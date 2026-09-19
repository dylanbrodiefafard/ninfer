#include "targets/qwen4/native_runtime.h"

#include "targets/qwen4/native_runtime_layout.h"
#include "targets/qwen4/native_decoder.h"
#include "targets/qwen4/native_draft_runtime.h"
#include "core/decode_graph.h"
#include "ninfer/ops/cast.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/gated_residual.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace ninfer::targets::qwen4 {
namespace {
constexpr int D=2560,F=10240,V=248320;
NativeState::Config state_config(const NativeRuntimeConfig& c) {
    return {c.requests,c.context_tokens,c.kv_tokens,c.verify_width,
        c.kv_dtype==KvCacheStorage::Nvfp4?ops::QsaKvFormat::NVFP4G16:ops::QsaKvFormat::BF16};
}
std::size_t staging_bytes(const NativeRuntimeConfig& c) {
    return std::size_t(native_layout::columns(c))*(2560+4*4)+std::size_t(c.requests)*5*4;
}
template<class Model> std::uint64_t allocation_bytes(const Model& model,const NativeRuntimeConfig& c) {
    native_layout::validate(c);
    const auto scratch=native_layout::scratch(native_layout::precision(model),c);
    return NativeState::device_bytes(state_config(c))+native_layout::shared_bytes(c)+
        scratch.workspace+scratch.qsa+native_layout::graph_allowance(c)+
        (c.vision?VisionProgram::device_bytes(c.vision_patches,c.vision_segments):0)+
        (c.mtp||c.dflash?NativeDraftRuntime::device_bytes(model,c)+NativeDraftRuntime::graph_allowance(c):0);
}
Tensor view(const Tensor& storage,std::initializer_list<int> shape) {
    Tensor result(storage.data,storage.dtype,shape);
    if(result.bytes()>storage.bytes()) throw std::logic_error("Qwen4 runtime workspace extent");
    return result;
}
void d2d(void* to,const void* from,std::size_t bytes,cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(to,from,bytes,cudaMemcpyDeviceToDevice,stream));
}
int checkpoint(NativeCheckpoint kind) {
    const int k=static_cast<int>(kind);
    if(k<0 || k>1) throw std::invalid_argument("Qwen4 checkpoint kind");
    return k;
}
}

struct NativeRuntime::Impl {
    const NativeModelView& model;
    NativeRuntimeConfig config;
    DeviceContext& device;
    NativeState state;
    WorkspaceArena shared,workspace;
    DeviceBuffer qsa;
    native_layout::Shared buffers;
    PinnedHostBuffer staging;
    std::unique_ptr<VisionProgram> vision;
    std::unique_ptr<NativeDraftRuntime> draft;
    std::array<std::array<DecodeGraphExecutable,4>,2> graphs;
    ops::PreparedNgramRowConfig ngram;
    std::array<std::array<int,2>,4> history;
    std::array<std::array<std::array<int,2>,4>,2> saved_history;
    // Fixed capacities: assignment during retain/restore never grows these vectors.
    std::array<std::array<std::vector<int>,4>,3> visual_columns;
    std::vector<int> row_ids;
    int* ids=nullptr; int* positions=nullptr; int* slots=nullptr; int* lengths=nullptr;
    int* frontiers=nullptr; int* counts=nullptr; int* compact=nullptr;
    void* packed=nullptr;
    int width=0,batch=0;
    bool prepared=false,enqueued=false,recorded=false,poisoned=false;
    std::size_t graph_observed=0;

    Impl(const NativeModelView& m,const NativeRuntimeConfig& c,DeviceContext& d):
        model(m),config(c),device(d),state(state_config(c),d.stream),
        shared(native_layout::shared_bytes(c)),
        workspace(native_layout::scratch(native_layout::precision(m),c).workspace),
        qsa(native_layout::scratch(native_layout::precision(m),c).qsa),
        buffers(native_layout::allocate(shared,c)),staging(staging_bytes(c)),
        ngram(ops::prepare_ngram_row_config(m.ngram)),row_ids(16*native_layout::columns(c)) {
        const int n=native_layout::columns(c);
        ids=static_cast<int*>(staging.data());positions=ids+n;slots=positions+3*n;
        lengths=slots+c.requests;frontiers=lengths+c.requests;counts=frontiers+c.requests;
        compact=counts+c.requests;packed=compact+c.requests;
        for(auto& h:history) h={m.ngram.eos_token_id,m.ngram.eos_token_id};
        for(auto& image:visual_columns) for(auto& cols:image) cols.reserve(native_layout::visual_capacity(c));
        if(c.vision) vision=std::make_unique<VisionProgram>(m.vision,c.vision_patches,c.vision_segments);
        if(c.mtp||c.dflash) draft=std::make_unique<NativeDraftRuntime>(m,c,d);
        if(c.use_cuda_graph) {
            device.synchronize();std::size_t before=0,total=0;
            CUDA_CHECK(cudaMemGetInfo(&before,&total));
            for(int b=1;b<=c.requests;++b) {
                capture(b,1,false);
                if(c.mtp||c.dflash) capture(b,c.verify_width,true);
            }
            device.synchronize();std::size_t after=0;CUDA_CHECK(cudaMemGetInfo(&after,&total));
            graph_observed=before>after?before-after:0;
            if(graph_observed>native_layout::graph_allowance(c))
                throw std::runtime_error("Qwen4 CUDA graphs exceed admitted startup allowance");
        }
    }
    ~Impl() { (void)cudaStreamSynchronize(device.stream); }
    void check() const {
        if(poisoned) throw std::runtime_error("Qwen4 runtime requires teardown after failed GPU transaction");
    }
    void idle() const {check();if(prepared) throw std::logic_error("Qwen4 pending batch must be resolved");}
    void slot(int s) const {if(s<0 || s>=config.requests) throw std::invalid_argument("Qwen4 runtime slot");}
    Tensor visual(int image,int s) const {
        return buffers.visual.slice(3,image,1).slice(2,s,1).reshape({D,native_layout::visual_capacity(config)});
    }
    void upload(Tensor target,const void* source) {
        CUDA_CHECK(cudaMemcpyAsync(target.data,source,target.bytes(),cudaMemcpyHostToDevice,device.stream));
    }
    ops::QsaBatchControls controls(int w,int b) const {
        return {view(buffers.slots,{b}),view(buffers.valid,{b}),view(buffers.frontiers,{b}),
                view(buffers.positions,{3,w,b})};
    }
    NativeDecoderViews decoder_views(int w,int b) const {
        return {view(buffers.residual,{D,4,w,b}),view(buffers.ple_embedding,{D,w,b}),
            view(buffers.mixed,{D,w,b}),view(buffers.block,{D,w,b}),view(buffers.scale,{4,w,b}),
            view(buffers.routes,{10,w,b}),view(buffers.probabilities,{10,w,b}),
            view(buffers.selected,{2051,w,b}),view(buffers.selected_count,{w,b}),
            config.dflash?view(buffers.features,{12800,w,b}):Tensor{},view(buffers.compact_rows,{b})};
    }
    void body(int w,int b,bool record,int visible) {
        workspace.reset();const int n=w*b;
        auto ple=view(buffers.ple_embedding,{160,16,n});
        if(model.ple_table.format==NativePleFormat::Nvfp4) {
            auto rows=view(buffers.packed_rows,{94,16,n});
            ops::ple_nvfp4_decode_rows(rows,ple,device.stream);
        } else {
            auto rows=view(buffers.packed_rows,{160,16,n});
            ops::ple_fp8_decode_rows(rows,model.ple_table.fp8_scale_bits,ple,device.stream);
        }
        auto embedding=view(buffers.embedding,{D,n});
        auto residual=view(buffers.residual,{D,4,n});
        ops::gated_residual_broadcast(embedding,residual,device.stream);
        auto scratch=Tensor(qsa.p,DType::U8,{static_cast<int>(qsa.bytes)});
        enqueue_native_decoder(model.layers,model.ple,state,controls(w,b),visible,
            decoder_views(w,b),record,workspace,scratch,device.stream,
            b==1 && w>16 && lengths[0]==w && !record?slots[0]:-1);
        auto output=view(buffers.mixed,{D,n});const auto& g=model.final_gr;
        ops::gated_residual_read(residual,g.norm,g.down,g.up,output,workspace,device.stream);
        auto logits=view(buffers.logits,{V,n});
        ops::linear(output,model.output_head,logits,ops::LinearPolicy::A16Only,workspace,device.stream);
    }
    void capture(int b,int w,bool record) {
        DecodeGraphDefinition definition;
        definition.capture(device.stream,[&]{body(w,b,record,config.context_tokens);});
        auto& graph=graphs[record?1:0][b-1];graph.instantiate(definition);graph.upload(device.stream);
    }
    void visual_image(int from,int to,int s) {
        const auto& source=visual_columns[from][s];
        if(config.vision && !source.empty())
            d2d(visual(to,s).data,visual(from,s).data,std::size_t(D)*source.size()*2,device.stream);
        visual_columns[to][s]=source;
    }
};

NativeRuntime::NativeRuntime(const NativeModelView& m,const NativeRuntimeConfig& c,DeviceContext& d) {
    native_layout::validate(c);
    if(c.dflash&&!m.dflash) throw std::invalid_argument("Qwen4 artifact has no DFlash component");
    impl_=std::make_unique<Impl>(m,c,d);
}
NativeRuntime::~NativeRuntime()=default;
std::uint64_t NativeRuntime::device_bytes(const NativeModelView& m,const NativeRuntimeConfig& c) {return allocation_bytes(m,c);}
std::uint64_t NativeRuntime::device_bytes(const NativeArtifactPlan& m,const NativeRuntimeConfig& c) {return allocation_bytes(m,c);}
std::uint64_t NativeRuntime::pinned_bytes(const NativeRuntimeConfig& c) {
    native_layout::validate(c);
    return staging_bytes(c)+16+(c.vision?VisionProgram::pinned_bytes(c.vision_patches,c.vision_segments):0)+
        (c.mtp||c.dflash?NativeDraftRuntime::pinned_bytes(c):0);
}
MemorySummary NativeRuntime::memory_summary() const {
    const auto& p=*impl_;MemorySummary m;m.device=p.device.device;
    m.max_context=p.config.context_tokens;m.kv_capacity=pages_for_tokens(p.config.kv_tokens)*64;
    m.kv_capacity_page_groups=pages_for_tokens(p.config.kv_tokens);
    m.kv_capacity_max_page_groups=p.config.requests*pages_for_tokens(p.config.context_tokens);
    m.kv_cache=p.config.kv_dtype;
    const auto state_bytes=NativeState::device_bytes(state_config(p.config));
    m.sequence={state_bytes,state_bytes,state_bytes};
    m.workspace={p.workspace.capacity()+p.qsa.bytes,p.workspace.used()+p.qsa.bytes,p.workspace.peak_used()+p.qsa.bytes};
    m.request_transient={p.shared.capacity(),p.shared.used(),p.shared.peak_used()};
    m.runtime_reservation_bytes=device_bytes(p.model,p.config);
    m.minimum_runtime_reservation_bytes=m.runtime_reservation_bytes;
    m.cuda_graph_allowance_bytes=native_layout::graph_allowance(p.config)+NativeDraftRuntime::graph_allowance(p.config);
    m.cuda_graph_observed_bytes=p.graph_observed+(p.draft?p.draft->observed_graph_bytes():0);
    m.workspace_logical_peak_bytes=p.workspace.peak_used();return m;
}
void NativeRuntime::reset_memory_peaks() {impl_->workspace.reset_peak();impl_->shared.reset_peak();}
void NativeRuntime::synchronize() {impl_->device.synchronize();}
void NativeRuntime::reset(int s) {
    auto& p=*impl_;p.idle();p.slot(s);p.device.synchronize();p.state.reset(s);
    if(p.draft) p.draft->reset(s);
    p.history[s]={p.model.ngram.eos_token_id,p.model.ngram.eos_token_id};
    for(auto& image:p.visual_columns) image[s].clear();
}
int NativeRuntime::frontier(int s) const {impl_->check();return impl_->state.frontier(s);}
bool NativeRuntime::can_reserve(int s,int maximum,std::span<const int> evictions) const {
    const auto& p=*impl_;p.check();return p.state.can_reserve(s,maximum,evictions)&&
        (!p.draft||p.draft->can_reserve(s,maximum,evictions));
}
bool NativeRuntime::reserve(int s,int maximum) {
    auto& p=*impl_;p.idle();if(!can_reserve(s,maximum)) return false;
    if(maximum<p.state.frontier(s) || (p.draft&&maximum<p.draft->frontier(s)))
        throw std::invalid_argument("Qwen4 reservation precedes live state");
    try {p.state.reserve(s,maximum);if(p.draft)p.draft->reserve(s,maximum);}
    catch(...) {p.poisoned=true;throw;}
    return true;
}
void NativeRuntime::release_reservation(int s) {
    auto& p=*impl_;p.idle();p.device.synchronize();p.state.release_reservation(s);
    if(p.draft)p.draft->release_reservation(s);
}

void NativeRuntime::prepare(std::span<const NativeInputRow> rows,int envelope) {
    auto& p=*impl_;p.idle();
    const int b=rows.size(),n=b*envelope;
    if(b<1||b>p.config.requests||envelope<1||n>native_layout::columns(p.config)||
       (b==1&&envelope>std::max(p.config.prefill_width,p.config.verify_width))||
       (b>1&&envelope>p.config.verify_width)) throw std::invalid_argument("Qwen4 input envelope");
    std::array<bool,4> seen{};
    for(const auto& row:rows) {
        p.slot(row.slot);
        if(seen[row.slot]||row.token_ids.empty()||row.token_ids.size()>std::size_t(envelope)||
           row.positions.size()!=row.token_ids.size()||
           row.token_ids.size()>std::size_t(p.config.context_tokens-p.state.frontier(row.slot)))
            throw std::invalid_argument("Qwen4 input row domain");
        seen[row.slot]=true;
        for(auto id:row.token_ids) if(id<0||id>=kNativeTokenDomain) throw std::invalid_argument("Qwen4 input token domain");
        for(auto position:row.positions) for(int axis:position)
            if(axis<0||axis>=262144) throw std::invalid_argument("Qwen4 input source position");
    }
    // Reuse of pinned input/staging is legal only after every prior consumer drains.
    p.device.synchronize();
    std::fill_n(p.ids,n,p.model.ngram.eos_token_id);std::fill_n(p.positions,3*n,0);
    for(int i=0;i<b;++i) {
        const auto& row=rows[i];p.slots[i]=row.slot;p.lengths[i]=row.token_ids.size();
        p.frontiers[i]=p.state.frontier(row.slot);p.compact[i]=i;
        auto h=p.history[row.slot];
        for(int t=0;t<envelope;++t) {
            const int col=i*envelope+t;
            if(t<p.lengths[i]) {
                p.ids[col]=row.token_ids[t];std::copy_n(row.positions[t].data(),3,p.positions+3*col);
            }
            const auto step=ops::ngram_row_ids_host_step(p.ids[col],h,p.ngram);
            std::copy(step.row_ids.begin(),step.row_ids.end(),p.row_ids.begin()+16*col);
            if(t<p.lengths[i])h=step.new_history;
        }
    }
    try {
        for(int i=0;i<b;++i)p.state.materialize(p.slots[i],p.frontiers[i]+p.lengths[i]);
        p.upload(view(p.buffers.ids,{envelope,b}),p.ids);
        p.upload(view(p.buffers.positions,{3,envelope,b}),p.positions);
        p.upload(view(p.buffers.slots,{b}),p.slots);p.upload(view(p.buffers.valid,{b}),p.lengths);
        p.upload(view(p.buffers.frontiers,{b}),p.frontiers);p.upload(view(p.buffers.compact_rows,{b}),p.compact);
        const std::span<const int> rowids(p.row_ids.data(),16*n);
        if(p.model.ple_table.format==NativePleFormat::Nvfp4) {
            auto device_rows=view(p.buffers.packed_rows,{94,16,n});
            ops::ple_nvfp4_stage_rows_batch(p.model.ple_table.nvfp4,rowids,n,p.packed,std::size_t(2560)*n,
                device_rows,p.device.stream);
        } else {
            auto device_rows=view(p.buffers.packed_rows,{160,16,n});
            ops::ple_fp8_stage_rows_batch(p.model.ple_table.fp8,rowids,n,p.packed,std::size_t(2560)*n,
                device_rows,p.device.stream);
        }
        auto token_ids=view(p.buffers.ids,{n});auto embedding=view(p.buffers.embedding,{D,n});
        ops::embedding(token_ids,p.model.token_embedding,embedding,p.device.stream);
        // Byte-only copies replace the represented embedding at media columns. This staging
        // occurs outside the compute graph; subsequent broadcast/decoder math remains compact.
        for(int i=0;i<b;++i) {
            const auto& columns=p.visual_columns[0][p.slots[i]];
            auto it=std::lower_bound(columns.begin(),columns.end(),p.frontiers[i]);
            for(;it!=columns.end()&&*it<p.frontiers[i]+p.lengths[i];++it) {
                const auto source=it-columns.begin();const int col=i*envelope+*it-p.frontiers[i];
                d2d(static_cast<std::byte*>(embedding.data)+std::size_t(col)*D*2,
                    static_cast<const std::byte*>(p.visual(0,p.slots[i]).data)+source*D*2,D*2,p.device.stream);
            }
        }
        p.width=envelope;p.batch=b;p.prepared=true;p.enqueued=false;
    } catch(...) {p.poisoned=true;throw;}
}
NativeBatchOutput NativeRuntime::enqueue(bool record) {
    auto& p=*impl_;p.check();
    if(!p.prepared||p.enqueued||(record&&(p.width<2||p.width>p.config.verify_width)))
        throw std::logic_error("Qwen4 enqueue boundary");
    try {
        const bool graph=p.config.use_cuda_graph&&
            ((!record&&p.width==1)||(record&&p.width==p.config.verify_width&&(p.config.mtp||p.config.dflash)));
        if(graph)p.graphs[record?1:0][p.batch-1].launch(p.device.stream);
        else {
            int visible=1;for(int i=0;i<p.batch;++i)visible=std::max(visible,p.frontiers[i]+p.lengths[i]);
            p.body(p.width,p.batch,record,visible);
        }
        p.enqueued=true;p.recorded=record;
    } catch(...) {p.poisoned=true;throw;}
    return {view(p.buffers.logits,{V,p.width,p.batch}),view(p.buffers.residual,{F,p.width,p.batch}),
        p.config.dflash?view(p.buffers.features,{12800,p.width,p.batch}):Tensor{},view(p.buffers.valid,{p.batch})};
}
void NativeRuntime::commit(std::span<const NativeCommitRow> rows) {
    auto& p=*impl_;p.check();
    if(!p.prepared||!p.enqueued||rows.size()!=std::size_t(p.batch))throw std::logic_error("Qwen4 commit boundary");
    std::array<bool,4> seen{};
    for(const auto& row:rows) {
        int i=0;for(;i<p.batch&&p.slots[i]!=row.slot;++i){}
        if(i==p.batch||seen[i]||row.input_count<0||row.input_count>p.lengths[i]||
           (!p.recorded&&row.input_count!=0&&row.input_count!=p.lengths[i]))
            throw std::invalid_argument("Qwen4 commit prefix");
        seen[i]=true;p.counts[i]=row.input_count;
    }
    try {
        auto device_counts=view(p.buffers.counts,{p.batch});
        p.state.commit({p.slots,std::size_t(p.batch)},{p.lengths,std::size_t(p.batch)},
            {p.counts,std::size_t(p.batch)},p.width,p.recorded,view(p.buffers.ids,{p.width,p.batch}),
            view(p.buffers.residual,{F,p.width,p.batch}),view(p.buffers.slots,{p.batch}),device_counts);
        for(int i=0;i<p.batch;++i)for(int t=0;t<p.counts[i];++t) {
            auto& h=p.history[p.slots[i]];h={h[1],p.ids[i*p.width+t]};
        }
        p.prepared=p.enqueued=false;
    } catch(...) {p.poisoned=true;throw;}
}
void NativeRuntime::discard() {
    auto& p=*impl_;p.check();if(!p.prepared)return;
    if(!p.enqueued) {
        try {p.state.discard_materialization({p.slots,std::size_t(p.batch)});p.prepared=false;}
        catch(...) {p.poisoned=true;throw;}
        return;
    }
    std::array<NativeCommitRow,4> rows{};for(int i=0;i<p.batch;++i)rows[i]={p.slots[i],0};
    commit({rows.data(),std::size_t(p.batch)});
}

void NativeRuntime::retain(int s,NativeCheckpoint kind) {
    auto& p=*impl_;p.idle();p.slot(s);const int k=checkpoint(kind);
    try {p.state.retain(s,kind);if(p.draft)p.draft->retain(s,kind);
        p.visual_image(0,k+1,s);p.saved_history[k][s]=p.history[s];p.device.synchronize();}
    catch(...) {p.poisoned=true;throw;}
}
void NativeRuntime::restore(int s,NativeCheckpoint kind) {
    auto& p=*impl_;p.idle();p.slot(s);const int k=checkpoint(kind);
    if(!has_retained(s,kind))throw std::logic_error("Qwen4 complete checkpoint absent");
    try {p.state.restore(s,kind);if(p.draft)p.draft->restore(s,kind);
        p.visual_image(k+1,0,s);p.history[s]=p.saved_history[k][s];p.device.synchronize();}
    catch(...) {p.poisoned=true;throw;}
}
void NativeRuntime::evict_retained(int s,NativeCheckpoint kind) {
    auto& p=*impl_;p.idle();p.slot(s);const int k=checkpoint(kind);
    p.device.synchronize();p.state.evict_retained(s,kind);if(p.draft)p.draft->evict_retained(s,kind);
    p.visual_columns[k+1][s].clear();
}
bool NativeRuntime::has_retained(int s,NativeCheckpoint kind) const {
    const auto& p=*impl_;p.check();return p.state.has_retained(s,kind)&&(!p.draft||p.draft->has_retained(s,kind));
}
void NativeRuntime::prepare_vision(int s,std::span<const float> patches,std::span<const VisionGrid> grids,
                                  std::span<const std::int32_t> columns) {
    auto& p=*impl_;p.idle();p.slot(s);
    if(!p.vision)throw std::invalid_argument("Qwen4 Vision not enabled at startup");
    const auto control=prepare_vision_control(grids);
    if(control.patches>p.config.vision_patches||control.segments.size()>std::size_t(p.config.vision_segments+1)||
       patches.size()!=std::size_t(control.patches)*1536||columns.size()!=std::size_t(control.patches/4)||
       columns.size()>std::size_t(native_layout::visual_capacity(p.config)))
        throw std::invalid_argument("Qwen4 Vision request capacity");
    int previous=-1;for(int col:columns) {
        if(col<=previous||col>=p.config.context_tokens)throw std::invalid_argument("Qwen4 Vision placement columns");
        previous=col;
    }
    try {
        p.device.synchronize();
        auto source=view(p.buffers.patch_fp32,{1536,control.patches});
        auto represented=view(p.buffers.patch_bf16,{1536,control.patches});
        // The owning prepared prompt remains alive until the explicit drain below.
        p.upload(source,patches.data());ops::cast_fp32_to_bf16(source,represented,p.device.stream);
        p.vision->configure(control,p.device.stream);auto encoded=p.vision->execute(represented,p.device.stream);
        d2d(p.visual(0,s).data,encoded.data,encoded.bytes(),p.device.stream);p.device.synchronize();
        p.visual_columns[0][s].assign(columns.begin(),columns.end());
    } catch(...) {p.poisoned=true;throw;}
}
void NativeRuntime::clear_vision(int s) {auto& p=*impl_;p.idle();p.slot(s);p.visual_columns[0][s].clear();}

Tensor NativeRuntime::extend_mtp(std::span<const NativeMtpInputRow> rows,int envelope) {
    auto& p=*impl_;p.idle();if(!p.config.mtp)throw std::logic_error("Qwen4 MTP disabled");
    try {return p.draft->extend_mtp(rows,envelope);}catch(...) {p.poisoned=true;throw;}
}
Tensor NativeRuntime::draft_mtp(std::span<const int> slots,std::span<const TokenId> tokens,
                              std::span<const std::array<int,3>> positions) {
    auto& p=*impl_;p.idle();if(!p.config.mtp)throw std::logic_error("Qwen4 MTP disabled");
    try {return p.draft->draft_mtp(slots,tokens,positions);}catch(...) {p.poisoned=true;throw;}
}
void NativeRuntime::discard_mtp(int s) {
    auto& p=*impl_;p.check();if(!p.config.mtp)throw std::logic_error("Qwen4 MTP disabled");
    try {p.draft->discard_mtp(s);}catch(...) {p.poisoned=true;throw;}
}
Tensor NativeRuntime::reseed_mtp(std::span<const int> slots,std::span<const TokenId> tokens,
                               std::span<const std::array<int,3>> positions) {
    auto& p=*impl_;p.idle();if(!p.config.mtp)throw std::logic_error("Qwen4 MTP disabled");
    try {return p.draft->reseed_mtp(slots,tokens,positions);}catch(...) {p.poisoned=true;throw;}
}
void NativeRuntime::append_dflash(int s,const Tensor& features,std::span<const std::int32_t> positions) {
    auto& p=*impl_;p.idle();if(!p.config.dflash)throw std::logic_error("Qwen4 DFlash disabled");
    try {p.draft->append_dflash(s,features,positions);}catch(...) {p.poisoned=true;throw;}
}
Tensor NativeRuntime::draft_dflash(std::span<const int> slots,std::span<const TokenId> anchors,
                                  std::span<const std::int32_t> positions,int width) {
    auto& p=*impl_;p.idle();if(!p.config.dflash)throw std::logic_error("Qwen4 DFlash disabled");
    try {return p.draft->draft_dflash(slots,anchors,positions,width);}catch(...) {p.poisoned=true;throw;}
}
} // namespace ninfer::targets::qwen4
