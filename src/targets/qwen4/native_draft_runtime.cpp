#include "targets/qwen4/native_draft_runtime.h"
#include "targets/qwen4/native_runtime.h"
#include "targets/qwen4/native_runtime_layout.h"
#include "core/decode_graph.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/linear.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace ninfer::targets::qwen4 {
namespace {
constexpr int D=2560,F=10240,V=248320,S=2051;
struct Layout {
    PagedKVPoolLayout pages;
    std::array<TensorRegion,4> carry,ids,count,target;
    std::array<TensorRegion,2> tail_k,tail_v,tail_raw,tail_pos;
    TensorRegion tokens,positions,valid,slots,frontiers,cache_positions,hidden,embedding,logits,frozen_ids,frozen_count,features;
    std::size_t bytes=0,pinned=0;
    int width=0,columns=0;
};
bool enabled(const NativeRuntimeConfig& c) {return c.mtp || c.dflash;}
Layout plan(const NativeRuntimeConfig& c) {
    Layout p;
    if(!enabled(c)) return p;
    if(c.mtp && c.dflash) throw std::invalid_argument("one native draft backend must be selected");
    if(c.requests<1 || c.requests>4 || c.context_tokens<1 || c.context_tokens>262144 ||
       c.kv_tokens<c.context_tokens || c.verify_width<2 || c.verify_width>16 ||
       c.prefill_width<1 || c.prefill_width>4096)
        throw std::invalid_argument("native draft startup geometry");
    p.width=std::max({c.prefill_width,c.verify_width,c.dflash?7:1});
    p.columns=std::max(p.width,std::min(p.width,16)*c.requests);
    LayoutBuilder b;
    const auto extra=c.mtp?c.requests*pages_for_tokens(c.verify_width-2):0;
    PagedKVPoolSpec spec{pages_for_tokens(c.kv_tokens)+extra,pages_for_tokens(c.context_tokens),c.requests,
        c.mtp?PagedKVPlaneOrder::PageMajor:PagedKVPlaneOrder::HeadMajor,{}};
    if(c.mtp) spec.planes={{DType::BF16,256,2},{DType::BF16,256,2},{DType::BF16,128,1},{DType::I32,3,1}};
    else for(int i=0;i<10;++i) spec.planes.push_back({DType::BF16,256,2});
    p.pages=plan_paged_kv_pool(b,spec);
    if(c.mtp) {
        for(int i=0;i<4;++i) {
            p.carry[i]=b.add_tensor(DType::BF16,{F,c.requests},256,"MTP carried state");
            p.target[i]=b.add_tensor(DType::BF16,{F,c.requests},256,"last actual target state");
            p.ids[i]=b.add_tensor(DType::I32,{S,c.requests},256,"MTP frozen domain");
            p.count[i]=b.add_tensor(DType::I32,{c.requests},256,"MTP frozen count");
        }
        for(int i=0;i<2;++i) {
            p.tail_k[i]=b.add_tensor(DType::BF16,{256,2,c.requests},256,"retained MTP last K");
            p.tail_v[i]=b.add_tensor(DType::BF16,{256,2,c.requests},256,"retained MTP last V");
            p.tail_raw[i]=b.add_tensor(DType::BF16,{128,c.requests},256,"retained MTP last index key");
            p.tail_pos[i]=b.add_tensor(DType::I32,{3,c.requests},256,"retained MTP last position");
        }
        p.hidden=b.add_tensor(DType::BF16,{F,p.columns},256,"compact MTP target/carry input");
        p.frozen_ids=b.add_tensor(DType::I32,{S,c.requests},256,"compact frozen IDs");
        p.frozen_count=b.add_tensor(DType::I32,{c.requests},256,"compact frozen count");
    } else p.features=b.add_tensor(DType::BF16,{12800,p.width},256,"accepted DFlash features");
    p.tokens=b.add_tensor(DType::I32,{p.columns},256,"draft token IDs");
    p.positions=b.add_tensor(DType::I32,{3,p.columns},256,"draft source positions");
    p.cache_positions=b.add_tensor(DType::I32,{p.columns},256,"draft cache positions");
    p.valid=b.add_tensor(DType::I32,{c.requests},256,"draft valid widths");
    p.slots=b.add_tensor(DType::I32,{c.requests},256,"draft table rows");
    p.frontiers=b.add_tensor(DType::I32,{c.requests},256,"draft frontiers");
    p.embedding=b.add_tensor(DType::BF16,{D,p.columns},256,"shared target embeddings");
    p.logits=b.add_tensor(DType::BF16,{V,p.columns},256,"shared target head logits");
    p.bytes=b.finish(256);p.pinned=std::size_t(5*p.columns+3*c.requests)*4;
    return p;
}
void copy(const Tensor& dst,const Tensor& src,cudaStream_t stream) {
    if(dst.bytes()!=src.bytes()) throw std::logic_error("native draft state copy shape");
    if(dst.data==src.data) return;
    CUDA_CHECK(cudaMemcpyAsync(dst.data,src.data,dst.bytes(),cudaMemcpyDeviceToDevice,stream));
}
struct Graph {int width,batch;bool frozen;DecodeGraphDefinition definition;DecodeGraphExecutable executable;};
}

struct NativeDraftRuntime::Impl {
    const NativeModelView& model;NativeRuntimeConfig config;cudaStream_t stream;
    Layout layout;DeviceBuffer backing;std::unique_ptr<PinnedHostBuffer> staging;std::unique_ptr<PagedKVPool> pool;
    std::array<PagedKVAllocation,4> allocation;
    std::array<int,4> frontier{},seed_frontier{};std::array<bool,4> active{};
    std::array<std::array<int,4>,2> retained_frontier{};
    std::array<std::array<bool,4>,2> retained{};
    std::array<Tensor,4> carry,ids,count,target;
    std::unique_ptr<MtpProgram> mtp;std::unique_ptr<DFlashProgram> dflash;
    std::unique_ptr<WorkspaceArena> head_workspace;std::vector<Graph> graphs;std::uint64_t graph_observed=0;

    Tensor view(const TensorRegion& r) const {return r.bind({backing.p,backing.bytes});}
    Tensor field(const TensorRegion& r,DType dtype,std::initializer_list<int> shape) const {
        return Tensor(view(r).data,dtype,shape);
    }
    ops::QsaPagedStateView qsa() const {
        return {ops::QsaKvFormat::BF16,pool->plane(0),pool->plane(1),{},{},pool->plane(2),pool->plane(3),pool->block_tables()};
    }
    std::array<PagedKVBatchLayerView,5> draft_caches() const {
        std::array<PagedKVBatchLayerView,5> result;
        for(int i=0;i<5;++i) result[i]={.k_pages=pool->plane(i*2),.v_pages=pool->plane(i*2+1),
            .block_tables=pool->block_tables(),.head_dim=256,.num_kv_heads=2,.dtype=DType::BF16};
        return result;
    }
    static std::size_t head_bytes(const NativeRuntimeConfig& c,QType type) {
        if(!enabled(c)) return 0;
        const auto p=plan(c);
        return std::max<std::size_t>(256,ops::linear_workspace_capacity_bytes(
            type,V,D,ops::LinearPolicy::A16Only,1,p.columns));
    }
    Impl(const NativeModelView& m,const NativeRuntimeConfig& c,DeviceContext& device)
        :model(m),config(c),stream(device.stream),layout(plan(c)),backing(layout.bytes),
         staging(enabled(c)?std::make_unique<PinnedHostBuffer>(layout.pinned):nullptr),
         head_workspace(enabled(c)?std::make_unique<WorkspaceArena>(head_bytes(c,m.output_head.qtype)):nullptr) {
        seed_frontier.fill(-1);
        if(!enabled(c)) return;
        CUDA_CHECK(cudaMemsetAsync(backing.p,0,backing.bytes,stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        pool=std::make_unique<PagedKVPool>(DeviceSpan{backing.p,backing.bytes},layout.pages);
        if(c.mtp) {
            for(int i=0;i<4;++i) {carry[i]=view(layout.carry[i]);ids[i]=view(layout.ids[i]);
                count[i]=view(layout.count[i]);target[i]=view(layout.target[i]);}
            mtp=std::make_unique<MtpProgram>(m.mtp,layout.width,c.requests);
        } else {
            if(!m.dflash) throw std::invalid_argument("native artifact has no DFlash companion");
            dflash=std::make_unique<DFlashProgram>(*m.dflash,c.context_tokens,layout.width,c.requests,draft_caches());
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if(c.use_cuda_graph) {
            std::size_t before=0,total=0;CUDA_CHECK(cudaMemGetInfo(&before,&total));
            for(int b=1;b<=c.requests;++b) {
                if(c.mtp) {
                    for(int w=1;w<=c.verify_width;++w) capture(w,b,false);
                    capture(1,b,true);
                } else for(int w=1;w<=7;++w) capture(w,b,false);
            }
            if(c.mtp && c.prefill_width>c.verify_width) capture(c.prefill_width,1,false);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            std::size_t after=0;CUDA_CHECK(cudaMemGetInfo(&after,&total));graph_observed=before>after?before-after:0;
            if(graph_observed>NativeDraftRuntime::graph_allowance(c))
                throw std::runtime_error("native draft CUDA graphs exceed admitted startup allowance");
        }
    }
    void check(int slot) const {
        if(slot<0 || slot>=config.requests) throw std::invalid_argument("native draft slot");
    }
    void require(bool want_mtp) const {
        if(want_mtp?!config.mtp:!config.dflash) throw std::logic_error("wrong native draft backend");
    }
    void image(int from,int to,int slot) {
        if(!mtp) return;
        copy(carry[to].slice(1,slot,1),carry[from].slice(1,slot,1),stream);
        copy(ids[to].slice(1,slot,1),ids[from].slice(1,slot,1),stream);
        copy(count[to].slice(0,slot,1),count[from].slice(0,slot,1),stream);
        copy(target[to].slice(1,slot,1),target[from].slice(1,slot,1),stream);
    }
    Tensor compute(int width,int batch,bool frozen) {
        auto token=field(layout.tokens,DType::I32,{width*batch});
        auto embedding=field(layout.embedding,DType::BF16,{D,width*batch});
        auto output=field(layout.logits,DType::BF16,{V,width*batch});
        if(mtp) {
            ops::embedding(token,model.token_embedding,embedding,stream);
            const ops::QsaBatchControls controls{
                field(layout.slots,DType::I32,{batch}),field(layout.valid,DType::I32,{batch}),
                field(layout.frontiers,DType::I32,{batch}),field(layout.positions,DType::I32,{3,width,batch})};
            auto frozen_ids=field(layout.frozen_ids,DType::I32,{S,1,batch});
            auto frozen_count=field(layout.frozen_count,DType::I32,{1,batch});
            auto result=mtp->execute(embedding.view({D,width,batch}),field(layout.hidden,DType::BF16,{F,width,batch}),
                controls,qsa(),config.context_tokens,frozen?&frozen_ids:nullptr,frozen?&frozen_count:nullptr,stream);
            auto input=result.logit_input.view({D,width*batch});
            ops::linear_packed_sequences(input,model.output_head,output,ops::LinearPolicy::A16Only,*head_workspace,stream,width);
        } else {
            auto anchors=field(layout.tokens,DType::I32,{batch});
            auto positions=field(layout.positions,DType::I32,{batch});
            auto lengths=field(layout.frontiers,DType::I32,{batch});
            auto valid=field(layout.valid,DType::I32,{batch});auto slots=field(layout.slots,DType::I32,{batch});
            auto logits=output.view({V,width,batch});
            dflash->draft(anchors,positions,lengths,valid,slots,model.token_embedding,model.output_head,logits,stream);
        }
        return output.view({V,width,batch});
    }
    void capture(int width,int batch,bool frozen) {
        Graph g{width,batch,frozen,{},{}};
        g.definition.capture(stream,[&]{(void)compute(width,batch,frozen);});
        g.executable.instantiate(g.definition);g.executable.upload(stream);graphs.push_back(std::move(g));
    }
    Tensor enqueue(int width,int batch,bool frozen) {
        for(auto& g:graphs) if(g.width==width && g.batch==batch && g.frozen==frozen) {
            g.executable.launch(stream);return field(layout.logits,DType::BF16,{V,width,batch});
        }
        return compute(width,batch,frozen);
    }
    void upload(const TensorRegion& region,std::span<const int> words,std::size_t& offset) {
        if(offset+words.size_bytes()>layout.pinned) throw std::logic_error("native draft pinned capacity");
        auto* p=static_cast<std::byte*>(staging->data())+offset;
        std::memcpy(p,words.data(),words.size_bytes());
        CUDA_CHECK(cudaMemcpyAsync(view(region).data,p,words.size_bytes(),cudaMemcpyHostToDevice,stream));
        offset+=words.size_bytes();
    }
    void controls(std::span<const int> slots,std::span<const int> lengths,int width,
                  std::span<const int> tokens,std::span<const int> positions) {
        std::vector<int> frontiers;for(int s:slots) frontiers.push_back(frontier[s]);
        std::size_t offset=0;upload(layout.slots,slots,offset);upload(layout.valid,lengths,offset);
        upload(layout.frontiers,frontiers,offset);upload(layout.tokens,tokens,offset);upload(layout.positions,positions,offset);
    }
    void validate_rows(std::span<const int> slots) const {
        if(slots.empty() || slots.size()>std::size_t(config.requests)) throw std::invalid_argument("native draft compact batch");
        std::array<bool,4> seen{};
        for(int slot:slots) {check(slot);if(seen[slot] || !active[slot] || !allocation[slot].valid())
            throw std::invalid_argument("native draft inactive/duplicate row");seen[slot]=true;}
    }
    void tail(int kind,int slot,bool restore,int logical_frontier=-1) {
        const int end=logical_frontier<0?frontier[slot]:logical_frontier;
        if(!mtp || end==0) return;
        const int logical=end-1,page=allocation[slot].page_ids()[logical/64],within=logical%64;
        for(int plane=0;plane<4;++plane) {
            const auto& src=pool->plane(plane);
            Tensor saved=plane==0?view(layout.tail_k[kind]).slice(2,slot,1):
                plane==1?view(layout.tail_v[kind]).slice(2,slot,1):
                plane==2?view(layout.tail_raw[kind]).slice(1,slot,1):view(layout.tail_pos[kind]).slice(1,slot,1);
            const int heads=plane<2?2:1,leading=plane<2?256:plane==2?128:3;
            const std::size_t word=plane==3?4:2;
            for(int h=0;h<heads;++h) {
                auto* physical=static_cast<std::byte*>(src.data)+std::size_t(leading)*(within+64*(h+heads*page))*word;
                auto* image=static_cast<std::byte*>(saved.data)+std::size_t(h)*leading*word;
                CUDA_CHECK(cudaMemcpyAsync(restore?physical:image,restore?image:physical,leading*word,cudaMemcpyDeviceToDevice,stream));
            }
        }
    }
};

NativeDraftRuntime::NativeDraftRuntime(const NativeModelView& m,const NativeRuntimeConfig& c,DeviceContext& d)
    :impl_(std::make_unique<Impl>(m,c,d)) {}
NativeDraftRuntime::~NativeDraftRuntime() {if(impl_) (void)cudaStreamSynchronize(impl_->stream);}
std::uint64_t NativeDraftRuntime::device_bytes(const NativeModelView& m,const NativeRuntimeConfig& c) {
    if(!enabled(c)) return 0;
    const auto p=plan(c);
    return p.bytes+Impl::head_bytes(c,m.output_head.qtype)+(c.mtp?MtpProgram::device_bytes(p.width,c.requests):DFlashProgram::device_bytes(c.context_tokens,p.width,c.requests));
}
std::uint64_t NativeDraftRuntime::device_bytes(const NativeArtifactPlan& a,const NativeRuntimeConfig& c) {
    if(c.dflash && !a.dflash) throw std::invalid_argument("native artifact has no DFlash companion");
    if(!enabled(c)) return 0;
    const auto p=plan(c);
    const auto type=native_layout::qtype(a.tensors.at("lm_head.weight").format);
    return p.bytes+Impl::head_bytes(c,type)+(c.mtp?MtpProgram::device_bytes(p.width,c.requests):DFlashProgram::device_bytes(c.context_tokens,p.width,c.requests));
}
std::uint64_t NativeDraftRuntime::pinned_bytes(const NativeRuntimeConfig& c) {return plan(c).pinned;}
std::uint64_t NativeDraftRuntime::graph_allowance(const NativeRuntimeConfig& c) {
    if(!enabled(c) || !c.use_cuda_graph) return 0;
    (void)plan(c);
    const int count=c.mtp?c.requests*(c.verify_width+1)+(c.prefill_width>c.verify_width?1:0):c.requests*7;
    return std::uint64_t(count)*12*1024*1024;
}
std::uint64_t NativeDraftRuntime::observed_graph_bytes() const noexcept {return impl_->graph_observed;}
ops::QsaPagedStateView NativeDraftRuntime::mtp_state() const {impl_->require(true);return impl_->qsa();}
NativeMtpSeedView NativeDraftRuntime::mtp_seed() const {
    auto& s=*impl_;s.require(true);return {s.carry[0],s.target[0],s.ids[0],s.count[0]};
}
PagedKVBatchLayerView NativeDraftRuntime::dflash_cache(int layer) const {
    auto& s=*impl_;s.require(false);
    if(layer<0 || layer>=5) throw std::invalid_argument("native DFlash diagnostic layer");
    return s.draft_caches()[layer];
}
int NativeDraftRuntime::frontier(int slot) const {impl_->check(slot);return impl_->frontier[slot];}
bool NativeDraftRuntime::can_reserve(int slot,int maximum,std::span<const int> evictions) const {
    auto& s=*impl_;s.check(slot);if(!enabled(s.config)) return true;
    if(maximum<1 || maximum>s.config.context_tokens) return false;
    const auto needed=pages_for_tokens(std::min(s.config.context_tokens,maximum+(s.mtp?s.config.verify_width-2:0)));
    std::uint64_t used=s.pool->entitled_pages();
    if(s.allocation[slot].valid()) used-=s.allocation[slot].page_entitlement();
    std::array<bool,4> seen{};
    for(int e:evictions) {s.check(e);if(e==slot || seen[e] || s.active[e] || !s.retained[0][e])
        throw std::invalid_argument("native draft eviction set");seen[e]=true;
        if(s.allocation[e].valid()) used-=s.allocation[e].page_entitlement();}
    return used+needed<=s.pool->page_group_count();
}
void NativeDraftRuntime::reserve(int slot,int maximum) {
    auto& s=*impl_;s.check(slot);if(!enabled(s.config)) return;
    if(maximum<s.frontier[slot]) throw std::invalid_argument("native draft reservation is shorter than live frontier");
    if(!can_reserve(slot,maximum)) throw std::runtime_error("native draft shared KV reservation exhausted");
    const auto needed=pages_for_tokens(std::min(s.config.context_tokens,maximum+(s.mtp?s.config.verify_width-2:0)));
    if(s.allocation[slot].valid()) s.allocation[slot].set_page_entitlement(needed);
    else s.allocation[slot]=s.pool->reserve(needed);
    if(s.allocation[slot].bound_row()!=slot) s.allocation[slot].bind_row(slot,s.stream);
    s.active[slot]=true;
}
void NativeDraftRuntime::release_reservation(int slot) {
    auto& s=*impl_;s.check(slot);if(!enabled(s.config)) return;
    discard_mtp(slot);
    if(s.allocation[slot].valid()) {s.allocation[slot].trim_tokens(s.frontier[slot]);
        s.allocation[slot].cancel_unmapped_entitlement();s.allocation[slot].unbind_row();}
    s.active[slot]=false;
}
void NativeDraftRuntime::reset(int slot) {
    auto& s=*impl_;s.check(slot);CUDA_CHECK(cudaStreamSynchronize(s.stream));
    s.allocation[slot].release();s.frontier[slot]=0;s.seed_frontier[slot]=-1;s.active[slot]=false;
    for(int k=0;k<2;++k) {s.retained[k][slot]=false;s.retained_frontier[k][slot]=0;}
}
void NativeDraftRuntime::retain(int slot,NativeCheckpoint kind) {
    auto& s=*impl_;s.check(slot);const int k=int(kind);
    if(k<0 || k>1) throw std::invalid_argument("native draft checkpoint kind");
    if(!enabled(s.config)) return;
    discard_mtp(slot);s.image(0,k+2,slot);s.tail(k,slot,false);CUDA_CHECK(cudaStreamSynchronize(s.stream));
    s.retained[k][slot]=true;s.retained_frontier[k][slot]=s.frontier[slot];
}
void NativeDraftRuntime::restore(int slot,NativeCheckpoint kind) {
    auto& s=*impl_;s.check(slot);const int k=int(kind);
    if(k<0 || k>1) throw std::invalid_argument("native draft checkpoint kind");
    if(!enabled(s.config)) return;
    if(!s.retained[k][slot]) throw std::logic_error("missing native draft checkpoint");
    discard_mtp(slot);const int restored=s.retained_frontier[k][slot];s.image(k+2,0,slot);s.tail(k,slot,true,restored);
    CUDA_CHECK(cudaStreamSynchronize(s.stream));
    s.frontier[slot]=restored;
    if(s.allocation[slot].valid()) s.allocation[slot].trim_tokens(s.frontier[slot]);
    for(int other=0;other<2;++other) if(s.retained_frontier[other][slot]>s.frontier[slot]) s.retained[other][slot]=false;
}
void NativeDraftRuntime::evict_retained(int slot,NativeCheckpoint kind) {
    auto& s=*impl_;s.check(slot);const int k=int(kind);
    if(k<0 || k>1) throw std::invalid_argument("native draft checkpoint kind");
    if(!enabled(s.config)) return;
    s.retained[k][slot]=false;
    if(!s.active[slot] && k==0) {
        CUDA_CHECK(cudaStreamSynchronize(s.stream));s.allocation[slot].release();s.frontier[slot]=0;s.seed_frontier[slot]=-1;
        for(int image=0;image<2;++image) {s.retained[image][slot]=false;s.retained_frontier[image][slot]=0;}
    }
}
bool NativeDraftRuntime::has_retained(int slot,NativeCheckpoint kind) const {
    auto& s=*impl_;s.check(slot);const int k=int(kind);
    if(k<0 || k>1) throw std::invalid_argument("native draft checkpoint kind");
    return !enabled(s.config) || s.retained[k][slot];
}

Tensor NativeDraftRuntime::extend_mtp(std::span<const NativeMtpInputRow> rows,int width) {
    auto& s=*impl_;s.require(true);const int batch=int(rows.size());
    if(width<1 || width>s.layout.width || batch<1 || batch>s.config.requests || (batch>1 && width>16))
        throw std::invalid_argument("native MTP extend envelope");
    std::vector<int> slots,lengths,tokens(std::size_t(width)*batch),positions(std::size_t(width)*batch*3);
    for(int b=0;b<batch;++b) {
        const auto& r=rows[b];s.check(r.slot);const int n=int(r.next_tokens.size());
        if(n<1 || n>width || r.source_positions.size()!=std::size_t(n) ||
           r.target_hidden.dtype!=DType::BF16 || !r.target_hidden.data || !r.target_hidden.is_contiguous() ||
           r.target_hidden.ne[0]!=F || r.target_hidden.ne[1]!=n || r.target_hidden.ne[2]!=1 || r.target_hidden.ne[3]!=1 ||
           s.seed_frontier[r.slot]>=0 || s.frontier[r.slot]+n>s.config.context_tokens ||
           !s.allocation[r.slot].valid() || pages_for_tokens(s.frontier[r.slot]+n)>s.allocation[r.slot].page_entitlement())
            throw std::invalid_argument("native MTP requires actual aligned target rows after draft discard");
        slots.push_back(r.slot);lengths.push_back(n);
        for(int t=0;t<n;++t) {
            if(r.next_tokens[t]<0 || r.next_tokens[t]>=kNativeTokenDomain)
                throw std::invalid_argument("native MTP next token outside tokenizer domain");
            tokens[b*width+t]=r.next_tokens[t];
            for(int a=0;a<3;++a) {if(r.source_positions[t][a]<0 || r.source_positions[t][a]>=262144) throw std::invalid_argument("native MTP source position");
                positions[(b*width+t)*3+a]=r.source_positions[t][a];}
        }
    }
    s.validate_rows(slots);CUDA_CHECK(cudaStreamSynchronize(s.stream));
    CUDA_CHECK(cudaMemsetAsync(s.view(s.layout.hidden).data,0,std::size_t(F)*width*batch*2,s.stream));
    for(int b=0;b<batch;++b) {
        const auto& r=rows[b];s.allocation[r.slot].materialize_tokens(s.frontier[r.slot]+lengths[b],s.stream);
        auto dst=s.field(s.layout.hidden,DType::BF16,{F,width,batch}).slice(2,b,1).slice(1,0,lengths[b]);
        copy(dst,r.target_hidden,s.stream);
    }
    s.controls(slots,lengths,width,tokens,positions);auto result=s.enqueue(width,batch,false);
    const auto output=s.mtp->output(width,batch);
    const auto selected=s.mtp->selected_ids(width,batch),counts=s.mtp->selected_count(width,batch);
    for(int b=0;b<batch;++b) {
        const int slot=slots[b],last=lengths[b]-1;
        copy(s.carry[0].slice(1,slot,1),output.carried_hidden.slice(2,b,1).slice(1,last,1),s.stream);
        copy(s.ids[0].slice(1,slot,1),selected.slice(2,b,1).slice(1,last,1),s.stream);
        copy(s.count[0].slice(0,slot,1),counts.slice(1,b,1).slice(0,last,1),s.stream);
        copy(s.target[0].slice(1,slot,1),rows[b].target_hidden.slice(1,last,1),s.stream);
    }
    CUDA_CHECK(cudaStreamSynchronize(s.stream));
    for(int b=0;b<batch;++b) s.frontier[slots[b]]+=lengths[b];
    return result;
}

Tensor NativeDraftRuntime::draft_mtp(std::span<const int> slots,std::span<const TokenId> proposals,
    std::span<const std::array<int,3>> source_positions) {
    auto& s=*impl_;s.require(true);s.validate_rows(slots);const int batch=int(slots.size());
    if(proposals.size()!=slots.size() || source_positions.size()!=slots.size())
        throw std::invalid_argument("native MTP draft rows");
    std::vector<int> positions(batch*3),lengths(batch,1);
    for(int b=0;b<batch;++b) {
        const int slot=slots[b];
        if(proposals[b]<0 || proposals[b]>=kNativeTokenDomain || s.frontier[slot]<1 ||
           s.frontier[slot]>=s.config.context_tokens ||
           pages_for_tokens(s.frontier[slot]+1)>s.allocation[slot].page_entitlement())
            throw std::invalid_argument("native MTP draft reservation/seed");
        for(int a=0;a<3;++a) {if(source_positions[b][a]<0 || source_positions[b][a]>=262144) throw std::invalid_argument("native MTP source position");
            positions[b*3+a]=source_positions[b][a];}
    }
    CUDA_CHECK(cudaStreamSynchronize(s.stream));
    for(int b=0;b<batch;++b) {
        const int slot=slots[b];
        if(s.seed_frontier[slot]<0) {s.image(0,1,slot);s.seed_frontier[slot]=s.frontier[slot];}
        s.allocation[slot].materialize_tokens(s.frontier[slot]+1,s.stream);
        copy(s.field(s.layout.hidden,DType::BF16,{F,batch}).slice(1,b,1),s.carry[0].slice(1,slot,1),s.stream);
        copy(s.field(s.layout.frozen_ids,DType::I32,{S,batch}).slice(1,b,1),s.ids[0].slice(1,slot,1),s.stream);
        copy(s.field(s.layout.frozen_count,DType::I32,{batch}).slice(0,b,1),s.count[0].slice(0,slot,1),s.stream);
    }
    s.controls(slots,lengths,1,proposals,positions);auto result=s.enqueue(1,batch,true);
    const auto output=s.mtp->output(1,batch);
    for(int b=0;b<batch;++b) copy(s.carry[0].slice(1,slots[b],1),output.carried_hidden.slice(2,b,1),s.stream);
    CUDA_CHECK(cudaStreamSynchronize(s.stream));
    for(int slot:slots) ++s.frontier[slot];
    return result;
}
void NativeDraftRuntime::discard_mtp(int slot) {
    auto& s=*impl_;s.check(slot);if(!s.mtp || s.seed_frontier[slot]<0) return;
    s.image(1,0,slot);CUDA_CHECK(cudaStreamSynchronize(s.stream));
    s.frontier[slot]=s.seed_frontier[slot];s.seed_frontier[slot]=-1;
    s.allocation[slot].trim_tokens(s.frontier[slot]);
}
Tensor NativeDraftRuntime::reseed_mtp(std::span<const int> slots,std::span<const TokenId> tokens,
    std::span<const std::array<int,3>> positions) {
    auto& s=*impl_;s.require(true);s.validate_rows(slots);
    if(tokens.size()!=slots.size() || positions.size()!=slots.size()) throw std::invalid_argument("native MTP reseed rows");
    for(std::size_t b=0;b<slots.size();++b) {
        const int slot=slots[b];
        const int committed=s.seed_frontier[slot]<0?s.frontier[slot]:s.seed_frontier[slot];
        if(committed<1 || tokens[b]<0 || tokens[b]>=kNativeTokenDomain ||
           std::any_of(positions[b].begin(),positions[b].end(),[](int v){return v<0 || v>=262144;}))
            throw std::invalid_argument("native MTP reseed needs a committed source position");
    }
    std::vector<NativeMtpInputRow> rows;
    for(std::size_t b=0;b<slots.size();++b) {
        const int slot=slots[b];discard_mtp(slot);--s.frontier[slot];
        rows.push_back({slot,s.target[0].slice(1,slot,1),tokens.subspan(b,1),positions.subspan(b,1)});
    }
    return extend_mtp(rows,1);
}

void NativeDraftRuntime::append_dflash(int slot,const Tensor& features,std::span<const int> positions) {
    auto& s=*impl_;s.require(false);s.validate_rows(std::span(&slot,1));const int width=int(positions.size());
    if(width<1 || width>s.layout.width || features.dtype!=DType::BF16 || !features.data ||
       !features.is_contiguous() || features.ne[0]!=12800 || features.ne[1]!=width || features.ne[2]!=1 || features.ne[3]!=1 ||
       s.frontier[slot]+width>s.config.context_tokens ||
       pages_for_tokens(s.frontier[slot]+width)>s.allocation[slot].page_entitlement())
        throw std::invalid_argument("native DFlash accepted feature prefix");
    for(int t=0;t<width;++t) if(positions[t]!=s.frontier[slot]+t)
        throw std::invalid_argument("native PixelML DFlash requires causal token ordinals, not target MRoPE coordinates");
    CUDA_CHECK(cudaStreamSynchronize(s.stream));s.allocation[slot].materialize_tokens(s.frontier[slot]+width,s.stream);
    auto input=s.field(s.layout.features,DType::BF16,{12800,width,1});copy(input,features,s.stream);
    std::size_t offset=0;s.upload(s.layout.slots,std::span(&slot,1),offset);
    s.upload(s.layout.valid,std::span(&width,1),offset);s.upload(s.layout.positions,positions,offset);
    s.upload(s.layout.cache_positions,positions,offset);
    auto rope=s.field(s.layout.positions,DType::I32,{width,1}),cache=s.field(s.layout.cache_positions,DType::I32,{width,1});
    auto counts=s.field(s.layout.valid,DType::I32,{1}),rows=s.field(s.layout.slots,DType::I32,{1});
    s.dflash->append_accepted_context(input,rope,cache,counts,rows,
        {std::uint32_t(width),std::uint32_t(width)},s.stream);
    CUDA_CHECK(cudaStreamSynchronize(s.stream));
    s.frontier[slot]+=width;
}
Tensor NativeDraftRuntime::draft_dflash(std::span<const int> slots,std::span<const TokenId> anchors,
    std::span<const int> positions,int width) {
    auto& s=*impl_;s.require(false);s.validate_rows(slots);
    if(width<1 || width>7 || anchors.size()!=slots.size() || positions.size()!=slots.size())
        throw std::invalid_argument("native DFlash compact proposal block");
    for(std::size_t b=0;b<slots.size();++b) if(anchors[b]<0 || anchors[b]>=kNativeTokenDomain ||
        positions[b]!=s.frontier[slots[b]] || positions[b]+width>s.config.context_tokens)
            throw std::invalid_argument("native DFlash source ordinal or block capacity");
    CUDA_CHECK(cudaStreamSynchronize(s.stream));const std::vector<int> valid(slots.size(),width);
    s.controls(slots,valid,width,anchors,positions);return s.enqueue(width,int(slots.size()),false);
}
} // namespace ninfer::targets::qwen4
