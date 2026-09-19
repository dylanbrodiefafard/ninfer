#include "targets/qwen4/engine_program.h"
#include "core/device.h"
#include "core/layout.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/nll_from_logits.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/speculative_round.h"
#include "runtime/contract/sampling.h"
#include "runtime/contract/score.h"
#include "runtime/contract/tool_masks.h"
#include "runtime/contract/typical_cycle.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace ninfer::targets::qwen4 {
namespace {
using Clock=std::chrono::steady_clock;
constexpr int vocabulary=kNativeTokenDomain,physical=kNativeHeadRows;
constexpr int mask_words=(vocabulary+31)/32;
double seconds(Clock::time_point start) {return std::chrono::duration<double>(Clock::now()-start).count();}
void copy(const Tensor& dst,const Tensor& src,cudaStream_t stream) {
    if(dst.bytes()!=src.bytes()) throw std::logic_error("Qwen4 control copy geometry");
    CUDA_CHECK(cudaMemcpyAsync(dst.data,src.data,src.bytes(),cudaMemcpyDeviceToDevice,stream));
}
Tensor column(const Tensor& t,int row,int width,int index,int channels) {
    return {static_cast<std::byte*>(t.data)+std::size_t(row*width+index)*channels*2,
            DType::BF16,{channels,1}};
}
std::array<int,3> position(const text::qwen::PreparedPromptData& prompt,int t) {
    return {prompt.position_axis(0)[t],prompt.position_axis(1)[t],prompt.position_axis(2)[t]};
}
bool media_prefix_equal(const text::qwen::PreparedPromptData& a,
                        const text::qwen::PreparedPromptData& b,int frontier) {
    auto live=[&](const text::qwen::VisionItem& item) {
        return std::any_of(item.token_spans.begin(),item.token_spans.end(),
            [&](const auto& span){return span.begin<std::size_t(frontier);});
    };
    auto ai=a.vision_items.begin(),bi=b.vision_items.begin();
    for(;;) {
        while(ai!=a.vision_items.end()&&!live(*ai)) ++ai;
        while(bi!=b.vision_items.end()&&!live(*bi)) ++bi;
        if(ai==a.vision_items.end()||bi==b.vision_items.end())
            return ai==a.vision_items.end()&&bi==b.vision_items.end();
        if(ai->modality!=bi->modality||ai->content_digest!=bi->content_digest||
           ai->grid.temporal!=bi->grid.temporal||ai->grid.height!=bi->grid.height||
           ai->grid.width!=bi->grid.width||ai->token_spans.size()!=bi->token_spans.size()) return false;
        for(std::size_t j=0;j<ai->token_spans.size();++j)
            if(ai->token_spans[j].begin!=bi->token_spans[j].begin||
               ai->token_spans[j].count!=bi->token_spans[j].count) return false;
        ++ai;++bi;
    }
}
struct SamplingStorage {
    Tensor counts,last_logits,mtp_logits,saved_logits,saved_mtp,checkpoint_logits,checkpoint_mtp;
};
template<class Arena> SamplingStorage lane_storage(Arena& arena) {
    return {arena.alloc(DType::I32,{vocabulary}),arena.alloc(DType::BF16,{physical}),
        arena.alloc(DType::BF16,{physical}),arena.alloc(DType::BF16,{physical}),
        arena.alloc(DType::BF16,{physical}),arena.alloc(DType::BF16,{physical}),
        arena.alloc(DType::BF16,{physical})};
}
struct BatchStorage {
    Tensor configs,masks,mask_configs,logits,ids,positions,extents,lengths,anchors,
           drafts,target,licensed,counts,accepted,nll,score_targets;
};
template<class Arena> BatchStorage batch_storage(Arena& a,int c,int w,int prefill) {
    return {a.alloc(DType::I32,{int((sizeof(ops::SamplingConfig)+3)/4),c}),
        a.alloc(DType::I32,{mask_words,w,c}),
        a.alloc(DType::I32,{int((sizeof(ops::SamplingConfig)+3)/4),c}),
        a.alloc(DType::BF16,{physical,c}),a.alloc(DType::I32,{w,c}),
        a.alloc(DType::I32,{c}),a.alloc(DType::I32,{c}),a.alloc(DType::I32,{c}),
        a.alloc(DType::I32,{c}),a.alloc(DType::I32,{std::max(1,w-1),c}),
        a.alloc(DType::I32,{w,c}),a.alloc(DType::I32,{w,c}),
        a.alloc(DType::I32,{c}),a.alloc(DType::I32,{c}),a.alloc(DType::FP32,{prefill}),
        a.alloc(DType::I32,{prefill})};
}
std::size_t storage_bytes(const NativeRuntimeConfig& c) {
    WorkspaceLayoutBuilder a;
    for(int i=0;i<c.requests;++i) (void)lane_storage(a);
    (void)batch_storage(a,c.requests,c.verify_width,c.prefill_width);
    return a.peak_bytes();
}
std::size_t workspace_bytes(const NativeRuntimeConfig& c) {
    const auto sampling=ops::sampling_workspace_capacity_bytes(vocabulary,1,c.requests);
    const auto spec=c.verify_width>1?ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
        vocabulary,1,c.verify_width-1,1,c.requests):0;
    return std::max<std::size_t>({sampling,spec,256});
}
}

NativeRuntimeConfig native_runtime_config(const EngineOptions& o,std::uint32_t kv_tokens) {
    if(o.kv_ram_capacity_bytes||o.kv_disk_capacity_bytes||!o.kv_disk_location.empty())
        throw std::invalid_argument("Qwen4 supports GPU prefix retention, not host KV tiers; PLE RAM residency is separate");
    if(o.sage_attn||o.keep_frac!=1.0F||o.xattn_tau!=1.0F)
        throw std::invalid_argument("Qwen4 uses source QSA selection, not dense-family attention approximations");
    if(o.kv_cache!=KvCacheStorage::Nvfp4&&o.kv_cache!=KvCacheStorage::BFloat16)
        throw std::invalid_argument("Qwen4 QSA KV storage must be nvfp4 or bf16");
    if(o.prefill_chunk<1||o.prefill_chunk>4096||o.max_context<1||o.max_context>262144||
       o.max_concurrency<1||o.max_concurrency>4||kv_tokens<64||kv_tokens%64)
        throw std::invalid_argument("Qwen4 startup execution capacities are invalid");
    if(o.speculative.adaptive_draft||o.speculative.proposal_head!=ProposalHead::Full)
        throw std::invalid_argument("Qwen4 source companions use fixed-width full-vocabulary proposals");
    if(o.context_checkpoint_marks&&!o.context_checkpoint_marks->empty())
        throw std::invalid_argument("Qwen4 supports one prompt/rewrite checkpoint, not a context-checkpoint ladder");
    NativeRuntimeConfig c;c.requests=o.max_concurrency;c.context_tokens=o.max_context;c.kv_tokens=kv_tokens;
    c.prefill_width=o.prefill_chunk;c.kv_dtype=o.kv_cache;c.vision=o.enable_vision;c.use_cuda_graph=o.use_cuda_graph;
    c.mtp=o.speculative.backend==SpeculativeBackend::Mtp;c.dflash=o.speculative.backend==SpeculativeBackend::DFlash;
    if(o.speculative.backend!=SpeculativeBackend::None&&!c.mtp&&!c.dflash)
        throw std::invalid_argument("Qwen4 speculative backend is invalid");
    if(o.speculative.draft_tokens>15) throw std::invalid_argument("Qwen4 source draft width exceeds 15");
    const int drafts=(c.mtp||c.dflash)?int(o.speculative.draft_tokens?o.speculative.draft_tokens:3):0;
    if(drafts>15||(c.dflash&&drafts>7)) throw std::invalid_argument("Qwen4 source draft width exceeds the qualified envelope");
    if(o.speculative.dflash_verify_width&&o.speculative.dflash_verify_width!=std::uint32_t(drafts+1))
        throw std::invalid_argument("Qwen4 DFlash uses source chain verification, not packed-tree verification");
    if(!c.mtp&&!c.dflash&&(o.speculative.draft_tokens||o.speculative.dflash_verify_width))
        throw std::invalid_argument("Qwen4 draft controls require a speculative backend");
    c.verify_width=std::max(2,drafts+1);
    return c;
}
std::size_t engine_control_device_bytes(const NativeRuntimeConfig& c) {
    return storage_bytes(c)+workspace_bytes(c);
}

struct EngineProgram::Impl {
    struct Lane {
        SamplingStorage gpu;
        text::qwen::PreparedPromptData prompt,saved_prompt,checkpoint_prompt;
        std::vector<TokenId> inputs,saved_inputs,generated,checkpoint_inputs;
        std::vector<std::array<int,3>> input_positions,saved_positions,checkpoint_positions;
        RequestPlan plan;
        const text::qwen::OutputSession* output=nullptr;
        ops::SamplingConfig sampling;
        GenerationTimings timings;
        SpeculativeStats stats;
        Clock::time_point start;
        TokenId anchor=0,saved_anchor=0,checkpoint_anchor=0;
        int cursor=0,pending_count=0,saved_frontier=0,checkpoint_frontier=0,capture_frontier=0;
        int captured_checkpoint=0,restored_checkpoint=0;
        PrefixReusePath checkpoint_path=PrefixReusePath::RestoreContextCheckpoint;
        bool active=false,prefilling=false,prefill_pending=false,reasoning=false,retained=false;
        std::uint64_t use_tick=0;
    };
    NativeRuntimeConfig config;
    EngineOptions options;
    DeviceContext& device;
    NativeRuntime runtime;
    DeviceArena storage,workspace;
    std::array<Lane,4> lanes;
    BatchStorage batch;
    runtime::ToolMaskExchange masks;
    PinnedHostBuffer host;
    std::array<ops::SamplingConfig,4> configs{};
    std::array<const text::qwen::OutputSession*,4> outputs{};
    std::array<int,4> pending_slots{},pending_widths{},licensed_counts{};
    std::array<TokenId,64> licensed{},verified{};
    std::array<bool,4> cycle{};
    NativeBatchOutput pending_output;
    int pending_batch=0,pending_envelope=0;
    std::uint64_t tick=0;

    Impl(const NativeModelView& model,const NativeRuntimeConfig& c,const EngineOptions& o,
         DeviceContext& d):config(c),options(o),device(d),runtime(model,c,d),storage(storage_bytes(c)),
         workspace(workspace_bytes(c)),batch(allocate_lanes_and_batch()),
         masks(batch.masks,batch.mask_configs),host(std::max<std::size_t>(65536,c.prefill_width*8)) {
        CUDA_CHECK(cudaMemsetAsync(storage.base(),0,storage.capacity(),device.stream));
        for(int i=0;i<c.requests;++i) {
            lanes[i].inputs.reserve(c.context_tokens);lanes[i].saved_inputs.reserve(c.context_tokens);
            lanes[i].generated.reserve(c.context_tokens);
            lanes[i].input_positions.reserve(c.context_tokens);lanes[i].saved_positions.reserve(c.context_tokens);
        }
        device.synchronize();
    }
    BatchStorage allocate_lanes_and_batch() {
        for(int i=0;i<config.requests;++i) lanes[i].gpu=lane_storage(storage);
        return batch_storage(storage,config.requests,config.verify_width,config.prefill_width);
    }
    Lane& lane(std::uint32_t id) {
        if(id>=std::uint32_t(config.requests)) throw std::out_of_range("Qwen4 request slot");
        return lanes[id];
    }
    const Lane& lane(std::uint32_t id) const {return const_cast<Impl*>(this)->lane(id);}
    void upload(const Tensor& dst,const void* data,std::size_t bytes) {
        if(bytes>dst.bytes()) throw std::logic_error("Qwen4 control upload capacity");
        CUDA_CHECK(cudaMemcpyAsync(dst.data,data,bytes,cudaMemcpyHostToDevice,device.stream));
    }
    void download(void* dst,const Tensor& src,std::size_t bytes) {
        if(bytes>src.bytes()||bytes>host.size()) throw std::logic_error("Qwen4 control download capacity");
        CUDA_CHECK(cudaMemcpyAsync(host.data(),src.data,bytes,cudaMemcpyDeviceToHost,device.stream));
        device.synchronize();std::memcpy(dst,host.data(),bytes);
    }
    ops::SamplingConfig sampling(Lane& s) {
        auto config=s.sampling;
        config.typical_exclude=runtime::typical_exclude_for_sample(s.reasoning,config.p_less,
                                                                 config.temperature,s.generated);
        return config;
    }
    void bind_masks(std::span<const int> slots) {
        for(std::size_t b=0;b<slots.size();++b) {
            auto& s=lane(slots[b]);configs[b]=sampling(s);outputs[b]=s.output;
            cycle[b]=configs[b].typical_exclude>=0;
        }
        masks.bind({outputs.data(),slots.size()},{configs.data(),slots.size()});
    }
    void sample_roots(std::span<const int> slots,int purpose) {
        bind_masks(slots);
        std::array<int,4> positions{};
        for(std::size_t b=0;b<slots.size();++b) {
            auto& s=lane(slots[b]);copy(column(batch.logits,b,1,0,physical),s.gpu.last_logits,device.stream);
            configs[b]=masks.root(b,device.stream);
            positions[b]=runtime.frontier(slots[b])-1;
        }
        upload(batch.configs,configs.data(),slots.size()*sizeof(ops::SamplingConfig));
        upload(batch.positions,positions.data(),slots.size()*4);
        Tensor logits(batch.logits.data,DType::BF16,{physical,int(slots.size())});
        Tensor result(batch.licensed.data,DType::I32,{int(slots.size())});
        Tensor keys(batch.positions.data,DType::I32,{int(slots.size())});
        workspace.reset();ops::sample(logits,result,vocabulary,
            static_cast<const ops::SamplingConfig*>(batch.configs.data),keys,purpose,workspace,device.stream);
        download(licensed.data(),result,slots.size()*4);
        for(std::size_t b=0;b<slots.size();++b) {licensed_counts[b]=1;lane(slots[b]).pending_count=1;}
    }
    void private_extend(int slot,const NativeBatchOutput& out,int row,int width,int count,
                        std::span<const TokenId> next,std::span<const std::array<int,3>> positions) {
        if(config.mtp&&count) {
            Tensor hidden(column(out.carried_hidden,row,width,0,10240).data,DType::BF16,{10240,count});
            NativeMtpInputRow input{slot,hidden,next.first(count),positions.first(count)};
            Tensor logits=runtime.extend_mtp({&input,1},count);
            copy(lane(slot).gpu.mtp_logits,column(logits,0,count,count-1,physical),device.stream);
        }
        if(config.dflash&&count) {
            std::array<int,4096> logical{};
            const int start=runtime.frontier(slot)-count;
            for(int t=0;t<count;++t) logical[t]=start+t;
            Tensor features(column(out.dflash_features,row,width,0,12800).data,DType::BF16,{12800,count});
            runtime.append_dflash(slot,features,{logical.data(),std::size_t(count)});
        }
    }
    void capture_prompt(int slot) {
        auto& s=lane(slot);runtime.retain(slot,NativeCheckpoint::Prompt);
        s.checkpoint_prompt=s.prompt;s.checkpoint_prompt.release_media_payload();
        s.checkpoint_inputs=s.inputs;s.checkpoint_positions=s.input_positions;
        s.checkpoint_frontier=runtime.frontier(slot);s.captured_checkpoint=s.checkpoint_frontier;
        s.checkpoint_anchor=s.cursor<int(s.prompt.token_ids.size())?s.prompt.token_ids[s.cursor]:s.anchor;
        copy(s.gpu.checkpoint_logits,s.gpu.last_logits,device.stream);
        copy(s.gpu.checkpoint_mtp,s.gpu.mtp_logits,device.stream);
    }
    runtime::PrefillStepResult advance(std::uint32_t id) {
        auto& s=lane(id);
        if(!s.prefilling||pending_batch) throw std::logic_error("Qwen4 prefill boundary");
        const auto start=Clock::now();
        int remaining=int(s.prompt.token_ids.size())-s.cursor;
        int count=std::min(config.prefill_width,remaining);
        if(s.capture_frontier>s.cursor) count=std::min(count,s.capture_frontier-s.cursor);
        NativeBatchOutput out;
        std::vector<std::array<int,3>> positions;
        if(count) {
            positions.reserve(count);
            for(int t=0;t<count;++t) positions.push_back(position(s.prompt,s.cursor+t));
            NativeInputRow row{int(id),{s.prompt.token_ids.data()+s.cursor,std::size_t(count)},positions};
            runtime.prepare({&row,1},count);out=runtime.enqueue(false);
            NativeCommitRow commit{int(id),count};runtime.commit({&commit,1});
            copy(s.gpu.last_logits,column(out.logits,0,count,count-1,physical),device.stream);
            s.inputs.insert(s.inputs.end(),row.token_ids.begin(),row.token_ids.end());
            s.input_positions.insert(s.input_positions.end(),positions.begin(),positions.end());
            s.cursor+=count;
        }
        const bool complete=s.cursor==int(s.prompt.token_ids.size());
        if(complete) {
            const int slot=int(id);sample_roots({&slot,1},ops::kSamplePurposePrefill);
            s.anchor=licensed[0];s.prefill_pending=true;s.prefilling=false;
            if(!count&&config.mtp) {
                const auto pos=position(s.prompt,s.cursor-1);
                auto logits=runtime.reseed_mtp({&slot,1},{&s.anchor,1},{&pos,1});
                copy(s.gpu.mtp_logits,Tensor(logits.data,DType::BF16,{physical}),device.stream);
            }
        }
        if(count) {
            std::vector<TokenId> next(count);
            for(int t=0;t<count;++t) next[t]=(s.cursor-count+t+1<int(s.prompt.token_ids.size()))?
                s.prompt.token_ids[s.cursor-count+t+1]:s.anchor;
            private_extend(id,out,0,count,count,next,positions);
        }
        if(s.capture_frontier==s.cursor&&s.capture_frontier>0) capture_prompt(id);
        runtime.synchronize();
        s.timings.prefill_seconds+=seconds(start);
        s.timings.total_seconds=seconds(s.start);
        if(complete) s.timings.first_token_seconds=s.timings.total_seconds;
        return {{s.plan.value.prompt_tokens,s.plan.reuse,s.plan.reuse?s.plan.reuse_path:
            PrefixReusePath::FullReset,s.plan.reuse?PrefixReuseSource::VramResident:PrefixReuseSource::None},
            {complete?std::span<const TokenId>(licensed.data(),1):std::span<const TokenId>()},
            std::uint32_t(count),complete,complete};
    }
};

EngineProgram::EngineProgram(const NativeModelView& m,const NativeRuntimeConfig& c,
    const EngineOptions& o,DeviceContext& d):impl_(std::make_unique<Impl>(m,c,o,d)) {}
EngineProgram::~EngineProgram() {try {impl_->runtime.synchronize();}catch(...) {}}

RequestBasePlan EngineProgram::plan_request_base(const text::qwen::PreparedPrompt& prompt,
    const runtime::ResolvedExecutionOptions& options) {
    const auto& data=text::qwen::PreparedPromptAccess::view(prompt);
    if(data.token_ids.empty()||data.token_ids.size()>std::size_t(impl_->config.context_tokens))
        throw std::invalid_argument("Qwen4 prompt exceeds context capacity");
    for(TokenId id:data.token_ids) if(id<0||id>=vocabulary)
        throw std::invalid_argument("Qwen4 input is outside the tokenizer domain");
    RequestBasePlan plan;plan.options=options;
    if(options.capture_context_checkpoint&&!context_checkpoint_capture_available(
        options.allow_prefix_reuse,impl_->options.speculative.backend))
        throw std::invalid_argument("capture_context_checkpoint requires prefix reuse and a speculative backend");
    plan.value.prompt_tokens=data.token_ids.size();
    plan.value.requested_output_tokens=options.requested_output_tokens;
    plan.value.effective_output_tokens=std::min<std::uint32_t>(options.requested_output_tokens,
        impl_->config.context_tokens-data.token_ids.size()+1);
    plan.value.effective_limit_reason=plan.value.effective_output_tokens<options.requested_output_tokens?
        FinishReason::ContextCapacity:FinishReason::OutputLimit;
    plan.value.admission={1,(plan.value.prompt_tokens+std::max(1U,plan.value.effective_output_tokens)-1+63)/64,0};
    plan.value.service_work_quanta=plan.value.prompt_tokens+plan.value.effective_output_tokens;
    return plan;
}
RequestPlan EngineProgram::plan_request_for_lane(std::uint32_t id,
    const text::qwen::PreparedPrompt& prompt,const RequestBasePlan& base) {
    auto& s=impl_->lane(id);const auto& data=text::qwen::PreparedPromptAccess::view(prompt);
    RequestPlan plan{base.value,base.options,0};
    if(!s.retained||!base.options.allow_prefix_reuse||base.options.force_cold_prefill||!data.identity.reusable) return plan;
    const auto consider=[&](NativeCheckpoint kind,int n,const auto& identity,const auto& inputs,
                            const auto& positions,PrefixReusePath path) {
        if(n<=int(plan.reuse)||!impl_->runtime.has_retained(id,kind)||!identity.identity.reusable||
           std::size_t(n)>data.token_ids.size()||inputs.size()!=std::size_t(n)||
           !std::equal(inputs.begin(),inputs.end(),data.token_ids.begin())||!media_prefix_equal(identity,data,n)) return;
        for(int t=0;t<n;++t) if(position(data,t)!=positions[t]) return;
        plan.reuse=n;plan.checkpoint=kind;plan.reuse_path=path;
        plan.value.reusable_prompt_tokens=n;plan.value.reuse_source=PrefixReuseSource::VramResident;
    };
    consider(NativeCheckpoint::Retained,s.saved_frontier,s.saved_prompt,s.saved_inputs,s.saved_positions,PrefixReusePath::AppendAtFrontier);
    consider(NativeCheckpoint::Prompt,s.checkpoint_frontier,s.checkpoint_prompt,s.checkpoint_inputs,s.checkpoint_positions,s.checkpoint_path);
    return plan;
}
bool EngineProgram::can_admit_lane(std::uint32_t id,const RequestPlan& p) const noexcept {
    try{return impl_->runtime.can_reserve(id,p.value.prompt_tokens+std::max(1U,p.value.effective_output_tokens)-1);}catch(...){return false;}
}
bool EngineProgram::can_admit_lane_after_retained_eviction(std::uint32_t id,const RequestPlan& p) const noexcept {
    return can_admit_lane(id,p);
}
bool EngineProgram::can_admit_lane_after_releasing(std::uint32_t id,const RequestPlan& p,
    std::span<const std::uint32_t> released) const noexcept {
    try {
        std::array<int,4> slots{};if(released.size()>slots.size()) return false;
        std::size_t count=0;for(auto slot:released) if(slot!=id) slots[count++]=slot;
        return impl_->runtime.can_reserve(id,p.value.prompt_tokens+std::max(1U,p.value.effective_output_tokens)-1,
                                          {slots.data(),count});
    }catch(...){return false;}
}
runtime::AdmissionResources EngineProgram::admission_capacity() const noexcept {
    return {std::uint32_t(impl_->config.requests),std::uint32_t(impl_->config.kv_tokens/64),0};
}
runtime::PrefillStepResult EngineProgram::start_prefill_lane(std::uint32_t id,
    text::qwen::PreparedPrompt&& prompt,RequestPlan&& plan,runtime::TransientRegion,
    const text::qwen::OutputSession* output) {
    auto& p=*impl_;auto& s=p.lane(id);
    if(s.active||p.pending_batch) throw std::logic_error("Qwen4 occupied request slot");
    s.plan=std::move(plan);s.prompt=text::qwen::PreparedPromptAccess::take(std::move(prompt));
    s.output=output;s.generated.clear();s.timings={};s.stats={};s.start=Clock::now();
    s.stats.backend=p.options.speculative.backend;s.stats.enabled=p.config.mtp||p.config.dflash;
    s.stats.draft_window=s.stats.enabled?p.config.verify_width-1:0;
    s.stats.accepted_per_position.assign(s.stats.draft_window,0);
    s.stats.rounds_per_draft.assign(s.stats.draft_window+1,0);
    s.reasoning=s.prompt.starts_in_reasoning;
    s.sampling={};const auto& source=s.plan.options.sampling;
    s.sampling.temperature=source.temperature;s.sampling.top_k=source.top_k;
    s.sampling.top_p=source.top_p;s.sampling.min_p=source.min_p;
    s.sampling.presence_penalty=source.presence_penalty;s.sampling.frequency_penalty=source.frequency_penalty;
    s.sampling.p_less=source.p_less;s.sampling.seed=source.seed;
    s.sampling.token_counts=static_cast<int*>(s.gpu.counts.data);
    s.sampling.suppressed_token_count=s.plan.options.suppressed_token_count;
    std::copy_n(s.plan.options.suppressed_token_ids.begin(),s.sampling.suppressed_token_count,s.sampling.suppressed_tokens);
    CUDA_CHECK(cudaMemsetAsync(s.gpu.counts.data,0,s.gpu.counts.bytes(),p.device.stream));
    if(s.plan.reuse) {
        p.runtime.restore(id,s.plan.checkpoint);
        if(s.plan.checkpoint==NativeCheckpoint::Prompt) {
            s.saved_inputs=s.checkpoint_inputs;s.saved_positions=s.checkpoint_positions;s.saved_prompt=s.checkpoint_prompt;
            s.saved_frontier=s.checkpoint_frontier;s.saved_anchor=s.checkpoint_anchor;
            copy(s.gpu.saved_logits,s.gpu.checkpoint_logits,p.device.stream);
            copy(s.gpu.saved_mtp,s.gpu.checkpoint_mtp,p.device.stream);
            s.restored_checkpoint=s.saved_frontier;
        } else s.restored_checkpoint=0;
        s.inputs=s.saved_inputs;s.input_positions=s.saved_positions;
        copy(s.gpu.last_logits,s.gpu.saved_logits,p.device.stream);
        copy(s.gpu.mtp_logits,s.gpu.saved_mtp,p.device.stream);
    } else {
        p.runtime.evict_retained(id);s.retained=false;s.saved_frontier=0;
        p.runtime.reset(id);s.inputs.clear();s.input_positions.clear();s.checkpoint_frontier=0;s.restored_checkpoint=0;
    }
    if(!p.runtime.reserve(id,s.plan.value.prompt_tokens+std::max(1U,s.plan.value.effective_output_tokens)-1))
        throw std::runtime_error("Qwen4 request page entitlement is no longer available");
    if(s.plan.reuse&&s.plan.checkpoint==NativeCheckpoint::Prompt) p.runtime.retain(id);
    // The last private key includes the old next-token seed. Recompute it after every
    // restore, once the exclusive private page mapping is bound for this request.
    if(p.config.mtp&&s.plan.reuse&&s.plan.reuse<s.prompt.token_ids.size()) {
        const int slot=id;const TokenId next=s.prompt.token_ids[s.plan.reuse];const auto pos=s.saved_positions.back();
        auto logits=p.runtime.reseed_mtp({&slot,1},{&next,1},{&pos,1});
        copy(s.gpu.mtp_logits,Tensor(logits.data,DType::BF16,{physical}),p.device.stream);
    }
    s.captured_checkpoint=0;s.capture_frontier=0;
    if(s.plan.options.allow_prefix_reuse&&s.prompt.identity.rewrite_checkpoint) {
        s.capture_frontier=s.prompt.identity.rewrite_checkpoint->frontier;
        s.checkpoint_path=s.prompt.identity.rewrite_checkpoint->kind==text::qwen::RewriteCheckpointKind::TurnClosure?
            PrefixReusePath::RestoreTurnCheckpoint:PrefixReusePath::RestoreResponseCheckpoint;
    }
    if(s.plan.options.capture_context_checkpoint) {
        s.capture_frontier=s.prompt.token_ids.size();s.checkpoint_path=PrefixReusePath::RestoreContextCheckpoint;
    }
    s.cursor=s.plan.reuse;s.active=true;s.prefilling=true;s.prefill_pending=false;s.pending_count=0;
    if(s.prompt.has_media()) {
        std::vector<VisionGrid> grids;std::vector<int> columns;
        for(const auto& item:s.prompt.vision_items) {
            grids.push_back({item.grid.temporal,item.grid.height,item.grid.width});
            for(const auto& span:item.token_spans) for(std::size_t t=0;t<span.count;++t) columns.push_back(span.begin+t);
        }
        const auto start=Clock::now();p.runtime.prepare_vision(id,s.prompt.patches,grids,columns);
        p.runtime.synchronize();s.timings.vision_seconds=seconds(start);s.prompt.release_media_payload();
    } else p.runtime.clear_vision(id);
    return p.advance(id);
}
runtime::PrefillStepResult EngineProgram::advance_prefill_lane(std::uint32_t id) {return impl_->advance(id);}

runtime::BatchedGeneratedRound EngineProgram::decode_batch(std::span<const std::uint32_t> ids,
    std::span<const runtime::RoundBudget> budgets) {
    auto& p=*impl_;
    if(ids.empty()||ids.size()>std::size_t(p.config.requests)||ids.size()!=budgets.size()||p.pending_batch)
        throw std::invalid_argument("Qwen4 decode batch boundary");
    const int B=ids.size();int K=(p.config.mtp||p.config.dflash)?p.config.verify_width-1:0;
    std::array<int,4> slots{},base{},anchors{},extents{};
    const auto start=Clock::now();
    for(int b=0;b<B;++b) {
        auto& s=p.lane(ids[b]);
        if(!s.active||s.prefilling||s.prefill_pending||s.pending_count||!budgets[b].generated_tokens_remaining)
            throw std::logic_error("Qwen4 decode requires a published anchor");
        for(int j=0;j<b;++j) if(ids[j]==ids[b]) throw std::invalid_argument("duplicate Qwen4 decode slot");
        slots[b]=ids[b];base[b]=p.runtime.frontier(ids[b]);anchors[b]=s.anchor;
        const int room=p.config.context_tokens-base[b];
        if(room<1) throw std::invalid_argument("Qwen4 context exhausted");
        K=std::min({K,int(budgets[b].generated_tokens_remaining)-1,room-1});
    }
    std::array<TokenId,60> drafts{};
    if(K&&p.config.mtp) {
        std::array<TokenId,4> previous{};
        std::array<std::array<int,3>,4> source_positions{};
        for(int j=0;j<K;++j) {
            Tensor logits;
            if(j==0) {
                for(int b=0;b<B;++b) copy(column(p.batch.logits,b,1,0,physical),p.lane(slots[b]).gpu.mtp_logits,p.device.stream);
                logits={p.batch.logits.data,DType::BF16,{physical,B}};
            } else {
                for(int b=0;b<B;++b) {
                    const int pos=base[b]+j-1+p.lane(slots[b]).prompt.rope_delta;
                    source_positions[b]={pos,pos,pos};
                }
                logits=p.runtime.draft_mtp({slots.data(),std::size_t(B)},
                    {previous.data(),std::size_t(B)},{source_positions.data(),std::size_t(B)});
                logits=Tensor(logits.data,DType::BF16,{physical,B});
            }
            Tensor tokens(p.batch.target.data,DType::I32,{B});
            ops::argmax(logits,tokens,vocabulary,p.device.stream);
            p.download(previous.data(),tokens,B*4);
            for(int b=0;b<B;++b) drafts[b*K+j]=previous[b];
        }
    } else if(K&&p.config.dflash) {
        std::array<int,4> rope{};
        // PixelML training constructs scalar positions from causal token ordinals. Target
        // multimodal MRoPE is represented in the taps, not reused as the drafter's RoPE.
        for(int b=0;b<B;++b) rope[b]=base[b];
        auto logits=p.runtime.draft_dflash({slots.data(),std::size_t(B)},
            {anchors.data(),std::size_t(B)},{rope.data(),std::size_t(B)},K);
        Tensor flat(logits.data,DType::BF16,{physical,K*B});
        Tensor tokens(p.batch.drafts.data,DType::I32,{K*B});
        ops::argmax(flat,tokens,vocabulary,p.device.stream);p.download(drafts.data(),tokens,K*B*4);
    }
    const int W=K+1;
    std::array<std::array<std::array<int,3>,16>,4> positions{};
    std::array<NativeInputRow,4> rows{};
    for(int b=0;b<B;++b) {
        p.verified[b*W]=anchors[b];
        for(int j=1;j<W;++j) p.verified[b*W+j]=drafts[b*K+j-1];
        for(int j=0;j<W;++j) {
            const int pos=base[b]+j+p.lane(slots[b]).prompt.rope_delta;
            positions[b][j]={pos,pos,pos};
        }
        rows[b]={slots[b],{p.verified.data()+b*W,std::size_t(W)},
                 {positions[b].data(),std::size_t(W)}};
        extents[b]=K;p.pending_widths[b]=W;p.pending_slots[b]=slots[b];
    }
    p.runtime.prepare({rows.data(),std::size_t(B)},W);
    p.pending_output=p.runtime.enqueue(W>1);
    p.bind_masks({slots.data(),std::size_t(B)});
    p.upload(p.batch.positions,base.data(),B*4);
    if(K) {
        std::array<int,4> widths{};std::fill_n(widths.begin(),B,W);
        p.upload(p.batch.ids,p.verified.data(),W*B*4);
        p.upload(p.batch.counts,widths.data(),B*4);
        Tensor inputs(p.batch.ids.data,DType::I32,{W,B}),valid(p.batch.counts.data,DType::I32,{B});
        const auto* configs=p.masks.enqueue(inputs,nullptr,valid,p.device.stream);
        Tensor logits(p.pending_output.logits.data,DType::BF16,{physical,W*B});
        Tensor selected(p.batch.target.data,DType::I32,{W*B});
        ops::argmax(logits,selected,vocabulary,configs,W,p.device.stream);
        p.upload(p.batch.drafts,drafts.data(),K*B*4);p.upload(p.batch.extents,extents.data(),B*4);
        p.upload(p.batch.lengths,base.data(),B*4);p.upload(p.batch.anchors,anchors.data(),B*4);
        Tensor target(p.batch.target.data,DType::I32,{W,B}),draft(p.batch.drafts.data,DType::I32,{K,B}),
            extent(p.batch.extents.data,DType::I32,{B}),length(p.batch.lengths.data,DType::I32,{B}),
            anchor(p.batch.anchors.data,DType::I32,{B}),licensed(p.batch.licensed.data,DType::I32,{W,B}),
            counts(p.batch.counts.data,DType::I32,{B}),accepted(p.batch.accepted.data,DType::I32,{B});
        p.workspace.reset();ops::speculative_accept_greedy_drafts(target,p.pending_output.logits,draft,
            extent,length,anchor,licensed,counts,accepted,vocabulary,configs,p.workspace,p.device.stream);
        p.download(p.licensed.data(),licensed,W*B*4);
        p.download(p.licensed_counts.data(),counts,B*4);p.masks.rethrow_error();
    } else {
        for(int b=0;b<B;++b) p.configs[b]=p.masks.root(b,p.device.stream);
        p.upload(p.batch.configs,p.configs.data(),B*sizeof(ops::SamplingConfig));
        Tensor logits(p.pending_output.logits.data,DType::BF16,{physical,B}),
            result(p.batch.licensed.data,DType::I32,{B}),keys(p.batch.positions.data,DType::I32,{B});
        p.workspace.reset();ops::sample(logits,result,vocabulary,
            static_cast<const ops::SamplingConfig*>(p.batch.configs.data),keys,
            ops::kSamplePurposeDecode,p.workspace,p.device.stream);
        p.download(p.licensed.data(),result,B*4);std::fill_n(p.licensed_counts.begin(),B,1);
    }
    p.pending_batch=B;p.pending_envelope=W;
    for(int b=0;b<B;++b) {
        auto& s=p.lane(slots[b]);s.pending_count=p.licensed_counts[b];
        s.timings.decode_seconds+=seconds(start);s.timings.total_seconds=seconds(s.start);
    }
    return {{p.licensed.data(),std::size_t(W*B)},{p.licensed_counts.data(),std::size_t(B)},
            std::uint32_t(W),p.cycle};
}

void EngineProgram::resolve_pending_batch(std::span<const std::uint32_t> ids,
    std::span<const std::uint32_t> counts,std::span<const std::uint8_t> terminal,
    std::span<const std::uint8_t> cancelled,std::span<const std::uint8_t> rejected) {
    auto& p=*impl_;const int B=p.pending_batch,W=p.pending_envelope;
    if(ids.size()!=std::size_t(B)||!B||counts.size()!=ids.size()||terminal.size()!=ids.size()||
       cancelled.size()!=ids.size()||(!rejected.empty()&&rejected.size()!=ids.size()))
        throw std::invalid_argument("Qwen4 resolution membership");
    std::array<NativeCommitRow,4> commits{};
    std::array<int,4> base{};
    for(int b=0;b<B;++b) {
        if(ids[b]!=std::uint32_t(p.pending_slots[b])||counts[b]>std::uint32_t(p.licensed_counts[b]))
            throw std::invalid_argument("Qwen4 resolution is not a licensed prefix");
        const int count=(cancelled[b]||(!rejected.empty()&&rejected[b]))?0:counts[b];
        if(!count&&!cancelled[b]&&(rejected.empty()||!rejected[b]))
            throw std::invalid_argument("Qwen4 zero publication needs cancellation or rejection");
        commits[b]={int(ids[b]),count};base[b]=p.runtime.frontier(ids[b]);
    }
    p.runtime.synchronize();
    for(int b=0;b<B;++b) {
        auto& s=p.lane(ids[b]);
        runtime::rollback_sampling_counts(s.sampling,{p.licensed.data()+b*W+commits[b].input_count,
            std::size_t(p.licensed_counts[b]-commits[b].input_count)});
        if(p.config.mtp) p.runtime.discard_mtp(ids[b]);
    }
    p.runtime.commit({commits.data(),std::size_t(B)});
    std::array<NativeMtpInputRow,4> mtp_rows{};
    std::array<std::array<std::array<int,3>,16>,4> positions{};
    int mtp_count=0,mtp_width=0;
    for(int b=0;b<B;++b) {
        auto& s=p.lane(ids[b]);const int count=commits[b].input_count;
        if(count) {
            for(int j=0;j<count;++j) {
                const int pos=base[b]+j+s.prompt.rope_delta;positions[b][j]={pos,pos,pos};
                s.inputs.push_back(p.verified[b*W+j]);s.input_positions.push_back(positions[b][j]);
                s.generated.push_back(p.licensed[b*W+j]);
            }
            s.anchor=p.licensed[b*W+count-1];
            copy(s.gpu.last_logits,column(p.pending_output.logits,b,W,count-1,physical),p.device.stream);
            if(p.config.mtp) {
                Tensor hidden(column(p.pending_output.carried_hidden,b,W,0,10240).data,DType::BF16,{10240,count});
                mtp_rows[mtp_count++]={int(ids[b]),hidden,{p.licensed.data()+b*W,std::size_t(count)},
                                      {positions[b].data(),std::size_t(count)}};
                mtp_width=std::max(mtp_width,count);
            }
            if(p.config.dflash) {
                std::array<int,16> logical{};for(int j=0;j<count;++j) logical[j]=base[b]+j;
                Tensor features(column(p.pending_output.dflash_features,b,W,0,12800).data,DType::BF16,{12800,count});
                p.runtime.append_dflash(ids[b],features,{logical.data(),std::size_t(count)});
            }
            if(W>1) {
                ++s.stats.rounds;s.stats.drafted_tokens+=W-1;s.stats.accepted_tokens+=count-1;
                s.stats.live_draft_tokens=W-1;
                ++s.stats.rounds_per_draft[W-1];
                for(int j=0;j<count-1;++j) ++s.stats.accepted_per_position[j];
            } else if(s.stats.enabled) ++s.stats.fallback_steps;
        }
        s.pending_count=0;
    }
    if(mtp_count) {
        Tensor logits=p.runtime.extend_mtp({mtp_rows.data(),std::size_t(mtp_count)},mtp_width);
        for(int b=0;b<mtp_count;++b) copy(p.lane(mtp_rows[b].slot).gpu.mtp_logits,
            column(logits,b,mtp_width,mtp_rows[b].next_tokens.size()-1,physical),p.device.stream);
    }
    p.runtime.synchronize();p.pending_batch=0;p.pending_envelope=0;p.pending_output={};
    for(int b=0;b<B;++b) if(terminal[b]||cancelled[b]) retain_lane(ids[b]);
}

void EngineProgram::retain_lane(std::uint32_t id) {
    auto& p=*impl_;auto& s=p.lane(id);
    if(!s.active||s.prefilling||s.prefill_pending||s.pending_count)
        throw std::logic_error("Qwen4 retention requires published state");
    p.runtime.retain(id);s.saved_inputs=s.inputs;s.saved_positions=s.input_positions;
    s.saved_prompt=s.prompt;s.saved_prompt.release_media_payload();s.saved_anchor=s.anchor;
    s.saved_frontier=p.runtime.frontier(id);copy(s.gpu.saved_logits,s.gpu.last_logits,p.device.stream);
    copy(s.gpu.saved_mtp,s.gpu.mtp_logits,p.device.stream);p.runtime.synchronize();
    p.runtime.release_reservation(id);s.retained=true;s.active=false;s.output=nullptr;s.use_tick=++p.tick;
}
bool EngineProgram::revert_cancelled_prefill_lane(std::uint32_t id) {
    auto& p=*impl_;auto& s=p.lane(id);
    if(!s.retained||!p.runtime.has_retained(id)) return false;
    p.runtime.synchronize();
    if(s.prefill_pending) runtime::rollback_sampling_counts(s.sampling,{&s.anchor,1});
    p.runtime.restore(id);s.inputs=s.saved_inputs;s.input_positions=s.saved_positions;
    s.prompt=s.saved_prompt;s.anchor=s.saved_anchor;
    copy(s.gpu.last_logits,s.gpu.saved_logits,p.device.stream);copy(s.gpu.mtp_logits,s.gpu.saved_mtp,p.device.stream);
    if(p.config.mtp) {
        const int slot=id;const auto pos=s.saved_positions.back();
        auto logits=p.runtime.reseed_mtp({&slot,1},{&s.anchor,1},{&pos,1});
        copy(s.gpu.mtp_logits,Tensor(logits.data,DType::BF16,{physical}),p.device.stream);
    }
    p.runtime.synchronize();p.runtime.release_reservation(id);s.active=false;s.prefilling=false;
    s.prefill_pending=false;s.pending_count=0;s.output=nullptr;return true;
}
void EngineProgram::abort_lane(std::uint32_t id) noexcept {
    try {
        auto& p=*impl_;auto& s=p.lane(id);p.runtime.synchronize();
        if(p.pending_batch) {
            p.runtime.discard();
            for(int row=0;row<p.pending_batch;++row) p.lane(p.pending_slots[row]).pending_count=0;
            p.pending_batch=0;p.pending_output={};
        }
        p.runtime.reset(id);p.runtime.evict_retained(id);p.runtime.evict_retained(id,NativeCheckpoint::Prompt);
        p.runtime.release_reservation(id);s.active=false;s.retained=false;s.prefilling=false;
        s.prefill_pending=false;s.pending_count=0;s.output=nullptr;s.saved_frontier=0;
    }catch(...) {}
}
bool EngineProgram::has_retained_lane(std::uint32_t id) const noexcept {return impl_->lanes[id].retained;}
std::uint64_t EngineProgram::retained_use_tick(std::uint32_t id) const noexcept {return impl_->lanes[id].use_tick;}
void EngineProgram::evict_retained_lane(std::uint32_t id) noexcept {
    try {
        auto& s=impl_->lane(id);impl_->runtime.evict_retained(id);impl_->runtime.evict_retained(id,NativeCheckpoint::Prompt);
        s.retained=false;s.saved_frontier=0;s.checkpoint_frontier=0;
        s.saved_inputs.clear();s.saved_positions.clear();s.saved_prompt={};
    }catch(...) {}
}

void EngineProgram::set_suppressed_tokens_lane(std::uint32_t id,std::span<const TokenId> tokens) {
    auto& c=impl_->lane(id).sampling;
    if(tokens.size()>ops::SamplingConfig::kMaximumSuppressedTokens) throw std::invalid_argument("too many suppressed tokens");
    for(auto token:tokens) if(token<0||token>=vocabulary) throw std::invalid_argument("suppressed token domain");
    c.suppressed_token_count=tokens.size();std::copy(tokens.begin(),tokens.end(),c.suppressed_tokens);
}
void EngineProgram::clear_suppressed_tokens_lane(std::uint32_t id) {impl_->lane(id).sampling.suppressed_token_count=0;}
void EngineProgram::set_typical_cycle_reasoning_lane(std::uint32_t id,bool enabled) {impl_->lane(id).reasoning=enabled;}
void EngineProgram::resolve_prefill_lane(std::uint32_t id,bool terminal) {
    auto& s=impl_->lane(id);if(!s.prefill_pending) throw std::logic_error("Qwen4 prefill has no publication");
    s.generated.push_back(s.anchor);s.prefill_pending=false;s.pending_count=0;
    if(terminal) retain_lane(id);
}
void EngineProgram::synchronize_all() {impl_->runtime.synchronize();}
GenerationTimings EngineProgram::generation_timings_lane(std::uint32_t id) const noexcept {return impl_->lanes[id].timings;}
SpeculativeStats EngineProgram::speculative_stats_lane(std::uint32_t id) const noexcept {return impl_->lanes[id].stats;}
std::uint32_t EngineProgram::captured_context_checkpoint_tokens_lane(std::uint32_t id) const noexcept {return impl_->lanes[id].captured_checkpoint;}
std::uint32_t EngineProgram::restored_context_checkpoint_tokens_lane(std::uint32_t id) const noexcept {return impl_->lanes[id].restored_checkpoint;}
MemorySummary EngineProgram::memory_summary() const noexcept {
    auto result=impl_->runtime.memory_summary();
    result.sequence.capacity_bytes+=impl_->storage.capacity();result.sequence.used_bytes+=impl_->storage.used();
    result.sequence.peak_used_bytes+=impl_->storage.peak_used();
    result.workspace.capacity_bytes+=impl_->workspace.capacity();result.workspace.peak_used_bytes+=impl_->workspace.peak_used();
    return result;
}
void EngineProgram::reset_memory_peaks() noexcept {impl_->runtime.reset_memory_peaks();impl_->storage.reset_peak();impl_->workspace.reset_peak();}

ScoreResult EngineProgram::score(text::qwen::PreparedPrompt&& prompt,RequestPlan&&,
    runtime::TransientRegion,ScoreOptions options) {
    auto& p=*impl_;
    if(p.pending_batch||std::any_of(p.lanes.begin(),p.lanes.begin()+p.config.requests,
        [](const auto& lane){return lane.active;})) throw std::logic_error("Qwen4 score needs an idle engine");
    auto data=text::qwen::PreparedPromptAccess::take(std::move(prompt));
    if(data.token_ids.size()<2||data.token_ids.size()>std::size_t(p.config.context_tokens))
        throw std::invalid_argument("Qwen4 scoring prompt length");
    for(auto token:data.token_ids) if(token<0||token>=vocabulary) throw std::invalid_argument("Qwen4 scoring token domain");
    // Scoring bypasses the generation scheduler, but shares its exclusive page pool.
    // Plan the minimum LRU retained eviction set before changing any live ownership.
    std::vector<int> candidates,victims;
    for(int slot=1;slot<p.config.requests;++slot) if(p.lane(slot).retained) candidates.push_back(slot);
    std::sort(candidates.begin(),candidates.end(),[&](int a,int b){return p.lane(a).use_tick<p.lane(b).use_tick;});
    for(int slot:candidates) {
        if(p.runtime.can_reserve(0,data.token_ids.size(),victims)) break;
        victims.push_back(slot);
    }
    if(!p.runtime.can_reserve(0,data.token_ids.size(),victims))
        throw std::runtime_error("Qwen4 scoring page capacity");
    for(int slot:victims) evict_retained_lane(slot);
    evict_retained_lane(0);p.runtime.reset(0);
    if(!p.runtime.reserve(0,data.token_ids.size())) throw std::runtime_error("Qwen4 scoring page capacity");
    try {
        if(data.has_media()) {
            std::vector<VisionGrid> grids;std::vector<int> columns;
            for(const auto& item:data.vision_items) {
                grids.push_back({item.grid.temporal,item.grid.height,item.grid.width});
                for(const auto& span:item.token_spans) for(std::size_t j=0;j<span.count;++j) columns.push_back(span.begin+j);
            }
            p.runtime.prepare_vision(0,data.patches,grids,columns);
        } else p.runtime.clear_vision(0);
        ScoreResult result;result.schedule=options.schedule;result.prompt_tokens=data.token_ids.size();
        result.skip_tokens=resolve_score_skip(result.prompt_tokens,options.skip_tokens);
        if(options.schedule==ScoreSchedule::Decode)
            result.skip_tokens=resolve_decode_prefix(result.prompt_tokens,result.skip_tokens);
        else if(options.schedule!=ScoreSchedule::Prefill) throw std::invalid_argument("invalid score schedule");
        const auto start=Clock::now();
        std::vector<float> values(p.config.prefill_width);
        std::vector<std::array<int,3>> positions(p.config.prefill_width);
        for(int begin=0;begin<int(data.token_ids.size())-1;) {
            const int chunk=options.schedule==ScoreSchedule::Decode?
                (begin<int(result.skip_tokens)?std::min(p.config.prefill_width,int(result.skip_tokens)-begin):1):p.config.prefill_width;
            const int count=std::min(chunk,int(data.token_ids.size())-1-begin);
            for(int j=0;j<count;++j) positions[j]=position(data,begin+j);
            NativeInputRow row{0,{data.token_ids.data()+begin,std::size_t(count)},
                               {positions.data(),std::size_t(count)}};
            p.runtime.prepare({&row,1},count);auto out=p.runtime.enqueue(false);
            NativeCommitRow commit{0,count};p.runtime.commit({&commit,1});
            p.upload(p.batch.score_targets,data.token_ids.data()+begin+1,count*4);
            Tensor logits(out.logits.data,DType::BF16,{physical,count}),
                targets(p.batch.score_targets.data,DType::I32,{count}),nll(p.batch.nll.data,DType::FP32,{count});
            ops::nll_from_logits(logits,targets,nll,vocabulary,p.device.stream);p.download(values.data(),nll,count*4);
            for(int j=0;j<count;++j) if(begin+j>=int(result.skip_tokens)) {
                record_score_nll(result,values[j]);
            }
            begin+=count;
        }
        if(result.tokens_scored) {
            result.mean_nll=result.sum_nll/result.tokens_scored;result.perplexity=std::exp(result.mean_nll);
        }
        result.score_seconds=seconds(start);p.runtime.reset(0);p.runtime.release_reservation(0);return result;
    } catch(...) {p.runtime.synchronize();p.runtime.reset(0);p.runtime.release_reservation(0);throw;}
}

} // namespace ninfer::targets::qwen4
