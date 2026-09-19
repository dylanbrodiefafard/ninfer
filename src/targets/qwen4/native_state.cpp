#include "targets/qwen4/native_state.h"

#include "core/device.h"
#include "ninfer/ops/gdn_replay.h"
#include "ninfer/ops/ple.h"
#include "ninfer/ops/ngram_embedding.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace ninfer::targets::qwen4 {
namespace {
constexpr int F=10240,Cnv=3,Hv=48;
struct StateLayout {
    PagedKVPoolLayout pages;
    std::array<LinearAttentionStatePoolLayout,4> gdn;
    std::array<TensorRegion,4> ple,carry,history;
    GdnReplayRecordLayout replay;
    TensorRegion ple_replay;
    std::size_t bytes=0;
};
StateLayout plan(NativeState::Config c) {
    if(c.requests<1 || c.requests>4 || c.context<1 || c.context>262144 ||
       c.kv_tokens<c.context || c.kv_tokens>std::int64_t(c.requests)*pages_for_tokens(c.context)*64 ||
       pages_for_tokens(c.kv_tokens)<std::uint32_t(c.requests) || c.record_width<2 || c.record_width>16 ||
       (c.kv!=ops::QsaKvFormat::BF16 && c.kv!=ops::QsaKvFormat::NVFP4G16))
        throw std::invalid_argument("Qwen4 native state capacities");
    LayoutBuilder b; StateLayout p;
    PagedKVPoolSpec kv{pages_for_tokens(c.kv_tokens),pages_for_tokens(c.context),c.requests,
                      PagedKVPlaneOrder::PageMajor,{}};
    for(int l=0;l<12;++l) {
        const bool packed=c.kv==ops::QsaKvFormat::NVFP4G16;
        for(int side=0;side<2;++side) kv.planes.push_back({packed?DType::U8:DType::BF16,packed?128:256,2});
        if(packed) for(int side=0;side<2;++side) kv.planes.push_back({DType::FP8_E4M3FN,16,2});
        kv.planes.push_back({DType::BF16,128,1});
        kv.planes.push_back({DType::I32,3,1});
    }
    p.pages=plan_paged_kv_pool(b,kv);
    const LinearAttentionStatePoolSpec recurrent{36,F,Cnv,Hv,128,128,c.requests,DType::BF16};
    for(int image=0;image<4;++image) {
        p.gdn[image]=plan_linear_attention_state_pool(b,recurrent);
        p.ple[image]=b.add_tensor(DType::BF16,{F,9,c.requests},256,"PLE state");
        p.carry[image]=b.add_tensor(DType::BF16,{F,c.requests},256,"GR continuation");
        p.history[image]=b.add_tensor(DType::I32,{2,c.requests},256,"raw token history");
    }
    p.replay=plan_gdn_replay_records(b,{36,c.requests,c.record_width,F,48,48,128,128});
    p.ple_replay=b.add_tensor(DType::BF16,{F,c.record_width,c.requests},256,"PLE records");
    p.bytes=b.finish(256);return p;
}
void copy(const Tensor& to,const Tensor& from,cudaStream_t stream) {
    if(to.bytes()!=from.bytes()) throw std::logic_error("Qwen4 state transfer extent");
    CUDA_CHECK(cudaMemcpyAsync(to.data,from.data,to.bytes(),cudaMemcpyDeviceToDevice,stream));
}
}

struct NativeState::Impl {
    Config config;cudaStream_t stream;StateLayout layout;DeviceBuffer backing;
    PagedKVPool pool;
    std::array<LinearAttentionStatePool,4> recurrent;
    std::array<Tensor,4> ple,carry,history;
    std::array<PagedKVAllocation,4> allocation;
    std::array<int,4> frontier{};
    std::array<std::array<int,4>,2> retained_frontier{};
    std::array<std::array<bool,4>,2> retained{};
    std::array<bool,4> active{};
    PinnedHostBuffer commit_counts{16};
    Impl(Config c,cudaStream_t s):config(c),stream(s),layout(plan(c)),backing(layout.bytes),
        pool({backing.p,backing.bytes},layout.pages) {
        const DeviceSpan bytes{backing.p,backing.bytes};
        for(int i=0;i<4;++i) {
            recurrent[i]=LinearAttentionStatePool(bytes,layout.gdn[i]);
            ple[i]=layout.ple[i].bind(bytes);carry[i]=layout.carry[i].bind(bytes);history[i]=layout.history[i].bind(bytes);
        }
        for(int i=0;i<config.requests;++i) clear(i);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    void check(int slot) const {
        if(slot<0 || slot>=config.requests) throw std::invalid_argument("Qwen4 native state slot");
    }
    void clear(int slot) {
        check(slot);
        for(int i=0;i<4;++i) {
            recurrent[i].zero_slot(slot,stream);
            auto p=ple[i].slice(2,slot,1),h=carry[i].slice(1,slot,1);
            CUDA_CHECK(cudaMemsetAsync(p.data,0,p.bytes(),stream));
            CUDA_CHECK(cudaMemsetAsync(h.data,0,h.bytes(),stream));
            const std::array<int,2> eos{248044,248044};
            // Synchronous stack-source upload at an explicit reset boundary.
            CUDA_CHECK(cudaMemcpy(history[i].slice(1,slot,1).data,eos.data(),8,cudaMemcpyHostToDevice));
        }
        frontier[slot]=0;active[slot]=false;
        for(int k=0;k<2;++k) {retained[k][slot]=false;retained_frontier[k][slot]=0;}
    }
    void image(int from,int to,int slot) {
        for(int l=0;l<36;++l) {
            copy(recurrent[to].conv_slot(l,slot),recurrent[from].conv_slot(l,slot),stream);
            copy(recurrent[to].recurrent_slot(l,slot),recurrent[from].recurrent_slot(l,slot),stream);
        }
        copy(ple[to].slice(2,slot,1),ple[from].slice(2,slot,1),stream);
        copy(carry[to].slice(1,slot,1),carry[from].slice(1,slot,1),stream);
        copy(history[to].slice(1,slot,1),history[from].slice(1,slot,1),stream);
    }
};

NativeState::NativeState(Config c,cudaStream_t s):impl_(std::make_unique<Impl>(c,s)) {}
NativeState::~NativeState() { if(impl_) (void)cudaStreamSynchronize(impl_->stream); }
std::size_t NativeState::device_bytes(Config c) {return plan(c).bytes;}
int NativeState::frontier(int slot) const {impl_->check(slot);return impl_->frontier[slot];}
bool NativeState::can_reserve(int slot,int maximum,std::span<const int> evictions) const {
    auto& s=*impl_;s.check(slot);
    if(maximum<1 || maximum>s.config.context) return false;
    const auto requested=pages_for_tokens(maximum);
    std::uint64_t used=s.pool.entitled_pages();
    if(s.allocation[slot].valid()) used-=s.allocation[slot].page_entitlement();
    std::array<bool,4> seen{};
    for(int e:evictions) {
        s.check(e);
        if(e==slot || seen[e] || s.active[e] || !s.retained[0][e]) throw std::invalid_argument("Qwen4 eviction set");
        seen[e]=true;
        used-=s.allocation[e].valid()?s.allocation[e].page_entitlement():0;
    }
    return used+requested<=s.pool.page_group_count();
}
void NativeState::reserve(int slot,int maximum) {
    auto& s=*impl_;s.check(slot);
    if(maximum<s.frontier[slot]) throw std::invalid_argument("Qwen4 reservation precedes live frontier");
    if(!can_reserve(slot,maximum)) throw std::runtime_error("Qwen4 shared KV reservation exhausted");
    const auto pages=pages_for_tokens(maximum);
    if(s.allocation[slot].valid()) s.allocation[slot].set_page_entitlement(pages);
    else s.allocation[slot]=s.pool.reserve(pages);
    if(s.allocation[slot].bound_row()!=slot) s.allocation[slot].bind_row(slot,s.stream);
    s.active[slot]=true;
}
void NativeState::release_reservation(int slot) {
    auto& s=*impl_;s.check(slot);
    if(s.allocation[slot].valid()) {
        s.allocation[slot].trim_tokens(s.frontier[slot]);
        s.allocation[slot].cancel_unmapped_entitlement();
        s.allocation[slot].unbind_row();
    }
    s.active[slot]=false;
}
void NativeState::materialize(int slot,int maximum) {
    auto& s=*impl_;s.check(slot);
    if(maximum<s.frontier[slot] || maximum>s.config.context || !s.allocation[slot].valid())
        throw std::invalid_argument("Qwen4 KV materialization domain");
    s.allocation[slot].materialize_tokens(maximum,s.stream);
}
void NativeState::discard_materialization(std::span<const int> slots) {
    auto& s=*impl_;std::array<bool,4> seen{};
    if(slots.empty() || slots.size()>std::size_t(s.config.requests))
        throw std::invalid_argument("Qwen4 discard materialization rows");
    for(int slot:slots) {s.check(slot);if(seen[slot])throw std::invalid_argument("Qwen4 duplicate discard slot");seen[slot]=true;}
    CUDA_CHECK(cudaStreamSynchronize(s.stream));
    for(int slot:slots)if(s.allocation[slot].valid())s.allocation[slot].trim_tokens(s.frontier[slot]);
}
void NativeState::reset(int slot) {
    auto& s=*impl_;s.check(slot);CUDA_CHECK(cudaStreamSynchronize(s.stream));
    s.allocation[slot].release();s.clear(slot);
}
LinearAttentionStatePool& NativeState::gdn(bool provisional) {return impl_->recurrent[provisional?1:0];}
Tensor NativeState::ple(bool provisional) const {return impl_->ple[provisional?1:0];}
Tensor NativeState::continuation() const {return impl_->carry[0];}
Tensor NativeState::token_history() const {return impl_->history[0];}
ops::QsaPagedStateView NativeState::qsa(int layer) const {
    auto& s=*impl_;if(layer<0 || layer>=12) throw std::invalid_argument("Qwen4 QSA layer");
    const bool packed=s.config.kv==ops::QsaKvFormat::NVFP4G16;int i=layer*(packed?6:4);
    ops::QsaPagedStateView view;view.format=s.config.kv;
    view.k=s.pool.plane(i++);view.v=s.pool.plane(i++);
    if(packed) {view.k_scales=s.pool.plane(i++);view.v_scales=s.pool.plane(i++);}
    view.raw_index_keys=s.pool.plane(i++);view.positions=s.pool.plane(i++);
    view.block_tables=s.pool.block_tables();return view;
}
GdnReplayRecords NativeState::records(int width,int batch) const {
    auto& s=*impl_;
    if(width<2 || width>s.config.record_width || batch<1 || batch>s.config.requests)
        throw std::invalid_argument("Qwen4 record geometry");
    const DeviceSpan storage{s.backing.p,s.backing.bytes};
    GdnReplayRecords result;
    result.spec={36,batch,width,F,48,48,128,128};
    result.conv=Tensor(s.layout.replay.conv.bind(storage).data,DType::BF16,{F,width,batch*36});
    result.key=Tensor(s.layout.replay.key.bind(storage).data,DType::BF16,{128,48,width,batch*36});
    result.value=Tensor(s.layout.replay.value.bind(storage).data,DType::BF16,{128,48,width,batch*36});
    result.gate=Tensor(s.layout.replay.gate.bind(storage).data,DType::FP32,{2,48,width,batch*36});
    return result;
}
Tensor NativeState::ple_records(int width,int batch) const {
    auto& s=*impl_;
    if(width<1 || width>s.config.record_width || batch<1 || batch>s.config.requests)
        throw std::invalid_argument("Qwen4 PLE record geometry");
    return Tensor(s.layout.ple_replay.bind({s.backing.p,s.backing.bytes}).data,DType::BF16,{F,width,batch});
}

void NativeState::commit(std::span<const int> slots,std::span<const int> lengths,
    std::span<const int> counts,int width,bool recorded,const Tensor& raw_ids,const Tensor& residual,
    const Tensor& device_slots,Tensor& device_counts) {
    auto& s=*impl_;const int batch=int(slots.size());
    if(batch<1 || batch>s.config.requests || lengths.size()!=slots.size() || counts.size()!=slots.size() || width<1 ||
       raw_ids.dtype!=DType::I32 || !raw_ids.is_contiguous() || raw_ids.ne[0]!=width || raw_ids.ne[1]!=batch ||
       raw_ids.ne[2]!=1 || raw_ids.ne[3]!=1 || residual.dtype!=DType::BF16 || !residual.is_contiguous() ||
       residual.ne[0]!=F || residual.ne[1]!=width || residual.ne[2]!=batch || residual.ne[3]!=1 ||
       device_slots.dtype!=DType::I32 || device_slots.ne[0]!=batch || !device_slots.is_contiguous() ||
       device_counts.dtype!=DType::I32 || device_counts.ne[0]!=batch || !device_counts.is_contiguous())
        throw std::invalid_argument("Qwen4 state commit geometry");
    std::array<bool,4> seen{};bool partial=false;
    for(int b=0;b<batch;++b) {
        const int slot=slots[b];s.check(slot);
        if(seen[slot] || lengths[b]<1 || lengths[b]>width || counts[b]<0 || counts[b]>lengths[b] ||
           s.frontier[slot]+lengths[b]>s.config.context || !s.allocation[slot].valid() ||
           s.allocation[slot].mapped_token_capacity()<std::uint32_t(s.frontier[slot]+lengths[b]))
            throw std::invalid_argument("Qwen4 state commit prefix");
        seen[slot]=true;partial|=counts[b]>0 && counts[b]<lengths[b];
    }
    if((partial && !recorded) || (recorded && (width<2 || width>s.config.record_width)))
        throw std::invalid_argument("Qwen4 partial commit requires recorded verification");
    CUDA_CHECK(cudaStreamSynchronize(s.stream));
    std::memcpy(s.commit_counts.data(),counts.data(),counts.size_bytes());
    CUDA_CHECK(cudaMemcpyAsync(device_counts.data,s.commit_counts.data(),counts.size_bytes(),cudaMemcpyHostToDevice,s.stream));
    if(recorded) {
        std::array<ops::GdnReplayFoldRow,4> rows{};
        for(int b=0;b<batch;++b) rows[b]={slots[b],counts[b]};
        ops::gdn_replay_fold(records(width,batch),s.recurrent[0].all_layers_view(),
            std::span(rows).first(batch),s.stream);
        auto record=ple_records(width,batch);
        ops::ple_commit_prefix(record,raw_ids,device_counts,device_slots,s.ple[0],s.history[0],s.stream);
    } else {
        for(int b=0;b<batch;++b) if(counts[b]) {
            const int slot=slots[b];
            for(int l=0;l<36;++l) {
                copy(s.recurrent[0].conv_slot(l,slot),s.recurrent[1].conv_slot(l,slot),s.stream);
                copy(s.recurrent[0].recurrent_slot(l,slot),s.recurrent[1].recurrent_slot(l,slot),s.stream);
            }
            copy(s.ple[0].slice(2,slot,1),s.ple[1].slice(2,slot,1),s.stream);
        }
        ops::ngram_history_commit(raw_ids,device_counts,device_slots,s.history[0],s.stream);
    }
    for(int b=0;b<batch;++b) if(counts[b]) {
        const auto* source=static_cast<const std::uint16_t*>(residual.data)+(std::size_t(b)*width+counts[b]-1)*F;
        CUDA_CHECK(cudaMemcpyAsync(s.carry[0].slice(1,slots[b],1).data,source,F*2,cudaMemcpyDeviceToDevice,s.stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(s.stream));
    for(int b=0;b<batch;++b) {
        s.frontier[slots[b]]+=counts[b];s.allocation[slots[b]].trim_tokens(s.frontier[slots[b]]);
    }
}
void NativeState::retain(int slot,NativeCheckpoint kind) {
    const int k=int(kind);if(k<0 || k>1) throw std::invalid_argument("Qwen4 checkpoint kind");
    auto& s=*impl_;s.check(slot);s.image(0,2+k,slot);CUDA_CHECK(cudaStreamSynchronize(s.stream));
    s.retained_frontier[k][slot]=s.frontier[slot];s.retained[k][slot]=true;
}
void NativeState::restore(int slot,NativeCheckpoint kind) {
    const int k=int(kind);if(k<0 || k>1) throw std::invalid_argument("Qwen4 checkpoint kind");
    auto& s=*impl_;s.check(slot);
    if(!s.retained[k][slot]) throw std::logic_error("Qwen4 missing retained state");
    s.image(2+k,0,slot);CUDA_CHECK(cudaStreamSynchronize(s.stream));
    s.frontier[slot]=s.retained_frontier[k][slot];
    if(s.allocation[slot].valid()) s.allocation[slot].trim_tokens(s.frontier[slot]);
    for(int other=0;other<2;++other) if(s.retained_frontier[other][slot]>s.frontier[slot]) s.retained[other][slot]=false;
}
void NativeState::evict_retained(int slot,NativeCheckpoint kind) {
    const int k=int(kind);if(k<0 || k>1) throw std::invalid_argument("Qwen4 checkpoint kind");
    auto& s=*impl_;s.check(slot);s.retained[k][slot]=false;
    if(!s.active[slot] && k==0) {
        CUDA_CHECK(cudaStreamSynchronize(s.stream));s.allocation[slot].release();s.clear(slot);
    }
}
bool NativeState::has_retained(int slot,NativeCheckpoint kind) const {
    const int k=int(kind);if(k<0 || k>1) throw std::invalid_argument("Qwen4 checkpoint kind");
    impl_->check(slot);return impl_->retained[k][slot];
}
std::uint32_t NativeState::entitled_pages() const {return impl_->pool.entitled_pages();}
} // namespace ninfer::targets::qwen4
