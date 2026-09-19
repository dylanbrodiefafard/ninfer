#include "targets/qwen4/transaction.h"
#include "core/layout.h"
#include "ninfer/ops/gdn_replay.h"
#include "ninfer/ops/ple.h"
#include "runtime/contract/sampling.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace ninfer::targets::qwen4 {
namespace {
constexpr int channels=10240,vocabulary=248320,reset_token=248044;
struct Snapshot {
    NativeFirstBlockState state;
    Tensor history,carry,counts;
};
template<class Allocator> Snapshot allocate_snapshot(Allocator& a,int capacity) {
    Snapshot s;
    for(auto& g:s.state.gdn) {
        g.conv=a.alloc(DType::BF16,{channels,3});
        g.recurrence=a.alloc(DType::FP32,{128,128,48});
    }
    s.state.ple_conv=a.alloc(DType::BF16,{channels,9});
    s.state.qsa={ops::QsaKvFormat::BF16,a.alloc(DType::BF16,{256,capacity,2}),
        a.alloc(DType::BF16,{256,capacity,2}),{},{},a.alloc(DType::BF16,{128,capacity}),
        a.alloc(DType::I32,{3,capacity})};
    s.history=a.alloc(DType::I32,{2});
    s.carry=a.alloc(DType::BF16,{2560,4});
    s.counts=a.alloc(DType::I32,{vocabulary});
    return s;
}
std::size_t snapshot_bytes(int capacity) {
    WorkspaceLayoutBuilder builder;
    (void)allocate_snapshot(builder,capacity);
    return builder.peak_bytes();
}
void copy(const Tensor& to,const Tensor& from,cudaStream_t stream) {
    if(to.bytes()!=from.bytes()) throw std::logic_error("native transaction copy geometry");
    CUDA_CHECK(cudaMemcpyAsync(to.data,from.data,to.bytes(),cudaMemcpyDeviceToDevice,stream));
}
void copy_state(const NativeFirstBlockState& to,const NativeFirstBlockState& from,
                int frontier,cudaStream_t stream) {
    for(int layer=0;layer<3;++layer) {
        copy(to.gdn[layer].conv,from.gdn[layer].conv,stream);
        copy(to.gdn[layer].recurrence,from.gdn[layer].recurrence,stream);
    }
    copy(to.ple_conv,from.ple_conv,stream);
    // Growing cache ownership is only [0,frontier); provisional suffixes are invisible.
    if(!frontier) return;
    for(auto [a,b]:std::array<std::pair<Tensor,Tensor>,2>{{{to.qsa.k,from.qsa.k},{to.qsa.v,from.qsa.v}}}) {
        CUDA_CHECK(cudaMemcpy2DAsync(a.data,a.ne[0]*a.ne[1]*2,b.data,b.ne[0]*b.ne[1]*2,
                                    a.ne[0]*frontier*2,2,cudaMemcpyDeviceToDevice,stream));
    }
    CUDA_CHECK(cudaMemcpyAsync(to.qsa.raw_index_keys.data,from.qsa.raw_index_keys.data,
        std::size_t(frontier)*128*2,cudaMemcpyDeviceToDevice,stream));
    CUDA_CHECK(cudaMemcpyAsync(to.qsa.positions.data,from.qsa.positions.data,
        std::size_t(frontier)*3*4,cudaMemcpyDeviceToDevice,stream));
}
void require_token(TokenId token) {
    if(token<0 || token>=vocabulary) throw std::invalid_argument("native transaction token domain");
}
}

struct NativeTransaction::Slot {
    explicit Slot(int capacity):snapshot_storage(snapshot_bytes(capacity)),
        saved(allocate_snapshot(snapshot_storage,capacity)),controls(256),history(8),carry(channels*2),
        counts(vocabulary*4),host(256),published(capacity),saved_published(capacity) {}
    DeviceArena snapshot_storage;
    Snapshot saved;
    DeviceBuffer controls,history,carry,counts;
    PinnedHostBuffer host;
    std::vector<TokenId> published,saved_published;
    std::array<TokenId,16> verified{};
    Tensor output_hidden;
    TokenId anchor=0,saved_anchor=0;
    std::array<int,3> position{},saved_position{};
    int pending=0,published_count=0,saved_count=0,saved_frontier=0;
    bool retained=false,poisoned=false,ready=false;
};

NativeTransaction::NativeTransaction(const LoadedNativeFirstBlock& model,int requests,int capacity,
                                     int max_width,cudaStream_t stream)
    :program_(model,requests,capacity,max_width),requests_(requests),capacity_(capacity),
     max_width_(max_width),stream_(stream) {
    if(max_width<1 || max_width>16) throw std::invalid_argument("native transaction width must be1..16");
    for(int i=0;i<requests_;++i) {slots_[i]=std::make_unique<Slot>(capacity);reset(i,0,{0,0,0});}
}
NativeTransaction::~NativeTransaction() { (void)cudaStreamSynchronize(stream_); }
NativeTransaction::Slot& NativeTransaction::slot(int index) const {
    if(index<0 || index>=requests_) throw std::out_of_range("native transaction slot");
    return *slots_[index];
}
void NativeTransaction::reset(int index,TokenId next,std::array<int,3> position) {
    require_token(next);
    for(int value:position) if(value<0) throw std::invalid_argument("native transaction negative position");
    auto& s=slot(index);
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    program_.reset(index,stream_);
    auto* host=static_cast<int*>(s.host.data());
    host[0]=reset_token;host[1]=reset_token;
    CUDA_CHECK(cudaMemcpyAsync(s.history.p,host,8,cudaMemcpyHostToDevice,stream_));
    CUDA_CHECK(cudaMemsetAsync(s.controls.p,0,s.controls.bytes,stream_));
    CUDA_CHECK(cudaMemsetAsync(s.carry.p,0,s.carry.bytes,stream_));
    CUDA_CHECK(cudaMemsetAsync(s.counts.p,0,s.counts.bytes,stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    s.anchor=next;s.position=position;s.pending=0;s.published_count=0;s.retained=false;s.poisoned=false;
    s.ready=false;s.output_hidden={};
}
void NativeTransaction::prepare(int index,std::span<const TokenId> verified,
    std::span<const std::array<int,3>> positions,const Tensor& hidden,const Tensor& embedding) {
    auto& s=slot(index);
    const int width=static_cast<int>(verified.size());
    if(s.poisoned || s.pending || width<1 || width>max_width_ || positions.size()!=verified.size() ||
       verified.front()!=s.anchor) throw std::invalid_argument("native transaction preparation boundary");
    for(int t=0;t<width;++t) {
        require_token(verified[t]);
        for(int axis=0;axis<3;++axis) {
            if(s.position[axis]>std::numeric_limits<int>::max()-width ||
               positions[t][axis]!=s.position[axis]+t)
                throw std::invalid_argument("native transaction text position progression");
        }
    }
    s.pending=width;
    s.ready=false;s.output_hidden={};
    try {
        program_.prepare(index,positions,hidden,embedding,stream_);
        std::copy(verified.begin(),verified.end(),s.verified.begin());
        auto* host=static_cast<int*>(s.host.data());
        std::copy(verified.begin(),verified.end(),host);
        CUDA_CHECK(cudaMemcpyAsync(s.controls.p,host,width*4,cudaMemcpyHostToDevice,stream_));
    } catch(...) {s.poisoned=true;throw;}
}
NativeFirstBlockOutput NativeTransaction::enqueue(int index,const NativeBlockTrace* trace) {
    auto& s=slot(index);
    if(s.poisoned || !s.pending || s.ready) throw std::logic_error("native transaction enqueue boundary");
    cudaStreamCaptureStatus capture;
    CUDA_CHECK(cudaStreamIsCapturing(stream_,&capture));
    if(capture!=cudaStreamCaptureStatusNone)
        throw std::logic_error("native transaction graph execution requires typed capture_graph");
    try {
        auto output=program_.enqueue(index,s.pending,s.pending>1,stream_,trace);
        s.output_hidden=output.hidden;s.ready=true;
        return output;
    } catch(...) {s.poisoned=true;throw;}
}
int NativeTransaction::frontier(int index) const {return program_.frontier(index);}
TokenId NativeTransaction::anchor(int index) const {return slot(index).anchor;}
std::array<int,3> NativeTransaction::anchor_position(int index) const {return slot(index).position;}
std::span<const TokenId> NativeTransaction::published(int index) const {
    const auto& s=slot(index);return {s.published.data(),static_cast<std::size_t>(s.published_count)};
}
NativeFirstBlockState NativeTransaction::state(int index) const {return program_.state(index,false);}
NativeFirstBlockRecords NativeTransaction::records(int index) const {
    const auto& s=slot(index);
    if(!s.ready || s.poisoned) throw std::logic_error("native records require an enqueued transaction");
    return program_.records(index,s.pending);
}
Tensor NativeTransaction::raw_token_history(int index) const {return {slot(index).history.p,DType::I32,{2}};}
Tensor NativeTransaction::continuation(int index) const {return {slot(index).carry.p,DType::BF16,{2560,4}};}
Tensor NativeTransaction::sampling_counts(int index) const {return {slot(index).counts.p,DType::I32,{vocabulary}};}

void NativeTransaction::resolve(std::span<const NativeResolution> rows) {
    if(rows.empty() || rows.size()>static_cast<std::size_t>(requests_))
        throw std::invalid_argument("native transaction resolution rows");
    unsigned mask=0;
    for(const auto& row:rows) {
        const auto& s=slot(row.slot);
        const auto count=row.decision.accepted_tokens;
        if((mask&(1U<<row.slot)) || s.poisoned || !s.pending ||
           count>row.licensed.tokens.size() || row.licensed.tokens.size()>static_cast<std::size_t>(s.pending) ||
           (row.decision.reject_generated_round!=(count==0)) || (count && !s.ready) ||
           s.published_count+count>static_cast<unsigned>(capacity_) ||
           (row.sampling.token_counts && row.sampling.token_counts!=s.counts.p))
            throw std::invalid_argument("native transaction licensed publication boundary");
        mask|=1U<<row.slot;
        for(std::size_t i=0;i<row.licensed.tokens.size();++i) {
            require_token(row.licensed.tokens[i]);
            if(i+1<row.licensed.tokens.size() && row.licensed.tokens[i]!=s.verified[i+1])
                throw std::invalid_argument("native transaction licensed draft prefix differs");
        }
    }
    try {
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        for(const auto& row:rows) {
            auto& s=slot(row.slot);
            const int count=static_cast<int>(row.decision.accepted_tokens),width=s.pending;
            runtime::rollback_sampling_counts(row.sampling,row.licensed.tokens.subspan(count));
            if(!count) continue;
            auto committed=program_.state(row.slot,false);
            Tensor ple_records;
            if(width>1) {
                const auto records=program_.records(row.slot,width);
                const std::array fold{ops::GdnReplayFoldRow{.linear_state_slot=0,.commit_columns=count}};
                for(int layer=0;layer<3;++layer)
                    ops::gdn_replay_fold(records.gdn[layer],committed.gdn[layer].conv,
                        committed.gdn[layer].recurrence,fold,stream_);
                ple_records=records.ple_conv;
            } else {
                const auto provisional=program_.state(row.slot,true);
                for(int layer=0;layer<3;++layer) {
                    copy(committed.gdn[layer].conv,provisional.gdn[layer].conv,stream_);
                    copy(committed.gdn[layer].recurrence,provisional.gdn[layer].recurrence,stream_);
                }
                ple_records=Tensor(static_cast<std::byte*>(provisional.ple_conv.data)+8*channels*2,
                                   DType::BF16,{channels,1});
            }
            auto* host=static_cast<int*>(s.host.data());host[16]=count;host[17]=0;
            auto* controls=static_cast<int*>(s.controls.p);
            CUDA_CHECK(cudaMemcpyAsync(controls+16,host+16,8,cudaMemcpyHostToDevice,stream_));
            Tensor tokens(controls,DType::I32,{width}),counts(controls+16,DType::I32,{1}),
                slots(controls+17,DType::I32,{1}),history=raw_token_history(row.slot);
            ops::ple_commit_prefix(ple_records,tokens,counts,slots,committed.ple_conv,history,stream_);
            // The last layer feature panel is slot-owned and stable until the next prepare.
            Tensor accepted_hidden(static_cast<std::byte*>(s.output_hidden.data)+(count-1)*channels*2,
                                   DType::BF16,{2560,4});
            copy(continuation(row.slot),accepted_hidden,stream_);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        for(const auto& row:rows) {
            auto& s=slot(row.slot);const int count=static_cast<int>(row.decision.accepted_tokens);
            program_.publish_frontier(row.slot,count);s.pending=0;s.ready=false;s.output_hidden={};
            if(!count) continue;
            std::copy_n(row.licensed.tokens.begin(),count,s.published.begin()+s.published_count);
            s.published_count+=count;s.anchor=row.licensed.tokens[count-1];
            for(auto& coordinate:s.position) coordinate+=count;
        }
    } catch(...) {
        for(const auto& row:rows) slot(row.slot).poisoned=true;
        throw;
    }
}

void NativeTransaction::retain(int index) {
    auto& s=slot(index);
    if(s.pending || s.poisoned) throw std::logic_error("native retain requires committed boundary");
    s.retained=false;
    copy_state(s.saved.state,state(index),frontier(index),stream_);
    copy(s.saved.history,raw_token_history(index),stream_);
    copy(s.saved.carry,continuation(index),stream_);
    copy(s.saved.counts,sampling_counts(index),stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    s.saved_frontier=frontier(index);s.saved_anchor=s.anchor;s.saved_position=s.position;
    s.saved_count=s.published_count;
    std::copy_n(s.published.begin(),s.published_count,s.saved_published.begin());s.retained=true;
}
void NativeTransaction::restore(int index) {
    auto& s=slot(index);
    if(!s.retained || (s.pending && !s.poisoned)) throw std::logic_error("native restore requires retained boundary");
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    program_.discard(index);s.pending=0;s.ready=false;s.output_hidden={};
    copy_state(state(index),s.saved.state,s.saved_frontier,stream_);
    copy(raw_token_history(index),s.saved.history,stream_);
    copy(continuation(index),s.saved.carry,stream_);
    copy(sampling_counts(index),s.saved.counts,stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    program_.restore_frontier(index,s.saved_frontier);s.anchor=s.saved_anchor;s.position=s.saved_position;
    s.published_count=s.saved_count;
    std::copy_n(s.saved_published.begin(),s.saved_count,s.published.begin());s.poisoned=false;
}

std::unique_ptr<NativeTransactionGraph> NativeTransaction::capture_graph(std::span<const int> slots) {
    return std::unique_ptr<NativeTransactionGraph>(new NativeTransactionGraph(*this,slots));
}
NativeTransactionGraph::NativeTransactionGraph(NativeTransaction& owner,std::span<const int> slots)
    :owner_(owner),rows_(static_cast<int>(slots.size())) {
    if(rows_<1 || rows_>owner_.requests_) throw std::invalid_argument("native graph captured row count");
    unsigned mask=0;
    for(int row=0;row<rows_;++row) {
        auto& s=owner_.slot(slots[row]);
        if((mask&(1U<<slots[row])) || !s.pending || s.ready || s.poisoned)
            throw std::invalid_argument("native graph capture preparation boundary");
        mask|=1U<<slots[row];slots_[row]=slots[row];widths_[row]=s.pending;
    }
    CUDA_CHECK(cudaStreamSynchronize(owner_.stream_));
    try {
        definition_.capture(owner_.stream_,[&] {
            for(int row=0;row<rows_;++row) {
                const auto output=owner_.program_.enqueue(slots_[row],widths_[row],widths_[row]>1,owner_.stream_);
                outputs_[row]=output.hidden;
            }
        });
        executable_.instantiate(definition_);executable_.upload(owner_.stream_);
    } catch(...) {
        for(int row=0;row<rows_;++row) owner_.slot(slots_[row]).poisoned=true;
        throw;
    }
}
NativeTransactionGraph::~NativeTransactionGraph() { (void)cudaStreamSynchronize(owner_.stream_); }
void NativeTransactionGraph::launch() {
    for(int row=0;row<rows_;++row) {
        const auto& s=owner_.slot(slots_[row]);
        if(s.poisoned || s.ready || s.pending!=widths_[row])
            throw std::logic_error("native graph launch differs from captured pending slots/widths");
    }
    try {
        executable_.launch(owner_.stream_);
        for(int row=0;row<rows_;++row) {
            auto& s=owner_.slot(slots_[row]);s.output_hidden=outputs_[row];s.ready=true;
        }
    } catch(...) {
        for(int row=0;row<rows_;++row) owner_.slot(slots_[row]).poisoned=true;
        throw;
    }
}
} // namespace ninfer::targets::qwen4
