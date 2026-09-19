#pragma once
#include "targets/qwen4/mtp.h"
#include "core/paged_kv_cache.h"
#include "core/decode_graph.h"

namespace ninfer::test::qwen4 {
struct MtpControl {int frontier=0;std::vector<int> ids,positions,visible,offsets;};
inline MtpControl prepare_mtp_control(int frontier,std::span<const std::array<int,3>> positions) {
    if(frontier<0 || positions.empty() || frontier+positions.size()>4096) throw std::invalid_argument("fixture MTP prefix");
    MtpControl c;c.frontier=frontier;c.offsets.push_back(0);
    for(std::size_t t=0;t<positions.size();++t) {
        c.ids.push_back(frontier+int(t));
        for(int p:positions[t]) {if(p<0) throw std::invalid_argument("fixture MTP position");c.positions.push_back(p);}
        for(int i=0;i<=frontier+int(t);++i)c.visible.push_back(i);
        c.offsets.push_back(int(c.visible.size()));
    }
    return c;
}
// C1 bounded source-oracle harness. It owns one real page and adapts existing trace records;
// product request/retention/graph ownership is independently exercised by NativeDraftRuntime.
class MtpTestProgram {
    static constexpr int D=2560,F=10240,S=2051;
    static PagedKVPoolLayout layout() {
        LayoutBuilder builder;
        return plan_paged_kv_pool(builder,{1,1,1,PagedKVPlaneOrder::PageMajor,
            {{DType::BF16,256,2},{DType::BF16,256,2},{DType::BF16,128,1},{DType::I32,3,1}}});
    }
    PagedKVPoolLayout layout_;
    DeviceBuffer backing_;
    PagedKVPool pool_;
    PagedKVAllocation allocation_;
    targets::qwen4::MtpProgram program_;
    DeviceBuffer control_,positions_,carry_,saved_carry_,seed_ids_,seed_count_,draft_embedding_;
    DecodeGraphDefinition definition_;
    DecodeGraphExecutable graph_;
    int capacity_,maximum_,frontier_=0,saved_frontier_=-1;
    ops::QsaPagedStateView paged() const {
        return {ops::QsaKvFormat::BF16,pool_.plane(0),pool_.plane(1),{},{},pool_.plane(2),pool_.plane(3),pool_.block_tables()};
    }
    ops::QsaBatchControls controls(int width) const {
        auto* p=static_cast<int*>(control_.p);
        return {Tensor(p,DType::I32,{1}),Tensor(p+1,DType::I32,{1}),Tensor(p+2,DType::I32,{1}),
            Tensor(positions_.p,DType::I32,{3,width,1})};
    }
    targets::qwen4::MtpOutput compute(const Tensor& e,const Tensor& h,int width,bool frozen,
                                    cudaStream_t stream,const targets::qwen4::MtpTrace* trace) {
        auto ids=selected_ids().view({S,1,1}),count=selected_count().view({1,1});
        return program_.execute(e.view({D,width,1}),h.view({F,width,1}),controls(width),paged(),64,
            frozen?&ids:nullptr,frozen?&count:nullptr,stream,trace);
    }
    targets::qwen4::MtpOutput execute(const Tensor& e,const Tensor& h,const MtpControl& c,
                                    bool frozen,cudaStream_t stream,const targets::qwen4::MtpTrace* trace) {
        const int width=int(c.ids.size());
        if(width<1 || width>maximum_ || c.frontier!=frontier_ || frontier_+width>capacity_ ||
           c.positions.size()!=std::size_t(width)*3 || (frozen && (width!=1 || frontier_==0)) ||
           e.dtype!=DType::BF16 || h.dtype!=DType::BF16 || e.ne[0]!=D || h.ne[0]!=F ||
           e.ne[1]!=width || h.ne[1]!=width || !e.is_contiguous() || !h.is_contiguous() ||
           (reinterpret_cast<std::uintptr_t>(e.data)&15U) || (reinterpret_cast<std::uintptr_t>(h.data)&15U))
            throw std::invalid_argument("bounded MTP fixture controls");
        for(int t=0;t<width;++t) if(c.ids[t]!=frontier_+t) throw std::invalid_argument("bounded MTP fixture IDs");
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const std::array<int,3> words{0,width,frontier_};control_.copy_from_host(words.data(),12);
        positions_.copy_from_host(c.positions.data(),c.positions.size()*4);
        if(frozen && saved_frontier_<0) {
            CUDA_CHECK(cudaMemcpyAsync(saved_carry_.p,carry_.p,F*2,cudaMemcpyDeviceToDevice,stream));
            saved_frontier_=frontier_;
        }
        targets::qwen4::MtpOutput output;
        if(frozen && !trace) {
            CUDA_CHECK(cudaMemcpyAsync(draft_embedding_.p,e.data,D*2,cudaMemcpyDeviceToDevice,stream));
            graph_.launch(stream);output=program_.output(1,1);
        } else output=compute(e,h,width,frozen,stream,trace);
        if(!frozen) {
            auto ids=program_.selected_ids(width,1),count=program_.selected_count(width,1);
            CUDA_CHECK(cudaMemcpyAsync(seed_ids_.p,static_cast<int*>(ids.data)+(width-1)*S,S*4,cudaMemcpyDeviceToDevice,stream));
            CUDA_CHECK(cudaMemcpyAsync(seed_count_.p,static_cast<int*>(count.data)+width-1,4,cudaMemcpyDeviceToDevice,stream));
        }
        CUDA_CHECK(cudaMemcpyAsync(carry_.p,static_cast<std::byte*>(output.carried_hidden.data)+(width-1)*F*2,
                                  F*2,cudaMemcpyDeviceToDevice,stream));
        frontier_+=width;return output;
    }
public:
    MtpTestProgram(const targets::qwen4::MtpWeights& weights,int capacity,int maximum,cudaStream_t stream)
        :layout_(layout()),backing_(layout_.block_tables.region.offset+layout_.block_tables.region.bytes),
         pool_({backing_.p,backing_.bytes},layout_),program_(weights,maximum,1),control_(12),positions_(maximum*12),
         carry_(F*2),saved_carry_(F*2),seed_ids_(S*4),seed_count_(4),draft_embedding_(D*2),
         capacity_(capacity),maximum_(maximum) {
        if(capacity<1 || capacity>64 || maximum<1 || maximum>capacity) throw std::invalid_argument("bounded MTP fixture capacity");
        allocation_=pool_.reserve(1);allocation_.bind_row(0,stream);allocation_.materialize_tokens(capacity,stream);
        CUDA_CHECK(cudaMemsetAsync(control_.p,0,12,stream));CUDA_CHECK(cudaStreamSynchronize(stream));
        definition_.capture(stream,[&]{(void)compute(Tensor(draft_embedding_.p,DType::BF16,{D,1}),
            Tensor(carry_.p,DType::BF16,{F,1}),1,true,stream,nullptr);});
        graph_.instantiate(definition_);graph_.upload(stream);CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    targets::qwen4::MtpOutput extend(const Tensor& e,const Tensor& h,const MtpControl& c,
        cudaStream_t stream,const targets::qwen4::MtpTrace* trace=nullptr) {return execute(e,h,c,false,stream,trace);}
    targets::qwen4::MtpOutput draft(const Tensor& e,const MtpControl& c,
        cudaStream_t stream,const targets::qwen4::MtpTrace* trace=nullptr) {
        return execute(e,Tensor(carry_.p,DType::BF16,{F,1}),c,true,stream,trace);
    }
    void discard_drafts(cudaStream_t stream) {
        if(saved_frontier_<0) return;
        CUDA_CHECK(cudaMemcpyAsync(carry_.p,saved_carry_.p,F*2,cudaMemcpyDeviceToDevice,stream));
        frontier_=saved_frontier_;saved_frontier_=-1;
    }
    void reset(cudaStream_t) {frontier_=0;saved_frontier_=-1;}
    int frontier() const {return frontier_;}
    Tensor selected_ids() const {return Tensor(seed_ids_.p,DType::I32,{S,1});}
    Tensor selected_count() const {return Tensor(seed_count_.p,DType::I32,{1});}
    ops::QsaStateView state() const {
        return {ops::QsaKvFormat::BF16,pool_.plane(0).view({256,64,2}),pool_.plane(1).view({256,64,2}),{},{},
            pool_.plane(2).view({128,64}),pool_.plane(3).view({3,64})};
    }
};
} // namespace ninfer::test::qwen4
