#include "targets/qwen4/native_state.h"
#include "core/device.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::targets::qwen4;
namespace {
constexpr int F=10240,R=128*128*48,C=3,W=3;
void upload(const Tensor& t,const void* data,cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(t.data,data,t.bytes(),cudaMemcpyHostToDevice,stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}
int run(ops::QsaKvFormat format) {
    DeviceContext device;
    NativeState state({C,130,256,W,format},device.stream);
    for(int slot=0;slot<C;++slot) state.reserve(slot,64);
    for(int slot=0;slot<C;++slot) state.materialize(slot,3);
    auto records=state.records(W,C);
    const std::vector<int> slots{2,0,1},valid{3,3,3},counts{1,0,3};
    std::vector<float> rec(R*C),conv(F*3*C);
    std::vector<std::uint16_t> raw(F*W*C),zero(128*48*W*C,0);
    std::vector<float> gates(2*48*W*C,0);
    const float g=std::log(.5F);
    for(std::size_t i=0;i<gates.size();i+=2) gates[i]=g;
    for(int layer=0;layer<36;++layer) {
        for(int slot=0;slot<C;++slot) {
            std::fill_n(rec.begin()+slot*R,R,float(layer+1)*.001F+slot*.01F);
            std::fill_n(conv.begin()+slot*F*3,F*3,float(layer+1)*.001F+slot*.01F);
        }
        auto dc=to_device_bf16(conv);
        upload(state.gdn(false).recurrent[layer],rec.data(),device.stream);
        CUDA_CHECK(cudaMemcpyAsync(state.gdn(false).conv[layer].data,dc.p,conv.size()*2,cudaMemcpyDeviceToDevice,device.stream));
        device.synchronize();
        const auto record=records.layer(layer,C);
        for(int b=0;b<C;++b) for(int t=0;t<W;++t)
            std::fill_n(raw.begin()+(b*W+t)*F,F,f32_to_bf16(float(layer*16+b*4+t+1)/512));
        upload(record.conv,raw.data(),device.stream);upload(record.key,zero.data(),device.stream);
        upload(record.value,zero.data(),device.stream);upload(record.gate,gates.data(),device.stream);
    }
    std::vector<std::uint16_t> ple(F*W*C),hidden(F*W*C);
    for(int b=0;b<C;++b) for(int t=0;t<W;++t) {
        std::fill_n(ple.begin()+(b*W+t)*F,F,f32_to_bf16(float(b*4+t+1)/16));
        std::fill_n(hidden.begin()+(b*W+t)*F,F,f32_to_bf16(float(40+b*4+t)));
    }
    auto pr=state.ple_records(W,C);upload(pr,ple.data(),device.stream);
    const std::vector<int> ids{248044,17,18, 19,20,21, 22,248044,24};
    auto di=to_device(ids),ds=to_device(slots),dc=to_device(counts),dh=to_device(hidden);
    Tensor ti(di.p,DType::I32,{W,C}),ts(ds.p,DType::I32,{C}),tc(dc.p,DType::I32,{C}),
        th(dh.p,DType::BF16,{F,W,C});
    state.commit(slots,valid,counts,W,true,ti,th,ts,tc);
    int failures=0;
    for(int b=0;b<C;++b) if(state.frontier(slots[b])!=counts[b]) ++failures;
    for(int layer=0;layer<36;++layer) {
        const auto actual=from_device<float>(state.gdn(false).recurrent[layer].data,R*C);
        const auto actual_conv=from_device<std::uint16_t>(state.gdn(false).conv[layer].data,F*3*C);
        for(int b=0;b<C;++b) {
            const int slot=slots[b];
            double expected=float(layer+1)*.001F+slot*.01F;
            for(int t=0;t<counts[b];++t) expected=float(std::exp(double(g))*expected);
            for(int d=0;d<R;++d) if(!std::isfinite(actual[slot*R+d]) ||
                std::abs(actual[slot*R+d]-expected)>1e-6*std::max(1.,std::abs(expected))) {++failures;break;}
            for(int t=0;t<3;++t) {
                const int source=counts[b]+t-3;
                const auto expected_bits=source<0?f32_to_bf16(float(layer+1)*.001F+slot*.01F):
                    f32_to_bf16(float(layer*16+b*4+source+1)/512);
                for(int d=0;d<F;++d) if(actual_conv[(slot*3+t)*F+d]!=expected_bits) {++failures;break;}
            }
        }
    }
    failures+=verify_exact("full36 prefix raw history",from_device<int>(state.token_history().data,6),
        std::vector<int>{248044,248044,248044,24,248044,248044});
    auto ple_actual=from_device<std::uint16_t>(state.ple(false).data,F*9*C);
    for(int b=0;b<C;++b) for(int t=0;t<9;++t) {
        const int source=counts[b]+t-9;
        const auto expected=source<0?std::uint16_t(0):f32_to_bf16(float(b*4+source+1)/16);
        for(int d=0;d<F;++d) if(ple_actual[(slots[b]*9+t)*F+d]!=expected) {++failures;break;}
    }
    state.retain(2,NativeCheckpoint::Retained);
    const auto checkpoint=from_device<float>(state.gdn(false).recurrent[35].data,R*C);
    state.materialize(2,2);
    for(int l=0;l<36;++l) {
        auto x=state.gdn(true).recurrent_slot(l,2),v=state.gdn(true).conv_slot(l,2);
        CUDA_CHECK(cudaMemsetAsync(x.data,0,x.bytes(),device.stream));
        CUDA_CHECK(cudaMemsetAsync(v.data,0,v.bytes(),device.stream));
    }
    auto ple_next=state.ple(true).slice(2,2,1);
    CUDA_CHECK(cudaMemsetAsync(ple_next.data,0,ple_next.bytes(),device.stream));
    const std::array<int,1> one_slot{2},one{1};
    auto d_one_slot=to_device(std::vector<int>{2}),d_one_count=to_device(std::vector<int>{1});
    Tensor single_ids(di.p,DType::I32,{1,1}),single_hidden(dh.p,DType::BF16,{F,1,1}),
        single_slots(d_one_slot.p,DType::I32,{1}),single_counts(d_one_count.p,DType::I32,{1});
    state.commit(one_slot,one,one,1,false,single_ids,single_hidden,single_slots,single_counts);
    state.retain(2,NativeCheckpoint::Prompt);
    const auto check_prospective_shorter_request=[&] {
        const int frontier=state.frontier(2);
        const auto pages=state.entitled_pages();
        if(frontier<=1 || !state.can_reserve(2,1)) ++failures;
        bool rejected=false;
        try {state.reserve(2,1);} catch(const std::invalid_argument&) {rejected=true;}
        if(!rejected || state.frontier(2)!=frontier || state.entitled_pages()!=pages) ++failures;
    };
    // Admission describes a prospective request, but reserve cannot truncate live state.
    check_prospective_shorter_request();
    state.restore(2,NativeCheckpoint::Retained);
    state.reserve(2,1); // The same maximum is legal after restoring the earlier prefix.
    failures+=verify_exact("full36 retained layer35 restored",
        from_device<float>(state.gdn(false).recurrent[35].data,R*C),checkpoint);
    if(state.frontier(2)!=1 || state.has_retained(2,NativeCheckpoint::Prompt)) ++failures;
    state.retain(1);state.release_reservation(1);
    if(state.can_reserve(2,130) || !state.can_reserve(2,130,std::array<int,1>{1})) ++failures;
    state.evict_retained(1);
    state.reserve(2,130);state.materialize(2,130);
    if(state.entitled_pages()!=4) ++failures;
    state.restore(2); // Restoring a non-page-aligned checkpoint releases the provisional pages.
    state.release_reservation(2);
    if(state.entitled_pages()!=2) ++failures;
    state.reserve(2,64);state.materialize(2,2);
    state.commit(one_slot,one,one,1,false,single_ids,single_hidden,single_slots,single_counts);
    state.retain(2);state.release_reservation(2);
    check_prospective_shorter_request(); // Also applies to an idle retained request.
    state.reset(2);
    state.reserve(2,1); // Reset, like earlier restore, makes the prospective admission real.
    if(state.frontier(2)!=0) ++failures;
    // Cancel before enqueue: a later shorter append must not inherit materialized tail pages.
    state.reserve(2,128);state.materialize(2,128);
    const auto entitlement=state.entitled_pages();
    state.discard_materialization(one_slot);
    if(state.frontier(2)!=0 || state.entitled_pages()!=entitlement) ++failures;
    state.materialize(2,1);
    state.commit(one_slot,one,one,1,false,single_ids,single_hidden,single_slots,single_counts);
    if(state.frontier(2)!=1) ++failures;
    return failures;
}
int idle_score_admission() {
    DeviceContext device;NativeState state({2,128,128,2,ops::QsaKvFormat::NVFP4G16},device.stream);
    state.reserve(1,64);state.materialize(1,64);
    auto ids=to_device(std::vector<int>(64,17)),hidden=to_device(std::vector<std::uint16_t>(F*64,0));
    auto slot=to_device(std::vector<int>{1}),count=to_device(std::vector<int>{64});
    Tensor ti(ids.p,DType::I32,{64,1}),th(hidden.p,DType::BF16,{F,64,1}),
        ts(slot.p,DType::I32,{1}),tc(count.p,DType::I32,{1});
    const std::array<int,1> slots{1},counts{64};
    state.commit(slots,counts,counts,64,false,ti,th,ts,tc);
    state.retain(1);state.release_reservation(1);
    int failures=0;const auto pages=state.entitled_pages();
    if(state.can_reserve(0,128) || !state.can_reserve(0,128,slots) ||
       state.entitled_pages()!=pages || state.frontier(1)!=64 || !state.has_retained(1)) ++failures;
    state.evict_retained(1);state.reserve(0,128);
    if(state.entitled_pages()!=2 || state.frontier(1)!=0 || state.has_retained(1)) ++failures;
    std::cout<<"idle score admission reclaims retained peer prefix: "<<failures<<'\n';return failures;
}
}
int main() {
    if(const int unavailable=require_cuda()) return unavailable;
    const int failures=run(ops::QsaKvFormat::BF16)+run(ops::QsaKvFormat::NVFP4G16)+idle_score_admission();
    std::cout<<(failures?"FAIL ":"PASS ")<<"full native Qwen4 state owner "<<failures<<'\n';
    return failures?1:0;
}
