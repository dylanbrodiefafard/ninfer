#include "ninfer/ops/ple.h"
#include "core/device.h"
#include "ops/op_tester.h"

#include <array>
#include <numeric>

using namespace ninfer;
using namespace ninfer::test;
namespace {
constexpr int channels=10240,batch=4;
int run(int width) {
    DeviceContext device;
    std::vector<std::uint16_t> initial(channels*9*batch),records(channels*width*batch);
    std::vector<int> tokens(width*batch),history(2*batch);
    for(std::size_t i=0;i<initial.size();++i) initial[i]=std::uint16_t(i*37+11);
    for(std::size_t i=0;i<records.size();++i) records[i]=std::uint16_t(i*71+101);
    std::iota(tokens.begin(),tokens.end(),248000);
    tokens[width]=248044; // Row 1 accepts one token: EOS must survive in its newest history word.
    tokens[2*width+std::min(width,15)-1]=248044; // Retained again after the second graph replay.
    std::iota(history.begin(),history.end(),11);
    auto dr=to_device(records),dt=to_device(tokens),dh=to_device(history);
    GuardedDeviceBuffer dc(initial.size()*2);
    CUDA_CHECK(cudaMemcpy(dc.data(),initial.data(),initial.size()*2,cudaMemcpyHostToDevice));
    auto dn=to_device_i32({0,1,std::min(9,width),width}),ds=to_device_i32({3,1,0,2});
    Tensor r(dr.p,DType::BF16,{channels,width,batch}),t(dt.p,DType::I32,{width,batch}),
        n(dn.p,DType::I32,{batch}),s(ds.p,DType::I32,{batch}),
        c(dc.data(),DType::BF16,{channels,9,batch}),h(dh.p,DType::I32,{2,batch});
    // Independent exact sequence oracle: concatenate then retain each stream's tail.
    const auto reference=[&](const std::array<int,batch>& count,const std::array<int,batch>& slots) {
        for(int b=0;b<batch;++b) {
            if(count[b]==0) continue;
            const int slot=slots[b];
            std::vector<std::uint16_t> columns(initial.begin()+slot*9*channels,
                                               initial.begin()+(slot+1)*9*channels);
            columns.insert(columns.end(),records.begin()+b*width*channels,
                           records.begin()+(b*width+count[b])*channels);
            std::copy(columns.end()-9*channels,columns.end(),initial.begin()+slot*9*channels);
            std::vector<int> sequence{history[slot*2],history[slot*2+1]};
            sequence.insert(sequence.end(),tokens.begin()+b*width,tokens.begin()+b*width+count[b]);
            history[slot*2]=sequence[sequence.size()-2];history[slot*2+1]=sequence.back();
        }
    };
    cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    CUDA_CHECK(cudaStreamBeginCapture(device.stream,cudaStreamCaptureModeThreadLocal));
    ops::ple_commit_prefix(r,t,n,s,c,h,device.stream);
    CUDA_CHECK(cudaStreamEndCapture(device.stream,&graph));
    CUDA_CHECK(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
    int failures=0;
    for(int pass=0;pass<2;++pass) {
        const std::array<int,batch> counts=pass==0?std::array<int,batch>{0,1,std::min(9,width),width}:
            std::array<int,batch>{std::min(3,width),std::min(7,width),std::min(15,width),0};
        const std::array<int,batch> slots=pass==0?std::array<int,batch>{3,1,0,2}:
                                                  std::array<int,batch>{2,0,3,1};
        dn.copy_from_host(counts.data(),sizeof(counts));ds.copy_from_host(slots.data(),sizeof(slots));
        reference(counts,slots);
        CUDA_CHECK(cudaGraphLaunch(executable,device.stream));device.synchronize();
        failures+=verify_exact("PLE accepted prefix, zero row, permutation and graph history",
            from_device<std::uint16_t>(dc.data(),initial.size()),initial);
        failures+=verify_exact("PLE accepted raw token history incl EOS",from_device<int>(dh,history.size()),history);
    }
    failures+=verify_exact("PLE rejected records remain immutable",from_device<std::uint16_t>(dr,records.size()),records);
    failures+=verify_exact("PLE verification tokens remain immutable",from_device<int>(dt,tokens.size()),tokens);
    failures+=dc.verify_guards("PLE commit state");
    CUDA_CHECK(cudaGraphExecDestroy(executable));CUDA_CHECK(cudaGraphDestroy(graph));
    return failures;
}
}
int main() { return (run(1)+run(7)+run(16))?1:0; }
