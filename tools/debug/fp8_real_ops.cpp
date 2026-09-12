// Execute public FP8 Ops on retained represented weights/activations.
// CPU oracle and artifact decoding are deliberately outside this executable.
// Build in ninfer-builder after the ordinary CMake build:
// nvcc -O2 -std=c++20 -arch=sm_120a -x cu -I/src/include -I/src/src -I/src/bench/ops
//   -c /src/tools/debug/fp8_real_ops.cpp -o /src/out/fp8-328-release/fp8_real_ops.o
// g++ /src/out/fp8-328-release/fp8_real_ops.o -o /src/out/fp8-328-release/fp8_real_ops
//   /build/src/libninfer_ops.a /build/src/libninfer_nvfp4_tma.a /build/src/libninfer_core.a
//   -L/usr/local/cuda/lib64 -lcudart -lcuda -ldl -lpthread
#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer_bench_common.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
using namespace ninfer;

std::vector<char> read(const std::string& path) {
    std::ifstream f(path,std::ios::binary|std::ios::ate);
    if (!f) throw std::runtime_error("read: "+path);
    std::vector<char> data(f.tellg());
    f.seekg(0); f.read(data.data(),data.size());
    if (!f) throw std::runtime_error("short read: "+path);
    return data;
}
void save(const std::string& path,void* data,std::size_t size) {
    std::vector<char> host(size);
    CUDA_CHECK(cudaMemcpy(host.data(),data,size,cudaMemcpyDeviceToHost));
    std::ofstream f(path,std::ios::binary);
    f.write(host.data(),host.size());
    if (!f) throw std::runtime_error("write: "+path);
}
int main(int argc,char** argv) {
    try {
        if (argc!=7 && argc!=8) throw std::runtime_error("usage: fp8_real_ops directory N K T a16|a8 linear|add|swiglu|attn [bench-swiglu]");
        const std::string root=argv[1], mode=argv[6];
        const int n=std::stoi(argv[2]), k=std::stoi(argv[3]), t=std::stoi(argv[4]);
        const auto policy=std::string(argv[5])=="a8" ? ops::LinearPolicy::AllowA8 : ops::LinearPolicy::A16Only;
        auto payload=read(root+"/weight.bin"), input=read(root+"/input.bin");
        const auto scale_offset=(std::size_t(n)*k+255)/256*256;
        if (payload.size()!=scale_offset+2*n || input.size()!=std::size_t(k)*t*2)
            throw std::runtime_error("input size mismatch");
        void *wd=nullptr,*xd=nullptr,*yd=nullptr;
        CUDA_CHECK(cudaMalloc(&wd,payload.size())); CUDA_CHECK(cudaMalloc(&xd,input.size()));
        CUDA_CHECK(cudaMalloc(&yd,std::size_t(n)*t*2));
        CUDA_CHECK(cudaMemcpy(wd,payload.data(),payload.size(),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(xd,input.data(),input.size(),cudaMemcpyHostToDevice));
        Weight w{}; w.payload=w.qdata=wd; w.payload_bytes=payload.size();
        w.scales=static_cast<char*>(wd)+scale_offset; w.qtype=QType::FP8_E4M3FN_ROW_BF16S;
        w.layout=QuantLayout::RowScale; w.scale_dtype=DType::BF16;
        w.n=n; w.k=k; w.ndim=2; w.group=w.group_size=k;
        w.shape[0]=w.padded_shape[0]=n; w.shape[1]=w.padded_shape[1]=k;
        w.scale_ne[0]=n; w.scale_nb[0]=2;
        for (int i=1;i<4;++i) { w.scale_ne[i]=1; w.scale_nb[i]=2LL*n; }
        for (int i=2;i<4;++i) w.shape[i]=w.padded_shape[i]=1;
        Tensor x(xd,DType::BF16,{k,t}), y(yd,DType::BF16,{mode=="swiglu"?n/2:n,t});
        const auto scratch = mode=="attn" ? ops::attn_input_proj_workspace_capacity_bytes(w.qtype,n,k,policy,t,t)
            : mode=="swiglu" ? ops::linear_swiglu_workspace_capacity_bytes(w.qtype,n,k,policy,t,t)
            : mode=="add" ? ops::linear_add_workspace_capacity_bytes(w.qtype,n,k,policy,t,t)
            : ops::linear_workspace_capacity_bytes(w.qtype,n,k,policy,t,t);
        WorkspaceArena work(std::max<std::size_t>(scratch,256));
        if (mode=="linear") ops::linear(x,w,y,policy,work,nullptr);
        else if (mode=="swiglu") ops::linear_swiglu(x,w,y,policy,work,nullptr);
        else if (mode=="add") {
            auto residual=read(root+"/residual.bin");
            if (residual.size()!=std::size_t(n)*t*2) throw std::runtime_error("residual size");
            CUDA_CHECK(cudaMemcpy(yd,residual.data(),residual.size(),cudaMemcpyHostToDevice));
            ops::linear_add(x,w,y,policy,work,nullptr);
        } else if (mode=="attn") {
            auto* p=static_cast<std::uint16_t*>(yd);
            Tensor q(p,DType::BF16,{6144,t}), key(p+6144*t,DType::BF16,{1024,t});
            Tensor gate(p+7168*t,DType::BF16,{6144,t}), value(p+13312*t,DType::BF16,{1024,t});
            ops::attn_input_proj(x,w,q,gate,key,value,policy,work,nullptr);
        } else throw std::runtime_error("unknown mode");
        CUDA_CHECK(cudaDeviceSynchronize());
        if (argc==8) {
            if (mode!="swiglu" || std::string(argv[7])!="bench-swiglu")
                throw std::runtime_error("timing mode supports SwiGLU only");
            DeviceBuffer flush(256ULL<<20);
            const auto timing=bench::measure_cold_launch([&](cudaStream_t stream) {
                ops::linear_swiglu(x,w,y,policy,work,stream);
            },flush,nullptr,3,20);
            std::cout<<"median_us="<<timing.median_us<<" min_us="<<timing.min_us
                     <<" p95_us="<<timing.p95_us<<'\n';
        }
        save(root+"/output.bin",yd,std::size_t(mode=="swiglu"?n/2:n)*t*2);
        CUDA_CHECK(cudaFree(yd)); CUDA_CHECK(cudaFree(xd)); CUDA_CHECK(cudaFree(wd));
        return 0;
    } catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
