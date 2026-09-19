#include "ops/launcher/linear_bias.h"
#include "ops/linear/bf16/bf16_config.h"
#include "ops/linear/bf16/bf16_gemm_mma_config.h"
#include "core/device.h"
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {
struct BiasedOutput {
    __nv_bfloat16* output;
    const __nv_bfloat16* bias;
    int rows;
    __device__ __forceinline__ BiasedOutput tile(int) const { return *this; }
    __device__ __forceinline__ void store(int row,int token,float value) const {
        output[static_cast<std::int64_t>(token)*rows+row]=__float2bfloat16_rn(value+__bfloat162float(bias[row]));
    }
};
template<int N,int K,bool Full>
void launch(const Tensor& x,const Weight& w,const Tensor& bias,Tensor& out,cudaStream_t stream) {
    using Geometry=Bf16GemvGeometry<N,K>;
    using Schedule=Bf16MmaProductionSchedule<Geometry>;
    if constexpr(Schedule::kSharedBytes>48*1024) {
        static const auto attr=cudaFuncSetAttribute(bf16_gemm_mma_kernel<Geometry,Schedule,Full,BiasedOutput>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,Schedule::kSharedBytes);
        CUDA_CHECK(attr);
    }
    const int blocks=((N+Schedule::kBlockRows-1)/Schedule::kBlockRows)*
                      ((x.ne[1]+Schedule::kBlockCols-1)/Schedule::kBlockCols);
    bf16_gemm_mma_kernel<Geometry,Schedule,Full><<<blocks,Schedule::kThreads,Schedule::kSharedBytes,stream>>>(
        static_cast<const __nv_bfloat16*>(x.data),static_cast<const __nv_bfloat16*>(w.qdata),
        BiasedOutput{static_cast<__nv_bfloat16*>(out.data),static_cast<const __nv_bfloat16*>(bias.data),N},x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}
template<int N,int K>
void dispatch(const Tensor& x,const Weight& w,const Tensor& b,Tensor& y,cudaStream_t s) {
    using Schedule=Bf16MmaProductionSchedule<Bf16GemvGeometry<N,K>>;
    if(x.ne[1]%Schedule::kBlockCols==0) launch<N,K,true>(x,w,b,y,s);
    else launch<N,K,false>(x,w,b,y,s);
}
}
void linear_bias_launch(const Tensor& x,const Weight& w,const Tensor& b,Tensor& y,cudaStream_t s) {
    if(w.n==1152 && w.k==1536) dispatch<1152,1536>(x,w,b,y,s);
    else if(w.n==3456 && w.k==1152) dispatch<3456,1152>(x,w,b,y,s);
    else if(w.n==1152 && w.k==1152) dispatch<1152,1152>(x,w,b,y,s);
    else if(w.n==4304 && w.k==1152) dispatch<4304,1152>(x,w,b,y,s);
    else if(w.n==1152 && w.k==4304) dispatch<1152,4304>(x,w,b,y,s);
    else if(w.n==4608 && w.k==4608) dispatch<4608,4608>(x,w,b,y,s);
    else if(w.n==2560 && w.k==4608) dispatch<2560,4608>(x,w,b,y,s);
    else throw std::invalid_argument("linear_bias launcher: unsupported shape");
}
} // namespace ninfer::ops::detail
