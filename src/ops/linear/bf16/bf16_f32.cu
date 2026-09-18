#include "ops/linear/bf16/bf16_launch.h"
#include "ops/linear/bf16/bf16_gemv.cuh"
#include "ops/linear/bf16/bf16_small_t.cuh"
#include "ops/linear/bf16/bf16_gemm_mma_config.h"
#include "core/device.h"

#include <array>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {
// Private component precision policy. The canonical arithmetic kernels and crossover remain
// unchanged; only the otherwise-unobservable projection output avoids a premature BF16 cast.
struct F32Output {
    float* data;
    int rows;
    __device__ __forceinline__ void store(int row,float value) const { data[row]=value; }
    __device__ __forceinline__ void store(int row,int token,float value) const {
        data[static_cast<std::int64_t>(token)*rows+row]=value;
    }
    __device__ __forceinline__ F32Output tile(int) const { return *this; }
};
struct F32Epilogue {
    __device__ __forceinline__ void operator()(const F32Output& output,int row,float value) const {
        output.store(row,value);
    }
};
template<class Geometry,int Tokens>
void small(const Tensor& x,const Weight& weight,Tensor& out,cudaStream_t stream) {
    using Schedule=typename Bf16LinearSmallTProductionSchedule<Geometry,Tokens>::Type;
    bf16_small_t_inner_kernel<Geometry,Tokens,Schedule>
        <<<Geometry::kOutputRows/Schedule::kRowsPerCta,Schedule::kThreads,0,stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),static_cast<const __nv_bfloat16*>(weight.qdata),
            F32Output{static_cast<float*>(out.data),Geometry::kOutputRows});
    CUDA_CHECK(cudaGetLastError());
}
template<class Geometry,std::size_t... I>
constexpr auto small_launchers(std::index_sequence<I...>) {
    return std::array<Bf16Launch,sizeof...(I)>{&small<Geometry,2+static_cast<int>(I)>...};
}
template<class Geometry,bool Full>
void mma(const Tensor& x,const Weight& weight,Tensor& out,cudaStream_t stream) {
    using Schedule=Bf16MmaProductionSchedule<Geometry>;
    if constexpr(Schedule::kSharedBytes>48*1024) {
        static const auto attr=cudaFuncSetAttribute(bf16_gemm_mma_kernel<Geometry,Schedule,Full,F32Output>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,Schedule::kSharedBytes);
        CUDA_CHECK(attr);
    }
    const int blocks=(Geometry::kOutputRows/Schedule::kBlockRows)*
        ((x.ne[1]+Schedule::kBlockCols-1)/Schedule::kBlockCols);
    bf16_gemm_mma_kernel<Geometry,Schedule,Full>
        <<<blocks,Schedule::kThreads,Schedule::kSharedBytes,stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),static_cast<const __nv_bfloat16*>(weight.qdata),
            F32Output{static_cast<float*>(out.data),Geometry::kOutputRows},x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}
template<class Geometry>
void launch(const Tensor& x,const Weight& weight,Tensor& out,cudaStream_t stream) {
    if(x.ne[1]==1) {
        using Schedule=Bf16LinearDecodeSchedule<Geometry>;
        bf16_gemv_kernel<Geometry,Schedule,F32Output,F32Epilogue>
            <<<Geometry::kOutputRows/Schedule::kRowsPerCta,Schedule::kThreads,0,stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),static_cast<const __nv_bfloat16*>(weight.qdata),
                F32Output{static_cast<float*>(out.data),Geometry::kOutputRows});
        CUDA_CHECK(cudaGetLastError());
    } else if(x.ne[1]<=kBf16LinearSmallTDispatchEnd) {
        constexpr auto launchers=small_launchers<Geometry>(
            std::make_index_sequence<kBf16LinearSmallTDispatchEnd-1>{});
        launchers[x.ne[1]-2](x,weight,out,stream);
    } else {
        using Schedule=Bf16MmaProductionSchedule<Geometry>;
        if(x.ne[1]%Schedule::kBlockCols==0) { mma<Geometry,true>(x,weight,out,stream); }
        else { mma<Geometry,false>(x,weight,out,stream); }
    }
}
}

void launch_bf16_f32(const Tensor& x,const Weight& weight,Tensor& out,cudaStream_t stream) {
    if(x.dtype!=DType::BF16 || out.dtype!=DType::FP32 || x.ne[1]<1 ||
       weight.qtype!=QType::BF16_CTRL || weight.k!=2560) {
        throw std::invalid_argument("private BF16 projection: expected BF16 input/weight and FP32 output");
    }
    if(weight.n==512) { launch<Bf16GemvGeometry<512,2560>>(x,weight,out,stream); }
    else if(weight.n==640) { launch<Bf16GemvGeometry<640,2560>>(x,weight,out,stream); }
    else { throw std::invalid_argument("private BF16 FP32 projection: unsupported exact shape"); }
}
}
