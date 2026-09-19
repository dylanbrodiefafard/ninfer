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
struct GdnQkvOutput {
    __nv_bfloat16* data;
    __device__ __forceinline__ GdnQkvOutput tile(int) const { return *this; }
    __device__ __forceinline__ void store(int row, int token, double value) const {
        data[static_cast<std::int64_t>(token) * 10240 + row] = __double2bfloat16(value);
    }
};
__device__ __forceinline__ void gdn_pair_add(float value, float& high, float& low) {
    const float sum = __fadd_rn(high, value);
    const float recovered = __fsub_rn(sum, high);
    const float error = __fadd_rn(__fsub_rn(high, __fsub_rn(sum, recovered)),
                                  __fsub_rn(value, recovered));
    high = sum;
    low = __fadd_rn(low, error);
}

// The existing decode/small-T row, K-phase and weight-reuse schedules, with compensated
// accumulators and warp reduction instead of the ordinary Linear single-float profile.
template<int Tokens, class Schedule>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm)
void gdn_qkv_small(const __nv_bfloat16* x, const __nv_bfloat16* weight, GdnQkvOutput output) {
    using Geometry = Bf16GemvGeometry<10240,2560>;
    static_assert(Schedule::kWarpsPerRow == 1);
    constexpr int phases = Geometry::kInputRows / (32 * Schedule::kValuesPerLane);
    const int lane = threadIdx.x & 31;
    const int row0 = blockIdx.x * Schedule::kRowsPerCta +
                    (threadIdx.x / 32) * Schedule::kRowsPerWarp;
    float high[Schedule::kRowsPerWarp][Tokens] = {};
    float low[Schedule::kRowsPerWarp][Tokens] = {};
#pragma unroll Schedule::kPhaseUnroll
    for (int iteration = 0; iteration < phases; ++iteration) {
        const int phase = bf16_phase_index<Schedule,phases>(iteration,row0);
        Bf16GemvPack<Schedule::kValuesPerLane> packed_weights[Schedule::kRowsPerWarp];
#pragma unroll
        for (int row = 0; row < Schedule::kRowsPerWarp; ++row) {
            packed_weights[row] = load_bf16_weight_phase<Geometry,Schedule>(
                weight,row0+row,phase,0,lane);
        }
#pragma unroll
        for (int token = 0; token < Tokens; ++token) {
            const auto activation = load_bf16_activation_phase<Geometry,Schedule>(
                x + token * Geometry::kInputRows,phase,0,lane);
#pragma unroll
            for (int row = 0; row < Schedule::kRowsPerWarp; ++row) {
#pragma unroll
                for (int pair = 0; pair < Schedule::kValuesPerLane/2; ++pair) {
                    const auto w = bf16x2_bits_to_float2(packed_weights[row].words[pair]);
                    const auto a = bf16x2_bits_to_float2(activation.words[pair]);
                    gdn_pair_add(__fmul_rn(w.x,a.x),high[row][token],low[row][token]);
                    gdn_pair_add(__fmul_rn(w.y,a.y),high[row][token],low[row][token]);
                }
            }
        }
    }
#pragma unroll
    for (int row = 0; row < Schedule::kRowsPerWarp; ++row) {
#pragma unroll
        for (int token = 0; token < Tokens; ++token) {
            float h = high[row][token], l = low[row][token];
#pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                const float other_h = __shfl_down_sync(0xffffffffU,h,offset);
                const float other_l = __shfl_down_sync(0xffffffffU,l,offset);
                if (lane < offset) {
                    gdn_pair_add(other_h,h,l);
                    gdn_pair_add(other_l,h,l);
                }
            }
            if (lane == 0) output.store(row0+row,token,static_cast<double>(h)+static_cast<double>(l));
        }
    }
}
template<int Tokens>
void gdn_small(const Tensor& x,const Weight& weight,Tensor& out,cudaStream_t stream) {
    using Geometry = Bf16GemvGeometry<10240,2560>;
    if constexpr (Tokens == 1) {
        using Schedule = Bf16LinearDecodeSchedule<Geometry>;
        gdn_qkv_small<Tokens,Schedule><<<10240/Schedule::kRowsPerCta,Schedule::kThreads,0,stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),static_cast<const __nv_bfloat16*>(weight.qdata),
            GdnQkvOutput{static_cast<__nv_bfloat16*>(out.data)});
    } else {
        using Schedule = typename Bf16LinearSmallTProductionSchedule<Geometry,Tokens>::Type;
        gdn_qkv_small<Tokens,Schedule><<<10240/Schedule::kRowsPerCta,Schedule::kThreads,0,stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),static_cast<const __nv_bfloat16*>(weight.qdata),
            GdnQkvOutput{static_cast<__nv_bfloat16*>(out.data)});
    }
    CUDA_CHECK(cudaGetLastError());
}
template<std::size_t... I>
constexpr auto gdn_small_launchers(std::index_sequence<I...>) {
    return std::array<Bf16Launch,sizeof...(I)>{&gdn_small<1+static_cast<int>(I)>...};
}
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

void launch_bf16_gdn_qkv(const Tensor& x,const Weight& weight,Tensor& out,cudaStream_t stream) {
    if (x.dtype != DType::BF16 || out.dtype != DType::BF16 || x.ne[1] < 1 ||
        weight.qtype != QType::BF16_CTRL || weight.n != 10240 || weight.k != 2560) {
        throw std::invalid_argument("native GDN BF16 QKV: unsupported shape or representation");
    }
    using Geometry = Bf16GemvGeometry<10240,2560>;
    // This compensated profile has a separately measured crossover; generic Linear is unchanged.
    constexpr int kGdnQkvSmallTEnd = 20;
    if (x.ne[1] <= kGdnQkvSmallTEnd) {
        constexpr auto launchers = gdn_small_launchers(
            std::make_index_sequence<kGdnQkvSmallTEnd>{});
        launchers[x.ne[1]-1](x,weight,out,stream);
        return;
    }
    using Schedule = Bf16MmaProductionSchedule<Geometry>;
    static const auto attr = cudaFuncSetAttribute(
        bf16_gemm_mma_kernel<Geometry,Schedule,false,GdnQkvOutput,true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,Schedule::kSharedBytes);
    CUDA_CHECK(attr);
    const int blocks = (Geometry::kOutputRows / Schedule::kBlockRows) *
        ((x.ne[1] + Schedule::kBlockCols - 1) / Schedule::kBlockCols);
    bf16_gemm_mma_kernel<Geometry,Schedule,false,GdnQkvOutput,true>
        <<<blocks,Schedule::kThreads,Schedule::kSharedBytes,stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(weight.qdata),
            GdnQkvOutput{static_cast<__nv_bfloat16*>(out.data)},x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}
}
