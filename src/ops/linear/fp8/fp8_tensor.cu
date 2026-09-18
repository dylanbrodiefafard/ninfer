#include "ops/linear/fp8/fp8_tensor.h"
#include "core/device.h"
#include "ops/common/warp.cuh"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_gemv.cuh"
#include "ops/linear/fp8/fp8_small_t.cuh"
#include "ops/linear/fp8/fp8_a16_gemm_mma.cuh"
#include "ops/linear/fp8/fp8_a8_schedule.cuh"
#include <cuda_fp8.h>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ninfer::ops::detail {
namespace {

bool aligned(const void* p, std::uintptr_t n) {
    return p && (reinterpret_cast<std::uintptr_t>(p) & (n - 1)) == 0;
}

void validate_profile(int n, int k, LinearPolicy policy) {
    if (!is_fp8_tensor_problem(n, k) ||
        (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA8)) {
        throw std::invalid_argument("tensor-calibrated FP8: unsupported shape/policy");
    }
}

template<int K>
__global__ void guarded_pack(const __nv_bfloat16* input, const float* multiplier,
                            std::uint8_t* codes, float* scales) {
    const int token = blockIdx.x;
    constexpr int pairs=(K/2+255)/256;
    float2 values[pairs];
    float maximum=0.F;
    __shared__ float maxima[8],token_scale;
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
#pragma unroll
    for(int item=0;item<pairs;++item) {
        const int column=(threadIdx.x+item*256)*2;
        values[item]=column<K?bf16x2_bits_to_float2(load_vec<std::uint32_t>(
            input+static_cast<std::int64_t>(token)*K+column)):make_float2(0.F,0.F);
        maximum=fmaxf(maximum,fmaxf(fabsf(values[item].x),fabsf(values[item].y)));
    }
    maximum=warp_max(maximum);
    if(lane==0) { maxima[warp]=maximum; }
    __syncthreads();
    if(warp==0) {
        maximum=warp_max(lane<8?maxima[lane]:0.F);
        if(lane==0) { token_scale=fmaxf(multiplier[1],__fdiv_rn(maximum,448.F)); }
    }
    __syncthreads();
    const float scale=token_scale;
#pragma unroll
    for(int item=0;item<pairs;++item) {
        const int column=(threadIdx.x+item*256)*2;
        const float2 pair=values[item];
        const float2 divided = make_float2(__fdiv_rn(pair.x, scale), __fdiv_rn(pair.y, scale));
        if(column<K) { *reinterpret_cast<std::uint16_t*>(codes + static_cast<std::int64_t>(token) * K + column) =
            __nv_cvt_float2_to_fp8x2(divided, __NV_SATFINITE, __NV_E4M3);
        }
    }
    if (threadIdx.x == 0) { scales[token] = scale; }
}

template<class Output>
Output output_view(Tensor& out,int rows) {
    if constexpr (std::is_same_v<Output,Fp8ContiguousF32Output>) {
        return {static_cast<float*>(out.data),rows};
    } else { return {static_cast<__nv_bfloat16*>(out.data),rows}; }
}

template<class Geometry, int Tokens,class Output>
void small(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const Fp8TensorMultiplier scales{static_cast<const float*>(weight.scales)};
    const auto output=output_view<Output>(out,Geometry::kOutputRows);
    if constexpr (Tokens == 1) {
        using Schedule = typename Fp8LinearDecodeProductionSchedule<Geometry>::Type;
        fp8_gemv_kernel<Geometry, Schedule>
            <<<Geometry::kOutputRows / Schedule::kRowsPerCta, Schedule::kThreads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const std::uint8_t*>(weight.qdata), scales, output);
    } else {
        using Schedule = typename Fp8LinearSmallTProductionSchedule<Geometry, Tokens>::Type;
        constexpr int tiles = (Tokens + Schedule::kTokenTile - 1) / Schedule::kTokenTile;
        fp8_small_t_kernel<Geometry, Tokens, Schedule>
            <<<Geometry::kOutputRows / Schedule::kRowsPerCta * tiles, Schedule::kThreads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const std::uint8_t*>(weight.qdata), scales, output);
    }
    CUDA_CHECK(cudaGetLastError());
}

template<class Geometry, bool Full,class Output>
void a16_mma(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Schedule = Fp8A16GemmSchedule<64,128,64,32,16,2,2>;
    const dim3 grid(Geometry::kOutputRows / Schedule::kBlockRows,
                    (x.ne[1] + Schedule::kBlockTokens - 1) / Schedule::kBlockTokens);
    const auto output=output_view<Output>(out,Geometry::kOutputRows);
    fp8_a16_gemm_mma_kernel<Geometry, Schedule, Full>
        <<<grid, Schedule::kThreads, Schedule::kSharedBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
            Fp8TensorMultiplier{static_cast<const float*>(weight.scales)}, output, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

template<class Geometry, bool Full,class Output>
void a8_mma(const Weight& weight, Tensor& out, Fp8A8Workspace scratch, int tokens,
            cudaStream_t stream) {
    using Schedule = typename Fp8LinearA8ProductionSchedule<Geometry>::Type;
    if constexpr (Schedule::kSharedBytes > 48 * 1024) {
        static const cudaError_t attribute = cudaFuncSetAttribute(
            fp8_mma_kernel<Geometry, Schedule, Full, Fp8IdentityEpilogue, Output,
                           Fp8MmaIdentityRows, false, Fp8TensorMultiplier>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, Schedule::kSharedBytes);
        CUDA_CHECK(attribute);
    }
    const int blocks = (Geometry::kOutputRows / Schedule::kBlockRows) *
                       ((tokens + Schedule::kBlockTokens - 1) / Schedule::kBlockTokens);
    const auto output=output_view<Output>(out,Geometry::kOutputRows);
    fp8_mma_kernel<Geometry, Schedule, Full>
        <<<blocks, Schedule::kThreads, Schedule::kSharedBytes, stream>>>(
            scratch.codes, scratch.scales, static_cast<const std::uint8_t*>(weight.qdata),
            Fp8TensorMultiplier{static_cast<const float*>(weight.scales)}, tokens,
            Fp8IdentityEpilogue{}, output);
    CUDA_CHECK(cudaGetLastError());
}

template<class Geometry,class Output=Fp8ContiguousOutput>
void run(const Tensor& x, const Weight& weight, Tensor& out, bool a8,
         Fp8A8Workspace scratch, cudaStream_t stream) {
    if constexpr (!std::is_same_v<Output,Fp8ContiguousF32Output>) {
        if (a8) {
            using Schedule = typename Fp8LinearA8ProductionSchedule<Geometry>::Type;
            if (x.ne[1] % Schedule::kBlockTokens == 0) { a8_mma<Geometry,true,Output>(weight,out,scratch,x.ne[1],stream); }
            else { a8_mma<Geometry,false,Output>(weight,out,scratch,x.ne[1],stream); }
            return;
        }
    }
    if (x.ne[1] >= fp8_exact_a16_gemm_first_t(resolve_fp8_problem(weight.n, weight.k))) {
        if (x.ne[1] % 128 == 0) { a16_mma<Geometry,true,Output>(x,weight,out,stream); }
        else { a16_mma<Geometry,false,Output>(x,weight,out,stream); }
        return;
    }
    // Reuse the existing canonical four-token A16 panels, never repack model weights.
    for (int offset=0; offset<x.ne[1]; offset+=4) {
        const int count=std::min(4,x.ne[1]-offset);
        const Tensor input=x.slice(1,offset,count); Tensor output=out.slice(1,offset,count);
        switch(count) {
        case 1: small<Geometry,1,Output>(input,weight,output,stream); break;
        case 2: small<Geometry,2,Output>(input,weight,output,stream); break;
        case 3: small<Geometry,3,Output>(input,weight,output,stream); break;
        case 4: small<Geometry,4,Output>(input,weight,output,stream); break;
        }
    }
}

void launch(const Tensor& x,const Weight& w,Tensor& out,bool a8,Fp8A8Workspace scratch,cudaStream_t stream) {
    switch(resolve_fp8_problem(w.n,w.k)) {
#define TENSOR_CASE(Name) case Fp8Problem::Name: run<Fp8##Name##Geometry>(x,w,out,a8,scratch,stream); return
    TENSOR_CASE(Rows10240K2560); TENSOR_CASE(Rows6144K2560); TENSOR_CASE(Rows12288K2560);
    TENSOR_CASE(Rows512K2560); TENSOR_CASE(Rows2560K6144); TENSOR_CASE(Rows640K2560);
    TENSOR_CASE(Rows2560K640);
#undef TENSOR_CASE
    default: throw std::invalid_argument("tensor-calibrated FP8: unsupported problem");
    }
}
} // namespace

bool is_fp8_tensor_problem(std::int32_t n,std::int32_t k) {
    return (k==2560 && (n==10240 || n==6144 || n==12288 || n==512 || n==640)) ||
           (n==2560 && (k==6144 || k==640));
}

std::int32_t fp8_tensor_a8_first_t(std::int32_t n,std::int32_t k) {
    // Packing-inclusive public-Linear measurements, not row-scale route cutovers.
    if (n==10240 && k==2560) { return 9; }
    if (n==6144 && k==2560) { return 13; }
    if (n==12288 && k==2560) { return 9; }
    if ((n==512 || n==640) && k==2560) { return 25; }
    if (n==2560 && k==6144) { return 24; }
    if (n==2560 && k==640) { return 9; }
    throw std::invalid_argument("tensor-calibrated FP8: unsupported problem");
}

void validate_fp8_tensor_weight(const Weight& w,const char* op) {
    const auto bytes=static_cast<std::uint64_t>(w.n)*static_cast<std::uint64_t>(w.k);
    const auto scale_offset=(bytes+3)&~std::uint64_t{3};
    if (!is_fp8_tensor_problem(w.n,w.k) || w.qtype!=QType::FP8_E4M3FN_TENSOR_F32M ||
        w.layout!=QuantLayout::TensorCalibrated || w.scale_dtype!=DType::FP32 ||
        w.ndim!=2 || w.shape[0]!=w.n || w.shape[1]!=w.k || w.shape[2]!=1 || w.shape[3]!=1 ||
        w.padded_shape[0]!=w.n || w.padded_shape[1]!=w.k || w.padded_shape[2]!=1 || w.padded_shape[3]!=1 ||
        w.scale_ne[0]!=2 || w.scale_ne[1]!=1 || w.scale_ne[2]!=1 || w.scale_ne[3]!=1 ||
        w.scale_nb[0]!=4 || w.scale_nb[1]!=8 || w.scale_nb[2]!=8 || w.scale_nb[3]!=8 ||
        w.payload==nullptr || w.qdata!=w.payload || w.scales!=static_cast<const std::byte*>(w.payload)+scale_offset ||
        w.qhigh || w.high_plane_bytes || w.payload_bytes<scale_offset+8 ||
        !aligned(w.qdata,16) || !aligned(w.scales,4)) {
        throw std::invalid_argument(std::string(op)+": invalid tensor-calibrated FP8 weight");
    }
}

std::size_t fp8_tensor_workspace_capacity_bytes(int n,int k,LinearPolicy policy,int min_t,int max_t) {
    validate_profile(n,k,policy);
    if(min_t<=0 || max_t<min_t) { throw std::invalid_argument("tensor FP8: invalid token interval"); }
    return policy==LinearPolicy::AllowA8 && max_t>=fp8_tensor_a8_first_t(n,k)
        ? fp8_a8_workspace_capacity_bytes(max_t,k):0;
}

void launch_fp8_tensor_quantize(const Tensor& x,const Weight& w,Fp8A8Workspace scratch,cudaStream_t stream) {
    switch(w.k) {
#define PACK_CASE(K) case K: guarded_pack<K><<<x.ne[1],256,0,stream>>>( \
        static_cast<const __nv_bfloat16*>(x.data),static_cast<const float*>(w.scales), \
        scratch.codes,scratch.scales); break
    PACK_CASE(640); PACK_CASE(2560); PACK_CASE(6144);
#undef PACK_CASE
    default: throw std::invalid_argument("tensor FP8 pack: unsupported input width");
    }
    CUDA_CHECK(cudaGetLastError());
}

void fp8_tensor_dispatch(const Tensor& x,const Weight& w,Tensor& out,LinearPolicy policy,
                         WorkspaceArena* workspace,cudaStream_t stream) {
    validate_profile(w.n,w.k,policy); validate_fp8_tensor_weight(w,"linear");
    const bool a8=policy==LinearPolicy::AllowA8 && x.ne[1]>=fp8_tensor_a8_first_t(w.n,w.k);
    if(!a8) { launch(x,w,out,false,{},stream); return; }
    if(!workspace) { throw std::invalid_argument("tensor FP8 A8 requires caller workspace"); }
    auto scope=workspace->scope();
    const auto scratch=allocate_fp8_a8_workspace(*workspace,x.ne[1],w.k);
    launch_fp8_tensor_quantize(x,w,scratch,stream); launch(x,w,out,true,scratch,stream);
}

void fp8_tensor_key_f32(const Tensor& x,const Weight& w,Tensor& out,cudaStream_t stream) {
    validate_profile(w.n,w.k,LinearPolicy::A16Only);validate_fp8_tensor_weight(w,"qsa key");
    if(w.n!=512 || w.k!=2560 || out.dtype!=DType::FP32) {
        throw std::invalid_argument("qsa tensor FP8 key: expected [512,2560] FP32 output");
    }
    run<Fp8Rows512K2560Geometry,Fp8ContiguousF32Output>(x,w,out,false,{},stream);
}
} // namespace ninfer::ops::detail
