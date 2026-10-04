#include "ops/launcher/rmsnorm_a4.h"
#include "core/device.h"
#include "ops/common/warp.cuh"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {
// One 512-thread CTA per token. Each lane holds five BF16 pairs. Every eight adjacent lanes
// own one G16 group in each register slot. A CTA sum reduction and barrier publish the inverse
// RMS; subgroup shuffles then publish each group's native codec without shared staging.
// Shared memory holds 16 warp sums and one inverse. Every active CTA lane reaches the reduction
// barriers; padded CTAs return uniformly. Input is BF16[5120,T], weight BF16[5120]; outputs are
// tiled A4, optionally BF16.
template <bool PublishNormalized>
__global__ __launch_bounds__(512) void rmsnorm_a4_kernel(const __nv_bfloat162* x,
                                                         const __nv_bfloat162* weight, float eps,
                                                         __nv_bfloat162* normalized,
                                                         std::uint8_t* codes, std::uint8_t* scales,
                                                         float divisor, int tokens) {
    constexpr int kBlock = 512;
    const int lane       = static_cast<int>(threadIdx.x);
    const int token      = static_cast<int>(blockIdx.x);
    if (token >= tokens) {
        if (lane < 320) {
            const std::int64_t tile = static_cast<std::int64_t>(token / 256) * 20 + lane / 16;
            scales[tile * 4096 + (token % 256) * 16 + lane % 16] = 0;
        }
        return;
    }
    const std::int64_t base = static_cast<std::int64_t>(token) * 2560;
    __nv_bfloat162 values[5];
    float sum = 0.0F;
#pragma unroll
    for (int slot = 0; slot < 5; ++slot) {
        values[slot]   = x[base + lane + slot * kBlock];
        const float2 v = __bfloat1622float2(values[slot]);
        sum += v.x * v.x + v.y * v.y;
    }
    __shared__ float sums[16];
    __shared__ float inverse;
    const float total = block_reduce_sum<kBlock>(sum, sums);
    if (lane == 0) { inverse = rsqrtf(total / 5120.0F + eps); }
    __syncthreads();
#pragma unroll
    for (int slot = 0; slot < 5; ++slot) {
        const int pair = lane + slot * kBlock;
        const float2 v = __bfloat1622float2(values[slot]);
        const float2 w = __bfloat1622float2(weight[pair]);
        const auto rounded =
            __floats2bfloat162_rn(v.x * inverse * (1.0F + w.x), v.y * inverse * (1.0F + w.y));
        if constexpr (PublishNormalized) { normalized[base + pair] = rounded; }
        const float2 y = __bfloat1622float2(rounded);
        float maximum  = fmaxf(fabsf(y.x), fabsf(y.y));
#pragma unroll
        for (int shift = 4; shift > 0; shift >>= 1) {
            maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffU, maximum, shift, 8));
        }
        const auto scale =
            __nv_cvt_float_to_fp8(__fdiv_rn(divisor * maximum, 6.0F), __NV_SATFINITE, __NV_E4M3);
        std::uint32_t packed = 0;
        if (scale != 0) {
            const float decoded = decode_nvfp4_e4m3(scale);
            const float a       = __fdiv_rn(y.x * divisor, decoded);
            const float b       = __fdiv_rn(y.y * divisor, decoded);
            asm volatile("{ .reg .b8 pair; cvt.rn.satfinite.e2m1x2.f32 pair, %2, %1; "
                         "cvt.u32.u8 %0, pair; }"
                         : "=r"(packed)
                         : "f"(a), "f"(b));
        }
        codes[base + pair] = static_cast<std::uint8_t>(packed);
        if ((lane & 7) == 0) {
            const int group         = pair / 8;
            const std::int64_t tile = static_cast<std::int64_t>(token / 256) * 20 + group / 16;
            scales[tile * 4096 + (token % 256) * 16 + group % 16] = scale;
        }
    }
}
} // namespace

void rmsnorm_a4_launch(const Tensor& x, const Tensor& weight, float eps, Tensor* normalized,
                       A4Activation& activation, cudaStream_t stream) {
    const auto* input = static_cast<const __nv_bfloat162*>(x.data);
    const auto* gains = static_cast<const __nv_bfloat162*>(weight.data);
    auto* codes       = static_cast<std::uint8_t*>(activation.codes.data);
    auto* scales      = static_cast<std::uint8_t*>(activation.scales.data);
    if (normalized != nullptr) {
        rmsnorm_a4_kernel<true><<<((activation.tokens + 255) / 256) * 256, 512, 0, stream>>>(
            input, gains, eps, static_cast<__nv_bfloat162*>(normalized->data), codes, scales,
            activation.divisor, activation.tokens);
    } else {
        rmsnorm_a4_kernel<false><<<((activation.tokens + 255) / 256) * 256, 512, 0, stream>>>(
            input, gains, eps, nullptr, codes, scales, activation.divisor, activation.tokens);
    }
    CUDA_CHECK(cudaGetLastError());
}
} // namespace ninfer::ops::detail
