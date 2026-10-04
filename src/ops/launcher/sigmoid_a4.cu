#include "core/device.h"
#include "core/tensor.h"
#include "ninfer/ops/a4_activation.h"
#include "ops/common/bf16_vector.cuh"
#include "ops/common/math.cuh"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

// Each 256-thread CTA owns consecutive eight-element packs; two adjacent lanes own G16.
// Three CTAs cover one token, so padding CTAs return uniformly before reading input or
// shuffling. Padding writes only zero scales. There is no shared memory or barrier.
// Inputs are aligned BF16[6144,T]; caller-owned outputs are disjoint and 256-byte aligned.
// The validated 6144*T < 2^31 extent bounds every pack index. Optional BF16 diagnostic
// publishes exactly the represented value being encoded.
template <bool Diagnostic>
__global__
__launch_bounds__(256) void sigmoid_a4_kernel(const Bf16x8Pack* gate, const Bf16x8Pack* input,
                                              Bf16x8Pack* normalized, std::uint32_t* codes,
                                              std::uint8_t* scales, float divisor, int tokens) {
    const int task  = static_cast<int>(blockIdx.x) * 256 + static_cast<int>(threadIdx.x);
    const int token = task / 768;
    const int group = (task % 768) / 2;
    if (token >= tokens) {
        if ((task & 1) == 0) {
            const std::int64_t tile = static_cast<std::int64_t>(token / 256) * 24 + group / 16;
            scales[tile * 4096 + (token % 256) * 16 + group % 16] = 0;
        }
        return;
    }
    const auto g = load_vec<Bf16x8Pack>(gate + task);
    auto value   = load_vec<Bf16x8Pack>(input + task);
    float2 represented[4];
    float maximum = 0;
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        const float2 gv   = __bfloat1622float2(g.pair[pair]);
        const float2 xv   = __bfloat1622float2(value.pair[pair]);
        value.pair[pair]  = __floats2bfloat162_rn(xv.x * sigmoid(gv.x), xv.y * sigmoid(gv.y));
        represented[pair] = __bfloat1622float2(value.pair[pair]);
        maximum = fmaxf(maximum, fmaxf(fabsf(represented[pair].x), fabsf(represented[pair].y)));
    }
    if constexpr (Diagnostic) { store_vec(normalized + task, value); }
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffffU, maximum, 1, 2));
    const auto scale =
        __nv_cvt_float_to_fp8(__fdiv_rn(divisor * maximum, 6.0F), __NV_SATFINITE, __NV_E4M3);
    std::uint32_t packed = 0;
    if (scale != 0) {
        const float decoded = decode_nvfp4_e4m3(scale);
#pragma unroll
        for (int pair = 0; pair < 4; ++pair) {
            represented[pair].x = __fdiv_rn(represented[pair].x * divisor, decoded);
            represented[pair].y = __fdiv_rn(represented[pair].y * divisor, decoded);
        }
        asm volatile("{ .reg .b8 b0,b1,b2,b3; "
                     "cvt.rn.satfinite.e2m1x2.f32 b0,%2,%1; "
                     "cvt.rn.satfinite.e2m1x2.f32 b1,%4,%3; "
                     "cvt.rn.satfinite.e2m1x2.f32 b2,%6,%5; "
                     "cvt.rn.satfinite.e2m1x2.f32 b3,%8,%7; "
                     "mov.b32 %0,{b0,b1,b2,b3}; }"
                     : "=r"(packed)
                     : "f"(represented[0].x), "f"(represented[0].y), "f"(represented[1].x),
                       "f"(represented[1].y), "f"(represented[2].x), "f"(represented[2].y),
                       "f"(represented[3].x), "f"(represented[3].y));
    }
    codes[task] = packed;
    if ((task & 1) == 0) {
        const std::int64_t tile = static_cast<std::int64_t>(token / 256) * 24 + group / 16;
        scales[tile * 4096 + (token % 256) * 16 + group % 16] = scale;
    }
}

} // namespace

void sigmoid_mul_a4_launch(const Tensor& gate, const Tensor& input, Tensor* normalized,
                           A4Activation& output, cudaStream_t stream) {
    const auto* g    = static_cast<const Bf16x8Pack*>(gate.data);
    const auto* x    = static_cast<const Bf16x8Pack*>(input.data);
    auto* codes      = static_cast<std::uint32_t*>(output.codes.data);
    auto* scales     = static_cast<std::uint8_t*>(output.scales.data);
    const int blocks = ((output.tokens + 255) / 256) * 768;
    if (normalized != nullptr) {
        sigmoid_a4_kernel<true>
            <<<blocks, 256, 0, stream>>>(g, x, static_cast<Bf16x8Pack*>(normalized->data), codes,
                                         scales, output.divisor, output.tokens);
    } else {
        sigmoid_a4_kernel<false><<<blocks, 256, 0, stream>>>(g, x, nullptr, codes, scales,
                                                             output.divisor, output.tokens);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
