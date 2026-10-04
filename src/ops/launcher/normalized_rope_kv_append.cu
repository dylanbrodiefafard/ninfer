#include "ops/launcher/normalized_rope_kv_append.h"

#include "core/device.h"
#include "ops/common/warp.cuh"
#include "ops/kernel/rope.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

// Grid x owns one accepted-prefix token, grid y one compact row. The eight 32-thread warps
// own its eight K heads; lane l owns split-half pairs 2l and 2l+1 (four features). Shared
// memory contains 64 sine/cosine coefficients reused by every head. The count predicate is
// uniform before the sole CTA barrier; warp reductions use all 32 active lanes. Inputs and
// cache are four-byte aligned, D/H=128/8, and distinct selected lanes and at most 2048
// sequential positions make all stores disjoint. Global offsets use 64 bits for padded caches.
// The private arithmetic profile rounds normalized pairs to BF16 in registers before rotation;
// it retains the qualified plain D128 norm reduction without materializing a global tensor.
__global__ void normalized_rope_kv_append_kernel(
    const __nv_bfloat162* __restrict__ k, const std::uint32_t* __restrict__ v,
    const __nv_bfloat162* __restrict__ gamma, const std::int32_t* __restrict__ positions,
    const std::int32_t* __restrict__ counts, const std::int32_t* __restrict__ lanes,
    __nv_bfloat162* __restrict__ cache_k, std::uint32_t* __restrict__ cache_v, int width,
    int padded_capacity, float epsilon, float theta) {
    const int token = static_cast<int>(blockIdx.x);
    const int batch = static_cast<int>(blockIdx.y);
    if (token >= counts[batch]) { return; }
    const std::int64_t column = static_cast<std::int64_t>(batch) * width + token;
    const int position        = positions[column];
    __shared__ float sine[64];
    __shared__ float cosine[64];
    if (threadIdx.x < 64) {
        const int pair = static_cast<int>(threadIdx.x);
        if (theta == 1.0e7F) {
            fixed_sincos<RopeKernelMode::DflashText1D>(positions + column, 1, 0, pair, &sine[pair],
                                                       &cosine[pair]);
        } else {
            constexpr double kInvTwoPi = 1.59154943091895336e-01;
            constexpr double kTwoPi    = 6.28318530717958648e+00;
            const double frequency     = pow(static_cast<double>(theta), -2.0 * pair / 128.0);
            const double angle         = static_cast<double>(position) * frequency;
            const float reduced = static_cast<float>(angle - nearbyint(angle * kInvTwoPi) * kTwoPi);
            sincosf(reduced, &sine[pair], &cosine[pair]);
        }
    }
    const int lane            = static_cast<int>(threadIdx.x) & 31;
    const int head            = static_cast<int>(threadIdx.x) >> 5;
    const std::int64_t source = (column * 8 + head) * 64 + lane;
    const float2 lo           = __bfloat1622float2(k[source]);
    const float2 hi           = __bfloat1622float2(k[source + 32]);
    float squares             = lo.x * lo.x + lo.y * lo.y + hi.x * hi.x + hi.y * hi.y;
    squares                   = warp_reduce_sum(squares);
    float inv                 = lane == 0 ? rsqrtf(squares * (1.0F / 128.0F) + epsilon) : 0.0F;
    inv                       = __shfl_sync(kFullWarpMask, inv, 0);
    const float2 gain_lo      = __bfloat1622float2(gamma[lane]);
    const float2 gain_hi      = __bfloat1622float2(gamma[lane + 32]);
    const float2 z_lo =
        __bfloat1622float2(__floats2bfloat162_rn(lo.x * inv * gain_lo.x, lo.y * inv * gain_lo.y));
    const float2 z_hi =
        __bfloat1622float2(__floats2bfloat162_rn(hi.x * inv * gain_hi.x, hi.y * inv * gain_hi.y));
    __syncthreads();
    const int pair = 2 * lane;
    const int slot = position & 2047;
    const std::int64_t destination =
        ((static_cast<std::int64_t>(lanes[batch]) * 8 + head) * padded_capacity + slot) * 64 + lane;
    cache_k[destination] =
        __floats2bfloat162_rn(z_lo.x * cosine[pair] - z_hi.x * sine[pair],
                              z_lo.y * cosine[pair + 1] - z_hi.y * sine[pair + 1]);
    cache_k[destination + 32] =
        __floats2bfloat162_rn(z_hi.x * cosine[pair] + z_lo.x * sine[pair],
                              z_hi.y * cosine[pair + 1] + z_lo.y * sine[pair + 1]);
    cache_v[destination]      = v[source];
    cache_v[destination + 32] = v[source + 32];
}

} // namespace

void normalized_rope_kv_append_launch(const Tensor& k, const Tensor& v, const Tensor& gamma,
                                      const Tensor& positions, const Tensor& counts,
                                      const Tensor& lanes, float epsilon, float theta,
                                      KVCacheAppendPrefixExecutionEnvelope envelope,
                                      CyclicKVCacheLayerView cache, cudaStream_t stream) {
    if (envelope.max_count == 0) { return; }
    const dim3 grid(envelope.max_count, static_cast<unsigned>(k.ne[3]), 1);
    normalized_rope_kv_append_kernel<<<grid, 256, 0, stream>>>(
        static_cast<const __nv_bfloat162*>(k.data), static_cast<const std::uint32_t*>(v.data),
        static_cast<const __nv_bfloat162*>(gamma.data),
        static_cast<const std::int32_t*>(positions.data),
        static_cast<const std::int32_t*>(counts.data), static_cast<const std::int32_t*>(lanes.data),
        static_cast<__nv_bfloat162*>(cache.k.data), static_cast<std::uint32_t*>(cache.v.data),
        k.ne[2], static_cast<int>(cache.padded_capacity), epsilon, theta);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
