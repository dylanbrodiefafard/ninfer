#pragma once

// Implements: include/ninfer/ops/stream_mean.h
// Match: D=2560, four streams, BF16, 16-byte aligned residual and output columns.

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kStreamMeanVector  = 8; // BF16 channels per 16-byte access
inline constexpr int kStreamMeanThreads = 2560 / kStreamMeanVector;

// Grid: one CTA per token t (grid.x = T); block: 320 threads, thread j owning channels
// [8j, 8j+8). Each thread reads its 16-byte chunk of the four streams of token t, sums them in
// FP32, scales by 1/4 exactly, and stores one 16-byte BF16 chunk at out + t * out_stride_elems.
// No shared memory or synchronization. 64-bit token offsets admit any positive T.
__global__ void __launch_bounds__(kStreamMeanThreads)
    stream_mean_kernel(const uint4* __restrict__ residual, __nv_bfloat16* __restrict__ out,
                       std::int64_t out_stride_elems) {
    constexpr int kHidden        = 2560;
    constexpr int kStreams       = 4;
    constexpr int kChunks        = kHidden / kStreamMeanVector;
    const std::int64_t token     = blockIdx.x;
    const int chunk              = static_cast<int>(threadIdx.x);
    const uint4* source          = residual + token * (kStreams * kChunks) + chunk;
    float sum[kStreamMeanVector] = {};
#pragma unroll
    for (int stream = 0; stream < kStreams; ++stream) {
        const uint4 packed = source[stream * kChunks];
        const auto* pairs  = reinterpret_cast<const __nv_bfloat162*>(&packed);
#pragma unroll
        for (int pair = 0; pair < kStreamMeanVector / 2; ++pair) {
            const float2 value = __bfloat1622float2(pairs[pair]);
            sum[2 * pair] += value.x;
            sum[2 * pair + 1] += value.y;
        }
    }
    uint4 result;
    auto* result_pairs = reinterpret_cast<__nv_bfloat162*>(&result);
#pragma unroll
    for (int pair = 0; pair < kStreamMeanVector / 2; ++pair) {
        result_pairs[pair] =
            __floats2bfloat162_rn(0.25F * sum[2 * pair], 0.25F * sum[2 * pair + 1]);
    }
    *reinterpret_cast<uint4*>(out + token * out_stride_elems + chunk * kStreamMeanVector) = result;
}

} // namespace ninfer::ops
