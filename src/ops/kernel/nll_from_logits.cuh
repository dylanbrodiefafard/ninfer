#pragma once

#include <cuda_bf16.h>
#include <cstdint>
#include <math_constants.h>

namespace ninfer::ops {

inline constexpr int kNllFromLogitsBlock = 256;

__device__ __forceinline__ float nll_warp_max(float value) {
    constexpr unsigned int kMask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(kMask, value, offset));
    }
    return value;
}

__device__ __forceinline__ float nll_warp_sum(float value) {
    constexpr unsigned int kMask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(kMask, value, offset);
    }
    return value;
}

__launch_bounds__(kNllFromLogitsBlock) __global__
    void nll_from_logits_kernel(const __nv_bfloat16* logits, const std::int32_t* targets,
                                float* out, std::int32_t valid_rows, std::int32_t physical_rows) {
    const std::int32_t t    = static_cast<std::int32_t>(blockIdx.x);
    const std::int64_t base = static_cast<std::int64_t>(t) * physical_rows;
    const int lane          = threadIdx.x & 31;
    const int warp          = threadIdx.x >> 5;
    const int warps         = blockDim.x >> 5;

    float thread_max = -CUDART_INF_F;
    for (std::int32_t v = static_cast<std::int32_t>(threadIdx.x); v < valid_rows; v += blockDim.x) {
        thread_max = fmaxf(thread_max, __bfloat162float(logits[base + v]));
    }
    thread_max = nll_warp_max(thread_max);

    __shared__ float warp_max[8];
    __shared__ float warp_sum[8];
    if (lane == 0) { warp_max[warp] = thread_max; }
    __syncthreads();
    float col_max = (lane < warps) ? warp_max[lane] : -CUDART_INF_F;
    if (warp == 0) { col_max = nll_warp_max(col_max); }
    if (threadIdx.x == 0) { warp_max[0] = col_max; }
    __syncthreads();
    col_max = warp_max[0];

    float thread_sum = 0.0f;
    for (std::int32_t v = static_cast<std::int32_t>(threadIdx.x); v < valid_rows; v += blockDim.x) {
        thread_sum += expf(__bfloat162float(logits[base + v]) - col_max);
    }
    thread_sum = nll_warp_sum(thread_sum);
    if (lane == 0) { warp_sum[warp] = thread_sum; }
    __syncthreads();
    float col_sum = (lane < warps) ? warp_sum[lane] : 0.0f;
    if (warp == 0) { col_sum = nll_warp_sum(col_sum); }
    if (threadIdx.x == 0) {
        const std::int32_t target = targets[t];
        const float target_logit  = __bfloat162float(logits[base + target]);
        out[t]                    = col_max + logf(col_sum) - target_logit;
    }
}

} // namespace ninfer::ops
