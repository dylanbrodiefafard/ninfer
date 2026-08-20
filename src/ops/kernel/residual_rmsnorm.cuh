#pragma once

// ninfer::ops - fused residual + RMSNorm kernel over contiguous BF16 rows.
//
// One kernel replaces the `residual_add` + `rmsnorm` launch pair:
//   x_new = x + y                 (in-place residual update, identical to residual_add)
//   out   = rmsnorm(x_new) * (1 + weight)   (the per-layer Offset epilogue, identical to rmsnorm)
//
// Bit-exact against the unfused (residual_add + rmsnorm) pair: x_new is computed in-register
// (the same BF16 the residual_add stores, then written in-place) and the per-row RMS uses the
// same block_reduce_sum<Block> reduce the standalone rmsnorm uses, so `out` is byte-identical.
// `x` is updated in-place (the running residual) and `out` holds the normalized hidden for the
// downstream GEMV.

#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

// CTA-per-row fused residual + RMSNorm. One CTA owns one row of `rows`. Mirrors
// rmsnorm_cta_bf16x2_kernel's geometry + reduce (so the RMS is bit-exact with the standalone
// rmsnorm) with the residual_add folded in: each pair reads x (residual) + y (the mixer/MLP
// delta), computes x_new = x + y, stores it in-place, and accumulates the per-row sum of
// squares of x_new. The per-row RMS (the same block_reduce_sum<Block> the rmsnorm uses) then
// scales x_new by (1 + weight) (the Offset epilogue, the per-layer unit_offset norm) into `out`.
template <int Block, int MaxPairsPerThread>
__launch_bounds__(Block) __global__
    void residual_rmsnorm_cta_bf16x2_kernel(const __nv_bfloat162* y, __nv_bfloat162* x,
                                             const __nv_bfloat162* weight, __nv_bfloat162* out,
                                             std::int32_t d, std::int64_t rows, float eps) {
    static_assert(Block % kWarpSize == 0);
    const std::int64_t row            = static_cast<std::int64_t>(blockIdx.x);
    if (row >= rows) { return; }

    const int pairs            = d / 2;
    const int pairs_per_thread = pairs / Block;
    const std::int64_t row_base = row * static_cast<std::int64_t>(pairs);
    __nv_bfloat162 values[MaxPairsPerThread];
    float sum = 0.0f;

#pragma unroll
    for (int k = 0; k < MaxPairsPerThread; ++k) {
        if (k < pairs_per_thread) {
            const int pair = static_cast<int>(threadIdx.x) + k * Block;
            // x_new = x + y (the residual update, in-place), identical to residual_add.
            const float2 xf = __bfloat1622float2(x[row_base + pair]);
            const float2 yf = __bfloat1622float2(y[row_base + pair]);
            values[k] = __floats2bfloat162_rn(xf.x + yf.x, xf.y + yf.y);
            x[row_base + pair] = values[k];  // in-place residual update (the residual_add)
            const float2 xn = __bfloat1622float2(values[k]);
            sum += xn.x * xn.x + xn.y * xn.y;
        }
    }

    __shared__ float warp_sums[Block / kWarpSize];
    __shared__ float inv_shared;
    const float block_sum = block_reduce_sum<Block>(sum, warp_sums);
    if (threadIdx.x == 0) {
        inv_shared = rsqrtf(block_sum / static_cast<float>(d) + eps);
    }
    __syncthreads();
    const float inv = inv_shared;

#pragma unroll
    for (int k = 0; k < MaxPairsPerThread; ++k) {
        if (k < pairs_per_thread) {
            const int pair = static_cast<int>(threadIdx.x) + k * Block;
            const float2 xn = __bfloat1622float2(values[k]);
            const float2 wf = __bfloat1622float2(weight[pair]);
            // The per-layer Offset epilogue (unit_offset): out = x_new * inv * (1 + weight).
            out[row_base + pair] =
                __floats2bfloat162_rn(xn.x * inv * (wf.x + 1.0f), xn.y * inv * (wf.y + 1.0f));
        }
    }
}

} // namespace ninfer::ops