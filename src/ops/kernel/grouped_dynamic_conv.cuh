#pragma once

// Implements: include/ninfer/ops/grouped_dynamic_conv.h
// Match: D=5120, G=320, group_size=16, kernel=2, contiguous BF16 activations.

#include "ops/common/math.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kGroupedDynamicConvGemmBlock  = 256;
inline constexpr int kGroupedDynamicConvConvBlock  = 256;

__device__ __forceinline__ float grouped_dynamic_conv_block_sum(float value) {
    __shared__ float partial[kGroupedDynamicConvGemmBlock];
    partial[threadIdx.x] = value;
    __syncthreads();
    for (int stride = kGroupedDynamicConvGemmBlock / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) { partial[threadIdx.x] += partial[threadIdx.x + stride]; }
        __syncthreads();
    }
    return partial[0];
}

// out is contiguous BF16 [N, columns] with K-fastest activations [K, columns].
__global__ void grouped_dynamic_conv_bf16_gemv_kernel(const __nv_bfloat16* x,
                                                      const __nv_bfloat16* weight,
                                                      __nv_bfloat16* out, std::int32_t n_rows,
                                                      std::int32_t k_rows) {
    const std::int32_t row = static_cast<std::int32_t>(blockIdx.x);
    const std::int32_t col = static_cast<std::int32_t>(blockIdx.y);
    if (row >= n_rows) { return; }

    const std::int64_t k64   = static_cast<std::int64_t>(k_rows);
    const __nv_bfloat16* w   = weight + static_cast<std::int64_t>(row) * k64;
    const __nv_bfloat16* col_x = x + static_cast<std::int64_t>(col) * k64;
    float acc                = 0.0f;
    for (std::int32_t k = static_cast<std::int32_t>(threadIdx.x); k < k_rows;
         k += kGroupedDynamicConvGemmBlock) {
        acc += __bfloat162float(w[k]) * __bfloat162float(col_x[k]);
    }
    acc = grouped_dynamic_conv_block_sum(acc);
    if (threadIdx.x == 0) {
        out[static_cast<std::int64_t>(col) * n_rows + row] = __float2bfloat16_rn(acc);
    }
}

template <bool kPrepare>
__global__ void grouped_dynamic_conv_kernel(const __nv_bfloat16* hidden,
                                            const __nv_bfloat16* base_kernel,
                                            const __nv_bfloat16* dynamic_src, __nv_bfloat16* out,
                                            __nv_bfloat16* finish_dynamic, std::int32_t tokens,
                                            std::int32_t batch) {
    constexpr std::int32_t kD      = 5120;
    constexpr std::int32_t kGroups = 320;
    constexpr std::int32_t kGroup  = 16;
    constexpr std::int32_t kProj   = 1280;

    const std::int32_t d   = static_cast<std::int32_t>(blockIdx.x) * blockDim.x +
                           static_cast<std::int32_t>(threadIdx.x);
    const std::int32_t col = static_cast<std::int32_t>(blockIdx.y);
    const std::int32_t columns = tokens * batch;
    if (d >= kD || col >= columns) { return; }

    const std::int32_t t     = col % tokens;
    const std::int32_t group = d / kGroup;
    const std::int64_t hid   = static_cast<std::int64_t>(col) * kD + d;
    const float x0           = __bfloat162float(hidden[hid]);
    const float x1           = (t == 0)
                                   ? 0.0f
                                   : __bfloat162float(hidden[hid - static_cast<std::int64_t>(kD)]);

    const std::int32_t phase = kPrepare ? 0 : 1;
    const float base0 =
        __bfloat162float(base_kernel[static_cast<std::int64_t>(phase) * (kD * 2) + d]);
    const float base1 =
        __bfloat162float(base_kernel[static_cast<std::int64_t>(phase) * (kD * 2) + kD + d]);

    float dyn0 = 0.0f;
    float dyn1 = 0.0f;
    if constexpr (kPrepare) {
        const std::int64_t proj_col = static_cast<std::int64_t>(col) * kProj;
        dyn0 = __bfloat162float(dynamic_src[proj_col + group]);
        dyn1 = __bfloat162float(dynamic_src[proj_col + kGroups + group]);
        if ((d % kGroup) == 0) {
            finish_dynamic[static_cast<std::int64_t>(col) * (kGroups * 2) + group] =
                dynamic_src[proj_col + 2 * kGroups + group];
            finish_dynamic[static_cast<std::int64_t>(col) * (kGroups * 2) + kGroups + group] =
                dynamic_src[proj_col + 3 * kGroups + group];
        }
    } else {
        const std::int64_t dyn_col = static_cast<std::int64_t>(col) * (kGroups * 2);
        dyn0                       = __bfloat162float(dynamic_src[dyn_col + group]);
        dyn1                       = __bfloat162float(dynamic_src[dyn_col + kGroups + group]);
    }

    out[hid] = __float2bfloat16_rn((base0 + dyn0) * x0 + (base1 + dyn1) * x1);
}

} // namespace ninfer::ops
