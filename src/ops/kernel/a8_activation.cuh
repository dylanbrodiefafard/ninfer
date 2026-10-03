#pragma once

// ninfer::ops - device side of the row-scaled E4M3 activation codec (include/ninfer/ops/
// a8_activation.h). Producers that hold one activation column per CTA in registers use these
// helpers to publish the column's scale and codes without a separate quantize launch. The
// arithmetic is the projections' quantizer's (src/ops/linear/fp8/fp8_a8.cu): an order-independent
// max, FP32 max/448, FP32 reciprocal, saturating round-to-nearest E4M3 conversion.

#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace ninfer::ops {

// Block-wide column scale. Every thread of the CTA passes the maximum |value| over the elements
// it holds; returns max/448, or 0 for an all-zero column. `warp_maxima` is shared storage for
// Block/32 floats and `column_scale` one shared float. Contains two __syncthreads(), so every
// thread of the CTA must call it.
template <int Block>
__device__ __forceinline__ float a8_block_column_scale(float local_maximum, float* warp_maxima,
                                                       float* column_scale) {
    static_assert(Block % kWarpSize == 0 && Block / kWarpSize <= kWarpSize);
    constexpr int kWarps = Block / kWarpSize;
    const int lane       = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp       = static_cast<int>(threadIdx.x) / kWarpSize;
    float maximum        = warp_max(local_maximum);
    if (lane == 0) { warp_maxima[warp] = maximum; }
    __syncthreads();
    if (warp == 0) {
        maximum = warp_max(lane < kWarps ? warp_maxima[lane] : 0.0F);
        if (lane == 0) { *column_scale = maximum > 0.0F ? maximum / 448.0F : 0.0F; }
    }
    __syncthreads();
    return *column_scale;
}

__device__ __forceinline__ float a8_inverse_scale(float scale) {
    return scale > 0.0F ? 1.0F / scale : 0.0F;
}

__device__ __forceinline__ float a8_pair_maximum(const __nv_bfloat162& values) {
    const float2 v = __bfloat1622float2(values);
    return fmaxf(fabsf(v.x), fabsf(v.y));
}

// Two adjacent E4M3 codes, low byte first, for an adjacent BF16 pair.
__device__ __forceinline__ std::uint16_t a8_encode_pair(const __nv_bfloat162& values,
                                                        float inverse) {
    const float2 v = __bfloat1622float2(values);
    return __nv_cvt_float2_to_fp8x2(make_float2(v.x * inverse, v.y * inverse), __NV_SATFINITE,
                                    __NV_E4M3);
}

} // namespace ninfer::ops
