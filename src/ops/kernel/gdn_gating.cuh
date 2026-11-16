#pragma once

// ninfer::ops - gdn_gating kernel: elementwise GDN gate prep over [48,T].
// Transcendentals use fp32 CUDA math functions, not polynomial approximations.

#include "ops/common/math.cuh"

#include <cuda_bf16.h>

#include <cmath>
#include <cstdint>

namespace ninfer::ops {

__device__ __forceinline__ float gdn_gate_input(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

__device__ __forceinline__ float gdn_gate_input(float value) { return value; }

// One thread per (h,t) output element, grid-stride. Element i = 48 t + h reads a and b at
// t * input_stride + h, so a/b are either separate [48,T] tensors (input_stride 48) or the two
// row blocks of one packed [96,T] projection (input_stride 96, b = a + 48).
template <class Input>
__global__ void gdn_gating_kernel(const Input* a, const Input* b, std::int32_t input_stride,
                                  const float* A_log, const float* dt_bias, float* g, float* beta,
                                  std::int64_t n) {
    const std::int64_t start  = blockIdx.x * static_cast<std::int64_t>(blockDim.x) + threadIdx.x;
    const std::int64_t stride = static_cast<std::int64_t>(gridDim.x) * blockDim.x;
    for (std::int64_t i = start; i < n; i += stride) {
        const int h              = static_cast<int>(i % 48);
        const std::int64_t input = (i / 48) * input_stride + h;
        const float av           = gdn_gate_input(a[input]);
        const float bv           = gdn_gate_input(b[input]);
        const float sp           = softplus(av + dt_bias[h]);
        g[i]                     = -expf(A_log[h]) * sp;
        beta[i]                  = sigmoid(bv);
    }
}

} // namespace ninfer::ops
