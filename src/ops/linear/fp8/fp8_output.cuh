#pragma once

#include "ops/common/memory.cuh"
#include "ops/common/math.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

struct Fp8TensorMultiplier {
    const float* values;
};

__device__ __forceinline__ float fp8_weight_scale(const __nv_bfloat16* scales, int row) {
    return __bfloat162float(scales[row]);
}
__device__ __forceinline__ float fp8_weight_scale(Fp8TensorMultiplier scales, int) {
    return scales.values[0];
}
__device__ __forceinline__ float2 fp8_weight_scale_pair(const __nv_bfloat16* scales, int row) {
    return bf16x2_bits_to_float2(load_vec<std::uint32_t>(scales + row));
}
__device__ __forceinline__ float2 fp8_weight_scale_pair(Fp8TensorMultiplier scales, int) {
    const float scale = scales.values[0];
    return make_float2(scale, scale);
}

struct Fp8IdentityEpilogue {
    __device__ __forceinline__ float apply(std::int32_t, std::int32_t, float value) const {
        return value;
    }
};

struct Fp8ContiguousOutput {
    __nv_bfloat16* data;
    std::int32_t rows;

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t token,
                                          float value) const {
        data[static_cast<std::int64_t>(token) * rows + parent_row] = __float2bfloat16_rn(value);
    }

    __device__ __forceinline__ void store_vector(std::int32_t parent_row, std::int32_t token,
                                                 uint4 values) const {
        auto* destination = data + static_cast<std::int64_t>(token) * rows + parent_row;
        store_vec(destination, values);
    }
};

// Private normalized-key projection output; never round through BF16 first.
struct Fp8ContiguousF32Output {
    float* data;
    std::int32_t rows;
    __device__ __forceinline__ void store(std::int32_t row,std::int32_t token,float value) const {
        data[static_cast<std::int64_t>(token)*rows+row]=value;
    }
};

} // namespace ninfer::ops::detail
