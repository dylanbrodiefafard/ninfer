#pragma once

#include "ops/ggml_block_linear/ggml_codebook_data.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

__device__ __forceinline__ std::uint16_t load_u16_le(const std::uint8_t* bytes) {
    return static_cast<std::uint16_t>(bytes[0]) |
           (static_cast<std::uint16_t>(bytes[1]) << 8U);
}

__device__ __forceinline__ std::uint32_t load_u32_le(const std::uint8_t* bytes) {
    return static_cast<std::uint32_t>(bytes[0]) |
           (static_cast<std::uint32_t>(bytes[1]) << 8U) |
           (static_cast<std::uint32_t>(bytes[2]) << 16U) |
           (static_cast<std::uint32_t>(bytes[3]) << 24U);
}

__device__ __forceinline__ float load_f16_le(const std::uint8_t* bytes) {
    return __half2float(__ushort_as_half(load_u16_le(bytes)));
}

__device__ __forceinline__ int signed_i8(std::uint8_t word) {
    return word < 128U ? static_cast<int>(word) : static_cast<int>(word) - 256;
}

__device__ __forceinline__ void qk_scale_min(const std::uint8_t* packed, int group,
                                              int& scale, int& minimum) {
    if (group < 4) {
        scale   = packed[group] & 0x3fU;
        minimum = packed[group + 4] & 0x3fU;
        return;
    }
    scale = (packed[group + 4] & 0x0fU) | ((packed[group - 4] >> 6U) << 4U);
    minimum = (packed[group + 4] >> 4U) | ((packed[group] >> 6U) << 4U);
}

template <QType Format>
struct GgmlDecoder;

template <>
struct GgmlDecoder<QType::GGML_Q8_0> {
    static constexpr int block_values = 32;
    static constexpr int block_bytes  = 34;

    __device__ static float value(const std::uint8_t* block, int index) {
        return load_f16_le(block) * static_cast<float>(signed_i8(block[2 + index]));
    }
};

template <>
struct GgmlDecoder<QType::GGML_Q4_K> {
    static constexpr int block_values = 256;
    static constexpr int block_bytes  = 144;

    __device__ static float value(const std::uint8_t* block, int index) {
        const int group = index >> 5;
        const int lane  = index & 31;
        int scale;
        int minimum;
        qk_scale_min(block + 4, group, scale, minimum);
        const int shift = 4 * (group & 1);
        const int code  = (block[16 + 32 * (group >> 1) + lane] >> shift) & 0x0fU;
        return load_f16_le(block) * static_cast<float>(scale * code) -
               load_f16_le(block + 2) * static_cast<float>(minimum);
    }
};

template <>
struct GgmlDecoder<QType::GGML_Q5_K> {
    static constexpr int block_values = 256;
    static constexpr int block_bytes  = 176;

    __device__ static float value(const std::uint8_t* block, int index) {
        const int group = index >> 5;
        const int lane  = index & 31;
        int scale;
        int minimum;
        qk_scale_min(block + 4, group, scale, minimum);
        const int shift = 4 * (group & 1);
        const int low   = (block[48 + 32 * (group >> 1) + lane] >> shift) & 0x0fU;
        const int high  = (block[16 + lane] >> group) & 1U;
        const int code  = low | (high << 4);
        return load_f16_le(block) * static_cast<float>(scale * code) -
               load_f16_le(block + 2) * static_cast<float>(minimum);
    }
};

template <>
struct GgmlDecoder<QType::GGML_Q6_K> {
    static constexpr int block_values = 256;
    static constexpr int block_bytes  = 210;

    __device__ static float value(const std::uint8_t* block, int index) {
        const int half    = index >> 7;
        const int within  = index & 127;
        const int group32 = within >> 5;
        const int lane    = within & 31;
        const int low_byte = 64 * half + lane + 32 * (group32 & 1);
        const int low = (block[low_byte] >> (4 * (group32 >> 1))) & 0x0fU;
        const int high = (block[128 + 32 * half + lane] >> (2 * group32)) & 0x03U;
        const int code  = (low | (high << 4)) - 32;
        const int group_scale = signed_i8(block[192 + (index >> 4)]);
        return load_f16_le(block + 208) * static_cast<float>(group_scale * code);
    }
};

template <>
struct GgmlDecoder<QType::GGML_IQ1_S> {
    static constexpr int block_values = 256;
    static constexpr int block_bytes  = 50;

    __device__ static float value(const std::uint8_t* block, int index) {
        const int group = index >> 5;
        const int lane8 = (index >> 3) & 3;
        const int item  = index & 7;
        const std::uint16_t control = load_u16_le(block + 34 + 2 * group);
        const int grid_index = block[2 + 4 * group + lane8] |
                               (((control >> (3 * lane8)) & 7U) << 8U);
        const int digit = (kIq1SPackedGrid[grid_index] >> (2 * item)) & 3U;
        const float grid_value = static_cast<float>(digit - 1);
        const float delta      = (control & 0x8000U) != 0 ? -0.125F : 0.125F;
        const int multiplier   = 2 * ((control >> 12U) & 7U) + 1;
        return load_f16_le(block) * static_cast<float>(multiplier) * (grid_value + delta);
    }
};

template <>
struct GgmlDecoder<QType::GGML_IQ2_XXS> {
    static constexpr int block_values = 256;
    static constexpr int block_bytes  = 66;

    __device__ static float value(const std::uint8_t* block, int index) {
        const int group = index >> 5;
        const int lane8 = (index >> 3) & 3;
        const int item  = index & 7;
        const std::uint8_t* words = block + 2 + 8 * group;
        const std::uint32_t low   = load_u32_le(words);
        const std::uint32_t high  = load_u32_le(words + 4);
        const int grid_index      = (low >> (8 * lane8)) & 0xffU;
        const int sign_index      = (high >> (7 * lane8)) & 0x7fU;
        const int parity          = __popc(static_cast<unsigned>(sign_index)) & 1;
        const int sign_bits       = sign_index | (parity << 7);
        const int digit = (kIq2XxsPackedGrid[grid_index] >> (2 * item)) & 3U;
        const int magnitude = digit == 0 ? 8 : (digit == 1 ? 25 : 43);
        const float group_scale =
            load_f16_le(block) * (0.5F + static_cast<float>(high >> 28U)) * 0.25F;
        return (sign_bits & (1 << item)) != 0
                   ? -group_scale * static_cast<float>(magnitude)
                   : group_scale * static_cast<float>(magnitude);
    }
};

template <>
struct GgmlDecoder<QType::GGML_IQ4_NL> {
    static constexpr int block_values = 32;
    static constexpr int block_bytes  = 18;

    __device__ static float value(const std::uint8_t* block, int index) {
        const std::uint8_t packed = block[2 + (index & 15)];
        const int code = index < 16 ? packed & 0x0fU : packed >> 4U;
        return load_f16_le(block) * static_cast<float>(kIq4NlGrid[code]);
    }
};

template <QType Format>
__global__ void ggml_block_linear_kernel(const __nv_bfloat16* x, const std::uint8_t* weights,
                                         __nv_bfloat16* out, std::int32_t k,
                                         std::uint64_t row_bytes) {
    using Decoder = GgmlDecoder<Format>;
    const auto* row = weights + static_cast<std::uint64_t>(blockIdx.x) * row_bytes;
    float partial   = 0.0F;
    for (int column = threadIdx.x; column < k; column += blockDim.x) {
        const int block_index = column / Decoder::block_values;
        const int block_item  = column % Decoder::block_values;
        const float weight =
            Decoder::value(row + static_cast<std::uint64_t>(block_index) * Decoder::block_bytes,
                           block_item);
        partial = fmaf(weight, __bfloat162float(x[column]), partial);
    }

    constexpr unsigned full_warp = 0xffffffffU;
    for (int offset = 16; offset > 0; offset >>= 1) {
        partial += __shfl_down_sync(full_warp, partial, offset);
    }

    __shared__ float warp_sums[8];
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    if (lane == 0) { warp_sums[warp] = partial; }
    __syncthreads();
    if (warp == 0) {
        float total = lane < 8 ? warp_sums[lane] : 0.0F;
        for (int offset = 16; offset > 0; offset >>= 1) {
            total += __shfl_down_sync(full_warp, total, offset);
        }
        if (lane == 0) { out[blockIdx.x] = __float2bfloat16_rn(total); }
    }
}

} // namespace ninfer::ops::detail
