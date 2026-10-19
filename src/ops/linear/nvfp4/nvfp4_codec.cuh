#pragma once

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace ninfer::ops::detail {

__device__ __forceinline__ float2 decode_nvfp4_e2m1x2(std::uint8_t storage) {
    __nv_fp4x2_e2m1 value;
    value.__x = storage;
    return static_cast<float2>(value);
}

__device__ __forceinline__ float decode_nvfp4_e4m3(std::uint8_t storage) {
    __nv_fp8x2_e4m3 value;
    value.__x = static_cast<std::uint16_t>(storage) | (static_cast<std::uint16_t>(storage) << 8);
    return static_cast<float2>(value).x;
}

struct alignas(8) Nvfp4QuantizedK16 {
    std::uint32_t codes_lo;
    std::uint32_t codes_hi;
    std::uint8_t scale;
};

static_assert(alignof(Nvfp4QuantizedK16) == 8);

__device__ __forceinline__ void
pack_nvfp4_e2m1x16(const float2 (&values)[8], std::uint32_t& codes_lo, std::uint32_t& codes_hi) {
    asm volatile("{\n"
                 ".reg .b8 b0;\n"
                 ".reg .b8 b1;\n"
                 ".reg .b8 b2;\n"
                 ".reg .b8 b3;\n"
                 ".reg .b8 b4;\n"
                 ".reg .b8 b5;\n"
                 ".reg .b8 b6;\n"
                 ".reg .b8 b7;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b0, %3, %2;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b1, %5, %4;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b2, %7, %6;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b3, %9, %8;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b4, %11, %10;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b5, %13, %12;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b6, %15, %14;\n"
                 "cvt.rn.satfinite.e2m1x2.f32 b7, %17, %16;\n"
                 "mov.b32 %0, {b0,b1,b2,b3};\n"
                 "mov.b32 %1, {b4,b5,b6,b7};\n"
                 "}\n"
                 : "=r"(codes_lo), "=r"(codes_hi)
                 : "f"(values[0].x), "f"(values[0].y), "f"(values[1].x), "f"(values[1].y),
                   "f"(values[2].x), "f"(values[2].y), "f"(values[3].x), "f"(values[3].y),
                   "f"(values[4].x), "f"(values[4].y), "f"(values[5].x), "f"(values[5].y),
                   "f"(values[6].x), "f"(values[6].y), "f"(values[7].x), "f"(values[7].y));
}

// Quantizes 16 BF16 values held as eight BF16x2 words (element 2i in the low half of word i).
__device__ __forceinline__ Nvfp4QuantizedK16
quantize_nvfp4_k16_bits(const std::uint32_t (&represented)[8], float input_scale_divisor) {
    float2 values[8];
    float max_abs = 0.0F;
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        values[pair] = bf16x2_bits_to_float2(represented[pair]);
        max_abs      = fmaxf(max_abs, fabsf(values[pair].x));
        max_abs      = fmaxf(max_abs, fabsf(values[pair].y));
    }

    Nvfp4QuantizedK16 result{};
    const float scale_unencoded = __fdiv_rn(input_scale_divisor * max_abs, 6.0F);
    result.scale                = __nv_cvt_float_to_fp8(scale_unencoded, __NV_SATFINITE, __NV_E4M3);
    if (result.scale == 0) { return result; }

    const float decoded_scale = decode_nvfp4_e4m3(result.scale);
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        values[pair].x = __fdiv_rn(values[pair].x * input_scale_divisor, decoded_scale);
        values[pair].y = __fdiv_rn(values[pair].y * input_scale_divisor, decoded_scale);
    }
    pack_nvfp4_e2m1x16(values, result.codes_lo, result.codes_hi);
    return result;
}

__device__ __forceinline__ Nvfp4QuantizedK16 quantize_nvfp4_k16(const __nv_bfloat16* source,
                                                                float input_scale_divisor) {
    const uint4 packed0                = load_vec<uint4>(source);
    const uint4 packed1                = load_vec<uint4>(source + 8);
    const std::uint32_t represented[8] = {
        packed0.x, packed0.y, packed0.z, packed0.w, packed1.x, packed1.y, packed1.z, packed1.w,
    };
    return quantize_nvfp4_k16_bits(represented, input_scale_divisor);
}

// One BF16 pair of a 16-value NVFP4 group whose eight pairs sit in eight consecutive lanes
// 8j..8j+7 of a warp, in group order. Returns the pair's code byte (first value in the low
// nibble); every lane of the group receives the group's E4M3 scale. The arithmetic is
// quantize_nvfp4_k16's, so codes and scale are bit-identical to it. All 32 lanes of the warp
// must call it together (the amax uses three xor shuffles inside each eight-lane group).
struct Nvfp4QuantizedPair {
    std::uint8_t code;
    std::uint8_t scale;
};

__device__ __forceinline__ Nvfp4QuantizedPair
quantize_nvfp4_pair_group8(const __nv_bfloat162& pair, float input_scale_divisor) {
    float2 value = __bfloat1622float2(pair);
    // Seeded with 0 like quantize_nvfp4_k16, so an all-NaN group also encodes scale 0.
    float max_abs = fmaxf(fmaxf(0.0F, fabsf(value.x)), fabsf(value.y));
#pragma unroll
    for (int offset = 1; offset < 8; offset <<= 1) {
        max_abs = fmaxf(max_abs, __shfl_xor_sync(0xffffffffU, max_abs, offset));
    }
    Nvfp4QuantizedPair result{};
    const float scale_unencoded = __fdiv_rn(input_scale_divisor * max_abs, 6.0F);
    result.scale                = __nv_cvt_float_to_fp8(scale_unencoded, __NV_SATFINITE, __NV_E4M3);
    if (result.scale == 0) { return result; }
    const float decoded_scale = decode_nvfp4_e4m3(result.scale);
    value.x                   = __fdiv_rn(value.x * input_scale_divisor, decoded_scale);
    value.y                   = __fdiv_rn(value.y * input_scale_divisor, decoded_scale);
    std::uint16_t code        = 0;
    asm("{\n"
        ".reg .b8 b;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b, %2, %1;\n"
        "cvt.u16.u8 %0, b;\n"
        "}\n"
        : "=h"(code)
        : "f"(value.x), "f"(value.y));
    result.code = static_cast<std::uint8_t>(code);
    return result;
}

__device__ __forceinline__ Nvfp4QuantizedK16 quantize_nvfp4_f32x16(const float* source) {
    float2 values[8];
    float max_abs = 0.0F;
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        values[pair] = make_float2(source[2 * pair], source[2 * pair + 1]);
        max_abs      = fmaxf(max_abs, fabsf(values[pair].x));
        max_abs      = fmaxf(max_abs, fabsf(values[pair].y));
    }

    Nvfp4QuantizedK16 result{};
    const float scale_unencoded = __fdiv_rn(max_abs, 6.0F);
    result.scale                = __nv_cvt_float_to_fp8(scale_unencoded, __NV_SATFINITE, __NV_E4M3);
    if (result.scale == 0) { return result; }

    const float decoded_scale = decode_nvfp4_e4m3(result.scale);
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        values[pair].x = __fdiv_rn(values[pair].x, decoded_scale);
        values[pair].y = __fdiv_rn(values[pair].y, decoded_scale);
    }
    pack_nvfp4_e2m1x16(values, result.codes_lo, result.codes_hi);
    return result;
}

} // namespace ninfer::ops::detail
