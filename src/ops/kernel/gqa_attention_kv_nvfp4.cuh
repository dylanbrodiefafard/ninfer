#pragma once

#include "ops/common/math.cuh"
#include "ops/kernel/paged_kv_address.cuh"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kGqaNvfp4HeadDim      = 256;
inline constexpr int kGqaNvfp4Group        = 16;
inline constexpr int kGqaNvfp4Groups       = kGqaNvfp4HeadDim / kGqaNvfp4Group;
inline constexpr int kGqaNvfp4CodeWidth    = kGqaNvfp4HeadDim / 2;
inline constexpr int kGqaNvfp4K64          = kGqaNvfp4HeadDim / 64;
inline constexpr int kGqaNvfp4CodeRowBytes = kGqaNvfp4CodeWidth;

template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_nvfp4_code_index(int physical_page, int kv_head,
                                                             int code_byte, int page_offset) {
    return paged_kv_element_offset<kGqaNvfp4CodeWidth, Geometry::KVHeads>(
        physical_page, kv_head, page_offset, code_byte);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_nvfp4_scale_index(int physical_page, int kv_head,
                                                              int group, int page_offset) {
    return paged_kv_element_offset<kGqaNvfp4Groups, Geometry::KVHeads>(physical_page, kv_head,
                                                                       page_offset, group);
}

__device__ __forceinline__ int gqa_nvfp4_swizzle_byte(int row, int logical_byte) {
    const int logical_segment  = logical_byte >> 4;
    const int byte_in_segment  = logical_byte & 15;
    const int physical_segment = logical_segment ^ (row & 7);
    return physical_segment * 16 + byte_in_segment;
}

__device__ __forceinline__ int4 gqa_nvfp4_dequant_bf16x8(const std::uint8_t* codes4, float scale) {
    unsigned packed[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 pair = detail::decode_nvfp4_e2m1x2(codes4[i]);
        packed[i]         = pack_bf16x2(pair.x * scale, pair.y * scale);
    }
    return make_int4(static_cast<int>(packed[0]), static_cast<int>(packed[1]),
                     static_cast<int>(packed[2]), static_cast<int>(packed[3]));
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_nvfp4_src_index(int kv_head, int d, int token) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kGqaNvfp4HeadDim) *
               (static_cast<std::int64_t>(kv_head) +
                static_cast<std::int64_t>(Geometry::KVHeads) * token);
}

__device__ __forceinline__ void gqa_nvfp4_quantize_bf16x16(const __nv_bfloat16* source,
                                                           std::uint32_t& codes_lo,
                                                           std::uint32_t& codes_hi,
                                                           std::uint8_t& scale) {
    const detail::Nvfp4QuantizedK16 packed = detail::quantize_nvfp4_k16(source, 1.0F);
    codes_lo                               = packed.codes_lo;
    codes_hi                               = packed.codes_hi;
    scale                                  = packed.scale;
}

} // namespace ninfer::ops
