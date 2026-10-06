#pragma once

// ninfer::ops - device side of the NVFP4 activation codec (include/ninfer/ops/a4_activation.h).
// Producers that hold an activation column's BF16 pairs in consecutive lanes publish its codes and
// tiled scales without a separate quantize launch; the arithmetic is the W4A4 projections'
// quantizer's (quantize_nvfp4_k16), so both are bit-identical.

#include "ninfer/ops/a4_activation.h"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops {

// Byte offset of scale (token, group) in the tiled scale plane of an activation with
// GroupsPerRow 16-value groups per column.
template <int GroupsPerRow>
__device__ __forceinline__ std::int64_t a4_scale_offset(int token, int group) {
    static_assert(GroupsPerRow % kA4ScaleTileGroups == 0);
    constexpr int kTilesPerRow = GroupsPerRow / kA4ScaleTileGroups;
    const std::int64_t tile = static_cast<std::int64_t>(token / kA4ScaleTileTokens) * kTilesPerRow +
                              group / kA4ScaleTileGroups;
    return tile * (kA4ScaleTileTokens * kA4ScaleTileGroups) +
           (token % kA4ScaleTileTokens) * kA4ScaleTileGroups + group % kA4ScaleTileGroups;
}

// Publishes BF16 pair `pair` (elements 2*pair and 2*pair + 1) of column `token` of an
// [InputRows, T] activation. Pairs 8j..8j+7 of a column must sit in eight consecutive lanes
// 8j'..8j'+7 of one warp, in order, and all 32 lanes must call together (see
// detail::quantize_nvfp4_pair_group8). The group's first lane writes its scale.
template <int InputRows>
__device__ __forceinline__ void a4_publish_pair(std::uint8_t* codes, std::uint8_t* scales,
                                                float input_scale_divisor, int token, int pair,
                                                const __nv_bfloat162& value) {
    const detail::Nvfp4QuantizedPair quantized =
        detail::quantize_nvfp4_pair_group8(value, input_scale_divisor);
    codes[static_cast<std::int64_t>(token) * (InputRows / 2) + pair] = quantized.code;
    if ((pair & 7) == 0) {
        scales[a4_scale_offset<InputRows / 16>(token, pair / 8)] = quantized.scale;
    }
}

// Publishes 16-value group `group` of column `token` of an [InputRows, T] activation from its
// BF16 values held as two vectors of eight (element 0 in the low half of lo.x).
template <int InputRows>
__device__ __forceinline__ void a4_publish_group16(std::uint8_t* codes, std::uint8_t* scales,
                                                   float input_scale_divisor, int token, int group,
                                                   uint4 lo, uint4 hi) {
    const std::uint32_t bits[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
    const detail::Nvfp4QuantizedK16 quantized =
        detail::quantize_nvfp4_k16_bits(bits, input_scale_divisor);
    store_vec(codes + static_cast<std::int64_t>(token) * (InputRows / 2) + group * 8,
              make_uint2(quantized.codes_lo, quantized.codes_hi));
    scales[a4_scale_offset<InputRows / 16>(token, group)] = quantized.scale;
}

// Writes zero scales for every group of column `token`, a padding token of the last 16-token
// fragment, from `threads` cooperating threads with indices `thread`.
template <int InputRows>
__device__ __forceinline__ void a4_zero_scales(std::uint8_t* scales, int token, int thread,
                                               int threads) {
    for (int group = thread; group < InputRows / 16; group += threads) {
        scales[a4_scale_offset<InputRows / 16>(token, group)] = 0;
    }
}

} // namespace ninfer::ops
