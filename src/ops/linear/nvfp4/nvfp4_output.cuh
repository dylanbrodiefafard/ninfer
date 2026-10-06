#pragma once

#include "ops/common/memory.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

// The work distributor issues CTAs in linear order with blockIdx.x fastest. Launched as
// grid(weight-row tiles, token tiles), every co-resident CTA would hold a different weight tile
// and the weight matrix would stream from memory once per token tile. Deriving both tile indices
// from the linear CTA id with the token tile fastest keeps the CTAs that share a weight tile
// together, so the matrix is read from DRAM once and the other token tiles hit L2.
// block_x is the weight-row tile and block_y the token tile, as in the launch grid.
__device__ __forceinline__ void nvfp4_raster_token_tiles_fastest(int& block_x, int& block_y) {
    const int rows = static_cast<int>(gridDim.y);
    const int linear =
        static_cast<int>(blockIdx.y) * static_cast<int>(gridDim.x) + static_cast<int>(blockIdx.x);
    block_y = linear % rows;
    block_x = linear / rows;
}

// Number of a warp's 16-token MMA fragments, out of Fragments, that hold at least one of the
// token_count real tokens when the warp's first fragment starts at token warp_token_begin. Only
// the last token tile of a launch is partial; its later fragments hold padding rows only.
template <int Fragments>
__device__ __forceinline__ int nvfp4_active_token_fragments(int warp_token_begin, int token_count) {
    const int remaining = token_count - warp_token_begin;
    if (remaining <= 0) { return 0; }
    return min((remaining + 15) / 16, Fragments);
}

// Calls visitor.template operator()<Active>() with Active = active, 0 <= active <= Fragments, so
// the consumer loop keeps compile-time fragment bounds: a full tile runs the fully unrolled
// loop, and a partial tile issues no shared-memory loads or MMAs for padding fragments.
// A padding fragment's accumulators stay zero; the epilogue drops its stores.
template <int Fragments, class Visitor>
__device__ __forceinline__ void nvfp4_with_active_token_fragments(int active,
                                                                  const Visitor& visitor) {
    if constexpr (Fragments == 0) {
        visitor.template operator()<0>();
    } else if (active >= Fragments) {
        visitor.template operator()<Fragments>();
    } else {
        nvfp4_with_active_token_fragments<Fragments - 1>(active, visitor);
    }
}

struct Nvfp4IdentityEpilogue {
    __device__ __forceinline__ float apply(std::int32_t, std::int32_t, float value) const {
        return value;
    }
};

struct Nvfp4ContiguousOutput {
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

} // namespace ninfer::ops::detail
