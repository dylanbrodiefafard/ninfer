#pragma once

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

struct W8OutputTile {
    __nv_bfloat16* data;
    std::int32_t leading_dim;
    std::int32_t parent_row_begin;

    __device__ __forceinline__ __nv_bfloat16* at(std::int32_t parent_row, std::int32_t col) const {
        return data + static_cast<std::int64_t>(col) * leading_dim + parent_row - parent_row_begin;
    }

    __device__ __forceinline__ bool valid(std::int32_t parent_row, std::int32_t total_rows) const {
        return parent_row < total_rows;
    }
};

struct W8ContiguousOutput {
    __nv_bfloat16* data;
    std::int32_t leading_dim;

    __device__ __forceinline__ std::int32_t row_begin(std::int32_t block,
                                                      std::int32_t tile_rows) const {
        return block * tile_rows;
    }

    __device__ __forceinline__ W8OutputTile tile(std::int32_t /*parent_row_begin*/) const {
        return {data, leading_dim, 0};
    }
};

} // namespace ninfer::ops::detail
