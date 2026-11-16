#pragma once

#include "ops/common/memory.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Split store of an attention-input parent in physical row order query, key, output gate, value,
// with QueryRows query/gate rows and KeyValueRows key/value rows, into four contiguous BF16
// allocations q/gate [QueryRows,T] and k/v [KeyValueRows,T]. A vector store covers eight rows
// starting at a multiple of eight, which never crosses a range boundary, and needs 16-byte-aligned
// bases.
template <std::int32_t QueryRows, std::int32_t KeyValueRows>
struct Nvfp4AttentionInputOutput {
    static_assert(QueryRows % 128 == 0 && KeyValueRows % 128 == 0);

    static constexpr std::int32_t kKeyBegin   = QueryRows;
    static constexpr std::int32_t kGateBegin  = kKeyBegin + KeyValueRows;
    static constexpr std::int32_t kValueBegin = kGateBegin + QueryRows;
    static constexpr std::int32_t kParentRows = kValueBegin + KeyValueRows;

    __nv_bfloat16* query;
    __nv_bfloat16* key;
    __nv_bfloat16* gate;
    __nv_bfloat16* value;

    __device__ __forceinline__ __nv_bfloat16* destination(std::int32_t parent_row,
                                                          std::int32_t token) const {
        if (parent_row < kKeyBegin) {
            return query + static_cast<std::int64_t>(token) * QueryRows + parent_row;
        }
        if (parent_row < kGateBegin) {
            return key + static_cast<std::int64_t>(token) * KeyValueRows + parent_row - kKeyBegin;
        }
        if (parent_row < kValueBegin) {
            return gate + static_cast<std::int64_t>(token) * QueryRows + parent_row - kGateBegin;
        }
        return value + static_cast<std::int64_t>(token) * KeyValueRows + parent_row - kValueBegin;
    }

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t token,
                                          float result) const {
        *destination(parent_row, token) = __float2bfloat16_rn(result);
    }

    __device__ __forceinline__ void store_vector(std::int32_t parent_row, std::int32_t token,
                                                 uint4 values) const {
        store_vec(destination(parent_row, token), values);
    }
};

// The [14336,5120] parent: 24 query/gate heads and 4 key/value heads of 256 rows.
using Nvfp4AttentionInput5120Output = Nvfp4AttentionInputOutput<6144, 1024>;
// The Flash-Next [13312,2560] parent: 24 query/gate heads and 2 key/value heads of 256 rows.
using Nvfp4AttentionInput2560Output = Nvfp4AttentionInputOutput<6144, 512>;

} // namespace ninfer::ops::detail
