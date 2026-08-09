#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kKVCacheAppendPrefixHeadDim = 128;
inline constexpr int kKVCacheAppendPrefixHeads   = 8;
inline constexpr int kKVCacheAppendPrefixWindow  = 4096;
inline constexpr int kKVCacheAppendPrefixPage    = 64;

__device__ __forceinline__ void kv_cache_append_prefix_copy_cyclic_unit(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ cache_k, __nv_bfloat16* __restrict__ cache_v, int token,
    int unit_in_token, int slot, int padded_capacity) {
    constexpr int Bf16PerUnit  = 16;
    constexpr int UnitsPerHead = kKVCacheAppendPrefixHeadDim / Bf16PerUnit;
    const int kv_head          = unit_in_token / UnitsPerHead;
    const int d                = (unit_in_token - kv_head * UnitsPerHead) * Bf16PerUnit;
    const std::int64_t src =
        static_cast<std::int64_t>(d) + static_cast<std::int64_t>(kKVCacheAppendPrefixHeadDim) *
                                           (kv_head + kKVCacheAppendPrefixHeads * token);
    const std::int64_t dst = static_cast<std::int64_t>(d) +
                             static_cast<std::int64_t>(kKVCacheAppendPrefixHeadDim) *
                                 (slot + static_cast<std::int64_t>(padded_capacity) * kv_head);

    const int4 k0                               = *reinterpret_cast<const int4*>(&k[src]);
    const int4 v0                               = *reinterpret_cast<const int4*>(&v[src]);
    *reinterpret_cast<int4*>(&cache_k[dst])     = k0;
    *reinterpret_cast<int4*>(&cache_v[dst])     = v0;
    const int4 k1                               = *reinterpret_cast<const int4*>(&k[src + 8]);
    const int4 v1                               = *reinterpret_cast<const int4*>(&v[src + 8]);
    *reinterpret_cast<int4*>(&cache_k[dst + 8]) = k1;
    *reinterpret_cast<int4*>(&cache_v[dst + 8]) = v1;
}

__device__ __forceinline__ void kv_cache_append_prefix_copy_paged_unit(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ cache_k, __nv_bfloat16* __restrict__ cache_v, int token,
    int unit_in_token, int page_offset, int physical_page, int physical_pages) {
    constexpr int Bf16PerUnit  = 16;
    constexpr int UnitsPerHead = kKVCacheAppendPrefixHeadDim / Bf16PerUnit;
    const int kv_head          = unit_in_token / UnitsPerHead;
    const int d                = (unit_in_token - kv_head * UnitsPerHead) * Bf16PerUnit;
    const std::int64_t src =
        static_cast<std::int64_t>(d) + static_cast<std::int64_t>(kKVCacheAppendPrefixHeadDim) *
                                           (kv_head + kKVCacheAppendPrefixHeads * token);
    const std::int64_t dst =
        static_cast<std::int64_t>(d) +
        static_cast<std::int64_t>(kKVCacheAppendPrefixHeadDim) *
            (page_offset + kKVCacheAppendPrefixPage * (physical_page + physical_pages * kv_head));

    const int4 k0                               = *reinterpret_cast<const int4*>(&k[src]);
    const int4 v0                               = *reinterpret_cast<const int4*>(&v[src]);
    *reinterpret_cast<int4*>(&cache_k[dst])     = k0;
    *reinterpret_cast<int4*>(&cache_v[dst])     = v0;
    const int4 k1                               = *reinterpret_cast<const int4*>(&k[src + 8]);
    const int4 v1                               = *reinterpret_cast<const int4*>(&v[src + 8]);
    *reinterpret_cast<int4*>(&cache_k[dst + 8]) = k1;
    *reinterpret_cast<int4*>(&cache_v[dst + 8]) = v1;
}

__global__ void kv_cache_append_prefix_cyclic_kernel(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    const std::int32_t* __restrict__ positions, const std::int32_t* __restrict__ commit_count,
    __nv_bfloat16* __restrict__ cache_k, __nv_bfloat16* __restrict__ cache_v, int min_count,
    int max_count, int padded_capacity) {
    constexpr int UnitsPerToken  = kKVCacheAppendPrefixHeads * 8;
    constexpr int TokensPerBlock = 256 / UnitsPerToken;
    static_assert(TokensPerBlock * UnitsPerToken == 256);
    const int count = commit_count[0];
    if (count < min_count || count > max_count) return;

    const int local         = static_cast<int>(threadIdx.x);
    const int local_token   = local / UnitsPerToken;
    const int unit_in_token = local - local_token * UnitsPerToken;
    const int token         = static_cast<int>(blockIdx.x) * TokensPerBlock + local_token;
    if (token >= count) return;
    const int position = positions[token];
    const int slot     = position & (kKVCacheAppendPrefixWindow - 1);
    kv_cache_append_prefix_copy_cyclic_unit(k, v, cache_k, cache_v, token, unit_in_token, slot,
                                            padded_capacity);
}

__global__ void kv_cache_append_prefix_paged_kernel(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    const std::int32_t* __restrict__ positions, const std::int32_t* __restrict__ commit_count,
    __nv_bfloat16* __restrict__ cache_k, __nv_bfloat16* __restrict__ cache_v,
    const std::int32_t* __restrict__ block_table, int physical_pages, int min_count,
    int max_count) {
    constexpr int UnitsPerToken  = kKVCacheAppendPrefixHeads * 8;
    constexpr int TokensPerBlock = 256 / UnitsPerToken;
    static_assert(TokensPerBlock * UnitsPerToken == 256);
    const int count = commit_count[0];
    if (count < min_count || count > max_count) return;

    const int local         = static_cast<int>(threadIdx.x);
    const int local_token   = local / UnitsPerToken;
    const int unit_in_token = local - local_token * UnitsPerToken;
    const int lane          = local & 31;
    const int token         = static_cast<int>(blockIdx.x) * TokensPerBlock + local_token;
    int position            = 0;
    int physical_page       = 0;
    if (lane == 0 && token < count) {
        position      = positions[token];
        physical_page = block_table[position >> 6];
    }
    position      = __shfl_sync(0xffffffffu, position, 0);
    physical_page = __shfl_sync(0xffffffffu, physical_page, 0);
    if (token < count) {
        kv_cache_append_prefix_copy_paged_unit(k, v, cache_k, cache_v, token, unit_in_token,
                                               position & (kKVCacheAppendPrefixPage - 1),
                                               physical_page, physical_pages);
    }
}

} // namespace ninfer::ops
