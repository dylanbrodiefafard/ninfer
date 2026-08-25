#pragma once

#include "ops/kernel/gqa_attention_geometry.cuh"
#include "ops/kernel/paged_kv_address.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kGqaCompactHeadDim = 256;

template <typename Geometry, typename Code>
__device__ __forceinline__ void gqa_kv_compact_copy_codes(Code* __restrict__ plane,
                                                          std::int32_t src_page,
                                                          std::int32_t dst_page,
                                                          std::int32_t src_off,
                                                          std::int32_t dst_off) {
    for (int head = 0; head < Geometry::KVHeads; ++head) {
        Code* src = plane + paged_kv_element_offset<kGqaCompactHeadDim, Geometry::KVHeads>(
                                src_page, head, src_off, 0);
        Code* dst = plane + paged_kv_element_offset<kGqaCompactHeadDim, Geometry::KVHeads>(
                                dst_page, head, dst_off, 0);
        for (int d = static_cast<int>(threadIdx.x); d < kGqaCompactHeadDim;
             d += static_cast<int>(blockDim.x)) {
            dst[d] = src[d];
        }
    }
}

template <typename Geometry>
__device__ __forceinline__ void gqa_kv_compact_copy_scales(__half* __restrict__ plane,
                                                           std::int32_t src_page,
                                                           std::int32_t dst_page,
                                                           std::int32_t src_off,
                                                           std::int32_t dst_off) {
    constexpr int kGroups = kGqaCompactHeadDim / 64;
    for (int head = 0; head < Geometry::KVHeads; ++head) {
        __half* src =
            plane + paged_kv_element_offset<kGroups, Geometry::KVHeads>(src_page, head, src_off, 0);
        __half* dst =
            plane + paged_kv_element_offset<kGroups, Geometry::KVHeads>(dst_page, head, dst_off, 0);
        for (int g = static_cast<int>(threadIdx.x); g < kGroups; g += static_cast<int>(blockDim.x)) {
            dst[g] = src[g];
        }
    }
}

template <typename Geometry, typename Code, bool HasScale>
__global__ void gqa_kv_compact_path_kernel(Code* __restrict__ cache_k, Code* __restrict__ cache_v,
                                           __half* __restrict__ cache_k_scale,
                                           __half* __restrict__ cache_v_scale,
                                           const std::int32_t* __restrict__ block_tables,
                                           const std::int32_t* __restrict__ kv_table_rows,
                                           const std::int32_t* __restrict__ prefix_lengths,
                                           const std::int32_t* __restrict__ path,
                                           const std::int32_t* __restrict__ counts,
                                           std::int32_t logical_pages, std::int32_t width) {
    const std::int32_t row   = static_cast<std::int32_t>(blockIdx.x);
    const std::int32_t count = counts[row];
    const std::int32_t prefix = prefix_lengths[row];
    const std::int32_t* table =
        block_tables + static_cast<std::int64_t>(kv_table_rows[row]) * logical_pages;
    const std::int32_t* row_path = path + static_cast<std::int64_t>(row) * width;
    for (std::int32_t i = 0; i < count; ++i) {
        const std::int32_t src_pos = prefix + row_path[i];
        const std::int32_t dst_pos = prefix + i;
        if (src_pos != dst_pos) {
            const std::int32_t src_page = paged_kv_physical_page(table, src_pos);
            const std::int32_t dst_page = paged_kv_physical_page(table, dst_pos);
            const std::int32_t src_off  = src_pos & kPagedKVPageMask;
            const std::int32_t dst_off  = dst_pos & kPagedKVPageMask;
            gqa_kv_compact_copy_codes<Geometry>(cache_k, src_page, dst_page, src_off, dst_off);
            gqa_kv_compact_copy_codes<Geometry>(cache_v, src_page, dst_page, src_off, dst_off);
            if constexpr (HasScale) {
                gqa_kv_compact_copy_scales<Geometry>(cache_k_scale, src_page, dst_page, src_off,
                                                     dst_off);
                gqa_kv_compact_copy_scales<Geometry>(cache_v_scale, src_page, dst_page, src_off,
                                                     dst_off);
            }
        }
        __syncthreads();
    }
}

} // namespace ninfer::ops
