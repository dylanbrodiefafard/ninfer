#pragma once

// ninfer::ops - split-KV GQA small-T attention shared scaffolding. The bf16 and
// int8 partial kernels live in gqa_attention_decode_bf16.cuh and
// gqa_attention_decode_i8.cuh respectively; they are fully separate kernels (no
// shared body) so each KV format can be optimized independently. This header owns
// only what both share: layout constants, device helpers, and the split reducer.

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"
#include "ops/kernel/gqa_attention_geometry.cuh"
#include "ops/kernel/paged_kv_address.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kGqaHeadDim = 256;

struct GqaAppendInput {
    static constexpr bool writes_cache = true;
    const __nv_bfloat16* k;
    const __nv_bfloat16* v;
};

struct GqaCachedInput {
    static constexpr bool writes_cache = false;
};

template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_cache_index(int physical_page, int kv_head, int d,
                                                        int page_offset) {
    return paged_kv_element_offset<kGqaHeadDim, Geometry::KVHeads>(physical_page, kv_head,
                                                                   page_offset, d);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_q_index(int q_head, int d, int token = 0) {
    return static_cast<std::int64_t>(d) + static_cast<std::int64_t>(kGqaHeadDim) *
                                              (static_cast<std::int64_t>(q_head) +
                                               static_cast<std::int64_t>(Geometry::QHeads) * token);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_kv_new_index(int kv_head, int d, int token = 0) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kGqaHeadDim) *
               (static_cast<std::int64_t>(kv_head) +
                static_cast<std::int64_t>(Geometry::KVHeads) * token);
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_partial_acc_index(int q_head, int d, int token,
                                                              int split, int tokens) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kGqaHeadDim) *
               (static_cast<std::int64_t>(q_head) +
                static_cast<std::int64_t>(Geometry::QHeads) *
                    (static_cast<std::int64_t>(token) + static_cast<std::int64_t>(tokens) * split));
}

template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_partial_stat_index(int q_head, int token, int split,
                                                               int tokens) {
    return static_cast<std::int64_t>(q_head) +
           static_cast<std::int64_t>(Geometry::QHeads) *
               (static_cast<std::int64_t>(token) + static_cast<std::int64_t>(tokens) * split);
}

template <typename Geometry>
__device__ __forceinline__ bool gqa_valid_q_head(int kv_head, int q_head) {
    return kv_head >= 0 && kv_head < Geometry::KVHeads && q_head >= kv_head * Geometry::GroupSize &&
           q_head < (kv_head + 1) * Geometry::GroupSize && q_head < Geometry::QHeads;
}

template <typename Geometry>
__device__ __forceinline__ int gqa_small_t_default_splits(int window) {
    int target_keys_per_split = 480 / Geometry::DecodeSplitScale;
    if (window <= 4096) {
        target_keys_per_split = 64 / Geometry::DecodeSplitScale;
    } else if (window <= 8198) {
        target_keys_per_split = 128 / Geometry::DecodeSplitScale;
    } else if (window <= 16390) {
        target_keys_per_split = 256 / Geometry::DecodeSplitScale;
    }
    constexpr int kMinSplits = 4 * Geometry::DecodeSplitScale;
    int splits               = div_up(window, target_keys_per_split);
    splits                   = splits > kMinSplits ? splits : kMinSplits;
    return splits < Geometry::DecodeSplits ? splits : Geometry::DecodeSplits;
}

template <typename Geometry, bool Int8>
__device__ __forceinline__ int gqa_small_t_active_splits(int window, int launch_capacity,
                                                         int tokens) {
    if (window <= 0) { return launch_capacity; }
    int splits = 0;
    if constexpr (Int8) {
        if (tokens == 5 && window > 128 && window <= 512) {
            splits = div_up(window, 32 / Geometry::DecodeSplitScale);
        } else if (tokens == 6 && window > 128 && window <= 160) {
            splits = div_up(window, 24 / Geometry::DecodeSplitScale);
        } else if (tokens == 6 && window > 5000 && window <= 8198) {
            splits             = div_up(window, 192 / Geometry::DecodeSplitScale);
            constexpr int kMin = 4 * Geometry::DecodeSplitScale;
            constexpr int kMax = 42 * Geometry::DecodeSplitScale;
            splits             = splits > kMin ? splits : kMin;
            splits             = splits < kMax ? splits : kMax;
        } else {
            splits = gqa_small_t_default_splits<Geometry>(window);
        }
    } else {
        splits = gqa_small_t_default_splits<Geometry>(window);
    }
    return splits < launch_capacity ? splits : launch_capacity;
}

__device__ __forceinline__ bool gqa_key_in_causal_split(int row, int row_count, int key,
                                                        int split_start, int split_end, int qabs) {
    return row < row_count && key >= split_start && key < split_end && key <= qabs;
}

template <bool TreeMasked>
__device__ __forceinline__ bool gqa_tree_allows_key(int key, int prefix_length, int packed_width,
                                                    int ancestor_bits) {
    if constexpr (!TreeMasked) {
        (void)key;
        (void)prefix_length;
        (void)packed_width;
        (void)ancestor_bits;
        return true;
    }
    if (key < prefix_length) { return true; }
    const int packed = key - prefix_length;
    return packed >= 0 && packed < packed_width &&
           ((static_cast<unsigned>(ancestor_bits) >> packed) & 1u) != 0u;
}

__device__ __forceinline__ int gqa_small_t_tc_swz(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

__device__ __forceinline__ int gqa_small_t_tc_swz32(int row, int col) {
    return (((col >> 3) ^ (row & 3)) << 3) | (col & 7);
}

// Signed int8 QK MMA, k=32 contraction. A = 16x32 s8 (4 regs/thread, 4 s8 each),
// B = 8x32 s8 col-major (2 regs/thread), D = 16x8 s32 (4 regs/thread). The A/B
// register byte layout is identical to the m16n8k16 bf16 fragments loaded by
// ldmatrix_x4/x2 over a d-contiguous int8 tile reinterpreted as
// b16 (two packed int8 per 16-bit lane), so the same ldmatrix helpers and XOR
// swizzle feed this MMA. The s32 accumulator layout matches the bf16 f32
// accumulator (c0/c1 -> row groupID, c2/c3 -> row groupID+8), so score
// consumption is unchanged; only per-64-group scale rescale differs.
template <typename Geometry>
__device__ __forceinline__ void gqa_small_t_tc_row_to_qt(int row, int tokens, int kv_head,
                                                         int& q_head, int& token) {
    token             = row / Geometry::GroupSize;
    const int local_q = row - token * Geometry::GroupSize;
    q_head            = kv_head * Geometry::GroupSize + local_q;
}

template <typename Geometry, int DChunk, bool Int8, bool MultiBatch, bool Masked, bool Offset>
__launch_bounds__(256) __global__ void gqa_attention_small_t_reduce_output_kernel(
    const __nv_bfloat16* partial_acc, const float* partial_m, const float* partial_l,
    const std::int32_t* positions, const std::int32_t* valid_columns, std::int32_t tokens,
    std::int32_t full_width, std::int32_t column_begin, std::int32_t batch_size,
    std::int32_t split_count, __nv_bfloat16* out) {
    static_assert(DChunk > 0 && DChunk <= kGqaHeadDim);

    const int q_head      = static_cast<int>(blockIdx.x);
    const int d_start     = static_cast<int>(blockIdx.y) * DChunk;
    const int flat_column = static_cast<int>(blockIdx.z);
    int batch             = 0;
    int token             = flat_column;
    if constexpr (MultiBatch) {
        batch = flat_column / tokens;
        token = flat_column - batch * tokens;
    }
    const int tid = threadIdx.x;
    if (q_head >= Geometry::QHeads || token >= tokens) { return; }
    if constexpr (MultiBatch) {
        if (batch >= batch_size) { return; }
    }

    if constexpr (Offset) { positions += column_begin; }
    if constexpr (MultiBatch) { positions += batch * full_width; }
    const int last_pos = positions[tokens - 1];
    int output_column  = token;
    if constexpr (Offset) { output_column += column_begin; }
    if constexpr (MultiBatch) { output_column += batch * full_width; }

    if constexpr (MultiBatch) {
        const std::int64_t partial_acc_row = static_cast<std::int64_t>(batch) * kGqaHeadDim *
                                              Geometry::QHeads * tokens * split_count;
        const std::int64_t partial_stat_row =
            static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * split_count;
        partial_acc += partial_acc_row;
        partial_m   += partial_stat_row;
        partial_l   += partial_stat_row;
    }

    const int window = last_pos + 1;
    const int active_split_count =
        gqa_small_t_active_splits<Geometry, Int8>(window, split_count, tokens);

    // Staged reduce: bulk-stage this block's partial column (the DChunk x active-splits
    // acc slice + the m/l stat vectors) into dynamic smem with cp.async, then run the
    // m/l/acc tree over smem. The prior design ran three serial chains of global
    // loads (one per pass over the active splits); at 85-170 splits x 96-576 blocks
    // on 170 SMs that was latency-bound (long_scoreboard). Staging is a fully
    // parallel bulk copy; the smem chain removes global-load latency from the serial
    // path. The staged values are bit-identical to the direct loads, so the numerics
    // (and the 4-way quarter sum order) are unchanged.
    //
    // Dynamic smem (launcher-sized): [m_s (active x f32) | l_s (active x f32) |
    // 16B pad | acc_s (split-major [s][DChunk] bf16)], so the 16B staging chunks are
    // 1:1 copies of the global rows (d is fastest in the workspace, 16B-aligned).
    extern __shared__ std::uint8_t smem_raw[];
    float* m_s = reinterpret_cast<float*>(smem_raw);
    float* l_s = m_s + split_count;
    __nv_bfloat16* acc_s =
        reinterpret_cast<__nv_bfloat16*>((reinterpret_cast<std::uintptr_t>(l_s + split_count) + 15) &
                                          ~std::uintptr_t(15));

    const std::int64_t acc_base = gqa_partial_acc_index<Geometry>(q_head, d_start, token, 0, tokens);
    const std::int64_t stat_base =
        gqa_partial_stat_index<Geometry>(q_head, token, 0, tokens);
    const std::int64_t acc_step =
        static_cast<std::int64_t>(tokens) * Geometry::QHeads * kGqaHeadDim;  // bf16 elems/split
    const std::int64_t stat_step =
        static_cast<std::int64_t>(tokens) * Geometry::QHeads;  // f32 elems/split
    constexpr int kAccChunks = DChunk * 2 / 16;  // 16B chunks per split row
    const int acc_chunks = active_split_count * kAccChunks;
    for (int c = tid; c < acc_chunks; c += 256) {
        const int s = c / kAccChunks;
        const int i = c - s * kAccChunks;
        cp_async<16>(&acc_s[s * DChunk + 8 * i], partial_acc + acc_base + acc_step * s + 8 * i);
    }
    for (int s = tid; s < active_split_count; s += 256) {
        cp_async<4>(&m_s[s], partial_m + stat_base + stat_step * s);
        cp_async<4>(&l_s[s], partial_l + stat_base + stat_step * s);
    }
    cp_commit();
    cp_wait<0>();
    __syncthreads();

    __shared__ float reduce[256];

    // 4-way split parallelism: the 256 threads map onto DChunk d-lanes with
    // tpd = 256/DChunk threads per d, each owning a quarter of the split range
    // (the old code left 256-active_split threads idle and ran pass 3 fully
    // serial across all 85 splits per d). Exact decompositions: max and sum
    // both split cleanly across disjoint quarters; the quad results combine in
    // shared memory (the quarter lanes are DChunk/tpd warps apart, no shfl).
    const int d_local = tid % DChunk;
    const int tpd     = 256 / DChunk;
    const int qn      = tpd < 4 ? tpd : 4;
    const int q       = tid / DChunk;
    const int S4      = (active_split_count + qn - 1) / qn;
    const int s0      = q * S4;
    const int s1      = s0 + S4 < active_split_count ? s0 + S4 : active_split_count;

    float local_m = -CUDART_INF_F;
    for (int split = s0; split < s1; ++split) {
        local_m = fmaxf(local_m, m_s[split]);
    }
    reduce[d_local * tpd + q] = local_m;
    __syncthreads();
    if (tid < DChunk) {
        float gmax = reduce[tid * tpd];
#pragma unroll
        for (int i = 1; i < tpd; ++i) { gmax = fmaxf(gmax, reduce[tid * tpd + i]); }
        reduce[tid] = gmax;
    }
    __syncthreads();
    // The per-(q_head, token, split) stats are d-independent, so every d-group
    // computed the same quarter max; the group result (reduce[0]) is the head
    // max. No cross-d tree: summing/maxing across the d-groups would replicate
    // the identical value DChunk times.
    const float head_m = reduce[0];
    __syncthreads();

    if (head_m == -CUDART_INF_F) {
        if (q == 0) {
            const int d = d_start + d_local;
            out[gqa_q_index<Geometry>(q_head, d, output_column)] = __float2bfloat16(0.0f);
        }
        return;
    }

    float local_l = 0.0f;
    for (int split = s0; split < s1; ++split) {
        const float tile_l = l_s[split];
        if (tile_l > 0.0f) {
            local_l += tile_l * expf(m_s[split] - head_m);
        }
    }
    reduce[d_local * tpd + q] = local_l;
    __syncthreads();
    if (tid < DChunk) {
        float gsum = 0.0f;
#pragma unroll
        for (int i = 0; i < tpd; ++i) { gsum += reduce[tid * tpd + i]; }
        reduce[tid] = gsum;
    }
    __syncthreads();
    // Each d-group holds the full (quarter-summed) l of its (q_head, token); the
    // value is d-independent, so any group's result is the head sum.
    const float head_l = reduce[0];

    const int d = d_start + d_local;
    if (d >= kGqaHeadDim) { return; }

    float numerator_q = 0.0f;
    if (head_l > 0.0f) {
        for (int split = s0; split < s1; ++split) {
            const float tile_l = l_s[split];
            if (tile_l <= 0.0f) { continue; }
            numerator_q +=
                __bfloat162float(acc_s[split * DChunk + d_local]) * expf(m_s[split] - head_m);
        }
    }
    reduce[d_local * tpd + q] = numerator_q;
    __syncthreads();
    if (q == 0) {
        float numerator = 0.0f;
#pragma unroll
        for (int i = 0; i < tpd; ++i) { numerator += reduce[d_local * tpd + i]; }
        bool valid = true;
        if constexpr (Masked) {
            int absolute_column = token;
            if constexpr (Offset) { absolute_column += column_begin; }
            valid = absolute_column < valid_columns[batch];
        }
        const float value = (valid && head_l > 0.0f) ? numerator / head_l : 0.0f;
        out[gqa_q_index<Geometry>(q_head, d, output_column)] = __float2bfloat16(value);
    }
}

} // namespace ninfer::ops
