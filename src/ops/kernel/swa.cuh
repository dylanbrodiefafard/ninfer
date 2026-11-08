#pragma once

#include "ops/common/math.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kSwaHeadDim = 128;
inline constexpr int kSwaQHeads  = 32;
inline constexpr int kSwaKVHeads = 8;
inline constexpr int kSwaGroup   = 4;

// Shared-memory rows are 128 BF16 values (256 bytes). The 16-byte chunk index is XOR-swizzled
// with the row's low three bits so ldmatrix reads of eight consecutive rows hit distinct banks.
__device__ __forceinline__ int swa_swz(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

__device__ __forceinline__ unsigned swa_swz_addr(unsigned lane_base, unsigned ck, unsigned as,
                                                 unsigned r) {
    return lane_base + ((ck | as) ^ r);
}

// q/out: contiguous [128,32,T] for one batch row.
__device__ __forceinline__ std::int64_t swa_q_index(int q_head, int d, int token) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kSwaHeadDim) *
               (static_cast<std::int64_t>(q_head) + static_cast<std::int64_t>(kSwaQHeads) * token);
}

// query_k/query_v: contiguous [128,8,T] for one batch row.
__device__ __forceinline__ std::int64_t swa_query_kv_index(int kv_head, int d, int token) {
    return static_cast<std::int64_t>(d) + static_cast<std::int64_t>(kSwaHeadDim) *
                                              (static_cast<std::int64_t>(kv_head) +
                                               static_cast<std::int64_t>(kSwaKVHeads) * token);
}

// Cyclic context K/V: [128,padded_context,8] for one lane; absolute position p is at slot p mod W.
__device__ __forceinline__ std::int64_t swa_context_index(int kv_head, int d, int slot,
                                                          int padded_context) {
    return static_cast<std::int64_t>(d) + static_cast<std::int64_t>(kSwaHeadDim) *
                                              (static_cast<std::int64_t>(slot) +
                                               static_cast<std::int64_t>(padded_context) * kv_head);
}

// Split partial accumulators: [128,32,T,split] for one batch row.
template <int Tokens>
__device__ __forceinline__ std::int64_t swa_partial_index(int q_head, int d, int token, int split) {
    return static_cast<std::int64_t>(d) +
           static_cast<std::int64_t>(kSwaHeadDim) *
               (static_cast<std::int64_t>(q_head) +
                static_cast<std::int64_t>(kSwaQHeads) *
                    (static_cast<std::int64_t>(token) + static_cast<std::int64_t>(Tokens) * split));
}

// Split running maxima and sums: [32,T,split] for one batch row.
template <int Tokens>
__device__ __forceinline__ std::int64_t swa_stat_index(int q_head, int token, int split) {
    return static_cast<std::int64_t>(q_head) +
           static_cast<std::int64_t>(kSwaQHeads) *
               (static_cast<std::int64_t>(token) + static_cast<std::int64_t>(Tokens) * split);
}

// A CTA owns one KV head; its MMA row r is query token r/4 and query head kv_head*4 + r%4.
__device__ __forceinline__ void swa_row_to_qt(int row, int kv_head, int& q_head, int& token) {
    token             = row / kSwaGroup;
    const int q_local = row - token * kSwaGroup;
    q_head            = kv_head * kSwaGroup + q_local;
}

// Stages KeyBlock key rows of one KV head into dst with cp.async. A query tile reads the
// temporary query K/V rows [0,valid_keys); a context tile reads absolute positions
// [key0,key0+valid_keys) from their cyclic slots. Rows past valid_keys are zero-filled.
template <int KeyBlock, int Threads>
__device__ __forceinline__ void swa_stage_tile(__nv_bfloat16* dst, const __nv_bfloat16* context,
                                               const __nv_bfloat16* query, int key0, int valid_keys,
                                               bool query_tile, int kv_head, int padded_context,
                                               int window, int tid) {
    constexpr int VecsPerRow = kSwaHeadDim / 8;
    for (int chunk = tid; chunk < KeyBlock * VecsPerRow; chunk += Threads) {
        const int row      = chunk / VecsPerRow;
        const int d        = (chunk - row * VecsPerRow) * 8;
        const bool live    = row < valid_keys;
        const int safe_row = live ? row : 0;
        const int slot     = (live ? key0 + row : 0) & (window - 1);
        const __nv_bfloat16* src =
            query_tile ? query + swa_query_kv_index(kv_head, d, safe_row)
                       : context + swa_context_index(kv_head, d, slot, padded_context);
        __nv_bfloat16* smem = &dst[row * kSwaHeadDim + swa_swz(row, d)];
        cp_async_zfill<16, Cache::cg>(smem, src, live ? 16 : 0);
    }
}

/**
 * Split-KV flash-attention partial for one (KV head, split, batch row).
 *
 * Grid (8, split_capacity, B); block WarpsPerCta*32 threads with WarpsPerCta = ceil(T/4). Warp w
 * owns MMA rows [16w,16w+16), covering the 4*T (query token, query head) rows of its KV head.
 * The populated context [L-min(L,W-1),L) is cut into KeyBlock-row tiles divided evenly among the
 * active splits; the last active split also attends to the temporary query block. With
 * DirectOutput the single split normalizes and writes out; otherwise it writes unnormalized BF16
 * accumulators and FP32 running maxima/sums for swa_reduce_kernel.
 *
 * Shared memory: two swizzled [KeyBlock,128] BF16 buffers (K then V), first reused to stage Q
 * before every Q fragment moves into registers. K for the next tile is prefetched while the
 * current V tile is consumed; cp_wait/__syncthreads pairs separate every buffer reuse.
 */
template <int Tokens, int WarpsPerCta, int KeyBlock, bool DirectOutput>
__launch_bounds__(WarpsPerCta * 32, 2) __global__ void swa_split_partial_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ query_k,
    const __nv_bfloat16* __restrict__ query_v, const std::int32_t* __restrict__ positions,
    const std::int32_t* __restrict__ valid_columns, const std::int32_t* __restrict__ lanes,
    const __nv_bfloat16* __restrict__ context_k, const __nv_bfloat16* __restrict__ context_v,
    int padded_context, int max_context, int window, int split_capacity, float scale,
    __nv_bfloat16* __restrict__ partial_acc, float* __restrict__ partial_m,
    float* __restrict__ partial_l, __nv_bfloat16* __restrict__ out) {
    static_assert(Tokens >= 1 && Tokens <= 16);
    static_assert(WarpsPerCta == (Tokens + 3) / 4);
    static_assert(KeyBlock == 32 || KeyBlock == 64);

    constexpr int D             = kSwaHeadDim;
    constexpr int Wc            = WarpsPerCta;
    constexpr int Threads       = Wc * 32;
    constexpr int Br            = Wc * 16;
    constexpr int RowCount      = Tokens * kSwaGroup;
    constexpr int QKNt          = KeyBlock / 8;
    constexpr int QKKs          = D / 16;
    constexpr int PVNt          = D / 8;
    constexpr int PVKs          = KeyBlock / 16;
    constexpr int RowBytes      = D * static_cast<int>(sizeof(__nv_bfloat16));
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;

    static_assert(RowCount <= Br);
    static_assert(Br <= 2 * KeyBlock);
    const int kv_head = static_cast<int>(blockIdx.x);
    const int split   = static_cast<int>(blockIdx.y);
    const int batch   = static_cast<int>(blockIdx.z);
    const int tid     = static_cast<int>(threadIdx.x);
    const int warp    = tid >> 5;
    const int lane    = tid & 31;

    constexpr std::int64_t QueryElements   = static_cast<std::int64_t>(D) * kSwaQHeads * Tokens;
    constexpr std::int64_t QueryKvElements = static_cast<std::int64_t>(D) * kSwaKVHeads * Tokens;
    constexpr std::int64_t PartialElements = QueryElements;
    constexpr std::int64_t StatElements    = static_cast<std::int64_t>(kSwaQHeads) * Tokens;
    q += QueryElements * batch;
    query_k += QueryKvElements * batch;
    query_v += QueryKvElements * batch;
    out += QueryElements * batch;
    partial_acc += PartialElements * split_capacity * batch;
    partial_m += StatElements * split_capacity * batch;
    partial_l += StatElements * split_capacity * batch;
    const int valid = valid_columns[batch];
    positions += static_cast<std::int64_t>(Tokens) * batch;
    const std::int64_t lane_elements = static_cast<std::int64_t>(D) * padded_context * kSwaKVHeads;
    context_k += lane_elements * lanes[batch];
    context_v += lane_elements * lanes[batch];
    const int length = positions[0];
    if (kv_head >= kSwaKVHeads || split >= split_capacity || length < 0 || length > max_context ||
        valid < 1 || valid > Tokens) {
        return;
    }

    const int context_count = min(length, window - 1);
    const int context_start = length - context_count;
    const int context_tiles = (context_count + KeyBlock - 1) / KeyBlock;
    const int active_splits = context_tiles > 0 ? min(context_tiles, split_capacity) : 1;
    if (split >= active_splits) { return; }

    const int tile_begin =
        static_cast<int>((static_cast<std::int64_t>(context_tiles) * split) / active_splits);
    const int tile_end =
        static_cast<int>((static_cast<std::int64_t>(context_tiles) * (split + 1)) / active_splits);
    const bool owns_query        = split == active_splits - 1;
    const int context_tile_count = tile_end - tile_begin;
    const int iterations         = context_tile_count + (owns_query ? 1 : 0);

    extern __shared__ __align__(16) __nv_bfloat16 shared[];
    __nv_bfloat16* k_s = shared;
    __nv_bfloat16* v_s = shared + KeyBlock * D;

    // The two K/V buffers together hold at least Br rows. Use them once as Q staging, then retain
    // all Q MMA fragments in registers for the complete split.
    for (int chunk = tid; chunk < Br * (D / 8); chunk += Threads) {
        const int row = chunk / (D / 8);
        const int d   = (chunk - row * (D / 8)) * 8;
        int q_head = 0, token = 0;
        swa_row_to_qt(row, kv_head, q_head, token);
        const bool live          = row < RowCount && token < valid;
        const __nv_bfloat16* src = q + swa_q_index(live ? q_head : 0, d, live ? token : 0);
        __nv_bfloat16* dst       = &shared[row * D + swa_swz(row, d)];
        cp_async_zfill<16, Cache::cg>(dst, src, live ? 16 : 0);
    }
    cp_commit();
    cp_wait<0>();
    __syncthreads();

    const int gid = lane >> 2;
    const int lid = lane & 3;

    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin    = lane & 7;
    const int b_koff   = ((lane >> 3) & 1) << 3;

    const int warp_row0   = warp * 16;
    const int row0        = warp_row0 + gid;
    const int row1        = row0 + 8;
    const int q_position0 = positions[row0 < RowCount ? row0 / kSwaGroup : 0];
    const int q_position1 = positions[row1 < RowCount ? row1 / kSwaGroup : 0];
    unsigned af_q[QKKs][4];
#pragma unroll
    for (int ks = 0; ks < QKKs; ++ks) {
        const int row = warp_row0 + a_rowoff;
        const int col = ks * 16 + a_coloff;
        ldmatrix_x4(af_q[ks][0], af_q[ks][1], af_q[ks][2], af_q[ks][3],
                    smem_addr(&shared[row * D + swa_swz(row, col)]));
    }
    __syncthreads();

    float acc[PVNt][4];
#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
#pragma unroll
        for (int item = 0; item < 4; ++item) { acc[n][item] = 0.0f; }
    }
    float m0 = -CUDART_INF_F;
    float m1 = -CUDART_INF_F;
    float l0 = 0.0f;
    float l1 = 0.0f;

    const unsigned v_sbase     = smem_addr(v_s);
    const unsigned v_lane_base = v_sbase + static_cast<unsigned>(((lane >> 3) & 1) * 8 * RowBytes) +
                                 static_cast<unsigned>(b_rin * RowBytes);
    const unsigned v_as        = static_cast<unsigned>((lane >> 4) << 4);
    const unsigned v_r         = static_cast<unsigned>(b_rin << 4);

    auto tile_metadata = [&](int iteration, bool& is_query, int& key0, int& valid_keys) {
        is_query = iteration >= context_tile_count;
        if (is_query) {
            key0       = 0;
            valid_keys = valid;
        } else {
            key0       = context_start + (tile_begin + iteration) * KeyBlock;
            valid_keys = min(KeyBlock, length - key0);
        }
    };

    bool current_is_query = false;
    int current_key0      = 0;
    int current_valid     = 0;
    tile_metadata(0, current_is_query, current_key0, current_valid);
    swa_stage_tile<KeyBlock, Threads>(k_s, context_k, query_k, current_key0, current_valid,
                                      current_is_query, kv_head, padded_context, window, tid);
    cp_commit();

    for (int iteration = 0; iteration < iterations; ++iteration) {
        cp_wait<0>();
        __syncthreads();

        swa_stage_tile<KeyBlock, Threads>(v_s, context_v, query_v, current_key0, current_valid,
                                          current_is_query, kv_head, padded_context, window, tid);
        cp_commit();

        float score[QKNt][4];
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0f;
#pragma unroll
            for (int ks = 0; ks < QKKs; ++ks) {
                unsigned bf[2];
                const int brow = nt * 8 + b_rin;
                const int bcol = ks * 16 + b_koff;
                ldmatrix_x2(bf[0], bf[1], smem_addr(&k_s[brow * D + swa_swz(brow, bcol)]));
                mma_bf16(score[nt][0], score[nt][1], score[nt][2], score[nt][3], af_q[ks][0],
                         af_q[ks][1], af_q[ks][2], af_q[ks][3], bf[0], bf[1]);
            }
        }

        cp_wait<0>();
        __syncthreads();

        bool next_is_query = false;
        int next_key0      = 0;
        int next_valid     = 0;
        if (iteration + 1 < iterations) {
            tile_metadata(iteration + 1, next_is_query, next_key0, next_valid);
            swa_stage_tile<KeyBlock, Threads>(k_s, context_k, query_k, next_key0, next_valid,
                                              next_is_query, kv_head, padded_context, window, tid);
            cp_commit();
        }

        // Context keys more than W-1 positions behind a query row are outside its window; the
        // temporary query block is always inside it.
        float block_m0 = -CUDART_INF_F;
        float block_m1 = -CUDART_INF_F;
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            const int col0       = nt * 8 + 2 * lid;
            const int col1       = col0 + 1;
            const bool row0_live = row0 < RowCount && row0 / kSwaGroup < valid;
            const bool row1_live = row1 < RowCount && row1 / kSwaGroup < valid;
            const bool allow00 =
                row0_live && col0 < current_valid &&
                (current_is_query || current_key0 + col0 >= q_position0 - (window - 1));
            const bool allow01 =
                row0_live && col1 < current_valid &&
                (current_is_query || current_key0 + col1 >= q_position0 - (window - 1));
            const bool allow10 =
                row1_live && col0 < current_valid &&
                (current_is_query || current_key0 + col0 >= q_position1 - (window - 1));
            const bool allow11 =
                row1_live && col1 < current_valid &&
                (current_is_query || current_key0 + col1 >= q_position1 - (window - 1));
            score[nt][0] = allow00 ? score[nt][0] * scale : -CUDART_INF_F;
            score[nt][1] = allow01 ? score[nt][1] * scale : -CUDART_INF_F;
            score[nt][2] = allow10 ? score[nt][2] * scale : -CUDART_INF_F;
            score[nt][3] = allow11 ? score[nt][3] * scale : -CUDART_INF_F;
            block_m0     = fmaxf(block_m0, fmaxf(score[nt][0], score[nt][1]));
            block_m1     = fmaxf(block_m1, fmaxf(score[nt][2], score[nt][3]));
        }
        block_m0 = warp_max<4>(block_m0, FullMask);
        block_m1 = warp_max<4>(block_m1, FullMask);

        const float next_m0 = fmaxf(m0, block_m0);
        const float next_m1 = fmaxf(m1, block_m1);
        const float alpha0  = m0 == -CUDART_INF_F ? 0.0f : exp2_approx((m0 - next_m0) * Log2E);
        const float alpha1  = m1 == -CUDART_INF_F ? 0.0f : exp2_approx((m1 - next_m1) * Log2E);

        unsigned p_frag[PVKs][4];
        float block_l0 = 0.0f;
        float block_l1 = 0.0f;
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            const float p00 =
                score[nt][0] > -CUDART_INF_F ? exp2_approx((score[nt][0] - next_m0) * Log2E) : 0.0f;
            const float p01 =
                score[nt][1] > -CUDART_INF_F ? exp2_approx((score[nt][1] - next_m0) * Log2E) : 0.0f;
            const float p10 =
                score[nt][2] > -CUDART_INF_F ? exp2_approx((score[nt][2] - next_m1) * Log2E) : 0.0f;
            const float p11 =
                score[nt][3] > -CUDART_INF_F ? exp2_approx((score[nt][3] - next_m1) * Log2E) : 0.0f;
            block_l0 += p00 + p01;
            block_l1 += p10 + p11;
            const int pk = nt >> 1;
            if ((nt & 1) == 0) {
                p_frag[pk][0] = pack_bf16x2(p00, p01);
                p_frag[pk][1] = pack_bf16x2(p10, p11);
            } else {
                p_frag[pk][2] = pack_bf16x2(p00, p01);
                p_frag[pk][3] = pack_bf16x2(p10, p11);
            }
        }
        block_l0 = warp_sum<4>(block_l0, FullMask);
        block_l1 = warp_sum<4>(block_l1, FullMask);

        l0 = l0 * alpha0 + block_l0;
        l1 = l1 * alpha1 + block_l1;
        m0 = next_m0;
        m1 = next_m1;
#pragma unroll
        for (int n = 0; n < PVNt; ++n) {
            acc[n][0] *= alpha0;
            acc[n][1] *= alpha0;
            acc[n][2] *= alpha1;
            acc[n][3] *= alpha1;
        }

        constexpr int PVTilePairs = (PVNt + 1) / 2;
        constexpr int PVLoads     = PVKs * PVTilePairs;
        unsigned vf[2][4];
        ldmatrix_x4_t(vf[0][0], vf[0][1], vf[0][2], vf[0][3],
                      swa_swz_addr(v_lane_base, 0u, v_as, v_r));
#pragma unroll
        for (int load = 0; load < PVLoads; ++load) {
            const int pk   = load / PVTilePairs;
            const int n2   = (load % PVTilePairs) * 2;
            const int cur  = load & 1;
            const int next = cur ^ 1;
            if (load + 1 < PVLoads) {
                const int next_pk = (load + 1) / PVTilePairs;
                const int next_n2 = ((load + 1) % PVTilePairs) * 2;
                ldmatrix_x4_t(
                    vf[next][0], vf[next][1], vf[next][2], vf[next][3],
                    swa_swz_addr(v_lane_base + static_cast<unsigned>(next_pk * 16 * RowBytes),
                                 static_cast<unsigned>(next_n2 << 4), v_as, v_r));
            }
            mma_bf16(acc[n2][0], acc[n2][1], acc[n2][2], acc[n2][3], p_frag[pk][0], p_frag[pk][1],
                     p_frag[pk][2], p_frag[pk][3], vf[cur][0], vf[cur][1]);
            if (n2 + 1 < PVNt) {
                mma_bf16(acc[n2 + 1][0], acc[n2 + 1][1], acc[n2 + 1][2], acc[n2 + 1][3],
                         p_frag[pk][0], p_frag[pk][1], p_frag[pk][2], p_frag[pk][3], vf[cur][2],
                         vf[cur][3]);
            }
        }

        current_is_query = next_is_query;
        current_key0     = next_key0;
        current_valid    = next_valid;
    }

    if constexpr (!DirectOutput) {
        if (lid == 0) {
            if (row0 < RowCount) {
                int q_head = 0, token = 0;
                swa_row_to_qt(row0, kv_head, q_head, token);
                partial_m[swa_stat_index<Tokens>(q_head, token, split)] = m0;
                partial_l[swa_stat_index<Tokens>(q_head, token, split)] = l0;
            }
            if (row1 < RowCount) {
                int q_head = 0, token = 0;
                swa_row_to_qt(row1, kv_head, q_head, token);
                partial_m[swa_stat_index<Tokens>(q_head, token, split)] = m1;
                partial_l[swa_stat_index<Tokens>(q_head, token, split)] = l1;
            }
        }
    }

#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
        const int d0 = n * 8 + 2 * lid;
        if (row0 < RowCount) {
            int q_head = 0, token = 0;
            swa_row_to_qt(row0, kv_head, q_head, token);
            if constexpr (DirectOutput) {
                const float inv_l = l0 > 0.0f ? 1.0f / l0 : 0.0f;
                const auto dst    = swa_q_index(q_head, d0, token);
                store_vec(&out[dst], pack_bf16x2(acc[n][0] * inv_l, acc[n][1] * inv_l));
            } else {
                const auto dst = swa_partial_index<Tokens>(q_head, d0, token, split);
                store_vec(&partial_acc[dst], pack_bf16x2(acc[n][0], acc[n][1]));
            }
        }
        if (row1 < RowCount) {
            int q_head = 0, token = 0;
            swa_row_to_qt(row1, kv_head, q_head, token);
            if constexpr (DirectOutput) {
                const float inv_l = l1 > 0.0f ? 1.0f / l1 : 0.0f;
                const auto dst    = swa_q_index(q_head, d0, token);
                store_vec(&out[dst], pack_bf16x2(acc[n][2] * inv_l, acc[n][3] * inv_l));
            } else {
                const auto dst = swa_partial_index<Tokens>(q_head, d0, token, split);
                store_vec(&partial_acc[dst], pack_bf16x2(acc[n][2], acc[n][3]));
            }
        }
    }
}

/**
 * Combines split partials into normalized BF16 output.
 *
 * Grid (ceil(32*T/WarpsPerBlock), 1, B); block WarpsPerBlock*32 threads. Warp w of block x owns
 * output row x*WarpsPerBlock+w = token*32 + q_head; lane l owns head dimensions l + 32*i for
 * i<4. Shared memory holds each warp's per-split rescale weights (at most 128 splits); only
 * __syncwarp separates their write and read because no warp reads another warp's row.
 */
template <int Tokens, int KeyBlock, int WarpsPerBlock>
__launch_bounds__(WarpsPerBlock * 32, 2) __global__
    void swa_reduce_kernel(const __nv_bfloat16* __restrict__ partial_acc,
                           const float* __restrict__ partial_m, const float* __restrict__ partial_l,
                           const std::int32_t* __restrict__ positions,
                           const std::int32_t* __restrict__ valid_columns, int max_context,
                           int window, int split_capacity, __nv_bfloat16* __restrict__ out) {
    static_assert(WarpsPerBlock >= 1 && WarpsPerBlock <= 8);
    constexpr int MaxSplits = 128;
    constexpr unsigned Mask = 0xffffffffu;
    __shared__ float weights[WarpsPerBlock][MaxSplits];

    const int warp       = static_cast<int>(threadIdx.x) >> 5;
    const int lane       = static_cast<int>(threadIdx.x) & 31;
    const int batch      = static_cast<int>(blockIdx.z);
    const int output_row = static_cast<int>(blockIdx.x) * WarpsPerBlock + warp;
    const int token      = output_row / kSwaQHeads;
    const int q_head     = output_row - token * kSwaQHeads;
    if (warp >= WarpsPerBlock || token >= Tokens) return;

    constexpr std::int64_t QueryElements =
        static_cast<std::int64_t>(kSwaHeadDim) * kSwaQHeads * Tokens;
    constexpr std::int64_t StatElements = static_cast<std::int64_t>(kSwaQHeads) * Tokens;
    partial_acc += QueryElements * split_capacity * batch;
    partial_m += StatElements * split_capacity * batch;
    partial_l += StatElements * split_capacity * batch;
    positions += static_cast<std::int64_t>(Tokens) * batch;
    out += QueryElements * batch;

    const int length = positions[0];
    if (length < 0 || length > max_context || token >= valid_columns[batch]) {
#pragma unroll
        for (int item = 0; item < 4; ++item) {
            const int d                        = lane + item * 32;
            out[swa_q_index(q_head, d, token)] = __float2bfloat16(0.0f);
        }
        return;
    }
    const int context_count = min(length, window - 1);
    const int context_tiles = (context_count + KeyBlock - 1) / KeyBlock;
    const int active_splits = context_tiles > 0 ? min(context_tiles, split_capacity) : 1;

    float local_m = -CUDART_INF_F;
    for (int split = lane; split < active_splits; split += 32) {
        local_m = fmaxf(local_m, partial_m[swa_stat_index<Tokens>(q_head, token, split)]);
    }
    const float global_m = warp_max<32>(local_m, Mask);

    float local_l = 0.0f;
    for (int split = lane; split < active_splits; split += 32) {
        const auto stat      = swa_stat_index<Tokens>(q_head, token, split);
        const float weight   = expf(partial_m[stat] - global_m);
        weights[warp][split] = weight;
        local_l += partial_l[stat] * weight;
    }
    const float global_l = warp_sum<32>(local_l, Mask);
    __syncwarp(Mask);

#pragma unroll
    for (int item = 0; item < 4; ++item) {
        const int d     = lane + item * 32;
        float numerator = 0.0f;
        for (int split = 0; split < active_splits; ++split) {
            numerator +=
                __bfloat162float(partial_acc[swa_partial_index<Tokens>(q_head, d, token, split)]) *
                weights[warp][split];
        }
        const float value                  = global_l > 0.0f ? numerator / global_l : 0.0f;
        out[swa_q_index(q_head, d, token)] = __float2bfloat16(value);
    }
}

} // namespace ninfer::ops
