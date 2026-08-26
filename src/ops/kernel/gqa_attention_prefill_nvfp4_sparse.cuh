#pragma once

// Exact-NVFP4 GQA prompt kernel with a skip-list K loop. MMA matches
// gqa_attention_prefill_nvfp4.cuh; the K iterator walks a keep_list produced by
// Sparge meansim (keep_frac, in-CTA) or XAttention (xattn_tau, separate select).
//
// XAttention ranking is a prepass (not fused into the MMA CTA). Pack K once to
// bf16 [kv_head, j, s, d], then score the official inverse reshape on one B=128
// plane per MMA CTA: A[i,j] = sum_s Q[15-s+i*16]·K[s+j*16] / (√d·S). find_blocks
// softmaxes over L/S, block-sums into paper B=128 (two Bc=64 pages), greedy
// mass ≥ τ, then expands each kept block to its pages. Force-keep is paper
// find_blocks_chunked: key block 0, the local diagonal B=128 block, and the
// last visible B=128 block. Remainder Q-blocks with <S rows still score one partial
// i-row (skip q_rel >= q_rows) instead of sink-only. Scoring is the paper
// inverse-stride GEMM C[8, n_j] = Qflat[8, S·d] @ packed[n_j, S·d]ᵀ on tensor
// cores (M padded to 16).

#include "ops/kernel/gqa_attention_kv_nvfp4.cuh"
#include "ops/kernel/gqa_attention_prefill_common.cuh"
#include "ops/launcher/gqa_xattn_scratch.h"
#include "ninfer/ops/gqa_attention.h"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

inline constexpr int kGqaPrefillNvfp4Warps    = 16;
inline constexpr int kGqaPrefillNvfp4Threads  = kGqaPrefillNvfp4Warps * 32;
// 128 query rows halves K/V reread versus Br=64: each CTA still streams the
// full key history once, but the grid has half as many tiles.
inline constexpr int kGqaPrefillNvfp4Br       = 128;
inline constexpr int kGqaPrefillNvfp4Bc       = 64;
inline constexpr int kGqaPrefillNvfp4Groups   = kGqaNvfp4Groups;
inline constexpr int kGqaPrefillNvfp4CodeW    = kGqaNvfp4CodeWidth;
inline constexpr int kGqaPrefillNvfp4RowTiles = kGqaPrefillNvfp4Br / 16;
inline constexpr int kGqaPrefillNvfp4DConsumers =
    kGqaPrefillNvfp4Warps / kGqaPrefillNvfp4RowTiles;

inline constexpr int kGqaPrefillNvfp4QBytes     = kGqaPrefillNvfp4Br * kGqaPrefillNvfp4CodeW;
inline constexpr int kGqaPrefillNvfp4QScaleBytes = kGqaPrefillNvfp4Br * kGqaPrefillNvfp4Groups;
inline constexpr int kGqaPrefillNvfp4KBytes     = kGqaPrefillNvfp4Bc * kGqaPrefillNvfp4CodeW;
inline constexpr int kGqaPrefillNvfp4VBytes     = kGqaPrefillNvfp4Bc * kGqaPrefillNvfp4CodeW;
inline constexpr int kGqaPrefillNvfp4VStageBytes =
    kGqaPrefillNvfp4Bc * kGqaPrefillHeadDim * static_cast<int>(sizeof(__nv_bfloat16));
inline constexpr int kGqaPrefillNvfp4PBytes =
    kGqaPrefillNvfp4Br * kGqaPrefillNvfp4Bc * static_cast<int>(sizeof(__nv_bfloat16));
inline constexpr int kGqaPrefillNvfp4ScaleBytes = 2 * kGqaPrefillNvfp4Bc * kGqaPrefillNvfp4Groups;
inline constexpr int kGqaPrefillNvfp4StatsBytes =
    2 * kGqaPrefillNvfp4Br * static_cast<int>(sizeof(float));
inline constexpr int kGqaPrefillNvfp4MmaSmemBytes =
    kGqaPrefillNvfp4QBytes + kGqaPrefillNvfp4QScaleBytes + kGqaPrefillNvfp4KBytes +
    kGqaPrefillNvfp4VBytes + kGqaPrefillNvfp4VStageBytes + kGqaPrefillNvfp4PBytes +
    kGqaPrefillNvfp4ScaleBytes + kGqaPrefillNvfp4StatsBytes;
inline constexpr int kGqaPrefillNvfp4RankTiles = 4096; // 262144 / Bc
// SM120 5090 opt-in is 99 KiB/block. Ranking overlays the MMA arena, then the
// compacted keep-list (uint16 tile ids) lives in the leftover tail.
inline constexpr int kGqaPrefillNvfp4KeepBytes =
    kGqaPrefillNvfp4RankTiles * static_cast<int>(sizeof(std::uint16_t)) +
    static_cast<int>(sizeof(int));
inline constexpr int kGqaPrefillNvfp4SmemBytes = 101376;

inline constexpr int kXAttnStride = kGqaXattnStride;
inline constexpr int kXAttnBlock  = kGqaPrefillNvfp4Br; // paper B=128, one greedy plane per MMA CTA
inline constexpr int kXAttnIRows  = kGqaXattnIRows;
// Pack is one 64-key page: four warps write the four inverse-stride columns.
inline constexpr int kXAttnPackWarps   = 4;
inline constexpr int kXAttnPackThreads = kXAttnPackWarps * 32;
// Inverse-stride GEMM tile: C[8, BN] += A[8, BK] @ B[BN, BK]ᵀ with M padded to 16.
// 20 KiB smem → several CTAs/SM. Grid z walks n_j / BN.
inline constexpr int kXAttnScoreBM      = 16;
inline constexpr int kXAttnScoreBN      = 64;
inline constexpr int kXAttnScoreBK      = 64;
inline constexpr int kXAttnScoreK       = kXAttnStride * kGqaPrefillHeadDim;
inline constexpr int kXAttnScoreStages  = 2;
inline constexpr int kXAttnScoreWarpsN  = 4;
inline constexpr int kXAttnScoreWarps   = kXAttnScoreWarpsN;
inline constexpr int kXAttnScoreThreads = kXAttnScoreWarps * 32;
inline constexpr int kXAttnScoreWN      = kXAttnScoreBN / kXAttnScoreWarpsN;
inline constexpr int kXAttnPageCodeBytes  = kGqaPrefillNvfp4Bc * kGqaNvfp4CodeWidth;
inline constexpr int kXAttnPageScaleBytes = kGqaPrefillNvfp4Bc * kGqaNvfp4Groups;
inline constexpr int kXAttnScoreSmemBytes =
    kXAttnScoreStages * (kXAttnScoreBM + kXAttnScoreBN) * kXAttnScoreBK *
    static_cast<int>(sizeof(__nv_bfloat16));

static_assert(kXAttnBlock == kGqaPrefillNvfp4Br);
static_assert(kGqaPrefillNvfp4Br == kGqaXattnPrefillBr);
static_assert(kGqaPrefillNvfp4Bc == kGqaXattnPrefillBc);
static_assert(kGqaXattnFindB == 2 * kGqaXattnPrefillBc);
static_assert(kGqaPrefillHeadDim == kGqaXattnHeadDim);
static_assert(kGqaPrefillNvfp4RankTiles == kGqaXattnRankTiles);
static_assert(kXAttnIRows == 8);
static_assert(kXAttnScoreK == 4096);
static_assert(kXAttnScoreK % kXAttnScoreBK == 0);
static_assert(kXAttnScoreBN % kXAttnScoreWN == 0);
static_assert(kXAttnScoreWN % 8 == 0);
static_assert(kXAttnScoreSmemBytes <= 101376);

struct GqaXattnScratchView {
    __nv_bfloat16* packed;
    float* logits;
    float* mass;
    std::uint16_t* keep;
    int* count;
    int n_br;
    int n_kb;
    int n_j;
    int keep_stride;
    int q_heads;
};

inline GqaXattnScratchView gqa_xattn_bind_scratch(void* p, int q_heads, int kv_heads, int n_br,
                                                 int n_kb) {
    const int n_j = n_kb * (kGqaPrefillNvfp4Bc / kXAttnStride);
    auto align256 = [](std::uintptr_t x) { return (x + 255u) & ~std::uintptr_t{255}; };
    auto* u                 = static_cast<std::uint8_t*>(p);
    std::uintptr_t o        = 0;
    GqaXattnScratchView v{};
    v.q_heads      = q_heads;
    v.n_br         = n_br;
    v.n_kb         = n_kb;
    v.n_j          = n_j;
    v.keep_stride  = n_kb;
    auto take      = [&](std::size_t n) {
        o              = align256(o);
        std::uint8_t* r = u + o;
        o += n;
        return r;
    };
    v.packed = reinterpret_cast<__nv_bfloat16*>(
        take(sizeof(__nv_bfloat16) * static_cast<std::size_t>(kv_heads) * n_j * kXAttnStride *
             kGqaPrefillHeadDim));
    v.logits = reinterpret_cast<float*>(
        take(sizeof(float) * static_cast<std::size_t>(q_heads) * n_br * kXAttnIRows * n_j));
    v.mass   = reinterpret_cast<float*>(
        take(sizeof(float) * static_cast<std::size_t>(q_heads) * n_br * n_kb));
    v.keep   = reinterpret_cast<std::uint16_t*>(
        take(sizeof(std::uint16_t) * static_cast<std::size_t>(q_heads) * n_br * n_kb));
    v.count  = reinterpret_cast<int*>(take(sizeof(int) * static_cast<std::size_t>(q_heads) * n_br));
    return v;
}

inline int gqa_xattn_score_smem_bytes() { return kXAttnScoreSmemBytes; }

inline constexpr int kXAttnFinishSmemBytes =
    2 * kGqaPrefillNvfp4RankTiles * static_cast<int>(sizeof(float)) +
    kGqaPrefillNvfp4RankTiles * static_cast<int>(sizeof(int)) + kGqaPrefillNvfp4RankTiles;

static_assert(kXAttnFinishSmemBytes <= 101376);

static_assert(kGqaPrefillNvfp4Groups == 16);
static_assert(kGqaPrefillNvfp4DConsumers == 2);
static_assert(kGqaPrefillNvfp4MmaSmemBytes == 87040);
static_assert(kGqaPrefillNvfp4SmemBytes <= 101376);
static_assert(2 * kGqaPrefillNvfp4RankTiles * static_cast<int>(sizeof(float)) +
                  kGqaPrefillNvfp4RankTiles * static_cast<int>(sizeof(int)) +
                  kGqaPrefillNvfp4RankTiles +
                  kGqaPrefillHeadDim * static_cast<int>(sizeof(float)) <=
              kGqaPrefillNvfp4MmaSmemBytes);

template <int N, int Threads>
__device__ void gqa_bitonic_sort_desc(float* keys, int* ids, int tid) {
    static_assert((N & (N - 1)) == 0, "bitonic length must be a power of two");
    for (int k = 2; k <= N; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = tid; i < N; i += Threads) {
                const int ixj = i ^ j;
                if (ixj > i) {
                    const bool want_i_better = (i & k) == 0;
                    const bool i_better =
                        keys[i] > keys[ixj] ||
                        (keys[i] == keys[ixj] && ids[i] >= 0 &&
                         (ids[ixj] < 0 || ids[i] < ids[ixj]));
                    if (i_better != want_i_better) {
                        const float tk = keys[i];
                        keys[i]        = keys[ixj];
                        keys[ixj]      = tk;
                        const int ti   = ids[i];
                        ids[i]         = ids[ixj];
                        ids[ixj]       = ti;
                    }
                }
            }
            __syncthreads();
        }
    }
}

// Pairwise online-softmax merge: (m,z) ⊕ (m2,z2).
__device__ __forceinline__ void gqa_xattn_combine_mz(float& m, float& z, float m2, float z2) {
    if (m2 == m) {
        if (m != -CUDART_INF_F) { z += z2; }
        return;
    }
    if (m2 > m) {
        z = z2 + ((m == -CUDART_INF_F) ? 0.0f : z * expf(m - m2));
        m = m2;
    } else {
        z += (m2 == -CUDART_INF_F) ? 0.0f : z2 * expf(m2 - m);
    }
}

// noinline so the four specializations stay out of the MMA register file.
// nsort is 256, 512, 2048, or 4096; callers pad keys/ids to that length.
__device__ __noinline__ void gqa_xattn_sort_desc(float* keys, int* ids, int nsort, int tid) {
    if (nsort <= 256) {
        gqa_bitonic_sort_desc<256, kGqaPrefillNvfp4Threads>(keys, ids, tid);
    } else if (nsort <= 512) {
        gqa_bitonic_sort_desc<512, kGqaPrefillNvfp4Threads>(keys, ids, tid);
    } else if (nsort <= 2048) {
        gqa_bitonic_sort_desc<2048, kGqaPrefillNvfp4Threads>(keys, ids, tid);
    } else {
        gqa_bitonic_sort_desc<4096, kGqaPrefillNvfp4Threads>(keys, ids, tid);
    }
}

__device__ __forceinline__ int gqa_xattn_n_i_rows(int q_rows) {
    if (q_rows <= 0) { return 0; }
    const int n = q_rows / kXAttnStride;
    return n > 0 ? n : 1;
}

__device__ __forceinline__ int gqa_xattn_logit_index(int q_head, int n_br, int br, int i, int n_j,
                                                    int j) {
    return (((q_head * n_br + br) * kXAttnIRows + i) * n_j) + j;
}

__device__ __forceinline__ int gqa_xattn_mass_index(int q_head, int n_br, int br, int n_kb, int kb) {
    return (q_head * n_br + br) * n_kb + kb;
}

__device__ __forceinline__ std::int64_t gqa_xattn_packed_row(int kv_head, int n_j, int j, int s) {
    return (static_cast<std::int64_t>(kv_head) * n_j + j) * kXAttnStride * kGqaPrefillHeadDim +
           static_cast<std::int64_t>(s) * kGqaPrefillHeadDim;
}

// 64-wide XOR swizzle matching bf16_gemm_mma (ldmatrix-friendly).
__device__ __forceinline__ int gqa_xattn_gemm_swz(int row, int col) {
    return (col & ~63) + ((((col & 63) >> 3) ^ (row & 7)) << 3) + (col & 7);
}

// Dequant each NVFP4 K page once into packed bf16 [kv_head, j, s, d]. Grid is
// (kv_head, kb); four warps write the four inverse-stride columns of the page.
template <typename Geometry, typename Metadata>
__global__ __launch_bounds__(kXAttnPackThreads, 8) void gqa_xattn_pack_kernel(
    const std::uint8_t* __restrict__ cache_k, const std::uint8_t* __restrict__ cache_k_scale,
    Metadata metadata, const std::int32_t* __restrict__ positions, std::int32_t width,
    std::int32_t xattn_min_len, int n_kb_cap, int n_j_cap, __nv_bfloat16* __restrict__ packed) {
    constexpr int D     = kGqaPrefillHeadDim;
    constexpr int Bc    = kGqaPrefillNvfp4Bc;
    constexpr int CodeW = kGqaNvfp4CodeWidth;
    constexpr int Groups = kGqaNvfp4Groups;

    __shared__ __align__(16) std::uint8_t k_codes[kXAttnPageCodeBytes];
    __shared__ __align__(16) std::uint8_t k_scale[kXAttnPageScaleBytes];

    const int kv_head = static_cast<int>(blockIdx.x);
    const int kb      = static_cast<int>(blockIdx.y);
    const int tid     = static_cast<int>(threadIdx.x);
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int tokens  = metadata.valid_tokens(width);
    if (kv_head >= Geometry::KVHeads || kb >= n_kb_cap || tokens <= 0) { return; }

    const int base_pos      = positions[0];
    const int max_query_abs = base_pos + tokens - 1;
    if (max_query_abs + 1 < xattn_min_len) { return; }
    const int tile_k0 = kb * Bc;
    if (tile_k0 > max_query_abs) { return; }

    const std::int32_t* block_table = metadata.block_table();
    const int physical_page         = block_table[kb];
    for (int key_l = tid; key_l < Bc; key_l += kXAttnPackThreads) {
        const int key = tile_k0 + key_l;
        if (key <= max_query_abs) {
            const std::int64_t off =
                gqa_nvfp4_scale_index<Geometry>(physical_page, kv_head, 0, key_l);
            ninfer::ops::cp_async<16>(&k_scale[key_l * Groups], &cache_k_scale[off]);
        } else {
            store_vec(&k_scale[key_l * Groups], make_int4(0, 0, 0, 0));
        }
    }
    for (int chunk = tid; chunk < Bc * (CodeW / 16); chunk += kXAttnPackThreads) {
        const int key_l   = chunk / (CodeW / 16);
        const int seg     = chunk - key_l * (CodeW / 16);
        const int logical = seg * 16;
        const int key     = tile_k0 + key_l;
        std::uint8_t* dst = &k_codes[key_l * CodeW + logical];
        if (key <= max_query_abs) {
            const std::int64_t off =
                gqa_nvfp4_code_index<Geometry>(physical_page, kv_head, logical, key_l);
            cp_async<16, Cache::cg>(dst, &cache_k[off]);
        } else {
            store_vec(dst, make_int4(0, 0, 0, 0));
        }
    }
    ninfer::ops::cp_commit();
    ninfer::ops::cp_wait<0>();
    __syncthreads();

    const int j = kb * (Bc / kXAttnStride) + warp;
    if (j >= n_j_cap) { return; }
#pragma unroll
    for (int s = 0; s < kXAttnStride; ++s) {
        const int key_l               = warp * kXAttnStride + s;
        const std::uint8_t* key_codes = &k_codes[key_l * CodeW];
        const std::uint8_t* key_sc    = &k_scale[key_l * Groups];
        __nv_bfloat16* dst            = packed + gqa_xattn_packed_row(kv_head, n_j_cap, j, s);
#pragma unroll
        for (int d2 = lane; d2 < (D / 2); d2 += 32) {
            const float2 pair = detail::decode_nvfp4_e2m1x2(key_codes[d2]);
            const float sv    = detail::decode_nvfp4_e4m3(key_sc[d2 >> 3]);
            const int d       = d2 << 1;
            dst[d]            = __float2bfloat16(pair.x * sv);
            dst[d + 1]        = __float2bfloat16(pair.y * sv);
        }
    }
}

// Inverse-stride GEMM: C[i, j] = sum_s Q[15-s+i·S] · packed[j, s] / (√d · S).
// Grid is (q_head, br, n_j / BN). Same math as the CUDA-core estimate; tensor-core
// m16n8k16 with M padded to 16.
template <typename Geometry, typename Metadata>
__global__ __launch_bounds__(kXAttnScoreThreads, 4) void gqa_xattn_score_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ packed, Metadata metadata,
    const std::int32_t* __restrict__ positions, float scale, std::int32_t width,
    std::int32_t xattn_min_len, int n_br, int n_j_cap, float* __restrict__ logits) {
    constexpr int D          = kGqaPrefillHeadDim;
    constexpr int QRowStride = D * Geometry::QHeads;
    constexpr int BM         = kXAttnScoreBM;
    constexpr int BN         = kXAttnScoreBN;
    constexpr int BK         = kXAttnScoreBK;
    constexpr int K          = kXAttnScoreK;
    constexpr int S          = kXAttnScoreStages;
    constexpr int WN         = kXAttnScoreWN;
    constexpr int NT         = WN / 8;
    constexpr int KSUB       = BK / 16;
    constexpr int THREADS    = kXAttnScoreThreads;
    constexpr int kTiles     = K / BK;

    extern __shared__ __align__(16) unsigned char xattn_score_smem[];
    auto* As = reinterpret_cast<__nv_bfloat16*>(xattn_score_smem);
    auto* Bs = As + S * BM * BK;

    const int q_head = static_cast<int>(blockIdx.x);
    const int br     = static_cast<int>(blockIdx.y);
    const int tile_n = static_cast<int>(blockIdx.z);
    const int tid    = static_cast<int>(threadIdx.x);
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const int wn     = warp;
    const int tokens = metadata.valid_tokens(width);
    if (q_head >= Geometry::QHeads || br >= n_br || tokens <= 0) { return; }

    const int kv_head       = q_head / Geometry::GroupSize;
    const int base_pos      = positions[0];
    const int max_query_abs = base_pos + tokens - 1;
    if (max_query_abs + 1 < xattn_min_len) { return; }

    const int q_start = br * kXAttnBlock;
    if (q_start >= tokens) { return; }
    const int q_rows = min(kXAttnBlock, tokens - q_start);
    const int n_i    = gqa_xattn_n_i_rows(q_rows);
    if (n_i <= 0) { return; }

    const int q_abs_max = base_pos + q_start + q_rows - 1;
    const int n_j       = min(n_j_cap, q_abs_max / kXAttnStride + 1);
    const int n0        = tile_n * BN;
    if (n0 >= n_j) { return; }

    const int gid          = lane >> 2;
    const int lid          = lane & 3;
    const int a_matrix     = lane >> 3;
    const int a_inner_row  = lane & 7;
    const int a_row_offset = a_inner_row + ((a_matrix & 1) << 3);
    const int a_col_offset = (a_matrix >> 1) << 3;
    const int b_inner_row  = lane & 7;
    const int b_k_offset   = ((lane >> 3) & 1) << 3;

    const __nv_bfloat16* q_block = q + gqa_prefill_q_index<Geometry>(q_head, 0, q_start);
    const __nv_bfloat16* b_base  = packed + gqa_xattn_packed_row(kv_head, n_j_cap, 0, 0);
    const float inv_s            = scale / static_cast<float>(kXAttnStride);

    float accum[NT][4] = {};

    auto stage_inputs = [&](int stage, int k_tile) {
        const int k0  = k_tile * BK;
        auto* a_stage = As + stage * BM * BK;
        auto* b_stage = Bs + stage * BN * BK;
        for (int item = tid; item < BM * (BK / 8); item += THREADS) {
            const int row = item / (BK / 8);
            const int kk  = (item - row * (BK / 8)) * 8;
            auto* dst     = &a_stage[row * BK + gqa_xattn_gemm_swz(row, kk)];
            if (row >= n_i) {
                store_vec(dst, make_int4(0, 0, 0, 0));
                continue;
            }
            const int k     = k0 + kk;
            const int s     = k >> 8;
            const int d     = k & 255;
            const int q_rel = (kXAttnStride - 1 - s) + row * kXAttnStride;
            if (q_rel < q_rows) {
                cp_async<16, Cache::cg>(dst, &q_block[q_rel * QRowStride + d]);
            } else {
                store_vec(dst, make_int4(0, 0, 0, 0));
            }
        }
        for (int item = tid; item < BN * (BK / 8); item += THREADS) {
            const int col = item / (BK / 8);
            const int kk  = (item - col * (BK / 8)) * 8;
            auto* dst     = &b_stage[col * BK + gqa_xattn_gemm_swz(col, kk)];
            const int j   = n0 + col;
            if (j < n_j) {
                cp_async<16, Cache::cg>(dst, &b_base[static_cast<std::int64_t>(j) * K + k0 + kk]);
            } else {
                store_vec(dst, make_int4(0, 0, 0, 0));
            }
        }
    };

#pragma unroll
    for (int stage = 0; stage < S; ++stage) {
        stage_inputs(stage, stage);
        ninfer::ops::cp_commit();
    }

#pragma unroll 1
    for (int k_tile = 0; k_tile < kTiles; ++k_tile) {
        const int stage = k_tile % S;
        if (k_tile + S <= kTiles) {
            ninfer::ops::cp_wait<S - 1>();
        } else {
            ninfer::ops::cp_wait<0>();
        }
        __syncthreads();

        unsigned a_frag[2][4];
        unsigned b_frag[2][NT][2];
        auto load_frags = [&](int k_step, unsigned(&a)[4], unsigned(&b)[NT][2]) {
            const int arow = a_row_offset;
            const int acol = k_step * 16 + a_col_offset;
            ldmatrix_x4(a[0], a[1], a[2], a[3],
                        smem_addr(&As[stage * BM * BK + arow * BK + gqa_xattn_gemm_swz(arow, acol)]));
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                const int brow = wn * WN + ni * 8 + b_inner_row;
                const int bcol = k_step * 16 + b_k_offset;
                ldmatrix_x2(b[ni][0], b[ni][1],
                            smem_addr(
                                &Bs[stage * BN * BK + brow * BK + gqa_xattn_gemm_swz(brow, bcol)]));
            }
        };
        load_frags(0, a_frag[0], b_frag[0]);
#pragma unroll
        for (int k_step = 0; k_step < KSUB; ++k_step) {
            const int slot = k_step & 1;
            if (k_step + 1 < KSUB) { load_frags(k_step + 1, a_frag[slot ^ 1], b_frag[slot ^ 1]); }
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                mma_bf16(accum[ni][0], accum[ni][1], accum[ni][2], accum[ni][3], a_frag[slot][0],
                         a_frag[slot][1], a_frag[slot][2], a_frag[slot][3], b_frag[slot][ni][0],
                         b_frag[slot][ni][1]);
            }
        }

        __syncthreads();
        const int next = k_tile + S;
        if (next < kTiles) {
            stage_inputs(stage, next);
            ninfer::ops::cp_commit();
        }
    }

#pragma unroll
    for (int ni = 0; ni < NT; ++ni) {
        const int j0 = n0 + wn * WN + ni * 8 + 2 * lid;
        const int j1 = j0 + 1;
        const int i  = gid;
#pragma unroll
        for (int t = 0; t < 2; ++t) {
            const int j     = t == 0 ? j0 : j1;
            const float raw = t == 0 ? accum[ni][0] : accum[ni][1];
            if (j >= n_j) { continue; }
            float logit = -CUDART_INF_F;
            if (i < n_i) {
                const int k_abs_max_j = j * kXAttnStride + (kXAttnStride - 1);
                const int q_abs_max_i =
                    base_pos + q_start + min(i * kXAttnStride + (kXAttnStride - 1), q_rows - 1);
                if (k_abs_max_j <= q_abs_max_i) { logit = raw * inv_s; }
            }
            logits[gqa_xattn_logit_index(q_head, n_br, br, i, n_j_cap, j)] = logit;
        }
    }
}

template <typename Geometry, typename Metadata>
__global__ void gqa_xattn_softmax_mass_kernel(Metadata metadata,
                                              const std::int32_t* __restrict__ positions,
                                              std::int32_t width, std::int32_t xattn_min_len,
                                              int n_br, int n_j_cap, int n_kb_cap,
                                              const float* __restrict__ logits,
                                              float* __restrict__ mass) {
    constexpr int Bc            = kGqaPrefillNvfp4Bc;
    constexpr unsigned FullMask = 0xffffffffu;
    __shared__ float red[kGqaPrefillNvfp4Warps];

    const int q_head = static_cast<int>(blockIdx.x);
    const int br     = static_cast<int>(blockIdx.y);
    const int tid    = static_cast<int>(threadIdx.x);
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const int tokens = metadata.valid_tokens(width);
    if (q_head >= Geometry::QHeads || br >= n_br || tokens <= 0) { return; }

    const int base_pos = positions[0];
    const int q_start  = br * kXAttnBlock;
    if (q_start >= tokens) { return; }
    const int q_rows        = min(kXAttnBlock, tokens - q_start);
    const int max_query_abs = base_pos + q_start + q_rows - 1;
    if (max_query_abs + 1 < xattn_min_len) { return; }
    const int n_i    = gqa_xattn_n_i_rows(q_rows);
    const int n_j    = min(n_j_cap, max_query_abs / kXAttnStride + 1);
    const int kb_lim = min(n_kb_cap, max_query_abs / Bc + 1);

    for (int kb = tid; kb < kb_lim; kb += kGqaPrefillNvfp4Threads) {
        mass[gqa_xattn_mass_index(q_head, n_br, br, n_kb_cap, kb)] = 0.0f;
    }
    __syncthreads();
    if (n_i <= 0) { return; }

    auto block_max = [&](float v) -> float {
#pragma unroll
        for (int r = 16; r; r >>= 1) { v = fmaxf(v, __shfl_xor_sync(FullMask, v, r)); }
        if (lane == 0) { red[warp] = v; }
        __syncthreads();
        v = (tid < kGqaPrefillNvfp4Warps) ? red[tid] : -CUDART_INF_F;
        if (warp == 0) {
#pragma unroll
            for (int r = 16; r; r >>= 1) { v = fmaxf(v, __shfl_xor_sync(FullMask, v, r)); }
            if (lane == 0) { red[0] = v; }
        }
        __syncthreads();
        return red[0];
    };
    auto block_sum = [&](float v) -> float {
#pragma unroll
        for (int r = 16; r; r >>= 1) { v += __shfl_xor_sync(FullMask, v, r); }
        if (lane == 0) { red[warp] = v; }
        __syncthreads();
        v = (tid < kGqaPrefillNvfp4Warps) ? red[tid] : 0.0f;
        if (warp == 0) {
#pragma unroll
            for (int r = 16; r; r >>= 1) { v += __shfl_xor_sync(FullMask, v, r); }
            if (lane == 0) { red[0] = v; }
        }
        __syncthreads();
        return red[0];
    };

    for (int i = 0; i < kXAttnIRows; ++i) {
        if (i >= n_i) { continue; }
        float mx = -CUDART_INF_F;
        for (int j = tid; j < n_j; j += kGqaPrefillNvfp4Threads) {
            mx = fmaxf(mx, logits[gqa_xattn_logit_index(q_head, n_br, br, i, n_j_cap, j)]);
        }
        mx = block_max(mx);
        float z = 0.0f;
        for (int j = tid; j < n_j; j += kGqaPrefillNvfp4Threads) {
            const float v = logits[gqa_xattn_logit_index(q_head, n_br, br, i, n_j_cap, j)];
            z += (v == -CUDART_INF_F) ? 0.0f : expf(v - mx);
        }
        z                 = block_sum(z);
        const float inv_z = z > 0.0f ? 1.0f / z : 0.0f;
        for (int j = tid; j < n_j; j += kGqaPrefillNvfp4Threads) {
            const float v = logits[gqa_xattn_logit_index(q_head, n_br, br, i, n_j_cap, j)];
            if (v == -CUDART_INF_F || inv_z == 0.0f) { continue; }
            const int kb = j / (Bc / kXAttnStride);
            if (kb < kb_lim) {
                atomicAdd(&mass[gqa_xattn_mass_index(q_head, n_br, br, n_kb_cap, kb)],
                          expf(v - mx) * inv_z);
            }
        }
        __syncthreads();
    }
}

template <typename Geometry, typename Metadata>
__global__ void gqa_xattn_finish_kernel(Metadata metadata, const std::int32_t* __restrict__ positions,
                                        std::int32_t width, float xattn_tau,
                                        std::int32_t xattn_min_len, int n_br, int n_kb_cap,
                                        const float* __restrict__ mass, std::uint16_t* __restrict__ keep,
                                        int* __restrict__ count) {
    constexpr int Bc = kGqaPrefillNvfp4Bc;

    extern __shared__ __align__(16) unsigned char xattn_finish_smem[];
    float* scores       = reinterpret_cast<float*>(xattn_finish_smem);
    int* sort_ids       = reinterpret_cast<int*>(scores + kGqaPrefillNvfp4RankTiles);
    unsigned char* mark = reinterpret_cast<unsigned char*>(sort_ids + kGqaPrefillNvfp4RankTiles);

    const int q_head = static_cast<int>(blockIdx.x);
    const int br     = static_cast<int>(blockIdx.y);
    const int tid    = static_cast<int>(threadIdx.x);
    const int tokens = metadata.valid_tokens(width);
    if (q_head >= Geometry::QHeads || br >= n_br || tokens <= 0) { return; }

    const int base_pos = positions[0];
    const int q0       = br * kGqaPrefillNvfp4Br;
    if (q0 >= tokens) { return; }
    const int tile_rows     = min(kGqaPrefillNvfp4Br, tokens - q0);
    const int max_query_abs = base_pos + q0 + tile_rows - 1;
    const int key_blocks    = min(max_query_abs / Bc + 1, kGqaPrefillNvfp4RankTiles);
    const int slot          = q_head * n_br + br;
    std::uint16_t* dst      = keep + slot * n_kb_cap;

    const bool identity = (max_query_abs + 1) < xattn_min_len;
    if (identity) {
        for (int kb = tid; kb < key_blocks; kb += kGqaPrefillNvfp4Threads) {
            dst[kb] = static_cast<std::uint16_t>(kb);
        }
        if (tid == 0) { count[slot] = key_blocks; }
        return;
    }

    for (int kb = tid; kb < key_blocks; kb += kGqaPrefillNvfp4Threads) { mark[kb] = 0; }
    __syncthreads();

    constexpr int PagesPerFind = kGqaXattnFindB / Bc;
    const int n_blocks         = (key_blocks + PagesPerFind - 1) / PagesPerFind;
    if (tid == 0) {
        float z = 0.0f;
        for (int b = 0; b < n_blocks; ++b) {
            const int p0 = b * PagesPerFind;
            float m      = mass[gqa_xattn_mass_index(q_head, n_br, br, n_kb_cap, p0)];
            if (p0 + 1 < key_blocks) {
                m += mass[gqa_xattn_mass_index(q_head, n_br, br, n_kb_cap, p0 + 1)];
            }
            scores[b] = m;
            z += m;
        }
        const float inv_z = z > 0.0f ? 1.0f / z : 0.0f;
        for (int b = 0; b < n_blocks; ++b) { scores[b] *= inv_z; }
    }
    __syncthreads();
    const int nsort = n_blocks <= 256 ? 256 : n_blocks <= 512 ? 512 : n_blocks <= 2048 ? 2048 : 4096;
    for (int i = tid; i < nsort; i += kGqaPrefillNvfp4Threads) {
        if (i < n_blocks) {
            sort_ids[i] = i;
        } else {
            scores[i]   = -CUDART_INF_F;
            sort_ids[i] = -1;
        }
    }
    __syncthreads();
    gqa_xattn_sort_desc(scores, sort_ids, nsort, tid);
    if (tid == 0) {
        float kept = 0.0f;
        for (int i = 0; i < nsort; ++i) {
            if (kept >= xattn_tau) { break; }
            const int b   = sort_ids[i];
            const float s = scores[i];
            if (b < 0 || s <= 0.0f) { break; }
            const int p0 = b * PagesPerFind;
            mark[p0]     = 1;
            if (p0 + 1 < key_blocks) { mark[p0 + 1] = 1; }
            kept += s;
        }
        const int forced[3] = {0, min((base_pos + q0) / kGqaXattnFindB, n_blocks - 1),
                               n_blocks - 1};
        for (int f = 0; f < 3; ++f) {
            const int p0 = forced[f] * PagesPerFind;
            mark[p0]     = 1;
            if (p0 + 1 < key_blocks) { mark[p0 + 1] = 1; }
        }
        int n = 0;
        for (int kb = 0; kb < key_blocks; ++kb) {
            if (mark[kb]) { dst[n++] = static_cast<std::uint16_t>(kb); }
        }
        count[slot] = n;
    }
}

template <typename Geometry, typename Metadata>
// Occupancy-1: 16 warps × 128 regs = 65536. 512-thread CTA cannot exceed 128.
__global__ __maxnreg__(128) void gqa_attention_prefill_nvfp4_sparse_kernel(
    const __nv_bfloat16* __restrict__ q, const std::uint8_t* __restrict__ cache_k,
    const std::uint8_t* __restrict__ cache_v, const std::uint8_t* __restrict__ cache_k_scale,
    const std::uint8_t* __restrict__ cache_v_scale, const float* __restrict__ k_mean,
    Metadata metadata, const std::int32_t* __restrict__ positions, float scale,
    __nv_bfloat16* __restrict__ out, std::int32_t width, float keep_frac, float xattn_tau,
    std::int32_t xattn_min_len, GqaS3PrefillDump* dump, const std::uint16_t* xattn_keep_list,
    const int* xattn_keep_count, int xattn_keep_stride, int xattn_n_br) {
    constexpr int D             = kGqaPrefillHeadDim;
    constexpr int Br            = kGqaPrefillNvfp4Br;
    constexpr int Bc            = kGqaPrefillNvfp4Bc;
    constexpr int Groups        = kGqaPrefillNvfp4Groups;
    constexpr int CodeW         = kGqaPrefillNvfp4CodeW;
    constexpr int QKNt          = Bc / 8;
    constexpr int K64s          = kGqaNvfp4K64;
    constexpr int PVNtPerWarp   = D / (kGqaPrefillNvfp4DConsumers * 8);
    constexpr int PVKs          = Bc / 16;
    constexpr int ProducerWarps = kGqaPrefillNvfp4RowTiles;
    constexpr int VWorkerWarps  = kGqaPrefillNvfp4Warps - ProducerWarps;
    constexpr int WorkerThreads = VWorkerWarps * 32;
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;

    static_assert(PVNtPerWarp == 16);

    extern __shared__ __align__(16) unsigned char smem_raw[];
    std::uint8_t* q_codes = smem_raw;
    std::uint8_t* q_scale = q_codes + kGqaPrefillNvfp4QBytes;
    std::uint8_t* k_codes = q_scale + kGqaPrefillNvfp4QScaleBytes;
    std::uint8_t* v_codes = k_codes + kGqaPrefillNvfp4KBytes;
    __nv_bfloat16* v_bf16 = reinterpret_cast<__nv_bfloat16*>(v_codes + kGqaPrefillNvfp4VBytes);
    __nv_bfloat16* p_s    = reinterpret_cast<__nv_bfloat16*>(
        reinterpret_cast<unsigned char*>(v_bf16) + kGqaPrefillNvfp4VStageBytes);
    std::uint8_t* k_scale_s =
        reinterpret_cast<std::uint8_t*>(reinterpret_cast<unsigned char*>(p_s) +
                                        kGqaPrefillNvfp4PBytes);
    std::uint8_t* v_scale_s = k_scale_s + Bc * Groups;
    float* alpha_s          = reinterpret_cast<float*>(v_scale_s + Bc * Groups);
    float* final_l_s        = alpha_s + Br;
    // Ranking overlay (scores/marks/q_mean) occupies the MMA arena before Q is
    // quantized. The compacted keep-list survives in the tail past MMA smem.
    float* proxy_scores = reinterpret_cast<float*>(smem_raw);
    float* xattn_scores1 = proxy_scores + kGqaPrefillNvfp4RankTiles;
    int* sort_ids        = reinterpret_cast<int*>(xattn_scores1 + kGqaPrefillNvfp4RankTiles);
    unsigned char* keep_mark =
        reinterpret_cast<unsigned char*>(sort_ids + kGqaPrefillNvfp4RankTiles);
    // Sparge q_mean (D). XAttn overlays Q bf16 + tile masses later in the branch.
    float* q_mean_s = reinterpret_cast<float*>(keep_mark + kGqaPrefillNvfp4RankTiles);
    unsigned char* keep_tail =
        smem_raw + ((kGqaPrefillNvfp4MmaSmemBytes + 15) & ~15);
    std::uint16_t* keep_u16 = reinterpret_cast<std::uint16_t*>(keep_tail);
    int* k_keep_count       = reinterpret_cast<int*>(keep_u16 + kGqaPrefillNvfp4RankTiles);
    // Block-reduce scratch in the keep-list tail (overwritten by compact after ranking).
    float* red_score = reinterpret_cast<float*>(keep_tail);
    int* red_kb      = reinterpret_cast<int*>(keep_tail + 16 * sizeof(float));

    const int q_block = static_cast<int>(blockIdx.x);
    const int q_head  = static_cast<int>(blockIdx.y);
    const int tid     = static_cast<int>(threadIdx.x);
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int q0      = q_block * Br;
    const int kv_head = q_head / Geometry::GroupSize;
    const int tokens  = metadata.valid_tokens(width);
    if (q_head >= Geometry::QHeads || q0 >= width) { return; }
    if (q0 >= tokens) {
        gqa_prefill_zero_output_rows<Geometry>(out, q_head, q0, min(q0 + Br, width), tid,
                                               kGqaPrefillNvfp4Threads);
        return;
    }
    const int base_pos              = positions[0];
    const std::int32_t* block_table = metadata.block_table();

    const int tile_rows     = min(Br, tokens - q0);
    const int max_query_abs = base_pos + q0 + tile_rows - 1;
    const int key_blocks    = min(max_query_abs / Bc + 1, kGqaPrefillNvfp4RankTiles);

    auto issue_kv_tile = [&](int tile_k0) {
        const int physical_page = block_table[tile_k0 >> kPagedKVPageShift];
        for (int key_l = tid; key_l < Bc; key_l += kGqaPrefillNvfp4Threads) {
            const int key = tile_k0 + key_l;
            if (key <= max_query_abs) {
                const std::int64_t off =
                    gqa_nvfp4_scale_index<Geometry>(physical_page, kv_head, 0, key_l);
                ninfer::ops::cp_async<16>(&k_scale_s[key_l * Groups], &cache_k_scale[off]);
                ninfer::ops::cp_async<16>(&v_scale_s[key_l * Groups], &cache_v_scale[off]);
            } else {
                store_vec(&k_scale_s[key_l * Groups], make_int4(0, 0, 0, 0));
                store_vec(&v_scale_s[key_l * Groups], make_int4(0, 0, 0, 0));
            }
        }
        for (int chunk = tid; chunk < Bc * (CodeW / 16); chunk += kGqaPrefillNvfp4Threads) {
            const int key_l   = chunk / (CodeW / 16);
            const int seg     = chunk - key_l * (CodeW / 16);
            const int logical = seg * 16;
            const int key     = tile_k0 + key_l;
            const int phys    = gqa_nvfp4_swizzle_byte(key_l, logical);
            std::uint8_t* kd  = &k_codes[key_l * CodeW + phys];
            std::uint8_t* vd  = &v_codes[key_l * CodeW + logical];
            if (key <= max_query_abs) {
                const std::int64_t off =
                    gqa_nvfp4_code_index<Geometry>(physical_page, kv_head, logical, key_l);
                cp_async<16, Cache::cg>(kd, &cache_k[off]);
                cp_async<16, Cache::cg>(vd, &cache_v[off]);
            } else {
                store_vec(kd, make_int4(0, 0, 0, 0));
                store_vec(vd, make_int4(0, 0, 0, 0));
            }
        }

        ninfer::ops::cp_commit();
    };

    (void)xattn_tau;
    (void)xattn_min_len;
    const bool sparge = keep_frac > 0.0f && keep_frac < 1.0f && k_mean != nullptr;

    // Same keep-set as a lane-0 left-to-right scan: higher score, then lower kb.
    auto reduce_argmax = [&](float best, int best_kb, float loser_score, int loser_kb) -> int {
#pragma unroll
        for (int r = 16; r; r >>= 1) {
            const float ov = __shfl_xor_sync(FullMask, best, r);
            const int ok   = __shfl_xor_sync(FullMask, best_kb, r);
            if (ov > best || (ov == best && ok >= 0 && (best_kb < 0 || ok < best_kb))) {
                best    = ov;
                best_kb = ok;
            }
        }
        if (lane == 0) {
            red_score[warp] = best;
            red_kb[warp]    = best_kb;
        }
        __syncthreads();
        if (warp == 0) {
            best    = (lane < kGqaPrefillNvfp4Warps) ? red_score[lane] : loser_score;
            best_kb = (lane < kGqaPrefillNvfp4Warps) ? red_kb[lane] : loser_kb;
#pragma unroll
            for (int r = 16; r; r >>= 1) {
                const float ov = __shfl_xor_sync(FullMask, best, r);
                const int ok   = __shfl_xor_sync(FullMask, best_kb, r);
                if (ov > best || (ov == best && ok >= 0 && (best_kb < 0 || ok < best_kb))) {
                    best    = ov;
                    best_kb = ok;
                }
            }
            if (lane == 0) {
                red_score[0] = best;
                red_kb[0]    = best_kb;
            }
        }
        __syncthreads();
        return red_kb[0];
    };

    auto argmax_unmarked = [&](int n) -> int {
        float best  = -CUDART_INF_F;
        int best_kb = 0;
        for (int kb = tid; kb < n; kb += kGqaPrefillNvfp4Threads) {
            if (keep_mark[kb]) { continue; }
            const float s = proxy_scores[kb];
            if (s > best || (s == best && kb < best_kb)) {
                best    = s;
                best_kb = kb;
            }
        }
        return reduce_argmax(best, best_kb, -CUDART_INF_F, 0);
    };

    auto compact_keep = [&]() {
        if (tid == 0) {
            int n = 0;
            for (int kb = 0; kb < key_blocks; ++kb) {
                if (keep_mark[kb]) { keep_u16[n++] = static_cast<std::uint16_t>(kb); }
            }
            *k_keep_count = n;
        }
    };

    for (int kb = tid; kb < key_blocks; kb += kGqaPrefillNvfp4Threads) { keep_mark[kb] = 0; }
    __syncthreads();

    if (xattn_keep_list != nullptr && xattn_keep_count != nullptr && xattn_n_br > 0) {
        const int slot = q_head * xattn_n_br + q_block;
        const int nkeep = min(xattn_keep_count[slot], key_blocks);
        const std::uint16_t* src =
            xattn_keep_list + static_cast<std::int64_t>(slot) * xattn_keep_stride;
        for (int i = tid; i < nkeep; i += kGqaPrefillNvfp4Threads) { keep_u16[i] = src[i]; }
        if (tid == 0) { *k_keep_count = nkeep; }
    } else if (sparge) {
        if (tid < D) {
            float acc = 0.0f;
            for (int row = 0; row < tile_rows; ++row) {
                acc += __bfloat162float(
                    __ldg(&q[gqa_prefill_q_index<Geometry>(q_head, tid, q0 + row)]));
            }
            q_mean_s[tid] = acc / static_cast<float>(tile_rows);
        }
        __syncthreads();
        for (int kb = warp; kb < key_blocks; kb += kGqaPrefillNvfp4Warps) {
            const int physical_page = block_table[kb];
            float acc               = 0.0f;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int d            = i * 32 + lane;
                const std::int64_t off = paged_kv_element_offset<4, Geometry::KVHeads>(
                    physical_page, kv_head, d >> 2, d & 3);
                acc += q_mean_s[d] * __ldg(&k_mean[off]);
            }
#pragma unroll
            for (int r = 16; r; r >>= 1) { acc += __shfl_xor_sync(FullMask, acc, r); }
            if (lane == 0) { proxy_scores[kb] = acc; }
        }
        __syncthreads();
        const int topk     = min(max(1, static_cast<int>(keep_frac * key_blocks)), key_blocks);
        const int k_sinks  = max(1, static_cast<int>(key_blocks * keep_frac * 0.2f));
        const int k_window = max(1, static_cast<int>(key_blocks * keep_frac * 0.4f));
        for (int sel = 0; sel < topk; ++sel) {
            const int best_kb = argmax_unmarked(key_blocks);
            if (tid == 0) { keep_mark[best_kb] = 1; }
            __syncthreads();
        }
        for (int kb = tid; kb < key_blocks; kb += kGqaPrefillNvfp4Threads) {
            if (kb < k_sinks || kb >= key_blocks - k_window) { keep_mark[kb] = 1; }
        }
        __syncthreads();
        compact_keep();
    } else {
        for (int kb = tid; kb < key_blocks; kb += kGqaPrefillNvfp4Threads) {
            keep_u16[kb] = static_cast<std::uint16_t>(kb);
        }
        if (tid == 0) { *k_keep_count = key_blocks; }
    }
    __syncthreads();
    if (q_block == 0 && dump != nullptr && dump->keep_list != nullptr &&
        dump->tile_count != nullptr && warp == 0 && lane == 0) {
        const int ncopy = min(*k_keep_count, dump->max_tiles);
        for (int i = 0; i < ncopy; ++i) {
            dump->keep_list[q_head * dump->max_tiles + i] = keep_u16[i];
        }
        dump->tile_count[q_head] = *k_keep_count;
    }
    const int n_tiles = *k_keep_count;
    __syncthreads();
    for (int i = tid; i < Br * CodeW; i += kGqaPrefillNvfp4Threads) { q_codes[i] = 0; }
    for (int i = tid; i < Br * Groups; i += kGqaPrefillNvfp4Threads) { q_scale[i] = 0; }
    __syncthreads();
    for (int unit = tid; unit < Br * Groups; unit += kGqaPrefillNvfp4Threads) {
        const int row = unit / Groups;
        const int grp = unit - row * Groups;
        std::uint32_t lo = 0, hi = 0;
        std::uint8_t sc = 0;
        if (row < tile_rows) {
            gqa_nvfp4_quantize_bf16x16(
                &q[gqa_prefill_q_index<Geometry>(q_head, grp * kGqaNvfp4Group, q0 + row)], lo, hi,
                sc);
        }
        const int phys = gqa_nvfp4_swizzle_byte(row, grp * 8);
        *reinterpret_cast<std::uint32_t*>(q_codes + row * CodeW + phys)     = lo;
        *reinterpret_cast<std::uint32_t*>(q_codes + row * CodeW + phys + 4) = hi;
        q_scale[row * Groups + grp]                                         = sc;
    }
    __syncthreads();
    if (n_tiles > 0) {
        issue_kv_tile(static_cast<int>(keep_u16[0]) * Bc);
        ninfer::ops::cp_wait<0>();
    }
    __syncthreads();

    const int gid           = lane >> 2;
    const int lid           = lane & 3;
    const int a_matrix      = lane >> 3;
    const int a_row_offset  = (lane & 7) + ((a_matrix & 1) << 3);
    const int a_column_byte = (a_matrix >> 1) * 16;
    const int b_row_offset  = lane & 7;
    const int b_column_byte = ((lane >> 3) & 1) * 16;
    const int sfa_row       = ((lane & 1) << 3) | (lane >> 2);
    const int sfb_row       = lane >> 2;
    const int a_coloff      = (a_matrix >> 1) << 3;
    const int b_rin         = lane & 7;
    const int b_koff        = ((lane >> 3) & 1) << 3;

    float acc[PVNtPerWarp][4];
#pragma unroll
    for (int n = 0; n < PVNtPerWarp; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[n][i] = 0.0f; }
    }
    float running_m0     = -CUDART_INF_F;
    float running_m1     = -CUDART_INF_F;
    float running_l0     = 0.0f;
    float running_l1     = 0.0f;
    const float scale_l2 = scale * Log2E;

    for (int ki = 0; ki < n_tiles; ++ki) {
        const int kb = static_cast<int>(keep_u16[ki]);
        const int k0 = kb * Bc;
        if (warp < ProducerWarps) {
            const int row_base = warp * 16;
            float score[QKNt][4];
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0f;
            }

#pragma unroll
            for (int k64 = 0; k64 < K64s; ++k64) {
                const int row           = row_base + a_row_offset;
                const int logical_byte  = k64 * 32 + a_column_byte;
                const int physical_byte = gqa_nvfp4_swizzle_byte(row, logical_byte);
                unsigned af[4];
                ldmatrix_x4(af[0], af[1], af[2], af[3],
                            smem_addr(q_codes + row * CodeW + physical_byte));
                const int scale_row = row_base + sfa_row;
                // MMA block_scale only consumes SFA from even-pair lanes and SFB from
                // lane%4==0; the other lanes duplicate those rows.
                unsigned sfa = 0;
                if ((lane & 2) == 0) {
                    sfa = *reinterpret_cast<const unsigned*>(
                        &q_scale[scale_row * Groups + k64 * 4]);
                }
#pragma unroll
                for (int nt = 0; nt < QKNt; ++nt) {
                    const int brow       = nt * 8 + b_row_offset;
                    const int b_logical  = k64 * 32 + b_column_byte;
                    const int b_physical = gqa_nvfp4_swizzle_byte(brow, b_logical);
                    unsigned bf[2];
                    ldmatrix_x2(bf[0], bf[1], smem_addr(k_codes + brow * CodeW + b_physical));
                    const int b_scale_row = nt * 8 + sfb_row;
                    unsigned sfb          = 0;
                    if ((lane & 3) == 0) {
                        sfb = *reinterpret_cast<const unsigned*>(
                            &k_scale_s[b_scale_row * Groups + k64 * 4]);
                    }
                    mma_nvfp4_e4m3(score[nt][0], score[nt][1], score[nt][2], score[nt][3], af[0],
                                   af[1], af[2], af[3], bf[0], bf[1], sfa, sfb);
                }
            }

            const int row0             = row_base + gid;
            const int row1             = row0 + 8;
            const int qabs0            = row0 < tile_rows ? base_pos + q0 + row0 : -1;
            const int qabs1            = row1 < tile_rows ? base_pos + q0 + row1 : -1;
            const bool full_score_tile = q0 + Br <= tokens && k0 + Bc - 1 <= base_pos + q0;
            float bm0                  = -CUDART_INF_F;
            float bm1                  = -CUDART_INF_F;
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int key0 = k0 + nt * 8 + 2 * lid;
                const int key1 = key0 + 1;
                if (!full_score_tile) {
                    score[nt][0] = key0 <= qabs0 ? score[nt][0] : -CUDART_INF_F;
                    score[nt][1] = key1 <= qabs0 ? score[nt][1] : -CUDART_INF_F;
                    score[nt][2] = key0 <= qabs1 ? score[nt][2] : -CUDART_INF_F;
                    score[nt][3] = key1 <= qabs1 ? score[nt][3] : -CUDART_INF_F;
                }
                bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
                bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
            }
            bm0 = warp_max<4>(bm0, FullMask);
            bm1 = warp_max<4>(bm1, FullMask);

            const float nm0        = fmaxf(running_m0, bm0);
            const float nm1        = fmaxf(running_m1, bm1);
            const float nm0_scaled = nm0 * scale_l2;
            const float nm1_scaled = nm1 * scale_l2;
            const float alpha0     = running_m0 == -CUDART_INF_F
                                         ? 0.0f
                                         : exp2_approx(__fmaf_rn(running_m0, scale_l2, -nm0_scaled));
            const float alpha1     = running_m1 == -CUDART_INF_F
                                         ? 0.0f
                                         : exp2_approx(__fmaf_rn(running_m1, scale_l2, -nm1_scaled));
            float bl0              = 0.0f;
            float bl1              = 0.0f;
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int col0  = nt * 8 + 2 * lid;
                const int col1  = col0 + 1;
                const float p00 = score[nt][0] > -CUDART_INF_F
                                      ? exp2_approx(__fmaf_rn(score[nt][0], scale_l2, -nm0_scaled))
                                      : 0.0f;
                const float p01 = score[nt][1] > -CUDART_INF_F
                                      ? exp2_approx(__fmaf_rn(score[nt][1], scale_l2, -nm0_scaled))
                                      : 0.0f;
                const float p10 = score[nt][2] > -CUDART_INF_F
                                      ? exp2_approx(__fmaf_rn(score[nt][2], scale_l2, -nm1_scaled))
                                      : 0.0f;
                const float p11 = score[nt][3] > -CUDART_INF_F
                                      ? exp2_approx(__fmaf_rn(score[nt][3], scale_l2, -nm1_scaled))
                                      : 0.0f;
                bl0 += p00 + p01;
                bl1 += p10 + p11;
                p_s[row0 * Bc + gqa_prefill_swz(row0, col0)] = __float2bfloat16(p00);
                p_s[row0 * Bc + gqa_prefill_swz(row0, col1)] = __float2bfloat16(p01);
                p_s[row1 * Bc + gqa_prefill_swz(row1, col0)] = __float2bfloat16(p10);
                p_s[row1 * Bc + gqa_prefill_swz(row1, col1)] = __float2bfloat16(p11);
            }
            bl0        = warp_sum<4>(bl0, FullMask);
            bl1        = warp_sum<4>(bl1, FullMask);
            running_l0 = __fmaf_rn(running_l0, alpha0, bl0);
            running_l1 = __fmaf_rn(running_l1, alpha1, bl1);
            running_m0 = nm0;
            running_m1 = nm1;
            if (lid == 0) {
                alpha_s[row0] = alpha0;
                alpha_s[row1] = alpha1;
            }
        } else if (warp < ProducerWarps + VWorkerWarps) {
            const int worker_tid = tid - ProducerWarps * 32;
#pragma unroll 1
            for (int chunk = worker_tid; chunk < Bc * (D / 8); chunk += WorkerThreads) {
                const int key_l    = chunk / (D / 8);
                const int dc       = chunk - key_l * (D / 8);
                const int d        = dc * 8;
                const int key      = k0 + key_l;
                __nv_bfloat16* dst = &v_bf16[key_l * D + gqa_prefill_swz(key_l, d)];
                if (key <= max_query_abs) {
                    const int grp = d >> 4;
                    const float vs =
                        detail::decode_nvfp4_e4m3(v_scale_s[key_l * Groups + grp]);
                    store_vec(dst, gqa_nvfp4_dequant_bf16x8(&v_codes[key_l * CodeW + d / 2], vs));
                } else {
                    store_vec(dst, make_int4(0, 0, 0, 0));
                }
            }
        }
        __syncthreads();

        const bool has_next = ki + 1 < n_tiles;
        if (has_next) { issue_kv_tile(static_cast<int>(keep_u16[ki + 1]) * Bc); }

        const int row_tile = warp % kGqaPrefillNvfp4RowTiles;
        const int d_slice  = warp / kGqaPrefillNvfp4RowTiles;
        const int row_base = row_tile * 16;
        const float alpha0 = alpha_s[row_base + gid];
        const float alpha1 = alpha_s[row_base + gid + 8];
#pragma unroll
        for (int n = 0; n < PVNtPerWarp; ++n) {
            acc[n][0] *= alpha0;
            acc[n][1] *= alpha0;
            acc[n][2] *= alpha1;
            acc[n][3] *= alpha1;
        }

#pragma unroll
        for (int k = 0; k < PVKs; ++k) {
            unsigned pf[4];
            const int pcol = k * 16 + a_coloff;
            ldmatrix_x4(pf[0], pf[1], pf[2], pf[3],
                        smem_addr(&p_s[(row_base + a_row_offset) * Bc +
                                       gqa_prefill_swz(row_base + a_row_offset, pcol)]));
#pragma unroll
            for (int n = 0; n < PVNtPerWarp; ++n) {
                const int global_n = d_slice * PVNtPerWarp + n;
                unsigned vf[2];
                const int vrow = k * 16 + b_koff + b_rin;
                const int vcol = global_n * 8;
                ldmatrix_x2_t(vf[0], vf[1],
                              smem_addr(&v_bf16[vrow * D + gqa_prefill_swz(vrow, vcol)]));
                mma_bf16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], pf[0], pf[1], pf[2], pf[3],
                         vf[0], vf[1]);
            }
        }
        if (has_next) { ninfer::ops::cp_wait<0>(); }
        __syncthreads();
    }

    if (warp < ProducerWarps && lid == 0) {
        const int row0  = warp * 16 + gid;
        const int row1  = row0 + 8;
        final_l_s[row0] = running_l0;
        final_l_s[row1] = running_l1;
    }
    __syncthreads();

    const int row_tile = warp % kGqaPrefillNvfp4RowTiles;
    const int d_slice  = warp / kGqaPrefillNvfp4RowTiles;
    const int row_base = row_tile * 16;
    const int row0     = row_base + gid;
    const int row1     = row0 + 8;
    const float inv_l0 = final_l_s[row0] > 0.0f ? __frcp_rn(final_l_s[row0]) : 0.0f;
    const float inv_l1 = final_l_s[row1] > 0.0f ? __frcp_rn(final_l_s[row1]) : 0.0f;
#pragma unroll
    for (int n = 0; n < PVNtPerWarp; ++n) {
        const int d0 = (d_slice * PVNtPerWarp + n) * 8 + 2 * lid;
        if (row0 < tile_rows) {
            *reinterpret_cast<unsigned*>(
                &out[gqa_prefill_q_index<Geometry>(q_head, d0, q0 + row0)]) =
                pack_bf16x2(acc[n][0] * inv_l0, acc[n][1] * inv_l0);
        }
        if (row1 < tile_rows) {
            *reinterpret_cast<unsigned*>(
                &out[gqa_prefill_q_index<Geometry>(q_head, d0, q0 + row1)]) =
                pack_bf16x2(acc[n][2] * inv_l1, acc[n][3] * inv_l1);
        }
    }
}

} // namespace ninfer::ops
