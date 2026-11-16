#pragma once

// Implements: include/ninfer/ops/dflash2_path_select.h

#include "ops/common/math.h"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"
#include "ninfer/ops/sampling.h"

#include <cuda_bf16.h>
#include <climits>
#include <cstdint>
#include <math_constants.h>

namespace ninfer::ops {

inline constexpr int kDflash2PathSelectBlock            = 256;
inline constexpr int kWarpSizeDevice                    = 32;
inline constexpr int kDflash2PathSelectK                = 16;
inline constexpr int kDflash2PathSelectRank             = 256;
inline constexpr int kDflash2PathSelectSuccStride       = kDflash2PathSelectRank + 2;
inline constexpr int kDflash2PathSelectGemmBlock        = 256;
inline constexpr int kDflash2PathSelectRngPurposeDevice = 16;
// Mirrors kDflash2TreeMaxWidth in the public header, which this device header cannot include.
inline constexpr int kDflash2TreeMaxWidthDevice = 16;

struct Dflash2CodebookDevice {
    const __nv_bfloat16* bf16        = nullptr;
    const std::uint8_t* nvfp4_codes  = nullptr;
    const std::uint8_t* nvfp4_scales = nullptr;
    float inv_dw                     = 0.0f;
};

__device__ __forceinline__ float dflash2_e2m1_value(std::uint8_t code) {
    constexpr float kMags[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float mag          = kMags[code & 7u];
    return (code & 8u) ? -mag : mag;
}

__device__ __forceinline__ __nv_bfloat16 dflash2_codebook_load(Dflash2CodebookDevice book,
                                                               int token, int rank) {
    if (token < 0) { token = 0; }
    if (book.bf16 != nullptr) {
        return book.bf16[static_cast<std::int64_t>(token) * kDflash2PathSelectRank + rank];
    }
    constexpr int kTiles = kDflash2PathSelectRank / 64;
    const std::uint8_t packed =
        book.nvfp4_codes[static_cast<std::int64_t>(token) * (kDflash2PathSelectRank / 2) +
                         rank / 2];
    const std::uint8_t nibble    = (rank & 1) != 0 ? static_cast<std::uint8_t>(packed >> 4)
                                                   : static_cast<std::uint8_t>(packed & 0x0f);
    const int group              = rank / 16;
    const int m_tile             = token / 128;
    const int row_inner          = token - m_tile * 128;
    const std::int64_t scale_off = static_cast<std::int64_t>(m_tile * kTiles + group / 4) * 512 +
                                   static_cast<std::int64_t>(row_inner & 31) * 16 +
                                   static_cast<std::int64_t>(row_inner >> 5) * 4 + (group & 3);
    const float value = dflash2_e2m1_value(nibble) *
                        detail::decode_nvfp4_e4m3(book.nvfp4_scales[scale_off]) * book.inv_dw;
    return __float2bfloat16_rn(value);
}

__device__ __forceinline__ float dflash2_path_select_block_sum(float value) {
    __shared__ float partial[kDflash2PathSelectGemmBlock];
    partial[threadIdx.x] = value;
    __syncthreads();
    for (int stride = kDflash2PathSelectGemmBlock / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) { partial[threadIdx.x] += partial[threadIdx.x + stride]; }
        __syncthreads();
    }
    return partial[0];
}

__global__ void dflash2_path_select_bf16_gemv_kernel(const __nv_bfloat16* x,
                                                     const __nv_bfloat16* weight,
                                                     __nv_bfloat16* out, std::int32_t n_rows,
                                                     std::int32_t k_rows) {
    const std::int32_t row = static_cast<std::int32_t>(blockIdx.x);
    const std::int32_t col = static_cast<std::int32_t>(blockIdx.y);
    if (row >= n_rows) { return; }

    const std::int64_t k64     = static_cast<std::int64_t>(k_rows);
    const __nv_bfloat16* w     = weight + static_cast<std::int64_t>(row) * k64;
    const __nv_bfloat16* col_x = x + static_cast<std::int64_t>(col) * k64;
    float acc                  = 0.0f;
    for (std::int32_t k = static_cast<std::int32_t>(threadIdx.x); k < k_rows;
         k += kDflash2PathSelectGemmBlock) {
        acc += __bfloat162float(w[k]) * __bfloat162float(col_x[k]);
    }
    acc = dflash2_path_select_block_sum(acc);
    if (threadIdx.x == 0) {
        out[static_cast<std::int64_t>(col) * n_rows + row] = __float2bfloat16_rn(acc);
    }
}

__device__ __forceinline__ unsigned long long dflash2_path_select_splitmix64(unsigned long long x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

__device__ __forceinline__ float dflash2_path_select_uniform(unsigned long long seed, int position,
                                                             int purpose, unsigned int hop) {
    unsigned long long key = seed;
    key                    = dflash2_path_select_splitmix64(
        key ^ (static_cast<unsigned long long>(static_cast<unsigned int>(position)) *
               0xD1B54A32D192ED03ull));
    key = dflash2_path_select_splitmix64(
        key ^ (static_cast<unsigned long long>(static_cast<unsigned int>(purpose)) << 21) ^
        (static_cast<unsigned long long>(hop) * 0x2545F4914F6CDD1Dull));
    const unsigned int bits = static_cast<unsigned int>(key >> 40);
    return static_cast<float>(bits) * (1.0f / 16777216.0f);
}

// Column top-16: dflash2_column_topk_split_kernel then dflash2_column_topk_merge_kernel.
//
// Selection keys. Each represented logit maps to one 64-bit key whose unsigned order is the Op's
// rank order (greater value first, lower row first among equal values):
//   bits 48..33  16-bit order key of the BF16 value: bits | 0x8000 for a positive value and
//                ~bits & 0xffff for a negative one, with -0 folded onto +0 so equal values tie.
//                Finite values land in [0x0080, 0xff7f], so a finite logit's key is never 0;
//   bits 32..1   ~row as 32 bits, so the lower row of two equal values has the greater key;
//   bit 0        set for a -0 logit, restoring the represented sign of zero on decode. Rows are
//                unique, so this bit never decides an order.
// Non-finite logits and empty slots have key 0 and are never selected. Distinct rows give
// distinct keys, so the top-16 is the 16 greatest nonzero keys without ties.
inline constexpr int kDflash2TopkWarps = kDflash2PathSelectBlock / kWarpSizeDevice;
// Keys each thread holds, loaded as 16-byte groups of eight BF16 logits.
inline constexpr int kDflash2TopkItems     = 16;
inline constexpr int kDflash2TopkRowGroup  = 8;
inline constexpr int kDflash2TopkBlockKeys = kDflash2PathSelectBlock * kDflash2TopkItems;
// Mirrors kDflash2PathSelectMaxLogitRows in the public header, which this device header cannot
// include; the launcher rejects a larger V, so split scratch sized from it cannot overflow. The
// merge block holds every split's 16 keys, so the split count is bounded by
// kDflash2TopkBlockKeys / 16.
inline constexpr int kDflash2TopkMaxLogitRows = 262144;
inline constexpr int kDflash2TopkMaxSplits =
    (kDflash2TopkMaxLogitRows + kDflash2TopkBlockKeys - 1) / kDflash2TopkBlockKeys;
static_assert(kDflash2PathSelectBlock == 256 && kDflash2TopkWarps == 8);
static_assert(kDflash2TopkItems % kDflash2TopkRowGroup == 0 && kDflash2TopkItems % 2 == 0);
static_assert(kDflash2TopkMaxSplits * kDflash2PathSelectK <= kDflash2TopkBlockKeys);

__device__ __forceinline__ std::uint64_t dflash2_topk_key(std::uint16_t bits, int row) {
    if ((bits & 0x7f80u) == 0x7f80u) { return 0; }
    const std::uint64_t negative_zero = bits == 0x8000u ? 1u : 0u;
    if (negative_zero != 0) { bits = 0; }
    const std::uint32_t order =
        (bits & 0x8000u) != 0 ? (~static_cast<std::uint32_t>(bits) & 0xffffu) : (bits | 0x8000u);
    return (static_cast<std::uint64_t>(order) << 33) |
           (static_cast<std::uint64_t>(~static_cast<std::uint32_t>(row)) << 1) | negative_zero;
}

__device__ __forceinline__ int dflash2_topk_key_row(std::uint64_t key) {
    return static_cast<int>(~static_cast<std::uint32_t>(key >> 1));
}

__device__ __forceinline__ float dflash2_topk_key_value(std::uint64_t key) {
    const auto order   = static_cast<std::uint32_t>(key >> 33);
    std::uint32_t bits = (order & 0x8000u) != 0 ? (order & 0x7fffu) : (~order & 0xffffu);
    if ((key & 1u) != 0) { bits = 0x8000u; }
    return __uint_as_float(bits << 16);
}

__device__ __forceinline__ std::uint64_t dflash2_topk_max(std::uint64_t a, std::uint64_t b) {
    return a > b ? a : b;
}

__device__ __forceinline__ std::uint64_t dflash2_topk_min(std::uint64_t a, std::uint64_t b) {
    return a < b ? a : b;
}

// Bitonic sort of one key per lane; lane l returns the l-th greatest key of the warp.
__device__ __forceinline__ std::uint64_t dflash2_warp_sort_desc(std::uint64_t value, int lane) {
#pragma unroll
    for (int size = 2; size <= kWarpSizeDevice; size <<= 1) {
#pragma unroll
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            const std::uint64_t other = __shfl_xor_sync(0xffffffffu, value, stride);
            const bool descending     = (lane & size) == 0;
            const bool lower          = (lane & stride) == 0;
            value                     = lower == descending ? dflash2_topk_max(value, other)
                                                            : dflash2_topk_min(value, other);
        }
    }
    return value;
}

// Bitonic merge of a warp-wide bitonic sequence into descending lane order.
__device__ __forceinline__ std::uint64_t dflash2_warp_merge_desc(std::uint64_t value, int lane) {
#pragma unroll
    for (int stride = kWarpSizeDevice / 2; stride > 0; stride >>= 1) {
        const std::uint64_t other = __shfl_xor_sync(0xffffffffu, value, stride);
        value =
            (lane & stride) == 0 ? dflash2_topk_max(value, other) : dflash2_topk_min(value, other);
    }
    return value;
}

// Merges two descending lists whose 16 greatest keys sit in lanes 0..15 of `upper` and `lower`.
// Lanes 0..15 return the 16 greatest keys of both, descending.
__device__ __forceinline__ std::uint64_t dflash2_warp_merge_top16(std::uint64_t upper,
                                                                  std::uint64_t lower, int lane) {
    const std::uint64_t mirrored = __shfl_sync(0xffffffffu, lower, kWarpSizeDevice - 1 - lane);
    return dflash2_warp_merge_desc(lane < kDflash2PathSelectK ? upper : mirrored, lane);
}

struct Dflash2TopkShared {
    std::uint64_t warp_top[kDflash2TopkWarps * kDflash2PathSelectK];
    std::uint64_t candidates[kDflash2PathSelectK * kDflash2TopkItems];
    int count;
};

// Exact top-16 of the block's keys, kDflash2TopkItems per thread (0 = empty).
//
// threshold is the 16th greatest per-thread maximum. Sixteen distinct keys reach it, so every
// top-16 key is >= threshold, and a key >= threshold lies in one of the exactly 16 threads whose
// maximum reaches it. When fewer than 16 threads hold a nonzero key, threshold is 0 and those at
// most 15 threads supply every nonzero key. Either way at most 16 * kDflash2TopkItems keys, one
// per thread of the block, become candidates, and a candidate's rank among the candidates is its
// rank in the block. emit(rank, key) runs once for every rank in [0,16) on the thread that owns
// it; ranks beyond the number of nonzero keys emit key 0.
//
// Synchronization: one barrier publishes the warp lists, every warp derives the same threshold
// from them, and a second barrier publishes the compacted candidates.
template <class Emit>
__device__ __forceinline__ void dflash2_block_top16(const std::uint64_t (&keys)[kDflash2TopkItems],
                                                    Dflash2TopkShared& shared, Emit&& emit) {
    static_assert(kDflash2PathSelectK * kDflash2TopkItems <= kDflash2PathSelectBlock);
    const int tid  = static_cast<int>(threadIdx.x);
    const int lane = tid % kWarpSizeDevice;
    const int warp = tid / kWarpSizeDevice;

    std::uint64_t thread_max = keys[0];
#pragma unroll
    for (int item = 1; item < kDflash2TopkItems; ++item) {
        thread_max = dflash2_topk_max(thread_max, keys[item]);
    }
    const std::uint64_t warp_sorted = dflash2_warp_sort_desc(thread_max, lane);
    if (lane < kDflash2PathSelectK) {
        shared.warp_top[warp * kDflash2PathSelectK + lane] = warp_sorted;
    }
    if (tid == 0) { shared.count = 0; }
    __syncthreads();

    // Register r holds warp list 2r descending in lanes 0..15 and list 2r+1 ascending in lanes
    // 16..31, a bitonic sequence; three merge levels leave the block's 16 greatest thread maxima
    // in lanes 0..15.
    std::uint64_t level[kDflash2TopkWarps / 2];
#pragma unroll
    for (int r = 0; r < kDflash2TopkWarps / 2; ++r) {
        const int base = 2 * r * kDflash2PathSelectK;
        const int slot =
            lane < kDflash2PathSelectK ? base + lane : base + 3 * kDflash2PathSelectK - 1 - lane;
        level[r] = dflash2_warp_merge_desc(shared.warp_top[slot], lane);
    }
    const std::uint64_t first     = dflash2_warp_merge_top16(level[0], level[1], lane);
    const std::uint64_t second    = dflash2_warp_merge_top16(level[2], level[3], lane);
    const std::uint64_t threshold = __shfl_sync(
        0xffffffffu, dflash2_warp_merge_top16(first, second, lane), kDflash2PathSelectK - 1);

#pragma unroll
    for (int item = 0; item < kDflash2TopkItems; ++item) {
        if (keys[item] != 0 && keys[item] >= threshold) {
            shared.candidates[atomicAdd(&shared.count, 1)] = keys[item];
        }
    }
    __syncthreads();

    const int count = shared.count;
    if (tid < count) {
        const std::uint64_t key = shared.candidates[tid];
        int rank                = 0;
        for (int other = 0; other < count; ++other) {
            rank += shared.candidates[other] > key ? 1 : 0;
        }
        if (rank < kDflash2PathSelectK) { emit(rank, key); }
    } else if (tid < kDflash2PathSelectK) {
        emit(tid, std::uint64_t{0});
    }
}

__device__ __forceinline__ std::int64_t dflash2_column_index(int tokens, int t, int b) {
    return static_cast<std::int64_t>(b) * tokens + t;
}

// grid (splits, T, B), block 256, splits = ceil(V / kDflash2TopkBlockKeys). CTA (s,t,b) reduces
// rows [s * kDflash2TopkBlockKeys, (s + 1) * kDflash2TopkBlockKeys) of column (t,b) to its 16
// greatest keys, written in rank order to split_keys[((b*T + t) * splits + s) * 16 + rank].
// Thread tid holds row groups s * kDflash2TopkBlockKeys + 8 * (g * 256 + tid) + [0,8) for
// g < kDflash2TopkItems / 8, so each group load is coalesced across the block. vector_rows
// requires vocab % 8 == 0 and a 16-byte aligned logits base: every group then lies entirely
// inside or outside the column and loads as one 16-byte vector. Otherwise rows load one by one.
__launch_bounds__(kDflash2PathSelectBlock) __global__
    void dflash2_column_topk_split_kernel(const __nv_bfloat16* logits, std::uint64_t* split_keys,
                                          std::int32_t vocab, std::int32_t splits,
                                          bool vector_rows) {
    __shared__ Dflash2TopkShared shared;
    const int split  = static_cast<int>(blockIdx.x);
    const int tokens = static_cast<int>(gridDim.y);
    const int tid    = static_cast<int>(threadIdx.x);
    const auto column =
        dflash2_column_index(tokens, static_cast<int>(blockIdx.y), static_cast<int>(blockIdx.z));
    const auto* column_bits =
        reinterpret_cast<const std::uint16_t*>(logits) + column * static_cast<std::int64_t>(vocab);

    std::uint64_t keys[kDflash2TopkItems];
#pragma unroll
    for (int group = 0; group < kDflash2TopkItems / kDflash2TopkRowGroup; ++group) {
        const int row0            = split * kDflash2TopkBlockKeys +
                                    (group * kDflash2PathSelectBlock + tid) * kDflash2TopkRowGroup;
        std::uint64_t* group_keys = keys + group * kDflash2TopkRowGroup;
        if (row0 >= vocab) {
#pragma unroll
            for (int item = 0; item < kDflash2TopkRowGroup; ++item) { group_keys[item] = 0; }
        } else if (vector_rows) {
            const uint4 packed           = *reinterpret_cast<const uint4*>(column_bits + row0);
            const std::uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
            for (int word = 0; word < 4; ++word) {
                group_keys[2 * word] =
                    dflash2_topk_key(static_cast<std::uint16_t>(words[word]), row0 + 2 * word);
                group_keys[2 * word + 1] = dflash2_topk_key(
                    static_cast<std::uint16_t>(words[word] >> 16), row0 + 2 * word + 1);
            }
        } else {
#pragma unroll
            for (int item = 0; item < kDflash2TopkRowGroup; ++item) {
                const int row    = row0 + item;
                group_keys[item] = row < vocab ? dflash2_topk_key(column_bits[row], row) : 0;
            }
        }
    }

    std::uint64_t* out =
        split_keys + (column * splits + split) * static_cast<std::int64_t>(kDflash2PathSelectK);
    dflash2_block_top16(keys, shared, [&](int rank, std::uint64_t key) { out[rank] = key; });
}

// grid (T, B), block 256. CTA (t,b) merges the column's splits * 16 <= kDflash2TopkBlockKeys
// split keys (16-byte loads of the key pair at offset 2 * (g * 256 + tid)) into
// cand_val/cand_idx[(b*T + t) * 16 + rank] in rank order. A rank with no finite logit gets value
// -inf and row = rank (rank < 16 <= vocab). Rows map through logit_token_ids when it is non-null.
// split_keys must be 16-byte aligned.
__launch_bounds__(kDflash2PathSelectBlock) __global__
    void dflash2_column_topk_merge_kernel(const std::uint64_t* split_keys, float* cand_val,
                                          int* cand_idx, const std::int32_t* logit_token_ids,
                                          std::int32_t splits) {
    __shared__ Dflash2TopkShared shared;
    const int tid     = static_cast<int>(threadIdx.x);
    const auto column = dflash2_column_index(
        static_cast<int>(gridDim.x), static_cast<int>(blockIdx.x), static_cast<int>(blockIdx.y));
    const int count = splits * kDflash2PathSelectK;
    const auto* src = reinterpret_cast<const ulonglong2*>(split_keys + column * count);

    // At most kDflash2TopkMaxSplits * 16 keys exist, so only the first kPairs key pairs per
    // thread can be live; the rest are compile-time empty.
    constexpr int kPairs =
        kDflash2TopkMaxSplits * kDflash2PathSelectK / (2 * kDflash2PathSelectBlock);
    static_assert(kPairs * 2 * kDflash2PathSelectBlock ==
                      kDflash2TopkMaxSplits * kDflash2PathSelectK &&
                  2 * kPairs <= kDflash2TopkItems);
    std::uint64_t keys[kDflash2TopkItems] = {};
#pragma unroll
    for (int pair = 0; pair < kPairs; ++pair) {
        const int at = pair * kDflash2PathSelectBlock + tid;
        ulonglong2 two{0, 0};
        if (2 * at < count) { two = src[at]; }
        keys[2 * pair]     = two.x;
        keys[2 * pair + 1] = two.y;
    }

    const std::int64_t dst = column * kDflash2PathSelectK;
    dflash2_block_top16(keys, shared, [&](int rank, std::uint64_t key) {
        const int row        = key != 0 ? dflash2_topk_key_row(key) : rank;
        cand_val[dst + rank] = key != 0 ? dflash2_topk_key_value(key) : -CUDART_INF_F;
        cand_idx[dst + rank] = logit_token_ids != nullptr ? logit_token_ids[row] : row;
    });
}

__device__ __forceinline__ float dflash2_markov_score_serial(const __nv_bfloat16* hidden_proj,
                                                             const __nv_bfloat16* pred_code,
                                                             const __nv_bfloat16* succ_code,
                                                             std::int64_t h_col, int prev, int cand,
                                                             float unary) {
    float acc               = unary;
    const std::int64_t pred = static_cast<std::int64_t>(prev) * kDflash2PathSelectRank;
    const std::int64_t succ = static_cast<std::int64_t>(cand) * kDflash2PathSelectRank;
    for (int r = 0; r < kDflash2PathSelectRank; ++r) {
        const float hr = __bfloat162float(hidden_proj[h_col + r]);
        const float pr = __bfloat162float(pred_code[pred + r]);
        const float sr = __bfloat162float(succ_code[succ + r]);
        acc += (pr * hr) * sr;
    }
    return acc;
}

__device__ __forceinline__ float dflash2_markov_score_staged(const float* hidden,
                                                             const __nv_bfloat16* pred,
                                                             const __nv_bfloat16* succ,
                                                             float unary) {
    float acc = unary;
    for (int r = 0; r < kDflash2PathSelectRank; ++r) {
        const float pr = __bfloat162float(pred[r]);
        const float sr = __bfloat162float(succ[r]);
        acc += (pr * hidden[r]) * sr;
    }
    return acc;
}

__launch_bounds__(kDflash2PathSelectBlock) __global__ void dflash2_path_select_kernel(
    const float* cand_val, const int* cand_idx, const __nv_bfloat16* hidden_proj,
    Dflash2CodebookDevice pred_code, Dflash2CodebookDevice succ_code, const std::int32_t* anchors,
    const std::int32_t* logical_positions, std::int32_t* path, std::int32_t* selector_ids,
    float* selector_q, std::int32_t tokens, std::int32_t batch, const SamplingConfig* configs,
    unsigned long long seed_xor, std::int32_t position_offset, bool force_greedy) {
    const int b   = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    if (b >= batch) { return; }
    const SamplingConfig cfg = configs[b];
    // q written below is the law each draft is drawn from (one-hot when greedy); verification
    // accepts with min(1, p/q) under every target sampler, p-less included. P-less rows draw at
    // their own draft temperature: the shortlist softmax at the p-less target temperature itself
    // accepts less than argmax, while a lower one accepts more.
    const float temperature =
        force_greedy ? 0.0f : (cfg.p_less != 0 ? cfg.draft_temperature : cfg.temperature);
    const unsigned long long seed = cfg.seed ^ seed_xor;

    __shared__ float scores[kDflash2PathSelectK];
    __shared__ float sm_val[kDflash2PathSelectK];
    __shared__ int sm_idx[kDflash2PathSelectK];
    __shared__ float sm_h[kDflash2PathSelectRank];
    __shared__ __nv_bfloat16 sm_pred[kDflash2PathSelectRank];
    __shared__ __nv_bfloat16 sm_succ[kDflash2PathSelectK * kDflash2PathSelectSuccStride];
    __shared__ int prev_id;

    if (tid == 0) { prev_id = anchors[b]; }
    __syncthreads();

    for (std::int32_t t = 0; t < tokens; ++t) {
        const std::int64_t col   = dflash2_column_index(tokens, t, b);
        const std::int64_t h_col = col * kDflash2PathSelectRank;
        if (tid < kDflash2PathSelectK) {
            sm_val[tid] = cand_val[col * kDflash2PathSelectK + tid];
            sm_idx[tid] = cand_idx[col * kDflash2PathSelectK + tid];
        }
        __syncthreads();

        if (tid < kDflash2PathSelectRank) {
            sm_h[tid]    = __bfloat162float(hidden_proj[h_col + tid]);
            sm_pred[tid] = dflash2_codebook_load(pred_code, prev_id, tid);
        }
        for (int i = tid; i < kDflash2PathSelectK * kDflash2PathSelectRank;
             i += kDflash2PathSelectBlock) {
            const int c = i / kDflash2PathSelectRank;
            const int r = i - c * kDflash2PathSelectRank;
            sm_succ[c * kDflash2PathSelectSuccStride + r] =
                dflash2_codebook_load(succ_code, sm_idx[c], r);
        }
        __syncthreads();

        if (tid < kDflash2PathSelectK) {
            scores[tid] = dflash2_markov_score_staged(
                sm_h, sm_pred, sm_succ + tid * kDflash2PathSelectSuccStride, sm_val[tid]);
        }
        __syncthreads();

        if (tid == 0) {
            int pick = 0;
            if (temperature > 0.0f) {
                float m = scores[0];
                for (int c = 1; c < kDflash2PathSelectK; ++c) { m = fmaxf(m, scores[c]); }
                const float inv_temp = 1.0f / temperature;
                float sum            = 0.0f;
                for (int c = 0; c < kDflash2PathSelectK; ++c) {
                    scores[c] = expf((scores[c] - m) * inv_temp);
                    sum += scores[c];
                }
                // Keyed by the round's first position and the hop. Block verification's accepted
                // length depends on drafts past it, so a draft uniform must not be reused by the
                // next round's hops at the same absolute position.
                const int round_start = logical_positions[b] + position_offset + 1;
                const float u    = dflash2_path_select_uniform(seed, round_start,
                                                               kDflash2PathSelectRngPurposeDevice,
                                                               static_cast<unsigned int>(t));
                const float goal = u * sum;
                float run        = 0.0f;
                pick             = kDflash2PathSelectK - 1;
                for (int c = 0; c < kDflash2PathSelectK; ++c) {
                    run += scores[c];
                    if (goal < run) {
                        pick = c;
                        break;
                    }
                }
            } else {
                float best  = scores[0];
                int best_id = sm_idx[0];
                for (int c = 1; c < kDflash2PathSelectK; ++c) {
                    const float s = scores[c];
                    const int id  = sm_idx[c];
                    if (s > best || (s == best && id < best_id)) {
                        best    = s;
                        best_id = id;
                        pick    = c;
                    }
                }
            }

            const int chosen = sm_idx[pick];
            path[static_cast<std::int64_t>(t) + static_cast<std::int64_t>(b) * tokens] = chosen;
            if (selector_ids != nullptr) {
                const std::int64_t sel_base =
                    (static_cast<std::int64_t>(t) + static_cast<std::int64_t>(b) * tokens) *
                    kDflash2PathSelectK;
                float qsum = 0.0f;
                if (temperature > 0.0f) {
                    for (int c = 0; c < kDflash2PathSelectK; ++c) { qsum += scores[c]; }
                }
                for (int c = 0; c < kDflash2PathSelectK; ++c) {
                    selector_ids[sel_base + c] = sm_idx[c];
                    if (selector_q != nullptr) {
                        if (temperature > 0.0f) {
                            selector_q[sel_base + c] = qsum > 0.0f ? scores[c] / qsum : 0.0f;
                        } else {
                            selector_q[sel_base + c] = c == pick ? 1.0f : 0.0f;
                        }
                    }
                }
            }
            prev_id = chosen;
        }
        __syncthreads();
    }
}

// Best-first Markov draft tree. One CTA per batch row. The tree grows by repeatedly popping the
// frontier entry with the greatest path log-probability, where a child's log-probability is the
// 16-way log-softmax of the selector's Markov scores (unary + bilinear term) under its parent.
// Path log-probabilities are non-increasing along a path, so the popped set is prefix-closed and
// is the out_width-1 node tree of greatest summed path probability under the Markov model.
// Shared memory: candidate ids/unary per column (16), the projected hidden column of the popped
// node's child depth, one predecessor row, 16 successor rows, and a frontier pool of at most
// 16 + 15*16 entries. Thread 0 performs the pop; threads 0..15 score the 16 children.
__launch_bounds__(kDflash2PathSelectBlock) __global__ void dflash2_tree_select_kernel(
    const float* cand_val, const int* cand_idx, const __nv_bfloat16* hidden_proj,
    Dflash2CodebookDevice pred_code, Dflash2CodebookDevice succ_code, const std::int32_t* anchors,
    const std::int32_t* frontiers, std::int32_t* verify_ids, std::int32_t* parent_index,
    std::int32_t* cache_positions, std::int32_t* rope_positions, std::int32_t* ancestor_mask,
    std::int32_t* valid_columns, std::int32_t tokens, std::int32_t batch, std::int32_t out_width,
    const SamplingConfig* configs, float p_less_tree_temperature) {
    constexpr int kExpand = kDflash2TreeMaxWidthDevice;
    constexpr int kPool   = kDflash2PathSelectK * kExpand;
    const int kOut        = out_width;
    const int b           = static_cast<int>(blockIdx.x);
    const int tid         = static_cast<int>(threadIdx.x);
    if (b >= batch || kOut < 2 || kOut > kExpand) { return; }

    __shared__ int node_id[kExpand];
    __shared__ int node_parent[kExpand];
    __shared__ int node_depth[kExpand];
    __shared__ float node_score[kExpand];
    __shared__ float pool_score[kPool];
    __shared__ int pool_parent[kPool];
    __shared__ int pool_cand[kPool];
    __shared__ int pool_depth[kPool];
    __shared__ int pool_n;
    __shared__ int live;
    __shared__ int expand_node;
    __shared__ float sm_h[kDflash2PathSelectRank];
    __shared__ __nv_bfloat16 sm_pred[kDflash2PathSelectRank];
    __shared__ __nv_bfloat16 sm_succ[kDflash2PathSelectK * kDflash2PathSelectSuccStride];
    __shared__ float child_score[kDflash2PathSelectK];
    __shared__ int child_id[kDflash2PathSelectK];

    if (tid == 0) {
        node_id[0]     = anchors[b];
        node_parent[0] = -1;
        node_depth[0]  = 0;
        node_score[0]  = 0.0f;
        pool_n         = 0;
        live           = 1;
        expand_node    = 0;
    }
    __syncthreads();

    while (true) {
        // Expand expand_node: score its 16 children in column t = depth(expand_node).
        const int parent = expand_node;
        const int t      = node_depth[parent];
        if (t < tokens) {
            const std::int64_t col = dflash2_column_index(tokens, t, b);
            if (tid < kDflash2PathSelectRank) {
                sm_h[tid]    = __bfloat162float(hidden_proj[col * kDflash2PathSelectRank + tid]);
                sm_pred[tid] = dflash2_codebook_load(pred_code, node_id[parent], tid);
            }
            if (tid < kDflash2PathSelectK) {
                child_id[tid] = cand_idx[col * kDflash2PathSelectK + tid];
            }
            __syncthreads();
            for (int i = tid; i < kDflash2PathSelectK * kDflash2PathSelectRank;
                 i += kDflash2PathSelectBlock) {
                const int c = i / kDflash2PathSelectRank;
                const int r = i - c * kDflash2PathSelectRank;
                sm_succ[c * kDflash2PathSelectSuccStride + r] =
                    dflash2_codebook_load(succ_code, child_id[c], r);
            }
            __syncthreads();
            if (tid < kDflash2PathSelectK) {
                child_score[tid] = dflash2_markov_score_staged(
                    sm_h, sm_pred, sm_succ + tid * kDflash2PathSelectSuccStride,
                    cand_val[col * kDflash2PathSelectK + tid]);
            }
            __syncthreads();
            // Warp 0 normalizes the 16 children (lane c holds child c) and appends them to the
            // frontier pool; the pool holds at most kExpand expansions of 16 children.
            if (tid < kWarpSizeDevice) {
                // A p-less row samples a flatter target; its tree tracks that law with a hotter
                // per-parent softmax.
                const float inv_temp = configs != nullptr && configs[b].p_less != 0
                                           ? 1.0f / p_less_tree_temperature
                                           : 1.0f;
                const bool child     = tid < kDflash2PathSelectK;
                const float x        = child ? child_score[tid] * inv_temp : -CUDART_INF_F;
                float mx             = x;
                for (int offset = 16; offset > 0; offset >>= 1) {
                    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffU, mx, offset));
                }
                float sum = child ? __expf(x - mx) : 0.0f;
                for (int offset = 16; offset > 0; offset >>= 1) {
                    sum += __shfl_xor_sync(0xffffffffU, sum, offset);
                }
                const float lse = mx + __logf(sum);
                const int base  = pool_n;
                if (child) {
                    pool_score[base + tid]  = node_score[parent] + x - lse;
                    pool_parent[base + tid] = parent;
                    pool_cand[base + tid]   = child_id[tid];
                    pool_depth[base + tid]  = t + 1;
                }
                __syncwarp();
                if (tid == 0) { pool_n = base + kDflash2PathSelectK; }
            }
        }
        __syncthreads();
        // Warp 0 pops the frontier entry of greatest path log-probability; ties take the lower
        // token id, then the earlier pool entry.
        if (tid < kWarpSizeDevice) {
            float best_score = -CUDART_INF_F;
            int best_cand    = INT_MAX;
            int best         = -1;
            for (int i = tid; i < pool_n; i += kWarpSizeDevice) {
                if (pool_parent[i] < 0) { continue; }
                const float sc = pool_score[i];
                const int cand = pool_cand[i];
                if (best < 0 || sc > best_score || (sc == best_score && cand < best_cand)) {
                    best_score = sc;
                    best_cand  = cand;
                    best       = i;
                }
            }
            for (int offset = 16; offset > 0; offset >>= 1) {
                const float o_score = __shfl_xor_sync(0xffffffffU, best_score, offset);
                const int o_cand    = __shfl_xor_sync(0xffffffffU, best_cand, offset);
                const int o_best    = __shfl_xor_sync(0xffffffffU, best, offset);
                const bool take =
                    o_best >= 0 &&
                    (best < 0 || o_score > best_score ||
                     (o_score == best_score &&
                      (o_cand < best_cand || (o_cand == best_cand && o_best < best))));
                if (take) {
                    best_score = o_score;
                    best_cand  = o_cand;
                    best       = o_best;
                }
            }
            if (tid == 0) {
                if (best >= 0 && live < kOut) {
                    const int n       = live;
                    node_id[n]        = pool_cand[best];
                    node_parent[n]    = pool_parent[best];
                    node_depth[n]     = pool_depth[best];
                    node_score[n]     = pool_score[best];
                    pool_parent[best] = -1;
                    expand_node       = n;
                    ++live;
                } else {
                    expand_node = -1;
                }
            }
        }
        __syncthreads();
        if (expand_node < 0 || live >= kOut) { break; }
    }

    if (tid == 0) {
        const int e     = frontiers[b];
        const int out_n = live < kOut ? live : kOut;
        // Emit columns in depth-first preorder with children in descending path probability, so
        // the most probable path is a contiguous spine (parent = previous column) and only
        // sibling subtrees resume from an earlier column.
        int order[kExpand];
        int column_of[kExpand];
        int stack[kExpand];
        int top      = 0;
        int placed   = 0;
        stack[top++] = 0;
        while (top > 0) {
            const int n     = stack[--top];
            column_of[n]    = placed;
            order[placed++] = n;
            // Push children lowest-probability first so the best child is popped next.
            int kids[kExpand];
            int nk = 0;
            for (int c = 1; c < out_n; ++c) {
                if (node_parent[c] == n) { kids[nk++] = c; }
            }
            for (int i = 1; i < nk; ++i) {
                const int v = kids[i];
                int j       = i - 1;
                while (j >= 0 && node_score[kids[j]] > node_score[v]) {
                    kids[j + 1] = kids[j];
                    --j;
                }
                kids[j + 1] = v;
            }
            for (int i = 0; i < nk; ++i) { stack[top++] = kids[i]; }
        }
        for (int i = 0; i < out_n; ++i) {
            const int n                   = order[i];
            verify_ids[b * kOut + i]      = node_id[n];
            parent_index[b * kOut + i]    = i == 0 ? -1 : column_of[node_parent[n]];
            cache_positions[b * kOut + i] = e + i;
            rope_positions[b * kOut + i]  = e + node_depth[n];
            int mask                      = 0;
            int cur                       = n;
            while (cur >= 0) {
                mask |= 1 << column_of[cur];
                cur = node_parent[cur];
            }
            ancestor_mask[b * kOut + i] = mask;
        }
        const int last = out_n > 0 ? out_n - 1 : 0;
        for (int col = out_n; col < kOut; ++col) {
            verify_ids[b * kOut + col]      = verify_ids[b * kOut + last];
            parent_index[b * kOut + col]    = parent_index[b * kOut + last];
            cache_positions[b * kOut + col] = e + col;
            rope_positions[b * kOut + col]  = rope_positions[b * kOut + last];
            ancestor_mask[b * kOut + col]   = ancestor_mask[b * kOut + last];
        }
        valid_columns[b] = out_n;
    }
}

} // namespace ninfer::ops
