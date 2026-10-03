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
inline constexpr int kDflash2PathSelectTopkSplits       = 32;
inline constexpr int kDflash2PathSelectMaxBatchDevice   = 8;
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

__device__ __forceinline__ bool dflash2_logit_better(float value, int index, float best_value,
                                                     int best_index) {
    return value > best_value || (value == best_value && index < best_index);
}

__device__ __forceinline__ void dflash2_insert_topk(float* vals, int* idxs, int cap, float value,
                                                    int index) {
    if (cap <= 0 || index < 0 || isnan(value) || isinf(value) ||
        !dflash2_logit_better(value, index, vals[cap - 1], idxs[cap - 1])) {
        return;
    }
    int pos = cap - 1;
    while (pos > 0 && dflash2_logit_better(value, index, vals[pos - 1], idxs[pos - 1])) {
        vals[pos] = vals[pos - 1];
        idxs[pos] = idxs[pos - 1];
        --pos;
    }
    vals[pos] = value;
    idxs[pos] = index;
}

__device__ __forceinline__ std::int64_t dflash2_column_index(int tokens, int t, int b) {
    return static_cast<std::int64_t>(b) * tokens + t;
}

__launch_bounds__(kDflash2PathSelectBlock) __global__
    void dflash2_column_topk_split_kernel(const __nv_bfloat16* logits, float* split_val,
                                          int* split_idx, std::int32_t vocab, std::int32_t tokens,
                                          std::int32_t batch) {
    const int split = static_cast<int>(blockIdx.x);
    const int t     = static_cast<int>(blockIdx.y);
    const int b     = static_cast<int>(blockIdx.z);
    const int tid   = static_cast<int>(threadIdx.x);
    if (split >= kDflash2PathSelectTopkSplits || t >= tokens || b >= batch) { return; }

    const int chunk = (vocab + kDflash2PathSelectTopkSplits - 1) / kDflash2PathSelectTopkSplits;
    const int v0    = split * chunk;
    const int v1    = v0 + chunk < vocab ? v0 + chunk : vocab;

    float local_val[kDflash2PathSelectK];
    int local_idx[kDflash2PathSelectK];
#pragma unroll
    for (int j = 0; j < kDflash2PathSelectK; ++j) {
        local_val[j] = -CUDART_INF_F;
        local_idx[j] = INT_MAX;
    }
    if (v0 < v1) {
        const std::int64_t logit_col =
            dflash2_column_index(tokens, t, b) * static_cast<std::int64_t>(vocab);
        // The optimized production shortlist leaves at most 16 rows per thread in each
        // split. Keep those rows unsorted and select the split's ranks cooperatively below;
        // sorting every thread's private list costs more than the vocabulary scan.
        if (v1 - v0 <= kDflash2PathSelectBlock * kDflash2PathSelectK) {
            int count = 0;
            for (int v = v0 + tid; v < v1; v += kDflash2PathSelectBlock) {
                const float value = __bfloat162float(logits[logit_col + v]);
                if (!isnan(value) && !isinf(value)) {
                    local_val[count] = value;
                    local_idx[count] = v;
                    ++count;
                }
            }
        } else {
            for (int v = v0 + tid; v < v1; v += kDflash2PathSelectBlock) {
                dflash2_insert_topk(local_val, local_idx, kDflash2PathSelectK,
                                    __bfloat162float(logits[logit_col + v]), v);
            }
        }
    }

    __shared__ float warp_val[8 * kDflash2PathSelectK];
    __shared__ int warp_idx[8 * kDflash2PathSelectK];
    const int warp = tid / 32;
    const int lane = tid % 32;
    if (v1 - v0 <= kDflash2PathSelectBlock * kDflash2PathSelectK) {
        for (int j = 0; j < kDflash2PathSelectK; ++j) {
            float best_val = -CUDART_INF_F;
            int best_idx   = INT_MAX;
#pragma unroll
            for (int item = 0; item < kDflash2PathSelectK; ++item) {
                if (local_idx[item] != INT_MAX &&
                    dflash2_logit_better(local_val[item], local_idx[item], best_val, best_idx)) {
                    best_val = local_val[item];
                    best_idx = local_idx[item];
                }
            }
            for (int offset = 16; offset > 0; offset >>= 1) {
                const float other_val = __shfl_down_sync(0xffffffffu, best_val, offset);
                const int other_idx   = __shfl_down_sync(0xffffffffu, best_idx, offset);
                if (dflash2_logit_better(other_val, other_idx, best_val, best_idx)) {
                    best_val = other_val;
                    best_idx = other_idx;
                }
            }
            const int selected_idx = __shfl_sync(0xffffffffu, best_idx, 0);
            if (lane == 0) {
                const int dst = warp * kDflash2PathSelectK + j;
                warp_val[dst] = best_val;
                warp_idx[dst] = best_idx;
            }
#pragma unroll
            for (int item = 0; item < kDflash2PathSelectK; ++item) {
                if (local_idx[item] == selected_idx) { local_idx[item] = INT_MAX; }
            }
        }
    } else {
        __shared__ float top_val[kDflash2PathSelectBlock * kDflash2PathSelectK];
        __shared__ int top_idx[kDflash2PathSelectBlock * kDflash2PathSelectK];
        const int local_base = tid * kDflash2PathSelectK;
#pragma unroll
        for (int j = 0; j < kDflash2PathSelectK; ++j) {
            top_val[local_base + j] = local_val[j];
            top_idx[local_base + j] = local_idx[j];
        }
        __syncthreads();
        if (lane == 0) {
            float merged_val[kDflash2PathSelectK];
            int merged_idx[kDflash2PathSelectK];
            for (int j = 0; j < kDflash2PathSelectK; ++j) {
                merged_val[j] = -CUDART_INF_F;
                merged_idx[j] = INT_MAX;
            }
            const int thread0 = warp * 32;
            for (int thread = thread0; thread < thread0 + 32; ++thread) {
                const int row = thread * kDflash2PathSelectK;
                for (int j = 0; j < kDflash2PathSelectK; ++j) {
                    if (top_idx[row + j] == INT_MAX) { continue; }
                    dflash2_insert_topk(merged_val, merged_idx, kDflash2PathSelectK,
                                        top_val[row + j], top_idx[row + j]);
                }
            }
            const int wbase = warp * kDflash2PathSelectK;
            for (int j = 0; j < kDflash2PathSelectK; ++j) {
                warp_val[wbase + j] = merged_val[j];
                warp_idx[wbase + j] = merged_idx[j];
            }
        }
    }
    __syncthreads();

    if (tid == 0) {
        float merged_val[kDflash2PathSelectK];
        int merged_idx[kDflash2PathSelectK];
        for (int j = 0; j < kDflash2PathSelectK; ++j) {
            merged_val[j] = -CUDART_INF_F;
            merged_idx[j] = INT_MAX;
        }
        for (int p = 0; p < 8 * kDflash2PathSelectK; ++p) {
            if (warp_idx[p] == INT_MAX) { continue; }
            dflash2_insert_topk(merged_val, merged_idx, kDflash2PathSelectK, warp_val[p],
                                warp_idx[p]);
        }
        const std::int64_t out =
            (dflash2_column_index(tokens, t, b) * kDflash2PathSelectTopkSplits + split) *
            kDflash2PathSelectK;
        for (int j = 0; j < kDflash2PathSelectK; ++j) {
            split_val[out + j] = merged_val[j];
            split_idx[out + j] = merged_idx[j];
        }
    }
}

__launch_bounds__(kDflash2PathSelectBlock) __global__
    void dflash2_column_topk_merge_kernel(const float* split_val, const int* split_idx,
                                          float* cand_val, int* cand_idx,
                                          const std::int32_t* logit_token_ids, std::int32_t vocab,
                                          std::int32_t tokens, std::int32_t batch) {
    const int t   = static_cast<int>(blockIdx.x);
    const int b   = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    if (t >= tokens || b >= batch) { return; }

    const std::int64_t column = dflash2_column_index(tokens, t, b);
    const std::int64_t src    = column * kDflash2PathSelectTopkSplits * kDflash2PathSelectK;
    constexpr int kCandidates = kDflash2PathSelectTopkSplits * kDflash2PathSelectK;
    constexpr int kPerThread  = kCandidates / kDflash2PathSelectBlock;
    static_assert(kPerThread * kDflash2PathSelectBlock == kCandidates);
    float local_val[kPerThread];
    int local_idx[kPerThread];
#pragma unroll
    for (int item = 0; item < kPerThread; ++item) {
        const int offset = tid + item * kDflash2PathSelectBlock;
        local_val[item]  = split_val[src + offset];
        local_idx[item]  = split_idx[src + offset];
    }

    __shared__ float reduce_val[kDflash2PathSelectBlock];
    __shared__ int reduce_idx[kDflash2PathSelectBlock];
    const std::int64_t dst = column * kDflash2PathSelectK;
    // The split stage contributes 32 exact top-16 lists. Select each final rank with
    // the same value/index ordering as insertion_topk, but spread the 512 candidates
    // across the block instead of serially merging them on thread zero.
    for (int rank = 0; rank < kDflash2PathSelectK; ++rank) {
        float best_val = -CUDART_INF_F;
        int best_idx   = INT_MAX;
#pragma unroll
        for (int item = 0; item < kPerThread; ++item) {
            if (local_idx[item] != INT_MAX &&
                dflash2_logit_better(local_val[item], local_idx[item], best_val, best_idx)) {
                best_val = local_val[item];
                best_idx = local_idx[item];
            }
        }
        reduce_val[tid] = best_val;
        reduce_idx[tid] = best_idx;
        __syncthreads();
        for (int stride = kDflash2PathSelectBlock / 2; stride > 0; stride >>= 1) {
            if (tid < stride &&
                dflash2_logit_better(reduce_val[tid + stride], reduce_idx[tid + stride],
                                     reduce_val[tid], reduce_idx[tid])) {
                reduce_val[tid] = reduce_val[tid + stride];
                reduce_idx[tid] = reduce_idx[tid + stride];
            }
            __syncthreads();
        }
        const int selected_idx = reduce_idx[0];
        if (tid == 0) {
            const int row = selected_idx != INT_MAX ? selected_idx : (rank < vocab ? rank : 0);
            cand_val[dst + rank] = reduce_val[0];
            cand_idx[dst + rank] = logit_token_ids != nullptr ? logit_token_ids[row] : row;
        }
#pragma unroll
        for (int item = 0; item < kPerThread; ++item) {
            if (local_idx[item] == selected_idx) { local_idx[item] = INT_MAX; }
        }
        __syncthreads();
    }
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
    unsigned long long seed_xor, std::int32_t position_offset, bool force_greedy,
    float p_less_draft_temperature_scale) {
    const int b   = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    if (b >= batch) { return; }
    const SamplingConfig cfg = configs[b];
    // q written below is the law each draft is drawn from (one-hot when greedy); verification
    // accepts with min(1, p/q) under every target sampler, p-less included. P-less rows draw at
    // their own draft temperature: the shortlist softmax at the p-less target temperature itself
    // accepts less than argmax, while a lower one accepts more.
    const float temperature =
        force_greedy ? 0.0f
                     : (cfg.p_less != 0 ? cfg.draft_temperature * p_less_draft_temperature_scale
                                        : cfg.temperature);
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
