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

inline constexpr int kDflash2PathSelectBlock     = 256;
inline constexpr int kDflash2PathSelectK         = 16;
inline constexpr int kDflash2PathSelectRank      = 256;
inline constexpr int kDflash2PathSelectSuccStride  = kDflash2PathSelectRank + 2;
inline constexpr int kDflash2PathSelectGemmBlock = 256;
inline constexpr int kDflash2PathSelectRngPurposeDevice = 16;
inline constexpr int kDflash2PathSelectTopkSplits = 32;
inline constexpr int kDflash2PathSelectMaxBatchDevice = 8;

struct Dflash2CodebookDevice {
    const __nv_bfloat16* bf16          = nullptr;
    const std::uint8_t* nvfp4_codes    = nullptr;
    const std::uint8_t* nvfp4_scales   = nullptr;
    float inv_dw                       = 0.0f;
};

__device__ __forceinline__ float dflash2_e2m1_value(std::uint8_t code) {
    constexpr float kMags[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float mag          = kMags[code & 7u];
    return (code & 8u) ? -mag : mag;
}

__device__ __forceinline__ __nv_bfloat16 dflash2_codebook_load(Dflash2CodebookDevice book, int token,
                                                              int rank) {
    if (token < 0) { token = 0; }
    if (book.bf16 != nullptr) {
        return book.bf16[static_cast<std::int64_t>(token) * kDflash2PathSelectRank + rank];
    }
    constexpr int kTiles = kDflash2PathSelectRank / 64;
    const std::uint8_t packed =
        book.nvfp4_codes[static_cast<std::int64_t>(token) * (kDflash2PathSelectRank / 2) + rank / 2];
    const std::uint8_t nibble =
        (rank & 1) != 0 ? static_cast<std::uint8_t>(packed >> 4) : static_cast<std::uint8_t>(packed & 0x0f);
    const int group      = rank / 16;
    const int m_tile     = token / 128;
    const int row_inner  = token - m_tile * 128;
    const std::int64_t scale_off =
        static_cast<std::int64_t>(m_tile * kTiles + group / 4) * 512 +
        static_cast<std::int64_t>(row_inner & 31) * 16 + static_cast<std::int64_t>(row_inner >> 5) * 4 +
        (group & 3);
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

__device__ __forceinline__ unsigned long long dflash2_path_select_splitmix64(
    unsigned long long x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

__device__ __forceinline__ float dflash2_path_select_uniform(unsigned long long seed, int position,
                                                              int purpose) {
    unsigned long long key = seed;
    key                    = dflash2_path_select_splitmix64(
        key ^ (static_cast<unsigned long long>(static_cast<unsigned int>(position)) *
               0xD1B54A32D192ED03ull));
    key = dflash2_path_select_splitmix64(
        key ^ (static_cast<unsigned long long>(static_cast<unsigned int>(purpose)) << 21));
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

__device__ __forceinline__ std::int64_t dflash2_column_index(int batch, int tokens, int t, int b) {
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
            dflash2_column_index(batch, tokens, t, b) * static_cast<std::int64_t>(vocab);
        for (int v = v0 + tid; v < v1; v += kDflash2PathSelectBlock) {
            dflash2_insert_topk(local_val, local_idx, kDflash2PathSelectK,
                                __bfloat162float(logits[logit_col + v]), v);
        }
    }

    __shared__ float top_val[kDflash2PathSelectBlock * kDflash2PathSelectK];
    __shared__ int top_idx[kDflash2PathSelectBlock * kDflash2PathSelectK];
    __shared__ float warp_val[8 * kDflash2PathSelectK];
    __shared__ int warp_idx[8 * kDflash2PathSelectK];
    const int local_base = tid * kDflash2PathSelectK;
#pragma unroll
    for (int j = 0; j < kDflash2PathSelectK; ++j) {
        top_val[local_base + j] = local_val[j];
        top_idx[local_base + j] = local_idx[j];
    }
    __syncthreads();

    const int warp = tid / 32;
    const int lane = tid % 32;
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
                dflash2_insert_topk(merged_val, merged_idx, kDflash2PathSelectK, top_val[row + j],
                                    top_idx[row + j]);
            }
        }
        const int wbase = warp * kDflash2PathSelectK;
        for (int j = 0; j < kDflash2PathSelectK; ++j) {
            warp_val[wbase + j] = merged_val[j];
            warp_idx[wbase + j] = merged_idx[j];
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
            (dflash2_column_index(batch, tokens, t, b) * kDflash2PathSelectTopkSplits + split) *
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
    if (t >= tokens || b >= batch || tid != 0) { return; }

    const std::int64_t column = dflash2_column_index(batch, tokens, t, b);
    const std::int64_t src    = column * kDflash2PathSelectTopkSplits * kDflash2PathSelectK;
    float out_val[kDflash2PathSelectK];
    int out_idx[kDflash2PathSelectK];
    for (int j = 0; j < kDflash2PathSelectK; ++j) {
        out_val[j] = -CUDART_INF_F;
        out_idx[j] = INT_MAX;
    }
    for (int p = 0; p < kDflash2PathSelectTopkSplits * kDflash2PathSelectK; ++p) {
        if (split_idx[src + p] == INT_MAX) { continue; }
        dflash2_insert_topk(out_val, out_idx, kDflash2PathSelectK, split_val[src + p],
                            split_idx[src + p]);
    }
    for (int c = 0; c < kDflash2PathSelectK; ++c) {
        if (out_idx[c] != INT_MAX) { continue; }
        out_idx[c] = c < vocab ? c : 0;
        out_val[c] = -CUDART_INF_F;
    }
    if (logit_token_ids != nullptr) {
        for (int c = 0; c < kDflash2PathSelectK; ++c) {
            const int row = out_idx[c];
            out_idx[c]    = (row >= 0 && row < vocab) ? logit_token_ids[row] : 0;
        }
    }
    const std::int64_t dst = column * kDflash2PathSelectK;
    for (int c = 0; c < kDflash2PathSelectK; ++c) {
        cand_val[dst + c] = out_val[c];
        cand_idx[dst + c] = out_idx[c];
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

__launch_bounds__(kDflash2PathSelectBlock) __global__
    void dflash2_path_select_kernel(const float* cand_val, const int* cand_idx,
                                    const __nv_bfloat16* hidden_proj, Dflash2CodebookDevice pred_code,
                                    Dflash2CodebookDevice succ_code, const std::int32_t* anchors,
                                     const std::int32_t* logical_positions, std::int32_t* path,
                                     std::int32_t* selector_ids, float* selector_q,
                                    std::int32_t tokens, std::int32_t batch,
                                    const SamplingConfig* configs, unsigned long long seed_xor,
                                     std::int32_t position_offset, bool force_greedy) {
    const int b   = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    if (b >= batch) { return; }
    const SamplingConfig cfg = configs[b];
    // P-less temperature parameterizes the target distribution, not the draft
    // scores. Adaptive chain at the product T=2 softmax would otherwise flatten
    // the 16-way proposal while Leviathan still sees that q.
    const float temperature =
        (force_greedy || cfg.p_less != 0) ? 0.0f : cfg.temperature;
    const unsigned long long seed  = cfg.seed ^ seed_xor;

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
        const std::int64_t col   = dflash2_column_index(batch, tokens, t, b);
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
            sm_succ[c * kDflash2PathSelectSuccStride + r]  = dflash2_codebook_load(succ_code, sm_idx[c], r);
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
                const int position =
                    logical_positions[b] + position_offset + static_cast<int>(t) + 1;
                const float u = dflash2_path_select_uniform(
                    seed, position, kDflash2PathSelectRngPurposeDevice);
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

__launch_bounds__(kDflash2PathSelectBlock) __global__
    void dflash2_tree_select_kernel(const float* cand_val, const int* cand_idx,
                                    const __nv_bfloat16* hidden_proj, Dflash2CodebookDevice pred_code,
                                    Dflash2CodebookDevice succ_code, const std::int32_t* anchors,
                                    const std::int32_t* frontiers, std::int32_t* verify_ids,
                                    std::int32_t* parent_index, std::int32_t* cache_positions,
                                    std::int32_t* rope_positions, std::int32_t* ancestor_mask,
                                    std::int32_t* valid_columns, std::int32_t tokens,
                                    std::int32_t batch, std::int32_t out_width) {
    constexpr int kExpand   = 16;
    constexpr int kFrontier = 2;
    const int kOut          = out_width;
    const int b             = static_cast<int>(blockIdx.x);
    const int tid           = static_cast<int>(threadIdx.x);
    if (b >= batch || kOut < 2 || kOut > kExpand) { return; }

    __shared__ float sm_val[kDflash2PathSelectK];
    __shared__ int sm_idx[kDflash2PathSelectK];
    __shared__ int node_id[kExpand];
    __shared__ int node_parent[kExpand];
    __shared__ int node_depth[kExpand];
    __shared__ float node_score[kExpand];
    __shared__ int frontier[kFrontier];
    __shared__ int frontier_n;
    __shared__ int live;
    __shared__ float sm_h[kDflash2PathSelectRank];
    __shared__ __nv_bfloat16 sm_pred[kFrontier * kDflash2PathSelectRank];
    __shared__ __nv_bfloat16 sm_succ[kDflash2PathSelectK * kDflash2PathSelectSuccStride];
    __shared__ float pair_score[kFrontier * kDflash2PathSelectK];

    if (tid == 0) {
        node_id[0]     = anchors[b];
        node_parent[0] = -1;
        node_depth[0]  = 0;
        node_score[0]  = 0.0f;
        frontier[0]    = 0;
        frontier_n     = 1;
        live           = 1;
    }
    __syncthreads();

    for (std::int32_t t = 0; t < tokens; ++t) {
        const std::int64_t col   = dflash2_column_index(batch, tokens, t, b);
        const std::int64_t h_col = col * kDflash2PathSelectRank;
        if (tid < kDflash2PathSelectK) {
            sm_val[tid] = cand_val[col * kDflash2PathSelectK + tid];
            sm_idx[tid] = cand_idx[col * kDflash2PathSelectK + tid];
        }
        __syncthreads();

        if (tid < kDflash2PathSelectRank) {
            sm_h[tid] = __bfloat162float(hidden_proj[h_col + tid]);
        }
        for (int i = tid; i < kFrontier * kDflash2PathSelectRank; i += kDflash2PathSelectBlock) {
            const int f    = i / kDflash2PathSelectRank;
            const int r    = i - f * kDflash2PathSelectRank;
            const int prev = f < frontier_n ? node_id[frontier[f]] : 0;
            sm_pred[i] = dflash2_codebook_load(pred_code, prev, r);
        }
        for (int i = tid; i < kDflash2PathSelectK * kDflash2PathSelectRank;
             i += kDflash2PathSelectBlock) {
            const int c = i / kDflash2PathSelectRank;
            const int r = i - c * kDflash2PathSelectRank;
            sm_succ[c * kDflash2PathSelectSuccStride + r] = dflash2_codebook_load(succ_code, sm_idx[c], r);
        }
        __syncthreads();

        if (tid < kFrontier * kDflash2PathSelectK) {
            const int f = tid / kDflash2PathSelectK;
            const int c = tid - f * kDflash2PathSelectK;
            if (f < frontier_n) {
                pair_score[tid] = dflash2_markov_score_staged(
                    sm_h, sm_pred + f * kDflash2PathSelectRank,
                    sm_succ + c * kDflash2PathSelectSuccStride, sm_val[c]);
            } else {
                pair_score[tid] = -CUDART_INF_F;
            }
        }
        __syncthreads();

        if (tid == 0) {
            float best_score[kFrontier];
            int best_parent[kFrontier];
            int best_cand[kFrontier];
            for (int s = 0; s < kFrontier; ++s) {
                best_score[s]  = -CUDART_INF_F;
                best_parent[s] = -1;
                best_cand[s]   = INT_MAX;
            }

            const int parents = frontier_n;
            for (int f = 0; f < parents; ++f) {
                const int pcol   = frontier[f];
                const float base = node_score[pcol];
                for (int c = 0; c < kDflash2PathSelectK; ++c) {
                    const int cand    = sm_idx[c];
                    const float joint = base + pair_score[f * kDflash2PathSelectK + c];
                    int slot = kFrontier;
                    for (int s = 0; s < kFrontier; ++s) {
                        if (joint > best_score[s] ||
                            (joint == best_score[s] &&
                             (cand < best_cand[s] ||
                              (cand == best_cand[s] && pcol < best_parent[s])))) {
                            slot = s;
                            break;
                        }
                    }
                    if (slot == kFrontier) { continue; }
                    for (int s = kFrontier - 1; s > slot; --s) {
                        best_score[s]  = best_score[s - 1];
                        best_parent[s] = best_parent[s - 1];
                        best_cand[s]   = best_cand[s - 1];
                    }
                    best_score[slot]  = joint;
                    best_parent[slot] = pcol;
                    best_cand[slot]   = cand;
                }
            }

            const int depth = static_cast<int>(t) + 1;
            int next_n      = 0;
            for (int s = 0; s < kFrontier; ++s) {
                if (best_parent[s] < 0) { continue; }
                const int col_i = live;
                if (col_i >= kExpand) { break; }
                node_id[col_i]     = best_cand[s];
                node_parent[col_i] = best_parent[s];
                node_depth[col_i]  = depth;
                node_score[col_i]  = best_score[s];
                frontier[next_n]   = col_i;
                ++next_n;
                ++live;
            }
            frontier_n = next_n > 0 ? next_n : frontier_n;
        }
        __syncthreads();
    }

    if (tid == 0) {
        const int e     = frontiers[b];
        const int out_n = live < kOut ? live : kOut;
        for (int i = 0; i < out_n; ++i) {
            verify_ids[b * kOut + i]      = node_id[i];
            parent_index[b * kOut + i]    = i == 0 ? -1 : node_parent[i];
            cache_positions[b * kOut + i] = e + i;
            rope_positions[b * kOut + i]  = e + node_depth[i];
            int mask                      = 0;
            int cur                       = i;
            while (cur >= 0) {
                mask |= 1 << cur;
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
