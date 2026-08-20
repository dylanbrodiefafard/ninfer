#pragma once

// Implements: include/ninfer/ops/dflash2_path_select.h

#include "ops/common/math.h"

#include <cuda_bf16.h>
#include <climits>
#include <cstdint>
#include <math_constants.h>

namespace ninfer::ops {

inline constexpr int kDflash2PathSelectBlock     = 256;
inline constexpr int kDflash2PathSelectK         = 16;
inline constexpr int kDflash2PathSelectRank      = 256;
inline constexpr int kDflash2PathSelectGemmBlock = 256;
inline constexpr int kDflash2PathSelectRngPurposeDevice = 16;

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

__device__ __forceinline__ float dflash2_path_select_uniform(unsigned long long seed, int t, int b,
                                                             int purpose) {
    unsigned long long key = seed;
    key                    = dflash2_path_select_splitmix64(
        key ^ (static_cast<unsigned long long>(static_cast<unsigned int>(t)) *
               0xD1B54A32D192ED03ull));
    key = dflash2_path_select_splitmix64(
        key ^ (static_cast<unsigned long long>(static_cast<unsigned int>(b)) *
               0xC2B2AE3D27D4EB4Full));
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
    if (cap <= 0 || !dflash2_logit_better(value, index, vals[cap - 1], idxs[cap - 1])) { return; }
    int pos = cap - 1;
    while (pos > 0 && dflash2_logit_better(value, index, vals[pos - 1], idxs[pos - 1])) {
        vals[pos] = vals[pos - 1];
        idxs[pos] = idxs[pos - 1];
        --pos;
    }
    vals[pos] = value;
    idxs[pos] = index;
}

__launch_bounds__(kDflash2PathSelectBlock) __global__
    void dflash2_path_select_kernel(const __nv_bfloat16* logits, const __nv_bfloat16* hidden_proj,
                                    const __nv_bfloat16* pred_code, const __nv_bfloat16* succ_code,
                                    const std::int32_t* anchors, const std::int32_t* logit_token_ids,
                                    std::int32_t* path, std::int32_t vocab, std::int32_t tokens,
                                    std::int32_t batch, std::int32_t codebook_rows,
                                    float temperature, unsigned long long seed) {
    const int b   = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    if (b >= batch) { return; }

    __shared__ float top_val[kDflash2PathSelectBlock * kDflash2PathSelectK];
    __shared__ int top_idx[kDflash2PathSelectBlock * kDflash2PathSelectK];
    __shared__ float cand_val[kDflash2PathSelectK];
    __shared__ int cand_idx[kDflash2PathSelectK];
    __shared__ float scores[kDflash2PathSelectK];
    __shared__ int prev_id;

    if (tid == 0) { prev_id = anchors[b]; }
    __syncthreads();

    float local_val[kDflash2PathSelectK];
    int local_idx[kDflash2PathSelectK];

    for (std::int32_t t = 0; t < tokens; ++t) {
        const std::int64_t logit_col =
            (static_cast<std::int64_t>(b) * tokens + t) * static_cast<std::int64_t>(vocab);
        const std::int64_t h_col =
            (static_cast<std::int64_t>(b) * tokens + t) * kDflash2PathSelectRank;

#pragma unroll
        for (int j = 0; j < kDflash2PathSelectK; ++j) {
            local_val[j] = -CUDART_INF_F;
            local_idx[j] = INT_MAX;
        }
        for (std::int32_t v = tid; v < vocab; v += kDflash2PathSelectBlock) {
            const float logit = __bfloat162float(logits[logit_col + v]);
            dflash2_insert_topk(local_val, local_idx, kDflash2PathSelectK, logit, v);
        }
        const int local_base = tid * kDflash2PathSelectK;
#pragma unroll
        for (int j = 0; j < kDflash2PathSelectK; ++j) {
            top_val[local_base + j] = local_val[j];
            top_idx[local_base + j] = local_idx[j];
        }
        __syncthreads();

        if (tid == 0) {
            for (int j = 0; j < kDflash2PathSelectK; ++j) {
                cand_val[j] = -CUDART_INF_F;
                cand_idx[j] = INT_MAX;
            }
            const int merge_n = kDflash2PathSelectBlock * kDflash2PathSelectK;
            for (int p = 0; p < merge_n; ++p) {
                if (top_idx[p] == INT_MAX) { continue; }
                dflash2_insert_topk(cand_val, cand_idx, kDflash2PathSelectK, top_val[p], top_idx[p]);
            }
            if (logit_token_ids != nullptr) {
                for (int c = 0; c < kDflash2PathSelectK; ++c) {
                    cand_idx[c] = logit_token_ids[cand_idx[c]];
                }
            }

            const int prev = prev_id;
            for (int c = 0; c < kDflash2PathSelectK; ++c) {
                const int cand          = cand_idx[c];
                float acc               = cand_val[c];
                const std::int64_t pred = static_cast<std::int64_t>(prev) * kDflash2PathSelectRank;
                const std::int64_t succ = static_cast<std::int64_t>(cand) * kDflash2PathSelectRank;
                for (int r = 0; r < kDflash2PathSelectRank; ++r) {
                    const float hr = __bfloat162float(hidden_proj[h_col + r]);
                    const float pr = __bfloat162float(pred_code[pred + r]);
                    const float sr = __bfloat162float(succ_code[succ + r]);
                    acc += (pr * hr) * sr;
                }
                scores[c] = acc;
            }

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
                const float u    = dflash2_path_select_uniform(
                    seed, static_cast<int>(t), b, kDflash2PathSelectRngPurposeDevice);
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
                float best = scores[0];
                int best_id = cand_idx[0];
                for (int c = 1; c < kDflash2PathSelectK; ++c) {
                    const float s = scores[c];
                    const int id  = cand_idx[c];
                    if (s > best || (s == best && id < best_id)) {
                        best    = s;
                        best_id = id;
                        pick    = c;
                    }
                }
            }

            const int chosen                                      = cand_idx[pick];
            path[static_cast<std::int64_t>(t) + static_cast<std::int64_t>(b) * tokens] = chosen;
            prev_id                                               = chosen;
            (void)codebook_rows;
        }
        __syncthreads();
    }
}

} // namespace ninfer::ops
