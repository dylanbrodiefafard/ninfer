#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

inline constexpr std::int32_t kDflash2PathSelectTopK           = 16;
inline constexpr std::int32_t kDflash2PathSelectRank           = 256;
inline constexpr std::int32_t kDflash2PathSelectHidden         = 5120;
inline constexpr std::int32_t kDflash2PathSelectCodebookRows   = 248320;
inline constexpr std::int32_t kDflash2PathSelectShortlistRows  = 131072;
inline constexpr std::int32_t kDflash2PathSelectMaxBatch       = 8;
inline constexpr std::int32_t kDflash2PathSelectMaxWidthWhenBatched = 16;
inline constexpr int kDflash2PathSelectRngPurpose              = 16;

/**
 * Op: dflash2_path_select
 *
 * Math / indexing:
 *   For each column (t,b), let rows[0..15] be the unsorted top-16 logit rows of logits[:,t,b]:
 *   the 16 largest represented logits, with lower row index winning logit ties.
 *   candidates[c] = logit_token_ids[rows[c]] when logit_token_ids is non-null, else rows[c].
 *   unary[c] = logits[rows[c], t, b]. Project the hidden state
 *
 *     h[r,t,b] = sum_{k=0}^{5119} W_h[r,k] * hidden[k,t,b],   r in [0,256).
 *
 *   prev[-1,b] = anchors[b]. Anchors are vocabulary ids (not shortlist rows). For t = 0..T-1
 *   and each candidate c,
 *
 *     score[t,b,c] = unary[c]
 *       + sum_{r=0}^{255} (pred_code[r, prev[t-1,b]] * h[r,t,b]) * succ_code[r, candidates[c]].
 *
 *   If temperature <= 0, path[t,b] is the candidate token with the greatest score; equal scores
 *   select the lower token id. If temperature > 0, the 16 scores are softmax-normalized after
 *   dividing by temperature and one candidate is drawn by inverse-CDF using the counter-based
 *   uniform
 *
 *     u = splitmix64(seed, t, b, purpose=16) in [0,1).
 *
 *   Then prev[t,b] = path[t,b]. Candidate order does not affect the selected token.
 *
 * Logical shapes:
 *   logits is contiguous BF16 [V,T] or [V,T,B] with V>=16. hidden is contiguous BF16 [5120,T] or
 *   [5120,T,B]. pred_code and succ_code are contiguous BF16 [256, codebook_rows] (rank fastest).
 *   When logit_token_ids is null, codebook_rows >= V (identity: logit row is the token id).
 *   When logit_token_ids is non-null it is contiguous I32 [V] mapping each logit row to a token
 *   id; codebook_rows >= 248320 and is not required to be >= V. Product uses codebook_rows=248320
 *   and V in {131072, 248320}. anchors is contiguous I32 [B], or a scalar I32 for B=1. path is
 *   contiguous I32 [T] or [T,B]. Every anchor and every selected token id is in [0, codebook_rows).
 *   T is any positive value at B=1; B=2..8 admits T=1..16.
 *
 * Supported domain:
 *   hidden_projection is BF16_CTRL Contiguous [256,5120], or a Linear-registered Q4G64_F16S /
 *   W8G32_F16S RowSplit problem of that logical shape. The Q4/W8 route calls ops::linear.
 *
 * Numeric:
 *   Top-k and greedy path ids are exact functions of the represented BF16 logits/scores. The
 *   score formula is evaluated by the oracle in FP64 from represented inputs and the logical FP32
 *   dequantized W_h. Stochastic draws are a function of (seed, t, b, purpose); the Op does not
 *   promise a particular host RNG bitstream as a public numeric output.
 *
 * Effects:
 *   Writes all of path. Inputs are unchanged. path must not alias logits, hidden, codebooks,
 *   anchors, or any projection-weight plane.
 *
 * Workspace:
 *   Caller-owned transient storage sized by dflash2_path_select_workspace_capacity_bytes() holds
 *   the [256,T*B] hidden projection and any Linear child scratch.
 */
[[nodiscard]] std::size_t dflash2_path_select_workspace_capacity_bytes(QType qtype,
                                                                       std::int32_t min_tokens,
                                                                       std::int32_t max_tokens,
                                                                       std::int32_t batch = 1);

void dflash2_path_select(const Tensor& logits, const Tensor& hidden,
                         const Weight& hidden_projection, const Tensor& pred_code,
                         const Tensor& succ_code, const Tensor& anchors, float temperature,
                         unsigned long long seed, Tensor& path, WorkspaceArena& workspace,
                         cudaStream_t stream, const Tensor* logit_token_ids = nullptr);

} // namespace ninfer::ops
