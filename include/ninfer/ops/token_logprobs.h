#pragma once

#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

// Largest number of ranked alternatives one slot reports.
inline constexpr std::int32_t kMaximumTopLogprobs = 20;

/**
 * Op: token_logprobs
 *
 * Math / indexing:
 *   Slot (i,b), 0 <= i < W and 0 <= b < B, is active iff row_enabled[b] != 0 and
 *   (counts == nullptr or i < counts[b]). For an active slot let
 *
 *     c     = columns == nullptr ? i : columns[i,b]
 *     z_v   = float(logits[v,c,b])                      for 0 <= v < token_domain
 *     lse   = log(sum_v exp(z_v))
 *     r_0, r_1, ...  = the ids v ordered by z_v descending, lower id first among equal z_v
 *
 *     token_logprob[i,b]  = z_{tokens[i,b]} - lse
 *     top_ids[k,i,b]      = r_k                         for 0 <= k < K
 *     top_logprobs[k,i,b] = z_{r_k} - lse               for 0 <= k < K
 *
 *   This is the log-softmax of the represented logits at temperature 1 over the whole token
 *   domain: no sampling temperature, penalty, truncation, or eligibility mask participates.
 *
 * Logical shapes:
 *   logits is BF16 [physical_rows,C,B]; tokens is I32 [W,B]; row_enabled is I32 [B]; the optional
 *   counts is I32 [B]; the optional columns is I32 [W,B]; token_logprob is FP32 [W,B]; top_ids is
 *   I32 [K,W,B]; top_logprobs is FP32 [K,W,B]. W >= 1, B >= 1, 1 <= K <= kMaximumTopLogprobs,
 *   and K <= token_domain <= physical_rows. Without columns, W <= C. For every active slot,
 *   tokens[i,b] is in [0,token_domain) and columns[i,b], when supplied, is in [0,C).
 *
 * Supported domain:
 *   row_enabled and counts are contiguous. Every other tensor may be a strided view along its
 *   W (or C) and B axes, so a caller can bind a prefix of a larger panel directly; the leading
 *   physical_rows and K axes are dense. Logits of active slots are finite.
 *
 * Numeric:
 *   BF16 logits decode exactly to FP32. top_ids is exact, including the lower-id tie break and
 *   +0 == -0. token_logprob and top_logprobs are FP32 approximations of the formula.
 *
 * Effects:
 *   Writes the three outputs of every active slot and leaves every inactive slot's outputs
 *   untouched. Inputs are unchanged; no output overlaps an input or another output.
 *
 * Workspace:
 *   None.
 *
 * Execution:
 *   Launch shape depends only on W and B, so a captured launch stays valid for every
 *   row_enabled, counts, and columns value.
 */
void token_logprobs(const Tensor& logits, const Tensor& tokens, const Tensor& row_enabled,
                    const Tensor* counts, const Tensor* columns, std::int32_t token_domain,
                    Tensor& token_logprob, Tensor& top_ids, Tensor& top_logprobs,
                    cudaStream_t stream);

} // namespace ninfer::ops
