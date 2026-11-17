#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

#include <cstdint>

namespace ninfer::ops {

// Qwen3.8-Flash-Next routing: 512 routed experts, ten per token, plus one shared expert.
inline constexpr std::int32_t kMoeExperts = 512;
inline constexpr std::int32_t kMoeTopK    = 10;
// Rows of the packed control projection whose FP32 output moe_route consumes: the 512 router
// rows, the shared-expert gate row, and zero rows padding the matrix to a 32-row tile multiple.
inline constexpr std::int32_t kMoeSharedGateRow = kMoeExperts;
inline constexpr std::int32_t kMoeControlRows   = 544;

/**
 * Op: moe_route
 *
 * For each token t, with r = logits[0..511, t] (router logits) and s = logits[512, t] (the
 * shared-expert gate logit):
 *
 *   p             = softmax(r)                                   over all 512 experts
 *   expert_ids    = the ten experts of largest p in descending order; equal FP32 logits keep
 *                   the lower expert id first (a stable descending sort)
 *   route_weights = p[expert_ids] / sum(p[expert_ids])           (renormalized over the ten)
 *   shared_scale  = sigmoid(s)
 *
 * logits is FP32 [544,T] (rows 513..543 are ignored); expert_ids is I32 [10,T]; route_weights is
 * FP32 [10,T]; shared_scale is FP32 [T]. All are contiguous, 16-byte aligned and pairwise
 * disjoint; T >= 1; logits are finite. The oracle evaluates p in FP64 from the represented FP32
 * logits. Ids are exact. The implementation evaluates the weights as exp(r_i - r_max) normalized
 * over the selected ten in FP32, which equals the renormalized full softmax mathematically. No
 * workspace.
 *
 * With nonempty valid_columns, tokens are packed as t = b * packed_width + c, T = B *
 * packed_width. valid_columns is device I32 [B], contiguous, 16-byte aligned, disjoint from
 * every other tensor; the caller guarantees 1 <= valid_columns[b] <= packed_width. For each
 * padding column c >= valid_columns[b], evaluate the formula from logits at
 * b * packed_width + valid_columns[b] - 1 instead. Valid columns are unchanged, including
 * their implementation's output bits. Empty valid_columns means all T columns are valid
 * and requires packed_width == 1. All inputs and outputs live until stream completes.
 * Invalid host-visible arguments throw std::invalid_argument. Asynchronous on stream and
 * graph-capturable for fixed T and packed_width, with device counts allowed to vary on replay.
 */
void moe_route(const Tensor& logits, Tensor& expert_ids, Tensor& route_weights,
               Tensor& shared_scale, const Tensor& valid_columns, std::int32_t packed_width,
               cudaStream_t stream);

} // namespace ninfer::ops
