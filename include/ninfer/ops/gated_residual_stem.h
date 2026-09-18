#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include <cuda_runtime.h>

namespace ninfer::ops {

[[nodiscard]] std::size_t gated_residual_stem_workspace_capacity_bytes(std::int32_t max_tokens);

/** Source-defined four-stream stem, not an MTP rollout or transaction.
 * e [2560,T], R [2560,4,T], norm_e [2560], norm_R [10240] and output [2560,4,T]
 * are contiguous BF16. Both weights are contiguous BF16 [2560,2560]. For each token:
 *   en = e / sqrt(mean(e^2)+1e-6) * (1+norm_e)
 *   rn = vec(R) / sqrt(mean(vec(R)^2)+1e-6) * (1+norm_R)
 *   output[:,j] = W_hidden * rn[2560*j:2560*(j+1)] + W_embedding * en
 * Hidden normalization is ONCE over 10240, never separately per branch. The independent
 * ideal evaluates this whole formula in FP64 from represented public inputs; private BF16
 * normalization/projection staging is not semantic. Only the final output is observable.
 * T in [1,4096]. All input/weight/output/workspace storage is pairwise disjoint. Caller owns
 * workspace; scope is restored after enqueue. No persistent state or hidden allocation.
 * Async/graph-capturable on stream. Numerical profile: relative L2 <= .01, max absolute
 * error <= .005 + .02*maxabs(ideal), finite. This does not admit full MTP execution.
 */
void gated_residual_stem(const Tensor& embedding, const Tensor& hidden,
                         const Tensor& embedding_norm, const Tensor& hidden_norm,
                         const Weight& embedding_weight, const Weight& hidden_weight,
                         Tensor& out, WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops
