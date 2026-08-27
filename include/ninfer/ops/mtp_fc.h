#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Op: mtp_fc
 *
 * Math / indexing:
 *   out[:, t] = W @ concat(embedding_norm[:, t], hidden_norm[:, t])
 *
 * This is the packed MTP `fc` projection. Concatenation is along the reduction axis: the first
 * D activation rows are `embedding_norm` and the second D rows are `hidden_norm`. The ideal
 * result is the Linear oracle on that concatenated activation.
 *
 * Logical shapes:
 *   BF16 embedding_norm and hidden_norm [D,T], NVFP4 weight [N, 2D], BF16 out [N,T], all
 *   contiguous and 16-byte aligned. The registered domain is D=5120, N=5120 (Qwen3.6-27B MTP
 *   `fc`). T is every positive column count.
 *
 * Numeric:
 *   Same Linear A16 NVFP4 criterion as `linear` on `[5120,10240]`. The implementation may load
 *   the two activation halves directly rather than materializing the concatenated vector.
 *
 * Effects:
 *   Writes the full output. Inputs, weight, and output must not alias.
 *
 * Workspace:
 *   None for the current A16 route.
 */
void mtp_fc(const Tensor& embedding_norm, const Tensor& hidden_norm, const Weight& weight,
            Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
