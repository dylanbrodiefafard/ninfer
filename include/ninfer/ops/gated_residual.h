#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>

namespace ninfer::ops {

/** Caller-owned transient capacity for the fixed Qwen4-preview C=1/T=1 GR profile. */
[[nodiscard]] std::size_t gated_residual_workspace_capacity_bytes();

/**
 * Op: gated_residual_read
 *
 * For contiguous BF16 residual branches R [2560,4], independently applies RMSNorm with epsilon
 * 1e-6 and the converted GGUF FP32 gamma [10240]. GGUF has already folded the source checkpoint's
 * zero-centered unit offset into gamma. For their branch-major concatenation Rhat:
 *
 *   u = SiLU(W_down Rhat / 4)
 *   G = reshape(sigmoid(W_up u), [4,2560])
 *   x = (1/4) * sum_i G_i * Rhat_i.
 *
 * down_weight is GGML Q8_0 [320,10240], up_weight is GGML Q8_0 [10240,320], and x is the
 * represented BF16 [2560] output. This verifier entry admits exactly C=1/T=1. The ideal oracle
 * exact-decodes both matrices and evaluates the complete formula naively in FP64 from represented
 * inputs; internal BF16 staging is an implementation profile, not an additional semantic output.
 * Inputs, weights, output, and live workspace are pairwise non-overlapping. Execution is
 * asynchronous on stream.
 */
void gated_residual_read(const Tensor& residual, const Tensor& norm_weight,
                         const Weight& down_weight, const Weight& up_weight, Tensor& x,
                         WorkspaceArena& workspace, cudaStream_t stream);

/**
 * Op: gated_residual_read_write
 *
 * Applies the complete read and publishes represented BF16 write_scale [4]:
 *
 *   write_scale = 2 * sigmoid(W_write Rhat / 4).
 *
 * write_weight is FP32 physical [10240,4], mathematical [4,10240]. Other effects, oracle,
 * aliasing, and execution rules match gated_residual_read.
 */
void gated_residual_read_write(const Tensor& residual, const Tensor& norm_weight,
                               const Weight& down_weight, const Weight& up_weight,
                               const Tensor& write_weight, Tensor& x, Tensor& write_scale,
                               WorkspaceArena& workspace, cudaStream_t stream);

/**
 * Op: gated_residual_inject
 *
 * For BF16 block_output [2560] and represented BF16 write_scale [4], writes
 * `residual_out[:,i] = residual[:,i] + write_scale[i] * block_output`. residual_out is BF16
 * [2560,4] and may alias residual exactly; every other overlap is forbidden. There is no
 * workspace or persistent state.
 */
void gated_residual_inject(const Tensor& residual, const Tensor& block_output,
                           const Tensor& write_scale, Tensor& residual_out,
                           cudaStream_t stream);

} // namespace ninfer::ops
