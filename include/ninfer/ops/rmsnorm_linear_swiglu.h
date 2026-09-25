#pragma once

#include "ninfer/ops/linear_swiglu.h"

namespace ninfer::ops {

/**
 * RMSNorm with unit-offset weights followed by NVFP4/A8 LinearSwiGLU.
 * x is contiguous BF16 [5120,T], norm_weight is BF16 [5120], gate_up is
 * NVFP4 [34816,5120], and out is BF16 [17408,T], with positive T.
 * The normalized input is explicitly rounded to BF16 before row-scaled E4M3
 * activation quantization. Stored NVFP4 weight codes/scales are unchanged.
 * The canonical formula is SwiGLU(W * BF16(RMSNorm(x,1+norm_weight,eps)));
 * A8 codec distortion is checked separately from arithmetic residual.
 * Workspace is caller-owned transient storage. Inputs and output do not alias.
 * x/out require 16-byte alignment; norm_weight requires 4-byte alignment.
 */
[[nodiscard]] std::size_t rmsnorm_linear_swiglu_workspace_capacity_bytes(std::int32_t tokens);

void rmsnorm_linear_swiglu(const Tensor& x, const Tensor& norm_weight, float eps,
                          const Weight& gate_up, Tensor& out, WorkspaceArena& workspace,
                          cudaStream_t stream);

} // namespace ninfer::ops
