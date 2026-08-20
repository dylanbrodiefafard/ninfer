#pragma once

// ninfer::ops::detail - residual_rmsnorm launcher: fused residual + RMSNorm over contiguous
// BF16 rows (one launch, replacing the residual_add + rmsnorm pair).
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// x += y (in-place residual update, the residual_add), then out = rmsnorm(x) * (1 + weight)
// (the per-layer unit_offset / Offset RMSNorm). Bit-exact against the unfused pair.
void residual_rmsnorm_launch(const Tensor& y, Tensor& x, const Tensor& weight, float eps,
                             Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail