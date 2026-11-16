#pragma once

// ninfer::ops::detail - private launch prototype for stream_mean. Included by the wrapper and
// defined by the launcher. Not part of the semantic contract.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// Assumes the wrapper validated geometry, alignment, column stride, and disjointness.
void stream_mean_launch(const Tensor& residual, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
