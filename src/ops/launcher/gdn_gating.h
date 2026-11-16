#pragma once

// ninfer::ops::detail - private launch prototype for gdn_gating.

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void gdn_gating_launch(const Tensor& a, const Tensor& b, const Tensor& A_log, const Tensor& dt_bias,
                       Tensor& g, Tensor& beta, cudaStream_t stream);

// ab is the validated FP32 [96,T] packed projection (a rows 0..47, b rows 48..95).
void gdn_gating_packed_launch(const Tensor& ab, const Tensor& A_log, const Tensor& dt_bias,
                              Tensor& g, Tensor& beta, cudaStream_t stream);

} // namespace ninfer::ops::detail
