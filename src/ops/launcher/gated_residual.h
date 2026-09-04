#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void gated_residual_normalize_launch(const Tensor& residual, const Tensor& norm_weight,
                                     Tensor& normalized, cudaStream_t stream);
void gated_residual_activate_launch(Tensor& low_rank, cudaStream_t stream);
void gated_residual_mix_launch(const Tensor& normalized, const Tensor& up_logits, Tensor& x,
                               cudaStream_t stream);
void gated_residual_write_launch(const Tensor& normalized, const Tensor& write_weight,
                                 Tensor& write_scale, cudaStream_t stream);
void gated_residual_inject_launch(const Tensor& residual, const Tensor& block_output,
                                  const Tensor& write_scale, Tensor& residual_out,
                                  cudaStream_t stream);

} // namespace ninfer::ops::detail
