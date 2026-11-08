#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void w8_pair_simt_r8_c4_launch(bool full, const Tensor& x, const Weight& first_weight,
                               const Weight& second_weight, Tensor& first_out, Tensor& second_out,
                               cudaStream_t stream);
void w8_pair_simt_r8_c8_launch(bool full, const Tensor& x, const Weight& first_weight,
                               const Weight& second_weight, Tensor& first_out, Tensor& second_out,
                               cudaStream_t stream);
void w8_pair_gemm_mma_launch(bool full, const Tensor& x, const Weight& first_weight,
                             const Weight& second_weight, Tensor& first_out, Tensor& second_out,
                             cudaStream_t stream);

} // namespace ninfer::ops::detail
