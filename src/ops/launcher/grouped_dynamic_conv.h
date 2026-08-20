#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

void grouped_dynamic_conv_bf16_gemv_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                           cudaStream_t stream);
void grouped_dynamic_conv_prepare_launch(const Tensor& hidden, const Tensor& base_kernel,
                                         const Tensor& projection, Tensor& prepared,
                                         Tensor& finish_dynamic, cudaStream_t stream);
void grouped_dynamic_conv_finish_launch(const Tensor& hidden, const Tensor& base_kernel,
                                        const Tensor& finish_dynamic, Tensor& out,
                                        cudaStream_t stream);

} // namespace ninfer::ops::detail
