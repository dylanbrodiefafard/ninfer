#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void gated_delta_net_layer_control_launch(const Tensor& x, const Tensor& a_weight,
                                          const Tensor& b_weight, const Tensor& ssm_a,
                                          const Tensor& dt_bias, Tensor& g, Tensor& beta,
                                          cudaStream_t stream);

void gated_delta_net_layer_conv_launch(const Tensor& projected_qkv, const Tensor& conv_weight,
                                       const Tensor& conv_state_in, Tensor& conv_state_out,
                                       Tensor& q, Tensor& k, Tensor& v, cudaStream_t stream);

void gated_delta_net_layer_norm_launch(const Tensor& recurrent, const Tensor& z,
                                       const Tensor& norm_weight, Tensor& normalized_gated,
                                       cudaStream_t stream);

} // namespace ninfer::ops::detail
