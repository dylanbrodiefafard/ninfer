#pragma once

#include "ninfer/ops/ple.h"

namespace ninfer::ops::detail {

void ple_iq4_nl_decode_rows_launch(const Tensor& device_rows, Tensor& embedding,
                                   cudaStream_t stream);
void ple_gate_launch(const Tensor& residual, const Tensor& projected_key,
                     const Tensor& projected_value, const Tensor& key_norm_weight,
                     const Tensor& query_norm_weight, Tensor& gated, cudaStream_t stream);
void ple_conv_input_launch(const Tensor& gated, const Tensor& conv_norm_weight,
                           Tensor& current_state, cudaStream_t stream);
void ple_conv_inject_launch(const Tensor& residual, const Tensor& gated,
                            const Tensor& conv_weight, const Tensor& old_state,
                            const Tensor& current_state, Tensor& out, cudaStream_t stream);
void ple_state_update_launch(const Tensor& old_state, const Tensor& current_state,
                             Tensor& new_state, cudaStream_t stream);

} // namespace ninfer::ops::detail
