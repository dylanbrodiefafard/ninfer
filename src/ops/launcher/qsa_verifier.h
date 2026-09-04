#pragma once

#include "ninfer/ops/qsa.h"

namespace ninfer::ops::detail {

void qsa_bf16_project_launch(const Tensor& x, const Weight& weight, Tensor& out,
                             cudaStream_t stream);
void qsa_core_norm_rope_launch(const Tensor& raw_query_gate, const Tensor& raw_key,
                               const Tensor& position, const Tensor& query_norm,
                               const Tensor& key_norm, Tensor& query, Tensor& key,
                               cudaStream_t stream);
void qsa_output_gate_launch(const Tensor& attention, const Tensor& raw_query_gate, Tensor& gated,
                            cudaStream_t stream);

} // namespace ninfer::ops::detail
