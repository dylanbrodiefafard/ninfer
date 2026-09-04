#pragma once

#include "ninfer/ops/qwen4_sparse_moe.h"

namespace ninfer::ops::detail {

void qwen4_sparse_moe_route_launch(const Tensor& x, const Weight& router, Tensor& logits,
                                   Tensor& selected_ids, Tensor& selected_weights,
                                   cudaStream_t stream);

void qwen4_sparse_moe_resident_route_launch(
    const Tensor& x, const Weight& router, const Tensor& shared_gate, Tensor& logits,
    Tensor& selected_ids, Tensor& selected_weights, Tensor& shared_gate_value,
    cudaStream_t stream);

void qwen4_sparse_moe_shared_gate_up_swiglu_launch(
    const Tensor& x, const Weight& gate, const Weight& up, Tensor& activated,
    cudaStream_t stream);

void qwen4_sparse_moe_indexed_gate_up_swiglu_launch(
    const Tensor& x, const Weight& gate_bank, const Weight& up_bank,
    const Tensor& selected_ids, Tensor& activated, cudaStream_t stream);

void qwen4_sparse_moe_indexed_down_finish_launch(
    const Tensor& activated, const Weight& expert_bank, const Tensor& selected_ids,
    const Tensor& selected_weights, const Tensor& shared, const Tensor& shared_gate,
    Tensor& destination, cudaStream_t stream);

void qwen4_sparse_moe_swiglu_launch(const Tensor& gate, const Tensor& up, Tensor& activated,
                                    cudaStream_t stream);

void qwen4_sparse_moe_zero_routed_launch(Tensor& routed, cudaStream_t stream);

void qwen4_sparse_moe_accumulate_launch(const Tensor& expert, const Tensor& selected_weights,
                                        std::int32_t rank, Tensor& routed,
                                        cudaStream_t stream);

void qwen4_sparse_moe_shared_gate_launch(const Tensor& x, const Tensor& shared_gate,
                                         Tensor& gate_value, cudaStream_t stream);

void qwen4_sparse_moe_finish_launch(const Tensor& routed, const Tensor& shared,
                                    const Tensor& shared_gate_value, Tensor& destination,
                                    cudaStream_t stream);

} // namespace ninfer::ops::detail
