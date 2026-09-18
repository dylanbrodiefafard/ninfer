#pragma once
#include "ops/linear/fp8/fp8_dispatch.h"
#include "ops/linear/fp8/fp8_a8_plan.h"

namespace ninfer::ops::detail {
bool is_fp8_tensor_problem(std::int32_t n, std::int32_t k);
std::int32_t fp8_tensor_a8_first_t(std::int32_t n, std::int32_t k);
void validate_fp8_tensor_weight(const Weight& weight, const char* operation);
std::size_t fp8_tensor_workspace_capacity_bytes(std::int32_t n, std::int32_t k,
    LinearPolicy policy, std::int32_t min_tokens, std::int32_t max_tokens);
void fp8_tensor_dispatch(const Tensor& x, const Weight& weight, Tensor& out,
    LinearPolicy policy, WorkspaceArena* workspace, cudaStream_t stream);
// Private QSA key route: exactly [512,2560], FP32 accumulator output until norm/RoPE.
void fp8_tensor_key_f32(const Tensor& x,const Weight& weight,Tensor& out,
    cudaStream_t stream);
// Explicit temporary codec used by the tensor-calibrated A8 route and exact codec tests.
void launch_fp8_tensor_quantize(const Tensor& x, const Weight& weight,
    Fp8A8Workspace scratch, cudaStream_t stream);
} // namespace ninfer::ops::detail
