#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

[[nodiscard]] std::size_t nvfp4_linear_swiglu_workspace_capacity_bytes(LinearPolicy policy,
                                                                       std::int32_t min_tokens,
                                                                       std::int32_t max_tokens);

void nvfp4_linear_swiglu_decode_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                       cudaStream_t stream);
void nvfp4_linear_swiglu_small_t_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                        cudaStream_t stream);
void nvfp4_linear_swiglu_w4a4_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                     WorkspaceArena& workspace, cudaStream_t stream);
// The W4A4 projection of an already quantized [5120, tokens] activation (codes and tiled scales).
void nvfp4_linear_swiglu_w4a4_project(const Weight& weight, std::int32_t tokens,
                                      Nvfp4W4a4Workspace activation, Tensor& out,
                                      cudaStream_t stream);
// The same projection publishing its [17408, tokens] output as the NVFP4 activation of the down
// projection (codes and tiled scales, output_input_scale_divisor) instead of BF16.
void nvfp4_linear_swiglu_w4a4_project_a4(const Weight& weight, std::int32_t tokens,
                                         Nvfp4W4a4Workspace activation, Nvfp4W4a4Workspace output,
                                         float output_input_scale_divisor, cudaStream_t stream);
void nvfp4_linear_swiglu_w4a8_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                     WorkspaceArena& workspace, cudaStream_t stream);
void nvfp4_rmsnorm_linear_swiglu_launch(const Tensor& x, const Tensor& norm_weight, float eps,
                                        const Weight& gate_up, Tensor& out,
                                        WorkspaceArena& workspace, cudaStream_t stream);

void nvfp4_linear_swiglu_dispatch(const Tensor& x, const Weight& weight, Tensor& out,
                                  LinearPolicy policy, WorkspaceArena& workspace,
                                  cudaStream_t stream);

} // namespace ninfer::ops::detail
