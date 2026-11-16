#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"
#include "ops/linear/fp8/fp8_a8_plan.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// input_rows selects the registered parent: [14336,5120] or [13312,2560].
[[nodiscard]] std::size_t nvfp4_attn_input_workspace_capacity_bytes(std::int32_t input_rows,
                                                                    LinearPolicy policy,
                                                                    std::int32_t min_tokens,
                                                                    std::int32_t max_tokens);

// The T=1 GEMV of either registered parent, selected by weight.k.
void nvfp4_attn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                    Tensor& k, Tensor& v, cudaStream_t stream);

// The BF16-activation W4A8-family MMA of the [13312,2560] parent at T>=2.
void nvfp4_attn_input_a16_mma_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                     Tensor& k, Tensor& v, cudaStream_t stream);

void nvfp4_attn_input_small_t_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                     Tensor& k, Tensor& v, cudaStream_t stream);

void nvfp4_attn_input_w4a4_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                  Tensor& k, Tensor& v, Nvfp4W4a4Workspace workspace,
                                  cudaStream_t stream);

// The W4A4 projection of an already quantized [5120, tokens] activation (codes and tiled scales).
void nvfp4_attn_input_w4a4_project(const Weight& weight, std::int32_t tokens,
                                   Nvfp4W4a4Workspace activation, Tensor& q, Tensor& gate,
                                   Tensor& k, Tensor& v, cudaStream_t stream);

void nvfp4_attn_input_w4a8_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                  Tensor& k, Tensor& v, Fp8A8Workspace workspace,
                                  cudaStream_t stream);
// The W4A8 projection of an already quantized [5120, tokens] activation.
void nvfp4_attn_input_w4a8_project(const Weight& weight, std::int32_t tokens,
                                   Fp8A8Workspace activation, Tensor& q, Tensor& gate, Tensor& k,
                                   Tensor& v, cudaStream_t stream);

void nvfp4_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                               Tensor& k, Tensor& v, LinearPolicy policy, WorkspaceArena* workspace,
                               cudaStream_t stream);

} // namespace ninfer::ops::detail
