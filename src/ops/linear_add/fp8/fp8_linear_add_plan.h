#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// Widths from which AllowA8 runs the A8 residual projection. Narrower widths run A16 under either
// policy, and A16Only is registered only there: its verify launches span at most
// kDflash2TreeMaxWidth columns.
inline constexpr std::int32_t kFp8LinearAddFirstA8Residual6144  = 22;
inline constexpr std::int32_t kFp8LinearAddFirstA8Residual17408 = 25;

[[nodiscard]] constexpr std::int32_t fp8_linear_add_first_a8(std::int32_t input_rows) noexcept {
    return input_rows == 6144 ? kFp8LinearAddFirstA8Residual6144
                              : kFp8LinearAddFirstA8Residual17408;
}

[[nodiscard]] std::size_t fp8_linear_add_workspace_capacity_bytes(std::int32_t output_rows,
                                                                  std::int32_t input_rows,
                                                                  LinearPolicy policy,
                                                                  std::int32_t min_tokens,
                                                                  std::int32_t max_tokens);

void fp8_linear_add_decode_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                  cudaStream_t stream);
void fp8_linear_add_small_t_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                   cudaStream_t stream);
void fp8_linear_add_a8_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                              WorkspaceArena& workspace, cudaStream_t stream);

void fp8_linear_add_dispatch(const Tensor& x, const Weight& weight, Tensor& residual,
                             LinearPolicy policy, WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
