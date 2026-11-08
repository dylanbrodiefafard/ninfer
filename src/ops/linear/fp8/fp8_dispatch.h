#pragma once

#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

[[nodiscard]] std::size_t fp8_linear_workspace_capacity_bytes(std::int32_t output_rows,
                                                              std::int32_t input_rows,
                                                              LinearPolicy policy,
                                                              std::int32_t min_tokens,
                                                              std::int32_t max_tokens);

// The FP8 vocabulary projection [248320,5120]; every other FP8 problem throws
// std::invalid_argument. It requires no workspace.
void fp8_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                  cudaStream_t stream);

} // namespace ninfer::ops::detail
