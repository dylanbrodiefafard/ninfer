#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

void nll_from_logits_launch(const Tensor& logits, const Tensor& targets, Tensor& out,
                            std::int32_t valid_rows, cudaStream_t stream);

} // namespace ninfer::ops::detail
