#pragma once

// ninfer::ops::detail - private launch prototype for argmax.

#include "core/tensor.h"
#include "ninfer/ops/sampling.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void argmax_launch(const Tensor& logits, Tensor& out, std::int32_t valid_rows, cudaStream_t stream);
void argmax_launch(const Tensor& logits, Tensor& out, std::int32_t valid_rows,
                   const SamplingConfig* configs, std::int32_t columns_per_config,
                   cudaStream_t stream);

} // namespace ninfer::ops::detail
