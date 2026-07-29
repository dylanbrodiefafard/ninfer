#pragma once

#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void bf16_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                   cudaStream_t stream);

} // namespace ninfer::ops::detail
