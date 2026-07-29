#pragma once

#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void nvfp4_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                    cudaStream_t stream);

} // namespace ninfer::ops::detail
