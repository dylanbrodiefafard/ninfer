#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void ggml_block_linear_launch(const Tensor& x, const Weight& w, Tensor& out,
                              cudaStream_t stream);

} // namespace ninfer::ops::detail
