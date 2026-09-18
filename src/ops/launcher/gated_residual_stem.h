#pragma once
#include "core/tensor.h"
#include <cuda_runtime.h>
namespace ninfer::ops::detail {
void gated_residual_stem_add_launch(const Tensor& embedding, const Tensor& hidden,
                                    Tensor& out, cudaStream_t stream);
}
