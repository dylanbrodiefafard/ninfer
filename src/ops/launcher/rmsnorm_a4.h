#pragma once
#include "ninfer/ops/rmsnorm.h"

namespace ninfer::ops::detail {
void rmsnorm_a4_launch(const Tensor& x, const Tensor& weight, float eps, Tensor* normalized,
                       A4Activation& activation, cudaStream_t stream);
}
