#pragma once
#include "ninfer/ops/sigmoid_mul.h"

namespace ninfer::ops::detail {
void sigmoid_mul_a4_launch(const Tensor& gate, const Tensor& input, Tensor* normalized,
                           A4Activation& output, cudaStream_t stream);
}
