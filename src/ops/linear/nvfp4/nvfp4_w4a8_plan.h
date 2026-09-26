#pragma once

#include "ops/linear/fp8/fp8_a8_plan.h"

namespace ninfer::ops::detail {

void launch_nvfp4_w4a8(const Tensor& x, const Weight& weight, Tensor& out,
                      Fp8A8Workspace workspace, cudaStream_t stream);
void launch_nvfp4_a16_mma(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
