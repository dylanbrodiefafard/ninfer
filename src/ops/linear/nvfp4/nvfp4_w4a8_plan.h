#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// The BF16-activation tensor-core route of the DFlash feature, QKV and attention-output
// projections at T>=2, which runs on the W4A8 MMA family.
void launch_nvfp4_a16_mma(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
