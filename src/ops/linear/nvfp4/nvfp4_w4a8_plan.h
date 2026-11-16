#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// The BF16-activation tensor-core route at T>=2 of the DFlash feature, QKV and attention-output
// projections, the Flash-Next drafter's feature, QKV, attention-output, MLP and conv projections,
// and the Flash-Next mixer output projection, which runs on the W4A8 MMA family.
void launch_nvfp4_a16_mma(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
