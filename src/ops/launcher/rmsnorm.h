#pragma once

// ninfer::ops::detail - private launch prototype for rmsnorm.

#include "core/tensor.h"
#include "ninfer/ops/a4_activation.h"
#include "ninfer/ops/a8_activation.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void rmsnorm_launch(const Tensor& x, const Tensor& weight, float eps, bool unit_offset,
                    const Tensor* z, Tensor& out, cudaStream_t stream);

// D=5120 unit-offset RMSNorm publishing the A8 activation of its BF16 output, and that BF16
// output too when `out` is non-null. Inputs already validated by the wrapper.
void rmsnorm_a8_launch(const Tensor& x, const Tensor& weight, float eps, Tensor* out,
                       A8Activation& activation, cudaStream_t stream);

// D=5120 unit-offset RMSNorm publishing the NVFP4 activation of its BF16 output, and that BF16
// output too when `out` is non-null. Inputs already validated by the wrapper.
void rmsnorm_a4_launch(const Tensor& x, const Tensor& weight, float eps, Tensor* out,
                       A4Activation& activation, cudaStream_t stream);

// Gated RMSNorm over [128, 48, T] heads publishing one A8 column per token. Inputs already
// validated by the wrapper.
void gated_rmsnorm_a8_launch(const Tensor& x, const Tensor& weight, const Tensor& z, float eps,
                             A8Activation& activation, cudaStream_t stream);

// Gated RMSNorm over [128, 48, T] heads publishing only the A4 activation of its [6144, T] BF16
// output. Inputs already validated by the wrapper.
void gated_rmsnorm_a4_launch(const Tensor& x, const Tensor& weight, const Tensor& z, float eps,
                             A4Activation& activation, cudaStream_t stream);

void dual_offset_rmsnorm_launch(const Tensor& x0, const Tensor& weight0, const Tensor& x1,
                                const Tensor& weight1, float eps, Tensor& out0, Tensor& out1,
                                cudaStream_t stream);

} // namespace ninfer::ops::detail
