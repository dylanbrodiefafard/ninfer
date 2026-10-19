#pragma once

#include "core/tensor.h"
#include "ninfer/ops/a4_activation.h"
#include "ninfer/ops/a8_activation.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Applies RMS normalization over ne[0] and an elementwise SiLU gate. For each logical row r:
 *
 *   inv_r    = 1 / sqrt((1/D) * sum_d x[d,r]^2 + eps)
 *   ideal[d,r] = x[d,r] * inv_r * weight[d] * SiLU(z[d,r]).
 *
 * `x`, `z`, and `out` are same-shaped contiguous BF16 tensors, `weight` is contiguous BF16 [D],
 * and eps is positive and finite. This form does not apply a unit offset to weight. Inputs and
 * output must not overlap. The oracle evaluates `ideal` naively in FP64 from the represented
 * inputs. The BF16 output is promoted and compared directly with that result; output storage
 * rounding belongs to the Op's numerical criterion, not the oracle. Kernel reduction, staging,
 * and accumulator precision are implementation choices. There is no workspace or persistent state
 * side effect.
 */
void gated_rmsnorm(const Tensor& x, const Tensor& weight, const Tensor& z, float eps, Tensor& out,
                   cudaStream_t stream);

/**
 * gated_rmsnorm over x/z [128, 48, T] (D=128 per head) whose BF16 result, bit-identical to
 * gated_rmsnorm's, is published only as an A8 activation (ninfer/ops/a8_activation.h) of
 * [6144,T]: one scale per token column covers all 48 heads. weight is BF16 [128].
 *
 * x/z are contiguous, 16-byte aligned BF16 with T >= 1; activation.codes is 16-byte aligned. No
 * input overlaps an output. The codes/scales are checked exactly against the A8 codec of the
 * gated_rmsnorm output. Invalid arguments throw std::invalid_argument. No workspace.
 */
void gated_rmsnorm_a8(const Tensor& x, const Tensor& weight, const Tensor& z, float eps,
                      A8Activation& activation, cudaStream_t stream);

/**
 * gated_rmsnorm over x/z [128, 48, T] (D=128 per head) whose BF16 result, bit-identical to
 * gated_rmsnorm's, is published only as an NVFP4 activation (ninfer/ops/a4_activation.h) of
 * [6144,T] for a weight with activation.input_scale_divisor. weight is BF16 [128].
 *
 * x/z are contiguous, 16-byte aligned BF16 with T >= 1. No input overlaps an output. The
 * codes/scales are checked exactly against the A4 codec of the gated_rmsnorm output. Invalid
 * arguments throw std::invalid_argument. No workspace.
 */
void gated_rmsnorm_a4(const Tensor& x, const Tensor& weight, const Tensor& z, float eps,
                      A4Activation& activation, cudaStream_t stream);

} // namespace ninfer::ops
