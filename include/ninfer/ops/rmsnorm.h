#pragma once

#include "core/tensor.h"
#include "ninfer/ops/a8_activation.h"

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

/**
 * Applies RMSNorm over the fastest dimension D=ne[0]. For each logical row r:
 *
 *   inv_r    = 1 / sqrt((1/D) * sum_d x[d,r]^2 + eps)
 *   gain[d]  = unit_offset ? 1 + weight[d] : weight[d]
 *   ideal[d,r] = x[d,r] * inv_r * gain[d].
 *
 * `x` and `out` are same-shaped contiguous BF16 tensors, weight is contiguous BF16 [D], and eps
 * is positive and finite. Input, weight, and output must not overlap. The oracle evaluates `ideal`
 * naively in FP64 from the represented inputs. The BF16 output is promoted and compared directly
 * with that result; output storage rounding belongs to the Op's numerical criterion, not the
 * oracle. Kernel reduction, staging, and accumulator precision are implementation choices. There
 * is no workspace or persistent state side effect.
 */
void rmsnorm(const Tensor& x, const Tensor& weight, float eps, bool unit_offset, Tensor& out,
             cudaStream_t stream);

/**
 * Unit-offset RMSNorm over D=5120 that publishes its output as an A8 activation
 * (ninfer/ops/a8_activation.h): activation encodes BF16(ideal) column by column, where `ideal` is
 * rmsnorm's with unit_offset = true. When `normalized` is non-null the same BF16 values are also
 * written there, bit-identical to rmsnorm(x, weight, eps, true, *normalized).
 *
 * x is contiguous BF16 [5120,T] with T >= 1 (no further dimensions), weight contiguous BF16
 * [5120], `normalized` (optional) has x's shape, and `activation` is [5120,T]. x/normalized/
 * activation.codes are 16-byte aligned; weight is 4-byte aligned. No input overlaps an output.
 * The BF16 values follow rmsnorm's criterion; the codes/scales are checked exactly against the A8
 * codec of those values. Invalid arguments throw std::invalid_argument. No workspace.
 */
void rmsnorm_a8(const Tensor& x, const Tensor& weight, float eps, Tensor* normalized,
                A8Activation& activation, cudaStream_t stream);

} // namespace ninfer::ops
