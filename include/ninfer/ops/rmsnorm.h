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

/**
 * Applies two independent offset RMS normalizations. For panel p in {0,1}, row r, and d:
 *
 *   inv_p[r] = 1 / sqrt(sum_j x_p[j,r]^2 / 5120 + eps)
 *   ideal_p[d,r] = x_p[d,r] * inv_p[r] * (1 + weight_p[d]).
 *
 * Each input and output is contiguous BF16 [5120,T] with the same positive T; each weight is
 * contiguous BF16 [5120]. eps is positive and finite. All six tensors have disjoint storage.
 * Each output is separate BF16 storage; the panels never share a statistic or weight. The RMS
 * family FP64 oracle applies independently to each panel, from represented inputs. The dual
 * BF16 profile bounds relative L2 by 1.85e-3 and gross error by 1e-5 + 4e-3 * max(abs(ideal_p));
 * both bounds must hold with finite outputs. The gross coefficient admits the BF16 rounding
 * envelope (1/256) plus FP32 normalization error. Private arithmetic and reduction order are
 * implementation choices.
 *
 * Only the two outputs are written. There is no workspace, allocation, or persistent state.
 * Operands remain valid through execution on stream; capture and replay use stable addresses.
 * Invalid shape, dtype, layout, scalar, pointer, or overlap throws before submitting work.
 */
void dual_offset_rmsnorm(const Tensor& x0, const Tensor& weight0, const Tensor& x1,
                         const Tensor& weight1, float eps, Tensor& out0, Tensor& out1,
                         cudaStream_t stream);

} // namespace ninfer::ops
