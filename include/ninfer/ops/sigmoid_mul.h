#pragma once

#include "core/tensor.h"
#include "ninfer/ops/a4_activation.h"
#include "ninfer/ops/a8_activation.h"

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

/**
 * Elementwise sigmoid gate:
 *
 *   ideal[i] = x[i] * (1 / (1 + exp(-gate[i]))).
 *
 * `gate` and `x` are non-overlapping, same-shaped contiguous BF16 tensors. The oracle evaluates
 * `ideal` in FP64 from the represented inputs. The updated BF16 x is promoted and compared directly
 * with that result; output storage rounding belongs to the Op's numerical criterion, not the
 * oracle. Private kernel arithmetic is implementation-defined. The Op uses no workspace or other
 * persistent state.
 */
void sigmoid_mul(const Tensor& gate, Tensor& x, cudaStream_t stream);

/**
 * sigmoid_mul over gate/x [6144,T] whose BF16 result, bit-identical to sigmoid_mul's, is
 * published only as an A8 activation (ninfer/ops/a8_activation.h) of [6144,T]; x is not
 * modified.
 *
 * gate/x are contiguous, 16-byte aligned BF16 with T >= 1; activation.codes is 16-byte aligned.
 * No input overlaps an output. The codes/scales are checked exactly against the A8 codec of the
 * sigmoid_mul output. Invalid arguments throw std::invalid_argument. No workspace.
 */
void sigmoid_mul_a8(const Tensor& gate, const Tensor& x, A8Activation& activation,
                    cudaStream_t stream);

/**
 * Publishes the A4 codec of BF16(x * sigmoid(gate)) without modifying either input.
 * The domain is aligned contiguous BF16 gate/x[6144,T] and tiled A4[6144,T].
 * Optional normalized publishes the same represented BF16 values; all operands are disjoint.
 * The primary oracle is the complete FP64 formula, explicit BF16 cast, and independent exact
 * signed G16 codec. Private FP32 evaluation may differ at BF16 cast ties. There is no workspace
 * or persistent state; caller-owned outputs remain live through stream completion. Invalid
 * shapes, dtypes, pointers, alignment, divisors or overlaps throw before submitting work.
 */
void sigmoid_mul_a4(const Tensor& gate, const Tensor& x, Tensor* normalized,
                    A4Activation& activation, cudaStream_t stream);

} // namespace ninfer::ops
