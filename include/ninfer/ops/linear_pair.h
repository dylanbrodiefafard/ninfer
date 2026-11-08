#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Computes two same-shaped projections of one input:
 *
 *   first_out[:,t]  = linear(x[:,t], first_weight)
 *   second_out[:,t] = linear(x[:,t], second_weight).
 *
 * `x` is contiguous BF16 [5120,T], both outputs are distinct contiguous BF16 [1024,T], and both
 * weights are RowSplit W8G32_F16S [1024,5120] with FP16 scales. Every positive T is admitted.
 * Numeric semantics are those of linear(). Inputs, outputs, and weight planes must be mutually
 * non-overlapping. The Op requires no caller-owned transient storage and has no persistent state
 * side effect.
 */
void linear_pair(const Tensor& x, const Weight& first_weight, const Weight& second_weight,
                 Tensor& first_out, Tensor& second_out, cudaStream_t stream);

} // namespace ninfer::ops
