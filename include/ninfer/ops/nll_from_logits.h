#pragma once

#include "core/tensor.h"

#include <cstdint>

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

/**
 * Per-column teacher-forced negative log likelihood from BF16 logits:
 *
 *   nll[t] = logsumexp_{0 <= v < valid_rows}(float(logits[v,t])) - float(logits[targets[t], t])
 *
 * `logits` is contiguous BF16 [physical_rows,T], `targets` is contiguous I32 [T] with each
 * target in [0, valid_rows), and `out` is contiguous FP32 [T]. 1 <= valid_rows <= physical_rows.
 * Physical rows [valid_rows, physical_rows) do not participate. `out` must not overlap logits or
 * targets. The Op has no workspace and changes no state other than writing all of `out`.
 */
void nll_from_logits(const Tensor& logits, const Tensor& targets, Tensor& out,
                     std::int32_t valid_rows, cudaStream_t stream);

} // namespace ninfer::ops
