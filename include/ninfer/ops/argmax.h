#pragma once

#include "core/tensor.h"
#include "ninfer/ops/sampling.h"

#include <cstdint>

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

/**
 * Computes one vocabulary argmax per column:
 *
 *   out[t] = min argmax_{0 <= v < valid_rows} float(logits[v,t]).
 *
 * `logits` is contiguous BF16 [physical_rows,T], `out` is contiguous I32 [T], and
 * 1 <= valid_rows <= physical_rows. Physical rows [valid_rows,physical_rows) do not
 * participate. Equal maxima select the lowest row index. `out` must not overlap `logits`.
 * The Op has no workspace and changes no state other than writing all of `out`.
 */
void argmax(const Tensor& logits, Tensor& out, std::int32_t valid_rows, cudaStream_t stream);

// Greedy selection with row-local token suppression. Config b owns each consecutive
// `columns_per_config` columns; temperature, filters, penalties, RNG, and counts are ignored.
// Each row's config must leave at least one token in valid_rows eligible.
void argmax(const Tensor& logits, Tensor& out, std::int32_t valid_rows,
            const SamplingConfig* configs, std::int32_t columns_per_config, cudaStream_t stream);

} // namespace ninfer::ops
