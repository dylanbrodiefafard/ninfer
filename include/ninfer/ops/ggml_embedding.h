#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

/**
 * Decode one row of the preview's device-resident Q4_K token embedding.
 *
 * `weight` is an unrepacked GGML Q4_K rank-two matrix `[vocabulary,2560]`, `token_id` is a
 * checked host scalar, and `out` is contiguous BF16 `[2560]`. The GPU exact-decodes the selected
 * row and rounds each value once at the BF16 output boundary. The operation performs no host
 * floating-point work, allocation, or persistent mutation.
 */
void ggml_q4_k_embedding_row(const Weight& weight, std::int32_t token_id, Tensor& out,
                             cudaStream_t stream);

} // namespace ninfer::ops
