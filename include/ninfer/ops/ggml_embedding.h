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

/**
 * Decode T rows of the same preview Q4_K token embedding.
 *
 * `token_ids` is contiguous device I32 `[T]`, `out` is contiguous BF16 `[2560,T]`, and T is in
 * [1,4096]. Every token id is promised by the caller to be in `[0,weight.n)`. Each represented
 * output is the exact selected stored value rounded once to BF16. The operation performs no host
 * floating-point work, allocation, or persistent mutation. Output may not overlap ids or weight.
 */
void ggml_q4_k_embedding(const Weight& weight, const Tensor& token_ids, Tensor& out,
                         cudaStream_t stream);

} // namespace ninfer::ops
