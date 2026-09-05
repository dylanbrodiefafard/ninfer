#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

void ggml_q4_k_embedding_row_launch(const Weight& weight, std::int32_t token_id, Tensor& out,
                                    cudaStream_t stream);
void ggml_q4_k_embedding_launch(const Weight& weight, const Tensor& token_ids, Tensor& out,
                                cudaStream_t stream);

} // namespace ninfer::ops::detail
