#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Launches one block per (slot,row) of the validated token_logprobs operands.
void token_logprobs_launch(const Tensor& logits, const Tensor& tokens, const Tensor& row_enabled,
                           const Tensor* counts, const Tensor* columns, std::int32_t token_domain,
                           Tensor& token_logprob, Tensor& top_ids, Tensor& top_logprobs,
                           cudaStream_t stream);

} // namespace ninfer::ops::detail
