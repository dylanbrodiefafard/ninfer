#pragma once

#include "core/tensor.h"
#include "ninfer/ops/ngram_embedding.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

using NgramRowLaunchConfig = PreparedNgramRowConfig;

void ngram_row_ids_launch(const Tensor& input_ids, const Tensor& valid_tokens,
                          const Tensor& old_history, const NgramRowLaunchConfig& config,
                          Tensor& row_ids, Tensor& new_history, cudaStream_t stream);

} // namespace ninfer::ops::detail
