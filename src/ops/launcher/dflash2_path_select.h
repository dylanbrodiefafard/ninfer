#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

void dflash2_path_select_bf16_gemv_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                          cudaStream_t stream);
void dflash2_path_select_launch(const Tensor& logits, const Tensor& hidden_proj,
                                const Tensor& pred_code, const Tensor& succ_code,
                                const Tensor& anchors, Tensor& path, float temperature,
                                unsigned long long seed, cudaStream_t stream,
                                const Tensor* logit_token_ids);

} // namespace ninfer::ops::detail
