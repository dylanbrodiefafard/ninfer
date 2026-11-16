#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// The launcher trusts the wrapper's validation: logits FP32 [544,tokens], expert_ids I32
// [10,tokens], route_weights FP32 [10,tokens], shared_scale FP32 [tokens], tokens >= 1.
void moe_route_launch(const float* logits, std::int32_t* expert_ids, float* route_weights,
                      float* shared_scale, std::int32_t tokens, cudaStream_t stream);

} // namespace ninfer::ops::detail
