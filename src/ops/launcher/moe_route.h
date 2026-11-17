#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// The launcher trusts the wrapper's validation: logits FP32 [544,tokens], expert_ids I32
// [10,tokens], route_weights FP32 [10,tokens], shared_scale FP32 [tokens], tokens >= 1.
// valid_columns is null (packed_width=1), or I32 [tokens/packed_width] with counts in
// [1,packed_width]. All tensors are disjoint; packed_width divides tokens.
void moe_route_launch(const float* logits, std::int32_t* expert_ids, float* route_weights,
                      float* shared_scale, std::int32_t tokens, const std::int32_t* valid_columns,
                      std::int32_t packed_width, cudaStream_t stream);

} // namespace ninfer::ops::detail
