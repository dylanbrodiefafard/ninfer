#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// The launchers trust the wrapper's validation (ninfer/ops/moe_expert_partition.h): shapes,
// alignment, disjointness, 1 <= tokens <= 48. local_experts is the U32 [16] bitset; the compact
// outputs have room for `tokens` columns.
void moe_expert_partition_launch(const std::uint16_t* x, const std::int32_t* expert_ids,
                                 const float* route_weights, const std::uint32_t* local_experts,
                                 std::int32_t* compact_map, std::uint32_t* compact_count,
                                 std::uint16_t* compact_x, std::int32_t* compact_ids,
                                 float* compact_weights, std::int32_t tokens, cudaStream_t stream);

// helper has helper_columns columns (>= tokens).
void moe_expert_partial_join_launch(const float* partial, const std::int32_t* compact_map,
                                    const float* helper, std::int32_t helper_columns,
                                    std::uint16_t* y, std::int32_t tokens, cudaStream_t stream);

} // namespace ninfer::ops::detail
