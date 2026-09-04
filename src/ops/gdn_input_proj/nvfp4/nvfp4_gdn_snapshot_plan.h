#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

enum class Nvfp4GdnConvScheduleId {
    DecodeFusedA16,
    SmallTFusedA16,
    Materialized,
};

struct Nvfp4GdnConvPlan {
    Nvfp4GdnConvScheduleId schedule;
};

Nvfp4GdnConvPlan nvfp4_gdn_conv_resolve_plan(LinearPolicy policy, std::int32_t tokens,
                                             std::int32_t batch_size);

[[nodiscard]] std::size_t nvfp4_gdn_decode_columns_workspace_bytes();

[[nodiscard]] std::size_t nvfp4_gdn_snapshot_workspace_capacity_bytes(LinearPolicy policy,
                                                                      std::int32_t min_tokens,
                                                                      std::int32_t max_tokens);

void nvfp4_gdn_snapshot_decode_launch(const Tensor& x, const Weight& weight,
                                      const Tensor& conv_weight, Tensor& conv_states,
                                      const Tensor& valid_columns, const Tensor& initial_slot,
                                      const Tensor& snapshot_base_slot, Tensor& query, Tensor& key,
                                      Tensor& value, Tensor& z, cudaStream_t stream);

void nvfp4_gdn_record_decode_launch(const Tensor& x, const Weight& weight,
                                    const Tensor& conv_weight, const Tensor& conv_states,
                                    const Tensor& valid_columns, const Tensor& initial_slot,
                                    Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value,
                                    Tensor& z, cudaStream_t stream);

void nvfp4_gdn_snapshot_decode_columns(const Tensor& x, const Weight& weight,
                                       const Tensor& conv_weight, Tensor& conv_states,
                                       const Tensor& valid_columns, const Tensor& initial_slot,
                                       const Tensor& snapshot_base_slot, Tensor& query, Tensor& key,
                                       Tensor& value, Tensor& z, WorkspaceArena& workspace,
                                       cudaStream_t stream);

void nvfp4_gdn_record_decode_columns(const Tensor& x, const Weight& weight,
                                     const Tensor& conv_weight, const Tensor& conv_states,
                                     const Tensor& valid_columns, const Tensor& initial_slot,
                                     Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value,
                                     Tensor& z, WorkspaceArena& workspace, cudaStream_t stream);

void nvfp4_gdn_snapshot_small_t_launch(const Tensor& x, const Weight& weight,
                                       const Tensor& conv_weight, Tensor& conv_states,
                                       const Tensor& valid_columns, const Tensor& initial_slot,
                                       const Tensor& snapshot_base_slot, Tensor& query, Tensor& key,
                                       Tensor& value, Tensor& z, cudaStream_t stream);

void nvfp4_gdn_record_small_t_launch(const Tensor& x, const Weight& weight,
                                     const Tensor& conv_weight, const Tensor& conv_states,
                                     const Tensor& valid_columns, const Tensor& initial_slot,
                                     Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value,
                                     Tensor& z, cudaStream_t stream,
                                     const std::int32_t* parent_index = nullptr);

// Packed T=2..16 record: one launch of T=1 GEMV+FP32 conv per sequence. Weights stay in L2
// across the token loop; each column matches ordinary-decode GEMV+BF16 3-tap history.
void nvfp4_gdn_record_t1_fused_launch(const Tensor& x, const Weight& weight,
                                      const Tensor& conv_weight, const Tensor& conv_states,
                                      const Tensor& valid_columns, const Tensor& initial_slot,
                                      Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value,
                                      Tensor& z, cudaStream_t stream,
                                      const std::int32_t* parent_index = nullptr);

void nvfp4_gdn_snapshot_post_launch(const Tensor& projected, const Tensor& conv_weight,
                                    Tensor& conv_states, const Tensor& valid_columns,
                                    const Tensor& initial_slot, const Tensor& snapshot_base_slot,
                                    Tensor& query, Tensor& key, Tensor& value, cudaStream_t stream);

void nvfp4_gdn_record_post_launch(const Tensor& conv_record, const Tensor& conv_weight,
                                  const Tensor& conv_states, const Tensor& valid_columns,
                                  const Tensor& initial_slot, Tensor& query, Tensor& key,
                                  Tensor& value, cudaStream_t stream);

void nvfp4_gdn_snapshot_dispatch(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                                 Tensor& conv_states, const Tensor& valid_columns,
                                 const Tensor& initial_slot, const Tensor& snapshot_base_slot,
                                 Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                                 LinearPolicy policy, WorkspaceArena& workspace,
                                 cudaStream_t stream);

} // namespace ninfer::ops::detail
