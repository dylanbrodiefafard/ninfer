#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

[[nodiscard]] std::size_t nvfp4_gdn_snapshot_workspace_capacity_bytes(LinearPolicy policy,
                                                                      std::int32_t min_tokens,
                                                                      std::int32_t max_tokens);

void nvfp4_gdn_snapshot_decode_launch(const Tensor& x, const Weight& weight,
                                      const Tensor& conv_weight, Tensor& conv_states,
                                      const Tensor& initial_slot, Tensor& query, Tensor& key,
                                      Tensor& value, Tensor& z, cudaStream_t stream);

void nvfp4_gdn_snapshot_small_t_launch(const Tensor& x, const Weight& weight,
                                       const Tensor& conv_weight, Tensor& conv_states,
                                       const Tensor& initial_slot, Tensor& query, Tensor& key,
                                       Tensor& value, Tensor& z, cudaStream_t stream);

void nvfp4_gdn_snapshot_post_launch(const Tensor& projected, const Tensor& conv_weight,
                                    Tensor& conv_states, const Tensor& initial_slot, Tensor& query,
                                    Tensor& key, Tensor& value, Tensor& z, cudaStream_t stream);

void nvfp4_gdn_snapshot_dispatch(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                                 Tensor& conv_states, const Tensor& initial_slot, Tensor& query,
                                 Tensor& key, Tensor& value, Tensor& z, LinearPolicy policy,
                                 WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
