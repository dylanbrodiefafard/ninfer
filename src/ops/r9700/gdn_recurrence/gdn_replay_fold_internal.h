#pragma once

#include "core/gdn_replay_records.h"
#include "core/linear_attention_state.h"
#include "core/tensor.h"
#include "ninfer/ops/gdn_replay.h"
#include "ninfer/types.h"
#include "ops/r9700/gdn_recurrence/gdn_replay_fold_impl.h"

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail::gated_delta_net {

struct alignas(8) GdnReplayFoldKernelRow {
    std::int32_t linear_state_slot;
    std::int32_t commit_columns;
    std::int32_t path_length;
    std::int32_t path[16];
};

struct alignas(16) GdnReplayFoldKernelRows {
    GdnReplayFoldKernelRow row[kMaximumConcurrency];
};

// Validated device arguments of one layer's deferred fold (gdn_replay_fold_layer), shared by the
// standalone kernel and fused consumers (the FP8 GDN front).
[[nodiscard]] fold::LayerArgs replay_fold_layer_args(const GdnLayerFold& layer_fold);
void launch_replay_fold_layer(const fold::LayerArgs& args, hipStream_t stream);
void launch_replay_prepare_layer(const fold::LayerArgs& args, const fold::Planes& staged,
                                 hipStream_t stream);

void launch_replay_fold(const GdnReplayRecords& records, LinearAttentionStateAllLayersView states,
                        const GdnReplayFoldKernelRows& rows, std::int32_t active_rows,
                        hipStream_t stream);

} // namespace ninfer::ops::detail::gated_delta_net
