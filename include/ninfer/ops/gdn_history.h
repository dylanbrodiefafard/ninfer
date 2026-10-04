#pragma once

#include "core/gdn_history.h"
#include "ninfer/ops/gdn_replay.h"

namespace ninfer::ops {

// Closed accepted-history profile: represented BF16 Q/K/V and FP32 controls/checkpoint obey the
// complete GDN recurrence. Raw records remain observable. FP32 normalized keys and innovations
// are derived from that invocation's actual parent state, including checkpoint+retained history.
// A selected commit must consume the same provisional records and an ancestor-complete path.
// The 48-layer/16-QK-head/48-value-head/D128 profile uses capacity four, stable startup storage,
// one stream and unique current slots. Arbitrary edited raw records cannot use this profile.
void gated_delta_net_history_record(const Tensor& q, const Tensor& k, const Tensor& v,
                                    const Tensor& g, const Tensor& beta, float scale,
                                    const Tensor& states, const Tensor& valid_columns,
                                    const Tensor& initial_slots, Tensor& key_record,
                                    Tensor& value_record, Tensor& gate_record, Tensor& out,
                                    const GdnHistoryLayer& history, cudaStream_t stream,
                                    const Tensor* tree_schedule = nullptr);

// Rows map compact provisional/raw record rows to distinct absolute current slots. commit_columns
// includes the anchor even when zero drafts were accepted; only an abort/retry/cancel commits zero.
// Append retained FP32 innovations, or materialize the complete checkpoint when count reaches four.
// Conv3 commits every accepted round. Counts publish only after all layers finish their transition.
void gdn_history_commit(const GdnReplayRecords& records, const GdnHistory& history,
                        LinearAttentionStateAllLayersView states,
                        std::span<const GdnReplayFoldRow> rows, cudaStream_t stream);

// Materialize the selected current slots before a dense consumer/snapshot; no raw/provisional
// record is read and no convolution history changes. Every selected length becomes zero.
void gdn_history_materialize(const GdnReplayRecords& records, const GdnHistory& history,
                             LinearAttentionStateAllLayersView states,
                             std::span<const std::int32_t> slots, cudaStream_t stream);

// Discard retained history only after the caller has installed/reset the dense checkpoint, or
// while discarding the logical slot before its next ordered restore/acquisition.
void gdn_history_reset_slot(const GdnHistory& history, std::int32_t slot, cudaStream_t stream);
void gdn_history_reset_all(const GdnHistory& history, cudaStream_t stream);

} // namespace ninfer::ops
