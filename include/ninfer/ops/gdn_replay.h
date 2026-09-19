#pragma once

#include "core/gdn_replay_records.h"
#include "core/linear_attention_state.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <span>

namespace ninfer::ops {

struct GdnReplayFoldRow {
    std::int32_t linear_state_slot;
    std::int32_t commit_columns;
    std::array<std::int32_t, 16> path{};
    std::int32_t path_length = -1;
};

/**
 * Op: gdn_replay_fold
 *
 * Replays each row's accepted records across every registered GDN layer and updates the
 * caller-selected absolute linear-attention state slot in place. rows[b] always maps to physical
 * record row b; rows are not filtered, compressed, or reordered. The active row count is
 * rows.size() and must be in [1,records.spec.record_capacity].
 *
 * linear_state_slot is in [0,states.spec.slot_count), is distinct across active rows, and is the
 * same absolute slot used to produce that row's records. commit_columns is in [0,T] and is the
 * linear/MTP packed prefix. Zero is a strict no-op for the row: no record or state is read and
 * neither recurrent state nor convolution history is written. path_length < 0 keeps that prefix
 * behavior: the Op consumes raw key/value/{g,beta} records in packed order [0,commit_columns) and
 * sets convolution history to tail_3(old_history || conv_record[0:commit_columns]).
 *
 * path_length == 0 is also a strict no-op. path_length > 0 replays records at
 * path[0], path[1], ..., path[path_length-1] in that time order. Each path[i] is in [0,T). The
 * convolution history is tail_3(old_history || conv_record[path[0]], ...,
 * conv_record[path[path_length-1]]).
 *
 * Supported (layers,QK heads,value heads,conv channels) are (48,16,48,10240),
 * (30,16,32,8192), and (36,48,48,10240). Key and value dimensions are 128. Each accepted
 * transition is the normalized gated_delta_net recurrence with raw BF16 key/value, FP32
 * {g,beta}, and FP32 recurrent state; QK head = floor(value_head / (value_heads/QK_heads)).
 * Thus the 48/48 geometry uses already-expanded keys with identity head assignment.
 * The Op owns no workspace or metadata allocation, does not read query or generate token
 * output, and supports stream capture. The four record planes are read-only, disjoint, and
 * do not overlap either state region.
 * Row controls are host values copied into kernel arguments: a captured graph fixes slots,
 * commit counts, and paths. Changing acceptance requires an eager call or a fresh capture.
 */
void gdn_replay_fold(const GdnReplayRecords& records, LinearAttentionStateAllLayersView states,
                     std::span<const GdnReplayFoldRow> rows, cudaStream_t stream);

/**
 * Single-layer form of the same transition, exclusively for expanded 48/48 heads and
 * 10240 convolution channels. conv_states is contiguous BF16 [10240,3,S] and
 * recurrent_states is contiguous FP32 [128,128,48,S], S>0. Records are contiguous
 * conv BF16 [10240,T,B], key/value BF16 [128,48,T,B], and gate FP32 [2,48,T,B],
 * with T=2..16, B=1..4; each plane and state base is 256-byte aligned. rows.size()
 * is in [1,B]. This overload consumes only these actual layer tensors, with no
 * allocation, projection recomputation, or reads/writes of other model layers.
 * Prefix/path, zero-acceptance, disjointness, and capture rules are as above.
 */
void gdn_replay_fold(const GdnReplayRecordLayer& records, Tensor& conv_states,
                     Tensor& recurrent_states, std::span<const GdnReplayFoldRow> rows,
                     cudaStream_t stream);

} // namespace ninfer::ops
