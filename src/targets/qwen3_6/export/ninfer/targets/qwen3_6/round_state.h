#pragma once

#include "core/layout.h"
#include "core/tensor.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace ninfer::targets::qwen3_6 {

inline constexpr std::uint32_t kMtpDecodeMaximumDrafts = 5;
inline constexpr std::uint32_t kMtpDecodeMaximumWidth  = kMtpDecodeMaximumDrafts + 1;

struct RoundStateSpec {
    std::int32_t hidden          = 0;
    std::int32_t output_rows     = 0;
    std::uint32_t batch_capacity = 1;
    std::uint32_t draft_window   = 0;
    bool enable_mtp              = false;
};

// Stable pinned/device transfer format for ordinary decode. The full fixed-size object is copied
// once per round; only its exact-B prefixes are consumed by the model schedule.
struct OrdinaryDecodeIngress {
    std::array<TokenId, kMaximumConcurrency> tokens{};
    std::array<std::int32_t, kMaximumConcurrency> cache_positions{};
    std::array<std::int32_t, kMaximumConcurrency> rope_positions{};
    std::array<std::int32_t, kMaximumConcurrency> text_kv_table_rows{};
    std::array<std::int32_t, kMaximumConcurrency> linear_state_read_slots{};
    std::array<std::int32_t, kMaximumConcurrency> linear_state_snapshot_base_slots{};
    std::array<std::int32_t, kMaximumConcurrency> continuation_slots{};
    std::array<ops::SamplingConfig, kMaximumConcurrency> sampling{};
};

struct OrdinaryDecodeEgress {
    std::array<TokenId, kMaximumConcurrency> sampled_tokens{};
};

// Stable pinned/device transfer formats for concurrent MTP decode. The arrays use the maximum
// product domain; RoundState binds only the configured [K,C] and [K+1,C] prefixes.
struct MtpDecodeIngress {
    std::array<TokenId, kMaximumConcurrency> anchors{};
    std::array<std::int32_t, kMaximumConcurrency> base_frontiers{};
    std::array<std::int32_t, kMaximumConcurrency> remaining_budgets{};
    std::array<std::int32_t, kMaximumConcurrency> current_extents{};
    std::array<std::int32_t, kMaximumConcurrency> target_valid_columns{};
    std::array<TokenId, kMaximumConcurrency * kMtpDecodeMaximumDrafts> current_drafts{};
    std::array<std::int32_t, kMaximumConcurrency * kMtpDecodeMaximumWidth> target_rope_positions{};
    std::array<std::int32_t, kMaximumConcurrency> text_kv_table_rows{};
    std::array<std::int32_t, kMaximumConcurrency> mtp_kv_table_rows{};
    std::array<std::int32_t, kMaximumConcurrency> linear_state_read_slots{};
    std::array<std::int32_t, kMaximumConcurrency> linear_state_snapshot_base_slots{};
    std::array<std::int32_t, kMaximumConcurrency> continuation_slots{};
    std::array<std::int32_t, kMaximumConcurrency> rope_deltas{};
    std::array<ops::SamplingConfig, kMaximumConcurrency> sampling{};
};

struct MtpDecodeEgress {
    std::array<TokenId, kMaximumConcurrency * kMtpDecodeMaximumWidth> licensed_tokens{};
    std::array<std::int32_t, kMaximumConcurrency> licensed_counts{};
    std::array<std::int32_t, kMaximumConcurrency> accepted_drafts{};
    // Step-major: all B rows for proposal step 0, followed by all B rows for step 1, etc.
    std::array<TokenId, kMaximumConcurrency * kMtpDecodeMaximumDrafts> next_drafts{};
    std::array<std::int32_t, kMaximumConcurrency> next_extents{};
};

struct OrdinaryDecodeStateLayout {
    LayoutRegion ingress;
    LayoutRegion egress;
    TensorRegion logits;
    TensorRegion hidden;
};

struct SpeculativeRoundStateLayout {
    TensorRegion target_argmax;
    TensorRegion draft_tokens;
    TensorRegion current_proposal_extent;
    TensorRegion round_tokens;
    TensorRegion produced_count;
    TensorRegion target_input_ids;
    TensorRegion target_positions;
    TensorRegion accepted_drafts;
};

struct MtpPrefillStateLayout {
    TensorRegion position;
    TensorRegion ar_hidden;
};

struct MtpDecodeStateLayout {
    LayoutRegion ingress;
    LayoutRegion egress;
    TensorRegion verify_ids;
    TensorRegion target_positions;
    TensorRegion target_argmax;
    TensorRegion target_logits;
    TensorRegion target_hidden;
    TensorRegion target_continuation_hidden;
    TensorRegion alignment_ids;
    TensorRegion alignment_hidden;
    TensorRegion ar_hidden;
    TensorRegion next_hidden;
    TensorRegion ar_positions;
    TensorRegion ar_rope_positions;
    TensorRegion ar_valid_columns;
};

struct RoundStateLayout {
    RoundStateSpec spec;
    OrdinaryDecodeStateLayout ordinary;
    TensorRegion token;
    TensorRegion pos;
    TensorRegion rope_pos;
    TensorRegion rope_delta;
    TensorRegion logits;
    TensorRegion verify_hidden;
    TensorRegion text_kv_table_row;
    TensorRegion backend_kv_table_row;
    TensorRegion linear_state_read_slot;
    TensorRegion linear_state_snapshot_base_slot;
    SpeculativeRoundStateLayout speculative;
    std::optional<MtpPrefillStateLayout> mtp;
    std::optional<MtpDecodeStateLayout> mtp_decode;
    bool complete = false;
};

struct OrdinaryDecodeState {
    DeviceSpan ingress;
    DeviceSpan egress;
    Tensor tokens;
    Tensor cache_positions;
    Tensor rope_positions;
    Tensor text_kv_table_rows;
    Tensor linear_state_read_slots;
    Tensor linear_state_snapshot_base_slots;
    Tensor continuation_slots;
    const ops::SamplingConfig* sampling = nullptr;
    Tensor sampled_tokens;
    Tensor logits;
    Tensor hidden;

    OrdinaryDecodeState() = default;
    OrdinaryDecodeState(DeviceSpan backing, const OrdinaryDecodeStateLayout& layout,
                        std::uint32_t batch_capacity);
};

// The two planning calls expose one deliberate exact-target extension seam after verify_hidden.
// This lets a target retain its schedule-sized prefill activation at the established physical
// address without making that activation part of the family round contract.
[[nodiscard]] RoundStateLayout begin_round_state_layout(LayoutBuilder& builder,
                                                        const RoundStateSpec& spec);
void complete_round_state_layout(LayoutBuilder& builder, RoundStateLayout& layout);

struct SpeculativeRoundState {
    Tensor target_argmax;
    Tensor draft_tokens;
    Tensor current_proposal_extent;
    Tensor round_tokens;
    Tensor produced_count;
    Tensor target_input_ids;
    Tensor target_positions;
    Tensor accepted_drafts;

    SpeculativeRoundState() = default;
    SpeculativeRoundState(DeviceSpan backing, const SpeculativeRoundStateLayout& layout);
};

struct MtpPrefillState {
    Tensor position;
    Tensor ar_hidden;

    MtpPrefillState() = default;
    MtpPrefillState(DeviceSpan backing, const MtpPrefillStateLayout& layout);
};

struct MtpDecodeState {
    DeviceSpan ingress;
    DeviceSpan egress;
    Tensor anchors;
    Tensor base_frontiers;
    Tensor remaining_budgets;
    Tensor current_extents;
    Tensor target_valid_columns;
    Tensor current_drafts;
    Tensor target_rope_positions;
    Tensor text_kv_table_rows;
    Tensor mtp_kv_table_rows;
    Tensor linear_state_read_slots;
    Tensor linear_state_snapshot_base_slots;
    Tensor continuation_slots;
    Tensor rope_deltas;
    const ops::SamplingConfig* sampling = nullptr;
    Tensor licensed_tokens;
    Tensor licensed_counts;
    Tensor accepted_drafts;
    Tensor next_drafts;
    Tensor next_extents;
    Tensor verify_ids;
    Tensor target_positions;
    Tensor target_argmax;
    Tensor target_logits;
    Tensor target_hidden;
    Tensor target_continuation_hidden;
    Tensor alignment_ids;
    Tensor alignment_hidden;
    Tensor ar_hidden;
    Tensor next_hidden;
    Tensor ar_positions;
    Tensor ar_rope_positions;
    Tensor ar_valid_columns;

    MtpDecodeState() = default;
    MtpDecodeState(DeviceSpan backing, const MtpDecodeStateLayout& layout,
                   std::uint32_t batch_capacity, std::uint32_t draft_window);
};

struct RoundState {
    OrdinaryDecodeState ordinary;
    Tensor token;
    Tensor pos;
    Tensor rope_pos;
    Tensor rope_delta;
    Tensor logits;
    Tensor verify_hidden;
    Tensor text_kv_table_row;
    Tensor backend_kv_table_row;
    Tensor linear_state_read_slot;
    Tensor linear_state_snapshot_base_slot;
    SpeculativeRoundState speculative;
    std::optional<MtpPrefillState> mtp;
    std::optional<MtpDecodeState> mtp_decode;

    RoundState() = default;
    RoundState(DeviceSpan backing, const RoundStateLayout& layout);
};

} // namespace ninfer::targets::qwen3_6
