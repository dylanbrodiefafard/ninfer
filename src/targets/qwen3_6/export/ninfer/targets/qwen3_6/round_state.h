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

struct OrdinaryDecodeStateLayout {
    LayoutRegion ingress;
    LayoutRegion egress;
    TensorRegion logits;
    TensorRegion hidden;
};

struct SpeculativeRoundStateLayout {
    TensorRegion target_argmax;
    TensorRegion draft_tokens;
    TensorRegion round_tokens;
    TensorRegion produced_count;
    TensorRegion target_input_ids;
    TensorRegion target_positions;
    TensorRegion accepted_drafts;
    TensorRegion stats;
};

struct MtpRoundStateLayout {
    TensorRegion alignment_ids;
    TensorRegion position;
    TensorRegion ar_hidden;
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
    std::optional<MtpRoundStateLayout> mtp;
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
    Tensor round_tokens;
    Tensor produced_count;
    Tensor target_input_ids;
    Tensor target_positions;
    Tensor accepted_drafts;
    Tensor stats;

    SpeculativeRoundState() = default;
    SpeculativeRoundState(DeviceSpan backing, const SpeculativeRoundStateLayout& layout);
};

struct MtpRoundState {
    Tensor alignment_ids;
    Tensor position;
    Tensor ar_hidden;

    MtpRoundState() = default;
    MtpRoundState(DeviceSpan backing, const MtpRoundStateLayout& layout);
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
    std::optional<MtpRoundState> mtp;

    RoundState() = default;
    RoundState(DeviceSpan backing, const RoundStateLayout& layout);
};

} // namespace ninfer::targets::qwen3_6
