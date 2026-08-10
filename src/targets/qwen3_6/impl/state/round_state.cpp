#include <ninfer/targets/qwen3_6/round_state.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace ninfer::targets::qwen3_6 {
namespace {

constexpr std::size_t kArenaAlign = 256;

TensorRegion add_tensor(LayoutBuilder& builder, DType dtype,
                        std::initializer_list<std::int32_t> shape, const char* label) {
    return builder.add_tensor(dtype, shape, kArenaAlign, label);
}

std::int32_t checked_i32(std::uint64_t value, const char* label) {
    if (value > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument(label);
    }
    return static_cast<std::int32_t>(value);
}

void validate_spec(const RoundStateSpec& spec) {
    if (spec.hidden <= 0) { throw std::invalid_argument("RoundState hidden must be positive"); }
    if (spec.output_rows <= 0) {
        throw std::invalid_argument("RoundState output_rows must be positive");
    }
    if (spec.enable_mtp && spec.draft_window == 0) {
        throw std::invalid_argument("RoundState cannot enable MTP with an empty draft window");
    }
    if (spec.enable_mtp && spec.draft_window > kMtpDecodeMaximumDrafts) {
        throw std::invalid_argument("RoundState MTP draft window exceeds the decode frame domain");
    }
    if (spec.batch_capacity == 0 || spec.batch_capacity > kMaximumConcurrency) {
        throw std::invalid_argument("RoundState batch capacity must be in [1,8]");
    }
    (void)checked_i32(static_cast<std::uint64_t>(spec.draft_window) + 1ULL,
                      "RoundState draft window exceeds int32");
}

} // namespace

RoundStateLayout begin_round_state_layout(LayoutBuilder& builder, const RoundStateSpec& spec) {
    validate_spec(spec);
    const std::int32_t columns = checked_i32(static_cast<std::uint64_t>(spec.draft_window) + 1ULL,
                                             "RoundState columns exceed int32");
    RoundStateLayout layout;
    layout.spec = spec;
    layout.ordinary.ingress =
        builder.add(sizeof(OrdinaryDecodeIngress), 256, "ordinary decode ingress");
    layout.ordinary.egress =
        builder.add(sizeof(OrdinaryDecodeEgress), 256, "ordinary decode egress");
    layout.ordinary.logits = add_tensor(
        builder, DType::BF16,
        {spec.output_rows, checked_i32(spec.batch_capacity, "RoundState batch capacity")},
        "ordinary decode logits");
    layout.ordinary.hidden =
        add_tensor(builder, DType::BF16,
                   {spec.hidden, checked_i32(spec.batch_capacity, "RoundState batch capacity")},
                   "ordinary decode hidden");
    layout.token      = add_tensor(builder, DType::I32, {1}, "step token");
    layout.pos        = add_tensor(builder, DType::I32, {1}, "step position");
    layout.rope_pos   = add_tensor(builder, DType::I32, {1}, "step rope position");
    layout.rope_delta = add_tensor(builder, DType::I32, {1}, "step rope delta");
    layout.logits = add_tensor(builder, DType::BF16, {spec.output_rows, columns}, "step logits");
    layout.verify_hidden =
        add_tensor(builder, DType::BF16, {spec.hidden, columns}, "step verify hidden");
    layout.text_kv_table_row    = add_tensor(builder, DType::I32, {1}, "step Text KV table row");
    layout.backend_kv_table_row = add_tensor(builder, DType::I32, {1}, "step backend KV table row");
    return layout;
}

OrdinaryDecodeState::OrdinaryDecodeState(DeviceSpan backing,
                                         const OrdinaryDecodeStateLayout& layout,
                                         std::uint32_t batch_capacity) {
    if (batch_capacity == 0 || batch_capacity > kMaximumConcurrency) {
        throw std::invalid_argument("ordinary decode batch capacity must be in [1,8]");
    }
    static_assert(std::is_standard_layout_v<OrdinaryDecodeIngress>);
    static_assert(std::is_standard_layout_v<OrdinaryDecodeEgress>);

    ingress                   = layout.ingress.bind(backing);
    egress                    = layout.egress.bind(backing);
    const auto count          = static_cast<std::int32_t>(batch_capacity);
    const auto ingress_tensor = [&](std::size_t offset, DType dtype) {
        return Tensor(static_cast<unsigned char*>(ingress.data) + offset, dtype, {count});
    };
    tokens          = ingress_tensor(offsetof(OrdinaryDecodeIngress, tokens), DType::I32);
    cache_positions = ingress_tensor(offsetof(OrdinaryDecodeIngress, cache_positions), DType::I32);
    rope_positions  = ingress_tensor(offsetof(OrdinaryDecodeIngress, rope_positions), DType::I32);
    text_kv_table_rows =
        ingress_tensor(offsetof(OrdinaryDecodeIngress, text_kv_table_rows), DType::I32);
    linear_state_read_slots =
        ingress_tensor(offsetof(OrdinaryDecodeIngress, linear_state_read_slots), DType::I32);
    linear_state_snapshot_base_slots = ingress_tensor(
        offsetof(OrdinaryDecodeIngress, linear_state_snapshot_base_slots), DType::I32);
    continuation_slots =
        ingress_tensor(offsetof(OrdinaryDecodeIngress, continuation_slots), DType::I32);
    sampling = reinterpret_cast<const ops::SamplingConfig*>(
        static_cast<const unsigned char*>(ingress.data) +
        offsetof(OrdinaryDecodeIngress, sampling));
    sampled_tokens = Tensor(static_cast<unsigned char*>(egress.data) +
                                offsetof(OrdinaryDecodeEgress, sampled_tokens),
                            DType::I32, {count});
    logits         = layout.logits.bind(backing);
    hidden         = layout.hidden.bind(backing);
}

void complete_round_state_layout(LayoutBuilder& builder, RoundStateLayout& layout) {
    if (layout.complete) { throw std::logic_error("RoundState layout is already complete"); }
    validate_spec(layout.spec);
    const std::int32_t columns =
        checked_i32(static_cast<std::uint64_t>(layout.spec.draft_window) + 1ULL,
                    "RoundState columns exceed int32");
    const std::int32_t drafts = checked_i32(std::max<std::uint64_t>(1ULL, layout.spec.draft_window),
                                            "RoundState drafts exceed int32");
    const auto i32            = [&](std::int32_t count, const char* label) {
        return add_tensor(builder, DType::I32, {count}, label);
    };
    layout.linear_state_read_slot              = i32(1, "step Linear Attention read slot");
    layout.linear_state_snapshot_base_slot     = i32(1, "step Linear Attention snapshot base slot");
    layout.speculative.target_argmax           = i32(columns, "step target argmax");
    layout.speculative.draft_tokens            = i32(drafts, "step draft tokens");
    layout.speculative.current_proposal_extent = i32(1, "step current proposal extent");
    layout.speculative.round_tokens            = i32(columns, "step speculative round tokens");
    layout.speculative.produced_count          = i32(1, "step speculative produced count");
    layout.speculative.target_input_ids        = i32(columns, "step target input ids");
    layout.speculative.target_positions        = i32(columns, "step target positions");
    layout.speculative.accepted_drafts         = i32(1, "step accepted drafts");
    if (layout.spec.enable_mtp) {
        layout.mtp.emplace();
        const auto ar_steps =
            checked_i32(std::max<std::uint64_t>(1ULL, layout.spec.draft_window - 1ULL),
                        "RoundState MTP AR steps exceed int32");
        layout.mtp->position  = i32(1, "MTP prefill autoregressive position");
        layout.mtp->ar_hidden = add_tensor(builder, DType::BF16, {layout.spec.hidden, 1},
                                           "MTP prefill autoregressive hidden");

        layout.mtp_decode.emplace();
        MtpDecodeStateLayout& decode = *layout.mtp_decode;
        decode.ingress = builder.add(sizeof(MtpDecodeIngress), kArenaAlign, "MTP decode ingress");
        decode.egress  = builder.add(sizeof(MtpDecodeEgress), kArenaAlign, "MTP decode egress");
        const auto batch =
            checked_i32(layout.spec.batch_capacity, "RoundState MTP batch capacity exceeds int32");
        decode.verify_ids =
            add_tensor(builder, DType::I32, {columns, batch}, "MTP decode verify ids");
        decode.target_positions =
            add_tensor(builder, DType::I32, {columns, batch}, "MTP decode target positions");
        decode.target_argmax =
            add_tensor(builder, DType::I32, {columns, batch}, "MTP decode target argmax");
        decode.target_logits =
            add_tensor(builder, DType::BF16, {layout.spec.output_rows, columns, batch},
                       "MTP decode target logits");
        decode.target_hidden = add_tensor(
            builder, DType::BF16, {layout.spec.hidden, columns, batch}, "MTP decode target hidden");
        decode.target_continuation_hidden =
            add_tensor(builder, DType::BF16, {layout.spec.hidden, batch},
                       "MTP decode target continuation hidden");
        decode.alignment_ids =
            add_tensor(builder, DType::I32, {columns, batch}, "MTP decode alignment ids");
        decode.alignment_hidden =
            add_tensor(builder, DType::BF16, {layout.spec.hidden, columns, batch},
                       "MTP decode alignment hidden");
        decode.ar_hidden = add_tensor(builder, DType::BF16, {layout.spec.hidden, batch},
                                      "MTP decode autoregressive hidden");
        decode.next_hidden =
            add_tensor(builder, DType::BF16, {layout.spec.hidden, batch}, "MTP decode next hidden");
        decode.ar_positions      = add_tensor(builder, DType::I32, {batch, ar_steps},
                                              "MTP decode autoregressive positions");
        decode.ar_rope_positions = add_tensor(builder, DType::I32, {batch, ar_steps},
                                              "MTP decode autoregressive rope positions");
        decode.ar_valid_columns  = add_tensor(builder, DType::I32, {batch, ar_steps},
                                              "MTP decode autoregressive valid columns");
    }
    layout.complete = true;
}

SpeculativeRoundState::SpeculativeRoundState(DeviceSpan backing,
                                             const SpeculativeRoundStateLayout& layout)
    : target_argmax(layout.target_argmax.bind(backing)),
      draft_tokens(layout.draft_tokens.bind(backing)),
      current_proposal_extent(layout.current_proposal_extent.bind(backing)),
      round_tokens(layout.round_tokens.bind(backing)),
      produced_count(layout.produced_count.bind(backing)),
      target_input_ids(layout.target_input_ids.bind(backing)),
      target_positions(layout.target_positions.bind(backing)),
      accepted_drafts(layout.accepted_drafts.bind(backing)) {}

MtpPrefillState::MtpPrefillState(DeviceSpan backing, const MtpPrefillStateLayout& layout)
    : position(layout.position.bind(backing)), ar_hidden(layout.ar_hidden.bind(backing)) {}

MtpDecodeState::MtpDecodeState(DeviceSpan backing, const MtpDecodeStateLayout& layout,
                               std::uint32_t batch_capacity, std::uint32_t draft_window) {
    if (batch_capacity == 0 || batch_capacity > kMaximumConcurrency || draft_window == 0 ||
        draft_window > kMtpDecodeMaximumDrafts) {
        throw std::invalid_argument("MTP decode state dimensions are outside the supported domain");
    }
    static_assert(std::is_standard_layout_v<MtpDecodeIngress>);
    static_assert(std::is_standard_layout_v<MtpDecodeEgress>);
    const auto batch          = static_cast<std::int32_t>(batch_capacity);
    const auto drafts         = static_cast<std::int32_t>(draft_window);
    const auto width          = drafts + 1;
    const auto steps          = std::max(drafts - 1, 1);
    ingress                   = layout.ingress.bind(backing);
    egress                    = layout.egress.bind(backing);
    const auto ingress_tensor = [&](std::size_t offset, DType dtype,
                                    std::initializer_list<std::int32_t> shape) {
        return Tensor(static_cast<unsigned char*>(ingress.data) + offset, dtype, shape);
    };
    const auto egress_tensor = [&](std::size_t offset, DType dtype,
                                   std::initializer_list<std::int32_t> shape) {
        return Tensor(static_cast<unsigned char*>(egress.data) + offset, dtype, shape);
    };
    anchors = ingress_tensor(offsetof(MtpDecodeIngress, anchors), DType::I32, {batch});
    base_frontiers =
        ingress_tensor(offsetof(MtpDecodeIngress, base_frontiers), DType::I32, {batch});
    remaining_budgets =
        ingress_tensor(offsetof(MtpDecodeIngress, remaining_budgets), DType::I32, {batch});
    current_extents =
        ingress_tensor(offsetof(MtpDecodeIngress, current_extents), DType::I32, {batch});
    target_valid_columns =
        ingress_tensor(offsetof(MtpDecodeIngress, target_valid_columns), DType::I32, {batch});
    current_drafts =
        ingress_tensor(offsetof(MtpDecodeIngress, current_drafts), DType::I32, {drafts, batch});
    target_rope_positions = ingress_tensor(offsetof(MtpDecodeIngress, target_rope_positions),
                                           DType::I32, {width, batch});
    text_kv_table_rows =
        ingress_tensor(offsetof(MtpDecodeIngress, text_kv_table_rows), DType::I32, {batch});
    mtp_kv_table_rows =
        ingress_tensor(offsetof(MtpDecodeIngress, mtp_kv_table_rows), DType::I32, {batch});
    linear_state_read_slots =
        ingress_tensor(offsetof(MtpDecodeIngress, linear_state_read_slots), DType::I32, {batch});
    linear_state_snapshot_base_slots = ingress_tensor(
        offsetof(MtpDecodeIngress, linear_state_snapshot_base_slots), DType::I32, {batch});
    continuation_slots =
        ingress_tensor(offsetof(MtpDecodeIngress, continuation_slots), DType::I32, {batch});
    rope_deltas = ingress_tensor(offsetof(MtpDecodeIngress, rope_deltas), DType::I32, {batch});
    sampling    = reinterpret_cast<const ops::SamplingConfig*>(
        static_cast<const unsigned char*>(ingress.data) + offsetof(MtpDecodeIngress, sampling));

    licensed_tokens =
        egress_tensor(offsetof(MtpDecodeEgress, licensed_tokens), DType::I32, {width, batch});
    licensed_counts =
        egress_tensor(offsetof(MtpDecodeEgress, licensed_counts), DType::I32, {batch});
    accepted_drafts =
        egress_tensor(offsetof(MtpDecodeEgress, accepted_drafts), DType::I32, {batch});
    next_drafts =
        egress_tensor(offsetof(MtpDecodeEgress, next_drafts), DType::I32, {batch, drafts});
    next_extents     = egress_tensor(offsetof(MtpDecodeEgress, next_extents), DType::I32, {batch});
    verify_ids       = layout.verify_ids.bind(backing);
    target_positions = layout.target_positions.bind(backing);
    target_argmax    = layout.target_argmax.bind(backing);
    target_logits    = layout.target_logits.bind(backing);
    target_hidden    = layout.target_hidden.bind(backing);
    target_continuation_hidden = layout.target_continuation_hidden.bind(backing);
    alignment_ids              = layout.alignment_ids.bind(backing);
    alignment_hidden           = layout.alignment_hidden.bind(backing);
    ar_hidden                  = layout.ar_hidden.bind(backing);
    next_hidden                = layout.next_hidden.bind(backing);
    ar_positions               = layout.ar_positions.bind(backing);
    ar_rope_positions          = layout.ar_rope_positions.bind(backing);
    ar_valid_columns           = layout.ar_valid_columns.bind(backing);
    if (ar_positions.ne[0] != batch || ar_positions.ne[1] != steps) {
        throw std::logic_error("MTP decode AR layout does not match its configured dimensions");
    }
}

RoundState::RoundState(DeviceSpan backing, const RoundStateLayout& layout) {
    if (!layout.complete) { throw std::invalid_argument("RoundState layout is incomplete"); }
    ordinary          = OrdinaryDecodeState(backing, layout.ordinary, layout.spec.batch_capacity);
    token             = layout.token.bind(backing);
    pos               = layout.pos.bind(backing);
    rope_pos          = layout.rope_pos.bind(backing);
    rope_delta        = layout.rope_delta.bind(backing);
    logits            = layout.logits.bind(backing);
    verify_hidden     = layout.verify_hidden.bind(backing);
    text_kv_table_row = layout.text_kv_table_row.bind(backing);
    backend_kv_table_row            = layout.backend_kv_table_row.bind(backing);
    linear_state_read_slot          = layout.linear_state_read_slot.bind(backing);
    linear_state_snapshot_base_slot = layout.linear_state_snapshot_base_slot.bind(backing);
    speculative                     = SpeculativeRoundState(backing, layout.speculative);
    if (layout.mtp) { mtp.emplace(backing, *layout.mtp); }
    if (layout.mtp_decode) {
        mtp_decode.emplace(backing, *layout.mtp_decode, layout.spec.batch_capacity,
                           layout.spec.draft_window);
    }
}

} // namespace ninfer::targets::qwen3_6
