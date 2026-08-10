#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"

#include "ninfer/ops/mtp_round.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/scalar.h"

#include <cuda_runtime.h>

#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {
void mtp_bridge_and_propose(State& state, const Tensor& next_token, const Tensor& previous_hidden,
                            std::int32_t position, std::span<const std::int32_t> rope_position,
                            bool build_proposal, const Tensor* next_embedding) {
    if (!state.mtp_kv.valid() || !state.io.mtp) {
        throw std::logic_error("MTP bridge requires MTP storage");
    }
    if (rope_position.size() != 3) {
        throw std::invalid_argument("MTP bridge requires one three-axis rope position");
    }
    state.work.reset();
    TextContext card(state.device, state.model, state.work, state.text_kv, state.linear_attention,
                     state.io, state.prefill_hidden, state.prefill_chunk, state.text_kv_base,
                     state.mtp_kv);
    configure_text_card(card, state);

    Tensor position_view = state.io.speculative.target_positions.slice(0, 0, 1);
    ops::set_i32_scalar(position_view, position, state.device.stream);
    Tensor mtp_hidden         = state.io.mtp->ar_hidden;
    Tensor logits             = state.io.logits.slice(1, 0, 1);
    Tensor draft0             = state.io.speculative.draft_tokens.slice(0, 0, 1);
    Tensor rope_position_view = state.work.alloc(DType::I32, {1, 3});
    CUDA_CHECK(cudaMemcpyAsync(rope_position_view.data, rope_position.data(),
                               rope_position.size_bytes(), cudaMemcpyHostToDevice,
                               state.device.stream));
    const auto bridge_visible = static_cast<std::uint32_t>(position + 1);
    const ops::GqaExecutionEnvelope bridge_envelope{bridge_visible, bridge_visible};
    card.mtp_forward_batch(next_token, previous_hidden, position_view, bridge_envelope, mtp_hidden,
                           build_proposal ? 0 : -1, build_proposal ? &logits : nullptr,
                           build_proposal ? &draft0 : nullptr, &rope_position_view, next_embedding);
    if (!build_proposal) { return; }

    if (state.mtp_proposal_extent == 0 ||
        state.mtp_proposal_extent >
            static_cast<std::uint32_t>(state.io.speculative.draft_tokens.ne[0])) {
        throw std::logic_error("MTP bridge proposal extent is outside the configured window");
    }

    Tensor ar_position = state.io.mtp->position.slice(0, 0, 1);
    ops::set_i32_scalar(ar_position, position + 1, state.device.stream);
    for (int i = 1; i < static_cast<int>(state.mtp_proposal_extent); ++i) {
        Tensor previous_token = state.io.speculative.draft_tokens.slice(0, i - 1, 1);
        Tensor next_draft     = state.io.speculative.draft_tokens.slice(0, i, 1);
        Tensor next_hidden    = state.prefill_hidden.slice(1, i, 1);
        const auto visible    = static_cast<std::uint32_t>(position + i + 1);
        const ops::GqaExecutionEnvelope envelope{visible, visible};
        card.mtp_forward_ar_step(previous_token, state.io.mtp->ar_hidden, ar_position, envelope,
                                 next_hidden, logits, next_draft);
        CUDA_CHECK(cudaMemcpyAsync(state.io.mtp->ar_hidden.data, next_hidden.data,
                                   state.io.mtp->ar_hidden.bytes(), cudaMemcpyDeviceToDevice,
                                   state.device.stream));
        ops::increment_i32_scalar(ar_position, state.device.stream);
    }
}

auto mtp_decode_batch_body(State& state, std::int32_t batch_size, std::uint32_t k,
                           MtpGqaEnvelopes envelopes) {
    return [&state, batch_size, k, envelopes] {
        if (batch_size <= 0 || batch_size > static_cast<std::int32_t>(kMaximumConcurrency) ||
            k == 0 || k > kMtpDecodeMaximumDrafts || !state.io.mtp_decode ||
            state.mtp_host_ingress == nullptr || state.mtp_host_egress == nullptr ||
            state.continuation_hidden_store == nullptr) {
            throw std::logic_error("MTP decode batch state is incomplete");
        }

        qwen3_6::MtpDecodeState& frame = *state.io.mtp_decode;
        const std::int32_t width       = static_cast<std::int32_t>(k) + 1;
        CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, state.mtp_host_ingress,
                                   sizeof(qwen3_6::MtpDecodeIngress), cudaMemcpyHostToDevice,
                                   state.device.stream));

        TextContext card(state.device, state.model, state.work, state.text_kv,
                         state.linear_attention, state.io, state.prefill_hidden,
                         state.prefill_chunk, state.text_kv_base, state.mtp_kv);
        configure_text_card(card, state);

        Tensor anchors           = frame.anchors.slice(0, 0, batch_size);
        Tensor frontiers         = frame.base_frontiers.slice(0, 0, batch_size);
        Tensor budgets           = frame.remaining_budgets.slice(0, 0, batch_size);
        Tensor current_extents   = frame.current_extents.slice(0, 0, batch_size);
        Tensor target_valid      = frame.target_valid_columns.slice(0, 0, batch_size);
        Tensor current_drafts    = frame.current_drafts.slice(1, 0, batch_size);
        Tensor target_rope       = frame.target_rope_positions.slice(1, 0, batch_size);
        Tensor text_rows         = frame.text_kv_table_rows.slice(0, 0, batch_size);
        Tensor mtp_rows          = frame.mtp_kv_table_rows.slice(0, 0, batch_size);
        Tensor read_slots        = frame.linear_state_read_slots.slice(0, 0, batch_size);
        Tensor snapshot_slots    = frame.linear_state_snapshot_base_slots.slice(0, 0, batch_size);
        Tensor continuation      = frame.continuation_slots.slice(0, 0, batch_size);
        Tensor rope_deltas       = frame.rope_deltas.slice(0, 0, batch_size);
        Tensor verify_ids        = frame.verify_ids.slice(1, 0, batch_size);
        Tensor target_positions  = frame.target_positions.slice(1, 0, batch_size);
        Tensor target_tokens     = frame.target_argmax.slice(1, 0, batch_size);
        Tensor target_logits     = frame.target_logits.slice(2, 0, batch_size);
        Tensor target_hidden     = frame.target_hidden.slice(2, 0, batch_size);
        Tensor selected_hidden   = frame.target_continuation_hidden.slice(1, 0, batch_size);
        Tensor licensed_tokens   = frame.licensed_tokens.slice(1, 0, batch_size);
        Tensor licensed_counts   = frame.licensed_counts.slice(0, 0, batch_size);
        Tensor accepted          = frame.accepted_drafts.slice(0, 0, batch_size);
        Tensor next_extents      = frame.next_extents.slice(0, 0, batch_size);
        Tensor alignment_ids     = frame.alignment_ids.slice(1, 0, batch_size);
        Tensor alignment_hidden  = frame.alignment_hidden.slice(2, 0, batch_size);
        Tensor ar_hidden         = frame.ar_hidden.slice(1, 0, batch_size);
        Tensor next_hidden       = frame.next_hidden.slice(1, 0, batch_size);
        Tensor ar_positions      = frame.ar_positions.slice(0, 0, batch_size);
        Tensor ar_rope_positions = frame.ar_rope_positions.slice(0, 0, batch_size);
        Tensor ar_valid_columns  = frame.ar_valid_columns.slice(0, 0, batch_size);
        Tensor next_drafts       = frame.next_drafts.slice(0, 0, batch_size);

        ops::speculative_prepare_verify_inputs(anchors, current_drafts, frontiers, current_extents,
                                               verify_ids, target_positions, state.device.stream);
        card.target_verify_batch(verify_ids, target_positions, target_rope, target_valid, text_rows,
                                 read_slots, snapshot_slots, envelopes.target_verify, target_hidden,
                                 target_logits, target_tokens);
        ops::speculative_accept_greedy_drafts(target_tokens, target_logits, current_drafts,
                                              current_extents, frontiers, anchors, licensed_tokens,
                                              licensed_counts, accepted, TextConfig::token_domain,
                                              frame.sampling, state.work, state.device.stream);
        ops::speculative_select_accepted_hidden(target_hidden, accepted, selected_hidden,
                                                state.device.stream);
        ops::scatter(selected_hidden, continuation, *state.continuation_hidden_store,
                     state.device.stream);

        ops::mtp_prepare_next_round(
            verify_ids, anchors, accepted, frontiers, budgets, licensed_counts, rope_deltas,
            alignment_ids, next_extents, ar_positions, ar_rope_positions, ar_valid_columns,
            static_cast<std::int32_t>(state.text_kv.max_context()), state.device.stream);
        card.mtp_forward_decode_batch(alignment_ids, target_hidden, target_positions, target_rope,
                                      licensed_counts, mtp_rows, envelopes.batch, alignment_hidden);
        ops::speculative_select_accepted_hidden(alignment_hidden, accepted, ar_hidden,
                                                state.device.stream);

        Tensor proposal_logits = state.io.ordinary.logits.slice(1, 0, batch_size);
        Tensor draft0          = next_drafts.slice(1, 0, 1).view({batch_size});
        card.mtp_propose_batch(ar_hidden, proposal_logits, draft0);
        for (std::uint32_t step = 0; step + 1 < k; ++step) {
            Tensor previous =
                next_drafts.slice(1, static_cast<std::int32_t>(step), 1).view({batch_size});
            Tensor next =
                next_drafts.slice(1, static_cast<std::int32_t>(step + 1), 1).view({batch_size});
            Tensor position =
                ar_positions.slice(1, static_cast<std::int32_t>(step), 1).view({1, batch_size});
            Tensor rope = ar_rope_positions.slice(1, static_cast<std::int32_t>(step), 1)
                              .view({1, batch_size});
            Tensor valid =
                ar_valid_columns.slice(1, static_cast<std::int32_t>(step), 1).view({batch_size});
            Tensor previous_batch    = previous.view({1, batch_size});
            Tensor hidden_batch      = ar_hidden.view({TextConfig::hidden, 1, batch_size});
            Tensor next_hidden_batch = next_hidden.view({TextConfig::hidden, 1, batch_size});
            card.mtp_forward_decode_batch(previous_batch, hidden_batch, position, rope, valid,
                                          mtp_rows, envelopes.ar[step], next_hidden_batch);
            card.mtp_propose_batch(next_hidden, proposal_logits, next);
            CUDA_CHECK(cudaMemcpyAsync(ar_hidden.data, next_hidden.data, ar_hidden.bytes(),
                                       cudaMemcpyDeviceToDevice, state.device.stream));
        }

        CUDA_CHECK(cudaMemcpyAsync(state.mtp_host_egress, frame.egress.data,
                                   sizeof(qwen3_6::MtpDecodeEgress), cudaMemcpyDeviceToHost,
                                   state.device.stream));
    };
}

void warm_capture_mtp_decode_batch(State& state, std::int32_t batch_size, std::uint32_t k,
                                   MtpGqaEnvelopes envelopes, const GraphPrepare& prepare,
                                   DecodeGraphDefinition& definition) {
    auto body = mtp_decode_batch_body(state, batch_size, k, envelopes);
    warm_capture(state, definition, prepare, body);
}

void mtp_decode_batch(State& state, std::int32_t batch_size, std::uint32_t k,
                      MtpGqaEnvelopes envelopes, DecodeGraphExecutable* executable) {
    auto body = mtp_decode_batch_body(state, batch_size, k, envelopes);
    run_prepared(state, executable, body);
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
