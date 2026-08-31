#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/panel_copy.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"

#include "ninfer/ops/mtp_round.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/scalar.h"

#include <cuda_runtime.h>

#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {
void mtp_bridge_and_propose(PrefillContext& state, const Tensor& next_token,
                            const Tensor& previous_hidden, std::int32_t position,
                            std::span<const std::int32_t> rope_position, bool build_proposal,
                            const Tensor* next_embedding) {
    if (!state.mtp_kv.valid() || !state.execution.io.mtp) {
        throw std::logic_error("MTP bridge requires MTP storage");
    }
    if (rope_position.size() != 3) {
        throw std::invalid_argument("MTP bridge requires one three-axis rope position");
    }
    state.execution.work.reset();
    TextContext card(state.execution.device, state.execution.model, state.execution.work,
                     state.text_kv, state.execution.linear_attention, state.execution.io,
                     state.execution.prefill_hidden, state.execution.prefill_chunk,
                     state.text_kv_base, state.mtp_kv, &state.text_cache, state.mtp_cache);
    configure_text_card(card, state.execution, state.sampling, state.current_state_slot,
                        state.rewrite_checkpoint_state_slot, state.mtp_proposal_extent);

    Tensor position_view = state.execution.io.mtp->target_positions.slice(0, 0, 1);
    ops::set_i32_scalar(position_view, position, state.execution.device.stream);
    Tensor mtp_hidden         = state.execution.io.mtp->ar_hidden;
    Tensor logits             = state.execution.io.logits.slice(1, 0, 1);
    Tensor draft0             = state.execution.io.mtp->draft_tokens.slice(0, 0, 1);
    Tensor rope_position_view = state.execution.work.alloc(DType::I32, {1, 3});
    CUDA_CHECK(cudaMemcpyAsync(rope_position_view.data, rope_position.data(),
                               rope_position.size_bytes(), cudaMemcpyHostToDevice,
                               state.execution.device.stream));
    const auto bridge_visible = static_cast<std::uint32_t>(position + 1);
    const ops::GqaExecutionEnvelope bridge_envelope{bridge_visible, bridge_visible};
    card.mtp_forward_batch(next_token, previous_hidden, position_view, bridge_envelope, mtp_hidden,
                           build_proposal ? 0 : -1, build_proposal ? &logits : nullptr,
                           build_proposal ? &draft0 : nullptr, &rope_position_view, next_embedding);
    if (!build_proposal) { return; }

    if (state.mtp_proposal_extent == 0 ||
        state.mtp_proposal_extent >
            static_cast<std::uint32_t>(state.execution.io.mtp->draft_tokens.ne[0])) {
        throw std::logic_error("MTP bridge proposal extent is outside the configured window");
    }

    Tensor ar_position = state.execution.io.mtp->position.slice(0, 0, 1);
    ops::set_i32_scalar(ar_position, position + 1, state.execution.device.stream);
    // Ping-pong ar_hidden <-> scratch so MTP AR steps avoid a full-hidden D2D each round.
    Tensor ar_a = state.execution.io.mtp->ar_hidden;
    Tensor ar_b = state.execution.prefill_hidden.slice(1, 0, 1);
    for (int i = 1; i < static_cast<int>(state.mtp_proposal_extent); ++i) {
        Tensor previous_token = state.execution.io.mtp->draft_tokens.slice(0, i - 1, 1);
        Tensor next_draft     = state.execution.io.mtp->draft_tokens.slice(0, i, 1);
        const bool from_a     = (i % 2) == 1;
        Tensor& src_hidden    = from_a ? ar_a : ar_b;
        Tensor& dst_hidden    = from_a ? ar_b : ar_a;
        const auto visible    = static_cast<std::uint32_t>(position + i + 1);
        const ops::GqaExecutionEnvelope envelope{visible, visible};
        card.mtp_forward_ar_step(previous_token, src_hidden, ar_position, envelope, dst_hidden,
                                 logits, next_draft);
        ops::increment_i32_scalar(ar_position, state.execution.device.stream);
    }
    // Odd step count ends in ar_b; keep the resident AR hidden in ar_a for the next round.
    if (((state.mtp_proposal_extent - 1) % 2) == 1) {
        CUDA_CHECK(cudaMemcpyAsync(ar_a.data, ar_b.data, ar_a.bytes(), cudaMemcpyDeviceToDevice,
                                   state.execution.device.stream));
    }
}

auto mtp_decode_batch_body(MtpBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                           MtpGqaEnvelopes envelopes) {
    return [&state, batch_size, k, envelopes] {
        if (batch_size <= 0 || batch_size > static_cast<std::int32_t>(kMaximumConcurrency) ||
            k == 0 || k > kMtpDecodeMaximumDrafts) {
            throw std::logic_error("MTP decode batch state is incomplete");
        }

        qwen3_6::MtpDecodeState& frame = state.frame;
        const std::int32_t width       = static_cast<std::int32_t>(k) + 1;
        CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, &state.host_ingress,
                                   sizeof(qwen3_6::MtpDecodeIngress), cudaMemcpyHostToDevice,
                                   state.execution.device.stream));

        TextContext card(state.execution.device, state.execution.model, state.execution.work, {},
                         state.execution.linear_attention, state.execution.io,
                         state.execution.prefill_hidden, state.execution.prefill_chunk, 0, {},
                         &state.text_cache, &state.mtp_cache);
        Tensor anchors           = frame.anchors.slice(0, 0, batch_size);
        Tensor frontiers         = frame.base_frontiers.slice(0, 0, batch_size);
        Tensor budgets           = frame.remaining_budgets.slice(0, 0, batch_size);
        Tensor current_extents   = frame.current_extents.slice(0, 0, batch_size);
        Tensor target_valid      = frame.target_valid_columns.slice(0, 0, batch_size);
        Tensor current_drafts    = frame.current_drafts.slice(1, 0, batch_size);
        Tensor target_rope       = frame.target_rope_positions.slice(1, 0, batch_size);
        Tensor text_rows         = frame.text_kv_table_rows.slice(0, 0, batch_size);
        Tensor mtp_rows          = frame.mtp_kv_table_rows.slice(0, 0, batch_size);
        Tensor lanes             = frame.lanes.slice(0, 0, batch_size);
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

        const std::int32_t k_ceil = frame.current_drafts.ne[0];
        const bool compact =
            static_cast<std::int32_t>(k) < k_ceil; // LLD Capture/run: no extra nodes when k==N
        if (compact) {
            state.execution.work.reset();
            auto copy_panel = [&](Tensor src, std::int32_t rows) {
                Tensor dst =
                    state.execution.work.alloc(DType::I32, {rows, batch_size});
                qwen3_6::copy_i32_panel(dst, src.slice(0, 0, rows),
                                        state.execution.device.stream);
                return dst;
            };
            current_drafts   = copy_panel(current_drafts, static_cast<std::int32_t>(k));
            target_rope      = copy_panel(target_rope, width);
            Tensor compact_verify =
                state.execution.work.alloc(DType::I32, {width, batch_size});
            Tensor compact_pos =
                state.execution.work.alloc(DType::I32, {width, batch_size});
            verify_ids       = compact_verify;
            target_positions = compact_pos;
            target_tokens    = state.execution.work.alloc(DType::I32, {width, batch_size});
            licensed_tokens  = state.execution.work.alloc(DType::I32, {width, batch_size});
            alignment_ids    = state.execution.work.alloc(DType::I32, {width, batch_size});
            target_logits    = state.execution.work.alloc(
                DType::BF16, {TextConfig::output_rows, width, batch_size});
            target_hidden = state.execution.work.alloc(
                DType::BF16, {TextConfig::hidden, width, batch_size});
            alignment_hidden = state.execution.work.alloc(
                DType::BF16, {TextConfig::hidden, width, batch_size});
            if (k > 1) {
                ar_positions = ar_positions.slice(1, 0, static_cast<std::int32_t>(k) - 1);
                ar_rope_positions =
                    ar_rope_positions.slice(1, 0, static_cast<std::int32_t>(k) - 1);
                ar_valid_columns =
                    ar_valid_columns.slice(1, 0, static_cast<std::int32_t>(k) - 1);
            }
        }

        ops::speculative_prepare_verify_inputs(anchors, current_drafts, frontiers, current_extents,
                                               verify_ids, target_positions,
                                               state.execution.device.stream);
        target_verify_accept(state.execution, state.continuation_hidden_store, card,
                             TargetVerifyFrameView{
                                 .ids             = verify_ids,
                                 .cache_positions = target_positions,
                                 .rope_positions  = target_rope,
                                 .valid_columns   = target_valid,
                                 .kv_table_rows   = text_rows,
                                 .lanes           = lanes,
                                 .target_hidden   = target_hidden,
                                 .target_logits   = target_logits,
                                 .target_tokens   = target_tokens,
                                 .drafts          = current_drafts,
                                 .current_extents = current_extents,
                                 .frontiers       = frontiers,
                                 .anchors         = anchors,
                                 .licensed_tokens = licensed_tokens,
                                 .licensed_counts = licensed_counts,
                                 .accepted_drafts = accepted,
                                 .selected_hidden = selected_hidden,
                                 .replay_records  = state.execution.replay_records,
                                 .sampling        = frame.sampling,
                             },
                             envelopes.target_verify, !compact);
        if (compact) {
            Tensor licensed_frame =
                frame.licensed_tokens.slice(1, 0, batch_size).slice(0, 0, width);
            qwen3_6::copy_i32_panel(licensed_frame, licensed_tokens,
                                    state.execution.device.stream);
            qwen3_6::copy_strided_width_panel(frame.target_hidden.slice(2, 0, batch_size),
                                              target_hidden, state.execution.device.stream);
        }

        ops::mtp_prepare_next_round(verify_ids, anchors, accepted, frontiers, budgets,
                                    licensed_counts, rope_deltas, alignment_ids, next_extents,
                                    ar_positions, ar_rope_positions, ar_valid_columns,
                                    static_cast<std::int32_t>(state.text_cache.max_context()),
                                    state.execution.device.stream);
        card.mtp_forward_decode_batch(alignment_ids, target_hidden, target_positions, target_rope,
                                      licensed_counts, mtp_rows, envelopes.batch, alignment_hidden);
        ops::speculative_select_accepted_hidden(alignment_hidden, accepted, ar_hidden,
                                                state.execution.device.stream);

        Tensor proposal_logits = frame.proposal_logits.slice(1, 0, batch_size);
        Tensor draft0          = next_drafts.slice(1, 0, 1).view({batch_size});
        card.mtp_propose_batch(ar_hidden, proposal_logits, draft0);
        // Alternate ar_hidden <-> next_hidden. For MTP3 (k=3) both AR steps are captured with
        // fixed addresses and the final hidden already lands in ar_hidden — no D2D.
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
            Tensor previous_batch = previous.view({1, batch_size});
            const bool from_ar    = (step % 2U) == 0U;
            Tensor src_hidden =
                (from_ar ? ar_hidden : next_hidden).view({TextConfig::hidden, 1, batch_size});
            Tensor dst_hidden =
                (from_ar ? next_hidden : ar_hidden).view({TextConfig::hidden, 1, batch_size});
            Tensor& propose_hidden = from_ar ? next_hidden : ar_hidden;
            card.mtp_forward_decode_batch(previous_batch, src_hidden, position, rope, valid,
                                          mtp_rows, envelopes.ar[step], dst_hidden);
            card.mtp_propose_batch(propose_hidden, proposal_logits, next);
        }
        if (((k - 1) % 2U) == 1U) {
            CUDA_CHECK(cudaMemcpyAsync(ar_hidden.data, next_hidden.data, ar_hidden.bytes(),
                                       cudaMemcpyDeviceToDevice, state.execution.device.stream));
        }

        CUDA_CHECK(cudaMemcpyAsync(&state.host_egress, frame.egress.data,
                                   sizeof(qwen3_6::MtpDecodeEgress), cudaMemcpyDeviceToHost,
                                   state.execution.device.stream));
    };
}

void capture_mtp_decode_batch(MtpBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                              MtpGqaEnvelopes envelopes, DecodeGraphDefinition& definition) {
    auto body = mtp_decode_batch_body(state, batch_size, k, envelopes);
    capture_graph(state, definition, body);
}

void mtp_decode_batch(MtpBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                      MtpGqaEnvelopes envelopes, DecodeGraphExecutable* executable) {
    auto body = mtp_decode_batch_body(state, batch_size, k, envelopes);
    run_prepared(state, executable, body);
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
