#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"

#include "ninfer/ops/sampling.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/scalar.h"

#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {
namespace {

auto ordinary_body(State& state, bool align_mtp, ops::GqaExecutionEnvelope envelope) {
    auto record = [&state, align_mtp, envelope] {
        if (align_mtp && !state.io.mtp) {
            throw std::logic_error("MTP alignment requires MTP round state");
        }
        TextContext card(state.device, state.model, state.work, state.text_kv,
                         state.linear_attention, state.io, state.prefill_hidden,
                         state.prefill_chunk, state.text_kv_base,
                         align_mtp ? state.mtp_kv : qwen3_6::PagedKVCacheView());
        configure_text_card(card, state);

        Tensor verify_id = state.io.speculative.target_input_ids.slice(0, 0, 1);
        Tensor position  = state.io.speculative.target_positions.slice(0, 0, 1);
        ops::assign_i32_scalar(state.io.token, verify_id, state.device.stream);
        ops::assign_i32_scalar(state.io.pos, position, state.device.stream);
        target_verify(card, state, verify_id, position, envelope);
        if (state.dflash != nullptr) {
            ops::set_i32_scalar(state.dflash->commit_count, 1, state.device.stream);
            Tensor features  = state.dflash->target_features.slice(1, 0, 1);
            Tensor positions = state.dflash->feature_positions.slice(0, 0, 1);
            dflash_append_context(state, features, positions, state.dflash->commit_count, {1, 1});
        }

        Tensor logits = state.io.logits.slice(1, 0, 1);
        ops::sample(logits, state.io.token, TextConfig::token_domain, state.sampling, state.io.pos,
                    ops::kSamplePurposeDecode, state.work, state.device.stream);

        if (align_mtp) {
            Tensor hidden     = state.io.verify_hidden.slice(1, 0, 1);
            Tensor mtp_hidden = state.io.mtp->ar_hidden;
            card.mtp_forward_batch(state.io.token, hidden, position, envelope, mtp_hidden, -1,
                                   nullptr, nullptr);
        }

        ops::increment_i32_scalar(state.io.pos, state.device.stream);
        ops::increment_i32_scalar(state.io.rope_pos, state.device.stream);
        ops::assign_i32_scalar(state.io.linear_state_snapshot_base_slot,
                               state.io.linear_state_read_slot, state.device.stream);
        if (state.mtp_kv.valid() || state.dflash != nullptr) {
            Tensor fallback_steps = state.io.speculative.stats.slice(0, 3, 1);
            ops::increment_i64_scalar(fallback_steps, state.device.stream);
        }
    };
    return record;
}

auto ordinary_batch_body(State& state, std::int32_t batch_size,
                         ops::GqaExecutionEnvelope envelope) {
    return [&state, batch_size, envelope] {
        if (batch_size <= 0 || batch_size > static_cast<std::int32_t>(kMaximumConcurrency) ||
            state.ordinary_host_ingress == nullptr || state.ordinary_host_egress == nullptr ||
            state.continuation_hidden_store == nullptr) {
            throw std::logic_error("ordinary decode batch state is incomplete");
        }

        qwen3_6::OrdinaryDecodeState& ordinary = state.io.ordinary;
        CUDA_CHECK(cudaMemcpyAsync(ordinary.ingress.data, state.ordinary_host_ingress,
                                   sizeof(qwen3_6::OrdinaryDecodeIngress), cudaMemcpyHostToDevice,
                                   state.device.stream));

        TextContext card(state.device, state.model, state.work, state.text_kv,
                         state.linear_attention, state.io, state.prefill_hidden,
                         state.prefill_chunk, state.text_kv_base);
        configure_text_card(card, state);

        Tensor tokens          = ordinary.tokens.slice(0, 0, batch_size);
        Tensor cache_positions = ordinary.cache_positions.slice(0, 0, batch_size);
        Tensor rope_positions  = ordinary.rope_positions.slice(0, 0, batch_size);
        Tensor kv_rows         = ordinary.text_kv_table_rows.slice(0, 0, batch_size);
        Tensor read_slots      = ordinary.linear_state_read_slots.slice(0, 0, batch_size);
        Tensor snapshot_slots  = ordinary.linear_state_snapshot_base_slots.slice(0, 0, batch_size);
        Tensor continuation_slots = ordinary.continuation_slots.slice(0, 0, batch_size);
        Tensor hidden             = ordinary.hidden.slice(1, 0, batch_size);
        Tensor logits             = ordinary.logits.slice(1, 0, batch_size);
        Tensor sampled            = ordinary.sampled_tokens.slice(0, 0, batch_size);

        card.ordinary_decode_batch(tokens, cache_positions, rope_positions, kv_rows, read_slots,
                                   snapshot_slots, envelope, hidden, logits);
        ops::scatter(hidden, continuation_slots, *state.continuation_hidden_store,
                     state.device.stream);
        ops::sample(logits, sampled, TextConfig::token_domain, ordinary.sampling, cache_positions,
                    ops::kSamplePurposeDecode, state.work, state.device.stream);
        CUDA_CHECK(cudaMemcpyAsync(state.ordinary_host_egress, ordinary.egress.data,
                                   sizeof(qwen3_6::OrdinaryDecodeEgress), cudaMemcpyDeviceToHost,
                                   state.device.stream));
    };
}

} // namespace

void warm_capture_ordinary_round(State& state, bool align_mtp, ops::GqaExecutionEnvelope envelope,
                                 const GraphPrepare& prepare, DecodeGraphDefinition& definition) {
    auto body = ordinary_body(state, align_mtp, envelope);
    warm_capture(state, definition, prepare, body);
}

void ordinary_round(State& state, bool align_mtp, ops::GqaExecutionEnvelope envelope,
                    DecodeGraphExecutable* executable) {
    auto body = ordinary_body(state, align_mtp, envelope);
    run_prepared(state, executable, body);
}

void warm_capture_ordinary_decode_batch(State& state, std::int32_t batch_size,
                                        ops::GqaExecutionEnvelope envelope,
                                        const GraphPrepare& prepare,
                                        DecodeGraphDefinition& definition) {
    auto body = ordinary_batch_body(state, batch_size, envelope);
    warm_capture(state, definition, prepare, body);
}

void ordinary_decode_batch(State& state, std::int32_t batch_size,
                           ops::GqaExecutionEnvelope envelope, DecodeGraphExecutable* executable) {
    auto body = ordinary_batch_body(state, batch_size, envelope);
    run_prepared(state, executable, body);
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
