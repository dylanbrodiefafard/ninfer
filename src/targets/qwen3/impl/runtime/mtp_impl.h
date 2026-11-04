#include "targets/qwen3/impl/runtime/instance.h"
#include "targets/qwen3/impl/runtime/panel_copy.h"
#include "targets/qwen3/impl/runtime/schedule.h"

#include "ninfer/ops/mtp_round.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/scalar.h"

#include <hip/hip_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::targets::qwen3::detail::NINFER_QWEN3_RUNTIME_NS::schedule {
// NOLINTNEXTLINE(misc-anonymous-namespace-in-header): single-TU fragment (runtime.hip)
namespace {

class MtpKvTransactionBatch {
public:
    explicit MtpKvTransactionBatch(std::size_t count) : size_(count) {
        if (count == 0U || count > transactions_.size()) {
            throw std::invalid_argument("MTP KV transaction batch size must be in [1,16]");
        }
    }

    MtpKvTransactionBatch(const MtpKvTransactionBatch&)            = delete;
    MtpKvTransactionBatch& operator=(const MtpKvTransactionBatch&) = delete;

    void append(qwen3::PagedKVTransaction transaction) {
        if (next_ >= size_) { throw std::logic_error("MTP KV transaction batch overflow"); }
        bindings_[next_] = &transactions_[next_].emplace(std::move(transaction));
        ++next_;
    }

    [[nodiscard]] std::span<qwen3::PagedKVTransaction* const> binding() const {
        if (next_ != size_) { throw std::logic_error("MTP KV transaction batch is incomplete"); }
        return {bindings_.data(), size_};
    }

    void commit() {
        if (next_ != size_) { throw std::logic_error("MTP KV transaction batch is incomplete"); }
        for (std::size_t row = 0; row < size_; ++row) {
            committed_frontiers_[row] = transactions_[row].value().commit();
        }
        committed_ = true;
    }

    [[nodiscard]] std::uint32_t committed_frontier(std::size_t row) const {
        if (!committed_ || row >= size_) {
            throw std::logic_error("MTP KV transaction frontiers are unavailable");
        }
        return committed_frontiers_[row];
    }

private:
    std::array<std::optional<qwen3::PagedKVTransaction>, kMaximumConcurrency> transactions_{};
    std::array<qwen3::PagedKVTransaction*, kMaximumConcurrency> bindings_{};
    std::array<std::uint32_t, kMaximumConcurrency> committed_frontiers_{};
    std::size_t size_ = 0;
    std::size_t next_ = 0;
    bool committed_   = false;
};

void require_i32_panel(const Tensor& tensor, std::int32_t width, std::int32_t batch,
                       const char* label) {
    if (width <= 0 || batch <= 0 || tensor.dtype != DType::I32 || tensor.data == nullptr ||
        tensor.ne[0] != width || tensor.ne[1] != batch || tensor.ne[2] != 1 || tensor.ne[3] != 1 ||
        !tensor.is_contiguous()) {
        throw std::invalid_argument(std::string(label) + " must be contiguous I32 [width,batch]");
    }
}

void require_i32_vector(const Tensor& tensor, std::int32_t count, const char* label) {
    if (count <= 0 || tensor.dtype != DType::I32 || tensor.data == nullptr ||
        tensor.ne[0] != count || tensor.ne[1] != 1 || tensor.ne[2] != 1 || tensor.ne[3] != 1 ||
        !tensor.is_contiguous()) {
        throw std::invalid_argument(std::string(label) + " must be contiguous I32 [batch]");
    }
}

void begin_transaction_segments(std::span<qwen3::PagedKVTransaction* const> transactions,
                                const Tensor& positions, const Tensor& device_counts,
                                std::span<const std::int32_t> host_maximum_counts) {
    const std::int32_t width = positions.ne[0];
    const std::int32_t batch = device_counts.ne[0];
    require_i32_panel(positions, width, batch, "MTP KV positions");
    require_i32_vector(device_counts, batch, "MTP KV active counts");
    if (transactions.size() != static_cast<std::size_t>(batch) ||
        (!host_maximum_counts.empty() &&
         host_maximum_counts.size() != static_cast<std::size_t>(batch))) {
        throw std::logic_error("MTP KV authorities do not match the packed batch");
    }
    const auto* all_positions = static_cast<const std::int32_t*>(positions.data);
    const auto* all_counts    = static_cast<const std::int32_t*>(device_counts.data);
    for (std::int32_t row = 0; row < batch; ++row) {
        const std::int32_t maximum = host_maximum_counts.empty()
                                         ? width
                                         : host_maximum_counts[static_cast<std::size_t>(row)];
        auto* transaction          = transactions[static_cast<std::size_t>(row)];
        if (transaction == nullptr || maximum <= 0 || maximum > width) {
            throw std::logic_error("MTP KV row has an invalid authority or append maximum");
        }
        transaction->begin_device_segment(all_positions + static_cast<std::size_t>(row) *
                                                              static_cast<std::size_t>(width),
                                          static_cast<std::size_t>(maximum), all_counts + row);
    }
}

void end_transaction_segments(std::span<qwen3::PagedKVTransaction* const> transactions,
                              hipStream_t stream) {
    for (qwen3::PagedKVTransaction* transaction : transactions) {
        if (transaction == nullptr) {
            throw std::logic_error("MTP KV segment binding contains null");
        }
        transaction->end_device_segment(stream);
    }
}

template <class Invoke>
void run_single_mtp_append(PrefillContext& state, TextContext& card, const Tensor& position,
                           Invoke&& invoke) {
    if (state.mtp_cache == nullptr || state.mtp_kv_allocation == nullptr ||
        state.mtp_kv_publication == nullptr || state.mtp_kv_status == nullptr) {
        throw std::logic_error("MTP bridge has no FP8-K/INT4-V transaction authority");
    }
    require_i32_vector(position, 1, "MTP bridge position");
    const std::uint32_t expected = state.mtp_kv_publication->valid_frontier + 1U;
    MtpKvTransactionBatch transactions(1);
    transactions.append(state.mtp_cache->begin_device_append(
        *state.mtp_kv_allocation, *state.mtp_kv_publication,
        static_cast<const std::int32_t*>(position.data), 1U,
        qwen3::PagedKVTransactionWorkspace{.status = state.mtp_kv_status}));
    card.set_mtp_kv_transactions(transactions.binding());
    std::forward<Invoke>(invoke)();
    transactions.commit();
    if (state.mtp_kv_publication->valid_frontier != expected) {
        throw std::logic_error("MTP bridge transaction published a wrong frontier");
    }
}

} // namespace

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
    TextContext card(state.execution.device, state.execution.model,
                     state.execution.linear_execution, state.execution.work, state.text_kv,
                     state.execution.linear_attention, state.execution.io,
                     state.execution.prefill_hidden, state.execution.prefill_chunk,
                     state.text_kv_base, state.mtp_kv, state.mtp_cache);
    configure_text_card(card, state.execution, state.sampling, state.current_state_slot,
                        state.mtp_proposal_extent);

    Tensor position_view = state.execution.io.mtp->target_positions.slice(0, 0, 1);
    ops::set_i32_scalar(position_view, position, state.execution.device.stream);
    Tensor mtp_hidden         = state.execution.io.mtp->ar_hidden;
    Tensor logits             = state.execution.io.logits.slice(1, 0, 1);
    Tensor draft0             = state.execution.io.mtp->draft_tokens.slice(0, 0, 1);
    Tensor rope_position_view = state.execution.work.alloc(DType::I32, {1, 3});
    HIP_CHECK(hipMemcpyAsync(rope_position_view.data, rope_position.data(),
                             rope_position.size_bytes(), hipMemcpyHostToDevice,
                             state.execution.device.stream));
    run_single_mtp_append(state, card, position_view, [&] {
        card.mtp_forward_batch(next_token, previous_hidden, position_view, mtp_hidden,
                               build_proposal ? 0 : -1, build_proposal ? &logits : nullptr,
                               build_proposal ? &draft0 : nullptr, &rope_position_view,
                               next_embedding);
    });
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
        run_single_mtp_append(state, card, ar_position, [&] {
            card.mtp_forward_ar_step(previous_token, src_hidden, ar_position, dst_hidden, logits,
                                     next_draft);
        });
        ops::increment_i32_scalar(ar_position, state.execution.device.stream);
    }
    // Odd step count ends in ar_b; keep the resident AR hidden in ar_a for the next round.
    if (((state.mtp_proposal_extent - 1) % 2) == 1) {
        HIP_CHECK(hipMemcpyAsync(ar_a.data, ar_b.data, ar_a.bytes(), hipMemcpyDeviceToDevice,
                                 state.execution.device.stream));
    }
    const std::uint32_t committed_frontier = static_cast<std::uint32_t>(position) + 1U;
    if (state.mtp_kv_publication->valid_frontier != committed_frontier) {
        state.mtp_cache->truncate_publication(*state.mtp_kv_allocation, *state.mtp_kv_publication,
                                              committed_frontier);
    }
}

auto mtp_decode_batch_body(MtpBatchContext& state, std::int32_t batch_size, std::uint32_t k) {
    return [&state, batch_size, k] {
        if (batch_size <= 0 || batch_size > static_cast<std::int32_t>(kMaximumConcurrency) ||
            k == 0 || k > kMtpDecodeMaximumDrafts) {
            throw std::logic_error("MTP decode batch state is incomplete");
        }

        qwen3::MtpDecodeState& frame = state.frame;
        const std::int32_t width     = static_cast<std::int32_t>(k) + 1;
        HIP_CHECK(hipMemcpyAsync(frame.ingress.data, &state.host_ingress,
                                 sizeof(qwen3::MtpDecodeIngress), hipMemcpyHostToDevice,
                                 state.execution.device.stream));

        TextContext card(
            state.execution.device, state.execution.model, state.execution.linear_execution,
            state.execution.work, {}, state.execution.linear_attention, state.execution.io,
            state.execution.prefill_hidden, state.execution.prefill_chunk, 0, {}, &state.mtp_cache);
        if (!state.text_kv_transactions.empty()) {
            card.set_text_kv_transactions(state.text_kv_transactions);
        }
        Tensor anchors           = frame.anchors.slice(0, 0, batch_size);
        Tensor base_frontiers    = frame.base_frontiers.slice(0, 0, batch_size);
        Tensor frontiers         = frame.updated_frontiers.slice(0, 0, batch_size);
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
                Tensor dst = state.execution.work.alloc(DType::I32, {rows, batch_size});
                qwen3::copy_i32_panel(dst, src.slice(0, 0, rows), state.execution.device.stream);
                return dst;
            };
            current_drafts        = copy_panel(current_drafts, static_cast<std::int32_t>(k));
            target_rope           = copy_panel(target_rope, width);
            Tensor compact_verify = state.execution.work.alloc(DType::I32, {width, batch_size});
            // The Program opens target-cache transactions before entering this schedule. Keep the
            // exact packed position panel in persistent round storage rather than scratch whose
            // address and lifetime are owned by the workspace arena.
            Tensor compact_pos =
                frame.target_positions
                    .view({static_cast<std::int32_t>(frame.target_positions.numel())})
                    .slice(0, 0, width * batch_size)
                    .view({width, batch_size});
            verify_ids       = compact_verify;
            target_positions = compact_pos;
            target_tokens    = state.execution.work.alloc(DType::I32, {width, batch_size});
            licensed_tokens  = state.execution.work.alloc(DType::I32, {width, batch_size});
            alignment_ids    = state.execution.work.alloc(DType::I32, {width, batch_size});
            target_logits    = state.execution.work.alloc(
                DType::BF16, {TextConfig::output_rows, width, batch_size});
            target_hidden =
                state.execution.work.alloc(DType::BF16, {TextConfig::hidden, width, batch_size});
            alignment_hidden =
                state.execution.work.alloc(DType::BF16, {TextConfig::hidden, width, batch_size});
            if (k > 1) {
                ar_positions      = ar_positions.slice(1, 0, static_cast<std::int32_t>(k) - 1);
                ar_rope_positions = ar_rope_positions.slice(1, 0, static_cast<std::int32_t>(k) - 1);
                ar_valid_columns  = ar_valid_columns.slice(1, 0, static_cast<std::int32_t>(k) - 1);
            }
        }

        qwen3::copy_i32_panel(frontiers, base_frontiers, state.execution.device.stream);
        ops::speculative_prepare_verify_inputs(anchors, current_drafts, base_frontiers,
                                               current_extents, verify_ids, target_positions,
                                               state.execution.device.stream);
        begin_transaction_segments(state.text_kv_transactions, target_positions, target_valid, {});
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
                                 .token_masks     = state.token_masks,
                             },
                             !compact);
        qwen3::record_round_logprobs(frame.logprobs, target_logits, licensed_tokens,
                                     &licensed_counts, nullptr, TextConfig::token_domain,
                                     state.execution.device.stream);
        end_transaction_segments(state.text_kv_transactions, state.execution.device.stream);
        if (compact) {
            Tensor licensed_frame =
                frame.licensed_tokens.slice(1, 0, batch_size).slice(0, 0, width);
            qwen3::copy_i32_panel(licensed_frame, licensed_tokens, state.execution.device.stream);
            qwen3::copy_strided_width_panel(frame.target_hidden.slice(2, 0, batch_size),
                                            target_hidden, state.execution.device.stream);
        }

        ops::mtp_prepare_next_round(verify_ids, anchors, accepted, frontiers, budgets,
                                    licensed_counts, rope_deltas, alignment_ids, next_extents,
                                    ar_positions, ar_rope_positions, ar_valid_columns,
                                    static_cast<std::int32_t>(state.text_cache.max_context()),
                                    state.execution.device.stream);
        begin_transaction_segments(
            state.mtp_kv_transactions, target_positions, licensed_counts,
            {state.host_ingress.target_valid_columns.data(), static_cast<std::size_t>(batch_size)});
        card.set_mtp_kv_transactions(state.mtp_kv_transactions);
        card.mtp_forward_decode_batch(alignment_ids, target_hidden, target_positions, target_rope,
                                      licensed_counts, mtp_rows, alignment_hidden);
        end_transaction_segments(state.mtp_kv_transactions, state.execution.device.stream);
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
            begin_transaction_segments(state.mtp_kv_transactions, position, valid, {});
            card.set_mtp_kv_transactions(state.mtp_kv_transactions);
            card.mtp_forward_decode_batch(previous_batch, src_hidden, position, rope, valid,
                                          mtp_rows, dst_hidden);
            end_transaction_segments(state.mtp_kv_transactions, state.execution.device.stream);
            card.mtp_propose_batch(propose_hidden, proposal_logits, next);
        }
        if (((k - 1) % 2U) == 1U) {
            HIP_CHECK(hipMemcpyAsync(ar_hidden.data, next_hidden.data, ar_hidden.bytes(),
                                     hipMemcpyDeviceToDevice, state.execution.device.stream));
        }

        HIP_CHECK(hipMemcpyAsync(&state.host_egress, frame.egress.data,
                                 sizeof(qwen3::MtpDecodeEgress), hipMemcpyDeviceToHost,
                                 state.execution.device.stream));
    };
}

void capture_mtp_decode_batch(MtpBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                              DecodeGraphDefinition& definition,
                              ops::PersistentDecodeStorage& storage) {
    auto body = mtp_decode_batch_body(state, batch_size, k);
    capture_graph(state, definition, batch_size, storage, body);
}

void mtp_decode_batch(MtpBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                      DecodeGraphExecutable* executable) {
    auto body = mtp_decode_batch_body(state, batch_size, k);
    run_prepared(state, executable, body);
}

} // namespace ninfer::targets::qwen3::detail::NINFER_QWEN3_RUNTIME_NS::schedule
