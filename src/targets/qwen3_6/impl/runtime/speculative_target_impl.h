#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"

#include "ninfer/ops/scatter.h"
#include "ninfer/ops/speculative_round.h"

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule {

void target_verify_accept(ExecutionCore& execution, Tensor& continuation_hidden_store,
                          TextContext& card, TargetVerifyFrameView frame,
                          ops::GqaExecutionEnvelope envelope, bool reset_workspace) {
    if (frame.replay_records == nullptr) {
        throw std::logic_error("speculative target verify has no ReplaySSM record storage");
    }
    const bool tree = frame.tree_verify;
    if (tree != (frame.parent_index.data != nullptr) ||
        tree != (frame.ancestor_mask.data != nullptr) ||
        tree != (frame.prefix_lengths.data != nullptr) ||
        tree != (frame.accepted_column.data != nullptr) ||
        tree != (frame.fold_path.data != nullptr)) {
        throw std::logic_error("speculative tree verify frame is incomplete");
    }
    card.set_gdn_state_action(GdnStateAction::RecordForReplay, frame.replay_records);
    if (tree) {
        card.set_tree_verify(&frame.parent_index, &frame.ancestor_mask, &frame.prefix_lengths);
    }
    if (frame.feature_sink != nullptr) {
        card.target_verify_batch(frame.ids, frame.cache_positions, frame.rope_positions,
                                 frame.valid_columns, frame.kv_table_rows, frame.lanes, envelope,
                                 frame.target_hidden, frame.target_logits, frame.target_tokens,
                                 *frame.feature_sink, reset_workspace);
    } else {
        card.target_verify_batch(frame.ids, frame.cache_positions, frame.rope_positions,
                                 frame.valid_columns, frame.kv_table_rows, frame.lanes, envelope,
                                 frame.target_hidden, frame.target_logits, frame.target_tokens,
                                 reset_workspace);
    }
    if (tree) {
        ops::speculative_accept_tree_drafts(
            frame.target_tokens, frame.target_logits, frame.ids, frame.parent_index,
            frame.valid_columns, frame.current_extents, frame.frontiers, frame.anchors,
            frame.licensed_tokens, frame.licensed_counts, frame.accepted_drafts,
            frame.accepted_column, frame.fold_path, TextConfig::token_domain, frame.sampling,
            execution.work, execution.device.stream);
        ops::speculative_select_accepted_hidden(frame.target_hidden, frame.accepted_column,
                                                frame.selected_hidden, execution.device.stream);
        card.set_tree_verify(nullptr, nullptr, nullptr);
    } else {
        ops::speculative_accept_greedy_drafts(
            frame.target_tokens, frame.target_logits, frame.drafts, frame.current_extents,
            frame.frontiers, frame.anchors, frame.licensed_tokens, frame.licensed_counts,
            frame.accepted_drafts, TextConfig::token_domain, frame.sampling, execution.work,
            execution.device.stream,
            frame.draft_selector_ids.data != nullptr ? &frame.draft_selector_ids : nullptr,
            frame.draft_selector_q.data != nullptr ? &frame.draft_selector_q : nullptr);
        ops::speculative_select_accepted_hidden(frame.target_hidden, frame.accepted_drafts,
                                                frame.selected_hidden, execution.device.stream);
    }
    ops::scatter(frame.selected_hidden, frame.lanes, continuation_hidden_store,
                 execution.device.stream);
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS::schedule
