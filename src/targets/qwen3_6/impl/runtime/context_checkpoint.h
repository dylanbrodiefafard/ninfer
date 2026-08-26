#pragma once

#include "prefix_identity.h"

#include <ninfer/types.h>

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::targets::qwen3_6::detail {

enum class ContextCheckpointKind : std::uint32_t {
    Ladder       = 0,
    TurnRollback = 1,
    OnDemand     = 2, // reserved; not written this cut
};

[[nodiscard]] constexpr ninfer::PrefixReusePath
reuse_path_for_context_checkpoint_kind(ContextCheckpointKind kind) noexcept {
    return kind == ContextCheckpointKind::TurnRollback
               ? ninfer::PrefixReusePath::RestoreTurnRollback
               : ninfer::PrefixReusePath::RestoreContextCheckpoint;
}

struct PrefillReuseHead {
    std::uint32_t frontier     = 0;
    ContextCheckpointKind kind = ContextCheckpointKind::Ladder;
};

// Prefill chunk-end thresholds. Advertised restore frontier is the committed chunk end
// at or past the mark, never the raw named size (24000, 36000, 150000, ...).
inline constexpr std::array<std::uint32_t, 6> kPrefillContextMarks = {
    24576u, 36864u, 53248u, 77824u, 102400u, 151552u};

[[nodiscard]] constexpr std::optional<std::uint32_t>
next_prefill_context_mark(std::uint32_t frontier) noexcept {
    for (const std::uint32_t mark : kPrefillContextMarks) {
        if (mark > frontier) { return mark; }
    }
    return std::nullopt;
}

// Prefill ladder capture follows `--no-prefix-reuse` and requires MTP or DFlash.
[[nodiscard]] constexpr bool capture_prefill_context_checkpoints(bool allow_prefix_reuse,
                                                                 bool speculative) noexcept {
    return allow_prefix_reuse && speculative;
}

// Restore to R keeps heads at R so a later request can hit R after current moves on.
[[nodiscard]] constexpr bool retain_context_checkpoint_head(std::uint32_t head_frontier,
                                                            std::uint32_t restore_frontier) noexcept {
    return head_frontier <= restore_frontier;
}

// Freeze after a committed prefill chunk when the chunk end has reached the next mark.
// Advertised restore frontier is that chunk end, not the raw named mark.
[[nodiscard]] constexpr bool should_freeze_prefill_context_checkpoint(
    bool prefill_chunk, bool capture_enabled, std::uint32_t chunk_end, std::uint32_t next_mark,
    bool already_has_head_at_end, bool prefix_items_complete) noexcept {
    if (!prefill_chunk || !capture_enabled) { return false; }
    if (next_mark == 0 || chunk_end < next_mark) { return false; }
    if (already_has_head_at_end || !prefix_items_complete) { return false; }
    return true;
}

[[nodiscard]] constexpr std::uint32_t
advertised_context_checkpoint_frontier(std::uint32_t chunk_end) noexcept {
    return chunk_end;
}

// Span between two advertised head frontiers. Public restore/capture fields report the
// absolute head F, not this increment.
[[nodiscard]] constexpr std::uint32_t newly_frozen_context_checkpoint_tokens(
    std::uint32_t advertised_frontier, std::uint32_t existing_frontier) noexcept {
    return advertised_frontier > existing_frontier ? advertised_frontier - existing_frontier : 0;
}

[[nodiscard]] constexpr bool staging_holds_restore_identity(
    bool occupied, std::uint32_t staging_lane, PrefixHash128 staging_hash,
    std::uint32_t staging_frontier, std::uint32_t lane, PrefixHash128 hash,
    std::uint32_t frontier) noexcept {
    return occupied && staging_lane == lane && staging_frontier == frontier &&
           staging_hash == hash;
}

[[nodiscard]] constexpr bool is_rewrite_checkpoint_restore(ninfer::PrefixReusePath path) noexcept {
    return path == ninfer::PrefixReusePath::RestoreTurnCheckpoint ||
           path == ninfer::PrefixReusePath::RestoreResponseCheckpoint;
}

[[nodiscard]] constexpr bool is_staged_checkpoint_restore(ninfer::PrefixReusePath path) noexcept {
    return path == ninfer::PrefixReusePath::RestoreContextCheckpoint ||
           path == ninfer::PrefixReusePath::RestoreTurnRollback;
}

[[nodiscard]] constexpr bool is_complete_checkpoint_restore(ninfer::PrefixReusePath path) noexcept {
    return is_rewrite_checkpoint_restore(path) || is_staged_checkpoint_restore(path);
}

// Occupy-append pin of the completed frontier before suffix prefill. Exact-hit
// (prompt_tokens == E) must not replace an older rollback head.
[[nodiscard]] constexpr bool should_capture_turn_rollback(
    ninfer::PrefixReusePath reuse, std::uint32_t execution_frontier, std::uint32_t prompt_tokens,
    bool capture_enabled, bool tail_hidden_valid, bool already_has_head_at_e,
    bool prefix_items_complete) noexcept {
    if (reuse != ninfer::PrefixReusePath::AppendAtFrontier) { return false; }
    if (execution_frontier == 0 || prompt_tokens <= execution_frontier) { return false; }
    if (!capture_enabled || !tail_hidden_valid || already_has_head_at_e || !prefix_items_complete) {
        return false;
    }
    return true;
}

// RestoreContextCheckpoint with mtp_kv_valid >= F-1 is ready; it is not forced to FullReset.
[[nodiscard]] constexpr bool mtp_complete_checkpoint_ready(ninfer::PrefixReusePath reuse,
                                                           std::uint32_t reuse_base,
                                                           std::uint32_t mtp_kv_valid) noexcept {
    return is_complete_checkpoint_restore(reuse) && reuse_base != 0 &&
           mtp_kv_valid >= reuse_base - 1;
}

// Planner FullReset fallback when MTP cannot legally continue the selected reuse path.
[[nodiscard]] constexpr bool mtp_prefix_reuse_ready(ninfer::PrefixReusePath reuse,
                                                    std::uint32_t reuse_base,
                                                    std::uint32_t mtp_kv_valid,
                                                    bool tail_hidden_valid,
                                                    bool mtp_cache) noexcept {
    if (reuse == ninfer::PrefixReusePath::FullReset) { return true; }
    if (!mtp_cache) { return false; }
    if (reuse == ninfer::PrefixReusePath::AppendAtFrontier) {
        return tail_hidden_valid && (reuse_base == 0 || mtp_kv_valid >= reuse_base - 1);
    }
    return mtp_complete_checkpoint_ready(reuse, reuse_base, mtp_kv_valid);
}

// Staging D2D restore is legal only when a host head exists for that identity.
[[nodiscard]] constexpr bool restore_may_d2d_staging(bool staging_holds_identity,
                                                     bool head_present) noexcept {
    return staging_holds_identity && head_present;
}

[[nodiscard]] constexpr bool mtp_bridge_reads_rewrite_hidden(ninfer::PrefixReusePath path) noexcept {
    return is_rewrite_checkpoint_restore(path);
}

struct PrefillReuseSelection {
    ninfer::PrefixReusePath path = ninfer::PrefixReusePath::FullReset;
    std::uint32_t frontier       = 0;
};

// Current matching frontier always wins. Otherwise the longest complete rewrite or
// staged head. Same-F rewrite vs rollback/ladder keeps rewrite (`>` not `>=`).
[[nodiscard]] inline PrefillReuseSelection select_resident_prefill_reuse(
    bool current_matches, std::uint32_t execution_frontier, bool rewrite_matches,
    std::uint32_t rewrite_frontier, ninfer::PrefixReusePath rewrite_path,
    std::span<const PrefillReuseHead> matching_heads) {
    if (current_matches && execution_frontier != 0) {
        return {ninfer::PrefixReusePath::AppendAtFrontier, execution_frontier};
    }
    PrefillReuseSelection best;
    if (rewrite_matches && rewrite_frontier != 0 && rewrite_frontier > best.frontier) {
        best.path     = rewrite_path;
        best.frontier = rewrite_frontier;
    }
    for (const PrefillReuseHead& head : matching_heads) {
        if (head.frontier > best.frontier) {
            best.path     = reuse_path_for_context_checkpoint_kind(head.kind);
            best.frontier = head.frontier;
        }
    }
    return best;
}

[[nodiscard]] inline PrefillReuseSelection select_resident_prefill_reuse(
    bool current_matches, std::uint32_t execution_frontier, bool rewrite_matches,
    std::uint32_t rewrite_frontier, ninfer::PrefixReusePath rewrite_path,
    std::span<const std::uint32_t> matching_ladder_frontiers) {
    std::vector<PrefillReuseHead> heads;
    heads.reserve(matching_ladder_frontiers.size());
    for (const std::uint32_t frontier : matching_ladder_frontiers) {
        heads.push_back(PrefillReuseHead{frontier, ContextCheckpointKind::Ladder});
    }
    return select_resident_prefill_reuse(current_matches, execution_frontier, rewrite_matches,
                                         rewrite_frontier, rewrite_path, heads);
}

[[nodiscard]] constexpr bool occupy_drops_rewrite_ahead_of_restore(
    ninfer::PrefixReusePath reuse, bool rewrite_valid, std::uint32_t rewrite_frontier,
    std::uint32_t restore_frontier) noexcept {
    return is_staged_checkpoint_restore(reuse) && rewrite_valid &&
           rewrite_frontier > restore_frontier;
}

[[nodiscard]] constexpr bool occupy_clears_context_checkpoints(
    ninfer::PrefixReusePath reuse) noexcept {
    return reuse == ninfer::PrefixReusePath::FullReset;
}

[[nodiscard]] constexpr bool occupy_keeps_context_checkpoint_head(
    ninfer::PrefixReusePath reuse, std::uint32_t head_frontier, std::uint32_t new_base) noexcept {
    if (occupy_clears_context_checkpoints(reuse)) { return false; }
    if (reuse == ninfer::PrefixReusePath::AppendAtFrontier ||
        is_staged_checkpoint_restore(reuse) || is_rewrite_checkpoint_restore(reuse)) {
        return retain_context_checkpoint_head(head_frontier, new_base);
    }
    return false;
}

} // namespace ninfer::targets::qwen3_6::detail
