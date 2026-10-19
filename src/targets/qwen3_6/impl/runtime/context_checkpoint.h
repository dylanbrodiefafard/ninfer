#pragma once

#include "prefix_identity.h"

#include <ninfer/types.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
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

// Upper bound on the checkpoint images all lanes own at once; it sizes the Program's startup
// pool. A lane owns at most one turn-rollback image plus one ladder image per mark it has
// crossed: its k-th ladder image lies at or past marks[k-1], marks above the sequence capacity
// are unreachable, and every image lies inside the lane's resident Main Text KV, which all
// lanes share within kv_capacity tokens. The bound maximizes the lanes' ladder images under
// that shared budget (lanes are interchangeable, so nonincreasing per-lane counts suffice) and
// adds one rollback image per lane. An entry restored from a cache written with other marks can
// exceed it; acquisition then skips the optional image instead of allocating.
[[nodiscard]] inline std::size_t
context_checkpoint_image_pool_capacity(std::uint32_t max_concurrency,
                                       std::span<const std::uint32_t> marks,
                                       std::uint32_t sequence_capacity, std::uint64_t kv_capacity) {
    std::size_t reachable = 0;
    while (reachable < marks.size() && marks[reachable] <= sequence_capacity) { ++reachable; }
    const auto best = [&](const auto& self, std::uint32_t lanes, std::size_t max_k,
                          std::uint64_t budget) -> std::size_t {
        std::size_t result = 0;
        if (lanes == 0) { return result; }
        for (std::size_t k = max_k + 1; k-- > 0;) {
            if (result >= k * lanes) { break; }
            const std::uint64_t cost = k == 0 ? 0 : marks[k - 1];
            if (cost > budget) { continue; }
            result = std::max(result, k + self(self, lanes - 1, k, budget - cost));
        }
        return result;
    };
    return static_cast<std::size_t>(max_concurrency) +
           best(best, max_concurrency, reachable, kv_capacity);
}

// Prefill chunk-end thresholds. Advertised restore frontier is the committed chunk end
// at or past the mark, never the raw named size (24000, 36000, 150000, ...).
inline constexpr std::array<std::uint32_t, 6> kPrefillContextMarks = {24576u, 36864u,  53248u,
                                                                      77824u, 102400u, 151552u};

[[nodiscard]] constexpr std::optional<std::uint32_t>
next_prefill_context_mark(std::uint32_t frontier, std::span<const std::uint32_t> marks) noexcept {
    for (const std::uint32_t mark : marks) {
        if (mark > frontier) { return mark; }
    }
    return std::nullopt;
}

[[nodiscard]] constexpr std::optional<std::uint32_t>
next_prefill_context_mark(std::uint32_t frontier) noexcept {
    return next_prefill_context_mark(frontier, kPrefillContextMarks);
}

[[nodiscard]] constexpr std::uint32_t
first_prefill_context_mark(std::span<const std::uint32_t> marks) noexcept {
    return marks.empty() ? 0 : marks.front();
}

[[nodiscard]] inline std::vector<std::uint32_t>
resolved_prefill_context_marks(const std::optional<std::vector<std::uint32_t>>& configured) {
    if (!configured) { return {kPrefillContextMarks.begin(), kPrefillContextMarks.end()}; }
    return *configured;
}

inline void validate_configured_context_checkpoint_marks(
    const std::optional<std::vector<std::uint32_t>>& configured, ninfer::SpeculativeBackend spec) {
    if (!configured) { return; }
    const std::vector<std::uint32_t>& marks = *configured;
    if (marks.size() > ninfer::kMaxContextCheckpointMarks) {
        throw std::invalid_argument("context-checkpoint mark table accepts at most 16 marks");
    }
    std::uint32_t previous = 0;
    for (const std::uint32_t mark : marks) {
        if (mark == 0 || mark <= previous) {
            throw std::invalid_argument(
                "context-checkpoint marks must be strictly increasing and greater than 0");
        }
        previous = mark;
    }
    if (!marks.empty() && spec == SpeculativeBackend::None) {
        throw std::invalid_argument(
            "custom context-checkpoint marks require speculative backend mtp or dflash");
    }
}

// Prefill ladder capture follows `--no-prefix-reuse` and requires MTP or DFlash.
[[nodiscard]] constexpr bool capture_prefill_context_checkpoints(bool allow_prefix_reuse,
                                                                 bool speculative) noexcept {
    return allow_prefix_reuse && speculative;
}

// Restore to R keeps heads at R so a later request can hit R after current moves on.
[[nodiscard]] constexpr bool
retain_context_checkpoint_head(std::uint32_t head_frontier,
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
[[nodiscard]] constexpr std::uint32_t
newly_frozen_context_checkpoint_tokens(std::uint32_t advertised_frontier,
                                       std::uint32_t existing_frontier) noexcept {
    return advertised_frontier > existing_frontier ? advertised_frontier - existing_frontier : 0;
}

[[nodiscard]] constexpr bool staging_holds_restore_identity(bool occupied,
                                                            std::uint32_t staging_lane,
                                                            PrefixHash128 staging_hash,
                                                            std::uint32_t staging_frontier,
                                                            std::uint32_t lane, PrefixHash128 hash,
                                                            std::uint32_t frontier) noexcept {
    return occupied && staging_lane == lane && staging_frontier == frontier && staging_hash == hash;
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
[[nodiscard]] constexpr bool
should_capture_turn_rollback(ninfer::PrefixReusePath reuse, std::uint32_t execution_frontier,
                             std::uint32_t prompt_tokens, bool capture_enabled,
                             bool tail_hidden_valid, bool already_has_head_at_e,
                             bool prefix_items_complete) noexcept {
    if (reuse != ninfer::PrefixReusePath::AppendAtFrontier) { return false; }
    if (execution_frontier == 0 || prompt_tokens <= execution_frontier) { return false; }
    if (!capture_enabled || !tail_hidden_valid || already_has_head_at_e || !prefix_items_complete) {
        return false;
    }
    return true;
}

// Occupy-time exact-hit / decode-only pin. Independent of reuse path; does not
// replace should_capture_turn_rollback (auto still requires AppendAtFrontier
// and prompt_tokens > E).
[[nodiscard]] constexpr bool
should_capture_exact_hit_pin(bool request_pin, std::uint32_t execution_frontier,
                             std::uint32_t prompt_tokens, bool capture_enabled,
                             bool tail_hidden_valid, bool already_has_head_at_e,
                             bool prefix_items_complete) noexcept {
    if (!request_pin || execution_frontier == 0 || prompt_tokens != execution_frontier) {
        return false;
    }
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

[[nodiscard]] constexpr bool
mtp_bridge_reads_rewrite_hidden(ninfer::PrefixReusePath path) noexcept {
    return is_rewrite_checkpoint_restore(path);
}

struct PrefillReuseSelection {
    ninfer::PrefixReusePath path = ninfer::PrefixReusePath::FullReset;
    std::uint32_t frontier       = 0;
};

// Current matching frontier always wins. Otherwise the longest complete rewrite or
// staged head. Same-F rewrite vs rollback/ladder keeps rewrite (`>` not `>=`).
[[nodiscard]] inline PrefillReuseSelection
select_resident_prefill_reuse(bool current_matches, std::uint32_t execution_frontier,
                              bool rewrite_matches, std::uint32_t rewrite_frontier,
                              ninfer::PrefixReusePath rewrite_path,
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

[[nodiscard]] inline PrefillReuseSelection
select_resident_prefill_reuse(bool current_matches, std::uint32_t execution_frontier,
                              bool rewrite_matches, std::uint32_t rewrite_frontier,
                              ninfer::PrefixReusePath rewrite_path,
                              std::span<const std::uint32_t> matching_ladder_frontiers) {
    std::vector<PrefillReuseHead> heads;
    heads.reserve(matching_ladder_frontiers.size());
    for (const std::uint32_t frontier : matching_ladder_frontiers) {
        heads.push_back(PrefillReuseHead{frontier, ContextCheckpointKind::Ladder});
    }
    return select_resident_prefill_reuse(current_matches, execution_frontier, rewrite_matches,
                                         rewrite_frontier, rewrite_path, heads);
}

// Resident checkpoint head reduced to the identity fields the reuse decision consumes.
struct ContextCheckpointRef {
    std::uint32_t frontier = 0;
    PrefixHash128 hash{};
    ContextCheckpointKind kind = ContextCheckpointKind::Ladder;
};

// Resident sequence state consumed by the reuse decision. This is the detail-level view of
// the runtime's richer ResidentStateView: the owner maps into it on every planning pass.
struct ResidentReuseState {
    const std::vector<TokenId>* ledger     = nullptr;
    const ResidentPrefixIdentity* identity = nullptr;
    std::uint32_t execution_frontier       = 0;
    bool rewrite_valid                     = false;
    RewriteCheckpointKind rewrite_kind     = RewriteCheckpointKind::TurnClosure;
    std::uint32_t rewrite_frontier         = 0;
    std::uint32_t mtp_kv_valid             = 0;
    std::uint32_t dflash_context_frontier  = 0;
    bool tail_hidden_valid                 = false;
    bool backend_image_present             = false;
    std::vector<ContextCheckpointRef> context_checkpoints;
};

// The restore path a rewrite checkpoint kind replays through.
[[nodiscard]] constexpr ninfer::PrefixReusePath
rewrite_restore_path(RewriteCheckpointKind kind) noexcept {
    return kind == RewriteCheckpointKind::TurnClosure
               ? ninfer::PrefixReusePath::RestoreTurnCheckpoint
               : ninfer::PrefixReusePath::RestoreResponseCheckpoint;
}

struct ReuseBackendPolicy {
    ninfer::SpeculativeBackend backend = ninfer::SpeculativeBackend::None;
    bool mtp_cache_present             = false;
    bool dflash_present                = false;
    bool dflash_full_layers            = false;
};

// Filter every candidate before ranking, including candidates in different RAM
// or disk entries. An unusable longer frontier cannot hide an earlier usable head.
[[nodiscard]] inline bool reuse_candidate_ready(const ResidentReuseState& state,
                                                ninfer::PrefixReusePath path,
                                                std::uint32_t frontier, std::size_t prompt_tokens,
                                                const ReuseBackendPolicy& policy) noexcept {
    if (path == ninfer::PrefixReusePath::AppendAtFrontier) {
        if (prompt_tokens == frontier && !state.tail_hidden_valid) { return false; }
        if (policy.backend == ninfer::SpeculativeBackend::DFlash &&
            state.dflash_context_frontier != frontier) {
            return false;
        }
    }
    if (policy.backend == ninfer::SpeculativeBackend::Mtp &&
        !mtp_prefix_reuse_ready(path, frontier, state.mtp_kv_valid, state.tail_hidden_valid,
                                policy.mtp_cache_present)) {
        return false;
    }
    if (is_rewrite_checkpoint_restore(path) &&
        policy.backend == ninfer::SpeculativeBackend::DFlash) {
        if (!policy.dflash_present || state.dflash_context_frontier < frontier) { return false; }
        if (policy.dflash_full_layers &&
            !dflash_rewrite_checkpoint_ready(state.backend_image_present,
                                             state.dflash_context_frontier, frontier)) {
            return false;
        }
    }
    return true;
}

// The resident prefix-reuse decision for one request. A matching execution frontier wins as
// an append, but each speculative backend gates the append on its continuation state before
// the checkpoint branch runs: DFlash needs its context at the frontier, MTP needs tail
// hidden + MTP KV there. An unready append falls through to a usable rewrite or staged
// checkpoint instead of forcing a FullReset later. Among the non-append candidates the
// longest matching head that the backend can legally continue wins. `prompt_hashes` is
// prefix_hash_chain(prompt), computed once per prompt by its owner; only checkpoint heads
// consult it, so it may be empty when the state has none.
[[nodiscard]] inline PrefillReuseSelection
decide_resident_reuse(const ResidentReuseState& state, const PreparedPromptData& prompt,
                      std::span<const PrefixHash128> prompt_hashes,
                      ninfer::SpeculativeBackend backend, bool mtp_cache_present,
                      bool dflash_present, bool dflash_full_layers) {
    if (!state.context_checkpoints.empty() && prompt_hashes.size() != prompt.token_ids.size() + 1) {
        throw std::invalid_argument("prompt prefix hash chain does not cover the prompt");
    }
    const ReuseBackendPolicy policy{backend, mtp_cache_present, dflash_present, dflash_full_layers};
    const auto ready = [&](ninfer::PrefixReusePath path, std::uint32_t frontier) {
        return reuse_candidate_ready(state, path, frontier, prompt.token_ids.size(), policy);
    };
    const bool current_matches =
        state.execution_frontier != 0 &&
        ready(ninfer::PrefixReusePath::AppendAtFrontier, state.execution_frontier) &&
        prefix_matches(prompt, *state.ledger, *state.identity, state.execution_frontier);
    bool rewrite_matches                 = false;
    ninfer::PrefixReusePath rewrite_path = ninfer::PrefixReusePath::FullReset;
    std::uint32_t rewrite_frontier       = 0;
    std::vector<PrefillReuseHead> matching_heads;
    if (!current_matches) {
        // The rewrite checkpoint is the resident prefix itself: an identical represented
        // prefix has the identical hash, so the prefix comparison alone decides it.
        if (state.rewrite_valid && state.rewrite_frontier != 0 &&
            prefix_matches(prompt, *state.ledger, *state.identity, state.rewrite_frontier) &&
            ready(rewrite_restore_path(state.rewrite_kind), state.rewrite_frontier)) {
            rewrite_matches  = true;
            rewrite_frontier = state.rewrite_frontier;
            rewrite_path     = rewrite_restore_path(state.rewrite_kind);
        }
        matching_heads.reserve(state.context_checkpoints.size());
        for (const ContextCheckpointRef& head : state.context_checkpoints) {
            if (head.frontier != 0 && head.frontier <= prompt.token_ids.size() &&
                prompt_hashes[head.frontier] == head.hash &&
                prefix_matches(prompt, *state.ledger, *state.identity, head.frontier) &&
                ready(reuse_path_for_context_checkpoint_kind(head.kind), head.frontier)) {
                matching_heads.push_back(PrefillReuseHead{head.frontier, head.kind});
            }
        }
    }
    return select_resident_prefill_reuse(current_matches, state.execution_frontier, rewrite_matches,
                                         rewrite_frontier, rewrite_path, matching_heads);
}

[[nodiscard]] constexpr bool
occupy_drops_rewrite_ahead_of_restore(ninfer::PrefixReusePath reuse, bool rewrite_valid,
                                      std::uint32_t rewrite_frontier,
                                      std::uint32_t restore_frontier) noexcept {
    return is_staged_checkpoint_restore(reuse) && rewrite_valid &&
           rewrite_frontier > restore_frontier;
}

[[nodiscard]] constexpr bool
occupy_clears_context_checkpoints(ninfer::PrefixReusePath reuse) noexcept {
    return reuse == ninfer::PrefixReusePath::FullReset;
}

[[nodiscard]] constexpr bool occupy_keeps_context_checkpoint_head(ninfer::PrefixReusePath reuse,
                                                                  std::uint32_t head_frontier,
                                                                  std::uint32_t new_base) noexcept {
    if (occupy_clears_context_checkpoints(reuse)) { return false; }
    if (reuse == ninfer::PrefixReusePath::AppendAtFrontier || is_staged_checkpoint_restore(reuse) ||
        is_rewrite_checkpoint_restore(reuse)) {
        return retain_context_checkpoint_head(head_frontier, new_base);
    }
    return false;
}

} // namespace ninfer::targets::qwen3_6::detail
