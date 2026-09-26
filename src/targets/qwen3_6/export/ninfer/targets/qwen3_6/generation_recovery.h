#pragma once

#include "ninfer/types.h"
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace ninfer::targets::qwen3_6 {

// Captures text-only thinking input for bounded internal regeneration.
// Tool evidence: two consecutive single-call rounds, identical parsed
// arguments and tool results, and a substantial repeated reasoning passage.
// An intervening user turn, different call/result, or media disables this proof.
// It is not a claim to infer arbitrary external tool side effects.
class GenerationRecoveryContext {
public:
    static constexpr std::uint32_t maximum_attempts = 2;
    [[nodiscard]] static std::shared_ptr<const GenerationRecoveryContext> analyze(const PromptInput& input);
    [[nodiscard]] bool has_repeated_tool_history() const noexcept { return !repeated_call_.name.empty(); }
    [[nodiscard]] bool repeats(std::span<const ToolCall> calls, std::string_view reasoning) const;
    [[nodiscard]] const PromptInput& input() const noexcept { return input_; }
    // Only the new messages. Empty calls are one system notice. Calls are one
    // assistant message plus one tool message per call. History is not copied
    // and reasoning_content is not cleared.
    [[nodiscard]] std::vector<ChatMessage> recovery_insert(std::span<const ToolCall> calls,
                                                           std::uint32_t attempt) const;

private:
    PromptInput input_;
    ToolCall repeated_call_;
    std::string repeated_reasoning_;
};

// Open generation that recovery is allowed to close, and the empty-think close
// that finishes it. The prologue is not repeated inside the close.
inline constexpr std::string_view kRecoveryThinkingPrologue = "<|im_start|>assistant\n<think>\n";
inline constexpr std::string_view kRecoveryTurnClose        = "\n</think>\n\n<|im_end|>\n";

inline constexpr std::string_view kRecoveryPrologueExhausted =
    "open generation is not the thinking prologue";
inline constexpr std::string_view kRecoveryBudgetExhausted =
    "repaired request cannot preserve its admitted output budget";
inline constexpr std::string_view kRecoveryLaneExhausted =
    "reserved lane cannot be rebuilt safely";

// False when any piece is empty, the prefix does not end with the prologue ids,
// or a piece's decoded text differs from the text that was encoded.
[[nodiscard]] bool recovery_suffix_tokens_ok(std::span<const TokenId> prefix,
                                              std::span<const TokenId> prologue,
                                              std::span<const TokenId> turn_close,
                                              std::span<const TokenId> insert,
                                              std::string_view prologue_text,
                                              std::string_view prologue_decoded,
                                              std::string_view turn_close_text,
                                              std::string_view turn_close_decoded,
                                              std::string_view insert_text,
                                              std::string_view insert_decoded) noexcept;

// False when the spliced prompt cannot keep `remaining` output tokens inside `capacity`.
[[nodiscard]] bool recovery_output_budget_preserved(std::uint32_t spliced, std::uint32_t capacity,
                                                    std::uint32_t remaining) noexcept;

enum class RecoveryPrefillRoute : std::uint8_t {
    ResidentSuffix,
    HostRam,
    HostDisk,
    Cold,
    Exhaust,
};

struct RecoveryPrefillDecision {
    RecoveryPrefillRoute route = RecoveryPrefillRoute::Cold;
    std::string_view detail{};
};

struct RecoveryPrefillInput {
    bool splice_accepted                 = false;
    bool allow_prefix_reuse              = false;
    bool force_cold_prefill              = false;
    bool lane_retained                   = false;
    std::uint32_t resident_reusable_tokens = 0;
    bool can_admit_lane                  = false;
    bool output_budget_preserved         = false;
    bool pages_fit                       = false;
    std::uint32_t ram_reusable_tokens    = 0;
    std::uint32_t disk_reusable_tokens   = 0;
    bool host_restore_failed             = false;
};

// The only recovery prefill branch. A positive resident reuse does not consult
// the host counts. Host counts are used only after the resident plan reused nothing.
[[nodiscard]] RecoveryPrefillDecision route_recovery_prefill(const RecoveryPrefillInput& input) noexcept;

// The splice prefix. A non-empty resident ledger copy wins. The original prompt
// is the prefix only when that copy is empty.
[[nodiscard]] std::span<const TokenId> recovery_splice_prefix(
    std::span<const TokenId> resident_prefix, std::span<const TokenId> original_prompt) noexcept;

} // namespace ninfer::targets::qwen3_6
