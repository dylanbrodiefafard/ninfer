#pragma once

#include "ninfer/types.h"
#include <memory>
#include <span>
#include <string_view>

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
    // Empty calls select reasoning-only recovery, with a system notice rather
    // than any invented assistant call or tool result.
    [[nodiscard]] PromptInput repair(std::span<const ToolCall> calls, std::uint32_t attempt) const;

private:
    PromptInput input_;
    ToolCall repeated_call_;
    std::string repeated_reasoning_;
};

} // namespace ninfer::targets::qwen3_6
