#pragma once

#include "ninfer/types.h"
#include "runtime/contract/types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace ninfer::targets::qwen3_6 {

inline constexpr std::size_t kTokenDomain = 248077;

struct FrontendResources;
struct PreparedPromptData;
class Frontend;
class FrontendTestAccess;
class PreparedPromptAccess;
class EncodedHistoryPrepare;
class GenerationRecoveryContext;

class PreparedPrompt {
public:
    PreparedPrompt() noexcept;
    ~PreparedPrompt();
    PreparedPrompt(PreparedPrompt&&) noexcept;
    PreparedPrompt& operator=(PreparedPrompt&&) noexcept;

    PreparedPrompt(const PreparedPrompt&)            = delete;
    PreparedPrompt& operator=(const PreparedPrompt&) = delete;

    [[nodiscard]] PromptSummary summary() const;
    [[nodiscard]] double prepare_seconds() const noexcept;
    [[nodiscard]] explicit operator bool() const noexcept;

private:
    explicit PreparedPrompt(std::unique_ptr<PreparedPromptData> data) noexcept;
    std::unique_ptr<PreparedPromptData> data_;

    friend class Frontend;
    friend class FrontendTestAccess;
    friend class PreparedPromptAccess;
    friend class EncodedHistoryPrepare;
};

class PublishedOutput {
public:
    using iterator       = std::array<OutputDelta, 2>::iterator;
    using const_iterator = std::array<OutputDelta, 2>::const_iterator;

    PublishedOutput()                                  = default;
    PublishedOutput(const PublishedOutput&)            = default;
    PublishedOutput& operator=(const PublishedOutput&) = default;
    PublishedOutput(PublishedOutput&& other) noexcept;
    PublishedOutput& operator=(PublishedOutput&& other) noexcept;

    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

    [[nodiscard]] std::size_t size() const noexcept { return size_; }

    [[nodiscard]] iterator begin() noexcept { return values_.begin(); }

    [[nodiscard]] const_iterator begin() const noexcept { return values_.begin(); }

    [[nodiscard]] iterator end() noexcept { return values_.begin() + size_; }

    [[nodiscard]] const_iterator end() const noexcept { return values_.begin() + size_; }

    [[nodiscard]] OutputDelta& back() noexcept { return values_[size_ - 1]; }

    [[nodiscard]] const OutputDelta& back() const noexcept { return values_[size_ - 1]; }

    void clear() noexcept;
    void push_back(OutputDelta value);

private:
    std::array<OutputDelta, 2> values_{};
    std::size_t size_ = 0;
};

class OutputSession {
public:
    OutputSession() noexcept;
    ~OutputSession();
    OutputSession(OutputSession&&) noexcept;
    OutputSession& operator=(OutputSession&&) noexcept;

    OutputSession(const OutputSession&)            = delete;
    OutputSession& operator=(const OutputSession&) = delete;

    [[nodiscard]] runtime::OutputDecision preview(std::span<const TokenId> tokens,
                                                  std::uint32_t budget_remaining,
                                                  FinishReason limit_reason);
    [[nodiscard]] runtime::OutputDecision preview_terminal(FinishReason reason);
    [[nodiscard]] PublishedOutput commit_preview() noexcept;
    void discard_preview() noexcept;
    [[nodiscard]] std::uint32_t reasoning_tokens() const noexcept;
    [[nodiscard]] bool in_reasoning() const noexcept;
    [[nodiscard]] bool model_stop_tokens_allowed() const noexcept;
    [[nodiscard]] bool reasoning_cycle_exclusion_allowed(TokenId token) const;
    [[nodiscard]] bool has_tool_grammar() const noexcept;
    [[nodiscard]] std::span<const ToolCall> tool_calls() const noexcept;
    [[nodiscard]] std::shared_ptr<const GenerationRecoveryContext> generation_recovery_context() const noexcept;
    [[nodiscard]] bool terminal() const noexcept;
    // Read-only snapshot of committed grammar, node-major. Node zero is the
    // committed root; other nodes consume their token from an earlier parent.
    void fill_tool_masks(std::span<const TokenId> tokens,
                         std::span<const std::int32_t> parents,
                         std::span<std::uint32_t> words) const;

private:
    class Impl;
    explicit OutputSession(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend class Frontend;
};

class Frontend {
public:
    Frontend(const Frontend&);
    Frontend& operator=(const Frontend&);
    Frontend(Frontend&&) noexcept;
    Frontend& operator=(Frontend&&) noexcept;
    ~Frontend();

    [[nodiscard]] PreparedPrompt prepare(PromptInput input) const;
    [[nodiscard]] std::uint32_t count_tokens(PromptInput input) const;
    [[nodiscard]] PreparedPrompt prepare_tokens(std::vector<TokenId> token_ids,
                                                bool allow_prefix_identity = true) const;
    [[nodiscard]] PromptCapabilities prompt_capabilities() const noexcept;
    [[nodiscard]] OutputSession make_output_session(const PreparedPrompt& prompt,
                                                    const StopPolicy& caller_stop,
                                                    const OutputOptions& output = {}) const;
    [[nodiscard]] const StopPolicy& default_stop_policy() const noexcept;

private:
    class Impl;
    explicit Frontend(std::shared_ptr<const Impl> impl) noexcept;
    std::shared_ptr<const Impl> impl_;

    friend class FrontendTestAccess;
    friend class EncodedHistoryPrepare;
    friend Frontend make_frontend(const FrontendResources& resources, bool vision_enabled);
};

[[nodiscard]] Frontend make_frontend(const FrontendResources& resources, bool vision_enabled);

// Diagnostic only for requests without declared tools. Recognizes complete
// envelopes but never produces executable calls or guesses argument types.
[[nodiscard]] std::vector<std::string>
unconstrained_tool_call_names(std::string_view text, std::size_t max_name_length);

} // namespace ninfer::targets::qwen3_6
