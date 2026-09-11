#pragma once

#include "ninfer/types.h"
#include "targets/qwen3_6/impl/frontend/tokenizer.h"

#include <nlohmann/json.hpp>
#include <xgrammar/xgrammar.h>

#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::targets::qwen3_6 {

struct ToolGrammarData {
    struct Definition {
        std::string name;
        nlohmann::ordered_json parameters;
        std::optional<xgrammar::CompiledGrammar> arguments_validator;
    };
    std::vector<Definition> definitions;
    xgrammar::CompiledGrammar compiled;
    [[nodiscard]] std::optional<ToolCall> decode_call(std::string_view text) const;
};

namespace frontend_internal {

// The OutputSession owns this state. Draft-mask construction forks committed
// state and never advances it; only a committed publication advances grammar.
class ToolGrammarState {
public:
    static constexpr std::size_t mask_words = (248077 + 31) / 32;
    explicit ToolGrammarState(std::shared_ptr<const ToolGrammarData> grammar,
                              std::vector<TokenId> ignored_model_stops = {});
    void preview(std::span<const TokenId> tokens);
    void commit_preview() noexcept;
    void discard_preview() noexcept;

    // Node zero is the committed root; its token/parent are ignored. Each other
    // node consumes its draft token from an earlier parent. Output is node-major,
    // including the final bonus column. Unreachable nodes have an unrestricted
    // mask, not an empty sampling domain: their parent forbids their acceptance.
    void fill_masks(std::span<const TokenId> tokens,
                    std::span<const std::int32_t> parents,
                    std::span<std::uint32_t> words) const;

private:
    bool accept(xgrammar::GrammarMatcher& matcher, TokenId token) const;
    std::shared_ptr<const ToolGrammarData> grammar_;
    std::vector<TokenId> ignored_model_stops_;
    xgrammar::GrammarMatcher committed_;
    std::optional<xgrammar::GrammarMatcher> preview_;
};

// One compiler per resident frontend/tokenizer. Request grammars are immutable;
// mutable matching/publication state belongs to the request's OutputSession.
class ToolGrammarCompiler {
public:
    explicit ToolGrammarCompiler(std::shared_ptr<const Tokenizer> tokenizer);
    [[nodiscard]] std::shared_ptr<const ToolGrammarData>
    compile(std::span<const std::string> tools, bool starts_in_reasoning);

private:
    std::shared_ptr<const Tokenizer> tokenizer_;
    std::mutex mutex_;
    std::optional<xgrammar::GrammarCompiler> compiler_;
};

} // namespace frontend_internal
} // namespace ninfer::targets::qwen3_6
