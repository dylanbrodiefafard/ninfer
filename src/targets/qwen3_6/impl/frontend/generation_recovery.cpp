#include <ninfer/targets/qwen3_6/generation_recovery.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace ninfer::targets::qwen3_6 {
namespace {

bool same_call(const ToolCall& a, const ToolCall& b) {
    if (a.name != b.name) { return false; }
    const auto x = nlohmann::json::parse(a.arguments_json, nullptr, false);
    const auto y = nlohmann::json::parse(b.arguments_json, nullptr, false);
    return !x.is_discarded() && !y.is_discarded() && x == y;
}

bool same_result(const ChatMessage& a, const ChatMessage& b) {
    if (a.parts.size() != b.parts.size()) { return false; }
    for (std::size_t i = 0; i < a.parts.size(); ++i) {
        if (a.parts[i].kind != MessagePartKind::Text || b.parts[i].kind != MessagePartKind::Text ||
            a.parts[i].text != b.parts[i].text) { return false; }
    }
    return true;
}

std::vector<std::string_view> words(std::string_view text) {
    std::vector<std::string_view> result;
    std::size_t begin = 0;
    while (begin < text.size()) {
        while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin]))) { ++begin; }
        auto end = begin;
        while (end < text.size() && !std::isspace(static_cast<unsigned char>(text[end]))) { ++end; }
        if (end > begin) { result.push_back(text.substr(begin, end - begin)); }
        begin = end;
    }
    return result;
}

bool repeated_reasoning(std::string_view a, std::string_view b) {
    constexpr std::size_t window = 64;
    const auto x = words(a), y = words(b);
    if (x.size() < window || y.size() < window) { return false; }
    // Hashes locate candidates only; exact words establish the match.
    std::unordered_multimap<std::size_t, std::size_t> starts;
    for (std::size_t i = 0; i + window <= x.size(); ++i) {
        starts.emplace(std::hash<std::string_view>{}(x[i]), i);
    }
    for (std::size_t i = 0; i + window <= y.size(); ++i) {
        const auto [begin, end] = starts.equal_range(std::hash<std::string_view>{}(y[i]));
        for (auto match = begin; match != end; ++match) {
            if (std::equal(y.begin() + i, y.begin() + i + window, x.begin() + match->second)) {
                return true;
            }
        }
    }
    return false;
}

} // namespace

std::shared_ptr<const GenerationRecoveryContext> GenerationRecoveryContext::analyze(const PromptInput& input) {
    if (!input.options.enable_thinking || !input.options.add_generation_prompt) { return {}; }
    for (const auto& message : input.messages) {
        for (const auto& part : message.parts) {
            if (part.kind != MessagePartKind::Text) { return {}; }
        }
    }
    auto result = std::make_shared<GenerationRecoveryContext>();
    result->input_ = input;
    if (input.options.tool_jsons.empty() || input.messages.size() < 4) { return result; }
    const auto n = input.messages.size();
    const auto& first = input.messages[n - 4];
    const auto& first_result = input.messages[n - 3];
    const auto& second = input.messages[n - 2];
    const auto& second_result = input.messages[n - 1];
    if (first.role != ChatRole::Assistant || second.role != ChatRole::Assistant ||
        first_result.role != ChatRole::Tool || second_result.role != ChatRole::Tool ||
        first.tool_calls.size() != 1 || second.tool_calls.size() != 1 ||
        first_result.tool_call_id != first.tool_calls[0].id ||
        second_result.tool_call_id != second.tool_calls[0].id ||
        !same_call(first.tool_calls[0], second.tool_calls[0]) ||
        !same_result(first_result, second_result) ||
        !repeated_reasoning(first.reasoning_content, second.reasoning_content)) { return result; }
    result->repeated_call_ = second.tool_calls[0];
    result->repeated_reasoning_ = second.reasoning_content;
    return result;
}

bool GenerationRecoveryContext::repeats(std::span<const ToolCall> calls, std::string_view reasoning) const {
    return has_repeated_tool_history() && std::any_of(calls.begin(), calls.end(), [&](const ToolCall& call) {
        return same_call(call, repeated_call_);
    }) && repeated_reasoning(reasoning, repeated_reasoning_);
}

PromptInput GenerationRecoveryContext::repair(std::span<const ToolCall> calls, std::uint32_t attempt) const {
    if (attempt == 0 || attempt > maximum_attempts) {
        throw std::invalid_argument("invalid bounded generation recovery attempt");
    }
    PromptInput repaired = input_;
    for (auto& message : repaired.messages) {
        if (message.role == ChatRole::Assistant) { message.reasoning_content.clear(); }
    }
    if (calls.empty()) {
        ChatMessage notice;
        notice.role = ChatRole::System;
        notice.parts.push_back(MessagePart{.kind = MessagePartKind::Text,
            .text = "NInfer engine recovery notice, attempt " + std::to_string(attempt) +
                    ": the previous generation repeated its reasoning without reaching an answer "
                    "or tool call. That failed reasoning is omitted from this internal retry. "
                    "No tool was executed by this recovery. The original user task and actual tool "
                    "results above are unchanged. This is a continuation, not a fresh session. "
                    "Continue from the current task state instead of restarting initial orientation. "
                    "Repeat a completed inspection only to obtain specific new information. "
                    "Use the existing results to take a concrete next step: "
                    "produce the needed answer or issue a relevant tool call. Do not merely repeat "
                    "an intention to act. This notice is engine feedback, not a new user request."});
        repaired.messages.push_back(std::move(notice));
        return repaired;
    }
    ChatMessage candidate;
    candidate.role = ChatRole::Assistant;
    candidate.tool_calls.assign(calls.begin(), calls.end());
    for (std::size_t i = 0; i < candidate.tool_calls.size(); ++i) {
        candidate.tool_calls[i].id = "ninfer_rejected_" + std::to_string(attempt) + "_" + std::to_string(i);
    }
    repaired.messages.push_back(std::move(candidate));
    const auto& proposed = repaired.messages.back().tool_calls;
    std::vector<ChatMessage> feedback;
    for (const auto& call : proposed) {
        ChatMessage message;
        message.role = ChatRole::Tool;
        message.tool_call_id = call.id;
        message.parts.push_back(MessagePart{.kind = MessagePartKind::Text,
            .text = "NInfer did not execute this proposed call. It detected repeated reasoning and "
                    "the same completed call twice with unchanged results. This is engine recovery "
                    "feedback, not a new tool result. Use the results already in the conversation "
                    "and take a substantive next step toward the user's task. Changing only a "
                    "read limit or rephrasing the same operation is not progress. If a repeat is "
                    "genuinely necessary, identify what changed or what new information it obtains."});
        feedback.push_back(std::move(message));
    }
    for (auto& message : feedback) { repaired.messages.push_back(std::move(message)); }
    return repaired;
}

} // namespace ninfer::targets::qwen3_6
