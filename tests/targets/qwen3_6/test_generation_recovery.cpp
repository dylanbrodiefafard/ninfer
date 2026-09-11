#include <ninfer/targets/qwen3_6/generation_recovery.h>
#include "runtime/contract/reasoning_recovery.h"
#include <iostream>

using namespace ninfer;
using ninfer::targets::qwen3_6::GenerationRecoveryContext;

int main() {
    std::string reasoning;
    for (int i = 0; i < 80; ++i) { reasoning += "step" + std::to_string(i) + " "; }
    PromptInput input;
    input.options.enable_thinking = true;
    input.options.tool_jsons.push_back(R"({"type":"function","function":{"name":"read"}})");
    ChatMessage user;
    user.role = ChatRole::User;
    user.parts.push_back(MessagePart{.kind = MessagePartKind::Text, .text = "Preserve my task."});
    input.messages.push_back(user);
    for (int i = 0; i < 2; ++i) {
        ChatMessage assistant;
        assistant.role = ChatRole::Assistant;
        assistant.reasoning_content = reasoning;
        assistant.tool_calls.push_back(ToolCall{.id = "call" + std::to_string(i), .name = "read",
            .arguments_json = i ? R"({"limit":100,"filePath":"a.cpp"})"
                                : R"({"filePath":"a.cpp","limit":100})"});
        input.messages.push_back(assistant);
        ChatMessage result;
        result.role = ChatRole::Tool;
        result.tool_call_id = assistant.tool_calls[0].id;
        result.parts.push_back(MessagePart{.kind = MessagePartKind::Text, .text = "unchanged file"});
        input.messages.push_back(result);
    }
    int failures = 0;
    auto check = [&](bool condition, const char* label) {
        if (!condition) { ++failures; std::cerr << label << '\n'; }
    };
    auto context = GenerationRecoveryContext::analyze(input);
    check(context && context->has_repeated_tool_history(),
          "confirmed unchanged repeated tool/reasoning was not detected");
    if (!context) { return 1; }
    const auto calls = input.messages[3].tool_calls;
    check(context->repeats(calls, reasoning), "canonical argument ordering changed duplicate identity");
    // A completed duplicate remains evidence, but explicit caller termination
    // wins over retry/exhaustion. Model EOS alone must still permit recovery.
    const std::vector<TokenId> generated{12, 13, 99};
    StopPolicy caller_stop;
    caller_stop.strings.push_back(StopString{.text = "STOP"});
    using ninfer::runtime::generation_recovery_allowed_at_finish;
    check(!generation_recovery_allowed_at_finish(FinishReason::StopString, generated, caller_stop),
          "completed duplicate bypassed the caller's string stop");
    check(generation_recovery_allowed_at_finish(FinishReason::StopToken, generated, caller_stop),
          "natural model EOS disabled duplicate-call recovery");
    caller_stop.token_ids = {99};
    check(!generation_recovery_allowed_at_finish(FinishReason::StopToken, generated, caller_stop),
          "completed duplicate bypassed an explicitly requested token stop");
    caller_stop.token_ids = {12};
    check(generation_recovery_allowed_at_finish(FinishReason::StopToken, generated, caller_stop),
          "a nonterminal historical token was mistaken for the caller's stop");
    check(!generation_recovery_allowed_at_finish(FinishReason::Cancelled, generated, caller_stop),
          "cancellation permitted a recovery attempt");
    check(generation_recovery_allowed_at_finish(FinishReason::OutputLimit, generated, caller_stop) &&
              generation_recovery_allowed_at_finish(FinishReason::ContextCapacity, generated, caller_stop),
          "a repeated passage at the budget boundary bypassed recovery exhaustion");
    check(!context->repeats(calls, "The user requested a fresh verification after changing the file."),
          "a repeated call without repeated reasoning triggered recovery");
    auto different_range = calls;
    different_range[0].arguments_json = R"({"filePath":"a.cpp","limit":100,"offset":101})";
    check(!context->repeats(different_range, reasoning), "reading a new range triggered recovery");
    auto changed = input;
    changed.messages.back().parts[0].text = "changed file";
    check(!GenerationRecoveryContext::analyze(changed)->has_repeated_tool_history(), "changed results were ignored");
    changed = input;
    changed.messages.insert(changed.messages.begin() + 3, user);
    check(!GenerationRecoveryContext::analyze(changed)->has_repeated_tool_history(), "intervening user instruction was ignored");
    changed = input;
    changed.messages[1].tool_calls[0].name = "edit";
    check(!GenerationRecoveryContext::analyze(changed)->has_repeated_tool_history(), "intervening different operation was ignored");
    changed = input;
    changed.messages.back().tool_call_id = "unrelated";
    check(!GenerationRecoveryContext::analyze(changed)->has_repeated_tool_history(), "unmatched result was treated as unchanged evidence");
    changed = input;
    changed.messages[1].reasoning_content = "Checking status.";
    check(!GenerationRecoveryContext::analyze(changed)->has_repeated_tool_history(), "a brief repeated poll triggered recovery");
    const auto repaired = context->repair(calls, 1);
    check(repaired.messages.size() == input.messages.size() + 2,
          "repair did not append the rejected proposal and feedback");
    check(repaired.messages.front().parts[0].text == "Preserve my task." &&
              repaired.messages[2].parts[0].text == "unchanged file" &&
              repaired.messages[4].parts[0].text == "unchanged file",
          "repair changed user instructions or real tool results");
    check(repaired.messages[1].reasoning_content.empty() &&
              repaired.messages[3].reasoning_content.empty() &&
              !input.messages[1].reasoning_content.empty(),
          "repair did not isolate removal of closed reasoning from the original input");
    check(repaired.messages.back().parts[0].text.find("did not execute") != std::string::npos &&
              repaired.messages.back().tool_call_id == repaired.messages[5].tool_calls[0].id,
          "repair fabricated execution or lost tool-result correspondence");
    const auto reasoning_repair = context->repair({}, 1);
    check(reasoning_repair.messages.size() == input.messages.size() + 1 &&
              reasoning_repair.messages.back().role == ChatRole::System &&
              reasoning_repair.messages.back().tool_calls.empty() &&
              reasoning_repair.messages.back().tool_call_id.empty(),
          "reasoning retry invented a call, tool result, or user message");
    for (std::size_t i = 0; i < input.messages.size(); ++i) {
        const auto& before = input.messages[i];
        const auto& after = reasoning_repair.messages[i];
        check(before.role == after.role && before.parts.size() == after.parts.size() &&
                  before.tool_call_id == after.tool_call_id &&
                  before.tool_calls.size() == after.tool_calls.size(), "retry changed message structure");
        for (std::size_t p = 0; p < before.parts.size(); ++p) {
            check(before.parts[p].text == after.parts[p].text, "retry changed task or actual tool result");
        }
        for (std::size_t c = 0; c < before.tool_calls.size(); ++c) {
            check(before.tool_calls[c].id == after.tool_calls[c].id &&
                      before.tool_calls[c].name == after.tool_calls[c].name &&
                      before.tool_calls[c].arguments_json == after.tool_calls[c].arguments_json,
                  "retry changed a historical call");
        }
    }
    PromptInput plain;
    plain.messages.push_back(user);
    plain.options.enable_thinking = true;
    const auto plain_context = GenerationRecoveryContext::analyze(plain);
    check(plain_context && !plain_context->has_repeated_tool_history(),
          "tool-free reasoning cannot recover");
    plain.options.enable_thinking = false;
    check(!GenerationRecoveryContext::analyze(plain), "non-thinking request retained recovery input");
    plain.options.enable_thinking = true;
    plain.messages[0].parts[0].kind = MessagePartKind::Media;
    check(!GenerationRecoveryContext::analyze(plain), "text-only repair admitted media");
    std::cout << "generation recovery failures=" << failures << '\n';
    return failures ? 1 : 0;
}
