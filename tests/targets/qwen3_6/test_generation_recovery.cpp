#include <ninfer/targets/qwen3_6/generation_recovery.h>
#include "runtime/contract/reasoning_recovery.h"
#include <algorithm>
#include <iostream>
#include <span>
#include <stdexcept>
#include <vector>

using namespace ninfer;
using ninfer::targets::qwen3_6::GenerationRecoveryContext;
using ninfer::targets::qwen3_6::RecoveryPrefillInput;
using ninfer::targets::qwen3_6::RecoveryPrefillRoute;
using ninfer::targets::qwen3_6::kRecoveryBudgetExhausted;
using ninfer::targets::qwen3_6::kRecoveryLaneExhausted;
using ninfer::targets::qwen3_6::kRecoveryPrologueExhausted;
using ninfer::targets::qwen3_6::recovery_output_budget_preserved;
using ninfer::targets::qwen3_6::recovery_splice_prefix;
using ninfer::targets::qwen3_6::recovery_suffix_tokens_ok;
using ninfer::targets::qwen3_6::route_recovery_prefill;

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
    const auto& stored = context->input();
    check(stored.messages[0].parts[0].text == "Preserve my task." &&
              stored.messages[1].reasoning_content == reasoning &&
              stored.messages[2].parts[0].text == "unchanged file" &&
              stored.messages[4].parts[0].text == "unchanged file",
          "recovery insert rewrote stored user text, reasoning, or tool results");
    const auto notice = context->recovery_insert({}, 1);
    const auto second_notice = context->recovery_insert({}, 2);
    check(notice.size() == 1 && notice[0].role == ChatRole::System &&
              notice[0].tool_calls.empty() && notice[0].tool_call_id.empty() &&
              notice[0].parts[0].text.find("attempt 1") != std::string::npos &&
              notice[0].parts[0].text.find("failed reasoning is omitted") != std::string::npos,
          "reasoning retry did not return one system notice for this attempt");
    check(second_notice.size() == 1 &&
              second_notice[0].parts[0].text.find("attempt 2") != std::string::npos &&
              second_notice[0].parts[0].text != notice[0].parts[0].text,
          "attempt 2 did not stack a distinct system notice");
    check(notice[0].role != ChatRole::User && stored.messages.size() > notice.size(),
          "history was a prefix of the reasoning insert");
    std::vector<ToolCall> proposed = {
        {.id = "old0", .name = "read", .arguments_json = R"({"a":1})"},
        {.id = "old1", .name = "read", .arguments_json = R"({"b":2})"},
    };
    const auto rejected = context->recovery_insert(proposed, 2);
    check(rejected.size() == 3 && rejected[0].role == ChatRole::Assistant &&
              rejected[0].reasoning_content.empty() && rejected[0].parts.empty() &&
              rejected[0].tool_calls.size() == 2 &&
              rejected[0].tool_calls[0].id == "ninfer_rejected_2_0" &&
              rejected[0].tool_calls[1].id == "ninfer_rejected_2_1" &&
              rejected[1].role == ChatRole::Tool &&
              rejected[1].tool_call_id == "ninfer_rejected_2_0" &&
              rejected[2].role == ChatRole::Tool &&
              rejected[2].tool_call_id == "ninfer_rejected_2_1" &&
              rejected[1].parts[0].text.find("did not execute") != std::string::npos &&
              rejected[2].parts[0].text.find("did not execute") != std::string::npos,
          "two rejected calls did not become one empty assistant plus matching tool feedback");
    check(proposed[0].id == "old0" && proposed[1].id == "old1",
          "recovery insert rewrote the caller's tool-call ids");
    for (const auto& message : rejected) {
        check(message.role != ChatRole::User, "a recovery insert used the user role");
    }
    check(stored.messages.size() > rejected.size() &&
              stored.messages[0].role != rejected[0].role,
          "history was a prefix of the rejected-call insert");
    auto throws_attempt = [&](std::uint32_t attempt) {
        try {
            (void)context->recovery_insert({}, attempt);
        } catch (const std::invalid_argument&) { return true; }
        return false;
    };
    check(throws_attempt(0) && throws_attempt(3), "attempt 0 or 3 was accepted");
    check(stored.messages[1].reasoning_content == reasoning &&
              stored.messages[0].parts[0].text == "Preserve my task." &&
              stored.messages[4].parts[0].text == "unchanged file",
          "a rejected insert cleared stored reasoning or tool-result text");
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
    const std::vector<TokenId> prologue{4, 5, 6};
    const std::vector<TokenId> prefix{1, 2, 3, 4, 5, 6};
    const std::vector<TokenId> close{7, 8};
    const std::vector<TokenId> insert_ids{9, 9};
    check(recovery_suffix_tokens_ok(prefix, prologue, close, insert_ids, "prologue", "prologue",
                                    "close", "close", "insert", "insert"),
          "a prologue-terminated prefix with matching decodes was refused");
    std::vector<TokenId> spliced_ids = prefix;
    spliced_ids.insert(spliced_ids.end(), close.begin(), close.end());
    spliced_ids.insert(spliced_ids.end(), insert_ids.begin(), insert_ids.end());
    spliced_ids.insert(spliced_ids.end(), prologue.begin(), prologue.end());
    const std::vector<TokenId> expected_suffix{1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 4, 5, 6};
    check(spliced_ids == expected_suffix, "the suffix concat did not copy the prologue ids");
    const std::vector<TokenId> empty_ids;
    check(!recovery_suffix_tokens_ok(std::vector<TokenId>{1, 2, 3, 4, 5, 9}, prologue, close,
                                    insert_ids, "prologue", "prologue", "close", "close", "insert",
                                    "insert"),
          "a prefix that does not end in the prologue was accepted");
    check(!recovery_suffix_tokens_ok(prefix, prologue, empty_ids, insert_ids, "prologue", "prologue",
                                    "close", "close", "insert", "insert"),
          "an empty close was accepted");
    check(!recovery_suffix_tokens_ok(prefix, prologue, close, empty_ids, "prologue", "prologue", "close",
                                    "close", "insert", "insert"),
          "an empty insert was accepted");
    check(!recovery_suffix_tokens_ok(prefix, empty_ids, close, insert_ids, "prologue", "prologue", "close",
                                    "close", "insert", "insert"),
          "an empty prologue was accepted");
    check(!recovery_suffix_tokens_ok(prefix, prologue, close, insert_ids, "prologue", "prologuX",
                                    "close", "close", "insert", "insert"),
          "a prologue decode that differs by one byte was accepted");
    check(!recovery_suffix_tokens_ok(prefix, prologue, close, insert_ids, "prologue", "prologue",
                                    "close", "closX", "insert", "insert"),
          "a close decode that differs by one byte was accepted");
    check(!recovery_suffix_tokens_ok(prefix, prologue, close, insert_ids, "prologue", "prologue",
                                    "close", "close", "insert", "inserX"),
          "an insert decode that differs by one byte was accepted");

    const std::vector<TokenId> original_prompt{10, 11, 12};
    const std::vector<TokenId> stacked_prompt{10, 11, 12, 20, 21, 10, 11, 12};
    const auto same_ids = [](std::span<const TokenId> got, const std::vector<TokenId>& expected) {
        return got.size() == expected.size() &&
               std::equal(expected.begin(), expected.end(), got.begin());
    };
    check(same_ids(recovery_splice_prefix(stacked_prompt, original_prompt), stacked_prompt),
          "a held ledger was replaced by the original prompt");
    check(same_ids(recovery_splice_prefix({}, original_prompt), original_prompt),
          "a missing ledger was not replaced by the original prompt");

    RecoveryPrefillInput resident;
    resident.splice_accepted          = true;
    resident.allow_prefix_reuse       = true;
    resident.lane_retained            = true;
    resident.resident_reusable_tokens = 1000;
    resident.can_admit_lane           = true;
    resident.output_budget_preserved  = true;
    resident.pages_fit                = true;
    resident.ram_reusable_tokens      = 5000;
    resident.disk_reusable_tokens     = 9000;
    const auto route = [](RecoveryPrefillInput input) { return route_recovery_prefill(input); };
    auto refused = resident;
    refused.splice_accepted = false;
    const auto refused_decision = route(refused);
    check(refused_decision.route == RecoveryPrefillRoute::Exhaust &&
              refused_decision.detail == kRecoveryPrologueExhausted,
          "a refused splice did not exhaust as a missing thinking prologue");
    auto over_budget = resident;
    over_budget.output_budget_preserved = false;
    auto over_pages = resident;
    over_pages.pages_fit = false;
    check(route(over_budget).route == RecoveryPrefillRoute::Exhaust &&
              route(over_budget).detail == kRecoveryBudgetExhausted &&
              route(over_pages).route == RecoveryPrefillRoute::Exhaust &&
              route(over_pages).detail == kRecoveryBudgetExhausted,
          "a lost output budget or page reservation did not exhaust");
    auto blocked = resident;
    blocked.can_admit_lane = false;
    check(route(blocked).route == RecoveryPrefillRoute::Exhaust &&
              route(blocked).detail == kRecoveryLaneExhausted,
          "a lane that cannot be admitted did not exhaust");
    auto forced = resident;
    forced.force_cold_prefill = true;
    auto disabled = resident;
    disabled.allow_prefix_reuse = false;
    check(route(forced).route == RecoveryPrefillRoute::Cold &&
              route(disabled).route == RecoveryPrefillRoute::Cold,
          "forced or disabled prefix reuse did not stay cold");
    auto blocked_and_forced = blocked;
    blocked_and_forced.force_cold_prefill = true;
    check(route(blocked_and_forced).route == RecoveryPrefillRoute::Cold,
          "forced cold prefill was exhausted by a lane that cannot be admitted");
    check(route(resident).route == RecoveryPrefillRoute::ResidentSuffix,
          "a retained resident prefix consulted the larger host tiers");
    auto host_disk = resident;
    host_disk.resident_reusable_tokens = 0;
    host_disk.ram_reusable_tokens      = 10;
    host_disk.disk_reusable_tokens     = 50;
    check(route(host_disk).route == RecoveryPrefillRoute::HostDisk,
          "a longer disk checkpoint did not win the resident miss");
    auto host_ram = host_disk;
    host_ram.ram_reusable_tokens  = 50;
    host_ram.disk_reusable_tokens = 50;
    check(route(host_ram).route == RecoveryPrefillRoute::HostRam,
          "an equal host checkpoint did not prefer RAM");
    auto neither = host_disk;
    neither.ram_reusable_tokens  = 0;
    neither.disk_reusable_tokens = 0;
    check(route(neither).route == RecoveryPrefillRoute::Cold,
          "a resident miss with no host checkpoint did not cold-prefill");
    auto stale = resident;
    stale.lane_retained            = false;
    stale.resident_reusable_tokens = 1000;
    stale.ram_reusable_tokens      = 10;
    stale.disk_reusable_tokens     = 50;
    check(route(stale).route == RecoveryPrefillRoute::HostDisk,
          "a stale resident count on an unretained lane hid the host checkpoint");
    auto restored = host_disk;
    restored.host_restore_failed = true;
    check(route(restored).route == RecoveryPrefillRoute::Cold && route(restored).detail.empty(),
          "a failed host restore did not cold-prefill");

    check(recovery_output_budget_preserved(8, 8, 1) &&
              !recovery_output_budget_preserved(8, 8, 2) &&
              !recovery_output_budget_preserved(9, 8, 1),
          "a spliced prompt at context capacity did not keep exactly one output token");
    check(recovery_output_budget_preserved(140, 200, 50) &&
              !recovery_output_budget_preserved(160, 200, 50),
          "the spliced length did not replace admitted-plus-generated accounting");
    std::cout << "generation recovery failures=" << failures << '\n';
    return failures ? 1 : 0;
}
