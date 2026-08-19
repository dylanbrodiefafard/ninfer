#include "ninfer/engine.h"
#include "serve/request.h"
#include "serve/translate.h"

#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace ninfer::serve;

int fail(const std::string& message) {
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

int check(bool condition, const std::string& message) { return condition ? 0 : fail(message); }

const char* kPrepend = "Server policy: answer in one sentence.";

ninfer::EngineOptions engine_options(const char* artifact, std::size_t ram_bytes = 0) {
    ninfer::EngineOptions options;
    options.artifact_path         = artifact;
    options.max_context           = 4096;
    options.kv_capacity           = ninfer::KvCapacityPolicy::explicit_capacity(4096);
    options.prefill_chunk         = 1024;
    options.kv_ram_capacity_bytes = ram_bytes;
    options.enable_vision         = false;
    options.speculative.backend   = ninfer::SpeculativeBackend::None;
    return options;
}

ninfer::OwnedMedia unused_media(const ContentPart&) {
    throw std::logic_error("media acquisition was not expected");
}

ResolvedPromptSemantics prompt_semantics() {
    ResolvedPromptSemantics semantics;
    semantics.enable_thinking   = true;
    semantics.preserve_thinking = false;
    semantics.reasoning_effort  = std::nullopt;
    return semantics;
}

ninfer::RequestOptions request_options() {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = 8;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = true;
    options.stop.include_model_defaults       = false;
    return options;
}

GenerationRequest
request_from_turns(std::initializer_list<std::pair<ninfer::ChatRole, std::string>> turns) {
    GenerationRequest request;
    for (const auto& turn_spec : turns) {
        ChatTurn turn;
        turn.role = turn_spec.first;
        ContentPart part;
        part.kind = ContentKind::Text;
        part.text = turn_spec.second;
        turn.content.push_back(std::move(part));
        request.messages.push_back(std::move(turn));
    }
    return request;
}

std::string joined_text(const ninfer::ChatMessage& message) {
    std::string text;
    for (const ninfer::MessagePart& part : message.parts) {
        if (part.kind == ninfer::MessagePartKind::Text) { text += part.text; }
    }
    return text;
}

void attach_weather_tool(GenerationRequest& request) {
    ToolDefinition tool;
    tool.name            = "get_weather";
    tool.definition_json = R"({"type":"function","function":{"name":"get_weather","description":"Fetch weather","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}})";
    request.tools.push_back(std::move(tool));
}

GenerationRequest weather_user_request() {
    GenerationRequest request = request_from_turns({{ninfer::ChatRole::User, "weather?"}});
    attach_weather_tool(request);
    return request;
}

GenerationRequest weather_tool_loop_request() {
    GenerationRequest request = request_from_turns({
        {ninfer::ChatRole::User, "weather?"},
        {ninfer::ChatRole::Assistant, ""},
        {ninfer::ChatRole::Tool, R"({"temp":20})"},
    });
    request.messages[1].tool_calls.push_back(
        {.id = "call_1", .name = "get_weather", .arguments_json = R"({"city":"Paris"})"});
    request.messages[2].tool_call_id = "call_1";
    attach_weather_tool(request);
    return request;
}

ninfer::PromptInput translate(const GenerationRequest& request, std::string_view prepend) {
    return to_prompt_input(request, prompt_semantics(), unused_media, prepend);
}

int require_eight_tokens(const ninfer::GenerationResult& result, const char* label) {
    if (result.generated_token_ids.size() >= 8) { return 0; }
    std::cerr << "FAIL: " << label << " generated " << result.generated_token_ids.size()
              << " tokens, expected at least 8\n";
    return 1;
}

int exercise_artifact(const char* artifact) {
    ninfer::Engine engine(engine_options(artifact));
    const ninfer::RequestOptions options = request_options();
    int failures                         = 0;

    const GenerationRequest hello          = request_from_turns({{ninfer::ChatRole::User, "hello"}});
    const ninfer::PromptInput with_prepend = translate(hello, kPrepend);
    const ninfer::PromptInput without      = translate(hello, {});
    const std::uint32_t count_with         = engine.count_tokens(with_prepend);
    const std::uint32_t count_without      = engine.count_tokens(without);
    failures += check(count_with > count_without,
                      "prepended prompt is not longer than the unprepended prompt");
    const ninfer::PreparedPrompt prepared = engine.prepare(with_prepend);
    failures += check(count_with == prepared.summary().prompt_tokens,
                      "count_tokens does not match prepare prompt_tokens with prepend");
    if (failures != 0) { return failures; }

    const ninfer::PromptInput turn1 =
        translate(request_from_turns({{ninfer::ChatRole::User, "q1"}}), kPrepend);
    const ninfer::GenerationResult first = engine.generate(engine.prepare(turn1), options);
    failures += require_eight_tokens(first, "turn 1");
    if (failures != 0) { return failures; }

    GenerationRequest turn2_req = request_from_turns({
        {ninfer::ChatRole::User, "q1"},
        {ninfer::ChatRole::Assistant, first.content},
        {ninfer::ChatRole::User, "q2"},
    });
    turn2_req.messages[1].reasoning_content = first.reasoning;
    const ninfer::PromptInput turn2         = translate(turn2_req, kPrepend);
    const ninfer::GenerationResult second   = engine.generate(engine.prepare(turn2), options);
    failures += require_eight_tokens(second, "turn 2");
    failures += check(second.prefix_reuse_source == ninfer::PrefixReuseSource::VramResident,
                      "turn 2 did not reuse the prepended prefix from VRAM");
    failures += check(second.reused_prompt_tokens > 0, "turn 2 did not reuse a prepended prefix");
    ninfer::PromptInput closed               = turn1;
    closed.options.add_generation_prompt     = false;
    const std::uint32_t closed_tokens        = engine.count_tokens(closed);
    failures += check(second.reused_prompt_tokens >= closed_tokens,
                      "turn 2 reused fewer tokens than the closed turn-1 prompt");
    failures += check(turn2.messages[0].role == ninfer::ChatRole::System &&
                          joined_text(turn2.messages[0]).starts_with(kPrepend),
                      "turn 2 leading message is not the prepended System");
    if (failures != 0) { return failures; }

    const ninfer::PromptInput miss        = translate(turn2_req, {});
    const ninfer::GenerationResult missed = engine.generate(engine.prepare(miss), options);
    failures += require_eight_tokens(missed, "miss control");
    failures +=
        check(missed.reused_prompt_tokens == 0, "empty-prepend turn 2 unexpectedly reused a prefix");
    if (failures != 0) { return failures; }

    const ninfer::PromptInput tool_turn1 = translate(weather_user_request(), kPrepend);
    failures += check(!tool_turn1.options.tool_jsons.empty(), "tool turn 1 dropped tool_jsons");
    failures += check(tool_turn1.messages[0].role == ninfer::ChatRole::System &&
                          joined_text(tool_turn1.messages[0]).starts_with(kPrepend),
                      "tool turn 1 leading message is not the prepended System");
    const ninfer::GenerationResult tool_first =
        engine.generate(engine.prepare(tool_turn1), options);
    failures += require_eight_tokens(tool_first, "tool turn 1");
    if (failures != 0) { return failures; }

    const ninfer::PromptInput tool_turn2 = translate(weather_tool_loop_request(), kPrepend);
    failures += check(!tool_turn2.options.tool_jsons.empty(), "tool turn 2 dropped tool_jsons");
    failures += check(tool_turn2.messages[2].tool_calls.size() == 1,
                      "tool turn 2 lost the assistant tool call");
    const ninfer::GenerationResult tool_second =
        engine.generate(engine.prepare(tool_turn2), options);
    failures += require_eight_tokens(tool_second, "tool turn 2");
    failures +=
        check(tool_second.reused_prompt_tokens > 0, "tool turn 2 did not reuse a prepended prefix");
    ninfer::PromptInput tool_closed           = tool_turn1;
    tool_closed.options.add_generation_prompt = false;
    const std::uint32_t tool_closed_tokens    = engine.count_tokens(tool_closed);
    if (tool_second.reused_prompt_tokens < tool_closed_tokens) {
        std::cerr << "FAIL: tool turn 2 reused " << tool_second.reused_prompt_tokens
                  << " tokens, expected at least " << tool_closed_tokens << '\n';
        ++failures;
    }
    if (failures != 0) { return failures; }

    const ninfer::PromptInput tool_miss = translate(weather_tool_loop_request(), {});
    const ninfer::GenerationResult tool_missed =
        engine.generate(engine.prepare(tool_miss), options);
    failures += require_eight_tokens(tool_missed, "tool miss control");
    failures += check(tool_missed.reused_prompt_tokens == 0,
                      "empty-prepend tool turn 2 unexpectedly reused a prefix");
    return failures;
}

int exercise_ram_artifact(const char* artifact) {
    constexpr std::size_t kRamBytes = 1024ULL * 1024ULL * 1024ULL;
    ninfer::Engine engine(engine_options(artifact, kRamBytes));
    if (engine.memory_summary().kv_ram_capacity_bytes != kRamBytes) {
        return fail("RAM-tier prepend engine did not enable host KV capacity");
    }
    const ninfer::RequestOptions options = request_options();

    const ninfer::PromptInput turn1 =
        translate(request_from_turns({{ninfer::ChatRole::User, "q1"}}), kPrepend);
    const ninfer::GenerationResult first = engine.generate(engine.prepare(turn1), options);
    if (const int rc = require_eight_tokens(first, "RAM turn 1"); rc != 0) { return rc; }

    const auto captures_before = engine.runtime_stats().kv_ram_captures;
    const ninfer::PromptInput other =
        translate(request_from_turns({{ninfer::ChatRole::User, "unrelated"}}), kPrepend);
    const ninfer::GenerationResult evictor = engine.generate(engine.prepare(other), options);
    if (const int rc = require_eight_tokens(evictor, "RAM evictor"); rc != 0) { return rc; }
    if (evictor.prefix_reuse_path != ninfer::PrefixReusePath::FullReset ||
        evictor.prefix_reuse_source != ninfer::PrefixReuseSource::None) {
        return fail("RAM evictor reused the prepended first chat instead of spilling it");
    }
    if (engine.runtime_stats().kv_ram_captures <= captures_before) {
        return fail("RAM evictor did not capture the prepended first chat");
    }

    GenerationRequest turn2_req = request_from_turns({
        {ninfer::ChatRole::User, "q1"},
        {ninfer::ChatRole::Assistant, first.content},
        {ninfer::ChatRole::User, "q2"},
    });
    turn2_req.messages[1].reasoning_content = first.reasoning;
    const ninfer::PromptInput turn2         = translate(turn2_req, kPrepend);
    const auto restores_before              = engine.runtime_stats().kv_ram_restores;
    const ninfer::GenerationResult second   = engine.generate(engine.prepare(turn2), options);
    if (const int rc = require_eight_tokens(second, "RAM turn 2"); rc != 0) { return rc; }
    if (second.prefix_reuse_source != ninfer::PrefixReuseSource::HostRam ||
        second.reused_prompt_tokens == 0) {
        std::cerr << "FAIL: RAM turn 2 reuse_source="
                  << static_cast<int>(second.prefix_reuse_source) << " reused "
                  << second.reused_prompt_tokens << '\n';
        return 1;
    }
    if (engine.runtime_stats().kv_ram_restores != restores_before + 1) {
        return fail("RAM turn 2 did not restore the prepended first chat from host KV");
    }
    ninfer::PromptInput closed           = turn1;
    closed.options.add_generation_prompt = false;
    const std::uint32_t closed_tokens    = engine.count_tokens(closed);
    if (second.reused_prompt_tokens < closed_tokens) {
        std::cerr << "FAIL: RAM turn 2 reused " << second.reused_prompt_tokens
                  << " tokens, expected at least " << closed_tokens << '\n';
        return 1;
    }
    return 0;
}

} // namespace

int main() {
    const char* groupwise = std::getenv("NINFER_QWEN3_6_27B_WEIGHTS");
    const char* nvfp4     = std::getenv("NINFER_QWEN3_6_27B_NVFP4_WEIGHTS");
    if ((groupwise == nullptr || *groupwise == '\0') && (nvfp4 == nullptr || *nvfp4 == '\0')) {
        std::cout << "skip: neither NINFER_QWEN3_6_27B_WEIGHTS nor "
                     "NINFER_QWEN3_6_27B_NVFP4_WEIGHTS is set\n";
        return 77;
    }
    int failures = 0;
    if (groupwise != nullptr && *groupwise != '\0') {
        failures += exercise_artifact(groupwise);
        failures += exercise_ram_artifact(groupwise);
    }
    if (nvfp4 != nullptr && *nvfp4 != '\0') {
        failures += exercise_artifact(nvfp4);
        failures += exercise_ram_artifact(nvfp4);
    }
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
