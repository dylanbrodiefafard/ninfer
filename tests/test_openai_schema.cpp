// Contract test for the OpenAI serving layer: request parsing (string + parts
// content, unsupported-feature rejection), response/chunk/models/error
// serialization shapes, and finish_reason mapping. This is the schema boundary
// consumed by external OpenAI clients.

#include "serve/openai_schema.h"
#include "serve/request.h"
#include "serve/tool_call_parser.h"
#include "serve/translate.h"

#include <nlohmann/json.hpp>

#include <array>
#include <cstddef>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <string>
#include <string_view>
#include <utility>

namespace {

using Json = nlohmann::json;
using namespace ninfer::serve;

int fail(const std::string& message) {
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

int check(bool condition, const std::string& message) { return condition ? 0 : fail(message); }

bool throws_api(const std::function<void()>& f) {
    try {
        f();
    } catch (const ApiException&) { return true; } catch (...) {
        return false;
    }
    return false;
}

std::string api_code(const std::function<void()>& f) {
    try {
        f();
    } catch (const ApiException& error) { return error.error().code; } catch (...) {
        return "wrong_exception";
    }
    return {};
}

RequestLimits default_limits() {
    RequestLimits limits;
    limits.default_max_tokens = 512;
    return limits;
}

ServeOptions default_server() { return ServeOptions{}; }

ninfer::PromptCapabilities effort_capabilities() {
    ninfer::PromptCapabilities capabilities;
    capabilities.enable_thinking                 = true;
    capabilities.reasoning_effort.low            = true;
    capabilities.reasoning_effort.medium         = true;
    capabilities.reasoning_effort.xhigh          = true;
    capabilities.reasoning_effort.default_effort = ninfer::ReasoningEffort::XHigh;
    return capabilities;
}

ninfer::OwnedMedia fake_media(const ContentPart& part) {
    ninfer::OwnedMedia media;
    media.kind =
        part.kind == ContentKind::Image ? ninfer::MediaKind::Image : ninfer::MediaKind::Video;
    media.bytes.push_back(0);
    media.media_type = part.source.media_type;
    return media;
}

ninfer::PromptInput translate(const GenerationRequest& req, std::string_view system_prepend = {}) {
    const ServeOptions server = default_server();
    return to_prompt_input(req, resolve_prompt_semantics(req, server, effort_capabilities()),
                           fake_media, system_prepend);
}

std::string joined_text(const ninfer::ChatMessage& message) {
    std::string text;
    for (const ninfer::MessagePart& part : message.parts) {
        if (part.kind == ninfer::MessagePartKind::Text) { text += part.text; }
    }
    return text;
}

// Strip "data: " prefix and trailing blank line from an SSE event, returning the
// parsed JSON payload.
Json parse_sse(const std::string& event) {
    const std::string prefix = "data: ";
    const std::string suffix = "\n\n";
    if (event.rfind(prefix, 0) != 0 || event.size() < prefix.size() + suffix.size()) {
        throw std::runtime_error("bad SSE framing: " + event);
    }
    const std::string json =
        event.substr(prefix.size(), event.size() - prefix.size() - suffix.size());
    return Json::parse(json);
}

int test_parse_string_content() {
    int failures                = 0;
    const Json body             = {{"model", "qwen3.6-27b"},
                                   {"messages", Json::array({Json{{"role", "user"}, {"content", "hello"}}})}};
    const GenerationRequest req = parse_chat_completion_request(body, default_limits());
    failures += check(req.model == "qwen3.6-27b", "model parsed");
    failures += check(req.messages.size() == 1, "one message parsed");
    failures += check(req.messages[0].role == ninfer::ChatRole::User, "role parsed");
    failures += check(req.messages[0].content.size() == 1, "one content part");
    failures += check(req.messages[0].content[0].kind == ContentKind::Text, "text part kind");
    failures += check(req.messages[0].content[0].text == "hello", "text part content");
    failures += check(!req.stream, "stream defaults false");
    failures += check(req.max_tokens == 512, "max_tokens default applied");
    failures += check(!req.max_tokens_set, "max_tokens_set false when defaulted");
    return failures;
}

int test_ninfer_capture_object() {
    const Json base = {{"model", "m"},
                       {"messages", Json::array({Json{{"role", "user"}, {"content", "hello"}}})}};
    int failures    = 0;
    failures += check(!parse_chat_completion_request(base, default_limits()).capture_context_checkpoint,
                      "omitted ninfer does not pin");
    Json pin          = base;
    pin["ninfer"]     = Json{{"capture_context_checkpoint", true}};
    const GenerationRequest pinned = parse_chat_completion_request(pin, default_limits());
    failures += check(pinned.capture_context_checkpoint, "ninfer capture flag parsed");
    failures +=
        check(to_request_options(pinned, default_server()).execution.capture_context_checkpoint,
              "ninfer capture flag reached ExecutionOptions");
    Json unknown      = base;
    unknown["ninfer"] = Json{{"capture_context_checkpoint", true}, {"foo", 1}};
    failures += check(api_code([&] { (void)parse_chat_completion_request(unknown, default_limits()); }) ==
                          "ninfer_option_not_supported",
                      "unknown ninfer key rejected");
    Json not_bool      = base;
    not_bool["ninfer"] = Json{{"capture_context_checkpoint", 1}};
    failures += check(throws_api([&] { (void)parse_chat_completion_request(not_bool, default_limits()); }),
                      "non-bool capture flag rejected");
    Json not_object      = base;
    not_object["ninfer"] = true;
    failures +=
        check(throws_api([&] { (void)parse_chat_completion_request(not_object, default_limits()); }),
              "non-object ninfer rejected");
    Json ninfer_null      = base;
    ninfer_null["ninfer"] = nullptr;
    failures +=
        check(throws_api([&] { (void)parse_chat_completion_request(ninfer_null, default_limits()); }),
              "Chat ninfer null is rejected");
    return failures;
}

int test_preserve_thinking_options() {
    const Json base = {
        {"model", "m"},
        {"messages", Json::array({Json{{"role", "user"}, {"content", "hello"}}})},
    };
    int failures = 0;

    Json kwargs                    = base;
    kwargs["chat_template_kwargs"] = Json{{"preserve_thinking", true}};
    const GenerationRequest kwargs_request =
        parse_chat_completion_request(kwargs, default_limits());
    failures += check(kwargs_request.preserve_thinking == true,
                      "chat_template_kwargs preserve_thinking parsed");
    failures += check(translate(kwargs_request).options.preserve_thinking,
                      "resolved preserve_thinking reached PromptInput");

    Json alias                 = base;
    alias["preserve_thinking"] = false;
    failures +=
        check(parse_chat_completion_request(alias, default_limits()).preserve_thinking == false,
              "top-level preserve_thinking alias parsed");

    Json same                 = kwargs;
    same["preserve_thinking"] = true;
    failures +=
        check(parse_chat_completion_request(same, default_limits()).preserve_thinking == true,
              "matching preserve_thinking values rejected");

    Json nulls                    = base;
    nulls["preserve_thinking"]    = nullptr;
    nulls["chat_template_kwargs"] = Json{{"preserve_thinking", nullptr}, {"future", nullptr}};
    failures +=
        check(!parse_chat_completion_request(nulls, default_limits()).preserve_thinking.has_value(),
              "null preserve_thinking did not remain omitted");

    Json conflict                 = kwargs;
    conflict["preserve_thinking"] = false;
    failures +=
        check(throws_api([&] { (void)parse_chat_completion_request(conflict, default_limits()); }),
              "conflicting preserve_thinking values accepted");

    Json bad_kwargs                    = base;
    bad_kwargs["chat_template_kwargs"] = true;
    failures += check(
        throws_api([&] { (void)parse_chat_completion_request(bad_kwargs, default_limits()); }),
        "non-object chat_template_kwargs accepted");
    Json bad_value                    = base;
    bad_value["chat_template_kwargs"] = Json{{"preserve_thinking", "yes"}};
    failures +=
        check(throws_api([&] { (void)parse_chat_completion_request(bad_value, default_limits()); }),
              "non-boolean preserve_thinking accepted");
    Json unknown                    = base;
    unknown["chat_template_kwargs"] = Json{{"preserve_thinking", true}, {"foo", 1}};
    failures +=
        check(throws_api([&] { (void)parse_chat_completion_request(unknown, default_limits()); }),
              "unknown non-null chat template option accepted");
    return failures;
}

int test_reasoning_effort() {
    const Json base = {
        {"model", "m"},
        {"messages", Json::array({Json{{"role", "user"}, {"content", "hello"}}})},
    };
    int failures = 0;

    Json low                            = base;
    low["reasoning_effort"]             = "low";
    const GenerationRequest low_request = parse_chat_completion_request(low, default_limits());
    failures += check(low_request.reasoning_effort == RequestedReasoningEffort::Low,
                      "Chat Completions reasoning_effort was not parsed");
    const ninfer::PromptInput low_prompt = translate(low_request);
    failures += check(low_prompt.options.enable_thinking &&
                          low_prompt.options.reasoning_effort == ninfer::ReasoningEffort::Low,
                      "Chat Completions low effort did not reach PromptInput");

    Json none                = base;
    none["reasoning_effort"] = "none";
    const ninfer::PromptInput none_prompt =
        translate(parse_chat_completion_request(none, default_limits()));
    failures += check(!none_prompt.options.enable_thinking && !none_prompt.options.reasoning_effort,
                      "Chat Completions none effort did not disable thinking");

    for (const auto& [wire, expected] :
         std::array<std::pair<const char*, RequestedReasoningEffort>, 6>{
             {{"minimal", RequestedReasoningEffort::Minimal},
              {"medium", RequestedReasoningEffort::Medium},
              {"high", RequestedReasoningEffort::High},
              {"xhigh", RequestedReasoningEffort::XHigh},
              {"max", RequestedReasoningEffort::Max},
              {"none", RequestedReasoningEffort::None}}}) {
        Json body                = base;
        body["reasoning_effort"] = wire;
        failures += check(parse_chat_completion_request(body, default_limits()).reasoning_effort ==
                              expected,
                          std::string("Chat Completions did not accept protocol effort ") + wire);
    }

    Json high                            = base;
    high["reasoning_effort"]             = "high";
    const GenerationRequest high_request = parse_chat_completion_request(high, default_limits());
    failures += check(api_code([&] {
                          (void)resolve_prompt_semantics(high_request, default_server(),
                                                         effort_capabilities());
                      }) == "reasoning_effort_not_supported",
                      "protocol-valid high effort was not rejected by template capability");

    ninfer::PromptCapabilities toggle_capabilities;
    toggle_capabilities.enable_thinking = true;
    failures += check(api_code([&] {
                          (void)resolve_prompt_semantics(low_request, default_server(),
                                                         toggle_capabilities);
                      }) == "reasoning_effort_not_supported",
                      "reasoning effort was accepted without template support");

    Json conflict               = low;
    conflict["enable_thinking"] = false;
    const GenerationRequest conflicting_request =
        parse_chat_completion_request(conflict, default_limits());
    failures += check(api_code([&] {
                          (void)resolve_prompt_semantics(conflicting_request, default_server(),
                                                         effort_capabilities());
                      }) == "conflicting_template_option",
                      "conflicting enable_thinking and reasoning_effort were accepted");

    Json invalid                = base;
    invalid["reasoning_effort"] = "ultra";
    failures +=
        check(throws_api([&] { (void)parse_chat_completion_request(invalid, default_limits()); }),
              "unknown Chat Completions reasoning effort was accepted");
    invalid["reasoning_effort"] = 1;
    failures +=
        check(throws_api([&] { (void)parse_chat_completion_request(invalid, default_limits()); }),
              "non-string Chat Completions reasoning effort was accepted");
    return failures;
}

int test_parse_parts_and_flatten() {
    int failures    = 0;
    const Json body = {
        {"model", "m"},
        {"messages",
         Json::array({Json{{"role", "user"},
                           {"content", Json::array({Json{{"type", "text"}, {"text", "a"}},
                                                    Json{{"type", "text"}, {"text", "b"}}})}}})}};
    const GenerationRequest req = parse_chat_completion_request(body, default_limits());
    failures += check(req.messages[0].content.size() == 2, "two content parts");
    const ninfer::PromptInput prompt = translate(req);
    failures += check(prompt.messages.size() == 1, "flattened to one message");
    failures += check(joined_text(prompt.messages[0]) == "a\nb", "text parts joined");
    return failures;
}

int test_parse_media_in_translate() {
    const Json body = {
        {"model", "m"},
        {"messages",
         Json::array({Json{
             {"role", "user"},
             {"content",
              Json::array(
                  {Json{{"type", "image_url"},
                        {"image_url", Json{{"url", "data:image/png;base64,AA=="}}}},
                   Json{{"type", "video_url"},
                        {"video_url", Json{{"url", "https://example.test/clip.mp4"}}}}})}}})}};
    const GenerationRequest req      = parse_chat_completion_request(body, default_limits());
    const ninfer::PromptInput prompt = translate(req);
    int failures                     = 0;
    failures += check(req.messages[0].content[0].kind == ContentKind::Image,
                      "image content kind preserved");
    failures += check(req.messages[0].content[0].source.kind ==
                          ninfer::product::media_acquire::SourceKind::Data,
                      "image data URI source preserved");
    failures += check(prompt.messages[0].parts[0].kind == ninfer::MessagePartKind::Media &&
                          prompt.messages[0].parts[0].media.kind == ninfer::MediaKind::Image,
                      "image translated to structured chat part");
    failures += check(req.messages[0].content[1].kind == ContentKind::Video,
                      "video content kind preserved");
    failures += check(req.messages[0].content[1].source.kind ==
                          ninfer::product::media_acquire::SourceKind::Url,
                      "video URL source preserved");
    failures += check(prompt.messages[0].parts[1].kind == ninfer::MessagePartKind::Media &&
                          prompt.messages[0].parts[1].media.kind == ninfer::MediaKind::Video,
                      "video translated to structured chat part");
    return failures;
}

int test_instruction_roles_preserved() {
    const Json body = {
        {"model", "m"},
        {"messages", Json::array({Json{{"role", "developer"}, {"content", "be terse"}},
                                  Json{{"role", "user"}, {"content", "hi"}},
                                  Json{{"role", "system"}, {"content", "new context"}}})}};
    const GenerationRequest req      = parse_chat_completion_request(body, default_limits());
    const ninfer::PromptInput prompt = translate(req);
    int failures                     = 0;
    failures +=
        check(req.messages.size() == 3 && req.messages[0].role == ninfer::ChatRole::Developer &&
                  req.messages[1].role == ninfer::ChatRole::User &&
                  req.messages[2].role == ninfer::ChatRole::System,
              "schema did not preserve ordered developer/system roles");
    failures += check(prompt.messages.size() == 3 &&
                          prompt.messages[0].role == ninfer::ChatRole::Developer &&
                          prompt.messages[1].role == ninfer::ChatRole::User &&
                          prompt.messages[2].role == ninfer::ChatRole::System,
                      "translation changed roles before target-specific lowering");
    return failures;
}

int test_reject_unsupported() {
    int failures    = 0;
    const Json base = {{"model", "m"},
                       {"messages", Json::array({Json{{"role", "user"}, {"content", "hi"}}})}};

    Json n2 = base;
    n2["n"] = 2;
    failures +=
        check(throws_api([&] { (void)parse_chat_completion_request(n2, default_limits()); }),
              "n>1 rejected");

    Json custom_tool = base;
    custom_tool["tools"] =
        Json::array({Json{{"type", "custom"}, {"custom", Json{{"name", "search"}}}}});
    failures += check(
        throws_api([&] { (void)parse_chat_completion_request(custom_tool, default_limits()); }),
        "custom tools rejected");

    Json functions         = base;
    functions["functions"] = Json::array({Json::object()});
    failures +=
        check(throws_api([&] { (void)parse_chat_completion_request(functions, default_limits()); }),
              "deprecated functions rejected");

    Json function_call             = base;
    function_call["function_call"] = "auto";
    failures += check(
        throws_api([&] { (void)parse_chat_completion_request(function_call, default_limits()); }),
        "deprecated function_call rejected");

    Json rf               = base;
    rf["response_format"] = Json{{"type", "json_object"}};
    failures +=
        check(throws_api([&] { (void)parse_chat_completion_request(rf, default_limits()); }),
              "json response_format rejected");

    Json rf_text               = base;
    rf_text["response_format"] = Json{{"type", "text"}};
    bool text_ok               = true;
    try {
        (void)parse_chat_completion_request(rf_text, default_limits());
    } catch (...) { text_ok = false; }
    failures += check(text_ok, "text response_format accepted");

    Json no_model = {{"messages", Json::array({Json{{"role", "user"}, {"content", "hi"}}})}};
    failures +=
        check(throws_api([&] { (void)parse_chat_completion_request(no_model, default_limits()); }),
              "missing model rejected");

    Json function_role = {
        {"model", "m"}, {"messages", Json::array({Json{{"role", "function"}, {"content", "x"}}})}};
    failures += check(
        throws_api([&] { (void)parse_chat_completion_request(function_role, default_limits()); }),
        "function role rejected");
    return failures;
}

int test_parse_function_tools_and_choices() {
    int failures = 0;
    const Json tool =
        Json{{"type", "function"},
             {"function",
              Json{{"name", "get_weather"},
                   {"description", "Fetch weather"},
                   {"parameters", Json{{"type", "object"},
                                       {"properties", Json{{"city", Json{{"type", "string"}}}}},
                                       {"required", Json::array({"city"})}}},
                   {"strict", true}}}};
    const Json base = {{"model", "m"},
                       {"messages", Json::array({Json{{"role", "user"}, {"content", "hi"}}})},
                       {"tools", Json::array({tool})}};

    GenerationRequest req = parse_chat_completion_request(base, default_limits());
    failures += check(req.tools.size() == 1, "one tool parsed");
    failures += check(req.tools[0].name == "get_weather", "tool name parsed");
    failures += check(req.tools[0].description == "Fetch weather", "tool description parsed");
    failures += check(req.tools[0].strict, "tool strict metadata parsed");
    failures += check(Json::parse(req.tools[0].parameters_json).at("required").at(0) == "city",
                      "tool parameters carried");
    failures += check(Json::parse(req.tools[0].definition_json).at("type") == "function",
                      "tool definition json carried");
    failures += check(req.tool_choice.mode == ToolChoiceMode::Auto, "default tool choice is auto");
    failures += check(req.uses_tools(), "tools enabled by default");
    failures += check(to_request_options(req, default_server()).output.preserve_special_tokens,
                      "active tools preserve special tokens in Engine output");

    Json none           = base;
    none["tool_choice"] = "none";
    req                 = parse_chat_completion_request(none, default_limits());
    failures += check(req.tool_choice.mode == ToolChoiceMode::None, "tool_choice none parsed");
    failures += check(!req.uses_tools(), "tool_choice none disables tools");
    failures += check(!to_request_options(req, default_server()).output.preserve_special_tokens,
                      "disabled tools do not preserve special tokens");

    Json required           = base;
    required["tool_choice"] = "required";
    req                     = parse_chat_completion_request(required, default_limits());
    failures +=
        check(req.tool_choice.mode == ToolChoiceMode::Required, "tool_choice required parsed");

    Json named           = base;
    named["tool_choice"] = Json{{"type", "function"}, {"function", Json{{"name", "get_weather"}}}};
    req                  = parse_chat_completion_request(named, default_limits());
    failures += check(req.tool_choice.mode == ToolChoiceMode::Named, "named tool_choice parsed");
    failures += check(req.tool_choice.name == "get_weather", "named tool_choice name parsed");

    Json unknown           = base;
    unknown["tool_choice"] = Json{{"type", "function"}, {"function", Json{{"name", "missing"}}}};
    failures +=
        check(throws_api([&] { (void)parse_chat_completion_request(unknown, default_limits()); }),
              "unknown named tool_choice rejected");
    return failures;
}

int test_parse_tool_history_messages() {
    int failures    = 0;
    const Json body = {
        {"model", "m"},
        {"messages",
         Json::array(
             {Json{{"role", "user"}, {"content", "weather?"}},
              Json{{"role", "assistant"},
                   {"content", nullptr},
                   {"tool_calls",
                    Json::array({Json{{"id", "call_1"},
                                      {"type", "function"},
                                      {"function", Json{{"name", "get_weather"},
                                                        {"arguments", R"({"city":"Paris"})"}}}}})}},
              Json{{"role", "tool"}, {"tool_call_id", "call_1"}, {"content", R"({"temp":20})"}}})}};
    const GenerationRequest req = parse_chat_completion_request(body, default_limits());
    failures += check(req.messages.size() == 3, "tool history message count");
    failures += check(req.messages[1].tool_calls.size() == 1, "assistant tool call parsed");
    failures += check(req.messages[1].tool_calls[0].id == "call_1", "tool call id parsed");
    failures += check(req.messages[1].tool_calls[0].name == "get_weather", "tool call name parsed");
    failures += check(req.messages[1].tool_calls[0].arguments_json == R"({"city":"Paris"})",
                      "tool call arguments parsed");
    failures += check(req.messages[2].role == ninfer::ChatRole::Tool, "tool role parsed");
    failures += check(req.messages[2].tool_call_id == "call_1", "tool_call_id parsed");
    failures +=
        check(req.messages[2].content.at(0).text == R"({"temp":20})", "tool content parsed");
    failures += check(to_request_options(req, default_server()).output.preserve_special_tokens,
                      "tool history preserves special tokens in Engine output");

    Json bad_args                                                     = body;
    bad_args["messages"][1]["tool_calls"][0]["function"]["arguments"] = R"(["Paris"])";
    failures +=
        check(throws_api([&] { (void)parse_chat_completion_request(bad_args, default_limits()); }),
              "non-object tool call arguments rejected");
    return failures;
}

int test_parse_stop_and_max_tokens() {
    int failures          = 0;
    Json body             = {{"model", "m"},
                             {"messages", Json::array({Json{{"role", "user"}, {"content", "hi"}}})},
                             {"stop", Json::array({"</s>", "STOP"})},
                             {"max_completion_tokens", 42}};
    GenerationRequest req = parse_chat_completion_request(body, default_limits());
    failures += check(req.stop_strings.size() == 2, "two stop strings");
    failures += check(req.stop_strings[0] == "</s>", "stop string 0");
    failures += check(req.max_tokens == 42 && req.max_tokens_set, "max_completion_tokens alias");
    const ninfer::RequestOptions options = to_request_options(req, default_server());
    failures += check(options.execution.requested_output_tokens == 42,
                      "max_completion_tokens reaches Engine options");
    failures += check(options.stop.strings.size() == 2 && options.stop.strings[0].text == "</s>" &&
                          options.stop.strings[1].text == "STOP",
                      "stop strings reach Engine options");

    Json single = {{"model", "m"},
                   {"messages", Json::array({Json{{"role", "user"}, {"content", "hi"}}})},
                   {"stop", "END"}};
    req         = parse_chat_completion_request(single, default_limits());
    failures +=
        check(req.stop_strings.size() == 1 && req.stop_strings[0] == "END", "single stop string");
    return failures;
}

int test_parse_sampling_carried() {
    int failures                = 0;
    const Json body             = {{"model", "m"},
                                   {"messages", Json::array({Json{{"role", "user"}, {"content", "hi"}}})},
                                   {"temperature", 0.7},
                                   {"top_p", 0.9},
                                   {"seed", 123},
                                   {"logit_bias", Json{{"5", -1.5}}}};
    const GenerationRequest req = parse_chat_completion_request(body, default_limits());
    failures += check(req.sampling.temperature.has_value() && *req.sampling.temperature == 0.7,
                      "temperature carried");
    failures +=
        check(req.sampling.top_p.has_value() && *req.sampling.top_p == 0.9, "top_p carried");
    failures += check(req.sampling.seed.has_value() && *req.sampling.seed == 123u, "seed carried");
    failures +=
        check(req.sampling.logit_bias.count(5) == 1 && req.sampling.logit_bias.at(5) == -1.5,
              "logit_bias carried");
    const ninfer::RequestOptions options = to_request_options(req, default_server());
    failures += check(options.execution.sampling.temperature == 0.7F,
                      "temperature reaches Engine overrides");
    failures += check(options.execution.sampling.top_p == 0.9F, "top_p reaches Engine overrides");
    failures += check(options.execution.sampling.seed == 123u, "seed reaches Engine overrides");
    failures +=
        check(!options.execution.sampling.top_k && !options.execution.sampling.presence_penalty,
              "omitted request fields unexpectedly replaced model defaults");
    return failures;
}

int test_response_serialization() {
    int failures = 0;
    const CompletionUsage usage{10, 3};
    const Json j = Json::parse(
        make_chat_completion_response("id-1", "m", 111, "hello world", "", "stop", usage));
    failures += check(j.at("object") == "chat.completion", "response object");
    failures += check(j.at("id") == "id-1", "response id");
    failures +=
        check(j.at("choices").at(0).at("message").at("role") == "assistant", "assistant role");
    failures += check(j.at("choices").at(0).at("message").at("content") == "hello world",
                      "response content");
    // Empty reasoning must not emit the reasoning_content key at all.
    failures += check(!j.at("choices").at(0).at("message").contains("reasoning_content"),
                      "no reasoning_content when reasoning empty");
    failures +=
        check(j.at("choices").at(0).at("finish_reason") == "stop", "response finish_reason");
    failures += check(j.at("usage").at("prompt_tokens") == 10, "usage prompt_tokens");
    failures += check(j.at("usage").at("completion_tokens") == 3, "usage completion_tokens");
    failures += check(j.at("usage").at("total_tokens") == 13, "usage total_tokens");
    failures += check(!j.at("usage").contains("prompt_tokens_details"),
                      "usage without timings has no prompt_tokens_details");
    failures += check(!j.contains("timings"), "top-level timings field present");

    CompletionTimings timings = make_completion_timings(10, 3, 0.25, 1.0);
    timings.kv_ram_capacity_bytes = 1048576;
    timings.kv_ram_used_bytes     = 524288;
    timings.kv_ram_entry_count    = 1;
    timings.kv_ram_captures       = 4;
    timings.kv_ram_restores       = 2;
    timings.kv_ram_evictions      = 0;
    timings.kv_ram_drops          = 1;
    timings.kv_ram_save_ms        = 8.0;
    timings.kv_ram_load_ms        = 14.0;
    timings.prefix_reuse_source   = ninfer::PrefixReuseSource::HostRam;
    timings.prompt_per_second     = 12.34567;
    timings.ttft_ms               = 358.024679;
    const Json jt = Json::parse(
        make_chat_completion_response("id-1", "m", 111, "hello world", "", "stop", usage, &timings));
    const Json& usage_t   = jt.at("usage");
    const Json& ptd       = usage_t.at("prompt_tokens_details");
    const Json& ninfer_ptd = ptd.at("ninfer");
    failures += check(usage_t.contains("prompt_tokens_details"),
                      "usage does not nest prompt_tokens_details");
    failures += check(!usage_t.contains("prompt_ms"), "usage duplicates stats at top level");
    failures += check(!usage_t.contains("prompt_eval_count"), "usage still emits Ollama aliases");
    failures += check(!usage_t.contains("reuse_source"), "usage duplicates reuse_source at top level");
    failures += check(!jt.contains("timings"), "top-level timings still present");
    failures += check(ptd.at("cached_tokens") == 0, "ptd cached_tokens zero on cache miss");
    failures += check(ninfer_ptd.at("ttft_ms") == 358.025, "ttft_ms rounded to three decimals");
    failures += check(ninfer_ptd.at("reuse_source") == "host_ram", "reuse_source host_ram");
    failures += check(ninfer_ptd.at("prefix_reuse_path") == "full_reset",
                      "prefix_reuse_path defaults to full_reset");
    failures += check(ninfer_ptd.at("prefill").at("ms") == 250.0, "prefill ms");
    failures += check(ninfer_ptd.at("prefill").at("tokens") == 10,
                      "prefill tokens is the computed suffix");
    failures += check(ninfer_ptd.at("prefill").at("tok_s") == 12.346,
                      "prefill rates are not rounded to three decimals");
    failures += check(ninfer_ptd.at("decode").at("ms") == 1000.0, "decode ms");
    failures += check(ninfer_ptd.at("decode").at("tokens") == 2,
                      "decode tokens exclude the prefill-sampled first token");
    // First completion token is attributed to prefill, so 3 completions = 2 decode tokens
    // over 1.0s.
    failures += check(ninfer_ptd.at("decode").at("tok_s") == 2.0, "decode tok_s");
    const Json& kv_ram = ninfer_ptd.at("kv_ram");
    failures += check(kv_ram.at("used_bytes") == 524288, "kv_ram used_bytes gauge");
    failures += check(kv_ram.at("entry_count") == 1, "kv_ram entry_count gauge");
    failures += check(kv_ram.at("save_ms") == 8.0, "kv_ram per-request save_ms");
    failures += check(kv_ram.at("load_ms") == 14.0, "kv_ram per-request load_ms");
    failures += check(kv_ram.at("lifetime").at("captures") == 4, "kv_ram lifetime captures");
    failures += check(kv_ram.at("lifetime").at("restores") == 2, "kv_ram lifetime restores");
    failures += check(kv_ram.at("lifetime").at("evictions") == 0, "kv_ram lifetime evictions");
    failures += check(kv_ram.at("lifetime").at("drops") == 1, "kv_ram lifetime drops");
    failures += check(!kv_ram.contains("capacity_bytes"), "kv_ram serializes pin capacity");

    // completion_tokens_details carries only OpenAI-standard keys.
    const Json& ctd = usage_t.at("completion_tokens_details");
    failures += check(ctd.at("reasoning_tokens") == 0, "ctd reasoning_tokens zero");
    failures += check(!ctd.contains("accepted_prediction_tokens"),
                      "ctd emits prediction tokens without speculation");

    CompletionTimings zero_timings = make_completion_timings(10, 3, 0.25, 1.0);
    const Json jz = Json::parse(make_chat_completion_response(
        "id-1", "m", 111, "hello world", "", "stop", usage, &zero_timings));
    failures += check(!jz.at("usage").at("prompt_tokens_details").at("ninfer").contains("kv_ram"),
                      "usage emits the KV RAM block when the tier is off");
    failures += check(jz.at("usage").at("prompt_tokens_details").at("ninfer").at("reuse_source") == "none",
                      "reuse_source omitted on cache miss");
    failures += check(!jz.at("usage").contains("reuse_source"),
                      "usage duplicates reuse_source at top level");

    zero_timings.prefix_reuse_source = ninfer::PrefixReuseSource::VramResident;
    zero_timings.prefix_reuse_path   = ninfer::PrefixReusePath::RestoreTurnCheckpoint;
    zero_timings.prompt_reused_n     = 4;
    const Json jturn = Json::parse(make_chat_completion_response(
        "id-1", "m", 111, "hello world", "", "stop", usage, &zero_timings));
    failures += check(jturn.at("usage").at("prompt_tokens_details").at("cached_tokens") == 4,
                      "turn restore still reports cached_tokens");
    failures += check(jturn.at("usage").at("prompt_tokens_details").at("ninfer")
                          .at("context_checkpoint")
                          .at("restored_tokens") == 0,
                      "turn restore must not copy cached_tokens into restored_tokens");

    zero_timings.prefix_reuse_path = ninfer::PrefixReusePath::RestoreContextCheckpoint;
    zero_timings.prompt_reused_n   = 0;
    const Json jv = Json::parse(make_chat_completion_response(
        "id-1", "m", 111, "hello world", "", "stop", usage, &zero_timings));
    failures += check(jv.at("usage").at("prompt_tokens_details").at("ninfer").at("reuse_source") ==
                          "vram_resident",
                      "reuse_source vram_resident");
    failures += check(jv.at("usage").at("prompt_tokens_details").at("ninfer").at("prefix_reuse_path") ==
                          "restore_context_checkpoint",
                      "prefix_reuse_path restore_context_checkpoint");
    zero_timings.prefix_reuse_path = ninfer::PrefixReusePath::RestoreTurnRollback;
    const Json jrollback = Json::parse(make_chat_completion_response(
        "id-1", "m", 111, "hello world", "", "stop", usage, &zero_timings));
    failures += check(jrollback.at("usage").at("prompt_tokens_details").at("ninfer")
                          .at("prefix_reuse_path") == "restore_turn_rollback",
                      "prefix_reuse_path restore_turn_rollback");
    zero_timings.prefix_reuse_path = ninfer::PrefixReusePath::RestoreContextCheckpoint;
    failures += check(jv.at("usage").at("prompt_tokens_details").at("ninfer").at("context_checkpoint")
                          .at("restored_tokens") == 0,
                      "context_checkpoint restored_tokens without cached_tokens");
    zero_timings.prompt_reused_n                    = 36864;
    zero_timings.restored_context_checkpoint_tokens = 36864;
    zero_timings.captured_context_checkpoint_tokens = 0;
    const Json jspan = Json::parse(make_chat_completion_response(
        "id-1", "m", 111, "hello world", "", "stop", usage, &zero_timings));
    failures += check(jspan.at("usage").at("prompt_tokens_details").at("cached_tokens") == 36864,
                      "cached_tokens remains the full reused prefix");
    failures += check(jspan.at("usage").at("prompt_tokens_details").at("ninfer").at("context_checkpoint")
                          .at("restored_tokens") == 36864,
                      "restored_tokens is the restored head frontier");
    failures += check(jspan.at("usage").at("prompt_tokens_details").at("ninfer").at("context_checkpoint")
                          .at("captured_tokens") == 0,
                      "restore-only request reports captured_tokens 0");
    failures += check(jspan.at("usage").at("prompt_tokens_details").at("cached_tokens").get<int>() ==
                          jspan.at("usage")
                              .at("prompt_tokens_details")
                              .at("ninfer")
                              .at("context_checkpoint")
                              .at("restored_tokens")
                              .get<int>(),
                      "restored_tokens matches cached_tokens on a staged restore");
    zero_timings.prompt_reused_n                    = 0;
    zero_timings.restored_context_checkpoint_tokens = 0;
    zero_timings.captured_context_checkpoint_tokens = 24576;
    const Json jmade = Json::parse(make_chat_completion_response(
        "id-1", "m", 111, "hello world", "", "stop", usage, &zero_timings));
    failures += check(jmade.at("usage").at("prompt_tokens_details").at("cached_tokens") == 0,
                      "freeze-only cached_tokens stays 0");
    failures += check(jmade.at("usage").at("prompt_tokens_details").at("ninfer").at("context_checkpoint")
                          .at("captured_tokens") == 24576,
                      "captured_tokens is this request's freeze frontier");
    failures += check(jmade.at("usage").at("prompt_tokens_details").at("ninfer").at("context_checkpoint")
                          .at("restored_tokens") == 0,
                      "freeze-only restored_tokens stays 0");
    zero_timings.prompt_reused_n                    = 36864;
    zero_timings.restored_context_checkpoint_tokens = 36864;
    zero_timings.captured_context_checkpoint_tokens = 102400;
    const Json jboth = Json::parse(make_chat_completion_response(
        "id-1", "m", 111, "hello world", "", "stop", usage, &zero_timings));
    failures += check(jboth.at("usage").at("prompt_tokens_details").at("cached_tokens") == 36864,
                      "combined cached_tokens stays the full reused prefix");
    failures += check(jboth.at("usage").at("prompt_tokens_details").at("ninfer").at("context_checkpoint")
                          .at("restored_tokens") == 36864,
                      "combined restored_tokens is the restore frontier");
    failures += check(jboth.at("usage").at("prompt_tokens_details").at("ninfer").at("context_checkpoint")
                          .at("captured_tokens") == 102400,
                      "combined captured_tokens is the freeze frontier");

    // Prefill rates cover the non-reused suffix only: a 6-token cached prefix must not
    // inflate prefill tok_s (4 computed tokens over 0.25s = 16, not 40).
    CompletionTimings reused_timings =
        make_completion_timings(10, 3, 0.25, 1.0, 0, 0, 0.0, 0.0, 6);
    failures += check(reused_timings.prompt_reused_n == 6, "prompt_reused_n recorded");
    failures +=
        check(reused_timings.prompt_per_second == 16.0, "prompt_per_second uses computed tokens");
    failures += check(reused_timings.prompt_per_token_ms == 62.5, "prompt_per_token_ms");
    failures += check(reused_timings.predicted_n == 2, "predicted_n is decode eval tokens");
    const Json jre = Json::parse(make_chat_completion_response(
        "id-1", "m", 111, "hello world", "", "stop", usage, &reused_timings));
    failures += check(jre.at("usage").at("prompt_tokens_details").at("cached_tokens") == 6,
                      "cached_tokens serialized");
    failures += check(jre.at("usage").at("prompt_tokens_details").at("ninfer")
                         .at("prefill")
                         .at("tokens") == 4,
                      "serialized prefill tokens reflects computed suffix");
    failures += check(jre.at("usage").at("prompt_tokens_details").at("ninfer")
                         .at("prefill")
                         .at("tok_s") == 16.0,
                      "serialized prefill tok_s reflects computed suffix");

    failures += check(decode_eval_tokens(0) == 0 && decode_eval_tokens(1) == 0 &&
                          decode_eval_tokens(2) == 1,
                      "decode_eval_tokens is max(0, n-1)");
    failures += check(prefill_eval_tokens(10, 6) == 4 && prefill_eval_tokens(4, 6) == 0,
                      "prefill_eval_tokens clamps reused to the prompt");
    CompletionTimings one_token = make_completion_timings(10, 1, 0.25, 1.0);
    failures += check(one_token.predicted_n == 0 && one_token.predicted_per_second == 0.0,
                      "one completion token is entirely prefill; decode tok_s is 0");
    CompletionTimings two_tokens = make_completion_timings(10, 2, 0.25, 0.5);
    failures += check(two_tokens.predicted_n == 1 && two_tokens.predicted_per_second == 2.0,
                      "two completions: one decode token over 0.5s");

    // Speculation + reasoning populate the OpenAI-standard completion details.
    CompletionTimings spec_timings = make_completion_timings(10, 3, 0.25, 1.0, 8, 5);
    spec_timings.reasoning_tokens = 2;
    const Json jsp = Json::parse(make_chat_completion_response(
        "id-1", "m", 111, "hello world", "", "stop", usage, &spec_timings));
    const Json& ctd_sp = jsp.at("usage").at("completion_tokens_details");
    failures += check(ctd_sp.at("reasoning_tokens") == 2, "ctd reasoning_tokens carried");
    failures += check(ctd_sp.at("accepted_prediction_tokens") == 5, "ctd accepted prediction");
    failures += check(ctd_sp.at("rejected_prediction_tokens") == 3, "ctd rejected prediction");

    // Non-empty reasoning is attached as message.reasoning_content, content stays answer-only.
    const Json jr = Json::parse(make_chat_completion_response("id-2", "m", 111, "the answer",
                                                              "let me think", "stop", usage));
    failures += check(jr.at("choices").at(0).at("message").at("content") == "the answer",
                      "reasoning response content is answer only");
    failures +=
        check(jr.at("choices").at(0).at("message").at("reasoning_content") == "let me think",
              "reasoning_content carried");
    return failures;
}

int test_tool_response_serialization() {
    int failures = 0;
    const CompletionUsage usage{12, 6};
    const std::vector<ToolCall> calls = {
        ToolCall{"call_abc", "get_weather", R"({"city":"Paris"})"}};
    const Json j = Json::parse(
        make_chat_completion_tool_response("id-tool", "m", 222, "", "need weather", calls, usage));

    failures += check(j.at("object") == "chat.completion", "tool response object");
    const Json& choice = j.at("choices").at(0);
    failures += check(choice.at("finish_reason") == "tool_calls", "tool finish reason");
    const Json& message = choice.at("message");
    failures += check(message.at("role") == "assistant", "tool message role");
    failures += check(message.at("content").is_null(), "empty tool content is null");
    failures += check(message.at("reasoning_content") == "need weather", "tool reasoning carried");
    const Json& call = message.at("tool_calls").at(0);
    failures += check(call.at("id") == "call_abc", "tool call id");
    failures += check(call.at("type") == "function", "tool call type");
    failures += check(call.at("function").at("name") == "get_weather", "tool function name");
    failures += check(call.at("function").at("arguments") == R"({"city":"Paris"})",
                      "tool function arguments");
    failures += check(j.at("usage").at("total_tokens") == 18, "tool usage total");

    const Json with_content = Json::parse(make_chat_completion_tool_response(
        "id-tool-2", "m", 223, "Calling weather.", "", calls, usage));
    failures +=
        check(with_content.at("choices").at(0).at("message").at("content") == "Calling weather.",
              "tool content prefix carried");
    return failures;
}

int test_chunk_serialization() {
    int failures    = 0;
    const Json role = parse_sse(make_chat_chunk_role("id", "m", 1, false));
    failures += check(role.at("object") == "chat.completion.chunk", "chunk object");
    failures += check(role.at("choices").at(0).at("delta").at("role") == "assistant", "role delta");
    failures += check(!role.contains("usage"), "no usage key when include_usage=false");

    const Json content = parse_sse(make_chat_chunk_content("id", "m", 1, "tok", false));
    failures +=
        check(content.at("choices").at(0).at("delta").at("content") == "tok", "content delta");

    // Reasoning deltas carry reasoning_content (not content) so clients render them
    // as a separate thinking channel.
    const Json reasoning = parse_sse(make_chat_chunk_reasoning("id", "m", 1, "why", false));
    failures += check(reasoning.at("choices").at(0).at("delta").at("reasoning_content") == "why",
                      "reasoning delta");
    failures += check(!reasoning.at("choices").at(0).at("delta").contains("content"),
                      "reasoning delta has no content key");

    // When usage reporting is on, content-bearing chunks carry usage: null.
    const Json role_usage = parse_sse(make_chat_chunk_role("id", "m", 1, true));
    failures += check(role_usage.contains("usage") && role_usage.at("usage").is_null(),
                      "role usage null when include_usage=true");
    const Json content_usage = parse_sse(make_chat_chunk_content("id", "m", 1, "x", true));
    failures += check(content_usage.contains("usage") && content_usage.at("usage").is_null(),
                      "content usage null when include_usage=true");

    // Final chunk carries finish_reason with an empty delta and no usage stats.
    const Json final_chunk = parse_sse(make_chat_chunk_final("id", "m", 1, "length", true));
    failures += check(final_chunk.at("choices").at(0).at("finish_reason") == "length",
                      "final finish_reason");
    failures += check(final_chunk.at("choices").at(0).at("delta").empty(), "final delta empty");
    failures += check(final_chunk.contains("usage") && final_chunk.at("usage").is_null(),
                      "final usage null (stats live on dedicated chunk)");

    const Json final_no_usage = parse_sse(make_chat_chunk_final("id", "m", 1, "stop", false));
    failures += check(!final_no_usage.contains("usage"), "no usage key when include_usage=false");

    // Dedicated usage chunk: empty choices, populated usage.
    const CompletionUsage usage{2, 5};
    const Json usage_chunk = parse_sse(make_chat_chunk_usage("id", "m", 1, usage));
    failures += check(usage_chunk.at("choices").is_array() && usage_chunk.at("choices").empty(),
                      "usage chunk has empty choices");
    failures +=
        check(usage_chunk.at("usage").at("prompt_tokens") == 2, "usage chunk prompt_tokens");
    failures += check(usage_chunk.at("usage").at("total_tokens") == 7, "usage chunk total");

    CompletionTimings ram_timings = make_completion_timings(2, 5, 0.25, 1.0);
    ram_timings.kv_ram_capacity_bytes = 1048576;
    ram_timings.kv_ram_used_bytes     = 524288;
    ram_timings.kv_ram_entry_count    = 1;
    ram_timings.kv_ram_captures       = 3;
    ram_timings.kv_ram_restores       = 2;
    ram_timings.kv_ram_evictions      = 0;
    ram_timings.kv_ram_drops          = 1;
    ram_timings.kv_ram_save_ms        = 8.0;
    ram_timings.kv_ram_load_ms        = 14.0;
    ram_timings.prefix_reuse_source   = ninfer::PrefixReuseSource::VramResident;
    const Json ram_chunk = parse_sse(make_chat_chunk_usage("id", "m", 1, usage, &ram_timings));
    const Json& ram_ninfer = ram_chunk.at("usage").at("prompt_tokens_details").at("ninfer");
    failures += check(ram_chunk.at("usage").contains("prompt_tokens_details"),
                      "usage chunk does not nest prompt_tokens_details");
    failures += check(!ram_chunk.at("usage").contains("kv_ram_used_bytes"),
                      "usage chunk duplicates kv stats at top level");
    failures += check(ram_ninfer.at("kv_ram").at("used_bytes") == 524288,
                      "usage chunk kv_ram used_bytes");
    failures += check(ram_ninfer.at("kv_ram").at("load_ms") == 14.0, "usage chunk kv_ram load_ms");
    failures += check(ram_ninfer.at("reuse_source") == "vram_resident", "usage chunk reuse_source");

    CompletionTimings ckpt_timings = make_completion_timings(2, 5, 0.25, 1.0);
    ckpt_timings.prefix_reuse_source                = ninfer::PrefixReuseSource::HostRam;
    ckpt_timings.prefix_reuse_path                  = ninfer::PrefixReusePath::RestoreContextCheckpoint;
    ckpt_timings.prompt_reused_n                    = 36864;
    ckpt_timings.restored_context_checkpoint_tokens = 36864;
    ckpt_timings.captured_context_checkpoint_tokens = 102400;
    const Json ckpt_chunk =
        parse_sse(make_chat_chunk_usage("id", "m", 1, usage, &ckpt_timings));
    const Json& ckpt_ninfer = ckpt_chunk.at("usage").at("prompt_tokens_details").at("ninfer");
    failures += check(ckpt_chunk.at("usage").at("prompt_tokens_details").at("cached_tokens") == 36864,
                      "stream usage chunk cached_tokens is the full reused prefix");
    failures += check(ckpt_ninfer.at("prefix_reuse_path") == "restore_context_checkpoint",
                      "stream usage chunk prefix_reuse_path");
    failures += check(ckpt_ninfer.at("context_checkpoint").at("restored_tokens") == 36864,
                      "stream usage chunk restored_tokens is the restored head frontier");
    failures += check(ckpt_ninfer.at("context_checkpoint").at("captured_tokens") == 102400,
                      "stream usage chunk captured_tokens is the freeze frontier");
    failures += check(ckpt_chunk.at("usage").at("prompt_tokens_details").at("cached_tokens").get<int>() ==
                          ckpt_ninfer.at("context_checkpoint").at("restored_tokens").get<int>(),
                      "stream usage chunk restored_tokens matches cached_tokens");
    const Json finish_ckpt = parse_sse(
        make_chat_chunk_final("id", "m", 1, "stop", true, &ckpt_timings, &usage));
    failures += check(finish_ckpt.at("usage").at("prompt_tokens_details").at("ninfer")
                          .at("context_checkpoint")
                          .at("restored_tokens") == 36864,
                      "finish-with-usage chunk restored_tokens");
    failures += check(finish_ckpt.at("usage").at("prompt_tokens_details").at("ninfer")
                          .at("context_checkpoint")
                          .at("captured_tokens") == 102400,
                      "finish-with-usage chunk captured_tokens");

    failures += check(sse_done() == "data: [DONE]\n\n", "done sentinel");
    return failures;
}

int test_tool_chunk_serialization() {
    int failures                      = 0;
    const std::vector<ToolCall> calls = {
        ToolCall{"call_abc", "get_weather", R"({"city":"Paris"})"}};
    const Json chunk = parse_sse(make_chat_chunk_tool_calls("id", "m", 1, calls, true));
    failures += check(chunk.at("object") == "chat.completion.chunk", "tool chunk object");
    const Json& delta = chunk.at("choices").at(0).at("delta");
    const Json& call  = delta.at("tool_calls").at(0);
    failures += check(call.at("index") == 0, "tool chunk index");
    failures += check(call.at("id") == "call_abc", "tool chunk id");
    failures += check(call.at("type") == "function", "tool chunk type");
    failures += check(call.at("function").at("name") == "get_weather", "tool chunk name");
    failures +=
        check(call.at("function").at("arguments") == R"({"city":"Paris"})", "tool chunk arguments");
    failures +=
        check(chunk.contains("usage") && chunk.at("usage").is_null(), "tool chunk usage null");

    const Json final_chunk = parse_sse(make_chat_chunk_final("id", "m", 1, "tool_calls", true));
    failures += check(final_chunk.at("choices").at(0).at("finish_reason") == "tool_calls",
                      "tool final finish reason");
    return failures;
}

int test_models_and_error() {
    int failures    = 0;
    const Json list = Json::parse(make_models_list("qwen3.6-27b", 1));
    failures += check(list.at("object") == "list", "models list object");
    failures += check(list.at("data").at(0).at("id") == "qwen3.6-27b", "models list id");
    failures += check(list.at("data").at(0).at("object") == "model", "models list entry object");
    failures += check(list.at("data").at(0).at("owned_by") == "ninfer", "models list owner");

    const Json one = Json::parse(make_model_object("qwen3.6-27b", 1));
    failures += check(one.at("id") == "qwen3.6-27b" && one.at("object") == "model", "model object");
    failures += check(one.at("owned_by") == "ninfer", "model owner");

    ApiError error;
    error.status   = 400;
    error.type     = "invalid_request_error";
    error.message  = "bad";
    error.param    = "messages";
    const Json err = Json::parse(make_error_body(error));
    failures += check(err.at("error").at("message") == "bad", "error message");
    failures += check(err.at("error").at("type") == "invalid_request_error", "error type");
    failures += check(err.at("error").at("param") == "messages", "error param");
    failures += check(err.at("error").at("code").is_null(), "error code null when empty");
    return failures;
}

int test_finish_reason_wire() {
    int failures = 0;
    failures += check(std::string(finish_reason_wire(ninfer::FinishReason::StopToken)) == "stop",
                      "stop token wire");
    failures +=
        check(std::string(finish_reason_wire(ninfer::FinishReason::OutputLimit)) == "length",
              "output limit wire");
    failures += check(std::string(finish_reason_wire(ninfer::FinishReason::Cancelled)) == "stop",
                      "cancelled maps to stop");
    return failures;
}

int test_system_prepend_merge() {
    int failures = 0;

    const GenerationRequest user_only = parse_chat_completion_request(
        Json{{"model", "m"},
             {"messages", Json::array({Json{{"role", "user"}, {"content", "hello"}}})}},
        default_limits());
    const ninfer::PromptInput inserted = translate(user_only, "P");
    failures += check(inserted.messages.size() == 2, "user-only prepend message count");
    failures += check(inserted.messages[0].role == ninfer::ChatRole::System &&
                          joined_text(inserted.messages[0]) == "P",
                      "user-only prepend did not insert a leading System");
    failures += check(inserted.messages[1].role == ninfer::ChatRole::User &&
                          joined_text(inserted.messages[1]) == "hello",
                      "user-only prepend did not keep the original user turn");

    const GenerationRequest leading_system = parse_chat_completion_request(
        Json{{"model", "m"},
             {"messages", Json::array({Json{{"role", "system"}, {"content", "Be brief."}},
                                      Json{{"role", "user"}, {"content", "hello"}}})}},
        default_limits());
    const ninfer::PromptInput merged = translate(leading_system, "P");
    failures += check(merged.messages.size() == 2, "leading system prepend message count");
    failures += check(merged.messages[0].role == ninfer::ChatRole::System &&
                          joined_text(merged.messages[0]) == "P\n\nBe brief.",
                      "leading system text was not prepended");

    const GenerationRequest empty_system = parse_chat_completion_request(
        Json{{"model", "m"},
             {"messages", Json::array({Json{{"role", "system"}, {"content", ""}},
                                      Json{{"role", "user"}, {"content", "hello"}}})}},
        default_limits());
    const ninfer::PromptInput empty_merged = translate(empty_system, "P");
    failures += check(empty_merged.messages[0].role == ninfer::ChatRole::System &&
                          joined_text(empty_merged.messages[0]) == "P",
                      "empty system prepend added a blank separator");

    const GenerationRequest late_system = parse_chat_completion_request(
        Json{{"model", "m"},
             {"messages", Json::array({Json{{"role", "user"}, {"content", "hello"}},
                                      Json{{"role", "system"}, {"content", "later"}}})}},
        default_limits());
    const ninfer::PromptInput late = translate(late_system, "P");
    failures += check(late.messages.size() == 3, "late system prepend message count");
    failures += check(late.messages[0].role == ninfer::ChatRole::System &&
                          joined_text(late.messages[0]) == "P",
                      "late system prepend did not insert a leading System");
    failures += check(late.messages[1].role == ninfer::ChatRole::User &&
                          joined_text(late.messages[1]) == "hello",
                      "late system prepend moved the user turn");
    failures += check(late.messages[2].role == ninfer::ChatRole::System &&
                          joined_text(late.messages[2]) == "later",
                      "late system turn was not left in place");

    const GenerationRequest follow_up = parse_chat_completion_request(
        Json{{"model", "m"},
             {"messages", Json::array({Json{{"role", "system"}, {"content", "Be brief."}},
                                      Json{{"role", "user"}, {"content", "q1"}},
                                      Json{{"role", "assistant"}, {"content", "old"}},
                                      Json{{"role", "user"}, {"content", "q2"}}})}},
        default_limits());
    const ninfer::PromptInput follow = translate(follow_up, "P");
    failures += check(follow.messages[0].role == ninfer::ChatRole::System &&
                          joined_text(follow.messages[0]).starts_with("P\n\nBe brief."),
                      "follow-up prepend did not target the leading system");

    const Json tool =
        Json{{"type", "function"},
             {"function",
              Json{{"name", "get_weather"},
                   {"description", "Fetch weather"},
                   {"parameters", Json{{"type", "object"},
                                       {"properties", Json{{"city", Json{{"type", "string"}}}}},
                                       {"required", Json::array({"city"})}}},
                   {"strict", true}}}};
    const GenerationRequest with_tools = parse_chat_completion_request(
        Json{{"model", "m"},
             {"messages", Json::array({Json{{"role", "user"}, {"content", "hello"}}})},
             {"tools", Json::array({tool})}},
        default_limits());
    const ninfer::PromptInput tooled = translate(with_tools, "P");
    failures += check(!tooled.options.tool_jsons.empty(), "tools were dropped when prepending");
    failures += check(tooled.messages[0].role == ninfer::ChatRole::System &&
                          joined_text(tooled.messages[0]) == "P",
                      "tools request prepend did not insert a leading System");

    const GenerationRequest tool_history = parse_chat_completion_request(
        Json{{"model", "m"},
             {"messages",
              Json::array(
                  {Json{{"role", "user"}, {"content", "weather?"}},
                   Json{{"role", "assistant"},
                        {"content", nullptr},
                        {"tool_calls",
                         Json::array({Json{{"id", "call_1"},
                                           {"type", "function"},
                                           {"function", Json{{"name", "get_weather"},
                                                             {"arguments", R"({"city":"Paris"})"}}}}})}},
                   Json{{"role", "tool"},
                        {"tool_call_id", "call_1"},
                        {"content", R"({"temp":20})"}}})},
             {"tools", Json::array({tool})}},
        default_limits());
    const ninfer::PromptInput looped = translate(tool_history, "P");
    failures += check(!looped.options.tool_jsons.empty(), "tool-loop tools were dropped when prepending");
    failures += check(looped.messages[0].role == ninfer::ChatRole::System &&
                          joined_text(looped.messages[0]) == "P",
                      "tool-loop prepend did not insert a leading System");
    failures += check(looped.messages[2].tool_calls.size() == 1 &&
                          looped.messages[2].tool_calls[0].name == "get_weather",
                      "tool-loop assistant tool call was lost");
    failures += check(looped.messages[3].role == ninfer::ChatRole::Tool &&
                          joined_text(looped.messages[3]) == R"({"temp":20})",
                      "tool-loop tool result was lost");
    return failures;
}

} // namespace

int run_sse_bench() {
    using ninfer::serve::make_chat_chunk_content;
    const std::string id    = "chatcmpl-bench";
    const std::string model = "qwen3.8-27b";
    const std::string delta = " the";
    auto nlohmann_chunk     = [&] {
        nlohmann::json payload = {{"id", id},
                                  {"object", "chat.completion.chunk"},
                                  {"created", 1},
                                  {"model", model},
                                  {"choices", nlohmann::json::array(
                                                 {nlohmann::json{{"index", 0},
                                                                 {"delta", nlohmann::json{{"content", delta}}},
                                                                 {"finish_reason", nullptr}}})}};
        return std::string("data: ") + payload.dump() + "\n\n";
    };
    auto time_ns = [](auto fn, int repeats) {
        (void)fn();
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < repeats; ++i) { (void)fn(); }
        return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start)
                   .count() /
               repeats;
    };
    constexpr int repeats = 50000;
    const double old_ns   = time_ns(nlohmann_chunk, repeats);
    const double new_ns =
        time_ns([&] { return make_chat_chunk_content(id, model, 1, delta, false); }, repeats);
    std::cerr << "sse_bench delta=\"" << delta << "\" nlohmann_ns=" << old_ns
              << " handwritten_ns=" << new_ns << " speedup=" << (old_ns / new_ns) << '\n';
    const std::string long_delta(256, 'x');
    const double old_long = time_ns(
        [&] {
            nlohmann::json payload = {
                {"id", id},
                {"object", "chat.completion.chunk"},
                {"created", 1},
                {"model", model},
                {"choices", nlohmann::json::array({nlohmann::json{
                    {"index", 0},
                    {"delta", nlohmann::json{{"content", long_delta}}},
                    {"finish_reason", nullptr}}})}};
            return std::string("data: ") + payload.dump() + "\n\n";
        },
        repeats);
    const double new_long =
        time_ns([&] { return make_chat_chunk_content(id, model, 1, long_delta, false); }, repeats);
    std::cerr << "sse_bench delta_bytes=256 nlohmann_ns=" << old_long
              << " handwritten_ns=" << new_long << " speedup=" << (old_long / new_long) << '\n';
    return 0;
}

// Open WebUI chat fb0d11ba (2026-08-30): reasoning closed, then this answer text was
// stored as output_text with no function_call item. Same model had executed
// fetch_url as a native tool_call minutes earlier. GenerationService only parses
// Qwen markup when the request uses_tools() or has tool history.
int test_owui_youtube_fetch_url_logged_turn() {
    const std::string content =
        "I'll look up what that video is about so I can summarize it for you.\n\n"
        "<tool_call>\n<function=fetch_url>\n<parameter=url>\n"
        "https://www.youtube.com/watch?v=7cEdPWh9hqU</parameter>\n"
        "</function>\n</tool_call>";
    const std::string reasoning =
        "The user shared a YouTube link without any instructions. Per my instructions: "
        "\"When given a YouTube link and no instructions provide a detailed summary of the video.\"\n\n"
        "I need to fetch the YouTube URL to get details about the video. Let me use fetch_url on it.\n";
    const Json user_message =
        Json{{"role", "user"},
             {"content", "https://youtu.be/7cEdPWh9hqU?si=48PBda0VDwvAzclN"}};
    const Json fetch_tool =
        Json{{"type", "function"},
             {"function",
              Json{{"name", "fetch_url"},
                   {"description", "Fetch and extract the main text content from a web page URL."},
                   {"parameters",
                    Json{{"type", "object"},
                         {"properties",
                          Json{{"url", Json{{"type", "string"},
                                            {"description", "The URL to fetch content from"}}}}},
                         {"required", Json::array({"url"})}}}}}};
    const CompletionUsage usage{1284, 115};

    int failures = 0;

    const GenerationRequest no_tools = parse_chat_completion_request(
        Json{{"model", "angel-gpt27b-5090"},
             {"messages", Json::array({user_message})},
             {"stream", true}},
        default_limits());
    failures += check(!no_tools.uses_tools() && !no_tools.has_tool_history(),
                      "logged request without tools is not tool-capable");

    const Json leaked = Json::parse(make_chat_completion_response(
        "chatcmpl-owui", "angel-gpt27b-5090", 1788077000, content, reasoning, "stop", usage));
    const Json& leaked_message = leaked.at("choices").at(0).at("message");
    failures += check(leaked.at("choices").at(0).at("finish_reason") == "stop",
                      "tools-off finish_reason is stop");
    failures += check(leaked_message.at("content") == content,
                      "tools-off content keeps the logged xml");
    failures += check(leaked_message.at("reasoning_content") == reasoning,
                      "tools-off reasoning matches the logged think block");
    failures += check(!leaked_message.contains("tool_calls"),
                      "tools-off response has no tool_calls");

    const GenerationRequest with_tools = parse_chat_completion_request(
        Json{{"model", "angel-gpt27b-5090"},
             {"messages", Json::array({user_message})},
             {"stream", true},
             {"tools", Json::array({fetch_tool})}},
        default_limits());
    failures += check(with_tools.uses_tools(), "logged request with fetch_url is tool-capable");

    const ParsedToolCallOutput parsed = parse_qwen_tool_call_output(content, 64);
    failures += check(parsed.is_tool_call_response, "logged xml parses when tools are on");
    failures += check(parsed.content ==
                          "I'll look up what that video is about so I can summarize it for you.",
                      "tools-on content is the spoken prefix only");
    failures += check(parsed.tool_calls.size() == 1 && parsed.tool_calls[0].name == "fetch_url",
                      "tools-on tool name is fetch_url");
    const Json args = Json::parse(parsed.tool_calls[0].arguments_json);
    failures += check(args.at("url") == "https://www.youtube.com/watch?v=7cEdPWh9hqU",
                      "tools-on url argument");

    const Json native = Json::parse(make_chat_completion_tool_response(
        "chatcmpl-owui", "angel-gpt27b-5090", 1788077000, parsed.content, reasoning,
        parsed.tool_calls, usage));
    const Json& native_message = native.at("choices").at(0).at("message");
    failures += check(native.at("choices").at(0).at("finish_reason") == "tool_calls",
                      "tools-on finish_reason is tool_calls");
    failures += check(native_message.at("content") == parsed.content,
                      "tools-on serialized content has no xml");
    failures += check(native_message.at("content").get<std::string>().find("<tool_call>") ==
                          std::string::npos,
                      "tools-on content does not contain tool markup");
    failures += check(native_message.at("reasoning_content") == reasoning,
                      "tools-on reasoning is unchanged");
    failures += check(native_message.at("tool_calls").at(0).at("function").at("name") ==
                          "fetch_url",
                      "tools-on wire tool name");

    ToolCallStreamFilter filtered;
    std::string streamed_with_tools = filtered.feed(content);
    streamed_with_tools += filtered.finish(true);
    failures += check(streamed_with_tools == parsed.content,
                      "tools-on stream holds the xml out of chat");
    return failures;
}

int main() {
    if (std::getenv("NINFER_BENCH_SSE") != nullptr) { return run_sse_bench(); }
    int failures = 0;
    failures += test_parse_string_content();
    failures += test_ninfer_capture_object();
    failures += test_preserve_thinking_options();
    failures += test_reasoning_effort();
    failures += test_parse_parts_and_flatten();
    failures += test_instruction_roles_preserved();
    failures += test_parse_media_in_translate();
    failures += test_reject_unsupported();
    failures += test_parse_function_tools_and_choices();
    failures += test_parse_tool_history_messages();
    failures += test_parse_stop_and_max_tokens();
    failures += test_parse_sampling_carried();
    failures += test_response_serialization();
    failures += test_tool_response_serialization();
    failures += test_owui_youtube_fetch_url_logged_turn();
    failures += test_chunk_serialization();
    failures += test_tool_chunk_serialization();
    failures += test_models_and_error();
    failures += test_finish_reason_wire();
    failures += test_system_prepend_merge();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
