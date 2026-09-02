#include <ninfer/targets/qwen3_6/frontend.h>
#include <ninfer/targets/qwen3_6/frontend_resources.h>

#include "targets/qwen3_6/impl/frontend/chat_template.h"
#include "targets/qwen3_6/impl/frontend/processor.h"
#include "targets/qwen3_6/impl/frontend/test_access.h"
#include "targets/qwen3_6/impl/frontend/tokenizer.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

using Frontend          = ninfer::targets::qwen3_6::Frontend;
using FrontendFactory   = ninfer::targets::qwen3_6::FrontendTestAccess;
using FrontendResources = ninfer::targets::qwen3_6::FrontendResources;
using PublishedOutput   = ninfer::targets::qwen3_6::PublishedOutput;
namespace fi            = ninfer::targets::qwen3_6::frontend_internal;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

std::string read_file(const char* path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) { throw std::runtime_error(std::string("failed to open test resource: ") + path); }
    return std::string(std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>());
}

std::string read_template_fixture(const char* path) {
    std::string source = read_file(path);
    if (!source.empty() && source.back() == '\n') { source.pop_back(); }
    return source;
}

const std::string& thinking_toggle_template_source() {
    static const std::string source = read_template_fixture(
        NINFER_SOURCE_DIR "/tests/fixtures/frontend/thinking_toggle_chat_template.jinja");
    return source;
}

const std::string& reasoning_effort_template_source() {
    static const std::string source = read_template_fixture(
        NINFER_SOURCE_DIR "/tests/fixtures/frontend/reasoning_effort_chat_template.jinja");
    return source;
}

const fi::CompiledChatTemplate& thinking_toggle_template() {
    static const fi::CompiledChatTemplate value =
        fi::CompiledChatTemplate::resolve(thinking_toggle_template_source());
    return value;
}

const fi::CompiledChatTemplate& reasoning_effort_template() {
    static const fi::CompiledChatTemplate value =
        fi::CompiledChatTemplate::resolve(reasoning_effort_template_source());
    return value;
}

const fi::Tokenizer& official_tokenizer() {
    static const std::string tokenizer_dir = [] {
        const char* env = std::getenv("NINFER_OFFICIAL_TOKENIZER_DIR");
        const char* candidates[] = {
            env,
            "/home/neroued/models/llm/qwen/Qwen3.6-27B/base-hf-bf16",
            "/ssdpool2nvme/local_llm/ninfer-dylan2/profiles/bench/official-tokenizer",
        };
        for (const char* dir : candidates) {
            if (dir == nullptr || dir[0] == '\0') { continue; }
            std::ifstream stream(std::string(dir) + "/tokenizer.json", std::ios::binary);
            if (stream) { return std::string(dir); }
        }
        throw std::runtime_error("official tokenizer.json was not found");
    }();
    static const std::string tokenizer_json = read_file((tokenizer_dir + "/tokenizer.json").c_str());
    static const std::string tokenizer_config_json =
        read_file((tokenizer_dir + "/tokenizer_config.json").c_str());
    static const std::string generation_config_json =
        read_file((tokenizer_dir + "/generation_config.json").c_str());
    static const fi::Tokenizer tokenizer({.tokenizer_json         = tokenizer_json,
                                          .tokenizer_config_json  = tokenizer_config_json,
                                          .generation_config_json = generation_config_json});
    return tokenizer;
}

nlohmann::json added(int id, std::string content, bool special = false) {
    return nlohmann::json{{"id", id},
                          {"content", std::move(content)},
                          {"single_word", false},
                          {"lstrip", false},
                          {"rstrip", false},
                          {"normalized", false},
                          {"special", special}};
}

nlohmann::json decoder_added(std::string content, bool special = false) {
    nlohmann::json value = added(0, std::move(content), special);
    value.erase("id");
    return value;
}

FrontendResources resources(const std::string& chat_template = thinking_toggle_template_source()) {
    FrontendResources result;
    result.chat_template_jinja  = chat_template;
    const nlohmann::json tokens = nlohmann::json::array(
        {added(1, "helloST"), added(2, "OPtail"), added(3, "thought</thi"),
         added(4, "nk>\n\nanswer"), added(6, "<eos>", true), added(7, "<0.0 seconds>"),
         added(14, "   \n"), added(15, "answer"),
         added(16, "<tool_"), added(17, "call>"), added(18, "<function=f>"),
         added(19, "</function>"), added(20, "</tool_call>"), added(21, "<tool_call>"),
         added(22, "preface"),
         added(30, "user\n"), added(31, "assistant\n"), added(32, "\n"),
         added(248045, "<|im_start|>", true), added(248046, "<|im_end|>", true),
         added(248053, "<|vision_start|>", true), added(248054, "<|vision_end|>", true),
         added(248056, "<|image_pad|>", true), added(248057, "<|video_pad|>", true),
         added(248068, "<think>"), added(248069, "</think>")});
    result.tokenizer_json = nlohmann::json{
        {"model",
         {{"type", "BPE"},
          {"vocab", {{"x", 0}, {"ä", 10}, {"¸", 11}, {"Ń", 12}}},
          {"merges", nlohmann::json::array()}}},
        {"added_tokens",
         tokens}}.dump();

    nlohmann::json decoder = nlohmann::json::object();
    for (const nlohmann::json& token : tokens) {
        nlohmann::json value = token;
        const std::string id = std::to_string(value.at("id").get<int>());
        value.erase("id");
        decoder[id] = std::move(value);
    }
    decoder["248070"]            = decoder_added("<|audio_start|>", true);
    decoder["248071"]            = decoder_added("<|audio_end|>", true);
    decoder["248072"]            = decoder_added("<tts_pad>", true);
    decoder["248073"]            = decoder_added("<tts_text_bos>", true);
    decoder["248074"]            = decoder_added("<tts_text_eod>", true);
    decoder["248075"]            = decoder_added("<tts_text_bos_single>", true);
    decoder["248076"]            = decoder_added("<|audio_pad|>", true);
    result.tokenizer_config_json = nlohmann::json{
        {"add_bos_token", false},
        {"add_prefix_space", false},
        {"pad_token", "<|endoftext|>"},
        {"chat_template", result.chat_template_jinja},
        {"added_tokens_decoder",
         std::move(decoder)}}.dump();
    result.generation_config_json = R"({"eos_token_id":[6]})";
    result.preprocessor_config_json =
        R"({"patch_size":16,"temporal_patch_size":2,"merge_size":2,"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],"size":{"shortest_edge":4096,"longest_edge":16777216}})";
    result.video_preprocessor_config_json =
        R"({"patch_size":16,"temporal_patch_size":2,"merge_size":2,"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],"size":{"shortest_edge":4096,"longest_edge":25165824}})";
    return result;
}

std::vector<std::uint8_t> gradient_ppm() {
    std::vector<std::uint8_t> ppm;
    const std::string header = "P6\n64 64\n255\n";
    for (const char byte : header) {
        ppm.push_back(static_cast<std::uint8_t>(static_cast<unsigned char>(byte)));
    }
    for (int index = 0; index < 64 * 64; ++index) {
        ppm.push_back(static_cast<std::uint8_t>(index & 0xff));
        ppm.push_back(static_cast<std::uint8_t>((index * 3) & 0xff));
        ppm.push_back(static_cast<std::uint8_t>((index * 7) & 0xff));
    }
    return ppm;
}

std::vector<std::uint8_t> block_ppm(int width, int height, std::uint8_t value) {
    const std::string header =
        "P6\n" + std::to_string(width) + ' ' + std::to_string(height) + "\n255\n";
    std::vector<std::uint8_t> ppm;
    ppm.reserve(header.size() +
                static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3);
    for (const char byte : header) {
        ppm.push_back(static_cast<std::uint8_t>(static_cast<unsigned char>(byte)));
    }
    ppm.insert(ppm.end(), static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 3,
               value);
    return ppm;
}

ninfer::PromptInput image_text_input(std::vector<std::uint8_t> bytes, std::string text,
                                     std::string source_name) {
    ninfer::MessagePart image;
    image.kind              = ninfer::MessagePartKind::Media;
    image.media.kind        = ninfer::MediaKind::Image;
    image.media.bytes       = std::move(bytes);
    image.media.media_type  = "image/x-portable-pixmap";
    image.media.source_name = std::move(source_name);

    ninfer::ChatMessage message;
    message.role = ninfer::ChatRole::User;
    message.parts.push_back(std::move(image));
    if (!text.empty()) {
        message.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = std::move(text), .media = {}});
    }
    ninfer::PromptInput input;
    input.messages.push_back(std::move(message));
    return input;
}

ninfer::PromptInput image_input() {
    ninfer::MessagePart image;
    image.kind              = ninfer::MessagePartKind::Media;
    image.media.kind        = ninfer::MediaKind::Image;
    image.media.bytes       = gradient_ppm();
    image.media.media_type  = "image/x-portable-pixmap";
    image.media.source_name = "inline.ppm";
    ninfer::ChatMessage message;
    message.role = ninfer::ChatRole::User;
    message.parts.push_back(std::move(image));
    ninfer::PromptInput input;
    input.messages.push_back(std::move(message));
    return input;
}

bool near(float actual, float expected) { return std::abs(actual - expected) < 1.0e-6F; }

constexpr std::array<std::uint8_t, 32> kGradientDigest{
    0x1e, 0x8c, 0xd9, 0x22, 0x40, 0xfa, 0x10, 0x62, 0x7b, 0x60, 0x86, 0x8e, 0xe9, 0x66, 0x41, 0xa2,
    0x4d, 0x21, 0xff, 0xc7, 0xe9, 0xa2, 0x2b, 0x34, 0xc0, 0xec, 0x99, 0x84, 0x6c, 0xa9, 0xa4, 0x8a,
};

std::string channel_text(const PublishedOutput& output, ninfer::OutputChannel channel) {
    std::string result;
    for (const ninfer::OutputDelta& delta : output) {
        if (delta.channel == channel) { result += delta.text; }
    }
    return result;
}

fi::ChatMessage chat_message(ninfer::ChatRole role, std::string content) {
    fi::ChatMessage message;
    message.role = role;
    message.parts.push_back(fi::ChatPart::text_part(std::move(content)));
    return message;
}

fi::RenderedChat render_chat(std::vector<fi::ChatMessage> messages,
                             fi::ChatRenderOptions options = {}) {
    return thinking_toggle_template().render(messages, std::move(options));
}

std::string render_chat_text(std::vector<fi::ChatMessage> messages,
                             fi::ChatRenderOptions options = {}) {
    return render_chat(std::move(messages), std::move(options)).text;
}

template <class Callable>
bool throws_invalid_argument(Callable&& callable) {
    try {
        callable();
    } catch (const std::invalid_argument&) { return true; }
    return false;
}

int test_official_tokenizer_merge() {
    const fi::Tokenizer& tokenizer = official_tokenizer();

    constexpr std::array<std::pair<const char*, int>, 7> appended = {{
        {"<|audio_start|>", 248070},
        {"<|audio_end|>", 248071},
        {"<tts_pad>", 248072},
        {"<tts_text_bos>", 248073},
        {"<tts_text_eod>", 248074},
        {"<tts_text_bos_single>", 248075},
        {"<|audio_pad|>", 248076},
    }};
    int failures = check(tokenizer.has_exact_token_domain(248077),
                         "official tokenizer merge left a hole in the token domain");
    for (const auto& [text, id] : appended) {
        const std::vector<int> encoded = tokenizer.encode(text);
        failures += check(encoded == std::vector<int>{id} && tokenizer.is_special_token(id) &&
                              tokenizer.decode_token_bytes(id) == text,
                          "official tokenizer_config.json token did not merge exactly");
    }

    FrontendResources conflicting = resources();
    nlohmann::json config         = nlohmann::json::parse(conflicting.tokenizer_config_json);
    config["added_tokens_decoder"]["248045"]["special"] = false;
    conflicting.tokenizer_config_json                   = config.dump();
    failures += check(
        throws_invalid_argument([&] {
            fi::Tokenizer invalid({.tokenizer_json         = conflicting.tokenizer_json,
                                   .tokenizer_config_json  = conflicting.tokenizer_config_json,
                                   .generation_config_json = conflicting.generation_config_json});
        }),
        "conflicting tokenizer/tokenizer_config added-token definitions were accepted");
    return failures;
}

int test_official_chat_template() {
    int failures = 0;
    failures += check(render_chat_text({chat_message(ninfer::ChatRole::User, "hello")}) ==
                          "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n",
                      "ordinary user prompt differs from the official template");

    fi::ChatRenderOptions no_generation;
    no_generation.add_generation_prompt = false;
    failures +=
        check(render_chat_text({chat_message(ninfer::ChatRole::System, "  be concise  "),
                                chat_message(ninfer::ChatRole::User, "hello")},
                               no_generation) == "<|im_start|>system\nbe concise<|im_end|>\n"
                                                 "<|im_start|>user\nhello<|im_end|>\n",
              "leading system prompt differs from the official template");
    failures += check(render_chat_text({chat_message(ninfer::ChatRole::System, ""),
                                        chat_message(ninfer::ChatRole::User, "hello")},
                                       no_generation) ==
                          "<|im_start|>system\n<|im_end|>\n<|im_start|>user\nhello<|im_end|>\n",
                      "empty leading system prompt differs from the official template");

    fi::ChatMessage tool_assistant = chat_message(ninfer::ChatRole::Assistant, "");
    tool_assistant.tool_calls.push_back(
        {.id = "", .name = "f", .arguments_json = R"({"flag":true,"nested":{"x":[1,2]}})"});
    failures += check(render_chat_text({chat_message(ninfer::ChatRole::User, "hi"), tool_assistant},
                                       no_generation) ==
                          "<|im_start|>user\nhi<|im_end|>\n"
                          "<|im_start|>assistant\n<think>\n\n</think>\n\n"
                          "<tool_call>\n<function=f>\n<parameter=flag>\ntrue\n</parameter>\n"
                          "<parameter=nested>\n{\"x\": [1, 2]}\n</parameter>\n"
                          "</function>\n</tool_call><|im_end|>\n",
                      "nested or boolean tool arguments differ from official JSON rendering");

    fi::ChatRenderOptions no_thinking;
    no_thinking.enable_thinking = false;
    failures +=
        check(render_chat_text({chat_message(ninfer::ChatRole::User, "q1"),
                                chat_message(ninfer::ChatRole::Assistant,
                                             "<think>\nold thought\n</think>\n\nold answer"),
                                chat_message(ninfer::ChatRole::User, "q2")},
                               no_thinking) == "<|im_start|>user\nq1<|im_end|>\n"
                                               "<|im_start|>assistant\nold answer<|im_end|>\n"
                                               "<|im_start|>user\nq2<|im_end|>\n"
                                               "<|im_start|>assistant\n<think>\n\n</think>\n\n",
              "thinking history differs from the official template");

    fi::ChatMessage lookup = chat_message(ninfer::ChatRole::Assistant, "");
    lookup.tool_calls.push_back(
        {.id = "", .name = "lookup", .arguments_json = R"({"city":"Paris"})"});
    failures +=
        check(render_chat_text({chat_message(ninfer::ChatRole::User, "weather?"), lookup,
                                chat_message(ninfer::ChatRole::Tool, "sunny"),
                                chat_message(ninfer::ChatRole::Tool, "20C"),
                                chat_message(ninfer::ChatRole::User, "thanks")},
                               no_generation) ==
                  "<|im_start|>user\nweather?<|im_end|>\n"
                  "<|im_start|>assistant\n<tool_call>\n<function=lookup>\n"
                  "<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call><|im_end|>\n"
                  "<|im_start|>user\n<tool_response>\nsunny\n</tool_response>\n"
                  "<tool_response>\n20C\n</tool_response><|im_end|>\n"
                  "<|im_start|>user\nthanks<|im_end|>\n",
              "tool-response grouping differs from the official template");

    fi::ChatRenderOptions tools = no_generation;
    tools.tool_jsons.push_back(
        R"({"type":"function","function":{"name":"f","description":"d","parameters":{"type":"object","properties":{"flag":{"type":"boolean"}}}}})");
    const std::string tools_rendered =
        render_chat_text({chat_message(ninfer::ChatRole::System, "be exact"),
                          chat_message(ninfer::ChatRole::User, "hi")},
                         tools);
    failures += check(
        tools_rendered.find("\n{\"type\": \"function\", \"function\": {\"name\": \"f\", "
                            "\"description\": \"d\", \"parameters\": {\"type\": \"object\", "
                            "\"properties\": {\"flag\": {\"type\": \"boolean\"}}}}}\n</tools>") !=
                std::string::npos &&
            tools_rendered.ends_with(
                "</IMPORTANT>\n\nbe exact<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n"),
        "tools system block differs from official tojson rendering");

    failures += check(throws_invalid_argument([&] {
                          (void)render_chat({chat_message(ninfer::ChatRole::System, "only")},
                                            no_generation);
                      }),
                      "message history without a user query was accepted");
    return failures;
}

int test_ordered_instruction_turns() {
    fi::ChatRenderOptions no_generation;
    no_generation.add_generation_prompt = false;

    const std::string leading_developer =
        render_chat_text({chat_message(ninfer::ChatRole::Developer, "policy"),
                          chat_message(ninfer::ChatRole::User, "hi")},
                         no_generation);
    int failures = check(leading_developer == "<|im_start|>system\npolicy<|im_end|>\n"
                                              "<|im_start|>user\nhi<|im_end|>\n",
                         "leading developer did not use the existing Qwen system path");

    const std::string late_system =
        render_chat_text({chat_message(ninfer::ChatRole::User, "hi"),
                          chat_message(ninfer::ChatRole::System, "  current diagnostics  ")},
                         no_generation);
    failures += check(late_system == "<|im_start|>user\nhi<|im_end|>\n"
                                     "<|im_start|>system\ncurrent diagnostics<|im_end|>\n",
                      "late system turn was not rendered at its original position");
    failures += check(
        render_chat_text({chat_message(ninfer::ChatRole::User, "hi"),
                          chat_message(ninfer::ChatRole::Developer, "  current diagnostics  ")},
                         no_generation) == late_system,
        "developer and system did not lower to the same in-place Qwen block");

    const std::string stable_history =
        render_chat_text({chat_message(ninfer::ChatRole::System, "stable policy"),
                          chat_message(ninfer::ChatRole::User, "hi")},
                         no_generation);
    const std::string appended_diagnostics =
        render_chat_text({chat_message(ninfer::ChatRole::System, "stable policy"),
                          chat_message(ninfer::ChatRole::User, "hi"),
                          chat_message(ninfer::ChatRole::System, "current diagnostics")},
                         no_generation);
    failures += check(appended_diagnostics.starts_with(stable_history) &&
                          appended_diagnostics.substr(stable_history.size()) ==
                              "<|im_start|>system\ncurrent diagnostics<|im_end|>\n",
                      "appended diagnostics changed the stable serialized history prefix");
    const std::vector<int> stable_tokens   = official_tokenizer().encode(stable_history);
    const std::vector<int> appended_tokens = official_tokenizer().encode(appended_diagnostics);
    failures +=
        check(appended_tokens.size() > stable_tokens.size() &&
                  std::equal(stable_tokens.begin(), stable_tokens.end(), appended_tokens.begin()),
              "appended diagnostics changed the stable token prefix");

    fi::ChatRenderOptions tools = no_generation;
    tools.tool_jsons.push_back(
        R"({"type":"function","function":{"name":"inspect","parameters":{"type":"object"}}})");
    const std::string tools_with_late_system =
        render_chat_text({chat_message(ninfer::ChatRole::System, "stable policy"),
                          chat_message(ninfer::ChatRole::User, "hi"),
                          chat_message(ninfer::ChatRole::System, "current diagnostics")},
                         tools);
    const std::size_t tools_position  = tools_with_late_system.find("# Tools");
    const std::size_t policy_position = tools_with_late_system.find("stable policy");
    const std::size_t user_position   = tools_with_late_system.find("<|im_start|>user\nhi");
    const std::size_t diagnostics_position =
        tools_with_late_system.find("<|im_start|>system\ncurrent diagnostics");
    failures +=
        check(tools_position != std::string::npos && policy_position != std::string::npos &&
                  user_position != std::string::npos && diagnostics_position != std::string::npos &&
                  tools_with_late_system.find("# Tools", tools_position + 1) == std::string::npos &&
                  tools_position < policy_position && policy_position < user_position &&
                  user_position < diagnostics_position,
              "late system duplicated or moved the leading tools/instruction block");

    const fi::RenderedChat generated =
        render_chat({chat_message(ninfer::ChatRole::User, "hi"),
                     chat_message(ninfer::ChatRole::System, "current diagnostics")});
    const std::string assistant_header = "<|im_start|>assistant\n";
    const std::size_t header           = generated.text.rfind(assistant_header);
    failures +=
        check(header != std::string::npos && generated.rewrite_checkpoint &&
                  generated.rewrite_checkpoint->kind ==
                      ninfer::targets::qwen3_6::RewriteCheckpointKind::TurnClosure &&
                  generated.rewrite_checkpoint->offset == header + assistant_header.size() &&
                  generated.text.find("current diagnostics<|im_end|>\n", 0) < header,
              "late system was not included before the generation rewrite boundary");

    fi::ChatMessage invalid = chat_message(ninfer::ChatRole::System, "diagnostics");
    invalid.tool_calls.push_back({.id = "call", .name = "f", .arguments_json = "{}"});
    failures += check(throws_invalid_argument([&] {
                          (void)render_chat({chat_message(ninfer::ChatRole::User, "hi"), invalid},
                                            no_generation);
                      }),
                      "system turn carrying assistant tool metadata was accepted");

    fi::ChatMessage media_instruction = chat_message(ninfer::ChatRole::Developer, "diagnostics");
    media_instruction.parts.push_back(fi::ChatPart::image({}));
    failures +=
        check(throws_invalid_argument([&] {
                  (void)render_chat({chat_message(ninfer::ChatRole::User, "hi"), media_instruction},
                                    no_generation);
              }),
              "developer turn carrying media was accepted");

    fi::ChatMessage invalid_role = chat_message(ninfer::ChatRole::User, "bad");
    invalid_role.role            = static_cast<ninfer::ChatRole>(255);
    failures +=
        check(throws_invalid_argument([&] {
                  (void)render_chat({chat_message(ninfer::ChatRole::User, "hi"), invalid_role},
                                    no_generation);
              }),
              "invalid typed chat role was accepted");
    return failures;
}

int test_reasoning_effort_chat_template() {
    constexpr std::string_view low_instructions =
        "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly "
        "to the conclusion without unnecessary elaboration.";
    constexpr std::string_view xhigh_instructions =
        "Reasoning effort is set to xhigh. Please think carefully through the task, validate key "
        "assumptions, consider plausible alternatives, and prioritize correctness, consistency, "
        "and clarity in the final answer.";

    const ninfer::PromptCapabilities toggle_capabilities =
        thinking_toggle_template().capabilities();
    const ninfer::PromptCapabilities effort_capabilities =
        reasoning_effort_template().capabilities();
    int failures = check(toggle_capabilities.enable_thinking &&
                             !toggle_capabilities.reasoning_effort.default_effort &&
                             !toggle_capabilities.reasoning_effort.low &&
                             !toggle_capabilities.reasoning_effort.medium &&
                             !toggle_capabilities.reasoning_effort.xhigh,
                         "thinking-toggle template advertised reasoning effort");
    failures += check(
        effort_capabilities.enable_thinking && effort_capabilities.reasoning_effort.low &&
            effort_capabilities.reasoning_effort.medium &&
            effort_capabilities.reasoning_effort.xhigh &&
            effort_capabilities.reasoning_effort.default_effort == ninfer::ReasoningEffort::XHigh,
        "reasoning-effort template did not advertise its complete capability set");

    const auto render_effort = [](ninfer::ReasoningEffort effort) {
        fi::ChatRenderOptions options;
        options.reasoning_effort = effort;
        return reasoning_effort_template()
            .render({chat_message(ninfer::ChatRole::User, "hello")}, options)
            .text;
    };
    const std::string tail = "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n";
    failures += check(
        reasoning_effort_template().render({chat_message(ninfer::ChatRole::User, "hello")}).text ==
            "<|im_start|>system\n" + std::string(xhigh_instructions) + "<|im_end|>\n" + tail,
        "reasoning-effort template did not apply its xhigh default");
    failures +=
        check(render_effort(ninfer::ReasoningEffort::Low) ==
                  "<|im_start|>system\n" + std::string(low_instructions) + "<|im_end|>\n" + tail,
              "low reasoning effort did not render the official instruction");
    failures += check(render_effort(ninfer::ReasoningEffort::Medium) == tail,
                      "medium reasoning effort injected an instruction");

    fi::ChatRenderOptions disabled;
    disabled.enable_thinking = false;
    failures += check(reasoning_effort_template()
                              .render({chat_message(ninfer::ChatRole::System, ""),
                                       chat_message(ninfer::ChatRole::User, "hello")},
                                      disabled)
                              .text == "<|im_start|>user\nhello<|im_end|>\n"
                                       "<|im_start|>assistant\n<think>\n\n</think>\n\n",
                      "disabled thinking did not suppress effort and an empty system turn");
    disabled.reasoning_effort = ninfer::ReasoningEffort::Low;
    failures += check(throws_invalid_argument([&] {
                          (void)reasoning_effort_template().render(
                              {chat_message(ninfer::ChatRole::User, "hello")}, disabled);
                      }),
                      "reasoning effort and disabled thinking were accepted together");

    fi::ChatRenderOptions unsupported;
    unsupported.reasoning_effort = ninfer::ReasoningEffort::Low;
    failures += check(throws_invalid_argument([&] {
                          (void)thinking_toggle_template().render(
                              {chat_message(ninfer::ChatRole::User, "hello")}, unsupported);
                      }),
                      "thinking-toggle template accepted reasoning effort");

    fi::ChatMessage previous   = chat_message(ninfer::ChatRole::Assistant, "old answer");
    previous.reasoning_content = "old thought";
    fi::ChatRenderOptions no_generation;
    no_generation.add_generation_prompt = false;
    no_generation.reasoning_effort      = ninfer::ReasoningEffort::Medium;
    const std::string preserved         = reasoning_effort_template()
                                      .render({chat_message(ninfer::ChatRole::User, "q1"), previous,
                                               chat_message(ninfer::ChatRole::User, "q2")},
                                              no_generation)
                                      .text;
    failures += check(preserved == "<|im_start|>user\nq1<|im_end|>\n"
                                   "<|im_start|>assistant\n<think>\nold thought\n</think>\n\n"
                                   "old answer<|im_end|>\n"
                                   "<|im_start|>user\nq2<|im_end|>\n",
                      "reasoning-effort template did not preserve prior thinking by default");
    no_generation.preserve_thinking = false;
    failures += check(reasoning_effort_template()
                              .render({chat_message(ninfer::ChatRole::User, "q1"), previous,
                                       chat_message(ninfer::ChatRole::User, "q2")},
                                      no_generation)
                              .text == "<|im_start|>user\nq1<|im_end|>\n"
                                       "<|im_start|>assistant\nold answer<|im_end|>\n"
                                       "<|im_start|>user\nq2<|im_end|>\n",
                      "explicit preserve_thinking=false did not remove prior thinking");

    fi::ChatMessage empty_arguments = chat_message(ninfer::ChatRole::Assistant, "");
    empty_arguments.tool_calls.push_back({.id = "", .name = "f", .arguments_json = ""});
    failures += check(
        reasoning_effort_template()
            .render({chat_message(ninfer::ChatRole::User, "call"), empty_arguments}, no_generation)
            .text == "<|im_start|>user\ncall<|im_end|>\n"
                     "<|im_start|>assistant\n<tool_call>\n<function=f>\n</function>\n"
                     "</tool_call><|im_end|>\n",
        "empty tool arguments did not follow the reasoning-effort template");
    return failures;
}

int test_reasoning_effort_empty_history_think() {
    const auto render = [](std::vector<fi::ChatMessage> messages, fi::ChatRenderOptions options) {
        return reasoning_effort_template().render(std::move(messages), std::move(options)).text;
    };

    fi::ChatRenderOptions medium_closed;
    medium_closed.add_generation_prompt = false;
    medium_closed.reasoning_effort      = ninfer::ReasoningEffort::Medium;

    fi::ChatMessage empty_history = chat_message(ninfer::ChatRole::Assistant, "old answer");
    const std::vector<fi::ChatMessage> closed_empty{
        chat_message(ninfer::ChatRole::User, "q1"), empty_history,
        chat_message(ninfer::ChatRole::User, "q2")};
    constexpr std::string_view closed_empty_desired =
        "<|im_start|>user\nq1<|im_end|>\n"
        "<|im_start|>assistant\nold answer<|im_end|>\n"
        "<|im_start|>user\nq2<|im_end|>\n";

    fi::ChatRenderOptions preserve_closed = medium_closed;
    preserve_closed.preserve_thinking     = true;
    int failures =
        check(render(closed_empty, preserve_closed) == closed_empty_desired,
              "preserve-on empty reasoning still wrapped a history think block");

    fi::ChatRenderOptions preserve_generate = preserve_closed;
    preserve_generate.add_generation_prompt = true;
    failures += check(render(closed_empty, preserve_generate) ==
                          std::string(closed_empty_desired) +
                              "<|im_start|>assistant\n<think>\n",
                      "preserve-on empty history reasoning still wrapped before the prologue");

    fi::ChatMessage whitespace_history = empty_history;
    whitespace_history.reasoning_content = "  \n\t  ";
    failures += check(render({chat_message(ninfer::ChatRole::User, "q1"), whitespace_history,
                              chat_message(ninfer::ChatRole::User, "q2")},
                             preserve_closed) == closed_empty_desired,
                      "whitespace-only reasoning_content was treated as real thoughts");

    fi::ChatMessage empty_arguments = chat_message(ninfer::ChatRole::Assistant, "");
    empty_arguments.tool_calls.push_back({.id = "", .name = "f", .arguments_json = ""});
    fi::ChatRenderOptions tool_preserve_off = medium_closed;
    tool_preserve_off.preserve_thinking     = false;
    constexpr std::string_view empty_tool_desired =
        "<|im_start|>user\ncall<|im_end|>\n"
        "<|im_start|>assistant\n<tool_call>\n<function=f>\n</function>\n"
        "</tool_call><|im_end|>\n";
    failures += check(render({chat_message(ninfer::ChatRole::User, "call"), empty_arguments},
                             tool_preserve_off) == empty_tool_desired,
                      "preserve-off tool-loop still synthesized an empty history think wrapper");

    fi::ChatMessage first_empty = chat_message(ninfer::ChatRole::Assistant, "a1");
    fi::ChatMessage second_kept = chat_message(ninfer::ChatRole::Assistant, "a2");
    second_kept.reasoning_content = "thought2";
    failures += check(render({chat_message(ninfer::ChatRole::User, "q1"), first_empty,
                              chat_message(ninfer::ChatRole::User, "q2"), second_kept},
                             preserve_closed) ==
                          "<|im_start|>user\nq1<|im_end|>\n"
                          "<|im_start|>assistant\na1<|im_end|>\n"
                          "<|im_start|>user\nq2<|im_end|>\n"
                          "<|im_start|>assistant\n<think>\nthought2\n</think>\n\n"
                          "a2<|im_end|>\n",
                      "empty-history skip was conversation-global or dropped real thoughts");

    fi::ChatMessage stuffed = chat_message(ninfer::ChatRole::Assistant,
                                           "<think>\nstuffed\n</think>\n\nbody");
    failures += check(render({chat_message(ninfer::ChatRole::User, "q1"), stuffed,
                              chat_message(ninfer::ChatRole::User, "q2")},
                             preserve_closed) ==
                          "<|im_start|>user\nq1<|im_end|>\n"
                          "<|im_start|>assistant\n<think>\nstuffed\n</think>\n\n"
                          "body<|im_end|>\n"
                          "<|im_start|>user\nq2<|im_end|>\n",
                      "effort template scraped or prepended an empty wrapper around stuffed think");

    fi::ChatMessage padded_thought = chat_message(ninfer::ChatRole::Assistant, "old answer");
    padded_thought.reasoning_content = "  thought  ";
    failures += check(render({chat_message(ninfer::ChatRole::User, "q1"), padded_thought,
                              chat_message(ninfer::ChatRole::User, "q2")},
                             preserve_closed) ==
                          "<|im_start|>user\nq1<|im_end|>\n"
                          "<|im_start|>assistant\n<think>\nthought\n</think>\n\n"
                          "old answer<|im_end|>\n"
                          "<|im_start|>user\nq2<|im_end|>\n",
                      "trimmed non-empty reasoning_content was dropped as empty");

    fi::ChatMessage reasoned_tool = chat_message(ninfer::ChatRole::Assistant, "");
    reasoned_tool.reasoning_content = "thought";
    reasoned_tool.tool_calls.push_back({.id = "", .name = "f", .arguments_json = ""});
    constexpr std::string_view reasoned_tool_desired =
        "<|im_start|>user\ncall<|im_end|>\n"
        "<|im_start|>assistant\n<think>\nthought\n</think>\n\n"
        "<tool_call>\n<function=f>\n</function>\n"
        "</tool_call><|im_end|>\n";
    fi::ChatRenderOptions tool_preserve_on = medium_closed;
    tool_preserve_on.preserve_thinking     = true;
    failures += check(render({chat_message(ninfer::ChatRole::User, "call"), reasoned_tool},
                             tool_preserve_on) == reasoned_tool_desired,
                      "empty-body tool turn dropped a real history think wrapper");

    fi::ChatMessage reasoned_note = reasoned_tool;
    reasoned_note.parts.front().text = "note";
    failures += check(render({chat_message(ninfer::ChatRole::User, "call"), reasoned_note},
                             tool_preserve_on) ==
                          "<|im_start|>user\ncall<|im_end|>\n"
                          "<|im_start|>assistant\n<think>\nthought\n</think>\n\n"
                          "note\n\n<tool_call>\n<function=f>\n</function>\n"
                          "</tool_call><|im_end|>\n",
                      "body-then-tool spacing lost the extra blank line before tool XML");

    failures += check(render(closed_empty, tool_preserve_off) == closed_empty_desired,
                      "preserve-off closed empty reasoning synthesized a think wrapper");

    fi::ChatRenderOptions toggle_preserve;
    toggle_preserve.add_generation_prompt = false;
    toggle_preserve.preserve_thinking     = true;
    failures += check(render_chat_text({chat_message(ninfer::ChatRole::User, "q1"),
                                        chat_message(ninfer::ChatRole::Assistant, "old answer"),
                                        chat_message(ninfer::ChatRole::User, "q2")},
                                       toggle_preserve) ==
                          "<|im_start|>user\nq1<|im_end|>\n"
                          "<|im_start|>assistant\n<think>\n\n</think>\n\n"
                          "old answer<|im_end|>\n"
                          "<|im_start|>user\nq2<|im_end|>\n",
                      "thinking-toggle preserve-on empty reasoning lost its empty think wrapper");

    fi::ChatRenderOptions effort_generate;
    effort_generate.reasoning_effort = ninfer::ReasoningEffort::Medium;
    failures += check(render({chat_message(ninfer::ChatRole::User, "please <|think_off|> now")},
                             effort_generate) ==
                          "<|im_start|>user\nplease <|think_off|> now<|im_end|>\n"
                          "<|im_start|>assistant\n<think>\n",
                      "user <|think_off|> text toggled thinking or was stripped");
    failures += check(render({chat_message(ninfer::ChatRole::User, "please <|think_on|> now")},
                             effort_generate) ==
                          "<|im_start|>user\nplease <|think_on|> now<|im_end|>\n"
                          "<|im_start|>assistant\n<think>\n",
                      "user <|think_on|> text toggled thinking or was stripped");

    const std::string assistant_header = "<|im_start|>assistant\n";
    const fi::RenderedChat empty_replay =
        reasoning_effort_template().render(closed_empty, preserve_generate);
    failures += check(empty_replay.rewrite_checkpoint &&
                          empty_replay.rewrite_checkpoint->kind ==
                              ninfer::targets::qwen3_6::RewriteCheckpointKind::ResponseReplay &&
                          empty_replay.rewrite_checkpoint->offset == empty_replay.text.size() &&
                          empty_replay.text.ends_with("<think>\n"),
                      "empty history reasoning moved the preserve-on thinking replay checkpoint");

    fi::ChatMessage kept_history = empty_history;
    kept_history.reasoning_content = "old thought";
    const fi::RenderedChat kept_replay = reasoning_effort_template().render(
        {chat_message(ninfer::ChatRole::User, "q1"), kept_history,
         chat_message(ninfer::ChatRole::User, "q2")},
        preserve_generate);
    failures += check(kept_replay.rewrite_checkpoint &&
                          kept_replay.rewrite_checkpoint->kind ==
                              ninfer::targets::qwen3_6::RewriteCheckpointKind::ResponseReplay &&
                          kept_replay.rewrite_checkpoint->offset == kept_replay.text.size() &&
                          kept_replay.text.ends_with("<think>\n"),
                      "kept history thoughts moved the preserve-on thinking replay checkpoint");

    fi::ChatRenderOptions preserve_nothinking = preserve_generate;
    preserve_nothinking.enable_thinking       = false;
    preserve_nothinking.reasoning_effort.reset();
    const fi::RenderedChat off_replay =
        reasoning_effort_template().render(closed_empty, preserve_nothinking);
    failures += check(off_replay.rewrite_checkpoint &&
                          off_replay.rewrite_checkpoint->kind ==
                              ninfer::targets::qwen3_6::RewriteCheckpointKind::ResponseReplay &&
                          off_replay.rewrite_checkpoint->offset == off_replay.text.size() &&
                          off_replay.text.ends_with("<think>\n\n</think>\n\n"),
                      "empty history reasoning moved the preserve-on no-thinking replay checkpoint");

    fi::ChatRenderOptions tool_generate = tool_preserve_off;
    tool_generate.add_generation_prompt = true;
    const fi::RenderedChat tool_closure = reasoning_effort_template().render(
        {chat_message(ninfer::ChatRole::User, "call"), empty_arguments}, tool_generate);
    const std::size_t first_header = tool_closure.text.find(assistant_header);
    failures += check(first_header != std::string::npos && tool_closure.rewrite_checkpoint &&
                          tool_closure.rewrite_checkpoint->kind ==
                              ninfer::targets::qwen3_6::RewriteCheckpointKind::TurnClosure &&
                          tool_closure.rewrite_checkpoint->offset ==
                              first_header + assistant_header.size(),
                      "preserve-off tool-loop did not keep TurnClosure at the first assistant header");

    failures += check(render({chat_message(ninfer::ChatRole::User, "call"), reasoned_tool},
                             tool_preserve_off) == reasoned_tool_desired,
                      "preserve-off tool-loop dropped a real history think wrapper");
    return failures;
}

int test_rewrite_checkpoint_trace() {
    const std::string assistant_header = "<|im_start|>assistant\n";
    fi::ChatMessage first              = chat_message(ninfer::ChatRole::Assistant, "");
    first.reasoning_content            = "first thought";
    first.parts.front().text           = "first answer";
    fi::ChatMessage second             = chat_message(ninfer::ChatRole::Assistant, "");
    second.reasoning_content           = "second thought";
    second.parts.front().text          = "second answer";

    const std::vector<fi::ChatMessage> tool_loop{
        chat_message(ninfer::ChatRole::User, "question"), first,
        chat_message(ninfer::ChatRole::Tool, "result one"), second,
        chat_message(ninfer::ChatRole::Tool, "result two")};
    const fi::RenderedChat open    = render_chat(tool_loop);
    const std::size_t first_header = open.text.find(assistant_header);
    int failures =
        check(first_header != std::string::npos && open.rewrite_checkpoint &&
                  open.rewrite_checkpoint->kind ==
                      ninfer::targets::qwen3_6::RewriteCheckpointKind::TurnClosure &&
                  open.rewrite_checkpoint->offset == first_header + assistant_header.size(),
              "tool loop did not retain its first assistant turn-closure boundary");

    fi::ChatRenderOptions preserve;
    preserve.preserve_thinking         = true;
    const fi::RenderedChat preserved   = render_chat(tool_loop, preserve);
    const std::size_t preserved_header = preserved.text.rfind(assistant_header);
    failures += check(preserved_header != std::string::npos && preserved.rewrite_checkpoint &&
                          preserved.rewrite_checkpoint->kind ==
                              ninfer::targets::qwen3_6::RewriteCheckpointKind::ResponseReplay &&
                          preserved.rewrite_checkpoint->offset == preserved.text.size() &&
                          preserved.text.ends_with("<think>\n"),
                      "preserve_thinking did not publish the complete generation prologue");

    preserve.enable_thinking           = false;
    const fi::RenderedChat nonthinking = render_chat(tool_loop, preserve);
    failures += check(nonthinking.rewrite_checkpoint &&
                          nonthinking.rewrite_checkpoint->kind ==
                              ninfer::targets::qwen3_6::RewriteCheckpointKind::ResponseReplay &&
                          nonthinking.rewrite_checkpoint->offset == nonthinking.text.size() &&
                          nonthinking.text.ends_with("<think>\n\n</think>\n\n"),
                      "non-thinking response replay did not retain its complete generation "
                      "prologue");

    std::vector<fi::ChatMessage> next_turn = tool_loop;
    next_turn.push_back(chat_message(ninfer::ChatRole::User, "next question"));
    const fi::RenderedChat next    = render_chat(next_turn);
    const std::size_t final_header = next.text.rfind(assistant_header);
    failures += check(final_header != std::string::npos && next.rewrite_checkpoint &&
                          next.rewrite_checkpoint->kind ==
                              ninfer::targets::qwen3_6::RewriteCheckpointKind::TurnClosure &&
                          next.rewrite_checkpoint->offset == final_header + assistant_header.size(),
                      "new user turn did not move the rewrite boundary to its generation opener");

    fi::ChatRenderOptions no_generation;
    no_generation.add_generation_prompt = false;
    const fi::RenderedChat no_assistant =
        render_chat({chat_message(ninfer::ChatRole::User, "question")}, no_generation);
    failures += check(!no_assistant.rewrite_checkpoint,
                      "boundary-less prompt unexpectedly published a rewrite boundary");

    no_generation.preserve_thinking                     = true;
    const fi::RenderedChat preserved_without_generation = render_chat(tool_loop, no_generation);
    failures += check(!preserved_without_generation.rewrite_checkpoint,
                      "response-replay boundary was published without a generation opener");
    no_generation.preserve_thinking = false;

    const fi::RenderedChat wrapped = render_chat(
        {chat_message(ninfer::ChatRole::User, "question"), first,
         chat_message(ninfer::ChatRole::User, "<tool_response>compat result</tool_response>"),
         second},
        no_generation);
    const std::size_t wrapped_first = wrapped.text.find(assistant_header);
    failures +=
        check(wrapped.rewrite_checkpoint &&
                  wrapped.rewrite_checkpoint->kind ==
                      ninfer::targets::qwen3_6::RewriteCheckpointKind::TurnClosure &&
                  wrapped.rewrite_checkpoint->offset == wrapped_first + assistant_header.size(),
              "bare tool-response wrapper incorrectly advanced the real user turn");
    return failures;
}

int test_official_resource_guards() {
    FrontendResources stale_pad     = resources();
    nlohmann::json tokenizer_config = nlohmann::json::parse(stale_pad.tokenizer_config_json);
    tokenizer_config["pad_token"]   = "<|vision_pad|>";
    stale_pad.tokenizer_config_json = tokenizer_config.dump();
    int failures =
        check(throws_invalid_argument([&] { (void)FrontendFactory::create_component(stale_pad); }),
              "stale Unsloth pad-token policy was accepted");

    FrontendResources mismatched       = resources();
    nlohmann::json mismatched_config   = nlohmann::json::parse(mismatched.tokenizer_config_json);
    mismatched_config["chat_template"] = reasoning_effort_template_source();
    mismatched.tokenizer_config_json   = mismatched_config.dump();
    failures +=
        check(throws_invalid_argument([&] { (void)FrontendFactory::create_component(mismatched); }),
              "different standalone and tokenizer-config chat templates were accepted");

    FrontendResources unknown = resources("{{ messages }}");
    failures +=
        check(throws_invalid_argument([&] { (void)FrontendFactory::create_component(unknown); }),
              "unknown chat template was accepted");

    const Frontend effort_frontend =
        FrontendFactory::create_component(resources(reasoning_effort_template_source()), false);
    const ninfer::PromptCapabilities capabilities = effort_frontend.prompt_capabilities();
    failures +=
        check(capabilities.reasoning_effort.low && capabilities.reasoning_effort.medium &&
                  capabilities.reasoning_effort.xhigh &&
                  capabilities.reasoning_effort.default_effort == ninfer::ReasoningEffort::XHigh,
              "Frontend did not expose capabilities from its loaded chat template");

    return failures;
}

int test_text_and_image_prepare(const Frontend& frontend) {
    ninfer::ChatMessage text_message;
    text_message.role = ninfer::ChatRole::User;
    text_message.parts.push_back(
        ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text, .text = "x", .media = {}});
    ninfer::PromptInput text_input;
    text_input.messages.push_back(std::move(text_message));
    auto text             = frontend.prepare(std::move(text_input));
    const auto& text_data = FrontendFactory::inspect(text);
    const std::vector<ninfer::TokenId> expected{248045, 30, 0, 248046, 32, 248045, 31, 248068, 32};
    int failures =
        check(text_data.token_ids == expected, "text frontend did not render/tokenize chat");
    failures += check(text_data.identity.rewrite_checkpoint &&
                          text_data.identity.rewrite_checkpoint->kind ==
                              ninfer::targets::qwen3_6::RewriteCheckpointKind::TurnClosure &&
                          text_data.identity.rewrite_checkpoint->frontier == 7 &&
                          text_data.starts_in_reasoning && !text_data.has_media(),
                      "text frontend did not preserve prefix/thinking identity");
    failures +=
        check(text_data.position_axis(0).back() == 8 && text_data.position_axis(1).back() == 8 &&
                  text_data.position_axis(2).back() == 8,
              "text frontend did not construct axis-major positions");

    ninfer::ChatMessage preserved_message;
    preserved_message.role = ninfer::ChatRole::User;
    preserved_message.parts.push_back(
        ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text, .text = "x", .media = {}});
    ninfer::PromptInput preserved_input;
    preserved_input.messages.push_back(std::move(preserved_message));
    preserved_input.options.preserve_thinking = true;
    const auto preserved_prompt               = frontend.prepare(std::move(preserved_input));
    const auto& preserved_data                = FrontendFactory::inspect(preserved_prompt);
    failures += check(preserved_data.identity.rewrite_checkpoint &&
                          preserved_data.identity.rewrite_checkpoint->kind ==
                              ninfer::targets::qwen3_6::RewriteCheckpointKind::ResponseReplay &&
                          preserved_data.identity.rewrite_checkpoint->frontier ==
                              preserved_data.token_ids.size(),
                      "preserve-thinking prompt did not publish a prompt-frontier response "
                      "checkpoint");

    ninfer::ChatMessage nonthinking_message;
    nonthinking_message.role = ninfer::ChatRole::User;
    nonthinking_message.parts.push_back(
        ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text, .text = "x", .media = {}});
    ninfer::PromptInput nonthinking_input;
    nonthinking_input.messages.push_back(std::move(nonthinking_message));
    nonthinking_input.options.preserve_thinking = true;
    nonthinking_input.options.enable_thinking   = false;
    const auto nonthinking_prompt               = frontend.prepare(std::move(nonthinking_input));
    const auto& nonthinking_data                = FrontendFactory::inspect(nonthinking_prompt);
    failures += check(nonthinking_data.identity.rewrite_checkpoint &&
                          nonthinking_data.identity.rewrite_checkpoint->kind ==
                              ninfer::targets::qwen3_6::RewriteCheckpointKind::ResponseReplay &&
                          nonthinking_data.identity.rewrite_checkpoint->frontier ==
                              nonthinking_data.token_ids.size() &&
                          !nonthinking_data.starts_in_reasoning,
                      "non-thinking prompt did not publish a prompt-frontier response checkpoint");

    ninfer::MessagePart image;
    image.kind              = ninfer::MessagePartKind::Media;
    image.media.kind        = ninfer::MediaKind::Image;
    image.media.bytes       = gradient_ppm();
    image.media.media_type  = "image/x-portable-pixmap";
    image.media.source_name = "inline.ppm";
    ninfer::ChatMessage image_message;
    image_message.role = ninfer::ChatRole::User;
    image_message.parts.push_back(std::move(image));
    ninfer::PromptInput image_input;
    image_input.messages.push_back(std::move(image_message));
    auto prepared             = frontend.prepare(std::move(image_input));
    const auto& prepared_data = FrontendFactory::inspect(prepared);
    failures += check(prepared_data.has_media() && prepared_data.vision_items.size() == 1,
                      "image frontend did not retain one Vision item");
    if (!prepared_data.vision_items.empty()) {
        const auto& item = prepared_data.vision_items.front();
        failures +=
            check(item.grid.temporal == 1 && item.grid.height == 4 && item.grid.width == 4 &&
                      item.patch_count == 16 && item.content_digest == kGradientDigest &&
                      item.token_spans.size() == 1 && item.token_spans.front().count == 4,
                  "image frontend grid/patch/placeholder geometry is incorrect");
        if (!item.token_spans.empty()) {
            const std::size_t span = item.token_spans.front().begin;
            failures += check(
                prepared_data.position_axis(0)[span] == prepared_data.position_axis(1)[span] &&
                    prepared_data.position_axis(1)[span] == prepared_data.position_axis(2)[span] &&
                    prepared_data.position_axis(1)[span + 2] ==
                        prepared_data.position_axis(1)[span] + 1 &&
                    prepared_data.position_axis(2)[span + 1] ==
                        prepared_data.position_axis(2)[span] + 1,
                "image frontend MRoPE positions are incorrect");
        }
    }
    failures += check(
        prepared_data.patches.size() == 16 * 1536 && prepared_data.prepare.raw_patches == 16 &&
            prepared_data.prepare.vision_tokens == 4 && prepared_data.identity.reusable &&
            prepared_data.identity.rewrite_checkpoint &&
            prepared_data.identity.rewrite_checkpoint->kind ==
                ninfer::targets::qwen3_6::RewriteCheckpointKind::TurnClosure &&
            prepared_data.identity.rewrite_checkpoint->frontier < prepared_data.token_ids.size(),
        "image frontend did not own the expected patch payload and identity");
    if (prepared_data.patches.size() == 16 * 1536) {
        failures += check(near(prepared_data.patches[0], -1.0F) &&
                              near(prepared_data.patches[1], 1.0F / 127.5F - 1.0F) &&
                              near(prepared_data.patches[256], -1.0F) &&
                              near(prepared_data.patches[1536], 16.0F / 127.5F - 1.0F),
                          "image frontend patch normalization/order is incorrect");
    }
    return failures;
}

int test_multimodal_prompt_over_removed_32k_cap(const Frontend& frontend) {
    const std::string long_text(40'000, 'x');
    const std::uint32_t counted =
        frontend.count_tokens(image_text_input(gradient_ppm(), long_text, "long-context.ppm"));
    const auto prepared =
        frontend.prepare(image_text_input(gradient_ppm(), long_text, "long-context.ppm"));
    const auto& data = FrontendFactory::inspect(prepared);

    int failures = check(counted > 32'768 && data.token_ids.size() == counted,
                         "multimodal prompt retained the removed 32K frontend token cap");
    failures += check(data.has_media() && data.vision_items.size() == 1,
                      "long multimodal prompt lost its Vision item");
    return failures;
}

int test_attention_pairs_are_diagnostic(const Frontend& frontend) {
    constexpr std::uint64_t kRemovedAttentionPairLimit = 128ULL * 1024ULL * 1024ULL;
    const auto prepared =
        frontend.prepare(image_text_input(block_ppm(2048, 1536, 127), {}, "large-grid.ppm"));
    const auto& data = FrontendFactory::inspect(prepared);

    int failures = check(data.prepare.attention_pairs > kRemovedAttentionPairLimit,
                         "test image did not exceed the removed attention-pair threshold");
    failures += check(data.prepare.raw_patches == 12'288 && data.prepare.vision_tokens == 3'072 &&
                          data.vision_items.size() == 1,
                      "large image did not retain its expected Vision geometry");
    return failures;
}

int test_video_prepare(const Frontend& frontend) {
    ninfer::MessagePart video;
    video.kind              = ninfer::MessagePartKind::Media;
    video.media.kind        = ninfer::MediaKind::Video;
    video.media.bytes       = gradient_ppm();
    video.media.media_type  = "image/x-portable-pixmap";
    video.media.source_name = "single-frame.ppm";
    ninfer::ChatMessage message;
    message.role = ninfer::ChatRole::User;
    message.parts.push_back(std::move(video));
    ninfer::PromptInput input;
    input.messages.push_back(std::move(message));

    auto prepared             = frontend.prepare(std::move(input));
    const auto& prepared_data = FrontendFactory::inspect(prepared);
    int failures = check(prepared_data.vision_items.size() == 1 && prepared_data.has_media(),
                         "video frontend did not retain one Vision item");
    if (!prepared_data.vision_items.empty()) {
        const auto& item = prepared_data.vision_items.front();
        failures +=
            check(item.modality == ninfer::targets::qwen3_6::PromptModality::Video &&
                      item.grid.temporal == 1 && item.grid.height == 4 && item.grid.width == 4 &&
                      item.patch_count == 16 && item.content_digest == kGradientDigest &&
                      item.timestamps.size() == 1 && item.timestamps.front() == 0.0 &&
                      item.token_spans.size() == 1 && item.token_spans.front().count == 4,
                  "video frontend temporal/grid/placeholder metadata is incorrect");
    }
    failures +=
        check(prepared_data.patches.size() == 16 * 1536 &&
                  near(prepared_data.patches[0], prepared_data.patches[256]) &&
                  prepared_data.prepare.raw_patches == 16 &&
                  prepared_data.prepare.vision_tokens == 4 && prepared_data.identity.reusable,
              "video frontend did not duplicate the odd temporal frame correctly");
    return failures;
}

int test_cross_round_stop(const Frontend& frontend) {
    auto prompt = frontend.prepare_tokens({0});
    ninfer::StopPolicy stop;
    stop.strings.push_back(ninfer::StopString{.text = "STOP"});
    auto session = frontend.make_output_session(prompt, stop);

    const auto first_decision =
        session.preview(std::array<ninfer::TokenId, 1>{1}, 2, ninfer::FinishReason::OutputLimit);
    int failures     = check(first_decision.accepted_tokens == 1 && !first_decision.finished(),
                             "cross-round stop ended before the stop string was complete");
    const auto first = session.commit_preview();
    failures += check(channel_text(first, ninfer::OutputChannel::Content) == "hello",
                      "cross-round stop did not retain the ambiguous suffix");

    const auto second_decision =
        session.preview(std::array<ninfer::TokenId, 1>{2}, 1, ninfer::FinishReason::OutputLimit);
    failures += check(second_decision.accepted_tokens == 1 &&
                          second_decision.finish_reason == ninfer::FinishReason::StopString,
                      "cross-round stop did not select the exact terminal token prefix");
    const auto second = session.commit_preview();
    failures += check(second.empty(), "stop marker or same-token suffix leaked to output");
    return failures;
}

int test_same_token_stop_priority(const Frontend& frontend) {
    auto prompt = frontend.prepare_tokens({0});
    ninfer::StopPolicy stop;
    stop.strings = {
        ninfer::StopString{.text = "tail", .include_in_output = true},
        ninfer::StopString{.text = "OPtail"},
        ninfer::StopString{.text = "OP", .include_in_output = true},
    };
    auto session = frontend.make_output_session(prompt, stop);
    const auto decision =
        session.preview(std::array<ninfer::TokenId, 1>{2}, 2, ninfer::FinishReason::OutputLimit);
    int failures      = check(decision.accepted_tokens == 1 &&
                                  decision.finish_reason == ninfer::FinishReason::StopString,
                              "same-token stop strings did not select a terminal prefix");
    const auto output = session.commit_preview();
    failures += check(output.empty(),
                      "same-token stops did not prefer the earliest byte and declaration order");
    return failures;
}

int test_terminal_flush(const Frontend& frontend) {
    auto prompt = frontend.prepare_tokens({0});
    ninfer::StopPolicy stop;
    stop.strings.push_back(ninfer::StopString{.text = "STOP"});
    auto session = frontend.make_output_session(prompt, stop);

    const auto first_decision =
        session.preview(std::array<ninfer::TokenId, 1>{1}, 2, ninfer::FinishReason::OutputLimit);
    int failures     = check(first_decision.accepted_tokens == 1 && !first_decision.finished(),
                             "terminal flush setup unexpectedly finished");
    const auto first = session.commit_preview();
    failures += check(channel_text(first, ninfer::OutputChannel::Content) == "hello",
                      "terminal flush setup did not retain the possible stop suffix");

    const auto terminal = session.preview_terminal(ninfer::FinishReason::Cancelled);
    failures += check(terminal.accepted_tokens == 0 &&
                          terminal.finish_reason == ninfer::FinishReason::Cancelled,
                      "between-round terminal preview returned the wrong decision");
    const auto flushed = session.commit_preview();
    failures += check(channel_text(flushed, ninfer::OutputChannel::Content) == "ST",
                      "between-round terminal preview lost the pending stop suffix");
    return failures;
}

int test_reasoning_split(const Frontend& frontend) {
    ninfer::ChatMessage message;
    message.role = ninfer::ChatRole::User;
    message.parts.push_back(
        ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text, .text = "x", .media = {}});
    ninfer::PromptInput input;
    input.messages.push_back(std::move(message));
    input.options.add_generation_prompt = true;
    input.options.enable_thinking       = true;
    auto prompt                         = frontend.prepare(std::move(input));
    auto session                        = frontend.make_output_session(prompt, {});
    int failures = check(session.in_reasoning(), "thinking output session did not start open");
    auto raw_session =
        frontend.make_output_session(prompt, {}, ninfer::OutputOptions{.raw = true});
    failures += check(!raw_session.in_reasoning(), "raw output session entered reasoning state");
    const std::array<ninfer::TokenId, 2> tokens{3, 4};
    const auto decision = session.preview(tokens, 3, ninfer::FinishReason::OutputLimit);
    failures += check(decision.accepted_tokens == 2 && !decision.finished(),
                      "reasoning close unexpectedly made a nonterminal preview terminal");
    const auto output   = session.commit_preview();
    failures += check(channel_text(output, ninfer::OutputChannel::Reasoning) == "thought",
                      "reasoning channel did not remove the close marker");
    failures += check(channel_text(output, ninfer::OutputChannel::Content) == "answer",
                      "content channel did not strip the post-thinking separator");
    failures += check(session.reasoning_tokens() == 2,
                      "reasoning token usage did not count accepted reasoning tokens exactly");
    failures += check(!session.in_reasoning(),
                      "thinking output session remained open after the close marker");

    ninfer::ChatMessage direct_message;
    direct_message.role = ninfer::ChatRole::User;
    direct_message.parts.push_back(
        ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text, .text = "x", .media = {}});
    ninfer::PromptInput direct_input;
    direct_input.messages.push_back(std::move(direct_message));
    direct_input.options.add_generation_prompt = true;
    direct_input.options.enable_thinking       = false;
    auto direct_prompt                         = frontend.prepare(std::move(direct_input));
    auto direct_session                        = frontend.make_output_session(direct_prompt, {});
    failures += check(!direct_session.in_reasoning(),
                      "non-thinking output session entered reasoning state");
    return failures;
}

int test_structured_model_stop_eligibility(const Frontend& frontend) {
    ninfer::ChatMessage message;
    message.role = ninfer::ChatRole::User;
    message.parts.push_back(
        ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text, .text = "x", .media = {}});
    ninfer::PromptInput thinking_input;
    thinking_input.messages.push_back(std::move(message));
    thinking_input.options.enable_thinking = true;
    auto thinking_prompt = frontend.prepare(std::move(thinking_input));
    auto thinking        = frontend.make_output_session(thinking_prompt, {});

    int failures = check(!thinking.model_stop_tokens_allowed(),
                         "model stop was eligible during reasoning");
    auto decision = thinking.preview(std::array<ninfer::TokenId, 1>{248069}, 8,
                                     ninfer::FinishReason::OutputLimit);
    failures += check(!decision.finished() && !decision.reject_generated_round,
                      "reasoning close was rejected");
    (void)thinking.commit_preview();
    failures += check(!thinking.in_reasoning() && !thinking.model_stop_tokens_allowed(),
                      "model stop was eligible before post-reasoning content");

    decision = thinking.preview(std::array<ninfer::TokenId, 1>{14}, 7,
                                ninfer::FinishReason::OutputLimit);
    (void)thinking.commit_preview();
    failures += check(!thinking.model_stop_tokens_allowed(),
                      "whitespace made the post-reasoning model stop eligible");

    decision = thinking.preview(std::array<ninfer::TokenId, 1>{6}, 6,
                                ninfer::FinishReason::OutputLimit);
    failures += check(decision.reject_generated_round && decision.accepted_tokens == 0 &&
                          !decision.finished(),
                      "invalid empty-answer model stop was not rejected");
    thinking.discard_preview();
    failures += check(!thinking.model_stop_tokens_allowed(),
                      "discarding an invalid stop changed output state");

    decision = thinking.preview(std::array<ninfer::TokenId, 1>{15}, 6,
                                ninfer::FinishReason::OutputLimit);
    (void)thinking.commit_preview();
    failures += check(thinking.model_stop_tokens_allowed(),
                      "answer content did not make the model stop eligible");
    decision = thinking.preview(std::array<ninfer::TokenId, 1>{6}, 5,
                                ninfer::FinishReason::OutputLimit);
    failures += check(decision.finished() &&
                          decision.finish_reason == ninfer::FinishReason::StopToken &&
                          !decision.reject_generated_round,
                      "established answer did not stop normally");
    (void)thinking.commit_preview();

    ninfer::ChatMessage tool_message;
    tool_message.role = ninfer::ChatRole::User;
    tool_message.parts.push_back(
        ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text, .text = "x", .media = {}});
    ninfer::ChatMessage assistant_call;
    assistant_call.role = ninfer::ChatRole::Assistant;
    assistant_call.parts.push_back(
        ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text, .text = "", .media = {}});
    assistant_call.tool_calls.push_back(
        ninfer::ToolCall{.id = "", .name = "f", .arguments_json = "{}"});
    ninfer::ChatMessage next_message;
    next_message.role = ninfer::ChatRole::User;
    next_message.parts.push_back(
        ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text, .text = "x", .media = {}});
    ninfer::PromptInput tool_input;
    tool_input.messages.push_back(std::move(tool_message));
    tool_input.messages.push_back(std::move(assistant_call));
    tool_input.messages.push_back(std::move(next_message));
    tool_input.options.enable_thinking = false;
    auto tool_prompt = frontend.prepare(std::move(tool_input));
    auto tool = frontend.make_output_session(tool_prompt, {});
    failures += check(!tool.model_stop_tokens_allowed(),
                      "tools-enabled empty output allowed a model stop");

    ninfer::ChatMessage ignored_user_call;
    ignored_user_call.role = ninfer::ChatRole::User;
    ignored_user_call.parts.push_back(
        ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text, .text = "x", .media = {}});
    ignored_user_call.tool_calls.push_back(
        ninfer::ToolCall{.id = "", .name = "f", .arguments_json = "{}"});
    ninfer::PromptInput ignored_user_call_input;
    ignored_user_call_input.messages.push_back(std::move(ignored_user_call));
    ignored_user_call_input.options.enable_thinking = false;
    auto ignored_user_call_prompt = frontend.prepare(std::move(ignored_user_call_input));
    auto ignored_user_call_output = frontend.make_output_session(ignored_user_call_prompt, {});
    failures += check(ignored_user_call_output.model_stop_tokens_allowed(),
                      "ignored non-assistant tool_calls enabled structured tool output");

    decision = tool.preview(std::array<ninfer::TokenId, 1>{22}, 16,
                            ninfer::FinishReason::OutputLimit);
    (void)tool.commit_preview();
    failures += check(tool.model_stop_tokens_allowed(),
                      "ordinary pre-tool content did not allow a model stop");

    const std::array<ninfer::TokenId, 2> partial_and_stop{16, 6};
    decision = tool.preview(partial_and_stop, 15, ninfer::FinishReason::OutputLimit);
    failures += check(decision.reject_generated_round && decision.accepted_tokens == 0,
                      "same-round partial tool opener did not reject its model stop");
    tool.discard_preview();
    failures += check(tool.model_stop_tokens_allowed(),
                      "rejected tool round changed committed structure");

    decision = tool.preview(std::array<ninfer::TokenId, 1>{16}, 15,
                            ninfer::FinishReason::OutputLimit);
    (void)tool.commit_preview();
    failures += check(!tool.model_stop_tokens_allowed(),
                      "ambiguous tool opener allowed a model stop");
    decision = tool.preview(std::array<ninfer::TokenId, 1>{17}, 14,
                            ninfer::FinishReason::OutputLimit);
    (void)tool.commit_preview();
    failures += check(!tool.model_stop_tokens_allowed(),
                      "open tool call allowed a model stop");
    decision = tool.preview(std::array<ninfer::TokenId, 2>{18, 19}, 13,
                            ninfer::FinishReason::OutputLimit);
    (void)tool.commit_preview();
    failures += check(!tool.model_stop_tokens_allowed(),
                      "incomplete tool call allowed a model stop");
    decision = tool.preview(std::array<ninfer::TokenId, 1>{20}, 11,
                            ninfer::FinishReason::OutputLimit);
    (void)tool.commit_preview();
    failures += check(tool.model_stop_tokens_allowed(),
                      "complete tool call did not allow a model stop");

    const std::array<ninfer::TokenId, 5> parallel_call_and_stop{21, 18, 19, 20, 6};
    decision = tool.preview(parallel_call_and_stop, 10, ninfer::FinishReason::OutputLimit);
    failures += check(decision.finished() && decision.accepted_tokens == 5 &&
                          decision.finish_reason == ninfer::FinishReason::StopToken,
                      "complete parallel tool call did not stop in the same round");
    (void)tool.commit_preview();

    auto marker_literal = frontend.make_output_session(tool_prompt, {});
    decision = marker_literal.preview(std::array<ninfer::TokenId, 5>{21, 18, 21, 19, 20}, 6,
                                      ninfer::FinishReason::OutputLimit);
    (void)marker_literal.commit_preview();
    failures += check(marker_literal.model_stop_tokens_allowed(),
                      "tool opener text inside a parameter was treated as a nested call");

    ninfer::StopPolicy no_model_defaults;
    no_model_defaults.include_model_defaults = false;
    auto no_defaults = frontend.make_output_session(tool_prompt, no_model_defaults);
    decision = no_defaults.preview(std::array<ninfer::TokenId, 1>{22}, 4,
                                   ninfer::FinishReason::OutputLimit);
    (void)no_defaults.commit_preview();
    decision = no_defaults.preview(partial_and_stop, 3, ninfer::FinishReason::OutputLimit);
    failures += check(decision.reject_generated_round && decision.accepted_tokens == 0,
                      "protected model stop depended on caller default-stop policy");
    no_defaults.discard_preview();

    auto raw = frontend.make_output_session(
        tool_prompt, {},
        ninfer::OutputOptions{.raw = true, .preserve_special_tokens = false});
    failures += check(raw.model_stop_tokens_allowed(),
                      "raw output unexpectedly applied structured stop eligibility");
    decision = raw.preview(std::array<ninfer::TokenId, 1>{6}, 1,
                           ninfer::FinishReason::OutputLimit);
    failures += check(decision.finished() && !decision.reject_generated_round,
                      "raw output rejected its model stop");
    (void)raw.commit_preview();
    return failures;
}

int test_utf8_and_hidden_eos(const Frontend& frontend) {
    auto prompt             = frontend.prepare_tokens({0});
    auto session            = frontend.make_output_session(prompt, {});
    int failures            = 0;
    std::uint32_t remaining = 4;
    for (const ninfer::TokenId token : {10, 11}) {
        const auto decision = session.preview(std::array<ninfer::TokenId, 1>{token}, remaining,
                                              ninfer::FinishReason::OutputLimit);
        failures += check(decision.accepted_tokens == 1 && !decision.finished(),
                          "partial UTF-8 token unexpectedly ended generation");
        const auto output = session.commit_preview();
        remaining -= decision.accepted_tokens;
        failures += check(output.empty(), "partial UTF-8 codepoint was published");
    }
    const auto complete_decision = session.preview(std::array<ninfer::TokenId, 1>{12}, remaining,
                                                   ninfer::FinishReason::OutputLimit);
    failures += check(complete_decision.accepted_tokens == 1 && !complete_decision.finished(),
                      "complete UTF-8 token unexpectedly ended generation");
    const auto complete = session.commit_preview();
    failures += check(channel_text(complete, ninfer::OutputChannel::Content) == "中",
                      "UTF-8 codepoint was not published when complete");

    auto eos_prompt         = frontend.prepare_tokens({0});
    auto eos_session        = frontend.make_output_session(eos_prompt, {});
    const auto eos_decision = eos_session.preview(std::array<ninfer::TokenId, 1>{6}, 2,
                                                  ninfer::FinishReason::OutputLimit);
    failures += check(eos_decision.accepted_tokens == 1 &&
                          eos_decision.finish_reason == ninfer::FinishReason::StopToken,
                      "default EOS token did not end generation");
    const auto eos = eos_session.commit_preview();
    failures += check(eos.empty(), "default EOS token was published");

    auto raw_prompt  = frontend.prepare_tokens({0});
    auto raw_session = frontend.make_output_session(
        raw_prompt, {}, ninfer::OutputOptions{.raw = true, .preserve_special_tokens = false});
    const auto raw_eos_decision = raw_session.preview(std::array<ninfer::TokenId, 1>{6}, 2,
                                                      ninfer::FinishReason::OutputLimit);
    failures += check(raw_eos_decision.accepted_tokens == 1 &&
                          raw_eos_decision.finish_reason == ninfer::FinishReason::StopToken,
                      "raw EOS token did not end generation");
    const auto raw_eos = raw_session.commit_preview();
    failures += check(channel_text(raw_eos, ninfer::OutputChannel::Content) == "<eos>",
                      "raw output did not preserve the terminal special token");
    return failures;
}

int test_disabled_vision() {
    const Frontend frontend = FrontendFactory::create_component(resources(), false);
    int failures = check(throws_invalid_argument([&] { (void)frontend.prepare(image_input()); }),
                         "Vision-disabled frontend accepted media during prepare");
    failures += check(throws_invalid_argument([&] { (void)frontend.count_tokens(image_input()); }),
                      "Vision-disabled frontend accepted media during token counting");

    ninfer::ChatMessage message;
    message.role = ninfer::ChatRole::User;
    message.parts.push_back(
        ninfer::MessagePart{.kind = ninfer::MessagePartKind::Text, .text = "x", .media = {}});
    ninfer::PromptInput input;
    input.messages.push_back(std::move(message));
    failures += check(frontend.prepare(std::move(input)).summary().prompt_tokens != 0,
                      "Vision-disabled frontend rejected a text prompt");
    return failures;
}

} // namespace

int run_encode_bench() {
    const fi::Tokenizer& tokenizer = official_tokenizer();
    const std::string paragraph =
        "Write a concise systems explanation of paged KV cache reuse, speculative decoding, "
        "and why host-side tokenization can hide under GPU prefill at 32k tokens. Include "
        "ASCII punctuation, numbers 0123456789, and a few Chinese characters: 缓存复用。\n";

    auto time_ms = [](auto fn, int repeats) {
        (void)fn();
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < repeats; ++i) { (void)fn(); }
        const auto elapsed = std::chrono::steady_clock::now() - start;
        return std::chrono::duration<double, std::milli>(elapsed).count() / repeats;
    };

    auto json_object = [](std::string path, std::string content) {
        nlohmann::json object = {{"path", std::move(path)}, {"content", std::move(content)}};
        return object.dump();
    };
    auto blob = [](std::size_t bytes) {
        std::string text;
        text.reserve(bytes);
        while (text.size() < bytes) {
            text += "fn handle_request(slot: u32, tokens: &[u32]) { /* kv-page ";
            text += std::to_string(text.size());
            text += " */ }\n";
        }
        text.resize(bytes);
        return text;
    };
    auto tool_schema = [](int count) {
        std::vector<std::string> tools;
        tools.reserve(static_cast<std::size_t>(count));
        for (int i = 0; i < count; ++i) {
            tools.push_back(nlohmann::json{{"type", "function"},
                                           {"function",
                                            {{"name", "read_file_" + std::to_string(i)},
                                             {"description", "Read a workspace file"},
                                             {"parameters",
                                              {{"type", "object"},
                                               {"properties",
                                                {{"path", {{"type", "string"}}},
                                                 {"content", {{"type", "string"}}}}},
                                               {"required", nlohmann::json::array({"path"})}}}}}}
                               .dump());
        }
        return tools;
    };
    auto plain_messages = [&](int target_tokens) {
        const std::vector<int> unit_ids = tokenizer.encode(paragraph);
        const int unit_n                = std::max(1, static_cast<int>(unit_ids.size()));
        const int copies                = (target_tokens + unit_n - 1) / unit_n;
        std::string body;
        body.reserve(paragraph.size() * static_cast<std::size_t>(copies));
        for (int i = 0; i < copies; ++i) { body += paragraph; }
        return std::vector<fi::ChatMessage>{chat_message(ninfer::ChatRole::User, std::move(body))};
    };
    auto tool_messages = [&](int calls, std::size_t content_bytes, int history_turns) {
        std::vector<fi::ChatMessage> messages;
        messages.push_back(chat_message(ninfer::ChatRole::User, "Refactor the listed files."));
        for (int turn = 0; turn < history_turns; ++turn) {
            fi::ChatMessage assistant = chat_message(ninfer::ChatRole::Assistant, "");
            assistant.tool_calls.reserve(static_cast<std::size_t>(calls));
            for (int i = 0; i < calls; ++i) {
                assistant.tool_calls.push_back(
                    {.id             = "",
                     .name           = "read_file_" + std::to_string(i),
                     .arguments_json = json_object("src/mod_" + std::to_string(turn) + "_" +
                                                       std::to_string(i) + ".rs",
                                                   blob(content_bytes))});
            }
            messages.push_back(std::move(assistant));
            for (int i = 0; i < calls; ++i) {
                messages.push_back(chat_message(ninfer::ChatRole::Tool, blob(content_bytes / 4)));
            }
            messages.push_back(chat_message(ninfer::ChatRole::User, "Continue with the next batch."));
        }
        return messages;
    };

    struct Case {
        const char* name;
        std::vector<fi::ChatMessage> messages;
        fi::ChatRenderOptions options;
        int repeats;
    };
    std::vector<Case> cases;
    fi::ChatRenderOptions with_tools;
    with_tools.tool_jsons = tool_schema(16);
    fi::ChatRenderOptions tools_100_opts;
    tools_100_opts.tool_jsons = tool_schema(100);
    fi::ChatRenderOptions tools_200_opts;
    tools_200_opts.tool_jsons = tool_schema(200);
    cases.push_back({"plain_2k", plain_messages(2048), {}, 24});
    cases.push_back({"plain_8k", plain_messages(8192), {}, 8});
    cases.push_back({"plain_16k", plain_messages(16384), {}, 6});
    cases.push_back({"plain_32k", plain_messages(32768), {}, 4});
    cases.push_back({"plain_150k", plain_messages(150000), {}, 3});
    cases.push_back({"large_tools", tool_messages(16, 4 * 1024, 1), with_tools, 6});
    cases.push_back({"tools_100_parallel", tool_messages(100, 4 * 1024, 1), tools_100_opts, 4});
    cases.push_back({"tools_200_150k", tool_messages(200, 1536, 1), tools_200_opts, 3});
    cases.push_back({"tools_100_turns", tool_messages(4, 4 * 1024, 25), tools_100_opts, 4});
    cases.push_back({"massive_tools", tool_messages(32, 32 * 1024, 2), with_tools, 3});
    cases.push_back({"tool_loop_history", tool_messages(4, 1024, 8), with_tools, 6});

    std::cerr << std::fixed << std::setprecision(3);
    std::cerr << "case tokens bytes render_ms encode_ms prepare_ms two_pass_ms decode_ms "
                 "per_token_decode_ms checkpoint\n";
    for (Case& test : cases) {
        const double render_ms =
            time_ms([&] { return render_chat(test.messages, test.options); }, test.repeats);
        const fi::RenderedChat rendered = render_chat(test.messages, test.options);
        const double encode_ms =
            time_ms([&] { return fi::encode_rendered_chat(tokenizer, rendered); }, test.repeats);
        if (!rendered.rewrite_checkpoint) {
            std::cerr << test.name << " missing rewrite checkpoint\n";
            return 1;
        }
        const std::size_t checkpoint = rendered.rewrite_checkpoint->offset;
        const double full_ms = time_ms([&] { return tokenizer.encode(rendered.text); }, test.repeats);
        const double prefix_ms = time_ms(
            [&] {
                return tokenizer.encode(std::string_view(rendered.text).substr(0, checkpoint));
            },
            test.repeats);
        const fi::EncodedChat encoded = fi::encode_rendered_chat(tokenizer, rendered);
        const double decode_ms =
            time_ms([&] { return tokenizer.decode(encoded.input_ids); }, test.repeats);
        const double per_token_ms = time_ms(
            [&] {
                std::string out;
                out.reserve(rendered.text.size());
                for (const int id : encoded.input_ids) { out += tokenizer.decode_token_bytes(id); }
                return out;
            },
            test.repeats);
        std::cerr << test.name << ' ' << encoded.input_ids.size() << ' ' << rendered.text.size()
                  << ' ' << render_ms << ' ' << encode_ms << ' ' << (render_ms + encode_ms) << ' '
                  << (full_ms + prefix_ms) << ' ' << decode_ms << ' ' << per_token_ms << ' '
                  << (encoded.rewrite_checkpoint ? encoded.rewrite_checkpoint->frontier : 0)
                  << '\n';
    }
    return 0;
}

int main() {
    if (std::getenv("NINFER_BENCH_ENCODE") != nullptr) { return run_encode_bench(); }
    const FrontendResources owned = resources();
    const Frontend frontend       = FrontendFactory::create_component(owned);
    int failures                  = 0;
    failures += test_official_tokenizer_merge();
    failures += test_official_chat_template();
    failures += test_ordered_instruction_turns();
    failures += test_reasoning_effort_chat_template();
    failures += test_reasoning_effort_empty_history_think();
    failures += test_rewrite_checkpoint_trace();
    failures += test_official_resource_guards();
    failures += test_text_and_image_prepare(frontend);
    failures += test_multimodal_prompt_over_removed_32k_cap(frontend);
    failures += test_attention_pairs_are_diagnostic(frontend);
    failures += test_video_prepare(frontend);
    failures += test_cross_round_stop(frontend);
    failures += test_same_token_stop_priority(frontend);
    failures += test_terminal_flush(frontend);
    failures += test_reasoning_split(frontend);
    failures += test_structured_model_stop_eligibility(frontend);
    failures += test_utf8_and_hidden_eos(frontend);
    failures += test_disabled_vision();
    return failures == 0 ? 0 : 1;
}
