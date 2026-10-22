#include <ninfer/targets/qwen3_6/frontend.h>

#include <ninfer/targets/qwen3_6/frontend_resources.h>
#include <ninfer/targets/qwen3_6/prepared_prompt.h>
#include <ninfer/targets/qwen3_6/generation_recovery.h>

#include "targets/qwen3_6/impl/frontend/chat_template.h"
#include "targets/qwen3_6/impl/frontend/encoded_history_cache.h"
#include "targets/qwen3_6/impl/frontend/processor.h"
#include "targets/qwen3_6/impl/frontend/test_access.h"
#include "targets/qwen3_6/impl/frontend/tokenizer.h"
#include "targets/qwen3_6/impl/frontend/token_grammar.h"
#include "text/unicode.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_6 {
namespace {

using Json   = nlohmann::json;
using Clock  = std::chrono::steady_clock;
namespace fi = frontend_internal;

constexpr std::string_view kThinkClose      = "</think>";
constexpr std::string_view kToolOpen        = "<tool_call>";
constexpr std::string_view kToolClose       = "</tool_call>";
constexpr std::string_view kUtf8Replacement = "\xef\xbf\xbd";
constexpr double kRescaleFactor             = 1.0 / 255.0;
constexpr double kVideoFps                  = 2.0;
constexpr int kVideoMinFrames               = 4;
constexpr int kVideoMaxFrames               = 768;

constexpr std::array<std::pair<std::string_view, TokenId>, 4> kVisionSpecialTokens = {{
    {"<|vision_start|>", 248053},
    {"<|vision_end|>", 248054},
    {"<|image_pad|>", 248056},
    {"<|video_pad|>", 248057},
}};

constexpr std::array<std::pair<std::string_view, TokenId>, 7> kConfigOnlyTokens = {{
    {"<|audio_start|>", 248070},
    {"<|audio_end|>", 248071},
    {"<tts_pad>", 248072},
    {"<tts_text_bos>", 248073},
    {"<tts_text_eod>", 248074},
    {"<tts_text_bos_single>", 248075},
    {"<|audio_pad|>", 248076},
}};

Json parse_resource_json(std::string_view bytes, std::string_view name) {
    try {
        Json result = Json::parse(bytes);
        if (!result.is_object()) {
            throw std::invalid_argument(std::string(name) + " must contain a JSON object");
        }
        return result;
    } catch (const nlohmann::json::exception& error) {
        throw std::invalid_argument("malformed " + std::string(name) + ": " + error.what());
    }
}

std::int64_t require_integer(const Json& object, std::string_view field,
                             std::string_view resource) {
    const std::string key(field);
    if (!object.contains(key) || !object.at(key).is_number_integer()) {
        throw std::invalid_argument(std::string(resource) + "." + key + " must be an integer");
    }
    return object.at(key).get<std::int64_t>();
}

double require_number(const Json& object, std::string_view field, std::string_view resource) {
    const std::string key(field);
    if (!object.contains(key) || !object.at(key).is_number()) {
        throw std::invalid_argument(std::string(resource) + "." + key + " must be a number");
    }
    return object.at(key).get<double>();
}

double number_or_default(const Json& object, std::string_view field, std::string_view resource,
                         double default_value) {
    const std::string key(field);
    return object.contains(key) ? require_number(object, field, resource) : default_value;
}

std::int64_t integer_or_default(const Json& object, std::string_view field,
                                std::string_view resource, std::int64_t default_value) {
    const std::string key(field);
    return object.contains(key) ? require_integer(object, field, resource) : default_value;
}

const Json& require_object(const Json& object, std::string_view field, std::string_view resource) {
    const std::string key(field);
    if (!object.contains(key) || !object.at(key).is_object()) {
        throw std::invalid_argument(std::string(resource) + "." + key + " must be an object");
    }
    return object.at(key);
}

std::uint64_t positive_u64(std::int64_t value, std::string_view field) {
    if (value <= 0) { throw std::invalid_argument(std::string(field) + " must be positive"); }
    return static_cast<std::uint64_t>(value);
}

void validate_pixel_pipeline(const Json& config, std::string_view resource) {
    if (require_integer(config, "patch_size", resource) != 16 ||
        require_integer(config, "temporal_patch_size", resource) != 2 ||
        require_integer(config, "merge_size", resource) != 2) {
        throw std::invalid_argument(std::string(resource) +
                                    " does not match the compiled Vision patch geometry");
    }
    const auto require_half_triplet = [&](std::string_view field) {
        const std::string key(field);
        if (!config.contains(key) || !config.at(key).is_array() || config.at(key).size() != 3) {
            throw std::invalid_argument(std::string(resource) + "." + key +
                                        " must contain three values");
        }
        for (const Json& value : config.at(key)) {
            if (!value.is_number() || value.get<double>() != 0.5) {
                throw std::invalid_argument(std::string(resource) + "." + key +
                                            " does not match the compiled normalization");
            }
        }
    };
    require_half_triplet("image_mean");
    require_half_triplet("image_std");
    if (number_or_default(config, "rescale_factor", resource, kRescaleFactor) != kRescaleFactor) {
        throw std::invalid_argument(std::string(resource) +
                                    ".rescale_factor does not match the compiled normalization");
    }
}

fi::ProcessorOptions processor_options(const FrontendResources& resources) {
    const Json image =
        parse_resource_json(resources.preprocessor_config_json, "preprocessor_config.json");
    const Json video = parse_resource_json(resources.video_preprocessor_config_json,
                                           "video_preprocessor_config.json");
    validate_pixel_pipeline(image, "preprocessor_config.json");
    validate_pixel_pipeline(video, "video_preprocessor_config.json");

    const Json& image_size = require_object(image, "size", "preprocessor_config.json");
    const Json& video_size = require_object(video, "size", "video_preprocessor_config.json");

    fi::ProcessorOptions options;
    options.image_min_pixels =
        positive_u64(require_integer(image_size, "shortest_edge", "preprocessor_config.json.size"),
                     "image shortest_edge");
    options.image_max_pixels =
        positive_u64(require_integer(image_size, "longest_edge", "preprocessor_config.json.size"),
                     "image longest_edge");
    options.video_min_pixels = positive_u64(
        require_integer(video_size, "shortest_edge", "video_preprocessor_config.json.size"),
        "video shortest_edge");
    options.video_max_pixels = positive_u64(
        require_integer(video_size, "longest_edge", "video_preprocessor_config.json.size"),
        "video longest_edge");
    options.video_fps =
        number_or_default(video, "fps", "video_preprocessor_config.json", kVideoFps);
    options.video_min_frames = static_cast<int>(
        integer_or_default(video, "min_frames", "video_preprocessor_config.json", kVideoMinFrames));
    options.video_max_frames = static_cast<int>(
        integer_or_default(video, "max_frames", "video_preprocessor_config.json", kVideoMaxFrames));
    if (options.video_fps != kVideoFps || options.video_min_frames != kVideoMinFrames ||
        options.video_max_frames != kVideoMaxFrames) {
        throw std::invalid_argument(
            "video_preprocessor_config.json does not match registered sampling defaults");
    }

    return options;
}

void validate_tokenizer_config(const FrontendResources& resources) {
    const Json tokenizer_config =
        parse_resource_json(resources.tokenizer_config_json, "tokenizer_config.json");
    if (tokenizer_config.value("add_bos_token", true) ||
        tokenizer_config.value("add_prefix_space", true)) {
        throw std::invalid_argument(
            "tokenizer_config.json does not match Qwen3.6 tokenizer prefix semantics");
    }
    if (!tokenizer_config.contains("pad_token") || !tokenizer_config.at("pad_token").is_string() ||
        tokenizer_config.at("pad_token").get<std::string>() != "<|endoftext|>") {
        throw std::invalid_argument(
            "tokenizer_config.json does not use the official <|endoftext|> pad token");
    }
    if (!tokenizer_config.contains("chat_template") ||
        !tokenizer_config.at("chat_template").is_string()) {
        throw std::invalid_argument(
            "tokenizer_config.json.chat_template must contain the loaded chat template");
    }
    if (tokenizer_config.at("chat_template").get_ref<const std::string&>() !=
        resources.chat_template_jinja) {
        throw std::invalid_argument(
            "tokenizer_config.json.chat_template does not match frontend/chat_template.jinja");
    }
}

fi::CompiledChatTemplate compile_chat_template(const FrontendResources& resources) {
    validate_tokenizer_config(resources);
    return fi::CompiledChatTemplate::resolve(resources.chat_template_jinja);
}

[[noreturn]] void throw_processor_error(const fi::ProcessorError& error) {
    switch (error.kind()) {
    case fi::ProcessorErrorKind::BudgetExceeded:
        throw RequestError(RequestErrorKind::MediaBudgetExceeded, error.what());
    }
    throw std::logic_error("unknown Qwen3.6 processor error kind");
}

void validate_registered_tokenizer(const fi::Tokenizer& tokenizer) {
    if (!tokenizer.has_exact_token_domain(kTokenDomain)) {
        throw std::invalid_argument(
            "artifact tokenizer does not expose the registered 248077-token domain");
    }
    for (const auto& [text, expected] : kVisionSpecialTokens) {
        const std::vector<int> encoded = tokenizer.encode(text);
        if (encoded.size() != 1 || encoded.front() != expected) {
            throw std::invalid_argument("artifact tokenizer does not match registered Vision token "
                                        "IDs");
        }
    }
    for (const auto& [text, expected] : kConfigOnlyTokens) {
        const std::vector<int> encoded = tokenizer.encode(text);
        if (encoded.size() != 1 || encoded.front() != expected ||
            !tokenizer.is_special_token(expected)) {
            throw std::invalid_argument(
                "artifact tokenizer does not merge official tokenizer_config.json tokens");
        }
    }
}

std::vector<fi::ChatMessage> convert_messages(std::vector<ChatMessage> messages) {
    std::vector<fi::ChatMessage> result;
    result.reserve(messages.size());
    for (ChatMessage& source : messages) {
        fi::ChatMessage target;
        target.role              = source.role;
        target.reasoning_content = std::move(source.reasoning_content);
        target.tool_call_id      = std::move(source.tool_call_id);
        target.tool_calls.reserve(source.tool_calls.size());
        for (ToolCall& call : source.tool_calls) {
            target.tool_calls.push_back(
                fi::ToolCall{.id             = std::move(call.id),
                             .name           = std::move(call.name),
                             .arguments_json = std::move(call.arguments_json)});
        }
        target.parts.reserve(source.parts.size());
        for (MessagePart& part : source.parts) {
            if (part.kind == MessagePartKind::Text) {
                target.parts.push_back(fi::ChatPart::text_part(std::move(part.text)));
                continue;
            }
            if (part.media.bytes.empty()) {
                throw std::invalid_argument("frontend media input contains no owning bytes");
            }
            fi::MediaData media;
            media.source_name = std::move(part.media.source_name);
            media.media_type  = std::move(part.media.media_type);
            media.bytes       = std::move(part.media.bytes);
            target.parts.push_back(part.media.kind == MediaKind::Image
                                       ? fi::ChatPart::image(std::move(media))
                                       : fi::ChatPart::video(std::move(media)));
        }
        result.push_back(std::move(target));
    }
    return result;
}

fi::ChatRenderOptions render_options(const PromptOptions& options) {
    return fi::ChatRenderOptions{.add_generation_prompt = options.add_generation_prompt,
                                 .enable_thinking       = options.enable_thinking,
                                 .reasoning_effort      = options.reasoning_effort,
                                 .preserve_thinking     = options.preserve_thinking,
                                 .add_vision_id         = options.add_vision_id,
                                 .tool_jsons            = options.tool_jsons};
}

std::uint32_t checked_token_count(std::size_t count) {
    if (count == 0) {
        throw std::invalid_argument("prepared prompt must contain at least one token");
    }
    if (count > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("prepared prompt token count exceeds the target domain");
    }
    return static_cast<std::uint32_t>(count);
}

void assign_text_positions(PreparedPromptData& prompt) {
    const std::size_t count = prompt.token_ids.size();
    prompt.token_types.assign(count, 0);
    prompt.positions.resize(count * 3);
    for (std::size_t axis = 0; axis < 3; ++axis) {
        for (std::size_t index = 0; index < count; ++index) {
            if (index > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
                throw std::invalid_argument("prepared prompt position exceeds int32 range");
            }
            prompt.positions[axis * count + index] = static_cast<std::int32_t>(index);
        }
    }
    prompt.rope_delta = 0;
}

bool enables_tool_output(const PromptInput& input) {
    if (!input.options.tool_jsons.empty()) { return true; }
    return std::any_of(
        input.messages.begin(), input.messages.end(), [](const ChatMessage& message) {
            return message.role == ChatRole::Tool ||
                   (message.role == ChatRole::Assistant && !message.tool_calls.empty());
        });
}

std::unique_ptr<PreparedPromptData> make_text_prompt_data(fi::EncodedChat encoded,
                                                          const PromptOptions& options,
                                                          bool tool_output_enabled,
                                                          Clock::time_point start) {
    auto prepared              = std::make_unique<PreparedPromptData>();
    PreparedPromptData& result = *prepared;
    result.token_ids.assign(encoded.input_ids.begin(), encoded.input_ids.end());
    result.identity.rewrite_checkpoint = encoded.rewrite_checkpoint;
    result.turn_closure_frontiers      = std::move(encoded.turn_closure_frontiers);
    result.system_prompt_frontier      = encoded.system_prompt_frontier;
    assign_text_positions(result);
    (void)checked_token_count(result.token_ids.size());
    result.identity.reusable   = true;
    result.starts_in_reasoning = options.add_generation_prompt && options.enable_thinking;
    result.tool_output_enabled = tool_output_enabled;
    result.prepare.seconds     = std::chrono::duration<double>(Clock::now() - start).count();
    return prepared;
}

VisionItem convert_vision_item(fi::VisionItem item);

std::unique_ptr<PreparedPromptData> make_media_prompt_data(fi::ProcessedInput processed,
                                                           const PromptOptions& options,
                                                           bool tool_output_enabled,
                                                           Clock::time_point start) {
    auto prepared              = std::make_unique<PreparedPromptData>();
    PreparedPromptData& result = *prepared;
    result.token_ids.assign(processed.input_ids.begin(), processed.input_ids.end());
    result.token_types = std::move(processed.token_types);
    result.positions   = std::move(processed.positions);
    result.rope_delta  = processed.rope_delta;
    result.patches     = std::move(processed.patches);
    result.vision_items.reserve(processed.vision_items.size());
    for (fi::VisionItem& item : processed.vision_items) {
        result.vision_items.push_back(convert_vision_item(std::move(item)));
    }
    result.prepare.media_items         = processed.stats.media_items;
    result.prepare.raw_patches         = processed.stats.raw_patches;
    result.prepare.vision_tokens       = processed.stats.vision_tokens;
    result.prepare.attention_pairs     = processed.stats.attention_pairs;
    result.prepare.patch_bytes         = processed.stats.patch_bytes;
    result.identity.rewrite_checkpoint = processed.rewrite_checkpoint;
    result.turn_closure_frontiers      = std::move(processed.turn_closure_frontiers);
    result.system_prompt_frontier      = processed.system_prompt_frontier;
    (void)checked_token_count(result.token_ids.size());
    result.identity.reusable   = true;
    result.starts_in_reasoning = options.add_generation_prompt && options.enable_thinking;
    result.tool_output_enabled = tool_output_enabled;
    result.prepare.seconds     = std::chrono::duration<double>(Clock::now() - start).count();
    return prepared;
}

VisionItem convert_vision_item(fi::VisionItem item) {
    VisionItem result;
    result.modality =
        item.modality == fi::Modality::Image ? PromptModality::Image : PromptModality::Video;
    result.grid = VisionGrid{.temporal = item.grid.t, .height = item.grid.h, .width = item.grid.w};
    result.patch_begin    = item.patch_begin;
    result.patch_count    = item.patch_count;
    result.content_digest = item.content_digest;
    result.timestamps     = std::move(item.timestamps);
    result.token_spans.reserve(item.token_spans.size());
    for (const fi::TokenSpan span : item.token_spans) {
        result.token_spans.push_back(TokenSpan{.begin = span.begin, .count = span.count});
    }
    return result;
}

StopPolicy merge_stop_policy(const fi::Tokenizer& tokenizer, const StopPolicy& caller) {
    StopPolicy result;
    result.publish_stop_token = caller.publish_stop_token;
    const auto append_token   = [&](TokenId token) {
        if (!tokenizer.is_valid_token(token)) {
            throw std::invalid_argument("stop token id is outside the checkpoint vocabulary: " +
                                        std::to_string(token));
        }
        if (std::find(result.token_ids.begin(), result.token_ids.end(), token) ==
            result.token_ids.end()) {
            result.token_ids.push_back(token);
        }
    };
    if (caller.include_model_defaults) {
        for (const int token : tokenizer.default_stop_token_ids()) { append_token(token); }
    }
    for (const TokenId token : caller.token_ids) { append_token(token); }

    result.strings.reserve(caller.strings.size());
    for (const StopString& stop : caller.strings) {
        if (stop.text.empty()) { throw std::invalid_argument("stop string must not be empty"); }
        (void)ninfer::text::unicode_internal::utf8_codepoints(stop.text, "stop string");
        const auto duplicate = std::find_if(
            result.strings.begin(), result.strings.end(), [&](const StopString& existing) {
                return existing.text == stop.text && existing.channel == stop.channel &&
                       existing.include_in_output == stop.include_in_output;
            });
        if (duplicate == result.strings.end()) { result.strings.push_back(stop); }
    }
    return result;
}

std::size_t channel_index(OutputChannel channel) noexcept {
    return channel == OutputChannel::Reasoning ? 0 : 1;
}

void append_delta(PublishedOutput& output, OutputChannel channel, std::string text) {
    if (text.empty()) { return; }
    if (!output.empty() && output.back().channel == channel) {
        output.back().text += text;
    } else {
        output.push_back(OutputDelta{.channel = channel, .text = std::move(text)});
    }
}

// Decodes the complete prefix of generated bytes, replacing each maximal ill-formed subpart with
// U+FFFD (Unicode 3.9 "substitution of maximal subparts"). A trailing incomplete but still
// well-formed sequence stays pending for the next token. Sampling can emit any byte-level token,
// so malformed output must degrade to replacement characters rather than fail the request.
std::string consume_generated_utf8(std::string& pending) {
    std::string decoded;
    decoded.reserve(pending.size());
    std::size_t offset = 0;
    while (offset < pending.size()) {
        const auto lead    = static_cast<unsigned char>(pending[offset]);
        std::size_t length = 0;
        if (lead <= 0x7fU) {
            decoded.push_back(pending[offset]);
            ++offset;
            continue;
        } else if (lead >= 0xc2U && lead <= 0xdfU) {
            length = 2;
        } else if (lead >= 0xe0U && lead <= 0xefU) {
            length = 3;
        } else if (lead >= 0xf0U && lead <= 0xf4U) {
            length = 4;
        } else {
            decoded.append(kUtf8Replacement);
            ++offset;
            continue;
        }

        bool malformed = false;
        for (std::size_t index = 1; index < length; ++index) {
            if (offset + index >= pending.size()) {
                pending.erase(0, offset);
                return decoded;
            }
            const auto byte      = static_cast<unsigned char>(pending[offset + index]);
            unsigned int minimum = 0x80U;
            unsigned int maximum = 0xbfU;
            if (index == 1) {
                if (lead == 0xe0U) {
                    minimum = 0xa0U;
                } else if (lead == 0xedU) {
                    maximum = 0x9fU;
                } else if (lead == 0xf0U) {
                    minimum = 0x90U;
                } else if (lead == 0xf4U) {
                    maximum = 0x8fU;
                }
            }
            if (byte < minimum || byte > maximum) {
                // The byte that cannot continue this sequence starts the next iteration, so valid
                // following text is kept.
                decoded.append(kUtf8Replacement);
                offset += index;
                malformed = true;
                break;
            }
        }
        if (malformed) { continue; }

        decoded.append(pending, offset, length);
        offset += length;
    }
    pending.clear();
    return decoded;
}

std::size_t longest_suffix_prefix(std::string_view text, std::string_view marker,
                                  bool allow_complete = false) {
    const std::size_t maximum = std::min(text.size(), marker.size());
    for (std::size_t size = maximum; size != 0; --size) {
        if (!allow_complete && size == marker.size()) { continue; }
        if (text.substr(text.size() - size) == marker.substr(0, size)) { return size; }
    }
    return 0;
}

struct DecoderState {
    std::string utf8_pending;
    std::string think_marker_pending;
    std::array<std::string, 2> stop_pending;
    bool in_reasoning               = false;
    bool constrained_content        = false;
    bool strip_content_leading      = false;
    bool terminal                   = false;
    bool require_initial_content    = false;
    bool has_non_whitespace_content = false;
    bool in_tool_call               = false;
    std::string tool_marker_pending;
    std::uint64_t decoded_bytes    = 0;
    std::uint32_t reasoning_tokens = 0;
    ReasoningLimitStats reasoning_limit;
    // IDs consisting only of a pending textual close prefix. Charged conservatively
    // until the complete marker proves they were framing rather than body.
    std::uint32_t reasoning_marker_tokens = 0;
    bool reasoning_limit_retry_pending    = false;
};

// Only generated IDs in this response can establish the replay. Once the first
// reasoning close freezes the reference, speculative forks share immutable IDs
// and copy only cursors/flags. No historical prompt tokens enter this proof.
struct ReasoningReplayState {
    static constexpr std::size_t minimum_reference = 256;
    bool enabled                                   = false;
    bool active                                    = false;
    bool repaired                                  = false;
    bool retry_required                            = false;
    bool accepted_answer                           = false;
    std::size_t matched                            = 0;
    std::shared_ptr<const std::vector<TokenId>> reference;

    [[nodiscard]] bool close_allowed() const noexcept {
        return enabled && active && !repaired && reference && matched == reference->size();
    }

    void freeze(std::span<const TokenId> history, bool dedicated_close) {
        if (!enabled || reference) { return; }
        if (!dedicated_close || history.size() < minimum_reference) {
            enabled = false;
            return;
        }
        reference = std::make_shared<const std::vector<TokenId>>(history.begin(), history.end());
        active    = true;
    }

    void observe(TokenId token, std::optional<OutputChannel> channel, bool leading_gap,
                 bool model_stop, bool substantive) {
        if (!enabled || !reference || model_stop) { return; }
        if (active && channel == OutputChannel::Content && matched < reference->size() &&
            token == (*reference)[matched]) {
            ++matched;
            if (repaired && matched == reference->size()) { retry_required = true; }
            return;
        }
        if (active && leading_gap && matched == 0) { return; }
        // Divergence releases a previously held nonempty copy as ordinary answer
        // content. Its semantic relevance is not inferred by this framing guard.
        if (active && matched != 0) { accepted_answer = true; }
        active = false;
        if (channel == OutputChannel::Content && substantive) { accepted_answer = true; }
    }
};

bool suppress_reasoning_end(const DecoderState& state, bool raw, bool grammar_managed) {
    return !raw && !state.in_reasoning && !grammar_managed;
}

void track_structured_content(DecoderState& state, std::string_view text,
                              bool tool_output_enabled) {
    if (text.empty()) { return; }
    if (!state.has_non_whitespace_content) {
        state.has_non_whitespace_content = std::any_of(
            text.begin(), text.end(), [](unsigned char byte) { return std::isspace(byte) == 0; });
    }
    if (!tool_output_enabled) { return; }

    state.tool_marker_pending.append(text);
    for (;;) {
        if (!state.in_tool_call) {
            const std::size_t open = state.tool_marker_pending.find(kToolOpen);
            if (open == std::string::npos) {
                const std::size_t hold =
                    longest_suffix_prefix(state.tool_marker_pending, kToolOpen);
                state.tool_marker_pending.erase(0, state.tool_marker_pending.size() - hold);
                break;
            }
            state.in_tool_call = true;
            state.tool_marker_pending.erase(0, open + kToolOpen.size());
            continue;
        }
        const std::size_t close = state.tool_marker_pending.find(kToolClose);
        if (close != std::string::npos) {
            state.in_tool_call = false;
            state.tool_marker_pending.erase(0, close + kToolClose.size());
            continue;
        }
        const std::size_t hold = longest_suffix_prefix(state.tool_marker_pending, kToolClose);
        state.tool_marker_pending.erase(0, state.tool_marker_pending.size() - hold);
        break;
    }
}

bool has_ambiguous_tool_marker(const DecoderState& state, bool tool_output_enabled) {
    if (!tool_output_enabled || state.in_tool_call) { return false; }
    return !state.tool_marker_pending.empty();
}

bool in_tool_region(const DecoderState& state, bool tool_output_enabled) {
    return state.in_tool_call || has_ambiguous_tool_marker(state, tool_output_enabled);
}

std::optional<OutputChannel> attributed_channel(bool was_reasoning, bool was_tool,
                                                bool was_stripping, const DecoderState& after,
                                                std::string_view bytes, bool withheld_stop,
                                                bool raw, bool tool_output_enabled) {
    if (bytes.empty() || withheld_stop) { return std::nullopt; }
    if (was_reasoning) {
        return after.in_reasoning ? std::optional{OutputChannel::Reasoning} : std::nullopt;
    }
    // Raw publication does not interpret tool envelopes.
    if (raw || (!was_tool && !in_tool_region(after, tool_output_enabled) &&
                !(tool_output_enabled && bytes.find(kToolOpen) != std::string_view::npos) &&
                !(was_stripping && after.strip_content_leading))) {
        return OutputChannel::Content;
    }
    return std::nullopt;
}

bool model_stop_allowed(const DecoderState& state, bool raw, bool tool_output_enabled) {
    if (raw) { return true; }
    if (state.in_reasoning) { return false; }
    if (state.require_initial_content && !state.has_non_whitespace_content) { return false; }
    return !in_tool_region(state, tool_output_enabled);
}

struct StopMatch {
    bool found                      = false;
    std::uint32_t committed_tokens  = 0;
    std::uint64_t byte_cut          = 0;
    std::uint32_t declaration_order = 0;
    PublishedOutput output;
};

bool stop_match_precedes(std::uint32_t committed_tokens, std::uint64_t byte_cut,
                         std::uint32_t declaration_order, const StopMatch& current) noexcept {
    if (!current.found) { return true; }
    if (committed_tokens != current.committed_tokens) {
        return committed_tokens < current.committed_tokens;
    }
    if (byte_cut != current.byte_cut) { return byte_cut < current.byte_cut; }
    return declaration_order < current.declaration_order;
}

std::size_t stop_hold_size(std::string_view text, OutputChannel channel, const StopPolicy& policy) {
    std::size_t hold = 0;
    for (const StopString& stop : policy.strings) {
        if (stop.channel != channel) { continue; }
        hold = std::max(hold, longest_suffix_prefix(text, stop.text));
    }
    return hold;
}

void feed_channel(DecoderState& state, OutputChannel channel, std::string_view text,
                  const StopPolicy& policy, PublishedOutput& emitted,
                  std::uint32_t committed_tokens, StopMatch* best_match) {
    if (text.empty()) { return; }
    std::size_t string_stops = 0;
    for (const StopString& stop : policy.strings) {
        if (stop.channel == channel) { ++string_stops; }
    }
    if (string_stops == 0) {
        append_delta(emitted, channel, std::string(text));
        state.decoded_bytes += text.size();
        return;
    }
    std::string combined          = state.stop_pending[channel_index(channel)];
    const std::size_t old_pending = combined.size();
    combined.append(text);
    const std::uint64_t combined_start = state.decoded_bytes - old_pending;

    if (best_match != nullptr) {
        for (std::size_t declaration = 0; declaration < policy.strings.size(); ++declaration) {
            const StopString& stop = policy.strings[declaration];
            if (stop.channel != channel) { continue; }
            const std::size_t found = combined.find(stop.text);
            if (found == std::string::npos) { continue; }
            const std::uint64_t byte_cut = combined_start + found;
            const auto order             = static_cast<std::uint32_t>(declaration);
            if (!stop_match_precedes(committed_tokens, byte_cut, order, *best_match)) { continue; }

            PublishedOutput candidate = emitted;
            append_delta(candidate, channel, combined.substr(0, found));
            if (stop.include_in_output) { append_delta(candidate, channel, stop.text); }
            *best_match = StopMatch{.found             = true,
                                    .committed_tokens  = committed_tokens,
                                    .byte_cut          = byte_cut,
                                    .declaration_order = order,
                                    .output            = std::move(candidate)};
        }
    }

    const std::size_t hold = stop_hold_size(combined, channel, policy);
    append_delta(emitted, channel, combined.substr(0, combined.size() - hold));
    state.stop_pending[channel_index(channel)] = combined.substr(combined.size() - hold);
    state.decoded_bytes += text.size();
}

void close_channel(DecoderState& state, OutputChannel channel, PublishedOutput& emitted) {
    std::string& pending = state.stop_pending[channel_index(channel)];
    append_delta(emitted, channel, std::move(pending));
    pending.clear();
}

void feed_content(DecoderState& state, std::string text, const StopPolicy& policy,
                  PublishedOutput& emitted, std::uint32_t committed_tokens, StopMatch* best_match,
                  bool tool_output_enabled) {
    if (state.strip_content_leading) {
        std::size_t begin = 0;
        while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin])) != 0) {
            ++begin;
        }
        text.erase(0, begin);
        if (!text.empty()) { state.strip_content_leading = false; }
    }
    track_structured_content(state, text, tool_output_enabled);
    feed_channel(state, OutputChannel::Content, text, policy, emitted, committed_tokens,
                 best_match);
}

void feed_decoded_text(DecoderState& state, std::string_view text, const StopPolicy& policy,
                       PublishedOutput& emitted, std::uint32_t committed_tokens,
                       StopMatch* best_match, bool tool_output_enabled) {
    if (!state.in_reasoning) {
        feed_content(state, std::string(text), policy, emitted, committed_tokens, best_match,
                     tool_output_enabled);
        return;
    }

    state.think_marker_pending.append(text);
    const std::size_t marker = state.think_marker_pending.find(kThinkClose);
    if (marker != std::string::npos) {
        feed_channel(state, OutputChannel::Reasoning,
                     std::string_view(state.think_marker_pending).substr(0, marker), policy,
                     emitted, committed_tokens, best_match);
        close_channel(state, OutputChannel::Reasoning, emitted);
        std::string content = state.think_marker_pending.substr(marker + kThinkClose.size());
        state.think_marker_pending.clear();
        state.in_reasoning          = false;
        state.strip_content_leading = !state.constrained_content;
        feed_content(state, std::move(content), policy, emitted, committed_tokens, best_match,
                     tool_output_enabled);
        return;
    }

    const std::size_t hold = longest_suffix_prefix(state.think_marker_pending, kThinkClose, true);
    const std::size_t safe = state.think_marker_pending.size() - hold;
    feed_channel(state, OutputChannel::Reasoning,
                 std::string_view(state.think_marker_pending).substr(0, safe), policy, emitted,
                 committed_tokens, best_match);
    state.think_marker_pending.erase(0, safe);
}

void feed_token_bytes(DecoderState& state, std::string_view bytes, const StopPolicy& policy,
                      PublishedOutput& emitted, std::uint32_t committed_tokens,
                      StopMatch* best_match, bool tool_output_enabled) {
    state.utf8_pending.append(bytes);
    const std::string text = consume_generated_utf8(state.utf8_pending);
    if (text.empty()) { return; }
    feed_decoded_text(state, text, policy, emitted, committed_tokens, best_match,
                      tool_output_enabled);
}

// A native close must not complete an unfinished UTF-8 sequence or absorb a
// textual marker prefix. Both pending fragments remain in the reasoning channel.
void flush_reasoning_fragments(DecoderState& state, const StopPolicy& policy,
                               PublishedOutput& emitted, std::uint32_t committed_tokens,
                               StopMatch* match, bool tool_output_enabled) {
    if (!state.utf8_pending.empty()) {
        state.utf8_pending.clear();
        feed_decoded_text(state, kUtf8Replacement, policy, emitted, committed_tokens, match,
                          tool_output_enabled);
    }
    if (!state.think_marker_pending.empty()) {
        feed_channel(state, OutputChannel::Reasoning, state.think_marker_pending, policy, emitted,
                     committed_tokens, match);
        state.think_marker_pending.clear();
    }
    state.reasoning_marker_tokens = 0;
}

void charge_reasoning_token(DecoderState& state, std::string_view bytes, bool canonical_close,
                            bool stop_token) {
    auto& limit = state.reasoning_limit;
    if (!state.in_reasoning || limit.budget_tokens == 0 || canonical_close || stop_token) {
        return;
    }
    ++limit.used_tokens;
    const std::string candidate = state.think_marker_pending + std::string(bytes);
    if (state.utf8_pending.empty() && !candidate.empty() && kThinkClose.starts_with(candidate)) {
        ++state.reasoning_marker_tokens;
        if (candidate == kThinkClose) {
            limit.used_tokens -= state.reasoning_marker_tokens;
            state.reasoning_marker_tokens = 0;
        }
    } else if (state.utf8_pending.empty() && candidate.starts_with(kThinkClose)) {
        // The current ID contains closing framing followed by answer text, but
        // contributes no reasoning body. Earlier mixed body/marker IDs stay charged.
        limit.used_tokens -= state.reasoning_marker_tokens + 1;
        state.reasoning_marker_tokens = 0;
    } else {
        state.reasoning_marker_tokens = 0;
    }
}

void terminalize(DecoderState& state, const StopPolicy& policy, PublishedOutput& emitted,
                 std::uint32_t committed_tokens, bool tool_output_enabled) {
    if (!state.utf8_pending.empty()) {
        // A token budget can end between byte-level tokens of one code point.
        // Publish the standard replacement character rather than an invalid
        // UTF-8 suffix; the logical token prefix remains exact.
        state.utf8_pending.clear();
        feed_decoded_text(state, kUtf8Replacement, policy, emitted, committed_tokens, nullptr,
                          tool_output_enabled);
    }
    if (state.in_reasoning) {
        feed_channel(state, OutputChannel::Reasoning, state.think_marker_pending, policy, emitted,
                     committed_tokens, nullptr);
        state.think_marker_pending.clear();
        close_channel(state, OutputChannel::Reasoning, emitted);
    } else {
        close_channel(state, OutputChannel::Content, emitted);
    }
    state.stop_pending = {};
    state.terminal     = true;
}

DecoderState terminal_state(DecoderState state) {
    state.utf8_pending.clear();
    state.think_marker_pending.clear();
    state.stop_pending = {};
    state.terminal     = true;
    return state;
}

} // namespace

class Frontend::Impl {
public:
    Impl(const FrontendResources& resources, bool registered_checkpoint, bool vision_enabled_)
        : chat_template(compile_chat_template(resources)),
          tokenizer(std::make_shared<const fi::Tokenizer>(
              fi::TokenizerResources{.tokenizer_json         = resources.tokenizer_json,
                                     .tokenizer_config_json  = resources.tokenizer_config_json,
                                     .generation_config_json = resources.generation_config_json})),
          token_grammar(tokenizer), processor(processor_options(resources)),
          vision_enabled(vision_enabled_) {
        if (registered_checkpoint) { validate_registered_tokenizer(*tokenizer); }
        const auto closing = tokenizer->encode(kThinkClose);
        if (closing.size() != 1 || tokenizer->decode_token_bytes(closing.front()) != kThinkClose) {
            throw std::invalid_argument("tokenizer must encode the reasoning close as one token");
        }
        reasoning_end_token = closing.front();
        for (const int token : tokenizer->default_stop_token_ids()) {
            if (!tokenizer->is_valid_token(token)) {
                throw std::invalid_argument(
                    "generation_config.json contains a stop token outside the vocabulary");
            }
            defaults.token_ids.push_back(token);
        }
    }

    fi::CompiledChatTemplate chat_template;
    std::shared_ptr<const fi::Tokenizer> tokenizer;
    mutable fi::TokenGrammarCompiler token_grammar;
    fi::ProcessorOptions processor;
    StopPolicy defaults;
    TokenId reasoning_end_token = -1;
    bool vision_enabled         = true;
};

class OutputSession::Impl {
public:
    struct ToolPublication {
        std::string pending;
        std::vector<ToolCall> calls;
        bool selected_tool = false;

        void filter(PublishedOutput& output, const TokenGrammarData& grammar, bool terminal) {
            PublishedOutput visible;
            for (auto& delta : output) {
                if (delta.channel == OutputChannel::Reasoning) {
                    append_delta(visible, delta.channel, std::move(delta.text));
                    continue;
                }
                pending += delta.text;
                for (;;) {
                    const auto open = pending.find(kToolOpen);
                    if (open == std::string::npos) {
                        if (!calls.empty() &&
                            std::all_of(pending.begin(), pending.end(),
                                        [](unsigned char c) { return std::isspace(c) != 0; })) {
                            if (terminal) { pending.clear(); }
                            break;
                        }
                        const auto suffix = longest_suffix_prefix(pending, kToolOpen);
                        // The grammar reserves <tool_call before its final >.
                        // Preserve shorter ambiguous prose (e.g. "a <"), but
                        // never publish a confirmed trigger cut by a caller stop.
                        const auto hold = !terminal || suffix == kToolOpen.size() - 1 ? suffix : 0;
                        append_delta(visible, OutputChannel::Content,
                                     pending.substr(0, pending.size() - hold));
                        pending.erase(0, pending.size() - hold);
                        break;
                    }
                    const bool separator =
                        !calls.empty() &&
                        std::all_of(pending.begin(), pending.begin() + open,
                                    [](unsigned char c) { return std::isspace(c) != 0; });
                    if (!separator) {
                        append_delta(visible, OutputChannel::Content, pending.substr(0, open));
                    }
                    selected_tool = true;
                    pending.erase(0, open);
                    auto close     = pending.find(kToolClose);
                    bool completed = false;
                    while (close != std::string::npos) {
                        const auto end = close + kToolClose.size();
                        if (auto call =
                                grammar.decode_call(std::string_view(pending).substr(0, end))) {
                            calls.push_back(std::move(*call));
                            pending.erase(0, end);
                            completed = true;
                            break;
                        }
                        close = pending.find(kToolClose, end);
                    }
                    if (!completed) { break; }
                }
            }
            // A caller stop/cancellation/output budget can interrupt a valid
            // envelope. Never convert that partial call into an executable call
            // or leak its markup into a streamed prose answer.
            if (terminal) {
                selected_tool = selected_tool || pending.starts_with("<tool_call");
                const bool separator =
                    !calls.empty() &&
                    std::all_of(pending.begin(), pending.end(),
                                [](unsigned char c) { return std::isspace(c) != 0; });
                if (!pending.starts_with("<tool_call") && !separator) {
                    append_delta(visible, OutputChannel::Content, std::move(pending));
                }
                pending.clear();
            }
            output = std::move(visible);
        }
    };

    Impl(std::shared_ptr<const fi::Tokenizer> tokenizer_, StopPolicy policy_, OutputOptions output,
         bool starts_in_reasoning, bool tool_output_enabled_, TokenId reasoning_end_token_,
         std::span<const TokenId> model_stop_tokens_,
         std::shared_ptr<const TokenGrammarData> grammar,
         std::shared_ptr<const GenerationRecoveryContext> recovery, bool generation_recovery)
        : tokenizer(std::move(tokenizer_)), policy(std::move(policy_)),
          model_stop_tokens(model_stop_tokens_.begin(), model_stop_tokens_.end()),
          preserve_special(output.raw || output.preserve_special_tokens),
          preserve_model_stops(output.raw || output.preserve_special_tokens), raw(output.raw),
          tool_output_enabled(tool_output_enabled_), reasoning_end_token(reasoning_end_token_) {
        state.reasoning_limit.budget_tokens = output.thinking_token_budget.value_or(0);
        state.in_reasoning                  = starts_in_reasoning && !output.raw;
        state.require_initial_content = !output.raw && (state.in_reasoning || tool_output_enabled);
        grammar_data                  = std::move(grammar);
        state.constrained_content     = grammar_data && grammar_data->definitions.empty();
        replay.enabled = generation_recovery && state.in_reasoning && !state.constrained_content;
        if (state.constrained_content) { state.require_initial_content = false; }
        preserve_special = preserve_special || state.constrained_content;
        recovery_context = std::move(recovery);
        if (grammar_data) {
            std::vector<TokenId> ignored;
            for (const auto token : model_stop_tokens) {
                if (std::find(policy.token_ids.begin(), policy.token_ids.end(), token) ==
                    policy.token_ids.end()) {
                    ignored.push_back(token);
                }
            }
            token_grammar =
                std::make_unique<fi::TokenGrammarState>(grammar_data, std::move(ignored));
        }
    }

    std::shared_ptr<const fi::Tokenizer> tokenizer;
    StopPolicy policy;
    std::vector<TokenId> model_stop_tokens;
    bool preserve_special                      = false;
    bool preserve_model_stops                  = false;
    bool raw                                   = false;
    bool tool_output_enabled                   = false;
    TokenId reasoning_end_token                = -1;
    runtime::ReasoningLimitAction limit_action = runtime::ReasoningLimitAction::Close;
    DecoderState state;
    DecoderState preview_state;
    ReasoningReplayState replay;
    ReasoningReplayState preview_replay;
    // Preview may append; discard truncates to the committed size. Immutable
    // reference snapshots are created only at the first dedicated close.
    std::vector<TokenId> reasoning_history;
    std::size_t committed_reasoning_size = 0;
    std::vector<std::uint8_t> preview_ignored_framing;
    std::vector<std::uint8_t> preview_reasoning_closes;
    std::uint32_t preview_replay_discarded = 0;
    std::uint32_t preview_replay_repairs   = 0;
    std::uint32_t replay_discarded         = 0;
    std::uint32_t replay_repairs           = 0;
    PublishedOutput replay_held;
    PublishedOutput preview_output;
    // One entry per token of the pending preview; see preview_token_channels().
    std::vector<std::optional<OutputChannel>> preview_channels;
    bool preview_ready = false;
    std::unique_ptr<fi::TokenGrammarState> token_grammar;
    std::shared_ptr<const TokenGrammarData> grammar_data;
    std::shared_ptr<const GenerationRecoveryContext> recovery_context;
    ToolPublication tools;
    ToolPublication preview_tools;
};

std::span<const std::int32_t> PreparedPromptData::position_axis(int axis) const {
    if (axis < 0 || axis >= 3 || positions.size() != token_ids.size() * 3) {
        throw std::out_of_range("invalid prepared-prompt position axis");
    }
    return std::span<const std::int32_t>(positions).subspan(
        static_cast<std::size_t>(axis) * token_ids.size(), token_ids.size());
}

PreparedPrompt::PreparedPrompt() noexcept = default;

PreparedPrompt::PreparedPrompt(std::unique_ptr<PreparedPromptData> data) noexcept
    : data_(std::move(data)) {}

PreparedPrompt::~PreparedPrompt()                                    = default;
PreparedPrompt::PreparedPrompt(PreparedPrompt&&) noexcept            = default;
PreparedPrompt& PreparedPrompt::operator=(PreparedPrompt&&) noexcept = default;

PromptSummary PreparedPrompt::summary() const {
    if (data_ == nullptr) { throw std::logic_error("prepared prompt is empty"); }
    return PromptSummary{.prompt_tokens = checked_token_count(data_->token_ids.size()),
                         .has_media     = data_->has_media()};
}

double PreparedPrompt::prepare_seconds() const noexcept {
    return data_ != nullptr ? data_->prepare.seconds : 0.0;
}

PreparedPrompt::operator bool() const noexcept { return data_ != nullptr; }

PublishedOutput::PublishedOutput(PublishedOutput&& other) noexcept
    : values_(std::move(other.values_)), size_(std::exchange(other.size_, 0)) {}

PublishedOutput& PublishedOutput::operator=(PublishedOutput&& other) noexcept {
    if (this != &other) {
        values_ = std::move(other.values_);
        size_   = std::exchange(other.size_, 0);
    }
    return *this;
}

void PublishedOutput::clear() noexcept {
    for (std::size_t index = 0; index < size_; ++index) { values_[index] = {}; }
    size_ = 0;
}

void PublishedOutput::push_back(OutputDelta value) {
    // Withheld publications can span rounds; adjacent records still belong to one channel.
    if (size_ != 0 && values_[size_ - 1].channel == value.channel) {
        auto& previous = values_[size_ - 1];
        previous.text += value.text;
        previous.logprobs.insert(previous.logprobs.end(),
                                 std::make_move_iterator(value.logprobs.begin()),
                                 std::make_move_iterator(value.logprobs.end()));
        return;
    }
    if (size_ == values_.size()) {
        throw std::logic_error("output decoder produced more than two channel transitions");
    }
    values_[size_++] = std::move(value);
}

OutputSession::OutputSession() noexcept                           = default;
OutputSession::~OutputSession()                                   = default;
OutputSession::OutputSession(OutputSession&&) noexcept            = default;
OutputSession& OutputSession::operator=(OutputSession&&) noexcept = default;

OutputSession::OutputSession(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

runtime::OutputDecision OutputSession::preview(std::span<const TokenId> tokens,
                                               std::uint32_t budget_remaining,
                                               FinishReason limit_reason) {
    if (impl_ == nullptr) { throw std::logic_error("output session is empty"); }
    if (impl_->state.terminal) { throw std::logic_error("output session is already terminal"); }
    if (impl_->preview_ready) { throw std::logic_error("output session already has a preview"); }
    if (tokens.empty()) {
        throw std::invalid_argument("cannot preview an empty generated-token round");
    }
    if (tokens.size() > budget_remaining) {
        throw std::invalid_argument("generated-token round exceeds the remaining budget");
    }
    if (limit_reason != FinishReason::OutputLimit &&
        limit_reason != FinishReason::ContextCapacity) {
        throw std::invalid_argument("generated-token budget has an invalid limit reason");
    }

    impl_->preview_state  = impl_->state;
    impl_->preview_replay = impl_->replay;
    impl_->reasoning_history.resize(impl_->committed_reasoning_size);
    impl_->preview_ignored_framing.clear();
    impl_->preview_reasoning_closes.clear();
    impl_->preview_replay_discarded = 0;
    impl_->preview_replay_repairs   = 0;
    impl_->preview_output.clear();

    const bool stops_blocked_at_start =
        !model_stop_allowed(impl_->state, impl_->raw, impl_->tool_output_enabled);
    bool saw_tool_region = in_tool_region(impl_->state, impl_->tool_output_enabled);

    impl_->preview_channels.clear();
    const auto complete = [&](std::uint32_t count, FinishReason reason,
                              bool reject_generated_round = false) {
        impl_->preview_channels.resize(count);
        impl_->preview_ignored_framing.resize(count);
        impl_->preview_reasoning_closes.resize(count);
        if (impl_->token_grammar) {
            impl_->token_grammar->preview(tokens.first(count), impl_->preview_ignored_framing,
                                          impl_->preview_reasoning_closes);
        }
        if (impl_->grammar_data && !impl_->grammar_data->definitions.empty() && !impl_->raw) {
            impl_->preview_tools = impl_->tools;
            impl_->preview_tools.filter(impl_->preview_output, *impl_->grammar_data,
                                        reason != FinishReason::None);
        }
        impl_->preview_ready = true;
        return runtime::OutputDecision{.accepted_tokens        = count,
                                       .finish_reason          = reason,
                                       .reject_generated_round = reject_generated_round};
    };

    for (std::size_t index = 0; index < tokens.size(); ++index) {
        const std::uint32_t count = static_cast<std::uint32_t>(index + 1);
        const TokenId token       = tokens[index];
        if (!impl_->tokenizer->is_valid_token(token)) {
            throw std::out_of_range("generated token is outside the checkpoint vocabulary: " +
                                    std::to_string(token));
        }

        if (suppress_reasoning_end(impl_->preview_state, impl_->raw,
                                   impl_->grammar_data != nullptr) &&
            !impl_->preview_replay.close_allowed() && token == impl_->reasoning_end_token) {
            throw std::logic_error("sampler selected a reasoning close outside reasoning");
        }
        if (impl_->preview_state.in_reasoning &&
            impl_->preview_state.reasoning_limit.budget_tokens != 0 &&
            impl_->preview_state.reasoning_limit.used_tokens >=
                impl_->preview_state.reasoning_limit.budget_tokens &&
            token != impl_->reasoning_end_token &&
            std::find(impl_->policy.token_ids.begin(), impl_->policy.token_ids.end(), token) ==
                impl_->policy.token_ids.end()) {
            throw std::logic_error("sampler crossed the reasoning-token boundary without a close");
        }
        if (impl_->preview_state.in_reasoning) { ++impl_->preview_state.reasoning_tokens; }

        const bool stop_token =
            std::find(impl_->policy.token_ids.begin(), impl_->policy.token_ids.end(), token) !=
            impl_->policy.token_ids.end();
        const bool model_stop =
            std::find(impl_->model_stop_tokens.begin(), impl_->model_stop_tokens.end(), token) !=
            impl_->model_stop_tokens.end();
        if (model_stop &&
            !model_stop_allowed(impl_->preview_state, impl_->raw, impl_->tool_output_enabled)) {
            impl_->preview_state  = impl_->state;
            impl_->preview_replay = impl_->replay;
            impl_->reasoning_history.resize(impl_->committed_reasoning_size);
            impl_->preview_output.clear();
            return complete(0, FinishReason::None, true);
        }
        DecoderState before_state;
        PublishedOutput before_output;
        if (stop_token && !impl_->policy.publish_stop_token) {
            before_state  = impl_->preview_state;
            before_output = impl_->preview_output;
        }

        StopMatch match;
        const bool repair_close =
            impl_->preview_replay.close_allowed() && token == impl_->reasoning_end_token;
        impl_->preview_ignored_framing.push_back(repair_close ? 1 : 0);
        impl_->preview_reasoning_closes.push_back(0);
        if (repair_close && !stop_token) {
            impl_->preview_replay_discarded =
                static_cast<std::uint32_t>(impl_->preview_replay.matched);
            impl_->preview_replay_repairs              = 1;
            impl_->preview_replay.repaired             = true;
            impl_->preview_replay.matched              = 0;
            impl_->preview_state.strip_content_leading = true;
            impl_->preview_channels.push_back(std::nullopt);
            if (count == budget_remaining) {
                terminalize(impl_->preview_state, impl_->policy, impl_->preview_output, count,
                            impl_->tool_output_enabled);
                return complete(count, limit_reason);
            }
            // Never merge post-repair prose/tools with the replay being discarded.
            return complete(count, FinishReason::None);
        }
        const std::string_view bytes = impl_->tokenizer->decode_token_bytes(
            token, !impl_->preserve_special || (model_stop && !impl_->preserve_model_stops));
        const bool was_reasoning      = impl_->preview_state.in_reasoning;
        const bool was_end_suppressed = suppress_reasoning_end(impl_->preview_state, impl_->raw,
                                                               impl_->grammar_data != nullptr);
        const bool was_stripping      = impl_->preview_state.strip_content_leading;
        const bool was_tool    = in_tool_region(impl_->preview_state, impl_->tool_output_enabled);
        const bool had_content = impl_->preview_state.has_non_whitespace_content;
        const bool semantic_close = was_reasoning &&
                                    impl_->preview_state.reasoning_limit.budget_tokens != 0 &&
                                    impl_->preview_state.reasoning_limit.used_tokens >=
                                        impl_->preview_state.reasoning_limit.budget_tokens &&
                                    token == impl_->reasoning_end_token;
        const bool forced_close   = semantic_close && !stop_token;
        if (semantic_close) { impl_->preview_reasoning_closes.back() = 1; }
        if (was_reasoning && token == impl_->reasoning_end_token &&
            impl_->preview_state.reasoning_limit.budget_tokens != 0) {
            flush_reasoning_fragments(impl_->preview_state, impl_->policy, impl_->preview_output,
                                      count, &match, impl_->tool_output_enabled);
        }
        charge_reasoning_token(impl_->preview_state, bytes, token == impl_->reasoning_end_token,
                               stop_token);
        feed_token_bytes(impl_->preview_state, bytes, impl_->policy, impl_->preview_output, count,
                         &match, impl_->tool_output_enabled);
        // A budget replacement needs semantic answer evidence, not diagnostic EOS bytes.
        if (model_stop && stop_token && impl_->preview_state.reasoning_limit.budget_tokens != 0) {
            impl_->preview_state.has_non_whitespace_content = had_content;
        }
        if (forced_close && !match.found) { ++impl_->preview_state.reasoning_limit.forced_closes; }
        const bool at_reasoning_limit = impl_->preview_state.in_reasoning &&
                                        impl_->preview_state.reasoning_limit.budget_tokens != 0 &&
                                        impl_->preview_state.reasoning_limit.used_tokens >=
                                            impl_->preview_state.reasoning_limit.budget_tokens;
        if (at_reasoning_limit) {
            flush_reasoning_fragments(impl_->preview_state, impl_->policy, impl_->preview_output,
                                      count, &match, impl_->tool_output_enabled);
            impl_->preview_state.reasoning_limit.limit_reached = true;
        }
        const auto channel =
            attributed_channel(was_reasoning, was_tool, was_stripping, impl_->preview_state, bytes,
                               stop_token && !impl_->policy.publish_stop_token, impl_->raw,
                               impl_->tool_output_enabled);
        impl_->preview_channels.push_back(channel);
        if (impl_->preview_replay.enabled && !impl_->preview_replay.reference) {
            if (channel == OutputChannel::Reasoning) { impl_->reasoning_history.push_back(token); }
            if (was_reasoning && !impl_->preview_state.in_reasoning) {
                impl_->preview_replay.freeze(impl_->reasoning_history,
                                             token == impl_->reasoning_end_token);
            }
        } else if (!was_reasoning) {
            impl_->preview_replay.observe(
                token, channel, was_stripping && impl_->preview_state.strip_content_leading,
                model_stop && stop_token,
                std::any_of(bytes.begin(), bytes.end(),
                            [](unsigned char byte) { return std::isspace(byte) == 0; }));
        }

        if (match.found) {
            impl_->preview_state  = terminal_state(std::move(impl_->preview_state));
            impl_->preview_output = std::move(match.output);
            return complete(match.committed_tokens, FinishReason::StopString);
        }

        if (stop_token) {
            if (!impl_->policy.publish_stop_token) {
                impl_->preview_state  = std::move(before_state);
                impl_->preview_output = std::move(before_output);
            }
            terminalize(impl_->preview_state, impl_->policy, impl_->preview_output, count,
                        impl_->tool_output_enabled);
            return complete(count, FinishReason::StopToken);
        }

        if (at_reasoning_limit || forced_close) {
            if (count == budget_remaining) {
                terminalize(impl_->preview_state, impl_->policy, impl_->preview_output, count,
                            impl_->tool_output_enabled);
                return complete(count, limit_reason);
            }
            if (at_reasoning_limit && impl_->limit_action == runtime::ReasoningLimitAction::Retry &&
                impl_->preview_state.reasoning_limit.retries == 0) {
                impl_->preview_state.reasoning_limit_retry_pending = true;
            }
            // The following target position must be sampled with the new mask.
            return complete(count, FinishReason::None);
        }

        if (impl_->preview_replay.retry_required) {
            if (count == budget_remaining) {
                terminalize(impl_->preview_state, impl_->policy, impl_->preview_output, count,
                            impl_->tool_output_enabled);
                return complete(count, limit_reason);
            }
            return complete(count, FinishReason::None);
        }

        // Following columns use the round's original exclusions. Commit through a framing
        // transition, then sample the continuation in its updated eligibility domain.
        if ((was_end_suppressed != suppress_reasoning_end(impl_->preview_state, impl_->raw,
                                                          impl_->grammar_data != nullptr) ||
             (!impl_->grammar_data &&
              impl_->preview_replay.close_allowed() != impl_->replay.close_allowed())) &&
            index + 1 < tokens.size()) {
            return complete(count, FinishReason::None);
        }
        saw_tool_region =
            saw_tool_region || in_tool_region(impl_->preview_state, impl_->tool_output_enabled);
        // A round sampled with model stops excluded cannot emit <|im_end|> after
        // </tool_call>. Extra-accepted tokens past that close reopen another call
        // and keep suppression on until the output budget. Commit only through the
        // first completing close; the next round samples with stops eligible.
        if (stops_blocked_at_start && saw_tool_region &&
            model_stop_allowed(impl_->preview_state, impl_->raw, impl_->tool_output_enabled) &&
            index + 1 < tokens.size()) {
            return complete(count, FinishReason::None);
        }
    }

    const auto count = static_cast<std::uint32_t>(tokens.size());
    if (tokens.size() == budget_remaining) {
        terminalize(impl_->preview_state, impl_->policy, impl_->preview_output, count,
                    impl_->tool_output_enabled);
        return complete(count, limit_reason);
    }
    return complete(count, FinishReason::None);
}

runtime::OutputDecision OutputSession::preview_terminal(FinishReason reason) {
    if (impl_ == nullptr) { throw std::logic_error("output session is empty"); }
    if (impl_->state.terminal) { throw std::logic_error("output session is already terminal"); }
    if (impl_->preview_ready) { throw std::logic_error("output session already has a preview"); }
    if (reason == FinishReason::None || reason == FinishReason::StopString ||
        reason == FinishReason::StopToken) {
        throw std::invalid_argument("invalid between-round terminal decoder reason");
    }
    impl_->preview_state  = impl_->state;
    impl_->preview_replay = impl_->replay;
    impl_->reasoning_history.resize(impl_->committed_reasoning_size);
    impl_->preview_replay_discarded = 0;
    impl_->preview_replay_repairs   = 0;
    impl_->preview_output.clear();
    impl_->preview_channels.clear();
    terminalize(impl_->preview_state, impl_->policy, impl_->preview_output, 0,
                impl_->tool_output_enabled);
    if (impl_->token_grammar) { impl_->token_grammar->preview({}); }
    if (impl_->grammar_data && !impl_->grammar_data->definitions.empty() && !impl_->raw) {
        impl_->preview_tools = impl_->tools;
        impl_->preview_tools.filter(impl_->preview_output, *impl_->grammar_data, true);
    }
    impl_->preview_ready = true;
    return runtime::OutputDecision{.accepted_tokens = 0, .finish_reason = reason};
}

std::span<const std::optional<OutputChannel>>
OutputSession::preview_token_channels() const noexcept {
    if (impl_ == nullptr || !impl_->preview_ready) { return {}; }
    return impl_->preview_channels;
}

PublishedOutput OutputSession::commit_preview() noexcept {
    if (impl_ == nullptr || !impl_->preview_ready) { std::terminate(); }
    using std::swap;
    if (impl_->token_grammar) { impl_->token_grammar->commit_preview(); }
    if (impl_->grammar_data && !impl_->grammar_data->definitions.empty() && !impl_->raw) {
        swap(impl_->tools, impl_->preview_tools);
    }
    swap(impl_->state, impl_->preview_state);
    swap(impl_->replay, impl_->preview_replay);
    impl_->committed_reasoning_size = impl_->reasoning_history.size();
    impl_->replay_discarded         = impl_->preview_replay_discarded;
    impl_->replay_repairs           = impl_->preview_replay_repairs;
    PublishedOutput output          = std::move(impl_->preview_output);
    impl_->preview_output.clear();
    impl_->preview_ready = false;
    return output;
}

ReasoningReplayPublication OutputSession::filter_reasoning_replay(PublishedOutput output,
                                                                  FinishReason reason,
                                                                  bool caller_terminated) {
    if (!impl_ || impl_->preview_ready) {
        throw std::logic_error("replay publication requires a committed output session");
    }
    ReasoningReplayPublication result;
    auto& replay     = impl_->replay;
    const auto flush = [&] {
        result.output = std::move(impl_->replay_held);
        for (auto& delta : output) { result.output.push_back(std::move(delta)); }
        impl_->replay_held.clear();
    };
    if (caller_terminated) {
        flush();
        replay.enabled          = false;
        replay.active           = false;
        replay.retry_required   = false;
        impl_->replay_discarded = impl_->replay_repairs = 0;
        return result;
    }
    if (!replay.enabled) {
        flush();
        return result;
    }
    if (impl_->tools.selected_tool) { replay.accepted_answer = true; }
    const bool terminal_without_answer =
        reason != FinishReason::None && replay.repaired && !replay.accepted_answer;
    result.retry_required           = replay.retry_required || terminal_without_answer;
    result.framing_repairs          = std::exchange(impl_->replay_repairs, 0);
    result.discarded_content_tokens = std::exchange(impl_->replay_discarded, 0);
    if (result.retry_required) {
        result.discarded_content_tokens += static_cast<std::uint32_t>(replay.matched);
        replay.matched = 0;
    }
    if (result.framing_repairs != 0 || result.retry_required) {
        impl_->replay_held.clear();
        // The accepted preview ends exactly at the repair/proof token, so no
        // post-repair answer can share these content deltas.
        for (auto& delta : output) {
            if (delta.channel == OutputChannel::Reasoning) {
                result.output.push_back(std::move(delta));
            }
        }
        return result;
    }
    if (replay.active && reason == FinishReason::None) {
        for (auto& delta : output) {
            if (delta.channel == OutputChannel::Content) {
                impl_->replay_held.push_back(std::move(delta));
            } else {
                result.output.push_back(std::move(delta));
            }
        }
        return result;
    }
    // An uncorroborated copy may be a legitimate answer. Divergence or its
    // terminal boundary publishes all held bytes/logprobs unchanged and in order.
    flush();
    return result;
}

void OutputSession::discard_preview() noexcept {
    if (impl_ == nullptr || !impl_->preview_ready) { std::terminate(); }
    if (impl_->token_grammar) { impl_->token_grammar->discard_preview(); }
    impl_->preview_state  = impl_->state;
    impl_->preview_replay = impl_->replay;
    impl_->reasoning_history.resize(impl_->committed_reasoning_size);
    impl_->preview_output.clear();
    impl_->preview_ready = false;
}

std::uint32_t OutputSession::reasoning_tokens() const noexcept {
    return impl_ != nullptr ? impl_->state.reasoning_tokens : 0;
}

bool OutputSession::in_reasoning() const noexcept {
    return impl_ != nullptr && impl_->state.in_reasoning;
}

ReasoningLimitStats OutputSession::reasoning_limit_stats() const noexcept {
    return impl_ ? impl_->state.reasoning_limit : ReasoningLimitStats{};
}

bool OutputSession::reasoning_limit_retry_required() const noexcept {
    return impl_ && !impl_->state.terminal && impl_->state.reasoning_limit_retry_pending;
}

std::optional<TokenId> OutputSession::reasoning_limit_forced_close_token() const noexcept {
    if (impl_ && impl_->limit_action == runtime::ReasoningLimitAction::Close &&
        impl_->state.in_reasoning && impl_->state.reasoning_limit.budget_tokens != 0 &&
        impl_->state.reasoning_limit.used_tokens >= impl_->state.reasoning_limit.budget_tokens &&
        !impl_->state.terminal) {
        return impl_->reasoning_end_token;
    }
    return std::nullopt;
}

bool OutputSession::has_answer_content() const noexcept {
    return impl_ && impl_->state.reasoning_limit.budget_tokens != 0 &&
           impl_->state.has_non_whitespace_content && !impl_->tools.selected_tool;
}

bool OutputSession::preview_free_text_content() const noexcept {
    return impl_ && impl_->preview_ready && !impl_->raw &&
           !impl_->preview_state.constrained_content && !impl_->preview_tools.selected_tool;
}

bool OutputSession::model_stop_tokens_allowed() const noexcept {
    return impl_ == nullptr ||
           model_stop_allowed(impl_->state, impl_->raw, impl_->tool_output_enabled);
}

std::optional<TokenId> OutputSession::suppressed_reasoning_end_token() const noexcept {
    if (impl_ && !has_token_masks() &&
        suppress_reasoning_end(impl_->state, impl_->raw, impl_->grammar_data != nullptr)) {
        return impl_->reasoning_end_token;
    }
    return std::nullopt;
}

bool OutputSession::reasoning_cycle_exclusion_allowed(TokenId token) const {
    if (!in_reasoning() || impl_->raw || token < 0) { return false; }
    if (std::find(impl_->model_stop_tokens.begin(), impl_->model_stop_tokens.end(), token) !=
        impl_->model_stop_tokens.end()) {
        return false;
    }
    const auto candidate = impl_->state.think_marker_pending +
                           std::string(impl_->tokenizer->decode_token_bytes(token));
    return candidate.find(kThinkClose) == std::string::npos &&
           longest_suffix_prefix(candidate, kThinkClose, true) == 0;
}

bool OutputSession::has_token_grammar() const noexcept {
    return impl_ && impl_->token_grammar != nullptr;
}

bool OutputSession::has_token_masks() const noexcept {
    return has_token_grammar() || (impl_ && ((impl_->replay.enabled && impl_->replay.active) ||
                                             impl_->state.reasoning_limit.budget_tokens != 0));
}

std::vector<std::string> unconstrained_tool_call_names(std::string_view text,
                                                       std::size_t max_name_length) {
    const auto first = text.find(kToolOpen);
    if (first == std::string_view::npos) { return {}; }
    text.remove_prefix(first);
    auto whitespace = [&] {
        while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front()))) {
            text.remove_prefix(1);
        }
    };
    auto consume = [&](std::string_view marker) {
        whitespace();
        if (!text.starts_with(marker)) { return false; }
        text.remove_prefix(marker.size());
        return true;
    };
    std::vector<std::string> names;
    while (!text.empty()) {
        if (!consume(kToolOpen) || !consume("<function=")) { return {}; }
        const auto end = text.find('>');
        if (end == 0 || end == std::string_view::npos || end > max_name_length) { return {}; }
        const auto name = text.substr(0, end);
        if (!std::all_of(name.begin(), name.end(),
                         [](unsigned char c) { return std::isalnum(c) || c == '_' || c == '-'; })) {
            return {};
        }
        names.emplace_back(name);
        text.remove_prefix(end + 1);
        for (;;) {
            whitespace();
            if (text.starts_with("</function>")) { break; }
            if (!consume("<parameter=")) { return {}; }
            const auto key_end = text.find('>');
            if (key_end == 0 || key_end == std::string_view::npos) { return {}; }
            text.remove_prefix(key_end + 1);
            const auto value_end = text.find("</parameter>");
            if (value_end == std::string_view::npos) { return {}; }
            text.remove_prefix(value_end + std::string_view("</parameter>").size());
        }
        if (!consume("</function>") || !consume(kToolClose)) { return {}; }
        whitespace();
    }
    return names;
}

std::span<const ToolCall> OutputSession::tool_calls() const noexcept {
    return impl_ ? std::span<const ToolCall>(impl_->tools.calls) : std::span<const ToolCall>{};
}

std::shared_ptr<const GenerationRecoveryContext>
OutputSession::generation_recovery_context() const noexcept {
    return impl_ ? impl_->recovery_context : nullptr;
}

bool OutputSession::terminal() const noexcept { return impl_ && impl_->state.terminal; }

void OutputSession::fill_token_masks(std::span<const TokenId> tokens,
                                     std::span<const std::int32_t> parents,
                                     std::span<std::uint32_t> words) const {
    if (!has_token_masks()) { throw std::logic_error("output session has no token mask policy"); }
    if (!impl_->replay.enabled && impl_->state.reasoning_limit.budget_tokens == 0) {
        impl_->token_grammar->fill_masks(tokens, parents, words);
        return;
    }
    if (tokens.empty() || tokens.size() != parents.size()) {
        throw std::invalid_argument("invalid replay mask hypothesis shape");
    }

    struct Hypothesis {
        DecoderState decoder;
        ReasoningReplayState replay;
        std::vector<TokenId> reasoning_suffix;
    };

    std::vector<Hypothesis> nodes(tokens.size());
    std::vector<std::uint8_t> allowed(tokens.size());
    std::vector<std::uint8_t> reasoning_closes(tokens.size());
    nodes[0].decoder = impl_->state;
    nodes[0].replay  = impl_->replay;
    const StopPolicy no_stops;
    for (std::size_t node = 0; node < nodes.size(); ++node) {
        if (node != 0) {
            const auto parent = parents[node];
            if (parent < 0 || static_cast<std::size_t>(parent) >= node) {
                throw std::invalid_argument("replay mask parent must precede child");
            }
            nodes[node]      = nodes[parent];
            auto& hypothesis = nodes[node];
            auto& decoder    = hypothesis.decoder;
            auto& replay     = hypothesis.replay;
            const auto token = tokens[node];
            if (replay.close_allowed() && token == impl_->reasoning_end_token) {
                replay.repaired               = true;
                replay.matched                = 0;
                decoder.strip_content_leading = true;
            } else {
                const bool model_stop =
                    std::find(impl_->model_stop_tokens.begin(), impl_->model_stop_tokens.end(),
                              token) != impl_->model_stop_tokens.end();
                const auto bytes = impl_->tokenizer->decode_token_bytes(
                    token,
                    !impl_->preserve_special || (model_stop && !impl_->preserve_model_stops));
                const bool was_reasoning = decoder.in_reasoning;
                const bool was_tool      = in_tool_region(decoder, impl_->tool_output_enabled);
                const bool was_stripping = decoder.strip_content_leading;
                PublishedOutput ignored;
                if (was_reasoning && token == impl_->reasoning_end_token &&
                    decoder.reasoning_limit.budget_tokens != 0) {
                    flush_reasoning_fragments(decoder, no_stops, ignored, 1, nullptr,
                                              impl_->tool_output_enabled);
                }
                charge_reasoning_token(decoder, bytes, token == impl_->reasoning_end_token,
                                       model_stop);
                feed_token_bytes(decoder, bytes, no_stops, ignored, 1, nullptr,
                                 impl_->tool_output_enabled);
                const auto channel =
                    attributed_channel(was_reasoning, was_tool, was_stripping, decoder, bytes,
                                       false, false, impl_->tool_output_enabled);
                if (replay.enabled && !replay.reference) {
                    if (channel == OutputChannel::Reasoning) {
                        hypothesis.reasoning_suffix.push_back(token);
                    }
                    if (was_reasoning && !decoder.in_reasoning) {
                        // Only a hypothetical first close materializes a reference;
                        // ordinary replay forks share the committed immutable vector.
                        std::vector<TokenId> history(impl_->reasoning_history.begin(),
                                                     impl_->reasoning_history.begin() +
                                                         impl_->committed_reasoning_size);
                        history.insert(history.end(), hypothesis.reasoning_suffix.begin(),
                                       hypothesis.reasoning_suffix.end());
                        replay.freeze(history, token == impl_->reasoning_end_token);
                    }
                } else if (!was_reasoning) {
                    const bool terminal_model_stop =
                        model_stop &&
                        std::find(impl_->policy.token_ids.begin(), impl_->policy.token_ids.end(),
                                  token) != impl_->policy.token_ids.end();
                    replay.observe(token, channel, was_stripping && decoder.strip_content_leading,
                                   terminal_model_stop,
                                   std::any_of(bytes.begin(), bytes.end(), [](unsigned char byte) {
                                       return std::isspace(byte) == 0;
                                   }));
                }
            }
        }
        allowed[node]           = nodes[node].replay.close_allowed() ? 1 : 0;
        const auto& limit_state = nodes[node].decoder;
        reasoning_closes[node] =
            limit_state.in_reasoning && limit_state.reasoning_limit.budget_tokens != 0 &&
            limit_state.reasoning_limit.used_tokens >= limit_state.reasoning_limit.budget_tokens;
    }
    if (impl_->token_grammar) {
        impl_->token_grammar->fill_masks(tokens, parents, words, impl_->reasoning_end_token,
                                         allowed,
                                         impl_->grammar_data->starts_in_reasoning
                                             ? std::span<const std::uint8_t>(reasoning_closes)
                                             : std::span<const std::uint8_t>{});
    } else {
        if (words.size() != tokens.size() * fi::TokenGrammarState::mask_words) {
            throw std::invalid_argument("invalid replay token mask shape");
        }
        std::fill(words.begin(), words.end(), ~std::uint32_t{0});
        const auto token = static_cast<std::size_t>(impl_->reasoning_end_token);
        for (std::size_t node = 0; node < nodes.size(); ++node) {
            if (!nodes[node].decoder.in_reasoning && !allowed[node]) {
                words[node * fi::TokenGrammarState::mask_words + token / 32] &=
                    ~(std::uint32_t{1} << (token % 32));
            }
        }
    }
    const auto close = static_cast<std::size_t>(impl_->reasoning_end_token);
    for (std::size_t node = 0; node < nodes.size(); ++node) {
        const auto& state = nodes[node].decoder;
        if (!state.in_reasoning || state.reasoning_limit.budget_tokens == 0 ||
            state.reasoning_limit.used_tokens < state.reasoning_limit.budget_tokens) {
            continue;
        }
        auto row = words.subspan(node * fi::TokenGrammarState::mask_words,
                                 fi::TokenGrammarState::mask_words);
        if ((row[close / 32] & (std::uint32_t{1} << (close % 32))) == 0) {
            throw RequestError(RequestErrorKind::InvalidOutputConstraint,
                               "reasoning-limit close conflicts with output grammar");
        }
        std::fill(row.begin(), row.end(), 0);
        row[close / 32] = std::uint32_t{1} << (close % 32);
    }
}

Frontend::Frontend(std::shared_ptr<const Impl> impl) noexcept : impl_(std::move(impl)) {}

Frontend::Frontend(const Frontend&)                = default;
Frontend& Frontend::operator=(const Frontend&)     = default;
Frontend::Frontend(Frontend&&) noexcept            = default;
Frontend& Frontend::operator=(Frontend&&) noexcept = default;
Frontend::~Frontend()                              = default;

Frontend make_frontend(const FrontendResources& resources, bool vision_enabled) {
    return Frontend(std::make_shared<const Frontend::Impl>(resources, true, vision_enabled));
}

Frontend FrontendTestAccess::create_component(const FrontendResources& resources,
                                              bool vision_enabled) {
    return Frontend(std::make_shared<const Frontend::Impl>(resources, false, vision_enabled));
}

const PreparedPromptData& PreparedPromptAccess::view(const PreparedPrompt& prompt) {
    if (prompt.data_ == nullptr) { throw std::invalid_argument("prepared prompt is empty"); }
    return *prompt.data_;
}

PreparedPromptData PreparedPromptAccess::take(PreparedPrompt&& prompt) {
    if (prompt.data_ == nullptr) { throw std::invalid_argument("prepared prompt is empty"); }
    auto data = std::move(prompt).data_;
    return std::move(*data);
}

const PreparedPromptData& FrontendTestAccess::inspect(const PreparedPrompt& prompt) {
    return PreparedPromptAccess::view(prompt);
}

PreparedPrompt Frontend::prepare(PromptInput input) const {
    const auto start               = Clock::now();
    const PromptOptions& options   = input.options;
    const bool tool_output_enabled = !options.output_constraint && enables_tool_output(input);
    const auto grammar =
        options.add_generation_prompt
            ? impl_->token_grammar.compile(options.tool_jsons, options.enable_thinking,
                                           options.output_constraint)
            : nullptr;
    if (options.output_constraint && !options.add_generation_prompt) {
        throw RequestError(RequestErrorKind::InvalidOutputConstraint,
                           "output constraints require a generation prompt");
    }
    const auto recovery = GenerationRecoveryContext::analyze(input);
    const auto finish   = [&](std::unique_ptr<PreparedPromptData> data) {
        data->token_grammar       = grammar;
        data->generation_recovery = recovery;
        return PreparedPrompt(std::move(data));
    };
    std::vector<fi::ChatMessage> messages = convert_messages(std::move(input.messages));
    const bool has_media =
        std::any_of(messages.begin(), messages.end(),
                    [](const fi::ChatMessage& message) { return message.has_media(); });
    if (has_media && !impl_->vision_enabled) {
        throw std::invalid_argument("Vision is disabled for this Engine");
    }

    if (has_media) {
        fi::Processor processor(*impl_->tokenizer, impl_->chat_template, impl_->processor);
        fi::ProcessedInput processed;
        try {
            processed = processor.process(messages, render_options(options));
        } catch (const fi::ProcessorError& error) { throw_processor_error(error); }
        return finish(
            make_media_prompt_data(std::move(processed), options, tool_output_enabled, start));
    }
    const fi::RenderedChat rendered =
        impl_->chat_template.render(messages, render_options(options));
    fi::EncodedChat encoded = fi::encode_rendered_chat(*impl_->tokenizer, rendered);
    return finish(make_text_prompt_data(std::move(encoded), options, tool_output_enabled, start));
}

std::uint32_t Frontend::count_tokens(PromptInput input) const {
    const PromptOptions& options          = input.options;
    std::vector<fi::ChatMessage> messages = convert_messages(std::move(input.messages));
    const bool has_media =
        std::any_of(messages.begin(), messages.end(),
                    [](const fi::ChatMessage& message) { return message.has_media(); });
    if (has_media && !impl_->vision_enabled) {
        throw std::invalid_argument("Vision is disabled for this Engine");
    }
    if (!has_media) {
        const fi::RenderedChat rendered =
            impl_->chat_template.render(messages, render_options(options));
        return checked_token_count(
            impl_->tokenizer->encode(rendered.text, {}, rendered.literal_spans).size());
    }

    fi::Processor processor(*impl_->tokenizer, impl_->chat_template, impl_->processor);
    try {
        return checked_token_count(
            processor.process(messages, render_options(options)).input_ids.size());
    } catch (const fi::ProcessorError& error) { throw_processor_error(error); }
}

PromptCapabilities Frontend::prompt_capabilities() const noexcept {
    return impl_ != nullptr ? impl_->chat_template.capabilities() : PromptCapabilities{};
}

PreparedPrompt Frontend::prepare_tokens(std::vector<TokenId> token_ids,
                                        bool allow_prefix_identity) const {
    const auto start = Clock::now();
    (void)checked_token_count(token_ids.size());
    for (const TokenId token : token_ids) {
        if (!impl_->tokenizer->is_valid_token(token)) {
            throw std::out_of_range("prompt token is outside the checkpoint vocabulary: " +
                                    std::to_string(token));
        }
    }
    auto prepared              = std::make_unique<PreparedPromptData>();
    PreparedPromptData& result = *prepared;
    result.token_ids           = std::move(token_ids);
    assign_text_positions(result);
    result.identity.reusable = allow_prefix_identity;
    result.prepare.seconds   = std::chrono::duration<double>(Clock::now() - start).count();
    return PreparedPrompt(std::move(prepared));
}

namespace {

std::optional<std::vector<TokenId>>
encode_round_trip(const fi::Tokenizer& tokenizer, std::string_view text, std::string& decoded,
                  std::span<const fi::ByteSpan> literal_spans = {}) {
    decoded.clear();
    if (text.empty()) { return std::nullopt; }
    const std::vector<int> encoded = tokenizer.encode(text, {}, literal_spans);
    if (encoded.empty()) { return std::nullopt; }
    std::vector<TokenId> ids;
    ids.reserve(encoded.size());
    for (const int id : encoded) {
        if (!tokenizer.is_valid_token(id)) { return std::nullopt; }
        decoded += tokenizer.decode_token_bytes(id);
        ids.push_back(static_cast<TokenId>(id));
    }
    if (decoded != text) { return std::nullopt; }
    return ids;
}

std::optional<std::vector<TokenId>>
encode_recovery_fragment(const fi::Tokenizer& tokenizer,
                         const fi::CompiledChatTemplate& chat_template, const PromptInput& source,
                         std::span<const ChatMessage> insert, std::string& fragment,
                         std::string& decoded) {
    std::vector<ChatMessage> owned(insert.begin(), insert.end());
    const fi::ChatRenderOptions fragment_options{
        .add_generation_prompt = false,
        .enable_thinking       = source.options.enable_thinking,
        .reasoning_effort      = source.options.reasoning_effort,
        .preserve_thinking     = source.options.preserve_thinking,
        .add_vision_id         = false,
        .tool_jsons            = {},
    };
    const fi::RenderedFragment rendered_fragment =
        chat_template.render_fragment(convert_messages(std::move(owned)), fragment_options);
    fragment = rendered_fragment.text;
    return encode_round_trip(tokenizer, fragment, decoded, rendered_fragment.literal_spans);
}

bool insert_enables_tool_output(const PromptInput& source, std::span<const ChatMessage> insert) {
    if (enables_tool_output(source)) { return true; }
    return std::any_of(insert.begin(), insert.end(), [](const ChatMessage& message) {
        return message.role == ChatRole::Tool ||
               (message.role == ChatRole::Assistant && !message.tool_calls.empty());
    });
}

} // namespace

std::uint32_t Frontend::recovery_prompt_token_bound(const PreparedPrompt& prompt,
                                                    const GenerationRecoveryContext& context,
                                                    std::uint32_t maximum_attempts) const {
    if (!impl_ || !prompt.data_ || maximum_attempts > GenerationRecoveryContext::maximum_attempts) {
        throw std::invalid_argument("invalid bounded recovery envelope");
    }
    const auto original = checked_token_count(prompt.data_->token_ids.size());
    if (maximum_attempts == 0) { return original; }
    std::array<std::vector<TokenId>, 2> prologues;
    std::array<std::size_t, 2> closes{};
    const std::array prologue_text{kRecoveryThinkingPrologue, kRecoveryNonThinkingPrologue};
    const std::array close_text{kRecoveryTurnClose, kRecoveryContentTurnClose};
    for (std::size_t phase = 0; phase != 2; ++phase) {
        std::string decoded;
        auto opener = encode_round_trip(*impl_->tokenizer, prologue_text[phase], decoded);
        auto close  = encode_round_trip(*impl_->tokenizer, close_text[phase], decoded);
        if (!opener || !close) {
            throw std::invalid_argument("native recovery framing cannot round-trip");
        }
        prologues[phase] = std::move(*opener);
        closes[phase]    = close->size();
    }
    std::array<std::optional<std::uint64_t>, 2> growth;
    for (std::size_t phase = 0; phase != 2; ++phase) {
        const auto& ids = prologues[phase];
        if (prompt.data_->token_ids.size() >= ids.size() &&
            std::equal(ids.rbegin(), ids.rend(), prompt.data_->token_ids.rbegin())) {
            growth[phase] = 0;
        }
    }
    if (!growth[0] && !growth[1]) {
        throw std::invalid_argument("recovery envelope requires a native generation prologue");
    }
    for (std::uint32_t attempt = 1; attempt <= maximum_attempts; ++attempt) {
        std::size_t maximum_fragment = 0;
        for (const auto& insert : context.recovery_insert_variants(attempt)) {
            std::string fragment, decoded;
            const auto ids = encode_recovery_fragment(*impl_->tokenizer, impl_->chat_template,
                                                      context.input(), insert, fragment, decoded);
            if (!ids) { throw std::invalid_argument("native recovery feedback cannot round-trip"); }
            maximum_fragment = std::max(maximum_fragment, ids->size());
        }
        std::array<std::optional<std::uint64_t>, 2> next;
        for (std::size_t prior = 0; prior != 2; ++prior) {
            const auto prior_growth = growth[prior];
            if (!prior_growth) { continue; }
            for (std::size_t phase = 0; phase != 2; ++phase) {
                const auto value =
                    *prior_growth + closes[prior] + maximum_fragment + prologues[phase].size();
                next[phase] = std::max(next[phase].value_or(0), value);
            }
        }
        growth = next;
    }
    const auto bound = static_cast<std::uint64_t>(original) +
                       std::max(growth[0].value_or(0), growth[1].value_or(0));
    if (bound > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("recovery envelope exceeds the target token domain");
    }
    return static_cast<std::uint32_t>(bound);
}

std::optional<PreparedPrompt> Frontend::splice_recovery_prompt(
    std::vector<TokenId> prefix, const PromptInput& source, std::span<const ChatMessage> insert,
    std::shared_ptr<const GenerationRecoveryContext> recovery, bool enable_thinking) const {
    if (impl_ == nullptr || prefix.empty() || insert.empty()) { return std::nullopt; }
    try {
        std::string thinking_decoded;
        std::string content_decoded;
        const auto thinking_ids =
            encode_round_trip(*impl_->tokenizer, kRecoveryThinkingPrologue, thinking_decoded);
        const auto content_ids =
            encode_round_trip(*impl_->tokenizer, kRecoveryNonThinkingPrologue, content_decoded);
        if (!thinking_ids || !content_ids) { return std::nullopt; }
        const auto ends_with = [&](const auto& ids) {
            return prefix.size() >= ids.size() &&
                   std::equal(ids.rbegin(), ids.rend(), prefix.rbegin());
        };
        const bool prior_thinking = ends_with(*thinking_ids);
        if (!prior_thinking && !ends_with(*content_ids)) { return std::nullopt; }
        const auto& prior_ids = prior_thinking ? *thinking_ids : *content_ids;
        const auto prior_text =
            prior_thinking ? kRecoveryThinkingPrologue : kRecoveryNonThinkingPrologue;
        const auto& prior_decoded = prior_thinking ? thinking_decoded : content_decoded;
        const auto& prologue_ids  = enable_thinking ? *thinking_ids : *content_ids;
        const auto close_text     = prior_thinking ? kRecoveryTurnClose : kRecoveryContentTurnClose;
        std::string close_decoded;
        std::string fragment_decoded;
        const auto close_ids = encode_round_trip(*impl_->tokenizer, close_text, close_decoded);
        if (!close_ids) { return std::nullopt; }

        std::string fragment;
        const auto fragment_ids = encode_recovery_fragment(
            *impl_->tokenizer, impl_->chat_template, source, insert, fragment, fragment_decoded);
        if (!fragment_ids ||
            !recovery_suffix_tokens_ok(prefix, prior_ids, *close_ids, *fragment_ids, prior_text,
                                       prior_decoded, close_text, close_decoded, fragment,
                                       fragment_decoded)) {
            return std::nullopt;
        }

        prefix.reserve(prefix.size() + close_ids->size() + fragment_ids->size() +
                       prologue_ids.size());
        prefix.insert(prefix.end(), close_ids->begin(), close_ids->end());
        prefix.insert(prefix.end(), fragment_ids->begin(), fragment_ids->end());
        prefix.insert(prefix.end(), prologue_ids.begin(), prologue_ids.end());

        const auto start = Clock::now();
        auto data        = std::make_unique<PreparedPromptData>();
        data->token_ids  = std::move(prefix);
        assign_text_positions(*data);
        (void)checked_token_count(data->token_ids.size());
        data->identity.reusable           = true;
        data->identity.rewrite_checkpoint = RewriteCheckpointSpec{
            .kind     = RewriteCheckpointKind::ResponseReplay,
            .frontier = static_cast<std::uint32_t>(data->token_ids.size()),
        };
        data->starts_in_reasoning = enable_thinking;
        data->tool_output_enabled =
            !source.options.output_constraint && insert_enables_tool_output(source, insert);
        data->token_grammar = impl_->token_grammar.compile(
            source.options.tool_jsons, enable_thinking, source.options.output_constraint);
        data->generation_recovery = std::move(recovery);
        data->prepare.seconds     = std::chrono::duration<double>(Clock::now() - start).count();
        return PreparedPrompt(std::move(data));
    } catch (const std::invalid_argument&) {
        return std::nullopt;
    } catch (const std::out_of_range&) { return std::nullopt; }
}

OutputSession Frontend::make_output_session(const PreparedPrompt& prompt,
                                            const StopPolicy& caller_stop,
                                            const OutputOptions& output, bool generation_recovery,
                                            const OutputSession* recovery_source,
                                            runtime::ReasoningLimitAction action) const {
    if (prompt.data_ == nullptr) { throw std::invalid_argument("prepared prompt is empty"); }
    if (output.thinking_token_budget &&
        (*output.thinking_token_budget == 0 || output.raw ||
         (!prompt.data_->starts_in_reasoning && !recovery_source))) {
        throw std::invalid_argument(
            "thinking_token_budget requires positive budget and thinking output");
    }
    StopPolicy policy = merge_stop_policy(*impl_->tokenizer, caller_stop);
    if (output.raw) { policy.publish_stop_token = true; }
    auto session                = OutputSession(std::make_unique<OutputSession::Impl>(
        impl_->tokenizer, std::move(policy), output, prompt.data_->starts_in_reasoning,
        prompt.data_->tool_output_enabled, impl_->reasoning_end_token, impl_->defaults.token_ids,
        prompt.data_->token_grammar, prompt.data_->generation_recovery, generation_recovery));
    session.impl_->limit_action = action;
    if (recovery_source && recovery_source->impl_) {
        session.impl_->state.reasoning_limit = recovery_source->impl_->state.reasoning_limit;
        if (recovery_source->impl_->state.reasoning_limit_retry_pending) {
            ++session.impl_->state.reasoning_limit.retries;
        }
    }
    if (generation_recovery && recovery_source && recovery_source->impl_ &&
        recovery_source->impl_->replay.repaired && recovery_source->impl_->replay.reference &&
        !output.raw && !session.impl_->state.constrained_content) {
        session.impl_->replay.enabled   = true;
        session.impl_->replay.active    = true;
        session.impl_->replay.repaired  = true;
        session.impl_->replay.reference = recovery_source->impl_->replay.reference;
    }
    return session;
}

const StopPolicy& Frontend::default_stop_policy() const noexcept { return impl_->defaults; }

std::string_view Frontend::token_bytes(TokenId token) const {
    if (!impl_->tokenizer->is_valid_token(token)) {
        throw std::out_of_range("token is outside the checkpoint vocabulary: " +
                                std::to_string(token));
    }
    return impl_->tokenizer->decode_token_bytes(token);
}

PreparedPrompt EncodedHistoryPrepare::prepare(const Frontend& frontend, PromptInput input,
                                              frontend_internal::EncodedHistoryCache& cache,
                                              bool cache_system_prompt) {
    const auto start               = Clock::now();
    const PromptOptions& options   = input.options;
    const bool tool_output_enabled = !options.output_constraint && enables_tool_output(input);
    const auto grammar =
        options.add_generation_prompt
            ? frontend.impl_->token_grammar.compile(options.tool_jsons, options.enable_thinking,
                                                    options.output_constraint)
            : nullptr;
    if (options.output_constraint && !options.add_generation_prompt) {
        throw RequestError(RequestErrorKind::InvalidOutputConstraint,
                           "output constraints require a generation prompt");
    }
    const auto recovery = GenerationRecoveryContext::analyze(input);
    const auto finish   = [&](std::unique_ptr<PreparedPromptData> data) {
        data->token_grammar       = grammar;
        data->generation_recovery = recovery;
        return PreparedPrompt(std::move(data));
    };
    auto rendering                        = render_options(options);
    rendering.mark_system_prompt          = cache_system_prompt;
    std::vector<fi::ChatMessage> messages = convert_messages(std::move(input.messages));
    const bool has_media =
        std::any_of(messages.begin(), messages.end(),
                    [](const fi::ChatMessage& message) { return message.has_media(); });
    if (has_media && !frontend.impl_->vision_enabled) {
        throw std::invalid_argument("Vision is disabled for this Engine");
    }
    if (has_media) {
        fi::last_host_encode_observation = {};
        fi::Processor processor(*frontend.impl_->tokenizer, frontend.impl_->chat_template,
                                frontend.impl_->processor);
        fi::ProcessedInput processed;
        try {
            processed = processor.process(messages, rendering);
        } catch (const fi::ProcessorError& error) { throw_processor_error(error); }
        return finish(
            make_media_prompt_data(std::move(processed), options, tool_output_enabled, start));
    }
    fi::EncodedChat encoded = fi::encode_chat_with_cache(
        *frontend.impl_->tokenizer, frontend.impl_->chat_template, messages, rendering, cache);
    return finish(make_text_prompt_data(std::move(encoded), options, tool_output_enabled, start));
}

std::uint32_t EncodedHistoryPrepare::count_tokens(const Frontend& frontend, PromptInput input,
                                                  frontend_internal::EncodedHistoryCache& cache) {
    const PromptOptions& options          = input.options;
    std::vector<fi::ChatMessage> messages = convert_messages(std::move(input.messages));
    const bool has_media =
        std::any_of(messages.begin(), messages.end(),
                    [](const fi::ChatMessage& message) { return message.has_media(); });
    if (has_media && !frontend.impl_->vision_enabled) {
        throw std::invalid_argument("Vision is disabled for this Engine");
    }
    if (!has_media) {
        const fi::EncodedChat encoded =
            fi::encode_chat_with_cache(*frontend.impl_->tokenizer, frontend.impl_->chat_template,
                                       messages, render_options(options), cache);
        return checked_token_count(encoded.input_ids.size());
    }
    fi::last_host_encode_observation = {};
    fi::Processor processor(*frontend.impl_->tokenizer, frontend.impl_->chat_template,
                            frontend.impl_->processor);
    try {
        return checked_token_count(
            processor.process(messages, render_options(options)).input_ids.size());
    } catch (const fi::ProcessorError& error) { throw_processor_error(error); }
}

} // namespace ninfer::targets::qwen3_6
