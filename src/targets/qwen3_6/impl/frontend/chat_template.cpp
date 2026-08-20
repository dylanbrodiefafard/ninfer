#include "targets/qwen3_6/impl/frontend/chat_template.h"

#include "targets/qwen3_6/impl/frontend/digest.h"

#include <nlohmann/json.hpp>

#include <array>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::targets::qwen3_6::frontend_internal {
namespace {

using OrderedJson = nlohmann::ordered_json;

constexpr Sha256Digest kThinkingToggleTemplateDigest{
    0xe8, 0x4f, 0x32, 0xa2, 0x3f, 0xdd, 0xa2, 0x76, 0x89, 0xf8, 0x68, 0xaa, 0x4a, 0x1a, 0x56, 0x21,
    0xf4, 0x11, 0x33, 0xe5, 0x1a, 0x48, 0xd7, 0xf3, 0xef, 0xcb, 0xea, 0x28, 0x39, 0x57, 0x42, 0x59,
};

constexpr Sha256Digest kReasoningEffortTemplateDigest{
    0xc3, 0xcf, 0x9e, 0x34, 0xab, 0xf4, 0xf9, 0xe3, 0x6c, 0x2d, 0x72, 0x16, 0x5a, 0xa9, 0xc1, 0x32,
    0xd3, 0xe2, 0xa7, 0x25, 0xb6, 0xc2, 0x58, 0x6a, 0xaa, 0x3a, 0x8a, 0xf9, 0xd7, 0xa8, 0x10, 0x41,
};

constexpr std::string_view kLowReasoningInstructions =
    "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to "
    "the conclusion without unnecessary elaboration.";

constexpr std::string_view kXHighReasoningInstructions =
    "Reasoning effort is set to xhigh. Please think carefully through the task, validate key "
    "assumptions, consider plausible alternatives, and prioritize correctness, consistency, and "
    "clarity in the final answer.";

bool is_instruction_role(ChatRole role) noexcept {
    return role == ChatRole::System || role == ChatRole::Developer;
}

void validate_instruction_message(const ChatMessage& message) {
    if (message.has_media()) {
        throw std::invalid_argument(
            "system and developer messages cannot contain images or videos");
    }
    if (!message.reasoning_content.empty() || !message.tool_calls.empty() ||
        !message.tool_call_id.empty()) {
        throw std::invalid_argument("system and developer messages may contain only text content");
    }
}

std::string trim_ascii_whitespace(const std::string& text) {
    std::size_t begin = 0;
    while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin])) != 0) {
        ++begin;
    }

    std::size_t end = text.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1])) != 0) { --end; }
    return text.substr(begin, end - begin);
}

bool starts_with(const std::string& text, std::string_view prefix) {
    return text.size() >= prefix.size() && text.compare(0, prefix.size(), prefix) == 0;
}

bool ends_with(const std::string& text, std::string_view suffix) {
    return text.size() >= suffix.size() &&
           text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

long last_real_user_query(const std::vector<ChatMessage>& messages) {
    for (long i = static_cast<long>(messages.size()) - 1; i >= 0; --i) {
        const ChatMessage& message = messages[static_cast<std::size_t>(i)];
        if (message.role != ChatRole::User) { continue; }
        const std::string content = trim_ascii_whitespace(message.rendered_content());
        if (!(starts_with(content, "<tool_response>") && ends_with(content, "</tool_response>"))) {
            return i;
        }
    }
    throw std::invalid_argument("no user query found in chat messages");
}

std::string lstrip_newlines(std::string text) {
    std::size_t begin = 0;
    while (begin < text.size() && text[begin] == '\n') { ++begin; }
    return text.substr(begin);
}

std::string rstrip_newlines(std::string text) {
    std::size_t end = text.size();
    while (end > 0 && text[end - 1] == '\n') { --end; }
    return text.substr(0, end);
}

// Split an assistant turn into (reasoning, content) exactly as the Qwen3.6 jinja
// does when reasoning_content is not provided: reasoning is the text between the
// last <think> and the first </think>; content is everything after the last
// </think>. When there is no </think> the whole thing is content and reasoning is
// empty.
struct ThinkParts {
    std::string reasoning;
    std::string content;
};

ThinkParts derive_think_parts(const std::string& content) {
    ThinkParts parts;
    const std::size_t first_close = content.find("</think>");
    if (first_close == std::string::npos) {
        parts.content = content;
        return parts;
    }
    // reasoning = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n')
    std::string before          = rstrip_newlines(content.substr(0, first_close));
    const std::size_t last_open = before.rfind("<think>");
    std::string reasoning       = (last_open == std::string::npos)
                                      ? before
                                      : before.substr(last_open + std::string("<think>").size());
    parts.reasoning             = lstrip_newlines(std::move(reasoning));
    // content = content.split('</think>')[-1].lstrip('\n')
    const std::size_t last_close = content.rfind("</think>");
    parts.content = lstrip_newlines(content.substr(last_close + std::string("</think>").size()));
    return parts;
}

constexpr std::string_view kToolInstructions =
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
    "<tool_call>\n"
    "<function=example_function_name>\n"
    "<parameter=example_parameter_1>\n"
    "value_1\n"
    "</parameter>\n"
    "<parameter=example_parameter_2>\n"
    "This is the value for the second parameter\n"
    "that can span\n"
    "multiple lines\n"
    "</parameter>\n"
    "</function>\n"
    "</tool_call>\n\n"
    "<IMPORTANT>\n"
    "Reminder:\n"
    "- Function calls MUST follow the specified format: an inner <function=...></function> block "
    "must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n"
    "- You may provide optional reasoning for your function call in natural language BEFORE the "
    "function call, but NOT after\n"
    "- If there is no function call available, answer the question like normal with your current "
    "knowledge and do not tell the user about function calls\n"
    "</IMPORTANT>";

struct JsonWalkError {};

void skip_json_ws(std::string_view text, std::size_t& index) {
    while (index < text.size()) {
        const char byte = text[index];
        if (byte != ' ' && byte != '\t' && byte != '\n' && byte != '\r') { return; }
        ++index;
    }
}

bool json_ident_continue(char byte) noexcept {
    return (byte >= '0' && byte <= '9') || (byte >= 'a' && byte <= 'z') ||
           (byte >= 'A' && byte <= 'Z') || byte == '_';
}

std::uint32_t json_hex4(std::string_view text, std::size_t index) {
    if (index + 4 > text.size()) { throw JsonWalkError{}; }
    std::uint32_t value = 0;
    for (std::size_t i = 0; i < 4; ++i) {
        const char byte = text[index + i];
        value <<= 4U;
        if (byte >= '0' && byte <= '9') {
            value |= static_cast<std::uint32_t>(byte - '0');
        } else if (byte >= 'a' && byte <= 'f') {
            value |= static_cast<std::uint32_t>(byte - 'a' + 10);
        } else if (byte >= 'A' && byte <= 'F') {
            value |= static_cast<std::uint32_t>(byte - 'A' + 10);
        } else {
            throw JsonWalkError{};
        }
    }
    return value;
}

void append_utf8(std::string& out, std::uint32_t codepoint) {
    if (codepoint < 0x80U) {
        out.push_back(static_cast<char>(codepoint));
    } else if (codepoint < 0x800U) {
        out.push_back(static_cast<char>(0xC0U | (codepoint >> 6U)));
        out.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    } else if (codepoint < 0x10000U) {
        out.push_back(static_cast<char>(0xE0U | (codepoint >> 12U)));
        out.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
        out.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    } else if (codepoint <= 0x10FFFFU) {
        out.push_back(static_cast<char>(0xF0U | (codepoint >> 18U)));
        out.push_back(static_cast<char>(0x80U | ((codepoint >> 12U) & 0x3FU)));
        out.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
        out.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    } else {
        throw JsonWalkError{};
    }
}

void append_json_escaped(std::string& out, std::string_view text) {
    out.push_back('"');
    for (const unsigned char byte : text) {
        switch (byte) {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\b':
            out += "\\b";
            break;
        case '\f':
            out += "\\f";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            if (byte < 0x20U) {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "\\u%04x", byte);
                out += buf;
            } else {
                out.push_back(static_cast<char>(byte));
            }
            break;
        }
    }
    out.push_back('"');
}

std::string decode_json_string(std::string_view text, std::size_t& index) {
    skip_json_ws(text, index);
    if (index >= text.size() || text[index] != '"') { throw JsonWalkError{}; }
    ++index;
    std::string out;
    while (index < text.size()) {
        const unsigned char byte = static_cast<unsigned char>(text[index++]);
        if (byte == '"') { return out; }
        if (byte == '\\') {
            if (index >= text.size()) { throw JsonWalkError{}; }
            const char escape = text[index++];
            switch (escape) {
            case '"':
            case '\\':
            case '/':
                out.push_back(escape);
                break;
            case 'b':
                out.push_back('\b');
                break;
            case 'f':
                out.push_back('\f');
                break;
            case 'n':
                out.push_back('\n');
                break;
            case 'r':
                out.push_back('\r');
                break;
            case 't':
                out.push_back('\t');
                break;
            case 'u': {
                std::uint32_t codepoint = json_hex4(text, index);
                index += 4;
                if (codepoint >= 0xD800U && codepoint <= 0xDBFFU) {
                    if (index + 6 > text.size() || text[index] != '\\' || text[index + 1] != 'u') {
                        throw JsonWalkError{};
                    }
                    index += 2;
                    const std::uint32_t low = json_hex4(text, index);
                    index += 4;
                    if (low < 0xDC00U || low > 0xDFFFU) { throw JsonWalkError{}; }
                    codepoint = 0x10000U + ((codepoint - 0xD800U) << 10U) + (low - 0xDC00U);
                } else if (codepoint >= 0xDC00U && codepoint <= 0xDFFFU) {
                    throw JsonWalkError{};
                }
                append_utf8(out, codepoint);
                break;
            }
            default:
                throw JsonWalkError{};
            }
        } else if (byte < 0x20U) {
            throw JsonWalkError{};
        } else {
            out.push_back(static_cast<char>(byte));
        }
    }
    throw JsonWalkError{};
}

void append_tojson_value(std::string& out, std::string_view text, std::size_t& index);

void append_tojson_value(std::string& out, std::string_view text, std::size_t& index) {
    skip_json_ws(text, index);
    if (index >= text.size()) { throw JsonWalkError{}; }
    const char byte = text[index];
    if (byte == '"') {
        const std::string decoded = decode_json_string(text, index);
        append_json_escaped(out, decoded);
        return;
    }
    if (byte == '{') {
        ++index;
        out.push_back('{');
        skip_json_ws(text, index);
        bool first = true;
        while (index < text.size() && text[index] != '}') {
            if (!first) {
                if (text[index] != ',') { throw JsonWalkError{}; }
                ++index;
                skip_json_ws(text, index);
                if (index < text.size() && text[index] == '}') { throw JsonWalkError{}; }
                out += ", ";
            }
            first               = false;
            const std::string key = decode_json_string(text, index);
            append_json_escaped(out, key);
            skip_json_ws(text, index);
            if (index >= text.size() || text[index] != ':') { throw JsonWalkError{}; }
            ++index;
            out += ": ";
            append_tojson_value(out, text, index);
            skip_json_ws(text, index);
        }
        if (index >= text.size() || text[index] != '}') { throw JsonWalkError{}; }
        ++index;
        out.push_back('}');
        return;
    }
    if (byte == '[') {
        ++index;
        out.push_back('[');
        skip_json_ws(text, index);
        bool first = true;
        while (index < text.size() && text[index] != ']') {
            if (!first) {
                if (text[index] != ',') { throw JsonWalkError{}; }
                ++index;
                skip_json_ws(text, index);
                if (index < text.size() && text[index] == ']') { throw JsonWalkError{}; }
                out += ", ";
            }
            first = false;
            append_tojson_value(out, text, index);
            skip_json_ws(text, index);
        }
        if (index >= text.size() || text[index] != ']') { throw JsonWalkError{}; }
        ++index;
        out.push_back(']');
        return;
    }
    if (text.compare(index, 4, "true") == 0 &&
        (index + 4 == text.size() || !json_ident_continue(text[index + 4]))) {
        out += "true";
        index += 4;
        return;
    }
    if (text.compare(index, 5, "false") == 0 &&
        (index + 5 == text.size() || !json_ident_continue(text[index + 5]))) {
        out += "false";
        index += 5;
        return;
    }
    if (text.compare(index, 4, "null") == 0 &&
        (index + 4 == text.size() || !json_ident_continue(text[index + 4]))) {
        out += "null";
        index += 4;
        return;
    }
    const std::size_t start = index;
    if (text[index] == '-') { ++index; }
    if (index >= text.size() || text[index] < '0' || text[index] > '9') { throw JsonWalkError{}; }
    if (text[index] == '0') {
        ++index;
        if (index < text.size() && text[index] >= '0' && text[index] <= '9') {
            throw JsonWalkError{};
        }
    } else {
        while (index < text.size() && text[index] >= '0' && text[index] <= '9') { ++index; }
    }
    if (index < text.size() && text[index] == '.') {
        ++index;
        if (index >= text.size() || text[index] < '0' || text[index] > '9') { throw JsonWalkError{}; }
        while (index < text.size() && text[index] >= '0' && text[index] <= '9') { ++index; }
    }
    if (index < text.size() && (text[index] == 'e' || text[index] == 'E')) {
        ++index;
        if (index < text.size() && (text[index] == '+' || text[index] == '-')) { ++index; }
        if (index >= text.size() || text[index] < '0' || text[index] > '9') { throw JsonWalkError{}; }
        while (index < text.size() && text[index] >= '0' && text[index] <= '9') { ++index; }
    }
    out.append(text.data() + start, index - start);
}

std::string tojson_text(const OrderedJson& value) {
    if (value.is_array()) {
        std::string rendered = "[";
        for (std::size_t index = 0; index < value.size(); ++index) {
            if (index != 0) { rendered += ", "; }
            rendered += tojson_text(value[index]);
        }
        rendered += "]";
        return rendered;
    }
    if (value.is_object()) {
        std::string rendered = "{";
        std::size_t index    = 0;
        for (auto it = value.begin(); it != value.end(); ++it, ++index) {
            if (index != 0) { rendered += ", "; }
            rendered += OrderedJson(it.key()).dump();
            rendered += ": ";
            rendered += tojson_text(it.value());
        }
        rendered += "}";
        return rendered;
    }
    return value.dump();
}

std::string tojson_text_from_json(std::string_view json) {
    try {
        std::size_t index = 0;
        std::string out;
        out.reserve(json.size() + json.size() / 8);
        append_tojson_value(out, json, index);
        skip_json_ws(json, index);
        if (index != json.size()) { throw JsonWalkError{}; }
        return out;
    } catch (const JsonWalkError&) { return tojson_text(OrderedJson::parse(json)); }
}

std::string parameter_text(const OrderedJson& value) {
    if (value.is_string()) { return value.get<std::string>(); }
    return tojson_text(value);
}

std::string render_tool_call_from_json(const ToolCall& call) {
    std::size_t index = 0;
    skip_json_ws(call.arguments_json, index);
    if (index >= call.arguments_json.size() || call.arguments_json[index] != '{') {
        throw JsonWalkError{};
    }
    ++index;
    std::string rendered;
    rendered += "<tool_call>\n<function=";
    rendered += call.name;
    rendered += ">\n";
    skip_json_ws(call.arguments_json, index);
    bool first = true;
    while (index < call.arguments_json.size() && call.arguments_json[index] != '}') {
        if (!first) {
            if (call.arguments_json[index] != ',') { throw JsonWalkError{}; }
            ++index;
            skip_json_ws(call.arguments_json, index);
            if (index < call.arguments_json.size() && call.arguments_json[index] == '}') {
                throw JsonWalkError{};
            }
        }
        first = false;
        const std::string key = decode_json_string(call.arguments_json, index);
        skip_json_ws(call.arguments_json, index);
        if (index >= call.arguments_json.size() || call.arguments_json[index] != ':') {
            throw JsonWalkError{};
        }
        ++index;
        skip_json_ws(call.arguments_json, index);
        std::string value;
        if (index < call.arguments_json.size() && call.arguments_json[index] == '"') {
            value = decode_json_string(call.arguments_json, index);
        } else {
            append_tojson_value(value, call.arguments_json, index);
        }
        rendered += "<parameter=";
        rendered += key;
        rendered += ">\n";
        rendered += value;
        rendered += "\n</parameter>\n";
        skip_json_ws(call.arguments_json, index);
    }
    if (index >= call.arguments_json.size() || call.arguments_json[index] != '}') {
        throw JsonWalkError{};
    }
    ++index;
    skip_json_ws(call.arguments_json, index);
    if (index != call.arguments_json.size()) { throw JsonWalkError{}; }
    rendered += "</function>\n</tool_call>";
    return rendered;
}

std::string render_tool_call(const ToolCall& call, bool allow_empty_arguments) {
    if (allow_empty_arguments && call.arguments_json.empty()) {
        return "<tool_call>\n<function=" + call.name + ">\n</function>\n</tool_call>";
    }
    try {
        return render_tool_call_from_json(call);
    } catch (const JsonWalkError&) {
        OrderedJson args = OrderedJson::parse(call.arguments_json);
        if (!args.is_object()) {
            throw std::invalid_argument("tool call arguments must be a JSON object");
        }
        std::string rendered;
        rendered += "<tool_call>\n<function=";
        rendered += call.name;
        rendered += ">\n";
        for (auto it = args.begin(); it != args.end(); ++it) {
            rendered += "<parameter=";
            rendered += it.key();
            rendered += ">\n";
            rendered += parameter_text(it.value());
            rendered += "\n</parameter>\n";
        }
        rendered += "</function>\n</tool_call>";
        return rendered;
    }
}

std::string render_tools_system_block(const std::vector<std::string>& tool_jsons,
                                      const std::string& leading_instruction,
                                      std::string_view reasoning_instructions) {
    std::string rendered;
    rendered += "<|im_start|>system\n";
    if (!reasoning_instructions.empty()) {
        rendered += reasoning_instructions;
        rendered += "\n\n";
    }
    rendered += "# Tools\n\nYou have access to the following functions:\n\n<tools>";
    for (const std::string& tool : tool_jsons) {
        rendered += "\n";
        rendered += tojson_text_from_json(tool);
    }
    rendered += "\n</tools>";
    rendered += std::string(kToolInstructions);
    if (!leading_instruction.empty()) {
        rendered += "\n\n";
        rendered += leading_instruction;
    }
    rendered += "<|im_end|>\n";
    return rendered;
}

std::string_view resolve_reasoning_instructions(ChatTemplateSemantics semantics,
                                                const ChatRenderOptions& options) {
    if (semantics == ChatTemplateSemantics::ThinkingToggle) {
        if (options.reasoning_effort) {
            throw std::invalid_argument("loaded chat template does not support reasoning effort");
        }
        return {};
    }
    if (!options.enable_thinking) {
        if (options.reasoning_effort) {
            throw std::invalid_argument(
                "reasoning effort cannot be combined with disabled thinking");
        }
        return {};
    }

    switch (options.reasoning_effort.value_or(ReasoningEffort::XHigh)) {
    case ReasoningEffort::Low:
        return kLowReasoningInstructions;
    case ReasoningEffort::Medium:
        return {};
    case ReasoningEffort::XHigh:
        return kXHighReasoningInstructions;
    }
    throw std::invalid_argument("invalid reasoning effort");
}

} // namespace

bool ChatMessage::has_media() const noexcept {
    for (const ChatPart& part : parts) {
        if (part.kind != ChatPartKind::Text) { return true; }
    }
    return false;
}

std::string ChatMessage::rendered_content(bool add_vision_id, int* image_count,
                                          int* video_count) const {
    int local_images = 0;
    int local_videos = 0;
    int& images      = image_count == nullptr ? local_images : *image_count;
    int& videos      = video_count == nullptr ? local_videos : *video_count;
    std::string out;
    for (const ChatPart& part : parts) {
        switch (part.kind) {
        case ChatPartKind::Text:
            out += part.text;
            break;
        case ChatPartKind::Image:
            ++images;
            if (add_vision_id) { out += "Picture " + std::to_string(images) + ": "; }
            out += "<|vision_start|><|image_pad|><|vision_end|>";
            break;
        case ChatPartKind::Video:
            ++videos;
            if (add_vision_id) { out += "Video " + std::to_string(videos) + ": "; }
            out += "<|vision_start|><|video_pad|><|vision_end|>";
            break;
        }
    }
    return out;
}

CompiledChatTemplate CompiledChatTemplate::resolve(std::string_view source) {
    const Sha256Digest digest = sha256(source);
    if (digest == kThinkingToggleTemplateDigest) {
        return CompiledChatTemplate(ChatTemplateSemantics::ThinkingToggle);
    }
    if (digest == kReasoningEffortTemplateDigest) {
        return CompiledChatTemplate(ChatTemplateSemantics::ReasoningEffort);
    }
    throw std::invalid_argument("unsupported frontend/chat_template.jinja (sha256 " +
                                sha256_hex(digest) + ")");
}

PromptCapabilities CompiledChatTemplate::capabilities() const noexcept {
    PromptCapabilities result;
    result.enable_thinking = true;
    if (semantics_ == ChatTemplateSemantics::ReasoningEffort) {
        result.reasoning_effort.low            = true;
        result.reasoning_effort.medium         = true;
        result.reasoning_effort.xhigh          = true;
        result.reasoning_effort.default_effort = ReasoningEffort::XHigh;
    }
    return result;
}

RenderedChat CompiledChatTemplate::render(const std::vector<ChatMessage>& messages,
                                          ChatRenderOptions options) const {
    if (messages.empty()) { throw std::invalid_argument("chat messages must not be empty"); }

    const bool effort_template = semantics_ == ChatTemplateSemantics::ReasoningEffort;
    const std::string_view reasoning_instructions =
        resolve_reasoning_instructions(semantics_, options);

    std::size_t message_begin = 0;
    std::string leading_instruction;
    if (is_instruction_role(messages[0].role)) {
        validate_instruction_message(messages[0]);
        leading_instruction = trim_ascii_whitespace(messages[0].rendered_content());
        message_begin       = 1;
    }

    std::string rendered;
    const bool has_tools = !options.tool_jsons.empty();
    if (has_tools) {
        rendered += render_tools_system_block(options.tool_jsons, leading_instruction,
                                              reasoning_instructions);
    } else if (message_begin == 1) {
        if (!effort_template || !leading_instruction.empty() || !reasoning_instructions.empty()) {
            rendered += "<|im_start|>system\n";
            if (!reasoning_instructions.empty()) {
                rendered += reasoning_instructions;
                if (!leading_instruction.empty()) { rendered += "\n\n"; }
            }
            rendered += leading_instruction;
            rendered += "<|im_end|>\n";
        }
    } else if (!reasoning_instructions.empty()) {
        rendered += "<|im_start|>system\n";
        rendered += reasoning_instructions;
        rendered += "<|im_end|>\n";
    }

    const long last_query_index  = last_real_user_query(messages);
    const bool preserve_thinking = options.preserve_thinking.value_or(effort_template);
    std::optional<RewriteCheckpointByteSpec> rewrite_checkpoint;

    int image_count = 0;
    int video_count = 0;
    for (std::size_t i = 0; i < messages.size(); ++i) {
        const ChatMessage& message = messages[i];
        if (i < message_begin) { continue; }
        if (is_instruction_role(message.role)) { validate_instruction_message(message); }
        const std::string content = trim_ascii_whitespace(
            message.rendered_content(options.add_vision_id, &image_count, &video_count));
        if (is_instruction_role(message.role)) {
            rendered += "<|im_start|>system\n";
            rendered += content;
            rendered += "<|im_end|>\n";
            continue;
        }
        if (message.role == ChatRole::User) {
            rendered += "<|im_start|>user\n";
            rendered += content;
            rendered += "<|im_end|>\n";
            continue;
        }
        if (message.role == ChatRole::Tool) {
            const bool opens_group = i > 0 && messages[i - 1].role != ChatRole::Tool;
            const bool closes_group =
                i + 1 == messages.size() || messages[i + 1].role != ChatRole::Tool;
            if (opens_group) { rendered += "<|im_start|>user"; }
            rendered += "\n<tool_response>\n";
            rendered += content;
            rendered += "\n</tool_response>";
            if (closes_group) { rendered += "<|im_end|>\n"; }
            continue;
        }

        if (message.role != ChatRole::Assistant) {
            throw std::invalid_argument("unsupported chat role value");
        }

        // assistant
        std::string reasoning;
        std::string body = content;
        if (!message.reasoning_content.empty()) {
            reasoning = message.reasoning_content;
        } else if (!effort_template) {
            ThinkParts parts = derive_think_parts(content);
            reasoning        = std::move(parts.reasoning);
            body             = std::move(parts.content);
        }
        reasoning = trim_ascii_whitespace(reasoning);

        const bool keep_thinking = preserve_thinking || (static_cast<long>(i) > last_query_index);
        rendered += "<|im_start|>assistant\n";
        if (!preserve_thinking && !rewrite_checkpoint && static_cast<long>(i) > last_query_index) {
            rewrite_checkpoint = RewriteCheckpointByteSpec{
                .kind = RewriteCheckpointKind::TurnClosure, .offset = rendered.size()};
        }
        // Official Qwen3.8 Jinja still wraps whenever keep_thinking. The C++ clone
        // omits an empty reasoning wrapper so history does not inject the
        // no-thinking cue `<think>\n\n</think>\n\n`.
        if (keep_thinking && !(effort_template && reasoning.empty())) {
            rendered += "<think>\n";
            rendered += reasoning;
            rendered += "\n</think>\n\n";
        }
        rendered += body;
        if (!message.tool_calls.empty()) {
            const bool body_has_text = !trim_ascii_whitespace(body).empty();
            for (std::size_t call_index = 0; call_index < message.tool_calls.size(); ++call_index) {
                if (call_index == 0) {
                    if (body_has_text) { rendered += "\n\n"; }
                } else {
                    rendered += "\n";
                }
                rendered += render_tool_call(message.tool_calls[call_index], effort_template);
            }
        }
        rendered += "<|im_end|>\n";
    }

    if (options.add_generation_prompt) {
        rendered += "<|im_start|>assistant\n";
        if (!preserve_thinking && !rewrite_checkpoint) {
            rewrite_checkpoint = RewriteCheckpointByteSpec{
                .kind = RewriteCheckpointKind::TurnClosure, .offset = rendered.size()};
        }
        if (options.enable_thinking) {
            rendered += "<think>\n";
        } else {
            rendered += "<think>\n\n</think>\n\n";
        }
        if (preserve_thinking) {
            // Response replay retains the deterministic generation prologue. This is the prompt
            // frontier for both thinking modes, so capturing it does not split off a tiny final
            // prefill unit. The complete rendered prefix is tokenized independently below.
            rewrite_checkpoint = RewriteCheckpointByteSpec{
                .kind = RewriteCheckpointKind::ResponseReplay, .offset = rendered.size()};
        }
    }
    return RenderedChat{.text = std::move(rendered), .rewrite_checkpoint = rewrite_checkpoint};
}

} // namespace ninfer::targets::qwen3_6::frontend_internal
