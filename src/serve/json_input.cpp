#include "serve/json_input.h"

#include "serve/request.h"
#include "text/json_input.h"

#include <optional>
#include <string>

namespace ninfer::serve {

nlohmann::json parse_request_body(std::string_view source, JsonRequestProtocol protocol) {
    const auto parsed = text::parse_json_numbers(source);
    const auto check  = [&](const std::string& path, bool tool = false,
                            const std::string& name = {}) {
        if (const auto location = text::inexact_schema_number(parsed, path)) {
            ApiError error;
            error.status = 400;
            error.message =
                (name.empty() ? "" : "tool " + name + ": ") +
                "schema numeric value cannot be preserved by the JSON representation at " +
                *location;
            error.param = *location;
            error.code  = tool ? "invalid_tool_schema" : "invalid_output_constraint";
            throw ApiException(std::move(error));
        }
    };
    std::optional<std::string> format_path;
    switch (protocol) {
    case JsonRequestProtocol::ChatCompletions:
        format_path = "/response_format";
        break;
    case JsonRequestProtocol::Responses:
        format_path = "/text/format";
        break;
    case JsonRequestProtocol::AnthropicMessages:
        break;
    }
    if (format_path) {
        const nlohmann::ordered_json::json_pointer format_pointer(format_path.value());
        if (parsed.value.contains(format_pointer)) {
            const auto& format = parsed.value.at(format_pointer);
            if (format.is_object() && format.contains("type") && format["type"] == "json_schema") {
                check(format_path.value() + (protocol == JsonRequestProtocol::ChatCompletions
                                                 ? "/json_schema/schema"
                                                 : "/schema"));
            }
        }
    }
    if (parsed.value.is_object() && parsed.value.contains("tools") &&
        parsed.value["tools"].is_array()) {
        const auto& tools = parsed.value["tools"];
        for (std::size_t index = 0; index < tools.size(); ++index) {
            const auto& tool = tools[index];
            if (!tool.is_object()) { continue; }
            const auto prefix = "/tools/" + std::to_string(index);
            if (protocol == JsonRequestProtocol::ChatCompletions) {
                if (!tool.contains("function") || !tool["function"].is_object()) { continue; }
                const auto& function = tool["function"];
                const auto name      = function.contains("name") && function["name"].is_string()
                                           ? function["name"].get<std::string>()
                                           : std::string{};
                check(prefix + "/function/parameters", true, name);
            } else {
                const auto name = tool.contains("name") && tool["name"].is_string()
                                      ? tool["name"].get<std::string>()
                                      : std::string{};
                check(prefix + (protocol == JsonRequestProtocol::Responses ? "/parameters"
                                                                           : "/input_schema"),
                      true, name);
            }
        }
    }
    return parsed.value;
}

} // namespace ninfer::serve
