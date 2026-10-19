#pragma once

#include <nlohmann/json.hpp>

#include <string_view>

namespace ninfer::serve {

enum class JsonRequestProtocol { ChatCompletions, Responses, AnthropicMessages };

// Returns an owning DOM after validating schema numbers before forwarding loses their spelling.
// Throws ApiException for an inexact assertion and a JSON exception for malformed input.
[[nodiscard]] nlohmann::json parse_request_body(std::string_view source,
                                                JsonRequestProtocol protocol);

} // namespace ninfer::serve
