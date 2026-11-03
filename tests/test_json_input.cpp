#include "text/json_input.h"
#include "serve/json_input.h"
#include "serve/request.h"
#include "serve/anthropic_schema.h"
#include "serve/openai_schema.h"
#include "serve/responses_schema.h"
#include "product/prompt_input/prompt_input.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <unistd.h>

#include <iostream>
#include <utility>
#include <stdexcept>
#include <string>

namespace {
struct MessagesFile {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
                                 ("ninfer-json-source-" + std::to_string(::getpid()) + ".json");

    ~MessagesFile() {
        std::error_code error;
        std::filesystem::remove(path, error);
    }
};
} // namespace

int main() {
    try {
        for (const auto& [source, expected] :
             {std::pair{R"({"const":9007199254740993.0})", "/const"},
              std::pair{R"({"minimum":9007199254740993})", ""}, std::pair{R"({"const":0.1})", ""},
              std::pair{R"({"const":100e-2})", ""}, std::pair{R"({"const":-0.0})", ""},
              std::pair{R"({"const":1.00000000000000001})", "/const"},
              std::pair{R"({"const":1e-324})", "/const"},
              std::pair{R"({"maxLength":9007199254740993.0})", "/maxLength"},
              std::pair{R"({"enum":[9007199254740993.0]})", "/enum/0"},
              std::pair{R"({"description":1.00000000000000001})", ""},
              std::pair{R"({"const":1.00000000000000001,"const":1})", ""},
              std::pair{R"({"const":{"a":[1.00000000000000001]}})", "/const/a/0"},
              std::pair{R"({"properties":{"a/b~":{"minimum":1.00000000000000001}}})",
                        "/properties/a~1b~0/minimum"},
              std::pair{R"({"prefixItems":[{"enum":[1.00000000000000001]}]})",
                        "/prefixItems/0/enum/0"}}) {
            const auto parsed   = ninfer::text::parse_json_numbers(source);
            const auto location = ninfer::text::inexact_schema_number(parsed);
            if (location.value_or("") != expected) {
                throw std::runtime_error(std::string(source) + ": wrong precision disposition");
            }
        }
        const auto exact = ninfer::text::parse_json_numbers(R"({"const":9007199254740993})");
        if (!exact.value.at("const").is_number_integer() ||
            exact.value.at("const").get<std::uint64_t>() != UINT64_C(9007199254740993)) {
            throw std::runtime_error("exact integer source was rounded through binary64");
        }
        const auto request = ninfer::text::parse_json_numbers(
            R"({"temperature":1.00000000000000001,"schema":{"const":1}})");
        if (ninfer::text::inexact_schema_number(request, "/schema")) {
            throw std::runtime_error("ordinary request number was treated as a schema assertion");
        }
        using Protocol = ninfer::serve::JsonRequestProtocol;

        struct SchemaCase {
            Protocol protocol;
            const char* source;
            const char* pointer;
        };

        for (
            const auto& sample : {
                SchemaCase{
                    Protocol::ChatCompletions,
                    R"({"response_format":{"type":"json_schema","json_schema":{"schema":{"const":9007199254740993.0}}}})",
                    "/response_format/json_schema/schema/const"},
                SchemaCase{
                    Protocol::Responses,
                    R"({"text":{"format":{"type":"json_schema","schema":{"const":9007199254740993.0}}}})",
                    "/text/format/schema/const"},
                SchemaCase{
                    Protocol::ChatCompletions,
                    R"({"tools":[{"function":{"parameters":{"const":9007199254740993.0}}}]})",
                    "/tools/0/function/parameters/const"},
                SchemaCase{
                    Protocol::Responses,
                    R"({"tools":[{"type":"function","name":"f","function":{},"parameters":{"const":9007199254740993.0}}]})",
                    "/tools/0/parameters/const"},
                SchemaCase{
                    Protocol::AnthropicMessages,
                    R"({"tools":[{"name":"f","function":{},"input_schema":{"const":9007199254740993.0}}]})",
                    "/tools/0/input_schema/const"},
            }) {
            bool rejected = false;
            try {
                (void)ninfer::serve::parse_request_body(sample.source, sample.protocol);
            } catch (const ninfer::serve::ApiException& error) {
                rejected = error.error().status == 400 && error.error().param == sample.pointer;
            }
            if (!rejected) {
                throw std::runtime_error(std::string("HTTP schema lost source precision at ") +
                                         sample.pointer);
            }
        }
        // Other protocols' schema-shaped extension fields are not active assertions.
        for (const auto protocol : {ninfer::serve::JsonRequestProtocol::Responses,
                                    ninfer::serve::JsonRequestProtocol::AnthropicMessages}) {
            (void)ninfer::serve::parse_request_body(
                R"({"tools":[{"function":{"parameters":{"const":9007199254740993.0}}}]})",
                protocol);
        }
        (void)ninfer::serve::parse_request_body(
            R"({"response_format":{"type":"json_object","json_schema":{"schema":{"const":9007199254740993.0}}}})",
            ninfer::serve::JsonRequestProtocol::ChatCompletions);
        const auto unsupported_format = ninfer::serve::parse_request_body(
            R"({"model":"local","max_tokens":16,"messages":[{"role":"user","content":"hello"}],"output_config":{"format":{"type":"json_schema","schema":{"const":9007199254740993.0}}}})",
            Protocol::AnthropicMessages);
        bool unsupported_classification = false;
        try {
            (void)ninfer::serve::parse_messages_request(unsupported_format, {});
        } catch (const ninfer::serve::ApiException& error) {
            unsupported_classification = error.error().code == "output_config_option_not_supported";
        }
        if (!unsupported_classification) {
            throw std::runtime_error(
                "unsupported Anthropic output format changed error classification");
        }
        // Response grammars emit properties in schema order, and tool schemas are rendered into
        // the prompt as supplied. A key-sorting representation would make a JSON response begin
        // with whichever property sorts first, for example a list before the fields it describes.
        const std::string ordered_schema =
            R"({"type":"object","properties":{"title":{"type":"string"},"tags":{"type":"array"},"people":{"type":"array"}},"required":["title","tags","people"]})";
        const auto chat = ninfer::serve::parse_chat_completion_request(
            ninfer::serve::parse_request_body(
                R"({"model":"local","messages":[{"role":"user","content":"hello"}],"response_format":{"type":"json_schema","json_schema":{"name":"result","schema":)" +
                    ordered_schema + "}}}",
                Protocol::ChatCompletions),
            {});
        if (!chat.output_constraint || chat.output_constraint->source != ordered_schema) {
            throw std::runtime_error("Chat response schema lost property order");
        }
        const auto responses = ninfer::serve::parse_responses_request(
            ninfer::serve::parse_request_body(
                R"({"model":"local","input":"hello","text":{"format":{"type":"json_schema","name":"result","schema":)" +
                    ordered_schema + "}}}",
                Protocol::Responses),
            {});
        if (!responses.generation.output_constraint ||
            responses.generation.output_constraint->source != ordered_schema) {
            throw std::runtime_error("Responses schema lost property order");
        }
        const auto tools = ninfer::serve::parse_chat_completion_request(
            ninfer::serve::parse_request_body(
                R"({"model":"local","messages":[{"role":"user","content":"hello"}],"tools":[{"type":"function","function":{"name":"f","parameters":)" +
                    ordered_schema + "}}]}",
                Protocol::ChatCompletions),
            {});
        if (tools.tools.size() != 1 || tools.tools[0].parameters_json != ordered_schema) {
            throw std::runtime_error("Chat tool schema lost property order");
        }
        MessagesFile messages;
        {
            std::ofstream stream(messages.path);
            stream.exceptions(std::ios::failbit | std::ios::badbit);
            stream
                << R"({"messages":[{"role":"user","content":"hello"}],"tools":[{"type":"function","function":{"name":"f","parameters":)"
                << ordered_schema << "}}]}";
        }
        const auto file_prompt = ninfer::product::prompt_from_messages(messages.path, false, false);
        if (file_prompt.options.tool_jsons.size() != 1 ||
            file_prompt.options.tool_jsons[0].find(ordered_schema) == std::string::npos) {
            throw std::runtime_error("messages-file tool schema lost property order");
        }
        {
            std::ofstream stream(messages.path);
            stream.exceptions(std::ios::failbit | std::ios::badbit);
            stream
                << R"({"messages":[{"role":"user","content":"hello"}],"tools":[{"type":"function","function":{"name":"precise","parameters":{"const":9007199254740993.0}}}]})";
        }
        bool rejected_messages = false;
        try {
            (void)ninfer::product::prompt_from_messages(messages.path, false, false);
        } catch (const std::invalid_argument& error) {
            rejected_messages =
                std::string(error.what()).find("/tools/0/function/parameters/const") !=
                std::string::npos;
        }
        if (!rejected_messages)
            throw std::runtime_error("messages file lost schema source precision");
        std::cout << "PASS: decimal schema source values, paths, and property order\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
