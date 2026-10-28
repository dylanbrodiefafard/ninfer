#include "serve/generation_service.h"
#include "serve/http_server.h"
#include "serve/metrics.h"
#include "serve/openai_schema.h"
#include "serve/anthropic_schema.h"
#include "serve/responses_schema.h"
#include "serve/translate.h"

#include <nlohmann/json.hpp>

#include <iostream>
#include <string>

namespace {

using Json = nlohmann::json;
using ninfer::serve::ServeOptions;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    int failures = 0;
    for (const bool available : {false, true}) {
        httplib::Response response;
        ninfer::serve::handle_health(available, response);
        failures += check(response.status == (available ? 200 : 503) &&
                              Json::parse(response.body) ==
                                  Json{{"status", available ? "ok" : "unavailable"}} &&
                              response.get_header_value("Content-Type") == "application/json",
                          "health did not reflect inference availability");
    }

    failures += check(ninfer::serve::is_unauthenticated_path("/health") &&
                          ninfer::serve::is_unauthenticated_path("/metrics") &&
                          ninfer::serve::is_unauthenticated_path("/metrics.json") &&
                          !ninfer::serve::is_unauthenticated_path("/v1/chat/completions"),
                      "metrics routes are authenticated");
    ServeOptions options;
    options.max_request_bytes = 1234;

    const auto conflict = ninfer::serve::request_error_to_api_error(
        ninfer::RequestError(ninfer::RequestErrorKind::CacheWriteConflict,
                             "capture_context_checkpoint=true requires cache_write=true because it "
                             "creates a retained context checkpoint."));
    failures += check(conflict.status == 400 && conflict.code == "cache_write_conflict" &&
                          conflict.param == "ninfer.capture_context_checkpoint",
                      "Engine conflict maps to HTTP 400");

    const auto tool_schema = ninfer::serve::request_error_to_api_error(ninfer::RequestError(
        ninfer::RequestErrorKind::InvalidToolSchema, "unsupported tool schema assertion: not"));
    failures +=
        check(tool_schema.status == 400 && tool_schema.type == "invalid_request_error" &&
                  tool_schema.code == "invalid_tool_schema" && tool_schema.param == "tools" &&
                  tool_schema.message == "unsupported tool schema assertion: not",
              "tool schema rejection lost its HTTP classification or diagnostic");

    {
        using namespace ninfer::serve;
        const Json schema{{"type", "object"}, {"properties", Json::object()}};
        const auto chat_tool = [&](const char* name) {
            return Json{{"type", "function"},
                        {"function", Json{{"name", name}, {"parameters", schema}}}};
        };
        const auto response_tool = [&](const char* name) {
            return Json{{"type", "function"}, {"name", name}, {"parameters", schema}};
        };
        const auto anthropic_tool = [&](const char* name) {
            return Json{{"name", name}, {"input_schema", schema}};
        };
        Json base{{"model", "test"},
                  {"messages", Json::array({Json{{"role", "user"}, {"content", "hi"}}})}};
        auto chat_body     = base;
        chat_body["tools"] = Json::array({chat_tool("first"), chat_tool("selected")});
        chat_body["tool_choice"] =
            Json{{"type", "function"}, {"function", Json{{"name", "selected"}}}};
        auto chat                    = parse_chat_completion_request(chat_body, RequestLimits{});
        auto anthropic_body          = base;
        anthropic_body["max_tokens"] = 16;
        anthropic_body["tools"] =
            Json::array({anthropic_tool("first"), anthropic_tool("selected")});
        anthropic_body["tool_choice"] = Json{{"type", "tool"}, {"name", "selected"}};
        auto anthropic                = parse_messages_request(anthropic_body, RequestLimits{});
        auto responses =
            parse_responses_request(
                Json{{"model", "test"},
                     {"input", "hi"},
                     {"tools", Json::array({response_tool("first"), response_tool("selected")})},
                     {"tool_choice", Json{{"type", "function"}, {"name", "selected"}}}},
                RequestLimits{})
                .generation;
        for (const auto& [request, expected] :
             {std::pair{chat, "/tools/1/function/parameters/$defs/range"},
              std::pair{anthropic, "/tools/1/input_schema/$defs/range"},
              std::pair{responses, "/tools/1/parameters/$defs/range"}}) {
            const auto error = request_error_to_api_error(
                ninfer::RequestError(ninfer::RequestErrorKind::InvalidToolSchema,
                                     "unsupported numeric range", {},
                                     ninfer::RequestErrorLocation{"/$defs/range", 0}),
                &request);
            failures += check(
                error.param == expected && error.message.find("selected") != std::string::npos,
                "named tool selection lost the original wire index, schema field, or name");
            failures += check(to_prompt_input(request, ResolvedRequestSemantics{}, {})
                                      .options.tool_jsons.size() == 1,
                              "named tool diagnostic fixture did not select one Engine tool");
        }
        chat.output_constraint =
            ninfer::OutputConstraint{ninfer::OutputConstraintKind::JsonSchema, "{}"};
        const auto response_error = ninfer::RequestError(
            ninfer::RequestErrorKind::InvalidOutputConstraint, "unsupported numeric range", {},
            ninfer::RequestErrorLocation{"/properties/value", std::nullopt});
        failures += check(request_error_to_api_error(response_error, &chat).param ==
                              "/response_format/json_schema/schema/properties/value",
                          "Chat schema source pointer changed");
        responses.output_constraint = chat.output_constraint;
        failures += check(request_error_to_api_error(response_error, &responses).param ==
                              "/text/format/schema/properties/value",
                          "Responses schema source pointer changed");
    }

    const ninfer::serve::ApiError media_budget = ninfer::serve::request_error_to_api_error(
        ninfer::RequestError(ninfer::RequestErrorKind::MediaBudgetExceeded,
                             "vision tokens exceed processor budget"));
    failures += check(media_budget.status == 400 && media_budget.code == "media_budget_exceeded",
                      "media resource rejection did not map to HTTP 400");
    ninfer::GenerationRecoveryStats recovery;
    recovery.attempts         = 2;
    recovery.cycle_exclusions = 5;
    const auto exhausted      = ninfer::serve::request_error_to_api_error(ninfer::RequestError(
        ninfer::RequestErrorKind::RecoveryExhausted, "bounded recovery exhausted", recovery));
    failures +=
        check(exhausted.status == 500 && exhausted.type == "server_error" &&
                  exhausted.code == "generation_recovery_exhausted" && exhausted.param.empty(),
              "recovery exhaustion lost its explicit request-local error");
    failures += check(exhausted.recovery.attempts == 2 && exhausted.recovery.cycle_exclusions == 5,
                      "Engine recovery totals did not reach the API error");
    const ninfer::serve::ApiError context_limit = ninfer::serve::request_error_to_api_error(
        ninfer::RequestError(ninfer::RequestErrorKind::ContextLengthExceeded,
                             "prepared prompt has 200 tokens, exceeding Engine max_context 128"));
    failures +=
        check(context_limit.status == 400 && context_limit.code == "context_length_exceeded" &&
                  context_limit.message.find("200 tokens") != std::string::npos &&
                  context_limit.message.find("128") != std::string::npos,
              "context rejection lost its HTTP classification or capacity details");

    auto capture_code = [](bool requested, bool reuse, ninfer::SpeculativeBackend spec) {
        try {
            ninfer::serve::reject_unavailable_context_checkpoint_capture(requested, reuse, spec);
            return std::string();
        } catch (const ninfer::serve::ApiException& error) { return error.error().code; }
    };
    auto capture_status = [](bool requested, bool reuse, ninfer::SpeculativeBackend spec) {
        try {
            ninfer::serve::reject_unavailable_context_checkpoint_capture(requested, reuse, spec);
            return 0;
        } catch (const ninfer::serve::ApiException& error) { return error.error().status; }
    };
    failures += check(
        ninfer::context_checkpoint_capture_available(true, ninfer::SpeculativeBackend::Mtp) &&
            ninfer::context_checkpoint_capture_available(true,
                                                         ninfer::SpeculativeBackend::DFlash) &&
            !ninfer::context_checkpoint_capture_available(true, ninfer::SpeculativeBackend::None) &&
            !ninfer::context_checkpoint_capture_available(false, ninfer::SpeculativeBackend::Mtp),
        "pin availability is not prefix-reuse plus a speculative backend");
    failures += check(capture_code(true, true, ninfer::SpeculativeBackend::None) ==
                              "context_checkpoint_unavailable" &&
                          capture_status(true, true, ninfer::SpeculativeBackend::None) == 400,
                      "no-spec capture did not return HTTP 400 context_checkpoint_unavailable");
    failures += check(capture_code(true, false, ninfer::SpeculativeBackend::Mtp) ==
                          "context_checkpoint_unavailable",
                      "--no-prefix-reuse capture did not return context_checkpoint_unavailable");
    failures += check(capture_code(false, true, ninfer::SpeculativeBackend::None).empty() &&
                          capture_code(true, true, ninfer::SpeculativeBackend::Mtp).empty() &&
                          capture_code(true, true, ninfer::SpeculativeBackend::DFlash).empty(),
                      "available or unrequested capture was rejected");

    httplib::Request messages_request;
    messages_request.path = "/v1/messages";
    httplib::Response messages_response;
    messages_response.status = 413;
    const auto messages_result =
        ninfer::serve::handle_unrendered_http_error(options, messages_request, messages_response);
    const Json messages_body = Json::parse(messages_response.body);
    failures += check(messages_result == httplib::Server::HandlerResponse::Handled &&
                          messages_body.at("type") == "error" &&
                          messages_body.at("error").at("type") == "invalid_request_error" &&
                          messages_body.at("error").at("message").get<std::string>().find(
                              "1234 bytes") != std::string::npos,
                      "empty Anthropic 413 did not become a payload-limit error");

    httplib::Request openai_request;
    openai_request.path = "/v1/responses";
    httplib::Response openai_response;
    openai_response.status = 413;
    const auto openai_result =
        ninfer::serve::handle_unrendered_http_error(options, openai_request, openai_response);
    const Json openai_body = Json::parse(openai_response.body);
    failures += check(openai_result == httplib::Server::HandlerResponse::Handled &&
                          openai_body.at("error").at("code") == "request_too_large" &&
                          openai_body.at("error").at("message").get<std::string>().find(
                              "1234 bytes") != std::string::npos,
                      "empty OpenAI 413 did not become a payload-limit error");

    httplib::Response authored_response;
    authored_response.status = 413;
    authored_response.set_content(R"({"error":{"code":"application_error"}})", "application/json");
    const std::string authored_body = authored_response.body;
    const auto authored_result =
        ninfer::serve::handle_unrendered_http_error(options, openai_request, authored_response);
    failures += check(authored_result == httplib::Server::HandlerResponse::Unhandled &&
                          authored_response.body == authored_body,
                      "application-authored 413 was overwritten by the payload-limit handler");

    httplib::Response other_response;
    other_response.status = 400;
    const auto other_result =
        ninfer::serve::handle_unrendered_http_error(options, openai_request, other_response);
    failures += check(other_result == httplib::Server::HandlerResponse::Unhandled &&
                          other_response.body.empty(),
                      "non-413 response was changed by the payload-limit handler");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
