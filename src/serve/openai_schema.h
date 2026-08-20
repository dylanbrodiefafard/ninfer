#pragma once

// OpenAI wire-format layer: parses request JSON into the internal GenerationRequest
// and serializes internal results back into OpenAI Chat Completions bodies/chunks.
// This layer knows nothing about the engine; it only speaks the OpenAI schema.

#include "serve/request.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <optional>
#include <string>

namespace ninfer::serve {

// ApiError, ApiException, RequestLimits, and CompletionUsage are the wire-format
// independent request/error types; they live in request.h and are shared by the
// OpenAI and Anthropic schema layers.

// Parse an already-decoded JSON body into a GenerationRequest. Throws ApiException
// on malformed or unsupported requests (n>1, tools, non-text response_format, ...).
GenerationRequest parse_chat_completion_request(const nlohmann::json& body,
                                                const RequestLimits& limits);

std::optional<bool> parse_openai_preserve_thinking(const nlohmann::json& body);
std::optional<bool> parse_openai_enable_thinking(const nlohmann::json& body);

[[nodiscard]] CompletionTimings make_completion_timings(int prompt_tokens, int completion_tokens,
                                                         double prefill_seconds,
                                                         double decode_seconds, int draft_n = 0,
                                                         int draft_n_accepted = 0,
                                                         double prefill_tail_tok_s = 0.0,
                                                         double prefill_tail_window_s = 0.0,
                                                         int prompt_reused = 0);

// Non-streaming chat completion response body (JSON string). When `reasoning` is
// non-empty it is attached as `message.reasoning_content` (the DeepSeek/vLLM-style
// convention consumed by Chatbox, Open WebUI, etc.), leaving `content` = answer.
// The usage object carries OpenAI-standard details sub-objects: `cached_tokens` and
// engine stats under the `ninfer` namespace in `prompt_tokens_details` (prefill /
// decode rates, KV-RAM tier stats), and reasoning / speculative-decoding token
// counts in `completion_tokens_details`. Each stat is stored exactly once.
std::string make_chat_completion_response(const std::string& id, const std::string& model,
                                          std::int64_t created, const std::string& content,
                                          const std::string& reasoning, const char* finish_reason,
                                          const CompletionUsage& usage,
                                          const CompletionTimings* timings = nullptr);
std::string make_chat_completion_tool_response(const std::string& id, const std::string& model,
                                               std::int64_t created, const std::string& content,
                                               const std::string& reasoning,
                                               const std::vector<ToolCall>& tool_calls,
                                               const CompletionUsage& usage,
                                               const CompletionTimings* timings = nullptr);

// Streaming SSE event strings ("data: {...}\n\n"). The first chunk carries the
// assistant role; reasoning chunks carry `reasoning_content` deltas (the <think>
// block), content chunks carry `content` deltas; the final chunk carries the
// finish_reason with an empty delta. Per the OpenAI stream_options contract, when
// usage reporting is enabled every content-bearing chunk carries `usage: null`
// and a single dedicated usage chunk (empty choices) is emitted before [DONE];
// pass include_usage accordingly.
std::string make_chat_chunk_role(const std::string& id, const std::string& model,
                                 std::int64_t created, bool include_usage);
std::string make_chat_chunk_reasoning(const std::string& id, const std::string& model,
                                      std::int64_t created, const std::string& delta_text,
                                      bool include_usage);
std::string make_chat_chunk_content(const std::string& id, const std::string& model,
                                    std::int64_t created, const std::string& delta_text,
                                    bool include_usage);
std::string make_chat_chunk_tool_calls(const std::string& id, const std::string& model,
                                       std::int64_t created,
                                       const std::vector<ToolCall>& tool_calls, bool include_usage);
std::string make_chat_chunk_final(const std::string& id, const std::string& model,
                                  std::int64_t created, const char* finish_reason,
                                  bool include_usage,
                                  const CompletionTimings* timings = nullptr,
                                  const CompletionUsage* usage = nullptr);
// Dedicated usage chunk: `choices: []` with the request's token usage. Emitted
// before [DONE] (and whenever include_usage is requested). Carries the usage object
// described above: OpenAI-standard `prompt_tokens_details` (`cached_tokens`, `ninfer`
// engine stats — prefill/decode rates, reuse source, and when host KV RAM is enabled
// the same live occupancy, copy times, and lifetime counters as the serve `[req] done`
// line) and OpenAI-standard `completion_tokens_details` (reasoning and speculative
// prediction token counts). Rates and millisecond fields are rounded to three
// decimal places.
std::string make_chat_chunk_usage(const std::string& id, const std::string& model,
                                  std::int64_t created, const CompletionUsage& usage,
                                  const CompletionTimings* timings = nullptr);
std::string sse_done();

// /v1/models payloads.
std::string make_models_list(const std::string& model_id, std::int64_t created);
std::string make_model_object(const std::string& model_id, std::int64_t created);

// Error object body.
std::string make_error_body(const ApiError& error);

// Identifiers / timestamps.
std::string new_chat_completion_id();
std::int64_t unix_time_now();

} // namespace ninfer::serve
