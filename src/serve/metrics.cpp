#include "serve/metrics.h"

#include "product/speculative_options.h"
#include "serve/request_log.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <initializer_list>
#include <iterator>
#include <map>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::serve {
namespace {

using Json = nlohmann::json;

// Same value as ninfer::kPagedKVPageSize. Serve does not include the core KV header.
constexpr std::uint32_t kKvPageTokens = 64;
// DFlash draft_tokens upper bound; see product/speculative_options.h.
constexpr std::size_t kMaxDraftTokens = 15;

// ---------------------------------------------------------------------------
// Histogram bounds. A Histogram type carries its bounds, so observation and
// rendering cannot disagree.

struct Bound {
    double value;
    const char* text;
};

// Request, phase, and copy wall clocks: sub-millisecond CPU phases up to
// long reasoning generations.
constexpr Bound kLatencyBounds[] = {
    {0.001, "0.001"}, {0.0025, "0.0025"}, {0.005, "0.005"}, {0.01, "0.01"}, {0.025, "0.025"},
    {0.05, "0.05"},   {0.1, "0.1"},       {0.25, "0.25"},   {0.5, "0.5"},   {1, "1"},
    {2.5, "2.5"},     {5, "5"},           {10, "10"},       {30, "30"},     {60, "60"},
    {120, "120"},     {300, "300"},       {600, "600"},     {1200, "1200"},
};
// Decode seconds per output token: 500 tok/s down to 4 tok/s.
constexpr Bound kInterTokenBounds[] = {
    {0.002, "0.002"}, {0.003, "0.003"},   {0.004, "0.004"}, {0.005, "0.005"}, {0.006, "0.006"},
    {0.008, "0.008"}, {0.01, "0.01"},     {0.0125, "0.0125"}, {0.015, "0.015"}, {0.02, "0.02"},
    {0.025, "0.025"}, {0.03, "0.03"},     {0.04, "0.04"},   {0.05, "0.05"},   {0.075, "0.075"},
    {0.1, "0.1"},     {0.25, "0.25"},
};
constexpr Bound kTokenBounds[] = {
    {32, "32"},       {64, "64"},       {128, "128"},     {256, "256"},       {512, "512"},
    {1024, "1024"},   {2048, "2048"},   {4096, "4096"},   {8192, "8192"},     {16384, "16384"},
    {32768, "32768"}, {65536, "65536"}, {131072, "131072"}, {262144, "262144"},
};
constexpr Bound kDecodeRateBounds[] = {
    {5, "5"},     {10, "10"},   {25, "25"},   {50, "50"},   {75, "75"},   {100, "100"}, {150, "150"},
    {200, "200"}, {250, "250"}, {300, "300"}, {400, "400"}, {500, "500"}, {750, "750"}, {1000, "1000"},
};
// Prefill runs at thousands of tokens per second.
constexpr Bound kPrefillRateBounds[] = {
    {250, "250"},     {500, "500"},     {1000, "1000"},   {2000, "2000"},   {3000, "3000"},
    {4000, "4000"},   {5000, "5000"},   {6000, "6000"},   {8000, "8000"},   {10000, "10000"},
    {12500, "12500"}, {15000, "15000"}, {20000, "20000"}, {30000, "30000"},
};
constexpr Bound kCountBounds[] = {
    {1, "1"}, {2, "2"}, {3, "3"}, {4, "4"}, {5, "5"}, {6, "6"}, {8, "8"}, {10, "10"}, {12, "12"}, {15, "15"},
};

template <const auto& Bounds>
constexpr bool strictly_increasing() {
    for (std::size_t i = 1; i < std::size(Bounds); ++i) {
        if (!(Bounds[i - 1].value < Bounds[i].value)) { return false; }
    }
    return true;
}

template <const auto& Bounds>
struct Histogram {
    static_assert(strictly_increasing<Bounds>());
    static constexpr std::span<const Bound> bounds{Bounds};

    // Per-bucket counts; the last entry is the +Inf overflow.
    std::array<std::uint64_t, std::size(Bounds) + 1> buckets{};
    double sum          = 0;
    std::uint64_t count = 0;

    void observe(double value) noexcept {
        if (!std::isfinite(value)) { return; }
        const auto it = std::lower_bound(bounds.begin(), bounds.end(), value,
                                         [](const Bound& bound, double v) { return bound.value < v; });
        ++buckets[static_cast<std::size_t>(it - bounds.begin())];
        sum += value;
        ++count;
    }
};

using LatencyHistogram     = Histogram<kLatencyBounds>;
using InterTokenHistogram  = Histogram<kInterTokenBounds>;
using TokenHistogram       = Histogram<kTokenBounds>;
using DecodeRateHistogram  = Histogram<kDecodeRateBounds>;
using PrefillRateHistogram = Histogram<kPrefillRateBounds>;
using CountHistogram       = Histogram<kCountBounds>;

// ---------------------------------------------------------------------------
// Closed label vocabularies. Array order is the storage index.

constexpr const char* kProtocols[] = {"openai_chat", "openai_responses", "anthropic_messages"};
constexpr const char* kResults[]   = {"cancelled", "error", "rejected", "success"};
constexpr const char* kBools[]     = {"false", "true"};
constexpr std::size_t kProtocolCount = std::size(kProtocols);
constexpr std::size_t kResultCount   = std::size(kResults);
static_assert(static_cast<std::size_t>(MetricsProtocol::AnthropicMessages) + 1 == kProtocolCount);
static_assert(static_cast<std::size_t>(GenerationResult::Success) + 1 == kResultCount);

enum Phase : std::size_t {
    kPhasePrepareCpu,
    kPhaseMediaWait,
    kPhaseMediaFetch,
    kPhaseQueue,
    kPhaseCopyHold,
    kPhaseVision,
    kPhasePrefill,
    kPhaseDecode,
    kPhaseRecovery,
    kPhaseHttpTail,
    kPhaseCount,
};
constexpr const char* kPhases[kPhaseCount] = {"prepare_cpu", "media_wait", "media_fetch", "queue",
                                              "copy_hold",   "vision",     "prefill",     "decode",
                                              "recovery",    "http_tail"};

constexpr const char* kFinishReasons[] = {"none",        "output_limit", "context_capacity",
                                          "stop_token",  "stop_string",  "cancelled",
                                          "tool_calls",  "unknown"};
constexpr const char* kPrefixPaths[] = {
    "full_reset",
    "append_frontier",
    "restore_turn_checkpoint",
    "restore_response_checkpoint",
    "restore_context_checkpoint",
    "restore_turn_rollback",
    "unknown",
};
constexpr const char* kPrefixSources[] = {"none", "vram_resident", "host_ram", "host_disk", "unknown"};
constexpr const char* kRecoveryKinds[] = {
    "cycle_exclusion", "retry_triggered", "retry_started",
    "retry_prefill_complete", "finished", "exhausted",
};
// Direct causes carried verbatim by non-exhausted events.
constexpr const char* kDirectRecoveryCauses[] = {
    "reasoning_cycle", "repeated_reasoning", "duplicate_tool_call", "cancelled",
    "tool_calls",      "output_limit",       "context_capacity",    "stop",
};
constexpr const char* kRecoveryCauses[] = {
    "reasoning_cycle", "repeated_reasoning", "duplicate_tool_call", "cancelled",
    "tool_calls",      "output_limit",       "context_capacity",    "stop",
    "retry_budget",    "output_budget",      "prologue",            "lane_rebuild",
    "other",
};
constexpr const char* kApiErrorCodes[] = {
    "invalid_api_key",
    "request_too_large",
    "server_overloaded",
    "request_queue_timeout",
    "service_unavailable",
    "generation_recovery_exhausted",
    "client_disconnected",
    "context_length_exceeded",
    "context_checkpoint_unavailable",
    "vision_disabled",
    "media_budget_exceeded",
    "media_fetch_failed",
    "media_fetch_timeout",
    "invalid_media",
    "invalid_tool_schema",
    "modality_not_supported",
    "reasoning_effort_not_supported",
    "conflicting_template_option",
    "response_not_found",
    "response_store_capacity_exceeded",
    "compaction_not_supported",
    "background_not_supported",
    "include_not_supported",
    "invalid_pagination",
    "n_not_supported",
    "tools_not_supported",
    "tool_type_not_supported",
    "tool_choice_not_supported",
    "strict_tools_not_supported",
    "response_format_not_supported",
    "structured_outputs_not_supported",
    "file_inputs_not_supported",
    "audio_inputs_not_supported",
    "image_detail_not_supported",
    "item_type_not_supported",
    "phase_not_supported",
    "encrypted_reasoning_not_supported",
    "reasoning_summary_not_supported",
    "reasoning_option_not_supported",
    "parameter_not_supported",
    "unknown_parameter",
    "parallel_tool_calls_not_supported",
    "logprobs_not_supported",
    "truncation_not_supported",
    "service_tier_not_supported",
    "stream_option_not_supported",
    "text_option_not_supported",
    "chat_template_option_not_supported",
    "ninfer_option_not_supported",
    "output_config_option_not_supported",
    "unsupported_role",
    "invalid_message_order",
    "invalid_value",
    "unnamed",
    "other",
};
constexpr const char* kTiers[]   = {"disk", "ram"};
constexpr const char* kCopyOps[] = {"h2d", "load", "save"};
constexpr const char* kMethods[] = {"DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT", "other"};

constexpr std::size_t kFinishCount   = std::size(kFinishReasons);
constexpr std::size_t kPathCount     = std::size(kPrefixPaths);
constexpr std::size_t kSourceCount   = std::size(kPrefixSources);
constexpr std::size_t kKindCount     = std::size(kRecoveryKinds);
constexpr std::size_t kCauseCount    = std::size(kRecoveryCauses);
constexpr std::size_t kApiCodeCount  = std::size(kApiErrorCodes);
constexpr std::size_t kTierCount     = std::size(kTiers);
constexpr std::size_t kCopyOpCount   = std::size(kCopyOps);
constexpr std::size_t kMethodCount   = std::size(kMethods);

constexpr HttpRouteClass kRoutes[] = {
    {"health", "/health"},
    {"metrics", "/metrics"},
    {"metrics", "/metrics.json"},
    {"models", "/v1/models"},
    {"models", "/v1/models/{id}"},
    {"openai_chat", "/v1/chat/completions"},
    {"openai_responses", "/v1/responses"},
    {"openai_responses", "/v1/responses/input_tokens"},
    {"openai_responses", "/v1/responses/compact"},
    {"openai_responses", "/v1/responses/{id}"},
    {"openai_responses", "/v1/responses/{id}/input_items"},
    {"openai_responses", "/v1/responses/{id}/cancel"},
    {"anthropic_messages", "/v1/messages"},
    {"anthropic_messages", "/v1/messages/count_tokens"},
    {"other", "other"},
};
constexpr std::size_t kRouteCount = std::size(kRoutes);

// Index of `value` in `names`, or `fallback` when absent.
template <std::size_t N>
[[nodiscard]] std::size_t index_of(const char* const (&names)[N], std::string_view value,
                                   std::size_t fallback) noexcept {
    for (std::size_t i = 0; i < N; ++i) {
        if (value == names[i]) { return i; }
    }
    return fallback;
}

[[nodiscard]] std::size_t route_index(std::string_view path) noexcept {
    for (std::size_t i = 0; i + 1 < kRouteCount; ++i) {
        if (path == kRoutes[i].route) { return i; }
    }
    constexpr std::string_view kModelPrefix = "/v1/models/";
    if (path.starts_with(kModelPrefix) && path.size() > kModelPrefix.size() &&
        path.find('/', kModelPrefix.size()) == std::string_view::npos) {
        return 4;
    }
    constexpr std::string_view kResponsePrefix = "/v1/responses/";
    if (path.starts_with(kResponsePrefix) && path.size() > kResponsePrefix.size()) {
        const std::string_view rest = path.substr(kResponsePrefix.size());
        const auto slash            = rest.find('/');
        if (slash == std::string_view::npos) { return 9; }
        if (slash > 0) {
            const std::string_view tail = rest.substr(slash);
            if (tail == "/input_items") { return 10; }
            if (tail == "/cancel") { return 11; }
        }
    }
    return kRouteCount - 1;
}

[[nodiscard]] std::size_t recovery_kind_index(ninfer::RecoveryEventKind kind) noexcept {
    switch (kind) {
    case ninfer::RecoveryEventKind::CycleExclusion:
        return 0;
    case ninfer::RecoveryEventKind::RetryTriggered:
        return 1;
    case ninfer::RecoveryEventKind::RetryStarted:
        return 2;
    case ninfer::RecoveryEventKind::RetryPrefillComplete:
        return 3;
    case ninfer::RecoveryEventKind::Finished:
        return 4;
    case ninfer::RecoveryEventKind::Exhausted:
        return 5;
    }
    return 5;
}

[[nodiscard]] const char* yn(bool value) { return value ? "true" : "false"; }

// ---------------------------------------------------------------------------
// Stored state.

struct Startup {
    double max_context                = 0;
    double prefill_chunk              = 0;
    double pending_timeout_seconds    = 0;
    double default_max_tokens         = 0;
    double keep_frac                  = 1;
    double xattn_tau                  = 1;
    double xattn_min_len              = static_cast<double>(kDefaultXattnMinLen);
    double dflash_verify_width        = 0;
    double configured_draft_tokens    = 0;
    double load_seconds               = 0;
    double cuda_graph_allowance_bytes = 0;
    double cuda_graph_observed_bytes  = 0;
    double arena_weights              = 0;
    double arena_sequence             = 0;
    double arena_workspace            = 0;
    double arena_request_transient    = 0;
    double start_time_unix_s          = 0;
    double max_concurrency            = 0;
    double max_pending_requests       = 0;
    double response_max_records       = 0;
    double response_max_bytes         = 0;
    // Sorted by key.
    std::vector<std::pair<const char*, std::string>> engine_labels;
    std::vector<std::pair<const char*, std::string>> build_labels;
};

struct Series {
    // Key: route << 24 | method << 16 | status (0 renders as "other").
    std::map<std::uint32_t, std::uint64_t> http_requests;
    std::array<LatencyHistogram, kRouteCount> http_duration{};
    std::array<std::uint64_t, kApiCodeCount> api_errors{};

    // [protocol][stream][result][thinking][tools]
    std::uint64_t generation_requests[kProtocolCount][2][kResultCount][2][2]{};
    std::array<LatencyHistogram, kProtocolCount> ttft{};
    std::array<LatencyHistogram, kProtocolCount> e2e{};
    std::array<InterTokenHistogram, kProtocolCount> inter_token{};
    std::array<LatencyHistogram, kPhaseCount> phase{};
    LatencyHistogram kv_copy[kTierCount][kCopyOpCount]{};
    TokenHistogram prompt_tokens;
    TokenHistogram completion_tokens;
    TokenHistogram reasoning_tokens;
    TokenHistogram computed_prefill_tokens;
    DecodeRateHistogram output_tokens_per_second;
    PrefillRateHistogram prefill_tokens_per_second;
    std::array<std::uint64_t, kFinishCount> finish_reason{};
    std::uint64_t tool_calls          = 0;
    std::uint64_t ignored_tool_markup = 0;
    std::uint64_t media_requests      = 0;
    std::array<std::uint64_t, kProtocolCount> token_count_requests{};

    std::uint64_t prefix_reuse[kPathCount][kSourceCount]{};
    std::array<std::uint64_t, kSourceCount> prefix_hit_tokens{};
    std::uint64_t prefix_query_tokens          = 0;
    std::uint64_t checkpoint_restored_tokens   = 0;
    std::uint64_t checkpoint_captured_tokens   = 0;
    std::uint64_t checkpoint_capture_requests  = 0;

    std::uint64_t speculative_rounds          = 0;
    std::uint64_t speculative_draft_tokens    = 0;
    std::uint64_t speculative_accepted_tokens = 0;
    std::uint64_t speculative_fallback_steps  = 0;
    // Index = draft position.
    std::array<std::uint64_t, kMaxDraftTokens> speculative_accepted_by_position{};
    // Index = k - 1.
    std::array<std::uint64_t, kMaxDraftTokens> speculative_rounds_by_k{};
    CountHistogram speculative_live_k;

    std::uint64_t recovery_events[kKindCount][kCauseCount]{};
    std::uint64_t recovery_cycle_exclusions           = 0;
    std::uint64_t recovery_discarded_reasoning_tokens = 0;
    std::uint64_t recovery_discarded_tool_calls       = 0;
    CountHistogram recovery_attempts;
};

void observe_terminal_recovery(Series& series, const ninfer::GenerationRecoveryStats& stats) {
    series.recovery_cycle_exclusions += stats.cycle_exclusions;
    series.recovery_discarded_reasoning_tokens += stats.discarded_reasoning_tokens;
    series.recovery_discarded_tool_calls += stats.discarded_tool_calls;
    if (stats.attempts > 0) { series.recovery_attempts.observe(static_cast<double>(stats.attempts)); }
}

void observe_success(Series& series, std::size_t protocol, const GenerationOutcome& outcome) {
    const GenerationMetrics& metrics = outcome.metrics;
    series.ttft[protocol].observe(metrics.ttft_seconds);
    series.e2e[protocol].observe(metrics.total_seconds);
    const int decode_tokens =
        decode_eval_tokens(outcome.completion_tokens, metrics.recovery.prefill_samples);
    if (metrics.decode_seconds > 0.0 && decode_tokens > 0) {
        series.inter_token[protocol].observe(metrics.decode_seconds / static_cast<double>(decode_tokens));
        series.output_tokens_per_second.observe(static_cast<double>(decode_tokens) / metrics.decode_seconds);
    }
    const double phases[kPhaseCount] = {
        metrics.prepare_cpu_seconds,
        metrics.media_wait_seconds,
        metrics.media_fetch_seconds,
        metrics.queued_seconds,
        metrics.copy_hold_seconds,
        metrics.vision_seconds,
        metrics.prefill_seconds,
        metrics.decode_seconds,
        metrics.recovery.prepare_seconds + metrics.recovery.prefill_seconds,
        metrics.http_tail_seconds,
    };
    for (std::size_t i = 0; i < kPhaseCount; ++i) {
        if (phases[i] > 0.0) { series.phase[i].observe(phases[i]); }
    }
    const double copies[kTierCount][kCopyOpCount] = {
        {metrics.kv_disk_h2d_seconds, metrics.kv_disk_load_seconds, metrics.kv_disk_save_seconds},
        {0.0, metrics.kv_ram_load_seconds, metrics.kv_ram_save_seconds},
    };
    for (std::size_t tier = 0; tier < kTierCount; ++tier) {
        for (std::size_t op = 0; op < kCopyOpCount; ++op) {
            if (copies[tier][op] > 0.0) { series.kv_copy[tier][op].observe(copies[tier][op]); }
        }
    }
    if (outcome.prompt_tokens >= 0) {
        series.prompt_tokens.observe(static_cast<double>(outcome.prompt_tokens));
    }
    if (outcome.completion_tokens >= 0) {
        series.completion_tokens.observe(static_cast<double>(outcome.completion_tokens));
    }
    if (outcome.reasoning_tokens > 0) {
        series.reasoning_tokens.observe(static_cast<double>(outcome.reasoning_tokens));
    }
    const int computed =
        prefill_eval_tokens(outcome.prompt_tokens, static_cast<int>(metrics.prefix_cache_hit_tokens));
    if (computed >= 0) { series.computed_prefill_tokens.observe(static_cast<double>(computed)); }
    if (metrics.prefill_tail_tok_s > 0.0) {
        series.prefill_tokens_per_second.observe(metrics.prefill_tail_tok_s);
    } else if (metrics.prefill_seconds > 0.0 && computed > 0) {
        series.prefill_tokens_per_second.observe(static_cast<double>(computed) / metrics.prefill_seconds);
    }

    const char* reason = outcome.tool_calls.empty() ? finish_reason_name(outcome.finish_reason)
                                                     : "tool_calls";
    ++series.finish_reason[index_of(kFinishReasons, reason, kFinishCount - 1)];
    series.tool_calls += outcome.tool_calls.size();
    series.ignored_tool_markup += outcome.ignored_qwen_tool_call_names.size();

    const std::size_t path =
        index_of(kPrefixPaths, prefix_reuse_path_name(metrics.prefix_reuse_path), kPathCount - 1);
    const std::size_t source = index_of(
        kPrefixSources, prefix_reuse_source_name(metrics.prefix_reuse_source), kSourceCount - 1);
    ++series.prefix_reuse[path][source];
    series.prefix_hit_tokens[source] += metrics.prefix_cache_hit_tokens;
    series.prefix_query_tokens += static_cast<std::uint64_t>(std::max(outcome.prompt_tokens, 0));
    series.checkpoint_restored_tokens += metrics.restored_context_checkpoint_tokens;
    series.checkpoint_captured_tokens += metrics.captured_context_checkpoint_tokens;

    if (metrics.speculative_backend == SpeculativeBackend::None) { return; }
    series.speculative_rounds += metrics.speculative_rounds;
    series.speculative_draft_tokens += metrics.speculative_draft_tokens;
    series.speculative_accepted_tokens += metrics.speculative_accepted_tokens;
    series.speculative_fallback_steps += metrics.speculative_fallback_steps;
    const std::size_t positions =
        std::min(metrics.speculative_accepted_per_position.size(), kMaxDraftTokens);
    for (std::size_t position = 0; position < positions; ++position) {
        series.speculative_accepted_by_position[position] +=
            metrics.speculative_accepted_per_position[position];
    }
    // rounds_per_draft is indexed by k; index 0 is unused.
    const std::size_t widths =
        std::min(metrics.speculative_rounds_per_draft.size(), kMaxDraftTokens + 1);
    for (std::size_t k = 1; k < widths; ++k) {
        series.speculative_rounds_by_k[k - 1] += metrics.speculative_rounds_per_draft[k];
    }
    if (metrics.speculative_live_draft_tokens > 0) {
        series.speculative_live_k.observe(static_cast<double>(metrics.speculative_live_draft_tokens));
    }
}

// ---------------------------------------------------------------------------
// Prometheus text.

void append_number(std::string& out, double value) {
    if (!std::isfinite(value)) {
        out.append(std::isnan(value) ? "NaN" : value > 0 ? "+Inf" : "-Inf");
        return;
    }
    char buf[64];
    const auto result = std::to_chars(buf, buf + sizeof(buf), value);
    out.append(buf, result.ptr);
}

void append_number(std::string& out, std::uint64_t value) {
    char buf[24];
    const auto result = std::to_chars(buf, buf + sizeof(buf), value);
    out.append(buf, result.ptr);
}

void append_escaped(std::string& out, std::string_view value) {
    for (const char ch : value) {
        if (ch == '\\' || ch == '"') {
            out.push_back('\\');
            out.push_back(ch);
        } else if (ch == '\n') {
            out.append("\\n");
        } else {
            out.push_back(ch);
        }
    }
}

using LabelList = std::initializer_list<std::pair<std::string_view, std::string_view>>;

// `a="x",b="y"` without braces. Closed values need no escaping.
[[nodiscard]] std::string label_text(LabelList labels) {
    std::string out;
    for (const auto& [key, value] : labels) {
        if (!out.empty()) { out.push_back(','); }
        out.append(key);
        out.append("=\"");
        out.append(value);
        out.push_back('"');
    }
    return out;
}

class TextWriter {
public:
    TextWriter() { out_.reserve(96 * 1024); }

    void family(std::string_view name, std::string_view help, std::string_view type) {
        name_ = name;
        out_.append("# HELP ").append(name).push_back(' ');
        out_.append(help).push_back('\n');
        out_.append("# TYPE ").append(name).push_back(' ');
        out_.append(type).push_back('\n');
    }

    template <typename Value>
    void sample(std::string_view labels, Value value) {
        line("", labels, value);
    }

    template <typename Value>
    void gauge(std::string_view name, std::string_view help, Value value) {
        family(name, help, "gauge");
        sample("", value);
    }

    template <typename Value>
    void counter(std::string_view name, std::string_view help, Value value) {
        family(name, help, "counter");
        sample("", value);
    }

    // Emits `le` first; every histogram label name sorts after it.
    template <const auto& Bounds>
    void histogram(std::string_view labels, const Histogram<Bounds>& histogram) {
        std::uint64_t cumulative = 0;
        for (std::size_t i = 0; i <= std::size(Bounds); ++i) {
            cumulative += histogram.buckets[i];
            out_.append(name_).append("_bucket{le=\"");
            out_.append(i < std::size(Bounds) ? Bounds[i].text : "+Inf").push_back('"');
            if (!labels.empty()) { out_.append(",").append(labels); }
            out_.append("} ");
            append_number(out_, cumulative);
            out_.push_back('\n');
        }
        line("_sum", labels, histogram.sum);
        line("_count", labels, histogram.count);
    }

    template <const auto& Bounds>
    void histogram(std::string_view name, std::string_view help, const Histogram<Bounds>& histogram) {
        family(name, help, "histogram");
        this->histogram("", histogram);
    }

    [[nodiscard]] std::string take() { return std::move(out_); }

private:
    template <typename Value>
    void line(std::string_view suffix, std::string_view labels, Value value) {
        out_.append(name_).append(suffix);
        if (!labels.empty()) { out_.append("{").append(labels).push_back('}'); }
        out_.push_back(' ');
        append_number(out_, value);
        out_.push_back('\n');
    }

    std::string out_;
    std::string_view name_;
};

[[nodiscard]] std::string info_labels(const std::vector<std::pair<const char*, std::string>>& labels) {
    std::string out;
    for (const auto& [key, value] : labels) {
        if (!out.empty()) { out.push_back(','); }
        out.append(key).append("=\"");
        append_escaped(out, value);
        out.push_back('"');
    }
    return out;
}

} // namespace

struct MetricsSnapshotData {
    std::shared_ptr<const Startup> startup;
    Series series;
    ScrapeInputs inputs;
};

struct ServeMetricsState {
    mutable std::mutex mutex;
    std::shared_ptr<const Startup> startup = std::make_shared<const Startup>();
    Series series;
};

namespace {

std::string render_text(const MetricsSnapshotData& data) {
    const Startup& startup            = *data.startup;
    const Series& series              = data.series;
    const ninfer::RuntimeStats& stats = data.inputs.stats;
    TextWriter w;

    w.family("ninfer_engine_info", "Resident engine identity.", "gauge");
    w.sample(info_labels(startup.engine_labels), std::uint64_t{1});
    w.family("ninfer_build_info", "CUDA build identity.", "gauge");
    w.sample(info_labels(startup.build_labels), std::uint64_t{1});
    w.gauge("ninfer_max_context_tokens", "Configured max context tokens.", startup.max_context);
    w.gauge("ninfer_prefill_chunk_tokens", "Configured prefill chunk tokens.", startup.prefill_chunk);
    w.gauge("ninfer_pending_timeout_seconds", "Pending FIFO timeout.", startup.pending_timeout_seconds);
    w.gauge("ninfer_default_max_tokens", "Default max output tokens.", startup.default_max_tokens);
    w.gauge("ninfer_keep_frac", "Sparse attention keep fraction.", startup.keep_frac);
    w.gauge("ninfer_xattn_tau", "XAttention mass threshold.", startup.xattn_tau);
    w.gauge("ninfer_xattn_min_len", "Cached length at or below which XAttention stays dense.",
            startup.xattn_min_len);
    w.gauge("ninfer_dflash_verify_width", "DFlash verify width. 0 is automatic.",
            startup.dflash_verify_width);
    w.gauge("ninfer_speculative_configured_draft_tokens", "Configured draft tokens.",
            startup.configured_draft_tokens);
    w.gauge("ninfer_engine_load_seconds", "Artifact load seconds.", startup.load_seconds);
    w.gauge("ninfer_cuda_graph_allowance_bytes", "CUDA graph allowance at startup.",
            startup.cuda_graph_allowance_bytes);
    w.gauge("ninfer_cuda_graph_observed_bytes", "CUDA graph bytes observed at startup.",
            startup.cuda_graph_observed_bytes);
    w.family("ninfer_arena_capacity_bytes", "Arena capacity at startup.", "gauge");
    w.sample(R"(arena="request_transient")", startup.arena_request_transient);
    w.sample(R"(arena="sequence")", startup.arena_sequence);
    w.sample(R"(arena="weights")", startup.arena_weights);
    w.sample(R"(arena="workspace")", startup.arena_workspace);
    w.gauge("ninfer_server_start_time_seconds", "Server attach time as unix seconds.",
            startup.start_time_unix_s);
    w.gauge("ninfer_scheduler_max_concurrency", "Configured resident requests.", startup.max_concurrency);
    w.gauge("ninfer_scheduler_max_pending_requests", "Configured pending FIFO depth.",
            startup.max_pending_requests);

    w.family("ninfer_http_requests_total", "HTTP requests by route.", "counter");
    for (const auto& [key, value] : series.http_requests) {
        const HttpRouteClass& route = kRoutes[key >> 24];
        const std::uint32_t status  = key & 0xFFFFU;
        const std::string status_text = status == 0 ? std::string("other") : std::to_string(status);
        w.sample(label_text({{"method", kMethods[(key >> 16) & 0xFFU]},
                             {"protocol", route.protocol},
                             {"route", route.route},
                             {"status", status_text}}),
                 value);
    }
    // Like the request counter, only routes that have served a request are exposed.
    w.family("ninfer_http_request_duration_seconds", "HTTP handler duration.", "histogram");
    for (std::size_t i = 0; i < kRouteCount; ++i) {
        if (series.http_duration[i].count == 0) { continue; }
        w.histogram(label_text({{"protocol", kRoutes[i].protocol}, {"route", kRoutes[i].route}}),
                    series.http_duration[i]);
    }
    w.gauge("ninfer_http_in_flight_requests", "Requests occupying a serve slot.",
            static_cast<std::uint64_t>(data.inputs.http_in_flight));
    w.family("ninfer_api_errors_total", "API errors by closed code.", "counter");
    for (std::size_t i = 0; i < kApiCodeCount; ++i) {
        w.sample(label_text({{"code", kApiErrorCodes[i]}}), series.api_errors[i]);
    }

    w.family("ninfer_generation_requests_total", "HTTP generation attempts.", "counter");
    for (std::size_t protocol = 0; protocol < kProtocolCount; ++protocol) {
        for (std::size_t stream = 0; stream < 2; ++stream) {
            for (std::size_t result = 0; result < kResultCount; ++result) {
                for (std::size_t thinking = 0; thinking < 2; ++thinking) {
                    for (std::size_t tools = 0; tools < 2; ++tools) {
                        w.sample(label_text({{"protocol", kProtocols[protocol]},
                                             {"result", kResults[result]},
                                             {"stream", kBools[stream]},
                                             {"thinking", kBools[thinking]},
                                             {"tools", kBools[tools]}}),
                                 series.generation_requests[protocol][stream][result][thinking][tools]);
                    }
                }
            }
        }
    }
    const auto per_protocol = [&](const char* name, const char* help, const auto& histograms) {
        w.family(name, help, "histogram");
        for (std::size_t i = 0; i < kProtocolCount; ++i) {
            w.histogram(label_text({{"protocol", kProtocols[i]}}), histograms[i]);
        }
    };
    per_protocol("ninfer_generation_ttft_seconds", "Time to first token.", series.ttft);
    per_protocol("ninfer_generation_e2e_seconds", "Engine end to end seconds.", series.e2e);
    per_protocol("ninfer_generation_inter_token_latency_seconds", "Decode seconds per output token.",
                 series.inter_token);
    w.family("ninfer_generation_phase_seconds", "Closed generation phase seconds.", "histogram");
    for (std::size_t i = 0; i < kPhaseCount; ++i) {
        w.histogram(label_text({{"phase", kPhases[i]}}), series.phase[i]);
    }
    w.family("ninfer_generation_kv_copy_seconds", "Per-request KV copy seconds.", "histogram");
    for (std::size_t tier = 0; tier < kTierCount; ++tier) {
        for (std::size_t op = 0; op < kCopyOpCount; ++op) {
            w.histogram(label_text({{"op", kCopyOps[op]}, {"tier", kTiers[tier]}}), series.kv_copy[tier][op]);
        }
    }
    w.histogram("ninfer_generation_prompt_tokens", "Prompt tokens.", series.prompt_tokens);
    w.histogram("ninfer_generation_completion_tokens", "Completion tokens.", series.completion_tokens);
    w.histogram("ninfer_generation_reasoning_tokens", "Reasoning tokens.", series.reasoning_tokens);
    w.histogram("ninfer_generation_computed_prefill_tokens", "Prompt tokens excluding prefix hits.",
                series.computed_prefill_tokens);
    w.histogram("ninfer_generation_output_tokens_per_second", "Decode output tokens per second.",
                series.output_tokens_per_second);
    w.histogram("ninfer_generation_prefill_tokens_per_second", "Prefill tokens per second.",
                series.prefill_tokens_per_second);
    w.family("ninfer_generation_finish_reason_total", "Successful finish reasons.", "counter");
    for (std::size_t i = 0; i < kFinishCount; ++i) {
        w.sample(label_text({{"reason", kFinishReasons[i]}}), series.finish_reason[i]);
    }
    w.counter("ninfer_generation_tool_calls_total", "Returned tool calls.", series.tool_calls);
    w.counter("ninfer_generation_ignored_tool_markup_total", "Ignored undeclared tool markup.",
              series.ignored_tool_markup);
    w.counter("ninfer_generation_media_requests_total", "Generation attempts whose prompt had media.",
              series.media_requests);
    w.family("ninfer_token_count_requests_total", "Token-count requests.", "counter");
    for (std::size_t i = 0; i < kProtocolCount; ++i) {
        w.sample(label_text({{"protocol", kProtocols[i]}}), series.token_count_requests[i]);
    }

    w.gauge("ninfer_scheduler_running_requests", "Requests with a lane.",
            std::uint64_t{stats.running_requests});
    w.gauge("ninfer_scheduler_prefilling_requests", "Requests in prefill.",
            std::uint64_t{stats.prefilling_requests});
    w.gauge("ninfer_scheduler_decode_ready_requests", "Requests waiting for decode.",
            std::uint64_t{stats.decode_ready_requests});
    w.gauge("ninfer_scheduler_waiting_requests", "Requests in the pending FIFO.",
            std::uint64_t{stats.waiting_requests});
    w.counter("ninfer_engine_computed_prefill_tokens_total", "Prompt tokens evaluated by prefill.",
              stats.computed_prefill_tokens);
    w.counter("ninfer_engine_committed_decode_tokens_total", "Tokens committed by decode rounds.",
              stats.committed_decode_tokens);
    w.counter("ninfer_engine_decode_rounds_total", "Decode batch executions.", stats.decode_rounds);
    w.counter("ninfer_engine_decode_row_rounds_total", "Decode rows across batches.",
              stats.decode_row_rounds);
    w.family("ninfer_gpu_kv_pages", "GPU KV pages by pool and state.", "gauge");
    constexpr const char* kPools[]      = {"main", "spec"};
    constexpr const char* kPageStates[] = {"capacity", "entitled", "free", "mapped"};
    const std::uint32_t pages[2][4] = {
        {stats.gpu_kv_main_capacity_pages, stats.gpu_kv_main_entitled_pages,
         stats.gpu_kv_main_free_pages, stats.gpu_kv_main_mapped_pages},
        {stats.gpu_kv_spec_capacity_pages, stats.gpu_kv_spec_entitled_pages,
         stats.gpu_kv_spec_free_pages, stats.gpu_kv_spec_mapped_pages},
    };
    for (std::size_t pool = 0; pool < 2; ++pool) {
        for (std::size_t state = 0; state < 4; ++state) {
            w.sample(label_text({{"pool", kPools[pool]}, {"state", kPageStates[state]}}),
                     std::uint64_t{pages[pool][state]});
        }
    }
    w.family("ninfer_gpu_kv_capacity_tokens", "GPU KV token capacity by pool.", "gauge");
    for (std::size_t pool = 0; pool < 2; ++pool) {
        w.sample(label_text({{"pool", kPools[pool]}}), std::uint64_t{pages[pool][0]} * kKvPageTokens);
    }
    w.gauge("ninfer_kv_ram_capacity_bytes", "Host RAM prefix cache capacity.",
            std::uint64_t{stats.kv_ram_capacity_bytes});
    w.gauge("ninfer_kv_ram_used_bytes", "Host RAM prefix cache occupancy.",
            std::uint64_t{stats.kv_ram_used_bytes});
    w.gauge("ninfer_kv_ram_entries", "Host RAM prefix cache entries.",
            std::uint64_t{stats.kv_ram_entry_count});
    w.counter("ninfer_kv_ram_captures_total", "Host RAM prefix captures.", stats.kv_ram_captures);
    w.counter("ninfer_kv_ram_restores_total", "Host RAM prefix restores.", stats.kv_ram_restores);
    w.counter("ninfer_kv_ram_evictions_total", "Host RAM prefix evictions.", stats.kv_ram_evictions);
    w.counter("ninfer_kv_ram_drops_total", "Host RAM prefix drops.", stats.kv_ram_drops);
    w.counter("ninfer_kv_ram_save_seconds_total", "Host RAM prefix save seconds.",
              stats.kv_ram_save_seconds);
    w.counter("ninfer_kv_ram_load_seconds_total", "Host RAM prefix load seconds.",
              stats.kv_ram_load_seconds);
    w.gauge("ninfer_kv_disk_capacity_bytes", "Disk prefix cache capacity.",
            std::uint64_t{stats.kv_disk_capacity_bytes});
    w.gauge("ninfer_kv_disk_used_bytes", "Disk prefix cache occupancy.",
            std::uint64_t{stats.kv_disk_used_bytes});
    w.gauge("ninfer_kv_disk_entries", "Disk prefix cache entries.",
            std::uint64_t{stats.kv_disk_entry_count});
    w.counter("ninfer_kv_disk_captures_total", "Disk prefix captures.", stats.kv_disk_captures);
    w.counter("ninfer_kv_disk_restores_total", "Disk prefix restores.", stats.kv_disk_restores);
    w.counter("ninfer_kv_disk_evictions_total", "Disk prefix evictions.", stats.kv_disk_evictions);
    w.counter("ninfer_kv_disk_drops_total", "Disk prefix drops.", stats.kv_disk_drops);
    w.counter("ninfer_kv_disk_save_seconds_total", "Disk prefix save seconds.",
              stats.kv_disk_save_seconds);
    w.counter("ninfer_kv_disk_load_seconds_total", "Disk prefix load seconds.",
              stats.kv_disk_load_seconds);
    w.counter("ninfer_kv_disk_h2d_seconds_total", "Disk restore host to device seconds.",
              stats.kv_disk_h2d_seconds);
    w.counter("ninfer_kv_cache_fallbacks_total", "Prefix restores that fell back to cold prefill.",
              stats.kv_cache_fallbacks);

    w.family("ninfer_prefix_reuse_requests_total", "Prefix reuse path and source.", "counter");
    for (std::size_t path = 0; path < kPathCount; ++path) {
        for (std::size_t source = 0; source < kSourceCount; ++source) {
            w.sample(label_text({{"path", kPrefixPaths[path]}, {"source", kPrefixSources[source]}}),
                     series.prefix_reuse[path][source]);
        }
    }
    w.family("ninfer_prefix_cache_hit_tokens_total", "Prefix hit tokens by source.", "counter");
    for (std::size_t source = 0; source < kSourceCount; ++source) {
        w.sample(label_text({{"source", kPrefixSources[source]}}), series.prefix_hit_tokens[source]);
    }
    w.counter("ninfer_prefix_cache_query_tokens_total", "Prefix query tokens.",
              series.prefix_query_tokens);
    w.counter("ninfer_context_checkpoint_restored_tokens_total", "Context checkpoint tokens restored.",
              series.checkpoint_restored_tokens);
    w.counter("ninfer_context_checkpoint_captured_tokens_total", "Context checkpoint tokens captured.",
              series.checkpoint_captured_tokens);
    w.counter("ninfer_context_checkpoint_capture_requests_total",
              "Requests that asked for a context checkpoint.", series.checkpoint_capture_requests);

    w.counter("ninfer_speculative_rounds_total", "Speculative rounds.", series.speculative_rounds);
    w.counter("ninfer_speculative_draft_tokens_total", "Draft tokens proposed.",
              series.speculative_draft_tokens);
    w.counter("ninfer_speculative_accepted_tokens_total", "Draft tokens accepted.",
              series.speculative_accepted_tokens);
    w.counter("ninfer_speculative_fallback_steps_total", "Speculative fallback steps.",
              series.speculative_fallback_steps);
    w.family("ninfer_speculative_accepted_tokens_position_total", "Accepted draft tokens by position.",
             "counter");
    for (std::size_t position = 0; position < kMaxDraftTokens; ++position) {
        w.sample(label_text({{"position", std::to_string(position)}}),
                 series.speculative_accepted_by_position[position]);
    }
    w.family("ninfer_speculative_rounds_by_k_total", "Speculative rounds by draft k.", "counter");
    for (std::size_t k = 1; k <= kMaxDraftTokens; ++k) {
        w.sample(label_text({{"k", std::to_string(k)}}), series.speculative_rounds_by_k[k - 1]);
    }
    w.histogram("ninfer_speculative_live_k", "Live draft width.", series.speculative_live_k);

    w.family("ninfer_recovery_events_total", "Published recovery events.", "counter");
    for (std::size_t kind = 0; kind < kKindCount; ++kind) {
        for (std::size_t cause = 0; cause < kCauseCount; ++cause) {
            w.sample(label_text({{"cause", kRecoveryCauses[cause]}, {"kind", kRecoveryKinds[kind]}}),
                     series.recovery_events[kind][cause]);
        }
    }
    w.counter("ninfer_recovery_cycle_exclusions_total", "True cycle exclusions on terminal requests.",
              series.recovery_cycle_exclusions);
    w.counter("ninfer_recovery_discarded_reasoning_tokens_total", "Reasoning tokens discarded by recovery.",
              series.recovery_discarded_reasoning_tokens);
    w.counter("ninfer_recovery_discarded_tool_calls_total", "Tool calls discarded by recovery.",
              series.recovery_discarded_tool_calls);
    w.histogram("ninfer_recovery_attempts", "Recovery attempts on terminal requests.",
                series.recovery_attempts);

    w.gauge("ninfer_response_store_records", "Stored response records.",
            static_cast<std::uint64_t>(data.inputs.response_records));
    w.gauge("ninfer_response_store_bytes", "Stored response bytes.",
            static_cast<std::uint64_t>(data.inputs.response_bytes));
    w.gauge("ninfer_response_store_max_records", "Response store record cap.",
            startup.response_max_records);
    w.gauge("ninfer_response_store_max_bytes", "Response store byte cap.", startup.response_max_bytes);
    return w.take();
}

// ---------------------------------------------------------------------------
// JSON.

template <const auto& Bounds>
[[nodiscard]] Json summary(const Histogram<Bounds>& histogram) {
    return Json{{"count", histogram.count}, {"sum", histogram.sum}};
}

template <typename Histograms>
[[nodiscard]] Json summary_all(const Histograms& histograms) {
    std::uint64_t count = 0;
    double sum          = 0;
    for (const auto& histogram : histograms) {
        count += histogram.count;
        sum += histogram.sum;
    }
    return Json{{"count", count}, {"sum", sum}};
}

Json render_json(const MetricsSnapshotData& data) {
    const Startup& startup            = *data.startup;
    const Series& series              = data.series;
    const ninfer::RuntimeStats& stats = data.inputs.stats;

    Json engine = Json::object();
    for (const auto& [key, value] : startup.engine_labels) { engine[key] = value; }
    engine["max_context_tokens"]                  = startup.max_context;
    engine["prefill_chunk_tokens"]                = startup.prefill_chunk;
    engine["pending_timeout_seconds"]             = startup.pending_timeout_seconds;
    engine["default_max_tokens"]                  = startup.default_max_tokens;
    engine["keep_frac"]                           = startup.keep_frac;
    engine["xattn_tau"]                           = startup.xattn_tau;
    engine["xattn_min_len"]                       = startup.xattn_min_len;
    engine["dflash_verify_width"]                 = startup.dflash_verify_width;
    engine["speculative_configured_draft_tokens"] = startup.configured_draft_tokens;
    engine["load_seconds"]                        = startup.load_seconds;

    Json build = Json::object();
    for (const auto& [key, value] : startup.build_labels) { build[key] = value; }

    Json gpu_kv = Json::object();
    const std::uint32_t pools[2][4] = {
        {stats.gpu_kv_main_capacity_pages, stats.gpu_kv_main_entitled_pages,
         stats.gpu_kv_main_mapped_pages, stats.gpu_kv_main_free_pages},
        {stats.gpu_kv_spec_capacity_pages, stats.gpu_kv_spec_entitled_pages,
         stats.gpu_kv_spec_mapped_pages, stats.gpu_kv_spec_free_pages},
    };
    constexpr const char* kPoolNames[] = {"main", "spec"};
    for (std::size_t pool = 0; pool < 2; ++pool) {
        gpu_kv[kPoolNames[pool]] = Json{
            {"capacity", pools[pool][0]},
            {"entitled", pools[pool][1]},
            {"mapped", pools[pool][2]},
            {"free", pools[pool][3]},
            {"capacity_tokens", std::uint64_t{pools[pool][0]} * kKvPageTokens},
        };
    }

    Json reuse_requests = Json::object();
    for (std::size_t path = 0; path < kPathCount; ++path) {
        Json sources = Json::object();
        for (std::size_t source = 0; source < kSourceCount; ++source) {
            sources[kPrefixSources[source]] = series.prefix_reuse[path][source];
        }
        reuse_requests[kPrefixPaths[path]] = std::move(sources);
    }
    Json hit_tokens = Json::object();
    for (std::size_t source = 0; source < kSourceCount; ++source) {
        hit_tokens[kPrefixSources[source]] = series.prefix_hit_tokens[source];
    }

    Json events = Json::object();
    for (std::size_t kind = 0; kind < kKindCount; ++kind) {
        Json causes = Json::object();
        for (std::size_t cause = 0; cause < kCauseCount; ++cause) {
            causes[kRecoveryCauses[cause]] = series.recovery_events[kind][cause];
        }
        events[kRecoveryKinds[kind]] = std::move(causes);
    }

    Json finish = Json::object();
    for (std::size_t i = 0; i < kFinishCount; ++i) { finish[kFinishReasons[i]] = series.finish_reason[i]; }
    std::uint64_t requests_total = 0;
    for (const auto& by_stream : series.generation_requests) {
        for (const auto& by_result : by_stream) {
            for (const auto& by_thinking : by_result) {
                for (const auto& by_tools : by_thinking) {
                    for (const std::uint64_t value : by_tools) { requests_total += value; }
                }
            }
        }
    }

    Json phases = Json::object();
    for (std::size_t i = 0; i < kPhaseCount; ++i) { phases[kPhases[i]] = summary(series.phase[i]); }

    return Json{
        {"engine", std::move(engine)},
        {"build", std::move(build)},
        {"start_time_unix_s", startup.start_time_unix_s},
        {"scheduler",
         {{"running", stats.running_requests},
          {"prefilling", stats.prefilling_requests},
          {"decode_ready", stats.decode_ready_requests},
          {"waiting", stats.waiting_requests},
          {"max_concurrency", startup.max_concurrency},
          {"max_pending_requests", startup.max_pending_requests},
          {"computed_prefill_tokens", stats.computed_prefill_tokens},
          {"committed_decode_tokens", stats.committed_decode_tokens},
          {"decode_rounds", stats.decode_rounds},
          {"decode_row_rounds", stats.decode_row_rounds}}},
        {"http", {{"in_flight", data.inputs.http_in_flight}}},
        {"gpu_kv", std::move(gpu_kv)},
        {"kv_ram",
         {{"capacity_bytes", stats.kv_ram_capacity_bytes},
          {"used_bytes", stats.kv_ram_used_bytes},
          {"entries", stats.kv_ram_entry_count},
          {"captures", stats.kv_ram_captures},
          {"restores", stats.kv_ram_restores},
          {"evictions", stats.kv_ram_evictions},
          {"drops", stats.kv_ram_drops},
          {"save_seconds", stats.kv_ram_save_seconds},
          {"load_seconds", stats.kv_ram_load_seconds}}},
        {"kv_disk",
         {{"capacity_bytes", stats.kv_disk_capacity_bytes},
          {"used_bytes", stats.kv_disk_used_bytes},
          {"entries", stats.kv_disk_entry_count},
          {"captures", stats.kv_disk_captures},
          {"restores", stats.kv_disk_restores},
          {"evictions", stats.kv_disk_evictions},
          {"drops", stats.kv_disk_drops},
          {"save_seconds", stats.kv_disk_save_seconds},
          {"load_seconds", stats.kv_disk_load_seconds},
          {"h2d_seconds", stats.kv_disk_h2d_seconds},
          {"cache_fallbacks", stats.kv_cache_fallbacks}}},
        {"prefix_reuse",
         {{"requests", std::move(reuse_requests)},
          {"hit_tokens", std::move(hit_tokens)},
          {"query_tokens", series.prefix_query_tokens},
          {"restored_tokens", series.checkpoint_restored_tokens},
          {"captured_tokens", series.checkpoint_captured_tokens},
          {"capture_requests", series.checkpoint_capture_requests}}},
        {"speculative",
         {{"rounds", series.speculative_rounds},
          {"draft_tokens", series.speculative_draft_tokens},
          {"accepted_tokens", series.speculative_accepted_tokens},
          {"fallback_steps", series.speculative_fallback_steps},
          {"accepted_by_position", series.speculative_accepted_by_position},
          {"rounds_by_k", series.speculative_rounds_by_k},
          {"live_k", summary(series.speculative_live_k)}}},
        {"recovery",
         {{"events", std::move(events)},
          {"cycle_exclusions", series.recovery_cycle_exclusions},
          {"discarded_reasoning_tokens", series.recovery_discarded_reasoning_tokens},
          {"discarded_tool_calls", series.recovery_discarded_tool_calls},
          {"attempts", summary(series.recovery_attempts)}}},
        {"generation",
         {{"requests_total", requests_total},
          {"finish_reason", std::move(finish)},
          {"tool_calls", series.tool_calls},
          {"ignored_tool_markup", series.ignored_tool_markup},
          {"media_requests", series.media_requests},
          {"ttft", summary_all(series.ttft)},
          {"e2e", summary_all(series.e2e)},
          {"inter_token_latency", summary_all(series.inter_token)},
          {"output_tokens_per_second", summary(series.output_tokens_per_second)},
          {"prefill_tokens_per_second", summary(series.prefill_tokens_per_second)},
          {"prompt_tokens", summary(series.prompt_tokens)},
          {"completion_tokens", summary(series.completion_tokens)},
          {"reasoning_tokens", summary(series.reasoning_tokens)},
          {"computed_prefill_tokens", summary(series.computed_prefill_tokens)}}},
        {"phases", std::move(phases)},
        {"response_store",
         {{"records", data.inputs.response_records},
          {"bytes", data.inputs.response_bytes},
          {"max_records", startup.response_max_records},
          {"max_bytes", startup.response_max_bytes}}},
    };
}

} // namespace

MetricsSnapshot::MetricsSnapshot()                                 = default;
MetricsSnapshot::~MetricsSnapshot()                                = default;
MetricsSnapshot::MetricsSnapshot(MetricsSnapshot&&) noexcept            = default;
MetricsSnapshot& MetricsSnapshot::operator=(MetricsSnapshot&&) noexcept = default;

std::string MetricsSnapshot::prometheus_text() const {
    return data_ ? render_text(*data_) : std::string();
}

std::string MetricsSnapshot::json() const { return data_ ? render_json(*data_).dump() : "{}"; }

ServeMetrics::ServeMetrics() : state_(std::make_unique<ServeMetricsState>()) {}
ServeMetrics::~ServeMetrics() = default;

void ServeMetrics::attach(const ServeOptions& options, const ninfer::LoadSummary& load,
                          const ninfer::MemorySummary& memory, std::string_view model_id,
                          std::string_view cuda_compile_version,
                          std::string_view cuda_runtime_version) {
    auto startup                        = std::make_shared<Startup>();
    startup->max_context                = options.max_context;
    startup->prefill_chunk              = options.prefill_chunk;
    startup->pending_timeout_seconds    = static_cast<double>(options.pending_timeout_ms) / 1000.0;
    startup->default_max_tokens         = options.default_max_tokens;
    startup->keep_frac                  = options.keep_frac;
    startup->xattn_tau                  = options.xattn_tau;
    startup->dflash_verify_width        = options.speculative.dflash_verify_width;
    startup->configured_draft_tokens    = options.speculative.draft_tokens;
    startup->load_seconds               = load.load_seconds;
    startup->cuda_graph_allowance_bytes = static_cast<double>(memory.cuda_graph_allowance_bytes);
    startup->cuda_graph_observed_bytes  = static_cast<double>(memory.cuda_graph_observed_bytes);
    startup->arena_weights              = static_cast<double>(memory.weights.capacity_bytes);
    startup->arena_sequence             = static_cast<double>(memory.sequence.capacity_bytes);
    startup->arena_workspace            = static_cast<double>(memory.workspace.capacity_bytes);
    startup->arena_request_transient    = static_cast<double>(memory.request_transient.capacity_bytes);
    startup->start_time_unix_s =
        std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
    startup->max_concurrency      = options.max_concurrency;
    startup->max_pending_requests = options.max_pending_requests;
    startup->response_max_records = static_cast<double>(options.response_store_max_records);
    startup->response_max_bytes   = static_cast<double>(options.response_store_max_bytes);
    const char* checkpoints = "default";
    if (options.context_checkpoint_marks.has_value()) {
        checkpoints = options.context_checkpoint_marks->empty() ? "off" : "custom";
    }
    startup->engine_labels = {
        {"adaptive_draft", yn(options.speculative.adaptive_draft)},
        {"auth", yn(!options.api_key.empty())},
        {"context_checkpoints", checkpoints},
        {"cors", yn(options.enable_cors)},
        {"cuda_graph", yn(options.use_cuda_graph)},
        {"device", std::to_string(options.device)},
        {"generation_recovery", yn(options.generation_recovery)},
        {"greedy", yn(options.greedy)},
        {"kv_capacity_mode", kv_capacity_mode_name(options.kv_capacity.mode)},
        {"kv_disk_compress", options.kv_disk_compress == ninfer::KvDiskCompress::Zstd ? "zstd" : "off"},
        {"kv_dtype", kv_cache_name(options.kv_cache)},
        {"model_id", std::string(model_id)},
        {"p_less", yn(options.sampling_overrides.p_less)},
        {"prefix_reuse", yn(options.allow_prefix_reuse)},
        {"proposal_head", proposal_head_name(options.speculative.proposal_head)},
        {"sage_attn", yn(options.sage_attn)},
        {"spec", product::speculative_backend_name(options.speculative.backend)},
        {"system_prepend", yn(!options.system_prepend.empty())},
        {"target", load.target},
        {"vision", yn(options.enable_vision)},
        {"weights_id", load.weights_id},
    };
    startup->build_labels = {{"cuda_compile_version", std::string(cuda_compile_version)},
                             {"cuda_runtime_version", std::string(cuda_runtime_version)}};
    std::lock_guard lock(state_->mutex);
    state_->startup = std::move(startup);
}

void ServeMetrics::observe_generation(const GenerationObservation& observation) {
    const auto protocol = static_cast<std::size_t>(observation.protocol);
    const auto result   = static_cast<std::size_t>(observation.result);
    std::lock_guard lock(state_->mutex);
    Series& series = state_->series;
    ++series.generation_requests[protocol][observation.stream][result][observation.thinking]
                                [observation.tools];
    if (observation.capture_requested) { ++series.checkpoint_capture_requests; }
    if (observation.has_media) { ++series.media_requests; }
    if (observation.result == GenerationResult::Rejected) { return; }
    if (observation.outcome != nullptr) {
        observe_terminal_recovery(series, observation.outcome->metrics.recovery);
    } else if (observation.recovery != nullptr) {
        observe_terminal_recovery(series, *observation.recovery);
    }
    if (observation.result == GenerationResult::Success && observation.outcome != nullptr) {
        observe_success(series, protocol, *observation.outcome);
    }
}

void ServeMetrics::observe_recovery_event(const ninfer::RecoveryEvent& event) {
    const std::size_t kind  = recovery_kind_index(event.kind);
    const std::size_t cause = index_of(
        kRecoveryCauses, prometheus_recovery_cause(kRecoveryKinds[kind], event.cause), kCauseCount - 1);
    std::lock_guard lock(state_->mutex);
    ++state_->series.recovery_events[kind][cause];
}

void ServeMetrics::observe_http(std::string_view path, std::string_view method, int status,
                                double seconds) {
    const std::size_t route        = route_index(path);
    const std::size_t method_index = index_of(kMethods, method, kMethodCount - 1);
    const std::uint32_t status_key =
        status >= 100 && status <= 599 ? static_cast<std::uint32_t>(status) : 0U;
    const std::uint32_t key = static_cast<std::uint32_t>(route) << 24 |
                              static_cast<std::uint32_t>(method_index) << 16 | status_key;
    std::lock_guard lock(state_->mutex);
    ++state_->series.http_requests[key];
    if (seconds >= 0.0) { state_->series.http_duration[route].observe(seconds); }
}

void ServeMetrics::observe_api_error(std::string_view code) {
    const std::size_t index = index_of(kApiErrorCodes, code.empty() ? "unnamed" : code,
                                       kApiCodeCount - 1);
    std::lock_guard lock(state_->mutex);
    ++state_->series.api_errors[index];
}

void ServeMetrics::observe_token_count(MetricsProtocol protocol) {
    std::lock_guard lock(state_->mutex);
    ++state_->series.token_count_requests[static_cast<std::size_t>(protocol)];
}

MetricsSnapshot ServeMetrics::snapshot(const ScrapeInputs& inputs) const {
    auto data    = std::make_unique<MetricsSnapshotData>();
    data->inputs = inputs;
    {
        std::lock_guard lock(state_->mutex);
        data->startup = state_->startup;
        data->series  = state_->series;
    }
    MetricsSnapshot snapshot;
    snapshot.data_ = std::move(data);
    return snapshot;
}

void handle_metrics(const MetricsSnapshot& snapshot, httplib::Response& response) {
    response.status = 200;
    response.set_content(snapshot.prometheus_text(), kPrometheusContentType);
}

void handle_metrics_json(const MetricsSnapshot& snapshot, httplib::Response& response) {
    response.status = 200;
    response.set_content(snapshot.json(), "application/json");
}

HttpRouteClass classify_http_route(std::string_view path) { return kRoutes[route_index(path)]; }

bool is_unauthenticated_path(std::string_view path) {
    return path == "/health" || path == "/metrics" || path == "/metrics.json";
}

MetricsProtocol metrics_protocol(std::string_view request_log_protocol) {
    if (request_log_protocol == "openai_chat_completions") { return MetricsProtocol::OpenAiChat; }
    if (request_log_protocol == "openai_responses") { return MetricsProtocol::OpenAiResponses; }
    if (request_log_protocol == "anthropic_messages") { return MetricsProtocol::AnthropicMessages; }
    throw std::logic_error("unknown request-log protocol: " + std::string(request_log_protocol));
}

const char* prometheus_recovery_cause(std::string_view kind, std::string_view cause) {
    for (const char* direct : kDirectRecoveryCauses) {
        if (cause == direct) { return direct; }
    }
    if (kind == "exhausted") {
        const auto has = [cause](std::string_view needle) {
            return cause.find(needle) != std::string_view::npos;
        };
        if (has("retry or output-token budget")) { return "retry_budget"; }
        if (has("cannot preserve its admitted output budget")) { return "output_budget"; }
        if (has("not the thinking prologue")) { return "prologue"; }
        if (has("cannot be rebuilt safely")) { return "lane_rebuild"; }
    }
    return "other";
}

} // namespace ninfer::serve
