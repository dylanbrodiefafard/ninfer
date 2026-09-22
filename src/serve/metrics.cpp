#include "serve/metrics.h"

#include "product/speculative_options.h"
#include "serve/request_log.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ninfer::serve {
namespace {

using Json = nlohmann::json;

// Same value as ninfer::kPagedKVPageSize. Serve does not include the core KV header.
constexpr std::uint32_t kKvPageTokens = 64;

constexpr double kSecondBounds[] = {0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5,
                                    5,     10,   30,    60,   120,  300};
constexpr const char* kSecondTexts[] = {"0.005", "0.01", "0.025", "0.05", "0.1", "0.25", "0.5",
                                        "1",     "2.5",  "5",     "10",   "30",  "60",   "120",
                                        "300"};
constexpr double kTokenBounds[] = {32,   64,   128,   256,    512,    1024,   2048,
                                   4096, 8192, 16384, 32768,  65536,  131072};
constexpr const char* kTokenTexts[] = {"32",   "64",   "128",  "256",   "512",   "1024",  "2048",
                                       "4096", "8192", "16384", "32768", "65536", "131072"};
constexpr double kToksBounds[] = {5, 10, 25, 50, 75, 100, 150, 200, 250, 300, 400, 500, 750, 1000};
constexpr const char* kToksTexts[] = {"5",  "10",  "25",  "50",  "75",  "100", "150",
                                      "200", "250", "300", "400", "500", "750", "1000"};
constexpr double kDraftBounds[] = {1, 2, 3, 4, 5, 6, 8, 10, 12, 15};
constexpr const char* kDraftTexts[] = {"1", "2", "3", "4", "5", "6", "8", "10", "12", "15"};

constexpr const char* kProtocols[] = {"openai_chat", "openai_responses", "anthropic_messages"};
constexpr const char* kStreams[]   = {"false", "true"};
constexpr const char* kResults[]   = {"cancelled", "error", "rejected", "success"};
constexpr const char* kBools[]     = {"false", "true"};
constexpr const char* kPhases[]    = {"prepare_cpu", "media_wait", "media_fetch", "queue",
                                      "copy_hold",   "vision",      "prefill",     "decode",
                                      "recovery",    "http_tail"};
constexpr const char* kFinishReasons[] = {"none",     "output_limit", "context_capacity", "stop_token",
                                          "stop_string", "cancelled",  "tool_calls",       "unknown"};
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
    "cycle_exclusion", "retry_triggered",        "retry_started",
    "retry_prefill_complete", "finished",        "exhausted",
};
constexpr const char* kRecoveryCauses[] = {
    "reasoning_cycle",     "repeated_reasoning", "duplicate_tool_call", "cancelled",
    "tool_calls",          "output_limit",       "context_capacity",    "stop",
    "reservation_exceeded", "retry_budget",      "lane_rebuild",        "other",
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
constexpr const char* kTiers[] = {"disk", "ram"};
constexpr const char* kCopyOps[] = {"h2d", "load", "save"};
constexpr const char* kMethods[] = {"DELETE", "GET", "OPTIONS", "PATCH", "POST", "PUT"};

enum class BucketKind { None, Seconds, Tokens, Toks, Draft };

struct BoundSet {
    const double* values = nullptr;
    const char* const* texts = nullptr;
    std::size_t count = 0;
};

[[nodiscard]] BoundSet bounds_for(BucketKind kind) {
    switch (kind) {
    case BucketKind::Seconds:
        return {kSecondBounds, kSecondTexts, std::size(kSecondBounds)};
    case BucketKind::Tokens:
        return {kTokenBounds, kTokenTexts, std::size(kTokenBounds)};
    case BucketKind::Toks:
        return {kToksBounds, kToksTexts, std::size(kToksBounds)};
    case BucketKind::Draft:
        return {kDraftBounds, kDraftTexts, std::size(kDraftBounds)};
    case BucketKind::None:
        break;
    }
    return {};
}

[[nodiscard]] BucketKind bucket_kind(std::string_view family) {
    if (family == "ninfer_http_request_duration_seconds" ||
        family == "ninfer_generation_ttft_seconds" || family == "ninfer_generation_e2e_seconds" ||
        family == "ninfer_generation_inter_token_latency_seconds" ||
        family == "ninfer_generation_phase_seconds" ||
        family == "ninfer_generation_kv_copy_seconds") {
        return BucketKind::Seconds;
    }
    if (family == "ninfer_generation_prompt_tokens" ||
        family == "ninfer_generation_completion_tokens" ||
        family == "ninfer_generation_reasoning_tokens" ||
        family == "ninfer_generation_computed_prefill_tokens") {
        return BucketKind::Tokens;
    }
    if (family == "ninfer_generation_output_tokens_per_second" ||
        family == "ninfer_generation_prefill_tokens_per_second") {
        return BucketKind::Toks;
    }
    if (family == "ninfer_speculative_live_k" || family == "ninfer_recovery_attempts") {
        return BucketKind::Draft;
    }
    return BucketKind::None;
}

struct Labels {
    std::vector<std::pair<std::string, std::string>> pairs;

    Labels() = default;
    Labels(std::initializer_list<std::pair<const char*, const char*>> items) {
        pairs.reserve(items.size());
        for (const auto& item : items) { pairs.emplace_back(item.first, item.second); }
        sort();
    }

    void add(std::string key, std::string value) { pairs.emplace_back(std::move(key), std::move(value)); }

    void sort() {
        std::sort(pairs.begin(), pairs.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
    }
};

[[nodiscard]] std::string series_key(std::string_view family, const Labels& labels) {
    std::string key(family);
    key.push_back('\n');
    for (const auto& [name, value] : labels.pairs) {
        key.append(name);
        key.push_back('=');
        key.append(value);
        key.push_back('\n');
    }
    return key;
}

[[nodiscard]] Labels labels_from_key(std::string_view key) {
    Labels labels;
    const auto split = key.find('\n');
    if (split == std::string_view::npos) { return labels; }
    std::string_view rest = key.substr(split + 1);
    while (!rest.empty()) {
        const auto end = rest.find('\n');
        const auto row = rest.substr(0, end);
        const auto eq  = row.find('=');
        if (eq != std::string_view::npos) {
            labels.pairs.emplace_back(std::string(row.substr(0, eq)), std::string(row.substr(eq + 1)));
        }
        if (end == std::string_view::npos) { break; }
        rest.remove_prefix(end + 1);
    }
    return labels;
}

[[nodiscard]] std::string escape_label(std::string_view value) {
    std::string out;
    out.reserve(value.size());
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
    return out;
}

void append_number(std::string& out, double value) {
    if (!std::isfinite(value)) {
        out.append("0");
        return;
    }
    char buf[64];
    if (value >= 0.0 && value == std::trunc(value) &&
        value <= static_cast<double>(std::numeric_limits<std::uint64_t>::max())) {
        const auto result =
            std::to_chars(buf, buf + sizeof(buf), static_cast<std::uint64_t>(value));
        out.append(buf, result.ptr);
        return;
    }
    const auto result =
        std::to_chars(buf, buf + sizeof(buf), value, std::chars_format::general, 15);
    if (result.ec != std::errc{}) {
        out.append("0");
        return;
    }
    out.append(buf, result.ptr);
}

[[nodiscard]] std::string format_labels(const Labels& labels) {
    if (labels.pairs.empty()) { return {}; }
    std::string out = "{";
    bool first = true;
    for (const auto& [key, value] : labels.pairs) {
        if (!first) { out.push_back(','); }
        first = false;
        out.append(key);
        out.append("=\"");
        out.append(escape_label(value));
        out.push_back('"');
    }
    out.push_back('}');
    return out;
}

[[nodiscard]] const char* yn(bool value) { return value ? "true" : "false"; }

[[nodiscard]] const char* known_method(std::string_view method) {
    for (const char* known : kMethods) {
        if (method == known) { return known; }
    }
    return "other";
}

[[nodiscard]] const char* known_api_code(std::string_view code) {
    if (code.empty()) { return "unnamed"; }
    for (const char* known : kApiErrorCodes) {
        if (code == known) { return known; }
    }
    return "other";
}

[[nodiscard]] const char* known_protocol(std::string_view protocol) {
    for (const char* known : kProtocols) {
        if (protocol == known) { return known; }
    }
    return "openai_chat";
}

[[nodiscard]] bool contains_text(std::string_view haystack, std::string_view needle) {
    return haystack.find(needle) != std::string_view::npos;
}

struct HistState {
    std::vector<std::uint64_t> buckets;
    double sum          = 0;
    std::uint64_t count = 0;
};

[[nodiscard]] HistState make_hist(BucketKind kind) {
    HistState state;
    const BoundSet bounds = bounds_for(kind);
    state.buckets.assign(bounds.count + 1, 0);
    return state;
}

struct Sample {
    Labels labels;
    double value          = 0;
    double sum            = 0;
    std::uint64_t count   = 0;
    std::vector<std::uint64_t> buckets;
};

struct Family {
    std::string name;
    const char* help = "";
    const char* type = "gauge";
    BucketKind buckets = BucketKind::None;
    std::vector<Sample> samples;
};

struct Startup {
    double max_context                 = 0;
    double prefill_chunk               = 0;
    double pending_timeout_seconds     = 0;
    double default_max_tokens          = 0;
    double keep_frac                   = 1;
    double xattn_tau                   = 1;
    double xattn_min_len               = static_cast<double>(kDefaultXattnMinLen);
    double dflash_verify_width         = 0;
    double configured_draft_tokens     = 0;
    double load_seconds                = 0;
    double cuda_graph_allowance_bytes  = 0;
    double cuda_graph_observed_bytes   = 0;
    double arena_weights               = 0;
    double arena_sequence              = 0;
    double arena_workspace             = 0;
    double arena_request_transient     = 0;
    double start_time_unix_s           = 0;
    double max_concurrency             = 0;
    double max_pending_requests        = 0;
    double response_max_records        = 0;
    double response_max_bytes          = 0;
    Labels engine_labels;
    Labels build_labels;
};

void add_info(std::vector<Family>& out, const char* name, const char* help, const Labels& labels) {
    Family family;
    family.name = name;
    family.help = help;
    family.type = "gauge";
    Sample sample;
    sample.labels = labels;
    sample.value  = 1;
    family.samples.push_back(std::move(sample));
    out.push_back(std::move(family));
}

void add_gauge(std::vector<Family>& out, const char* name, const char* help, double value) {
    Family family;
    family.name = name;
    family.help = help;
    family.type = "gauge";
    Sample sample;
    sample.value = value;
    family.samples.push_back(std::move(sample));
    out.push_back(std::move(family));
}

void add_absolute(std::vector<Family>& out, const char* name, const char* help, double value) {
    Family family;
    family.name = name;
    family.help = help;
    family.type = "counter";
    Sample sample;
    sample.value = value;
    family.samples.push_back(std::move(sample));
    out.push_back(std::move(family));
}

void add_labeled_gauges(std::vector<Family>& out, const char* name, const char* help,
                        std::vector<Sample> samples) {
    Family family;
    family.name    = name;
    family.help    = help;
    family.type    = "gauge";
    family.samples = std::move(samples);
    out.push_back(std::move(family));
}

Sample labeled_value(std::initializer_list<std::pair<const char*, const char*>> labels, double value) {
    Sample sample;
    sample.labels = Labels(labels);
    sample.value  = value;
    return sample;
}

void add_mapped_counter(std::vector<Family>& out, const char* name, const char* help,
                        const std::map<std::string, std::uint64_t>& counters) {
    Family family;
    family.name = name;
    family.help = help;
    family.type = "counter";
    const std::string prefix = std::string(name) + "\n";
    for (const auto& [key, value] : counters) {
        if (key.compare(0, prefix.size(), prefix) != 0) { continue; }
        Sample sample;
        sample.labels = labels_from_key(key);
        sample.value  = static_cast<double>(value);
        family.samples.push_back(std::move(sample));
    }
    out.push_back(std::move(family));
}

void add_mapped_histogram(std::vector<Family>& out, const char* name, const char* help,
                          BucketKind kind, const std::map<std::string, HistState>& hists) {
    Family family;
    family.name    = name;
    family.help    = help;
    family.type    = "histogram";
    family.buckets = kind;
    const std::string prefix = std::string(name) + "\n";
    for (const auto& [key, state] : hists) {
        if (key.compare(0, prefix.size(), prefix) != 0) { continue; }
        Sample sample;
        sample.labels  = labels_from_key(key);
        sample.sum     = state.sum;
        sample.count   = state.count;
        sample.buckets = state.buckets;
        family.samples.push_back(std::move(sample));
    }
    out.push_back(std::move(family));
}

[[nodiscard]] const char* label_value(const Labels& labels, const char* key) {
    for (const auto& [name, value] : labels.pairs) {
        if (name == key) { return value.c_str(); }
    }
    return "";
}

void append_help_type(std::string& out, const Family& family) {
    out.append("# HELP ");
    out.append(family.name);
    out.push_back(' ');
    out.append(family.help);
    out.push_back('\n');
    out.append("# TYPE ");
    out.append(family.name);
    out.push_back(' ');
    out.append(family.type);
    out.push_back('\n');
}

void append_sample_line(std::string& out, const std::string& name, const std::string& suffix,
                        const Labels& labels, double value) {
    out.append(name);
    out.append(suffix);
    out.append(format_labels(labels));
    out.push_back(' ');
    append_number(out, value);
    out.push_back('\n');
}

std::string render_families(const std::vector<Family>& families) {
    std::string out;
    out.reserve(256 * 1024);
    for (const Family& family : families) {
        append_help_type(out, family);
        if (family.type == std::string_view("histogram")) {
            const BoundSet bounds = bounds_for(family.buckets);
            for (const Sample& sample : family.samples) {
                std::uint64_t cumulative = 0;
                const std::size_t finite = std::min(bounds.count, sample.buckets.size());
                for (std::size_t i = 0; i < finite; ++i) {
                    cumulative += sample.buckets[i];
                    Labels labels = sample.labels;
                    labels.add("le", bounds.texts[i]);
                    labels.sort();
                    append_sample_line(out, family.name, "_bucket", labels,
                                       static_cast<double>(cumulative));
                }
                if (!sample.buckets.empty()) {
                    for (std::size_t i = finite; i < sample.buckets.size(); ++i) {
                        cumulative += sample.buckets[i];
                    }
                }
                Labels inf = sample.labels;
                inf.add("le", "+Inf");
                inf.sort();
                append_sample_line(out, family.name, "_bucket", inf, static_cast<double>(cumulative));
                append_sample_line(out, family.name, "_sum", sample.labels, sample.sum);
                append_sample_line(out, family.name, "_count", sample.labels,
                                   static_cast<double>(sample.count));
            }
        } else {
            for (const Sample& sample : family.samples) {
                append_sample_line(out, family.name, "", sample.labels, sample.value);
            }
        }
    }
    return out;
}

[[nodiscard]] double unlabeled(const Family& family) {
    return family.samples.empty() ? 0.0 : family.samples.front().value;
}

[[nodiscard]] const Sample* find_labels(const Family& family, const char* key, const char* value) {
    for (const Sample& sample : family.samples) {
        if (std::string_view(label_value(sample.labels, key)) == value) { return &sample; }
    }
    return nullptr;
}

void add_hist_json_all(Json& object, const char* field, const Family& family) {
    std::uint64_t count = 0;
    double sum          = 0;
    for (const Sample& sample : family.samples) {
        count += sample.count;
        sum += sample.sum;
    }
    object[field] = Json{{"count", count}, {"sum", sum}};
}

Json render_json(const std::vector<Family>& families) {
    Json json = Json::object();
    json["engine"]          = Json::object();
    json["build"]           = Json::object();
    json["scheduler"]       = Json::object();
    json["http"]            = Json::object();
    json["gpu_kv"]          = Json{{"main", Json::object()}, {"spec", Json::object()}};
    json["kv_ram"]          = Json::object();
    json["kv_disk"]         = Json::object();
    json["prefix_reuse"]    = Json{{"requests", Json::object()}, {"hit_tokens", Json::object()}};
    json["speculative"]     = Json::object();
    json["recovery"]        = Json{{"events", Json::object()}};
    json["generation"]      = Json::object();
    json["phases"]          = Json::object();
    json["response_store"]  = Json::object();
    json["start_time_unix_s"] = 0.0;

    const auto family_named = [&](const char* name) -> const Family* {
        for (const Family& family : families) {
            if (family.name == name) { return &family; }
        }
        return nullptr;
    };

    if (const Family* info = family_named("ninfer_engine_info")) {
        if (!info->samples.empty()) {
            for (const auto& [key, value] : info->samples.front().labels.pairs) {
                json["engine"][key] = value;
            }
        }
    }
    if (const Family* info = family_named("ninfer_build_info")) {
        if (!info->samples.empty()) {
            for (const auto& [key, value] : info->samples.front().labels.pairs) {
                json["build"][key] = value;
            }
        }
    }

    const auto gauge_into = [&](const char* name, Json& object, const char* field) {
        if (const Family* family = family_named(name)) { object[field] = unlabeled(*family); }
    };
    gauge_into("ninfer_max_context_tokens", json["engine"], "max_context_tokens");
    gauge_into("ninfer_prefill_chunk_tokens", json["engine"], "prefill_chunk_tokens");
    gauge_into("ninfer_pending_timeout_seconds", json["engine"], "pending_timeout_seconds");
    gauge_into("ninfer_default_max_tokens", json["engine"], "default_max_tokens");
    gauge_into("ninfer_keep_frac", json["engine"], "keep_frac");
    gauge_into("ninfer_xattn_tau", json["engine"], "xattn_tau");
    gauge_into("ninfer_xattn_min_len", json["engine"], "xattn_min_len");
    gauge_into("ninfer_dflash_verify_width", json["engine"], "dflash_verify_width");
    gauge_into("ninfer_speculative_configured_draft_tokens", json["engine"],
               "speculative_configured_draft_tokens");
    gauge_into("ninfer_engine_load_seconds", json["engine"], "load_seconds");
    gauge_into("ninfer_server_start_time_seconds", json, "start_time_unix_s");
    gauge_into("ninfer_scheduler_running_requests", json["scheduler"], "running");
    gauge_into("ninfer_scheduler_prefilling_requests", json["scheduler"], "prefilling");
    gauge_into("ninfer_scheduler_decode_ready_requests", json["scheduler"], "decode_ready");
    gauge_into("ninfer_scheduler_waiting_requests", json["scheduler"], "waiting");
    gauge_into("ninfer_scheduler_max_concurrency", json["scheduler"], "max_concurrency");
    gauge_into("ninfer_scheduler_max_pending_requests", json["scheduler"], "max_pending_requests");
    gauge_into("ninfer_http_in_flight_requests", json["http"], "in_flight");
    gauge_into("ninfer_kv_ram_capacity_bytes", json["kv_ram"], "capacity_bytes");
    gauge_into("ninfer_kv_ram_used_bytes", json["kv_ram"], "used_bytes");
    gauge_into("ninfer_kv_ram_entries", json["kv_ram"], "entries");
    gauge_into("ninfer_kv_disk_capacity_bytes", json["kv_disk"], "capacity_bytes");
    gauge_into("ninfer_kv_disk_used_bytes", json["kv_disk"], "used_bytes");
    gauge_into("ninfer_kv_disk_entries", json["kv_disk"], "entries");
    gauge_into("ninfer_response_store_records", json["response_store"], "records");
    gauge_into("ninfer_response_store_bytes", json["response_store"], "bytes");
    gauge_into("ninfer_response_store_max_records", json["response_store"], "max_records");
    gauge_into("ninfer_response_store_max_bytes", json["response_store"], "max_bytes");

    const auto counter_into = [&](const char* name, Json& object, const char* field) {
        if (const Family* family = family_named(name)) { object[field] = unlabeled(*family); }
    };
    counter_into("ninfer_engine_computed_prefill_tokens_total", json["scheduler"],
                 "computed_prefill_tokens");
    counter_into("ninfer_engine_committed_decode_tokens_total", json["scheduler"],
                 "committed_decode_tokens");
    counter_into("ninfer_engine_decode_rounds_total", json["scheduler"], "decode_rounds");
    counter_into("ninfer_engine_decode_row_rounds_total", json["scheduler"], "decode_row_rounds");
    counter_into("ninfer_kv_ram_captures_total", json["kv_ram"], "captures");
    counter_into("ninfer_kv_ram_restores_total", json["kv_ram"], "restores");
    counter_into("ninfer_kv_ram_evictions_total", json["kv_ram"], "evictions");
    counter_into("ninfer_kv_ram_drops_total", json["kv_ram"], "drops");
    counter_into("ninfer_kv_ram_save_seconds_total", json["kv_ram"], "save_seconds");
    counter_into("ninfer_kv_ram_load_seconds_total", json["kv_ram"], "load_seconds");
    counter_into("ninfer_kv_disk_captures_total", json["kv_disk"], "captures");
    counter_into("ninfer_kv_disk_restores_total", json["kv_disk"], "restores");
    counter_into("ninfer_kv_disk_evictions_total", json["kv_disk"], "evictions");
    counter_into("ninfer_kv_disk_drops_total", json["kv_disk"], "drops");
    counter_into("ninfer_kv_disk_save_seconds_total", json["kv_disk"], "save_seconds");
    counter_into("ninfer_kv_disk_load_seconds_total", json["kv_disk"], "load_seconds");
    counter_into("ninfer_kv_disk_h2d_seconds_total", json["kv_disk"], "h2d_seconds");
    counter_into("ninfer_kv_cache_fallbacks_total", json["kv_disk"], "cache_fallbacks");
    counter_into("ninfer_prefix_cache_query_tokens_total", json["prefix_reuse"], "query_tokens");
    counter_into("ninfer_context_checkpoint_restored_tokens_total", json["prefix_reuse"],
                 "restored_tokens");
    counter_into("ninfer_context_checkpoint_captured_tokens_total", json["prefix_reuse"],
                 "captured_tokens");
    counter_into("ninfer_context_checkpoint_capture_requests_total", json["prefix_reuse"],
                 "capture_requests");
    counter_into("ninfer_speculative_rounds_total", json["speculative"], "rounds");
    counter_into("ninfer_speculative_draft_tokens_total", json["speculative"], "draft_tokens");
    counter_into("ninfer_speculative_accepted_tokens_total", json["speculative"], "accepted_tokens");
    counter_into("ninfer_speculative_fallback_steps_total", json["speculative"], "fallback_steps");
    counter_into("ninfer_recovery_cycle_exclusions_total", json["recovery"], "cycle_exclusions");
    counter_into("ninfer_recovery_discarded_reasoning_tokens_total", json["recovery"],
                 "discarded_reasoning_tokens");
    counter_into("ninfer_recovery_discarded_tool_calls_total", json["recovery"],
                 "discarded_tool_calls");
    counter_into("ninfer_generation_tool_calls_total", json["generation"], "tool_calls");
    counter_into("ninfer_generation_ignored_tool_markup_total", json["generation"],
                 "ignored_tool_markup");
    counter_into("ninfer_generation_media_requests_total", json["generation"], "media_requests");

    if (const Family* pages = family_named("ninfer_gpu_kv_pages")) {
        for (const char* pool : {"main", "spec"}) {
            for (const char* state : {"capacity", "entitled", "mapped", "free"}) {
                for (const Sample& sample : pages->samples) {
                    if (std::string_view(label_value(sample.labels, "pool")) == pool &&
                        std::string_view(label_value(sample.labels, "state")) == state) {
                        json["gpu_kv"][pool][state] = sample.value;
                    }
                }
            }
        }
    }
    if (const Family* tokens = family_named("ninfer_gpu_kv_capacity_tokens")) {
        for (const Sample& sample : tokens->samples) {
            json["gpu_kv"][label_value(sample.labels, "pool")]["capacity_tokens"] = sample.value;
        }
    }
    if (const Family* reuse = family_named("ninfer_prefix_reuse_requests_total")) {
        for (const Sample& sample : reuse->samples) {
            json["prefix_reuse"]["requests"][label_value(sample.labels, "path")]
                                [label_value(sample.labels, "source")] = sample.value;
        }
    }
    if (const Family* hits = family_named("ninfer_prefix_cache_hit_tokens_total")) {
        for (const Sample& sample : hits->samples) {
            json["prefix_reuse"]["hit_tokens"][label_value(sample.labels, "source")] = sample.value;
        }
    }
    if (const Family* positions = family_named("ninfer_speculative_accepted_tokens_position_total")) {
        Json values = Json::array();
        for (int position = 0; position < 15; ++position) {
            double value = 0;
            for (const Sample& sample : positions->samples) {
                if (label_value(sample.labels, "position") == std::to_string(position)) {
                    value = sample.value;
                }
            }
            values.push_back(value);
        }
        json["speculative"]["accepted_by_position"] = std::move(values);
    }
    if (const Family* rounds = family_named("ninfer_speculative_rounds_by_k_total")) {
        Json values = Json::array();
        for (int k = 1; k <= 15; ++k) {
            double value = 0;
            for (const Sample& sample : rounds->samples) {
                if (label_value(sample.labels, "k") == std::to_string(k)) { value = sample.value; }
            }
            values.push_back(value);
        }
        json["speculative"]["rounds_by_k"] = std::move(values);
    }
    if (const Family* events = family_named("ninfer_recovery_events_total")) {
        for (const Sample& sample : events->samples) {
            json["recovery"]["events"][label_value(sample.labels, "kind")]
                            [label_value(sample.labels, "cause")] = sample.value;
        }
    }
    if (const Family* finish = family_named("ninfer_generation_finish_reason_total")) {
        Json reasons = Json::object();
        for (const Sample& sample : finish->samples) {
            reasons[label_value(sample.labels, "reason")] = sample.value;
        }
        json["generation"]["finish_reason"] = std::move(reasons);
    }
    if (const Family* requests = family_named("ninfer_generation_requests_total")) {
        double total = 0;
        for (const Sample& sample : requests->samples) { total += sample.value; }
        json["generation"]["requests_total"] = total;
    }
    if (const Family* family = family_named("ninfer_generation_ttft_seconds")) {
        add_hist_json_all(json["generation"], "ttft", *family);
    }
    if (const Family* family = family_named("ninfer_generation_e2e_seconds")) {
        add_hist_json_all(json["generation"], "e2e", *family);
    }
    if (const Family* family = family_named("ninfer_generation_inter_token_latency_seconds")) {
        add_hist_json_all(json["generation"], "inter_token_latency", *family);
    }
    if (const Family* family = family_named("ninfer_generation_output_tokens_per_second")) {
        add_hist_json_all(json["generation"], "output_tokens_per_second", *family);
    }
    if (const Family* family = family_named("ninfer_generation_prefill_tokens_per_second")) {
        add_hist_json_all(json["generation"], "prefill_tokens_per_second", *family);
    }
    if (const Family* family = family_named("ninfer_generation_prompt_tokens")) {
        add_hist_json_all(json["generation"], "prompt_tokens", *family);
    }
    if (const Family* family = family_named("ninfer_generation_completion_tokens")) {
        add_hist_json_all(json["generation"], "completion_tokens", *family);
    }
    if (const Family* family = family_named("ninfer_generation_reasoning_tokens")) {
        add_hist_json_all(json["generation"], "reasoning_tokens", *family);
    }
    if (const Family* family = family_named("ninfer_generation_computed_prefill_tokens")) {
        add_hist_json_all(json["generation"], "computed_prefill_tokens", *family);
    }
    if (const Family* family = family_named("ninfer_speculative_live_k")) {
        add_hist_json_all(json["speculative"], "live_k", *family);
    }
    if (const Family* family = family_named("ninfer_recovery_attempts")) {
        add_hist_json_all(json["recovery"], "attempts", *family);
    }
    if (const Family* family = family_named("ninfer_generation_phase_seconds")) {
        for (const char* phase : kPhases) {
            const Sample* sample = find_labels(*family, "phase", phase);
            json["phases"][phase] = Json{{"count", sample == nullptr ? 0 : sample->count},
                                         {"sum", sample == nullptr ? 0.0 : sample->sum}};
        }
    }
    return json;
}

std::vector<Family> collect_families(const MetricsSnapshotData& data);

} // namespace

struct MetricsSnapshotData {
    Startup startup;
    ninfer::RuntimeStats stats;
    std::size_t http_in_flight   = 0;
    std::size_t response_records = 0;
    std::size_t response_bytes   = 0;
    std::map<std::string, std::uint64_t> counters;
    std::map<std::string, HistState> histograms;
};

struct ServeMetricsState {
    std::mutex mutex;
    Startup startup;
    std::map<std::string, std::uint64_t> counters;
    std::map<std::string, HistState> histograms;

    void precreate_counter(const char* family, Labels labels) {
        counters.emplace(series_key(family, labels), 0);
    }

    void precreate_histogram(const char* family, Labels labels) {
        histograms.emplace(series_key(family, labels), make_hist(bucket_kind(family)));
    }

    void inc(const char* family, Labels labels, std::uint64_t amount) {
        if (amount == 0) { return; }
        const auto it = counters.find(series_key(family, labels));
        if (it == counters.end()) { return; }
        it->second += amount;
    }

    void inc_open(const char* family, Labels labels, std::uint64_t amount) {
        if (amount == 0) { return; }
        counters[series_key(family, labels)] += amount;
    }

    void observe(const char* family, Labels labels, double value) {
        if (!std::isfinite(value)) { return; }
        const std::string key = series_key(family, labels);
        auto it = histograms.find(key);
        if (it == histograms.end()) {
            it = histograms.emplace(key, make_hist(bucket_kind(family))).first;
        }
        HistState& state = it->second;
        const BoundSet bounds = bounds_for(bucket_kind(family));
        std::size_t index = bounds.count;
        for (std::size_t i = 0; i < bounds.count; ++i) {
            if (value <= bounds.values[i]) {
                index = i;
                break;
            }
        }
        if (index < state.buckets.size()) { ++state.buckets[index]; }
        state.sum += value;
        ++state.count;
    }
};

namespace {

void precreate(ServeMetricsState& state) {
    for (const char* protocol : kProtocols) {
        for (const char* stream : kStreams) {
            for (const char* result : kResults) {
                for (const char* thinking : kBools) {
                    for (const char* tools : kBools) {
                        state.precreate_counter(
                            "ninfer_generation_requests_total",
                            Labels{{"protocol", protocol},
                                   {"result", result},
                                   {"stream", stream},
                                   {"thinking", thinking},
                                   {"tools", tools}});
                    }
                }
            }
        }
    }
    for (const char* protocol : kProtocols) {
        state.precreate_histogram("ninfer_generation_ttft_seconds", Labels{{"protocol", protocol}});
        state.precreate_histogram("ninfer_generation_e2e_seconds", Labels{{"protocol", protocol}});
        state.precreate_histogram("ninfer_generation_inter_token_latency_seconds",
                                  Labels{{"protocol", protocol}});
        state.precreate_counter("ninfer_token_count_requests_total", Labels{{"protocol", protocol}});
    }
    for (const char* phase : kPhases) {
        state.precreate_histogram("ninfer_generation_phase_seconds", Labels{{"phase", phase}});
    }
    for (const char* tier : kTiers) {
        for (const char* op : kCopyOps) {
            state.precreate_histogram("ninfer_generation_kv_copy_seconds",
                                      Labels{{"op", op}, {"tier", tier}});
        }
    }
    state.precreate_histogram("ninfer_generation_prompt_tokens", {});
    state.precreate_histogram("ninfer_generation_completion_tokens", {});
    state.precreate_histogram("ninfer_generation_reasoning_tokens", {});
    state.precreate_histogram("ninfer_generation_computed_prefill_tokens", {});
    state.precreate_histogram("ninfer_generation_output_tokens_per_second", {});
    state.precreate_histogram("ninfer_generation_prefill_tokens_per_second", {});
    state.precreate_histogram("ninfer_speculative_live_k", {});
    state.precreate_histogram("ninfer_recovery_attempts", {});
    for (const char* reason : kFinishReasons) {
        state.precreate_counter("ninfer_generation_finish_reason_total", Labels{{"reason", reason}});
    }
    state.precreate_counter("ninfer_generation_tool_calls_total", {});
    state.precreate_counter("ninfer_generation_ignored_tool_markup_total", {});
    state.precreate_counter("ninfer_generation_media_requests_total", {});
    for (const char* code : kApiErrorCodes) {
        state.precreate_counter("ninfer_api_errors_total", Labels{{"code", code}});
    }
    for (const char* path : kPrefixPaths) {
        for (const char* source : kPrefixSources) {
            state.precreate_counter("ninfer_prefix_reuse_requests_total",
                                    Labels{{"path", path}, {"source", source}});
        }
    }
    for (const char* source : kPrefixSources) {
        state.precreate_counter("ninfer_prefix_cache_hit_tokens_total", Labels{{"source", source}});
    }
    state.precreate_counter("ninfer_prefix_cache_query_tokens_total", {});
    state.precreate_counter("ninfer_context_checkpoint_restored_tokens_total", {});
    state.precreate_counter("ninfer_context_checkpoint_captured_tokens_total", {});
    state.precreate_counter("ninfer_context_checkpoint_capture_requests_total", {});
    state.precreate_counter("ninfer_speculative_rounds_total", {});
    state.precreate_counter("ninfer_speculative_draft_tokens_total", {});
    state.precreate_counter("ninfer_speculative_accepted_tokens_total", {});
    state.precreate_counter("ninfer_speculative_fallback_steps_total", {});
    for (int position = 0; position <= 14; ++position) {
        state.precreate_counter("ninfer_speculative_accepted_tokens_position_total",
                                Labels{{"position", std::to_string(position).c_str()}});
    }
    for (int k = 1; k <= 15; ++k) {
        state.precreate_counter("ninfer_speculative_rounds_by_k_total",
                                Labels{{"k", std::to_string(k).c_str()}});
    }
    state.precreate_counter("ninfer_recovery_cycle_exclusions_total", {});
    state.precreate_counter("ninfer_recovery_discarded_reasoning_tokens_total", {});
    state.precreate_counter("ninfer_recovery_discarded_tool_calls_total", {});
    for (const char* kind : kRecoveryKinds) {
        for (const char* cause : kRecoveryCauses) {
            state.precreate_counter("ninfer_recovery_events_total",
                                    Labels{{"cause", cause}, {"kind", kind}});
        }
    }
}

[[nodiscard]] const char* recovery_kind_name(ninfer::RecoveryEventKind kind) {
    switch (kind) {
    case ninfer::RecoveryEventKind::CycleExclusion:
        return "cycle_exclusion";
    case ninfer::RecoveryEventKind::RetryTriggered:
        return "retry_triggered";
    case ninfer::RecoveryEventKind::RetryStarted:
        return "retry_started";
    case ninfer::RecoveryEventKind::RetryPrefillComplete:
        return "retry_prefill_complete";
    case ninfer::RecoveryEventKind::Finished:
        return "finished";
    case ninfer::RecoveryEventKind::Exhausted:
        return "exhausted";
    }
    return "other";
}

void observe_success(ServeMetricsState& state, const GenerationObservation& observation) {
    const GenerationOutcome& outcome = *observation.outcome;
    const GenerationMetrics& metrics = outcome.metrics;
    const char* protocol = known_protocol(observation.protocol);
    const Labels protocol_labels{{"protocol", protocol}};
    if (std::isfinite(metrics.ttft_seconds)) {
        state.observe("ninfer_generation_ttft_seconds", protocol_labels, metrics.ttft_seconds);
    }
    if (std::isfinite(metrics.total_seconds)) {
        state.observe("ninfer_generation_e2e_seconds", protocol_labels, metrics.total_seconds);
    }
    const int decode_tokens =
        decode_eval_tokens(outcome.completion_tokens, metrics.recovery.prefill_samples);
    if (metrics.decode_seconds > 0.0 && decode_tokens > 0) {
        state.observe("ninfer_generation_inter_token_latency_seconds", protocol_labels,
                      metrics.decode_seconds / static_cast<double>(decode_tokens));
        state.observe("ninfer_generation_output_tokens_per_second", {},
                      static_cast<double>(decode_tokens) / metrics.decode_seconds);
    }
    const double phases[] = {
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
    for (std::size_t i = 0; i < std::size(kPhases); ++i) {
        if (phases[i] > 0.0) {
            state.observe("ninfer_generation_phase_seconds", Labels{{"phase", kPhases[i]}}, phases[i]);
        }
    }
    const double copies[][3] = {
        {metrics.kv_disk_h2d_seconds, metrics.kv_disk_load_seconds, metrics.kv_disk_save_seconds},
        {0.0, metrics.kv_ram_load_seconds, metrics.kv_ram_save_seconds},
    };
    for (std::size_t tier = 0; tier < 2; ++tier) {
        for (std::size_t op = 0; op < 3; ++op) {
            if (copies[tier][op] > 0.0) {
                state.observe("ninfer_generation_kv_copy_seconds",
                              Labels{{"op", kCopyOps[op]}, {"tier", kTiers[tier]}}, copies[tier][op]);
            }
        }
    }
    if (outcome.prompt_tokens >= 0) {
        state.observe("ninfer_generation_prompt_tokens", {}, static_cast<double>(outcome.prompt_tokens));
    }
    if (outcome.completion_tokens >= 0) {
        state.observe("ninfer_generation_completion_tokens", {},
                      static_cast<double>(outcome.completion_tokens));
    }
    if (outcome.reasoning_tokens > 0) {
        state.observe("ninfer_generation_reasoning_tokens", {},
                      static_cast<double>(outcome.reasoning_tokens));
    }
    const int computed =
        prefill_eval_tokens(outcome.prompt_tokens, static_cast<int>(metrics.prefix_cache_hit_tokens));
    if (computed >= 0) {
        state.observe("ninfer_generation_computed_prefill_tokens", {}, static_cast<double>(computed));
    }
    if (metrics.prefill_tail_tok_s > 0.0) {
        state.observe("ninfer_generation_prefill_tokens_per_second", {}, metrics.prefill_tail_tok_s);
    } else if (metrics.prefill_seconds > 0.0 && computed > 0) {
        state.observe("ninfer_generation_prefill_tokens_per_second", {},
                      static_cast<double>(computed) / metrics.prefill_seconds);
    }
    const char* reason = outcome.tool_calls.empty() ? finish_reason_name(outcome.finish_reason)
                                                     : "tool_calls";
    bool known_reason = false;
    for (const char* candidate : kFinishReasons) {
        if (std::string_view(reason) == candidate) { known_reason = true; }
    }
    state.inc("ninfer_generation_finish_reason_total",
              Labels{{"reason", known_reason ? reason : "unknown"}}, 1);
    state.inc("ninfer_generation_tool_calls_total", {}, outcome.tool_calls.size());
    state.inc("ninfer_generation_ignored_tool_markup_total", {},
              outcome.ignored_qwen_tool_call_names.size());
    state.inc("ninfer_prefix_reuse_requests_total",
              Labels{{"path", prefix_reuse_path_name(metrics.prefix_reuse_path)},
                     {"source", prefix_reuse_source_name(metrics.prefix_reuse_source)}},
              1);
    state.inc("ninfer_prefix_cache_hit_tokens_total",
              Labels{{"source", prefix_reuse_source_name(metrics.prefix_reuse_source)}},
              metrics.prefix_cache_hit_tokens);
    state.inc("ninfer_prefix_cache_query_tokens_total", {},
              static_cast<std::uint64_t>(std::max(outcome.prompt_tokens, 0)));
    state.inc("ninfer_context_checkpoint_restored_tokens_total", {},
              metrics.restored_context_checkpoint_tokens);
    state.inc("ninfer_context_checkpoint_captured_tokens_total", {},
              metrics.captured_context_checkpoint_tokens);
    if (metrics.speculative_backend == SpeculativeBackend::None) { return; }
    state.inc("ninfer_speculative_rounds_total", {}, metrics.speculative_rounds);
    state.inc("ninfer_speculative_draft_tokens_total", {}, metrics.speculative_draft_tokens);
    state.inc("ninfer_speculative_accepted_tokens_total", {}, metrics.speculative_accepted_tokens);
    state.inc("ninfer_speculative_fallback_steps_total", {}, metrics.speculative_fallback_steps);
    const std::size_t positions =
        std::min<std::size_t>(metrics.speculative_accepted_per_position.size(), 15);
    for (std::size_t position = 0; position < positions; ++position) {
        state.inc("ninfer_speculative_accepted_tokens_position_total",
                  Labels{{"position", std::to_string(position).c_str()}},
                  metrics.speculative_accepted_per_position[position]);
    }
    for (int k = 1; k <= 15; ++k) {
        if (static_cast<std::size_t>(k) >= metrics.speculative_rounds_per_draft.size()) { break; }
        state.inc("ninfer_speculative_rounds_by_k_total",
                  Labels{{"k", std::to_string(k).c_str()}},
                  metrics.speculative_rounds_per_draft[static_cast<std::size_t>(k)]);
    }
    if (metrics.speculative_live_draft_tokens > 0) {
        state.observe("ninfer_speculative_live_k", {}, metrics.speculative_live_draft_tokens);
    }
}

void observe_terminal_recovery(ServeMetricsState& state, const ninfer::GenerationRecoveryStats& stats) {
    state.inc("ninfer_recovery_cycle_exclusions_total", {}, stats.cycle_exclusions);
    state.inc("ninfer_recovery_discarded_reasoning_tokens_total", {}, stats.discarded_reasoning_tokens);
    state.inc("ninfer_recovery_discarded_tool_calls_total", {}, stats.discarded_tool_calls);
    if (stats.attempts > 0) {
        state.observe("ninfer_recovery_attempts", {}, static_cast<double>(stats.attempts));
    }
}

std::vector<Family> collect_families(const MetricsSnapshotData& data) {
    std::vector<Family> families;
    families.reserve(kMetricFamilyCount);
    const Startup& startup = data.startup;
    const ninfer::RuntimeStats& stats = data.stats;
    add_info(families, "ninfer_engine_info", "Resident engine identity.", startup.engine_labels);
    add_info(families, "ninfer_build_info", "CUDA build identity.", startup.build_labels);
    add_gauge(families, "ninfer_max_context_tokens", "Configured max context tokens.",
              startup.max_context);
    add_gauge(families, "ninfer_prefill_chunk_tokens", "Configured prefill chunk tokens.",
              startup.prefill_chunk);
    add_gauge(families, "ninfer_pending_timeout_seconds", "Pending FIFO timeout.",
              startup.pending_timeout_seconds);
    add_gauge(families, "ninfer_default_max_tokens", "Default max output tokens.",
              startup.default_max_tokens);
    add_gauge(families, "ninfer_keep_frac", "Sparse attention keep fraction.", startup.keep_frac);
    add_gauge(families, "ninfer_xattn_tau", "Sparse attention tau.", startup.xattn_tau);
    add_gauge(families, "ninfer_xattn_min_len", "Cross-attention minimum length.",
              startup.xattn_min_len);
    add_gauge(families, "ninfer_dflash_verify_width", "DFlash verify width. 0 is automatic.",
              startup.dflash_verify_width);
    add_gauge(families, "ninfer_speculative_configured_draft_tokens", "Configured draft tokens.",
              startup.configured_draft_tokens);
    add_gauge(families, "ninfer_engine_load_seconds", "Artifact load seconds.", startup.load_seconds);
    add_gauge(families, "ninfer_cuda_graph_allowance_bytes", "CUDA graph allowance at startup.",
              startup.cuda_graph_allowance_bytes);
    add_gauge(families, "ninfer_cuda_graph_observed_bytes", "CUDA graph bytes observed at startup.",
              startup.cuda_graph_observed_bytes);
    add_labeled_gauges(
        families, "ninfer_arena_capacity_bytes", "Arena capacity at startup.",
        {labeled_value({{"arena", "request_transient"}}, startup.arena_request_transient),
         labeled_value({{"arena", "sequence"}}, startup.arena_sequence),
         labeled_value({{"arena", "weights"}}, startup.arena_weights),
         labeled_value({{"arena", "workspace"}}, startup.arena_workspace)});
    add_gauge(families, "ninfer_server_start_time_seconds", "Server attach time as unix seconds.",
              startup.start_time_unix_s);
    add_gauge(families, "ninfer_scheduler_max_concurrency", "Configured resident requests.",
              startup.max_concurrency);
    add_gauge(families, "ninfer_scheduler_max_pending_requests", "Configured pending FIFO depth.",
              startup.max_pending_requests);
    add_mapped_counter(families, "ninfer_http_requests_total", "HTTP requests by route.",
                       data.counters);
    add_mapped_histogram(families, "ninfer_http_request_duration_seconds", "HTTP handler duration.",
                         BucketKind::Seconds, data.histograms);
    add_gauge(families, "ninfer_http_in_flight_requests", "Requests occupying a serve slot.",
              static_cast<double>(data.http_in_flight));
    add_mapped_counter(families, "ninfer_api_errors_total", "API errors by closed code.",
                       data.counters);
    add_mapped_counter(families, "ninfer_generation_requests_total", "HTTP generation attempts.",
                       data.counters);
    add_mapped_histogram(families, "ninfer_generation_ttft_seconds", "Time to first token.",
                         BucketKind::Seconds, data.histograms);
    add_mapped_histogram(families, "ninfer_generation_e2e_seconds", "Engine end to end seconds.",
                         BucketKind::Seconds, data.histograms);
    add_mapped_histogram(families, "ninfer_generation_inter_token_latency_seconds",
                         "Decode seconds per output token.", BucketKind::Seconds, data.histograms);
    add_mapped_histogram(families, "ninfer_generation_phase_seconds", "Closed generation phase seconds.",
                         BucketKind::Seconds, data.histograms);
    add_mapped_histogram(families, "ninfer_generation_kv_copy_seconds", "Per-request KV copy seconds.",
                         BucketKind::Seconds, data.histograms);
    add_mapped_histogram(families, "ninfer_generation_prompt_tokens", "Prompt tokens.",
                         BucketKind::Tokens, data.histograms);
    add_mapped_histogram(families, "ninfer_generation_completion_tokens", "Completion tokens.",
                         BucketKind::Tokens, data.histograms);
    add_mapped_histogram(families, "ninfer_generation_reasoning_tokens", "Reasoning tokens.",
                         BucketKind::Tokens, data.histograms);
    add_mapped_histogram(families, "ninfer_generation_computed_prefill_tokens",
                         "Prompt tokens excluding prefix hits.", BucketKind::Tokens, data.histograms);
    add_mapped_histogram(families, "ninfer_generation_output_tokens_per_second",
                         "Decode output tokens per second.", BucketKind::Toks, data.histograms);
    add_mapped_histogram(families, "ninfer_generation_prefill_tokens_per_second",
                         "Prefill tokens per second.", BucketKind::Toks, data.histograms);
    add_mapped_counter(families, "ninfer_generation_finish_reason_total", "Successful finish reasons.",
                       data.counters);
    add_mapped_counter(families, "ninfer_generation_tool_calls_total", "Returned tool calls.",
                       data.counters);
    add_mapped_counter(families, "ninfer_generation_ignored_tool_markup_total",
                       "Ignored undeclared tool markup.", data.counters);
    add_mapped_counter(families, "ninfer_generation_media_requests_total",
                       "Generations whose prompt had media.", data.counters);
    add_mapped_counter(families, "ninfer_token_count_requests_total", "Token-count requests.",
                       data.counters);
    add_gauge(families, "ninfer_scheduler_running_requests", "Requests with a lane.",
              stats.running_requests);
    add_gauge(families, "ninfer_scheduler_prefilling_requests", "Requests in prefill.",
              stats.prefilling_requests);
    add_gauge(families, "ninfer_scheduler_decode_ready_requests", "Requests waiting for decode.",
              stats.decode_ready_requests);
    add_gauge(families, "ninfer_scheduler_waiting_requests", "Requests in the pending FIFO.",
              stats.waiting_requests);
    add_absolute(families, "ninfer_engine_computed_prefill_tokens_total",
                 "Prompt tokens evaluated by prefill.",
                 static_cast<double>(stats.computed_prefill_tokens));
    add_absolute(families, "ninfer_engine_committed_decode_tokens_total",
                 "Tokens committed by decode rounds.",
                 static_cast<double>(stats.committed_decode_tokens));
    add_absolute(families, "ninfer_engine_decode_rounds_total", "Decode batch executions.",
                 static_cast<double>(stats.decode_rounds));
    add_absolute(families, "ninfer_engine_decode_row_rounds_total", "Decode rows across batches.",
                 static_cast<double>(stats.decode_row_rounds));
    add_labeled_gauges(
        families, "ninfer_gpu_kv_pages", "GPU KV pages by pool and state.",
        {labeled_value({{"pool", "main"}, {"state", "capacity"}}, stats.gpu_kv_main_capacity_pages),
         labeled_value({{"pool", "main"}, {"state", "entitled"}}, stats.gpu_kv_main_entitled_pages),
         labeled_value({{"pool", "main"}, {"state", "free"}}, stats.gpu_kv_main_free_pages),
         labeled_value({{"pool", "main"}, {"state", "mapped"}}, stats.gpu_kv_main_mapped_pages),
         labeled_value({{"pool", "spec"}, {"state", "capacity"}}, stats.gpu_kv_spec_capacity_pages),
         labeled_value({{"pool", "spec"}, {"state", "entitled"}}, stats.gpu_kv_spec_entitled_pages),
         labeled_value({{"pool", "spec"}, {"state", "free"}}, stats.gpu_kv_spec_free_pages),
         labeled_value({{"pool", "spec"}, {"state", "mapped"}}, stats.gpu_kv_spec_mapped_pages)});
    add_labeled_gauges(
        families, "ninfer_gpu_kv_capacity_tokens", "GPU KV token capacity by pool.",
        {labeled_value({{"pool", "main"}},
                       static_cast<double>(stats.gpu_kv_main_capacity_pages) * kKvPageTokens),
         labeled_value({{"pool", "spec"}},
                       static_cast<double>(stats.gpu_kv_spec_capacity_pages) * kKvPageTokens)});
    add_gauge(families, "ninfer_kv_ram_capacity_bytes", "Host RAM prefix cache capacity.",
              static_cast<double>(stats.kv_ram_capacity_bytes));
    add_gauge(families, "ninfer_kv_ram_used_bytes", "Host RAM prefix cache occupancy.",
              static_cast<double>(stats.kv_ram_used_bytes));
    add_gauge(families, "ninfer_kv_ram_entries", "Host RAM prefix cache entries.",
              static_cast<double>(stats.kv_ram_entry_count));
    add_absolute(families, "ninfer_kv_ram_captures_total", "Host RAM prefix captures.",
                 static_cast<double>(stats.kv_ram_captures));
    add_absolute(families, "ninfer_kv_ram_restores_total", "Host RAM prefix restores.",
                 static_cast<double>(stats.kv_ram_restores));
    add_absolute(families, "ninfer_kv_ram_evictions_total", "Host RAM prefix evictions.",
                 static_cast<double>(stats.kv_ram_evictions));
    add_absolute(families, "ninfer_kv_ram_drops_total", "Host RAM prefix drops.",
                 static_cast<double>(stats.kv_ram_drops));
    add_absolute(families, "ninfer_kv_ram_save_seconds_total", "Host RAM prefix save seconds.",
                 stats.kv_ram_save_seconds);
    add_absolute(families, "ninfer_kv_ram_load_seconds_total", "Host RAM prefix load seconds.",
                 stats.kv_ram_load_seconds);
    add_gauge(families, "ninfer_kv_disk_capacity_bytes", "Disk prefix cache capacity.",
              static_cast<double>(stats.kv_disk_capacity_bytes));
    add_gauge(families, "ninfer_kv_disk_used_bytes", "Disk prefix cache occupancy.",
              static_cast<double>(stats.kv_disk_used_bytes));
    add_gauge(families, "ninfer_kv_disk_entries", "Disk prefix cache entries.",
              static_cast<double>(stats.kv_disk_entry_count));
    add_absolute(families, "ninfer_kv_disk_captures_total", "Disk prefix captures.",
                 static_cast<double>(stats.kv_disk_captures));
    add_absolute(families, "ninfer_kv_disk_restores_total", "Disk prefix restores.",
                 static_cast<double>(stats.kv_disk_restores));
    add_absolute(families, "ninfer_kv_disk_evictions_total", "Disk prefix evictions.",
                 static_cast<double>(stats.kv_disk_evictions));
    add_absolute(families, "ninfer_kv_disk_drops_total", "Disk prefix drops.",
                 static_cast<double>(stats.kv_disk_drops));
    add_absolute(families, "ninfer_kv_disk_save_seconds_total", "Disk prefix save seconds.",
                 stats.kv_disk_save_seconds);
    add_absolute(families, "ninfer_kv_disk_load_seconds_total", "Disk prefix load seconds.",
                 stats.kv_disk_load_seconds);
    add_absolute(families, "ninfer_kv_disk_h2d_seconds_total", "Disk restore host to device seconds.",
                 stats.kv_disk_h2d_seconds);
    add_absolute(families, "ninfer_kv_cache_fallbacks_total",
                 "Prefix restores that fell back to cold prefill.",
                 static_cast<double>(stats.kv_cache_fallbacks));
    add_mapped_counter(families, "ninfer_prefix_reuse_requests_total", "Prefix reuse path and source.",
                       data.counters);
    add_mapped_counter(families, "ninfer_prefix_cache_hit_tokens_total", "Prefix hit tokens by source.",
                       data.counters);
    add_mapped_counter(families, "ninfer_prefix_cache_query_tokens_total", "Prefix query tokens.",
                       data.counters);
    add_mapped_counter(families, "ninfer_context_checkpoint_restored_tokens_total",
                       "Context checkpoint tokens restored.", data.counters);
    add_mapped_counter(families, "ninfer_context_checkpoint_captured_tokens_total",
                       "Context checkpoint tokens captured.", data.counters);
    add_mapped_counter(families, "ninfer_context_checkpoint_capture_requests_total",
                       "Requests that asked for a context checkpoint.", data.counters);
    add_mapped_counter(families, "ninfer_speculative_rounds_total", "Speculative rounds.",
                       data.counters);
    add_mapped_counter(families, "ninfer_speculative_draft_tokens_total", "Draft tokens proposed.",
                       data.counters);
    add_mapped_counter(families, "ninfer_speculative_accepted_tokens_total", "Draft tokens accepted.",
                       data.counters);
    add_mapped_counter(families, "ninfer_speculative_fallback_steps_total", "Speculative fallback steps.",
                       data.counters);
    add_mapped_counter(families, "ninfer_speculative_accepted_tokens_position_total",
                       "Accepted draft tokens by position.", data.counters);
    add_mapped_counter(families, "ninfer_speculative_rounds_by_k_total", "Speculative rounds by draft k.",
                       data.counters);
    add_mapped_histogram(families, "ninfer_speculative_live_k", "Live draft width.", BucketKind::Draft,
                         data.histograms);
    add_mapped_counter(families, "ninfer_recovery_events_total", "Published recovery events.",
                       data.counters);
    add_mapped_counter(families, "ninfer_recovery_cycle_exclusions_total",
                       "True cycle exclusions on terminal requests.", data.counters);
    add_mapped_counter(families, "ninfer_recovery_discarded_reasoning_tokens_total",
                       "Reasoning tokens discarded by recovery.", data.counters);
    add_mapped_counter(families, "ninfer_recovery_discarded_tool_calls_total",
                       "Tool calls discarded by recovery.", data.counters);
    add_mapped_histogram(families, "ninfer_recovery_attempts", "Recovery attempts on terminal requests.",
                         BucketKind::Draft, data.histograms);
    add_gauge(families, "ninfer_response_store_records", "Stored response records.",
              static_cast<double>(data.response_records));
    add_gauge(families, "ninfer_response_store_bytes", "Stored response bytes.",
              static_cast<double>(data.response_bytes));
    add_gauge(families, "ninfer_response_store_max_records", "Response store record cap.",
              startup.response_max_records);
    add_gauge(families, "ninfer_response_store_max_bytes", "Response store byte cap.",
              startup.response_max_bytes);
    if (families.size() != kMetricFamilyCount) {
        throw std::logic_error("metric family count drifted from kMetricFamilies");
    }
    for (std::size_t i = 0; i < families.size(); ++i) {
        if (families[i].name != kMetricFamilies[i]) {
            throw std::logic_error("metric family order drifted at " + families[i].name);
        }
    }
    return families;
}

} // namespace

MetricsSnapshot::MetricsSnapshot() = default;
MetricsSnapshot::~MetricsSnapshot() = default;
MetricsSnapshot::MetricsSnapshot(const MetricsSnapshot& other) {
    if (other.data_) { data_ = std::make_unique<MetricsSnapshotData>(*other.data_); }
}
MetricsSnapshot& MetricsSnapshot::operator=(const MetricsSnapshot& other) {
    if (this != &other) {
        data_ = other.data_ ? std::make_unique<MetricsSnapshotData>(*other.data_) : nullptr;
    }
    return *this;
}
MetricsSnapshot::MetricsSnapshot(MetricsSnapshot&&) noexcept = default;
MetricsSnapshot& MetricsSnapshot::operator=(MetricsSnapshot&&) noexcept = default;

ServeMetrics::ServeMetrics() : state_(std::make_unique<ServeMetricsState>()) { precreate(*state_); }
ServeMetrics::~ServeMetrics() = default;

void ServeMetrics::attach(const ServeOptions& options, const ninfer::LoadSummary& load,
                          const ninfer::MemorySummary& memory, std::string model_id,
                          std::string cuda_compile_version, std::string cuda_runtime_version) {
    Startup startup;
    startup.max_context             = options.max_context;
    startup.prefill_chunk           = options.prefill_chunk;
    startup.pending_timeout_seconds = static_cast<double>(options.pending_timeout_ms) / 1000.0;
    startup.default_max_tokens      = options.default_max_tokens;
    startup.keep_frac               = options.keep_frac;
    startup.xattn_tau               = options.xattn_tau;
    startup.xattn_min_len           = static_cast<double>(kDefaultXattnMinLen);
    startup.dflash_verify_width     = options.speculative.dflash_verify_width;
    startup.configured_draft_tokens = options.speculative.draft_tokens;
    startup.load_seconds            = load.load_seconds;
    startup.cuda_graph_allowance_bytes =
        static_cast<double>(memory.cuda_graph_allowance_bytes);
    startup.cuda_graph_observed_bytes = static_cast<double>(memory.cuda_graph_observed_bytes);
    startup.arena_weights             = static_cast<double>(memory.weights.capacity_bytes);
    startup.arena_sequence            = static_cast<double>(memory.sequence.capacity_bytes);
    startup.arena_workspace           = static_cast<double>(memory.workspace.capacity_bytes);
    startup.arena_request_transient   = static_cast<double>(memory.request_transient.capacity_bytes);
    startup.start_time_unix_s =
        std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
    startup.max_concurrency      = options.max_concurrency;
    startup.max_pending_requests = options.max_pending_requests;
    startup.response_max_records = static_cast<double>(options.response_store_max_records);
    startup.response_max_bytes   = static_cast<double>(options.response_store_max_bytes);
    const char* checkpoints = "default";
    if (options.context_checkpoint_marks.has_value()) {
        checkpoints = options.context_checkpoint_marks->empty() ? "off" : "custom";
    }
    startup.engine_labels = Labels{
        {"adaptive_draft", yn(options.speculative.adaptive_draft)},
        {"auth", yn(!options.api_key.empty())},
        {"context_checkpoints", checkpoints},
        {"cors", yn(options.enable_cors)},
        {"cuda_graph", yn(options.use_cuda_graph)},
        {"device", std::to_string(options.device).c_str()},
        {"generation_recovery", yn(options.generation_recovery)},
        {"greedy", yn(options.greedy)},
        {"kv_capacity_mode", kv_capacity_mode_name(options.kv_capacity.mode)},
        {"kv_disk_compress", options.kv_disk_compress == ninfer::KvDiskCompress::Zstd ? "zstd" : "off"},
        {"kv_dtype", kv_cache_name(options.kv_cache)},
        {"model_id", model_id.c_str()},
        {"p_less", yn(options.sampling_overrides.p_less)},
        {"prefix_reuse", yn(options.allow_prefix_reuse)},
        {"proposal_head", proposal_head_name(options.speculative.proposal_head)},
        {"sage_attn", yn(options.sage_attn)},
        {"spec", product::speculative_backend_name(options.speculative.backend)},
        {"system_prepend", yn(!options.system_prepend.empty())},
        {"target", load.target.c_str()},
        {"vision", yn(options.enable_vision)},
        {"weights_id", load.weights_id.c_str()},
    };
    startup.build_labels = Labels{{"cuda_compile_version", cuda_compile_version.c_str()},
                                  {"cuda_runtime_version", cuda_runtime_version.c_str()}};
    std::lock_guard lock(state_->mutex);
    state_->startup = std::move(startup);
}

void observe_generation(ServeMetrics& metrics, const GenerationObservation& observation) {
    std::lock_guard lock(metrics.state_->mutex);
    ServeMetricsState& state = *metrics.state_;
    const char* result = "error";
    for (const char* known : kResults) {
        if (observation.result == known) { result = known; }
    }
    state.inc("ninfer_generation_requests_total",
              Labels{{"protocol", known_protocol(observation.protocol)},
                     {"result", result},
                     {"stream", yn(observation.stream)},
                     {"thinking", yn(observation.thinking)},
                     {"tools", yn(observation.tools)}},
              1);
    if (observation.capture_requested) {
        state.inc("ninfer_context_checkpoint_capture_requests_total", {}, 1);
    }
    if (observation.has_media &&
        (std::string_view(result) == "success" || std::string_view(result) == "rejected")) {
        state.inc("ninfer_generation_media_requests_total", {}, 1);
    }
    const bool terminal = std::string_view(result) == "success" || std::string_view(result) == "error" ||
                          std::string_view(result) == "cancelled";
    if (terminal) {
        if (observation.outcome != nullptr) {
            observe_terminal_recovery(state, observation.outcome->metrics.recovery);
        } else if (observation.recovery != nullptr) {
            observe_terminal_recovery(state, *observation.recovery);
        }
    }
    if (std::string_view(result) == "success" && observation.outcome != nullptr) {
        observe_success(state, observation);
    }
}

void observe_recovery_event(ServeMetrics& metrics, const ninfer::RecoveryEvent& event) {
    std::lock_guard lock(metrics.state_->mutex);
    const char* kind = recovery_kind_name(event.kind);
    const char* cause = prometheus_recovery_cause(kind, event.cause);
    metrics.state_->inc("ninfer_recovery_events_total", Labels{{"cause", cause}, {"kind", kind}}, 1);
}

void observe_http(ServeMetrics& metrics, std::string_view protocol, std::string_view route,
                  std::string_view method, int status, double seconds) {
    std::string status_text = (status >= 100 && status <= 599) ? std::to_string(status) : "other";
    Labels labels{{"method", known_method(method)},
                  {"protocol", protocol.empty() ? "other" : std::string(protocol).c_str()},
                  {"route", route.empty() ? "other" : std::string(route).c_str()},
                  {"status", status_text.c_str()}};
    std::lock_guard lock(metrics.state_->mutex);
    metrics.state_->inc_open("ninfer_http_requests_total", labels, 1);
    if (std::isfinite(seconds) && seconds >= 0.0) {
        Labels duration{{"protocol", protocol.empty() ? "other" : std::string(protocol).c_str()},
                        {"route", route.empty() ? "other" : std::string(route).c_str()}};
        metrics.state_->observe("ninfer_http_request_duration_seconds", std::move(duration), seconds);
    }
}

void observe_api_error(ServeMetrics& metrics, std::string_view code) {
    std::lock_guard lock(metrics.state_->mutex);
    metrics.state_->inc("ninfer_api_errors_total", Labels{{"code", known_api_code(code)}}, 1);
}

void observe_token_count(ServeMetrics& metrics, std::string_view protocol) {
    std::lock_guard lock(metrics.state_->mutex);
    bool known = false;
    for (const char* candidate : kProtocols) {
        if (protocol == candidate) { known = true; }
    }
    if (!known) { return; }
    metrics.state_->inc("ninfer_token_count_requests_total",
                        Labels{{"protocol", std::string(protocol).c_str()}}, 1);
}

MetricsSnapshot fill_metrics_snapshot(ServeMetrics& metrics, const ScrapeInputs& inputs) {
    auto data = std::make_unique<MetricsSnapshotData>();
    {
        std::lock_guard lock(metrics.state_->mutex);
        data->startup    = metrics.state_->startup;
        data->counters   = metrics.state_->counters;
        data->histograms = metrics.state_->histograms;
    }
    data->stats            = inputs.stats;
    data->http_in_flight   = inputs.http_in_flight;
    data->response_records = inputs.response_records;
    data->response_bytes   = inputs.response_bytes;
    MetricsSnapshot snapshot;
    snapshot.data_ = std::move(data);
    return snapshot;
}

std::string render_prometheus_text(const MetricsSnapshot& snapshot) {
    if (!snapshot.data_) { return {}; }
    return render_families(collect_families(*snapshot.data_));
}

std::string render_metrics_json(const MetricsSnapshot& snapshot) {
    if (!snapshot.data_) { return "{}"; }
    return render_json(collect_families(*snapshot.data_)).dump();
}

void handle_metrics(const MetricsSnapshot& snapshot, httplib::Response& response) {
    response.status = 200;
    response.set_content(render_prometheus_text(snapshot), kPrometheusContentType);
}

void handle_metrics_json(const MetricsSnapshot& snapshot, httplib::Response& response) {
    response.status = 200;
    response.set_content(render_metrics_json(snapshot), "application/json");
}

HttpRouteClass classify_http_route(std::string_view path) {
    if (path == "/health") { return {"health", "/health"}; }
    if (path == "/metrics") { return {"metrics", "/metrics"}; }
    if (path == "/metrics.json") { return {"metrics", "/metrics.json"}; }
    if (path == "/v1/models") { return {"models", "/v1/models"}; }
    if (path == "/v1/chat/completions") { return {"openai_chat", "/v1/chat/completions"}; }
    if (path == "/v1/responses") { return {"openai_responses", "/v1/responses"}; }
    if (path == "/v1/responses/input_tokens") {
        return {"openai_responses", "/v1/responses/input_tokens"};
    }
    if (path == "/v1/responses/compact") { return {"openai_responses", "/v1/responses/compact"}; }
    if (path == "/v1/messages") { return {"anthropic_messages", "/v1/messages"}; }
    if (path == "/v1/messages/count_tokens") {
        return {"anthropic_messages", "/v1/messages/count_tokens"};
    }
    constexpr std::string_view kModelPrefix = "/v1/models/";
    if (path.starts_with(kModelPrefix) && path.size() > kModelPrefix.size() &&
        path.find('/', kModelPrefix.size()) == std::string_view::npos) {
        return {"models", "/v1/models/{id}"};
    }
    constexpr std::string_view kResponsePrefix = "/v1/responses/";
    if (path.starts_with(kResponsePrefix) && path.size() > kResponsePrefix.size()) {
        const std::string_view rest = path.substr(kResponsePrefix.size());
        constexpr std::string_view kItems = "/input_items";
        constexpr std::string_view kCancel = "/cancel";
        if (rest.size() > kItems.size() && rest.ends_with(kItems) &&
            rest.find('/') == rest.size() - kItems.size()) {
            return {"openai_responses", "/v1/responses/{id}/input_items"};
        }
        if (rest.size() > kCancel.size() && rest.ends_with(kCancel) &&
            rest.find('/') == rest.size() - kCancel.size()) {
            return {"openai_responses", "/v1/responses/{id}/cancel"};
        }
        if (rest.find('/') == std::string_view::npos) {
            return {"openai_responses", "/v1/responses/{id}"};
        }
    }
    return {"other", "other"};
}

bool is_unauthenticated_path(std::string_view path) {
    return path == "/health" || path == "/metrics" || path == "/metrics.json";
}

const char* prometheus_protocol(std::string_view request_log_protocol) {
    if (request_log_protocol == "openai_chat_completions" || request_log_protocol == "openai_chat") {
        return "openai_chat";
    }
    if (request_log_protocol == "openai_responses") { return "openai_responses"; }
    if (request_log_protocol == "anthropic_messages") { return "anthropic_messages"; }
    return "other";
}

const char* prometheus_recovery_cause(std::string_view kind, std::string_view cause) {
    if (cause == "reasoning_cycle" || cause == "repeated_reasoning" || cause == "duplicate_tool_call" ||
        cause == "cancelled" || cause == "tool_calls" || cause == "output_limit" ||
        cause == "context_capacity" || cause == "stop") {
        if (cause == "reasoning_cycle") { return "reasoning_cycle"; }
        if (cause == "repeated_reasoning") { return "repeated_reasoning"; }
        if (cause == "duplicate_tool_call") { return "duplicate_tool_call"; }
        if (cause == "cancelled") { return "cancelled"; }
        if (cause == "tool_calls") { return "tool_calls"; }
        if (cause == "output_limit") { return "output_limit"; }
        if (cause == "context_capacity") { return "context_capacity"; }
        return "stop";
    }
    if (kind == "exhausted") {
        if (contains_text(cause, "reservation")) { return "reservation_exceeded"; }
        if (contains_text(cause, "output-token budget") || contains_text(cause, "retry or output-token") ||
            contains_text(cause, "no retry")) {
            return "retry_budget";
        }
        if (contains_text(cause, "reserved lane cannot be rebuilt safely")) { return "lane_rebuild"; }
    }
    return "other";
}

} // namespace ninfer::serve
