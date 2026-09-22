#include "serve/metrics.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <locale>
#include <set>
#include <string>
#include <thread>

namespace {

using ninfer::serve::GenerationObservation;
using ninfer::serve::GenerationOutcome;
using ninfer::serve::ScrapeInputs;
using ninfer::serve::ServeMetrics;
using ninfer::serve::kMetricFamilies;
using ninfer::serve::kMetricFamilyCount;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

bool contains(const std::string& text, const std::string& needle) {
    return text.find(needle) != std::string::npos;
}

std::string between(const std::string& text, const std::string& start, const std::string& end) {
    const auto at = text.find(start);
    if (at == std::string::npos) { return {}; }
    const auto from = at + start.size();
    const auto stop = text.find(end, from);
    if (stop == std::string::npos) { return {}; }
    return text.substr(from, stop - from);
}

struct CommaDecimal : std::numpunct<char> {
    char do_decimal_point() const override { return ','; }
    char do_thousands_sep() const override { return '.'; }
    std::string do_grouping() const override { return "\3"; }
};

GenerationOutcome success_outcome() {
    GenerationOutcome outcome;
    outcome.prompt_tokens     = 128;
    outcome.completion_tokens = 9;
    outcome.reasoning_tokens  = 0;
    outcome.finish_reason     = ninfer::FinishReason::StopToken;
    outcome.tool_calls.resize(2);
    outcome.metrics.speculative_backend          = ninfer::SpeculativeBackend::Mtp;
    outcome.metrics.speculative_accepted_tokens  = 6;
    outcome.metrics.speculative_draft_tokens     = 12;
    outcome.metrics.speculative_rounds           = 3;
    outcome.metrics.prefix_cache_hit_tokens      = 40;
    outcome.metrics.prefix_reuse_path            = ninfer::PrefixReusePath::AppendAtFrontier;
    outcome.metrics.prefix_reuse_source          = ninfer::PrefixReuseSource::HostRam;
    outcome.metrics.recovery.cycle_exclusions    = 4;
    outcome.metrics.queued_seconds               = 0.2;
    outcome.metrics.copy_hold_seconds            = 0.05;
    outcome.metrics.decode_seconds               = 1.0;
    outcome.metrics.ttft_seconds                 = 0.3;
    outcome.metrics.total_seconds                = 1.4;
    outcome.metrics.prefill_seconds              = 0.1;
    return outcome;
}

} // namespace

int main() {
    int failures = 0;
    ServeMetrics metrics;
    const std::locale classic = std::locale::classic();
    std::locale::global(std::locale(classic, new CommaDecimal));

    ninfer::LoadSummary load;
    load.target     = "qwen3.8-27b";
    load.model_id   = "id \"quoted\"\nline";
    load.weights_id = "weights";
    ninfer::serve::ServeOptions options;
    options.speculative.backend = ninfer::SpeculativeBackend::Mtp;
    metrics.attach(options, load, {}, "id \"quoted\"\nline", "13.1", "13.1");

    ScrapeInputs inputs;
    inputs.stats.kv_ram_used_bytes = 4096;
    auto snapshot = ninfer::serve::fill_metrics_snapshot(metrics, inputs);
    const std::string text = ninfer::serve::render_prometheus_text(snapshot);

    failures += check(text.find("# HELP ninfer_engine_info ") != std::string::npos &&
                          text.find("# HELP ninfer_engine_info ") < text.find("# TYPE ninfer_engine_info "),
                      "HELP does not precede TYPE");
    failures += check(contains(text, "le=\"+Inf\""), "histogram is missing +Inf");
    failures += check(contains(text, "le=\"0.005\""), "histogram bound is not locale-independent");
    failures += check(contains(text, "model_id=\"id \\\"quoted\\\"\\nline\""),
                      "label escaping dropped quote or newline");
    failures += check(text.find(',') == std::string::npos || contains(text, "le=\"0.005\""),
                      "locale changed a histogram bound");
    const auto sum_at = text.find("ninfer_generation_phase_seconds_sum");
    failures += check(sum_at != std::string::npos && text.find(',', sum_at) != sum_at,
                      "phase histogram sum is missing");

    inputs.stats.kv_cache_fallbacks = 10;
    const std::string first = ninfer::serve::render_prometheus_text(
        ninfer::serve::fill_metrics_snapshot(metrics, inputs));
    inputs.stats.kv_cache_fallbacks = 20;
    const std::string second = ninfer::serve::render_prometheus_text(
        ninfer::serve::fill_metrics_snapshot(metrics, inputs));
    failures += check(contains(second, "ninfer_kv_cache_fallbacks_total 20\n") &&
                          !contains(second, "ninfer_kv_cache_fallbacks_total 30"),
                      "absolute counter accumulated across scrapes");
    failures += check(contains(first, "ninfer_kv_cache_fallbacks_total 10\n"),
                      "first absolute counter scrape lost its value");

    std::set<std::string> rendered;
    std::string::size_type cursor = 0;
    while ((cursor = text.find("# TYPE ", cursor)) != std::string::npos) {
        cursor += 7;
        const auto end = text.find(' ', cursor);
        rendered.insert(text.substr(cursor, end - cursor));
    }
    std::set<std::string> catalog;
    for (std::size_t i = 0; i < kMetricFamilyCount; ++i) { catalog.emplace(kMetricFamilies[i]); }
    failures += check(rendered == catalog, "rendered families do not match kMetricFamilies");

    ServeMetrics fresh_metrics;
    const std::string fresh = ninfer::serve::render_prometheus_text(
        ninfer::serve::fill_metrics_snapshot(fresh_metrics, {}));
    for (const char* phase : {"prepare_cpu", "media_wait", "media_fetch", "queue", "copy_hold",
                              "vision", "prefill", "decode", "recovery", "http_tail"}) {
        failures += check(contains(fresh, std::string("phase=\"") + phase + "\""),
                          "fresh render is missing a phase");
    }
    failures += check(contains(fresh, "path=\"append_frontier\"") && contains(fresh, "source=\"host_ram\"") &&
                          contains(fresh, "kind=\"cycle_exclusion\"") &&
                          contains(fresh, "reason=\"tool_calls\"") &&
                          contains(fresh, "code=\"invalid_api_key\"") &&
                          contains(fresh, "position=\"0\"") && contains(fresh, "position=\"14\"") &&
                          contains(fresh, "k=\"1\"}") && contains(fresh, "k=\"15\"}"),
                      "fresh render is missing a closed label");

    GenerationOutcome outcome = success_outcome();
    outcome.metrics.speculative_accepted_per_position = {1, 2};
    outcome.metrics.speculative_rounds_per_draft      = {0, 3};
    outcome.metrics.speculative_live_draft_tokens     = 4;
    GenerationObservation observation;
    observation.protocol = "openai_chat";
    observation.result   = "success";
    observation.outcome  = &outcome;
    ninfer::serve::observe_generation(metrics, observation);
    const std::string observed = ninfer::serve::render_prometheus_text(
        ninfer::serve::fill_metrics_snapshot(metrics, inputs));
    failures += check(contains(observed, "ninfer_speculative_accepted_tokens_total 6\n"),
                      "accepted tokens were not recorded");
    failures += check(contains(observed, "ninfer_prefix_cache_hit_tokens_total{source=\"host_ram\"} 40\n"),
                      "prefix hit tokens were not recorded");
    failures += check(contains(observed, "ninfer_recovery_cycle_exclusions_total 4\n"),
                      "cycle exclusions were logged instead of counted");
    failures += check(contains(observed, "ninfer_generation_tool_calls_total 2\n"),
                      "tool calls were not recorded");
    failures += check(contains(observed, "ninfer_generation_phase_seconds_count{phase=\"queue\"} 1\n") &&
                          contains(observed, "ninfer_generation_phase_seconds_count{phase=\"copy_hold\"} 1\n") &&
                          contains(observed, "ninfer_generation_phase_seconds_count{phase=\"decode\"} 1\n"),
                      "phase histogram did not observe queue copy_hold and decode");
    const std::string decode_le_1 = between(
        observed, "ninfer_generation_phase_seconds_bucket{le=\"0.5\",phase=\"decode\"} ", "\n");
    const std::string decode_le_2 = between(
        observed, "ninfer_generation_phase_seconds_bucket{le=\"2.5\",phase=\"decode\"} ", "\n");
    failures += check(decode_le_1 == "0" && decode_le_2 == "1",
                      "histogram buckets are not cumulative");

    for (const std::uint32_t exclusions : {1U, 2U, 4U}) {
        ninfer::RecoveryEvent event;
        event.kind             = ninfer::RecoveryEventKind::CycleExclusion;
        event.cause            = "reasoning_cycle";
        event.cycle_exclusions = exclusions;
        ninfer::serve::observe_recovery_event(metrics, event);
    }
    GenerationObservation terminal;
    terminal.result   = "error";
    terminal.recovery = &outcome.metrics.recovery;
    ninfer::serve::observe_generation(metrics, terminal);
    const std::string recovery = ninfer::serve::render_prometheus_text(
        ninfer::serve::fill_metrics_snapshot(metrics, inputs));
    failures += check(
        contains(recovery,
                 "ninfer_recovery_events_total{cause=\"reasoning_cycle\",kind=\"cycle_exclusion\"} 3\n"),
        "cycle exclusion events were not counted once each");
    failures += check(contains(recovery, "ninfer_recovery_cycle_exclusions_total 8\n"),
                      "terminal cycle exclusions did not add the true total");

    failures += check(
        std::string(ninfer::serve::prometheus_recovery_cause(
            "exhausted", "prefix reservation exceeded")) == "reservation_exceeded" &&
            std::string(ninfer::serve::prometheus_recovery_cause(
                "exhausted", "output-token budget exhausted")) == "retry_budget" &&
            std::string(ninfer::serve::prometheus_recovery_cause(
                "exhausted", "no retry remains")) == "retry_budget" &&
            std::string(ninfer::serve::prometheus_recovery_cause(
                "exhausted", "reserved lane cannot be rebuilt safely")) == "lane_rebuild" &&
            std::string(ninfer::serve::prometheus_recovery_cause(
                "exhausted", "repaired request cannot preserve its admitted output budget")) ==
                "other" &&
            std::string(ninfer::serve::prometheus_recovery_cause("exhausted", "garbage")) == "other",
        "exhausted causes did not map to the closed set");

    GenerationOutcome skipped;
    skipped.prompt_tokens          = 32;
    skipped.completion_tokens      = 1;
    skipped.metrics.decode_seconds = 0;
    skipped.metrics.prefill_seconds = 0;
    skipped.metrics.prefix_cache_hit_tokens = 32;
    GenerationObservation skip;
    skip.result  = "success";
    skip.outcome = &skipped;
    ServeMetrics skip_metrics;
    ninfer::serve::observe_generation(skip_metrics, skip);
    const std::string skipped_text = ninfer::serve::render_prometheus_text(
        ninfer::serve::fill_metrics_snapshot(skip_metrics, {}));
    failures += check(contains(skipped_text, "ninfer_generation_output_tokens_per_second_count 0\n") &&
                          contains(skipped_text, "ninfer_generation_inter_token_latency_seconds_count{protocol=\"openai_chat\"} 0\n") &&
                          contains(skipped_text, "ninfer_generation_phase_seconds_count{phase=\"prefill\"} 0\n"),
                      "zero decode or a full prefix hit observed a sample");

    const std::string json_text = ninfer::serve::render_metrics_json(
        ninfer::serve::fill_metrics_snapshot(metrics, inputs));
    const auto json = nlohmann::json::parse(json_text);
    for (const char* key : {"engine", "build", "scheduler", "http", "gpu_kv", "kv_ram", "kv_disk",
                            "prefix_reuse", "speculative", "recovery", "generation", "phases",
                            "response_store", "start_time_unix_s"}) {
        failures += check(json.contains(key), "metrics json is missing a required key");
    }
    failures += check(json.at("kv_ram").at("used_bytes") == 4096, "kv_ram.used_bytes drifted from the snapshot");
    failures += check(json.at("phases").at("decode").at("count") == 1, "phases.decode.count drifted from the fixture");

    auto routed = ninfer::serve::classify_http_route("/v1/responses/abc/cancel");
    failures += check(std::string(routed.route) == "/v1/responses/{id}/cancel",
                      "response cancel was not classified before the id route");
    failures += check(ninfer::serve::is_unauthenticated_path("/metrics") &&
                          ninfer::serve::is_unauthenticated_path("/metrics.json") &&
                          ninfer::serve::is_unauthenticated_path("/health") &&
                          !ninfer::serve::is_unauthenticated_path("/v1/chat/completions"),
                      "metrics paths are not on the unauthenticated allowlist");

    httplib::Server server;
    server.Get("/metrics", [&](const httplib::Request&, httplib::Response& response) {
        ninfer::serve::handle_metrics(snapshot, response);
    });
    server.Get("/metrics.json", [&](const httplib::Request&, httplib::Response& response) {
        ninfer::serve::handle_metrics_json(snapshot, response);
    });
    const int port = server.bind_to_any_port("127.0.0.1");
    failures += check(port > 0, "metrics test server did not bind");
    std::thread thread([&] { server.listen_after_bind(); });
    httplib::Client client("127.0.0.1", port);
    client.set_connection_timeout(std::chrono::seconds(2));
    client.set_read_timeout(std::chrono::seconds(2));
    if (const auto response = client.Get("/metrics")) {
        const auto type = response->get_header_value("Content-Type");
        failures += check(type == ninfer::serve::kPrometheusContentType &&
                              response->body.find("ninfer_engine_info") != std::string::npos,
                          "GET /metrics did not return Prometheus text");
    } else {
        failures += check(false, "GET /metrics failed");
    }
    if (const auto response = client.Get("/metrics.json")) {
        failures += check(nlohmann::json::parse(response->body).contains("engine"),
                          "GET /metrics.json did not return the snapshot");
    } else {
        failures += check(false, "GET /metrics.json failed");
    }
    server.stop();
    thread.join();
    std::locale::global(classic);
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
