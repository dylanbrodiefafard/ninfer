#pragma once

// Prometheus text and JSON snapshots for ninfer-serve. HttpServer owns one
// ServeMetrics. The GPU executor never touches it. Every label set is closed
// and stored in fixed-index arrays, so an observation is a mutex-held array
// update and a scrape copies one flat value. fill_metrics_snapshot is the only
// reader of RuntimeStats into a snapshot; both encodings render that snapshot
// and do not sample the engine again.

#include "serve/generation_service.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace httplib {
struct Response;
}

namespace ninfer::serve {

inline constexpr const char* kPrometheusContentType =
    "text/plain; version=0.0.4; charset=utf-8";

// Rendered family inventory, in exposition order.
inline constexpr std::string_view kMetricFamilies[] = {
    "ninfer_engine_info",
    "ninfer_build_info",
    "ninfer_max_context_tokens",
    "ninfer_prefill_chunk_tokens",
    "ninfer_pending_timeout_seconds",
    "ninfer_default_max_tokens",
    "ninfer_keep_frac",
    "ninfer_xattn_tau",
    "ninfer_xattn_min_len",
    "ninfer_dflash_verify_width",
    "ninfer_speculative_configured_draft_tokens",
    "ninfer_engine_load_seconds",
    "ninfer_cuda_graph_allowance_bytes",
    "ninfer_cuda_graph_observed_bytes",
    "ninfer_arena_capacity_bytes",
    "ninfer_server_start_time_seconds",
    "ninfer_scheduler_max_concurrency",
    "ninfer_scheduler_max_pending_requests",
    "ninfer_http_requests_total",
    "ninfer_http_request_duration_seconds",
    "ninfer_http_in_flight_requests",
    "ninfer_api_errors_total",
    "ninfer_generation_requests_total",
    "ninfer_generation_ttft_seconds",
    "ninfer_generation_e2e_seconds",
    "ninfer_generation_inter_token_latency_seconds",
    "ninfer_generation_phase_seconds",
    "ninfer_generation_kv_copy_seconds",
    "ninfer_generation_prompt_tokens",
    "ninfer_generation_completion_tokens",
    "ninfer_generation_reasoning_tokens",
    "ninfer_generation_computed_prefill_tokens",
    "ninfer_generation_output_tokens_per_second",
    "ninfer_generation_prefill_tokens_per_second",
    "ninfer_generation_finish_reason_total",
    "ninfer_generation_tool_calls_total",
    "ninfer_generation_ignored_tool_markup_total",
    "ninfer_generation_media_requests_total",
    "ninfer_token_count_requests_total",
    "ninfer_scheduler_running_requests",
    "ninfer_scheduler_prefilling_requests",
    "ninfer_scheduler_decode_ready_requests",
    "ninfer_scheduler_waiting_requests",
    "ninfer_engine_computed_prefill_tokens_total",
    "ninfer_engine_committed_decode_tokens_total",
    "ninfer_engine_decode_rounds_total",
    "ninfer_engine_decode_row_rounds_total",
    "ninfer_gpu_kv_pages",
    "ninfer_gpu_kv_capacity_tokens",
    "ninfer_kv_ram_capacity_bytes",
    "ninfer_kv_ram_used_bytes",
    "ninfer_kv_ram_entries",
    "ninfer_kv_ram_captures_total",
    "ninfer_kv_ram_restores_total",
    "ninfer_kv_ram_evictions_total",
    "ninfer_kv_ram_drops_total",
    "ninfer_kv_ram_save_seconds_total",
    "ninfer_kv_ram_load_seconds_total",
    "ninfer_kv_disk_capacity_bytes",
    "ninfer_kv_disk_used_bytes",
    "ninfer_kv_disk_entries",
    "ninfer_kv_disk_captures_total",
    "ninfer_kv_disk_restores_total",
    "ninfer_kv_disk_evictions_total",
    "ninfer_kv_disk_drops_total",
    "ninfer_kv_disk_save_seconds_total",
    "ninfer_kv_disk_load_seconds_total",
    "ninfer_kv_disk_h2d_seconds_total",
    "ninfer_kv_cache_fallbacks_total",
    "ninfer_prefix_reuse_requests_total",
    "ninfer_prefix_cache_hit_tokens_total",
    "ninfer_prefix_cache_query_tokens_total",
    "ninfer_context_checkpoint_restored_tokens_total",
    "ninfer_context_checkpoint_captured_tokens_total",
    "ninfer_context_checkpoint_capture_requests_total",
    "ninfer_speculative_rounds_total",
    "ninfer_speculative_draft_tokens_total",
    "ninfer_speculative_accepted_tokens_total",
    "ninfer_speculative_fallback_steps_total",
    "ninfer_speculative_accepted_tokens_position_total",
    "ninfer_speculative_rounds_by_k_total",
    "ninfer_speculative_live_k",
    "ninfer_recovery_events_total",
    "ninfer_recovery_cycle_exclusions_total",
    "ninfer_recovery_discarded_reasoning_tokens_total",
    "ninfer_recovery_discarded_tool_calls_total",
    "ninfer_recovery_attempts",
    "ninfer_response_store_records",
    "ninfer_response_store_bytes",
    "ninfer_response_store_max_records",
    "ninfer_response_store_max_bytes",
};

inline constexpr std::size_t kMetricFamilyCount = std::size(kMetricFamilies);

enum class MetricsProtocol : std::uint8_t { OpenAiChat, OpenAiResponses, AnthropicMessages };

enum class GenerationResult : std::uint8_t { Cancelled, Error, Rejected, Success };

struct HttpRouteClass {
    const char* protocol = "other";
    const char* route    = "other";
};

[[nodiscard]] HttpRouteClass classify_http_route(std::string_view path);
[[nodiscard]] bool is_unauthenticated_path(std::string_view path);
// Maps a request-log protocol name; an unknown name is a programming error.
[[nodiscard]] MetricsProtocol metrics_protocol(std::string_view request_log_protocol);
// Closed Prometheus cause. `kind` is the recovery event kind name. Exhausted
// details collapse to their decision; JSONL keeps the raw string.
[[nodiscard]] const char* prometheus_recovery_cause(std::string_view kind,
                                                    std::string_view cause);

struct ScrapeInputs {
    ninfer::RuntimeStats stats;
    std::size_t http_in_flight   = 0;
    std::size_t response_records = 0;
    std::size_t response_bytes   = 0;
};

// One HTTP generation terminal or reject. `outcome` is set for a finished run.
// `recovery` is set when the error path has stats and no outcome.
struct GenerationObservation {
    MetricsProtocol protocol = MetricsProtocol::OpenAiChat;
    bool stream              = false;
    GenerationResult result  = GenerationResult::Success;
    bool thinking            = false;
    bool tools               = false;
    bool capture_requested   = false;
    bool has_media           = false;
    const GenerationOutcome* outcome = nullptr;
    const ninfer::GenerationRecoveryStats* recovery = nullptr;
};

struct MetricsSnapshotData;
struct ServeMetricsState;

class MetricsSnapshot {
public:
    MetricsSnapshot();
    ~MetricsSnapshot();
    MetricsSnapshot(MetricsSnapshot&&) noexcept;
    MetricsSnapshot& operator=(MetricsSnapshot&&) noexcept;

    [[nodiscard]] std::string prometheus_text() const;
    [[nodiscard]] std::string json() const;

private:
    friend class ServeMetrics;
    std::unique_ptr<MetricsSnapshotData> data_;
};

class ServeMetrics {
public:
    ServeMetrics();
    ~ServeMetrics();
    ServeMetrics(const ServeMetrics&)            = delete;
    ServeMetrics& operator=(const ServeMetrics&) = delete;
    ServeMetrics(ServeMetrics&&)                 = delete;
    ServeMetrics& operator=(ServeMetrics&&)      = delete;

    // Called once before the server accepts connections.
    void attach(const ServeOptions& options, const ninfer::LoadSummary& load,
                const ninfer::MemorySummary& memory, std::string_view model_id,
                std::string_view cuda_compile_version, std::string_view cuda_runtime_version);

    void observe_generation(const GenerationObservation& observation);
    void observe_recovery_event(const ninfer::RecoveryEvent& event);
    void observe_http(std::string_view path, std::string_view method, int status, double seconds);
    void observe_api_error(std::string_view code);
    void observe_token_count(MetricsProtocol protocol);

    [[nodiscard]] MetricsSnapshot snapshot(const ScrapeInputs& inputs) const;

private:
    std::unique_ptr<ServeMetricsState> state_;
};

void handle_metrics(const MetricsSnapshot& snapshot, httplib::Response& response);
void handle_metrics_json(const MetricsSnapshot& snapshot, httplib::Response& response);

} // namespace ninfer::serve
