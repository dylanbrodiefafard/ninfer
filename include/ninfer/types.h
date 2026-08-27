#pragma once

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace ninfer {

using TokenId = std::int32_t;

inline constexpr std::uint32_t kMaximumConcurrency = 8;
inline constexpr std::size_t kMaxContextCheckpointMarks = 16;

[[nodiscard]] inline std::vector<std::uint32_t>
parse_context_checkpoint_marks_flag(std::string_view raw,
                                    const char* flag = "--context-checkpoints") {
    if (raw == "off") { return {}; }
    if (raw.empty()) {
        throw std::invalid_argument(std::string(flag) + " must be off or a comma-separated list");
    }
    std::vector<std::uint32_t> marks;
    std::size_t begin = 0;
    while (true) {
        const std::size_t comma = raw.find(',', begin);
        const std::string_view token =
            comma == std::string_view::npos ? raw.substr(begin) : raw.substr(begin, comma - begin);
        if (token.empty()) {
            throw std::invalid_argument(std::string(flag) + " contains an empty mark");
        }
        std::uint32_t value = 0;
        const auto parsed   = std::from_chars(token.data(), token.data() + token.size(), value);
        if (parsed.ec != std::errc{} || parsed.ptr != token.data() + token.size() || value == 0) {
            throw std::invalid_argument(std::string(flag) + " marks must be positive integers");
        }
        if (!marks.empty() && value <= marks.back()) {
            throw std::invalid_argument(std::string(flag) + " marks must be strictly increasing");
        }
        if (marks.size() >= kMaxContextCheckpointMarks) {
            throw std::invalid_argument(std::string(flag) + " accepts at most 16 marks");
        }
        marks.push_back(value);
        if (comma == std::string_view::npos) { break; }
        begin = comma + 1;
    }
    return marks;
}

enum class KvCacheStorage : std::uint8_t {
    BFloat16,
    Int8Group64,
    Nvfp4,
};

enum class KvCapacityMode : std::uint8_t {
    Explicit,
    Automatic,
};

inline constexpr std::size_t kDefaultKvCapacityHeadroomBytes = 1024ULL * 1024ULL * 1024ULL;

struct KvCapacityPolicy {
    KvCapacityMode mode                  = KvCapacityMode::Explicit;
    std::uint32_t explicit_tokens        = 2048;
    std::size_t automatic_headroom_bytes = 0;

    [[nodiscard]] static constexpr KvCapacityPolicy
    explicit_capacity(std::uint32_t tokens) noexcept {
        return KvCapacityPolicy{KvCapacityMode::Explicit, tokens, 0};
    }

    [[nodiscard]] static constexpr KvCapacityPolicy
    automatic(std::size_t headroom_bytes = kDefaultKvCapacityHeadroomBytes) noexcept {
        return KvCapacityPolicy{KvCapacityMode::Automatic, 0, headroom_bytes};
    }
};

enum class ProposalHead : std::uint8_t {
    Full,
    Optimized,
};

enum class SpeculativeBackend : std::uint8_t {
    None,
    Mtp,
    DFlash,
};

[[nodiscard]] constexpr bool context_checkpoint_capture_available(
    bool allow_prefix_reuse, SpeculativeBackend spec) noexcept {
    return allow_prefix_reuse && spec != SpeculativeBackend::None;
}

struct SpeculativeOptions {
    SpeculativeBackend backend = SpeculativeBackend::None;
    std::uint32_t draft_tokens = 0;
    ProposalHead proposal_head = ProposalHead::Full;
    // Packed/chain DFlash verify width. 0 selects the k-dependent default.
    std::uint32_t dflash_verify_width = 0;
};

struct LoadProgress {
    std::function<void(std::string_view phase, std::uint64_t done, std::uint64_t total)> callback;
};

struct EngineOptions {
    std::filesystem::path artifact_path;
    int device                         = 0;
    std::uint32_t max_context          = 2048; // Exact logical ceiling of each request.
    KvCapacityPolicy kv_capacity       = KvCapacityPolicy::explicit_capacity(2048);
    std::uint32_t max_concurrency      = 1;
    std::uint32_t max_pending_requests = 16;
    std::uint32_t pending_timeout_ms   = 30000;
    std::uint32_t prefill_chunk        = 4096;
    std::size_t kv_ram_capacity_bytes  = 0;
    // nullopt = default prefill ladder; empty = disable automatic ladder (`off`).
    std::optional<std::vector<std::uint32_t>> context_checkpoint_marks;
    KvCacheStorage kv_cache            = KvCacheStorage::Nvfp4;
    // Sage3-style FP4-PV compute recipe (SageAttention3): only issuable with KvCacheStorage::Nvfp4.
    bool sage_attn                     = false;
    // Prefill tile-skip on exact NVFP4 (not Sage3). 1.0 = dense. Mutually exclusive
    // with xattn_tau < 1. Requires kv_cache == Nvfp4 and sage_attn == false.
    float keep_frac = 1.0f;
    // XAttention mass threshold τ on exact NVFP4 prefill. 1.0 = dense. Mutually
    // exclusive with keep_frac < 1. Requires kv_cache == Nvfp4 and sage_attn == false.
    float xattn_tau = 1.0f;
    // Skip XAttention ranking below this cached length; short prefixes stay dense.
    std::int32_t xattn_min_len = 8192;
    SpeculativeOptions speculative;
    bool enable_vision  = false;
    bool use_cuda_graph = true;
    LoadProgress load_progress;
};

inline constexpr std::int32_t kDefaultXattnMinLen = 8192;

inline float parse_unit_interval_flag(const char* raw, const char* flag) {
    char* end = nullptr;
    const float parsed = std::strtof(raw, &end);
    if (end == raw || *end != '\0' || !(parsed > 0.0f && parsed <= 1.0f)) {
        throw std::invalid_argument(std::string(flag) + " must be a float in (0, 1]");
    }
    return parsed;
}

inline void validate_sparse_attn_flags(KvCacheStorage kv_cache, bool sage_attn, float keep_frac,
                                       float xattn_tau) {
    if (!(keep_frac > 0.0f && keep_frac <= 1.0f)) {
        throw std::invalid_argument("--keep-frac must be a float in (0, 1]");
    }
    if (!(xattn_tau > 0.0f && xattn_tau <= 1.0f)) {
        throw std::invalid_argument("--xattn-tau must be a float in (0, 1]");
    }
    if (keep_frac < 1.0f && xattn_tau < 1.0f) {
        throw std::invalid_argument("--keep-frac and --xattn-tau are mutually exclusive");
    }
    if (sage_attn && (keep_frac < 1.0f || xattn_tau < 1.0f)) {
        throw std::invalid_argument(
            "--sage is exact-S3 only; --keep-frac / --xattn-tau require --kv-dtype nvfp4 without "
            "--sage");
    }
    if ((keep_frac < 1.0f || xattn_tau < 1.0f) && kv_cache != KvCacheStorage::Nvfp4) {
        throw std::invalid_argument("--keep-frac / --xattn-tau require --kv-dtype nvfp4");
    }
}

enum class SamplingMode : std::uint8_t {
    Thinking,
    NonThinking,
};

// Immutable model-owned values used when a request does not override a sampling field. Seed is
// deliberately excluded: it is an execution choice rather than a model recommendation.
struct SamplingPreset {
    float temperature       = 0.0F;
    std::int32_t top_k      = 0;
    float top_p             = 1.0F;
    float min_p             = 0.0F;
    float presence_penalty  = 0.0F;
    float frequency_penalty = 0.0F;
};

struct ModelSamplingDefaults {
    SamplingPreset thinking;
    SamplingPreset non_thinking;

    [[nodiscard]] constexpr const SamplingPreset& for_mode(SamplingMode mode) const noexcept {
        return mode == SamplingMode::Thinking ? thinking : non_thinking;
    }
};

// Public request-side overrides. std::nullopt means "use the registered model/mode default";
// explicit zero remains a real override (including temperature=0 for exact argmax).
struct SamplingOverrides {
    std::optional<float> temperature;
    std::optional<std::int32_t> top_k;
    std::optional<float> top_p;
    std::optional<float> min_p;
    std::optional<float> presence_penalty;
    std::optional<float> frequency_penalty;
    std::optional<std::uint64_t> seed;
};

// Complete parameters after Engine resolution. Target runtimes consume only this type.
struct ResolvedSamplingParameters {
    float temperature       = 0.0F;
    std::int32_t top_k      = 0;
    float top_p             = 1.0F;
    float min_p             = 0.0F;
    float presence_penalty  = 0.0F;
    float frequency_penalty = 0.0F;
    std::uint64_t seed      = 0;
};

enum class OutputChannel : std::uint8_t {
    Content,
    Reasoning,
};

struct StopString {
    std::string text;
    OutputChannel channel  = OutputChannel::Content;
    bool include_in_output = false;
};

struct StopPolicy {
    std::vector<TokenId> token_ids;
    std::vector<StopString> strings;
    bool include_model_defaults = true;
    bool publish_stop_token     = false;
};

struct ExecutionOptions {
    SamplingOverrides sampling;
    std::uint32_t requested_output_tokens = 0;
    bool allow_prefix_reuse               = true;
    bool capture_context_checkpoint       = false;
};

struct OutputOptions {
    bool raw                     = false;
    bool preserve_special_tokens = false;
};

struct RequestOptions {
    ExecutionOptions execution;
    StopPolicy stop;
    OutputOptions output;
};

// Owns a bounded host-input reservation whose lifetime may cross from request preparation into
// Engine execution. The concrete reservation is product-owned; Engine only releases it once the
// target reports that the retained host payload is no longer needed.
class HostInputLease {
public:
    HostInputLease() noexcept = default;

    explicit HostInputLease(std::shared_ptr<void> owner) noexcept : owner_(std::move(owner)) {}

    HostInputLease(HostInputLease&&) noexcept            = default;
    HostInputLease& operator=(HostInputLease&&) noexcept = default;

    HostInputLease(const HostInputLease&)            = delete;
    HostInputLease& operator=(const HostInputLease&) = delete;

    void reset() noexcept { owner_.reset(); }

    [[nodiscard]] explicit operator bool() const noexcept { return owner_ != nullptr; }

private:
    std::shared_ptr<void> owner_;
};

enum class MediaKind : std::uint8_t {
    Image,
    Video,
};

struct OwnedMedia {
    MediaKind kind = MediaKind::Image;
    std::vector<std::uint8_t> bytes;
    std::string media_type;
    std::string source_name;
};

struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments_json;
};

// Wire-independent conversation authority. Protocol adapters preserve these roles and their
// ordering; a target frontend owns any model-specific role lowering.
enum class ChatRole : std::uint8_t {
    System,
    Developer,
    User,
    Assistant,
    Tool,
};

enum class MessagePartKind : std::uint8_t {
    Text,
    Media,
};

struct MessagePart {
    MessagePartKind kind = MessagePartKind::Text;
    std::string text;
    OwnedMedia media;
};

struct ChatMessage {
    ChatRole role = ChatRole::User;
    std::vector<MessagePart> parts;
    std::string reasoning_content;
    std::vector<ToolCall> tool_calls;
    std::string tool_call_id;
};

enum class ReasoningEffort : std::uint8_t {
    Low,
    Medium,
    XHigh,
};

struct ReasoningEffortCapabilities {
    bool low    = false;
    bool medium = false;
    bool xhigh  = false;
    std::optional<ReasoningEffort> default_effort;

    [[nodiscard]] constexpr bool supports(ReasoningEffort effort) const noexcept {
        switch (effort) {
        case ReasoningEffort::Low:
            return low;
        case ReasoningEffort::Medium:
            return medium;
        case ReasoningEffort::XHigh:
            return xhigh;
        }
        return false;
    }
};

struct PromptCapabilities {
    bool enable_thinking = false;
    ReasoningEffortCapabilities reasoning_effort;
};

struct PromptOptions {
    bool add_generation_prompt = true;
    bool enable_thinking       = true;
    std::optional<ReasoningEffort> reasoning_effort;
    bool preserve_thinking = false;
    bool add_vision_id     = false;
    std::vector<std::string> tool_jsons;
};

struct PromptInput {
    std::vector<ChatMessage> messages;
    PromptOptions options;
};

enum class RequestErrorKind : std::uint8_t {
    ContextLengthExceeded,
    MediaBudgetExceeded,
    Overloaded,
    QueueTimeout,
    Unavailable,
};

class RequestError final : public std::invalid_argument {
public:
    RequestError(RequestErrorKind kind, std::string message)
        : std::invalid_argument(std::move(message)), kind_(kind) {}

    [[nodiscard]] RequestErrorKind kind() const noexcept { return kind_; }

private:
    RequestErrorKind kind_;
};

struct PromptSummary {
    std::uint32_t prompt_tokens = 0;
    bool has_media              = false;
};

enum class FinishReason : std::uint8_t {
    None,
    OutputLimit,
    ContextCapacity,
    StopToken,
    StopString,
    Cancelled,
};

struct OutputDelta {
    OutputChannel channel = OutputChannel::Content;
    std::string text;
};

class OutputSink {
public:
    virtual ~OutputSink()                   = default;
    virtual void publish(OutputDelta delta) = 0;
};

class CancellationView {
public:
    CancellationView() = default;
    explicit CancellationView(std::function<bool()> requested);

    [[nodiscard]] bool requested() const;
    [[nodiscard]] bool armed() const noexcept;

private:
    std::function<bool()> requested_;
};

struct GenerationTimings {
    double prepare_seconds     = 0.0;
    double first_token_seconds = 0.0;
    double vision_seconds      = 0.0;
    double prefill_seconds     = 0.0;
    double decode_seconds      = 0.0;
    double total_seconds       = 0.0;
    // Prefill throughput over the trailing window (<= 1s) of prefill execution, i.e. the
    // steady-state rate once warm. Degenerates to the overall prefill average when the
    // prefill is shorter than the window. 0 when prefill did not process prompt tokens
    // (fully reused prefix).
    double prefill_tail_tok_s     = 0.0;
    double prefill_tail_window_s  = 0.0;
};

struct SpeculativeStats {
    SpeculativeBackend backend    = SpeculativeBackend::None;
    bool enabled                  = false;
    std::uint32_t draft_window    = 0;
    std::uint64_t rounds          = 0;
    std::uint64_t drafted_tokens  = 0;
    std::uint64_t accepted_tokens = 0;
    std::uint64_t fallback_steps  = 0;
    std::vector<std::uint64_t> accepted_per_position;
};

enum class PrefixReusePath : std::uint8_t {
    FullReset,
    AppendAtFrontier,
    RestoreTurnCheckpoint,
    RestoreResponseCheckpoint,
    RestoreContextCheckpoint,
    RestoreTurnRollback,
};

enum class PrefixReuseSource : std::uint8_t {
    None,
    VramResident,
    HostRam,
};

struct GenerationResult {
    PromptSummary prompt;
    std::vector<TokenId> generated_token_ids;
    std::string content;
    std::string reasoning;
    std::uint32_t reasoning_tokens     = 0;
    FinishReason finish_reason         = FinishReason::None;
    std::uint32_t reused_prompt_tokens = 0;
    PrefixReusePath prefix_reuse_path  = PrefixReusePath::FullReset;
    PrefixReuseSource prefix_reuse_source = PrefixReuseSource::None;
    // Absolute staged-checkpoint head frontiers this request restored or wrote;
    // 0 if none. restored is the matching ladder or turn-rollback F (the same
    // length as reused_prompt_tokens on those paths). captured is the advertised
    // ladder freeze, occupy-append rollback pin, or exact-hit request pin written
    // this request.
    std::uint32_t captured_context_checkpoint_tokens = 0;
    std::uint32_t restored_context_checkpoint_tokens = 0;
    // CUDA D2H/H2D elapsed for this request's admission spills and RAM restore.
    double kv_ram_save_seconds = 0;
    double kv_ram_load_seconds = 0;
    GenerationTimings timings;
    SpeculativeStats speculative;
};

struct ArenaMemorySummary {
    std::size_t capacity_bytes  = 0;
    std::size_t used_bytes      = 0;
    std::size_t peak_used_bytes = 0;
};

struct MemorySummary {
    int device                                = 0;
    std::uint32_t max_context                 = 0;
    KvCapacityMode kv_capacity_mode           = KvCapacityMode::Explicit;
    std::uint32_t kv_capacity                 = 0; // Resolved page-aligned Main KV capacity.
    std::uint32_t kv_capacity_page_groups     = 0;
    std::uint32_t kv_capacity_max_page_groups = 0;
    KvCacheStorage kv_cache                   = KvCacheStorage::Nvfp4;
    ArenaMemorySummary weights;
    ArenaMemorySummary sequence;
    ArenaMemorySummary workspace;
    ArenaMemorySummary request_transient;
    std::size_t minimum_runtime_reservation_bytes = 0;
    std::size_t kv_capacity_increment_bytes       = 0;
    std::size_t runtime_reservation_bytes         = 0;
    std::size_t available_after_weights_bytes     = 0;
    std::size_t available_after_startup_bytes     = 0;
    std::size_t kv_capacity_headroom_bytes        = 0;
    std::size_t planned_slack_bytes               = 0;
    std::size_t workspace_logical_peak_bytes      = 0;
    std::size_t cuda_graph_allowance_bytes        = 0;
    std::size_t cuda_graph_observed_bytes         = 0;
    std::size_t kv_payload_bytes                  = 0;
    std::size_t kv_ram_capacity_bytes             = 0;
    // Live host-RAM residents only (claimed included). Not pinned-arena occupancy;
    // a retired copy may still occupy the pin until its D2H/H2D event is reaped.
    std::size_t kv_ram_used_bytes                 = 0;
    std::size_t kv_ram_entry_count                = 0;
};

// Monotonic execution counters plus one boundary-consistent scheduler snapshot. Consumers derive
// interval throughput by subtracting two snapshots and dividing by their own monotonic wall time.
struct RuntimeStats {
    // Actual prompt tokens evaluated by prefill; resident prefix hits are excluded.
    std::uint64_t computed_prefill_tokens = 0;
    // Tokens committed by decode rounds; the first token emitted by prefill is excluded.
    std::uint64_t committed_decode_tokens = 0;
    // Decode batch executions and the sum of their batch sizes.
    std::uint64_t decode_rounds         = 0;
    std::uint64_t decode_row_rounds     = 0;
    std::uint32_t running_requests      = 0;
    std::uint32_t prefilling_requests   = 0;
    std::uint32_t decode_ready_requests = 0;
    std::uint32_t waiting_requests      = 0;
    std::uint64_t kv_ram_captures       = 0;
    std::uint64_t kv_ram_restores       = 0;
    std::uint64_t kv_ram_evictions      = 0;
    std::uint64_t kv_ram_drops          = 0;
    double kv_ram_save_seconds          = 0;
    double kv_ram_load_seconds          = 0;
    std::size_t kv_ram_capacity_bytes   = 0;
    std::size_t kv_ram_used_bytes       = 0;
    std::size_t kv_ram_entry_count      = 0;
};

enum class ScoreSchedule : std::uint8_t {
    Prefill,
    Decode,
};

[[nodiscard]] inline const char* score_schedule_name(ScoreSchedule schedule) noexcept {
    switch (schedule) {
    case ScoreSchedule::Prefill:
        return "prefill";
    case ScoreSchedule::Decode:
        return "decode";
    }
    return "unknown";
}

struct ScoreOptions {
    ScoreSchedule schedule = ScoreSchedule::Prefill;
    // nullopt: skip the first prompt_tokens/2 positions so scored tokens have KV context.
    // 0: score every teacher-forced position except the last prompt token.
    std::optional<std::uint32_t> skip_tokens;
};

// Teacher-forced NLL at or above this is reported as a terrible token (p <= exp(-10) ≈ 4.5e-5).
inline constexpr double kScoreTerribleNll = 10.0;

struct ScoreResult {
    ScoreSchedule schedule      = ScoreSchedule::Prefill;
    std::uint32_t prompt_tokens = 0;
    std::uint32_t skip_tokens   = 0;
    std::uint32_t tokens_scored = 0;
    std::uint32_t non_finite    = 0;
    std::uint32_t terrible_tokens = 0;
    double sum_nll              = 0.0;
    double mean_nll             = 0.0;
    double max_nll              = 0.0;
    double perplexity           = 0.0;
    double score_seconds        = 0.0;
    std::vector<float> token_nlls;
};

struct LoadSummary {
    std::string target;
    std::string model_id;
    std::string weights_id;
    double load_seconds                = 0.0;
    double upload_seconds              = 0.0;
    std::uint64_t artifact_bytes_read  = 0;
    std::uint64_t host_to_device_bytes = 0;
    std::uint64_t peak_staging_bytes   = 0;
    std::size_t tensor_count           = 0;
    std::size_t resource_count         = 0;
};

} // namespace ninfer
