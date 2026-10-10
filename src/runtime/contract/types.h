#pragma once

#include "ninfer/types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>

namespace ninfer::runtime {

// Optional cache failures distinguish source integrity, stale plans, and destination pressure.
// CUDA failures and execution errors must retain their original exception.
enum class CacheRestoreFailureKind : std::uint8_t {
    InvalidSource,
    StalePlan,
    DestinationUnavailable
};

class CacheRestoreFailure : public std::runtime_error {
public:
    explicit CacheRestoreFailure(
        const std::string& message,
        CacheRestoreFailureKind kind = CacheRestoreFailureKind::InvalidSource)
        : std::runtime_error(message), kind_(kind) {}

    [[nodiscard]] CacheRestoreFailureKind kind() const noexcept { return kind_; }
private:
    CacheRestoreFailureKind kind_;
};

enum class CacheReadIntent : std::uint8_t { Consume, Fork };
enum class CacheRetention : std::uint8_t { Ordinary, Disposable };

using ::ninfer::FinishReason;
using ::ninfer::KvCapacityMode;
using ::ninfer::KvCapacityPolicy;
using ::ninfer::OutputChannel;
using ::ninfer::ResolvedSamplingParameters;
using ::ninfer::StopPolicy;
using ::ninfer::StopString;
using ::ninfer::TokenId;

// Engine has already selected the registered model/mode preset, applied every explicit override,
// and validated these values before constructing the runtime request.
struct ResolvedExecutionOptions {
    static constexpr std::size_t kMaximumSuppressedTokens = 4;

    ResolvedSamplingParameters sampling;
    std::uint32_t requested_output_tokens = 0;
    bool allow_prefix_reuse               = true;
    // Internal cache recovery bypasses existing images without disabling capture.
    bool force_cold_prefill         = false;
    bool cache_write                = true;
    bool capture_context_checkpoint = false;
    // The target scores every generated token of the request (RoundLogprobs).
    bool token_logprobs = false;
    std::array<TokenId, kMaximumSuppressedTokens> suppressed_token_ids{};
    std::uint32_t suppressed_token_count = 0;
};

struct ResolvedRequestOptions {
    ResolvedExecutionOptions execution;
    StopPolicy stop;
    OutputOptions output;
};

struct OutputDecision {
    // Tokens to commit from the licensed round. A non-terminal decision may be a
    // proper prefix (structured-output cut). Zero only with reject_generated_round.
    std::uint32_t accepted_tokens = 0;
    FinishReason finish_reason    = FinishReason::None;
    bool reject_generated_round   = false;

    [[nodiscard]] bool finished() const noexcept { return finish_reason != FinishReason::None; }
};

// Complete request-lifetime ownership in the three independently exhausted admission domains.
// Values are already rounded to the physical allocation granularity by the target.
struct AdmissionResources {
    std::uint32_t active_lanes     = 0;
    std::uint32_t main_kv_pages    = 0;
    std::uint32_t backend_kv_pages = 0;
};

struct RequestPlanSummary {
    std::uint32_t prompt_tokens           = 0;
    std::uint32_t reusable_prompt_tokens  = 0;
    std::uint32_t requested_output_tokens = 0;
    std::uint32_t effective_output_tokens = 0;
    FinishReason effective_limit_reason   = FinishReason::None;
    std::size_t transient_bytes           = 0;
    std::size_t transient_alignment       = 1;
    AdmissionResources admission;
    std::uint64_t service_work_quanta       = 0;
    std::uint64_t ram_entry_id              = 0;
    std::uint64_t disk_entry_id             = 0;
    std::uint64_t disk_hash_f_lo            = 0;
    std::uint64_t disk_hash_f_hi            = 0;
    std::uint32_t disk_execution_frontier   = 0;
    std::uint64_t disk_committed_generation = 0;
    PrefixReusePath disk_reuse_path         = PrefixReusePath::FullReset;
    PrefixReuseSource reuse_source          = PrefixReuseSource::None;
};

struct BeginSummary {
    std::uint32_t prompt_tokens           = 0;
    std::uint32_t reused_prompt_tokens    = 0;
    PrefixReusePath prefix_reuse_path     = PrefixReusePath::FullReset;
    PrefixReuseSource prefix_reuse_source = PrefixReuseSource::None;
};

// Logprob records parallel to a round's token storage: token slot s (row * row_stride + i for a
// batched round) owns token_logprobs[s] and the kMaximumTopLogprobs ranked alternatives starting
// at top_ids / top_logprobs [s * kMaximumTopLogprobs]. Only the slots of produced tokens in rows
// whose request set ResolvedExecutionOptions::token_logprobs hold values. The spans are empty
// when no row of the round asked.
struct RoundLogprobs {
    std::span<const float> token_logprobs;
    std::span<const TokenId> top_ids;
    std::span<const float> top_logprobs;

    [[nodiscard]] bool empty() const noexcept { return token_logprobs.empty(); }

    // The record of token slot `slot`, truncated to top_count alternatives.
    [[nodiscard]] TokenLogprob record(std::size_t slot, TokenId token,
                                      std::uint32_t top_count) const {
        if (top_count > kMaximumTopLogprobs || slot >= token_logprobs.size() ||
            (slot + 1) * kMaximumTopLogprobs > top_ids.size() ||
            top_ids.size() != top_logprobs.size()) {
            throw std::logic_error("round logprob record is outside its storage");
        }
        TokenLogprob out;
        out.token     = token;
        out.logprob   = token_logprobs[slot];
        out.top_count = top_count;
        for (std::uint32_t rank = 0; rank < top_count; ++rank) {
            const std::size_t at = slot * kMaximumTopLogprobs + rank;
            out.top[rank]        = TokenAlternative{top_ids[at], top_logprobs[at]};
        }
        return out;
    }
};

struct GeneratedRound {
    std::span<const TokenId> tokens;
    RoundLogprobs logprobs;
};

struct BatchedGeneratedRound {
    std::span<const TokenId> tokens;
    RoundLogprobs logprobs;
    std::span<const std::int32_t> row_counts;
    std::uint32_t row_stride = 1;
    // True when the sampler was armed with a protected, valid cycle exclusion for
    // that row's root selection. This does not claim the unmodified draw differed.
    std::array<bool, kMaximumConcurrency> cycle_exclusions{};
};

struct PrefillStepResult {
    BeginSummary summary;
    GeneratedRound round;
    std::uint32_t processed_prompt_tokens = 0;
    bool complete                         = false;
    bool host_input_consumed              = false;
};

struct RoundBudget {
    std::uint32_t generated_tokens_remaining = 0;
};

// Target-produced affine reservation curve for one Main KV physical-capacity axis. The byte
// values come from complete target physical layout plans, not from a model geometry formula in
// the common runtime.
struct SequenceCapacityCurve {
    std::uint32_t main_page_tokens                   = 0;
    std::uint32_t minimum_main_page_groups           = 0;
    std::uint32_t maximum_main_page_groups           = 0;
    std::size_t minimum_device_reservation_bytes     = 0;
    std::size_t bytes_per_additional_main_page_group = 0;

    [[nodiscard]] std::size_t reservation_bytes(std::uint32_t main_page_groups) const;
    [[nodiscard]] std::uint32_t resolved_tokens(std::uint32_t main_page_groups) const;
};

struct KvCapacityResolution {
    KvCapacityMode mode                              = KvCapacityMode::Explicit;
    std::uint32_t main_page_groups                   = 0;
    std::uint32_t maximum_main_page_groups           = 0;
    std::uint32_t resolved_tokens                    = 0;
    std::size_t minimum_runtime_reservation_bytes    = 0;
    std::size_t bytes_per_additional_main_page_group = 0;
    std::size_t runtime_reservation_bytes            = 0;
    std::size_t available_after_weights_bytes        = 0;
    std::size_t available_after_startup_bytes        = 0;
    std::size_t automatic_headroom_bytes             = 0;
    std::size_t planned_slack_bytes                  = 0;
};

} // namespace ninfer::runtime
