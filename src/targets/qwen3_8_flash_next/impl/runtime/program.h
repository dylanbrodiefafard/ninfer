#pragma once

// The Qwen3.8-Flash-Next Program of 1..12 concurrent lanes: the n-gram table, lane state over one
// shared paged QSA pool, row feeder, expert cache and Forward over a LoadedModel, sized from the
// device at construction.

#include "config.h"
#include "load/binder.h"
#include "ngram_table.h"
#include "runtime/expert_residency.h"
#include "runtime/forward.h"
#include "runtime/kv_extents.h"
#include "runtime/ple_feeder.h"
#include "runtime/dflash.h"
#include "ninfer/types.h"
#include "runtime/state.h"

#include "core/device.h"
#ifdef NINFER_ENABLE_HIP_EXPERT_HELPER
#    include "runtime/expert_helper.h"
#endif

#include <cuda_runtime.h> // cudaStream_t

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ninfer::targets::qwen3_8_flash_next {

struct ProgramOptions {
    std::filesystem::path ngram_table;
    // Longest context one request may reach (prompt plus output), in [1,
    // ops::kQsaMaximumCapacity].
    std::int32_t context_tokens = 4096;
    // Concurrent requests, in [1, kMaxLanes]: lanes 0 .. lanes - 1, each with its own GDN, PLE and
    // n-gram history; decode graphs are captured for every batch size 1 .. lanes.
    std::int32_t lanes = 1;
    // Tokens of QSA state shared by all lanes (the --kv-capacity pool, rounded up to whole
    // 64-token pages; for example 500K tokens at 4 lanes or 600K at 6). nullopt: lanes *
    // context_tokens, so every lane can reach the full context. The expert cache takes the device
    // memory left after it, and the pool lends it whole extents of pages no lane reserves
    // (KvExpertExtents).
    std::optional<std::int64_t> kv_capacity_tokens;
    // Storage format of the QSA K/V planes; NVFP4G16 (about 11.4 KiB of QSA state per token over
    // the 12 QSA layers) or BF16 (about 28.6 KiB).
    ops::QsaKvFormat kv_format = ops::QsaKvFormat::NVFP4G16;
    // Forward options; the Program sets forward.expert_window_bytes from expert_window_layers.
    ForwardOptions forward;
    SpeculativeOptions speculative;
    // Qualification/benchmark control: exercise the adaptive path at a deterministic fixed arm.
    std::optional<std::uint32_t> forced_adaptive_k;
    // Each of the two streaming windows holds this fraction of the largest layer's routed-expert
    // records (rounded down to 256 bytes); 0 disables streaming. In [0, 1].
    double expert_window_layers = 1.0;
    // Expert-cache fixed slot bytes; by default every byte free after the model, state, Forward and
    // kDeviceReserveBytes. The flex slots over the QSA pool's idle extents come on top.
    std::optional<std::size_t> expert_cache_bytes;
    // Optional calibration routing count per expert, layer-major [48,512], for the cache's warm
    // start (ExpertResidency).
    std::span<const std::uint64_t> calib_counts;
    // Internal startup placement control; product/Engine plumbing is separate. OFF builds reject
    // a supplied PCI address before allocating lane state; nullopt preserves the ordinary path.
    std::optional<std::string> expert_helper_pci;
};

// Throws std::invalid_argument for unsupported backend, chain or proposal-head options.
void validate_speculative_options(const SpeculativeOptions& options);

// A lane: one admitted request's slot, in [0, lanes).
using Lane = std::int32_t;
// Prefix reuse resumes prompt execution; a live continuation resumes decode without sampling.
enum class RestorePhase : std::uint8_t { Prompt, Decoding };

// Ranked alternatives a sampled token reports when LaneSampling::top_logprobs is set.
inline constexpr std::int32_t kTopLogprobs = 20;

/**
 * A lane's sampling of the tokens prefill_step and decode produce, applied by ops::sample over the
 * token domain [0, kTokenDomain) with the Program's per-lane token counts (penalties, counting the
 * tokens the lane sampled at positive temperature since open_lane). temperature <= 0 is greedy
 * (lowest id on ties).
 */
struct LaneSampling {
    float temperature       = 0.0F;
    std::int32_t top_k      = 0;
    float top_p             = 1.0F;
    float min_p             = 0.0F;
    float presence_penalty  = 0.0F;
    float frequency_penalty = 0.0F;
    bool p_less             = false;
    std::uint64_t seed      = 0;
    // Tokens never sampled (at most ops::SamplingConfig::kMaximumSuppressedTokens).
    std::vector<std::int32_t> suppressed_tokens;
    // Cycle-exit continuation excluded from the typical set, or -1.
    std::int32_t typical_exclude = -1;
    // Whether prefill_step and decode report the kTopLogprobs ranked alternatives.
    bool top_logprobs = false;
};

/** Result of one prefill_step: the prompt tokens it appended, and the first sampled token once
 *  the whole prompt is in. */
struct PrefillStep {
    std::int32_t consumed   = 0;
    bool complete           = false;
    std::int32_t next_token = -1;   // valid when complete
    float next_logprob      = 0.0F; // log p(next_token) under the unmodified distribution
    // When complete and the lane's sampling sets top_logprobs: the kTopLogprobs most probable ids
    // of the unmodified distribution, descending (lower id on ties), and their log probabilities.
    std::array<std::int32_t, kTopLogprobs> top_ids{};
    std::array<float, kTopLogprobs> top_logprobs{};
};

/** Where the last decode round's time went (tooling). */
struct RoundTiming {
    // Host n-gram row gather of the round's input tokens, in seconds.
    double gather_seconds = 0.0;
    // Elapsed stream time across the graph pair in milliseconds, including exposed gather waits.
    // The host gather overlaps this interval; the two fields are not additive.
    float replay_ms = 0.0F;
};

/** Device bytes of the Program's parts, for memory reporting. */
struct ProgramMemory {
    std::size_t state_bytes        = 0; // GDN, PLE and paged QSA state of every lane
    std::size_t activation_bytes   = 0;
    std::size_t workspace_bytes    = 0;
    std::size_t window_bytes       = 0; // cache bytes the prefill streaming windows borrow
    std::size_t expert_cache_bytes = 0; // the fixed slots' own allocations
    // Record bytes of the F3 flex slots: QSA pool pages (inside state_bytes) that the pool lends
    // to the cache while no lane reserves them.
    std::size_t flex_cache_bytes = 0;
#ifdef NINFER_ENABLE_HIP_EXPERT_HELPER
    std::optional<ExpertHelperMemory> helper;
#endif
};

/**
 * Owns the parts of the Program and wires them in dependency order: the NgramTable (whose hash
 * constants must equal the artifact's), the State of every lane, the PleRowFeeder, the
 * KvExpertExtents over the State's QSA pool, the ExpertResidency with those extents' flex slots,
 * and the Forward with its decode graphs. With helper placement enabled, ExpertHelper is built
 * after fixed-capacity planning and before ExpertResidency: it owns the immutable HIP banks,
 * mailbox and stable CUDA panels, and its records are excluded from the residency's warm start.
 * Every lane operation that maps QSA pages first takes
 * back the extents it needs (KvExpertExtents::make_room), and every one that releases pages lends
 * fully free extents back (give_back), so only pages no lane reserves hold expert records. The
 * model must outlive the Program; construction synchronizes the device stream while warming the
 * cache and capturing the graphs, and throws on any component's failure: std::invalid_argument for
 * invalid options, std::runtime_error on a hash-constant mismatch or when the requested cache does
 * not fit.
 */
class Program {
public:
    // Device bytes left free for the CUDA context, CUDA Graph instantiation and allocator slack.
    static constexpr std::size_t kDeviceReserveBytes = std::size_t{1} << 30;

    Program(const LoadedModel& model, const ProgramOptions& options, DeviceContext& device);

    Program(const Program&)            = delete;
    Program& operator=(const Program&) = delete;

    // ---- Lanes: the request lifecycle the Engine adapter drives -------------------------------
    //
    // Calls come from one thread. A lane is open from open_lane until close_lane; between those,
    // its prompt is appended by prefill_step calls (one super-chunk each, the unit at which other
    // lanes' decode rounds may run) and then its tokens by decode rounds. Every call returns with
    // its device work complete enough for the host results it reports.

    [[nodiscard]] std::int32_t lanes() const noexcept;
    // Shared QSA pool size and the tokens not yet reserved by open lanes. Pages lent to the expert
    // cache count as unreserved: an admitted lane takes them back before mapping them.
    [[nodiscard]] std::int64_t kv_capacity_tokens() const noexcept;
    [[nodiscard]] std::int64_t kv_free_tokens() const noexcept;
    // Whether open_lane(reserve_tokens) would succeed now: a free lane and enough free pages.
    [[nodiscard]] bool can_admit(std::int64_t reserve_tokens) const noexcept;
    // Opens a lane at position 0 with pages reserved for reserve_tokens (prompt plus output
    // bound, at most context_tokens), or nullopt when can_admit is false.
    [[nodiscard]] std::optional<Lane> open_lane(std::int64_t reserve_tokens,
                                                const LaneSampling& sampling);
    // Releases the lane and its pages; valid at any point after open_lane (abort included).
    void close_lane(Lane lane);
    // Moves an open lane's exact frontier and page reservation at an execution boundary
    // (State::resize): growth maps pages now, shrinking unmaps the pages beyond the new
    // reservation. The caller restores recurrent and PLE state separately before executing from an
    // older frontier. Throws std::out_of_range for an invalid extent and std::bad_alloc when the
    // pool cannot fund the growth.
    void resize_lane(Lane lane, std::int32_t frontier, std::int64_t reserve_tokens);
    // Starts another request on an open retained lane at an exact complete-state frontier. If
    // image is non-null it supplies continuation_bytes() pinned bytes for that prefix; otherwise
    // the current continuation must already describe it. Resets request-local sampling counts
    // and masks, restores raw n-gram history from prefix, and reserves the incoming lifetime.
    void reuse_lane(Lane lane, std::int64_t reserve_tokens, std::span<const std::int32_t> prefix,
                    const LaneSampling& sampling, const void* image = nullptr);
    // Installs a host-tier image on a freshly opened, empty lane after its QSA transfer is
    // ordered before the compute stream. The pinned complete image must live through stream
    // completion; prefix supplies the exact raw n-gram history. The caller restores sampling
    // counts separately for a live decoding continuation, after this call resets them. Decoding
    // preserves calibration; cumulative_stats, when supplied for that phase, is copied from the
    // same logical request under this Program's startup options. Prompt resets calibration.
    void restore_lane(Lane lane, std::span<const std::int32_t> prefix, const LaneSampling& sampling,
                      const void* image, RestorePhase phase = RestorePhase::Prompt,
                      const SpeculativeStats* cumulative_stats = nullptr);
    // Replaces the lane's sampling; applies from the lane's next sampled token on. Throws
    // std::invalid_argument for invalid parameters.
    void set_sampling(Lane lane, const LaneSampling& sampling);
    // Restricts the lane's next sampled token (the completing prefill_step or the lane's next
    // decode row) to the tokens whose bit is set in `words` (kAllowedTokenWords U32; bit v of word
    // v / 32 marks token v eligible), intersected with the suppressed tokens. The words are copied;
    // the mask is consumed by that one sampled token, so a caller re-arms it for every token it
    // constrains. It never affects next_logprob or the top log probabilities. At least one
    // eligible token must remain.
    void set_allowed_tokens(Lane lane, std::span<const std::uint32_t> words);
    // Drops an armed mask that no token has consumed yet.
    void clear_allowed_tokens(Lane lane);
    // Tokens the lane holds (prompt and output so far).
    [[nodiscard]] std::int32_t committed(Lane lane) const;

    // Appends the first min(remaining.size(), super_chunk_tokens) tokens of `remaining` (the
    // prompt tokens not yet appended, non-empty) to the lane; when that completes the prompt,
    // samples the first output token. Requires the lane to be in its prompt phase and committed +
    // remaining <= reserved tokens. Other lanes' prefill_step and decode calls may run between
    // two steps of one lane. A positive token_limit cuts an internal checkpoint segment; tail
    // requests a saved distribution at that segment's end without sampling early. The adapter
    // keeps these segments inside one scheduling quantum. The next super-chunk's rows prefetch
    // while the step's device work runs and are discarded if the next step does not continue it.
    [[nodiscard]] PrefillStep prefill_step(Lane lane, std::span<const std::int32_t> remaining,
                                           std::int32_t token_limit = 0,
                                           PrefillTail tail         = PrefillTail::Skip);
    // Samples the first output token from retained tail logits, including an exact-prefix hit.
    [[nodiscard]] PrefillStep finish_prompt(Lane lane);

    // One decode round: lanes[i] (distinct, prompt complete, committed < reserved tokens)
    // appends tokens[i]; next[i] receives its sampled next token and, when next_logprob is not
    // empty, next_logprob[i] its log probability under the unmodified distribution. When top_ids
    // and top_logprobs are not empty (each lanes.size() * kTopLogprobs), row i's kTopLogprobs
    // entries [i * kTopLogprobs, (i + 1) * kTopLogprobs) receive the ranked alternatives as in
    // PrefillStep, for every row. A round with both empty skips the log-probability pass.
    // 1 <= lanes.size() <= lanes(); two graph replays of batch size lanes.size(), then the stream
    // is synchronized.
    void decode(std::span<const Lane> lanes, std::span<const std::int32_t> tokens,
                std::span<std::int32_t> next, std::span<float> next_logprob = {},
                std::span<std::int32_t> top_ids = {}, std::span<float> top_logprobs = {});

    // Teacher-forced log-probabilities log p(tokens[i] | tokens[0..i)) for i >= 1, computed on a
    // free lane that is closed again before returning; throws std::runtime_error when no lane or
    // pages are free.
    [[nodiscard]] std::vector<float> score(std::span<const std::int32_t> tokens);

    [[nodiscard]] ProgramMemory memory() const noexcept;

    // ---- Tooling access (parity driver) ---------------------------------------------------

    [[nodiscard]] SpeculativeRows decode_round(std::span<const Lane> lanes,
                                               std::span<const std::int32_t> anchors,
                                               std::span<const std::uint32_t> output_limits,
                                               RoundLogprobs logprobs);
    void resolve_round(std::span<const std::int32_t> counts);
    void flush_context(std::span<const Lane> lanes);
    [[nodiscard]] SpeculativeStats speculative_stats(Lane lane) const;

    [[nodiscard]] bool pending_speculative() const noexcept { return pending_speculative_; }

    [[nodiscard]] Forward& forward() noexcept { return forward_; }

    [[nodiscard]] State& state() noexcept { return state_; }

    [[nodiscard]] NgramTable& ngram_table() noexcept { return table_; }

    [[nodiscard]] ExpertResidency& experts() noexcept { return residency_; }

    [[nodiscard]] const KvExpertExtents& extents() const noexcept { return extents_; }

    [[nodiscard]] const RoundTiming& last_round() const noexcept { return last_round_; }

private:
    enum class LanePhase : std::uint8_t { Closed, Prompt, Decoding };

    struct LaneControl {
        LanePhase phase = LanePhase::Closed;
        LaneSampling sampling;
        // An eligibility mask is in the lane's State storage for its next sampled token.
        bool mask_armed = false;
        SpeculativeStats speculative;
    };

    // The lane's control after checking that it is open; throws std::out_of_range otherwise.
    [[nodiscard]] LaneControl& open_control(Lane lane);
    // The device sampling configuration of the lane's next token.
    [[nodiscard]] ops::SamplingConfig sampling_config(Lane lane);
    // Copies the staged eligibility masks of every lane to the device when any was staged since
    // the last upload; enqueued before the sampling call that reads them.
    void upload_masks();

    NgramTable table_;
    State state_;
    PleRowFeeder feeder_;
    std::unique_ptr<Dflash> dflash_;
    // Withholds the pool's extents before residency_ writes warm records into them.
    KvExpertExtents extents_;
#ifdef NINFER_ENABLE_HIP_EXPERT_HELPER
    std::unique_ptr<ExpertHelper> helper_;
#endif
    ExpertResidency residency_;
    Forward forward_;
    cudaStream_t stream_;
    std::array<LaneControl, kMaxLanes> lanes_{};
    RoundTiming last_round_;
    // Pinned U32 [lanes][State::kAllowedTokenWordStride], the host image of State's eligibility
    // bitsets (lane l's words at l * stride), and whether it changed since the last upload.
    PinnedHostBuffer mask_staging_;
    bool masks_staged_ = false;
    SpeculativeOptions speculative_options_;
    std::optional<std::uint32_t> forced_adaptive_k_;

    std::array<Lane, kMaxLanes> pending_lanes_{};
    std::array<std::int32_t, kMaxLanes> pending_extents_{};
    std::array<ops::SamplingConfig, kMaxLanes> round_sampling_{};
    std::array<std::int32_t, kMaxLanes> ordinary_tokens_{}, ordinary_counts_{},
        ordinary_accepted_{};
    std::array<float, kMaxLanes> ordinary_logprobs_{};
    std::array<TokenId, kTopLogprobs * kMaxLanes> ordinary_top_ids_{};
    std::array<float, kTopLogprobs * kMaxLanes> ordinary_top_logprobs_{};
    SpeculativeRows pending_round_;
    std::int32_t pending_k_   = 0;
    bool pending_batch_       = false;
    bool pending_speculative_ = false;
    [[nodiscard]] std::int32_t choose_k(std::int32_t batch);
    void initialize_speculative(Lane lane, RestorePhase phase = RestorePhase::Prompt);
};

} // namespace ninfer::targets::qwen3_8_flash_next
