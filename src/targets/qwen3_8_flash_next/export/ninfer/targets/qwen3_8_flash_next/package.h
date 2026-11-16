#pragma once

// The registered qwen3.8-flash-next/f3-fp8-nvfp4 package: identity, Engine option checks, loading,
// the shared Qwen3.6 frontend, and EngineProgram, the adapter through which the Engine's
// ConcurrentExecutor drives the target's own lane Program (impl/runtime/program.h).
//
// Adapter mapping (executor call -> Program call, or the fixed answer for a feature the target
// lacks). Executor lanes and Program lanes are distinct indices: an executor lane maps to the
// Program lane its open_lane returned. A closed turn keeps that binding until consuming reuse or
// eviction; disposable reuse copies into another independently allocated Program lane.
//
//   plan_request_base, plan_request_for_lane   host plan: pages for prompt + output - 1 tokens
//   can_admit_lane(_after_*)                   executor lane free && Program::can_admit
//   admission_capacity                         {lanes, kv_capacity_tokens / 64, 0}
//   start_prefill_lane, advance_prefill_lane   open_lane or reuse_lane, then prefill_step
//                                              (one super-chunk) per call; masks re-armed before
//                                              the completing step
//   prefill_remaining_tokens                   prompt tokens not yet appended
//   prefill_suspendable / resume_prefill_lane  true / nothing to restage (per-lane state)
//   prefill_mixable / decode_batch_with_prefill false / std::logic_error
//   decode_batch                               mask arming and one compact decode_round(); rows
//                                              with no eligible token close and report an error
//   resolve_prefill_lane, resolve_pending_batch authoritative licensed prefix publication; native
//                                              DFlash, including K0, defers model/count commits
//                                              and preserves state on zero publication
//   abort_lane                                 close_lane
//   set_suppressed_tokens_lane & co.           set_sampling before the next sampled token
//   score                                      Program::score (prefill schedule only)
//   memory_summary / kv_gpu_snapshot           Program::memory / reserved pages
//   retained lanes, prefix reuse               current frontier, typed frontend rewrite and one
//                                              occupy rollback image; exact token prefix at that
//                                              frontier, complete GDN/PLE/tail state, no LCP
//                                              rollback
//   fork_retained_lane                         independent page/continuation copy, source unchanged
//   RAM/disk tiers                             shared FIFO/pack store, exact QSA and complete
//                                              target-owned continuation images
//   windows, live suspension                   funded output windows, secured complete-state RAM
//                                              images with request and sampling history preserved
//   speculation                                optional native DFlash2, chain K0..3, full shared
//                                              head; complete context and request-stat transport
//   context checkpoints                        absent (rejected at startup)

#include "ninfer/types.h"
#include "runtime/contract/transient_region.h"
#include "runtime/contract/types.h"
#include "targets/qwen3_6/impl/runtime/kv_disk_snapshot.h"
#include "targets/qwen3_6/impl/runtime/kv_gpu_snapshot.h"
#include "targets/qwen3_6/impl/runtime/kv_ram_snapshot.h"
#include <ninfer/targets/qwen3_6/frontend.h>
#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace ninfer {

struct DeviceContext;

namespace artifact {
class Reader;
struct LoadProgress;
} // namespace artifact

namespace targets::qwen3_8_flash_next {

class LoadedModel;
class EngineProgram;

/**
 * A request's plan, independent of the lane it runs on. Its summary carries the executor's
 * admission accounting: one lane and QSA pages of the prompt plus the funded output window
 * (full completion with RAM off). The last sampled token is never appended. There are no backend
 * pages or request-transient bytes.
 */
class RequestBasePlan {
public:
    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept { return summary_; }

private:
    friend class EngineProgram;

    runtime::RequestPlanSummary summary_;
    // Tokens open_lane reserves.
    std::uint32_t reserve_tokens_ = 0;
    ResolvedSamplingParameters sampling_;
    std::vector<TokenId> suppressed_tokens_;
    bool token_logprobs_ = false;
    bool allow_reuse_    = false;
    bool force_cold_     = false;
    bool cache_write_    = true;
    std::optional<qwen3_6::RewriteCheckpointSpec> rewrite_;
};

/** A base plan bound to the executor lane it was planned for. */
class RequestPlan {
public:
    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept {
        return base_.summary();
    }

private:
    friend class EngineProgram;

    RequestBasePlan base_;
    std::uint32_t lane_ = 0;
};

/**
 * The ConcurrentExecutor's Program surface over the target Program (see the mapping above). It
 * owns the target Program and the per-lane request bookkeeping the Program does not hold: the
 * unappended prompt, the sampled-but-unresolved token, the output-session binding that supplies
 * token masks and cycle-exclusion policy, and per-request timings. Calls come from the executor's
 * worker thread, except memory_summary, kv_gpu_snapshot and the snapshot queries, which take no
 * lane state; the executor serializes them with execution.
 */
class EngineProgram {
public:
    using PreparedPrompt = qwen3_6::PreparedPrompt;
    using OutputSession  = qwen3_6::OutputSession;

    EngineProgram(const LoadedModel& model, const EngineOptions& options, DeviceContext& device);
    ~EngineProgram();

    EngineProgram(const EngineProgram&)            = delete;
    EngineProgram& operator=(const EngineProgram&) = delete;
    EngineProgram(EngineProgram&&)                 = delete;
    EngineProgram& operator=(EngineProgram&&)      = delete;

    // ---- Planning and admission --------------------------------------------------------------

    [[nodiscard]] RequestBasePlan
    plan_request_base(const PreparedPrompt& prompt,
                      const runtime::ResolvedExecutionOptions& options);
    [[nodiscard]] RequestPlan plan_request_for_lane(std::uint32_t lane,
                                                    const PreparedPrompt& prompt,
                                                    const RequestBasePlan& base);
    // Shared host tiers use the same complete-state prefix proof as retained VRAM lanes.
    [[nodiscard]] RequestPlan plan_ram_reuse(const PreparedPrompt& prompt,
                                             const RequestBasePlan& base);
    [[nodiscard]] RequestPlan plan_disk_reuse(const PreparedPrompt& prompt,
                                              const RequestBasePlan& base);
    [[nodiscard]] bool can_admit_lane(std::uint32_t lane, const RequestPlan& plan) const noexcept;

    [[nodiscard]] bool
    can_admit_lane_after_retained_eviction(std::uint32_t lane,
                                           const RequestPlan& plan) const noexcept;
    [[nodiscard]] bool
    can_admit_lane_after_releasing(std::uint32_t lane, const RequestPlan& plan,
                                   std::span<const std::uint32_t> release_lanes) const noexcept;

    [[nodiscard]] runtime::AdmissionResources admission_capacity() const noexcept;

    [[nodiscard]] bool chunked_execution_enabled() const noexcept;
    [[nodiscard]] runtime::AdmissionResources
    completion_resources(const RequestBasePlan& base, std::uint32_t prompt_token_bound) const;

    // ---- Funded execution windows and secured live suspension --------------------------------

    [[nodiscard]] bool shrink_window(std::uint32_t lane, const runtime::ExecutionWindow& window);
    [[nodiscard]] runtime::ExecutionWindow plan_window(std::uint32_t lane,
                                                       std::uint32_t output_tokens) const;
    [[nodiscard]] runtime::ExecutionWindow plan_suspended_window(std::uint64_t owner,
                                                                 std::uint32_t output_tokens) const;
    [[nodiscard]] bool grant_window(std::uint32_t lane, const runtime::ExecutionWindow& window);

    [[nodiscard]] bool reserve_suspension_credit(std::uint64_t owner, std::size_t bytes);
    void release_suspension_credit(std::uint64_t owner);

    [[nodiscard]] bool start_suspend(std::uint32_t lane, std::uint64_t owner);
    [[nodiscard]] bool suspension_ready(std::uint64_t owner) const;
    void finish_suspend(std::uint32_t lane, std::uint64_t owner);
    [[nodiscard]] bool start_resume(std::uint32_t lane, std::uint64_t owner,
                                    const runtime::ExecutionWindow& window);
    [[nodiscard]] bool resume_ready(std::uint64_t owner) const;
    void finish_resume(std::uint32_t lane, std::uint64_t owner);

    void discard_suspended(std::uint64_t owner);

    // ---- Prefill and decode ------------------------------------------------------------------

    // Opens or claims a retained Program lane, then prefills the first suffix super-chunk (an
    // exact hit only finalizes from retained logits). Internal rewrite captures do not create
    // scheduling boundaries. `output`
    // must stay alive until the lane closes; `transient` is empty (no request transient) and
    // `pace` is not used, because the super-chunk is the target's only interruption unit.
    [[nodiscard]] runtime::PrefillStepResult
    start_prefill_lane(std::uint32_t lane, PreparedPrompt&& prompt, RequestPlan&& plan,
                       runtime::TransientRegion transient, runtime::PrefillPace pace,
                       const OutputSession* output = nullptr);
    [[nodiscard]] runtime::PrefillStepResult advance_prefill_lane(std::uint32_t lane,
                                                                  runtime::PrefillPace pace);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_batch(std::span<const std::uint32_t> lanes,
                 std::span<const runtime::RoundBudget> budgets);
    [[nodiscard]] std::uint32_t prefill_remaining_tokens(std::uint32_t lane) const;
    // Every lane's prompt state (GDN, QSA, PLE and n-gram history) is its own, so a suspended
    // owner resumes with nothing to restage.
    [[nodiscard]] bool prefill_suspendable(std::uint32_t lane) const;
    void resume_prefill_lane(std::uint32_t lane) const;

    [[nodiscard]] bool prefill_mixable(std::uint32_t /*lane*/) const noexcept { return false; }

    [[nodiscard]] runtime::MixedGeneratedRound
    decode_batch_with_prefill(std::span<const std::uint32_t> lanes,
                              std::span<const runtime::RoundBudget> budgets,
                              std::uint32_t owner_lane);
    void set_suppressed_tokens_lane(std::uint32_t lane, std::span<const TokenId> tokens);
    void clear_suppressed_tokens_lane(std::uint32_t lane);
    void set_typical_cycle_reasoning_lane(std::uint32_t lane, bool enabled);
    void resolve_prefill_lane(std::uint32_t lane, bool terminal);
    void resolve_pending_batch(std::span<const std::uint32_t> lanes,
                               std::span<const std::uint32_t> accepted_tokens,
                               std::span<const std::uint8_t> terminal,
                               std::span<const std::uint8_t> cancelled,
                               std::span<const std::uint8_t> rejected = {});
    void abort_lane(std::uint32_t lane) noexcept;

    // ---- Retained lanes and exact complete-state prefix reuse --------------------------------

    [[nodiscard]] bool retain_reusable_lane(std::uint32_t lane);
    [[nodiscard]] bool copy_reusable_prompt(std::uint32_t lane, std::uint32_t prompt_tokens,
                                            std::vector<TokenId>& tokens,
                                            std::uint32_t& rewrite_frontier) const;
    [[nodiscard]] bool revert_cancelled_prefill_lane(std::uint32_t lane);
    [[nodiscard]] bool has_retained_lane(std::uint32_t lane) const noexcept;
    [[nodiscard]] bool retained_lane_cache_writable(std::uint32_t lane) const noexcept;
    [[nodiscard]] bool retained_lane_precedes(std::uint32_t left,
                                              std::uint32_t right) const noexcept;

    void fork_retained_lane(std::uint32_t destination, std::uint32_t source,
                            const RequestPlan& plan);
    void retarget_plan_to_ram(RequestPlan& plan, std::uint64_t entry_id);
    [[nodiscard]] bool preserve_retained_lane(std::uint32_t lane, std::uint64_t& entry_id,
                                              bool may_block, bool& deferred);

    void evict_retained_lane(std::uint32_t lane);
    void mark_turn_closed(std::uint32_t lane) noexcept;

    [[nodiscard]] bool capture_retained_lane(std::uint32_t lane,
                                             std::uint64_t* ram_entry_id = nullptr,
                                             bool may_block = true, bool* deferred = nullptr,
                                             std::span<const std::uint64_t> attempt_ram_ids = {});

    // ---- Shared host KV tiers --------------------------------------------------------------

    void restore_ram_entry(std::uint32_t lane, std::uint64_t entry_id, const RequestPlan& plan);
    void restore_disk_entry(std::uint32_t lane, std::uint64_t entry_id, const RequestPlan& plan);

    [[nodiscard]] bool ram_restore_ready(std::uint64_t entry_id) const;
    [[nodiscard]] bool disk_restore_ready(std::uint64_t entry_id) const;
    [[nodiscard]] bool kv_ram_reclaim_pending() const;

    void claim_ram_entry(std::uint64_t entry_id);
    void release_ram_entry(std::uint64_t entry_id);
    void finish_ram_restore(std::uint64_t entry_id, runtime::CacheReadIntent intent);
    [[nodiscard]] bool claim_disk_entry(std::uint64_t entry_id, std::uint32_t expected_frontier,
                                        std::uint64_t hash_lo, std::uint64_t hash_hi,
                                        std::uint32_t expected_reuse_base,
                                        PrefixReusePath expected_reuse,
                                        std::uint64_t expected_committed_generation);
    void release_disk_entry(std::uint64_t entry_id);
    void invalidate_disk_entry(std::uint64_t entry_id);
    void consume_disk_entry(std::uint64_t entry_id);
    void prefetch_disk_plan(std::uint64_t entry_id, const RequestPlan& plan);

    void pump_disk_restore();
    void cancel_disk_restore();
    void begin_copy_hold_cancel();
    [[nodiscard]] bool copy_hold_cancel_settled() const;

    void discard_ram_capture(std::uint64_t ram_id);

    void shutdown_kv_tiers(const LoadProgress& progress = {});
    void set_cache_idle(bool idle);
    [[nodiscard]] qwen3_6::detail::KvRamSnapshot kv_ram_snapshot() const noexcept;
    [[nodiscard]] qwen3_6::detail::KvRamCopySeconds harvest_kv_ram_copy_seconds();
    [[nodiscard]] std::optional<qwen3_6::detail::KvDiskSnapshot> try_kv_disk_snapshot() const;
    [[nodiscard]] qwen3_6::detail::KvDiskCopySeconds harvest_kv_disk_copy_seconds();

    [[nodiscard]] qwen3_6::detail::KvGpuSnapshot kv_gpu_snapshot() const noexcept;

    [[nodiscard]] bool kv_ram_copies_ready() const;
    [[nodiscard]] bool kv_disk_copies_ready() const;
    [[nodiscard]] bool kv_disk_restore_failed() const;
    [[nodiscard]] bool kv_copies_ready() const;
    void wait_kv_ram_copies_on_compute();
    void wait_kv_ram_copies();
    void wait_kv_disk_copies();
    [[nodiscard]] std::uint64_t pending_disk_restore_ticket() const noexcept;
    [[nodiscard]] std::uint64_t kv_ram_index_version() const noexcept;
    [[nodiscard]] std::uint64_t kv_disk_index_version() const;

    // Drains the device; every Program call already returns with its work complete.
    void synchronize_all();

    // ---- Reporting ---------------------------------------------------------------------------

    [[nodiscard]] GenerationTimings generation_timings_lane(std::uint32_t lane) const noexcept;

    [[nodiscard]] SpeculativeStats speculative_stats_lane(std::uint32_t lane) const;

    [[nodiscard]] std::uint32_t
    captured_context_checkpoint_tokens_lane(std::uint32_t /*lane*/) const noexcept {
        return 0;
    }

    [[nodiscard]] std::uint32_t
    restored_context_checkpoint_tokens_lane(std::uint32_t /*lane*/) const noexcept {
        return 0;
    }

    [[nodiscard]] MemorySummary memory_summary() const noexcept;

    void reset_memory_peaks() const noexcept {}

    // Teacher-forced NLL over the prompt on a free Program lane; only ScoreSchedule::Prefill
    // (the target has no teacher-forced decode path), else std::invalid_argument.
    [[nodiscard]] ScoreResult score(PreparedPrompt&& prompt, RequestPlan&& plan,
                                    runtime::TransientRegion transient, ScoreOptions options = {});

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

struct Package {
    static constexpr bool supports_context_checkpoints = false;
    static constexpr std::string_view model_id         = "qwen3.8-flash-next";
    static constexpr std::string_view weights_id       = "f3-fp8-nvfp4";
    static constexpr std::string_view target_key       = "qwen3_8_flash_next";

    using LoadedModel     = qwen3_8_flash_next::LoadedModel;
    using Frontend        = qwen3_6::Frontend;
    using PreparedPrompt  = qwen3_6::PreparedPrompt;
    using OutputSession   = qwen3_6::OutputSession;
    using Program         = EngineProgram;
    using RequestBasePlan = qwen3_8_flash_next::RequestBasePlan;
    using RequestPlan     = qwen3_8_flash_next::RequestPlan;

    // The checkpoint's published Thinking and Instruct presets.
    [[nodiscard]] static ModelSamplingDefaults sampling_defaults(std::string_view model);
    // Rejects unsupported draft windows/heads/verify widths, context checkpoints, Vision,
    // INT8 KV, sparse or Sage attention, eager execution, a missing
    // n-gram table, and a context beyond the QSA state capacity; std::invalid_argument names the
    // option.
    static void validate_options(const EngineOptions& options);
    // Loads the opened artifact (identity already dispatched by the registry).
    [[nodiscard]] static std::unique_ptr<LoadedModel>
    load(const artifact::Reader& reader, DeviceContext& device, artifact::LoadProgress* progress);
    [[nodiscard]] static Frontend make_frontend(const LoadedModel& model);
    // Builds the Program over all free device memory after the model: QSA state for the Engine's
    // KV capacity (Automatic: max_concurrency * max_context), then the expert cache.
    [[nodiscard]] static std::unique_ptr<Program>
    create_program(const LoadedModel& model, const EngineOptions& options, DeviceContext& device);
};

} // namespace targets::qwen3_8_flash_next
} // namespace ninfer
