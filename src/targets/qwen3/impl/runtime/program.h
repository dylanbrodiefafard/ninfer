#pragma once
#include "targets/qwen3/impl/runtime/instance.h"
// Qwen3 family runtime implementation; instantiated only by exact variants.

#include "core/arena.h"
#include "core/device.h"
#include "core/gdn_replay_records.h"
#include "ninfer/ops/sampling.h"
#include "core/decode_graph.h"
#include <ninfer/targets/qwen3/prepared_prompt.h>
#include <ninfer/targets/qwen3/generation_recovery.h>

#include "targets/qwen3/impl/runtime/context_checkpoint.h"
#include "targets/qwen3/impl/runtime/adaptive_draft.h"
#include "targets/qwen3/impl/runtime/p_less_draft_calibration.h"
#include "targets/qwen3/impl/runtime/context_checkpoint_image.h"
#include "targets/qwen3/impl/runtime/kv_gpu_snapshot.h"
#include "targets/qwen3/impl/runtime/kv_ram_cache.h"
#include "targets/qwen3/impl/runtime/kv_disk_cache.h"
#include "targets/qwen3/impl/runtime/layouts.h"
#include "targets/qwen3/impl/runtime/dflash_context.h"
#include "targets/qwen3/impl/runtime/linear_state_slots.h"
#include "targets/qwen3/impl/runtime/prefill_schedule.h"
#include "targets/qwen3/impl/runtime/prefix_identity.h"
#include "targets/qwen3/impl/runtime/prompt_embedding_staging.h"
#include "targets/qwen3/impl/runtime/text_context.h"
#include "targets/qwen3/impl/runtime/token_masks.h"
#include "targets/qwen3/impl/runtime/vision_context.h"
#include "targets/qwen3/impl/runtime/vision_prefill.h"

#include <cstdint>
#include <array>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>
#include <unordered_map>

namespace ninfer::targets::qwen3::detail::NINFER_QWEN3_RUNTIME_NS {
namespace schedule {
struct PrefillContext;
}

using PreparedPromptData    = qwen3::PreparedPromptData;
using RewriteCheckpointKind = qwen3::RewriteCheckpointKind;
using RewriteCheckpointSpec = qwen3::RewriteCheckpointSpec;

using ReusePath = ninfer::PrefixReusePath;

[[nodiscard]] constexpr bool is_rewrite_checkpoint_restore(ReusePath path) noexcept {
    return qwen3::detail::is_rewrite_checkpoint_restore(path);
}

[[nodiscard]] constexpr bool is_complete_checkpoint_restore(ReusePath path) noexcept {
    return qwen3::detail::is_complete_checkpoint_restore(path);
}

[[nodiscard]] constexpr ReusePath restore_path(RewriteCheckpointKind kind) noexcept {
    return qwen3::detail::rewrite_restore_path(kind);
}

enum class RewriteCheckpointAction : std::uint8_t {
    Drop,
    KeepExisting,
    ReclassifyExisting,
    CaptureNew,
    DeferCapture,
};

enum class MtpBridgeMode : std::uint8_t {
    None,
    BeforeSuffix,
    AfterExactHit,
};

} // namespace ninfer::targets::qwen3::detail::NINFER_QWEN3_RUNTIME_NS

namespace ninfer::targets::qwen3::detail {

template <>
struct RequestBasePlanImpl<NINFER_QWEN3_VARIANT> {
    runtime::RequestPlanSummary summary;
    ops::SamplingConfig sampling;
    std::uint32_t text_kv_page_entitlement    = 0;
    std::uint32_t backend_kv_page_entitlement = 0;
    std::shared_ptr<const qwen3::VisionControl> vision_control;
    std::optional<qwen3::RewriteCheckpointSpec> rewrite_checkpoint;
    // prefix_hash_chain(prompt), computed once at submission for a reusable prompt when a
    // checkpoint or host tier can key on it, and shared by every lane, RAM, and disk plan of
    // the request and by the occupied sequence.
    std::shared_ptr<const std::vector<PrefixHash128>> prompt_hashes;
    bool cache_write                = true;
    bool allow_prefix_reuse         = false;
    bool force_cold_prefill         = false;
    bool capture_context_checkpoint = false;
    bool token_logprobs             = false;
};

template <>
struct RequestPlanImpl<NINFER_QWEN3_VARIANT> {
    runtime::RequestPlanSummary summary;
    NINFER_QWEN3_RUNTIME_NS::ReusePath reuse = NINFER_QWEN3_RUNTIME_NS::ReusePath::FullReset;
    std::uint32_t reuse_base                 = 0;
    NINFER_QWEN3_RUNTIME_NS::MtpBridgeMode mtp_bridge =
        NINFER_QWEN3_RUNTIME_NS::MtpBridgeMode::None;
    bool prepare_mtp = false;
    std::optional<qwen3::detail::VisionPrefillPlan> vision;
    NINFER_QWEN3_RUNTIME_NS::RewriteCheckpointAction rewrite_checkpoint_action =
        NINFER_QWEN3_RUNTIME_NS::RewriteCheckpointAction::Drop;
    std::optional<qwen3::RewriteCheckpointSpec> rewrite_checkpoint_capture;
    ops::SamplingConfig sampling;
    std::uint32_t text_kv_page_entitlement       = 0;
    std::uint32_t backend_kv_page_entitlement    = 0;
    PrefixReuseSource reuse_source               = PrefixReuseSource::None;
    std::uint64_t ram_entry_id                   = 0;
    std::uint32_t system_prompt_cache_hit_tokens = 0;
    std::uint64_t disk_entry_id                  = 0;
    PrefixHash128 disk_hash_f{};
    std::uint32_t disk_execution_frontier      = 0;
    std::uint64_t disk_committed_generation    = 0;
    runtime::CacheReadIntent cache_read_intent = runtime::CacheReadIntent::Consume;
    bool capture_context_checkpoints           = false;
    bool capture_context_checkpoint            = false;
    bool capture_system_prompt                 = false;
    bool token_logprobs                        = false;
    std::shared_ptr<const std::vector<PrefixHash128>> prompt_hashes;
};

} // namespace ninfer::targets::qwen3::detail

namespace ninfer::targets::qwen3::detail::NINFER_QWEN3_RUNTIME_NS {

using RequestPlanImpl     = qwen3::detail::RequestPlanImpl<Variant>;
using RequestBasePlanImpl = qwen3::detail::RequestBasePlanImpl<Variant>;

enum class PendingKind : std::uint8_t {
    None,
    Begin,
    Ordinary,
    Speculative,
};

// Host resolution words of one segmented transaction batch: pinned, so the status/cursor
// readbacks stay asynchronous copies ordered before the round synchronization.
struct KvResolutionWords {
    std::uint32_t* status = nullptr;
    std::uint32_t* cursor = nullptr;
};

struct PendingCandidate {
    PendingKind kind               = PendingKind::None;
    std::uint32_t base_E           = 0;
    std::uint32_t base_S           = 0;
    std::uint32_t prompt_tokens    = 0;
    std::uint32_t produced         = 0;
    std::uint32_t text_kv_appended = 0;
    std::uint32_t drafted          = 0;
    std::uint32_t round_k          = 0;
    std::uint32_t verify_width     = 0;
    bool tree_verify               = false;
    qwen3::AdaptiveDraftState adaptive_before;
};

enum class Lifecycle : std::uint8_t {
    Empty,
    Prefilling,
    Active,
    Pending,
    Complete,
};

struct RewriteCheckpoint {
    bool valid                 = false;
    RewriteCheckpointKind kind = RewriteCheckpointKind::TurnClosure;
    std::uint32_t frontier     = 0;
};

using ContextCheckpointHead = qwen3::detail::ContextCheckpointHead;
using PinnedImage           = qwen3::detail::PinnedImage;

struct ContextCheckpointIndex {
    std::uint32_t frontier = 0;
    qwen3::detail::PrefixHash128 hash{};
    qwen3::detail::ContextCheckpointKind kind = qwen3::detail::ContextCheckpointKind::Ladder;
};

struct SequenceKVBundle {
    PagedKVAllocation text;
    std::optional<PagedKVAllocation> backend;
};

struct DecodeGraphProfile {
    std::uint32_t batch_size             = 1;
    std::uint32_t draft_tokens           = 0;
    std::uint32_t min_execution_frontier = 0;
    std::uint32_t max_execution_frontier = 0;
    std::uint32_t topology_class         = 0; // Startup warmup metadata only.
    DecodeGraphDefinition definition;
    DecodeGraphExecutable executable;
};

struct DecodeGraphFamily {
    std::vector<DecodeGraphProfile> profiles;
};

// Model state for continuing one logical sequence after its request has finished.
// Kept separate from request, output, sampling and round-control state.
enum class CacheCaptureMode : std::uint8_t { ReachableTurn, CompleteOriginal };

struct SequenceState {
    std::optional<SequenceKVBundle> kv;
    Tensor tail_hidden;
    Tensor rewrite_checkpoint_hidden;
    std::uint32_t lane = 0;

    std::uint32_t execution_frontier = 0;
    std::uint32_t ledger_frontier    = 0;
    std::vector<TokenId> ledger;
    qwen3::detail::ResidentPrefixIdentity prefix_identity;
    // Hashes of (ledger, prefix_identity) prefixes; truncated wherever either is rewritten.
    qwen3::detail::ResidentPrefixHashes prefix_hashes;
    std::int32_t rope_delta = 0;
    qwen3::PagedKVPublication text_kv_publication;
    qwen3::PagedKVPublication mtp_kv_publication;
    std::uint32_t dflash_context_frontier = 0;
    std::array<TokenId, qwen3::kMtpDecodeMaximumDrafts> mtp_drafts{};
    std::uint32_t mtp_draft_count           = 0;
    bool tail_hidden_valid                  = false;
    bool retained                           = false;
    runtime::CacheRetention cache_retention = runtime::CacheRetention::Ordinary;
    // A fork has already installed the selected checkpoint as current state, without carrying
    // auxiliary reuse images into the destination.
    bool fork_initialized = false;
    // The retained request stopped (stop token or string) without a tool call. With its own
    // TurnClosure checkpoint at its generation opener (closure_frontier), the next preserve-off
    // prompt re-renders the reply without reasoning and diverges there, so RAM/disk tiers store
    // the lane cut at that checkpoint (capture_cuts_at_rewrite).
    bool turn_closed = false;
    // The latest request's own TurnClosure checkpoint when it sits at that request's generation
    // opener (preserve off, outside a tool loop); 0 otherwise. Only that checkpoint is cut at.
    std::uint32_t closure_frontier = 0;
    std::uint64_t use_tick         = 0;
    std::uint64_t disk_entry_id    = 0;
    RewriteCheckpoint rewrite_checkpoint;
    // Lane-owned pinned image of the rewrite checkpoint's GDN slot and DFlash cyclic lane,
    // carved from the Program's checkpoint-image slab; rewrite_checkpoint says whether its
    // contents are valid. Its fence covers every asynchronous copy that reads or writes it. The
    // generation advances whenever its contents are replaced, so a device staging copy can prove
    // it still matches.
    ContextCheckpointHead rewrite_image;
    std::uint64_t rewrite_image_generation = 0;
    std::vector<ContextCheckpointHead> context_checkpoints;
    std::uint32_t next_context_mark = 0;
    // Set by a HostRam or HostDisk staged restore, which unpacks the matching head into current
    // with the entry's KV. Occupy skips a second unpack when it matches this (base, hash).
    // Cleared at end of occupy and in clear_lane so a later VRAM staged restore still unpacks.
    std::uint32_t tier_unpacked_context_base = 0;
    qwen3::detail::PrefixHash128 tier_unpacked_context_hash{};
    // A committed DFlash chain round whose ReplaySSM fold is deferred into the lane's next
    // verification forward: `deferred_fold_columns` (> 0 while pending) columns of physical
    // record row `deferred_fold_row`. Until it is applied, the lane's GDN slot holds the state
    // before that round and the record row must not be overwritten.
    std::uint32_t deferred_fold_columns = 0;
    std::uint32_t deferred_fold_row     = 0;
};

// Request/round control is not retained with a reusable SequenceState. A later concurrent Engine
// gives every occupied request slot its own instance of this state.
struct RequestControl {
    // Borrowed from the occupying Engine request until its lane is resolved.
    const qwen3::OutputSession* output = nullptr;
    Lifecycle lifecycle                = Lifecycle::Empty;
    PendingCandidate pending;
    ops::SamplingConfig sampling_host;
    ops::SamplingConfig prefill_sampling_host;
    GenerationTimings timings;
    SpeculativeStats speculative_stats;

    // Absolute staged-checkpoint head frontiers; 0 if none. Survive prefill.reset()
    // and abort_lane/clear_lane so completion can still read them. captured is the
    // advertised freeze or rollback pin written this request; restored is the
    // matching head F on a staged restore.
    std::uint32_t captured_context_checkpoint_tokens = 0;
    std::uint32_t restored_context_checkpoint_tokens = 0;

    struct Prefill {
        PreparedPromptData prompt;
        std::optional<VisionPrefillPlan> vision_plan;
        std::unique_ptr<schedule::VisionPrefillSession> vision;
        runtime::TransientRegion transient;
        std::optional<RewriteCheckpointSpec> rewrite_checkpoint_capture;
        std::optional<std::uint32_t> system_prompt_capture;
        std::uint32_t base                           = 0;
        std::uint32_t cursor                         = 0;
        std::uint32_t prompt_tokens                  = 0;
        std::uint32_t initial_mtp_extent             = 0;
        double elapsed_seconds                       = 0.0;
        bool host_input_consumed_pending             = false;
        bool prepare_mtp                             = false;
        ReusePath reuse                              = ReusePath::FullReset;
        PrefixReuseSource reuse_source               = PrefixReuseSource::None;
        std::uint32_t system_prompt_cache_hit_tokens = 0;
        MtpBridgeMode mtp_bridge                     = MtpBridgeMode::None;
        bool capture_context_checkpoints             = false;
        // Per-chunk prefill step records (tokens processed, wall seconds) for the
        // tail-window throughput metric. One entry per advance_prefill call; the
        // finalizing call includes its sampling/bridge time.
        std::vector<std::uint32_t> step_tokens;
        std::vector<double> step_seconds;
    };

    std::optional<Prefill> prefill;
    qwen3::AdaptiveDraftState adaptive;
    bool typical_cycle_reasoning = false;
    // The occupying request reports token logprobs: its rounds set the frame's row flag.
    bool token_logprobs                   = false;
    std::uint32_t prompt_tokens           = 0;
    std::uint32_t effective_output_tokens = 0;
    std::uint32_t window_output_tokens    = 0;
};

class ProgramImplCore {
public:
    ProgramImplCore(const LoadedModelData& model_in, const SequencePlanImpl& plan,
                    DeviceContext& device_in, std::unique_ptr<HostPinnedArena> kv_ram_arena);
    ~ProgramImplCore() noexcept;

    [[nodiscard]] RequestBasePlan
    plan_request_base(const PreparedPromptData& prompt,
                      const runtime::ResolvedExecutionOptions& options);
    [[nodiscard]] RequestPlan plan_request_for_lane(std::uint32_t lane,
                                                    const PreparedPromptData& prompt,
                                                    const RequestBasePlan& base_plan);
    [[nodiscard]] RequestPlan plan_ram_reuse(const PreparedPromptData& prompt,
                                             const RequestBasePlan& base_plan);
    [[nodiscard]] RequestPlan plan_disk_reuse(const PreparedPromptData& prompt,
                                              const RequestBasePlan& base_plan);
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

    [[nodiscard]] runtime::PrefillStepResult
    start_prefill_lane(std::uint32_t lane, PreparedPromptData&& prompt, RequestPlan&& plan,
                       runtime::TransientRegion transient,
                       const qwen3::OutputSession* output = nullptr, bool decode_waiting = false);
    // With decode_waiting and a configured mixed_forward, a mixable owner returns without
    // progress (0 tokens, incomplete): its first slice runs in the next mixed round.
    // decode_waiting bounds the step's forward by mixed_forward (when configured) instead of
    // prefill_chunk.
    [[nodiscard]] runtime::PrefillStepResult advance_prefill_lane(std::uint32_t lane,
                                                                  bool decode_waiting = false);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_batch(std::span<const std::uint32_t> lanes,
                 std::span<const runtime::RoundBudget> budgets);
    // True when the staged owner's next step can share a DFlash round's target forward: a
    // text-only prompt with prompt tokens left and no MTP bridge.
    [[nodiscard]] bool prefill_mixable(std::uint32_t lane) const noexcept;
    // One DFlash round for `lanes` that also advances the owner's next chunk (the mixed forward
    // width less the round's live verify columns) in the same target forward.
    [[nodiscard]] runtime::MixedGeneratedRound
    decode_batch_with_prefill(std::span<const std::uint32_t> lanes,
                              std::span<const runtime::RoundBudget> budgets,
                              std::uint32_t prefill_lane);
    void set_suppressed_tokens_lane(std::uint32_t lane, std::span<const TokenId> tokens);
    void clear_suppressed_tokens_lane(std::uint32_t lane);
    void set_typical_cycle_reasoning_lane(std::uint32_t lane, bool enabled);
    // Binds the batch's output sessions and the sampling configs the round runs with (its host
    // ingress), so masked accept configs match the drafted rows.
    void bind_token_mask_batch(std::span<const std::uint32_t> lanes,
                               std::span<const ops::SamplingConfig> configs);
    [[nodiscard]] bool any_token_logprobs(std::span<const std::uint32_t> lanes) const;
    void resolve_prefill_lane(std::uint32_t lane, bool terminal);
    void resolve_pending_batch(std::span<const std::uint32_t> lanes,
                               std::span<const std::uint32_t> accepted_tokens,
                               std::span<const std::uint8_t> terminal,
                               std::span<const std::uint8_t> cancelled,
                               std::span<const std::uint8_t> rejected = {});
    void abort_lane(std::uint32_t lane) noexcept;
    // Active sequences are retained in place. An already retained lane stays.
    // False leaves the caller to drop the lane; this does not clear it.
    [[nodiscard]] bool retain_reusable_lane(std::uint32_t lane);
    // Copies the prompt prefix of a live or retained lane. `rewrite_frontier` is 0
    // when that lane has no rewrite checkpoint inside the prompt.
    [[nodiscard]] bool copy_reusable_prompt(std::uint32_t lane, std::uint32_t prompt_tokens,
                                            std::vector<TokenId>& tokens,
                                            std::uint32_t& rewrite_frontier) const;
    [[nodiscard]] bool revert_cancelled_prefill_lane(std::uint32_t lane);
    [[nodiscard]] bool has_retained_lane(std::uint32_t lane) const noexcept;
    [[nodiscard]] std::uint64_t retained_use_tick(std::uint32_t lane) const noexcept;
    [[nodiscard]] bool retained_lane_cache_writable(std::uint32_t lane) const noexcept;
    [[nodiscard]] bool retained_lane_precedes(std::uint32_t left,
                                              std::uint32_t right) const noexcept;
    // Copies only the selected reusable prefix into an empty independently reserved lane.
    // The executor protects the retained source until the compute stream completes the copy.
    void fork_retained_lane(std::uint32_t destination, std::uint32_t source,
                            const RequestPlan& plan);
    // A full-original RAM capture preserves exactly the state described by the resident plan.
    void retarget_plan_to_ram(RequestPlan& plan, std::uint64_t entry_id);
    void evict_retained_lane(std::uint32_t lane) noexcept;
    // The retained lane's request finished without a tool call (see SequenceState::turn_closed).
    void mark_turn_closed(std::uint32_t lane) noexcept;
    // `may_block` permits a synchronous disk spill to free RAM; callers pass
    // false while other lanes are decoding. Without it a full RAM tier whose
    // entries are not yet on disk starts their spill on the disk worker and
    // returns false with `*deferred` set; retry after kv_ram_reclaim_pending() clears.
    // `attempt_ram_ids` are entries captured earlier in the same admission attempt, which a
    // deferral rolls back; they are never spilled or dropped to make room for this capture.
    [[nodiscard]] bool
    capture_retained_lane(std::uint32_t lane, std::uint64_t* ram_entry_id = nullptr,
                          bool may_block = true, bool* deferred = nullptr,
                          std::span<const std::uint64_t> attempt_ram_ids = {},
                          CacheCaptureMode mode = CacheCaptureMode::ReachableTurn);
    [[nodiscard]] bool preserve_retained_lane(std::uint32_t lane, std::uint64_t& entry_id,
                                              bool may_block, bool& deferred);
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
    void prefetch_disk_window(std::uint64_t entry_id, std::uint32_t text_pages,
                              std::uint32_t backend_pages);
    void prefetch_disk_plan(std::uint64_t entry_id, const RequestPlan& plan);
    void pump_disk_restore();
    void cancel_disk_restore();
    // A cancelled or failed copy-hold admission stops further restore work and fences the copies
    // already queued, so the executor can keep decoding until they settle instead of draining them.
    void begin_copy_hold_cancel();
    [[nodiscard]] bool copy_hold_cancel_settled() const;
    void discard_ram_capture(std::uint64_t ram_id);
    void shutdown_kv_tiers(LoadProgress progress = {});
    void set_cache_idle(bool idle);
    void install_pending_disk_restore_checkpoints();
    [[nodiscard]] qwen3::detail::KvRamSnapshot kv_ram_snapshot() const noexcept;
    qwen3::detail::KvRamCopySeconds harvest_kv_ram_copy_seconds();
    [[nodiscard]] std::optional<qwen3::detail::KvDiskSnapshot>
    try_kv_disk_snapshot() const noexcept;
    qwen3::detail::KvDiskCopySeconds harvest_kv_disk_copy_seconds();
    [[nodiscard]] qwen3::detail::KvGpuSnapshot kv_gpu_snapshot() const noexcept;
    [[nodiscard]] bool kv_ram_copies_ready() const;
    [[nodiscard]] bool kv_disk_copies_ready() const;
    [[nodiscard]] bool kv_disk_restore_failed() const;
    [[nodiscard]] bool kv_copies_ready() const;
    void wait_kv_ram_copies_on_compute();
    void wait_kv_ram_copies();
    void wait_kv_disk_copies();
    [[nodiscard]] std::uint64_t pending_disk_restore_ticket() const noexcept;
    void synchronize_all();
    [[nodiscard]] std::uint64_t kv_ram_index_version() const noexcept;
    [[nodiscard]] std::uint64_t kv_disk_index_version() const noexcept;
    [[nodiscard]] GenerationTimings generation_timings_lane(std::uint32_t lane) const noexcept;
    [[nodiscard]] SpeculativeStats speculative_stats_lane(std::uint32_t lane) const noexcept;
    [[nodiscard]] std::uint32_t
    captured_context_checkpoint_tokens_lane(std::uint32_t lane) const noexcept;
    [[nodiscard]] std::uint32_t
    restored_context_checkpoint_tokens_lane(std::uint32_t lane) const noexcept;

    [[nodiscard]] MemorySummary memory_summary() const noexcept;

    void reset_memory_peaks() noexcept;

    [[nodiscard]] ScoreResult score(PreparedPromptData&& prompt, RequestPlan&& plan,
                                    runtime::TransientRegion transient, ScoreOptions options);

    const LoadedModelData& model;
    DeviceContext& device;
    const WeightsProfile weights_profile;
    const std::uint32_t capacity;
    const std::uint32_t kv_capacity;
    const std::uint32_t max_concurrency;
    const std::uint32_t prefill_chunk;
    const std::uint32_t mixed_forward;
    const std::uint32_t draft_window;
    const std::uint32_t dflash_verify_width;
    const bool adaptive_draft;
    // Pinned DFlash2 p-less draft temperature; unset, chain rounds use p_less_calibration.
    const std::optional<float> p_less_draft_temperature;
    qwen3::PLessDraftCalibration p_less_calibration{};
    static_assert(qwen3::kPLessCalibrationHops == qwen3::kDFlashDecodeMaximumDrafts);

    [[nodiscard]] bool calibrates_p_less_drafts() const {
        return !p_less_draft_temperature.has_value() &&
               speculative_backend == SpeculativeBackend::DFlash;
    }

    const std::vector<std::uint32_t> captured_ks;
    std::array<qwen3::AdaptiveRoundTimeState, kMaximumConcurrency> adaptive_t_by_batch{};
    std::array<qwen3::AdaptiveBatchKState, kMaximumConcurrency> adaptive_batch_k_by_c{};
    const SpeculativeBackend speculative_backend;
    const std::vector<std::uint32_t> context_marks;
    const ProposalHead proposal_head;
    const bool vision_enabled;
    const bool use_device_graph;
    const std::size_t kv_payload_bytes;
    const std::size_t kv_ram_capacity_bytes;
    const std::uint32_t kv_ram_system_prompts;
    const std::size_t expected_graph_definition_count;
    const std::size_t expected_graph_executable_count;
    const std::size_t kv_disk_capacity_bytes;
    const std::size_t graph_allowance_bytes;
    std::size_t graph_observed_bytes = 0;
    const WorkspacePlan workspace_plan;

    DeviceArena persistent;
    DeviceArena workspace_storage;
    WorkspaceArena work;
    std::unique_ptr<qwen3::DecoderState> decoder;
    std::optional<GdnReplayRecords> replay_records;
    std::optional<DFlashPersistentState> dflash;
    std::unique_ptr<typename Variant::ExecutionState> linear_execution;
    qwen3::RoundState io;
    Tensor prefill_hidden;
    Tensor sampling_config;
    std::unique_ptr<qwen3::TokenMaskExchange> token_masks;
    Tensor token_counts;
    Tensor tail_hidden_store;
    Tensor rewrite_checkpoint_hidden_store;
    Tensor staging_hidden;
    std::unique_ptr<PromptEmbeddingStaging> prompt_embedding;

    // Every pinned checkpoint image (per-lane rewrite images and the context-checkpoint pool),
    // prefaulted and registered at startup. Declared before every head that views it.
    std::optional<HostPinnedArena> checkpoint_image_slab_;
    std::array<SequenceState, kMaximumConcurrency> sequences;
    std::array<RequestControl, kMaximumConcurrency> requests;

    DecodeGraphFamily ordinary_graphs;
    DecodeGraphFamily mtp_graphs;
    DecodeGraphFamily dflash_graphs;

    // Pinned landing area of the prefill-sampled token and its logprob record.
    struct PrefillRoundHost {
        TokenId token = 0;
        std::array<std::uint32_t,
                   static_cast<std::size_t>(qwen3::kDFlashDecodeMaximumWidth) * kMaximumConcurrency>
            nonfinite_logits{};
        qwen3::RoundLogprobRecords<1> logprobs;
    };

    PinnedHostBuffer round_host;
    TokenId* host_tokens                                 = nullptr;
    qwen3::RoundLogprobRecords<1>* host_prefill_logprobs = nullptr;
    // Pinned status/cursor words of the Text (false) and MTP (true) segmented KV batches.
    PinnedHostBuffer kv_resolution_host;

    [[nodiscard]] KvResolutionWords kv_resolution_words(bool mtp) const noexcept {
        auto* words = static_cast<std::uint32_t*>(kv_resolution_host.data()) +
                      (mtp ? 2U * kMaximumConcurrency : 0U);
        return {words, words + kMaximumConcurrency};
    }

    std::optional<PinnedHostBuffer> ordinary_host;
    qwen3::OrdinaryDecodeIngress* ordinary_host_ingress = nullptr;
    qwen3::OrdinaryDecodeEgress* ordinary_host_egress   = nullptr;
    std::optional<PinnedHostBuffer> mtp_host;
    qwen3::MtpDecodeIngress* mtp_host_ingress = nullptr;
    qwen3::MtpDecodeEgress* mtp_host_egress   = nullptr;
    std::optional<PinnedHostBuffer> dflash_host;
    qwen3::DFlashDecodeIngress* dflash_host_ingress = nullptr;
    qwen3::DFlashDecodeEgress* dflash_host_egress   = nullptr;

    std::size_t workspace_logical_peak_bytes = 0;
    std::optional<qwen3::detail::KVRamCache> kv_ram_cache_;
    std::optional<qwen3::detail::KVDiskCache> kv_disk_cache_;
    std::uint64_t next_use_tick_ = 1;
    bool kv_tiers_shutdown_      = false;
    std::optional<std::uint32_t> pending_disk_checkpoint_lane_;
    // Pool heads the disk restore workers decode into; the Program keeps them until the restore
    // installs them or cancel_restore() has quiesced every writer.
    std::vector<ContextCheckpointHead> pending_disk_checkpoint_heads_;
    std::uint64_t pending_disk_restore_ticket_ = 0;

private:
    struct SuspendedRequest {
        SequenceState sequence;
        RequestControl request;
        std::uint64_t entry_id              = 0;
        std::size_t sampling_counts_offset  = 0;
        bool sampling_counts_present        = false;
        std::size_t pending_features_offset = 0;
        std::size_t checkpoint_count        = 0;
        // Image page floors remain available to admission without decoding host descriptors.
        std::uint32_t text_pages                                = 0;
        std::uint32_t backend_pages                             = 0;
        std::uint32_t lane                                      = 0;
        enum class Phase { Capturing, Parked, Restoring } phase = Phase::Capturing;
    };

    std::unordered_map<std::uint64_t, SuspendedRequest> suspended_requests_;
    [[nodiscard]] runtime::ExecutionWindow
    execution_window(std::uint32_t prompt_tokens, std::uint32_t output_tokens,
                     std::size_t identity_extra_bytes, std::size_t checkpoint_count,
                     runtime::AdmissionResources minimum = {}) const;
    [[nodiscard]] bool select_imported_checkpoint(std::uint32_t frontier,
                                                  qwen3::detail::ContextCheckpointKind kind,
                                                  std::span<std::uint8_t> selected) const;
    void clear_lane(SequenceState& sequence, RequestControl& request) noexcept;
    // Applies the pending deferred folds of `lanes` (all lanes when empty) with the eager
    // all-layer fold, before anything else reads their GDN state or rewrites their records.
    void flush_deferred_gdn_folds(std::span<const std::uint32_t> lanes = {});
    void flush_deferred_gdn_fold(std::uint32_t lane);
    void retain_committed_sequence(SequenceState& sequence, RequestControl& request);
    void ordered_reset(SequenceState& sequence);
    void zero_dflash_local_cache(CyclicKVCache& cache);
    void prepare_graphs();
    void install_sampling(SequenceState& sequence, RequestControl& request,
                          const ops::SamplingConfig& config);
    void set_device_i32(Tensor& tensor, std::int32_t value);
    void copy_tail(SequenceState& sequence, const Tensor& source);
    void copy_round_token();
    void copy_prefill_logprobs();
    void clear_nonfinite_logits(std::uint32_t lane);
    void copy_nonfinite_logits();
    void validate_finite_logits(std::uint32_t lane, std::uint32_t columns) const;
    void resolve_non_speculative_pending(SequenceState& sequence, RequestControl& request,
                                         std::uint32_t accepted_tokens, bool terminal);
    // Runs the owner's chunk inside a mixed DFlash round instead of alone: receives the owner's
    // schedule state, nominal extent, final-candidate flag and rewrite capture frontier.
    using MixedChunkRunner = std::function<schedule::PrefillChunkResult(
        schedule::PrefillContext&, std::uint32_t, bool, std::optional<std::uint32_t>)>;
    // Binds the prefill owner's root sampling (tool-grammar root mask included) on its lane.
    void bind_prefill_sampling(SequenceState& sequence, RequestControl& request);
    [[nodiscard]] runtime::PrefillStepResult
    advance_prefill(SequenceState& sequence, RequestControl& request, bool decode_waiting,
                    const MixedChunkRunner* mixed = nullptr, std::uint32_t step_cap = 0);

    // Largest prompt extent of one prefill step: while decode rows wait, a separate step's
    // forward is the configured mixed forward width.
    [[nodiscard]] std::uint32_t prefill_step_tokens(bool decode_waiting) const noexcept {
        return decode_waiting && mixed_forward != 0 ? mixed_forward : prefill_chunk;
    }

    // Step extent the admission projection counts with: the least a non-final step advances
    // while decode rows wait. A mixed step's owner gets at least the forward less every other
    // lane's full verify panel, and can leave the cursor unaligned, after which a step above the
    // irregular split may split at it, so the extent is capped at the split.
    [[nodiscard]] std::uint32_t projected_step_tokens() const noexcept {
        if (mixed_forward == 0) { return prefill_chunk; }
        return std::min(mixed_forward - (max_concurrency - 1U) * dflash_verify_width,
                        kIrregularPrefillSplit);
    }

    // Target width of a mixed round: the owner fills it beside the round's live verify panels.
    [[nodiscard]] std::uint32_t mixed_forward_tokens() const noexcept { return mixed_forward; }

    void enqueue_dflash_context_append(std::span<const std::uint32_t> lanes,
                                       std::span<const std::uint32_t> starts,
                                       std::span<const std::uint32_t> counts);
    void validate_licensed_tokens(std::span<const TokenId> tokens) const;
    void mark_workspace_usage(std::size_t phase_bytes) noexcept;
    void order_persistent_decode_after_transfers(std::size_t batch_size,
                                                 const DecodeGraphExecutable* executable);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_ordinary_batch(std::span<const std::uint32_t> lanes,
                          std::span<const runtime::RoundBudget> budgets);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_mtp_batch(std::span<const std::uint32_t> lanes,
                     std::span<const runtime::RoundBudget> budgets);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_dflash_batch(std::span<const std::uint32_t> lanes,
                        std::span<const runtime::RoundBudget> budgets,
                        std::optional<std::uint32_t> prefill_lane = std::nullopt,
                        runtime::PrefillStepResult* prefill_step  = nullptr);
    void reserve_sequence_kv(SequenceState& sequence, std::uint32_t text_pages,
                             std::uint32_t backend_pages);
    void resize_sequence_kv_entitlement(SequenceState& sequence, std::uint32_t text_pages,
                                        std::uint32_t backend_pages);
    void bind_sequence_kv(SequenceState& sequence);
    void unbind_sequence_kv(SequenceState& sequence) noexcept;
    void truncate_sequence_mtp_publication(SequenceState& sequence,
                                           std::uint32_t retained_frontier);
    void materialize_sequence_kv(SequenceState& sequence, std::uint32_t main_tokens,
                                 std::uint32_t backend_tokens = 0);
    void trim_sequence_kv(SequenceState& sequence, std::uint32_t main_tokens,
                          std::uint32_t backend_tokens = 0);
    void release_sequence_growth_entitlement(SequenceState& sequence) noexcept;
    [[nodiscard]] PagedKVPool* backend_kv_pool() noexcept;
    [[nodiscard]] const PagedKVPool* backend_kv_pool() const noexcept;
    [[nodiscard]] std::uint32_t backend_kv_valid(const SequenceState& sequence) const noexcept;
    [[nodiscard]] qwen3::PagedKVCacheView text_kv_view(const SequenceState& sequence) const;
    [[nodiscard]] qwen3::PagedKVCacheView mtp_kv_view(const SequenceState& sequence) const;

    struct ResidentStateView {
        const std::vector<TokenId>* ledger                    = nullptr;
        const qwen3::detail::ResidentPrefixIdentity* identity = nullptr;
        std::uint32_t execution_frontier                      = 0;
        RewriteCheckpoint rewrite_checkpoint;
        std::uint32_t text_kv_valid           = 0;
        std::uint32_t mtp_kv_valid            = 0;
        std::uint32_t dflash_context_frontier = 0;
        bool tail_hidden_valid                = false;
        bool backend_image_present            = false;
        std::vector<ContextCheckpointIndex> context_checkpoints;
    };

    void apply_reuse_decision(RequestPlanImpl& plan, const ResidentStateView& view,
                              const PreparedPromptData& prompt, const RequestBasePlanImpl& base);
    void finish_request_plan(RequestPlanImpl& plan, const ResidentStateView* view,
                             const PreparedPromptData& prompt, const RequestBasePlanImpl& base);
    void capture_system_prompt(SequenceState& sequence, RequestControl::Prefill& staged,
                               std::uint32_t processed_tokens);
    [[nodiscard]] bool capture_cuts_at_rewrite(const SequenceState& sequence) const noexcept;
    [[nodiscard]] qwen3::detail::RamCaptureSource
    ram_capture_source(SequenceState& sequence, qwen3::detail::ResidentPrefixIdentity& cut_identity,
                       CacheCaptureMode mode);
    [[nodiscard]] qwen3::detail::RamCaptureSource
    cut_ram_capture_source(SequenceState& sequence,
                           qwen3::detail::ResidentPrefixIdentity& cut_identity);
    void accumulate_prefill_nll(std::span<const TokenId> ids, std::uint32_t chunk_begin,
                                std::uint32_t chunk_tokens, std::uint32_t skip,
                                ScoreResult& result);
    void accumulate_decode_nll(const Tensor& logits, TokenId target, ScoreResult& result,
                               DeviceArena& score_workspace);
    void run_prefill_score(PreparedPromptData&& prompt, RequestPlan&& plan,
                           runtime::TransientRegion transient, std::span<const TokenId> ids,
                           std::uint32_t skip, ScoreResult& result);
    void run_decode_score(PreparedPromptData&& prompt, runtime::TransientRegion transient,
                          std::span<const TokenId> ids, std::uint32_t prefix, ScoreResult& result);
    void maybe_freeze_context_checkpoint(SequenceState& sequence, RequestControl& request,
                                         std::uint32_t chunk_tokens);
    void maybe_capture_turn_rollback(SequenceState& sequence, RequestControl& request,
                                     const PreparedPromptData& prompt, ReusePath reuse,
                                     std::uint32_t base, std::uint32_t prompt_tokens,
                                     bool capture_enabled, bool request_pin, bool cut_restore);
    void restore_context_checkpoint_state(SequenceState& sequence, std::uint32_t base);
    void allocate_rewrite_image(SequenceState& sequence);
    void capture_rewrite_image(SequenceState& sequence);
    void restore_rewrite_checkpoint_state(SequenceState& sequence);
    [[nodiscard]] qwen3::detail::RewriteStateHostTarget
    rewrite_state_host_target(SequenceState& sequence);
    void restore_dflash_cyclic_from_head(SequenceState& sequence,
                                         const ContextCheckpointHead& head);
    void snapshot_dflash_cyclic_to_staging(std::int32_t lane);
    void pack_dflash_cyclic_to_head(ContextCheckpointHead& head);
    [[nodiscard]] PinnedImage carve_checkpoint_image(std::size_t bytes);
    // An unowned pool head, or none once every head is owned (the image is optional).
    [[nodiscard]] std::optional<ContextCheckpointHead> acquire_context_checkpoint_head() noexcept;
    void record_context_checkpoint_head_use(ContextCheckpointHead& head, hipStream_t stream);
    void recycle_context_checkpoint_head(ContextCheckpointHead&& head) noexcept;
    void drop_context_checkpoints_after(SequenceState& sequence, std::uint32_t frontier) noexcept;
    void clear_context_checkpoints(SequenceState& sequence) noexcept;
    void install_ram_context_checkpoints(SequenceState& sequence,
                                         const qwen3::detail::RamRestoredHost& host,
                                         std::uint64_t entry_id, std::uint32_t reuse_base,
                                         bool live_restore = false);
    void recycle_pending_disk_checkpoint_heads() noexcept;
    [[nodiscard]] bool staging_holds(std::uint32_t lane, qwen3::detail::PrefixHash128 hash,
                                     std::uint32_t frontier) const noexcept;
    [[nodiscard]] bool captures_context_checkpoints() const noexcept;
    void fence_staging_copies() noexcept;
    void unoccupy_staging() noexcept;
    void reload_turn_rollback_into_staging(std::uint32_t lane, qwen3::detail::PrefixHash128 hash,
                                           std::uint32_t frontier);

    struct ContextCheckpointStaging {
        bool occupied          = false;
        std::uint32_t lane     = 0;
        std::uint32_t frontier = 0;
        qwen3::detail::PrefixHash128 hash{};
        qwen3::detail::ContextCheckpointKind kind = qwen3::detail::ContextCheckpointKind::Ladder;
        // Set instead of occupied when the slot and DFlash staging lane hold `lane`'s rewrite
        // image at rewrite_generation, so a rewrite restore can copy them device-to-device.
        bool rewrite                     = false;
        std::uint64_t rewrite_generation = 0;
        hipEvent_t d2d_done              = nullptr;
        hipEvent_t copies_done           = nullptr;
    };

    ContextCheckpointStaging staging_;
    hipEvent_t copy_hold_cancel_fence_ = nullptr;
    // Startup-carved heads of one layout (GDN, staging hidden, DFlash cyclic); capacity is
    // context_checkpoint_image_pool_capacity and never grows.
    std::vector<ContextCheckpointHead> context_checkpoint_pool_;
};

} // namespace ninfer::targets::qwen3::detail::NINFER_QWEN3_RUNTIME_NS

namespace ninfer::targets::qwen3::detail {

template <>
class ProgramImpl<NINFER_QWEN3_VARIANT> final : public NINFER_QWEN3_RUNTIME_NS::ProgramImplCore {
public:
    using NINFER_QWEN3_RUNTIME_NS::ProgramImplCore::ProgramImplCore;
};

} // namespace ninfer::targets::qwen3::detail
