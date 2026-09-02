#pragma once
#include "targets/qwen3_6/impl/runtime/instance.h"
// Qwen3.6 family runtime implementation; instantiated only by exact variants.

#include "core/arena.h"
#include "core/device.h"
#include "core/gdn_replay_records.h"
#include "ninfer/ops/sampling.h"
#include "core/decode_graph.h"
#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include "targets/qwen3_6/impl/runtime/adaptive_draft.h"
#include "targets/qwen3_6/impl/runtime/context_checkpoint.h"
#include "targets/qwen3_6/impl/runtime/kv_ram_cache.h"
#include "targets/qwen3_6/impl/runtime/layouts.h"
#include "targets/qwen3_6/impl/runtime/dflash_context.h"
#include "targets/qwen3_6/impl/runtime/linear_state_slots.h"
#include "targets/qwen3_6/impl/runtime/prefix_identity.h"
#include "targets/qwen3_6/impl/runtime/text_context.h"
#include "targets/qwen3_6/impl/runtime/vision_context.h"
#include "targets/qwen3_6/impl/runtime/vision_prefill.h"

#include <cstdint>
#include <array>
#include <memory>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

using PreparedPromptData    = qwen3_6::PreparedPromptData;
using RewriteCheckpointKind = qwen3_6::RewriteCheckpointKind;
using RewriteCheckpointSpec = qwen3_6::RewriteCheckpointSpec;

using ReusePath = ninfer::PrefixReusePath;

[[nodiscard]] constexpr bool is_rewrite_checkpoint_restore(ReusePath path) noexcept {
    return qwen3_6::detail::is_rewrite_checkpoint_restore(path);
}

[[nodiscard]] constexpr bool is_complete_checkpoint_restore(ReusePath path) noexcept {
    return qwen3_6::detail::is_complete_checkpoint_restore(path);
}

[[nodiscard]] constexpr ReusePath restore_path(RewriteCheckpointKind kind) noexcept {
    return qwen3_6::detail::rewrite_restore_path(kind);
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

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS

namespace ninfer::targets::qwen3_6::detail {

template <>
struct RequestBasePlanImpl<NINFER_QWEN36_VARIANT> {
    runtime::RequestPlanSummary summary;
    ops::SamplingConfig sampling;
    std::uint32_t text_kv_page_entitlement    = 0;
    std::uint32_t backend_kv_page_entitlement = 0;
    std::shared_ptr<const qwen3_6::VisionControl> vision_control;
    std::size_t vision_transient_bytes = 0;
    std::optional<qwen3_6::RewriteCheckpointSpec> rewrite_checkpoint;
    bool allow_prefix_reuse = false;
    bool capture_context_checkpoint = false;
};

template <>
struct RequestPlanImpl<NINFER_QWEN36_VARIANT> {
    runtime::RequestPlanSummary summary;
    NINFER_QWEN36_RUNTIME_NS::ReusePath reuse = NINFER_QWEN36_RUNTIME_NS::ReusePath::FullReset;
    std::uint32_t reuse_base                  = 0;
    NINFER_QWEN36_RUNTIME_NS::MtpBridgeMode mtp_bridge =
        NINFER_QWEN36_RUNTIME_NS::MtpBridgeMode::None;
    bool prepare_mtp = false;
    std::optional<NINFER_QWEN36_RUNTIME_NS::VisionPrefillPlan> vision;
    NINFER_QWEN36_RUNTIME_NS::RewriteCheckpointAction rewrite_checkpoint_action =
        NINFER_QWEN36_RUNTIME_NS::RewriteCheckpointAction::Drop;
    std::optional<qwen3_6::RewriteCheckpointSpec> rewrite_checkpoint_capture;
    ops::SamplingConfig sampling;
    std::uint32_t text_kv_page_entitlement    = 0;
    std::uint32_t backend_kv_page_entitlement = 0;
    PrefixReuseSource reuse_source            = PrefixReuseSource::None;
    std::uint64_t ram_entry_id                = 0;
    bool capture_context_checkpoints          = false;
    bool capture_context_checkpoint           = false;
};

} // namespace ninfer::targets::qwen3_6::detail

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

using RequestPlanImpl     = qwen3_6::detail::RequestPlanImpl<Variant>;
using RequestBasePlanImpl = qwen3_6::detail::RequestBasePlanImpl<Variant>;

enum class PendingKind : std::uint8_t {
    None,
    Begin,
    Ordinary,
    Speculative,
};

struct PendingCandidate {
    PendingKind kind            = PendingKind::None;
    std::uint32_t base_E        = 0;
    std::uint32_t base_S        = 0;
    std::uint32_t prompt_tokens = 0;
    std::uint32_t produced      = 0;
    std::uint32_t drafted       = 0;
    std::uint32_t round_k       = 0;
    std::uint32_t verify_width  = 0;
    bool tree_verify            = false;
    qwen3_6::AdaptiveDraftState adaptive_before;
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

struct ContextCheckpointHead {
    std::uint32_t frontier = 0;
    qwen3_6::detail::PrefixHash128 hash{};
    qwen3_6::detail::ContextCheckpointKind kind = qwen3_6::detail::ContextCheckpointKind::Ladder;
    std::optional<PinnedHostBuffer> conv;
    std::optional<PinnedHostBuffer> recurrent;
    std::optional<PinnedHostBuffer> hidden;
    std::optional<PinnedHostBuffer> dflash;
    cudaEvent_t copies_done = nullptr;

    ContextCheckpointHead() = default;
    ContextCheckpointHead(const ContextCheckpointHead&)            = delete;
    ContextCheckpointHead& operator=(const ContextCheckpointHead&) = delete;
    ContextCheckpointHead(ContextCheckpointHead&& other) noexcept { *this = std::move(other); }
    ContextCheckpointHead& operator=(ContextCheckpointHead&& other) noexcept {
        if (this == &other) { return *this; }
        release();
        frontier     = other.frontier;
        hash         = other.hash;
        kind         = other.kind;
        conv         = std::move(other.conv);
        recurrent    = std::move(other.recurrent);
        hidden       = std::move(other.hidden);
        dflash       = std::move(other.dflash);
        copies_done  = other.copies_done;
        other.copies_done = nullptr;
        other.frontier    = 0;
        other.kind        = qwen3_6::detail::ContextCheckpointKind::Ladder;
        return *this;
    }
    ~ContextCheckpointHead() { release(); }

    void wait_copies() const {
        if (copies_done != nullptr) { CUDA_CHECK(cudaEventSynchronize(copies_done)); }
    }

    void release() noexcept {
        if (copies_done != nullptr) {
            (void)cudaEventSynchronize(copies_done);
            (void)cudaEventDestroy(copies_done);
            copies_done = nullptr;
        }
        conv.reset();
        recurrent.reset();
        hidden.reset();
        dflash.reset();
    }
};

struct ContextCheckpointIndex {
    std::uint32_t frontier = 0;
    qwen3_6::detail::PrefixHash128 hash{};
    qwen3_6::detail::ContextCheckpointKind kind = qwen3_6::detail::ContextCheckpointKind::Ladder;
};

struct SequenceKVBundle {
    PagedKVAllocation text;
    std::optional<PagedKVAllocation> backend;
};

struct DecodeGraphProfile {
    std::uint32_t batch_size             = 1;
    std::uint32_t min_execution_frontier = 0;
    std::uint32_t max_execution_frontier = 0;
    std::uint32_t topology_class         = 0;
    std::uint32_t draft_k                = 0;
    std::uint32_t verify_width           = 0;
    DecodeGraphDefinition definition;
};

struct DecodeGraphTopology {
    std::uint32_t topology_class = 0;
    DecodeGraphExecutable executable;
    std::optional<std::size_t> installed_profile;
};

struct DecodeGraphFamily {
    std::vector<DecodeGraphProfile> profiles;
    std::vector<DecodeGraphTopology> topologies;
};

// Target model continuation for one logical sequence. This state remains meaningful after the
// request which produced it has finished, so it is deliberately separate from request lifecycle,
// output, sampling, and round-control state.
struct SequenceState {
    std::optional<SequenceKVBundle> kv;
    Tensor tail_hidden;
    Tensor rewrite_checkpoint_hidden;
    std::uint32_t lane = 0;

    std::uint32_t execution_frontier = 0;
    std::uint32_t ledger_frontier    = 0;
    std::vector<TokenId> ledger;
    qwen3_6::detail::ResidentPrefixIdentity prefix_identity;
    std::int32_t rope_delta               = 0;
    std::uint32_t text_kv_valid           = 0;
    std::uint32_t mtp_kv_valid            = 0;
    std::uint32_t dflash_context_frontier = 0;
    std::array<TokenId, qwen3_6::kMtpDecodeMaximumDrafts> mtp_drafts{};
    std::uint32_t mtp_draft_count = 0;
    bool tail_hidden_valid        = false;
    bool retained                 = false;
    std::uint64_t use_tick        = 0;
    RewriteCheckpoint rewrite_checkpoint;
    std::vector<ContextCheckpointHead> context_checkpoints;
    std::uint32_t next_context_mark = 0;
};

// Request/round control is not retained with a reusable SequenceState. A later concurrent Engine
// gives every occupied request slot its own instance of this state.
struct RequestControl {
    Lifecycle lifecycle = Lifecycle::Empty;
    PendingCandidate pending;
    ops::SamplingConfig sampling_host;
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
        std::uint32_t base               = 0;
        std::uint32_t cursor             = 0;
        std::uint32_t prompt_tokens      = 0;
        std::uint32_t initial_mtp_extent = 0;
        double elapsed_seconds           = 0.0;
        bool host_input_consumed_pending = false;
        bool prepare_mtp                 = false;
        ReusePath reuse                  = ReusePath::FullReset;
        PrefixReuseSource reuse_source   = PrefixReuseSource::None;
        MtpBridgeMode mtp_bridge         = MtpBridgeMode::None;
        bool capture_context_checkpoints = false;
        // Per-chunk prefill step records (tokens processed, wall seconds) for the
        // tail-window throughput metric. One entry per advance_prefill call; the
        // finalizing call includes its sampling/bridge time.
        std::vector<std::uint32_t> step_tokens;
        std::vector<double> step_seconds;
    };

    std::optional<Prefill> prefill;
    qwen3_6::AdaptiveDraftState adaptive;
};

class ProgramImplCore {
public:
    ProgramImplCore(const LoadedModelData& model, const SequencePlanImpl& plan,
                    DeviceContext& device);
    ~ProgramImplCore() noexcept;

    [[nodiscard]] RequestBasePlan
    plan_request_base(const PreparedPromptData& prompt,
                      const runtime::ResolvedExecutionOptions& options);
    [[nodiscard]] RequestPlan plan_request_for_lane(std::uint32_t lane,
                                                    const PreparedPromptData& prompt,
                                                    const RequestBasePlan& base);
    [[nodiscard]] RequestPlan plan_ram_reuse(const PreparedPromptData& prompt,
                                             const RequestBasePlan& base);
    [[nodiscard]] bool can_admit_lane(std::uint32_t lane, const RequestPlan& plan) const noexcept;
    [[nodiscard]] bool
    can_admit_lane_after_retained_eviction(std::uint32_t lane,
                                           const RequestPlan& plan) const noexcept;
    [[nodiscard]] bool can_admit_lane_after_releasing(std::uint32_t lane, const RequestPlan& plan,
                                                      std::span<const std::uint32_t> release_lanes)
        const noexcept;
    [[nodiscard]] runtime::AdmissionResources admission_capacity() const noexcept;
    [[nodiscard]] runtime::PrefillStepResult start_prefill_lane(std::uint32_t lane,
                                                                PreparedPromptData&& prompt,
                                                                RequestPlan&& plan,
                                                                runtime::TransientRegion transient);
    [[nodiscard]] runtime::PrefillStepResult advance_prefill_lane(std::uint32_t lane);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_batch(std::span<const std::uint32_t> lanes,
                 std::span<const runtime::RoundBudget> budgets);
    void set_suppressed_tokens_lane(std::uint32_t lane, std::span<const TokenId> tokens);
    void clear_suppressed_tokens_lane(std::uint32_t lane);
    void resolve_prefill_lane(std::uint32_t lane, bool terminal);
    void resolve_pending_batch(std::span<const std::uint32_t> lanes,
                               std::span<const std::uint32_t> accepted_tokens,
                               std::span<const std::uint8_t> terminal,
                               std::span<const std::uint8_t> cancelled,
                               std::span<const std::uint8_t> rejected = {});
    void abort_lane(std::uint32_t lane) noexcept;
    void retain_lane(std::uint32_t lane);
    [[nodiscard]] bool revert_cancelled_prefill_lane(std::uint32_t lane);
    [[nodiscard]] bool has_retained_lane(std::uint32_t lane) const noexcept;
    [[nodiscard]] std::uint64_t retained_use_tick(std::uint32_t lane) const noexcept;
    void evict_retained_lane(std::uint32_t lane) noexcept;
    [[nodiscard]] bool capture_retained_lane(std::uint32_t lane);
    void restore_ram_entry(std::uint32_t lane, std::uint64_t entry_id, const RequestPlan& plan);
    void claim_ram_entry(std::uint64_t entry_id);
    void release_ram_entry(std::uint64_t entry_id);
    void consume_ram_entry(std::uint64_t entry_id);
    [[nodiscard]] qwen3_6::detail::KvRamSnapshot kv_ram_snapshot() const noexcept;
    qwen3_6::detail::KvRamCopySeconds harvest_kv_ram_copy_seconds();
    [[nodiscard]] bool kv_ram_copies_ready() const;
    void wait_kv_ram_copies_on_compute();
    void wait_kv_ram_copies();
    void synchronize_all();
    [[nodiscard]] std::uint64_t kv_ram_index_version() const noexcept;
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
    const std::uint32_t capacity;
    const std::uint32_t kv_capacity;
    const std::uint32_t max_concurrency;
    const std::uint32_t prefill_chunk;
    const std::uint32_t draft_window;
    const std::uint32_t dflash_verify_width;
    const bool adaptive_draft;
    const std::vector<std::uint32_t> captured_ks;
    const std::vector<float> adaptive_round_time;
    const SpeculativeBackend speculative_backend;
    const std::vector<std::uint32_t> context_marks;
    const DType kv_dtype;
    const std::int32_t kv_quant_group;
    const ProposalHead proposal_head;
    const float keep_frac;
    const float xattn_tau;
    const std::int32_t xattn_min_len;
    const bool vision_enabled;
    const bool use_cuda_graph;
    const std::size_t kv_payload_bytes;
    const std::size_t kv_ram_capacity_bytes;
    const std::size_t graph_allowance_bytes;
    std::size_t graph_observed_bytes = 0;
    const WorkspacePlan workspace_plan;

    DeviceArena persistent;
    DeviceArena workspace_storage;
    WorkspaceArena work;
    std::unique_ptr<qwen3_6::DecoderState> decoder;
    std::optional<GdnReplayRecords> replay_records;
    std::optional<DFlashPersistentState> dflash;
    qwen3_6::RoundState io;
    Tensor prefill_hidden;
    Tensor sampling_config;
    Tensor token_counts;
    Tensor tail_hidden_store;
    Tensor rewrite_checkpoint_hidden_store;
    Tensor staging_hidden;

    std::array<SequenceState, kMaximumConcurrency> sequences;
    std::array<RequestControl, kMaximumConcurrency> requests;

    DecodeGraphFamily ordinary_graphs;
    DecodeGraphFamily mtp_graphs;
    DecodeGraphFamily dflash_graphs;

    PinnedHostBuffer round_host;
    TokenId* host_tokens = nullptr;
    std::optional<PinnedHostBuffer> ordinary_host;
    qwen3_6::OrdinaryDecodeIngress* ordinary_host_ingress = nullptr;
    qwen3_6::OrdinaryDecodeEgress* ordinary_host_egress   = nullptr;
    std::optional<PinnedHostBuffer> mtp_host;
    qwen3_6::MtpDecodeIngress* mtp_host_ingress = nullptr;
    qwen3_6::MtpDecodeEgress* mtp_host_egress   = nullptr;
    std::optional<PinnedHostBuffer> dflash_host;
    qwen3_6::DFlashDecodeIngress* dflash_host_ingress = nullptr;
    qwen3_6::DFlashDecodeEgress* dflash_host_egress   = nullptr;

    std::size_t workspace_logical_peak_bytes = 0;
    std::optional<qwen3_6::detail::KVRamCache> kv_ram_cache_;
    std::uint64_t next_use_tick_ = 1;

private:
    void clear_lane(SequenceState& sequence, RequestControl& request) noexcept;
    void retain_committed_sequence(SequenceState& sequence, RequestControl& request);
    void ordered_reset(SequenceState& sequence);
    void prepare_graphs();
    void install_sampling(SequenceState& sequence, RequestControl& request,
                          const ops::SamplingConfig& config);
    void set_device_i32(Tensor& tensor, std::int32_t value);
    void copy_tail(SequenceState& sequence, const Tensor& source);
    void copy_round_token();
    void resolve_non_speculative_pending(SequenceState& sequence, RequestControl& request,
                                         std::uint32_t accepted_tokens, bool terminal);
    [[nodiscard]] runtime::PrefillStepResult advance_prefill(SequenceState& sequence,
                                                             RequestControl& request);
    void enqueue_dflash_context_append(std::span<const std::uint32_t> lanes,
                                       std::span<const std::uint32_t> starts,
                                       std::span<const std::uint32_t> counts);
    void validate_licensed_tokens(std::span<const TokenId> tokens) const;
    void mark_workspace_usage(std::size_t phase_bytes) noexcept;
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_ordinary_batch(std::span<const std::uint32_t> lanes,
                          std::span<const runtime::RoundBudget> budgets);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_mtp_batch(std::span<const std::uint32_t> lanes,
                     std::span<const runtime::RoundBudget> budgets);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_dflash_batch(std::span<const std::uint32_t> lanes,
                        std::span<const runtime::RoundBudget> budgets);
    void reserve_sequence_kv(SequenceState& sequence, std::uint32_t text_pages,
                             std::uint32_t backend_pages);
    void resize_sequence_kv_entitlement(SequenceState& sequence, std::uint32_t text_pages,
                                        std::uint32_t backend_pages);
    void bind_sequence_kv(SequenceState& sequence);
    void unbind_sequence_kv(SequenceState& sequence) noexcept;
    void materialize_sequence_kv(SequenceState& sequence, std::uint32_t main_tokens,
                                 std::uint32_t backend_tokens = 0);
    void trim_sequence_kv(SequenceState& sequence, std::uint32_t main_tokens,
                          std::uint32_t backend_tokens = 0);
    void release_sequence_growth_entitlement(SequenceState& sequence) noexcept;
    [[nodiscard]] qwen3_6::PagedKVCache* backend_kv_cache() noexcept;
    [[nodiscard]] const qwen3_6::PagedKVCache* backend_kv_cache() const noexcept;
    [[nodiscard]] std::uint32_t backend_kv_valid(const SequenceState& sequence) const noexcept;
    [[nodiscard]] qwen3_6::PagedKVCacheView text_kv_view(const SequenceState& sequence) const;
    [[nodiscard]] qwen3_6::PagedKVCacheView mtp_kv_view(const SequenceState& sequence) const;

    struct ResidentStateView {
        const std::vector<TokenId>* ledger                    = nullptr;
        const qwen3_6::detail::ResidentPrefixIdentity* identity = nullptr;
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
    [[nodiscard]] qwen3_6::detail::RamCaptureSource ram_capture_source(const SequenceState& sequence);
    void accumulate_prefill_nll(std::span<const TokenId> ids, std::uint32_t chunk_begin,
                                std::uint32_t chunk_tokens, std::uint32_t skip, ScoreResult& result);
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
                                     bool capture_enabled, bool request_pin);
    void restore_context_checkpoint_state(SequenceState& sequence, std::uint32_t base);
    void restore_dflash_cyclic_from_head(SequenceState& sequence, const ContextCheckpointHead& head);
    void snapshot_dflash_cyclic_to_staging(std::int32_t lane);
    void pack_dflash_cyclic_to_head(ContextCheckpointHead& head);
    [[nodiscard]] ContextCheckpointHead acquire_context_checkpoint_head(
        std::size_t conv_bytes, std::size_t recurrent_bytes, std::size_t hidden_bytes,
        std::size_t dflash_bytes);
    void record_context_checkpoint_head_use(ContextCheckpointHead& head, cudaStream_t stream);
    void recycle_context_checkpoint_head(ContextCheckpointHead&& head);
    void drop_context_checkpoints_after(SequenceState& sequence, std::uint32_t frontier) noexcept;
    void clear_context_checkpoints(SequenceState& sequence) noexcept;
    void install_ram_context_checkpoints(SequenceState& sequence,
                                         const qwen3_6::detail::RamRestoredHost& host);
    [[nodiscard]] bool staging_holds(std::uint32_t lane, qwen3_6::detail::PrefixHash128 hash,
                                     std::uint32_t frontier) const noexcept;
    [[nodiscard]] bool captures_context_checkpoints() const noexcept;
    void fence_staging_copies() noexcept;
    void unoccupy_staging() noexcept;
    void reload_turn_rollback_into_staging(std::uint32_t lane, qwen3_6::detail::PrefixHash128 hash,
                                           std::uint32_t frontier);

    struct ContextCheckpointStaging {
        bool occupied             = false;
        std::uint32_t lane        = 0;
        std::uint32_t frontier    = 0;
        qwen3_6::detail::PrefixHash128 hash{};
        qwen3_6::detail::ContextCheckpointKind kind =
            qwen3_6::detail::ContextCheckpointKind::Ladder;
        cudaEvent_t d2d_done    = nullptr;
        cudaEvent_t copies_done = nullptr;
    };
    ContextCheckpointStaging staging_;
    std::vector<ContextCheckpointHead> context_checkpoint_pool_;
};

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS

namespace ninfer::targets::qwen3_6::detail {

template <>
class ProgramImpl<NINFER_QWEN36_VARIANT> final : public NINFER_QWEN36_RUNTIME_NS::ProgramImplCore {
public:
    using NINFER_QWEN36_RUNTIME_NS::ProgramImplCore::ProgramImplCore;
};

} // namespace ninfer::targets::qwen3_6::detail
