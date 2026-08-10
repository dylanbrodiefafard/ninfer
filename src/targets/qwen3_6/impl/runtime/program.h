#pragma once
#include "targets/qwen3_6/impl/runtime/instance.h"
// Qwen3.6 family runtime implementation; instantiated only by exact variants.

#include "core/arena.h"
#include "ninfer/ops/sampling.h"
#include "core/decode_graph.h"
#include <ninfer/targets/qwen3_6/prepared_prompt.h>

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
#include <vector>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

using PreparedPromptData = qwen3_6::PreparedPromptData;

enum class ReusePath : std::uint8_t {
    FullReset,
    AppendAtFrontier,
    RestoreBoundary,
};

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS

namespace ninfer::targets::qwen3_6::detail {

template <>
struct RequestPlanImpl<NINFER_QWEN36_VARIANT> {
    runtime::RequestPlanSummary summary;
    NINFER_QWEN36_RUNTIME_NS::ReusePath reuse = NINFER_QWEN36_RUNTIME_NS::ReusePath::FullReset;
    std::uint32_t reuse_base                  = 0;
    bool needs_mtp_bridge                     = false;
    bool prepare_mtp                          = false;
    bool prepare_dflash                       = false;
    std::optional<NINFER_QWEN36_RUNTIME_NS::VisionPrefillPlan> vision;
    std::optional<std::uint32_t> snapshot_boundary;
    ops::SamplingConfig sampling;
    std::uint32_t text_kv_page_entitlement    = 0;
    std::uint32_t backend_kv_page_entitlement = 0;
};

} // namespace ninfer::targets::qwen3_6::detail

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

using RequestPlanImpl = qwen3_6::detail::RequestPlanImpl<Variant>;

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
};

enum class Lifecycle : std::uint8_t {
    Empty,
    Prefilling,
    Active,
    Pending,
    Resident,
    Invalid,
};

struct PrefixCheckpoint {
    bool valid             = false;
    std::uint32_t boundary = 0;
    bool hidden_valid      = false;
    bool mtp_prefix_valid  = false;
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
    std::optional<DFlashPersistentState> dflash;
    std::optional<SequenceKVBundle> kv;
    Tensor tail_hidden;
    Tensor boundary_hidden;
    std::uint32_t lane                 = 0;
    std::int32_t linear_state_base     = 0;
    std::int32_t linear_state_capacity = 0;

    std::uint32_t execution_frontier = 0;
    std::uint32_t ledger_frontier    = 0;
    std::vector<TokenId> ledger;
    qwen3_6::detail::ResidentPrefixIdentity prefix_identity;
    std::int32_t rope_delta                = 0;
    std::int32_t current_linear_state_slot = 0;
    std::uint32_t text_kv_valid            = 0;
    std::uint32_t mtp_kv_valid             = 0;
    std::uint32_t dflash_context_frontier  = 0;
    std::uint32_t dflash_proposal_frontier = 0;
    TokenId dflash_proposal_anchor         = 0;
    std::uint64_t request_epoch            = 0;
    std::uint64_t dflash_proposal_epoch    = 0;
    std::uint32_t pending_context_base     = 0;
    std::uint32_t pending_context_count    = 0;
    bool pending_context_valid             = false;
    bool dflash_boundary_valid             = false;
    std::uint32_t dflash_boundary_frontier = 0;
    bool ordinary_tail                     = false;
    bool drafts_ready                      = false;
    bool tail_hidden_valid                 = false;
    PrefixCheckpoint boundary;
};

// Request/round control is not retained with a reusable SequenceState. A later concurrent Engine
// gives every occupied request slot its own instance of this state.
struct RequestControl {
    Lifecycle lifecycle = Lifecycle::Empty;
    PendingCandidate pending;
    ops::SamplingConfig sampling_host;
    GenerationTimings timings;

    struct OrdinaryPrefill {
        PreparedPromptData prompt;
        std::optional<VisionPrefillPlan> vision_plan;
        std::unique_ptr<schedule::VisionPrefillSession> vision;
        runtime::TransientRegion transient;
        std::optional<std::uint32_t> snapshot_boundary;
        std::uint32_t base               = 0;
        std::uint32_t cursor             = 0;
        std::uint32_t prompt_tokens      = 0;
        double elapsed_seconds           = 0.0;
        bool host_input_consumed_pending = false;
    };

    std::optional<OrdinaryPrefill> ordinary_prefill;
};

class ProgramImplCore {
public:
    ProgramImplCore(const LoadedModelData& model, const SequencePlanImpl& plan,
                    DeviceContext& device);
    ~ProgramImplCore() noexcept;

    [[nodiscard]] RequestPlan plan_request(const PreparedPromptData& prompt,
                                           const ExecutionOptions& options) const;
    [[nodiscard]] runtime::BeginResult begin(PreparedPromptData&& prompt, RequestPlan&& plan,
                                             runtime::TransientRegion transient);
    [[nodiscard]] runtime::GeneratedRound decode_round(runtime::RoundBudget budget);

    [[nodiscard]] RequestPlan plan_request_for_lane(std::uint32_t lane,
                                                    const PreparedPromptData& prompt,
                                                    const ExecutionOptions& options);
    [[nodiscard]] bool can_admit_lane(std::uint32_t lane, const RequestPlan& plan) const noexcept;
    [[nodiscard]] bool
    can_admit_lane_after_retained_eviction(std::uint32_t lane,
                                           const RequestPlan& plan) const noexcept;
    [[nodiscard]] runtime::PrefillStepResult
    start_ordinary_prefill_lane(std::uint32_t lane, PreparedPromptData&& prompt, RequestPlan&& plan,
                                runtime::TransientRegion transient);
    [[nodiscard]] runtime::PrefillStepResult advance_ordinary_prefill_lane(std::uint32_t lane);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_ordinary_batch(std::span<const std::uint32_t> lanes,
                          std::span<const runtime::RoundBudget> budgets);
    void resolve_pending_lane(std::uint32_t lane, std::uint32_t accepted_tokens, bool terminal);
    void abort_lane(std::uint32_t lane) noexcept;
    [[nodiscard]] bool has_retained_lane(std::uint32_t lane) const noexcept;
    void evict_retained_lane(std::uint32_t lane) noexcept;
    [[nodiscard]] GenerationTimings generation_timings_lane(std::uint32_t lane) const noexcept;

    void resolve_pending(std::uint32_t accepted_tokens, bool terminal);

    void finish_active();
    void abort_request() noexcept;

    [[nodiscard]] std::uint32_t materialized_tokens() const noexcept;
    [[nodiscard]] MemorySummary memory_summary() const noexcept;
    [[nodiscard]] SpeculativeStats speculative_stats() const;

    [[nodiscard]] GenerationTimings generation_timings() const noexcept {
        return active_request().timings;
    }

    void reset_memory_peaks() noexcept;

    const LoadedModelData& model;
    DeviceContext& device;
    const std::uint32_t capacity;
    const std::uint32_t max_concurrency;
    const std::uint32_t prefill_chunk;
    const std::uint32_t draft_window;
    const SpeculativeBackend speculative_backend;
    const DType kv_dtype;
    const std::int32_t kv_quant_group;
    const ProposalHead proposal_head;
    const bool vision_enabled;
    const bool use_cuda_graph;
    const std::size_t kv_payload_bytes;
    const std::size_t graph_allowance_bytes;
    const WorkspacePlan workspace_plan;

    DeviceArena persistent;
    DeviceArena workspace_storage;
    WorkspaceArena work;
    std::unique_ptr<qwen3_6::DecoderState> decoder;
    qwen3_6::RoundState io;
    Tensor prefill_hidden;
    Tensor sampling_config;
    Tensor token_counts;
    Tensor tail_hidden_store;
    Tensor boundary_hidden_store;

    std::array<SequenceState, kMaximumConcurrency> sequences;
    std::array<RequestControl, kMaximumConcurrency> requests;

    DecodeGraphFamily ordinary_graphs;
    DecodeGraphFamily ordinary_aligned_graphs;
    DecodeGraphFamily mtp_graphs;
    DecodeGraphFamily dflash_initial_graphs;
    DecodeGraphFamily dflash_steady_graphs;

    PinnedHostBuffer round_host;
    std::int32_t* host_count = nullptr;
    TokenId* host_tokens     = nullptr;
    PinnedHostBuffer ordinary_host;
    qwen3_6::OrdinaryDecodeIngress* ordinary_host_ingress = nullptr;
    qwen3_6::OrdinaryDecodeEgress* ordinary_host_egress   = nullptr;

    std::size_t workspace_logical_peak_bytes = 0;

private:
    [[nodiscard]] SequenceState& active_sequence() noexcept { return sequences[active_lane_]; }

    [[nodiscard]] const SequenceState& active_sequence() const noexcept {
        return sequences[active_lane_];
    }

    [[nodiscard]] RequestControl& active_request() noexcept { return requests[active_lane_]; }

    [[nodiscard]] const RequestControl& active_request() const noexcept {
        return requests[active_lane_];
    }

    std::uint32_t active_lane_ = 0;
    void make_invalid() noexcept;
    void ordered_reset();
    void prepare_graphs();
    void install_sampling(const ops::SamplingConfig& config);
    void set_device_i32(Tensor& tensor, std::int32_t value);
    void copy_tail(const Tensor& source);
    void copy_round_token();
    [[nodiscard]] runtime::PrefillStepResult advance_ordinary_prefill();
    void flush_dflash_context_prefix(std::uint32_t count);
    void validate_licensed_tokens(std::span<const TokenId> tokens) const;
    void mark_workspace_usage(std::size_t phase_bytes) noexcept;
    void reserve_sequence_kv(std::uint32_t text_pages, std::uint32_t backend_pages);
    void resize_sequence_kv_entitlement(std::uint32_t text_pages, std::uint32_t backend_pages);
    void bind_sequence_kv();
    void unbind_sequence_kv() noexcept;
    void materialize_sequence_kv(std::uint32_t main_tokens, std::uint32_t backend_tokens = 0);
    void trim_sequence_kv(std::uint32_t main_tokens, std::uint32_t backend_tokens = 0);
    void release_sequence_growth_entitlement() noexcept;
    [[nodiscard]] qwen3_6::PagedKVCache* backend_kv_cache() noexcept;
    [[nodiscard]] const qwen3_6::PagedKVCache* backend_kv_cache() const noexcept;
    [[nodiscard]] std::uint32_t backend_kv_valid() const noexcept;
    [[nodiscard]] qwen3_6::PagedKVCacheView text_kv_view() const;
    [[nodiscard]] qwen3_6::PagedKVCacheView mtp_kv_view() const;
    [[nodiscard]] qwen3_6::PagedKVCacheView dflash_full_kv_view() const;
};

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS

namespace ninfer::targets::qwen3_6::detail {

template <>
class ProgramImpl<NINFER_QWEN36_VARIANT> final : public NINFER_QWEN36_RUNTIME_NS::ProgramImplCore {
public:
    using NINFER_QWEN36_RUNTIME_NS::ProgramImplCore::ProgramImplCore;
};

} // namespace ninfer::targets::qwen3_6::detail
