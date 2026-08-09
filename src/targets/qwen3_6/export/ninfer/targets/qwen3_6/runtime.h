#pragma once

#include "ninfer/types.h"
#include "runtime/contract/transient_region.h"
#include "runtime/contract/types.h"
#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace ninfer {
struct DeviceContext;
}

namespace ninfer::targets::qwen3_6 {

enum class TextPhase {
    Prefill,
    Verify,
};

struct GraphExecutionProfile {
    std::uint32_t min            = 0;
    std::uint32_t max            = 0;
    std::uint32_t topology_class = 0;
};

namespace detail {
template <class Variant>
struct SequencePlanImpl;
template <class Variant>
struct RequestPlanImpl;
template <class Variant>
class ProgramImpl;
} // namespace detail

// These are the complete family execution types. Exact packages bind them to a private Variant;
// target selection remains outside this layer and happens once in the closed Engine registry.
template <class Variant>
class SequencePlan {
public:
    SequencePlan(SequencePlan&&) noexcept;
    SequencePlan& operator=(SequencePlan&&) noexcept;
    ~SequencePlan();

    SequencePlan(const SequencePlan&)            = delete;
    SequencePlan& operator=(const SequencePlan&) = delete;

    [[nodiscard]] std::uint32_t capacity() const noexcept;
    [[nodiscard]] std::uint32_t max_concurrency() const noexcept;
    [[nodiscard]] std::size_t device_reservation_bytes() const noexcept;
    [[nodiscard]] std::size_t workspace_capacity_bytes() const noexcept;
    [[nodiscard]] std::size_t request_transient_capacity_bytes() const noexcept;

public:
    // Family-private construction/storage seam; exact packages expose only the completed alias.
    explicit SequencePlan(std::unique_ptr<detail::SequencePlanImpl<Variant>> impl) noexcept;
    std::unique_ptr<detail::SequencePlanImpl<Variant>> impl_;

    template <class V>
    friend SequencePlan<V> plan_sequence(DeviceContext&, const EngineOptions&,
                                         typename V::WeightsProfile);
    template <class V>
    friend class detail::ProgramImpl;
};

template <class Variant>
class RequestPlan {
public:
    RequestPlan(RequestPlan&&) noexcept;
    RequestPlan& operator=(RequestPlan&&) noexcept;
    ~RequestPlan();

    RequestPlan(const RequestPlan&)            = delete;
    RequestPlan& operator=(const RequestPlan&) = delete;

    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept;

public:
    // Family-private construction/storage seam. This header is repository-internal; exact
    // packages expose only the completed alias and never inspect this pointer.
    explicit RequestPlan(std::unique_ptr<detail::RequestPlanImpl<Variant>> impl) noexcept;
    std::unique_ptr<detail::RequestPlanImpl<Variant>> impl_;
};

template <class Variant>
class Program {
public:
    ~Program() noexcept;

    Program(const Program&)            = delete;
    Program& operator=(const Program&) = delete;
    Program(Program&&)                 = delete;
    Program& operator=(Program&&)      = delete;

    [[nodiscard]] RequestPlan<Variant> plan_request(const PreparedPrompt& prompt,
                                                    const ExecutionOptions& options) const;
    [[nodiscard]] runtime::BeginResult begin(PreparedPrompt&& prompt, RequestPlan<Variant>&& plan,
                                             runtime::TransientRegion transient);
    [[nodiscard]] runtime::GeneratedRound decode_round(runtime::RoundBudget budget);

    // Engine-internal fixed-lane execution surface. The public Engine owns scheduling; Program
    // owns the target state images and executes one immutable ordinary batch membership.
    [[nodiscard]] RequestPlan<Variant> plan_request_for_lane(std::uint32_t lane,
                                                             const PreparedPrompt& prompt,
                                                             const ExecutionOptions& options);
    [[nodiscard]] bool can_admit_lane(std::uint32_t lane,
                                      const RequestPlan<Variant>& plan) const noexcept;
    [[nodiscard]] bool
    can_admit_lane_after_retained_eviction(std::uint32_t lane,
                                           const RequestPlan<Variant>& plan) const noexcept;
    [[nodiscard]] runtime::PrefillStepResult
    start_ordinary_prefill_lane(std::uint32_t lane, PreparedPrompt&& prompt,
                                RequestPlan<Variant>&& plan, runtime::TransientRegion transient);
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
    [[nodiscard]] GenerationTimings generation_timings() const noexcept;
    void reset_memory_peaks() noexcept;

private:
    explicit Program(std::unique_ptr<detail::ProgramImpl<Variant>> impl) noexcept;
    std::unique_ptr<detail::ProgramImpl<Variant>> impl_;

    template <class V>
    friend std::unique_ptr<Program<V>> create_program(const typename V::ModelView&,
                                                      typename V::WeightsProfile, SequencePlan<V>&&,
                                                      DeviceContext&);
};

template <class Variant>
[[nodiscard]] SequencePlan<Variant> plan_sequence(DeviceContext& device,
                                                  const EngineOptions& options,
                                                  typename Variant::WeightsProfile weights_profile);

template <class Variant>
[[nodiscard]] std::unique_ptr<Program<Variant>>
create_program(const typename Variant::ModelView& model,
               typename Variant::WeightsProfile weights_profile, SequencePlan<Variant>&& plan,
               DeviceContext& device);

} // namespace ninfer::targets::qwen3_6
