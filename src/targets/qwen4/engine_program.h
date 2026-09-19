#pragma once

#include "targets/qwen4/native_runtime.h"
#include "runtime/contract/transient_region.h"
#include "runtime/contract/types.h"
#include "text/qwen/prepared_prompt.h"

namespace ninfer::targets::qwen4 {

[[nodiscard]] NativeRuntimeConfig native_runtime_config(const EngineOptions&,std::uint32_t kv_tokens);
[[nodiscard]] std::size_t engine_control_device_bytes(const NativeRuntimeConfig&);

struct RequestBasePlan {
    runtime::RequestPlanSummary value;
    runtime::ResolvedExecutionOptions options;
    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept { return value; }
};
struct RequestPlan {
    runtime::RequestPlanSummary value;
    runtime::ResolvedExecutionOptions options;
    std::uint32_t reuse = 0;
    NativeCheckpoint checkpoint = NativeCheckpoint::Retained;
    PrefixReusePath reuse_path = PrefixReusePath::AppendAtFrontier;
    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept { return value; }
};

// Concrete Qwen4 control owner. The common executor owns admission ordering/publication;
// NativeRuntime owns GPU mathematics and complete target state. This class binds those two
// contracts, without a second tokenizer, scheduler, sampling law or execution graph.
class EngineProgram {
public:
    EngineProgram(const NativeModelView&, const NativeRuntimeConfig&, const EngineOptions&,
                  DeviceContext&);
    ~EngineProgram();
    EngineProgram(const EngineProgram&) = delete;
    EngineProgram& operator=(const EngineProgram&) = delete;
    [[nodiscard]] RequestBasePlan plan_request_base(const text::qwen::PreparedPrompt&,
        const runtime::ResolvedExecutionOptions&);
    [[nodiscard]] RequestPlan plan_request_for_lane(std::uint32_t,
        const text::qwen::PreparedPrompt&, const RequestBasePlan&);
    [[nodiscard]] bool can_admit_lane(std::uint32_t,const RequestPlan&) const noexcept;
    [[nodiscard]] bool can_admit_lane_after_retained_eviction(std::uint32_t,
        const RequestPlan&) const noexcept;
    [[nodiscard]] bool can_admit_lane_after_releasing(std::uint32_t,const RequestPlan&,
        std::span<const std::uint32_t>) const noexcept;
    [[nodiscard]] runtime::AdmissionResources admission_capacity() const noexcept;
    [[nodiscard]] runtime::PrefillStepResult start_prefill_lane(std::uint32_t,
        text::qwen::PreparedPrompt&&,RequestPlan&&,runtime::TransientRegion,
        const text::qwen::OutputSession* = nullptr);
    [[nodiscard]] runtime::PrefillStepResult advance_prefill_lane(std::uint32_t);
    [[nodiscard]] runtime::BatchedGeneratedRound decode_batch(std::span<const std::uint32_t>,
        std::span<const runtime::RoundBudget>);
    void set_suppressed_tokens_lane(std::uint32_t,std::span<const TokenId>);
    void clear_suppressed_tokens_lane(std::uint32_t);
    void set_typical_cycle_reasoning_lane(std::uint32_t,bool);
    void resolve_prefill_lane(std::uint32_t,bool terminal);
    void resolve_pending_batch(std::span<const std::uint32_t>,
        std::span<const std::uint32_t>,std::span<const std::uint8_t>,
        std::span<const std::uint8_t>,std::span<const std::uint8_t> = {});
    void abort_lane(std::uint32_t) noexcept;
    void retain_lane(std::uint32_t);
    [[nodiscard]] bool revert_cancelled_prefill_lane(std::uint32_t);
    [[nodiscard]] bool has_retained_lane(std::uint32_t) const noexcept;
    [[nodiscard]] std::uint64_t retained_use_tick(std::uint32_t) const noexcept;
    void evict_retained_lane(std::uint32_t) noexcept;
    void synchronize_all();
    [[nodiscard]] GenerationTimings generation_timings_lane(std::uint32_t) const noexcept;
    [[nodiscard]] SpeculativeStats speculative_stats_lane(std::uint32_t) const noexcept;
    [[nodiscard]] std::uint32_t captured_context_checkpoint_tokens_lane(std::uint32_t) const noexcept;
    [[nodiscard]] std::uint32_t restored_context_checkpoint_tokens_lane(std::uint32_t) const noexcept;
    [[nodiscard]] MemorySummary memory_summary() const noexcept;
    void reset_memory_peaks() noexcept;
    [[nodiscard]] ScoreResult score(text::qwen::PreparedPrompt&&,RequestPlan&&,
        runtime::TransientRegion,ScoreOptions = {});
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

struct Package {
    static constexpr bool supports_host_kv_tiers = false;
    static constexpr std::string_view model_id = "qwen4/native-preview";
    static constexpr std::string_view target_key = "qwen4";
    using LoadedModel = LoadedNativeModel;
    using Frontend = text::qwen::Frontend;
    using PreparedPrompt = text::qwen::PreparedPrompt;
    using OutputSession = text::qwen::OutputSession;
    using RequestBasePlan = qwen4::RequestBasePlan;
    using RequestPlan = qwen4::RequestPlan;
    using Program = EngineProgram;
};

} // namespace ninfer::targets::qwen4
