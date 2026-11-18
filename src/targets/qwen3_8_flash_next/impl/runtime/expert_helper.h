#pragma once

#include "core/expert_mailbox.h"
#include "runtime/expert_helper_placement.h"

#include <memory>
#include <string_view>

namespace ninfer::targets::qwen3_8_flash_next {
class LoadedModel;
struct ExpertSlotPlan;

struct ExpertHelperMemory {
    std::size_t free_at_start_bytes = 0, reserve_bytes = kHelperReserveBytes;
    std::size_t record_bytes = 0, nonrecord_device_bytes = 0;
    std::size_t cuda_device_bytes = 0, mailbox_host_bytes = 0;
    std::array<std::int32_t, 2> record_counts{};
};

struct ExpertHelperCudaView {
    ExpertMailboxView mailbox;
    const std::uint32_t* local_eligibility;
    float* local_partial;
    float* remote_partial;
    // Compact VRAM sources: partition writes these, then core owns the guarded mapped copy.
    std::uint16_t* compact_x;
    std::int32_t* compact_ids;
    float* compact_weights;
    std::int32_t* compact_map;
    std::uint32_t* compact_count;
};

// One immutable helper and one serialized round owner for this Program. Construct after State,
// DFlash and extents planning but before ExpertResidency/Forward. CUDA/HIP stream owners and the
// LoadedModel outlive it. Destruction cancels/joins before destroying graphs or any GPU storage.
// Internal activation is startup-only; flag-off construction never calls this class.
class ExpertHelper {
public:
    ExpertHelper(const LoadedModel& model, const ExpertSlotPlan& fixed_primary,
                 std::string_view pci, std::int32_t lanes, bool speculative, void* cuda_stream,
                 std::size_t primary_flex_slots);
    ~ExpertHelper();
    ExpertHelper(const ExpertHelper&)            = delete;
    ExpertHelper& operator=(const ExpertHelper&) = delete;
    [[nodiscard]] static std::size_t cuda_device_bytes(std::uint32_t max_tokens);
    [[nodiscard]] const ExpertSlotPlan& primary_plan() const noexcept;
    [[nodiscard]] const ExpertHelperPlacement& placement() const noexcept;
    [[nodiscard]] ExpertHelperCudaView cuda_view() const noexcept;
    [[nodiscard]] ExpertHelperMemory memory() const noexcept;
    // No CPU per-layer work. begin launches the pre-instantiated 48-layer HIP graph before CUDA A.
    // finish joins both streams and reports any first error before callers may license output.
    [[nodiscard]] std::uint64_t begin_round(std::uint32_t tokens);
    void finish_round();
    // Idle-boundary check used by every Forward entry point; poisoned state cannot be reused.
    void check_failure() const;
    void cancel_round() noexcept;
    // Host abort: joins both runtimes and retires execution tracking. Benign cancellation
    // leaves reusable idle storage; real failures remain visible through check_failure().
    void cancel_and_drain() noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace ninfer::targets::qwen3_8_flash_next
