#pragma once

// Qwen3.8-Flash-Next routed-expert residency (docs/maintainer/qwen3.8-flash-next-plan.md, "Expert
// residency design"): every routed-expert record in device-mapped pinned host memory, a VRAM slot
// cache of two record size classes, and the device tables that ops::moe_expert_cache_admit/commit
// and ops::moe_experts read. Replacement runs on the device inside those Ops; this file owns the
// storage, its sizing and the initial state.

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/moe_expert_cache.h"
#include "ninfer/ops/moe_experts.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::targets::qwen3_8_flash_next {

inline constexpr std::int32_t kExpertLayers  = 48;
inline constexpr std::int32_t kRoutedExperts = ops::kMoeExperts;

// Bytes of one pinned host chunk of the production store: 13 chunks for the 50.3 GB store.
inline constexpr std::size_t kExpertHostChunkBytes = std::size_t{4} << 30;

/**
 * Pinned, device-mapped host memory holding one record per routed (layer, expert).
 *
 * codecs gives the codec of every expert, layer-major ([layer * 512 + expert]); its size is
 * layers * 512 with 1 <= layers <= 48. Records of ops::moe_expert_record_bytes(codec) bytes are
 * packed in that order, at 256-byte aligned offsets, into HostPinnedArena chunks of at most
 * chunk_bytes (a chunk always holds at least one record). Records start zeroed. Invalid arguments
 * throw std::invalid_argument; allocation and CUDA registration failures propagate.
 *
 * The binder fills each record once with write_record before device work reads it; concurrent
 * write_record calls for distinct records are safe. Records are then immutable: moe_experts reads
 * them through device_addresses() while the store lives. Moving the store keeps every address.
 */
class ExpertHostStore {
public:
    ExpertHostStore(std::span<const ops::MoeExpertCodec> codecs, std::size_t chunk_bytes);

    ExpertHostStore(const ExpertHostStore&)            = delete;
    ExpertHostStore& operator=(const ExpertHostStore&) = delete;
    ExpertHostStore(ExpertHostStore&&) noexcept        = default;
    ExpertHostStore& operator=(ExpertHostStore&&)      = default;
    ~ExpertHostStore()                                 = default;

    // Copies record, whose size must equal the expert's record bytes, into the store. Throws
    // std::out_of_range or std::invalid_argument.
    void write_record(std::int32_t layer, std::int32_t expert, std::span<const std::byte> record);

    [[nodiscard]] std::span<const std::byte> record(std::int32_t layer, std::int32_t expert) const;
    [[nodiscard]] ops::MoeExpertCodec codec(std::int32_t layer, std::int32_t expert) const;

    [[nodiscard]] std::int32_t layers() const noexcept { return layers_; }

    [[nodiscard]] std::size_t chunk_count() const noexcept { return chunks_.size(); }

    // Every record's device-mapped address with its codec in bit 0, layer-major.
    [[nodiscard]] std::span<const std::uint64_t> device_addresses() const noexcept {
        return device_addresses_;
    }

    // Records of each codec, indexed by ops::MoeExpertCodec.
    [[nodiscard]] std::array<std::int32_t, ops::kMoeCacheCodecTags> codec_counts() const noexcept {
        return codec_counts_;
    }

private:
    [[nodiscard]] std::size_t index(std::int32_t layer, std::int32_t expert) const;

    std::int32_t layers_ = 0;
    std::vector<HostPinnedArena> chunks_;
    std::vector<ops::MoeExpertCodec> codecs_;
    std::vector<std::byte*> records_;
    std::vector<std::uint64_t> device_addresses_;
    std::array<std::int32_t, ops::kMoeCacheCodecTags> codec_counts_{};
};

// Slots of each pool, indexed by ops::MoeExpertCodec.
struct ExpertSlotPlan {
    std::array<std::int32_t, ops::kMoeCacheCodecTags> slots{};

    [[nodiscard]] std::size_t bytes() const noexcept;
};

// Share of the slot budget given to the NVFP4 pool, in percent. The plan's cache simulation on
// recorded decode routing (two LRU pools, fill on miss) found PCIe time per step flat for 600 to
// 1,200 NVFP4 slots at both 20 GB and 23.4 GB, and worse when the pool holds every NVFP4 expert
// (effectively pinning them); 11% of the budget lands inside that optimum at both sizes.
inline constexpr std::size_t kNvfp4BudgetPercent = 11;

/**
 * Splits a VRAM byte budget between the two slot pools: the NVFP4 pool takes kNvfp4BudgetPercent
 * of the budget, F3 slots take the rest, and NVFP4 slots then grow into budget the F3 pool cannot
 * use. No pool exceeds its expert count. For 20 GB and the chosen store this is 795 NVFP4 and
 * 8,915 F3 slots. Exact; never throws.
 */
[[nodiscard]] ExpertSlotPlan
plan_expert_slots(std::size_t budget_bytes,
                  std::array<std::int32_t, ops::kMoeCacheCodecTags> expert_counts);

// Replacement policy, from replaying lane-batched decode traces through the exact admit model
// (tools/parity/qwen3_8_flash_next/replay_cache.py; docs/maintainer/qwen3.8-flash-next-plan.md,
// "Expert admission policy"): GCLOCK counters capped at 8 save 4-11% of miss bytes against CLOCK
// bits at C = 1..6, and the frequency filter, with counts halved every 8 x (total slots) lookups,
// saves a further 2-8% at C >= 2 (16% at C = 4 and 6 together) but costs about 10% at C = 1, so
// the Forward applies it to decode rounds of two or more lanes only.
inline constexpr std::int32_t kExpertReferenceLimit  = 8;
inline constexpr std::int64_t kFrequencyHalvingSlots = 8;

// AllEligible warms zero-count eligible records into fixed slots only; it requires every
// eligible record of each codec to fit. Calibrated preserves ordinary fixed+flex warm ordering.
enum class ExpertWarmStart : std::uint8_t { Calibrated, AllEligible };

/**
 * The device expert cache over an ExpertHostStore of L layers: the slot pools of plan, their CLOCK
 * state, and the tables (I64 holding U64 addresses, column l = layer l)
 *
 *   expert_table [512,L]: the address moe_experts reads each expert from (slot or mapped host
 *                         record, codec in bit 0); column l is that layer's MoeExpertBank table;
 *   host_table   [512,L]: every record's tagged mapped host address (read only);
 *   fill_table   [512,L]: pending fills of the current layer call (0 between calls), which
 *                         moe_expert_cache_fill copies and moe_expert_cache_commit publishes.
 *
 * Each table starts 4096-byte aligned, so every layer column (4096 bytes) is too. cache() carries
 * the slot index I32 [512,L] (the inverse of the owners, which the Ops maintain),
 * reference_limit kExpertReferenceLimit and the frequency state (counts I32 [512,L], zeroed,
 * halved every kFrequencyHalvingSlots x (total slots) lookups).
 *
 * The F3 pool has plan.slots[F3] fixed slots in its own allocation and, after them, one flex slot
 * per entry of flex_addresses (ops::MoeExpertSlotPool): device addresses of F3-record-sized,
 * 256-byte aligned, pairwise disjoint memory that another owner lends (KvExpertExtents, pages of
 * the shared QSA pool). The memory must be idle during construction and stay valid for this
 * object's lifetime; its owner borrows and returns the flex slots through the Ops. The NVFP4 pool
 * has fixed slots only. The frequency halving period counts every slot, fixed and flex.
 *
 * calib_counts is empty or holds one calibration routing count per expert, layer-major. Warm
 * start: each pool receives its codec's experts with nonzero count in descending count order
 * (ties to the lower layer-major index), as many as it has slots, fixed and flex. The hottest
 * takes the last slot (the highest flex slot, which KvExpertExtents takes back last) and the
 * coldest admitted one follows the empty slots. A nonempty local_eligibility U32 [L,16]
 * excludes zero bits from warm start. AllEligible instead warms every eligible record, including
 * zero counts, into fixed slots (not reclaimable flex); every eligible codec must fit its fixed
 * pool. Every reference bit and hand is 0, so CLOCK first
 * reclaims empty slots, then the coldest records. Without counts every slot starts empty and every
 * entry names the host record. The constructor copies the warm records on stream and synchronizes
 * it; it throws std::invalid_argument for invalid arguments and propagates CUDA failures.
 *
 * The store must outlive this object. The views stay valid for its lifetime; the Program passes
 * them to the moe_expert_cache Ops and moe_experts, which own every later state change.
 */
class ExpertResidency {
public:
    ExpertResidency(const ExpertHostStore& store, const ExpertSlotPlan& plan,
                    std::span<const std::uint64_t> flex_addresses,
                    std::span<const std::uint64_t> calib_counts, cudaStream_t stream,
                    std::span<const std::uint32_t> local_eligibility = {},
                    ExpertWarmStart warm_start                       = ExpertWarmStart::Calibrated);

    ExpertResidency(const ExpertResidency&)            = delete;
    ExpertResidency& operator=(const ExpertResidency&) = delete;
    ExpertResidency(ExpertResidency&&) noexcept        = default;
    ExpertResidency& operator=(ExpertResidency&&)      = default;
    ~ExpertResidency()                                 = default;

    [[nodiscard]] ops::MoeExpertCache cache();
    [[nodiscard]] Tensor expert_table();
    [[nodiscard]] Tensor host_table();
    [[nodiscard]] Tensor fill_table();

    // The fixed slots of each pool.
    [[nodiscard]] const ExpertSlotPlan& plan() const noexcept { return plan_; }

    // The F3 pool's flex slots.
    [[nodiscard]] std::int32_t flex_slots() const noexcept { return flex_slots_; }

private:
    std::int32_t layers_ = 0;
    ExpertSlotPlan plan_;
    std::int32_t flex_slots_ = 0;
    std::array<DeviceBuffer, ops::kMoeCacheCodecTags> slots_;
    // Tables (each layer column 4096-byte aligned), the slot index, hands, the flex address table,
    // and per-pool owner and reference arrays (fixed and flex slots), carved once from metadata_.
    DeviceArena metadata_;
    DeviceSpan expert_table_;
    DeviceSpan host_table_;
    DeviceSpan fill_table_;
    DeviceSpan slot_index_;
    DeviceSpan clock_hands_;
    DeviceSpan frequency_counts_;
    DeviceSpan frequency_lookups_;
    DeviceSpan flex_address_;
    std::array<DeviceSpan, ops::kMoeCacheCodecTags> owner_;
    std::array<DeviceSpan, ops::kMoeCacheCodecTags> referenced_;
};

} // namespace ninfer::targets::qwen3_8_flash_next
