#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// One slot pool as the kernels see it: fixed slot s < fixed_count lives at base + s *
// record_bytes, flex slot s >= fixed_count at flex_address[s - fixed_count]. count (fixed plus
// flex) == 0 marks a pool without slots, whose pointers are null; flex_address is null when
// count == fixed_count.
struct MoeExpertSlotPoolArgs {
    std::uint64_t base                = 0;
    std::uint64_t record_bytes        = 0;
    const std::uint64_t* flex_address = nullptr;
    std::int32_t* owner               = nullptr;
    std::uint8_t* referenced          = nullptr;
    std::int32_t fixed_count          = 0;
    std::int32_t count                = 0;
};

// The replacement policy of one admit: reference counters capped at reference_limit (1 = CLOCK
// bits) and, when counts is not null, the frequency state over `layers` layers (counts I32
// [512,layers], lookups I32 [1]); filter applies the frequency admission filter (requires counts).
struct MoeAdmitPolicyArgs {
    std::int32_t reference_limit = 1;
    bool filter                  = false;
    std::int32_t* counts         = nullptr;
    std::int32_t* lookups        = nullptr;
    std::int32_t halving_lookups = 0;
    std::int32_t layers          = 0;
};

// The launchers trust the wrapper's validation (shapes, alignment, disjointness, layer range).
// The tables are [512,L] layer-major (slot_index I32, the others U64); id_count is 10 * T;
// counters is null or the I64 [2,2] (kind, codec) counters, summed as U64; eligible is null or
// the layer's U32 [16] eligibility bitset. route_counts is null or I64 [512,L], incremented once
// per routed-set entry before eligibility filtering.
void moe_expert_cache_admit_launch(const std::int32_t* ids, std::int32_t id_count,
                                   std::uint64_t* expert_ptr, const std::uint64_t* host_ptr,
                                   std::uint64_t* fill_ptr, std::int32_t* slot_index,
                                   MoeExpertSlotPoolArgs pool0, MoeExpertSlotPoolArgs pool1,
                                   std::int32_t* clock_hands, std::int32_t layer,
                                   MoeAdmitPolicyArgs policy, std::uint64_t* counters,
                                   const std::uint32_t* eligible, std::uint64_t* route_counts,
                                   cudaStream_t stream);

// Whether moe_expert_cache_lend_launch borrows (owner -2, evicting) or returns (owner -1).
enum class MoeCacheLend : std::uint8_t { Borrow, Return };

// Lends slots [first, first + count) of `pool`, an absolute slot range. expert_ptr, host_ptr
// and slot_index are the tables a borrow evicts through, null for a return.
void moe_expert_cache_lend_launch(MoeExpertSlotPoolArgs pool, std::uint64_t* expert_ptr,
                                  const std::uint64_t* host_ptr, std::int32_t* slot_index,
                                  std::int32_t first, std::int32_t count, MoeCacheLend lend,
                                  cudaStream_t stream);

// source_ptr names mapped host or device records (codec in bit 0); blocks is in [1, 1024].
void moe_expert_cache_fill_launch(const std::uint64_t* source_ptr, const std::uint64_t* fill_ptr,
                                  std::int32_t layer, std::int32_t blocks, cudaStream_t stream);

// expert_ptr and host_ptr are the layer's columns; window_base is 0 when window_bytes is 0.
void moe_expert_cache_window_launch(const std::uint64_t* expert_ptr, const std::uint64_t* host_ptr,
                                    std::uint64_t window_base, std::uint64_t window_bytes,
                                    std::uint64_t* window_ptr, cudaStream_t stream);

// expert_ptr and host_ptr are the layer's columns; predicted is I32 [10,tokens]; staging_base is 0
// when budget_bytes is 0 and nothing can be placed; eligible is null or the layer's U32 [16]
// eligibility bitset.
void moe_expert_cache_stage_launch(const std::int32_t* predicted, std::int32_t tokens,
                                   std::int32_t candidates, const std::uint64_t* expert_ptr,
                                   const std::uint64_t* host_ptr, std::uint64_t staging_base,
                                   std::uint64_t budget_bytes, std::uint64_t* stage_ptr,
                                   std::uint64_t* source, const std::uint32_t* eligible,
                                   cudaStream_t stream);

// fill_ptr is the layer's column.
void moe_expert_cache_defer_launch(const std::uint64_t* fill_ptr, const std::uint64_t* stage_ptr,
                                   std::uint64_t* deferred, cudaStream_t stream);

void moe_expert_cache_commit_launch(std::uint64_t* expert_ptr, std::uint64_t* fill_ptr,
                                    std::int32_t layer, cudaStream_t stream);

} // namespace ninfer::ops::detail
