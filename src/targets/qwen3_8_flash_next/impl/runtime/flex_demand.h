#pragma once

#include <algorithm>
#include <cstdint>

namespace ninfer::targets::qwen3_8_flash_next {

// The KV space the expert cache can use as flex slots, in whole extents (kv_extents.h). The
// demand is static for a Program: it depends only on the admission domain of the F3 pool, the
// fixed F3 slots and the extent geometry.
struct FlexDemand {
    // Admittable F3 records that cannot be resident outside flex slots.
    std::int64_t slots = 0;
    // Extents that hold those slots, bounded by the flex region.
    std::int64_t extents = 0;
    std::int64_t tokens  = 0;
};

// slots = max(0, admittable_f3_records - fixed_f3_slots): the admission domain is the set of F3
// records the residency may admit (every F3 record, or the helper placement's local subset), and
// NVFP4 records never use flex slots. extents = ceil(slots / slots_per_extent), capped at the
// region's `region_extents`, since a demand beyond the region is not satisfiable by KV;
// tokens = extents * extent_tokens. A region without slots (slots_per_extent == 0) has no demand.
[[nodiscard]] constexpr FlexDemand plan_flex_demand(std::int64_t admittable_f3_records,
                                                    std::int64_t fixed_f3_slots,
                                                    std::int64_t slots_per_extent,
                                                    std::int64_t region_extents,
                                                    std::int64_t extent_tokens) noexcept {
    FlexDemand demand;
    demand.slots = std::max<std::int64_t>(0, admittable_f3_records - fixed_f3_slots);
    if (slots_per_extent > 0) {
        demand.extents =
            std::min(region_extents, (demand.slots + slots_per_extent - 1) / slots_per_extent);
    }
    demand.tokens = demand.extents * extent_tokens;
    return demand;
}

} // namespace ninfer::targets::qwen3_8_flash_next
