#pragma once

#include "ninfer/ops/moe_expert_record.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::targets::qwen3_8_flash_next {
inline constexpr std::size_t kHelperReserveBytes       = std::size_t{1} << 30;
inline constexpr std::size_t kHelperUploadStagingBytes = std::size_t{16} << 20;

struct ExpertHelperPlacement {
    // Layer-major U32[48,16], bit e%32 set iff the expert remains local/eligible.
    std::vector<std::uint32_t> local_eligibility;
    std::array<std::int32_t, 2> helper_counts{}, remaining_counts{};
    std::array<std::size_t, 2> helper_bytes{};
    std::array<std::size_t, 48> layer_bytes{};
    std::array<std::array<std::int32_t, 2>, 48> layer_counts{};
    std::size_t budget_bytes    = 0;
    std::uint64_t helper_routes = 0, total_routes = 0;
    bool full_coverage = false;
    [[nodiscard]] bool helper_owns(std::size_t index) const;
};

// Exact static placement from fixed (never flex) capacity. Counts and codecs are [48,512].
// Protect each codec's global hottest F pairs including zeros, then admit its complement, all if
// it fits or by decreasing count/bytes (ties lower index). Ratios use widened integer products.
// free_helper_bytes includes the 1 GiB reserve. Invalid shapes/codecs/capacities throw.
[[nodiscard]] ExpertHelperPlacement plan_expert_helper(std::span<const ops::MoeExpertCodec> codecs,
                                                       std::span<const std::uint64_t> counts,
                                                       std::array<std::int32_t, 2> fixed,
                                                       std::size_t free_bytes);

// Position-sensitive modular U64 checksum of an exact immutable record span, matching the raw
// HIP checksum kernel. Host loads use memcpy, not potentially unaligned typed dereferences.
[[nodiscard]] std::uint64_t expert_record_checksum(std::span<const std::byte> bytes);
} // namespace ninfer::targets::qwen3_8_flash_next
