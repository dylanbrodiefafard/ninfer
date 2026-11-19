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

enum class ExpertPlacementTier : std::uint8_t { Primary, Helper, Tail };

struct ExpertHelperPlacement {
    // Layer-major U32[48,16] bitsets, bit e%32 of word e/32: primary_mask marks P and
    // local_eligibility marks P union T, that is everything outside H.
    std::vector<std::uint32_t> primary_mask, local_eligibility;
    std::array<std::int32_t, 2> primary_counts{}, tail_counts{};
    std::array<std::size_t, 2> primary_bytes{}, tail_bytes{};
    std::uint64_t primary_routes = 0, tail_routes = 0;
    std::array<std::int32_t, 2> helper_counts{}, remaining_counts{};
    std::array<std::size_t, 2> helper_bytes{};
    std::array<std::size_t, 48> layer_bytes{};
    std::array<std::array<std::int32_t, 2>, 48> layer_counts{};
    std::size_t budget_bytes    = 0;
    std::uint64_t helper_routes = 0, total_routes = 0;
    bool full_coverage = false; // T is empty: P and H hold every record
    [[nodiscard]] ExpertPlacementTier tier(std::size_t index) const;
    [[nodiscard]] bool helper_owns(std::size_t index) const;
};

// Exact three-tier static placement from fixed (never flex) capacity. Counts and codecs are
// [48,512]. P: each codec's global hottest `fixed` pairs, including zeros, ties by lower
// layer-major index. H: the complement of P, whole if it fits the budget, otherwise admitted by
// decreasing count/record_bytes (ties lower index), skipping records that do not fit. T: every
// other record. Ratios use widened integer products. free_bytes is the helper's free VRAM and
// includes the 1 GiB reserve. Invalid shapes, codecs or capacities throw.
[[nodiscard]] ExpertHelperPlacement plan_expert_helper(std::span<const ops::MoeExpertCodec> codecs,
                                                       std::span<const std::uint64_t> counts,
                                                       std::array<std::int32_t, 2> fixed,
                                                       std::size_t free_bytes);

// Position-sensitive modular U64 checksum of an exact immutable record span, matching the raw
// HIP checksum kernel. Host loads use memcpy, not potentially unaligned typed dereferences.
[[nodiscard]] std::uint64_t expert_record_checksum(std::span<const std::byte> bytes);
} // namespace ninfer::targets::qwen3_8_flash_next
