#pragma once

#include <ninfer/targets/qwen3_6/prepared_prompt.h>
#include "targets/qwen3_6/impl/runtime/vision_prefill.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace ninfer::targets::qwen3_6::detail {

inline constexpr std::uint32_t kPrefillChunkAlignment = 128;
inline constexpr std::uint32_t kIrregularPrefillSplit = 4096;
// Mixed-forward widths are whole 256-token activation scale tiles, where the W4A4 projections
// run full TMA tiles.
inline constexpr std::uint32_t kMixedForwardAlignment = 256;

// Large aligned extents use the full 8192-token workspace efficiently. A large
// unaligned tail uses 4096 first so only the smaller final unit pays the tail cost.
[[nodiscard]] inline std::uint32_t select_prefill_chunk(std::uint32_t remaining,
                                                        std::uint32_t maximum) noexcept {
    const std::uint32_t nominal = std::min(remaining, maximum);
    if (maximum > kIrregularPrefillSplit && nominal > kIrregularPrefillSplit &&
        nominal % kPrefillChunkAlignment != 0) {
        return kIrregularPrefillSplit;
    }
    return nominal;
}

[[nodiscard]] inline std::uint64_t prefill_chunk_count(std::uint32_t tokens,
                                                       std::uint32_t maximum) noexcept {
    std::uint64_t count = 0;
    while (tokens != 0) {
        tokens -= select_prefill_chunk(tokens, maximum);
        ++count;
    }
    return count;
}

[[nodiscard]] inline std::uint32_t
cap_prefill_at_frontiers(std::uint32_t begin, std::uint32_t length,
                         std::span<const std::uint32_t> frontiers) noexcept {
    // PreparedPrompt stores these in rendered turn order, hence token order.
    const auto next = std::upper_bound(frontiers.begin(), frontiers.end(), begin);
    return next == frontiers.end() ? length : std::min(length, *next - begin);
}

// Admission uses the execution chunk policy, including the irregular-tail rule.
// Adding one quantum per split is insufficient: splitting an aligned 8192-token
// prompt at 100 produces 100 + 4096 + 3996, rather than two execution steps.
[[nodiscard]] inline std::uint64_t
projected_prefill_work(const PreparedPromptData& prompt, std::uint32_t reuse_base,
                       std::uint32_t maximum, std::span<const VisionUseSpan> vision_uses,
                       std::optional<std::uint32_t> rewrite_frontier) noexcept {
    const auto tokens = static_cast<std::uint32_t>(prompt.token_ids.size());
    if (tokens == reuse_base) { return 1; }
    std::uint64_t result = 0;
    for (std::uint32_t begin = reuse_base; begin < tokens;) {
        auto length = select_prefill_chunk(tokens - begin, maximum);
        if (rewrite_frontier && *rewrite_frontier > begin) {
            length = std::min(length, *rewrite_frontier - begin);
        }
        if (prompt.has_media()) {
            length = cap_prefill_at_frontiers(begin, length, prompt.turn_closure_frontiers);
        }
        length = select_vision_prefill_chunk(vision_uses, begin, length).length;
        begin += length;
        ++result;
    }
    return result;
}

} // namespace ninfer::targets::qwen3_6::detail
