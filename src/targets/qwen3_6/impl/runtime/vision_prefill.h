#pragma once
#include <ninfer/targets/qwen3_6/vision_control.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::targets::qwen3_6::detail {

struct VisionUseSpan {
    // MTP consumes the first visual embedding one column before Text does.
    std::uint32_t begin      = 0;
    std::uint32_t text_begin = 0;
    std::uint32_t end        = 0;
    std::uint32_t item_index = 0;
    std::size_t output_begin = 0;
};

struct VisionPrefillPlan {
    std::shared_ptr<const qwen3_6::VisionControl> control;
    std::vector<VisionUseSpan> uses;
};

struct VisionChunkSelection {
    std::uint32_t length = 0;
    std::optional<std::size_t> use_index;
};

// Keep the Text split at the first image column, matching a reused text-only
// prefix. Its preceding chunk still needs the image's embeddings for MTP's
// shifted final input. Between items, cap at the next shifted consumer because
// a chunk can reference only one item's embeddings.
[[nodiscard]] inline VisionChunkSelection select_vision_prefill_chunk(
    std::span<const VisionUseSpan> uses, std::uint32_t begin, std::uint32_t length) noexcept {
    std::uint32_t end = begin + length;
    std::optional<std::size_t> active;
    for (std::size_t index = 0; index < uses.size(); ++index) {
        const VisionUseSpan& use = uses[index];
        if (use.end <= begin) { continue; }
        if (use.begin >= end) { break; }
        if (!active) {
            if (begin < use.text_begin) {
                end = std::min(end, use.text_begin);
                if (use.begin < end) { active = index; }
                break;
            }
            active = index;
        } else {
            end = std::min(end, use.begin);
            break;
        }
    }
    return {end - begin, active};
}

} // namespace ninfer::targets::qwen3_6::detail
