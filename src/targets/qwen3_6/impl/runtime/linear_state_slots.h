#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

/** Qwen3.6's target-local mapping from a stable request lane to its device GDN state slot.
 *
 * Slots `[0,C)` hold each lane's current committed state. MTP/DFlash engines append one
 * Engine-wide staging slot `C`. Rewrite checkpoints are lane-owned host images, not slots.
 */
struct LinearStateSlots {
    [[nodiscard]] static std::int32_t state_slot_count(std::uint32_t max_concurrency,
                                                       bool staging_slot = false) {
        const std::uint32_t extra = staging_slot ? 1U : 0U;
        if (max_concurrency == 0 ||
            max_concurrency >
                static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) - extra) {
            throw std::invalid_argument("Qwen3.6 Linear Attention concurrency is invalid");
        }
        return static_cast<std::int32_t>(max_concurrency + extra);
    }

    [[nodiscard]] static std::int32_t staging_state_slot(std::uint32_t max_concurrency) {
        return state_slot_count(max_concurrency, true) - 1;
    }

    [[nodiscard]] static std::int32_t current_state_slot(std::uint32_t lane,
                                                         std::uint32_t max_concurrency) {
        if (lane >= max_concurrency) {
            throw std::out_of_range("Qwen3.6 Linear Attention lane is out of range");
        }
        return static_cast<std::int32_t>(lane);
    }
};

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
