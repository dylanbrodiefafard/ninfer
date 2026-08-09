#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

/** Qwen3.6's target-local mapping from semantic Linear Attention roles to flat pool slots. */
struct LinearStateSlots {
    [[nodiscard]] static std::int32_t required_slot_count(std::uint32_t draft_window) {
        if (draft_window >
            static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max() - 2)) {
            throw std::overflow_error("Qwen3.6 Linear Attention slot count exceeds int32");
        }
        return static_cast<std::int32_t>(draft_window) + 2;
    }

    [[nodiscard]] static constexpr std::int32_t
    prefill_working_slot(std::int32_t base = 0) noexcept {
        return base;
    }

    [[nodiscard]] static constexpr std::int32_t
    verify_snapshot_base_slot(std::int32_t base = 0) noexcept {
        return base;
    }

    [[nodiscard]] static std::int32_t prefix_boundary_slot(std::int32_t base,
                                                           std::int32_t slot_count) {
        if (slot_count < 2) {
            throw std::invalid_argument("Qwen3.6 Linear Attention requires a boundary slot");
        }
        return base + slot_count - 1;
    }

    [[nodiscard]] static std::int32_t prefix_boundary_slot(std::int32_t slot_count) {
        return prefix_boundary_slot(0, slot_count);
    }

    [[nodiscard]] static std::int32_t committed_snapshot_slot(std::uint32_t committed_extent,
                                                              std::int32_t base,
                                                              std::int32_t slot_count) {
        if (committed_extent == 0 || committed_extent >= static_cast<std::uint32_t>(slot_count)) {
            throw std::out_of_range(
                "Qwen3.6 committed Linear Attention extent exceeds snapshot slots");
        }
        return verify_snapshot_base_slot(base) + static_cast<std::int32_t>(committed_extent) - 1;
    }

    [[nodiscard]] static std::int32_t committed_snapshot_slot(std::uint32_t committed_extent,
                                                              std::int32_t slot_count) {
        return committed_snapshot_slot(committed_extent, 0, slot_count);
    }
};

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
