#pragma once

#include "core/arena.h"

#include <cstdint>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace ninfer::ops {

/**
 * Non-owning signed NVFP4 G16 activation. Codes pack consecutive logical values, low nibble
 * first, in token-major order. For represented BF16 v and positive finite divisor d:
 * scale = E4M3_satfinite_rn(FP32(d * max(abs(v))) / 6), then codes are
 * E2M1_satfinite_rn(FP32(d * v) / decoded_scale); a zero scale publishes zero codes.
 * Products and divisions round in FP32. Decoded values are code * decoded_scale / d.
 *
 * Tiled scales occupy token-tile-major [256 tokens,16 groups] tiles; within each tile the
 * address is token*16+group. Rows must be 5120, 6144 or 17408. Scale storage rounds tokens up to
 * 256; every padding scale is zero. Codes occupy only the represented tokens. Both spans are
 * caller-owned, disjoint, 256-byte aligned and remain live through stream completion.
 */
struct A4Activation {
    DeviceSpan codes;
    DeviceSpan scales;
    std::int32_t rows;
    std::int32_t tokens;
    float divisor;
};

/** Allocates disjoint native tiled storage from an arena or its layout simulation. */
template <class Allocator>
[[nodiscard]] A4Activation allocate_a4_activation(Allocator& allocator, std::int32_t rows,
                                                  std::int32_t tokens, float divisor) {
    if ((rows != 5120 && rows != 6144 && rows != 17408) || tokens <= 0 ||
        tokens > std::numeric_limits<std::int32_t>::max() / rows || !(divisor > 0.0F) ||
        !std::isfinite(divisor)) {
        throw std::invalid_argument("A4 activation: invalid target shape or index extent");
    }
    const auto padded_tokens = ((static_cast<std::size_t>(tokens) + 255) / 256) * 256;
    return {allocator.alloc_bytes(static_cast<std::size_t>(rows) * tokens / 2, 256),
            allocator.alloc_bytes(static_cast<std::size_t>(rows) * padded_tokens / 16, 256), rows,
            tokens, divisor};
}

} // namespace ninfer::ops
