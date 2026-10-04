#pragma once

#include "ninfer/ops/a4_activation.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {

inline void require_disjoint(const void* a, std::size_t a_bytes, const void* b,
                             std::size_t b_bytes) {
    const auto begin_a = reinterpret_cast<std::uintptr_t>(a);
    const auto begin_b = reinterpret_cast<std::uintptr_t>(b);
    if (a_bytes > std::numeric_limits<std::uintptr_t>::max() - begin_a ||
        b_bytes > std::numeric_limits<std::uintptr_t>::max() - begin_b ||
        (begin_a < begin_b + b_bytes && begin_b < begin_a + a_bytes)) {
        throw std::invalid_argument("A4 activation: overlapping or overflowing storage");
    }
}

inline void validate_a4_activation(const A4Activation& x) {
    if ((x.rows != 5120 && x.rows != 6144 && x.rows != 17408) || x.tokens <= 0 ||
        x.tokens > std::numeric_limits<std::int32_t>::max() / x.rows || !(x.divisor > 0.0F) ||
        !std::isfinite(x.divisor)) {
        throw std::invalid_argument("A4 activation: invalid target shape, index extent or divisor");
    }
    const auto rows          = static_cast<std::size_t>(x.rows);
    const auto tokens        = static_cast<std::size_t>(x.tokens);
    const auto padded_tokens = ((tokens + 255) / 256) * 256;
    if (x.codes.data == nullptr || x.scales.data == nullptr ||
        (reinterpret_cast<std::uintptr_t>(x.codes.data) % 256) != 0 ||
        (reinterpret_cast<std::uintptr_t>(x.scales.data) % 256) != 0 ||
        x.codes.bytes < rows * tokens / 2 || x.scales.bytes < rows * padded_tokens / 16) {
        throw std::invalid_argument(
            "A4 activation: requires aligned codes and padded tiled scales");
    }
    require_disjoint(x.codes.data, x.codes.bytes, x.scales.data, x.scales.bytes);
}

} // namespace ninfer::ops::detail
