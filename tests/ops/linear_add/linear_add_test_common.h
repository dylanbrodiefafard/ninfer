#pragma once

#include "core/tensor.h"

#include <cstdint>
#include <span>
#include <string_view>

namespace ninfer::test::linear_add {

struct ShapeCase {
    std::int32_t n;
    std::int32_t k;
    std::uint32_t seed;
    std::span<const std::int32_t> route_starts;
    std::span<const std::int32_t> route_interiors;
};

bool cuda_available();

// Checks BF16_CTRL linear_add at every route boundary of `shape` against the FP64 oracle.
int run_shape(std::string_view label, const ShapeCase& shape);

} // namespace ninfer::test::linear_add
