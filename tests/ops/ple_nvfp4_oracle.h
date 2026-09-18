#pragma once

#include "ops/op_tester.h"

#include <cmath>
#include <cstdint>
#include <cstring>

namespace ninfer::test {

// Independent registered-codec formula, not a CUDA codec or production lookup table.
// FP32 reconstruction is a specified decode boundary, not a copied private reduction.
inline std::uint16_t ple_nvfp4_oracle(const std::uint8_t* record, int feature) {
    const unsigned code = (record[feature / 2] >> (4 * (feature % 2))) & 15;
    const int exponent = (code >> 1) & 3;
    const double magnitude = exponent == 0 ? (code & 1) * 0.5
        : std::ldexp(1.0 + (code & 1) * 0.5, exponent - 1);
    const double value = std::copysign(magnitude, code & 8 ? -1.0 : 1.0);
    const unsigned scale = record[80 + feature / 16];
    const int scale_exponent = (scale >> 3) & 15;
    const double block = scale_exponent == 0 ? std::ldexp(double(scale & 7), -9)
        : std::ldexp(1.0 + double(scale & 7) / 8.0, scale_exponent - 7);
    const std::uint32_t bits = std::uint32_t(record[90]) | (std::uint32_t(record[91]) << 8) |
        (std::uint32_t(record[92]) << 16) | (std::uint32_t(record[93]) << 24);
    float multiplier;
    std::memcpy(&multiplier, &bits, 4);
    return f32_to_bf16(static_cast<float>(value * block * double(multiplier)));
}

} // namespace ninfer::test
