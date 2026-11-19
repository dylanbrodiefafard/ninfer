#pragma once

// Test-side reference of the routed-expert A8 activation codec (moe_activation_quantize in
// ninfer/ops/moe_experts.h; the cross-card contract shared with the R9700 expert helper). Pure host
// code, independent of the device implementation: the E4M3 code is found by an exhaustive
// nearest-value search over the finite codes with ties to the even code, not by a conversion
// instruction. Shared by CUDA and HIP Op tests.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace ninfer::test::moe_a8 {

// The exact value of an E4M3FN byte (0x7F and 0xFF, the NaNs, are never produced by the codec).
inline double e4m3fn_value(std::uint8_t bits) {
    const int exponent     = (bits >> 3) & 0xF;
    const int mantissa     = bits & 0x7;
    const double magnitude = exponent == 0 ? std::ldexp(mantissa / 8.0, -6)
                                           : std::ldexp(1.0 + mantissa / 8.0, exponent - 7);
    return (bits & 0x80) != 0 ? -magnitude : magnitude;
}

// Round-to-nearest-even of an FP32 value to E4M3FN with saturation to +-448: the code of the
// nearest finite magnitude (0x00..0x7E, values exact in double so distances and ties are exact),
// the even code on a tie (E4M3 codes are ordered by magnitude, so the even code has the even
// mantissa), and the sign bit of the input, so a negative value that rounds to zero gives 0x80.
inline std::uint8_t e4m3fn_round(float value) {
    const double magnitude = std::abs(static_cast<double>(value));
    double best_distance   = std::numeric_limits<double>::infinity();
    std::uint8_t best      = 0;
    for (int code = 0; code <= 0x7E; ++code) {
        const double distance = std::abs(magnitude - e4m3fn_value(static_cast<std::uint8_t>(code)));
        if (distance < best_distance || (distance == best_distance && (code & 1) == 0)) {
            best_distance = distance;
            best          = static_cast<std::uint8_t>(code);
        }
    }
    return std::signbit(value) ? static_cast<std::uint8_t>(best | 0x80) : best;
}

struct Encoded {
    std::vector<std::uint8_t> codes; // [rows][K]
    std::vector<float> scales;       // [rows]
};

// The codec of rows of K represented BF16 values (given as floats): scale = max|x| / 448 and
// codes = RNE(x / scale) in FP32 arithmetic; an all-zero row has scale +0 and +0 codes.
inline Encoded encode(std::span<const float> values, std::size_t k) {
    if (k == 0 || values.size() % k != 0) {
        throw std::invalid_argument("moe_a8::encode: values must be whole rows of K");
    }
    const std::size_t rows = values.size() / k;
    Encoded out{std::vector<std::uint8_t>(values.size(), 0), std::vector<float>(rows, 0.0F)};
    for (std::size_t r = 0; r < rows; ++r) {
        const float* row = values.data() + r * k;
        float maximum    = 0.0F;
        for (std::size_t i = 0; i < k; ++i) { maximum = std::max(maximum, std::abs(row[i])); }
        if (maximum == 0.0F) { continue; }
        const float scale = maximum / 448.0F;
        out.scales[r]     = scale;
        for (std::size_t i = 0; i < k; ++i) { out.codes[r * k + i] = e4m3fn_round(row[i] / scale); }
    }
    return out;
}

// The represented values decode(code) * scale of an encoding, exact in double.
inline std::vector<double> decode(const Encoded& encoded, std::size_t k) {
    std::vector<double> values(encoded.codes.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = e4m3fn_value(encoded.codes[i]) * static_cast<double>(encoded.scales[i / k]);
    }
    return values;
}

} // namespace ninfer::test::moe_a8
