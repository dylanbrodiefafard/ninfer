#pragma once

#include "ops/quantized_weight.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace ninfer::test {

// Independent exhaustive nearest-code oracle for the NVFP4 activation codec. This is a
// supplemental represented-input control; canonical Op references still use original BF16 x.
struct Nvfp4ActivationReference {
    std::vector<std::uint8_t> codes, scales;
    std::vector<float> represented;
};

template <class Decode>
inline std::uint8_t nearest_positive_code(double value, int count, Decode decode) {
    double distance = std::numeric_limits<double>::infinity();
    std::uint8_t best = 0;
    for (int code = 0; code < count; ++code) {
        const double d = std::abs(value - decode(code));
        if (d < distance || (d == distance && (code & 1) == 0)) {
            distance = d;
            best = static_cast<std::uint8_t>(code);
        }
    }
    return best;
}

inline Nvfp4ActivationReference nvfp4_activation_reference(std::span<const float> input,
                                                           float divisor) {
    if (input.size() % 16 != 0 || divisor <= 0) {
        throw std::invalid_argument("NVFP4 activation reference requires K16 groups and divisor>0");
    }
    Nvfp4ActivationReference out{std::vector<std::uint8_t>(input.size() / 2),
        std::vector<std::uint8_t>(input.size() / 16), std::vector<float>(input.size())};
    for (std::size_t base = 0; base < input.size(); base += 16) {
        float maximum = 0;
        for (std::size_t i = 0; i < 16; ++i) { maximum = std::max(maximum, std::abs(input[base + i])); }
        // These FP32 boundaries define encoding, not the matrix/convolution oracle arithmetic.
        const float scale_value = (divisor * maximum) / 6.0F;
        const auto scale_code = nearest_positive_code(scale_value, 127,
            [](int code) { return quantized_weight::detail::decode_e4m3fn(code); });
        out.scales[base / 16] = scale_code;
        if (scale_code == 0) { continue; }
        const float scale = static_cast<float>(quantized_weight::detail::decode_e4m3fn(scale_code));
        for (std::size_t i = 0; i < 16; ++i) {
            const float normalized = (input[base + i] * divisor) / scale;
            auto code = nearest_positive_code(std::abs(normalized), 8,
                [](int code) { return quantized_weight::detail::decode_e2m1(code); });
            if (std::signbit(normalized)) { code |= 8; }
            out.codes[(base + i) / 2] |= code << (4 * (i & 1));
            out.represented[base + i] = static_cast<float>(
                quantized_weight::detail::decode_e2m1(code) * scale / divisor);
        }
    }
    return out;
}

} // namespace ninfer::test
