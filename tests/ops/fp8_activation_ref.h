#pragma once
#include "ops/nvfp4_activation_ref.h"

namespace ninfer::test {
struct Fp8ActivationReference {
    std::vector<std::uint8_t> codes;
    std::vector<float> scales, represented;
};

inline Fp8ActivationReference fp8_activation_reference(std::span<const float> input, int k) {
    Fp8ActivationReference out{std::vector<std::uint8_t>(input.size()),
        std::vector<float>(input.size() / k), std::vector<float>(input.size())};
    for (std::size_t base = 0; base < input.size(); base += k) {
        float maximum = 0;
        for (int i = 0; i < k; ++i) maximum = std::max(maximum, std::abs(input[base + i]));
        const float scale = maximum / 448.0F;
        const float inverse = scale > 0 ? 1.0F / scale : 0.0F;
        out.scales[base / k] = scale;
        for (int i = 0; i < k; ++i) {
            const float normalized = input[base + i] * inverse;
            auto code = nearest_positive_code(std::abs(normalized), 127,
                [](int c) { return quantized_weight::detail::decode_e4m3fn(c); });
            if (std::signbit(normalized)) code |= 128;
            out.codes[base + i] = code;
            out.represented[base + i] = static_cast<float>(quantized_weight::detail::decode_e4m3fn(code) * scale);
        }
    }
    return out;
}
} // namespace ninfer::test
