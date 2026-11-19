// moe_activation_quantize against the independent exact host codec (moe_a8_activation_reference.h):
// codes and scales must be bit-identical, which is the cross-card contract with the R9700 helper.
//
// Rows cover all-zero rows (including -0), rows whose scale is exactly a power of two so that x /
// scale lands exactly on E4M3 round-to-nearest-even ties (normal and subnormal), signed zeros and
// negative values that round to zero inside a nonzero row, the +-448 extremes, rows with a large
// outlier (most codes subnormal), rows of BF16-subnormal magnitude (an FP32-subnormal scale), and
// random rows of varied magnitude.
#include "ninfer/ops/moe_experts.h"
#include "ops/moe_a8_activation_reference.h"
#include "ops/op_tester.h"
#include "ops/sanitizer_scope.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <iterator>
#include <random>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kHidden = ops::kMoeHidden;

// Row r of the case: the special rows first, then random rows.
std::vector<float> make_rows(int tokens, std::uint32_t seed) {
    std::vector<float> x(static_cast<std::size_t>(tokens) * kHidden, 0.0F);
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal(0.0F, 1.0F);
    std::uniform_int_distribution<int> magnitude(-30, 30);
    std::uniform_int_distribution<int> position(0, kHidden - 1);
    for (int t = 0; t < tokens; ++t) {
        float* row = x.data() + static_cast<std::size_t>(t) * kHidden;
        switch (t % 8) {
        case 0: // all zero, with negative zeros: scale +0, every code +0
            for (int k = 0; k < kHidden; k += 3) { row[k] = -0.0F; }
            break;
        case 1: { // scale 1 exactly: x / scale = x lands on E4M3 ties
            const float ties[] = {
                1.0625F,                // between 1 and 1.125: even code 0x38 (1.0)
                1.1875F,                // between 1.125 and 1.25: even code 0x3A (1.25)
                -1.0625F,               // signed tie
                std::ldexp(1.5F, -9),   // subnormal tie 2^-9 / 2^-8: code 0x02
                std::ldexp(1.0F, -10),  // tie 0 / 2^-9: +0 (0x00)
                -std::ldexp(1.0F, -10), // the same, negative: -0 (0x80)
                std::ldexp(2.5F, -9),   // tie 2 / 3 x 2^-9: code 0x02
                std::ldexp(7.5F, -9),   // tie between the largest subnormal and 2^-6: 0x08
                -0.0F,                  // -0 inside a nonzero row: 0x80
                -std::ldexp(1.0F, -12), // rounds to zero, negative: 0x80
                448.0F,                 // the largest code 0x7E
                -448.0F,                // -448
                240.0F,
                232.0F,
                248.0F}; // exact, tie 224/240 -> 224, tie 240/256 -> 256
            for (int k = 0; k < kHidden; ++k) {
                row[k] = ties[static_cast<std::size_t>(k) % std::size(ties)];
            }
            row[kHidden - 1] = 448.0F;
            break;
        }
        case 2: // scale 2^-6 exactly (max 7): scaled ties
            for (int k = 0; k < kHidden; ++k) {
                row[k] = std::ldexp(k % 2 == 0 ? 1.0625F : -std::ldexp(1.5F, -9), -6);
            }
            row[7] = 7.0F;
            break;
        case 3: // one large outlier: most codes subnormal
            for (int k = 0; k < kHidden; ++k) { row[k] = normal(rng); }
            row[position(rng)] = 30000.0F;
            break;
        case 4: // BF16 values of FP32-subnormal magnitude: the scale is an FP32 subnormal
            for (int k = 0; k < kHidden; ++k) { row[k] = std::ldexp(normal(rng), -128); }
            break;
        default: { // random magnitudes
            const float scale = std::ldexp(1.0F, magnitude(rng));
            for (int k = 0; k < kHidden; ++k) { row[k] = scale * normal(rng); }
            break;
        }
        }
    }
    round_to_bf16(x);
    return x;
}

int run_case(int tokens, std::uint32_t seed) {
    const std::string label        = "moe_activation_quantize T=" + std::to_string(tokens);
    const std::vector<float> x     = make_rows(tokens, seed);
    const moe_a8::Encoded expected = moe_a8::encode(x, static_cast<std::size_t>(kHidden));
    const DeviceBuffer x_device    = to_device_bf16(x);
    GuardedDeviceBuffer codes(x.size());
    GuardedDeviceBuffer scales(static_cast<std::size_t>(tokens) * sizeof(float));
    codes.fill(0xA5);
    scales.fill(0xA5);
    Tensor x_view(x_device.p, DType::BF16, {kHidden, tokens});
    Tensor codes_view(codes.data(), DType::FP8_E4M3FN, {kHidden, tokens});
    Tensor scales_view(scales.data(), DType::FP32, {tokens});
    ops::moe_activation_quantize(x_view, codes_view, scales_view, nullptr);
    cuda_synchronize();
    int failures = verify_exact((label + " codes").c_str(),
                                from_device<std::uint8_t>(codes.data(), x.size()), expected.codes);
    // Bit patterns, so +0 and -0 differ.
    std::vector<std::uint32_t> got_bits(static_cast<std::size_t>(tokens));
    std::vector<std::uint32_t> expected_bits(got_bits.size());
    const std::vector<float> got_scales = from_device<float>(scales.data(), got_bits.size());
    std::memcpy(got_bits.data(), got_scales.data(), got_bits.size() * sizeof(float));
    std::memcpy(expected_bits.data(), expected.scales.data(), got_bits.size() * sizeof(float));
    failures += verify_exact((label + " scales").c_str(), got_bits, expected_bits);
    failures += codes.verify_guards(label + " codes");
    failures += scales.verify_guards(label + " scales");
    return failures;
}

int invalid_argument_cases() {
    constexpr int kTokens = 2;
    DeviceBuffer x(static_cast<std::size_t>(kHidden) * kTokens * sizeof(std::uint16_t));
    DeviceBuffer codes(static_cast<std::size_t>(kHidden) * kTokens);
    DeviceBuffer scales(kTokens * sizeof(float));
    const auto call = [&](DType codes_dtype, void* codes_data) {
        Tensor x_view(x.p, DType::BF16, {kHidden, kTokens});
        Tensor codes_view(codes_data, codes_dtype, {kHidden, kTokens});
        Tensor scales_view(scales.p, DType::FP32, {kTokens});
        ops::moe_activation_quantize(x_view, codes_view, scales_view, nullptr);
    };
    int failures = 0;
    failures += expect_invalid_argument([&] { call(DType::U8, codes.p); },
                                        "moe_activation_quantize accepted U8 codes");
    failures += expect_invalid_argument([&] { call(DType::FP8_E4M3FN, x.p); },
                                        "moe_activation_quantize accepted codes aliasing x");
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    if (require_cuda() != 0) { return 1; }
    try {
        int failures = 0;
        if (sanitizer_scope(argc, argv)) {
            failures += run_case(9, 3U);
        } else {
            failures += invalid_argument_cases();
            for (const int tokens : {1, 8, 9, 300, 1031}) {
                failures += run_case(tokens, 100U + static_cast<std::uint32_t>(tokens));
            }
        }
        std::cout << (failures == 0 ? "OK" : "FAIL") << " moe_activation_quantize\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "moe_activation_quantize: " << error.what() << '\n';
        return 1;
    }
}
