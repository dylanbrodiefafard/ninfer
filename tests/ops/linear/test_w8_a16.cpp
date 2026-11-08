#include "ops/linear/linear_test_common.h"

#include <array>
#include <exception>
#include <iostream>
#include <string>

namespace {

using namespace ninfer::test::linear;

constexpr Invocation a16(std::int32_t t) { return {t}; }

constexpr Invocation convenience(std::int32_t t) { return {t, CallForm::A16Convenience}; }

int w8_a16_conformance() {
    int failures = 0;

    constexpr std::array kN248320K5120{
        a16(1),  a16(5),  a16(6),  a16(10), a16(15), a16(16), a16(17), a16(20), a16(32),
        a16(33), a16(34), a16(40), a16(41), a16(42), a16(48), a16(49), a16(64), a16(65),
    };
    failures += run_shape("W8_A16", ActivationCompute::A16, make_w8g32_f16s_weight,
                          {248320, 5120, 197U, Comparison::Sampled, false, kN248320K5120});
    // Verify LM head: every chain width across C=2..6 shares one weight pass of at most 48
    // columns (W=7 C=6 is 42, W=8 C=6 is 48).
    constexpr std::array kVocabularyBatches{2, 3, 4, 5, 6};
    for (const std::int32_t width : {2, 3, 4, 5, 6, 7, 8}) {
        failures += run_packed_sequences_matches_panels(
            "W8_A16 vocabulary packed W" + std::to_string(width), make_w8g32_f16s_weight, 248320,
            5120, 199U + static_cast<std::uint32_t>(width), width, kVocabularyBatches);
    }

    constexpr std::array kN5120K10240{
        a16(1),  a16(4),  a16(5),  a16(8),  a16(9),  a16(16), a16(17), a16(24),
        a16(25), a16(32), a16(33), a16(40), a16(41), a16(48), a16(49), a16(128),
    };
    failures += run_shape("W8_A16", ActivationCompute::A16, make_w8g32_f16s_weight,
                          {5120, 10240, 211U, Comparison::Sampled, false, kN5120K10240});

    constexpr std::array kN256K5120{
        a16(1), a16(4), a16(5), a16(16), a16(17), a16(128),
    };
    failures += run_shape("W8_A16", ActivationCompute::A16, make_w8g32_f16s_weight,
                          {256, 5120, 293U, Comparison::Sampled, false, kN256K5120});

    constexpr std::array kN1024K5120{
        convenience(1), a16(4), a16(5), a16(16), a16(17), a16(128),
    };
    failures += run_shape("W8_A16", ActivationCompute::A16, make_w8g32_f16s_weight,
                          {1024, 5120, 223U, Comparison::Full, true, kN1024K5120});

    constexpr std::array kN1280K5120{
        a16(1), a16(4), a16(5), a16(16), a16(17), a16(128),
    };
    failures += run_shape("W8_A16", ActivationCompute::A16, make_w8g32_f16s_weight,
                          {1280, 5120, 307U, Comparison::Sampled, false, kN1280K5120});

    constexpr std::array kN6144K5120{
        a16(1), a16(4), a16(5), a16(16), a16(17), a16(128),
    };
    failures += run_shape("W8_A16", ActivationCompute::A16, make_w8g32_f16s_weight,
                          {6144, 5120, 227U, Comparison::Sampled, false, kN6144K5120});

    constexpr std::array kN14336K5120{
        a16(1),  a16(4),  a16(5),  a16(8),  a16(9),  a16(16), a16(17), a16(24),
        a16(25), a16(32), a16(33), a16(40), a16(41), a16(48), a16(49), a16(128),
    };
    failures += run_shape("W8_A16", ActivationCompute::A16, make_w8g32_f16s_weight,
                          {14336, 5120, 229U, Comparison::Sampled, false, kN14336K5120});

    constexpr std::array kN34816K5120{
        a16(1),  a16(4),  a16(5),  a16(8),  a16(9),  a16(15), a16(16), a16(17), a16(21),  a16(22),
        a16(24), a16(25), a16(32), a16(33), a16(40), a16(41), a16(48), a16(49), a16(128),
    };
    failures += run_shape("W8_A16", ActivationCompute::A16, make_w8g32_f16s_weight,
                          {34816, 5120, 233U, Comparison::Sampled, false, kN34816K5120});

    constexpr std::array kN5120K6144{
        a16(1),  a16(4),  a16(5),  a16(8),  a16(9),  a16(16), a16(17), a16(24),
        a16(25), a16(32), a16(33), a16(40), a16(41), a16(48), a16(49), a16(128),
    };
    failures += run_shape("W8_A16", ActivationCompute::A16, make_w8g32_f16s_weight,
                          {5120, 6144, 239U, Comparison::Sampled, false, kN5120K6144});

    constexpr std::array kN5120K17408{
        a16(1),  a16(4),  a16(5),  a16(8),  a16(9),  a16(16), a16(17), a16(24), a16(25),
        a16(30), a16(31), a16(32), a16(33), a16(40), a16(41), a16(48), a16(49), a16(128),
    };
    failures += run_shape("W8_A16", ActivationCompute::A16, make_w8g32_f16s_weight,
                          {5120, 17408, 241U, Comparison::Sampled, false, kN5120K17408});

    constexpr std::array kN5120K25600{
        a16(1), a16(4), a16(5), a16(16), a16(17), a16(128),
    };
    failures += run_shape("W8_A16", ActivationCompute::A16, make_w8g32_f16s_weight,
                          {5120, 25600, 311U, Comparison::Sampled, false, kN5120K25600});

    constexpr std::array kN5120K4096{
        a16(1), a16(4), a16(5), a16(16), a16(17), a16(128),
    };
    failures += run_shape("W8_A16", ActivationCompute::A16, make_w8g32_f16s_weight,
                          {5120, 4096, 313U, Comparison::Sampled, false, kN5120K4096});

    constexpr std::array kN4608K4608{
        a16(1), a16(8), a16(9), a16(11), a16(12), a16(13), a16(256), a16(257), a16(384), a16(32768),
    };
    failures += run_shape("W8_A16", ActivationCompute::A16, make_w8g32_f16s_weight,
                          {4608, 4608, 277U, Comparison::Sampled, false, kN4608K4608});

    constexpr std::array kN5120K4608{
        a16(1), a16(4), a16(5), a16(6), a16(128), a16(384), a16(32768),
    };
    failures += run_shape("W8_A16", ActivationCompute::A16, make_w8g32_f16s_weight,
                          {5120, 4608, 281U, Comparison::Sampled, false, kN5120K4608});

    return failures;
}

} // namespace

int main() {
    if (!ninfer::test::linear::cuda_available()) {
        std::cerr << "FAIL: no usable CUDA device\n";
        return 1;
    }

    try {
        const int failures = w8_a16_conformance();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " W8_A16 Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "W8_A16 Linear: " << error.what() << '\n';
        return 1;
    }
}
