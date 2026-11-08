#include "ops/linear/linear_test_common.h"

#include <array>
#include <exception>
#include <iostream>

namespace {

using namespace ninfer::test::linear;

constexpr Invocation a16(std::int32_t t) { return {t}; }

int q5_a16_conformance() {
    int failures = 0;

    constexpr std::array kN1152K1152{
        a16(4),   a16(76),   a16(80),   a16(636),  a16(640),  a16(700),  a16(704),
        a16(708), a16(828),  a16(832),  a16(836),  a16(896),  a16(900),  a16(960),
        a16(964), a16(1024), a16(1028), a16(1088), a16(1092), a16(1536), a16(131072),
    };
    failures += run_shape("Q5_A16", ActivationCompute::A16, make_q5g64_f16s_weight,
                          {1152, 1152, 179U, Comparison::Sampled, false, kN1152K1152});

    constexpr std::array kN1152K4304{
        a16(4), a16(120), a16(124), a16(1148), a16(1152), a16(1536), a16(131072),
    };
    failures += run_shape("Q5_A16", ActivationCompute::A16, make_q5g64_f16s_weight,
                          {1152, 4304, 181U, Comparison::Sampled, false, kN1152K4304});

    return failures;
}

} // namespace

int main() {
    if (!ninfer::test::linear::cuda_available()) {
        std::cerr << "FAIL: no usable CUDA device\n";
        return 1;
    }

    try {
        const int failures = q5_a16_conformance();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " Q5_A16 Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Q5_A16 Linear: " << error.what() << '\n';
        return 1;
    }
}
