#include "ops/linear/linear_test_common.h"

#include <array>
#include <exception>
#include <iostream>

namespace {

using namespace ninfer::test::linear;

constexpr Invocation a16(std::int32_t t) { return {t}; }

constexpr Invocation convenience(std::int32_t t) { return {t, CallForm::A16Convenience}; }

int q6_a16_conformance() {
    int failures = 0;

    constexpr std::array kN248320K5120{
        a16(1),
        a16(6),
        a16(7),
        a16(128),
    };
    failures += run_shape("Q6_A16", ActivationCompute::A16, make_q6g64_f16s_weight,
                          {248320, 5120, 191U, Comparison::Sampled, false, kN248320K5120});

    constexpr std::array kN248320K2048{
        a16(1), a16(4), a16(5), a16(8), a16(9), a16(64), a16(65), a16(128),
    };
    failures += run_shape("Q6_A16", ActivationCompute::A16, make_q6g64_f16s_weight,
                          {248320, 2048, 193U, Comparison::Sampled, false, kN248320K2048});

    constexpr std::array kN1152K1536Full{
        convenience(4),
        a16(96),
    };
    failures += run_shape("Q6_A16", ActivationCompute::A16, make_q6g64_f16s_weight,
                          {1152, 1536, 197U, Comparison::Full, true, kN1152K1536Full});

    constexpr std::array kN1152K1536Large{
        a16(100), a16(704), a16(708),  a16(828),  a16(832),  a16(836),  a16(896),    a16(900),
        a16(960), a16(964), a16(1024), a16(1028), a16(1088), a16(1092), a16(131072),
    };
    failures += run_shape("Q6_A16", ActivationCompute::A16, make_q6g64_f16s_weight,
                          {1152, 1536, 197U, Comparison::Sampled, false, kN1152K1536Large});

    return failures;
}

} // namespace

int main() {
    if (!ninfer::test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 0;
    }

    try {
        const int failures = q6_a16_conformance();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " Q6_A16 Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Q6_A16 Linear: " << error.what() << '\n';
        return 1;
    }
}
