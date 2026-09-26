#include "ops/linear_swiglu/linear_swiglu_test_common.h"

#include <array>
#include <exception>
#include <iostream>
#include <string>

int main() {
    using namespace ninfer;
    using namespace ninfer::test::linear_swiglu;

    try {
        constexpr std::array<std::int32_t, 11> kA16Cases{1, 4, 5, 8, 10, 15, 16, 17, 18, 19, 20};
        constexpr std::array<std::int32_t, 19> kA4Cases{
            1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 24, 36, 48, 49, 128, 1024, 2048, 4096,
        };
        int failures = 0;
        failures += run_profile(
            "rmsnorm_linear_swiglu/nvfp4-a8",
            Profile{QType::NVFP4, 34816, 5120, 17408, 0x8317U, ActivationCompute::A8, true},
            std::array<std::int32_t, 15>{1, 3, 4, 5, 6, 8, 12, 16, 20, 24, 25, 30, 32, 33, 36},
            std::array<std::int32_t, 4>{4, 5, 24, 36});
        failures += run_profile("LinearSwiGLU NVFP4_A8",
            {QType::NVFP4, 34816, 5120, 17408, 1803U, ActivationCompute::A8},
            std::array<std::int32_t, 15>{4, 5, 6, 8, 10, 12, 15, 16, 18, 20, 24, 25, 30, 33, 36});
        // Every verify width uses A8; aggregated C=2..6 rows equal their W-panels.
        for (const bool fused_norm : {false, true}) {
            const Profile a8{QType::NVFP4, 34816, 5120, 17408, 1805U, ActivationCompute::A8,
                             fused_norm};
            const char* name = fused_norm ? "rmsnorm_linear_swiglu/nvfp4-a8" : "LinearSwiGLU NVFP4_A8";
            failures += run_packed_matches_panels(std::string(name) + " W2 panels", a8, 2,
                std::array<std::int32_t, 5>{4, 6, 8, 10, 12});
            failures += run_packed_matches_panels(std::string(name) + " W3 panels", a8, 3,
                std::array<std::int32_t, 5>{6, 9, 12, 15, 18});
            failures += run_packed_matches_panels(std::string(name) + " W4 panels", a8, 4,
                std::array<std::int32_t, 5>{8, 12, 16, 20, 24});
            failures += run_packed_matches_panels(std::string(name) + " W5 panels", a8, 5,
                std::array<std::int32_t, 5>{10, 15, 20, 25, 30});
            failures += run_packed_matches_panels(std::string(name) + " W6 panels", a8, 6,
                std::array<std::int32_t, 5>{12, 18, 24, 30, 36});
        }
        failures += run_profile("LinearSwiGLU NVFP4_A16",
                                {QType::NVFP4, 34816, 5120, 17408, 1801U, ActivationCompute::A16},
                                kA16Cases);
        failures +=
            run_profile("LinearSwiGLU NVFP4_A4",
                        {QType::NVFP4, 34816, 5120, 17408, 1803U, ActivationCompute::A4}, kA4Cases);
        failures += run_column0_matches_decode(
            "LinearSwiGLU NVFP4_A16 packed-col0",
            {QType::NVFP4, 34816, 5120, 17408, 1801U, ActivationCompute::A16},
            std::array<std::int32_t, 5>{2, 8, 12, 16, 20});
        failures += run_packed_matches_panels(
            "LinearSwiGLU NVFP4_A16 W5 panels",
            {QType::NVFP4, 34816, 5120, 17408, 1801U, ActivationCompute::A16}, 5,
            std::array<std::int32_t, 3>{10, 15, 20});
        failures += run_packed_matches_panels(
            "LinearSwiGLU NVFP4_A16 W4 panels",
            {QType::NVFP4, 34816, 5120, 17408, 1801U, ActivationCompute::A16}, 4,
            std::array<std::int32_t, 3>{8, 12, 16});
        failures += run_packed_matches_panels(
            "LinearSwiGLU NVFP4_A16 W6 panels",
            {QType::NVFP4, 34816, 5120, 17408, 1801U, ActivationCompute::A16}, 6,
            std::array<std::int32_t, 2>{12, 18});
        std::cout << (failures == 0 ? "OK" : "FAIL") << " LinearSwiGLU NVFP4 correctness\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "LinearSwiGLU NVFP4 test failed: " << error.what() << '\n';
        return 1;
    }
}
