#include "ops/linear_swiglu/linear_swiglu_test_common.h"

#include <array>
#include <exception>
#include <iostream>

int main() {
    using namespace ninfer;
    using namespace ninfer::test::linear_swiglu;

    try {
        constexpr std::array<std::int32_t, 11> kA16Cases{1, 4, 5, 8, 10, 15, 16, 17, 18, 19, 20};
        constexpr std::array<std::int32_t, 19> kA4Cases{
            1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 24, 36, 48, 49, 128, 1024, 2048, 4096,
        };
        int failures = 0;
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
        std::cout << (failures == 0 ? "OK" : "FAIL") << " LinearSwiGLU NVFP4 correctness\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "LinearSwiGLU NVFP4 test failed: " << error.what() << '\n';
        return 1;
    }
}
