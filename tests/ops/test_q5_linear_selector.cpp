#include "ops/linear/q5/q5_dispatch.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {

using ninfer::ops::LinearPolicy;
using ninfer::ops::detail::Q5Launch;
using namespace ninfer::ops::detail;

struct RouteCase {
    std::int32_t n;
    std::int32_t k;
    std::int32_t t;
    Q5Launch expected;
};

int failures = 0;

void expect_route(const RouteCase& test, LinearPolicy policy) {
    try {
        const Q5Launch actual = select_q5_launch(test.n, test.k, test.t, policy);
        if (actual != test.expected) {
            std::cerr << "wrong Q5 route N=" << test.n << " K=" << test.k << " T=" << test.t
                      << " policy=" << static_cast<int>(policy) << '\n';
            ++failures;
        }
    } catch (const std::exception& error) {
        std::cerr << "Q5 selector rejected N=" << test.n << " K=" << test.k << " T=" << test.t
                  << ": " << error.what() << '\n';
        ++failures;
    }
}

void route_registry() {
    constexpr std::array<RouteCase, 59> cases{{
        {1024, 5120, 1, launch_q5_simt_r8_c4},
        {1024, 5120, 4, launch_q5_simt_r8_c4},
        {1024, 5120, 5, launch_q5_simt_r8_c8},
        {1024, 5120, 16, launch_q5_simt_r8_c8},
        {1024, 5120, 17, launch_q5_mma_r64_c128},
        {1024, 5120, 262141, launch_q5_mma_r64_c128},
        {6144, 5120, 1, launch_q5_gemv_r16_s2_x},
        {6144, 5120, 2, launch_q5_simt_split4_exact},
        {6144, 5120, 6, launch_q5_simt_split4_exact},
        {6144, 5120, 7, launch_q5_simt_r8_c8},
        {6144, 5120, 24, launch_q5_simt_r8_c8},
        {6144, 5120, 25, launch_q5_mma_r64_c64},
        {6144, 5120, 64, launch_q5_mma_r64_c64},
        {6144, 5120, 65, launch_q5_mma_r64_c128},
        {6144, 5120, 262141, launch_q5_mma_r64_c128},
        {7168, 5120, 1, launch_q5_gemv_r16_s2_x},
        {7168, 5120, 2, launch_q5_simt_split4_exact},
        {7168, 5120, 6, launch_q5_simt_split4_exact},
        {7168, 5120, 7, launch_q5_simt_r8_c4},
        {7168, 5120, 16, launch_q5_simt_r8_c4},
        {7168, 5120, 17, launch_q5_mma_r64_c128},
        {5120, 6144, 1, launch_q5_simt_r8_c4},
        {5120, 6144, 2, launch_q5_simt_split2_exact},
        {5120, 6144, 6, launch_q5_simt_split2_exact},
        {5120, 6144, 7, launch_q5_simt_r8_c8},
        {5120, 6144, 24, launch_q5_simt_r8_c8},
        {5120, 6144, 25, launch_q5_mma_r64_c128},
        {5120, 17408, 1, launch_q5_simt_r8_c4},
        {5120, 17408, 2, launch_q5_simt_split2_exact},
        {5120, 17408, 6, launch_q5_simt_split2_exact},
        {5120, 17408, 7, launch_q5_simt_r8_c8},
        {5120, 17408, 24, launch_q5_simt_r8_c8},
        {5120, 17408, 25, launch_q5_mma_r64_c128},
        {1152, 1152, 4, launch_q5_simt_r8_c4},
        {1152, 1152, 76, launch_q5_simt_r8_c4},
        {1152, 1152, 80, launch_q5_mma_r64_c64},
        {1152, 1152, 636, launch_q5_mma_r64_c64},
        {1152, 1152, 640, launch_q5_mma_r64_c128},
        {1152, 1152, 700, launch_q5_mma_r64_c128},
        {1152, 1152, 704, launch_q5_mma_r64_c64},
        {1152, 1152, 708, launch_q5_mma_r64_c128},
        {1152, 1152, 828, launch_q5_mma_r64_c128},
        {1152, 1152, 832, launch_q5_mma_r64_c64},
        {1152, 1152, 836, launch_q5_mma_r64_c128},
        {1152, 1152, 896, launch_q5_mma_r64_c128},
        {1152, 1152, 900, launch_q5_mma_r64_c64},
        {1152, 1152, 960, launch_q5_mma_r64_c64},
        {1152, 1152, 964, launch_q5_mma_r64_c128},
        {1152, 1152, 1024, launch_q5_mma_r64_c128},
        {1152, 1152, 1028, launch_q5_mma_r64_c64},
        {1152, 1152, 1088, launch_q5_mma_r64_c64},
        {1152, 1152, 1092, launch_q5_mma_r64_c128},
        {1152, 1152, 131072, launch_q5_mma_r64_c128},
        {1152, 4304, 4, launch_q5_simt_r8_c4},
        {1152, 4304, 120, launch_q5_simt_r8_c4},
        {1152, 4304, 124, launch_q5_mma_r64_c64},
        {1152, 4304, 1148, launch_q5_mma_r64_c64},
        {1152, 4304, 1152, launch_q5_mma_r64_c128},
        {1152, 4304, 131072, launch_q5_mma_r64_c128},
    }};

    for (const RouteCase& test : cases) {
        expect_route(test, LinearPolicy::A16Only);
        expect_route(test, LinearPolicy::AllowA8);
    }
}

void rejection_contract() {
    constexpr std::array<std::array<std::int32_t, 3>, 12> invalid{{
        {{1024, 5120, 0}},
        {{1024, 5120, -1}},
        {{1025, 5120, 1}},
        {{1024, 4096, 1}},
        {{1152, 1152, 1}},
        {{1152, 1152, 5}},
        {{1152, 1152, 131076}},
        {{1152, 4304, 86}},
        {{1152, 4304, 131071}},
        {{1152, 4352, 4}},
        {{6144, 2048, 1}},
        {{0, 5120, 1}},
    }};
    for (const auto& problem : invalid) {
        try {
            (void)select_q5_a16_launch(problem[0], problem[1], problem[2]);
            std::cerr << "Q5 selector accepted invalid N=" << problem[0] << " K=" << problem[1]
                      << " T=" << problem[2] << '\n';
            ++failures;
        } catch (const std::invalid_argument&) {}
    }

    for (const LinearPolicy policy : {LinearPolicy::AllowA4, static_cast<LinearPolicy>(255)}) {
        try {
            (void)select_q5_launch(6144, 5120, 1, policy);
            std::cerr << "Q5 selector accepted policy=" << static_cast<int>(policy) << '\n';
            ++failures;
        } catch (const std::invalid_argument&) {}
    }
}

} // namespace

int main() {
    route_registry();
    rejection_contract();
    std::cout << (failures ? "FAIL" : "OK") << " Q5 Linear selector\n";
    return failures ? 1 : 0;
}
