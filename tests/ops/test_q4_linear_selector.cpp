#include "ops/common/token_slices.h"
#include "ops/linear/q4/q4_dispatch.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using ninfer::ops::LinearPolicy;
using ninfer::ops::detail::Q4Launch;
using namespace ninfer::ops::detail;

struct RouteCase {
    std::int32_t n;
    std::int32_t k;
    std::int32_t t;
    Q4Launch expected;
};

int failures = 0;

void expect_route(const RouteCase& test, LinearPolicy policy) {
    try {
        const Q4Launch actual = select_q4_launch(test.n, test.k, test.t, policy);
        if (actual != test.expected) {
            std::cerr << "wrong Q4 route N=" << test.n << " K=" << test.k << " T=" << test.t
                      << " policy=" << static_cast<int>(policy) << '\n';
            ++failures;
        }
    } catch (const std::exception& error) {
        std::cerr << "Q4 selector rejected N=" << test.n << " K=" << test.k << " T=" << test.t
                  << ": " << error.what() << '\n';
        ++failures;
    }
}

void route_registry() {
    constexpr std::array<RouteCase, 52> cases{{
        {1024, 5120, 1, launch_q4_gemv_r1_w8_direct},
        {1024, 5120, 2, launch_q4_simt_r8_c4},
        {1024, 5120, 15, launch_q4_simt_r8_c4},
        {1024, 5120, 16, launch_q4_simt_r8_c8},
        {1024, 5120, 17, launch_q4_mma_r64_c128},
        {1024, 5120, 262141, launch_q4_mma_r64_c128},
        {4096, 5120, 1, launch_q4_gemv_r1_w8_direct},
        {4096, 5120, 2, launch_q4_simt_r8_c4},
        {4096, 5120, 4, launch_q4_simt_r8_c4},
        {4096, 5120, 5, launch_q4_simt_r8_c8},
        {4096, 5120, 16, launch_q4_simt_r8_c8},
        {4096, 5120, 17, launch_q4_mma_r64_c128},
        {6144, 5120, 1, launch_q4_gemv_r1_w8_direct},
        {6144, 5120, 2, launch_q4_simt_r8_c4},
        {6144, 5120, 7, launch_q4_simt_r8_c4},
        {6144, 5120, 8, launch_q4_simt_r8_c8},
        {6144, 5120, 16, launch_q4_simt_r8_c8},
        {6144, 5120, 17, launch_q4_mma_r64_c128},
        {7168, 5120, 1, launch_q4_gemv_r1_w8_direct},
        {7168, 5120, 2, launch_q4_simt_r8_c4},
        {7168, 5120, 7, launch_q4_simt_r8_c4},
        {7168, 5120, 8, launch_q4_simt_r8_c8},
        {7168, 5120, 9, launch_q4_simt_r8_c4},
        {7168, 5120, 15, launch_q4_simt_r8_c4},
        {7168, 5120, 16, launch_q4_simt_r8_c8},
        {7168, 5120, 17, launch_q4_mma_r64_c128},
        {34816, 5120, 1, launch_q4_gemv_r1_w8_direct},
        {34816, 5120, 2, launch_q4_simt_r8_c4},
        {34816, 5120, 4, launch_q4_simt_r8_c4},
        {34816, 5120, 5, launch_q4_simt_r8_c8},
        {34816, 5120, 16, launch_q4_simt_r8_c8},
        {34816, 5120, 17, launch_q4_mma_r64_c128},
        {131072, 5120, 1, launch_q4_gemv_r4_w1_direct},
        {131072, 5120, 2, launch_q4_mma_r64_c128},
        {131072, 2048, 1, launch_q4_gemv_r4_w1_direct},
        {131072, 2048, 2, launch_q4_mma_r64_c128},
        {3456, 1152, 4, launch_q4_simt_r8_c4},
        {3456, 1152, 36, launch_q4_simt_r8_c4},
        {3456, 1152, 40, launch_q4_mma_r64_c64},
        {3456, 1152, 320, launch_q4_mma_r64_c64},
        {3456, 1152, 324, launch_q4_mma_r64_c128},
        {3456, 1152, 131072, launch_q4_mma_r64_c128},
        {4304, 1152, 4, launch_q4_simt_r8_c4},
        {4304, 1152, 8, launch_q4_simt_r8_c8},
        {4304, 1152, 12, launch_q4_simt_r8_c4},
        {4304, 1152, 16, launch_q4_simt_r8_c8},
        {4304, 1152, 24, launch_q4_simt_r8_c8},
        {4304, 1152, 28, launch_q4_mma_r64_c64},
        {4304, 1152, 320, launch_q4_mma_r64_c64},
        {4304, 1152, 324, launch_q4_mma_r64_c128},
        {4304, 1152, 131072, launch_q4_mma_r64_c128},
        {4304, 1152, 36, launch_q4_mma_r64_c64},
    }};

    for (const RouteCase& test : cases) {
        expect_route(test, LinearPolicy::A16Only);
        expect_route(test, LinearPolicy::AllowA8);
    }
}

void rejection_contract() {
    constexpr std::array<std::array<std::int32_t, 3>, 11> invalid{{
        {{1024, 5120, 0}},
        {{1024, 5120, -1}},
        {{1025, 5120, 1}},
        {{1024, 4096, 1}},
        {{3456, 1152, 3}},
        {{3456, 1152, 5}},
        {{3456, 1152, 131076}},
        {{4304, 1152, 1}},
        {{4304, 1152, 131071}},
        {{131072, 1152, 1}},
        {{0, 5120, 1}},
    }};
    for (const auto& problem : invalid) {
        try {
            (void)select_q4_a16_launch(problem[0], problem[1], problem[2]);
            std::cerr << "Q4 selector accepted invalid N=" << problem[0] << " K=" << problem[1]
                      << " T=" << problem[2] << '\n';
            ++failures;
        } catch (const std::invalid_argument&) {}
    }

    for (const LinearPolicy policy : {LinearPolicy::AllowA4, static_cast<LinearPolicy>(255)}) {
        try {
            (void)select_q4_launch(1024, 5120, 1, policy);
            std::cerr << "Q4 selector accepted policy=" << static_cast<int>(policy) << '\n';
            ++failures;
        } catch (const std::invalid_argument&) {}
    }
}

void token_slice_boundaries() {
    constexpr std::int32_t tile     = 4;
    constexpr std::int32_t capacity = tile * kCudaGridYLimit;
    const auto collect              = [](std::int32_t tokens) {
        std::vector<std::array<std::int32_t, 2>> slices;
        for_each_token_slice(tokens, tile, [&](std::int32_t offset, std::int32_t count) {
            slices.push_back({offset, count});
        });
        return slices;
    };

    const auto one   = collect(capacity);
    const auto two   = collect(capacity + 1);
    const auto three = collect(2 * capacity + 7);
    if (one != std::vector<std::array<std::int32_t, 2>>{{{0, capacity}}} ||
        two != std::vector<std::array<std::int32_t, 2>>{{{0, capacity}, {capacity, 1}}} ||
        three != std::vector<std::array<std::int32_t, 2>>{
                     {{0, capacity}, {capacity, capacity}, {2 * capacity, 7}}}) {
        std::cerr << "token slicing produced wrong grid-y boundaries\n";
        ++failures;
    }
}

} // namespace

int main() {
    route_registry();
    rejection_contract();
    token_slice_boundaries();
    std::cout << (failures ? "FAIL" : "OK") << " Q4 Linear selector\n";
    return failures ? 1 : 0;
}
