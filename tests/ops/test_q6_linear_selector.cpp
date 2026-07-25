#include "ops/linear/q6/q6_dispatch.h"

#include <array>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using ninfer::ops::LinearPolicy;
using ninfer::ops::detail::Q6Launch;
using namespace ninfer::ops::detail;

struct RouteCase {
    std::int32_t n;
    std::int32_t k;
    std::int32_t t;
    Q6Launch expected;
};

int failures = 0;

void fail(const std::string& label, const std::string& message) {
    std::cerr << "FAIL " << label << ": " << message << '\n';
    ++failures;
}

void expect_route(const RouteCase& test) {
    for (const LinearPolicy policy : {LinearPolicy::A16Only, LinearPolicy::AllowA8}) {
        try {
            const Q6Launch actual = select_q6_launch(test.n, test.k, test.t, policy);
            if (actual != test.expected) {
                fail("route", "unexpected launcher for [" + std::to_string(test.n) + "," +
                                  std::to_string(test.k) + "] T=" + std::to_string(test.t));
            }
        } catch (const std::exception& error) {
            fail("route", "selector threw for registered point: " + std::string(error.what()));
        }
    }

    try {
        if (select_q6_a16_launch(test.n, test.k, test.t) != test.expected) {
            fail("A16 selector", "direct A16 selector returned a different launcher");
        }
    } catch (const std::exception& error) {
        fail("A16 selector", "direct A16 selector threw: " + std::string(error.what()));
    }
}

void registered_routes() {
    constexpr std::array<RouteCase, 47> cases{{
        {248320, 5120, 1, launch_q6_simt_r8_c4},
        {248320, 5120, 6, launch_q6_simt_r8_c4},
        {248320, 5120, 7, launch_q6_mma_r64_c128},
        {248320, 5120, 8, launch_q6_mma_r64_c128},
        {248320, 2048, 1, launch_q6_simt_r8_c4},
        {248320, 2048, 3, launch_q6_simt_r8_c4},
        {248320, 2048, 4, launch_q6_simt_r8_c4},
        {248320, 2048, 5, launch_q6_simt_r8_c8},
        {248320, 2048, 7, launch_q6_simt_r8_c8},
        {248320, 2048, 8, launch_q6_simt_r8_c8},
        {248320, 2048, 9, launch_q6_mma_r64_c64},
        {248320, 2048, 64, launch_q6_mma_r64_c64},
        {248320, 2048, 65, launch_q6_mma_r64_c128},
        {248320, 2048, 66, launch_q6_mma_r64_c128},
        {1152, 1536, 4, launch_q6_simt_r8_c4},
        {1152, 1536, 92, launch_q6_simt_r8_c4},
        {1152, 1536, 96, launch_q6_simt_r8_c4},
        {1152, 1536, 100, launch_q6_mma_r64_c64},
        {1152, 1536, 700, launch_q6_mma_r64_c64},
        {1152, 1536, 704, launch_q6_mma_r64_c64},
        {1152, 1536, 708, launch_q6_mma_r64_c128},
        {1152, 1536, 824, launch_q6_mma_r64_c128},
        {1152, 1536, 828, launch_q6_mma_r64_c128},
        {1152, 1536, 832, launch_q6_mma_r64_c64},
        {1152, 1536, 836, launch_q6_mma_r64_c128},
        {1152, 1536, 892, launch_q6_mma_r64_c128},
        {1152, 1536, 896, launch_q6_mma_r64_c128},
        {1152, 1536, 900, launch_q6_mma_r64_c64},
        {1152, 1536, 904, launch_q6_mma_r64_c64},
        {1152, 1536, 956, launch_q6_mma_r64_c64},
        {1152, 1536, 960, launch_q6_mma_r64_c64},
        {1152, 1536, 964, launch_q6_mma_r64_c128},
        {1152, 1536, 968, launch_q6_mma_r64_c128},
        {1152, 1536, 1020, launch_q6_mma_r64_c128},
        {1152, 1536, 1024, launch_q6_mma_r64_c128},
        {1152, 1536, 1028, launch_q6_mma_r64_c64},
        {1152, 1536, 1032, launch_q6_mma_r64_c64},
        {1152, 1536, 1084, launch_q6_mma_r64_c64},
        {1152, 1536, 1088, launch_q6_mma_r64_c64},
        {1152, 1536, 1092, launch_q6_mma_r64_c128},
        {1152, 1536, 1096, launch_q6_mma_r64_c128},
        {1152, 1536, 131068, launch_q6_mma_r64_c128},
        {1152, 1536, 131072, launch_q6_mma_r64_c128},
        {248320, 5120, 131073, launch_q6_mma_r64_c128},
        {248320, 2048, 131073, launch_q6_mma_r64_c128},
        {248320, 5120, 5, launch_q6_simt_r8_c4},
        {248320, 2048, 63, launch_q6_mma_r64_c64},
    }};

    for (const RouteCase& test : cases) { expect_route(test); }
}

void rejection_contract() {
    constexpr std::array<std::array<std::int32_t, 3>, 9> rejected{{
        {65536, 5120, 1},
        {248320, 4096, 1},
        {248320, 5120, 0},
        {248320, 2048, -1},
        {1152, 1536, 1},
        {1152, 1536, 5},
        {1152, 1536, 98},
        {1152, 1536, 131076},
        {1152, 1152, 4},
    }};

    for (const auto& problem : rejected) {
        try {
            (void)select_q6_a16_launch(problem[0], problem[1], problem[2]);
            fail("rejection", "A16 selector accepted an unsupported point");
        } catch (const std::invalid_argument&) {
        } catch (const std::exception& error) {
            fail("rejection", "A16 selector threw wrong exception: " + std::string(error.what()));
        }
    }

    for (const LinearPolicy policy : {LinearPolicy::AllowA4, static_cast<LinearPolicy>(999)}) {
        try {
            (void)select_q6_launch(248320, 5120, 1, policy);
            fail("policy rejection", "selector accepted an unsupported policy");
        } catch (const std::invalid_argument&) {
        } catch (const std::exception& error) {
            fail("policy rejection",
                 "selector threw wrong exception: " + std::string(error.what()));
        }
    }
}

} // namespace

int main() {
    registered_routes();
    rejection_contract();

    std::cout << (failures == 0 ? "OK" : "FAIL") << " Q6 Linear selector\n";
    return failures == 0 ? 0 : 1;
}
