#include "targets/qwen3_8_flash_next/impl/runtime/speculative_policy.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>

namespace {
namespace fn = ninfer::targets::qwen3_8_flash_next;

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

void cases() {
    constexpr std::array expected{3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1};
    for (std::size_t batch = 1; batch <= expected.size(); ++batch) {
        require(fn::speculative_batch_policy(batch).k == expected[batch - 1],
                "batch-size table selected the wrong default arm");
    }
    fn::SpeculativeRequestPolicy healthy, poor;
    // Both requests see the same C5 rounds; the failed request cannot train the healthy one.
    for (std::int32_t round = 0; round < 63; ++round) {
        healthy.observe(5, 2, 2, 2);
        poor.observe(5, 2, 2, 0);
    }
    require(poor.ceiling == 3, "step-down ignored two-window hysteresis");
    poor.observe(5, 2, 2, 0);
    require(poor.ceiling == 0, "first-hop collapse did not disable losing drafting");
    require(healthy.ceiling == 3, "another request's rejection changed healthy policy");

    fn::SpeculativeRequestPolicy tail;
    for (std::int32_t round = 0; round < 64; ++round) { tail.observe(1, 3, 3, 2); }
    require(tail.ceiling == 2, "poor third hop did not step down to K2");
    for (std::int32_t round = 0; round < 64; ++round) { tail.observe(1, 2, 2, 1); }
    require(tail.ceiling == 1, "poor second hop did not step down to K1");

    fn::SpeculativeRequestPolicy transient;
    for (std::int32_t round = 0; round < 32; ++round) { transient.observe(1, 3, 3, 0); }
    for (std::int32_t round = 0; round < 32; ++round) { transient.observe(1, 3, 3, 3); }
    for (std::int32_t round = 0; round < 32; ++round) { transient.observe(1, 3, 3, 0); }
    require(transient.ceiling == 3, "separated bad windows triggered a step-down");
    for (std::int32_t round = 0; round < 32; ++round) { transient.observe(5, 2, 2, 0); }
    require(transient.ceiling == 3, "different proposal widths shared hysteresis");

    fn::SpeculativeRequestPolicy conditional;
    // Unreached hops are not failures: every reached second/third hop accepts.
    for (std::int32_t round = 0; round < 320; ++round) {
        conditional.observe(1, 3, 3, round % 2 == 0 ? 0 : 3);
    }
    require(conditional.ceiling == 3, "conditional acceptance counted unreached hops as failures");
    fn::SpeculativeRequestPolicy bounded;
    for (std::int32_t round = 0; round < 128; ++round) { bounded.observe(1, 3, 1, 1); }
    require(bounded.ceiling == 3 && bounded.reached[1] == 0 && bounded.reached[2] == 0,
            "unlicensed proposal hops trained the request policy");
}
} // namespace

int main() {
    try {
        cases();
        std::cout << "Flash-Next speculation policy: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
