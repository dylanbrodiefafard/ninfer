#include "ops/linear_pair/linear_pair_test_common.h"

#include <array>
#include <exception>
#include <iostream>

namespace {

using ninfer::test::linear_pair::ShapeCase;

int w8_a16_conformance() {
    int failures = 0;

    constexpr std::array<std::int32_t, 2> kK5120RouteStarts{5, 57};
    constexpr std::array<std::int32_t, 3> kK5120RouteInteriors{2, 24, 128};
    failures += ninfer::test::linear_pair::run_w8_a16_shape(
        "W8_A16 LinearPair", ShapeCase{5120, 431U, kK5120RouteStarts, kK5120RouteInteriors});

    return failures;
}

} // namespace

int main() {
    if (!ninfer::test::linear_pair::cuda_available()) {
        std::cerr << "FAIL: no usable CUDA device\n";
        return 1;
    }

    try {
        const int failures = w8_a16_conformance();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " W8_A16 LinearPair\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "W8_A16 LinearPair: " << error.what() << '\n';
        return 1;
    }
}
