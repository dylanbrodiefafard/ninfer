#include "ninfer/ops/linear.h"

#include "ops/direct_bf16_weight.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <span>
#include <thread>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::direct_bf16_weight;

constexpr std::int32_t kRows   = 14336;
constexpr std::int32_t kHidden = 5120;
constexpr ReductionCriterion kA16Tolerance{1.0 / 256.0, 1.0 / 256.0, 2.0 / 256.0};

std::vector<std::uint16_t> make_activation_bits() {
    std::vector<std::uint16_t> result(kHidden);
    for (std::int32_t column = 0; column < kHidden; ++column) {
        const int centered = ((column * 29 + 17) & 0xff) - 128;
        result[static_cast<std::size_t>(column)] =
            f32_to_bf16(static_cast<float>(centered) * (1.0F / 512.0F));
    }
    return result;
}

std::vector<float> materialize(std::span<const std::uint16_t> bits) {
    std::vector<float> result(bits.size());
    for (std::size_t index = 0; index < bits.size(); ++index) {
        result[index] = bf16_to_f32(bits[index]);
    }
    return result;
}

std::vector<double> oracle_all_rows(const HostWeight& weight, std::span<const float> activation) {
    std::vector<double> result(kRows);
    const unsigned available   = std::max(1U, std::thread::hardware_concurrency());
    const std::int32_t threads = std::min(kRows, static_cast<std::int32_t>(available));
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(threads));
    for (std::int32_t thread = 0; thread < threads; ++thread) {
        const std::int32_t begin =
            static_cast<std::int32_t>((static_cast<std::int64_t>(kRows) * thread) / threads);
        const std::int32_t end =
            static_cast<std::int32_t>((static_cast<std::int64_t>(kRows) * (thread + 1)) / threads);
        workers.emplace_back([&, begin, end] {
            for (std::int32_t row = begin; row < end; ++row) {
                result[static_cast<std::size_t>(row)] = dot_fp64(weight, row, activation);
            }
        });
    }
    for (std::thread& worker : workers) { worker.join(); }
    return result;
}

int run_bf16_linear() {
    DeviceWeight weight(make_patterned(kRows, kHidden, 401U));
    const std::vector<std::uint16_t> activation_bits = make_activation_bits();
    const std::vector<float> activation              = materialize(activation_bits);
    const std::vector<double> expected               = oracle_all_rows(weight.host, activation);
    DeviceBuffer device_activation                   = to_device(activation_bits);
    GuardedDeviceBuffer guarded_output(static_cast<std::size_t>(kRows) * sizeof(std::uint16_t));
    guarded_output.fill(0xff);

    Tensor x(device_activation.p, DType::BF16, {kHidden, 1});
    Tensor output(guarded_output.data(), DType::BF16, {kRows, 1});
    ops::linear(x, weight.view(), output, ops::LinearPolicy::A16Only, nullptr);
    cuda_synchronize();

    int failures = guarded_output.verify_guards("BF16_A16 Linear output");
    const std::vector<std::uint16_t> output_bits =
        from_device<std::uint16_t>(guarded_output.data(), kRows);
    std::vector<double> actual(kRows);
    for (std::int32_t row = 0; row < kRows; ++row) {
        const std::uint16_t bits = output_bits[static_cast<std::size_t>(row)];
        if (!std::isfinite(bf16_to_f32(bits))) {
            std::cerr << "BF16_A16 Linear output row " << row << " is not finite\n";
            ++failures;
        }
        actual[static_cast<std::size_t>(row)] = bf16_to_f32(bits);
    }
    failures +=
        verify_reduction("BF16_A16 Linear [14336,5120] T=1", actual, expected, kA16Tolerance);
    const std::vector<std::uint16_t> activation_after =
        from_device<std::uint16_t>(device_activation, activation_bits.size());
    if (activation_after != activation_bits) {
        std::cerr << "BF16_A16 Linear modified its activation\n";
        ++failures;
    }
    failures += weight.verify_preserved("BF16_A16 Linear weight");
    return failures;
}

} // namespace

int main() {
    if (ninfer::test::cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    try {
        const int failures = run_bf16_linear();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " BF16_A16 Linear\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "BF16_A16 Linear: " << error.what() << '\n';
        return 1;
    }
}
