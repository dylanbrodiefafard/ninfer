#include "ninfer/ops/nll_from_logits.h"
#include "ops/op_tester.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

std::vector<double> nll_oracle(const std::vector<std::uint16_t>& logits,
                               const std::vector<std::int32_t>& targets, std::int32_t physical_rows,
                               std::int32_t valid_rows, std::int32_t tokens) {
    std::vector<double> expected(static_cast<std::size_t>(tokens));
    for (std::int32_t token = 0; token < tokens; ++token) {
        const std::size_t base = static_cast<std::size_t>(token) * physical_rows;
        float max_value        = bf16_to_f32(logits[base]);
        for (std::int32_t row = 1; row < valid_rows; ++row) {
            max_value = std::max(max_value, bf16_to_f32(logits[base + static_cast<std::size_t>(row)]));
        }
        double sum = 0.0;
        for (std::int32_t row = 0; row < valid_rows; ++row) {
            sum += std::exp(static_cast<double>(bf16_to_f32(logits[base + static_cast<std::size_t>(row)])) -
                            static_cast<double>(max_value));
        }
        const std::int32_t target = targets[static_cast<std::size_t>(token)];
        const double target_logit =
            static_cast<double>(bf16_to_f32(logits[base + static_cast<std::size_t>(target)]));
        expected[static_cast<std::size_t>(token)] =
            static_cast<double>(max_value) + std::log(sum) - target_logit;
    }
    return expected;
}

int run_case(std::int32_t physical_rows, std::int32_t valid_rows, std::int32_t tokens,
             double abs_tol = 2.0e-3) {
    std::vector<std::uint16_t> logits(static_cast<std::size_t>(physical_rows) *
                                      static_cast<std::size_t>(tokens));
    std::vector<std::int32_t> targets(static_cast<std::size_t>(tokens));
    for (std::int32_t token = 0; token < tokens; ++token) {
        const std::size_t base = static_cast<std::size_t>(token) * physical_rows;
        for (std::int32_t row = 0; row < physical_rows; ++row) {
            const std::uint32_t mixed = static_cast<std::uint32_t>(row) * 1664525u +
                                        static_cast<std::uint32_t>(token + 1) * 1013904223u;
            const float value  = -8.0f + static_cast<float>(mixed % 2048u) * (1.0f / 256.0f);
            logits[base + static_cast<std::size_t>(row)] = f32_to_bf16(value);
        }
        targets[static_cast<std::size_t>(token)] = (token * 17 + 3) % valid_rows;
        if (valid_rows < physical_rows) {
            logits[base + static_cast<std::size_t>(valid_rows)] = f32_to_bf16(64.0f);
        }
    }
    const auto expected = nll_oracle(logits, targets, physical_rows, valid_rows, tokens);

    GuardedDeviceBuffer device_logits(logits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_targets(targets.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer device_output(static_cast<std::size_t>(tokens) * sizeof(float));
    device_logits.copy_from_host(logits.data(), logits.size() * sizeof(std::uint16_t));
    device_targets.copy_from_host(targets.data(), targets.size() * sizeof(std::int32_t));
    device_output.fill(0xcd);

    Tensor logits_tensor(device_logits.data(), DType::BF16, {physical_rows, tokens});
    Tensor targets_tensor(device_targets.data(), DType::I32, {tokens});
    Tensor output_tensor(device_output.data(), DType::FP32, {tokens});
    ops::nll_from_logits(logits_tensor, targets_tensor, output_tensor, valid_rows, nullptr);
    cuda_synchronize();

    std::vector<float> host(static_cast<std::size_t>(tokens));
    device_output.copy_to_host(host.data(), host.size() * sizeof(float));
    std::vector<double> actual(host.begin(), host.end());
    const std::string label = "nll_from_logits rows=" + std::to_string(physical_rows) +
                              " valid=" + std::to_string(valid_rows) +
                              " T=" + std::to_string(tokens);
    int failures = verify_pointwise(label, actual, expected, PointwiseCriterion{abs_tol, 1.0e-5});
    failures += device_output.verify_guards("nll_from_logits output");
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cerr << "FAIL: no usable CUDA device\n";
        return 1;
    }
    int failures = 0;
    failures += run_case(128, 128, 1);
    failures += run_case(256, 200, 4);
    failures += run_case(1024, 1024, 3);
    failures += run_case(248320, 248077, 1);
    failures += run_case(248320, 248077, 64, 5.0e-3);
    if (failures != 0) {
        std::cerr << "nll_from_logits failures=" << failures << '\n';
        return 1;
    }
    std::cout << "nll_from_logits: PASS\n";
    return 0;
}
