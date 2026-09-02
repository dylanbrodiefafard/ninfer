#include "ninfer/ops/argmax.h"
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

std::vector<std::int32_t> argmax_oracle(const std::vector<std::uint16_t>& logits,
                                        std::int32_t physical_rows, std::int32_t tokens,
                                        std::int32_t valid_rows) {
    std::vector<std::int32_t> expected(static_cast<std::size_t>(tokens));
    for (std::int32_t token = 0; token < tokens; ++token) {
        const std::size_t base = static_cast<std::size_t>(token) * physical_rows;
        std::int32_t best      = 0;
        float best_value       = bf16_to_f32(logits[base]);
        for (std::int32_t row = 1; row < valid_rows; ++row) {
            const float value = bf16_to_f32(logits[base + row]);
            if (value > best_value) {
                best       = row;
                best_value = value;
            }
        }
        expected[static_cast<std::size_t>(token)] = best;
    }
    return expected;
}

std::vector<std::uint16_t> make_logits(std::int32_t physical_rows, std::int32_t tokens,
                                       std::int32_t valid_rows) {
    std::vector<std::uint16_t> logits(static_cast<std::size_t>(physical_rows) * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        const std::size_t base = static_cast<std::size_t>(token) * physical_rows;
        for (std::int32_t row = 0; row < physical_rows; ++row) {
            const std::uint32_t mixed = static_cast<std::uint32_t>(row) * 1664525u +
                                        static_cast<std::uint32_t>(token + 1) * 1013904223u;
            const float value  = -24.0f + static_cast<float>(mixed % 3072u) * (1.0f / 256.0f);
            logits[base + row] = f32_to_bf16(value);
        }

        std::int32_t first  = 17 + token * 7919;
        std::int32_t second = valid_rows - 1 - token * 65537;
        first %= valid_rows;
        second %= valid_rows;
        if (second < 0) { second += valid_rows; }
        if (first == second) { second = (second + 1) % valid_rows; }
        if (second < first) {
            const std::int32_t temporary = first;
            first                        = second;
            second                       = temporary;
        }
        logits[base + first]  = f32_to_bf16(32.0f + static_cast<float>(token));
        logits[base + second] = logits[base + first];

        if (valid_rows < physical_rows) {
            logits[base + valid_rows]        = f32_to_bf16(64.0f);
            logits[base + physical_rows - 1] = f32_to_bf16(96.0f);
        }
    }
    return logits;
}

int run_case(std::int32_t physical_rows, std::int32_t valid_rows, std::int32_t tokens) {
    const auto logits   = make_logits(physical_rows, tokens, valid_rows);
    const auto expected = argmax_oracle(logits, physical_rows, tokens, valid_rows);

    GuardedDeviceBuffer device_logits(logits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_output(static_cast<std::size_t>(tokens) * sizeof(std::int32_t));
    device_logits.copy_from_host(logits.data(), logits.size() * sizeof(std::uint16_t));
    device_output.fill(0xcd);

    Tensor logits_tensor(device_logits.data(), DType::BF16, {physical_rows, tokens});
    Tensor output_tensor(device_output.data(), DType::I32, {tokens});
    ops::argmax(logits_tensor, output_tensor, valid_rows, nullptr);
    cuda_synchronize();

    const auto actual =
        from_device<std::int32_t>(device_output.data(), static_cast<std::size_t>(tokens));
    const auto logits_after = from_device<std::uint16_t>(device_logits.data(), logits.size());
    const std::string label = "argmax rows=" + std::to_string(physical_rows) +
                              " valid=" + std::to_string(valid_rows) +
                              " T=" + std::to_string(tokens);

    int failures = 0;
    failures += verify_exact(label.c_str(), actual, expected);
    failures += verify_exact((label + " preserves logits").c_str(), logits_after, logits);
    failures += device_logits.verify_guards((label + " logits").c_str());
    failures += device_output.verify_guards((label + " output").c_str());
    return failures;
}

int run_suppressed_case(std::int32_t physical_rows, std::int32_t valid_rows,
                        const char* label) {
    constexpr std::int32_t width         = 3;
    constexpr std::int32_t batch         = 2;
    constexpr std::int32_t tokens        = width * batch;
    std::vector<std::uint16_t> logits(static_cast<std::size_t>(physical_rows) * tokens,
                                      f32_to_bf16(-20.0f));
    for (int column = 0; column < tokens; ++column) {
        const int suppressed = column < width ? 11 : 17;
        const std::size_t base = static_cast<std::size_t>(column) * physical_rows;
        logits[base + suppressed] = f32_to_bf16(9.0f);
        logits[base + 23]         = f32_to_bf16(8.0f);
    }
    for (int token = 0; token < valid_rows; ++token) {
        logits[static_cast<std::size_t>(token)] = f32_to_bf16(-INFINITY);
    }
    std::vector<ops::SamplingConfig> configs(batch);
    configs[0].suppressed_token_count = 2;
    configs[0].suppressed_tokens[0]   = 0;
    configs[0].suppressed_tokens[1]   = 11;
    configs[1].suppressed_token_count = 1;
    configs[1].suppressed_tokens[0]   = 17;

    DeviceBuffer device_logits  = to_device(logits);
    DeviceBuffer device_configs = to_device(configs);
    GuardedDeviceBuffer device_output(static_cast<std::size_t>(tokens) * sizeof(std::int32_t));
    Tensor logits_tensor(device_logits.p, DType::BF16, {physical_rows, tokens});
    Tensor output_tensor(device_output.data(), DType::I32, {tokens});
    ops::argmax(logits_tensor, output_tensor, valid_rows,
                static_cast<const ops::SamplingConfig*>(device_configs.p), width, nullptr);
    cuda_synchronize();

    std::vector<std::int32_t> expected(tokens, 23);
    expected[0] = 1;
    int failures = verify_exact(
        label, from_device<std::int32_t>(device_output.data(), static_cast<std::size_t>(tokens)),
        expected);
    failures += device_output.verify_guards(label);
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cerr << "FAIL: no usable CUDA device\n";
        return 1;
    }

    int failures = 0;
    failures += run_case(248320, 248077, 1);
    failures += run_case(248320, 248077, 6);
    failures += run_case(248320, 248077, 15);
    failures += run_case(248320, 248077, 128);
    failures += run_case(131072, 131072, 1);
    failures += run_case(131072, 131072, 15);
    failures += run_case(131072, 131072, 120);
    failures += run_suppressed_case(1024, 1000, "argmax row-local suppressed tokens small");
    failures += run_suppressed_case(248320, 248077,
                                    "argmax row-local suppressed tokens tiled");
    std::cout << (failures ? "FAIL" : "OK") << " argmax\n";
    return failures ? 1 : 0;
}
