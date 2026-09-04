#include "ops/linear_add/linear_add_test_common.h"

#include "ninfer/ops/linear_add.h"
#include "ops/direct_bf16_weight.h"
#include "ops/op_tester.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using ninfer::test::linear_add::ShapeCase;
using ninfer::test::linear_add::WeightFormat;

template <class Callable>
int expect_invalid(Callable&& callable, const char* label) {
    try {
        callable();
    } catch (const std::invalid_argument&) { return 0; } catch (const std::exception& error) {
        std::cerr << label << ": expected invalid_argument, got " << error.what() << '\n';
        return 1;
    }
    std::cerr << label << ": expected invalid_argument\n";
    return 1;
}

int bf16_a16_rejections() {
    int failures = 0;
    failures += expect_invalid(
        [] {
            (void)ninfer::ops::linear_add_workspace_capacity_bytes(ninfer::QType::BF16_CTRL, 5120,
                                                                   6143, 1, 32);
        },
        "BF16_A16 LinearAdd workspace shape");

    ninfer::DeviceBuffer input(static_cast<std::size_t>(6144) * sizeof(std::uint16_t));
    ninfer::DeviceBuffer residual(static_cast<std::size_t>(5120) * sizeof(std::uint16_t));
    ninfer::DeviceBuffer weight_storage(256);
    ninfer::Weight weight{};
    weight.qtype  = ninfer::QType::BF16_CTRL;
    weight.layout = ninfer::QuantLayout::RowSplit;
    weight.qdata  = weight_storage.p;
    weight.n      = 5120;
    weight.k      = 6144;
    ninfer::Tensor x(input.p, ninfer::DType::BF16, {6144, 1});
    ninfer::Tensor out(residual.p, ninfer::DType::BF16, {5120, 1});
    ninfer::WorkspaceArena workspace(1);
    failures += expect_invalid([&] { ninfer::ops::linear_add(x, weight, out, workspace, nullptr); },
                               "BF16_A16 LinearAdd weight layout");
    return failures;
}

int bf16_a16_conformance() {
    constexpr std::array<std::int32_t, 3> kRouteStarts{2, 5, 49};
    constexpr std::array<std::int32_t, 21> kRouteInteriors{
        4,  5,  8,  10, 15, 16, 20,  24,  28,  32, 36,
        44, 48, 52, 60, 64, 127, 128, 129, 1024, 1536,
    };
    return ninfer::test::linear_add::run_shape(
               "BF16_A16 LinearAdd", WeightFormat::BF16,
               ShapeCase{5120, 6144, 431U, kRouteStarts, kRouteInteriors}) +
           bf16_a16_rejections();
}

int bf16_w5_aggregate_matches_panels() {
    constexpr std::int32_t kN        = 5120;
    constexpr std::int32_t kK        = 6144;
    constexpr std::int32_t kWidth    = 5;
    constexpr std::int32_t kMaxBatch = 4;
    constexpr std::int32_t kMaxT     = kWidth * kMaxBatch;
    ninfer::test::direct_bf16_weight::DeviceWeight weight(
        ninfer::test::direct_bf16_weight::make_patterned(kN, kK, 449U));
    std::vector<std::uint16_t> activation(static_cast<std::size_t>(kK) * kMaxT);
    std::vector<std::uint16_t> residual(static_cast<std::size_t>(kN) * kMaxT);
    for (std::size_t index = 0; index < activation.size(); ++index) {
        const int value = static_cast<int>((index * 17U + 29U) & 0xffU) - 128;
        activation[index] = ninfer::test::f32_to_bf16(static_cast<float>(value) / 256.0F);
    }
    for (std::size_t index = 0; index < residual.size(); ++index) {
        const int value = static_cast<int>((index * 23U + 11U) & 0xffU) - 128;
        residual[index] = ninfer::test::f32_to_bf16(static_cast<float>(value) / 128.0F);
    }
    ninfer::DeviceBuffer device_activation(activation.size() * sizeof(std::uint16_t));
    device_activation.copy_from_host(activation.data(), device_activation.bytes);

    int failures = 0;
    for (std::int32_t batch = 2; batch <= kMaxBatch; ++batch) {
        const std::int32_t tokens       = kWidth * batch;
        const std::size_t output_words = static_cast<std::size_t>(kN) * tokens;
        const std::size_t output_bytes = output_words * sizeof(std::uint16_t);
        ninfer::test::GuardedDeviceBuffer aggregate(output_bytes);
        ninfer::test::GuardedDeviceBuffer panels(output_bytes);
        aggregate.copy_from_host(residual.data(), output_bytes);
        panels.copy_from_host(residual.data(), output_bytes);
        ninfer::WorkspaceArena workspace(256);

        ninfer::Tensor aggregate_x(device_activation.p, ninfer::DType::BF16, {kK, tokens});
        ninfer::Tensor aggregate_y(aggregate.data(), ninfer::DType::BF16, {kN, tokens});
        ninfer::ops::linear_add(aggregate_x, weight.view(), aggregate_y, workspace, nullptr);
        for (std::int32_t row = 0; row < batch; ++row) {
            auto* input = static_cast<std::uint8_t*>(device_activation.p) +
                          static_cast<std::int64_t>(row) * kWidth * kK * sizeof(std::uint16_t);
            auto* output = static_cast<std::uint8_t*>(panels.data()) +
                           static_cast<std::int64_t>(row) * kWidth * kN * sizeof(std::uint16_t);
            ninfer::Tensor panel_x(input, ninfer::DType::BF16, {kK, kWidth});
            ninfer::Tensor panel_y(output, ninfer::DType::BF16, {kN, kWidth});
            ninfer::ops::linear_add(panel_x, weight.view(), panel_y, workspace, nullptr);
        }
        ninfer::test::cuda_check(cudaDeviceSynchronize(), "synchronize BF16 W5 aggregate parity");

        std::vector<std::uint16_t> aggregate_bits(output_words);
        std::vector<std::uint16_t> panel_bits(output_words);
        aggregate.copy_to_host(aggregate_bits.data(), output_bytes);
        panels.copy_to_host(panel_bits.data(), output_bytes);
        const std::string label = "BF16 linear_add W5 aggregate parity C=" +
                                  std::to_string(batch);
        if (aggregate_bits != panel_bits) {
            std::cerr << label << ": aggregate and panel outputs differ\n";
            ++failures;
        }
        failures += aggregate.verify_guards(label + " aggregate");
        failures += panels.verify_guards(label + " panels");
    }
    failures += weight.verify_preserved("BF16 W5 aggregate parity weight");
    return failures;
}

} // namespace

int main() {
    if (!ninfer::test::linear_add::cuda_available()) {
        std::cerr << "FAIL: no usable CUDA device\n";
        return 1;
    }

    try {
        const int failures = bf16_a16_conformance() + bf16_w5_aggregate_matches_panels();
        std::cout << (failures == 0 ? "OK" : "FAIL") << " BF16_A16 LinearAdd\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "BF16_A16 LinearAdd: " << error.what() << '\n';
        return 1;
    }
}
