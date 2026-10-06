#include "ninfer/ops/linear.h"

#include "core/decode_graph.h"
#include "core/device.h"
#include "ops/linear/context_kv_oracle.h"
#include "ops/op_tester.h"
#include "ops/sanitizer_scope.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {
using namespace ninfer;
namespace oracle = ninfer::test::context_kv;

constexpr test::ReductionCriterion kA16Criterion{1.0 / 256, 1.0 / 256, 2.0 / 256};

int check_admission(DeviceContext& device, const Weight& weight) {
    DeviceBuffer input(static_cast<std::size_t>(oracle::kHidden) * 3 * 2);
    DeviceBuffer key_storage(1024 * 3 * 2);
    DeviceBuffer value_storage(1024 * 3 * 2);
    Tensor x(input.p, DType::BF16, {oracle::kHidden, 3});
    Tensor key(key_storage.p, DType::BF16, {1024, 3});
    Tensor value(value_storage.p, DType::BF16, {1024, 3});
    int failures = test::expect_invalid_argument(
        [&] { ops::linear_kv_projection(x, weight, key, key, device.stream, 3); },
        "context KV accepted overlapping K/V outputs");
    Weight forged          = weight;
    forged.n               = 2048;
    forged.shape[0]        = 2048;
    forged.padded_shape[0] = 2048;
    failures += test::expect_invalid_argument(
        [&] { ops::linear_kv_projection(x, forged, key, value, device.stream, 3); },
        "context KV accepted a fabricated subset Weight");
    Weight truncated        = weight;
    truncated.payload_bytes = truncated.payload_bytes / 3;
    failures += test::expect_invalid_argument(
        [&] { ops::linear_kv_projection(x, truncated, key, value, device.stream, 3); },
        "context KV accepted an incomplete original payload");
    failures += test::expect_invalid_argument(
        [&] { ops::linear_kv_projection(x, weight, key, value, device.stream, 2); },
        "context KV accepted a partial packed sequence");
    return failures;
}

int run_case(DeviceContext& device, const Weight& weight, const std::vector<double>& logical,
             std::int32_t width, std::int32_t batch) {
    const auto tokens  = width * batch;
    const auto words   = static_cast<std::size_t>(1024) * tokens;
    const char* format = weight.qtype == QType::NVFP4        ? "NVFP4"
                         : weight.qtype == QType::W8G32_F16S ? "W8"
                                                             : "Q4";
    const auto label   = std::string("context KV ") + format + " W=" + std::to_string(width) +
                         " B=" + std::to_string(batch);
    std::vector<float> activation(static_cast<std::size_t>(oracle::kHidden) * tokens);
    test::fill_uniform(activation, static_cast<std::uint32_t>(width * 13 + batch), -0.5F, 0.5F);
    test::round_to_bf16(activation);
    auto input = test::to_device_bf16(activation);
    test::GuardedDeviceBuffer key_storage(words * 2);
    test::GuardedDeviceBuffer value_storage(words * 2);
    Tensor x(input.p, DType::BF16, {oracle::kHidden, tokens});
    Tensor key(key_storage.data(), DType::BF16, {1024, tokens});
    Tensor value(value_storage.data(), DType::BF16, {1024, tokens});
    const auto launch = [&] {
        ops::linear_kv_projection(x, weight, key, value, device.stream, width);
    };

    // Full short-append outputs are checked. For ingest, complete dots cover all output rows
    // at every 48- and 64-column tile boundary and the last column; every stored output is scanned
    // finite.
    std::vector<std::int32_t> columns;
    for (std::int32_t column = 0; column < tokens; ++column) {
        if (width <= 8 || column % 48 == 0 || column % 48 == 47 || column % 64 == 0 ||
            column % 64 == 63 || column == tokens - 1) {
            columns.push_back(column);
        }
    }
    std::vector<float> selected_activation;
    selected_activation.reserve(columns.size() * oracle::kHidden);
    for (const auto column : columns) {
        const auto first = activation.begin() + static_cast<std::int64_t>(column) * oracle::kHidden;
        selected_activation.insert(selected_activation.end(), first, first + oracle::kHidden);
    }
    auto reference = oracle::project(logical, selected_activation);
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    int failures = 0;
    for (int replay = 0; replay < 3; ++replay) {
        key_storage.fill(0xff);
        value_storage.fill(0xff);
        // Fixture cudaMemset uses the default stream; compute uses a nonblocking stream.
        test::cuda_synchronize(nullptr);
        if (replay == 0) {
            launch();
            device.synchronize();
            definition.capture(device.stream, launch);
            graph.instantiate(definition);
        } else {
            // Replay changes represented inputs at stable addresses, detecting stale capture data.
            if (replay == 2) {
                for (float& number : activation) { number = -number; }
                std::vector<std::uint16_t> bits(activation.size());
                std::transform(activation.begin(), activation.end(), bits.begin(),
                               test::f32_to_bf16);
                input.copy_from_host(bits.data(), input.bytes);
                for (double& number : reference) { number = -number; }
            }
            graph.launch(device.stream);
            device.synchronize();
        }
        failures += key_storage.verify_guards(label + " key");
        failures += value_storage.verify_guards(label + " value");
        const auto key_bits   = test::from_device<std::uint16_t>(key.data, words);
        const auto value_bits = test::from_device<std::uint16_t>(value.data, words);
        std::vector<double> actual_key(columns.size() * 1024);
        std::vector<double> actual_value(columns.size() * 1024);
        std::vector<double> reference_key(columns.size() * 1024);
        std::vector<double> reference_value(columns.size() * 1024);
        for (std::size_t index = 0; index < words; ++index) {
            if (!std::isfinite(test::bf16_to_f32(key_bits[index])) ||
                !std::isfinite(test::bf16_to_f32(value_bits[index]))) {
                throw std::runtime_error(label + ": non-finite output");
            }
        }
        for (std::size_t selected = 0; selected < columns.size(); ++selected) {
            const auto column = static_cast<std::size_t>(columns[selected]);
            for (std::size_t row = 0; row < 1024; ++row) {
                actual_key[selected * 1024 + row] =
                    test::bf16_to_f32(key_bits[column * 1024 + row]);
                actual_value[selected * 1024 + row] =
                    test::bf16_to_f32(value_bits[column * 1024 + row]);
                reference_key[selected * 1024 + row] =
                    reference[selected * oracle::kSelectedRows + row];
                reference_value[selected * 1024 + row] =
                    reference[selected * oracle::kSelectedRows + row + 1024];
            }
        }
        failures +=
            test::verify_reduction(label + " key", actual_key, reference_key, kA16Criterion);
        failures +=
            test::verify_reduction(label + " value", actual_value, reference_value, kA16Criterion);
        const auto preserved = test::from_device<std::uint16_t>(input, activation.size());
        for (std::size_t index = 0; index < preserved.size(); ++index) {
            if (preserved[index] != test::f32_to_bf16(activation[index])) {
                throw std::runtime_error(label + ": input modified");
            }
        }
        if (replay == 0 && batch > 1) {
            for (std::int32_t sequence = 0; sequence < batch; ++sequence) {
                Tensor key_panel   = key.slice(1, sequence * width, width);
                Tensor value_panel = value.slice(1, sequence * width, width);
                ops::linear_kv_projection(x.slice(1, sequence * width, width), weight, key_panel,
                                          value_panel, device.stream, width);
            }
            device.synchronize();
            if (test::from_device<std::uint16_t>(key.data, words) != key_bits ||
                test::from_device<std::uint16_t>(value.data, words) != value_bits) {
                throw std::runtime_error(label + ": packed sequence changed its C=1 result");
            }
        }
    }
    return failures;
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (test::require_cuda() != 0) { return 1; }
        DeviceContext device;
        constexpr std::uint32_t kSeed = 831;
        auto packed                   = oracle::make_weight(kSeed, 128.0F);
        const auto logical            = oracle::decode_selected(packed);
        oracle::verify_exact_codec(logical, kSeed, 128.0F);
        auto payload      = test::to_device(packed.payload);
        const auto weight = packed.device_weight(payload.p);
        int failures      = check_admission(device, weight);
        if (test::sanitizer_scope(argc, argv)) {
            // Cover GEMV, both narrow MMA schedules, M48 (with a partial ingest tile at 160), and
            // the M64N32, M64N64 and M64N128 prefill tiles with partial last tiles.
            for (const auto& shape :
                 {std::array{1, 1}, std::array{3, 2}, std::array{3, 6}, std::array{8, 6},
                  std::array{100, 1}, std::array{160, 1}, std::array{300, 1}, std::array{600, 1}}) {
                failures += run_case(device, weight, logical, shape[0], shape[1]);
            }
        } else {
            for (const auto batch : {1, 2, 6}) {
                for (const auto width : {1, 2, 3, 4, 5, 6, 7, 8, 12}) {
                    failures += run_case(device, weight, logical, width, batch);
                }
            }
            // Widths inside every prefill tier of nvfp4_dflash_kv_a16_tile, partial last tiles
            // included.
            for (const auto width : {100, 128, 160, 300, 400, 512, 700, 900, 1000, 2048}) {
                failures += run_case(device, weight, logical, width, 1);
            }
        }
        auto second_packed        = oracle::make_weight(kSeed + 1, 37.5F);
        const auto second_logical = oracle::decode_selected(second_packed);
        oracle::verify_exact_codec(second_logical, kSeed + 1, 37.5F);
        auto second_payload = test::to_device(second_packed.payload);
        failures +=
            run_case(device, second_packed.device_weight(second_payload.p), second_logical, 4, 2);
        if (test::from_device<std::uint8_t>(payload, packed.payload.size()) != packed.payload) {
            throw std::runtime_error("context KV: persistent weight modified");
        }
        if (!test::sanitizer_scope(argc, argv)) {
            for (const auto qtype : {QType::W8G32_F16S, QType::Q4G64_F16S}) {
                auto row_split = test::quantized_weight::make_patterned_weight(
                    qtype, oracle::kParentRows, oracle::kHidden, kSeed + 2,
                    {test::quantized_weight::RowSplitScalePattern::Small,
                     test::quantized_weight::RowSplitCodePattern::Hashed});
                const auto row_split_logical = oracle::decode_selected_row_split(row_split);
                auto row_split_payload       = test::to_device(row_split.payload);
                const auto row_split_weight  = row_split.device_weight(row_split_payload.p);
                failures += check_admission(device, row_split_weight);
                for (const auto& shape : {std::array{1, 1}, std::array{3, 1}, std::array{3, 6},
                                          std::array{8, 6}, std::array{12, 1}, std::array{12, 2},
                                          std::array{12, 6}, std::array{128, 1}}) {
                    failures +=
                        run_case(device, row_split_weight, row_split_logical, shape[0], shape[1]);
                }
                if (test::from_device<std::uint8_t>(row_split_payload, row_split.payload.size()) !=
                    row_split.payload) {
                    throw std::runtime_error("context KV: original RowSplit weight modified");
                }
            }
        }
        std::cout << (failures == 0 ? "OK" : "FAIL") << " A16 context KV projection\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "context KV projection: " << error.what() << '\n';
        return 1;
    }
}
