#include "ops/linear_pair/linear_pair_test_common.h"

#include "ninfer/ops/linear_pair.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::test::linear_pair {
namespace {

constexpr std::int32_t kOutputRows       = 1024;
constexpr std::size_t kOutputScanWords   = 1U << 20;
constexpr std::int32_t kDFlashParentRows = 6144;
constexpr std::int32_t kDFlashFirstRow   = 4096;
constexpr std::int32_t kDFlashSecondRow  = 5120;

// Both observable projections use the same A16 arithmetic profile. T and the selected pair
// launcher never select another correctness criterion.
constexpr ReductionCriterion kLinearPairA16Tolerance{
    2.9e-3,
    4.0e-3,
    3.8e-3,
};

struct LogicalWeight {
    std::int32_t k;
    std::uint64_t code_offset;
    std::uint64_t scale_offset;
    std::vector<std::int32_t> oracle_rows;
    std::vector<float> oracle_weight;

    Weight device_view(void* device_payload, std::size_t payload_bytes) const {
        Weight result{};
        result.payload          = device_payload;
        result.payload_bytes    = payload_bytes;
        result.high_plane_bytes = 0;
        result.qtype            = QType::W8G32_F16S;
        result.group_size       = 32;
        result.ndim             = 2;
        result.shape[0]         = kOutputRows;
        result.shape[1]         = k;
        result.padded_shape[0]  = kOutputRows;
        result.padded_shape[1]  = k;
        result.qdata            = static_cast<std::uint8_t*>(device_payload) + code_offset;
        result.qhigh            = nullptr;
        result.scales           = static_cast<std::uint8_t*>(device_payload) + scale_offset;
        result.n                = kOutputRows;
        result.k                = k;
        result.group            = 32;
        result.layout           = QuantLayout::RowSplit;
        result.scale_dtype      = DType::FP16;
        return result;
    }
};

struct PairFixture {
    bool shared_payload;
    std::vector<std::uint8_t> first_payload;
    std::vector<std::uint8_t> second_payload;
    LogicalWeight first;
    LogicalWeight second;
};

std::size_t checked_elements(std::int32_t first, std::int32_t second, const char* label) {
    if (first <= 0 || second <= 0) {
        throw std::invalid_argument(std::string("linear_pair test: invalid ") + label + " extent");
    }
    const std::size_t a = static_cast<std::size_t>(first);
    const std::size_t b = static_cast<std::size_t>(second);
    if (a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error(std::string("linear_pair test: ") + label + " size overflow");
    }
    return a * b;
}

std::uint64_t mix64(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

float bits_to_float(std::uint32_t bits) {
    float value = 0.0F;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

float bf16_to_float(std::uint16_t bits) {
    return bits_to_float(static_cast<std::uint32_t>(bits) << 16);
}

float fp16_to_float(std::uint16_t bits) {
    const std::uint32_t sign = (static_cast<std::uint32_t>(bits) & 0x8000U) << 16;
    std::uint32_t exponent   = (static_cast<std::uint32_t>(bits) >> 10) & 0x1fU;
    std::uint32_t fraction   = static_cast<std::uint32_t>(bits) & 0x03ffU;
    if (exponent == 0) {
        if (fraction == 0) { return bits_to_float(sign); }
        int unbiased = -14;
        while ((fraction & 0x0400U) == 0) {
            fraction <<= 1;
            --unbiased;
        }
        fraction &= 0x03ffU;
        return bits_to_float(sign | (static_cast<std::uint32_t>(unbiased + 127) << 23) |
                             (fraction << 13));
    }
    if (exponent == 31) { return bits_to_float(sign | 0x7f800000U | (fraction << 13)); }
    exponent = exponent - 15 + 127;
    return bits_to_float(sign | (exponent << 23) | (fraction << 13));
}

void store_u16_le(std::vector<std::uint8_t>& bytes, std::size_t offset, std::uint16_t value) {
    bytes[offset]     = static_cast<std::uint8_t>(value & 0xffU);
    bytes[offset + 1] = static_cast<std::uint8_t>(value >> 8);
}

std::uint16_t load_u16_le(const std::vector<std::uint8_t>& bytes, std::size_t offset) {
    return static_cast<std::uint16_t>(bytes[offset]) |
           static_cast<std::uint16_t>(static_cast<std::uint16_t>(bytes[offset + 1]) << 8);
}

std::vector<std::int32_t> sampled_indices(std::int32_t extent) {
    std::vector<std::int32_t> result;
    for (const std::int32_t index :
         {0, 1, extent / 4, extent / 2, (3 * extent) / 4, extent - 2, extent - 1}) {
        if (index >= 0 && index < extent &&
            std::find(result.begin(), result.end(), index) == result.end()) {
            result.push_back(index);
        }
    }
    std::sort(result.begin(), result.end());
    return result;
}

std::vector<std::int32_t> conformance_tokens(const ShapeCase& shape) {
    std::vector<std::int32_t> result{1};
    for (const std::int32_t boundary : shape.route_starts) {
        if (boundary <= 1) {
            throw std::invalid_argument("linear_pair test: route starts must be greater than one");
        }
        result.push_back(boundary - 1);
        result.push_back(boundary);
        if (boundary != std::numeric_limits<std::int32_t>::max()) {
            result.push_back(boundary + 1);
        }
    }
    for (const std::int32_t interior : shape.route_interiors) {
        if (interior <= 0) {
            throw std::invalid_argument("linear_pair test: route interior must be positive");
        }
        result.push_back(interior);
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

std::vector<std::uint8_t> make_w8_payload(std::int32_t rows, std::int32_t k, std::uint32_t seed) {
    if ((k % 32) != 0) { throw std::invalid_argument("linear_pair test: W8 K must divide 32"); }
    const std::int32_t groups_per_row = k / 32;
    const std::size_t group_count     = checked_elements(rows, groups_per_row, "weight groups");
    const std::size_t code_bytes      = checked_elements(rows, k, "weight codes");
    std::vector<std::uint8_t> result(code_bytes + group_count * sizeof(std::uint16_t), 0);

    std::array<std::array<std::uint8_t, 32>, 256> patterns{};
    for (std::size_t pattern = 0; pattern < patterns.size(); ++pattern) {
        std::uint64_t state = mix64((static_cast<std::uint64_t>(seed) << 32) | pattern);
        for (std::uint8_t& value : patterns[pattern]) {
            state = mix64(state);
            value = static_cast<std::uint8_t>(state >> 56);
            if (value == 0x80U) { value = 0x81U; }
        }
    }

    constexpr std::array<std::uint16_t, 4> kScaleBits{
        0x1800U,
        0x1c00U,
        0x2000U,
        0x2400U,
    };
    for (std::size_t group = 0; group < group_count; ++group) {
        const std::uint64_t mixed = mix64(group ^ (static_cast<std::uint64_t>(seed) << 17));
        const std::size_t pattern = static_cast<std::size_t>(mixed & 0xffU);
        std::memcpy(result.data() + group * 32U, patterns[pattern].data(), 32U);
        store_u16_le(result, code_bytes + group * sizeof(std::uint16_t),
                     kScaleBits[static_cast<std::size_t>((mixed >> 8) & 3U)]);
    }
    return result;
}

LogicalWeight make_logical_weight(const std::vector<std::uint8_t>& payload,
                                  std::int32_t physical_rows, std::int32_t first_row,
                                  std::int32_t k, std::span<const std::int32_t> oracle_rows) {
    const std::int32_t groups_per_row  = k / 32;
    const std::size_t code_plane_bytes = checked_elements(physical_rows, k, "code plane");
    LogicalWeight result{
        k,
        static_cast<std::uint64_t>(first_row) * static_cast<std::uint64_t>(k),
        code_plane_bytes + static_cast<std::uint64_t>(first_row) *
                               static_cast<std::uint64_t>(groups_per_row) * sizeof(std::uint16_t),
        std::vector<std::int32_t>(oracle_rows.begin(), oracle_rows.end()),
        {},
    };
    result.oracle_weight.resize(
        checked_elements(static_cast<std::int32_t>(oracle_rows.size()), k, "oracle weight"));
    for (std::size_t oracle_row = 0; oracle_row < oracle_rows.size(); ++oracle_row) {
        const std::int32_t physical_row = first_row + oracle_rows[oracle_row];
        for (std::int32_t column = 0; column < k; ++column) {
            const auto code = static_cast<std::int8_t>(
                payload[static_cast<std::size_t>(physical_row) * k + column]);
            const std::size_t group = static_cast<std::size_t>(physical_row) * groups_per_row +
                                      static_cast<std::size_t>(column / 32);
            const float scale = fp16_to_float(
                load_u16_le(payload, code_plane_bytes + group * sizeof(std::uint16_t)));
            result.oracle_weight[oracle_row * static_cast<std::size_t>(k) +
                                 static_cast<std::size_t>(column)] =
                static_cast<float>(code) * scale;
        }
    }
    return result;
}

PairFixture make_pair_fixture(std::int32_t k, std::uint32_t seed,
                              std::span<const std::int32_t> oracle_rows) {
    if (k == 5120) {
        PairFixture result{};
        result.shared_payload = false;
        result.first_payload  = make_w8_payload(kOutputRows, k, seed);
        result.second_payload = make_w8_payload(kOutputRows, k, seed + 1U);
        result.first  = make_logical_weight(result.first_payload, kOutputRows, 0, k, oracle_rows);
        result.second = make_logical_weight(result.second_payload, kOutputRows, 0, k, oracle_rows);
        return result;
    }
    if (k == 2048) {
        PairFixture result{};
        result.shared_payload = true;
        result.first_payload  = make_w8_payload(kDFlashParentRows, k, seed);
        result.first = make_logical_weight(result.first_payload, kDFlashParentRows, kDFlashFirstRow,
                                           k, oracle_rows);
        result.second = make_logical_weight(result.first_payload, kDFlashParentRows,
                                            kDFlashSecondRow, k, oracle_rows);
        return result;
    }
    throw std::invalid_argument("linear_pair test: unsupported K");
}

std::vector<std::uint16_t> make_activation(std::int32_t k, std::int32_t t, std::uint32_t seed) {
    std::vector<std::uint16_t> result(checked_elements(k, t, "activation"));
    std::vector<std::uint16_t> patterns(static_cast<std::size_t>(256) * k);
    for (int offset = 0; offset < 256; ++offset) {
        for (std::int32_t column = 0; column < k; ++column) {
            const int raw = (column * 17 + offset) & 0xff;
            patterns[static_cast<std::size_t>(offset) * k + column] =
                test::f32_to_bf16(static_cast<float>(raw - 128) * (1.0F / 256.0F));
        }
    }
    for (std::int32_t token = 0; token < t; ++token) {
        const int offset =
            static_cast<int>((static_cast<std::uint64_t>(token) * 31U + seed * 13U) & 0xffU);
        std::memcpy(result.data() + static_cast<std::size_t>(token) * k,
                    patterns.data() + static_cast<std::size_t>(offset) * k,
                    static_cast<std::size_t>(k) * sizeof(std::uint16_t));
    }
    return result;
}

std::vector<double> linear_pair_oracle(const LogicalWeight& weight,
                                       const std::vector<std::uint16_t>& activation,
                                       std::span<const std::int32_t> columns) {
    const std::int32_t oracle_n = static_cast<std::int32_t>(weight.oracle_rows.size());
    std::vector<double> result(
        checked_elements(oracle_n, static_cast<std::int32_t>(columns.size()), "oracle output"));
    for (std::size_t selected_column = 0; selected_column < columns.size(); ++selected_column) {
        const std::int32_t token = columns[selected_column];
        const std::uint16_t* activation_column =
            activation.data() + static_cast<std::size_t>(token) * weight.k;
        for (std::int32_t oracle_row = 0; oracle_row < oracle_n; ++oracle_row) {
            double sum = 0.0;
            const float* weight_row =
                weight.oracle_weight.data() + static_cast<std::size_t>(oracle_row) * weight.k;
            for (std::int32_t column = 0; column < weight.k; ++column) {
                sum += static_cast<double>(weight_row[column]) *
                       static_cast<double>(bf16_to_float(activation_column[column]));
            }
            result[selected_column * static_cast<std::size_t>(oracle_n) + oracle_row] = sum;
        }
    }
    return result;
}

struct OutputRead {
    int failures = 0;
    std::vector<double> selected;
};

OutputRead read_output(const void* device, std::int32_t t, std::span<const std::int32_t> rows,
                       std::span<const std::int32_t> columns, std::string_view label) {
    const std::size_t total_words = checked_elements(kOutputRows, t, "output");
    std::vector<std::size_t> wanted;
    wanted.reserve(rows.size() * columns.size());
    for (const std::int32_t column : columns) {
        for (const std::int32_t row : rows) {
            wanted.push_back(static_cast<std::size_t>(column) * kOutputRows + row);
        }
    }

    OutputRead result;
    result.selected.resize(wanted.size());
    std::vector<std::uint16_t> chunk(std::min(kOutputScanWords, total_words));
    std::size_t wanted_index    = 0;
    std::size_t poison_count    = 0;
    std::size_t nonfinite_count = 0;
    for (std::size_t begin = 0; begin < total_words; begin += chunk.size()) {
        const std::size_t count = std::min(chunk.size(), total_words - begin);
        test::cuda_check(
            cudaMemcpy(chunk.data(),
                       static_cast<const std::uint8_t*>(device) + begin * sizeof(std::uint16_t),
                       count * sizeof(std::uint16_t), cudaMemcpyDeviceToHost),
            "copy linear_pair output");
        for (const std::uint16_t bits : std::span<const std::uint16_t>(chunk.data(), count)) {
            if (bits == 0xffffU) { ++poison_count; }
            if ((bits & 0x7f80U) == 0x7f80U) { ++nonfinite_count; }
        }
        while (wanted_index < wanted.size() && wanted[wanted_index] < begin + count) {
            result.selected[wanted_index] =
                static_cast<double>(bf16_to_float(chunk[wanted[wanted_index] - begin]));
            ++wanted_index;
        }
    }
    if (poison_count != 0) {
        std::cerr << label << ": output retains " << poison_count << " poison values\n";
        ++result.failures;
    }
    if (nonfinite_count != 0) {
        std::cerr << label << ": output contains " << nonfinite_count << " non-finite values\n";
        ++result.failures;
    }
    return result;
}

int compare_output(std::string_view label, std::span<const double> actual,
                   std::span<const double> reference) {
    return verify_reduction(label, actual, reference, kLinearPairA16Tolerance);
}

int verify_preserved(const test::GuardedDeviceBuffer& device,
                     std::span<const std::uint8_t> expected, const char* label) {
    std::vector<std::uint8_t> actual(expected.size());
    device.copy_to_host(actual.data(), actual.size());
    if (std::equal(actual.begin(), actual.end(), expected.begin(), expected.end())) { return 0; }
    std::cerr << label << ": input payload was modified\n";
    return 1;
}

} // namespace

bool cuda_available() { return !test::cuda_unavailable(); }

int run_w8_a16_shape(std::string_view label, const ShapeCase& shape) {
    const std::vector<std::int32_t> tokens = conformance_tokens(shape);
    if (tokens.empty()) { throw std::invalid_argument("linear_pair test: no token cases"); }
    const std::int32_t maximum_t = tokens.back();

    const std::vector<std::int32_t> oracle_rows = sampled_indices(kOutputRows);
    PairFixture fixture = make_pair_fixture(shape.k, shape.seed, oracle_rows);
    const std::vector<std::uint16_t> activation =
        make_activation(shape.k, maximum_t, shape.seed + 2U);

    test::GuardedDeviceBuffer device_activation(activation.size() * sizeof(std::uint16_t));
    device_activation.copy_from_host(activation.data(), device_activation.bytes());
    test::GuardedDeviceBuffer first_device_weight(fixture.first_payload.size());
    first_device_weight.copy_from_host(fixture.first_payload.data(), fixture.first_payload.size());
    test::GuardedDeviceBuffer second_device_weight(
        fixture.shared_payload ? 1U : fixture.second_payload.size());
    if (!fixture.shared_payload) {
        second_device_weight.copy_from_host(fixture.second_payload.data(),
                                            fixture.second_payload.size());
    }

    void* const second_payload =
        fixture.shared_payload ? first_device_weight.data() : second_device_weight.data();
    const std::size_t second_payload_bytes =
        fixture.shared_payload ? fixture.first_payload.size() : fixture.second_payload.size();
    const Weight first_weight =
        fixture.first.device_view(first_device_weight.data(), fixture.first_payload.size());
    const Weight second_weight = fixture.second.device_view(second_payload, second_payload_bytes);
    WorkspaceArena workspace(256);

    int failures = 0;
    for (const std::int32_t t : tokens) {
        const std::size_t output_words = checked_elements(kOutputRows, t, "output");
        test::GuardedDeviceBuffer first_output(output_words * sizeof(std::uint16_t));
        test::GuardedDeviceBuffer second_output(output_words * sizeof(std::uint16_t));
        first_output.fill(0xff);
        second_output.fill(0xff);

        Tensor input(device_activation.data(), DType::BF16, {shape.k, t});
        Tensor first(first_output.data(), DType::BF16, {kOutputRows, t});
        Tensor second(second_output.data(), DType::BF16, {kOutputRows, t});
        workspace.reset();

        const std::string case_label =
            std::string(label) + " [1024," + std::to_string(shape.k) + "] T=" + std::to_string(t);
        try {
            ops::linear_pair(input, first_weight, second_weight, first, second, workspace, nullptr);
            test::cuda_check(cudaDeviceSynchronize(), "synchronize linear_pair");
        } catch (const std::exception& error) {
            std::cerr << case_label << ": unexpected exception: " << error.what() << '\n';
            ++failures;
            continue;
        }

        failures += first_output.verify_guards((case_label + " first").c_str());
        failures += second_output.verify_guards((case_label + " second").c_str());
        const std::vector<std::int32_t> columns = sampled_indices(t);
        const OutputRead first_actual =
            read_output(first_output.data(), t, oracle_rows, columns, case_label + " first");
        const OutputRead second_actual =
            read_output(second_output.data(), t, oracle_rows, columns, case_label + " second");
        failures += first_actual.failures + second_actual.failures;

        const std::vector<double> first_reference =
            linear_pair_oracle(fixture.first, activation, columns);
        const std::vector<double> second_reference =
            linear_pair_oracle(fixture.second, activation, columns);
        failures += compare_output(case_label + " first", first_actual.selected, first_reference);
        failures +=
            compare_output(case_label + " second", second_actual.selected, second_reference);
    }

    failures += device_activation.verify_guards("linear_pair activation");
    failures += first_device_weight.verify_guards("linear_pair first weight");
    if (!fixture.shared_payload) {
        failures += second_device_weight.verify_guards("linear_pair second weight");
    }
    failures += verify_preserved(
        device_activation,
        std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(activation.data()),
                                      activation.size() * sizeof(std::uint16_t)),
        "linear_pair activation");
    failures +=
        verify_preserved(first_device_weight, fixture.first_payload, "linear_pair first weight");
    if (!fixture.shared_payload) {
        failures += verify_preserved(second_device_weight, fixture.second_payload,
                                     "linear_pair second weight");
    }
    return failures;
}

} // namespace ninfer::test::linear_pair
