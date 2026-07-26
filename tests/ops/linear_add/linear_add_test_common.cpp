#include "ops/linear_add/linear_add_test_common.h"

#include "ninfer/ops/linear_add.h"
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
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::test::linear_add {
namespace {

constexpr std::size_t kOutputScanWords = 1U << 20;

// One criterion for the complete A16 fused Op. It is not selected by T, route, or kernel.
constexpr ReductionCriterion kLinearAddA16Tolerance{
    4.0e-3,
    4.0e-3,
    5.5e-3,
};

struct PackedWeight {
    WeightFormat format;
    std::int32_t n;
    std::int32_t k;
    std::int32_t group_size;
    std::uint64_t high_offset;
    std::uint64_t high_bytes;
    std::uint64_t scale_offset;
    std::vector<std::uint8_t> payload;
    std::vector<std::int32_t> oracle_rows;
    std::vector<float> oracle_weight;

    Weight device_view(void* device_payload) const {
        Weight result{};
        result.payload          = device_payload;
        result.payload_bytes    = payload.size();
        result.high_plane_bytes = high_bytes;
        result.qtype = format == WeightFormat::Q5G64F16S ? QType::Q5G64_F16S : QType::W8G32_F16S;
        result.group_size      = static_cast<std::uint32_t>(group_size);
        result.ndim            = 2;
        result.shape[0]        = n;
        result.shape[1]        = k;
        result.padded_shape[0] = n;
        result.padded_shape[1] = k;
        result.qdata           = device_payload;
        result.qhigh =
            high_bytes == 0 ? nullptr : static_cast<std::uint8_t*>(device_payload) + high_offset;
        result.scales      = static_cast<std::uint8_t*>(device_payload) + scale_offset;
        result.n           = n;
        result.k           = k;
        result.group       = group_size;
        result.layout      = QuantLayout::RowSplit;
        result.scale_dtype = DType::FP16;
        return result;
    }
};

std::size_t checked_elements(std::int32_t first, std::int32_t second, const char* label) {
    if (first <= 0 || second <= 0) {
        throw std::invalid_argument(std::string("linear_add test: invalid ") + label + " extent");
    }
    const std::size_t a = static_cast<std::size_t>(first);
    const std::size_t b = static_cast<std::size_t>(second);
    if (a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error(std::string("linear_add test: ") + label + " size overflow");
    }
    return a * b;
}

std::size_t align_up(std::size_t value, std::size_t alignment) {
    if (value > std::numeric_limits<std::size_t>::max() - (alignment - 1)) {
        throw std::overflow_error("linear_add test: alignment overflow");
    }
    return ((value + alignment - 1) / alignment) * alignment;
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

std::vector<std::int32_t> all_indices(std::int32_t extent) {
    std::vector<std::int32_t> result(static_cast<std::size_t>(extent));
    std::iota(result.begin(), result.end(), 0);
    return result;
}

std::vector<std::int32_t> conformance_tokens(const ShapeCase& shape) {
    std::vector<std::int32_t> result{1};
    for (const std::int32_t boundary : shape.route_starts) {
        if (boundary <= 1) {
            throw std::invalid_argument("linear_add test: route starts must be greater than one");
        }
        result.push_back(boundary - 1);
        result.push_back(boundary);
        if (boundary != std::numeric_limits<std::int32_t>::max()) {
            result.push_back(boundary + 1);
        }
    }
    for (const std::int32_t interior : shape.route_interiors) {
        if (interior <= 0) {
            throw std::invalid_argument("linear_add test: route interior must be positive");
        }
        result.push_back(interior);
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

PackedWeight make_q5_weight(std::int32_t n, std::int32_t k, std::uint32_t seed,
                            std::span<const std::int32_t> oracle_rows) {
    if ((k % 64) != 0) { throw std::invalid_argument("linear_add test: Q5 K must divide 64"); }
    const std::int32_t groups_per_row = k / 64;
    const std::size_t group_count     = checked_elements(n, groups_per_row, "Q5 weight groups");
    const std::size_t base_bytes      = group_count * 32U;
    const std::size_t high_offset     = align_up(base_bytes, 256);
    const std::size_t high_bytes      = group_count * 8U;
    const std::size_t scale_offset    = high_offset + align_up(high_bytes, 256);

    PackedWeight result{
        WeightFormat::Q5G64F16S,
        n,
        k,
        64,
        high_offset,
        high_bytes,
        scale_offset,
        std::vector<std::uint8_t>(scale_offset + group_count * sizeof(std::uint16_t), 0),
        std::vector<std::int32_t>(oracle_rows.begin(), oracle_rows.end()),
        {},
    };

    std::array<std::array<std::uint8_t, 32>, 256> biased_low_patterns{};
    std::array<std::array<std::uint8_t, 8>, 256> biased_high_patterns{};
    std::array<std::array<std::uint8_t, 32>, 256> random_low_patterns{};
    std::array<std::array<std::uint8_t, 8>, 256> random_high_patterns{};
    for (std::size_t pattern = 0; pattern < random_low_patterns.size(); ++pattern) {
        std::uint64_t state = mix64((static_cast<std::uint64_t>(seed) << 32) | pattern);
        for (std::uint8_t& value : random_low_patterns[pattern]) {
            state = mix64(state);
            value = static_cast<std::uint8_t>(state >> 56);
        }
        for (std::uint8_t& value : random_high_patterns[pattern]) {
            state = mix64(state);
            value = static_cast<std::uint8_t>(state >> 56);
        }
        const int negative_lane = static_cast<int>(pattern % 64U);
        for (int lane = 0; lane < 64; ++lane) {
            const int code =
                lane == negative_lane ? -16 : 2 + static_cast<int>((pattern + lane) % 14U);
            const std::uint32_t word = static_cast<std::uint32_t>(code) & 0x1fU;
            std::uint8_t& packed = biased_low_patterns[pattern][static_cast<std::size_t>(lane / 2)];
            if ((lane & 1) == 0) {
                packed = static_cast<std::uint8_t>((packed & 0xf0U) | (word & 0x0fU));
            } else {
                packed = static_cast<std::uint8_t>((packed & 0x0fU) | ((word & 0x0fU) << 4));
            }
            biased_high_patterns[pattern][static_cast<std::size_t>(lane / 8)] |=
                static_cast<std::uint8_t>(((word >> 4) & 1U) << (lane & 7));
        }
    }

    constexpr std::array<std::uint16_t, 4> kScaleBits{
        0x2000U,
        0x2400U,
        0x2800U,
        0x2c00U,
    };
    for (std::size_t group = 0; group < group_count; ++group) {
        const std::uint64_t mixed = mix64(group ^ (static_cast<std::uint64_t>(seed) << 17));
        const std::size_t pattern = static_cast<std::size_t>(mixed & 0xffU);
        const std::int32_t row =
            static_cast<std::int32_t>(group / static_cast<std::size_t>(groups_per_row));
        const bool oracle_row =
            std::binary_search(result.oracle_rows.begin(), result.oracle_rows.end(), row);
        const auto& low = oracle_row ? random_low_patterns[pattern] : biased_low_patterns[pattern];
        const auto& high =
            oracle_row ? random_high_patterns[pattern] : biased_high_patterns[pattern];
        std::memcpy(result.payload.data() + group * 32U, low.data(), 32U);
        std::memcpy(result.payload.data() + high_offset + group * 8U, high.data(), 8U);
        store_u16_le(result.payload, scale_offset + group * sizeof(std::uint16_t),
                     kScaleBits[static_cast<std::size_t>((mixed >> 8) & 3U)]);
    }

    result.oracle_weight.resize(checked_elements(
        static_cast<std::int32_t>(result.oracle_rows.size()), k, "Q5 oracle weight"));
    for (std::size_t oracle_row = 0; oracle_row < result.oracle_rows.size(); ++oracle_row) {
        const std::int32_t physical_row = result.oracle_rows[oracle_row];
        for (std::int32_t column = 0; column < k; ++column) {
            const std::size_t group = static_cast<std::size_t>(physical_row) * groups_per_row +
                                      static_cast<std::size_t>(column / 64);
            const std::uint8_t packed =
                result.payload[group * 32U + static_cast<std::size_t>((column % 64) / 2)];
            std::uint32_t code      = (column & 1) == 0 ? static_cast<std::uint32_t>(packed & 0x0fU)
                                                        : static_cast<std::uint32_t>(packed >> 4);
            const int lane          = column % 64;
            const std::uint8_t high = result.payload[high_offset + group * 8U + lane / 8];
            code |= static_cast<std::uint32_t>((high >> (lane & 7)) & 1U) << 4;
            const int signed_code =
                (code & 0x10U) != 0U ? static_cast<int>(code) - 32 : static_cast<int>(code);
            const float scale = fp16_to_float(
                load_u16_le(result.payload, scale_offset + group * sizeof(std::uint16_t)));
            result.oracle_weight[oracle_row * static_cast<std::size_t>(k) +
                                 static_cast<std::size_t>(column)] =
                static_cast<float>(signed_code) * scale;
        }
    }
    return result;
}

PackedWeight make_w8_weight(std::int32_t n, std::int32_t k, std::uint32_t seed,
                            std::span<const std::int32_t> oracle_rows) {
    if ((k % 32) != 0) { throw std::invalid_argument("linear_add test: W8 K must divide 32"); }
    const std::int32_t groups_per_row = k / 32;
    const std::size_t group_count     = checked_elements(n, groups_per_row, "W8 weight groups");
    const std::size_t code_bytes      = checked_elements(n, k, "W8 codes");
    const std::size_t scale_offset    = code_bytes;

    PackedWeight result{
        WeightFormat::W8G32F16S,
        n,
        k,
        32,
        0,
        0,
        scale_offset,
        std::vector<std::uint8_t>(scale_offset + group_count * sizeof(std::uint16_t), 0),
        std::vector<std::int32_t>(oracle_rows.begin(), oracle_rows.end()),
        {},
    };

    std::array<std::array<std::uint8_t, 32>, 256> biased_patterns{};
    std::array<std::array<std::uint8_t, 32>, 256> random_patterns{};
    for (std::size_t pattern = 0; pattern < random_patterns.size(); ++pattern) {
        std::uint64_t state = mix64((static_cast<std::uint64_t>(seed) << 32) | pattern);
        for (std::uint8_t& value : random_patterns[pattern]) {
            state = mix64(state);
            value = static_cast<std::uint8_t>(state >> 56);
            if (value == 0x80U) { value = 0x81U; }
        }
        const int negative_lane = static_cast<int>(pattern % 32U);
        for (int lane = 0; lane < 32; ++lane) {
            const int code =
                lane == negative_lane ? -127 : 32 + static_cast<int>((pattern + lane) % 96U);
            biased_patterns[pattern][static_cast<std::size_t>(lane)] =
                static_cast<std::uint8_t>(static_cast<std::int8_t>(code));
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
        const std::int32_t row =
            static_cast<std::int32_t>(group / static_cast<std::size_t>(groups_per_row));
        const bool oracle_row =
            std::binary_search(result.oracle_rows.begin(), result.oracle_rows.end(), row);
        const auto& selected = oracle_row ? random_patterns[pattern] : biased_patterns[pattern];
        std::memcpy(result.payload.data() + group * 32U, selected.data(), 32U);
        store_u16_le(result.payload, scale_offset + group * sizeof(std::uint16_t),
                     kScaleBits[static_cast<std::size_t>((mixed >> 8) & 3U)]);
    }

    result.oracle_weight.resize(checked_elements(
        static_cast<std::int32_t>(result.oracle_rows.size()), k, "W8 oracle weight"));
    for (std::size_t oracle_row = 0; oracle_row < result.oracle_rows.size(); ++oracle_row) {
        const std::int32_t physical_row = result.oracle_rows[oracle_row];
        for (std::int32_t column = 0; column < k; ++column) {
            const std::size_t group = static_cast<std::size_t>(physical_row) * groups_per_row +
                                      static_cast<std::size_t>(column / 32);
            const auto code = static_cast<std::int8_t>(
                result.payload[static_cast<std::size_t>(physical_row) * k + column]);
            const float scale = fp16_to_float(
                load_u16_le(result.payload, scale_offset + group * sizeof(std::uint16_t)));
            result.oracle_weight[oracle_row * static_cast<std::size_t>(k) +
                                 static_cast<std::size_t>(column)] =
                static_cast<float>(code) * scale;
        }
    }
    return result;
}

std::vector<std::uint16_t> make_activation(std::int32_t k, std::int32_t t, std::uint32_t seed) {
    std::vector<std::uint16_t> result(checked_elements(k, t, "activation"));
    std::vector<std::uint16_t> patterns(static_cast<std::size_t>(256) * k);
    for (int offset = 0; offset < 256; ++offset) {
        for (std::int32_t column = 0; column < k; ++column) {
            const int raw = 32 + ((column * 17 + offset) & 0x5f);
            patterns[static_cast<std::size_t>(offset) * k + column] =
                test::f32_to_bf16(static_cast<float>(raw) * (1.0F / 256.0F));
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

std::vector<std::uint16_t> make_residual(std::int32_t n, std::int32_t t, std::uint32_t seed) {
    std::vector<std::uint16_t> result(checked_elements(n, t, "residual"));
    for (std::int32_t token = 0; token < t; ++token) {
        for (std::int32_t row = 0; row < n; ++row) {
            const int raw = static_cast<int>((static_cast<std::uint64_t>(row) * 23U +
                                              static_cast<std::uint64_t>(token) * 41U + seed * 7U) &
                                             0xffU);
            result[static_cast<std::size_t>(token) * n + row] =
                test::f32_to_bf16(static_cast<float>(raw - 128) * (1.0F / 512.0F));
        }
    }
    return result;
}

std::vector<double> linear_add_oracle(const PackedWeight& weight,
                                      const std::vector<std::uint16_t>& activation,
                                      const std::vector<std::uint16_t>& residual,
                                      std::span<const std::int32_t> columns) {
    const std::int32_t oracle_n = static_cast<std::int32_t>(weight.oracle_rows.size());
    std::vector<double> result(
        checked_elements(oracle_n, static_cast<std::int32_t>(columns.size()), "oracle output"));
    for (std::size_t selected_column = 0; selected_column < columns.size(); ++selected_column) {
        const std::int32_t token = columns[selected_column];
        for (std::int32_t oracle_row = 0; oracle_row < oracle_n; ++oracle_row) {
            double sum = 0.0;
            const float* weight_row =
                weight.oracle_weight.data() + static_cast<std::size_t>(oracle_row) * weight.k;
            const std::uint16_t* activation_column =
                activation.data() + static_cast<std::size_t>(token) * weight.k;
            for (std::int32_t column = 0; column < weight.k; ++column) {
                sum += static_cast<double>(weight_row[column]) *
                       static_cast<double>(bf16_to_float(activation_column[column]));
            }
            const std::int32_t physical_row =
                weight.oracle_rows[static_cast<std::size_t>(oracle_row)];
            // This is the complete fused formula. There is deliberately no private projection
            // materialization or BF16 rounding between GEMM and residual addition.
            result[selected_column * static_cast<std::size_t>(oracle_n) + oracle_row] =
                sum + static_cast<double>(bf16_to_float(
                          residual[static_cast<std::size_t>(token) * weight.n + physical_row]));
        }
    }
    return result;
}

struct OutputRead {
    int failures = 0;
    std::vector<double> selected;
};

OutputRead read_output(const void* device, std::int32_t n, std::int32_t t,
                       std::span<const std::int32_t> rows, std::span<const std::int32_t> columns,
                       std::span<const std::uint16_t> initial, std::string_view label) {
    const std::size_t total_words = checked_elements(n, t, "output");
    std::vector<std::size_t> wanted;
    wanted.reserve(rows.size() * columns.size());
    for (const std::int32_t column : columns) {
        for (const std::int32_t row : rows) {
            wanted.push_back(static_cast<std::size_t>(column) * n + row);
        }
    }

    OutputRead result;
    result.selected.resize(wanted.size());
    std::vector<std::uint16_t> chunk(std::min(kOutputScanWords, total_words));
    std::size_t wanted_index    = 0;
    std::size_t unchanged_count = 0;
    std::size_t nonfinite_count = 0;
    for (std::size_t begin = 0; begin < total_words; begin += chunk.size()) {
        const std::size_t count = std::min(chunk.size(), total_words - begin);
        test::cuda_check(
            cudaMemcpy(chunk.data(),
                       static_cast<const std::uint8_t*>(device) + begin * sizeof(std::uint16_t),
                       count * sizeof(std::uint16_t), cudaMemcpyDeviceToHost),
            "copy linear_add output");
        for (std::size_t index = 0; index < count; ++index) {
            const std::uint16_t bits = chunk[index];
            const std::int32_t row =
                static_cast<std::int32_t>((begin + index) % static_cast<std::size_t>(n));
            if (bits == initial[begin + index] &&
                !std::binary_search(rows.begin(), rows.end(), row)) {
                ++unchanged_count;
            }
            if ((bits & 0x7f80U) == 0x7f80U) { ++nonfinite_count; }
        }
        while (wanted_index < wanted.size() && wanted[wanted_index] < begin + count) {
            result.selected[wanted_index] =
                static_cast<double>(bf16_to_float(chunk[wanted[wanted_index] - begin]));
            ++wanted_index;
        }
    }
    if (unchanged_count != 0) {
        std::cerr << label << ": " << unchanged_count
                  << " non-oracle residual elements were not observably updated\n";
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
    if (actual.empty() || actual.size() != reference.size()) {
        std::cerr << label << ": invalid comparison sizes\n";
        return 1;
    }

    const ReductionStats stats = compute_reduction_stats(actual.data(), reference.data(),
                                                         static_cast<std::int64_t>(actual.size()));
    const double gross_limit   = gross_error_limit(stats, kLinearAddA16Tolerance);
    if (reduction_passes(stats, static_cast<std::int64_t>(actual.size()), kLinearAddA16Tolerance)) {
        return 0;
    }

    std::cerr << label << ": numerical mismatch rel_l2=" << stats.relative_l2
              << " limit=" << kLinearAddA16Tolerance.relative_l2
              << " max_abs=" << stats.maximum_absolute_error << " gross_limit=" << gross_limit
              << " index=" << stats.maximum_error_index << " actual=" << stats.actual_at_maximum
              << " reference=" << stats.reference_at_maximum << '\n';
    return 1;
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

int run_shape(std::string_view label, WeightFormat format, const ShapeCase& shape) {
    const std::vector<std::int32_t> tokens = conformance_tokens(shape);
    if (tokens.empty()) { throw std::invalid_argument("linear_add test: no token cases"); }
    const std::int32_t maximum_t = tokens.back();

    const std::vector<std::int32_t> oracle_rows = sampled_indices(shape.n);
    PackedWeight host_weight                    = format == WeightFormat::Q5G64F16S
                                                      ? make_q5_weight(shape.n, shape.k, shape.seed, oracle_rows)
                                                      : make_w8_weight(shape.n, shape.k, shape.seed, oracle_rows);
    const std::vector<std::uint16_t> activation =
        make_activation(shape.k, maximum_t, shape.seed + 1U);
    const std::vector<std::uint16_t> residual = make_residual(shape.n, maximum_t, shape.seed + 2U);
    const std::vector<std::int32_t> all_columns = all_indices(maximum_t);
    const std::vector<double> full_reference =
        linear_add_oracle(host_weight, activation, residual, all_columns);

    test::GuardedDeviceBuffer device_activation(activation.size() * sizeof(std::uint16_t));
    device_activation.copy_from_host(activation.data(), device_activation.bytes());
    test::GuardedDeviceBuffer device_weight(host_weight.payload.size());
    device_weight.copy_from_host(host_weight.payload.data(), host_weight.payload.size());
    const Weight weight = host_weight.device_view(device_weight.data());

    const std::size_t workspace_bytes =
        ops::linear_add_workspace_bytes(shape.n, shape.k, maximum_t);
    WorkspaceArena workspace(std::max<std::size_t>(workspace_bytes, 256));

    int failures = 0;
    for (const std::int32_t t : tokens) {
        const std::size_t output_words = checked_elements(shape.n, t, "output");
        test::GuardedDeviceBuffer output(output_words * sizeof(std::uint16_t));
        output.copy_from_host(residual.data(), output.bytes());

        Tensor input(device_activation.data(), DType::BF16, {shape.k, t});
        Tensor residual_out(output.data(), DType::BF16, {shape.n, t});
        workspace.reset();

        const std::string case_label = std::string(label) + " [" + std::to_string(shape.n) + "," +
                                       std::to_string(shape.k) + "] T=" + std::to_string(t);
        try {
            ops::linear_add(input, weight, residual_out, workspace, nullptr);
            test::cuda_check(cudaDeviceSynchronize(), "synchronize linear_add");
        } catch (const std::exception& error) {
            std::cerr << case_label << ": unexpected exception: " << error.what() << '\n';
            ++failures;
            continue;
        }

        failures += output.verify_guards(case_label.c_str());
        const std::vector<std::int32_t> columns = all_indices(t);
        const OutputRead actual =
            read_output(output.data(), shape.n, t, oracle_rows, columns,
                        std::span<const std::uint16_t>(residual.data(), output_words), case_label);
        failures += actual.failures;
        failures += compare_output(
            case_label, actual.selected,
            std::span<const double>(
                full_reference.data(),
                checked_elements(static_cast<std::int32_t>(oracle_rows.size()), t, "reference")));
    }

    failures += device_activation.verify_guards("linear_add activation");
    failures += device_weight.verify_guards("linear_add weight");
    failures += verify_preserved(
        device_activation,
        std::span<const std::uint8_t>(reinterpret_cast<const std::uint8_t*>(activation.data()),
                                      activation.size() * sizeof(std::uint16_t)),
        "linear_add activation");
    failures += verify_preserved(device_weight, host_weight.payload, "linear_add weight");
    return failures;
}

} // namespace ninfer::test::linear_add
