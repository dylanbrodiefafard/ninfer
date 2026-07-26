#include "ops/linear/linear_test_common.h"

#include "core/arena.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace ninfer::test::linear {
namespace {

constexpr std::size_t kOutputGuardBytes  = 256;
constexpr std::uint8_t kOutputGuardByte  = 0xa5;
constexpr std::uint8_t kOutputPoisonByte = 0xff;
constexpr std::size_t kOutputScanWords   = 1U << 20;
constexpr int kOracleTBlock              = 8;

struct FormatSpec {
    QType qtype;
    std::int32_t bits;
    std::int32_t group_size;
    std::int32_t base_bytes_per_group;
    std::int32_t high_bytes_per_group;
};

constexpr FormatSpec kQ4Spec{QType::Q4G64_F16S, 4, 64, 32, 0};
constexpr FormatSpec kQ5Spec{QType::Q5G64_F16S, 5, 64, 32, 8};
constexpr FormatSpec kQ6Spec{QType::Q6G64_F16S, 6, 64, 32, 16};
constexpr FormatSpec kW8Spec{QType::W8G32_F16S, 8, 32, 32, 0};

// The criterion belongs to the activation compute path, not to a private kernel, schedule, or
// launcher selected inside that path.
constexpr ReductionCriterion tolerance_for(ActivationCompute activation_compute) {
    switch (activation_compute) {
    case ActivationCompute::A16:
        return {3.0e-3, 4.0e-3, 3.5e-3};
    }
    throw std::invalid_argument("linear test: unknown activation compute path");
}

constexpr const char* activation_compute_name(ActivationCompute activation_compute) {
    switch (activation_compute) {
    case ActivationCompute::A16:
        return "A16";
    }
    throw std::invalid_argument("linear test: unknown activation compute path");
}

constexpr const char* comparison_name(Comparison comparison) {
    switch (comparison) {
    case Comparison::Full:
        return "full";
    case Comparison::Sampled:
        return "sampled";
    }
    throw std::invalid_argument("linear test: unknown comparison mode");
}

bool report_statistics() {
    static const bool enabled = [] {
        const char* value = std::getenv("NINFER_LINEAR_REPORT_STATS");
        return value != nullptr && value[0] != '\0' && std::strcmp(value, "0") != 0;
    }();
    return enabled;
}

std::size_t checked_elements(std::int32_t first, std::int32_t second, const char* label) {
    if (first <= 0 || second <= 0) {
        throw std::invalid_argument(std::string("linear test: invalid ") + label + " extent");
    }
    const auto a = static_cast<std::size_t>(first);
    const auto b = static_cast<std::size_t>(second);
    if (a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error(std::string("linear test: ") + label + " size overflow");
    }
    return a * b;
}

std::size_t align_up(std::size_t value, std::size_t alignment) {
    if (value > std::numeric_limits<std::size_t>::max() - (alignment - 1)) {
        throw std::overflow_error("linear test: alignment overflow");
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

std::uint32_t float_to_bits(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

float bf16_to_float(std::uint16_t bits) {
    return bits_to_float(static_cast<std::uint32_t>(bits) << 16);
}

std::uint16_t float_to_bf16(float value) {
    std::uint32_t bits = float_to_bits(value);
    if ((bits & 0x7fffffffU) > 0x7f800000U) {
        return static_cast<std::uint16_t>((bits >> 16) | 0x0040U);
    }
    bits += 0x7fffU + ((bits >> 16) & 1U);
    return static_cast<std::uint16_t>(bits >> 16);
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

int sign_extend(std::uint32_t value, int bits) {
    const std::uint32_t sign  = 1U << (bits - 1);
    const std::uint32_t range = 1U << bits;
    return (value & sign) != 0U ? static_cast<int>(value) - static_cast<int>(range)
                                : static_cast<int>(value);
}

void store_code(std::vector<std::uint8_t>& payload, const FormatSpec& spec,
                std::uint64_t high_plane_offset, std::size_t group_index, int lane, int code) {
    if (spec.bits == 8) {
        payload[group_index * static_cast<std::size_t>(spec.base_bytes_per_group) +
                static_cast<std::size_t>(lane)] =
            static_cast<std::uint8_t>(static_cast<std::int8_t>(code));
        return;
    }

    const std::uint32_t word = static_cast<std::uint32_t>(code) & ((1U << spec.bits) - 1U);
    const std::size_t base_offset =
        group_index * static_cast<std::size_t>(spec.base_bytes_per_group) +
        static_cast<std::size_t>(lane / 2);
    const std::uint8_t nibble = static_cast<std::uint8_t>(word & 0x0fU);
    if ((lane & 1) == 0) {
        payload[base_offset] = static_cast<std::uint8_t>((payload[base_offset] & 0xf0U) | nibble);
    } else {
        payload[base_offset] =
            static_cast<std::uint8_t>((payload[base_offset] & 0x0fU) | (nibble << 4));
    }

    const int high_bits = spec.bits - 4;
    for (int bit = 0; bit < high_bits; ++bit) {
        const int stream_bit = lane * high_bits + bit;
        const std::size_t byte_offset =
            static_cast<std::size_t>(high_plane_offset) +
            group_index * static_cast<std::size_t>(spec.high_bytes_per_group) +
            static_cast<std::size_t>(stream_bit / 8);
        payload[byte_offset] |=
            static_cast<std::uint8_t>(((word >> (4 + bit)) & 1U) << (stream_bit & 7));
    }
}

int load_code(const SimulatedLinearWeight& weight, const FormatSpec& spec, std::int32_t row,
              std::int32_t column) {
    const std::int32_t groups_per_row = weight.padded_k / spec.group_size;
    const std::int32_t group          = column / spec.group_size;
    const int lane                    = column % spec.group_size;
    const std::size_t group_index =
        static_cast<std::size_t>(row) * static_cast<std::size_t>(groups_per_row) +
        static_cast<std::size_t>(group);

    if (spec.bits == 8) {
        const std::uint8_t word =
            weight
                .packed_payload[group_index * static_cast<std::size_t>(spec.base_bytes_per_group) +
                                static_cast<std::size_t>(lane)];
        return static_cast<int>(static_cast<std::int8_t>(word));
    }

    const std::uint8_t base =
        weight.packed_payload[group_index * static_cast<std::size_t>(spec.base_bytes_per_group) +
                              static_cast<std::size_t>(lane / 2)];
    std::uint32_t word  = (lane & 1) == 0 ? (base & 0x0fU) : (base >> 4);
    const int high_bits = spec.bits - 4;
    for (int bit = 0; bit < high_bits; ++bit) {
        const int stream_bit = lane * high_bits + bit;
        const std::size_t byte_offset =
            static_cast<std::size_t>(weight.high_plane_offset) +
            group_index * static_cast<std::size_t>(spec.high_bytes_per_group) +
            static_cast<std::size_t>(stream_bit / 8);
        word |= static_cast<std::uint32_t>(
                    (weight.packed_payload[byte_offset] >> (stream_bit & 7)) & 1U)
                << (4 + bit);
    }
    return sign_extend(word, spec.bits);
}

std::vector<std::int32_t> checked_oracle_rows(std::int32_t n, std::span<const std::int32_t> rows) {
    std::vector<std::int32_t> result(rows.begin(), rows.end());
    if (result.empty()) { throw std::invalid_argument("linear test: oracle row set is empty"); }
    for (const std::int32_t row : result) {
        if (row < 0 || row >= n) {
            throw std::invalid_argument("linear test: oracle row is outside the weight");
        }
    }
    std::sort(result.begin(), result.end());
    if (std::adjacent_find(result.begin(), result.end()) != result.end()) {
        throw std::invalid_argument("linear test: duplicate oracle row");
    }
    return result;
}

SimulatedLinearWeight make_weight(const FormatSpec& spec, std::int32_t n, std::int32_t k,
                                  std::uint32_t seed, std::span<const std::int32_t> oracle_rows) {
    if (n <= 0 || k <= 0) {
        throw std::invalid_argument("linear test: weight shape must be positive");
    }

    SimulatedLinearWeight result;
    result.qtype       = spec.qtype;
    result.n           = n;
    result.k           = k;
    result.padded_k    = static_cast<std::int32_t>(align_up(static_cast<std::size_t>(k), 128));
    result.group_size  = spec.group_size;
    result.oracle_rows = checked_oracle_rows(n, oracle_rows);

    const std::int32_t groups_per_row = result.padded_k / spec.group_size;
    const std::size_t group_count     = checked_elements(n, groups_per_row, "weight groups");
    const std::size_t base_bytes =
        group_count * static_cast<std::size_t>(spec.base_bytes_per_group);
    result.high_plane_offset  = align_up(base_bytes, 256);
    result.high_plane_bytes   = group_count * static_cast<std::size_t>(spec.high_bytes_per_group);
    result.scale_plane_offset = result.high_plane_offset + align_up(result.high_plane_bytes, 256);
    const std::size_t scale_bytes = group_count * sizeof(std::uint16_t);
    result.packed_payload.assign(static_cast<std::size_t>(result.scale_plane_offset) + scale_bytes,
                                 0);

    std::array<std::array<std::uint8_t, 32>, 256> base_patterns{};
    std::array<std::array<std::uint8_t, 16>, 256> high_patterns{};
    for (std::size_t pattern = 0; pattern < base_patterns.size(); ++pattern) {
        std::uint64_t state = mix64(static_cast<std::uint64_t>(seed) << 32 | pattern);
        for (std::size_t byte = 0; byte < base_patterns[pattern].size(); ++byte) {
            state              = mix64(state + byte);
            std::uint8_t value = static_cast<std::uint8_t>(state >> 56);
            if (spec.bits == 8 && value == 0x80U) { value = 0x81U; }
            base_patterns[pattern][byte] = value;
        }
        for (std::size_t byte = 0; byte < high_patterns[pattern].size(); ++byte) {
            state                        = mix64(state + 0x100U + byte);
            high_patterns[pattern][byte] = static_cast<std::uint8_t>(state >> 56);
        }
    }

    constexpr std::array<std::uint16_t, 4> kScaleBits{
        0x2000U, // 0.0078125
        0x2400U, // 0.015625
        0x2800U, // 0.03125
        0x2c00U, // 0.0625
    };
    for (std::size_t group_index = 0; group_index < group_count; ++group_index) {
        const std::int32_t group_in_row =
            static_cast<std::int32_t>(group_index % static_cast<std::size_t>(groups_per_row));
        const std::int32_t first_column = group_in_row * spec.group_size;
        const std::int32_t valid_lanes  = std::clamp(k - first_column, 0, spec.group_size);
        const std::uint64_t mixed = mix64(group_index ^ (static_cast<std::uint64_t>(seed) << 17));
        const std::size_t pattern = static_cast<std::size_t>(mixed & 0xffU);

        if (valid_lanes == spec.group_size) {
            std::memcpy(result.packed_payload.data() +
                            group_index * static_cast<std::size_t>(spec.base_bytes_per_group),
                        base_patterns[pattern].data(),
                        static_cast<std::size_t>(spec.base_bytes_per_group));
            if (spec.high_bytes_per_group != 0) {
                std::memcpy(result.packed_payload.data() +
                                static_cast<std::size_t>(result.high_plane_offset) +
                                group_index * static_cast<std::size_t>(spec.high_bytes_per_group),
                            high_patterns[pattern].data(),
                            static_cast<std::size_t>(spec.high_bytes_per_group));
            }
        } else if (valid_lanes != 0) {
            for (int lane = 0; lane < valid_lanes; ++lane) {
                const std::uint64_t lane_bits = mix64(mixed + static_cast<std::uint64_t>(lane));
                int code                      = 0;
                if (spec.bits == 8) {
                    code = static_cast<int>(lane_bits % 255U) - 127;
                } else {
                    code = sign_extend(static_cast<std::uint32_t>(lane_bits) &
                                           ((1U << spec.bits) - 1U),
                                       spec.bits);
                }
                store_code(result.packed_payload, spec, result.high_plane_offset, group_index, lane,
                           code);
            }
        }

        const std::uint16_t scale =
            valid_lanes == 0 ? 0U : kScaleBits[static_cast<std::size_t>((mixed >> 8) & 3U)];
        store_u16_le(result.packed_payload,
                     static_cast<std::size_t>(result.scale_plane_offset) +
                         group_index * sizeof(std::uint16_t),
                     scale);
    }

    result.oracle_weight.resize(
        checked_elements(static_cast<std::int32_t>(result.oracle_rows.size()), k, "oracle weight"));
    for (std::size_t oracle_row = 0; oracle_row < result.oracle_rows.size(); ++oracle_row) {
        const std::int32_t physical_row = result.oracle_rows[oracle_row];
        for (std::int32_t column = 0; column < k; ++column) {
            const std::int32_t group = column / spec.group_size;
            const std::size_t group_index =
                static_cast<std::size_t>(physical_row) * static_cast<std::size_t>(groups_per_row) +
                static_cast<std::size_t>(group);
            const std::uint16_t scale_bits = load_u16_le(
                result.packed_payload, static_cast<std::size_t>(result.scale_plane_offset) +
                                           group_index * sizeof(std::uint16_t));
            const float scale = fp16_to_float(scale_bits);
            result.oracle_weight[oracle_row * static_cast<std::size_t>(k) +
                                 static_cast<std::size_t>(column)] =
                static_cast<float>(load_code(result, spec, physical_row, column)) * scale;
        }
    }
    return result;
}

class GuardedOutput {
public:
    explicit GuardedOutput(std::size_t words)
        : storage_(words * sizeof(std::uint16_t), kOutputGuardBytes, kOutputGuardByte) {
        poison();
    }

    void* data() { return storage_.data(); }

    const void* data() const { return storage_.data(); }

    void poison() { storage_.fill(kOutputPoisonByte); }

    int verify_guards(std::string_view label) const { return storage_.verify_guards(label); }

private:
    GuardedDeviceBuffer storage_;
};

std::vector<std::int32_t> all_indices(std::int32_t extent) {
    std::vector<std::int32_t> indices(static_cast<std::size_t>(extent));
    std::iota(indices.begin(), indices.end(), 0);
    return indices;
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

std::vector<std::uint16_t> make_activation(std::int32_t k, std::int32_t t, std::uint32_t seed) {
    const std::size_t elements = checked_elements(k, t, "activation");
    std::vector<std::uint16_t> result(elements);
    std::vector<std::uint16_t> patterns(static_cast<std::size_t>(256) *
                                        static_cast<std::size_t>(k));
    for (int offset = 0; offset < 256; ++offset) {
        for (std::int32_t column = 0; column < k; ++column) {
            const int raw     = (column * 17 + offset) & 0xff;
            const float value = static_cast<float>(raw - 128) * (1.0F / 256.0F);
            patterns[static_cast<std::size_t>(offset) * k + column] = float_to_bf16(value);
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

std::vector<float> materialize_activation(const std::vector<std::uint16_t>& bits, std::int32_t k,
                                          std::span<const std::int32_t> columns) {
    std::vector<float> result(
        checked_elements(k, static_cast<std::int32_t>(columns.size()), "oracle activation"));
    for (std::size_t oracle_column = 0; oracle_column < columns.size(); ++oracle_column) {
        const std::uint16_t* source =
            bits.data() + static_cast<std::size_t>(columns[oracle_column]) * k;
        float* destination = result.data() + oracle_column * static_cast<std::size_t>(k);
        for (std::int32_t column = 0; column < k; ++column) {
            destination[column] = bf16_to_float(source[column]);
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
                       std::string_view label) {
    const std::size_t total_words = checked_elements(n, t, "output");
    std::vector<std::size_t> wanted;
    wanted.reserve(rows.size() * columns.size());
    for (const std::int32_t column : columns) {
        for (const std::int32_t row : rows) {
            wanted.push_back(static_cast<std::size_t>(column) * n + static_cast<std::size_t>(row));
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
        cuda_check(
            cudaMemcpy(chunk.data(),
                       static_cast<const std::uint8_t*>(device) + begin * sizeof(std::uint16_t),
                       count * sizeof(std::uint16_t), cudaMemcpyDeviceToHost),
            "copy linear output");
        for (std::size_t index = 0; index < count; ++index) {
            const std::uint16_t bits = chunk[index];
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
                   std::span<const double> reference, ActivationCompute activation_compute,
                   Comparison comparison) {
    if (actual.empty() || actual.size() != reference.size()) {
        std::cerr << label << ": invalid comparison sizes\n";
        return 1;
    }

    const ReductionCriterion tolerance = tolerance_for(activation_compute);
    const ReductionStats stats         = compute_reduction_stats(actual.data(), reference.data(),
                                                                 static_cast<std::int64_t>(actual.size()));
    if (stats.first_non_finite >= 0) {
        std::cerr << label << ": comparison contains a non-finite value at index "
                  << stats.first_non_finite << '\n';
        return 1;
    }

    const double gross_limit = gross_error_limit(stats, tolerance);
    if (report_statistics()) {
        const double peak_relative =
            stats.maximum_absolute_error / std::max(stats.maximum_absolute_reference, 1.0e-30);
        const double relative_l2_ratio = stats.relative_l2 / tolerance.relative_l2;
        const double gross_ratio       = stats.maximum_absolute_error / gross_limit;
        std::printf(
            "LINEAR_STATS activation_compute=%s comparison=%s count=%zu rel_l2=%.17g rmse=%.17g "
            "reference_rms=%.17g max_abs=%.17g max_reference=%.17g peak_relative=%.17g "
            "relative_l2_limit_ratio=%.17g gross_limit_ratio=%.17g case=%.*s\n",
            activation_compute_name(activation_compute), comparison_name(comparison), actual.size(),
            stats.relative_l2, stats.root_mean_squared_error, stats.reference_root_mean_square,
            stats.maximum_absolute_error, stats.maximum_absolute_reference, peak_relative,
            relative_l2_ratio, gross_ratio, static_cast<int>(label.size()), label.data());
    }
    if (reduction_passes(stats, static_cast<std::int64_t>(actual.size()), tolerance)) { return 0; }

    std::cerr << label << ": numerical mismatch" << " rel_l2=" << stats.relative_l2
              << " limit=" << tolerance.relative_l2 << " max_abs=" << stats.maximum_absolute_error
              << " gross_limit=" << gross_limit << " index=" << stats.maximum_error_index
              << " actual=" << stats.actual_at_maximum
              << " reference=" << stats.reference_at_maximum << '\n';
    return 1;
}

} // namespace

Weight SimulatedLinearWeight::device_weight(void* device_payload) const {
    Weight weight{};
    weight.payload          = device_payload;
    weight.payload_bytes    = packed_payload.size();
    weight.high_plane_bytes = high_plane_bytes;
    weight.qtype            = qtype;
    weight.group_size       = static_cast<std::uint32_t>(group_size);
    weight.qdata            = device_payload;
    weight.qhigh            = high_plane_bytes == 0
                                  ? nullptr
                                  : static_cast<std::uint8_t*>(device_payload) + high_plane_offset;
    weight.scales           = static_cast<std::uint8_t*>(device_payload) + scale_plane_offset;
    weight.n                = n;
    weight.k                = k;
    weight.group            = group_size;
    weight.layout           = QuantLayout::RowSplit;
    weight.scale_dtype      = DType::FP16;
    weight.ndim             = 2;
    weight.shape[0]         = n;
    weight.shape[1]         = k;
    weight.padded_shape[0]  = n;
    weight.padded_shape[1]  = padded_k;
    return weight;
}

SimulatedLinearWeight make_q4g64_f16s_weight(std::int32_t n, std::int32_t k, std::uint32_t seed,
                                             std::span<const std::int32_t> oracle_rows) {
    return make_weight(kQ4Spec, n, k, seed, oracle_rows);
}

SimulatedLinearWeight make_q5g64_f16s_weight(std::int32_t n, std::int32_t k, std::uint32_t seed,
                                             std::span<const std::int32_t> oracle_rows) {
    return make_weight(kQ5Spec, n, k, seed, oracle_rows);
}

SimulatedLinearWeight make_q6g64_f16s_weight(std::int32_t n, std::int32_t k, std::uint32_t seed,
                                             std::span<const std::int32_t> oracle_rows) {
    return make_weight(kQ6Spec, n, k, seed, oracle_rows);
}

SimulatedLinearWeight make_w8g32_f16s_weight(std::int32_t n, std::int32_t k, std::uint32_t seed,
                                             std::span<const std::int32_t> oracle_rows) {
    return make_weight(kW8Spec, n, k, seed, oracle_rows);
}

void cpu_linear_gemm_fp64(const float* weight, const float* activation, double* output,
                          std::int32_t n, std::int32_t k, std::int32_t t) {
    if (weight == nullptr || activation == nullptr || output == nullptr || n <= 0 || k <= 0 ||
        t <= 0) {
        throw std::invalid_argument("linear test: invalid FP64 GEMM argument");
    }

    const unsigned hardware_threads = std::max(1U, std::thread::hardware_concurrency());
    const std::int32_t thread_count = std::min(n, static_cast<std::int32_t>(hardware_threads));
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(thread_count));
    for (std::int32_t thread = 0; thread < thread_count; ++thread) {
        const std::int32_t row_begin =
            static_cast<std::int32_t>((static_cast<std::int64_t>(n) * thread) / thread_count);
        const std::int32_t row_end =
            static_cast<std::int32_t>((static_cast<std::int64_t>(n) * (thread + 1)) / thread_count);
        workers.emplace_back([=] {
            for (std::int32_t row = row_begin; row < row_end; ++row) {
                const float* weight_row = weight + static_cast<std::size_t>(row) * k;
                for (std::int32_t token_begin = 0; token_begin < t; token_begin += kOracleTBlock) {
                    const std::int32_t active = std::min(kOracleTBlock, t - token_begin);
                    std::array<double, kOracleTBlock> accumulators{};
                    for (std::int32_t column = 0; column < k; ++column) {
                        const double weight_value = static_cast<double>(weight_row[column]);
                        for (std::int32_t token = 0; token < active; ++token) {
                            accumulators[static_cast<std::size_t>(token)] +=
                                weight_value *
                                static_cast<double>(
                                    activation[static_cast<std::size_t>(token_begin + token) * k +
                                               column]);
                        }
                    }
                    for (std::int32_t token = 0; token < active; ++token) {
                        output[static_cast<std::size_t>(token_begin + token) * n + row] =
                            accumulators[static_cast<std::size_t>(token)];
                    }
                }
            }
        });
    }
    for (std::thread& worker : workers) { worker.join(); }
}

bool cuda_available() { return !test::cuda_unavailable(); }

int run_shape(std::string_view label, ActivationCompute activation_compute,
              WeightGenerator generator, const ShapeCase& shape) {
    if (shape.invocations.empty()) {
        throw std::invalid_argument("linear test: shape has no invocations");
    }
    const auto maximum = std::max_element(
        shape.invocations.begin(), shape.invocations.end(),
        [](const Invocation& left, const Invocation& right) { return left.t < right.t; });
    if (maximum->t <= 0) {
        throw std::invalid_argument("linear test: token extent must be positive");
    }

    const std::vector<std::int32_t> oracle_rows =
        shape.comparison == Comparison::Full ? all_indices(shape.n) : sampled_indices(shape.n);
    SimulatedLinearWeight host_weight = generator(shape.n, shape.k, shape.seed, oracle_rows);
    const std::vector<std::uint16_t> activation_bits =
        make_activation(shape.k, maximum->t, shape.seed + 1U);

    DeviceBuffer device_activation(activation_bits.size() * sizeof(std::uint16_t));
    device_activation.copy_from_host(activation_bits.data(), device_activation.bytes);
    DeviceBuffer device_weight(host_weight.packed_payload.size());
    device_weight.copy_from_host(host_weight.packed_payload.data(), device_weight.bytes);
    const Weight weight = host_weight.device_weight(device_weight.p);
    WorkspaceArena workspace(64U << 20);

    std::vector<double> full_reference;
    if (shape.comparison == Comparison::Full) {
        const std::vector<std::int32_t> columns = all_indices(maximum->t);
        const std::vector<float> activation =
            materialize_activation(activation_bits, shape.k, columns);
        full_reference.resize(checked_elements(shape.n, maximum->t, "full reference"));
        cpu_linear_gemm_fp64(host_weight.oracle_weight.data(), activation.data(),
                             full_reference.data(), shape.n, shape.k, maximum->t);
    }

    int failures = 0;
    for (const Invocation& invocation : shape.invocations) {
        const std::string case_label = std::string(label) + " [" + std::to_string(shape.n) + "," +
                                       std::to_string(shape.k) +
                                       "] T=" + std::to_string(invocation.t);
        GuardedOutput output(checked_elements(shape.n, invocation.t, "guarded output"));
        Tensor input(device_activation.p, DType::BF16, {shape.k, invocation.t});
        Tensor destination(output.data(), DType::BF16, {shape.n, invocation.t});
        workspace.reset();
        try {
            if (invocation.call_form == CallForm::A16Convenience) {
                ops::linear(input, weight, destination, workspace, nullptr);
            } else {
                ops::linear(input, weight, destination, invocation.policy, workspace, nullptr);
            }
            cuda_check(cudaDeviceSynchronize(), "synchronize linear");
        } catch (const std::exception& error) {
            std::cerr << case_label << ": unexpected exception: " << error.what() << '\n';
            ++failures;
            continue;
        }

        failures += output.verify_guards(case_label);
        const std::vector<std::int32_t> columns = shape.comparison == Comparison::Full
                                                      ? all_indices(invocation.t)
                                                      : sampled_indices(invocation.t);
        OutputRead actual =
            read_output(output.data(), shape.n, invocation.t, oracle_rows, columns, case_label);
        failures += actual.failures;

        if (shape.comparison == Comparison::Full) {
            failures +=
                compare_output(case_label, actual.selected,
                               std::span<const double>(
                                   full_reference.data(),
                                   checked_elements(shape.n, invocation.t, "reference prefix")),
                               activation_compute, shape.comparison);
        } else {
            const std::vector<float> activation =
                materialize_activation(activation_bits, shape.k, columns);
            std::vector<double> reference(
                checked_elements(static_cast<std::int32_t>(oracle_rows.size()),
                                 static_cast<std::int32_t>(columns.size()), "sampled reference"));
            cpu_linear_gemm_fp64(host_weight.oracle_weight.data(), activation.data(),
                                 reference.data(), static_cast<std::int32_t>(oracle_rows.size()),
                                 shape.k, static_cast<std::int32_t>(columns.size()));
            failures += compare_output(case_label, actual.selected, reference, activation_compute,
                                       shape.comparison);
        }
    }

    if (shape.verify_input_preservation) {
        std::vector<std::uint16_t> activation_after(activation_bits.size());
        device_activation.copy_to_host(activation_after.data(), device_activation.bytes);
        if (activation_after != activation_bits) {
            std::cerr << label << ": linear modified its activation input\n";
            ++failures;
        }
        std::vector<std::uint8_t> weight_after(host_weight.packed_payload.size());
        device_weight.copy_to_host(weight_after.data(), device_weight.bytes);
        if (weight_after != host_weight.packed_payload) {
            std::cerr << label << ": linear modified its persistent weight\n";
            ++failures;
        }
    }
    return failures;
}

} // namespace ninfer::test::linear
