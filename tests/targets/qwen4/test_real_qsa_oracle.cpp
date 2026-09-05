#include "targets/qwen4/real_oracle_common.h"

#include "ninfer/ops/gated_residual.h"
#include "ninfer/ops/qsa.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace verifier = ninfer::targets::qwen4::verifier;
namespace ops = ninfer::ops;
using ninfer::DType;
using ninfer::DeviceContext;
using ninfer::DeviceBuffer;
using ninfer::QType;
using ninfer::QuantLayout;
using ninfer::Tensor;
using ninfer::Weight;
using ninfer::WorkspaceArena;
using ninfer::test::GuardedDeviceBuffer;
using ninfer::test::PointwiseCriterion;
using ninfer::test::bf16_to_f32;
using ninfer::test::error_stats_enabled;
using ninfer::test::f32_to_bf16;
using ninfer::test::from_device;
using ninfer::test::from_device_bf16;
using ninfer::test::to_device;
using ninfer::test::verify_exact;
using ninfer::test::verify_pointwise;
using namespace ninfer::test::qwen4::real_oracle;

namespace {

constexpr std::int32_t kHidden = 2560;
constexpr std::int32_t kIndex = 128;
constexpr std::int32_t kIndexHeads = 4;
constexpr std::int32_t kHeadDim = 256;
constexpr std::int32_t kQueryHeads = 24;
constexpr std::int32_t kKvHeads = 2;
constexpr std::int32_t kQueryGateRows = kQueryHeads * kHeadDim * 2;
constexpr std::int32_t kOutputColumns = kQueryHeads * kHeadDim;
constexpr double kFloatUnitRoundoff = 0x1p-24;
constexpr double kMinimumBf16RoundBound = 0x1p-134;

// Pinned llama_tokenize output for one paragraph including its terminal LF. The 601-token
// fixture repeats this sequence six times and then omits the final LF from the seventh copy.
constexpr std::array<std::int32_t, 86> kFrozenParagraph = {
    48, 16451, 17120, 22188, 11988, 3817, 19039, 888, 264, 2716, 8097, 40701, 13, 561,
    1558, 15339, 1754, 3299, 303, 1906, 321, 54004, 1092, 3905, 1727, 13, 3931, 921,
    13224, 20480, 16338, 1528, 11, 6326, 13224, 62586, 6575, 2193, 11, 321, 32335,
    11312, 5000, 3955, 10885, 13, 1061, 14648, 13901, 5533, 13983, 19464, 12, 23,
    1414, 11, 14733, 59429, 11, 321, 3213, 10885, 364, 799, 2526, 10756, 14751,
    1931, 19221, 3136, 13, 11116, 7193, 369, 33625, 17066, 5721, 11, 524, 264,
    3591, 883, 3992, 4131, 13, 198,
};

// This is the fixed complete-composite criterion owned by test_qsa.cpp.
constexpr PointwiseCriterion kCompleteCriterion{/*absolute=*/0.025, /*relative=*/0.02};

struct Dot {
    double value = 0.0;
    double absolute_sum = 0.0;
};

struct Projection {
    std::vector<double> ideal;
    std::vector<double> represented;
    std::vector<double> production_to_ideal_bound;
};

struct NormProjection {
    std::vector<double> represented;
    std::vector<double> production_to_represented_bound;
};

double represented_bf16(double value) {
    return static_cast<double>(bf16_to_f32(f32_to_bf16(static_cast<float>(value))));
}

double sigmoid(double value) {
    if (value >= 0.0) { return 1.0 / (1.0 + std::exp(-value)); }
    const double exponential = std::exp(value);
    return exponential / (1.0 + exponential);
}

double accumulation_and_bf16_bound(const Dot& dot, double arithmetic_steps) {
    const double gamma = arithmetic_steps * kFloatUnitRoundoff /
                         (1.0 - arithmetic_steps * kFloatUnitRoundoff);
    const double accumulation = gamma * dot.absolute_sum;
    const double bf16_rounding =
        std::max((std::abs(dot.value) + accumulation) / 255.0,
                 kMinimumBf16RoundBound);
    return accumulation + bf16_rounding;
}

void require_q5(const Weight& weight, std::int32_t rows, std::int32_t columns,
                const char* name) {
    const std::size_t row_bytes =
        static_cast<std::size_t>(columns) / kQ5BlockValues * kQ5BlockBytes;
    if (columns % static_cast<std::int32_t>(kQ5BlockValues) != 0 ||
        weight.qtype != QType::GGML_Q5_K || weight.layout != QuantLayout::GgmlBlockRow ||
        weight.n != rows || weight.k != columns || weight.qdata == nullptr ||
        weight.payload_bytes != static_cast<std::size_t>(rows) * row_bytes) {
        throw std::logic_error(std::string("Qwen4 real QSA malformed ") + name);
    }
}

void require_bf16(const Weight& weight, std::int32_t rows, const char* name) {
    if (weight.qtype != QType::BF16_CTRL || weight.layout != QuantLayout::Contiguous ||
        weight.n != rows || weight.k != kHidden || weight.qdata == nullptr ||
        weight.payload_bytes != static_cast<std::size_t>(rows) * kHidden * sizeof(std::uint16_t)) {
        throw std::logic_error(std::string("Qwen4 real QSA malformed ") + name);
    }
}

Dot q5_dot(std::span<const std::uint8_t> matrix, std::int32_t row, std::int32_t rows,
           std::int32_t columns, std::span<const double> input) {
    const std::size_t row_bytes =
        static_cast<std::size_t>(columns) / kQ5BlockValues * kQ5BlockBytes;
    if (matrix.size() != static_cast<std::size_t>(rows) * row_bytes || row < 0 || row >= rows ||
        input.size() != static_cast<std::size_t>(columns)) {
        throw std::logic_error("Qwen4 real QSA Q5_K oracle received malformed storage");
    }
    const auto* encoded = matrix.data() + static_cast<std::size_t>(row) * row_bytes;
    Dot result;
    for (std::int32_t column = 0; column < columns; ++column) {
        const double product = ggml_q5_k_value(encoded, column) * input[column];
        result.value += product;
        result.absolute_sum += std::abs(product);
    }
    return result;
}

Projection q5_project(std::span<const std::uint8_t> matrix, std::int32_t rows,
                      std::int32_t columns, std::span<const double> input) {
    Projection result{std::vector<double>(rows), std::vector<double>(rows),
                      std::vector<double>(rows)};
    const double arithmetic_steps = 5.0 * columns + 8.0;
    for (std::int32_t row = 0; row < rows; ++row) {
        const Dot dot = q5_dot(matrix, row, rows, columns, input);
        result.ideal[row] = dot.value;
        result.represented[row] = represented_bf16(dot.value);
        result.production_to_ideal_bound[row] =
            accumulation_and_bf16_bound(dot, arithmetic_steps);
    }
    return result;
}

Projection bf16_project(std::span<const std::uint16_t> matrix, std::int32_t rows,
                        std::span<const double> input) {
    if (matrix.size() != static_cast<std::size_t>(rows) * kHidden ||
        input.size() != static_cast<std::size_t>(kHidden)) {
        throw std::logic_error("Qwen4 real QSA BF16 oracle received malformed storage");
    }
    Projection result{std::vector<double>(rows), std::vector<double>(rows),
                      std::vector<double>(rows)};
    const double arithmetic_steps = 2.0 * kHidden + 8.0;
    for (std::int32_t row = 0; row < rows; ++row) {
        Dot dot;
        for (std::int32_t column = 0; column < kHidden; ++column) {
            const double product =
                static_cast<double>(bf16_to_f32(matrix[static_cast<std::size_t>(row) * kHidden +
                                                       column])) *
                input[column];
            dot.value += product;
            dot.absolute_sum += std::abs(product);
        }
        result.ideal[row] = dot.value;
        result.represented[row] = represented_bf16(dot.value);
        result.production_to_ideal_bound[row] =
            accumulation_and_bf16_bound(dot, arithmetic_steps);
    }
    return result;
}

NormProjection core_norm_rope(const Projection& raw, std::int32_t heads,
                              std::int32_t raw_stride, std::span<const float> gamma,
                              const std::array<std::int32_t, 3>& position) {
    if (raw.represented.size() != static_cast<std::size_t>(heads) * raw_stride ||
        gamma.size() != kHeadDim) {
        throw std::logic_error("Qwen4 real QSA norm/RoPE oracle shape changed");
    }
    NormProjection result{std::vector<double>(static_cast<std::size_t>(heads) * kHeadDim),
                          std::vector<double>(static_cast<std::size_t>(heads) * kHeadDim)};
    for (std::int32_t head = 0; head < heads; ++head) {
        const std::size_t raw_base = static_cast<std::size_t>(head) * raw_stride;
        const std::size_t out_base = static_cast<std::size_t>(head) * kHeadDim;
        double represented_sum_squares = 0.0;
        double raw_error_sum_squares = 0.0;
        std::array<double, kHeadDim> raw_error{};
        for (std::int32_t d = 0; d < kHeadDim; ++d) {
            const std::size_t source = raw_base + d;
            represented_sum_squares += raw.represented[source] * raw.represented[source];
            raw_error[d] = raw.production_to_ideal_bound[source] +
                           std::abs(raw.represented[source] - raw.ideal[source]);
            raw_error_sum_squares += raw_error[d] * raw_error[d];
        }
        const double represented_norm = std::sqrt(represented_sum_squares);
        const double raw_error_norm = std::sqrt(raw_error_sum_squares);
        const double represented_rms =
            std::sqrt(represented_sum_squares / kHeadDim + 1.0e-6);
        const double production_rms_lower =
            std::sqrt(std::pow(std::max(represented_norm - raw_error_norm, 0.0), 2) /
                          kHeadDim +
                      1.0e-6);
        const double rms_error = raw_error_norm / std::sqrt(static_cast<double>(kHeadDim));
        std::array<double, kHeadDim> normalized{};
        std::array<double, kHeadDim> normalized_bound{};
        for (std::int32_t d = 0; d < kHeadDim; ++d) {
            const double normalized_value =
                raw.represented[raw_base + d] / represented_rms * gamma[d];
            const double norm_perturbation =
                std::abs(static_cast<double>(gamma[d])) *
                (raw_error[d] / production_rms_lower +
                 std::abs(raw.represented[raw_base + d]) * rms_error /
                     (production_rms_lower * represented_rms));
            normalized[d] = normalized_value;
            normalized_bound[d] = norm_perturbation;
        }
        for (std::int32_t d = 0; d < kHeadDim; ++d) {
            double rotated = normalized[d];
            double evaluation_bound = normalized_bound[d] + 1.0e-4;
            if (d < 64) {
                const std::int32_t pair = d & 31;
                const double inverse_frequency =
                    std::pow(1.0e7, -static_cast<double>(pair) / 32.0);
                const double phase =
                    static_cast<double>(position[static_cast<std::size_t>(pair % 3)]) *
                    inverse_frequency;
                const double cosine = std::cos(phase);
                const double sine = std::sin(phase);
                const double lo = normalized[pair];
                const double hi = normalized[pair + 32];
                rotated = d < 32 ? lo * cosine - hi * sine
                                 : hi * cosine + lo * sine;
                evaluation_bound = normalized_bound[pair] + normalized_bound[pair + 32] +
                                   1.0e-4;
            }
            // Carry the independently bounded projection perturbation through RMSNorm/MRoPE,
            // then account for both the production and oracle BF16 stores.
            const double represented = represented_bf16(rotated);
            const double production_round =
                std::max((std::abs(rotated) + evaluation_bound) / 255.0,
                         kMinimumBf16RoundBound);
            result.represented[out_base + d] = represented;
            result.production_to_represented_bound[out_base + d] =
                evaluation_bound + production_round + std::abs(represented - rotated);
        }
    }
    return result;
}

std::vector<double> normalized_index_rope(
    std::span<const double> raw, std::int32_t heads, std::span<const float> gamma,
    const std::array<std::int32_t, 3>& position) {
    if (raw.size() != static_cast<std::size_t>(heads) * kIndex || gamma.size() != kIndex) {
        throw std::logic_error("Qwen4 real QSA index oracle shape changed");
    }
    std::vector<double> result(raw.size());
    for (std::int32_t head = 0; head < heads; ++head) {
        const std::size_t base = static_cast<std::size_t>(head) * kIndex;
        double sum_squares = 0.0;
        for (std::int32_t d = 0; d < kIndex; ++d) {
            sum_squares += raw[base + d] * raw[base + d];
        }
        const double inverse_rms = 1.0 / std::sqrt(sum_squares / kIndex + 1.0e-6);
        std::array<double, kIndex> normalized{};
        for (std::int32_t d = 0; d < kIndex; ++d) {
            normalized[d] = raw[base + d] * inverse_rms * gamma[d];
        }
        for (std::int32_t d = 0; d < kIndex; ++d) {
            double value = normalized[d];
            if (d < 64) {
                const std::int32_t pair = d & 31;
                const double inverse_frequency =
                    std::pow(1.0e7, -static_cast<double>(pair) / 32.0);
                const double phase =
                    static_cast<double>(position[static_cast<std::size_t>(pair % 3)]) *
                    inverse_frequency;
                const double cosine = std::cos(phase);
                const double sine = std::sin(phase);
                value = d < 32 ? normalized[pair] * cosine - normalized[pair + 32] * sine
                               : normalized[pair + 32] * cosine + normalized[pair] * sine;
            }
            result[base + d] = value;
        }
    }
    return result;
}

std::size_t code_index(int byte, int token, int head, int capacity) {
    return static_cast<std::size_t>(byte) + 128U * (token + capacity * head);
}

std::size_t scale_index(int group, int token, int head, int capacity) {
    return static_cast<std::size_t>(group) + 16U * (token + capacity * head);
}

double decode_e2m1(std::uint8_t nibble) {
    constexpr double magnitude[]{0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
    return (nibble & 8U) == 0 ? magnitude[nibble & 7U] : -magnitude[nibble & 7U];
}

double decode_e4m3(std::uint8_t bits) {
    const double sign = (bits & 0x80U) == 0 ? 1.0 : -1.0;
    const unsigned magnitude = bits & 0x7fU;
    const unsigned exponent = magnitude >> 3U;
    const unsigned fraction = magnitude & 7U;
    if (exponent == 0) { return sign * std::ldexp(static_cast<double>(fraction), -9); }
    if (exponent == 15 && fraction == 7) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return sign * std::ldexp(static_cast<double>(8U + fraction),
                             static_cast<int>(exponent) - 10);
}

double decode_cache(std::span<const std::uint8_t> codes,
                    std::span<const std::uint8_t> scales, int d, int token, int head,
                    int capacity) {
    const std::uint8_t packed = codes[code_index(d / 2, token, head, capacity)];
    const std::uint8_t nibble = (d & 1) == 0 ? packed & 0x0fU : packed >> 4U;
    return decode_e2m1(nibble) *
           decode_e4m3(scales[scale_index(d / 16, token, head, capacity)]);
}

std::uint8_t encode_e4m3_rne(float value) {
    if (!std::isfinite(value) || value < 0.0F) {
        throw std::logic_error("Qwen4 real QSA E4M3 scale input is invalid");
    }
    std::uint8_t best = 0;
    double best_distance = INFINITY;
    for (unsigned code = 0; code <= 0x7eU; ++code) {
        const double candidate = decode_e4m3(static_cast<std::uint8_t>(code));
        const double distance = std::abs(static_cast<double>(value) - candidate);
        if (distance < best_distance ||
            (distance == best_distance && (code & 1U) == 0U && (best & 1U) != 0U)) {
            best = static_cast<std::uint8_t>(code);
            best_distance = distance;
        }
    }
    return best;
}

std::uint8_t encode_e2m1_rne(float value) {
    constexpr std::array<double, 8> magnitude = {0.0, 0.5, 1.0, 1.5,
                                                  2.0, 3.0, 4.0, 6.0};
    if (!std::isfinite(value)) {
        throw std::logic_error("Qwen4 real QSA E2M1 input is non-finite");
    }
    const double absolute = std::abs(static_cast<double>(value));
    unsigned best = 0;
    double best_distance = INFINITY;
    for (unsigned code = 0; code < magnitude.size(); ++code) {
        const double distance = std::abs(absolute - magnitude[code]);
        if (distance < best_distance ||
            (distance == best_distance && (code & 1U) == 0U && (best & 1U) != 0U)) {
            best = code;
            best_distance = distance;
        }
    }
    return static_cast<std::uint8_t>(best | (std::signbit(value) ? 8U : 0U));
}

struct HostQsaState {
    explicit HostQsaState(std::int32_t capacity)
        : capacity(capacity),
          k_codes(static_cast<std::size_t>(128) * capacity * kKvHeads),
          v_codes(k_codes.size()),
          k_scales(static_cast<std::size_t>(16) * capacity * kKvHeads),
          v_scales(k_scales.size()),
          raw_index_keys(static_cast<std::size_t>(kIndex) * capacity),
          positions(static_cast<std::size_t>(3) * capacity) {}

    std::int32_t capacity;
    std::vector<std::uint8_t> k_codes;
    std::vector<std::uint8_t> v_codes;
    std::vector<std::uint8_t> k_scales;
    std::vector<std::uint8_t> v_scales;
    std::vector<std::uint16_t> raw_index_keys;
    std::vector<std::int32_t> positions;
};

HostQsaState copy_qsa_state(const ops::QsaStateView& state) {
    const std::int32_t capacity = state.raw_index_keys.ne[1];
    HostQsaState result(capacity);
    result.k_codes = from_device<std::uint8_t>(state.k_codes.data, state.k_codes.numel());
    result.v_codes = from_device<std::uint8_t>(state.v_codes.data, state.v_codes.numel());
    result.k_scales = from_device<std::uint8_t>(state.k_scales.data, state.k_scales.numel());
    result.v_scales = from_device<std::uint8_t>(state.v_scales.data, state.v_scales.numel());
    result.raw_index_keys =
        from_device<std::uint16_t>(state.raw_index_keys.data, state.raw_index_keys.numel());
    result.positions =
        from_device<std::int32_t>(state.positions.data, state.positions.numel());
    return result;
}

struct GuardedQsaState {
    explicit GuardedQsaState(const HostQsaState& source)
        : k_codes(source.k_codes.size()),
          v_codes(source.v_codes.size()),
          k_scales(source.k_scales.size()),
          v_scales(source.v_scales.size()),
          raw_index_keys(source.raw_index_keys.size() * sizeof(std::uint16_t)),
          positions(source.positions.size() * sizeof(std::int32_t)),
          view{
              Tensor(k_codes.data(), DType::U8, {128, source.capacity, kKvHeads}),
              Tensor(v_codes.data(), DType::U8, {128, source.capacity, kKvHeads}),
              Tensor(k_scales.data(), DType::FP8_E4M3FN, {16, source.capacity, kKvHeads}),
              Tensor(v_scales.data(), DType::FP8_E4M3FN, {16, source.capacity, kKvHeads}),
              Tensor(raw_index_keys.data(), DType::BF16, {kIndex, source.capacity}),
              Tensor(positions.data(), DType::I32, {3, source.capacity}),
          } {
        k_codes.copy_from_host(source.k_codes.data(), source.k_codes.size());
        v_codes.copy_from_host(source.v_codes.data(), source.v_codes.size());
        k_scales.copy_from_host(source.k_scales.data(), source.k_scales.size());
        v_scales.copy_from_host(source.v_scales.data(), source.v_scales.size());
        raw_index_keys.copy_from_host(source.raw_index_keys.data(),
                                      source.raw_index_keys.size() * sizeof(std::uint16_t));
        positions.copy_from_host(source.positions.data(),
                                 source.positions.size() * sizeof(std::int32_t));
    }

    int verify_guards() const {
        int failures = k_codes.verify_guards("Qwen4 accumulated QSA K codes");
        failures += v_codes.verify_guards("Qwen4 accumulated QSA V codes");
        failures += k_scales.verify_guards("Qwen4 accumulated QSA K scales");
        failures += v_scales.verify_guards("Qwen4 accumulated QSA V scales");
        failures += raw_index_keys.verify_guards("Qwen4 accumulated QSA raw index keys");
        failures += positions.verify_guards("Qwen4 accumulated QSA positions");
        return failures;
    }

    GuardedDeviceBuffer k_codes;
    GuardedDeviceBuffer v_codes;
    GuardedDeviceBuffer k_scales;
    GuardedDeviceBuffer v_scales;
    GuardedDeviceBuffer raw_index_keys;
    GuardedDeviceBuffer positions;
    ops::QsaStateView view;
};

void store_known_nvfp4_group(std::vector<std::uint8_t>& codes,
                             std::vector<std::uint8_t>& scales, std::int32_t token,
                             std::int32_t head, std::int32_t group, std::int32_t capacity,
                             std::uint8_t scale_bits,
                             const std::array<float, 16>& code_values) {
    scales[scale_index(group, token, head, capacity)] = scale_bits;
    for (std::int32_t pair = 0; pair < 8; ++pair) {
        const std::uint8_t low = encode_e2m1_rne(code_values[2 * pair]);
        const std::uint8_t high = encode_e2m1_rne(code_values[2 * pair + 1]);
        codes[code_index(group * 8 + pair, token, head, capacity)] =
            static_cast<std::uint8_t>(low | (high << 4U));
    }
}

void encode_nvfp4_group(std::vector<std::uint8_t>& codes,
                        std::vector<std::uint8_t>& scales, std::int32_t token,
                        std::int32_t head, std::int32_t group, std::int32_t capacity,
                        std::span<const double, 16> values) {
    float maximum = 0.0F;
    std::array<float, 16> represented{};
    for (std::size_t lane = 0; lane < represented.size(); ++lane) {
        represented[lane] = static_cast<float>(values[lane]);
        maximum = std::max(maximum, std::abs(represented[lane]));
    }
    const std::uint8_t scale_bits = encode_e4m3_rne(maximum / 6.0F);
    scales[scale_index(group, token, head, capacity)] = scale_bits;
    const float scale = static_cast<float>(decode_e4m3(scale_bits));
    for (std::int32_t pair = 0; pair < 8; ++pair) {
        const float low_value = scale == 0.0F ? 0.0F : represented[2 * pair] / scale;
        const float high_value = scale == 0.0F ? 0.0F : represented[2 * pair + 1] / scale;
        const std::uint8_t low = encode_e2m1_rne(low_value);
        const std::uint8_t high = encode_e2m1_rne(high_value);
        codes[code_index(group * 8 + pair, token, head, capacity)] =
            static_cast<std::uint8_t>(low | (high << 4U));
    }
}

void encode_nvfp4_vector(HostQsaState& state, bool key, std::int32_t token,
                         std::int32_t head, std::span<const double> values) {
    if (values.size() != kHeadDim) {
        throw std::logic_error("Qwen4 real QSA NVFP4 vector has the wrong width");
    }
    auto& codes = key ? state.k_codes : state.v_codes;
    auto& scales = key ? state.k_scales : state.v_scales;
    for (std::int32_t group = 0; group < kHeadDim / 16; ++group) {
        const auto group_values = std::span<const double, 16>(values.data() + group * 16, 16);
        encode_nvfp4_group(codes, scales, token, head, group, state.capacity, group_values);
    }
}

void seed_prefix(HostQsaState& state, const NormProjection& core_query,
                 std::span<const double> index_query,
                 std::span<const float> index_key_gamma) {
    constexpr std::array<float, 8> key_magnitude = {0.5F, 1.0F, 1.5F, 2.0F,
                                                     -0.5F, -1.0F, -1.5F, -2.0F};
    constexpr std::array<float, 8> value_magnitude = {-0.5F, 0.5F, -1.0F, 1.0F,
                                                       -1.5F, 1.5F, -2.0F, 2.0F};
    if (core_query.represented.size() != static_cast<std::size_t>(kQueryHeads) * kHeadDim ||
        index_query.size() != static_cast<std::size_t>(kIndexHeads) * kIndex ||
        index_key_gamma.size() != kIndex) {
        throw std::logic_error("Qwen4 real QSA seed witness shape changed");
    }
    for (std::int32_t token = 0; token < 8; ++token) {
        for (std::int32_t head = 0; head < kKvHeads; ++head) {
            const std::int32_t query_head = head * (kQueryHeads / kKvHeads);
            for (std::int32_t group = 0; group < kHeadDim / 16; ++group) {
                std::array<float, 16> key_values{};
                std::array<float, 16> value_values{};
                for (std::int32_t lane = 0; lane < 16; ++lane) {
                    const std::int32_t d = group * 16 + lane;
                    const double q = core_query.represented[
                        static_cast<std::size_t>(query_head) * kHeadDim + d];
                    const float sign = std::signbit(q) ? -1.0F : 1.0F;
                    key_values[lane] = sign * key_magnitude[token];
                    value_values[lane] = (head == 0 ? 1.0F : -1.0F) *
                                         value_magnitude[token] *
                                         ((d & 1) == 0 ? 1.0F : -1.0F);
                }
                // These are deliberately hand-packed known codes. E4M3 0x18 is 1/16, keeping
                // the resulting attention logits nonuniform without saturating its softmax.
                store_known_nvfp4_group(state.k_codes, state.k_scales, token, head, group,
                                         state.capacity, 0x18U, key_values);
                store_known_nvfp4_group(state.v_codes, state.v_scales, token, head, group,
                                         state.capacity, 0x38U, value_values);
            }
        }
    }

    // The first complete block has an exact zero score. The second block's four identical raw
    // keys align with the summed independently normalized real index queries, so it must rank
    // first; without the real index-query projection both blocks tie and lower block 0 wins.
    for (std::int32_t d = 0; d < kIndex; ++d) {
        double summed_query = 0.0;
        for (std::int32_t head = 0; head < kIndexHeads; ++head) {
            summed_query += index_query[static_cast<std::size_t>(head) * kIndex + d];
        }
        const double aligned = summed_query * static_cast<double>(index_key_gamma[d]);
        const std::uint16_t bits = f32_to_bf16(aligned < 0.0 ? -1.0F : 1.0F);
        for (std::int32_t token = 4; token < 8; ++token) {
            state.raw_index_keys[static_cast<std::size_t>(token) * kIndex + d] = bits;
        }
    }
}

double selector_block_score(const HostQsaState& state, std::int32_t first_id,
                            std::span<const double> query,
                            std::span<const float> key_gamma) {
    std::array<double, kIndex> pooled{};
    double sum_squares = 0.0;
    for (std::int32_t d = 0; d < kIndex; ++d) {
        double sum = 0.0;
        for (std::int32_t token = 0; token < 4; ++token) {
            sum += bf16_to_f32(
                state.raw_index_keys[static_cast<std::size_t>(first_id + token) * kIndex + d]);
        }
        pooled[d] = represented_bf16(sum * 0.25);
        sum_squares += pooled[d] * pooled[d];
    }
    const double inverse_rms = 1.0 / std::sqrt(sum_squares / kIndex + 1.0e-6);
    std::array<double, kIndex> normalized{};
    for (std::int32_t d = 0; d < kIndex; ++d) {
        normalized[d] = pooled[d] * inverse_rms * key_gamma[d];
    }
    const std::array<std::int32_t, 3> position = {
        state.positions[static_cast<std::size_t>(3) * first_id],
        state.positions[static_cast<std::size_t>(3) * first_id + 1],
        state.positions[static_cast<std::size_t>(3) * first_id + 2],
    };
    std::array<double, kIndex> rotated = normalized;
    for (std::int32_t pair = 0; pair < 32; ++pair) {
        const double inverse_frequency =
            std::pow(1.0e7, -static_cast<double>(pair) / 32.0);
        const double phase = static_cast<double>(position[pair % 3]) * inverse_frequency;
        const double cosine = std::cos(phase);
        const double sine = std::sin(phase);
        rotated[pair] = normalized[pair] * cosine - normalized[pair + 32] * sine;
        rotated[pair + 32] = normalized[pair + 32] * cosine + normalized[pair] * sine;
    }
    double score = 0.0;
    for (std::int32_t head = 0; head < kIndexHeads; ++head) {
        double dot = 0.0;
        for (std::int32_t d = 0; d < kIndex; ++d) {
            dot += query[static_cast<std::size_t>(head) * kIndex + d] * rotated[d];
        }
        score += std::max(dot, 0.0);
    }
    return score / std::sqrt(static_cast<double>(kIndex));
}

int verify_bounded(const char* label, std::span<const double> actual,
                   std::span<const double> expected, std::span<const double> bound) {
    if (actual.size() != expected.size() || actual.size() != bound.size()) {
        throw std::logic_error(std::string(label) + " bound shape mismatch");
    }
    int violations = 0;
    double maximum_ratio = 0.0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const double error = std::abs(actual[i] - expected[i]);
        const double ratio = bound[i] == 0.0 ? (error == 0.0 ? 0.0 : INFINITY)
                                             : error / bound[i];
        maximum_ratio = std::max(maximum_ratio, ratio);
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]) ||
            !std::isfinite(bound[i]) || error > bound[i]) {
            ++violations;
        }
    }
    if (violations != 0) {
        std::cerr << "FAIL: " << label << " violations=" << violations
                  << " max_bound_ratio=" << maximum_ratio << '\n';
    }
    if (error_stats_enabled()) {
        std::cout << "OP_ERROR_STATS kind=derived_bound count=" << actual.size()
                  << " max_bound_ratio=" << maximum_ratio << " case=" << label << '\n';
    }
    return violations == 0 ? 0 : 1;
}

int run_accumulated_qsa_cell(const verifier::LoadedModel& model, DeviceContext& device) {
    constexpr std::int32_t current_id = 227;
    const ops::QsaVerifierWeights& weights = *model.view().layers[3].qsa;

    HostQsaState initial_state(verifier::kQsaCapacity);
    GuardedDeviceBuffer qsa_x(static_cast<std::size_t>(kHidden) * sizeof(std::uint16_t));
    qsa_x.fill(0xcd);
    {
        verifier::Program program(model, device, verifier::DiagnosticSnapshots::Enabled);
        program.reset();
        for (std::int32_t position = 0; position < current_id; ++position) {
            const std::int32_t token =
                kFrozenParagraph[static_cast<std::size_t>(position) % kFrozenParagraph.size()];
            const std::int32_t target = kFrozenParagraph[
                (static_cast<std::size_t>(position) + 1U) % kFrozenParagraph.size()];
            (void) program.execute_token(token, target);
        }
        if (program.frontier() != current_id || !program.state().qsa()[3].has_value()) {
            throw std::logic_error("Qwen4 accumulated QSA pre-append frontier/state changed");
        }
        initial_state = copy_qsa_state(*program.state().qsa()[3]);

        const std::int32_t token =
            kFrozenParagraph[static_cast<std::size_t>(current_id) % kFrozenParagraph.size()];
        const std::int32_t target = kFrozenParagraph[
            (static_cast<std::size_t>(current_id) + 1U) % kFrozenParagraph.size()];
        const verifier::TokenResultView result = program.execute_token(token, target);
        if (result.token_index != current_id || result.gr.size() != verifier::kLayerCount ||
            result.gr[2].layer != 2 || result.gr[2].ffn_residual.dtype != DType::BF16 ||
            result.gr[2].ffn_residual.ne[0] != kHidden ||
            result.gr[2].ffn_residual.ne[1] != 4) {
            throw std::logic_error("Qwen4 accumulated QSA layer-2 residual diagnostic changed");
        }

        Tensor x_tensor(qsa_x.data(), DType::BF16, {kHidden});
        WorkspaceArena gr_workspace(ops::gated_residual_workspace_capacity_bytes());
        const verifier::GrWeights& layer3_gr = model.view().layers[3].attention_gr;
        ops::gated_residual_read(result.gr[2].ffn_residual, layer3_gr.norm, layer3_gr.down,
                                 layer3_gr.up, x_tensor, gr_workspace, device.stream);
        device.synchronize();
    }

    const std::vector<std::uint16_t> residual_bits =
        from_device<std::uint16_t>(qsa_x.data(), kHidden);
    std::vector<double> residual(kHidden);
    std::transform(residual_bits.begin(), residual_bits.end(), residual.begin(),
                   [](std::uint16_t bits) { return static_cast<double>(bf16_to_f32(bits)); });

    const auto index_query_bits = copy_device_values<std::uint16_t>(
        weights.index_query.qdata, static_cast<std::size_t>(kIndex * kIndexHeads) * kHidden);
    const auto index_key_bits = copy_device_values<std::uint16_t>(
        weights.index_key.qdata, static_cast<std::size_t>(kIndex) * kHidden);
    const auto query_gate_bytes =
        copy_device_bytes(weights.core_query_gate.qdata, weights.core_query_gate.payload_bytes);
    const auto key_bytes =
        copy_device_bytes(weights.core_key.qdata, weights.core_key.payload_bytes);
    const auto value_bytes =
        copy_device_bytes(weights.core_value.qdata, weights.core_value.payload_bytes);
    const auto output_bytes =
        copy_device_bytes(weights.output.qdata, weights.output.payload_bytes);
    const auto index_query_gamma =
        copy_device_values<float>(weights.index_query_norm.data, kIndex);
    const auto index_key_gamma = copy_device_values<float>(weights.index_key_norm.data, kIndex);
    const auto core_query_gamma =
        copy_device_values<float>(weights.core_query_norm.data, kHeadDim);
    const auto core_key_gamma =
        copy_device_values<float>(weights.core_key_norm.data, kHeadDim);

    const Projection index_query =
        bf16_project(index_query_bits, kIndex * kIndexHeads, residual);
    const Projection index_key = bf16_project(index_key_bits, kIndex, residual);
    const Projection raw_query_gate =
        q5_project(query_gate_bytes, kQueryGateRows, kHidden, residual);
    const Projection raw_key =
        q5_project(key_bytes, kKvHeads * kHeadDim, kHidden, residual);
    const Projection raw_value =
        q5_project(value_bytes, kKvHeads * kHeadDim, kHidden, residual);
    const std::array<std::int32_t, 3> position{current_id, current_id, current_id};
    const NormProjection query =
        core_norm_rope(raw_query_gate, kQueryHeads, 2 * kHeadDim, core_query_gamma, position);
    const NormProjection key =
        core_norm_rope(raw_key, kKvHeads, kHeadDim, core_key_gamma, position);
    const std::vector<double> normalized_index_query =
        normalized_index_rope(index_query.represented, kIndexHeads, index_query_gamma, position);

    HostQsaState expected_state = initial_state;
    for (std::int32_t head = 0; head < kKvHeads; ++head) {
        encode_nvfp4_vector(
            expected_state, true, current_id, head,
            std::span<const double>(key.represented.data() +
                                        static_cast<std::size_t>(head) * kHeadDim,
                                    kHeadDim));
        encode_nvfp4_vector(
            expected_state, false, current_id, head,
            std::span<const double>(raw_value.represented.data() +
                                        static_cast<std::size_t>(head) * kHeadDim,
                                    kHeadDim));
    }
    for (std::int32_t d = 0; d < kIndex; ++d) {
        expected_state.raw_index_keys[static_cast<std::size_t>(current_id) * kIndex + d] =
            f32_to_bf16(static_cast<float>(index_key.represented[d]));
    }
    for (std::int32_t axis = 0; axis < 3; ++axis) {
        expected_state.positions[static_cast<std::size_t>(3) * current_id + axis] = position[axis];
    }

    struct RankedBlock {
        std::int32_t rank;
        double score;
    };
    constexpr std::int32_t selected_count = current_id + 1;
    constexpr std::int32_t complete_blocks = selected_count / 4;
    std::vector<RankedBlock> blocks;
    blocks.reserve(complete_blocks);
    for (std::int32_t block = 0; block < complete_blocks; ++block) {
        const double score = selector_block_score(expected_state, block * 4,
                                                  normalized_index_query, index_key_gamma);
        if (!std::isfinite(score)) {
            throw std::logic_error("Qwen4 accumulated QSA selector score is non-finite");
        }
        blocks.push_back({block, score});
    }
    std::stable_sort(blocks.begin(), blocks.end(), [](const RankedBlock& lhs,
                                                       const RankedBlock& rhs) {
        return lhs.score > rhs.score || (lhs.score == rhs.score && lhs.rank < rhs.rank);
    });
    std::vector<std::int32_t> expected_selected(ops::kQsaSelectedCapacity, -1);
    for (std::size_t rank = 0; rank < blocks.size(); ++rank) {
        for (std::int32_t lane = 0; lane < 4; ++lane) {
            expected_selected[rank * 4 + static_cast<std::size_t>(lane)] =
                blocks[rank].rank * 4 + lane;
        }
    }
    for (std::int32_t id = complete_blocks * 4; id < selected_count; ++id) {
        expected_selected[static_cast<std::size_t>(complete_blocks) * 4 +
                          static_cast<std::size_t>(id - complete_blocks * 4)] = id;
    }
    const auto [minimum_score, maximum_score] = std::minmax_element(
        blocks.begin(), blocks.end(),
        [](const RankedBlock& lhs, const RankedBlock& rhs) { return lhs.score < rhs.score; });
    bool selector_reordered = false;
    for (std::size_t rank = 0; rank < blocks.size(); ++rank) {
        selector_reordered |= blocks[rank].rank != static_cast<std::int32_t>(rank);
    }
    if (!(maximum_score->score > minimum_score->score) || !selector_reordered) {
        throw std::logic_error("Qwen4 accumulated QSA selector witness is degenerate");
    }

    std::vector<std::int32_t> visible_ids(selected_count);
    std::iota(visible_ids.begin(), visible_ids.end(), 0);
    const std::vector<std::int32_t> visible_offsets{0, selected_count};
    const std::vector<std::int32_t> token_id{current_id};
    const std::vector<std::int32_t> position_values(position.begin(), position.end());
    DeviceBuffer device_token_id = to_device(token_id);
    DeviceBuffer device_position = to_device(position_values);
    DeviceBuffer device_visible_ids = to_device(visible_ids);
    DeviceBuffer device_visible_offsets = to_device(visible_offsets);
    GuardedQsaState device_state(initial_state);
    GuardedDeviceBuffer device_selected(
        static_cast<std::size_t>(ops::kQsaSelectedCapacity) * sizeof(std::int32_t));
    GuardedDeviceBuffer device_count(sizeof(std::int32_t));
    GuardedDeviceBuffer device_output(static_cast<std::size_t>(kHidden) * sizeof(std::uint16_t));
    GuardedDeviceBuffer workspace(ops::qsa_verifier_workspace_bytes());
    device_selected.fill(0xcd);
    device_count.fill(0xcd);
    device_output.fill(0xcd);
    workspace.fill(0xcd);

    Tensor x_tensor(qsa_x.data(), DType::BF16, {kHidden});
    Tensor token_id_tensor(device_token_id.p, DType::I32, {1});
    Tensor position_tensor(device_position.p, DType::I32, {3});
    Tensor visible_ids_tensor(device_visible_ids.p, DType::I32, {selected_count});
    Tensor visible_offsets_tensor(device_visible_offsets.p, DType::I32, {2});
    Tensor selected_tensor(device_selected.data(), DType::I32, {ops::kQsaSelectedCapacity});
    Tensor count_tensor(device_count.data(), DType::I32, {1});
    Tensor output_tensor(device_output.data(), DType::BF16, {kHidden});
    Tensor workspace_tensor(workspace.data(), DType::U8,
                            {static_cast<std::int32_t>(workspace.bytes())});
    ops::qsa_verifier_token(x_tensor, token_id_tensor, position_tensor, visible_ids_tensor,
                            visible_offsets_tensor, weights, device_state.view, selected_tensor,
                            count_tensor, output_tensor, workspace_tensor, device.stream);
    device.synchronize();

    int failures = 0;
    failures += verify_exact("Qwen4 accumulated QSA selected count",
                             from_device<std::int32_t>(device_count.data(), 1),
                             std::vector<std::int32_t>{selected_count});
    failures += verify_exact(
        "Qwen4 accumulated QSA exact all-visible selector order and padding",
        from_device<std::int32_t>(device_selected.data(), ops::kQsaSelectedCapacity),
        expected_selected);

    HostQsaState actual_state = copy_qsa_state(device_state.view);
    std::vector<std::uint8_t> actual_current_k_codes;
    std::vector<std::uint8_t> actual_current_v_codes;
    std::vector<std::uint8_t> expected_current_k_codes;
    std::vector<std::uint8_t> expected_current_v_codes;
    std::vector<std::uint8_t> actual_current_k_scales;
    std::vector<std::uint8_t> actual_current_v_scales;
    std::vector<std::uint8_t> expected_current_k_scales;
    std::vector<std::uint8_t> expected_current_v_scales;
    for (std::int32_t head = 0; head < kKvHeads; ++head) {
        for (std::int32_t byte = 0; byte < kHeadDim / 2; ++byte) {
            const std::size_t index = code_index(byte, current_id, head, initial_state.capacity);
            actual_current_k_codes.push_back(actual_state.k_codes[index]);
            actual_current_v_codes.push_back(actual_state.v_codes[index]);
            expected_current_k_codes.push_back(expected_state.k_codes[index]);
            expected_current_v_codes.push_back(expected_state.v_codes[index]);
            actual_state.k_codes[index] = initial_state.k_codes[index];
            actual_state.v_codes[index] = initial_state.v_codes[index];
        }
        for (std::int32_t group = 0; group < kHeadDim / 16; ++group) {
            const std::size_t index =
                scale_index(group, current_id, head, initial_state.capacity);
            actual_current_k_scales.push_back(actual_state.k_scales[index]);
            actual_current_v_scales.push_back(actual_state.v_scales[index]);
            expected_current_k_scales.push_back(expected_state.k_scales[index]);
            expected_current_v_scales.push_back(expected_state.v_scales[index]);
            actual_state.k_scales[index] = initial_state.k_scales[index];
            actual_state.v_scales[index] = initial_state.v_scales[index];
        }
    }
    failures += verify_exact("Qwen4 accumulated QSA independently encoded current K codes",
                             actual_current_k_codes, expected_current_k_codes);
    failures += verify_exact("Qwen4 accumulated QSA independently encoded current V codes",
                             actual_current_v_codes, expected_current_v_codes);
    failures += verify_exact("Qwen4 accumulated QSA independently encoded current K scales",
                             actual_current_k_scales, expected_current_k_scales);
    failures += verify_exact("Qwen4 accumulated QSA independently encoded current V scales",
                             actual_current_v_scales, expected_current_v_scales);
    failures += verify_exact("Qwen4 accumulated QSA untouched K code plane", actual_state.k_codes,
                             initial_state.k_codes);
    failures += verify_exact("Qwen4 accumulated QSA untouched V code plane", actual_state.v_codes,
                             initial_state.v_codes);
    failures += verify_exact("Qwen4 accumulated QSA untouched K scale plane",
                             actual_state.k_scales, initial_state.k_scales);
    failures += verify_exact("Qwen4 accumulated QSA untouched V scale plane",
                             actual_state.v_scales, initial_state.v_scales);

    std::vector<double> actual_raw_key(kIndex);
    std::vector<double> raw_key_bound(kIndex);
    for (std::int32_t d = 0; d < kIndex; ++d) {
        const std::size_t index = static_cast<std::size_t>(current_id) * kIndex + d;
        actual_raw_key[d] = bf16_to_f32(actual_state.raw_index_keys[index]);
        actual_state.raw_index_keys[index] = initial_state.raw_index_keys[index];
        raw_key_bound[d] = index_key.production_to_ideal_bound[d] +
                           std::abs(index_key.represented[d] - index_key.ideal[d]);
    }
    failures += verify_bounded("qwen4_accumulated_qsa_raw_index_key", actual_raw_key,
                               index_key.represented, raw_key_bound);
    failures += verify_exact("Qwen4 accumulated QSA untouched raw-index-key plane",
                             actual_state.raw_index_keys, initial_state.raw_index_keys);
    failures += verify_exact(
        "Qwen4 accumulated QSA exact current three-axis position",
        std::vector<std::int32_t>(actual_state.positions.begin() + 3 * current_id,
                                  actual_state.positions.begin() + 3 * current_id + 3),
        position_values);
    std::copy(initial_state.positions.begin() + 3 * current_id,
              initial_state.positions.begin() + 3 * current_id + 3,
              actual_state.positions.begin() + 3 * current_id);
    failures += verify_exact("Qwen4 accumulated QSA untouched position plane",
                             actual_state.positions, initial_state.positions);

    std::vector<double> gated(kOutputColumns);
    double maximum_logit_spread = 0.0;
    for (std::int32_t head = 0; head < kQueryHeads; ++head) {
        const std::int32_t kv_head = head / (kQueryHeads / kKvHeads);
        std::vector<double> logits(selected_count);
        double maximum = -INFINITY;
        double minimum = INFINITY;
        for (std::int32_t item = 0; item < selected_count; ++item) {
            const std::int32_t id = expected_selected[static_cast<std::size_t>(item)];
            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                logits[item] +=
                    query.represented[static_cast<std::size_t>(head) * kHeadDim + d] *
                    decode_cache(expected_state.k_codes, expected_state.k_scales, d, id,
                                 kv_head, expected_state.capacity);
            }
            logits[item] /= std::sqrt(static_cast<double>(kHeadDim));
            maximum = std::max(maximum, logits[item]);
            minimum = std::min(minimum, logits[item]);
        }
        maximum_logit_spread = std::max(maximum_logit_spread, maximum - minimum);
        std::vector<double> probabilities(selected_count);
        double denominator = 0.0;
        for (std::int32_t item = 0; item < selected_count; ++item) {
            probabilities[item] = std::exp(logits[item] - maximum);
            denominator += probabilities[item];
        }
        for (double& probability : probabilities) { probability /= denominator; }
        for (std::int32_t d = 0; d < kHeadDim; ++d) {
            double attention = 0.0;
            for (std::int32_t item = 0; item < selected_count; ++item) {
                attention += probabilities[item] *
                             decode_cache(expected_state.v_codes, expected_state.v_scales, d,
                                          expected_selected[item], kv_head,
                                          expected_state.capacity);
            }
            const double represented_attention = represented_bf16(attention);
            const std::size_t gate_index =
                static_cast<std::size_t>(head) * 2 * kHeadDim + kHeadDim + d;
            gated[static_cast<std::size_t>(head) * kHeadDim + d] = represented_bf16(
                represented_attention * sigmoid(raw_query_gate.represented[gate_index]));
        }
    }
    if (!(maximum_logit_spread > 0.25) || !std::isfinite(maximum_logit_spread)) {
        throw std::logic_error("Qwen4 accumulated QSA attention witness is degenerate");
    }
    const Projection projected_output =
        q5_project(output_bytes, kHidden, kOutputColumns, gated);
    std::vector<double> expected_output(kHidden);
    std::transform(projected_output.ideal.begin(), projected_output.ideal.end(),
                   expected_output.begin(),
                   [](double value) { return represented_bf16(value); });
    failures += verify_pointwise("Qwen4 accumulated layer-3 QSA complete 228-item FP64 oracle",
                                 from_device_bf16(device_output.data(), kHidden), expected_output,
                                 kCompleteCriterion);

    if (error_stats_enabled()) {
        std::cout << "OP_ERROR_STATS kind=witness selector_score_spread="
                  << maximum_score->score - minimum_score->score
                  << " attention_logit_spread=" << maximum_logit_spread
                  << " selected_count=" << selected_count
                  << " case=Qwen4_accumulated_QSA_position_227\n";
    }
    failures += device_state.verify_guards();
    failures += device_selected.verify_guards("Qwen4 accumulated QSA selected ids");
    failures += device_count.verify_guards("Qwen4 accumulated QSA selected count");
    failures += device_output.verify_guards("Qwen4 accumulated QSA output");
    failures += workspace.verify_guards("Qwen4 accumulated QSA workspace");
    failures += qsa_x.verify_guards("Qwen4 accumulated QSA represented input");
    std::cout << (failures == 0 ? "OK" : "FAIL")
              << " qwen4_real_qsa_accumulated_position_227_cell\n";
    return failures;
}

} // namespace

int ninfer::test::qwen4::real_oracle::run_qsa_cell(const verifier::LoadedModel& model,
                                                   DeviceContext& device) {
    if (!model.view().layers[3].qsa.has_value()) {
        throw std::logic_error("Qwen4 layer 3 is not a QSA layer");
    }
    const ops::QsaVerifierWeights& weights = *model.view().layers[3].qsa;
    require_bf16(weights.index_query, kIndex * kIndexHeads, "index query");
    require_bf16(weights.index_key, kIndex, "index key");
    require_q5(weights.core_query_gate, kQueryGateRows, kHidden, "core query/gate");
    require_q5(weights.core_key, kKvHeads * kHeadDim, kHidden, "core key");
    require_q5(weights.core_value, kKvHeads * kHeadDim, kHidden, "core value");
    require_q5(weights.output, kHidden, kOutputColumns, "output");
    if (weights.index_query_norm.dtype != DType::FP32 ||
        weights.index_query_norm.numel() != kIndex ||
        weights.index_key_norm.dtype != DType::FP32 || weights.index_key_norm.numel() != kIndex ||
        weights.core_query_norm.dtype != DType::FP32 ||
        weights.core_query_norm.numel() != kHeadDim || weights.core_key_norm.dtype != DType::FP32 ||
        weights.core_key_norm.numel() != kHeadDim) {
        throw std::logic_error("Qwen4 layer-3 QSA norm binding changed");
    }

    GuardedDeviceBuffer qsa_x(static_cast<std::size_t>(kHidden) * sizeof(std::uint16_t));
    qsa_x.fill(0xcd);
    {
        verifier::Program program(model, device, verifier::DiagnosticSnapshots::Enabled);
        program.reset();
        const verifier::TokenResultView program_result = program.execute_token(0, 0);
        if (program_result.gr.size() != verifier::kLayerCount ||
            program_result.gr[2].layer != 2 ||
            program_result.gr[2].ffn_residual.dtype != DType::BF16 ||
            program_result.gr[2].ffn_residual.ne[0] != kHidden ||
            program_result.gr[2].ffn_residual.ne[1] != 4) {
            throw std::logic_error("Qwen4 Program layer-2 FFN residual snapshot changed");
        }
        DeviceBuffer layer2_ffn_residual(static_cast<std::size_t>(kHidden) * 4 *
                                         sizeof(std::uint16_t));
        CUDA_CHECK(cudaMemcpyAsync(layer2_ffn_residual.p, program_result.gr[2].ffn_residual.data,
                                   layer2_ffn_residual.bytes, cudaMemcpyDeviceToDevice,
                                   device.stream));
        Tensor layer2_ffn_residual_tensor(layer2_ffn_residual.p, DType::BF16, {kHidden, 4});
        Tensor qsa_x_tensor(qsa_x.data(), DType::BF16, {kHidden});
        WorkspaceArena gr_workspace(ops::gated_residual_workspace_capacity_bytes());
        const verifier::GrWeights& layer3_gr = model.view().layers[3].attention_gr;
        ops::gated_residual_read(layer2_ffn_residual_tensor, layer3_gr.norm, layer3_gr.down,
                                 layer3_gr.up, qsa_x_tensor, gr_workspace, device.stream);
        device.synchronize();
    }
    const std::vector<std::uint16_t> residual_bits =
        from_device<std::uint16_t>(qsa_x.data(), kHidden);
    std::vector<double> residual(kHidden);
    std::transform(residual_bits.begin(), residual_bits.end(), residual.begin(),
                   [](std::uint16_t bits) { return static_cast<double>(bf16_to_f32(bits)); });

    const auto index_query_bits = copy_device_values<std::uint16_t>(
        weights.index_query.qdata, static_cast<std::size_t>(kIndex * kIndexHeads) * kHidden);
    const auto index_key_bits = copy_device_values<std::uint16_t>(
        weights.index_key.qdata, static_cast<std::size_t>(kIndex) * kHidden);
    const auto query_gate_bytes =
        copy_device_bytes(weights.core_query_gate.qdata, weights.core_query_gate.payload_bytes);
    const auto key_bytes =
        copy_device_bytes(weights.core_key.qdata, weights.core_key.payload_bytes);
    const auto value_bytes =
        copy_device_bytes(weights.core_value.qdata, weights.core_value.payload_bytes);
    const auto output_bytes =
        copy_device_bytes(weights.output.qdata, weights.output.payload_bytes);
    const auto index_query_gamma =
        copy_device_values<float>(weights.index_query_norm.data, kIndex);
    const auto index_key_gamma = copy_device_values<float>(weights.index_key_norm.data, kIndex);
    const auto core_query_gamma =
        copy_device_values<float>(weights.core_query_norm.data, kHeadDim);
    const auto core_key_gamma = copy_device_values<float>(weights.core_key_norm.data, kHeadDim);

    const Projection index_query =
        bf16_project(index_query_bits, kIndex * kIndexHeads, residual);
    const Projection index_key = bf16_project(index_key_bits, kIndex, residual);
    const Projection raw_query_gate =
        q5_project(query_gate_bytes, kQueryGateRows, kHidden, residual);
    const Projection raw_key = q5_project(key_bytes, kKvHeads * kHeadDim, kHidden, residual);
    const Projection raw_value = q5_project(value_bytes, kKvHeads * kHeadDim, kHidden, residual);
    const std::array<std::int32_t, 3> position{3, 5, 7};
    const NormProjection query =
        core_norm_rope(raw_query_gate, kQueryHeads, 2 * kHeadDim, core_query_gamma, position);
    const NormProjection key =
        core_norm_rope(raw_key, kKvHeads, kHeadDim, core_key_gamma, position);

    const std::vector<double> normalized_index_query =
        normalized_index_rope(index_query.represented, kIndexHeads, index_query_gamma, position);
    const std::vector<double> normalized_index_key =
        normalized_index_rope(index_key.represented, 1, index_key_gamma, position);
    if (!std::all_of(normalized_index_query.begin(), normalized_index_query.end(),
                     [](double value) { return std::isfinite(value); }) ||
        !std::all_of(normalized_index_key.begin(), normalized_index_key.end(),
                     [](double value) { return std::isfinite(value); })) {
        throw std::logic_error("Qwen4 real QSA index oracle produced a non-finite value");
    }

    verifier::State clean_state;
    clean_state.reset(device.stream);
    device.synchronize();
    if (!clean_state.qsa()[3].has_value()) {
        throw std::logic_error("Qwen4 clean state has no layer-3 QSA planes");
    }
    ops::QsaStateView state = *clean_state.qsa()[3];
    const std::int32_t state_capacity = state.raw_index_keys.ne[1];
    if (state_capacity != verifier::kQsaCapacity) {
        throw std::logic_error("Qwen4 layer-3 QSA state capacity changed");
    }
    HostQsaState initial_state(state_capacity);
    seed_prefix(initial_state, query, normalized_index_query, index_key_gamma);
    HostQsaState expected_state = initial_state;
    constexpr std::int32_t current_id = 8;
    for (std::int32_t head = 0; head < kKvHeads; ++head) {
        encode_nvfp4_vector(
            expected_state, true, current_id, head,
            std::span<const double>(key.represented.data() +
                                        static_cast<std::size_t>(head) * kHeadDim,
                                    kHeadDim));
        encode_nvfp4_vector(
            expected_state, false, current_id, head,
            std::span<const double>(raw_value.represented.data() +
                                        static_cast<std::size_t>(head) * kHeadDim,
                                    kHeadDim));
    }
    for (std::int32_t d = 0; d < kIndex; ++d) {
        expected_state.raw_index_keys[static_cast<std::size_t>(current_id) * kIndex + d] =
            f32_to_bf16(static_cast<float>(index_key.represented[d]));
    }
    for (std::int32_t axis = 0; axis < 3; ++axis) {
        expected_state.positions[static_cast<std::size_t>(3) * current_id + axis] = position[axis];
    }

    CUDA_CHECK(cudaMemcpy(state.k_codes.data, initial_state.k_codes.data(),
                          initial_state.k_codes.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.v_codes.data, initial_state.v_codes.data(),
                          initial_state.v_codes.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.k_scales.data, initial_state.k_scales.data(),
                          initial_state.k_scales.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.v_scales.data, initial_state.v_scales.data(),
                          initial_state.v_scales.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.raw_index_keys.data, initial_state.raw_index_keys.data(),
                          initial_state.raw_index_keys.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.positions.data, initial_state.positions.data(),
                          initial_state.positions.size() * sizeof(std::int32_t),
                          cudaMemcpyHostToDevice));

    const double block_zero_score =
        selector_block_score(expected_state, 0, normalized_index_query, index_key_gamma);
    const double block_one_score =
        selector_block_score(expected_state, 4, normalized_index_query, index_key_gamma);
    if (block_zero_score != 0.0 || !(block_one_score > block_zero_score) ||
        !std::isfinite(block_one_score)) {
        throw std::logic_error("Qwen4 real QSA index-query ranking witness is ineffective");
    }

    const std::vector<std::int32_t> token_id{current_id};
    const std::vector<std::int32_t> position_values(position.begin(), position.end());
    std::vector<std::int32_t> visible_ids(current_id + 1);
    std::iota(visible_ids.begin(), visible_ids.end(), 0);
    const std::vector<std::int32_t> visible_offsets{0, current_id + 1};
    DeviceBuffer device_token_id = to_device(token_id);
    DeviceBuffer device_position = to_device(position_values);
    DeviceBuffer device_visible_ids = to_device(visible_ids);
    DeviceBuffer device_visible_offsets = to_device(visible_offsets);
    GuardedDeviceBuffer device_selected(
        static_cast<std::size_t>(ops::kQsaSelectedCapacity) * sizeof(std::int32_t));
    GuardedDeviceBuffer device_count(sizeof(std::int32_t));
    GuardedDeviceBuffer device_output(static_cast<std::size_t>(kHidden) * sizeof(std::uint16_t));
    GuardedDeviceBuffer workspace(ops::qsa_verifier_workspace_bytes());
    device_selected.fill(0xcd);
    device_count.fill(0xcd);
    device_output.fill(0xcd);
    workspace.fill(0xcd);

    Tensor x_tensor(qsa_x.data(), DType::BF16, {kHidden});
    Tensor token_id_tensor(device_token_id.p, DType::I32, {1});
    Tensor position_tensor(device_position.p, DType::I32, {3});
    Tensor visible_ids_tensor(device_visible_ids.p, DType::I32,
                              {static_cast<std::int32_t>(visible_ids.size())});
    Tensor visible_offsets_tensor(device_visible_offsets.p, DType::I32, {2});
    Tensor selected_tensor(device_selected.data(), DType::I32, {ops::kQsaSelectedCapacity});
    Tensor count_tensor(device_count.data(), DType::I32, {1});
    Tensor output_tensor(device_output.data(), DType::BF16, {kHidden});
    Tensor workspace_tensor(workspace.data(), DType::U8,
                            {static_cast<std::int32_t>(workspace.bytes())});
    ops::qsa_verifier_token(x_tensor, token_id_tensor, position_tensor, visible_ids_tensor,
                            visible_offsets_tensor, weights, state, selected_tensor,
                            count_tensor, output_tensor, workspace_tensor, device.stream);
    device.synchronize();

    int failures = 0;
    failures += verify_exact("Qwen4 real QSA selected count",
                             from_device<std::int32_t>(device_count.data(), 1),
                             std::vector<std::int32_t>{current_id + 1});
    std::vector<std::int32_t> expected_selected(ops::kQsaSelectedCapacity, -1);
    constexpr std::array<std::int32_t, 9> selected_prefix = {4, 5, 6, 7, 0, 1, 2, 3, 8};
    std::copy(selected_prefix.begin(), selected_prefix.end(), expected_selected.begin());
    failures += verify_exact(
        "Qwen4 real QSA index-query ranked blocks and current-token tail",
        from_device<std::int32_t>(device_selected.data(), ops::kQsaSelectedCapacity),
        expected_selected);

    auto actual_k_codes = from_device<std::uint8_t>(state.k_codes.data, state.k_codes.bytes());
    auto actual_v_codes = from_device<std::uint8_t>(state.v_codes.data, state.v_codes.bytes());
    auto actual_k_scales =
        from_device<std::uint8_t>(state.k_scales.data, state.k_scales.bytes());
    auto actual_v_scales =
        from_device<std::uint8_t>(state.v_scales.data, state.v_scales.bytes());
    auto actual_raw_key_bits =
        from_device<std::uint16_t>(state.raw_index_keys.data, state.raw_index_keys.numel());
    auto actual_positions =
        from_device<std::int32_t>(state.positions.data, state.positions.numel());

    std::vector<std::uint8_t> actual_current_k_codes;
    std::vector<std::uint8_t> actual_current_v_codes;
    std::vector<std::uint8_t> expected_current_k_codes;
    std::vector<std::uint8_t> expected_current_v_codes;
    std::vector<std::uint8_t> actual_current_k_scales;
    std::vector<std::uint8_t> actual_current_v_scales;
    std::vector<std::uint8_t> expected_current_k_scales;
    std::vector<std::uint8_t> expected_current_v_scales;
    for (std::int32_t head = 0; head < kKvHeads; ++head) {
        for (std::int32_t byte = 0; byte < kHeadDim / 2; ++byte) {
            const std::size_t index = code_index(byte, current_id, head, state_capacity);
            actual_current_k_codes.push_back(actual_k_codes[index]);
            actual_current_v_codes.push_back(actual_v_codes[index]);
            expected_current_k_codes.push_back(expected_state.k_codes[index]);
            expected_current_v_codes.push_back(expected_state.v_codes[index]);
            actual_k_codes[index] = initial_state.k_codes[index];
            actual_v_codes[index] = initial_state.v_codes[index];
        }
        for (std::int32_t group = 0; group < kHeadDim / 16; ++group) {
            const std::size_t index = scale_index(group, current_id, head, state_capacity);
            actual_current_k_scales.push_back(actual_k_scales[index]);
            actual_current_v_scales.push_back(actual_v_scales[index]);
            expected_current_k_scales.push_back(expected_state.k_scales[index]);
            expected_current_v_scales.push_back(expected_state.v_scales[index]);
            actual_k_scales[index] = initial_state.k_scales[index];
            actual_v_scales[index] = initial_state.v_scales[index];
        }
    }
    failures += verify_exact("Qwen4 real QSA independently encoded current K codes",
                             actual_current_k_codes, expected_current_k_codes);
    failures += verify_exact("Qwen4 real QSA independently encoded current V codes",
                             actual_current_v_codes, expected_current_v_codes);
    failures += verify_exact("Qwen4 real QSA independently encoded current K scales",
                             actual_current_k_scales, expected_current_k_scales);
    failures += verify_exact("Qwen4 real QSA independently encoded current V scales",
                             actual_current_v_scales, expected_current_v_scales);
    failures += verify_exact("Qwen4 real QSA untouched K code plane", actual_k_codes,
                             initial_state.k_codes);
    failures += verify_exact("Qwen4 real QSA untouched V code plane", actual_v_codes,
                             initial_state.v_codes);
    failures += verify_exact("Qwen4 real QSA untouched K scale plane", actual_k_scales,
                             initial_state.k_scales);
    failures += verify_exact("Qwen4 real QSA untouched V scale plane", actual_v_scales,
                             initial_state.v_scales);

    std::vector<double> actual_raw_key(kIndex);
    std::vector<double> raw_key_bound(kIndex);
    for (std::int32_t d = 0; d < kIndex; ++d) {
        const std::size_t index = static_cast<std::size_t>(current_id) * kIndex + d;
        actual_raw_key[d] = bf16_to_f32(actual_raw_key_bits[index]);
        actual_raw_key_bits[index] = initial_state.raw_index_keys[index];
        raw_key_bound[d] = index_key.production_to_ideal_bound[d] +
                           std::abs(index_key.represented[d] - index_key.ideal[d]);
    }
    failures += verify_bounded("qwen4_real_qsa_raw_index_key", actual_raw_key,
                               index_key.represented, raw_key_bound);
    failures += verify_exact("Qwen4 real QSA untouched raw-index-key plane",
                             actual_raw_key_bits, initial_state.raw_index_keys);
    failures += verify_exact(
        "Qwen4 real QSA stored nonzero three-axis position",
        std::vector<std::int32_t>(actual_positions.begin() + 3 * current_id,
                                  actual_positions.begin() + 3 * current_id + 3),
        position_values);
    std::fill(actual_positions.begin() + 3 * current_id,
              actual_positions.begin() + 3 * current_id + 3, 0);
    failures += verify_exact("Qwen4 real QSA untouched position plane", actual_positions,
                             initial_state.positions);

    std::vector<double> decoded_current_k(static_cast<std::size_t>(kKvHeads) * kHeadDim);
    std::vector<double> decoded_current_v(static_cast<std::size_t>(kKvHeads) * kHeadDim);
    std::vector<double> k_bound(decoded_current_k.size());
    std::vector<double> v_bound(decoded_current_v.size());
    for (std::int32_t head = 0; head < kKvHeads; ++head) {
        for (std::int32_t group = 0; group < kHeadDim / 16; ++group) {
            double maximum_staged_k = 0.0;
            double maximum_staged_v = 0.0;
            for (std::int32_t lane = 0; lane < 16; ++lane) {
                const std::size_t index = static_cast<std::size_t>(head) * kHeadDim +
                                          group * 16 + lane;
                maximum_staged_k =
                    std::max(maximum_staged_k,
                             std::abs(key.represented[index]) +
                                 key.production_to_represented_bound[index]);
                const double value_stage_error =
                    raw_value.production_to_ideal_bound[index] +
                    std::abs(raw_value.represented[index] - raw_value.ideal[index]);
                maximum_staged_v =
                    std::max(maximum_staged_v,
                             std::abs(raw_value.represented[index]) + value_stage_error);
            }
            const double k_codec_bound = (17.0 / 96.0) * maximum_staged_k;
            const double v_codec_bound = (17.0 / 96.0) * maximum_staged_v;
            for (std::int32_t lane = 0; lane < 16; ++lane) {
                const std::size_t index = static_cast<std::size_t>(head) * kHeadDim +
                                          group * 16 + lane;
                const std::int32_t d = group * 16 + lane;
                decoded_current_k[index] = decode_cache(
                    expected_state.k_codes, expected_state.k_scales, d, current_id, head,
                    state_capacity);
                decoded_current_v[index] = decode_cache(
                    expected_state.v_codes, expected_state.v_scales, d, current_id, head,
                    state_capacity);
                k_bound[index] = key.production_to_represented_bound[index] + k_codec_bound;
                v_bound[index] = raw_value.production_to_ideal_bound[index] +
                                 std::abs(raw_value.represented[index] - raw_value.ideal[index]) +
                                 v_codec_bound;
            }
        }
    }
    failures += verify_bounded("qwen4_real_qsa_nvfp4_k", decoded_current_k,
                               key.represented, k_bound);
    failures += verify_bounded("qwen4_real_qsa_nvfp4_v", decoded_current_v,
                               raw_value.represented, v_bound);

    // Complete independent formula over the independently constructed expected cache. The
    // prefix deliberately creates nonuniform logits and the current real projected key/value is
    // included as the ninth item; production-mutated cache storage is never an oracle input.
    std::vector<double> gated(kOutputColumns);
    double maximum_logit_spread = 0.0;
    double maximum_current_probability = 0.0;
    for (std::int32_t head = 0; head < kQueryHeads; ++head) {
        const std::int32_t kv_head = head / (kQueryHeads / kKvHeads);
        std::array<double, selected_prefix.size()> logits{};
        double maximum = -INFINITY;
        double minimum = INFINITY;
        for (std::size_t item = 0; item < selected_prefix.size(); ++item) {
            const std::int32_t id = selected_prefix[item];
            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                logits[item] +=
                    query.represented[static_cast<std::size_t>(head) * kHeadDim + d] *
                    decode_cache(expected_state.k_codes, expected_state.k_scales, d, id,
                                 kv_head, state_capacity);
            }
            logits[item] /= std::sqrt(static_cast<double>(kHeadDim));
            maximum = std::max(maximum, logits[item]);
            minimum = std::min(minimum, logits[item]);
        }
        maximum_logit_spread = std::max(maximum_logit_spread, maximum - minimum);
        std::array<double, selected_prefix.size()> probabilities{};
        double denominator = 0.0;
        for (std::size_t item = 0; item < selected_prefix.size(); ++item) {
            probabilities[item] = std::exp(logits[item] - maximum);
            denominator += probabilities[item];
        }
        for (double& probability : probabilities) { probability /= denominator; }
        maximum_current_probability =
            std::max(maximum_current_probability, probabilities.back());
        for (std::int32_t d = 0; d < kHeadDim; ++d) {
            double attention = 0.0;
            for (std::size_t item = 0; item < selected_prefix.size(); ++item) {
                attention += probabilities[item] *
                             decode_cache(expected_state.v_codes, expected_state.v_scales, d,
                                          selected_prefix[item], kv_head, state_capacity);
            }
            const double represented_attention = represented_bf16(attention);
            const std::size_t raw_gate_index =
                static_cast<std::size_t>(head) * 2 * kHeadDim + kHeadDim + d;
            gated[static_cast<std::size_t>(head) * kHeadDim + d] = represented_bf16(
                represented_attention * sigmoid(raw_query_gate.represented[raw_gate_index]));
        }
    }
    if (!(maximum_logit_spread > 0.25) || !(maximum_current_probability > 1.0e-3) ||
        !(maximum_current_probability < 1.0) || !std::isfinite(maximum_logit_spread)) {
        throw std::logic_error(
            "Qwen4 real QSA nonuniform core-attention witness is ineffective: spread=" +
            std::to_string(maximum_logit_spread) +
            " current_probability=" + std::to_string(maximum_current_probability));
    }
    if (error_stats_enabled()) {
        std::cout << "OP_ERROR_STATS kind=witness selector_block0=" << block_zero_score
                  << " selector_block1=" << block_one_score
                  << " max_logit_spread=" << maximum_logit_spread
                  << " max_current_probability=" << maximum_current_probability
                  << " case=Qwen4_real_QSA_nonuniform_prefix\n";
    }
    const Projection output = q5_project(output_bytes, kHidden, kOutputColumns, gated);
    std::vector<double> expected_output(kHidden);
    std::transform(output.ideal.begin(), output.ideal.end(), expected_output.begin(),
                   [](double value) { return represented_bf16(value); });
    failures += verify_pointwise("Qwen4 real layer-3 QSA complete FP64 oracle",
                                 from_device_bf16(device_output.data(), kHidden), expected_output,
                                 kCompleteCriterion);

    failures += device_selected.verify_guards("Qwen4 real QSA selected ids");
    failures += device_count.verify_guards("Qwen4 real QSA selected count");
    failures += device_output.verify_guards("Qwen4 real QSA output");
    failures += workspace.verify_guards("Qwen4 real QSA workspace");
    failures += qsa_x.verify_guards("Qwen4 real QSA represented input");
    failures += run_accumulated_qsa_cell(model, device);
    std::cout << (failures == 0 ? "OK" : "FAIL") << " qwen4_real_qsa_oracle_cell\n";
    return failures;
}
