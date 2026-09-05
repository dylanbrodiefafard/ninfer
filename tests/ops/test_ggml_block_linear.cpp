#include "ninfer/ops/ggml_block_linear.h"
#include "ops/ggml_iq_oracle.h"
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
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

struct FormatSpec {
    QType qtype;
    const char* name;
    std::int32_t block_values;
    std::int32_t block_bytes;
    int arithmetic_steps_per_value;
    std::int32_t real_n;
    std::int32_t real_k;
};

constexpr std::array<FormatSpec, 7> kFormats = {{
    {QType::GGML_Q8_0, "Q8_0", 32, 34, 3, 320, 10240},
    {QType::GGML_Q4_K, "Q4_K", 256, 144, 5, 248320, 2560},
    {QType::GGML_Q5_K, "Q5_K", 256, 176, 5, 640, 2560},
    {QType::GGML_Q6_K, "Q6_K", 256, 210, 3, 640, 2560},
    {QType::GGML_IQ1_S, "IQ1_S", 256, 50, 4, 640, 2560},
    {QType::GGML_IQ2_XXS, "IQ2_XXS", 256, 66, 5, 640, 2560},
    {QType::GGML_IQ4_NL, "IQ4_NL", 32, 18, 3, 2560, 640},
}};

std::uint16_t read_u16(const std::uint8_t* bytes) {
    return static_cast<std::uint16_t>(bytes[0]) |
           (static_cast<std::uint16_t>(bytes[1]) << 8U);
}

std::uint32_t read_u32(const std::uint8_t* bytes) {
    return static_cast<std::uint32_t>(bytes[0]) |
           (static_cast<std::uint32_t>(bytes[1]) << 8U) |
           (static_cast<std::uint32_t>(bytes[2]) << 16U) |
           (static_cast<std::uint32_t>(bytes[3]) << 24U);
}

double half_to_double(std::uint16_t word) {
    const double sign = (word & 0x8000U) != 0 ? -1.0 : 1.0;
    const unsigned exponent = (word >> 10U) & 0x1fU;
    const unsigned fraction = word & 0x3ffU;
    if (exponent == 0) { return sign * std::ldexp(static_cast<double>(fraction), -24); }
    if (exponent == 31) {
        return fraction == 0 ? sign * std::numeric_limits<double>::infinity()
                             : std::numeric_limits<double>::quiet_NaN();
    }
    return sign * std::ldexp(static_cast<double>(1024U + fraction),
                             static_cast<int>(exponent) - 25);
}

int signed_byte(std::uint8_t word) {
    return word < 128U ? static_cast<int>(word) : static_cast<int>(word) - 256;
}

std::pair<int, int> oracle_scale_min(const std::uint8_t* table, int group) {
    if (group < 4) { return {table[group] & 63, table[group + 4] & 63}; }
    const int scale = (table[group + 4] & 15) + 16 * (table[group - 4] >> 6U);
    const int minimum = (table[group + 4] >> 4U) + 16 * (table[group] >> 6U);
    return {scale, minimum};
}

std::vector<double> oracle_decode_block(const FormatSpec& format,
                                        const std::uint8_t* block) {
    std::vector<double> values(static_cast<std::size_t>(format.block_values));
    if (format.qtype == QType::GGML_Q8_0) {
        const double d = half_to_double(read_u16(block));
        for (int i = 0; i < 32; ++i) { values[i] = d * signed_byte(block[2 + i]); }
        return values;
    }
    if (format.qtype == QType::GGML_Q4_K || format.qtype == QType::GGML_Q5_K) {
        const double d    = half_to_double(read_u16(block));
        const double dmin = half_to_double(read_u16(block + 2));
        const int low_offset  = format.qtype == QType::GGML_Q4_K ? 16 : 48;
        const int high_offset = 16;
        for (int group = 0; group < 8; ++group) {
            const auto [scale, minimum] = oracle_scale_min(block + 4, group);
            for (int lane = 0; lane < 32; ++lane) {
                const int packed = block[low_offset + 32 * (group / 2) + lane];
                int code = group % 2 == 0 ? packed & 15 : packed >> 4;
                if (format.qtype == QType::GGML_Q5_K) {
                    code += 16 * ((block[high_offset + lane] >> group) & 1U);
                }
                values[32 * group + lane] = d * scale * code - dmin * minimum;
            }
        }
        return values;
    }
    if (format.qtype == QType::GGML_Q6_K) {
        const double d = half_to_double(read_u16(block + 208));
        for (int half = 0; half < 2; ++half) {
            for (int group = 0; group < 4; ++group) {
                for (int lane = 0; lane < 32; ++lane) {
                    const int index = 128 * half + 32 * group + lane;
                    const int low_word = block[64 * half + lane + 32 * (group % 2)];
                    const int low = group < 2 ? low_word & 15 : low_word >> 4;
                    const int high = (block[128 + 32 * half + lane] >> (2 * group)) & 3;
                    const int code = low + 16 * high - 32;
                    values[index] = d * signed_byte(block[192 + index / 16]) * code;
                }
            }
        }
        return values;
    }
    if (format.qtype == QType::GGML_IQ1_S) {
        const double d = half_to_double(read_u16(block));
        for (int group = 0; group < 8; ++group) {
            const std::uint16_t control = read_u16(block + 34 + 2 * group);
            const double delta = (control & 0x8000U) != 0 ? -0.125 : 0.125;
            const int multiplier = 2 * ((control >> 12U) & 7U) + 1;
            for (int lane = 0; lane < 4; ++lane) {
                const int grid = block[2 + 4 * group + lane] +
                                 256 * ((control >> (3 * lane)) & 7U);
                const auto& witness = iq_oracle::iq1_grid(grid);
                for (int item = 0; item < 8; ++item) {
                    values[32 * group + 8 * lane + item] =
                        d * multiplier * (static_cast<double>(witness[item]) + delta);
                }
            }
        }
        return values;
    }
    if (format.qtype == QType::GGML_IQ2_XXS) {
        const double d = half_to_double(read_u16(block));
        for (int group = 0; group < 8; ++group) {
            const std::uint32_t grids = read_u32(block + 2 + 8 * group);
            const std::uint32_t signs_scales = read_u32(block + 6 + 8 * group);
            const double db = d * (0.5 + static_cast<double>(signs_scales >> 28U)) / 4.0;
            for (int lane = 0; lane < 4; ++lane) {
                const int grid = (grids >> (8 * lane)) & 255U;
                int sign_bits = (signs_scales >> (7 * lane)) & 127U;
                sign_bits |= (__builtin_popcount(static_cast<unsigned>(sign_bits)) & 1) << 7;
                const auto& witness = iq_oracle::iq2_grid(grid);
                for (int item = 0; item < 8; ++item) {
                    const double magnitude = witness[item];
                    const int sign = (sign_bits & (1 << item)) != 0 ? -1 : 1;
                    values[32 * group + 8 * lane + item] = db * magnitude * sign;
                }
            }
        }
        return values;
    }

    const double d = half_to_double(read_u16(block));
    for (int lane = 0; lane < 16; ++lane) {
        const std::uint8_t packed = block[2 + lane];
        values[lane] = d * iq_oracle::kIq4Nl[packed & 15U];
        values[16 + lane] = d * iq_oracle::kIq4Nl[packed >> 4U];
    }
    return values;
}

void write_u16(std::uint8_t* bytes, std::uint16_t value) {
    bytes[0] = static_cast<std::uint8_t>(value & 0xffU);
    bytes[1] = static_cast<std::uint8_t>(value >> 8U);
}

std::vector<std::uint8_t> make_row(const FormatSpec& format, std::int32_t k,
                                   std::uint32_t seed) {
    const std::size_t blocks = static_cast<std::size_t>(k / format.block_values);
    std::vector<std::uint8_t> row(blocks * static_cast<std::size_t>(format.block_bytes));
    std::mt19937 generator(seed);
    std::uniform_int_distribution<int> byte_distribution(0, 255);
    for (auto& byte : row) { byte = static_cast<std::uint8_t>(byte_distribution(generator)); }
    for (std::size_t block_index = 0; block_index < blocks; ++block_index) {
        auto* block = row.data() + block_index * static_cast<std::size_t>(format.block_bytes);
        const std::uint16_t d = (block_index & 1U) == 0 ? 0x2000U : 0x9c00U;
        if (format.qtype == QType::GGML_Q6_K) {
            write_u16(block + 208, d);
        } else {
            write_u16(block, d);
        }
        if (format.qtype == QType::GGML_Q4_K || format.qtype == QType::GGML_Q5_K) {
            write_u16(block + 2, (block_index & 1U) == 0 ? 0x1800U : 0x9800U);
        } else if (format.qtype == QType::GGML_IQ1_S) {
            // Every block independently exercises low indices 0/1/2 and the maximum 2047.
            // Restricting the randomized fixture to test-owned anchors keeps its oracle wholly
            // independent of the production 2048-entry table.
            constexpr std::array<int, 4> grids = {0, 1, 2, 2047};
            for (int group = 0; group < 8; ++group) {
                std::uint16_t control = 0;
                for (int lane = 0; lane < 4; ++lane) {
                    const int grid = grids[(seed + block_index + group + lane) & 3U];
                    block[2 + 4 * group + lane] = static_cast<std::uint8_t>(grid);
                    control |= static_cast<std::uint16_t>(grid >> 8) << (3 * lane);
                }
                const std::uint16_t multiplier =
                    static_cast<std::uint16_t>((seed + block_index + group) & 7U);
                const std::uint16_t delta =
                    ((seed + block_index + group) & 1U) != 0 ? 0x8000U : 0U;
                write_u16(block + 34 + 2 * group,
                          static_cast<std::uint16_t>(control | (multiplier << 12U) | delta));
            }
        } else if (format.qtype == QType::GGML_IQ2_XXS) {
            constexpr std::array<int, 4> grids = {0, 1, 2, 255};
            for (int group = 0; group < 8; ++group) {
                std::uint32_t signs = 0;
                for (int lane = 0; lane < 4; ++lane) {
                    block[2 + 8 * group + lane] = static_cast<std::uint8_t>(
                        grids[(seed + block_index + group + lane) & 3U]);
                    const std::uint32_t sign =
                        (seed + 19U * block_index + 11U * group + 29U * lane) & 127U;
                    signs |= sign << (7 * lane);
                }
                const std::uint32_t scale =
                    static_cast<std::uint32_t>((seed + block_index + group) & 15U) << 28U;
                const std::uint32_t control = signs | scale;
                for (int byte = 0; byte < 4; ++byte) {
                    block[6 + 8 * group + byte] =
                        static_cast<std::uint8_t>(control >> (8 * byte));
                }
            }
        } else if (format.qtype == QType::GGML_IQ4_NL) {
            // Each half covers all 16 codes, with seed-dependent permutations across rows.
            for (int lane = 0; lane < 16; ++lane) {
                const int low = static_cast<int>((seed + block_index + 5U * lane) & 15U);
                const int high = static_cast<int>((seed + 3U * block_index + 7U * lane) & 15U);
                block[2 + lane] = static_cast<std::uint8_t>((high << 4) | low);
            }
        }
    }
    return row;
}

std::vector<float> make_input(std::int32_t k, std::int32_t tokens) {
    std::vector<float> x(static_cast<std::size_t>(k) * static_cast<std::size_t>(tokens));
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t i = 0; i < k; ++i) {
            x[static_cast<std::size_t>(token) * k + i] =
                static_cast<float>((i * 17 + token * 23 + (i * token) % 29) % 61 - 30) / 64.0F;
        }
    }
    round_to_bf16(x);
    return x;
}

struct OracleDot {
    double value = 0.0;
    double absolute_sum = 0.0;
};

OracleDot oracle_dot(const FormatSpec& format, const std::vector<std::uint8_t>& row,
                     const std::vector<float>& x, std::int32_t token, std::int32_t k) {
    OracleDot result;
    const std::size_t blocks = row.size() / static_cast<std::size_t>(format.block_bytes);
    for (std::size_t block = 0; block < blocks; ++block) {
        const auto values = oracle_decode_block(
            format, row.data() + block * static_cast<std::size_t>(format.block_bytes));
        for (std::size_t item = 0; item < values.size(); ++item) {
            const std::size_t column = block * values.size() + item;
            const double product =
                values[item] * static_cast<double>(x[static_cast<std::size_t>(token) * k + column]);
            result.value += product;
            result.absolute_sum += std::abs(product);
        }
    }
    return result;
}

double error_bound(const FormatSpec& format, std::int32_t k, const OracleDot& oracle) {
    constexpr double float_unit_roundoff = 0x1p-24;
    const double operations =
        static_cast<double>(format.arithmetic_steps_per_value) * static_cast<double>(k) + 8.0;
    const double gamma = operations * float_unit_roundoff /
                         (1.0 - operations * float_unit_roundoff);
    const double accumulation = gamma * oracle.absolute_sum;
    const double bf16_rounding =
        std::max((std::abs(oracle.value) + accumulation) / 255.0, 0x1p-134);
    return accumulation + bf16_rounding;
}

Weight make_weight(void* payload, std::size_t bytes, const FormatSpec& format, std::int32_t n,
                   std::int32_t k) {
    Weight weight{};
    weight.payload         = payload;
    weight.payload_bytes   = bytes;
    weight.qdata           = payload;
    weight.qtype           = format.qtype;
    weight.group_size      = static_cast<std::uint32_t>(format.block_values);
    weight.group           = format.block_values;
    weight.layout          = QuantLayout::GgmlBlockRow;
    weight.n               = n;
    weight.k               = k;
    weight.ndim            = 2;
    weight.shape[0]        = n;
    weight.shape[1]        = k;
    weight.padded_shape[0] = n;
    weight.padded_shape[1] = k;
    return weight;
}

int run_case(const FormatSpec& format, std::int32_t n, std::int32_t k,
             std::int32_t tokens, bool repeat_row, cudaStream_t stream, const char* profile) {
    const auto first_row = make_row(format, k, 0x51a6U + static_cast<unsigned>(format.qtype));
    std::vector<std::uint8_t> payload(first_row.size() * static_cast<std::size_t>(n));
    for (std::int32_t row = 0; row < n; ++row) {
        auto generated = repeat_row
                             ? first_row
                             : make_row(format, k, 0x51a6U + static_cast<unsigned>(format.qtype) +
                                                     static_cast<unsigned>(row) * 17U);
        std::copy(generated.begin(), generated.end(),
                  payload.begin() + static_cast<std::size_t>(row) * first_row.size());
    }
    const auto x = make_input(k, tokens);
    std::vector<OracleDot> first_oracles(static_cast<std::size_t>(tokens));
    for (std::int32_t token = 0; token < tokens; ++token) {
        first_oracles[static_cast<std::size_t>(token)] =
            oracle_dot(format, first_row, x, token, k);
    }

    DeviceBuffer device_payload = to_device(payload);
    DeviceBuffer device_x       = to_device_bf16(x);
    const std::size_t output_count =
        static_cast<std::size_t>(n) * static_cast<std::size_t>(tokens);
    GuardedDeviceBuffer device_out(output_count * sizeof(std::uint16_t));
    device_out.fill(0xcd);
    // GuardedDeviceBuffer initializes on the default stream. Finish test setup before launching
    // the Op on an intentionally nonblocking stream.
    cuda_synchronize();
    Tensor x_tensor(device_x.p, DType::BF16, {k, tokens});
    Tensor out_tensor(device_out.data(), DType::BF16, {n, tokens});
    Weight weight = make_weight(device_payload.p, payload.size(), format, n, k);
    ops::ggml_block_linear(x_tensor, weight, out_tensor, stream);
    cuda_synchronize(stream);

    const auto actual = from_device_bf16(device_out.data(), output_count);
    double maximum_ratio = 0.0;
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t row = 0; row < n; ++row) {
            const OracleDot expected =
                repeat_row
                    ? first_oracles[static_cast<std::size_t>(token)]
                    : oracle_dot(
                          format,
                          std::vector<std::uint8_t>(
                              payload.begin() + static_cast<std::size_t>(row) * first_row.size(),
                              payload.begin() +
                                  static_cast<std::size_t>(row + 1) * first_row.size()),
                          x, token, k);
            const std::size_t index =
                static_cast<std::size_t>(token) * static_cast<std::size_t>(n) + row;
            const double limit = error_bound(format, k, expected);
            const double error = std::abs(actual[index] - expected.value);
            maximum_ratio = std::max(maximum_ratio, error / limit);
            if (!std::isfinite(actual[index]) || error > limit) {
                std::cerr << format.name << ' ' << profile << " row " << row << " token "
                          << token << " error=" << error << " bound=" << limit
                          << " reference=" << expected.value << '\n';
                return 1;
            }
        }
    }
    if (error_stats_enabled()) {
        std::cout << "OP_ERROR_STATS kind=derived_bound format=" << format.name
                  << " profile=" << profile << " max_bound_ratio=" << maximum_ratio << '\n';
    }
    return device_out.verify_guards(std::string(format.name) + " " + profile);
}

int run_qwen4_replay_shape_case(const FormatSpec& format, std::int32_t n, std::int32_t k,
                                std::int32_t tokens, cudaStream_t stream) {
    constexpr std::size_t pattern_count = 4;
    std::array<std::vector<std::uint8_t>, pattern_count> rows;
    for (std::size_t pattern = 0; pattern < pattern_count; ++pattern) {
        rows[pattern] = make_row(
            format, k, 0x45a7U + static_cast<std::uint32_t>(format.qtype) * 257U +
                           static_cast<std::uint32_t>(pattern) * 131U);
    }
    const std::size_t row_bytes = rows[0].size();
    std::vector<std::uint8_t> payload(row_bytes * static_cast<std::size_t>(n));
    for (std::int32_t row = 0; row < n; ++row) {
        const auto& source = rows[static_cast<std::size_t>(row) % pattern_count];
        std::copy(source.begin(), source.end(),
                  payload.begin() + static_cast<std::size_t>(row) * row_bytes);
    }
    const auto x = make_input(k, tokens);
    std::vector<OracleDot> oracle(pattern_count * static_cast<std::size_t>(tokens));
    for (std::size_t pattern = 0; pattern < pattern_count; ++pattern) {
        for (std::int32_t token = 0; token < tokens; ++token) {
            oracle[pattern * static_cast<std::size_t>(tokens) + token] =
                oracle_dot(format, rows[pattern], x, token, k);
        }
    }

    DeviceBuffer device_payload = to_device(payload);
    DeviceBuffer device_x       = to_device_bf16(x);
    const std::size_t output_count =
        static_cast<std::size_t>(n) * static_cast<std::size_t>(tokens);
    GuardedDeviceBuffer device_out(output_count * sizeof(std::uint16_t));
    device_out.fill(0xcd);
    cuda_synchronize();
    Tensor x_tensor(device_x.p, DType::BF16, {k, tokens});
    Tensor out_tensor(device_out.data(), DType::BF16, {n, tokens});
    Weight weight = make_weight(device_payload.p, payload.size(), format, n, k);
    ops::ggml_block_linear(x_tensor, weight, out_tensor, stream);
    cuda_synchronize(stream);

    const auto actual = from_device_bf16(device_out.data(), output_count);
    double maximum_ratio = 0.0;
    int failures          = 0;
    for (std::int32_t token = 0; token < tokens && failures == 0; ++token) {
        for (std::int32_t row = 0; row < n; ++row) {
            const OracleDot& expected =
                oracle[(static_cast<std::size_t>(row) % pattern_count) *
                           static_cast<std::size_t>(tokens) +
                       static_cast<std::size_t>(token)];
            const std::size_t index =
                static_cast<std::size_t>(token) * static_cast<std::size_t>(n) + row;
            const double limit = error_bound(format, k, expected);
            const double error = std::abs(actual[index] - expected.value);
            maximum_ratio      = std::max(maximum_ratio, error / limit);
            if (!std::isfinite(actual[index]) || error > limit) {
                std::cerr << format.name << " qwen4-replay-boundary-t" << tokens << " row "
                          << row << " token " << token << " error=" << error
                          << " bound=" << limit << " reference=" << expected.value << '\n';
                failures = 1;
                break;
            }
        }
    }
    if (error_stats_enabled()) {
        std::cout << "OP_ERROR_STATS kind=derived_bound format=" << format.name
                  << " profile=qwen4-replay-boundary-t" << tokens
                  << " max_bound_ratio=" << maximum_ratio << '\n';
    }
    const std::string label = std::string(format.name) + " qwen4 replay";
    failures += device_out.verify_guards(label + " boundary");
    const std::string weights_label = label + " weights unchanged";
    failures += verify_exact(weights_label.c_str(),
                             from_device<std::uint8_t>(device_payload, payload.size()), payload);
    std::vector<std::uint16_t> represented_x(x.size());
    std::transform(x.begin(), x.end(), represented_x.begin(), f32_to_bf16);
    const std::string input_label = label + " input unchanged";
    failures += verify_exact(input_label.c_str(),
                             from_device<std::uint16_t>(device_x, represented_x.size()),
                             represented_x);
    return failures;
}

template <class Function>
int expect_invalid(Function&& function, const char* label) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        return 0;
    }
    std::cerr << label << " was accepted\n";
    return 1;
}

int validation_cases() {
    const FormatSpec& format = kFormats.front();
    constexpr std::int32_t n = 3;
    constexpr std::int32_t k = 32;
    const auto payload = make_row(format, k, 7);
    std::vector<std::uint8_t> rows(payload.size() * n);
    for (std::int32_t row = 0; row < n; ++row) {
        std::copy(payload.begin(), payload.end(), rows.begin() + row * payload.size());
    }
    auto x = make_input(k, 2);
    DeviceBuffer device_payload = to_device(rows);
    DeviceBuffer device_x       = to_device_bf16(x);
    DeviceBuffer device_out(static_cast<std::size_t>(n) * 2 * sizeof(std::uint16_t));
    Tensor x_tensor(device_x.p, DType::BF16, {k, 2});
    Tensor out_tensor(device_out.p, DType::BF16, {n, 2});
    const Weight valid = make_weight(device_payload.p, rows.size(), format, n, k);
    int failures = 0;

    Weight wrong = valid;
    wrong.layout = QuantLayout::RowSplit;
    failures += expect_invalid(
        [&] { ops::ggml_block_linear(x_tensor, wrong, out_tensor, nullptr); }, "wrong layout");
    wrong = valid;
    --wrong.payload_bytes;
    failures += expect_invalid(
        [&] { ops::ggml_block_linear(x_tensor, wrong, out_tensor, nullptr); }, "short payload");
    wrong = valid;
    wrong.ndim = 3;
    failures += expect_invalid(
        [&] { ops::ggml_block_linear(x_tensor, wrong, out_tensor, nullptr); }, "rank-three bank");
    Tensor aliased_out(device_x.p, DType::BF16, {n, 2});
    failures += expect_invalid(
        [&] { ops::ggml_block_linear(x_tensor, valid, aliased_out, nullptr); }, "output alias");
    Tensor wrong_tokens_out(device_out.p, DType::BF16, {n});
    failures += expect_invalid(
        [&] { ops::ggml_block_linear(x_tensor, valid, wrong_tokens_out, nullptr); },
        "output token mismatch");
    Tensor too_wide_x(device_x.p, DType::BF16, {k, 4097});
    failures += expect_invalid(
        [&] { ops::ggml_block_linear(too_wide_x, valid, out_tensor, nullptr); },
        "T above verifier ceiling");
    Tensor empty_x = x_tensor;
    empty_x.ne[1] = 0;
    failures += expect_invalid(
        [&] { ops::ggml_block_linear(empty_x, valid, out_tensor, nullptr); },
        "zero-token input");
    Weight unsupported = valid;
    unsupported.qtype  = QType::NVFP4;
    failures += expect_invalid(
        [&] { ops::ggml_block_linear(x_tensor, unsupported, out_tensor, nullptr); },
        "unsupported qtype");
    return failures;
}

} // namespace

int main() {
    try {
        if (const int unavailable = require_cuda(); unavailable != 0) { return unavailable; }
        cudaStream_t stream = nullptr;
        cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
        int failures = validation_cases();
        for (const auto& format : kFormats) {
            failures += run_case(format, 3, 2 * format.block_values, 1, false, stream,
                                 "representative-t1");
            failures += run_case(format, 3, 2 * format.block_values, 16, false, stream,
                                 "aggregate-t16");
            failures += run_case(format, 3, 2 * format.block_values, 17, false, stream,
                                 "aggregate-tail-t17");
            failures +=
                run_case(format, format.real_n, format.real_k, 1, true, stream, "real-shape-t1");
        }
        failures += run_case(kFormats[2], 3, 6144, 1, false, stream,
                             "q5-output-projection-k");
        failures += run_case(kFormats[2], 3, 512, 2, false, stream, "q5-aggregate-t2");
        failures += run_case(kFormats[2], 3, 512, 15, false, stream, "q5-aggregate-t15");
        failures += run_case(kFormats[2], 3, 512, 128, false, stream,
                             "aggregate-t128");
        failures += run_qwen4_replay_shape_case(kFormats[2], 10240, 2560, 511, stream);
        failures += run_qwen4_replay_shape_case(kFormats[2], 10240, 2560, 512, stream);
        failures += run_qwen4_replay_shape_case(kFormats[2], 10240, 2560, 513, stream);
        failures += run_qwen4_replay_shape_case(kFormats[3], 2560, 6144, 511, stream);
        failures += run_qwen4_replay_shape_case(kFormats[3], 2560, 6144, 512, stream);
        failures += run_qwen4_replay_shape_case(kFormats[3], 2560, 6144, 513, stream);
        failures += run_case(kFormats[0], 1, 32, 4096, true, stream,
                             "aggregate-t4096");
        cuda_check(cudaStreamDestroy(stream), "cudaStreamDestroy");
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
