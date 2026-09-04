#include "ninfer/ops/ggml_block_linear.h"
#include "ops/ggml_block_linear/ggml_codebook_data.h"
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
                const std::uint16_t packed = ops::detail::kIq1SPackedGrid[grid];
                for (int item = 0; item < 8; ++item) {
                    const int trit = (packed >> (2 * item)) & 3U;
                    values[32 * group + 8 * lane + item] =
                        d * multiplier * (static_cast<double>(trit - 1) + delta);
                }
            }
        }
        return values;
    }
    if (format.qtype == QType::GGML_IQ2_XXS) {
        const double d = half_to_double(read_u16(block));
        constexpr std::array<int, 3> magnitudes = {8, 25, 43};
        for (int group = 0; group < 8; ++group) {
            const std::uint32_t grids = read_u32(block + 2 + 8 * group);
            const std::uint32_t signs_scales = read_u32(block + 6 + 8 * group);
            const double db = d * (0.5 + static_cast<double>(signs_scales >> 28U)) / 4.0;
            for (int lane = 0; lane < 4; ++lane) {
                const int grid = (grids >> (8 * lane)) & 255U;
                int sign_bits = (signs_scales >> (7 * lane)) & 127U;
                sign_bits |= (__builtin_popcount(static_cast<unsigned>(sign_bits)) & 1) << 7;
                const std::uint16_t packed = ops::detail::kIq2XxsPackedGrid[grid];
                for (int item = 0; item < 8; ++item) {
                    const int digit = (packed >> (2 * item)) & 3U;
                    const double magnitude = magnitudes[static_cast<std::size_t>(digit)];
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
        values[lane] = d * ops::detail::kIq4NlGrid[packed & 15U];
        values[16 + lane] = d * ops::detail::kIq4NlGrid[packed >> 4U];
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
        }
    }
    return row;
}

std::vector<float> make_input(std::int32_t k) {
    std::vector<float> x(static_cast<std::size_t>(k));
    for (std::int32_t i = 0; i < k; ++i) {
        x[static_cast<std::size_t>(i)] = static_cast<float>((i * 17) % 61 - 30) / 64.0F;
    }
    round_to_bf16(x);
    return x;
}

struct OracleDot {
    double value = 0.0;
    double absolute_sum = 0.0;
};

OracleDot oracle_dot(const FormatSpec& format, const std::vector<std::uint8_t>& row,
                     const std::vector<float>& x) {
    OracleDot result;
    const std::size_t blocks = row.size() / static_cast<std::size_t>(format.block_bytes);
    for (std::size_t block = 0; block < blocks; ++block) {
        const auto values = oracle_decode_block(
            format, row.data() + block * static_cast<std::size_t>(format.block_bytes));
        for (std::size_t item = 0; item < values.size(); ++item) {
            const double product = values[item] *
                                   static_cast<double>(x[block * values.size() + item]);
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

int run_case(const FormatSpec& format, std::int32_t n, std::int32_t k, bool repeat_row,
             cudaStream_t stream, const char* profile) {
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
    const auto x = make_input(k);
    const OracleDot first_oracle = oracle_dot(format, first_row, x);

    DeviceBuffer device_payload = to_device(payload);
    DeviceBuffer device_x       = to_device_bf16(x);
    GuardedDeviceBuffer device_out(static_cast<std::size_t>(n) * sizeof(std::uint16_t));
    device_out.fill(0xcd);
    // GuardedDeviceBuffer initializes on the default stream. Finish test setup before launching
    // the Op on an intentionally nonblocking stream.
    cuda_synchronize();
    Tensor x_tensor(device_x.p, DType::BF16, {k});
    Tensor out_tensor(device_out.data(), DType::BF16, {n});
    Weight weight = make_weight(device_payload.p, payload.size(), format, n, k);
    ops::ggml_block_linear(x_tensor, weight, out_tensor, stream);
    cuda_synchronize(stream);

    const auto actual = from_device_bf16(device_out.data(), static_cast<std::size_t>(n));
    double maximum_ratio = 0.0;
    for (std::int32_t row = 0; row < n; ++row) {
        const OracleDot expected =
            repeat_row
                ? first_oracle
                : oracle_dot(format,
                             std::vector<std::uint8_t>(
                                 payload.begin() + static_cast<std::size_t>(row) * first_row.size(),
                                 payload.begin() +
                                     static_cast<std::size_t>(row + 1) * first_row.size()),
                             x);
        const double limit = error_bound(format, k, expected);
        const double error = std::abs(actual[static_cast<std::size_t>(row)] - expected.value);
        maximum_ratio = std::max(maximum_ratio, error / limit);
        if (!std::isfinite(actual[static_cast<std::size_t>(row)]) || error > limit) {
            std::cerr << format.name << ' ' << profile << " row " << row << " error=" << error
                      << " bound=" << limit << " reference=" << expected.value << '\n';
            return 1;
        }
    }
    if (error_stats_enabled()) {
        std::cout << "OP_ERROR_STATS kind=derived_bound format=" << format.name
                  << " profile=" << profile << " max_bound_ratio=" << maximum_ratio << '\n';
    }
    return device_out.verify_guards(std::string(format.name) + " " + profile);
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
    auto x = make_input(k);
    DeviceBuffer device_payload = to_device(rows);
    DeviceBuffer device_x       = to_device_bf16(x);
    DeviceBuffer device_out(static_cast<std::size_t>(n) * sizeof(std::uint16_t));
    Tensor x_tensor(device_x.p, DType::BF16, {k});
    Tensor out_tensor(device_out.p, DType::BF16, {n});
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
    Tensor aliased_out(device_x.p, DType::BF16, {n});
    failures += expect_invalid(
        [&] { ops::ggml_block_linear(x_tensor, valid, aliased_out, nullptr); }, "output alias");
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
            failures += run_case(format, 3, 2 * format.block_values, false, stream,
                                 "representative");
            failures += run_case(format, format.real_n, format.real_k, true, stream, "real-shape");
        }
        failures += run_case(kFormats[2], 3, 6144, false, stream, "q5-output-projection-k");
        cuda_check(cudaStreamDestroy(stream), "cudaStreamDestroy");
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
