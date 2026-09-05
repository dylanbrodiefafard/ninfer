#include "ninfer/ops/gated_delta_net_layer.h"

#include "ops/op_tester.h"
#include "ops/launcher/gated_delta_net_layer.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kHidden     = 2560;
constexpr std::int32_t kQkHeads    = 16;
constexpr std::int32_t kValueHeads = 48;
constexpr std::int32_t kHeadDim    = 128;
constexpr std::int32_t kQkRows     = kQkHeads * kHeadDim;
constexpr std::int32_t kValueRows  = kValueHeads * kHeadDim;
constexpr std::int32_t kQkvRows    = 2 * kQkRows + kValueRows;
constexpr std::int32_t kPrompt     = 3;

// The complete T=1 profile crosses represented BF16 projection/conv/norm boundaries and retains
// measured headroom while detecting any omitted normalization, head repeat, gate, decay, or scale.
constexpr ReductionCriterion kOutputCriterion{/*relative_l2=*/9.0e-3,
                                               /*gross_absolute=*/2.5e-4,
                                               /*gross_relative_to_max_reference=*/5.0e-3};
constexpr ReductionCriterion kStateCriterion{/*relative_l2=*/6.5e-3,
                                              /*gross_absolute=*/2.0e-5,
                                              /*gross_relative_to_max_reference=*/4.0e-3};
constexpr ReductionCriterion kConvStateCriterion{/*relative_l2=*/5.0e-5,
                                                  /*gross_absolute=*/5.0e-6,
                                                  /*gross_relative_to_max_reference=*/1.0e-5};

double sigmoid(double value) {
    if (value >= 0.0) { return 1.0 / (1.0 + std::exp(-value)); }
    const double exponential = std::exp(value);
    return exponential / (1.0 + exponential);
}

double softplus(double value) {
    if (value > 20.0) { return value; }
    if (value < -20.0) { return std::exp(value); }
    return std::log1p(std::exp(value));
}

double silu(double value) { return value * sigmoid(value); }

double represented_bf16(double value) {
    return static_cast<double>(bf16_to_f32(f32_to_bf16(static_cast<float>(value))));
}

int conv_source_layout_witness() {
    constexpr int channel = 7;
    constexpr int key_channel = kQkRows + 11;
    std::vector<float> projected(kQkvRows, 0.0F);
    std::vector<float> state(static_cast<std::size_t>(kQkvRows) * 3, 0.0F);
    std::vector<float> weight(static_cast<std::size_t>(kQkvRows) * 4, 0.0F);
    projected[channel] = 4.0F;
    state[channel] = 1.0F;
    state[kQkvRows + channel] = 2.0F;
    state[2 * kQkvRows + channel] = 3.0F;
    weight[4 * channel] = 1.0F;
    weight[4 * channel + 1] = 2.0F;
    weight[4 * channel + 2] = 4.0F;
    weight[4 * channel + 3] = 8.0F;
    projected[key_channel] = 4.0F;
    state[key_channel] = 1.0F;
    state[kQkvRows + key_channel] = 2.0F;
    state[2 * kQkvRows + key_channel] = 3.0F;
    weight[4 * key_channel] = 1.0F;
    weight[4 * key_channel + 1] = 2.0F;
    weight[4 * key_channel + 2] = 4.0F;
    weight[4 * key_channel + 3] = 8.0F;
    round_to_bf16(projected);
    round_to_bf16(state);
    auto d_projected = to_device_bf16(projected);
    auto d_state = to_device_bf16(state);
    auto d_weight = to_device_f32(weight);
    GuardedDeviceBuffer d_state_out(state.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer d_q(kValueRows * sizeof(std::uint16_t));
    GuardedDeviceBuffer d_k(kValueRows * sizeof(std::uint16_t));
    GuardedDeviceBuffer d_v(kValueRows * sizeof(std::uint16_t));
    d_state_out.fill(0);
    d_q.fill(0);
    d_k.fill(0);
    d_v.fill(0);
    Tensor projected_t(d_projected.p, DType::BF16, {kQkvRows});
    Tensor weight_t(d_weight.p, DType::FP32, {4, kQkvRows});
    Tensor state_t(d_state.p, DType::BF16, {kQkvRows, 3});
    Tensor state_out_t(d_state_out.data(), DType::BF16, {kQkvRows, 3});
    Tensor q_t(d_q.data(), DType::BF16, {kHeadDim, kValueHeads});
    Tensor k_t(d_k.data(), DType::BF16, {kHeadDim, kValueHeads});
    Tensor v_t(d_v.data(), DType::BF16, {kHeadDim, kValueHeads});
    ops::detail::gated_delta_net_layer_conv_launch(projected_t, weight_t, state_t, state_out_t,
                                                    q_t, k_t, v_t, nullptr);
    cuda_synchronize();
    const auto q_bits = from_device<std::uint16_t>(d_q.data(), kValueRows);
    const auto state_bits = from_device<std::uint16_t>(d_state_out.data(), state.size());
    // 1*1 + 2*2 + 4*3 + 8*4 = 49. A tap-major transpose reads zeros at this channel.
    int failures = q_bits[channel] == f32_to_bf16(49.0F) ? 0 : 1;
    failures += q_bits[kQkRows + channel] == f32_to_bf16(49.0F) ? 0 : 1;
    failures += q_bits[2 * kQkRows + channel] == f32_to_bf16(49.0F) ? 0 : 1;
    const auto k_bits = from_device<std::uint16_t>(d_k.data(), kValueRows);
    constexpr int key_index = key_channel - kQkRows;
    failures += k_bits[key_index] == f32_to_bf16(49.0F) ? 0 : 1;
    failures += k_bits[kQkRows + key_index] == f32_to_bf16(49.0F) ? 0 : 1;
    failures += k_bits[2 * kQkRows + key_index] == f32_to_bf16(49.0F) ? 0 : 1;
    failures += state_bits[channel] == f32_to_bf16(2.0F) ? 0 : 1;
    failures += state_bits[kQkvRows + channel] == f32_to_bf16(3.0F) ? 0 : 1;
    failures += state_bits[2 * kQkvRows + channel] == f32_to_bf16(4.0F) ? 0 : 1;
    if (failures != 0) { std::cerr << "FAIL GDN GGUF channel-major conv layout witness\n"; }
    failures += d_state_out.verify_guards("GDN layout state");
    failures += d_q.verify_guards("GDN layout q");
    failures += d_k.verify_guards("GDN layout k");
    failures += d_v.verify_guards("GDN layout v");
    return failures;
}

std::vector<std::uint16_t> bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    std::transform(values.begin(), values.end(), bits.begin(), f32_to_bf16);
    return bits;
}

std::uint16_t read_u16(const std::uint8_t* bytes) {
    return static_cast<std::uint16_t>(bytes[0]) |
           (static_cast<std::uint16_t>(bytes[1]) << 8U);
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

std::pair<int, int> scale_min(const std::uint8_t* table, int group) {
    if (group < 4) { return {table[group] & 63, table[group + 4] & 63}; }
    return {(table[group + 4] & 15) + 16 * (table[group - 4] >> 6U),
            (table[group + 4] >> 4U) + 16 * (table[group] >> 6U)};
}

std::vector<double> decode_block(QType qtype, const std::uint8_t* block) {
    std::vector<double> values(256);
    if (qtype == QType::GGML_Q5_K) {
        const double d = half_to_double(read_u16(block));
        const double dmin = half_to_double(read_u16(block + 2));
        for (int group = 0; group < 8; ++group) {
            const auto [scale, minimum] = scale_min(block + 4, group);
            for (int lane = 0; lane < 32; ++lane) {
                const int packed = block[48 + 32 * (group / 2) + lane];
                int code = group % 2 == 0 ? packed & 15 : packed >> 4;
                code += 16 * ((block[16 + lane] >> group) & 1U);
                values[32 * group + lane] = d * scale * code - dmin * minimum;
            }
        }
        return values;
    }
    const double d = half_to_double(read_u16(block + 208));
    for (int half = 0; half < 2; ++half) {
        for (int group = 0; group < 4; ++group) {
            for (int lane = 0; lane < 32; ++lane) {
                const int index = 128 * half + 32 * group + lane;
                const int low_word = block[64 * half + lane + 32 * (group % 2)];
                const int low = group < 2 ? low_word & 15 : low_word >> 4;
                const int high = (block[128 + 32 * half + lane] >> (2 * group)) & 3;
                values[index] = d * signed_byte(block[192 + index / 16]) *
                                (low + 16 * high - 32);
            }
        }
    }
    return values;
}

void write_u16(std::uint8_t* bytes, std::uint16_t value) {
    bytes[0] = static_cast<std::uint8_t>(value & 0xffU);
    bytes[1] = static_cast<std::uint8_t>(value >> 8U);
}

std::vector<std::uint8_t> make_quant(QType qtype, std::int32_t rows, std::int32_t columns,
                                     std::uint32_t seed) {
    const std::size_t block_bytes = qtype == QType::GGML_Q5_K ? 176U : 210U;
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(rows) * (columns / 256) * block_bytes);
    std::mt19937 generator(seed);
    std::uniform_int_distribution<int> distribution(0, 255);
    for (auto& byte : bytes) { byte = static_cast<std::uint8_t>(distribution(generator)); }
    for (std::size_t offset = 0; offset < bytes.size(); offset += block_bytes) {
        auto* block = bytes.data() + offset;
        if (qtype == QType::GGML_Q5_K) {
            write_u16(block, 0x0080U);
            write_u16(block + 2, 0x0040U);
        } else {
            write_u16(block + 208, 0x0080U);
        }
    }
    return bytes;
}

Weight ggml_weight(void* data, std::uint64_t bytes, QType qtype, std::int32_t rows,
                   std::int32_t columns) {
    Weight weight{};
    weight.payload         = data;
    weight.payload_bytes   = bytes;
    weight.qdata           = data;
    weight.qtype           = qtype;
    weight.layout          = QuantLayout::GgmlBlockRow;
    weight.n               = rows;
    weight.k               = columns;
    weight.group           = 256;
    weight.group_size      = 256;
    weight.ndim            = 2;
    weight.shape[0]        = rows;
    weight.shape[1]        = columns;
    weight.padded_shape[0] = rows;
    weight.padded_shape[1] = columns;
    return weight;
}

struct Fixture {
    QType input_qtype;
    std::vector<std::uint8_t> qkv;
    std::vector<std::uint8_t> z;
    std::vector<float> a;
    std::vector<float> b;
    std::vector<float> conv;
    std::vector<float> ssm_a;
    std::vector<float> dt_bias;
    std::vector<float> norm;
    std::vector<std::uint8_t> output;
    DeviceBuffer d_qkv;
    DeviceBuffer d_z;
    DeviceBuffer d_a;
    DeviceBuffer d_b;
    DeviceBuffer d_conv;
    DeviceBuffer d_ssm_a;
    DeviceBuffer d_dt_bias;
    DeviceBuffer d_norm;
    DeviceBuffer d_output;

    explicit Fixture(QType input_format)
        : input_qtype(input_format),
          qkv(make_quant(input_format, kQkvRows, kHidden, 8101U)),
          z(make_quant(input_format, kValueRows, kHidden, 8102U)),
          a(static_cast<std::size_t>(kValueHeads) * kHidden),
          b(static_cast<std::size_t>(kValueHeads) * kHidden),
          conv(static_cast<std::size_t>(kQkvRows) * 4), ssm_a(kValueHeads),
          dt_bias(kValueHeads), norm(kHeadDim),
          output(make_quant(QType::GGML_Q6_K, kHidden, kValueRows, 8109U)) {
        fill_uniform(a, 8103U, -0.008F, 0.008F);
        fill_uniform(b, 8104U, -0.008F, 0.008F);
        fill_uniform(conv, 8105U, -0.20F, 0.20F);
        fill_uniform(ssm_a, 8106U, -0.25F, -0.02F);
        ssm_a[0] = -157.984F;
        ssm_a[17] = -0.0279F;
        fill_uniform(dt_bias, 8107U, -0.5F, 0.5F);
        fill_uniform(norm, 8108U, 0.5F, 1.5F);
        d_qkv = to_device(qkv);
        d_z = to_device(z);
        d_a = to_device_f32(a);
        d_b = to_device_f32(b);
        d_conv = to_device_f32(conv);
        d_ssm_a = to_device_f32(ssm_a);
        d_dt_bias = to_device_f32(dt_bias);
        d_norm = to_device_f32(norm);
        d_output = to_device(output);
    }

    ops::GatedDeltaNetLayerWeights views() {
        return {
            ggml_weight(d_qkv.p, d_qkv.bytes, input_qtype, kQkvRows, kHidden),
            ggml_weight(d_z.p, d_z.bytes, input_qtype, kValueRows, kHidden),
            Tensor(d_a.p, DType::FP32, {kHidden, kValueHeads}),
            Tensor(d_b.p, DType::FP32, {kHidden, kValueHeads}),
            Tensor(d_conv.p, DType::FP32, {4, kQkvRows}),
            Tensor(d_ssm_a.p, DType::FP32, {kValueHeads}),
            Tensor(d_dt_bias.p, DType::FP32, {kValueHeads}),
            Tensor(d_norm.p, DType::FP32, {kHeadDim}),
            ggml_weight(d_output.p, d_output.bytes, QType::GGML_Q6_K, kHidden, kValueRows),
        };
    }
};

struct OracleResult {
    std::vector<double> output;
    std::vector<double> conv_state;
    std::vector<double> ssm_state;
};

std::vector<double> project(const std::vector<float>& weight, std::int32_t rows,
                            const std::vector<float>& input, std::int32_t tokens) {
    std::vector<double> result(static_cast<std::size_t>(rows) * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        const std::size_t input_base = static_cast<std::size_t>(token) * kHidden;
        for (std::int32_t row = 0; row < rows; ++row) {
            const std::size_t weight_base = static_cast<std::size_t>(row) * kHidden;
            double sum = 0.0;
            for (std::int32_t d = 0; d < kHidden; ++d) {
                sum += static_cast<double>(weight[weight_base + d]) * input[input_base + d];
            }
            result[static_cast<std::size_t>(token) * rows + row] = sum;
        }
    }
    return result;
}

std::vector<double> project_quant(const std::vector<std::uint8_t>& weight, QType qtype,
                                  std::int32_t rows, std::int32_t columns,
                                  const std::vector<double>& input) {
    const std::size_t block_bytes = qtype == QType::GGML_Q5_K ? 176U : 210U;
    const std::size_t row_bytes = static_cast<std::size_t>(columns / 256) * block_bytes;
    std::vector<double> result(rows);
    for (std::int32_t row = 0; row < rows; ++row) {
        double sum = 0.0;
        const auto* row_data = weight.data() + static_cast<std::size_t>(row) * row_bytes;
        for (std::int32_t block_index = 0; block_index < columns / 256; ++block_index) {
            const auto decoded = decode_block(qtype, row_data + block_index * block_bytes);
            for (std::int32_t item = 0; item < 256; ++item) {
                sum += decoded[item] * input[static_cast<std::size_t>(block_index) * 256 + item];
            }
        }
        result[row] = sum;
    }
    return result;
}

OracleResult oracle(const Fixture& fixture, const std::vector<float>& input,
                    const std::vector<float>& initial_conv,
                    const std::vector<float>& initial_ssm, std::int32_t tokens) {
    std::vector<double> raw(static_cast<std::size_t>(kQkvRows) * tokens);
    std::vector<double> z(static_cast<std::size_t>(kValueRows) * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        std::vector<double> token_input(kHidden);
        for (std::int32_t d = 0; d < kHidden; ++d) {
            token_input[d] = input[static_cast<std::size_t>(token) * kHidden + d];
        }
        const auto raw_token =
            project_quant(fixture.qkv, fixture.input_qtype, kQkvRows, kHidden, token_input);
        const auto z_token =
            project_quant(fixture.z, fixture.input_qtype, kValueRows, kHidden, token_input);
        for (std::int32_t row = 0; row < kQkvRows; ++row) {
            raw[static_cast<std::size_t>(token) * kQkvRows + row] =
                represented_bf16(raw_token[row]);
        }
        for (std::int32_t row = 0; row < kValueRows; ++row) {
            z[static_cast<std::size_t>(token) * kValueRows + row] =
                represented_bf16(z_token[row]);
        }
    }
    const std::vector<double> a   = project(fixture.a, kValueHeads, input, tokens);
    const std::vector<double> b   = project(fixture.b, kValueHeads, input, tokens);
    std::vector<double> q(static_cast<std::size_t>(kQkRows) * tokens);
    std::vector<double> k(static_cast<std::size_t>(kQkRows) * tokens);
    std::vector<double> v(static_cast<std::size_t>(kValueRows) * tokens);
    std::vector<double> conv_state(static_cast<std::size_t>(kQkvRows) * 3);
    for (std::int32_t channel = 0; channel < kQkvRows; ++channel) {
        double s0 = initial_conv[channel];
        double s1 = initial_conv[kQkvRows + channel];
        double s2 = initial_conv[2 * kQkvRows + channel];
        for (std::int32_t token = 0; token < tokens; ++token) {
            const double p = raw[static_cast<std::size_t>(token) * kQkvRows + channel];
            const std::size_t weight_base = static_cast<std::size_t>(channel) * 4;
            const double sum = static_cast<double>(fixture.conv[weight_base]) * s0 +
                               static_cast<double>(fixture.conv[weight_base + 1]) * s1 +
                               static_cast<double>(fixture.conv[weight_base + 2]) * s2 +
                               static_cast<double>(fixture.conv[weight_base + 3]) * p;
            const double value = represented_bf16(silu(sum));
            if (channel < kQkRows) {
                q[static_cast<std::size_t>(token) * kQkRows + channel] = value;
            } else if (channel < 2 * kQkRows) {
                k[static_cast<std::size_t>(token) * kQkRows + channel - kQkRows] = value;
            } else {
                v[static_cast<std::size_t>(token) * kValueRows + channel - 2 * kQkRows] = value;
            }
            s0 = s1;
            s1 = s2;
            s2 = p;
        }
        conv_state[channel]                = represented_bf16(s0);
        conv_state[kQkvRows + channel]     = represented_bf16(s1);
        conv_state[2 * kQkvRows + channel] = represented_bf16(s2);
    }

    std::vector<double> q_norm(q.size());
    std::vector<double> k_norm(k.size());
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t head = 0; head < kQkHeads; ++head) {
            const std::size_t base = static_cast<std::size_t>(token * kQkHeads + head) * kHeadDim;
            double q_sum = 0.0;
            double k_sum = 0.0;
            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                q_sum += q[base + d] * q[base + d];
                k_sum += k[base + d] * k[base + d];
            }
            const double q_inverse = 1.0 / std::sqrt(q_sum + 1.0e-6);
            const double k_inverse = 1.0 / std::sqrt(k_sum + 1.0e-6);
            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                q_norm[base + d] = q[base + d] * q_inverse;
                k_norm[base + d] = k[base + d] * k_inverse;
            }
        }
    }

    std::vector<double> state(initial_ssm.begin(), initial_ssm.end());
    std::vector<double> recurrent(static_cast<std::size_t>(kValueRows) * tokens);
    std::vector<double> delta(kHeadDim);
    const double query_scale = 1.0 / std::sqrt(static_cast<double>(kHeadDim));
    for (std::int32_t head = 0; head < kValueHeads; ++head) {
        // GGUF stores V-side heads tiled by repeat group, so represented head h consumes h%16.
        // The random per-head state/control fixtures make floor(h/3) a strict wrong-layout path.
        const std::int32_t qk_head = head % kQkHeads;
        const std::size_t state_base = static_cast<std::size_t>(head) * kHeadDim * kHeadDim;
        for (std::int32_t token = 0; token < tokens; ++token) {
            const std::size_t qk_base =
                static_cast<std::size_t>(token * kQkHeads + qk_head) * kHeadDim;
            const std::size_t value_base =
                static_cast<std::size_t>(token * kValueHeads + head) * kHeadDim;
            const std::size_t control = static_cast<std::size_t>(token) * kValueHeads + head;
            const double g = static_cast<double>(static_cast<float>(
                static_cast<double>(fixture.ssm_a[head]) *
                softplus(a[control] + fixture.dt_bias[head])));
            const double beta =
                static_cast<double>(static_cast<float>(sigmoid(b[control])));
            const double alpha = std::exp(g);
            for (std::int32_t row = 0; row < kHeadDim; ++row) {
                double dot = 0.0;
                const std::size_t row_base = state_base + static_cast<std::size_t>(row) * kHeadDim;
                for (std::int32_t column = 0; column < kHeadDim; ++column) {
                    dot += state[row_base + column] * k_norm[qk_base + column];
                }
                delta[row] = beta * (v[value_base + row] - alpha * dot);
            }
            for (std::int32_t row = 0; row < kHeadDim; ++row) {
                const std::size_t row_base = state_base + static_cast<std::size_t>(row) * kHeadDim;
                for (std::int32_t column = 0; column < kHeadDim; ++column) {
                    state[row_base + column] = static_cast<double>(static_cast<float>(
                        alpha * state[row_base + column] +
                        delta[row] * k_norm[qk_base + column]));
                }
                double dot = 0.0;
                for (std::int32_t column = 0; column < kHeadDim; ++column) {
                    dot += state[row_base + column] * q_norm[qk_base + column];
                }
                recurrent[value_base + row] = represented_bf16(query_scale * dot);
            }
        }
    }

    std::vector<double> normalized_gated(recurrent.size());
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t head = 0; head < kValueHeads; ++head) {
            const std::size_t base =
                static_cast<std::size_t>(token * kValueHeads + head) * kHeadDim;
            double sum = 0.0;
            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                sum += recurrent[base + d] * recurrent[base + d];
            }
            const double inverse = 1.0 / std::sqrt(sum / kHeadDim + 1.0e-6);
            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                normalized_gated[base + d] = represented_bf16(
                    recurrent[base + d] * inverse * fixture.norm[d] *
                    sigmoid(represented_bf16(z[base + d])));
            }
        }
    }
    std::vector<double> output(static_cast<std::size_t>(kHidden) * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        std::vector<double> token_input(kValueRows);
        for (std::int32_t d = 0; d < kValueRows; ++d) {
            token_input[d] = normalized_gated[static_cast<std::size_t>(token) * kValueRows + d];
        }
        const auto token_output =
            project_quant(fixture.output, QType::GGML_Q6_K, kHidden, kValueRows, token_input);
        for (std::int32_t row = 0; row < kHidden; ++row) {
            output[static_cast<std::size_t>(token) * kHidden + row] = token_output[row];
        }
    }
    return {std::move(output), std::move(conv_state), std::move(state)};
}

int run_complete_case(Fixture& fixture, std::int32_t tokens, const char* label) {
    std::vector<float> input(static_cast<std::size_t>(kHidden) * tokens);
    std::vector<float> initial_conv(static_cast<std::size_t>(kQkvRows) * 3);
    std::vector<float> initial_ssm(static_cast<std::size_t>(kHeadDim) * kHeadDim * kValueHeads);
    fill_uniform(input, 8201U, -0.20F, 0.20F);
    fill_uniform(initial_conv, 8202U, -0.05F, 0.05F);
    fill_uniform(initial_ssm, 8203U, -0.002F, 0.002F);
    round_to_bf16(input);
    round_to_bf16(initial_conv);
    const OracleResult expected = oracle(fixture, input, initial_conv, initial_ssm, tokens);
    DeviceBuffer d_input      = to_device_bf16(input);
    DeviceBuffer d_conv_in    = to_device_bf16(initial_conv);
    DeviceBuffer d_ssm_in     = to_device_f32(initial_ssm);
    GuardedDeviceBuffer d_conv_out(initial_conv.size() * 2);
    GuardedDeviceBuffer d_ssm_out(initial_ssm.size() * sizeof(float));
    GuardedDeviceBuffer d_output(static_cast<std::size_t>(kHidden) * tokens * 2);
    d_conv_out.fill(0xff);
    d_ssm_out.fill(0xff);
    d_output.fill(0xff);
    Tensor x(d_input.p, DType::BF16, {kHidden, tokens});
    Tensor conv_in(d_conv_in.p, DType::BF16, {kQkvRows, 3});
    Tensor conv_out(d_conv_out.data(), DType::BF16, {kQkvRows, 3});
    Tensor ssm_in(d_ssm_in.p, DType::FP32, {kHeadDim, kHeadDim, kValueHeads});
    Tensor ssm_out(d_ssm_out.data(), DType::FP32, {kHeadDim, kHeadDim, kValueHeads});
    Tensor output(d_output.data(), DType::BF16, {kHidden, tokens});
    auto weights = fixture.views();
    const std::size_t full_bytes = ops::gated_delta_net_layer_workspace_capacity_bytes(tokens);
    WorkspaceArena full_workspace(full_bytes);
    ops::gated_delta_net_layer(x, weights, conv_in, conv_out, ssm_in, ssm_out, output,
                               full_workspace, nullptr);
    cuda_synchronize();

    int failures = verify_reduction(std::string(label) + " panel output",
                                    from_device_bf16(d_output.data(), expected.output.size()),
                                    expected.output, kOutputCriterion);
    failures += verify_reduction(std::string(label) + " distinct conv state",
                                 from_device_bf16(d_conv_out.data(), initial_conv.size()),
                                 expected.conv_state, kConvStateCriterion);
    std::vector<double> actual_ssm(initial_ssm.size());
    const std::vector<float> actual_ssm_f32 =
        from_device<float>(d_ssm_out.data(), initial_ssm.size());
    std::copy(actual_ssm_f32.begin(), actual_ssm_f32.end(), actual_ssm.begin());
    failures += verify_reduction(std::string(label) + " distinct SSM state", actual_ssm,
                                 expected.ssm_state,
                                 kStateCriterion);
    const std::string rollback_conv_label = std::string(label) + " rollback conv input";
    const std::string rollback_ssm_label = std::string(label) + " rollback SSM input";
    failures += verify_exact(rollback_conv_label.c_str(), from_device<std::uint16_t>(
                                 d_conv_in, initial_conv.size()), bf16_bits(initial_conv));
    failures += verify_exact(rollback_ssm_label.c_str(),
                             from_device<float>(d_ssm_in, initial_ssm.size()), initial_ssm);

    DeviceBuffer sequential_conv = to_device_bf16(initial_conv);
    DeviceBuffer sequential_ssm  = to_device_f32(initial_ssm);
    GuardedDeviceBuffer sequential_output(input.size() * 2);
    sequential_output.fill(0xff);
    Tensor sequential_conv_tensor(sequential_conv.p, DType::BF16, {kQkvRows, 3});
    Tensor sequential_ssm_tensor(sequential_ssm.p, DType::FP32,
                                 {kHeadDim, kHeadDim, kValueHeads});
    Tensor sequential_output_tensor(sequential_output.data(), DType::BF16, {kHidden, tokens});
    const std::size_t step_bytes = ops::gated_delta_net_layer_workspace_capacity_bytes();
    WorkspaceArena step_workspace(step_bytes);
    for (std::int32_t token = 0; token < tokens; ++token) {
        Tensor x_step   = x.slice(1, token, 1);
        Tensor out_step = sequential_output_tensor.slice(1, token, 1);
        ops::gated_delta_net_layer(x_step, weights, sequential_conv_tensor,
                                   sequential_conv_tensor, sequential_ssm_tensor,
                                   sequential_ssm_tensor, out_step, step_workspace, nullptr);
    }
    cuda_synchronize();
    failures += verify_reduction(std::string(label) + " repeated T=1 output",
                                 from_device_bf16(sequential_output.data(), input.size()),
                                 expected.output, kOutputCriterion);
    const std::vector<double> sequential_ssm_actual = [&] {
        const std::vector<float> values =
            from_device<float>(sequential_ssm, initial_ssm.size());
        return std::vector<double>(values.begin(), values.end());
    }();
    failures += verify_reduction(std::string(label) + " repeated T=1 SSM state", sequential_ssm_actual,
                                 expected.ssm_state, kStateCriterion);
    failures += verify_reduction(std::string(label) + " repeated T=1 conv state",
                                 from_device_bf16(sequential_conv, initial_conv.size()),
                                 expected.conv_state, kConvStateCriterion);
    failures += d_conv_out.verify_guards(std::string(label) + " conv state");
    failures += d_ssm_out.verify_guards(std::string(label) + " SSM state");
    failures += d_output.verify_guards(std::string(label) + " output");
    failures += sequential_output.verify_guards(std::string(label) + " repeated output");
    if (full_workspace.used() != 0 || full_workspace.peak_used() != full_bytes ||
        step_workspace.used() != 0 || step_workspace.peak_used() != step_bytes) {
        std::cerr << "FAIL " << label << " workspace query/high-water mismatch\n";
        ++failures;
    }
    return failures;
}

int run_partition_case(Fixture& fixture, std::int32_t tokens,
                       std::initializer_list<std::int32_t> chunks, const char* label) {
    std::int32_t total = 0;
    std::int32_t maximum_chunk = 0;
    for (const std::int32_t chunk : chunks) {
        total += chunk;
        maximum_chunk = std::max(maximum_chunk, chunk);
    }
    if (total != tokens || maximum_chunk <= 0) {
        std::cerr << "FAIL invalid GDN layer partition fixture\n";
        return 1;
    }

    std::vector<float> input(static_cast<std::size_t>(kHidden) * tokens);
    std::vector<float> initial_conv(static_cast<std::size_t>(kQkvRows) * 3);
    std::vector<float> initial_ssm(static_cast<std::size_t>(kHeadDim) * kHeadDim * kValueHeads);
    fill_uniform(input, 9100U + tokens, -0.20F, 0.20F);
    fill_uniform(initial_conv, 9200U + tokens, -0.05F, 0.05F);
    fill_uniform(initial_ssm, 9300U + tokens, -0.002F, 0.002F);
    round_to_bf16(input);
    round_to_bf16(initial_conv);

    DeviceBuffer d_input = to_device_bf16(input);
    DeviceBuffer full_conv = to_device_bf16(initial_conv);
    DeviceBuffer full_ssm = to_device_f32(initial_ssm);
    GuardedDeviceBuffer full_output(input.size() * sizeof(std::uint16_t));
    DeviceBuffer partition_conv = to_device_bf16(initial_conv);
    DeviceBuffer partition_ssm = to_device_f32(initial_ssm);
    GuardedDeviceBuffer partition_output(input.size() * sizeof(std::uint16_t));
    full_output.fill(0xff);
    partition_output.fill(0xff);

    Tensor x(d_input.p, DType::BF16, {kHidden, tokens});
    Tensor full_conv_t(full_conv.p, DType::BF16, {kQkvRows, 3});
    Tensor full_ssm_t(full_ssm.p, DType::FP32, {kHeadDim, kHeadDim, kValueHeads});
    Tensor full_output_t(full_output.data(), DType::BF16, {kHidden, tokens});
    Tensor partition_conv_t(partition_conv.p, DType::BF16, {kQkvRows, 3});
    Tensor partition_ssm_t(partition_ssm.p, DType::FP32, {kHeadDim, kHeadDim, kValueHeads});
    Tensor partition_output_t(partition_output.data(), DType::BF16, {kHidden, tokens});
    auto weights = fixture.views();

    const std::size_t full_bytes = ops::gated_delta_net_layer_workspace_capacity_bytes(tokens);
    WorkspaceArena full_workspace(full_bytes);
    ops::gated_delta_net_layer(x, weights, full_conv_t, full_conv_t, full_ssm_t, full_ssm_t,
                               full_output_t, full_workspace, nullptr);

    const std::size_t partition_bytes =
        ops::gated_delta_net_layer_workspace_capacity_bytes(maximum_chunk);
    WorkspaceArena partition_workspace(partition_bytes);
    std::int32_t offset = 0;
    for (const std::int32_t chunk : chunks) {
        Tensor x_chunk = x.slice(1, offset, chunk);
        Tensor output_chunk = partition_output_t.slice(1, offset, chunk);
        ops::gated_delta_net_layer(x_chunk, weights, partition_conv_t, partition_conv_t,
                                   partition_ssm_t, partition_ssm_t, output_chunk,
                                   partition_workspace, nullptr);
        offset += chunk;
    }
    cuda_synchronize();

    const std::string prefix = std::string(label) + " T=" + std::to_string(tokens);
    const auto full_output_values = from_device_bf16(full_output.data(), input.size());
    const auto partition_output_values = from_device_bf16(partition_output.data(), input.size());
    std::vector<double> full_reference(full_output_values.begin(), full_output_values.end());
    int failures = verify_reduction(prefix + " partition output", partition_output_values,
                                    full_reference, kOutputCriterion);
    failures += verify_exact((prefix + " partition conv state").c_str(),
                             from_device<std::uint16_t>(partition_conv, initial_conv.size()),
                             from_device<std::uint16_t>(full_conv, initial_conv.size()));
    const auto full_ssm_values = from_device<float>(full_ssm, initial_ssm.size());
    const auto partition_ssm_values = from_device<float>(partition_ssm, initial_ssm.size());
    failures += verify_reduction(prefix + " partition SSM state",
                                 std::vector<double>(partition_ssm_values.begin(),
                                                     partition_ssm_values.end()),
                                 std::vector<double>(full_ssm_values.begin(), full_ssm_values.end()),
                                 kStateCriterion);
    failures += full_output.verify_guards((prefix + " full output").c_str());
    failures += partition_output.verify_guards((prefix + " partition output").c_str());
    if (full_workspace.used() != 0 || full_workspace.peak_used() != full_bytes ||
        partition_workspace.used() != 0 ||
        partition_workspace.peak_used() != partition_bytes) {
        std::cerr << "FAIL " << prefix << " workspace query/high-water mismatch\n";
        ++failures;
    }
    return failures;
}

} // namespace

int main() {
    if (const int unavailable = require_cuda()) { return unavailable; }
    Fixture q5_fixture(QType::GGML_Q5_K);
    int failures = conv_source_layout_witness();
    failures += run_complete_case(q5_fixture, kPrompt, "GDN Q5_K/Q6_K");
    failures += run_partition_case(q5_fixture, 64, {31, 33}, "GDN chunk boundary");
    failures += run_partition_case(q5_fixture, 65, {64, 1}, "GDN chunk tail");
    Fixture q6_fixture(QType::GGML_Q6_K);
    failures += run_complete_case(q6_fixture, 1, "GDN layer-2 Q6_K/Q6_K");
    try {
        const std::size_t broad = ops::gated_delta_net_layer_workspace_capacity_bytes(4096);
        if (broad <= ops::gated_delta_net_layer_workspace_capacity_bytes(65)) {
            std::cerr << "FAIL GDN broad workspace capacity\n";
            ++failures;
        }
    } catch (const std::invalid_argument&) {
        std::cerr << "FAIL GDN rejected T=4096 capacity\n";
        ++failures;
    }
    try {
        (void)ops::gated_delta_net_layer_workspace_capacity_bytes(4097);
        std::cerr << "FAIL GDN accepted T=4097 capacity\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
    std::cout << (failures ? "FAIL" : "OK") << " gated_delta_net_layer\n";
    return failures == 0 ? 0 : 1;
}
