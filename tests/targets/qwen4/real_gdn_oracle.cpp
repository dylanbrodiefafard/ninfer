#include "targets/qwen4/real_oracle_common.h"

#include "ninfer/ops/gated_delta_net_layer.h"
#include "ninfer/ops/gated_residual.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace verifier = ninfer::targets::qwen4::verifier;
namespace ops = ninfer::ops;
using ninfer::DType;
using ninfer::DeviceBuffer;
using ninfer::DeviceContext;
using ninfer::QType;
using ninfer::QuantLayout;
using ninfer::Tensor;
using ninfer::Weight;
using ninfer::WorkspaceArena;
using ninfer::test::GuardedDeviceBuffer;
using ninfer::test::ReductionCriterion;
using ninfer::test::bf16_to_f32;
using ninfer::test::f32_to_bf16;
using ninfer::test::from_device;
using ninfer::test::from_device_bf16;
using ninfer::test::to_device;
using ninfer::test::verify_exact;
using ninfer::test::verify_reduction;
using namespace ninfer::test::qwen4::real_oracle;

namespace {

constexpr std::int32_t kHidden = 2560;
constexpr std::int32_t kBranches = 4;
constexpr std::int32_t kQkHeads = 16;
constexpr std::int32_t kValueHeads = 48;
constexpr std::int32_t kHeadDim = 128;
constexpr std::int32_t kQkRows = kQkHeads * kHeadDim;
constexpr std::int32_t kValueRows = kValueHeads * kHeadDim;
constexpr std::int32_t kQkvRows = 2 * kQkRows + kValueRows;
constexpr std::size_t kQ6BlockBytes = 210;
constexpr std::size_t kQ6BlockValues = 256;

// These are the complete T=1 criteria owned by tests/ops/test_gated_delta_net_layer.cpp. The
// actual-artifact cells change only represented x/state and packed weights.
constexpr ReductionCriterion kOutputCriterion{/*relative_l2=*/9.0e-3,
                                               /*gross_absolute=*/2.5e-4,
                                               /*gross_relative_to_max_reference=*/5.0e-3};
constexpr ReductionCriterion kRecurrenceCriterion{/*relative_l2=*/6.5e-3,
                                                   /*gross_absolute=*/2.0e-5,
                                                   /*gross_relative_to_max_reference=*/4.0e-3};
constexpr ReductionCriterion kConvCriterion{/*relative_l2=*/5.0e-5,
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

double ggml_q6_k_value(const std::uint8_t* row, std::int32_t column) {
    const auto* block =
        row + static_cast<std::size_t>(column / kQ6BlockValues) * kQ6BlockBytes;
    const int item = column % static_cast<std::int32_t>(kQ6BlockValues);
    const int half = item / 128;
    const int within_half = item % 128;
    const int group = within_half / 32;
    const int lane = within_half % 32;
    const int low_word = block[64 * half + lane + 32 * (group % 2)];
    const int low = group < 2 ? low_word & 15 : low_word >> 4;
    const int high = (block[128 + 32 * half + lane] >> (2 * group)) & 3;
    const int scale = signed_i8(block[192 + item / 16]);
    return binary16_to_double(read_u16(block + 208)) * scale * (low + 16 * high - 32);
}

std::size_t quant_row_bytes(QType qtype, std::int32_t columns) {
    if (columns % 256 != 0) {
        throw std::logic_error("Qwen4 real GDN packed columns changed");
    }
    if (qtype == QType::GGML_Q5_K) {
        return static_cast<std::size_t>(columns) / kQ5BlockValues * kQ5BlockBytes;
    }
    if (qtype == QType::GGML_Q6_K) {
        return static_cast<std::size_t>(columns) / kQ6BlockValues * kQ6BlockBytes;
    }
    throw std::logic_error("Qwen4 real GDN received an unsupported packed format");
}

double quant_value(QType qtype, const std::uint8_t* row, std::int32_t column) {
    if (qtype == QType::GGML_Q5_K) { return ggml_q5_k_value(row, column); }
    if (qtype == QType::GGML_Q6_K) { return ggml_q6_k_value(row, column); }
    throw std::logic_error("Qwen4 real GDN received an unsupported packed format");
}

void require_quant(const Weight& weight, QType qtype, std::int32_t rows,
                   std::int32_t columns, const char* name) {
    const std::size_t expected = static_cast<std::size_t>(rows) *
                                 quant_row_bytes(qtype, columns);
    if (weight.qtype != qtype || weight.layout != QuantLayout::GgmlBlockRow ||
        weight.n != rows || weight.k != columns || weight.qdata == nullptr ||
        weight.payload_bytes != expected) {
        throw std::logic_error(std::string("Qwen4 real GDN malformed ") + name);
    }
}

std::vector<double> quant_project(std::span<const std::uint8_t> matrix, QType qtype,
                                  std::int32_t rows, std::int32_t columns,
                                  std::span<const double> input, bool represent_output) {
    const std::size_t row_bytes = quant_row_bytes(qtype, columns);
    if (matrix.size() != static_cast<std::size_t>(rows) * row_bytes ||
        input.size() != static_cast<std::size_t>(columns)) {
        throw std::logic_error("Qwen4 real GDN quantized oracle received malformed storage");
    }
    std::vector<double> output(static_cast<std::size_t>(rows));
    for (std::int32_t row = 0; row < rows; ++row) {
        const auto* encoded = matrix.data() + static_cast<std::size_t>(row) * row_bytes;
        double sum = 0.0;
        for (std::int32_t column = 0; column < columns; ++column) {
            sum += quant_value(qtype, encoded, column) * input[column];
        }
        output[row] = represent_output ? represented_bf16(sum) : sum;
    }
    return output;
}

std::vector<double> fp32_project(std::span<const float> matrix, std::int32_t rows,
                                 std::span<const double> input) {
    if (matrix.size() != static_cast<std::size_t>(rows) * kHidden ||
        input.size() != static_cast<std::size_t>(kHidden)) {
        throw std::logic_error("Qwen4 real GDN FP32 oracle received malformed storage");
    }
    std::vector<double> output(static_cast<std::size_t>(rows));
    for (std::int32_t row = 0; row < rows; ++row) {
        double sum = 0.0;
        const std::size_t base = static_cast<std::size_t>(row) * kHidden;
        for (std::int32_t column = 0; column < kHidden; ++column) {
            sum += static_cast<double>(matrix[base + column]) * input[column];
        }
        output[row] = sum;
    }
    return output;
}

std::vector<std::uint16_t> make_conv_state() {
    std::vector<std::uint16_t> state(static_cast<std::size_t>(kQkvRows) * 3);
    for (std::int32_t history = 0; history < 3; ++history) {
        for (std::int32_t channel = 0; channel < kQkvRows; ++channel) {
            const float magnitude =
                0.012F + 0.004F * history + 0.0001F * static_cast<float>(channel % 23);
            const float value = ((channel + history) & 1) == 0 ? magnitude : -magnitude;
            state[static_cast<std::size_t>(history) * kQkvRows + channel] =
                f32_to_bf16(value);
        }
    }
    return state;
}

std::vector<float> make_recurrence_state() {
    std::vector<float> state(static_cast<std::size_t>(kValueHeads) * kHeadDim * kHeadDim);
    for (std::int32_t head = 0; head < kValueHeads; ++head) {
        for (std::int32_t row = 0; row < kHeadDim; ++row) {
            for (std::int32_t column = 0; column < kHeadDim; ++column) {
                const std::size_t index =
                    (static_cast<std::size_t>(head) * kHeadDim + row) * kHeadDim + column;
                const float magnitude =
                    0.00015F + 0.000002F * static_cast<float>((7 * head + 3 * row + column) % 31);
                state[index] = ((head + row + column) & 1) == 0 ? magnitude : -magnitude;
            }
        }
    }
    return state;
}

struct HostWeights {
    QType input_qtype{};
    std::vector<std::uint8_t> qkv;
    std::vector<std::uint8_t> z;
    std::vector<float> a;
    std::vector<float> b;
    std::vector<float> conv;
    std::vector<float> ssm_a;
    std::vector<float> dt_bias;
    std::vector<float> norm;
    std::vector<std::uint8_t> output;
};

HostWeights copy_weights(const ops::GatedDeltaNetLayerWeights& weights, QType input_qtype) {
    require_quant(weights.qkv, input_qtype, kQkvRows, kHidden, "qkv");
    require_quant(weights.z, input_qtype, kValueRows, kHidden, "z");
    require_quant(weights.output, QType::GGML_Q6_K, kHidden, kValueRows, "output");
    if (weights.a.dtype != DType::FP32 || weights.a.ne[0] != kHidden ||
        weights.a.ne[1] != kValueHeads || weights.b.dtype != DType::FP32 ||
        weights.b.ne[0] != kHidden || weights.b.ne[1] != kValueHeads ||
        weights.conv.dtype != DType::FP32 || weights.conv.ne[0] != 4 ||
        weights.conv.ne[1] != kQkvRows || weights.ssm_a.dtype != DType::FP32 ||
        weights.ssm_a.numel() != kValueHeads || weights.dt_bias.dtype != DType::FP32 ||
        weights.dt_bias.numel() != kValueHeads || weights.norm.dtype != DType::FP32 ||
        weights.norm.numel() != kHeadDim) {
        throw std::logic_error("Qwen4 real GDN FP32 binding changed");
    }
    return {
        input_qtype,
        copy_device_bytes(weights.qkv.qdata, weights.qkv.payload_bytes),
        copy_device_bytes(weights.z.qdata, weights.z.payload_bytes),
        copy_device_values<float>(weights.a.data, static_cast<std::size_t>(kValueHeads) * kHidden),
        copy_device_values<float>(weights.b.data, static_cast<std::size_t>(kValueHeads) * kHidden),
        copy_device_values<float>(weights.conv.data, static_cast<std::size_t>(kQkvRows) * 4),
        copy_device_values<float>(weights.ssm_a.data, kValueHeads),
        copy_device_values<float>(weights.dt_bias.data, kValueHeads),
        copy_device_values<float>(weights.norm.data, kHeadDim),
        copy_device_bytes(weights.output.qdata, weights.output.payload_bytes),
    };
}

struct OracleResult {
    std::vector<double> output;
    std::vector<double> conv_state;
    std::vector<double> recurrence_state;
};

OracleResult gdn_oracle(const HostWeights& weights, std::span<const std::uint16_t> x_bits,
                        std::span<const std::uint16_t> initial_conv_bits,
                        std::span<const float> initial_recurrence) {
    if (x_bits.size() != static_cast<std::size_t>(kHidden) ||
        initial_conv_bits.size() != static_cast<std::size_t>(kQkvRows) * 3 ||
        initial_recurrence.size() !=
            static_cast<std::size_t>(kValueHeads) * kHeadDim * kHeadDim) {
        throw std::logic_error("Qwen4 real GDN oracle received malformed represented state");
    }
    std::vector<double> x(x_bits.size());
    std::transform(x_bits.begin(), x_bits.end(), x.begin(),
                   [](std::uint16_t bits) { return static_cast<double>(bf16_to_f32(bits)); });
    std::vector<double> conv_in(initial_conv_bits.size());
    std::transform(initial_conv_bits.begin(), initial_conv_bits.end(), conv_in.begin(),
                   [](std::uint16_t bits) { return static_cast<double>(bf16_to_f32(bits)); });

    const std::vector<double> raw =
        quant_project(weights.qkv, weights.input_qtype, kQkvRows, kHidden, x, true);
    const std::vector<double> z =
        quant_project(weights.z, weights.input_qtype, kValueRows, kHidden, x, true);
    const std::vector<double> a = fp32_project(weights.a, kValueHeads, x);
    const std::vector<double> b = fp32_project(weights.b, kValueHeads, x);

    std::vector<double> q(static_cast<std::size_t>(kQkRows));
    std::vector<double> k(static_cast<std::size_t>(kQkRows));
    std::vector<double> v(static_cast<std::size_t>(kValueRows));
    std::vector<double> conv_out(static_cast<std::size_t>(kQkvRows) * 3);
    for (std::int32_t channel = 0; channel < kQkvRows; ++channel) {
        const double convolution =
            static_cast<double>(weights.conv[static_cast<std::size_t>(channel) * 4]) *
                conv_in[channel] +
            static_cast<double>(weights.conv[static_cast<std::size_t>(channel) * 4 + 1]) *
                conv_in[kQkvRows + channel] +
            static_cast<double>(weights.conv[static_cast<std::size_t>(channel) * 4 + 2]) *
                conv_in[2 * kQkvRows + channel] +
            static_cast<double>(weights.conv[static_cast<std::size_t>(channel) * 4 + 3]) *
                raw[channel];
        const double represented = represented_bf16(silu(convolution));
        if (channel < kQkRows) {
            q[channel] = represented;
        } else if (channel < 2 * kQkRows) {
            k[channel - kQkRows] = represented;
        } else {
            v[channel - 2 * kQkRows] = represented;
        }
        conv_out[channel] = conv_in[kQkvRows + channel];
        conv_out[kQkvRows + channel] = conv_in[2 * kQkvRows + channel];
        conv_out[2 * kQkvRows + channel] = raw[channel];
    }

    std::vector<double> q_norm(static_cast<std::size_t>(kQkRows));
    std::vector<double> k_norm(static_cast<std::size_t>(kQkRows));
    for (std::int32_t head = 0; head < kQkHeads; ++head) {
        const std::size_t base = static_cast<std::size_t>(head) * kHeadDim;
        double q_square_sum = 0.0;
        double k_square_sum = 0.0;
        for (std::int32_t dimension = 0; dimension < kHeadDim; ++dimension) {
            q_square_sum += q[base + dimension] * q[base + dimension];
            k_square_sum += k[base + dimension] * k[base + dimension];
        }
        const double q_inverse = 1.0 / std::sqrt(q_square_sum + 1.0e-6);
        const double k_inverse = 1.0 / std::sqrt(k_square_sum + 1.0e-6);
        for (std::int32_t dimension = 0; dimension < kHeadDim; ++dimension) {
            q_norm[base + dimension] = q[base + dimension] * q_inverse;
            k_norm[base + dimension] = k[base + dimension] * k_inverse;
        }
    }

    std::vector<double> recurrence(initial_recurrence.begin(), initial_recurrence.end());
    std::vector<double> recurrent(static_cast<std::size_t>(kValueRows));
    std::array<double, kHeadDim> delta{};
    const double query_scale = 1.0 / std::sqrt(static_cast<double>(kHeadDim));
    for (std::int32_t head = 0; head < kValueHeads; ++head) {
        const std::int32_t qk_head = head % kQkHeads;
        const std::size_t qk_base = static_cast<std::size_t>(qk_head) * kHeadDim;
        const std::size_t value_base = static_cast<std::size_t>(head) * kHeadDim;
        const std::size_t state_base = static_cast<std::size_t>(head) * kHeadDim * kHeadDim;
        const double g = static_cast<double>(static_cast<float>(
            static_cast<double>(weights.ssm_a[head]) *
            softplus(a[head] + static_cast<double>(weights.dt_bias[head]))));
        const double beta = static_cast<double>(static_cast<float>(sigmoid(b[head])));
        const double alpha = std::exp(g);
        for (std::int32_t row = 0; row < kHeadDim; ++row) {
            double dot = 0.0;
            const std::size_t row_base = state_base + static_cast<std::size_t>(row) * kHeadDim;
            for (std::int32_t column = 0; column < kHeadDim; ++column) {
                dot += recurrence[row_base + column] * k_norm[qk_base + column];
            }
            delta[row] = beta * (v[value_base + row] - alpha * dot);
        }
        for (std::int32_t row = 0; row < kHeadDim; ++row) {
            const std::size_t row_base = state_base + static_cast<std::size_t>(row) * kHeadDim;
            for (std::int32_t column = 0; column < kHeadDim; ++column) {
                recurrence[row_base + column] = static_cast<double>(static_cast<float>(
                    alpha * recurrence[row_base + column] +
                    delta[row] * k_norm[qk_base + column]));
            }
            double dot = 0.0;
            for (std::int32_t column = 0; column < kHeadDim; ++column) {
                dot += recurrence[row_base + column] * q_norm[qk_base + column];
            }
            recurrent[value_base + row] = represented_bf16(query_scale * dot);
        }
    }

    std::vector<double> normalized_gated(static_cast<std::size_t>(kValueRows));
    for (std::int32_t head = 0; head < kValueHeads; ++head) {
        const std::size_t base = static_cast<std::size_t>(head) * kHeadDim;
        double square_sum = 0.0;
        for (std::int32_t dimension = 0; dimension < kHeadDim; ++dimension) {
            square_sum += recurrent[base + dimension] * recurrent[base + dimension];
        }
        const double inverse_rms =
            1.0 / std::sqrt(square_sum / static_cast<double>(kHeadDim) + 1.0e-6);
        for (std::int32_t dimension = 0; dimension < kHeadDim; ++dimension) {
            normalized_gated[base + dimension] = represented_bf16(
                recurrent[base + dimension] * inverse_rms * weights.norm[dimension] *
                sigmoid(z[base + dimension]));
        }
    }
    const std::vector<double> output = quant_project(
        weights.output, QType::GGML_Q6_K, kHidden, kValueRows, normalized_gated, false);
    return {output, conv_out, recurrence};
}

int run_layer(const char* label, const ops::GatedDeltaNetLayerWeights& weights,
              QType input_qtype, std::span<const std::uint16_t> x_bits,
              DeviceContext& device) {
    const HostWeights host_weights = copy_weights(weights, input_qtype);
    const std::vector<std::uint16_t> initial_conv = make_conv_state();
    const std::vector<float> initial_recurrence = make_recurrence_state();
    const OracleResult expected =
        gdn_oracle(host_weights, x_bits, initial_conv, initial_recurrence);

    DeviceBuffer device_x = to_device(std::vector<std::uint16_t>(x_bits.begin(), x_bits.end()));
    DeviceBuffer device_conv_in = to_device(initial_conv);
    DeviceBuffer device_recurrence_in = to_device(initial_recurrence);
    GuardedDeviceBuffer device_conv_out(initial_conv.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_recurrence_out(initial_recurrence.size() * sizeof(float));
    GuardedDeviceBuffer device_output(static_cast<std::size_t>(kHidden) * sizeof(std::uint16_t));
    device_conv_out.fill(0xcd);
    device_recurrence_out.fill(0xcd);
    device_output.fill(0xcd);
    Tensor x_tensor(device_x.p, DType::BF16, {kHidden});
    Tensor conv_in_tensor(device_conv_in.p, DType::BF16, {kQkvRows, 3});
    Tensor conv_out_tensor(device_conv_out.data(), DType::BF16, {kQkvRows, 3});
    Tensor recurrence_in_tensor(device_recurrence_in.p, DType::FP32,
                                {kHeadDim, kHeadDim, kValueHeads});
    Tensor recurrence_out_tensor(device_recurrence_out.data(), DType::FP32,
                                 {kHeadDim, kHeadDim, kValueHeads});
    Tensor output_tensor(device_output.data(), DType::BF16, {kHidden});
    WorkspaceArena workspace(ops::gated_delta_net_layer_workspace_capacity_bytes());
    ops::gated_delta_net_layer(x_tensor, weights, conv_in_tensor, conv_out_tensor,
                               recurrence_in_tensor, recurrence_out_tensor, output_tensor,
                               workspace, device.stream);
    device.synchronize();

    int failures = verify_reduction(std::string(label) + " output",
                                    from_device_bf16(device_output.data(), kHidden),
                                    expected.output, kOutputCriterion);
    const std::vector<std::uint16_t> conv_values =
        from_device<std::uint16_t>(device_conv_out.data(), initial_conv.size());
    const std::size_t column_values = static_cast<std::size_t>(kQkvRows);
    failures += verify_exact(
        (std::string(label) + " exact shifted convolution history").c_str(),
        std::vector<std::uint16_t>(conv_values.begin(),
                                   conv_values.begin() + 2 * column_values),
        std::vector<std::uint16_t>(initial_conv.begin() + column_values,
                                   initial_conv.end()));
    std::vector<double> projected_conv(column_values);
    std::transform(conv_values.begin() + 2 * column_values, conv_values.end(),
                   projected_conv.begin(), [](std::uint16_t bits) {
                       return static_cast<double>(bf16_to_f32(bits));
                   });
    failures += verify_reduction(
        std::string(label) + " projected convolution state", projected_conv,
        std::span<const double>(expected.conv_state).subspan(2 * column_values),
        kConvCriterion);
    // The recurrence state crosses ideal-FP64 oracle dots and natural production FP32 reductions,
    // so it retains the established complete-Op numerical criterion.
    const std::vector<float> recurrence_values =
        from_device<float>(device_recurrence_out.data(), initial_recurrence.size());
    failures += verify_reduction(
        std::string(label) + " recurrence state",
        std::vector<double>(recurrence_values.begin(), recurrence_values.end()),
        expected.recurrence_state, kRecurrenceCriterion);
    failures += device_output.verify_guards(std::string(label) + " output");
    failures += device_conv_out.verify_guards(std::string(label) + " convolution state");
    failures += device_recurrence_out.verify_guards(std::string(label) + " recurrence state");
    return failures;
}

std::vector<std::uint16_t> layer_zero_program_x(const verifier::LoadedModel& model,
                                                DeviceContext& device) {
    // PLE is injected only before layer 1, so a reset State plus token embedding is the exact
    // layer-0 residual boundary. The attention GR result is the represented public GDN x.
    verifier::State state;
    state.reset(device.stream);
    state.embed_token(model.view().token_embedding, 48, device.stream);
    GuardedDeviceBuffer device_x(static_cast<std::size_t>(kHidden) * sizeof(std::uint16_t));
    device_x.fill(0xcd);
    Tensor x_tensor(device_x.data(), DType::BF16, {kHidden});
    WorkspaceArena workspace(ops::gated_residual_workspace_capacity_bytes());
    const verifier::GrWeights& gr = model.view().layers[0].attention_gr;
    ops::gated_residual_read(state.residual(), gr.norm, gr.down, gr.up, x_tensor, workspace,
                             device.stream);
    device.synchronize();
    if (device_x.verify_guards("Qwen4 real layer-0 GDN x") != 0) {
        throw std::logic_error("Qwen4 layer-0 attention GR overwrote its x fixture");
    }
    return from_device<std::uint16_t>(device_x.data(), kHidden);
}

std::vector<std::uint16_t> layer_two_program_x(const verifier::LoadedModel& model,
                                               DeviceContext& device) {
    // The diagnostic snapshot is the represented residual after layer 1's FFN injection. Copy it
    // while the TokenResultView is valid, then evaluate layer 2's attention GR to obtain the exact
    // public x boundary. No GDN output or private GDN intermediate is used as an oracle input.
    GuardedDeviceBuffer device_x(static_cast<std::size_t>(kHidden) * sizeof(std::uint16_t));
    device_x.fill(0xcd);
    {
        verifier::Program program(model, device, verifier::DiagnosticSnapshots::Enabled);
        program.reset();
        const verifier::TokenResultView result = program.execute_token(48, 16451);
        if (result.gr.size() != verifier::kLayerCount || result.gr[1].layer != 1 ||
            result.gr[1].ffn_residual.dtype != DType::BF16 ||
            result.gr[1].ffn_residual.ne[0] != kHidden ||
            result.gr[1].ffn_residual.ne[1] != 4) {
            throw std::logic_error("Qwen4 Program layer-1 FFN residual snapshot changed");
        }
        DeviceBuffer layer_one_residual(static_cast<std::size_t>(kHidden) * kBranches *
                                        sizeof(std::uint16_t));
        CUDA_CHECK(cudaMemcpyAsync(layer_one_residual.p, result.gr[1].ffn_residual.data,
                                   layer_one_residual.bytes, cudaMemcpyDeviceToDevice,
                                   device.stream));
        Tensor residual_tensor(layer_one_residual.p, DType::BF16, {kHidden, kBranches});
        Tensor x_tensor(device_x.data(), DType::BF16, {kHidden});
        WorkspaceArena workspace(ops::gated_residual_workspace_capacity_bytes());
        const verifier::GrWeights& gr = model.view().layers[2].attention_gr;
        ops::gated_residual_read(residual_tensor, gr.norm, gr.down, gr.up, x_tensor, workspace,
                                 device.stream);
        device.synchronize();
    }
    if (device_x.verify_guards("Qwen4 real layer-2 GDN x") != 0) {
        throw std::logic_error("Qwen4 layer-2 attention GR overwrote its x fixture");
    }
    return from_device<std::uint16_t>(device_x.data(), kHidden);
}

} // namespace

int ninfer::test::qwen4::real_oracle::run_gdn_cell(const verifier::LoadedModel& model,
                                                   DeviceContext& device) {
    if (!model.view().layers[0].gdn || !model.view().layers[2].gdn) {
        throw std::logic_error("Qwen4 real GDN layer schedule changed");
    }
    const std::vector<std::uint16_t> layer_zero_x = layer_zero_program_x(model, device);
    const std::vector<std::uint16_t> layer_two_x = layer_two_program_x(model, device);
    int failures = run_layer("Qwen4 real layer-0 Q5_K GDN", *model.view().layers[0].gdn,
                             QType::GGML_Q5_K, layer_zero_x, device);
    failures += run_layer("Qwen4 real layer-2 Q6_K GDN", *model.view().layers[2].gdn,
                          QType::GGML_Q6_K, layer_two_x, device);
    std::cout << (failures == 0 ? "OK" : "FAIL") << " qwen4_real_gdn_oracle_cell\n";
    return failures;
}
