#include "ninfer/ops/gdn_gating_proj.h"

#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <initializer_list>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

struct Geometry {
    const char* label;
    std::int32_t hidden;
    std::int32_t heads;
    bool parent_weight;
};

constexpr Geometry kQwen27{"qwen3_6_27b", 5120, 48, false};
constexpr Geometry kQwen35{"qwen3_6_35b_a3b", 2048, 32, true};

constexpr ReductionCriterion kGdnProjectionFp32{/*relative_l2=*/1.4e-6,
                                                /*gross_absolute=*/5.0e-7,
                                                /*gross_relative_to_max_reference=*/2.5e-6};
constexpr ReductionCriterion kGdnNormOutputBf16{/*relative_l2=*/1.75e-3,
                                                /*gross_absolute=*/1.0e-4,
                                                /*gross_relative_to_max_reference=*/4.0e-3};
constexpr ReductionCriterion kGdnNormControlFp32{/*relative_l2=*/8.0e-4,
                                                 /*gross_absolute=*/1.5e-4,
                                                 /*gross_relative_to_max_reference=*/1.05e-3};

double softplus(double value) {
    return std::max(value, 0.0) + std::log1p(std::exp(-std::abs(value)));
}

double sigmoid(double value) {
    if (value >= 0.0) { return 1.0 / (1.0 + std::exp(-value)); }
    const double e = std::exp(value);
    return e / (1.0 + e);
}

std::vector<std::uint16_t> bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { bits[i] = f32_to_bf16(values[i]); }
    return bits;
}

std::vector<double> read_fp32(const void* device, std::size_t elements) {
    const std::vector<float> values = from_device<float>(device, elements);
    return {values.begin(), values.end()};
}

int verify_normwise(const std::string& label, const std::vector<double>& actual,
                    const std::vector<double>& expected, const ReductionCriterion& criterion) {
    return verify_reduction(label.c_str(), actual, expected, criterion);
}

Weight bf16_weight(void* data, std::int32_t rows, std::int32_t hidden) {
    Weight weight{};
    weight.qtype           = QType::BF16_CTRL;
    weight.layout          = QuantLayout::Contiguous;
    weight.payload         = data;
    weight.payload_bytes   = static_cast<std::uint64_t>(rows) * hidden * sizeof(std::uint16_t);
    weight.qdata           = data;
    weight.ndim            = 2;
    weight.shape[0]        = rows;
    weight.shape[1]        = hidden;
    weight.padded_shape[0] = rows;
    weight.padded_shape[1] = hidden;
    weight.n               = rows;
    weight.k               = hidden;
    return weight;
}

std::vector<std::int32_t> oracle_tokens(std::int32_t tokens) {
    std::vector<std::int32_t> selected;
    if (tokens <= 128) {
        selected.reserve(static_cast<std::size_t>(tokens));
        for (std::int32_t token = 0; token < tokens; ++token) { selected.push_back(token); }
    } else {
        selected = {0, tokens / 2, tokens - 1};
    }
    return selected;
}

void projection_oracle(const Geometry& geometry, const std::vector<float>& x,
                       const std::vector<float>& a_weight, const std::vector<float>& b_weight,
                       const std::vector<float>& a_log, const std::vector<float>& dt_bias,
                       const std::vector<std::int32_t>& selected_tokens, std::vector<double>& g,
                       std::vector<double>& beta) {
    const std::size_t output_elements =
        static_cast<std::size_t>(geometry.heads) * selected_tokens.size();
    g.resize(output_elements);
    beta.resize(output_elements);

    for (std::size_t sample = 0; sample < selected_tokens.size(); ++sample) {
        const std::size_t x_base =
            static_cast<std::size_t>(selected_tokens[sample]) * geometry.hidden;
        for (std::int32_t head = 0; head < geometry.heads; ++head) {
            const std::size_t weight_base = static_cast<std::size_t>(head) * geometry.hidden;
            double projected_a            = 0.0;
            double projected_b            = 0.0;
            for (std::int32_t k = 0; k < geometry.hidden; ++k) {
                const double value = static_cast<double>(x[x_base + static_cast<std::size_t>(k)]);
                projected_a +=
                    static_cast<double>(a_weight[weight_base + static_cast<std::size_t>(k)]) *
                    value;
                projected_b +=
                    static_cast<double>(b_weight[weight_base + static_cast<std::size_t>(k)]) *
                    value;
            }
            const std::size_t output = sample * geometry.heads + head;
            g[output]                = -std::exp(static_cast<double>(a_log[head])) *
                        softplus(projected_a + static_cast<double>(dt_bias[head]));
            beta[output] = sigmoid(projected_b);
        }
    }
}

void norm_projection_oracle(const Geometry& geometry, const std::vector<float>& x,
                            const std::vector<float>& norm_weight,
                            const std::vector<float>& a_weight, const std::vector<float>& b_weight,
                            const std::vector<float>& a_log, const std::vector<float>& dt_bias,
                            std::int32_t tokens, double eps, std::vector<double>& h,
                            std::vector<double>& g, std::vector<double>& beta) {
    h.resize(static_cast<std::size_t>(geometry.hidden) * tokens);
    g.resize(static_cast<std::size_t>(geometry.heads) * tokens);
    beta.resize(static_cast<std::size_t>(geometry.heads) * tokens);
    std::vector<double> normalized(static_cast<std::size_t>(geometry.hidden));

    for (std::int32_t token = 0; token < tokens; ++token) {
        const std::size_t x_base = static_cast<std::size_t>(token) * geometry.hidden;
        double sum_squares       = 0.0;
        for (std::int32_t k = 0; k < geometry.hidden; ++k) {
            const double value = static_cast<double>(x[x_base + static_cast<std::size_t>(k)]);
            sum_squares += value * value;
        }
        const double inverse_rms =
            1.0 / std::sqrt(sum_squares / static_cast<double>(geometry.hidden) + eps);
        for (std::int32_t k = 0; k < geometry.hidden; ++k) {
            const double value = static_cast<double>(x[x_base + static_cast<std::size_t>(k)]) *
                                 inverse_rms * (1.0 + static_cast<double>(norm_weight[k]));
            normalized[static_cast<std::size_t>(k)] = value;
            h[x_base + static_cast<std::size_t>(k)] = value;
        }

        for (std::int32_t head = 0; head < geometry.heads; ++head) {
            const std::size_t weight_base = static_cast<std::size_t>(head) * geometry.hidden;
            double projected_a            = 0.0;
            double projected_b            = 0.0;
            for (std::int32_t k = 0; k < geometry.hidden; ++k) {
                const double value = normalized[static_cast<std::size_t>(k)];
                projected_a +=
                    static_cast<double>(a_weight[weight_base + static_cast<std::size_t>(k)]) *
                    value;
                projected_b +=
                    static_cast<double>(b_weight[weight_base + static_cast<std::size_t>(k)]) *
                    value;
            }
            const std::size_t output = static_cast<std::size_t>(token) * geometry.heads + head;
            g[output]                = -std::exp(static_cast<double>(a_log[head])) *
                        softplus(projected_a + static_cast<double>(dt_bias[head]));
            beta[output] = sigmoid(projected_b);
        }
    }
}

std::vector<double> select_tokens(const std::vector<double>& full,
                                  const std::vector<std::int32_t>& selected_tokens,
                                  std::int32_t rows) {
    std::vector<double> selected(static_cast<std::size_t>(rows) * selected_tokens.size());
    for (std::size_t sample = 0; sample < selected_tokens.size(); ++sample) {
        const std::size_t source = static_cast<std::size_t>(selected_tokens[sample]) * rows;
        std::copy_n(full.begin() + static_cast<std::ptrdiff_t>(source), rows,
                    selected.begin() + static_cast<std::ptrdiff_t>(sample * rows));
    }
    return selected;
}

int require_all_finite(const std::string& label, const std::vector<double>& values) {
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (!std::isfinite(values[i])) {
            std::cerr << label << ": output element " << i
                      << " was not written to a finite value\n";
            return 1;
        }
    }
    return 0;
}

int verify_inputs_unchanged(const std::string& label, const DeviceBuffer& device_x,
                            const std::vector<std::uint16_t>& x_bits,
                            const DeviceBuffer& device_weight,
                            const std::vector<std::uint16_t>& weight_bits,
                            const DeviceBuffer& device_a_log, const std::vector<float>& a_log,
                            const DeviceBuffer& device_dt_bias, const std::vector<float>& dt_bias) {
    int failures = 0;
    failures += verify_exact((label + " x immutable").c_str(),
                             from_device<std::uint16_t>(device_x, x_bits.size()), x_bits);
    failures +=
        verify_exact((label + " weight immutable").c_str(),
                     from_device<std::uint16_t>(device_weight, weight_bits.size()), weight_bits);
    failures += verify_exact((label + " A_log immutable").c_str(),
                             from_device<float>(device_a_log, a_log.size()), a_log);
    failures += verify_exact((label + " dt_bias immutable").c_str(),
                             from_device<float>(device_dt_bias, dt_bias.size()), dt_bias);
    return failures;
}

int run_projection_case(const Geometry& geometry, std::int32_t tokens, std::uint32_t seed) {
    std::vector<float> x(static_cast<std::size_t>(geometry.hidden) * tokens);
    std::vector<float> a_weight(static_cast<std::size_t>(geometry.heads) * geometry.hidden);
    std::vector<float> b_weight(static_cast<std::size_t>(geometry.heads) * geometry.hidden);
    std::vector<float> a_log(geometry.heads), dt_bias(geometry.heads);
    fill_uniform(x, seed, -1.0F, 1.0F);
    fill_uniform(a_weight, seed + 1u, -0.015F, 0.015F);
    fill_uniform(b_weight, seed + 2u, -0.015F, 0.015F);
    fill_uniform(a_log, seed + 3u, -2.0F, 1.0F);
    fill_uniform(dt_bias, seed + 4u, -1.0F, 1.0F);
    round_to_bf16(x);
    round_to_bf16(a_weight);
    round_to_bf16(b_weight);

    const std::vector<std::int32_t> selected = oracle_tokens(tokens);
    std::vector<double> reference_g, reference_beta;
    projection_oracle(geometry, x, a_weight, b_weight, a_log, dt_bias, selected, reference_g,
                      reference_beta);

    const std::vector<std::uint16_t> x_bits        = bf16_bits(x);
    std::vector<std::uint16_t> weight_bits         = bf16_bits(a_weight);
    const std::vector<std::uint16_t> b_weight_bits = bf16_bits(b_weight);
    if (geometry.parent_weight) {
        weight_bits.insert(weight_bits.end(), b_weight_bits.begin(), b_weight_bits.end());
    }
    DeviceBuffer device_x      = to_device(x_bits);
    DeviceBuffer device_weight = to_device(weight_bits);
    DeviceBuffer device_b_weight;
    if (!geometry.parent_weight) { device_b_weight = to_device(b_weight_bits); }
    DeviceBuffer device_a_log         = to_device(a_log);
    DeviceBuffer device_dt_bias       = to_device(dt_bias);
    const std::size_t output_elements = static_cast<std::size_t>(geometry.heads) * tokens;
    GuardedDeviceBuffer device_g(output_elements * sizeof(float));
    GuardedDeviceBuffer device_beta(output_elements * sizeof(float));
    device_g.fill(0xff);
    device_beta.fill(0xff);

    Tensor tensor_x(device_x.p, DType::BF16, {geometry.hidden, tokens});
    Tensor tensor_a_log(device_a_log.p, DType::FP32, {geometry.heads});
    Tensor tensor_dt_bias(device_dt_bias.p, DType::FP32, {geometry.heads});
    Tensor tensor_g(device_g.data(), DType::FP32, {geometry.heads, tokens});
    Tensor tensor_beta(device_beta.data(), DType::FP32, {geometry.heads, tokens});
    const std::size_t workspace_bytes = ops::gdn_gating_proj_workspace_capacity_bytes(
        geometry.heads, geometry.hidden, tokens, tokens);
    WorkspaceArena workspace(std::max<std::size_t>(256, workspace_bytes));

    if (geometry.parent_weight) {
        Weight parent = bf16_weight(device_weight.p, 2 * geometry.heads, geometry.hidden);
        ops::gdn_gating_proj(tensor_x, parent, tensor_a_log, tensor_dt_bias, workspace, tensor_g,
                             tensor_beta, nullptr);
    } else {
        Weight weight_a = bf16_weight(device_weight.p, geometry.heads, geometry.hidden);
        Weight weight_b = bf16_weight(device_b_weight.p, geometry.heads, geometry.hidden);
        ops::gdn_gating_proj(tensor_x, weight_a, weight_b, tensor_a_log, tensor_dt_bias, workspace,
                             tensor_g, tensor_beta, nullptr);
    }
    cuda_synchronize();

    const std::vector<double> full_g    = read_fp32(device_g.data(), output_elements);
    const std::vector<double> full_beta = read_fp32(device_beta.data(), output_elements);
    const std::string label =
        std::string("gdn_gating_proj ") + geometry.label + " T=" + std::to_string(tokens);
    int failures = 0;
    failures += require_all_finite(label + " g", full_g);
    failures += require_all_finite(label + " beta", full_beta);
    failures += verify_normwise(label + " g", select_tokens(full_g, selected, geometry.heads),
                                reference_g, kGdnProjectionFp32);
    failures += verify_normwise(label + " beta", select_tokens(full_beta, selected, geometry.heads),
                                reference_beta, kGdnProjectionFp32);
    failures += device_g.verify_guards((label + " g").c_str());
    failures += device_beta.verify_guards((label + " beta").c_str());
    failures += verify_inputs_unchanged(label, device_x, x_bits, device_weight, weight_bits,
                                        device_a_log, a_log, device_dt_bias, dt_bias);
    if (!geometry.parent_weight) {
        failures += verify_exact((label + " b_weight immutable").c_str(),
                                 from_device<std::uint16_t>(device_b_weight, b_weight_bits.size()),
                                 b_weight_bits);
    }
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    return failures;
}

int run_packed_column0_matches_decode(const Geometry& geometry, std::int32_t tokens,
                                      std::uint32_t seed) {
    std::vector<float> x(static_cast<std::size_t>(geometry.hidden) * tokens);
    std::vector<float> a_weight(static_cast<std::size_t>(geometry.heads) * geometry.hidden);
    std::vector<float> b_weight(static_cast<std::size_t>(geometry.heads) * geometry.hidden);
    std::vector<float> a_log(geometry.heads), dt_bias(geometry.heads);
    fill_uniform(x, seed, -1.0F, 1.0F);
    fill_uniform(a_weight, seed + 1u, -0.015F, 0.015F);
    fill_uniform(b_weight, seed + 2u, -0.015F, 0.015F);
    fill_uniform(a_log, seed + 3u, -2.0F, 1.0F);
    fill_uniform(dt_bias, seed + 4u, -1.0F, 1.0F);
    round_to_bf16(x);
    round_to_bf16(a_weight);
    round_to_bf16(b_weight);

    const std::vector<std::uint16_t> x_bits        = bf16_bits(x);
    const std::vector<std::uint16_t> a_bits        = bf16_bits(a_weight);
    const std::vector<std::uint16_t> b_bits        = bf16_bits(b_weight);
    DeviceBuffer device_x                          = to_device(x_bits);
    DeviceBuffer device_a                          = to_device(a_bits);
    DeviceBuffer device_b                          = to_device(b_bits);
    DeviceBuffer device_a_log                      = to_device(a_log);
    DeviceBuffer device_dt_bias                    = to_device(dt_bias);
    const std::size_t packed_elements = static_cast<std::size_t>(geometry.heads) * tokens;
    GuardedDeviceBuffer packed_g(packed_elements * sizeof(float));
    GuardedDeviceBuffer packed_beta(packed_elements * sizeof(float));
    GuardedDeviceBuffer decode_g(static_cast<std::size_t>(geometry.heads) * sizeof(float));
    GuardedDeviceBuffer decode_beta(static_cast<std::size_t>(geometry.heads) * sizeof(float));
    packed_g.fill(0xff);
    packed_beta.fill(0xff);
    decode_g.fill(0xff);
    decode_beta.fill(0xff);

    Tensor tensor_x(device_x.p, DType::BF16, {geometry.hidden, tokens});
    Tensor tensor_x1(device_x.p, DType::BF16, {geometry.hidden, 1});
    Tensor tensor_a_log(device_a_log.p, DType::FP32, {geometry.heads});
    Tensor tensor_dt_bias(device_dt_bias.p, DType::FP32, {geometry.heads});
    Tensor tensor_pg(packed_g.data(), DType::FP32, {geometry.heads, tokens});
    Tensor tensor_pb(packed_beta.data(), DType::FP32, {geometry.heads, tokens});
    Tensor tensor_dg(decode_g.data(), DType::FP32, {geometry.heads, 1});
    Tensor tensor_db(decode_beta.data(), DType::FP32, {geometry.heads, 1});
    Weight weight_a = bf16_weight(device_a.p, geometry.heads, geometry.hidden);
    Weight weight_b = bf16_weight(device_b.p, geometry.heads, geometry.hidden);

    const std::size_t packed_ws = ops::gdn_gating_proj_workspace_capacity_bytes(
        geometry.heads, geometry.hidden, tokens, tokens);
    const std::size_t decode_ws =
        ops::gdn_gating_proj_workspace_capacity_bytes(geometry.heads, geometry.hidden, 1, 1);
    WorkspaceArena packed_workspace(std::max<std::size_t>(256, packed_ws));
    WorkspaceArena decode_workspace(std::max<std::size_t>(256, decode_ws));
    ops::gdn_gating_proj(tensor_x, weight_a, weight_b, tensor_a_log, tensor_dt_bias,
                         packed_workspace, tensor_pg, tensor_pb, nullptr);
    ops::gdn_gating_proj(tensor_x1, weight_a, weight_b, tensor_a_log, tensor_dt_bias,
                         decode_workspace, tensor_dg, tensor_db, nullptr);
    cuda_synchronize();

    const std::vector<float> packed_g_host =
        from_device<float>(packed_g.data(), packed_elements);
    const std::vector<float> packed_beta_host =
        from_device<float>(packed_beta.data(), packed_elements);
    const std::vector<float> decode_g_host =
        from_device<float>(decode_g.data(), static_cast<std::size_t>(geometry.heads));
    const std::vector<float> decode_beta_host =
        from_device<float>(decode_beta.data(), static_cast<std::size_t>(geometry.heads));
    const std::string label = std::string("gdn_gating_proj packed-col0 ") + geometry.label +
                              " T=" + std::to_string(tokens);
    int failures = 0;
    failures += verify_exact(
        (label + " g").c_str(),
        std::vector<float>(packed_g_host.begin(),
                           packed_g_host.begin() + geometry.heads),
        decode_g_host);
    failures += verify_exact(
        (label + " beta").c_str(),
        std::vector<float>(packed_beta_host.begin(),
                           packed_beta_host.begin() + geometry.heads),
        decode_beta_host);
    return failures;
}

int run_packed_columns_match_decode(const Geometry& geometry, std::int32_t tokens,
                                    std::uint32_t seed) {
    std::vector<float> x(static_cast<std::size_t>(geometry.hidden) * tokens);
    std::vector<float> a_weight(static_cast<std::size_t>(geometry.heads) * geometry.hidden);
    std::vector<float> b_weight(static_cast<std::size_t>(geometry.heads) * geometry.hidden);
    std::vector<float> a_log(geometry.heads), dt_bias(geometry.heads);
    fill_uniform(x, seed, -1.0F, 1.0F);
    fill_uniform(a_weight, seed + 1u, -0.015F, 0.015F);
    fill_uniform(b_weight, seed + 2u, -0.015F, 0.015F);
    fill_uniform(a_log, seed + 3u, -2.0F, 1.0F);
    fill_uniform(dt_bias, seed + 4u, -1.0F, 1.0F);
    round_to_bf16(x);
    round_to_bf16(a_weight);
    round_to_bf16(b_weight);

    const std::vector<std::uint16_t> x_bits = bf16_bits(x);
    const std::vector<std::uint16_t> a_bits = bf16_bits(a_weight);
    const std::vector<std::uint16_t> b_bits = bf16_bits(b_weight);
    DeviceBuffer device_x                   = to_device(x_bits);
    DeviceBuffer device_a                   = to_device(a_bits);
    DeviceBuffer device_b                   = to_device(b_bits);
    DeviceBuffer device_a_log               = to_device(a_log);
    DeviceBuffer device_dt_bias             = to_device(dt_bias);
    const std::size_t packed_elements = static_cast<std::size_t>(geometry.heads) * tokens;
    GuardedDeviceBuffer packed_g(packed_elements * sizeof(float));
    GuardedDeviceBuffer packed_beta(packed_elements * sizeof(float));
    packed_g.fill(0xff);
    packed_beta.fill(0xff);

    Tensor tensor_x(device_x.p, DType::BF16, {geometry.hidden, tokens});
    Tensor tensor_a_log(device_a_log.p, DType::FP32, {geometry.heads});
    Tensor tensor_dt_bias(device_dt_bias.p, DType::FP32, {geometry.heads});
    Tensor tensor_pg(packed_g.data(), DType::FP32, {geometry.heads, tokens});
    Tensor tensor_pb(packed_beta.data(), DType::FP32, {geometry.heads, tokens});
    Weight weight_a = bf16_weight(device_a.p, geometry.heads, geometry.hidden);
    Weight weight_b = bf16_weight(device_b.p, geometry.heads, geometry.hidden);
    const std::size_t packed_ws = ops::gdn_gating_proj_workspace_capacity_bytes(
        geometry.heads, geometry.hidden, tokens, tokens);
    WorkspaceArena packed_workspace(std::max<std::size_t>(256, packed_ws));
    ops::gdn_gating_proj(tensor_x, weight_a, weight_b, tensor_a_log, tensor_dt_bias,
                         packed_workspace, tensor_pg, tensor_pb, nullptr);
    cuda_synchronize();
    const std::vector<float> packed_g_host =
        from_device<float>(packed_g.data(), packed_elements);
    const std::vector<float> packed_beta_host =
        from_device<float>(packed_beta.data(), packed_elements);

    int failures = 0;
    const std::size_t decode_ws =
        ops::gdn_gating_proj_workspace_capacity_bytes(geometry.heads, geometry.hidden, 1, 1);
    for (std::int32_t column = 0; column < tokens; ++column) {
        GuardedDeviceBuffer decode_g(static_cast<std::size_t>(geometry.heads) * sizeof(float));
        GuardedDeviceBuffer decode_beta(static_cast<std::size_t>(geometry.heads) * sizeof(float));
        decode_g.fill(0xff);
        decode_beta.fill(0xff);
        Tensor tensor_x1(
            static_cast<std::byte*>(device_x.p) +
                static_cast<std::size_t>(column) * geometry.hidden * sizeof(std::uint16_t),
            DType::BF16, {geometry.hidden, 1});
        Tensor tensor_dg(decode_g.data(), DType::FP32, {geometry.heads, 1});
        Tensor tensor_db(decode_beta.data(), DType::FP32, {geometry.heads, 1});
        WorkspaceArena decode_workspace(std::max<std::size_t>(256, decode_ws));
        ops::gdn_gating_proj(tensor_x1, weight_a, weight_b, tensor_a_log, tensor_dt_bias,
                             decode_workspace, tensor_dg, tensor_db, nullptr);
        cuda_synchronize();
        const std::vector<float> decode_g_host =
            from_device<float>(decode_g.data(), static_cast<std::size_t>(geometry.heads));
        const std::vector<float> decode_beta_host =
            from_device<float>(decode_beta.data(), static_cast<std::size_t>(geometry.heads));
        const std::size_t offset = static_cast<std::size_t>(column) * geometry.heads;
        const std::string label  = std::string("gdn_gating_proj packed-col ") + geometry.label +
                                  " T=" + std::to_string(tokens) +
                                  " col=" + std::to_string(column);
        failures += verify_exact(
            (label + " g").c_str(),
            std::vector<float>(packed_g_host.begin() + static_cast<std::ptrdiff_t>(offset),
                               packed_g_host.begin() + static_cast<std::ptrdiff_t>(offset) +
                                   geometry.heads),
            decode_g_host);
        failures += verify_exact(
            (label + " beta").c_str(),
            std::vector<float>(packed_beta_host.begin() + static_cast<std::ptrdiff_t>(offset),
                               packed_beta_host.begin() + static_cast<std::ptrdiff_t>(offset) +
                                   geometry.heads),
            decode_beta_host);
        if (failures != 0) { return failures; }
    }
    return failures;
}

int run_norm_packed_columns_match_decode(const Geometry& geometry, std::int32_t tokens,
                                         std::uint32_t seed,
                                         std::int32_t sequence_width = 5) {
    constexpr float kEps = 1.0e-6F;
    std::vector<float> x(static_cast<std::size_t>(geometry.hidden) * tokens);
    std::vector<float> norm_weight(static_cast<std::size_t>(geometry.hidden));
    std::vector<float> a_weight(static_cast<std::size_t>(geometry.heads) * geometry.hidden);
    std::vector<float> b_weight(static_cast<std::size_t>(geometry.heads) * geometry.hidden);
    std::vector<float> a_log(geometry.heads), dt_bias(geometry.heads);
    fill_uniform(x, seed, -1.0F, 1.0F);
    fill_uniform(norm_weight, seed + 1u, -0.2F, 0.2F);
    fill_uniform(a_weight, seed + 2u, -0.015F, 0.015F);
    fill_uniform(b_weight, seed + 3u, -0.015F, 0.015F);
    fill_uniform(a_log, seed + 4u, -2.0F, 1.0F);
    fill_uniform(dt_bias, seed + 5u, -1.0F, 1.0F);
    round_to_bf16(x);
    round_to_bf16(norm_weight);
    round_to_bf16(a_weight);
    round_to_bf16(b_weight);

    const std::vector<std::uint16_t> x_bits           = bf16_bits(x);
    const std::vector<std::uint16_t> norm_weight_bits = bf16_bits(norm_weight);
    const std::vector<std::uint16_t> a_bits           = bf16_bits(a_weight);
    const std::vector<std::uint16_t> b_bits           = bf16_bits(b_weight);
    DeviceBuffer device_x                             = to_device(x_bits);
    DeviceBuffer device_norm_weight                   = to_device(norm_weight_bits);
    DeviceBuffer device_a                             = to_device(a_bits);
    DeviceBuffer device_b                             = to_device(b_bits);
    DeviceBuffer device_a_log                         = to_device(a_log);
    DeviceBuffer device_dt_bias                       = to_device(dt_bias);

    const std::size_t packed_h      = static_cast<std::size_t>(geometry.hidden) * tokens;
    const std::size_t packed_ctrl   = static_cast<std::size_t>(geometry.heads) * tokens;
    GuardedDeviceBuffer packed_h_buf(packed_h * sizeof(std::uint16_t));
    GuardedDeviceBuffer packed_g(packed_ctrl * sizeof(float));
    GuardedDeviceBuffer packed_beta(packed_ctrl * sizeof(float));
    GuardedDeviceBuffer panel_h_buf(packed_h * sizeof(std::uint16_t));
    GuardedDeviceBuffer panel_g(packed_ctrl * sizeof(float));
    GuardedDeviceBuffer panel_beta(packed_ctrl * sizeof(float));
    packed_h_buf.fill(0xff);
    packed_g.fill(0xff);
    packed_beta.fill(0xff);
    panel_h_buf.fill(0xff);
    panel_g.fill(0xff);
    panel_beta.fill(0xff);

    Tensor tensor_x(device_x.p, DType::BF16, {geometry.hidden, tokens});
    Tensor tensor_norm(device_norm_weight.p, DType::BF16, {geometry.hidden});
    Tensor tensor_a_log(device_a_log.p, DType::FP32, {geometry.heads});
    Tensor tensor_dt_bias(device_dt_bias.p, DType::FP32, {geometry.heads});
    Tensor tensor_ph(packed_h_buf.data(), DType::BF16, {geometry.hidden, tokens});
    Tensor tensor_pg(packed_g.data(), DType::FP32, {geometry.heads, tokens});
    Tensor tensor_pb(packed_beta.data(), DType::FP32, {geometry.heads, tokens});
    Tensor tensor_panel_h(panel_h_buf.data(), DType::BF16, {geometry.hidden, tokens});
    Tensor tensor_panel_g(panel_g.data(), DType::FP32, {geometry.heads, tokens});
    Tensor tensor_panel_beta(panel_beta.data(), DType::FP32, {geometry.heads, tokens});
    Weight weight_a = bf16_weight(device_a.p, geometry.heads, geometry.hidden);
    Weight weight_b = bf16_weight(device_b.p, geometry.heads, geometry.hidden);

    const std::int32_t batch = tokens / sequence_width;
    const std::size_t packed_ws =
        ops::gdn_norm_gating_proj_packed_sequences_workspace_capacity_bytes(sequence_width, batch,
                                                                            batch);
    const std::size_t panel_ws = ops::gdn_norm_gating_proj_workspace_capacity_bytes(
        geometry.heads, geometry.hidden, sequence_width, sequence_width);
    WorkspaceArena packed_workspace(std::max<std::size_t>(256, packed_ws));
    ops::gdn_norm_gating_proj_packed_sequences(
        tensor_x, tensor_norm, kEps, weight_a, weight_b, tensor_a_log, tensor_dt_bias,
        packed_workspace, tensor_ph, tensor_pg, tensor_pb, nullptr, sequence_width);
    WorkspaceArena panel_workspace(std::max<std::size_t>(256, panel_ws));
    for (std::int32_t offset = 0; offset < tokens; offset += sequence_width) {
        Tensor panel_h_out    = tensor_panel_h.slice(1, offset, sequence_width);
        Tensor panel_g_out    = tensor_panel_g.slice(1, offset, sequence_width);
        Tensor panel_beta_out = tensor_panel_beta.slice(1, offset, sequence_width);
        ops::gdn_norm_gating_proj(
            tensor_x.slice(1, offset, sequence_width), tensor_norm, kEps, weight_a, weight_b,
            tensor_a_log, tensor_dt_bias, panel_workspace, panel_h_out, panel_g_out, panel_beta_out,
            nullptr);
    }
    cuda_synchronize();

    const std::vector<std::uint16_t> packed_h_host =
        from_device<std::uint16_t>(packed_h_buf.data(), packed_h);
    const std::vector<float> packed_g_host = from_device<float>(packed_g.data(), packed_ctrl);
    const std::vector<float> packed_beta_host =
        from_device<float>(packed_beta.data(), packed_ctrl);
    const std::string label = std::string("gdn_norm_gating_proj packed-panels ") +
                              geometry.label + " T=" + std::to_string(tokens) +
                              " W=" + std::to_string(sequence_width);
    int failures = verify_exact(
        (label + " h").c_str(), packed_h_host,
        from_device<std::uint16_t>(panel_h_buf.data(), packed_h));
    failures += verify_exact((label + " g").c_str(), packed_g_host,
                             from_device<float>(panel_g.data(), packed_ctrl));
    failures += verify_exact((label + " beta").c_str(), packed_beta_host,
                             from_device<float>(panel_beta.data(), packed_ctrl));
    failures += packed_h_buf.verify_guards("gdn_norm_gating_proj packed h");
    failures += packed_g.verify_guards("gdn_norm_gating_proj packed g");
    failures += packed_beta.verify_guards("gdn_norm_gating_proj packed beta");
    failures += panel_h_buf.verify_guards("gdn_norm_gating_proj panel h");
    failures += panel_g.verify_guards("gdn_norm_gating_proj panel g");
    failures += panel_beta.verify_guards("gdn_norm_gating_proj panel beta");
    if (packed_workspace.used() != 0 || packed_workspace.peak_used() != packed_ws) {
        std::cerr << label << ": packed workspace query/execution high-water mismatch\n";
        ++failures;
    }
    if (panel_workspace.used() != 0 || panel_workspace.peak_used() != panel_ws) {
        std::cerr << label << ": panel workspace query/execution high-water mismatch\n";
        ++failures;
    }
    return failures;
}

int run_norm_projection_case(const Geometry& geometry, std::int32_t tokens, std::uint32_t seed,
                             std::int32_t sequence_width = 0) {
    constexpr float kEps = 1.0e-6F;
    std::vector<float> x(static_cast<std::size_t>(geometry.hidden) * tokens);
    std::vector<float> norm_weight(static_cast<std::size_t>(geometry.hidden));
    std::vector<float> a_weight(static_cast<std::size_t>(geometry.heads) * geometry.hidden);
    std::vector<float> b_weight(static_cast<std::size_t>(geometry.heads) * geometry.hidden);
    std::vector<float> a_log(geometry.heads), dt_bias(geometry.heads);
    fill_uniform(x, seed, -1.0F, 1.0F);
    fill_uniform(norm_weight, seed + 1u, -0.2F, 0.2F);
    fill_uniform(a_weight, seed + 2u, -0.015F, 0.015F);
    fill_uniform(b_weight, seed + 3u, -0.015F, 0.015F);
    fill_uniform(a_log, seed + 4u, -2.0F, 1.0F);
    fill_uniform(dt_bias, seed + 5u, -1.0F, 1.0F);
    round_to_bf16(x);
    round_to_bf16(norm_weight);
    round_to_bf16(a_weight);
    round_to_bf16(b_weight);

    std::vector<double> reference_h, reference_g, reference_beta;
    norm_projection_oracle(geometry, x, norm_weight, a_weight, b_weight, a_log, dt_bias, tokens,
                           kEps, reference_h, reference_g, reference_beta);

    const std::vector<std::uint16_t> x_bits           = bf16_bits(x);
    const std::vector<std::uint16_t> norm_weight_bits = bf16_bits(norm_weight);
    std::vector<std::uint16_t> weight_bits            = bf16_bits(a_weight);
    const std::vector<std::uint16_t> b_weight_bits    = bf16_bits(b_weight);
    if (geometry.parent_weight) {
        weight_bits.insert(weight_bits.end(), b_weight_bits.begin(), b_weight_bits.end());
    }
    DeviceBuffer device_x           = to_device(x_bits);
    DeviceBuffer device_norm_weight = to_device(norm_weight_bits);
    DeviceBuffer device_weight      = to_device(weight_bits);
    DeviceBuffer device_b_weight;
    if (!geometry.parent_weight) { device_b_weight = to_device(b_weight_bits); }
    DeviceBuffer device_a_log          = to_device(a_log);
    DeviceBuffer device_dt_bias        = to_device(dt_bias);
    const std::size_t h_elements       = static_cast<std::size_t>(geometry.hidden) * tokens;
    const std::size_t control_elements = static_cast<std::size_t>(geometry.heads) * tokens;
    GuardedDeviceBuffer device_h(h_elements * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_g(control_elements * sizeof(float));
    GuardedDeviceBuffer device_beta(control_elements * sizeof(float));
    device_h.fill(0xff);
    device_g.fill(0xff);
    device_beta.fill(0xff);

    Tensor tensor_x(device_x.p, DType::BF16, {geometry.hidden, tokens});
    Tensor tensor_norm_weight(device_norm_weight.p, DType::BF16, {geometry.hidden});
    Tensor tensor_h(device_h.data(), DType::BF16, {geometry.hidden, tokens});
    Tensor tensor_a_log(device_a_log.p, DType::FP32, {geometry.heads});
    Tensor tensor_dt_bias(device_dt_bias.p, DType::FP32, {geometry.heads});
    Tensor tensor_g(device_g.data(), DType::FP32, {geometry.heads, tokens});
    Tensor tensor_beta(device_beta.data(), DType::FP32, {geometry.heads, tokens});
    const std::size_t workspace_bytes =
        sequence_width > 0
            ? ops::gdn_norm_gating_proj_packed_sequences_workspace_capacity_bytes(
                  sequence_width, tokens / sequence_width, tokens / sequence_width)
            : ops::gdn_norm_gating_proj_workspace_capacity_bytes(geometry.heads, geometry.hidden,
                                                                  tokens, tokens);
    WorkspaceArena workspace(std::max<std::size_t>(256, workspace_bytes));

    if (geometry.parent_weight) {
        Weight parent = bf16_weight(device_weight.p, 2 * geometry.heads, geometry.hidden);
        ops::gdn_norm_gating_proj(tensor_x, tensor_norm_weight, kEps, parent, tensor_a_log,
                                  tensor_dt_bias, workspace, tensor_h, tensor_g, tensor_beta,
                                  nullptr);
    } else {
        Weight weight_a = bf16_weight(device_weight.p, geometry.heads, geometry.hidden);
        Weight weight_b = bf16_weight(device_b_weight.p, geometry.heads, geometry.hidden);
        if (sequence_width > 0) {
            ops::gdn_norm_gating_proj_packed_sequences(
                tensor_x, tensor_norm_weight, kEps, weight_a, weight_b, tensor_a_log,
                tensor_dt_bias, workspace, tensor_h, tensor_g, tensor_beta, nullptr,
                sequence_width);
        } else {
            ops::gdn_norm_gating_proj(tensor_x, tensor_norm_weight, kEps, weight_a, weight_b,
                                      tensor_a_log, tensor_dt_bias, workspace, tensor_h, tensor_g,
                                      tensor_beta, nullptr);
        }
    }
    cuda_synchronize();

    const std::string label = std::string("gdn_norm_gating_proj ") + geometry.label +
                              " T=" + std::to_string(tokens) +
                              (sequence_width > 0 ? " packed-width=" +
                                                        std::to_string(sequence_width)
                                                  : "");
    int failures = 0;
    failures += verify_normwise(label + " h", from_device_bf16(device_h.data(), h_elements),
                                reference_h, kGdnNormOutputBf16);
    failures += verify_normwise(label + " g", read_fp32(device_g.data(), control_elements),
                                reference_g, kGdnNormControlFp32);
    failures += verify_normwise(label + " beta", read_fp32(device_beta.data(), control_elements),
                                reference_beta, kGdnNormControlFp32);
    failures += device_h.verify_guards((label + " h").c_str());
    failures += device_g.verify_guards((label + " g").c_str());
    failures += device_beta.verify_guards((label + " beta").c_str());
    failures += verify_inputs_unchanged(label, device_x, x_bits, device_weight, weight_bits,
                                        device_a_log, a_log, device_dt_bias, dt_bias);
    failures += verify_exact(
        (label + " norm_weight immutable").c_str(),
        from_device<std::uint16_t>(device_norm_weight, norm_weight_bits.size()), norm_weight_bits);
    if (!geometry.parent_weight) {
        failures += verify_exact((label + " b_weight immutable").c_str(),
                                 from_device<std::uint16_t>(device_b_weight, b_weight_bits.size()),
                                 b_weight_bits);
    }
    const bool invalid_workspace =
        workspace.used() != 0 || workspace.peak_used() != workspace_bytes;
    if (invalid_workspace) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    return failures;
}

int verify_workspace_capacity_contract(const Geometry& geometry,
                                       std::initializer_list<std::int32_t> route_endpoints) {
    const std::int32_t last = *std::max_element(route_endpoints.begin(), route_endpoints.end());
    const std::size_t interval =
        ops::gdn_gating_proj_workspace_capacity_bytes(geometry.heads, geometry.hidden, 1, last);
    std::size_t witness = 0;
    for (const std::int32_t tokens : route_endpoints) {
        witness = std::max(witness, ops::gdn_gating_proj_workspace_capacity_bytes(
                                        geometry.heads, geometry.hidden, tokens, tokens));
    }
    int failures = 0;
    if (interval != witness) {
        std::cerr << geometry.label << ": GDN control interval missed a route endpoint\n";
        ++failures;
    }
    const std::size_t norm_interval =
        ops::gdn_norm_gating_proj_workspace_capacity_bytes(geometry.heads, geometry.hidden, 1, 64);
    const std::size_t norm_witness = std::max(
        ops::gdn_norm_gating_proj_workspace_capacity_bytes(geometry.heads, geometry.hidden, 16, 16),
        ops::gdn_norm_gating_proj_workspace_capacity_bytes(geometry.heads, geometry.hidden, 64,
                                                           64));
    if (norm_interval != norm_witness) {
        std::cerr << geometry.label << ": GDN norm/control interval missed a route endpoint\n";
        ++failures;
    }
    return failures;
}

int verify_packed_workspace_rejections() {
    int failures = 0;
    const auto expect_invalid = [&](std::int32_t width, std::int32_t first,
                                    std::int32_t last) {
        try {
            (void)ops::gdn_norm_gating_proj_packed_sequences_workspace_capacity_bytes(width, first,
                                                                                      last);
            std::cerr << "gdn packed workspace accepted invalid W=" << width << " B=" << first
                      << ".." << last << '\n';
            ++failures;
        } catch (const std::invalid_argument&) {
        }
    };
    expect_invalid(0, 1, 1);
    expect_invalid(5, 0, 1);
    expect_invalid(5, 3, 2);
    expect_invalid(5, 1, 5);
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cerr << "FAIL: no usable CUDA device\n";
        return 1;
    }

    int failures = 0;
    failures += verify_workspace_capacity_contract(kQwen27, {1, 16, 1024, 2048, 4096, 4097});
    failures += verify_workspace_capacity_contract(kQwen35, {1, 127, 1024, 2048, 4096, 4097});
    failures += verify_packed_workspace_rejections();

    // Every registered 27B projection route, including predicated and full token tiles.
    // 16 is the last packed-GEMV width (product DFlash2 verify is 12); 17 is MMA split-8.
    for (const std::int32_t tokens : {1, 2, 6, 8, 12, 16, 17, 1024, 1025, 2049, 4097}) {
        failures +=
            run_projection_case(kQwen27, tokens, 0x1000u + static_cast<std::uint32_t>(tokens));
    }
    failures += run_packed_column0_matches_decode(kQwen27, 2, 0x5102u);
    failures += run_packed_column0_matches_decode(kQwen27, 6, 0x5106u);
    failures += run_packed_column0_matches_decode(kQwen27, 8, 0x5108u);
    failures += run_packed_column0_matches_decode(kQwen27, 12, 0x510cu);
    failures += run_packed_column0_matches_decode(kQwen27, 16, 0x5110u);
    failures += run_packed_columns_match_decode(kQwen27, 8, 0x5188u);
    failures += run_packed_columns_match_decode(kQwen27, 12, 0x518cu);
    failures += run_norm_packed_columns_match_decode(kQwen27, 5, 0x6105u);
    failures += run_norm_packed_columns_match_decode(kQwen27, 10, 0x610au);
    failures += run_norm_packed_columns_match_decode(kQwen27, 15, 0x610fu);
    failures += run_norm_packed_columns_match_decode(kQwen27, 20, 0x6114u);
    failures += run_norm_packed_columns_match_decode(kQwen27, 24, 0x6118u, 12);
    failures += run_norm_packed_columns_match_decode(kQwen27, 34, 0x6122u, 17);
    // Every registered 35B projection route and its contiguous-parent storage contract.
    for (const std::int32_t tokens : {1, 127, 128, 1024, 1025, 2049, 4097}) {
        failures +=
            run_projection_case(kQwen35, tokens, 0x2000u + static_cast<std::uint32_t>(tokens));
    }

    // 27B uses the composed implementation; 35B also qualifies both sides of its fused boundary.
    failures += run_norm_projection_case(kQwen27, 1, 0x3001u);
    failures += run_norm_projection_case(kQwen27, 5, 0x3005u);
    failures += run_norm_projection_case(kQwen27, 10, 0x300au, 5);
    failures += run_norm_projection_case(kQwen27, 15, 0x300fu, 5);
    failures += run_norm_projection_case(kQwen27, 20, 0x3014u, 5);
    failures += run_norm_projection_case(kQwen27, 24, 0x3018u, 12);
    failures += run_norm_projection_case(kQwen27, 64, 0x3040u);
    failures += run_norm_projection_case(kQwen35, 1, 0x4001u);
    failures += run_norm_projection_case(kQwen35, 16, 0x4010u);
    failures += run_norm_projection_case(kQwen35, 17, 0x4011u);
    failures += run_norm_projection_case(kQwen35, 64, 0x4040u);

    std::cout << (failures == 0 ? "OK" : "FAIL") << " gdn_gating_proj correctness\n";
    return failures == 0 ? 0 : 1;
}
