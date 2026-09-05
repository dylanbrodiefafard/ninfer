#include "targets/qwen4/real_oracle_common.h"

#include "ninfer/ops/gated_residual.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <stdexcept>
#include <vector>

namespace verifier = ninfer::targets::qwen4::verifier;
namespace ops = ninfer::ops;
using ninfer::DType;
using ninfer::DeviceBuffer;
using ninfer::QType;
using ninfer::Tensor;
using ninfer::Weight;
using ninfer::WorkspaceArena;
using ninfer::test::GuardedDeviceBuffer;
using ninfer::test::ReductionCriterion;
using ninfer::test::bf16_to_f32;
using ninfer::test::f32_to_bf16;
using ninfer::test::from_device_bf16;
using ninfer::test::to_device;
using ninfer::test::verify_reduction;
using namespace ninfer::test::qwen4::real_oracle;

namespace {

constexpr std::int32_t kHidden = 2560;
constexpr std::int32_t kBranches = 4;
constexpr std::int32_t kFlat = kHidden * kBranches;
constexpr std::int32_t kRank = 320;
constexpr std::size_t kQ4RowBytes =
    static_cast<std::size_t>(kHidden) / kQ4BlockValues * kQ4BlockBytes;

// These are the established live-verifier GR criteria from test_gated_residual.cpp. The real
// artifact cell changes only represented inputs and weights, not the implementation profile.
constexpr ReductionCriterion kReadCriterion{/*relative_l2=*/6.0e-3,
                                             /*gross_absolute=*/4.0e-3,
                                             /*gross_relative_to_max_reference=*/5.0e-3};
constexpr ReductionCriterion kScaleCriterion{/*relative_l2=*/3.5e-3,
                                              /*gross_absolute=*/1.5e-3,
                                              /*gross_relative_to_max_reference=*/3.0e-3};

std::vector<double> q8_project(std::span<const std::uint8_t> weight, std::int32_t rows,
                               std::int32_t columns, std::span<const double> input) {
    const std::size_t row_bytes =
        static_cast<std::size_t>(columns) / kQ8BlockValues * kQ8BlockBytes;
    if (weight.size() != static_cast<std::size_t>(rows) * row_bytes ||
        input.size() != static_cast<std::size_t>(columns)) {
        throw std::logic_error("real GR oracle received a malformed Q8_0 matrix");
    }
    std::vector<double> output(static_cast<std::size_t>(rows));
    for (std::int32_t row = 0; row < rows; ++row) {
        const auto* row_data = weight.data() + static_cast<std::size_t>(row) * row_bytes;
        double sum = 0.0;
        for (std::int32_t column = 0; column < columns; ++column) {
            sum += ggml_q8_0_value(row_data, column) * input[static_cast<std::size_t>(column)];
        }
        output[static_cast<std::size_t>(row)] = sum;
    }
    return output;
}

double sigmoid(double value) {
    if (value >= 0.0) { return 1.0 / (1.0 + std::exp(-value)); }
    const double exponential = std::exp(value);
    return exponential / (1.0 + exponential);
}

struct OracleResult {
    std::vector<double> mixed;
    std::vector<double> write_scale;
};

OracleResult gr_oracle(std::span<const float> residual, std::span<const float> norm,
                       std::span<const std::uint8_t> down,
                       std::span<const std::uint8_t> up, std::span<const float> write) {
    if (residual.size() != kFlat || norm.size() != kFlat ||
        write.size() != static_cast<std::size_t>(kBranches) * kFlat) {
        throw std::logic_error("real GR oracle received malformed represented inputs");
    }
    std::vector<double> normalized(kFlat);
    for (std::int32_t branch = 0; branch < kBranches; ++branch) {
        const std::size_t base = static_cast<std::size_t>(branch) * kHidden;
        double sum_squares = 0.0;
        for (std::int32_t dimension = 0; dimension < kHidden; ++dimension) {
            const double value = residual[base + dimension];
            sum_squares += value * value;
        }
        const double inverse_rms = 1.0 / std::sqrt(sum_squares / kHidden + 1.0e-6);
        for (std::int32_t dimension = 0; dimension < kHidden; ++dimension) {
            const std::size_t index = base + dimension;
            normalized[index] = residual[index] * inverse_rms * norm[index];
        }
    }

    std::vector<double> low_rank = q8_project(down, kRank, kFlat, normalized);
    for (double& value : low_rank) { value = (value / 4.0) * sigmoid(value / 4.0); }
    const std::vector<double> gates = q8_project(up, kFlat, kRank, low_rank);

    OracleResult result{std::vector<double>(kHidden), std::vector<double>(kBranches)};
    for (std::int32_t dimension = 0; dimension < kHidden; ++dimension) {
        double mixed = 0.0;
        for (std::int32_t branch = 0; branch < kBranches; ++branch) {
            const std::size_t index =
                static_cast<std::size_t>(branch) * kHidden + dimension;
            mixed += sigmoid(gates[index]) * normalized[index];
        }
        result.mixed[static_cast<std::size_t>(dimension)] = mixed / 4.0;
    }
    for (std::int32_t branch = 0; branch < kBranches; ++branch) {
        double projected = 0.0;
        const std::size_t base = static_cast<std::size_t>(branch) * kFlat;
        for (std::int32_t index = 0; index < kFlat; ++index) {
            projected += write[base + index] * normalized[static_cast<std::size_t>(index)];
        }
        result.write_scale[static_cast<std::size_t>(branch)] =
            2.0 * sigmoid(projected / 4.0);
    }
    return result;
}

std::vector<std::uint16_t> decode_token_zero_residual(const Weight& embedding) {
    if (embedding.qtype != QType::GGML_Q4_K || embedding.qdata == nullptr ||
        embedding.n != 248320 || embedding.k != kHidden ||
        embedding.payload_bytes < kQ4RowBytes) {
        throw std::logic_error("Qwen4 real GR test requires the bound Q4_K embedding");
    }
    const std::vector<std::uint8_t> encoded =
        copy_device_bytes(embedding.qdata, kQ4RowBytes);
    std::vector<std::uint16_t> residual(static_cast<std::size_t>(kFlat));
    for (std::int32_t dimension = 0; dimension < kHidden; ++dimension) {
        const std::uint16_t represented =
            f32_to_bf16(static_cast<float>(ggml_q4_k_value(encoded.data(), dimension)));
        for (std::int32_t branch = 0; branch < kBranches; ++branch) {
            residual[static_cast<std::size_t>(branch) * kHidden + dimension] = represented;
        }
    }
    return residual;
}

} // namespace

int ninfer::test::qwen4::real_oracle::run_gr_cell(const verifier::LoadedModel& model,
                                                  DeviceContext& device) {
    const verifier::GrWeights& weights = model.view().layers[0].attention_gr;
    if (weights.norm.dtype != DType::FP32 || weights.norm.numel() != kFlat ||
        weights.down.qtype != QType::GGML_Q8_0 || weights.down.n != kRank ||
        weights.down.k != kFlat || weights.up.qtype != QType::GGML_Q8_0 ||
        weights.up.n != kFlat || weights.up.k != kRank ||
        weights.inject.dtype != DType::FP32 ||
        weights.inject.numel() != static_cast<std::int64_t>(kBranches) * kFlat) {
        throw std::logic_error("Qwen4 layer-0 attention GR binding changed");
    }

    const std::vector<std::uint16_t> residual_bits =
        decode_token_zero_residual(model.view().token_embedding);
    std::vector<float> residual(residual_bits.size());
    std::transform(residual_bits.begin(), residual_bits.end(), residual.begin(),
                   [](std::uint16_t bits) { return bf16_to_f32(bits); });
    const std::vector<float> norm =
        copy_device_values<float>(weights.norm.data, static_cast<std::size_t>(kFlat));
    const std::vector<float> write = copy_device_values<float>(
        weights.inject.data, static_cast<std::size_t>(kBranches) * kFlat);
    const std::vector<std::uint8_t> down =
        copy_device_bytes(weights.down.qdata, weights.down.payload_bytes);
    const std::vector<std::uint8_t> up =
        copy_device_bytes(weights.up.qdata, weights.up.payload_bytes);
    const OracleResult expected = gr_oracle(residual, norm, down, up, write);

    DeviceBuffer device_residual = to_device(residual_bits);
    GuardedDeviceBuffer device_mixed(static_cast<std::size_t>(kHidden) * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_scale(static_cast<std::size_t>(kBranches) * sizeof(std::uint16_t));
    device_mixed.fill(0xcd);
    device_scale.fill(0xcd);
    Tensor residual_tensor(device_residual.p, DType::BF16, {kHidden, kBranches});
    Tensor mixed_tensor(device_mixed.data(), DType::BF16, {kHidden});
    Tensor scale_tensor(device_scale.data(), DType::BF16, {kBranches});
    WorkspaceArena workspace(ops::gated_residual_workspace_capacity_bytes());
    ops::gated_residual_read_write(residual_tensor, weights.norm, weights.down, weights.up,
                                   weights.inject, mixed_tensor, scale_tensor, workspace,
                                   device.stream);
    device.synchronize();

    int failures = verify_reduction(
        "Qwen4 real layer-0 attention GR mixed", from_device_bf16(device_mixed.data(), kHidden),
        expected.mixed, kReadCriterion);
    failures += verify_reduction(
        "Qwen4 real layer-0 attention GR write scale",
        from_device_bf16(device_scale.data(), kBranches), expected.write_scale, kScaleCriterion);
    failures += device_mixed.verify_guards("Qwen4 real layer-0 attention GR mixed");
    failures += device_scale.verify_guards("Qwen4 real layer-0 attention GR write scale");
    std::cout << (failures == 0 ? "OK" : "FAIL") << " qwen4_real_gr_oracle_cell\n";
    return failures;
}
