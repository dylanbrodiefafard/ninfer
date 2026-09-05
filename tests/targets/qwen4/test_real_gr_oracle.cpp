#include "targets/qwen4/real_oracle_common.h"

#include "ninfer/ops/gated_residual.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
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
using ninfer::test::PointwiseCriterion;
using ninfer::test::ReductionCriterion;
using ninfer::test::bf16_to_f32;
using ninfer::test::error_stats_enabled;
using ninfer::test::f32_to_bf16;
using ninfer::test::from_device_bf16;
using ninfer::test::to_device;
using ninfer::test::verify_pointwise;
using ninfer::test::verify_reduction;
using namespace ninfer::test::qwen4::real_oracle;

namespace {

constexpr std::int32_t kHidden = 2560;
constexpr std::int32_t kBranches = 4;
constexpr std::int32_t kFlat = kHidden * kBranches;
constexpr std::int32_t kRank = 320;
constexpr std::int32_t kVocabulary = 248320;
constexpr std::size_t kQ4RowBytes =
    static_cast<std::size_t>(kHidden) / kQ4BlockValues * kQ4BlockBytes;
constexpr std::size_t kQ8DownRowBytes =
    static_cast<std::size_t>(kFlat) / kQ8BlockValues * kQ8BlockBytes;
constexpr std::size_t kQ8UpRowBytes =
    static_cast<std::size_t>(kRank) / kQ8BlockValues * kQ8BlockBytes;
constexpr std::size_t kFinalDownBytes = static_cast<std::size_t>(kRank) * kQ8DownRowBytes;
constexpr std::size_t kFinalUpBytes = static_cast<std::size_t>(kFlat) * kQ8UpRowBytes;
constexpr std::size_t kOutputHeadBytes = static_cast<std::size_t>(kVocabulary) * kQ4RowBytes;
constexpr std::int32_t kFinalWitnessPosition = 221;
constexpr std::int32_t kFinalWitnessToken = 5533;
constexpr std::int32_t kFinalWitnessTarget = 13983;

// Pinned llama_tokenize output for one paragraph including its terminal LF. Position 221 is the
// first native gross discrepancy that was common to every recorded external execution profile.
// The external values select the witness only; every comparison below has an independent oracle.
constexpr std::array<std::int32_t, 86> kFrozenParagraph = {
    48, 16451, 17120, 22188, 11988, 3817, 19039, 888, 264, 2716, 8097, 40701, 13, 561,
    1558, 15339, 1754, 3299, 303, 1906, 321, 54004, 1092, 3905, 1727, 13, 3931, 921,
    13224, 20480, 16338, 1528, 11, 6326, 13224, 62586, 6575, 2193, 11, 321, 32335,
    11312, 5000, 3955, 10885, 13, 1061, 14648, 13901, 5533, 13983, 19464, 12, 23,
    1414, 11, 14733, 59429, 11, 321, 3213, 10885, 364, 799, 2526, 10756, 14751,
    1931, 19221, 3136, 13, 11116, 7193, 369, 33625, 17066, 5721, 11, 524, 264,
    3591, 883, 3992, 4131, 13, 198,
};
static_assert(kFrozenParagraph[kFinalWitnessPosition % kFrozenParagraph.size()] ==
              kFinalWitnessToken);
static_assert(kFrozenParagraph[(kFinalWitnessPosition + 1) % kFrozenParagraph.size()] ==
              kFinalWitnessTarget);

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

struct ReadOracleResult {
    std::vector<double> normalized;
    std::vector<double> mixed;
};

ReadOracleResult gr_read_oracle(std::span<const float> residual, std::span<const float> norm,
                                std::span<const std::uint8_t> down,
                                std::span<const std::uint8_t> up) {
    if (residual.size() != kFlat || norm.size() != kFlat) {
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

    std::vector<double> mixed_result(kHidden);
    for (std::int32_t dimension = 0; dimension < kHidden; ++dimension) {
        double mixed = 0.0;
        for (std::int32_t branch = 0; branch < kBranches; ++branch) {
            const std::size_t index =
                static_cast<std::size_t>(branch) * kHidden + dimension;
            mixed += sigmoid(gates[index]) * normalized[index];
        }
        mixed_result[static_cast<std::size_t>(dimension)] = mixed / 4.0;
    }
    return {std::move(normalized), std::move(mixed_result)};
}

OracleResult gr_oracle(std::span<const float> residual, std::span<const float> norm,
                       std::span<const std::uint8_t> down,
                       std::span<const std::uint8_t> up, std::span<const float> write) {
    if (write.size() != static_cast<std::size_t>(kBranches) * kFlat) {
        throw std::logic_error("real GR oracle received malformed represented write weights");
    }
    ReadOracleResult read = gr_read_oracle(residual, norm, down, up);
    OracleResult result{std::move(read.mixed), std::vector<double>(kBranches)};
    for (std::int32_t branch = 0; branch < kBranches; ++branch) {
        double projected = 0.0;
        const std::size_t base = static_cast<std::size_t>(branch) * kFlat;
        for (std::int32_t index = 0; index < kFlat; ++index) {
            projected += write[base + index] * read.normalized[static_cast<std::size_t>(index)];
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

void require_ggml_weight(const Weight& weight, QType qtype, std::int32_t rows,
                         std::int32_t columns, std::int32_t group,
                         std::size_t payload_bytes, const char* label) {
    if (weight.payload == nullptr || weight.qdata == nullptr || weight.payload != weight.qdata ||
        weight.qtype != qtype || weight.layout != QuantLayout::GgmlBlockRow ||
        weight.n != rows || weight.k != columns || weight.group != group ||
        weight.group_size != static_cast<std::uint32_t>(group) || weight.ndim != 2 ||
        weight.shape[0] != rows || weight.shape[1] != columns ||
        weight.padded_shape[0] != rows || weight.padded_shape[1] != columns ||
        weight.payload_bytes != payload_bytes) {
        throw std::logic_error(std::string("Qwen4 real final-path binding changed: ") + label);
    }
}

struct FinalPathCapture {
    std::vector<std::uint16_t> residual;
    std::vector<std::uint16_t> final_hidden;
    std::vector<std::uint16_t> logits;
    float nll = 0.0F;
};

FinalPathCapture capture_final_path(const verifier::LoadedModel& model, DeviceContext& device) {
    FinalPathCapture captured;
    {
        verifier::Program program(model, device, verifier::DiagnosticSnapshots::Disabled);
        program.reset();
        for (std::int32_t position = 0; position < kFinalWitnessPosition; ++position) {
            const std::int32_t token =
                kFrozenParagraph[static_cast<std::size_t>(position) % kFrozenParagraph.size()];
            const std::int32_t target = kFrozenParagraph[
                (static_cast<std::size_t>(position) + 1U) % kFrozenParagraph.size()];
            (void) program.execute_token(token, target);
        }
        const verifier::TokenResultView result =
            program.execute_token(kFinalWitnessToken, kFinalWitnessTarget);
        const Tensor residual = program.state().residual();
        if (result.token_index != kFinalWitnessPosition ||
            program.frontier() != kFinalWitnessPosition + 1 || residual.dtype != DType::BF16 ||
            residual.ne[0] != kHidden || residual.ne[1] != kBranches ||
            residual.numel() != kFlat || result.final_hidden.dtype != DType::BF16 ||
            result.final_hidden.numel() != kHidden || result.logits.dtype != DType::BF16 ||
            result.logits.numel() != kVocabulary || result.nll.dtype != DType::FP32 ||
            result.nll.numel() != 1) {
            throw std::logic_error("Qwen4 position-221 Program final-path view changed");
        }
        captured.residual = copy_device_values<std::uint16_t>(residual.data, kFlat);
        captured.final_hidden =
            copy_device_values<std::uint16_t>(result.final_hidden.data, kHidden);
        captured.logits =
            copy_device_values<std::uint16_t>(result.logits.data, kVocabulary);
        captured.nll = copy_device_values<float>(result.nll.data, 1).front();
    }
    return captured;
}

struct DotResult {
    double value = 0.0;
    double absolute_sum = 0.0;
};

DotResult q4_row_dot(const std::uint8_t* row, std::span<const float> input) {
    if (input.size() != kHidden) {
        throw std::logic_error("Qwen4 real output-head oracle received malformed input");
    }
    DotResult result;
    for (std::int32_t block_index = 0;
         block_index < kHidden / static_cast<std::int32_t>(kQ4BlockValues); ++block_index) {
        const auto* block =
            row + static_cast<std::size_t>(block_index) * kQ4BlockBytes;
        const double scale = binary16_to_double(read_u16(block));
        const double minimum_scale = binary16_to_double(read_u16(block + 2));
        for (std::int32_t group = 0; group < 8; ++group) {
            const auto [quant_scale, quant_minimum] = qk_scale_min(block + 4, group);
            const double group_scale = scale * quant_scale;
            const double group_minimum = minimum_scale * quant_minimum;
            for (std::int32_t lane = 0; lane < 32; ++lane) {
                const int packed = block[16 + 32 * (group / 2) + lane];
                const int code = group % 2 == 0 ? packed & 15 : packed >> 4;
                const double weight = group_scale * code - group_minimum;
                const std::size_t column =
                    static_cast<std::size_t>(block_index) * kQ4BlockValues +
                    static_cast<std::size_t>(group * 32 + lane);
                const double product = weight * static_cast<double>(input[column]);
                result.value += product;
                result.absolute_sum += std::abs(product);
            }
        }
    }
    return result;
}

double output_head_error_bound(const DotResult& oracle) {
    constexpr double kFloatUnitRoundoff = 0x1p-24;
    constexpr double kOperations = 5.0 * static_cast<double>(kHidden) + 8.0;
    constexpr double kGamma =
        kOperations * kFloatUnitRoundoff / (1.0 - kOperations * kFloatUnitRoundoff);
    const double accumulation = kGamma * oracle.absolute_sum;
    const double bf16_rounding =
        std::max((std::abs(oracle.value) + accumulation) / 255.0, 0x1p-134);
    return accumulation + bf16_rounding;
}

int verify_complete_output_head(const Weight& output_head,
                                std::span<const std::uint16_t> actual_bits,
                                std::span<const float> input) {
    if (actual_bits.size() != kVocabulary || input.size() != kHidden) {
        throw std::logic_error("Qwen4 real output-head comparison shape changed");
    }
    constexpr std::int32_t kChunkRows = 4096;
    const auto* device_rows = static_cast<const std::uint8_t*>(output_head.qdata);
    std::size_t violations = 0;
    std::int32_t first_violation = -1;
    double first_actual = 0.0;
    double first_reference = 0.0;
    double first_error = 0.0;
    double first_bound = 0.0;
    double maximum_ratio = 0.0;
    for (std::int32_t begin = 0; begin < kVocabulary; begin += kChunkRows) {
        const std::int32_t rows = std::min(kChunkRows, kVocabulary - begin);
        const std::size_t bytes = static_cast<std::size_t>(rows) * kQ4RowBytes;
        const std::vector<std::uint8_t> encoded = copy_device_bytes(
            device_rows + static_cast<std::size_t>(begin) * kQ4RowBytes, bytes);
        for (std::int32_t local = 0; local < rows; ++local) {
            const std::int32_t row = begin + local;
            const DotResult expected = q4_row_dot(
                encoded.data() + static_cast<std::size_t>(local) * kQ4RowBytes, input);
            const double actual = bf16_to_f32(actual_bits[static_cast<std::size_t>(row)]);
            const double bound = output_head_error_bound(expected);
            const double error = std::abs(actual - expected.value);
            const double ratio = error / bound;
            if (std::isfinite(ratio)) { maximum_ratio = std::max(maximum_ratio, ratio); }
            if (!std::isfinite(actual) || !std::isfinite(expected.value) ||
                !std::isfinite(bound) || error > bound) {
                if (first_violation < 0) {
                    first_violation = row;
                    first_actual = actual;
                    first_reference = expected.value;
                    first_error = error;
                    first_bound = bound;
                }
                ++violations;
            }
        }
    }
    if (error_stats_enabled()) {
        std::cout << "OP_ERROR_STATS kind=derived_bound format=Q4_K"
                     " profile=qwen4-real-position-221-head max_bound_ratio="
                  << maximum_ratio << '\n';
    }
    if (violations != 0) {
        std::cerr << "Qwen4 real position-221 complete output head violations=" << violations
                  << " first_row=" << first_violation << " actual=" << first_actual
                  << " reference=" << first_reference << " error=" << first_error
                  << " bound=" << first_bound << '\n';
        return 1;
    }
    return 0;
}

double nll_oracle(std::span<const std::uint16_t> logits, std::int32_t target) {
    if (logits.size() != kVocabulary || target < 0 || target >= kVocabulary) {
        throw std::logic_error("Qwen4 real NLL oracle received malformed represented inputs");
    }
    double maximum = -std::numeric_limits<double>::infinity();
    for (std::uint16_t bits : logits) {
        maximum = std::max(maximum, static_cast<double>(bf16_to_f32(bits)));
    }
    double sum = 0.0;
    for (std::uint16_t bits : logits) {
        sum += std::exp(static_cast<double>(bf16_to_f32(bits)) - maximum);
    }
    return maximum + std::log(sum) -
           static_cast<double>(bf16_to_f32(logits[static_cast<std::size_t>(target)]));
}

int run_final_path_cell(const verifier::LoadedModel& model, DeviceContext& device) {
    const verifier::GrWeights& weights = model.view().final_gr;
    const Weight& output_head = model.view().output_head;
    if (weights.norm.data == nullptr || weights.norm.dtype != DType::FP32 ||
        weights.norm.ne[0] != kFlat || weights.norm.numel() != kFlat ||
        weights.inject.data != nullptr) {
        throw std::logic_error("Qwen4 real final GR tensor binding changed");
    }
    require_ggml_weight(weights.down, QType::GGML_Q8_0, kRank, kFlat,
                        static_cast<std::int32_t>(kQ8BlockValues), kFinalDownBytes,
                        "final GR down");
    require_ggml_weight(weights.up, QType::GGML_Q8_0, kFlat, kRank,
                        static_cast<std::int32_t>(kQ8BlockValues), kFinalUpBytes,
                        "final GR up");
    require_ggml_weight(output_head, QType::GGML_Q4_K, kVocabulary, kHidden,
                        static_cast<std::int32_t>(kQ4BlockValues), kOutputHeadBytes,
                        "output head");

    const FinalPathCapture captured = capture_final_path(model, device);
    std::vector<float> residual(captured.residual.size());
    std::transform(captured.residual.begin(), captured.residual.end(), residual.begin(),
                   [](std::uint16_t bits) { return bf16_to_f32(bits); });
    std::vector<float> final_hidden(captured.final_hidden.size());
    std::transform(captured.final_hidden.begin(), captured.final_hidden.end(),
                   final_hidden.begin(),
                   [](std::uint16_t bits) { return bf16_to_f32(bits); });
    std::vector<double> final_hidden_actual(final_hidden.begin(), final_hidden.end());
    const std::vector<float> norm =
        copy_device_values<float>(weights.norm.data, static_cast<std::size_t>(kFlat));
    const std::vector<std::uint8_t> down =
        copy_device_bytes(weights.down.qdata, weights.down.payload_bytes);
    const std::vector<std::uint8_t> up =
        copy_device_bytes(weights.up.qdata, weights.up.payload_bytes);
    const ReadOracleResult final_gr = gr_read_oracle(residual, norm, down, up);

    int failures = verify_reduction("Qwen4 real position-221 final GR", final_hidden_actual,
                                    final_gr.mixed, kReadCriterion);
    failures += verify_complete_output_head(output_head, captured.logits, final_hidden);
    const std::array<double, 1> actual_nll = {static_cast<double>(captured.nll)};
    const std::array<double, 1> expected_nll = {
        nll_oracle(captured.logits, kFinalWitnessTarget)};
    failures += verify_pointwise("Qwen4 real position-221 NLL", actual_nll, expected_nll,
                                 PointwiseCriterion{/*absolute=*/2.0e-3,
                                                    /*relative=*/1.0e-5});
    std::cout << (failures == 0 ? "OK" : "FAIL")
              << " qwen4_real_final_gr_head_nll_position_221_cell\n";
    return failures;
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
    failures += run_final_path_cell(model, device);
    std::cout << (failures == 0 ? "OK" : "FAIL") << " qwen4_real_gr_oracle_cell\n";
    return failures;
}
