#include "targets/qwen4/real_oracle_common.h"

#include "ninfer/ops/ple.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <array>
#include <bit>
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
using ninfer::DeviceArena;
using ninfer::DeviceBuffer;
using ninfer::PinnedHostBuffer;
using ninfer::QType;
using ninfer::QuantLayout;
using ninfer::Tensor;
using ninfer::Weight;
using ninfer::test::GuardedDeviceBuffer;
using ninfer::test::PointwiseCriterion;
using ninfer::test::bf16_to_f32;
using ninfer::test::f32_to_bf16;
using ninfer::test::from_device;
using ninfer::test::from_device_bf16;
using ninfer::test::to_device;
using ninfer::test::verify_exact;
using ninfer::test::verify_pointwise;
using namespace ninfer::test::qwen4::real_oracle;

namespace {

constexpr std::int32_t kToken = 48;
constexpr std::int32_t kHidden = ops::kPleEmbeddingWidth;
constexpr std::int32_t kBranches = ops::kPleBranches;
constexpr std::int32_t kChannels = ops::kPleChannels;
constexpr std::int32_t kHistory = ops::kPleConvHistory;
constexpr std::size_t kQ8KeyRowBytes =
    static_cast<std::size_t>(kHidden) / kQ8BlockValues * kQ8BlockBytes;
constexpr std::size_t kHostGuardBytes = 256;
constexpr std::array<std::uint64_t, 3> kNgramMultiplier = {
    23703573157769ULL, 20109073645365ULL, 8052911324071ULL};
constexpr std::array<std::int32_t, ops::kPleHeads> kNgramPrime = {
    20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
    20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171};
constexpr std::array<std::int32_t, ops::kPleHeads> kNgramOffset = {
    0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
    160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114,
    300001275};

// This is the complete-formula PLE criterion owned by tests/ops/test_ple.cpp. The real cell
// changes represented inputs and weights, not the implementation profile.
constexpr PointwiseCriterion kCompleteCriterion{/*absolute=*/0.02, /*relative=*/0.01};
constexpr PointwiseCriterion kStateCriterion{/*absolute=*/1.0e-6, /*relative=*/0.01};

double represented_bf16(double value) {
    return static_cast<double>(bf16_to_f32(f32_to_bf16(static_cast<float>(value))));
}

double sigmoid(double value) {
    if (value >= 0.0) { return 1.0 / (1.0 + std::exp(-value)); }
    const double exponential = std::exp(value);
    return exponential / (1.0 + exponential);
}

double silu(double value) { return value * sigmoid(value); }

double iq4_nl_value(const std::uint8_t* row, std::int32_t column) {
    constexpr std::array<int, 16> codebook = {
        -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
    const auto* block = row + static_cast<std::size_t>(column / ops::kPleIq4NlBlockValues) *
                                 ops::kPleIq4NlBlockBytes;
    const int item = column % ops::kPleIq4NlBlockValues;
    const std::uint8_t packed = block[2 + (item & 15)];
    const std::uint8_t code = item < 16 ? packed & 15U : packed >> 4U;
    return binary16_to_double(read_u16(block)) * codebook[code];
}

std::array<std::int32_t, ops::kPleHeads> reset_history_rows() {
    const std::uint64_t reset = static_cast<std::uint64_t>(verifier::kPleResetToken);
    const std::uint64_t mixed2 = static_cast<std::uint64_t>(kToken) * kNgramMultiplier[0] ^
                                 reset * kNgramMultiplier[1];
    const std::uint64_t mixed3 = mixed2 ^ reset * kNgramMultiplier[2];
    std::array<std::int32_t, ops::kPleHeads> rows{};
    for (std::size_t head = 0; head < rows.size(); ++head) {
        const std::int64_t mixed = std::bit_cast<std::int64_t>(head < 8 ? mixed2 : mixed3);
        std::int64_t remainder = mixed % kNgramPrime[head];
        if (remainder < 0) { remainder += kNgramPrime[head]; }
        rows[head] = kNgramOffset[head] + static_cast<std::int32_t>(remainder);
    }
    return rows;
}

void require_q8(const Weight& weight, std::int32_t rows, const char* name) {
    const std::size_t expected = static_cast<std::size_t>(rows) * kQ8KeyRowBytes;
    if (weight.qtype != QType::GGML_Q8_0 || weight.layout != QuantLayout::GgmlBlockRow ||
        weight.n != rows || weight.k != kHidden || weight.qdata == nullptr ||
        weight.payload_bytes != expected) {
        throw std::logic_error(std::string("Qwen4 real PLE malformed ") + name);
    }
}

struct PleRowsFixture {
    std::array<std::int32_t, ops::kPleHeads> row_ids{};
    std::vector<std::uint8_t> staged;
    std::vector<std::uint16_t> embedding;
};

PleRowsFixture decode_ple_rows(const verifier::PleWeights& weights) {
    if (weights.table.data == nullptr || weights.table.rows == 0 ||
        weights.table.rows > UINT64_MAX / ops::kPleIq4NlRowBytes ||
        weights.table.bytes != weights.table.rows * ops::kPleIq4NlRowBytes) {
        throw std::logic_error("Qwen4 real PLE mapped table binding changed");
    }
    PleRowsFixture fixture{
        .row_ids = reset_history_rows(),
        .staged = std::vector<std::uint8_t>(ops::kPleStagedBytes),
        .embedding = std::vector<std::uint16_t>(static_cast<std::size_t>(kHidden)),
    };
    for (std::int32_t head = 0; head < ops::kPleHeads; ++head) {
        const std::int32_t row_id = fixture.row_ids[head];
        if (row_id < 0 || static_cast<std::uint64_t>(row_id) >= weights.table.rows) {
            throw std::logic_error("Qwen4 real PLE n-gram row is outside the mapped table");
        }
        const auto* row = weights.table.data +
                          static_cast<std::uint64_t>(row_id) * ops::kPleIq4NlRowBytes;
        std::copy_n(row, ops::kPleIq4NlRowBytes,
                    fixture.staged.begin() +
                        static_cast<std::size_t>(head) * ops::kPleIq4NlRowBytes);
        for (std::int32_t dimension = 0; dimension < ops::kPleRowWidth; ++dimension) {
            fixture.embedding[static_cast<std::size_t>(head) * ops::kPleRowWidth + dimension] =
                f32_to_bf16(static_cast<float>(iq4_nl_value(row, dimension)));
        }
    }
    return fixture;
}

std::vector<std::uint16_t> pre_layer_one_residual(const verifier::LoadedModel& model,
                                                  ninfer::DeviceContext& device) {
    verifier::Program program(model, device, verifier::DiagnosticSnapshots::Enabled);
    program.reset();
    const verifier::TokenResultView result = program.execute_token(kToken, 16451);
    if (result.gr.size() != verifier::kLayerCount || result.gr[0].layer != 0 ||
        result.gr[0].ffn_residual.dtype != DType::BF16 ||
        result.gr[0].ffn_residual.ne[0] != kHidden ||
        result.gr[0].ffn_residual.ne[1] != kBranches ||
        result.gr[0].ffn_residual.ne[2] != 1) {
        throw std::logic_error("Qwen4 Program layer-0 FFN residual snapshot changed");
    }
    // TokenResultView storage is Program-owned. Copy the represented boundary while it is valid;
    // no PLE output or private PLE intermediate becomes an oracle input.
    return from_device<std::uint16_t>(result.gr[0].ffn_residual.data, kChannels);
}

int verify_host_guards(const PinnedHostBuffer& storage) {
    const auto* bytes = static_cast<const std::uint8_t*>(storage.data());
    const bool prefix = std::all_of(bytes, bytes + kHostGuardBytes,
                                    [](std::uint8_t value) { return value == 0xa5; });
    const bool suffix = std::all_of(bytes + kHostGuardBytes + ops::kPleStagedBytes,
                                    bytes + storage.size(),
                                    [](std::uint8_t value) { return value == 0xa5; });
    if (prefix && suffix) { return 0; }
    std::cerr << "Qwen4 real PLE pinned stage guard was overwritten\n";
    return 1;
}

std::vector<std::uint16_t> make_initial_state() {
    std::vector<std::uint16_t> state(static_cast<std::size_t>(kChannels) * kHistory);
    for (std::int32_t history = 0; history < kHistory; ++history) {
        for (std::int32_t channel = 0; channel < kChannels; ++channel) {
            const float magnitude =
                0.01F + 0.002F * history + 0.0001F * static_cast<float>(channel % 19);
            const float value = ((channel + history) & 1) == 0 ? magnitude : -magnitude;
            state[static_cast<std::size_t>(history) * kChannels + channel] =
                f32_to_bf16(value);
        }
    }
    return state;
}

std::vector<double> q8_project(std::span<const std::uint8_t> matrix, std::int32_t rows,
                               std::span<const double> input) {
    if (matrix.size() != static_cast<std::size_t>(rows) * kQ8KeyRowBytes ||
        input.size() != static_cast<std::size_t>(kHidden)) {
        throw std::logic_error("Qwen4 real PLE Q8_0 oracle received malformed storage");
    }
    std::vector<double> output(static_cast<std::size_t>(rows));
    for (std::int32_t row = 0; row < rows; ++row) {
        const auto* encoded = matrix.data() + static_cast<std::size_t>(row) * kQ8KeyRowBytes;
        double sum = 0.0;
        for (std::int32_t column = 0; column < kHidden; ++column) {
            sum += ggml_q8_0_value(encoded, column) * input[column];
        }
        output[row] = represented_bf16(sum);
    }
    return output;
}

struct OracleResult {
    std::vector<double> output;
    std::vector<std::uint16_t> final_state;
};

OracleResult ple_oracle(std::span<const std::uint16_t> residual_bits,
                        std::span<const std::uint16_t> embedding_bits,
                        std::span<const std::uint16_t> initial_state,
                        std::span<const std::uint8_t> key_weight,
                        std::span<const std::uint8_t> value_weight,
                        std::span<const float> key_norm,
                        std::span<const float> query_norm,
                        std::span<const float> conv_norm,
                        std::span<const float> conv_weight) {
    if (residual_bits.size() != static_cast<std::size_t>(kChannels) ||
        embedding_bits.size() != static_cast<std::size_t>(kHidden) ||
        initial_state.size() != static_cast<std::size_t>(kChannels) * kHistory ||
        key_norm.size() != static_cast<std::size_t>(kChannels) ||
        query_norm.size() != static_cast<std::size_t>(kChannels) ||
        conv_norm.size() != static_cast<std::size_t>(kChannels) ||
        conv_weight.size() != static_cast<std::size_t>(kChannels) * 4) {
        throw std::logic_error("Qwen4 real PLE oracle received malformed represented inputs");
    }

    std::vector<double> residual(residual_bits.size());
    std::transform(residual_bits.begin(), residual_bits.end(), residual.begin(),
                   [](std::uint16_t bits) { return static_cast<double>(bf16_to_f32(bits)); });
    std::vector<double> embedding(embedding_bits.size());
    std::transform(embedding_bits.begin(), embedding_bits.end(), embedding.begin(),
                   [](std::uint16_t bits) { return static_cast<double>(bf16_to_f32(bits)); });
    std::vector<double> old_state(initial_state.size());
    std::transform(initial_state.begin(), initial_state.end(), old_state.begin(),
                   [](std::uint16_t bits) { return static_cast<double>(bf16_to_f32(bits)); });

    const std::vector<double> key = q8_project(key_weight, kChannels, embedding);
    const std::vector<double> value = q8_project(value_weight, kHidden, embedding);
    std::vector<double> gated(static_cast<std::size_t>(kChannels));
    std::vector<double> current(static_cast<std::size_t>(kChannels));

    for (std::int32_t branch = 0; branch < kBranches; ++branch) {
        const std::size_t base = static_cast<std::size_t>(branch) * kHidden;
        double key_square_sum = 0.0;
        double query_square_sum = 0.0;
        for (std::int32_t dimension = 0; dimension < kHidden; ++dimension) {
            key_square_sum += key[base + dimension] * key[base + dimension];
            query_square_sum += residual[base + dimension] * residual[base + dimension];
        }
        const double key_inverse_rms =
            1.0 / std::sqrt(key_square_sum / static_cast<double>(kHidden) + 1.0e-6);
        const double query_inverse_rms =
            1.0 / std::sqrt(query_square_sum / static_cast<double>(kHidden) + 1.0e-6);
        double dot = 0.0;
        for (std::int32_t dimension = 0; dimension < kHidden; ++dimension) {
            const std::size_t channel = base + dimension;
            const double normalized_key = key[channel] * key_inverse_rms * key_norm[channel];
            const double normalized_query =
                residual[channel] * query_inverse_rms * query_norm[channel];
            dot += normalized_key * normalized_query;
        }
        const double raw = dot / std::sqrt(static_cast<double>(kHidden));
        const double transformed =
            raw == 0.0 ? 0.0 : std::copysign(std::sqrt(std::max(std::abs(raw), 1.0e-6)), raw);
        const double gate = sigmoid(transformed);
        double gated_square_sum = 0.0;
        for (std::int32_t dimension = 0; dimension < kHidden; ++dimension) {
            const std::size_t channel = base + dimension;
            gated[channel] = gate * value[dimension];
            gated_square_sum += gated[channel] * gated[channel];
        }
        const double gated_inverse_rms =
            1.0 / std::sqrt(gated_square_sum / static_cast<double>(kHidden) + 1.0e-6);
        for (std::int32_t dimension = 0; dimension < kHidden; ++dimension) {
            const std::size_t channel = base + dimension;
            current[channel] = represented_bf16(
                gated[channel] * gated_inverse_rms * conv_norm[channel]);
        }
    }

    OracleResult result{std::vector<double>(static_cast<std::size_t>(kChannels)),
                        std::vector<std::uint16_t>(static_cast<std::size_t>(kChannels) * kHistory)};
    for (std::int32_t channel = 0; channel < kChannels; ++channel) {
        double convolution = 0.0;
        for (std::int32_t tap = 0; tap < 4; ++tap) {
            const double source = tap == 3
                                      ? current[channel]
                                      : old_state[static_cast<std::size_t>(3 * tap) * kChannels +
                                                  channel];
            convolution += conv_weight[static_cast<std::size_t>(channel) * 4 + tap] * source;
        }
        result.output[channel] = residual[channel] + gated[channel] + silu(convolution);
    }
    for (std::int32_t history = 0; history < kHistory - 1; ++history) {
        const auto source = initial_state.begin() +
                            static_cast<std::size_t>(history + 1) * kChannels;
        std::copy_n(source, kChannels,
                    result.final_state.begin() + static_cast<std::size_t>(history) * kChannels);
    }
    for (std::int32_t channel = 0; channel < kChannels; ++channel) {
        result.final_state[static_cast<std::size_t>(kHistory - 1) * kChannels + channel] =
            f32_to_bf16(static_cast<float>(current[channel]));
    }
    return result;
}

} // namespace

int ninfer::test::qwen4::real_oracle::run_ple_cell(const verifier::LoadedModel& model,
                                                   ninfer::DeviceContext& device) {
    const verifier::PleWeights& weights = model.view().ple;
    require_q8(weights.key, kChannels, "key projection");
    require_q8(weights.value, kHidden, "value projection");
    if (weights.key_norm.dtype != DType::FP32 || weights.key_norm.numel() != kChannels ||
        weights.query_norm.dtype != DType::FP32 || weights.query_norm.numel() != kChannels ||
        weights.conv_norm.dtype != DType::FP32 || weights.conv_norm.numel() != kChannels ||
        weights.conv.dtype != DType::FP32 || weights.conv.ne[0] != 4 ||
        weights.conv.ne[1] != kChannels || weights.conv.ne[2] != 1) {
        throw std::logic_error("Qwen4 real PLE FP32 binding changed");
    }

    const std::vector<std::uint16_t> residual = pre_layer_one_residual(model, device);
    const PleRowsFixture rows = decode_ple_rows(weights);
    const std::vector<std::uint16_t> initial_state = make_initial_state();
    const std::vector<std::uint8_t> key =
        copy_device_bytes(weights.key.qdata, weights.key.payload_bytes);
    const std::vector<std::uint8_t> value =
        copy_device_bytes(weights.value.qdata, weights.value.payload_bytes);
    const std::vector<float> key_norm =
        copy_device_values<float>(weights.key_norm.data, kChannels);
    const std::vector<float> query_norm =
        copy_device_values<float>(weights.query_norm.data, kChannels);
    const std::vector<float> conv_norm =
        copy_device_values<float>(weights.conv_norm.data, kChannels);
    const std::vector<float> conv_weight =
        copy_device_values<float>(weights.conv.data, static_cast<std::size_t>(kChannels) * 4);
    const OracleResult expected = ple_oracle(residual, rows.embedding, initial_state, key, value,
                                             key_norm, query_norm, conv_norm, conv_weight);

    DeviceBuffer device_residual = to_device(residual);
    DeviceBuffer device_old_state = to_device(initial_state);
    PinnedHostBuffer pinned_rows(ops::kPleStagedBytes + 2 * kHostGuardBytes);
    auto* pinned_bytes = static_cast<std::uint8_t*>(pinned_rows.data());
    std::fill_n(pinned_bytes, pinned_rows.size(), std::uint8_t{0xa5});
    void* pinned_stage = pinned_bytes + kHostGuardBytes;
    GuardedDeviceBuffer device_rows(ops::kPleStagedBytes);
    GuardedDeviceBuffer device_embedding(static_cast<std::size_t>(kHidden) *
                                         sizeof(std::uint16_t));
    GuardedDeviceBuffer device_new_state(initial_state.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_output(residual.size() * sizeof(std::uint16_t));
    device_rows.fill(0xcd);
    device_embedding.fill(0xcd);
    device_new_state.fill(0xcd);
    device_output.fill(0xcd);
    Tensor rows_tensor(device_rows.data(), DType::U8,
                       {ops::kPleIq4NlRowBytes, ops::kPleHeads});
    Tensor decoded_embedding_tensor(device_embedding.data(), DType::BF16,
                                    {ops::kPleRowWidth, ops::kPleHeads});
    ops::ple_iq4_nl_stage_rows(weights.table, rows.row_ids, pinned_stage,
                               ops::kPleStagedBytes, rows_tensor, device.stream);
    ops::ple_iq4_nl_decode_rows(rows_tensor, decoded_embedding_tensor, device.stream);
    Tensor residual_tensor(device_residual.p, DType::BF16, {kHidden, kBranches, 1});
    Tensor embedding_tensor = decoded_embedding_tensor.reshape({kHidden, 1});
    Tensor old_state_tensor(device_old_state.p, DType::BF16, {kChannels, kHistory});
    Tensor new_state_tensor(device_new_state.data(), DType::BF16, {kChannels, kHistory});
    Tensor output_tensor(device_output.data(), DType::BF16, {kHidden, kBranches, 1});
    DeviceArena workspace(ops::ple_workspace_capacity_bytes(1));
    ops::ple_inject(residual_tensor, embedding_tensor, weights.key, weights.value,
                    weights.key_norm, weights.query_norm, weights.conv_norm, weights.conv,
                    old_state_tensor, new_state_tensor, output_tensor, workspace, device.stream);
    device.synchronize();

    int failures = verify_exact(
        "Qwen4 real PLE exact pinned staged rows",
        std::vector<std::uint8_t>(pinned_bytes + kHostGuardBytes,
                                  pinned_bytes + kHostGuardBytes + ops::kPleStagedBytes),
        rows.staged);
    failures += verify_exact("Qwen4 real PLE exact device staged rows",
                             from_device<std::uint8_t>(device_rows.data(), rows.staged.size()),
                             rows.staged);
    failures += verify_exact(
        "Qwen4 real PLE exact decoded embedding",
        from_device<std::uint16_t>(device_embedding.data(), rows.embedding.size()),
        rows.embedding);
    failures += verify_pointwise("Qwen4 real PLE complete FP64 formula",
                                    from_device_bf16(device_output.data(), residual.size()),
                                    expected.output, kCompleteCriterion);
    const std::vector<std::uint16_t> actual_state =
        from_device<std::uint16_t>(device_new_state.data(), initial_state.size());
    const std::size_t retained_values = static_cast<std::size_t>(kChannels) * (kHistory - 1);
    failures += verify_exact(
        "Qwen4 real PLE exact retained history",
        std::vector<std::uint16_t>(actual_state.begin(), actual_state.begin() + retained_values),
        std::vector<std::uint16_t>(expected.final_state.begin(),
                                   expected.final_state.begin() + retained_values));
    std::vector<double> actual_current(static_cast<std::size_t>(kChannels));
    std::vector<double> expected_current(static_cast<std::size_t>(kChannels));
    for (std::size_t channel = 0; channel < actual_current.size(); ++channel) {
        actual_current[channel] = bf16_to_f32(actual_state[retained_values + channel]);
        expected_current[channel] = bf16_to_f32(expected.final_state[retained_values + channel]);
    }
    failures += verify_pointwise("Qwen4 real PLE computed BF16 state", actual_current,
                                 expected_current, kStateCriterion);
    failures += device_output.verify_guards("Qwen4 real PLE output");
    failures += device_new_state.verify_guards("Qwen4 real PLE final state");
    failures += device_rows.verify_guards("Qwen4 real PLE staged rows");
    failures += device_embedding.verify_guards("Qwen4 real PLE decoded embedding");
    failures += verify_host_guards(pinned_rows);
    std::cout << (failures == 0 ? "OK" : "FAIL") << " qwen4_real_ple_oracle_cell\n";
    return failures;
}
