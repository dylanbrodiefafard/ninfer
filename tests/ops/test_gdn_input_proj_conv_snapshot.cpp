#include "ninfer/ops/gdn_input_proj.h"

#include "ops/input_projection_test_common.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::input_projection;

namespace {

// This criterion belongs to the complete A16 fused projection/conv/snapshot Op.
constexpr ReductionCriterion kGdnInputProjConvSnapshotA16Tolerance{3.15e-3, 4.0e-3, 3.2e-3};
constexpr ReductionCriterion kGdnInputProjConvSnapshotA4Tolerance{0.16, 4.0e-3, 0.16};

constexpr std::int32_t kQueryRows = 2048;
constexpr std::int32_t kKeyRows   = 2048;

double silu_fp64(double value) {
    if (value >= 0.0) { return value / (1.0 + std::exp(-value)); }
    const double exponential = std::exp(value);
    return value * exponential / (1.0 + exponential);
}

std::vector<float> make_conv_weight(std::int32_t channels, std::uint32_t seed) {
    std::vector<float> weight(static_cast<std::size_t>(channels) * 4);
    fill_uniform(weight, seed, -0.02F, 0.02F);
    round_to_bf16(weight);
    return weight;
}

std::vector<std::uint16_t> make_state(std::int32_t channels, std::int32_t slots,
                                      std::int32_t initial_slot, std::uint32_t seed) {
    const std::size_t slot_stride = static_cast<std::size_t>(channels) * 3;
    std::vector<std::uint16_t> state(slot_stride * slots, 0xffffU);
    std::vector<float> initial(slot_stride);
    fill_uniform(initial, seed, -0.05F, 0.05F);
    round_to_bf16(initial);
    const std::size_t initial_base = static_cast<std::size_t>(initial_slot) * slot_stride;
    for (std::size_t index = 0; index < initial.size(); ++index) {
        state[initial_base + index] = f32_to_bf16(initial[index]);
    }
    return state;
}

struct SnapshotOracle {
    std::vector<double> query;
    std::vector<double> key;
    std::vector<double> value;
    std::vector<double> state;
};

template <class Projection>
SnapshotOracle
snapshot_oracle(std::int32_t value_rows, std::int32_t tokens, const std::vector<float>& conv_weight,
                std::span<const std::uint16_t> initial_state, Projection&& projection) {
    const std::int32_t channels = kQueryRows + kKeyRows + value_rows;
    SnapshotOracle oracle;
    const std::vector<std::int32_t> query_rows     = sampled_rows(kQueryRows);
    const std::vector<std::int32_t> key_rows       = sampled_rows(kKeyRows);
    const std::vector<std::int32_t> value_selected = sampled_rows(value_rows);
    const std::size_t sampled_channel_count =
        query_rows.size() + key_rows.size() + value_selected.size();
    oracle.query.reserve(query_rows.size() * static_cast<std::size_t>(tokens));
    oracle.key.reserve(key_rows.size() * static_cast<std::size_t>(tokens));
    oracle.value.reserve(value_selected.size() * static_cast<std::size_t>(tokens));
    oracle.state.reserve(sampled_channel_count * 3 * static_cast<std::size_t>(tokens));

    const auto evaluate = [&](std::int32_t global_row, std::vector<double>& output) {
        double state0        = bf16_to_f32(initial_state[global_row]);
        double state1        = bf16_to_f32(initial_state[channels + global_row]);
        double state2        = bf16_to_f32(initial_state[2 * channels + global_row]);
        const double weight0 = conv_weight[global_row];
        const double weight1 = conv_weight[channels + global_row];
        const double weight2 = conv_weight[2 * channels + global_row];
        const double weight3 = conv_weight[3 * channels + global_row];
        for (std::int32_t token = 0; token < tokens; ++token) {
            const double projected = projection(global_row, token);
            const double convolved =
                weight0 * state0 + weight1 * state1 + weight2 * state2 + weight3 * projected;
            output.push_back(silu_fp64(convolved));
            oracle.state.push_back(state1);
            oracle.state.push_back(state2);
            oracle.state.push_back(projected);
            state0 = state1;
            state1 = state2;
            state2 = projected;
        }
    };

    for (const std::int32_t row : query_rows) { evaluate(row, oracle.query); }
    for (const std::int32_t row : key_rows) { evaluate(kQueryRows + row, oracle.key); }
    for (const std::int32_t row : value_selected) {
        evaluate(kQueryRows + kKeyRows + row, oracle.value);
    }
    return oracle;
}

std::vector<double> gather_state(const std::vector<std::uint16_t>& full, std::int32_t channels,
                                 std::int32_t value_rows, std::int32_t tokens,
                                 std::int32_t snapshot_base_slot) {
    std::vector<double> gathered;
    const auto append = [&](std::int32_t global_row) {
        for (std::int32_t token = 0; token < tokens; ++token) {
            const std::size_t token_base =
                static_cast<std::size_t>(snapshot_base_slot + token) * 3 * channels;
            for (std::int32_t history = 0; history < 3; ++history) {
                gathered.push_back(bf16_to_f32(
                    full[token_base + static_cast<std::size_t>(history) * channels + global_row]));
            }
        }
    };
    for (const std::int32_t row : sampled_rows(kQueryRows)) { append(row); }
    for (const std::int32_t row : sampled_rows(kKeyRows)) { append(kQueryRows + row); }
    for (const std::int32_t row : sampled_rows(value_rows)) { append(kQueryRows + kKeyRows + row); }
    return gathered;
}

int verify_state_effects(std::string_view label, const std::vector<std::uint16_t>& before,
                         const std::vector<std::uint16_t>& after, std::int32_t channels,
                         std::int32_t tokens, std::int32_t slots, std::int32_t snapshot_base_slot) {
    const std::size_t slot_stride = static_cast<std::size_t>(channels) * 3;
    for (std::int32_t slot = snapshot_base_slot; slot < snapshot_base_slot + tokens; ++slot) {
        const std::size_t base = static_cast<std::size_t>(slot) * slot_stride;
        for (std::size_t index = 0; index < slot_stride; ++index) {
            if (!std::isfinite(bf16_to_f32(after[base + index]))) {
                std::cerr << label << ": state slot " << slot << " was not fully written\n";
                return 1;
            }
        }
    }
    for (std::int32_t slot = 0; slot < slots; ++slot) {
        if (slot >= snapshot_base_slot && slot < snapshot_base_slot + tokens) { continue; }
        const std::size_t base = static_cast<std::size_t>(slot) * slot_stride;
        if (!std::equal(before.begin() + static_cast<std::ptrdiff_t>(base),
                        before.begin() + static_cast<std::ptrdiff_t>(base + slot_stride),
                        after.begin() + static_cast<std::ptrdiff_t>(base))) {
            std::cerr << label << ": state slot " << slot << " was modified\n";
            return 1;
        }
    }
    return 0;
}

int verify_snapshot_outputs(
    std::string_view suffix, const GuardedBf16Tensor& query, const GuardedBf16Tensor& key,
    const GuardedBf16Tensor& value, std::int32_t value_rows, std::int32_t tokens,
    const SnapshotOracle& oracle,
    const ReductionCriterion& criterion = kGdnInputProjConvSnapshotA16Tolerance) {
    int failures = 0;
    failures += query.verify_guards("snapshot query" + std::string(suffix));
    failures += key.verify_guards("snapshot key" + std::string(suffix));
    failures += value.verify_guards("snapshot value" + std::string(suffix));
    failures += query.verify_fully_written("snapshot query" + std::string(suffix));
    failures += key.verify_fully_written("snapshot key" + std::string(suffix));
    failures += value.verify_fully_written("snapshot value" + std::string(suffix));
    failures += compare("snapshot query" + std::string(suffix),
                        gather_rows(query.values(), kQueryRows, 0, kQueryRows, tokens),
                        oracle.query, criterion);
    failures +=
        compare("snapshot key" + std::string(suffix),
                gather_rows(key.values(), kKeyRows, 0, kKeyRows, tokens), oracle.key, criterion);
    failures += compare("snapshot value" + std::string(suffix),
                        gather_rows(value.values(), value_rows, 0, value_rows, tokens),
                        oracle.value, criterion);
    return failures;
}

int run_q4_q5_case(DevicePackedWeight& query_key, DevicePackedWeight& value_z_weight,
                   std::int32_t tokens, std::int32_t initial_slot) {
    constexpr std::int32_t kHidden           = 5120;
    constexpr std::int32_t kValueRows        = 6144;
    constexpr std::int32_t kZRows            = 6144;
    constexpr std::int32_t kChannels         = 10240;
    constexpr std::int32_t kSnapshotBaseSlot = 1;
    const std::int32_t slots                 = std::max(tokens + 2, initial_slot + 1);
    const std::vector<float> activation      = make_bf16_activation(kHidden, tokens, 601U + tokens);
    const std::vector<std::uint16_t> activation_bits  = bf16_bits(activation);
    const std::vector<float> conv_weight              = make_conv_weight(kChannels, 607U);
    const std::vector<std::uint16_t> conv_weight_bits = bf16_bits(conv_weight);
    const std::vector<std::uint16_t> state_before =
        make_state(kChannels, slots, initial_slot, 613U + tokens);
    const std::vector<std::int32_t> initial_value{initial_slot};
    const std::vector<std::int32_t> snapshot_base_value{kSnapshotBaseSlot};

    DeviceBuffer device_activation    = to_device(activation_bits);
    DeviceBuffer device_conv_weight   = to_device(conv_weight_bits);
    DeviceBuffer device_initial       = to_device(initial_value);
    DeviceBuffer device_snapshot_base = to_device(snapshot_base_value);
    GuardedBf16Tensor state(kChannels * 3, slots);
    state.copy_from_bits(state_before);
    GuardedBf16Tensor query(kQueryRows, tokens);
    GuardedBf16Tensor key(kKeyRows, tokens);
    GuardedBf16Tensor value(kValueRows, tokens);
    GuardedBf16Tensor z(kZRows, tokens);
    Tensor x(device_activation.p, DType::BF16, {kHidden, tokens});
    Tensor conv(device_conv_weight.p, DType::BF16, {kChannels, 4});
    Tensor conv_state(state.data(), DType::BF16, {kChannels, 3, slots});
    Tensor initial(device_initial.p, DType::I32, {1});
    Tensor snapshot_base(device_snapshot_base.p, DType::I32, {1});
    Tensor q                          = query.tensor();
    Tensor k                          = key.tensor();
    Tensor v                          = value.tensor();
    Tensor z_output                   = z.tensor();
    const std::size_t workspace_bytes = ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
        kQueryRows, kKeyRows, kValueRows, tokens, tokens);
    WorkspaceArena workspace(std::max<std::size_t>(1, workspace_bytes));

    ops::gdn_input_proj_conv_snapshot(x, query_key.view(), value_z_weight.view(), conv, conv_state,
                                      initial, snapshot_base, q, k, v, z_output, workspace,
                                      nullptr);
    cuda_synchronize();

    const std::size_t initial_base = static_cast<std::size_t>(initial_slot) * 3 * kChannels;
    const std::span<const std::uint16_t> initial_state(state_before.data() + initial_base,
                                                       3 * kChannels);
    const SnapshotOracle oracle = snapshot_oracle(
        kValueRows, tokens, conv_weight, initial_state, [&](std::int32_t row, std::int32_t token) {
            const float* token_activation =
                activation.data() + static_cast<std::size_t>(token) * kHidden;
            if (row < kQueryRows + kKeyRows) {
                return quantized_weight::dot_fp64(query_key.host, row, token_activation, kHidden);
            }
            return quantized_weight::dot_fp64(value_z_weight.host, row - kQueryRows - kKeyRows,
                                              token_activation, kHidden);
        });
    const std::vector<std::uint16_t> state_after = state.bits();
    const std::string suffix                     = " Q4/Q5 A16 T=" + std::to_string(tokens) +
                               " initial=" + std::to_string(initial_slot) +
                               " base=" + std::to_string(kSnapshotBaseSlot);
    int failures = verify_snapshot_outputs(suffix, query, key, value, kValueRows, tokens, oracle);
    failures += compare("snapshot state" + suffix,
                        gather_state(state_after, kChannels, kValueRows, tokens, kSnapshotBaseSlot),
                        oracle.state, kGdnInputProjConvSnapshotA16Tolerance);
    failures += state.verify_guards("snapshot state" + suffix);
    failures += verify_state_effects("snapshot state" + suffix, state_before, state_after,
                                     kChannels, tokens, slots, kSnapshotBaseSlot);
    failures += z.verify_guards("snapshot z" + suffix);
    failures += z.verify_fully_written("snapshot z" + suffix);
    failures += compare(
        "snapshot z" + suffix, gather_rows(z.values(), kZRows, 0, kZRows, tokens),
        projection_oracle(value_z_weight.host, kValueRows, kZRows, activation, kHidden, tokens),
        kGdnInputProjConvSnapshotA16Tolerance);
    failures += verify_preserved("snapshot x" + suffix, device_activation, activation_bits);
    failures +=
        verify_preserved("snapshot conv weight" + suffix, device_conv_weight, conv_weight_bits);
    failures += verify_preserved("snapshot initial slot" + suffix, device_initial, initial_value);
    failures +=
        verify_preserved("snapshot base slot" + suffix, device_snapshot_base, snapshot_base_value);
    failures += query_key.verify_preserved("snapshot query/key weight" + suffix);
    failures += value_z_weight.verify_preserved("snapshot value/z weight" + suffix);
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << "snapshot" << suffix << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    return failures;
}

int run_q4_q5() {
    constexpr std::int32_t kHidden = 5120;
    DevicePackedWeight query_key(
        quantized_weight::make_patterned_weight(QType::Q4G64_F16S, 4096, kHidden, 617U));
    DevicePackedWeight value_z_weight(
        quantized_weight::make_patterned_weight(QType::Q5G64_F16S, 12288, kHidden, 619U));
    int failures = 0;
    // Cover every fixed Small-T specialization plus the first composed extent.
    for (const std::int32_t tokens : {1, 2, 3, 4, 5, 6, 7}) {
        const std::int32_t initial_slot = tokens == 5 ? 0 : tokens + 1;
        failures += run_q4_q5_case(query_key, value_z_weight, tokens, initial_slot);
    }
    return failures;
}

int run_w8_case(DevicePackedWeight& parent, std::int32_t tokens, std::int32_t initial_slot) {
    constexpr std::int32_t kHidden           = 2048;
    constexpr std::int32_t kValueRows        = 4096;
    constexpr std::int32_t kZRows            = 4096;
    constexpr std::int32_t kChannels         = 8192;
    constexpr std::int32_t kSnapshotBaseSlot = 1;
    const std::int32_t slots                 = std::max(tokens + 2, initial_slot + 1);
    const std::vector<float> activation      = make_bf16_activation(kHidden, tokens, 701U + tokens);
    const std::vector<std::uint16_t> activation_bits  = bf16_bits(activation);
    const std::vector<float> conv_weight              = make_conv_weight(kChannels, 709U);
    const std::vector<std::uint16_t> conv_weight_bits = bf16_bits(conv_weight);
    const std::vector<std::uint16_t> state_before =
        make_state(kChannels, slots, initial_slot, 719U + tokens);
    const std::vector<std::int32_t> initial_value{initial_slot};
    const std::vector<std::int32_t> snapshot_base_value{kSnapshotBaseSlot};

    DeviceBuffer device_activation    = to_device(activation_bits);
    DeviceBuffer device_conv_weight   = to_device(conv_weight_bits);
    DeviceBuffer device_initial       = to_device(initial_value);
    DeviceBuffer device_snapshot_base = to_device(snapshot_base_value);
    GuardedBf16Tensor state(kChannels * 3, slots);
    state.copy_from_bits(state_before);
    GuardedBf16Tensor query(kQueryRows, tokens);
    GuardedBf16Tensor key(kKeyRows, tokens);
    GuardedBf16Tensor value(kValueRows, tokens);
    GuardedBf16Tensor z(kZRows, tokens);
    Tensor x(device_activation.p, DType::BF16, {kHidden, tokens});
    Tensor conv(device_conv_weight.p, DType::BF16, {kChannels, 4});
    Tensor conv_state(state.data(), DType::BF16, {kChannels, 3, slots});
    Tensor initial(device_initial.p, DType::I32, {1});
    Tensor snapshot_base(device_snapshot_base.p, DType::I32, {1});
    Tensor q                          = query.tensor();
    Tensor k                          = key.tensor();
    Tensor v                          = value.tensor();
    Tensor z_output                   = z.tensor();
    const std::size_t workspace_bytes = ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
        kQueryRows, kKeyRows, kValueRows, tokens, tokens);
    WorkspaceArena workspace(std::max<std::size_t>(1, workspace_bytes));

    ops::gdn_input_proj_conv_snapshot(x, parent.view(), conv, conv_state, initial, snapshot_base, q,
                                      k, v, z_output, workspace, nullptr);
    cuda_synchronize();

    const std::size_t initial_base = static_cast<std::size_t>(initial_slot) * 3 * kChannels;
    const std::span<const std::uint16_t> initial_state(state_before.data() + initial_base,
                                                       3 * kChannels);
    const SnapshotOracle oracle = snapshot_oracle(
        kValueRows, tokens, conv_weight, initial_state, [&](std::int32_t row, std::int32_t token) {
            return quantized_weight::dot_fp64(
                parent.host, row, activation.data() + static_cast<std::size_t>(token) * kHidden,
                kHidden);
        });
    const std::vector<std::uint16_t> state_after = state.bits();
    const std::string suffix                     = " W8 A16 T=" + std::to_string(tokens) +
                               " initial=" + std::to_string(initial_slot) +
                               " base=" + std::to_string(kSnapshotBaseSlot);
    int failures = verify_snapshot_outputs(suffix, query, key, value, kValueRows, tokens, oracle);
    failures += compare("snapshot state" + suffix,
                        gather_state(state_after, kChannels, kValueRows, tokens, kSnapshotBaseSlot),
                        oracle.state, kGdnInputProjConvSnapshotA16Tolerance);
    failures += state.verify_guards("snapshot state" + suffix);
    failures += verify_state_effects("snapshot state" + suffix, state_before, state_after,
                                     kChannels, tokens, slots, kSnapshotBaseSlot);
    failures += z.verify_guards("snapshot z" + suffix);
    failures += z.verify_fully_written("snapshot z" + suffix);
    failures +=
        compare("snapshot z" + suffix, gather_rows(z.values(), kZRows, 0, kZRows, tokens),
                projection_oracle(parent.host, kChannels, kZRows, activation, kHidden, tokens),
                kGdnInputProjConvSnapshotA16Tolerance);
    failures += verify_preserved("snapshot x" + suffix, device_activation, activation_bits);
    failures +=
        verify_preserved("snapshot conv weight" + suffix, device_conv_weight, conv_weight_bits);
    failures += verify_preserved("snapshot initial slot" + suffix, device_initial, initial_value);
    failures +=
        verify_preserved("snapshot base slot" + suffix, device_snapshot_base, snapshot_base_value);
    failures += parent.verify_preserved("snapshot parent weight" + suffix);
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << "snapshot" << suffix << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    return failures;
}

int run_w8() {
    constexpr std::int32_t kHidden = 2048;
    DevicePackedWeight parent(
        quantized_weight::make_patterned_weight(QType::W8G32_F16S, 12288, kHidden, 727U));
    int failures = 0;
    // Representative registered T values around every current execution boundary.
    for (const std::int32_t tokens : {1, 2, 17}) {
        const std::int32_t initial_slot = tokens == 2 ? 0 : tokens + 1;
        failures += run_w8_case(parent, tokens, initial_slot);
    }
    return failures;
}

int run_nvfp4_case(DevicePackedWeight& parent, std::int32_t tokens, ops::LinearPolicy policy,
                   std::int32_t initial_slot) {
    constexpr std::int32_t kHidden           = 5120;
    constexpr std::int32_t kValueRows        = 6144;
    constexpr std::int32_t kZRows            = 6144;
    constexpr std::int32_t kChannels         = 10240;
    constexpr std::int32_t kRows             = kChannels + kZRows;
    constexpr std::int32_t kSnapshotBaseSlot = 1;
    const std::int32_t slots                 = std::max(tokens + 2, initial_slot + 1);
    const std::vector<float> activation =
        make_bf16_activation(kHidden, tokens, 809U + static_cast<std::uint32_t>(tokens));
    const std::vector<std::uint16_t> activation_bits  = bf16_bits(activation);
    const std::vector<float> conv_weight              = make_conv_weight(kChannels, 811U);
    const std::vector<std::uint16_t> conv_weight_bits = bf16_bits(conv_weight);
    const std::vector<std::uint16_t> state_before =
        make_state(kChannels, slots, initial_slot, 821U + static_cast<std::uint32_t>(tokens));
    const std::vector<std::int32_t> initial_value{initial_slot};
    const std::vector<std::int32_t> snapshot_base_value{kSnapshotBaseSlot};

    DeviceBuffer device_activation    = to_device(activation_bits);
    DeviceBuffer device_conv_weight   = to_device(conv_weight_bits);
    DeviceBuffer device_initial       = to_device(initial_value);
    DeviceBuffer device_snapshot_base = to_device(snapshot_base_value);
    GuardedBf16Tensor state(kChannels * 3, slots);
    state.copy_from_bits(state_before);
    GuardedBf16Tensor query(kQueryRows, tokens);
    GuardedBf16Tensor key(kKeyRows, tokens);
    GuardedBf16Tensor value(kValueRows, tokens);
    GuardedBf16Tensor z(kZRows, tokens);
    Tensor x(device_activation.p, DType::BF16, {kHidden, tokens});
    Tensor conv(device_conv_weight.p, DType::BF16, {kChannels, 4});
    Tensor conv_state(state.data(), DType::BF16, {kChannels, 3, slots});
    Tensor initial(device_initial.p, DType::I32, {1});
    Tensor snapshot_base(device_snapshot_base.p, DType::I32, {1});
    Tensor q                          = query.tensor();
    Tensor k                          = key.tensor();
    Tensor v                          = value.tensor();
    Tensor z_output                   = z.tensor();
    const std::size_t workspace_bytes = ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
        QType::NVFP4, kRows, kHidden, policy, tokens, tokens);
    WorkspaceArena workspace(std::max<std::size_t>(256, workspace_bytes));

    ops::gdn_input_proj_conv_snapshot(x, parent.view(), conv, conv_state, initial, snapshot_base, q,
                                      k, v, z_output, policy, workspace, nullptr);
    cuda_synchronize();

    const std::size_t initial_base = static_cast<std::size_t>(initial_slot) * 3 * kChannels;
    const std::span<const std::uint16_t> initial_state(state_before.data() + initial_base,
                                                       3 * kChannels);
    const SnapshotOracle oracle = snapshot_oracle(
        kValueRows, tokens, conv_weight, initial_state, [&](std::int32_t row, std::int32_t token) {
            return quantized_weight::dot_fp64(
                parent.host, row, activation.data() + static_cast<std::size_t>(token) * kHidden,
                kHidden);
        });
    const bool a4 = policy == ops::LinearPolicy::AllowA4 && tokens > 16;
    const ReductionCriterion& criterion =
        a4 ? kGdnInputProjConvSnapshotA4Tolerance : kGdnInputProjConvSnapshotA16Tolerance;
    const std::string suffix =
        std::string(" NVFP4 ") + (a4 ? "A4" : "A16") + " T=" + std::to_string(tokens) +
        " initial=" + std::to_string(initial_slot) + " base=" + std::to_string(kSnapshotBaseSlot);
    const std::vector<std::uint16_t> state_after = state.bits();
    int failures =
        verify_snapshot_outputs(suffix, query, key, value, kValueRows, tokens, oracle, criterion);
    failures += compare("snapshot state" + suffix,
                        gather_state(state_after, kChannels, kValueRows, tokens, kSnapshotBaseSlot),
                        oracle.state, criterion);
    failures += state.verify_guards("snapshot state" + suffix);
    failures += verify_state_effects("snapshot state" + suffix, state_before, state_after,
                                     kChannels, tokens, slots, kSnapshotBaseSlot);
    failures += z.verify_guards("snapshot z" + suffix);
    failures += z.verify_fully_written("snapshot z" + suffix);
    failures += compare(
        "snapshot z" + suffix, gather_rows(z.values(), kZRows, 0, kZRows, tokens),
        projection_oracle(parent.host, kChannels, kZRows, activation, kHidden, tokens), criterion);
    failures += verify_preserved("snapshot x" + suffix, device_activation, activation_bits);
    failures +=
        verify_preserved("snapshot conv weight" + suffix, device_conv_weight, conv_weight_bits);
    failures += verify_preserved("snapshot initial slot" + suffix, device_initial, initial_value);
    failures +=
        verify_preserved("snapshot base slot" + suffix, device_snapshot_base, snapshot_base_value);
    failures += parent.verify_preserved("snapshot parent weight" + suffix);
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << "snapshot" << suffix << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    return failures;
}

int run_nvfp4() {
    constexpr std::int32_t kHidden = 5120;
    constexpr std::int32_t kRows   = 16384;
    quantized_weight::PatternedWeightOptions options;
    options.weight_scale_divisor = 0.125F;
    options.input_scale_divisor  = 3.5F;
    DevicePackedWeight parent(
        quantized_weight::make_patterned_weight(QType::NVFP4, kRows, kHidden, 823U, options));

    int failures = 0;
    failures += run_nvfp4_case(parent, 1, ops::LinearPolicy::A16Only, 2);
    failures += run_nvfp4_case(parent, 16, ops::LinearPolicy::AllowA4, 17);
    failures += run_nvfp4_case(parent, 17, ops::LinearPolicy::AllowA4, 0);
    failures += run_nvfp4_case(parent, 1024, ops::LinearPolicy::AllowA4, 1025);
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    const std::size_t q4_interval =
        ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(2048, 2048, 6144, 1, 6);
    const std::size_t q4_witness =
        ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(2048, 2048, 6144, 4, 4);
    const std::size_t q4_right_endpoint =
        ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(2048, 2048, 6144, 6, 6);
    if (q4_interval != q4_witness || q4_witness == 0 || q4_right_endpoint != 0) {
        std::cerr << "Q4/Q5 snapshot interval did not retain its non-monotonic T=4 route\n";
        ++failures;
    }
    if (ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(2048, 2048, 4096, 1, 16) != 0 ||
        ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(2048, 2048, 4096, 1, 17) !=
            ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(2048, 2048, 4096, 17, 17)) {
        std::cerr << "W8 snapshot interval did not preserve its zero/nonzero route boundary\n";
        ++failures;
    }
    const std::size_t nvfp4_a4_17 = ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
        QType::NVFP4, 16384, 5120, ops::LinearPolicy::AllowA4, 17, 17);
    if (ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
            QType::NVFP4, 16384, 5120, ops::LinearPolicy::A16Only, 1, 16) != 0 ||
        ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
            QType::NVFP4, 16384, 5120, ops::LinearPolicy::AllowA4, 1, 16) != 0 ||
        nvfp4_a4_17 == 0 ||
        ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
            QType::NVFP4, 16384, 5120, ops::LinearPolicy::AllowA4, 1, 17) != nvfp4_a4_17) {
        std::cerr << "NVFP4 snapshot interval did not preserve its A16/A4 route boundary\n";
        ++failures;
    }
    failures += run_q4_q5();
    failures += run_w8();
    failures += run_nvfp4();
    std::cout << (failures == 0 ? "OK" : "FAIL") << " gdn_input_proj_conv_snapshot\n";
    return failures == 0 ? 0 : 1;
}
