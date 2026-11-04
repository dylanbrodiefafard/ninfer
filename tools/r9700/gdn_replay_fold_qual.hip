#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include "core/arena.h"
#include "core/device.h"
#include "core/decode_graph.h"
#include "core/gdn_replay_records.h"
#include "core/layout.h"
#include "core/linear_attention_state.h"
#include "ninfer/ops/gdn_replay.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/types.h"
#include "ops/r9700/gdn/gdn_ops.h"
#include "ops/r9700/linear/fp8_activation.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kLayers       = 48;
constexpr int kCapacity     = 2;
constexpr int kWidth        = 6;
constexpr int kConvChannels = 10240;
constexpr int kQkHeads      = 16;
constexpr int kValueHeads   = 48;
constexpr int kStateDim     = 128;
constexpr int kSlots        = 2;
constexpr int kOracleLayer  = 17;

double g_max_state_abs         = 0.0;
double g_max_state_relative_l2 = 0.0;

[[noreturn]] void fail(const std::string& message) { throw std::runtime_error(message); }

int select_r9700() {
    int count = 0;
    HIP_CHECK(hipGetDeviceCount(&count));
    int selected = -1;
    for (int ordinal = 0; ordinal < count; ++ordinal) {
        hipDeviceProp_t properties{};
        HIP_CHECK(hipGetDeviceProperties(&properties, ordinal));
        if (std::string(properties.gcnArchName).starts_with("gfx1201") &&
            properties.warpSize == 32) {
            if (selected >= 0) { fail("multiple wave32 gfx1201 devices found"); }
            selected = ordinal;
        }
    }
    if (selected < 0) { fail("no wave32 gfx1201 device found"); }
    return selected;
}

std::uint32_t mix(std::uint32_t value) {
    value ^= value >> 16U;
    value *= 0x7feb352dU;
    value ^= value >> 15U;
    value *= 0x846ca68bU;
    return value ^ (value >> 16U);
}

std::uint16_t bf16_bits(float value) {
    std::uint32_t word           = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t absolute = word & 0x7fffffffU;
    if ((absolute & 0x7f800000U) != 0x7f800000U) {
        word += 0x7fffU + ((word >> 16U) & 1U);
    } else if ((word & 0xffffU) != 0U) {
        word |= 0x10000U;
    }
    return static_cast<std::uint16_t>(word >> 16U);
}

hip_bfloat16 bf16(float value) {
    hip_bfloat16 result;
    result.data = bf16_bits(value);
    return result;
}

float fp32(hip_bfloat16 value) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(value.data) << 16U);
}

float pattern(std::uint32_t seed, float scale) {
    const int centered = static_cast<int>(mix(seed) % 2001U) - 1000;
    return static_cast<float>(centered) * scale / 1000.0F;
}

std::size_t record_column(int layer, int row, int token, int capacity) {
    return (static_cast<std::size_t>(layer) * capacity + row) * kWidth + token;
}

std::size_t key_index(int layer, int row, int token, int head, int dim, int capacity = kCapacity) {
    return (record_column(layer, row, token, capacity) * kQkHeads + head) * kStateDim + dim;
}

std::size_t value_index(int layer, int row, int token, int head, int dim,
                        int capacity = kCapacity) {
    return (record_column(layer, row, token, capacity) * kValueHeads + head) * kStateDim + dim;
}

std::size_t gate_index(int layer, int row, int token, int head, int component,
                       int capacity = kCapacity) {
    return (record_column(layer, row, token, capacity) * kValueHeads + head) * 2 + component;
}

std::size_t conv_index(int layer, int row, int token, int channel, int capacity = kCapacity) {
    return record_column(layer, row, token, capacity) * kConvChannels + channel;
}

std::vector<float> initial_state(int row) {
    std::vector<float> state(static_cast<std::size_t>(kValueHeads) * kStateDim * kStateDim);
    for (std::size_t index = 0; index < state.size(); ++index) {
        state[index] = pattern(0x10007U + static_cast<std::uint32_t>(row) * 131U +
                                   static_cast<std::uint32_t>(index),
                               0.006F);
    }
    return state;
}

std::vector<hip_bfloat16> initial_conv(int row) {
    std::vector<hip_bfloat16> state(static_cast<std::size_t>(3) * kConvChannels);
    for (std::size_t index = 0; index < state.size(); ++index) {
        state[index] = bf16(pattern(0x30011U + static_cast<std::uint32_t>(row) * 149U +
                                        static_cast<std::uint32_t>(index),
                                    0.05F));
    }
    return state;
}

struct Fixture {
    std::vector<hip_bfloat16> conv;
    std::vector<hip_bfloat16> key;
    std::vector<hip_bfloat16> value;
    std::vector<float> gate;
    int capacity = kCapacity;
};

Fixture make_fixture(const ninfer::GdnReplayRecords& records) {
    Fixture fixture{
        std::vector<hip_bfloat16>(static_cast<std::size_t>(records.conv.numel()), bf16(0.0F)),
        std::vector<hip_bfloat16>(static_cast<std::size_t>(records.key.numel()), bf16(0.0F)),
        std::vector<hip_bfloat16>(static_cast<std::size_t>(records.value.numel()), bf16(0.0F)),
        std::vector<float>(static_cast<std::size_t>(records.gate.numel()), 0.0F),
        records.spec.record_capacity,
    };
    const int capacity = fixture.capacity;
    for (int row = 0; row < capacity; ++row) {
        for (int token = 0; token < kWidth; ++token) {
            for (int channel = 0; channel < kConvChannels; ++channel) {
                fixture.conv[conv_index(kOracleLayer, row, token, channel, capacity)] =
                    bf16(pattern(0x50021U + row * 181U + token * 17U + channel, 0.08F));
            }
            for (int head = 0; head < kQkHeads; ++head) {
                for (int dim = 0; dim < kStateDim; ++dim) {
                    fixture.key[key_index(kOracleLayer, row, token, head, dim, capacity)] =
                        bf16(pattern(0x70001U + row * 193U + token * 19U + head * 7U + dim, 0.09F));
                }
            }
            for (int head = 0; head < kValueHeads; ++head) {
                for (int dim = 0; dim < kStateDim; ++dim) {
                    fixture.value[value_index(kOracleLayer, row, token, head, dim, capacity)] =
                        bf16(
                            pattern(0x90031U + row * 211U + token * 23U + head * 11U + dim, 0.07F));
                }
                fixture.gate[gate_index(kOracleLayer, row, token, head, 0, capacity)] =
                    -0.03F -
                    static_cast<float>(mix(0xb001U + row * 31U + token * 13U + head) % 700U) /
                        1000.0F;
                fixture.gate[gate_index(kOracleLayer, row, token, head, 1, capacity)] =
                    0.15F +
                    static_cast<float>(mix(0xd007U + row * 37U + token * 17U + head) % 700U) /
                        1000.0F;
            }
        }
    }
    return fixture;
}

std::vector<double> recurrent_oracle(const Fixture& fixture, int row,
                                     const std::vector<int>& sequence,
                                     const std::vector<float>& initial,
                                     std::vector<double>* outputs = nullptr,
                                     const Fixture* queries       = nullptr) {
    if (outputs != nullptr) outputs->clear();
    std::vector<double> state(initial.begin(), initial.end());
    std::array<double, static_cast<std::size_t>(kQkHeads) * kStateDim> normalized{};
    std::array<double, static_cast<std::size_t>(kQkHeads) * kStateDim> query{};
    for (int token : sequence) {
        for (int qk_head = 0; qk_head < kQkHeads; ++qk_head) {
            double norm2 = 0.0;
            for (int dim = 0; dim < kStateDim; ++dim) {
                const double value = fp32(fixture.key[key_index(kOracleLayer, row, token, qk_head,
                                                                dim, fixture.capacity)]);
                normalized[static_cast<std::size_t>(qk_head) * kStateDim + dim] = value;
                norm2 += value * value;
            }
            const double inverse = 1.0 / std::sqrt(norm2 + 1.0e-6);
            for (int dim = 0; dim < kStateDim; ++dim) {
                normalized[static_cast<std::size_t>(qk_head) * kStateDim + dim] *= inverse;
            }
            if (outputs != nullptr) {
                double query_norm2 = 0.0;
                for (int dim = 0; dim < kStateDim; ++dim) {
                    const auto index = static_cast<std::size_t>(qk_head) * kStateDim + dim;
                    query[index]     = fp32((queries != nullptr ? *queries : fixture)
                                                .key[key_index(kOracleLayer, row, token, qk_head, dim,
                                                               fixture.capacity)]);
                    query_norm2 += query[index] * query[index];
                }
                const double query_inverse = 1.0 / std::sqrt(query_norm2 + 1.0e-6);
                for (int dim = 0; dim < kStateDim; ++dim)
                    query[static_cast<std::size_t>(qk_head) * kStateDim + dim] *= query_inverse;
            }
        }
        for (int value_head = 0; value_head < kValueHeads; ++value_head) {
            const int qk_head  = value_head / 3;
            const double alpha = std::exp(static_cast<double>(fixture.gate[gate_index(
                kOracleLayer, row, token, value_head, 0, fixture.capacity)]));
            const double beta =
                fixture.gate[gate_index(kOracleLayer, row, token, value_head, 1, fixture.capacity)];
            for (int value_dim = 0; value_dim < kStateDim; ++value_dim) {
                const std::size_t state_base =
                    (static_cast<std::size_t>(value_head) * kStateDim + value_dim) * kStateDim;
                double state_key = 0.0;
                for (int key_dim = 0; key_dim < kStateDim; ++key_dim) {
                    state_key +=
                        state[state_base + key_dim] *
                        normalized[static_cast<std::size_t>(qk_head) * kStateDim + key_dim];
                }
                const double represented_value = fp32(fixture.value[value_index(
                    kOracleLayer, row, token, value_head, value_dim, fixture.capacity)]);
                const double delta = beta * (represented_value - alpha * state_key);
                for (int key_dim = 0; key_dim < kStateDim; ++key_dim) {
                    double& element = state[state_base + key_dim];
                    element =
                        alpha * element +
                        delta * normalized[static_cast<std::size_t>(qk_head) * kStateDim + key_dim];
                }
                if (outputs != nullptr) {
                    double dot = 0.0;
                    for (int key_dim = 0; key_dim < kStateDim; ++key_dim)
                        dot += state[state_base + key_dim] * query[qk_head * kStateDim + key_dim];
                    outputs->push_back(dot * static_cast<double>(1.F / std::sqrt(128.F)));
                }
            }
        }
    }
    return state;
}

std::vector<hip_bfloat16> conv_oracle(const Fixture& fixture, int row,
                                      const std::vector<int>& sequence,
                                      std::vector<hip_bfloat16> state) {
    for (int token : sequence) {
        for (int channel = 0; channel < kConvChannels; ++channel) {
            state[channel]                 = state[kConvChannels + channel];
            state[kConvChannels + channel] = state[2 * kConvChannels + channel];
            state[2 * kConvChannels + channel] =
                fixture.conv[conv_index(kOracleLayer, row, token, channel, fixture.capacity)];
        }
    }
    return state;
}

void require_state_close(const std::vector<float>& actual, const std::vector<double>& expected,
                         const std::string& label) {
    if (actual.size() != expected.size()) { fail(label + ": size mismatch"); }
    double maximum            = 0.0;
    double error2             = 0.0;
    double oracle2            = 0.0;
    std::size_t maximum_index = 0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const double error = static_cast<double>(actual[index]) - expected[index];
        if (std::abs(error) > maximum) {
            maximum       = std::abs(error);
            maximum_index = index;
        }
        error2 += error * error;
        oracle2 += expected[index] * expected[index];
    }
    const double relative   = std::sqrt(error2 / std::max(oracle2, 1.0e-30));
    g_max_state_abs         = std::max(g_max_state_abs, maximum);
    g_max_state_relative_l2 = std::max(g_max_state_relative_l2, relative);
    if (maximum > 1.0e-5 || relative > 3.0e-3) {
        fail(label + ": max_abs=" + std::to_string(maximum) +
             " rel_l2=" + std::to_string(relative) + " index=" + std::to_string(maximum_index));
    }
}

void require_bf16_exact(const std::vector<hip_bfloat16>& actual,
                        const std::vector<hip_bfloat16>& expected, const std::string& label) {
    if (actual.size() != expected.size()) { fail(label + ": size mismatch"); }
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (actual[index].data != expected[index].data) {
            fail(label + ": bit mismatch at " + std::to_string(index));
        }
    }
}

template <class T>
std::vector<T> copy_tensor(const ninfer::Tensor& tensor) {
    std::vector<T> result(static_cast<std::size_t>(tensor.numel()));
    HIP_CHECK(
        hipMemcpy(result.data(), tensor.data, result.size() * sizeof(T), hipMemcpyDeviceToHost));
    return result;
}

// Supplement the independent FP64 oracle with the state-publication invariant:
// folding recorded inputs must retain the ordinary snapshot arithmetic profile.
std::vector<float> snapshot_state(const Fixture& fixture, int row, const std::vector<int>& sequence,
                                  const std::vector<float>& initial, hipStream_t stream) {
    ninfer::DeviceBuffer q_storage(kStateDim * kQkHeads * 2), k_storage(kStateDim * kQkHeads * 2);
    ninfer::DeviceBuffer v_storage(kStateDim * kValueHeads * 2),
        out_storage(kStateDim * kValueHeads * 2);
    ninfer::DeviceBuffer g_storage(kValueHeads * 4), beta_storage(kValueHeads * 4);
    ninfer::DeviceBuffer state_storage(initial.size() * sizeof(float)),
        slot_storage(sizeof(std::int32_t));
    ninfer::Tensor q(q_storage.data(), ninfer::DType::BF16, {kStateDim, kQkHeads, 1, 1});
    ninfer::Tensor k(k_storage.data(), ninfer::DType::BF16, {kStateDim, kQkHeads, 1, 1});
    ninfer::Tensor v(v_storage.data(), ninfer::DType::BF16, {kStateDim, kValueHeads, 1, 1});
    ninfer::Tensor out(out_storage.data(), ninfer::DType::BF16, {kStateDim, kValueHeads, 1, 1});
    ninfer::Tensor g(g_storage.data(), ninfer::DType::FP32, {kValueHeads, 1, 1});
    ninfer::Tensor beta(beta_storage.data(), ninfer::DType::FP32, {kValueHeads, 1, 1});
    ninfer::Tensor state(state_storage.data(), ninfer::DType::FP32,
                         {kStateDim, kStateDim, kValueHeads, 1});
    ninfer::Tensor slot(slot_storage.data(), ninfer::DType::I32, {1});
    HIP_CHECK(hipMemsetAsync(slot.data, 0, slot.bytes(), stream));
    HIP_CHECK(
        hipMemcpyAsync(state.data, initial.data(), state.bytes(), hipMemcpyHostToDevice, stream));
    for (int token : sequence) {
        const auto* keys =
            fixture.key.data() + key_index(kOracleLayer, row, token, 0, 0, fixture.capacity);
        HIP_CHECK(hipMemcpyAsync(k.data, keys, k.bytes(), hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(q.data, keys, q.bytes(), hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(v.data,
                                 fixture.value.data() +
                                     value_index(kOracleLayer, row, token, 0, 0, fixture.capacity),
                                 v.bytes(), hipMemcpyHostToDevice, stream));
        std::array<float, kValueHeads> host_g{}, host_beta{};
        for (int head = 0; head < kValueHeads; ++head) {
            host_g[head] =
                fixture.gate[gate_index(kOracleLayer, row, token, head, 0, fixture.capacity)];
            host_beta[head] =
                fixture.gate[gate_index(kOracleLayer, row, token, head, 1, fixture.capacity)];
        }
        HIP_CHECK(hipMemcpyAsync(g.data, host_g.data(), g.bytes(), hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(beta.data, host_beta.data(), beta.bytes(), hipMemcpyHostToDevice,
                                 stream));
        ninfer::ops::gated_delta_net_snapshot(q, k, v, g, beta, 1.F / std::sqrt(float(kStateDim)),
                                              true, state, {}, slot, slot, out, stream);
        HIP_CHECK(hipStreamSynchronize(stream));
    }
    return copy_tensor<float>(state);
}

template <class Callable>
void require_invalid(Callable&& callable, const std::string& label) {
    try {
        callable();
    } catch (const std::invalid_argument&) { return; }
    fail(label + ": expected invalid_argument");
}

// One front launch (T verification columns of Qwen3.8 GDN controls) with or without a fused
// deferred fold; returns g, beta and the E4M3 image bytes, which the fold must not change.
std::vector<std::uint8_t> run_front(const ninfer::ops::GdnLayerFold* fold, hipStream_t stream) {
    namespace gdn              = ninfer::ops::r9700::gdn;
    namespace lin              = ninfer::ops::r9700::linear;
    constexpr unsigned kTokens = 16, kHidden = 5120, kHeads = 48;
    std::vector<hip_bfloat16> residual(kTokens * kHidden), norm(kHidden), a(kHeads * kHidden),
        b(kHeads * kHidden);
    for (std::size_t i = 0; i < residual.size(); ++i)
        residual[i] = bf16(pattern(0xe001U + i, 2.0F));
    for (std::size_t i = 0; i < norm.size(); ++i) norm[i] = bf16(pattern(0xe101U + i, 0.2F));
    for (std::size_t i = 0; i < a.size(); ++i) {
        a[i] = bf16(pattern(0xe201U + i, 0.02F));
        b[i] = bf16(pattern(0xe301U + i, 0.02F));
    }
    std::vector<float> a_log(kHeads), dt_bias(kHeads);
    for (unsigned h = 0; h < kHeads; ++h) {
        a_log[h]   = pattern(0xe401U + h, 0.5F);
        dt_bias[h] = pattern(0xe501U + h, 0.5F);
    }
    const std::size_t image_bytes = lin::fp8_activation_workspace_capacity_bytes(kTokens, kHidden);
    const std::size_t control_bytes = kTokens * kHeads * sizeof(float);
    ninfer::DeviceBuffer d_residual(residual.size() * 2), d_norm(norm.size() * 2),
        d_a(a.size() * 2), d_b(b.size() * 2), d_a_log(kHeads * 4), d_dt(kHeads * 4),
        d_g(control_bytes), d_beta(control_bytes), d_image(image_bytes);
    HIP_CHECK(
        hipMemcpy(d_residual.data(), residual.data(), d_residual.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_norm.data(), norm.data(), d_norm.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_a.data(), a.data(), d_a.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b.data(), b.data(), d_b.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_a_log.data(), a_log.data(), d_a_log.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_dt.data(), dt_bias.data(), d_dt.size(), hipMemcpyHostToDevice));
    d_image.fill(0);
    lin::Fp8ActivationWorkspace image{};
    HIP_CHECK(
        lin::fp8_bind_activation_workspace(d_image.data(), image_bytes, kTokens, kHidden, &image));
    HIP_CHECK(gdn::fp8_gdn_normalized_front(
        static_cast<const hip_bfloat16*>(d_residual.data()),
        static_cast<const hip_bfloat16*>(d_norm.data()), 1e-6F, true,
        static_cast<const hip_bfloat16*>(d_a.data()), static_cast<const hip_bfloat16*>(d_b.data()),
        static_cast<const float*>(d_a_log.data()), static_cast<const float*>(d_dt.data()),
        static_cast<float*>(d_g.data()), static_cast<float*>(d_beta.data()), image, stream, {},
        fold));
    HIP_CHECK(hipStreamSynchronize(stream));
    std::vector<std::uint8_t> out(2 * control_bytes + image.code_bytes + image.scale_bytes);
    HIP_CHECK(hipMemcpy(out.data(), d_g.data(), control_bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(
        hipMemcpy(out.data() + control_bytes, d_beta.data(), control_bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(out.data() + 2 * control_bytes, image.codes, image.code_bytes,
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(out.data() + 2 * control_bytes + image.code_bytes, image.scales,
                        image.scale_bytes, hipMemcpyDeviceToHost));
    return out;
}

// Deferred fold (ops::gdn_replay_fold_layer and its fusion into the FP8 GDN front): device rows
// whose record row differs from their position, one prefix row and one no-op row, must leave the
// oracle layer's state and history bitwise equal to the all-layer chain fold and leave every
// other layer untouched.
void qualify_deferred(const ninfer::GdnReplayRecords& records,
                      ninfer::LinearAttentionStatePool& states, const Fixture& fixture,
                      const std::array<std::vector<float>, kCapacity>& initial_recurrent,
                      const std::array<std::vector<hip_bfloat16>, kCapacity>& initial_history,
                      hipStream_t stream) {
    // Record row 1 (3 columns) folds into slot 0; record row 0 (6 columns) into slot 1.
    const std::array<int, kCapacity> slot_of_record{1, 0};
    const std::array<int, kCapacity> columns_of_record{kWidth, 3};
    const auto reset = [&] {
        for (int row = 0; row < kCapacity; ++row) {
            HIP_CHECK(hipMemcpyAsync(states.recurrent_slot(kOracleLayer, slot_of_record[row]).data,
                                     initial_recurrent[row].data(),
                                     initial_recurrent[row].size() * sizeof(float),
                                     hipMemcpyHostToDevice, stream));
            HIP_CHECK(hipMemcpyAsync(states.conv_slot(kOracleLayer, slot_of_record[row]).data,
                                     initial_history[row].data(),
                                     initial_history[row].size() * sizeof(hip_bfloat16),
                                     hipMemcpyHostToDevice, stream));
        }
        HIP_CHECK(hipStreamSynchronize(stream));
    };

    struct Snapshot {
        std::array<std::vector<float>, kSlots> recurrent;
        std::array<std::vector<hip_bfloat16>, kSlots> conv;
        std::vector<float> neighbor;
    };

    const auto snapshot = [&] {
        Snapshot result;
        for (int slot = 0; slot < kSlots; ++slot) {
            result.recurrent[slot] = copy_tensor<float>(states.recurrent_slot(kOracleLayer, slot));
            result.conv[slot] = copy_tensor<hip_bfloat16>(states.conv_slot(kOracleLayer, slot));
        }
        result.neighbor = copy_tensor<float>(states.recurrent_slot(kOracleLayer + 1, 0));
        return result;
    };
    const auto require_same = [&](const Snapshot& actual, const Snapshot& expected,
                                  const std::string& label) {
        for (int slot = 0; slot < kSlots; ++slot) {
            if (actual.recurrent[slot] != expected.recurrent[slot]) {
                fail(label + ": recurrent slot " + std::to_string(slot) + " differs");
            }
            require_bf16_exact(actual.conv[slot], expected.conv[slot],
                               label + ": conv slot " + std::to_string(slot));
        }
        if (actual.neighbor != expected.neighbor) fail(label + ": neighboring layer changed");
    };

    reset();
    std::array<ninfer::ops::GdnReplayFoldRow, kCapacity> chain{};
    for (int row = 0; row < kCapacity; ++row) {
        chain[row] = {.linear_state_slot = slot_of_record[row],
                      .commit_columns    = columns_of_record[row]};
    }
    ninfer::ops::gdn_replay_fold(records, states.all_layers_view(), chain, stream);
    HIP_CHECK(hipStreamSynchronize(stream));
    const Snapshot expected = snapshot();
    require_state_close(expected.recurrent[0],
                        recurrent_oracle(fixture, 1, {0, 1, 2}, initial_recurrent[1]),
                        "all-layer chain prefix 3");

    // Rows: position 0 takes record row 1, position 1 record row 0.
    ninfer::ops::GdnDeferredFoldRows host_rows{};
    host_rows.row[0] = {.linear_state_slot = 0, .commit_columns = 3, .record_row = 1};
    host_rows.row[1] = {.linear_state_slot = 1, .commit_columns = kWidth, .record_row = 0};
    ninfer::DeviceBuffer device_rows(sizeof(host_rows));
    HIP_CHECK(hipMemcpy(device_rows.data(), &host_rows, sizeof(host_rows), hipMemcpyHostToDevice));
    const auto layer_fold = [&](int layer) {
        return ninfer::ops::GdnLayerFold{
            .records   = &records,
            .recurrent = states.recurrent.at(static_cast<std::size_t>(layer)),
            .conv      = states.conv.at(static_cast<std::size_t>(layer)),
            .layer     = layer,
            .rows      = static_cast<const ninfer::ops::GdnDeferredFoldRows*>(device_rows.data()),
            .batch     = kCapacity,
        };
    };

    reset();
    for (int layer = 0; layer < kLayers; ++layer) {
        ninfer::ops::gdn_replay_fold_layer(layer_fold(layer), stream);
    }
    HIP_CHECK(hipStreamSynchronize(stream));
    require_same(snapshot(), expected, "per-layer deferred fold");

    const std::vector<std::uint8_t> plain_front = run_front(nullptr, stream);
    reset();
    for (int layer = 0; layer < kLayers; ++layer) {
        const ninfer::ops::GdnLayerFold fold  = layer_fold(layer);
        const std::vector<std::uint8_t> front = run_front(&fold, stream);
        if (front != plain_front) fail("front outputs changed by its fused fold");
    }
    require_same(snapshot(), expected, "front-fused deferred fold");

    // The next record round overwrites the persistent record planes. Preparation must stage
    // every old head before that overwrite, including when a row is a strict no-op.
    std::vector<std::unique_ptr<ninfer::DeviceBuffer>> storage;
    const auto allocate = [&](ninfer::DType dtype, std::initializer_list<std::int32_t> shape) {
        const ninfer::Tensor view(nullptr, dtype, shape);
        storage.push_back(std::make_unique<ninfer::DeviceBuffer>(view.bytes()));
        return ninfer::Tensor(storage.back()->data(), dtype, shape);
    };
    auto current           = fixture;
    const auto layer_first = key_index(kOracleLayer, 0, 0, 0, 0);
    for (std::size_t i = layer_first; i < layer_first + kCapacity * kWidth * kQkHeads * kStateDim;
         ++i)
        current.key[i] = bf16(-0.75F * fp32(current.key[i]));
    auto queries = current;
    for (std::size_t i = layer_first; i < layer_first + kCapacity * kWidth * kQkHeads * kStateDim;
         ++i)
        queries.key[i] =
            bf16(0.375F * fp32(current.key[i]) + (static_cast<int>(i % 17) - 8) / 16.F);
    auto q      = allocate(ninfer::DType::BF16, {128, 16, kWidth, kCapacity});
    auto k      = allocate(ninfer::DType::BF16, {128, 16, kWidth, kCapacity});
    auto v      = allocate(ninfer::DType::BF16, {128, 48, kWidth, kCapacity});
    auto g      = allocate(ninfer::DType::FP32, {48, kWidth, kCapacity});
    auto beta   = allocate(ninfer::DType::FP32, {48, kWidth, kCapacity});
    auto output = allocate(ninfer::DType::BF16, {128, 48, kWidth, kCapacity});
    auto slots  = allocate(ninfer::DType::I32, {kCapacity});
    auto valid  = allocate(ninfer::DType::I32, {kCapacity});
    auto live   = records.layer(kOracleLayer, 0, kCapacity);
    ninfer::ops::GdnRecordFold staged{allocate(ninfer::DType::BF16, {128, 16, kWidth, kCapacity}),
                                      allocate(ninfer::DType::BF16, {128, 48, kWidth, kCapacity}),
                                      allocate(ninfer::DType::FP32, {2, 48, kWidth, kCapacity}),
                                      nullptr};
    const std::array<int, kCapacity> current_slots{0, 1}, current_valid{kWidth, kWidth - 1};
    HIP_CHECK(hipMemcpy(slots.data, current_slots.data(), slots.bytes(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(valid.data, current_valid.data(), valid.bytes(), hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(q.data, queries.key.data() + layer_first, q.bytes(), hipMemcpyHostToDevice));
    HIP_CHECK(
        hipMemcpy(k.data, current.key.data() + layer_first, k.bytes(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(v.data, current.value.data() + value_index(kOracleLayer, 0, 0, 0, 0),
                        v.bytes(), hipMemcpyHostToDevice));
    std::vector<float> host_g(kCapacity * kWidth * 48), host_beta(host_g.size());
    for (int b = 0; b < kCapacity; ++b)
        for (int t = 0; t < kWidth; ++t)
            for (int h = 0; h < 48; ++h) {
                const auto index = (b * kWidth + t) * 48 + h;
                host_g[index]    = current.gate[gate_index(kOracleLayer, b, t, h, 0)];
                host_beta[index] = current.gate[gate_index(kOracleLayer, b, t, h, 1)];
            }
    HIP_CHECK(hipMemcpy(g.data, host_g.data(), g.bytes(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(beta.data, host_beta.data(), beta.bytes(), hipMemcpyHostToDevice));
    const auto restore_records = [&] {
        HIP_CHECK(hipMemcpy(live.key.data, fixture.key.data() + layer_first, live.key.bytes(),
                            hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(live.value.data,
                            fixture.value.data() + value_index(kOracleLayer, 0, 0, 0, 0),
                            live.value.bytes(), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(live.gate.data,
                            fixture.gate.data() + gate_index(kOracleLayer, 0, 0, 0, 0),
                            live.gate.bytes(), hipMemcpyHostToDevice));
    };
    const auto record = [&](const ninfer::ops::GdnRecordFold* fold) {
        ninfer::ops::gated_delta_net_replay_record(
            q, k, v, g, beta, 1.F / std::sqrt(128.F), states.recurrent.at(kOracleLayer), valid,
            slots, live.key, live.value, live.gate, output, stream, nullptr, nullptr, {}, fold);
    };
    for (bool noop : {false, true}) {
        host_rows.row[0].commit_columns = noop ? 0 : 3;
        HIP_CHECK(
            hipMemcpy(device_rows.data(), &host_rows, sizeof(host_rows), hipMemcpyHostToDevice));
        reset();
        restore_records();
        ninfer::ops::gdn_replay_fold_layer(layer_fold(kOracleLayer), stream);
        record(nullptr);
        HIP_CHECK(hipStreamSynchronize(stream));
        const auto expected_output = copy_tensor<hip_bfloat16>(output);
        const auto expected_state  = snapshot();
        const auto expected_key    = copy_tensor<hip_bfloat16>(live.key);
        const auto expected_value  = copy_tensor<hip_bfloat16>(live.value);
        const auto expected_gate   = copy_tensor<float>(live.gate);
        // Independent complete FP64 fold -> represented FP32 state -> current recurrence,
        // with independent Q and K, followed by the recurrence qualifier's BF16 output criterion.
        for (int b = 0; b < kCapacity; ++b) {
            const int source_row = host_rows.row[b].record_row;
            std::vector<int> old_sequence(host_rows.row[b].commit_columns);
            std::iota(old_sequence.begin(), old_sequence.end(), 0);
            const auto old_ideal =
                recurrent_oracle(fixture, source_row, old_sequence, initial_recurrent[source_row]);
            const std::vector<float> represented(old_ideal.begin(), old_ideal.end());
            std::vector<int> sequence(current_valid[b]);
            std::iota(sequence.begin(), sequence.end(), 0);
            std::vector<double> ideal;
            (void)recurrent_oracle(current, b, sequence, represented, &ideal, &queries);
            double error2 = 0.0, oracle2 = 0.0, maximum = 0.0;
            for (std::size_t i = 0; i < ideal.size(); ++i) {
                const double value = fp32(expected_output[b * kWidth * 48 * 128 + i]);
                const double error = value - ideal[i];
                maximum            = std::max(maximum, std::abs(error));
                error2 += error * error;
                oracle2 += ideal[i] * ideal[i];
            }
            if (maximum > 0.025 || std::sqrt(error2 / std::max(oracle2, 1e-30)) > 0.02)
                fail("prepared fold-record independent FP64 output mismatch");
            for (int t = current_valid[b]; t < kWidth; ++t)
                for (int i = 0; i < 48 * 128; ++i)
                    if (expected_output[(b * kWidth + t) * 48 * 128 + i].data != 0)
                        fail("prepared fold-record invalid suffix is not exact zero");
        }
        const auto fused = [&] {
            ninfer::ops::gdn_replay_prepare_layer(layer_fold(kOracleLayer), staged, stream);
            record(&staged);
        };
        const auto verify = [&] {
            HIP_CHECK(hipStreamSynchronize(stream));
            require_same(snapshot(), expected_state, "prepared fold-record state/history");
            require_bf16_exact(copy_tensor<hip_bfloat16>(output), expected_output,
                               "fold-record output");
            require_bf16_exact(copy_tensor<hip_bfloat16>(live.key), expected_key,
                               "fold-record key");
            require_bf16_exact(copy_tensor<hip_bfloat16>(live.value), expected_value,
                               "fold-record value");
            if (copy_tensor<float>(live.gate) != expected_gate) fail("fold-record gate differs");
        };
        reset();
        restore_records();
        fused();
        verify();
        ninfer::DecodeGraphDefinition definition;
        definition.capture(stream, fused);
        ninfer::DecodeGraphExecutable executable;
        executable.instantiate(definition);
        for (int replay = 0; replay < 3; ++replay) {
            reset();
            restore_records();
            executable.launch(stream);
            verify();
        }
    }
    reset();
    restore_records();
    host_rows.row[0].commit_columns = 3;
    HIP_CHECK(hipMemcpy(device_rows.data(), &host_rows, sizeof(host_rows), hipMemcpyHostToDevice));
    std::cout << "prepared fold-record FP64/output/state/alias/noop/graph=PASS\n";

    ninfer::ops::GdnLayerFold oversized = layer_fold(kOracleLayer);
    oversized.batch                     = kCapacity + 1;
    require_invalid([&] { ninfer::ops::gdn_replay_fold_layer(oversized, stream); },
                    "deferred batch above record capacity");
}

// Real W8 server batches: every lane has permuted source rows and state slots, including
// zero, short and full accepted prefixes. Fixture columns repeat after six, while the independent
// recurrence oracle evaluates all eight transitions rather than copying a device reduction.
void qualify_w8_batch(int batch, const Fixture& fixture, hipStream_t stream) {
    constexpr int width = 8;
    ninfer::LayoutBuilder builder;
    const auto layout = ninfer::plan_gdn_replay_records(builder, {.layers          = 48,
                                                                  .record_capacity = batch,
                                                                  .width           = width,
                                                                  .conv_channels   = 10240,
                                                                  .qk_heads        = 16,
                                                                  .value_heads     = 48,
                                                                  .key_dim         = 128,
                                                                  .value_dim       = 128});
    ninfer::DeviceBuffer record_storage(builder.finish(256));
    ninfer::GdnReplayRecords records({record_storage.data(), record_storage.size()}, layout);
    auto live = records.layer(0, 0, batch);
    std::vector<std::unique_ptr<ninfer::DeviceBuffer>> storage;
    const auto allocate = [&](ninfer::DType dtype, std::initializer_list<std::int32_t> shape) {
        const ninfer::Tensor view(nullptr, dtype, shape);
        storage.push_back(std::make_unique<ninfer::DeviceBuffer>(view.bytes()));
        return ninfer::Tensor(storage.back()->data(), dtype, shape);
    };
    auto state  = allocate(ninfer::DType::FP32, {128, 128, 48, batch});
    auto conv   = allocate(ninfer::DType::BF16, {10240, 3, batch});
    auto q      = allocate(ninfer::DType::BF16, {128, 16, width, batch});
    auto k      = allocate(ninfer::DType::BF16, {128, 16, width, batch});
    auto v      = allocate(ninfer::DType::BF16, {128, 48, width, batch});
    auto g      = allocate(ninfer::DType::FP32, {48, width, batch});
    auto beta   = allocate(ninfer::DType::FP32, {48, width, batch});
    auto output = allocate(ninfer::DType::BF16, {128, 48, width, batch});
    auto slots  = allocate(ninfer::DType::I32, {batch});
    auto valid  = allocate(ninfer::DType::I32, {batch});
    auto rows   = allocate(ninfer::DType::I32, {4, 8});
    ninfer::ops::GdnRecordFold staged{allocate(ninfer::DType::BF16, {128, 16, width, batch}),
                                      allocate(ninfer::DType::BF16, {128, 48, width, batch}),
                                      allocate(ninfer::DType::FP32, {2, 48, width, batch}),
                                      nullptr};
    std::vector<hip_bfloat16> old_k(batch * width * 16 * 128), old_v(batch * width * 48 * 128),
        old_c(batch * width * 10240), current_k(old_k.size()), current_q(old_k.size());
    std::vector<float> old_gate(batch * width * 48 * 2), host_g(batch * width * 48),
        host_beta(host_g.size());
    auto current     = fixture;
    const auto begin = key_index(kOracleLayer, 0, 0, 0, 0);
    for (std::size_t j = begin; j < begin + kCapacity * kWidth * 16 * 128; ++j)
        current.key[j] = bf16(-0.75F * fp32(current.key[j]));
    auto queries = current;
    for (std::size_t j = begin; j < begin + kCapacity * kWidth * 16 * 128; ++j)
        queries.key[j] =
            bf16(0.375F * fp32(current.key[j]) + (static_cast<int>(j % 17) - 8) / 16.F);
    for (int b = 0; b < batch; ++b)
        for (int t = 0; t < width; ++t) {
            const auto column = b * width + t;
            std::copy_n(fixture.key.data() + key_index(kOracleLayer, b % 2, t % kWidth, 0, 0),
                        16 * 128, old_k.data() + column * 16 * 128);
            std::copy_n(current.key.data() + key_index(kOracleLayer, b % 2, t % kWidth, 0, 0),
                        16 * 128, current_k.data() + column * 16 * 128);
            std::copy_n(queries.key.data() + key_index(kOracleLayer, b % 2, t % kWidth, 0, 0),
                        16 * 128, current_q.data() + column * 16 * 128);
            std::copy_n(fixture.value.data() + value_index(kOracleLayer, b % 2, t % kWidth, 0, 0),
                        48 * 128, old_v.data() + column * 48 * 128);
            std::copy_n(fixture.conv.data() + conv_index(kOracleLayer, b % 2, t % kWidth, 0), 10240,
                        old_c.data() + column * 10240);
            for (int h = 0; h < 48; ++h) {
                const auto index = (column * 48 + h) * 2;
                old_gate[index]  = fixture.gate[gate_index(kOracleLayer, b % 2, t % kWidth, h, 0)];
                old_gate[index + 1] =
                    fixture.gate[gate_index(kOracleLayer, b % 2, t % kWidth, h, 1)];
                host_g[index / 2]    = old_gate[index];
                host_beta[index / 2] = old_gate[index + 1];
            }
        }
    HIP_CHECK(hipMemcpy(q.data, current_q.data(), q.bytes(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(k.data, current_k.data(), k.bytes(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(v.data, old_v.data(), v.bytes(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(g.data, host_g.data(), g.bytes(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(beta.data, host_beta.data(), beta.bytes(), hipMemcpyHostToDevice));
    std::vector<int> host_slots(batch), host_valid(batch);
    ninfer::ops::GdnDeferredFoldRows host_rows{};
    constexpr int commits[8] = {0, 1, 2, 3, 6, 8, 4, 7};
    for (int b = 0; b < batch; ++b) {
        host_slots[b]    = (b + 3) % batch;
        host_valid[b]    = width - b % 3;
        host_rows.row[b] = {.linear_state_slot = host_slots[b],
                            .commit_columns    = commits[b],
                            .record_row        = (b + 1) % batch};
    }
    HIP_CHECK(hipMemcpy(slots.data, host_slots.data(), slots.bytes(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(valid.data, host_valid.data(), valid.bytes(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(rows.data, &host_rows, sizeof(host_rows), hipMemcpyHostToDevice));
    const ninfer::ops::GdnLayerFold fold{
        &records, state, conv, 0, static_cast<const ninfer::ops::GdnDeferredFoldRows*>(rows.data),
        batch};
    const auto reset = [&] {
        HIP_CHECK(hipMemcpy(live.key.data, old_k.data(), live.key.bytes(), hipMemcpyHostToDevice));
        HIP_CHECK(
            hipMemcpy(live.value.data, old_v.data(), live.value.bytes(), hipMemcpyHostToDevice));
        HIP_CHECK(
            hipMemcpy(live.gate.data, old_gate.data(), live.gate.bytes(), hipMemcpyHostToDevice));
        HIP_CHECK(
            hipMemcpy(live.conv.data, old_c.data(), live.conv.bytes(), hipMemcpyHostToDevice));
        for (int b = 0; b < batch; ++b) {
            const auto initial = initial_state(b);
            const auto history = initial_conv(b);
            HIP_CHECK(hipMemcpy(static_cast<float*>(state.data) + b * 48 * 128 * 128,
                                initial.data(), initial.size() * sizeof(float),
                                hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(static_cast<hip_bfloat16*>(conv.data) + b * 3 * 10240,
                                history.data(), history.size() * sizeof(hip_bfloat16),
                                hipMemcpyHostToDevice));
        }
    };
    const auto record = [&](const ninfer::ops::GdnRecordFold* old) {
        ninfer::ops::gated_delta_net_replay_record(q, k, v, g, beta, 1.F / std::sqrt(128.F), state,
                                                   valid, slots, live.key, live.value, live.gate,
                                                   output, stream, nullptr, nullptr, {}, old);
    };
    reset();
    ninfer::ops::gdn_replay_fold_layer(fold, stream);
    record(nullptr);
    HIP_CHECK(hipStreamSynchronize(stream));
    const auto expected_state  = copy_tensor<float>(state);
    const auto expected_conv   = copy_tensor<hip_bfloat16>(conv);
    const auto expected_output = copy_tensor<hip_bfloat16>(output);
    reset();
    ninfer::ops::gdn_replay_prepare_layer(fold, staged, stream);
    record(&staged);
    HIP_CHECK(hipStreamSynchronize(stream));
    if (copy_tensor<float>(state) != expected_state) fail("W8 folded state differs");
    require_bf16_exact(copy_tensor<hip_bfloat16>(conv), expected_conv, "W8 convolution");
    const auto actual = copy_tensor<hip_bfloat16>(output);
    require_bf16_exact(actual, expected_output, "W8 output");
    for (int b = 0; b < batch; ++b) {
        const int slot = host_slots[b], source = host_rows.row[b].record_row;
        std::vector<int> previous_sequence(commits[b]), current_sequence(host_valid[b]);
        for (std::size_t t = 0; t < previous_sequence.size(); ++t)
            previous_sequence[t] = t % kWidth;
        for (std::size_t t = 0; t < current_sequence.size(); ++t) current_sequence[t] = t % kWidth;
        const auto old_ideal =
            recurrent_oracle(fixture, source % 2, previous_sequence, initial_state(slot));
        require_state_close(
            std::vector<float>(expected_state.begin() + slot * 48 * 128 * 128,
                               expected_state.begin() + (slot + 1) * 48 * 128 * 128),
            old_ideal, "W8 independent state oracle");
        const std::vector<float> represented(old_ideal.begin(), old_ideal.end());
        std::vector<double> ideal;
        (void)recurrent_oracle(current, b % 2, current_sequence, represented, &ideal, &queries);
        double error2 = 0.0, oracle2 = 0.0, maximum = 0.0;
        for (std::size_t t = 0; t < ideal.size(); ++t) {
            const double error = fp32(actual[b * width * 48 * 128 + t]) - ideal[t];
            maximum            = std::max(maximum, std::abs(error));
            error2 += error * error;
            oracle2 += ideal[t] * ideal[t];
        }
        if (maximum > 0.025 || std::sqrt(error2 / std::max(oracle2, 1e-30)) > 0.02)
            fail("W8 independent output oracle mismatch");
        for (int t = host_valid[b]; t < width; ++t)
            for (int f = 0; f < 48 * 128; ++f)
                if (actual[(b * width + t) * 48 * 128 + f].data != 0)
                    fail("W8 inactive suffix not zero");
    }
    std::cout << "prepared fold-record B=" << batch << " W8 FP64/state/output/alias=PASS\n";
}

// A full compact batch: every kernarg row 0..C-1 folds its own distinct record row (prefix and
// tree-path modes, varied extents) into a permuted, distinct destination slot, so a wrong row
// index or a kernarg row table shorter than C fails the same oracle criteria as the B=2 case.
void qualify_full_batch(hipStream_t stream) {
    constexpr int batch = static_cast<int>(ninfer::kMaximumConcurrency);
    ninfer::LayoutBuilder record_builder;
    const ninfer::GdnReplayRecordLayout record_layout =
        ninfer::plan_gdn_replay_records(record_builder, {.layers          = kLayers,
                                                         .record_capacity = batch,
                                                         .width           = kWidth,
                                                         .conv_channels   = kConvChannels,
                                                         .qk_heads        = kQkHeads,
                                                         .value_heads     = kValueHeads,
                                                         .key_dim         = kStateDim,
                                                         .value_dim       = kStateDim});
    ninfer::DeviceBuffer record_storage(record_builder.finish(256));
    ninfer::GdnReplayRecords records({record_storage.data(), record_storage.size()}, record_layout);
    const Fixture fixture = make_fixture(records);
    HIP_CHECK(hipMemcpy(records.conv.data, fixture.conv.data(), records.conv.bytes(),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(records.key.data, fixture.key.data(), records.key.bytes(),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(records.value.data, fixture.value.data(), records.value.bytes(),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(records.gate.data, fixture.gate.data(), records.gate.bytes(),
                        hipMemcpyHostToDevice));

    ninfer::LayoutBuilder state_builder;
    const ninfer::LinearAttentionStatePoolLayout state_layout =
        ninfer::plan_linear_attention_state_pool(state_builder,
                                                 {.layers         = kLayers,
                                                  .conv_channels  = kConvChannels,
                                                  .conv_width     = 3,
                                                  .value_heads    = kValueHeads,
                                                  .value_head_dim = kStateDim,
                                                  .key_head_dim   = kStateDim,
                                                  .slot_count     = batch,
                                                  .conv_dtype     = ninfer::DType::BF16});
    ninfer::DeviceBuffer state_storage(state_builder.finish(256));
    state_storage.fill(0);
    ninfer::LinearAttentionStatePool states({state_storage.data(), state_storage.size()},
                                            state_layout);

    std::vector<ninfer::ops::GdnReplayFoldRow> rows(batch);
    std::vector<std::vector<int>> sequences(batch);
    std::vector<std::vector<float>> initial_recurrent(batch);
    std::vector<std::vector<hip_bfloat16>> initial_history(batch);
    for (int row = 0; row < batch; ++row) {
        // 5 is coprime to every power-of-two C, so destinations form a non-identity permutation.
        const int slot = (row * 5 + 3) % batch;
        rows[row]      = {.linear_state_slot = slot};
        if (row % 2 == 0) {
            const int columns        = 1 + (row / 2) % kWidth;
            rows[row].commit_columns = columns;
            sequences[row].resize(columns);
            std::iota(sequences[row].begin(), sequences[row].end(), 0);
        } else {
            const int length         = 2 + (row / 2) % (kWidth - 1);
            rows[row].commit_columns = 1; // Ignored in path mode.
            rows[row].path_length    = length;
            sequences[row].push_back(0);
            for (int step = 1; step < length; ++step)
                sequences[row].push_back(1 + (row + step * 2) % (kWidth - 1));
            std::copy(sequences[row].begin(), sequences[row].end(), rows[row].path.begin());
        }
        initial_recurrent[row] = initial_state(row);
        initial_history[row]   = initial_conv(row);
        HIP_CHECK(hipMemcpyAsync(
            states.recurrent_slot(kOracleLayer, slot).data, initial_recurrent[row].data(),
            initial_recurrent[row].size() * sizeof(float), hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(
            states.conv_slot(kOracleLayer, slot).data, initial_history[row].data(),
            initial_history[row].size() * sizeof(hip_bfloat16), hipMemcpyHostToDevice, stream));
    }
    ninfer::ops::gdn_replay_fold(records, states.all_layers_view(), rows, stream);
    HIP_CHECK(hipStreamSynchronize(stream));

    for (int row = 0; row < batch; ++row) {
        const int slot  = rows[row].linear_state_slot;
        const auto name = "B" + std::to_string(batch) + " row " + std::to_string(row);
        const std::vector<float> actual_state =
            copy_tensor<float>(states.recurrent_slot(kOracleLayer, slot));
        require_state_close(actual_state,
                            recurrent_oracle(fixture, row, sequences[row], initial_recurrent[row]),
                            name + " independent FP64 recurrence");
        const auto ordinary_state =
            snapshot_state(fixture, row, sequences[row], initial_recurrent[row], stream);
        for (std::size_t i = 0; i < actual_state.size(); ++i)
            if (std::bit_cast<std::uint32_t>(actual_state[i]) !=
                std::bit_cast<std::uint32_t>(ordinary_state[i]))
                fail(name + ": replay/ordinary state profile mismatch");
        require_bf16_exact(copy_tensor<hip_bfloat16>(states.conv_slot(kOracleLayer, slot)),
                           conv_oracle(fixture, row, sequences[row], initial_history[row]),
                           name + " convolution tail");
    }
    const auto untouched_layer =
        copy_tensor<float>(states.recurrent_slot(kOracleLayer + 1, batch - 1));
    if (std::any_of(untouched_layer.begin(), untouched_layer.end(),
                    [](float value) { return std::bit_cast<std::uint32_t>(value) != 0U; }))
        fail("B" + std::to_string(batch) + " record outer index bled into a neighboring layer");
    std::cout << "direct fold B=" << batch << " W=" << kWidth
              << " prefix/path FP64/state/conv=PASS\n";
}

void qualify(hipStream_t stream) {
    ninfer::LayoutBuilder record_builder;
    const ninfer::GdnReplayRecordLayout record_layout =
        ninfer::plan_gdn_replay_records(record_builder, {.layers          = kLayers,
                                                         .record_capacity = kCapacity,
                                                         .width           = kWidth,
                                                         .conv_channels   = kConvChannels,
                                                         .qk_heads        = kQkHeads,
                                                         .value_heads     = kValueHeads,
                                                         .key_dim         = kStateDim,
                                                         .value_dim       = kStateDim});
    ninfer::DeviceBuffer record_storage(record_builder.finish(256));
    ninfer::GdnReplayRecords records({record_storage.data(), record_storage.size()}, record_layout);
    const Fixture fixture = make_fixture(records);
    qualify_w8_batch(6, fixture, stream);
    qualify_w8_batch(8, fixture, stream);
    HIP_CHECK(hipMemcpy(records.conv.data, fixture.conv.data(), records.conv.bytes(),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(records.key.data, fixture.key.data(), records.key.bytes(),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(records.value.data, fixture.value.data(), records.value.bytes(),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(records.gate.data, fixture.gate.data(), records.gate.bytes(),
                        hipMemcpyHostToDevice));

    ninfer::LayoutBuilder state_builder;
    const ninfer::LinearAttentionStatePoolLayout state_layout =
        ninfer::plan_linear_attention_state_pool(state_builder,
                                                 {.layers         = kLayers,
                                                  .conv_channels  = kConvChannels,
                                                  .conv_width     = 3,
                                                  .value_heads    = kValueHeads,
                                                  .value_head_dim = kStateDim,
                                                  .key_head_dim   = kStateDim,
                                                  .slot_count     = kSlots,
                                                  .conv_dtype     = ninfer::DType::BF16});
    ninfer::DeviceBuffer state_storage(state_builder.finish(256));
    state_storage.fill(0);
    ninfer::LinearAttentionStatePool states({state_storage.data(), state_storage.size()},
                                            state_layout);

    const std::array<std::vector<float>, kCapacity> initial_recurrent{initial_state(0),
                                                                      initial_state(1)};
    const std::array<std::vector<hip_bfloat16>, kCapacity> initial_history{initial_conv(0),
                                                                           initial_conv(1)};
    // Physical row zero deliberately updates slot one; row one updates slot zero. This proves
    // record-row identity is not inferred from the destination state slot.
    const std::array<int, kCapacity> slots{1, 0};
    for (int row = 0; row < kCapacity; ++row) {
        HIP_CHECK(hipMemcpyAsync(
            states.recurrent_slot(kOracleLayer, slots[row]).data, initial_recurrent[row].data(),
            initial_recurrent[row].size() * sizeof(float), hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(
            states.conv_slot(kOracleLayer, slots[row]).data, initial_history[row].data(),
            initial_history[row].size() * sizeof(hip_bfloat16), hipMemcpyHostToDevice, stream));
    }

    ninfer::ops::GdnReplayFoldRow prefix{.linear_state_slot = slots[0], .commit_columns = kWidth};
    ninfer::ops::GdnReplayFoldRow path{};
    path.linear_state_slot = slots[1];
    path.commit_columns    = 1; // Ignored in path mode.
    path.path              = {0, 1, 4, 2, 5};
    path.path_length       = 5;
    const std::array rows{prefix, path};
    ninfer::ops::gdn_replay_fold(records, states.all_layers_view(), rows, stream);
    HIP_CHECK(hipStreamSynchronize(stream));

    const std::array<std::vector<int>, kCapacity> sequences{std::vector<int>{0, 1, 2, 3, 4, 5},
                                                            std::vector<int>{0, 1, 4, 2, 5}};
    for (int row = 0; row < kCapacity; ++row) {
        const std::vector<float> actual_state =
            copy_tensor<float>(states.recurrent_slot(kOracleLayer, slots[row]));
        require_state_close(actual_state,
                            recurrent_oracle(fixture, row, sequences[row], initial_recurrent[row]),
                            "row " + std::to_string(row) + " independent FP64 recurrence");
        const auto ordinary_state =
            snapshot_state(fixture, row, sequences[row], initial_recurrent[row], stream);
        std::size_t state_differences = 0;
        for (std::size_t i = 0; i < actual_state.size(); ++i)
            state_differences += std::bit_cast<std::uint32_t>(actual_state[i]) !=
                                 std::bit_cast<std::uint32_t>(ordinary_state[i]);
        if (state_differences)
            fail("replay/ordinary state profile mismatch row=" + std::to_string(row) +
                 " elements=" + std::to_string(state_differences));
        const std::vector<hip_bfloat16> actual_conv =
            copy_tensor<hip_bfloat16>(states.conv_slot(kOracleLayer, slots[row]));
        require_bf16_exact(actual_conv,
                           conv_oracle(fixture, row, sequences[row], initial_history[row]),
                           "row " + std::to_string(row) + " convolution tail");
    }

    // A layer with all-zero represented records and initial state remains exact zero, proving the
    // launch covers the all-layer domain without bleeding the oracle layer's record outer index.
    const auto untouched_layer = copy_tensor<float>(states.recurrent_slot(kOracleLayer + 1, 0));
    if (std::any_of(untouched_layer.begin(), untouched_layer.end(),
                    [](float value) { return std::bit_cast<std::uint32_t>(value) != 0U; })) {
        fail("record outer index bled into a neighboring layer");
    }

    const auto state_before_noop0 = copy_tensor<float>(states.recurrent_slot(kOracleLayer, 0));
    const auto state_before_noop1 = copy_tensor<float>(states.recurrent_slot(kOracleLayer, 1));
    const auto conv_before_noop0  = copy_tensor<hip_bfloat16>(states.conv_slot(kOracleLayer, 0));
    const auto conv_before_noop1  = copy_tensor<hip_bfloat16>(states.conv_slot(kOracleLayer, 1));
    ninfer::ops::GdnReplayFoldRow zero_prefix{.linear_state_slot = 0, .commit_columns = 0};
    ninfer::ops::GdnReplayFoldRow zero_path{.linear_state_slot = 1, .commit_columns = kWidth};
    zero_path.path_length = 0;
    const std::array noops{zero_prefix, zero_path};
    ninfer::ops::gdn_replay_fold(records, states.all_layers_view(), noops, stream);
    HIP_CHECK(hipStreamSynchronize(stream));
    if (copy_tensor<float>(states.recurrent_slot(kOracleLayer, 0)) != state_before_noop0 ||
        copy_tensor<float>(states.recurrent_slot(kOracleLayer, 1)) != state_before_noop1) {
        fail("zero-extent replay changed recurrent state");
    }
    require_bf16_exact(copy_tensor<hip_bfloat16>(states.conv_slot(kOracleLayer, 0)),
                       conv_before_noop0, "zero-prefix convolution no-op");
    require_bf16_exact(copy_tensor<hip_bfloat16>(states.conv_slot(kOracleLayer, 1)),
                       conv_before_noop1, "zero-path convolution no-op");

    std::array duplicate{prefix, path};
    duplicate[1].linear_state_slot = duplicate[0].linear_state_slot;
    require_invalid(
        [&] { ninfer::ops::gdn_replay_fold(records, states.all_layers_view(), duplicate, stream); },
        "duplicate destination slots");
    std::array invalid_path{path};
    invalid_path[0].path[0] = kWidth;
    require_invalid(
        [&] {
            ninfer::ops::gdn_replay_fold(records, states.all_layers_view(), invalid_path, stream);
        },
        "out-of-range path index");

    qualify_deferred(records, states, fixture, initial_recurrent, initial_history, stream);
    qualify_full_batch(stream);

    require_bf16_exact(copy_tensor<hip_bfloat16>(records.conv), fixture.conv,
                       "read-only convolution records");
    require_bf16_exact(copy_tensor<hip_bfloat16>(records.key), fixture.key,
                       "read-only key records");
    require_bf16_exact(copy_tensor<hip_bfloat16>(records.value), fixture.value,
                       "read-only value records");
    if (copy_tensor<float>(records.gate) != fixture.gate) {
        fail("read-only gate records changed");
    }
}

} // namespace

int main() {
    try {
        HIP_CHECK(hipSetDevice(select_r9700()));
        hipStream_t stream = nullptr;
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        try {
            qualify(stream);
            HIP_CHECK(hipStreamDestroy(stream));
        } catch (...) {
            (void)hipStreamDestroy(stream);
            throw;
        }
        std::cout << "r9700_gdn_replay_fold: PASS gfx1201 wave32 Qwen3.8 L48/Hq16/Hv48 "
                     "B=2,16 W=6 prefix/path/no-op/read-only ordinary-state-exact "
                     "deferred-layer/front-fused-bitwise max_state_abs="
                  << g_max_state_abs << " max_state_rel_l2=" << g_max_state_relative_l2 << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "r9700_gdn_replay_fold: FAIL " << error.what() << '\n';
        return 1;
    }
}
