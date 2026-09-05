#include "ninfer/ops/ple.h"
#include "ops/op_tester.h"
#include "ops/launcher/ple.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::size_t kQ8RowBytes = (ops::kPleEmbeddingWidth / 32) * 34;

std::uint16_t f16_bits(float value) {
    const __half encoded = __float2half_rn(value);
    std::uint16_t bits;
    std::memcpy(&bits, &encoded, sizeof(bits));
    return bits;
}

float f16_value(std::uint16_t bits) {
    __half encoded;
    std::memcpy(&encoded, &bits, sizeof(bits));
    return __half2float(encoded);
}

ops::PleMappedIq4NlTable mapped_view(const std::vector<std::uint8_t>& table, std::uint64_t rows) {
    return {table.data(), rows, table.size()};
}

double iq4_oracle(const std::uint8_t* block, int index) {
    constexpr int codebook[]{-127, -104, -83, -65, -49, -35, -22, -10,
                             1,    13,   25,  38,  53,  69,  89,  113};
    const std::uint16_t scale_bits =
        static_cast<std::uint16_t>(block[0]) | (static_cast<std::uint16_t>(block[1]) << 8U);
    const std::uint8_t packed = block[2 + (index & 15)];
    const int code = index < 16 ? packed & 0x0fU : packed >> 4U;
    return static_cast<double>(f16_value(scale_bits)) * codebook[code];
}

int conv_source_layout_witness() {
    constexpr int channel = 7;
    std::vector<float> residual(ops::kPleChannels, 0.0F);
    std::vector<float> gated(ops::kPleChannels, 0.0F);
    std::vector<float> weight(static_cast<std::size_t>(ops::kPleChannels) * 4, 0.0F);
    std::vector<float> old_state(static_cast<std::size_t>(ops::kPleChannels) * 9, 0.0F);
    std::vector<float> current(ops::kPleChannels, 0.0F);
    weight[4 * channel] = 1.0F;
    weight[4 * channel + 1] = 2.0F;
    weight[4 * channel + 2] = 4.0F;
    weight[4 * channel + 3] = 8.0F;
    old_state[channel] = 1.0F;                            // logical -9
    old_state[3 * ops::kPleChannels + channel] = 2.0F;   // logical -6
    old_state[6 * ops::kPleChannels + channel] = 3.0F;   // logical -3
    current[channel] = 4.0F;                             // logical 0
    auto d_residual = to_device_bf16(residual);
    auto d_gated = to_device_f32(gated);
    auto d_weight = to_device_f32(weight);
    auto d_old = to_device_bf16(old_state);
    auto d_current = to_device_bf16(current);
    GuardedDeviceBuffer d_output(ops::kPleChannels * sizeof(std::uint16_t));
    d_output.fill(0);
    Tensor residual_t(d_residual.p, DType::BF16, {ops::kPleEmbeddingWidth, 4, 1});
    Tensor gated_t(d_gated.p, DType::FP32, {ops::kPleChannels, 1});
    Tensor weight_t(d_weight.p, DType::FP32, {4, ops::kPleChannels});
    Tensor old_t(d_old.p, DType::BF16, {ops::kPleChannels, 9});
    Tensor current_t(d_current.p, DType::BF16, {ops::kPleChannels, 1});
    Tensor output_t(d_output.data(), DType::BF16, {ops::kPleEmbeddingWidth, 4, 1});
    ops::detail::ple_conv_inject_launch(residual_t, gated_t, weight_t, old_t, current_t,
                                         output_t, nullptr);
    cuda_synchronize();
    const auto output = from_device<std::uint16_t>(d_output.data(), ops::kPleChannels);
    // 1*1 + 2*2 + 4*3 + 8*4 = 49, whose SiLU rounds to exact BF16 49. Tap-major indexing reads
    // zeros at channel 7, so this is a strict transpose witness over source-order bytes.
    int failures = output[channel] == f32_to_bf16(49.0F) ? 0 : 1;
    if (failures != 0) { std::cerr << "FAIL PLE GGUF channel-major conv layout witness\n"; }
    failures += d_output.verify_guards("PLE layout output");
    return failures;
}

int staging_decode_case() {
    constexpr int rows = 23;
    std::vector<std::uint8_t> table(static_cast<std::size_t>(rows) * ops::kPleIq4NlRowBytes);
    for (int row = 0; row < rows; ++row) {
        for (int block = 0; block < 5; ++block) {
            auto* encoded = table.data() + static_cast<std::size_t>(row) * ops::kPleIq4NlRowBytes +
                            block * ops::kPleIq4NlBlockBytes;
            const std::uint16_t scale = f16_bits((row + block + 1) / 256.0F);
            encoded[0] = static_cast<std::uint8_t>(scale);
            encoded[1] = static_cast<std::uint8_t>(scale >> 8U);
            for (int lane = 0; lane < 16; ++lane) {
                encoded[2 + lane] = static_cast<std::uint8_t>(((row + block + lane + 7) & 15) << 4U) |
                                     static_cast<std::uint8_t>((row + 3 * block + lane) & 15);
            }
        }
    }
    const std::array<std::int32_t, ops::kPleHeads> ids{
        22, 0, 17, 3, 3, 9, 1, 21, 4, 16, 8, 12, 6, 19, 2, 14,
    };
    PinnedHostBuffer pinned(ops::kPleStagedBytes);
    GuardedDeviceBuffer staged(ops::kPleStagedBytes);
    GuardedDeviceBuffer output(static_cast<std::size_t>(ops::kPleHeads) * ops::kPleRowWidth *
                               sizeof(std::uint16_t));
    Tensor staged_tensor(staged.data(), DType::U8, {ops::kPleIq4NlRowBytes, ops::kPleHeads});
    Tensor output_tensor(output.data(), DType::BF16, {ops::kPleRowWidth, ops::kPleHeads});
    ops::ple_iq4_nl_stage_rows(mapped_view(table, rows), ids, pinned.data(), pinned.size(),
                               staged_tensor, nullptr);
    ops::ple_iq4_nl_decode_rows(staged_tensor, output_tensor, nullptr);
    cuda_synchronize();

    std::vector<std::uint8_t> expected_staged(ops::kPleStagedBytes);
    for (int head = 0; head < ops::kPleHeads; ++head) {
        std::memcpy(expected_staged.data() + head * ops::kPleIq4NlRowBytes,
                    table.data() + static_cast<std::size_t>(ids[head]) * ops::kPleIq4NlRowBytes,
                    ops::kPleIq4NlRowBytes);
    }
    std::vector<std::uint8_t> actual_pinned(ops::kPleStagedBytes);
    std::memcpy(actual_pinned.data(), pinned.data(), actual_pinned.size());
    int failures = verify_exact("PLE pinned selected rows", actual_pinned, expected_staged);
    failures += verify_exact("PLE device selected rows",
                             from_device<std::uint8_t>(staged.data(), ops::kPleStagedBytes),
                             expected_staged);

    const auto actual = from_device<std::uint16_t>(output.data(),
                                                   static_cast<std::size_t>(ops::kPleHeads) *
                                                       ops::kPleRowWidth);
    std::vector<std::uint16_t> expected(actual.size());
    for (int head = 0; head < ops::kPleHeads; ++head) {
        const auto* row = table.data() + static_cast<std::size_t>(ids[head]) * ops::kPleIq4NlRowBytes;
        for (int d = 0; d < ops::kPleRowWidth; ++d) {
            const auto* block = row + (d / 32) * 18;
            expected[d + ops::kPleRowWidth * head] =
                f32_to_bf16(static_cast<float>(iq4_oracle(block, d % 32)));
        }
    }
    failures += verify_exact("PLE exact IQ4_NL decode", actual, expected);
    failures += staged.verify_guards("PLE staged rows");
    failures += output.verify_guards("PLE decoded embedding");
    return failures;
}

int batched_staging_decode_case(int width) {
    constexpr int rows = 41;
    const std::size_t slots = static_cast<std::size_t>(ops::kPleHeads) * width;
    const std::size_t staged_bytes = static_cast<std::size_t>(ops::kPleStagedBytes) * width;
    std::vector<std::uint8_t> table(static_cast<std::size_t>(rows) * ops::kPleIq4NlRowBytes);
    for (int row = 0; row < rows; ++row) {
        for (int block = 0; block < 5; ++block) {
            auto* encoded = table.data() + static_cast<std::size_t>(row) * ops::kPleIq4NlRowBytes +
                            block * ops::kPleIq4NlBlockBytes;
            const std::uint16_t scale = f16_bits((2 * row + block + 3) / 512.0F);
            encoded[0] = static_cast<std::uint8_t>(scale);
            encoded[1] = static_cast<std::uint8_t>(scale >> 8U);
            for (int lane = 0; lane < 16; ++lane) {
                const int low = (row + 5 * block + lane) & 15;
                const int high = (3 * row + block + 2 * lane + 1) & 15;
                encoded[2 + lane] = static_cast<std::uint8_t>(low | (high << 4));
            }
        }
    }
    std::vector<std::int32_t> ids(slots);
    for (int token = 0; token < width; ++token) {
        for (int head = 0; head < ops::kPleHeads; ++head) {
            ids[head + ops::kPleHeads * token] =
                (token == width - 1 && (head == 3 || head == 11))
                    ? 7
                    : (17 * token + 5 * head + 2) % rows;
        }
    }

    PinnedHostBuffer pinned(staged_bytes);
    GuardedDeviceBuffer staged(staged_bytes);
    GuardedDeviceBuffer output(slots * ops::kPleRowWidth * sizeof(std::uint16_t));
    Tensor staged_tensor(staged.data(), DType::U8,
                         {ops::kPleIq4NlRowBytes, ops::kPleHeads, width});
    Tensor output_tensor(output.data(), DType::BF16,
                         {ops::kPleRowWidth, ops::kPleHeads, width});
    ops::ple_iq4_nl_stage_rows_batch(mapped_view(table, rows), ids, width, pinned.data(),
                                     pinned.size(), staged_tensor, nullptr);
    ops::ple_iq4_nl_decode_rows(staged_tensor, output_tensor, nullptr);
    cuda_synchronize();

    std::vector<std::uint8_t> expected_staged(staged_bytes);
    std::vector<std::uint16_t> expected(slots * ops::kPleRowWidth);
    for (std::size_t slot = 0; slot < slots; ++slot) {
        const auto* row = table.data() + static_cast<std::size_t>(ids[slot]) * ops::kPleIq4NlRowBytes;
        std::memcpy(expected_staged.data() + slot * ops::kPleIq4NlRowBytes, row,
                    ops::kPleIq4NlRowBytes);
        for (int d = 0; d < ops::kPleRowWidth; ++d) {
            expected[d + ops::kPleRowWidth * slot] = f32_to_bf16(static_cast<float>(
                iq4_oracle(row + (d / 32) * ops::kPleIq4NlBlockBytes, d % 32)));
        }
    }
    std::vector<std::uint8_t> actual_pinned(staged_bytes);
    std::memcpy(actual_pinned.data(), pinned.data(), staged_bytes);
    int failures = verify_exact("PLE batched pinned row bytes", actual_pinned, expected_staged);
    failures += verify_exact("PLE batched device row bytes",
                             from_device<std::uint8_t>(staged.data(), staged_bytes),
                             expected_staged);
    failures += verify_exact("PLE batched exact IQ4_NL decode",
                             from_device<std::uint16_t>(output.data(), expected.size()), expected);

    const std::vector<std::uint8_t> before =
        from_device<std::uint8_t>(staged.data(), staged_bytes);
    auto invalid_ids = ids;
    invalid_ids[ops::kPleHeads + 9] = rows;
    bool rejected = false;
    try {
        ops::ple_iq4_nl_stage_rows_batch(mapped_view(table, rows), invalid_ids, width,
                                         pinned.data(), pinned.size(), staged_tensor, nullptr);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    if (!rejected) {
        std::cerr << "FAIL PLE batched invalid row was accepted\n";
        ++failures;
    }
    failures += verify_exact("PLE rejected batch leaves device staging unchanged",
                             from_device<std::uint8_t>(staged.data(), staged_bytes), before);
    rejected = false;
    try {
        ops::ple_iq4_nl_stage_rows_batch(mapped_view(table, rows), ids, width, pinned.data(),
                                         staged_bytes - 1, staged_tensor, nullptr);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    if (!rejected) {
        std::cerr << "FAIL PLE batched short pinned slot was accepted\n";
        ++failures;
    }
    rejected = false;
    try {
        ops::ple_iq4_nl_stage_rows_batch(mapped_view(table, rows),
                                         std::span<const std::int32_t>(ids).first(ids.size() - 1),
                                         width, pinned.data(), pinned.size(), staged_tensor,
                                         nullptr);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    if (!rejected) {
        std::cerr << "FAIL PLE batched malformed row-id extent was accepted\n";
        ++failures;
    }
    failures += staged.verify_guards("PLE batched staged rows");
    failures += output.verify_guards("PLE batched decoded embedding");
    return failures;
}

struct Weights {
    Weights()
        : key_host(static_cast<std::size_t>(ops::kPleChannels) * kQ8RowBytes, 0),
          value_host(static_cast<std::size_t>(ops::kPleEmbeddingWidth) * kQ8RowBytes, 0),
          key_device(key_host.size()), value_device(value_host.size()) {
        scale_bits = f16_bits(1.0F / 127.0F);
        scale = f16_value(scale_bits);
        for (int row = 0; row < ops::kPleChannels; ++row) {
            auto* block = key_host.data() + static_cast<std::size_t>(row) * kQ8RowBytes;
            block[0] = static_cast<std::uint8_t>(scale_bits);
            block[1] = static_cast<std::uint8_t>(scale_bits >> 8U);
            block[2] = static_cast<std::uint8_t>((row & 1) == 0 ? 127 : 129);
        }
        for (int row = 0; row < ops::kPleEmbeddingWidth; ++row) {
            auto* block = value_host.data() + static_cast<std::size_t>(row) * kQ8RowBytes;
            block[0] = static_cast<std::uint8_t>(scale_bits);
            block[1] = static_cast<std::uint8_t>(scale_bits >> 8U);
            block[3] = 64;
        }
        key_device.copy_from_host(key_host.data(), key_host.size());
        value_device.copy_from_host(value_host.data(), value_host.size());
        key = make_weight(key_device, ops::kPleChannels);
        value = make_weight(value_device, ops::kPleEmbeddingWidth);
    }

    static Weight make_weight(DeviceBuffer& storage, int rows) {
        Weight weight{};
        weight.payload = storage.p;
        weight.payload_bytes = storage.bytes;
        weight.qtype = QType::GGML_Q8_0;
        weight.group_size = 32;
        weight.shape[0] = rows;
        weight.shape[1] = ops::kPleEmbeddingWidth;
        weight.padded_shape[0] = rows;
        weight.padded_shape[1] = ops::kPleEmbeddingWidth;
        weight.ndim = 2;
        weight.qdata = storage.p;
        weight.n = rows;
        weight.k = ops::kPleEmbeddingWidth;
        weight.group = 32;
        weight.layout = QuantLayout::GgmlBlockRow;
        return weight;
    }

    std::vector<std::uint8_t> key_host;
    std::vector<std::uint8_t> value_host;
    DeviceBuffer key_device;
    DeviceBuffer value_device;
    std::uint16_t scale_bits{};
    float scale{};
    Weight key;
    Weight value;
};

struct Inputs {
    explicit Inputs(int width) : width(width), residual(static_cast<std::size_t>(ops::kPleChannels) * width),
                                 embedding(static_cast<std::size_t>(ops::kPleEmbeddingWidth) * width),
                                 conv_weight(static_cast<std::size_t>(ops::kPleChannels) * 4),
                                 norm(ops::kPleChannels, 0.75F) {
        for (int token = 0; token < width; ++token) {
            embedding[ops::kPleEmbeddingWidth * token] = 1.0F + token * 0.125F;
            embedding[ops::kPleEmbeddingWidth * token + 1] = 0.5F + token * 0.125F;
            for (int branch = 0; branch < 4; ++branch) {
                const int agreement = (branch & 1) == 0 ? 1312 : 1248;
                for (int d = 0; d < ops::kPleEmbeddingWidth; ++d) {
                    const float key_sign = (d & 1) == 0 ? 1.0F : -1.0F;
                    residual[d + ops::kPleEmbeddingWidth * (branch + 4 * token)] =
                        d < agreement ? key_sign : -key_sign;
                }
            }
        }
        for (int tap = 0; tap < 4; ++tap) {
            for (int channel = 0; channel < ops::kPleChannels; ++channel) {
                conv_weight[static_cast<std::size_t>(channel) * 4 + tap] =
                    0.05F * (tap + 1) + 0.000001F * (channel % 17);
            }
        }
        round_to_bf16(residual);
        round_to_bf16(embedding);
    }

    int width;
    std::vector<float> residual;
    std::vector<float> embedding;
    std::vector<float> conv_weight;
    std::vector<float> norm;
};

struct RunResult {
    std::vector<std::uint16_t> output;
    std::vector<std::uint16_t> state;
    int guards = 0;
};

RunResult run_inject(std::span<const float> residual, std::span<const float> embedding, int width,
                     std::span<const std::uint16_t> old_state, const Inputs& inputs,
                     const Weights& weights, bool in_place_state,
                     bool in_place_residual = false) {
    std::vector<float> residual_vec(residual.begin(), residual.end());
    std::vector<float> embedding_vec(embedding.begin(), embedding.end());
    auto dresidual = to_device_bf16(residual_vec);
    auto dembedding = to_device_bf16(embedding_vec);
    auto dnorm_key = to_device_f32(inputs.norm);
    auto dnorm_query = to_device_f32(inputs.norm);
    auto dnorm_conv = to_device_f32(inputs.norm);
    auto dconv = to_device_f32(inputs.conv_weight);
    auto dold = to_device(std::vector<std::uint16_t>(old_state.begin(), old_state.end()));
    GuardedDeviceBuffer dnew(old_state.size_bytes());
    GuardedDeviceBuffer dout(residual.size_bytes());
    DeviceArena workspace(ops::ple_workspace_capacity_bytes(width));
    Tensor residual_t(dresidual.p, DType::BF16, {ops::kPleEmbeddingWidth, 4, width});
    Tensor embedding_t(dembedding.p, DType::BF16, {ops::kPleEmbeddingWidth, width});
    Tensor key_norm_t(dnorm_key.p, DType::FP32, {ops::kPleChannels});
    Tensor query_norm_t(dnorm_query.p, DType::FP32, {ops::kPleChannels});
    Tensor conv_norm_t(dnorm_conv.p, DType::FP32, {ops::kPleChannels});
    Tensor conv_t(dconv.p, DType::FP32, {4, ops::kPleChannels});
    Tensor old_t(dold.p, DType::BF16, {ops::kPleChannels, 9});
    Tensor new_t(dnew.data(), DType::BF16, {ops::kPleChannels, 9});
    Tensor out_t(dout.data(), DType::BF16, {ops::kPleEmbeddingWidth, 4, width});
    Tensor& state_out = in_place_state ? old_t : new_t;
    Tensor& output = in_place_residual ? residual_t : out_t;
    ops::ple_inject(residual_t, embedding_t, weights.key, weights.value, key_norm_t,
                    query_norm_t, conv_norm_t, conv_t, old_t, state_out, output, workspace, nullptr);
    cuda_synchronize();
    RunResult result;
    result.output = from_device<std::uint16_t>(in_place_residual ? dresidual.p : dout.data(),
                                               residual.size());
    result.state = from_device<std::uint16_t>(in_place_state ? dold.p : dnew.data(), old_state.size());
    if (!in_place_residual) { result.guards += dout.verify_guards("PLE injection output"); }
    if (!in_place_state) { result.guards += dnew.verify_guards("PLE new convolution state"); }
    return result;
}

struct OracleResult {
    std::vector<double> output;
    std::vector<std::uint16_t> state;
};

OracleResult oracle(const Inputs& input, const Weights& weights,
                    std::span<const std::uint16_t> initial_state) {
    std::vector<double> state(initial_state.size());
    std::transform(initial_state.begin(), initial_state.end(), state.begin(),
                   [](std::uint16_t bits) { return static_cast<double>(bf16_to_f32(bits)); });
    std::vector<double> current(static_cast<std::size_t>(ops::kPleChannels) * input.width);
    std::vector<double> output(static_cast<std::size_t>(ops::kPleChannels) * input.width);
    const double key_unit = static_cast<double>(weights.scale) * 127.0;
    const double value_unit = weights.scale * 64.0;
    for (int token = 0; token < input.width; ++token) {
        const double e0 = input.embedding[ops::kPleEmbeddingWidth * token];
        const double e1 = input.embedding[ops::kPleEmbeddingWidth * token + 1];
        const double key_abs = bf16_to_f32(f32_to_bf16(static_cast<float>(e0 * key_unit)));
        const double value = bf16_to_f32(f32_to_bf16(static_cast<float>(e1 * value_unit)));
        std::array<double, 4> gate{};
        std::array<double, 4> normalized_g{};
        for (int branch = 0; branch < 4; ++branch) {
            const double q_inv = 1.0 / std::sqrt(1.0 + 1.0e-6);
            const double k_inv = 1.0 / std::sqrt(key_abs * key_abs + 1.0e-6);
            double dot = 0.0;
            for (int d = 0; d < ops::kPleEmbeddingWidth; ++d) {
                const int channel = d + ops::kPleEmbeddingWidth * branch;
                const double gamma = input.norm[channel];
                const double key = ((d & 1) == 0 ? key_abs : -key_abs) * k_inv * gamma;
                dot += input.residual[channel + ops::kPleChannels * token] * q_inv * gamma * key;
            }
            const double raw = dot / std::sqrt(2560.0);
            const double transformed = raw == 0.0 ? 0.0 : std::copysign(std::sqrt(std::max(std::abs(raw), 1.0e-6)), raw);
            gate[branch] = 1.0 / (1.0 + std::exp(-transformed));
            const double g = gate[branch] * value;
            normalized_g[branch] = bf16_to_f32(f32_to_bf16(static_cast<float>(
                g / std::sqrt(g * g + 1.0e-6) * input.norm[branch * ops::kPleEmbeddingWidth])));
            for (int d = 0; d < ops::kPleEmbeddingWidth; ++d) {
                const int channel = d + ops::kPleEmbeddingWidth * branch;
                current[channel + ops::kPleChannels * token] = normalized_g[branch];
            }
        }
        for (int branch = 0; branch < 4; ++branch) {
            for (int d = 0; d < ops::kPleEmbeddingWidth; ++d) {
                const int channel = d + ops::kPleEmbeddingWidth * branch;
                double conv = 0.0;
                for (int tap = 0; tap < 4; ++tap) {
                    const int logical = token - 9 + 3 * tap;
                    const double n = logical < 0
                                         ? state[channel + ops::kPleChannels * (logical + 9)]
                                         : current[channel + ops::kPleChannels * logical];
                    conv += static_cast<double>(
                                input.conv_weight[static_cast<std::size_t>(channel) * 4 + tap]) * n;
                }
                const double silu = conv / (1.0 + std::exp(-conv));
                output[channel + ops::kPleChannels * token] =
                    static_cast<double>(input.residual[channel + ops::kPleChannels * token]) +
                    gate[branch] * value + silu;
            }
        }
    }

    std::vector<std::uint16_t> final_state(initial_state.size());
    for (int history = 0; history < 9; ++history) {
        const int logical = input.width - 9 + history;
        for (int channel = 0; channel < ops::kPleChannels; ++channel) {
            const std::size_t destination =
                static_cast<std::size_t>(history) * ops::kPleChannels + channel;
            if (logical < 0) {
                final_state[destination] =
                    initial_state[static_cast<std::size_t>(logical + 9) * ops::kPleChannels +
                                  channel];
            } else {
                final_state[destination] = f32_to_bf16(static_cast<float>(
                    current[static_cast<std::size_t>(logical) * ops::kPleChannels + channel]));
            }
        }
    }
    return {std::move(output), std::move(final_state)};
}

int injection_state_case() {
    constexpr int width = 4;
    Inputs input(width);
    Weights weights;
    std::vector<std::uint16_t> initial_state(static_cast<std::size_t>(ops::kPleChannels) * 9);
    for (int history = 0; history < 9; ++history) {
        for (int channel = 0; channel < ops::kPleChannels; ++channel) {
            const float magnitude =
                0.01F + 0.002F * history + 0.0001F * static_cast<float>(channel % 19);
            const float value = ((channel + history) & 1) == 0 ? magnitude : -magnitude;
            initial_state[static_cast<std::size_t>(history) * ops::kPleChannels + channel] =
                f32_to_bf16(value);
        }
    }
    const OracleResult expected = oracle(input, weights, initial_state);
    const RunResult one = run_inject(input.residual, input.embedding, width, initial_state, input,
                                     weights, false);
    int failures = one.guards;
    const auto actual = [&] {
        std::vector<double> values(one.output.size());
        for (std::size_t i = 0; i < values.size(); ++i) { values[i] = bf16_to_f32(one.output[i]); }
        return values;
    }();
    failures += verify_pointwise("PLE injection FP64 formula", actual, expected.output,
                                 PointwiseCriterion{0.02, 0.01});
    failures += verify_exact("PLE injection independent final BF16 state", one.state,
                             expected.state);
    const RunResult in_place = run_inject(input.residual, input.embedding, width, initial_state, input,
                                          weights, false, true);
    failures += verify_exact("PLE distinct vs in-place residual output", one.output,
                             in_place.output);
    failures += verify_exact("PLE distinct vs in-place residual state", one.state, in_place.state);
    failures += in_place.guards;

    const std::size_t channels = ops::kPleChannels;
    const RunResult first = run_inject(std::span(input.residual).first(2 * channels),
                                       std::span(input.embedding).first(2 * ops::kPleEmbeddingWidth),
                                       2, initial_state, input, weights, true);
    const RunResult second = run_inject(std::span(input.residual).subspan(2 * channels),
                                        std::span(input.embedding).subspan(2 * ops::kPleEmbeddingWidth),
                                        2, first.state, input, weights, false);
    std::vector<std::uint16_t> chunked = first.output;
    chunked.insert(chunked.end(), second.output.begin(), second.output.end());
    failures += verify_exact("PLE one-shot vs chunked output", one.output, chunked);
    failures += verify_exact("PLE one-shot vs chunked state", one.state, second.state);
    failures += first.guards + second.guards;

    std::vector<std::uint16_t> repeated_output;
    std::vector<std::uint16_t> repeated_state = initial_state;
    for (int token = 0; token < width; ++token) {
        const RunResult step = run_inject(
            std::span(input.residual).subspan(static_cast<std::size_t>(token) * channels, channels),
            std::span(input.embedding).subspan(static_cast<std::size_t>(token) * ops::kPleEmbeddingWidth,
                                                ops::kPleEmbeddingWidth),
            1, repeated_state, input, weights, true);
        repeated_output.insert(repeated_output.end(), step.output.begin(), step.output.end());
        repeated_state = step.state;
        failures += step.guards;
    }
    failures += verify_exact("PLE one-shot vs repeated T1 output", one.output, repeated_output);
    failures += verify_exact("PLE one-shot vs repeated T1 state", one.state, repeated_state);
    return failures;
}

} // namespace

int main() {
    if (require_cuda() != 0) { return 1; }
    static_assert(ops::kPleMaxStagedBytes == 5'898'240);
    int failures = 0;
    failures += conv_source_layout_witness();
    failures += staging_decode_case();
    for (const int width : {3, 16, 17, 128, 4096}) {
        failures += batched_staging_decode_case(width);
    }
    failures += injection_state_case();
    if (failures != 0) {
        std::cerr << "PLE tests failed: " << failures << '\n';
        return 1;
    }
    std::cout << "PLE tests passed\n";
    return 0;
}
