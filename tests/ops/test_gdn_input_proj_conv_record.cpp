#include "ninfer/ops/gdn_input_proj.h"
#include "core/device.h"

#include "ops/input_projection_test_common.h"
#include "ops/fp8_activation_ref.h"
#include "ninfer/ops/dflash2_path_select.h"
#include "ops/linear/fp8/fp8_a8_plan.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::input_projection;

namespace {

constexpr std::int32_t kQueryRows = 2048;
constexpr std::int32_t kKeyRows   = 2048;
constexpr ReductionCriterion kNvfp4RecordA16Tolerance{3.15e-3, 4.0e-3, 3.2e-3};
// BF16 final storage permits up to 1/256 relative rounding. In the outlier fixture,
// FP64 SiLU=16.9366 rounds correctly to BF16=16.875 (0.00364 relative error);
// a carried-history output 74.2587 rounds to 74.5 (0.00325 relative error).
// Keep the original A16 criterion unchanged; the A8 codec residual has this explicit
// storage bound plus the same absolute allowance for convolution-history rounding.
constexpr ReductionCriterion kNvfp4RecordA8ResidualTolerance{1.0 / 256.0, 4.0e-3, 1.0 / 256.0};
using OracleGroups                      = std::vector<std::vector<double>>;
double maximum_a8_canonical_relative_l2 = 0;
double maximum_a8_codec_relative_l2     = 0;
double maximum_a8_residual_relative_l2  = 0;

double silu_fp64(double value) {
    if (value >= 0.0) { return value / (1.0 + std::exp(-value)); }
    const double exponential = std::exp(value);
    return value * exponential / (1.0 + exponential);
}

std::vector<std::uint16_t> make_bf16_bits(std::size_t elements, std::uint32_t seed, float low,
                                          float high) {
    std::vector<float> values(elements);
    fill_uniform(values, seed, low, high);
    round_to_bf16(values);
    return bf16_bits(values);
}

int verify_equal(std::string_view label, const std::vector<std::uint16_t>& lhs,
                 const std::vector<std::uint16_t>& rhs) {
    if (lhs.size() == rhs.size() && std::equal(lhs.begin(), lhs.end(), rhs.begin())) { return 0; }
    std::cerr << label << ": BF16 bits differ\n";
    return 1;
}

int verify_zero_tail(std::string_view label, const std::vector<std::uint16_t>& values,
                     std::int32_t rows, std::int32_t width, std::int32_t batch,
                     const std::vector<std::int32_t>& valid_columns) {
    for (std::int32_t batch_row = 0; batch_row < batch; ++batch_row) {
        for (std::int32_t token = valid_columns[static_cast<std::size_t>(batch_row)]; token < width;
             ++token) {
            const std::size_t base = static_cast<std::size_t>(batch_row * width + token) * rows;
            for (std::int32_t row = 0; row < rows; ++row) {
                if (values[base + row] != 0) {
                    std::cerr << label << ": invalid tail is not exact zero\n";
                    return 1;
                }
            }
        }
    }
    return 0;
}

int verify_valid_record_equal(std::string_view label, const std::vector<std::uint16_t>& candidate,
                              const std::vector<std::uint16_t>& reference, std::int32_t channels,
                              std::int32_t width, std::int32_t batch,
                              const std::vector<std::int32_t>& valid_columns) {
    for (std::int32_t batch_row = 0; batch_row < batch; ++batch_row) {
        for (std::int32_t token = 0; token < valid_columns[static_cast<std::size_t>(batch_row)];
             ++token) {
            const std::size_t base = static_cast<std::size_t>(batch_row * width + token) * channels;
            if (!std::equal(candidate.begin() + static_cast<std::ptrdiff_t>(base),
                            candidate.begin() + static_cast<std::ptrdiff_t>(base + channels),
                            reference.begin() + static_cast<std::ptrdiff_t>(base))) {
                std::cerr << label << ": valid conv_record differs at batch=" << batch_row
                          << " token=" << token << '\n';
                return 1;
            }
        }
    }
    return 0;
}

int verify_record_oracle(
    std::string_view label, const quantized_weight::PackedWeight& parent,
    const std::vector<float>& activation, const std::vector<std::uint16_t>& conv_weight,
    const std::vector<std::uint16_t>& state, const std::vector<std::int32_t>& initial_slots,
    const std::vector<std::int32_t>& valid_columns, const std::vector<std::int32_t>& parent_indices,
    const GuardedBf16Tensor& query, const GuardedBf16Tensor& key, const GuardedBf16Tensor& value,
    const GuardedBf16Tensor& z, const GuardedBf16Tensor& record, std::int32_t hidden,
    std::int32_t value_rows, std::int32_t width, std::int32_t batch,
    ReductionCriterion criterion = kNvfp4RecordA16Tolerance, OracleGroups* oracle_outputs = nullptr,
    const OracleGroups* quantized_oracle = nullptr) {
    const std::int32_t channels             = kQueryRows + kKeyRows + value_rows;
    const std::int32_t z_rows               = parent.weight.n - channels;
    const std::vector<double> query_values  = query.values();
    const std::vector<double> key_values    = key.values();
    const std::vector<double> value_values  = value.values();
    const std::vector<double> z_values      = z.values();
    const std::vector<double> record_values = record.values();
    const std::size_t state_slot_stride     = static_cast<std::size_t>(channels) * 3;

    int failures             = 0;
    std::size_t oracle_group = 0;
    const auto check         = [&](const std::string& name, const std::vector<double>& actual,
                                   const std::vector<double>& expected) {
        if (oracle_outputs) { oracle_outputs->push_back(expected); }
        auto allowed = criterion;
        if (quantized_oracle) {
            const auto& quantized = quantized_oracle->at(oracle_group);
            const auto distortion =
                compute_reduction_stats(quantized.data(), expected.data(), expected.size());
            double sum_squared = 0, maximum = 0;
            for (double v : quantized) {
                sum_squared += v * v;
                maximum = std::max(maximum, std::abs(v));
            }
            // Triangle-inequality bound from the independently evaluated codec distortion
            // and unchanged arithmetic/storage allowance. Unlike a fixed percentage, this
            // remains meaningful for cancellation-heavy inputs and nonlinear convolution.
            allowed.relative_l2 =
                distortion.relative_l2 + criterion.relative_l2 *
                                             std::sqrt(sum_squared / quantized.size()) /
                                             std::max(distortion.reference_root_mean_square, 1e-30);
            allowed.gross_absolute = distortion.maximum_absolute_error + criterion.gross_absolute +
                                     criterion.gross_relative_to_max_reference * maximum;
            allowed.gross_relative_to_max_reference = 0;
            maximum_a8_codec_relative_l2 =
                std::max(maximum_a8_codec_relative_l2, distortion.relative_l2);
            maximum_a8_canonical_relative_l2 =
                std::max(maximum_a8_canonical_relative_l2,
                         compute_reduction_stats(actual.data(), expected.data(), expected.size())
                             .relative_l2);
        } else if (oracle_outputs) {
            maximum_a8_residual_relative_l2 =
                std::max(maximum_a8_residual_relative_l2,
                         compute_reduction_stats(actual.data(), expected.data(), expected.size())
                             .relative_l2);
        }
        ++oracle_group;
        return compare(name, actual, expected, allowed);
    };
    const auto verify_group = [&](std::string_view group_label, std::int32_t global_offset,
                                  std::int32_t rows, const std::vector<double>& output,
                                  bool convolved) {
        std::vector<double> actual;
        std::vector<double> expected;
        std::vector<double> record_actual;
        std::vector<double> record_expected;
        for (const std::int32_t local_row : sampled_rows(rows, 7)) {
            const std::int32_t global_row = global_offset + local_row;
            for (std::int32_t batch_row = 0; batch_row < batch; ++batch_row) {
                if (!convolved) {
                    for (std::int32_t token = 0; token < width; ++token) {
                        const std::int32_t flat_column = batch_row * width + token;
                        actual.push_back(
                            output[static_cast<std::size_t>(flat_column) * rows + local_row]);
                        expected.push_back(quantized_weight::dot_fp64(
                            parent, global_row,
                            activation.data() + static_cast<std::size_t>(flat_column) * hidden,
                            hidden));
                    }
                    continue;
                }
                const std::size_t initial_base =
                    static_cast<std::size_t>(initial_slots[static_cast<std::size_t>(batch_row)]) *
                    state_slot_stride;
                const std::array<double, 3> initial{
                    bf16_to_f32(state[initial_base + global_row]),
                    bf16_to_f32(state[initial_base + channels + global_row]),
                    bf16_to_f32(state[initial_base + 2ULL * channels + global_row]),
                };
                std::array<std::array<double, 3>, 16> saved{};
                std::array<double, 3> sequential = initial;
                for (std::int32_t token = 0; token < width; ++token) {
                    const std::int32_t flat_column = batch_row * width + token;
                    const double projected         = quantized_weight::dot_fp64(
                        parent, global_row,
                        activation.data() + static_cast<std::size_t>(flat_column) * hidden, hidden);
                    if (token >= valid_columns[static_cast<std::size_t>(batch_row)]) { continue; }
                    std::array<double, 3> history = sequential;
                    if (!parent_indices.empty()) {
                        const std::int32_t parent_column =
                            parent_indices[static_cast<std::size_t>(flat_column)];
                        history = parent_column < 0
                                      ? initial
                                      : saved[static_cast<std::size_t>(parent_column)];
                    }
                    const double conv =
                        bf16_to_f32(conv_weight[global_row]) * history[0] +
                        bf16_to_f32(conv_weight[channels + global_row]) * history[1] +
                        bf16_to_f32(conv_weight[2ULL * channels + global_row]) * history[2] +
                        bf16_to_f32(conv_weight[3ULL * channels + global_row]) * projected;
                    actual.push_back(
                        output[static_cast<std::size_t>(flat_column) * rows + local_row]);
                    expected.push_back(silu_fp64(conv));
                    record_actual.push_back(
                        record_values[static_cast<std::size_t>(flat_column) * channels +
                                      global_row]);
                    record_expected.push_back(projected);
                    const double saved_projection = round_persistent_bf16(projected);
                    const std::array<double, 3> next{history[1], history[2], saved_projection};
                    saved[static_cast<std::size_t>(token)] = next;
                    if (parent_indices.empty()) { sequential = next; }
                }
            }
        }
        failures +=
            check(std::string(label) + " FP64 " + std::string(group_label), actual, expected);
        if (convolved) {
            failures += check(std::string(label) + " FP64 record " + std::string(group_label),
                              record_actual, record_expected);
        }
    };
    verify_group("query", 0, kQueryRows, query_values, true);
    verify_group("key", kQueryRows, kKeyRows, key_values, true);
    verify_group("value", kQueryRows + kKeyRows, value_rows, value_values, true);
    verify_group("z", channels, z_rows, z_values, false);
    return failures;
}

// A tree whose parents form a chain reproduces the chain record bit for bit at the widest chain
// width, which both verify forms register.
int run_nvfp4_tree_chain_matches_sequential_fused() {
    constexpr auto policy             = ops::LinearPolicy::AllowA8;
    constexpr std::int32_t kHidden    = 5120;
    constexpr std::int32_t kValueRows = 6144;
    constexpr std::int32_t kZRows     = 6144;
    constexpr std::int32_t kChannels  = kQueryRows + kKeyRows + kValueRows;
    constexpr std::int32_t kRows      = 16384;
    constexpr std::int32_t kWidth     = ops::kDflash2MaxChainWidth;
    constexpr std::int32_t kSlots     = kWidth + 1;
    constexpr std::int32_t kInitial   = kWidth;

    quantized_weight::PatternedWeightOptions options;
    options.weight_scale_divisor = 0.125F;
    options.input_scale_divisor  = 3.5F;
    DevicePackedWeight parent(
        quantized_weight::make_patterned_weight(QType::NVFP4, kRows, kHidden, 1901U, options));

    const std::vector<float> packed_x = make_bf16_activation(kHidden, kWidth, 1911U);
    const std::vector<std::uint16_t> conv_weight_bits =
        make_bf16_bits(static_cast<std::size_t>(kChannels) * 4, 1913U, -0.02F, 0.02F);
    const std::vector<std::uint16_t> state_before =
        make_bf16_bits(static_cast<std::size_t>(kChannels) * 3 * kSlots, 1915U, -0.05F, 0.05F);
    const std::vector<std::int32_t> initial_slots{kInitial};
    std::vector<std::int32_t> chain_parent(static_cast<std::size_t>(kWidth), -1);
    for (std::int32_t token = 1; token < kWidth; ++token) {
        chain_parent[static_cast<std::size_t>(token)] = token - 1;
    }

    DeviceBuffer device_packed    = to_device_bf16(packed_x);
    DeviceBuffer device_conv      = to_device(conv_weight_bits);
    DeviceBuffer sequential_state = to_device(state_before);
    DeviceBuffer tree_state       = to_device(state_before);
    DeviceBuffer device_initial   = to_device(initial_slots);
    DeviceBuffer device_parent    = to_device(chain_parent);

    GuardedBf16Tensor seq_q(kQueryRows, kWidth);
    GuardedBf16Tensor seq_k(kKeyRows, kWidth);
    GuardedBf16Tensor seq_v(kValueRows, kWidth);
    GuardedBf16Tensor seq_z(kZRows, kWidth);
    GuardedBf16Tensor seq_record(kChannels, kWidth);
    GuardedBf16Tensor tree_q(kQueryRows, kWidth);
    GuardedBf16Tensor tree_k(kKeyRows, kWidth);
    GuardedBf16Tensor tree_v(kValueRows, kWidth);
    GuardedBf16Tensor tree_z(kZRows, kWidth);
    GuardedBf16Tensor tree_record(kChannels, kWidth);

    Tensor xw(device_packed.p, DType::BF16, {kHidden, kWidth, 1});
    Tensor conv(device_conv.p, DType::BF16, {kChannels, 4});
    Tensor seq_state(sequential_state.p, DType::BF16, {kChannels, 3, kSlots});
    Tensor rec_state(tree_state.p, DType::BF16, {kChannels, 3, kSlots});
    Tensor initial(device_initial.p, DType::I32, {1});
    Tensor parent_index(device_parent.p, DType::I32, {kWidth, 1});
    Tensor sq(seq_q.data(), DType::BF16, {kQueryRows, kWidth, 1});
    Tensor sk(seq_k.data(), DType::BF16, {kKeyRows, kWidth, 1});
    Tensor sv(seq_v.data(), DType::BF16, {kValueRows, kWidth, 1});
    Tensor sz(seq_z.data(), DType::BF16, {kZRows, kWidth, 1});
    Tensor sr(seq_record.data(), DType::BF16, {kChannels, kWidth, 1});
    Tensor tq(tree_q.data(), DType::BF16, {kQueryRows, kWidth, 1});
    Tensor tk(tree_k.data(), DType::BF16, {kKeyRows, kWidth, 1});
    Tensor tv(tree_v.data(), DType::BF16, {kValueRows, kWidth, 1});
    Tensor tz(tree_z.data(), DType::BF16, {kZRows, kWidth, 1});
    Tensor tr(tree_record.data(), DType::BF16, {kChannels, kWidth, 1});

    const std::size_t rec_bytes = ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
        QType::NVFP4, kRows, kHidden, policy, 1, kWidth, kWidth);
    WorkspaceArena seq_ws(std::max<std::size_t>(256, rec_bytes));
    WorkspaceArena tree_ws(std::max<std::size_t>(256, rec_bytes));
    ops::gdn_input_proj_conv_record(xw, parent.view(), conv, seq_state, Tensor{}, initial, sr, sq,
                                    sk, sv, sz, policy, seq_ws, nullptr);
    ops::gdn_input_proj_conv_record(xw, parent.view(), conv, rec_state, Tensor{}, initial, tr, tq,
                                    tk, tv, tz, policy, tree_ws, nullptr, &parent_index);
    cuda_synchronize();

    int failures = 0;
    failures +=
        verify_equal("NVFP4 fused chain-parent vs sequential query", tree_q.bits(), seq_q.bits());
    failures +=
        verify_equal("NVFP4 fused chain-parent vs sequential key", tree_k.bits(), seq_k.bits());
    failures +=
        verify_equal("NVFP4 fused chain-parent vs sequential value", tree_v.bits(), seq_v.bits());
    failures +=
        verify_equal("NVFP4 fused chain-parent vs sequential z", tree_z.bits(), seq_z.bits());
    failures += verify_equal("NVFP4 fused chain-parent vs sequential conv_record",
                             tree_record.bits(), seq_record.bits());
    failures += parent.verify_preserved("NVFP4 fused chain-parent parent weight");
    return failures;
}

int run_batched_record_qualification(QType qtype, ops::LinearPolicy policy) {
    // Batching must preserve the selected request-local arithmetic profile. Same-profile
    // B=1 launches check isolation; the independent FP64 oracle checks the complete formula.
    constexpr std::int32_t kHidden    = 5120;
    constexpr std::int32_t kValueRows = 6144;
    constexpr std::int32_t kZRows     = 6144;
    constexpr std::int32_t kChannels  = kQueryRows + kKeyRows + kValueRows;
    constexpr std::int32_t kRows      = 16384;

    quantized_weight::PatternedWeightOptions options;
    if (qtype == QType::NVFP4) {
        options.weight_scale_divisor = 0.125F;
        options.input_scale_divisor  = 3.5F;
    }
    DevicePackedWeight parent(
        quantized_weight::make_patterned_weight(qtype, kRows, kHidden, 2001U, options));

    std::vector<float> real_activation;
    if (const char* path = std::getenv("NINFER_GDN_REAL_ACTIVATIONS")) {
        std::ifstream input(path);
        if (!input) { throw std::runtime_error("cannot open real GDN activation fixture"); }
        float v;
        while (input >> v) { real_activation.push_back(v); }
        if (!input.eof() || real_activation.empty() || real_activation.size() % kHidden != 0) {
            throw std::runtime_error(
                "real GDN activation fixture must contain complete K=5120 columns");
        }
    }

    const auto run_shape = [&](std::int32_t width, std::int32_t batch,
                               std::vector<std::int32_t> valid_columns,
                               std::vector<std::int32_t> parent_indices, std::uint32_t seed,
                               std::vector<std::uint16_t>* carried_state   = nullptr,
                               std::vector<std::uint16_t>* reference_state = nullptr) {
        const std::int32_t aggregate = width * batch;
        // NVFP4 records under AllowA8 only; FP8 keeps A16 arithmetic under either policy.
        const bool a8            = qtype == QType::NVFP4;
        const bool quantized     = a8;
        const std::int32_t slots = aggregate + batch + 1;
        const bool dense         = valid_columns.empty();
        if (dense) { valid_columns.assign(static_cast<std::size_t>(batch), width); }
        if (!parent_indices.empty() &&
            parent_indices.size() != static_cast<std::size_t>(aggregate)) {
            throw std::invalid_argument("NVFP4 batched record parent fixture has wrong size");
        }

        std::vector<float> activation = make_bf16_activation(kHidden, aggregate, seed);
        if (quantized && !real_activation.empty()) {
            for (std::size_t i = 0; i < activation.size(); ++i) {
                activation[i] = real_activation[i % real_activation.size()];
            }
        } else if (quantized) {
            // Zero/underflow groups, signs, ties-to-even, scale saturation, and an outlier.
            constexpr std::array<float, 16> ties{6,     -6,     .25F,  -.25F,  .75F, -.75F,
                                                 1.25F, -1.25F, 1.75F, -1.75F, 2.5F, -2.5F,
                                                 3.5F,  -3.5F,  5,     -5};
            std::fill_n(activation.begin(), 16, 0.0F);
            std::fill_n(activation.begin() + 16, 16, 0.0001F);
            std::copy(ties.begin(), ties.end(), activation.begin() + 32);
            activation[48] = 1024.0F;
            activation[49] = -1024.0F;
            round_to_bf16(activation);
            if (a8) {
                // Exercise the row-scaled codec's all-zero row as well as the
                // signed outliers/ties in a separate represented input row.
                std::copy_n(activation.begin(), kHidden, activation.begin() + kHidden);
                std::fill_n(activation.begin(), kHidden, 0.0F);
            }
        }
        const std::vector<std::uint16_t> activation_bits = bf16_bits(activation);
        const std::vector<std::uint16_t> conv_weight_bits =
            make_bf16_bits(static_cast<std::size_t>(kChannels) * 4, seed + 1, -0.02F, 0.02F);
        const std::vector<std::uint16_t> state_before =
            carried_state && !carried_state->empty()
                ? *carried_state
                : make_bf16_bits(static_cast<std::size_t>(kChannels) * 3 * slots, seed + 2, -0.05F,
                                 0.05F);
        if (reference_state && reference_state->empty()) { *reference_state = state_before; }

        std::vector<std::int32_t> initial_slots(static_cast<std::size_t>(batch));
        for (std::int32_t batch_row = 0; batch_row < batch; ++batch_row) {
            initial_slots[static_cast<std::size_t>(batch_row)] = aggregate + batch_row;
        }

        DeviceBuffer device_x           = to_device_bf16(activation);
        DeviceBuffer device_conv_weight = to_device(conv_weight_bits);
        DeviceBuffer batched_state      = to_device(state_before);
        DeviceBuffer serial_state       = to_device(state_before);
        DeviceBuffer device_valid;
        if (!dense) { device_valid = to_device(valid_columns); }
        DeviceBuffer device_initial = to_device(initial_slots);
        DeviceBuffer device_parent;
        if (!parent_indices.empty()) { device_parent = to_device(parent_indices); }

        GuardedBf16Tensor batched_q(kQueryRows, aggregate);
        GuardedBf16Tensor batched_k(kKeyRows, aggregate);
        GuardedBf16Tensor batched_v(kValueRows, aggregate);
        GuardedBf16Tensor batched_z(kZRows, aggregate);
        GuardedBf16Tensor batched_record(kChannels, aggregate);
        GuardedBf16Tensor serial_q(kQueryRows, aggregate);
        GuardedBf16Tensor serial_k(kKeyRows, aggregate);
        GuardedBf16Tensor serial_v(kValueRows, aggregate);
        GuardedBf16Tensor serial_z(kZRows, aggregate);
        GuardedBf16Tensor serial_record(kChannels, aggregate);

        Tensor x(device_x.p, DType::BF16, {kHidden, width, batch});
        Tensor conv(device_conv_weight.p, DType::BF16, {kChannels, 4});
        Tensor batched_state_view(batched_state.p, DType::BF16, {kChannels, 3, slots});
        Tensor serial_state_view(serial_state.p, DType::BF16, {kChannels, 3, slots});
        Tensor valid;
        if (!dense) { valid = Tensor(device_valid.p, DType::I32, {batch}); }
        Tensor initial(device_initial.p, DType::I32, {batch});
        Tensor parent_index;
        if (!parent_indices.empty()) {
            parent_index = Tensor(device_parent.p, DType::I32, {width, batch});
        }
        Tensor bq(batched_q.data(), DType::BF16, {kQueryRows, width, batch});
        Tensor bk(batched_k.data(), DType::BF16, {kKeyRows, width, batch});
        Tensor bv(batched_v.data(), DType::BF16, {kValueRows, width, batch});
        Tensor bz(batched_z.data(), DType::BF16, {kZRows, width, batch});
        Tensor br(batched_record.data(), DType::BF16, {kChannels, width, batch});
        Tensor sq(serial_q.data(), DType::BF16, {kQueryRows, width, batch});
        Tensor sk(serial_k.data(), DType::BF16, {kKeyRows, width, batch});
        Tensor sv(serial_v.data(), DType::BF16, {kValueRows, width, batch});
        Tensor sz(serial_z.data(), DType::BF16, {kZRows, width, batch});
        Tensor sr(serial_record.data(), DType::BF16, {kChannels, width, batch});

        // Exercise the adaptive interval: every A8 width is fused, so [2,6] is sized by W6.
        const bool interval = qtype == QType::NVFP4 && policy == ops::LinearPolicy::AllowA8 &&
                              batch == 4 && width == 6;
        const std::size_t rec_bytes = ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
            qtype, kRows, kHidden, policy, batch, interval ? 2 : width, width);
        WorkspaceArena batched_ws(std::max<std::size_t>(256, rec_bytes));
        ops::gdn_input_proj_conv_record(x, parent.view(), conv, batched_state_view, valid, initial,
                                        br, bq, bk, bv, bz, policy, batched_ws, nullptr,
                                        parent_indices.empty() ? nullptr : &parent_index);
        const std::size_t serial_bytes = ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
            qtype, kRows, kHidden, policy, 1, width, width);
        WorkspaceArena serial_ws(std::max<std::size_t>(256, serial_bytes));
        for (std::int32_t batch_row = 0; batch_row < batch; ++batch_row) {
            Tensor valid_b;
            if (!dense) { valid_b = valid.slice(0, batch_row, 1); }
            Tensor parent_b;
            if (!parent_indices.empty()) {
                auto* row_parent = static_cast<std::int32_t*>(device_parent.p) +
                                   static_cast<std::ptrdiff_t>(batch_row) * width;
                parent_b         = Tensor(row_parent, DType::I32, {width, 1});
            }
            Tensor rb = sr.slice(2, batch_row, 1);
            Tensor qb = sq.slice(2, batch_row, 1);
            Tensor kb = sk.slice(2, batch_row, 1);
            Tensor vb = sv.slice(2, batch_row, 1);
            Tensor zb = sz.slice(2, batch_row, 1);
            ops::gdn_input_proj_conv_record(
                x.slice(2, batch_row, 1), parent.view(), conv, serial_state_view, valid_b,
                initial.slice(0, batch_row, 1), rb, qb, kb, vb, zb, policy, serial_ws, nullptr,
                parent_indices.empty() ? nullptr : &parent_b);
        }
        cuda_synchronize();

        const std::string label = std::string(qtype == QType::NVFP4 ? "NVFP4" : "FP8") +
                                  " batched vs serial B=" + std::to_string(batch) +
                                  " policy=" + std::to_string(static_cast<int>(policy)) +
                                  " W=" + std::to_string(width) +
                                  (parent_indices.empty() ? " sequential" : " tree");
        int failures            = 0;
        failures += verify_equal(label + " query", batched_q.bits(), serial_q.bits());
        failures += verify_equal(label + " key", batched_k.bits(), serial_k.bits());
        failures += verify_equal(label + " value", batched_v.bits(), serial_v.bits());
        failures += verify_equal(label + " z", batched_z.bits(), serial_z.bits());
        failures +=
            verify_valid_record_equal(label + " conv_record", batched_record.bits(),
                                      serial_record.bits(), kChannels, width, batch, valid_columns);
        failures += verify_zero_tail(label + " query", batched_q.bits(), kQueryRows, width, batch,
                                     valid_columns);
        failures += verify_zero_tail(label + " key", batched_k.bits(), kKeyRows, width, batch,
                                     valid_columns);
        failures += verify_zero_tail(label + " value", batched_v.bits(), kValueRows, width, batch,
                                     valid_columns);
        const ReductionCriterion criterion =
            qtype == QType::NVFP4 ? kNvfp4RecordA16Tolerance
                                  : ReductionCriterion{1.0 / 256.0, 1.0 / 256.0, 2.0 / 256.0};
        if (!quantized) {
            failures += verify_record_oracle(
                label, parent.host, activation, conv_weight_bits, state_before, initial_slots,
                valid_columns, parent_indices, batched_q, batched_k, batched_v, batched_z,
                batched_record, kHidden, kValueRows, width, batch, criterion);
        }
        if (quantized) {
            OracleGroups quantized_oracle;
            std::vector<float> represented;
            if (a8) {
                const auto encoded = fp8_activation_reference(activation, kHidden);
                represented        = encoded.represented;
                WorkspaceArena codec_workspace(
                    ops::detail::fp8_a8_workspace_capacity_bytes(aggregate, kHidden));
                const auto scratch =
                    ops::detail::allocate_fp8_a8_workspace(codec_workspace, aggregate, kHidden);
                ops::detail::launch_fp8_a8_quantize(x.view({kHidden, aggregate}), parent.view(),
                                                    scratch, nullptr);
                std::vector<std::uint8_t> codes(encoded.codes.size());
                std::vector<float> scales(encoded.scales.size());
                CUDA_CHECK(
                    cudaMemcpy(codes.data(), scratch.codes, codes.size(), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(scales.data(), scratch.scales, scales.size() * sizeof(float),
                                      cudaMemcpyDeviceToHost));
                if (codes != encoded.codes || scales != encoded.scales) {
                    std::cerr << label << ": FP8 activation codec mismatch\n";
                    ++failures;
                }
                if (qtype == QType::NVFP4) {
                    // A producer-published activation must reproduce the policy route exactly.
                    GuardedBf16Tensor a8_q(kQueryRows, aggregate);
                    GuardedBf16Tensor a8_k(kKeyRows, aggregate);
                    GuardedBf16Tensor a8_v(kValueRows, aggregate);
                    GuardedBf16Tensor a8_z(kZRows, aggregate);
                    GuardedBf16Tensor a8_record(kChannels, aggregate);
                    Tensor aq(a8_q.data(), DType::BF16, {kQueryRows, width, batch});
                    Tensor ak(a8_k.data(), DType::BF16, {kKeyRows, width, batch});
                    Tensor av(a8_v.data(), DType::BF16, {kValueRows, width, batch});
                    Tensor az(a8_z.data(), DType::BF16, {kZRows, width, batch});
                    Tensor ar(a8_record.data(), DType::BF16, {kChannels, width, batch});
                    const ops::A8Activation a8_input{
                        Tensor(scratch.codes, DType::FP8_E4M3FN, {kHidden, width, batch}),
                        Tensor(scratch.scales, DType::FP32, {width, batch})};
                    WorkspaceArena a8_ws(std::max<std::size_t>(256, rec_bytes));
                    ops::gdn_input_proj_conv_record(
                        a8_input, parent.view(), conv, batched_state_view, valid, initial, ar, aq,
                        ak, av, az, a8_ws, nullptr,
                        parent_indices.empty() ? nullptr : &parent_index);
                    cuda_synchronize();
                    failures +=
                        verify_equal(label + " A8 input query", a8_q.bits(), batched_q.bits());
                    failures +=
                        verify_equal(label + " A8 input key", a8_k.bits(), batched_k.bits());
                    failures +=
                        verify_equal(label + " A8 input value", a8_v.bits(), batched_v.bits());
                    failures += verify_equal(label + " A8 input z", a8_z.bits(), batched_z.bits());
                    failures += verify_valid_record_equal(label + " A8 input conv_record",
                                                          a8_record.bits(), batched_record.bits(),
                                                          kChannels, width, batch, valid_columns);
                }
            }
            failures += verify_record_oracle(
                label + " codec residual", parent.host, represented, conv_weight_bits, state_before,
                initial_slots, valid_columns, parent_indices, batched_q, batched_k, batched_v,
                batched_z, batched_record, kHidden, kValueRows, width, batch,
                kNvfp4RecordA8ResidualTolerance, &quantized_oracle);
            failures += verify_record_oracle(
                label + " canonical", parent.host, activation, conv_weight_bits, state_before,
                initial_slots, valid_columns, parent_indices, batched_q, batched_k, batched_v,
                batched_z, batched_record, kHidden, kValueRows, width, batch,
                kNvfp4RecordA8ResidualTolerance, nullptr, &quantized_oracle);
            if (reference_state) {
                // This second residual check carries independently evolved BF16 history,
                // rather than resetting to the candidate's represented incoming state.
                failures += verify_record_oracle(
                    label + " carried history", parent.host, represented, conv_weight_bits,
                    *reference_state, initial_slots, valid_columns, parent_indices, batched_q,
                    batched_k, batched_v, batched_z, batched_record, kHidden, kValueRows, width,
                    batch, kNvfp4RecordA8ResidualTolerance);
                *carried_state         = state_before;
                const auto record_bits = batched_record.bits();
                for (int b = 0; b < batch; ++b) {
                    std::vector<int> path;
                    for (int t = valid_columns[b] - 1; t >= 0;
                         t     = parent_indices.empty() ? t - 1 : parent_indices[b * width + t]) {
                        path.push_back(t);
                    }
                    std::reverse(path.begin(), path.end());
                    const std::size_t base =
                        static_cast<std::size_t>(initial_slots[b]) * 3 * kChannels;
                    for (int row = 0; row < kChannels; ++row) {
                        for (int t : path) {
                            (*carried_state)[base + row] = (*carried_state)[base + kChannels + row];
                            (*carried_state)[base + kChannels + row] =
                                (*carried_state)[base + 2 * static_cast<std::size_t>(kChannels) +
                                                 row];
                            (*carried_state)[base + 2 * static_cast<std::size_t>(kChannels) + row] =
                                record_bits[static_cast<std::size_t>(b * width + t) * kChannels +
                                            row];
                        }
                    }
                    // Evolve all rows inspected by the canonical oracle independently from
                    // decoded codes/scales. Other reference rows are never read by that oracle.
                    for (const auto& [offset, rows] : std::array<std::pair<int, int>, 3>{
                             {{0, kQueryRows},
                              {kQueryRows, kKeyRows},
                              {kQueryRows + kKeyRows, kValueRows}}}) {
                        for (int local : sampled_rows(rows, 7)) {
                            const int row = offset + local;
                            for (int t : path) {
                                const double projected = quantized_weight::dot_fp64(
                                    parent.host, row,
                                    represented.data() +
                                        static_cast<std::size_t>(b * width + t) * kHidden,
                                    kHidden);
                                (*reference_state)[base + row] =
                                    (*reference_state)[base + kChannels + row];
                                (*reference_state)[base + kChannels + row] =
                                    (*reference_state)[base +
                                                       2 * static_cast<std::size_t>(kChannels) +
                                                       row];
                                (*reference_state)[base + 2 * static_cast<std::size_t>(kChannels) +
                                                   row] =
                                    f32_to_bf16(
                                        static_cast<float>(round_persistent_bf16(projected)));
                            }
                        }
                    }
                }
            }
        }
        if (qtype == QType::FP8_E4M3FN_ROW_BF16S) {
            failures += verify_record_oracle(
                label + " serial", parent.host, activation, conv_weight_bits, state_before,
                initial_slots, valid_columns, parent_indices, serial_q, serial_k, serial_v,
                serial_z, serial_record, kHidden, kValueRows, width, batch, criterion);
        }
        failures += batched_q.verify_guards(label + " batched query");
        failures += batched_k.verify_guards(label + " batched key");
        failures += batched_v.verify_guards(label + " batched value");
        failures += batched_z.verify_guards(label + " batched z");
        failures += batched_record.verify_guards(label + " batched conv_record");
        failures += serial_q.verify_guards(label + " serial query");
        failures += serial_k.verify_guards(label + " serial key");
        failures += serial_v.verify_guards(label + " serial value");
        failures += serial_z.verify_guards(label + " serial z");
        failures += serial_record.verify_guards(label + " serial conv_record");
        failures += verify_preserved(label + " activation", device_x, activation_bits);
        failures += verify_preserved(label + " conv weight", device_conv_weight, conv_weight_bits);
        failures += verify_preserved(label + " batched state", batched_state, state_before);
        failures += verify_preserved(label + " serial state", serial_state, state_before);
        if (!dense) {
            failures += verify_preserved(label + " valid columns", device_valid, valid_columns);
        }
        failures += verify_preserved(label + " initial slots", device_initial, initial_slots);
        if (!parent_indices.empty()) {
            failures += verify_preserved(label + " parent index", device_parent, parent_indices);
        }
        if (batched_ws.used() != 0 || batched_ws.peak_used() != rec_bytes) {
            std::cerr << label << ": batched workspace query/execution mismatch\n";
            ++failures;
        }
        if (serial_ws.used() != 0 || serial_ws.peak_used() != serial_bytes) {
            std::cerr << label << ": serial workspace query/execution mismatch\n";
            ++failures;
        }
        return failures;
    };

    int failures = 0;
    failures += run_shape(4, 1, {}, {}, 1951U);
    failures += run_shape(4, 1, {3}, {}, 1957U);
    failures += run_shape(4, 1, {}, {-1, 0, 0, 1}, 1961U);
    failures += run_shape(4, 2, {}, {}, 1963U);
    failures += run_shape(4, 3, {4, 3, 1}, {}, 1967U);
    failures +=
        run_shape(4, 4, {4, 3, 2, 1}, {-1, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1}, 1969U);
    failures += run_shape(5, 1, {}, {}, 1973U);
    failures += run_shape(5, 1, {4}, {-1, 0, 0, 1, 2}, 1979U);
    failures += run_shape(6, 1, {}, {}, 1980U);
    failures += run_shape(6, 1, {5}, {-1, 0, 0, 1, 2, 3}, 1981U);
    failures += run_shape(2, 2, {}, {}, 1983U);
    failures += run_shape(2, 3, {}, {}, 1987U);
    failures += run_shape(2, 4, {2, 2, 1, 1}, {}, 1991U);
    failures += run_shape(5, 2, {}, {}, 2011U);
    failures += run_shape(5, 3, {}, {}, 2021U);
    failures += run_shape(5, 4, {}, {}, 2031U);
    failures += run_shape(5, 2, {5, 2}, {}, 2041U);
    failures += run_shape(5, 3, {5, 4, 1}, {}, 2051U);
    failures += run_shape(5, 4, {5, 4, 2, 1}, {}, 2061U);
    failures += run_shape(5, 2, {}, {-1, 0, 0, 1, 1, -1, 0, 1, 1, 3}, 2071U);
    failures += run_shape(5, 3, {5, 3, 2}, {-1, 0, 0, 1, 1, -1, 0, 1, 1, 3, -1, 0, 0, 2, 2}, 2077U);
    failures += run_shape(5, 4, {5, 4, 3, 2},
                          {-1, 0, 0, 1, 1, -1, 0, 1, 1, 3, -1, 0, 0, 2, 2, -1, 0, 1, 2, 3}, 2081U);
    // C=5..8 verify covers every chain width, plus ragged tree rows at the widest aggregates
    // (A8 W=8 B=6 is the 48-token single-tile boundary).
    for (int batch : {5, 6, 7, 8}) {
        for (int width : {2, 3, 4, 5, 6, 7, 8}) {
            failures += run_shape(width, batch, {}, {}, 2700U + width * 8 + batch);
        }
        for (int width : {5, 6, 7, 8}) {
            std::vector<int> valid(batch), parents(static_cast<std::size_t>(batch) * width);
            for (int b = 0; b < batch; ++b) {
                valid[b] = width - b % width;
                for (int t = 0; t < width; ++t) {
                    parents[b * width + t] = t == 0 ? -1 : (t - 1) / 2;
                }
            }
            failures += run_shape(width, batch, valid, parents, 2760U + width * 8 + batch);
        }
    }
    if (qtype == QType::NVFP4) {
        for (int width : {6, 7, 8}) {
            for (int batch : {1, 2, 3, 4, 7, 8}) {
                std::vector<int> valid(batch);
                for (int b = 0; b < batch; ++b) { valid[b] = b + 1 == batch ? width - 1 : width; }
                failures += run_shape(width, batch, valid, {}, 2300U + width * 4 + batch);
            }
        }
        // Depth-first draft trees at the product tree widths; W*B<=48 runs the fused epilogue and
        // W=16 C=4 the staged projection.
        for (int width : {12, 16}) {
            for (int batch : {1, 2, 3, 4}) {
                std::vector<int> valid(batch);
                std::vector<int> tree(static_cast<std::size_t>(width) * batch);
                for (int b = 0; b < batch; ++b) {
                    valid[b] = width - 2 * b;
                    for (int t = 0; t < width; ++t) {
                        // Spine 0..7, then siblings hanging off spine columns b, b+1, ...
                        tree[b * width + t] = t == 0 ? -1 : (t < 8 ? t - 1 : (t - 8 + b) % 7);
                    }
                }
                failures += run_shape(width, batch, valid, tree, 2500U + width * 4 + batch);
            }
        }
        for (int width : {4, 5, 6, 8}) {
            std::vector<std::uint16_t> carried, independent;
            for (int prefix = 1; prefix <= width; ++prefix) {
                std::vector<int> valid(4);
                for (int b = 0; b < 4; ++b) { valid[b] = 1 + (prefix - 1 + b) % width; }
                std::vector<int> parents;
                if ((prefix & 1) == 0) {
                    parents.resize(static_cast<std::size_t>(width) * 4);
                    for (int b = 0; b < 4; ++b) {
                        for (int t = 0; t < width; ++t) {
                            parents[b * width + t] = t == 0 ? -1 : (t - 1) / 2;
                        }
                    }
                }
                failures += run_shape(width, 4, valid, parents, 2400U + width * 16 + prefix,
                                      &carried, &independent);
            }
        }
    }
    if (qtype == QType::FP8_E4M3FN_ROW_BF16S) {
        for (int width : {3, 7, 8, 9, 10, 11, 16}) {
            for (int batch : {1, 2, 3, 4, 7, 8}) {
                std::vector<int> valid(batch);
                for (int b = 0; b < batch; ++b) { valid[b] = b + 1 == batch ? width - 1 : width; }
                std::vector<int> tree(static_cast<std::size_t>(width) * batch);
                for (int b = 0; b < batch; ++b)
                    for (int t = 0; t < width; ++t) tree[b * width + t] = t == 0 ? -1 : (t - 1) / 2;
                failures += run_shape(width, batch, valid, tree, 2200U + width * 4 + batch);
            }
        }
    }
    failures += parent.verify_preserved("batched record parent weight");
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cerr << "FAIL: no usable CUDA device\n";
        return 1;
    }

    int failures = 0;
    failures += run_nvfp4_tree_chain_matches_sequential_fused();
    failures += run_batched_record_qualification(QType::NVFP4, ops::LinearPolicy::AllowA8);
    std::cout << "A8 max group relative-L2: canonical=" << maximum_a8_canonical_relative_l2
              << " codec-distortion=" << maximum_a8_codec_relative_l2
              << " residual=" << maximum_a8_residual_relative_l2 << '\n';
    failures +=
        run_batched_record_qualification(QType::FP8_E4M3FN_ROW_BF16S, ops::LinearPolicy::A16Only);
    failures +=
        run_batched_record_qualification(QType::FP8_E4M3FN_ROW_BF16S, ops::LinearPolicy::AllowA8);
    std::cout << (failures == 0 ? "OK" : "FAIL") << " gdn_input_proj_conv_record\n";
    return failures == 0 ? 0 : 1;
}
