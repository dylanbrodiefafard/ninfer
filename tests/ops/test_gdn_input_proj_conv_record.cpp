#include "ninfer/ops/gdn_input_proj.h"

#include "ops/input_projection_test_common.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::input_projection;

namespace {

constexpr std::int32_t kQueryRows = 2048;
constexpr std::int32_t kKeyRows   = 2048;
constexpr ReductionCriterion kNvfp4RecordA16Tolerance{3.15e-3, 4.0e-3, 3.2e-3};

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

int verify_valid_record_equal(std::string_view label,
                              const std::vector<std::uint16_t>& candidate,
                              const std::vector<std::uint16_t>& reference,
                              std::int32_t channels, std::int32_t width, std::int32_t batch,
                              const std::vector<std::int32_t>& valid_columns) {
    for (std::int32_t batch_row = 0; batch_row < batch; ++batch_row) {
        for (std::int32_t token = 0;
             token < valid_columns[static_cast<std::size_t>(batch_row)]; ++token) {
            const std::size_t base =
                static_cast<std::size_t>(batch_row * width + token) * channels;
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

int verify_nvfp4_record_oracle(
    std::string_view label, const quantized_weight::PackedWeight& parent,
    const std::vector<float>& activation, const std::vector<std::uint16_t>& conv_weight,
    const std::vector<std::uint16_t>& state, const std::vector<std::int32_t>& initial_slots,
    const std::vector<std::int32_t>& valid_columns,
    const std::vector<std::int32_t>& parent_indices, const GuardedBf16Tensor& query,
    const GuardedBf16Tensor& key, const GuardedBf16Tensor& value, const GuardedBf16Tensor& z,
    const GuardedBf16Tensor& record, std::int32_t hidden, std::int32_t value_rows,
    std::int32_t width, std::int32_t batch) {
    const std::int32_t channels = kQueryRows + kKeyRows + value_rows;
    const std::int32_t z_rows   = parent.weight.n - channels;
    const std::vector<double> query_values = query.values();
    const std::vector<double> key_values   = key.values();
    const std::vector<double> value_values = value.values();
    const std::vector<double> z_values     = z.values();
    const std::vector<double> record_values = record.values();
    const std::size_t state_slot_stride = static_cast<std::size_t>(channels) * 3;

    int failures = 0;
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
                            activation.data() +
                                static_cast<std::size_t>(flat_column) * hidden,
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
                    const double projected = quantized_weight::dot_fp64(
                        parent, global_row,
                        activation.data() + static_cast<std::size_t>(flat_column) * hidden,
                        hidden);
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
                    actual.push_back(output[static_cast<std::size_t>(flat_column) * rows +
                                            local_row]);
                    expected.push_back(silu_fp64(conv));
                    record_actual.push_back(
                        record_values[static_cast<std::size_t>(flat_column) * channels +
                                      global_row]);
                    record_expected.push_back(projected);
                    const double saved_projection =
                        bf16_to_f32(f32_to_bf16(static_cast<float>(projected)));
                    const std::array<double, 3> next{history[1], history[2], saved_projection};
                    saved[static_cast<std::size_t>(token)] = next;
                    if (parent_indices.empty()) { sequential = next; }
                }
            }
        }
        failures += compare(std::string(label) + " FP64 " + std::string(group_label), actual,
                            expected, kNvfp4RecordA16Tolerance);
        if (convolved) {
            failures += compare(std::string(label) + " FP64 record " +
                                    std::string(group_label),
                                record_actual, record_expected, kNvfp4RecordA16Tolerance);
        }
    };
    verify_group("query", 0, kQueryRows, query_values, true);
    verify_group("key", kQueryRows, kKeyRows, key_values, true);
    verify_group("value", kQueryRows + kKeyRows, value_rows, value_values, true);
    verify_group("z", channels, z_rows, z_values, false);
    return failures;
}

int verify_conv_record(std::string_view label, const std::vector<std::uint16_t>& snapshot_state,
                       const std::vector<std::uint16_t>& record, std::int32_t channels,
                       std::int32_t width, std::int32_t batch,
                       const std::vector<std::int32_t>& valid_columns,
                       const std::vector<std::int32_t>& snapshot_bases) {
    const std::size_t slot_stride = static_cast<std::size_t>(channels) * 3;
    for (std::int32_t batch_row = 0; batch_row < batch; ++batch_row) {
        for (std::int32_t token = 0; token < valid_columns[static_cast<std::size_t>(batch_row)];
             ++token) {
            const std::size_t snapshot =
                static_cast<std::size_t>(snapshot_bases[static_cast<std::size_t>(batch_row)] +
                                         token) *
                    slot_stride +
                2ULL * channels;
            const std::size_t record_column =
                static_cast<std::size_t>(batch_row * width + token) * channels;
            if (!std::equal(snapshot_state.begin() + static_cast<std::ptrdiff_t>(snapshot),
                            snapshot_state.begin() +
                                static_cast<std::ptrdiff_t>(snapshot + channels),
                            record.begin() + static_cast<std::ptrdiff_t>(record_column))) {
                std::cerr << label << ": conv record differs from snapshot newest column\n";
                return 1;
            }
        }
    }
    return 0;
}

template <class SnapshotLaunch, class RecordLaunch>
int run_case(std::string_view label, std::int32_t hidden, std::int32_t value_rows,
             std::int32_t z_rows, std::int32_t width, std::int32_t batch,
             std::vector<std::int32_t> valid_columns, std::size_t snapshot_workspace_bytes,
             std::size_t record_workspace_bytes, SnapshotLaunch&& snapshot_launch,
             RecordLaunch&& record_launch, std::uint32_t seed) {
    const std::int32_t channels          = kQueryRows + kKeyRows + value_rows;
    const std::int32_t aggregate_columns = width * batch;
    const std::int32_t slots             = aggregate_columns + batch + 1;
    const bool dense                     = valid_columns.empty();
    if (valid_columns.empty()) { valid_columns.assign(static_cast<std::size_t>(batch), width); }

    const std::vector<float> activation = make_bf16_activation(hidden, aggregate_columns, seed);
    const std::vector<std::uint16_t> conv_weight_bits =
        make_bf16_bits(static_cast<std::size_t>(channels) * 4, seed + 1, -0.02F, 0.02F);
    const std::vector<std::uint16_t> state_before =
        make_bf16_bits(static_cast<std::size_t>(channels) * 3 * slots, seed + 2, -0.05F, 0.05F);

    std::vector<std::int32_t> initial_slots(static_cast<std::size_t>(batch));
    std::vector<std::int32_t> snapshot_bases(static_cast<std::size_t>(batch));
    for (std::int32_t batch_row = 0; batch_row < batch; ++batch_row) {
        snapshot_bases[static_cast<std::size_t>(batch_row)] = batch_row * width;
        initial_slots[static_cast<std::size_t>(batch_row)]  = aggregate_columns + batch_row;
    }

    DeviceBuffer device_x           = to_device_bf16(activation);
    DeviceBuffer device_conv_weight = to_device(conv_weight_bits);
    DeviceBuffer snapshot_state     = to_device(state_before);
    DeviceBuffer record_state       = to_device(state_before);
    DeviceBuffer device_valid;
    if (!dense) { device_valid = to_device(valid_columns); }
    DeviceBuffer device_initial  = to_device(initial_slots);
    DeviceBuffer device_snapshot = to_device(snapshot_bases);

    GuardedBf16Tensor snapshot_query(kQueryRows, aggregate_columns);
    GuardedBf16Tensor snapshot_key(kKeyRows, aggregate_columns);
    GuardedBf16Tensor snapshot_value(value_rows, aggregate_columns);
    GuardedBf16Tensor snapshot_z(z_rows, aggregate_columns);
    GuardedBf16Tensor record_query(kQueryRows, aggregate_columns);
    GuardedBf16Tensor record_key(kKeyRows, aggregate_columns);
    GuardedBf16Tensor record_value(value_rows, aggregate_columns);
    GuardedBf16Tensor record_z(z_rows, aggregate_columns);
    GuardedBf16Tensor conv_record(channels, aggregate_columns);

    Tensor x(device_x.p, DType::BF16, {hidden, width, batch});
    Tensor conv_weight(device_conv_weight.p, DType::BF16, {channels, 4});
    Tensor snapshot_state_view(snapshot_state.p, DType::BF16, {channels, 3, slots});
    Tensor record_state_view(record_state.p, DType::BF16, {channels, 3, slots});
    Tensor valid;
    if (!dense) { valid = Tensor(device_valid.p, DType::I32, {batch}); }
    Tensor initial(device_initial.p, DType::I32, {batch});
    Tensor snapshot_base(device_snapshot.p, DType::I32, {batch});
    Tensor snapshot_q(snapshot_query.data(), DType::BF16, {kQueryRows, width, batch});
    Tensor snapshot_k(snapshot_key.data(), DType::BF16, {kKeyRows, width, batch});
    Tensor snapshot_v(snapshot_value.data(), DType::BF16, {value_rows, width, batch});
    Tensor snapshot_z_view(snapshot_z.data(), DType::BF16, {z_rows, width, batch});
    Tensor record_q(record_query.data(), DType::BF16, {kQueryRows, width, batch});
    Tensor record_k(record_key.data(), DType::BF16, {kKeyRows, width, batch});
    Tensor record_v(record_value.data(), DType::BF16, {value_rows, width, batch});
    Tensor record_z_view(record_z.data(), DType::BF16, {z_rows, width, batch});
    Tensor conv_record_view(conv_record.data(), DType::BF16, {channels, width, batch});

    WorkspaceArena snapshot_workspace(std::max<std::size_t>(256, snapshot_workspace_bytes));
    WorkspaceArena record_workspace(std::max<std::size_t>(256, record_workspace_bytes));
    snapshot_launch(x, conv_weight, snapshot_state_view, valid, initial, snapshot_base, snapshot_q,
                    snapshot_k, snapshot_v, snapshot_z_view, snapshot_workspace);
    record_launch(x, conv_weight, record_state_view, valid, initial, conv_record_view, record_q,
                  record_k, record_v, record_z_view, record_workspace);
    cuda_synchronize();

    int failures = 0;
    failures +=
        verify_equal(std::string(label) + " query", snapshot_query.bits(), record_query.bits());
    failures += verify_equal(std::string(label) + " key", snapshot_key.bits(), record_key.bits());
    failures +=
        verify_equal(std::string(label) + " value", snapshot_value.bits(), record_value.bits());
    failures += verify_equal(std::string(label) + " z", snapshot_z.bits(), record_z.bits());
    failures += verify_zero_tail(std::string(label) + " query", record_query.bits(), kQueryRows,
                                 width, batch, valid_columns);
    failures += verify_zero_tail(std::string(label) + " key", record_key.bits(), kKeyRows, width,
                                 batch, valid_columns);
    failures += verify_zero_tail(std::string(label) + " value", record_value.bits(), value_rows,
                                 width, batch, valid_columns);

    const std::vector<std::uint16_t> snapshot_state_after =
        from_device<std::uint16_t>(snapshot_state, state_before.size());
    const std::vector<std::uint16_t> record_state_after =
        from_device<std::uint16_t>(record_state, state_before.size());
    failures += verify_conv_record(label, snapshot_state_after, conv_record.bits(), channels, width,
                                   batch, valid_columns, snapshot_bases);
    failures +=
        verify_equal(std::string(label) + " source state", state_before, record_state_after);

    failures += snapshot_query.verify_guards(std::string(label) + " snapshot query");
    failures += snapshot_key.verify_guards(std::string(label) + " snapshot key");
    failures += snapshot_value.verify_guards(std::string(label) + " snapshot value");
    failures += snapshot_z.verify_guards(std::string(label) + " snapshot z");
    failures += record_query.verify_guards(std::string(label) + " record query");
    failures += record_key.verify_guards(std::string(label) + " record key");
    failures += record_value.verify_guards(std::string(label) + " record value");
    failures += record_z.verify_guards(std::string(label) + " record z");
    failures += conv_record.verify_guards(std::string(label) + " conv record");
    if (snapshot_workspace.used() != 0 ||
        snapshot_workspace.peak_used() != snapshot_workspace_bytes) {
        std::cerr << label << ": snapshot workspace query/execution mismatch\n";
        ++failures;
    }
    if (record_workspace.used() != 0 || record_workspace.peak_used() != record_workspace_bytes) {
        std::cerr << label << ": record workspace query/execution mismatch\n";
        ++failures;
    }
    return failures;
}

int run_q4_q5() {
    constexpr std::int32_t kHidden    = 5120;
    constexpr std::int32_t kValueRows = 6144;
    constexpr std::int32_t kZRows     = 6144;
    DevicePackedWeight qk(
        quantized_weight::make_patterned_weight(QType::Q4G64_F16S, 4096, kHidden, 1401U));
    DevicePackedWeight value_z(
        quantized_weight::make_patterned_weight(QType::Q5G64_F16S, 12288, kHidden, 1403U));

    int failures   = 0;
    const auto run = [&](std::int32_t width, std::int32_t batch, std::vector<std::int32_t> valid,
                         std::uint32_t seed) {
        const std::size_t snapshot_bytes =
            ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                kQueryRows, kKeyRows, kValueRows, batch, width, width);
        const std::size_t record_bytes = ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
            kQueryRows, kKeyRows, kValueRows, batch, width, width);
        return run_case(
            "Q4/Q5 B=" + std::to_string(batch) + " T=" + std::to_string(width), kHidden, kValueRows,
            kZRows, width, batch, std::move(valid), snapshot_bytes, record_bytes,
            [&](const Tensor& x, const Tensor& conv, Tensor& state, const Tensor& valid_columns,
                const Tensor& initial, const Tensor& snapshot_base, Tensor& q, Tensor& k, Tensor& v,
                Tensor& z, WorkspaceArena& workspace) {
                ops::gdn_input_proj_conv_snapshot(x, qk.view(), value_z.view(), conv, state,
                                                  valid_columns, initial, snapshot_base, q, k, v, z,
                                                  workspace, nullptr);
            },
            [&](const Tensor& x, const Tensor& conv, const Tensor& state,
                const Tensor& valid_columns, const Tensor& initial, Tensor& record, Tensor& q,
                Tensor& k, Tensor& v, Tensor& z, WorkspaceArena& workspace) {
                ops::gdn_input_proj_conv_record(x, qk.view(), value_z.view(), conv, state,
                                                valid_columns, initial, record, q, k, v, z,
                                                workspace, nullptr);
            },
            seed);
    };
    failures += run(2, 1, {}, 1411U);
    failures += run(4, 1, {3}, 1421U);
    failures += run(7, 1, {5}, 1431U);
    failures += run(16, 1, {}, 1441U);
    failures += run(6, 4, {6, 5, 4, 3}, 1451U);
    failures += qk.verify_preserved("Q4 record qk weight");
    failures += value_z.verify_preserved("Q5 record value/z weight");
    return failures;
}

int run_w8() {
    constexpr std::int32_t kHidden    = 2048;
    constexpr std::int32_t kValueRows = 4096;
    constexpr std::int32_t kZRows     = 4096;
    DevicePackedWeight parent(
        quantized_weight::make_patterned_weight(QType::W8G32_F16S, 12288, kHidden, 1501U));

    int failures   = 0;
    const auto run = [&](std::int32_t width, std::int32_t batch, std::vector<std::int32_t> valid,
                         std::uint32_t seed) {
        const std::size_t snapshot_bytes =
            ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                kQueryRows, kKeyRows, kValueRows, batch, width, width);
        const std::size_t record_bytes = ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
            kQueryRows, kKeyRows, kValueRows, batch, width, width);
        return run_case(
            "W8 B=" + std::to_string(batch) + " T=" + std::to_string(width), kHidden, kValueRows,
            kZRows, width, batch, std::move(valid), snapshot_bytes, record_bytes,
            [&](const Tensor& x, const Tensor& conv, Tensor& state, const Tensor& valid_columns,
                const Tensor& initial, const Tensor& snapshot_base, Tensor& q, Tensor& k, Tensor& v,
                Tensor& z, WorkspaceArena& workspace) {
                ops::gdn_input_proj_conv_snapshot(x, parent.view(), conv, state, valid_columns,
                                                  initial, snapshot_base, q, k, v, z, workspace,
                                                  nullptr);
            },
            [&](const Tensor& x, const Tensor& conv, const Tensor& state,
                const Tensor& valid_columns, const Tensor& initial, Tensor& record, Tensor& q,
                Tensor& k, Tensor& v, Tensor& z, WorkspaceArena& workspace) {
                ops::gdn_input_proj_conv_record(x, parent.view(), conv, state, valid_columns,
                                                initial, record, q, k, v, z, workspace, nullptr);
            },
            seed);
    };
    failures += run(2, 1, {1}, 1511U);
    failures += run(16, 1, {}, 1521U);
    failures += run(16, 4, {16, 13, 9, 7}, 1531U);
    failures += parent.verify_preserved("W8 record parent weight");
    return failures;
}

int run_nvfp4() {
    constexpr std::int32_t kHidden    = 5120;
    constexpr std::int32_t kValueRows = 6144;
    constexpr std::int32_t kZRows     = 6144;
    constexpr std::int32_t kRows      = 16384;
    quantized_weight::PatternedWeightOptions options;
    options.weight_scale_divisor = 0.125F;
    options.input_scale_divisor  = 3.5F;
    DevicePackedWeight parent(
        quantized_weight::make_patterned_weight(QType::NVFP4, kRows, kHidden, 1601U, options));

    int failures   = 0;
    const auto run = [&](std::int32_t width, std::int32_t batch, std::vector<std::int32_t> valid,
                         ops::LinearPolicy policy, std::uint32_t seed) {
        std::vector<std::int32_t> valid_host = valid;
        if (valid_host.empty()) {
            valid_host.assign(static_cast<std::size_t>(batch), width);
        }
        const std::size_t snapshot_bytes =
            ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(QType::NVFP4, kRows, kHidden,
                                                                       policy, 1, 1, 1);
        const std::size_t record_bytes = ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
            QType::NVFP4, kRows, kHidden, policy, batch, width, width);
        return run_case(
            std::string("NVFP4 ") + (policy == ops::LinearPolicy::AllowA4 ? "A4" : "A16") +
                " B=" + std::to_string(batch) + " T=" + std::to_string(width),
            kHidden, kValueRows, kZRows, width, batch, std::move(valid), snapshot_bytes,
            record_bytes,
            [&, valid_host, policy, width, batch](
                const Tensor& x, const Tensor& conv, Tensor& state, const Tensor& valid_columns,
                const Tensor& initial, const Tensor& snapshot_base, Tensor& q, Tensor& k, Tensor& v,
                Tensor& z, WorkspaceArena& workspace) {
                (void)valid_columns;
                (void)initial;
                (void)snapshot_base;
                const std::int32_t hidden    = x.ne[0];
                const std::int32_t aggregate = width * batch;
                std::vector<std::int32_t> live(static_cast<std::size_t>(batch));
                for (std::int32_t b = 0; b < batch; ++b) {
                    live[static_cast<std::size_t>(b)] = aggregate + b;
                }
                auto column = [&](Tensor& tensor, std::int32_t b, std::int32_t t) {
                    auto* data = static_cast<std::uint8_t*>(tensor.data) +
                                 static_cast<std::int64_t>(b * width + t) * tensor.ne[0] *
                                     static_cast<std::int64_t>(sizeof(std::uint16_t));
                    return Tensor(data, DType::BF16, {tensor.ne[0], 1});
                };
                auto* x_bytes = static_cast<std::uint8_t*>(x.data);
                for (std::int32_t b = 0; b < batch; ++b) {
                    for (std::int32_t t = 0; t < width; ++t) {
                        const std::int32_t ok =
                            t < valid_host[static_cast<std::size_t>(b)] ? 1 : 0;
                        const std::int32_t base = b * width + t;
                        DeviceBuffer device_valid = to_device(std::vector<std::int32_t>{ok});
                        DeviceBuffer device_init =
                            to_device(std::vector<std::int32_t>{live[static_cast<std::size_t>(b)]});
                        DeviceBuffer device_base = to_device(std::vector<std::int32_t>{base});
                        Tensor x1(x_bytes + static_cast<std::int64_t>(base) * hidden *
                                                static_cast<std::int64_t>(sizeof(std::uint16_t)),
                                  DType::BF16, {hidden, 1});
                        Tensor q1      = column(q, b, t);
                        Tensor k1      = column(k, b, t);
                        Tensor v1      = column(v, b, t);
                        Tensor z1      = column(z, b, t);
                        Tensor valid_t(device_valid.p, DType::I32, {1});
                        Tensor init_t(device_init.p, DType::I32, {1});
                        Tensor base_t(device_base.p, DType::I32, {1});
                        ops::gdn_input_proj_conv_snapshot(x1, parent.view(), conv, state, valid_t,
                                                          init_t, base_t, q1, k1, v1, z1, policy,
                                                          workspace, nullptr);
                        if (ok != 0) { live[static_cast<std::size_t>(b)] = base; }
                    }
                }
            },
            [&](const Tensor& x, const Tensor& conv, const Tensor& state,
                const Tensor& valid_columns, const Tensor& initial, Tensor& record, Tensor& q,
                Tensor& k, Tensor& v, Tensor& z, WorkspaceArena& workspace) {
                ops::gdn_input_proj_conv_record(x, parent.view(), conv, state, valid_columns,
                                                initial, record, q, k, v, z, policy, workspace,
                                                nullptr);
            },
            seed);
    };
    failures += run(2, 1, {}, ops::LinearPolicy::A16Only, 1611U);
    failures += run(16, 1, {11}, ops::LinearPolicy::A16Only, 1621U);
    failures += run(3, 1, {2}, ops::LinearPolicy::AllowA4, 1631U);
    failures += run(4, 1, {}, ops::LinearPolicy::AllowA4, 1641U);
    failures += run(16, 1, {13}, ops::LinearPolicy::AllowA4, 1651U);
    failures += run(6, 3, {6, 4, 1}, ops::LinearPolicy::AllowA4, 1661U);
    failures += run(5, 3, {5, 5, 5}, ops::LinearPolicy::AllowA4, 1666U);
    failures += parent.verify_preserved("NVFP4 record parent weight");
    return failures;
}

int run_parent_index_tree() {
    constexpr std::int32_t kHidden    = 5120;
    constexpr std::int32_t kValueRows = 6144;
    constexpr std::int32_t kZRows     = 6144;
    constexpr std::int32_t kChannels  = kQueryRows + kKeyRows + kValueRows;
    constexpr std::int32_t kTreeWidth = 3;
    constexpr std::int32_t kSeqWidth  = 2;
    constexpr std::int32_t kBatch     = 2;
    constexpr std::int32_t kSlots     = 4;
    constexpr std::int32_t kInitial   = 3;

    DevicePackedWeight qk(
        quantized_weight::make_patterned_weight(QType::Q4G64_F16S, 4096, kHidden, 1701U));
    DevicePackedWeight value_z(
        quantized_weight::make_patterned_weight(QType::Q5G64_F16S, 12288, kHidden, 1703U));

    const std::vector<float> activation =
        make_bf16_activation(kHidden, kTreeWidth * kBatch, 1711U);
    const std::vector<std::uint16_t> conv_weight_bits =
        make_bf16_bits(static_cast<std::size_t>(kChannels) * 4, 1713U, -0.02F, 0.02F);
    const std::vector<std::uint16_t> state_before =
        make_bf16_bits(static_cast<std::size_t>(kChannels) * 3 * kSlots, 1715U, -0.05F, 0.05F);
    const std::vector<std::int32_t> initial_slots(static_cast<std::size_t>(kBatch), kInitial);
    std::vector<std::int32_t> parent_host(static_cast<std::size_t>(kTreeWidth * kBatch));
    for (std::int32_t batch = 0; batch < kBatch; ++batch) {
        parent_host[static_cast<std::size_t>(batch * kTreeWidth + 0)] = -1;
        parent_host[static_cast<std::size_t>(batch * kTreeWidth + 1)] = 0;
        parent_host[static_cast<std::size_t>(batch * kTreeWidth + 2)] = 0;
    }

    const auto pack_pair = [&](std::int32_t second) {
        std::vector<float> packed(static_cast<std::size_t>(kHidden) * kSeqWidth * kBatch);
        for (std::int32_t batch = 0; batch < kBatch; ++batch) {
            const std::array<std::int32_t, 2> tokens{0, second};
            for (std::int32_t dst = 0; dst < kSeqWidth; ++dst) {
                const std::int32_t src = tokens[static_cast<std::size_t>(dst)];
                const std::size_t src_base =
                    static_cast<std::size_t>((batch * kTreeWidth + src) * kHidden);
                const std::size_t dst_base =
                    static_cast<std::size_t>((batch * kSeqWidth + dst) * kHidden);
                std::copy_n(activation.begin() + static_cast<std::ptrdiff_t>(src_base), kHidden,
                            packed.begin() + static_cast<std::ptrdiff_t>(dst_base));
            }
        }
        return packed;
    };

    const auto launch_record = [&](const std::vector<float>& x_host, std::int32_t width,
                                   const Tensor* parent) {
        DeviceBuffer device_x           = to_device_bf16(x_host);
        DeviceBuffer device_conv_weight = to_device(conv_weight_bits);
        DeviceBuffer device_state       = to_device(state_before);
        DeviceBuffer device_initial     = to_device(initial_slots);
        DeviceBuffer device_parent;
        if (parent != nullptr) { device_parent = to_device(parent_host); }
        GuardedBf16Tensor query(kQueryRows, width * kBatch);
        GuardedBf16Tensor key(kKeyRows, width * kBatch);
        GuardedBf16Tensor value(kValueRows, width * kBatch);
        GuardedBf16Tensor z(kZRows, width * kBatch);
        GuardedBf16Tensor conv_record(kChannels, width * kBatch);
        Tensor x(device_x.p, DType::BF16, {kHidden, width, kBatch});
        Tensor conv_weight(device_conv_weight.p, DType::BF16, {kChannels, 4});
        Tensor state(device_state.p, DType::BF16, {kChannels, 3, kSlots});
        Tensor valid;
        Tensor initial(device_initial.p, DType::I32, {kBatch});
        Tensor q(query.data(), DType::BF16, {kQueryRows, width, kBatch});
        Tensor k(key.data(), DType::BF16, {kKeyRows, width, kBatch});
        Tensor v(value.data(), DType::BF16, {kValueRows, width, kBatch});
        Tensor z_view(z.data(), DType::BF16, {kZRows, width, kBatch});
        Tensor record(conv_record.data(), DType::BF16, {kChannels, width, kBatch});
        Tensor parent_tensor;
        const Tensor* parent_arg = nullptr;
        if (parent != nullptr) {
            parent_tensor = Tensor(device_parent.p, DType::I32, {width, kBatch});
            parent_arg    = &parent_tensor;
        }
        const std::size_t bytes = ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
            kQueryRows, kKeyRows, kValueRows, kBatch, width, width);
        WorkspaceArena workspace(std::max<std::size_t>(256, bytes));
        ops::gdn_input_proj_conv_record(x, qk.view(), value_z.view(), conv_weight, state, valid,
                                        initial, record, q, k, v, z_view, workspace, nullptr,
                                        parent_arg);
        cuda_synchronize();
        return std::tuple{query.bits(), key.bits(), value.bits(), z.bits(), conv_record.bits()};
    };

    Tensor dummy;
    const auto [tree_q, tree_k, tree_v, tree_z, tree_record] =
        launch_record(activation, kTreeWidth, &dummy);
    const auto [seq0_q, seq0_k, seq0_v, seq0_z, seq0_record] =
        launch_record(pack_pair(1), kSeqWidth, nullptr);
    const auto [seq1_q, seq1_k, seq1_v, seq1_z, seq1_record] =
        launch_record(pack_pair(2), kSeqWidth, nullptr);

    const auto compare = [&](std::string_view label, const std::vector<std::uint16_t>& tree_bits,
                             const std::vector<std::uint16_t>& seq_bits, std::int32_t rows,
                             std::int32_t tree_col, std::int32_t seq_col) {
        for (std::int32_t batch = 0; batch < kBatch; ++batch) {
            const std::size_t tree_base =
                static_cast<std::size_t>((batch * kTreeWidth + tree_col) * rows);
            const std::size_t seq_base =
                static_cast<std::size_t>((batch * kSeqWidth + seq_col) * rows);
            if (!std::equal(tree_bits.begin() + static_cast<std::ptrdiff_t>(tree_base),
                            tree_bits.begin() + static_cast<std::ptrdiff_t>(tree_base + rows),
                            seq_bits.begin() + static_cast<std::ptrdiff_t>(seq_base))) {
                std::cerr << "parent-index conv " << label << " mismatch\n";
                return 1;
            }
        }
        return 0;
    };

    int failures = 0;
    failures += compare("query child 0", tree_q, seq0_q, kQueryRows, 1, 1);
    failures += compare("query child 1", tree_q, seq1_q, kQueryRows, 2, 1);
    failures += compare("key child 0", tree_k, seq0_k, kKeyRows, 1, 1);
    failures += compare("key child 1", tree_k, seq1_k, kKeyRows, 2, 1);
    failures += compare("value child 0", tree_v, seq0_v, kValueRows, 1, 1);
    failures += compare("value child 1", tree_v, seq1_v, kValueRows, 2, 1);
    failures += compare("query parent", tree_q, seq0_q, kQueryRows, 0, 0);
    failures += compare("conv record child 0", tree_record, seq0_record, kChannels, 1, 1);
    failures += compare("conv record child 1", tree_record, seq1_record, kChannels, 2, 1);
    failures += qk.verify_preserved("Q4 parent-index qk weight");
    failures += value_z.verify_preserved("Q5 parent-index value/z weight");
    return failures;
}

int run_nvfp4_tree_column0_matches_decode() {
    constexpr std::int32_t kHidden     = 5120;
    constexpr std::int32_t kValueRows  = 6144;
    constexpr std::int32_t kZRows      = 6144;
    constexpr std::int32_t kChannels   = kQueryRows + kKeyRows + kValueRows;
    constexpr std::int32_t kRows       = 16384;
    constexpr std::int32_t kWidth      = 12;
    constexpr std::int32_t kSlots      = 4;
    constexpr std::int32_t kInitial    = 3;
    constexpr ReductionCriterion kA16{3.15e-3, 4.0e-3, 3.2e-3};

    quantized_weight::PatternedWeightOptions options;
    options.weight_scale_divisor = 0.125F;
    options.input_scale_divisor  = 3.5F;
    DevicePackedWeight parent(
        quantized_weight::make_patterned_weight(QType::NVFP4, kRows, kHidden, 1801U, options));

    const std::vector<float> packed_x = make_bf16_activation(kHidden, kWidth, 1811U);
    std::vector<float> decode_x(static_cast<std::size_t>(kHidden));
    std::copy_n(packed_x.begin(), kHidden, decode_x.begin());
    const std::vector<std::uint16_t> conv_weight_bits =
        make_bf16_bits(static_cast<std::size_t>(kChannels) * 4, 1813U, -0.02F, 0.02F);
    const std::vector<std::uint16_t> state_before =
        make_bf16_bits(static_cast<std::size_t>(kChannels) * 3 * kSlots, 1815U, -0.05F, 0.05F);
    const std::vector<std::int32_t> initial_slots{kInitial};
    const std::vector<std::int32_t> snapshot_bases{0};
    std::vector<std::int32_t> parent_host(static_cast<std::size_t>(kWidth), -1);
    for (std::int32_t token = 1; token < kWidth; ++token) {
        parent_host[static_cast<std::size_t>(token)] = token - 1;
    }

    DeviceBuffer device_packed = to_device_bf16(packed_x);
    DeviceBuffer device_decode = to_device_bf16(decode_x);
    DeviceBuffer device_conv   = to_device(conv_weight_bits);
    DeviceBuffer snapshot_state = to_device(state_before);
    DeviceBuffer record_state   = to_device(state_before);
    DeviceBuffer device_initial = to_device(initial_slots);
    DeviceBuffer device_base    = to_device(snapshot_bases);
    DeviceBuffer device_parent  = to_device(parent_host);

    GuardedBf16Tensor decode_q(kQueryRows, 1);
    GuardedBf16Tensor decode_k(kKeyRows, 1);
    GuardedBf16Tensor decode_v(kValueRows, 1);
    GuardedBf16Tensor decode_z(kZRows, 1);
    GuardedBf16Tensor tree_q(kQueryRows, kWidth);
    GuardedBf16Tensor tree_k(kKeyRows, kWidth);
    GuardedBf16Tensor tree_v(kValueRows, kWidth);
    GuardedBf16Tensor tree_z(kZRows, kWidth);
    GuardedBf16Tensor conv_record(kChannels, kWidth);

    Tensor x1(device_decode.p, DType::BF16, {kHidden, 1});
    Tensor xw(device_packed.p, DType::BF16, {kHidden, kWidth, 1});
    Tensor conv(device_conv.p, DType::BF16, {kChannels, 4});
    Tensor snap_state(snapshot_state.p, DType::BF16, {kChannels, 3, kSlots});
    Tensor rec_state(record_state.p, DType::BF16, {kChannels, 3, kSlots});
    Tensor initial(device_initial.p, DType::I32, {1});
    Tensor snap_base(device_base.p, DType::I32, {1});
    Tensor parent_index(device_parent.p, DType::I32, {kWidth, 1});
    Tensor q1 = decode_q.tensor();
    Tensor k1 = decode_k.tensor();
    Tensor v1 = decode_v.tensor();
    Tensor z1 = decode_z.tensor();
    Tensor qw(tree_q.data(), DType::BF16, {kQueryRows, kWidth, 1});
    Tensor kw(tree_k.data(), DType::BF16, {kKeyRows, kWidth, 1});
    Tensor vw(tree_v.data(), DType::BF16, {kValueRows, kWidth, 1});
    Tensor zw(tree_z.data(), DType::BF16, {kZRows, kWidth, 1});
    Tensor record(conv_record.data(), DType::BF16, {kChannels, kWidth, 1});

    const std::size_t snap_bytes = ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
        QType::NVFP4, kRows, kHidden, ops::LinearPolicy::AllowA4, 1, 1, 1);
    const std::size_t rec_bytes = ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
        QType::NVFP4, kRows, kHidden, ops::LinearPolicy::AllowA4, 1, kWidth, kWidth);
    WorkspaceArena snap_ws(std::max<std::size_t>(256, snap_bytes));
    WorkspaceArena rec_ws(std::max<std::size_t>(256, rec_bytes));

    ops::gdn_input_proj_conv_snapshot(x1, parent.view(), conv, snap_state, Tensor{}, initial,
                                      snap_base, q1, k1, v1, z1, ops::LinearPolicy::AllowA4,
                                      snap_ws, nullptr);
    ops::gdn_input_proj_conv_record(xw, parent.view(), conv, rec_state, Tensor{}, initial, record,
                                    qw, kw, vw, zw, ops::LinearPolicy::AllowA4, rec_ws, nullptr,
                                    &parent_index);
    cuda_synchronize();

    int failures = 0;
    failures += compare_column0("NVFP4 tree W=12 record query vs T=1 snapshot", tree_q, decode_q,
                                kQueryRows, kA16);
    failures += compare_column0("NVFP4 tree W=12 record key vs T=1 snapshot", tree_k, decode_k,
                                kKeyRows, kA16);
    failures += compare_column0("NVFP4 tree W=12 record value vs T=1 snapshot", tree_v, decode_v,
                                kValueRows, kA16);
    failures +=
        compare_column0("NVFP4 tree W=12 record z vs T=1 snapshot", tree_z, decode_z, kZRows, kA16);
    failures += parent.verify_preserved("NVFP4 tree W=12 parent weight");
    return failures;
}

int run_nvfp4_compose_chain_matches_snapshot() {
    constexpr std::int32_t kHidden    = 5120;
    constexpr std::int32_t kValueRows = 6144;
    constexpr std::int32_t kZRows     = 6144;
    constexpr std::int32_t kChannels  = kQueryRows + kKeyRows + kValueRows;
    constexpr std::int32_t kRows      = 16384;
    constexpr std::int32_t kWidth     = 12;
    constexpr std::int32_t kSlots     = kWidth + 1;
    constexpr std::int32_t kInitial   = kWidth;
    constexpr ReductionCriterion kA16{3.15e-3, 4.0e-3, 3.2e-3};

    quantized_weight::PatternedWeightOptions options;
    options.weight_scale_divisor = 0.125F;
    options.input_scale_divisor  = 3.5F;
    DevicePackedWeight parent(
        quantized_weight::make_patterned_weight(QType::NVFP4, kRows, kHidden, 1801U, options));

    const std::vector<float> packed_x = make_bf16_activation(kHidden, kWidth, 1811U);
    const std::vector<std::uint16_t> conv_weight_bits =
        make_bf16_bits(static_cast<std::size_t>(kChannels) * 4, 1813U, -0.02F, 0.02F);
    const std::vector<std::uint16_t> state_before =
        make_bf16_bits(static_cast<std::size_t>(kChannels) * 3 * kSlots, 1815U, -0.05F, 0.05F);
    const std::vector<std::int32_t> initial_slots{kInitial};
    std::vector<std::int32_t> parent_host(static_cast<std::size_t>(kWidth), -1);
    for (std::int32_t token = 1; token < kWidth; ++token) {
        parent_host[static_cast<std::size_t>(token)] = token - 1;
    }

    DeviceBuffer device_packed  = to_device_bf16(packed_x);
    DeviceBuffer device_conv    = to_device(conv_weight_bits);
    DeviceBuffer record_state   = to_device(state_before);
    DeviceBuffer snapshot_state = to_device(state_before);
    DeviceBuffer device_initial = to_device(initial_slots);
    DeviceBuffer device_parent  = to_device(parent_host);

    GuardedBf16Tensor compose_q(kQueryRows, kWidth);
    GuardedBf16Tensor compose_k(kKeyRows, kWidth);
    GuardedBf16Tensor compose_v(kValueRows, kWidth);
    GuardedBf16Tensor compose_z(kZRows, kWidth);
    GuardedBf16Tensor conv_record(kChannels, kWidth);

    Tensor xw(device_packed.p, DType::BF16, {kHidden, kWidth, 1});
    Tensor conv(device_conv.p, DType::BF16, {kChannels, 4});
    Tensor rec_state(record_state.p, DType::BF16, {kChannels, 3, kSlots});
    Tensor snap_state(snapshot_state.p, DType::BF16, {kChannels, 3, kSlots});
    Tensor initial(device_initial.p, DType::I32, {1});
    Tensor parent_index(device_parent.p, DType::I32, {kWidth, 1});
    Tensor cq(compose_q.data(), DType::BF16, {kQueryRows, kWidth, 1});
    Tensor ck(compose_k.data(), DType::BF16, {kKeyRows, kWidth, 1});
    Tensor cv(compose_v.data(), DType::BF16, {kValueRows, kWidth, 1});
    Tensor cz(compose_z.data(), DType::BF16, {kZRows, kWidth, 1});
    Tensor record(conv_record.data(), DType::BF16, {kChannels, kWidth, 1});

    const std::size_t rec_bytes = ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
        QType::NVFP4, kRows, kHidden, ops::LinearPolicy::AllowA4, 1, kWidth, kWidth);
    WorkspaceArena rec_ws(std::max<std::size_t>(256, rec_bytes));
    ops::gdn_input_proj_conv_record(xw, parent.view(), conv, rec_state, Tensor{}, initial, record,
                                    cq, ck, cv, cz, ops::LinearPolicy::AllowA4, rec_ws, nullptr,
                                    &parent_index);
    cuda_synchronize();

    const std::size_t snap_bytes = ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
        QType::NVFP4, kRows, kHidden, ops::LinearPolicy::AllowA4, 1, 1, 1);
    WorkspaceArena snap_ws(std::max<std::size_t>(256, snap_bytes));
    GuardedBf16Tensor decode_q(kQueryRows, 1);
    GuardedBf16Tensor decode_k(kKeyRows, 1);
    GuardedBf16Tensor decode_v(kValueRows, 1);
    GuardedBf16Tensor decode_z(kZRows, 1);

    int failures = 0;
    for (std::int32_t token = 0; token < kWidth; ++token) {
        std::vector<float> decode_x(static_cast<std::size_t>(kHidden));
        std::copy_n(packed_x.begin() + static_cast<std::ptrdiff_t>(token) * kHidden, kHidden,
                    decode_x.begin());
        DeviceBuffer device_decode = to_device_bf16(decode_x);
        const std::int32_t initial_slot = token == 0 ? kInitial : token - 1;
        const std::int32_t snapshot_base = token;
        DeviceBuffer device_token_initial = to_device(std::vector<std::int32_t>{initial_slot});
        DeviceBuffer device_token_base    = to_device(std::vector<std::int32_t>{snapshot_base});
        Tensor x1(device_decode.p, DType::BF16, {kHidden, 1});
        Tensor token_initial(device_token_initial.p, DType::I32, {1});
        Tensor token_base(device_token_base.p, DType::I32, {1});
        Tensor q1 = decode_q.tensor();
        Tensor k1 = decode_k.tensor();
        Tensor v1 = decode_v.tensor();
        Tensor z1 = decode_z.tensor();
        ops::gdn_input_proj_conv_snapshot(x1, parent.view(), conv, snap_state, Tensor{},
                                          token_initial, token_base, q1, k1, v1, z1,
                                          ops::LinearPolicy::AllowA4, snap_ws, nullptr);
        cuda_synchronize();
        const std::string suffix = " col=" + std::to_string(token);
        failures += compare_packed_column_to_decode(
            "NVFP4 compose vs T=1 snapshot query" + suffix, compose_q, token, decode_q, kQueryRows,
            kA16);
        failures += compare_packed_column_to_decode(
            "NVFP4 compose vs T=1 snapshot key" + suffix, compose_k, token, decode_k, kKeyRows, kA16);
        failures += compare_packed_column_to_decode(
            "NVFP4 compose vs T=1 snapshot value" + suffix, compose_v, token, decode_v, kValueRows,
            kA16);
        failures += compare_packed_column_to_decode(
            "NVFP4 compose vs T=1 snapshot z" + suffix, compose_z, token, decode_z, kZRows, kA16);
        if (failures != 0) { return failures; }
    }
    failures += parent.verify_preserved("NVFP4 compose vs T=1 snapshot parent weight");
    return failures;
}

int run_nvfp4_tree_chain_matches_sequential_fused() {
    constexpr std::int32_t kHidden    = 5120;
    constexpr std::int32_t kValueRows = 6144;
    constexpr std::int32_t kZRows     = 6144;
    constexpr std::int32_t kChannels  = kQueryRows + kKeyRows + kValueRows;
    constexpr std::int32_t kRows      = 16384;
    constexpr std::int32_t kWidth     = 12;
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

    DeviceBuffer device_packed   = to_device_bf16(packed_x);
    DeviceBuffer device_conv     = to_device(conv_weight_bits);
    DeviceBuffer sequential_state = to_device(state_before);
    DeviceBuffer tree_state      = to_device(state_before);
    DeviceBuffer device_initial  = to_device(initial_slots);
    DeviceBuffer device_parent   = to_device(chain_parent);

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
        QType::NVFP4, kRows, kHidden, ops::LinearPolicy::AllowA4, 1, kWidth, kWidth);
    WorkspaceArena seq_ws(std::max<std::size_t>(256, rec_bytes));
    WorkspaceArena tree_ws(std::max<std::size_t>(256, rec_bytes));
    ops::gdn_input_proj_conv_record(xw, parent.view(), conv, seq_state, Tensor{}, initial, sr, sq,
                                    sk, sv, sz, ops::LinearPolicy::AllowA4, seq_ws, nullptr);
    ops::gdn_input_proj_conv_record(xw, parent.view(), conv, rec_state, Tensor{}, initial, tr, tq,
                                    tk, tv, tz, ops::LinearPolicy::AllowA4, tree_ws, nullptr,
                                    &parent_index);
    cuda_synchronize();

    int failures = 0;
    failures += verify_equal("NVFP4 fused chain-parent vs sequential query", tree_q.bits(),
                             seq_q.bits());
    failures +=
        verify_equal("NVFP4 fused chain-parent vs sequential key", tree_k.bits(), seq_k.bits());
    failures += verify_equal("NVFP4 fused chain-parent vs sequential value", tree_v.bits(),
                             seq_v.bits());
    failures +=
        verify_equal("NVFP4 fused chain-parent vs sequential z", tree_z.bits(), seq_z.bits());
    failures += verify_equal("NVFP4 fused chain-parent vs sequential conv_record",
                             tree_record.bits(), seq_record.bits());
    failures += parent.verify_preserved("NVFP4 fused chain-parent parent weight");
    return failures;
}

int run_nvfp4_batched_matches_serial_fused() {
    // A multi-request record kernel must preserve each sequence's T=1 GEMV reduction and
    // BF16-history/FP32-convolution path. Independent B=1 launches are the exact arithmetic
    // reference; the decoded-weight FP64 oracle below independently checks the public formula.
    constexpr std::int32_t kHidden    = 5120;
    constexpr std::int32_t kValueRows = 6144;
    constexpr std::int32_t kZRows     = 6144;
    constexpr std::int32_t kChannels  = kQueryRows + kKeyRows + kValueRows;
    constexpr std::int32_t kRows      = 16384;

    quantized_weight::PatternedWeightOptions options;
    options.weight_scale_divisor = 0.125F;
    options.input_scale_divisor  = 3.5F;
    DevicePackedWeight parent(
        quantized_weight::make_patterned_weight(QType::NVFP4, kRows, kHidden, 2001U, options));

    const auto run_shape = [&](std::int32_t width, std::int32_t batch,
                               std::vector<std::int32_t> valid_columns,
                               std::vector<std::int32_t> parent_indices, std::uint32_t seed) {
        const std::int32_t aggregate = width * batch;
        const std::int32_t slots     = aggregate + batch + 1;
        const bool dense             = valid_columns.empty();
        if (dense) { valid_columns.assign(static_cast<std::size_t>(batch), width); }
        if (!parent_indices.empty() &&
            parent_indices.size() != static_cast<std::size_t>(aggregate)) {
            throw std::invalid_argument("NVFP4 batched record parent fixture has wrong size");
        }

        const std::vector<float> activation = make_bf16_activation(kHidden, aggregate, seed);
        const std::vector<std::uint16_t> activation_bits = bf16_bits(activation);
        const std::vector<std::uint16_t> conv_weight_bits =
            make_bf16_bits(static_cast<std::size_t>(kChannels) * 4, seed + 1, -0.02F, 0.02F);
        const std::vector<std::uint16_t> state_before =
            make_bf16_bits(static_cast<std::size_t>(kChannels) * 3 * slots, seed + 2, -0.05F, 0.05F);

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

        const std::size_t rec_bytes = ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
            QType::NVFP4, kRows, kHidden, ops::LinearPolicy::AllowA4, batch, width, width);
        WorkspaceArena batched_ws(std::max<std::size_t>(256, rec_bytes));
        ops::gdn_input_proj_conv_record(x, parent.view(), conv, batched_state_view, valid, initial,
                                        br, bq, bk, bv, bz, ops::LinearPolicy::AllowA4, batched_ws,
                                        nullptr,
                                        parent_indices.empty() ? nullptr : &parent_index);
        const std::size_t serial_bytes = ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
            QType::NVFP4, kRows, kHidden, ops::LinearPolicy::AllowA4, 1, width, width);
        WorkspaceArena serial_ws(std::max<std::size_t>(256, serial_bytes));
        for (std::int32_t batch_row = 0; batch_row < batch; ++batch_row) {
            Tensor valid_b;
            if (!dense) { valid_b = valid.slice(0, batch_row, 1); }
            Tensor parent_b;
            if (!parent_indices.empty()) {
                auto* row_parent = static_cast<std::int32_t*>(device_parent.p) +
                                   static_cast<std::ptrdiff_t>(batch_row) * width;
                parent_b = Tensor(row_parent, DType::I32, {width, 1});
            }
            Tensor rb = sr.slice(2, batch_row, 1);
            Tensor qb = sq.slice(2, batch_row, 1);
            Tensor kb = sk.slice(2, batch_row, 1);
            Tensor vb = sv.slice(2, batch_row, 1);
            Tensor zb = sz.slice(2, batch_row, 1);
            ops::gdn_input_proj_conv_record(
                x.slice(2, batch_row, 1), parent.view(), conv, serial_state_view, valid_b,
                initial.slice(0, batch_row, 1), rb, qb, kb, vb, zb, ops::LinearPolicy::AllowA4,
                serial_ws, nullptr, parent_indices.empty() ? nullptr : &parent_b);
        }
        cuda_synchronize();

        const std::string label = "NVFP4 batched vs serial B=" + std::to_string(batch) +
                                  " W=" + std::to_string(width) +
                                  (parent_indices.empty() ? " sequential" : " tree");
        int failures = 0;
        failures += verify_equal(label + " query", batched_q.bits(), serial_q.bits());
        failures += verify_equal(label + " key", batched_k.bits(), serial_k.bits());
        failures += verify_equal(label + " value", batched_v.bits(), serial_v.bits());
        failures += verify_equal(label + " z", batched_z.bits(), serial_z.bits());
        failures += verify_valid_record_equal(label + " conv_record", batched_record.bits(),
                                              serial_record.bits(), kChannels, width, batch,
                                              valid_columns);
        failures += verify_zero_tail(label + " query", batched_q.bits(), kQueryRows, width,
                                     batch, valid_columns);
        failures += verify_zero_tail(label + " key", batched_k.bits(), kKeyRows, width, batch,
                                     valid_columns);
        failures += verify_zero_tail(label + " value", batched_v.bits(), kValueRows, width,
                                     batch, valid_columns);
        failures += verify_nvfp4_record_oracle(
            label, parent.host, activation, conv_weight_bits, state_before, initial_slots,
            valid_columns, parent_indices, batched_q, batched_k, batched_v, batched_z,
            batched_record, kHidden, kValueRows, width, batch);
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
            failures +=
                verify_preserved(label + " parent index", device_parent, parent_indices);
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
    failures += run_shape(2, 2, {}, {}, 1981U);
    failures += run_shape(2, 3, {}, {}, 1987U);
    failures += run_shape(2, 4, {2, 2, 1, 1}, {}, 1991U);
    failures += run_shape(2, 4, {2, 2, 2, 1}, {-1, 0, -1, 0, -1, 0, -1, 0}, 1997U);
    failures += run_shape(5, 2, {}, {}, 2011U);
    failures += run_shape(5, 3, {}, {}, 2021U);
    failures += run_shape(5, 4, {}, {}, 2031U);
    failures += run_shape(5, 2, {5, 2}, {}, 2041U);
    failures += run_shape(5, 3, {5, 4, 1}, {}, 2051U);
    failures += run_shape(5, 4, {5, 4, 2, 1}, {}, 2061U);
    failures += run_shape(5, 2, {}, {-1, 0, 0, 1, 1, -1, 0, 1, 1, 3}, 2071U);
    failures += run_shape(5, 3, {5, 3, 2},
                          {-1, 0, 0, 1, 1, -1, 0, 1, 1, 3, -1, 0, 0, 2, 2}, 2077U);
    failures += run_shape(5, 4, {5, 4, 3, 2},
                          {-1, 0, 0, 1, 1, -1, 0, 1, 1, 3, -1, 0, 0, 2, 2, -1, 0, 1, 2, 3},
                          2081U);
    failures += parent.verify_preserved("NVFP4 batched vs serial parent weight");
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cerr << "FAIL: no usable CUDA device\n";
        return 1;
    }

    int failures = 0;
    failures += run_q4_q5();
    failures += run_w8();
    failures += run_nvfp4();
    failures += run_parent_index_tree();
    failures += run_nvfp4_tree_column0_matches_decode();
    failures += run_nvfp4_compose_chain_matches_snapshot();
    failures += run_nvfp4_tree_chain_matches_sequential_fused();
    failures += run_nvfp4_batched_matches_serial_fused();
    std::cout << (failures == 0 ? "OK" : "FAIL") << " gdn_input_proj_conv_record\n";
    return failures == 0 ? 0 : 1;
}
