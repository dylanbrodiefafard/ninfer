#include "ninfer/ops/gdn_input_proj.h"

#include "ops/input_projection_test_common.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
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
    failures += run(6, 8, {6, 5, 4, 3, 2, 1, 6, 2}, 1451U);
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
    failures += run(16, 8, {16, 13, 9, 7, 5, 3, 2, 1}, 1531U);
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
        const std::size_t snapshot_bytes =
            ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(QType::NVFP4, kRows, kHidden,
                                                                       policy, batch, width, width);
        const std::size_t record_bytes = ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
            QType::NVFP4, kRows, kHidden, policy, batch, width, width);
        return run_case(
            std::string("NVFP4 ") + (policy == ops::LinearPolicy::AllowA4 ? "A4" : "A16") +
                " B=" + std::to_string(batch) + " T=" + std::to_string(width),
            kHidden, kValueRows, kZRows, width, batch, std::move(valid), snapshot_bytes,
            record_bytes,
            [&](const Tensor& x, const Tensor& conv, Tensor& state, const Tensor& valid_columns,
                const Tensor& initial, const Tensor& snapshot_base, Tensor& q, Tensor& k, Tensor& v,
                Tensor& z, WorkspaceArena& workspace) {
                ops::gdn_input_proj_conv_snapshot(x, parent.view(), conv, state, valid_columns,
                                                  initial, snapshot_base, q, k, v, z, policy,
                                                  workspace, nullptr);
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
    std::cout << (failures == 0 ? "OK" : "FAIL") << " gdn_input_proj_conv_record\n";
    return failures == 0 ? 0 : 1;
}
