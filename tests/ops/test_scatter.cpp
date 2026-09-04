#include "ninfer/ops/prepare_ragged_prefix.h"
#include "ninfer/ops/scatter.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

std::vector<std::uint16_t> bit_pattern(std::size_t count, std::uint32_t seed) {
    std::vector<std::uint16_t> values(count);
    std::uint32_t state = seed;
    for (std::size_t i = 0; i < count; ++i) {
        state     = state * 1664525u + 1013904223u;
        values[i] = static_cast<std::uint16_t>((state >> 16) ^ static_cast<std::uint32_t>(i));
    }
    return values;
}

int scatter_case(std::int32_t rows, const std::vector<std::int32_t>& indices,
                 std::int32_t destination_columns) {
    const std::int32_t source_columns = static_cast<std::int32_t>(indices.size());
    const auto source = bit_pattern(static_cast<std::size_t>(rows) * source_columns, 0x1324'68acu);
    const auto destination =
        bit_pattern(static_cast<std::size_t>(rows) * destination_columns, 0x9876'4321u);
    auto expected = destination;
    for (std::int32_t source_column = 0; source_column < source_columns; ++source_column) {
        const std::int32_t destination_column = indices[static_cast<std::size_t>(source_column)];
        for (std::int32_t row = 0; row < rows; ++row) {
            expected[static_cast<std::size_t>(destination_column) * rows + row] =
                source[static_cast<std::size_t>(source_column) * rows + row];
        }
    }

    GuardedDeviceBuffer device_source(source.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_indices(indices.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer device_destination(destination.size() * sizeof(std::uint16_t));
    device_source.copy_from_host(source.data(), source.size() * sizeof(std::uint16_t));
    device_indices.copy_from_host(indices.data(), indices.size() * sizeof(std::int32_t));
    device_destination.copy_from_host(destination.data(),
                                      destination.size() * sizeof(std::uint16_t));

    Tensor source_tensor(device_source.data(), DType::BF16, {rows, source_columns});
    Tensor indices_tensor(device_indices.data(), DType::I32, {source_columns});
    Tensor destination_tensor(device_destination.data(), DType::BF16, {rows, destination_columns});
    ops::scatter(source_tensor, indices_tensor, destination_tensor, nullptr);
    cuda_synchronize();

    const std::string label =
        "scatter D=" + std::to_string(rows) + " V=" + std::to_string(source_columns);
    int failures = 0;
    failures += verify_exact(
        label.c_str(), from_device<std::uint16_t>(device_destination.data(), destination.size()),
        expected);
    failures +=
        verify_exact((label + " preserves source").c_str(),
                     from_device<std::uint16_t>(device_source.data(), source.size()), source);
    failures +=
        verify_exact((label + " preserves indices").c_str(),
                     from_device<std::int32_t>(device_indices.data(), indices.size()), indices);
    failures += device_source.verify_guards((label + " source").c_str());
    failures += device_indices.verify_guards((label + " indices").c_str());
    failures += device_destination.verify_guards((label + " destination").c_str());
    return failures;
}

int extract_case(std::int32_t source_rows, std::int32_t destination_rows,
                 std::int32_t source_offset, std::int32_t tokens) {
    const auto source = bit_pattern(static_cast<std::size_t>(source_rows) * tokens, 0x1357'9bdfu);
    std::vector<std::uint16_t> expected(static_cast<std::size_t>(destination_rows) * tokens);
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t row = 0; row < destination_rows; ++row) {
            expected[static_cast<std::size_t>(token) * destination_rows + row] =
                source[static_cast<std::size_t>(token) * source_rows + source_offset + row];
        }
    }

    GuardedDeviceBuffer device_source(source.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_destination(expected.size() * sizeof(std::uint16_t));
    device_source.copy_from_host(source.data(), source.size() * sizeof(std::uint16_t));
    device_destination.fill(0xcd);
    Tensor source_tensor(device_source.data(), DType::BF16, {source_rows, tokens});
    Tensor destination_tensor(device_destination.data(), DType::BF16, {destination_rows, tokens});
    ops::extract_bf16_columns(source_tensor, source_offset, destination_tensor, nullptr);
    cuda_synchronize();

    const std::string label = "extract_bf16_columns source=" + std::to_string(source_rows) +
                              " offset=" + std::to_string(source_offset) +
                              " rows=" + std::to_string(destination_rows) +
                              " T=" + std::to_string(tokens);
    int failures = 0;
    failures += verify_exact(label.c_str(),
                             from_device<std::uint16_t>(device_destination.data(), expected.size()),
                             expected);
    failures +=
        verify_exact((label + " preserves source").c_str(),
                     from_device<std::uint16_t>(device_source.data(), source.size()), source);
    failures += device_source.verify_guards((label + " source").c_str());
    failures += device_destination.verify_guards((label + " destination").c_str());
    return failures;
}

int batch_prefix_case() {
    constexpr std::int32_t rows     = 8;
    constexpr std::int32_t width    = 3;
    constexpr std::int32_t batch    = 2;
    constexpr std::int32_t capacity = 3;
    const std::vector<std::int32_t> lanes{2, 0};
    const std::vector<std::int32_t> valid{2, 3};
    const std::vector<std::int32_t> starts{10, 20};
    const std::vector<std::int32_t> ends{12, 23};
    const auto source = bit_pattern(static_cast<std::size_t>(rows * width * batch), 0x2468'ace0u);
    const auto initial =
        bit_pattern(static_cast<std::size_t>(rows * width * capacity), 0x3141'5926u);
    auto expected_pool = initial;
    for (std::int32_t b = 0; b < batch; ++b) {
        for (std::int32_t column = 0; column < valid[static_cast<std::size_t>(b)]; ++column) {
            for (std::int32_t row = 0; row < rows; ++row) {
                expected_pool[static_cast<std::size_t>(lanes[static_cast<std::size_t>(b)] * width +
                                                       column) *
                                  rows +
                              row] =
                    source[static_cast<std::size_t>(b * width + column) * rows + row];
            }
        }
    }

    DeviceBuffer device_source = to_device(source);
    DeviceBuffer device_lanes  = to_device(lanes);
    DeviceBuffer device_valid  = to_device(valid);
    DeviceBuffer device_starts = to_device(starts);
    DeviceBuffer device_ends   = to_device(ends);
    GuardedDeviceBuffer device_pool(initial.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_gathered(source.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_positions(static_cast<std::size_t>(width * batch) *
                                         sizeof(std::int32_t));
    GuardedDeviceBuffer device_counts(static_cast<std::size_t>(batch) * sizeof(std::int32_t));
    device_pool.copy_from_host(initial.data(), initial.size() * sizeof(std::uint16_t));
    device_gathered.fill(0xcd);
    device_positions.fill(0xef);
    device_counts.fill(0xab);

    Tensor source_tensor(device_source.p, DType::BF16, {rows, width, batch});
    Tensor lanes_tensor(device_lanes.p, DType::I32, {batch});
    Tensor valid_tensor(device_valid.p, DType::I32, {batch});
    Tensor pool_tensor(device_pool.data(), DType::BF16, {rows, width, capacity});
    ops::scatter_bf16_batch(source_tensor, lanes_tensor, valid_tensor, pool_tensor, nullptr);

    Tensor starts_tensor(device_starts.p, DType::I32, {batch});
    Tensor ends_tensor(device_ends.p, DType::I32, {batch});
    Tensor gathered_tensor(device_gathered.data(), DType::BF16, {rows, width, batch});
    Tensor positions_tensor(device_positions.data(), DType::I32, {width, batch});
    Tensor counts_tensor(device_counts.data(), DType::I32, {batch});
    ops::prepare_ragged_prefix(pool_tensor, lanes_tensor, starts_tensor, ends_tensor,
                               gathered_tensor, positions_tensor, counts_tensor, nullptr);
    cuda_synchronize();

    auto expected_gathered = source;
    std::fill(expected_gathered.begin() + static_cast<std::ptrdiff_t>(2 * rows),
              expected_gathered.begin() + static_cast<std::ptrdiff_t>(3 * rows), 0);
    const std::vector<std::int32_t> expected_positions{10, 11, 11, 20, 21, 22};

    int failures = verify_exact(
        "scatter_bf16_batch B=2 pool",
        from_device<std::uint16_t>(device_pool.data(), expected_pool.size()), expected_pool);
    failures +=
        verify_exact("prepare_ragged_prefix B=2 values",
                     from_device<std::uint16_t>(device_gathered.data(), expected_gathered.size()),
                     expected_gathered);
    failures +=
        verify_exact("prepare_ragged_prefix B=2 positions",
                     from_device<std::int32_t>(device_positions.data(), expected_positions.size()),
                     expected_positions);
    failures += verify_exact("prepare_ragged_prefix B=2 counts",
                             from_device<std::int32_t>(device_counts.data(), valid.size()), valid);
    failures += device_pool.verify_guards("scatter_bf16_batch B=2 pool guards");
    failures += device_gathered.verify_guards("prepare_ragged_prefix B=2 values guards");
    failures += device_positions.verify_guards("prepare_ragged_prefix B=2 positions guards");
    failures += device_counts.verify_guards("prepare_ragged_prefix B=2 counts guards");
    return failures;
}

int pitched_live_width_case() {
    // Adaptive DFlash stores pending_features at W_ceil and verify/scatter at W(k).
    // scatter dest is a dim-1 prefix of the W_ceil pool; prepare dest matches W_ceil.
    constexpr std::int32_t rows     = 8;
    constexpr std::int32_t w_ceil   = 12;
    constexpr std::int32_t vw       = 6;
    constexpr std::int32_t batch    = 2;
    constexpr std::int32_t capacity = 3;
    const std::vector<std::int32_t> lanes{2, 0};
    const std::vector<std::int32_t> valid{6, 4};
    const auto source = bit_pattern(static_cast<std::size_t>(rows * vw * batch), 0x51ed51edu);
    const auto initial =
        bit_pattern(static_cast<std::size_t>(rows * w_ceil * capacity), 0xa11ce110u);
    auto expected_pool = initial;
    for (std::int32_t b = 0; b < batch; ++b) {
        for (std::int32_t column = 0; column < valid[static_cast<std::size_t>(b)]; ++column) {
            for (std::int32_t row = 0; row < rows; ++row) {
                expected_pool[static_cast<std::size_t>(
                                  (lanes[static_cast<std::size_t>(b)] * w_ceil + column) * rows +
                                  row)] =
                    source[static_cast<std::size_t>((b * vw + column) * rows + row)];
            }
        }
    }

    DeviceBuffer device_source = to_device(source);
    DeviceBuffer device_lanes  = to_device(lanes);
    DeviceBuffer device_valid  = to_device(valid);
    GuardedDeviceBuffer device_pool(initial.size() * sizeof(std::uint16_t));
    device_pool.copy_from_host(initial.data(), initial.size() * sizeof(std::uint16_t));

    Tensor source_tensor(device_source.p, DType::BF16, {rows, vw, batch});
    Tensor lanes_tensor(device_lanes.p, DType::I32, {batch});
    Tensor valid_tensor(device_valid.p, DType::I32, {batch});
    Tensor pool_ceil(device_pool.data(), DType::BF16, {rows, w_ceil, capacity});
    Tensor pool_live = pool_ceil.slice(1, 0, vw);
    ops::scatter_bf16_batch(source_tensor, lanes_tensor, valid_tensor, pool_live, nullptr);
    cuda_synchronize();

    int failures = verify_exact(
        "scatter live W into W_ceil pending",
        from_device<std::uint16_t>(device_pool.data(), expected_pool.size()), expected_pool);
    failures += device_pool.verify_guards("scatter live W pool guards");

    bool threw = false;
    try {
        ops::scatter_bf16_batch(source_tensor, lanes_tensor, valid_tensor, pool_ceil, nullptr);
    } catch (const std::invalid_argument&) { threw = true; }
    failures += threw ? 0 : 1;
    if (!threw) {
        std::cerr << "FAIL: scatter_bf16_batch accepted W(k) source into W_ceil dest\n";
    }

    const std::vector<std::int32_t> starts{0, 10};
    const std::vector<std::int32_t> ends{6, 14};
    DeviceBuffer device_starts = to_device(starts);
    DeviceBuffer device_ends   = to_device(ends);
    GuardedDeviceBuffer device_gathered(static_cast<std::size_t>(rows * w_ceil * batch) *
                                        sizeof(std::uint16_t));
    GuardedDeviceBuffer device_positions(static_cast<std::size_t>(w_ceil * batch) *
                                         sizeof(std::int32_t));
    GuardedDeviceBuffer device_counts(static_cast<std::size_t>(batch) * sizeof(std::int32_t));
    device_gathered.fill(0xcd);
    device_positions.fill(0xef);
    device_counts.fill(0xab);
    Tensor gathered(device_gathered.data(), DType::BF16, {rows, w_ceil, batch});
    Tensor positions(device_positions.data(), DType::I32, {w_ceil, batch});
    Tensor counts(device_counts.data(), DType::I32, {batch});
    Tensor starts_tensor(device_starts.p, DType::I32, {batch});
    Tensor ends_tensor(device_ends.p, DType::I32, {batch});
    ops::prepare_ragged_prefix(pool_ceil, lanes_tensor, starts_tensor, ends_tensor, gathered,
                               positions, counts, nullptr);
    cuda_synchronize();

    std::vector<std::uint16_t> expected_gathered(static_cast<std::size_t>(rows * w_ceil * batch), 0);
    const std::vector<std::int32_t> expected_counts{6, 4};
    for (std::int32_t b = 0; b < batch; ++b) {
        const std::int32_t count = expected_counts[static_cast<std::size_t>(b)];
        for (std::int32_t column = 0; column < count; ++column) {
            for (std::int32_t row = 0; row < rows; ++row) {
                expected_gathered[static_cast<std::size_t>((b * w_ceil + column) * rows + row)] =
                    expected_pool[static_cast<std::size_t>(
                        (lanes[static_cast<std::size_t>(b)] * w_ceil + column) * rows + row)];
            }
        }
    }
    failures += verify_exact(
        "prepare_ragged_prefix from W_ceil pending",
        from_device<std::uint16_t>(device_gathered.data(), expected_gathered.size()),
        expected_gathered);
    failures += verify_exact("prepare_ragged_prefix W_ceil counts",
                             from_device<std::int32_t>(device_counts.data(), expected_counts.size()),
                             expected_counts);

    threw = false;
    GuardedDeviceBuffer device_narrow(static_cast<std::size_t>(rows * vw * batch) *
                                      sizeof(std::uint16_t));
    Tensor gathered_vw(device_narrow.data(), DType::BF16, {rows, vw, batch});
    GuardedDeviceBuffer device_pos_vw(static_cast<std::size_t>(vw * batch) * sizeof(std::int32_t));
    Tensor positions_vw(device_pos_vw.data(), DType::I32, {vw, batch});
    try {
        ops::prepare_ragged_prefix(pool_ceil, lanes_tensor, starts_tensor, ends_tensor,
                                   gathered_vw, positions_vw, counts, nullptr);
    } catch (const std::invalid_argument&) { threw = true; }
    failures += threw ? 0 : 1;
    if (!threw) {
        std::cerr << "FAIL: prepare_ragged_prefix accepted W_ceil source into W(k) dest\n";
    }

    GuardedDeviceBuffer device_compact(static_cast<std::size_t>(rows * vw * batch) *
                                       sizeof(std::uint16_t));
    device_compact.fill(0xcd);
    Tensor compact(device_compact.data(), DType::BF16, {rows, vw, batch});
    const std::size_t elem     = sizeof(std::uint16_t);
    const std::size_t live_row = static_cast<std::size_t>(rows) * static_cast<std::size_t>(vw) * elem;
    cuda_check(cudaMemcpy2D(compact.data, static_cast<std::size_t>(compact.nb[2]), gathered.data,
                            static_cast<std::size_t>(gathered.nb[2]), live_row,
                            static_cast<std::size_t>(batch), cudaMemcpyDeviceToDevice),
               "wide-to-narrow W_ceil pack");
    cuda_synchronize();
    std::vector<std::uint16_t> expected_compact(static_cast<std::size_t>(rows * vw * batch));
    for (std::int32_t b = 0; b < batch; ++b) {
        for (std::int32_t column = 0; column < vw; ++column) {
            for (std::int32_t row = 0; row < rows; ++row) {
                expected_compact[static_cast<std::size_t>((b * vw + column) * rows + row)] =
                    expected_gathered[static_cast<std::size_t>((b * w_ceil + column) * rows + row)];
            }
        }
    }
    failures += verify_exact(
        "pack W_ceil features down to W(k)",
        from_device<std::uint16_t>(device_compact.data(), expected_compact.size()),
        expected_compact);
    failures += device_compact.verify_guards("pack W_ceil compact guards");
    return failures;
}

// Host fold of a DFlash2 binary tree is never the identity 0,1,2,... spine, so
// gather_bf16_path actually runs. Compact row index is not the KV/feature lane:
// OpenCode C=2 can occupy lanes {1,0}. Indexing features by compact row would
// swap the two chats' pending features after a branched accept.
int gather_path_crossed_lanes_case() {
    constexpr std::int32_t rows     = 8;
    constexpr std::int32_t width    = 12;
    constexpr std::int32_t capacity = 3;
    constexpr std::int32_t batch    = 2;
    const std::vector<std::int32_t> lanes{2, 0};
    const std::vector<std::int32_t> counts{3, 4};
    const std::vector<std::int32_t> path{
        0, 2, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // compact row 0 → lane 2
        0, 1, 3, 7, 0, 0, 0, 0, 0, 0, 0, 0,  // compact row 1 → lane 0
    };
    const auto initial =
        bit_pattern(static_cast<std::size_t>(rows) * width * capacity, 0x0d15ea5eu);
    auto expected = initial;
    const auto column = [&](std::int32_t lane, std::int32_t col) -> std::size_t {
        return static_cast<std::size_t>(lane * width + col) * static_cast<std::size_t>(rows);
    };
    for (std::int32_t b = 0; b < batch; ++b) {
        const std::int32_t lane  = lanes[static_cast<std::size_t>(b)];
        const std::int32_t count = counts[static_cast<std::size_t>(b)];
        std::vector<std::uint16_t> saved(static_cast<std::size_t>(count) *
                                         static_cast<std::size_t>(rows));
        for (std::int32_t i = 0; i < count; ++i) {
            const std::int32_t src = path[static_cast<std::size_t>(b) * width + i];
            std::copy(initial.begin() + static_cast<std::ptrdiff_t>(column(lane, src)),
                      initial.begin() + static_cast<std::ptrdiff_t>(column(lane, src) + rows),
                      saved.begin() + static_cast<std::ptrdiff_t>(i * rows));
        }
        for (std::int32_t i = 0; i < count; ++i) {
            std::copy(saved.begin() + static_cast<std::ptrdiff_t>(i * rows),
                      saved.begin() + static_cast<std::ptrdiff_t>((i + 1) * rows),
                      expected.begin() + static_cast<std::ptrdiff_t>(column(lane, i)));
        }
    }

    GuardedDeviceBuffer device_features(initial.size() * sizeof(std::uint16_t));
    device_features.copy_from_host(initial.data(), initial.size() * sizeof(std::uint16_t));
    DeviceBuffer device_lanes  = to_device(lanes);
    DeviceBuffer device_path   = to_device(path);
    DeviceBuffer device_counts = to_device(counts);
    Tensor features_t(device_features.data(), DType::BF16, {rows, width, capacity});
    Tensor lanes_t(device_lanes.p, DType::I32, {batch});
    Tensor path_t(device_path.p, DType::I32, {width, batch});
    Tensor counts_t(device_counts.p, DType::I32, {batch});
    ops::gather_bf16_path(features_t, lanes_t, path_t, counts_t, nullptr);
    cuda_synchronize();

    int failures = verify_exact(
        "gather_bf16_path B=2 crossed lanes",
        from_device<std::uint16_t>(device_features.data(), expected.size()), expected);
    failures += device_features.verify_guards("gather_bf16_path B=2 crossed lanes guards");
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cerr << "FAIL: no usable CUDA device\n";
        return 1;
    }

    int failures = 0;
    failures += scatter_case(5120, {4, 0, 7, 2}, 9);
    failures += scatter_case(2048, {5, 1, 3}, 7);
    failures += extract_case(10240, 6144, 4096, 6);
    failures += extract_case(8192, 2048, 2048, 1);
    failures += batch_prefix_case();
    failures += pitched_live_width_case();
    failures += gather_path_crossed_lanes_case();
    std::cout << (failures ? "FAIL" : "OK") << " scatter and extract_bf16_columns\n";
    return failures ? 1 : 0;
}
