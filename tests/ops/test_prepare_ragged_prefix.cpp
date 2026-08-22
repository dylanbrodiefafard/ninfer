#include "ninfer/ops/prepare_ragged_prefix.h"
#include "ops/op_tester.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
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

std::size_t column_index(std::int32_t d, std::int32_t width, std::int32_t lane, std::int32_t column,
                         std::int32_t row) {
    return static_cast<std::size_t>(lane * width + column) * static_cast<std::size_t>(d) +
           static_cast<std::size_t>(row);
}

int run_case(const char* label, std::int32_t d, std::int32_t width, std::int32_t capacity,
             const std::vector<std::int32_t>& lanes, const std::vector<std::int32_t>& starts,
             const std::vector<std::int32_t>& ends) {
    const auto batch = static_cast<std::int32_t>(lanes.size());
    const auto source =
        bit_pattern(static_cast<std::size_t>(d) * width * capacity, 0x51ed'c0deu);
    std::vector<std::uint16_t> expected_dest(static_cast<std::size_t>(d) * width * batch, 0);
    std::vector<std::int32_t> expected_positions(static_cast<std::size_t>(width) * batch);
    std::vector<std::int32_t> expected_counts(static_cast<std::size_t>(batch));
    for (std::int32_t b = 0; b < batch; ++b) {
        const std::int32_t n = ends[static_cast<std::size_t>(b)] - starts[static_cast<std::size_t>(b)];
        expected_counts[static_cast<std::size_t>(b)] = n;
        const std::int32_t lane                      = lanes[static_cast<std::size_t>(b)];
        const std::int32_t start                     = starts[static_cast<std::size_t>(b)];
        for (std::int32_t column = 0; column < width; ++column) {
            const std::size_t pos_index =
                static_cast<std::size_t>(b) * width + static_cast<std::size_t>(column);
            if (column < n) {
                expected_positions[pos_index] = start + column;
                for (std::int32_t row = 0; row < d; ++row) {
                    expected_dest[column_index(d, width, b, column, row)] =
                        source[column_index(d, width, lane, column, row)];
                }
            } else {
                expected_positions[pos_index] = n == 0 ? start : start + n - 1;
            }
        }
    }

    GuardedDeviceBuffer device_source(source.size() * sizeof(std::uint16_t));
    DeviceBuffer device_lanes  = to_device(lanes);
    DeviceBuffer device_starts = to_device(starts);
    DeviceBuffer device_ends   = to_device(ends);
    GuardedDeviceBuffer device_dest(expected_dest.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_positions(expected_positions.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer device_counts(expected_counts.size() * sizeof(std::int32_t));
    device_source.copy_from_host(source.data(), device_source.bytes());
    device_dest.fill(0xcd);
    device_positions.fill(0xef);
    device_counts.fill(0xab);

    Tensor source_tensor(device_source.data(), DType::BF16, {d, width, capacity});
    Tensor lanes_tensor(device_lanes.p, DType::I32, {batch});
    Tensor starts_tensor(device_starts.p, DType::I32, {batch});
    Tensor ends_tensor(device_ends.p, DType::I32, {batch});
    Tensor dest_tensor(device_dest.data(), DType::BF16, {d, width, batch});
    Tensor positions_tensor(device_positions.data(), DType::I32, {width, batch});
    Tensor counts_tensor(device_counts.data(), DType::I32, {batch});
    ops::prepare_ragged_prefix(source_tensor, lanes_tensor, starts_tensor, ends_tensor, dest_tensor,
                               positions_tensor, counts_tensor, nullptr);
    cuda_synchronize();

    int failures = verify_exact((std::string(label) + " values").c_str(),
                                from_device<std::uint16_t>(device_dest.data(), expected_dest.size()),
                                expected_dest);
    failures += verify_exact((std::string(label) + " positions").c_str(),
                             from_device<std::int32_t>(device_positions.data(),
                                                       expected_positions.size()),
                             expected_positions);
    failures += verify_exact((std::string(label) + " counts").c_str(),
                             from_device<std::int32_t>(device_counts.data(), expected_counts.size()),
                             expected_counts);
    failures += verify_exact((std::string(label) + " preserves source").c_str(),
                             from_device<std::uint16_t>(device_source.data(), source.size()), source);
    failures += device_source.verify_guards((std::string(label) + " source").c_str());
    failures += device_dest.verify_guards((std::string(label) + " dest").c_str());
    failures += device_positions.verify_guards((std::string(label) + " positions").c_str());
    failures += device_counts.verify_guards((std::string(label) + " counts").c_str());
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cerr << "FAIL: no usable CUDA device\n";
        return 1;
    }

    int failures = 0;
    failures += run_case("prepare_ragged_prefix N=0", 8, 4, 2, {1}, {40}, {40});
    failures += run_case("prepare_ragged_prefix N=W", 8, 4, 2, {0}, {10}, {14});
    failures += run_case("prepare_ragged_prefix mixed", 64, 4, 4, {2, 0, 3}, {10, 20, 30},
                         {10, 22, 34});
    std::cout << (failures ? "FAIL" : "OK") << " prepare_ragged_prefix\n";
    return failures ? 1 : 0;
}
