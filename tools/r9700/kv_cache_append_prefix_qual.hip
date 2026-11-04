#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "ninfer/ops/kv_cache_append_prefix.h"
#include "ninfer/types.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kD      = 128;
constexpr int kHeads  = 8;
constexpr int kWidth  = 12;
constexpr int kPage   = 64;
constexpr int kWindow = 2048;

template <class T>
ninfer::DeviceBuffer upload(const std::vector<T>& host) {
    ninfer::DeviceBuffer result(host.size() * sizeof(T));
    result.copy_from_host(host.data(), result.size());
    return result;
}

template <class T>
std::vector<T> download(const ninfer::DeviceBuffer& device, std::size_t elements) {
    std::vector<T> result(elements);
    device.copy_to_host(result.data(), elements * sizeof(T));
    return result;
}

std::vector<std::uint16_t> bits(std::size_t count, std::uint32_t seed) {
    std::vector<std::uint16_t> result(count);
    for (std::size_t i = 0; i < count; ++i) {
        seed      = seed * 1664525U + 1013904223U;
        result[i] = static_cast<std::uint16_t>(seed >> 16U);
    }
    return result;
}

std::size_t input_index(int d, int head, int token, int batch) {
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kD) *
               (static_cast<std::size_t>(head) +
                static_cast<std::size_t>(kHeads) *
                    (static_cast<std::size_t>(token) + static_cast<std::size_t>(kWidth) * batch));
}

void require_equal(const std::vector<std::uint16_t>& actual,
                   const std::vector<std::uint16_t>& expected, const char* label) {
    if (actual == expected) { return; }
    const auto mismatch = std::mismatch(actual.begin(), actual.end(), expected.begin());
    throw std::runtime_error(std::string(label) + " differs at element " +
                             std::to_string(mismatch.first - actual.begin()));
}

void cyclic_oracle(std::vector<std::uint16_t>& cache_key, std::vector<std::uint16_t>& cache_value,
                   const std::vector<std::uint16_t>& key, const std::vector<std::uint16_t>& value,
                   const std::vector<std::int32_t>& positions,
                   const std::vector<std::int32_t>& counts, const std::vector<std::int32_t>& lanes,
                   int batch) {
    for (int row = 0; row < batch; ++row) {
        for (int token = 0; token < counts[row]; ++token) {
            const int slot =
                positions[static_cast<std::size_t>(kWidth) * row + token] & (kWindow - 1);
            for (int head = 0; head < kHeads; ++head) {
                for (int d = 0; d < kD; ++d) {
                    const std::size_t source = input_index(d, head, token, row);
                    const std::size_t destination =
                        static_cast<std::size_t>(d) +
                        static_cast<std::size_t>(kD) *
                            (static_cast<std::size_t>(slot) +
                             kWindow * (static_cast<std::size_t>(head) +
                                        static_cast<std::size_t>(kHeads) * lanes[row]));
                    cache_key[destination]   = key[source];
                    cache_value[destination] = value[source];
                }
            }
        }
    }
}

void run_cyclic(ninfer::DeviceContext& device) {
    constexpr int batch              = static_cast<int>(ninfer::kDFlashMaximumConcurrency);
    constexpr int lanes              = batch;
    const std::size_t input_elements = static_cast<std::size_t>(kD) * kHeads * kWidth * batch;
    const std::size_t cache_elements = static_cast<std::size_t>(kD) * kWindow * kHeads * lanes;
    const auto key                   = bits(input_elements, 0x12345678U);
    const auto value                 = bits(input_elements, 0x9abcdef0U);
    const auto initial_key           = bits(cache_elements, 0x31415926U);
    const auto initial_value         = bits(cache_elements, 0x27182818U);
    std::vector<std::int32_t> positions(static_cast<std::size_t>(kWidth) * batch);
    for (int row = 0; row < batch; ++row) {
        for (int token = 0; token < kWidth; ++token) {
            positions[static_cast<std::size_t>(kWidth) * row + token] = 2044 + row * 2049 + token;
        }
    }
    std::vector<std::int32_t> counts(batch, 0);
    // A fixed non-identity permutation of every lane.
    std::vector<std::int32_t> selectors(batch);
    for (int row = 0; row < batch; ++row) { selectors[row] = (row * 3 + batch - 1) % batch; }
    auto d_key         = upload(key);
    auto d_value       = upload(value);
    auto d_positions   = upload(positions);
    auto d_counts      = upload(counts);
    auto d_selectors   = upload(selectors);
    auto d_cache_key   = upload(initial_key);
    auto d_cache_value = upload(initial_value);
    const ninfer::Tensor key_tensor(d_key.data(), ninfer::DType::BF16, {kD, kHeads, kWidth, batch});
    const ninfer::Tensor value_tensor(d_value.data(), ninfer::DType::BF16,
                                      {kD, kHeads, kWidth, batch});
    const ninfer::Tensor position_tensor(d_positions.data(), ninfer::DType::I32, {kWidth, batch});
    const ninfer::Tensor count_tensor(d_counts.data(), ninfer::DType::I32, {batch});
    const ninfer::Tensor selector_tensor(d_selectors.data(), ninfer::DType::I32, {batch});
    const ninfer::CyclicKVCacheLayerView cache{
        .k = ninfer::Tensor(d_cache_key.data(), ninfer::DType::BF16, {kD, kWindow, kHeads, lanes}),
        .v =
            ninfer::Tensor(d_cache_value.data(), ninfer::DType::BF16, {kD, kWindow, kHeads, lanes}),
        .capacity        = kWindow,
        .padded_capacity = kWindow,
        .num_kv_heads    = kHeads,
        .head_dim        = kD,
        .lane_capacity   = lanes,
    };

    ninfer::DecodeGraphDefinition definition;
    definition.capture(device.stream, [&] {
        ninfer::ops::kv_cache_append_prefix(key_tensor, value_tensor, position_tensor, count_tensor,
                                            selector_tensor, {0, kWidth}, cache, device.stream);
    });
    ninfer::DecodeGraphExecutable executable;
    executable.instantiate(definition);

    for (const std::array<std::int32_t, batch>& replay_counts :
         {std::array<std::int32_t, batch>{0, 3, 7, 12, 1, 12, 5, 9},
          std::array<std::int32_t, batch>{12, 8, 2, 0, 11, 4, 12, 6}}) {
        counts.assign(replay_counts.begin(), replay_counts.end());
        d_counts.copy_from_host(counts.data(), d_counts.size());
        d_cache_key.copy_from_host(initial_key.data(), d_cache_key.size());
        d_cache_value.copy_from_host(initial_value.data(), d_cache_value.size());
        executable.launch(device.stream);
        device.synchronize();
        auto expected_key   = initial_key;
        auto expected_value = initial_value;
        cyclic_oracle(expected_key, expected_value, key, value, positions, counts, selectors,
                      batch);
        require_equal(download<std::uint16_t>(d_cache_key, cache_elements), expected_key,
                      "cyclic key");
        require_equal(download<std::uint16_t>(d_cache_value, cache_elements), expected_value,
                      "cyclic value");
    }
    require_equal(download<std::uint16_t>(d_key, input_elements), key, "cyclic source key");
    require_equal(download<std::uint16_t>(d_value, input_elements), value, "cyclic source value");

    bool rejected = false;
    try {
        auto wrong     = cache;
        wrong.capacity = 4096;
        ninfer::ops::kv_cache_append_prefix(key_tensor, value_tensor, position_tensor, count_tensor,
                                            selector_tensor, {0, kWidth}, wrong, device.stream);
    } catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) { throw std::runtime_error("non-Qwen DFlash cyclic geometry was accepted"); }
}

std::size_t paged_index(int d, int head, int position, int physical_pages,
                        const std::vector<std::int32_t>& tables, int row, int logical_pages) {
    const int page = tables[static_cast<std::size_t>(logical_pages) * row + position / kPage];
    return static_cast<std::size_t>(d) +
           static_cast<std::size_t>(kD) * (static_cast<std::size_t>(position % kPage) +
                                           static_cast<std::size_t>(kPage) *
                                               (static_cast<std::size_t>(page) +
                                                static_cast<std::size_t>(physical_pages) * head));
}

void run_paged(ninfer::DeviceContext& device) {
    constexpr int batch              = 2;
    constexpr int logical_pages      = 3;
    constexpr int physical_pages     = 4;
    const std::size_t input_elements = static_cast<std::size_t>(kD) * kHeads * kWidth * batch;
    const std::size_t cache_elements =
        static_cast<std::size_t>(kD) * kPage * physical_pages * kHeads;
    const auto key           = bits(input_elements, 0xabcdef01U);
    const auto value         = bits(input_elements, 0x10fedcbaU);
    const auto initial_key   = bits(cache_elements, 0x55aa55aaU);
    const auto initial_value = bits(cache_elements, 0xaa55aa55U);
    std::vector<std::int32_t> positions(static_cast<std::size_t>(kWidth) * batch);
    for (int row = 0; row < batch; ++row) {
        for (int token = 0; token < kWidth; ++token) {
            positions[static_cast<std::size_t>(kWidth) * row + token] = 60 + 5 * row + token;
        }
    }
    std::vector<std::int32_t> counts{12, 5};
    std::vector<std::int32_t> rows{0, 1};
    const std::vector<std::int32_t> tables{2, 0, 1, 3, 1, 0};
    auto d_key         = upload(key);
    auto d_value       = upload(value);
    auto d_positions   = upload(positions);
    auto d_counts      = upload(counts);
    auto d_rows        = upload(rows);
    auto d_tables      = upload(tables);
    auto d_cache_key   = upload(initial_key);
    auto d_cache_value = upload(initial_value);
    const ninfer::Tensor key_tensor(d_key.data(), ninfer::DType::BF16, {kD, kHeads, kWidth, batch});
    const ninfer::Tensor value_tensor(d_value.data(), ninfer::DType::BF16,
                                      {kD, kHeads, kWidth, batch});
    const ninfer::Tensor position_tensor(d_positions.data(), ninfer::DType::I32, {kWidth, batch});
    const ninfer::Tensor count_tensor(d_counts.data(), ninfer::DType::I32, {batch});
    const ninfer::Tensor row_tensor(d_rows.data(), ninfer::DType::I32, {batch});
    const ninfer::ops::KVCacheAppendPrefixPagedView cache{
        .key_pages    = ninfer::Tensor(d_cache_key.data(), ninfer::DType::BF16,
                                       {kD, kPage, physical_pages, kHeads}),
        .value_pages  = ninfer::Tensor(d_cache_value.data(), ninfer::DType::BF16,
                                       {kD, kPage, physical_pages, kHeads}),
        .block_tables = ninfer::Tensor(d_tables.data(), ninfer::DType::I32, {logical_pages, batch}),
        .head_dim     = kD,
        .num_kv_heads = kHeads,
    };
    ninfer::ops::kv_cache_append_prefix(key_tensor, value_tensor, position_tensor, count_tensor,
                                        row_tensor, {0, kWidth}, cache, device.stream);
    device.synchronize();
    auto expected_key   = initial_key;
    auto expected_value = initial_value;
    for (int row = 0; row < batch; ++row) {
        for (int token = 0; token < counts[row]; ++token) {
            const int position = positions[static_cast<std::size_t>(kWidth) * row + token];
            for (int head = 0; head < kHeads; ++head) {
                for (int d = 0; d < kD; ++d) {
                    const std::size_t source      = input_index(d, head, token, row);
                    const std::size_t destination = paged_index(d, head, position, physical_pages,
                                                                tables, rows[row], logical_pages);
                    expected_key[destination]     = key[source];
                    expected_value[destination]   = value[source];
                }
            }
        }
    }
    require_equal(download<std::uint16_t>(d_cache_key, cache_elements), expected_key, "paged key");
    require_equal(download<std::uint16_t>(d_cache_value, cache_elements), expected_value,
                  "paged value");

    bool rejected = false;
    try {
        const ninfer::Tensor alias_positions(d_key.data(), ninfer::DType::I32, {kWidth, batch});
        ninfer::ops::kv_cache_append_prefix(key_tensor, value_tensor, alias_positions, count_tensor,
                                            row_tensor, {0, kWidth}, cache, device.stream);
    } catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) { throw std::runtime_error("overlapping append inputs were accepted"); }
}

} // namespace

int main() {
    try {
        ninfer::DeviceContext device(0);
        if (!device.is_gfx1201() || device.props.warpSize != 32) {
            throw std::runtime_error("qualifier requires wave32 gfx1201");
        }
        run_cyclic(device);
        run_paged(device);
        std::cout << "kv_cache_append_prefix: PASS exact BF16 D128 Hkv8 W12 B8 "
                     "cyclic2048 graph_replay paged64\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "kv_cache_append_prefix: FAIL: " << error.what() << '\n';
        return 1;
    }
}
