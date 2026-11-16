// Qwen3.8-Flash-Next lane State: a lane reserves exactly ceil(reserve / 64) pages of the shared QSA
// pool and maps them in its block-table row; admission fails when no lane or not enough free pages
// remain; closing unmaps the row and returns the pages, which a reopened lane reuses; a lane
// cannot advance past its reservation; every one-lane selector is 16-byte aligned.

#include "runtime/state.h"

#include "core/device.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace fn = ninfer::targets::qwen3_8_flash_next;

namespace {

int failures = 0;

void expect(bool condition, const std::string& what) {
    if (!condition) {
        std::cerr << "failed: " << what << '\n';
        ++failures;
    }
}

// The block-table row of `lane` (logical page capacity entries).
std::vector<std::int32_t> table_row(fn::State& state, std::int32_t lane) {
    const ninfer::Tensor& tables = state.qsa(0).block_tables;
    std::vector<std::int32_t> row(static_cast<std::size_t>(tables.ne[0]));
    CUDA_CHECK(cudaMemcpy(row.data(),
                          static_cast<const std::int32_t*>(tables.data) +
                              static_cast<std::size_t>(lane) * row.size(),
                          row.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost));
    return row;
}

std::int32_t mapped(const std::vector<std::int32_t>& row) {
    return static_cast<std::int32_t>(
        std::ranges::count_if(row, [](std::int32_t page) { return page >= 0; }));
}

void run() {
    ninfer::DeviceContext device(0);
    // Three lanes of at most 640 tokens (10 pages) over a 1000-token pool (16 pages).
    fn::State state(3, 640, 1000, ninfer::ops::QsaKvFormat::NVFP4G16);
    expect(state.lane_capacity() == 640, "lane capacity is the page-rounded context");
    for (std::int32_t lane = 0; lane < 3; ++lane) {
        const ninfer::Tensor id = state.lane_id(lane);
        std::int32_t value      = -1;
        CUDA_CHECK(cudaMemcpy(&value, id.data, sizeof(value), cudaMemcpyDeviceToHost));
        expect(reinterpret_cast<std::uintptr_t>(id.data) % 16 == 0 && value == lane,
               "lane selector " + std::to_string(lane) + " is 16-byte aligned and holds its lane");
    }
    expect(state.pool_tokens() == 1024 && state.free_tokens() == 1024, "pool of 16 pages");
    expect(!state.can_open(641) && !state.can_open(0), "reservations outside [1, context]");

    const std::optional<std::int32_t> a = state.open(129, device.stream); // 3 pages
    const std::optional<std::int32_t> b = state.open(640, device.stream); // 10 pages
    CUDA_CHECK(cudaStreamSynchronize(device.stream));
    expect(a == 0 && b == 1, "lanes open lowest first");
    expect(state.free_tokens() == 3 * 64, "exactly ceil(reserve / 64) pages are reserved");
    expect(mapped(table_row(state, 0)) == 3 && mapped(table_row(state, 1)) == 10,
           "each lane maps its reserved pages");
    expect(!state.can_open(193) && state.can_open(192), "admission follows the free pages");
    expect(state.reserved_tokens(0) == 129 && state.committed(0) == 0, "lane bookkeeping");

    state.advance(0, 129);
    bool rejected = false;
    try {
        state.advance(0, 1);
    } catch (const std::out_of_range&) { rejected = true; }
    expect(rejected, "advancing past the reservation throws");

    const std::vector<std::int32_t> pages_b = table_row(state, 1);
    state.close(1, device.stream);
    CUDA_CHECK(cudaStreamSynchronize(device.stream));
    expect(mapped(table_row(state, 1)) == 0, "a closed lane's row is unmapped");
    expect(state.free_tokens() == 13 * 64 && !state.is_open(1), "closing returns the pages");

    const std::optional<std::int32_t> c = state.open(640, device.stream);
    CUDA_CHECK(cudaStreamSynchronize(device.stream));
    expect(c == 1, "the closed lane is reopened");
    std::vector<std::int32_t> reused = table_row(state, 1);
    std::vector<std::int32_t> before = pages_b;
    std::ranges::sort(reused);
    std::ranges::sort(before);
    expect(reused == before, "the reopened lane reuses the returned pages");
    expect(state.committed(1) == 0, "a reopened lane starts at position 0");

    const std::optional<std::int32_t> d = state.open(192, device.stream);
    expect(d == 2 && state.free_tokens() == 0, "the last pages are admitted");
    expect(!state.can_open(1) && !state.open(1, device.stream).has_value(),
           "no lane is free and no page remains");
    CUDA_CHECK(cudaStreamSynchronize(device.stream));

    // Exact continuation transport includes FP32 recurrent values, BF16 convolution/PLE history
    // and the BF16 tail distribution; it must not quantize or depend on the physical lane.
    for (std::int32_t layer = 0; layer < fn::kGdnLayers; ++layer) {
        const auto conv      = state.gdn_convolution(layer, 0);
        const auto recurrent = state.gdn_recurrent(layer, 0);
        CUDA_CHECK(cudaMemsetAsync(conv.data, layer + 1, conv.bytes(), device.stream));
        CUDA_CHECK(cudaMemsetAsync(recurrent.data, layer + 7, recurrent.bytes(), device.stream));
    }
    auto ple    = state.ple_history().slice(2, 0, 1);
    auto logits = state.tail_logits(0);
    CUDA_CHECK(cudaMemsetAsync(ple.data, 37, ple.bytes(), device.stream));
    CUDA_CHECK(cudaMemsetAsync(logits.data, 53, logits.bytes(), device.stream));
    ninfer::PinnedHostBuffer saved(state.continuation_bytes());
    ninfer::PinnedHostBuffer restored(state.continuation_bytes());
    state.pack_continuation(0, saved.data(), device.stream);
    state.unpack_continuation(1, saved.data(), device.stream);
    state.pack_continuation(1, restored.data(), device.stream);
    CUDA_CHECK(cudaStreamSynchronize(device.stream));
    expect(std::memcmp(saved.data(), restored.data(), saved.size()) == 0,
           "continuation survives a cross-lane pinned RAM round trip exactly");
    state.copy_prefix(1, 0, 129, device.stream);
    state.pack_continuation(1, restored.data(), device.stream);
    CUDA_CHECK(cudaStreamSynchronize(device.stream));
    expect(std::memcmp(saved.data(), restored.data(), saved.size()) == 0,
           "independent prefix copy preserves all continuation bytes");
    state.resize(0, 65, 65, device.stream);
    CUDA_CHECK(cudaStreamSynchronize(device.stream));
    expect(state.committed(0) == 65 && mapped(table_row(state, 0)) == 2 &&
               state.free_tokens() == 64,
           "retention releases future growth and keeps the partial final page");
    const auto retained_pages = table_row(state, 0);
    state.resize(0, 65, 129, device.stream);
    CUDA_CHECK(cudaStreamSynchronize(device.stream));
    const auto grown_pages = table_row(state, 0);
    expect(state.free_tokens() == 0 && grown_pages[0] == retained_pages[0] &&
               grown_pages[1] == retained_pages[1],
           "claiming retention preserves its page mapping and funds suffix growth");
}

} // namespace

int main() {
    try {
        run();
        if (failures != 0) {
            std::cerr << failures << " failures\n";
            return 1;
        }
        std::cout << "State: ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "State test failed: " << error.what() << '\n';
        return 1;
    }
}
