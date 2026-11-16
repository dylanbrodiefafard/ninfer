// Qwen3.8-Flash-Next KV extents as expert-cache flex slots: the extent geometry over real QSA pools
// (NVFP4 244 pages and 84 records, BF16 122 pages and 108 records; addresses aligned, disjoint and
// inside their extent's run of one plane), and the CacheHeld / KvLent state machine against an
// independent model: pool page states, the lanes' mapped pages, and the cache's device state
// (borrowed and returned flex owners, evicted table entries and slot index) after make_room and
// give_back, including the one-extent spare that give_back keeps free.

#include "runtime/expert_residency.h"
#include "runtime/kv_extents.h"
#include "runtime/state.h"

#include "core/device.h"
#include "ninfer/ops/moe_expert_cache.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace fn  = ninfer::targets::qwen3_8_flash_next;
namespace ops = ninfer::ops;

namespace {

int failures = 0;

void expect(bool condition, const std::string& what) {
    if (!condition) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

template <typename T>
std::vector<T> download(const void* device, std::size_t count) {
    std::vector<T> host(count);
    CUDA_CHECK(cudaMemcpy(host.data(), device, count * sizeof(T), cudaMemcpyDeviceToHost));
    return host;
}

constexpr std::size_t kF3Bytes = 1'996'800;

// Checks the geometry of a pool of `pages` pages against an independent computation from the plane
// strides.
void test_geometry(ops::QsaKvFormat format, std::uint32_t extent_pages, std::int32_t slots,
                   std::uint32_t pages) {
    const std::string label = format == ops::QsaKvFormat::BF16 ? "BF16" : "NVFP4";
    fn::State state(1, 64, std::int64_t{pages} * 64, format);
    ninfer::PagedKVPool& pool = state.pool();
    const fn::KvExpertExtents extents(pool, format, nullptr, nullptr);
    const std::uint32_t count = pages / extent_pages;
    expect(fn::kv_extent_pages(format) == extent_pages && extents.extent_pages() == extent_pages,
           label + " extent pages");
    expect(extents.slots_per_extent() == slots, label + " records per extent");
    expect(extents.extents() == static_cast<std::int32_t>(count) &&
               extents.first_page() == pages - count * extent_pages,
           label + " flex region");
    expect(pool.withheld_pages() == count * extent_pages &&
               pool.free_pages() == pages - count * extent_pages,
           label + " every extent starts withheld");

    std::vector<std::uint64_t> expected;
    for (std::uint32_t x = 0; x < count; ++x) {
        const std::uint32_t first = extents.first_page() + x * extent_pages;
        for (std::size_t p = 0; p < pool.plane_count(); ++p) {
            const ninfer::Tensor& plane = pool.plane(p);
            const auto page_bytes       = static_cast<std::uint64_t>(plane.nb[3]);
            const std::uint64_t run =
                reinterpret_cast<std::uint64_t>(plane.data) + first * page_bytes;
            const std::uint64_t end = run + extent_pages * page_bytes;
            for (std::uint64_t address = run; address + kF3Bytes <= end; address += kF3Bytes) {
                expected.push_back(address);
            }
        }
    }
    const auto addresses = extents.flex_addresses();
    expect(std::vector<std::uint64_t>(addresses.begin(), addresses.end()) == expected,
           label + " flex addresses");
    expect(std::ranges::all_of(addresses, [](std::uint64_t a) { return a % 256 == 0; }),
           label + " flex alignment");
    std::vector<std::uint64_t> sorted(addresses.begin(), addresses.end());
    std::ranges::sort(sorted);
    for (std::size_t i = 1; i < sorted.size(); ++i) {
        expect(sorted[i - 1] + kF3Bytes <= sorted[i], label + " flex slots are disjoint");
    }
    bool rejected = false;
    try {
        const fn::KvExpertExtents again(pool, format, nullptr, nullptr);
    } catch (const std::invalid_argument&) { rejected = true; }
    expect(rejected, label + " extents need an idle pool");
}

// The device state of the F3 flex slots and the tables after each transition, against the model.
struct Model {
    std::vector<std::int32_t> owner;         // flex slot owners
    std::vector<std::uint64_t> expert_table; // [512]
    std::vector<std::int32_t> slot_index;    // [512]
    std::vector<bool> lent;
};

void lend_model(Model& model, std::int32_t extent, std::int32_t per_extent,
                std::span<const std::uint64_t> host) {
    for (std::int32_t j = 0; j < per_extent; ++j) {
        std::int32_t& owner =
            model.owner[static_cast<std::size_t>(extent) * static_cast<std::size_t>(per_extent) +
                        static_cast<std::size_t>(j)];
        if (owner >= 0) {
            model.expert_table[static_cast<std::size_t>(owner)] =
                host[static_cast<std::size_t>(owner)];
            model.slot_index[static_cast<std::size_t>(owner)] = ops::kMoeCacheNoSlot;
        }
        owner = ops::kMoeCacheBorrowedSlot;
    }
    model.lent[static_cast<std::size_t>(extent)] = true;
}

void return_model(Model& model, std::int32_t extent, std::int32_t per_extent) {
    for (std::int32_t j = 0; j < per_extent; ++j) {
        model.owner[static_cast<std::size_t>(extent) * static_cast<std::size_t>(per_extent) +
                    static_cast<std::size_t>(j)] = ops::kMoeCacheEmptySlot;
    }
    model.lent[static_cast<std::size_t>(extent)] = false;
}

void test_state_machine() {
    ninfer::DeviceContext device(0);
    constexpr std::uint32_t kExtentPages = 244;
    constexpr std::uint32_t kLowPages    = 40;
    constexpr std::uint32_t kPages       = kLowPages + 2 * kExtentPages;
    constexpr std::int32_t kFixed        = 4;
    fn::State state(2, 320 * 64, std::int64_t{kPages} * 64, ops::QsaKvFormat::NVFP4G16);
    ninfer::PagedKVPool& pool = state.pool();
    fn::KvExpertExtents extents(pool, ops::QsaKvFormat::NVFP4G16, device.stream,
                                device.copy_stream);
    const std::int32_t per_extent = extents.slots_per_extent();
    const auto flex =
        static_cast<std::size_t>(extents.extents()) * static_cast<std::size_t>(per_extent);
    expect(extents.extents() == 2 && extents.first_page() == kLowPages, "test pool geometry");

    // One F3 layer, warm-started hottest first into every fixed and flex slot.
    const std::vector<ops::MoeExpertCodec> codecs(fn::kRoutedExperts, ops::MoeExpertCodec::F3G32);
    const fn::ExpertHostStore store(codecs, std::size_t{256} << 20);
    std::vector<std::uint64_t> counts(fn::kRoutedExperts);
    std::iota(counts.rbegin(), counts.rend(), 1); // expert e has count 512 - e
    fn::ExpertSlotPlan plan;
    plan.slots = {0, kFixed};
    fn::ExpertResidency residency(store, plan, extents.flex_addresses(), counts, device.stream);
    const std::span<const std::uint64_t> host = store.device_addresses();

    // Warm model: slot N - 1 - rank holds expert rank (the hottest), N = kFixed + flex.
    const std::size_t total = static_cast<std::size_t>(kFixed) + flex;
    Model model;
    model.owner.assign(flex, ops::kMoeCacheEmptySlot);
    model.expert_table.assign(host.begin(), host.end());
    model.slot_index.assign(fn::kRoutedExperts, ops::kMoeCacheNoSlot);
    model.lent.assign(2, false);
    const auto flex_addresses = extents.flex_addresses();
    const auto fixed_base = reinterpret_cast<std::uint64_t>(residency.cache().pools[1].slots.data);
    for (std::size_t rank = 0; rank < total; ++rank) {
        const std::size_t slot = total - 1 - rank;
        model.slot_index[rank] = static_cast<std::int32_t>(slot);
        if (slot >= static_cast<std::size_t>(kFixed)) {
            model.owner[slot - kFixed] = static_cast<std::int32_t>(rank);
            model.expert_table[rank]   = flex_addresses[slot - kFixed] | 1U;
        } else {
            model.expert_table[rank] = (fixed_base + slot * kF3Bytes) | 1U;
        }
    }

    const auto check = [&](const std::string& step) {
        CUDA_CHECK(cudaStreamSynchronize(device.stream));
        const ops::MoeExpertCache cache = residency.cache();
        const auto owners               = download<std::int32_t>(cache.pools[1].owner.data, total);
        expect(std::vector<std::int32_t>(owners.begin() + kFixed, owners.end()) == model.owner,
               step + ": flex owners");
        expect(download<std::uint64_t>(residency.expert_table().data, fn::kRoutedExperts) ==
                   model.expert_table,
               step + ": expert table");
        expect(download<std::int32_t>(cache.slot_index.data, fn::kRoutedExperts) ==
                   model.slot_index,
               step + ": slot index");
        std::uint32_t withheld = 0;
        for (std::int32_t x = 0; x < 2; ++x) {
            expect(extents.lent(x) == model.lent[static_cast<std::size_t>(x)],
                   step + ": extent " + std::to_string(x) + " state");
            if (!model.lent[static_cast<std::size_t>(x)]) { withheld += kExtentPages; }
        }
        expect(pool.withheld_pages() == withheld &&
                   pool.free_pages() + pool.mapped_pages() + withheld == kPages,
               step + ": page states partition the pool");
        // No mapped page lies in a withheld extent.
        for (std::int32_t lane = 0; lane < 2; ++lane) {
            if (!state.is_open(lane)) { continue; }
            for (const std::int32_t page : state.pages(lane).page_ids()) {
                const auto index = static_cast<std::uint32_t>(page);
                const bool held =
                    index >= kLowPages &&
                    !model.lent[static_cast<std::size_t>((index - kLowPages) / kExtentPages)];
                expect(!held, step + ": a lane maps a withheld page");
            }
        }
    };
    check("warm start");

    // Lane 0 needs 100 pages: the 40 low pages and extent 0.
    extents.make_room(100, residency);
    lend_model(model, 0, per_extent, host);
    const std::optional<std::int32_t> lane0 = state.open(100 * 64, device.stream);
    expect(lane0.has_value(), "lane 0 opens");
    check("lane 0 opened");
    extents.give_back(residency);
    check("give_back with lane 0 mapped");

    // Lane 1 needs 300 pages: 184 are free, so extent 1 is taken back too.
    extents.make_room(300, residency);
    lend_model(model, 1, per_extent, host);
    const std::optional<std::int32_t> lane1 = state.open(300 * 64, device.stream);
    expect(lane1.has_value(), "lane 1 opens");
    check("lane 1 opened");
    extents.make_room(10, residency);
    check("make_room with enough free pages");

    // Closing lane 1 frees extent 1 entirely, but lending it would leave 184 < 244 free pages.
    state.close(lane1.value(), device.stream);
    extents.give_back(residency);
    check("lane 1 closed, spare kept");

    // With every lane closed, extent 1 returns and extent 0 stays as the spare.
    state.close(lane0.value(), device.stream);
    extents.give_back(residency);
    return_model(model, 1, per_extent);
    check("every lane closed");

    // A request for the whole pool takes the returned, now empty, extent back as well.
    extents.make_room(kPages, residency);
    lend_model(model, 1, per_extent, host);
    check("every extent lent");
}

} // namespace

int main() {
    try {
        test_geometry(ops::QsaKvFormat::NVFP4G16, 244, 84, 3 * 244 + 17);
        test_geometry(ops::QsaKvFormat::BF16, 122, 108, 2 * 122 + 5);
        test_state_machine();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    std::cout << (failures == 0 ? "OK" : "FAIL") << " flash-next kv extents\n";
    return failures == 0 ? 0 : 1;
}
