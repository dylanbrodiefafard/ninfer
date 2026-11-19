// Qwen3.8-Flash-Next KV extents as expert-cache flex slots: the extent geometry over real QSA pools
// (NVFP4 244 pages and 84 records, BF16 122 pages and 108 records; addresses aligned, disjoint and
// inside their extent's run of one plane), and the CacheHeld / KvLent state machine against an
// independent model: pool page states, the lanes' mapped pages, and the cache's device state
// (borrowed and returned flex owners, evicted table entries and slot index) after make_room and
// give_back, including the one-extent spare that give_back keeps free.

#include "runtime/expert_residency.h"
#include "runtime/kv_extents.h"
#include "runtime/state.h"
#include "runtime/engine/idle_lane_policy.h"

#include "core/device.h"
#include "ninfer/ops/moe_expert_cache.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
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

void test_idle_reclaim_refill() {
    ninfer::DeviceContext device(0);
    constexpr std::uint32_t kExtent = 244;
    constexpr std::uint32_t kLow    = 40;
    constexpr std::uint32_t kPages  = kLow + 2 * kExtent;
    fn::State state(3, static_cast<std::int32_t>(kPages * 64), std::int64_t{kPages} * 64,
                    ops::QsaKvFormat::NVFP4G16);
    fn::KvExpertExtents extents(state.pool(), ops::QsaKvFormat::NVFP4G16, device.stream,
                                device.copy_stream);
    const std::vector<ops::MoeExpertCodec> codecs(fn::kRoutedExperts, ops::MoeExpertCodec::F3G32);
    fn::ExpertHostStore store(codecs, std::size_t{256} << 20);
    std::vector<std::byte> record(kF3Bytes, std::byte{0x5a});
    for (std::int32_t e = 0; e < extents.slots_per_extent(); ++e) {
        record[0] = static_cast<std::byte>(e);
        store.write_record(0, e, record);
    }
    fn::ExpertSlotPlan plan;
    plan.slots = {0, 4};
    fn::ExpertResidency cache(store, plan, extents.flex_addresses(), {}, device.stream);
    std::vector<std::int32_t> frequency(fn::kRoutedExperts);
    std::iota(frequency.rbegin(), frequency.rend(), 1);
    CUDA_CHECK(cudaMemcpyAsync(cache.cache().frequency.counts.data, frequency.data(),
                               frequency.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                               device.stream));
    extents.make_room(kPages, cache);
    const auto exempt = state.open(std::int64_t{kLow} * 64, device.stream).value();
    const auto first  = state.open(std::int64_t{kExtent} * 64, device.stream).value();
    const auto second = state.open(std::int64_t{kExtent} * 64, device.stream).value();
    expect(state.pool().free_pages() == 0 && extents.lent_extents() == 2,
           "reclaim test fills the pool to capacity");
    using Policy = ninfer::runtime::IdleLanePolicy;
    Policy idle(std::chrono::seconds(30));
    const auto start = Policy::Clock::time_point{};
    idle.idle(static_cast<std::uint32_t>(first), false, start);
    idle.idle(static_cast<std::uint32_t>(second), true, start + std::chrono::seconds(1));
    idle.idle(static_cast<std::uint32_t>(exempt), false, start + std::chrono::seconds(2));
    // RAM transport of complete originals is checked by the tier and Engine idle-timeout suites;
    // this boundary model checks the resulting page ownership and physical refill bytes.
    while (const auto expired = idle.expired(start + std::chrono::seconds(33))) {
        state.close(static_cast<std::int32_t>(expired->lane), device.stream);
        idle.activity(expired->lane);
        extents.give_back(cache);
    }
    expect(state.is_open(exempt) && !state.is_open(first) && !state.is_open(second),
           "timeout releases every nonexempt idle lane");
    expect(extents.lent_extents() == 1 && extents.lent(0) && !extents.lent(1),
           "timeout returns the top extent and keeps one spare");
    while (cache.service_refill(device.stream, device.copy_stream, true)) {
        device.synchronize_all();
    }
    CUDA_CHECK(cudaStreamSynchronize(device.stream));
    const auto pool   = cache.cache().pools[1];
    const auto owners = download<std::int32_t>(
        pool.owner.data, std::size_t{4} + static_cast<std::size_t>(cache.flex_slots()));
    const auto slots       = extents.slots_per_extent();
    std::int32_t populated = 0;
    for (std::int32_t j = 0; j < slots; ++j) {
        const auto flex_slot = slots + j;
        const auto owner     = owners[std::size_t{4} + static_cast<std::size_t>(flex_slot)];
        expect(owner == slots - 1 - j, "hottest refill belongs in the highest flex slot");
        if (owner >= 0) {
            ++populated;
            // NOLINTNEXTLINE(performance-no-int-to-ptr): the flex table stores device addresses.
            const auto* address = reinterpret_cast<const void*>(
                extents.flex_addresses()[static_cast<std::size_t>(flex_slot)]);
            const auto payload  = download<std::byte>(address, kF3Bytes);
            const auto original = store.record(0, owner);
            expect(std::equal(payload.begin(), payload.end(), original.begin(), original.end()),
                   "refill copies the immutable host record exactly");
        }
    }
    expect(populated == slots, "returned extent must refill every eligible flex slot");
    // Reclaim may arrive immediately after a background copy is issued; joining it is a stream
    // event dependency, and every following KV writer must observe that dependency.
    extents.make_room(kExtent + 1, cache);
    extents.give_back(cache);
    expect(cache.service_refill(device.stream, device.copy_stream, true),
           "returned empty flex slots start a background copy");
    extents.make_room(kExtent + 1, cache);
    const auto restored = state.open(std::int64_t{2} * kExtent * 64, device.stream).value();
    state.pool().zero_pages(state.pages(restored).page_ids(), device.stream);
    device.synchronize_all();
    // Servicing a stale unpublished reservation after KV reclaim would re-install its owner.
    // Together with the byte check this fails if make_room omits the refill join/publication.
    (void)cache.service_refill(device.stream, device.copy_stream, false);
    CUDA_CHECK(cudaStreamSynchronize(device.stream));
    // NOLINTNEXTLINE(performance-no-int-to-ptr): the flex table stores device addresses.
    const auto* reserved_address = reinterpret_cast<const void*>(extents.flex_addresses().back());
    const auto overwritten       = download<std::byte>(reserved_address, kF3Bytes);
    expect(std::ranges::all_of(overwritten, [](std::byte value) { return value == std::byte{0}; }),
           "refill must complete before reclaimed KV pages are zeroed");
    const auto reclaimed_owners = download<std::int32_t>(
        pool.owner.data, std::size_t{4} + static_cast<std::size_t>(cache.flex_slots()));
    expect(std::all_of(reclaimed_owners.begin() + 4, reclaimed_owners.end(),
                       [](std::int32_t owner) { return owner == ops::kMoeCacheBorrowedSlot; }),
           "refill publication must not re-install an owner in a KV-lent extent");
    expect(download<std::uint64_t>(cache.expert_table().data, 1).front() ==
               store.device_addresses().front(),
           "reclaim must leave the refilled expert on its host record");
    expect(state.pages(restored).page_ids().front() == static_cast<std::int32_t>(kLow),
           "a restored high lane relocates into the lowest free pages");
    state.close(restored, device.stream);
    state.close(exempt, device.stream);
    extents.give_back(cache);
    device.synchronize_all();
}

// Admission eligibility can fit entirely in low slots while high flex slots remain empty. A later
// KV borrow makes some of those records nonresident; no give_back occurs to request refill.
void test_borrow_restarts_refill() {
    ninfer::DeviceContext device(0);
    fn::State state(1, 64, std::int64_t{528} * 64, ops::QsaKvFormat::NVFP4G16);
    fn::KvExpertExtents extents(state.pool(), ops::QsaKvFormat::NVFP4G16, device.stream,
                                device.copy_stream);
    const std::vector<ops::MoeExpertCodec> codecs(fn::kRoutedExperts, ops::MoeExpertCodec::F3G32);
    const fn::ExpertHostStore store(codecs, std::size_t{256} << 20);
    fn::ExpertSlotPlan plan;
    plan.slots = {0, 4};
    std::array<std::uint32_t, ops::kMoeExpertMaskWords> eligibility{};
    eligibility[0] = (1U << 10) - 1U;
    // Records 100 to 103 are pinned in the four fixed slots; the tail is records 0 to 9.
    std::vector<std::int32_t> primary_slots(fn::kRoutedExperts, -1);
    std::iota(primary_slots.begin() + 100, primary_slots.begin() + 104, 0);
    fn::ExpertResidency cache(store, plan, extents.flex_addresses(), {}, device.stream,
                              {primary_slots, eligibility});
    std::array<std::int32_t, ops::kMoeTopK> ids{};
    std::iota(ids.begin(), ids.end(), 0);
    ninfer::DeviceBuffer routes_storage(sizeof(ids));
    routes_storage.copy_from_host(ids.data(), sizeof(ids));
    const ninfer::Tensor routes(routes_storage.p, ninfer::DType::I32, {ops::kMoeTopK, 1});
    auto table = cache.expert_table();
    auto fills = cache.fill_table();
    ops::moe_expert_cache_admit(routes, table, cache.host_table(), fills, cache.cache(), 0,
                                ops::MoeCacheAdmission::EveryMiss, device.stream);
    ops::moe_expert_cache_fill(cache.host_table(), fills, 0, device.stream);
    ops::moe_expert_cache_commit(table, fills, 0, device.stream);
    cache.request_refill();
    while (cache.service_refill(device.stream, device.copy_stream, true)) {
        device.synchronize_all();
    }
    const auto slots       = extents.slots_per_extent();
    const auto high_owners = [&] {
        const auto owners =
            download<std::int32_t>(cache.cache().pools[1].owner.data,
                                   std::size_t{4} + static_cast<std::size_t>(cache.flex_slots()));
        return std::count_if(owners.begin() + 4 + slots, owners.end(),
                             [](std::int32_t owner) { return owner >= 0; });
    };
    expect(high_owners() == 0, "all eligible records start resident below the empty high extent");
    extents.make_room(state.pool().free_pages() + 1, cache);
    expect(extents.lent_extents() == 1, "only the low extent is borrowed");
    while (cache.service_refill(device.stream, device.copy_stream, true)) {
        device.synchronize_all();
    }
    CUDA_CHECK(cudaStreamSynchronize(device.stream));
    expect(high_owners() == 6, "borrowing low owners must restart refill into empty high slots");
}

} // namespace

int main() {
    try {
        test_geometry(ops::QsaKvFormat::NVFP4G16, 244, 84, 3 * 244 + 17);
        test_geometry(ops::QsaKvFormat::BF16, 122, 108, 2 * 122 + 5);
        test_state_machine();
        test_idle_reclaim_refill();
        test_borrow_restarts_refill();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    std::cout << (failures == 0 ? "OK" : "FAIL") << " flash-next kv extents\n";
    return failures == 0 ? 0 : 1;
}
