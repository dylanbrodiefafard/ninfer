// Qwen3.8-Flash-Next expert residency: the slot-pool budget split (11% NVFP4) at 20 GB and 8 GB
// and at its caps; a one-layer pinned host store packed into several chunks
// (record contents, codec tags, alignment, disjointness, and device reads through the mapped
// addresses); the warm-started cache, with flex F3 slots at lent addresses after the fixed ones,
// against an independent selection of the hottest experts per codec (tables, owners, slot index,
// flex address table, reference bits, hands, slot contents, halving period); and one admit/commit
// step through the residency's own views.

#include "core/device.h"
#include "ninfer/ops/moe_expert_cache.h"
#include "runtime/expert_residency.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace fn  = ninfer::targets::qwen3_8_flash_next;
namespace ops = ninfer::ops;
using ninfer::DType;
using ninfer::Tensor;
using ops::MoeExpertCodec;

namespace {

int failures = 0;

void expect(bool condition, const std::string& what) {
    if (!condition) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

template <typename T>
std::vector<T> download(const void* device, std::size_t count) {
    std::vector<T> host(count);
    check(cudaMemcpy(host.data(), device, count * sizeof(T), cudaMemcpyDeviceToHost),
          "cudaMemcpy device-to-host");
    return host;
}

constexpr std::size_t kNvfp4Bytes = 2'764'800;
constexpr std::size_t kF3Bytes    = 1'996'800;

std::byte pattern(std::size_t entry, std::size_t offset) {
    return static_cast<std::byte>((entry * 131U + offset * 7U + (offset >> 9)) & 0xffU);
}

void test_plan() {
    const auto plan = [](std::size_t budget, std::int32_t nvfp4, std::int32_t f3) {
        return fn::plan_expert_slots(budget, {nvfp4, f3}).slots;
    };
    // 11% of the budget for NVFP4 slots (2.2 GB at 20 GB), F3 slots in the rest.
    expect(plan(20'000'000'000ULL, 1583, 22993) == std::array<std::int32_t, 2>{795, 8915},
           "20 GB plan");
    expect(plan(8'000'000'000ULL, 1583, 22993) == std::array<std::int32_t, 2>{318, 3566},
           "8 GB plan");
    // The F3 pool reaches its expert count; NVFP4 grows into the remaining budget.
    expect(plan(200'000'000ULL, 100, 10) == std::array<std::int32_t, 2>{65, 10}, "F3-capped plan");
    expect(plan(100'000'000'000ULL, 1583, 22993) == std::array<std::int32_t, 2>{1583, 22993},
           "plan above the store size");
    expect(plan(3'000'000ULL, 1583, 22993) == std::array<std::int32_t, 2>{0, 1}, "tiny plan");
    const fn::ExpertSlotPlan measured = fn::plan_expert_slots(20'000'000'000ULL, {1583, 22993});
    expect(measured.bytes() == 795 * kNvfp4Bytes + 8915 * kF3Bytes, "plan bytes");
    expect(measured.bytes() <= 20'000'000'000ULL, "plan within budget");
}

void test_store_and_residency() {
    // One layer: experts 5, 17, ... (every 16th from 5) and 300..309 are NVFP4, the rest F3.
    std::vector<MoeExpertCodec> codecs(fn::kRoutedExperts, MoeExpertCodec::F3G32);
    for (std::size_t e = 5; e < codecs.size(); e += 16) { codecs[e] = MoeExpertCodec::Nvfp4G16; }
    for (std::size_t e = 300; e < 310; ++e) { codecs[e] = MoeExpertCodec::Nvfp4G16; }
    const std::int32_t nvfp4_count = static_cast<std::int32_t>(
        std::count(codecs.begin(), codecs.end(), MoeExpertCodec::Nvfp4G16));
    constexpr std::size_t kChunkBytes = std::size_t{96} << 20;

    fn::ExpertHostStore store(codecs, kChunkBytes);
    expect(store.layers() == 1, "store layers");
    expect(store.codec_counts() ==
               std::array<std::int32_t, 2>{nvfp4_count, fn::kRoutedExperts - nvfp4_count},
           "codec counts");
    // Independent greedy chunk count.
    std::size_t chunks = 1;
    std::size_t used   = 0;
    for (const MoeExpertCodec codec : codecs) {
        const std::size_t bytes = codec == MoeExpertCodec::Nvfp4G16 ? kNvfp4Bytes : kF3Bytes;
        if (used + bytes > kChunkBytes) {
            ++chunks;
            used = 0;
        }
        used += bytes;
    }
    expect(store.chunk_count() == chunks && chunks > 1, "chunk count");

    std::vector<std::byte> record(kNvfp4Bytes);
    for (std::int32_t e = 0; e < fn::kRoutedExperts; ++e) {
        const std::size_t bytes = codecs[e] == MoeExpertCodec::Nvfp4G16 ? kNvfp4Bytes : kF3Bytes;
        for (std::size_t i = 0; i < bytes; ++i) { record[i] = pattern(e, i); }
        store.write_record(0, e, std::span<const std::byte>(record.data(), bytes));
    }

    const auto addresses = store.device_addresses();
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
    for (std::int32_t e = 0; e < fn::kRoutedExperts; ++e) {
        const std::uint64_t address = addresses[e];
        const std::size_t bytes = codecs[e] == MoeExpertCodec::Nvfp4G16 ? kNvfp4Bytes : kF3Bytes;
        expect((address & 1U) == static_cast<std::uint64_t>(codecs[e]), "address codec tag");
        expect(((address & ~std::uint64_t{1}) % 256) == 0, "record alignment");
        expect(store.codec(0, e) == codecs[e] && store.record(0, e).size() == bytes,
               "record codec and size");
        ranges.emplace_back(address & ~std::uint64_t{1}, (address & ~std::uint64_t{1}) + bytes);
        const auto stored = store.record(0, e);
        bool same         = true;
        for (std::size_t i = 0; i < bytes; i += 4099) { same = same && stored[i] == pattern(e, i); }
        expect(same && stored[bytes - 1] == pattern(e, bytes - 1), "record contents");
    }
    std::sort(ranges.begin(), ranges.end());
    for (std::size_t i = 1; i < ranges.size(); ++i) {
        expect(ranges[i - 1].second <= ranges[i].first, "records are disjoint");
    }

    ninfer::DeviceContext context;
    // The device reads mapped records through their device addresses.
    {
        ninfer::DeviceBuffer staging(kNvfp4Bytes);
        for (const std::int32_t e : {0, 5, 301, 511}) {
            const std::size_t bytes = store.record(0, e).size();
            // The test checks the stored integer address itself as a device pointer.
            const auto* source = reinterpret_cast<const void*>( // NOLINT(performance-no-int-to-ptr)
                addresses[e] & ~std::uint64_t{1});
            check(
                cudaMemcpyAsync(staging.p, source, bytes, cudaMemcpyDeviceToDevice, context.stream),
                "mapped record read");
            check(cudaStreamSynchronize(context.stream), "mapped record sync");
            const auto read = download<std::byte>(staging.p, bytes);
            expect(std::equal(read.begin(), read.end(), store.record(0, e).begin()),
                   "mapped device read of expert " + std::to_string(e));
        }
    }

    // Warm start: counts with ties and zeros; pools of 6 NVFP4 and 9 + 4 F3 slots, the flex
    // slots at descending record offsets of a separate buffer, so slot order differs from
    // address order and from the fixed array.
    std::vector<std::uint64_t> counts(fn::kRoutedExperts, 0);
    std::mt19937 rng(7);
    for (std::size_t e = 0; e < counts.size(); e += 3) { counts[e] = rng() % 50; }
    counts[301] = counts[302] = 1000; // a tie at the top of the NVFP4 pool
    fn::ExpertSlotPlan plan;
    plan.slots                  = {6, 9};
    constexpr std::size_t kFlex = 4;
    ninfer::DeviceBuffer flex_memory((kFlex + 1) * kF3Bytes);
    std::vector<std::uint64_t> flex_addresses;
    flex_addresses.reserve(kFlex);
    for (std::size_t j = 0; j < kFlex; ++j) {
        flex_addresses.push_back(reinterpret_cast<std::uint64_t>(flex_memory.p) +
                                 (kFlex - j) * kF3Bytes);
    }
    fn::ExpertResidency residency(store, plan, flex_addresses, counts, context.stream);
    expect(residency.flex_slots() == static_cast<std::int32_t>(kFlex), "flex slot count");
    const std::array<std::size_t, 2> total_slots{6, 9 + kFlex};

    std::array<std::vector<std::int32_t>, 2> expected_warm;
    for (std::size_t tag = 0; tag < 2; ++tag) {
        std::vector<std::int32_t> candidates;
        for (std::int32_t e = 0; e < fn::kRoutedExperts; ++e) {
            if (static_cast<std::size_t>(codecs[e]) == tag && counts[e] > 0) {
                candidates.push_back(e);
            }
        }
        std::sort(candidates.begin(), candidates.end(), [&](std::int32_t a, std::int32_t b) {
            return counts[a] != counts[b] ? counts[a] > counts[b] : a < b;
        });
        candidates.resize(std::min<std::size_t>(candidates.size(), total_slots[tag]));
        expected_warm[tag] = candidates;
    }
    expect(expected_warm[0].front() == 301 && expected_warm[0][1] == 302, "tie order");

    ops::MoeExpertCache cache = residency.cache();
    const Tensor expert_table = residency.expert_table();
    const Tensor host_table   = residency.host_table();
    const Tensor fill_table   = residency.fill_table();
    expect(expert_table.dtype == DType::I64 && expert_table.ne[0] == 512 && expert_table.ne[1] == 1,
           "expert table shape");
    expect(reinterpret_cast<std::uintptr_t>(expert_table.data) % 4096 == 0 &&
               reinterpret_cast<std::uintptr_t>(fill_table.data) % 4096 == 0,
           "table column alignment");
    std::vector<std::uint64_t> expected_table(addresses.begin(), addresses.end());
    std::vector<std::int32_t> expected_slot_index(512, ops::kMoeCacheNoSlot);
    for (std::size_t tag = 0; tag < 2; ++tag) {
        const ops::MoeExpertSlotPool& pool = cache.pools[tag];
        const auto fixed                   = static_cast<std::size_t>(plan.slots[tag]);
        const std::size_t slots            = total_slots[tag];
        const std::size_t bytes            = tag == 0 ? kNvfp4Bytes : kF3Bytes;
        expect(pool.record_bytes == bytes && pool.slots.bytes == fixed * bytes, "pool geometry");
        if (tag == 0) {
            expect(pool.flex_address.data == nullptr, "NVFP4 pool has no flex slots");
        } else {
            expect(pool.flex_address.dtype == DType::I64 &&
                       pool.flex_address.ne[0] == static_cast<std::int32_t>(kFlex) &&
                       download<std::uint64_t>(pool.flex_address.data, kFlex) == flex_addresses,
                   "flex address table");
        }
        expect(pool.owner.ne[0] == static_cast<std::int32_t>(slots) &&
                   pool.referenced.ne[0] == static_cast<std::int32_t>(slots),
               "owner and reference arrays cover fixed and flex slots");
        std::vector<std::int32_t> expected_owner(slots, ops::kMoeCacheEmptySlot);
        for (std::size_t rank = 0; rank < expected_warm[tag].size(); ++rank) {
            const std::int32_t e   = expected_warm[tag][rank];
            const std::size_t slot = slots - 1 - rank;
            expected_owner[slot]   = e;
            expected_slot_index[e] = static_cast<std::int32_t>(slot);
            const std::uint64_t address =
                slot < fixed ? reinterpret_cast<std::uint64_t>(pool.slots.data) + slot * bytes
                             : flex_addresses[slot - fixed];
            expected_table[e] = address | tag;
            // The test reads the stored integer address itself as a device pointer.
            // NOLINTNEXTLINE(performance-no-int-to-ptr): a device address as the tables hold it.
            const auto* const slot_bytes = reinterpret_cast<const void*>(address);
            const auto contents          = download<std::byte>(slot_bytes, bytes);
            expect(std::equal(contents.begin(), contents.end(), store.record(0, e).begin()),
                   "warm slot contents");
        }
        expect(download<std::int32_t>(pool.owner.data, slots) == expected_owner, "warm owners");
        const auto referenced = download<std::uint8_t>(pool.referenced.data, slots);
        expect(std::all_of(referenced.begin(), referenced.end(), [](auto v) { return v == 0; }),
               "warm reference bits");
    }
    expect(download<std::int32_t>(cache.clock_hands.data, 2) == std::vector<std::int32_t>{0, 0},
           "warm hands");
    expect(download<std::uint64_t>(expert_table.data, 512) == expected_table, "warm expert table");
    expect(cache.slot_index.dtype == DType::I32 && cache.slot_index.ne[0] == 512 &&
               cache.slot_index.ne[1] == 1,
           "slot index shape");
    expect(download<std::int32_t>(cache.slot_index.data, 512) == expected_slot_index,
           "warm slot index");
    expect(download<std::uint64_t>(host_table.data, 512) ==
               std::vector<std::uint64_t>(addresses.begin(), addresses.end()),
           "host table");
    const auto fills = download<std::uint64_t>(fill_table.data, 512);
    expect(std::all_of(fills.begin(), fills.end(), [](auto v) { return v == 0; }), "empty fills");
    expect(cache.frequency.halving_lookups ==
               fn::kFrequencyHalvingSlots * static_cast<std::int64_t>(6 + 9 + kFlex),
           "halving period over fixed and flex slots");

    // Three tiers over plan {1 NVFP4, 2 F3} fixed slots: NVFP4 301 and F3 0 and 1 are pinned, the
    // admission domain is the F3 records 2 and 7, and every other record (NVFP4 5, helper or
    // tail) keeps its mapped host address. Pinned slots are protected for good: owner -2, table
    // entry at the slot, slot index -1. Calibrated warm start fills only flex slots with eligible
    // records, never a fixed slot and never the hotter ineligible record 300.
    {
        std::vector<std::int32_t> primary_slots(512, -1);
        primary_slots[301] = 0;
        primary_slots[1]   = 0;
        primary_slots[0]   = 1;
        std::array<std::uint32_t, 16> admit{};
        admit[0] = (1U << 2) | (1U << 7);
        std::vector<std::uint64_t> tier_counts(512, 0);
        tier_counts[7]   = 9;
        tier_counts[300] = 1000;
        fn::ExpertResidency tiered(store, fn::ExpertSlotPlan{{1, 2}}, flex_addresses, tier_counts,
                                   context.stream, {primary_slots, admit});
        auto tier_cache    = tiered.cache();
        const auto index   = download<std::int32_t>(tier_cache.slot_index.data, 512);
        const auto table   = download<std::uint64_t>(tiered.expert_table().data, 512);
        const auto owner   = download<std::int32_t>(tier_cache.pools[1].owner.data, 2 + kFlex);
        const auto nv_own  = download<std::int32_t>(tier_cache.pools[0].owner.data, 1);
        const auto f3_base = reinterpret_cast<std::uint64_t>(tier_cache.pools[1].slots.data);
        const auto nv_base = reinterpret_cast<std::uint64_t>(tier_cache.pools[0].slots.data);
        expect(table[1] == (f3_base | 1U) && table[0] == ((f3_base + kF3Bytes) | 1U) &&
                   table[301] == nv_base,
               "pinned table entries name their fixed slots");
        expect(index[0] == -1 && index[1] == -1 && index[301] == -1, "pinned slot index stays -1");
        expect(owner[0] == ops::kMoeCacheBorrowedSlot && owner[1] == ops::kMoeCacheBorrowedSlot &&
                   nv_own[0] == ops::kMoeCacheBorrowedSlot,
               "pinned slots are protected");
        expect(owner[2 + kFlex - 1] == 7 && index[7] == static_cast<std::int32_t>(1 + kFlex) &&
                   table[7] == (flex_addresses[kFlex - 1] | 1U) && table[300] == addresses[300] &&
                   table[5] == addresses[5] && table[2] == addresses[2],
               "only the eligible record is warm, in the highest flex slot");
        for (const std::size_t e : {std::size_t{0}, std::size_t{1}}) {
            // NOLINTNEXTLINE(performance-no-int-to-ptr): a device address as the tables hold it.
            const auto* const slot = reinterpret_cast<const void*>(table[e] & ~std::uint64_t{1});
            const auto slot_bytes  = download<std::byte>(slot, kF3Bytes);
            expect(std::equal(slot_bytes.begin(), slot_bytes.end(), store.record(0, e).begin()),
                   "pinned slot contents");
        }
        expect(tier_cache.frequency.halving_lookups ==
                   fn::kFrequencyHalvingSlots * static_cast<std::int64_t>(kFlex),
               "halving period counts flex slots only");

        // Routing pinned, ineligible and one missing eligible record admits only the eligible one,
        // into an empty flex slot: CLOCK passes the protected fixed slots.
        ninfer::DeviceBuffer tier_ids(10 * sizeof(std::int32_t));
        const std::vector<std::int32_t> routed{2, 0, 1, 5, 301, 3, 2, 0, 1, 5};
        tier_ids.copy_from_host(routed.data(), tier_ids.bytes);
        ninfer::DeviceBuffer mask_storage(sizeof(admit));
        mask_storage.copy_from_host(admit.data(), mask_storage.bytes);
        const Tensor admit_view(mask_storage.p, DType::I32, {16});
        Tensor tier_expert = tiered.expert_table();
        Tensor tier_fill   = tiered.fill_table();
        ops::moe_expert_cache_admit(
            Tensor(tier_ids.p, DType::I32, {10, 1}), tier_expert, tiered.host_table(), tier_fill,
            tier_cache, 0, ops::MoeCacheAdmission::EveryMiss, context.stream, nullptr, &admit_view);
        ops::moe_expert_cache_commit(tier_expert, tier_fill, 0, context.stream);
        check(cudaStreamSynchronize(context.stream), "tiered admit");
        const auto after       = download<std::uint64_t>(tier_expert.data, 512);
        const auto after_owner = download<std::int32_t>(tier_cache.pools[1].owner.data, 2 + kFlex);
        expect(after[0] == table[0] && after[1] == table[1] && after[301] == table[301] &&
                   after[5] == addresses[5] && after[3] == addresses[3] &&
                   after[2] == (flex_addresses[0] | 1U),
               "admission leaves pinned and ineligible entries and admits record 2");
        expect(after_owner[0] == ops::kMoeCacheBorrowedSlot &&
                   after_owner[1] == ops::kMoeCacheBorrowedSlot && after_owner[2] == 2,
               "admission never takes a pinned slot");
    }

    // One admit/commit through the residency views: F3 expert 1 (count 0, not warm) takes F3 slot
    // 0, the coldest warm record (or an empty slot), whose owner returns to host memory; NVFP4
    // expert 301 (the hottest, slot 5) is a hit.
    ninfer::DeviceBuffer ids_storage(10 * sizeof(std::int32_t));
    std::vector<std::int32_t> ids{1, 301, 1, 301, 1, 301, 1, 301, 1, 301};
    ids_storage.copy_from_host(ids.data(), ids_storage.bytes);
    const Tensor ids_view(ids_storage.p, DType::I32, {10, 1});
    Tensor expert_view = residency.expert_table();
    Tensor fill_view   = residency.fill_table();
    ops::moe_expert_cache_admit(ids_view, expert_view, host_table, fill_view, cache, 0,
                                ops::MoeCacheAdmission::EveryMiss, context.stream);
    check(cudaStreamSynchronize(context.stream), "admit");
    const std::uint64_t first_f3_slot = reinterpret_cast<std::uint64_t>(cache.pools[1].slots.data);
    expect(download<std::uint64_t>(fill_view.data, 2)[1] == first_f3_slot, "admitted fill slot");
    ops::moe_expert_cache_commit(expert_view, fill_view, 0, context.stream);
    check(cudaStreamSynchronize(context.stream), "commit");
    const auto table = download<std::uint64_t>(expert_view.data, 512);
    expect(table[1] == (first_f3_slot | 1U), "committed table entry");
    if (expected_warm[1].size() == total_slots[1]) {
        const std::int32_t evicted = expected_warm[1].back();
        expect(table[evicted] == addresses[evicted], "evicted owner returns to host");
    }
    expect(download<std::uint8_t>(cache.pools[0].referenced.data, 6)[5] == 1, "hit reference bit");
}

} // namespace

int main() {
    try {
        test_plan();
        test_store_and_residency();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    std::cout << (failures == 0 ? "OK" : "FAIL") << " flash-next expert residency\n";
    return failures == 0 ? 0 : 1;
}
