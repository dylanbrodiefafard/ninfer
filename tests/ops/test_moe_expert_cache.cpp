// moe_expert_cache_admit / moe_expert_cache_commit / moe_expert_cache_borrow / _return against an
// independent host model of the contract's sequential policy (CLOCK bits or GCLOCK counters, with
// or without the frequency state and its admission filter, lent slots protected): exact equality
// of the expert, fill and host tables, the slot index (derived independently from the model's
// owners), slot owners, reference counters, hands, frequency state and full-set route counts after
// every admit, commit, borrow and return, over many random steps with skewed routing (hits,
// repeated experts across and within tokens), both codec pools, multi-layer interleaving, pools
// small enough that every
// slot is protected (exhaustion), pools wider than one 1024-slot scan window, an empty pool, pools
// with flex slots at scattered addresses (with and without fixed slots, lent in extents while a
// fixed run is lent as well), CLOCK ranges that start inside or past the fixed slots, and a
// captured multi-layer graph with a flex lend replayed with fresh routes. A pinned case (also the
// `--pinned` sanitizer case) puts the CLOCK range of each pool past fixed slots that hold pinned
// (owner -2) and cold unprotected records, so only flex slots are admittable: the hand starts
// near the end of the range and wraps to its first slot, admits exhaust the flex slots within one
// call, every flex extent is lent while misses arrive (nothing admitted, counters exact, hands
// unchanged), and no slot below the range ever changes owner. The lookup and miss counters are
// compared exactly after every admit that takes them (they accumulate from a random start; admits
// without them must leave them alone). After each exact comparison the model, and so the device
// state, is checked against the cache's consistency invariant; the host checks of the lend regions
// are exercised as well.

#include "core/decode_graph.h"
#include "core/device.h"
#include "ninfer/ops/moe_expert_cache.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kExperts = ops::kMoeExperts;
constexpr std::int32_t kTopK    = ops::kMoeTopK;
constexpr std::int32_t kPools   = ops::kMoeCacheCodecTags;
constexpr std::size_t kCounters = static_cast<std::size_t>(ops::kMoeCacheCounterKinds) * kPools;
// Synthetic mapped-host record addresses: never dereferenced by these Ops, far from any device
// allocation, 4 KiB apart.
constexpr std::uint64_t kHostBase = 0x7f00'0000'0000ULL;

// One pool as MoeExpertSlotPool defines it: `fixed` slots at base + s * record_bytes, then the
// flex slots at their listed addresses.
struct PoolModel {
    std::uint64_t base         = 0;
    std::uint64_t record_bytes = 0;
    std::int32_t fixed         = 0;
    // CLOCK visits only slots [clock_first, N).
    std::int32_t clock_first = 0;
    std::vector<std::uint64_t> flex;
    std::vector<std::int32_t> owner;
    std::vector<std::uint8_t> referenced;

    [[nodiscard]] std::uint64_t address(std::size_t slot) const {
        const auto first_flex = static_cast<std::size_t>(fixed);
        return slot < first_flex ? base + slot * record_bytes : flex.at(slot - first_flex);
    }
};

// The complete cache state as the contract defines it (tables are layer-major [512,L]).
struct CacheModel {
    std::int32_t layers = 0;
    std::vector<std::uint64_t> expert, host, fill, route_counts;
    std::array<PoolModel, kPools> pools;
    std::array<std::int32_t, kPools> hands{};
    // counters[kind + 2 codec], as the admit contract lays them out.
    std::array<std::int64_t, kCounters> counters{};
    std::int32_t reference_limit = 1;
    // Frequency state (empty counts: none), as MoeExpertFrequency defines it.
    std::vector<std::int32_t> counts;
    std::int32_t lookups         = 0;
    std::int32_t halving_lookups = 0;
};

// Index of a counter of codec tag in CacheModel::counters.
std::size_t counter_index(ops::MoeCacheCounter kind, std::size_t tag) {
    return static_cast<std::size_t>(kind) +
           static_cast<std::size_t>(ops::kMoeCacheCounterKinds) * tag;
}

// The contract's sequential policy, written from its text: hits raise the reference counter of
// the slot the expert owns up to the limit; the frequency state counts the routed set and halves
// at its period; misses in ascending id run CLOCK one slot at a time after checking that an
// unprotected slot of the pool's CLOCK range exists, the hand wrapping within that range, and
// with the filter a miss whose count does not exceed its victim
// owner's is not admitted. With count_events, every distinct routed expert adds one lookup of its
// codec and, when its entry names the host record on entry, one miss. Route counts include the
// full routed set; eligibility restricts only cache events, frequency and admission.
void model_admit(CacheModel& m, const std::vector<std::int32_t>& ids, std::int32_t layer,
                 bool count_events, bool filter, const std::vector<bool>* eligible = nullptr) {
    std::vector<bool> routed(kExperts, false);
    for (const std::int32_t id : ids) { routed[static_cast<std::size_t>(id)] = true; }
    for (std::int32_t e = 0; e < kExperts; ++e) {
        if (count_events && routed[e]) {
            ++m.route_counts[static_cast<std::size_t>(layer) * kExperts + e];
        }
        if (eligible != nullptr && !(*eligible)[e]) { routed[e] = false; }
    }
    if (count_events) {
        for (std::int32_t e = 0; e < kExperts; ++e) {
            const auto entry = static_cast<std::size_t>(layer) * kExperts + e;
            if (!routed[e]) { continue; }
            const std::size_t tag = m.host[entry] & 1U;
            ++m.counters[counter_index(ops::MoeCacheCounter::Lookups, tag)];
            if (m.expert[entry] == m.host[entry]) {
                ++m.counters[counter_index(ops::MoeCacheCounter::Misses, tag)];
            }
        }
    }
    const auto is_protected = [&](std::int32_t owner) {
        return owner == ops::kMoeCacheBorrowedSlot ||
               (owner >= 0 && owner / kExperts == layer && routed[owner % kExperts]);
    };
    for (std::int32_t e = 0; e < kExperts; ++e) {
        const auto entry = static_cast<std::size_t>(layer) * kExperts + e;
        if (!routed[e] || m.expert[entry] == m.host[entry]) { continue; }
        PoolModel& pool = m.pools[m.host[entry] & 1U];
        const auto it =
            std::find(pool.owner.begin(), pool.owner.end(), static_cast<std::int32_t>(entry));
        if (it == pool.owner.end()) {
            throw std::logic_error("model: resident expert without slot");
        }
        std::uint8_t& reference =
            pool.referenced[static_cast<std::size_t>(it - pool.owner.begin())];
        reference =
            static_cast<std::uint8_t>(std::min<std::int32_t>(reference + 1, m.reference_limit));
    }
    if (!m.counts.empty()) {
        std::int32_t routed_count = 0;
        for (std::int32_t e = 0; e < kExperts; ++e) {
            if (!routed[e]) { continue; }
            ++routed_count;
            std::int32_t& count = m.counts[static_cast<std::size_t>(layer) * kExperts + e];
            if (count < 0x7fffffff) { ++count; }
        }
        m.lookups += routed_count;
        if (m.lookups >= m.halving_lookups) {
            for (std::int32_t& count : m.counts) { count >>= 1; }
            m.lookups = 0;
        }
    }
    for (std::int32_t e = 0; e < kExperts; ++e) {
        const auto entry = static_cast<std::size_t>(layer) * kExperts + e;
        if (!routed[e] || m.expert[entry] != m.host[entry]) { continue; }
        const std::size_t tag    = m.host[entry] & 1U;
        PoolModel& pool          = m.pools[tag];
        const std::int32_t first = pool.clock_first;
        const auto range         = static_cast<std::int32_t>(pool.owner.size()) - first;
        if (std::none_of(pool.owner.begin() + first, pool.owner.end(),
                         [&](std::int32_t o) { return !is_protected(o); })) {
            continue;
        }
        std::int32_t& hand  = m.hands[tag];
        std::int32_t victim = -1;
        while (victim < 0) {
            const std::int32_t s = hand;
            hand                 = first + (hand - first + 1) % range;
            if (is_protected(pool.owner[s])) { continue; }
            if (pool.referenced[s] != 0) {
                --pool.referenced[s];
                continue;
            }
            victim = s;
        }
        const std::int32_t old = pool.owner[victim];
        if (filter && old >= 0 && m.counts[entry] <= m.counts[static_cast<std::size_t>(old)]) {
            continue;
        }
        if (old >= 0) { m.expert[old] = m.host[old]; }
        pool.owner[victim]      = static_cast<std::int32_t>(entry);
        pool.referenced[victim] = 1;
        m.fill[entry]           = pool.address(static_cast<std::size_t>(victim));
    }
}

// The slot index the contract defines: for every expert, the slot of its codec's pool that it
// owns, or -1. Derived from the owners alone, so the device table is checked against them.
std::vector<std::int32_t> model_slot_index(const CacheModel& m) {
    std::vector<std::int32_t> index(m.host.size(), ops::kMoeCacheNoSlot);
    for (std::size_t entry = 0; entry < m.host.size(); ++entry) {
        const PoolModel& pool = m.pools[m.host[entry] & 1U];
        const auto it =
            std::find(pool.owner.begin(), pool.owner.end(), static_cast<std::int32_t>(entry));
        if (it != pool.owner.end()) {
            index[entry] = static_cast<std::int32_t>(it - pool.owner.begin());
        }
    }
    return index;
}

// Lends `count` slots of a region from `first` (relative to the region's first slot).
void model_lend(CacheModel& m, std::int32_t pool_index, ops::MoeCacheSlotRegion region,
                std::int32_t first, std::int32_t count, bool borrow) {
    PoolModel& pool = m.pools[static_cast<std::size_t>(pool_index)];
    if (region == ops::MoeCacheSlotRegion::Flex) { first += pool.fixed; }
    for (std::int32_t s = first; s < first + count; ++s) {
        const std::int32_t owner = pool.owner[static_cast<std::size_t>(s)];
        if (borrow && owner >= 0) {
            m.expert[static_cast<std::size_t>(owner)] = m.host[static_cast<std::size_t>(owner)];
        }
        pool.owner[static_cast<std::size_t>(s)] =
            borrow ? ops::kMoeCacheBorrowedSlot : ops::kMoeCacheEmptySlot;
        pool.referenced[static_cast<std::size_t>(s)] = 0;
    }
}

void model_commit(CacheModel& m, std::int32_t layer) {
    for (std::int32_t e = 0; e < kExperts; ++e) {
        const auto entry = static_cast<std::size_t>(layer) * kExperts + e;
        if (m.fill[entry] != 0) {
            m.expert[entry] = m.fill[entry] | (m.expert[entry] & 1U);
            m.fill[entry]   = 0;
        }
    }
}

// A pool of `slots` fixed slots and `flex` flex slots (lent in extents of extent_slots), with the
// CLOCK range [clock_first, slots + flex).
struct PoolConfig {
    std::int32_t slots;
    std::uint64_t record_bytes;
    std::int32_t flex         = 0;
    std::int32_t extent_slots = 1;
    std::int32_t clock_first  = 0;
};

struct DevicePool {
    std::unique_ptr<DeviceBuffer> slots;
    std::unique_ptr<DeviceBuffer> flex_memory;
    std::unique_ptr<GuardedDeviceBuffer> flex_address;
    GuardedDeviceBuffer owner;
    GuardedDeviceBuffer referenced;
};

// Device copy of a CacheModel and the Op views over it.
class DeviceCache {
public:
    // Flex slots sit in one separate allocation at a random permutation of its record-sized
    // chunks, so their addresses follow neither slot order nor the fixed array.
    DeviceCache(CacheModel& model, const std::array<PoolConfig, kPools>& config, std::mt19937& rng)
        : expert_(table_bytes(model)), host_(table_bytes(model)), fill_(table_bytes(model)),
          slot_index_(table_bytes(model) / 2), hands_(kPools * sizeof(std::int32_t)),
          counters_(kCounters * sizeof(std::int64_t)), counts_(table_bytes(model) / 2),
          lookups_(sizeof(std::int32_t)), route_counts_(table_bytes(model)) {
        model.route_counts.resize(static_cast<std::size_t>(model.layers) * kExperts);
        for (std::int32_t p = 0; p < kPools; ++p) {
            const auto fixed  = static_cast<std::size_t>(config[p].slots);
            const auto flex   = static_cast<std::size_t>(config[p].flex);
            const auto slots  = static_cast<std::int32_t>(fixed + flex);
            const auto record = config[p].record_bytes;
            PoolModel& pool   = model.pools[p];
            pool.record_bytes = record;
            pool.fixed        = config[p].slots;
            pool.clock_first  = config[p].clock_first;
            pool.owner.assign(fixed + flex, ops::kMoeCacheEmptySlot);
            pool.referenced.assign(fixed + flex, 0);
            if (slots == 0) { continue; }
            pools_[p] = std::make_unique<DevicePool>(
                DevicePool{nullptr, nullptr, nullptr,
                           GuardedDeviceBuffer((fixed + flex) * sizeof(std::int32_t)),
                           GuardedDeviceBuffer(fixed + flex)});
            DevicePool& device           = *pools_[p];
            ops::MoeExpertSlotPool& view = cache_.pools[p];
            view.record_bytes            = record;
            view.clock_first             = config[p].clock_first;
            if (fixed != 0) {
                device.slots = std::make_unique<DeviceBuffer>(fixed * record);
                pool.base    = reinterpret_cast<std::uint64_t>(device.slots->p);
                view.slots   = {device.slots->p, device.slots->bytes};
            }
            if (flex != 0) {
                device.flex_memory = std::make_unique<DeviceBuffer>(flex * record);
                std::vector<std::size_t> chunk(flex);
                std::iota(chunk.begin(), chunk.end(), 0);
                std::shuffle(chunk.begin(), chunk.end(), rng);
                const auto memory = reinterpret_cast<std::uint64_t>(device.flex_memory->p);
                for (const std::size_t c : chunk) { pool.flex.push_back(memory + c * record); }
                device.flex_address =
                    std::make_unique<GuardedDeviceBuffer>(flex * sizeof(std::uint64_t));
                device.flex_address->copy_from_host(pool.flex.data(), device.flex_address->bytes());
                view.flex_address = Tensor(device.flex_address->data(), DType::I64,
                                           {static_cast<std::int32_t>(flex)});
            }
            view.owner      = Tensor(device.owner.data(), DType::I32, {slots});
            view.referenced = Tensor(device.referenced.data(), DType::U8, {slots});
        }
        cache_.slot_index      = Tensor(slot_index_.data(), DType::I32, {kExperts, model.layers});
        cache_.clock_hands     = Tensor(hands_.data(), DType::I32, {kPools});
        cache_.reference_limit = model.reference_limit;
        if (model.halving_lookups > 0) {
            cache_.frequency = ops::MoeExpertFrequency{
                .counts          = Tensor(counts_.data(), DType::I32, {kExperts, model.layers}),
                .lookups         = Tensor(lookups_.data(), DType::I32, {1}),
                .halving_lookups = model.halving_lookups,
            };
        }
        route_counts_view_ = Tensor(route_counts_.data(), DType::I64, {kExperts, model.layers});
        counters_view_ = Tensor(counters_.data(), DType::I64, {ops::kMoeCacheCounterKinds, kPools});
        expert_view_   = Tensor(expert_.data(), DType::I64, {kExperts, model.layers});
        host_view_     = Tensor(host_.data(), DType::I64, {kExperts, model.layers});
        fill_view_     = Tensor(fill_.data(), DType::I64, {kExperts, model.layers});
    }

    void upload(const CacheModel& m) {
        expert_.copy_from_host(m.expert.data(), expert_.bytes());
        host_.copy_from_host(m.host.data(), host_.bytes());
        fill_.copy_from_host(m.fill.data(), fill_.bytes());
        const std::vector<std::int32_t> slot_index = model_slot_index(m);
        slot_index_.copy_from_host(slot_index.data(), slot_index_.bytes());
        hands_.copy_from_host(m.hands.data(), hands_.bytes());
        counters_.copy_from_host(m.counters.data(), counters_.bytes());
        route_counts_.copy_from_host(m.route_counts.data(), route_counts_.bytes());
        if (!m.counts.empty()) {
            counts_.copy_from_host(m.counts.data(), counts_.bytes());
            lookups_.copy_from_host(&m.lookups, lookups_.bytes());
        }
        for (std::int32_t p = 0; p < kPools; ++p) {
            if (!pools_[p]) { continue; }
            pools_[p]->owner.copy_from_host(m.pools[p].owner.data(), pools_[p]->owner.bytes());
            pools_[p]->referenced.copy_from_host(m.pools[p].referenced.data(),
                                                 pools_[p]->referenced.bytes());
        }
    }

    // Exact comparison of every table, pool array and hand, plus guard checks.
    int compare(const CacheModel& m, const std::string& label) const {
        const std::size_t entries = m.expert.size();
        int failures = verify_exact((label + " expert_ptr").c_str(),
                                    from_device<std::uint64_t>(expert_.data(), entries), m.expert);
        failures += verify_exact((label + " host_ptr").c_str(),
                                 from_device<std::uint64_t>(host_.data(), entries), m.host);
        failures += verify_exact((label + " fill_ptr").c_str(),
                                 from_device<std::uint64_t>(fill_.data(), entries), m.fill);
        failures += verify_exact((label + " slot_index").c_str(),
                                 from_device<std::int32_t>(slot_index_.data(), entries),
                                 model_slot_index(m));
        failures += slot_index_.verify_guards(label + " slot_index");
        const auto hands = from_device<std::int32_t>(hands_.data(), kPools);
        failures += verify_exact((label + " clock_hands").c_str(), hands,
                                 std::vector<std::int32_t>(m.hands.begin(), m.hands.end()));
        failures += expert_.verify_guards(label + " expert_ptr");
        failures += fill_.verify_guards(label + " fill_ptr");
        failures += hands_.verify_guards(label + " clock_hands");
        failures += verify_exact((label + " counters").c_str(),
                                 from_device<std::int64_t>(counters_.data(), kCounters),
                                 std::vector<std::int64_t>(m.counters.begin(), m.counters.end()));
        failures += counters_.verify_guards(label + " counters");
        failures +=
            verify_exact((label + " route counts").c_str(),
                         from_device<std::uint64_t>(route_counts_.data(), entries), m.route_counts);
        failures += route_counts_.verify_guards(label + " route counts");
        if (!m.counts.empty()) {
            failures +=
                verify_exact((label + " frequency counts").c_str(),
                             from_device<std::int32_t>(counts_.data(), m.counts.size()), m.counts);
            failures += verify_exact((label + " frequency lookups").c_str(),
                                     from_device<std::int32_t>(lookups_.data(), 1),
                                     std::vector<std::int32_t>{m.lookups});
            failures += counts_.verify_guards(label + " frequency counts");
            failures += lookups_.verify_guards(label + " frequency lookups");
        }
        for (std::int32_t p = 0; p < kPools; ++p) {
            if (!pools_[p]) { continue; }
            const std::string pool = label + " pool " + std::to_string(p);
            const PoolModel& model = m.pools[p];
            failures +=
                verify_exact((pool + " owner").c_str(),
                             from_device<std::int32_t>(pools_[p]->owner.data(), model.owner.size()),
                             model.owner);
            failures += verify_exact(
                (pool + " referenced").c_str(),
                from_device<std::uint8_t>(pools_[p]->referenced.data(), model.referenced.size()),
                model.referenced);
            failures += pools_[p]->owner.verify_guards(pool + " owner");
            failures += pools_[p]->referenced.verify_guards(pool + " referenced");
            if (pools_[p]->flex_address) {
                failures += verify_exact(
                    (pool + " flex_address").c_str(),
                    from_device<std::uint64_t>(pools_[p]->flex_address->data(), model.flex.size()),
                    model.flex);
                failures += pools_[p]->flex_address->verify_guards(pool + " flex_address");
            }
        }
        return failures;
    }

    void admit(const Tensor& ids, std::int32_t layer, bool count_events, bool filter,
               cudaStream_t stream, const Tensor* eligible = nullptr) {
        ops::moe_expert_cache_admit(ids, expert_view_, host_view_, fill_view_, cache_, layer,
                                    filter ? ops::MoeCacheAdmission::FrequencyFilter
                                           : ops::MoeCacheAdmission::EveryMiss,
                                    stream, count_events ? &counters_view_ : nullptr, eligible,
                                    count_events ? &route_counts_view_ : nullptr);
    }

    void lend(std::int32_t pool, ops::MoeCacheSlotRegion region, std::int32_t first,
              std::int32_t count, bool borrow, cudaStream_t stream) {
        const auto codec = static_cast<ops::MoeExpertCodec>(pool);
        if (borrow) {
            ops::moe_expert_cache_borrow(expert_view_, host_view_, cache_, codec, region, first,
                                         count, stream);
        } else {
            ops::moe_expert_cache_return(cache_, codec, region, first, count, stream);
        }
    }

    void refill_select(Tensor& state, Tensor& source, Tensor& destination, cudaStream_t stream,
                       const Tensor* eligible) {
        ops::moe_expert_cache_refill_select(host_view_, cache_, state, source, destination, stream,
                                            eligible);
    }

    void refill_publish(Tensor& state, cudaStream_t stream) {
        ops::moe_expert_cache_refill_publish(expert_view_, host_view_, cache_, state, stream);
    }

    void commit(std::int32_t layer, cudaStream_t stream) {
        ops::moe_expert_cache_commit(expert_view_, fill_view_, layer, stream);
    }

private:
    static std::size_t table_bytes(const CacheModel& m) {
        return static_cast<std::size_t>(m.layers) * kExperts * sizeof(std::uint64_t);
    }

    GuardedDeviceBuffer expert_, host_, fill_, slot_index_, hands_, counters_, counts_, lookups_,
        route_counts_;
    std::array<std::unique_ptr<DevicePool>, kPools> pools_;
    ops::MoeExpertCache cache_;
    Tensor expert_view_, host_view_, fill_view_, counters_view_, route_counts_view_;
};

// Consistent random initial state: host addresses with random codecs (a fraction nvfp4_share of
// codec 0), a random fill of each pool by distinct experts of its codec, random reference bits and
// hands (in each CLOCK range; the first slot of an empty range). Requires DeviceCache to have set
// the pool bases, sizes and ranges.
void randomize_state(CacheModel& m, double nvfp4_share, double occupancy, double referenced,
                     std::mt19937& rng) {
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    const std::size_t entries = static_cast<std::size_t>(m.layers) * kExperts;
    m.host.resize(entries);
    std::array<std::vector<std::int32_t>, kPools> by_codec;
    for (std::size_t entry = 0; entry < entries; ++entry) {
        const std::uint64_t tag = unit(rng) < nvfp4_share ? 0 : 1;
        m.host[entry]           = (kHostBase + entry * 4096) | tag;
        by_codec[tag].push_back(static_cast<std::int32_t>(entry));
    }
    m.expert = m.host;
    m.fill.assign(entries, 0);
    for (std::int32_t p = 0; p < kPools; ++p) {
        PoolModel& pool = m.pools[p];
        std::shuffle(by_codec[p].begin(), by_codec[p].end(), rng);
        std::size_t next = 0;
        for (std::size_t s = 0; s < pool.owner.size(); ++s) {
            pool.referenced[s] =
                unit(rng) < referenced
                    ? static_cast<std::uint8_t>(
                          std::uniform_int_distribution<std::int32_t>(1, m.reference_limit)(rng))
                    : 0;
            if (unit(rng) < occupancy && next < by_codec[p].size()) {
                const std::int32_t code = by_codec[p][next++];
                pool.owner[s]           = code;
                m.expert[static_cast<std::size_t>(code)] =
                    pool.address(s) | static_cast<std::uint64_t>(p);
            }
        }
        const auto slots = static_cast<std::int32_t>(pool.owner.size());
        m.hands[p] =
            pool.clock_first == slots
                ? pool.clock_first
                : std::uniform_int_distribution<std::int32_t>(pool.clock_first, slots - 1)(rng);
    }
    // Counters start at random nonzero values, the first just below 2^32, so the comparison sees
    // accumulation rather than assignment and carries past 32 bits.
    for (std::int64_t& counter : m.counters) {
        counter = std::uniform_int_distribution<std::int64_t>(0, std::int64_t{1} << 40)(rng);
    }
    m.counters[0] = (std::int64_t{1} << 32) - 3;
    if (m.halving_lookups > 0) {
        // Small counts, so filter decisions go both ways and ties occur.
        m.counts.resize(entries);
        for (std::int32_t& count : m.counts) {
            count = std::uniform_int_distribution<std::int32_t>(0, 6)(rng);
        }
        m.lookups = std::uniform_int_distribution<std::int32_t>(0, m.halving_lookups - 1)(rng);
    }
}

// Skewed routes: expert ranks drawn as 512 u^3 through a per-layer permutation, so a few experts
// dominate (hits) and tokens repeat experts; every 7th token repeats one id within the token.
std::vector<std::int32_t> random_ids(std::int32_t tokens, std::int32_t layer, std::mt19937& rng) {
    std::vector<std::int32_t> permutation(kExperts);
    std::iota(permutation.begin(), permutation.end(), 0);
    std::mt19937 layer_rng(1000U + static_cast<std::uint32_t>(layer));
    std::shuffle(permutation.begin(), permutation.end(), layer_rng);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::vector<std::int32_t> ids(static_cast<std::size_t>(tokens) * kTopK);
    for (std::int32_t t = 0; t < tokens; ++t) {
        for (std::int32_t k = 0; k < kTopK; ++k) {
            const double u  = unit(rng);
            const auto rank = static_cast<std::int32_t>(std::floor(kExperts * u * u * u));
            ids[static_cast<std::size_t>(t) * kTopK + k] =
                permutation[std::min(rank, kExperts - 1)];
        }
        if (t % 7 == 3) {
            ids[static_cast<std::size_t>(t) * kTopK + 9] = ids[static_cast<std::size_t>(t) * kTopK];
        }
    }
    return ids;
}

// The cache's consistency invariant after a commit, checked on the model (equal to the device
// state once an exact comparison passed): every owned slot is named by its owner's table entry
// (with the codec tag), every non-host entry names a slot its expert owns, and fills are empty.
int check_consistency(const CacheModel& m, const std::string& label) {
    std::vector<std::int32_t> owned(m.expert.size(), 0);
    for (std::int32_t p = 0; p < kPools; ++p) {
        const PoolModel& pool = m.pools[p];
        for (std::size_t s = 0; s < pool.owner.size(); ++s) {
            const std::int32_t o = pool.owner[s];
            if (o < 0) { continue; }
            const std::uint64_t address = pool.address(s) | static_cast<std::uint64_t>(p);
            if (++owned[static_cast<std::size_t>(o)] != 1 ||
                m.expert[static_cast<std::size_t>(o)] != address ||
                (m.host[static_cast<std::size_t>(o)] & 1U) != static_cast<std::uint64_t>(p)) {
                std::cerr << label << ": slot " << s << " of pool " << p << " is inconsistent\n";
                return 1;
            }
        }
    }
    for (std::size_t entry = 0; entry < m.expert.size(); ++entry) {
        if (m.fill[entry] != 0 || (m.expert[entry] != m.host[entry] && owned[entry] != 1)) {
            std::cerr << label << ": table entry " << entry << " is inconsistent\n";
            return 1;
        }
    }
    return 0;
}

struct Scenario {
    const char* name;
    std::int32_t layers;
    std::array<PoolConfig, kPools> pools;
    double nvfp4_share;
    double occupancy;
    double referenced;
    std::int32_t rounds;
    // GCLOCK cap, the frequency period (0: no frequency state; then every admit admits every
    // miss) and whether slots are lent out and returned between layers: a fixed run of pool 1 and
    // the flex extents of both pools.
    std::int32_t reference_limit = 1;
    std::int32_t halving_lookups = 0;
    bool lend                    = false;
    // Admits take a random eligibility bitset (about 70% eligible) on two of every three steps.
    bool eligibility = false;
};

// The lending state of one eager run: a fixed run of pool 1 (the window's region) and which flex
// extents of each pool are lent. step() toggles one of them, chosen at random.
class Lender {
public:
    explicit Lender(const std::array<PoolConfig, kPools>& pools) : pools_(pools) {
        for (std::int32_t p = 0; p < kPools; ++p) {
            extents_[p].assign(static_cast<std::size_t>(pools[p].flex / pools[p].extent_slots),
                               false);
        }
    }

    [[nodiscard]] bool can_lend() const {
        return pools_[1].slots > 0 || !extents_[0].empty() || !extents_[1].empty();
    }

    // Borrows or returns one run on the device and in the model; returns the step's label.
    std::string step(DeviceCache& device, CacheModel& model, std::mt19937& rng,
                     cudaStream_t stream) {
        std::uniform_real_distribution<double> unit(0.0, 1.0);
        const bool has_flex = !extents_[0].empty() || !extents_[1].empty();
        if (has_flex && (pools_[1].slots == 0 || unit(rng) < 0.6)) {
            std::int32_t pool = extents_[0].empty() ? 1 : 0;
            if (!extents_[0].empty() && !extents_[1].empty()) { pool = unit(rng) < 0.3 ? 0 : 1; }
            std::vector<bool>& lent = extents_[pool];
            const auto extent = std::uniform_int_distribution<std::size_t>(0, lent.size() - 1)(rng);
            const std::int32_t slots = pools_[pool].extent_slots;
            const auto first         = static_cast<std::int32_t>(extent) * slots;
            const bool borrow        = !lent[extent];
            device.lend(pool, ops::MoeCacheSlotRegion::Flex, first, slots, borrow, stream);
            model_lend(model, pool, ops::MoeCacheSlotRegion::Flex, first, slots, borrow);
            lent[extent] = borrow;
            return std::string(borrow ? " borrow" : " return") + " flex extent " +
                   std::to_string(extent) + " of pool " + std::to_string(pool);
        }
        const std::int32_t fixed = pools_[1].slots;
        const bool borrow        = fixed_count_ == 0;
        if (borrow) {
            fixed_count_ =
                std::uniform_int_distribution<std::int32_t>(1, std::max(1, fixed / 3))(rng);
            fixed_first_ =
                std::uniform_int_distribution<std::int32_t>(0, fixed - fixed_count_)(rng);
        }
        device.lend(1, ops::MoeCacheSlotRegion::Fixed, fixed_first_, fixed_count_, borrow, stream);
        model_lend(model, 1, ops::MoeCacheSlotRegion::Fixed, fixed_first_, fixed_count_, borrow);
        if (!borrow) { fixed_count_ = 0; }
        return borrow ? " borrow fixed" : " return fixed";
    }

private:
    std::array<PoolConfig, kPools> pools_;
    std::array<std::vector<bool>, kPools> extents_;
    // The lent fixed run [fixed_first_, fixed_first_ + fixed_count_), empty when the count is 0.
    std::int32_t fixed_first_ = 0;
    std::int32_t fixed_count_ = 0;
};

// Eager steps: every round visits each layer in order with a random T, comparing the complete
// state after each admit, commit, borrow and return. Admits take the counters on two of every
// three steps and, with frequency state, filter on half of them. With lending, a fixed run of
// pool 1 or a flex extent is borrowed or returned before a quarter of the layer calls, so admits
// meet lent slots of both regions at once.
int run_eager(const Scenario& scenario, std::uint32_t seed, cudaStream_t stream) {
    std::mt19937 rng(seed);
    CacheModel model;
    model.layers          = scenario.layers;
    model.reference_limit = scenario.reference_limit;
    model.halving_lookups = scenario.halving_lookups;
    DeviceCache device(model, scenario.pools, rng);
    randomize_state(model, scenario.nvfp4_share, scenario.occupancy, scenario.referenced, rng);
    device.upload(model);
    int failures = check_consistency(model, std::string(scenario.name) + " initial");
    constexpr std::array<std::int32_t, 8> kTokens{1, 1, 2, 3, 4, 8, 16, 64};
    std::int32_t step = 0;
    Lender lender(scenario.pools);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    for (std::int32_t round = 0; round < scenario.rounds && failures == 0; ++round) {
        for (std::int32_t layer = 0; layer < scenario.layers && failures == 0; ++layer) {
            if (scenario.lend && lender.can_lend() && unit(rng) < 0.25) {
                const std::string label =
                    std::string(scenario.name) + " round " + std::to_string(round) + " layer " +
                    std::to_string(layer) + lender.step(device, model, rng, stream);
                cuda_synchronize(stream);
                failures += device.compare(model, label);
                failures += check_consistency(model, label);
            }
            const std::int32_t tokens =
                kTokens[std::uniform_int_distribution<std::size_t>(0, kTokens.size() - 1)(rng)];
            auto ids = random_ids(tokens, layer, rng);
            if (scenario.eligibility) {
                // Every restricted round routes both an eligible and an ineligible expert.
                ids[0] = 0;
                ids[1] = 1;
            }
            GuardedDeviceBuffer ids_storage(ids.size() * sizeof(std::int32_t));
            ids_storage.copy_from_host(ids.data(), ids_storage.bytes());
            const Tensor ids_view(ids_storage.data(), DType::I32, {kTopK, tokens});
            const std::string label = std::string(scenario.name) + " round " +
                                      std::to_string(round) + " layer " + std::to_string(layer) +
                                      " T=" + std::to_string(tokens);
            const bool count_events = step++ % 3 != 2;
            const bool filter       = !model.counts.empty() && unit(rng) < 0.5;
            const bool restricted   = scenario.eligibility && step % 3 != 0;
            std::vector<bool> eligible(kExperts, true);
            std::vector<std::uint32_t> words(ops::kMoeExpertMaskWords, 0);
            std::unique_ptr<GuardedDeviceBuffer> eligible_storage;
            std::optional<Tensor> eligible_view;
            if (restricted) {
                for (std::size_t e = 0; e < eligible.size(); ++e) {
                    eligible[e] = unit(rng) < 0.7;
                    if (eligible[e]) { words[e / 32] |= 1U << (e % 32); }
                }
                eligible[0] = true;
                eligible[1] = false;
                words[0]    = (words[0] | 1U) & ~2U;
                eligible_storage =
                    std::make_unique<GuardedDeviceBuffer>(words.size() * sizeof(std::uint32_t));
                eligible_storage->copy_from_host(words.data(), eligible_storage->bytes());
                eligible_view =
                    Tensor(eligible_storage->data(), DType::I32, {ops::kMoeExpertMaskWords});
            }
            device.admit(ids_view, layer, count_events, filter, stream,
                         eligible_view ? &*eligible_view : nullptr);
            cuda_synchronize(stream);
            model_admit(model, ids, layer, count_events, filter, restricted ? &eligible : nullptr);
            failures += device.compare(model, label + " admit");
            device.commit(layer, stream);
            cuda_synchronize(stream);
            model_commit(model, layer);
            failures += device.compare(model, label + " commit");
            failures += check_consistency(model, label);
            failures +=
                verify_exact((label + " ids").c_str(),
                             from_device<std::int32_t>(ids_storage.data(), ids.size()), ids);
        }
    }
    return failures;
}

// One captured graph holds admit (with the counters) and commit for every layer at a fixed T, with
// a flex extent of pool 1 borrowed after layer 1 and returned after the last layer, while another
// extent stays lent from before the capture; each replay uploads fresh routes into the captured
// ids buffers and is compared against the model.
int run_graph(std::uint32_t seed, cudaStream_t stream) {
    constexpr std::int32_t kLayers = 4;
    constexpr std::int32_t kTokens = 4;
    constexpr std::int32_t kExtent = 36;
    constexpr auto kFlex           = ops::MoeCacheSlotRegion::Flex;
    // GCLOCK counters and the frequency filter, so the captured kernels carry the full policy.
    const Scenario scenario{
        "graph", kLayers, {{{96, 512, 24, 12}, {1500, 256, 3 * kExtent, kExtent}}},
        0.1,     0.6,     0.5,
        0,       4,       300};
    std::mt19937 rng(seed);
    CacheModel model;
    model.layers          = kLayers;
    model.reference_limit = scenario.reference_limit;
    model.halving_lookups = scenario.halving_lookups;
    DeviceCache device(model, scenario.pools, rng);
    randomize_state(model, scenario.nvfp4_share, scenario.occupancy, scenario.referenced, rng);
    // Pool 1's hand starts just before its flex slots, so the replays admit into them.
    model.hands[1] = model.pools[1].fixed - 8;
    device.upload(model);
    device.lend(1, kFlex, 2 * kExtent, kExtent, true, stream);
    const std::vector<std::uint64_t>& flex = model.pools[1].flex;
    std::int32_t flex_fills                = 0;
    model_lend(model, 1, kFlex, 2 * kExtent, kExtent, true);
    std::vector<GuardedDeviceBuffer> ids_storage;
    std::vector<Tensor> ids_views;
    ids_storage.reserve(kLayers);
    ids_views.reserve(kLayers);
    for (std::int32_t layer = 0; layer < kLayers; ++layer) {
        ids_storage.emplace_back(static_cast<std::size_t>(kTokens) * kTopK * sizeof(std::int32_t));
    }
    for (std::int32_t layer = 0; layer < kLayers; ++layer) {
        ids_views.emplace_back(ids_storage[layer].data(), DType::I32,
                               std::initializer_list<std::int32_t>{kTopK, kTokens});
    }
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    definition.capture(stream, [&] {
        for (std::int32_t layer = 0; layer < kLayers; ++layer) {
            device.admit(ids_views[layer], layer, true, true, stream);
            device.commit(layer, stream);
            if (layer == 1) { device.lend(1, kFlex, 0, kExtent, true, stream); }
        }
        device.lend(1, kFlex, 0, kExtent, false, stream);
    });
    graph.instantiate(definition);
    int failures = 0;
    for (std::int32_t replay = 0; replay < 40 && failures == 0; ++replay) {
        std::vector<std::vector<std::int32_t>> routes;
        for (std::int32_t layer = 0; layer < kLayers; ++layer) {
            routes.push_back(random_ids(kTokens, layer, rng));
            ids_storage[layer].copy_from_host(routes.back().data(), ids_storage[layer].bytes());
        }
        graph.launch(stream);
        cuda_synchronize(stream);
        for (std::int32_t layer = 0; layer < kLayers; ++layer) {
            model_admit(model, routes[layer], layer, true, true);
            flex_fills += static_cast<std::int32_t>(
                std::count_if(model.fill.begin(), model.fill.end(), [&](std::uint64_t address) {
                    return std::find(flex.begin(), flex.end(), address) != flex.end();
                }));
            model_commit(model, layer);
            if (layer == 1) { model_lend(model, 1, kFlex, 0, kExtent, true); }
        }
        model_lend(model, 1, kFlex, 0, kExtent, false);
        const std::string label = "graph replay " + std::to_string(replay);
        failures += device.compare(model, label);
        failures += check_consistency(model, label);
    }
    if (failures == 0 && flex_fills == 0) {
        std::cerr << "graph: no replay admitted into a flex slot\n";
        ++failures;
    }
    return failures;
}

// The host checks of the lend regions: a run must lie inside the region it names, so a window
// cannot reach flex slots and an extent cannot reach fixed ones.
int run_region_checks(std::uint32_t seed, cudaStream_t stream) {
    std::mt19937 rng(seed);
    CacheModel model;
    model.layers = 1;
    DeviceCache device(model, {{{4, 256}, {8, 256, 6, 3}}}, rng);
    randomize_state(model, 0.5, 0.5, 0.5, rng);
    device.upload(model);
    int failures       = 0;
    const auto refused = [&](const char* label, std::int32_t pool, ops::MoeCacheSlotRegion region,
                             std::int32_t first, std::int32_t count, bool borrow) {
        try {
            device.lend(pool, region, first, count, borrow, stream);
        } catch (const std::invalid_argument&) { return; }
        std::cerr << "region check: " << label << " was accepted\n";
        ++failures;
    };
    constexpr auto kFixed = ops::MoeCacheSlotRegion::Fixed;
    constexpr auto kFlex  = ops::MoeCacheSlotRegion::Flex;
    refused("fixed run past the fixed slots", 1, kFixed, 6, 3, true);
    refused("flex run past the flex slots", 1, kFlex, 4, 3, true);
    refused("flex run of a pool without flex slots", 0, kFlex, 0, 1, true);
    refused("empty return", 1, kFlex, 0, 0, false);
    refused("negative first", 1, kFixed, -1, 2, false);
    // The last slots of each region are accepted.
    device.lend(1, kFixed, 5, 3, true, stream);
    model_lend(model, 1, kFixed, 5, 3, true);
    device.lend(1, kFlex, 3, 3, true, stream);
    model_lend(model, 1, kFlex, 3, 3, true);
    cuda_synchronize(stream);
    failures += device.compare(model, "region check");
    failures += check_consistency(model, "region check");
    return failures;
}

// Independent exact oracle: enumerate candidates, sort by the stated frequency/code ordering,
// and enumerate empty flex slots in reverse order. Synthetic addresses are never dereferenced;
// the extent integration test checks actual host-record copies and reclaim fences.
int run_refill(cudaStream_t stream) {
    std::mt19937 rng(391);
    int failures = 0;
    for (const auto layers : {1, 48}) {
        CacheModel model;
        model.layers          = layers;
        model.halving_lookups = 50000;
        DeviceCache device(model, {{{5, 256}, {7, 256, 84, 84}}}, rng);
        randomize_state(model, 0.2, 0.4, 0.5, rng);
        std::vector<std::uint32_t> mask(static_cast<std::size_t>(layers) * 16, 0x55555555U);
        GuardedDeviceBuffer mask_buffer(mask.size() * sizeof(std::uint32_t));
        mask_buffer.copy_from_host(mask.data(), mask_buffer.bytes());
        const Tensor eligible(mask_buffer.data(), DType::I32, {16, layers});
        GuardedDeviceBuffer state_buffer(2 * sizeof(std::int32_t));
        GuardedDeviceBuffer source_buffer(kExperts * sizeof(std::uint64_t));
        GuardedDeviceBuffer destination_buffer(kExperts * sizeof(std::uint64_t));
        Tensor state(state_buffer.data(), DType::I32, {2});
        Tensor source(source_buffer.data(), DType::I64, {kExperts});
        Tensor destination(destination_buffer.data(), DType::I64, {kExperts});
        // Borrowed high slots must not be selected; equal frequencies exercise exact tie breaks.
        model_lend(model, 1, ops::MoeCacheSlotRegion::Flex, 83, 1, true);
        for (int step = 0; step < 95; ++step) {
            device.upload(model);
            const auto index = model_slot_index(model);
            std::vector<int> candidates;
            for (std::size_t code = 0; code < model.host.size(); ++code) {
                if ((model.host[code] & 1U) != 0 && index[code] < 0 &&
                    (mask[code / 32] & (1U << (code % 32))) != 0) {
                    candidates.push_back(static_cast<int>(code));
                }
            }
            std::sort(candidates.begin(), candidates.end(), [&](int a, int b) {
                return model.counts[a] != model.counts[b] ? model.counts[a] > model.counts[b]
                                                          : a < b;
            });
            int slot = -1;
            for (int s = static_cast<int>(model.pools[1].owner.size()) - 1;
                 s >= model.pools[1].fixed; --s) {
                if (model.pools[1].owner[s] == ops::kMoeCacheEmptySlot) {
                    slot = s;
                    break;
                }
            }
            const int code = slot >= 0 && !candidates.empty() ? candidates.front() : -1;
            if (code < 0) { slot = -1; }
            device.refill_select(state, source, destination, stream, &eligible);
            cuda_synchronize(stream);
            failures += verify_exact("refill selection", from_device<int>(state.data, 2),
                                     std::vector<int>{code, slot});
            std::vector<std::uint64_t> expected_source(kExperts, 0),
                expected_destination(kExperts, 0);
            if (code >= 0) {
                expected_source[0]         = model.host[code];
                expected_destination[0]    = model.pools[1].address(static_cast<std::size_t>(slot));
                model.pools[1].owner[slot] = ops::kMoeCacheBorrowedSlot;
                model.pools[1].referenced[slot] = 0;
            }
            failures +=
                verify_exact("refill source", from_device<std::uint64_t>(source.data, kExperts),
                             expected_source);
            failures += verify_exact("refill destination",
                                     from_device<std::uint64_t>(destination.data, kExperts),
                                     expected_destination);
            failures += device.compare(model, "unpublished refill");
            if (code >= 0 && step % 3 == 0) {
                // Demand can install the same candidate elsewhere while its refill slot is
                // protected. Publication must discard the duplicate, not create two owners.
                const std::vector<std::int32_t> ids(kTopK, code % kExperts);
                GuardedDeviceBuffer routed(ids.size() * sizeof(std::int32_t));
                routed.copy_from_host(ids.data(), routed.bytes());
                const Tensor routes(routed.data(), DType::I32, {kTopK, 1});
                device.admit(routes, code / kExperts, false, false, stream);
                model_admit(model, ids, code / kExperts, false, false);
                device.commit(code / kExperts, stream);
                model_commit(model, code / kExperts);
                cuda_synchronize(stream);
            }
            if (code >= 0) {
                if (model_slot_index(model)[code] < 0) {
                    model.pools[1].owner[slot]      = code;
                    model.pools[1].referenced[slot] = 1;
                    model.expert[code] =
                        model.pools[1].address(static_cast<std::size_t>(slot)) | 1U;
                } else {
                    model.pools[1].owner[slot]      = ops::kMoeCacheEmptySlot;
                    model.pools[1].referenced[slot] = 0;
                }
            }
            device.refill_publish(state, stream);
            cuda_synchronize(stream);
            failures += device.compare(model, "published refill");
            failures += verify_exact("cleared refill reservation", from_device<int>(state.data, 2),
                                     std::vector<int>{-1, -1});
            failures += state_buffer.verify_guards("refill state");
            failures += source_buffer.verify_guards("refill source");
            failures += destination_buffer.verify_guards("refill destination");
            if (step == 90) {
                // Every record excluded: even an empty slot must stay empty.
                std::fill(mask.begin(), mask.end(), 0);
                mask_buffer.copy_from_host(mask.data(), mask_buffer.bytes());
            }
        }
    }
    return failures;
}

// The expert-helper residency shape: each pool's CLOCK range starts past its fixed slots, which
// hold pinned slots (owner -2) and a few cold, unreferenced records that a whole-pool CLOCK would
// evict first. Pool 0 has no flex slots (an empty range: nothing is admitted); pool 1 has two flex
// extents behind more than one scan window of fixed slots. Phases, each admit compared exactly
// against the model after admit and commit: (1) the hand starts three slots before the end and
// must wrap to the range's first slot; (2) with both extents lent every miss stays mapped, the
// counters still count it and the hands do not move; (3) with one extent back, wide calls have
// more misses than the extent has slots, so CLOCK exhausts the range within the call and stops;
// (4) both extents back. No slot below a range ever changes owner, on the device or in the model.
int run_pinned(std::uint32_t seed, cudaStream_t stream) {
    constexpr std::int32_t kLayers = 3;
    constexpr std::int32_t kExtent = 36;
    constexpr auto kFlex           = ops::MoeCacheSlotRegion::Flex;
    const std::array<PoolConfig, kPools> pools{
        {{40, 512, 0, 1, 40}, {1100, 256, 2 * kExtent, kExtent, 1100}}};
    std::mt19937 rng(seed);
    CacheModel model;
    model.layers          = kLayers;
    model.reference_limit = 8;
    model.halving_lookups = 2000;
    DeviceCache device(model, pools, rng);
    randomize_state(model, 0.2, 0.8, 0.5, rng);
    // Pin nine of every ten fixed slots; the rest keep their record with a cleared counter.
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    for (PoolModel& pool : model.pools) {
        for (std::int32_t s = 0; s < pool.clock_first; ++s) {
            const auto slot      = static_cast<std::size_t>(s);
            const std::int32_t o = pool.owner[slot];
            if (unit(rng) < 0.9) {
                if (o >= 0) {
                    model.expert[static_cast<std::size_t>(o)] =
                        model.host[static_cast<std::size_t>(o)];
                }
                pool.owner[slot] = ops::kMoeCacheBorrowedSlot;
            }
            pool.referenced[slot] = 0;
        }
    }
    model.hands[0] = model.pools[0].clock_first;
    model.hands[1] = static_cast<std::int32_t>(model.pools[1].owner.size()) - 3;
    device.upload(model);
    std::array<std::vector<std::int32_t>, kPools> below;
    for (std::int32_t p = 0; p < kPools; ++p) {
        const PoolModel& pool = model.pools[p];
        below[p].assign(pool.owner.begin(), pool.owner.begin() + pool.clock_first);
    }
    int failures = check_consistency(model, "pinned initial");

    // One admit and commit of layer `step % kLayers` with counters, compared after each; returns
    // the call's pool-1 misses and how many of them it admitted. Odd steps filter by frequency
    // unless every miss is to be admitted while a slot remains.
    struct F3Misses {
        std::int64_t misses   = 0;
        std::int64_t admitted = 0;
    };

    const auto f3_misses = [&] {
        return model.counters[counter_index(ops::MoeCacheCounter::Misses, 1)];
    };
    std::int32_t step = 0;
    const auto admit  = [&](std::int32_t tokens, const std::string& phase, bool every_miss) {
        const std::int32_t layer = step % kLayers;
        const bool filter        = !every_miss && step % 2 == 1;
        ++step;
        const auto ids = random_ids(tokens, layer, rng);
        GuardedDeviceBuffer ids_storage(ids.size() * sizeof(std::int32_t));
        ids_storage.copy_from_host(ids.data(), ids_storage.bytes());
        const std::string label          = "pinned " + phase + " step " + std::to_string(step);
        const std::int64_t misses_before = f3_misses();
        device.admit(Tensor(ids_storage.data(), DType::I32, {kTopK, tokens}), layer, true, filter,
                     stream);
        cuda_synchronize(stream);
        model_admit(model, ids, layer, true, filter);
        failures += device.compare(model, label + " admit");
        F3Misses result{f3_misses() - misses_before, 0};
        for (std::size_t entry = 0; entry < model.fill.size(); ++entry) {
            if (model.fill[entry] != 0 && (model.host[entry] & 1U) == 1) { ++result.admitted; }
        }
        device.commit(layer, stream);
        cuda_synchronize(stream);
        model_commit(model, layer);
        failures += device.compare(model, label + " commit");
        failures += check_consistency(model, label);
        for (std::int32_t p = 0; p < kPools; ++p) {
            const PoolModel& pool = model.pools[p];
            if (!std::equal(below[p].begin(), below[p].end(), pool.owner.begin())) {
                std::cerr << label << ": a slot below the CLOCK range of pool " << p
                          << " changed owner\n";
                ++failures;
            }
        }
        return result;
    };
    const auto lend = [&](std::int32_t extent, bool borrow) {
        device.lend(1, kFlex, extent * kExtent, kExtent, borrow, stream);
        model_lend(model, 1, kFlex, extent * kExtent, kExtent, borrow);
        cuda_synchronize(stream);
        failures += device.compare(model, std::string("pinned ") + (borrow ? "borrow" : "return"));
    };

    // (1) Hands lie in [clock_first, N), so a pool-1 hand below its start has wrapped.
    const std::int32_t start = model.hands[1];
    bool wrapped             = false;
    for (std::int32_t i = 0; i < 6 && failures == 0; ++i) {
        admit(16, "wrap", false);
        wrapped = wrapped || model.hands[1] < start;
    }
    if (!wrapped || model.hands[1] < model.pools[1].clock_first) {
        std::cerr << "pinned: the hand must wrap to the first slot of its range\n";
        ++failures;
    }
    // (2) Flex exhausted by lending.
    lend(0, true);
    lend(1, true);
    const std::array<std::int32_t, kPools> lent_hands = model.hands;
    F3Misses lent;
    for (std::int32_t i = 0; i < 6 && failures == 0; ++i) {
        const F3Misses call = admit(64, "lent", false);
        lent.misses += call.misses;
        lent.admitted += call.admitted;
    }
    if (model.hands != lent_hands || lent.misses == 0 || lent.admitted != 0) {
        std::cerr << "pinned: lent extents must leave the hands and admit no miss\n";
        ++failures;
    }
    // (3) One extent: wide calls without the filter exhaust its 36 slots, so the rest of their
    // misses stay mapped.
    lend(1, false);
    bool exhausted = false;
    for (std::int32_t i = 0; i < 6 && failures == 0; ++i) {
        const F3Misses call = admit(64, "one extent", true);
        exhausted           = exhausted || call.admitted < call.misses;
    }
    if (!exhausted) {
        std::cerr << "pinned: no call exhausted the range\n";
        ++failures;
    }
    // (4) Both extents back.
    lend(0, false);
    for (std::int32_t i = 0; i < 6 && failures == 0; ++i) { admit(4, "returned", false); }
    return failures;
}

} // namespace

int main(int argc, char** argv) {
    if (require_cuda() != 0) { return 1; }
    try {
        DeviceContext context;
        if (argc == 2 && std::string_view(argv[1]) == "--refill") {
            return run_refill(context.stream) == 0 ? 0 : 1;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--pinned") {
            return run_pinned(5, context.stream) == 0 ? 0 : 1;
        }
        if (argc != 1) {
            throw std::invalid_argument("expected no arguments, --refill or --pinned");
        }
        // Pool sizes: tiny pools are exhausted by one wide call (every slot protected), medium
        // pools churn, and pools above 1024 slots span several scan windows and wrap.
        // A saturated pool (every slot owned and referenced) makes its first miss sweep a whole
        // lap: more than one window, clearing every unprotected bit across the wrap. Flex pools
        // use the target's F3 extent widths (36 and 84 slots) and an NVFP4 pool of flex slots.
        // CLOCK ranges: pool 0's starts past its fixed slots (flex only), pool 1's inside its
        // fixed slots and beyond one scan window, so the hand wraps to a nonzero first slot while
        // lent runs fall on both sides of it.
        const std::array<Scenario, 16> scenarios{{
            {"tiny", 3, {{{3, 512}, {7, 256}}}, 0.25, 0.8, 0.5, 30},
            {"medium", 6, {{{40, 2764800}, {300, 1996800}}}, 0.1, 0.5, 0.5, 40},
            {"wide", 3, {{{1100, 256}, {2500, 512}}}, 0.15, 0.7, 0.5, 25},
            {"saturated", 8, {{{1500, 256}, {2100, 256}}}, 0.45, 1.0, 1.0, 12},
            {"cold", 2, {{{64, 256}, {1030, 256}}}, 0.3, 0.0, 0.0, 30},
            {"one pool", 2, {{{0, 256}, {200, 256}}}, 0.2, 0.5, 0.5, 30},
            // GCLOCK counters, the frequency filter with frequent halving, and lent slots, over
            // churning, multi-window and saturated pools.
            {"gclock", 6, {{{40, 512}, {300, 256}}}, 0.1, 0.5, 0.6, 40, 8, 0, true},
            {"filter", 6, {{{40, 512}, {300, 256}}}, 0.1, 0.9, 0.5, 40, 1, 97, false},
            {"gclock filter lend", 4, {{{60, 512}, {1300, 256}}}, 0.2, 0.8, 0.7, 30, 8, 500, true},
            {"saturated lend", 8, {{{200, 256}, {1100, 256}}}, 0.3, 1.0, 1.0, 12, 255, 4000, true},
            {"flex", 6, {{{40, 512, 12, 4}, {300, 256, 108, 36}}}, 0.1, 0.6, 0.5, 40, 8, 0, true},
            // The expert-helper split's eligibility bitset with the full policy and lending.
            {"eligible",
             4,
             {{{60, 512, 24, 12}, {700, 256, 252, 36}}},
             0.2,
             0.8,
             0.6,
             30,
             8,
             900,
             true,
             true},
            {"flex only", 3, {{{0, 256}, {0, 256, 1260, 36}}}, 0.0, 0.7, 0.5, 25, 8, 3000, true},
            {"flex wide filter",
             4,
             {{{100, 512, 84, 84}, {1300, 256, 840, 84}}},
             0.2,
             0.8,
             0.7,
             25,
             8,
             1500,
             true},
            {"flex saturated",
             8,
             {{{200, 256, 36, 36}, {1100, 256, 252, 36}}},
             0.3,
             1.0,
             1.0,
             12,
             255,
             4000,
             true},
            {"clock range",
             4,
             {{{60, 512, 24, 12, 60}, {1300, 256, 252, 36, 200}}},
             0.2,
             0.8,
             0.6,
             25,
             8,
             1500,
             true,
             true},
        }};
        int failures       = 0;
        std::uint32_t seed = 11;
        for (const Scenario& scenario : scenarios) {
            failures += run_eager(scenario, seed++, context.stream);
        }
        failures += run_graph(seed++, context.stream);
        failures += run_region_checks(seed, context.stream);
        failures += run_refill(context.stream);
        failures += run_pinned(seed + 1, context.stream);
        std::cout << (failures == 0 ? "OK" : "FAIL") << " moe_expert_cache\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "moe_expert_cache: " << error.what() << '\n';
        return 1;
    }
}
