// ninfer::ops - moe_expert_cache wrapper: implements the public api, validates parameters, and
// dispatches to the launcher. Host-compiled; never includes the kernel header.
#include "ninfer/ops/moe_expert_cache.h"

#include "core/device.h"
#include "core/dtype.h"
#include "ninfer/ops/moe_helper_contract.h"
#include "ops/launcher/moe_expert_cache.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::ops {
namespace {

bool contiguous_2d(const Tensor& t, DType dtype, std::size_t alignment) {
    return t.dtype == dtype && t.ne[0] >= 1 && t.ne[1] >= 1 && t.ne[2] == 1 && t.ne[3] == 1 &&
           t.is_contiguous() && t.data != nullptr &&
           (reinterpret_cast<std::uintptr_t>(t.data) % alignment) == 0;
}

struct ByteRange {
    std::uintptr_t begin = 0;
    std::size_t bytes    = 0;
};

ByteRange range_of(const Tensor& t) {
    return {reinterpret_cast<std::uintptr_t>(t.data), t.bytes()};
}

// Throws unless every range lies within the address space and no two ranges overlap.
void require_disjoint(const std::vector<ByteRange>& ranges, const char* message) {
    for (const ByteRange& range : ranges) {
        if (range.bytes > std::numeric_limits<std::uintptr_t>::max() - range.begin) {
            throw std::invalid_argument(message);
        }
    }
    for (std::size_t a = 0; a < ranges.size(); ++a) {
        for (std::size_t b = a + 1; b < ranges.size(); ++b) {
            if (ranges[a].begin < ranges[b].begin + ranges[b].bytes &&
                ranges[b].begin < ranges[a].begin + ranges[a].bytes) {
                throw std::invalid_argument(message);
            }
        }
    }
}

static_assert(kMoeExpertMaskWords * 32 == kMoeExperts);

// Validates an optional eligibility bitset (contiguous I32 [16], 4-byte aligned), appends its
// range for the disjointness check, and returns its words, or null when absent.
const std::uint32_t* validate_eligible(const Tensor* eligible, const char* op,
                                       std::vector<ByteRange>& ranges) {
    if (eligible == nullptr) { return nullptr; }
    if (!contiguous_2d(*eligible, DType::I32, sizeof(std::int32_t)) ||
        eligible->ne[0] != kMoeExpertMaskWords || eligible->ne[1] != 1) {
        throw std::invalid_argument(std::string(op) + ": eligible must be contiguous I32 [16]");
    }
    ranges.push_back(range_of(*eligible));
    return static_cast<const std::uint32_t*>(eligible->data);
}

bool table_of(const Tensor& t, DType dtype, std::int32_t layers) {
    return contiguous_2d(t, dtype, dtype_size(dtype)) && t.ne[0] == kMoeExperts &&
           t.ne[1] == layers;
}

bool table(const Tensor& t, std::int32_t layers) { return table_of(t, DType::I64, layers); }

// Validates an I64 [512,L] table and 0 <= layer < L, and returns L.
std::int32_t validate_table(const Tensor& table_view, std::int32_t layer, const std::string& op,
                            const std::string& name) {
    const std::int32_t layers = table_view.ne[1];
    if (!table(table_view, layers)) {
        throw std::invalid_argument(op + ": " + name + " must be contiguous I64 [512,L]");
    }
    if (layers > std::numeric_limits<std::int32_t>::max() / kMoeExperts) {
        throw std::invalid_argument(op + ": 512 L must fit int32");
    }
    if (layer < 0 || layer >= layers) {
        throw std::invalid_argument(op + ": layer must lie in [0,L)");
    }
    return layers;
}

// Validates two tables of equal L (named `first` and `second`) and the layer index; returns L.
std::int32_t validate_tables(const Tensor& first_table, const Tensor& second_table,
                             std::int32_t layer, const std::string& op, const char* first,
                             const char* second) {
    const std::int32_t layers = validate_table(first_table, layer, op, first);
    if (!table(second_table, layers)) {
        throw std::invalid_argument(op + ": " + first + " and " + second +
                                    " must be contiguous I64 [512,L]");
    }
    return layers;
}

// Validates a contiguous I64 [512] column.
void validate_column(const Tensor& column, const std::string& op, const char* name) {
    if (!contiguous_2d(column, DType::I64, sizeof(std::uint64_t)) || column.ne[0] != kMoeExperts ||
        column.ne[1] != 1) {
        throw std::invalid_argument(op + ": " + name + " must be contiguous I64 [512]");
    }
}

// Validates a pool's geometry and metadata views (MoeExpertSlotPool); the flex addresses are
// device values and stay the caller's precondition.
detail::MoeExpertSlotPoolArgs validate_pool(const MoeExpertSlotPool& pool, const std::string& op) {
    const bool no_fixed = pool.slots.data == nullptr && pool.slots.bytes == 0;
    const bool no_flex  = pool.flex_address.data == nullptr;
    if (no_fixed && no_flex) {
        if (pool.owner.data != nullptr || pool.referenced.data != nullptr) {
            throw std::invalid_argument(op +
                                        ": a pool without slots has null owner and referenced");
        }
        return {};
    }
    if (pool.record_bytes == 0 || pool.record_bytes % kMoeRecordAlignment != 0) {
        throw std::invalid_argument(op + ": pool record_bytes must be a nonzero multiple of 256");
    }
    if (!no_fixed &&
        (pool.slots.data == nullptr ||
         reinterpret_cast<std::uintptr_t>(pool.slots.data) % kMoeRecordAlignment != 0 ||
         pool.slots.bytes == 0 || pool.slots.bytes % pool.record_bytes != 0)) {
        throw std::invalid_argument(op + ": pool slots must be a 256-byte aligned whole number of "
                                         "records, or null and empty");
    }
    const std::size_t fixed = pool.slots.bytes / pool.record_bytes;
    std::size_t flex        = 0;
    if (!no_flex) {
        const Tensor& table = pool.flex_address;
        if (table.dtype != DType::I64 || table.ne[0] < 1 || table.ne[1] != 1 || table.ne[2] != 1 ||
            table.ne[3] != 1 || !table.is_contiguous() ||
            (reinterpret_cast<std::uintptr_t>(table.data) % sizeof(std::uint64_t)) != 0) {
            throw std::invalid_argument(op + ": pool flex_address must be contiguous I64 [X]");
        }
        flex = static_cast<std::size_t>(table.ne[0]);
    }
    if (fixed + flex > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument(op + ": pool slot count must fit int32");
    }
    const auto slots     = static_cast<std::int32_t>(fixed + flex);
    const auto vector_of = [slots](const Tensor& t, DType dtype) {
        return t.dtype == dtype && t.ne[0] == slots && t.ne[1] == 1 && t.ne[2] == 1 &&
               t.ne[3] == 1 && t.is_contiguous() && t.data != nullptr &&
               (reinterpret_cast<std::uintptr_t>(t.data) % dtype_size(dtype)) == 0;
    };
    if (!vector_of(pool.owner, DType::I32) || !vector_of(pool.referenced, DType::U8)) {
        throw std::invalid_argument(op +
                                    ": pool owner and referenced must be contiguous I32 [N] and "
                                    "U8 [N], N = slots.bytes / record_bytes + X");
    }
    return {reinterpret_cast<std::uint64_t>(pool.slots.data),
            pool.record_bytes,
            static_cast<const std::uint64_t*>(pool.flex_address.data),
            static_cast<std::int32_t*>(pool.owner.data),
            static_cast<std::uint8_t*>(pool.referenced.data),
            static_cast<std::int32_t>(fixed),
            slots};
}

// The pool's metadata ranges for the disjointness check: owner, referenced, flex_address and the
// fixed slot bytes, those that exist.
void append_pool_ranges(const MoeExpertSlotPool& pool, std::vector<ByteRange>& ranges) {
    if (pool.owner.data == nullptr) { return; }
    if (pool.slots.data != nullptr) {
        ranges.push_back({reinterpret_cast<std::uintptr_t>(pool.slots.data), pool.slots.bytes});
    }
    if (pool.flex_address.data != nullptr) { ranges.push_back(range_of(pool.flex_address)); }
    ranges.push_back(range_of(pool.owner));
    ranges.push_back(range_of(pool.referenced));
}

std::int32_t* validate_slot_index(const MoeExpertCache& cache, std::int32_t layers,
                                  const std::string& op) {
    if (!table_of(cache.slot_index, DType::I32, layers)) {
        throw std::invalid_argument(op + ": slot_index must be contiguous I32 [512,L]");
    }
    return static_cast<std::int32_t*>(cache.slot_index.data);
}

// The replacement policy of an admit over L layers, after validating the cache's reference limit
// and frequency state.
detail::MoeAdmitPolicyArgs validate_policy(const MoeExpertCache& cache, std::int32_t layers,
                                           MoeCacheAdmission admission) {
    if (cache.reference_limit < 1 || cache.reference_limit > kMoeCacheMaxReferenceLimit) {
        throw std::invalid_argument("moe_expert_cache_admit: reference_limit must lie in [1, " +
                                    std::to_string(kMoeCacheMaxReferenceLimit) + "]");
    }
    detail::MoeAdmitPolicyArgs policy;
    policy.reference_limit         = cache.reference_limit;
    const MoeExpertFrequency& freq = cache.frequency;
    if (freq.counts.data != nullptr) {
        if (!table_of(freq.counts, DType::I32, layers) ||
            !contiguous_2d(freq.lookups, DType::I32, sizeof(std::int32_t)) ||
            freq.lookups.ne[0] != 1 || freq.lookups.ne[1] != 1 || freq.halving_lookups < 1 ||
            freq.halving_lookups > std::numeric_limits<std::int32_t>::max() - kMoeExperts) {
            throw std::invalid_argument(
                "moe_expert_cache_admit: frequency counts must be contiguous I32 [512,L], "
                "lookups I32 [1] and halving_lookups in [1, 2^31 - 513]");
        }
        policy.counts          = static_cast<std::int32_t*>(freq.counts.data);
        policy.lookups         = static_cast<std::int32_t*>(freq.lookups.data);
        policy.halving_lookups = freq.halving_lookups;
        policy.layers          = layers;
    }
    policy.filter = admission == MoeCacheAdmission::FrequencyFilter;
    if (policy.filter && policy.counts == nullptr) {
        throw std::invalid_argument("moe_expert_cache_admit: the frequency filter needs the "
                                    "cache's frequency state");
    }
    return policy;
}

} // namespace

void moe_expert_cache_admit(const Tensor& ids, Tensor& expert_ptr, const Tensor& host_ptr,
                            Tensor& fill_ptr, const MoeExpertCache& cache, std::int32_t layer,
                            MoeCacheAdmission admission, cudaStream_t stream, Tensor* counters,
                            const Tensor* eligible, Tensor* route_counts) {
    const std::int32_t layers = validate_tables(expert_ptr, fill_ptr, layer,
                                                "moe_expert_cache_admit", "expert_ptr", "fill_ptr");
    if (!contiguous_2d(ids, DType::I32, sizeof(std::int32_t)) || ids.ne[0] != kMoeTopK) {
        throw std::invalid_argument("moe_expert_cache_admit: ids must be contiguous I32 [10,T]");
    }
    const std::int64_t id_count = ids.numel();
    if (id_count > std::numeric_limits<std::int32_t>::max()) {
        throw std::invalid_argument("moe_expert_cache_admit: 10 * T must fit int32");
    }
    if (!table(host_ptr, layers)) {
        throw std::invalid_argument(
            "moe_expert_cache_admit: host_ptr must be contiguous I64 [512,L]");
    }
    const detail::MoeExpertSlotPoolArgs pool0 =
        validate_pool(cache.pools[0], "moe_expert_cache_admit");
    const detail::MoeExpertSlotPoolArgs pool1 =
        validate_pool(cache.pools[1], "moe_expert_cache_admit");
    const Tensor& hands = cache.clock_hands;
    if (hands.dtype != DType::I32 || hands.ne[0] != kMoeCacheCodecTags || hands.ne[1] != 1 ||
        hands.ne[2] != 1 || hands.ne[3] != 1 || !hands.is_contiguous() || hands.data == nullptr ||
        (reinterpret_cast<std::uintptr_t>(hands.data) % sizeof(std::int32_t)) != 0) {
        throw std::invalid_argument("moe_expert_cache_admit: clock_hands must be I32 [2]");
    }

    const detail::MoeAdmitPolicyArgs policy = validate_policy(cache, layers, admission);
    std::int32_t* const slot_index = validate_slot_index(cache, layers, "moe_expert_cache_admit");

    std::vector<ByteRange> ranges{range_of(ids),      range_of(expert_ptr),
                                  range_of(host_ptr), range_of(fill_ptr),
                                  range_of(hands),    range_of(cache.slot_index)};
    if (policy.counts != nullptr) {
        ranges.push_back(range_of(cache.frequency.counts));
        ranges.push_back(range_of(cache.frequency.lookups));
    }
    if (counters != nullptr) {
        if (!contiguous_2d(*counters, DType::I64, sizeof(std::int64_t)) ||
            counters->ne[0] != kMoeCacheCounterKinds || counters->ne[1] != kMoeCacheCodecTags) {
            throw std::invalid_argument("moe_expert_cache_admit: counters must be I64 [2,2]");
        }
        ranges.push_back(range_of(*counters));
    }
    const std::uint32_t* const eligible_bits =
        validate_eligible(eligible, "moe_expert_cache_admit", ranges);
    if (route_counts != nullptr) {
        if (!table(*route_counts, layers)) {
            throw std::invalid_argument(
                "moe_expert_cache_admit: route_counts must be contiguous I64 [512,L]");
        }
        ranges.push_back(range_of(*route_counts));
    }
    for (const MoeExpertSlotPool& pool : cache.pools) { append_pool_ranges(pool, ranges); }
    require_disjoint(
        ranges,
        "moe_expert_cache_admit: tables, slots, pool metadata, counters, route counts and the "
        "eligibility bitset must be disjoint");

    detail::moe_expert_cache_admit_launch(
        static_cast<const std::int32_t*>(ids.data), static_cast<std::int32_t>(id_count),
        static_cast<std::uint64_t*>(expert_ptr.data),
        static_cast<const std::uint64_t*>(host_ptr.data),
        static_cast<std::uint64_t*>(fill_ptr.data), slot_index, pool0, pool1,
        static_cast<std::int32_t*>(hands.data), layer, policy,
        counters == nullptr ? nullptr : static_cast<std::uint64_t*>(counters->data), eligible_bits,
        route_counts == nullptr ? nullptr : static_cast<std::uint64_t*>(route_counts->data),
        stream);
}

namespace {

// Borrows (with the tables) or returns (expert_ptr and host_ptr null) a run of one region.
void lend(Tensor* expert_ptr, const Tensor* host_ptr, const MoeExpertCache& cache,
          MoeExpertCodec codec, MoeCacheSlotRegion region, std::int32_t first, std::int32_t count,
          detail::MoeCacheLend lend_kind, cudaStream_t stream) {
    const bool borrow = lend_kind == detail::MoeCacheLend::Borrow;
    const std::string op(borrow ? "moe_expert_cache_borrow" : "moe_expert_cache_return");
    const MoeExpertSlotPool& view            = cache.pools[static_cast<std::size_t>(codec)];
    const detail::MoeExpertSlotPoolArgs pool = validate_pool(view, op);
    const bool fixed                         = region == MoeCacheSlotRegion::Fixed;
    const std::int32_t region_first          = fixed ? 0 : pool.fixed_count;
    const std::int32_t region_slots = fixed ? pool.fixed_count : pool.count - pool.fixed_count;
    if (count < 1 || first < 0 || first > region_slots - count) {
        throw std::invalid_argument(op + ": the slot range must be nonempty and lie in the " +
                                    (fixed ? "fixed" : "flex") + " slots of the pool");
    }
    std::vector<ByteRange> ranges{range_of(view.owner), range_of(view.referenced)};
    std::int32_t* slot_index = nullptr;
    if (borrow) {
        const std::int32_t layers = expert_ptr->ne[1];
        if (!table(*expert_ptr, layers) || !table(*host_ptr, layers)) {
            throw std::invalid_argument(op +
                                        ": expert_ptr and host_ptr must be contiguous I64 [512,L]");
        }
        slot_index = validate_slot_index(cache, layers, op);
        ranges.push_back(range_of(*expert_ptr));
        ranges.push_back(range_of(*host_ptr));
        ranges.push_back(range_of(cache.slot_index));
    }
    require_disjoint(ranges, (op + ": tables and pool metadata must be disjoint").c_str());
    detail::moe_expert_cache_lend_launch(
        pool, borrow ? static_cast<std::uint64_t*>(expert_ptr->data) : nullptr,
        borrow ? static_cast<const std::uint64_t*>(host_ptr->data) : nullptr, slot_index,
        region_first + first, count, lend_kind, stream);
}

} // namespace

void moe_expert_cache_borrow(Tensor& expert_ptr, const Tensor& host_ptr,
                             const MoeExpertCache& cache, MoeExpertCodec codec,
                             MoeCacheSlotRegion region, std::int32_t first, std::int32_t count,
                             cudaStream_t stream) {
    lend(&expert_ptr, &host_ptr, cache, codec, region, first, count, detail::MoeCacheLend::Borrow,
         stream);
}

void moe_expert_cache_return(const MoeExpertCache& cache, MoeExpertCodec codec,
                             MoeCacheSlotRegion region, std::int32_t first, std::int32_t count,
                             cudaStream_t stream) {
    lend(nullptr, nullptr, cache, codec, region, first, count, detail::MoeCacheLend::Return,
         stream);
}

void moe_expert_cache_fill(const Tensor& source_ptr, const Tensor& fill_ptr, std::int32_t layer,
                           cudaStream_t stream, std::optional<std::int32_t> blocks) {
    validate_tables(source_ptr, fill_ptr, layer, "moe_expert_cache_fill", "source_ptr", "fill_ptr");
    require_disjoint({range_of(source_ptr), range_of(fill_ptr)},
                     "moe_expert_cache_fill: source_ptr and fill_ptr must be disjoint");
    constexpr std::int32_t kMaxFillBlocks = 1024;
    if (blocks.has_value() && (*blocks < 1 || *blocks > kMaxFillBlocks)) {
        throw std::invalid_argument("moe_expert_cache_fill: blocks must lie in [1, 1024]");
    }
    std::int32_t grid = 0;
    if (blocks.has_value()) {
        grid = *blocks;
    } else {
        int device_id = 0;
        CUDA_CHECK(cudaGetDevice(&device_id));
        int multiprocessors = 0;
        CUDA_CHECK(
            cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device_id));
        grid = moe_expert_cache_fill_blocks(multiprocessors);
    }
    detail::moe_expert_cache_fill_launch(static_cast<const std::uint64_t*>(source_ptr.data),
                                         static_cast<const std::uint64_t*>(fill_ptr.data), layer,
                                         grid, stream);
}

void moe_expert_cache_window(const Tensor& expert_ptr, const Tensor& host_ptr, std::int32_t layer,
                             DeviceSpan window, Tensor& window_ptr, cudaStream_t stream) {
    validate_tables(expert_ptr, host_ptr, layer, "moe_expert_cache_window", "expert_ptr",
                    "host_ptr");
    if (!contiguous_2d(window_ptr, DType::I64, sizeof(std::uint64_t)) ||
        window_ptr.ne[0] != kMoeExperts || window_ptr.ne[1] != 1) {
        throw std::invalid_argument("moe_expert_cache_window: window_ptr must be I64 [512]");
    }
    const bool empty = window.data == nullptr && window.bytes == 0;
    if (!empty && (window.data == nullptr ||
                   reinterpret_cast<std::uintptr_t>(window.data) % kMoeRecordAlignment != 0)) {
        throw std::invalid_argument(
            "moe_expert_cache_window: the window must be 256-byte aligned, or null and empty");
    }
    std::vector<ByteRange> ranges{range_of(expert_ptr), range_of(host_ptr), range_of(window_ptr)};
    if (!empty) { ranges.push_back({reinterpret_cast<std::uintptr_t>(window.data), window.bytes}); }
    require_disjoint(ranges, "moe_expert_cache_window: tables, window_ptr and window must be "
                             "disjoint");
    const std::size_t column = static_cast<std::size_t>(layer) * kMoeExperts;
    detail::moe_expert_cache_window_launch(
        static_cast<const std::uint64_t*>(expert_ptr.data) + column,
        static_cast<const std::uint64_t*>(host_ptr.data) + column,
        reinterpret_cast<std::uint64_t>(window.data), window.bytes,
        static_cast<std::uint64_t*>(window_ptr.data), stream);
}

void moe_expert_cache_stage(const Tensor& predicted, std::int32_t candidates,
                            const Tensor& expert_ptr, const Tensor& host_ptr, std::int32_t layer,
                            DeviceSpan staging, std::size_t budget_bytes, Tensor& stage_ptr,
                            Tensor& source, cudaStream_t stream, const Tensor* eligible) {
    const std::string op("moe_expert_cache_stage");
    validate_tables(expert_ptr, host_ptr, layer, op, "expert_ptr", "host_ptr");
    if (!contiguous_2d(predicted, DType::I32, sizeof(std::int32_t)) ||
        predicted.ne[0] != kMoeTopK || predicted.ne[1] > kMoeMaxTokens) {
        throw std::invalid_argument(op + ": predicted must be contiguous I32 [10,T], 1 <= T <= " +
                                    std::to_string(kMoeMaxTokens));
    }
    if (candidates < 1 || candidates > kMoeStageMaxCandidates) {
        throw std::invalid_argument(op + ": candidates must lie in [1,10]");
    }
    validate_column(stage_ptr, op, "stage_ptr");
    validate_column(source, op, "source");
    const bool empty = staging.data == nullptr;
    if (empty ? staging.bytes != 0 || budget_bytes != 0
              : reinterpret_cast<std::uintptr_t>(staging.data) % kMoeRecordAlignment != 0 ||
                    budget_bytes > staging.bytes) {
        throw std::invalid_argument(op + ": the staging span must be 256-byte aligned and hold "
                                         "the budget, or be null with a zero budget");
    }
    std::vector<ByteRange> ranges{range_of(predicted), range_of(expert_ptr), range_of(host_ptr),
                                  range_of(stage_ptr), range_of(source)};
    if (!empty) {
        ranges.push_back({reinterpret_cast<std::uintptr_t>(staging.data), staging.bytes});
    }
    const std::uint32_t* const eligible_bits = validate_eligible(eligible, op.c_str(), ranges);
    require_disjoint(ranges, (op + ": tensors and the staging span must be disjoint").c_str());
    const std::size_t column = static_cast<std::size_t>(layer) * kMoeExperts;
    detail::moe_expert_cache_stage_launch(
        static_cast<const std::int32_t*>(predicted.data), predicted.ne[1], candidates,
        static_cast<const std::uint64_t*>(expert_ptr.data) + column,
        static_cast<const std::uint64_t*>(host_ptr.data) + column,
        reinterpret_cast<std::uint64_t>(staging.data), budget_bytes,
        static_cast<std::uint64_t*>(stage_ptr.data), static_cast<std::uint64_t*>(source.data),
        eligible_bits, stream);
}

void moe_expert_cache_defer(const Tensor& fill_ptr, std::int32_t layer, const Tensor& stage_ptr,
                            Tensor& deferred, cudaStream_t stream) {
    const std::string op("moe_expert_cache_defer");
    validate_table(fill_ptr, layer, op, "fill_ptr");
    validate_column(stage_ptr, op, "stage_ptr");
    validate_column(deferred, op, "deferred");
    require_disjoint({range_of(fill_ptr), range_of(stage_ptr), range_of(deferred)},
                     (op + ": fill_ptr, stage_ptr and deferred must be disjoint").c_str());
    detail::moe_expert_cache_defer_launch(static_cast<const std::uint64_t*>(fill_ptr.data) +
                                              static_cast<std::size_t>(layer) * kMoeExperts,
                                          static_cast<const std::uint64_t*>(stage_ptr.data),
                                          static_cast<std::uint64_t*>(deferred.data), stream);
}

void moe_expert_cache_commit(Tensor& expert_ptr, Tensor& fill_ptr, std::int32_t layer,
                             cudaStream_t stream) {
    validate_tables(expert_ptr, fill_ptr, layer, "moe_expert_cache_commit", "expert_ptr",
                    "fill_ptr");
    require_disjoint({range_of(expert_ptr), range_of(fill_ptr)},
                     "moe_expert_cache_commit: expert_ptr and fill_ptr must be disjoint");
    detail::moe_expert_cache_commit_launch(static_cast<std::uint64_t*>(expert_ptr.data),
                                           static_cast<std::uint64_t*>(fill_ptr.data), layer,
                                           stream);
}

} // namespace ninfer::ops
