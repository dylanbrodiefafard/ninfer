#include "ops/launcher/moe_expert_cache.h"

#include "core/device.h"
#include "ninfer/ops/moe_expert_cache.h"
#include "ops/common/warp.cuh"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kAdmitThreads    = 1024;
constexpr int kAdmitWarps      = kAdmitThreads / kWarpSize;
constexpr int kRoutedMaskWords = kMoeExperts / kWarpSize;
constexpr int kExpertWarps     = kMoeExperts / kWarpSize;
constexpr int kCommitThreads   = kMoeExperts;
static_assert(kMoeExperts % kWarpSize == 0 && kMoeExperts <= kAdmitThreads);
static_assert(kAdmitWarps <= kWarpSize);
static_assert(kMoeCacheCodecTags == 2);

constexpr std::uint64_t kCodecTagMask = 1;

struct AdmitShared {
    // Bit e of the 512-bit set: expert e is routed by this call.
    unsigned routed[kRoutedMaskWords];
    // Misses of each pool in ascending expert id, and their count.
    int misses[kMoeCacheCodecTags][kMoeExperts];
    int miss_count[kMoeCacheCodecTags];
    // Per-warp miss counts, turned into exclusive offsets, while the miss lists are built.
    int miss_offset[kMoeCacheCodecTags][kExpertWarps];
    // Per-warp counts of routed experts of each pool (the Lookups counter).
    int lookups[kMoeCacheCodecTags][kExpertWarps];
    // Block scan of one CLOCK window: per-warp exclusive offsets and the window total.
    int scan_offset[kAdmitWarps];
    int scan_total;
    // Window index of the last victim the window takes.
    int last_victim;
    // Lookups of the call (both pools) and whether it halves the frequency counts.
    int lookup_total[kMoeCacheCodecTags];
    int halve;
};

__device__ __forceinline__ unsigned lanes_below(int lane) { return (1U << lane) - 1U; }

// address(slot) of the pool contract: the fixed array below fixed_count, the flex table above.
__device__ __forceinline__ std::uint64_t slot_address(const MoeExpertSlotPoolArgs& pool, int slot) {
    return slot < pool.fixed_count
               ? pool.base + static_cast<std::uint64_t>(slot) * pool.record_bytes
               : pool.flex_address[slot - pool.fixed_count];
}

__device__ __forceinline__ bool is_routed(const AdmitShared& shared, int expert) {
    return ((shared.routed[expert / kWarpSize] >> (expert % kWarpSize)) & 1U) != 0;
}

// Exclusive rank of flag among the block's threads in thread order; the window total is left in
// shared.scan_total. Contains two block barriers; every thread of the block calls it.
__device__ int block_exclusive_rank(bool flag, AdmitShared& shared, int lane, int warp) {
    const unsigned ballot = __ballot_sync(kFullWarpMask, flag);
    if (lane == 0) { shared.scan_offset[warp] = __popc(ballot); }
    __syncthreads();
    if (warp == 0) {
        const int count = lane < kAdmitWarps ? shared.scan_offset[lane] : 0;
        int inclusive   = count;
#pragma unroll
        for (int offset = 1; offset < kWarpSize; offset *= 2) {
            const int below = __shfl_up_sync(kFullWarpMask, inclusive, offset);
            if (lane >= offset) { inclusive += below; }
        }
        if (lane < kAdmitWarps) { shared.scan_offset[lane] = inclusive - count; }
        if (lane == kWarpSize - 1) { shared.scan_total = inclusive; }
    }
    __syncthreads();
    return shared.scan_offset[warp] + __popc(ballot & lanes_below(lane));
}

// Admits the misses of one pool by CLOCK, `window` = min(1024, N) consecutive hand positions at a
// time. Thread i examines position hand + i (mod N), so a window visits each slot at most once.
// Within a window only the visited slot itself changes its candidacy (not protected and a zero
// reference counter), so the window's first `need` candidates in hand order are exactly the
// victims a one-slot-at-a-time CLOCK takes, and the counters it decrements are those of the
// unprotected slots before the last victim (the whole window when it takes fewer than `need`).
// The frequency filter pairs the k-th miss with the k-th candidate, as the sequential policy
// does, and decides each pair independently; a rejected pair leaves its slot unchanged. Protection
// only grows during the call and a counter reaches zero within reference_limit visits, so
// (reference_limit + 1) N consecutive positions without a victim prove that every slot is
// protected: the remaining misses are not admitted and the hand returns to the position after the
// last victim, where the sequential policy, which checks for an unprotected slot before each miss,
// leaves it. Global writes of one window are visible to the next after its final barrier. The
// slot_index entries a window writes are distinct: each evicted owner holds one victim slot of
// the window, each admitted miss is a distinct non-resident expert, and the two sets are disjoint.
__device__ void admit_pool(AdmitShared& shared, int tag, const MoeExpertSlotPoolArgs& pool,
                           std::uint64_t* expert_ptr, const std::uint64_t* host_ptr,
                           std::uint64_t* fill_ptr, std::int32_t* slot_index,
                           std::int32_t* clock_hands, int layer, const MoeAdmitPolicyArgs& policy) {
    const int misses = shared.miss_count[tag];
    const int slots  = pool.count;
    if (misses == 0 || slots == 0) { return; }
    const int thread       = static_cast<int>(threadIdx.x);
    const int lane         = thread % kWarpSize;
    const int warp         = thread / kWarpSize;
    const int window       = slots < kAdmitThreads ? slots : kAdmitThreads;
    const int layer_base   = layer * kMoeExperts;
    const std::int64_t lap = static_cast<std::int64_t>(slots);

    int hand               = clock_hands[tag];
    int resume             = hand;
    int assigned           = 0;
    std::int64_t fruitless = 0;
    while (assigned < misses) {
        const bool active     = thread < window;
        std::int64_t position = static_cast<std::int64_t>(hand) + thread;
        if (position >= lap) { position -= lap; }
        int owner        = kMoeCacheEmptySlot;
        bool unprotected = false;
        int reference    = 0;
        if (active) {
            owner       = pool.owner[position];
            unprotected = owner == kMoeCacheEmptySlot ||
                          (owner >= 0 && (owner / kMoeExperts != layer ||
                                          !is_routed(shared, owner % kMoeExperts)));
            reference   = pool.referenced[position];
        }
        const bool referenced = reference != 0;
        const bool candidate  = unprotected && !referenced;
        const int rank        = block_exclusive_rank(candidate, shared, lane, warp);
        const int need        = misses - assigned;
        const int found       = shared.scan_total < need ? shared.scan_total : need;
        if (candidate && rank == found - 1) { shared.last_victim = thread; }
        __syncthreads();
        const int last  = found > 0 ? shared.last_victim : -1;
        const int limit = found == need ? last : window;
        if (unprotected && referenced && thread < limit) {
            pool.referenced[position] = static_cast<std::uint8_t>(reference - 1);
        }
        if (candidate && rank < need) {
            const int expert = shared.misses[tag][assigned + rank];
            const int code   = layer_base + expert;
            // The filter pairs miss `rank` with this victim; a rejected miss leaves the slot.
            const bool admit = !policy.filter || policy.counts == nullptr || owner < 0 ||
                               policy.counts[code] > policy.counts[owner];
            if (admit) {
                const int slot = static_cast<int>(position);
                if (owner >= 0) {
                    expert_ptr[owner] = host_ptr[owner];
                    slot_index[owner] = kMoeCacheNoSlot;
                }
                pool.owner[slot]      = code;
                pool.referenced[slot] = 1;
                slot_index[code]      = slot;
                fill_ptr[code]        = slot_address(pool, slot);
            }
        }
        // Every thread derives the same loop state from shared values.
        if (found > 0) {
            resume    = static_cast<int>((static_cast<std::int64_t>(hand) + last + 1) % lap);
            fruitless = window - 1 - last;
        } else {
            fruitless += window;
        }
        assigned += found;
        hand = static_cast<int>((static_cast<std::int64_t>(hand) + window) % lap);
        if (assigned == misses) { hand = resume; }
        // Ends the window: its writes become visible and its shared values may be reused.
        __syncthreads();
        if (fruitless >= (policy.reference_limit + 1) * lap) {
            hand = resume;
            break;
        }
    }
    if (thread == 0) { clock_hands[tag] = hand; }
}

// Grid 1, block 1024. Phase 1 marks the routed set in a shared 512-bit mask (shared atomics, so
// repeated ids need no ordering). Phase 2: thread e < 512 classifies a routed expert as a hit
// (increments the reference counter of its slot, slot_index[e], up to the limit; distinct hits
// own distinct slots) or a miss of its pool; warp ballots and per-warp offsets give the
// ascending miss lists, and thread c < 2 sums pool c's per-warp
// counts and, when counters is not null, adds its lookups and misses (the only writer of those
// two counters in the grid). Phase 3 (frequency state only): thread e counts its routed expert,
// thread 0 advances the lookups since the last halving, and after a barrier every thread halves a
// strided share of the counts when the period is reached. Phase 4 runs CLOCK for pool 0 then pool
// 1 (admit_pool), reading the counts after the barrier that ends phase 3's writes. Barriers
// separate the phases; all global state lives in the arguments.
__global__ void __launch_bounds__(kAdmitThreads) moe_expert_cache_admit_kernel(
    const std::int32_t* __restrict__ ids, int id_count, std::uint64_t* __restrict__ expert_ptr,
    const std::uint64_t* __restrict__ host_ptr, std::uint64_t* __restrict__ fill_ptr,
    std::int32_t* __restrict__ slot_index, MoeExpertSlotPoolArgs pool0, MoeExpertSlotPoolArgs pool1,
    std::int32_t* clock_hands, int layer, MoeAdmitPolicyArgs policy, std::uint64_t* counters) {
    __shared__ AdmitShared shared;
    const int thread = static_cast<int>(threadIdx.x);
    const int lane   = thread % kWarpSize;
    const int warp   = thread / kWarpSize;

    if (thread < kRoutedMaskWords) { shared.routed[thread] = 0; }
    __syncthreads();
    for (std::int64_t index = thread; index < id_count; index += kAdmitThreads) {
        const int expert = ids[index];
        atomicOr(&shared.routed[expert / kWarpSize], 1U << (expert % kWarpSize));
    }
    __syncthreads();

    bool miss[kMoeCacheCodecTags]   = {false, false};
    bool lookup[kMoeCacheCodecTags] = {false, false};
    if (thread < kMoeExperts && is_routed(shared, thread)) {
        const int entry             = layer * kMoeExperts + thread;
        const std::uint64_t address = expert_ptr[entry];
        const std::uint64_t host    = host_ptr[entry];
        const int tag               = static_cast<int>(host & kCodecTagMask);
        lookup[tag]                 = true;
        if (address == host) {
            miss[tag] = true;
        } else {
            const MoeExpertSlotPoolArgs& pool = tag == 0 ? pool0 : pool1;
            const int slot                    = slot_index[entry];
            const int reference               = pool.referenced[slot];
            pool.referenced[slot]             = static_cast<std::uint8_t>(
                reference < policy.reference_limit ? reference + 1 : reference);
        }
    }
    unsigned ballot[kMoeCacheCodecTags];
#pragma unroll
    for (int tag = 0; tag < kMoeCacheCodecTags; ++tag) {
        ballot[tag]                  = __ballot_sync(kFullWarpMask, miss[tag]);
        const unsigned lookup_ballot = __ballot_sync(kFullWarpMask, lookup[tag]);
        if (lane == 0 && warp < kExpertWarps) {
            shared.miss_offset[tag][warp] = __popc(ballot[tag]);
            shared.lookups[tag][warp]     = __popc(lookup_ballot);
        }
    }
    __syncthreads();
    if (thread < kMoeCacheCodecTags) {
        int total   = 0;
        int lookups = 0;
        for (int w = 0; w < kExpertWarps; ++w) {
            const int count               = shared.miss_offset[thread][w];
            shared.miss_offset[thread][w] = total;
            total += count;
            lookups += shared.lookups[thread][w];
        }
        shared.miss_count[thread]   = total;
        shared.lookup_total[thread] = lookups;
        if (counters != nullptr) {
            // Unsigned sums, so the contract's wrap modulo 2^64 is defined.
            std::uint64_t* codec = counters + kMoeCacheCounterKinds * thread;
            codec[static_cast<int>(MoeCacheCounter::Lookups)] +=
                static_cast<std::uint64_t>(lookups);
            codec[static_cast<int>(MoeCacheCounter::Misses)] += static_cast<std::uint64_t>(total);
        }
    }
    __syncthreads();
    if (policy.counts != nullptr) {
        // Frequency: count the call's routed experts, then halve every count when the lookups
        // since the last halving reach the period. Thread e owns entry e of the layer here.
        if (lookup[0] || lookup[1]) {
            std::int32_t& count = policy.counts[layer * kMoeExperts + thread];
            if (count < 0x7fffffff) { ++count; }
        }
        if (thread == 0) {
            const int seen  = *policy.lookups + shared.lookup_total[0] + shared.lookup_total[1];
            shared.halve    = seen >= policy.halving_lookups ? 1 : 0;
            *policy.lookups = shared.halve != 0 ? 0 : seen;
        }
        __syncthreads();
        if (shared.halve != 0) {
            const int entries = policy.layers * kMoeExperts;
            for (int entry = thread; entry < entries; entry += kAdmitThreads) {
                policy.counts[entry] >>= 1;
            }
        }
    }
#pragma unroll
    for (int tag = 0; tag < kMoeCacheCodecTags; ++tag) {
        if (miss[tag]) {
            shared.misses[tag][shared.miss_offset[tag][warp] +
                               __popc(ballot[tag] & lanes_below(lane))] = thread;
        }
    }
    __syncthreads();

    admit_pool(shared, 0, pool0, expert_ptr, host_ptr, fill_ptr, slot_index, clock_hands, layer,
               policy);
    admit_pool(shared, 1, pool1, expert_ptr, host_ptr, fill_ptr, slot_index, clock_hands, layer,
               policy);
}

// Grid 1, block 512: thread e publishes entry e of the layer's fill table, keeping the codec tag
// of the host address the entry holds until then.
__global__ void __launch_bounds__(kCommitThreads)
    moe_expert_cache_commit_kernel(std::uint64_t* __restrict__ expert_ptr,
                                   std::uint64_t* __restrict__ fill_ptr, int layer) {
    const int entry          = layer * kMoeExperts + static_cast<int>(threadIdx.x);
    const std::uint64_t slot = fill_ptr[entry];
    if (slot != 0) {
        expert_ptr[entry] = slot | (expert_ptr[entry] & kCodecTagMask);
        fill_ptr[entry]   = 0;
    }
}

constexpr int kFillThreads          = 256;
constexpr int kFillUnroll           = 4;
constexpr int kFillEntriesPerThread = kMoeExperts / kFillThreads;
static_assert(kMoeExperts % kFillThreads == 0 && kFillThreads / kWarpSize <= kWarpSize);
static_assert(kMoeNvfp4RecordBytes % 16 == 0 && kMoeF3RecordBytes % 16 == 0);

// The tables hold record and slot addresses as integers by contract.
__device__ __forceinline__ uint4* table_word(std::uint64_t bits) {
    return reinterpret_cast<uint4*>(bits); // NOLINT(performance-no-int-to-ptr)
}

// Grid moe_expert_cache_fill_blocks(SMs), block 256. Thread t owns fill entries 2t and 2t+1 of the
// layer's column; a warp scan and a scan of the eight warp totals give word_prefix[e], the first
// 16-byte word of pending record e in the concatenation of pending records (ascending e), with
// word_prefix[512] the total. After the barrier, global word index w = (blockIdx * 256 + thread)
// + k * gridDim * 256 strides over the total, kFillUnroll words per iteration with their loads
// issued before any store. The tables are only read, so CTAs never synchronize with each other.
__global__ void __launch_bounds__(kFillThreads)
    moe_expert_cache_fill_kernel(const std::uint64_t* __restrict__ host_ptr,
                                 const std::uint64_t* __restrict__ fill_ptr, int layer) {
    __shared__ std::uint64_t word_prefix[kMoeExperts + 1];
    __shared__ std::uint64_t source[kMoeExperts];
    __shared__ std::uint64_t target[kMoeExperts];
    __shared__ std::uint64_t warp_total[kFillThreads / kWarpSize];
    const int thread = static_cast<int>(threadIdx.x);
    const int lane   = thread % kWarpSize;
    const int warp   = thread / kWarpSize;

    std::uint64_t words[kFillEntriesPerThread];
    std::uint64_t mine = 0;
#pragma unroll
    for (int i = 0; i < kFillEntriesPerThread; ++i) {
        const int expert         = kFillEntriesPerThread * thread + i;
        const int entry          = layer * kMoeExperts + expert;
        const std::uint64_t slot = fill_ptr[entry];
        const std::uint64_t host = host_ptr[entry];
        const std::uint64_t record =
            (host & kCodecTagMask) == 0 ? kMoeNvfp4RecordBytes : kMoeF3RecordBytes;
        words[i]       = slot != 0 ? record / 16 : 0;
        source[expert] = host & ~kCodecTagMask;
        target[expert] = slot;
        mine += words[i];
    }
    std::uint64_t inclusive = mine;
#pragma unroll
    for (int offset = 1; offset < kWarpSize; offset *= 2) {
        const std::uint64_t below = __shfl_up_sync(kFullWarpMask, inclusive, offset);
        if (lane >= offset) { inclusive += below; }
    }
    if (lane == kWarpSize - 1) { warp_total[warp] = inclusive; }
    __syncthreads();
    std::uint64_t before = inclusive - mine;
    for (int w = 0; w < warp; ++w) { before += warp_total[w]; }
#pragma unroll
    for (int i = 0; i < kFillEntriesPerThread; ++i) {
        word_prefix[kFillEntriesPerThread * thread + i] = before;
        before += words[i];
    }
    if (thread == kFillThreads - 1) { word_prefix[kMoeExperts] = before; }
    __syncthreads();

    const std::uint64_t total  = word_prefix[kMoeExperts];
    const std::uint64_t stride = static_cast<std::uint64_t>(gridDim.x) * kFillThreads;
    for (std::uint64_t first = static_cast<std::uint64_t>(blockIdx.x) * kFillThreads + thread;
         first < total; first += stride * kFillUnroll) {
        uint4 value[kFillUnroll];
        uint4* destination[kFillUnroll];
#pragma unroll
        for (int k = 0; k < kFillUnroll; ++k) {
            const std::uint64_t word = first + static_cast<std::uint64_t>(k) * stride;
            destination[k]           = nullptr;
            if (word < total) {
                // The record holding `word`: the last e with word_prefix[e] <= word. Records with
                // no pending fill have equal prefixes, so the search lands on a pending record.
                int low  = 0;
                int high = kMoeExperts;
                while (high - low > 1) {
                    const int mid = (low + high) / 2;
                    if (word_prefix[mid] <= word) {
                        low = mid;
                    } else {
                        high = mid;
                    }
                }
                const std::uint64_t offset = (word - word_prefix[low]) * 16;
                // The record is device-mapped host memory, immutable while enqueued work reads it.
                value[k]       = __ldcs(table_word(source[low] + offset));
                destination[k] = table_word(target[low] + offset);
            }
        }
#pragma unroll
        for (int k = 0; k < kFillUnroll; ++k) {
            if (destination[k] != nullptr) { *destination[k] = value[k]; }
        }
    }
}

constexpr int kWindowThreads = kMoeExperts;
constexpr int kWindowWarps   = kWindowThreads / kWarpSize;
static_assert(kWindowWarps <= kWarpSize);

// Grid 1, block 512: thread e takes n_e (0 for a resident expert); a warp scan of the 64-bit
// sizes and a scan of the 16 warp totals in shared memory give the exclusive prefix p_e. One
// barrier separates the warp totals' writes from their reads.
__global__ void __launch_bounds__(kWindowThreads)
    moe_expert_cache_window_kernel(const std::uint64_t* __restrict__ expert_ptr,
                                   const std::uint64_t* __restrict__ host_ptr,
                                   std::uint64_t window_base, std::uint64_t window_bytes,
                                   std::uint64_t* __restrict__ window_ptr) {
    __shared__ std::uint64_t warp_total[kWindowWarps];
    const int expert = static_cast<int>(threadIdx.x);
    const int lane   = expert % kWarpSize;
    const int warp   = expert / kWarpSize;

    const std::uint64_t host = host_ptr[expert];
    const bool streamed      = expert_ptr[expert] == host;
    const std::uint64_t bytes =
        !streamed ? 0 : ((host & kCodecTagMask) == 0 ? kMoeNvfp4RecordBytes : kMoeF3RecordBytes);
    std::uint64_t inclusive = bytes;
#pragma unroll
    for (int offset = 1; offset < kWarpSize; offset *= 2) {
        const std::uint64_t below = __shfl_up_sync(kFullWarpMask, inclusive, offset);
        if (lane >= offset) { inclusive += below; }
    }
    if (lane == kWarpSize - 1) { warp_total[warp] = inclusive; }
    __syncthreads();
    std::uint64_t prefix = inclusive - bytes;
    for (int w = 0; w < warp; ++w) { prefix += warp_total[w]; }
    window_ptr[expert] =
        streamed && prefix + bytes <= window_bytes ? window_base + prefix : std::uint64_t{0};
}

constexpr int kLendThreads = 256;

// Grid ceil(count / 256), block 256: thread i of the grid owns slot first + i of the range, an
// absolute slot index of either region. Slots are distinct and every owner code occurs in at most
// one slot, so each table entry evicted belongs to exactly one thread; no shared memory, no
// barriers.
__global__ void __launch_bounds__(kLendThreads)
    moe_expert_cache_lend_kernel(MoeExpertSlotPoolArgs pool, std::uint64_t* __restrict__ expert_ptr,
                                 const std::uint64_t* __restrict__ host_ptr,
                                 std::int32_t* __restrict__ slot_index, int first, int count,
                                 MoeCacheLend lend) {
    const int index = static_cast<int>(blockIdx.x) * kLendThreads + static_cast<int>(threadIdx.x);
    if (index >= count) { return; }
    const int slot = first + index;
    if (lend == MoeCacheLend::Borrow) {
        const int owner = pool.owner[slot];
        if (owner >= 0) {
            expert_ptr[owner] = host_ptr[owner];
            slot_index[owner] = kMoeCacheNoSlot;
        }
        pool.owner[slot] = kMoeCacheBorrowedSlot;
    } else {
        pool.owner[slot] = kMoeCacheEmptySlot;
    }
    pool.referenced[slot] = 0;
}

} // namespace

void moe_expert_cache_window_launch(const std::uint64_t* expert_ptr, const std::uint64_t* host_ptr,
                                    std::uint64_t window_base, std::uint64_t window_bytes,
                                    std::uint64_t* window_ptr, cudaStream_t stream) {
    moe_expert_cache_window_kernel<<<1, kWindowThreads, 0, stream>>>(
        expert_ptr, host_ptr, window_base, window_bytes, window_ptr);
    CUDA_CHECK(cudaGetLastError());
}

void moe_expert_cache_fill_launch(const std::uint64_t* host_ptr, const std::uint64_t* fill_ptr,
                                  std::int32_t layer, std::int32_t blocks, cudaStream_t stream) {
    moe_expert_cache_fill_kernel<<<blocks, kFillThreads, 0, stream>>>(host_ptr, fill_ptr, layer);
    CUDA_CHECK(cudaGetLastError());
}

void moe_expert_cache_admit_launch(const std::int32_t* ids, std::int32_t id_count,
                                   std::uint64_t* expert_ptr, const std::uint64_t* host_ptr,
                                   std::uint64_t* fill_ptr, std::int32_t* slot_index,
                                   MoeExpertSlotPoolArgs pool0, MoeExpertSlotPoolArgs pool1,
                                   std::int32_t* clock_hands, std::int32_t layer,
                                   MoeAdmitPolicyArgs policy, std::uint64_t* counters,
                                   cudaStream_t stream) {
    moe_expert_cache_admit_kernel<<<1, kAdmitThreads, 0, stream>>>(
        ids, id_count, expert_ptr, host_ptr, fill_ptr, slot_index, pool0, pool1, clock_hands, layer,
        policy, counters);
    CUDA_CHECK(cudaGetLastError());
}

void moe_expert_cache_lend_launch(MoeExpertSlotPoolArgs pool, std::uint64_t* expert_ptr,
                                  const std::uint64_t* host_ptr, std::int32_t* slot_index,
                                  std::int32_t first, std::int32_t count, MoeCacheLend lend,
                                  cudaStream_t stream) {
    const int blocks = (count + kLendThreads - 1) / kLendThreads;
    moe_expert_cache_lend_kernel<<<blocks, kLendThreads, 0, stream>>>(
        pool, expert_ptr, host_ptr, slot_index, first, count, lend);
    CUDA_CHECK(cudaGetLastError());
}

void moe_expert_cache_commit_launch(std::uint64_t* expert_ptr, std::uint64_t* fill_ptr,
                                    std::int32_t layer, cudaStream_t stream) {
    moe_expert_cache_commit_kernel<<<1, kCommitThreads, 0, stream>>>(expert_ptr, fill_ptr, layer);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
