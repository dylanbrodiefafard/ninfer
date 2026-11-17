#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/moe_experts.h"
#include "ninfer/ops/moe_route.h"

#include <cuda_runtime.h> // cudaStream_t

#include <cstddef>
#include <cstdint>
#include <optional>

namespace ninfer::ops {

// One slot pool per record codec, indexed by MoeExpertCodec (bit 0 of a record address).
inline constexpr std::int32_t kMoeCacheCodecTags = 2;
inline constexpr std::int32_t kMoeCacheEmptySlot = -1;
// Owner of a slot lent out by moe_expert_cache_borrow: never a CLOCK victim until returned.
inline constexpr std::int32_t kMoeCacheBorrowedSlot = -2;
// MoeExpertCache::slot_index entry of an expert that no slot holds.
inline constexpr std::int32_t kMoeCacheNoSlot = -1;
// Largest reference_limit (the reference counters are U8).
inline constexpr std::int32_t kMoeCacheMaxReferenceLimit = 255;
static_assert(static_cast<int>(MoeExpertCodec::Nvfp4G16) == 0 &&
              static_cast<int>(MoeExpertCodec::F3G32) == 1);

// Kinds of moe_expert_cache_admit's optional counters I64 [2,2]: counters[kind + 2 c] counts
// events of codec c.
enum class MoeCacheCounter : std::int32_t { Lookups = 0, Misses = 1 };
inline constexpr std::int32_t kMoeCacheCounterKinds = 2;

/**
 * One size class of the device expert cache: N = F + X slots of record_bytes each, in two regions.
 *
 *   fixed slots s in [0, F): F = slots.bytes / record_bytes slots of one contiguous array,
 *              address(s) = slots.data + s * record_bytes;
 *   flex slots s in [F, N): X = flex_address.ne[0] slots at address(s) = flex_address[s - F]
 *              (I64 [X] holding U64 device addresses), memory that another owner lends to the
 *              cache while it holds nothing of its own, such as paged-KV pages without live KV;
 *   owner      I32 [N]: the owner code l * 512 + e of the record (expert e of layer l) held in
 *              slot s, -1 (kMoeCacheEmptySlot) when the slot is empty, or -2
 *              (kMoeCacheBorrowedSlot) while moe_expert_cache_borrow lends it out;
 *   referenced U8  [N]: the slot's CLOCK reference counter, in [0, reference_limit].
 *
 * Slot order is the CLOCK order, and moe_expert_cache_borrow lends runs of it within one region;
 * it does not follow addresses. record_bytes is a nonzero multiple of 256 (kMoeRecordAlignment),
 * slots.data is 256-byte aligned and slots.bytes is F * record_bytes, or slots is {nullptr, 0}
 * (F = 0); a pool without flex slots has null flex_address data (X = 0). Every flex address is
 * 256-byte aligned and names record_bytes bytes, disjoint from every other slot and from every
 * table, record and metadata array the Ops use (the device addresses cannot be checked on the
 * host; a violation is undefined behavior). flex_address is read only, and only by admit for a
 * slot it assigns; its contents, X and the slot order stay fixed while the views are in use (a
 * captured graph bakes them in), and lending a flex slot changes its owner state, never its
 * address. moe_expert_cache_fill writes moe_expert_record_bytes(c) bytes into a slot of pool c,
 * so a production pool uses at least that stride (exactly it in the target). A pool without slots
 * (N = 0) has null owner and referenced data, its record_bytes is ignored, and its records are
 * never admitted. N < 2^31. Non-owning views; this Op family never reads or writes the slot bytes
 * except moe_expert_cache_fill, which writes them.
 */
struct MoeExpertSlotPool {
    DeviceSpan slots;
    std::size_t record_bytes = 0;
    Tensor flex_address;
    Tensor owner;
    Tensor referenced;
};

/** The slots of a pool that moe_expert_cache_borrow and moe_expert_cache_return address. */
enum class MoeCacheSlotRegion : std::uint8_t {
    // Slots [0, F): their bytes are contiguous, so a borrowed run is one streaming window.
    Fixed,
    // Slots [F, N): lent back to the owner of their memory while it needs the bytes.
    Flex,
};

/**
 * Per-expert lookup frequencies for the admission filter, over L layers:
 *
 *   counts   I32 [512,L]: counts[e,l] is the number of admit calls of layer l whose routed set
 *            held e since the counters were zeroed, halved (floor) at every halving;
 *   lookups  I32 [1]: lookups (routed-set elements, every layer) since the last halving.
 *
 * Every counts value is in [0, 2^31 - 1] and lookups in [0, halving_lookups). A halving happens
 * when lookups reaches halving_lookups, in [1, 2^31 - 513] so that lookups plus one call's routed
 * set fits I32: every count is shifted right by one and lookups
 * restarts at 0. The caller zeroes both tensors before the first admit.
 */
struct MoeExpertFrequency {
    Tensor counts;
    Tensor lookups;
    std::int32_t halving_lookups = 0;
};

/**
 * The complete device state of the expert cache, global across layers: pools[c] holds records
 * of codec c; slot_index I32 [512,L] (contiguous, column l = layer l) is the inverse of the
 * owners: slot_index[e,l] = s when slot s of pool tag(host_ptr[e,l]) has owner l * 512 + e, and
 * kMoeCacheNoSlot (-1) when no slot does; clock_hands I32 [2] is the next CLOCK position of each
 * pool (0 <= hand < N, 0 for a pool without slots); reference_limit in
 * [1, kMoeCacheMaxReferenceLimit] caps the reference counters (1 is CLOCK's reference bit, K > 1
 * GCLOCK with counters up to K); frequency, when its counts are present (data not null), counts
 * every admit's lookups and enables MoeCacheAdmission::FrequencyFilter.
 */
struct MoeExpertCache {
    MoeExpertSlotPool pools[kMoeCacheCodecTags];
    Tensor slot_index;
    Tensor clock_hands;
    std::int32_t reference_limit = 1;
    MoeExpertFrequency frequency;
};

/** Which misses moe_expert_cache_admit admits into the pools. */
enum class MoeCacheAdmission : std::uint8_t {
    // Every miss for which an unprotected slot exists.
    EveryMiss,
    // A miss only when its frequency count exceeds that of the CLOCK victim's owner (a TinyLFU
    // filter); an empty victim slot always admits. Requires cache.frequency.
    FrequencyFilter,
};

/**
 * Op: moe_expert_cache_admit
 *
 * Math / indexing:
 *   Tables over L layers (I64 holding U64 bit patterns; column l is layer l):
 *     expert_ptr [512,L]  the record address moe_experts reads expert e of layer l from: the
 *                         mapped host record host_ptr[e,l], or address(s) | c of the slot s of
 *                         pool c that holds it, with c = tag(host_ptr[e,l]) = host_ptr[e,l] & 1;
 *     host_ptr   [512,L]  the device-mapped host record of every expert, codec in bit 0; read
 *                         only;
 *     fill_ptr   [512,L]  the untagged slot address moe_expert_cache_fill copies the record to,
 *                         or 0 for no fill.
 *   With U the set of ids in ids [10,T] and l = layer, a slot of a pool is "protected" when it is
 *   borrowed (owner -2) or its owner is l * 512 + u for some u in U (slots taken below become
 *   protected), and K = cache.reference_limit. Then:
 *   1. Hit: each u in U with expert_ptr[u,l] != host_ptr[u,l] sets referenced[s] =
 *      min(referenced[s] + 1, K) for the slot s = slot_index[u,l] that holds it (1 when K = 1).
 *   2. Frequency (when cache.frequency is present): counts[u,l] = min(counts[u,l] + 1, 2^31 - 1)
 *      for every u in U, then lookups += |U|; when lookups >= halving_lookups, every count of
 *      every layer is halved (floor) and lookups = 0.
 *   3. Miss: each u in U with expert_ptr[u,l] == host_ptr[u,l], in ascending u, in pool c =
 *      tag(host_ptr[u,l]) with N slots: if N = 0 or every slot is protected, u is not admitted
 *      and the pool is unchanged. Otherwise CLOCK picks a victim: repeat { s = hand; hand =
 *      (hand + 1) mod N; if s is protected continue; if referenced[s] > 0 { referenced[s] -= 1;
 *      continue } stop }. With MoeCacheAdmission::FrequencyFilter, when owner[s] = o >= 0 and
 *      counts[u,l] <= counts[o mod 512, o / 512] (after step 2), u is not admitted and slot s is
 *      left unchanged (the hand stays advanced). Otherwise, when owner[s] = o >= 0,
 *      expert_ptr[o mod 512, o / 512] = host_ptr[o mod 512, o / 512] and slot_index[o mod 512,
 *      o / 512] = -1; then owner[s] = l * 512 + u, referenced[s] = 1, slot_index[u,l] = s and
 *      fill_ptr[u,l] = address(s), a fixed or a flex slot alike.
 *   expert_ptr[u,l] of an admitted miss keeps its host address until moe_expert_cache_commit; a
 *   miss that is not admitted is read by moe_experts through its mapped host record.
 *   4. Counters (when counters is given), with the classification of steps 1 and 3 taken on
 *      entry:
 *      counters[Lookups + 2 c] += the number of u in U with tag(host_ptr[u,l]) = c, and
 *      counters[Misses + 2 c] += the number of those with expert_ptr[u,l] == host_ptr[u,l],
 *      admitted or not. U is a set: an expert routed by several (token, k) pairs of the call
 *      counts once, matching the one record read per miss that the fill or moe_experts makes.
 *      The sums wrap modulo 2^64; the caller zeroes the counters when it starts a measurement.
 *
 * Logical shapes:
 *   ids I32 [10,T], T >= 1 with 10 T < 2^31, every id in [0,512); expert_ptr, host_ptr and
 *   fill_ptr I64 [512,L], L >= 1 with 512 L < 2^31; 0 <= layer < L; counters, when not null,
 *   contiguous I64 [2,2] (kind, codec); frequency counts, when present, contiguous I32 [512,L]
 *   with lookups contiguous I32 [1]; cache.slot_index contiguous I32 [512,L].
 *
 * Supported domain:
 *   Exact integer and address arithmetic. On entry the state is consistent: every fill_ptr entry
 *   is 0; every referenced value is in [0, K]; every owner code occurs in at most one slot, of
 *   pool c = tag(host_ptr[e,l]) for owner code l * 512 + e, and slot_index is the inverse of the
 *   owners (MoeExpertCache); expert_ptr[e,l] is address(slot_index[e,l]) | c when slot_index[e,l]
 *   >= 0 and host_ptr[e,l] otherwise; host records lie outside every pool. Violations (device
 *   values the Op cannot check on the host) are undefined behavior.
 *
 * Effects:
 *   Writes expert_ptr only for evicted owners (any layer), slot_index only for evicted owners and
 *   admitted misses, fill_ptr only in column layer, owner, referenced and clock_hands of the
 *   pools, the frequency state when present, and the counters when given; nothing else. Between
 *   admit and commit an admitted miss already has its slot_index entry while its expert_ptr entry
 *   still names the host record.
 *   ids and host_ptr are read only. The state is consistent again after the commit. Invariants
 *   the schedule relies on:
 *   - a slot evicted by this call is never read by this layer's moe_experts: its owner is not
 *     routed here and its table entry already names the host record;
 *   - pools are global across layers: a victim may belong to any layer, whose earlier kernels
 *     precede this call in stream order and whose next call reads the host record;
 *   - a slot is published in expert_ptr only by moe_expert_cache_commit, which runs after the
 *     moe_expert_cache_fill that wrote it.
 *   Required order per layer call, with no other admit in between: (1) admit; (2)
 *   moe_expert_cache_fill, on the same stream or on a second stream that waits for an event
 *   recorded after admit; (3) either commit and then moe_experts with this layer's column of
 *   expert_ptr as its expert table (reading every admitted record from its slot), or that
 *   moe_experts with a MoeExpertsDeferral of this layer's fill_ptr column (or of its
 *   moe_expert_cache_defer merge with a staging column, which reads the fill column before commit)
 *   and an event recorded after the fill, and then commit. Commit always follows the fill. All
 *   tensors and pool metadata are pairwise disjoint device memory. Calls sharing counters run in
 *   stream order (one stream): the single CTA adds each counter with one plain read-modify-write,
 *   no atomics.
 *
 * Workspace: none.
 *
 * Execution:
 *   Asynchronous on stream with a fixed launch shape and no host round trip; graph-capturable
 *   for fixed T, layer and views. Deterministic. Invalid arguments throw std::invalid_argument.
 */
void moe_expert_cache_admit(const Tensor& ids, Tensor& expert_ptr, const Tensor& host_ptr,
                            Tensor& fill_ptr, const MoeExpertCache& cache, std::int32_t layer,
                            MoeCacheAdmission admission, cudaStream_t stream,
                            Tensor* counters = nullptr);

/**
 * Op: moe_expert_cache_borrow / moe_expert_cache_return
 *
 * Lends the slots s0 + [first, first + count) of pool c = tag(codec) out of the cache, where s0 =
 * 0 for MoeCacheSlotRegion::Fixed and s0 = F for MoeCacheSlotRegion::Flex: a fixed run's bytes,
 * slots.data + first * record_bytes onward, form one streaming window; a flex run goes back to the
 * owner of the memory behind its addresses. borrow evicts each slot's owner o >= 0
 * (expert_ptr[o mod 512, o / 512] = host_ptr[o mod 512, o / 512] and slot_index[o mod 512, o /
 * 512] = -1) and sets owner = kMoeCacheBorrowedSlot and referenced = 0. While borrowed a slot is
 * protected: no admit takes it, and nothing reads it through the tables. return gives the slots
 * back empty (owner = kMoeCacheEmptySlot, referenced = 0). Nothing else changes; the hand stays
 * where it is. A record is a copy of its immutable host record, so an eviction loses nothing.
 *
 * Preconditions: 1 <= count and first + count <= F (Fixed) or X (Flex) of the pool, which the
 * host checks; for borrow no slot of the range is borrowed, for return every slot of it is; the
 * cache state is consistent (admit's domain) and no fill is pending. expert_ptr and host_ptr are
 * I64 [512,L] and cache.slot_index I32 [512,L] as for admit, pairwise disjoint and disjoint from
 * the pool's owner and referenced arrays. The caller orders every reader of an
 * evicted record before borrow and every other user of the lent bytes between borrow and return
 * (stream order or events). Grid-stride over the range; no workspace; asynchronous and
 * graph-capturable; invalid arguments (as far as the host can see them) throw
 * std::invalid_argument.
 */
void moe_expert_cache_borrow(Tensor& expert_ptr, const Tensor& host_ptr,
                             const MoeExpertCache& cache, MoeExpertCodec codec,
                             MoeCacheSlotRegion region, std::int32_t first, std::int32_t count,
                             cudaStream_t stream);

void moe_expert_cache_return(const MoeExpertCache& cache, MoeExpertCodec codec,
                             MoeCacheSlotRegion region, std::int32_t first, std::int32_t count,
                             cudaStream_t stream);

/**
 * CTAs of one moe_expert_cache_fill launch for a device with `multiprocessors` SMs: one per SM.
 * On the RTX 5090 at PCIe Gen5 x16 one CTA per SM copies at the mapped-read roof: 47-49 GB/s for
 * 16 to 512 F3 records and 45 GB/s for three, against 51-54 GB/s for a copy-engine cudaMemcpy of
 * the same bytes. Two or four CTAs per SM, or 2, 8 or 16 loads in flight per thread, measure the
 * same or up to 3 GB/s less except at three records, so the fill's resident CTAs leave each SM
 * room for the kernels that run beside it.
 */
[[nodiscard]] constexpr std::int32_t moe_expert_cache_fill_blocks(std::int32_t multiprocessors) {
    return multiprocessors;
}

/**
 * CTAs of a staging fill (the records moe_expert_cache_stage places), which copies while another
 * layer's dense kernels run. On the RTX 5090 at PCIe Gen5 x16, 12 CTAs copy F3 records at 44-52
 * GB/s, within reach of the mapped-read roof, while 4 CTAs fall below it; the other SMs stay free
 * for the kernels beside the copy.
 */
inline constexpr std::int32_t kMoeStagingFillBlocks = 12;

/**
 * Op: moe_expert_cache_fill
 *
 * For every e in [0,512) with fill_ptr[e,layer] != 0, copies the moe_expert_record_bytes(c) bytes
 * of the source record at source_ptr[e,layer] with its codec tag c = source_ptr[e,layer] & 1
 * cleared, to the destination address fill_ptr[e,layer]. A source record is device-mapped pinned
 * host memory or device memory, read the same way: a host table, or the source column of
 * moe_expert_cache_stage, which names staged VRAM copies of host records. Writes only those
 * destination bytes; the tables are read only. The copy is exact.
 *
 * Shapes: source_ptr and fill_ptr are disjoint I64 [512,L] as for moe_expert_cache_admit's tables,
 * 0 <= layer < L. Column layer of fill_ptr comes from moe_expert_cache_admit (fixed or flex slots
 * of the codec's pool), moe_expert_cache_window (window placements) or moe_expert_cache_stage
 * (staging placements), the last two passed with a source column as single-layer tables: every
 * nonzero entry is a 256-byte aligned destination holding at least moe_expert_record_bytes(c)
 * bytes, disjoint from every other entry's destination, from every source record read and from
 * every table, and no enqueued work reads or writes those destinations until this call completes;
 * every source record read stays unchanged until then. The schedule guarantees this: admit assigns
 * only slots whose owners this layer does not route, commit and a moe_experts deferral wait for
 * this call, a window or staging span is refilled only after its previous readers are done, and a
 * staged source is rewritten only by a later staging fill.
 *
 * Grid: `blocks` CTAs of 256 threads (moe_expert_cache_fill_blocks(SM count) when not given, a
 * value in [1, 1024] otherwise), independent of how many fills are pending, so the launch is
 * graph-capturable and an empty column costs one table scan per CTA. Each CTA reads the column's
 * 512 fill entries and forms in shared memory the exclusive prefix of the pending records' 16-byte
 * words, the records concatenated in ascending e. The grid strides over that word range with four
 * independent 16-byte loads in flight per thread (consecutive threads take consecutive words, so a
 * warp reads 512 contiguous source bytes per load except where its words straddle two records),
 * finds each word's record by binary search of the prefix, and stores the word at the same offset
 * of its destination. Pending fills of the whole layer therefore share every CTA, and mapped reads
 * keep enough bytes in flight to run at PCIe rate.
 *
 * No workspace; asynchronous on stream, deterministic, and graph-capturable for fixed views, layer
 * and grid. Invalid arguments throw std::invalid_argument.
 */
void moe_expert_cache_fill(const Tensor& source_ptr, const Tensor& fill_ptr, std::int32_t layer,
                           cudaStream_t stream, std::optional<std::int32_t> blocks = std::nullopt);

/**
 * Op: moe_expert_cache_window
 *
 * Places one layer's non-resident records in a streaming window, for a chunk that routes to nearly
 * every expert and admits none. With l = layer, let N be the experts e in ascending order whose
 * expert_ptr[e,l] == host_ptr[e,l] (not resident in a slot), n_e = moe_expert_record_bytes(c) for
 * c = tag(host_ptr[e,l]), and p_e = sum of n_e' over the e' in N below e. Then window_ptr[e] =
 * window.data + p_e for e in N with p_e + n_e <= window.bytes, and 0 for every other e (resident,
 * or past the window, which leaves the record on its mapped host address). The placed records
 * therefore pack the window from its start in ascending e, each 256-byte aligned (record sizes are
 * multiples of 256).
 *
 * Shapes: expert_ptr and host_ptr I64 [512,L] as for moe_expert_cache_admit, 0 <= layer < L;
 * window_ptr contiguous I64 [512], 8-byte aligned, disjoint from both tables; window.data
 * 256-byte aligned, or null with window.bytes == 0 (every entry 0). The tables are read only.
 *
 * Consumers: window_ptr has the layout of a fill_ptr column, so moe_expert_cache_fill with
 * host_ptr's column l and window_ptr as single-layer tables copies the placed records, and
 * moe_experts reads them through a MoeExpertsDeferral of window_ptr after that fill.
 *
 * Grid: one CTA of 512 threads; thread e computes n_e for a non-resident e and the block scan of
 * the 512 sizes gives p_e. Exact; deterministic; asynchronous on stream and graph-capturable for
 * fixed views, layer and window. Invalid arguments throw std::invalid_argument.
 */
void moe_expert_cache_window(const Tensor& expert_ptr, const Tensor& host_ptr, std::int32_t layer,
                             DeviceSpan window, Tensor& window_ptr, cudaStream_t stream);

// Candidates per token moe_expert_cache_stage accepts: a prefix of moe_route's ten ranked ids.
inline constexpr std::int32_t kMoeStageMaxCandidates = kMoeTopK;

/**
 * Op: moe_expert_cache_stage
 *
 * Chooses which non-resident records of layer l = layer a staging fill copies into VRAM before the
 * layer routes, from a prediction of its routes. With k = candidates and the exact integer weights
 * W_j = 360360 / (4 + j), j in [0,10) (that is, 1 / (4 + j) scaled by lcm(4..13) = 360360):
 *
 *   score[e]   = sum of W_j over every (j, t) with j < k, t < T and predicted[j,t] = e (ids outside
 *                [0,512) contribute nothing);
 *   resident e : expert_ptr[e,l] != host_ptr[e,l];
 *   candidates : every non-resident e with score[e] > 0, in descending score, ties to the lower e;
 *   placement  : with used = 0, each candidate e in that order, of n_e = moe_expert_record_bytes(
 *                tag(host_ptr[e,l])) bytes, is placed at offset used when used + n_e <=
 *                budget_bytes (then used += n_e) and is skipped otherwise;
 *   stage_ptr[e] = staging.data + the offset of a placed e, else 0;
 *   source[e]    = stage_ptr[e] | tag(host_ptr[e,l]) for a placed e, else host_ptr[e,l].
 *
 * The weights rank an expert by summed reciprocal ranks, so an expert several tokens predict, or
 * one predicted first, comes before one predicted once at a late rank; integer weights make the
 * order exact. Placed records are 256-byte aligned (record sizes are multiples of 256).
 *
 * Consumers:
 *   1. moe_expert_cache_fill with host_ptr's column l and stage_ptr as single-layer tables copies
 *      the placed records into the staging span (the staging fill);
 *   2. after the staging fill, source is the source column of moe_expert_cache_fill for the
 *      layer's admitted misses (fill_ptr's column l from moe_expert_cache_admit), which copies a
 *      staged record from VRAM and any other from its host record;
 *   3. moe_expert_cache_defer merges stage_ptr into the layer's deferral, through which
 *      moe_experts reads the staged records that admit did not admit.
 *   Placement uses the residency on entry. Every source entry names bytes equal to the host
 *   record, so the admitted fill is exact for any admission; when the schedule changes no
 *   residency of layer l between this call and the layer's admit, every staged record moe_experts
 *   reads through the deferral is a miss, since admit never evicts a routed expert. This Op reads
 *   the tables and writes neither, so admit, commit and the replacement state evolve exactly as
 *   without staging. stage_ptr, source and the staging bytes stay unchanged until those consumers
 *   complete.
 *
 * Logical shapes: predicted I32 [10,T], 1 <= T <= 8192 (kMoeMaxTokens), contiguous;
 * 1 <= candidates <= kMoeStageMaxCandidates; expert_ptr and host_ptr I64 [512,L] as for
 * moe_expert_cache_admit, 0 <= layer < L; stage_ptr and source contiguous I64 [512], 8-byte
 * aligned; staging.data 256-byte aligned with budget_bytes <= staging.bytes, or null with
 * staging.bytes == budget_bytes == 0 (nothing is placed). Every tensor and the staging span are
 * pairwise disjoint; the tables and predicted are read only.
 *
 * Grid: one CTA of 512 threads. Integer shared-memory atomics accumulate the scores (exact and
 * independent of order); thread e then ranks its candidate by counting the candidates ahead of it,
 * one thread walks the ranked candidates while a record still fits, and thread e writes entry e of
 * both outputs. Deterministic; no workspace; asynchronous on stream and graph-capturable for fixed
 * views, layer, candidates, staging and budget. Invalid arguments throw std::invalid_argument.
 */
void moe_expert_cache_stage(const Tensor& predicted, std::int32_t candidates,
                            const Tensor& expert_ptr, const Tensor& host_ptr, std::int32_t layer,
                            DeviceSpan staging, std::size_t budget_bytes, Tensor& stage_ptr,
                            Tensor& source, cudaStream_t stream);

/**
 * Op: moe_expert_cache_defer
 *
 * The deferral slots of a staged layer's moe_experts (MoeExpertsDeferral::slots): for every e in
 * [0,512), deferred[e] = fill_ptr[e,layer] when it is nonzero (an admitted miss, read from the slot
 * its fill writes), else stage_ptr[e] (a staged record, or 0 for an expert read through the expert
 * table). moe_experts with expert_ptr's column layer as its table then reads every routed expert
 * from bytes equal to its host record: hits from their slots, admitted misses from their slots,
 * staged misses from the staging span, and the rest from their mapped host records, so its output
 * equals the unstaged call's bit for bit (the deferral contract). The deferral's ready event
 * follows both the admitted fill and the staging fill.
 *
 * Shapes: fill_ptr I64 [512,L] as for moe_expert_cache_admit, 0 <= layer < L, its column final
 * (after the layer's admit); stage_ptr from moe_expert_cache_stage of the same layer, which
 * completes before this call (stream order or an event); stage_ptr and deferred stay unchanged
 * until the layer's moe_experts completes; deferred contiguous I64 [512]; the three pairwise
 * disjoint and 8-byte aligned. One CTA of 512 threads, thread e writing entry e. Exact; no
 * workspace; asynchronous and graph-capturable for fixed views and layer; invalid arguments throw
 * std::invalid_argument.
 */
void moe_expert_cache_defer(const Tensor& fill_ptr, std::int32_t layer, const Tensor& stage_ptr,
                            Tensor& deferred, cudaStream_t stream);

/**
 * Op: moe_expert_cache_commit
 *
 * For every e in [0,512) with fill_ptr[e,layer] != 0: expert_ptr[e,layer] = fill_ptr[e,layer] |
 * (expert_ptr[e,layer] & 1) (the record's codec, carried by its host address until now) and
 * fill_ptr[e,layer] = 0. Nothing else changes. expert_ptr and fill_ptr are disjoint I64 [512,L]
 * as for moe_expert_cache_admit, 0 <= layer < L. Runs after the moe_expert_cache_fill that wrote
 * the slots completes (same stream, or ordered after it by an event), and after every read of
 * the fill column (a moe_experts deferral or moe_expert_cache_defer). No workspace; asynchronous
 * and graph-capturable; invalid arguments throw std::invalid_argument.
 */
void moe_expert_cache_commit(Tensor& expert_ptr, Tensor& fill_ptr, std::int32_t layer,
                             cudaStream_t stream);

} // namespace ninfer::ops
