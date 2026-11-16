#pragma once

// The shared QSA pool's idle pages as expert-cache slots (qwen3.8-flash-next-plan.md, "KV pages
// as expert-cache slots"): whole extents of consecutive pages that hold no KV are withheld from the
// pool and lent to the F3 slot pool as flex slots, and the KV pool takes them back before it maps
// their pages. A record is a copy of an immutable host record, so taking an extent back only
// evicts.

#include "runtime/cuda_event.h"
#include "runtime/expert_residency.h"

#include "core/paged_kv_cache.h"
#include "ninfer/ops/qsa.h"

#include <cuda_runtime.h> // cudaStream_t

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::targets::qwen3_8_flash_next {

// Pages of one extent. In every plane of a PageMajor pool, an extent's pages are one contiguous run
// of extent pages x the plane's page bytes, and the F3 records that fit those runs are its slots.
// NVFP4 QSA state: 244 pages (15,616 tokens) hold 84 records, two per 16,384-byte code and index
// key plane and one per 8,192-byte block-key plane of each of the 12 layers, 92.1% of the extent's
// 182 MB. BF16: 122 pages (7,808 tokens) hold 108 records, 94.2% of 229 MB. The larger NVFP4
// extent gains 13 points of byte use over 122 pages (36 records, 78.9%) at the cost of coarser
// granularity: about half an extent stays partly used at the KV frontier.
[[nodiscard]] std::uint32_t kv_extent_pages(ops::QsaKvFormat format);

/**
 * Lends the idle pages of the shared QSA pool to the F3 pool of the expert cache, in extents.
 *
 * Geometry. With P pool pages and E = kv_extent_pages(format), the flex region is the top
 * X = floor(P / E) extents: extent x covers pages first_page() + [x E, (x + 1) E), with
 * first_page() = P - X E. Pages below the region are never lent, and make_room restores the extent
 * nearest them first; the pool's allocator prefers a lane's own continuation and then the first
 * contiguous free run, so which free pages a lane maps is its choice, not this class's. Within
 * extent x every plane p of the pool contributes floor(E * page_bytes(p) / R) records of R =
 * moe_expert_record_bytes(F3) bytes at plane(p).data + (first page of x) * page_bytes(p) + k R,
 * plane after plane in pool order. These S records per extent are flex slots S x + j of the F3
 * pool, so flex_addresses() lists S X addresses, each 256-byte aligned (plane bases, page strides
 * and R are multiples of 256), pairwise disjoint and inside one extent's pages.
 *
 * States. Every extent is CacheHeld (its pages withheld from the pool, its slots the cache's) or
 * KvLent (its pages returned to the pool, its slots borrowed out of the cache, owner
 * kMoeCacheBorrowedSlot). Construction starts every extent CacheHeld while the pool is still idle;
 * it allocates no device memory. Invariants: a CacheHeld extent has no mapped page, so no lane's
 * block table names it, and a KvLent extent's slots are never admitted or read through the tables.
 *
 * Ordering. All device work is enqueued on the compute stream in call order. make_room enqueues
 * an extent's borrow before the caller maps its pages, so the borrow follows every enqueued reader
 * of the evicted records and precedes every write of the pages as KV, on this stream and on the
 * copy stream after DeviceContext::order_copy_after_compute. No fill may be pending: the caller
 * runs between Forward calls, whose fills the compute stream has joined. give_back enqueues an
 * extent's return after the stream has waited for the copy stream's work so far, so host-tier
 * transfers that read or wrote its pages as KV, and the block-table unmap of State::close or
 * State::resize, precede every fill of its slots. Disk-tier restores scatter pages on the disk
 * cache's own stream; they target a lane that stays open until the Engine has drained or cancelled
 * them, so a lane's pages are never free while such a copy is in flight. The join makes the
 * compute stream wait for every queued copy, including another lane's restore; it happens only
 * when an extent is actually returned.
 * Not thread-safe; the pool and the residency must outlive this object.
 */
class KvExpertExtents {
public:
    KvExpertExtents(PagedKVPool& pool, ops::QsaKvFormat format, cudaStream_t stream,
                    cudaStream_t copy_stream);

    KvExpertExtents(const KvExpertExtents&)            = delete;
    KvExpertExtents& operator=(const KvExpertExtents&) = delete;

    [[nodiscard]] std::uint32_t extent_pages() const noexcept { return extent_pages_; }

    [[nodiscard]] std::uint32_t first_page() const noexcept { return first_page_; }

    [[nodiscard]] std::int32_t extents() const noexcept {
        return static_cast<std::int32_t>(lent_.size());
    }

    [[nodiscard]] std::int32_t slots_per_extent() const noexcept { return slots_per_extent_; }

    // The S X flex slot addresses in flex index order, for ExpertResidency.
    [[nodiscard]] std::span<const std::uint64_t> flex_addresses() const noexcept {
        return flex_addresses_;
    }

    [[nodiscard]] bool lent(std::int32_t extent) const;
    [[nodiscard]] std::int32_t lent_extents() const noexcept;

    // Takes CacheHeld extents back, lowest first (the one nearest the KV frontier), until the pool
    // has at least `pages` free pages or every extent is lent. Free plus withheld pages cover
    // every materialization the pool's entitlement admits, so a shortfall remains only for growth
    // the pool refuses anyway, which then fails with the pool's own error.
    void make_room(std::uint32_t pages, ExpertResidency& cache);

    // Lends fully free KvLent extents back to the cache, highest first, while at least one extent
    // of free pages remains outside them. That spare extent keeps a lane that grows and shrinks
    // across an extent boundary from taking back and lending the same extent every request; each
    // such cycle would evict its S records, which then cross PCIe again. Churn larger than one
    // extent (a closed lane longer than E pages) still cycles extents, and once taken back one
    // extent stays with the KV pool as the spare; a pool of fewer than 2 E pages never lends its
    // only extent again.
    void give_back(ExpertResidency& cache);

private:
    [[nodiscard]] std::uint32_t extent_first_page(std::size_t extent) const noexcept;

    PagedKVPool& pool_;
    cudaStream_t stream_;
    cudaStream_t copy_stream_;
    std::uint32_t extent_pages_    = 0;
    std::uint32_t first_page_      = 0;
    std::int32_t slots_per_extent_ = 0;
    std::vector<std::uint64_t> flex_addresses_;
    // lent_[x]: extent x is KvLent.
    std::vector<bool> lent_;
    // Recorded on the copy stream by give_back; the compute stream waits for it before a return.
    CudaEvent copies_done_;
};

} // namespace ninfer::targets::qwen3_8_flash_next
