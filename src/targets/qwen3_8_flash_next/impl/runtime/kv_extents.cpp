#include "runtime/kv_extents.h"

#include "core/device.h"
#include "ninfer/ops/moe_expert_cache.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ninfer::targets::qwen3_8_flash_next {
namespace {

constexpr auto kFlexCodec = ops::MoeExpertCodec::F3G32;

// Bytes of one page in a PageMajor plane: the stride between consecutive pages.
std::size_t plane_page_bytes(const Tensor& plane) {
    if (plane.nb[3] <= 0) { throw std::logic_error("KvExpertExtents: invalid plane page stride"); }
    return static_cast<std::size_t>(plane.nb[3]);
}

} // namespace

std::uint32_t kv_extent_pages(ops::QsaKvFormat format) {
    switch (format) {
    case ops::QsaKvFormat::NVFP4G16:
        return 244;
    case ops::QsaKvFormat::BF16:
        return 122;
    }
    throw std::invalid_argument("KvExpertExtents: unknown QSA state format");
}

KvExpertExtents::KvExpertExtents(PagedKVPool& pool, ops::QsaKvFormat format, cudaStream_t stream,
                                 cudaStream_t copy_stream)
    : pool_(pool), stream_(stream), copy_stream_(copy_stream),
      extent_pages_(kv_extent_pages(format)) {
    if (pool.plane_order() != PagedKVPlaneOrder::PageMajor) {
        throw std::invalid_argument("KvExpertExtents: the pool must be PageMajor");
    }
    if (pool.mapped_pages() != 0 || pool.entitled_pages() != 0 || pool.withheld_pages() != 0) {
        throw std::invalid_argument("KvExpertExtents: the pool must be idle");
    }
    const auto record = static_cast<std::size_t>(ops::moe_expert_record_bytes(kFlexCodec));
    std::vector<std::size_t> records_per_plane(pool.plane_count());
    std::size_t per_extent = 0;
    for (std::size_t p = 0; p < pool.plane_count(); ++p) {
        records_per_plane[p] =
            std::size_t{extent_pages_} * plane_page_bytes(pool.plane(p)) / record;
        per_extent += records_per_plane[p];
    }
    const std::uint32_t pages = pool.page_group_count();
    // An extent that holds no record lends nothing, so such a geometry has no flex region.
    const std::uint32_t count = per_extent == 0 ? 0 : pages / extent_pages_;
    first_page_               = pages - count * extent_pages_;
    // Slot indices S x + j stay below 2^31 (the Op's I32 owner arrays).
    if (std::size_t{count} * per_extent >
        static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("KvExpertExtents: too many flex slots");
    }
    slots_per_extent_ = static_cast<std::int32_t>(per_extent);

    flex_addresses_.reserve(std::size_t{count} * per_extent);
    for (std::uint32_t x = 0; x < count; ++x) {
        const std::uint32_t first = extent_first_page(x);
        for (std::size_t p = 0; p < pool.plane_count(); ++p) {
            const Tensor& plane     = pool.plane(p);
            const std::size_t bytes = plane_page_bytes(plane);
            const auto base =
                reinterpret_cast<std::uint64_t>(plane.data) + std::uint64_t{first} * bytes;
            for (std::size_t k = 0; k < records_per_plane[p]; ++k) {
                flex_addresses_.push_back(base + k * record);
            }
        }
        pool_.withhold_pages(first, extent_pages_);
    }
    lent_.assign(count, false);
}

std::uint32_t KvExpertExtents::extent_first_page(std::size_t extent) const noexcept {
    return first_page_ + static_cast<std::uint32_t>(extent) * extent_pages_;
}

bool KvExpertExtents::lent(std::int32_t extent) const {
    return lent_.at(static_cast<std::size_t>(extent));
}

std::int32_t KvExpertExtents::lent_extents() const noexcept {
    return static_cast<std::int32_t>(std::count(lent_.begin(), lent_.end(), true));
}

void KvExpertExtents::make_room(std::uint32_t pages, ExpertResidency& cache) {
    bool borrowed = false;
    for (std::size_t x = 0; x < lent_.size() && pool_.free_pages() < pages; ++x) {
        if (lent_[x]) { continue; }
        // A refill reserves an unpublished slot. Join only its bounded copy before borrow
        // changes that reservation or KV overwrites its bytes; no host synchronization.
        cache.join_refill(stream_);
        Tensor expert_table = cache.expert_table();
        ops::moe_expert_cache_borrow(expert_table, cache.host_table(), cache.cache(), kFlexCodec,
                                     ops::MoeCacheSlotRegion::Flex,
                                     static_cast<std::int32_t>(x) * slots_per_extent_,
                                     slots_per_extent_, stream_);
        pool_.restore_pages(extent_first_page(x), extent_pages_);
        lent_[x] = true;
        borrowed = true;
    }
    // A scan can stop with every eligible record resident but high flex slots still empty.
    // Borrowing a lower extent can evict those records while leaving those high slots lent to
    // the cache. Retry only after a real borrow, and only when cache-held extents remain.
    if (borrowed && std::ranges::any_of(lent_, [](bool lent) { return !lent; })) {
        cache.request_refill();
    }
}

void KvExpertExtents::record_tier_copy(cudaEvent_t done) const {
    CUDA_CHECK(cudaStreamWaitEvent(copy_stream_, done, 0));
    CUDA_CHECK(cudaEventRecord(done, copy_stream_));
}

void KvExpertExtents::wait_tier_copy(cudaEvent_t done) const {
    CUDA_CHECK(cudaStreamWaitEvent(stream_, done, 0));
}

void KvExpertExtents::give_back(ExpertResidency& cache) {
    // Lending x withholds E free pages, so at least E stay free while 2 E are free before it.
    for (std::size_t x = lent_.size(); x-- > 0 && pool_.free_pages() >= 2 * extent_pages_;) {
        if (!lent_[x] || !pool_.pages_free(extent_first_page(x), extent_pages_)) { continue; }
        ops::moe_expert_cache_return(cache.cache(), kFlexCodec, ops::MoeCacheSlotRegion::Flex,
                                     static_cast<std::int32_t>(x) * slots_per_extent_,
                                     slots_per_extent_, stream_);
        pool_.withhold_pages(extent_first_page(x), extent_pages_);
        lent_[x] = false;
        cache.request_refill();
    }
}

} // namespace ninfer::targets::qwen3_8_flash_next
