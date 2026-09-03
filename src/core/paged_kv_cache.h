#pragma once

#include "core/layout.h"
#include "core/tensor.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace ninfer {

inline constexpr std::int32_t kPagedKVPageSize = 64;

/**
 * Non-owning, single-sequence view consumed by growing-cache Ops.
 *
 * Physical plane order is fixed by the owning homogeneous pool and validated by the consuming
 * Op. block_table is one contiguous I32 logical-block row.
 */
struct PagedKVLayerView {
    Tensor k_pages;
    Tensor v_pages;
    Tensor k_scale_pages;
    Tensor v_scale_pages;
    // Per-page (64-key tile) dequantized K mean for the Sparge meansim proxy
    // [4][64][kv_heads][pages]; sum[d] = (page_offset=d/4)*4 + d%4. Present when
    // sage_pv or keep_frac<1 on NVFP4; unused by XAttention.
    Tensor k_mean_pages;
    Tensor block_table;
    std::int32_t head_dim     = 0;
    std::int32_t num_kv_heads = 0;
    DType dtype               = DType::BF16;
    std::int32_t quant_group  = 0;
    // Sage3-style FP4-PV compute recipe (NVFP4 storage only).
    bool sage_pv               = false;
};

/**
 * Non-owning multi-sequence view consumed by batched growing-cache Ops.
 *
 * Physical planes and the complete block-table matrix are shared by every logical row in one
 * invocation. block_tables is contiguous I32 [logical_pages, table_rows]; the consuming Op
 * receives its per-row table selectors separately.
 */
struct PagedKVBatchLayerView {
    Tensor k_pages;
    Tensor v_pages;
    Tensor k_scale_pages;
    Tensor v_scale_pages;
    Tensor k_mean_pages;
    Tensor block_tables;
    std::int32_t head_dim     = 0;
    std::int32_t num_kv_heads = 0;
    DType dtype               = DType::BF16;
    std::int32_t quant_group  = 0;
    // Sage3-style FP4-PV compute recipe (NVFP4 storage only).
    bool sage_pv               = false;
};

// A pool plane is storage-only. Consumers assign K/V/layer meaning to plane indices.
struct PagedKVPlaneSpec {
    DType dtype                 = DType::BF16;
    std::int32_t leading_extent = 0;
    std::int32_t head_extent    = 0;
    std::size_t alignment       = 256;
};

enum class PagedKVPlaneOrder : std::uint8_t {
    PageMajor,
    HeadMajor,
};

struct PagedKVPoolSpec {
    std::uint32_t page_group_count      = 0;
    std::uint32_t logical_page_capacity = 0;
    std::int32_t table_rows             = 0;
    PagedKVPlaneOrder plane_order       = PagedKVPlaneOrder::PageMajor;
    std::vector<PagedKVPlaneSpec> planes;
};

struct PagedKVPlaneLayout {
    PagedKVPlaneSpec spec;
    TensorRegion storage;
};

struct PagedKVPoolLayout {
    PagedKVPoolSpec spec;
    std::vector<PagedKVPlaneLayout> planes;
    TensorRegion block_tables;

    [[nodiscard]] std::size_t payload_bytes() const noexcept;
    [[nodiscard]] std::size_t metadata_bytes() const noexcept;
};

// Immutable device scatter geometry for one packed logical page. The caller owns the device copy
// of this table and the packed device staging buffer; the raw transfer path allocates neither.
struct PagedKVScatterPlane {
    unsigned char* base       = nullptr;
    std::uint64_t packed_offset = 0;
    std::uint64_t packed_bytes  = 0;
    std::int64_t page_stride    = 0;
    std::int64_t row_pitch      = 0;
    std::uint32_t row_bytes     = 0;
    std::uint32_t rows          = 0;
};

struct PagedKVScatterPlan {
    std::vector<PagedKVScatterPlane> planes;
    std::size_t page_bytes      = 0;
    std::size_t max_plane_bytes = 0;
};

[[nodiscard]] PagedKVPoolLayout plan_paged_kv_pool(LayoutBuilder& builder,
                                                   const PagedKVPoolSpec& spec);

[[nodiscard]] std::uint32_t pages_for_tokens(std::uint32_t tokens) noexcept;

class PagedKVAllocation;
struct PagedKVResize;

class PagedKVPool {
public:
    PagedKVPool(DeviceSpan backing, const PagedKVPoolLayout& layout);

    PagedKVPool(const PagedKVPool&)            = delete;
    PagedKVPool& operator=(const PagedKVPool&) = delete;
    PagedKVPool(PagedKVPool&&)                 = delete;
    PagedKVPool& operator=(PagedKVPool&&)      = delete;

    [[nodiscard]] std::uint32_t page_group_count() const noexcept;
    [[nodiscard]] std::uint32_t logical_page_capacity() const noexcept;
    [[nodiscard]] std::int32_t table_row_count() const noexcept;
    [[nodiscard]] std::size_t plane_count() const noexcept;
    [[nodiscard]] PagedKVPlaneOrder plane_order() const noexcept;
    [[nodiscard]] const Tensor& plane(std::size_t index) const;
    [[nodiscard]] const Tensor& block_tables() const noexcept;
    [[nodiscard]] Tensor block_table_row(std::int32_t row) const;

    [[nodiscard]] std::uint32_t entitled_pages() const noexcept;
    [[nodiscard]] std::uint32_t mapped_pages() const noexcept;
    [[nodiscard]] std::uint32_t free_pages() const noexcept;
    [[nodiscard]] bool can_reserve(std::uint32_t page_entitlement) const noexcept;
    [[nodiscard]] PagedKVAllocation reserve(std::uint32_t page_entitlement);

    // Zeros only the named physical page groups across every storage plane.
    void zero_pages(std::span<const std::int32_t> page_ids, cudaStream_t stream = nullptr);

private:
    friend class PagedKVAllocation;
    friend void resize_paged_kv_bundle(std::span<const PagedKVResize> changes);

    [[nodiscard]] bool can_replace_entitlement(std::uint32_t old_pages,
                                               std::uint32_t new_pages) const noexcept;
    [[nodiscard]] std::vector<std::int32_t> take_pages(std::uint32_t count,
                                                       std::int32_t preferred_first);
    void return_pages(std::span<const std::int32_t> pages) noexcept;
    void add_entitlement(std::uint32_t pages) noexcept;
    void replace_entitlement(std::uint32_t old_pages, std::uint32_t new_pages) noexcept;
    void acquire_row(std::int32_t row);
    void release_row(std::int32_t row) noexcept;

    PagedKVPoolSpec spec_;
    std::vector<Tensor> planes_;
    Tensor block_tables_;
    std::vector<std::int32_t> free_page_ids_;
    std::vector<bool> row_in_use_;
    std::uint32_t entitled_pages_ = 0;
    std::uint32_t mapped_pages_   = 0;
};

class PagedKVAllocation {
public:
    PagedKVAllocation() noexcept = default;
    ~PagedKVAllocation();

    PagedKVAllocation(const PagedKVAllocation&)            = delete;
    PagedKVAllocation& operator=(const PagedKVAllocation&) = delete;
    PagedKVAllocation(PagedKVAllocation&& other) noexcept;
    PagedKVAllocation& operator=(PagedKVAllocation&& other) noexcept;

    [[nodiscard]] bool valid() const noexcept;
    [[nodiscard]] std::uint32_t page_entitlement() const noexcept;
    [[nodiscard]] std::uint32_t mapped_page_count() const noexcept;
    [[nodiscard]] std::uint32_t mapped_token_capacity() const noexcept;
    [[nodiscard]] std::int32_t bound_row() const noexcept;
    [[nodiscard]] std::span<const std::int32_t> page_ids() const noexcept;
    [[nodiscard]] bool belongs_to(const PagedKVPool& pool) const noexcept;

    void set_page_entitlement(std::uint32_t pages);
    void cancel_unmapped_entitlement() noexcept;
    void materialize_pages(std::uint32_t pages, cudaStream_t stream = nullptr);
    void materialize_tokens(std::uint32_t tokens, cudaStream_t stream = nullptr);
    void trim_pages(std::uint32_t pages);
    void trim_tokens(std::uint32_t tokens);

    void bind_row(std::int32_t row, cudaStream_t stream = nullptr);
    void publish_mapping(cudaStream_t stream = nullptr) const;
    void unbind_row() noexcept;
    [[nodiscard]] Tensor block_table() const;

    void release() noexcept;

private:
    friend class PagedKVPool;
    friend void resize_paged_kv_bundle(std::span<const PagedKVResize> changes);

    PagedKVAllocation(PagedKVPool& pool, std::uint32_t page_entitlement);
    void publish_range(std::uint32_t first_page, std::uint32_t page_count,
                       cudaStream_t stream) const;

    PagedKVPool* pool_ = nullptr;
    std::vector<std::int32_t> page_ids_;
    std::uint32_t page_entitlement_ = 0;
    std::int32_t bound_row_         = -1;
};

struct PagedKVReservation {
    PagedKVPool* pool              = nullptr;
    std::uint32_t page_entitlement = 0;
};

// Reserves every requested pool or leaves all pools unchanged.
[[nodiscard]] std::vector<PagedKVAllocation>
reserve_paged_kv_bundle(std::span<const PagedKVReservation> reservations);

struct PagedKVResize {
    PagedKVAllocation* allocation  = nullptr;
    std::uint32_t mapped_pages     = 0;
    std::uint32_t page_entitlement = 0;
};

// Atomically validates a retained-claim/truncate resize vector, then applies it at a GPU boundary.
void resize_paged_kv_bundle(std::span<const PagedKVResize> changes);

[[nodiscard]] std::size_t paged_kv_host_image_bytes(const PagedKVPool& pool,
                                                    std::uint32_t page_count);

void pack_paged_kv_allocation_to_host(const PagedKVAllocation& allocation, const PagedKVPool& pool,
                                      void* dst, cudaStream_t stream);

void unpack_paged_kv_allocation_from_host(PagedKVAllocation& allocation, const PagedKVPool& pool,
                                          const void* src, std::uint32_t src_page_count,
                                          std::uint32_t dst_extent, cudaStream_t stream);

[[nodiscard]] std::size_t paged_kv_logical_page_bytes(const PagedKVPool& pool);

void pack_paged_kv_logical_page_to_host(const PagedKVAllocation& allocation, const PagedKVPool& pool,
                                        std::uint32_t logical_index, void* dst, cudaStream_t stream);

void unpack_paged_kv_logical_page_from_host(PagedKVAllocation& allocation, const PagedKVPool& pool,
                                            const void* src, std::uint32_t logical_index,
                                            cudaStream_t stream);

[[nodiscard]] PagedKVScatterPlan make_paged_kv_scatter_plan(const PagedKVPool& pool);

void scatter_paged_kv_logical_page_from_device(
    PagedKVAllocation& allocation, const PagedKVPool& pool, const void* device_staging,
    const PagedKVScatterPlane* device_planes, std::size_t plane_count,
    std::size_t max_plane_bytes, std::uint32_t logical_index, cudaStream_t stream);

void gather_logical_page_from_host_image(const void* image, const PagedKVPool& pool,
                                         std::uint32_t src_page_count, std::uint32_t logical_index,
                                         void* dst);

void logical_page_host_slices(const void* image, const PagedKVPool& pool,
                              std::uint32_t src_page_count, std::uint32_t logical_index,
                              std::vector<std::pair<const void*, std::size_t>>& out);

} // namespace ninfer
