#include "core/paged_kv_cache.h"

#include "core/device.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace ninfer {
namespace {

__global__ void scatter_paged_kv_page(const unsigned char* packed,
                                      const PagedKVScatterPlane* planes,
                                      std::size_t plane_count, std::int32_t page_id,
                                      std::size_t max_plane_bytes) {
    const std::size_t plane_index = blockIdx.y;
    if (plane_index >= plane_count) { return; }
    __shared__ PagedKVScatterPlane plane;
    if (threadIdx.x == 0) { plane = planes[plane_index]; }
    __syncthreads();
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= plane.packed_bytes || index >= max_plane_bytes) { return; }
    const std::size_t row = index / plane.row_bytes;
    const std::size_t col = index - row * plane.row_bytes;
    plane.base[static_cast<std::int64_t>(page_id) * plane.page_stride +
               static_cast<std::int64_t>(row) * plane.row_pitch + col] =
        packed[plane.packed_offset + index];
}

} // namespace

PagedKVScatterPlan make_paged_kv_scatter_plan(const PagedKVPool& pool) {
    PagedKVScatterPlan out;
    out.planes.reserve(pool.plane_count());
    const PagedKVPlaneOrder order = pool.plane_order();
    for (std::size_t i = 0; i < pool.plane_count(); ++i) {
        const Tensor& tensor = pool.plane(i);
        PagedKVScatterPlane plane;
        plane.base          = static_cast<unsigned char*>(tensor.data);
        plane.packed_offset = out.page_bytes;
        if (order == PagedKVPlaneOrder::PageMajor) {
            if (tensor.nb[3] <= 0 || tensor.nb[3] > std::numeric_limits<std::uint32_t>::max()) {
                throw std::logic_error("Paged KV PageMajor scatter geometry is invalid");
            }
            plane.packed_bytes = static_cast<std::uint64_t>(tensor.nb[3]);
            plane.page_stride  = tensor.nb[3];
            plane.row_pitch    = tensor.nb[3];
            plane.row_bytes    = static_cast<std::uint32_t>(tensor.nb[3]);
            plane.rows         = 1;
        } else {
            if (tensor.ne[3] <= 0 || tensor.ne[3] > std::numeric_limits<std::uint32_t>::max() ||
                tensor.nb[2] <= 0 || tensor.nb[2] > std::numeric_limits<std::uint32_t>::max() ||
                tensor.nb[3] <= 0) {
                throw std::logic_error("Paged KV HeadMajor scatter geometry is invalid");
            }
            plane.packed_bytes = static_cast<std::uint64_t>(tensor.ne[3]) * tensor.nb[2];
            plane.page_stride  = tensor.nb[2];
            plane.row_pitch    = tensor.nb[3];
            plane.row_bytes    = static_cast<std::uint32_t>(tensor.nb[2]);
            plane.rows         = static_cast<std::uint32_t>(tensor.ne[3]);
        }
        out.page_bytes += static_cast<std::size_t>(plane.packed_bytes);
        out.max_plane_bytes = std::max(out.max_plane_bytes,
                                       static_cast<std::size_t>(plane.packed_bytes));
        out.planes.push_back(plane);
    }
    return out;
}

void scatter_paged_kv_logical_page_from_device(
    PagedKVAllocation& allocation, const PagedKVPool& pool, const void* device_staging,
    const PagedKVScatterPlane* device_planes, std::size_t plane_count,
    std::size_t max_plane_bytes, std::uint32_t logical_index, cudaStream_t stream) {
    if (!allocation.valid() || !allocation.belongs_to(pool)) {
        throw std::invalid_argument("Paged KV scatter requires an allocation from the named pool");
    }
    if (logical_index >= allocation.mapped_page_count()) {
        throw std::logic_error("Paged KV scatter logical index exceeds mapped pages");
    }
    if (device_staging == nullptr || device_planes == nullptr || plane_count != pool.plane_count() ||
        max_plane_bytes == 0) {
        throw std::invalid_argument("Paged KV scatter staging or geometry is invalid");
    }
    constexpr unsigned threads = 256;
    const unsigned blocks = static_cast<unsigned>((max_plane_bytes + threads - 1) / threads);
    const dim3 grid(blocks, static_cast<unsigned>(plane_count));
    scatter_paged_kv_page<<<grid, threads, 0, stream>>>(
        static_cast<const unsigned char*>(device_staging), device_planes, plane_count,
        allocation.page_ids()[logical_index], max_plane_bytes);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer
