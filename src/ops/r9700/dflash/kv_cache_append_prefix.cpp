#include "ninfer/ops/kv_cache_append_prefix.h"
#include "ninfer/types.h"

#include "core/paged_kv_cache.h"
#include "ops/r9700/dflash/kv_cache_append_prefix_launch.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <span>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kHeadDim       = 128;
constexpr std::int32_t kKvHeads       = 8;
constexpr std::int32_t kDFlash2Window = 2048;

void require_shape(const Tensor& tensor, std::int32_t n0, std::int32_t n1, std::int32_t n2,
                   std::int32_t n3, const char* label) {
    if (tensor.ne[0] != n0 || tensor.ne[1] != n1 || tensor.ne[2] != n2 || tensor.ne[3] != n3) {
        throw std::invalid_argument(std::string("kv_cache_append_prefix: invalid shape for ") +
                                    label);
    }
}

void require_tensor(const Tensor& tensor, DType dtype, const char* label) {
    if (tensor.dtype != dtype || tensor.data == nullptr || !tensor.is_contiguous()) {
        throw std::invalid_argument(std::string("kv_cache_append_prefix: invalid ") + label);
    }
}

void require_vector_alignment(const Tensor& tensor, const char* label) {
    if ((reinterpret_cast<std::uintptr_t>(tensor.data) & 15U) != 0U) {
        throw std::invalid_argument(std::string("kv_cache_append_prefix: unaligned ") + label);
    }
}

bool overlaps(const Tensor& first, const Tensor& second) {
    const auto first_address  = reinterpret_cast<std::uintptr_t>(first.data);
    const auto second_address = reinterpret_cast<std::uintptr_t>(second.data);
    if (first_address <= second_address) { return second_address - first_address < first.bytes(); }
    return first_address - second_address < second.bytes();
}

void require_disjoint(std::span<const Tensor* const> tensors) {
    for (std::size_t first = 0; first < tensors.size(); ++first) {
        for (std::size_t second = first + 1; second < tensors.size(); ++second) {
            if (overlaps(*tensors[first], *tensors[second])) {
                throw std::invalid_argument("kv_cache_append_prefix: operands overlap");
            }
        }
    }
}

void validate_common(const Tensor& key, const Tensor& value, const Tensor& positions,
                     const Tensor& counts, const Tensor& selectors,
                     KVCacheAppendPrefixExecutionEnvelope envelope, hipStream_t stream) {
    if (stream == nullptr) {
        throw std::invalid_argument("kv_cache_append_prefix: stream must be non-null");
    }
    require_tensor(key, DType::BF16, "key");
    require_tensor(value, DType::BF16, "value");
    require_tensor(positions, DType::I32, "positions");
    require_tensor(counts, DType::I32, "counts");
    require_tensor(selectors, DType::I32, "selectors");
    require_vector_alignment(key, "key");
    require_vector_alignment(value, "value");
    const std::int32_t width = key.ne[2];
    const std::int32_t batch = key.ne[3];
    if (width <= 0 || batch <= 0 || batch > static_cast<std::int32_t>(kDFlashMaximumConcurrency) ||
        envelope.min_count > envelope.max_count ||
        envelope.max_count > static_cast<std::uint32_t>(width)) {
        throw std::invalid_argument("kv_cache_append_prefix: invalid width, batch, or envelope");
    }
    require_shape(key, kHeadDim, kKvHeads, width, batch, "key");
    require_shape(value, kHeadDim, kKvHeads, width, batch, "value");
    require_shape(positions, width, batch, 1, 1, "positions");
    require_shape(counts, batch, 1, 1, 1, "counts");
    require_shape(selectors, batch, 1, 1, 1, "selectors");
    require_disjoint(std::array<const Tensor*, 5>{&key, &value, &positions, &counts, &selectors});
}

void validate_paged(const KVCacheAppendPrefixPagedView& cache) {
    require_tensor(cache.key_pages, DType::BF16, "paged key storage");
    require_tensor(cache.value_pages, DType::BF16, "paged value storage");
    require_tensor(cache.block_tables, DType::I32, "paged block tables");
    require_vector_alignment(cache.key_pages, "paged key storage");
    require_vector_alignment(cache.value_pages, "paged value storage");
    const std::int32_t physical_pages = cache.key_pages.ne[2];
    if (cache.head_dim != kHeadDim || cache.num_kv_heads != kKvHeads || physical_pages <= 0 ||
        cache.block_tables.ne[0] <= 0 || cache.block_tables.ne[1] <= 0) {
        throw std::invalid_argument("kv_cache_append_prefix: invalid paged cache geometry");
    }
    require_shape(cache.key_pages, kHeadDim, kPagedKVPageSize, physical_pages, kKvHeads,
                  "paged key storage");
    require_shape(cache.value_pages, kHeadDim, kPagedKVPageSize, physical_pages, kKvHeads,
                  "paged value storage");
    require_shape(cache.block_tables, cache.block_tables.ne[0], cache.block_tables.ne[1], 1, 1,
                  "paged block tables");
    require_disjoint(
        std::array<const Tensor*, 3>{&cache.key_pages, &cache.value_pages, &cache.block_tables});
}

void validate_cyclic(const CyclicKVCacheLayerView& cache) {
    require_tensor(cache.k, DType::BF16, "cyclic key storage");
    require_tensor(cache.v, DType::BF16, "cyclic value storage");
    require_vector_alignment(cache.k, "cyclic key storage");
    require_vector_alignment(cache.v, "cyclic value storage");
    if (cache.head_dim != kHeadDim || cache.num_kv_heads != kKvHeads ||
        cache.capacity != kDFlash2Window || cache.padded_capacity != kDFlash2Window ||
        cache.lane_capacity <= 0) {
        throw std::invalid_argument("kv_cache_append_prefix: invalid DFlash2 cyclic geometry");
    }
    require_shape(cache.k, kHeadDim, kDFlash2Window, kKvHeads, cache.lane_capacity,
                  "cyclic key storage");
    require_shape(cache.v, kHeadDim, kDFlash2Window, kKvHeads, cache.lane_capacity,
                  "cyclic value storage");
    require_disjoint(std::array<const Tensor*, 2>{&cache.k, &cache.v});
}

} // namespace

void kv_cache_append_prefix(const Tensor& k, const Tensor& v, const Tensor& positions,
                            const Tensor& counts, const Tensor& table_rows,
                            KVCacheAppendPrefixExecutionEnvelope envelope,
                            const KVCacheAppendPrefixPagedView& cache, hipStream_t stream) {
    validate_common(k, v, positions, counts, table_rows, envelope, stream);
    validate_paged(cache);
    require_disjoint(std::array<const Tensor*, 8>{&k, &v, &positions, &counts, &table_rows,
                                                  &cache.key_pages, &cache.value_pages,
                                                  &cache.block_tables});
    r9700::dflash::kv_cache_append_prefix_paged_launch(k, v, positions, counts, table_rows, cache,
                                                       envelope, stream);
}

void kv_cache_append_prefix(const Tensor& k, const Tensor& v, const Tensor& positions,
                            const Tensor& counts, const Tensor& lanes,
                            KVCacheAppendPrefixExecutionEnvelope envelope,
                            const CyclicKVCacheLayerView& cache, hipStream_t stream) {
    validate_common(k, v, positions, counts, lanes, envelope, stream);
    validate_cyclic(cache);
    require_disjoint(
        std::array<const Tensor*, 7>{&k, &v, &positions, &counts, &lanes, &cache.k, &cache.v});
    r9700::dflash::kv_cache_append_prefix_cyclic_launch(k, v, positions, counts, lanes, cache,
                                                        envelope, stream);
}

} // namespace ninfer::ops
