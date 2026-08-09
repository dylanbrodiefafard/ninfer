#include "ninfer/ops/kv_cache_append_prefix.h"

#include "ops/launcher/kv_cache_append_prefix.h"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kHeadDim = 128;
constexpr std::int32_t kKVHeads = 8;
constexpr std::uint32_t kWindow = 4096;

void require_shape(const Tensor& tensor, std::int32_t n0, std::int32_t n1, std::int32_t n2,
                   std::int32_t n3, const char* name) {
    if (tensor.ne[0] != n0 || tensor.ne[1] != n1 || tensor.ne[2] != n2 || tensor.ne[3] != n3) {
        throw std::invalid_argument("kv_cache_append_prefix: invalid shape for " +
                                    std::string(name));
    }
}

void require_contiguous_nonnull(const Tensor& tensor, const char* name) {
    if (!tensor.is_contiguous() || tensor.data == nullptr) {
        throw std::invalid_argument("kv_cache_append_prefix: " + std::string(name) +
                                    " must be contiguous and non-null");
    }
}

void require_vector_aligned(const Tensor& tensor, const char* name) {
    require_contiguous_nonnull(tensor, name);
    if ((reinterpret_cast<std::uintptr_t>(tensor.data) & 15u) != 0u) {
        throw std::invalid_argument("kv_cache_append_prefix: " + std::string(name) +
                                    " must be 16-byte aligned");
    }
}

detail::KVCacheAppendPrefixPlan validate_inputs(const Tensor& k, const Tensor& v,
                                                const Tensor& positions, const Tensor& commit_count,
                                                KVCacheAppendPrefixExecutionEnvelope envelope) {
    if (k.dtype != DType::BF16 || v.dtype != DType::BF16) {
        throw std::invalid_argument("kv_cache_append_prefix: k/v must be BF16");
    }
    if (positions.dtype != DType::I32 || commit_count.dtype != DType::I32) {
        throw std::invalid_argument("kv_cache_append_prefix: positions/count must be I32");
    }
    const std::int32_t tokens = k.ne[2];
    if (tokens < 1) { throw std::invalid_argument("kv_cache_append_prefix: T must be positive"); }
    require_shape(k, kHeadDim, kKVHeads, tokens, 1, "k");
    require_shape(v, kHeadDim, kKVHeads, tokens, 1, "v");
    require_shape(positions, tokens, 1, 1, 1, "positions");
    require_shape(commit_count, 1, 1, 1, 1, "commit_count");
    require_vector_aligned(k, "k");
    require_vector_aligned(v, "v");
    require_contiguous_nonnull(positions, "positions");
    require_contiguous_nonnull(commit_count, "commit_count");
    return detail::kv_cache_append_prefix_resolve_plan(tokens, envelope);
}

void validate_paged_cache(const PagedKVLayerView& cache,
                          KVCacheAppendPrefixExecutionEnvelope envelope) {
    if (cache.dtype != DType::BF16 || cache.quant_group != 0 || cache.num_kv_heads != kKVHeads ||
        cache.head_dim != kHeadDim || cache.k_pages.ne[2] <= 0 ||
        cache.k_pages.ne[2] != cache.v_pages.ne[2] || cache.block_table.ne[0] <= 0 ||
        envelope.max_count >
            static_cast<std::uint32_t>(cache.block_table.ne[0]) * kPagedKVPageSize) {
        throw std::invalid_argument("kv_cache_append_prefix: invalid paged cache");
    }
    const std::int32_t physical_pages = cache.k_pages.ne[2];
    if (cache.k_pages.dtype != DType::BF16 || cache.v_pages.dtype != DType::BF16 ||
        cache.k_pages.ne[0] != kHeadDim || cache.k_pages.ne[1] != kPagedKVPageSize ||
        cache.k_pages.ne[3] != kKVHeads || cache.v_pages.ne[0] != kHeadDim ||
        cache.v_pages.ne[1] != kPagedKVPageSize || cache.v_pages.ne[2] != physical_pages ||
        cache.v_pages.ne[3] != kKVHeads || cache.k_scale_pages.data != nullptr ||
        cache.v_scale_pages.data != nullptr || cache.block_table.dtype != DType::I32 ||
        cache.block_table.ne[1] != 1 || cache.block_table.ne[2] != 1 ||
        cache.block_table.ne[3] != 1) {
        throw std::invalid_argument("kv_cache_append_prefix: invalid paged cache tensors");
    }
    require_vector_aligned(cache.k_pages, "cache k pages");
    require_vector_aligned(cache.v_pages, "cache v pages");
    require_contiguous_nonnull(cache.block_table, "cache block table");
}

void validate_cyclic_cache(const CyclicKVCacheLayerView& cache,
                           KVCacheAppendPrefixExecutionEnvelope envelope) {
    if (cache.num_kv_heads != kKVHeads || cache.head_dim != kHeadDim || cache.capacity != kWindow ||
        cache.padded_capacity < cache.capacity ||
        cache.padded_capacity >
            static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        envelope.max_count > cache.capacity) {
        throw std::invalid_argument("kv_cache_append_prefix: invalid cyclic cache");
    }
    const auto padded = static_cast<std::int32_t>(cache.padded_capacity);
    if (cache.k.dtype != DType::BF16 || cache.v.dtype != DType::BF16 || cache.k.ne[0] != kHeadDim ||
        cache.k.ne[1] != padded || cache.k.ne[2] != kKVHeads || cache.k.ne[3] != 1 ||
        cache.v.ne[0] != kHeadDim || cache.v.ne[1] != padded || cache.v.ne[2] != kKVHeads ||
        cache.v.ne[3] != 1) {
        throw std::invalid_argument("kv_cache_append_prefix: invalid cyclic cache tensors");
    }
    require_vector_aligned(cache.k, "cache k");
    require_vector_aligned(cache.v, "cache v");
}

} // namespace

void kv_cache_append_prefix(const Tensor& k, const Tensor& v, const Tensor& positions,
                            const Tensor& commit_count,
                            KVCacheAppendPrefixExecutionEnvelope envelope, PagedKVLayerView cache,
                            cudaStream_t stream) {
    const auto plan = validate_inputs(k, v, positions, commit_count, envelope);
    validate_paged_cache(cache, envelope);
    detail::kv_cache_append_prefix_launch(k, v, positions, commit_count, cache, plan, stream);
}

void kv_cache_append_prefix(const Tensor& k, const Tensor& v, const Tensor& positions,
                            const Tensor& commit_count,
                            KVCacheAppendPrefixExecutionEnvelope envelope,
                            CyclicKVCacheLayerView cache, cudaStream_t stream) {
    const auto plan = validate_inputs(k, v, positions, commit_count, envelope);
    validate_cyclic_cache(cache, envelope);
    detail::kv_cache_append_prefix_launch(k, v, positions, commit_count, cache, plan, stream);
}

} // namespace ninfer::ops
