#include "ops/launcher/kv_cache_append_prefix.h"

#include "core/device.h"
#include "ops/kernel/kv_cache_append_prefix.cuh"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kBlock = 256;

void validate_plan(const Tensor& k, const KVCacheAppendPrefixPlan& plan) {
    if (plan.tokens != k.ne[2] || plan.min_count < 0 || plan.max_count < plan.min_count ||
        plan.max_count > plan.tokens) {
        throw std::invalid_argument("kv_cache_append_prefix: inconsistent plan");
    }
}

void launch_paged(const Tensor& k, const Tensor& v, const Tensor& positions,
                  const Tensor& commit_count, PagedKVLayerView cache,
                  const KVCacheAppendPrefixPlan& plan, cudaStream_t stream) {
    validate_plan(k, plan);
    if (plan.max_count == 0) return;
    auto* cache_k       = static_cast<__nv_bfloat16*>(cache.k_pages.data);
    auto* cache_v       = static_cast<__nv_bfloat16*>(cache.v_pages.data);
    const auto* input_k = static_cast<const __nv_bfloat16*>(k.data);
    const auto* input_v = static_cast<const __nv_bfloat16*>(v.data);
    const auto* pos     = static_cast<const std::int32_t*>(positions.data);
    const auto* count   = static_cast<const std::int32_t*>(commit_count.data);
    const auto* table   = static_cast<const std::int32_t*>(cache.block_table.data);

    const int grid = 1 + (plan.max_count - 1) / 4;
    kv_cache_append_prefix_paged_kernel<<<grid, kBlock, 0, stream>>>(
        input_k, input_v, pos, count, cache_k, cache_v, table, cache.k_pages.ne[2], plan.min_count,
        plan.max_count);
    CUDA_CHECK(cudaGetLastError());
}

void launch_cyclic(const Tensor& k, const Tensor& v, const Tensor& positions,
                   const Tensor& commit_count, CyclicKVCacheLayerView cache,
                   const KVCacheAppendPrefixPlan& plan, cudaStream_t stream) {
    validate_plan(k, plan);
    if (plan.max_count == 0) return;
    auto* cache_k       = static_cast<__nv_bfloat16*>(cache.k.data);
    auto* cache_v       = static_cast<__nv_bfloat16*>(cache.v.data);
    const auto* input_k = static_cast<const __nv_bfloat16*>(k.data);
    const auto* input_v = static_cast<const __nv_bfloat16*>(v.data);
    const auto* pos     = static_cast<const std::int32_t*>(positions.data);
    const auto* count   = static_cast<const std::int32_t*>(commit_count.data);
    const int padded    = static_cast<int>(cache.padded_capacity);

    const int grid = 1 + (plan.max_count - 1) / 4;
    kv_cache_append_prefix_cyclic_kernel<<<grid, kBlock, 0, stream>>>(
        input_k, input_v, pos, count, cache_k, cache_v, plan.min_count, plan.max_count, padded);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

KVCacheAppendPrefixPlan
kv_cache_append_prefix_resolve_plan(std::int32_t tokens,
                                    KVCacheAppendPrefixExecutionEnvelope envelope) {
    if (tokens < 1) {
        throw std::invalid_argument("kv_cache_append_prefix plan: T must be positive");
    }
    if (envelope.min_count > envelope.max_count ||
        envelope.max_count > static_cast<std::uint32_t>(tokens)) {
        throw std::invalid_argument("kv_cache_append_prefix plan: invalid execution envelope");
    }
    return {
        .tokens    = tokens,
        .min_count = static_cast<std::int32_t>(envelope.min_count),
        .max_count = static_cast<std::int32_t>(envelope.max_count),
    };
}

void kv_cache_append_prefix_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                   const Tensor& commit_count, PagedKVLayerView cache,
                                   const KVCacheAppendPrefixPlan& plan, cudaStream_t stream) {
    launch_paged(k, v, positions, commit_count, cache, plan, stream);
}

void kv_cache_append_prefix_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                   const Tensor& commit_count, CyclicKVCacheLayerView cache,
                                   const KVCacheAppendPrefixPlan& plan, cudaStream_t stream) {
    launch_cyclic(k, v, positions, commit_count, cache, plan, stream);
}

} // namespace ninfer::ops::detail
