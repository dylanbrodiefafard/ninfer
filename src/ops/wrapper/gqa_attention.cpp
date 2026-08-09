// ninfer::ops - GQA A1/A2/A3 validation and finite route dispatch.
#include "ninfer/ops/gqa_attention.h"

#include "core/layout.h"
#include "ops/launcher/gqa_attention.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kHeadDim                      = 256;
constexpr std::int32_t kQuantGroup                   = 64;
constexpr float kExpectedScale                       = 0.0625f;
constexpr std::int32_t kSmallTChunkTokens            = 6;
constexpr std::int32_t kMaximumVerifyTokens          = 16;
constexpr std::uint32_t kTwoChunkPromptVisibleKeys   = 512;
constexpr std::uint32_t kThreeChunkPromptVisibleKeys = 1024;

std::int32_t kv_heads_for_q_heads(std::int32_t q_heads, const char* op) {
    if (q_heads == 24) { return 4; }
    if (q_heads == 16) { return 2; }
    throw std::invalid_argument(std::string(op) + ": unsupported Q/KV head geometry");
}

void require_kv_heads(std::int32_t kv_heads, const char* op) {
    if (kv_heads != 4 && kv_heads != 2) {
        throw std::invalid_argument(std::string(op) + ": unsupported KV head geometry");
    }
}

void require_shape(const Tensor& tensor, std::int32_t n0, std::int32_t n1, std::int32_t n2,
                   std::int32_t n3, const char* op, const char* name) {
    if (tensor.ne[0] != n0 || tensor.ne[1] != n1 || tensor.ne[2] != n2 || tensor.ne[3] != n3) {
        throw std::invalid_argument(std::string(op) + ": invalid shape for " + name);
    }
}

void require_contiguous_nonnull(const Tensor& tensor, const char* op, const char* name) {
    if (!tensor.is_contiguous()) {
        throw std::invalid_argument(std::string(op) + ": " + name + " must be contiguous");
    }
    if (tensor.data == nullptr) {
        throw std::invalid_argument(std::string(op) + ": " + name + " data must be non-null");
    }
}

std::uint32_t validate_cache(const PagedKVLayerView& cache, std::int32_t kv_heads, const char* op) {
    if ((cache.dtype != DType::BF16 && cache.dtype != DType::I8) ||
        cache.num_kv_heads != kv_heads || cache.head_dim != kHeadDim) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache geometry or dtype");
    }
    if (cache.dtype == DType::BF16 && cache.quant_group != 0) {
        throw std::invalid_argument(std::string(op) + ": BF16 KV cache must not have quant_group");
    }
    if (cache.dtype == DType::I8 && cache.quant_group != kQuantGroup) {
        throw std::invalid_argument(std::string(op) + ": I8 KV cache must use quant_group 64");
    }

    const std::int32_t physical_pages = cache.k_pages.ne[3];
    const std::int32_t logical_pages  = cache.block_table.ne[0];
    const std::int64_t capacity       = static_cast<std::int64_t>(logical_pages) * kPagedKVPageSize;
    if (physical_pages <= 0 || logical_pages <= 0 ||
        capacity > std::numeric_limits<std::int32_t>::max()) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache capacity");
    }

    const DType code_dtype = cache.dtype == DType::I8 ? DType::I8 : DType::BF16;
    if (cache.k_pages.dtype != code_dtype || cache.v_pages.dtype != code_dtype) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache code dtype");
    }
    require_shape(cache.k_pages, kHeadDim, kPagedKVPageSize, kv_heads, physical_pages, op,
                  "cache k pages");
    require_shape(cache.v_pages, kHeadDim, kPagedKVPageSize, kv_heads, physical_pages, op,
                  "cache v pages");
    require_contiguous_nonnull(cache.k_pages, op, "cache k pages");
    require_contiguous_nonnull(cache.v_pages, op, "cache v pages");
    if (cache.block_table.dtype != DType::I32) {
        throw std::invalid_argument(std::string(op) + ": block table must be I32");
    }
    require_shape(cache.block_table, logical_pages, 1, 1, 1, op, "block table");
    require_contiguous_nonnull(cache.block_table, op, "block table");

    if (cache.dtype == DType::BF16) {
        if (cache.k_scale_pages.data != nullptr || cache.v_scale_pages.data != nullptr) {
            throw std::invalid_argument(std::string(op) + ": BF16 KV cache must not have scales");
        }
        return static_cast<std::uint32_t>(capacity);
    }

    constexpr std::int32_t groups = kHeadDim / kQuantGroup;
    if (cache.k_scale_pages.dtype != DType::FP16 || cache.v_scale_pages.dtype != DType::FP16) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache scale dtype");
    }
    require_shape(cache.k_scale_pages, groups, kPagedKVPageSize, kv_heads, physical_pages, op,
                  "cache k scale pages");
    require_shape(cache.v_scale_pages, groups, kPagedKVPageSize, kv_heads, physical_pages, op,
                  "cache v scale pages");
    require_contiguous_nonnull(cache.k_scale_pages, op, "cache k scale pages");
    require_contiguous_nonnull(cache.v_scale_pages, op, "cache v scale pages");
    return static_cast<std::uint32_t>(capacity);
}

void validate_envelope(GqaExecutionEnvelope envelope, const PagedKVLayerView& cache,
                       std::int32_t tokens, const char* op) {
    const std::uint32_t capacity = validate_cache(cache, cache.num_kv_heads, op);
    if (envelope.min_visible_keys == 0 || envelope.min_visible_keys > envelope.max_visible_keys ||
        envelope.max_visible_keys > kGqaAttentionMaximumVisibleKeys ||
        envelope.max_visible_keys > capacity) {
        throw std::invalid_argument(std::string(op) + ": invalid execution envelope");
    }
    if (envelope.max_visible_keys < static_cast<std::uint32_t>(tokens)) {
        throw std::invalid_argument(std::string(op) + ": execution envelope is shorter than T");
    }
}

void validate_attention_tensors(const Tensor& q, const Tensor& positions, const Tensor& out,
                                const PagedKVLayerView& cache, GqaExecutionEnvelope envelope,
                                float scale, const char* op) {
    if (q.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument(std::string(op) + ": q/out must be BF16");
    }
    if (positions.dtype != DType::I32) {
        throw std::invalid_argument(std::string(op) + ": positions must be I32");
    }
    if (!std::isfinite(scale) || std::abs(scale - kExpectedScale) > 1.0e-6f) {
        throw std::invalid_argument(std::string(op) + ": scale must be 1/sqrt(256)");
    }
    const std::int32_t q_heads  = q.ne[1];
    const std::int32_t kv_heads = kv_heads_for_q_heads(q_heads, op);
    const std::int32_t tokens   = q.ne[2];
    if (tokens <= 0) { throw std::invalid_argument(std::string(op) + ": T must be positive"); }
    require_shape(q, kHeadDim, q_heads, tokens, 1, op, "q");
    require_shape(positions, tokens, 1, 1, 1, op, "positions");
    require_shape(out, kHeadDim, q_heads, tokens, 1, op, "out");
    require_contiguous_nonnull(q, op, "q");
    require_contiguous_nonnull(positions, op, "positions");
    require_contiguous_nonnull(out, op, "out");
    if (cache.num_kv_heads != kv_heads) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache head geometry");
    }
    validate_envelope(envelope, cache, tokens, op);
}

struct SmallTWorkspace {
    Tensor acc;
    Tensor m;
    Tensor l;
};

template <class Allocator>
SmallTWorkspace allocate_small_t_workspace(Allocator& workspace, std::int32_t q_heads,
                                           std::int32_t tokens, std::int32_t splits) {
    return {
        workspace.alloc(DType::BF16, {kHeadDim, q_heads, tokens, splits}),
        workspace.alloc(DType::FP32, {q_heads, tokens, splits}),
        workspace.alloc(DType::FP32, {q_heads, tokens, splits}),
    };
}

template <typename Launch>
void for_each_small_t_chunk(const Tensor& q, const Tensor& positions, WorkspaceArena& workspace,
                            DType cache_dtype, GqaExecutionEnvelope envelope, Tensor& out,
                            Launch&& launch) {
    for (std::int32_t begin = 0; begin < q.ne[2]; begin += kSmallTChunkTokens) {
        const std::int32_t count = std::min(kSmallTChunkTokens, q.ne[2] - begin);
        auto chunk_scope         = workspace.scope();
        const std::int32_t splits =
            detail::gqa_attention_split_capacity(q.ne[1], count, cache_dtype, envelope);
        SmallTWorkspace partial = allocate_small_t_workspace(workspace, q.ne[1], count, splits);
        Tensor q_chunk          = q.slice(2, begin, count);
        Tensor position_chunk   = positions.slice(0, begin, count);
        Tensor out_chunk        = out.slice(2, begin, count);
        launch(begin, count, q_chunk, position_chunk, partial, out_chunk);
    }
}

void launch_chunked_small_t(const Tensor& q, const Tensor& k, const Tensor& v,
                            const Tensor& positions, float scale, PagedKVLayerView cache,
                            GqaExecutionEnvelope envelope, WorkspaceArena& workspace, Tensor& out,
                            cudaStream_t stream) {
    for_each_small_t_chunk(
        q, positions, workspace, cache.dtype, envelope, out,
        [&](std::int32_t begin, std::int32_t count, const Tensor& q_chunk,
            const Tensor& position_chunk, SmallTWorkspace& partial, Tensor& out_chunk) {
            Tensor k_chunk = k.slice(2, begin, count);
            Tensor v_chunk = v.slice(2, begin, count);
            detail::gqa_attention_small_t_launch(q_chunk, k_chunk, v_chunk, position_chunk, scale,
                                                 cache, envelope, partial.acc, partial.m, partial.l,
                                                 out_chunk, stream);
        });
}

void launch_cached_chunked_small_t(const Tensor& q, const Tensor& positions, float scale,
                                   const PagedKVLayerView& cache, GqaExecutionEnvelope envelope,
                                   WorkspaceArena& workspace, Tensor& out, cudaStream_t stream) {
    for_each_small_t_chunk(
        q, positions, workspace, cache.dtype, envelope, out,
        [&](std::int32_t, std::int32_t, const Tensor& q_chunk, const Tensor& position_chunk,
            SmallTWorkspace& partial, Tensor& out_chunk) {
            detail::gqa_attention_cached_small_t_launch(q_chunk, position_chunk, scale, cache,
                                                        envelope, partial.acc, partial.m, partial.l,
                                                        out_chunk, stream);
        });
}

} // namespace

namespace detail {

GqaAttentionRoute gqa_attention_resolve_route(std::int32_t q_heads, std::int32_t tokens,
                                              GqaExecutionEnvelope envelope) {
    if (tokens >= 1 && tokens <= kSmallTChunkTokens) { return GqaAttentionRoute::SmallT; }
    const std::uint32_t prompt_visible_keys = tokens <= 2 * kSmallTChunkTokens
                                                  ? kTwoChunkPromptVisibleKeys
                                                  : kThreeChunkPromptVisibleKeys;
    if (q_heads == 16 && tokens <= kMaximumVerifyTokens &&
        envelope.max_visible_keys > prompt_visible_keys) {
        return GqaAttentionRoute::ChunkedSmallT;
    }
    return GqaAttentionRoute::Prompt;
}

const char* gqa_attention_route_name(GqaAttentionRoute route) {
    switch (route) {
    case GqaAttentionRoute::SmallT:
        return "small_t";
    case GqaAttentionRoute::ChunkedSmallT:
        return "chunked_small_t";
    case GqaAttentionRoute::Prompt:
        return "prompt";
    }
    return "unknown";
}

} // namespace detail

std::size_t gqa_attention_workspace_capacity_bytes(std::int32_t q_heads, DType cache_dtype,
                                                   GqaExecutionEnvelope envelope,
                                                   std::int32_t min_tokens,
                                                   std::int32_t max_tokens) {
    (void)kv_heads_for_q_heads(q_heads, "gqa_attention workspace");
    if ((cache_dtype != DType::BF16 && cache_dtype != DType::I8) || min_tokens <= 0 ||
        max_tokens < min_tokens || envelope.min_visible_keys == 0 ||
        envelope.min_visible_keys > envelope.max_visible_keys ||
        envelope.max_visible_keys > kGqaAttentionMaximumVisibleKeys ||
        envelope.max_visible_keys < static_cast<std::uint32_t>(max_tokens)) {
        throw std::invalid_argument("gqa_attention workspace: invalid profile or interval");
    }

    const auto chunk_capacity = [&](std::int32_t tokens) {
        const std::int32_t splits =
            detail::gqa_attention_split_capacity(q_heads, tokens, cache_dtype, envelope);
        WorkspaceLayoutBuilder layout;
        (void)allocate_small_t_workspace(layout, q_heads, tokens, splits);
        return layout.peak_bytes(1);
    };
    const auto exact_capacity = [&](std::int32_t tokens) {
        const detail::GqaAttentionRoute route =
            detail::gqa_attention_resolve_route(q_heads, tokens, envelope);
        if (route == detail::GqaAttentionRoute::Prompt) { return std::size_t{0}; }
        if (route == detail::GqaAttentionRoute::SmallT) { return chunk_capacity(tokens); }
        std::size_t maximum = 0;
        for (std::int32_t begin = 0; begin < tokens; begin += kSmallTChunkTokens) {
            maximum =
                std::max(maximum, chunk_capacity(std::min(kSmallTChunkTokens, tokens - begin)));
        }
        return maximum;
    };

    std::size_t maximum = 0;
    if (min_tokens <= kMaximumVerifyTokens) {
        const std::int32_t last = std::min(max_tokens, kMaximumVerifyTokens);
        for (std::int32_t tokens = min_tokens; tokens <= last; ++tokens) {
            maximum = std::max(maximum, exact_capacity(tokens));
        }
    }
    return maximum;
}

void gqa_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
                   float scale, PagedKVLayerView cache, GqaExecutionEnvelope envelope,
                   WorkspaceArena& workspace, Tensor& out, cudaStream_t stream) {
    constexpr const char* op = "gqa_attention";
    validate_attention_tensors(q, positions, out, cache, envelope, scale, op);
    if (k.dtype != DType::BF16 || v.dtype != DType::BF16) {
        throw std::invalid_argument("gqa_attention: k/v must be BF16");
    }
    const std::int32_t tokens   = q.ne[2];
    const std::int32_t kv_heads = kv_heads_for_q_heads(q.ne[1], op);
    require_shape(k, kHeadDim, kv_heads, tokens, 1, op, "k");
    require_shape(v, kHeadDim, kv_heads, tokens, 1, op, "v");
    require_contiguous_nonnull(k, op, "k");
    require_contiguous_nonnull(v, op, "v");

    auto scope = workspace.scope();
    if (detail::gqa_attention_resolve_route(q.ne[1], tokens, envelope) ==
        detail::GqaAttentionRoute::ChunkedSmallT) {
        launch_chunked_small_t(q, k, v, positions, scale, cache, envelope, workspace, out, stream);
        return;
    }
    if (detail::gqa_attention_uses_small_t(tokens)) {
        const std::int32_t splits =
            detail::gqa_attention_split_capacity(q.ne[1], tokens, cache.dtype, envelope);
        SmallTWorkspace partial = allocate_small_t_workspace(workspace, q.ne[1], tokens, splits);
        detail::gqa_attention_launch(q, k, v, positions, scale, cache, envelope, &partial.acc,
                                     &partial.m, &partial.l, out, stream);
        return;
    }
    detail::gqa_attention_launch(q, k, v, positions, scale, cache, envelope, nullptr, nullptr,
                                 nullptr, out, stream);
}

void gqa_kv_append(const Tensor& k, const Tensor& v, const Tensor& positions,
                   PagedKVLayerView cache, cudaStream_t stream) {
    constexpr const char* op = "gqa_kv_append";
    if (k.dtype != DType::BF16 || v.dtype != DType::BF16) {
        throw std::invalid_argument("gqa_kv_append: k/v must be BF16");
    }
    if (positions.dtype != DType::I32) {
        throw std::invalid_argument("gqa_kv_append: positions must be I32");
    }
    const std::int32_t kv_heads = k.ne[1];
    require_kv_heads(kv_heads, op);
    const std::int32_t tokens = k.ne[2];
    if (tokens <= 0) { throw std::invalid_argument("gqa_kv_append: T must be positive"); }
    require_shape(k, kHeadDim, kv_heads, tokens, 1, op, "k");
    require_shape(v, kHeadDim, kv_heads, tokens, 1, op, "v");
    require_shape(positions, tokens, 1, 1, 1, op, "positions");
    require_contiguous_nonnull(k, op, "k");
    require_contiguous_nonnull(v, op, "v");
    require_contiguous_nonnull(positions, op, "positions");
    const std::uint32_t capacity = validate_cache(cache, kv_heads, op);
    if (static_cast<std::uint32_t>(tokens) > capacity) {
        throw std::invalid_argument("gqa_kv_append: T exceeds KV cache capacity");
    }
    detail::gqa_kv_append_launch(k, v, positions, cache, stream);
}

void gqa_attention_cached(const Tensor& q, const Tensor& positions, float scale,
                          const PagedKVLayerView& cache, GqaExecutionEnvelope envelope,
                          WorkspaceArena& workspace, Tensor& out, cudaStream_t stream) {
    constexpr const char* op = "gqa_attention_cached";
    validate_attention_tensors(q, positions, out, cache, envelope, scale, op);

    auto scope = workspace.scope();
    if (detail::gqa_attention_resolve_route(q.ne[1], q.ne[2], envelope) ==
        detail::GqaAttentionRoute::ChunkedSmallT) {
        launch_cached_chunked_small_t(q, positions, scale, cache, envelope, workspace, out, stream);
        return;
    }
    if (detail::gqa_attention_uses_small_t(q.ne[2])) {
        const std::int32_t splits =
            detail::gqa_attention_split_capacity(q.ne[1], q.ne[2], cache.dtype, envelope);
        SmallTWorkspace partial = allocate_small_t_workspace(workspace, q.ne[1], q.ne[2], splits);
        detail::gqa_attention_cached_small_t_launch(q, positions, scale, cache, envelope,
                                                    partial.acc, partial.m, partial.l, out, stream);
        return;
    }
    detail::gqa_attention_prompt_attention_launch(q, positions, scale, cache, out, stream);
}

} // namespace ninfer::ops
