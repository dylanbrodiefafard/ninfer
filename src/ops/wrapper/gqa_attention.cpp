// ninfer::ops - GQA A1/A2/A3 validation and finite route dispatch.
#include "ninfer/ops/gqa_attention.h"

#include "core/layout.h"
#include "ops/common/math.h"
#include "ops/launcher/gqa_attention.h"
#include "ops/launcher/gqa_xattn_scratch.h"

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
constexpr std::int32_t kNvfp4Group                   = 16;
constexpr std::int32_t kNvfp4CodeWidth               = 128;
constexpr float kExpectedScale                       = 0.0625f;
constexpr std::int32_t kSmallTChunkTokens            = 6;
constexpr std::int32_t kMaximumVerifyTokens          = 16;
constexpr std::int32_t kMaximumBatchSize             = 8;
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

bool supported_cache_dtype(DType dtype) {
    return dtype == DType::BF16 || dtype == DType::I8 || dtype == DType::U8;
}

void validate_cache_dtype(DType dtype, std::int32_t quant_group, const char* op) {
    if (!supported_cache_dtype(dtype)) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache geometry or dtype");
    }
    if (dtype == DType::BF16 && quant_group != 0) {
        throw std::invalid_argument(std::string(op) + ": BF16 KV cache must not have quant_group");
    }
    if (dtype == DType::I8 && quant_group != kQuantGroup) {
        throw std::invalid_argument(std::string(op) + ": I8 KV cache must use quant_group 64");
    }
    if (dtype == DType::U8 && quant_group != kNvfp4Group) {
        throw std::invalid_argument(std::string(op) + ": NVFP4 KV cache must use quant_group 16");
    }
}

std::int32_t cache_code_leading(DType dtype) { return dtype == DType::U8 ? kNvfp4CodeWidth : kHeadDim; }

DType cache_code_dtype(DType dtype) {
    if (dtype == DType::I8) { return DType::I8; }
    if (dtype == DType::U8) { return DType::U8; }
    return DType::BF16;
}

void validate_scale_planes(const Tensor& k_scale, const Tensor& v_scale, DType dtype,
                           std::int32_t kv_heads, std::int32_t physical_pages, const char* op) {
    const std::int32_t groups = dtype == DType::U8 ? kHeadDim / kNvfp4Group : kHeadDim / kQuantGroup;
    const DType scale_dtype   = dtype == DType::U8 ? DType::FP8_E4M3FN : DType::FP16;
    if (k_scale.dtype != scale_dtype || v_scale.dtype != scale_dtype) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache scale dtype");
    }
    require_shape(k_scale, groups, kPagedKVPageSize, kv_heads, physical_pages, op,
                  "cache k scale pages");
    require_shape(v_scale, groups, kPagedKVPageSize, kv_heads, physical_pages, op,
                  "cache v scale pages");
    require_contiguous_nonnull(k_scale, op, "cache k scale pages");
    require_contiguous_nonnull(v_scale, op, "cache v scale pages");
}

std::uint32_t validate_cache(const PagedKVLayerView& cache, std::int32_t kv_heads, const char* op) {
    if (!supported_cache_dtype(cache.dtype) || cache.num_kv_heads != kv_heads ||
        cache.head_dim != kHeadDim) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache geometry or dtype");
    }
    validate_cache_dtype(cache.dtype, cache.quant_group, op);
    if (cache.sage_pv && cache.dtype != DType::U8) {
        throw std::invalid_argument(std::string(op) + ": --sage (FP4-PV) requires NVFP4 KV storage");
    }

    const std::int32_t physical_pages = cache.k_pages.ne[3];
    const std::int32_t logical_pages  = cache.block_table.ne[0];
    const std::int64_t capacity       = static_cast<std::int64_t>(logical_pages) * kPagedKVPageSize;
    if (physical_pages <= 0 || logical_pages <= 0 ||
        capacity > std::numeric_limits<std::int32_t>::max()) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache capacity");
    }

    const DType code_dtype       = cache_code_dtype(cache.dtype);
    const std::int32_t code_lead = cache_code_leading(cache.dtype);
    if (cache.k_pages.dtype != code_dtype || cache.v_pages.dtype != code_dtype) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache code dtype");
    }
    require_shape(cache.k_pages, code_lead, kPagedKVPageSize, kv_heads, physical_pages, op,
                  "cache k pages");
    require_shape(cache.v_pages, code_lead, kPagedKVPageSize, kv_heads, physical_pages, op,
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

    validate_scale_planes(cache.k_scale_pages, cache.v_scale_pages, cache.dtype, kv_heads,
                          physical_pages, op);
    return static_cast<std::uint32_t>(capacity);
}

std::uint32_t validate_batch_cache(const PagedKVBatchLayerView& cache, std::int32_t kv_heads,
                                   const char* op) {
    if (!supported_cache_dtype(cache.dtype) || cache.num_kv_heads != kv_heads ||
        cache.head_dim != kHeadDim) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache geometry or dtype");
    }
    validate_cache_dtype(cache.dtype, cache.quant_group, op);
    if (cache.sage_pv && cache.dtype != DType::U8) {
        throw std::invalid_argument(std::string(op) + ": --sage (FP4-PV) requires NVFP4 KV storage");
    }

    const std::int32_t physical_pages = cache.k_pages.ne[3];
    const std::int32_t logical_pages  = cache.block_tables.ne[0];
    const std::int32_t table_rows     = cache.block_tables.ne[1];
    const std::int64_t capacity       = static_cast<std::int64_t>(logical_pages) * kPagedKVPageSize;
    if (physical_pages <= 0 || logical_pages <= 0 || table_rows <= 0 ||
        capacity > std::numeric_limits<std::int32_t>::max()) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache capacity");
    }

    const DType code_dtype       = cache_code_dtype(cache.dtype);
    const std::int32_t code_lead = cache_code_leading(cache.dtype);
    if (cache.k_pages.dtype != code_dtype || cache.v_pages.dtype != code_dtype) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache code dtype");
    }
    require_shape(cache.k_pages, code_lead, kPagedKVPageSize, kv_heads, physical_pages, op,
                  "cache k pages");
    require_shape(cache.v_pages, code_lead, kPagedKVPageSize, kv_heads, physical_pages, op,
                  "cache v pages");
    require_contiguous_nonnull(cache.k_pages, op, "cache k pages");
    require_contiguous_nonnull(cache.v_pages, op, "cache v pages");
    if (cache.block_tables.dtype != DType::I32) {
        throw std::invalid_argument(std::string(op) + ": block tables must be I32");
    }
    require_shape(cache.block_tables, logical_pages, table_rows, 1, 1, op, "block tables");
    require_contiguous_nonnull(cache.block_tables, op, "block tables");

    if (cache.dtype == DType::BF16) {
        if (cache.k_scale_pages.data != nullptr || cache.v_scale_pages.data != nullptr) {
            throw std::invalid_argument(std::string(op) + ": BF16 KV cache must not have scales");
        }
        return static_cast<std::uint32_t>(capacity);
    }

    validate_scale_planes(cache.k_scale_pages, cache.v_scale_pages, cache.dtype, kv_heads,
                          physical_pages, op);
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

void validate_batched_attention_tensors(const Tensor& q, const Tensor& positions,
                                        const Tensor& valid_columns, const Tensor& kv_table_rows,
                                        const Tensor& out, const PagedKVBatchLayerView& cache,
                                        GqaExecutionEnvelope envelope, float scale,
                                        const char* op) {
    if (q.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument(std::string(op) + ": q/out must be BF16");
    }
    const bool masked = valid_columns.data != nullptr;
    if (positions.dtype != DType::I32 || kv_table_rows.dtype != DType::I32 ||
        (masked && valid_columns.dtype != DType::I32)) {
        throw std::invalid_argument(std::string(op) + ": batch metadata must be I32");
    }
    if (!std::isfinite(scale) || std::abs(scale - kExpectedScale) > 1.0e-6f) {
        throw std::invalid_argument(std::string(op) + ": scale must be 1/sqrt(256)");
    }
    const std::int32_t q_heads  = q.ne[1];
    const std::int32_t kv_heads = kv_heads_for_q_heads(q_heads, op);
    const std::int32_t width    = q.ne[2];
    const std::int32_t batch    = q.ne[3];
    if (width <= 0 || batch <= 0 || batch > kMaximumBatchSize ||
        (batch > 1 && width > kMaximumVerifyTokens)) {
        throw std::invalid_argument(std::string(op) + ": unsupported B/W domain");
    }
    require_shape(q, kHeadDim, q_heads, width, batch, op, "q");
    require_shape(positions, width, batch, 1, 1, op, "positions");
    if (masked) { require_shape(valid_columns, batch, 1, 1, 1, op, "valid columns"); }
    require_shape(kv_table_rows, batch, 1, 1, 1, op, "KV table rows");
    require_shape(out, kHeadDim, q_heads, width, batch, op, "out");
    require_contiguous_nonnull(q, op, "q");
    require_contiguous_nonnull(positions, op, "positions");
    if (masked) { require_contiguous_nonnull(valid_columns, op, "valid columns"); }
    require_contiguous_nonnull(kv_table_rows, op, "KV table rows");
    require_contiguous_nonnull(out, op, "out");
    if (cache.num_kv_heads != kv_heads) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache head geometry");
    }
    const std::uint32_t capacity = validate_batch_cache(cache, kv_heads, op);
    if (cache.block_tables.ne[1] < batch || envelope.min_visible_keys == 0 ||
        envelope.min_visible_keys > envelope.max_visible_keys ||
        envelope.max_visible_keys > kGqaAttentionMaximumVisibleKeys ||
        envelope.max_visible_keys > capacity ||
        envelope.max_visible_keys < static_cast<std::uint32_t>(width)) {
        throw std::invalid_argument(std::string(op) + ": invalid execution envelope or table");
    }
}

void validate_tree_verify(const Tensor& ancestor_mask, const Tensor& prefix_lengths,
                          std::int32_t width, std::int32_t batch, const char* op) {
    const bool tree   = ancestor_mask.data != nullptr;
    const bool prefix = prefix_lengths.data != nullptr;
    if (tree != prefix) {
        throw std::invalid_argument(std::string(op) +
                                    ": ancestor_mask and prefix_lengths must both be empty or both "
                                    "be populated");
    }
    if (!tree) { return; }
    if (width > kMaximumVerifyTokens) {
        throw std::invalid_argument(std::string(op) + ": tree verify requires W<=16");
    }
    if (ancestor_mask.dtype != DType::I32 || prefix_lengths.dtype != DType::I32) {
        throw std::invalid_argument(std::string(op) + ": tree verify metadata must be I32");
    }
    require_shape(ancestor_mask, width, batch, 1, 1, op, "ancestor mask");
    require_shape(prefix_lengths, batch, 1, 1, 1, op, "prefix lengths");
    require_contiguous_nonnull(ancestor_mask, op, "ancestor mask");
    require_contiguous_nonnull(prefix_lengths, op, "prefix lengths");
}

struct SmallTWorkspace {
    Tensor acc;
    Tensor m;
    Tensor l;
};

template <class Allocator>
SmallTWorkspace allocate_small_t_workspace(Allocator& workspace, std::int32_t q_heads,
                                           std::int32_t tokens, std::int32_t splits,
                                           std::int32_t batch_size = 1) {
    return {
        workspace.alloc(DType::BF16, {kHeadDim, q_heads, tokens, splits * batch_size}),
        workspace.alloc(DType::FP32, {q_heads, tokens, splits * batch_size}),
        workspace.alloc(DType::FP32, {q_heads, tokens, splits * batch_size}),
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
                            const Tensor& positions, const Tensor& valid_columns,
                            const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
                            GqaExecutionEnvelope envelope, WorkspaceArena& workspace, Tensor& out,
                            cudaStream_t stream, const Tensor& ancestor_mask,
                            const Tensor& prefix_lengths) {
    for (std::int32_t begin = 0; begin < q.ne[2]; begin += kSmallTChunkTokens) {
        const std::int32_t count = std::min(kSmallTChunkTokens, q.ne[2] - begin);
        auto chunk_scope         = workspace.scope();
        const std::int32_t splits =
            detail::gqa_attention_split_capacity(q.ne[1], count, cache.dtype, envelope);
        SmallTWorkspace partial =
            allocate_small_t_workspace(workspace, q.ne[1], count, splits, q.ne[3]);
        detail::gqa_attention_small_t_launch(q, k, v, positions, valid_columns, table_rows, scale,
                                             cache, envelope, begin, count, partial.acc, partial.m,
                                             partial.l, out, stream, ancestor_mask, prefix_lengths);
    }
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

GqaAttentionRoute gqa_attention_resolve_route(std::int32_t q_heads, std::int32_t width,
                                              std::int32_t batch_size,
                                              GqaExecutionEnvelope envelope, bool tree_verify) {
    if (tree_verify) {
        if (width >= 1 && width <= kSmallTChunkTokens) { return GqaAttentionRoute::SmallT; }
        return GqaAttentionRoute::ChunkedSmallT;
    }
    if (width >= 1 && width <= kSmallTChunkTokens) { return GqaAttentionRoute::SmallT; }
    if (batch_size > 1) { return GqaAttentionRoute::ChunkedSmallT; }
    const std::uint32_t prompt_visible_keys =
        width <= 2 * kSmallTChunkTokens ? kTwoChunkPromptVisibleKeys : kThreeChunkPromptVisibleKeys;
    if (q_heads == 16 && width <= kMaximumVerifyTokens &&
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
                                                   std::int32_t batch_size, std::int32_t min_width,
                                                   std::int32_t max_width, float keep_frac,
                                                   bool tree_verify, float xattn_tau) {
    (void)kv_heads_for_q_heads(q_heads, "gqa_attention workspace");
    if ((cache_dtype != DType::BF16 && cache_dtype != DType::I8 && cache_dtype != DType::U8) || batch_size <= 0 ||
        batch_size > kMaximumBatchSize || min_width <= 0 || max_width < min_width ||
        (batch_size > 1 && max_width > kMaximumVerifyTokens) ||
        (tree_verify && max_width > kMaximumVerifyTokens) || envelope.min_visible_keys == 0 ||
        envelope.min_visible_keys > envelope.max_visible_keys ||
        envelope.max_visible_keys > kGqaAttentionMaximumVisibleKeys ||
        envelope.max_visible_keys < static_cast<std::uint32_t>(max_width)) {
        throw std::invalid_argument("gqa_attention workspace: invalid profile or interval");
    }

    const auto chunk_capacity = [&](std::int32_t width) {
        const std::int32_t splits =
            detail::gqa_attention_split_capacity(q_heads, width, cache_dtype, envelope);
        const std::int32_t launch_batch = batch_size;
        WorkspaceLayoutBuilder layout;
        (void)allocate_small_t_workspace(layout, q_heads, width, splits, launch_batch);
        // Sparge-decode tile-skip scratch (rank keep set): only a T=1 step on a
        // sage (U8) cache with keep_frac < 1 allocates the rank scratch, so only
        // that width carries the addition. Dry-run the same three allocations
        // through the layout builder so the arena's 256B alignment padding is
        // modeled exactly (a flat byte sum undercounts and overflows the arena).
        if (keep_frac < 1.0f && cache_dtype == DType::U8 && width == 1) {
            const std::int32_t kv_rows  =
                launch_batch * kv_heads_for_q_heads(q_heads, "gqa_attention workspace");
            const std::int32_t max_keep =
                (static_cast<std::int32_t>(envelope.max_visible_keys) + 31) / 32;
            (void)layout.alloc(DType::I32, {kv_rows, max_keep});
            (void)layout.alloc(DType::I32, {kv_rows});
            (void)layout.alloc(DType::I32, {kv_rows, splits + 1});
        }
        return layout.peak_bytes(1);
    };
    const auto exact_capacity = [&](std::int32_t width) {
        const detail::GqaAttentionRoute route =
            detail::gqa_attention_resolve_route(q_heads, width, batch_size, envelope, tree_verify);
        if (route == detail::GqaAttentionRoute::Prompt) {
            if (xattn_tau > 0.0f && xattn_tau < 1.0f && cache_dtype == DType::U8) {
                WorkspaceLayoutBuilder layout;
                const std::int32_t kv_heads =
                    kv_heads_for_q_heads(q_heads, "gqa_attention workspace");
                const int n_br = div_up(width, kGqaXattnPrefillBr);
                const int n_kb = gqa_xattn_n_kb(kGqaXattnRankTiles, envelope.max_visible_keys);
                (void)layout.alloc_bytes(
                    gqa_xattn_scratch_bytes(q_heads, kv_heads, n_br, n_kb));
                return layout.peak_bytes(1);
            }
            return std::size_t{0};
        }
        if (route == detail::GqaAttentionRoute::SmallT) { return chunk_capacity(width); }
        std::size_t maximum = 0;
        for (std::int32_t begin = 0; begin < width; begin += kSmallTChunkTokens) {
            maximum =
                std::max(maximum, chunk_capacity(std::min(kSmallTChunkTokens, width - begin)));
        }
        return maximum;
    };

    std::size_t maximum = 0;
    if (min_width <= kMaximumVerifyTokens) {
        const std::int32_t last = std::min(max_width, kMaximumVerifyTokens);
        for (std::int32_t width = min_width; width <= last; ++width) {
            maximum = std::max(maximum, exact_capacity(width));
        }
    }
    // Prompt widths historically needed no arena. XAttention ranker scratch
    // grows with W (n_br) and the envelope (n_kb), so the interval peak must
    // include the largest Prompt width, not only W<=16 SmallT/ChunkedSmallT.
    if (max_width > kMaximumVerifyTokens) {
        maximum = std::max(maximum, exact_capacity(max_width));
    }
    return maximum;
}

void gqa_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
                   const Tensor& valid_columns, const Tensor& kv_table_rows, float scale,
                   PagedKVBatchLayerView cache, GqaExecutionEnvelope envelope,
                   WorkspaceArena& workspace, Tensor& out, cudaStream_t stream, float keep_frac,
                   float xattn_tau, std::int32_t xattn_min_len, GqaS3PrefillDump* dump,
                   const Tensor& ancestor_mask, const Tensor& prefix_lengths) {
    constexpr const char* op = "gqa_attention";
    if (!(keep_frac > 0.0f && keep_frac <= 1.0f)) {
        throw std::invalid_argument("gqa_attention: keep_frac must be in (0, 1]");
    }
    if (!(xattn_tau > 0.0f && xattn_tau <= 1.0f)) {
        throw std::invalid_argument("gqa_attention: xattn_tau must be in (0, 1]");
    }
    if (keep_frac < 1.0f && xattn_tau < 1.0f) {
        throw std::invalid_argument("gqa_attention: keep_frac and xattn_tau are mutually exclusive");
    }
    if (cache.sage_pv && (keep_frac < 1.0f || xattn_tau < 1.0f)) {
        throw std::invalid_argument(
            "gqa_attention: sage_pv is exact-S3 only; keep_frac / xattn_tau require NVFP4 without "
            "sage");
    }
    if ((keep_frac < 1.0f || xattn_tau < 1.0f) && cache.dtype != DType::U8) {
        throw std::invalid_argument("gqa_attention: keep_frac / xattn_tau require NVFP4 KV");
    }
    if (xattn_min_len < 0) {
        throw std::invalid_argument("gqa_attention: xattn_min_len must be non-negative");
    }
    if (keep_frac < 1.0f && cache.k_mean_pages.data == nullptr) {
        throw std::invalid_argument("gqa_attention: keep_frac<1 requires the k_mean plane");
    }
    validate_batched_attention_tensors(q, positions, valid_columns, kv_table_rows, out, cache,
                                       envelope, scale, op);
    if (k.dtype != DType::BF16 || v.dtype != DType::BF16) {
        throw std::invalid_argument("gqa_attention: k/v must be BF16");
    }
    const std::int32_t width    = q.ne[2];
    const std::int32_t batch    = q.ne[3];
    const std::int32_t kv_heads = kv_heads_for_q_heads(q.ne[1], op);
    require_shape(k, kHeadDim, kv_heads, width, batch, op, "k");
    require_shape(v, kHeadDim, kv_heads, width, batch, op, "v");
    require_contiguous_nonnull(k, op, "k");
    require_contiguous_nonnull(v, op, "v");
    validate_tree_verify(ancestor_mask, prefix_lengths, width, batch, op);
    const bool tree_verify = ancestor_mask.data != nullptr;

    auto scope = workspace.scope();
    const detail::GqaAttentionRoute route =
        detail::gqa_attention_resolve_route(q.ne[1], width, batch, envelope, tree_verify);
    if (route == detail::GqaAttentionRoute::ChunkedSmallT) {
        launch_chunked_small_t(q, k, v, positions, valid_columns, kv_table_rows, scale, cache,
                               envelope, workspace, out, stream, ancestor_mask, prefix_lengths);
        return;
    }
    if (route == detail::GqaAttentionRoute::SmallT) {
        const std::int32_t splits =
            detail::gqa_attention_split_capacity(q.ne[1], width, cache.dtype, envelope);
        SmallTWorkspace partial =
            allocate_small_t_workspace(workspace, q.ne[1], width, splits, batch);
        // Decode/SmallT is always exact. Prefill skip flags on the card are ignored.
        detail::gqa_attention_small_t_launch(q, k, v, positions, valid_columns, kv_table_rows,
                                             scale, cache, envelope, 0, width, partial.acc,
                                             partial.m, partial.l, out, stream, ancestor_mask,
                                             prefix_lengths, 1.0f, {});
        return;
    }
    void* xattn_scratch = nullptr;
    if (xattn_tau < 1.0f && cache.dtype == DType::U8 &&
        envelope.max_visible_keys >= static_cast<std::uint32_t>(xattn_min_len)) {
        const int n_br = div_up(width, kGqaXattnPrefillBr);
        const int n_kb = gqa_xattn_n_kb(static_cast<int>(cache.block_tables.ne[0]),
                                        envelope.max_visible_keys);
        xattn_scratch = workspace
                            .alloc_bytes(gqa_xattn_scratch_bytes(q.ne[1], kv_heads, n_br, n_kb))
                            .data;
    }
    detail::gqa_attention_prompt_launch(q, k, v, positions, valid_columns, kv_table_rows, scale,
                                        cache, out, stream, keep_frac, xattn_tau, xattn_min_len,
                                        dump, xattn_scratch, envelope);
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

void gqa_kv_compact_path(PagedKVBatchLayerView cache, const Tensor& kv_table_rows,
                         const Tensor& prefix_lengths, const Tensor& path, const Tensor& counts,
                         cudaStream_t stream) {
    constexpr const char* op = "gqa_kv_compact_path";
    const std::int32_t batch = counts.ne[0];
    const std::int32_t width = path.ne[0];
    if (batch <= 0 || width <= 0 || path.ne[1] != batch) {
        throw std::invalid_argument("gqa_kv_compact_path: path must be I32 [W,B]");
    }
    if (kv_table_rows.dtype != DType::I32 || prefix_lengths.dtype != DType::I32 ||
        path.dtype != DType::I32 || counts.dtype != DType::I32 || kv_table_rows.ne[0] != batch ||
        prefix_lengths.ne[0] != batch) {
        throw std::invalid_argument("gqa_kv_compact_path: selector vectors must be I32 [B]");
    }
    validate_batch_cache(cache, cache.num_kv_heads, op);
    require_contiguous_nonnull(kv_table_rows, op, "kv table rows");
    require_contiguous_nonnull(prefix_lengths, op, "prefix lengths");
    require_contiguous_nonnull(path, op, "path");
    require_contiguous_nonnull(counts, op, "counts");
    detail::gqa_kv_compact_path_launch(cache, kv_table_rows, prefix_lengths, path, counts, stream);
}

void gqa_attention_cached(const Tensor& q, const Tensor& positions, float scale,
                          const PagedKVLayerView& cache, GqaExecutionEnvelope envelope,
                          WorkspaceArena& workspace, Tensor& out, cudaStream_t stream,
                          float keep_frac, GqaS3DecodeRankDump* rank_dump) {
    constexpr const char* op = "gqa_attention_cached";
    if (keep_frac <= 0.0f || keep_frac > 1.0f) {
        throw std::invalid_argument("gqa_attention_cached: keep_frac must be in (0, 1]");
    }
    validate_attention_tensors(q, positions, out, cache, envelope, scale, op);

    auto scope = workspace.scope();
    if (detail::gqa_attention_resolve_route(q.ne[1], q.ne[2], 1, envelope) ==
        detail::GqaAttentionRoute::ChunkedSmallT) {
        launch_cached_chunked_small_t(q, positions, scale, cache, envelope, workspace, out,
                                      stream);
        return;
    }
    if (detail::gqa_attention_uses_small_t(q.ne[2])) {
        const std::int32_t splits =
            detail::gqa_attention_split_capacity(q.ne[1], q.ne[2], cache.dtype, envelope);
        SmallTWorkspace partial = allocate_small_t_workspace(workspace, q.ne[1], q.ne[2], splits);
        detail::gqa_attention_cached_small_t_launch(q, positions, scale, cache, envelope,
                                                    partial.acc, partial.m, partial.l, out,
                                                    stream, 1.0f, {}, rank_dump);
        return;
    }
    detail::gqa_attention_prompt_attention_launch(q, positions, scale, cache, out, stream);
}

void gqa_attention_s3_dump(const Tensor& q, const Tensor& k, const Tensor& v,
                           const Tensor& positions, const Tensor& valid_columns,
                           const Tensor& kv_table_rows, float scale, PagedKVBatchLayerView cache,
                           GqaExecutionEnvelope envelope, WorkspaceArena& workspace, Tensor& out,
                           cudaStream_t stream, float keep_frac, GqaS3PrefillDump& dump) {
    constexpr const char* op = "gqa_attention";
    validate_batched_attention_tensors(q, positions, valid_columns, kv_table_rows, out, cache,
                                       envelope, scale, op);
    if (k.dtype != DType::BF16 || v.dtype != DType::BF16) {
        throw std::invalid_argument("gqa_attention_s3_dump: k/v must be BF16");
    }
    const std::int32_t width    = q.ne[2];
    const std::int32_t batch    = q.ne[3];
    const std::int32_t kv_heads = kv_heads_for_q_heads(q.ne[1], op);
    require_shape(k, kHeadDim, kv_heads, width, batch, op, "k");
    require_shape(v, kHeadDim, kv_heads, width, batch, op, "v");
    require_contiguous_nonnull(k, op, "k");
    require_contiguous_nonnull(v, op, "v");
    const detail::GqaAttentionRoute route =
        detail::gqa_attention_resolve_route(q.ne[1], width, batch, envelope);
    if (route != detail::GqaAttentionRoute::Prompt) {
        throw std::invalid_argument(
            "gqa_attention_s3_dump: the s3 prefill kernel only runs on the Prompt "
            "route (T > 6); the dump would silently stay empty");
    }
    if (cache.dtype != DType::U8 || !cache.sage_pv) {
        throw std::invalid_argument("gqa_attention_s3_dump: requires a Sage (U8 + sage_pv) cache");
    }
    auto scope = workspace.scope();
    detail::gqa_attention_prompt_launch(q, k, v, positions, valid_columns, kv_table_rows, scale,
                                       cache, out, stream, keep_frac, 1.0f, 8192, &dump);
}

} // namespace ninfer::ops
