#include "ninfer/ops/bidirectional_gqa_attention.h"
#include "ninfer/types.h"

#include "core/paged_kv_cache.h"
#include "ops/r9700/dflash/bidirectional_gqa_attention_launch.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kHeadDim = 128;
constexpr std::int32_t kQHeads  = 32;
constexpr std::int32_t kKvHeads = 8;
constexpr float kExpectedScale  = 0.08838834764831844055F;

void require_shape(const Tensor& tensor, std::int32_t n0, std::int32_t n1, std::int32_t n2,
                   std::int32_t n3, const char* name) {
    if (tensor.ne[0] != n0 || tensor.ne[1] != n1 || tensor.ne[2] != n2 || tensor.ne[3] != n3) {
        throw std::invalid_argument(std::string("bidirectional_gqa_attention: invalid shape for ") +
                                    name);
    }
}

void require_tensor(const Tensor& tensor, DType dtype, const char* name) {
    if (tensor.dtype != dtype || !tensor.is_contiguous() || tensor.data == nullptr) {
        throw std::invalid_argument(std::string("bidirectional_gqa_attention: invalid ") + name);
    }
}

std::uint32_t validate_context(const BidirectionalGqaBF16ContextView& context) {
    if (context.head_dim != kHeadDim || context.num_kv_heads != kKvHeads) {
        throw std::invalid_argument("bidirectional_gqa_attention: invalid context geometry");
    }
    require_tensor(context.key_pages, DType::BF16, "context key pages");
    require_tensor(context.value_pages, DType::BF16, "context value pages");
    require_tensor(context.block_tables, DType::I32, "context block tables");
    const std::int32_t physical_pages = context.key_pages.ne[2];
    require_shape(context.key_pages, kHeadDim, kPagedKVPageSize, physical_pages, kKvHeads,
                  "context key pages");
    require_shape(context.value_pages, kHeadDim, kPagedKVPageSize, physical_pages, kKvHeads,
                  "context value pages");
    if (physical_pages <= 0 || context.block_tables.ne[0] <= 0 || context.block_tables.ne[1] <= 0 ||
        context.block_tables.ne[2] != 1 || context.block_tables.ne[3] != 1) {
        throw std::invalid_argument("bidirectional_gqa_attention: invalid context capacity");
    }
    const std::uint64_t capacity =
        static_cast<std::uint64_t>(context.block_tables.ne[0]) * kPagedKVPageSize;
    if (capacity > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("bidirectional_gqa_attention: context capacity exceeds int32");
    }
    return static_cast<std::uint32_t>(capacity);
}

void validate_profile(GqaContextExecutionEnvelope envelope, std::int32_t min_tokens,
                      std::int32_t max_tokens, std::int32_t batch_size) {
    if (min_tokens < 1 || max_tokens < min_tokens || max_tokens > 16 || batch_size < 1 ||
        batch_size > static_cast<std::int32_t>(kDFlashMaximumConcurrency) ||
        envelope.min_context > envelope.max_context ||
        envelope.max_context >
            static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument(
            "bidirectional_gqa_attention: invalid envelope, token interval, or batch");
    }
}

} // namespace

std::size_t bidirectional_gqa_attention_workspace_capacity_bytes(
    GqaContextExecutionEnvelope envelope, std::int32_t min_tokens, std::int32_t max_tokens,
    std::int32_t batch_size) {
    validate_profile(envelope, min_tokens, max_tokens, batch_size);
    return 0;
}

void bidirectional_gqa_attention(const Tensor& q, const Tensor& query_k, const Tensor& query_v,
                                 const Tensor& context_lengths, const Tensor& valid_columns,
                                 const Tensor& table_rows, float scale,
                                 const BidirectionalGqaBF16ContextView& context,
                                 GqaContextExecutionEnvelope envelope, WorkspaceArena& workspace,
                                 Tensor& out, hipStream_t stream) {
    const std::int32_t tokens = q.ne[2];
    const std::int32_t batch  = q.ne[3];
    validate_profile(envelope, tokens, tokens, batch);
    require_tensor(q, DType::BF16, "q");
    require_tensor(query_k, DType::BF16, "query k");
    require_tensor(query_v, DType::BF16, "query v");
    require_tensor(context_lengths, DType::I32, "context lengths");
    require_tensor(valid_columns, DType::I32, "valid columns");
    require_tensor(table_rows, DType::I32, "table rows");
    require_tensor(out, DType::BF16, "output");
    require_shape(q, kHeadDim, kQHeads, tokens, batch, "q");
    require_shape(query_k, kHeadDim, kKvHeads, tokens, batch, "query k");
    require_shape(query_v, kHeadDim, kKvHeads, tokens, batch, "query v");
    require_shape(context_lengths, batch, 1, 1, 1, "context lengths");
    require_shape(valid_columns, batch, 1, 1, 1, "valid columns");
    require_shape(table_rows, batch, 1, 1, 1, "table rows");
    require_shape(out, kHeadDim, kQHeads, tokens, batch, "output");
    const std::uint32_t logical_capacity = validate_context(context);
    if (envelope.max_context > logical_capacity) {
        throw std::invalid_argument("bidirectional_gqa_attention: envelope exceeds context");
    }
    if (!std::isfinite(scale) || std::abs(scale - kExpectedScale) > 1.0e-7F) {
        throw std::invalid_argument("bidirectional_gqa_attention: scale must be 1/sqrt(128)");
    }
    if (out.data == q.data || out.data == query_k.data || out.data == query_v.data ||
        out.data == context.key_pages.data || out.data == context.value_pages.data) {
        throw std::invalid_argument("bidirectional_gqa_attention: output aliases an input");
    }
    (void)workspace;
    detail::bidirectional_gqa_attention_launch(q, query_k, query_v, context_lengths, valid_columns,
                                               table_rows, scale, context, envelope, out, stream);
}

} // namespace ninfer::ops
