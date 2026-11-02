#include "r9700_full_attention.h"

#include "ops/r9700/kv/fp8_int4_kv_attention.h"
#include "ops/r9700/kv/r9700_attention_profile.h"
#if defined(NINFER_R9700_XATTENTION_QUALIFICATION)
#    include "ops/r9700/kv/fp8_int4_kv_xattention.h"
#    if defined(NINFER_R9700_XATTENTION_KEEP_DIAGNOSTIC)
#        include "xattention_keep_trace.h"
#    endif
#endif
#include "targets/qwen3_8_27b/impl/config.h"

#include <algorithm>
#include <array>
#include <limits>

namespace ninfer::targets::qwen3_8_27b::detail {
namespace {

constexpr std::uint32_t kHeadDim                  = 256;
constexpr std::uint32_t kQueryHeads               = 24;
constexpr std::uint32_t kKvHeads                  = 4;
constexpr std::uint32_t kQueriesPerKv             = 6;
constexpr std::uint32_t kPageSize                 = 64;
constexpr std::uint32_t kMaximumRows              = kNativeContext;
constexpr float kQwenAttentionScale               = 0.0625F;
constexpr std::size_t kSequenceWorkspaceAlignment = 256U;

[[nodiscard]] bool layout_valid(Fp8KInt4VPlaneLayout layout) noexcept {
    return layout == Fp8KInt4VPlaneLayout::FeatureFastestPageMajor ||
           layout == Fp8KInt4VPlaneLayout::TokenFastestHeadMajor;
}

[[nodiscard]] ops::r9700::kv::Fp8Int4KvPlaneLayout
raw_layout(Fp8KInt4VPlaneLayout layout) noexcept {
    return layout == Fp8KInt4VPlaneLayout::TokenFastestHeadMajor
               ? ops::r9700::kv::Fp8Int4KvPlaneLayout::TokenFastest
               : ops::r9700::kv::Fp8Int4KvPlaneLayout::FeatureFastest;
}

[[nodiscard]] bool exact_contiguous_shape(const Tensor& tensor, DType dtype, std::int32_t d0,
                                          std::int32_t d1, std::int32_t d2 = 1,
                                          std::int32_t d3 = 1) noexcept {
    if (tensor.data == nullptr || tensor.dtype != dtype || tensor.ne[0] != d0 ||
        tensor.ne[1] != d1 || tensor.ne[2] != d2 || tensor.ne[3] != d3) {
        return false;
    }
    const std::int64_t element = static_cast<std::int64_t>(dtype_size(dtype));
    return tensor.nb[0] == element && tensor.nb[1] == element * d0 &&
           tensor.nb[2] == element * d0 * d1 && tensor.nb[3] == element * d0 * d1 * d2;
}

[[nodiscard]] bool contiguous_elements_at_least(const Tensor& tensor, DType dtype,
                                                std::uint64_t minimum) noexcept {
    if (tensor.data == nullptr || tensor.dtype != dtype) { return false; }
    std::uint64_t elements = 1;
    std::int64_t stride    = static_cast<std::int64_t>(dtype_size(dtype));
    for (int axis = 0; axis < 4; ++axis) {
        if (tensor.ne[axis] <= 0 || (tensor.ne[axis] != 1 && tensor.nb[axis] != stride)) {
            return false;
        }
        const auto extent = static_cast<std::uint64_t>(tensor.ne[axis]);
        if (elements > std::numeric_limits<std::uint64_t>::max() / extent) { return false; }
        elements *= extent;
        if (axis != 3 && tensor.ne[axis] != 1) {
            if (stride > std::numeric_limits<std::int64_t>::max() / tensor.ne[axis]) {
                return false;
            }
            stride *= tensor.ne[axis];
        }
    }
    return elements >= minimum;
}

[[nodiscard]] std::uint64_t tensor_elements(const Tensor& tensor) noexcept {
    std::uint64_t elements = 1;
    for (const std::int32_t extent : tensor.ne) {
        if (extent <= 0 || elements > std::numeric_limits<std::uint64_t>::max() /
                                          static_cast<std::uint64_t>(extent)) {
            return 0;
        }
        elements *= static_cast<std::uint64_t>(extent);
    }
    return elements;
}

} // namespace

std::size_t
r9700_full_attention_score_workspace_capacity_bytes(std::size_t visible_context) noexcept {
    if (!ops::r9700::kv::kFp8QkWmmaDecode) return 0U;
    constexpr std::size_t bytes_per_token = kQueryHeads * sizeof(float);
    if (visible_context == 0U ||
        visible_context > std::numeric_limits<std::size_t>::max() / bytes_per_token) {
        return 0U;
    }
    return visible_context * bytes_per_token;
}

std::size_t r9700_full_attention_workspace_capacity_bytes(std::uint32_t query_rows,
                                                          std::size_t visible_context,
                                                          bool tree_or_device_count) noexcept {
    std::size_t required = 0U;
    if (ops::r9700::kv::use_split512_attention(query_rows, visible_context, tree_or_device_count)) {
        required = ops::r9700::kv::fp8_int4_kv_attention_split512_workspace_capacity_bytes(
            query_rows, visible_context);
    }
    // Planners size one call for the maximum row count, which must cover every narrower host-fixed
    // call it bounds (a prefill tail or short append takes the packed or mid-row route).
    const std::uint32_t packed_rows = std::min(query_rows, 8U);
    if (packed_rows != 0U && ops::r9700::kv::use_packed_decode_attention(
                                 packed_rows, visible_context, tree_or_device_count)) {
        required =
            std::max(required, ops::r9700::kv::fp8_int4_kv_attention_packed_decode_workspace_bytes(
                                   visible_context, packed_rows));
    }
    if (!tree_or_device_count) {
        required = std::max(required,
                            ops::r9700::kv::fp8_int4_kv_attention_mid_rows_workspace_capacity_bytes(
                                visible_context, query_rows));
    }
    // Dense prefill is fused and needs no workspace; the qualification-only P129 tail reuses
    // the single-row score workspace after the dense launch.
    if (ops::r9700::kv::use_text_p129_wmma_tail(query_rows, visible_context)) {
        required = std::max(required,
                            r9700_full_attention_score_workspace_capacity_bytes(visible_context));
    }
    return required;
}

// Validates one call against its cache capability and forms the raw Op arguments (without route
// workspace).
static hipError_t raw_attention_args(const R9700FullAttentionArgs& a, hipStream_t stream,
                                     ops::r9700::kv::Fp8Int4KvAttentionArgs& raw) noexcept {
    if (!a.cache_read.valid_for(stream) ||
        (a.ancestor_masks == nullptr) != (a.prefix_lengths == nullptr) ||
        (a.ancestor_masks != nullptr && a.row_positions == nullptr) ||
        a.prefix_length_stride > 1U) {
        return hipErrorInvalidValue;
    }
    const auto& cache                    = a.cache_read.layer();
    const std::uint32_t mapped_pages     = a.cache_read.mapped_pages();
    const std::uint32_t visible_frontier = a.cache_read.visible_frontier();
    if (cache.head_dim != static_cast<std::int32_t>(kHeadDim) ||
        cache.num_kv_heads != static_cast<std::int32_t>(kKvHeads) ||
        (cache.value_group != 16 && cache.value_group != 32) ||
        !layout_valid(cache.plane_layouts.key) || !layout_valid(cache.plane_layouts.value) ||
        !layout_valid(cache.plane_layouts.value_scale)) {
        return hipErrorInvalidValue;
    }

    const std::int32_t rows = a.query.ne[2];
    if (rows <= 0 || rows > static_cast<std::int32_t>(kMaximumRows) ||
        !exact_contiguous_shape(a.query, DType::BF16, kHeadDim, kQueryHeads, rows) ||
        !exact_contiguous_shape(a.output, DType::FP32, kHeadDim, kQueryHeads, rows)) {
        return hipErrorInvalidValue;
    }

    const std::uint64_t key_elements               = tensor_elements(cache.key_fp8_pages);
    const std::uint64_t value_elements             = tensor_elements(cache.value_int4_pages);
    const std::uint64_t scale_elements             = tensor_elements(cache.value_scale_fp16_pages);
    constexpr std::uint64_t key_elements_per_token = kHeadDim * kKvHeads;
    if (!contiguous_elements_at_least(cache.key_fp8_pages, DType::FP8_E4M3FN, 1U) ||
        !contiguous_elements_at_least(cache.value_int4_pages, DType::U8, 1U) ||
        !contiguous_elements_at_least(cache.value_scale_fp16_pages, DType::FP16, 1U) ||
        key_elements == 0U || key_elements % key_elements_per_token != 0U) {
        return hipErrorInvalidValue;
    }
    const std::uint64_t physical_tokens  = key_elements / key_elements_per_token;
    const std::uint64_t groups_per_token = kHeadDim / static_cast<std::uint32_t>(cache.value_group);
    if (physical_tokens == 0U || physical_tokens % kPageSize != 0U ||
        value_elements != physical_tokens * kKvHeads * (kHeadDim / 2U) ||
        scale_elements != physical_tokens * kKvHeads * groups_per_token ||
        physical_tokens > std::numeric_limits<std::size_t>::max()) {
        return hipErrorInvalidValue;
    }

    const std::uint64_t table_elements = tensor_elements(cache.block_table);
    const std::uint64_t frontier       = visible_frontier;
    if (!contiguous_elements_at_least(cache.block_table, DType::I32, mapped_pages) ||
        mapped_pages > physical_tokens / kPageSize || mapped_pages > table_elements ||
        frontier > static_cast<std::uint64_t>(mapped_pages) * kPageSize ||
        frontier > table_elements * kPageSize ||
        frontier > std::numeric_limits<std::size_t>::max() ||
        frontier > std::numeric_limits<std::uint32_t>::max()) {
        return hipErrorInvalidValue;
    }

    raw = ops::r9700::kv::Fp8Int4KvAttentionArgs{
        .query            = static_cast<const hip_bfloat16*>(a.query.data),
        .key_codes        = static_cast<const std::uint8_t*>(cache.key_fp8_pages.data),
        .value_codes      = static_cast<const std::uint8_t*>(cache.value_int4_pages.data),
        .value_scale_bits = static_cast<const std::uint16_t*>(cache.value_scale_fp16_pages.data),
        .page_table       = a.cache_read.device_table_row() == nullptr
                                ? static_cast<const std::uint32_t*>(cache.block_table.data)
                                : a.cache_read.pool_block_tables(),
        .page_table_pages = mapped_pages,
        .page_table_row   = a.cache_read.device_table_row(),
        .page_table_row_stride = a.cache_read.pool_table_row_stride(),
        .page_table_row_count  = a.cache_read.pool_table_row_count(),
        .context               = static_cast<std::size_t>(frontier),
        .physical_tokens       = static_cast<std::size_t>(physical_tokens),
        .query_heads           = kQueryHeads,
        .kv_heads              = kKvHeads,
        .q_to_kv               = kQueriesPerKv,
        .value_group           = static_cast<std::uint32_t>(cache.value_group),
        .query_rows            = static_cast<std::uint32_t>(rows),
        .row_positions         = a.row_positions,
        .ancestor_masks        = a.ancestor_masks,
        .prefix_lengths        = a.prefix_lengths,
        .prefix_length_stride  = a.prefix_length_stride,
        .active_query_rows     = a.active_query_rows,
        .attention_scale       = kQwenAttentionScale,
        .key_layout            = raw_layout(cache.plane_layouts.key),
        .value_layout          = raw_layout(cache.plane_layouts.value),
        .value_scale_layout    = raw_layout(cache.plane_layouts.value_scale),
        .output                = static_cast<float*>(a.output.data),
    };
    return hipSuccess;
}

// The packed decode route's cell: host-fixed causal 1..8 rows, G16, token-fastest FP8 keys and
// feature-fastest INT4/FP16 values and scales.
[[nodiscard]] static bool packed_decode_route(const ops::r9700::kv::Fp8Int4KvAttentionArgs& raw,
                                              bool tree_or_device_count) noexcept {
    return ops::r9700::kv::use_packed_decode_attention(raw.query_rows, raw.context,
                                                       tree_or_device_count) &&
           raw.value_group == 16U &&
           raw.key_layout == ops::r9700::kv::Fp8Int4KvPlaneLayout::TokenFastest &&
           raw.value_layout == ops::r9700::kv::Fp8Int4KvPlaneLayout::FeatureFastest &&
           raw.value_scale_layout == ops::r9700::kv::Fp8Int4KvPlaneLayout::FeatureFastest;
}

static hipError_t launch_r9700_attention(const R9700FullAttentionArgs& a,
#if defined(NINFER_R9700_XATTENTION_QUALIFICATION)
                                         bool xattention_prefill, void* xattention_workspace,
                                         std::size_t xattention_workspace_bytes,
#endif
                                         hipStream_t stream) noexcept {
    ops::r9700::kv::Fp8Int4KvAttentionArgs raw{};
    if (const hipError_t error = raw_attention_args(a, stream, raw); error != hipSuccess) {
        return error;
    }
    const std::int32_t rows = a.query.ne[2];
#if defined(NINFER_R9700_XATTENTION_QUALIFICATION)
    if (xattention_prefill) {
        if (a.ancestor_masks != nullptr || a.prefix_lengths != nullptr ||
            a.active_query_rows != nullptr || rows < 2) {
            return hipErrorInvalidValue;
        }
#    if defined(NINFER_R9700_XATTENTION_KEEP_DIAGNOSTIC)
        const ops::r9700::kv::Fp8Int4KvXAttentionArgs traced{raw, xattention_workspace,
                                                             xattention_workspace_bytes};
        const auto status = ops::r9700::kv::fp8_int4_kv_xattention_prefill(traced, stream);
        return status == hipSuccess ? trace_xattention_keep(traced, stream) : status;
#    else
        return ops::r9700::kv::fp8_int4_kv_xattention_prefill(
            ops::r9700::kv::Fp8Int4KvXAttentionArgs{raw, xattention_workspace,
                                                    xattention_workspace_bytes},
            stream);
#    endif
    }
#endif
    const bool tree_or_device_count = a.ancestor_masks != nullptr || a.active_query_rows != nullptr;
    // Host-fixed rows that would underfill the unsplit dense grid take the split route first.
    if (ops::r9700::kv::use_mid_rows_attention(static_cast<std::uint32_t>(rows), raw.context,
                                               tree_or_device_count) &&
        a.ancestor_masks == nullptr && a.prefix_lengths == nullptr &&
        raw.key_layout == ops::r9700::kv::Fp8Int4KvPlaneLayout::TokenFastest &&
        raw.value_layout == ops::r9700::kv::Fp8Int4KvPlaneLayout::FeatureFastest &&
        raw.value_scale_layout == ops::r9700::kv::Fp8Int4KvPlaneLayout::FeatureFastest) {
        const std::size_t required = ops::r9700::kv::fp8_int4_kv_attention_mid_rows_workspace_bytes(
            raw.context, static_cast<std::uint32_t>(rows));
        if (required == 0U || a.workspace == nullptr || a.workspace_bytes < required) {
            return hipErrorInvalidValue;
        }
        auto mid            = raw;
        mid.score_workspace = static_cast<float*>(a.workspace);
        return ops::r9700::kv::fp8_int4_kv_attention_mid_rows(mid, stream);
    }
    const bool dense_prefill =
        ops::r9700::kv::use_dense_prefill_attention(static_cast<std::uint32_t>(rows),
                                                    raw.context) &&
        a.ancestor_masks == nullptr && a.prefix_lengths == nullptr &&
        raw.key_layout == ops::r9700::kv::Fp8Int4KvPlaneLayout::TokenFastest &&
        raw.value_layout == ops::r9700::kv::Fp8Int4KvPlaneLayout::FeatureFastest &&
        raw.value_scale_layout == ops::r9700::kv::Fp8Int4KvPlaneLayout::FeatureFastest;
    if (dense_prefill) {
        hipError_t error = ops::r9700::kv::fp8_int4_kv_attention_dense_prefill(raw, stream);
        if (error != hipSuccess || !ops::r9700::kv::use_text_p129_wmma_tail(
                                       static_cast<std::uint32_t>(rows), raw.context)) {
            return error;
        }
        const std::size_t tail_score_bytes =
            r9700_full_attention_score_workspace_capacity_bytes(raw.context);
        if (tail_score_bytes == 0U || a.workspace == nullptr ||
            a.workspace_bytes < tail_score_bytes) {
            return hipErrorInvalidValue;
        }
        // Overwrite only the final output row, leaving the 128-row prefix byte-for-byte on the
        // dense route.
        auto tail                          = raw;
        constexpr std::size_t row_elements = kQueryHeads * kHeadDim;
        tail.query += 128U * row_elements;
        tail.query_rows = 1U;
        tail.row_positions += 128U;
        tail.score_workspace = static_cast<float*>(a.workspace);
        tail.output += 128U * row_elements;
        return ops::r9700::kv::fp8_int4_kv_attention_wmma(tail, stream);
    }
    // Device active-row counts ride on packed decode, whose setup and merge zero inactive rows.
    if (packed_decode_route(raw, a.ancestor_masks != nullptr)) {
        const std::size_t required =
            ops::r9700::kv::fp8_int4_kv_attention_packed_decode_workspace_bytes(
                raw.context, static_cast<std::uint32_t>(rows));
        if (required == 0U || a.workspace == nullptr || a.workspace_bytes < required) {
            return hipErrorInvalidValue;
        }
        auto packed            = raw;
        packed.score_workspace = static_cast<float*>(a.workspace);
        return ops::r9700::kv::fp8_int4_kv_attention_packed_decode(packed, stream);
    }
    const bool split512 = ops::r9700::kv::use_split512_attention(static_cast<std::uint32_t>(rows),
                                                                 raw.context, tree_or_device_count);
    if (split512) {
        const std::size_t required = r9700_full_attention_workspace_capacity_bytes(
            static_cast<std::uint32_t>(rows), raw.context, tree_or_device_count);
        if (required == 0U || a.workspace == nullptr || a.workspace_bytes < required) {
            return hipErrorInvalidValue;
        }
        return ops::r9700::kv::fp8_int4_kv_attention_split512(raw, a.workspace, required, stream);
    }
    return ops::r9700::kv::fp8_int4_kv_attention_fused(raw, stream);
}

hipError_t r9700_qwen3_8_27b_full_attention(const R9700FullAttentionArgs& args,
                                            hipStream_t stream) noexcept {
    return launch_r9700_attention(args,
#if defined(NINFER_R9700_XATTENTION_QUALIFICATION)
                                  false, nullptr, 0U,
#endif
                                  stream);
}

std::size_t
r9700_full_attention_sequence_workspace_stride_bytes(std::uint32_t query_rows,
                                                     std::size_t visible_context) noexcept {
    if (!ops::r9700::kv::use_packed_decode_attention(query_rows, visible_context, false)) {
        return 0U;
    }
    const std::size_t bytes = ops::r9700::kv::fp8_int4_kv_attention_packed_decode_workspace_bytes(
        visible_context, query_rows);
    return (bytes + kSequenceWorkspaceAlignment - 1U) / kSequenceWorkspaceAlignment *
           kSequenceWorkspaceAlignment;
}

bool r9700_full_attention_sequences_supported(std::span<const R9700FullAttentionArgs> sequences,
                                              hipStream_t stream) noexcept {
    if (sequences.empty() || sequences.size() > ops::r9700::kv::kPackedDecodeMaximumSequences) {
        return false;
    }
    for (const R9700FullAttentionArgs& sequence : sequences) {
        ops::r9700::kv::Fp8Int4KvAttentionArgs raw{};
        if (sequence.ancestor_masks != nullptr || sequence.prefix_lengths != nullptr ||
            sequence.active_query_rows != nullptr ||
            raw_attention_args(sequence, stream, raw) != hipSuccess ||
            !packed_decode_route(raw, false)) {
            return false;
        }
    }
    return true;
}

hipError_t
r9700_qwen3_8_27b_full_attention_sequences(std::span<const R9700FullAttentionArgs> sequences,
                                           hipStream_t stream) noexcept {
    if (!r9700_full_attention_sequences_supported(sequences, stream)) {
        return hipErrorInvalidValue;
    }
    std::array<ops::r9700::kv::Fp8Int4KvAttentionArgs,
               ops::r9700::kv::kPackedDecodeMaximumSequences>
        raw{};
    for (std::size_t i = 0; i < sequences.size(); ++i) {
        const R9700FullAttentionArgs& sequence = sequences[i];
        if (const hipError_t error = raw_attention_args(sequence, stream, raw[i]);
            error != hipSuccess) {
            return error;
        }
        const std::size_t required =
            ops::r9700::kv::fp8_int4_kv_attention_packed_decode_workspace_bytes(raw[i].context,
                                                                                raw[i].query_rows);
        if (required == 0U || sequence.workspace == nullptr ||
            sequence.workspace_bytes < required) {
            return hipErrorInvalidValue;
        }
        raw[i].score_workspace = static_cast<float*>(sequence.workspace);
    }
    return ops::r9700::kv::fp8_int4_kv_attention_packed_decode(
        std::span<const ops::r9700::kv::Fp8Int4KvAttentionArgs>(raw.data(), sequences.size()),
        stream);
}

#if defined(NINFER_R9700_XATTENTION_QUALIFICATION)
std::size_t r9700_qwen3_8_27b_text_prefill_attention_workspace_capacity_bytes(
    std::uint32_t maximum_query_rows, std::size_t maximum_visible_context) noexcept {
    if (maximum_query_rows < 2U) return 0U;
    return ops::r9700::kv::fp8_int4_kv_xattention_workspace_capacity_bytes(
        kQueryHeads, kKvHeads, maximum_query_rows, maximum_visible_context);
}

hipError_t r9700_qwen3_8_27b_text_prefill_attention(const R9700FullAttentionArgs& args,
                                                    void* workspace, std::size_t workspace_bytes,
                                                    hipStream_t stream) noexcept {
    if (args.query.ne[2] < 2) return hipErrorInvalidValue;
    return launch_r9700_attention(args, true, workspace, workspace_bytes, stream);
}
#endif

} // namespace ninfer::targets::qwen3_8_27b::detail
