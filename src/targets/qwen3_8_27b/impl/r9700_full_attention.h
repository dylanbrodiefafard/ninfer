#pragma once

#include "core/fp8_int4_paged_kv_cache.h"

#include <ninfer/targets/qwen3/decoder_state.h>

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::targets::qwen3_8_27b::detail {

// Target-private bridge from the Qwen sequence publication/cache view into the owned R9700 A3
// Op. Qwen3.8 is presently registered through this package and has one fixed full-attention
// geometry: D256, Hq24, Hkv4, Q:KV=6, any positive T within the target's 262,144-token
// native context capacity, and scale 1/sqrt(256).
struct R9700FullAttentionArgs {
    Tensor query; // contiguous BF16 [256,24,T]
    qwen3::PagedKVLayerRead cache_read;
    const std::int32_t* row_positions = nullptr; // optional device I32 [T], each in [0,frontier)
    // Optional paired device I32 packed-tree visibility metadata: masks are [T], while prefixes
    // are [T] or one broadcast scalar. The raw Op validates represented values per row; this
    // bridge validates atomic pairing and the explicit zero-or-one prefix stride.
    const std::int32_t* ancestor_masks = nullptr;
    const std::int32_t* prefix_lengths = nullptr;
    std::uint32_t prefix_length_stride = 1; // zero broadcasts one prefix; one is rowwise
    // Optional device I32 scalar. Null activates all T rows; a valid scalar activates its prefix
    // and zeroes the fixed-width tail without reading tail metadata or cache state.
    const std::int32_t* active_query_rows = nullptr;
    // Caller-owned storage at the planner's stable peak. Dense initial-prefix calls use the
    // rectangular FP32 score plane plus maxima; decode uses the packed or split arena.
    void* workspace             = nullptr;
    std::size_t workspace_bytes = 0;
    Tensor output; // contiguous FP32 [256,24,T]
};

// Returns the exact caller-owned score storage for one ordinary causal WMMA call, or zero for an
// invalid/overflowing visible context or a compile-time score-streaming control build.
[[nodiscard]] std::size_t
r9700_full_attention_score_workspace_capacity_bytes(std::size_t visible_context) noexcept;

// Returns the caller-owned peak for every production route reachable within these envelope
// maxima, including an initial-prefix dense P<=4096 score plane. This is suitable for startup
// planning; a concrete call consumes only its exact selected-route prefix of the stable span.
[[nodiscard]] std::size_t
r9700_full_attention_workspace_capacity_bytes(std::uint32_t query_rows, std::size_t visible_context,
                                              bool tree_or_device_count) noexcept;

// Dispatches dense prefill through staged full-score GQA6. G16 causal 1..8-row decode
// (ordinary, MTP, DFlash chain verification), including device-selected active prefixes, uses
// the packed route for contexts 64..262144; tree T=4 uses split-512, and other cells the fused
// leaf. The default-off Text parity profile additionally overwrites only P129's tail with W1 WMMA.
[[nodiscard]] hipError_t r9700_qwen3_8_27b_full_attention(const R9700FullAttentionArgs& args,
                                                          hipStream_t stream) noexcept;

// Sequence batch of one compact decode round (at most kMaximumConcurrency sequences of one cache,
// each with its own read capability, positions, output and workspace): true when every sequence is
// a non-tree host-fixed call on the packed decode route, which then serves the whole batch as one
// split launch and one merge launch with each sequence's output bytes equal to its single call.
[[nodiscard]] bool
r9700_full_attention_sequences_supported(std::span<const R9700FullAttentionArgs> sequences,
                                         hipStream_t stream) noexcept;
// One sequence's 256-byte-aligned slice of the batch workspace, or zero off the packed route. It
// is monotonic in rows and context, so a stride sized at the envelope maxima covers every call.
[[nodiscard]] std::size_t
r9700_full_attention_sequence_workspace_stride_bytes(std::uint32_t query_rows,
                                                     std::size_t visible_context) noexcept;
[[nodiscard]] hipError_t
r9700_qwen3_8_27b_full_attention_sequences(std::span<const R9700FullAttentionArgs> sequences,
                                           hipStream_t stream) noexcept;

#if defined(NINFER_R9700_XATTENTION_QUALIFICATION)
// Private model-gate route compiled only into an explicitly configured qualification build.
// T>=2 ordinary prefill consumes the B128 XAttention evaluator; the Variant keeps T=1 on its
// existing dense leaf before entering this boundary.
[[nodiscard]] std::size_t r9700_qwen3_8_27b_text_prefill_attention_workspace_capacity_bytes(
    std::uint32_t maximum_query_rows, std::size_t maximum_visible_context) noexcept;
[[nodiscard]] hipError_t
r9700_qwen3_8_27b_text_prefill_attention(const R9700FullAttentionArgs& args, void* workspace,
                                         std::size_t workspace_bytes, hipStream_t stream) noexcept;
#endif

} // namespace ninfer::targets::qwen3_8_27b::detail
