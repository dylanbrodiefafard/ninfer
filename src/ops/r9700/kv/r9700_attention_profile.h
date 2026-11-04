#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace ninfer::ops::r9700::kv {

// Compile-time numerical/performance profile of the long-context split-512 leaf for tree or
// device-selected T=4 schedules. Zero is the separately built PPL control profile and retains the
// fused represented-BF16-Q leaf there.
// This is not an artifact, CLI, or runtime selector.
#ifndef NINFER_R9700_FP8_QK_WMMA
#    define NINFER_R9700_FP8_QK_WMMA 1
#endif
static_assert(NINFER_R9700_FP8_QK_WMMA == 0 || NINFER_R9700_FP8_QK_WMMA == 1,
              "R9700 FP8-Q/K WMMA profile must be zero or one");
inline constexpr bool kFp8QkWmmaDecode                     = NINFER_R9700_FP8_QK_WMMA != 0;
inline constexpr std::uint32_t kPackedDecodeMinimumContext = 64U;
inline constexpr std::size_t kPackedDecodeMaximumContext   = 262144U;
inline constexpr std::uint32_t kSplit512MinimumContext     = 8192U;
inline constexpr std::uint32_t kDensePrefillMinimumRows    = 128U;
inline constexpr std::uint32_t kDensePrefillMaximumRows    = 8192U;
inline constexpr std::size_t kDensePrefillMaximumContext   = 262144U;

// Qualification-only Text P129 WMMA tail candidate. Zero retains the dense P129 route; one admits
// the exact P129 tail overwrite. The packed decode route is a production route and is not gated by
// this flag.
#ifndef NINFER_R9700_TEXT_P129_WMMA_TAIL_CANDIDATE
#    define NINFER_R9700_TEXT_P129_WMMA_TAIL_CANDIDATE 0
#endif
static_assert(NINFER_R9700_TEXT_P129_WMMA_TAIL_CANDIDATE == 0 ||
                  NINFER_R9700_TEXT_P129_WMMA_TAIL_CANDIDATE == 1,
              "R9700 Text P129 WMMA tail candidate must be zero or one");
inline constexpr bool kTextP129WmmaTailCandidate = NINFER_R9700_TEXT_P129_WMMA_TAIL_CANDIDATE == 1;

// Causal prefill covers initial and appended chunks with absolute positions and the complete
// visible cache frontier.
[[nodiscard]] constexpr bool use_dense_prefill_attention(std::uint32_t query_rows,
                                                         std::size_t visible_context) noexcept {
    return query_rows >= kDensePrefillMinimumRows && query_rows <= kDensePrefillMaximumRows &&
           visible_context >= query_rows && visible_context <= kDensePrefillMaximumContext;
}

// Mid-row split: the dense-prefill tiles (4 KV heads x 32-row tiles per CTA, one resident CTA per
// compute unit) split over context chunks into FP32 partials, merged stably. The chunk count
// minimizes whole waves of kMidRowsWaveCtas CTAs times the keys each CTA streams plus a fixed
// per-CTA overhead, within at most kMidRowsMaximumChunks chunks of at least
// kMidRowsMinimumChunkKeys keys and kMidRowsPartialRows row-chunk partial slots (50.7 MiB).
inline constexpr std::uint32_t kMidRowsMinimumRows      = 9U;
inline constexpr std::uint32_t kMidRowsMaximumRows      = 1023U;
inline constexpr std::uint32_t kMidRowsWaveCtas         = 64U;
inline constexpr std::uint32_t kMidRowsMaximumChunks    = 64U;
inline constexpr std::uint32_t kMidRowsMinimumChunkKeys = 256U;
inline constexpr std::uint32_t kMidRowsPartialRows      = 2048U;
inline constexpr std::size_t kMidRowsChunkOverheadKeys  = 64U;

// Largest chunk count for (context, rows): monotonic in the context, so a workspace sized at a
// frontier envelope covers every smaller context.
[[nodiscard]] constexpr std::uint32_t mid_rows_chunk_limit(std::size_t visible_context,
                                                           std::uint32_t query_rows) noexcept {
    std::size_t limit = kMidRowsMaximumChunks;
    if (query_rows != 0U) limit = std::min<std::size_t>(limit, kMidRowsPartialRows / query_rows);
    limit = std::min<std::size_t>(limit, (visible_context + kMidRowsMinimumChunkKeys - 1U) /
                                             kMidRowsMinimumChunkKeys);
    return static_cast<std::uint32_t>(std::max<std::size_t>(limit, 1U));
}

[[nodiscard]] constexpr std::uint32_t mid_rows_chunks(std::size_t visible_context,
                                                      std::uint32_t query_rows) noexcept {
    const std::size_t ctas    = 4U * ((static_cast<std::size_t>(query_rows) + 31U) / 32U);
    const std::uint32_t limit = mid_rows_chunk_limit(visible_context, query_rows);
    std::uint32_t best        = 1U;
    std::size_t best_cost     = 0U;
    for (std::uint32_t chunks = 1U; chunks <= limit; ++chunks) {
        const std::size_t waves = (ctas * chunks + kMidRowsWaveCtas - 1U) / kMidRowsWaveCtas;
        const std::size_t cost =
            waves * ((visible_context + chunks - 1U) / chunks + kMidRowsChunkOverheadKeys);
        if (chunks == 1U || cost < best_cost) {
            best      = chunks;
            best_cost = cost;
        }
    }
    return best;
}

// Host-fixed causal rows above the packed decode route (1..8), such as appended turns, tool
// results and prompt tails, take the split so a short call against a long cache keeps every CTA
// busy: always below dense prefill's 128 rows, and up to 1023 rows whenever the wave model splits
// the context (it takes precedence over dense prefill there).
[[nodiscard]] constexpr bool use_mid_rows_attention(std::uint32_t query_rows,
                                                    std::size_t visible_context,
                                                    bool tree_or_device_count) noexcept {
    return !tree_or_device_count && query_rows >= kMidRowsMinimumRows &&
           query_rows <= kMidRowsMaximumRows && visible_context >= query_rows &&
           visible_context <= kDensePrefillMaximumContext &&
           (query_rows < kDensePrefillMinimumRows ||
            mid_rows_chunks(visible_context, query_rows) >= 2U);
}

// Production packed decode route for 1..8 causal rows per sequence (ordinary decode, MTP and
// DFlash chain verification): non-tree execution with optional device-selected active prefixes and
// contexts 64..262144. The caller must additionally require G16, token-fastest FP8 keys, and
// feature-fastest INT4/FP16 values and scales; other cells retain the split/fused fallback.
[[nodiscard]] constexpr bool use_packed_decode_attention(std::uint32_t query_rows,
                                                         std::size_t visible_context,
                                                         bool tree_or_device_count) noexcept {
    return (query_rows >= 1U && query_rows <= 8U) && !tree_or_device_count &&
           visible_context >= kPackedDecodeMinimumContext &&
           visible_context <= kPackedDecodeMaximumContext;
}

[[nodiscard]] constexpr bool use_text_p129_wmma_tail(std::uint32_t query_rows,
                                                     std::size_t visible_context) noexcept {
    return kTextP129WmmaTailCandidate && query_rows == 129U && visible_context == 129U;
}

// Long-context T=4 fallback for tree or device-selected row metadata. The target dispatches
// non-tree device-selected prefixes to packed decode before reaching this fallback.
[[nodiscard]] constexpr bool use_split512_attention(std::uint32_t query_rows,
                                                    std::size_t visible_context,
                                                    bool tree_or_device_count) noexcept {
    return kFp8QkWmmaDecode && visible_context >= kSplit512MinimumContext && query_rows == 4U &&
           tree_or_device_count;
}

} // namespace ninfer::ops::r9700::kv
