#pragma once

#include "ninfer/ops/qsa.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {

// The launchers trust the wrapper's validation (shapes, dtypes, 16-byte alignment, disjointness,
// 1 <= width <= 65535). Device pointers address the planes documented in ninfer/ops/qsa.h.

// Raw device planes of a QsaPagedState. Scale planes are null for BF16 state.
struct QsaStatePlanes {
    void* k                          = nullptr;
    void* v                          = nullptr;
    std::uint8_t* k_scales           = nullptr;
    std::uint8_t* v_scales           = nullptr;
    std::uint16_t* index_keys        = nullptr;
    std::int32_t* positions          = nullptr;
    float* block_keys                = nullptr;
    const std::int32_t* block_tables = nullptr; // I32 [logical_pages, table_rows]
    std::int32_t logical_pages       = 0;       // a lane's capacity is 64 * logical_pages
    std::int32_t table_rows          = 0;
    std::int32_t physical_pages      = 0;
};

// The packed [W, B] column layout of a lane-batched call: column t belongs to row t / width,
// whose block-table row is lanes[row]; slots[t] is its logical slot (-1 for padding).
struct QsaColumns {
    const std::int32_t* slots = nullptr;
    const std::int32_t* lanes = nullptr;
    std::int32_t width        = 0;
    std::int32_t rows         = 0;
};

void qsa_state_append_launch(QsaKvFormat format, const std::uint16_t* k, const std::uint16_t* v,
                             const std::uint16_t* index_keys, const std::int32_t* positions,
                             const QsaColumns& columns, const std::uint16_t* key_norm_weight,
                             const QsaStatePlanes& state, cudaStream_t stream);

// --- qsa_index_select --------------------------------------------------------------------------

// Upper bound of the FP32 block-score rows held at once; the queries are scored in chunks of at
// most max(1, kQsaScoreBytes / (4 * blocks)) so the workspace stays independent of W at long
// context. A chunk is a run of whole rows or a column range of one row (see the launcher).
inline constexpr std::size_t kQsaScoreBytes = std::size_t{32} << 20;

[[nodiscard]] constexpr std::int32_t qsa_score_chunk_queries(std::int32_t width,
                                                             std::int32_t blocks) {
    const auto rows = static_cast<std::int64_t>(kQsaScoreBytes / (sizeof(float) * blocks));
    return static_cast<std::int32_t>(std::clamp<std::int64_t>(rows, 1, width));
}

// Workspace of one selection, outermost dimension first: query FP32 [W*B][4][128] (normalized,
// rotated index queries) and scores FP32 [chunk][blocks] (unscaled ReLU sums of one query chunk).
// blocks is floor(max_visible_tokens / 4) > budget; with blocks <= budget every pointer is null.
struct QsaSelectWorkspace {
    float* query  = nullptr;
    float* scores = nullptr;
};

void qsa_index_select_launch(const std::uint16_t* index_query, const QsaStatePlanes& state,
                             const QsaColumns& columns, const std::uint16_t* query_norm_weight,
                             std::int32_t budget_blocks, std::int32_t max_visible_tokens,
                             std::int32_t* selected_ids, std::int32_t* selected_count,
                             const QsaSelectWorkspace& workspace, cudaStream_t stream);

// --- qsa_selected_attention --------------------------------------------------------------------

// Selected ranks one attention CTA processes per tile (16 per warp), and the fewest ranks a split
// is given.
inline constexpr std::int32_t kQsaAttentionKeyTile       = 64;
inline constexpr std::int32_t kQsaAttentionMinSplitSlots = 128;
// CTAs a launch aims for: two per SM of the 170-SM RTX 5090. Queries are split across CTAs only
// while W * 2 KV heads falls short of it.
inline constexpr std::int32_t kQsaAttentionTargetCtas = 340;

// Selected-slot ranks each CTA owns: a multiple of the key tile.
[[nodiscard]] constexpr std::int32_t qsa_attention_split_slots(std::int32_t width,
                                                               std::int32_t selected_slots) {
    const std::int32_t wanted =
        (kQsaAttentionTargetCtas + kQsaKvHeads * width - 1) / (kQsaKvHeads * width);
    const std::int32_t most =
        (selected_slots + kQsaAttentionMinSplitSlots - 1) / kQsaAttentionMinSplitSlots;
    const std::int32_t splits = std::clamp(wanted, 1, most);
    const std::int32_t slots  = (selected_slots + splits - 1) / splits;
    return (slots + kQsaAttentionKeyTile - 1) / kQsaAttentionKeyTile * kQsaAttentionKeyTile;
}

[[nodiscard]] constexpr std::int32_t qsa_attention_splits(std::int32_t width,
                                                          std::int32_t selected_slots) {
    const std::int32_t slots = qsa_attention_split_slots(width, selected_slots);
    return (selected_slots + slots - 1) / slots;
}

// Workspace of a split launch (splits > 1), outermost dimension first: accumulators FP32
// [width][24][splits][256] (unnormalized weighted value sums) and stats FP32
// [width][24][splits][2] (running maximum of the base-2 logits dot / 16 * log2(e), softmax
// denominator). Null for one split.
struct QsaAttentionWorkspace {
    float* accumulators = nullptr;
    float* stats        = nullptr;
};

// width is the packed column count W*B.
void qsa_selected_attention_launch(const std::uint16_t* q, const std::int32_t* selected_ids,
                                   const std::int32_t* selected_count, const QsaColumns& columns,
                                   QsaKvFormat format, const QsaStatePlanes& state,
                                   std::int32_t selected_slots, std::uint16_t* out,
                                   const QsaAttentionWorkspace& workspace, cudaStream_t stream);

} // namespace ninfer::ops::detail
