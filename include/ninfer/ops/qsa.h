#pragma once

#include "core/arena.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ninfer::ops {

// Qwen3.8-Flash-Next Qwen Sparse Attention (QSA) geometry. Core attention has 24 query heads of
// width 256 over two KV heads (query head h reads KV head h / 12). The indexer has four 128-wide
// query heads and one shared 128-wide key head and scores blocks of four consecutive tokens.
inline constexpr std::int32_t kQsaHeadDim         = 256;
inline constexpr std::int32_t kQsaQueryHeads      = 24;
inline constexpr std::int32_t kQsaKvHeads         = 2;
inline constexpr std::int32_t kQsaIndexHeadDim    = 128;
inline constexpr std::int32_t kQsaIndexQueryHeads = 4;
inline constexpr std::int32_t kQsaBlockTokens     = 4;
// The checkpoint's native position capacity, and therefore the largest state capacity.
inline constexpr std::int32_t kQsaMaximumCapacity = 262144;

/** Explicit storage format of the core K/V state planes; never inferred from nullable scales. */
enum class QsaKvFormat : std::uint8_t {
    BF16,
    // NVFP4 with one E4M3 scale per 16 consecutive features of a token's head and no tensor
    // divisor: scale = E4M3_rn_satfinite(amax / 6), code = E2M1_rn_satfinite(x / scale), with
    // every code zero when the scale encodes zero.
    NVFP4G16,
};

/**
 * Complete blocks a query may select. The checkpoint selects 512 blocks (2048 tokens); wider
 * budgets are startup choices for long-context retrieval.
 */
enum class QsaSelectBudget : std::int32_t {
    Blocks512  = 512,
    Blocks1024 = 1024,
    Blocks2048 = 2048,
};

/** Selected-id slots per query: every token of the budget's blocks plus an incomplete tail. */
[[nodiscard]] constexpr std::int32_t qsa_selected_capacity(QsaSelectBudget budget) {
    return kQsaBlockTokens * static_cast<std::int32_t>(budget) + kQsaBlockTokens - 1;
}

// QSA state pages: a page holds 64 consecutive logical token slots of one lane, i.e. 16 complete
// blocks, so a block never crosses a page. Equal to the paged KV pool's page size.
inline constexpr std::int32_t kQsaPageTokens = 64;
inline constexpr std::int32_t kQsaPageBlocks = kQsaPageTokens / kQsaBlockTokens;

/**
 * Persistent paged QSA state of one layer for every lane: the planes of a PageMajor paged pool
 * (one physical page id addresses the same page of every plane) plus its block-table matrix.
 * With P physical pages, every plane is a contiguous, 16-byte aligned [lead, 64, heads, P]
 * tensor (lead fastest, then the token's slot within the page, then the head, then the page):
 *
 *   k, v        BF16 [256,64,2,P] for QsaKvFormat::BF16, or U8 [128,64,2,P] NVFP4 codes (feature
 *               2i in the low nibble of byte i) for QsaKvFormat::NVFP4G16.
 *   k_scales,   FP8_E4M3FN [16,64,2,P], the scale of features 16g..16g+15, for NVFP4G16 only;
 *   v_scales    empty for BF16.
 *   index_keys  BF16 [128,64,1,P], the raw indexer key before its norm and rotary embedding.
 *   positions   I32 [1,64,3,P], the temporal/height/width MRoPE coordinates of each slot
 *               (axis-major within a page). Text tokens carry the same value on all three axes.
 *   block_keys  FP32 [32,64,1,P]: per page, the 16 sealed block keys of 128 features, block
 *               after block (key feature f of in-page block q is element 128 q + f of the page).
 *
 * block_tables is contiguous I32 [L, rows]: lane (table row) r maps its logical page p, i.e. its
 * logical slots [64p, 64p+64), to physical page block_tables[p, r]. A lane's capacity is
 * C = 64 L <= kQsaMaximumCapacity; slot s of block b = floor(s/4) of lane r lives at in-page
 * position s % 64 of page block_tables[floor(s/64), r]. A physical page outside [0,P) is treated
 * as unmapped: nothing is written to it, its slots are skipped by attention, and selection reads
 * no key from it (the caller maps every page holding a slot some query sees).
 *
 * K is the normalized and rotated core key; V the projected value. The sealed key of block b of a
 * lane is
 *
 *     pooled[b] = BF16_rn(0.25 * (((k[4b] + k[4b+1]) + k[4b+2]) + k[4b+3]))   (a semantic cast)
 *     key[b]    = rope_{4b}(norm(pooled[b], key_norm_weight))
 *
 * of the raw keys k[s] = index_keys at slot s and the position stored in the block's slots when
 * its last slot 4b+3 was last appended (see qsa_state_append); the sum and scale are FP32 in slot
 * order: the source model's FP32 mean, whose rounding before the BF16 cast decides which side of a
 * BF16 rounding midpoint the pooled key lands on, so it is part of the semantics, not an
 * implementation choice. norm(x, w) = x / sqrt(mean(x^2) + 1e-6) * (1 + w), and rope_p is the
 * 64-wide MRoPE at the three-axis position of slot p: it rotates the first 64 features as 32
 * rotate-half pairs (i, i+32) by the angle position[i % 3] * 1e7^(-2i/64), so pairs owned by the
 * temporal, height and width axes interleave (sections [11,11,10]); features 64..127 pass through.
 * Implementation profile: the norm and rotary embedding run in FP32 with the rotary angle reduced
 * in FP64.
 *
 * The view owns no frontier: which slots hold committed tokens and which pages a lane maps are
 * the caller's state. A block's sealed key reflects its current raw keys when the caller appends
 * a block's last slot no earlier than its other slots (the causal stream order, also after a
 * rollback that rewrites slots): block_keys is the selection's only source of block keys, so it
 * is read only for blocks every slot of which the querying token sees. Planes and the block
 * tables are pairwise disjoint.
 */
struct QsaPagedState {
    QsaKvFormat format = QsaKvFormat::BF16;
    Tensor k;
    Tensor v;
    Tensor k_scales;
    Tensor v_scales;
    Tensor index_keys;
    Tensor positions;
    Tensor block_keys;
    Tensor block_tables;
};

/**
 * The paged-pool plane specs of one QSA layer in the format, in QsaPagedState order: k, v,
 * (k_scales, v_scales for NVFP4G16 only), index_keys, positions, block_keys. Building a PageMajor
 * PagedKVPool from them (as the planes of one layer, possibly among other layers' planes) yields
 * planes the QSA Ops accept.
 */
[[nodiscard]] std::vector<PagedKVPlaneSpec> qsa_paged_plane_specs(QsaKvFormat format);

/*
 * Lane-batched calls. B rows of W columns each are packed column-major by row: column j of row b
 * is packed index t = b*W + j. lanes is device I32 [B]: row b reads and writes the state of
 * block-table row lanes[b]. slots is device I32 [W,B]: the logical slot of each column in its
 * lane, or -1 for a padding column. A row whose lane lies outside [0, rows) behaves as if every
 * one of its slots were -1. Every row is independent: the result of each column depends only on
 * its own inputs and its lane's state, so a batched call equals one call per row (for
 * qsa_selected_attention, one call per row of the same packed width; see there).
 */

/**
 * Op: qsa_state_append
 *
 * For every column t (row b, lane r = lanes[b]) with slots[t] >= 0, writes the column's core K/V,
 * raw indexer key and MRoPE position into slot s = slots[t] of lane r:
 *
 *   k, v        BF16 [256,2,W,B]: copied bit-exactly into BF16 state, or encoded per 16-feature
 *               group with the NVFP4G16 codec into NVFP4 state.
 *   index_keys  BF16 [128,W,B]: copied bit-exactly.
 *   positions   I32 [W,3,B] (axis-major per row: axis a of column j of row b is element
 *               3Wb + Wa + j): copied exactly.
 *   slots       I32 [W,B], lanes I32 [B], on the device.
 *
 * Then, for every written slot 4q+3 (the last slot of block q), seals block q of lane r:
 * its block key becomes key[q] (see QsaPagedState) of the raw keys and position the lane holds in
 * slots [4q, 4q+4) after every column of this call is written, with key_norm_weight the source
 * checkpoint's zero-centered BF16 [128] indexer key norm weight. Slots appended in earlier calls
 * are read from the state.
 *
 * The caller promises that the written (lane, slot) pairs are unique; a slot outside [0,C) or in
 * an unmapped page writes nothing. Every other slot and block key is left unchanged and no
 * frontier is advanced. 1 <= W*B <= 65535; every tensor is contiguous and 16-byte aligned, and
 * inputs and state are pairwise disjoint. No workspace. Invalid arguments throw
 * std::invalid_argument before any work is queued. Asynchronous on stream and graph-capturable
 * for fixed W, B and addresses.
 */
void qsa_state_append(const Tensor& k, const Tensor& v, const Tensor& index_keys,
                      const Tensor& positions, const Tensor& slots, const Tensor& lanes,
                      const Tensor& key_norm_weight, QsaPagedState& state, cudaStream_t stream);

/**
 * Workspace bytes of qsa_index_select for `queries` = W*B packed query columns whose visible
 * prefixes are at most max_visible_tokens long. Zero when no query can have more complete blocks
 * than the budget.
 */
[[nodiscard]] std::size_t qsa_index_select_workspace_bytes(std::int32_t queries,
                                                           std::int32_t max_visible_tokens,
                                                           QsaSelectBudget budget);

/**
 * Op: qsa_index_select
 *
 * Query column t (index_query BF16 [128,4,W,B]; row b, lane r = lanes[b]) occupies slot
 * s = slots[t] of lane r and sees the causal prefix of that lane's logical slots [0,s],
 * n = s + 1 tokens, all already appended. Complete block q holds slots [4q, 4q+4) for
 * q < Q = floor(n/4); slots [4Q, n) form the incomplete tail. With key[q] the lane's sealed key of
 * block q and norm and rope_s as defined at QsaPagedState:
 *
 *   query[h]  = rope_s(norm(index_query[:, h, t], query_norm_weight))
 *   score[q]  = sum_{h<4} ReLU(dot(query[h], key[q])) / sqrt(128)
 *
 * If Q <= budget every visible slot is selected. Otherwise the `budget` highest-scoring blocks
 * win, an exact score tie going to the lower block index. selected_ids[:, t] (I32 [S,W,B],
 * S = qsa_selected_capacity(budget)) lists the four logical slots of every selected block in
 * ascending block order, then the tail; the selection is a set, ordered for locality, since
 * attention over it does not depend on order. The remaining entries are -1 and
 * selected_count[t] (I32 [W,B]) is the number of listed slots.
 *
 * Implementation profile: the query norm, rotary embedding, dots and ReLU sum run in FP32 with
 * the rotary angle reduced in FP64; blocks are ranked by the unscaled ReLU sum, which orders like
 * the score. A query's scores do not depend on the other rows of the call.
 *
 * query_norm_weight is the source checkpoint's zero-centered BF16 [128] weight. A padding column
 * (slot -1) has count zero. max_visible_tokens, a host bound in [1,C], bounds every query's
 * prefix (s < max_visible_tokens); it sizes the launch, not the visibility. When some query ranks
 * blocks, block keys below floor(max_visible_tokens / 4) are read only for the blocks some query
 * of the same lane sees. A slot outside [0,max_visible_tokens) yields an empty column instead of
 * an out-of-bounds read. workspace must hold qsa_index_select_workspace_bytes(W*B,
 * max_visible_tokens, budget) and its allocations are released on return. 1 <= W*B <= 65535.
 * Every tensor is contiguous and 16-byte aligned; inputs, state and outputs are pairwise disjoint
 * and state is not mutated. Invalid arguments throw std::invalid_argument before any work is
 * queued. Asynchronous on stream and graph-capturable for fixed W, B, max_visible_tokens, budget
 * and addresses.
 */
void qsa_index_select(const Tensor& index_query, const QsaPagedState& state, const Tensor& slots,
                      const Tensor& lanes, const Tensor& query_norm_weight, QsaSelectBudget budget,
                      std::int32_t max_visible_tokens, Tensor& selected_ids, Tensor& selected_count,
                      WorkspaceArena& workspace, cudaStream_t stream);

/**
 * Workspace bytes of qsa_selected_attention for `queries` = W*B packed query columns and S
 * selected-id slots per query.
 */
[[nodiscard]] std::size_t qsa_selected_attention_workspace_bytes(std::int32_t queries,
                                                                 std::int32_t selected_slots);

/**
 * Op: qsa_selected_attention
 *
 * For query column t (row b, lane r = lanes[b]) and query head h, with KV head g = h / 12 and the
 * selected logical slots J = selected_ids[0 .. selected_count[t]-1, t] of lane r:
 *
 *   logit[j] = dot(q[:, h, t], K[:, j, g]) / 16                    (1/sqrt(256))
 *   out[:, h, t] = sum_{j in J} softmax(logit)[j] * V[:, j, g]
 *
 * K and V are read from the BF16 state exactly, or decoded from the NVFP4G16 state (E2M1 code
 * times its E4M3 scale). q and out are BF16 [256,24,W,B]; q is the normalized, rotated query.
 * selected_ids is I32 [S,W,B] with 1 <= S <= qsa_selected_capacity(QsaSelectBudget::Blocks2048),
 * selected_count I32 [W,B]. The caller promises that each listed prefix holds distinct slots; a
 * listed slot outside [0,C) or in an unmapped page is skipped rather than read. A count of zero, a
 * count outside [0,S], or a row whose lane is out of range produces an exact-zero output column.
 *
 * Implementation profile: logits are FP32-accumulated sums of exact BF16 products (BF16
 * tensor-core MMA; NVFP4 K decodes exactly to BF16); the softmax (max-subtracted exp and sum) is
 * FP32; each weight enters the BF16 tensor-core value sum as the sum of two BF16 values (error at
 * most 2^-16 of the weight, or 2^-134 for weights below the BF16 normal range) with FP32
 * accumulation; the output is rounded once to BF16. Short query batches split each query's
 * selected slots across CTAs and merge the FP32 partial softmax states; the split count depends
 * only on W*B and S, so a column's output is bit-identical to a call of the same packed width in
 * which the other columns are padding.
 *
 * workspace must hold qsa_selected_attention_workspace_bytes(W*B, S); its allocations are
 * released on return. 1 <= W*B <= 65535. Every tensor is contiguous and 16-byte aligned; inputs,
 * state and out are pairwise disjoint and state is not mutated. Invalid arguments throw
 * std::invalid_argument before any work is queued. Asynchronous on stream and graph-capturable
 * for fixed W, B, S and addresses.
 */
void qsa_selected_attention(const Tensor& q, const Tensor& selected_ids,
                            const Tensor& selected_count, const Tensor& lanes,
                            const QsaPagedState& state, Tensor& out, WorkspaceArena& workspace,
                            cudaStream_t stream);

} // namespace ninfer::ops
