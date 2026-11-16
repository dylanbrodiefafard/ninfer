#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

// Qwen3.8-Flash-Next n-gram (PLE) geometry: 16 table rows of 160 values per token (eight bigram
// then eight trigram heads), concatenated into one 2560-wide embedding, injected into the four
// 2560-wide residual streams through a dilated (3) depthwise convolution of width 4.
inline constexpr std::int32_t kPleRowsPerToken = 16;
inline constexpr std::int32_t kPleRowWidth     = 160;
inline constexpr std::int32_t kPleEmbedding    = kPleRowsPerToken * kPleRowWidth;
inline constexpr std::int32_t kPleHidden       = 2560;
inline constexpr std::int32_t kPleStreams      = 4;
inline constexpr std::int32_t kPleChannels     = kPleHidden * kPleStreams;
inline constexpr std::int32_t kPleConvDilation = 3;
inline constexpr std::int32_t kPleConvWidth    = 4;
inline constexpr std::int32_t kPleConvHistory  = (kPleConvWidth - 1) * kPleConvDilation;
// int8_row table encoding: one FP16 scale (value = code * scale) then 160 signed int8 codes.
inline constexpr std::int32_t kPleInt8RowBytes = 2 + kPleRowWidth;

/**
 * Op: ple_decode_rows
 *
 * embedding[r*160 + j, t] = BF16(code[r,t,j] * scale[r,t]) for the 16 rows of each token, where
 * rows is U8 [162,16,T] holding int8_row records (little-endian FP16 scale, then 160 int8 codes)
 * in head order. The product of an int8 and an FP16 is exact in FP32; its BF16 rounding is the
 * Op's only rounding. embedding is BF16 [2560,T]. Both are contiguous and disjoint, rows is
 * 2-byte and embedding 16-byte aligned, T >= 1. No workspace; asynchronous on stream.
 */
void ple_decode_rows(const Tensor& rows, Tensor& embedding, cudaStream_t stream);

/**
 * Source-checkpoint PLE weights: the zero-centered BF16 RMSNorm weights `norm_key`,
 * `norm_query` and `norm_conv` [10240] (gamma = 1 + w, one 2560 slice per stream) and the
 * depthwise convolution `conv` BF16 [4,10240] (four taps per channel, oldest to newest, taps
 * fastest). Contiguous device tensors. Shapes in this header list ne[0] (fastest) first.
 */
struct PleWeights {
    Tensor norm_key;
    Tensor norm_query;
    Tensor norm_conv;
    Tensor conv;
};

/**
 * Workspace bytes of ple_inject and ple_inject_snapshot for any call with W*B <= max_tokens
 * packed columns.
 */
[[nodiscard]] std::size_t ple_inject_workspace_bytes(std::int32_t max_tokens);

/**
 * Op: ple_inject
 *
 * Math / indexing:
 *   B independent sequences (rows) of W columns each. For column j of row b and stream i of
 *   residual R[:, i, j, b], with key = key_projection[:, j, b] (10240) and value =
 *   value_projection[:, j, b] (2560), the FP32 projections of the column's decoded embedding,
 *   RMSNorm (epsilon 1e-6) over each 2560 slice, and gamma = 1 + w:
 *
 *     K_i  = RMSNorm(key_i) * gamma_key_i         Q_i = RMSNorm(R_i) * gamma_query_i
 *     s    = (K_i . Q_i) / sqrt(2560)             g   = sign(s) * sqrt(max(|s|, 1e-6))
 *     G_i  = sigmoid(g) * value                   C_i = BF16(RMSNorm(G_i) * gamma_conv_i)
 *     conv[c,j] = sum_k conv[c,k] * C[c, j - 3*(3-k)]   (taps oldest to newest)
 *     R_i += G_i + SiLU(conv_i)
 *
 *   where C at a column before 0 is entry 9 + (j - 3*(3-k)) of the row's initial history: the
 *   history slot initial_state_slots[b] of conv_states, BF16 [9,10240] per slot holding, per
 *   channel, C of the nine previous tokens oldest first (positions fastest). C crosses the
 *   represented BF16 boundary of the persistent history before the convolution reads it, as the
 *   source model's state does; the oracle applies that rounding. Everything else is FP32 and R is
 *   rounded once to BF16.
 *
 * Logical shapes:
 *   residual BF16 [2560,4,W,B] (updated in place), key_projection FP32 [10240,W,B],
 *   value_projection FP32 [2560,W,B], conv_states BF16 [9,10240,S], valid_columns empty (every row
 *   has W valid columns) or I32 [B] with every value in [0,W], and I32 [B] slot selectors with
 *   every slot in [0,S). 1 <= B <= 12.
 *
 * Effects:
 *   A row's valid prefix [0, valid_columns[b]) is updated; its invalid tail columns of R are
 *   unchanged and do not enter its history. ple_inject: W >= 1 and initial and final history are
 *   the same slot, state_slots[b], which receives the last nine C of (old history, valid prefix)
 *   (unchanged for an empty prefix). Rows must name distinct slots. ple_inject_snapshot:
 *   1 <= W <= 16; row b reads initial_state_slots[b] and, after each valid column j, writes the
 *   last nine C through column j to slot snapshot_base_slots[b] + j; other slots are unchanged.
 *   The caller reserves disjoint complete [base, base+W) intervals and prevents one row from
 *   writing another row's initial slot; a row's own initial slot may lie in its interval.
 *
 *   Each row depends only on its own columns and history, and every column's arithmetic is
 *   independent of B, W, and the packing, so a batched call is bit-identical to one call per row,
 *   and chunked calls with carried history equal one call over the concatenation.
 *
 * Workspace / execution:
 *   All tensors are contiguous, 16-byte aligned and pairwise disjoint; the selectors are device
 *   tensors. workspace holds ple_inject_workspace_bytes(W*B) and is released on return. Invalid
 *   arguments throw std::invalid_argument before any work is queued. Asynchronous on stream;
 *   graph-capturable for fixed B, W and addresses (slot and valid values may change per replay).
 */
void ple_inject(Tensor& residual, const Tensor& key_projection, const Tensor& value_projection,
                const PleWeights& weights, Tensor& conv_states, const Tensor& valid_columns,
                const Tensor& state_slots, WorkspaceArena& workspace, cudaStream_t stream);

/** Snapshot form of ple_inject; see its contract. */
void ple_inject_snapshot(Tensor& residual, const Tensor& key_projection,
                         const Tensor& value_projection, const PleWeights& weights,
                         Tensor& conv_states, const Tensor& valid_columns,
                         const Tensor& initial_state_slots, const Tensor& snapshot_base_slots,
                         WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops
