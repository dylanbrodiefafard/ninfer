#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

inline constexpr std::int32_t kQsaHeadDim          = 256;
inline constexpr std::int32_t kQsaQueryHeads       = 24;
inline constexpr std::int32_t kQsaKvHeads          = 2;
inline constexpr std::int32_t kQsaIndexHeadDim     = 128;
inline constexpr std::int32_t kQsaIndexQueryHeads  = 4;
inline constexpr std::int32_t kQsaMaximumTokens    = 4096;
inline constexpr std::int32_t kQsaSelectedCapacity = 2051;

/**
 * Qwen4 C=1 QSA persistent planes. All tensors are contiguous and capacity is
 * startup-fixed in [1,4096]. The explicit BF16 baseline uses K/V BF16 [256,capacity,2]
 * and null scale tensors. The diagnostic NVFP4-G16 profile uses U8 [128,capacity,2], with the low
 * nibble representing the even feature and the high nibble the odd feature; scales are
 * FP8_E4M3FN [16,capacity,2]. Raw index keys are BF16 [128,capacity], and positions are I32
 * [3,capacity] in temporal/height/width order. Code planes are four-byte aligned for their packed
 * stores; BF16 K/V also require four-byte alignment. Raw index keys are two-byte aligned and
 * positions are four-byte aligned; all present planes are pairwise disjoint. Format selection is
 * explicit and never inferred from a checkpoint or from nullable scales.
 */
enum class QsaKvFormat { BF16, NVFP4G16 };

struct QsaStateView {
    QsaKvFormat format = QsaKvFormat::BF16;
    Tensor k;
    Tensor v;
    Tensor k_scales;
    Tensor v_scales;
    Tensor raw_index_keys;
    Tensor positions;
};

/** Native growing state. Every plane uses the same page-group IDs and page size 64.
 * K/V are BF16 [256,64,2,P] or NVFP4 U8 [128,64,2,P], with FP8_E4M3FN
 * scales [16,64,2,P]. Raw keys are BF16 [128,64,1,P], positions I32 [3,64,1,P].
 * block_tables is I32 [L,C], C=1..4. All planes are contiguous page-major, disjoint,
 * and caller-owned. Logical capacity is 64*L, at most the source context 262144.
 * Paging changes no codec or selection mathematics. This view owns no frontier.
 */
struct QsaPagedStateView {
    QsaKvFormat format = QsaKvFormat::BF16;
    Tensor k, v, k_scales, v_scales, raw_index_keys, positions, block_tables;
};

/** Device controls for exact compact B rows, not padded to C. table_rows [B] selects
 * distinct writable table rows; valid_columns [B] is in [0,W]; frontiers [B] is the
 * length before append. positions is I32 [3,W,B] (MRoPE coordinates, not KV ordinals).
 * Query j writes logical frontiers[b]+j and sees precisely [0,frontiers[b]+j].
 * Only j<valid_columns[b] is live. Mappings for all live reads/writes are materialized
 * and stable until the stream drains. No host readback occurs, including graph replay.
 */
struct QsaBatchControls {
    Tensor table_rows, valid_columns, frontiers, positions;
};

/** Native overloads: B=1 permits W=1..4096 prefill; B=2..4 permits W=1..16.
 * max_visible_keys is a host execution envelope in [1,64*L], covering every live
 * frontier+valid count. It controls launch extent, never logical visibility.
 * All storage is non-overlapping and remains alive through stream completion.
 * K/V [256,2,W,B], raw keys [128,W,B]. Invalid suffixes never write state.
 */
void qsa_state_append(const Tensor& k, const Tensor& v, const Tensor& raw_index_keys,
                      const QsaBatchControls& controls, QsaPagedStateView state,
                      std::int32_t max_visible_keys, cudaStream_t stream);

/** Raw query [128,4,W,B], selected I32 [2051,W,B], count I32 [W,B]. The complete
 * visible-rank blocks are [4*r,4*r+4); source pooling/norm/MRoPE and stable top512
 * are identical to the diagnostic CSR formula. Append the incomplete causal tail.
 * Invalid suffixes are count zero / selected -1. Scratch is independent of context
 * capacity: a fixed 512-block streaming merge; no capacity-sized shared sort.
 */
[[nodiscard]] std::size_t qsa_index_select_workspace_bytes(std::int32_t width,
                                                         std::int32_t batch);
void qsa_index_select(const Tensor& raw_query, const QsaPagedStateView& state,
                      const QsaBatchControls& controls, std::int32_t max_visible_keys,
                      const Tensor& query_norm_weight, const Tensor& key_norm_weight,
                      Tensor& selected_ids, Tensor& selected_count, Tensor& workspace,
                      cudaStream_t stream);

/** Q/out BF16 [256,24,W,B]. Selected [S,W,B], S<=2051, count [W,B]. The caller
 * promises unique visible logical IDs in each prefix, including an explicitly frozen
 * subset when requested. Invalid suffixes/empty selections produce exact zero.
 * No workspace or cache mutation. Reads the represented BF16/NVFP4 state directly.
 */
void qsa_selected_attention(const Tensor& q, const Tensor& selected_ids,
                            const Tensor& selected_count, const QsaPagedStateView& state,
                            const QsaBatchControls& controls, std::int32_t max_visible_keys,
                            Tensor& out, cudaStream_t stream);

/**
 * Append normalized/rotated K and projected V from BF16 [256,2,W], copying every bit unchanged
 * for BF16 state or encoding the exact registered NVFP4-G16 codec for diagnostic state, and
 * append BF16 raw index keys [128,W] plus I32 MRoPE positions [3,W] at I32 append_ids [W].
 * An id of -1 is an invalid suffix and writes nothing. Every other id is promised by the caller
 * to be unique and in the state's capacity. K and V are 16-byte aligned for their vectorized K16
 * loads; raw index keys are two-byte aligned and position/append ids are four-byte aligned. This Op
 * owns no frontier or commit decision. All input and state storage is pairwise non-overlapping.
 */
void qsa_state_append(const Tensor& k, const Tensor& v, const Tensor& raw_index_keys,
                      const Tensor& position_ids, const Tensor& append_ids, QsaStateView state,
                      cudaStream_t stream);

/** Workspace for qsa_index_select at BF16 raw-query shape [128,4,W]. */
[[nodiscard]] std::size_t qsa_index_select_workspace_bytes(std::int32_t width);

/**
 * Select visible-rank blocks from BF16 raw_query [128,4,W]. query_ids is I32 [W]. Visibility is
 * CSR: offsets I32 [W+1] and flat strictly-increasing visible_ids. Each valid query id is in its
 * own slice. Complete rank blocks of four are scored after the semantic FP32 mean -> BF16 cast,
 * RMSNorm with effective gamma, and 64-wide interleaved MRoPE. Effective gamma represents
 * the source zero-centered unit offset already added. Scores sum four ReLU dots / sqrt(128).
 * The highest 512 blocks win; ties choose the lower logical block rank. Their ids, in ranked
 * block order, are followed by the incomplete tail. selected_ids I32 [2051,W] is padded with -1;
 * selected_count I32 [W] gives the valid prefix. query_id -1 produces an empty column.
 *
 * query_norm_weight/key_norm_weight are FP32 [128]. State capacity is at most 4096. Workspace is
 * contiguous, at least four-byte-aligned U8 with at least
 * qsa_index_select_workspace_bytes(W) bytes. The trusted caller owns CSR/id validation; the kernel
 * guards every state access but reports malformed device data as an empty output rather than
 * synchronizing the stream to throw. All input, state, output, and workspace storage is pairwise
 * non-overlapping.
 */
void qsa_index_select(const Tensor& raw_query, const QsaStateView& state,
                      const Tensor& query_ids, const Tensor& visible_ids,
                      const Tensor& visible_offsets, const Tensor& query_norm_weight,
                      const Tensor& key_norm_weight, Tensor& selected_ids,
                      Tensor& selected_count, Tensor& workspace, cudaStream_t stream);

/**
 * Compute selected grouped-query attention from normalized/rotated BF16 q [256,24,W] and the
 * exact per-column selected prefixes. Query head h consumes KV head floor(h/12). Every K/V value
 * is read exactly from BF16 state or decoded from diagnostic NVFP4-G16 state, including values
 * appended earlier on the same stream.
 * Softmax uses FP32 max/subtract/exp/sum and the fixed 1/sqrt(256) scale. out is BF16 [256,24,W].
 * Empty/invalid queries are exact zero. selected_ids is I32 [S,W] for a caller-known bound S in
 * [1,2051]; the caller promises each valid selected prefix is unique, visible, and in range and
 * selected_count is I32 [W] with values in [0,S]. Workspace is caller-owned, 256-byte aligned U8
 * with at least
 * qsa_selected_attention_workspace_bytes() bytes. No state is mutated.
 */
[[nodiscard]] std::size_t qsa_selected_attention_workspace_bytes();
void qsa_selected_attention(const Tensor& q, const Tensor& selected_ids,
                            const Tensor& selected_count, const QsaStateView& state, Tensor& out,
                            Tensor& workspace, cudaStream_t stream);

struct QsaVerifierWeights {
    Weight index_query; // contiguous BF16_CTRL [512,2560]
    Weight index_key;   // contiguous BF16_CTRL [128,2560]
    Weight core_query_gate; // BF16/Q5_K/NVFP4/row- or tensor-scaled FP8 [12288,2560], query then gate
    Weight core_key;        // BF16/Q5_K/NVFP4/row- or tensor-scaled FP8 [512,2560]
    Weight core_value;      // BF16/Q5_K/NVFP4/row- or tensor-scaled FP8 [512,2560]
    Weight output;          // BF16/Q5_K/NVFP4/row- or tensor-scaled FP8 [2560,6144]
    Tensor index_query_norm; // effective FP32 gamma [128]
    Tensor index_key_norm;   // effective FP32 gamma [128]
    Tensor core_query_norm;  // effective FP32 gamma [256]
    Tensor core_key_norm;    // effective FP32 gamma [256]
};

/** Transient capacity at width W in [1,4096] and explicit core projection formats.
 * Each core projection admits BF16, diagnostic Q5_K, NVFP4 or row-scaled FP8 A16,
 * plus tensor-calibrated FP8 A16. QSA A8 projection profiles are not admitted.
 * Index projections remain BF16; state format is explicitly selected. */
[[nodiscard]] std::size_t qsa_verifier_workspace_bytes(
    std::int32_t width, QType query_gate, QType key, QType value, QType output);

/**
 * Actual-artifact C=1 QSA verifier composite for W in [1,4096]. x/out are BF16 [2560,W],
 * append_ids is I32 [W], position is I32 [3,W], and visibility is W CSR slices
 * (visible_offsets I32 [W+1]). The exact weights and controls are QsaVerifierWeights.
 * selected_ids/count expose the exact selector result as I32 [2051,W] and [W].
 *
 * The Op projects BF16 index queries/key, projects the per-head core query/gate parent plus K/V,
 * applies Q/K norms with effective gamma and 64-wide interleaved T/H/W MRoPE with theta 1e7,
 * appends
 * normalized/rotated K, projected V, raw index key, and position at token_id, selects visible-rank
 * blocks, and evaluates selected attention through the explicit BF16 or NVFP4-G16 state. Each 256-wide attention
 * head is multiplied by sigmoid of its represented raw gate, concatenated, and projected by the
 * output weight. Newly appended values are always consumed through their represented cache format.
 * Native BF16 and tensor-calibrated FP8 core K retain FP32 private output through norm/RoPE to avoid a
 * premature BF16 projection cast before normalization. The BF16 append boundary remains explicit;
 * other projection formats retain their qualified profile.
 *
 * This verifier entry owns no frontier, visibility construction, commit, or rollback. All storage
 * is caller-owned, non-overlapping except for no permitted aliases, and remains alive through the
 * stream. Workspace is contiguous U8, 256-byte aligned, and at least
 * qsa_verifier_workspace_bytes(W, query_type, key_type, value_type, output_type) for the
 * supplied core projection formats (including the native-BF16 key precision scratch).
 */
void qsa_verifier(const Tensor& x, const Tensor& append_ids, const Tensor& position,
                  const Tensor& visible_ids, const Tensor& visible_offsets,
                  const QsaVerifierWeights& weights, QsaStateView state,
                  Tensor& selected_ids, Tensor& selected_count, Tensor& out,
                  Tensor& workspace, cudaStream_t stream);

/** Same projection, norm/RoPE, cache append, selected attention and output formula as
 * qsa_verifier, but consumes caller-specified I32 selected_ids [2051,W] / count [W]
 * without recomputing or mutating them. The caller promises unique visible IDs in each
 * prefix and valid counts; those IDs may deliberately exclude newly appended rows.
 * Raw index keys and core K/V are still appended at the supplied logical append IDs.
 * This is the explicit-input primitive for the audited frozen-domain MTP profile, not
 * a claim that all upstream MTP implementations use identical tail semantics.
 * Workspace/alias/lifetime requirements match qsa_verifier; no visibility CSR exists.
 */
void qsa_verifier_selected(const Tensor& x, const Tensor& append_ids, const Tensor& position,
                           const QsaVerifierWeights& weights, QsaStateView state,
                           const Tensor& selected_ids, const Tensor& selected_count, Tensor& out,
                           Tensor& workspace, cudaStream_t stream);

/** Native paged composite with the same projection/norm/gate formula and public
 * cast boundaries as above. x/out [2560,W,B]; controls, selection and storage obey
 * the native overloads. Workspace is contiguous U8, 256-byte aligned. Invalid output
 * suffixes are zero; no frontier is advanced. Frozen selection is never mutated.
 */
[[nodiscard]] std::size_t qsa_verifier_workspace_bytes(
    std::int32_t width, std::int32_t batch, QType query_gate, QType key,
    QType value, QType output);
void qsa_verifier(const Tensor& x, const QsaBatchControls& controls,
                  std::int32_t max_visible_keys, const QsaVerifierWeights& weights,
                  QsaPagedStateView state, Tensor& selected_ids, Tensor& selected_count,
                  Tensor& out, Tensor& workspace, cudaStream_t stream);
void qsa_verifier_selected(const Tensor& x, const QsaBatchControls& controls,
                           std::int32_t max_visible_keys, const QsaVerifierWeights& weights,
                           QsaPagedStateView state, const Tensor& selected_ids,
                           const Tensor& selected_count, Tensor& out, Tensor& workspace,
                           cudaStream_t stream);

} // namespace ninfer::ops
