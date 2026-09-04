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
 * startup-fixed in [1,4096]. K/V use NVFP4-G16: codes are U8 [128,capacity,2], with the low
 * nibble representing the even feature and the high nibble the odd feature; scales are
 * FP8_E4M3FN [16,capacity,2]. Raw index keys are BF16 [128,capacity], and positions are I32
 * [3,capacity] in temporal/height/width order. Code planes are four-byte aligned for their packed
 * stores, raw index keys are two-byte aligned, and positions are four-byte aligned; all six planes
 * are pairwise disjoint.
 */
struct QsaStateView {
    Tensor k_codes;
    Tensor v_codes;
    Tensor k_scales;
    Tensor v_scales;
    Tensor raw_index_keys;
    Tensor positions;
};

/**
 * Encode normalized/rotated K and projected V from BF16 [256,2,W] into exact NVFP4-G16 and
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
 * RMSNorm with converted GGUF gamma, and 64-wide interleaved MRoPE. GGUF has already folded the
 * source zero-centered unit offset. Scores sum four ReLU dots / sqrt(128).
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
 * Compute one-token selected grouped-query attention from normalized/rotated BF16 q [256,24,1]
 * and the exact selected prefix. Query head h consumes KV head floor(h/12). Every K/V value is
 * decoded from the NVFP4-G16 state, including values appended earlier on the same stream. Softmax
 * uses FP32 max/subtract/exp/sum and the fixed 1/sqrt(256) scale. out is BF16 [256,24,1].
 * Empty/invalid queries are exact zero. selected_ids is I32 [S] for a caller-known bound S in
 * [1,2051]; the caller promises valid selected ids are unique, visible, and in range and
 * selected_count is in [0,S]. Workspace is caller-owned, 256-byte aligned U8 with at least
 * qsa_selected_attention_workspace_bytes() bytes. No state is mutated.
 */
[[nodiscard]] std::size_t qsa_selected_attention_workspace_bytes();
void qsa_selected_attention(const Tensor& q, const Tensor& selected_ids,
                            const Tensor& selected_count, const QsaStateView& state, Tensor& out,
                            Tensor& workspace, cudaStream_t stream);

struct QsaVerifierWeights {
    Weight index_query; // contiguous BF16_CTRL [512,2560]
    Weight index_key;   // contiguous BF16_CTRL [128,2560]
    Weight core_query_gate; // GGML Q5_K [12288,2560], per-head query then gate
    Weight core_key;        // GGML Q5_K [512,2560]
    Weight core_value;      // GGML Q5_K [512,2560]
    Weight output;          // GGML Q5_K [2560,6144]
    Tensor index_query_norm; // converted GGUF FP32 gamma [128]
    Tensor index_key_norm;   // converted GGUF FP32 gamma [128]
    Tensor core_query_norm;  // converted GGUF FP32 gamma [256]
    Tensor core_key_norm;    // converted GGUF FP32 gamma [256]
};

/** Fixed transient device capacity for qsa_verifier_token. */
[[nodiscard]] std::size_t qsa_verifier_workspace_bytes();

/**
 * Actual-artifact C=1/T=1 QSA verifier composite. x/out are BF16 [2560], token_id is I32 [1],
 * position is I32 [3], visibility is one CSR slice (visible_offsets I32 [2]). The exact weights
 * and controls are QsaVerifierWeights. selected_ids/count expose the exact selector result as I32
 * [2051] and [1].
 *
 * The Op projects BF16 index queries/key, projects the per-head core query/gate parent plus K/V,
 * applies Q/K norms with converted GGUF gamma and 64-wide interleaved T/H/W MRoPE with theta 1e7,
 * appends
 * normalized/rotated K, projected V, raw index key, and position at token_id, selects visible-rank
 * blocks, and evaluates selected attention through the NVFP4-G16 state. Each 256-wide attention
 * head is multiplied by sigmoid of its represented raw gate, concatenated, and projected by the
 * Q5_K output weight. Newly appended values are always consumed through the cache codec.
 *
 * This verifier entry owns no frontier, visibility construction, commit, or rollback. All storage
 * is caller-owned, non-overlapping except for no permitted aliases, and remains alive through the
 * stream. Workspace is contiguous U8, 256-byte aligned, and at least
 * qsa_verifier_workspace_bytes().
 */
void qsa_verifier_token(const Tensor& x, const Tensor& token_id, const Tensor& position,
                        const Tensor& visible_ids, const Tensor& visible_offsets,
                        const QsaVerifierWeights& weights, QsaStateView state,
                        Tensor& selected_ids, Tensor& selected_count, Tensor& out,
                        Tensor& workspace, cudaStream_t stream);

} // namespace ninfer::ops
