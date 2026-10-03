#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/sampling.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

inline constexpr std::int32_t kDflash2PathSelectTopK                = 16;
inline constexpr std::int32_t kDflash2PathSelectRank                = 256;
inline constexpr std::int32_t kDflash2PathSelectHidden              = 5120;
inline constexpr std::int32_t kDflash2PathSelectCodebookRows        = 248320;
inline constexpr std::int32_t kDflash2PathSelectShortlistRows       = 131072;
inline constexpr std::int32_t kDflash2PathSelectMaxBatch            = 8;
inline constexpr std::int32_t kDflash2PathSelectMaxWidthWhenBatched = 16;
inline constexpr int kDflash2PathSelectRngPurpose                   = 16;
inline constexpr std::int32_t kDflash2TreeMaxWidth                  = 16;
inline constexpr std::int32_t kDflash2VerifyWidth                   = 12;

/**
 * Op: dflash2_path_select
 *
 * Math / indexing:
 *   For each column (t,b), let rows[0..15] be the unsorted top-16 logit rows of logits[:,t,b]:
 *   the 16 largest represented logits, with lower row index winning logit ties.
 *   candidates[c] = logit_token_ids[rows[c]] when logit_token_ids is non-null, else rows[c].
 *   unary[c] = logits[rows[c], t, b]. Project the hidden state
 *
 *     h[r,t,b] = sum_{k=0}^{5119} W_h[r,k] * hidden[k,t,b],   r in [0,256).
 *
 *   prev[-1,b] = anchors[b]. Anchors are vocabulary ids (not shortlist rows). For t = 0..T-1
 *   and each candidate c,
 *
 *     score[t,b,c] = unary[c]
 *       + sum_{r=0}^{255} (pred_code[r, prev[t-1,b]] * h[r,t,b]) * succ_code[r, candidates[c]].
 *
 *   configs is a device-resident SamplingConfig[B] (same buffer the round copies into ingress).
 *   The draft temperature is configs[b].draft_temperature when configs[b].p_less != 0, else
 *   configs[b].temperature (force_greedy makes it 0). The p-less temperature parameterizes the
 *   target distribution in sampling.h, not these draft scores: a 16-way softmax at the p-less
 *   target temperature is nearly uniform, so p-less drafts use their own draft_temperature. If
 * the draft temperature is <= 0, path[t,b] is the candidate with the greatest score; equal scores
 * select the lower token id. Otherwise the 16 scores are softmax-normalized after dividing by the
 *   draft temperature and one candidate is drawn by inverse-CDF using
 *
 *     u = splitmix64(configs[b].seed ^ seed_xor,
 *                    logical_positions[b] + position_offset + 1,
 *                    purpose=16, hop=t) in [0,1),
 *
 *   keyed by the round's first position and the hop so a later round never reuses a draft
 *   uniform.
 *
 *   Then prev[t,b] = path[t,b]. Candidate order does not affect a greedy pick; the sampled
 *   inverse-CDF walk runs in the unsorted top-16 order.
 *   An internal force_greedy call may override the draft temperature for an intermediate
 *   refinement pass. When selector_ids / selector_q are non-null they receive the 16 candidate
 *   token ids and the proposal distribution q: one-hot at a greedy pick, else the 16-way softmax
 *   the draft was drawn from. Chain accept uses this q for truncated sampling and p-less alike.
 *   Null selectors leave q implicit one-hot at path[t,b].
 *
 * Logical shapes:
 *   logits is contiguous BF16 [V,T] or [V,T,B] with V>=16. hidden is contiguous BF16 [5120,T] or
 *   [5120,T,B]. pred_code and succ_code are contiguous BF16 [256, codebook_rows] (rank fastest)
 *   when the NVFP4 codebook weights are null.
 *   When logit_token_ids is null, codebook_rows >= V (identity: logit row is the token id).
 *   When logit_token_ids is non-null it is contiguous I32 [V] mapping each logit row to a token
 *   id; codebook_rows >= 248320 and is not required to be >= V. Product uses codebook_rows=248320
 *   and V in {131072, 248320}. anchors and logical_positions are contiguous I32 [B], or scalars
 *   for B=1. logical_positions[b] is the current anchor position. path is contiguous I32 [T] or
 *   [T,B]. selector_ids is contiguous I32 [16,T] or [16,T,B] and
 *   selector_q is contiguous FP32 of the same shape; both null or both non-null. Every anchor
 *   and every selected token id is in [0, codebook_rows).
 *   T is any positive value at B=1; B=2..8 admits T=1..16.
 *
 * Supported domain:
 *   hidden_projection is BF16_CTRL Contiguous [256,5120], or a Linear-registered Q4G64_F16S /
 *   W8G32_F16S RowSplit / NVFP4 BlockScale problem of that logical shape. The Q4/W8/NVFP4 route
 *   calls Linear at per-sequence T (not T*B) so C>1 does not select a different T-specialized
 *   kernel than C=1.
 *
 *   pred_code and succ_code are contiguous BF16 [256, codebook_rows] (rank fastest), or NVFP4
 *   BlockScale weights of logical shape [codebook_rows, 256] passed as pred_nvfp4/succ_nvfp4.
 *
 * Numeric:
 *   Top-k and greedy path ids are exact functions of the represented BF16 logits/scores. The
 *   score formula is evaluated by the oracle in FP64 from represented inputs and the logical FP32
 *   dequantized W_h. Stochastic draws are a function of (seeds[b], logical position, purpose); the
 * Op does not promise a particular host RNG bitstream as a public numeric output.
 *
 * Effects:
 *   Writes all of path. When selector tensors are provided, writes all of them. Inputs are
 *   unchanged. path must not alias logits, hidden, codebooks, anchors, selector tensors, or
 *   any projection-weight plane.
 *
 * Workspace:
 *   Caller-owned transient storage sized by dflash2_path_select_workspace_capacity_bytes() holds
 *   the [256,T*B] hidden projection, per-column top-16 scratch, and any Linear child scratch.
 */
[[nodiscard]] std::size_t dflash2_path_select_workspace_capacity_bytes(QType qtype,
                                                                       std::int32_t min_tokens,
                                                                       std::int32_t max_tokens,
                                                                       std::int32_t batch = 1);

void dflash2_path_select(const Tensor& logits, const Tensor& hidden,
                         const Weight& hidden_projection, const Tensor& pred_code,
                         const Tensor& succ_code, const Tensor& anchors,
                         const Tensor& logical_positions, const SamplingConfig* configs,
                         Tensor& path, WorkspaceArena& workspace, cudaStream_t stream,
                         const Tensor* logit_token_ids = nullptr,
                         const Weight* pred_nvfp4 = nullptr, const Weight* succ_nvfp4 = nullptr,
                         Tensor* selector_ids = nullptr, Tensor* selector_q = nullptr,
                         unsigned long long seed_xor = 0, std::int32_t position_offset = 0,
                         bool force_greedy = false);

/**
 * Op: dflash2_tree_select
 *
 * Builds a packed parent-conditioned draft tree from the same DFlash2 shortlist scores as
 * dflash2_path_select, without sampling. Block size T is logits.ne[1] and W = verify_ids.ne[0]
 * in [2, kDflash2TreeMaxWidth]. A child's log-probability is the 16-way log-softmax of its
 * parent's Markov scores (the selector's unary + bilinear term over column t's top-16, with the
 * parent token as predecessor); a node's path log-probability is the sum along its root path.
 * Starting from the anchor (column 0), the selector repeatedly adds the frontier node of greatest
 * path log-probability until W nodes exist or no frontier remains (depth is bounded by T); ties
 * select the lower token id. Path log-probabilities never increase along a path, so the W nodes
 * form the prefix-closed tree of greatest summed path probability under the Markov model.
 * Columns are emitted in depth-first preorder with children in descending path log-probability,
 * so the most probable path is a contiguous spine (parent = previous column). Unused columns copy
 * the last live column and are excluded by valid_columns. configs (device SamplingConfig[B], or
 * null) selects the per-parent softmax temperature: p_less_tree_temperature for p-less rows,
 * 1 otherwise; it shapes only which nodes are drafted, since tree drafts are verified by
 * sample-then-descend.
 *
 * cache_positions[j,b] = frontiers[b] + j. rope_positions[j,b] = frontiers[b] + depth[j].
 * ancestor_mask[j,b] bit i is set iff packed column i is an ancestor of j, including j.
 * parent_index[0,b] = -1; every other valid column has parent in [0, j).
 */
void dflash2_tree_select(const Tensor& logits, const Tensor& hidden,
                         const Weight& hidden_projection, const Tensor& pred_code,
                         const Tensor& succ_code, const Tensor& anchors, const Tensor& frontiers,
                         Tensor& verify_ids, Tensor& parent_index, Tensor& cache_positions,
                         Tensor& rope_positions, Tensor& ancestor_mask, Tensor& valid_columns,
                         WorkspaceArena& workspace, cudaStream_t stream,
                         const Tensor* logit_token_ids = nullptr,
                         const Weight* pred_nvfp4 = nullptr, const Weight* succ_nvfp4 = nullptr,
                         const SamplingConfig* configs = nullptr,
                         float p_less_tree_temperature = 1.0f);

} // namespace ninfer::ops
