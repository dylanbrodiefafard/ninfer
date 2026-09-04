#pragma once

#include "core/tensor.h"
#include "ninfer/ops/sampling.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

// Caller-owned transient capacity for every draft-count and batch-size pair in the inclusive
// domains. token_domain is the fixed sampling profile; invalid domains throw.
[[nodiscard]] std::size_t speculative_accept_greedy_drafts_workspace_capacity_bytes(
    std::int32_t token_domain, std::int32_t min_drafts, std::int32_t max_drafts,
    std::int32_t min_batch, std::int32_t max_batch);

// Caller-owned transient capacity for every packed-tree width and batch-size pair in the
// inclusive domains. Width is the packed verify width W (not K+1). Invalid domains throw.
[[nodiscard]] std::size_t speculative_accept_tree_drafts_workspace_capacity_bytes(
    std::int32_t token_domain, std::int32_t min_width, std::int32_t max_width,
    std::int32_t min_batch, std::int32_t max_batch);

/**
 * Op: speculative_prepare_verify_inputs
 *
 * Math / indexing:
 *   For row b and 0<=j<=K:
 *     verify_ids[j,b] = anchors[b]                         when j=0
 *                       drafts[j-1,b]                     when 0<j<=Pcur[b]
 *                       anchors[b]                        otherwise;
 *     positions[j,b]  = base_positions[b] + min(j,Pcur[b]).
 *
 * Logical shapes:
 *   All tensors are contiguous I32. anchors/base_positions/current_extents are [B], drafts is
 *   [K,B] with K>=1 and B>=1, and verify_ids/positions are [K+1,B]. Each current extent is in
 *   [0,K]. Inputs and outputs do not overlap.
 *
 * Effects:
 *   Writes every physical output element, including safe invalid-tail values. Inputs remain
 *   unchanged.
 *
 * Workspace:
 *   None.
 */
void speculative_prepare_verify_inputs(const Tensor& anchors, const Tensor& drafts,
                                       const Tensor& base_positions, const Tensor& current_extents,
                                       Tensor& verify_ids, Tensor& positions, cudaStream_t stream);

/**
 * Prepare only the target verification ids when the caller already owns the matching position
 * matrix. Shapes and id semantics are identical to speculative_prepare_verify_inputs; the
 * existing positions remain untouched.
 */
void speculative_prepare_verify_ids(const Tensor& anchors, const Tensor& drafts,
                                    const Tensor& current_extents, Tensor& verify_ids,
                                    cudaStream_t stream);

/**
 * Op: speculative_accept_greedy_drafts
 *
 * Algorithm:
 *   Independently for each row b, greedy mode accepts the longest available draft prefix matching
 *   target_tokens and commits the target token at the first mismatch (or the bonus column).
 *   Sampling mode applies configs[b] to each valid verification column, accepts draft i with
 *   probability min(1, p_i(d)/q_i(d)) (Leviathan), samples from max(0, p-q) on first rejection,
 *   and samples a bonus from column Pcur[b] when every available draft is accepted. Null
 *   selector_ids/selector_q is the one-hot draft convention: accept iff u < p_i(d) and the
 *   residual excludes d. Greedy mode ignores q. When configs[b].p_less is set, p is the p-less
 *   distribution from sampling.h rather than the top-k/top-p/min-p truncation, and selector q is
 *   ignored (one-hot at the drafted token, same as MTP): p-less temperature is not a draft
 *   softmax, and a 16-way q at that T would make Leviathan accept almost every top-16 copy. Hop 0
 *   is that Leviathan test; every later hop, and the bonus after a full accept, is greedy
 *   (accept the draft iff it equals the packed-column argmax, else emit that argmax). A hop-0
 *   p-less residual whose mass is numerically zero draws from p' rather than re-emitting the
 *   rejected draft. If admitted mass is zero, or a residual inverse-CDF with positive mass does
 *   not land on a survivor, the correction is the packed-column argmax.
 *   RNG domains are the speculative accept/correction/bonus SamplePurpose values and logical
 *   positions derived from the old length.
 *
 * Logical shapes:
 *   All Tensor storage is contiguous. target_tokens/licensed_tokens are I32 [K+1,B], drafts is
 *   I32 [K,B], logits is BF16 [physical_rows,K+1,B], and current_extents/lengths/anchors/
 *   licensed_counts/accepted are I32 [B]. Optional selector_ids is I32 [C,K,B] and selector_q
 *   is FP32 [C,K,B] with C>=1 (product C=16); both null or both non-null. token_domain is in
 *   [1,physical_rows], K>=1, B>=1, and configs points to a device-resident SamplingConfig[B].
 *   Tensor arguments, configs, and configs[b].token_counts do not overlap except for the
 *   explicitly mutated objects.
 *
 * Numeric:
 *   Sampling filtering, penalties, normalization, and RNG semantics are those of sampling.h.
 *
 * Effects:
 *   For each row, let A be the accepted draft count and L=A+1. licensed_tokens[0:A,b] receives
 *   accepted drafts, licensed_tokens[A,b] receives the correction/bonus token, and the remaining
 *   physical slots are zero. licensed_counts[b]=L; accepted[b]=A; anchors[b] becomes the
 *   correction/bonus token; lengths[b]+=L. In sampling mode, each produced token increments
 *   configs[b].token_counts when that pointer is non-null. Greedy mode does not update
 *   token_counts. current_extents and all other inputs remain unchanged. Request statistics are
 *   deliberately outside this Op.
 *
 * Workspace:
 *   Caller-owned transient storage reported by
 *   speculative_accept_greedy_drafts_workspace_capacity_bytes().
 */
void speculative_accept_greedy_drafts(const Tensor& target_tokens, const Tensor& logits,
                                      const Tensor& drafts, const Tensor& current_extents,
                                      Tensor& lengths, Tensor& anchors, Tensor& licensed_tokens,
                                      Tensor& licensed_counts, Tensor& accepted,
                                      std::int32_t token_domain, const SamplingConfig* configs,
                                      WorkspaceArena& workspace, cudaStream_t stream,
                                      const Tensor* selector_ids = nullptr,
                                      const Tensor* selector_q = nullptr);

/**
 * Op: speculative_accept_tree_drafts
 *
 * Greedy: from packed column 0, walk to the unique child whose token equals the target argmax;
 * otherwise emit that argmax as the correction. Sampling with p_less==0: at node u sample x from
 * the truncated target distribution; if x is a child of u, accept and continue, else emit x as
 * correction (SpecInfer membership). When configs[b].p_less is set, hop 0 is that membership
 * draw from p-less(p_LLM); every later hop walks only the packed-column argmax (greedy
 * correction if that argmax is not a child). Walks at most current_extents[b] accepted hops
 * (same budget as chain verify). fold_path lists packed columns of the processed path including
 * the root; accepted_column is the last processed packed index (hidden selector).
 * licensed_tokens are time-ordered accepted child ids plus the correction. accepted is the
 * accepted draft count. Sampling increments configs[b].token_counts for each produced token when
 * that pointer is non-null; greedy does not. Large-vocabulary sampling uses the same
 * partial/group pipeline as chain accept, with a parent-walk penalty overlay at each packed
 * column, then the hop-0 membership / later-hop greedy walk over the stored p-less moments.
 *
 * verify_ids/parent_index/fold_path/licensed_tokens are I32 [W,B]. target_tokens is I32 [W,B].
 * logits is BF16 [physical_rows,W,B]. current_extents/valid_columns and the other vectors are
 * I32 [B]. W is the packed verify width in [2,16] (product k=7 uses 12). lengths[b] is the
 * pre-round sequence length and is incremented by the produced count; it must not alias the
 * packed-window base used by gqa_kv_compact_path (E+path[i] → E+i).
 *
 * Workspace:
 *   Caller-owned transient storage reported by
 *   speculative_accept_tree_drafts_workspace_capacity_bytes().
 */
void speculative_accept_tree_drafts(const Tensor& target_tokens, const Tensor& logits,
                                    const Tensor& verify_ids, const Tensor& parent_index,
                                    const Tensor& valid_columns, const Tensor& current_extents,
                                    Tensor& lengths, Tensor& anchors, Tensor& licensed_tokens,
                                    Tensor& licensed_counts, Tensor& accepted,
                                    Tensor& accepted_column, Tensor& fold_path,
                                    std::int32_t token_domain, const SamplingConfig* configs,
                                    WorkspaceArena& workspace, cudaStream_t stream);

/**
 * Op: speculative_select_accepted_hidden
 *
 * Math / indexing:
 *   out[:,b] = hidden[:,selectors[b],b].
 *
 * Shape / numeric / effects:
 *   hidden is contiguous BF16 [D,T,B], selectors is contiguous I32 [B] with every value in [0,T),
 *   and out is distinct contiguous BF16 [D,B]. The Op exactly copies BF16 bits, writes all of out,
 *   and uses no workspace or other state.
 */
void speculative_select_accepted_hidden(const Tensor& hidden, const Tensor& selectors, Tensor& out,
                                        cudaStream_t stream);

/**
 * Op: proposal_remap_token_ids
 *
 * Math / indexing:
 *   proposal_tokens[i]' = id_map[proposal_tokens[i]] for every proposal token.
 *
 * Effects:
 *   Updates the contiguous non-empty I32 proposal_tokens vector in place; every input id is in
 *   [0,count), and id_map is a distinct device I32 array [count]. There is no workspace or other
 *   state side effect.
 */
void proposal_remap_token_ids(Tensor& proposal_tokens, const std::int32_t* id_map,
                              std::int32_t count, cudaStream_t stream);

} // namespace ninfer::ops
