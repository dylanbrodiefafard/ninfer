#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Returns the transient arena capacity required by gated_delta_net for the given geometry. It is
 * zero when the private implementation requires no transient storage. The state/head dimension is
 * fixed at 128; `value_heads` must be at least `qk_heads` and divisible by it. The query covers
 * every T in the inclusive interval and throws for an invalid profile or interval.
 */
[[nodiscard]] std::size_t gated_delta_net_workspace_capacity_bytes(std::int32_t qk_heads,
                                                                   std::int32_t value_heads,
                                                                   bool normalize_qk,
                                                                   std::int32_t min_tokens,
                                                                   std::int32_t max_tokens);

/**
 * Applies the Gated DeltaNet recurrence independently for each value head h. Let
 * G=value_heads/qk_heads; its Q/K head is qh=floor(h/G). Starting from S_h, for t in increasing
 * order:
 *
 *   alpha       = exp(g[h,t])
 *   delta       = beta[h,t] * (v[:,h,t] - alpha * S_h * k[:,qh,t])
 *   S_h         = alpha * S_h + outer(delta, k[:,qh,t])
 *   ideal[:,h,t] = scale * S_h * q[:,qh,t].
 *
 * Shapes/dtypes are contiguous q/k BF16 [128,Hqk,T], v/out BF16 [128,Hv,T], g/beta FP32 [Hv,T],
 * and state FP32 [128,128,Hv], where Hqk>=1, Hv>=Hqk, and Hv%Hqk==0. `scale` is 1/sqrt(128). When
 * `normalize_qk` is true, the recurrent implementation consumes raw q/k and applies
 * x / sqrt(sum(x^2) + 1e-6) independently to every 128-element row before using it. When false,
 * q/k are consumed as supplied. The oracle evaluates the complete recurrence and `ideal` naively
 * in FP64 from the represented inputs and FP32 initial state. The BF16 out is promoted and
 * compared directly with that result; output storage rounding belongs to the Op's numerical
 * criterion, not the oracle. Recurrent implementations may apply the normalization directly;
 * chunked implementations may use private normalized staging. The corresponding private storage
 * is included by gated_delta_net_workspace_capacity_bytes when `normalize_qk` is true.
 * Inputs and out do not overlap state or one another. `ws` supplies transient storage reported by
 * gated_delta_net_workspace_capacity_bytes; scratch is scoped to the call. T may be any positive
 * value.
 *
 * This overload reads and writes the same `ssm_state`, publishing the state after all T tokens.
 */
void gated_delta_net(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                     const Tensor& beta, float scale, bool normalize_qk, WorkspaceArena& ws,
                     Tensor& ssm_state, Tensor& out, cudaStream_t stream);

/**
 * Distinct-state form of the same recurrence. `ssm_state_out` receives the final state;
 * `ssm_state_in` and `ssm_state_out` may be disjoint or exactly the same storage. No other
 * arguments may overlap either state.
 */
void gated_delta_net(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                     const Tensor& beta, float scale, bool normalize_qk, WorkspaceArena& ws,
                     const Tensor& ssm_state_in, Tensor& ssm_state_out, Tensor& out,
                     cudaStream_t stream);

/**
 * Snapshot form for B independent recurrences. q/k are contiguous BF16 [128,Hqk,W,B], v/out are
 * BF16 [128,Hv,W,B], g/beta are FP32 [Hv,W,B], and `ssm_states` is contiguous FP32
 * [128,128,Hv,Slots]. `initial_state_slots` and `snapshot_base_slots` are contiguous I32 [B].
 * `valid_columns` is either contiguous I32 [B], with every value in [1,W], or an empty Tensor
 * meaning every row has W valid columns. B=1 accepts every positive W; B=2..8 accepts W=1..16.
 *
 * Row b starts from initial_state_slots[b] and writes the state after valid column j to
 * snapshot_base_slots[b]+j. Invalid-tail output columns are exact BF16 zero and do not mutate
 * state. The caller reserves disjoint complete [base,base+W) intervals and prevents one row from
 * overwriting another row's initial slot; a row may overwrite its own initial slot after loading
 * it. This form uses no arena allocation and `ssm_states` is the only persistent state mutated.
 */
void gated_delta_net_snapshot(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                              const Tensor& beta, float scale, bool normalize_qk,
                              Tensor& ssm_states, const Tensor& valid_columns,
                              const Tensor& initial_state_slots, const Tensor& snapshot_base_slots,
                              Tensor& out, cudaStream_t stream);

/**
 * Op: gated_delta_net_replay_record
 *
 * Evaluates B independent normalized Gated DeltaNet recurrences from absolute state-pool slots
 * without modifying any state. q/k are BF16 [128,Hq,T,B], v/out are BF16 [128,Hv,T,B], g/beta
 * are FP32 [Hv,T,B], and ssm_states is FP32 [128,128,Hv,S]. The ReplaySSM execution domain is
 * B=1..6 and T=2..16, with Hq=16 and Hv in {32,48}. scale is 1/sqrt(128).
 *
 * valid_columns is empty for dense rows or device I32 [B], with every caller-supplied extent in
 * [1,T]. initial_state_slots is device I32 [B] containing absolute slots in [0,S). For each valid
 * transition, key_record BF16 [128,Hq,T,B], value_record BF16 [128,Hv,T,B], and gate_record FP32
 * [2,Hv,T,B] receive bit-preserving copies of raw k, v, and {g,beta}. The invalid record suffix is
 * unchanged, while the invalid out suffix is exact BF16 zero. Inputs, state, records, and out are
 * pairwise non-overlapping.
 *
 * `tree_schedule` is null for sequential packed time: S_j = F(S_{j-1}, x_j) from the checkpoint
 * slot. When non-null it is the I32 [kGdnTreeScheduleWords,B] schedule written by
 * gated_delta_net_tree_schedule() for this round's parent_index and valid_columns, and the
 * recurrence is the tree rule S_j = F(S_parent[j], x_j) with S_parent = checkpoint for column 0.
 * Every column applies exactly the transitions of a sequential chain over its root path, in path
 * order, with the width-one decode arithmetic, so each packed column's output is bit-identical
 * to sequential execution of that path. No per-column state is materialized.
 */
void gated_delta_net_replay_record(const Tensor& q, const Tensor& k, const Tensor& v,
                                   const Tensor& g, const Tensor& beta, float scale,
                                   const Tensor& ssm_states, const Tensor& valid_columns,
                                   const Tensor& initial_state_slots, Tensor& key_record,
                                   Tensor& value_record, Tensor& gate_record, Tensor& out,
                                   cudaStream_t stream, const Tensor* tree_schedule = nullptr);

inline constexpr std::int32_t kGdnTreeMaxColumns = 16;
// Column c emits one step and, when its parent is neither column c-1 nor slotted, first replays
// its parent's root path (at most c steps, since a parent's depth is below c). The schedule
// therefore never exceeds sum_{c<16} (1 + c) steps.
inline constexpr std::int32_t kGdnTreeMaxSteps = kGdnTreeMaxColumns * (kGdnTreeMaxColumns + 1) / 2;
inline constexpr std::int32_t kGdnTreeScheduleWords = 1 + kGdnTreeMaxSteps;

/**
 * Op: gated_delta_net_tree_schedule
 *
 * Compiles each row's packed draft tree into the step list gated_delta_net_replay_record
 * executes. parent_index is device I32 [T,B] (T in [2,16]) in topological order: column 0 has
 * parent -1 and every other valid column j has parent in [0,j). valid_columns is empty (all T
 * columns valid) or device I32 [B] with extents in [1,T]. schedule is device I32
 * [kGdnTreeScheduleWords,B]: word 0 is the step count, then one word per step,
 * `column | code << 8`, where code bits 0-2 select the resume source (0 the live state, 1 the
 * checkpoint, 2+s branch slot s), bits 3-5 the branch slot to save into (0 none, 1+s slot s), and
 * bit 6 whether the step emits the column's records and output. Columns with two or more children
 * save their state while a slot is free (three slots; a slot is released when its branch's last
 * child starts); a column whose parent is neither the previous column nor slotted is preceded by
 * non-emitting replay steps over its parent's root path. A depth-first column order makes every
 * sibling resume from one slot load. The schedule depends only on parent_index and
 * valid_columns, so one launch serves every GDN layer of a round.
 */
void gated_delta_net_tree_schedule(const Tensor& parent_index, const Tensor& valid_columns,
                                   Tensor& schedule, cudaStream_t stream);

} // namespace ninfer::ops
