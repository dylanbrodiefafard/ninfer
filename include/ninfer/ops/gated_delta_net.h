#pragma once

#include "core/arena.h"
#include "core/cache_warm.h"
#include "core/tensor.h"

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

struct GdnRecordFold;

// Caller-owned transient capacity for an ordinary recurrence at this width. Zero means
// the route uses no workspace; the caller may pass its existing nonempty arena unchanged.
[[nodiscard]] std::size_t gated_delta_net_workspace_capacity_bytes(std::int32_t width,
                                                                   bool normalize_qk);

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
 * Shapes/dtypes are contiguous q/k BF16 [128,16,T], v/out BF16 [128,48,T], g/beta FP32 [48,T],
 * and state FP32 [128,128,48]. `scale` is 1/sqrt(128). When
 * `normalize_qk` is true, the recurrent implementation consumes raw q/k and applies
 * x / sqrt(sum(x^2) + 1e-6) independently to every 128-element row before using it. When false,
 * q/k are consumed as supplied. The oracle evaluates the complete recurrence and `ideal` naively
 * in FP64 from the represented inputs and FP32 initial state. The BF16 out is promoted and
 * compared directly with that result; output storage rounding belongs to the Op's numerical
 * criterion, not the oracle. Recurrent implementations may apply the normalization directly;
 * private arithmetic is implementation-defined. Inputs and out do not overlap state or one
 * another. Normalized ordinary prefill T>=256 uses caller-owned transient workspace.
 * The ordinary form accepts T=1..262144; snapshot and
 * replay forms retain their fixed W=1..16 domain.
 *
 * This overload reads and writes the same `ssm_state`, publishing the state after all T tokens.
 */
void gated_delta_net(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                     const Tensor& beta, float scale, bool normalize_qk, Tensor& ssm_state,
                     Tensor& out, WorkspaceArena& workspace, hipStream_t stream);

/**
 * Distinct-state form of the same recurrence. `ssm_state_out` receives the final state;
 * `ssm_state_in` and `ssm_state_out` may be disjoint or exactly the same storage. No other
 * arguments may overlap either state.
 */
void gated_delta_net(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                     const Tensor& beta, float scale, bool normalize_qk, const Tensor& ssm_state_in,
                     Tensor& ssm_state_out, Tensor& out, WorkspaceArena& workspace,
                     hipStream_t stream);

/**
 * Diagnostic ordinary recurrence form. In addition to the normal final state and output, writes
 * the complete FP32 state after `prefix_tokens` transitions to `prefix_state`. The trace state is
 * disjoint caller-owned storage with shape [128,128,48]. The deliberately narrow diagnostic
 * contract is T=129 and prefix_tokens=128: it compares the wide-prefill prefix with the exact
 * restored append frontier without adding a second production recurrence route.
 */
void gated_delta_net_trace_prefix_state(const Tensor& q, const Tensor& k, const Tensor& v,
                                        const Tensor& g, const Tensor& beta, float scale,
                                        bool normalize_qk, Tensor& ssm_state, Tensor& out,
                                        Tensor& prefix_state, std::int32_t prefix_tokens,
                                        hipStream_t stream);

/**
 * Snapshot form for B independent recurrences. q/k are contiguous BF16 [128,Hqk,W,B], v/out are
 * BF16 [128,Hv,W,B], g/beta are FP32 [Hv,W,B], and `ssm_states` is contiguous FP32
 * [128,128,Hv,Slots]. `initial_state_slots` and `snapshot_base_slots` are contiguous I32 [B].
 * `valid_columns` is either contiguous I32 [B], with every value in [1,W], or an empty Tensor
 * meaning every row has W valid columns. B=1..16 and W=1..16.
 *
 * Row b starts from initial_state_slots[b] and writes the state after valid column j to
 * snapshot_base_slots[b]+j. Invalid-tail output columns are exact BF16 zero and do not mutate
 * state. The caller reserves disjoint complete [base,base+W) intervals and prevents one row from
 * overwriting another row's initial slot; a row may overwrite its own initial slot after loading
 * it. This form uses no arena allocation and `ssm_states` is the only persistent state mutated.
 * `warm` (also on the replay form) is touched by CTAs past the recurrence grid
 * (core/cache_warm.h).
 */
void gated_delta_net_snapshot(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                              const Tensor& beta, float scale, bool normalize_qk,
                              Tensor& ssm_states, const Tensor& valid_columns,
                              const Tensor& initial_state_slots, const Tensor& snapshot_base_slots,
                              Tensor& out, hipStream_t stream, const CacheWarm& warm = {});

/**
 * Op: gated_delta_net_replay_record
 *
 * Evaluates B independent normalized Gated DeltaNet recurrences from absolute state-pool slots
 * without publishing the current unaccepted state. q/k are BF16 [128,Hq,T,B], v/out are BF16
 * [128,Hv,T,B], g/beta are FP32 [Hv,T,B], and ssm_states is FP32 [128,128,Hv,S]. The ReplaySSM
 * execution domain is B=1..16 and T=2..16, with Hq=16 and Hv=48. scale is 1/sqrt(128).
 *
 * valid_columns is empty for dense rows or device I32 [B], with every caller-supplied extent in
 * [1,T]. initial_state_slots is device I32 [B] containing absolute slots in [0,S). For each valid
 * transition, key_record BF16 [128,Hq,T,B], value_record BF16 [128,Hv,T,B], and gate_record FP32
 * [2,Hv,T,B] receive bit-preserving copies of raw k, v, and {g,beta}. The invalid record suffix is
 * unchanged, while the invalid out suffix is exact BF16 zero. Inputs, state, records, and out are
 * pairwise non-overlapping.
 *
 * `parent_index` is null or empty for sequential packed time: S_j = F(S_{j-1}, x_j) from the
 * checkpoint slot. When non-null it is contiguous I32 [T,B], or [T] when B=1, and the recurrence
 * is the tree rule S_j = F(S_parent[j], x_j). parent_index[0,b] is -1 and names the checkpoint
 * slot; every other valid column j has parent in [0,j). Packed order is not time. With a
 * workspace sized by gated_delta_net_replay_record_workspace_capacity_bytes(), parent states live
 * in HBM. Tree mode requires that caller-owned workspace; sequential record allocates no arena
 * workspace. With a non-null GdnRecordFold (sequential mode only), first replay its previous
 * committed raw records into each selected FP32 state slot, publish that committed state, then
 * compute the current round from it. The current round still does not publish unaccepted state.
 * The staged records must be disjoint from all current inputs, state, records and output, and
 * their slot selectors agree with initial_state_slots. Their normalization and FP32 transition
 * are the same as gdn_replay_fold; convolution history was published by prepare_layer already.
 */
[[nodiscard]] std::size_t
gated_delta_net_replay_record_workspace_capacity_bytes(std::int32_t value_heads, std::int32_t batch,
                                                       std::int32_t width);

void gated_delta_net_replay_record(const Tensor& q, const Tensor& k, const Tensor& v,
                                   const Tensor& g, const Tensor& beta, float scale,
                                   const Tensor& ssm_states, const Tensor& valid_columns,
                                   const Tensor& initial_state_slots, Tensor& key_record,
                                   Tensor& value_record, Tensor& gate_record, Tensor& out,
                                   hipStream_t stream, const Tensor* parent_index = nullptr,
                                   WorkspaceArena* workspace = nullptr, const CacheWarm& warm = {},
                                   const GdnRecordFold* fold = nullptr);

} // namespace ninfer::ops
