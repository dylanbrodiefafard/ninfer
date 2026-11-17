#pragma once

// ninfer::ops - fused GDN Q/K/V/Z input projections.

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/a4_activation.h"
#include "ninfer/ops/a8_activation.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

// FP8 single-parent domain: FP8_E4M3FN_ROW_BF16S RowScale [16384,5120] and [16384,2560], ordered
// [Q(2048),K(2048),V(6144),Z(6144)], exact BF16 row multipliers. Projection admits
// A16Only/AllowA8 at every positive T for [16384,5120] and A16Only at every positive T for
// [16384,2560]. Snapshot and record admit B=1..12. For [16384,5120] the snapshot is registered
// for W=1 and the record for W=2..16 (a tree record for W>=3), both with A16 arithmetic under
// either policy; [16384,2560] admits only A16Only, with the snapshot over its whole B/W domain and
// the record over W=1..16 (chains) and W=2..16 (trees). Record preserves the optional tree-parent
// operand in fused and materialized routes. Complete FP64 projection/convolution/SiLU and
// represented persistent state form the oracle; private activation quantization and reduction
// choices do not enter that oracle.

/**
 * Op: gdn_input_proj
 *
 * Single-parent GDN projection. Registered parent forms are NVFP4 BlockScaleK16M128x4 and
 * FP8_E4M3FN_ROW_BF16S RowScale, each [16384,5120] and [16384,2560] with stored row counts
 * [2048,2048,6144,6144]. x is contiguous BF16 [K,T] for the parent's input rows K, qkv is
 * contiguous BF16 [10240,T], and z is contiguous BF16 [6144,T] for any positive T.
 *
 * The first three ranges are written contiguously to qkv and the final range is written to z.
 * NVFP4 [16384,5120] admits A16Only and AllowA4. Its A16 route is registered at T=1 and below the
 * W4A4 cutover (T < kA4GdnInputMinTokens); AllowA4 permits private activation quantization from
 * that cutover, so it admits every positive T, while A16Only at a wider T throws
 * std::invalid_argument. NVFP4 [16384,2560] admits only A16Only, at every positive T. FP8
 * [16384,5120] admits A16Only and AllowA8 at every positive T; AllowA8 permits private row-scaled
 * E4M3 activation quantization from T=8. FP8 [16384,2560] admits only A16Only, at every positive
 * T. Every route writes the two independent final allocations directly. The complete projection
 * is evaluated against the same exact-decode/naive-FP64 oracle; activation quantization and the
 * production reduction profile are private effects covered by the selected criterion. x, qkv, and
 * z must be pairwise non-overlapping.
 *
 * The policy-bearing form uses caller-owned call-scoped transient storage sized by
 * gdn_input_proj_workspace_capacity_bytes(). A16 requires zero bytes. The convenience overload
 * selects A16Only and requires no transient workspace.
 */
[[nodiscard]] std::size_t
gdn_input_proj_workspace_capacity_bytes(QType parent_qtype, std::int32_t parent_rows,
                                        std::int32_t input_rows, LinearPolicy policy,
                                        std::int32_t min_tokens, std::int32_t max_tokens);

void gdn_input_proj(const Tensor& x, const Weight& query_key_value_z_weight, Tensor& qkv, Tensor& z,
                    LinearPolicy policy, WorkspaceArena& workspace, cudaStream_t stream);

/**
 * Applies the A16-only single-parent GDN projection without transient workspace.
 */
void gdn_input_proj(const Tensor& x, const Weight& query_key_value_z_weight, Tensor& qkv, Tensor& z,
                    cudaStream_t stream);

/**
 * The NVFP4 AllowA4 single-parent projection of an already quantized activation
 * (ninfer/ops/a4_activation.h) of [5120,T], T >= kA4GdnInputMinTokens (the widths at which
 * AllowA4 selects W4A4), whose input scale divisor equals the weight's. For the same BF16
 * activation qkv and z are bit-identical to the policy form's with AllowA4. The weight is NVFP4
 * [16384,5120]; qkv/z are as above. The activation, qkv, and z must not overlap. The oracle is the
 * policy form's, evaluated on the represented activation. Invalid arguments throw
 * std::invalid_argument. No workspace.
 */
void gdn_input_proj(const A4Activation& x, const Weight& query_key_value_z_weight, Tensor& qkv,
                    Tensor& z, cudaStream_t stream);

/**
 * Returns the transient capacity for a snapshot profile. `batch_size` is exact and the query
 * covers every W in the inclusive width interval. The registered [16384,5120] W=1 snapshot runs
 * one fused T=1 GEMV per sequence and needs no transient plane; any other width throws. The
 * [16384,2560] parent (A16Only) reports zero when the whole interval uses the fused FP8
 * projection/convolution; otherwise it reports 20480 * batch_size bytes per column of the
 * largest staged width within the interval.
 */
[[nodiscard]] std::size_t gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
    QType parent_qtype, std::int32_t parent_rows, std::int32_t input_rows, LinearPolicy policy,
    std::int32_t batch_size, std::int32_t min_width, std::int32_t max_width);

/**
 * Op: gdn_input_proj_conv_snapshot
 *
 * Math / indexing:
 *   For each row b, let p[:,j,b] be the concatenated q/k/value projection of x[:,j,b], while
 *   z[:,j,b] is the independent z projection. Starting from the BF16 width-three history selected
 *   by initial_state_slots[b], evaluate the width-four depthwise convolution over p, apply SiLU,
 *   and publish its channel ranges to query, key, and value. After valid column j, write the new
 *   width-three history to snapshot_base_slots[b]+j. Z bypasses convolution.
 *
 * Logical shapes:
 *   x [K,W,B], one q/k/value/z parent [16384,K] with K in {5120,2560}, conv_weight [10240,4],
 *   conv_states [10240,3,Slots] with Slots >= W*B, query/key [2048,W,B], value/z [6144,W,B], and
 *   I32 selectors [B]. The registered [16384,5120] domain is ordinary decode: W=1 and B=1..12.
 *   The [16384,2560] domain is B=1 at every positive W and B=2..12 at W=1..16. `valid_columns`
 *   is empty for a dense invocation or I32 [B] for a mixed-width batch. A mixed-width invocation
 *   has B>=1 and every valid extent lies in [1,W].
 *
 * Numeric:
 *   The oracle exact-decodes packed weights and evaluates projection, convolution, SiLU, z, and
 *   every snapshot value naively in FP64 from represented inputs. After each valid column,
 *   the new history entry is rounded to BF16 before it participates in the next column's
 *   convolution: this is a persistent-state boundary, not private projection staging.
 *   BF16 query/key/value/z are promoted and compared with ideal FP64 outputs under the
 *   named A16 criterion; snapshots are compared with the represented BF16 history.
 *   Unfused projection tensors are not observable cast boundaries; production routes use their
 *   natural private accumulator and staging precision within the permitted compute profiles.
 *
 * Effects:
 *   Each row writes query/key/value through its valid prefix and exact zero to its invalid tail;
 *   z is projected for all B*W safe input columns. A row writes only its valid destination state
 *   prefix. The caller reserves disjoint complete [base,base+W) intervals, prevents one row from
 *   overwriting another row's initial slot, and may overlap a row's own initial slot with its
 *   destination after that initial history has been loaded. Other slots are unchanged. Newly
 *   projected convolution channels remain private until entering the BF16 history transition.
 *
 * Registered parents are NVFP4 BlockScaleK16M128x4 and FP8_E4M3FN_ROW_BF16S RowScale
 * [16384,5120] and [16384,2560], all in q/k/value/z row order. At K=5120, NVFP4 admits every
 * policy and FP8 A16Only and AllowA8; every admitted policy runs the A16 fused T=1 GEMV once per
 * request, so each row keeps its single-request arithmetic profile. At K=2560 both qtypes admit
 * only A16Only: one A16 projection covers all W*B columns (the parent is read once per call),
 * and a slot-indexed convolution consumes its BF16-rounded q/k/value projection. The FP8
 * single-request decode fuses convolution into the projection epilogue; other profiles stage it
 * in the workspace. The projection's reduction route depends on the packed column count W*B,
 * so a row matches a B=1 call on the same inputs within the A16 criterion rather than bit for
 * bit; the snapshot history is the same BF16 projection in either case. Other widths,
 * policies, and parents throw std::invalid_argument.
 */
void gdn_input_proj_conv_snapshot(const Tensor& x, const Weight& query_key_value_z_weight,
                                  const Tensor& conv_weight, Tensor& conv_states,
                                  const Tensor& valid_columns, const Tensor& initial_state_slots,
                                  const Tensor& snapshot_base_slots, Tensor& query, Tensor& key,
                                  Tensor& value, Tensor& z, LinearPolicy policy, WorkspaceArena& ws,
                                  cudaStream_t stream);

/**
 * Applies the A16-only single-parent form over the same per-parent B/W domain.
 */
void gdn_input_proj_conv_snapshot(const Tensor& x, const Weight& query_key_value_z_weight,
                                  const Tensor& conv_weight, Tensor& conv_states,
                                  const Tensor& valid_columns, const Tensor& initial_state_slots,
                                  const Tensor& snapshot_base_slots, Tensor& query, Tensor& key,
                                  Tensor& value, Tensor& z, WorkspaceArena& ws,
                                  cudaStream_t stream);

/**
 * Returns the transient capacity for a record-producing profile. NVFP4 [16384,5120] is
 * registered for AllowA8 at W=2..16: it fuses projection/convolution when W*B<=48 and needs only
 * activation codes/scales; larger packed extents also stage the FP32 projection before
 * convolution. FP8 [16384,5120] and both [16384,2560] parents (A16Only) write their A16
 * projection into conv_record and need no storage. Interval capacity includes the largest
 * transient requirement across its admitted widths. conv_record is caller-owned.
 */
[[nodiscard]] std::size_t gdn_input_proj_conv_record_workspace_capacity_bytes(
    QType parent_qtype, std::int32_t parent_rows, std::int32_t input_rows, LinearPolicy policy,
    std::int32_t batch_size, std::int32_t min_width, std::int32_t max_width);

/**
 * Op: gdn_input_proj_conv_record
 *
 * For each batch row, evaluates the registered projection, width-four causal convolution, SiLU,
 * and q/k/value/z split from the BF16 history selected by initial_state_slots. The current
 * convolution consumes the projection route's private accumulator or its BF16 record (see the
 * per-parent routes below); the Op writes the BF16 projection to conv_record [C,T,B] for
 * subsequent history/replay. Query, key, and value are
 * zero in each row's invalid tail; z is projected for every physical column.
 *
 * The execution domain is B=1..12 and the DFlash2 verify widths: T=2..kDflash2MaxChainWidth
 * without `parent_index` (a chain) and T=kDflash2MinTreeWidth..kDflash2TreeMaxWidth with it (a
 * tree); FP8 also admits chains through T=16, and the [16384,2560] parents also admit T=1
 * chains. valid_columns is empty for dense input or device
 * I32 [B], with each caller-supplied extent in [1,T]. conv_states is a read-only BF16 [C,3,S]
 * state-pool view, and initial_state_slots contains absolute slots in [0,S). Source state is not
 * modified. Only the valid prefix of conv_record is semantically defined.
 *
 * `parent_index` is null or empty for sequential packed time. When non-null it is the same I32
 * [T,B] (or [T] when B=1) parent tensor used by gated_delta_net_replay_record: token 0 loads the
 * checkpoint history, every other valid token j loads the width-three history saved after its
 * parent, and the kernel saves the post-convolution history for j so siblings can share a parent.
 * NVFP4 aggregates B requests in the A8 MMA family and supplies FP32 current projections to
 * convolution, without a BF16 intermediate. Launches with W*B<=48 keep those projections in a
 * CTA-local tile and fuse convolution/record publication; wider launches use private FP32
 * storage. A8 uses row-scaled E4M3 activations and separately scaled K16 partials without
 * requantizing weights. Both routes round saved three-tap history through BF16 at each valid
 * column and never materialize a BF16 projection before convolution.
 *
 * All tensor operands, outputs, conv_record, source state, and live workspace must be disjoint.
 * Registered parents are NVFP4 and FP8_E4M3FN_ROW_BF16S [16384,5120] and [16384,2560], with x
 * [K,T,B] for the parent's K. NVFP4 [16384,5120] admits only AllowA8. FP8 [16384,5120] admits
 * A16Only/AllowA8 but uses qualified A16 arithmetic throughout W=2..16; its batches preserve the
 * C=1 panel reduction and fused/materialized convolution profile. Both [16384,2560] parents admit
 * only A16Only over B=1..12 with T=1..16 chains and T=2..16 trees: one A16 projection over all
 * T*B columns writes conv_record, and convolution consumes that BF16-represented projection
 * (directly in fused FP8 decode, from the record otherwise), so a row matches a B=1 call
 * within the A16 criterion rather than bit for bit. That projection and convolution are
 * the ones gdn_input_proj_conv_snapshot runs at the same T and B, so a chain's query/key/value/z
 * and its conv_record are bit-identical to that snapshot's outputs and newest history tap; at T=1
 * this is the read-only counterpart of a width-one decode step. Unregistered policies or widths
 * throw std::invalid_argument.
 */
void gdn_input_proj_conv_record(const Tensor& x, const Weight& query_key_value_z_weight,
                                const Tensor& conv_weight, const Tensor& conv_states,
                                const Tensor& valid_columns, const Tensor& initial_state_slots,
                                Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value,
                                Tensor& z, LinearPolicy policy, WorkspaceArena& workspace,
                                cudaStream_t stream, const Tensor* parent_index = nullptr);

/**
 * NVFP4 AllowA8 single-parent record-producing form over an already quantized activation
 * (ninfer/ops/a8_activation.h): x.codes is [5120,W,B] and x.scales [W,B], with W and B in the
 * policy form's NVFP4 domain. For the same BF16 activation every output is bit-identical to the
 * policy form with LinearPolicy::AllowA8; all other operands, the workspace capacity, and the
 * oracle (evaluated on the represented activation codes * scales) are that form's.
 */
void gdn_input_proj_conv_record(const A8Activation& x, const Weight& query_key_value_z_weight,
                                const Tensor& conv_weight, const Tensor& conv_states,
                                const Tensor& valid_columns, const Tensor& initial_state_slots,
                                Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value,
                                Tensor& z, WorkspaceArena& workspace, cudaStream_t stream,
                                const Tensor* parent_index = nullptr);

/**
 * Applies the A16-only single-parent record-producing form, which FP8 parents and the NVFP4
 * [16384,2560] parent admit.
 */
void gdn_input_proj_conv_record(const Tensor& x, const Weight& query_key_value_z_weight,
                                const Tensor& conv_weight, const Tensor& conv_states,
                                const Tensor& valid_columns, const Tensor& initial_state_slots,
                                Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value,
                                Tensor& z, WorkspaceArena& workspace, cudaStream_t stream,
                                const Tensor* parent_index = nullptr);

} // namespace ninfer::ops
