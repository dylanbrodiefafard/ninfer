#pragma once

// ninfer::ops - fused GDN Q/K/V/Z input projections.

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Op: gdn_input_proj
 *
 * Math / indexing:
 *   qkv[:,t] = concat(qk_weight * x[:,t], value_z_weight[0:6144,:] * x[:,t])
 *   z[:,t]   = value_z_weight[6144:12288,:] * x[:,t]
 *
 * Logical shapes:
 *   x [5120,T], qk weight/output rows 4096, value/z rows 6144 each, qkv [10240,T],
 *   z [6144,T]. T may be any positive value. x, qkv, and z are contiguous BF16.
 *   qk_weight is Q4G64_F16S RowSplit [4096,5120] and value_z_weight is one
 *   Q5G64_F16S RowSplit parent [12288,5120] in [value,z] row order, both with FP16 scales.
 *
 * Numeric:
 *   The oracle exact-decodes both weight parents and evaluates all four logical projections
 *   naively in FP64 from the represented input. The BF16 qkv and z outputs are promoted and
 *   compared directly with those ideal values; final output storage rounding belongs to
 *   GdnInputProj's named A16 criterion, not the oracle. Production routes may choose their
 *   private precision independently; every registered route writes both final allocations.
 *
 * Effects:
 *   Writes the full qkv and z outputs; inputs and outputs must not alias.
 *
 * Workspace:
 *   No transient bytes are required.
 */
void gdn_input_proj(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                    Tensor& qkv, Tensor& z, cudaStream_t stream);

/**
 * Single-parent GDN projection. Registered parent forms are:
 *
 * - W8G32_F16S RowSplit [12288,2048], with stored row counts [2048,2048,4096,4096];
 * - NVFP4 BlockScaleK16M128x4 [16384,5120], with stored row counts [2048,2048,6144,6144].
 *
 * The first three ranges are written contiguously to qkv and the final range is written to z.
 * W8 admits A16 only. NVFP4 admits A16Only and AllowA4; AllowA4 remains A16 for T<=16 and may
 * privately quantize the represented BF16 activation for larger T. Every route writes the two
 * independent final allocations directly. The complete projection is evaluated against the same
 * exact-decode/naive-FP64 oracle; activation quantization and the production reduction profile are
 * private effects covered by the selected criterion. x, qkv, and z must be pairwise
 * non-overlapping.
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
 * Returns the transient capacity required by gdn_input_proj_conv_snapshot. The 35B W8 route writes
 * final outputs from projection epilogues for exact T=1..16; the 27B Q4/Q5 route may reserve
 * private BF16 staging for its small-T kernels. Extents outside a target route's optimized range
 * use the composed implementation and require two BF16 [C,T] intermediates. The row geometry is
 * the fixed profile; the query covers the inclusive token interval and throws for invalid values.
 */
[[nodiscard]] std::size_t gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
    std::int32_t query_rows, std::int32_t key_rows, std::int32_t value_rows,
    std::int32_t min_tokens, std::int32_t max_tokens);

/**
 * Returns the transient capacity for the single-parent GDN snapshot form. The registered NVFP4
 * parent is [16384,5120]. A16Only is valid for T<=16 and requires no storage. AllowA4 resolves to
 * the same fused A16 route for T<=16; for larger T it reserves one private BF16 projection plus
 * the workspace required by the public W4A4 Linear route.
 */
[[nodiscard]] std::size_t gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
    QType parent_qtype, std::int32_t parent_rows, std::int32_t input_rows, LinearPolicy policy,
    std::int32_t min_tokens, std::int32_t max_tokens);

/**
 * Op: gdn_input_proj_conv_snapshot
 *
 * Math / indexing:
 *   Let p[:,t] be concat(qk_weight*x[:,t], value_z_weight[0:6144,:]*x[:,t]) and
 *   z[:,t] = value_z_weight[6144:12288,:]*x[:,t]. Starting from the BF16 width-three history
 *   selected by device I32 initial_slot, evaluate the width-four depthwise convolution over p,
 *   apply SiLU, and write its three channel ranges directly to query, key, and value. After token
 *   t, write the resulting width-three projection history to state slot t. Z bypasses convolution.
 *
 * Logical shapes:
 *   The 27B registered form has x [5120,T], Q4 q/k weight [4096,5120], one Q5 value/z parent
 *   [12288,5120], conv_weight [10240,4], conv_states [10240,3,Slots], query/key [2048,T],
 *   value [6144,T], and z [6144,T]. T is positive, Slots>=T, and the device initial_slot value
 *   is in [0,Slots).
 *
 * Numeric:
 *   The oracle exact-decodes packed weights and evaluates projection, convolution, SiLU, z, and
 *   every snapshot value naively in FP64 from represented inputs. BF16 query/key/value/z and
 *   snapshots are promoted and compared directly with those ideal values; their final storage
 *   rounding belongs to the Op's named A16 criterion, not the oracle. Former unfused projection
 *   tensors are not observable cast boundaries; production routes use their natural private
 *   accumulator and staging precision. This two-parent Q4/Q5 form does not quantize activation;
 *   the single-parent policy-bearing form below defines its own permitted compute profiles.
 *
 * Effects:
 *   Writes query/key/value/z and state slots [0,T); other slots are unchanged. Newly projected
 *   convolution channels remain private to the current call while each published snapshot is
 *   BF16.
 */
void gdn_input_proj_conv_snapshot(const Tensor& x, const Weight& qk_weight,
                                  const Weight& value_z_weight, const Tensor& conv_weight,
                                  Tensor& conv_states, const Tensor& initial_slot, Tensor& query,
                                  Tensor& key, Tensor& value, Tensor& z, WorkspaceArena& ws,
                                  cudaStream_t stream);

/**
 * Single-parent form of gdn_input_proj_conv_snapshot. Registered parents are W8G32_F16S RowSplit
 * [12288,2048] and NVFP4 BlockScaleK16M128x4 [16384,5120], both in q/k/value/z row order. W8
 * admits A16Only. NVFP4 admits A16Only for T<=16 and AllowA4 for every positive T. The NVFP4
 * AllowA4 route remains fused A16 through T=16, then privately quantizes the represented BF16
 * activation and composes public W4A4 Linear with one convolution/snapshot post kernel.
 */
void gdn_input_proj_conv_snapshot(const Tensor& x, const Weight& query_key_value_z_weight,
                                  const Tensor& conv_weight, Tensor& conv_states,
                                  const Tensor& initial_slot, Tensor& query, Tensor& key,
                                  Tensor& value, Tensor& z, LinearPolicy policy, WorkspaceArena& ws,
                                  cudaStream_t stream);

/**
 * Applies the A16-only single-parent form. For NVFP4 this convenience overload is valid only for
 * T<=16.
 */
void gdn_input_proj_conv_snapshot(const Tensor& x, const Weight& query_key_value_z_weight,
                                  const Tensor& conv_weight, Tensor& conv_states,
                                  const Tensor& initial_slot, Tensor& query, Tensor& key,
                                  Tensor& value, Tensor& z, WorkspaceArena& ws,
                                  cudaStream_t stream);

} // namespace ninfer::ops
