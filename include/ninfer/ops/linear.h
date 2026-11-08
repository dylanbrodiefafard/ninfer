#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * @brief Permitted private activation-compute profiles for a linear projection.
 *
 * The policy constrains private route selection; it does not select a kernel or prescribe a
 * particular MMA instruction. The public activation and output tensors remain BF16 for every
 * policy.
 */
enum class LinearPolicy : std::uint8_t {
    A16Only, ///< Admit only A16 compute profiles.
    AllowA8, ///< Admit either A16 or A8 compute profiles.
    AllowA4, ///< Admit either A16 or A4 compute profiles.
};

/**
 * Returns the caller-owned transient capacity required by Linear for every T in the inclusive
 * `[min_tokens,max_tokens]` interval. Invalid registered profiles, policies, or intervals throw;
 * a legal route that requires no transient storage returns zero.
 */
[[nodiscard]] std::size_t linear_workspace_capacity_bytes(QType qtype, std::int32_t output_rows,
                                                          std::int32_t input_rows,
                                                          LinearPolicy policy,
                                                          std::int32_t min_tokens,
                                                          std::int32_t max_tokens);

/**
 * @brief Applies a bias-free matrix projection independently to every input column.
 *
 * @details The ideal mathematical result is
 *
 * @f[
 *   \mathrm{ideal}_{n,t} =
 *   \sum_{k=0}^{K-1}
 *     \mathrm{FP32Dequant}(w)_{n,k}\,\mathrm{FP32}(x_{k,t}).
 * @f]
 *
 * `out` stores a BF16 approximation of this ideal result under the named numerical criterion for
 * the selected private activation-compute path.
 *
 * @par Logical tensors and layout
 * `x` is contiguous, non-null, 16-byte-aligned BF16 `[K,T]`, `w` has logical shape `[N,K]`, and
 * `out` is contiguous, non-null, 16-byte-aligned BF16 `[N,T]`. Every logical extent is positive;
 * in particular, `T=0` is invalid rather than a no-op. Dimension zero is stored fastest. The Op has
 * no bias, activation, residual addition, or transpose mode.
 *
 * @par Supported execution domain
 * Registered execution uses RowSplit Q4G64_F16S, Q5G64_F16S, Q6G64_F16S, or W8G32_F16S weights
 * with FP16 scales, block-scaled NVFP4 weights, or row-scaled FP8_E4M3FN_ROW_BF16S weights.
 * The FP8 problem is the vocabulary head
 * [248320,5120] at every positive T; the FP8 Text projections run through their fused Ops.
 * Its represented values are signed E4M3FN codes times exact BF16 row multipliers.
 * Each format owns a finite registry of exact physical weight problems and selects its kernel
 * internally; a valid encoding and alignment do not imply arbitrary N/K support. The NVFP4
 * problems serve the DFlash2 drafter and MTP; the NVFP4 Text projections run through their fused
 * Ops. Each NVFP4 problem admits T=1, plus the widths its callers launch (k is the draft window
 * in [1, kDflash2MaxDraftTokens], B a batch in [1, kDflash2MaxBatch]):
 * - `[5120,25600]`, `[6144,5120]` and `[5120,4096]` (DFlash feature, QKV and attention output):
 *   every positive T, A16-only.
 * - `[1280,5120]` (DFlash conv projection): A16-only passes of m proposal blocks of k + 1
 *   columns within 32 columns, and wider extents made of such passes.
 * - `[256,5120]` (DFlash selector): A16-only T=k.
 * - `[34816,5120]` and `[5120,17408]` (DFlash MLP gate/up and down): AllowA4 through the widest
 *   packed proposal, (kDflash2MaxDraftTokens + 1) * kDflash2MaxBatch columns, which selects W4A4
 *   from T=2 (gate/up) and T=3 (down); the down projection's T=2 is A16.
 * - `[5120,10240]` (MTP `fc`): every positive T; AllowA4 selects W4A4 at T>=8.
 * - `[14336,5120]`, `[16384,5120]` and `[5120,6144]`: T=1 only.
 * Other widths and policies throw std::invalid_argument. Registered W8 problems
 * `[5120,25600]`, `[5120,4096]`,
 * `[1280,5120]`, and `[256,5120]`, and registered Q4 problems `[5120,25600]`, `[5120,4096]`,
 * `[5120,17408]`, `[1280,5120]`, and `[256,5120]`, also accept every positive T. Registered Vision
 * problems accept raw-patch P in `{4,8,...,131072}` or merged-token V in `[1,32768]`; a matrix
 * column does not inherently represent a text token. BF16_CTRL, FP32_CTRL, and I32_CTRL weights are
 * unsupported; the BF16 Text projections run through their fused Ops.
 *
 * @par Numerical contract
 * Test fixture code materializes the persistent weight as its logical FP32 dequantized matrix.
 * The one Linear oracle accepts that matrix and the FP32 values represented by the BF16 activation,
 * evaluates every complete dot product with naive FP64 accumulation, and retains the FP64 result.
 * The BF16 output is promoted and compared against that result. Output representation,
 * accumulator precision, activation quantization, staging, reduction order, and kernel schedule
 * are private implementation effects covered by the named tolerance for the selected
 * activation-compute path; none is copied into the oracle. Kernel, schedule, template instance,
 * host launcher, and T region do not create separate criteria inside one path.
 *
 * @par Compute policy
 * `policy` specifies the permitted private activation-compute set. A permission does not require a
 * corresponding low-precision route: the resolved plan may remain A16 when that is the qualified
 * choice. Registered Q4/Q5/Q6/W8 formats admit
 * LinearPolicy::A16Only and LinearPolicy::AllowA8. NVFP4 admits A16Only and AllowA4; AllowA4
 * permits the private resolver to select either a qualified A16 route or activation quantization
 * to NVFP4. The selected route depends only on the registered problem and T. Weight codes and
 * per16 E4M3 scales remain unchanged: exact code expansion and separately scaled K16 partials
 * introduce no additional weight quantization. The canonical oracle retains original BF16 inputs;
 * independent codec attribution supplements its check. The FP8 vocabulary projection accepts
 * every policy and always uses A16. FP8 packed sequences retain independent per-sequence routes.
 * Callers that pack B independent sequences as `T = sequence_width * B` must use
 * `linear_packed_sequences`, which preserves the C=1 route by default and owns explicitly
 * qualified aggregate exceptions.
 *
 * @par Workspace
 * `workspace` is caller-owned call-scoped transient storage sized by
 * linear_workspace_capacity_bytes(). It must not overlap x, any weight plane, or out. Linear does
 * not allocate device memory internally.
 *
 * @param[in] x Contiguous, non-null, 16-byte-aligned BF16 input matrix `[K,T]`.
 * @param[in] w Logical weight matrix `[N,K]` in a registered persistent format and layout.
 * @param[out] out Contiguous, non-null, 16-byte-aligned BF16 output matrix `[N,T]`. It must not
 * overlap `x` or any weight plane.
 * @param[in] policy Permitted private activation-compute profiles.
 * @param[in,out] workspace Caller-owned transient arena.
 * @param[in] stream CUDA stream on which execution is enqueued.
 */
void linear(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
            WorkspaceArena& workspace, cudaStream_t stream);

/**
 * @brief Applies the A16-only form of the bias-free matrix projection.
 *
 * @details This overload admits only A16 compute and requires no transient workspace. All tensor,
 * weight, aliasing, and execution-domain requirements of the policy-bearing overload apply.
 *
 * @param[in] x Contiguous BF16 input matrix `[K,T]`.
 * @param[in] w Logical weight matrix `[N,K]` in a registered persistent format and layout.
 * @param[out] out Contiguous BF16 output matrix `[N,T]`.
 * @param[in] stream CUDA stream on which execution is enqueued.
 */
void linear(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);

/**
 * @brief Applies Linear independently to each packed sequence of `sequence_width` columns.
 *
 * @details Concurrent DFlash/target activations are stored as contiguous `[K, width*B]`. NVFP4
 * SmallT, Q4 draft-head, and W8/Q4 SIMT/MMA routes are specialized by exact T, so launching once
 * at `T=width*B` can select a different kernel than sequential C=1 (`T=width`). This entry
 * launches one Linear per sequence by default. Qualified exceptions aggregate the A16 W8
 * vocabulary W=2..8 profile in passes of at most 48 same-reduction columns and aggregate selected
 * NVFP4 DFlash (the conv projection in passes of at most 32 columns) and Q4 27B draft-head
 * profiles. `T` must be a positive multiple of `sequence_width`.
 * Tensor, weight, aliasing, policy, and workspace requirements match `linear`.
 *
 * @param[in] sequence_width Per-sequence column count (the C=1 T).
 */
void linear_packed_sequences(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                             WorkspaceArena& workspace, cudaStream_t stream,
                             std::int32_t sequence_width);

void linear_packed_sequences(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream,
                             std::int32_t sequence_width);

/**
 * @brief Projects the K and V row ranges of one registered fused QKV weight.
 *
 * @details `w` is the original immutable `[6144,5120]` descriptor in block-scaled NVFP4 or
 * RowSplit W8G32_F16S/Q4G64_F16S, including its full payload and original scale planes.
 * `x` is contiguous BF16 `[5120,T]`; `key` and
 * `value` are distinct contiguous BF16 `[1024,T]` outputs. The ideal results are the complete
 * dot products of represented BF16 inputs with exactly decoded original weight rows
 * `[4096,5120)` and `[5120,6144)`, respectively. Outputs use Linear's A16 numerical criterion;
 * activation quantization is not admitted. No Q output is produced.
 *
 * Tensor and weight-plane pointers must be non-null and 16-byte aligned. Outputs cannot overlap
 * each other, the input, or the original weight payload. Inputs and weights remain unchanged. `T`
 * and `sequence_width` are positive; `T` is a multiple of `sequence_width` representing one to
 * eight packed sequences. Packed requests retain their C=1 arithmetic route: NVFP4 width 1 uses
 * independent GEMV panels and widths >= 2 share the qualified ascending-K A16 MMA reduction.
 * RowSplit formats retain independent per-sequence Linear calls on their valid K/V row views.
 * No workspace or internal device allocation is required.
 *
 * Execution is enqueued on `stream`; caller-owned tensors and weight planes must remain live
 * until execution completes. Invalid tensors, original weight descriptors, or packed shapes
 * throw `std::invalid_argument`; CUDA failures propagate with launch context.
 */
void linear_kv_projection(const Tensor& x, const Weight& w, Tensor& key, Tensor& value,
                          cudaStream_t stream, std::int32_t sequence_width);

} // namespace ninfer::ops
