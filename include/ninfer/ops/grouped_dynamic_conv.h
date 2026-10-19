#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

inline constexpr std::int32_t kGroupedDynamicConvHidden    = 5120;
inline constexpr std::int32_t kGroupedDynamicConvGroupSize = 16;
inline constexpr std::int32_t kGroupedDynamicConvGroups    = 320;
inline constexpr std::int32_t kGroupedDynamicConvKernel    = 2;
inline constexpr std::int32_t kGroupedDynamicConvProjRows =
    2 * kGroupedDynamicConvKernel * kGroupedDynamicConvGroups; // 1280
inline constexpr std::int32_t kGroupedDynamicConvMaxBatch            = 8;
inline constexpr std::int32_t kGroupedDynamicConvMaxWidthWhenBatched = 16;

/**
 * Op: grouped_dynamic_conv_prepare / grouped_dynamic_conv_finish
 *
 * Math / indexing:
 *   Let D=5120, G=320, group_size=16, kernel=2. For each column (t,b) the projection is
 *
 *     proj[n,t,b] = sum_{k=0}^{D-1} W[n,k] * hidden[k,t,b],   n in [0,1280).
 *
 *   Split n = phase * 640 + offset * 320 + group with phase,offset in {0,1} and group in [0,G).
 *   Prepare uses phase 0 and stashes phase 1; finish uses the stashed phase-1 values. The causal
 *   grouped convolution at phase p is
 *
 *     values[d,t,b,0] = hidden[d,t,b]
 *     values[d,t,b,1] = hidden[d,t-1,b] if t>=1 else 0
 *     out[d,t,b]      = sum_{j=0,1}
 *                         (base[d,j,p] + dynamic[group(d),j,t,b]) * values[d,t,b,j]
 *
 *   with group(d)=floor(d/16). Prepare writes out into `prepared` with p=0 and writes
 *   dynamic[g,j,t,b] = proj[640 + j*320 + g, t, b] into `finish_dynamic`. Finish reads that stash
 *   and applies p=1. There is no persistent conv state; padding is zeros at the start of the
 *   supplied block.
 *
 * Logical shapes:
 *   hidden/prepared/out are contiguous BF16 [D,T] or [D,T,B]. finish_dynamic is contiguous BF16
 *   [G,2,T] or [G,2,T,B]. base_kernel is contiguous BF16 [D,2,2] stored D-fastest, then kernel
 *   offset, then phase (physical layout of a PyTorch [2,2,D] parameter). kernel_projection is a
 *   logical [1280,D] matrix. T is any positive value at B=1; B=2..8 admits T=1..16.
 *
 * Supported domain:
 *   Activations and base_kernel are BF16. kernel_projection is BF16_CTRL Contiguous [1280,D], or
 *   a Linear-registered Q4G64_F16S / W8G32_F16S RowSplit / NVFP4 BlockScale problem of the same
 *   logical shape. The Q4/W8/NVFP4 route uses Linear packed sequences to preserve the C=1
 *   reduction, including the qualified NVFP4 W=5 SIMT aggregate.
 *
 * Numeric:
 *   The oracle evaluates the complete formula in FP64 from the represented BF16 activations and
 *   the logical FP32 dequantized projection matrix. BF16 outputs are promoted and compared
 *   directly with that result. Output storage rounding, GEMM association, and kernel staging are
 *   implementation-defined.
 *
 * Effects:
 *   Prepare writes all of prepared and finish_dynamic. Finish writes all of out. Inputs other than
 *   those outputs are unchanged. prepared/out must not alias hidden, base_kernel, finish_dynamic,
 *   or any projection-weight plane. finish_dynamic must not alias hidden or base_kernel.
 *
 * Workspace:
 *   Prepare uses caller-owned transient storage sized by
 *   grouped_dynamic_conv_prepare_workspace_capacity_bytes() for the [1280,T*B] projection and any
 *   Linear child scratch. Finish uses none.
 */
[[nodiscard]] std::size_t grouped_dynamic_conv_prepare_workspace_capacity_bytes(
    QType qtype, std::int32_t min_tokens, std::int32_t max_tokens, std::int32_t batch = 1);

void grouped_dynamic_conv_prepare(const Tensor& hidden, const Tensor& base_kernel,
                                  const Weight& kernel_projection, Tensor& prepared,
                                  Tensor& finish_dynamic, WorkspaceArena& workspace,
                                  cudaStream_t stream);

void grouped_dynamic_conv_finish(const Tensor& hidden, const Tensor& base_kernel,
                                 const Tensor& finish_dynamic, Tensor& out, cudaStream_t stream);

/**
 * Op: grouped_dynamic_conv_finish_residual_rmsnorm
 *
 * Math / indexing:
 *   C[d,t,b] is the phase-1 convolution defined by grouped_dynamic_conv_finish.
 *   residual_ideal[d,t,b] = old_residual[d,t,b] + C[d,t,b].
 *   The published BF16 residual is the observable input to plain RMS normalization:
 *     out[d,t,b] = weight[d] * residual[d,t,b]
 *                    / sqrt(sum_i residual[i,t,b]^2 / 5120 + epsilon).
 *
 * Logical shapes / supported domain:
 *   hidden, residual, out are contiguous BF16 [5120,T] or [5120,T,B], with the finish domain.
 *   base_kernel and finish_dynamic have the finish shapes; weight is contiguous BF16 [5120].
 *   Every pointer is 4-byte aligned. epsilon is finite and positive. All operands are disjoint.
 *
 * Numeric:
 *   The complete residual oracle evaluates C and the residual sum in FP64 from represented
 *   inputs. Only the published residual BF16 boundary is semantic; finish staging is private.
 *   The normalization oracle evaluates the complete plain RMS formula in FP64 from that
 *   represented residual. Output rounding and private arithmetic follow numerical qualification.
 *
 * Effects / workspace / execution:
 *   Updates all of residual in place and writes all of out; other inputs are preserved.
 *   No workspace or persistent state. Operands remain valid through execution on stream;
 *   capture and replay require stable addresses. Invalid operands throw before submitting work.
 */
void grouped_dynamic_conv_finish_residual_rmsnorm(const Tensor& hidden, const Tensor& base_kernel,
                                                  const Tensor& finish_dynamic, Tensor& residual,
                                                  const Tensor& weight, float epsilon, Tensor& out,
                                                  cudaStream_t stream);

} // namespace ninfer::ops
