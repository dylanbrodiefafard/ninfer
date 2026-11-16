#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

inline constexpr std::int32_t kGroupedDynamicConvGroupSize           = 16;
inline constexpr std::int32_t kGroupedDynamicConvKernel              = 2;
inline constexpr std::int32_t kGroupedDynamicConvMaxBatch            = 12;
inline constexpr std::int32_t kGroupedDynamicConvMaxWidthWhenBatched = 16;

// Registered hidden widths D: 5120 (Qwen3.8-27B DFlash2) and 2560 (Qwen3.8-Flash-Next DFlash2).
[[nodiscard]] constexpr bool grouped_dynamic_conv_hidden_registered(std::int32_t hidden) noexcept {
    return hidden == 5120 || hidden == 2560;
}

// G = D / 16 groups of one hidden width.
[[nodiscard]] constexpr std::int32_t grouped_dynamic_conv_groups(std::int32_t hidden) noexcept {
    return hidden / kGroupedDynamicConvGroupSize;
}

// 2 * kernel * G projection rows of one hidden width: D/4.
[[nodiscard]] constexpr std::int32_t
grouped_dynamic_conv_projection_rows(std::int32_t hidden) noexcept {
    return 2 * kGroupedDynamicConvKernel * grouped_dynamic_conv_groups(hidden);
}

/**
 * Op: grouped_dynamic_conv_prepare / grouped_dynamic_conv_finish
 *
 * Math / indexing:
 *   Let D be a registered hidden width, G=D/16 the group count, group_size=16, kernel=2, and
 *   P=4G=D/4 the projection rows. For each column (t,b) the projection is
 *
 *     proj[n,t,b] = sum_{k=0}^{D-1} W[n,k] * hidden[k,t,b],   n in [0,P).
 *
 *   Split n = phase * 2G + offset * G + group with phase,offset in {0,1} and group in [0,G).
 *   Prepare uses phase 0 and stashes phase 1; finish uses the stashed phase-1 values. The causal
 *   grouped convolution at phase p is
 *
 *     values[d,t,b,0] = hidden[d,t,b]
 *     values[d,t,b,1] = hidden[d,t-1,b] if t>=1 else 0
 *     out[d,t,b]      = sum_{j=0,1}
 *                         (base[d,j,p] + dynamic[group(d),j,t,b]) * values[d,t,b,j]
 *
 *   with group(d)=floor(d/16). Prepare writes out into `prepared` with p=0 and writes
 *   dynamic[g,j,t,b] = proj[2G + j*G + g, t, b] into `finish_dynamic`. Finish reads that stash
 *   and applies p=1. There is no persistent conv state; padding is zeros at the start of the
 *   supplied block.
 *
 * Logical shapes:
 *   D is hidden.ne[0], one of the widths grouped_dynamic_conv_hidden_registered() admits.
 *   hidden/prepared/out are contiguous BF16 [D,T] or [D,T,B]. finish_dynamic is contiguous BF16
 *   [G,2,T] or [G,2,T,B]. base_kernel is contiguous BF16 [D,2,2] stored D-fastest, then kernel
 *   offset, then phase (physical layout of a PyTorch [2,2,D] parameter). kernel_projection is a
 *   logical [P,D] matrix. T is any positive value at B=1; B=2..12 admits T=1..16.
 *
 * Supported domain:
 *   Activations and base_kernel are BF16. kernel_projection is BF16_CTRL Contiguous [P,D] at
 *   either width, or a Linear-registered problem of the same logical shape: Q4G64_F16S /
 *   W8G32_F16S RowSplit or NVFP4 BlockScale at D=5120, NVFP4 BlockScale at D=2560. The
 *   quantized route uses Linear packed sequences to preserve the C=1 reduction, including the
 *   qualified NVFP4 aggregates (D=5120 conv-projection passes, the D=2560 A16 MMA).
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
 *   grouped_dynamic_conv_prepare_workspace_capacity_bytes() for the [P,T*B] projection and any
 *   Linear child scratch. Finish uses none. The query takes the projection qtype and D and throws
 *   for an unregistered pair.
 */
[[nodiscard]] std::size_t grouped_dynamic_conv_prepare_workspace_capacity_bytes(
    QType qtype, std::int32_t hidden, std::int32_t min_tokens, std::int32_t max_tokens,
    std::int32_t batch = 1);

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
 *                    / sqrt(sum_i residual[i,t,b]^2 / D + epsilon).
 *
 * Logical shapes / supported domain:
 *   hidden, residual, out are contiguous BF16 [D,T] or [D,T,B] at a registered width D, with the
 *   finish domain. base_kernel and finish_dynamic have the finish shapes; weight is contiguous
 *   BF16 [D].
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
