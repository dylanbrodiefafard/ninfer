#pragma once

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/linear.h"
#include "ninfer/types.h"

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

inline constexpr std::int32_t kGroupedDynamicConvHidden    = 5120;
inline constexpr std::int32_t kGroupedDynamicConvGroupSize = 16;
inline constexpr std::int32_t kGroupedDynamicConvGroups    = 320;
inline constexpr std::int32_t kGroupedDynamicConvKernel    = 2;
inline constexpr std::int32_t kGroupedDynamicConvProjRows =
    2 * kGroupedDynamicConvKernel * kGroupedDynamicConvGroups; // 1280
inline constexpr std::int32_t kGroupedDynamicConvMaxBatch =
    static_cast<std::int32_t>(kDFlashMaximumConcurrency);
inline constexpr std::int32_t kGroupedDynamicConvMaxWidthWhenBatched = 16;

/**
 * Op: grouped_dynamic_conv_prepare / grouped_dynamic_conv_finish /
 *     grouped_dynamic_conv_finish_normalized
 *
 * Math / indexing:
 *   Let D=5120, G=320, group_size=16, kernel=2. For each column (t,b) the caller supplies the
 *   dynamic-kernel projection
 *
 *     proj[n,t,b] = sum_{k=0}^{D-1} W[n,k] * hidden[k,t,b],   n in [0,1280),
 *
 *   evaluated by the Linear Op from the same hidden. Split n = phase * 640 + offset * 320 + group
 *   with phase,offset in {0,1} and group in [0,G). Prepare uses phase 0 and stashes phase 1;
 *   finish uses the stashed phase-1 values. The causal grouped convolution at phase p is
 *
 *     values[d,t,b,0] = hidden[d,t,b]
 *     values[d,t,b,1] = hidden[d,t-1,b] if t>=1 else 0
 *     out[d,t,b]      = sum_{j=0,1}
 *                         (base[d,j,p] + dynamic[group(d),j,t,b]) * values[d,t,b,j]
 *
 *   with group(d)=floor(d/16). Prepare publishes BF16(out) with p=0 to `prepared` and/or the Q4
 *   activation image of those values (ninfer/ops/linear.h), and writes
 *   dynamic[g,j,t,b] = proj[640 + j*320 + g, t, b] into `finish_dynamic`. Finish reads that stash,
 *   applies p=1 and adds the rounded result to the residual stream in place:
 *   residual[d,t,b] = BF16(residual[d,t,b] + BF16(out[d,t,b])). finish_normalized instead
 *   publishes that sum to residual_out (residual is unchanged) and also the next sublayer's
 *   RMSNorm normalized[d,t,b] = BF16(r[d,t,b] * rsqrt(mean_d(r[d,t,b]^2) + eps) * norm[d]) of the
 *   represented sum r (the BF16 seam ops::rmsnorm publishes), plus, when requested, the Q4
 *   activation image of normalized. There is no persistent conv state; padding is zeros at the
 *   start of the supplied block.
 *
 * Logical shapes:
 *   hidden/prepared/residual/residual_out/normalized are contiguous BF16 [D,T] or [D,T,B].
 *   projection is contiguous BF16 [1280,T] or [1280,T,B]. finish_dynamic is contiguous BF16
 *   [G,2,T] or [G,2,T,B]. base_kernel is contiguous BF16 [D,2,2] stored D-fastest, then kernel
 *   offset, then phase (physical layout of a PyTorch [2,2,D] parameter). norm is BF16 [D]. T is any
 *   positive value at B=1; B=2..8 admits T=1..16. An image is bound for (T*B, 5120).
 *
 * Numeric:
 *   The oracle evaluates the complete formula in FP64 from the represented BF16 activations and
 *   the represented projection. BF16 outputs are promoted and compared directly with that result;
 *   an image decodes exactly to the A8G64 codec of the published BF16 values. Output storage
 *   rounding and kernel staging are implementation-defined.
 *
 * Effects:
 *   Prepare writes all of finish_dynamic and every requested output (at least one of prepared and
 *   image). Finish updates all of residual. finish_normalized writes all of residual_out,
 *   normalized, and the requested image. Outputs alias no input and no other output; the image
 *   span overlaps no tensor operand. An image's status word is published by the launch's last
 *   finishing block through the target's completion words (ninfer/ops/linear.h).
 */
void grouped_dynamic_conv_prepare(const Tensor& hidden, const Tensor& base_kernel,
                                  const Tensor& projection, Tensor* prepared,
                                  Tensor& finish_dynamic, const Q4ActivationImageTarget* image,
                                  hipStream_t stream);

void grouped_dynamic_conv_finish(const Tensor& hidden, const Tensor& base_kernel,
                                 const Tensor& finish_dynamic, Tensor& residual,
                                 hipStream_t stream);

void grouped_dynamic_conv_finish_normalized(const Tensor& hidden, const Tensor& base_kernel,
                                            const Tensor& finish_dynamic, const Tensor& residual,
                                            Tensor& residual_out, const Tensor& norm, float eps,
                                            Tensor& normalized,
                                            const Q4ActivationImageTarget* image,
                                            hipStream_t stream);

} // namespace ninfer::ops
