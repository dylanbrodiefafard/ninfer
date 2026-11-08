#pragma once

// ninfer::ops - fused residual += W @ x.

#include "core/arena.h"
#include "core/tensor.h"
#include "ninfer/ops/a4_activation.h"
#include "ninfer/ops/a8_activation.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Returns the A16-only transient capacity required by LinearAdd for every T in the inclusive
 * [min_tokens,max_tokens] interval. The QType and dimensions are the fixed implementation profile.
 * Invalid profiles or intervals throw; a legal static-zero route returns zero.
 */
[[nodiscard]] std::size_t linear_add_workspace_capacity_bytes(QType qtype, std::int32_t output_rows,
                                                              std::int32_t input_rows,
                                                              std::int32_t min_tokens,
                                                              std::int32_t max_tokens);

[[nodiscard]] std::size_t linear_add_workspace_capacity_bytes(QType qtype, std::int32_t output_rows,
                                                              std::int32_t input_rows,
                                                              LinearPolicy policy,
                                                              std::int32_t min_tokens,
                                                              std::int32_t max_tokens);

/**
 * Op: linear_add
 *
 * Math / indexing:
 *   ideal[:,t] = residual[:,t] + Linear(x,w)[:,t].
 *
 * Logical shapes:
 *   Contiguous BF16 x [K,T] and residual [N,T]. Registered weights are NVFP4
 *   BlockScaleK16M128x4 [5120,6144] or [5120,17408], or BF16_CTRL Contiguous [5120,6144].
 *   T may be any positive value.
 *   FP8_E4M3FN_ROW_BF16S RowScale [5120,6144] and [5120,17408] also admit every positive T;
 *   codes are signed E4M3FN with exact BF16 row multipliers.
 *
 * Numeric:
 *   The oracle reads a registered BF16 weight directly or exact-decodes a registered packed
 *   weight, then evaluates `ideal` naively in FP64 from the represented inputs. The updated BF16
 *   residual is promoted and compared directly with that result; output storage rounding belongs
 *   to LinearAdd's selected A16, A8 or A4 criterion, not the oracle. Production routes may fuse or
 *   materialize the projection and may choose their natural accumulator, activation quantization,
 *   staging, and workspace precision; those private choices are not semantic rounding boundaries.
 *
 * Compute policy:
 *   BF16_CTRL admits only A16Only. NVFP4 admits A16Only, AllowA4 and AllowA8. Its A16 route is
 * registered at T=1 and below the projection's W4A4 cutover (T < kA4Residual6144MinTokens or
 * kA4Residual17408MinTokens); AllowA4 selects the A4 route from that cutover and AllowA8 selects
 * row-scaled E4M3 activation compute at T>=2, so both admit every positive T, while A16Only at a
 * wider T throws std::invalid_argument. Exact weight-code expansion and FP32 per16-scale
 * accumulation preserve the stored weights. FP8 admits AllowA8 at every positive T, with
 * caller-owned A8 activation workspace from T=22 (K=6144) or T=25 (K=17408); its A16 route, and
 * hence A16Only, is registered only below those widths.
 *
 * Effects:
 *   Updates the full residual tensor in place; x/weight must not alias residual.
 *
 * Workspace:
 *   Caller-owned transient storage reported by linear_add_workspace_capacity_bytes(), scoped to
 *   the call. There is no persistent state side effect.
 */
void linear_add(const Tensor& x, const Weight& w, Tensor& residual, WorkspaceArena& ws,
                cudaStream_t stream);

void linear_add(const Tensor& x, const Weight& w, Tensor& residual, LinearPolicy policy,
                WorkspaceArena& ws, cudaStream_t stream);

/**
 * LinearAdd of an already quantized activation: the NVFP4 AllowA8 route above with its row-scaled
 * E4M3 quantization supplied by the producer (ninfer/ops/a8_activation.h). For the same BF16
 * activation the residual update is bit-identical to linear_add(x, w, residual, AllowA8, ...).
 * w is NVFP4 [5120,6144] or [5120,17408]; x is [w.k,T] with T >= 2, the widths at which AllowA8
 * selects W4A8; residual is contiguous, 16-byte aligned BF16 [5120,T] and does not overlap x.
 * The oracle is LinearAdd's evaluated on the represented activation codes[k,t] * scales[t].
 * Invalid arguments throw std::invalid_argument. No workspace.
 */
void linear_add(const A8Activation& x, const Weight& w, Tensor& residual, cudaStream_t stream);

/**
 * LinearAdd of an already quantized activation: the NVFP4 AllowA4 route above with its NVFP4
 * quantization supplied by the producer (ninfer/ops/a4_activation.h), whose input scale divisor
 * equals w's. For the same BF16 activation the residual update is bit-identical to
 * linear_add(x, w, residual, AllowA4, ...). w is NVFP4 [5120,6144] (T >= kA4Residual6144MinTokens)
 * or [5120,17408] (T >= kA4Residual17408MinTokens); residual is contiguous, 16-byte aligned BF16
 * [5120,T] and does not overlap x. The oracle is LinearAdd's evaluated on the represented
 * activation. Invalid arguments throw std::invalid_argument. No workspace.
 */
void linear_add(const A4Activation& x, const Weight& w, Tensor& residual, cudaStream_t stream);

} // namespace ninfer::ops
