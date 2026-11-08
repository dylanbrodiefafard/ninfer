#pragma once

#include "core/tensor.h"
#include "ninfer/ops/a4_activation.h"
#include "ninfer/ops/a8_activation.h"
#include "ninfer/ops/linear.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Computes the single-parent Q/K/output-gate/V projection.
 *
 * The parent stores rows in physical order query, key, output gate, value while the public output
 * argument order is q, gate, k, v. Every route writes the four independently contiguous final
 * allocations directly; no packed parent output is materialized. The NVFP4 A4 profile may use
 * caller-owned transient storage for its private quantized activation.
 *
 * Registered parent forms are:
 *
 * - BF16_CTRL Contiguous `[14336,5120]`, with row counts `[6144,1024,6144,1024]`. `x` is
 *   BF16 `[5120,T]`, q/gate are BF16 `[6144,T]`, and k/v are BF16 `[1024,T]`.
 * - NVFP4 BlockScaleK16M128x4 `[14336,5120]`, with the same logical row and tensor shapes as
 *   BF16_CTRL.
 * - FP8_E4M3FN_ROW_BF16S RowScale `[14336,5120]`, with the same logical row and tensor shapes as
 *   BF16_CTRL. Its represented weight is each signed E4M3FN code times the exact BF16 row
 *   multiplier.
 *
 * `T` is the positive token extent of the Op contract. BF16_CTRL admits only
 * LinearPolicy::A16Only. NVFP4 admits A16Only, AllowA4 and AllowA8. Its A16 route is registered
 * at T=1 and below the W4A4 cutover (T < kA4AttnInputMinTokens); AllowA4 selects activation
 * quantization to NVFP4 from that cutover and AllowA8 selects row-scaled E4M3 activation compute
 * at T>=2, so both admit every positive T, while A16Only at a wider T throws
 * std::invalid_argument. The per16-scale partial accumulation preserves the original represented
 * NVFP4 weights. FP8 admits A16Only and AllowA8 at every positive T.
 *
 * The oracle evaluates every projection independently with naive FP64 accumulation from the
 * logical values represented by the persistent weight and BF16 activation. The final four BF16
 * stores belong to the Op's criterion for the selected activation-compute path.
 *
 * `workspace` is caller-owned call-scoped transient storage sized by
 * attn_input_proj_workspace_capacity_bytes(). It must not overlap the input, parent weight, or any
 * output. The Op does not allocate device memory internally.
 */
[[nodiscard]] std::size_t
attn_input_proj_workspace_capacity_bytes(QType parent_qtype, std::int32_t parent_rows,
                                         std::int32_t input_rows, LinearPolicy policy,
                                         std::int32_t min_tokens, std::int32_t max_tokens);

void attn_input_proj(const Tensor& x, const Weight& query_key_gate_value_weight, Tensor& q,
                     Tensor& gate, Tensor& k, Tensor& v, LinearPolicy policy,
                     WorkspaceArena& workspace, cudaStream_t stream);

/**
 * Applies the A16-only single-parent Q/K/output-gate/V projection without transient workspace.
 */
void attn_input_proj(const Tensor& x, const Weight& query_key_gate_value_weight, Tensor& q,
                     Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream);

/**
 * The NVFP4 AllowA8 single-parent projection of an already quantized activation
 * (ninfer/ops/a8_activation.h) of [5120,T], T >= 2 (the widths at which AllowA8 selects W4A8).
 * For the same BF16 activation the four outputs are bit-identical to the policy overload's. The
 * weight is NVFP4 [14336,5120]; q/gate/k/v are as above. The oracle is the overload's, evaluated on
 * the represented activation codes[k,t] * scales[t]. Invalid arguments throw
 * std::invalid_argument. No workspace.
 */
void attn_input_proj(const A8Activation& x, const Weight& query_key_gate_value_weight, Tensor& q,
                     Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream);

/**
 * The NVFP4 AllowA4 single-parent projection of an already quantized activation
 * (ninfer/ops/a4_activation.h) of [5120,T], T >= kA4AttnInputMinTokens (the widths at which
 * AllowA4 selects W4A4), whose input scale divisor equals the weight's. For the same BF16
 * activation the four outputs are bit-identical to the policy overload's with AllowA4. The weight
 * is NVFP4 [14336,5120]; q/gate/k/v are as above. The oracle is the overload's, evaluated on the
 * represented activation. Invalid arguments throw std::invalid_argument. No workspace.
 */
void attn_input_proj(const A4Activation& x, const Weight& query_key_gate_value_weight, Tensor& q,
                     Tensor& gate, Tensor& k, Tensor& v, cudaStream_t stream);

} // namespace ninfer::ops
