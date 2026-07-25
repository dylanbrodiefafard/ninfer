#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

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
 * the selected private compute profile.
 *
 * @par Logical tensors and layout
 * `x` is contiguous, non-null, 16-byte-aligned BF16 `[K,T]`, `w` has logical shape `[N,K]`, and
 * `out` is contiguous, non-null, 16-byte-aligned BF16 `[N,T]`. Every logical extent is positive;
 * in particular, `T=0` is invalid rather than a no-op. Dimension zero is stored fastest. The Op has
 * no bias, activation, residual addition, or transpose mode.
 *
 * @par Supported execution domain
 * Registered execution uses RowSplit Q4G64_F16S, Q5G64_F16S, Q6G64_F16S, or W8G32_F16S weights
 * with FP16 scales. Each format owns a finite registry of exact physical weight problems and
 * selects its kernel internally; a valid encoding and alignment do not imply arbitrary N/K
 * support. Text and MTP problems accept every positive column extent T. Registered Vision problems
 * accept raw-patch P in `{4,8,...,131072}` or merged-token V in `[1,32768]`; a matrix column does
 * not inherently represent a text token. BF16_CTRL has a reserved format-local boundary but
 * currently admits no pure Linear problem. FP32_CTRL is unsupported.
 *
 * @par Numerical contract
 * Test fixture code materializes the persistent weight as its logical FP32 dequantized matrix.
 * The one Linear oracle accepts that matrix and the FP32 values represented by the BF16 activation,
 * evaluates every complete dot product with naive FP64 accumulation, and retains the FP64 result.
 * The BF16 output is promoted and compared against that result. Output representation,
 * accumulator precision, activation quantization, staging, reduction order, and kernel schedule
 * are private implementation effects covered by the named tolerance for the selected profile;
 * none is copied into the oracle.
 *
 * @par Compute policy
 * `policy` specifies the permitted private activation-compute set. A permission does not require a
 * corresponding low-precision route: the resolved plan may remain A16 when that is the qualified
 * choice. BF16_CTRL admits only LinearPolicy::A16Only. Registered Q4/Q5/Q6/W8 formats admit
 * LinearPolicy::A16Only and LinearPolicy::AllowA8. No currently registered weight format admits
 * LinearPolicy::AllowA4.
 *
 * @param[in] x Contiguous, non-null, 16-byte-aligned BF16 input matrix `[K,T]`.
 * @param[in] w Logical weight matrix `[N,K]` in a registered persistent format and layout.
 * @param[out] out Contiguous, non-null, 16-byte-aligned BF16 output matrix `[N,T]`. It must not
 * overlap `x` or any weight plane.
 * @param[in] policy Permitted private activation-compute profiles.
 * @param[in,out] ws Caller-owned transient storage. It carries no semantic state beyond the call.
 * @param[in] stream CUDA stream on which execution is enqueued.
 */
void linear(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy, WorkspaceArena& ws,
            cudaStream_t stream);

/**
 * @brief Applies the A16-only form of the bias-free matrix projection.
 *
 * @details This overload is exactly equivalent to
 * `linear(x, w, out, LinearPolicy::A16Only, ws, stream)`. All tensor, weight, aliasing, workspace,
 * and execution-domain requirements of the policy-bearing overload apply.
 *
 * @param[in] x Contiguous BF16 input matrix `[K,T]`.
 * @param[in] w Logical weight matrix `[N,K]` in a registered persistent format and layout.
 * @param[out] out Contiguous BF16 output matrix `[N,T]`.
 * @param[in,out] ws Caller-owned transient storage.
 * @param[in] stream CUDA stream on which execution is enqueued.
 */
void linear(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws, cudaStream_t stream);

} // namespace ninfer::ops
