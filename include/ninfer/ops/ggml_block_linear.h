#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * @brief Decode one stored GGML block matrix and apply a single-column linear projection.
 *
 * This verifier Op computes `out[n] = sum_k decode(w[n,k]) * FP32(x[k])`, where `x` and
 * `out` are contiguous BF16 vectors and `w` is a rank-two `[N,K]` GGML block-row weight.
 * It admits exactly GGML Q8_0, Q4_K, Q5_K, Q6_K, IQ1_S, IQ2_XXS, and IQ4_NL. The device-resident
 * source block bytes are read in place; the Op performs no allocation, repacking, or persistent
 * mutation.
 * The bounded verifier domain is `1 <= N <= 248320`, `1 <= K <= 10240`, with K divisible by the
 * selected format's block value count.
 *
 * The independent oracle decodes each complete block into FP64, promotes the represented BF16
 * input to FP64, and evaluates every dot product with naive FP64 accumulation. Qualification uses
 * the derived pointwise bound `A + max((abs(ref)+A)/255, 2^-134)`, where
 * `A=gamma*sum(abs(w*x))`, `gamma=((p*K+8)*2^-24)/(1-(p*K+8)*2^-24)`, and format-specific
 * `p` is 3/5/5/3/4/5/3 for Q8_0/Q4_K/Q5_K/Q6_K/IQ1_S/IQ2_XXS/IQ4_NL. This accounts for the
 * route's FP32 decode/product/reduction profile and final BF16 representation without reproducing
 * those boundaries in the oracle.
 *
 * This is deliberately a C=1/T=1 numerical-verification route. It is not part of generic Linear
 * dispatch and makes no throughput claim. Input, output, and weight storage must not overlap the
 * output. Work is enqueued on `stream`; the caller owns every storage lifetime through completion.
 */
void ggml_block_linear(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
