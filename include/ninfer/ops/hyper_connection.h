#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

// Qwen3.8-Flash-Next hyper-connection geometry: four residual streams of the 2560-wide hidden
// state, read through a rank-320 gate.
inline constexpr std::int32_t kHyperConnectionHidden  = 2560;
inline constexpr std::int32_t kHyperConnectionStreams = 4;
inline constexpr std::int32_t kHyperConnectionFlat =
    kHyperConnectionHidden * kHyperConnectionStreams;
inline constexpr std::int32_t kHyperConnectionRank = 320;

/**
 * Weights of one hyper-connection read. `norm` is the zero-centered BF16 RMSNorm weight [10240]
 * (gamma = 1 + norm). `down` [320,10240] and `up` [10240,320] are FP8_E4M3FN_ROW_BF16S RowScale
 * weights (Weight n = rows, k = inputs): row-major E4M3FN codes and one BF16 multiplier per row in
 * the row-scale-v1 payload, so down[r,k] = code[r,k] * multiplier[r] exactly. `write` is the BF16
 * [4,10240] block-inject projection of a decoder block's read (Tensor ne {10240, 4}); the model's
 * final read has none and leaves write.data null. All are contiguous, 16-byte aligned device
 * tensors that stay valid and unmodified while a read runs.
 */
struct HyperConnectionWeights {
    Tensor norm;
    Weight down;
    Weight up;
    Tensor write;
};

/**
 * Op: hyper_connection_broadcast
 *
 * residual[d,i,t] = embedding[d,t] for every stream i. embedding is BF16 [2560,T], residual BF16
 * [2560,4,T] (stream-major within a token), T >= 1, both contiguous, 16-byte aligned and
 * disjoint. Exact copy; no workspace; asynchronous on stream and graph-capturable.
 */
void hyper_connection_broadcast(const Tensor& embedding, Tensor& residual, cudaStream_t stream);

/** Workspace bytes of hyper_connection_read for any T <= max_tokens. */
[[nodiscard]] std::size_t hyper_connection_read_workspace_bytes(std::int32_t max_tokens);

/**
 * Op: hyper_connection_read
 *
 * For each token t of residual R [2560,4,T], with Rhat_i the RMSNorm (epsilon 1e-6) of stream i
 * scaled by its 2560-value slice of (1 + norm), and Rhat the stream-major concatenation [10240]:
 *
 *   u           = SiLU(down Rhat / 4)                       [320]
 *   G           = reshape(sigmoid(up u), [4,2560])
 *   x[:,t]      = (1/4) sum_i G_i * Rhat_i                  BF16 [2560,T]
 *   write[i,t]  = 2 sigmoid((write Rhat)_i / 4)             FP32 [4,T], only when weights.write
 *                                                           is present
 *
 * down and up are the exact represented FP8 values (code times row multiplier). The oracle
 * evaluates these formulas in FP64 from the represented inputs and weights; x is the only rounded
 * output (BF16). Implementation profiles (private, covered by the same criteria): a projection
 * runs either in FP32 SIMT arithmetic or on BF16 Tensor Cores with FP32 accumulation, where the
 * exact codes, the BF16 write rows and the BF16 residual are exact operands and every FP32
 * operand is entered as a two-term BF16 split hi + lo (about 2^-17 relative). The Tensor Core
 * operands are, for down, code * (1 + norm) against R (from 160 tokens) or the code against
 * R * (1 + norm) (3 to 32 tokens), and for up the code against u. The up projection uses Tensor
 * Cores for 2 to 32 tokens and from 160, the down projection for 3 to 32 and from 160 tokens;
 * Rhat, the normalization, the gates and the mix are always FP32.
 * write_scale must be a valid FP32 [4,T] tensor exactly when weights.write is present and is
 * otherwise ignored. Tokens are independent. Inputs (the whole down and up payloads included),
 * outputs and workspace are pairwise disjoint.
 * workspace must hold hyper_connection_read_workspace_bytes(T); its allocations are released on
 * return. Invalid arguments throw std::invalid_argument. Asynchronous on stream and
 * graph-capturable for a fixed T and fixed addresses.
 */
void hyper_connection_read(const Tensor& residual, const HyperConnectionWeights& weights, Tensor& x,
                           Tensor& write_scale, WorkspaceArena& workspace, cudaStream_t stream);

/**
 * Op: hyper_connection_inject
 *
 * residual_out[d,i,t] = residual[d,i,t] + write_scale[i,t] * block_output[d,t], evaluated in FP32
 * and rounded once to BF16. residual and residual_out are BF16 [2560,4,T] and may alias exactly;
 * block_output is BF16 [2560,T] and write_scale FP32 [4,T], both disjoint from residual_out. All
 * are contiguous and 16-byte aligned. No workspace; asynchronous on stream and graph-capturable.
 */
void hyper_connection_inject(const Tensor& residual, const Tensor& block_output,
                             const Tensor& write_scale, Tensor& residual_out, cudaStream_t stream);

} // namespace ninfer::ops
