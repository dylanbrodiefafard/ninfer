#pragma once

#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

struct VisionPatchMergerWeights {
    Tensor norm_weight, norm_bias; // BF16 [1152], ordinary learned gamma/beta.
    Weight fc1;                   // contiguous BF16 [4608,4608]
    Tensor fc1_bias;              // BF16 [4608]
    Weight fc2;                   // contiguous BF16 [2560,4608]
    Tensor fc2_bias;              // BF16 [2560]
};

[[nodiscard]] std::size_t vision_patch_merger_workspace_bytes(std::int32_t groups);

/** Native preview Vision patch merger, with one closed mathematical output:
 * y = W2 GELU_exact(W1 concat(LayerNorm(x_patch,gamma,beta,eps=1e-6)) + b1) + b2.
 * LayerNorm uses population variance independently over each 1152-wide patch.
 * x is contiguous BF16 [1152,4,G], four patches per merge group in TL,TR,BL,BR order;
 * y is BF16 [2560,G]. Matrices and learned norm/bias parameters remain BF16.
 * G is positive. Intermediate normalization/projection/bias/GELU casts are private
 * implementation choices, not oracle boundaries; only represented inputs/output are public.
 * The independent complete FP64 oracle evaluates the ideal formula without private casts.
 * Inputs, weights, output and U8 caller-owned workspace are pairwise non-overlapping.
 * Workspace is contiguous, 256-byte aligned, and at least the queried size.
 * No state, allocation, spatial regrouping, frontend, or Vision tower execution is owned here.
 */
void vision_patch_merger(const Tensor& x,const VisionPatchMergerWeights& weights,
                         Tensor& y,Tensor& workspace,cudaStream_t stream);

} // namespace ninfer::ops
