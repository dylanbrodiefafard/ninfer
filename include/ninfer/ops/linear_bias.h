#pragma once
#include "core/tensor.h"
#include <cuda_runtime.h>

namespace ninfer::ops {
/** Complete biased native BF16 projection: ideal[n,t]=sum_k W[n,k]*x[k,t]+bias[n].
 * The independent oracle accumulates the complete formula in FP64 from represented BF16
 * public inputs; there is no semantic cast before bias. Output is contiguous BF16 [N,T].
 * x is contiguous BF16 [K,T], bias is BF16 [N], and W is contiguous BF16_CTRL [N,K].
 * Exact admitted geometries are (N,K)=(1152,1536),(3456,1152),(1152,1152),
 * (4304,1152),(1152,4304),(4608,4608),(2560,4608). T is positive. All storage is
 * pairwise nonoverlapping and16byte aligned. No workspace, allocation or persistent state.
 */
void linear_bias(const Tensor& x,const Weight& weight,const Tensor& bias,Tensor& out,cudaStream_t);
} // namespace ninfer::ops
