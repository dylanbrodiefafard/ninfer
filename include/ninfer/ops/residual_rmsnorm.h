#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

/**
 * Fused residual update + per-layer RMSNorm over the fastest dimension D=ne[0]. For each row r:
 *
 *   x[d,r] = x[d,r] + y[d,r]                       (in-place residual, identical to residual_add)
 *   inv_r  = 1 / sqrt((1/D) * sum_d x[d,r]^2 + eps)
 *   out[d,r] = x[d,r] * inv_r * (1 + weight[d])    (the unit_offset / Offset RMSNorm)
 *
 * `x` is updated in place (the running residual) and `out` holds the normalized hidden for the
 * downstream GEMV. `y`, `x`, `weight`, `out` are contiguous BF16; `y`/`x`/`out` are same-shaped
 * [rows, D] and `weight` is [D]. This replaces the `residual_add` + `rmsnorm(unit_offset=true)`
 * launch pair with a single launch, bit-exact against that pair (the same CTA reduce + epilogue
 * the standalone rmsnorm uses). No workspace or persistent state.
 */
void residual_rmsnorm(const Tensor& y, Tensor& x, const Tensor& weight, float eps, Tensor& out,
                      cudaStream_t stream);

} // namespace ninfer::ops