#pragma once

// ninfer::ops - fused gate/up projection followed by SwiGLU.

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Returns the maximum transient capacity required by LinearSwiGLU for any T in [1,max_tokens].
 * `max_tokens` sizes caller-owned storage; it does not cap later Op calls when sufficient storage
 * is provided.
 */
[[nodiscard]] std::size_t linear_swiglu_workspace_bytes(std::int32_t gate_up_rows,
                                                        std::int32_t max_tokens);

/**
 * Op: linear_swiglu
 *
 * Math / indexing:
 *   gate_up = Linear(x, gate_up_weight); M=gate_up_rows/2;
 *   ideal[i,t] = SiLU(gate_up[i,t]) * gate_up[M+i,t].
 *
 * Logical shapes / supported domain:
 *   T may be any positive value. The registered RowSplit profiles are:
 *   - Q4G64_F16S weight [34816,5120], x [5120,T], out [17408,T];
 *   - W8G32_F16S weight [12288,2048], x [2048,T], out [6144,T].
 *   Inputs and output are contiguous BF16; packed scales are FP16.
 *
 * Numeric:
 *   The oracle exact-decodes the registered weight and evaluates `ideal` naively in FP64 from the
 *   represented inputs. The BF16 output is promoted and compared directly with that result; output
 *   storage rounding belongs to LinearSwiGLU's named A16 criterion, not the oracle. Production
 *   routes may fuse or materialize gate/up and may choose their natural accumulator, staging, and
 *   workspace precision; those private choices are not semantic rounding boundaries.
 *
 * Effects:
 *   Writes the full output; x/weight and output must not alias.
 *
 * Workspace:
 *   Caller-owned transient storage reported by linear_swiglu_workspace_bytes(gate_up_rows,T),
 *   scoped to the call. The W8 profile requires zero bytes. There is no persistent state side
 *   effect.
 */
void linear_swiglu(const Tensor& x, const Weight& gate_up_weight, Tensor& out, WorkspaceArena& ws,
                   cudaStream_t stream);

} // namespace ninfer::ops
