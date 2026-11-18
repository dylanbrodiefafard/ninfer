#pragma once

#include "core/tensor.h"
#include "ninfer/ops/moe_helper_contract.h"

#include <cuda_runtime.h> // cudaStream_t

namespace ninfer::ops {

// The RTX 5090 ends of the Qwen3.8-Flash-Next expert-helper split
// (docs/maintainer/qwen3.8-flash-next-expert-helper.md, "Layer dataflow"): moe_expert_partition
// packs the tokens with helper-owned routes for the R9700, moe_experts_local_partial
// (ninfer/ops/moe_experts.h) computes the local share, and moe_expert_partial_join adds the
// helper's FP32 partials and rounds the MoE output once.
//
// Both Ops use a per-layer local-expert bitset local_experts: I32 [16] holding U32 bit patterns,
// bit e % 32 of word e / 32 set when expert e is local (kMoeExpertMaskWords).

/**
 * Op: moe_expert_partition
 *
 * With H the experts whose local_experts bit is clear, token t participates when one of its route
 * entries expert_ids[j,t] is in H. The N participating tokens, in ascending t, get compact rows
 * 0..N-1:
 *
 *   compact_map[t]          = the compact row of t, or -1 when t does not participate
 *   compact_count[0]        = N
 *   compact_x[:,r]          = x[:,t]                                  (r = compact_map[t] >= 0)
 *   compact_ids[j,r]        = expert_ids[j,t] when it is in H, else kMoeHelperInactiveExpert
 *   compact_weights[j,r]    = route_weights[j,t] when expert_ids[j,t] is in H, else 0
 *
 * Route-slot order j and the weights are unchanged (no renormalization). Columns r >= N of the
 * compact outputs are not written. The exact copy is the Op's semantics; its oracle is exact.
 * The three compact outputs are caller-owned stable CUDA VRAM panels, not mapped mailbox planes.
 * In the helper schedule core/expert_mailbox.h copies their populated prefixes into the mapped
 * request only after claiming its writer guard, then fences and release-publishes the header.
 * The helper side zips ids and weights into moe_helper_experts' MoeHelperRoute entries. The
 * compact map stays on CUDA; no separate per-token helper mask is produced or needed.
 *
 * Shapes: x BF16 [2560,T]; expert_ids I32 [10,T] with ids in [0,512) (moe_route's output; an id
 * outside is clamped into [0,512), the Op then reads only documented buffers); route_weights FP32
 * [10,T]; local_experts I32 [16]; compact_map I32 [T]; compact_count I32 [1] (a U32 count
 * consumers may read as such); compact_x BF16 [2560,C], compact_ids I32 [10,C] and
 * compact_weights FP32 [10,C] for a capacity C >= T; 1 <= T <= kMoeHelperMaxTokens. All are
 * contiguous and 16-byte aligned; every output is disjoint from every input and from the other
 * outputs.
 *
 * Execution: one CTA of 256 threads on `stream`, asynchronous, no host synchronization or
 * workspace; inputs (including local_experts) are read on the device, so a captured graph stays
 * valid while their contents change. The outputs are complete in stream order; publishing them
 * to the helper (and the fences that requires) belongs to the transport. Invalid shapes, dtypes,
 * alignment or aliasing throw std::invalid_argument before any work is queued.
 */
void moe_expert_partition(const Tensor& x, const Tensor& expert_ids, const Tensor& route_weights,
                          const Tensor& local_experts, Tensor& compact_map, Tensor& compact_count,
                          Tensor& compact_x, Tensor& compact_ids, Tensor& compact_weights,
                          cudaStream_t stream);

/**
 * Op: moe_expert_partial_join
 *
 *   y[:,t] = BF16(partial[:,t] + helper[:,compact_map[t]])   when compact_map[t] >= 0
 *   y[:,t] = BF16(partial[:,t])                               otherwise
 *
 * where the sum is one FP32 addition and BF16 rounds to nearest even: the MoE output's single
 * rounding (moe_experts_local_partial and moe_helper_experts' FP32 partials enter unrounded). A
 * token that does not participate reads no helper row, so a round with N = 0 never reads helper.
 *
 * Shapes: partial FP32 [2560,T]; compact_map I32 [T] from moe_expert_partition of the same T
 * (values in [-1, N)); helper FP32 [2560,C] with C >= T, whose columns [0,N) hold the helper's
 * outputs (other columns are not read); y BF16 [2560,T]; 1 <= T <= kMoeHelperMaxTokens. All are
 * contiguous and 16-byte aligned; y is disjoint from every input. A compact_map value outside
 * [-1, C) is a precondition violation (it is clamped into [-1, C), so the Op still reads only
 * documented buffers).
 *
 * Execution: asynchronous on `stream`, no workspace, graph-capturable. Invalid shapes, dtypes,
 * alignment or aliasing throw std::invalid_argument before any work is queued.
 */
void moe_expert_partial_join(const Tensor& partial, const Tensor& compact_map, const Tensor& helper,
                             Tensor& y, cudaStream_t stream);

} // namespace ninfer::ops
