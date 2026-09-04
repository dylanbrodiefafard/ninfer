#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::ops {

inline constexpr std::int32_t kQwen4SparseMoeHidden       = 2560;
inline constexpr std::int32_t kQwen4SparseMoeExperts      = 512;
inline constexpr std::int32_t kQwen4SparseMoeTopK         = 10;
inline constexpr std::int32_t kQwen4SparseMoeIntermediate = 640;
inline constexpr std::size_t kQwen4SparseMoeStageBytes    = 8'448'000;

/** Two read-only mapped-host expert banks with the same exact GGML format. */
struct Qwen4MappedRoutedGateUp {
    std::span<const std::byte> gate;
    std::span<const std::byte> up;
    QType qtype = QType::GGML_IQ1_S;
};

/** Persistent views for the fixed UD-IQ1_S Qwen4-preview sparse-MoE verifier. */
struct Qwen4SparseMoeWeights {
    Weight router;                          // contiguous FP32 [512,2560]
    Qwen4MappedRoutedGateUp routed_gate_up; // mapped IQ1_S or IQ2_XXS banks [512,640,2560]
    Weight routed_down;                     // device IQ4_NL [512,2560,640]
    Tensor shared_gate;                     // device FP32 [2560]
    Weight shared_gate_proj;                // device Q5_K or Q6_K [640,2560]
    Weight shared_up;                       // same format and shape as shared_gate_proj
    Weight shared_down;                     // device Q8_0 [2560,640]
};

/** Exact encoded bytes copied for ten gate/up expert pairs of the selected routed format. */
[[nodiscard]] std::size_t qwen4_sparse_moe_selected_stage_bytes(QType routed_qtype);

/** Caller-owned transient device capacity for the exact C=1/T=1 verifier profile. */
[[nodiscard]] std::size_t qwen4_sparse_moe_workspace_capacity_bytes();

/**
 * Op: qwen4_sparse_moe
 *
 * Computes the complete Qwen4-preview C=1/T=1 sparse-MoE formula. The GPU evaluates the FP32
 * 512-row router, FP32 softmax, stable `(ideal probability descending, expert id ascending)` top
 * ten, selected-weight renormalization, ten routed SwiGLU experts, the independently gated shared
 * SwiGLU expert, and their sum. Because ideal softmax is monotone, selected ids are ranked from
 * raw router logits so FP32 exponential underflow cannot change their order; selected weights are
 * the renormalized FP32 probabilities. destination is the BF16 Store result. selected_ids (I32
 * [10]) and selected_weights (FP32 [10]) are observable route outputs in rank order.
 *
 * routed_gate_up is read-only mapped host storage. After the GPU route completes, the Op copies
 * the selected ids into the first 40 bytes of pinned_stage, synchronizes stream once, overwrites
 * that slot with the ten exact gate/up matrix pairs, and enqueues one H2D transfer into
 * device_stage. The CPU performs no decode or floating-point work. pinned_stage and device_stage
 * each have capacity exactly kQwen4SparseMoeStageBytes; IQ1_S uses 6,400,000 bytes and IQ2_XXS
 * uses the full 8,448,000 bytes. No source bank is pinned, copied in full, or repacked.
 *
 * The production verifier composes ggml_block_linear for every quantized projection. Its private
 * projection and SwiGLU inputs therefore use BF16 storage, while routing, mixture accumulation,
 * and the shared scalar gate use FP32. The independent oracle exact-decodes every selected block
 * and evaluates the complete ideal formula naively in FP64 from represented public inputs. Fixed
 * normwise and finite gross criteria qualify the FP32/BF16 production profile without reproducing
 * its private staging boundaries in the oracle.
 *
 * All device operands, output, route outputs, device stage, and workspace are pairwise disjoint.
 * The mapped banks and pinned stage are mutually disjoint. The caller owns every view and must not
 * reuse either stage or workspace until stream completes. Work after the route synchronization is
 * asynchronous on stream. This verifier Op owns no scheduling, target, registry, or Engine path.
 */
void qwen4_sparse_moe(const Tensor& x, const Qwen4SparseMoeWeights& weights,
                      void* pinned_stage, std::size_t pinned_stage_bytes, Tensor& device_stage,
                      Tensor& selected_ids, Tensor& selected_weights, Tensor& destination,
                      WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::ops
