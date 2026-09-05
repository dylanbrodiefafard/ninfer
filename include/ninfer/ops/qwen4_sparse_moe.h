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
inline constexpr std::int32_t kQwen4SparseMoePipelineSlots = 2;
inline constexpr std::size_t kQwen4SparseMoeRankStageCapacityBytes = 844'800;
inline constexpr std::size_t kQwen4SparseMoePipelineStageBytes =
    kQwen4SparseMoePipelineSlots * kQwen4SparseMoeRankStageCapacityBytes;
inline constexpr std::int32_t kQwen4SparseMoePrefillMaxWidth = 4096;
inline constexpr std::int32_t kQwen4SparseMoePrefillGroupExperts = 32;
inline constexpr std::size_t kQwen4SparseMoePrefillGroupMatrixCapacityBytes =
    static_cast<std::size_t>(kQwen4SparseMoePrefillGroupExperts) *
    kQwen4SparseMoeRankStageCapacityBytes;
inline constexpr std::size_t kQwen4SparseMoePrefillOccurrenceCapacityBytes =
    static_cast<std::size_t>(kQwen4SparseMoeTopK) * kQwen4SparseMoePrefillMaxWidth *
    sizeof(std::int32_t);
inline constexpr std::size_t kQwen4SparseMoePrefillSlotCapacityBytes =
    kQwen4SparseMoePrefillGroupMatrixCapacityBytes +
    kQwen4SparseMoePrefillOccurrenceCapacityBytes;
inline constexpr std::size_t kQwen4SparseMoePrefillPipelineStageBytes =
    kQwen4SparseMoePipelineSlots * kQwen4SparseMoePrefillSlotCapacityBytes;
inline constexpr std::size_t kQwen4SparseMoePrefillHostScratchI32 =
    2ULL * kQwen4SparseMoeTopK * kQwen4SparseMoePrefillMaxWidth +
    4ULL * kQwen4SparseMoeExperts + 1ULL;
inline constexpr std::size_t kQwen4SparseMoePrefillHostScratchBytes =
    kQwen4SparseMoePrefillHostScratchI32 * sizeof(std::int32_t);

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

/** Device-resident expert banks for the exact Qwen4-preview sparse-MoE geometry. */
struct Qwen4ResidentSparseMoeWeights {
    Weight router;           // contiguous FP32 [512,2560]
    Weight routed_gate;      // device IQ1_S or IQ2_XXS [512,640,2560]
    Weight routed_up;        // same format and shape as routed_gate
    Weight routed_down;      // device IQ4_NL [512,2560,640]
    Tensor shared_gate;      // device FP32 [2560]
    Weight shared_gate_proj; // device Q5_K or Q6_K [640,2560]
    Weight shared_up;        // same format and shape as shared_gate_proj
    Weight shared_down;      // device Q8_0 [2560,640]
};

/** Exact encoded bytes copied for one gate/up expert pair of the selected routed format. */
[[nodiscard]] std::size_t qwen4_sparse_moe_rank_stage_bytes(QType routed_qtype);

/**
 * Caller-owned execution resources for the fixed two-stream/two-slot implementation.
 *
 * pinned_stage and device_stage each contain two consecutive
 * kQwen4SparseMoeRankStageCapacityBytes slots. transfer_stream is distinct from the compute
 * stream passed to qwen4_sparse_moe. route_ready orders the GPU route before the selected-id
 * D2H, ids_ready permits the one host wait, transfer_ready makes one slot visible to the compute
 * stream, and consumer_complete prevents its reuse until the preceding expert has consumed it.
 * Every event is caller-created with timing disabled; ids_ready and transfer_ready additionally
 * support host synchronization. At the start of a later call, ids_ready transitively proves both
 * slots free: current route_ready follows all preceding compute consumers, and current ids_ready
 * follows route_ready plus all preceding transfer work. Within one call, ready/complete events
 * explicitly protect slot reuse beginning at rank two. These resources affect only execution and
 * not the Op result. compute_stream binds every use of one pipeline/event set to one stream so the
 * cross-call ordering proof cannot be invalidated by stream substitution.
 */
struct Qwen4SparseMoePipeline {
    void* pinned_stage = nullptr;
    std::size_t pinned_stage_bytes = 0;
    Tensor device_stage;
    cudaStream_t transfer_stream = nullptr;
    cudaStream_t compute_stream = nullptr;
    cudaEvent_t route_ready = nullptr;
    cudaEvent_t ids_ready = nullptr;
    cudaEvent_t transfer_ready[kQwen4SparseMoePipelineSlots]{};
    cudaEvent_t consumer_complete[kQwen4SparseMoePipelineSlots]{};
};

/**
 * Caller-owned two-stream/two-slot resources for qwen4_sparse_moe_prefill. Each slot has room for
 * 32 exact IQ2_XXS gate/up pairs (the largest admitted routed format) followed by all 10*4096
 * integer occurrence indices. The fixed 54,394,880-byte pinned/device spans bound staging without
 * pinning a source bank. host_scratch is a fixed four-byte-aligned pinned span holding the one
 * selected-id D2H and bounded integer grouping tables, so the Op performs no runtime allocation.
 * It is mutually disjoint from both mapped banks and pinned_stage. Event ownership, fixed
 * compute-stream binding, and transitive cross-call lifetime rules are identical to
 * Qwen4SparseMoePipeline, but reuse is per unique-expert group rather than per rank.
 */
struct Qwen4SparseMoePrefillPipeline {
    void* pinned_stage = nullptr;
    std::size_t pinned_stage_bytes = 0;
    void* host_scratch = nullptr;
    std::size_t host_scratch_bytes = 0;
    Tensor device_stage;
    cudaStream_t transfer_stream = nullptr;
    cudaStream_t compute_stream = nullptr;
    cudaEvent_t route_ready = nullptr;
    cudaEvent_t ids_ready = nullptr;
    cudaEvent_t transfer_ready[kQwen4SparseMoePipelineSlots]{};
    cudaEvent_t consumer_complete[kQwen4SparseMoePipelineSlots]{};
};

/** Caller-owned transient device capacity for the exact C=1/T=1 verifier profile. */
[[nodiscard]] std::size_t qwen4_sparse_moe_workspace_capacity_bytes();

/** Caller-owned transient device capacity for qwen4_sparse_moe_prefill at exact width T. */
[[nodiscard]] std::size_t qwen4_sparse_moe_prefill_workspace_capacity_bytes(
    std::int32_t width);

/** Caller-owned transient device capacity for qwen4_sparse_moe_resident at exact width T. */
[[nodiscard]] std::size_t qwen4_sparse_moe_resident_workspace_capacity_bytes(
    std::int32_t width);

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
 * routed_gate_up is read-only mapped host storage. After the GPU route, its ids cross once to the
 * host. The CPU validates them and copies each selected gate/up pair byte-for-byte into one of two
 * pinned slots; it performs no decode or floating-point work. A distinct transfer stream copies
 * that rank to the paired device slot while the compute stream consumes the preceding rank. The
 * shared expert is queued while the selected ids cross to the host. Explicit ready/complete events
 * protect both slot lifetimes, and routed accumulation remains in rank order. No source bank is
 * pinned, copied in full, or repacked.
 *
 * The production verifier composes ggml_block_linear for every quantized projection. Its private
 * projection and SwiGLU inputs therefore use BF16 storage, while routing, mixture accumulation,
 * and the shared scalar gate use FP32. The independent oracle exact-decodes every selected block
 * and evaluates the complete ideal formula naively in FP64 from represented public inputs. Fixed
 * normwise and finite gross criteria qualify the FP32/BF16 production profile without reproducing
 * its private staging boundaries in the oracle.
 *
 * All device operands, output, route outputs, pipeline device stage, and workspace are pairwise
 * disjoint. The mapped banks and pipeline pinned stage are mutually disjoint. The caller owns both
 * streams, all six events, every view, and every lifetime. It must keep them alive and avoid stage
 * reuse until compute stream completion; that completion transitively covers the transfer stream.
 * The call performs one host event wait for selected ids plus one transfer-completion wait before
 * each rank-two-through-nine host-slot reuse, and is not CUDA-Graph capturable. This verifier Op
 * owns no scheduling, target, registry, or Engine path.
 */
void qwen4_sparse_moe(const Tensor& x, const Qwen4SparseMoeWeights& weights,
                      Qwen4SparseMoePipeline& pipeline,
                      Tensor& selected_ids, Tensor& selected_weights, Tensor& destination,
                      WorkspaceArena& workspace, cudaStream_t stream);

/**
 * T-wide staged sparse-MoE for C=1, T in [1,4096]. x/destination are BF16 [2560,T], and route
 * outputs are selected_ids I32 [10,T] and selected_weights FP32 [10,T], rank-fastest. Routing,
 * softmax/renormalization, every decode/projection/SwiGLU, shared-expert work, and final rank-order
 * FP32 accumulation execute on the GPU with the same represented BF16 boundaries as the scalar
 * Op. The CPU receives all 10*T ids in one D2H, performs only validation and integer grouping,
 * and copies every unique routed gate/up pair exactly once in ascending expert-id groups of at
 * most 32. Two fixed slots overlap group transfer and consumption. An occurrence can reference a
 * staged expert from any token/rank, and duplicate experts across tokens do not duplicate source
 * bytes. Source banks are never decoded, repacked, or fully copied on the CPU.
 *
 * All device operands, output, route outputs, device stage, and workspace are pairwise disjoint;
 * mapped banks and pinned stage are mutually disjoint. The caller owns both streams, all events,
 * every view, and every lifetime through compute-stream completion. This host-grouped route is
 * not CUDA-Graph capturable. The established qwen4_sparse_moe C=1/T=1 entry point and dispatch are
 * unchanged.
 */
void qwen4_sparse_moe_prefill(const Tensor& x, const Qwen4SparseMoeWeights& weights,
                              Qwen4SparseMoePrefillPipeline& pipeline,
                              Tensor& selected_ids, Tensor& selected_weights,
                              Tensor& destination, WorkspaceArena& workspace,
                              cudaStream_t stream);

/**
 * Op: qwen4_sparse_moe_resident
 *
 * Computes the same complete formula, observable route outputs, represented-format decoding,
 * BF16 projection/SwiGLU seams, rank-ordered FP32 routed accumulation, shared branch, and BF16
 * Store result as qwen4_sparse_moe independently for every token. x/destination are contiguous
 * BF16 [2560,T], selected_ids are I32 [10,T], and selected_weights are FP32 [10,T], with
 * rank-fastest route storage and T in [1,4096]. The routed gate/up banks are complete
 * device-resident rank-three IQ1_S or IQ2_XXS weights [512,640,2560]; routed down is a complete
 * device-resident IQ4_NL weight [512,2560,640]. Same-expert selections across tokens remain
 * independent occurrences. GPU-produced selected_ids dynamically index all three banks. No
 * selected id or weight byte crosses to the host, and the Op performs no copy, host
 * synchronization, event operation, allocation, or runtime repack.
 *
 * All inputs and weights are read-only. selected_ids, selected_weights, destination, and workspace
 * are pairwise disjoint from every input and weight. The width-specific workspace query is the
 * exact caller-owned capacity contract. T=1 retains the scalar fused implementation. Private
 * small-T dispatch repeats that implementation, while sufficiently wide calls use device-only
 * route grouping and exact-format expert aggregation; grouping order is private because every
 * occurrence writes its unique rank/token slot before the observable rank-order merge. The same
 * independent complete FP64 oracle and per-token output criteria used by
 * qwen4_sparse_moe qualify this implementation. Work is enqueued on stream; the caller owns every
 * device allocation and must retain it through stream completion. This eager verifier
 * implementation is not CUDA-Graph qualified.
 */
void qwen4_sparse_moe_resident(const Tensor& x,
                               const Qwen4ResidentSparseMoeWeights& weights,
                               Tensor& selected_ids, Tensor& selected_weights,
                               Tensor& destination, WorkspaceArena& workspace,
                               cudaStream_t stream);

} // namespace ninfer::ops
