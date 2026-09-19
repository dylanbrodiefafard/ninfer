#pragma once

#include "artifact/materializer.h"
#include "core/paged_kv_cache.h"
#include "ninfer/ops/kv_cache_append_prefix.h"

#include <array>
#include <filesystem>
#include <memory>

namespace ninfer::targets::qwen4 {

struct DFlashLayerWeights {
    Tensor input_norm, post_norm, query_norm, key_norm;
    Weight query, key, value, output, gate, up, down;
};
struct DFlashWeights {
    Weight feature;
    Tensor hidden_norm, final_norm;
    std::array<DFlashLayerWeights, 5> layers;
};

// Exact PixelML preview companion, not a registered Engine target or a DFlash2 head.
// Both source BF16 and locally converted NVFP4 matrices execute with A16 activations.
class LoadedDFlash {
public:
    static std::unique_ptr<LoadedDFlash> load(const std::filesystem::path&, DeviceContext&);
    const DFlashWeights& weights() const noexcept { return weights_; }
private:
    artifact::MaterializedArtifact backing_;
    DFlashWeights weights_;
};

enum class DFlashBoundary {
    FeatureProjection, FusedContext, ContextKeyProjection, ContextValue, ContextKeyNorm,
    ContextKey, InputNorm, QueryProjection, KeyProjection, Value, QueryNorm, KeyNorm,
    Query, Key, Attention, AttentionProjection, AttentionResidual, PostNorm, Gate, Up,
    Activated, Down, LayerOutput, FinalOutput
};
struct DFlashTrace {
    void* user = nullptr;
    // A synchronous host callback may enqueue observations on the supplied stream. Views expire
    // when the next Op reuses scratch. The callback must not allocate during graph capture.
    void (*observe)(void*, DFlashBoundary, int layer, const Tensor&, cudaStream_t) = nullptr;
};

// Startup-fixed, GPU-resident five-layer drafter over caller-owned accepted-context pages.
// Feature columns concatenate attention-GR block inputs at layers [4,16,24,36,44], in that order.
// Cache K/V are BF16 and contain only accepted target-context features. Query K/V never enter it.
// Callers own accepted frontiers, table-row selection and sampling/publication transactions; every
// row must select a distinct slot in [0,slots). Cache positions and RoPE coordinates are distinct
// inputs. All calls enqueue work without host/device synchronization or device allocation.
// The model, shared target weights, input/control tensors and this owner outlive queued consumers.
class DFlashProgram {
public:
    DFlashProgram(const DFlashWeights&, int context_capacity, int append_width, int slots,
                  const std::array<PagedKVBatchLayerView,5>& caches);
    [[nodiscard]] static std::size_t device_bytes(int context_capacity,int append_width,int slots);

    // Features BF16 [12800,W,B], positions I32 [W,B], counts/slots I32 [B]. Only each selected
    // prefix is committed; rejected suffix columns cannot change cache state. W<=append_width.
    void append_accepted_context(const Tensor& features, const Tensor& rope_positions,
        const Tensor& cache_positions, const Tensor& counts, const Tensor& slots,
        ops::KVCacheAppendPrefixExecutionEnvelope, cudaStream_t, DFlashTrace = {});

    // Embeddings BF16 [2560,K,B] are anchor then K-1 masks; K=1..7. Query zero predicts anchor+1.
    // All live queries see every live query K/V and context [0,context_lengths[b]). This method
    // changes no persistent cache. It returns program-owned BF16 [2560,K,B] final features.
    Tensor draft_embeddings(const Tensor& embeddings, const Tensor& rope_positions,
        const Tensor& context_lengths, const Tensor& valid_columns, const Tensor& slots,
        cudaStream_t, DFlashTrace = {});

    // Exact target-bound convenience schedule: prepare anchor/mask IDs and positions, gather the
    // shared BF16 target embedding, run the drafter, then use the shared target BF16 output head.
    // anchor_positions is I32 [B], independent of logical context lengths (e.g. shifted RoPE).
    // logits is caller-owned BF16 [248320,K,B]; no first query is discarded.
    void draft(const Tensor& anchors, const Tensor& anchor_positions,
        const Tensor& context_lengths, const Tensor& valid_columns,
        const Tensor& slots, const Weight& target_embedding, const Weight& target_head,
        Tensor& logits, cudaStream_t, DFlashTrace = {});

    PagedKVBatchLayerView cache(int layer) const;
private:
    struct Storage;
    DFlashProgram(const DFlashWeights&,int,int,int,const std::array<PagedKVBatchLayerView,5>&,const Storage&);
    const DFlashWeights& weights_;
    int context_capacity_, append_width_, slots_;
    std::array<PagedKVBatchLayerView,5> caches_;
    DeviceBuffer residual_, norm_, projected_, context_, q_raw_, q_, k_raw_, k_, v_,
        attention_, gate_, up_, activated_, ids_, positions_, embeddings_;
    WorkspaceArena workspace_;
};

} // namespace ninfer::targets::qwen4
