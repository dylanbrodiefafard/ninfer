#pragma once

#include "artifact/materializer.h"
#include "targets/qwen4/native_artifact.h"
#include "core/gdn_replay_records.h"
#include "ninfer/ops/gated_delta_net_layer.h"
#include "ninfer/ops/qsa.h"
#include "ninfer/ops/qwen4_sparse_moe.h"

#include <array>
#include <filesystem>
#include <memory>
#include <span>

namespace ninfer::targets::qwen4 {

// Exact first four source layers, not a configurable model or registered Engine identity.
// All execution preparation is offline. Loading binds/uploads represented artifact bytes only.
class LoadedNativeFirstBlock {
public:
    static std::unique_ptr<LoadedNativeFirstBlock> load(const std::filesystem::path& prepared_root,
        const std::filesystem::path& ple_component,DeviceContext&);
    const std::array<NativeLayerWeights,4>& layers() const noexcept { return layers_; }
    const NativePleWeights& ple() const noexcept { return ple_; }
private:
    std::array<artifact::MaterializedArtifact,5> backing_;
    std::array<NativeLayerWeights,4> layers_;
    NativePleWeights ple_;
};

struct NativeGdnState { Tensor conv,recurrence; };
struct NativeFirstBlockState {
    std::array<NativeGdnState,3> gdn;
    Tensor ple_conv;
    ops::QsaStateView qsa;
};
struct NativeFirstBlockRecords { std::array<GdnReplayRecordLayer,3> gdn; Tensor ple_conv; };
// Borrowed BF16 four-stream states, not attention-GR block inputs for DFlash feature taps.
// hidden is invalidated by this slot's next prepare; layer_hidden by its next enqueue.
struct NativeFirstBlockOutput { Tensor hidden; std::array<Tensor,4> layer_hidden; };
enum class NativeBlockTraceStage { Ple,AttentionRead,AttentionWrite,Mixer,AttentionInjected,
                                  MoeRead,MoeWrite,MoeIds,MoeWeights,Moe,MoeInjected };
struct NativeBlockTrace {
    void* context=nullptr;
    void (*capture)(void*,int,NativeBlockTraceStage,const Tensor&,cudaStream_t)=nullptr;
};

// GPU-resident bounded qualification schedule for actual layers 0,1,2,3 only.
// One ordered stream, startup-fixed C=1..4 slots, BF16 QSA KV, protected BF16 matrices,
// source NVFP4 routed banks and A16Only activations. PLE table/hash ownership is external;
// prepare receives already-decoded represented PLE embeddings and four-stream input hidden.
// No floating-point CPU execution, runtime repacking, or pageable copy occurs in enqueue.
class NativeFirstBlock {
public:
    NativeFirstBlock(const LoadedNativeFirstBlock&,int requests,int capacity,int max_width);
    ~NativeFirstBlock();
    // Host/pageable metadata and D2D input staging occur outside graph capture. Source inputs
    // and positions outlive their queued copies. Positions are independent of logical frontier.
    void prepare(int slot,std::span<const std::array<int,3>> positions,const Tensor& hidden,
                 const Tensor& ple_embedding,cudaStream_t);
    // Reads committed GDN/PLE state, writes disjoint provisional state and QSA append-only tail.
    // Optional records support widths2..16. Fixed-width enqueue may be CUDA-graph captured.
    NativeFirstBlockOutput enqueue(int slot,int width,bool record,cudaStream_t,
                                   const NativeBlockTrace* =nullptr);
    NativeFirstBlockState state(int slot,bool provisional) const;
    NativeFirstBlockRecords records(int slot,int width) const;
    // Full-width baseline commit. Partial acceptance is owned by the caller's transaction;
    // fold state first, then publish_frontier. Rejected QSA suffix rows remain logically invisible.
    void commit_all(int slot,cudaStream_t);
    void publish_frontier(int slot,int accepted);
    void discard(int slot);
    void reset(int slot,cudaStream_t);
    int frontier(int slot) const;
    int requests() const noexcept { return requests_; }
    int capacity() const noexcept { return capacity_; }
    int max_width() const noexcept { return max_width_; }
    // Snapshot owner restores all committed state first on the ordered stream, then restores
    // this logical frontier. No pending transaction may exist; QSA tail stays invisible.
    void restore_frontier(int slot,int frontier);
private:
    struct Slot;
    Slot& slot(int) const;
    const LoadedNativeFirstBlock& model_;
    int requests_,capacity_,max_width_;
    std::array<std::unique_ptr<Slot>,4> slots_;
    DeviceBuffer mixed_,block_,scale_,routes_,route_weights_,qsa_workspace_;
    WorkspaceArena workspace_;
};
} // namespace ninfer::targets::qwen4
