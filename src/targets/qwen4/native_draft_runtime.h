#pragma once

#include "targets/qwen4/native_artifact.h"
#include "targets/qwen4/native_state.h"
#include "ninfer/types.h"

namespace ninfer::targets::qwen4 {

struct NativeRuntimeConfig;
struct NativeMtpInputRow;
struct NativeMtpSeedView {Tensor carry,target,selected_ids,selected_count;};

// The one selected native speculative backend. Owns one shared exclusive page pool, compact
// batched compute scratch, fixed controls/graphs, and per-slot committed/provisional seed state.
// The complete NativeModelView and stream outlive this owner. No target mathematics, sampler,
// publication policy, CPU floating-point work, or ordinary-weight streaming is performed here.
class NativeDraftRuntime {
public:
    NativeDraftRuntime(const NativeModelView&,const NativeRuntimeConfig&,DeviceContext&);
    ~NativeDraftRuntime();
    NativeDraftRuntime(const NativeDraftRuntime&)=delete;
    NativeDraftRuntime& operator=(const NativeDraftRuntime&)=delete;
    static std::uint64_t device_bytes(const NativeModelView&,const NativeRuntimeConfig&);
    static std::uint64_t device_bytes(const NativeArtifactPlan&,const NativeRuntimeConfig&);
    static std::uint64_t pinned_bytes(const NativeRuntimeConfig&);
    // Conservative CUDA-driver allocation allowance, separate from exact owned buffers.
    static std::uint64_t graph_allowance(const NativeRuntimeConfig&);
    [[nodiscard]] std::uint64_t observed_graph_bytes() const noexcept;
    // Borrowed represented-state diagnostics, not an alternate state owner or mutation API.
    [[nodiscard]] ops::QsaPagedStateView mtp_state() const;
    [[nodiscard]] NativeMtpSeedView mtp_seed() const;
    [[nodiscard]] PagedKVBatchLayerView dflash_cache(int layer) const;

    bool can_reserve(int slot,int maximum,std::span<const int> evictions={}) const;
    void reserve(int slot,int maximum);
    void release_reservation(int slot);
    void reset(int slot);
    int frontier(int slot) const;
    void retain(int slot,NativeCheckpoint kind=NativeCheckpoint::Retained);
    void restore(int slot,NativeCheckpoint kind=NativeCheckpoint::Retained);
    void evict_retained(int slot,NativeCheckpoint kind=NativeCheckpoint::Retained);
    bool has_retained(int slot,NativeCheckpoint kind=NativeCheckpoint::Retained) const;

    Tensor extend_mtp(std::span<const NativeMtpInputRow>,int envelope);
    Tensor draft_mtp(std::span<const int> slots,std::span<const TokenId> preceding_proposals,
                     std::span<const std::array<int,3>> source_positions);
    void discard_mtp(int slot);
    // Uses independently saved last TARGET hidden, not last private carried hidden. Replaces
    // precisely the last private seed at its original position after target-prefix reuse.
    Tensor reseed_mtp(std::span<const int> slots,std::span<const TokenId> next_tokens,
                      std::span<const std::array<int,3>> source_positions);
    void append_dflash(int slot,const Tensor& accepted_features,
                       std::span<const std::int32_t> source_positions);
    Tensor draft_dflash(std::span<const int> slots,std::span<const TokenId> anchors,
                        std::span<const std::int32_t> anchor_positions,int draft_width);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::targets::qwen4
