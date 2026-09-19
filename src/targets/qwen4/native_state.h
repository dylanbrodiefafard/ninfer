#pragma once

#include "core/gdn_replay_records.h"
#include "core/linear_attention_state.h"
#include "core/paged_kv_cache.h"
#include "ninfer/ops/qsa.h"

#include <array>
#include <memory>
#include <span>

namespace ninfer::targets::qwen4 {

enum class NativeCheckpoint { Retained=0, Prompt=1 };

// Exact full-preview state ownership; no weights, sampling law, prompt identity or CPU math.
// The caller serializes GPU execution and calls publish only after all layer consumers drain.
// Retention is two typed complete logical-prefix checkpoints per slot over its exclusively owned
// allocation, not copied KV, shared active ownership, or arbitrary partial-prefix reuse.
class NativeState {
public:
    struct Config {
        int requests=1, context=4096, kv_tokens=4096, record_width=8;
        ops::QsaKvFormat kv=ops::QsaKvFormat::NVFP4G16;
    };
    NativeState(Config,cudaStream_t);
    ~NativeState();
    static std::size_t device_bytes(Config);
    bool can_reserve(int slot,int maximum,std::span<const int> evictions={}) const;
    void reserve(int slot,int maximum);
    void release_reservation(int slot);
    void materialize(int slot,int maximum);
    // Roll back prepared but unexecuted append ranges, retaining each growth entitlement.
    // No recurrent/history/frontier mutation. Consumers drain before physical pages return.
    void discard_materialization(std::span<const int> slots);
    void reset(int slot);
    int frontier(int slot) const;

    LinearAttentionStatePool& gdn(bool provisional);
    Tensor ple(bool provisional) const;
    Tensor continuation() const;
    Tensor token_history() const;
    ops::QsaPagedStateView qsa(int layer) const;
    // W=2..record_width and B=1..requests; storage is packed for the active width/rows.
    GdnReplayRecords records(int width,int batch) const;
    Tensor ple_records(int width,int batch) const;

    // Publish a previously enqueued exact batch. Counts correspond to compact input rows.
    // Whole-prefix commits copy provisional state. Partial W>=2 commits fold recorded inputs.
    // Zero leaves all committed state unchanged. raw_ids [W,B] preserves raw EOS history.
    // residual [10240,W,B] supplies the last committed pre-final-GR continuation.
    // Every argument and all counts are validated before any mutation or frontier publication.
    void commit(std::span<const int> slots,std::span<const int> lengths,
                std::span<const int> counts,int width,bool recorded,
                const Tensor& raw_ids,const Tensor& residual,
                const Tensor& device_slots,Tensor& device_counts);
    void retain(int slot,NativeCheckpoint kind=NativeCheckpoint::Retained);
    void restore(int slot,NativeCheckpoint kind=NativeCheckpoint::Retained);
    void evict_retained(int slot,NativeCheckpoint kind=NativeCheckpoint::Retained);
    bool has_retained(int slot,NativeCheckpoint kind=NativeCheckpoint::Retained) const;
    std::uint32_t entitled_pages() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace ninfer::targets::qwen4
