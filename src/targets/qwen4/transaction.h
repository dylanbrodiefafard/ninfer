#pragma once

#include "targets/qwen4/native_compute.h"
#include "runtime/contract/types.h"
#include "ninfer/ops/sampling.h"
#include "core/decode_graph.h"

namespace ninfer::targets::qwen4 {
class NativeTransactionGraph;

struct NativeResolution {
    int slot;
    runtime::GeneratedRound licensed;
    runtime::OutputDecision decision;
    ops::SamplingConfig sampling{};
};

// Qualification-only text speculative transaction over the exact first four native layers.
// It does not decide acceptance, apply a sampling law, register a target, or represent 48 layers.
// The caller uses the common sampler/accept Ops, whose provisional lengths/anchors are scratch;
// this owner's committed frontier/anchor remain authoritative. Non-null SamplingConfig counts
// must refer to sampling_counts(slot). Seed/config and output policy remain caller-owned.
// C=1..4, W=1..16; one ordered caller stream, model/stream outlive this owner. All buffers,
// including one retained checkpoint per slot and publication storage, are startup-fixed.
class NativeTransaction {
public:
    NativeTransaction(const LoadedNativeFirstBlock&,int requests,int capacity,int max_width,
                      cudaStream_t stream);
    ~NativeTransaction();
    NativeTransaction(const NativeTransaction&)=delete;
    NativeTransaction& operator=(const NativeTransaction&)=delete;

    void reset(int slot,TokenId anchor,std::array<int,3> anchor_position);
    // verified_tokens=[old anchor,drafts...]. Positions are text continuation coordinates:
    // anchor_position+t in each axis, independently of the logical cache frontier.
    // Preparation stages inputs outside capture. Source device tensors survive queued copies.
    void prepare(int slot,std::span<const TokenId> verified_tokens,
                 std::span<const std::array<int,3>> positions,const Tensor& hidden,
                 const Tensor& ple_embedding);
    // Eager execution. Exactly one enqueue or typed graph launch follows each preparation;
    // no host publication or device allocation occurs in this schedule.
    NativeFirstBlockOutput enqueue(int slot,const NativeBlockTrace* trace=nullptr);
    // Explicit capture owns exact prepared slot/width provenance and graph-stable output views.
    // Capturing does not execute/mark a row ready. Returned graph must not outlive this owner.
    std::unique_ptr<NativeTransactionGraph> capture_graph(std::span<const int> slots);
    // Drains provisional consumers, rolls back common sampler counts for unpublished suffixes,
    // folds exactly decision.accepted_tokens VERIFIED INPUT columns, drains, then publishes.
    // A proper output prefix makes its last licensed token the new anchor. Zero/rejected output
    // preserves every committed owner and logical RNG position. No correction/bonus is folded.
    // Validation precedes mutation. Execution failures poison affected slots until reset/restore.
    void resolve(std::span<const NativeResolution> rows);

    // Boundary-only, synchronous snapshots of all continuation owners and host publication.
    // Snapshot storage is owned, independent of later provisional/scratch writes.
    void retain(int slot);
    void restore(int slot);

    int frontier(int slot) const;
    TokenId anchor(int slot) const;
    std::array<int,3> anchor_position(int slot) const;
    std::span<const TokenId> published(int slot) const;
    NativeFirstBlockState state(int slot) const;
    // Represented pending transition records for independent qualification oracles, W>=2.
    NativeFirstBlockRecords records(int slot) const;
    Tensor raw_token_history(int slot) const;
    Tensor continuation(int slot) const;
    Tensor sampling_counts(int slot) const;
private:
    friend class NativeTransactionGraph;
    struct Slot;
    Slot& slot(int) const;
    NativeFirstBlock program_;
    int requests_,capacity_,max_width_;
    cudaStream_t stream_;
    std::array<std::unique_ptr<Slot>,4> slots_;
};

class NativeTransactionGraph {
public:
    ~NativeTransactionGraph();
    NativeTransactionGraph(const NativeTransactionGraph&)=delete;
    NativeTransactionGraph& operator=(const NativeTransactionGraph&)=delete;
    // Validates every captured row is newly prepared at its captured width. A successful
    // launch is the only graph path that marks records available for positive resolution.
    void launch();
private:
    friend class NativeTransaction;
    NativeTransactionGraph(NativeTransaction&,std::span<const int>);
    NativeTransaction& owner_;
    std::array<int,4> slots_{},widths_{};
    std::array<Tensor,4> outputs_{};
    int rows_=0;
    DecodeGraphDefinition definition_;
    DecodeGraphExecutable executable_;
};
} // namespace ninfer::targets::qwen4
