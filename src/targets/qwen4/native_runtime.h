#pragma once

#include "targets/qwen4/native_artifact.h"
#include "targets/qwen4/native_state.h"
#include "ninfer/types.h"

#include <array>
#include <cstdint>
#include <memory>
#include <span>

namespace ninfer::targets::qwen4 {

inline constexpr int kNativeTokenDomain = 248077;
inline constexpr int kNativeHeadRows = 248320;

// Startup-fixed exact preview geometry; these are execution capacities, never model topology.
// Every tensor matrix in NativeModelView is GPU resident, except the fully locked PLE table.
struct NativeRuntimeConfig {
    int requests = 1;
    int context_tokens = 4096;
    int kv_tokens = 4096;
    int prefill_width = 128;
    int verify_width = 8;
    int vision_patches = 65536;
    int vision_segments = 384;
    int vision_tokens_per_request = 32768;
    KvCacheStorage kv_dtype = KvCacheStorage::Nvfp4;
    bool vision = true;
    bool mtp = false;
    bool dflash = false;
    bool use_cuda_graph = true;
};

// One represented input prefix for an occupied request slot. Padding to the batch envelope is
// owner-private; only token_ids.size() rows are visible or committed. Positions are the original
// source T/H/W coordinates, independent of logical cache frontier. Host inputs survive prepare.
struct NativeInputRow {
    int slot = 0;
    std::span<const TokenId> token_ids;
    std::span<const std::array<int,3>> positions;
};
struct NativeBatchOutput {
    // All views are contiguous and owner-borrowed until the next prepare/enqueue. Compact row
    // order matches the input span; width is the fixed envelope, lengths names each live prefix.
    Tensor logits;          // BF16 [248320,W,B]; sampler must restrict domain to248077.
    Tensor carried_hidden;  // BF16 [10240,W,B], pre-final-GR, for source-aligned MTP.
    Tensor dflash_features; // BF16 [12800,W,B], attention-GR input taps [4,16,24,36,44].
    Tensor lengths;         // I32 [B], valid input widths.
};
struct NativeCommitRow { int slot = 0; int input_count = 0; };
struct NativeMtpInputRow {
    int slot = 0;
    Tensor target_hidden; // BF16 [10240,T], actual preceding target carried state.
    std::span<const TokenId> next_tokens;
    std::span<const std::array<int,3>> source_positions;
};

// Exact full48-layer target runtime. This owns only GPU mathematics/state, input staging and
// exact private draft components. Engine Program owns request/prompt identity, sampling law,
// token occurrence counts, published anchors, output policy and common scheduler interaction.
// One ordered DeviceContext stream; model outlives runtime and all consumers drain before
// destruction. All GPU/pinned workspaces, state images and graph addresses are startup-fixed.
// prepare uses qualified integer-only host hashing and bounded host PLE row copies; no CPU model math,
// inference-time table I/O, ordinary-weight streaming or device allocation is permitted.
class NativeRuntime {
public:
    NativeRuntime(const NativeModelView&, const NativeRuntimeConfig&, DeviceContext&);
    ~NativeRuntime();
    NativeRuntime(const NativeRuntime&) = delete;
    NativeRuntime& operator=(const NativeRuntime&) = delete;

    // Exact owned allocations plus an explicitly conservative CUDA-driver graph allowance,
    // including retained images, private backend caches and active-request capacity.
    // Excludes already-admitted model bytes; startup checks observed graph use against allowance.
    [[nodiscard]] static std::uint64_t device_bytes(const NativeModelView&, const NativeRuntimeConfig&);
    [[nodiscard]] static std::uint64_t device_bytes(const NativeArtifactPlan&, const NativeRuntimeConfig&);
    [[nodiscard]] static std::uint64_t pinned_bytes(const NativeRuntimeConfig&);
    [[nodiscard]] MemorySummary memory_summary() const;
    void reset_memory_peaks();

    void reset(int slot);
    [[nodiscard]] int frontier(int slot) const;
    // Exact exclusive P64 page entitlements, including active growth guarantees. Trial
    // eviction is non-mutating; reserve requires those retained owners already evicted and
    // rechecks atomically. The total configured pool is shared by all request slots.
    [[nodiscard]] bool can_reserve(int slot, int maximum_frontier,
        std::span<const int> evict_retained_slots = {}) const;
    [[nodiscard]] bool reserve(int slot, int maximum_frontier);
    void release_reservation(int slot);
    // prepare validates complete batch before mutation. At most one pending batch exists;
    // each input slot is unique, B=1..4, with 1..envelope live rows. B1 prefill width<=4096;
    // B2..4 width<=16; recorded verification width<=16. Constructor validates these bounds.
    void prepare(std::span<const NativeInputRow>, int envelope);
    [[nodiscard]] NativeBatchOutput enqueue(bool record_prefix);
    // Fold exactly verified INPUT columns, including old anchor, never an extra bonus token.
    // Zero discards all provisional effects. Every prepared row must be resolved once.
    // GPU consumer drain precedes host frontier publication; failure leaves no reusable slot.
    void commit(std::span<const NativeCommitRow>);
    void discard();
    void synchronize();

    // One complete retained logical prefix per slot, independent of later scratch/pending
    // writes. QSA page allocation has one exclusive owner, transferred on retain/restore;
    // there are no shared active owners or page refcounts. Stale tail bytes beyond the
    // retained frontier are unobservable and need not be preserved.
    // Includes GDN FP32+conv, QSA/cache/frontier/index/positions, PLE conv+raw history, carry,
    // Vision placement, MTP seed/provisional state and DFlash accepted-context state.
    void retain(int slot, NativeCheckpoint kind = NativeCheckpoint::Retained);
    void restore(int slot, NativeCheckpoint kind = NativeCheckpoint::Retained);
    void evict_retained(int slot, NativeCheckpoint kind = NativeCheckpoint::Retained);
    [[nodiscard]] bool has_retained(int slot, NativeCheckpoint kind = NativeCheckpoint::Retained) const;

    // Encode one or more complete media items, and retain their merged source-order features
    // at the requested absolute prompt columns. No raw-token/PLE replacement occurs: visual
    // features replace the embedding then broadcast to all four GR branches at those columns.
    void prepare_vision(int slot, std::span<const float> patches, std::span<const VisionGrid> grids,
                        std::span<const std::int32_t> prompt_columns);
    void clear_vision(int slot);

    // Source-aligned private MTP: R_t + embedding(next token) at original position t.
    // extend/draft return full shared-head logits; all private matrices stay resident/A16.
    // Caller slices accepted target rows, and discards drafts before target replay.
    [[nodiscard]] Tensor extend_mtp(std::span<const NativeMtpInputRow>, int envelope);
    [[nodiscard]] Tensor draft_mtp(std::span<const int> slots,
        std::span<const TokenId> preceding_proposals,
        std::span<const std::array<int,3>> source_positions); // BF16 [248320,1,B].
    void discard_mtp(int slot);
    // Replace the last seed after exact target-prefix reuse chooses a different anchor.
    // Truncate only the private last position, then extend saved target R_(frontier-1)
    // with the new anchor at that same original source position; no target state changes.
    [[nodiscard]] Tensor reseed_mtp(std::span<const int> slots,
        std::span<const TokenId> next_tokens,
        std::span<const std::array<int,3>> source_positions);

    // Only accepted target feature rows enter DFlash context. Noise never mutates that cache.
    void append_dflash(int slot, const Tensor& accepted_features,
                       std::span<const std::int32_t> source_positions);
    [[nodiscard]] Tensor draft_dflash(std::span<const int> slots,
        std::span<const TokenId> anchors, std::span<const std::int32_t> anchor_positions,
        int draft_width); // BF16 [248320,K,B], query0 included; anchor+mask source embeddings.
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::targets::qwen4
