#pragma once

// Persistent device state of the Qwen3.8-Flash-Next lanes (concurrent requests): per lane the 36
// GDN convolution and recurrent images, the PLE convolution history, the sampling token counts
// and an eligibility bitset, and the 12 QSA layers' state in one paged pool shared by every lane.
// The n-gram token histories live with the row feeder (ple_feeder.h), since only the host reads
// them.

#include "config.h"
#include "runtime/speculative_policy.h"

#include "core/arena.h"
#include "core/cyclic_kv_cache.h"
#include "core/linear_attention_state.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"
#include "ninfer/ops/qsa.h"
#include "ninfer/ops/speculative_round.h"

#include <cuda_runtime.h> // cudaStream_t

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace ninfer::targets::qwen3_8_flash_next {

/**
 * The recurrent and attention state of `lanes` lanes, in one device allocation. Lane l owns slot l
 * of every per-lane component:
 *
 *   gdn          LinearAttentionStatePool of `lanes` slots: per GDN layer the BF16 [10240,3,lanes]
 *                q|k|v convolution windows (oldest first) and the FP32 [128,128,48,lanes]
 *                recurrent states.
 *   qsa          one PageMajor PagedKVPool whose planes are the 12 QSA layers' state planes in
 *                the chosen format (ops::qsa_paged_plane_specs, layer after layer), so one physical
 *                page id addresses the same 64 slots of every layer, and whose block-table matrix
 *                I32 [lane_capacity / 64, lanes] maps lane l's logical pages in row l. Unmapped
 *                entries hold -1, which the QSA Ops treat as unmapped.
 *   ple_history  BF16 [9,10240,lanes] (ple_inject history slots): the PLE convolution input of
 *                the nine previous tokens per channel, oldest first.
 *   lane_ids     I32 [4,lanes] holding lane l at element 4 l: the I32 [1] selector tensors of
 *                one-lane calls, each 16-byte aligned as the QSA Ops require.
 *   token_counts I32 [248320,lanes]: column l counts the tokens lane l sampled at positive
 *                temperature (ops::sample's penalty counts; rows below kTokenDomain are used).
 *   allowed      U32 [kAllowedTokenWordStride,lanes]: column l is lane l's token-eligibility
 *                root bitset storage (ops::SamplingConfig::allowed_token_words). Forward owns
 *                the separate compact-row, per-hypothesis masks used during verification.
 *
 * Lanes are opened with a page reservation for their whole token bound: open() reserves and maps
 * ceil(reserve / 64) pages from the shared pool, publishes them in the lane's block-table row and
 * zeroes the lane's GDN, PLE and count state on the stream; close() returns them. A lane's
 * `committed` counts the tokens whose effects every state component holds, so it is the next
 * token's position and its QSA slot; the caller advances it after enqueueing the work that wrote
 * those tokens. QSA slots and block keys of reused pages need no clearing: a query reads neither
 * a slot beyond its own position nor the key of a block it does not see whole, and both are
 * written before they become visible.
 *
 * All device work is enqueued on the caller's stream in call order; host bookkeeping is immediate
 * and safe because every later use of a returned page is enqueued after its release on the same
 * stream. Not thread-safe.
 */
struct DflashLaneState {
    std::int32_t pending    = 0;
    float draft_temperature = 0.4F;
    std::array<double, ops::kPLessProposalCalibrationTemperatureCount> calibration_sum{};
    std::array<std::uint64_t, ops::kPLessProposalCalibrationTemperatureCount> calibration_count{};
    SpeculativeRequestPolicy policy;
};

class State {
public:
    // Words of one lane's eligibility bitset column: kAllowedTokenWords rounded up to 256 bytes.
    static constexpr std::int32_t kAllowedTokenWordStride = (kAllowedTokenWords + 63) / 64 * 64;

    // lanes in [1, kMaxLanes]; context_tokens (one lane's longest context) in [1,
    // ops::kQsaMaximumCapacity], rounded up to whole 64-token pages; pool_tokens (the shared QSA
    // pool) at least 1, rounded up to whole pages. Throws std::invalid_argument for invalid
    // arguments and propagates allocation failures; synchronizes the device while initializing.
    State(std::int32_t lanes, std::int32_t context_tokens, std::int64_t pool_tokens,
          ops::QsaKvFormat format, DflashExecution dflash = DflashExecution::Disabled);

    State(const State&)            = delete;
    State& operator=(const State&) = delete;

    [[nodiscard]] std::int32_t lanes() const noexcept { return lanes_; }

    // One lane's longest context (its logical QSA capacity), a page multiple.
    [[nodiscard]] std::int32_t lane_capacity() const noexcept { return lane_capacity_; }

    [[nodiscard]] ops::QsaKvFormat format() const noexcept { return format_; }

    // Tokens of the shared pool and those not reserved by open lanes (whole pages).
    [[nodiscard]] std::int64_t pool_tokens() const noexcept;
    [[nodiscard]] std::int64_t free_tokens() const noexcept;

    // Whether open(reserve_tokens) succeeds now.
    [[nodiscard]] bool can_open(std::int64_t reserve_tokens) const noexcept;
    // Opens the lowest closed lane at position 0 with ceil(reserve_tokens / 64) pages, or returns
    // nullopt when can_open is false.
    [[nodiscard]] std::optional<std::int32_t> open(std::int64_t reserve_tokens,
                                                   cudaStream_t stream);
    // Unmaps and releases an open lane's pages; throws std::out_of_range for a closed lane.
    void close(std::int32_t lane, cudaStream_t stream);

    [[nodiscard]] bool is_open(std::int32_t lane) const noexcept;
    // Accessors of an open lane; throw std::out_of_range for a closed or invalid lane.
    [[nodiscard]] std::int32_t committed(std::int32_t lane) const;
    [[nodiscard]] std::int64_t reserved_tokens(std::int32_t lane) const;
    // Records that `tokens` more tokens are held; throws std::out_of_range beyond the lane's
    // reservation.
    void advance(std::int32_t lane, std::int32_t tokens);

    // Adjusts a lane's exact frontier and page reservation at an execution boundary. Growing
    // preserves existing pages; shrinking releases only complete pages beyond the new bound.
    // The caller restores recurrent/PLE state separately before executing from an older frontier.
    void resize(std::int32_t lane, std::int32_t frontier, std::int64_t reserve_tokens,
                cudaStream_t stream);
    void reset_sampling(std::int32_t lane, cudaStream_t stream);

    // Exact continuation image: all GDN convolution/recurrent slots, PLE history and tail logits.
    // Host buffers must be pinned, at least continuation_bytes(), and live through stream
    // completion.
    [[nodiscard]] std::size_t continuation_bytes() const noexcept;
    void pack_continuation(std::int32_t lane, void* destination, cudaStream_t stream) const;
    void unpack_continuation(std::int32_t lane, const void* source, cudaStream_t stream);
    // Copies independent page payloads and the source's current continuation; both lanes must be
    // open and the destination's reservation and mapped extent must cover frontier. When frontier
    // precedes the source's current position, the caller must replace that continuation with its
    // exact checkpoint before executing. The source remains unchanged.
    void copy_prefix(std::int32_t destination, std::int32_t source, std::int32_t frontier,
                     cudaStream_t stream);
    [[nodiscard]] Tensor tail_logits(std::int32_t lane) const;
    // Raw transfer access for the shared host tiers; the allocation remains lane-owned.
    [[nodiscard]] PagedKVAllocation& pages(std::int32_t lane);

    [[nodiscard]] PagedKVPool& pool() noexcept { return *pool_; }

    // Whole-pool GDN tensors (the snapshot Ops' state operands) and one lane's slot of them.
    [[nodiscard]] LinearAttentionStateAllLayersView gdn_all_layers() const {
        return gdn_.all_layers_view();
    }

    [[nodiscard]] Tensor tail_logits_pool() const noexcept { return tail_logits_; }

    [[nodiscard]] Tensor sampling_counts_pool() const noexcept { return token_counts_; }

    [[nodiscard]] Tensor gdn_convolution_pool(std::int32_t gdn_layer) const;
    [[nodiscard]] Tensor gdn_recurrent_pool(std::int32_t gdn_layer) const;
    [[nodiscard]] Tensor gdn_convolution(std::int32_t gdn_layer, std::int32_t lane) const;
    [[nodiscard]] Tensor gdn_recurrent(std::int32_t gdn_layer, std::int32_t lane) const;
    [[nodiscard]] ops::QsaPagedState& qsa(std::int32_t qsa_layer);

    [[nodiscard]] Tensor& ple_history() noexcept { return ple_history_; }

    // I32 [1] holding `lane`, 16-byte aligned.
    [[nodiscard]] Tensor lane_id(std::int32_t lane) const;
    // Device I32 [248320] counts and U32 [kAllowedTokenWords] bitset storage of a lane.
    [[nodiscard]] std::int32_t* token_counts(std::int32_t lane) const;
    [[nodiscard]] std::uint32_t* allowed_tokens(std::int32_t lane) const;

    [[nodiscard]] CyclicKVCache& dflash_cache() { return *dflash_cache_; }

    [[nodiscard]] const CyclicKVCache& dflash_cache() const { return *dflash_cache_; }

    [[nodiscard]] Tensor dflash_pending_features() const noexcept {
        return dflash_pending_features_;
    }

    [[nodiscard]] std::int32_t dflash_pending_count(std::int32_t lane) const {
        return lane_states_.at(static_cast<std::size_t>(lane)).dflash.pending;
    }

    void set_dflash_pending_count(std::int32_t lane, std::int32_t count);
    [[nodiscard]] DflashLaneState& dflash_lane(std::int32_t lane);

    [[nodiscard]] bool has_dflash() const noexcept { return dflash_cache_ != nullptr; }

    [[nodiscard]] std::size_t device_bytes() const noexcept { return backing_.bytes; }

private:
    struct LaneState {
        bool open              = false;
        std::int32_t committed = 0;
        DflashLaneState dflash;
        std::int64_t reserved_tokens = 0;
        PagedKVAllocation pages;
    };

    [[nodiscard]] const LaneState& open_lane(std::int32_t lane) const;

    std::int32_t lanes_         = 0;
    std::int32_t lane_capacity_ = 0;
    ops::QsaKvFormat format_    = ops::QsaKvFormat::NVFP4G16;
    DeviceBuffer backing_;
    LinearAttentionStatePool gdn_;
    std::unique_ptr<PagedKVPool> pool_;
    std::array<ops::QsaPagedState, kQsaLayers> qsa_{};
    Tensor ple_history_;
    Tensor lane_ids_;
    Tensor token_counts_;
    Tensor allowed_;
    Tensor tail_logits_;
    std::unique_ptr<CyclicKVCache> dflash_cache_;
    Tensor dflash_pending_features_;
    // Declared after pool_ so that every allocation is released before the pool is destroyed.
    std::vector<LaneState> lane_states_;
};

} // namespace ninfer::targets::qwen3_8_flash_next
