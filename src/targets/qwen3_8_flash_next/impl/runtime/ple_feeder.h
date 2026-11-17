#pragma once

// Host-side supply of the n-gram (PLE) table rows each forward chunk consumes.

#include "config.h"
#include "ngram_table.h"
#include "runtime/cuda_event.h"

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t, cudaEvent_t

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <future>
#include <mutex>
#include <span>
#include <stop_token>
#include <thread>
#include <vector>

namespace ninfer::targets::qwen3_8_flash_next {

/**
 * Gathers the int8_row records of a chunk's tokens from the NgramTable into pinned staging memory
 * and uploads them as the U8 [162,16,T] rows tensor ple_decode_rows reads, for `lanes`
 * independent requests whose n-gram histories (the last two raw tokens of each) it keeps.
 *
 * Two staging buffers alternate: begin() starts the gather of one lane's next chunk on a worker
 * thread, which first waits for the earlier upload out of the same buffer to complete, so a
 * prefill gathers chunk k+1 while the host enqueues and the GPU runs chunk k. The worker gathers
 * the chunk in parts of part_tokens tokens and publishes each part as it completes; upload(part)
 * waits for that part only and enqueues its H2D copy, so the device can consume early parts while
 * later ones are still being read. A lane's history advances with every gathered token in call
 * order, so a lane's chunks must be begun in sequence order.
 *
 * Protocol: begin(), then upload() of parts 0, 1, ... in order; the gather ends with the last
 * part's upload, after which begin() may start the next one. cancel() abandons a pending gather
 * and restores its lane's history to what it was at begin(), so a prefetched chunk that is not
 * consumed (another lane's prefill runs first, or the lane closes) leaves no trace; the worker
 * stops after the part it is reading. gather_now() may run while a gather of another lane is
 * pending: it first waits for the worker to finish reading (the table serves one thread at a
 * time), then reads on the calling thread. cancel(), reset() of the pending gather's lane and
 * destruction wait for the worker. The table must outlive the feeder. A failed
 * gather rethrows from upload(); its lane's history is then unspecified and the lane must restart
 * with reset(). Anchor fetches use a persistent worker and immutable history snapshots; every table
 * reader joins any earlier worker before reading, so the shared row cache has one reader at a
 * time. Gathers reuse matching anchor records without committing their speculative histories.
 * Not thread-safe; one owner thread.
 */
class PleRowFeeder {
public:
    // max_tokens: the widest chunk, at least 1; lanes: the requests, at least 1. Allocates
    // 2 * max_tokens * 2592 pinned bytes and starts one persistent CPU worker for anchor reads.
    PleRowFeeder(NgramTable& table, std::int32_t max_tokens, std::int32_t lanes);
    ~PleRowFeeder();

    PleRowFeeder(const PleRowFeeder&)            = delete;
    PleRowFeeder& operator=(const PleRowFeeder&) = delete;

    // Starts a new request on `lane`: cancels the lane's pending gather, if any, and resets its
    // history to two missing predecessors. Throws std::out_of_range for an invalid lane.
    void reset(std::int32_t lane);
    // Rebinds the raw history at an exact retained prefix. Cancels this lane's speculative gather
    // before replacing its last two tokens; other lanes' prefetches are unchanged.
    void restore_history(std::int32_t lane, std::span<const std::int32_t> prefix);

    // Starts gathering the rows of `tokens` (copied; 1 <= size <= max_tokens) for `lane` in parts
    // of part_tokens >= 1 tokens (the last part may be shorter). Throws std::logic_error when a
    // gather is already pending, std::invalid_argument on a bad size or lane.
    void begin(std::int32_t lane, std::span<const std::int32_t> tokens, std::int32_t part_tokens);

    // Whether a gather of exactly `tokens` for `lane` is pending with no part uploaded yet: a
    // prefetch the lane's next chunk can consume.
    [[nodiscard]] bool pending_matches(std::int32_t lane,
                                       std::span<const std::int32_t> tokens) const noexcept;

    // The lane of the pending gather, or -1.
    [[nodiscard]] std::int32_t pending_lane() const noexcept;

    // Abandons the pending gather, if any: waits for its worker and restores its lane's history to
    // the value at begin(). Copies already enqueued by upload() still complete.
    void cancel() noexcept;

    // Waits for part `part` (the next one in order) of the pending gather and enqueues the copy
    // of its rows into `rows`, a contiguous device U8 tensor of exactly 2592 * W bytes for the
    // part's W tokens, on `stream`. The staging buffer is reused only after every copy out of it
    // completes. A failed gather rethrows here and ends the gather.
    void upload(Tensor& rows, std::int32_t part, cudaStream_t stream);

    // One decode round: gathers the records of tokens[i] after the history of lanes[i] (distinct
    // lanes, none of them the pending gather's) into `out` (exactly 2592 bytes per token) on the
    // calling thread (reusing matching prefetched anchors) and advances those histories.
    // Throws std::invalid_argument for repeated lanes or a size mismatch,
    // std::out_of_range for a lane outside [0, lanes), and
    // std::logic_error for the pending gather's lane.
    void gather_now(std::span<const std::int32_t> lanes, std::span<const std::int32_t> tokens,
                    std::span<std::uint8_t> out);

    // Gathers request-major chains [width,lanes.size()] using independent copies of each lane's
    // history. width is in [1,8]. Preserves committed histories, including on failure; distinct
    // lanes and no pending gather of a named lane are required, as for gather_now().
    void gather_proposals(std::span<const std::int32_t> lanes, std::span<const std::int32_t> tokens,
                          std::int32_t width, std::span<std::uint8_t> out);

    // Fetches the next input token of each distinct lane asynchronously from a snapshot of its
    // committed raw history. Does not advance histories. Later gathers reuse an entry only when
    // lane, token and both predecessors match; reset/restore joins the worker and drops the
    // lane's entry. Serializes with prefill reads and any earlier anchor fetch. I/O failures
    // belong to the fetched entries and rethrow only when a gather consumes a matching entry;
    // discarding an entry on reset/restore cannot prevent lane release.
    void prefetch_anchors(std::span<const std::int32_t> lanes,
                          std::span<const std::int32_t> tokens);

    // Advances only the committed prefix of each gathered proposal row, without table I/O.
    // Uses the same packed geometry and distinct-lane preconditions as gather_proposals().
    void commit_proposals(std::span<const std::int32_t> lanes, std::span<const std::int32_t> tokens,
                          std::int32_t width, std::span<const std::int32_t> counts);

private:
    struct Staging {
        // The event is polled only by cudaEventSynchronize on the worker; no timing.
        explicit Staging(std::size_t capacity_bytes)
            : bytes(capacity_bytes), uploaded(cudaEventDisableTiming | cudaEventBlockingSync) {}

        PinnedHostBuffer bytes;
        CudaEvent uploaded;
        std::vector<std::int32_t> tokens;
        std::int32_t part_tokens = 0;
        // One promise per part, fulfilled by the worker when the part's rows are staged.
        std::vector<std::promise<void>> parts;
        std::vector<std::future<void>> ready;
    };

    struct AnchorRows {
        std::int32_t lane  = -1;
        std::int32_t token = -1;
        NgramHistory history;
        std::array<std::uint8_t, kNgramTokenBytes> bytes{};
        std::exception_ptr failure{};
    };

    [[nodiscard]] NgramHistory& history(std::int32_t lane);
    void wait_pending() noexcept;
    void finish_anchors();
    void anchor_worker(const std::stop_token& stop);

    NgramTable& table_;
    std::int32_t max_tokens_;
    std::vector<NgramHistory> histories_;
    std::array<Staging, 2> staging_;
    std::size_t next_           = 0;
    std::size_t pending_buffer_ = 0;
    std::int32_t next_part_     = 0;
    // The pending gather's lane and that lane's history at begin().
    std::int32_t pending_lane_ = -1;
    NgramHistory pending_start_;
    // Set by cancel(); the worker checks it before each part and stops early.
    std::atomic<bool> cancel_requested_{false};
    std::future<void> pending_;
    std::vector<AnchorRows> anchors_;
    // Compact table output for the uncached columns, scattered into the pinned proposal frame.
    std::vector<std::uint8_t> proposal_rows_;
    enum class AnchorPhase : std::uint8_t { Idle, Reading, Ready };
    std::mutex anchor_mutex_;
    std::condition_variable_any anchor_changed_;
    std::array<AnchorRows, kMaxLanes> anchor_batch_;
    std::size_t anchor_count_ = 0;
    AnchorPhase anchor_phase_ = AnchorPhase::Idle;
    // Last member: joined before the synchronization objects and staging it uses are destroyed.
    std::jthread anchor_worker_;
};

} // namespace ninfer::targets::qwen3_8_flash_next
