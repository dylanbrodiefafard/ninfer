#include "runtime/ple_feeder.h"

#include "config.h"

#include "core/device.h"
#include "ninfer/ops/ple.h"

#include <algorithm>
#include <array>
#include <exception>
#include <stdexcept>
#include <string>

namespace ninfer::targets::qwen3_8_flash_next {

static_assert(kNgramHeads == ops::kPleRowsPerToken && kNgramRowWidth == ops::kPleRowWidth &&
              kNgramRecordBytes == ops::kPleInt8RowBytes);

namespace {

std::size_t staging_bytes(std::int32_t max_tokens) {
    if (max_tokens < 1) {
        throw std::invalid_argument("PleRowFeeder: max_tokens must be positive");
    }
    return static_cast<std::size_t>(max_tokens) * static_cast<std::size_t>(kNgramTokenBytes);
}

std::size_t lane_count(std::int32_t lanes) {
    if (lanes < 1) { throw std::invalid_argument("PleRowFeeder: lanes must be positive"); }
    return static_cast<std::size_t>(lanes);
}

} // namespace

PleRowFeeder::PleRowFeeder(NgramTable& table, std::int32_t max_tokens, std::int32_t lanes)
    : table_(table), max_tokens_(max_tokens), histories_(lane_count(lanes)),
      staging_{Staging(staging_bytes(max_tokens)), Staging(staging_bytes(max_tokens))},
      anchors_(lane_count(lanes)),
      anchor_worker_([this](const std::stop_token& stop) { anchor_worker(stop); }) {
    for (Staging& staging : staging_) {
        staging.tokens.reserve(static_cast<std::size_t>(max_tokens));
    }
}

PleRowFeeder::~PleRowFeeder() {
    wait_pending();
    anchor_worker_.request_stop();
    anchor_changed_.notify_all();
}

void PleRowFeeder::wait_pending() noexcept {
    if (pending_.valid()) { pending_.wait(); }
}

NgramHistory& PleRowFeeder::history(std::int32_t lane) {
    if (lane < 0 || static_cast<std::size_t>(lane) >= histories_.size()) {
        throw std::out_of_range("PleRowFeeder: lane " + std::to_string(lane) + " is out of range");
    }
    return histories_[static_cast<std::size_t>(lane)];
}

void PleRowFeeder::reset(std::int32_t lane) {
    NgramHistory& lane_history = history(lane);
    if (pending_lane_ == lane) { cancel(); }
    finish_anchors();
    anchors_.at(static_cast<std::size_t>(lane)) = AnchorRows{};
    lane_history                                = NgramHistory{};
}

void PleRowFeeder::restore_history(std::int32_t lane, std::span<const std::int32_t> prefix) {
    reset(lane);
    NgramHistory& restored = history(lane);
    if (!prefix.empty()) { restored.tokens[1] = prefix.back(); }
    if (prefix.size() >= 2) { restored.tokens[0] = prefix[prefix.size() - 2]; }
}

void PleRowFeeder::cancel() noexcept {
    if (pending_lane_ < 0) { return; }
    // The worker stops after the part it is reading; its failure, if any, is dropped with the
    // gather, and parts it never staged are never waited on.
    cancel_requested_.store(true, std::memory_order_relaxed);
    wait_pending();
    pending_                                            = {};
    histories_[static_cast<std::size_t>(pending_lane_)] = pending_start_;
    pending_lane_                                       = -1;
}

std::int32_t PleRowFeeder::pending_lane() const noexcept { return pending_lane_; }

bool PleRowFeeder::pending_matches(std::int32_t lane,
                                   std::span<const std::int32_t> tokens) const noexcept {
    if (pending_lane_ != lane || next_part_ != 0) { return false; }
    // The worker only reads the staged token list, so comparing it while the gather runs is safe.
    const std::vector<std::int32_t>& staged = staging_.at(pending_buffer_).tokens;
    return std::ranges::equal(staged, tokens);
}

void PleRowFeeder::begin(std::int32_t lane, std::span<const std::int32_t> tokens,
                         std::int32_t part_tokens) {
    if (pending_lane_ >= 0) { throw std::logic_error("PleRowFeeder: a gather is already pending"); }
    if (lane < 0 || static_cast<std::size_t>(lane) >= histories_.size() || tokens.empty() ||
        tokens.size() > static_cast<std::size_t>(max_tokens_) || part_tokens < 1) {
        throw std::invalid_argument("PleRowFeeder: chunk of " + std::to_string(tokens.size()) +
                                    " tokens is outside [1, " + std::to_string(max_tokens_) +
                                    "], its parts are empty, or its lane is invalid");
    }
    finish_anchors();
    Staging& staging = staging_.at(next_);
    staging.tokens.assign(tokens.begin(), tokens.end());
    staging.part_tokens     = part_tokens;
    const std::size_t parts = (tokens.size() + static_cast<std::size_t>(part_tokens) - 1) /
                              static_cast<std::size_t>(part_tokens);
    staging.parts.clear();
    staging.parts.resize(parts);
    staging.ready.clear();
    for (std::promise<void>& part : staging.parts) { staging.ready.push_back(part.get_future()); }
    pending_buffer_            = next_;
    next_                      = 1 - next_;
    next_part_                 = 0;
    pending_lane_              = lane;
    NgramHistory& lane_history = histories_[static_cast<std::size_t>(lane)];
    pending_start_             = lane_history;
    cancel_requested_.store(false, std::memory_order_relaxed);
    pending_ = std::async(std::launch::async, [this, &staging, &lane_history] {
        const auto width = static_cast<std::size_t>(staging.part_tokens);
        auto* out        = static_cast<std::uint8_t*>(staging.bytes.data());
        std::size_t part = 0;
        try {
            // The previous upload out of this buffer may still be in flight.
            CUDA_CHECK(cudaEventSynchronize(staging.uploaded.get()));
            for (; part < staging.parts.size(); ++part) {
                if (cancel_requested_.load(std::memory_order_relaxed)) { return; }
                const std::size_t first = part * width;
                const std::size_t count = std::min(width, staging.tokens.size() - first);
                const std::span<const std::int32_t> part_ids(staging.tokens.data() + first, count);
                table_.gather(part_ids, lane_history,
                              std::span<std::uint8_t>(out + first * kNgramTokenBytes,
                                                      count * kNgramTokenBytes));
                staging.parts[part].set_value();
            }
        } catch (...) {
            // The failed part and every later one report the failure.
            for (; part < staging.parts.size(); ++part) {
                staging.parts[part].set_exception(std::current_exception());
            }
        }
    });
}

void PleRowFeeder::upload(Tensor& rows, std::int32_t part, cudaStream_t stream) {
    if (pending_lane_ < 0) { throw std::logic_error("PleRowFeeder: no gather is pending"); }
    // The worker only reads staging.tokens and part_tokens, so they are safe to read while the
    // gather runs; validating first keeps a rejected call from consuming the gather.
    Staging& staging = staging_.at(pending_buffer_);
    if (part != next_part_) {
        throw std::logic_error("PleRowFeeder: parts must be uploaded in order");
    }
    const auto width        = static_cast<std::size_t>(staging.part_tokens);
    const std::size_t first = static_cast<std::size_t>(part) * width;
    const std::size_t count = std::min(width, staging.tokens.size() - first);
    const std::size_t bytes = count * static_cast<std::size_t>(kNgramTokenBytes);
    if (rows.dtype != DType::U8 || !rows.is_contiguous() || rows.bytes() != bytes ||
        rows.data == nullptr) {
        throw std::invalid_argument("PleRowFeeder: rows must be a contiguous U8 tensor of " +
                                    std::to_string(bytes) + " bytes");
    }
    const bool last = static_cast<std::size_t>(part) + 1 == staging.parts.size();
    try {
        staging.ready.at(static_cast<std::size_t>(part)).get();
    } catch (...) {
        pending_.wait();
        pending_      = {};
        pending_lane_ = -1;
        throw;
    }
    ++next_part_;
    CUDA_CHECK(cudaMemcpyAsync(rows.data,
                               static_cast<const std::uint8_t*>(staging.bytes.data()) +
                                   first * static_cast<std::size_t>(kNgramTokenBytes),
                               bytes, cudaMemcpyHostToDevice, stream));
    // Re-recorded after every part, so the buffer's next gather waits for every copy out of it
    // even when this gather is abandoned before its last part.
    CUDA_CHECK(cudaEventRecord(staging.uploaded.get(), stream));
    if (last) {
        std::future<void> worker = std::move(pending_);
        pending_lane_            = -1;
        worker.get();
    }
}

void PleRowFeeder::gather_now(std::span<const std::int32_t> lanes,
                              std::span<const std::int32_t> tokens, std::span<std::uint8_t> out) {
    if (lanes.size() != tokens.size() || lanes.empty()) {
        throw std::invalid_argument("PleRowFeeder: one lane per token is required");
    }
    std::array<std::int32_t, kMaxLanes> counts{};
    if (lanes.size() > counts.size()) {
        throw std::invalid_argument("PleRowFeeder: a round holds at most 12 lanes");
    }
    gather_proposals(lanes, tokens, 1, out);
    std::fill_n(counts.begin(), lanes.size(), 1);
    commit_proposals(lanes, tokens, 1, std::span<const std::int32_t>(counts).first(lanes.size()));
}

void PleRowFeeder::gather_proposals(std::span<const std::int32_t> lanes,
                                    std::span<const std::int32_t> tokens, std::int32_t width,
                                    std::span<std::uint8_t> out) {
    if (width < 1 || width > 8 || lanes.empty() || lanes.size() > kMaxLanes ||
        tokens.size() != lanes.size() * static_cast<std::size_t>(width)) {
        throw std::invalid_argument("PleRowFeeder: proposals require 1..12 chains of width 1..8");
    }
    if (out.size() != tokens.size() * static_cast<std::size_t>(kNgramTokenBytes)) {
        throw std::invalid_argument("PleRowFeeder: proposals need 2592 bytes per token");
    }
    std::array<NgramHistory, kMaxLanes> copies{};
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const auto earlier = lanes.first(row);
        if (std::find(earlier.begin(), earlier.end(), lanes[row]) != earlier.end()) {
            throw std::invalid_argument("PleRowFeeder: a proposal batch names a lane twice");
        }
        if (lanes[row] == pending_lane_) {
            throw std::logic_error("PleRowFeeder: the proposal lane has a pending gather");
        }
        copies[row] = history(lanes[row]);
    }
    wait_pending();
    finish_anchors();
    std::array<std::int32_t, 8 * kMaxLanes> missing_tokens{};
    std::array<NgramHistory*, 8 * kMaxLanes> histories{};
    std::array<std::size_t, 8 * kMaxLanes> destinations{};
    std::size_t missing = 0;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const auto first   = row * static_cast<std::size_t>(width);
        const auto& anchor = anchors_.at(static_cast<std::size_t>(lanes[row]));
        const bool cached  = anchor.lane == lanes[row] && anchor.token == tokens[first] &&
                             anchor.history.tokens == copies[row].tokens;
        if (cached) {
            if (anchor.failure) { std::rethrow_exception(anchor.failure); }
            std::ranges::copy(anchor.bytes,
                              out.begin() + static_cast<std::ptrdiff_t>(first * kNgramTokenBytes));
            copies[row].tokens = {copies[row].tokens[1], tokens[first]};
        }
        // Hashing all columns on the calling thread preserves each chain's predecessor order;
        // NgramTable advances the pointed-to copies while processing its compact token list.
        for (std::int32_t column = cached ? 1 : 0; column < width; ++column) {
            const auto index        = first + static_cast<std::size_t>(column);
            missing_tokens[missing] = tokens[index];
            histories[missing]      = &copies[row];
            destinations[missing]   = index;
            ++missing;
        }
    }
    if (missing == 0) { return; }
    proposal_rows_.resize(missing * static_cast<std::size_t>(kNgramTokenBytes));
    table_.gather(std::span<const std::int32_t>(missing_tokens).first(missing),
                  std::span<NgramHistory* const>(histories).first(missing), proposal_rows_);
    for (std::size_t i = 0; i < missing; ++i) {
        std::copy_n(proposal_rows_.data() + i * kNgramTokenBytes, kNgramTokenBytes,
                    out.data() + destinations[i] * kNgramTokenBytes);
    }
}

void PleRowFeeder::finish_anchors() {
    std::unique_lock lock(anchor_mutex_);
    if (anchor_phase_ == AnchorPhase::Idle) { return; }
    anchor_changed_.wait(lock, [this] { return anchor_phase_ == AnchorPhase::Ready; });
    for (std::size_t row = 0; row < anchor_count_; ++row) {
        const auto& anchor                                 = anchor_batch_[row];
        anchors_.at(static_cast<std::size_t>(anchor.lane)) = anchor;
    }
    anchor_phase_ = AnchorPhase::Idle;
}

void PleRowFeeder::prefetch_anchors(std::span<const std::int32_t> lanes,
                                    std::span<const std::int32_t> tokens) {
    if (lanes.empty() || lanes.size() > kMaxLanes || tokens.size() != lanes.size()) {
        throw std::invalid_argument("PleRowFeeder: an anchor fetch needs one token per lane");
    }

    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const auto earlier = lanes.first(row);
        if (lanes[row] == pending_lane_ || tokens[row] < 0 || tokens[row] >= kTokenDomain ||
            std::find(earlier.begin(), earlier.end(), lanes[row]) != earlier.end()) {
            throw std::invalid_argument("PleRowFeeder: invalid anchor fetch membership or token");
        }
        static_cast<void>(history(lanes[row]));
    }
    wait_pending();
    finish_anchors();
    {
        std::lock_guard lock(anchor_mutex_);
        anchor_count_ = lanes.size();
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            auto& anchor   = anchor_batch_[row];
            anchor.lane    = lanes[row];
            anchor.token   = tokens[row];
            anchor.history = history(lanes[row]);
            anchor.failure = {};
        }
        anchor_phase_ = AnchorPhase::Reading;
    }
    anchor_changed_.notify_all();
}

void PleRowFeeder::anchor_worker(const std::stop_token& stop) {
    while (!stop.stop_requested()) {
        std::unique_lock lock(anchor_mutex_);
        if (!anchor_changed_.wait(lock, stop,
                                  [this] { return anchor_phase_ == AnchorPhase::Reading; })) {
            return;
        }
        lock.unlock();
        // The owner cannot replace the batch until Ready is published under the mutex. All
        // canonical histories remain owner-thread-only; the worker advances local copies.
        std::array<std::int32_t, kMaxLanes> ids{};
        std::array<NgramHistory, kMaxLanes> copies{};
        std::array<NgramHistory*, kMaxLanes> histories{};
        std::array<std::uint8_t, kMaxLanes * kNgramTokenBytes> bytes{};
        for (std::size_t row = 0; row < anchor_count_; ++row) {
            ids[row]       = anchor_batch_[row].token;
            copies[row]    = anchor_batch_[row].history;
            histories[row] = &copies[row];
        }
        try {
            table_.gather(std::span<const std::int32_t>(ids).first(anchor_count_),
                          std::span<NgramHistory* const>(histories).first(anchor_count_),
                          std::span<std::uint8_t>(bytes).first(anchor_count_ * kNgramTokenBytes));
            for (std::size_t row = 0; row < anchor_count_; ++row) {
                std::copy_n(bytes.data() + row * kNgramTokenBytes, kNgramTokenBytes,
                            anchor_batch_[row].bytes.data());
            }
        } catch (...) {
            // Cancellation discards only that lane's entry. Surviving entries retain the read
            // failure, but joining the worker during lane cleanup never throws an I/O error.
            for (std::size_t row = 0; row < anchor_count_; ++row) {
                anchor_batch_[row].failure = std::current_exception();
            }
        }
        lock.lock();
        anchor_phase_ = AnchorPhase::Ready;
        lock.unlock();
        anchor_changed_.notify_all();
    }
}

void PleRowFeeder::commit_proposals(std::span<const std::int32_t> lanes,
                                    std::span<const std::int32_t> tokens, std::int32_t width,
                                    std::span<const std::int32_t> counts) {
    if (width < 1 || width > 8 || lanes.empty() || lanes.size() > kMaxLanes ||
        counts.size() != lanes.size() ||
        tokens.size() != lanes.size() * static_cast<std::size_t>(width)) {
        throw std::invalid_argument("PleRowFeeder: invalid proposal commit dimensions");
    }
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const auto earlier = lanes.first(row);
        if (counts[row] < 0 || counts[row] > width || lanes[row] == pending_lane_ ||
            std::find(earlier.begin(), earlier.end(), lanes[row]) != earlier.end()) {
            throw std::invalid_argument("PleRowFeeder: invalid proposal commit membership");
        }
        static_cast<void>(history(lanes[row]));
    }
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        auto& current = history(lanes[row]);
        for (std::int32_t column = 0; column < counts[row]; ++column) {
            current.tokens = {
                current.tokens[1],
                tokens[row * static_cast<std::size_t>(width) + static_cast<std::size_t>(column)]};
        }
    }
}

} // namespace ninfer::targets::qwen3_8_flash_next
