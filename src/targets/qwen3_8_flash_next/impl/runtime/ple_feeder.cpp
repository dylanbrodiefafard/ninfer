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
      staging_{Staging(staging_bytes(max_tokens)), Staging(staging_bytes(max_tokens))} {
    for (Staging& staging : staging_) {
        staging.tokens.reserve(static_cast<std::size_t>(max_tokens));
    }
}

PleRowFeeder::~PleRowFeeder() { wait_pending(); }

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
    lane_history = NgramHistory{};
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
    std::array<NgramHistory*, kMaxLanes> histories{};
    if (lanes.size() > histories.size()) {
        throw std::invalid_argument("PleRowFeeder: a round holds at most 12 lanes");
    }
    for (std::size_t i = 0; i < lanes.size(); ++i) {
        if (std::find(lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(i), lanes[i]) !=
            lanes.begin() + static_cast<std::ptrdiff_t>(i)) {
            throw std::invalid_argument("PleRowFeeder: a round names a lane twice");
        }
        if (lanes[i] == pending_lane_) {
            throw std::logic_error("PleRowFeeder: the lane has a pending gather");
        }
        histories.at(i) = &history(lanes[i]);
    }
    // The table serves one thread at a time: let a pending prefetch finish its reads first.
    wait_pending();
    table_.gather(tokens, std::span<NgramHistory* const>(histories.data(), lanes.size()), out);
}

void PleRowFeeder::gather_proposals(std::span<const std::int32_t> lanes,
                                    std::span<const std::int32_t> tokens, std::int32_t width,
                                    std::span<std::uint8_t> out) {
    if (width < 1 || width > 8 || lanes.empty() || lanes.size() > kMaxLanes ||
        tokens.size() != lanes.size() * static_cast<std::size_t>(width)) {
        throw std::invalid_argument("PleRowFeeder: proposals require 1..12 chains of width 1..8");
    }
    std::array<NgramHistory, kMaxLanes> copies{};
    std::array<NgramHistory*, 8 * kMaxLanes> histories{};
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const auto earlier = lanes.first(row);
        if (std::find(earlier.begin(), earlier.end(), lanes[row]) != earlier.end()) {
            throw std::invalid_argument("PleRowFeeder: a proposal batch names a lane twice");
        }
        if (lanes[row] == pending_lane_) {
            throw std::logic_error("PleRowFeeder: the proposal lane has a pending gather");
        }
        copies[row] = history(lanes[row]);
        for (std::int32_t column = 0; column < width; ++column) {
            histories[row * static_cast<std::size_t>(width) + static_cast<std::size_t>(column)] =
                &copies[row];
        }
    }
    wait_pending();
    table_.gather(tokens, std::span<NgramHistory* const>(histories.data(), tokens.size()), out);
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
