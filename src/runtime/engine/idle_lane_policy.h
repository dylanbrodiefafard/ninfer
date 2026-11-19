#pragma once

#include "ninfer/types.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>

namespace ninfer::runtime {

// Whether evicting idle lanes still frees KV the expert cache needs, in whole extents of the flex
// region: expiry applies only while the extents the cache can hold now (lendable_extents) are
// fewer than its flex demand plus the headroom. With headroom 0 eviction stops as soon as the
// demand is covered.
[[nodiscard]] constexpr bool idle_eviction_needed(std::int64_t lendable_extents,
                                                  std::int64_t demand_extents,
                                                  std::int64_t headroom_extents) noexcept {
    return lendable_extents < demand_extents + headroom_extents;
}

// Boundary-owned retention policy. Activity removes a lane from consideration; only a completed
// transition back to idle starts its timer. The latest currently idle lane is exempt, with a
// sequence number breaking equal clock ticks. Preservation and expiry copies do not constitute
// activity: waiting for completion or retrying a failed capture preserves the idle recency.
class IdleLanePolicy {
public:
    using Clock = std::chrono::steady_clock;

    explicit IdleLanePolicy(std::chrono::seconds timeout) : timeout_(timeout) {}

    [[nodiscard]] bool enabled() const noexcept { return timeout_.count() != 0; }

    void activity(std::uint32_t lane) {
        entries_.at(lane).reset();
        recheck_ = Clock::time_point{};
    }

    void idle(std::uint32_t lane, bool disposable, Clock::time_point now) {
        if (timeout_.count() == 0) { return; }
        entries_.at(lane) = Entry{now + timeout_, ++sequence_, disposable};
        recheck_          = Clock::time_point{};
    }

    // A successful capture releases no pages yet. Later boundaries query its completion; the
    // short wake-up deadline never changes idle recency or licenses release before completion.
    void await_capture(std::uint32_t lane, Clock::time_point now) {
        auto& entry     = require_idle(lane);
        entry.capturing = true;
        entry.deadline  = now + std::chrono::milliseconds(1);
    }

    [[nodiscard]] bool capturing(std::uint32_t lane) const {
        const auto& entry = entries_.at(lane);
        return entry && entry->capturing;
    }

    void retry(std::uint32_t lane, Clock::time_point now) {
        auto& entry     = require_idle(lane);
        entry.capturing = false;
        entry.deadline  = now + timeout_;
    }

    struct Expiry {
        std::uint32_t lane;
        bool disposable;
    };

    // The next boundary wake-up: the earliest deadline of a non-exempt lane. A lane whose expiry
    // was held back by due() is revisited no earlier than the recheck time, so a held lane never
    // makes the executor spin; a capture awaiting completion keeps its short deadline.
    [[nodiscard]] std::optional<Clock::time_point> deadline() const {
        const auto exempt = newest();
        std::optional<Clock::time_point> result;
        for (std::uint32_t lane = 0; lane < entries_.size(); ++lane) {
            const auto& entry = entries_.at(lane);
            if (!entry || lane == exempt) { continue; }
            const auto wake =
                entry->capturing ? entry->deadline : std::max(entry->deadline, recheck_);
            if (!result || wake < *result) { result = wake; }
        }
        return result;
    }

    // The expired non-exempt lane to process next: a lane in its capture phase first (it finishes
    // regardless of demand), otherwise the one idle longest.
    [[nodiscard]] std::optional<Expiry> expired(Clock::time_point now) const {
        const auto exempt = newest();
        std::optional<Expiry> result;
        bool result_capturing = false;
        auto sequence         = std::numeric_limits<std::uint64_t>::max();
        for (std::uint32_t lane = 0; lane < entries_.size(); ++lane) {
            const auto& entry = entries_.at(lane);
            if (!entry || lane == exempt || entry->deadline > now) { continue; }
            if ((entry->capturing && !result_capturing) ||
                (entry->capturing == result_capturing && entry->sequence < sequence)) {
                result           = Expiry{lane, entry->disposable};
                result_capturing = entry->capturing;
                sequence         = entry->sequence;
            }
        }
        return result;
    }

    // The expiry to process at this boundary. A lane in its capture phase always continues to
    // its close. A new expiry starts only while no lane is capturing (expiries never overlap; the
    // capturing lane's short deadline wakes the next boundary) and `eviction_needed` holds;
    // otherwise the lane stays retained with its timer expired, no capture or close is issued,
    // and the next wake is pushed out by kRecheckInterval. The caller re-evaluates
    // `eviction_needed` at every boundary, so the lane is evictable at any later boundary where
    // the condition holds. A lane becoming idle or active clears the pushed-out wake.
    [[nodiscard]] std::optional<Expiry> due(Clock::time_point now, bool eviction_needed) {
        const auto candidate = expired(now);
        if (!candidate || capturing(candidate->lane)) { return candidate; }
        if (std::ranges::any_of(entries_,
                                [](const auto& entry) { return entry && entry->capturing; })) {
            return std::nullopt;
        }
        if (eviction_needed) { return candidate; }
        recheck_ = now + kRecheckInterval;
        return std::nullopt;
    }

    static constexpr std::chrono::seconds kRecheckInterval{1};

private:
    struct Entry {
        Clock::time_point deadline;
        std::uint64_t sequence;
        bool disposable;
        bool capturing = false;
    };

    [[nodiscard]] Entry& require_idle(std::uint32_t lane) {
        auto& entry = entries_.at(lane);
        if (!entry) { throw std::logic_error("only retained idle lanes can expire"); }
        return *entry;
    }

    [[nodiscard]] std::optional<std::uint32_t> newest() const {
        std::optional<std::uint32_t> result;
        std::uint64_t sequence = 0;
        for (std::uint32_t lane = 0; lane < entries_.size(); ++lane) {
            const auto& entry = entries_.at(lane);
            if (entry && entry->sequence > sequence) {
                result   = lane;
                sequence = entry->sequence;
            }
        }
        return result;
    }

    std::chrono::seconds timeout_;
    Clock::time_point recheck_{};
    std::uint64_t sequence_ = 0;
    std::array<std::optional<Entry>, kMaximumConcurrency> entries_{};
};

} // namespace ninfer::runtime
