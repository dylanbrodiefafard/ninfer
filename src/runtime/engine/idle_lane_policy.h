#pragma once

#include "ninfer/types.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>

namespace ninfer::runtime {

// Boundary-owned retention policy. Activity removes a lane from consideration; only a completed
// transition back to idle starts its timer. The latest currently idle lane is exempt, with a
// sequence number breaking equal clock ticks. Preservation and expiry copies do not constitute
// activity: waiting for completion or retrying a failed capture preserves the idle recency.
class IdleLanePolicy {
public:
    using Clock = std::chrono::steady_clock;

    explicit IdleLanePolicy(std::chrono::seconds timeout) : timeout_(timeout) {}

    void activity(std::uint32_t lane) { entries_.at(lane).reset(); }

    void idle(std::uint32_t lane, bool disposable, Clock::time_point now) {
        if (timeout_.count() == 0) { return; }
        entries_.at(lane) = Entry{now + timeout_, ++sequence_, disposable};
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

    [[nodiscard]] std::optional<Clock::time_point> deadline() const {
        const auto exempt = newest();
        std::optional<Clock::time_point> result;
        for (std::uint32_t lane = 0; lane < entries_.size(); ++lane) {
            const auto& entry = entries_.at(lane);
            if (!entry || lane == exempt) { continue; }
            if (!result || entry->deadline < *result) { result = entry->deadline; }
        }
        return result;
    }

    [[nodiscard]] std::optional<Expiry> expired(Clock::time_point now) const {
        const auto exempt = newest();
        std::optional<Expiry> result;
        auto sequence = std::numeric_limits<std::uint64_t>::max();
        for (std::uint32_t lane = 0; lane < entries_.size(); ++lane) {
            const auto& entry = entries_.at(lane);
            if (entry && lane != exempt && entry->deadline <= now && entry->sequence < sequence) {
                result   = Expiry{lane, entry->disposable};
                sequence = entry->sequence;
            }
        }
        return result;
    }

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
    std::uint64_t sequence_ = 0;
    std::array<std::optional<Entry>, kMaximumConcurrency> entries_{};
};

} // namespace ninfer::runtime
