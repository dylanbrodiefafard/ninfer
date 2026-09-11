#pragma once

#include "ninfer/types.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <unordered_map>

namespace ninfer::runtime {

// The caller's unmerged stop policy distinguishes an explicit token stop from
// a natural model EOS. A caller stop ends the request even if completed calls
// also establish repetition; recovery must never turn it into another attempt.
[[nodiscard]] inline bool generation_recovery_allowed_at_finish(
    FinishReason reason, std::span<const TokenId> generated, const StopPolicy& caller_stop) {
    switch (reason) {
    case FinishReason::OutputLimit:
    case FinishReason::ContextCapacity:
        return true;
    case FinishReason::StopToken:
        return !generated.empty() &&
            std::find(caller_stop.token_ids.begin(), caller_stop.token_ids.end(), generated.back()) ==
                caller_stop.token_ids.end();
    case FinishReason::None:
    case FinishReason::StopString:
    case FinishReason::Cancelled:
        return false;
    }
    return false;
}

// Three non-overlapping occurrences of the same 256-token reasoning passage,
// plus >=4096 distinct generated tokens covered by repeat passages. Overlapping
// windows count their covered tokens only once. This leaves room for shorter
// loops to escape naturally without paying for a full-context retry.
// Unlike a fixed-period square, this survives small insertions/deletions elsewhere
// in a multi-paragraph loop. Hashes only locate candidates; exact IDs prove identity.
// Runtime supplies only the current generated thinking attempt, never prompt tokens.
class RepeatedReasoningSpan {
public:
    static constexpr std::size_t window = 256;
    static constexpr std::size_t minimum_repeated_tokens = 4096;
    [[nodiscard]] bool observe(std::span<const TokenId> generated) {
        if (generated.size() < processed_) { *this = {}; }
        while (processed_ < generated.size()) {
            const auto i = processed_++;
            if (i >= window) { hash_ -= atom(generated[i - window]) * power_; }
            hash_ = hash_ * base_ + atom(generated[i]);
            if (processed_ < window) { continue; }
            const auto begin = processed_ - window;
            const auto [first, end] = occurrences_.equal_range(hash_);
            bool found = false;
            for (auto candidate = first; candidate != end; ++candidate) {
                auto& occurrence = candidate->second;
                if (!std::equal(generated.begin() + begin, generated.begin() + processed_,
                                generated.begin() + occurrence.first_begin)) { continue; }
                found = true;
                if (begin >= occurrence.first_begin + window) {
                    repeated_tokens_ += processed_ - std::max(begin, covered_end_);
                    covered_end_ = processed_;
                }
                if (processed_ - occurrence.last_end >= window) {
                    occurrence.last_end = processed_;
                    has_three_ |= ++occurrence.copies >= 3;
                }
                if (has_three_ && repeated_tokens_ >= minimum_repeated_tokens) { return true; }
                break;
            }
            if (!found) { occurrences_.emplace(hash_, Occurrence{begin, processed_, 1}); }
        }
        return false;
    }
private:
    struct Occurrence { std::size_t first_begin, last_end, copies; };
    static constexpr std::uint64_t atom(TokenId token) {
        return static_cast<std::uint32_t>(token) + std::uint64_t{1};
    }
    static constexpr std::uint64_t base_ = 0x100000001b3ULL;
    static constexpr std::uint64_t power_ = [] {
        std::uint64_t result = 1;
        for (std::size_t i = 1; i < window; ++i) { result *= base_; }
        return result;
    }();
    std::size_t processed_ = 0;
    std::size_t repeated_tokens_ = 0, covered_end_ = 0;
    bool has_three_ = false;
    std::uint64_t hash_ = 0;
    std::unordered_multimap<std::uint64_t, Occurrence> occurrences_;
};

} // namespace ninfer::runtime
