#pragma once

#include "ninfer/types.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace ninfer::runtime {

// Thinking-only p-less cycle-exit: detect the least period p in [p_min, p_max] such that the
// last 2p generated token ids are a Hamming-near square. The continuation is x[n-p]. Exact
// identity is Hamming 0 and still wins as the least p.
//
// Allowed substitutions are 2p/p_max (integer divide), so p < 256 is exact. The traces that
// needed fuzz were ~500-token squares with one whitespace substitution — not 32-token
// near-repeats. A floor of 1 at p_min would treat "same 32-token template, one slot
// different" as a cycle; that is a common thinking pattern, not an attractor.
// Cost: early-out after budget+1 mismatches; worst case sum_{p=p_min}^{p_max} p compares.
inline constexpr std::size_t kTypicalCyclePeriodMin = 32;
inline constexpr std::size_t kTypicalCyclePeriodMax = 512;

struct TypicalCycle {
    std::size_t period     = 0;
    TokenId continuation = -1;
};

[[nodiscard]] inline std::size_t typical_cycle_hamming_max(std::size_t p) {
    return (2 * p) / kTypicalCyclePeriodMax;
}

// Hamming distance of generated[n-2p, n-p) vs generated[n-p, n). Aborts as soon as
// the count exceeds typical_cycle_hamming_max(p) (returned value is then max+1).
[[nodiscard]] inline std::size_t suffix_square_hamming(std::span<const TokenId> generated,
                                                       std::size_t p) {
    const std::size_t n   = generated.size();
    const std::size_t max = typical_cycle_hamming_max(p);
    const TokenId* a      = generated.data() + (n - 2 * p);
    const TokenId* b      = generated.data() + (n - p);
    std::size_t mism      = 0;
    for (std::size_t i = 0; i < p; ++i) {
        if (a[i] != b[i] && ++mism > max) { return mism; }
    }
    return mism;
}

// Least p in [p_min, p_max] with 2p <= n such that the last 2p generated ids are a
// Hamming-near square. Returns nullopt when no such period exists (including n < 2*p_min).
[[nodiscard]] inline std::optional<TypicalCycle>
least_square_period(std::span<const TokenId> generated,
                    std::size_t p_min = kTypicalCyclePeriodMin,
                    std::size_t p_max = kTypicalCyclePeriodMax) {
    const std::size_t n = generated.size();
    if (p_min == 0 || p_min > p_max || n < 2 * p_min) { return std::nullopt; }
    const std::size_t p_hi = std::min(p_max, n / 2);
    for (std::size_t p = p_min; p <= p_hi; ++p) {
        if (suffix_square_hamming(generated, p) <= typical_cycle_hamming_max(p)) {
            return TypicalCycle{p, generated[n - p]};
        }
    }
    return std::nullopt;
}

[[nodiscard]] inline TokenId typical_exclude_token(std::span<const TokenId> generated) {
    const auto hit = least_square_period(generated);
    return hit ? hit->continuation : TokenId{-1};
}

// Host packing gates. Identity (-1) unless thinking, p-less, T>0, and a square exists.
[[nodiscard]] inline TokenId typical_exclude_for_sample(bool in_reasoning, bool p_less,
                                                        float temperature,
                                                        std::span<const TokenId> generated) {
    if (!in_reasoning || !p_less || !(temperature > 0.0f)) { return -1; }
    return typical_exclude_token(generated);
}

} // namespace ninfer::runtime
