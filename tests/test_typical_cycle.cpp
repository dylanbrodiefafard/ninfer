// Suffix-square detector: exact at p<256, Hamming ≤ 2p/512 otherwise.
#include "runtime/contract/typical_cycle.h"

#include <iostream>
#include <span>
#include <utility>
#include <vector>

namespace {

using ninfer::TokenId;
using ninfer::runtime::TypicalCycle;
using ninfer::runtime::kTypicalCyclePeriodMax;
using ninfer::runtime::kTypicalCyclePeriodMin;
using ninfer::runtime::least_square_period;
using ninfer::runtime::suffix_square_hamming;
using ninfer::runtime::typical_cycle_hamming_max;
using ninfer::runtime::typical_exclude_for_sample;
using ninfer::runtime::typical_exclude_token;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

std::vector<TokenId> repeat_block(const std::vector<TokenId>& block, int copies) {
    std::vector<TokenId> out;
    out.reserve(block.size() * static_cast<std::size_t>(copies));
    for (int i = 0; i < copies; ++i) {
        out.insert(out.end(), block.begin(), block.end());
    }
    return out;
}

int detector_empty_and_short() {
    int failures = 0;
    failures += check(!least_square_period({}).has_value(), "empty span reported a square");
    const std::vector<TokenId> short_span(2 * kTypicalCyclePeriodMin - 1, TokenId{7});
    failures +=
        check(!least_square_period(short_span).has_value(), "n=63 identical ids reported a square");
    return failures;
}

int detector_min_period_square() {
    std::vector<TokenId> block(kTypicalCyclePeriodMin);
    for (std::size_t i = 0; i < block.size(); ++i) {
        block[i] = static_cast<TokenId>(1000 + static_cast<int>(i));
    }
    const auto twice = repeat_block(block, 2);
    const auto hit   = least_square_period(twice);
    int failures     = 0;
    failures += check(hit.has_value(), "exact 32-token square was not detected");
    if (hit) {
        failures += check(hit->period == kTypicalCyclePeriodMin, "min-period square was not p=32");
        failures += check(hit->continuation == block[0], "continuation was not x[n-p]");
        failures += check(typical_exclude_token(twice) == block[0],
                          "typical_exclude_token did not return the continuation");
    }
    const auto once = block;
    failures += check(!least_square_period(once).has_value(), "a single copy reported a square");
    return failures;
}

int detector_below_p_min() {
    std::vector<TokenId> block(kTypicalCyclePeriodMin - 1);
    for (std::size_t i = 0; i < block.size(); ++i) {
        block[i] = static_cast<TokenId>(static_cast<int>(i) + 20);
    }
    const auto twice = repeat_block(block, 2);
    return check(!least_square_period(twice).has_value(),
                 "period 31 square was admitted below p_min");
}

int detector_primitive_not_multiple() {
    std::vector<TokenId> block(kTypicalCyclePeriodMin);
    for (std::size_t i = 0; i < block.size(); ++i) {
        block[i] = static_cast<TokenId>(i + 1);
    }
    const auto four = repeat_block(block, 4);
    const auto hit  = least_square_period(four);
    int failures    = 0;
    failures += check(hit.has_value() && hit->period == kTypicalCyclePeriodMin,
                      "u^4 did not report the primitive period");
    return failures;
}

int detector_max_period() {
    std::vector<TokenId> block(kTypicalCyclePeriodMax);
    for (std::size_t i = 0; i < block.size(); ++i) {
        block[i] = static_cast<TokenId>(static_cast<int>(i) * 3 + 9);
    }
    const auto twice = repeat_block(block, 2);
    const auto hit   = least_square_period(twice);
    int failures     = 0;
    failures += check(hit.has_value() && hit->period == kTypicalCyclePeriodMax,
                      "period 512 square was not detected");
    std::vector<TokenId> too_long(kTypicalCyclePeriodMax + 1);
    for (std::size_t i = 0; i < too_long.size(); ++i) {
        too_long[i] = static_cast<TokenId>(static_cast<int>(i) + 50);
    }
    failures += check(!least_square_period(repeat_block(too_long, 2)).has_value(),
                      "period 513 square was admitted above p_max");
    return failures;
}

int detector_rotation_and_break() {
    std::vector<TokenId> block(kTypicalCyclePeriodMin);
    for (std::size_t i = 0; i < block.size(); ++i) {
        block[i] = static_cast<TokenId>(400 + static_cast<int>(i));
    }
    auto seq     = repeat_block(block, 2);
    int failures = 0;
    seq.push_back(block[0]);
    auto hit = least_square_period(seq);
    failures += check(hit.has_value() && hit->period == kTypicalCyclePeriodMin,
                      "rotated square after emitting c was not detected");
    failures += check(hit && hit->continuation == block[1],
                      "continuation after stay was not the next cycle token");
    seq.back() = block[0] + 999;
    failures +=
        check(!least_square_period(seq).has_value(), "one substitution at p=32 still reported a square");
    return failures;
}

int detector_prompt_excluded() {
    std::vector<TokenId> prompt(kTypicalCyclePeriodMin, TokenId{1});
    std::vector<TokenId> generated = prompt;
    generated.insert(generated.end(), prompt.begin(), prompt.end());
    // Detector is called on the generated span only. A prompt echo is not a generated square.
    const std::span<const TokenId> gen_only(generated.data() + prompt.size(), prompt.size());
    return check(!least_square_period(gen_only).has_value(),
                 "a single generated copy of the prompt was treated as a square");
}

int detector_short_period_is_exact() {
    std::vector<TokenId> block(kTypicalCyclePeriodMin);
    for (std::size_t i = 0; i < block.size(); ++i) {
        block[i] = static_cast<TokenId>(800 + static_cast<int>(i));
    }
    auto twice   = repeat_block(block, 2);
    twice.back() = block.back() + 1;
    int failures = 0;
    failures += check(typical_cycle_hamming_max(kTypicalCyclePeriodMin) == 0,
                      "p=32 Hamming budget was not 0");
    failures += check(suffix_square_hamming(twice, kTypicalCyclePeriodMin) == 1,
                      "p=32 Hamming-1 fixture was not distance 1");
    failures += check(!least_square_period(twice).has_value(),
                      "Hamming-1 period-32 square was admitted");
    return failures;
}

int detector_hamming_budget_scales() {
    int failures = 0;
    failures += check(typical_cycle_hamming_max(kTypicalCyclePeriodMin) == 0,
                      "p=32 Hamming budget was not 0");
    failures += check(typical_cycle_hamming_max(255) == 0, "p=255 Hamming budget was not 0");
    failures += check(typical_cycle_hamming_max(256) == 1, "p=256 Hamming budget was not 1");
    failures += check(typical_cycle_hamming_max(511) == 1, "p=511 Hamming budget was not 1");
    failures += check(typical_cycle_hamming_max(kTypicalCyclePeriodMax) == 2,
                      "p=512 Hamming budget was not 2");
    return failures;
}

int detector_fuzzy_budget() {
    constexpr std::size_t p256 = 256;
    std::vector<TokenId> mid_block(p256);
    for (std::size_t i = 0; i < mid_block.size(); ++i) {
        mid_block[i] = static_cast<TokenId>(2000 + static_cast<int>(i));
    }
    auto mid_twice               = repeat_block(mid_block, 2);
    mid_twice[mid_twice.size() - 17] = mid_block[p256 - 17] + 11;
    const auto mid_hit           = least_square_period(mid_twice);
    int failures                 = 0;
    failures += check(mid_hit.has_value() && mid_hit->period == p256,
                      "Hamming-1 period-256 square was not detected");
    failures += check(mid_hit && mid_hit->continuation == mid_block[0],
                      "p=256 near-square continuation was not x[n-p]");

    std::vector<TokenId> long_block(kTypicalCyclePeriodMax);
    for (std::size_t i = 0; i < long_block.size(); ++i) {
        long_block[i] = static_cast<TokenId>(3000 + static_cast<int>(i));
    }
    auto long_twice = repeat_block(long_block, 2);
    long_twice[long_twice.size() - 1]  = long_block.back() + 1;
    long_twice[long_twice.size() - 17] = long_block[kTypicalCyclePeriodMax - 17] + 1;
    failures += check(suffix_square_hamming(long_twice, kTypicalCyclePeriodMax) == 2,
                      "p=512 Hamming-2 fixture was not distance 2");
    const auto hit = least_square_period(long_twice);
    failures += check(hit.has_value() && hit->period == kTypicalCyclePeriodMax,
                      "Hamming-2 period-512 square was not detected");
    long_twice[long_twice.size() - 33] = long_block[kTypicalCyclePeriodMax - 33] + 1;
    failures += check(!least_square_period(long_twice).has_value(),
                      "Hamming-3 period-512 square was admitted");
    return failures;
}

int detector_non_square_permutation() {
    std::vector<TokenId> a(kTypicalCyclePeriodMin);
    std::vector<TokenId> b(kTypicalCyclePeriodMin);
    for (std::size_t i = 0; i < a.size(); ++i) {
        a[i] = static_cast<TokenId>(static_cast<int>(i) + 10);
        b[i] = a[kTypicalCyclePeriodMin - 1 - i];
    }
    std::vector<TokenId> seq = a;
    seq.insert(seq.end(), b.begin(), b.end());
    return check(!least_square_period(seq).has_value(),
                 "reversed 32-token suffix was treated as a square");
}

int host_gates() {
    std::vector<TokenId> block(kTypicalCyclePeriodMin, TokenId{8});
    block[0]               = 9;
    const auto twice       = repeat_block(block, 2);
    const TokenId expected = typical_exclude_token(twice);
    int failures           = 0;
    failures += check(typical_exclude_for_sample(true, true, 2.0f, twice) == expected,
                      "thinking p-less T>0 did not return the continuation");
    failures += check(typical_exclude_for_sample(false, true, 2.0f, twice) == -1,
                      "non-thinking still armed typical_exclude");
    failures += check(typical_exclude_for_sample(true, false, 2.0f, twice) == -1,
                      "non-p-less still armed typical_exclude");
    failures += check(typical_exclude_for_sample(true, true, 0.0f, twice) == -1,
                      "greedy T<=0 still armed typical_exclude");
    failures += check(typical_exclude_for_sample(true, true, 2.0f, {}) == -1,
                      "empty generated span armed typical_exclude");
    return failures;
}

// Lemma (singleton trap): if q = p_mode > 1/2 on a 2-mass distribution, V is the mode.
int singleton_trap_lemma() {
    auto typical_set = [](double q, double r) {
        const double L      = q * q + r * r;
        const bool mode_in  = q >= L;
        const bool other_in = r >= L;
        return std::pair<bool, bool>{mode_in, other_in};
    };
    int failures = 0;
    {
        const auto [mode_in, other_in] = typical_set(0.8, 0.2);
        failures += check(mode_in && !other_in, "q=0.8 2-mass did not collapse V to the mode");
    }
    {
        const auto [mode_in, other_in] = typical_set(0.51, 0.49);
        failures += check(mode_in && !other_in, "q=0.51 2-mass admitted the runner-up");
    }
    {
        const auto [mode_in, other_in] = typical_set(0.5, 0.5);
        failures += check(mode_in && other_in, "tied 2-mass did not keep both atoms");
    }
    // Proof obligation: r >= q^2+r^2 with q+r<=1 requires q<=1/2.
    for (double q = 0.51; q <= 1.0; q += 0.01) {
        const double r = 1.0 - q;
        const double L = q * q + r * r;
        if (r >= L) {
            std::cerr << "singleton trap counterexample q=" << q << " r=" << r << " L=" << L
                      << '\n';
            ++failures;
            break;
        }
    }
    return failures;
}

// Closed class: if Square_p and the next token is c, the suffix remains a square.
int closed_class_stay_leave() {
    std::vector<TokenId> u(kTypicalCyclePeriodMin);
    for (std::size_t i = 0; i < u.size(); ++i) { u[i] = static_cast<TokenId>(700 + static_cast<int>(i)); }
    auto seq           = repeat_block(u, 2);
    const TokenId c    = u[0];
    int failures       = 0;
    auto after_c       = seq;
    after_c.push_back(c);
    failures += check(least_square_period(after_c).has_value(),
                      "emitting c did not keep the cyclic class closed");
    auto after_exit = seq;
    after_exit.push_back(c + 1);
    failures += check(!least_square_period(after_exit).has_value(),
                      "emitting a non-continuation did not leave the class");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += detector_empty_and_short();
    failures += detector_min_period_square();
    failures += detector_below_p_min();
    failures += detector_primitive_not_multiple();
    failures += detector_max_period();
    failures += detector_rotation_and_break();
    failures += detector_prompt_excluded();
    failures += detector_short_period_is_exact();
    failures += detector_hamming_budget_scales();
    failures += detector_fuzzy_budget();
    failures += detector_non_square_permutation();
    failures += host_gates();
    failures += singleton_trap_lemma();
    failures += closed_class_stay_leave();
    std::cout << (failures == 0 ? "OK" : "FAIL") << " typical-cycle detector and topology lemmas\n";
    return failures == 0 ? 0 : 1;
}
