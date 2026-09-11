// Suffix-square detector: exact at p<256, Hamming ≤ 2p/512 otherwise, through p=2048.
#include "runtime/contract/typical_cycle.h"
#include "runtime/contract/reasoning_recovery.h"

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
                      "maximum-period square was not detected");
    std::vector<TokenId> too_long(kTypicalCyclePeriodMax + 1);
    for (std::size_t i = 0; i < too_long.size(); ++i) {
        too_long[i] = static_cast<TokenId>(static_cast<int>(i) + 50);
    }
    failures += check(!least_square_period(repeat_block(too_long, 2)).has_value(),
                      "period above p_max was admitted");
    return failures;
}

// The captured OpenCode DFLASH-SCHEDULE attractor repeats a roughly 546-token paragraph group.
// This is deliberately above the former 512-token ceiling.
int detector_captured_long_period() {
    constexpr std::size_t observed_period = 546;
    std::vector<TokenId> block(observed_period);
    for (std::size_t i = 0; i < block.size(); ++i) {
        block[i] = static_cast<TokenId>(5000 + static_cast<int>(i));
    }
    auto twice = repeat_block(block, 2);
    twice[twice.size() - 31] = block[observed_period - 31] + 1;
    twice[twice.size() - 79] = block[observed_period - 79] + 1;

    int failures = 0;
    failures += check(typical_cycle_hamming_max(observed_period) == 2,
                      "p=546 Hamming budget was not 2");
    const auto hit = least_square_period(twice);
    failures += check(hit.has_value() && hit->period == observed_period,
                      "captured Hamming-2 period-546 shape was not detected");
    failures += check(hit && hit->continuation == block[0],
                      "p=546 continuation was not x[n-p]");
    twice[twice.size() - 127] = block[observed_period - 127] + 1;
    failures += check(!least_square_period(twice).has_value(),
                      "Hamming-3 period-546 shape was admitted");
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
    failures += check(typical_cycle_hamming_max(512) == 2, "p=512 Hamming budget was not 2");
    failures += check(typical_cycle_hamming_max(1024) == 4,
                      "p=1024 Hamming budget was not 4");
    failures += check(typical_cycle_hamming_max(2048) == 8,
                      "p=2048 Hamming budget was not 8");
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

    constexpr std::size_t long_period = 1024;
    std::vector<TokenId> long_block(long_period);
    for (std::size_t i = 0; i < long_block.size(); ++i) {
        long_block[i] = static_cast<TokenId>(3000 + static_cast<int>(i));
    }
    auto long_twice = repeat_block(long_block, 2);
    long_twice[long_twice.size() - 1]   = long_block.back() + 1;
    long_twice[long_twice.size() - 17]  = long_block[long_period - 17] + 1;
    long_twice[long_twice.size() - 33]  = long_block[long_period - 33] + 1;
    long_twice[long_twice.size() - 49]  = long_block[long_period - 49] + 1;
    failures += check(suffix_square_hamming(long_twice, long_period) == 4,
                      "p=1024 Hamming-4 fixture was not distance 4");
    const auto hit = least_square_period(long_twice);
    failures += check(hit.has_value() && hit->period == long_period,
                      "Hamming-4 period-1024 square was not detected");
    long_twice[long_twice.size() - 65] = long_block[long_period - 65] + 1;
    failures += check(!least_square_period(long_twice).has_value(),
                      "Hamming-5 period-1024 square was admitted");
    return failures;
}

// Periods observed in exact T1.5 traces, plus the new inclusive boundary.
// Unique IDs isolate period/tolerance behavior from vocabulary coincidences.
int detector_new_long_periods() {
    int failures = 0;
    for (const auto [period, tolerance] :
         {std::pair<std::size_t, std::size_t>{1081, 4}, {1516, 5}, {1886, 7}, {2048, 8}}) {
        std::vector<TokenId> block(period);
        for (std::size_t i = 0; i < period; ++i) { block[i] = static_cast<TokenId>(10000 + i); }
        auto twice = repeat_block(block, 2);
        failures += check(!least_square_period(twice, 32, 1024),
                          "long-period regression unexpectedly fits the former ceiling");
        for (std::size_t i = 0; i < tolerance; ++i) { twice[period + 17*i] += 50000; }
        auto hit = least_square_period(twice);
        failures += check(hit && hit->period == period && hit->continuation == twice[period],
                          "new long-period tolerance boundary or continuation failed");
        twice[period + 17*tolerance] += 50000;
        failures += check(!least_square_period(twice),
                          "new long-period detector exceeded its mismatch tolerance");
    }
    return failures;
}

int persistent_recovery_evidence() {
    using ninfer::runtime::RepeatedReasoningSpan;
    int failures = 0;
    std::vector<TokenId> paragraph(256);
    for (std::size_t i = 0; i < paragraph.size(); ++i) { paragraph[i] = 10000 + i; }
    RepeatedReasoningSpan detector;
    failures += check(!detector.observe(repeat_block(paragraph, 2)),
                      "two paragraphs triggered a retry without persistence evidence");
    auto three = repeat_block(paragraph, 3);
    failures += check(!detector.observe(three), "a brief repeated passage triggered a costly retry");
    failures += check(detector.observe(repeat_block(paragraph, 17)),
                      "4096 redundant tokens with three copies did not recover");
    detector = {};
    auto varied = repeat_block(paragraph, 20);
    for (std::size_t copy = 0; copy < 20; ++copy) { varied[(copy + 1)*256 - 1] += copy; }
    failures += check(!detector.observe(varied),
                      "non-identical token passages triggered an exact-span retry");
    detector = {};
    const std::vector<TokenId> uniform(4352, 7);
    failures += check(!detector.observe(std::span(uniform).first(4351)),
                      "overlapping windows double-counted redundant token coverage");
    failures += check(detector.observe(uniform), "redundant-token boundary did not recover");
    std::vector<TokenId> long_reasoning(16000);
    for (std::size_t i = 0; i < long_reasoning.size(); ++i) { long_reasoning[i] = 10000 + i; }
    detector = {};
    failures += check(!detector.observe(long_reasoning),
                      "reasoning length alone triggered a retry");
    // Different gaps model insertions/deletions induced by one-token exclusions;
    // the repeated passage persists even though no stable cycle period exists.
    std::vector<TokenId> disrupted;
    TokenId gap = 90000;
    for (int copy = 0; copy < 17; ++copy) {
        for (int i = 0; i < 23 + copy; ++i) { disrupted.push_back(gap++); }
        disrupted.insert(disrupted.end(), paragraph.begin(), paragraph.end());
    }
    detector = {};
    bool recovered = false;
    for (std::size_t n = 1; n <= disrupted.size(); ++n) {
        recovered |= detector.observe(std::span(disrupted).first(n));
    }
    failures += check(recovered, "variable-period returning passage escaped recovery");
    failures += check(!detector.observe(paragraph), "a fresh shorter attempt retained old evidence");
    paragraph.resize(5000);
    for (std::size_t i = 0; i < paragraph.size(); ++i) { paragraph[i] = 10000 + i; }
    detector = {};
    auto long_copies = repeat_block(paragraph, 2);
    failures += check(!detector.observe(long_copies),
                      "two large repeated passages bypassed the three-copy requirement");
    long_copies.insert(long_copies.end(), paragraph.begin(), paragraph.begin() + 256);
    failures += check(detector.observe(long_copies),
                      "a multi-paragraph cycle above the square ceiling escaped recovery");
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
    failures += detector_captured_long_period();
    failures += detector_rotation_and_break();
    failures += detector_prompt_excluded();
    failures += detector_short_period_is_exact();
    failures += detector_hamming_budget_scales();
    failures += detector_fuzzy_budget();
    failures += detector_new_long_periods();
    failures += persistent_recovery_evidence();
    failures += detector_non_square_permutation();
    failures += host_gates();
    failures += singleton_trap_lemma();
    failures += closed_class_stay_leave();
    std::cout << (failures == 0 ? "OK" : "FAIL") << " typical-cycle detector and topology lemmas\n";
    return failures == 0 ? 0 : 1;
}
