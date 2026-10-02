#pragma once

// Family host policy for adaptive draft length. No CUDA.
//
// Objective: match the best captured constant-k policy. Mixing k forks the
// greedy DFlash/MTP path, so this is not a mixing bandit.
//
// Y(k) = 1 + sum_{i<k} q_i with q_i = prod_{j<=i} r_j and
// r_i = P(accepted > i | accepted > i-1).
//
// DFlash hop model. DFlash2 block attention makes r depend on the block length k:
//   r_{k,i} = clamp(1 - c_i * H[law][k][i], 0, kAdaptiveMaxHopRate).
// H is an engine-global rejection hazard per draft law, block length and hop, learned online
// only from exploration rounds. An exploration round (a hash of a Program-wide counter, so
// independent of request content) runs a uniformly drawn captured k as a one-round override;
// only full-extent rows count. Randomizing k gives every k the same content mix, so H carries
// no selection bias. Measured per-k differences are at most about a tenth of the hazard while
// request content moves it several-fold, so the shrinkage is heavy: H starts as the pooled
// hazard and leaves it only on substantial evidence. Three levels:
//   P_i      pooled hazard of hop i over every k (prior 0.35);
//   rho_k    the block length's hazard ratio, failures over expected failures sum(P_i) across
//            all of its hops (prior ratio 1);
//   H_{k,i}  = (failures + b * rho_k * P_i) / (trials + b).
// Exploration runs one round in kAdaptiveExploreEvery.
// c_i is the request's content factor on H: discounted failures over discounted expected
// failures sum(H) at hop i, using H in force before each round. Every recorded round decays
// all hops, so a hop the picker stops reaching relaxes to its prior, and that prior inherits
// half of c_{i-1}'s departure from 1. c only predicts; it never feeds H.
//
// MTP pooled model. MTP drafts autoregressively, so hop i does not depend on k: r_i is a
// discounted Beta, updated only when the prefix reached i, and E[Y] truncates at the first
// unseen hop (the truncation lands on MTP's cheapest arm k=3; counting unseen hops as accepted
// measured 5% slower there at C=1).
//
// T(k,C,L) = a_{C,k} + c_C L from online least squares (shared slope, per-k
// intercept), one table per batch size. Each captured k within the cap is
// measured once per batch size before exploitation; then argmax E[Y]/T.
// Switching k adds 1 ms to that arm's T. Ties keep the smaller k.

#include "ninfer/types.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::targets::qwen3_6 {

inline constexpr float kAdaptiveEwma                 = 32.0f;
inline constexpr float kAdaptiveEwmaAlpha            = 2.0f / (kAdaptiveEwma + 1.0f);
inline constexpr float kAdaptiveDiscount             = 1.0f - kAdaptiveEwmaAlpha;
inline constexpr float kAdaptiveBetaPrior            = 1.0f;
inline constexpr std::uint32_t kAdaptiveTBins        = 16;
inline constexpr std::uint32_t kAdaptiveMaxHops      = 16;
inline constexpr float kAdaptiveSwitchSeconds        = 0.001f;
inline constexpr std::uint32_t kAdaptiveHopTableMaxK = 8;
inline constexpr float kAdaptiveHazardPrior          = 1.0f;
inline constexpr float kAdaptiveHazardInherit        = 0.5f;
inline constexpr float kAdaptiveMaxHopRate           = 0.995f;
inline constexpr std::uint32_t kAdaptiveExploreEvery = 32;
inline constexpr float kAdaptivePooledHazardPrior    = 0.35f;
inline constexpr float kAdaptivePooledHazardTrials   = 4.0f;
inline constexpr float kAdaptiveBlockRatioPrior      = 64.0f; // expected failures
inline constexpr float kAdaptiveCellPriorTrials      = 256.0f;
inline constexpr float kAdaptiveCellDiscount         = 1.0f - 1.0f / 8192.0f;
inline constexpr float kAdaptivePooledDiscount       = 1.0f - 1.0f / 1024.0f;

// The drafter draws argmax for a greedy target, at the scaled draft temperature for a p-less
// target, and at the target temperature otherwise; each law has its own hop table.
enum class AdaptiveDraftLaw : std::uint8_t { Greedy, PLess, Sampled };
inline constexpr std::uint32_t kAdaptiveDraftLaws = 3;

// Engine-global DFlash hop hazards learned from exploration rounds (see the header).
// Block lengths above kAdaptiveHopTableMaxK share its row.
struct AdaptiveHopRates {
    float failures[kAdaptiveDraftLaws][kAdaptiveHopTableMaxK + 1][kAdaptiveHopTableMaxK] = {};
    float trials[kAdaptiveDraftLaws][kAdaptiveHopTableMaxK + 1][kAdaptiveHopTableMaxK]   = {};
    float pooled_failures[kAdaptiveDraftLaws][kAdaptiveHopTableMaxK]                     = {};
    float pooled_trials[kAdaptiveDraftLaws][kAdaptiveHopTableMaxK]                       = {};
    std::uint64_t decisions                                                              = 0;
};

struct AdaptiveDraftState {
    std::uint32_t live_k                    = 0;
    std::uint32_t rounds_at_k               = 0;
    std::uint32_t observed                  = 0;
    float alpha[kAdaptiveMaxHops]           = {};
    float beta[kAdaptiveMaxHops]            = {};
    std::uint16_t r_seen                    = 0;
    std::uint64_t rounds_hist[16]           = {};
    AdaptiveDraftLaw law                    = AdaptiveDraftLaw::Greedy;
    float hazard_failures[kAdaptiveMaxHops] = {};
    float hazard_expected[kAdaptiveMaxHops] = {};
};

struct AdaptiveRoundTimeState {
    float n[kAdaptiveTBins]      = {};
    float mean_L[kAdaptiveTBins] = {};
    float mean_T[kAdaptiveTBins] = {};
    float Sxx[kAdaptiveTBins]    = {};
    float Sxy[kAdaptiveTBins]    = {};
};

struct AdaptiveBatchKState {
    std::uint32_t live_k      = 0;
    std::uint32_t rounds_at_k = 0;
};

struct AdaptiveDraftConfig {
    std::span<const std::uint32_t> captured_ks;
    const AdaptiveRoundTimeState* round_time = nullptr;
    std::uint32_t length_tokens              = 0;
    float switch_seconds                     = kAdaptiveSwitchSeconds;
    AdaptiveHopRates* hop_rates              = nullptr; // DFlash hop model; null: MTP pooled
};

// DFlash captures {3..min(N, maximum_adaptive_k)}; each target bounds its set by the widths its
// verify routes serve without a slower fallback.
[[nodiscard]] inline std::vector<std::uint32_t>
adaptive_draft_ks(SpeculativeBackend backend, std::uint32_t n, bool adaptive,
                  std::uint32_t dflash_maximum_adaptive_k) {
    if (!adaptive || backend == SpeculativeBackend::None || n == 0) { return {n}; }
    std::vector<std::uint32_t> out;
    if (backend == SpeculativeBackend::Mtp) {
        for (std::uint32_t k = 3; k <= 5 && k <= n; ++k) { out.push_back(k); }
        return out.empty() ? std::vector<std::uint32_t>{n} : out;
    }
    // k=1/2 never beat k=3 at any C=1..6 on the A8 verify routes: C=1 round time is nearly flat
    // in k, and at C>=4 a k=4 round is cheaper than a k=1 round.
    if (n < 5) { return {n}; }
    for (std::uint32_t k = 3; k <= n && k <= dflash_maximum_adaptive_k; ++k) { out.push_back(k); }
    return out.empty() ? std::vector<std::uint32_t>{n} : out;
}

[[nodiscard]] inline std::uint32_t adaptive_k_index(std::span<const std::uint32_t> captured_ks,
                                                    std::uint32_t k) {
    for (std::uint32_t i = 0; i < captured_ks.size(); ++i) {
        if (captured_ks[i] == k) { return i; }
    }
    return 0;
}

[[nodiscard]] inline std::uint32_t adaptive_k_stride(std::uint32_t max_concurrency,
                                                     std::uint32_t max_planned_topology) {
    return max_concurrency * (1U + max_planned_topology);
}

[[nodiscard]] inline std::uint32_t adaptive_topology_class(std::uint32_t k_index,
                                                           std::uint32_t k_stride,
                                                           std::uint32_t planned_topology,
                                                           std::uint32_t max_concurrency,
                                                           std::uint32_t batch_size) {
    return k_index * k_stride + planned_topology * max_concurrency + (batch_size - 1U);
}

// Speculative verify graphs are captured with and without the tool-grammar mask
// exchange: a round whose batch has no grammar row skips it, and any grammar row
// selects the variant whose host-stream exchange overlaps target verification.
inline constexpr std::uint32_t kGrammarExchangeVariants = 2;

[[nodiscard]] constexpr std::uint32_t grammar_exchange_topology(std::uint32_t topology,
                                                                bool exchange) noexcept {
    return topology * kGrammarExchangeVariants + (exchange ? 1U : 0U);
}

[[nodiscard]] inline std::uint32_t
adaptive_snap_captured_k(std::span<const std::uint32_t> captured_ks, std::uint32_t k) {
    if (captured_ks.empty()) { return k; }
    for (std::uint32_t c : captured_ks) {
        if (c >= k) { return c; }
    }
    return captured_ks.back();
}

[[nodiscard]] inline std::uint32_t adaptive_batch_k(std::span<const std::uint32_t> row_k,
                                                    std::span<const std::uint32_t> captured_ks) {
    std::uint32_t batch_k = 0;
    for (std::uint32_t k : row_k) { batch_k = std::max(batch_k, k); }
    return adaptive_snap_captured_k(captured_ks, batch_k);
}

// Smallest captured k. First round selects from T if any exist; this is the
// fallback when every T is unknown. Not a throughput attractor.
[[nodiscard]] inline std::uint32_t adaptive_seed_k(std::span<const std::uint32_t> captured_ks,
                                                   SpeculativeBackend) {
    if (captured_ks.empty()) { return 0; }
    return captured_ks.front();
}

inline void seed_adaptive_draft_state(AdaptiveDraftState& state, std::uint32_t live_k,
                                      AdaptiveDraftLaw law) {
    state        = {};
    state.live_k = live_k;
    state.law    = law;
}

[[nodiscard]] inline float adaptive_pooled_hazard(const AdaptiveHopRates& rates,
                                                  AdaptiveDraftLaw law, std::uint32_t hop) {
    const auto l          = static_cast<std::uint32_t>(law);
    const std::uint32_t i = std::min(hop, kAdaptiveHopTableMaxK - 1U);
    return (rates.pooled_failures[l][i] +
            kAdaptivePooledHazardTrials * kAdaptivePooledHazardPrior) /
           (rates.pooled_trials[l][i] + kAdaptivePooledHazardTrials);
}

// Hazard ratio of block length k against the pooled hazard, over all of its hops.
[[nodiscard]] inline float adaptive_block_ratio(const AdaptiveHopRates& rates, AdaptiveDraftLaw law,
                                                std::uint32_t k) {
    const auto l           = static_cast<std::uint32_t>(law);
    const std::uint32_t kk = std::clamp(k, 1U, kAdaptiveHopTableMaxK);
    float failures         = 0.0f;
    float expected         = 0.0f;
    for (std::uint32_t i = 0; i < kk; ++i) {
        failures += rates.failures[l][kk][i];
        expected += rates.trials[l][kk][i] * adaptive_pooled_hazard(rates, law, i);
    }
    return (failures + kAdaptiveBlockRatioPrior) / (expected + kAdaptiveBlockRatioPrior);
}

// Rejection hazard of hop i in a block of length k.
[[nodiscard]] inline float adaptive_hop_hazard(const AdaptiveHopRates& rates, AdaptiveDraftLaw law,
                                               std::uint32_t k, std::uint32_t hop) {
    const auto l           = static_cast<std::uint32_t>(law);
    const std::uint32_t kk = std::clamp(k, 1U, kAdaptiveHopTableMaxK);
    const std::uint32_t i  = std::min(hop, kk - 1U);
    const float prior =
        adaptive_block_ratio(rates, law, kk) * adaptive_pooled_hazard(rates, law, i);
    const float h = (rates.failures[l][kk][i] + kAdaptiveCellPriorTrials * prior) /
                    (rates.trials[l][kk][i] + kAdaptiveCellPriorTrials);
    return std::min(h, 1.0f);
}

[[nodiscard]] inline std::uint64_t adaptive_explore_hash(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

inline void adaptive_observe_round_time(AdaptiveRoundTimeState& st, std::uint32_t k, float seconds,
                                        std::uint32_t length_tokens) {
    if (k >= kAdaptiveTBins || !(seconds > 0.0f)) { return; }
    const float L = static_cast<float>(length_tokens);
    if (!(st.n[k] > 0.0f)) {
        st.n[k]      = 1.0f;
        st.mean_L[k] = L;
        st.mean_T[k] = seconds;
        st.Sxx[k]    = 0.0f;
        st.Sxy[k]    = 0.0f;
        return;
    }
    st.n[k] += 1.0f;
    const float dL = L - st.mean_L[k];
    const float dT = seconds - st.mean_T[k];
    st.mean_L[k] += dL / st.n[k];
    st.mean_T[k] += dT / st.n[k];
    st.Sxx[k] += dL * (L - st.mean_L[k]);
    st.Sxy[k] += dL * (seconds - st.mean_T[k]);
}

namespace detail {

[[nodiscard]] inline bool captured_contains(std::span<const std::uint32_t> ks, std::uint32_t k) {
    return std::find(ks.begin(), ks.end(), k) != ks.end();
}

[[nodiscard]] inline bool r_seen_at(const AdaptiveDraftState& state, std::uint32_t i) {
    return i < kAdaptiveMaxHops && ((state.r_seen >> i) & 1U) != 0;
}

[[nodiscard]] inline float r_mean(const AdaptiveDraftState& state, std::uint32_t i) {
    if (!r_seen_at(state, i)) { return 0.0f; }
    const float den = state.alpha[i] + state.beta[i];
    if (!(den > 0.0f)) { return 0.0f; }
    return state.alpha[i] / den;
}

// Content factor on hop i's rejection hazard; the prior inherits half of c_{i-1}'s departure.
[[nodiscard]] inline float hazard_factor(const AdaptiveDraftState& state, std::uint32_t i,
                                         float prior) {
    return (state.hazard_failures[i] + kAdaptiveHazardPrior * prior) /
           (state.hazard_expected[i] + kAdaptiveHazardPrior);
}

// Block length k, of which the row uses the first `extent` hops.
[[nodiscard]] inline float hop_expected_tokens(const AdaptiveDraftState& state,
                                               const AdaptiveHopRates& rates, std::uint32_t k,
                                               std::uint32_t extent) {
    float e               = 1.0f;
    float run             = 1.0f;
    float prior           = 1.0f;
    const std::uint32_t n = std::min(extent, kAdaptiveMaxHops);
    for (std::uint32_t i = 0; i < n; ++i) {
        const float c = hazard_factor(state, i, prior);
        const float h = adaptive_hop_hazard(rates, state.law, k, i);
        run *= std::clamp(1.0f - c * h, 0.0f, kAdaptiveMaxHopRate);
        e += run;
        prior = 1.0f + kAdaptiveHazardInherit * (c - 1.0f);
    }
    return e;
}

// MTP pooled model: E[Y] truncates at the first hop the request has not observed.
[[nodiscard]] inline float expected_tokens(const AdaptiveDraftState& state, std::uint32_t k) {
    float e               = 1.0f;
    float run             = 1.0f;
    const std::uint32_t n = std::min(k, kAdaptiveMaxHops);
    for (std::uint32_t i = 0; i < n; ++i) {
        if (!r_seen_at(state, i)) { return e; }
        run *= r_mean(state, i);
        e += run;
    }
    return e;
}

[[nodiscard]] inline float pooled_slope(const AdaptiveRoundTimeState& st) {
    float sxx = 0.0f;
    float sxy = 0.0f;
    for (std::uint32_t k = 0; k < kAdaptiveTBins; ++k) {
        if (st.n[k] >= 2.0f) {
            sxx += st.Sxx[k];
            sxy += st.Sxy[k];
        }
    }
    if (!(sxx > 1.0f)) { return 0.0f; }
    return sxy / sxx;
}

inline void t_lookup(const AdaptiveRoundTimeState* st, std::uint32_t k, std::uint32_t L, float& t,
                     bool& measured) {
    t        = 0.0f;
    measured = false;
    if (st == nullptr || k >= kAdaptiveTBins || !(st->n[k] > 0.0f)) { return; }
    const float c = pooled_slope(*st);
    t             = st->mean_T[k] + c * (static_cast<float>(L) - st->mean_L[k]);
    if (!(t > 0.0f)) { t = st->mean_T[k]; }
    measured = t > 0.0f;
}

[[nodiscard]] inline std::uint32_t clamp_to_budget(std::span<const std::uint32_t> captured,
                                                   std::uint32_t k, std::uint32_t budget) {
    k = std::min(k, budget);
    if (captured.empty()) { return k; }
    if (captured_contains(captured, k) && k <= budget) { return k; }
    std::uint32_t down = captured.front();
    for (std::uint32_t c : captured) {
        if (c <= budget) { down = c; }
    }
    if (down > budget) { return captured.front(); }
    return down;
}

[[nodiscard]] inline float row_sum_e(std::span<const AdaptiveDraftState* const> states,
                                     std::span<const std::uint32_t> row_cap, std::uint32_t k,
                                     const AdaptiveHopRates* rates) {
    float sum_e = 0.0f;
    for (std::size_t r = 0; r < states.size(); ++r) {
        const AdaptiveDraftState* st = states[r];
        if (st == nullptr) { continue; }
        const std::uint32_t kr = r < row_cap.size() ? std::min(k, row_cap[r]) : k;
        if (kr == 0) { continue; }
        // The batch drafts a block of length k; a capped row uses its first kr hops.
        sum_e +=
            rates != nullptr ? hop_expected_tokens(*st, *rates, k, kr) : expected_tokens(*st, kr);
    }
    return sum_e;
}

} // namespace detail

[[nodiscard]] inline float adaptive_t_hat(const AdaptiveRoundTimeState& st, std::uint32_t k,
                                          std::uint32_t length_tokens) {
    float t       = 0.0f;
    bool measured = false;
    detail::t_lookup(&st, k, length_tokens, t, measured);
    (void)measured;
    return t;
}

[[nodiscard]] inline bool adaptive_t_measured(const AdaptiveRoundTimeState& st, std::uint32_t k) {
    return k < kAdaptiveTBins && st.n[k] > 0.0f;
}

inline void adaptive_observe_hops(AdaptiveDraftState& state, std::uint32_t accepted,
                                  std::uint32_t drafted, float discount = kAdaptiveDiscount) {
    const std::uint32_t n = std::min(drafted, kAdaptiveMaxHops);
    for (std::uint32_t i = 0; i < n; ++i) {
        if (i > 0 && accepted <= i - 1U) { break; }
        const float x = accepted > i ? 1.0f : 0.0f;
        if (!detail::r_seen_at(state, i)) {
            state.alpha[i] = kAdaptiveBetaPrior;
            state.beta[i]  = kAdaptiveBetaPrior;
            state.r_seen |= static_cast<std::uint16_t>(1U << i);
        }
        state.alpha[i] = discount * state.alpha[i] + x;
        state.beta[i]  = discount * state.beta[i] + (1.0f - x);
    }
}

// Hop i is a trial only when hops < i were accepted. The request's content sums decay every
// round and accumulate the hazard in force before the round. An exploration round whose row
// drafted the full block also counts its trials in the global table.
inline void adaptive_observe_hazards(AdaptiveDraftState& state, AdaptiveHopRates& rates,
                                     std::uint32_t accepted, std::uint32_t drafted,
                                     std::uint32_t round_k, bool explored,
                                     float discount = kAdaptiveDiscount) {
    const std::uint32_t n          = std::min({drafted, accepted + 1U, kAdaptiveMaxHops});
    float hazard[kAdaptiveMaxHops] = {};
    for (std::uint32_t i = 0; i < n; ++i) {
        hazard[i] = adaptive_hop_hazard(rates, state.law, round_k, i);
    }
    if (explored && drafted == round_k && round_k <= kAdaptiveHopTableMaxK) {
        const auto l = static_cast<std::uint32_t>(state.law);
        for (std::uint32_t k = 0; k <= kAdaptiveHopTableMaxK; ++k) {
            for (std::uint32_t i = 0; i < kAdaptiveHopTableMaxK; ++i) {
                rates.failures[l][k][i] *= kAdaptiveCellDiscount;
                rates.trials[l][k][i] *= kAdaptiveCellDiscount;
            }
        }
        for (std::uint32_t i = 0; i < kAdaptiveHopTableMaxK; ++i) {
            rates.pooled_failures[l][i] *= kAdaptivePooledDiscount;
            rates.pooled_trials[l][i] *= kAdaptivePooledDiscount;
        }
        for (std::uint32_t i = 0; i < n; ++i) {
            const float x = accepted > i ? 0.0f : 1.0f;
            rates.failures[l][round_k][i] += x;
            rates.trials[l][round_k][i] += 1.0f;
            rates.pooled_failures[l][i] += x;
            rates.pooled_trials[l][i] += 1.0f;
        }
    }
    for (std::uint32_t i = 0; i < kAdaptiveMaxHops; ++i) {
        state.hazard_failures[i] *= discount;
        state.hazard_expected[i] *= discount;
    }
    for (std::uint32_t i = 0; i < n; ++i) {
        state.hazard_failures[i] += accepted > i ? 0.0f : 1.0f;
        state.hazard_expected[i] += hazard[i];
    }
}

inline void adaptive_record_round(AdaptiveDraftState& state, std::uint32_t accepted,
                                  std::uint32_t drafted, std::uint32_t round_k,
                                  AdaptiveHopRates* rates, bool explored) {
    if (rates != nullptr) {
        adaptive_observe_hazards(state, *rates, accepted, drafted, round_k, explored);
    } else {
        adaptive_observe_hops(state, accepted, drafted);
    }
    state.observed += 1;
    if (!explored) { state.rounds_at_k += 1; } // an exploration override is not the live k
    if (round_k < 16) { state.rounds_hist[round_k] += 1; }
}

// One round in kAdaptiveExploreEvery overrides the picked k with a uniform captured k <= cap_k,
// once every such k has T measured for this batch size. Returns 0 for an ordinary round.
[[nodiscard]] inline std::uint32_t adaptive_explore_k(AdaptiveHopRates& rates,
                                                      std::span<const std::uint32_t> captured_ks,
                                                      const AdaptiveRoundTimeState& round_time,
                                                      std::uint32_t cap_k) {
    std::uint32_t candidates[kAdaptiveTBins] = {};
    std::uint32_t count                      = 0;
    for (std::uint32_t k : captured_ks) {
        if (k > cap_k || k >= kAdaptiveTBins) { continue; }
        if (!adaptive_t_measured(round_time, k)) { return 0; }
        candidates[count++] = k;
    }
    if (count < 2) { return 0; }
    const std::uint64_t h = adaptive_explore_hash(rates.decisions++);
    if ((h % kAdaptiveExploreEvery) != 0) { return 0; }
    return candidates[(h >> 32) % count];
}

inline void adaptive_assign_live_k(std::span<AdaptiveDraftState*> states, std::uint32_t k) {
    for (AdaptiveDraftState* state : states) {
        if (state == nullptr) { continue; }
        if (state->live_k != k) { state->rounds_at_k = 0; }
        state->live_k = k;
    }
}

// Round time need not grow smoothly with k: verify routes and tiles change with T=W*C (a C=4
// k=4 round is cheaper than k=3 or k=5). Shorter arms therefore bound nothing about an
// unmeasured arm. Each captured k within the cap is measured once per batch size (T is
// engine-global), then the policy takes argmax E[Y]/T.
[[nodiscard]] inline std::uint32_t
adaptive_select_k(const AdaptiveDraftConfig& cfg, std::span<const AdaptiveDraftState* const> states,
                  std::span<const std::uint32_t> row_cap, std::uint32_t cap_k,
                  std::uint32_t live_k) {
    if (cfg.captured_ks.empty() || states.empty()) { return cap_k; }

    for (std::uint32_t k : cfg.captured_ks) {
        if (k <= cap_k && (cfg.round_time == nullptr || !adaptive_t_measured(*cfg.round_time, k))) {
            return k;
        }
    }

    std::uint32_t best = 0;
    float best_s       = -1.0f;
    for (std::uint32_t k : cfg.captured_ks) {
        if (k > cap_k) { continue; }
        float t       = 0.0f;
        bool measured = false;
        detail::t_lookup(cfg.round_time, k, cfg.length_tokens, t, measured);
        if (!measured) { continue; }
        const float e = detail::row_sum_e(states, row_cap, k, cfg.hop_rates);
        if (!(e > 0.0f)) { continue; }
        const float t_eff = t + ((live_k != 0 && k != live_k) ? cfg.switch_seconds : 0.0f);
        const float sc    = e / t_eff;
        if (sc > best_s) {
            best_s = sc;
            best   = k;
        }
    }
    return best != 0 ? best : cfg.captured_ks.front();
}

[[nodiscard]] inline std::uint32_t adaptive_select_batch_k(
    std::span<const AdaptiveDraftState* const> states, std::span<const std::uint32_t> row_k,
    std::span<const std::uint32_t> captured_ks, const AdaptiveRoundTimeState* round_time,
    std::uint32_t length_tokens, std::uint32_t live_k, AdaptiveHopRates* rates) {
    AdaptiveDraftConfig cfg;
    cfg.captured_ks           = captured_ks;
    cfg.round_time            = round_time;
    cfg.length_tokens         = length_tokens;
    cfg.hop_rates             = rates;
    const std::uint32_t cap_k = adaptive_batch_k(row_k, captured_ks);
    return adaptive_select_k(cfg, states, row_k, cap_k, live_k);
}

[[nodiscard]] inline std::uint32_t
adaptive_batch_next(AdaptiveBatchKState& batch, std::span<const AdaptiveDraftState* const> states,
                    std::span<const std::uint32_t> row_k,
                    std::span<const std::uint32_t> captured_ks,
                    const AdaptiveRoundTimeState* round_time, std::uint32_t length_tokens,
                    AdaptiveHopRates* rates) {
    const std::uint32_t next = adaptive_select_batch_k(states, row_k, captured_ks, round_time,
                                                       length_tokens, batch.live_k, rates);
    if (batch.live_k != 0 && next == batch.live_k) {
        batch.rounds_at_k += 1;
    } else {
        batch.live_k      = next;
        batch.rounds_at_k = 0;
    }
    return batch.live_k;
}

inline std::uint32_t adaptive_draft_next(const AdaptiveDraftConfig& cfg, AdaptiveDraftState& state,
                                         std::uint32_t accepted, std::uint32_t drafted,
                                         std::uint32_t budget_extent, std::uint32_t round_k,
                                         bool explored) {
    if (cfg.captured_ks.empty()) { return state.live_k; }
    if (drafted == 0) { return std::min(state.live_k, budget_extent); }

    adaptive_record_round(state, accepted, drafted, round_k, cfg.hop_rates, explored);

    const AdaptiveDraftState* ptr = &state;
    const std::uint32_t row_cap[] = {budget_extent};
    const std::uint32_t cap_k =
        detail::clamp_to_budget(cfg.captured_ks, budget_extent, budget_extent);
    const std::uint32_t picked =
        adaptive_select_k(cfg, std::span<const AdaptiveDraftState* const>(&ptr, 1),
                          std::span<const std::uint32_t>(row_cap, 1), cap_k, state.live_k);
    const std::uint32_t next = detail::clamp_to_budget(cfg.captured_ks, picked, budget_extent);
    if (next != state.live_k) {
        state.live_k      = next;
        state.rounds_at_k = 0;
    }
    return state.live_k;
}

} // namespace ninfer::targets::qwen3_6
