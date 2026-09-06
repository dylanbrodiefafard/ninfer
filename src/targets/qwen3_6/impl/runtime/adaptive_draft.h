#pragma once

// Family host policy for adaptive draft length. No CUDA.
//
// Objective: match the best constant-k policy in {3,4,5}. Mixing k forks the
// greedy DFlash/MTP path, so this is not a mixing bandit.
//
// Y(k) = 1 + sum_{i<k} q_i with q_i = prod_{j<=i} r_j and
// r_i = P(accepted > i | accepted > i-1). r_i is a discounted Beta, updated
// only when the prefix reached i. Unseen r_i are not invented in E[Y]; they
// appear only as r=1 in an optimistic bound used to drop dominated k.
//
// T(k,C,L) = a_{C,k} + c_C L from online least squares (shared slope, per-k
// intercept). Unmeasured T(k) is extrapolated as max(T(k-1), 2 T(k-1)-T(k-2))
// and never schedules k just because a length bin is empty. k is probed at
// most once when that bound still thinks it can win.
//
// Lock argmax E[Y]/T among surviving k. No Thompson, no dwell, no hop CUSUM.
// Switching k adds 1 ms to that arm's T. Ties keep the smaller k.

#include "ninfer/types.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::targets::qwen3_6 {

inline constexpr float kAdaptiveEwma              = 32.0f;
inline constexpr float kAdaptiveEwmaAlpha         = 2.0f / (kAdaptiveEwma + 1.0f);
inline constexpr float kAdaptiveDiscount          = 1.0f - kAdaptiveEwmaAlpha;
inline constexpr float kAdaptiveBetaPrior         = 1.0f;
inline constexpr std::uint32_t kAdaptiveTBins     = 16;
inline constexpr float kAdaptiveSwitchSeconds     = 0.001f;

struct AdaptiveDraftState {
    std::uint32_t live_k      = 0;
    std::uint32_t rounds_at_k = 0;
    std::uint32_t observed    = 0;
    float alpha[5]            = {};
    float beta[5]             = {};
    std::uint8_t r_seen       = 0;
    std::uint64_t rounds_hist[16] = {};
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
};

[[nodiscard]] inline std::vector<std::uint32_t>
adaptive_draft_ks(SpeculativeBackend backend, std::uint32_t n, bool adaptive) {
    if (!adaptive || backend == SpeculativeBackend::None || n == 0) { return {n}; }
    std::vector<std::uint32_t> out;
    if (backend == SpeculativeBackend::Mtp) {
        for (std::uint32_t k = 3; k <= 5 && k <= n; ++k) { out.push_back(k); }
        return out.empty() ? std::vector<std::uint32_t>{n} : out;
    }
    if (n >= 5) { return {3, 4, 5}; }
    return {n};
}

[[nodiscard]] inline std::uint32_t
adaptive_k_index(std::span<const std::uint32_t> captured_ks, std::uint32_t k) {
    for (std::uint32_t i = 0; i < captured_ks.size(); ++i) {
        if (captured_ks[i] == k) { return i; }
    }
    return 0;
}

[[nodiscard]] inline std::uint32_t adaptive_k_stride(std::uint32_t max_concurrency,
                                                     std::uint32_t max_planned_topology) {
    return max_concurrency * (1U + max_planned_topology);
}

[[nodiscard]] inline std::uint32_t
adaptive_topology_class(std::uint32_t k_index, std::uint32_t k_stride,
                        std::uint32_t planned_topology, std::uint32_t max_concurrency,
                        std::uint32_t batch_size) {
    return k_index * k_stride + planned_topology * max_concurrency + (batch_size - 1U);
}

[[nodiscard]] inline std::uint32_t
adaptive_snap_captured_k(std::span<const std::uint32_t> captured_ks, std::uint32_t k) {
    if (captured_ks.empty()) { return k; }
    for (std::uint32_t c : captured_ks) {
        if (c >= k) { return c; }
    }
    return captured_ks.back();
}

[[nodiscard]] inline std::uint32_t
adaptive_batch_k(std::span<const std::uint32_t> row_k, std::span<const std::uint32_t> captured_ks) {
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

inline void seed_adaptive_draft_state(AdaptiveDraftState& state, std::uint32_t live_k) {
    state        = {};
    state.live_k = live_k;
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
    return i < 5 && ((state.r_seen >> i) & 1U) != 0;
}

[[nodiscard]] inline float r_mean(const AdaptiveDraftState& state, std::uint32_t i) {
    if (!r_seen_at(state, i)) { return 0.0f; }
    const float den = state.alpha[i] + state.beta[i];
    if (!(den > 0.0f)) { return 0.0f; }
    return state.alpha[i] / den;
}

[[nodiscard]] inline float expected_tokens(const AdaptiveDraftState& state, std::uint32_t k) {
    float e               = 1.0f;
    float run             = 1.0f;
    const std::uint32_t n = std::min(k, 5U);
    for (std::uint32_t i = 0; i < n; ++i) {
        if (!r_seen_at(state, i)) { return e; }
        run *= r_mean(state, i);
        e += run;
    }
    return e;
}

// Unseen r_i after an identified prefix are 1 (q_extra = last q). No data at
// hop 0 stays E[Y]=1 so a cold start does not invent a chain of successes.
[[nodiscard]] inline float expected_tokens_optimistic(const AdaptiveDraftState& state,
                                                      std::uint32_t k) {
    float e               = 1.0f;
    float run             = 1.0f;
    const std::uint32_t n = std::min(k, 5U);
    for (std::uint32_t i = 0; i < n; ++i) {
        if (!r_seen_at(state, i)) {
            if (i == 0) { return 1.0f; }
            e += run * static_cast<float>(n - i);
            return e;
        }
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
    if (st == nullptr || k >= kAdaptiveTBins) { return; }
    const float Lf = static_cast<float>(L);
    const float c  = pooled_slope(*st);
    if (st->n[k] > 0.0f) {
        t = st->mean_T[k] + c * (Lf - st->mean_L[k]);
        if (!(t > 0.0f)) { t = st->mean_T[k]; }
        measured = t > 0.0f;
        return;
    }
    if (k == 0) { return; }
    float t_prev = 0.0f;
    bool m_prev  = false;
    t_lookup(st, k - 1U, L, t_prev, m_prev);
    if (!(t_prev > 0.0f)) { return; }
    t = t_prev;
    if (k >= 2) {
        float t_prev2 = 0.0f;
        bool m2       = false;
        t_lookup(st, k - 2U, L, t_prev2, m2);
        if (t_prev2 > 0.0f) { t = std::max(t_prev, 2.0f * t_prev - t_prev2); }
    }
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
                                     bool optimistic) {
    float sum_e = 0.0f;
    for (std::size_t r = 0; r < states.size(); ++r) {
        const AdaptiveDraftState* st = states[r];
        if (st == nullptr) { continue; }
        const std::uint32_t kr = r < row_cap.size() ? std::min(k, row_cap[r]) : k;
        if (kr == 0) { continue; }
        sum_e += optimistic ? expected_tokens_optimistic(*st, kr) : expected_tokens(*st, kr);
    }
    return sum_e;
}

} // namespace detail

[[nodiscard]] inline float adaptive_t_hat(const AdaptiveRoundTimeState& st, std::uint32_t k,
                                          std::uint32_t length_tokens) {
    float t        = 0.0f;
    bool measured  = false;
    detail::t_lookup(&st, k, length_tokens, t, measured);
    (void)measured;
    return t;
}

[[nodiscard]] inline bool adaptive_t_measured(const AdaptiveRoundTimeState& st, std::uint32_t k) {
    return k < kAdaptiveTBins && st.n[k] > 0.0f;
}

inline void adaptive_observe_hops(AdaptiveDraftState& state, std::uint32_t accepted,
                                  std::uint32_t drafted, float discount = kAdaptiveDiscount) {
    const std::uint32_t n = std::min(drafted, 5U);
    for (std::uint32_t i = 0; i < n; ++i) {
        if (i > 0 && accepted <= i - 1U) { break; }
        const float x = accepted > i ? 1.0f : 0.0f;
        if (!detail::r_seen_at(state, i)) {
            state.alpha[i] = kAdaptiveBetaPrior;
            state.beta[i]  = kAdaptiveBetaPrior;
            state.r_seen |= static_cast<std::uint8_t>(1U << i);
        }
        state.alpha[i] = discount * state.alpha[i] + x;
        state.beta[i]  = discount * state.beta[i] + (1.0f - x);
    }
}

inline void adaptive_record_round(AdaptiveDraftState& state, std::uint32_t accepted,
                                  std::uint32_t drafted, std::uint32_t round_k) {
    adaptive_observe_hops(state, accepted, drafted);
    state.observed += 1;
    state.rounds_at_k += 1;
    if (round_k < 16) { state.rounds_hist[round_k] += 1; }
}

inline void adaptive_assign_live_k(std::span<AdaptiveDraftState*> states, std::uint32_t k) {
    for (AdaptiveDraftState* state : states) {
        if (state == nullptr) { continue; }
        if (state->live_k != k) { state->rounds_at_k = 0; }
        state->live_k = k;
    }
}

[[nodiscard]] inline std::uint32_t
adaptive_select_k(const AdaptiveDraftConfig& cfg,
                  std::span<const AdaptiveDraftState* const> states,
                  std::span<const std::uint32_t> row_cap, std::uint32_t cap_k,
                  std::uint32_t live_k) {
    if (cfg.captured_ks.empty() || states.empty()) { return cap_k; }

    std::uint32_t best = 0;
    float best_s       = -1.0f;
    for (std::uint32_t k : cfg.captured_ks) {
        if (k > cap_k) { continue; }
        float t       = 0.0f;
        bool measured = false;
        detail::t_lookup(cfg.round_time, k, cfg.length_tokens, t, measured);
        if (!measured || !(t > 0.0f)) { continue; }
        const float e = detail::row_sum_e(states, row_cap, k, false);
        if (!(e > 0.0f)) { continue; }
        const float t_eff =
            t + ((live_k != 0 && k != live_k) ? cfg.switch_seconds : 0.0f);
        if (!(t_eff > 0.0f)) { continue; }
        const float sc = e / t_eff;
        if (sc > best_s) {
            best_s = sc;
            best   = k;
        }
    }

    std::uint32_t probe = 0;
    for (std::uint32_t k : cfg.captured_ks) {
        if (k > cap_k) { continue; }
        float t       = 0.0f;
        bool measured = false;
        detail::t_lookup(cfg.round_time, k, cfg.length_tokens, t, measured);
        if (!measured) {
            probe = k;
            break;
        }
    }
    if (probe == 0) {
        return best != 0 ? best : cfg.captured_ks.front();
    }

    float t_lb       = 0.0f;
    bool probe_meas  = false;
    detail::t_lookup(cfg.round_time, probe, cfg.length_tokens, t_lb, probe_meas);
    if (!(t_lb > 0.0f) && best != 0) {
        bool dummy = false;
        detail::t_lookup(cfg.round_time, best, cfg.length_tokens, t_lb, dummy);
    }
    if (!(t_lb > 0.0f)) { return probe; }

    const float e_opt = detail::row_sum_e(states, row_cap, probe, true);
    const float t_eff =
        t_lb + ((live_k != 0 && probe != live_k) ? cfg.switch_seconds : 0.0f);
    if (!(e_opt > 0.0f) || !(t_eff > 0.0f)) {
        return best != 0 ? best : probe;
    }
    const float sc_opt = e_opt / t_eff;
    if (best == 0 || sc_opt > best_s) { return probe; }
    return best;
}

[[nodiscard]] inline std::uint32_t
adaptive_select_batch_k(std::span<const AdaptiveDraftState* const> states,
                        std::span<const std::uint32_t> row_k,
                        std::span<const std::uint32_t> captured_ks,
                        const AdaptiveRoundTimeState* round_time, std::uint32_t length_tokens,
                        std::uint32_t live_k) {
    AdaptiveDraftConfig cfg;
    cfg.captured_ks   = captured_ks;
    cfg.round_time    = round_time;
    cfg.length_tokens = length_tokens;
    const std::uint32_t cap_k = adaptive_batch_k(row_k, captured_ks);
    return adaptive_select_k(cfg, states, row_k, cap_k, live_k);
}

[[nodiscard]] inline std::uint32_t
adaptive_batch_next(AdaptiveBatchKState& batch, std::span<const AdaptiveDraftState* const> states,
                    std::span<const std::uint32_t> row_k, std::span<const std::uint32_t> captured_ks,
                    const AdaptiveRoundTimeState* round_time, std::uint32_t length_tokens) {
    const std::uint32_t next = adaptive_select_batch_k(states, row_k, captured_ks, round_time,
                                                       length_tokens, batch.live_k);
    if (batch.live_k != 0 && next == batch.live_k) { batch.rounds_at_k += 1; }
    else {
        batch.live_k      = next;
        batch.rounds_at_k = 0;
    }
    return batch.live_k;
}

inline std::uint32_t adaptive_draft_next(const AdaptiveDraftConfig& cfg, AdaptiveDraftState& state,
                                         std::uint32_t accepted, std::uint32_t drafted,
                                         std::uint32_t budget_extent, std::uint32_t round_k) {
    if (cfg.captured_ks.empty()) { return state.live_k; }
    if (drafted == 0) { return std::min(state.live_k, budget_extent); }

    adaptive_record_round(state, accepted, drafted, round_k);

    const AdaptiveDraftState* ptr = &state;
    const std::uint32_t row_cap[] = {budget_extent};
    const std::uint32_t cap_k =
        detail::clamp_to_budget(cfg.captured_ks, budget_extent, budget_extent);
    const std::uint32_t picked = adaptive_select_k(
        cfg, std::span<const AdaptiveDraftState* const>(&ptr, 1),
        std::span<const std::uint32_t>(row_cap, 1), cap_k, state.live_k);
    const std::uint32_t next = detail::clamp_to_budget(cfg.captured_ks, picked, budget_extent);
    if (next != state.live_k) {
        state.live_k      = next;
        state.rounds_at_k = 0;
    }
    return state.live_k;
}

} // namespace ninfer::targets::qwen3_6
