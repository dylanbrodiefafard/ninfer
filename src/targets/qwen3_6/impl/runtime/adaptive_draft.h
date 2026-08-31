#pragma once

// Family host policy for adaptive draft length. No CUDA. Variants supply captured_ks and T(K).

#include "ninfer/types.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <span>
#include <vector>

namespace ninfer::targets::qwen3_6 {

inline constexpr float kAdaptiveEwma           = 32.0f;
inline constexpr std::uint32_t kAdaptiveWarmupRounds = 8;
inline constexpr std::uint32_t kAdaptiveFirstRemoveWarmup = 32;
inline constexpr std::uint32_t kAdaptiveDflashFirstRemoveWarmup = 8;
inline constexpr std::uint32_t kAdaptiveK5WaiveWarmup = 8;
inline constexpr std::uint32_t kAdaptiveMinDwell     = 8;
inline constexpr float kAdaptiveDeltaAdd       = 0.05f;
inline constexpr float kAdaptiveDeltaRemove    = 0.02f;
inline constexpr float kAdaptiveDropSlotMax    = 0.30f;
inline constexpr float kAdaptiveClimbSlotMin   = 0.35f;
inline constexpr float kAdaptiveHighP0MinK     = 0.65f;
inline constexpr float kAdaptiveDropTo3Max     = 0.08f;
inline constexpr float kAdaptiveEwmaAlpha      = 2.0f / (kAdaptiveEwma + 1.0f);
inline constexpr float kAdaptiveSlotEps        = 1.0e-6f;
inline constexpr float kAdaptivePbarPrior      = 0.5f;
inline constexpr float kAdaptiveE5Campaign     = 2.94f;
inline constexpr float kAdaptiveE7Campaign     = 3.21f;

// C=1 NVFP4-KV relative round time, Qwen3.8-27B, RTX 5090. Index is K.
// MTP: ninfer_bench pp2048+tg256 and AIME 4096 agree (T3=13.53 ms).
// DFlash chain k=3/4/5 from AIME 4096. Packed-tree k=6/7 W=12 after origin opt:
// AIME T_ms 19.06 / 19.09 vs chain k=5 15.24 → rel 1.25 (was T[7]=1.43). High-accept
// CUDA/Python still lose tok/s to chain k=4/5, so the adaptive live set stays {3,4,5}.
inline constexpr float kAdaptiveMtpT[6]    = {0.0f, 0.0f, 0.0f, 1.00f, 1.10f, 1.16f};
inline constexpr float kAdaptiveDflashT[8] = {0.0f, 0.0f, 0.0f, 0.91f, 0.97f, 1.00f, 1.25f, 1.25f};
// C=2 NVFP4-KV relative round time from frozen homogeneous decode-saturation.
// DFlash T[3]=0.83: C=2 AIME/story/code 0.82–0.83. T[4]=0.93: measured 0.91–0.93
// (old 0.89 over-picked k=4 on mixed C=2). T[5]=1.00.
inline constexpr float kAdaptiveMtpC2T[6]    = {0.0f, 0.0f, 0.0f, 1.00f, 1.10f, 1.22f};
inline constexpr float kAdaptiveDflashC2T[8] = {0.0f, 0.0f, 0.0f, 0.83f, 0.93f, 1.00f, 0.0f, 0.0f};
// C=3 DFlash: T[4] collapses toward T[5] (homo 0.97–0.98). T[3]=0.86 (0.85–0.87).
inline constexpr float kAdaptiveDflashC3T[8] = {0.0f, 0.0f, 0.0f, 0.86f, 0.98f, 1.00f, 0.0f, 0.0f};
// C≥4 DFlash: dialogue C=4/C=8 + python C=4. T[4] retreats from the C=3 peak
// (C=4 0.94, C=8 0.93). T[3] 0.85 (C=4) / 0.82 (C=8); table uses C=4.
inline constexpr float kAdaptiveDflashC4T[8] = {0.0f, 0.0f, 0.0f, 0.85f, 0.94f, 1.00f, 0.0f, 0.0f};

[[nodiscard]] inline std::span<const float>
adaptive_dflash_round_time(std::uint32_t batch_size) {
    if (batch_size >= 4U) { return std::span<const float>(kAdaptiveDflashC4T); }
    if (batch_size >= 3U) { return std::span<const float>(kAdaptiveDflashC3T); }
    return std::span<const float>(kAdaptiveDflashC2T);
}

struct AdaptiveDraftState {
    std::uint32_t live_k      = 0;
    std::uint32_t rounds_at_k = 0;
    std::uint32_t observed    = 0;
    float s[5]                = {};
    std::uint8_t s_seen       = 0; // bit i: s[i] updated at least once; chain slots 0..4 only
    std::uint64_t rounds_hist[16] = {};
};

struct AdaptiveDraftConfig {
    std::span<const std::uint32_t> captured_ks;
    std::span<const float> round_time; // index = K; unused slots 0
    float delta_add                     = kAdaptiveDeltaAdd;
    float delta_remove                  = kAdaptiveDeltaRemove;
    std::uint32_t warmup_rounds         = kAdaptiveWarmupRounds;
    std::uint32_t first_remove_warmup   = kAdaptiveFirstRemoveWarmup;
    std::uint32_t min_dwell             = kAdaptiveMinDwell;
    float drop_slot_max                 = kAdaptiveDropSlotMax;
    float drop_to_3_max                 = kAdaptiveDropSlotMax;
    float climb_slot_min                = kAdaptiveClimbSlotMin;
    float high_p0_min_k                 = kAdaptiveHighP0MinK;
    float alpha                         = kAdaptiveEwmaAlpha;
};

[[nodiscard]] inline std::vector<std::uint32_t>
adaptive_draft_ks(SpeculativeBackend backend, std::uint32_t n, bool adaptive) {
    if (!adaptive || backend == SpeculativeBackend::None || n == 0) { return {n}; }
    std::vector<std::uint32_t> out;
    if (backend == SpeculativeBackend::Mtp) {
        for (std::uint32_t k = 3; k <= 5 && k <= n; ++k) { out.push_back(k); }
        return out.empty() ? std::vector<std::uint32_t>{n} : out;
    }
    // DFlash adaptive live set is chain k=3/4/5. Packed-tree k=6/7 is the native
    // window when N is frozen at 6 or 7 without --adaptive-draft. N=6 is not a
    // +1 chain step, so adaptive N=6 stays frozen {6} until tree is in the live set.
    if (n == 6) { return {n}; }
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

// Frozen k_index=0 → planned*C+(B-1), identical to today.
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

namespace detail {

[[nodiscard]] inline bool captured_contains(std::span<const std::uint32_t> ks, std::uint32_t k) {
    return std::find(ks.begin(), ks.end(), k) != ks.end();
}

[[nodiscard]] inline float lookup_t(std::span<const float> round_time, std::uint32_t k) {
    if (k >= round_time.size()) { return 0.0f; }
    return round_time[k];
}

[[nodiscard]] inline float mean_conditional_p(const AdaptiveDraftState& state) {
    float sum   = 0.0f;
    int count   = 0;
    for (int j = 1; j < 5; ++j) {
        const bool have = ((state.s_seen >> static_cast<unsigned>(j)) & 1U) != 0 &&
                          ((state.s_seen >> static_cast<unsigned>(j - 1)) & 1U) != 0;
        if (!have || !(state.s[j - 1] > kAdaptiveSlotEps)) { continue; }
        sum += state.s[j] / state.s[j - 1];
        ++count;
    }
    return count > 0 ? sum / static_cast<float>(count) : kAdaptivePbarPrior;
}

[[nodiscard]] inline float last_conditional_p(const AdaptiveDraftState& state) {
    for (int j = 4; j >= 1; --j) {
        const bool have = ((state.s_seen >> static_cast<unsigned>(j)) & 1U) != 0 &&
                          ((state.s_seen >> static_cast<unsigned>(j - 1)) & 1U) != 0;
        if (!have || !(state.s[j - 1] > kAdaptiveSlotEps)) { continue; }
        return state.s[j] / state.s[j - 1];
    }
    return kAdaptivePbarPrior;
}

// Unseen tail uses the more conservative of mean vs last observed conditional.
[[nodiscard]] inline float geometric_pbar(const AdaptiveDraftState& state) {
    return std::min(mean_conditional_p(state), last_conditional_p(state));
}

[[nodiscard]] inline float slot_s(const AdaptiveDraftState& state, int i, float pbar) {
    if (i < 0) { return 1.0f; }
    if (i < 5 && ((state.s_seen >> static_cast<unsigned>(i)) & 1U) != 0) { return state.s[i]; }
    if (i == 0) { return kAdaptivePbarPrior; }
    return slot_s(state, i - 1, pbar) * pbar;
}

[[nodiscard]] inline float expected_chain(const AdaptiveDraftState& state, std::uint32_t k) {
    const float pbar = geometric_pbar(state);
    float e          = 1.0f;
    const std::uint32_t n = std::min(k, 5U);
    for (std::uint32_t i = 0; i < n; ++i) { e += slot_s(state, static_cast<int>(i), pbar); }
    if (k > 5) {
        float s = slot_s(state, 4, pbar);
        for (std::uint32_t i = 5; i < k; ++i) {
            s *= pbar;
            e += s;
        }
    }
    return e;
}

[[nodiscard]] inline float expected_tokens(const AdaptiveDraftConfig& cfg,
                                           const AdaptiveDraftState& state, std::uint32_t k) {
    (void)cfg;
    return expected_chain(state, k);
}

[[nodiscard]] inline float score_k(const AdaptiveDraftConfig& cfg, const AdaptiveDraftState& state,
                                   std::uint32_t k) {
    const float t = lookup_t(cfg.round_time, k);
    if (!(t > 0.0f)) { return 0.0f; }
    return expected_tokens(cfg, state, k) / t;
}

inline void update_ewma(AdaptiveDraftState& state, std::uint32_t accepted, std::uint32_t drafted,
                        float alpha) {
    const std::uint32_t n = std::min(drafted, 5U);
    for (std::uint32_t i = 0; i < n; ++i) {
        const float obs = accepted > i ? 1.0f : 0.0f;
        if (((state.s_seen >> i) & 1U) == 0) {
            state.s[i] = kAdaptivePbarPrior;
            state.s_seen |= static_cast<std::uint8_t>(1U << i);
        }
        state.s[i] = (1.0f - alpha) * state.s[i] + alpha * obs;
    }
}

[[nodiscard]] inline std::uint32_t neighbor(std::span<const std::uint32_t> captured,
                                            std::uint32_t live, int delta) {
    if (captured.empty()) { return live; }
    const std::uint32_t want = static_cast<std::uint32_t>(static_cast<int>(live) + delta);
    return captured_contains(captured, want) ? want : live;
}

[[nodiscard]] inline bool slot_seen(const AdaptiveDraftState& state, std::uint32_t i) {
    return i < 5 && ((state.s_seen >> i) & 1U) != 0;
}

// Invert the p̄=0.5 EWMA origin so dead/hot thresholds apply to the implied rate.
// After 32 rounds a raw CUDA s[4]≈0.31 still sits above drop_slot_max=0.30; unbiased
// recovers μ≈0.28. A single first Bernoulli is also unbiased back to 0 or 1.
[[nodiscard]] inline float ewma_rate(float s, std::uint32_t n) {
    if (n == 0) { return s; }
    const float w = std::pow(1.0f - kAdaptiveEwmaAlpha, static_cast<float>(n));
    const float den = 1.0f - w;
    if (!(den > 1.0e-3f)) { return s; }
    return (s - kAdaptivePbarPrior * w) / den;
}

// Extra draft position of live K is s[K-1]. Unknown → not dead (do not drop).
[[nodiscard]] inline bool extra_slot_dead(const AdaptiveDraftState& state, std::uint32_t live_k,
                                          float drop_slot_max) {
    if (live_k == 0) { return false; }
    const std::uint32_t i = live_k - 1U;
    return slot_seen(state, i) && ewma_rate(state.s[i], state.observed) < drop_slot_max;
}

[[nodiscard]] inline bool last_slot_hot(const AdaptiveDraftState& state, std::uint32_t live_k,
                                        float climb_slot_min) {
    if (live_k == 0) { return false; }
    const std::uint32_t i = live_k - 1U;
    return slot_seen(state, i) && ewma_rate(state.s[i], state.observed) > climb_slot_min;
}

[[nodiscard]] inline std::uint32_t p0_floor_k(const AdaptiveDraftState& state,
                                              std::span<const std::uint32_t> captured,
                                              float high_p0) {
    if (!slot_seen(state, 0) || !(state.s[0] > high_p0)) { return captured.empty() ? 0 : captured.front(); }
    return captured_contains(captured, 4) ? 4U : captured.front();
}

} // namespace detail

// C>=2: pick captured K maximizing Σ_row E_row(K) / T_B(K). Ties keep the lower K.
// Candidate K cannot exceed max(row_k) (budget/live cap). B=1 must not call this.
// Unobserved p̄=0.5 prefers min T (k=3) and then extra slots never get drafted.
// Until every row has warmup observations, execute max(row_k).
[[nodiscard]] inline std::uint32_t
adaptive_batch_k_sum_score(std::span<const AdaptiveDraftState* const> states,
                           std::span<const std::uint32_t> row_k,
                           std::span<const std::uint32_t> captured_ks,
                           std::span<const float> round_time) {
    if (states.size() < 2 || captured_ks.empty()) {
        return adaptive_batch_k(row_k, captured_ks);
    }
    const std::uint32_t cap_k = adaptive_batch_k(row_k, captured_ks);
    for (const AdaptiveDraftState* state : states) {
        if (state != nullptr && state->observed < kAdaptiveWarmupRounds) {
            return cap_k;
        }
    }
    AdaptiveDraftConfig cfg;
    cfg.captured_ks = captured_ks;
    cfg.round_time  = round_time;
    std::uint32_t best = captured_ks.front();
    float best_score   = -1.0f;
    bool any           = false;
    for (std::uint32_t k : captured_ks) {
        if (k > cap_k) { continue; }
        const float t = detail::lookup_t(round_time, k);
        if (!(t > 0.0f)) { continue; }
        float sum_e = 0.0f;
        for (const AdaptiveDraftState* state : states) {
            if (state == nullptr) { continue; }
            sum_e += detail::expected_tokens(cfg, *state, k);
        }
        const float score = sum_e / t;
        if (!any || score > best_score) {
            best_score = score;
            best       = k;
            any        = true;
        }
    }
    return any ? best : cap_k;
}

// MTP attractor 4 (AIME peak). DFlash attractor 5 (AIME peak). Frozen |K|=1 keeps N.
[[nodiscard]] inline std::uint32_t adaptive_seed_k(std::span<const std::uint32_t> captured_ks,
                                                   SpeculativeBackend backend) {
    if (captured_ks.empty()) { return 0; }
    if (captured_ks.size() == 1) { return captured_ks[0]; }
    const std::uint32_t want = backend == SpeculativeBackend::Mtp ? 4U : 5U;
    if (detail::captured_contains(captured_ks, want)) { return want; }
    std::uint32_t best = captured_ks.front();
    for (std::uint32_t k : captured_ks) {
        if (k <= want) { best = k; }
    }
    return best;
}

inline void seed_adaptive_draft_state(AdaptiveDraftState& state, std::uint32_t live_k) {
    state        = {};
    state.live_k = live_k;
}

[[nodiscard]] inline std::uint32_t
adaptive_draft_next(const AdaptiveDraftConfig& cfg, AdaptiveDraftState& state,
                    std::uint32_t accepted, std::uint32_t drafted, std::uint32_t budget_extent,
                    std::uint32_t round_k) {
    if (cfg.captured_ks.empty()) { return state.live_k; }
    const std::uint32_t cap_max = cfg.captured_ks.back();
    if (drafted == 0) { return std::min(state.live_k, budget_extent); }

    detail::update_ewma(state, accepted, drafted, cfg.alpha);
    state.observed += 1;
    state.rounds_at_k += 1;
    if (round_k < 16) { state.rounds_hist[round_k] += 1; }

    std::uint32_t next = state.live_k;
    const bool dwell_ok = state.rounds_at_k >= cfg.min_dwell;
    const bool can_add  = dwell_ok && state.observed >= cfg.warmup_rounds;
    const bool can_remove =
        dwell_ok && state.observed >= cfg.first_remove_warmup;
    if (can_add || can_remove) {
        const float live_score = detail::score_k(cfg, state, state.live_k);
        const std::uint32_t k_hi = detail::neighbor(cfg.captured_ks, state.live_k, 1);
        const std::uint32_t k_lo = detail::neighbor(cfg.captured_ks, state.live_k, -1);
        const float hi_score     = detail::score_k(cfg, state, k_hi);
        const bool extra_hot =
            detail::last_slot_hot(state, state.live_k, cfg.climb_slot_min);
        const float t_live = detail::lookup_t(cfg.round_time, state.live_k);
        const float t_hi   = detail::lookup_t(cfg.round_time, k_hi);
        const bool t_sane  = t_hi > 0.0f && t_live > 0.0f && t_hi <= t_live * 1.20f;
        // Last-hot waives δ_add for 4→5 (Python / DFlash) and 3→4. r8 required
        // hi_score > live_score too; that reproduced r3 Python (266 vs 286, hist 4:439).
        const bool last_hot_probe = extra_hot && t_sane;
        const float drop_max =
            (state.live_k == 4 && k_lo == 3) ? cfg.drop_to_3_max : cfg.drop_slot_max;
        // 5→4: CUDA k=5 last slot is dead (s[4]=0.28) but score(4) loses to score(5)
        // because T[5]/T[4] is only 1.03. Waive only when hop 4 is still last-hot
        // (CUDA s[3]=0.38). AIME_15 s[3]=0.24 is cold — no waiver, stay at 5.
        // r9 fires that waiver at DFlash first-remove (8). Score 5→4 stays at 32.
        // r10 required 8 consecutive extra-dead rounds: Python 266 but CUDA 239
        // and AIME_15 186. Reverted.
        const bool k5_to_4 = state.live_k == 5 && k_lo == 4;
        const bool k5_score_ready =
            !k5_to_4 || state.observed >= kAdaptiveFirstRemoveWarmup;
        const bool dead_k5_to_4 =
            k5_to_4 && state.observed >= kAdaptiveK5WaiveWarmup &&
            detail::extra_slot_dead(state, state.live_k, drop_max) &&
            detail::last_slot_hot(state, 4U, cfg.climb_slot_min);
        if (can_add && k_hi != state.live_k && k_hi <= budget_extent &&
            (hi_score > live_score * (1.0f + cfg.delta_add) || last_hot_probe)) {
            next = k_hi;
        } else if (can_remove && k_lo != state.live_k &&
                   detail::extra_slot_dead(state, state.live_k, drop_max) &&
                   (dead_k5_to_4 ||
                    (k5_score_ready && detail::score_k(cfg, state, k_lo) >
                                           live_score * (1.0f + cfg.delta_remove)))) {
            next = k_lo;
        }
    }

    next = std::min({next, budget_extent, cap_max});
    const std::uint32_t floor_k =
        state.observed >= cfg.warmup_rounds
            ? detail::p0_floor_k(state, cfg.captured_ks, cfg.high_p0_min_k)
            : (cfg.captured_ks.empty() ? 0 : cfg.captured_ks.front());
    if (next < floor_k && floor_k <= budget_extent) { next = floor_k; }
    if (next < cfg.captured_ks.front()) { next = cfg.captured_ks.front(); }
    if (!detail::captured_contains(cfg.captured_ks, next)) {
        next = adaptive_snap_captured_k(cfg.captured_ks, next);
        if (next > budget_extent) {
            std::uint32_t down = cfg.captured_ks.front();
            for (std::uint32_t c : cfg.captured_ks) {
                if (c <= budget_extent) { down = c; }
            }
            next = down;
        }
    }
    if (next != state.live_k) {
        state.live_k      = next;
        state.rounds_at_k = 0;
    }
    return state.live_k;
}

} // namespace ninfer::targets::qwen3_6
