#include "targets/qwen3_6/impl/runtime/adaptive_draft.h"

#include "ninfer/types.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <initializer_list>
#include <utility>
#include <iostream>
#include <span>
#include <string_view>
#include <vector>

namespace {

namespace q36 = ninfer::targets::qwen3_6;

int failures = 0;

void expect(bool condition, std::string_view message) {
    if (condition) { return; }
    ++failures;
    std::cerr << "FAIL: " << message << '\n';
}

void expect_near(float got, float want, float tol, std::string_view message) {
    const float err = got > want ? got - want : want - got;
    expect(err <= tol, message);
}

void plant_r(q36::AdaptiveDraftState& state, std::uint32_t live_k,
             std::initializer_list<float> rs, std::uint32_t n = 256) {
    q36::seed_adaptive_draft_state(state, live_k, q36::AdaptiveDraftLaw::Greedy);
    state.observed    = n;
    state.rounds_at_k = 32;
    std::uint32_t i   = 0;
    for (float r : rs) {
        if (i >= q36::kAdaptiveMaxHops) { break; }
        const float nn   = static_cast<float>(n);
        state.alpha[i]   = r * nn + 1.0f;
        state.beta[i]    = (1.0f - r) * nn + 1.0f;
        state.r_seen |= static_cast<std::uint16_t>(1U << i);
        ++i;
    }
}

void plant_t(q36::AdaptiveRoundTimeState& st, std::uint32_t k, float seconds,
             std::uint32_t L = 512) {
    q36::adaptive_observe_round_time(st, k, seconds, L);
}

q36::AdaptiveDraftConfig cfg_of(std::span<const std::uint32_t> ks,
                                const q36::AdaptiveRoundTimeState& t, std::uint32_t L = 512,
                                float sw = 0.0f) {
    q36::AdaptiveDraftConfig cfg;
    cfg.captured_ks    = ks;
    cfg.round_time     = &t;
    cfg.length_tokens  = L;
    cfg.switch_seconds = sw;
    return cfg;
}

std::uint32_t pick(const q36::AdaptiveDraftConfig& cfg, const q36::AdaptiveDraftState& state,
                   std::uint32_t cap = 5, std::uint32_t live = 0) {
    const q36::AdaptiveDraftState* ptr = &state;
    const std::uint32_t row[]          = {cap};
    return q36::adaptive_select_k(cfg, std::span<const q36::AdaptiveDraftState* const>(&ptr, 1),
                                  std::span<const std::uint32_t>(row, 1), cap,
                                  live != 0 ? live : state.live_k);
}

void test_capture_set() {
    using ninfer::SpeculativeBackend;
    const auto eq = [](std::vector<std::uint32_t> got, std::vector<std::uint32_t> want,
                       std::string_view msg) { expect(got == want, msg); };
    eq(q36::adaptive_draft_ks(SpeculativeBackend::Mtp, 5, false, 7), {5}, "frozen MTP {N}");
    eq(q36::adaptive_draft_ks(SpeculativeBackend::Mtp, 5, true, 7), {3, 4, 5}, "MTP adaptive {3,4,5}");
    eq(q36::adaptive_draft_ks(SpeculativeBackend::DFlash, 5, true, 7), {3, 4, 5},
       "DFlash adaptive {3,4,5}");
    eq(q36::adaptive_draft_ks(SpeculativeBackend::DFlash, 7, true, 7), {3, 4, 5, 6, 7},
       "DFlash adaptive {3..7}");
    eq(q36::adaptive_draft_ks(SpeculativeBackend::DFlash, 15, true, 5), {3, 4, 5},
       "DFlash adaptive bounded by the target maximum");
    eq(q36::adaptive_draft_ks(SpeculativeBackend::DFlash, 4, true, 7), {4}, "DFlash N=4 frozen {4}");
    eq(q36::adaptive_draft_ks(SpeculativeBackend::DFlash, 7, false, 7), {7}, "frozen DFlash {N}");
}

void test_seed_is_captured_min() {
    using ninfer::SpeculativeBackend;
    const auto df = q36::adaptive_draft_ks(SpeculativeBackend::DFlash, 5, true, 7);
    const auto mt = q36::adaptive_draft_ks(SpeculativeBackend::Mtp, 5, true, 7);
    expect(q36::adaptive_seed_k(df, SpeculativeBackend::DFlash) == 3,
           "seed fallback is captured.front()");
    expect(q36::adaptive_seed_k(mt, SpeculativeBackend::Mtp) == 3, "MTP seed is also the smallest k");
    const std::uint32_t frozen[] = {7};
    expect(q36::adaptive_seed_k(frozen, SpeculativeBackend::DFlash) == 7, "frozen |K|=1 seeds N");
}

void test_topology_class() {
    const std::uint32_t C = 4;
    expect(q36::adaptive_k_stride(C, 0) == C, "27B k_stride = C");
    expect(q36::adaptive_topology_class(0, C, 0, C, 1) == 0, "frozen topology_class B=1");
    expect(q36::adaptive_topology_class(2, C, 0, C, 1) == 2U * C, "k=third class at B=1");
}

void test_y_is_one_plus_product_of_r() {
    q36::AdaptiveDraftState state;
    plant_r(state, 4, {0.80f, 0.625f, 0.60f});
    const float q0 = 0.80f;
    const float q1 = 0.80f * 0.625f;
    const float q2 = q1 * 0.60f;
    expect_near(q36::detail::expected_tokens(state, 3), 1.0f + q0 + q1 + q2, 0.02f,
                "Y(3) = 1 + r0 + r0 r1 + r0 r1 r2");
}

void test_r_updates_only_when_prefix_reached() {
    q36::AdaptiveDraftState state;
    q36::seed_adaptive_draft_state(state, 0, q36::AdaptiveDraftLaw::Greedy);
    q36::adaptive_observe_hops(state, 2, 5);
    expect((state.r_seen & 0x07U) == 0x07U, "accepted=2 drafted=5 observes r0,r1,r2");
    expect((state.r_seen & 0x18U) == 0, "r3 and r4 are not updated when the prefix died at 2");
    const float d = q36::kAdaptiveDiscount;
    const float p = q36::kAdaptiveBetaPrior;
    expect_near(q36::detail::r_mean(state, 0), (d * p + 1.0f) / (2.0f * d * p + 1.0f), 1e-5f,
                "accepted>0 is a success for r0");
    expect_near(q36::detail::r_mean(state, 1), (d * p + 1.0f) / (2.0f * d * p + 1.0f), 1e-5f,
                "accepted>1 is a success for r1");
    expect_near(q36::detail::r_mean(state, 2), (d * p) / (2.0f * d * p + 1.0f), 1e-5f,
                "accepted=2 is a failure for r2, not for later r_i");

    q36::AdaptiveDraftState deep;
    q36::seed_adaptive_draft_state(deep, 0, q36::AdaptiveDraftLaw::Greedy);
    q36::adaptive_observe_hops(deep, 7, 7);
    expect(deep.r_seen == 0x7FU, "a fully accepted k=7 round observes r0..r6");
    expect(q36::detail::r_mean(deep, 6) > 0.5f, "accepted=7 is a success for r6");
}

void test_pcur_zero_skips_update() {
    q36::AdaptiveDraftState state;
    q36::seed_adaptive_draft_state(state, 4, q36::AdaptiveDraftLaw::Greedy);
    const auto before        = state;
    const std::uint32_t ks[] = {3, 4, 5};
    q36::AdaptiveRoundTimeState t;
    auto cfg = cfg_of(ks, t);
    expect(q36::adaptive_draft_next(cfg, state, 0, 0, 5, 4, false) == 4, "pcur==0 keeps live_k");
    expect(state.observed == before.observed && state.r_seen == 0, "pcur==0 skips hop update");
}

void test_t_ols_shared_slope() {
    q36::AdaptiveRoundTimeState st;
    plant_t(st, 4, 0.020f, 0);
    plant_t(st, 4, 0.030f, 1000);
    expect_near(q36::adaptive_t_hat(st, 4, 500), 0.025f, 1e-5f,
                "T(4,L) interpolates two observations");
    plant_t(st, 5, 0.040f, 0);
    const float t5 = q36::adaptive_t_hat(st, 5, 1000);
    expect(t5 > 0.040f, "unobserved L for k=5 uses the shared slope from k=4");
    expect_near(t5 - 0.040f, 0.010f, 2e-3f, "shared c ≈ 0.010 / 1000 tokens");
}

void test_cold_start_runs_smallest_k() {
    q36::AdaptiveDraftState state;
    q36::seed_adaptive_draft_state(state, 0, q36::AdaptiveDraftLaw::Greedy);
    q36::AdaptiveRoundTimeState t;
    const std::uint32_t ks[] = {3, 4, 5};
    auto cfg                 = cfg_of(ks, t);
    expect(pick(cfg, state) == 3, "no T and no hops → captured.front()");
}

void test_measured_expensive_k5_is_dominated() {
    q36::AdaptiveDraftState state;
    plant_r(state, 4, {0.778f, 0.717f, 0.695f, 0.727f});
    q36::AdaptiveRoundTimeState t;
    plant_t(t, 3, 0.05586f);
    plant_t(t, 4, 0.04931f);
    plant_t(t, 5, 0.07270f);
    const std::uint32_t ks[] = {3, 4, 5};
    auto cfg                 = cfg_of(ks, t);
    expect(pick(cfg, state, 5, 4) == 4, "AIME C=4 T(5) cannot be paid by q4 <= q3");
}

void test_unmeasured_k5_is_probed_at_most_once_then_dropped() {
    q36::AdaptiveDraftState state;
    plant_r(state, 4, {0.778f, 0.717f, 0.695f, 0.727f});
    q36::AdaptiveRoundTimeState t;
    plant_t(t, 3, 0.05586f);
    plant_t(t, 4, 0.04931f);
    const std::uint32_t ks[] = {3, 4, 5};
    auto cfg                 = cfg_of(ks, t);
    expect(pick(cfg, state, 5, 4) == 5, "optimistic equal-T extra hop allows one k=5 probe");
    plant_t(t, 5, 0.07270f);
    expect(pick(cfg, state, 5, 5) == 4, "after T(5) is measured the arm is dominated");
}

void test_unmeasured_arm_is_probed_despite_slower_short_arms() {
    // Round time is not monotone in k (A16-era C=6: k=1/2 rounds cost more than k=3..5; A8 C=4:
    // k=4 is cheapest). Measured arms must not hide unmeasured, faster ones.
    q36::AdaptiveDraftState state;
    plant_r(state, 2, {0.61f, 0.40f});
    q36::AdaptiveRoundTimeState t;
    plant_t(t, 1, 0.036f);
    plant_t(t, 2, 0.041f);
    const std::uint32_t ks[] = {1, 2, 3, 4, 5};
    auto cfg                 = cfg_of(ks, t);
    expect(pick(cfg, state, 5, 1) == 3, "first unmeasured arm is probed");
    plant_t(t, 3, 0.022f);
    expect(pick(cfg, state, 5, 3) == 4, "each unmeasured arm is measured once");
    plant_t(t, 4, 0.024f);
    plant_t(t, 5, 0.027f);
    plant_r(state, 5, {0.61f, 0.40f, 0.40f, 0.40f, 0.40f});
    expect(pick(cfg, state, 5, 5) == 3, "then argmax E[Y]/T over measured arms");
}

void test_dominated_arm_is_measured_once_then_dropped() {
    q36::AdaptiveDraftState state;
    plant_r(state, 4, {0.80f, 0.70f, 0.60f, 0.0f});
    q36::AdaptiveRoundTimeState t;
    plant_t(t, 3, 0.020f);
    plant_t(t, 4, 0.024f);
    const std::uint32_t ks[] = {3, 4, 5};
    auto cfg                 = cfg_of(ks, t);
    expect(pick(cfg, state, 5, 4) == 5, "an unmeasured arm has no T bound: probe k=5 once");
    plant_t(t, 5, 0.028f);
    expect(pick(cfg, state, 5, 5) == 3, "dead last hop and rising T: lock 3 after measuring");
}

void test_unmeasured_k4_probed_at_most_once() {
    q36::AdaptiveDraftState state;
    plant_r(state, 3, {0.90f, 0.90f, 0.90f});
    q36::AdaptiveRoundTimeState t;
    plant_t(t, 3, 0.020f);
    const std::uint32_t ks[] = {3, 4, 5};
    auto cfg                 = cfg_of(ks, t);
    expect(pick(cfg, state, 5, 3) == 4, "T(3) known, equal-T extra hop probes k=4 once");
    plant_t(t, 4, 0.040f);
    expect(pick(cfg, state, 5, 4) == 5, "k=5 is still measured once");
    plant_t(t, 5, 0.060f);
    expect(pick(cfg, state, 5, 5) == 3, "after expensive T(4)/T(5) both arms are dominated");
}

// MTP keeps the truncated sum: unseen hops add nothing, so a request without hops takes the
// cheapest measured round.
void test_mtp_unseen_hop_stops_expected_tokens() {
    q36::AdaptiveDraftState planted;
    plant_r(planted, 3, {0.80f, 0.625f, 0.60f});
    expect_near(q36::detail::expected_tokens(planted, 5),
                q36::detail::expected_tokens(planted, 3), 1e-5f,
                "MTP: unseen r3,r4 do not add tokens");
    q36::AdaptiveDraftState state;
    q36::seed_adaptive_draft_state(state, 0, q36::AdaptiveDraftLaw::Greedy);
    q36::AdaptiveRoundTimeState t;
    plant_t(t, 3, 0.056f);
    plant_t(t, 4, 0.049f);
    plant_t(t, 5, 0.055f);
    const std::uint32_t ks[] = {3, 4, 5};
    auto cfg                 = cfg_of(ks, t);
    expect(pick(cfg, state) == 4, "MTP: no hop data takes the smallest measured T");
}

void test_ties_keep_smaller_k() {
    q36::AdaptiveDraftState state;
    plant_r(state, 5, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f});
    q36::AdaptiveRoundTimeState t;
    plant_t(t, 3, 0.020f);
    plant_t(t, 4, 0.020f);
    plant_t(t, 5, 0.020f);
    const std::uint32_t ks[] = {3, 4, 5};
    auto cfg                 = cfg_of(ks, t);
    expect(pick(cfg, state, 5, 3) == 3, "equal scores keep the smaller k");
}

void test_switch_cost_holds_live_k() {
    q36::AdaptiveDraftState state;
    plant_r(state, 4, {0.50f, 0.50f, 0.50f, 0.50f});
    q36::AdaptiveRoundTimeState t;
    plant_t(t, 3, 0.0190f);
    plant_t(t, 4, 0.0197f);
    plant_t(t, 5, 0.0400f);
    const std::uint32_t ks[] = {3, 4, 5};
    auto cfg                 = cfg_of(ks, t, 512, 0.0f);
    expect(pick(cfg, state, 5, 4) == 3, "without switch cost k=3 wins a ~0.7 ms T gap");
    cfg.switch_seconds = 0.001f;
    expect(pick(cfg, state, 5, 4) == 4, "1 ms switch cost holds k=4 across that gap");
}

void test_stationary_late_hop_does_not_force_k5() {
    q36::AdaptiveDraftState state;
    q36::seed_adaptive_draft_state(state, 4, q36::AdaptiveDraftLaw::Greedy);
    q36::AdaptiveRoundTimeState t;
    plant_t(t, 3, 0.022f);
    plant_t(t, 4, 0.020f);
    plant_t(t, 5, 0.040f);
    const std::uint32_t ks[] = {3, 4, 5};
    auto cfg                 = cfg_of(ks, t);
    cfg.switch_seconds       = 0.0f;
    for (int i = 0; i < 80; ++i) {
        (void)q36::adaptive_draft_next(cfg, state, 3, 4, 5, 4, false);
    }
    expect(state.live_k == 4, "stationary hop-3 failures do not inject k=5");
    expect((state.r_seen & 0x10U) == 0, "k=4 rounds never observe r4");
}

void test_batch_sum_e_over_t() {
    const std::uint32_t captured[] = {3, 4, 5};
    q36::AdaptiveRoundTimeState t;
    plant_t(t, 3, 0.83f);
    plant_t(t, 4, 0.93f);
    plant_t(t, 5, 1.20f);
    q36::AdaptiveDraftState hot;
    q36::AdaptiveDraftState cold;
    plant_r(hot, 4, {0.90f, 0.94f, 0.88f, 0.88f});
    plant_r(cold, 3, {0.40f, 0.35f, 0.36f});
    const q36::AdaptiveDraftState* mid[] = {&hot, &cold};
    const std::uint32_t rows[]           = {5, 5};
    const std::uint32_t picked =
        q36::adaptive_select_batch_k(mid, rows, captured, &t, 512, 0, nullptr);
    float best_s               = -1.0f;
    std::uint32_t want         = 3;
    for (std::uint32_t k : {3U, 4U, 5U}) {
        const float e  = q36::detail::expected_tokens(hot, k) + q36::detail::expected_tokens(cold, k);
        const float tk = q36::adaptive_t_hat(t, k, 512);
        const float sc = e / tk;
        if (sc > best_s) {
            best_s = sc;
            want   = k;
        }
    }
    expect(picked == want, "C=2 picks argmax Σ_r E[Y_r(k)] / T(k,L)");
}

void test_batch_row_budget_clips_expected_tokens() {
    const std::uint32_t captured[] = {3, 4, 5};
    q36::AdaptiveRoundTimeState t;
    plant_t(t, 3, 0.95f);
    plant_t(t, 4, 0.93f);
    plant_t(t, 5, 1.00f);
    q36::AdaptiveDraftState hot;
    plant_r(hot, 5, {0.90f, 0.94f, 0.88f, 0.88f, 0.76f});
    const q36::AdaptiveDraftState* mid[] = {&hot, &hot};
    const std::uint32_t rows[]           = {5, 3};
    const std::uint32_t picked =
        q36::adaptive_select_batch_k(mid, rows, captured, &t, 512, 0, nullptr);
    float best_s               = -1.0f;
    std::uint32_t want         = 3;
    for (std::uint32_t k : {3U, 4U, 5U}) {
        const float e = q36::detail::expected_tokens(hot, std::min(k, 5U)) +
                        q36::detail::expected_tokens(hot, std::min(k, 3U));
        const float sc = e / q36::adaptive_t_hat(t, k, 512);
        if (sc > best_s) {
            best_s = sc;
            want   = k;
        }
    }
    expect(picked == want, "C=2 scores E[Y_r | min(k, budget_r)] / T");
}

void test_batch_next_writes_executed_k() {
    const std::uint32_t captured[] = {3, 4, 5};
    q36::AdaptiveRoundTimeState t;
    plant_t(t, 3, 0.056f);
    plant_t(t, 4, 0.049f);
    plant_t(t, 5, 0.073f);
    q36::AdaptiveDraftState a;
    q36::AdaptiveDraftState b;
    plant_r(a, 3, {0.778f, 0.717f, 0.695f, 0.727f});
    plant_r(b, 5, {0.778f, 0.717f, 0.695f, 0.727f});
    const q36::AdaptiveDraftState* mid[] = {&a, &b};
    const std::uint32_t rows[]           = {5, 5};
    q36::AdaptiveBatchKState batch;
    const std::uint32_t k =
        q36::adaptive_batch_next(batch, mid, rows, captured, &t, 512, nullptr);
    expect(k == 4 && batch.live_k == 4, "batch live_k is the executed argmax, not a per-row pick");
    q36::AdaptiveDraftState* mut[] = {&a, &b};
    q36::adaptive_assign_live_k(mut, k);
    expect(a.live_k == 4 && b.live_k == 4, "C>=2 writes the same executed k onto every row");
}

void test_budget_clamp() {
    q36::AdaptiveDraftState state;
    plant_r(state, 5, {0.9f, 0.8f, 0.7f, 0.6f, 0.5f});
    q36::AdaptiveRoundTimeState t;
    plant_t(t, 3, 0.02f);
    plant_t(t, 4, 0.02f);
    plant_t(t, 5, 0.02f);
    const std::uint32_t ks[] = {3, 4, 5};
    auto cfg                 = cfg_of(ks, t);
    expect(q36::adaptive_draft_next(cfg, state, 5, 5, 3, 5, false) == 3,
           "budget_extent=3 clamps to captured k<=3");
}

void test_t_survives_request_seed() {
    q36::AdaptiveRoundTimeState st;
    plant_t(st, 4, 0.015f);
    q36::AdaptiveDraftState state;
    q36::seed_adaptive_draft_state(state, 0, q36::AdaptiveDraftLaw::Greedy);
    expect(q36::adaptive_t_measured(st, 4) && state.r_seen == 0,
           "T is server-global; hop posterior is per-request");
}


// DFlash hop model. Exploration rows are recorded directly so the arithmetic stays explicit.
void explore_rows(q36::AdaptiveHopRates& rates, q36::AdaptiveDraftLaw law, std::uint32_t k,
                  std::uint32_t accepted, int rows) {
    q36::AdaptiveDraftState scratch;
    q36::seed_adaptive_draft_state(scratch, k, law);
    for (int r = 0; r < rows; ++r) { q36::adaptive_record_round(scratch, accepted, k, k, &rates, true); }
}

float sum_of_products(std::initializer_list<float> rates) {
    float e = 1.0f, run = 1.0f;
    for (float r : rates) {
        run *= r;
        e += run;
    }
    return e;
}

void test_hop_cold_table_is_k_independent() {
    q36::AdaptiveHopRates rates;
    q36::AdaptiveDraftState state;
    q36::seed_adaptive_draft_state(state, 0, q36::AdaptiveDraftLaw::PLess);
    for (std::uint32_t k = 3; k <= 7; ++k) {
        expect_near(q36::adaptive_hop_hazard(rates, q36::AdaptiveDraftLaw::PLess, k, 0), 0.35f,
                    1e-6f, "a cold table starts every block length at the pooled prior");
    }
    expect_near(q36::detail::hop_expected_tokens(state, rates, 3, 3),
                sum_of_products({0.65f, 0.65f, 0.65f}), 1e-5f,
                "a cold request scores the prior hazard with no unseen optimism");
}

void test_hop_exploration_rows_alone_update_the_table() {
    q36::AdaptiveHopRates rates;
    q36::AdaptiveDraftState state;
    q36::seed_adaptive_draft_state(state, 5, q36::AdaptiveDraftLaw::Greedy);
    q36::adaptive_record_round(state, 2, 5, 5, &rates, false);
    expect(rates.trials[0][5][0] == 0.0f, "an ordinary round leaves the global table");
    q36::adaptive_record_round(state, 2, 3, 5, &rates, true);
    expect(rates.trials[0][5][0] == 0.0f, "a budget-clipped exploration row does not count");
    q36::adaptive_record_round(state, 2, 5, 5, &rates, true);
    expect(rates.trials[0][5][0] == 1.0f && rates.trials[0][5][2] == 1.0f &&
               rates.trials[0][5][3] == 0.0f && rates.failures[0][5][2] == 1.0f &&
               rates.failures[0][5][1] == 0.0f,
           "a full exploration row counts trials up to the failing hop");
    expect(state.rounds_at_k == 1, "exploration rounds do not count toward the incumbent k");
    // Exact hierarchy after that single row.
    const float p0 = (0.0f + 4.0f * 0.35f) / (1.0f + 4.0f);
    const float p2 = (1.0f + 4.0f * 0.35f) / (1.0f + 4.0f);
    const float pr[] = {p0, p0, p2, 0.35f, 0.35f};
    const float expected = pr[0] + pr[1] + pr[2];
    const float a        = q36::kAdaptiveBlockRatioPrior;
    const float b        = q36::kAdaptiveCellPriorTrials;
    const float rho      = (1.0f + a) / (expected + a);
    expect_near(q36::adaptive_block_ratio(rates, q36::AdaptiveDraftLaw::Greedy, 5), rho, 1e-6f,
                "rho_k = (failures + a) / (sum trials * P_i + a)");
    expect_near(q36::adaptive_hop_hazard(rates, q36::AdaptiveDraftLaw::Greedy, 5, 2),
                (1.0f + b * rho * p2) / (1.0f + b), 1e-6f,
                "H = (failures + b rho P) / (trials + b)");
}

void test_hop_block_ratio_shares_strength_across_hops() {
    // k=5 rows fail at hop 0 twice as often as k=4 rows: the ratio separates the two block
    // lengths at every hop, including hops neither has much data on.
    q36::AdaptiveHopRates rates;
    for (int r = 0; r < 1000; ++r) {
        explore_rows(rates, q36::AdaptiveDraftLaw::Greedy, 4, r % 5 == 0 ? 0U : 4U, 1);
        explore_rows(rates, q36::AdaptiveDraftLaw::Greedy, 5, r % 5 < 2 ? 0U : 5U, 1);
    }
    const float r4 = q36::adaptive_block_ratio(rates, q36::AdaptiveDraftLaw::Greedy, 4);
    const float r5 = q36::adaptive_block_ratio(rates, q36::AdaptiveDraftLaw::Greedy, 5);
    expect(r5 > 1.2f * r4, "the worse block length gets the larger hazard ratio");
    expect(q36::adaptive_hop_hazard(rates, q36::AdaptiveDraftLaw::Greedy, 5, 3) >
               q36::adaptive_hop_hazard(rates, q36::AdaptiveDraftLaw::Greedy, 4, 3),
           "the ratio carries to hops with little direct data");
    expect_near(q36::adaptive_block_ratio(rates, q36::AdaptiveDraftLaw::Greedy, 7), 1.0f, 1e-6f,
                "an untried block length keeps ratio 1");
    expect_near(q36::adaptive_block_ratio(rates, q36::AdaptiveDraftLaw::PLess, 5), 1.0f, 1e-6f,
                "draft laws are learned separately");
    expect_near(q36::adaptive_block_ratio(rates, q36::AdaptiveDraftLaw::Sampled, 5), 1.0f, 1e-6f,
                "temperature-sampled drafts do not share the greedy table");
}

void test_hop_picker_prefers_better_shorter_block() {
    const std::uint32_t captured[] = {4, 5};
    q36::AdaptiveRoundTimeState t;
    plant_t(t, 4, 0.010f);
    plant_t(t, 5, 0.010f);
    q36::AdaptiveHopRates rates;
    q36::AdaptiveDraftState state;
    q36::seed_adaptive_draft_state(state, 4, q36::AdaptiveDraftLaw::Greedy);
    auto cfg      = cfg_of(captured, t);
    cfg.hop_rates = &rates;
    expect(pick(cfg, state, 5, 4) == 5, "k-independent hazards at equal T prefer the longer block");
    for (int r = 0; r < 400; ++r) {
        explore_rows(rates, q36::AdaptiveDraftLaw::Greedy, 4, r % 4 == 0 ? 1U : 4U, 1);
        explore_rows(rates, q36::AdaptiveDraftLaw::Greedy, 5, r % 4 < 3 ? 0U : 5U, 1);
    }
    expect(pick(cfg, state, 5, 4) == 4, "learned block hazards keep k=4 when k=5 accepts less");
}

void test_hop_content_factor_normalizes_by_block_hazard() {
    q36::AdaptiveHopRates rates;
    q36::AdaptiveDraftState state;
    q36::seed_adaptive_draft_state(state, 7, q36::AdaptiveDraftLaw::Greedy);
    q36::adaptive_record_round(state, 7, 7, 7, &rates, false);
    const float c0 = 1.0f / 1.35f;
    const float r0 = 1.0f - c0 * 0.35f;
    const float p1 = 1.0f + 0.5f * (c0 - 1.0f);
    const float c1 = p1 / 1.35f;
    const float r1 = 1.0f - c1 * 0.35f;
    expect_near(q36::detail::hop_expected_tokens(state, rates, 5, 2), 1.0f + r0 + r0 * r1, 1e-5f,
                "accepted trials lower the content factor on the hazard");
}

void test_hop_batch_row_uses_block_length() {
    q36::AdaptiveHopRates rates;
    for (int r = 0; r < 200; ++r) { explore_rows(rates, q36::AdaptiveDraftLaw::Greedy, 5, 0, 1); }
    q36::AdaptiveDraftState state;
    q36::seed_adaptive_draft_state(state, 5, q36::AdaptiveDraftLaw::Greedy);
    const q36::AdaptiveDraftState* rows[] = {&state, &state};
    const std::uint32_t caps[]            = {5, 3};
    const float e = q36::detail::row_sum_e(rows, caps, 5, &rates);
    expect_near(e, q36::detail::hop_expected_tokens(state, rates, 5, 5) +
                       q36::detail::hop_expected_tokens(state, rates, 5, 3), 1e-5f,
                "a capped row in a k=5 batch uses the k=5 block hazards over its extent");
    expect(q36::detail::hop_expected_tokens(state, rates, 5, 3) <
               q36::detail::hop_expected_tokens(state, rates, 3, 3),
           "block length, not extent, selects the hazards");
}

void test_hop_decay_relaxes_unreached_hops() {
    q36::AdaptiveHopRates rates;
    q36::AdaptiveDraftState state;
    q36::seed_adaptive_draft_state(state, 7, q36::AdaptiveDraftLaw::Greedy);
    for (int round = 0; round < 8; ++round) { q36::adaptive_record_round(state, 5, 7, 7, &rates, false); }
    expect(state.hazard_failures[5] > 4.0f, "hop 5 failed every k=7 round");
    for (int round = 0; round < 200; ++round) { q36::adaptive_record_round(state, 0, 3, 3, &rates, false); }
    expect(state.hazard_failures[5] < 0.01f && state.hazard_expected[5] < 0.01f,
           "a hop the picker stops reaching decays back to its prior");
}

void test_hop_easy_content_does_not_saturate_unseen_hops() {
    q36::AdaptiveHopRates rates;
    q36::AdaptiveDraftState state;
    q36::seed_adaptive_draft_state(state, 5, q36::AdaptiveDraftLaw::Greedy);
    for (int round = 0; round < 64; ++round) { q36::adaptive_record_round(state, 5, 5, 5, &rates, false); }
    const float e3 = q36::detail::hop_expected_tokens(state, rates, 7, 3);
    const float e4 = q36::detail::hop_expected_tokens(state, rates, 7, 4);
    const float e5 = q36::detail::hop_expected_tokens(state, rates, 7, 5);
    const float e6 = q36::detail::hop_expected_tokens(state, rates, 7, 6);
    const float r4 = (e5 - e4) / (e4 - e3);
    const float r5 = (e6 - e5) / (e5 - e4);
    expect(r4 > 0.9f && r4 <= q36::kAdaptiveMaxHopRate + 1e-6f, "a seen easy hop approaches the cap");
    expect(r5 < r4 && r5 > 0.6f, "the unseen hop inherits half of the easiness, not the cap");
}

void test_hop_model_warm_server_new_request_is_not_locked_short() {
    q36::AdaptiveRoundTimeState t;
    plant_t(t, 3, 0.0150f);
    plant_t(t, 4, 0.0155f);
    plant_t(t, 5, 0.0160f);
    const std::uint32_t ks[] = {3, 4, 5};
    q36::AdaptiveHopRates rates;
    auto cfg      = cfg_of(ks, t, 512, 0.0f);
    cfg.hop_rates = &rates;
    q36::AdaptiveDraftState state;
    q36::seed_adaptive_draft_state(state, 0, q36::AdaptiveDraftLaw::Greedy);
    state.live_k = pick(cfg, state);
    expect(state.live_k == 5, "fresh request on measured near-flat T starts at the longest k");
    for (int round = 0; round < 64; ++round) {
        (void)q36::adaptive_draft_next(cfg, state, 2, state.live_k, 5, state.live_k, false);
    }
    expect(state.live_k == 3, "a prefix that dies at hop 2 settles on k=3");
    for (int round = 0; round < 64; ++round) {
        (void)q36::adaptive_draft_next(cfg, state, state.live_k, state.live_k, 5, state.live_k,
                                       false);
    }
    expect(state.live_k == 5, "once every drafted hop accepts, the deeper hops pull k back up");
}

void test_exploration_trigger_is_uniform_and_capped() {
    q36::AdaptiveRoundTimeState t;
    for (std::uint32_t k = 3; k <= 7; ++k) { plant_t(t, k, 0.015f); }
    const std::uint32_t ks[] = {3, 4, 5, 6, 7};
    q36::AdaptiveHopRates rates;
    std::array<std::uint32_t, 8> hits{};
    std::uint32_t total = 0;
    for (int i = 0; i < 64000; ++i) {
        const std::uint32_t k = q36::adaptive_explore_k(rates, ks, t, 5);
        if (k != 0) {
            expect(k >= 3 && k <= 5, "exploration stays within the cap");
            ++hits[k];
            ++total;
        }
    }
    expect(total > 1500 && total < 2500, "one round in 32 explores");
    expect(hits[3] > 500 && hits[4] > 500 && hits[5] > 500, "explored k is uniform over the cap");
    q36::AdaptiveHopRates replay;
    q36::AdaptiveHopRates again;
    bool same = true;
    for (int i = 0; i < 1000; ++i) {
        same = same && q36::adaptive_explore_k(replay, ks, t, 7) == q36::adaptive_explore_k(again, ks, t, 7);
    }
    expect(same, "the trigger is a deterministic function of the decision counter");
    q36::AdaptiveRoundTimeState partial;
    plant_t(partial, 3, 0.015f);
    q36::AdaptiveHopRates probe;
    std::uint32_t during_probe = 0;
    for (int i = 0; i < 4000; ++i) { during_probe += q36::adaptive_explore_k(probe, ks, partial, 7); }
    expect(during_probe == 0, "no exploration while a k within the cap is unmeasured");
}

void test_content_sums_decay_before_adding() {
    q36::AdaptiveHopRates rates;
    q36::AdaptiveDraftState state;
    q36::seed_adaptive_draft_state(state, 3, q36::AdaptiveDraftLaw::Greedy);
    q36::adaptive_record_round(state, 0, 3, 3, &rates, false);
    q36::adaptive_record_round(state, 0, 3, 3, &rates, false);
    expect_near(state.hazard_failures[0], 1.0f + q36::kAdaptiveDiscount, 1e-6f,
                "two failures: d * 1 + 1 (decay, then add)");
    expect_near(state.hazard_expected[0], 0.35f * (1.0f + q36::kAdaptiveDiscount), 1e-6f,
                "expected failures accumulate the hazard in force before each round");
}

} // namespace

int main() {
    test_capture_set();
    test_seed_is_captured_min();
    test_topology_class();
    test_y_is_one_plus_product_of_r();
    test_r_updates_only_when_prefix_reached();
    test_pcur_zero_skips_update();
    test_t_ols_shared_slope();
    test_cold_start_runs_smallest_k();
    test_measured_expensive_k5_is_dominated();
    test_unmeasured_k5_is_probed_at_most_once_then_dropped();
    test_unmeasured_arm_is_probed_despite_slower_short_arms();
    test_dominated_arm_is_measured_once_then_dropped();
    test_unmeasured_k4_probed_at_most_once();
    test_mtp_unseen_hop_stops_expected_tokens();
    test_ties_keep_smaller_k();
    test_switch_cost_holds_live_k();
    test_stationary_late_hop_does_not_force_k5();
    test_batch_sum_e_over_t();
    test_batch_row_budget_clips_expected_tokens();
    test_batch_next_writes_executed_k();
    test_budget_clamp();
    test_t_survives_request_seed();
    test_hop_cold_table_is_k_independent();
    test_hop_exploration_rows_alone_update_the_table();
    test_hop_block_ratio_shares_strength_across_hops();
    test_hop_picker_prefers_better_shorter_block();
    test_hop_content_factor_normalizes_by_block_hazard();
    test_hop_batch_row_uses_block_length();
    test_hop_decay_relaxes_unreached_hops();
    test_hop_easy_content_does_not_saturate_unseen_hops();
    test_hop_model_warm_server_new_request_is_not_locked_short();
    test_exploration_trigger_is_uniform_and_capped();
    test_content_sums_decay_before_adding();
    if (failures != 0) {
        std::cerr << failures << " adaptive draft host checks failed\n";
        return 1;
    }
    std::cout << "adaptive draft host checks passed\n";
    return 0;
}
