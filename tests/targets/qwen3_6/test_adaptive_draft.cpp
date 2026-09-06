#include "targets/qwen3_6/impl/runtime/adaptive_draft.h"

#include "ninfer/types.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <initializer_list>
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
    q36::seed_adaptive_draft_state(state, live_k);
    state.observed    = n;
    state.rounds_at_k = 32;
    std::uint32_t i   = 0;
    for (float r : rs) {
        if (i >= 5) { break; }
        const float nn   = static_cast<float>(n);
        state.alpha[i]   = r * nn + 1.0f;
        state.beta[i]    = (1.0f - r) * nn + 1.0f;
        state.r_seen |= static_cast<std::uint8_t>(1U << i);
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
    eq(q36::adaptive_draft_ks(SpeculativeBackend::Mtp, 5, false), {5}, "frozen MTP {N}");
    eq(q36::adaptive_draft_ks(SpeculativeBackend::Mtp, 5, true), {3, 4, 5}, "MTP adaptive {3,4,5}");
    eq(q36::adaptive_draft_ks(SpeculativeBackend::DFlash, 7, true), {3, 4, 5},
       "DFlash N>=5 {3,4,5}");
    eq(q36::adaptive_draft_ks(SpeculativeBackend::DFlash, 4, true), {4}, "DFlash N=4 frozen {4}");
    eq(q36::adaptive_draft_ks(SpeculativeBackend::DFlash, 7, false), {7}, "frozen DFlash {N}");
}

void test_seed_is_captured_min() {
    using ninfer::SpeculativeBackend;
    const auto df = q36::adaptive_draft_ks(SpeculativeBackend::DFlash, 5, true);
    const auto mt = q36::adaptive_draft_ks(SpeculativeBackend::Mtp, 5, true);
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

void test_unseen_r_not_added_to_mean_y() {
    q36::AdaptiveDraftState state;
    plant_r(state, 3, {0.80f, 0.625f, 0.60f});
    const float y3 = q36::detail::expected_tokens(state, 3);
    expect_near(q36::detail::expected_tokens(state, 5), y3, 1e-5f,
                "unseen r3,r4 do not add tokens to E[Y]");
    const float q2 = 0.80f * 0.625f * 0.60f;
    expect_near(q36::detail::expected_tokens_optimistic(state, 5), y3 + 2.0f * q2, 0.02f,
                "optimistic bound fills unseen r_i with 1");
}

void test_r_updates_only_when_prefix_reached() {
    q36::AdaptiveDraftState state;
    q36::seed_adaptive_draft_state(state, 0);
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
}

void test_pcur_zero_skips_update() {
    q36::AdaptiveDraftState state;
    q36::seed_adaptive_draft_state(state, 4);
    const auto before        = state;
    const std::uint32_t ks[] = {3, 4, 5};
    q36::AdaptiveRoundTimeState t;
    auto cfg = cfg_of(ks, t);
    expect(q36::adaptive_draft_next(cfg, state, 0, 0, 5, 4) == 4, "pcur==0 keeps live_k");
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

void test_unmeasured_t5_extrapolates_and_is_at_least_t4() {
    q36::AdaptiveRoundTimeState st;
    plant_t(st, 3, 0.056f, 512);
    plant_t(st, 4, 0.049f, 512);
    expect(q36::adaptive_t_measured(st, 4), "T(4) is measured");
    expect(!q36::adaptive_t_measured(st, 5), "T(5) is not measured");
    const float t5 = q36::adaptive_t_hat(st, 5, 512);
    expect(t5 >= 0.049f - 1e-6f, "T(5) >= T(4) even when 2 T(4)-T(3) is smaller");
}

void test_cold_start_runs_smallest_k() {
    q36::AdaptiveDraftState state;
    q36::seed_adaptive_draft_state(state, 0);
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

void test_small_q_extrapolation_skips_k5_probe() {
    q36::AdaptiveDraftState state;
    plant_r(state, 4, {0.80f, 0.70f, 0.60f, 0.0f});
    q36::AdaptiveRoundTimeState t;
    plant_t(t, 3, 0.020f);
    plant_t(t, 4, 0.024f);
    const std::uint32_t ks[] = {3, 4, 5};
    auto cfg                 = cfg_of(ks, t);
    expect(pick(cfg, state, 5, 4) == 3,
           "dead last hop and rising T: lock 3; unmeasured k=5 stays dominated");
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
    expect(pick(cfg, state, 5, 4) == 3, "after expensive T(4) the arm is dominated");
}

void test_argmin_t_when_hops_unseen() {
    q36::AdaptiveDraftState state;
    q36::seed_adaptive_draft_state(state, 0);
    q36::AdaptiveRoundTimeState t;
    plant_t(t, 3, 0.056f);
    plant_t(t, 4, 0.049f);
    plant_t(t, 5, 0.073f);
    const std::uint32_t ks[] = {3, 4, 5};
    auto cfg                 = cfg_of(ks, t);
    expect(pick(cfg, state) == 4, "no hop data: E[Y]=1, pick the smallest measured T");
}

void test_ties_keep_smaller_k() {
    q36::AdaptiveDraftState state;
    plant_r(state, 3, {0.0f, 0.0f, 0.0f});
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
    plant_r(state, 4, {0.50f, 0.50f, 0.50f});
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
    q36::seed_adaptive_draft_state(state, 4);
    q36::AdaptiveRoundTimeState t;
    plant_t(t, 3, 0.022f);
    plant_t(t, 4, 0.020f);
    plant_t(t, 5, 0.040f);
    const std::uint32_t ks[] = {3, 4, 5};
    auto cfg                 = cfg_of(ks, t);
    cfg.switch_seconds       = 0.0f;
    for (int i = 0; i < 80; ++i) {
        (void)q36::adaptive_draft_next(cfg, state, 3, 4, 5, 4);
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
        q36::adaptive_select_batch_k(mid, rows, captured, &t, 512, 0);
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
        q36::adaptive_select_batch_k(mid, rows, captured, &t, 512, 0);
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
        q36::adaptive_batch_next(batch, mid, rows, captured, &t, 512);
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
    expect(q36::adaptive_draft_next(cfg, state, 5, 5, 3, 5) == 3,
           "budget_extent=3 clamps to captured k<=3");
}

void test_t_survives_request_seed() {
    q36::AdaptiveRoundTimeState st;
    plant_t(st, 4, 0.015f);
    q36::AdaptiveDraftState state;
    q36::seed_adaptive_draft_state(state, 0);
    expect(q36::adaptive_t_measured(st, 4) && state.r_seen == 0,
           "T is server-global; hop posterior is per-request");
}

void test_optimistic_cold_start_does_not_invent_all_success() {
    q36::AdaptiveDraftState state;
    q36::seed_adaptive_draft_state(state, 0);
    expect_near(q36::detail::expected_tokens_optimistic(state, 5), 1.0f, 1e-6f,
                "no r_i seen → optimistic E[Y] stays 1");
}

} // namespace

int main() {
    test_capture_set();
    test_seed_is_captured_min();
    test_topology_class();
    test_y_is_one_plus_product_of_r();
    test_unseen_r_not_added_to_mean_y();
    test_r_updates_only_when_prefix_reached();
    test_pcur_zero_skips_update();
    test_t_ols_shared_slope();
    test_unmeasured_t5_extrapolates_and_is_at_least_t4();
    test_cold_start_runs_smallest_k();
    test_measured_expensive_k5_is_dominated();
    test_unmeasured_k5_is_probed_at_most_once_then_dropped();
    test_small_q_extrapolation_skips_k5_probe();
    test_unmeasured_k4_probed_at_most_once();
    test_argmin_t_when_hops_unseen();
    test_ties_keep_smaller_k();
    test_switch_cost_holds_live_k();
    test_stationary_late_hop_does_not_force_k5();
    test_batch_sum_e_over_t();
    test_batch_row_budget_clips_expected_tokens();
    test_batch_next_writes_executed_k();
    test_budget_clamp();
    test_t_survives_request_seed();
    test_optimistic_cold_start_does_not_invent_all_success();
    if (failures != 0) {
        std::cerr << failures << " adaptive draft host checks failed\n";
        return 1;
    }
    std::cout << "adaptive draft host checks passed\n";
    return 0;
}
