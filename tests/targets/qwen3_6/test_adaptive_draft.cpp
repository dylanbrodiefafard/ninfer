#include "targets/qwen3_6/impl/runtime/adaptive_draft.h"

#include "ninfer/types.h"

#include <array>
#include <cmath>
#include <cstdint>
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

std::array<float, 8> mtp_t() {
    std::array<float, 8> t{};
    t[3] = 1.00f;
    t[4] = 1.10f;
    t[5] = 1.16f;
    return t;
}

std::array<float, 8> dflash_t() {
    std::array<float, 8> t{};
    t[3] = 0.91f;
    t[4] = 0.97f;
    t[5] = 1.00f;
    return t;
}

q36::AdaptiveDraftConfig cfg_of(std::span<const std::uint32_t> ks, std::span<const float> t,
                                std::uint32_t warmup = 0, std::uint32_t dwell = 0,
                                std::uint32_t first_remove = 0) {
    return q36::AdaptiveDraftConfig{
        .captured_ks          = ks,
        .round_time           = t,
        .warmup_rounds        = warmup,
        .first_remove_warmup  = first_remove,
        .min_dwell            = dwell,
    };
}

void test_named_constants() {
    expect(q36::kAdaptiveEwma == 32.0f, "kAdaptiveEwma");
    expect(q36::kAdaptiveWarmupRounds == 8, "kAdaptiveWarmupRounds");
    expect(q36::kAdaptiveFirstRemoveWarmup == 32, "kAdaptiveFirstRemoveWarmup");
    expect(q36::kAdaptiveDflashFirstRemoveWarmup == 8, "kAdaptiveDflashFirstRemoveWarmup");
    expect(q36::kAdaptiveK5WaiveWarmup == 8, "kAdaptiveK5WaiveWarmup");
    expect(q36::kAdaptiveMinDwell == 8, "kAdaptiveMinDwell");
    expect(q36::kAdaptiveDeltaAdd == 0.05f, "kAdaptiveDeltaAdd");
    expect(q36::kAdaptiveDeltaRemove == 0.02f, "kAdaptiveDeltaRemove");
    expect(q36::kAdaptiveDropSlotMax == 0.30f, "kAdaptiveDropSlotMax");
    expect(q36::kAdaptiveClimbSlotMin == 0.35f, "kAdaptiveClimbSlotMin");
    expect(q36::kAdaptiveHighP0MinK == 0.65f, "kAdaptiveHighP0MinK");
    expect(q36::kAdaptiveDropTo3Max == 0.08f, "kAdaptiveDropTo3Max");
    expect_near(q36::kAdaptiveEwmaAlpha, 2.0f / 33.0f, 1e-6f, "alpha = 2/(32+1)");
    expect(q36::kAdaptiveMtpT[3] == 1.00f && q36::kAdaptiveMtpT[4] == 1.10f &&
               q36::kAdaptiveMtpT[5] == 1.16f,
           "measured MTP T(K)");
    expect(q36::kAdaptiveMtpC2T[3] == 1.00f && q36::kAdaptiveMtpC2T[4] == 1.10f &&
               q36::kAdaptiveMtpC2T[5] == 1.22f,
           "measured MTP C=2 T(K)");
    expect(q36::kAdaptiveDflashT[3] == 0.91f && q36::kAdaptiveDflashT[4] == 0.97f &&
               q36::kAdaptiveDflashT[5] == 1.00f && q36::kAdaptiveDflashT[6] == 1.25f &&
               q36::kAdaptiveDflashT[7] == 1.25f,
           "measured DFlash T(K)");
    expect(q36::kAdaptiveDflashC2T[3] == 0.83f && q36::kAdaptiveDflashC2T[4] == 0.93f &&
               q36::kAdaptiveDflashC2T[5] == 1.00f,
           "measured DFlash C=2 T(K)");
    expect(q36::kAdaptiveDflashC3T[3] == 0.86f && q36::kAdaptiveDflashC3T[4] == 0.98f &&
               q36::kAdaptiveDflashC3T[5] == 1.00f,
           "measured DFlash C=3 T(K)");
    expect(q36::kAdaptiveDflashC4T[3] == 0.85f && q36::kAdaptiveDflashC4T[4] == 0.94f &&
               q36::kAdaptiveDflashC4T[5] == 1.00f,
           "measured DFlash C>=4 T(K)");
    expect(q36::adaptive_dflash_round_time(2).data() == q36::kAdaptiveDflashC2T,
           "B=2 uses C=2 T");
    expect(q36::adaptive_dflash_round_time(3).data() == q36::kAdaptiveDflashC3T,
           "B=3 uses C=3 T");
    expect(q36::adaptive_dflash_round_time(4).data() == q36::kAdaptiveDflashC4T &&
               q36::adaptive_dflash_round_time(8).data() == q36::kAdaptiveDflashC4T,
           "B>=4 uses C=4 T");
    expect(q36::kAdaptiveE5Campaign == 2.94f && q36::kAdaptiveE7Campaign == 3.21f,
           "campaign E priors");
}

void test_capture_set() {
    using ninfer::SpeculativeBackend;
    const auto eq = [](std::vector<std::uint32_t> got, std::vector<std::uint32_t> want,
                       std::string_view msg) {
        expect(got == want, msg);
    };
    eq(q36::adaptive_draft_ks(SpeculativeBackend::Mtp, 5, false), {5}, "frozen MTP {N}");
    eq(q36::adaptive_draft_ks(SpeculativeBackend::Mtp, 5, true), {3, 4, 5}, "MTP adaptive {3,4,5}");
    eq(q36::adaptive_draft_ks(SpeculativeBackend::Mtp, 4, true), {3, 4}, "MTP N=4");
    eq(q36::adaptive_draft_ks(SpeculativeBackend::Mtp, 3, true), {3}, "MTP N=3");
    eq(q36::adaptive_draft_ks(SpeculativeBackend::Mtp, 2, true), {2}, "MTP N=2 captures {N} only");
    eq(q36::adaptive_draft_ks(SpeculativeBackend::Mtp, 1, true), {1}, "MTP N=1 captures {N} only");
    eq(q36::adaptive_draft_ks(SpeculativeBackend::DFlash, 7, true), {3, 4, 5},
       "DFlash N>=7 {3,4,5}");
    eq(q36::adaptive_draft_ks(SpeculativeBackend::DFlash, 11, true), {3, 4, 5},
       "DFlash N=11 still {3,4,5}");
    eq(q36::adaptive_draft_ks(SpeculativeBackend::DFlash, 6, true), {6}, "DFlash N=6 frozen");
    eq(q36::adaptive_draft_ks(SpeculativeBackend::DFlash, 5, true), {3, 4, 5}, "DFlash N=5 {3,4,5}");
    eq(q36::adaptive_draft_ks(SpeculativeBackend::DFlash, 4, true), {4}, "DFlash N=4 frozen {4}");
    eq(q36::adaptive_draft_ks(SpeculativeBackend::DFlash, 7, false), {7}, "frozen DFlash {N}");
    eq(q36::adaptive_draft_ks(SpeculativeBackend::None, 5, true), {5}, "no backend {N}");
    for (std::uint32_t k : q36::adaptive_draft_ks(SpeculativeBackend::Mtp, 5, true)) {
        expect(k != 1 && k != 2 && k != 6, "MTP adaptive never 1/2/6");
    }
    for (std::uint32_t k : q36::adaptive_draft_ks(SpeculativeBackend::DFlash, 7, true)) {
        expect(k != 6 && k != 7, "DFlash adaptive never 6/7");
    }
}

void test_seed_k() {
    using ninfer::SpeculativeBackend;
    const auto mtp5 = q36::adaptive_draft_ks(SpeculativeBackend::Mtp, 5, true);
    const auto df7  = q36::adaptive_draft_ks(SpeculativeBackend::DFlash, 7, true);
    const auto mtp3 = q36::adaptive_draft_ks(SpeculativeBackend::Mtp, 3, true);
    const auto fz7  = q36::adaptive_draft_ks(SpeculativeBackend::DFlash, 7, false);
    expect(q36::adaptive_seed_k(mtp5, SpeculativeBackend::Mtp) == 4, "MTP N=5 seeds 4");
    expect(q36::adaptive_seed_k(df7, SpeculativeBackend::DFlash) == 5, "DFlash N=7 seeds 5");
    expect(q36::adaptive_seed_k(mtp3, SpeculativeBackend::Mtp) == 3, "MTP N=3 seeds 3");
    expect(q36::adaptive_seed_k(fz7, SpeculativeBackend::DFlash) == 7, "frozen DFlash seeds N");
}

void test_topology_class() {
    const std::uint32_t C = 4;
    expect(q36::adaptive_k_stride(C, 0) == C, "27B k_stride = C");
    for (std::uint32_t B = 1; B <= C; ++B) {
        expect(q36::adaptive_topology_class(0, C, 0, C, B) == (B - 1U),
               "frozen topology_class = B-1");
    }
    expect(q36::adaptive_topology_class(2, C, 0, C, 1) == 2U * C, "k=third class at B=1");
    expect(q36::adaptive_topology_class(1, C, 0, C, 3) == C + 2U, "k_index=1 B=3");
    expect(q36::adaptive_k_stride(C, 3) == C * 4U, "35B k_stride uses max planned");
    expect(q36::adaptive_topology_class(1, C * 4U, 2, C, 2) == 1U * C * 4U + 2U * C + 1U,
           "folded 35B class");
}

void test_batch_k() {
    const std::uint32_t captured[] = {3, 4, 5};
    const std::uint32_t rows[]     = {3, 5, 4};
    expect(q36::adaptive_batch_k(rows, captured) == 5, "batch_k = max(row_k)");
    const std::uint32_t hole_rows[] = {3, 3};
    expect(q36::adaptive_batch_k(hole_rows, captured) == 3, "dense 3 is captured");
    const std::uint32_t dflash[] = {4, 5};
    const std::uint32_t mixed[]  = {4, 4};
    expect(q36::adaptive_batch_k(mixed, dflash) == 4, "DFlash max 4 stays 4");
    expect(q36::adaptive_snap_captured_k(dflash, 6) == 5, "snap above 5 clamps to 5");
    const std::uint32_t snap[] = {4};
    expect(q36::adaptive_batch_k(snap, dflash) == 4, "max 4 needs no snap");

    const std::uint32_t zeros[] = {0, 0};
    expect(q36::adaptive_snap_captured_k(dflash, 0) == 4, "snap(0) is captured.front()");
    expect(q36::adaptive_batch_k(zeros, dflash) == 4, "raw_max==0 on {4,5} uses k=4");
    expect(q36::adaptive_batch_k(zeros, captured) == 3, "raw_max==0 on MTP uses k_min");
    const std::uint32_t frozen_n[] = {7};
    expect(q36::adaptive_batch_k(zeros, frozen_n) == 7, "frozen |K|=1 snap(0) stays N");
    const std::uint32_t mixed_hi[] = {0, 5, 0};
    expect(q36::adaptive_batch_k(mixed_hi, dflash) == 5, "one live k=5 still selects 5");
    const std::uint32_t mixed_lo[] = {0, 4};
    expect(q36::adaptive_batch_k(mixed_lo, dflash) == 4, "max 4 with a zero row stays 4");
    std::span<const std::uint32_t> empty_rows{};
    expect(q36::adaptive_batch_k(empty_rows, dflash) == 4, "empty row span snaps to front");
}

void pump(q36::AdaptiveDraftState& state, const q36::AdaptiveDraftConfig& cfg, std::uint32_t k,
          std::uint32_t accepted, std::uint32_t drafted, std::uint32_t n, std::uint32_t budget) {
    for (std::uint32_t i = 0; i < n; ++i) {
        (void)q36::adaptive_draft_next(cfg, state, accepted, drafted, budget, k);
        k = state.live_k;
    }
}

void plant(q36::AdaptiveDraftState& state, std::uint32_t live_k,
           std::initializer_list<float> slots) {
    q36::seed_adaptive_draft_state(state, live_k);
    // Large n so ewma_rate on planted μ is identity (slots were not EWMA'd from 0.5).
    state.observed    = 256;
    state.rounds_at_k = 16;
    std::uint32_t i   = 0;
    for (float s : slots) {
        if (i >= 5) { break; }
        state.s[i] = s;
        state.s_seen |= static_cast<std::uint8_t>(1U << i);
        ++i;
    }
}

void test_batch_k_sum_score() {
    const std::uint32_t captured[] = {3, 4, 5};
    const auto t                   = std::span<const float>(q36::kAdaptiveMtpC2T);
    q36::AdaptiveDraftState a;
    q36::AdaptiveDraftState b;
    q36::seed_adaptive_draft_state(a, 4);
    q36::seed_adaptive_draft_state(b, 5);
    const q36::AdaptiveDraftState* states[] = {&a, &b};
    const std::uint32_t rows[]              = {4, 5};
    expect(q36::adaptive_batch_k_sum_score(states, rows, captured, t) == 5,
           "unobserved C=2 explores max(row_k), not min T");

    const std::uint32_t ks[] = {3, 4, 5};
    const auto t1            = mtp_t();
    auto cfg                 = cfg_of(ks, t1, 0, 0);
    pump(a, cfg, 5, 5, 5, 8, 5);
    pump(b, cfg, 5, 5, 5, 8, 5);
    a.live_k = 5;
    b.live_k = 5;
    const std::uint32_t hot_rows[] = {5, 5};
    expect(q36::adaptive_batch_k_sum_score(states, hot_rows, captured, t) == 5,
           "full-accept C=2 rows keep k=5");

    const auto td = std::span<const float>(q36::kAdaptiveDflashC2T);
    q36::AdaptiveDraftState da;
    q36::AdaptiveDraftState db;
    q36::seed_adaptive_draft_state(da, 4);
    q36::seed_adaptive_draft_state(db, 5);
    const q36::AdaptiveDraftState* dflash_states[] = {&da, &db};
    const std::uint32_t dflash_rows[]              = {4, 5};
    expect(q36::adaptive_batch_k_sum_score(dflash_states, dflash_rows, captured, td) == 5,
           "unobserved C=2 DFlash explores max(row_k)");

    // Borderline 4-vs-5: old T[4]=0.89 picks 4; measured T[4]=0.93 picks 5.
    q36::AdaptiveDraftState ha;
    q36::AdaptiveDraftState hb;
    plant(ha, 5, {0.90f, 0.85f, 0.75f, 0.45f, 0.32f});
    plant(hb, 5, {0.90f, 0.85f, 0.75f, 0.45f, 0.32f});
    const q36::AdaptiveDraftState* hot_pair[] = {&ha, &hb};
    const std::uint32_t hot_cap[]             = {5, 5};
    const float t_old[8] = {0.0f, 0.0f, 0.0f, 0.83f, 0.89f, 1.00f};
    expect(q36::adaptive_batch_k_sum_score(hot_pair, hot_cap, captured, t_old) == 4,
           "optimistic T[4]=0.89 over-picks k=4");
    expect(q36::adaptive_batch_k_sum_score(hot_pair, hot_cap, captured, td) == 5,
           "measured C=2 T[4]=0.93 picks k=5");
    const auto t3 = std::span<const float>(q36::kAdaptiveDflashC3T);
    const auto t4 = std::span<const float>(q36::kAdaptiveDflashC4T);
    expect(q36::adaptive_batch_k_sum_score(hot_pair, hot_cap, captured, t3) == 5,
           "C=3 T[4]~T[5] keeps k=5 on this E");
    expect(q36::adaptive_batch_k_sum_score(hot_pair, hot_cap, captured, t4) == 5,
           "C>=4 T[4]=0.94 still picks k=5 on this E");

    q36::AdaptiveDraftState cold;
    q36::seed_adaptive_draft_state(cold, 3);
    pump(cold, cfg, 3, 0, 3, 8, 5);
    const q36::AdaptiveDraftState* mixed[] = {&cold, &cold};
    const std::uint32_t cold_rows[]        = {3, 3};
    expect(q36::adaptive_batch_k_sum_score(mixed, cold_rows, captured, t) == 3,
           "dead extra slots stay at capped k=3");

    // logic+dialogue / csv+scifi: 3-vs-5 wants, empirical shared K is 4.
    q36::AdaptiveDraftState logic;
    q36::AdaptiveDraftState dlg;
    plant(logic, 5, {0.90f, 0.85f, 0.75f, 0.66f, 0.23f});
    plant(dlg, 3, {0.40f, 0.14f, 0.05f, 0.02f, 0.01f});
    const q36::AdaptiveDraftState* mid[] = {&logic, &dlg};
    const std::uint32_t mid_rows[]       = {5, 3};
    expect(q36::adaptive_batch_k_sum_score(mid, mid_rows, captured, td) == 4,
           "C=2 mixed 3-vs-5 with mid E picks shared k=4, not min or max");
}

void test_pcur_zero() {
    const std::uint32_t ks[] = {3, 4, 5};
    const auto t             = mtp_t();
    auto cfg                 = cfg_of(ks, t, 0, 0);
    q36::AdaptiveDraftState state;
    q36::seed_adaptive_draft_state(state, 4);
    const auto before = state;
    const std::uint32_t next =
        q36::adaptive_draft_next(cfg, state, 0, 0, 5, 4);
    expect(next == 4, "pcur==0 keeps live_k");
    expect(state.observed == before.observed && state.s_seen == 0, "pcur==0 skips EWMA");
}

void test_warmup_and_dwell() {
    const std::uint32_t ks[] = {3, 4, 5};
    const auto t             = mtp_t();
    auto cfg                 = cfg_of(ks, t, q36::kAdaptiveWarmupRounds, q36::kAdaptiveMinDwell,
                                      q36::kAdaptiveFirstRemoveWarmup);
    q36::AdaptiveDraftState state;
    q36::seed_adaptive_draft_state(state, 4);
    // Budget 4: equal-prior slots otherwise score-climb 4→5 before first-remove.
    pump(state, cfg, 4, 0, 4, q36::kAdaptiveFirstRemoveWarmup - 1, 4);
    expect(state.live_k == 4, "first-remove warmup blocks 4→3");
    pump(state, cfg, 4, 0, 4, q36::kAdaptiveMinDwell, 4);
    expect(state.live_k == 3, "remove after first-remove warmup+dwell when extra slot is dead");
}

void test_first_observation_and_ewma() {
    const std::uint32_t ks[] = {3, 4, 5};
    const auto t             = mtp_t();
    auto cfg                 = cfg_of(ks, t, 100, 100);
    q36::AdaptiveDraftState state;
    q36::seed_adaptive_draft_state(state, 5);
    (void)q36::adaptive_draft_next(cfg, state, 5, 5, 5, 5);
    expect(state.s_seen == 0x1f, "all five chain slots seen");
    const float first = (1.0f - q36::kAdaptiveEwmaAlpha) * q36::kAdaptivePbarPrior +
                        q36::kAdaptiveEwmaAlpha * 1.0f;
    expect_near(state.s[0], first, 1e-5f, "first observation EWMAs from p̄=0.5");
    expect_near(state.s[4], first, 1e-5f, "first observation EWMAs from p̄=0.5");
    (void)q36::adaptive_draft_next(cfg, state, 0, 5, 5, 5);
    expect_near(state.s[0], (1.0f - q36::kAdaptiveEwmaAlpha) * first, 1e-5f, "second sample EWMAs");
}

void test_geometric_add() {
    const std::uint32_t ks[] = {3, 4, 5};
    const auto t             = mtp_t();
    auto cfg                 = cfg_of(ks, t, 0, 0);
    q36::AdaptiveDraftState state;
    q36::seed_adaptive_draft_state(state, 3);
    pump(state, cfg, 3, 3, 3, 8, 5);
    expect(state.live_k >= 4, "geometric add climbs from 3");
}

void test_remove() {
    const std::uint32_t ks[] = {3, 4, 5};
    const auto t             = mtp_t();
    auto cfg                 = cfg_of(ks, t, 0, 0);
    q36::AdaptiveDraftState state;
    q36::seed_adaptive_draft_state(state, 5);
    pump(state, cfg, 5, 0, 5, 80, 5);
    expect(state.live_k < 5, "zero accept removes from 5");
}

void test_budget_clamp() {
    const std::uint32_t ks[] = {3, 4, 5};
    const auto t             = mtp_t();
    auto cfg                 = cfg_of(ks, t, 0, 0);
    q36::AdaptiveDraftState state;
    q36::seed_adaptive_draft_state(state, 5);
    const std::uint32_t next = q36::adaptive_draft_next(cfg, state, 5, 5, 3, 5);
    expect(next == 3, "budget_extent clamps to captured k<=budget");
}

void test_dflash_chain_4_5() {
    const std::uint32_t ks[] = {4, 5};
    const auto t             = dflash_t();
    auto cfg                 = cfg_of(ks, t, 0, 0);
    q36::AdaptiveDraftState state;
    q36::seed_adaptive_draft_state(state, 5);
    pump(state, cfg, 5, 0, 5, 80, 5);
    expect(state.live_k == 4, "zero accept drops 5→4");

    q36::seed_adaptive_draft_state(state, 4);
    pump(state, cfg, 4, 4, 4, 12, 5);
    expect(state.live_k == 5, "full accept climbs 4→5");

    q36::seed_adaptive_draft_state(state, 5);
    pump(state, cfg, 5, 5, 5, 40, 5);
    expect(state.live_k == 5, "full accept at 5 does not drop: extra slot is hot");
}

void test_dflash_chain_3_4_5() {
    const std::uint32_t ks[] = {3, 4, 5};
    const auto t             = dflash_t();
    auto cfg                 = cfg_of(ks, t, 0, 0);
    cfg.drop_to_3_max        = q36::kAdaptiveDropTo3Max;
    q36::AdaptiveDraftState state;

    q36::seed_adaptive_draft_state(state, 5);
    pump(state, cfg, 5, 0, 5, 80, 5);
    expect(state.live_k == 3, "zero accept on {3,4,5} drops through 4 to 3");

    // zh-dialogue frozen k=4: extra of 4 is dead and score(3) wins.
    plant(state, 4, {0.402f, 0.156f, 0.054f, 0.014f});
    (void)q36::adaptive_draft_next(cfg, state, 1, 4, 5, 4);
    expect(state.live_k == 3, "dialogue-like 4→3 when extra is dead and score 3 wins");

    // AIME frozen k=4: extra of 4 is still dead (<0.30) but score 4 beats 3.
    plant(state, 4, {0.713f, 0.480f, 0.302f, 0.205f});
    (void)q36::adaptive_draft_next(cfg, state, 3, 4, 5, 4);
    expect(state.live_k == 4, "AIME-like extra-dead 4 does not drop to 3 on score");

    // EN story frozen k=4: extra 0.115 is above drop_to_3_max=0.08, so 4 holds.
    plant(state, 4, {0.697f, 0.443f, 0.247f, 0.115f});
    (void)q36::adaptive_draft_next(cfg, state, 2, 4, 5, 4);
    expect(state.live_k == 4, "story-like extra of 4 is not dead enough for 4→3");

    plant(state, 3, {0.90f, 0.80f, 0.50f});
    (void)q36::adaptive_draft_next(cfg, state, 3, 3, 5, 3);
    expect(state.live_k == 4, "last-hot 3→4 needs extra of 3 > 0.35");

    plant(state, 3, {0.402f, 0.156f, 0.054f});
    (void)q36::adaptive_draft_next(cfg, state, 1, 3, 5, 3);
    expect(state.live_k == 3, "dialogue extra of 3 is cold; does not last-hot to 4");

    // CUDA frozen k=5: s[4]=0.28 is dead, but score(4) from those slots loses to
    // score(5) at T[5]/T[4]=1.03. Waive the score gate only for 5→4.
    const float cuda_mu[] = {0.781f, 0.624f, 0.494f, 0.379f, 0.282f};
    const float aime_mu[] = {0.75f, 0.52f, 0.338f, 0.235f, 0.164f};
    auto load_prior = [&](q36::AdaptiveDraftState& st, const float* mu, std::uint32_t n) {
        q36::seed_adaptive_draft_state(st, 5);
        st.observed    = n;
        st.rounds_at_k = 16;
        const float w  = std::pow(1.0f - q36::kAdaptiveEwmaAlpha, static_cast<float>(n));
        for (int i = 0; i < 5; ++i) {
            st.s[static_cast<std::size_t>(i)] =
                mu[i] + (q36::kAdaptivePbarPrior - mu[i]) * w;
            st.s_seen |= static_cast<std::uint8_t>(1U << i);
        }
    };
    load_prior(state, cuda_mu, q36::kAdaptiveK5WaiveWarmup - 2);
    (void)q36::adaptive_draft_next(cfg, state, 4, 5, 5, 5);
    expect(state.live_k == 5, "5→4 waiver waits for 8 observations");
    load_prior(state, cuda_mu, q36::kAdaptiveK5WaiveWarmup - 1);
    (void)q36::adaptive_draft_next(cfg, state, 4, 5, 5, 5);
    expect(state.live_k == 4, "CUDA-like dead extra drops 5→4 at 8");

    // Score 5→4 still waits 32. Cold hop-4 so the last-hot waiver cannot fire.
    plant(state, 5, {0.75f, 0.52f, 0.338f, 0.235f, 0.164f});
    state.observed = 8;
    (void)q36::adaptive_draft_next(cfg, state, 3, 5, 5, 5);
    expect(state.live_k == 5, "5→4 score path waits for 32 when hop 4 is cold");

    plant(state, 5, {0.781f, 0.624f, 0.494f, 0.379f, 0.282f});
    (void)q36::adaptive_draft_next(cfg, state, 3, 5, 5, 5);
    expect(state.live_k == 4, "CUDA-like dead k=5 extra drops 5→4 without score");
    plant(state, 4, {0.95f, 0.85f, 0.70f, 0.20f, 0.10f});
    (void)q36::adaptive_draft_next(cfg, state, 3, 4, 5, 4);
    expect(state.live_k == 4, "k=4 with cold last slot and losing score(5) stays at 4");

    // Frozen AIME_15 k=5: s[4] is dead but s[3] is not last-hot. Score(4) also
    // loses at T[5]/T[4]=1.03, so 5 holds. CUDA waiver must not fire here.
    plant(state, 5, {0.75f, 0.52f, 0.338f, 0.235f, 0.164f});
    (void)q36::adaptive_draft_next(cfg, state, 3, 5, 5, 5);
    expect(state.live_k == 5, "AIME-like cold s[3] does not waive 5→4");

    // Production at 8/32 is EWMA-from-0.5. Unbias recovers μ for dead/hot.
    const float w8  = std::pow(1.0f - q36::kAdaptiveEwmaAlpha, 8.0f);
    const float w32 = std::pow(1.0f - q36::kAdaptiveEwmaAlpha, 32.0f);
    auto prior_at   = [](float mu, float w) { return mu + (q36::kAdaptivePbarPrior - mu) * w; };
    q36::AdaptiveDraftState from_prior;
    q36::seed_adaptive_draft_state(from_prior, 5);
    from_prior.observed = 8;
    for (int i = 0; i < 5; ++i) {
        from_prior.s[static_cast<std::size_t>(i)] = prior_at(cuda_mu[i], w8);
        from_prior.s_seen |= static_cast<std::uint8_t>(1U << i);
    }
    expect(q36::detail::extra_slot_dead(from_prior, 5, q36::kAdaptiveDropSlotMax),
           "CUDA EWMA-from-0.5 extra of 5 is dead at 8 after unbias");
    expect(q36::detail::last_slot_hot(from_prior, 4, q36::kAdaptiveClimbSlotMin),
           "CUDA EWMA-from-0.5 hop 4 is last-hot at 8 after unbias");
    for (int i = 0; i < 5; ++i) {
        from_prior.s[static_cast<std::size_t>(i)] = prior_at(aime_mu[i], w8);
    }
    expect(!q36::detail::last_slot_hot(from_prior, 4, q36::kAdaptiveClimbSlotMin),
           "AIME EWMA-from-0.5 hop 4 is cold at 8 after unbias");
    from_prior.observed = 32;
    for (int i = 0; i < 5; ++i) {
        from_prior.s[static_cast<std::size_t>(i)] = prior_at(cuda_mu[i], w32);
    }
    expect(q36::detail::extra_slot_dead(from_prior, 5, q36::kAdaptiveDropSlotMax) &&
               q36::detail::last_slot_hot(from_prior, 4, q36::kAdaptiveClimbSlotMin),
           "CUDA EWMA-from-0.5 at 32 still last-hot + extra-dead");
    for (int i = 0; i < 5; ++i) {
        from_prior.s[static_cast<std::size_t>(i)] = prior_at(aime_mu[i], w32);
    }
    expect(q36::detail::extra_slot_dead(from_prior, 5, q36::kAdaptiveDropSlotMax),
           "AIME EWMA-from-0.5 extra of 5 is dead after unbias");
    expect(!q36::detail::last_slot_hot(from_prior, 4, q36::kAdaptiveClimbSlotMin),
           "AIME EWMA-from-0.5 hop 4 is cold after unbias");

    // Python frozen k=5: s[4]=0.47 stays above drop_slot_max; extra is not dead.
    plant(state, 5, {0.880f, 0.760f, 0.640f, 0.556f, 0.474f});
    (void)q36::adaptive_draft_next(cfg, state, 5, 5, 5, 5);
    expect(state.live_k == 5, "Python-like hot k=5 extra holds 5");
}

void test_seed_isolates_request() {
    const std::uint32_t ks[] = {4, 5};
    const auto t             = dflash_t();
    auto cfg                 = cfg_of(ks, t, 0, 0);
    q36::AdaptiveDraftState state;
    q36::seed_adaptive_draft_state(state, 5);
    pump(state, cfg, 5, 0, 5, 80, 5);
    expect(state.live_k == 4 && state.observed > 0, "pre-seed dropped to 4");
    q36::seed_adaptive_draft_state(state, 5);
    expect(state.live_k == 5, "new request reseeds attractor 5");
    expect(state.rounds_at_k == 0 && state.observed == 0 && state.s_seen == 0, "EWMA wiped");
    for (float s : state.s) { expect(s == 0.0f, "s[] wiped on seed"); }
}

void test_hysteresis_equal_score_holds() {
    const std::uint32_t ks[] = {3, 4, 5};
    auto t                   = mtp_t();
    t[5]                     = 100.0f;
    auto cfg                 = cfg_of(ks, t, 0, 0);
    q36::AdaptiveDraftState state;
    q36::seed_adaptive_draft_state(state, 4);
    (void)q36::adaptive_draft_next(cfg, state, 4, 4, 5, 4);
    expect(state.live_k == 4, "expensive K+1 loses to live even at full accept");
}

void test_hot_extra_slot_holds_k4() {
    const std::uint32_t ks[] = {3, 4, 5};
    const auto t             = mtp_t();
    auto cfg                 = cfg_of(ks, t, 0, 0);
    q36::AdaptiveDraftState state;
    q36::seed_adaptive_draft_state(state, 4);
    pump(state, cfg, 4, 4, 4, 40, 5);
    expect(state.live_k >= 4, "full accept at 4 does not dump to 3");
}

void test_p0_floor_blocks_k3() {
    const std::uint32_t ks[] = {3, 4, 5};
    const auto t             = mtp_t();
    auto cfg                 = cfg_of(ks, t, 0, 0);
    q36::AdaptiveDraftState state;
    q36::seed_adaptive_draft_state(state, 4);
    // High s0 (accept first draft every round) with a dead extra slot would otherwise 4→3.
    pump(state, cfg, 4, 1, 4, 40, 5);
    expect(state.live_k >= 4, "s0>0.65 floors min_k at 4");
}

void test_hot_last_slot_climbs_without_delta() {
    const std::uint32_t ks[] = {3, 4, 5};
    const auto t             = mtp_t();
    auto cfg                 = cfg_of(ks, t, 0, 0);
    q36::AdaptiveDraftState state;
    q36::seed_adaptive_draft_state(state, 3);
    pump(state, cfg, 3, 3, 3, 12, 5);
    expect(state.live_k >= 4, "hot s[2] climbs 3→4 even when score hysteresis is tight");
}

void test_hot_last_slot_probes_k5() {
    const std::uint32_t ks[] = {3, 4, 5};
    auto t                   = mtp_t();
    t[5]                     = 1.28f;
    auto cfg                 = cfg_of(ks, t, 0, 0);
    q36::AdaptiveDraftState state;
    q36::seed_adaptive_draft_state(state, 4);
    pump(state, cfg, 4, 4, 4, 16, 5);
    expect(state.live_k == 5, "hot last slot probes 4→5 when score_hi still beats live");
}

void test_cold_last_slot_does_not_probe() {
    const std::uint32_t ks[] = {3, 4, 5};
    auto t                   = mtp_t();
    t[5]                     = 1.28f;
    auto cfg                 = cfg_of(ks, t, 0, 0, 1000);
    q36::AdaptiveDraftState state;
    q36::seed_adaptive_draft_state(state, 4);
    pump(state, cfg, 4, 2, 4, 16, 5);
    expect(state.live_k == 4, "cold extra slot does not probe 4→5 without δ_add");
}

void test_seen_k5_slot_may_waive() {
    const std::uint32_t ks[] = {3, 4, 5};
    auto t                   = mtp_t();
    t[5]                     = 1.28f;
    auto cfg                 = cfg_of(ks, t, 0, 0);
    q36::AdaptiveDraftState state;
    q36::seed_adaptive_draft_state(state, 5);
    pump(state, cfg, 5, 5, 5, 4, 5);
    expect((state.s_seen & (1U << 4)) != 0, "k=5 observes s[4]");
    state.live_k      = 4;
    state.rounds_at_k = 8;
    pump(state, cfg, 4, 4, 4, 8, 5);
    expect(state.live_k == 5, "after s[4] is seen, last-hot may waive δ_add to return to 5");
}

void test_p0_floor_waits_for_warmup() {
    const std::uint32_t ks[] = {3, 4, 5};
    const auto t             = mtp_t();
    auto cfg                 = cfg_of(ks, t, q36::kAdaptiveWarmupRounds, 0, 1000);
    q36::AdaptiveDraftState state;
    q36::seed_adaptive_draft_state(state, 3);
    pump(state, cfg, 3, 1, 3, 1, 5);
    expect(state.live_k == 3, "first boolean s[0]=1 does not floor to 4");
    pump(state, cfg, 3, 1, 3, q36::kAdaptiveWarmupRounds - 1, 5);
    expect(state.live_k >= 4, "p0 floor applies once observed >= warmup");
}

void test_conservative_unseen_tail() {
    q36::AdaptiveDraftState state;
    q36::seed_adaptive_draft_state(state, 4);
    state.s[0]    = 0.90f;
    state.s[1]    = 0.80f;
    state.s[2]    = 0.70f;
    state.s[3]    = 0.20f;
    state.s_seen  = 0x0F;
    const float mean = q36::detail::mean_conditional_p(state);
    const float last = q36::detail::last_conditional_p(state);
    const float pbar = q36::detail::geometric_pbar(state);
    expect(last < mean, "last conditional below mean on a dying tail");
    expect(pbar == last, "unseen tail uses min(mean, last)");
}

} // namespace

int main() {
    test_named_constants();
    test_capture_set();
    test_seed_k();
    test_topology_class();
    test_batch_k();
    test_batch_k_sum_score();
    test_pcur_zero();
    test_warmup_and_dwell();
    test_first_observation_and_ewma();
    test_geometric_add();
    test_remove();
    test_budget_clamp();
    test_dflash_chain_4_5();
    test_dflash_chain_3_4_5();
    test_seed_isolates_request();
    test_hysteresis_equal_score_holds();
    test_hot_extra_slot_holds_k4();
    test_p0_floor_blocks_k3();
    test_hot_last_slot_climbs_without_delta();
    test_hot_last_slot_probes_k5();
    test_cold_last_slot_does_not_probe();
    test_seen_k5_slot_may_waive();
    test_p0_floor_waits_for_warmup();
    test_conservative_unseen_tail();
    if (failures != 0) {
        std::cerr << failures << " adaptive draft host checks failed\n";
        return 1;
    }
    std::cout << "adaptive draft host checks passed\n";
    return 0;
}
