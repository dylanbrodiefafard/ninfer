#include "core/paged_kv_cache.h"
#include "runtime/engine/admission_policy.h"

#include <array>
#include <iostream>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    using ninfer::runtime::ActiveAdmissionSnapshot;
    using ninfer::runtime::AdmissionResources;
    using ninfer::runtime::BackfillClass;

    int failures = 0;
    const AdmissionResources capacity{
        .active_lanes     = 4,
        .main_kv_pages    = 160,
        .backend_kv_pages = 128,
    };
    const AdmissionResources head{
        .active_lanes     = 1,
        .main_kv_pages    = 64,
        .backend_kv_pages = 48,
    };
    std::array<ActiveAdmissionSnapshot, 2> incumbents{
        ActiveAdmissionSnapshot{
            .request_id            = 1,
            .resources             = {1, 64, 32},
            .remaining_work_quanta = 100,
        },
        ActiveAdmissionSnapshot{
            .request_id            = 2,
            .resources             = {1, 48, 64},
            .remaining_work_quanta = 20,
        },
    };

    const auto protection = ninfer::runtime::make_admission_protection(
        7, 10, head, std::span<const ActiveAdmissionSnapshot>(incumbents), capacity);
    failures += check(protection.donor_count == 1 && protection.donor_ids[0] == 2 &&
                          protection.temporal_credit == 20,
                      "release frontier did not select the earliest sufficient incumbent");
    failures += check(ninfer::runtime::protection_frontier_distance(protection, incumbents) == 20,
                      "frontier distance did not follow the frozen donor");

    const AdmissionResources persistent_candidate{1, 24, 40};
    failures += check(ninfer::runtime::persistent_backfill_is_safe(protection, incumbents,
                                                                   persistent_candidate, capacity),
                      "future resource surplus rejected a persistent-safe backfill");
    failures += check(!ninfer::runtime::persistent_backfill_is_safe(
                          protection, incumbents, AdmissionResources{1, 40, 60}, capacity),
                      "persistent backfill borrowed protected future capacity");

    std::array<ActiveAdmissionSnapshot, 3> with_persistent{
        incumbents[0],
        incumbents[1],
        ActiveAdmissionSnapshot{
            .request_id            = 3,
            .resources             = persistent_candidate,
            .remaining_work_quanta = 50,
            .backfill_epoch        = 7,
            .backfill_class        = BackfillClass::Persistent,
        },
    };
    failures += check(!ninfer::runtime::persistent_backfill_is_safe(
                          protection, with_persistent, AdmissionResources{1, 9, 9}, capacity),
                      "persistent ledger failed to accumulate earlier backfills");

    std::array<ActiveAdmissionSnapshot, 2> after_donor{
        incumbents[0],
        ActiveAdmissionSnapshot{
            .request_id            = 4,
            .resources             = {1, 32, 64},
            .remaining_work_quanta = 8,
            .backfill_epoch        = 7,
            .backfill_class        = BackfillClass::Temporal,
        },
    };
    failures += check(ninfer::runtime::protection_frontier_distance(protection, after_donor) == 0,
                      "later temporal work changed the frozen frontier");
    failures += check(
        ninfer::runtime::protected_head_safe_without_temporal(protection, after_donor, capacity),
        "released frontier did not mature behind a temporal borrower");

    failures += check(
        !ninfer::runtime::admission_resources_fit(AdmissionResources{1, 161, 1}, capacity) &&
            !ninfer::runtime::admission_resources_fit(AdmissionResources{1, 1, 129}, capacity),
        "independent KV pools were incorrectly treated as interchangeable capacity");

    // Scaled analog of C=3 / max_context=80k / kv_capacity=100k: page size 64, max_context 512
    // (8 pages), pool 640 tokens (10 pages). Small chat 128 prompt + 8 output → 135 reserved → 3
    // pages; large continuation 200 + 48 → 247 reserved → 4 pages; tiny backfill 64 + 2 → 65
    // reserved → 2 pages.
    failures += check(ninfer::pages_for_tokens(128U + 8U - 1U) == 3 &&
                          ninfer::pages_for_tokens(200U + 48U - 1U) == 4 &&
                          ninfer::pages_for_tokens(64U + 2U - 1U) == 2 &&
                          ninfer::pages_for_tokens(640) == 10,
                      "C=3 shared-pool token reservations do not map to 3/4/2/10 pages");
    failures += check(ninfer::runtime::admission_resources_fit({3, 9, 0}, {3, 10, 0}),
                      "three 3-page chats should fit in a 10-page C=3 pool");
    failures += check(!ninfer::runtime::admission_resources_fit({3, 12, 0}, {3, 10, 0}),
                      "three 4-page chats should not fit in a 10-page C=3 pool");
    failures += check(ninfer::runtime::admission_resources_fit({2, 8, 0}, {3, 10, 0}),
                      "two 4-page chats should leave 2 leftover pages");

    const AdmissionResources pool{.active_lanes = 3, .main_kv_pages = 10, .backend_kv_pages = 0};
    const AdmissionResources large{.active_lanes = 1, .main_kv_pages = 4, .backend_kv_pages = 0};
    const AdmissionResources tiny{.active_lanes = 1, .main_kv_pages = 2, .backend_kv_pages = 0};
    std::array<ActiveAdmissionSnapshot, 2> two_large{
        ActiveAdmissionSnapshot{.request_id = 11, .resources = large, .remaining_work_quanta = 48},
        ActiveAdmissionSnapshot{.request_id = 12, .resources = large, .remaining_work_quanta = 40},
    };
    const auto pooled = ninfer::runtime::make_admission_protection(
        3, 13, large, std::span<const ActiveAdmissionSnapshot>(two_large), pool);
    failures += check(pooled.donor_count == 1 && pooled.donor_ids[0] == 12,
                      "C=3 leftover-2 protection did not freeze the earliest 4-page donor");
    failures += check(
        ninfer::runtime::persistent_backfill_is_safe(pooled, two_large, tiny, pool),
        "2-page fits-now backfill was rejected with 2 leftover pages and a free lane");
    failures += check(
        !ninfer::runtime::persistent_backfill_is_safe(pooled, two_large, large, pool),
        "4-page candidate backfilled into leftover-2 capacity reserved for the protected head");
    failures += check(!ninfer::runtime::persistent_backfill_is_safe(
                          pooled, two_large, AdmissionResources{1, 3, 0}, pool),
                      "3-page candidate stole a leftover page from the protected head");

    std::array<ActiveAdmissionSnapshot, 3> after_tiny{
        two_large[0],
        two_large[1],
        ActiveAdmissionSnapshot{.request_id     = 14,
                                .resources      = tiny,
                                .remaining_work_quanta = 2,
                                .backfill_epoch = 3,
                                .backfill_class = BackfillClass::Persistent},
    };
    failures += check(!ninfer::runtime::persistent_backfill_is_safe(pooled, after_tiny, tiny, pool),
                      "second 2-page backfill ignored the first persistent occupant of leftover pages");

    const AdmissionResources five{.active_lanes = 1, .main_kv_pages = 5, .backend_kv_pages = 0};
    std::array<ActiveAdmissionSnapshot, 2> two_five{
        ActiveAdmissionSnapshot{.request_id = 21, .resources = five, .remaining_work_quanta = 10},
        ActiveAdmissionSnapshot{.request_id = 22, .resources = five, .remaining_work_quanta = 8},
    };
    const auto leftover0 = ninfer::runtime::make_admission_protection(
        4, 23, five, std::span<const ActiveAdmissionSnapshot>(two_five), pool);
    failures += check(
        !ninfer::runtime::persistent_backfill_is_safe(leftover0, two_five, tiny, pool),
        "2-page backfill entered a leftover-0 5+5 occupancy that cannot spare pages");

    const AdmissionResources two_lanes{.active_lanes = 2, .main_kv_pages = 10, .backend_kv_pages = 0};
    const auto no_lane = ninfer::runtime::make_admission_protection(
        5, 33, large, std::span<const ActiveAdmissionSnapshot>(two_large), two_lanes);
    failures += check(
        !ninfer::runtime::persistent_backfill_is_safe(no_lane, two_large, tiny, two_lanes),
        "2-page leftover fit ignored that both lanes are occupied");

    // Decode debt state machine: one ready donor in C=3 freezes two admissions. A terminal
    // prefill freeing its lane does not refund debt, so a continuous stream cannot starve decode.
    ninfer::runtime::DecodeAdmissionBurst burst;
    burst.observe_membership(3, 1, 1);
    failures += check(burst.remaining_budget() == 2 && burst.allows_admission() &&
                          burst.admissions() == 0,
                      "first decode membership did not freeze all free lanes as debt");
    burst.consume_admission(); // terminal or cancelled prefill: active slots can still be one.
    failures += check(burst.remaining_budget() == 1 && burst.admissions() == 1,
                      "terminal admission incorrectly refunded decode debt");
    burst.observe_membership(3, 1, 1);
    failures += check(burst.remaining_budget() == 1,
                      "later boundary recomputed rather than preserving frozen debt");
    burst.consume_admission();
    failures += check(!burst.allows_admission() && burst.remaining_budget() == 0 &&
                          burst.admissions() == 2,
                      "C-1 admissions did not force the donor decode");
    bool exhausted_rejected = false;
    try {
        burst.consume_admission();
    } catch (const std::logic_error&) {
        exhausted_rejected = true;
    }
    failures += check(exhausted_rejected, "exhausted decode debt admitted another prefill");
    burst.complete_decode();
    failures += check(!burst.remaining_budget() && burst.admissions() == 0 &&
                          burst.allows_admission(),
                      "decode completion did not start a fresh admission interval");

    // Empty membership is startup/pure-prefill mode, not decode debt. Membership transitions
    // reset stale debt, while malformed worker snapshots are rejected before scheduling work.
    burst.observe_membership(3, 0, 0);
    failures += check(!burst.remaining_budget() && burst.allows_admission(),
                      "pure-prefill mode inherited decode admission debt");
    bool invalid_membership_rejected = false;
    try {
        burst.observe_membership(3, 1, 2);
    } catch (const std::logic_error&) {
        invalid_membership_rejected = true;
    }
    failures += check(invalid_membership_rejected,
                      "membership larger than active slots was accepted");
    bool over_capacity_rejected = false;
    try {
        burst.observe_membership(3, 4, 1);
    } catch (const std::logic_error&) {
        over_capacity_rejected = true;
    }
    failures += check(over_capacity_rejected, "active slots beyond concurrency were accepted");
    bool absent_budget_rejected = false;
    try {
        burst.consume_admission();
    } catch (const std::logic_error&) {
        absent_budget_rejected = true;
    }
    failures += check(absent_budget_rejected, "startup admission consumed nonexistent decode debt");
    burst.observe_membership(3, 3, 2);
    failures += check(burst.remaining_budget() == 0 && !burst.allows_admission(),
                      "full active set incorrectly admitted between decode rounds");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
