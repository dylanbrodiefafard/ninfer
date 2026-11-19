#include "runtime/engine/idle_lane_policy.h"

#include <chrono>
#include <iostream>

int main() {
    using Policy = ninfer::runtime::IdleLanePolicy;
    using namespace std::chrono_literals;
    const auto start  = Policy::Clock::time_point{};
    int failures      = 0;
    const auto expect = [&](bool value, const char* message) {
        if (!value) {
            std::cerr << message << '\n';
            ++failures;
        }
    };
    Policy policy(30s);
    expect(!policy.deadline(), "admission/active time must not start an idle timer");
    policy.idle(0, false, start + 100s);
    expect(!policy.deadline(), "the most recently idle lane must be exempt");
    policy.idle(1, false, start + 110s);
    expect(policy.deadline() == start + 130s, "timeout must start at completion, not admission");
    expect(!policy.expired(start + 129s), "idle lane expired before its timeout");
    const auto first = policy.expired(start + 130s);
    expect(first && first->lane == 0 && !first->disposable, "ordinary original must migrate");
    policy.activity(0);
    expect(!policy.deadline(), "reattachment/fork activity must cancel the old timer");
    policy.idle(0, true, start + 140s);
    const auto second = policy.expired(start + 140s);
    expect(second && second->lane == 1, "becoming idle must restart recency and its timer");
    policy.activity(1);
    policy.idle(1, false, start + 150s);
    expect(!policy.expired(start + 169s), "disposable fork source timer was not restarted");
    const auto disposable = policy.expired(start + 170s);
    expect(disposable && disposable->lane == 0 && disposable->disposable,
           "disposable idle state must release, not offload");
    policy.retry(0, start + 170s);
    expect(!policy.expired(start + 199s), "failed capture must wait another full timeout");
    const auto retried = policy.expired(start + 200s);
    expect(retried && retried->lane == 0, "capture retry changed the latest-idle exemption");
    policy.activity(0);
    expect(!policy.expired(start + 1000s), "the sole idle lane must remain exempt");
    Policy migration(30s);
    migration.idle(2, false, start);
    migration.idle(0, false, start + 1s);
    migration.idle(1, false, start + 100s);
    const auto oldest = migration.expired(start + 200s);
    expect(oldest && oldest->lane == 2, "expiry must choose idle age, not lane index");
    migration.await_capture(2, start + 200s);
    expect(migration.capturing(2),
           "a successful capture must retain its source pending completion");
    const auto other = migration.expired(start + 200s);
    expect(other && other->lane == 0, "an outstanding capture must not steal the exemption");
    migration.retry(0, start + 200s);
    migration.await_capture(2, start + 201s);
    expect(!migration.expired(start + 201s), "completion waits must not become tight retries");
    migration.retry(2, start + 202s);
    expect(!migration.capturing(2) && !migration.expired(start + 229s),
           "failed capture must preserve both lanes and the newest-idle exemption");
    const auto retry_old = migration.expired(start + 232s);
    expect(retry_old && retry_old->lane == 2,
           "capture/retry must not promote an old idle lane over a newer one");
    migration.await_capture(2, start + 232s);
    migration.activity(2);
    expect(!migration.capturing(2), "reattachment must cancel an outstanding expiry decision");

    // Demand-based stop: expired lanes stay retained while free KV covers demand + headroom.
    expect(ninfer::runtime::idle_eviction_needed(14, 2, 13) &&
               !ninfer::runtime::idle_eviction_needed(15, 2, 13) &&
               ninfer::runtime::idle_eviction_needed(1, 2, 0) &&
               !ninfer::runtime::idle_eviction_needed(2, 2, 0),
           "eviction threshold must be lendable extents < demand + headroom extents");
    Policy demand(30s);
    demand.idle(0, false, start);
    demand.idle(1, false, start + 1s);
    demand.idle(2, false, start + 2s);
    const auto now = start + 40s;
    expect(!demand.due(now, false), "covered demand must not start an expiry");
    const auto held = demand.expired(now);
    expect(!demand.capturing(0) && held && held->lane == 0,
           "a held lane must stay retained with its timer expired");
    expect(demand.deadline() == now + Policy::kRecheckInterval,
           "a held expiry must wake at the recheck interval, not spin");
    expect(!demand.due(now + 500ms, false) &&
               demand.deadline() == now + 500ms + Policy::kRecheckInterval,
           "every boundary re-evaluates the condition and keeps the wake bounded");
    const auto resumed = demand.due(now + 600ms, true);
    expect(resumed && resumed->lane == 0,
           "a later boundary under pressure must evict oldest first");
    demand.await_capture(0, now + 600ms);
    // The capture phase continues to its close while demand is covered, and before other lanes.
    const auto closing = demand.due(now + 602ms, false);
    expect(closing && closing->lane == 0, "an in-flight capture must close without pressure");
    demand.activity(0);
    expect(!demand.due(now + 603ms, false), "covered demand must again hold the next lane");
    expect(demand.deadline() == now + 603ms + Policy::kRecheckInterval,
           "the next lane must wait for the recheck interval");
    demand.idle(0, false, now + 604ms);
    const auto exempt_skipped = demand.due(now + 700ms, true);
    expect(exempt_skipped && exempt_skipped->lane == 1,
           "exemption is unchanged: the newest idle lane 0 is skipped");
    // Expiries never overlap: while one lane captures, an unstarted expired lane waits even
    // under pressure, and the capturing lane's close is unaffected.
    Policy overlap(30s);
    overlap.idle(0, false, start);
    overlap.idle(1, false, start + 1s);
    overlap.idle(2, false, start + 2s);
    overlap.await_capture(0, start + 40s);
    expect(!overlap.due(start + 40s, true), "a new expiry must not start during a capture");
    const auto close_first = overlap.due(start + 41s, true);
    expect(close_first && close_first->lane == 0, "the capturing lane must still close");
    overlap.activity(0);
    const auto next = overlap.due(start + 41s, true);
    expect(next && next->lane == 1, "the next expiry starts once no lane is capturing");
    // A lane that becomes idle or active clears an earlier hold's pushed-out wake.
    Policy hold(30s);
    hold.idle(0, false, start);
    hold.idle(1, false, start + 1s);
    hold.idle(2, false, start + 2s);
    expect(!hold.due(start + 40s, false) && hold.deadline() == start + 41s, "hold must arm");
    hold.activity(2);
    expect(hold.deadline() == start + 30s, "activity must clear the pushed-out wake");
    // A younger capturing lane is processed ahead of an older lane that has not started.
    Policy order(30s);
    order.idle(0, false, start);
    order.idle(1, false, start + 1s);
    order.idle(2, false, start + 2s);
    order.await_capture(1, start + 40s);
    const auto capture_first = order.due(start + 41s, false);
    expect(capture_first && capture_first->lane == 1,
           "a lane in its capture phase must precede unstarted expiries");

    Policy disabled(0s);
    disabled.idle(0, false, start);
    disabled.idle(1, true, start);
    expect(!disabled.deadline() && !disabled.expired(start + 1000s), "zero must disable eviction");
    return failures == 0 ? 0 : 1;
}
