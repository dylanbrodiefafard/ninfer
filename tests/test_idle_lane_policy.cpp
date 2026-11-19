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

    Policy disabled(0s);
    disabled.idle(0, false, start);
    disabled.idle(1, true, start);
    expect(!disabled.deadline() && !disabled.expired(start + 1000s), "zero must disable eviction");
    return failures == 0 ? 0 : 1;
}
