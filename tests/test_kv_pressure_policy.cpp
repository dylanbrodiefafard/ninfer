#include "runtime/engine/kv_pressure_policy.h"

#include <array>
#include <iostream>
#include <vector>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

ninfer::runtime::KvPressureRequest request(std::uint64_t id, bool blocked = false) {
    return {.id               = id,
            .queue_order      = id,
            .resident         = true,
            .growth_blocked   = blocked,
            .suspend_ready    = true,
            .suspension_bytes = 100};
}

} // namespace

int main() {
    using ninfer::runtime::KvPressurePolicy;
    using ninfer::runtime::KvPressureReason;
    using ninfer::runtime::KvPressureRequest;
    int failures = 0;

    KvPressurePolicy policy(4, 12);
    std::vector<KvPressureRequest> active{request(1, true), request(2), request(3)};
    active[1].completion_member = true;
    policy.observe(active, true);
    failures += check(!policy.pressure_needed() && !policy.draining(),
                      "single blocked lane disturbed runnable peers");
    failures += check(policy.ranked_victims(active, KvPressureReason::Growth).count == 0,
                      "single blocked lane triggered growth eviction");
    auto victims = policy.ranked_victims(active, KvPressureReason::WaitingReplacement);
    failures += check(victims.count == 1 && victims.sets[0].mask == 1,
                      "queued replacement evicted an unblocked or completion member");
    failures += check(policy.oldest_incumbent() == 1 && policy.incumbent_ids().size() == 3,
                      "pressure did not freeze original incumbent identities");

    // Early completion of bypassed work must not refund the finite allowance. Removing a
    // nonoldest incumbent also cannot reset the epoch while fresh work continues arriving.
    for (std::uint64_t id = 4; id < 7; ++id) {
        failures += check(policy.allows_ingress(active.size(), false),
                          "finite replacement allowance ended prematurely");
        policy.record_ingress();
        active.push_back(request(id));
        policy.observe(active, true);
        active.pop_back();
        policy.observe(active, true);
    }
    failures += check(policy.draining() && policy.bypass_remaining() == 0 &&
                          !policy.allows_ingress(active.size(), false),
                      "fresh ingress renewed exhausted pressure credit");
    active.erase(active.begin() + 2);
    policy.observe(active, true);
    failures += check(policy.draining() && policy.oldest_incumbent() == 1,
                      "nonoldest completion cleared pressure drain");
    active.erase(active.begin());
    policy.observe(active, true);
    failures += check(!policy.draining() && !policy.oldest_incumbent(),
                      "oldest completion did not release pressure epoch");

    KvPressurePolicy pressure(4, 12);
    std::array rows{request(1, true), request(2, true), request(3), request(4)};
    rows[3].completion_member = true;
    rows[0].suspension_bytes  = 300;
    pressure.observe(rows, true);
    failures += check(pressure.draining() && pressure.pressure_needed(),
                      "second blocked request did not open evacuation drain");
    victims = pressure.ranked_victims(rows, KvPressureReason::Growth);
    failures += check(victims.count == 7 && victims.sets[0].mask == 2 &&
                          victims.sets[1].mask == 1 && victims.sets[2].mask == 4,
                      "victims did not prefer minimal blocked captures by transfer cost");
    // Exact geometry is deliberately external. Simulate only a two-victim post-transition
    // fit: the first feasible ranked set must remain minimal and exclude protected members.
    const auto feasible = std::find_if(victims.candidates().begin(), victims.candidates().end(),
                                       [](const auto& set) { return set.mask == 3; });
    failures += check(feasible != victims.candidates().end() && feasible->count == 2,
                      "multi-victim pressure candidate was omitted");
    failures += check(std::all_of(victims.candidates().begin(), victims.candidates().end(),
                                  [](const auto& set) { return (set.mask & 8U) == 0; }),
                      "completion member became a suspension victim");

    rows[0].useful_chunk_since_resume = false;
    victims                           = pressure.ranked_victims(rows, KvPressureReason::Growth);
    failures += check(std::all_of(victims.candidates().begin(), victims.candidates().end(),
                                  [](const auto& set) { return (set.mask & 1U) == 0; }),
                      "recent resume was suspended before useful progress");
    pressure.observe(rows, false);
    victims = pressure.ranked_victims(rows, KvPressureReason::Growth);
    failures += check(std::any_of(victims.candidates().begin(), victims.candidates().end(),
                                  [](const auto& set) { return set.mask == 1; }),
                      "no-runnable emergency could not evacuate a recent resume");
    rows[1].suspend_ready = false;
    victims               = pressure.ranked_victims(rows, KvPressureReason::Growth);
    failures += check(std::all_of(victims.candidates().begin(), victims.candidates().end(),
                                  [](const auto& set) { return (set.mask & 2U) == 0; }),
                      "unfunded or unsettled request became an evacuation victim");

    // Suspended identities retain original arrival precedence and consume logical admission
    // capacity although they no longer occupy a physical lane.
    std::array waiting{request(10), request(2), request(6)};
    waiting[0].resident = false;
    waiting[1].resident = false;
    failures += check(KvPressurePolicy::oldest_suspended(waiting) == 2,
                      "resume order lost original request age");
    KvPressurePolicy bounded(2, 5);
    bounded.observe(waiting, true);
    failures += check(!bounded.allows_ingress(3, true) && !bounded.allows_ingress(5, false),
                      "new ingress bypassed suspended work or logical admission bound");
    waiting[1].resident = true;
    bounded.observe(waiting, true);
    failures +=
        check(bounded.oldest_incumbent() == 2 && KvPressurePolicy::oldest_suspended(waiting) == 10,
              "physical resume reset the incumbent's logical identity");

    KvPressurePolicy alone(1, 2);
    std::array single{request(1, true)};
    single[0].completion_member = true;
    alone.observe(single, false);
    failures += check(alone.draining() && !alone.allows_ingress(1, false) &&
                          alone.ranked_victims(single, KvPressureReason::Growth).count == 0,
                      "single protected member tried to spill instead of using exclusive growth");

    return failures == 0 ? 0 : 1;
}
