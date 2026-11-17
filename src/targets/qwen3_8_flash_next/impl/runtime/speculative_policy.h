#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace ninfer::targets::qwen3_8_flash_next {

// RTX 5090, NVFP4 KV, 4096-token prompts and 512 greedy outputs. Each width has its own
// conditional acceptance calibration: the symmetric drafter changes proposals with width.
// Costs are round latency / K0 round latency, not seconds per published token.
struct SpeculativeBatchPolicy {
    std::int32_t k;
    std::array<std::array<double, 3>, 3> acceptance;
    std::array<double, 3> root_cost;
};

// batch is the checked compact decode membership in [1,12].
[[nodiscard]] inline const SpeculativeBatchPolicy& speculative_batch_policy(std::size_t batch) {
    static constexpr std::array<SpeculativeBatchPolicy, 8> table{{
        {3, {{{0.777, 0, 0}, {0.688, 0.755, 0}, {0.663, 0.765, 0.782}}}, {1.42, 1.73, 1.78}},
        {3, {{{0.762, 0, 0}, {0.733, 0.739, 0}, {0.707, 0.755, 0.762}}}, {1.43, 1.81, 2.03}},
        {2, {{{0.793, 0, 0}, {0.782, 0.773, 0}, {0.773, 0.751, 0.746}}}, {1.56, 2.00, 2.61}},
        {2, {{{0.835, 0, 0}, {0.844, 0.809, 0}, {0.807, 0.795, 0.805}}}, {1.65, 2.19, 2.59}},
        {2, {{{0.823, 0, 0}, {0.831, 0.770, 0}, {0.778, 0.782, 0.790}}}, {2.32, 2.23, 3.07}},
        {1, {{{0.848, 0, 0}, {0.820, 0.820, 0}, {0.802, 0.847, 0.856}}}, {1.78, 2.43, 3.07}},
        {1, {{{0.853, 0, 0}, {0.830, 0.820, 0}, {0.778, 0.797, 0.808}}}, {1.71, 2.35, 3.09}},
        {1, {{{0.853, 0, 0}, {0.830, 0.820, 0}, {0.778, 0.797, 0.808}}}, {1.71, 2.35, 3.09}},
    }};
    // C7 and C9..12 use the measured high-concurrency C8 policy; they are not measured optima.
    return table.at(std::min(batch, table.size()) - 1);
}

// Request-owned continuation metadata. Only complete, published accept walks are observations;
// rejected and truncated packets must not train the policy. No other request's timing or
// acceptance participates. The ceiling only decreases, including after a batch-size change.
struct SpeculativeRequestPolicy {
    std::array<std::uint32_t, 3> reached{}, accepted{}, low_windows{};
    std::int32_t width   = 0;
    std::int32_t ceiling = 3;

    // k is the executed proposal width minus one, in [1,3]; extent is in [1,k] and
    // accepted_tokens is in [0,extent]. The Program admits only fully published walks.
    void observe(std::size_t batch, std::int32_t k, std::int32_t extent,
                 std::int32_t accepted_tokens) {
        if (k != width) {
            // Do not mix evidence from different symmetric proposal widths.
            reached     = {};
            accepted    = {};
            low_windows = {};
            width       = k;
        }
        const auto& reference = speculative_batch_policy(batch);
        for (std::int32_t hop = 0; hop < extent && hop <= accepted_tokens; ++hop) {
            const auto index = static_cast<std::size_t>(hop);
            ++reached[index];
            accepted[index] += accepted_tokens > hop ? 1U : 0U;
            if (reached[index] != 32) { continue; }
            const double rate = static_cast<double>(accepted[index]) / 32;
            const auto arm    = static_cast<std::size_t>(k - 1);
            // Two disjoint low windows avoid reacting to an unlucky accept walk. A modest
            // acceptance change is not a reason to abandon the measured best fixed arm.
            low_windows[index] =
                rate < 0.55 * reference.acceptance[arm][index] ? low_windows[index] + 1 : 0;
            reached[index]  = 0;
            accepted[index] = 0;
            if (low_windows[index] < 2) { continue; }
            auto limit = std::max(1, hop);
            // Even perfect later hops cannot repay this width's measured round cost. The
            // first-hop collapse therefore warrants K0 rather than another drafting arm.
            if (hop == 0 && 1 + k * rate < reference.root_cost[arm]) { limit = 0; }
            ceiling            = std::min(ceiling, limit);
            low_windows[index] = 0;
        }
    }
};

} // namespace ninfer::targets::qwen3_8_flash_next
