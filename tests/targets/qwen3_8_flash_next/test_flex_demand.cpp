#include "targets/qwen3_8_flash_next/impl/runtime/flex_demand.h"

#include <cstdint>
#include <iostream>

namespace {
namespace fn = ninfer::targets::qwen3_8_flash_next;

int failures = 0;

void expect(bool value, const char* message) {
    if (!value) {
        std::cerr << message << '\n';
        ++failures;
    }
}

} // namespace

int main() {
    // NVFP4 geometry: 244 pages = 15,616 tokens and 84 F3 slots per extent, 12 extents.
    constexpr std::int64_t kTokens = 244 * 64;
    expect(kTokens == 15616, "NVFP4 extent is 15,616 tokens");
    const auto base = fn::plan_flex_demand(10000, 8915, 84, 12, kTokens);
    expect(base.slots == 1085 && base.extents == 13 - 1 && base.tokens == 12 * kTokens,
           "ceil(1085 / 84) = 13 extents is bounded by the 12-extent region");
    const auto exact = fn::plan_flex_demand(8915 + 168, 8915, 84, 12, kTokens);
    expect(exact.slots == 168 && exact.extents == 2 && exact.tokens == 2 * kTokens,
           "a demand of whole extents needs no rounding");
    const auto round_up = fn::plan_flex_demand(8915 + 169, 8915, 84, 12, kTokens);
    expect(round_up.extents == 3 && round_up.tokens == 3 * kTokens,
           "one slot beyond an extent needs the next whole extent");
    const auto covered = fn::plan_flex_demand(8915, 8915, 84, 12, kTokens);
    expect(covered.slots == 0 && covered.extents == 0 && covered.tokens == 0,
           "records that fit the fixed slots have no flex demand");
    const auto surplus = fn::plan_flex_demand(100, 8915, 84, 12, kTokens);
    expect(surplus.slots == 0 && surplus.tokens == 0, "surplus fixed slots never go negative");
    // A helper placement narrows the admission domain: fewer admittable records, same fixed slots.
    const auto helper = fn::plan_flex_demand(8915 + 84, 8915, 84, 12, kTokens);
    expect(helper.extents == 1 && helper.tokens == kTokens, "a narrower domain shrinks demand");
    // BF16 geometry: 122 pages = 7,808 tokens and 108 slots per extent.
    const auto bf16 = fn::plan_flex_demand(1000 + 109, 1000, 108, 5, 122 * 64);
    expect(bf16.slots == 109 && bf16.extents == 2 && bf16.tokens == 2 * 7808,
           "BF16 geometry rounds to its own extent size");
    const auto none = fn::plan_flex_demand(5000, 1000, 0, 0, kTokens);
    expect(none.extents == 0 && none.tokens == 0, "a pool without a flex region has no demand");
    return failures == 0 ? 0 : 1;
}
