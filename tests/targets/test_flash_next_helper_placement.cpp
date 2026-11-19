#include "runtime/expert_helper_placement.h"
#include "runtime/expert_helper_profile.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <limits>
#include <numeric>
#include <set>
#include <stdexcept>
#include <vector>

using namespace ninfer::targets::qwen3_8_flash_next;
using ninfer::ops::MoeExpertCodec;

namespace {
// Independent rational comparison by continued fractions: no cross product, floating point or
// production comparator. True iff a/b > c/d, including 64-bit numerators near the range limit.
bool ratio_greater(std::uint64_t a, std::uint64_t b, std::uint64_t c, std::uint64_t d) {
    bool reverse = false;
    while (true) {
        if (a / b != c / d) { return reverse ? a / b < c / d : a / b > c / d; }
        a %= b;
        c %= d;
        if (a == 0 || c == 0) { return reverse ? (a == 0 && c != 0) : (a != 0 && c == 0); }
        std::swap(a, b);
        std::swap(c, d);
        reverse = !reverse;
    }
}

void require(bool condition) {
    if (!condition) { throw std::runtime_error("placement independent model mismatch"); }
}

void check(std::span<const MoeExpertCodec> codecs, std::span<const std::uint64_t> counts,
           std::array<std::int32_t, 2> fixed, std::size_t budget, bool expect_full) {
    const auto actual = plan_expert_helper(codecs, counts, fixed, budget + kHelperReserveBytes);
    std::vector<bool> protected_core(counts.size(), false), helper(counts.size(), false);
    for (std::size_t codec = 0; codec < 2; ++codec) {
        std::set<std::pair<std::uint64_t, std::size_t>> ranked;
        for (std::size_t i = 0; i < counts.size(); ++i) {
            if (static_cast<std::size_t>(codecs[i]) == codec) {
                ranked.emplace(counts[i], counts.size() - i);
            }
        }
        for (std::int32_t slot = 0; slot < fixed[codec]; ++slot) {
            const auto it                              = std::prev(ranked.end());
            protected_core[counts.size() - it->second] = true;
            ranked.erase(it);
        }
    }
    const auto bytes = [&](std::size_t i) {
        return static_cast<std::uint64_t>(codecs[i] == MoeExpertCodec::F3G32 ? 1996800 : 2764800);
    };
    const auto ranking = [&](std::size_t a, std::size_t b) {
        if (ratio_greater(counts[a], bytes(a), counts[b], bytes(b))) { return true; }
        if (ratio_greater(counts[b], bytes(b), counts[a], bytes(a))) { return false; }
        return a < b;
    };
    std::set<std::size_t, decltype(ranking)> candidates(ranking);
    for (std::size_t i = 0; i < counts.size(); ++i) {
        if (!protected_core[i]) { candidates.insert(i); }
    }
    std::size_t available = budget / 256 * 256, used = 0;
    std::array<std::int32_t, 2> helper_counts{}, local_counts{};
    std::array<std::size_t, 2> helper_bytes{};
    std::array<std::size_t, 48> layer_bytes{};
    std::array<std::array<std::int32_t, 2>, 48> layer_counts{};
    std::array<std::int32_t, 2> primary_counts{}, tail_counts{};
    std::array<std::size_t, 2> primary_bytes{}, tail_bytes{};
    std::uint64_t helper_routes = 0, total_routes = 0, primary_routes = 0, tail_routes = 0;
    for (const auto i : candidates) {
        if (bytes(i) <= available) {
            helper[i] = true;
            available -= bytes(i);
        }
    }
    for (std::size_t i = 0; i < counts.size(); ++i) {
        require(actual.helper_owns(i) == helper[i]);
        require(!helper[i] || !protected_core[i]);
        const auto tag = static_cast<std::size_t>(codecs[i]);
        total_routes += counts[i];
        if (helper[i]) {
            used += bytes(i);
            helper_bytes[tag] += bytes(i);
            layer_bytes[i / 512] += bytes(i);
            ++helper_counts[tag];
            ++layer_counts[i / 512][tag];
            helper_routes += counts[i];
        } else {
            ++local_counts[tag];
        }
        require((actual.tier(i) == ExpertPlacementTier::Primary) == protected_core[i]);
        require((actual.tier(i) == ExpertPlacementTier::Helper) == helper[i]);
        if (protected_core[i]) {
            ++primary_counts[tag];
            primary_bytes[tag] += bytes(i);
            primary_routes += counts[i];
        } else if (!helper[i]) {
            ++tail_counts[tag];
            tail_bytes[tag] += bytes(i);
            tail_routes += counts[i];
        }
    }
    // Admission covers exactly the F3 records of T; every P record owns the next fixed slot of its
    // codec in ascending index order, and no other record has a slot.
    std::array<std::int32_t, 2> next_slot{};
    std::int32_t admitted = 0;
    require(actual.admission_mask.size() == counts.size() / 32 &&
            actual.primary_slot.size() == counts.size());
    for (std::size_t i = 0; i < counts.size(); ++i) {
        const bool tail_f3 = !protected_core[i] && !helper[i] && codecs[i] == MoeExpertCodec::F3G32;
        require(((actual.admission_mask[i / 32] >> (i % 32)) & 1U) == (tail_f3 ? 1U : 0U));
        admitted += tail_f3 ? 1 : 0;
        require(actual.primary_slot[i] ==
                (protected_core[i] ? next_slot[static_cast<std::size_t>(codecs[i])]++ : -1));
    }
    require(actual.admission_count == admitted);
    require(actual.primary_counts == primary_counts && actual.tail_counts == tail_counts);
    require(actual.primary_bytes == primary_bytes && actual.tail_bytes == tail_bytes);
    require(actual.primary_routes == primary_routes && actual.tail_routes == tail_routes);
    require(primary_routes + helper_routes + tail_routes == total_routes);
    for (std::size_t tag = 0; tag < 2; ++tag) {
        require(primary_counts[tag] == fixed[tag]);
        require(primary_counts[tag] + helper_counts[tag] + tail_counts[tag] ==
                static_cast<std::int32_t>(std::ranges::count_if(
                    codecs, [&](MoeExpertCodec c) { return static_cast<std::size_t>(c) == tag; })));
    }
    require(actual.helper_counts == helper_counts && actual.remaining_counts == local_counts);
    require(actual.helper_bytes == helper_bytes && actual.layer_bytes == layer_bytes &&
            actual.layer_counts == layer_counts);
    require(actual.helper_routes == helper_routes && actual.total_routes == total_routes);
    require(actual.full_coverage == expect_full &&
            (tail_counts == std::array<std::int32_t, 2>{}) == expect_full && used <= budget);
    std::printf("placement budget=%zu fixed=%d/%d helper=%d/%d bytes=%zu full=%d exact=ok\n",
                budget, fixed[0], fixed[1], helper_counts[0], helper_counts[1], used, expect_full);
}

// Independent restatement of the documented slot split: 11% of the budget to NVFP4 slots, F3 slots
// take the rest, NVFP4 slots grow into what F3 cannot use; no pool exceeds its codec count.
std::array<std::int32_t, 2> fixed_slots(std::uint64_t budget) {
    const std::uint64_t nvfp4_record = 2764800, f3_record = 1996800, nvfp4_total = 1583,
                        f3_total = 22993;
    auto fit = [](std::uint64_t bytes, std::uint64_t record, std::uint64_t total) {
        return std::min(bytes / record, total);
    };
    const auto nvfp4 = fit(budget / 100 * 11, nvfp4_record, nvfp4_total);
    const auto f3    = fit(budget - nvfp4 * nvfp4_record, f3_record, f3_total);
    return {static_cast<std::int32_t>(fit(budget - f3 * f3_record, nvfp4_record, nvfp4_total)),
            static_cast<std::int32_t>(f3)};
}
} // namespace

int main() {
    try {
        std::vector<MoeExpertCodec> codecs(48 * 512, MoeExpertCodec::F3G32);
        // A codec inventory with the production counts but deliberately not the production map;
        // placement must use the actual map, not infer it from CALIB rank.
        for (std::size_t i = 0; i < 1583; ++i) {
            codecs[(i * 13) % codecs.size()] = MoeExpertCodec::Nvfp4G16;
        }
        check(codecs, kExpertHelperCalibCounts, {700, 8000}, std::size_t{31} << 30, true);
        check(codecs, kExpertHelperCalibCounts, {500, 6000}, std::size_t{31} << 30, false);
        check(codecs, kExpertHelperCalibCounts, {700, 8000}, 0, false);
        check(codecs, kExpertHelperCalibCounts, {0, 0}, 2764800 - 256,
              false); // skip a larger record and keep looking
        check(codecs, kExpertHelperCalibCounts, {1583, 22993}, 0,
              true); // no flex, full fixed coverage
        // Small KV: the fixed pools hold at least 17.0 GB, so P plus a 31 GiB H leaves T empty.
        // 1M-token KV with 8 lanes: a fixed budget near 12 GB leaves a nonempty T.
        for (const std::uint64_t budget :
             {std::uint64_t{24'600'000'000}, std::uint64_t{19'950'000'000},
              std::uint64_t{17'100'000'000}}) {
            check(codecs, kExpertHelperCalibCounts, fixed_slots(budget), std::size_t{31} << 30,
                  true);
        }
        for (const std::uint64_t budget :
             {std::uint64_t{12'000'000'000}, std::uint64_t{15'000'000'000}}) {
            check(codecs, kExpertHelperCalibCounts, fixed_slots(budget), std::size_t{31} << 30,
                  false);
        }
        std::vector<std::uint64_t> zeros(codecs.size(), 0);
        // H budget edge: exactly every record fits; 256 bytes less leaves a nonempty T.
        constexpr std::size_t kAll = 1583 * std::size_t{2764800} + 22993 * std::size_t{1996800};
        check(codecs, zeros, {0, 0}, kAll, true);
        check(codecs, zeros, {0, 0}, kAll - 256, false);
        check(codecs, zeros, {1, 1}, std::size_t{60} << 30, true); // zeros included, stable ties
        zeros[0] = 100;
        zeros[1] = 1;
        check(codecs, zeros, {0, 0}, 2764800 - 256, false); // highest ratio NVFP4 cannot fit
        zeros[1] = 0;
        zeros[0] = std::numeric_limits<std::uint64_t>::max();
        check(codecs, zeros, {0, 0}, 2764800, false); // cross products exceed U64
        bool invalid = false;
        try {
            static_cast<void>(plan_expert_helper(codecs, zeros, {0, 0}, kHelperReserveBytes - 1));
        } catch (const std::invalid_argument&) { invalid = true; }
        require(invalid);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
