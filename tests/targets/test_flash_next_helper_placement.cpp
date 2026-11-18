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
    std::uint64_t helper_routes = 0, total_routes = 0;
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
    }
    require(actual.helper_counts == helper_counts && actual.remaining_counts == local_counts);
    require(actual.helper_bytes == helper_bytes && actual.layer_bytes == layer_bytes &&
            actual.layer_counts == layer_counts);
    require(actual.helper_routes == helper_routes && actual.total_routes == total_routes);
    require(actual.full_coverage == expect_full && used <= budget);
    std::printf("placement budget=%zu fixed=%d/%d helper=%d/%d bytes=%zu full=%d exact=ok\n",
                budget, fixed[0], fixed[1], helper_counts[0], helper_counts[1], used, expect_full);
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
        std::vector<std::uint64_t> zeros(codecs.size(), 0);
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
