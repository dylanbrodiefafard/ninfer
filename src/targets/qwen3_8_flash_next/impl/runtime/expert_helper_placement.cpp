#include "runtime/expert_helper_placement.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace ninfer::targets::qwen3_8_flash_next {
namespace {
constexpr std::size_t kEntries = 48 * 512;

std::size_t record_bytes(ops::MoeExpertCodec codec) {
    return static_cast<std::size_t>(ops::moe_expert_record_bytes(codec));
}
} // namespace

bool ExpertHelperPlacement::helper_owns(std::size_t index) const {
    if (index >= kEntries || local_eligibility.size() != kEntries / 32) {
        throw std::out_of_range("helper ownership index");
    }
    return (local_eligibility[index / 32] & (std::uint32_t{1} << (index % 32))) == 0;
}

ExpertHelperPlacement plan_expert_helper(std::span<const ops::MoeExpertCodec> codecs,
                                         std::span<const std::uint64_t> counts,
                                         std::array<std::int32_t, 2> fixed,
                                         std::size_t free_bytes) {
    if (codecs.size() != kEntries || counts.size() != kEntries || fixed[0] < 0 || fixed[1] < 0 ||
        free_bytes < kHelperReserveBytes) {
        throw std::invalid_argument("expert helper placement requires [48,512], nonnegative fixed "
                                    "capacity and 1 GiB reserve");
    }
    ExpertHelperPlacement result;
    result.budget_bytes = (free_bytes - kHelperReserveBytes) / 256 * 256;
    result.local_eligibility.assign(kEntries / 32, ~std::uint32_t{0});
    std::array<std::vector<std::size_t>, 2> by_codec;
    for (std::size_t i = 0; i < kEntries; ++i) {
        const auto tag = static_cast<std::size_t>(codecs[i]);
        if (tag >= 2) { throw std::invalid_argument("expert helper placement: unsupported codec"); }
        by_codec[tag].push_back(i);
        if (counts[i] > std::numeric_limits<std::uint64_t>::max() - result.total_routes) {
            throw std::overflow_error("expert helper CALIB sum overflow");
        }
        result.total_routes += counts[i];
    }
    std::vector<std::size_t> complement;
    for (std::size_t tag = 0; tag < 2; ++tag) {
        auto& order = by_codec[tag];
        if (static_cast<std::size_t>(fixed[tag]) > order.size()) {
            throw std::invalid_argument("expert helper fixed capacity exceeds codec count");
        }
        std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
            return counts[a] != counts[b] ? counts[a] > counts[b] : a < b;
        });
        complement.insert(complement.end(), order.begin() + fixed[tag], order.end());
        result.remaining_counts[tag] = static_cast<std::int32_t>(order.size());
    }
    std::sort(complement.begin(), complement.end(), [&](std::size_t a, std::size_t b) {
        const auto lhs = static_cast<unsigned __int128>(counts[a]) * record_bytes(codecs[b]);
        const auto rhs = static_cast<unsigned __int128>(counts[b]) * record_bytes(codecs[a]);
        return lhs != rhs ? lhs > rhs : a < b;
    });
    std::size_t left = result.budget_bytes;
    for (const auto i : complement) {
        const auto bytes = record_bytes(codecs[i]);
        if (bytes > left) { continue; }
        left -= bytes;
        const auto tag = static_cast<std::size_t>(codecs[i]);
        result.local_eligibility[i / 32] &= ~(std::uint32_t{1} << (i % 32));
        ++result.helper_counts[tag];
        --result.remaining_counts[tag];
        result.helper_bytes[tag] += bytes;
        result.layer_bytes[i / 512] += bytes;
        ++result.layer_counts[i / 512][tag];
        result.helper_routes += counts[i];
    }
    result.full_coverage = result.remaining_counts == fixed;
    return result;
}

std::uint64_t expert_record_checksum(std::span<const std::byte> bytes) {
    if (bytes.empty() || bytes.size() % 8 != 0) {
        throw std::invalid_argument("record checksum needs a nonempty U64 span");
    }
    std::uint64_t sum = 0;
    for (std::size_t i = 0; i < bytes.size() / 8; ++i) {
        std::uint64_t value = 0;
        std::memcpy(&value, bytes.data() + i * 8, 8);
        value ^= static_cast<std::uint64_t>(i);
        value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
        value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
        sum += value ^ (value >> 31);
    }
    return sum;
}
} // namespace ninfer::targets::qwen3_8_flash_next
