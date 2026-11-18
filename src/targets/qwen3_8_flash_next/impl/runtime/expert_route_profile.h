#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <vector>

namespace ninfer::targets::qwen3_8_flash_next {

// Owning lifetime decode/verify routed-set totals, layer-major [48,512]. Codec order is
// NVFP4, F3G32. Prefill is excluded; rejected verification proposals are included.
struct ExpertRouteProfile {
    std::vector<std::uint64_t> decode_lookups;
    std::array<std::uint64_t, 2> lookups{};
    std::array<std::uint64_t, 2> misses{};
    std::uint64_t miss_bytes = 0;
    std::uint64_t rounds     = 0;
};

// A single Program owns path. Closes a same-directory temporary file before atomic rename.
// Invalid shape or I/O failure throws; the prior snapshot remains intact on write failure.
void write_expert_route_profile(const std::filesystem::path& path,
                                const ExpertRouteProfile& profile, double uptime_s);

} // namespace ninfer::targets::qwen3_8_flash_next
