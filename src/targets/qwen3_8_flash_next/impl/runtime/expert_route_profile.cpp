#include "runtime/expert_route_profile.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace ninfer::targets::qwen3_8_flash_next {

void write_expert_route_profile(const std::filesystem::path& path,
                                const ExpertRouteProfile& profile, double uptime_s) {
    constexpr std::size_t kLayers  = 48;
    constexpr std::size_t kExperts = 512;
    if (profile.decode_lookups.size() != kLayers * kExperts) {
        throw std::invalid_argument("expert route profile requires [48,512] counts");
    }
    const double lookups =
        static_cast<double>(profile.lookups[0]) + static_cast<double>(profile.lookups[1]);
    const double misses =
        static_cast<double>(profile.misses[0]) + static_cast<double>(profile.misses[1]);
    const double hit_rate     = lookups == 0 ? 0 : 1 - misses / lookups;
    const double mb_per_round = profile.rounds == 0 ? 0
                                                    : static_cast<double>(profile.miss_bytes) /
                                                          static_cast<double>(profile.rounds) / 1e6;
    const std::filesystem::path temporary = path.string() + ".tmp";
    try {
        std::ofstream out;
        out.exceptions(std::ios::failbit | std::ios::badbit);
        out.open(temporary, std::ios::out | std::ios::trunc);
        out << std::setprecision(17)
            << "{\"version\":1,\"identity\":{\"model_id\":\"qwen3.8-flash-next\","
               "\"weights_id\":\"f3-fp8-nvfp4\"},\"uptime_s\":"
            << uptime_s << ",\"rounds\":" << profile.rounds
            << ",\"codecs\":{\"nvfp4\":{\"decode_lookups\":" << profile.lookups[0]
            << ",\"decode_misses\":" << profile.misses[0]
            << "},\"f3g32\":{\"decode_lookups\":" << profile.lookups[1]
            << ",\"decode_misses\":" << profile.misses[1] << "}},\"hit_rate\":" << hit_rate
            << ",\"miss_bytes\":" << profile.miss_bytes << ",\"decode_lookups\":[";
        for (std::size_t layer = 0; layer < kLayers; ++layer) {
            if (layer != 0) { out << ','; }
            out << '[';
            for (std::size_t expert = 0; expert < kExperts; ++expert) {
                if (expert != 0) { out << ','; }
                out << profile.decode_lookups[layer * kExperts + expert];
            }
            out << ']';
        }
        out << "]}\n";
        out.close();
        std::filesystem::rename(temporary, path);
    } catch (const std::exception& error) {
        throw std::runtime_error("expert route profile '" + path.string() + "': " + error.what());
    }
    std::clog << "expert route profile: decode hit rate=" << hit_rate * 100
              << "% miss MB/round=" << mb_per_round << " rounds=" << profile.rounds << '\n';
}

} // namespace ninfer::targets::qwen3_8_flash_next
