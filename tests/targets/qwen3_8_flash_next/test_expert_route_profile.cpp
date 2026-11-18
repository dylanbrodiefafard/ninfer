#include "runtime/expert_route_profile.h"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include <unistd.h>

int main() {
    using namespace ninfer::targets::qwen3_8_flash_next;
    try {
        const auto path = std::filesystem::temp_directory_path() /
                          ("ninfer-route-profile-" + std::to_string(getpid()) + ".json");
        ExpertRouteProfile profile;
        profile.decode_lookups.resize(48 * 512);
        profile.decode_lookups[511] = 9007199254740993ULL;
        profile.lookups             = {9007199254740993ULL, 10};
        profile.misses              = {1, 2};
        profile.miss_bytes          = 6758400;
        profile.rounds              = 3;
        write_expert_route_profile(path, profile, 12.5);
        std::ifstream input(path);
        const auto json = nlohmann::json::parse(input);
        if (json.at("version") != 1 || json.at("rounds") != 3 ||
            json.at("identity").at("model_id") != "qwen3.8-flash-next" ||
            json.at("decode_lookups").size() != 48 ||
            json.at("decode_lookups").at(0).size() != 512 ||
            json.at("decode_lookups").at(0).at(511).get<std::uint64_t>() != 9007199254740993ULL ||
            json.at("miss_bytes") != 6758400 || json.at("uptime_s") != 12.5 ||
            json.at("codecs").at("f3g32").at("decode_misses") != 2) {
            throw std::runtime_error("route profile JSON lost shape, identity or exact counts");
        }
        input.close();
        profile.decode_lookups[511] = 0;
        profile.rounds              = 4;
        write_expert_route_profile(path, profile, 13);
        input.open(path);
        const auto replaced = nlohmann::json::parse(input);
        input.close();
        if (replaced.at("rounds") != 4 || replaced.at("decode_lookups").at(0).at(511) != 0 ||
            std::filesystem::exists(path.string() + ".tmp")) {
            throw std::runtime_error("route profile replacement was not complete");
        }
        bool rejected = false;
        profile.decode_lookups.clear();
        try {
            write_expert_route_profile(path, profile, 0);
        } catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected) { throw std::runtime_error("route profile accepted an invalid shape"); }
        if (!std::filesystem::remove(path)) { throw std::runtime_error("profile cleanup failed"); }
        std::cout << "OK expert route profile JSON\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
