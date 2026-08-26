#pragma once

#include <cstdint>
#include <sstream>
#include <string>
#include <string_view>

namespace ninfer::product {

// Absolute head frontiers as `restored:F` / `captured:F`. `separator` is ',' on
// the serve `[req] done` line and ' ' on the CLI summary. Both zero returns `empty`.
[[nodiscard]] inline std::string format_context_checkpoint_frontiers(
    std::uint32_t restored, std::uint32_t captured, char separator,
    std::string_view empty = {}) {
    if (restored == 0 && captured == 0) { return std::string(empty); }
    std::ostringstream out;
    bool first = true;
    if (restored != 0) {
        out << "restored:" << restored;
        first = false;
    }
    if (captured != 0) {
        if (!first) { out << separator; }
        out << "captured:" << captured;
    }
    return out.str();
}

} // namespace ninfer::product
