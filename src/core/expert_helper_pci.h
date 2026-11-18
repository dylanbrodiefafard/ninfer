#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer {
// Strict domain:bus:device.function, normalized to lowercase. No device/runtime access.
[[nodiscard]] inline std::string normalize_helper_pci(std::string_view pci) {
    if (pci.size() != 12 || pci[4] != ':' || pci[7] != ':' || pci[10] != '.') {
        throw std::invalid_argument("--expert-helper-pci must be domain:bus:device.function");
    }
    std::string result(pci);
    for (std::size_t i = 0; i < result.size(); ++i) {
        if (i == 4 || i == 7 || i == 10) { continue; }
        char& c = result[i];
        if (c >= 'A' && c <= 'F') { c = static_cast<char>(c - 'A' + 'a'); }
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'))) {
            throw std::invalid_argument("--expert-helper-pci contains a non-hex digit");
        }
    }
    if (result[8] > '1' || result[11] > '7') {
        throw std::invalid_argument("--expert-helper-pci device/function is out of range");
    }
    return result;
}
} // namespace ninfer
