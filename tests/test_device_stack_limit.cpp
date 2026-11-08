#include "core/device.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <vector>

// Every compiled kernel must fit the per-thread stack DeviceContext sets. A larger frame makes
// its first launch grow the driver's local-memory reservation outside the Automatic KV
// capacity plan. Kernel optimization may change stack sizes; a failure does not mean the new
// frame is wrong, only that its memory must be accounted for: reduce the frame, or raise
// kDeviceStackBytesPerThread and re-measure the published KV capacity tables. The libraries are
// the static archives that hold all product device code; bench, tool and test kernels are not
// linked into the product.
namespace {

struct FunctionUsage {
    std::string name;
    std::size_t stack_bytes = 0;
};

std::size_t field(const std::string& line, const std::string& key) {
    const auto at = line.find(" " + key + ":");
    if (at == std::string::npos) { return 0; }
    return std::stoull(line.substr(at + key.size() + 2));
}

// Owns a popen stream; close() reports the child's exit status.
class Pipe {
public:
    // Runs the CMake-configured cuobjdump on build-tree archive paths; no external input.
    // NOLINTNEXTLINE(bugprone-command-processor)
    explicit Pipe(const std::string& command) : file_(popen(command.c_str(), "r")) {
        if (file_ == nullptr) { throw std::runtime_error("cannot run " + command); }
    }

    Pipe(const Pipe&)            = delete;
    Pipe& operator=(const Pipe&) = delete;

    ~Pipe() {
        if (file_ != nullptr) { pclose(file_); }
    }

    [[nodiscard]] FILE* get() const noexcept { return file_; }

    [[nodiscard]] bool close_succeeded() {
        const int status = pclose(file_);
        file_            = nullptr;
        return status != -1 && WIFEXITED(status) && WEXITSTATUS(status) == 0;
    }

private:
    FILE* file_;
};

std::vector<FunctionUsage> resource_usage(const std::string& library) {
    const std::string command =
        std::string("\"") + NINFER_CUOBJDUMP + "\" --dump-resource-usage \"" + library + "\"";
    Pipe pipe(command);
    std::vector<FunctionUsage> functions;
    std::string current;
    std::array<char, 8192> buffer{};
    std::string line;
    while (std::fgets(buffer.data(), static_cast<int>(buffer.size()), pipe.get()) != nullptr) {
        line += buffer.data();
        if (line.empty() || line.back() != '\n') { continue; }
        const auto begin = line.find_first_not_of(' ');
        if (begin != std::string::npos && line.compare(begin, 9, "Function ") == 0) {
            current = line.substr(begin + 9, line.find_last_not_of(":\n") - begin - 8);
        } else if (!current.empty() && line.find(" STACK:") != std::string::npos) {
            // LOCAL is non-ABI local memory; both draw on the same per-thread reservation.
            functions.push_back({current, field(line, "STACK") + field(line, "LOCAL")});
            current.clear();
        }
        line.clear();
    }
    if (!pipe.close_succeeded()) { throw std::runtime_error("cuobjdump failed for " + library); }
    return functions;
}

} // namespace

int main() try {
    std::vector<std::string> libraries;
    std::istringstream list(NINFER_KERNEL_LIBRARIES);
    for (std::string library; std::getline(list, library, '|');) {
        if (!library.empty()) { libraries.push_back(library); }
    }
    std::size_t functions = 0;
    std::size_t largest   = 0;
    int failures          = 0;
    for (const auto& library : libraries) {
        const auto usage = resource_usage(library);
        if (usage.empty()) {
            std::cerr << "FAIL no device functions found in " << library << '\n';
            ++failures;
        }
        for (const auto& function : usage) {
            ++functions;
            largest = std::max(largest, function.stack_bytes);
            if (function.stack_bytes > ninfer::kDeviceStackBytesPerThread) {
                std::cerr << "FAIL " << function.stack_bytes << " B stack exceeds "
                          << ninfer::kDeviceStackBytesPerThread << " B: " << function.name << '\n';
                ++failures;
            }
        }
    }
    std::cout << (failures ? "FAIL" : "OK") << " device stack limit: " << functions
              << " functions, largest " << largest << " B, limit "
              << ninfer::kDeviceStackBytesPerThread << " B\n";
    if (failures) {
        std::cerr << "A kernel stack frame grew past the device stack reservation. Either reduce "
                     "the frame (usually register spills) or raise kDeviceStackBytesPerThread in "
                     "src/core/device.h; each byte costs about 0.25 MiB of device memory on the "
                     "RTX 5090, taken from Automatic KV capacity, so re-measure the published "
                     "capacity tables.\n";
    } else if (largest < ninfer::kDeviceStackBytesPerThread) {
        std::cout << "kDeviceStackBytesPerThread can be lowered to " << largest
                  << " B to return the unused reservation to KV capacity.\n";
    }
    return failures ? 1 : 0;
} catch (const std::exception& error) {

    std::cerr << "FAIL " << error.what() << '\n';
    return 1;
}
