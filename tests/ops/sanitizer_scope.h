#pragma once

// Sanitizer scope for Op tests.
//
// A test started with `--sanitizer` runs its sanitizer cases instead of its full oracle sweep:
// each kernel route the qwen3.8-27b NVFP4 DFlash2 flow dispatches, once, at its boundary shapes.
// compute-sanitizer cost grows with launches and tracked memory, not with the number of oracle
// comparisons, so these cases keep real model widths and cut the case count. Tests registered
// under the CTest label `sanitizer` are what the compute-sanitizer gates run; see
// docs/maintainer/code-quality.md.

#include <string_view>

namespace ninfer::test {

[[nodiscard]] inline bool sanitizer_scope(int argc, char** argv) {
    return argc == 2 && std::string_view(argv[1]) == "--sanitizer";
}

} // namespace ninfer::test
