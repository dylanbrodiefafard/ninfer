#pragma once

#include "ninfer/types.h"

#include <filesystem>

namespace ninfer::product {

enum class SpeculativeSelection { ArtifactDefault, Explicit };

// Resolves CLI/server defaults from the immutable artifact directory without reading weights
// or allocating device memory. Explicit selections pass through unchanged. Reader errors propagate.
[[nodiscard]] SpeculativeOptions
resolve_speculative_startup(const std::filesystem::path& artifact_path,
                            SpeculativeSelection selection, const SpeculativeOptions& requested);

} // namespace ninfer::product
