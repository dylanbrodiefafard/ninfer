#pragma once

#include "ninfer/types.h"

#include <filesystem>

namespace ninfer::product {

enum class SpeculativeSelection { ArtifactDefault, Explicit };

// Resolves CLI/server defaults from the immutable artifact directory without reading weights
// or allocating device memory. Also validates the optional helper against artifact identity and
// build capability. Explicit speculation passes through unchanged. Reader/validation errors
// propagate.
[[nodiscard]] SpeculativeOptions
resolve_speculative_startup(const std::filesystem::path& artifact_path,
                            SpeculativeSelection selection, const SpeculativeOptions& requested,
                            const std::optional<std::string>& expert_helper_pci = std::nullopt);

} // namespace ninfer::product
