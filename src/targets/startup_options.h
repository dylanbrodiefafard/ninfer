#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace ninfer::targets {
// Resolves Flash-Next's RAM-on default of 30 seconds, otherwise 0. Explicit nonzero values
// require that identity and a RAM tier; throws std::invalid_argument without device work.
[[nodiscard]] std::uint32_t resolve_kv_idle_timeout(std::optional<std::uint32_t> requested,
                                                    std::size_t ram_bytes,
                                                    std::string_view model_id);

// Checks lexical PCI and build capability before Engine creates any device context.
void validate_expert_helper_build(const std::optional<std::string>& pci);

// Host-only registry admission, shared with product preflight and the Flash-Next parity tool.
// An absent option is a no-op. Rejects malformed PCI, wrong identity and unavailable build
// capability without initializing either GPU runtime. Throws std::invalid_argument.
void validate_expert_helper_option(const std::optional<std::string>& pci,
                                   std::string_view model_id);
} // namespace ninfer::targets
