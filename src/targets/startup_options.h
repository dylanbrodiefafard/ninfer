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

inline constexpr std::int64_t kDefaultKvIdleHeadroomTokens = 200000;

// Resolves the KV tokens of lendable extents beyond the expert cache's flex demand below which
// idle expiry continues (default 200000; 0 evicts only to cover the demand). An explicit value
// requires the identity and an effective idle timeout (resolve_kv_idle_timeout of the same
// arguments is nonzero); throws std::invalid_argument without device work.
[[nodiscard]] std::int64_t resolve_kv_idle_headroom(std::optional<std::uint32_t> headroom,
                                                    std::optional<std::uint32_t> timeout,
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
