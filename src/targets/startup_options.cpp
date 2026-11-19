#include "targets/startup_options.h"

#include "core/expert_helper_pci.h"

namespace ninfer::targets {
std::uint32_t resolve_kv_idle_timeout(std::optional<std::uint32_t> requested, std::size_t ram_bytes,
                                      std::string_view model_id) {
    const bool flash_next = model_id == "qwen3.8-flash-next";
    if (requested.value_or(0) != 0) {
        if (!flash_next) {
            throw std::invalid_argument(
                "--kv-idle-timeout applies only to qwen3.8-flash-next, not '" +
                std::string(model_id) + "'");
        }
        if (ram_bytes == 0) {
            throw std::invalid_argument("--kv-idle-timeout requires --kv-ram-capacity");
        }
    }
    return requested.value_or(flash_next && ram_bytes != 0 ? 30U : 0U);
}

std::int64_t resolve_kv_idle_headroom(std::optional<std::uint32_t> headroom,
                                      std::optional<std::uint32_t> timeout, std::size_t ram_bytes,
                                      std::string_view model_id) {
    const bool enabled = resolve_kv_idle_timeout(timeout, ram_bytes, model_id) != 0;
    if (headroom && model_id != "qwen3.8-flash-next") {
        throw std::invalid_argument("--kv-idle-headroom applies only to qwen3.8-flash-next, not '" +
                                    std::string(model_id) + "'");
    }
    if (headroom && !enabled) {
        throw std::invalid_argument("--kv-idle-headroom requires an enabled --kv-idle-timeout");
    }
    return headroom.value_or(kDefaultKvIdleHeadroomTokens);
}

void validate_expert_helper_build(const std::optional<std::string>& pci) {
    if (!pci) { return; }
    static_cast<void>(normalize_helper_pci(*pci));
#ifndef NINFER_ENABLE_HIP_EXPERT_HELPER
    throw std::invalid_argument(
        "expert helper requires a build with NINFER_ENABLE_HIP_EXPERT_HELPER=ON");
#endif
}

void validate_expert_helper_option(const std::optional<std::string>& pci,
                                   std::string_view model_id) {
    if (!pci) { return; }
    if (model_id != "qwen3.8-flash-next") {
        throw std::invalid_argument(
            "--expert-helper-pci applies only to qwen3.8-flash-next, not '" +
            std::string(model_id) + "'");
    }
    validate_expert_helper_build(pci);
}
} // namespace ninfer::targets
