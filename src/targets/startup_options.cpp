#include "targets/startup_options.h"

#include "core/expert_helper_pci.h"

namespace ninfer::targets {
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
