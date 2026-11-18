#include "product/startup_options.h"

#include "artifact/reader.h"
#include "targets/startup_options.h"

namespace ninfer::product {

SpeculativeOptions
resolve_speculative_startup(const std::filesystem::path& artifact_path,
                            SpeculativeSelection selection, const SpeculativeOptions& requested,
                            const std::optional<std::string>& expert_helper_pci) {
    if (selection == SpeculativeSelection::Explicit && !expert_helper_pci) { return requested; }
    const artifact::Reader reader(artifact_path);
    targets::validate_expert_helper_option(expert_helper_pci, reader.identity().model_id);
    if (selection == SpeculativeSelection::Explicit) { return requested; }
    if (reader.find("dflash/feature_projection") == nullptr) { return requested; }
    if (reader.identity().model_id == "qwen3.8-flash-next" &&
        reader.identity().weights_id == "f3-fp8-nvfp4") {
        SpeculativeOptions resolved;
        resolved.backend        = SpeculativeBackend::DFlash;
        resolved.draft_tokens   = 3;
        resolved.adaptive_draft = true;
        return resolved;
    }
    if (reader.identity().model_id != "qwen3.8-27b" || reader.identity().weights_id != "nvfp4") {
        return requested;
    }
    SpeculativeOptions resolved;
    resolved.backend        = SpeculativeBackend::DFlash;
    resolved.draft_tokens   = 7;
    resolved.proposal_head  = ProposalHead::Optimized;
    resolved.adaptive_draft = true;
    return resolved;
}

} // namespace ninfer::product
