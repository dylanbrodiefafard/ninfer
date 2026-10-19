#include "product/startup_options.h"

#include "artifact/reader.h"

namespace ninfer::product {

SpeculativeOptions resolve_speculative_startup(const std::filesystem::path& artifact_path,
                                               SpeculativeSelection selection,
                                               const SpeculativeOptions& requested) {
    if (selection == SpeculativeSelection::Explicit) { return requested; }
    const artifact::Reader reader(artifact_path);
    if (reader.identity().model_id != "qwen3.8-27b" || reader.identity().weights_id != "nvfp4" ||
        reader.find("dflash/feature_projection") == nullptr) {
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
