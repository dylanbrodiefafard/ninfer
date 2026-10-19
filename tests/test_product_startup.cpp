#include "artifact_fixture.h"
#include "product/startup_options.h"

#include <exception>
#include <iostream>

int main() {
    using namespace ninfer;
    using product::SpeculativeSelection;
    using product::resolve_speculative_startup;
    using test::artifact_fixture::Json;
    using test::artifact_fixture::write_fixture;
    try {
        Json directory  = {{"identity", {{"model_id", "qwen3.8-27b"}, {"weights_id", "nvfp4"}}},
                           {"objects", Json::array({{{"name", "tokenizer"},
                                                     {"kind", "resource"},
                                                     {"encoding", "raw-bytes-v1"},
                                                     {"offset", 0},
                                                     {"bytes", 1}}})}};
        const auto base = write_fixture(directory, "startup_base");
        if (resolve_speculative_startup(base.path, SpeculativeSelection::ArtifactDefault, {})
                .backend != SpeculativeBackend::None) {
            throw std::runtime_error("base artifact enabled unrequested speculation");
        }
        directory["objects"].push_back({{"name", "dflash/feature_projection"},
                                        {"kind", "tensor"},
                                        {"shape", {1}},
                                        {"format", "BF16"},
                                        {"layout", "contiguous-le-v1"},
                                        {"offset", 256},
                                        {"bytes", 2}});
        const auto combined = write_fixture(directory, "startup_combined");
        const auto automatic =
            resolve_speculative_startup(combined.path, SpeculativeSelection::ArtifactDefault, {});
        if (automatic.backend != SpeculativeBackend::DFlash || automatic.draft_tokens != 7 ||
            automatic.proposal_head != ProposalHead::Optimized || !automatic.adaptive_draft) {
            throw std::runtime_error("DFlash artifact did not select production startup defaults");
        }
        if (resolve_speculative_startup(combined.path, SpeculativeSelection::Explicit, {})
                .backend != SpeculativeBackend::None) {
            throw std::runtime_error("explicit none was overridden by the artifact");
        }
        SpeculativeOptions explicit_settings;
        explicit_settings.backend      = SpeculativeBackend::DFlash;
        explicit_settings.draft_tokens = 3;
        const auto explicit_dflash     = resolve_speculative_startup(
            combined.path, SpeculativeSelection::Explicit, explicit_settings);
        if (explicit_dflash.backend != SpeculativeBackend::DFlash ||
            explicit_dflash.draft_tokens != 3 || explicit_dflash.adaptive_draft ||
            explicit_dflash.proposal_head != ProposalHead::Full) {
            throw std::runtime_error("explicit DFlash settings were overridden by the artifact");
        }
        std::cout << "ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
