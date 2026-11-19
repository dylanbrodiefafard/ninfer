#include "artifact_fixture.h"
#include "product/startup_options.h"
#include "targets/startup_options.h"

#include <exception>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>

int main() {
    using namespace ninfer;
    using product::SpeculativeSelection;
    using product::resolve_speculative_startup;
    using test::artifact_fixture::Json;
    using test::artifact_fixture::write_fixture;
    try {
        using targets::resolve_kv_idle_timeout;
        if (resolve_kv_idle_timeout({}, 1024, "qwen3.8-flash-next") != 30 ||
            resolve_kv_idle_timeout({}, 0, "qwen3.8-flash-next") != 0 ||
            resolve_kv_idle_timeout({}, 1024, "qwen3.8-27b") != 0 ||
            resolve_kv_idle_timeout(0, 0, "qwen3.8-27b") != 0 ||
            resolve_kv_idle_timeout(5, 1024, "qwen3.8-flash-next") != 5) {
            throw std::runtime_error("idle timeout defaults or explicit override are incorrect");
        }
        for (const auto model : {"qwen3.8-27b", "qwen3.8-flash-next"}) {
            bool rejected = false;
            try {
                (void)resolve_kv_idle_timeout(5, 0, model);
            } catch (const std::invalid_argument& error) {
                rejected = std::string(error.what()).find("--kv-idle-timeout") != std::string::npos;
            }
            if (!rejected) {
                throw std::runtime_error("idle timeout accepted unsupported startup");
            }
        }
        bool wrong_identity = false;
        try {
            (void)resolve_kv_idle_timeout(5, 1024, "qwen3.8-27b");
        } catch (const std::invalid_argument&) { wrong_identity = true; }
        if (!wrong_identity) { throw std::runtime_error("27B accepted RAM-on idle timeout"); }

        using targets::resolve_kv_idle_headroom;
        if (resolve_kv_idle_headroom({}, {}, 1024, "qwen3.8-flash-next") != 200000 ||
            resolve_kv_idle_headroom({}, {}, 0, "qwen3.8-27b") != 200000 ||
            resolve_kv_idle_headroom(0, {}, 1024, "qwen3.8-flash-next") != 0 ||
            resolve_kv_idle_headroom(5, 7, 1024, "qwen3.8-flash-next") != 5) {
            throw std::runtime_error("idle headroom default or explicit value is incorrect");
        }
        for (const auto& startup :
             {std::tuple<const char*, std::optional<std::uint32_t>, std::size_t>{
                  "qwen3.8-flash-next", std::nullopt, 0}, // RAM off: no timeout
              {"qwen3.8-flash-next", 0U, 1024},           // timeout disabled
              {"qwen3.8-27b", std::nullopt, 0}}) {
            bool rejected = false;
            try {
                (void)resolve_kv_idle_headroom(5, std::get<1>(startup), std::get<2>(startup),
                                               std::get<0>(startup));
            } catch (const std::invalid_argument& error) {
                rejected =
                    std::string(error.what()).find("--kv-idle-headroom") != std::string::npos;
            }
            if (!rejected) {
                throw std::runtime_error("idle headroom accepted unsupported startup");
            }
        }

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
        directory["identity"] = {{"model_id", "qwen3.8-flash-next"},
                                 {"weights_id", "f3-fp8-nvfp4"}};
        const auto flash_next = write_fixture(directory, "startup_flash_next_dflash");
        const auto native =
            resolve_speculative_startup(flash_next.path, SpeculativeSelection::ArtifactDefault, {});
        if (native.backend != SpeculativeBackend::DFlash || native.draft_tokens != 3 ||
            native.proposal_head != ProposalHead::Full || !native.adaptive_draft) {
            throw std::runtime_error("Flash-Next companion did not select adaptive K3 defaults");
        }
        if (resolve_speculative_startup(flash_next.path, SpeculativeSelection::Explicit, {})
                .backend != SpeculativeBackend::None) {
            throw std::runtime_error("explicit none was overridden for Flash-Next");
        }
        directory["objects"].erase(1);
        const auto flash_next_base = write_fixture(directory, "startup_flash_next_base");
        if (resolve_speculative_startup(flash_next_base.path, SpeculativeSelection::ArtifactDefault,
                                        {})
                .backend != SpeculativeBackend::None) {
            throw std::runtime_error("base Flash-Next artifact enabled speculation");
        }
        const std::optional<std::string> helper = "0000:13:00.0";
        for (const auto selection :
             {SpeculativeSelection::ArtifactDefault, SpeculativeSelection::Explicit}) {
            bool wrong_target = false;
            try {
                (void)resolve_speculative_startup(base.path, selection, {}, helper);
            } catch (const std::invalid_argument& error) {
                wrong_target =
                    std::string(error.what())
                        .find("--expert-helper-pci applies only to qwen3.8-flash-next") !=
                    std::string::npos;
            }
            if (!wrong_target) { throw std::runtime_error("27B accepted an expert helper"); }
            bool build_rejected = false;
            try {
                const auto resolved =
                    resolve_speculative_startup(flash_next.path, selection, {}, helper);
                if ((selection == SpeculativeSelection::Explicit &&
                     resolved.backend != SpeculativeBackend::None) ||
                    (selection == SpeculativeSelection::ArtifactDefault &&
                     resolved.backend != SpeculativeBackend::DFlash)) {
                    throw std::runtime_error("helper preflight changed speculation selection");
                }
            } catch (const std::invalid_argument& error) {
                build_rejected =
                    std::string(error.what())
                        .find("requires a build with NINFER_ENABLE_HIP_EXPERT_HELPER=ON") !=
                    std::string::npos;
                if (!build_rejected) { throw; }
            }
#ifdef NINFER_ENABLE_HIP_EXPERT_HELPER
            if (build_rejected) {
                throw std::runtime_error("enabled build rejected helper preflight");
            }
#else
            if (!build_rejected) {
                throw std::runtime_error("OFF build accepted an expert helper");
            }
#endif
        }
        for (const char* pci : {"", "13:00.0", "0000:13:00.8"}) {
            bool malformed = false;
            try {
                (void)resolve_speculative_startup(flash_next.path, SpeculativeSelection::Explicit,
                                                  {}, std::string(pci));
            } catch (const std::invalid_argument& error) {
                malformed =
                    std::string(error.what()).find("--expert-helper-pci") != std::string::npos;
            }
            if (!malformed) { throw std::runtime_error("helper preflight accepted malformed PCI"); }
        }
        std::cout << "ok\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
