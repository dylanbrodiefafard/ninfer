#pragma once

#include "ninfer/types.h"

#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::product {

[[nodiscard]] inline SpeculativeBackend parse_speculative_backend(std::string_view value) {
    if (value == "none") { return SpeculativeBackend::None; }
    if (value == "dflash") { return SpeculativeBackend::DFlash; }
    throw std::invalid_argument("invalid speculative backend: " + std::string(value));
}

[[nodiscard]] inline const char* speculative_backend_name(SpeculativeBackend backend) noexcept {
    switch (backend) {
    case SpeculativeBackend::None:
        return "none";
    case SpeculativeBackend::Mtp:
        return "mtp";
    case SpeculativeBackend::DFlash:
        return "dflash";
    }
    return "unknown";
}

inline void validate_speculative_cli_options(const SpeculativeOptions& options) {
    validate_speculative_backend(options.backend);
    if (options.backend == SpeculativeBackend::None) {
        if (options.draft_tokens != 0 || options.proposal_head != ProposalHead::Full ||
            options.dflash_verify_width != 0 || options.adaptive_draft ||
            options.dflash_p_less_draft_temperature.has_value()) {
            throw std::invalid_argument(
                "--draft-tokens, --lm-head-draft, --dflash-verify-width, and --adaptive-draft "
                "require --spec dflash");
        }
        return;
    }
    if (options.draft_tokens == 0 || options.draft_tokens > 15) {
        throw std::invalid_argument("--spec dflash requires --draft-tokens in [1,15]");
    }
    if (options.dflash_verify_width != 0 &&
        (options.dflash_verify_width < 2 || options.dflash_verify_width > 16)) {
        throw std::invalid_argument("--dflash-verify-width must be in [2,16]");
    }
}

} // namespace ninfer::product
