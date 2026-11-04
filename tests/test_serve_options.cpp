#include "serve/serve_options.h"
#include "serve/translate.h"

#include <array>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ninfer::serve;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

ServeOptions parse(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (std::string& argument : arguments) { argv.push_back(argument.data()); }
    return parse_serve_options(static_cast<int>(argv.size()), argv.data());
}

ninfer::RequestOptions test_request_options(const GenerationRequest& request,
                                            const ServeOptions& server) {
    return to_request_options(request, server, resolve_request_semantics(request, server, {}));
}

} // namespace

int main() {
    int failures = 0;
    failures += check(ServeOptions{}.pending_timeout_ms == 300000,
                      "default pending timeout must be five minutes");

    static_assert(ninfer::kMaximumConcurrency == 16);

    const auto system_cache = parse({"ninfer-serve", "model.ninfer", "--kv-ram-capacity", "1024",
                                     "--kv-ram-system-prompts", "3"});
    failures += check(system_cache.kv_ram_system_prompts == 3 &&
                          system_cache.kv_ram_capacity_bytes == 1024ULL * 1024ULL * 1024ULL,
                      "system prompt count did not use the RAM byte budget");
    const auto system_off = parse({"ninfer-serve", "model.ninfer", "--kv-ram-system-prompts", "0"});
    failures += check(system_off.kv_ram_system_prompts == 0, "zero did not disable system caching");
    for (const char* count : {"1", "-1", "4294967296"}) {
        bool rejected = false;
        try {
            (void)parse({"ninfer-serve", "model.ninfer", "--kv-ram-system-prompts", count});
        } catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected, "system prompt limit accepted missing RAM or invalid count");
    }
    failures +=
        check(!ServeOptions{}.default_thinking_token_budget &&
                  !parse({"ninfer-serve", "model.ninfer"}).default_thinking_token_budget &&
                  !parse({"ninfer-serve", "model.ninfer", "--default-thinking-token-budget", "off"})
                       .default_thinking_token_budget,
              "server default reasoning budget must remain unlimited");
    for (const auto& [wire, expected] : std::array<std::pair<const char*, std::uint32_t>, 3>{
             {{"1", 1}, {"16384", 16384}, {"4294967295", 4294967295U}}}) {
        failures +=
            check(parse({"ninfer-serve", "model.ninfer", "--default-thinking-token-budget", wire})
                          .default_thinking_token_budget == expected,
                  "server default reasoning cap did not preserve uint32 value");
    }
    for (const char* wire : {"0", "-1", "1.5", "4294967296", "18446744073709551616", "on"}) {
        bool rejected = false;
        try {
            (void)parse({"ninfer-serve", "model.ninfer", "--default-thinking-token-budget", wire});
        } catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected, "server accepted invalid default reasoning budget");
    }
    failures += check(!parse({"ninfer-serve", "model.ninfer"}).default_reasoning_effort &&
                          !parse({"ninfer-serve", "model.ninfer", "--default-reasoning-effort",
                                  "low", "--default-reasoning-effort", "template"})
                               .default_reasoning_effort,
                      "omitted or template default effort must use the artifact default");
    for (const auto& [wire, expected] :
         std::array<std::pair<const char*, ninfer::ReasoningEffort>, 3>{
             {{"low", ninfer::ReasoningEffort::Low},
              {"medium", ninfer::ReasoningEffort::Medium},
              {"xhigh", ninfer::ReasoningEffort::XHigh}}}) {
        const auto server =
            parse({"ninfer-serve", "model.ninfer", "--default-reasoning-effort", wire});
        failures += check(server.default_reasoning_effort == expected,
                          "default effort flag did not preserve the selected effort");
    }
    for (const char* wire : {"none", "minimal", "high", "max", "off", ""}) {
        bool rejected = false;
        try {
            (void)parse({"ninfer-serve", "model.ninfer", "--default-reasoning-effort", wire});
        } catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected, "server accepted invalid default reasoning effort");
    }
    auto low_server = parse({"ninfer-serve", "model.ninfer", "--default-reasoning-effort", "low"});
    ninfer::PromptCapabilities effort_capabilities;
    effort_capabilities.enable_thinking  = true;
    effort_capabilities.reasoning_effort = {.low            = true,
                                            .medium         = true,
                                            .xhigh          = true,
                                            .default_effort = ninfer::ReasoningEffort::XHigh};
    GenerationRequest effort_request;
    effort_request.messages.resize(1);
    const auto inherit = resolve_request_semantics(effort_request, low_server, effort_capabilities);
    failures +=
        check(inherit.enable_thinking && inherit.reasoning_effort == ninfer::ReasoningEffort::Low &&
                  to_prompt_input(effort_request, inherit, {}).options.reasoning_effort ==
                      ninfer::ReasoningEffort::Low,
              "omitted request effort did not reach PromptInput as server default low");
    effort_request.reasoning_effort = RequestedReasoningEffort::Medium;
    failures += check(resolve_request_semantics(effort_request, low_server, effort_capabilities)
                              .reasoning_effort == ninfer::ReasoningEffort::Medium,
                      "explicit request effort did not override server default");
    effort_request.reasoning_effort = RequestedReasoningEffort::None;
    const auto no_reasoning =
        resolve_request_semantics(effort_request, low_server, effort_capabilities);
    failures += check(!no_reasoning.enable_thinking && !no_reasoning.reasoning_effort,
                      "none did not disable thinking under a low server default");
    effort_request.reasoning_effort.reset();
    effort_request.enable_thinking = false;
    const auto disabled =
        resolve_request_semantics(effort_request, low_server, effort_capabilities);
    failures += check(!disabled.enable_thinking && !disabled.reasoning_effort,
                      "disabled thinking inherited server effort");
    low_server.enable_thinking     = false;
    effort_request.enable_thinking = true;
    failures += check(resolve_request_semantics(effort_request, low_server, effort_capabilities)
                              .reasoning_effort == ninfer::ReasoningEffort::Low,
                      "request enabling thinking did not inherit low effort");
    const auto default_spec = parse({"ninfer-serve", "model.ninfer"});
    failures += check(default_spec.speculative.backend == ninfer::SpeculativeBackend::DFlash &&
                          default_spec.speculative.draft_tokens == 7 &&
                          default_spec.speculative.adaptive_draft &&
                          default_spec.speculative.proposal_head == ninfer::ProposalHead::Optimized,
                      "defaults must enable DFlash, adaptive K7, and the optimized head");
    const auto no_spec = parse({"ninfer-serve", "model.ninfer", "--spec", "none"});
    failures +=
        check(no_spec.speculative.backend == ninfer::SpeculativeBackend::None &&
                  no_spec.speculative.draft_tokens == 0 && !no_spec.speculative.adaptive_draft &&
                  no_spec.speculative.proposal_head == ninfer::ProposalHead::Full,
              "--spec none retained speculative allocations");
    const auto full_head = parse({"ninfer-serve", "model.ninfer", "--no-lm-head-draft"});
    failures += check(full_head.speculative.proposal_head == ninfer::ProposalHead::Full,
                      "--no-lm-head-draft did not select the full proposal head");
    const auto optimized_head =
        parse({"ninfer-serve", "model.ninfer", "--no-lm-head-draft", "--lm-head-draft"});
    failures += check(optimized_head.speculative.proposal_head == ninfer::ProposalHead::Optimized,
                      "last proposal-head option did not win");
    const auto mtp =
        parse({"ninfer-serve", "model.ninfer", "--spec", "mtp", "--draft-tokens", "3"});
    failures += check(mtp.speculative.backend == ninfer::SpeculativeBackend::Mtp &&
                          mtp.speculative.draft_tokens == 3 && !mtp.speculative.adaptive_draft &&
                          mtp.speculative.proposal_head == ninfer::ProposalHead::Full,
                      "DFlash defaults leaked into MTP");
    bool stray_spec_options_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--lm-head-draft", "--spec", "none"});
    } catch (const std::invalid_argument&) { stray_spec_options_rejected = true; }
    failures +=
        check(stray_spec_options_rejected, "--spec none accepted an enabled speculative head");

    const auto dflash_defaults = parse({"ninfer-serve", "model.ninfer", "--spec", "dflash"});
    failures += check(dflash_defaults.speculative.draft_tokens == 7 &&
                          dflash_defaults.speculative.adaptive_draft,
                      "DFlash must default to adaptive drafting with a maximum of seven");
    const auto fixed_dflash = parse({"ninfer-serve", "model.ninfer", "--spec", "dflash",
                                     "--draft-tokens", "5", "--no-adaptive-draft"});
    failures += check(fixed_dflash.speculative.draft_tokens == 5 &&
                          !fixed_dflash.speculative.adaptive_draft,
                      "explicit fixed DFlash length was not preserved");
    const auto adaptive_override = parse({"ninfer-serve", "model.ninfer", "--spec", "dflash",
                                          "--no-adaptive-draft", "--adaptive-draft"});
    failures += check(adaptive_override.speculative.adaptive_draft,
                      "last adaptive-draft option did not win");
    bool zero_drafts_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--spec", "dflash", "--draft-tokens", "0"});
    } catch (const std::invalid_argument&) { zero_drafts_rejected = true; }
    failures += check(zero_drafts_rejected, "explicit zero drafts silently selected the default");

    const ServeOptions defaults = parse({"ninfer-serve", "model.ninfer"});
    failures += check(defaults.prefill_chunk == ninfer::kDefaultPrefillChunk,
                      "serve prefill chunk diverges from the product default");
    failures += check(defaults.allow_prefix_reuse, "prefix reuse is not enabled by default");
    failures +=
        check(!defaults.preserve_thinking, "thinking history is unexpectedly preserved by default");
    failures += check(defaults.system_prepend.empty(), "system prepend is not empty by default");
    failures += check(!defaults.enable_vision, "Vision is not disabled by default");
    failures += check(defaults.request_log_jsonl.empty(),
                      "request JSONL logging is not disabled by default");
    failures += check(defaults.log_stats_interval_ms == 5000,
                      "periodic throughput interval default mismatch");
    failures += check(defaults.kv_capacity.mode == ninfer::KvCapacityMode::Automatic &&
                          defaults.kv_capacity.automatic_headroom_bytes ==
                              ninfer::kDefaultKvCapacityHeadroomBytes,
                      "default KV capacity must use automatic sizing");
    failures += check(defaults.speculative.backend == ninfer::SpeculativeBackend::DFlash,
                      "DFlash is not enabled by default");
    failures += check(defaults.response_store_max_records == kDefaultResponseStoreRecords &&
                          defaults.response_store_max_bytes == kDefaultResponseStoreBytes,
                      "Responses store defaults mismatch");
    failures += check(!defaults.model_id_override.has_value(),
                      "model id override is unexpectedly configured by default");
    failures += check(
        !defaults.sampling_overrides.temperature && !defaults.sampling_overrides.top_p &&
            !defaults.sampling_overrides.top_k && !defaults.sampling_overrides.presence_penalty &&
            !defaults.sampling_overrides.frequency_penalty && defaults.sampling_overrides.p_less,
        "server did not enable p-less by default");
    failures += check(!defaults.speculative.dflash_p_less_draft_temperature.has_value(),
                      "p-less draft temperature is pinned by default");
    const ServeOptions draft_temperature =
        parse({"ninfer-serve", "model.ninfer", "--spec", "dflash", "--draft-tokens", "5",
               "--dflash-p-less-draft-temperature", "0.6"});
    failures += check(draft_temperature.speculative.dflash_p_less_draft_temperature == 0.6f,
                      "--dflash-p-less-draft-temperature did not set SpeculativeOptions");
    bool negative_draft_temperature_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--dflash-p-less-draft-temperature", "-1"});
    } catch (const std::invalid_argument&) { negative_draft_temperature_rejected = true; }
    failures += check(negative_draft_temperature_rejected,
                      "a negative --dflash-p-less-draft-temperature was accepted");
    failures += check(resolve_public_model_id(defaults, "artifact-model") == "artifact-model",
                      "artifact model id was not selected by default");

    const ServeOptions model_alias =
        parse({"ninfer-serve", "model.ninfer", "--model-id", "deployment-alias"});
    failures +=
        check(model_alias.model_id_override == "deployment-alias" &&
                  resolve_public_model_id(model_alias, "artifact-model") == "deployment-alias",
              "explicit model id did not override the artifact identity");

    bool empty_model_id_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--model-id", ""});
    } catch (const std::invalid_argument&) { empty_model_id_rejected = true; }
    failures += check(empty_model_id_rejected, "empty --model-id was accepted");

    const ServeOptions dflash = parse({"ninfer-serve", "model.ninfer", "--spec", "dflash",
                                       "--draft-tokens", "5", "--lm-head-draft"});
    failures += check(dflash.speculative.backend == ninfer::SpeculativeBackend::DFlash,
                      "--spec dflash did not select DFlash");
    failures += check(dflash.speculative.draft_tokens == 5,
                      "--draft-tokens did not preserve the DFlash window");
    failures += check(dflash.speculative.proposal_head == ninfer::ProposalHead::Optimized,
                      "--lm-head-draft did not select the optimized proposal head");
    failures += check(dflash.speculative.dflash_verify_width == 0,
                      "DFlash verify width default is not auto");

    const ServeOptions dflash_w6 = parse({"ninfer-serve", "model.ninfer", "--spec", "dflash",
                                          "--draft-tokens", "5", "--dflash-verify-width", "6"});
    failures += check(dflash_w6.speculative.dflash_verify_width == 6,
                      "--dflash-verify-width did not preserve the chain width");

    bool mtp_width_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--spec", "mtp", "--draft-tokens", "3",
                     "--dflash-verify-width", "6"});
    } catch (const std::invalid_argument&) { mtp_width_rejected = true; }
    failures += check(mtp_width_rejected, "--dflash-verify-width was accepted with MTP");

    const ServeOptions dflash_vision = parse(
        {"ninfer-serve", "model.ninfer", "--spec", "dflash", "--draft-tokens", "5", "--vision"});
    failures += check(dflash_vision.enable_vision &&
                          dflash_vision.speculative.backend == ninfer::SpeculativeBackend::DFlash,
                      "DFlash and Vision were not accepted together");

    const auto implicit_backend = parse({"ninfer-serve", "model.ninfer", "--draft-tokens", "3"});
    failures += check(implicit_backend.speculative.backend == ninfer::SpeculativeBackend::DFlash &&
                          implicit_backend.speculative.draft_tokens == 3,
                      "--draft-tokens did not configure the default DFlash backend");

    const ServeOptions configured = parse(
        {"ninfer-serve", "model.ninfer", "--no-prefix-reuse", "--vision", "--max-concurrency", "4",
         "--max-pending-requests", "12", "--pending-timeout-ms", "2500", "--max-context", "4096",
         "--kv-capacity", "8192", "--log-stats-interval-ms", "0", "--preserve-thinking"});
    failures += check(!configured.allow_prefix_reuse,
                      "--no-prefix-reuse did not disable server prefix reuse");
    const ServeOptions off_marks =
        parse({"ninfer-serve", "model.ninfer", "--context-checkpoints", "off"});
    failures += check(off_marks.context_checkpoint_marks.has_value() &&
                          off_marks.context_checkpoint_marks->empty(),
                      "--context-checkpoints off did not disable the ladder");
    const ServeOptions custom_marks =
        parse({"ninfer-serve", "model.ninfer", "--context-checkpoints", "8192,16384"});
    failures += check(
        custom_marks.context_checkpoint_marks ==
            std::optional<std::vector<std::uint32_t>>(std::vector<std::uint32_t>{8192u, 16384u}),
        "--context-checkpoints custom list was not parsed");
    bool bad_marks = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--context-checkpoints", "16384,8192"});
    } catch (const std::invalid_argument&) { bad_marks = true; }
    failures += check(bad_marks, "non-increasing context-checkpoints was accepted");
    bool zero_mark = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--context-checkpoints", "0"});
    } catch (const std::invalid_argument&) { zero_mark = true; }
    failures += check(zero_mark, "zero context-checkpoint mark was accepted");
    bool too_many = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--context-checkpoints",
                     "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17"});
    } catch (const std::invalid_argument&) { too_many = true; }
    failures += check(too_many, "17 context-checkpoint marks were accepted");
    failures +=
        check(serve_usage_text("ninfer-serve").find("--context-checkpoints") != std::string::npos,
              "serve help omits --context-checkpoints");
    failures +=
        check(serve_usage_text("ninfer-serve").find("--max-concurrency 1..16") != std::string::npos,
              "serve help omits the product concurrency range");
    failures += check(configured.enable_vision, "--vision did not enable Vision");
    failures +=
        check(configured.preserve_thinking, "--preserve-thinking did not reach serving options");

    const ServeOptions prepended =
        parse({"ninfer-serve", "model.ninfer", "--system-prepend", "Stay terse."});
    failures += check(prepended.generation_recovery, "generation recovery must default to enabled");
    const ServeOptions no_recovery =
        parse({"ninfer-serve", "model.ninfer", "--no-generation-recovery"});
    failures += check(!no_recovery.generation_recovery,
                      "--no-generation-recovery did not disable recovery");
    failures += check(prepended.system_prepend == "Stay terse.",
                      "--system-prepend did not store the provided text");

    bool empty_prepend_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--system-prepend", ""});
    } catch (const std::invalid_argument&) { empty_prepend_rejected = true; }
    failures += check(empty_prepend_rejected, "empty --system-prepend was accepted");
    failures +=
        check(configured.max_concurrency == 4, "--max-concurrency did not reach serving options");
    bool excessive_concurrency_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--max-concurrency", "17"});
    } catch (const std::invalid_argument&) { excessive_concurrency_rejected = true; }
    failures += check(excessive_concurrency_rejected,
                      "--max-concurrency accepted more than sixteen active requests");
    failures += check(parse({"ninfer-serve", "model.ninfer", "--spec", "mtp", "--draft-tokens", "5",
                             "--max-concurrency", "16"})
                              .max_concurrency == 16,
                      "--max-concurrency rejected sixteen active requests");
    failures += check(configured.max_context == 4096 &&
                          configured.kv_capacity.mode == ninfer::KvCapacityMode::Explicit &&
                          configured.kv_capacity.explicit_tokens == 8192,
                      "context and KV capacity options were not kept distinct");
    failures += check(configured.max_pending_requests == 12,
                      "--max-pending-requests did not reach serving options");
    failures += check(configured.pending_timeout_ms == 2500,
                      "--pending-timeout-ms did not reach serving options");
    failures += check(configured.log_stats_interval_ms == 0,
                      "--log-stats-interval-ms did not disable periodic reporting");

    const ServeOptions response_store = parse(
        {"ninfer-serve", "model.ninfer", "--response-store-max-records", "42",
         "--response-store-max-mib", "8", "--response-store-location", "/tmp/response-history"});
    failures += check(response_store.response_store_max_records == 42 &&
                          response_store.response_store_max_bytes == (8ULL << 20) &&
                          response_store.response_store_location == "/tmp/response-history",
                      "Responses store limits did not reach serving options");

    const ServeOptions sampling =
        parse({"ninfer-serve", "model.ninfer", "--temperature", "0", "--top-p", "0.9", "--top-k",
               "40", "--min-p", "0.1", "--presence-penalty", "1.25", "--frequency-penalty", "-0.5",
               "--seed", "0"});
    failures += check(sampling.sampling_overrides.temperature == 0.0F &&
                          sampling.sampling_overrides.top_p == 0.9F &&
                          sampling.sampling_overrides.top_k == 40 &&
                          sampling.sampling_overrides.min_p == 0.1F &&
                          sampling.sampling_overrides.presence_penalty == 1.25F &&
                          sampling.sampling_overrides.frequency_penalty == -0.5F &&
                          sampling.sampling_overrides.seed == 0,
                      "server sampling flags did not preserve explicit values and zeros");

    const ServeOptions production = parse({"ninfer-serve", "model.ninfer", "--no-p-less-sampling"});
    failures += check(!production.sampling_overrides.p_less,
                      "--no-p-less-sampling did not reach serving options");
    failures +=
        check(serve_usage_text("ninfer-serve").find("--no-p-less-sampling") != std::string::npos,
              "serve help omits --no-p-less-sampling");

    GenerationRequest request;
    request.max_tokens = 1;
    ninfer::PromptCapabilities prompt_capabilities;
    prompt_capabilities.enable_thinking = true;
    failures += check(test_request_options(request, defaults).execution.allow_prefix_reuse,
                      "default server policy did not reach Engine options");
    failures += check(!test_request_options(request, configured).execution.allow_prefix_reuse,
                      "disabled server policy did not reach Engine options");
    request.capture_context_checkpoint = true;
    failures += check(test_request_options(request, defaults).execution.capture_context_checkpoint,
                      "request capture_context_checkpoint did not reach Engine options");
    request.capture_context_checkpoint              = false;
    const ninfer::RequestOptions inherited_sampling = test_request_options(request, sampling);
    failures += check(inherited_sampling.execution.sampling.temperature == 0.0F &&
                          inherited_sampling.execution.sampling.top_p == 0.9F &&
                          inherited_sampling.execution.sampling.seed == 0,
                      "server sampling overrides did not reach Engine options");
    failures += check(test_request_options(request, defaults).execution.sampling.p_less,
                      "default p-less mode did not reach Engine overrides");
    failures += check(!test_request_options(request, production).execution.sampling.p_less,
                      "--no-p-less-sampling did not reach Engine overrides");
    request.sampling.temperature = 1.1;
    failures +=
        check(test_request_options(request, sampling).execution.sampling.temperature == 1.1F,
              "request sampling override did not win over the server override");
    failures +=
        check(resolve_request_semantics(request, configured, prompt_capabilities).preserve_thinking,
              "server preserve-thinking default was not resolved");
    request.preserve_thinking = false;
    failures += check(
        !resolve_request_semantics(request, configured, prompt_capabilities).preserve_thinking,
        "request preserve-thinking override did not win");

    failures +=
        check(serve_usage_text("ninfer-serve").find("--no-prefix-reuse") != std::string::npos,
              "serve help omits --no-prefix-reuse");
    const ServeOptions eager = parse({"ninfer-serve", "model.ninfer", "--no-device-graph"});
    failures += check(!eager.use_device_graph,
                      "--no-device-graph did not disable serving Device Graph decode");
    failures +=
        check(serve_usage_text("ninfer-serve").find("--no-device-graph") != std::string::npos,
              "serve help omits --no-device-graph");
    failures +=
        check(serve_usage_text("ninfer-serve").find("--preserve-thinking") != std::string::npos,
              "serve help omits --preserve-thinking");
    failures +=
        check(serve_usage_text("ninfer-serve").find("--system-prepend") != std::string::npos,
              "serve help omits --system-prepend");
    failures += check(serve_usage_text("ninfer-serve").find("--vision") != std::string::npos,
                      "serve help omits --vision");
    failures +=
        check(serve_usage_text("ninfer-serve").find("--log-stats-interval-ms") != std::string::npos,
              "serve help omits --log-stats-interval-ms");
    failures += check(serve_usage_text("ninfer-serve").find("--kv-capacity") != std::string::npos,
                      "serve help omits --kv-capacity");
    failures +=
        check(serve_usage_text("ninfer-serve").find("--kv-ram-capacity") != std::string::npos,
              "serve help omits --kv-ram-capacity");
    failures +=
        check(serve_usage_text("ninfer-serve").find("--kv-disk-capacity") != std::string::npos,
              "serve help omits --kv-disk-capacity");
    failures +=
        check(serve_usage_text("ninfer-serve").find("--kv-disk-location") != std::string::npos,
              "serve help omits --kv-disk-location");
    failures +=
        check(serve_usage_text("ninfer-serve").find("--kv-disk-compress") != std::string::npos,
              "serve help omits --kv-disk-compress");
    failures += check(
        serve_usage_text("ninfer-serve").find("pinned host KV prefix-cache capacity in MiB") !=
            std::string::npos,
        "serve help omits KV RAM MiB wording");
    failures += check(serve_usage_text("ninfer-serve").find("--response-store-max-mib") !=
                          std::string::npos,
                      "serve help omits Responses store limits");
    failures +=
        check(serve_usage_text("ninfer-serve").find("identity.model_id") != std::string::npos,
              "serve help omits the artifact-derived model id default");

    const ServeOptions inherited =
        parse({"ninfer-serve", "model.ninfer", "--max-context", "16384"});
    failures += check(inherited.max_context == 16384 &&
                          inherited.kv_capacity.mode == ninfer::KvCapacityMode::Automatic,
                      "changing max context disabled automatic KV sizing");

    const ServeOptions automatic = parse({"ninfer-serve", "model.ninfer", "--kv-capacity", "auto"});
    failures += check(automatic.kv_capacity.mode == ninfer::KvCapacityMode::Automatic &&
                          automatic.kv_capacity.explicit_tokens == 0 &&
                          automatic.kv_capacity.automatic_headroom_bytes ==
                              ninfer::kDefaultKvCapacityHeadroomBytes,
                      "--kv-capacity auto did not select automatic sizing");
    const ServeOptions headroom =
        parse({"ninfer-serve", "model.ninfer", "--kv-capacity-headroom", "512"});
    failures +=
        check(headroom.kv_capacity.mode == ninfer::KvCapacityMode::Automatic &&
                  headroom.kv_capacity.automatic_headroom_bytes == 512ULL * 1024ULL * 1024ULL,
              "--kv-capacity-headroom did not set automatic headroom in MiB");
    bool headroom_rejected = false;
    try {
        (void)parse({"ninfer-serve", "model.ninfer", "--kv-capacity", "8192",
                     "--kv-capacity-headroom", "512"});
    } catch (const std::invalid_argument&) { headroom_rejected = true; }
    failures += check(headroom_rejected, "--kv-capacity-headroom without auto was accepted");

    const ServeOptions ram_off = parse({"ninfer-serve", "model.ninfer"});
    failures +=
        check(ram_off.kv_ram_capacity_bytes == 0, "omitted --kv-ram-capacity did not default off");
    const ServeOptions ram_explicit_off =
        parse({"ninfer-serve", "model.ninfer", "--kv-ram-capacity", "off"});
    failures += check(ram_explicit_off.kv_ram_capacity_bytes == 0,
                      "--kv-ram-capacity off did not disable the tier");
    const ServeOptions ram_mib = parse({"ninfer-serve", "model.ninfer", "--kv-ram-capacity", "1"});
    failures += check(ram_mib.kv_ram_capacity_bytes == 1024ULL * 1024ULL,
                      "--kv-ram-capacity 1 did not convert MiB to bytes");
    auto reject_ram = [&](const char* value, const char* message) {
        bool rejected = false;
        try {
            parse({"ninfer-serve", "model.ninfer", "--kv-ram-capacity", value});
        } catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected, message);
    };
    reject_ram("0", "--kv-ram-capacity 0 was accepted");
    reject_ram("1.5", "--kv-ram-capacity float was accepted");
    reject_ram("", "--kv-ram-capacity empty was accepted");
    reject_ram("auto", "--kv-ram-capacity non-decimal was accepted");
    reject_ram("17592186044416", "--kv-ram-capacity overflow was accepted");

    const ServeOptions disk_off = parse({"ninfer-serve", "model.ninfer"});
    failures += check(disk_off.kv_disk_capacity_bytes == 0 && disk_off.kv_disk_location.empty() &&
                          disk_off.kv_disk_compress == ninfer::KvDiskCompress::Off,
                      "omitted disk flags did not default off");
    const ServeOptions disk_on =
        parse({"ninfer-serve", "model.ninfer", "--kv-ram-capacity", "1", "--kv-disk-capacity", "2",
               "--kv-disk-location", "/tmp/ninfer-kv-disk", "--kv-disk-compress", "zstd"});
    failures += check(disk_on.kv_disk_capacity_bytes == 2ULL * 1024ULL * 1024ULL &&
                          disk_on.kv_disk_location == "/tmp/ninfer-kv-disk" &&
                          disk_on.kv_disk_compress == ninfer::KvDiskCompress::Zstd,
                      "disk flags did not parse");
    auto reject_disk = [&](std::vector<std::string> args, const char* message) {
        bool rejected = false;
        try {
            parse(std::move(args));
        } catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected, message);
    };
    reject_disk({"ninfer-serve", "model.ninfer", "--kv-disk-capacity", "1"},
                "disk capacity without location was accepted");
    reject_disk({"ninfer-serve", "model.ninfer", "--kv-disk-location", "/tmp/x"},
                "disk location without capacity was accepted");
    reject_disk(
        {"ninfer-serve", "model.ninfer", "--kv-disk-capacity", "1", "--kv-disk-location", "/tmp/x"},
        "disk without RAM was accepted");
    reject_disk(
        {"ninfer-serve", "model.ninfer", "--kv-ram-capacity", "1", "--kv-disk-capacity", "0"},
        "--kv-disk-capacity 0 was accepted");
    reject_disk({"ninfer-serve", "model.ninfer", "--kv-ram-capacity", "1", "--kv-disk-capacity",
                 "1", "--kv-disk-location", "/tmp/x", "--kv-disk-compress", "lz4"},
                "unknown --kv-disk-compress was accepted");

    const ServeOptions logged = parse({"ninfer-serve", "model.ninfer", "--request-log-jsonl",
                                       "requests.jsonl", "--api-key", "do-not-log"});
    failures += check(logged.request_log_jsonl == "requests.jsonl",
                      "--request-log-jsonl did not preserve its path");
    failures +=
        check(serve_usage_text("ninfer-serve").find("--request-log-jsonl") != std::string::npos,
              "serve help omits --request-log-jsonl");
    bool secret_present    = false;
    bool redaction_present = false;
    for (const std::string& argument : logged.startup_argv) {
        secret_present    = secret_present || argument == "do-not-log";
        redaction_present = redaction_present || argument == "<redacted>";
    }
    failures += check(!secret_present, "startup argv retained the API key");
    failures += check(redaction_present, "startup argv omitted the API-key redaction marker");

    const std::string help = serve_usage_text("ninfer-serve");
    failures += check(help.find("--kv-capacity") != std::string::npos &&
                          help.find("--kv-ram-capacity") != std::string::npos,
                      "serve help omits the fixed-cache capacity controls");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
