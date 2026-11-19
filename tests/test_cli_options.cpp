#include "cli/options.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

ninfer::cli::Options parse(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (std::string& argument : arguments) { argv.push_back(argument.data()); }
    return ninfer::cli::parse_options(static_cast<int>(argv.size()), argv.data());
}

} // namespace

int main() {
    int failures = 0;
    failures += check(!parse({"ninfer", "model.ninfer", "--prompt", "hi"}).expert_helper_pci,
                      "CLI enabled an expert helper by default");
    failures += check(
        parse({"ninfer", "model.ninfer", "--prompt", "hi", "--expert-helper-pci", "ABCD:EF:1F.7"})
                .expert_helper_pci == "abcd:ef:1f.7",
        "CLI did not normalize the complete helper PCI address");
    for (const char* pci : {"", "13:00.0", "0000:13:20.0", "0000:13:00.8", "0000:gg:00.0",
                            "0000:13:00.00", " 0000:13:00.0", "0000:13:00.0x"}) {
        bool rejected = false;
        try {
            (void)parse({"ninfer", "model.ninfer", "--prompt", "hi", "--expert-helper-pci", pci});
        } catch (const std::invalid_argument& error) {
            rejected = std::string(error.what()).find("--expert-helper-pci") != std::string::npos;
        }
        failures += check(rejected, "CLI accepted a malformed or empty helper PCI address");
    }
    bool missing_helper = false;
    try {
        (void)parse({"ninfer", "model.ninfer", "--prompt", "hi", "--expert-helper-pci"});
    } catch (const std::invalid_argument&) { missing_helper = true; }
    failures += check(missing_helper, "CLI accepted a missing helper PCI value");
    failures += check(!parse({"ninfer", "model.ninfer", "--prompt", "hi"}).thinking_token_budget,
                      "CLI enabled a reasoning budget by default");
    failures += check(
        parse({"ninfer", "model.ninfer", "--prompt", "hi", "--thinking-token-budget", "16384"})
                .thinking_token_budget == 16384,
        "CLI lost the reasoning token budget");
    for (const auto& args :
         {std::vector<std::string>{"--thinking-token-budget", "0"},
          std::vector<std::string>{"--thinking-token-budget", "4294967296"},
          std::vector<std::string>{"--thinking-token-budget", "8", "--no-thinking"},
          std::vector<std::string>{"--thinking-token-budget", "8", "--raw-output"}}) {
        bool rejected = false;
        try {
            std::vector<std::string> arguments{"ninfer", "model.ninfer", "--prompt", "hi"};
            arguments.insert(arguments.end(), args.begin(), args.end());
            (void)parse(std::move(arguments));
        } catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected, "CLI accepted an invalid reasoning budget configuration");
    }
    bool mtp_rejected = false;
    try {
        (void)parse({"ninfer", "model.ninfer", "--spec", "mtp", "--draft-tokens", "3"});
    } catch (const std::invalid_argument&) { mtp_rejected = true; }
    failures += check(mtp_rejected, "MTP execution was accepted");


    failures += check(!parse({"ninfer", "model.ninfer", "--prompt", "hi"}).kv_idle_timeout,
                      "idle timeout default must be target-resolved");
    for (const char* seconds : {"0", "15"}) {
        failures +=
            check(parse({"ninfer", "model.ninfer", "--prompt", "hi", "--kv-idle-timeout", seconds})
                          .kv_idle_timeout == static_cast<std::uint32_t>(std::stoul(seconds)),
                  "CLI lost idle timeout seconds");
    }
    bool negative_idle_timeout = false;
    try {
        (void)parse({"ninfer", "model.ninfer", "--prompt", "hi", "--kv-idle-timeout", "-1"});
    } catch (const std::invalid_argument&) { negative_idle_timeout = true; }
    failures += check(negative_idle_timeout, "CLI accepted negative idle timeout");

    const auto system_cache = parse({"ninfer", "model.ninfer", "--prompt", "hi",
                                     "--kv-ram-capacity", "1024", "--kv-ram-system-prompts", "3"});
    failures += check(system_cache.kv_ram_system_prompts == 3 &&
                          system_cache.kv_ram_capacity_bytes == 1024ULL * 1024ULL * 1024ULL,
                      "system prompt count did not use the RAM byte budget");
    const auto system_off =
        parse({"ninfer", "model.ninfer", "--prompt", "hi", "--kv-ram-system-prompts", "0"});
    failures += check(system_off.kv_ram_system_prompts == 0, "zero did not disable system caching");
    for (const char* count : {"1", "-1", "4294967296"}) {
        bool rejected = false;
        try {
            (void)parse(
                {"ninfer", "model.ninfer", "--prompt", "hi", "--kv-ram-system-prompts", count});
        } catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected, "system prompt limit accepted missing RAM or invalid count");
    }

    const auto constrained =
        parse({"ninfer", "model.ninfer", "--prompt", "hi", "--json-schema", "schema.json"});
    failures +=
        check(constrained.json_schema_path == "schema.json", "JSON schema file was not parsed");
    const auto grammar =
        parse({"ninfer", "model.ninfer", "--prompt", "hi", "--grammar", "answer.ebnf"});
    failures += check(grammar.grammar_path == "answer.ebnf", "grammar file was not parsed");
    bool conflict_rejected = false;
    try {
        (void)parse({"ninfer", "model.ninfer", "--prompt", "hi", "--json-object", "--grammar",
                     "answer.ebnf"});
    } catch (const std::invalid_argument&) { conflict_rejected = true; }
    failures += check(conflict_rejected, "CLI accepted competing output constraints");

    const ninfer::cli::Options pin =
        parse({"ninfer", "model.ninfer", "--prompt", "hi", "--capture-context-checkpoint"});
    failures += check(pin.capture_context_checkpoint,
                      "--capture-context-checkpoint did not set the request pin");
    failures += check(pin.context_checkpoint_marks.empty(),
                      "omitted --context-checkpoints did not leave the ladder off");

    const ninfer::cli::Options on =
        parse({"ninfer", "model.ninfer", "--prompt", "hi", "--context-checkpoints", "on"});
    failures += check(
        std::ranges::equal(on.context_checkpoint_marks, ninfer::kStandardContextCheckpointMarks),
        "--context-checkpoints on did not select the standard ladder");

    const ninfer::cli::Options off =
        parse({"ninfer", "model.ninfer", "--prompt", "hi", "--context-checkpoints", "off"});
    failures += check(off.context_checkpoint_marks.empty(),
                      "--context-checkpoints off did not disable the ladder");

    const ninfer::cli::Options custom =
        parse({"ninfer", "model.ninfer", "--prompt", "hi", "--context-checkpoints", "8192,16384"});
    failures += check(custom.context_checkpoint_marks == std::vector<std::uint32_t>{8192u, 16384u},
                      "--context-checkpoints custom list was not parsed");

    const ninfer::cli::Options flash_next =
        parse({"ninfer", "model.ninfer", "--prompt", "hi", "--ngram-table", "table.bin",
               "--qsa-select-budget", "2048", "--expert-route-profile", "routes.json"});
    failures += check(flash_next.ngram_table_path == "table.bin" &&
                          flash_next.qsa_select_budget == ninfer::QsaSelectBudget::Blocks2048,
                      "--ngram-table / --qsa-select-budget were not parsed");
    failures += check(flash_next.expert_route_profile_path == "routes.json",
                      "--expert-route-profile was not parsed");
    bool empty_profile = false;
    try {
        (void)parse({"ninfer", "model.ninfer", "--prompt", "hi", "--expert-route-profile", ""});
    } catch (const std::invalid_argument&) { empty_profile = true; }
    failures += check(empty_profile, "an empty --expert-route-profile was accepted");
    failures += check(!on.qsa_select_budget.has_value() && on.ngram_table_path.empty() &&
                          on.expert_route_profile_path.empty(),
                      "omitted Flash-Next options were not left unset");
    bool bad_budget = false;
    try {
        (void)parse({"ninfer", "model.ninfer", "--prompt", "hi", "--qsa-select-budget", "768"});
    } catch (const std::invalid_argument&) { bad_budget = true; }
    failures += check(bad_budget, "an unsupported --qsa-select-budget was accepted");

    failures += check(ninfer::cli::usage_text("ninfer").find("--capture-context-checkpoint") !=
                          std::string::npos,
                      "CLI help omits --capture-context-checkpoint");
    failures +=
        check(ninfer::cli::usage_text("ninfer").find("--no-p-less-sampling") != std::string::npos,
              "CLI help omits --no-p-less-sampling");

    const ninfer::cli::Options defaults = parse({"ninfer", "model.ninfer", "--prompt", "hi"});
    failures += check(defaults.kv_capacity.mode == ninfer::KvCapacityMode::Automatic &&
                          defaults.kv_capacity.automatic_headroom_bytes ==
                              ninfer::kDefaultKvCapacityHeadroomBytes,
                      "CLI did not default to automatic KV capacity");
    const auto headroom =
        parse({"ninfer", "model.ninfer", "--prompt", "hi", "--kv-capacity-headroom", "512"});
    failures += check(headroom.kv_capacity.automatic_headroom_bytes == 512ULL * 1024ULL * 1024ULL,
                      "CLI headroom override did not apply to default automatic sizing");
    const auto disabled = parse({"ninfer", "model.ninfer", "--prompt", "hi", "--spec", "none"});
    failures +=
        check(disabled.speculative.backend == ninfer::SpeculativeBackend::None &&
                  disabled.speculative_selection == ninfer::product::SpeculativeSelection::Explicit,
              "CLI --spec none did not disable artifact-default speculation");
    failures += check(defaults.prefill_chunk == 4096, "CLI prefill chunk default is not 4096");
    failures += check(defaults.sampling.p_less, "CLI did not enable p-less by default");
    const ninfer::cli::Options production = parse(
        {"ninfer", "model.ninfer", "--prompt", "hi", "--no-p-less-sampling", "--top-p", "0.5"});
    failures += check(!production.sampling.p_less && production.sampling.top_p == 0.5F,
                      "--no-p-less-sampling did not opt into the production sampler");

    failures += check(!defaults.speculative.dflash_p_less_draft_temperature.has_value(),
                      "CLI p-less draft temperature is pinned by default");
    const ninfer::cli::Options greedy_drafts =
        parse({"ninfer", "model.ninfer", "--prompt", "hi", "--spec", "dflash", "--draft-tokens",
               "5", "--dflash-p-less-draft-temperature", "0"});
    failures += check(greedy_drafts.speculative.dflash_p_less_draft_temperature == 0.0F,
                      "--dflash-p-less-draft-temperature did not set SpeculativeOptions");

    const ninfer::cli::Options dflash_vision =
        parse({"ninfer", "model.ninfer", "--prompt", "hi", "--spec", "dflash", "--draft-tokens",
               "3", "--vision"});
    failures += check(dflash_vision.enable_vision &&
                          dflash_vision.speculative.backend == ninfer::SpeculativeBackend::DFlash,
                      "CLI did not accept DFlash and Vision together");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
