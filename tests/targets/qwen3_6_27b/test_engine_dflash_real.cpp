#include "ninfer/engine.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

ninfer::EngineOptions base_engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path  = artifact;
    options.max_context    = 256;
    options.kv_capacity    = ninfer::KvCapacityPolicy::explicit_capacity(256);
    options.prefill_chunk  = 128;
    options.kv_cache       = ninfer::KvCacheStorage::BFloat16;
    options.use_cuda_graph = true;
    options.enable_vision  = false;
    return options;
}

ninfer::EngineOptions speculative_engine_options(const char* artifact,
                                                 ninfer::SpeculativeBackend backend,
                                                 std::uint32_t draft_tokens,
                                                 std::uint32_t max_concurrency) {
    ninfer::EngineOptions options     = base_engine_options(artifact);
    options.speculative.backend       = backend;
    options.speculative.draft_tokens  = draft_tokens;
    options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    options.max_concurrency           = max_concurrency;
    return options;
}

ninfer::RequestOptions greedy_options(std::uint32_t outputs) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = outputs;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = false;
    options.stop.include_model_defaults       = false;
    return options;
}

void dump_tokens(const char* tag, const std::vector<ninfer::TokenId>& tokens) {
    std::cerr << tag << " [" << tokens.size() << "]:";
    for (ninfer::TokenId token : tokens) { std::cerr << ' ' << token; }
    std::cerr << '\n';
}

void dump_speculative(const char* tag, const ninfer::SpeculativeStats& stats) {
    std::cerr << tag << " rounds=" << stats.rounds << " drafted=" << stats.drafted_tokens
              << " accepted=" << stats.accepted_tokens << " fallback=" << stats.fallback_steps
              << " per_pos=";
    for (std::uint64_t count : stats.accepted_per_position) { std::cerr << ' ' << count; }
    std::cerr << '\n';
}

int report_mismatch(const char* label, const ninfer::GenerationResult& result,
                    const std::vector<ninfer::TokenId>& want) {
    const auto mismatch =
        std::mismatch(result.generated_token_ids.begin(), result.generated_token_ids.end(),
                      want.begin(), want.end());
    std::cerr << label << " diverged at "
              << static_cast<std::size_t>(mismatch.first - result.generated_token_ids.begin())
              << ": dflash="
              << (mismatch.first == result.generated_token_ids.end() ? -1 : *mismatch.first)
              << " oracle=" << (mismatch.second == want.end() ? -1 : *mismatch.second) << '\n';
    dump_tokens("  got   ", result.generated_token_ids);
    dump_tokens("  oracle", want);
    dump_speculative("  spec  ", result.speculative);
    return 1;
}

bool prefix_matches(const std::vector<ninfer::TokenId>& got,
                    const std::vector<ninfer::TokenId>& baseline, std::size_t count) {
    return got.size() == count &&
           std::equal(got.begin(), got.end(), baseline.begin(),
                      baseline.begin() + static_cast<std::ptrdiff_t>(count));
}

int check_dflash_load(const ninfer::Engine& engine) {
    const ninfer::LoadSummary load = engine.load_summary();
    if (load.target != "qwen3_8_27b" || load.weights_id != "nvfp4" ||
        load.host_to_device_bytes == 0) {
        std::cerr << "DFlash2 Engine materialized an invalid artifact payload: target="
                  << load.target << " weights=" << load.weights_id << '\n';
        return 1;
    }
    return 0;
}

int check_speculative(const ninfer::GenerationResult& result, const char* label) {
    if (result.speculative.backend != ninfer::SpeculativeBackend::DFlash ||
        result.speculative.rounds == 0) {
        std::cerr << label << " did not execute speculative decode\n";
        return 1;
    }
    return 0;
}

int greedy_oracle(ninfer::Engine& engine, const std::vector<ninfer::TokenId>& prompt,
                  std::vector<ninfer::TokenId>& oracle, const char* label) {
    const ninfer::GenerationResult seq =
        engine.generate(engine.prepare_tokens(prompt), greedy_options(24));
    if (seq.generated_token_ids.size() != 24 || check_speculative(seq, label) != 0) {
        std::cerr << label << " C=1 oracle did not generate 24 tokens\n";
        return 1;
    }
    oracle = seq.generated_token_ids;
    return 0;
}

int run_overlapping_c3(ninfer::Engine& engine,
                       const std::array<std::vector<ninfer::TokenId>, 3>& prompts,
                       const std::array<std::vector<ninfer::TokenId>, 3>& oracles,
                       const char* label) {
    constexpr std::array<std::uint32_t, 3> lengths{19, 13, 7};
    auto a = engine.submit(engine.prepare_tokens(prompts[0]), greedy_options(lengths[0]));
    auto b = engine.submit(engine.prepare_tokens(prompts[1]), greedy_options(lengths[1]));
    auto c = engine.submit(engine.prepare_tokens(prompts[2]), greedy_options(lengths[2]));
    const std::array<ninfer::GenerationResult, 3> results{a.wait(), b.wait(), c.wait()};
    const char* names[] = {"A", "B", "C"};
    int failed          = 0;
    for (std::size_t i = 0; i < 3; ++i) {
        if (!prefix_matches(results[i].generated_token_ids, oracles[i], lengths[i]) ||
            check_speculative(results[i], label) != 0) {
            std::cerr << label << " request " << names[i]
                      << " diverged from the C=1 greedy oracle\n";
            dump_tokens((std::string(label) + " oracle " + names[i]).c_str(),
                        std::vector<ninfer::TokenId>(oracles[i].begin(),
                                                     oracles[i].begin() + lengths[i]));
            dump_tokens((std::string(label) + " got    " + names[i]).c_str(),
                        results[i].generated_token_ids);
            dump_speculative((std::string(label) + " spec   " + names[i]).c_str(),
                             results[i].speculative);
            report_mismatch(label, results[i],
                            std::vector<ninfer::TokenId>(oracles[i].begin(),
                                                         oracles[i].begin() + lengths[i]));
            failed = 1;
        }
    }
    return failed;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_QWEN3_8_27B_NVFP4_DFLASH_WEIGHTS");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: NINFER_QWEN3_8_27B_NVFP4_DFLASH_WEIGHTS is not set\n";
        return 77;
    }

    // NVFP4 AllowA4 verify at T>=4 uses W4A4 attention; ordinary T=1 decode stays A16.
    // Product k=4/k=5 chain is T=5/T=6 SmallT. Overlapping Graph C=3 is checked against
    // C=1 DFlash of the same k, not against MTP k=3 (T=4).
    const std::array<std::vector<ninfer::TokenId>, 3> prompts{
        std::vector<ninfer::TokenId>{
            248045, 846,    198, 109266, 3709,  96220, 117443, 97913,
            1710,   248046, 198, 248045, 74455, 198,   248068, 198,
        },
        std::vector<ninfer::TokenId>{
            248045, 846,    198, 109266, 4120,  96220, 117443, 97913,
            1710,   248046, 198, 248045, 74455, 198,   248068, 198,
        },
        std::vector<ninfer::TokenId>{
            248045, 846,    198, 109266, 5200,  96220, 117443, 97913,
            1710,   248046, 198, 248045, 74455, 198,   248068, 198,
        },
    };

    auto run_k = [&](std::uint32_t draft_tokens, const char* label) -> int {
        ninfer::Engine engine(speculative_engine_options(
            artifact, ninfer::SpeculativeBackend::DFlash, draft_tokens, 3));
        if (const int result = check_dflash_load(engine); result != 0) { return result; }
        std::array<std::vector<ninfer::TokenId>, 3> oracles;
        for (std::size_t i = 0; i < 3; ++i) {
            if (const int result = greedy_oracle(engine, prompts[i], oracles[i], label);
                result != 0) {
                return result;
            }
        }
        if (oracles[0] == oracles[1] || oracles[0] == oracles[2] || oracles[1] == oracles[2]) {
            std::cerr << label << " C=1 oracles are not distinct across the three prompts\n";
            return 1;
        }
        if (const int result = run_overlapping_c3(engine, prompts, oracles, label); result != 0) {
            return result;
        }
        std::cout << "ok " << label << '\n' << std::flush;
        return 0;
    };

    if (const int result = run_k(4, "DFlash2 k=4 chain C=3"); result != 0) { return result; }
    if (const int result = run_k(5, "DFlash2 k=5 chain C=3"); result != 0) { return result; }
    return 0;
}
