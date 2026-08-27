#include "ninfer/engine.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

ninfer::EngineOptions mtp_engine_options(const char* artifact, std::uint32_t draft_tokens,
                                         std::uint32_t max_concurrency) {
    ninfer::EngineOptions options;
    options.artifact_path             = artifact;
    options.max_context               = 256;
    options.kv_capacity               = ninfer::KvCapacityPolicy::explicit_capacity(256);
    options.prefill_chunk             = 128;
    options.kv_cache                  = ninfer::KvCacheStorage::Nvfp4;
    options.use_cuda_graph            = true;
    options.enable_vision             = false;
    options.max_concurrency           = max_concurrency;
    options.speculative.backend       = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens  = draft_tokens;
    options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
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
              << '\n';
}

int check_load(const ninfer::Engine& engine) {
    const ninfer::LoadSummary load = engine.load_summary();
    if (load.target != "qwen3_8_27b" || load.weights_id != "nvfp4" ||
        load.host_to_device_bytes == 0) {
        std::cerr << "MTP NVFP4 Engine materialized an invalid artifact: target=" << load.target
                  << " weights=" << load.weights_id << '\n';
        return 1;
    }
    return 0;
}

int check_speculative(const ninfer::GenerationResult& result, const char* label) {
    if (result.speculative.backend != ninfer::SpeculativeBackend::Mtp ||
        result.speculative.rounds == 0) {
        std::cerr << label << " did not execute MTP decode\n";
        dump_speculative("  spec", result.speculative);
        return 1;
    }
    return 0;
}

int greedy_oracle(ninfer::Engine& engine, const std::vector<ninfer::TokenId>& prompt,
                  std::vector<ninfer::TokenId>& oracle, const char* label) {
    const ninfer::GenerationResult seq =
        engine.generate(engine.prepare_tokens(prompt), greedy_options(24));
    if (seq.generated_token_ids.size() != 24 || check_speculative(seq, label) != 0) {
        std::cerr << label << " C=1 oracle did not generate 24 MTP tokens\n";
        dump_tokens("  got", seq.generated_token_ids);
        dump_speculative("  spec", seq.speculative);
        return 1;
    }
    oracle = seq.generated_token_ids;
    return 0;
}

int run_overlapping(ninfer::Engine& engine, std::span<const std::vector<ninfer::TokenId>> prompts,
                    std::span<const std::uint32_t> lengths, const char* label) {
    std::vector<ninfer::GenerationHandle> handles;
    handles.reserve(prompts.size());
    for (std::size_t i = 0; i < prompts.size(); ++i) {
        handles.push_back(engine.submit(engine.prepare_tokens(prompts[i]), greedy_options(lengths[i])));
    }
    int failed = 0;
    for (std::size_t i = 0; i < handles.size(); ++i) {
        const ninfer::GenerationResult result = handles[i].wait();
        if (result.generated_token_ids.size() != lengths[i] ||
            check_speculative(result, label) != 0) {
            std::cerr << label << " request " << i << " did not complete MTP decode\n";
            dump_tokens("  got", result.generated_token_ids);
            dump_speculative("  spec", result.speculative);
            failed = 1;
        }
    }
    return failed;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_QWEN3_8_27B_NVFP4_MTP_WEIGHTS");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: NINFER_QWEN3_8_27B_NVFP4_MTP_WEIGHTS is not set\n";
        return 77;
    }

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
        ninfer::Engine engine(mtp_engine_options(artifact, draft_tokens, 3));
        if (const int result = check_load(engine); result != 0) { return result; }

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

        const std::array<std::uint32_t, 2> c2_lengths{19, 13};
        if (const int result =
                run_overlapping(engine, std::span(prompts.data(), 2), c2_lengths,
                                (std::string(label) + " C=2").c_str());
            result != 0) {
            return result;
        }

        const std::array<std::uint32_t, 3> c3_lengths{19, 13, 7};
        if (const int result = run_overlapping(engine, prompts, c3_lengths,
                                               (std::string(label) + " C=3").c_str());
            result != 0) {
            return result;
        }
        std::cout << "ok " << label << '\n' << std::flush;
        return 0;
    };

    if (const int result = run_k(3, "MTP NVFP4 k=3"); result != 0) { return result; }
    if (const int result = run_k(5, "MTP NVFP4 k=5"); result != 0) { return result; }
    return 0;
}
