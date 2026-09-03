#include "ninfer/engine.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <new>
#include <string>
#include <utility>
#include <vector>

namespace {

ninfer::EngineOptions base_engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path  = artifact;
    options.max_context    = 512;
    options.kv_capacity    = ninfer::KvCapacityPolicy::explicit_capacity(512);
    options.prefill_chunk  = 128;
    options.kv_cache       = ninfer::KvCacheStorage::Nvfp4;
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

ninfer::EngineOptions adaptive_engine_options(const char* artifact,
                                              ninfer::SpeculativeBackend backend,
                                              std::uint32_t draft_tokens,
                                              std::uint32_t max_concurrency) {
    ninfer::EngineOptions options =
        speculative_engine_options(artifact, backend, draft_tokens, max_concurrency);
    options.speculative.adaptive_draft = true;
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

ninfer::RequestOptions p_less_options(std::uint32_t outputs, std::uint64_t seed) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens     = outputs;
    options.execution.sampling.temperature        = 2.0F;
    options.execution.sampling.top_k              = 1;
    options.execution.sampling.top_p              = 0.1F;
    options.execution.sampling.min_p              = 0.9F;
    options.execution.sampling.presence_penalty   = 2.0F;
    options.execution.sampling.frequency_penalty  = 2.0F;
    options.execution.sampling.seed               = seed;
    options.execution.sampling.p_less             = true;
    options.execution.allow_prefix_reuse          = false;
    options.stop.include_model_defaults           = false;
    return options;
}

ninfer::RequestOptions greedy_reuse(std::uint32_t outputs, bool reuse) {
    ninfer::RequestOptions options          = greedy_options(outputs);
    options.execution.allow_prefix_reuse    = reuse;
    return options;
}

std::vector<ninfer::TokenId> resume_prefix(const std::vector<ninfer::TokenId>& keep,
                                           const std::vector<ninfer::TokenId>& generated) {
    std::vector<ninfer::TokenId> prefix = keep;
    if (!generated.empty()) {
        prefix.insert(prefix.end(), generated.begin(), generated.end() - 1);
    }
    return prefix;
}

void dump_tokens(const char* tag, const std::vector<ninfer::TokenId>& tokens) {
    std::cerr << tag << " [" << tokens.size() << "]:";
    for (ninfer::TokenId token : tokens) { std::cerr << ' ' << token; }
    std::cerr << '\n';
}

void dump_speculative(const char* tag, const ninfer::SpeculativeStats& stats) {
    std::cerr << tag << " rounds=" << stats.rounds << " drafted=" << stats.drafted_tokens
              << " accepted=" << stats.accepted_tokens << " fallback=" << stats.fallback_steps
              << " live_k=" << stats.live_draft_tokens << " per_pos=";
    for (std::uint64_t count : stats.accepted_per_position) { std::cerr << ' ' << count; }
    std::cerr << '\n';
}

int report_mismatch(const char* label, const ninfer::GenerationResult& result,
                    const std::vector<ninfer::TokenId>& want, const char* want_name) {
    const auto mismatch =
        std::mismatch(result.generated_token_ids.begin(), result.generated_token_ids.end(),
                      want.begin(), want.end());
    std::cerr << label << " diverged at "
              << static_cast<std::size_t>(mismatch.first - result.generated_token_ids.begin())
              << ": dflash="
              << (mismatch.first == result.generated_token_ids.end() ? -1 : *mismatch.first)
              << ' ' << want_name << '='
              << (mismatch.second == want.end() ? -1 : *mismatch.second) << '\n';
    dump_tokens("  got   ", result.generated_token_ids);
    std::cerr << "  " << want_name << " [" << want.size() << "]:";
    for (ninfer::TokenId token : want) { std::cerr << ' ' << token; }
    std::cerr << '\n';
    dump_speculative("  spec  ", result.speculative);
    return 1;
}

std::size_t match_prefix_length(const std::vector<ninfer::TokenId>& got,
                                const std::vector<ninfer::TokenId>& want) {
    const auto mismatch =
        std::mismatch(got.begin(), got.end(), want.begin(), want.end());
    return static_cast<std::size_t>(mismatch.first - got.begin());
}

bool relax_oracle() { return std::getenv("NINFER_DFLASH_TEST_RELAX_ORACLE") != nullptr; }

int check_speculative(const ninfer::GenerationResult& result, const char* label);

enum class OracleKind { TargetOnly, C1DFlash };

// Target-only: packed/T=1 numerical identity. First token must agree. A later greedy
// flip fails the default run; NINFER_DFLASH_TEST_RELAX_ORACLE=1 prints it and continues.
// C=1 DFlash: overlapping C>1 must match sequential C=1 of the same k. Never relaxed —
// that comparison is row isolation, not packed-versus-T=1 drift. Flattening NVFP4
// GDN conv-record to T=W*B compose (W4A4 GEMM + BF16 conv) flipped greedy col 0 vs
// C=1 fused SmallT+FP32; the Op guard is run_nvfp4_batched_matches_serial_fused.
int check_tokens(const char* label, const ninfer::GenerationResult& result,
                 const std::vector<ninfer::TokenId>& want, OracleKind kind) {
    const char* want_name = kind == OracleKind::TargetOnly ? "target-only" : "C=1 DFlash";
    if (check_speculative(result, label) != 0) { return 1; }
    if (result.generated_token_ids.size() != want.size()) {
        std::cerr << label << " generated " << result.generated_token_ids.size()
                  << " tokens, " << want_name << ' ' << want.size() << '\n';
        dump_tokens("  got   ", result.generated_token_ids);
        std::cerr << "  " << want_name << " [" << want.size() << "]:";
        for (ninfer::TokenId token : want) { std::cerr << ' ' << token; }
        std::cerr << '\n';
        return 1;
    }
    const std::size_t matched = match_prefix_length(result.generated_token_ids, want);
    if (matched < want.size()) { (void)report_mismatch(label, result, want, want_name); }
    if (matched == 0) {
        std::cerr << label << " first generated token disagrees with " << want_name << '\n';
        return 1;
    }
    if (result.speculative.drafted_tokens > 0 && result.speculative.accepted_tokens == 0) {
        std::cerr << label << " drafted tokens but accepted none\n";
        return 1;
    }
    const bool allow_relax = kind == OracleKind::TargetOnly && relax_oracle();
    if (matched != want.size() && !allow_relax) { return 1; }
    if (matched < want.size()) {
        std::cerr << label << " greedy match " << matched << '/' << want.size()
                  << " vs " << want_name
                  << " (NINFER_DFLASH_TEST_RELAX_ORACLE=1 continuing)\n";
    }
    return 0;
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

// Regression: live serve uses k=5 W=6 (chain verify). Decode materializes frontier+W while
// entitlement used to omit W for non-tree mode, so short max_tokens on a page-aligned prompt
// threw "Paged KV materialize extent is outside entitlement" and poisoned the executor.
int exercise_chain_verify_short_output_entitlement(const char* artifact) {
    ninfer::EngineOptions options =
        speculative_engine_options(artifact, ninfer::SpeculativeBackend::DFlash, 5, 1);
    options.speculative.dflash_verify_width = 6;
    options.kv_cache                        = ninfer::KvCacheStorage::Nvfp4;
    options.max_context                     = 256;
    options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(256);
    if (std::getenv("NINFER_DFLASH_TEST_NO_GRAPH") != nullptr) {
        options.use_cuda_graph = false;
    }
    ninfer::Engine engine(options);
    if (const int result = check_dflash_load(engine); result != 0) { return result; }

    // reserved_context = 60 + 5 - 1 = 64 (exact page). Without verify-width headroom,
    // first decode materialize(60+6) needs a second page and throws.
    std::vector<ninfer::TokenId> prompt(60, 198);
    prompt[0] = 248045;
    const ninfer::GenerationResult result =
        engine.generate(engine.prepare_tokens(prompt), greedy_options(5));
    if (result.finish_reason != ninfer::FinishReason::OutputLimit ||
        result.generated_token_ids.size() != 5) {
        std::cerr << "chain-verify short-output entitlement: finish="
                  << static_cast<int>(result.finish_reason)
                  << " gen=" << result.generated_token_ids.size() << '\n';
        return 1;
    }
    if (check_speculative(result, "chain-verify short-output entitlement") != 0) { return 1; }
    return 0;
}

int check_adaptive_dflash(const ninfer::GenerationResult& result, const char* label) {
    if (check_speculative(result, label) != 0) { return 1; }
    std::uint64_t hist = 0;
    for (std::uint64_t count : result.speculative.rounds_per_draft) { hist += count; }
    if (hist != result.speculative.rounds) {
        std::cerr << label << " rounds_per_draft sum " << hist << " != rounds "
                  << result.speculative.rounds << '\n';
        return 1;
    }
    const std::uint32_t live = result.speculative.live_draft_tokens;
    if (live != 3 && live != 4 && live != 5) {
        std::cerr << label << " live_draft_tokens " << live << " not in {3,4,5}\n";
        dump_speculative("  spec", result.speculative);
        return 1;
    }
    return 0;
}

int greedy_oracle(ninfer::Engine& engine, const std::vector<ninfer::TokenId>& prompt,
                  std::vector<ninfer::TokenId>& oracle, const char* label,
                  ninfer::GenerationResult* result_out = nullptr) {
    const ninfer::GenerationResult seq =
        engine.generate(engine.prepare_tokens(prompt), greedy_options(24));
    if (seq.generated_token_ids.size() != 24 || check_speculative(seq, label) != 0) {
        std::cerr << label << " C=1 oracle did not generate 24 tokens\n";
        return 1;
    }
    oracle = seq.generated_token_ids;
    if (result_out != nullptr) { *result_out = seq; }
    return 0;
}

int exercise_p_less_product_tree(ninfer::Engine& engine,
                                 const std::vector<ninfer::TokenId>& prompt) {
    constexpr const char* label = "DFlash2 p-less k=7 W=12";
    constexpr std::uint64_t seed = 0x123456789abcdef0ULL;
    const ninfer::GenerationResult first =
        engine.generate(engine.prepare_tokens(prompt), p_less_options(32, seed));
    const ninfer::GenerationResult replay =
        engine.generate(engine.prepare_tokens(prompt), p_less_options(32, seed));
    ninfer::RequestOptions top1_options = p_less_options(32, seed);
    top1_options.execution.sampling.p_less = false;
    const ninfer::GenerationResult top1 =
        engine.generate(engine.prepare_tokens(prompt), top1_options);
    if (first.finish_reason != ninfer::FinishReason::OutputLimit ||
        replay.finish_reason != ninfer::FinishReason::OutputLimit ||
        top1.finish_reason != ninfer::FinishReason::OutputLimit ||
        first.generated_token_ids.size() != 32 || replay.generated_token_ids.size() != 32 ||
        top1.generated_token_ids.size() != 32) {
        std::cerr << label << " did not complete seeded p-less/replay/top-1 requests\n";
        dump_tokens("  first", first.generated_token_ids);
        dump_tokens("  replay", replay.generated_token_ids);
        dump_tokens("  top-1", top1.generated_token_ids);
        return 1;
    }
    if (check_speculative(first, label) != 0 || check_speculative(replay, label) != 0 ||
        check_speculative(top1, label) != 0) {
        return 1;
    }
    if (first.speculative.live_draft_tokens != 7 || first.speculative.drafted_tokens == 0 ||
        first.speculative.accepted_tokens == 0) {
        std::cerr << label << " did not execute and accept from the native draft tree\n";
        dump_speculative("  spec", first.speculative);
        return 1;
    }
    if (first.generated_token_ids != replay.generated_token_ids ||
        first.speculative.rounds != replay.speculative.rounds ||
        first.speculative.drafted_tokens != replay.speculative.drafted_tokens ||
        first.speculative.accepted_tokens != replay.speculative.accepted_tokens) {
        std::cerr << label << " changed under identical seeded CUDA-graph replay\n";
        dump_tokens("  first", first.generated_token_ids);
        dump_tokens("  replay", replay.generated_token_ids);
        dump_speculative("  first", first.speculative);
        dump_speculative("  replay", replay.speculative);
        return 1;
    }
    if (first.generated_token_ids == top1.generated_token_ids) {
        std::cerr << label << " produced the top-1 control sequence; p-less was not observed\n";
        dump_tokens("  p-less", first.generated_token_ids);
        dump_tokens("  top-1", top1.generated_token_ids);
        return 1;
    }
    std::cout << "ok " << label << '\n' << std::flush;
    return 0;
}

int exercise_p_less_target_likelihood(const char* artifact,
                                      const std::vector<ninfer::TokenId>& prompt) {
    constexpr const char* label = "DFlash2 adaptive p-less target likelihood";
    constexpr std::uint64_t seed = 15446143373561885318ULL;
    // This short seeded route is a fast guard for the broad 3,000-token verifier-drift
    // diagnosis; target-only teacher forcing is the independent distribution oracle.
    std::vector<ninfer::TokenId> corpus = prompt;
    {
        ninfer::Engine engine(adaptive_engine_options(
            artifact, ninfer::SpeculativeBackend::DFlash, 7, 1));
        const ninfer::GenerationResult generated =
            engine.generate(engine.prepare_tokens(prompt), p_less_options(64, seed));
        if (generated.generated_token_ids.size() != 64 ||
            check_adaptive_dflash(generated, label) != 0) {
            std::cerr << label << " generation did not complete\n";
            return 1;
        }
        corpus.insert(corpus.end(), generated.generated_token_ids.begin(),
                      generated.generated_token_ids.end());
    }

    ninfer::Engine baseline(base_engine_options(artifact));
    ninfer::ScoreOptions score_options;
    score_options.schedule = ninfer::ScoreSchedule::Decode;
    score_options.skip_tokens = static_cast<std::uint32_t>(prompt.size() - 1);
    const ninfer::ScoreResult score =
        baseline.score(baseline.prepare_tokens(std::move(corpus), false), score_options);
    if (score.tokens_scored != 64 || score.non_finite != 0 || score.mean_nll > 0.5 ||
        score.max_nll > 4.0) {
        std::cerr << label << " drifted from the ordinary target distribution: scored="
                  << score.tokens_scored << " non_finite=" << score.non_finite
                  << " mean_nll=" << score.mean_nll << " max_nll=" << score.max_nll << '\n';
        return 1;
    }
    std::cout << "ok " << label << " mean_nll=" << score.mean_nll
              << " max_nll=" << score.max_nll << '\n';
    return 0;
}

std::vector<ninfer::TokenId> token_prefix(const std::vector<ninfer::TokenId>& tokens,
                                          std::uint32_t length, const char* label) {
    if (tokens.size() < length) {
        std::cerr << label << " oracle has " << tokens.size() << " tokens, need " << length
                  << '\n';
        return {};
    }
    return std::vector<ninfer::TokenId>(tokens.begin(), tokens.begin() + length);
}

int run_overlapping_c3(ninfer::Engine& engine,
                       const std::array<std::vector<ninfer::TokenId>, 3>& prompts,
                       const std::array<std::vector<ninfer::TokenId>, 3>& dflash_oracles,
                       const std::array<std::vector<ninfer::TokenId>, 3>& target_oracles,
                       const char* label) {
    // Per-row isolation: each overlapping request must match sequential C=1 DFlash of the
    // same k. Target-only is a second oracle; a packed/T=1 greedy flip is not row mixing.
    constexpr std::array<std::uint32_t, 3> lengths{19, 13, 7};
    auto a = engine.submit(engine.prepare_tokens(prompts[0]), greedy_options(lengths[0]));
    auto b = engine.submit(engine.prepare_tokens(prompts[1]), greedy_options(lengths[1]));
    auto c = engine.submit(engine.prepare_tokens(prompts[2]), greedy_options(lengths[2]));
    std::array<ninfer::GenerationResult, 3> results;
    try {
        results = {a.wait(), b.wait(), c.wait()};
    } catch (const std::exception& error) {
        std::cerr << label << " concurrent speculative verify threw: " << error.what() << '\n';
        return 1;
    }
    const char* names[] = {"A", "B", "C"};
    int failed          = 0;
    for (std::size_t i = 0; i < 3; ++i) {
        const std::string request = std::string(label) + " request " + names[i];
        const std::string dflash_label = request + " C=1 DFlash";
        const std::string target_label = request + " target-only";
        const std::vector<ninfer::TokenId> dflash_want =
            token_prefix(dflash_oracles[i], lengths[i], dflash_label.c_str());
        const std::vector<ninfer::TokenId> target_want =
            token_prefix(target_oracles[i], lengths[i], target_label.c_str());
        if (dflash_want.empty() || target_want.empty()) { return 1; }
        const std::string vs_dflash = request + " vs C=1 DFlash";
        const std::string vs_target = request + " vs target-only";
        if (check_tokens(vs_dflash.c_str(), results[i], dflash_want, OracleKind::C1DFlash) != 0) {
            failed = 1;
        }
        if (check_tokens(vs_target.c_str(), results[i], target_want, OracleKind::TargetOnly) != 0) {
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

    if (std::getenv("NINFER_DFLASH_TEST_SKIP_ENTITLEMENT") == nullptr) {
        std::cerr << "dflash_real: chain-verify short-output Main KV entitlement\n";
        if (const int result = exercise_chain_verify_short_output_entitlement(artifact);
            result != 0) {
            return result;
        }
    }

    // Chain W=k+1 uses packed SmallT / W4A4 at T=W. Ordinary decode stays T=1 GEMV.
    // C=1 vs target-only can flip a later greedy token (k=4 prompt 0 token 21).
    // C>1 must still match saved C=1 DFlash of the same k (row isolation).
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

    if (std::getenv("NINFER_DFLASH_TEST_SKIP_LIKELIHOOD") == nullptr) {
        if (const int result = exercise_p_less_target_likelihood(artifact, prompts[0]);
            result != 0) {
            return result;
        }
    }

    auto run_k = [&](std::uint32_t draft_tokens, const char* label) -> int {
        std::array<std::vector<ninfer::TokenId>, 3> target_oracles;
        const char* only_prompt = std::getenv("NINFER_DFLASH_TEST_ONLY_PROMPT");
        {
            // A DFlash self-comparison can miss a verifier that consistently commits the
            // wrong token. Compare against the same artifact with speculative decoding disabled.
            ninfer::Engine baseline(base_engine_options(artifact));
            for (std::size_t i = 0; i < prompts.size(); ++i) {
                if (only_prompt != nullptr && std::string(only_prompt) != std::to_string(i)) {
                    continue;
                }
                const ninfer::GenerationResult result =
                    baseline.generate(baseline.prepare_tokens(prompts[i]), greedy_options(24));
                if (result.generated_token_ids.size() != 24) {
                    std::cerr << label << " baseline did not generate 24 tokens\n";
                    return 1;
                }
                target_oracles[i] = result.generated_token_ids;
            }
        }

        ninfer::EngineOptions dflash_options =
            speculative_engine_options(artifact, ninfer::SpeculativeBackend::DFlash, draft_tokens, 3);
        if (std::getenv("NINFER_DFLASH_TEST_MAX1") != nullptr) {
            dflash_options.max_concurrency = 1;
        }
        if (std::getenv("NINFER_DFLASH_TEST_NO_GRAPH") != nullptr) {
            dflash_options.use_cuda_graph = false;
        }
        if (std::getenv("NINFER_DFLASH_TEST_FULL_HEAD") != nullptr) {
            dflash_options.speculative.proposal_head = ninfer::ProposalHead::Full;
        }
        if (std::getenv("NINFER_DFLASH_TEST_CHAIN") != nullptr) {
            dflash_options.speculative.dflash_verify_width = draft_tokens + 1;
        }
        if (const char* verify_width = std::getenv("NINFER_DFLASH_TEST_VERIFY_WIDTH")) {
            dflash_options.speculative.dflash_verify_width =
                static_cast<std::uint32_t>(std::atoi(verify_width));
        }
        try {
            ninfer::Engine engine(dflash_options);
            if (const int result = check_dflash_load(engine); result != 0) { return result; }
            std::array<std::vector<ninfer::TokenId>, 3> dflash_oracles;
            int failed = 0;
            for (std::size_t i = 0; i < 3; ++i) {
                if (only_prompt != nullptr &&
                    std::string(only_prompt) != std::to_string(i)) {
                    continue;
                }
                ninfer::GenerationResult dflash_result;
                if (const int result = greedy_oracle(engine, prompts[i], dflash_oracles[i], label,
                                                     &dflash_result);
                    result != 0) {
                    return result;
                }
                // Packed/T=1 identity. Do not return here: C>1 isolation still uses
                // the saved C=1 DFlash tokens even when a later greedy token flips.
                const std::string vs_target =
                    std::string(label) + " prompt " + std::to_string(i) + " vs target-only";
                if (check_tokens(vs_target.c_str(), dflash_result, target_oracles[i],
                                 OracleKind::TargetOnly) != 0) {
                    failed = 1;
                }
            }
            if (only_prompt == nullptr && (target_oracles[0] == target_oracles[1] ||
                                           target_oracles[0] == target_oracles[2] ||
                                           target_oracles[1] == target_oracles[2])) {
                std::cerr << label
                          << " target-only oracles are not distinct across the three prompts\n";
                return 1;
            }
            if (only_prompt == nullptr && (dflash_oracles[0] == dflash_oracles[1] ||
                                           dflash_oracles[0] == dflash_oracles[2] ||
                                           dflash_oracles[1] == dflash_oracles[2])) {
                std::cerr << label
                          << " C=1 DFlash oracles are not distinct across the three prompts\n";
                return 1;
            }
            if (only_prompt == nullptr && dflash_options.max_concurrency >= 3 &&
                run_overlapping_c3(engine, prompts, dflash_oracles, target_oracles, label) != 0) {
                failed = 1;
            }
            if (draft_tokens == 7 && dflash_options.speculative.dflash_verify_width == 0 &&
                (only_prompt == nullptr || std::string(only_prompt) == "0") &&
                exercise_p_less_product_tree(engine, prompts[0]) != 0) {
                failed = 1;
            }
            if (failed != 0) { return failed; }
        } catch (const std::bad_alloc&) {
            std::cerr << label << " engine std::bad_alloc (k=" << draft_tokens
                      << " max_conc=" << dflash_options.max_concurrency
                      << " graph=" << dflash_options.use_cuda_graph << ")\n";
            return 1;
        }
        std::cout << "ok " << label << '\n' << std::flush;
        return 0;
    };

    const char* only_k = std::getenv("NINFER_DFLASH_TEST_ONLY_K");
    if (only_k == nullptr || std::string(only_k) == "7") {
        std::string label = "DFlash2 k=7 W=12 tree C=3";
        if (std::getenv("NINFER_DFLASH_TEST_CHAIN") != nullptr) {
            label = "DFlash2 k=7 W=8 chain C=3";
        } else if (const char* width = std::getenv("NINFER_DFLASH_TEST_VERIFY_WIDTH")) {
            label = "DFlash2 k=7 W=" + std::string(width) + " override C=3";
        }
        if (const int result = run_k(7, label.c_str()); result != 0) { return result; }
    }
    if (only_k == nullptr || std::string(only_k) == "1") {
        if (const int result = run_k(1, "DFlash2 k=1 chain C=3"); result != 0) { return result; }
    }
    if (only_k == nullptr || std::string(only_k) == "4") {
        if (const int result = run_k(4, "DFlash2 k=4 chain C=3"); result != 0) { return result; }
    }
    if (only_k == nullptr || std::string(only_k) == "5") {
        if (const int result = run_k(5, "DFlash2 k=5 chain C=3"); result != 0) { return result; }
    }

    {
        const char* label = "DFlash2 adaptive N=7 {3,4,5}";
        ninfer::Engine engine(adaptive_engine_options(
            artifact, ninfer::SpeculativeBackend::DFlash, 7, 3));
        if (const int result = check_dflash_load(engine); result != 0) { return result; }
        // Prefill commits the first generated token. Seven tokens leave remaining=6 on
        // the first DFlash round: budget_extent=5, seed live_k=5, W(k)=6 under W_ceil=6.
        const ninfer::GenerationResult compact =
            engine.generate(engine.prepare_tokens(prompts[0]), greedy_options(7));
        if (compact.generated_token_ids.size() != 7 || check_speculative(compact, label) != 0) {
            std::cerr << label << " compact k=5 under N=7 did not complete\n";
            dump_tokens("  got", compact.generated_token_ids);
            dump_speculative("  spec", compact.speculative);
            return 1;
        }
        if (compact.speculative.rounds_per_draft.size() < 6 ||
            compact.speculative.rounds_per_draft[5] == 0) {
            std::cerr << label << " compact run did not record k=5 rounds under N=7\n";
            dump_speculative("  spec", compact.speculative);
            return 1;
        }
        const std::uint64_t decode_before = engine.runtime_stats().committed_decode_tokens;
        auto compact_a = engine.submit(engine.prepare_tokens(prompts[0]), greedy_options(7));
        auto compact_b = engine.submit(engine.prepare_tokens(prompts[1]), greedy_options(7));
        const ninfer::GenerationResult compact_ra = compact_a.wait();
        const ninfer::GenerationResult compact_rb = compact_b.wait();
        if (compact_ra.generated_token_ids.size() != 7 ||
            compact_rb.generated_token_ids.size() != 7 ||
            check_speculative(compact_ra, label) != 0 ||
            check_speculative(compact_rb, label) != 0 ||
            compact_ra.speculative.rounds_per_draft.size() < 6 ||
            compact_ra.speculative.rounds_per_draft[5] == 0) {
            std::cerr << label << " compact C=2 k=5 under N=7 failed\n";
            dump_speculative("  A", compact_ra.speculative);
            dump_speculative("  B", compact_rb.speculative);
            return 1;
        }
        // Each seven-token request commits its first token in prefill, then six tokens in decode.
        // A terminal wait must not become observable before those decode counters are published.
        if (engine.runtime_stats().committed_decode_tokens < decode_before + 12) {
            std::cerr << label << " terminal waits preceded committed decode-stat publication\n";
            return 1;
        }
        const ninfer::GenerationResult seq =
            engine.generate(engine.prepare_tokens(prompts[0]), greedy_options(24));
        if (seq.generated_token_ids.size() != 24 || check_speculative(seq, label) != 0) {
            std::cerr << label << " C=1 did not complete\n";
            dump_tokens("  got", seq.generated_token_ids);
            dump_speculative("  spec", seq.speculative);
            return 1;
        }
        std::uint64_t hist = 0;
        for (std::uint64_t count : seq.speculative.rounds_per_draft) { hist += count; }
        if (hist != seq.speculative.rounds) {
            std::cerr << label << " rounds_per_draft sum mismatch\n";
            return 1;
        }
        const std::uint32_t live = seq.speculative.live_draft_tokens;
        if (live != 3 && live != 4 && live != 5) {
            std::cerr << label << " live_draft_tokens " << live << " not in {3,4,5}\n";
            return 1;
        }
        auto a = engine.submit(engine.prepare_tokens(prompts[0]), greedy_options(19));
        auto b = engine.submit(engine.prepare_tokens(prompts[1]), greedy_options(13));
        auto c = engine.submit(engine.prepare_tokens(prompts[2]), greedy_options(7));
        const std::array<ninfer::GenerationResult, 3> results{a.wait(), b.wait(), c.wait()};
        const std::array<std::uint32_t, 3> lengths{19, 13, 7};
        for (std::size_t i = 0; i < 3; ++i) {
            if (results[i].generated_token_ids.size() != lengths[i] ||
                check_speculative(results[i], label) != 0) {
                std::cerr << label << " C=3 request " << i << " failed\n";
                dump_tokens("  got", results[i].generated_token_ids);
                dump_speculative("  spec", results[i].speculative);
                return 1;
            }
        }
        std::cout << "ok " << label << '\n' << std::flush;
    }

    {
        const char* label = "DFlash2 terminal delivery and queue telemetry";
        ninfer::Engine engine(adaptive_engine_options(
            artifact, ninfer::SpeculativeBackend::DFlash, 7, 1));
        if (const int result = check_dflash_load(engine); result != 0) { return result; }

        struct DiscardingSink final : ninfer::OutputSink {
            void publish(ninfer::OutputDelta) override {}
        } sink;
        auto terminal = engine.submit(engine.prepare_tokens(prompts[0]), greedy_options(7));
        bool rejected = false;
        try {
            (void)terminal.wait(&sink);
        } catch (const std::logic_error&) {
            rejected = true;
        }
        if (!rejected) {
            std::cerr << label << " accepted a sink for terminal-only delivery\n";
            return 1;
        }
        if (terminal.wait().generated_token_ids.size() != 7) {
            std::cerr << label << " terminal request did not complete after sink rejection\n";
            return 1;
        }

        auto active = engine.submit(engine.prepare_tokens(prompts[0]), greedy_options(16));
        auto queued = engine.submit(engine.prepare_tokens(prompts[1]), greedy_options(7));
        if (engine.runtime_stats().waiting_requests == 0) {
            std::cerr << label << " full engine did not publish queued depth at ingress\n";
            return 1;
        }
        if (active.wait().generated_token_ids.size() != 16 ||
            queued.wait().generated_token_ids.size() != 7) {
            std::cerr << label << " queued requests did not complete\n";
            return 1;
        }
        std::cout << "ok " << label << '\n' << std::flush;
    }

    {
        const char* label = "DFlash2 adaptive RAM reseed {3,4,5}";
        ninfer::EngineOptions options =
            adaptive_engine_options(artifact, ninfer::SpeculativeBackend::DFlash, 7, 1);
        options.kv_ram_capacity_bytes = 1024ULL * 1024ULL * 1024ULL;
        ninfer::Engine engine(options);
        if (const int result = check_dflash_load(engine); result != 0) { return result; }
        if (engine.memory_summary().kv_ram_capacity_bytes == 0) {
            std::cerr << label << " RAM tier is disabled\n";
            return 1;
        }
        const ninfer::GenerationResult first =
            engine.generate(engine.prepare_tokens(prompts[0]), greedy_reuse(8, false));
        if (first.generated_token_ids.size() != 8 ||
            check_adaptive_dflash(first, label) != 0 ||
            first.speculative.live_draft_tokens != 5) {
            std::cerr << label << " source did not stay at seeded k=5\n";
            dump_speculative("  spec", first.speculative);
            return 1;
        }
        const auto captures_before = engine.runtime_stats().kv_ram_captures;
        const ninfer::GenerationResult evictor =
            engine.generate(engine.prepare_tokens(prompts[1]), greedy_reuse(4, false));
        if (evictor.generated_token_ids.size() != 4 ||
            engine.runtime_stats().kv_ram_captures <= captures_before) {
            std::cerr << label << " evictor did not capture the first chat to RAM\n";
            return 1;
        }
        const std::vector<ninfer::TokenId> history =
            resume_prefix(prompts[0], first.generated_token_ids);
        const auto restores_before = engine.runtime_stats().kv_ram_restores;
        const ninfer::GenerationResult hit =
            engine.generate(engine.prepare_tokens(history), greedy_reuse(4, true));
        if (hit.prefix_reuse_source != ninfer::PrefixReuseSource::HostRam) {
            std::cerr << label << " restore source is "
                      << static_cast<int>(hit.prefix_reuse_source) << ", expected HostRam\n";
            return 1;
        }
        if (engine.runtime_stats().kv_ram_restores != restores_before + 1) {
            std::cerr << label << " kv_ram_restores did not increment\n";
            return 1;
        }
        if (hit.generated_token_ids.size() != 4 || check_adaptive_dflash(hit, label) != 0 ||
            hit.speculative.live_draft_tokens != 5) {
            std::cerr << label << " RAM restore leaked live_k off the seeded attractor\n";
            dump_speculative("  hit", hit.speculative);
            return 1;
        }
        std::cout << "ok " << label << '\n' << std::flush;
    }

    {
        const char* label = "DFlash2 adaptive RAM restore in flight {3,4,5}";
        ninfer::EngineOptions options =
            adaptive_engine_options(artifact, ninfer::SpeculativeBackend::DFlash, 7, 2);
        options.kv_ram_capacity_bytes = 1024ULL * 1024ULL * 1024ULL;
        ninfer::Engine engine(options);
        if (const int result = check_dflash_load(engine); result != 0) { return result; }
        const ninfer::GenerationResult first =
            engine.generate(engine.prepare_tokens(prompts[0]), greedy_reuse(8, false));
        const ninfer::GenerationResult second =
            engine.generate(engine.prepare_tokens(prompts[1]), greedy_reuse(4, false));
        if (first.generated_token_ids.size() != 8 || second.generated_token_ids.size() != 4) {
            std::cerr << label << " C=2 fill did not complete\n";
            return 1;
        }
        const auto captures_before = engine.runtime_stats().kv_ram_captures;
        const ninfer::GenerationResult evictor =
            engine.generate(engine.prepare_tokens(prompts[2]), greedy_reuse(4, false));
        if (evictor.generated_token_ids.size() != 4 ||
            engine.runtime_stats().kv_ram_captures <= captures_before) {
            std::cerr << label << " third request did not dump a chat to RAM\n";
            return 1;
        }
        const std::vector<ninfer::TokenId> history =
            resume_prefix(prompts[0], first.generated_token_ids);
        // Restored exact-prefix generate overlaps a compact k=5 decode (greedy 7 →
        // remaining=6 after prefill, budget_extent=5, W(k)=6 under W_ceil=6) on the
        // other lane while copy_stream H2D-restores cyclic DFlash KV / GDN / pages.
        auto restored_h =
            engine.submit(engine.prepare_tokens(history), greedy_reuse(4, true));
        auto inflight_h =
            engine.submit(engine.prepare_tokens(prompts[1]), greedy_reuse(7, false));
        const ninfer::GenerationResult hit      = restored_h.wait();
        const ninfer::GenerationResult inflight = inflight_h.wait();
        if (hit.generated_token_ids.size() != 4 || inflight.generated_token_ids.size() != 7 ||
            check_adaptive_dflash(hit, label) != 0 ||
            check_adaptive_dflash(inflight, label) != 0 ||
            hit.speculative.live_draft_tokens != 5) {
            std::cerr << label << " overlapping RAM restore dropped live_k or failed compact\n";
            dump_speculative("  hit", hit.speculative);
            dump_speculative("  inflight", inflight.speculative);
            return 1;
        }
        if (inflight.speculative.rounds_per_draft.size() < 6 ||
            inflight.speculative.rounds_per_draft[5] == 0) {
            std::cerr << label << " inflight compact did not record k=5 rounds under N=7\n";
            dump_speculative("  inflight", inflight.speculative);
            return 1;
        }
        if (hit.prefix_reuse_source != ninfer::PrefixReuseSource::HostRam &&
            hit.prefix_reuse_source != ninfer::PrefixReuseSource::VramResident) {
            std::cerr << label << " restore source is "
                      << static_cast<int>(hit.prefix_reuse_source) << '\n';
            return 1;
        }
        std::cout << "ok " << label << '\n' << std::flush;
    }
    return 0;
}
