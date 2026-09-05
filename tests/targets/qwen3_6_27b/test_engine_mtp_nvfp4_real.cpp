#include "ninfer/engine.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <optional>
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

ninfer::EngineOptions mtp_adaptive_options(const char* artifact, std::uint32_t draft_tokens,
                                           std::uint32_t max_concurrency) {
    ninfer::EngineOptions options = mtp_engine_options(artifact, draft_tokens, max_concurrency);
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

ninfer::RequestOptions greedy_reuse(std::uint32_t outputs, bool reuse) {
    ninfer::RequestOptions options = greedy_options(outputs);
    options.execution.allow_prefix_reuse = reuse;
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
              << " live_k=" << stats.live_draft_tokens << '\n';
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

int check_adaptive_stats(const ninfer::GenerationResult& result, std::uint32_t ceiling,
                         std::initializer_list<std::uint32_t> captured, const char* label) {
    if (check_speculative(result, label) != 0) { return 1; }
    std::uint64_t hist = 0;
    for (std::uint64_t count : result.speculative.rounds_per_draft) { hist += count; }
    if (hist != result.speculative.rounds) {
        std::cerr << label << " rounds_per_draft sum " << hist << " != rounds "
                  << result.speculative.rounds << '\n';
        return 1;
    }
    const std::uint32_t live = result.speculative.live_draft_tokens;
    bool in_set              = false;
    for (std::uint32_t k : captured) {
        if (k == live) { in_set = true; }
    }
    if (!in_set || live > ceiling) {
        std::cerr << label << " live_draft_tokens " << live << " is outside the captured set\n";
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

int check_k4_b4_isolation(ninfer::Engine& engine,
                          std::span<const std::vector<ninfer::TokenId>> seeds,
                          const char* label) {
    std::array<std::vector<ninfer::TokenId>, 5> prompts{
        seeds[0], seeds[1], seeds[2], seeds[0], seeds[1]};
    constexpr std::array<ninfer::TokenId, 5> fills{3709, 4120, 5200, 96220, 74455};
    for (std::size_t i = 0; i < prompts.size(); ++i) { prompts[i].resize(896, fills[i]); }

    const auto run_wave = [&](const std::array<std::size_t, 4>& order, const char* wave)
        -> std::optional<std::array<std::vector<ninfer::TokenId>, 5>> {
        std::array<decltype(engine.prepare_tokens(prompts[0])), 4> prepared{
            engine.prepare_tokens(prompts[order[0]]), engine.prepare_tokens(prompts[order[1]]),
            engine.prepare_tokens(prompts[order[2]]), engine.prepare_tokens(prompts[order[3]])};
        const ninfer::RuntimeStats before = engine.runtime_stats();
        std::array<ninfer::GenerationHandle, 4> handles{
            engine.submit(std::move(prepared[0]), greedy_options(24)),
            engine.submit(std::move(prepared[1]), greedy_options(24)),
            engine.submit(std::move(prepared[2]), greedy_options(24)),
            engine.submit(std::move(prepared[3]), greedy_options(24))};

        std::array<std::vector<ninfer::TokenId>, 5> outputs;
        for (std::size_t row = 0; row < handles.size(); ++row) {
            const ninfer::GenerationResult result = handles[row].wait();
            if (result.generated_token_ids.size() != 24 || check_speculative(result, label) != 0) {
                std::cerr << label << ' ' << wave << " row " << row
                          << " did not complete MTP decode\n";
                dump_tokens("  got", result.generated_token_ids);
                dump_speculative("  spec", result.speculative);
                return std::nullopt;
            }
            outputs[order[row]] = result.generated_token_ids;
        }
        (void)engine.memory_summary();
        const ninfer::RuntimeStats after = engine.runtime_stats();
        const std::uint64_t rounds       = after.decode_rounds - before.decode_rounds;
        const std::uint64_t rows         = after.decode_row_rounds - before.decode_row_rounds;
        if (rounds == 0 || rows <= 3 * rounds) {
            std::cerr << label << ' ' << wave << " did not exercise a B=4 decode round: rounds="
                      << rounds << " rows=" << rows << '\n';
            return std::nullopt;
        }
        return outputs;
    };

    const auto base = run_wave({0, 1, 2, 3}, "base");
    const auto permuted = run_wave({3, 2, 0, 1}, "permuted");
    const auto partner = run_wave({0, 1, 2, 4}, "alternate-partner");
    if (!base || !permuted || !partner) { return 1; }
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = i + 1; j < 4; ++j) {
            if ((*base)[i] == (*base)[j]) {
                std::cerr << label << " base prompts " << i << " and " << j
                          << " did not produce distinguishable outputs\n";
                return 1;
            }
        }
    }
    for (std::size_t i = 0; i < 3; ++i) {
        if ((*partner)[i] == (*partner)[4]) {
            std::cerr << label << " alternate partner did not produce a distinguishable output\n";
            return 1;
        }
    }
    if (*base != *permuted || (*base)[0] != (*partner)[0] || (*base)[1] != (*partner)[1] ||
        (*base)[2] != (*partner)[2]) {
        std::cerr << label
                  << " changed a request output after a B=4 row permutation or partner change\n";
        return 1;
    }
    return 0;
}

int check_prefill_first_batch(ninfer::Engine& engine,
                              std::span<const std::vector<ninfer::TokenId>> seeds,
                              const char* label) {
    std::vector<ninfer::TokenId> prompt_a = seeds[0];
    std::vector<ninfer::TokenId> prompt_b = seeds[1];
    std::vector<ninfer::TokenId> prompt_c = seeds[2];
    prompt_a.resize(896, 3709);
    prompt_b.resize(896, 4120);
    prompt_c.resize(896, 5200);

    const auto run_pair = [&](const std::vector<ninfer::TokenId>& first,
                              const std::vector<ninfer::TokenId>& second,
                              const char* wave)
        -> std::optional<std::array<std::vector<ninfer::TokenId>, 2>> {
        auto prepared_first  = engine.prepare_tokens(first);
        auto prepared_second = engine.prepare_tokens(second);
        const ninfer::RuntimeStats before = engine.runtime_stats();
        auto handle_first = engine.submit(std::move(prepared_first), greedy_options(2));
        auto handle_second = engine.submit(std::move(prepared_second), greedy_options(2));
        const ninfer::GenerationResult result_first = handle_first.wait();
        const ninfer::GenerationResult result_second = handle_second.wait();
        (void)engine.memory_summary(); // Fence the worker's counter publication at this boundary.
        const ninfer::RuntimeStats after = engine.runtime_stats();

        if (result_first.generated_token_ids.size() != 2 ||
            result_second.generated_token_ids.size() != 2) {
            std::cerr << label << ' ' << wave
                      << " prefill-first pair did not generate two tokens per lane\n";
            return std::nullopt;
        }
        const std::uint64_t rounds = after.decode_rounds - before.decode_rounds;
        const std::uint64_t rows   = after.decode_row_rounds - before.decode_row_rounds;
        if (rounds != 1 || rows != 2) {
            std::cerr << label << ' ' << wave << " prefill-first pair ran " << rounds
                      << " decode rounds / " << rows
                      << " row rounds, expected one maximal B=2 decode\n";
            return std::nullopt;
        }
        return std::array<std::vector<ninfer::TokenId>, 2>{result_first.generated_token_ids,
                                                           result_second.generated_token_ids};
    };

    const auto ab = run_pair(prompt_a, prompt_b, "A/B");
    const auto ac = run_pair(prompt_a, prompt_c, "A/C");
    const auto bc = run_pair(prompt_b, prompt_c, "B/C");
    if (!ab || !ac || !bc) { return 1; }
    if ((*ab)[0] == (*ab)[1] || (*ab)[0] != (*ac)[0] || (*ab)[1] != (*bc)[0] ||
        (*ac)[1] != (*bc)[1]) {
        std::cerr << label
                  << " exact-B=2 partner waves did not preserve per-request output ownership\n";
        return 1;
    }
    return 0;
}

int check_terminal_prefill_debt(ninfer::Engine& engine,
                                std::span<const std::vector<ninfer::TokenId>> seeds,
                                const char* label) {
    std::array<std::vector<ninfer::TokenId>, 4> prompts{seeds[0], seeds[1], seeds[2], seeds[1]};
    constexpr std::array<ninfer::TokenId, 4> fills{3709, 4120, 5200, 96220};
    for (std::size_t i = 0; i < prompts.size(); ++i) { prompts[i].resize(896, fills[i]); }

    std::array<decltype(engine.prepare_tokens(prompts[0])), 4> prepared{
        engine.prepare_tokens(prompts[0]), engine.prepare_tokens(prompts[1]),
        engine.prepare_tokens(prompts[2]), engine.prepare_tokens(prompts[3])};
    auto donor = engine.submit(std::move(prepared[0]), greedy_options(2));
    auto terminal_b = engine.submit(std::move(prepared[1]), greedy_options(1));
    auto terminal_c = engine.submit(std::move(prepared[2]), greedy_options(1));
    auto terminal_d = engine.submit(std::move(prepared[3]), greedy_options(1));

    const ninfer::GenerationResult donor_result = donor.wait();
    const ninfer::GenerationResult result_b = terminal_b.wait();
    const ninfer::GenerationResult result_c = terminal_c.wait();
    const ninfer::GenerationResult result_d = terminal_d.wait();
    (void)engine.memory_summary();

    if (result_b.generated_token_ids.size() != 1 || result_c.generated_token_ids.size() != 1 ||
        result_d.generated_token_ids.size() != 1 ||
        donor_result.generated_token_ids.size() != 2) {
        std::cerr << label
                  << " terminal-prefill burst did not preserve donor and terminal completions\n";
        return 1;
    }
    return 0;
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

    {
        const char* label = "MTP NVFP4 k=4 B=4 isolation";
        ninfer::EngineOptions options = mtp_engine_options(artifact, 4, 4);
        options.max_context = 1024;
        options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(
            options.max_context * options.max_concurrency);
        ninfer::Engine engine(options);
        if (const int result = check_load(engine); result != 0) { return result; }
        if (const int result = check_k4_b4_isolation(engine, prompts, label); result != 0) {
            return result;
        }
        std::cout << "ok " << label << '\n' << std::flush;
    }

    {
        const char* label = "MTP NVFP4 prefill-first scheduling";
        ninfer::EngineOptions options = mtp_engine_options(artifact, 3, 3);
        options.max_context = 1024;
        options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(
            options.max_context * options.max_concurrency);
        ninfer::Engine engine(options);
        if (const int result = check_load(engine); result != 0) { return result; }
        if (const int result = check_prefill_first_batch(engine, prompts, label); result != 0) {
            return result;
        }
        if (const int result = check_terminal_prefill_debt(engine, prompts, label); result != 0) {
            return result;
        }
        std::cout << "ok " << label << '\n' << std::flush;
    }

    {
        const char* label = "MTP NVFP4 adaptive N=5";
        ninfer::Engine engine(mtp_adaptive_options(artifact, 5, 3));
        if (const int result = check_load(engine); result != 0) { return result; }
        const ninfer::GenerationResult seq =
            engine.generate(engine.prepare_tokens(prompts[0]), greedy_options(24));
        if (seq.generated_token_ids.size() != 24 ||
            check_adaptive_stats(seq, 5, {3, 4, 5}, label) != 0) {
            std::cerr << label << " did not complete adaptive MTP decode\n";
            dump_tokens("  got", seq.generated_token_ids);
            dump_speculative("  spec", seq.speculative);
            return 1;
        }
        const std::array<std::uint32_t, 3> c3_lengths{19, 13, 7};
        std::vector<ninfer::GenerationHandle> handles;
        for (std::size_t i = 0; i < 3; ++i) {
            handles.push_back(
                engine.submit(engine.prepare_tokens(prompts[i]), greedy_options(c3_lengths[i])));
        }
        for (std::size_t i = 0; i < handles.size(); ++i) {
            const ninfer::GenerationResult result = handles[i].wait();
            if (result.generated_token_ids.size() != c3_lengths[i] ||
                check_adaptive_stats(result, 5, {3, 4, 5}, label) != 0) {
                std::cerr << label << " C=3 request " << i << " failed\n";
                dump_tokens("  got", result.generated_token_ids);
                dump_speculative("  spec", result.speculative);
                return 1;
            }
        }
        std::cout << "ok " << label << '\n' << std::flush;
    }

    {
        const char* label = "MTP NVFP4 adaptive compact k=4 C=2";
        ninfer::Engine engine(mtp_adaptive_options(artifact, 5, 2));
        if (const int result = check_load(engine); result != 0) { return result; }
        auto ha = engine.submit(engine.prepare_tokens(prompts[0]), greedy_options(6));
        auto hb = engine.submit(engine.prepare_tokens(prompts[1]), greedy_options(6));
        const ninfer::GenerationResult ra = ha.wait();
        const ninfer::GenerationResult rb = hb.wait();
        if (ra.generated_token_ids.size() != 6 || rb.generated_token_ids.size() != 6 ||
            check_adaptive_stats(ra, 5, {3, 4, 5}, label) != 0 ||
            check_adaptive_stats(rb, 5, {3, 4, 5}, label) != 0) {
            std::cerr << label << " did not complete compact MTP decode\n";
            dump_speculative("  A", ra.speculative);
            dump_speculative("  B", rb.speculative);
            return 1;
        }
        if (ra.speculative.rounds_per_draft.size() < 5 ||
            ra.speculative.rounds_per_draft[4] == 0) {
            std::cerr << label << " did not record k=4 compact rounds under N=5\n";
            dump_speculative("  A", ra.speculative);
            return 1;
        }
        std::cout << "ok " << label << '\n' << std::flush;
    }

    {
        const char* label = "MTP NVFP4 adaptive RAM reseed";
        ninfer::EngineOptions options = mtp_adaptive_options(artifact, 5, 1);
        options.kv_ram_capacity_bytes = 1024ULL * 1024ULL * 1024ULL;
        ninfer::Engine engine(options);
        if (const int result = check_load(engine); result != 0) { return result; }
        if (engine.memory_summary().kv_ram_capacity_bytes == 0) {
            std::cerr << label << " RAM tier is disabled\n";
            return 1;
        }
        const ninfer::GenerationResult first =
            engine.generate(engine.prepare_tokens(prompts[0]), greedy_reuse(8, false));
        if (first.generated_token_ids.size() != 8 ||
            check_adaptive_stats(first, 5, {3, 4, 5}, label) != 0 ||
            first.speculative.rounds_per_draft.size() < 5 ||
            first.speculative.rounds_per_draft[4] == 0) {
            std::cerr << label << " source did not record seeded k=4 rounds\n";
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
        if (hit.generated_token_ids.size() != 4 ||
            check_adaptive_stats(hit, 5, {3, 4, 5}, label) != 0) {
            std::cerr << label << " RAM restore leaked live_k off the captured set\n";
            dump_speculative("  hit", hit.speculative);
            return 1;
        }
        std::cout << "ok " << label << '\n' << std::flush;
    }

    {
        const char* label = "MTP NVFP4 adaptive RAM restore in flight";
        ninfer::EngineOptions options = mtp_adaptive_options(artifact, 5, 2);
        options.kv_ram_capacity_bytes = 1024ULL * 1024ULL * 1024ULL;
        ninfer::Engine engine(options);
        if (const int result = check_load(engine); result != 0) { return result; }
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
        auto restored_h =
            engine.submit(engine.prepare_tokens(history), greedy_reuse(4, true));
        auto inflight_h =
            engine.submit(engine.prepare_tokens(prompts[1]), greedy_reuse(4, false));
        const ninfer::GenerationResult hit      = restored_h.wait();
        const ninfer::GenerationResult inflight = inflight_h.wait();
        if (hit.generated_token_ids.size() != 4 || inflight.generated_token_ids.size() != 4 ||
            check_adaptive_stats(hit, 5, {3, 4, 5}, label) != 0 ||
            check_adaptive_stats(inflight, 5, {3, 4, 5}, label) != 0) {
            std::cerr << label << " overlapping RAM restore dropped live_k\n";
            dump_speculative("  hit", hit.speculative);
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
