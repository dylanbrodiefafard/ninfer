#include "ninfer/engine.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
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
