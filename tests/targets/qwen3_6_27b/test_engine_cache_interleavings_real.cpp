#include "ninfer/engine.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

using Tokens = std::vector<ninfer::TokenId>;
constexpr std::size_t kRamBytes = 2ULL << 30;

ninfer::RequestOptions greedy(std::uint32_t outputs, bool reuse) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = outputs;
    options.execution.sampling.temperature = 0.0F;
    options.execution.allow_prefix_reuse = reuse;
    options.stop.include_model_defaults = false;
    return options;
}

Tokens prompt(std::uint32_t lane, bool filler = false) {
    return {248045, 846, 198, static_cast<ninfer::TokenId>(5834 + lane + (filler ? 100 : 0)),
            248046, 198};
}

Tokens history(Tokens input, const ninfer::GenerationResult& result) {
    if (result.generated_token_ids.empty()) {
        throw std::runtime_error("published cancellation left no committed token");
    }
    input.insert(input.end(), result.generated_token_ids.begin(),
                 result.generated_token_ids.end() - 1);
    return input;
}

void equal_slice(const Tokens& actual, const Tokens& reference, std::size_t offset, const char* stage) {
    if (offset + actual.size() > reference.size() ||
        !std::equal(actual.begin(), actual.end(), reference.begin() + offset)) {
        std::cerr << stage << " offset=" << offset << " actual=";
        for (const auto token : actual) { std::cerr << token << ','; }
        std::cerr << " reference=";
        for (std::size_t i = offset; i < std::min(reference.size(), offset + actual.size()); ++i) {
            std::cerr << reference[i] << ',';
        }
        std::cerr << '\n';
        throw std::runtime_error("cache continuation differs from fresh greedy computation");
    }
}

struct DiskDirectory {
    explicit DiskDirectory(std::uint32_t concurrency)
        : path(std::filesystem::temp_directory_path() /
               ("ninfer-cache-interleavings-" + std::to_string(::getpid()) + "-" +
                std::to_string(concurrency))) {
        std::filesystem::remove_all(path);
        std::filesystem::create_directories(path);
    }
    ~DiskDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
    std::filesystem::path path;
};

void exercise(const char* artifact, std::uint32_t concurrency, bool disk_enabled,
              ninfer::SpeculativeBackend backend, bool early_cancel = false) {
    DiskDirectory disk(concurrency);
    ninfer::EngineOptions options;
    options.artifact_path = artifact;
    options.max_concurrency = concurrency;
    options.max_context = 256;
    options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(concurrency * 256);
    options.prefill_chunk = 128;
    options.kv_ram_capacity_bytes = kRamBytes;
    options.enable_vision = false;
    options.speculative.backend = backend;
    if (backend != ninfer::SpeculativeBackend::None) {
        options.speculative.draft_tokens = 4;
        options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    }
    if (disk_enabled) {
        options.kv_disk_capacity_bytes = 4ULL << 30;
        options.kv_disk_location = disk.path;
    }
    ninfer::Engine engine(options);
    // Ordinary cancellation follows the first publication; speculative cancellation
    // follows the second so each canceled request executes its backend. Enumerate
    // every subset of consumers at that boundary. Submission
    // order is fixed; worker/CUDA schedules are not claimed to be exhaustive here.
    // Real kernels, per-lane GDN slots, retained-page eviction and both cache workers
    // participate in every row, which a host state-machine oracle cannot exercise.
    for (std::uint32_t mask = 0; mask < (1U << concurrency); ++mask) {
        std::array<Tokens, 4> inputs;
        for (std::uint32_t lane = 0; lane < concurrency; ++lane) {
            const auto seed = engine.generate(engine.prepare_tokens(prompt(lane)), greedy(8, false));
            if (seed.generated_token_ids.size() != 8 ||
                (backend != ninfer::SpeculativeBackend::None && seed.speculative.rounds == 0)) {
                throw std::runtime_error("cache matrix seed did not complete");
            }
            inputs[lane] = history(prompt(lane), seed);
        }
        // Replace every completed lane so each wave begins with host-resident sources.
        for (std::uint32_t lane = 0; lane < concurrency; ++lane) {
            (void)engine.generate(engine.prepare_tokens(prompt(lane, true)), greedy(2, false));
        }

        const auto before_wave = engine.runtime_stats();
        std::array<ninfer::GenerationResult, 4> results;
        std::array<std::exception_ptr, 4> errors;
        std::vector<std::jthread> consumers;
        for (std::uint32_t lane = 0; lane < concurrency; ++lane) {
            auto handle = engine.submit(engine.prepare_tokens(inputs[lane]), greedy(32, true),
                                        ninfer::OutputDelivery::Streaming);
            consumers.emplace_back([&, lane, handle = std::move(handle)]() mutable {
                struct Published : ninfer::OutputSink {
                    std::uint32_t publications = 0;
                    void publish(ninfer::OutputDelta) override { ++publications; }
                } sink;
                ninfer::CancellationView cancel([&] {
                    const std::uint32_t required =
                        (backend == ninfer::SpeculativeBackend::None || early_cancel) ? 1U : 2U;
                    return (mask & (1U << lane)) != 0 && sink.publications >= required;
                });
                try {
                    results[lane] = handle.wait(&sink, cancel);
                } catch (...) {
                    errors[lane] = std::current_exception();
                }
            });
        }
        consumers.clear(); // Join every consumer before reading its result or the mask.
        const auto after_wave = engine.runtime_stats();
        if (mask == 0 && concurrency > 1 &&
            after_wave.decode_row_rounds - before_wave.decode_row_rounds <=
                after_wave.decode_rounds - before_wave.decode_rounds) {
            throw std::runtime_error("cache wave did not execute any overlapping decode batch");
        }
        for (std::uint32_t lane = 0; lane < concurrency; ++lane) {
            if (errors[lane]) { std::rethrow_exception(errors[lane]); }
            const auto& result = results[lane];
            if (result.prefix_reuse_source != ninfer::PrefixReuseSource::HostRam ||
                result.reused_prompt_tokens != inputs[lane].size()) {
                throw std::runtime_error("cancellation matrix did not restore the requested RAM source");
            }
            const bool cancelled = (mask & (1U << lane)) != 0;
            if (backend != ninfer::SpeculativeBackend::None && !early_cancel &&
                result.speculative.rounds == 0) {
                throw std::runtime_error("cache wave did not publish a speculative round");
            }
            if (cancelled ? result.finish_reason != ninfer::FinishReason::Cancelled
                          : result.generated_token_ids.size() != 32) {
                throw std::runtime_error("cancellation matrix did not reach its assigned terminal state");
            }
        }

        // Capture retained states, then continue each history. Cancellation at a
        // committed boundary can retain; cancellation of an in-flight ordinary
        // round must release its overwritten state. Either way, recomputation or
        // reuse must match fresh output, and completed peers must remain reusable.
        for (std::uint32_t lane = 0; lane < concurrency; ++lane) {
            (void)engine.generate(engine.prepare_tokens(prompt(lane, true)), greedy(2, false));
        }
        for (std::uint32_t lane = 0; lane < concurrency; ++lane) {
            const auto resumed_input = history(inputs[lane], results[lane]);
            // Compare one next-token decision for the exact same represented
            // history. This does not assert multi-token equivalence between
            // fresh prefill and cached decode arithmetic routes.
            const auto resumed = engine.generate(engine.prepare_tokens(resumed_input), greedy(1, true));
            if ((mask & (1U << lane)) == 0 &&
                (resumed.prefix_reuse_source != ninfer::PrefixReuseSource::HostRam ||
                 resumed.reused_prompt_tokens != resumed_input.size())) {
                throw std::runtime_error("completed peer did not survive RAM eviction/restore");
            }
            const auto fresh = engine.generate(engine.prepare_tokens(resumed_input), greedy(1, false));
            if (resumed.generated_token_ids.size() != 1 || fresh.generated_token_ids.size() != 1) {
                throw std::runtime_error("next-token continuation did not produce one token");
            }
            equal_slice(resumed.generated_token_ids, fresh.generated_token_ids, 0, "resume");
        }
        const auto stats = engine.runtime_stats();
        if (stats.running_requests != 0 || stats.waiting_requests != 0 || stats.kv_ram_drops != 0) {
            throw std::runtime_error("cancellation matrix leaked a request or dropped its cache fixture");
        }
        std::cerr << "cache interleavings " << (disk_enabled ? "RAM+disk" : "RAM")
                  << " C=" << concurrency << " early=" << early_cancel
                  << " cancellation mask=" << mask << " passed\n";
    }
    if (disk_enabled && engine.runtime_stats().kv_disk_captures == 0) {
        throw std::runtime_error("combined cache matrix did not execute any disk spill");
    }
}

} // namespace

int main(int argc, char** argv) {
    const std::string selected = argc == 3 && std::string(argv[1]) == "--backend"
                                     ? argv[2] : "ordinary";
    if ((argc != 1 && argc != 3) ||
        (argc == 3 && std::string(argv[1]) != "--backend") ||
        (selected != "ordinary" && selected != "mtp" && selected != "dflash")) {
        std::cerr << "usage: cache_interleavings_real [--backend ordinary|mtp|dflash]\n";
        return 1;
    }
    const auto backend = selected == "dflash" ? ninfer::SpeculativeBackend::DFlash
                         : selected == "mtp" ? ninfer::SpeculativeBackend::Mtp
                                              : ninfer::SpeculativeBackend::None;
    const char* artifact = std::getenv(selected == "dflash"
        ? "NINFER_QWEN3_8_27B_NVFP4_DFLASH_WEIGHTS"
        : "NINFER_QWEN3_6_27B_NVFP4_WEIGHTS");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: set "
                  << (selected == "dflash" ? "NINFER_QWEN3_8_27B_NVFP4_DFLASH_WEIGHTS"
                                            : "NINFER_QWEN3_6_27B_NVFP4_WEIGHTS")
                  << " to the Qwen3.8 NVFP4 artifact with the selected backend\n";
        return 77;
    }
    try {
        if (backend == ninfer::SpeculativeBackend::Mtp) {
            // A first-publication cancellation can leave an invalid-tail image
            // at the same frontier as a later fresh capture. It must not shadow
            // that usable entry during the next wave's RAM/disk matching.
            for (const bool disk : {false, true}) {
                exercise(artifact, 2, disk, backend, true);
            }
        }
        for (const bool disk : {false, true}) {
            for (std::uint32_t concurrency = 1; concurrency <= 4; ++concurrency) {
                exercise(artifact, concurrency, disk, backend);
            }
        }
    } catch (const std::exception& error) {
        if (backend == ninfer::SpeculativeBackend::Mtp &&
            std::string(error.what()).find("mtp/") != std::string::npos) {
            std::cout << "skip: MTP cache interleavings need MTP artifact objects\n";
            return 77;
        }
        std::cerr << "cache interleavings: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
