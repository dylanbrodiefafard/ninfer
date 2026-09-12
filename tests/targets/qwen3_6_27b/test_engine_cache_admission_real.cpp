#include "ninfer/engine.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

ninfer::RequestOptions request_options() {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = 2;
    options.execution.sampling.temperature = 0.0F;
    options.execution.allow_prefix_reuse = false;
    options.stop.include_model_defaults = false;
    return options;
}

void exercise(const char* artifact, std::uint32_t concurrency) {
    ninfer::EngineOptions options;
    options.artifact_path = artifact;
    options.max_concurrency = concurrency;
    options.max_context = (concurrency + 1) * 64;
    options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity((concurrency + 1) * 64);
    options.prefill_chunk = 128;
    options.kv_ram_capacity_bytes = 1ULL << 20;
    ninfer::Engine engine(options);

    // Fill every free lane with a distinct one-page retained bundle. Each GDN
    // image exceeds the RAM budget, but no capture is needed until all lanes fill.
    for (std::uint32_t lane = 0; lane < concurrency; ++lane) {
        std::vector<ninfer::TokenId> tokens(32, 198);
        tokens.front() = 1000 + lane;
        const auto result = engine.generate(engine.prepare_tokens(tokens), request_options());
        if (result.generated_token_ids.size() != 2) {
            throw std::runtime_error("retained-lane fixture did not complete");
        }
    }
    const auto before = engine.runtime_stats();
    if (before.kv_ram_captures != 0 || before.kv_ram_drops != 0) {
        throw std::runtime_error("fixture did not fill empty lanes without capture");
    }

    // This request needs the entire pool. For C > 1, admission must reclaim
    // both its target lane and other retained lanes despite every capture dropping.
    std::vector<ninfer::TokenId> tokens(concurrency * 64 + 1, 198);
    tokens.front() = 2200;
    const auto result = engine.generate(engine.prepare_tokens(tokens), request_options());
    if (result.generated_token_ids.size() != 2 ||
        result.prefix_reuse_source != ninfer::PrefixReuseSource::None) {
        throw std::runtime_error("cache drop prevented a fresh request from completing");
    }
    const auto after = engine.runtime_stats();
    if (after.kv_ram_drops != before.kv_ram_drops + concurrency ||
        after.kv_ram_captures != 0 || after.kv_ram_evictions != 0) {
        throw std::runtime_error("admission did not drop each uncapturable victim exactly once");
    }

    // The executor remains usable after reclaiming all victims.
    tokens.assign(32, 198);
    tokens.front() = 3300;
    if (engine.generate(engine.prepare_tokens(tokens), request_options()).generated_token_ids.size()
        != 2) {
        throw std::runtime_error("cache-drop admission poisoned the next request");
    }

    if (concurrency > 1) {
        // Make every lane dirty again, then replace two while their requests
        // overlap. A cache drop must neither fail admission nor abort a peer.
        for (std::uint32_t lane = 0; lane < concurrency; ++lane) {
            tokens.front() = 4000 + lane;
            (void)engine.generate(engine.prepare_tokens(tokens), request_options());
        }
        auto overlap_options = request_options();
        overlap_options.execution.requested_output_tokens = 32;
        tokens.assign(16, 198);
        tokens.front() = 5000;
        const auto drops = engine.runtime_stats().kv_ram_drops;
        auto a = engine.submit(engine.prepare_tokens(tokens), overlap_options);
        tokens.front() = 6000;
        auto b = engine.submit(engine.prepare_tokens(tokens), overlap_options);
        bool overlapped = false;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (std::chrono::steady_clock::now() < deadline) {
            const auto stats = engine.runtime_stats();
            if (stats.running_requests >= 2) { overlapped = true; break; }
            if (stats.running_requests == 0 && stats.waiting_requests == 0) { break; }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        const auto result_a = a.wait();
        const auto result_b = b.wait();
        if (!overlapped || result_a.generated_token_ids.size() != 32 ||
            result_b.generated_token_ids.size() != 32 ||
            engine.runtime_stats().kv_ram_drops != drops + 2) {
            throw std::runtime_error("concurrent dirty-lane admission did not survive cache drops");
        }
    }
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_QWEN3_6_27B_NVFP4_WEIGHTS");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: set NINFER_QWEN3_6_27B_NVFP4_WEIGHTS to the Qwen3.8 NVFP4 artifact\n";
        return 77;
    }
    int failures = 0;
    for (std::uint32_t concurrency = 1; concurrency <= 4; ++concurrency) {
        try {
            exercise(artifact, concurrency);
            std::cout << "cache-drop admission C=" << concurrency << " passed\n";
        } catch (const std::exception& error) {
            std::cerr << "cache-drop admission C=" << concurrency << ": " << error.what() << '\n';
            ++failures;
        }
    }
    return failures == 0 ? 0 : 1;
}
