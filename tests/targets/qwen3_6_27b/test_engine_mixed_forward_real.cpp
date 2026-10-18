// Mixed DFlash rounds on the real artifact: a prompt that prefills while another request decodes
// advances inside the decode rounds' target forward (EngineOptions::mixed_forward). With the
// prefill chunk equal to the mixed-forward width, the owner's chunks have the same extents
// whether they run alone or mixed. Owner output must equal its solo run; background streams
// must equal a reference with the same background requests and no prefill owner.

#include "ninfer/engine.h"

#include <chrono>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using Tokens = std::vector<ninfer::TokenId>;

constexpr std::uint32_t kLaneOutputs   = 256;
constexpr std::uint32_t kLanePrompt    = 96;
constexpr std::uint32_t kMaximumTokens = 12288;

ninfer::RequestOptions greedy(std::uint32_t outputs) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = outputs;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = false;
    options.stop.include_model_defaults       = false;
    return options;
}

// Deterministic in-vocabulary tokens; the content only needs to differ between prompts.
Tokens prompt_tokens(std::uint32_t count, std::uint32_t seed) {
    Tokens tokens(count);
    std::uint32_t state = seed;
    for (auto& token : tokens) {
        state = state * 1664525U + 1013904223U;
        token = static_cast<ninfer::TokenId>(1000U + (state >> 8) % 60000U);
    }
    return tokens;
}

void exercise(const char* artifact, std::uint32_t width, std::uint32_t decode_lanes,
              std::uint32_t rounds) {
    ninfer::EngineOptions options;
    options.artifact_path   = artifact;
    options.max_concurrency = decode_lanes + 1;
    options.max_context     = kMaximumTokens;
    options.kv_capacity =
        ninfer::KvCapacityPolicy::explicit_capacity((decode_lanes + 1) * kMaximumTokens);
    options.prefill_chunk = width;
    if (width != 4096) {
        options.mixed_forward        = width;
        options.mixed_forward_rounds = rounds;
    }
    options.enable_vision             = false;
    options.speculative.backend       = ninfer::SpeculativeBackend::DFlash;
    options.speculative.draft_tokens  = 5;
    options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    ninfer::Engine engine(options);

    std::vector<Tokens> lane_prompts;
    lane_prompts.reserve(decode_lanes);
    for (std::uint32_t lane = 0; lane < decode_lanes; ++lane) {
        lane_prompts.push_back(prompt_tokens(kLanePrompt, 17 + lane));
    }
    // Submit the same background batch in the reference. The D5 owner stops after its
    // prefill token so it cannot change that batch to six lanes during later decode.
    std::vector<ninfer::GenerationHandle> references;
    references.reserve(decode_lanes);
    for (const auto& prompt : lane_prompts) {
        references.push_back(engine.submit(engine.prepare_tokens(prompt), greedy(kLaneOutputs)));
    }
    std::vector<Tokens> lane_reference;
    lane_reference.reserve(decode_lanes);
    for (auto& reference : references) {
        lane_reference.push_back(reference.wait().generated_token_ids);
    }
    const Tokens owner_prompt         = prompt_tokens(2 * width + 76, 29);
    const std::uint32_t owner_outputs = decode_lanes == 1 ? 24U : 1U;
    const auto owner_solo =
        engine.generate(engine.prepare_tokens(owner_prompt), greedy(owner_outputs));

    std::vector<ninfer::GenerationHandle> lanes;
    lanes.reserve(decode_lanes);
    for (const auto& prompt : lane_prompts) {
        lanes.push_back(engine.submit(engine.prepare_tokens(prompt), greedy(kLaneOutputs)));
    }
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (engine.runtime_stats().decode_ready_requests != decode_lanes) {
        if (std::chrono::steady_clock::now() >= deadline) {
            throw std::runtime_error("the decoding lanes never became decode-ready");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    const auto before = engine.runtime_stats();
    const auto owner_mixed =
        engine.generate(engine.prepare_tokens(owner_prompt), greedy(owner_outputs));
    const auto after = engine.runtime_stats();
    if (after.decode_ready_requests != decode_lanes) {
        throw std::runtime_error("a background lane completed before the mixed-batch check");
    }

    // The first chunk runs on admission; the remaining full chunk and tail must each execute
    // inside a decode round. Owner generation after prefill cannot increment this counter.
    if (after.mixed_decode_rounds - before.mixed_decode_rounds != 2) {
        throw std::runtime_error("the owner did not execute both staged chunks in mixed rounds");
    }
    if (owner_mixed.generated_token_ids != owner_solo.generated_token_ids) {
        throw std::runtime_error("mixed rounds changed the prefill owner's greedy stream");
    }
    for (std::uint32_t lane = 0; lane < decode_lanes; ++lane) {
        const Tokens mixed_tokens = lanes[lane].wait().generated_token_ids;
        if (mixed_tokens != lane_reference[lane]) {
            std::size_t first = 0;
            while (first < mixed_tokens.size() && first < lane_reference[lane].size() &&
                   mixed_tokens[first] == lane_reference[lane][first]) {
                ++first;
            }
            throw std::runtime_error("mixed rounds changed decoding lane " + std::to_string(lane) +
                                     " at token " + std::to_string(first));
        }
    }
    std::cout << "mixed forward " << width << "x" << rounds << " D" << decode_lanes << " passed ("
              << after.mixed_decode_rounds - before.mixed_decode_rounds
              << " mixed decode rounds)\n";
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_QWEN3_8_27B_NVFP4_DFLASH_WEIGHTS");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: set NINFER_QWEN3_8_27B_NVFP4_DFLASH_WEIGHTS\n";
        return 77;
    }
    try {
        exercise(artifact, 4096, 3, 1);
        for (const std::uint32_t rounds : {1U, 2U}) { exercise(artifact, 512, 1, rounds); }
        for (const std::uint32_t width : {1024U, 2048U}) { exercise(artifact, width, 5, 1); }
    } catch (const std::exception& error) {
        std::cerr << "mixed forward: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
