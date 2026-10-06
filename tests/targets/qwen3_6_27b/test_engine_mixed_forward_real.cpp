// Mixed DFlash rounds on the real artifact: a prompt that prefills while another request decodes
// advances inside the decode rounds' target forward (EngineOptions::mixed_forward). With the
// prefill chunk equal to the mixed-forward width, the owner's chunks have the same extents
// whether they run alone or mixed, so both greedy streams must equal their solo runs.

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

constexpr std::uint32_t kWidth         = 512;
constexpr std::uint32_t kLaneOutputs   = 256;
constexpr std::uint32_t kOwnerOutputs  = 24;
constexpr std::uint32_t kOwnerTokens   = 2 * kWidth + 76;
constexpr std::uint32_t kLanePrompt    = 96;
constexpr std::uint32_t kMaximumTokens = 4096;

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

void exercise(const char* artifact, std::uint32_t rounds) {
    ninfer::EngineOptions options;
    options.artifact_path        = artifact;
    options.max_concurrency      = 2;
    options.max_context          = kMaximumTokens;
    options.kv_capacity          = ninfer::KvCapacityPolicy::explicit_capacity(2 * kMaximumTokens);
    options.prefill_chunk        = kWidth;
    options.mixed_forward        = kWidth;
    options.mixed_forward_rounds = rounds;
    options.enable_vision        = false;
    options.speculative.backend  = ninfer::SpeculativeBackend::DFlash;
    options.speculative.draft_tokens  = 5;
    options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    ninfer::Engine engine(options);

    const Tokens lane_prompt  = prompt_tokens(kLanePrompt, 17);
    const Tokens owner_prompt = prompt_tokens(kOwnerTokens, 29);
    const auto lane_solo =
        engine.generate(engine.prepare_tokens(lane_prompt), greedy(kLaneOutputs));
    const auto owner_solo =
        engine.generate(engine.prepare_tokens(owner_prompt), greedy(kOwnerOutputs));

    auto lane           = engine.submit(engine.prepare_tokens(lane_prompt), greedy(kLaneOutputs));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (engine.runtime_stats().decode_ready_requests != 1) {
        if (std::chrono::steady_clock::now() >= deadline) {
            throw std::runtime_error("the decoding lane never became decode-ready");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    const auto before = engine.runtime_stats();
    const auto owner_mixed =
        engine.generate(engine.prepare_tokens(owner_prompt), greedy(kOwnerOutputs));
    const auto after      = engine.runtime_stats();
    const auto lane_mixed = lane.wait();

    // The first chunk runs on admission; the remaining full chunk and tail must each execute
    // inside a decode round. Owner generation after prefill cannot increment this counter.
    if (after.mixed_decode_rounds - before.mixed_decode_rounds != 2) {
        throw std::runtime_error("the owner did not execute both staged chunks in mixed rounds");
    }
    if (owner_mixed.generated_token_ids != owner_solo.generated_token_ids) {
        throw std::runtime_error("mixed rounds changed the prefill owner's greedy stream");
    }
    if (lane_mixed.generated_token_ids != lane_solo.generated_token_ids) {
        throw std::runtime_error("mixed rounds changed the decoding lane's greedy stream");
    }
    std::cout << "mixed forward " << kWidth << "x" << rounds << " passed ("
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
        for (const std::uint32_t rounds : {1U, 2U}) { exercise(artifact, rounds); }
    } catch (const std::exception& error) {
        std::cerr << "mixed forward: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
