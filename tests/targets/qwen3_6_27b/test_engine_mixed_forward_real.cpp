// Mixed DFlash rounds on the real artifact: a prompt that prefills while another request decodes
// advances inside the decode rounds' target forward (EngineOptions::mixed_forward). With the
// prefill chunk equal to the mixed-forward width, the owner's chunks have the same extents
// whether they run alone or mixed. Owner output must equal its solo run; background streams
// must equal a reference with the same background requests and no prefill owner.

#include "ninfer/engine.h"

#include <chrono>
#include <cmath>
#include <memory>
#include <array>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>
#include <utility>
#include <algorithm>

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
    options.prefill_chunk            = width;
    options.mixed_forward_min_tokens = 0;
    options.mixed_forward_rounds     = rounds;
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

ninfer::EngineOptions isolation_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path   = artifact;
    options.max_concurrency = 4;
    options.max_context     = kMaximumTokens;
    options.kv_capacity     = ninfer::KvCapacityPolicy::explicit_capacity(4 * kMaximumTokens);
    options.prefill_chunk   = 512;
    options.mixed_forward   = 512;
    options.mixed_forward_min_tokens  = 0;
    options.mixed_forward_rounds      = 1;
    options.enable_vision             = false;
    options.speculative.backend       = ninfer::SpeculativeBackend::DFlash;
    options.speculative.draft_tokens  = 5;
    options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    return options;
}

void wait_decode_ready(ninfer::Engine& engine, std::uint32_t lanes) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    while (engine.runtime_stats().decode_ready_requests != lanes) {
        if (std::chrono::steady_clock::now() >= deadline) {
            throw std::runtime_error("isolation decode lanes did not become ready");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

using LaneStreams = std::array<Tokens, 3>;

LaneStreams background_reference(ninfer::Engine& engine,
                                 const std::array<std::uint32_t, 3>& seeds) {
    std::array<ninfer::GenerationHandle, 3> handles;
    for (std::size_t lane = 0; lane < handles.size(); ++lane) {
        handles[lane] = engine.submit(
            engine.prepare_tokens(prompt_tokens(kLanePrompt, seeds[lane])), greedy(kLaneOutputs));
    }
    LaneStreams result;
    for (std::size_t lane = 0; lane < handles.size(); ++lane) {
        result[lane] = handles[lane].wait().generated_token_ids;
    }
    if (result[0] == result[1] || result[0] == result[2] || result[1] == result[2]) {
        throw std::runtime_error("isolation reference streams cannot distinguish lane swaps");
    }
    return result;
}

ninfer::ChatMessage chat_text(ninfer::ChatRole role, std::string value) {
    ninfer::ChatMessage message;
    message.role = role;
    message.parts.push_back({.kind = ninfer::MessagePartKind::Text, .text = std::move(value)});
    return message;
}

std::pair<ninfer::PromptInput, Tokens> prime_incremental_chat(ninfer::Engine& engine) {
    ninfer::PromptInput input;
    input.options.enable_thinking = false;
    std::string history;
    for (int i = 0; i < 64; ++i) { history += "Explain attention and recurrent state. "; }
    input.messages.push_back(chat_text(ninfer::ChatRole::User, std::move(history)));
    const auto first = engine.generate(engine.prepare(input), greedy(16));
    if (first.generated_token_ids.size() != 16) {
        throw std::runtime_error("incremental isolation prime stopped early");
    }
    auto assistant              = chat_text(ninfer::ChatRole::Assistant, first.content);
    assistant.reasoning_content = first.reasoning;
    input.messages.push_back(std::move(assistant));
    std::string followup;
    for (int i = 0; i < 128; ++i) {
        followup += "Describe how independent sequence state is retained. ";
    }
    input.messages.push_back(chat_text(ninfer::ChatRole::User, std::move(followup)));
    return {std::move(input), first.generated_token_ids};
}

void exercise_incremental_isolation(const char* artifact) {
    Tokens prime_reference;
    ninfer::GenerationResult owner_reference;
    LaneStreams lane_reference;
    auto followup_options                         = greedy(1);
    followup_options.execution.allow_prefix_reuse = true;
    {
        ninfer::Engine engine(isolation_options(artifact));
        auto [input, prime_ids] = prime_incremental_chat(engine);
        prime_reference         = std::move(prime_ids);
        owner_reference = engine.generate(engine.prepare(std::move(input)), followup_options);
        lane_reference  = background_reference(engine, {17, 18, 19});
    }
    // A separate Engine prevents the solo followup from becoming a cached hit. Priming first
    // also puts the owner in a different physical slot from its reserved mixed ingress row.
    ninfer::Engine engine(isolation_options(artifact));
    auto [input, prime_ids] = prime_incremental_chat(engine);
    if (prime_ids != prime_reference) { throw std::runtime_error("incremental primes differ"); }
    std::array<ninfer::GenerationHandle, 3> backgrounds;
    for (std::size_t lane = 0; lane < backgrounds.size(); ++lane) {
        backgrounds[lane] = engine.submit(engine.prepare_tokens(prompt_tokens(
                                              kLanePrompt, 17 + static_cast<std::uint32_t>(lane))),
                                          greedy(kLaneOutputs));
    }
    wait_decode_ready(engine, 3);
    const auto before   = engine.runtime_stats();
    const auto owner    = engine.generate(engine.prepare(std::move(input)), followup_options);
    const auto after    = engine.runtime_stats();
    const auto computed = owner.prompt.prompt_tokens - owner.reused_prompt_tokens;
    if (owner.reused_prompt_tokens == 0 ||
        owner.prefix_reuse_path == ninfer::PrefixReusePath::FullReset ||
        owner.prefix_reuse_source != ninfer::PrefixReuseSource::VramResident || computed <= 512 ||
        after.computed_prefill_tokens - before.computed_prefill_tokens != computed ||
        after.mixed_decode_rounds <= before.mixed_decode_rounds ||
        after.decode_ready_requests != 3) {
        throw std::runtime_error("incremental owner did not restore and execute mixed suffix work");
    }
    if (owner.generated_token_ids != owner_reference.generated_token_ids ||
        owner.reused_prompt_tokens != owner_reference.reused_prompt_tokens ||
        owner.prefix_reuse_path != owner_reference.prefix_reuse_path) {
        throw std::runtime_error("incremental mixed owner differs from matching restored solo");
    }
    for (std::size_t lane = 0; lane < backgrounds.size(); ++lane) {
        if (backgrounds[lane].wait().generated_token_ids != lane_reference[lane]) {
            throw std::runtime_error("incremental owner changed background lane " +
                                     std::to_string(lane));
        }
    }
    std::cout << "incremental D3/W512 owner and background isolation passed\n";
}

void exercise_compact_isolation(const char* artifact) {
    const Tokens first_prompt  = prompt_tokens(4096 + 76, 91);
    const Tokens second_prompt = prompt_tokens(4096 + 76, 92);
    Tokens first_reference;
    Tokens second_reference;
    LaneStreams original_reference;
    LaneStreams replacement_reference;
    {
        ninfer::Engine engine(isolation_options(artifact));
        first_reference =
            engine.generate(engine.prepare_tokens(first_prompt), greedy(1)).generated_token_ids;
        second_reference =
            engine.generate(engine.prepare_tokens(second_prompt), greedy(1)).generated_token_ids;
        original_reference    = background_reference(engine, {17, 18, 19});
        replacement_reference = background_reference(engine, {17, 47, 19});
    }
    ninfer::Engine engine(isolation_options(artifact));
    std::array<ninfer::GenerationHandle, 3> backgrounds;
    for (std::size_t lane = 0; lane < backgrounds.size(); ++lane) {
        backgrounds[lane] = engine.submit(engine.prepare_tokens(prompt_tokens(
                                              kLanePrompt, 17 + static_cast<std::uint32_t>(lane))),
                                          greedy(kLaneOutputs));
    }
    wait_decode_ready(engine, 3);
    const auto before   = engine.runtime_stats();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    std::optional<ninfer::GenerationResult> cancelled;
    std::exception_ptr cancel_error;
    std::jthread cancel_middle([&](std::stop_token stop) {
        try {
            cancelled = backgrounds[1].wait(
                nullptr, ninfer::CancellationView([&] {
                    return stop.stop_requested() || std::chrono::steady_clock::now() >= deadline ||
                           engine.runtime_stats().mixed_decode_rounds > before.mixed_decode_rounds;
                }));
        } catch (...) { cancel_error = std::current_exception(); }
    });
    auto first_owner = engine.submit(engine.prepare_tokens(first_prompt), greedy(1));
    cancel_middle.join();
    if (cancel_error) { std::rethrow_exception(cancel_error); }
    const auto compact = engine.runtime_stats();
    if (!cancelled || cancelled->finish_reason != ninfer::FinishReason::Cancelled ||
        compact.decode_ready_requests != 2 || compact.prefilling_requests != 1 ||
        compact.mixed_decode_rounds <= before.mixed_decode_rounds) {
        throw std::runtime_error("middle lane did not cancel while mixed owner remained staged");
    }
    // The single-prefill-owner rule keeps this replacement pending until the first owner finishes.
    // Surviving
    // physical lanes therefore execute compact mixed rows before the vacant slot is reused.
    auto replacement =
        engine.submit(engine.prepare_tokens(prompt_tokens(kLanePrompt, 47)), greedy(kLaneOutputs));
    const auto first = first_owner.wait();
    if (first.generated_token_ids != first_reference ||
        engine.runtime_stats().mixed_decode_rounds <= compact.mixed_decode_rounds) {
        throw std::runtime_error("compact membership did not preserve the mixed owner");
    }
    wait_decode_ready(engine, 3);
    const auto rebound = engine.runtime_stats();
    const auto second  = engine.generate(engine.prepare_tokens(second_prompt), greedy(1));
    const auto after   = engine.runtime_stats();
    if (second.generated_token_ids != second_reference ||
        after.mixed_decode_rounds <= rebound.mixed_decode_rounds ||
        after.decode_ready_requests != 3) {
        throw std::runtime_error("slot reuse did not preserve the next mixed owner");
    }
    for (const std::size_t lane : {0U, 2U}) {
        if (backgrounds[lane].wait().generated_token_ids != original_reference[lane]) {
            throw std::runtime_error("compaction changed survivor lane " + std::to_string(lane));
        }
    }
    if (replacement.wait().generated_token_ids != replacement_reference[1] ||
        cancelled->generated_token_ids.size() > original_reference[1].size() ||
        !std::equal(cancelled->generated_token_ids.begin(), cancelled->generated_token_ids.end(),
                    original_reference[1].begin())) {
        throw std::runtime_error("cancelled or replacement lane stream changed");
    }
    std::cout << "compact D3/D2 membership and replacement D3 isolation passed\n";
}

// Both owners reuse separate resident prefixes. The large owner's ladder/rollback state must
// survive suspension; different logprob settings expose shared output-record bindings. Draft K
// stays fixed: adaptive draft picks K from measured wall-clock round times, and greedy output is
// not identical across K, so the primes and the decoder stream compared across two Engines would
// otherwise depend on timing.
void exercise_cached_prefill_interruption(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path                      = artifact;
    options.max_concurrency                    = 3;
    options.max_context                        = 16384;
    options.kv_capacity                        = ninfer::KvCapacityPolicy::explicit_capacity(32768);
    options.context_checkpoint_marks           = std::vector<std::uint32_t>{4096, 8192};
    options.enable_vision                      = false;
    options.speculative.backend                = ninfer::SpeculativeBackend::DFlash;
    options.speculative.draft_tokens           = 5;
    options.speculative.proposal_head          = ninfer::ProposalHead::Optimized;
    auto large_options                         = greedy(1);
    large_options.execution.allow_prefix_reuse = true;
    large_options.output.raw                   = true;
    large_options.output.top_logprobs          = 2;
    auto small_options                         = greedy(1);
    small_options.execution.allow_prefix_reuse = true;
    const auto background_tokens               = prompt_tokens(kLanePrompt, 947);

    struct Primed {
        Tokens large;
        Tokens small;
    };

    const auto prime = [](ninfer::Engine& engine) {
        Primed prompts{prompt_tokens(2048, 731), prompt_tokens(128, 839)};
        const auto first_large = engine.generate(engine.prepare_tokens(prompts.large), greedy(16));
        const auto first_small = engine.generate(engine.prepare_tokens(prompts.small), greedy(1));
        prompts.large.insert(prompts.large.end(), first_large.generated_token_ids.begin(),
                             first_large.generated_token_ids.end());
        prompts.small.insert(prompts.small.end(), first_small.generated_token_ids.begin(),
                             first_small.generated_token_ids.end());
        const auto large_suffix = prompt_tokens(12288, 1013);
        const auto small_suffix = prompt_tokens(240, 1117);
        prompts.large.insert(prompts.large.end(), large_suffix.begin(), large_suffix.end());
        prompts.small.insert(prompts.small.end(), small_suffix.begin(), small_suffix.end());
        return prompts;
    };
    Primed reference_prompts;
    ninfer::GenerationResult large_reference;
    ninfer::GenerationResult small_reference;
    Tokens decode_reference;
    {
        ninfer::Engine engine(options);
        reference_prompts = prime(engine);
        large_reference =
            engine.generate(engine.prepare_tokens(reference_prompts.large), large_options);
        small_reference =
            engine.generate(engine.prepare_tokens(reference_prompts.small), small_options);
        decode_reference = engine.generate(engine.prepare_tokens(background_tokens), greedy(512))
                               .generated_token_ids;
    }
    ninfer::Engine engine(options);
    auto prompts = prime(engine);
    if (prompts.large != reference_prompts.large || prompts.small != reference_prompts.small) {
        throw std::runtime_error("cached interruption prime streams differ");
    }
    auto large_prompt = engine.prepare_tokens(std::move(prompts.large));
    auto small_prompt = engine.prepare_tokens(std::move(prompts.small));
    auto background   = engine.submit(engine.prepare_tokens(background_tokens), greedy(512));
    wait_decode_ready(engine, 1);
    const auto before          = engine.runtime_stats();
    const auto large_submitted = std::chrono::steady_clock::now();
    auto large                 = engine.submit(std::move(large_prompt), large_options);
    const auto deadline        = large_submitted + std::chrono::seconds(60);
    while (engine.runtime_stats().computed_prefill_tokens < before.computed_prefill_tokens + 4096) {
        if (std::chrono::steady_clock::now() >= deadline) {
            throw std::runtime_error("cached owner did not reach its first chunk");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    const auto small_submitted = std::chrono::steady_clock::now();
    auto small                 = engine.submit(std::move(small_prompt), small_options);
    const auto small_result    = small.wait();
    const auto large_result    = large.wait();
    const double gap   = std::chrono::duration<double>(small_submitted - large_submitted).count();
    const auto matches = [](const ninfer::GenerationResult& actual,
                            const ninfer::GenerationResult& reference) {
        return actual.generated_token_ids == reference.generated_token_ids &&
               actual.reused_prompt_tokens == reference.reused_prompt_tokens &&
               actual.reused_prompt_tokens != 0 &&
               actual.prefix_reuse_path == reference.prefix_reuse_path &&
               actual.prefix_reuse_source == ninfer::PrefixReuseSource::VramResident;
    };
    if (small_result.timings.first_token_seconds + gap >=
            large_result.timings.first_token_seconds ||
        !matches(large_result, large_reference) || !matches(small_result, small_reference) ||
        large_result.captured_context_checkpoint_tokens == 0 ||
        large_result.captured_context_checkpoint_tokens !=
            large_reference.captured_context_checkpoint_tokens ||
        large_result.content_logprobs.size() != 1 || large_reference.content_logprobs.size() != 1 ||
        !std::isfinite(large_result.content_logprobs.front().logprob) ||
        !std::isfinite(large_reference.content_logprobs.front().logprob) ||
        std::abs(large_result.content_logprobs.front().logprob -
                 large_reference.content_logprobs.front().logprob) > 1.0e-4F) {
        throw std::runtime_error("cached interruption changed prefix, logprob or owner state");
    }
    if (background.wait().generated_token_ids != decode_reference) {
        throw std::runtime_error("cached interruption changed the B=1 decoder stream");
    }
    std::cout << "cached spare-lane interruption/checkpoint/logprob isolation passed\n";
}

// The sanitizer route targets ownership transfer directly; reference generation and the full
// decode histories are qualified separately by exercise_adaptive_scheduling.
void exercise_prefill_resume_memory(const char* artifact, std::uint32_t draft_tokens = 5) {
    ninfer::EngineOptions options;
    options.artifact_path              = artifact;
    options.max_concurrency            = 3;
    options.max_context                = 16384;
    options.kv_capacity                = ninfer::KvCapacityPolicy::explicit_capacity(32768);
    options.enable_vision              = false;
    options.speculative.backend        = ninfer::SpeculativeBackend::DFlash;
    options.speculative.draft_tokens   = draft_tokens;
    options.speculative.adaptive_draft = true;
    options.speculative.proposal_head  = ninfer::ProposalHead::Optimized;
    ninfer::Engine engine(options);
    auto large_options                = greedy(1);
    large_options.output.raw          = true;
    large_options.output.top_logprobs = 2;
    auto large_prompt                 = engine.prepare_tokens(prompt_tokens(12288, 731));
    auto small_prompt                 = engine.prepare_tokens(prompt_tokens(240, 839));
    auto background =
        engine.submit(engine.prepare_tokens(prompt_tokens(kLanePrompt, 947)), greedy(512));
    wait_decode_ready(engine, 1);
    const auto before          = engine.runtime_stats();
    const auto large_submitted = std::chrono::steady_clock::now();
    auto large                 = engine.submit(std::move(large_prompt), large_options);
    const auto deadline        = large_submitted + std::chrono::seconds(120);
    while (engine.runtime_stats().computed_prefill_tokens < before.computed_prefill_tokens + 4096) {
        if (std::chrono::steady_clock::now() >= deadline) {
            throw std::runtime_error("resume memory route did not reach its first chunk");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    const auto small_submitted = std::chrono::steady_clock::now();
    auto small                 = engine.submit(std::move(small_prompt), greedy(1));
    const auto short_result    = small.wait();
    const auto long_result     = large.wait();
    const auto after           = engine.runtime_stats();
    const double gap = std::chrono::duration<double>(small_submitted - large_submitted).count();
    if (short_result.timings.first_token_seconds + gap >= long_result.timings.first_token_seconds ||
        short_result.generated_token_ids.size() != 1 ||
        long_result.generated_token_ids.size() != 1 || long_result.content_logprobs.size() != 1 ||
        !std::isfinite(long_result.content_logprobs.front().logprob) ||
        after.computed_prefill_tokens - before.computed_prefill_tokens != 12528 ||
        after.mixed_decode_rounds - before.mixed_decode_rounds != 2 ||
        after.decode_ready_requests != 1 || after.prefilling_requests != 0) {
        throw std::runtime_error("resume memory route lost an owner, output record or chunk");
    }
    const auto decoded = background.wait(nullptr, ninfer::CancellationView([] { return true; }));
    if (decoded.finish_reason != ninfer::FinishReason::Cancelled ||
        engine.runtime_stats().running_requests != 0) {
        throw std::runtime_error("resume memory route did not release the final decoder");
    }
    std::cout << "adaptive-draft K" << draft_tokens << " spare-lane resume memory route passed\n";
}

// Same chunk extents and B=1 decoder profile isolate resumed lane state from numerical-profile
// differences. Short requests stop at their prefill token, so they never change decode membership.
void exercise_adaptive_scheduling(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path             = artifact;
    options.max_concurrency           = 3;
    options.max_context               = 16384;
    options.kv_capacity               = ninfer::KvCapacityPolicy::explicit_capacity(32768);
    options.enable_vision             = false;
    options.speculative.backend       = ninfer::SpeculativeBackend::DFlash;
    options.speculative.draft_tokens  = 5;
    options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    auto engine                       = std::make_unique<ninfer::Engine>(options);
    auto owner_options                = greedy(1);
    owner_options.output.raw          = true;
    owner_options.output.top_logprobs = 2;
    const auto large_tokens           = prompt_tokens(12288, 731);
    const auto small_tokens           = prompt_tokens(240, 839);
    const auto large_reference =
        engine->generate(engine->prepare_tokens(large_tokens), owner_options);
    const auto small_reference = engine->generate(engine->prepare_tokens(small_tokens), greedy(1));
    const auto background_tokens = prompt_tokens(kLanePrompt, 947);
    const auto background_reference =
        engine->generate(engine->prepare_tokens(background_tokens), greedy(512));
    auto background = engine->submit(engine->prepare_tokens(background_tokens), greedy(512));
    wait_decode_ready(*engine, 1);

    // Frontier boundary cases: after the admission chunk exactly 4095/4096/4097 tokens remain.
    for (const auto count : {8191U, 8192U, 8193U}) {
        auto prompt               = engine->prepare_tokens(prompt_tokens(count, count + 17));
        const auto before         = engine->runtime_stats();
        const auto result         = engine->generate(std::move(prompt), greedy(1));
        const auto after          = engine->runtime_stats();
        const auto expected_mixed = count == 8191 ? 0U : 1U;
        if (after.mixed_decode_rounds - before.mixed_decode_rounds != expected_mixed ||
            result.prompt.prompt_tokens - result.reused_prompt_tokens != count) {
            throw std::runtime_error("adaptive 4095/4096/4097 frontier dispatched the wrong route");
        }
    }

    if (background.wait().generated_token_ids != background_reference.generated_token_ids) {
        throw std::runtime_error("adaptive frontier cases changed the B=1 decoder stream");
    }
    for (const auto threshold : {0U, 4096U}) {
        options.mixed_forward_min_tokens = threshold;
        // Reference requests leave retained lanes. This scenario needs an unused spare lane,
        // rather than requiring interruption to evict a completed request's retained allocation.
        engine.reset();
        engine     = std::make_unique<ninfer::Engine>(options);
        background = engine->submit(engine->prepare_tokens(background_tokens), greedy(512));
        wait_decode_ready(*engine, 1);

        auto small_prompt          = engine->prepare_tokens(small_tokens);
        auto large_prompt          = engine->prepare_tokens(large_tokens);
        const auto before          = engine->runtime_stats();
        const auto large_submitted = std::chrono::steady_clock::now();
        auto large                 = engine->submit(std::move(large_prompt), owner_options);
        const auto deadline        = large_submitted + std::chrono::seconds(60);
        while (engine->runtime_stats().computed_prefill_tokens <
               before.computed_prefill_tokens + 4096) {
            if (std::chrono::steady_clock::now() >= deadline) {
                throw std::runtime_error("large prefill did not reach its first chunk boundary");
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        const auto small_submitted = std::chrono::steady_clock::now();
        auto small                 = engine->submit(std::move(small_prompt), greedy(1));
        const auto small_result    = small.wait();
        const auto large_result    = large.wait();
        const auto completed       = std::chrono::steady_clock::now();
        const auto after           = engine->runtime_stats();
        const double submission_gap =
            std::chrono::duration<double>(small_submitted - large_submitted).count();
        const bool short_first = small_result.timings.first_token_seconds + submission_gap <
                                 large_result.timings.first_token_seconds;
        if (short_first != (threshold != 0) ||
            large_result.generated_token_ids != large_reference.generated_token_ids ||
            small_result.generated_token_ids != small_reference.generated_token_ids ||
            large_result.content_logprobs.size() != 1 ||
            large_reference.content_logprobs.size() != 1 ||
            !std::isfinite(large_result.content_logprobs.front().logprob) ||
            !std::isfinite(large_reference.content_logprobs.front().logprob) ||
            std::abs(large_result.content_logprobs.front().logprob -
                     large_reference.content_logprobs.front().logprob) > 1.0e-4F ||
            after.running_requests != 1 || after.prefilling_requests != 0) {
            throw std::runtime_error(
                "spare-lane resume: short TTFT=" +
                std::to_string(small_result.timings.first_token_seconds) +
                " submit_gap=" + std::to_string(submission_gap) + " large TTFT=" +
                std::to_string(large_result.timings.first_token_seconds) + " large_match=" +
                std::to_string(large_result.generated_token_ids ==
                               large_reference.generated_token_ids) +
                " small_match=" +
                std::to_string(small_result.generated_token_ids ==
                               small_reference.generated_token_ids) +
                " logprobs=" + std::to_string(large_result.content_logprobs.size()) +
                " ref_logprobs=" + std::to_string(large_reference.content_logprobs.size()) +
                " running=" + std::to_string(after.running_requests) +
                " prefilling=" + std::to_string(after.prefilling_requests));
        }
        if (background.wait().generated_token_ids != background_reference.generated_token_ids) {
            throw std::runtime_error(
                "adaptive prefill interruption changed the B=1 decoder stream");
        }
        const double service_seconds =
            std::chrono::duration<double>(completed - large_submitted).count();
        std::cout << "adaptive 4K frontier and spare-lane resume isolation passed; threshold "
                  << threshold << ", short TTFT "
                  << small_result.timings.first_token_seconds * 1000.0 << " ms, large TTFT "
                  << large_result.timings.first_token_seconds * 1000.0 << " ms, effective prefill "
                  << (12288.0 + 240.0) / service_seconds << " tok/s, aggregate decode "
                  << static_cast<double>(after.committed_decode_tokens -
                                         before.committed_decode_tokens) /
                         service_seconds
                  << " tok/s\n";
    }
}

} // namespace

int main(int argc, char** argv) {
    const char* artifact = std::getenv("NINFER_QWEN3_8_27B_NVFP4_DFLASH_WEIGHTS");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: set NINFER_QWEN3_8_27B_NVFP4_DFLASH_WEIGHTS\n";
        return 77;
    }
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--incremental-isolation") {
            exercise_incremental_isolation(artifact);
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--compact-isolation") {
            exercise_compact_isolation(artifact);
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--cached-prefill-interruption") {
            exercise_cached_prefill_interruption(artifact);
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--prefill-resume-memory") {
            exercise_prefill_resume_memory(artifact);
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--prefill-resume-k7") {
            exercise_prefill_resume_memory(artifact, 7);
            return 0;
        }
        if (argc == 2 && std::string_view(argv[1]) == "--adaptive-scheduling") {
            exercise_adaptive_scheduling(artifact);
            return 0;
        }
        if (argc != 1) { throw std::invalid_argument("unknown mixed-forward test argument"); }
        exercise_cached_prefill_interruption(artifact);
        exercise_prefill_resume_memory(artifact);
        exercise_prefill_resume_memory(artifact, 7);
        exercise_adaptive_scheduling(artifact);
        exercise_incremental_isolation(artifact);
        exercise_compact_isolation(artifact);
        exercise(artifact, 4096, 3, 1);
        for (const std::uint32_t rounds : {1U, 2U}) { exercise(artifact, 512, 1, rounds); }
        for (const std::uint32_t width : {1024U, 2048U}) { exercise(artifact, width, 5, 1); }
    } catch (const std::exception& error) {
        std::cerr << "mixed forward: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
