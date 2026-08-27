#include "ninfer/engine.h"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

ninfer::EngineOptions engine_options(const char* artifact) {
    ninfer::EngineOptions options;
    options.artifact_path             = artifact;
    options.max_context               = 4096;
    options.kv_capacity               = ninfer::KvCapacityPolicy::explicit_capacity(4096);
    options.prefill_chunk             = 1024;
    options.speculative.backend       = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens  = 3;
    options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    options.enable_vision             = true;
    return options;
}

ninfer::EngineOptions dflash_engine_options(const char* artifact) {
    ninfer::EngineOptions options     = engine_options(artifact);
    options.speculative.backend       = ninfer::SpeculativeBackend::DFlash;
    options.speculative.draft_tokens  = 3;
    options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    options.enable_vision             = false;
    return options;
}

ninfer::RequestOptions greedy_reuse(bool reuse, std::uint32_t tokens) {
    ninfer::RequestOptions result;
    result.execution.requested_output_tokens = tokens;
    result.execution.sampling.temperature    = 0.0F;
    result.execution.allow_prefix_reuse      = reuse;
    result.stop.include_model_defaults       = false;
    return result;
}

std::vector<ninfer::TokenId> padded_ids(ninfer::TokenId pad, std::size_t count) {
    std::vector<ninfer::TokenId> ids(count, 198);
    if (count != 0) { ids[0] = pad; }
    return ids;
}

int verify_loaded_product(const ninfer::Engine& engine);

std::vector<std::uint8_t> gradient_ppm() {
    std::vector<std::uint8_t> ppm;
    const std::string header = "P6\n64 64\n255\n";
    ppm.insert(ppm.end(), header.begin(), header.end());
    for (int index = 0; index < 64 * 64; ++index) {
        ppm.push_back(static_cast<std::uint8_t>(index & 0xff));
        ppm.push_back(static_cast<std::uint8_t>((index * 3) & 0xff));
        ppm.push_back(static_cast<std::uint8_t>((index * 7) & 0xff));
    }
    return ppm;
}

ninfer::PromptInput chinese_chat(bool enable_thinking) {
    ninfer::ChatMessage message;
    message.role = ninfer::ChatRole::User;
    message.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = "你好，简单介绍一下你自己。", .media = {}});
    ninfer::PromptInput input;
    input.messages.push_back(std::move(message));
    input.options.enable_thinking = enable_thinking;
    return input;
}

const char* reuse_path_name(ninfer::PrefixReusePath path) {
    switch (path) {
    case ninfer::PrefixReusePath::FullReset:
        return "full_reset";
    case ninfer::PrefixReusePath::AppendAtFrontier:
        return "append_frontier";
    case ninfer::PrefixReusePath::RestoreTurnCheckpoint:
        return "restore_turn_checkpoint";
    case ninfer::PrefixReusePath::RestoreResponseCheckpoint:
        return "restore_response_checkpoint";
    case ninfer::PrefixReusePath::RestoreContextCheckpoint:
        return "restore_context_checkpoint";
    case ninfer::PrefixReusePath::RestoreTurnRollback:
        return "restore_turn_rollback";
    }
    return "unknown";
}

ninfer::ChatMessage text_turn(ninfer::ChatRole role, std::string text) {
    ninfer::ChatMessage message;
    message.role = role;
    message.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = std::move(text), .media = {}});
    return message;
}

int exercise_max_tokens_chat_followup(ninfer::Engine& engine) {
    auto options = [](bool reuse, std::uint32_t tokens) {
        ninfer::RequestOptions result;
        result.execution.requested_output_tokens = tokens;
        result.execution.sampling.temperature    = 0.0F;
        result.execution.allow_prefix_reuse      = reuse;
        result.stop.include_model_defaults       = false;
        return result;
    };

    ninfer::PromptInput first_input;
    first_input.messages.push_back(text_turn(ninfer::ChatRole::User, "Say hello in one sentence."));
    first_input.options.enable_thinking   = true;
    first_input.options.preserve_thinking = false;

    const ninfer::GenerationResult first =
        engine.generate(engine.prepare(first_input), options(false, 8));
    std::cerr << "max-tokens follow-up first: finish="
              << static_cast<int>(first.finish_reason) << " gen=" << first.generated_token_ids.size()
              << " reasoning_tokens=" << first.reasoning_tokens
              << " content_bytes=" << first.content.size()
              << " reasoning_bytes=" << first.reasoning.size() << '\n';
    if (first.finish_reason != ninfer::FinishReason::OutputLimit ||
        first.generated_token_ids.size() != 8) {
        std::cerr << "max-tokens follow-up source did not stop at the requested output limit\n";
        return 1;
    }

    ninfer::PromptInput followup = first_input;
    ninfer::ChatMessage assistant = text_turn(ninfer::ChatRole::Assistant, first.content);
    assistant.reasoning_content   = first.reasoning;
    followup.messages.push_back(std::move(assistant));
    followup.messages.push_back(text_turn(ninfer::ChatRole::User, "Now say it in French."));

    const ninfer::GenerationResult reused =
        engine.generate(engine.prepare(std::move(followup)), options(true, 4));
    std::cerr << "max-tokens follow-up second: path=" << reuse_path_name(reused.prefix_reuse_path)
              << " source=" << static_cast<int>(reused.prefix_reuse_source)
              << " reused=" << reused.reused_prompt_tokens
              << " prompt=" << reused.prompt.prompt_tokens
              << " gen=" << reused.generated_token_ids.size() << '\n';
    if (reused.prefix_reuse_path == ninfer::PrefixReusePath::FullReset ||
        reused.reused_prompt_tokens == 0) {
        std::cerr << "max-tokens chat follow-up FullReset (needs fix)\n";
        return 1;
    }
    return 0;
}

int exercise_cancel_retain_rollback(ninfer::Engine& engine) {
    auto options = [](bool reuse, std::uint32_t tokens) {
        ninfer::RequestOptions result;
        result.execution.requested_output_tokens = tokens;
        result.execution.sampling.temperature    = 0.0F;
        result.execution.allow_prefix_reuse      = reuse;
        result.stop.include_model_defaults       = false;
        return result;
    };

    ninfer::PromptInput first_input;
    first_input.messages.push_back(text_turn(ninfer::ChatRole::User, "Say hello in one sentence."));
    first_input.options.enable_thinking   = true;
    first_input.options.preserve_thinking = false;

    ninfer::PreparedPrompt first_prompt = engine.prepare(first_input);

    struct CancelAfterPublish : ninfer::OutputSink {
        std::atomic<int> published{0};
        std::atomic<bool> stop{false};
        void publish(ninfer::OutputDelta) override {
            if (published.fetch_add(1) + 1 >= 2) { stop.store(true); }
        }
    } sink;
    ninfer::CancellationView cancel([&] { return sink.stop.load(); });

    const ninfer::GenerationResult first =
        engine.generate(std::move(first_prompt), options(false, 32), &sink, cancel);
    std::cerr << "cancel retain first: finish=" << static_cast<int>(first.finish_reason)
              << " gen=" << first.generated_token_ids.size() << '\n';
    if (first.finish_reason != ninfer::FinishReason::Cancelled ||
        first.generated_token_ids.empty()) {
        std::cerr << "cancel retain source did not stop with published tokens\n";
        return 1;
    }

    const ninfer::GenerationResult restored =
        engine.generate(engine.prepare(first_input), options(true, 4));
    std::cerr << "cancel retain rollback: path=" << reuse_path_name(restored.prefix_reuse_path)
              << " reused=" << restored.reused_prompt_tokens
              << " prompt=" << restored.prompt.prompt_tokens << '\n';
    if (restored.prefix_reuse_path != ninfer::PrefixReusePath::RestoreTurnCheckpoint ||
        restored.reused_prompt_tokens == 0) {
        std::cerr << "rolling back a cancelled decode did not restore the turn checkpoint\n";
        return 1;
    }
    return 0;
}

int exercise_prefill_cancel_keeps_checkpoint(ninfer::Engine& engine) {
    auto options = [](bool reuse, std::uint32_t tokens) {
        ninfer::RequestOptions result;
        result.execution.requested_output_tokens = tokens;
        result.execution.sampling.temperature    = 0.0F;
        result.execution.allow_prefix_reuse      = reuse;
        result.stop.include_model_defaults       = false;
        return result;
    };

    ninfer::PromptInput first_input;
    first_input.messages.push_back(text_turn(ninfer::ChatRole::User, "Say hello in one sentence."));
    first_input.options.enable_thinking   = true;
    first_input.options.preserve_thinking = false;

    const ninfer::GenerationResult first =
        engine.generate(engine.prepare(first_input), options(false, 8));
    std::cerr << "prefill-cancel first: finish=" << static_cast<int>(first.finish_reason)
              << " prompt=" << first.prompt.prompt_tokens
              << " gen=" << first.generated_token_ids.size() << '\n';
    if (first.finish_reason != ninfer::FinishReason::OutputLimit ||
        first.generated_token_ids.size() != 8) {
        std::cerr << "prefill-cancel source did not stop at the requested output limit\n";
        return 1;
    }

    std::string long_user;
    long_user.reserve(3500 * 9);
    for (int index = 0; index < 3500; ++index) { long_user += "continue "; }

    ninfer::PromptInput followup = first_input;
    ninfer::ChatMessage assistant = text_turn(ninfer::ChatRole::Assistant, first.content);
    assistant.reasoning_content   = first.reasoning;
    followup.messages.push_back(std::move(assistant));
    followup.messages.push_back(text_turn(ninfer::ChatRole::User, std::move(long_user)));

    const std::uint64_t prefill_before = engine.runtime_stats().computed_prefill_tokens;
    std::atomic<bool> stop{false};
    ninfer::CancellationView cancel([&] {
        const ninfer::RuntimeStats stats = engine.runtime_stats();
        if (stats.prefilling_requests > 0 || stats.computed_prefill_tokens > prefill_before) {
            stop.store(true);
        }
        return stop.load();
    });
    const ninfer::GenerationResult cancelled =
        engine.generate(engine.prepare(std::move(followup)), options(true, 4), nullptr, cancel);
    std::cerr << "prefill-cancel second: finish=" << static_cast<int>(cancelled.finish_reason)
              << " path=" << reuse_path_name(cancelled.prefix_reuse_path)
              << " reused=" << cancelled.reused_prompt_tokens
              << " prompt=" << cancelled.prompt.prompt_tokens << '\n';
    if (cancelled.finish_reason != ninfer::FinishReason::Cancelled) {
        std::cerr << "long follow-up was not cancelled during suffix prefill\n";
        return 1;
    }

    const ninfer::GenerationResult restored =
        engine.generate(engine.prepare(first_input), options(true, 4));
    std::cerr << "prefill-cancel rollback: path=" << reuse_path_name(restored.prefix_reuse_path)
              << " reused=" << restored.reused_prompt_tokens
              << " prompt=" << restored.prompt.prompt_tokens << '\n';
    if (restored.prefix_reuse_path != ninfer::PrefixReusePath::AppendAtFrontier ||
        restored.prompt.prompt_tokens != first.prompt.prompt_tokens ||
        restored.reused_prompt_tokens == 0 ||
        restored.reused_prompt_tokens >= first.prompt.prompt_tokens) {
        std::cerr << "cancelled suffix prefill did not revert live state to the turn checkpoint\n";
        return 1;
    }
    return 0;
}

int exercise_first_prompt_cancel_aborts(ninfer::Engine& engine) {
    const std::vector<ninfer::TokenId> prompt = padded_ids(203, 1536);
    const std::uint64_t before = engine.runtime_stats().computed_prefill_tokens;
    ninfer::CancellationView cancel([&] {
        return engine.runtime_stats().computed_prefill_tokens >= before + 1024;
    });
    const ninfer::GenerationResult cancelled =
        engine.generate(engine.prepare_tokens(prompt), greedy_reuse(true, 1), nullptr, cancel);
    std::cerr << "first-prompt cancel: finish=" << static_cast<int>(cancelled.finish_reason)
              << " captured=" << cancelled.captured_context_checkpoint_tokens << '\n';
    if (cancelled.finish_reason != ninfer::FinishReason::Cancelled) {
        std::cerr << "first-prompt prepare_tokens cancel did not cancel\n";
        return 1;
    }
    if (cancelled.captured_context_checkpoint_tokens != 0 ||
        cancelled.restored_context_checkpoint_tokens != 0) {
        std::cerr << "first-prompt cancel published a context-checkpoint head\n";
        return 1;
    }
    std::vector<ninfer::TokenId> probe = padded_ids(203, 64);
    probe.insert(probe.end(), 8, 200);
    const ninfer::GenerationResult after =
        engine.generate(engine.prepare_tokens(probe), greedy_reuse(true, 4));
    std::cerr << "first-prompt cancel probe: path=" << reuse_path_name(after.prefix_reuse_path)
              << " reused=" << after.reused_prompt_tokens << '\n';
    if (after.prefix_reuse_path == ninfer::PrefixReusePath::RestoreContextCheckpoint) {
        std::cerr << "first-prompt cancel left a hittable ladder head\n";
        return 1;
    }
    const ninfer::GenerationResult retry =
        engine.generate(engine.prepare_tokens(prompt), greedy_reuse(true, 1));
    if (retry.prefix_reuse_path != ninfer::PrefixReusePath::FullReset) {
        std::cerr << "retrying a cancelled first prompt was not FullReset: path="
                  << reuse_path_name(retry.prefix_reuse_path) << '\n';
        return 1;
    }
    return 0;
}

int exercise_append_cancel_keeps_turn_rollback(ninfer::Engine& engine) {
    const std::vector<ninfer::TokenId> pin_src = padded_ids(236, 256);
    const ninfer::GenerationResult pin_r1 =
        engine.generate(engine.prepare_tokens(pin_src), greedy_reuse(true, 2));
    std::cerr << "append-cancel pin first: gen=" << pin_r1.generated_token_ids.size() << '\n';
    if (pin_r1.generated_token_ids.size() != 2) {
        std::cerr << "append-cancel source did not generate two tokens\n";
        return 1;
    }

    std::vector<ninfer::TokenId> pin_follow = pin_src;
    pin_follow.insert(pin_follow.end(), pin_r1.generated_token_ids.begin(),
                      pin_r1.generated_token_ids.end());
    pin_follow.resize(2500, 198);
    const std::uint64_t before = engine.runtime_stats().computed_prefill_tokens;
    ninfer::CancellationView cancel([&] {
        return engine.runtime_stats().computed_prefill_tokens >= before + 1024;
    });
    const ninfer::GenerationResult cancelled =
        engine.generate(engine.prepare_tokens(pin_follow), greedy_reuse(true, 1), nullptr, cancel);
    std::cerr << "append-cancel suffix: finish=" << static_cast<int>(cancelled.finish_reason)
              << " path=" << reuse_path_name(cancelled.prefix_reuse_path) << '\n';
    if (cancelled.finish_reason != ninfer::FinishReason::Cancelled) {
        std::cerr << "append suffix was not cancelled during prefill\n";
        return 1;
    }

    std::vector<ninfer::TokenId> probe = pin_src;
    probe.insert(probe.end(), pin_r1.generated_token_ids.begin(), pin_r1.generated_token_ids.end());
    probe.insert(probe.end(), 8, 200);
    const ninfer::GenerationResult pin_after =
        engine.generate(engine.prepare_tokens(probe), greedy_reuse(true, 4));
    std::cerr << "append-cancel probe: path=" << reuse_path_name(pin_after.prefix_reuse_path)
              << " reused=" << pin_after.reused_prompt_tokens << '\n';
    if ((pin_after.prefix_reuse_path != ninfer::PrefixReusePath::AppendAtFrontier &&
         pin_after.prefix_reuse_path != ninfer::PrefixReusePath::RestoreTurnRollback) ||
        pin_after.reused_prompt_tokens == 0) {
        std::cerr << "cancelled append suffix did not keep the previous turn\n";
        return 1;
    }
    return 0;
}

int exercise_cancel_ladder_heads(const char* artifact) {
    ninfer::EngineOptions options = engine_options(artifact);
    options.max_context           = 32768;
    options.kv_capacity           = ninfer::KvCapacityPolicy::explicit_capacity(32768);
    options.prefill_chunk         = 4096;
    options.enable_vision         = false;
    ninfer::Engine engine(options);
    if (const int rc = verify_loaded_product(engine); rc != 0) { return rc; }

    constexpr std::uint32_t kMark = 24576;
    auto cancel_first = [&](ninfer::TokenId pad, std::uint32_t prompt_tokens,
                            std::uint64_t extra_tokens, const char* label) {
        const auto prompt = padded_ids(pad, prompt_tokens);
        const auto before = engine.runtime_stats().computed_prefill_tokens;
        ninfer::CancellationView cancel([&] {
            return engine.runtime_stats().computed_prefill_tokens >= before + extra_tokens;
        });
        const ninfer::GenerationResult cancelled =
            engine.generate(engine.prepare_tokens(prompt), greedy_reuse(true, 1), nullptr, cancel);
        std::cerr << label << " finish=" << static_cast<int>(cancelled.finish_reason)
                  << " captured=" << cancelled.captured_context_checkpoint_tokens << '\n';
        if (cancelled.finish_reason != ninfer::FinishReason::Cancelled) {
            std::cerr << label << " did not cancel\n";
            return 1;
        }
        if (extra_tokens < kMark && cancelled.captured_context_checkpoint_tokens != 0) {
            std::cerr << label << " captured a ladder head before the freeze chunk\n";
            return 1;
        }
        std::vector<ninfer::TokenId> diverge = padded_ids(pad, kMark);
        diverge.insert(diverge.end(), 8, 200);
        const ninfer::GenerationResult after =
            engine.generate(engine.prepare_tokens(diverge), greedy_reuse(true, 4));
        std::cerr << label << " probe path=" << reuse_path_name(after.prefix_reuse_path)
                  << " reused=" << after.reused_prompt_tokens << '\n';
        if (after.prefix_reuse_path == ninfer::PrefixReusePath::RestoreContextCheckpoint) {
            std::cerr << label << " left a hittable ladder head\n";
            return 1;
        }
        const ninfer::GenerationResult evict =
            engine.generate(engine.prepare_tokens(padded_ids(216, 8)), greedy_reuse(false, 1));
        if (evict.generated_token_ids.size() != 1) {
            std::cerr << label << " eviction did not complete\n";
            return 1;
        }
        return 0;
    };
    if (const int rc =
            cancel_first(203, kMark, 4096, "ladder cancel before freeze");
        rc != 0) {
        return rc;
    }

    const auto pin_src = padded_ids(236, 256);
    const ninfer::GenerationResult pin_r1 =
        engine.generate(engine.prepare_tokens(pin_src), greedy_reuse(true, 2));
    if (pin_r1.generated_token_ids.size() != 2) {
        std::cerr << "ladder pin first visit did not generate\n";
        return 1;
    }
    std::vector<ninfer::TokenId> pin_follow = pin_src;
    pin_follow.insert(pin_follow.end(), pin_r1.generated_token_ids.begin(),
                      pin_r1.generated_token_ids.end());
    pin_follow.resize(8192, 198);
    const auto before_pin = engine.runtime_stats().computed_prefill_tokens;
    ninfer::CancellationView pin_cancel([&] {
        return engine.runtime_stats().computed_prefill_tokens >= before_pin + 4096;
    });
    const ninfer::GenerationResult pin_cancelled =
        engine.generate(engine.prepare_tokens(pin_follow), greedy_reuse(true, 1), nullptr,
                        pin_cancel);
    std::cerr << "ladder pin cancel: finish=" << static_cast<int>(pin_cancelled.finish_reason)
              << " path=" << reuse_path_name(pin_cancelled.prefix_reuse_path) << '\n';
    if (pin_cancelled.finish_reason != ninfer::FinishReason::Cancelled) {
        std::cerr << "ladder pin suffix was not cancelled\n";
        return 1;
    }
    std::vector<ninfer::TokenId> pin_probe = pin_src;
    pin_probe.insert(pin_probe.end(), pin_r1.generated_token_ids.begin(),
                     pin_r1.generated_token_ids.end());
    pin_probe.insert(pin_probe.end(), 8, 200);
    const ninfer::GenerationResult pin_after =
        engine.generate(engine.prepare_tokens(pin_probe), greedy_reuse(true, 4));
    std::cerr << "ladder pin probe: path=" << reuse_path_name(pin_after.prefix_reuse_path)
              << " reused=" << pin_after.reused_prompt_tokens << '\n';
    if ((pin_after.prefix_reuse_path != ninfer::PrefixReusePath::AppendAtFrontier &&
         pin_after.prefix_reuse_path != ninfer::PrefixReusePath::RestoreTurnRollback) ||
        pin_after.reused_prompt_tokens == 0) {
        std::cerr << "cancel after rollback pin did not keep the previous turn\n";
        return 1;
    }
    return 0;
}

int exercise_registered_frontend(const ninfer::Engine& engine) {
    if (engine.count_tokens(chinese_chat(true)) != 16) {
        std::cerr << "registered tokenizer/chat template changed the thinking prompt golden\n";
        return 1;
    }
    if (engine.count_tokens(chinese_chat(false)) != 18) {
        std::cerr << "registered tokenizer/chat template changed the no-thinking prompt golden\n";
        return 1;
    }
    return 0;
}

int exercise_full_prefill_chunk(ninfer::Engine& engine) {
    constexpr std::size_t kChunkTokens = 1024;
    std::vector<ninfer::TokenId> prompt(kChunkTokens, 198);
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = 1;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.allow_prefix_reuse      = false;
    options.stop.include_model_defaults       = false;

    const ninfer::GenerationResult result =
        engine.generate(engine.prepare_tokens(std::move(prompt)), options);
    if (result.generated_token_ids.size() != 1 ||
        result.finish_reason != ninfer::FinishReason::OutputLimit) {
        std::cerr << "full-chunk prefill did not complete through the planned workspace\n";
        return 1;
    }
    return 0;
}

int exercise_zero_suffix_reuse(ninfer::Engine& engine, const std::vector<ninfer::TokenId>& prompt) {
    ninfer::RequestOptions baseline_options;
    baseline_options.execution.requested_output_tokens = 8;
    baseline_options.execution.sampling.temperature    = 0.0F;
    baseline_options.execution.allow_prefix_reuse      = false;
    baseline_options.stop.include_model_defaults       = false;
    const ninfer::GenerationResult baseline =
        engine.generate(engine.prepare_tokens(prompt), baseline_options);
    if (baseline.generated_token_ids.size() != 8) {
        std::cerr << "zero-suffix baseline did not generate eight tokens\n";
        return 1;
    }

    std::vector<ninfer::TokenId> exact_frontier = prompt;
    exact_frontier.insert(exact_frontier.end(), baseline.generated_token_ids.begin(),
                          baseline.generated_token_ids.end() - 1);

    ninfer::RequestOptions reuse_options;
    reuse_options.execution.requested_output_tokens = 2;
    reuse_options.execution.sampling.temperature    = 0.0F;
    reuse_options.execution.allow_prefix_reuse      = true;
    reuse_options.stop.include_model_defaults       = false;
    const ninfer::GenerationResult reused =
        engine.generate(engine.prepare_tokens(exact_frontier), reuse_options);
    if (reused.reused_prompt_tokens != exact_frontier.size()) {
        std::cerr << "zero-suffix reuse count is " << reused.reused_prompt_tokens << ", expected "
                  << exact_frontier.size() << '\n';
        return 1;
    }
    if (reused.generated_token_ids.size() != 2 ||
        reused.generated_token_ids[0] != baseline.generated_token_ids.back()) {
        std::cerr << "zero-suffix reuse did not resume from the retained target frontier\n";
        return 1;
    }
    return 0;
}

int exercise_prefix(ninfer::Engine& engine) {
    ninfer::RequestOptions first_options;
    first_options.execution.requested_output_tokens = 5;
    first_options.execution.sampling.temperature    = 0.0F;
    first_options.stop.include_model_defaults       = false;

    const std::vector<ninfer::TokenId> prompt{248045, 846, 198, 5834, 248046, 198};
    const ninfer::GenerationResult first =
        engine.generate(engine.prepare_tokens(prompt), first_options);
    if (first.generated_token_ids.size() != 5) {
        std::cerr << "first request did not generate five tokens\n";
        return 1;
    }

    std::vector<ninfer::TokenId> continuation = prompt;
    continuation.insert(continuation.end(), first.generated_token_ids.begin(),
                        first.generated_token_ids.end());
    continuation.push_back(198);

    ninfer::RequestOptions reuse_options;
    reuse_options.execution.requested_output_tokens = 5;
    reuse_options.execution.sampling.temperature    = 0.0F;
    reuse_options.execution.allow_prefix_reuse      = true;
    reuse_options.stop.include_model_defaults       = false;
    const ninfer::GenerationResult reused =
        engine.generate(engine.prepare_tokens(continuation), reuse_options);

    const std::uint32_t expected_reuse =
        static_cast<std::uint32_t>(prompt.size() + first.generated_token_ids.size() - 1);
    if (reused.reused_prompt_tokens != expected_reuse) {
        std::cerr << "append reuse count is " << reused.reused_prompt_tokens << ", expected "
                  << expected_reuse << '\n';
        return 1;
    }

    if (const int result = exercise_zero_suffix_reuse(engine, prompt); result != 0) {
        return result;
    }

    return 0;
}

int exercise_rewrite_checkpoints(ninfer::Engine& engine) {
    auto text_message = [](ninfer::ChatRole role, std::string text) {
        ninfer::ChatMessage message;
        message.role = role;
        message.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = std::move(text), .media = {}});
        return message;
    };
    auto assistant_call = [&](std::string reasoning, std::string id, std::string key) {
        ninfer::ChatMessage message = text_message(ninfer::ChatRole::Assistant, "");
        message.reasoning_content   = std::move(reasoning);
        message.tool_calls.push_back(ninfer::ToolCall{
            .id = std::move(id), .name = "lookup", .arguments_json = "{\"key\":\"" + key + "\"}"});
        return message;
    };
    auto input_with_history = [&](int completed_responses, bool preserve_thinking) {
        ninfer::PromptInput input;
        input.messages.push_back(text_message(
            ninfer::ChatRole::User,
            "Use the lookup results to determine the deterministic checkpoint value."));
        if (completed_responses >= 1) {
            input.messages.push_back(
                assistant_call("The first lookup should be alpha.", "call_alpha", "alpha"));
            ninfer::ChatMessage tool =
                text_message(ninfer::ChatRole::Tool, "{\"value\":17,\"next\":\"beta\"}");
            tool.tool_call_id = "call_alpha";
            input.messages.push_back(std::move(tool));
        }
        if (completed_responses >= 2) {
            input.messages.push_back(
                assistant_call("The alpha result requests beta.", "call_beta", "beta"));
            ninfer::ChatMessage tool = text_message(ninfer::ChatRole::Tool, "{\"value\":25}");
            tool.tool_call_id        = "call_beta";
            input.messages.push_back(std::move(tool));
        }
        input.options.preserve_thinking = preserve_thinking;
        input.options.tool_jsons.push_back(
            R"({"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{"key":{"type":"string"}},"required":["key"]}}})");
        return input;
    };
    auto options = [](bool reuse) {
        ninfer::RequestOptions result;
        result.execution.requested_output_tokens = 4;
        result.execution.sampling.temperature    = 0.0F;
        result.execution.allow_prefix_reuse      = reuse;
        result.stop.include_model_defaults       = false;
        return result;
    };

    const ninfer::GenerationResult first =
        engine.generate(engine.prepare(input_with_history(0, true)), options(false));
    if (first.generated_token_ids.size() != 4 ||
        first.prefix_reuse_path != ninfer::PrefixReusePath::FullReset) {
        std::cerr << "response-checkpoint source request did not complete from a cold lane\n";
        return 1;
    }

    const ninfer::GenerationResult exact_replay =
        engine.generate(engine.prepare(input_with_history(0, false)), options(true));
    if (exact_replay.generated_token_ids.size() != 4 ||
        exact_replay.prefix_reuse_path != ninfer::PrefixReusePath::RestoreResponseCheckpoint ||
        exact_replay.reused_prompt_tokens == 0) {
        std::cerr << "prompt-frontier response checkpoint was not restored on an exact replay\n";
        return 1;
    }

    const ninfer::GenerationResult first_replay =
        engine.generate(engine.prepare(input_with_history(1, true)), options(true));
    if (first_replay.generated_token_ids.size() != 4 ||
        first_replay.prefix_reuse_path != ninfer::PrefixReusePath::RestoreResponseCheckpoint ||
        first_replay.reused_prompt_tokens == 0) {
        std::cerr << "normalized first response did not restore its response checkpoint: path="
                  << static_cast<int>(first_replay.prefix_reuse_path)
                  << " reused=" << first_replay.reused_prompt_tokens << '\n';
        return 1;
    }

    const ninfer::GenerationResult second_replay =
        engine.generate(engine.prepare(input_with_history(2, true)), options(true));
    if (second_replay.generated_token_ids.size() != 4 ||
        second_replay.prefix_reuse_path != ninfer::PrefixReusePath::RestoreResponseCheckpoint ||
        second_replay.reused_prompt_tokens <= first_replay.reused_prompt_tokens) {
        std::cerr << "rolling response checkpoint did not advance across the tool loop: first="
                  << first_replay.reused_prompt_tokens
                  << " second=" << second_replay.reused_prompt_tokens << '\n';
        return 1;
    }

    const ninfer::GenerationResult mode_change =
        engine.generate(engine.prepare(input_with_history(2, false)), options(true));
    if (mode_change.generated_token_ids.size() != 4 ||
        mode_change.prefix_reuse_path != ninfer::PrefixReusePath::RestoreResponseCheckpoint ||
        mode_change.reused_prompt_tokens == 0) {
        std::cerr << "preserve-thinking policy change discarded a compatible response checkpoint: "
                  << "path=" << static_cast<int>(mode_change.prefix_reuse_path)
                  << " reused=" << mode_change.reused_prompt_tokens << '\n';
        return 1;
    }
    return 0;
}

int exercise_vision(ninfer::Engine& engine) {
    const auto image_bytes = gradient_ppm();
    auto image_part        = [](const std::vector<std::uint8_t>& bytes, std::string name) {
        ninfer::MessagePart image;
        image.kind              = ninfer::MessagePartKind::Media;
        image.media.kind        = ninfer::MediaKind::Image;
        image.media.bytes       = bytes;
        image.media.media_type  = "image/x-portable-pixmap";
        image.media.source_name = std::move(name);
        return image;
    };
    auto assistant_message = [](const ninfer::GenerationResult& result) {
        ninfer::ChatMessage message;
        message.role              = ninfer::ChatRole::Assistant;
        message.reasoning_content = result.reasoning;
        message.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = result.content, .media = {}});
        return message;
    };
    auto first_input = [&](const std::vector<std::uint8_t>& bytes) {
        ninfer::ChatMessage message;
        message.role = ninfer::ChatRole::User;
        message.parts.push_back(image_part(bytes, "inline.ppm"));
        message.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = "What is visible?", .media = {}});
        ninfer::PromptInput input;
        input.messages.push_back(std::move(message));
        input.options.enable_thinking = false;
        return input;
    };
    auto followup_input = [&](const std::vector<std::uint8_t>& bytes,
                              const ninfer::GenerationResult& first) {
        ninfer::PromptInput input = first_input(bytes);
        input.messages.push_back(assistant_message(first));
        ninfer::ChatMessage followup;
        followup.role = ninfer::ChatRole::User;
        followup.parts.push_back(ninfer::MessagePart{
            .kind = ninfer::MessagePartKind::Text, .text = "Give one more detail.", .media = {}});
        input.messages.push_back(std::move(followup));
        return input;
    };
    auto appended_media_input =
        [&](const std::vector<std::uint8_t>& old_bytes, const ninfer::GenerationResult& first,
            const ninfer::GenerationResult& second, const std::vector<std::uint8_t>& new_bytes) {
            ninfer::PromptInput input = followup_input(old_bytes, first);
            input.messages.push_back(assistant_message(second));
            ninfer::ChatMessage followup;
            followup.role = ninfer::ChatRole::User;
            followup.parts.push_back(image_part(new_bytes, "second.ppm"));
            followup.parts.push_back(ninfer::MessagePart{
                .kind = ninfer::MessagePartKind::Text, .text = "Compare the images.", .media = {}});
            input.messages.push_back(std::move(followup));
            return input;
        };

    auto options = [](bool reuse) {
        ninfer::RequestOptions result;
        result.execution.requested_output_tokens = 2;
        result.execution.sampling.temperature    = 0.0F;
        result.execution.allow_prefix_reuse      = reuse;
        result.stop.include_model_defaults       = false;
        return result;
    };

    const ninfer::GenerationResult first =
        engine.generate(engine.prepare(first_input(image_bytes)), options(false));
    if (!first.prompt.has_media || first.generated_token_ids.size() != 2 ||
        first.finish_reason != ninfer::FinishReason::OutputLimit) {
        std::cerr << "real Vision request did not complete through the public Engine\n";
        return 1;
    }

    const ninfer::GenerationResult reused =
        engine.generate(engine.prepare(followup_input(image_bytes, first)), options(true));
    if (reused.reused_prompt_tokens == 0 || reused.timings.vision_seconds != 0.0 ||
        reused.generated_token_ids.size() != 2) {
        std::cerr << "same-media continuation did not reuse the resident Vision prefix: reused="
                  << reused.reused_prompt_tokens << " vision=" << reused.timings.vision_seconds
                  << '\n';
        return 1;
    }

    std::vector<std::uint8_t> second_image = image_bytes;
    second_image.back() ^= 0x5aU;
    const ninfer::GenerationResult appended = engine.generate(
        engine.prepare(appended_media_input(image_bytes, first, reused, second_image)),
        options(true));
    if (appended.reused_prompt_tokens == 0 || !(appended.timings.vision_seconds > 0.0) ||
        appended.generated_token_ids.size() != 2) {
        std::cerr << "new-media suffix did not preserve the old multimodal prefix: reused="
                  << appended.reused_prompt_tokens << " vision=" << appended.timings.vision_seconds
                  << '\n';
        return 1;
    }

    const ninfer::GenerationResult baseline = engine.generate(
        engine.prepare(appended_media_input(image_bytes, first, reused, second_image)),
        options(false));
    if (baseline.generated_token_ids != appended.generated_token_ids) {
        std::cerr << "multimodal prefix reuse changed greedy output\n";
        return 1;
    }

    std::vector<std::uint8_t> changed_prefix = image_bytes;
    changed_prefix[changed_prefix.size() - 2] ^= 0x33U;
    const ninfer::GenerationResult miss = engine.generate(
        engine.prepare(appended_media_input(changed_prefix, first, reused, second_image)),
        options(true));
    if (miss.reused_prompt_tokens != 0) {
        std::cerr << "changed media content incorrectly reused placeholder-token KV\n";
        return 1;
    }

    ninfer::RequestOptions mtp_options            = options(false);
    mtp_options.execution.requested_output_tokens = 5;
    const ninfer::GenerationResult mtp_baseline =
        engine.generate(engine.prepare(first_input(image_bytes)), mtp_options);
    if (mtp_baseline.generated_token_ids.size() != 5 ||
        mtp_baseline.generated_token_ids[0] == mtp_baseline.generated_token_ids[1]) {
        std::cerr << "multimodal stop fixture did not produce distinct leading tokens\n";
        return 1;
    }
    ninfer::RequestOptions stop_options = mtp_options;
    stop_options.stop.token_ids.push_back(mtp_baseline.generated_token_ids[1]);
    const ninfer::GenerationResult stopped =
        engine.generate(engine.prepare(first_input(image_bytes)), stop_options);
    if (stopped.finish_reason != ninfer::FinishReason::StopToken ||
        stopped.generated_token_ids.size() != 2 ||
        stopped.generated_token_ids[0] != mtp_baseline.generated_token_ids[0] ||
        stopped.generated_token_ids[1] != mtp_baseline.generated_token_ids[1]) {
        std::cerr << "multimodal custom stop did not terminate at the selected token\n";
        return 1;
    }
    const ninfer::GenerationResult stopped_reuse =
        engine.generate(engine.prepare(followup_input(image_bytes, stopped)), options(true));
    if (stopped_reuse.reused_prompt_tokens == 0 || stopped_reuse.timings.vision_seconds != 0.0) {
        std::cerr << "multimodal stop discarded its reusable boundary: reused="
                  << stopped_reuse.reused_prompt_tokens
                  << " vision=" << stopped_reuse.timings.vision_seconds << '\n';
        return 1;
    }

    // Exact registered rendering prefix before the first image-pad column:
    // <|im_start|>user\n<|vision_start|>. Reusing it places the MTP bridge directly on the first
    // Vision merger column rather than on an ordinary token embedding.
    const std::vector<ninfer::TokenId> visual_prefix{248045, 846, 198, 248053};
    ninfer::RequestOptions source_options            = options(false);
    source_options.execution.requested_output_tokens = 1;
    const ninfer::GenerationResult bridge_source =
        engine.generate(engine.prepare_tokens(visual_prefix), source_options);
    ninfer::RequestOptions bridge_options            = options(true);
    bridge_options.execution.requested_output_tokens = 5;
    const ninfer::GenerationResult visual_bridge =
        engine.generate(engine.prepare(first_input(image_bytes)), bridge_options);
    if (bridge_source.generated_token_ids.size() != 1 ||
        visual_bridge.reused_prompt_tokens != visual_prefix.size() ||
        !(visual_bridge.timings.vision_seconds > 0.0) || visual_bridge.speculative.rounds == 0) {
        std::cerr << "visual MTP bridge did not append the prefix and enter speculative decode: "
                  << "source_outputs=" << bridge_source.generated_token_ids.size()
                  << " reused=" << visual_bridge.reused_prompt_tokens
                  << " vision=" << visual_bridge.timings.vision_seconds
                  << " rounds=" << visual_bridge.speculative.rounds
                  << " fallbacks=" << visual_bridge.speculative.fallback_steps << '\n';
        return 1;
    }
    ninfer::RequestOptions bridge_baseline_options       = bridge_options;
    bridge_baseline_options.execution.allow_prefix_reuse = false;
    const ninfer::GenerationResult visual_bridge_baseline =
        engine.generate(engine.prepare(first_input(image_bytes)), bridge_baseline_options);
    if (visual_bridge.generated_token_ids != visual_bridge_baseline.generated_token_ids) {
        std::cerr << "visual MTP bridge changed greedy output relative to full prefill\n";
        return 1;
    }
    return 0;
}

int verify_loaded_product(const ninfer::Engine& engine) {
    const ninfer::LoadSummary load = engine.load_summary();
    if ((load.target != "qwen3_6_27b" && load.target != "qwen3_8_27b") ||
        (load.weights_id != "groupwise-int" && load.weights_id != "nvfp4") ||
        load.host_to_device_bytes == 0 || load.artifact_bytes_read < load.host_to_device_bytes) {
        std::cerr << "Engine construction has an invalid load summary: target=" << load.target
                  << " weights=" << load.weights_id << '\n';
        return 1;
    }
    const ninfer::MemorySummary memory = engine.memory_summary();
    const bool missing_transient =
        engine.options().enable_vision && memory.request_transient.capacity_bytes == 0;
    if (memory.weights.capacity_bytes == 0 || memory.weights.used_bytes == 0 ||
        memory.weights.used_bytes > memory.weights.capacity_bytes ||
        memory.sequence.capacity_bytes == 0 || memory.sequence.used_bytes == 0 ||
        memory.sequence.used_bytes > memory.sequence.capacity_bytes ||
        memory.workspace.capacity_bytes == 0 || missing_transient ||
        memory.request_transient.used_bytes != 0 || memory.cuda_graph_allowance_bytes == 0) {
        std::cerr << "Engine construction has incomplete materialized backing\n";
        return 1;
    }
    return 0;
}

} // namespace

int exercise_artifact(const char* artifact) {
    int vision_rc = 0;
    {
        ninfer::Engine engine(engine_options(artifact));
        if (const int result = verify_loaded_product(engine); result != 0) { return result; }
        if (const int result = exercise_max_tokens_chat_followup(engine); result != 0) {
            return result;
        }
        if (const int result = exercise_cancel_retain_rollback(engine); result != 0) {
            return result;
        }
        if (const int result = exercise_prefill_cancel_keeps_checkpoint(engine); result != 0) {
            return result;
        }
        if (engine.load_summary().target == "qwen3_6_27b") {
            if (const int result = exercise_registered_frontend(engine); result != 0) {
                return result;
            }
        } else {
            std::cerr << "skip registered frontend goldens for " << engine.load_summary().target
                      << '\n';
        }
        if (const int result = exercise_full_prefill_chunk(engine); result != 0) { return result; }
        if (const int result = exercise_prefix(engine); result != 0) { return result; }
        if (const int result = exercise_rewrite_checkpoints(engine); result != 0) { return result; }
        if (const int result = exercise_first_prompt_cancel_aborts(engine); result != 0) {
            return result;
        }
        if (const int result = exercise_append_cancel_keeps_turn_rollback(engine); result != 0) {
            return result;
        }
        if (engine.load_summary().target == "qwen3_6_27b") {
            vision_rc = exercise_vision(engine);
        } else {
            std::cerr << "skip vision greedy reuse on " << engine.load_summary().target
                      << " (RestoreTurnCheckpoint vs FullReset diverges)\n";
        }
    }

    std::cerr << "ladder first-prompt cancel\n";
    if (const int result = exercise_cancel_ladder_heads(artifact); result != 0) { return result; }
    return vision_rc;
}

int main() {
    const char* groupwise = std::getenv("NINFER_QWEN3_6_27B_WEIGHTS");
    const char* nvfp4     = std::getenv("NINFER_QWEN3_6_27B_NVFP4_WEIGHTS");
    const char* dflash    = std::getenv("NINFER_QWEN3_8_27B_NVFP4_DFLASH_WEIGHTS");
    if ((groupwise == nullptr || *groupwise == '\0') && (nvfp4 == nullptr || *nvfp4 == '\0') &&
        (dflash == nullptr || *dflash == '\0')) {
        std::cout << "skip: set NINFER_QWEN3_6_27B_WEIGHTS, "
                     "NINFER_QWEN3_6_27B_NVFP4_WEIGHTS, or "
                     "NINFER_QWEN3_8_27B_NVFP4_DFLASH_WEIGHTS\n";
        return 77;
    }
    if (groupwise != nullptr && *groupwise != '\0') {
        if (const int result = exercise_artifact(groupwise); result != 0) { return result; }
    }
    if (nvfp4 != nullptr && *nvfp4 != '\0') {
        if (const int result = exercise_artifact(nvfp4); result != 0) { return result; }
    }
    if (dflash != nullptr && *dflash != '\0') {
        ninfer::Engine engine(dflash_engine_options(dflash));
        if (const int result = verify_loaded_product(engine); result != 0) { return result; }
        std::cerr << "dflash cancel retain\n";
        if (const int result = exercise_cancel_retain_rollback(engine); result != 0) {
            return result;
        }
        if (const int result = exercise_prefill_cancel_keeps_checkpoint(engine); result != 0) {
            return result;
        }
        if (const int result = exercise_append_cancel_keeps_turn_rollback(engine); result != 0) {
            return result;
        }
    }
    std::cout << "ok\n";
    return 0;
}
