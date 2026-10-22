#include "ninfer/engine.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

ninfer::RequestOptions request_options() {
    ninfer::RequestOptions options;
    options.execution.sampling.temperature = 0.0F;
    options.stop.include_model_defaults    = false;
    return options;
}

void require_same_output(const ninfer::GenerationResult& actual,
                         const ninfer::GenerationResult& expected) {
    require(actual.generated_token_ids == expected.generated_token_ids &&
                actual.finish_reason == expected.finish_reason,
            "system restore changed cold greedy output");
}

struct CancelAfterOutput : ninfer::OutputSink {
    bool requested = false;

    void publish(ninfer::OutputDelta) override { requested = true; }
};

// Exercise boundary capture during interleaved prefill/decode, cancellation, and subsequent
// reusable restores through the public Engine. The exact GPU cancellation phase is not forced.
int qualify_system_prompts(const std::filesystem::path& artifact, std::string_view backend,
                           std::uint32_t concurrency) {
    require(backend == "none" || backend == "dflash",
            "system cache backend must be none or dflash");
    ninfer::EngineOptions options;
    options.artifact_path        = artifact;
    options.max_context          = 4096;
    options.kv_capacity          = ninfer::KvCapacityPolicy::explicit_capacity(4096 * concurrency);
    options.max_concurrency      = concurrency;
    options.prefill_chunk        = 512;
    options.mixed_forward        = 512;
    options.mixed_forward_rounds = 1;
    options.mixed_forward_min_tokens = 0;
    options.kv_ram_capacity_bytes    = 2ULL << 30;
    options.kv_ram_system_prompts    = 2;
    options.generation_recovery      = false;
    options.enable_vision            = false;
    options.kv_cache                 = ninfer::KvCacheStorage::Nvfp4;
    if (backend != "none") {
        options.speculative.backend       = ninfer::SpeculativeBackend::DFlash;
        options.speculative.draft_tokens  = 7;
        options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    }
    std::string system = "You are a concise technical assistant. Follow the user's instruction.\n";
    for (unsigned i = 0; i < 80; ++i) {
        system += "Archive entry " + std::to_string(i) +
                  ": cedar trees grow beside the river. This is background context.\n";
    }
    const auto input = [&](std::string user, unsigned system_variant = 0) {
        ninfer::PromptInput value;
        value.options.enable_thinking = false;
        ninfer::ChatMessage instruction;
        instruction.role = ninfer::ChatRole::System;
        instruction.parts.push_back(ninfer::MessagePart{
            .text = (system_variant != 0 ? "Additional instruction " +
                                               std::to_string(system_variant) + ": be precise.\n"
                                         : "") +
                    system});
        value.messages.push_back(std::move(instruction));
        ninfer::ChatMessage message;
        message.parts.push_back(ninfer::MessagePart{.text = std::move(user)});
        value.messages.push_back(std::move(message));
        return value;
    };
    auto request                              = request_options();
    request.execution.cache_write             = true;
    request.execution.requested_output_tokens = 32;
    const std::string question                = "Name the capital of France.";
    ninfer::GenerationResult cold;
    {
        ninfer::Engine engine(options);
        cold = engine.generate(engine.prepare(input(question)), request);
        require(cold.reused_prompt_tokens == 0, "fresh Engine unexpectedly reused a prefix");
    }
    ninfer::Engine engine(options);
    (void)engine.generate(engine.prepare(input("What is three plus five?")), request);
    auto warm = engine.generate(engine.prepare(input(question)), request);
    require_same_output(warm, cold);
    require(warm.system_prompt_cache_hit_tokens > 0 &&
                warm.system_prompt_cache_hit_tokens == warm.reused_prompt_tokens,
            "sequential different-user request did not restore the system snapshot");
    const auto boundary              = warm.reused_prompt_tokens;
    auto disposable                  = request;
    disposable.execution.cache_write = false;
    const auto before_disposable     = engine.runtime_stats();
    auto branch =
        engine.generate(engine.prepare(input("Write the integer after nine.")), disposable);
    require(branch.system_prompt_cache_hit_tokens == boundary &&
                engine.runtime_stats().kv_ram_system_prompt_used_bytes ==
                    before_disposable.kv_ram_system_prompt_used_bytes &&
                engine.runtime_stats().kv_ram_system_prompt_entries == 1,
            "disposable request failed to reuse without writing system state");
    (void)engine.generate(engine.prepare(input("Write the integer after ten.", 3)), disposable);
    require(engine.runtime_stats().kv_ram_system_prompt_entries == 1,
            "disposable miss captured a system prefix");
    auto no_reuse                         = request;
    no_reuse.execution.allow_prefix_reuse = false;
    const auto uncached =
        engine.generate(engine.prepare(input("Write the integer after eleven.", 4)), no_reuse);
    require(uncached.reused_prompt_tokens == 0 &&
                engine.runtime_stats().kv_ram_system_prompt_entries == 1,
            "prefix opt-out reused or captured a system prefix");
    // Queue all admissions before the first cold prefill completes. Only the first computes
    // the shared system prefix; later FIFO admissions must refresh their RAM plans.
    auto prompt_a            = engine.prepare(input("What is three plus five?", 1));
    auto prompt_b            = engine.prepare(input(question, 1));
    auto prompt_c            = engine.prepare(input("Give the chemical symbol for oxygen.", 1));
    auto first               = engine.submit(std::move(prompt_a), request);
    auto queued_b            = engine.submit(std::move(prompt_b), request);
    auto queued_c            = engine.submit(std::move(prompt_c), request);
    const auto first_result  = first.wait();
    const auto queued_result = queued_b.wait();
    const auto queued_third  = queued_c.wait();
    require(first_result.system_prompt_cache_hit_tokens == 0 &&
                queued_result.system_prompt_cache_hit_tokens > 0 &&
                queued_third.system_prompt_cache_hit_tokens ==
                    queued_result.system_prompt_cache_hit_tokens,
            "queued first-user variants did not share the newly captured system prefix");
    // Ordinary decode batch shape can change floating-point rounding; queued behavior is
    // checked by exact source/frontier and completion. Exact cache parity is sequential above.
    require(queued_result.prefix_reuse_source == ninfer::PrefixReuseSource::HostRam &&
                queued_third.prefix_reuse_source == ninfer::PrefixReuseSource::HostRam &&
                queued_result.generated_token_ids.size() == 32 &&
                queued_third.generated_token_ids.size() == 32,
            "queued same-system admissions did not complete with RAM reuse");
    const auto captures                           = engine.runtime_stats().kv_ram_captures;
    auto long_decode                              = request;
    long_decode.execution.requested_output_tokens = 128;
    auto decode = engine.submit(engine.prepare(input("Count the integers in order.")), long_decode);
    // This different system is a cold capture. It must stop exactly at the unaligned system
    // boundary while another lane decodes; DFlash takes the mixed-target-forward route.
    auto mixed =
        engine.submit(engine.prepare(input("Give the chemical symbol for oxygen.", 2)), request);
    // The third admission can use the retained snapshot while the earlier chats decode.
    auto active_shared =
        engine.submit(engine.prepare(input("Write the integer after forty-one.")), request);
    const auto mixed_result  = mixed.wait();
    const auto decode_result = decode.wait();
    const auto active_result = active_shared.wait();
    require(active_result.system_prompt_cache_hit_tokens == boundary,
            "new job beside active chats did not reuse the retained system snapshot");
    require(mixed_result.generated_token_ids.size() == 32 &&
                decode_result.generated_token_ids.size() == 128,
            "concurrent system capture failed to complete both requests");
    require(engine.runtime_stats().kv_ram_system_prompt_entries == 2 &&
                engine.runtime_stats().kv_ram_captures > captures,
            "mixed prefill did not retain its second system prefix");
    CancelAfterOutput sink;
    const ninfer::CancellationView cancellation([&] { return sink.requested; });
    const auto cancelled = engine.generate(engine.prepare(input("List the days of the week.")),
                                           long_decode, &sink, cancellation);
    require(sink.requested && cancelled.finish_reason == ninfer::FinishReason::Cancelled,
            "system-cache cancellation was not observed");
    warm = engine.generate(engine.prepare(input(question)), request);
    require_same_output(warm, cold);
    if (concurrency == 8) {
        std::vector<ninfer::GenerationHandle> handles;
        handles.reserve(concurrency);
        for (std::uint32_t lane = 0; lane < concurrency; ++lane) {
            handles.push_back(engine.submit(
                engine.prepare(input("Write a short sentence about item " + std::to_string(lane))),
                request));
        }
        for (auto& handle : handles) {
            const auto result = handle.wait();
            require(result.system_prompt_cache_hit_tokens == boundary &&
                        result.prefix_reuse_source == ninfer::PrefixReuseSource::HostRam &&
                        result.generated_token_ids.size() == 32,
                    "eight-lane wave did not complete with retained system reuse");
        }
    }
    const auto stats = engine.runtime_stats();
    require(stats.kv_ram_system_prompt_entries == 2 && stats.kv_ram_system_prompt_restores >= 3 &&
                stats.kv_cache_fallbacks == 0 &&
                stats.kv_ram_used_bytes <= stats.kv_ram_capacity_bytes,
            "retained system state/budget was lost after concurrent or cancelled work");
    std::cout << "system_prompt_cache: PASS backend=" << backend << " C=" << concurrency
              << " boundary=" << boundary << " entries=" << stats.kv_ram_system_prompt_entries
              << " restores=" << stats.kv_ram_system_prompt_restores
              << " bytes=" << stats.kv_ram_system_prompt_used_bytes
              << " cold_ttft_s=" << cold.timings.first_token_seconds
              << " warm_ttft_s=" << warm.timings.first_token_seconds
              << "; queued same-system admissions, active-chat reuse, exact cold greedy output, "
                 "mixed capture, cancellation, no fallback\n";
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    const char* artifact = std::getenv("NINFER_QWEN3_8_27B_NVFP4_DFLASH_WEIGHTS");
    if (artifact == nullptr || !std::filesystem::is_regular_file(artifact)) {
        std::cout << "skip: set NINFER_QWEN3_8_27B_NVFP4_DFLASH_WEIGHTS\n";
        return 77;
    }
    try {
        require(argc <= 3 && (argc != 3 || std::string_view(argv[2]) == "--c8"),
                "usage: system_prompt_cache_real_test [none|dflash] [--c8]");
        return qualify_system_prompts(artifact, argc >= 2 ? argv[1] : "none", argc == 3 ? 8U : 2U);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
