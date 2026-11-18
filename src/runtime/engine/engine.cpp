#include "ninfer/engine.h"

#include "core/device.h"
#include "runtime/contract/sampling.h"
#include "runtime/contract/types.h"
#include "runtime/engine/concurrent_executor.h"
#include "targets/qwen3_6/impl/frontend/encoded_history_cache.h"
#include "targets/registry.h"
#include "targets/startup_options.h"

#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include <cstdint>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>

namespace ninfer {
namespace {

runtime::ResolvedRequestOptions resolve_request_options(const ModelSamplingDefaults& defaults,
                                                        SamplingMode mode, RequestOptions options) {
    runtime::ResolvedRequestOptions resolved;
    resolved.execution.sampling =
        runtime::resolve_sampling(defaults, mode, options.execution.sampling);
    resolved.execution.requested_output_tokens    = options.execution.requested_output_tokens;
    resolved.execution.allow_prefix_reuse         = options.execution.allow_prefix_reuse;
    resolved.execution.cache_write                = options.execution.cache_write;
    resolved.execution.capture_context_checkpoint = options.execution.capture_context_checkpoint;
    if (options.output.top_logprobs && *options.output.top_logprobs > kMaximumTopLogprobs) {
        throw std::invalid_argument("top_logprobs exceeds kMaximumTopLogprobs");
    }
    resolved.execution.token_logprobs = options.output.top_logprobs.has_value();
    resolved.stop                     = std::move(options.stop);
    resolved.output                   = options.output;
    return resolved;
}

EngineOptions checked_startup_options(EngineOptions options) {
    targets::validate_expert_helper_build(options.expert_helper_pci);
    return options;
}

std::string context_capacity_error(std::uint32_t prompt_tokens, std::uint32_t max_context) {
    return "prepared prompt has " + std::to_string(prompt_tokens) +
           " tokens, exceeding Engine max_context " + std::to_string(max_context);
}

} // namespace

class PreparedPrompt::Impl {
public:
    Impl(PromptSummary prompt_summary, double frontend_seconds, SamplingMode mode,
         targets::qwen3_6::PreparedPrompt prepared)
        : summary(prompt_summary), prepare_seconds(frontend_seconds), sampling_mode(mode),
          value(std::move(prepared)) {}

    PromptSummary summary;
    double prepare_seconds     = 0.0;
    SamplingMode sampling_mode = SamplingMode::Thinking;
    targets::qwen3_6::PreparedPrompt value;
};

PreparedPrompt::PreparedPrompt() noexcept                            = default;
PreparedPrompt::~PreparedPrompt()                                    = default;
PreparedPrompt::PreparedPrompt(PreparedPrompt&&) noexcept            = default;
PreparedPrompt& PreparedPrompt::operator=(PreparedPrompt&&) noexcept = default;

PreparedPrompt::PreparedPrompt(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

const PromptSummary& PreparedPrompt::summary() const noexcept {
    if (impl_ == nullptr) {
        static const PromptSummary empty;
        return empty;
    }
    return impl_->summary;
}

std::span<const TokenId> PreparedPrompt::token_ids() const {
    if (impl_ == nullptr) { return {}; }
    return targets::qwen3_6::PreparedPromptAccess::view(impl_->value).token_ids;
}

PreparedPrompt::operator bool() const noexcept { return impl_ != nullptr; }

class GenerationHandle::Impl {
public:
    class Concept {
    public:
        virtual ~Concept() = default;
        virtual GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) = 0;
    };

    template <class Submission>
    class Model final : public Concept {
    public:
        Model(std::shared_ptr<void> keep_alive, Submission submission)
            : keep_alive_(std::move(keep_alive)), submission_(std::move(submission)) {}

        GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) override {
            return submission_.wait(sink, cancellation);
        }

    private:
        std::shared_ptr<void> keep_alive_;
        Submission submission_;
    };

    template <class Submission>
    Impl(std::shared_ptr<void> keep_alive, Submission submission,
         ResolvedSamplingParameters sampling)
        : state_(std::make_unique<Model<Submission>>(std::move(keep_alive), std::move(submission))),
          sampling_(sampling) {}

    GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) {
        return state_->wait(sink, cancellation);
    }

    [[nodiscard]] const ResolvedSamplingParameters& resolved_sampling() const noexcept {
        return sampling_;
    }

private:
    std::unique_ptr<Concept> state_;
    ResolvedSamplingParameters sampling_;
};

GenerationHandle::GenerationHandle() noexcept                              = default;
GenerationHandle::~GenerationHandle()                                      = default;
GenerationHandle::GenerationHandle(GenerationHandle&&) noexcept            = default;
GenerationHandle& GenerationHandle::operator=(GenerationHandle&&) noexcept = default;

GenerationHandle::GenerationHandle(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

GenerationHandle::operator bool() const noexcept { return impl_ != nullptr; }

const ResolvedSamplingParameters& GenerationHandle::resolved_sampling() const noexcept {
    static const ResolvedSamplingParameters empty;
    return impl_ != nullptr ? impl_->resolved_sampling() : empty;
}

GenerationResult GenerationHandle::wait(OutputSink* sink, const CancellationView& cancellation) {
    if (impl_ == nullptr) { throw std::logic_error("GenerationHandle is empty"); }
    std::unique_ptr<Impl> impl = std::move(impl_);
    return impl->wait(sink, cancellation);
}

class Engine::Impl {
public:
    // One registered target's executor Instance and the executor that drives it.
    template <class Instance>
    struct TargetRuntime {
        std::unique_ptr<Instance> instance;
        std::unique_ptr<runtime::ConcurrentExecutor<Instance>> executor;
    };

    using ActiveRuntime = std::variant<TargetRuntime<targets::Qwen3_6_27BInstance>,
                                       TargetRuntime<targets::Qwen3_8FlashNextInstance>>;
    static_assert(std::variant_size_v<ActiveRuntime> == 2);

    explicit Impl(EngineOptions engine_options)
        : options(checked_startup_options(std::move(engine_options))), device(options.device),
          active(start(targets::construct_target(options, device))) {}

    ~Impl() noexcept {
        device.bind_to_current_thread_noexcept();
        // Destroying the executor joins its worker, so no CUDA work is issued after this point.
        for_each_runtime([](auto& target) noexcept { target.executor.reset(); });
        // The final drain only orders teardown of device resources owned by later members; a
        // destructor has no caller to report a sticky device error to, and every allocation is
        // released by its own RAII owner regardless.
        try {
            device.synchronize_all();
            // NOLINTNEXTLINE(bugprone-empty-catch): unreportable teardown drain; see above.
        } catch (const std::exception&) {}
    }

    Impl(const Impl&)            = delete;
    Impl& operator=(const Impl&) = delete;
    Impl(Impl&&)                 = delete;
    Impl& operator=(Impl&&)      = delete;

    // Calls fn(instance, executor) for the active target. Both targets share the Qwen3.6
    // frontend types, so callers see one prompt and output vocabulary.
    template <class Fn>
    decltype(auto) visit(Fn&& fn) const {
        return std::visit(
            [&](const auto& target) -> decltype(auto) {
                if (target.instance == nullptr || target.executor == nullptr) {
                    throw std::logic_error("concurrent Engine executor is unavailable");
                }
                return fn(*target.instance, *target.executor);
            },
            active);
    }

    [[nodiscard]] const targets::qwen3_6::Frontend& frontend() const {
        return visit([](const auto& instance, auto&) -> const targets::qwen3_6::Frontend& {
            return instance.loaded->frontend;
        });
    }

    [[nodiscard]] std::uint32_t capacity() const {
        return visit([](const auto& instance, auto&) { return instance.capacity; });
    }

    // Applies fn to the active runtime without std::visit's bad_variant_access path: the
    // variant is never valueless, because start() constructs it once and nothing assigns it.
    template <class Fn>
    void for_each_runtime(Fn&& fn) noexcept {
        if (auto* target = std::get_if<0>(&active)) { fn(*target); }
        if (auto* target = std::get_if<1>(&active)) { fn(*target); }
    }

    [[nodiscard]] bool available() const {
        return std::visit(
            [](const auto& target) {
                return target.executor != nullptr && target.executor->available();
            },
            active);
    }

    EngineOptions options;
    DeviceContext device;
    LoadSummary load;
    ModelSamplingDefaults sampling_defaults;
    ActiveRuntime active;
    mutable targets::qwen3_6::frontend_internal::EncodedHistoryCache host_encode_cache;

private:
    ActiveRuntime start(targets::ConstructedTarget constructed) {
        load              = std::move(constructed.load);
        sampling_defaults = constructed.sampling_defaults;
        return std::visit(
            [&](auto& instance) -> ActiveRuntime {
                using Instance = typename std::decay_t<decltype(instance)>::element_type;
                TargetRuntime<Instance> target{.instance = std::move(instance), .executor = {}};
                target.executor = std::make_unique<runtime::ConcurrentExecutor<Instance>>(
                    *target.instance, options);
                return target;
            },
            constructed.active);
    }
};

bool Engine::context_checkpoint_capture_available(bool allow_prefix_reuse) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    if (!context_checkpoint_mode_available(allow_prefix_reuse,
                                           impl_->options.speculative.backend)) {
        return false;
    }
    return impl_->visit([](const auto& instance, auto&) {
        using Package = typename std::remove_cvref_t<decltype(instance)>::Package;
        return Package::supports_context_checkpoints;
    });
}

Engine::Engine(EngineOptions options) {
    validate_speculative_backend(options.speculative.backend);
    impl_ = std::make_shared<Impl>(std::move(options));
}

Engine::~Engine()                            = default;
Engine::Engine(Engine&&) noexcept            = default;
Engine& Engine::operator=(Engine&&) noexcept = default;

PreparedPrompt Engine::prepare(PromptInput input) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    const SamplingMode sampling_mode =
        input.options.enable_thinking ? SamplingMode::Thinking : SamplingMode::NonThinking;
    auto prepared = targets::qwen3_6::EncodedHistoryPrepare::prepare(
        impl_->frontend(), std::move(input), impl_->host_encode_cache,
        impl_->options.kv_ram_system_prompts != 0);
    PromptSummary info           = prepared.summary();
    const std::uint32_t capacity = impl_->capacity();
    if (info.prompt_tokens > capacity) {
        throw RequestError(RequestErrorKind::ContextLengthExceeded,
                           context_capacity_error(info.prompt_tokens, capacity));
    }
    const double seconds = prepared.prepare_seconds();
    return PreparedPrompt(
        std::make_unique<PreparedPrompt::Impl>(info, seconds, sampling_mode, std::move(prepared)));
}

PreparedPrompt Engine::prepare_tokens(std::vector<TokenId> token_ids,
                                      bool allow_prefix_identity) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    auto prepared = impl_->frontend().prepare_tokens(std::move(token_ids), allow_prefix_identity);
    PromptSummary info           = prepared.summary();
    const std::uint32_t capacity = impl_->capacity();
    if (info.prompt_tokens > capacity) {
        throw RequestError(RequestErrorKind::ContextLengthExceeded,
                           context_capacity_error(info.prompt_tokens, capacity));
    }
    const double seconds = prepared.prepare_seconds();
    return PreparedPrompt(std::make_unique<PreparedPrompt::Impl>(
        info, seconds, SamplingMode::Thinking, std::move(prepared)));
}

std::uint32_t Engine::count_tokens(PromptInput input) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return targets::qwen3_6::EncodedHistoryPrepare::count_tokens(
        impl_->frontend(), std::move(input), impl_->host_encode_cache);
}

PromptCapabilities Engine::prompt_capabilities() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->frontend().prompt_capabilities();
}

std::string Engine::token_bytes(TokenId token) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::string(impl_->frontend().token_bytes(token));
}

ModelSamplingDefaults Engine::sampling_defaults() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->sampling_defaults;
}

GenerationHandle Engine::submit(PreparedPrompt prompt, RequestOptions options,
                                OutputDelivery delivery,
                                std::chrono::steady_clock::time_point pending_deadline,
                                HostInputLease host_input) {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    if (prompt.impl_ == nullptr) { throw std::invalid_argument("PreparedPrompt is empty"); }

    struct HostInputGuard {
        PreparedPrompt* prompt;
        HostInputLease* lease;

        ~HostInputGuard() {
            if (static_cast<bool>(*lease)) {
                prompt->impl_.reset();
                lease->reset();
            }
        }
    } host_input_guard{&prompt, &host_input};

    runtime::ResolvedRequestOptions resolved_options = resolve_request_options(
        impl_->sampling_defaults, prompt.impl_->sampling_mode, std::move(options));
    const ResolvedSamplingParameters resolved_sampling = resolved_options.execution.sampling;

    const PromptSummary prompt_summary = prompt.impl_->summary;
    if (prompt_summary.prompt_tokens > impl_->options.max_context) {
        throw RequestError(
            RequestErrorKind::ContextLengthExceeded,
            context_capacity_error(prompt_summary.prompt_tokens, impl_->options.max_context));
    }
    const double prepare_seconds = prompt.impl_->prepare_seconds;
    if (!resolved_options.execution.cache_write &&
        resolved_options.execution.capture_context_checkpoint) {
        throw RequestError(RequestErrorKind::CacheWriteConflict,
                           "capture_context_checkpoint=true requires cache_write=true because it "
                           "creates a retained context checkpoint.");
    }
    if (resolved_options.execution.capture_context_checkpoint &&
        !context_checkpoint_capture_available(resolved_options.execution.allow_prefix_reuse)) {
        throw std::invalid_argument(
            "capture_context_checkpoint requires a supported target, prefix reuse and DFlash");
    }
    if (resolved_options.execution.requested_output_tokens == 0) {
        struct ImmediateSubmission {
            GenerationResult result;

            GenerationResult wait(OutputSink*, const CancellationView& cancellation) {
                if (cancellation.requested()) { result.finish_reason = FinishReason::Cancelled; }
                return std::move(result);
            }
        } immediate;

        immediate.result.prompt                  = prompt_summary;
        immediate.result.finish_reason           = FinishReason::OutputLimit;
        immediate.result.timings.prepare_seconds = prepare_seconds;
        immediate.result.timings.total_seconds   = prepare_seconds;
        prompt.impl_.reset();
        host_input.reset();
        return GenerationHandle(std::make_unique<GenerationHandle::Impl>(
            impl_, std::move(immediate), resolved_sampling));
    }

    return impl_->visit([&](const auto&, auto& executor) {
        auto submission = executor.submit(std::move(prompt.impl_->value), prompt_summary,
                                          prepare_seconds, std::move(resolved_options), delivery,
                                          pending_deadline, std::move(host_input));
        return GenerationHandle(std::make_unique<GenerationHandle::Impl>(
            impl_, std::move(submission), resolved_sampling));
    });
}

GenerationResult Engine::generate(PreparedPrompt prompt, RequestOptions options, OutputSink* sink,
                                  const CancellationView& cancellation) {
    const OutputDelivery delivery =
        sink == nullptr ? OutputDelivery::TerminalOnly : OutputDelivery::Streaming;
    return submit(std::move(prompt), std::move(options), delivery).wait(sink, cancellation);
}

ScoreResult Engine::score(PreparedPrompt prompt, ScoreOptions options) {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    if (prompt.impl_ == nullptr) { throw std::invalid_argument("PreparedPrompt is empty"); }
    const PromptSummary prompt_summary = prompt.impl_->summary;
    if (prompt_summary.prompt_tokens > impl_->options.max_context) {
        throw RequestError(
            RequestErrorKind::ContextLengthExceeded,
            context_capacity_error(prompt_summary.prompt_tokens, impl_->options.max_context));
    }
    if (prompt_summary.prompt_tokens < 2) {
        throw std::invalid_argument("score requires at least two prompt tokens");
    }
    if (options.schedule == ScoreSchedule::Decode && prompt_summary.prompt_tokens < 3) {
        throw std::invalid_argument("decode score requires at least three prompt tokens");
    }
    return impl_->visit([&](const auto&, auto& executor) {
        return executor.score(std::move(prompt.impl_->value), options);
    });
}

const EngineOptions& Engine::options() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->options;
}

LoadSummary Engine::load_summary() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->load;
}

MemorySummary Engine::memory_summary() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->visit([](const auto&, auto& executor) { return executor.memory_summary(); });
}

bool Engine::available() const { return impl_ != nullptr && impl_->available(); }

RuntimeStats Engine::runtime_stats() const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return impl_->visit([](const auto&, auto& executor) { return executor.runtime_stats(); });
}

void Engine::reset_memory_peaks() noexcept {
    if (impl_ == nullptr) { return; }
    impl_->for_each_runtime([](auto& target) noexcept {
        if (target.executor != nullptr) { target.executor->reset_memory_peaks(); }
    });
}

} // namespace ninfer
