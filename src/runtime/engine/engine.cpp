#include "ninfer/engine.h"

#include "core/device.h"
#include "runtime/engine/ordinary_executor.h"
#include "runtime/generation/generation_controller.h"
#include "targets/registry.h"

#include <mutex>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

namespace ninfer {

class PreparedPrompt::Impl {
public:
    Impl(PromptSummary prompt_summary, double frontend_seconds,
         targets::qwen3_6::PreparedPrompt prepared)
        : summary(std::move(prompt_summary)), prepare_seconds(frontend_seconds),
          value(std::move(prepared)) {}

    PromptSummary summary;
    double prepare_seconds = 0.0;
    targets::qwen3_6::PreparedPrompt value;
};

PreparedPrompt::PreparedPrompt() noexcept                            = default;
PreparedPrompt::~PreparedPrompt()                                    = default;
PreparedPrompt::PreparedPrompt(PreparedPrompt&&) noexcept            = default;
PreparedPrompt& PreparedPrompt::operator=(PreparedPrompt&&) noexcept = default;

PreparedPrompt::PreparedPrompt(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

const PromptSummary& PreparedPrompt::summary() const noexcept {
    static const PromptSummary empty;
    return impl_ != nullptr ? impl_->summary : empty;
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
    Impl(std::shared_ptr<void> keep_alive, Submission submission)
        : state_(
              std::make_unique<Model<Submission>>(std::move(keep_alive), std::move(submission))) {}

    GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) {
        return state_->wait(sink, cancellation);
    }

private:
    std::unique_ptr<Concept> state_;
};

GenerationHandle::GenerationHandle() noexcept                              = default;
GenerationHandle::~GenerationHandle()                                      = default;
GenerationHandle::GenerationHandle(GenerationHandle&&) noexcept            = default;
GenerationHandle& GenerationHandle::operator=(GenerationHandle&&) noexcept = default;

GenerationHandle::GenerationHandle(std::unique_ptr<Impl> impl) noexcept : impl_(std::move(impl)) {}

GenerationHandle::operator bool() const noexcept { return impl_ != nullptr; }

GenerationResult GenerationHandle::wait(OutputSink* sink, const CancellationView& cancellation) {
    if (impl_ == nullptr) { throw std::logic_error("GenerationHandle is empty"); }
    std::unique_ptr<Impl> impl = std::move(impl_);
    return impl->wait(sink, cancellation);
}

class Engine::Impl {
public:
    using Executor27 = runtime::OrdinaryExecutor<targets::Qwen3_6_27BInstance>;
    using Executor35 = runtime::OrdinaryExecutor<targets::Qwen3_6_35BA3BInstance>;
    using Executor =
        std::variant<std::monostate, std::unique_ptr<Executor27>, std::unique_ptr<Executor35>>;

    explicit Impl(EngineOptions engine_options)
        : options(std::move(engine_options)), device(options.device) {
        auto constructed = targets::construct_target(options, device);
        active           = std::move(constructed.active);
        load             = std::move(constructed.load);
        if (options.speculative.backend == SpeculativeBackend::None) {
            executor = std::visit(
                [&](auto& target_ptr) -> Executor {
                    using Instance =
                        typename std::remove_reference_t<decltype(target_ptr)>::element_type;
                    if constexpr (std::is_same_v<Instance, targets::Qwen3_6_27BInstance>) {
                        return std::make_unique<Executor27>(*target_ptr, options);
                    } else {
                        return std::make_unique<Executor35>(*target_ptr, options);
                    }
                },
                active);
        }
    }

    ~Impl() noexcept {
        executor.emplace<std::monostate>();
        try {
            device.synchronize();
        } catch (...) {}
    }

    GenerationResult run_legacy(std::unique_ptr<PreparedPrompt::Impl> prompt,
                                RequestOptions request_options, HostInputLease host_input,
                                OutputSink* sink, const CancellationView& cancellation) {
        std::scoped_lock lock(legacy_generation_mutex);
        return std::visit(
            [&](auto& target_ptr) -> GenerationResult {
                if (target_ptr == nullptr) {
                    throw std::logic_error("Engine target is not active");
                }
                if (prompt->summary.prompt_tokens > target_ptr->capacity) {
                    throw RequestError(RequestErrorKind::ContextLengthExceeded,
                                       "prepared prompt exceeds Engine context capacity");
                }
                auto output = target_ptr->loaded->frontend.make_output_session(
                    prompt->value, request_options.stop, request_options.output);
                auto controller =
                    runtime::run_one(*target_ptr->program, std::move(prompt->value),
                                     std::move(output), target_ptr->request_memory, request_options,
                                     cancellation, sink, std::move(host_input));

                GenerationResult result;
                result.prompt              = prompt->summary;
                result.generated_token_ids = std::move(controller.generated_token_ids);
                result.content             = std::move(controller.content);
                result.reasoning           = std::move(controller.reasoning);
                result.finish_reason       = controller.summary.finish_reason;
                if (controller.summary.begin) {
                    result.reused_prompt_tokens = controller.summary.begin->reused_prompt_tokens;
                    result.timings              = target_ptr->program->generation_timings();
                    result.speculative          = target_ptr->program->speculative_stats();
                }
                result.timings.prepare_seconds = prompt->prepare_seconds;
                if (result.timings.prefill_seconds == 0.0) {
                    result.timings.prefill_seconds = controller.prefill_seconds;
                }
                result.timings.decode_seconds = controller.decode_seconds;
                result.timings.first_token_seconds =
                    prompt->prepare_seconds + controller.first_token_seconds;
                result.timings.total_seconds = prompt->prepare_seconds + controller.total_seconds;
                return result;
            },
            active);
    }

    EngineOptions options;
    DeviceContext device;
    targets::ActiveTarget active;
    LoadSummary load;
    Executor executor;
    mutable std::mutex legacy_generation_mutex;
};

Engine::Engine(EngineOptions options) : impl_(std::make_shared<Impl>(std::move(options))) {}

Engine::~Engine()                            = default;
Engine::Engine(Engine&&) noexcept            = default;
Engine& Engine::operator=(Engine&&) noexcept = default;

PreparedPrompt Engine::prepare(PromptInput input) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [&](const auto& target_ptr) -> PreparedPrompt {
            if (target_ptr == nullptr) { throw std::logic_error("Engine target is not active"); }
            auto prepared      = target_ptr->loaded->frontend.prepare(std::move(input));
            PromptSummary info = prepared.summary();
            if (info.prompt_tokens > target_ptr->capacity) {
                throw RequestError(RequestErrorKind::ContextLengthExceeded,
                                   "prepared prompt exceeds Engine context capacity");
            }
            const double seconds = prepared.prepare_seconds();
            return PreparedPrompt(
                std::make_unique<PreparedPrompt::Impl>(info, seconds, std::move(prepared)));
        },
        impl_->active);
}

PreparedPrompt Engine::prepare_tokens(std::vector<TokenId> token_ids,
                                      bool allow_prefix_identity) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [&](const auto& target_ptr) -> PreparedPrompt {
            if (target_ptr == nullptr) { throw std::logic_error("Engine target is not active"); }
            auto prepared      = target_ptr->loaded->frontend.prepare_tokens(std::move(token_ids),
                                                                             allow_prefix_identity);
            PromptSummary info = prepared.summary();
            if (info.prompt_tokens > target_ptr->capacity) {
                throw RequestError(RequestErrorKind::ContextLengthExceeded,
                                   "prepared prompt exceeds Engine context capacity");
            }
            const double seconds = prepared.prepare_seconds();
            return PreparedPrompt(
                std::make_unique<PreparedPrompt::Impl>(info, seconds, std::move(prepared)));
        },
        impl_->active);
}

std::uint32_t Engine::count_tokens(PromptInput input) const {
    if (impl_ == nullptr) { throw std::logic_error("Engine is moved from"); }
    return std::visit(
        [&](const auto& target_ptr) {
            if (target_ptr == nullptr) { throw std::logic_error("Engine target is not active"); }
            return target_ptr->loaded->frontend.count_tokens(std::move(input));
        },
        impl_->active);
}

GenerationHandle Engine::submit(PreparedPrompt prompt, RequestOptions options,
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

    const PromptSummary prompt_summary = prompt.impl_->summary;
    if (prompt_summary.prompt_tokens > impl_->options.max_context) {
        throw RequestError(RequestErrorKind::ContextLengthExceeded,
                           "prepared prompt exceeds Engine context capacity");
    }
    const double prepare_seconds = prompt.impl_->prepare_seconds;
    if (options.execution.requested_output_tokens == 0) {
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
        return GenerationHandle(
            std::make_unique<GenerationHandle::Impl>(impl_, std::move(immediate)));
    }

    if (impl_->options.speculative.backend == SpeculativeBackend::None) {
        return std::visit(
            [&](auto& executor) -> GenerationHandle {
                using Executor = std::remove_cvref_t<decltype(executor)>;
                if constexpr (std::is_same_v<Executor, std::monostate>) {
                    throw std::logic_error("ordinary Engine executor is unavailable");
                } else {
                    auto submission = executor->submit(
                        std::move(prompt.impl_->value), prompt_summary, prepare_seconds,
                        std::move(options), pending_deadline, std::move(host_input));
                    return GenerationHandle(
                        std::make_unique<GenerationHandle::Impl>(impl_, std::move(submission)));
                }
            },
            impl_->executor);
    }

    struct LegacySubmission {
        std::shared_ptr<Impl> engine;
        HostInputLease host_input;
        std::unique_ptr<PreparedPrompt::Impl> prompt;
        RequestOptions options;

        GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) {
            return engine->run_legacy(std::move(prompt), std::move(options), std::move(host_input),
                                      sink, cancellation);
        }
    } legacy{impl_, std::move(host_input), std::move(prompt.impl_), std::move(options)};

    return GenerationHandle(std::make_unique<GenerationHandle::Impl>(impl_, std::move(legacy)));
}

GenerationResult Engine::generate(PreparedPrompt prompt, RequestOptions options, OutputSink* sink,
                                  const CancellationView& cancellation) {
    return submit(std::move(prompt), std::move(options)).wait(sink, cancellation);
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
    if (impl_->options.speculative.backend == SpeculativeBackend::None) {
        return std::visit(
            [](const auto& executor) -> MemorySummary {
                using Executor = std::remove_cvref_t<decltype(executor)>;
                if constexpr (std::is_same_v<Executor, std::monostate>) {
                    throw std::logic_error("ordinary Engine executor is unavailable");
                } else {
                    return executor->memory_summary();
                }
            },
            impl_->executor);
    }
    std::scoped_lock lock(impl_->legacy_generation_mutex);
    return std::visit(
        [](const auto& target_ptr) {
            if (target_ptr == nullptr) { throw std::logic_error("Engine target is not active"); }
            MemorySummary out     = target_ptr->program->memory_summary();
            out.request_transient = target_ptr->request_memory.summary();
            return out;
        },
        impl_->active);
}

void Engine::reset_memory_peaks() noexcept {
    if (impl_ == nullptr) { return; }
    if (impl_->options.speculative.backend == SpeculativeBackend::None) {
        std::visit(
            [](auto& executor) {
                using Executor = std::remove_cvref_t<decltype(executor)>;
                if constexpr (!std::is_same_v<Executor, std::monostate>) {
                    executor->reset_memory_peaks();
                }
            },
            impl_->executor);
        return;
    }
    std::unique_lock lock(impl_->legacy_generation_mutex, std::defer_lock);
    try {
        lock.lock();
    } catch (...) { return; }
    std::visit(
        [](auto& target_ptr) {
            if (target_ptr != nullptr) {
                target_ptr->program->reset_memory_peaks();
                target_ptr->request_memory.reset_peak();
            }
        },
        impl_->active);
}

} // namespace ninfer
