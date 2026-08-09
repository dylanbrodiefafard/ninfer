#pragma once

#include "ninfer/types.h"
#include "runtime/engine/request_memory.h"
#include "runtime/generation/generation_budget.h"
#include "targets/qwen3_6/export/ninfer/targets/qwen3_6/frontend.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace ninfer::runtime {

template <class Instance>
class OrdinaryExecutor {
    struct Request;

public:
    using Package = typename Instance::Package;
    using Program = typename Package::Program;
    using Plan    = typename Package::RequestPlan;
    using Clock   = std::chrono::steady_clock;

    OrdinaryExecutor(Instance& instance, const EngineOptions& options)
        : instance_(instance), max_concurrency_(options.max_concurrency),
          max_outstanding_(static_cast<std::size_t>(options.max_concurrency) +
                           options.max_pending_requests),
          pending_timeout_(std::chrono::milliseconds(options.pending_timeout_ms)) {
        if (max_concurrency_ == 0 || max_concurrency_ > kMaximumConcurrency ||
            options.max_pending_requests == 0 || pending_timeout_.count() <= 0) {
            throw std::invalid_argument("ordinary executor bounds are invalid");
        }
        worker_ = std::thread([this] { worker_loop(); });
    }

    ~OrdinaryExecutor() noexcept {
        {
            std::lock_guard lock(queue_mutex_);
            stopping_ = true;
        }
        queue_cv_.notify_all();
        if (worker_.joinable()) { worker_.join(); }
    }

    OrdinaryExecutor(const OrdinaryExecutor&)            = delete;
    OrdinaryExecutor& operator=(const OrdinaryExecutor&) = delete;

    class Submission {
    public:
        Submission() noexcept = default;

        ~Submission() { reset(); }

        Submission(Submission&& other) noexcept
            : owner_(std::exchange(other.owner_, nullptr)), request_(std::move(other.request_)) {}

        Submission& operator=(Submission&& other) noexcept {
            if (this != &other) {
                reset();
                owner_   = std::exchange(other.owner_, nullptr);
                request_ = std::move(other.request_);
            }
            return *this;
        }

        Submission(const Submission&)            = delete;
        Submission& operator=(const Submission&) = delete;

        [[nodiscard]] explicit operator bool() const noexcept { return request_ != nullptr; }

        GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) {
            if (owner_ == nullptr || request_ == nullptr) {
                throw std::logic_error("ordinary submission is empty");
            }
            OrdinaryExecutor* owner = std::exchange(owner_, nullptr);
            return owner->wait_for_request(std::exchange(request_, nullptr), sink, cancellation);
        }

    private:
        Submission(OrdinaryExecutor& owner, std::shared_ptr<Request> request) noexcept
            : owner_(&owner), request_(std::move(request)) {}

        void reset() noexcept {
            if (owner_ != nullptr && request_ != nullptr) {
                owner_->abandon_request(std::move(request_));
            }
            owner_ = nullptr;
        }

        OrdinaryExecutor* owner_ = nullptr;
        std::shared_ptr<Request> request_;

        friend class OrdinaryExecutor;
    };

    Submission submit(targets::qwen3_6::PreparedPrompt prompt, PromptSummary prompt_summary,
                      double prepare_seconds, RequestOptions options,
                      Clock::time_point pending_deadline = {}) {
        const Clock::time_point submitted = Clock::now();
        if (pending_deadline == Clock::time_point{}) {
            pending_deadline = submitted + pending_timeout_;
        }
        if (submitted >= pending_deadline) {
            throw RequestError(RequestErrorKind::QueueTimeout,
                               "inference request expired before submission");
        }

        {
            std::lock_guard lock(queue_mutex_);
            if (stopping_ || failed_) {
                throw RequestError(RequestErrorKind::Unavailable,
                                   "inference engine is unavailable");
            }
            if (outstanding_ >= max_outstanding_) {
                throw RequestError(RequestErrorKind::Overloaded, "inference request queue is full");
            }
            ++outstanding_;
        }

        std::shared_ptr<Request> request;
        try {
            auto output = instance_.loaded->frontend.make_output_session(prompt, options.stop,
                                                                         options.output);
            request     = std::make_shared<Request>(std::move(prompt), std::move(output),
                                                    prompt_summary, prepare_seconds, std::move(options),
                                                    pending_deadline, submitted);
        } catch (...) {
            release_reserved_capacity();
            throw;
        }

        {
            std::lock_guard lock(queue_mutex_);
            if (stopping_ || failed_) {
                --outstanding_;
                throw RequestError(RequestErrorKind::Unavailable,
                                   "inference engine is unavailable");
            }
            pending_.push_back(request);
        }
        queue_cv_.notify_one();
        return Submission(*this, std::move(request));
    }

    GenerationResult generate(targets::qwen3_6::PreparedPrompt prompt, PromptSummary prompt_summary,
                              double prepare_seconds, RequestOptions options, OutputSink* sink,
                              const CancellationView& cancellation,
                              Clock::time_point pending_deadline = {}) {
        return submit(std::move(prompt), prompt_summary, prepare_seconds, std::move(options),
                      pending_deadline)
            .wait(sink, cancellation);
    }

    [[nodiscard]] MemorySummary memory_summary() const {
        std::scoped_lock lock(execution_mutex_);
        MemorySummary out     = instance_.program->memory_summary();
        out.request_transient = instance_.request_memory.summary();
        return out;
    }

    void reset_memory_peaks() noexcept {
        try {
            std::scoped_lock lock(execution_mutex_);
            instance_.program->reset_memory_peaks();
            instance_.request_memory.reset_peak();
        } catch (...) {}
    }

private:
    GenerationResult wait_for_request(std::shared_ptr<Request> request, OutputSink* sink,
                                      const CancellationView& cancellation) {
        struct ConsumerGuard {
            OrdinaryExecutor* owner;
            std::shared_ptr<Request> request;

            ~ConsumerGuard() { owner->release_consumer(request); }
        } guard{this, request};

        std::exception_ptr caller_error;
        for (;;) {
            std::vector<OutputDelta> events;
            bool done = false;
            {
                std::unique_lock lock(request->mutex);
                request->cv.wait_for(lock, std::chrono::milliseconds(10),
                                     [&] { return request->done || !request->events.empty(); });
                events.swap(request->events);
                done = request->done;
            }

            if (caller_error == nullptr && sink != nullptr) {
                try {
                    for (OutputDelta& event : events) { sink->publish(std::move(event)); }
                } catch (...) {
                    caller_error = std::current_exception();
                    request->cancelled.store(true, std::memory_order_release);
                    queue_cv_.notify_one();
                }
            }

            if (caller_error == nullptr) {
                try {
                    if (cancellation.requested()) {
                        request->cancelled.store(true, std::memory_order_release);
                        queue_cv_.notify_one();
                    }
                } catch (...) {
                    caller_error = std::current_exception();
                    request->cancelled.store(true, std::memory_order_release);
                    queue_cv_.notify_one();
                }
            }
            if (!done) { continue; }

            if (caller_error != nullptr) { std::rethrow_exception(caller_error); }
            std::lock_guard lock(request->mutex);
            if (request->error != nullptr) { std::rethrow_exception(request->error); }
            return std::move(request->result);
        }
    }

    struct Request {
        Request(targets::qwen3_6::PreparedPrompt input,
                targets::qwen3_6::OutputSession output_session, PromptSummary summary,
                double frontend_seconds, RequestOptions request_options, Clock::time_point limit,
                Clock::time_point submit_time)
            : prompt(std::move(input)), output(std::move(output_session)), prompt_summary(summary),
              prepare_seconds(frontend_seconds), options(std::move(request_options)),
              deadline(limit), submitted(submit_time) {}

        targets::qwen3_6::PreparedPrompt prompt;
        targets::qwen3_6::OutputSession output;
        PromptSummary prompt_summary;
        double prepare_seconds = 0.0;
        RequestOptions options;
        Clock::time_point deadline;
        Clock::time_point submitted;
        std::optional<Clock::time_point> first_token;
        std::optional<GenerationBudget> budget;
        std::optional<BeginSummary> begin;
        std::vector<TokenId> generated;
        std::string content;
        std::string reasoning;
        std::optional<std::uint32_t> lane;
        std::atomic<bool> cancelled{false};
        bool decode_ready = false;

        std::mutex mutex;
        std::condition_variable cv;
        std::vector<OutputDelta> events;
        GenerationResult result;
        std::exception_ptr error;
        bool done              = false;
        bool consumer_released = false;
        bool capacity_released = false;
    };

    void append_output(const std::shared_ptr<Request>& request,
                       targets::qwen3_6::PublishedOutput output) {
        if (output.empty()) { return; }
        {
            std::lock_guard lock(request->mutex);
            for (OutputDelta& delta : output) {
                std::string& full = delta.channel == OutputChannel::Reasoning ? request->reasoning
                                                                              : request->content;
                full += delta.text;
                request->events.push_back(std::move(delta));
            }
        }
        request->cv.notify_one();
    }

    void release_reserved_capacity() noexcept {
        std::lock_guard lock(queue_mutex_);
        if (outstanding_ != 0) { --outstanding_; }
    }

    void release_consumer(const std::shared_ptr<Request>& request) noexcept {
        bool release = false;
        {
            std::lock_guard lock(request->mutex);
            request->consumer_released = true;
            if (request->done && !request->capacity_released) {
                request->capacity_released = true;
                release                    = true;
            }
        }
        if (release) { release_reserved_capacity(); }
    }

    void abandon_request(std::shared_ptr<Request> request) noexcept {
        request->cancelled.store(true, std::memory_order_release);
        queue_cv_.notify_one();
        release_consumer(request);
    }

    bool mark_completed(const std::shared_ptr<Request>& request) noexcept {
        bool release = false;
        {
            std::lock_guard lock(request->mutex);
            if (request->consumer_released && !request->capacity_released) {
                request->capacity_released = true;
                release                    = true;
            }
        }
        return release;
    }

    void complete_error(const std::shared_ptr<Request>& request, std::exception_ptr error) {
        {
            std::lock_guard lock(request->mutex);
            if (request->done) { return; }
            request->error = std::move(error);
            request->done  = true;
        }
        if (mark_completed(request)) { release_reserved_capacity(); }
        request->cv.notify_one();
    }

    void complete_success(const std::shared_ptr<Request>& request, FinishReason reason) {
        GenerationResult result;
        result.prompt                  = request->prompt_summary;
        result.generated_token_ids     = std::move(request->generated);
        result.content                 = std::move(request->content);
        result.reasoning               = std::move(request->reasoning);
        result.finish_reason           = reason;
        result.timings.prepare_seconds = request->prepare_seconds;
        if (request->begin) { result.reused_prompt_tokens = request->begin->reused_prompt_tokens; }
        if (request->lane) {
            result.timings = instance_.program->generation_timings_lane(*request->lane);
            result.timings.prepare_seconds = request->prepare_seconds;
        }
        if (request->first_token) {
            result.timings.first_token_seconds =
                request->prepare_seconds +
                std::chrono::duration<double>(*request->first_token - request->submitted).count();
        }
        result.timings.total_seconds =
            request->prepare_seconds +
            std::chrono::duration<double>(Clock::now() - request->submitted).count();
        {
            std::lock_guard lock(request->mutex);
            if (request->done) { return; }
            request->result = std::move(result);
            request->done   = true;
        }
        if (mark_completed(request)) { release_reserved_capacity(); }
        request->cv.notify_one();
    }

    void complete_cancelled(const std::shared_ptr<Request>& request) {
        (void)request->output.preview_terminal(FinishReason::Cancelled);
        append_output(request, request->output.commit_preview());
        complete_success(request, FinishReason::Cancelled);
    }

    bool resolve_round(const std::shared_ptr<Request>& request, TokenId token,
                       bool cancel_at_boundary) {
        const std::uint32_t lane = *request->lane;
        if (cancel_at_boundary) {
            (void)request->output.preview_terminal(FinishReason::Cancelled);
            instance_.program->abort_lane(lane);
            append_output(request, request->output.commit_preview());
            complete_success(request, FinishReason::Cancelled);
            return true;
        }

        const std::span<const TokenId> tokens(&token, 1);
        const OutputDecision decision = request->output.preview(
            tokens, request->budget->remaining(), request->budget->limit_reason());
        if (decision.accepted_tokens != 1) {
            throw std::logic_error("ordinary output policy did not accept its licensed token");
        }
        request->generated.push_back(token);
        instance_.program->resolve_pending_lane(lane, 1, decision.finished());
        request->budget->commit(1);
        auto published = request->output.commit_preview();
        if (!request->first_token) { request->first_token = Clock::now(); }
        append_output(request, std::move(published));
        if (decision.finished()) {
            complete_success(request, decision.finish_reason);
            return true;
        }
        return false;
    }

    void invalidate_admission_plans() noexcept {
        planned_request_.reset();
        for (auto& plan : admission_plans_) { plan.reset(); }
    }

    void remove_completed_slot(std::uint32_t lane) {
        slots_[lane].reset();
        invalidate_admission_plans();
    }

    [[nodiscard]] std::array<bool, kMaximumConcurrency> snapshot_cancellations() const noexcept {
        std::array<bool, kMaximumConcurrency> cancelled{};
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] != nullptr) {
                cancelled[lane] = slots_[lane]->cancelled.load(std::memory_order_acquire);
            }
        }
        return cancelled;
    }

    void
    cancel_active_requests(const std::array<bool, kMaximumConcurrency>& cancelled_at_boundary) {
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            const auto& request = slots_[lane];
            if (request == nullptr || !cancelled_at_boundary[lane]) { continue; }
            instance_.program->abort_lane(lane);
            if (prefill_lane_ && *prefill_lane_ == lane) {
                instance_.request_memory.deactivate();
                prefill_lane_.reset();
            }
            complete_cancelled(request);
            remove_completed_slot(lane);
        }
    }

    void expire_pending_requests() {
        std::vector<std::shared_ptr<Request>> cancelled;
        std::vector<std::shared_ptr<Request>> expired;
        {
            std::lock_guard lock(queue_mutex_);
            const auto now = Clock::now();
            for (auto it = pending_.begin(); it != pending_.end();) {
                if ((*it)->cancelled.load(std::memory_order_acquire)) {
                    cancelled.push_back(*it);
                    it = pending_.erase(it);
                } else if (now >= (*it)->deadline) {
                    expired.push_back(*it);
                    it = pending_.erase(it);
                } else {
                    ++it;
                }
            }
        }
        if (!cancelled.empty() || !expired.empty()) { invalidate_admission_plans(); }
        for (const auto& request : cancelled) { complete_cancelled(request); }
        for (const auto& request : expired) {
            complete_error(request, std::make_exception_ptr(RequestError(
                                        RequestErrorKind::QueueTimeout,
                                        "inference request expired while waiting for admission")));
        }
    }

    [[nodiscard]] std::vector<std::uint32_t> ready_lanes() const {
        std::vector<std::uint32_t> lanes;
        lanes.reserve(max_concurrency_);
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] != nullptr && slots_[lane]->decode_ready) { lanes.push_back(lane); }
        }
        return lanes;
    }

    void resolve_prefill_step(const std::shared_ptr<Request>& request,
                              const PrefillStepResult& step) {
        if (!step.complete) { return; }
        if (!request->lane) { throw std::logic_error("completed prefill has no request lane"); }
        if (prefill_lane_ && *request->lane == *prefill_lane_) {
            instance_.request_memory.deactivate();
            prefill_lane_.reset();
        }
        request->begin = step.summary;
        if (step.round.tokens.size() != 1) {
            throw std::logic_error("ordinary prefill did not license exactly one token");
        }
        if (resolve_round(request, step.round.tokens.front(), false)) {
            remove_completed_slot(*request->lane);
        } else {
            request->decode_ready = true;
        }
    }

    void run_prefill_step() {
        if (!prefill_lane_) { throw std::logic_error("no request owns staged prefill"); }
        const std::uint32_t lane = *prefill_lane_;
        const auto request       = slots_[lane];
        if (request == nullptr || request->decode_ready) {
            throw std::logic_error("staged prefill lane has invalid request state");
        }
        const PrefillStepResult step = instance_.program->advance_ordinary_prefill_lane(lane);
        resolve_prefill_step(request, step);
    }

    void plan_fifo_head(const std::shared_ptr<Request>& request) {
        if (planned_request_ == request) { return; }
        invalidate_admission_plans();
        planned_request_ = request;
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] == nullptr) {
                admission_plans_[lane].emplace(instance_.program->plan_request_for_lane(
                    lane, request->prompt, request->options.execution));
            }
        }
    }

    [[nodiscard]] bool plan_can_run_with_current_prefill(const Plan& plan) const noexcept {
        const RequestPlanSummary& summary = plan.summary();
        return !prefill_lane_ || summary.reusable_prompt_tokens == summary.prompt_tokens;
    }

    bool try_admit_one() {
        std::shared_ptr<Request> request;
        {
            std::lock_guard lock(queue_mutex_);
            if (pending_.empty()) { return false; }
            request = pending_.front();
        }
        if (request->cancelled.load(std::memory_order_acquire)) { return false; }

        std::optional<std::uint32_t> selected_lane;
        std::uint32_t selected_reuse = 0;
        bool requires_eviction       = false;
        try {
            plan_fifo_head(request);
            for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
                if (!admission_plans_[lane] ||
                    !plan_can_run_with_current_prefill(*admission_plans_[lane])) {
                    continue;
                }
                const Plan& plan          = *admission_plans_[lane];
                const std::uint32_t reuse = plan.summary().reusable_prompt_tokens;
                if (instance_.program->can_admit_lane(lane, plan) &&
                    (!selected_lane || reuse > selected_reuse)) {
                    selected_lane  = lane;
                    selected_reuse = reuse;
                }
            }

            if (!selected_lane) {
                for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
                    if (!admission_plans_[lane] ||
                        !plan_can_run_with_current_prefill(*admission_plans_[lane])) {
                        continue;
                    }
                    const Plan& plan          = *admission_plans_[lane];
                    const std::uint32_t reuse = plan.summary().reusable_prompt_tokens;
                    if (instance_.program->can_admit_lane_after_retained_eviction(lane, plan) &&
                        (!selected_lane || reuse > selected_reuse)) {
                        selected_lane     = lane;
                        selected_reuse    = reuse;
                        requires_eviction = true;
                    }
                }
            }
        } catch (...) {
            {
                std::lock_guard lock(queue_mutex_);
                if (!pending_.empty() && pending_.front() == request) { pending_.pop_front(); }
            }
            invalidate_admission_plans();
            complete_error(request, std::current_exception());
            return true;
        }
        if (Clock::now() >= request->deadline) {
            {
                std::lock_guard lock(queue_mutex_);
                if (!pending_.empty() && pending_.front() == request) { pending_.pop_front(); }
            }
            invalidate_admission_plans();
            complete_error(request, std::make_exception_ptr(RequestError(
                                        RequestErrorKind::QueueTimeout,
                                        "inference request expired while waiting for admission")));
            return true;
        }
        if (request->cancelled.load(std::memory_order_acquire)) {
            {
                std::lock_guard lock(queue_mutex_);
                if (!pending_.empty() && pending_.front() == request) { pending_.pop_front(); }
            }
            invalidate_admission_plans();
            complete_cancelled(request);
            return true;
        }
        if (!selected_lane) { return false; }

        const std::uint32_t lane = *selected_lane;
        if (requires_eviction) {
            for (std::uint32_t retained_lane = 0;
                 retained_lane < max_concurrency_ &&
                 !instance_.program->can_admit_lane(lane, *admission_plans_[lane]);
                 ++retained_lane) {
                if (retained_lane != lane && slots_[retained_lane] == nullptr &&
                    instance_.program->has_retained_lane(retained_lane)) {
                    instance_.program->evict_retained_lane(retained_lane);
                    admission_plans_[retained_lane].reset();
                }
            }
            if (!instance_.program->can_admit_lane(lane, *admission_plans_[lane])) {
                throw std::logic_error("retained eviction did not make admission feasible");
            }
        }

        Plan selected_plan = std::move(*admission_plans_[lane]);
        invalidate_admission_plans();

        {
            std::lock_guard lock(queue_mutex_);
            if (pending_.empty() || pending_.front() != request) { return false; }
            pending_.pop_front();
        }
        if (request->cancelled.load(std::memory_order_acquire)) {
            complete_cancelled(request);
            return true;
        }

        const RequestPlanSummary summary = selected_plan.summary();
        const bool needs_prefill         = summary.reusable_prompt_tokens < summary.prompt_tokens;
        bool target_started              = false;
        try {
            request->budget.emplace(summary.effective_output_tokens,
                                    summary.effective_limit_reason);
            request->generated.reserve(summary.effective_output_tokens);
            request->lane = lane;
            slots_[lane]  = request;

            TransientRegion transient;
            if (needs_prefill) {
                instance_.request_memory.activate(summary.transient_bytes,
                                                  summary.transient_alignment);
                prefill_lane_ = lane;
                transient     = instance_.request_memory.region();
            }
            target_started                = true;
            const PrefillStepResult first = instance_.program->start_ordinary_prefill_lane(
                lane, std::move(request->prompt), std::move(selected_plan), transient);
            if (!first.complete && (!prefill_lane_ || *prefill_lane_ != lane)) {
                throw std::logic_error("partial prefill did not retain its execution owner");
            }
            resolve_prefill_step(request, first);
        } catch (...) {
            const std::exception_ptr error = std::current_exception();
            if (target_started) { instance_.program->abort_lane(lane); }
            if (prefill_lane_ && *prefill_lane_ == lane) {
                instance_.request_memory.deactivate();
                prefill_lane_.reset();
            }
            slots_[lane].reset();
            invalidate_admission_plans();
            complete_error(request, error);
            throw;
        }
        return true;
    }

    void run_decode_round(const std::vector<std::uint32_t>& lanes,
                          const std::array<bool, kMaximumConcurrency>& cancelled_at_boundary) {
        std::array<RoundBudget, kMaximumConcurrency> budgets{};
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            budgets[row] = slots_[lanes[row]]->budget->round_budget();
        }
        const BatchedGeneratedRound round = instance_.program->decode_ordinary_batch(
            lanes, std::span<const RoundBudget>(budgets.data(), lanes.size()));
        if (round.tokens.size() != lanes.size()) {
            throw std::logic_error("ordinary batch returned an invalid row count");
        }
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            const std::uint32_t lane = lanes[row];
            const auto request       = slots_[lane];
            if (resolve_round(request, round.tokens[row], cancelled_at_boundary[lane])) {
                remove_completed_slot(lane);
            }
        }
    }

    void fail_all(std::exception_ptr error) noexcept {
        std::vector<std::shared_ptr<Request>> pending;
        {
            std::lock_guard lock(queue_mutex_);
            failed_ = true;
            pending.assign(pending_.begin(), pending_.end());
            pending_.clear();
        }
        if (prefill_lane_) {
            instance_.request_memory.deactivate();
            prefill_lane_.reset();
        }
        invalidate_admission_plans();
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] != nullptr) {
                instance_.program->abort_lane(lane);
                complete_error(slots_[lane], error);
                slots_[lane].reset();
            }
        }
        for (const auto& request : pending) { complete_error(request, error); }
    }

    void worker_loop() noexcept {
        bool previous_unit_was_decode = false;
        for (;;) {
            {
                std::unique_lock lock(queue_mutex_);
                if (!stopping_ && pending_.empty()) {
                    bool active = false;
                    for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
                        active = active || slots_[lane] != nullptr;
                    }
                    if (!active) {
                        queue_cv_.wait(lock, [&] { return stopping_ || !pending_.empty(); });
                    }
                }
                if (stopping_) {
                    lock.unlock();
                    fail_all(std::make_exception_ptr(RequestError(
                        RequestErrorKind::Unavailable, "inference engine is shutting down")));
                    return;
                }
            }

            try {
                std::scoped_lock execution_lock(execution_mutex_);
                expire_pending_requests();
                const auto cancelled_at_boundary = snapshot_cancellations();
                cancel_active_requests(cancelled_at_boundary);
                std::vector<std::uint32_t> lanes = ready_lanes();

                bool have_pending = false;
                {
                    std::lock_guard lock(queue_mutex_);
                    have_pending = !pending_.empty();
                }
                if (prefill_lane_) {
                    if (have_pending && try_admit_one()) {
                        previous_unit_was_decode = false;
                        continue;
                    }
                    if (!lanes.empty() && !previous_unit_was_decode) {
                        run_decode_round(lanes, cancelled_at_boundary);
                        previous_unit_was_decode = true;
                    } else {
                        run_prefill_step();
                        previous_unit_was_decode = false;
                    }
                    continue;
                }

                if (have_pending && (lanes.empty() || previous_unit_was_decode)) {
                    if (try_admit_one()) {
                        previous_unit_was_decode = false;
                        continue;
                    }
                }

                lanes = ready_lanes();
                if (!lanes.empty()) {
                    run_decode_round(lanes, cancelled_at_boundary);
                    previous_unit_was_decode = true;
                    continue;
                }
                if (have_pending && try_admit_one()) {
                    previous_unit_was_decode = false;
                    continue;
                }
            } catch (...) {
                fail_all(std::current_exception());
                return;
            }
        }
    }

    Instance& instance_;
    const std::uint32_t max_concurrency_;
    const std::size_t max_outstanding_;
    const std::chrono::milliseconds pending_timeout_;

    mutable std::mutex execution_mutex_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::deque<std::shared_ptr<Request>> pending_;
    std::size_t outstanding_ = 0;
    std::array<std::shared_ptr<Request>, kMaximumConcurrency> slots_{};
    std::optional<std::uint32_t> prefill_lane_;
    std::shared_ptr<Request> planned_request_;
    std::array<std::optional<Plan>, kMaximumConcurrency> admission_plans_{};
    bool stopping_ = false;
    bool failed_   = false;
    std::thread worker_;
};

} // namespace ninfer::runtime
