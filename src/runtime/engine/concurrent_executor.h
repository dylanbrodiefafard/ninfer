#pragma once

// Small fixed-capacity request scheduling and batched decode execution for every backend.

#include "ninfer/types.h"
#include "runtime/contract/types.h"
#include "runtime/engine/admission_policy.h"
#include "runtime/engine/request_memory.h"
#include "runtime/generation/generation_budget.h"
#include "targets/qwen3_6/export/ninfer/targets/qwen3_6/frontend.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace ninfer::runtime {

template <class Instance>
class ConcurrentExecutor {
    struct Request;

public:
    using Package  = typename Instance::Package;
    using Program  = typename Package::Program;
    using BasePlan = typename Package::RequestBasePlan;
    using Plan     = typename Package::RequestPlan;
    using Clock    = std::chrono::steady_clock;

    ConcurrentExecutor(Instance& instance, const EngineOptions& options)
        : instance_(instance), max_concurrency_(options.max_concurrency),
          max_outstanding_(static_cast<std::size_t>(options.max_concurrency) +
                           options.max_pending_requests),
          pending_timeout_(std::chrono::milliseconds(options.pending_timeout_ms)),
          admission_capacity_(instance.program->admission_capacity()),
          load_progress_(options.load_progress) {
        if (max_concurrency_ == 0 || max_concurrency_ > kMaximumConcurrency ||
            options.max_pending_requests == 0 || pending_timeout_.count() <= 0) {
            throw std::invalid_argument("concurrent executor bounds are invalid");
        }
        if (admission_capacity_.active_lanes != max_concurrency_ ||
            admission_capacity_.main_kv_pages == 0) {
            throw std::logic_error("target admission capacity does not match the Engine");
        }
        worker_ = std::thread([this] { worker_loop(); });
    }

    ~ConcurrentExecutor() noexcept {
        {
            std::lock_guard lock(queue_mutex_);
            stopping_ = true;
        }
        signal_control(true);
        if (worker_.joinable()) { worker_.join(); }
    }

    ConcurrentExecutor(const ConcurrentExecutor&)            = delete;
    ConcurrentExecutor& operator=(const ConcurrentExecutor&) = delete;

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

        GenerationResult wait(OutputSink* sink, const CancellationView& cancellation) {
            if (owner_ == nullptr || request_ == nullptr) {
                throw std::logic_error("concurrent submission is empty");
            }
            if (sink != nullptr && request_->delivery != OutputDelivery::Streaming) {
                throw std::logic_error("terminal-only submission cannot consume an OutputSink");
            }
            ConcurrentExecutor* owner = std::exchange(owner_, nullptr);
            return owner->wait_for_request(std::exchange(request_, nullptr), sink, cancellation);
        }

    private:
        Submission(ConcurrentExecutor& owner, std::shared_ptr<Request> request) noexcept
            : owner_(&owner), request_(std::move(request)) {}

        void reset() noexcept {
            if (owner_ != nullptr && request_ != nullptr) {
                owner_->abandon_request(std::move(request_));
            }
            owner_ = nullptr;
        }

        ConcurrentExecutor* owner_ = nullptr;
        std::shared_ptr<Request> request_;

        friend class ConcurrentExecutor;
    };

    Submission submit(targets::qwen3_6::PreparedPrompt prompt, PromptSummary prompt_summary,
                      double prepare_seconds, ResolvedRequestOptions options,
                      OutputDelivery delivery,
                      Clock::time_point pending_deadline = {}, HostInputLease host_input = {}) {
        const Clock::time_point submitted = Clock::now();
        if (pending_deadline == Clock::time_point{}) {
            pending_deadline = submitted + pending_timeout_;
        }
        if (submitted >= pending_deadline) {
            throw RequestError(RequestErrorKind::QueueTimeout,
                               "inference request expired before submission");
        }

        std::uint64_t request_id = 0;
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
            request_id = next_request_id_++;
        }

        std::shared_ptr<Request> request;
        try {
            auto output = instance_.loaded->frontend.make_output_session(prompt, options.stop,
                                                                         options.output);
            if (!output.model_stop_tokens_allowed()) {
                const auto& stop_ids = instance_.loaded->frontend.default_stop_policy().token_ids;
                if (stop_ids.size() > options.execution.suppressed_token_ids.size()) {
                    throw std::logic_error("model defines too many default stop tokens");
                }
                options.execution.suppressed_token_count =
                    static_cast<std::uint32_t>(stop_ids.size());
                std::copy(stop_ids.begin(), stop_ids.end(),
                          options.execution.suppressed_token_ids.begin());
            }
            request = std::make_shared<Request>(request_id, std::move(prompt), std::move(output),
                                                prompt_summary, prepare_seconds, std::move(options),
                                                delivery, pending_deadline, submitted,
                                                std::move(host_input));
            request->base_plan.emplace(
                instance_.program->plan_request_base(request->prompt, request->options.execution));
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
            published_waiting_requests_.store(static_cast<std::uint32_t>(pending_.size()),
                                              std::memory_order_relaxed);
        }
        signal_control();
        return Submission(*this, std::move(request));
    }

    [[nodiscard]] MemorySummary memory_summary() const {
        std::scoped_lock lock(execution_mutex_);
        MemorySummary out                      = instance_.program->memory_summary();
        out.request_transient                  = instance_.request_memory.summary();
        const KvCapacityResolution& resolution = instance_.kv_capacity_resolution;
        out.kv_capacity_mode                   = resolution.mode;
        out.kv_capacity_page_groups            = resolution.main_page_groups;
        out.kv_capacity_max_page_groups        = resolution.maximum_main_page_groups;
        out.minimum_runtime_reservation_bytes  = resolution.minimum_runtime_reservation_bytes;
        out.kv_capacity_increment_bytes        = resolution.bytes_per_additional_main_page_group;
        out.runtime_reservation_bytes          = resolution.runtime_reservation_bytes;
        out.available_after_weights_bytes      = resolution.available_after_weights_bytes;
        out.available_after_startup_bytes      = resolution.available_after_startup_bytes;
        out.kv_capacity_headroom_bytes         = resolution.automatic_headroom_bytes;
        out.planned_slack_bytes                = resolution.planned_slack_bytes;
        return out;
    }

    [[nodiscard]] ScoreResult score(targets::qwen3_6::PreparedPrompt prompt,
                                    ScoreOptions options = {}) {
        std::scoped_lock execution_lock(execution_mutex_);
        {
            std::lock_guard lock(queue_mutex_);
            if (stopping_ || failed_) {
                throw RequestError(RequestErrorKind::Unavailable,
                                   "inference engine is unavailable");
            }
            if (outstanding_ != 0 || copy_hold_ || prefill_lane_) {
                throw RequestError(RequestErrorKind::Overloaded, "score requires an idle Engine");
            }
            for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
                if (slots_[lane] != nullptr) {
                    throw RequestError(RequestErrorKind::Overloaded,
                                       "score requires an idle Engine");
                }
            }
        }

        ResolvedExecutionOptions execution;
        execution.sampling.temperature    = 0.0F;
        execution.requested_output_tokens = std::max(1u, prompt.summary().prompt_tokens);
        execution.allow_prefix_reuse      = false;
        BasePlan base                     = instance_.program->plan_request_base(prompt, execution);
        Plan plan                         = instance_.program->plan_request_for_lane(0, prompt, base);
        const RequestPlanSummary summary = plan.summary();
        instance_.request_memory.activate(summary.transient_bytes, summary.transient_alignment);
        try {
            ScoreResult result =
                instance_.program->score(std::move(prompt), std::move(plan),
                                         instance_.request_memory.region(), options);
            instance_.request_memory.deactivate();
            return result;
        } catch (...) {
            instance_.request_memory.deactivate();
            throw;
        }
    }

    [[nodiscard]] RuntimeStats runtime_stats() const {
        std::lock_guard lock(stats_mutex_);
        RuntimeStats out              = published_stats_;
        out.waiting_requests          = published_waiting_requests_.load(std::memory_order_relaxed);
        out.computed_prefill_tokens   = published_computed_prefill_tokens_.load(
            std::memory_order_relaxed);
        out.committed_decode_tokens   = published_committed_decode_tokens_.load(
            std::memory_order_relaxed);
        out.decode_rounds             = published_decode_rounds_.load(std::memory_order_relaxed);
        out.decode_row_rounds         = published_decode_row_rounds_.load(std::memory_order_relaxed);
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
    void publish_hot_runtime_counters() noexcept {
        published_computed_prefill_tokens_.store(cumulative_stats_.computed_prefill_tokens,
                                                  std::memory_order_relaxed);
        published_committed_decode_tokens_.store(cumulative_stats_.committed_decode_tokens,
                                                  std::memory_order_relaxed);
        published_decode_rounds_.store(cumulative_stats_.decode_rounds,
                                       std::memory_order_relaxed);
        published_decode_row_rounds_.store(cumulative_stats_.decode_row_rounds,
                                           std::memory_order_relaxed);
    }

    void publish_runtime_stats() {
        RuntimeStats snapshot = cumulative_stats_;
        {
            std::lock_guard lock(queue_mutex_);
            snapshot.waiting_requests = static_cast<std::uint32_t>(pending_.size());
            published_waiting_requests_.store(snapshot.waiting_requests, std::memory_order_relaxed);
        }
        snapshot.prefilling_requests = prefill_lane_.has_value() ? 1U : 0U;
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] == nullptr) { continue; }
            ++snapshot.running_requests;
            if (slots_[lane]->decode_ready) { ++snapshot.decode_ready_requests; }
        }
        const auto ram                    = instance_.program->kv_ram_snapshot();
        snapshot.kv_ram_captures          = ram.captures;
        snapshot.kv_ram_restores          = ram.restores;
        snapshot.kv_ram_evictions         = ram.evictions;
        snapshot.kv_ram_drops             = ram.drops;
        snapshot.kv_ram_save_seconds      = ram.save_seconds;
        snapshot.kv_ram_load_seconds      = ram.load_seconds;
        snapshot.kv_ram_capacity_bytes    = ram.capacity_bytes;
        snapshot.kv_ram_used_bytes        = ram.used_bytes;
        snapshot.kv_ram_entry_count       = ram.entry_count;
        const auto disk                   = instance_.program->kv_disk_snapshot();
        snapshot.kv_disk_captures         = disk.captures;
        snapshot.kv_disk_restores         = disk.restores;
        snapshot.kv_disk_evictions        = disk.evictions;
        snapshot.kv_disk_drops            = disk.drops;
        snapshot.kv_disk_save_seconds     = disk.save_seconds;
        snapshot.kv_disk_load_seconds     = disk.load_seconds;
        snapshot.kv_disk_capacity_bytes   = disk.capacity_bytes;
        snapshot.kv_disk_used_bytes       = disk.used_bytes;
        snapshot.kv_disk_entry_count      = disk.entry_count;
        std::lock_guard lock(stats_mutex_);
        published_stats_ = snapshot;
    }

    GenerationResult wait_for_request(std::shared_ptr<Request> request, OutputSink* sink,
                                      const CancellationView& cancellation) {
        struct ConsumerGuard {
            ConcurrentExecutor* owner;
            std::shared_ptr<Request> request;

            ~ConsumerGuard() { owner->release_consumer(request); }
        } guard{this, request};

        std::exception_ptr caller_error;
        std::vector<OutputDelta> events;
        for (;;) {
            events.clear();
            bool done = false;
            {
                std::unique_lock lock(request->mutex);
                if (cancellation.armed()) {
                    request->cv.wait_for(lock, std::chrono::milliseconds(10),
                                         [&] { return request->done || !request->events.empty(); });
                } else {
                    request->cv.wait(lock,
                                     [&] { return request->done || !request->events.empty(); });
                }
                events.swap(request->events);
                done = request->done;
            }

            if (caller_error == nullptr && sink != nullptr) {
                try {
                    for (OutputDelta& event : events) { sink->publish(std::move(event)); }
                } catch (...) {
                    caller_error = std::current_exception();
                    request->cancelled.store(true, std::memory_order_release);
                    signal_control();
                }
            }

            if (caller_error == nullptr) {
                try {
                    if (cancellation.requested()) {
                        request->cancelled.store(true, std::memory_order_release);
                        signal_control();
                    }
                } catch (...) {
                    caller_error = std::current_exception();
                    request->cancelled.store(true, std::memory_order_release);
                    signal_control();
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
        Request(std::uint64_t request_identity, targets::qwen3_6::PreparedPrompt input,
                targets::qwen3_6::OutputSession output_session, PromptSummary summary,
                double frontend_seconds, ResolvedRequestOptions request_options,
                OutputDelivery output_delivery, Clock::time_point limit,
                Clock::time_point submit_time, HostInputLease input_lease)
            : id(request_identity), host_input(std::move(input_lease)), prompt(std::move(input)),
              output(std::move(output_session)), prompt_summary(summary),
              prepare_seconds(frontend_seconds), options(std::move(request_options)),
              delivery(output_delivery), deadline(limit), submitted(submit_time),
              stop_suppression_active(options.execution.suppressed_token_count != 0) {}

        const std::uint64_t id;
        HostInputLease host_input;
        targets::qwen3_6::PreparedPrompt prompt;
        targets::qwen3_6::OutputSession output;
        PromptSummary prompt_summary;
        double prepare_seconds = 0.0;
        ResolvedRequestOptions options;
        OutputDelivery delivery = OutputDelivery::TerminalOnly;
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
        bool stop_suppression_active = false;

        std::optional<BasePlan> base_plan;
        std::array<std::optional<Plan>, kMaximumConcurrency> lane_plans{};
        std::array<std::uint64_t, kMaximumConcurrency> lane_plan_versions{};
        std::optional<Plan> ram_plan;
        std::uint64_t ram_index_version = 0;
        std::optional<Plan> disk_plan;
        std::uint64_t disk_index_version = 0;
        AdmissionResources admission_resources;
        std::uint64_t remaining_service_work = 0;
        std::uint64_t backfill_epoch         = 0;
        BackfillClass backfill_class         = BackfillClass::None;
        double kv_ram_save_seconds           = 0;
        double kv_ram_load_seconds           = 0;
        double kv_disk_save_seconds          = 0;
        double kv_disk_load_seconds          = 0;
        double kv_disk_h2d_seconds           = 0;

        std::mutex mutex;
        std::condition_variable cv;
        std::vector<OutputDelta> events;
        GenerationResult result;
        std::exception_ptr error;
        bool done              = false;
        bool consumer_released = false;
        bool capacity_released = false;
    };

    struct RoundMembership {
        std::array<std::uint32_t, kMaximumConcurrency> lanes{};
        std::array<RoundBudget, kMaximumConcurrency> budgets{};
        std::size_t size = 0;

        [[nodiscard]] bool empty() const noexcept { return size == 0; }

        [[nodiscard]] std::span<const std::uint32_t> lane_span() const noexcept {
            return {lanes.data(), size};
        }

        [[nodiscard]] std::span<const RoundBudget> budget_span() const noexcept {
            return {budgets.data(), size};
        }
    };

    struct ActiveAdmissionSet {
        std::array<ActiveAdmissionSnapshot, kMaximumConcurrency> requests{};
        std::size_t size = 0;

        [[nodiscard]] std::span<const ActiveAdmissionSnapshot> span() const noexcept {
            return {requests.data(), size};
        }
    };

    enum class AdmissionProgress : std::uint8_t {
        None,
        ControlProgress,
        CopyHold,
        RanGpuUnit,
    };

    struct LaneChoice {
        std::uint32_t lane         = 0;
        bool evict_retained        = false;
        std::uint64_t ram_entry_id = 0;
        std::uint64_t disk_entry_id = 0;
    };

    struct CopyHold {
        std::shared_ptr<Request> request;
        std::uint32_t lane           = 0;
        Plan plan;
        bool ram_hit                 = false;
        std::uint64_t ram_entry_id   = 0;
        bool ram_claimed             = false;
        bool ram_consumed            = false;
        bool disk_hit                = false;
        std::uint64_t disk_entry_id  = 0;
        bool disk_claimed            = false;
        bool disk_consumed           = false;
        std::uint64_t disk_restore_epoch = 0;
        std::vector<std::uint32_t> victim_lanes;
        bool victims_evicted         = false;
        bool restored                = false;
        bool needs_prefill           = false;
    };

    [[nodiscard]] static bool membership_contains(const RoundMembership& membership,
                                                  std::uint32_t lane) noexcept {
        for (std::size_t i = 0; i < membership.size; ++i) {
            if (membership.lanes[i] == lane) { return true; }
        }
        return false;
    }

    void append_output(const std::shared_ptr<Request>& request,
                       targets::qwen3_6::PublishedOutput output, bool notify = true) {
        if (output.empty()) { return; }
        for (OutputDelta& delta : output) {
            std::string& full = delta.channel == OutputChannel::Reasoning ? request->reasoning
                                                                          : request->content;
            full += delta.text;
        }
        if (request->delivery == OutputDelivery::TerminalOnly) { return; }
        {
            std::lock_guard lock(request->mutex);
            for (OutputDelta& delta : output) { request->events.push_back(std::move(delta)); }
        }
        if (notify) { request->cv.notify_one(); }
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
        signal_control();
        release_consumer(request);
    }

    void signal_control(bool all = false) noexcept {
        control_dirty_.store(true, std::memory_order_release);
        if (all) {
            queue_cv_.notify_all();
        } else {
            queue_cv_.notify_one();
        }
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

    void release_planning_state(const std::shared_ptr<Request>& request) noexcept {
        request->base_plan.reset();
        for (auto& plan : request->lane_plans) { plan.reset(); }
        request->ram_plan.reset();
        request->ram_index_version = 0;
        request->disk_plan.reset();
        request->disk_index_version = 0;
    }

    void retire_request(const std::shared_ptr<Request>& request) {
        if (request->lane) {
            const std::uint32_t lane = *request->lane;
            if (slots_[lane] == request) { remove_completed_slot(lane); }
        }
        publish_runtime_stats();
    }

    void complete_error(std::shared_ptr<Request> request, std::exception_ptr error) {
        release_planning_state(request);
        request->prompt = {};
        request->host_input.reset();
        retire_request(request);
        {
            std::lock_guard lock(request->mutex);
            if (request->done) { return; }
            request->error = std::move(error);
            request->done  = true;
        }
        if (mark_completed(request)) { release_reserved_capacity(); }
        request->cv.notify_one();
    }

    void complete_success(std::shared_ptr<Request> request, FinishReason reason) {
        release_planning_state(request);
        request->prompt = {};
        request->host_input.reset();
        GenerationResult result;
        result.prompt                  = request->prompt_summary;
        result.generated_token_ids     = std::move(request->generated);
        result.content                 = std::move(request->content);
        result.reasoning               = std::move(request->reasoning);
        result.reasoning_tokens        = request->output.reasoning_tokens();
        result.finish_reason           = reason;
        result.timings.prepare_seconds = request->prepare_seconds;
        if (request->begin) {
            result.reused_prompt_tokens   = request->begin->reused_prompt_tokens;
            result.prefix_reuse_path      = request->begin->prefix_reuse_path;
            result.prefix_reuse_source    = request->begin->prefix_reuse_source;
        }
        if (request->lane) {
            result.captured_context_checkpoint_tokens =
                instance_.program->captured_context_checkpoint_tokens_lane(*request->lane);
            result.restored_context_checkpoint_tokens =
                instance_.program->restored_context_checkpoint_tokens_lane(*request->lane);
        }
        result.kv_ram_save_seconds = request->kv_ram_save_seconds;
        result.kv_ram_load_seconds = request->kv_ram_load_seconds;
        result.kv_disk_save_seconds = request->kv_disk_save_seconds;
        result.kv_disk_load_seconds = request->kv_disk_load_seconds;
        result.kv_disk_h2d_seconds  = request->kv_disk_h2d_seconds;
        if (request->lane) {
            result.timings = instance_.program->generation_timings_lane(*request->lane);
            result.timings.prepare_seconds = request->prepare_seconds;
            result.speculative = instance_.program->speculative_stats_lane(*request->lane);
        }
        if (request->first_token) {
            result.timings.first_token_seconds =
                request->prepare_seconds +
                std::chrono::duration<double>(*request->first_token - request->submitted).count();
        }
        result.timings.total_seconds =
            request->prepare_seconds +
            std::chrono::duration<double>(Clock::now() - request->submitted).count();
        retire_request(request);
        {
            std::lock_guard lock(request->mutex);
            if (request->done) { return; }
            request->result = std::move(result);
            request->done   = true;
        }
        if (mark_completed(request)) { release_reserved_capacity(); }
        request->cv.notify_one();
    }

    void complete_cancelled(std::shared_ptr<Request> request) {
        (void)request->output.preview_terminal(FinishReason::Cancelled);
        append_output(request, request->output.commit_preview(), false);
        complete_success(request, FinishReason::Cancelled);
    }

    bool resolve_round(const std::shared_ptr<Request>& request, TokenId token,
                       bool cancel_at_boundary) {
        const std::uint32_t lane = *request->lane;
        if (cancel_at_boundary) {
            (void)request->output.preview_terminal(FinishReason::Cancelled);
            instance_.program->resolve_prefill_lane(lane, true);
            append_output(request, request->output.commit_preview(), false);
            complete_success(request, FinishReason::Cancelled);
            return true;
        }

        const std::span<const TokenId> tokens(&token, 1);
        const OutputDecision decision = request->output.preview(
            tokens, request->budget->remaining(), request->budget->limit_reason());
        if (decision.reject_generated_round) {
            throw std::logic_error("non-speculative sampler selected a suppressed model stop token");
        }
        if (decision.accepted_tokens != 1) {
            throw std::logic_error("prefill output policy did not accept its licensed token");
        }
        request->generated.push_back(token);
        instance_.program->resolve_prefill_lane(lane, decision.finished());
        request->budget->commit(1);
        auto published = request->output.commit_preview();
        if (!decision.finished()) { synchronize_stop_suppression(request, lane); }
        if (!request->first_token) { request->first_token = Clock::now(); }
        append_output(request, std::move(published), !decision.finished());
        if (decision.finished()) {
            complete_success(request, decision.finish_reason);
            return true;
        }
        return false;
    }

    void invalidate_lane_plans(std::uint32_t lane) noexcept { ++lane_plan_versions_[lane]; }

    void enable_stop_suppression(const std::shared_ptr<Request>& request, std::uint32_t lane) {
        if (request->stop_suppression_active) { return; }
        const auto& stop_ids = instance_.loaded->frontend.default_stop_policy().token_ids;
        instance_.program->set_suppressed_tokens_lane(lane, stop_ids);
        request->stop_suppression_active = true;
    }

    void synchronize_stop_suppression(const std::shared_ptr<Request>& request,
                                      std::uint32_t lane) {
        const bool suppress = !request->output.model_stop_tokens_allowed();
        if (suppress) {
            enable_stop_suppression(request, lane);
        } else if (request->stop_suppression_active) {
            instance_.program->clear_suppressed_tokens_lane(lane);
            request->stop_suppression_active = false;
        }
    }

    void remove_completed_slot(std::uint32_t lane) {
        slots_[lane].reset();
        invalidate_lane_plans(lane);
    }

    void consume_service_work(const std::shared_ptr<Request>& request, std::uint64_t work) {
        if (work == 0 || work > request->remaining_service_work) {
            throw std::logic_error("request service projection consumed " + std::to_string(work) +
                                   " quanta with " +
                                   std::to_string(request->remaining_service_work) + " remaining");
        }
        request->remaining_service_work -= work;
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
            if (copy_hold_ && copy_hold_->lane == lane) { drain_copy_hold_before_abort(); }
            if (request->decode_ready) {
                instance_.program->retain_lane(lane);
            } else if (!instance_.program->revert_cancelled_prefill_lane(lane)) {
                instance_.program->abort_lane(lane);
            }
            if (prefill_lane_ && *prefill_lane_ == lane) {
                instance_.request_memory.deactivate();
                prefill_lane_.reset();
            }
            complete_cancelled(request);
        }
    }

    [[nodiscard]] bool expire_pending_requests() {
        std::vector<std::shared_ptr<Request>> cancelled;
        std::vector<std::shared_ptr<Request>> expired;
        bool have_pending = false;
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
            have_pending = !pending_.empty();
            published_waiting_requests_.store(static_cast<std::uint32_t>(pending_.size()),
                                              std::memory_order_relaxed);
        }
        if (protection_) {
            const auto removed_protected = [&](const std::shared_ptr<Request>& request) {
                return request->id == protection_->head_request_id;
            };
            if (std::any_of(cancelled.begin(), cancelled.end(), removed_protected) ||
                std::any_of(expired.begin(), expired.end(), removed_protected)) {
                protection_.reset();
            }
        }
        for (const auto& request : cancelled) { complete_cancelled(request); }
        for (const auto& request : expired) {
            complete_error(request, std::make_exception_ptr(RequestError(
                                        RequestErrorKind::QueueTimeout,
                                        "inference request expired while waiting for admission")));
        }
        return have_pending;
    }

    [[nodiscard]] RoundMembership build_round_membership() const {
        RoundMembership membership;
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            const auto& request = slots_[lane];
            if (request == nullptr || !request->decode_ready) { continue; }
            if (!request->budget) {
                throw std::logic_error("decode-ready request has no generation budget");
            }
            membership.lanes[membership.size]   = lane;
            membership.budgets[membership.size] = request->budget->round_budget();
            ++membership.size;
        }
        return membership;
    }

    [[nodiscard]] ActiveAdmissionSet active_admission_set() const {
        ActiveAdmissionSet active;
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            const auto& request = slots_[lane];
            if (request == nullptr) { continue; }
            if (request->admission_resources.active_lanes == 0 ||
                request->remaining_service_work == 0) {
                throw std::logic_error("active request has no admission accounting");
            }
            active.requests[active.size++] = ActiveAdmissionSnapshot{
                .request_id            = request->id,
                .resources             = request->admission_resources,
                .remaining_work_quanta = request->remaining_service_work,
                .backfill_epoch        = request->backfill_epoch,
                .backfill_class        = request->backfill_class,
            };
        }
        return active;
    }

    [[nodiscard]] bool resolve_prefill_step(const std::shared_ptr<Request>& request,
                                            const PrefillStepResult& step,
                                            bool cancel_at_boundary) {
        cumulative_stats_.computed_prefill_tokens += step.processed_prompt_tokens;
        publish_hot_runtime_counters();
        consume_service_work(request, 1);
        if (step.host_input_consumed || step.complete) { request->host_input.reset(); }
        if (cancel_at_boundary) {
            if (!request->lane) { throw std::logic_error("cancelled prefill has no request lane"); }
            const std::uint32_t lane = *request->lane;
            if (prefill_lane_ && lane == *prefill_lane_) {
                instance_.request_memory.deactivate();
                prefill_lane_.reset();
            }
            if (step.complete) {
                instance_.program->resolve_prefill_lane(lane, true);
            } else if (!instance_.program->revert_cancelled_prefill_lane(lane)) {
                instance_.program->abort_lane(lane);
            }
            complete_cancelled(request);
            return true;
        }
        if (!step.complete) { return false; }
        if (!request->lane) { throw std::logic_error("completed prefill has no request lane"); }
        if (prefill_lane_ && *request->lane == *prefill_lane_) {
            instance_.request_memory.deactivate();
            prefill_lane_.reset();
        }
        request->begin = step.summary;
        if (step.round.tokens.size() != 1) {
            throw std::logic_error("prefill did not license exactly one token");
        }
        if (!resolve_round(request, step.round.tokens.front(), false)) {
            request->decode_ready = true;
        }
        return true;
    }

    void run_prefill_step() {
        if (!prefill_lane_) { throw std::logic_error("no request owns staged prefill"); }
        const std::uint32_t lane = *prefill_lane_;
        const auto request       = slots_[lane];
        if (request == nullptr || request->decode_ready) {
            throw std::logic_error("staged prefill lane has invalid request state");
        }
        const PrefillStepResult step  = instance_.program->advance_prefill_lane(lane);
        const bool cancel_at_boundary = request->cancelled.load(std::memory_order_acquire);
        if (resolve_prefill_step(request, step, cancel_at_boundary)) { publish_runtime_stats(); }
    }

    [[nodiscard]] std::vector<std::shared_ptr<Request>> pending_snapshot() const {
        std::lock_guard lock(queue_mutex_);
        return {pending_.begin(), pending_.end()};
    }

    [[nodiscard]] bool erase_pending(const std::shared_ptr<Request>& request) {
        std::lock_guard lock(queue_mutex_);
        const auto it = std::find(pending_.begin(), pending_.end(), request);
        if (it == pending_.end()) { return false; }
        pending_.erase(it);
        published_waiting_requests_.store(static_cast<std::uint32_t>(pending_.size()),
                                          std::memory_order_relaxed);
        return true;
    }

    void clear_protection_if_head(const std::shared_ptr<Request>& request) noexcept {
        if (protection_ && protection_->head_request_id == request->id) { protection_.reset(); }
    }

    void ensure_base_plan(const std::shared_ptr<Request>& request) {
        if (!request->base_plan) {
            request->base_plan.emplace(
                instance_.program->plan_request_base(request->prompt, request->options.execution));
        }
        const RequestPlanSummary& summary = request->base_plan->summary();
        if (summary.admission.active_lanes != 1 || summary.service_work_quanta == 0) {
            throw std::logic_error("target request plan has invalid admission accounting");
        }
    }

    void ensure_lane_plan(const std::shared_ptr<Request>& request, std::uint32_t lane) {
        if (slots_[lane] != nullptr) { return; }
        if (request->lane_plan_versions[lane] == lane_plan_versions_[lane] &&
            request->lane_plans[lane]) {
            return;
        }
        request->lane_plans[lane].reset();
        request->lane_plans[lane].emplace(
            instance_.program->plan_request_for_lane(lane, request->prompt, *request->base_plan));
        request->lane_plan_versions[lane] = lane_plan_versions_[lane];
    }

    void ensure_ram_candidate(const std::shared_ptr<Request>& request) {
        if (!request->options.execution.allow_prefix_reuse) {
            request->ram_plan.reset();
            return;
        }
        const std::uint64_t version = instance_.program->kv_ram_index_version();
        if (request->ram_index_version == version) { return; }
        request->ram_plan.reset();
        Plan plan =
            instance_.program->plan_ram_reuse(request->prompt, *request->base_plan);
        if (plan.summary().reusable_prompt_tokens > 0 &&
            plan.summary().ram_entry_id != 0 &&
            plan.summary().reuse_source == PrefixReuseSource::HostRam) {
            request->ram_plan.emplace(std::move(plan));
        }
        request->ram_index_version = version;
    }

    void ensure_disk_candidate(const std::shared_ptr<Request>& request) {
        if (!request->options.execution.allow_prefix_reuse) {
            request->disk_plan.reset();
            return;
        }
        const std::uint64_t version = instance_.program->kv_disk_index_version();
        if (request->disk_index_version == version) { return; }
        request->disk_plan.reset();
        Plan plan =
            instance_.program->plan_disk_reuse(request->prompt, *request->base_plan);
        if (plan.summary().reusable_prompt_tokens > 0 &&
            plan.summary().disk_entry_id != 0 &&
            plan.summary().reuse_source == PrefixReuseSource::HostDisk) {
            request->disk_plan.emplace(std::move(plan));
        }
        request->disk_index_version = version;
    }

    [[nodiscard]] std::optional<LaneChoice>
    find_admission_lane(const std::shared_ptr<Request>& request) {
        ensure_ram_candidate(request);
        ensure_disk_candidate(request);
        const Plan* ram_plan = request->ram_plan ? &*request->ram_plan : nullptr;
        const std::uint32_t ram_reuse =
            ram_plan != nullptr ? ram_plan->summary().reusable_prompt_tokens : 0U;
        const std::uint64_t ram_entry_id =
            ram_plan != nullptr ? ram_plan->summary().ram_entry_id : 0ULL;
        const Plan* disk_plan = request->disk_plan ? &*request->disk_plan : nullptr;
        const std::uint32_t disk_reuse =
            disk_plan != nullptr ? disk_plan->summary().reusable_prompt_tokens : 0U;
        const std::uint64_t disk_entry_id =
            disk_plan != nullptr ? disk_plan->summary().disk_entry_id : 0ULL;

        auto first_host_lane = [&](const Plan& host_plan,
                                   bool after_eviction) -> std::optional<std::uint32_t> {
            std::optional<std::uint32_t> dirty;
            std::uint64_t dirty_tick = 0;
            for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
                if (slots_[lane] != nullptr) { continue; }
                const bool feasible =
                    after_eviction
                        ? instance_.program->can_admit_lane_after_retained_eviction(lane, host_plan)
                        : instance_.program->can_admit_lane(lane, host_plan);
                if (!feasible) { continue; }
                if (!instance_.program->has_retained_lane(lane)) { return lane; }
                const std::uint64_t tick = instance_.program->retained_use_tick(lane);
                if (!dirty || tick < dirty_tick) {
                    dirty      = lane;
                    dirty_tick = tick;
                }
            }
            return dirty;
        };

        std::optional<LaneChoice> selected;
        std::uint32_t selected_reuse = 0;
        bool selected_dirty          = false;
        auto consider_vram           = [&](std::uint32_t lane, std::uint32_t reuse, bool evict) {
            const bool dirty = instance_.program->has_retained_lane(lane);
            if (selected && reuse < selected_reuse) { return; }
            if (selected && reuse == selected_reuse) {
                if (!selected_dirty) { return; }
                if (dirty) {
                    const std::uint64_t selected_tick =
                        instance_.program->retained_use_tick(selected->lane);
                    const std::uint64_t tick = instance_.program->retained_use_tick(lane);
                    if (tick >= selected_tick) { return; }
                }
            }
            selected       = LaneChoice{.lane = lane, .evict_retained = evict, .ram_entry_id = 0,
                                  .disk_entry_id = 0};
            selected_reuse = reuse;
            selected_dirty = dirty;
        };
        auto consider_host = [&](std::uint32_t reuse, std::uint64_t ram_id, std::uint64_t disk_id,
                                 std::optional<std::uint32_t> lane, bool evict) {
            if (!lane || reuse == 0) { return; }
            if (selected && reuse <= selected_reuse) { return; }
            selected = LaneChoice{.lane           = *lane,
                                  .evict_retained = evict,
                                  .ram_entry_id   = ram_id,
                                  .disk_entry_id  = disk_id};
            selected_reuse = reuse;
            selected_dirty = instance_.program->has_retained_lane(*lane);
        };

        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] != nullptr) { continue; }
            ensure_lane_plan(request, lane);
            const Plan& plan          = *request->lane_plans[lane];
            const std::uint32_t reuse = plan.summary().reusable_prompt_tokens;
            if (instance_.program->can_admit_lane(lane, plan)) {
                consider_vram(lane, reuse, false);
            }
        }
        if (ram_plan != nullptr && ram_reuse > 0) {
            consider_host(ram_reuse, ram_entry_id, 0, first_host_lane(*ram_plan, false), false);
        }
        if (disk_plan != nullptr && disk_reuse > 0) {
            consider_host(disk_reuse, 0, disk_entry_id, first_host_lane(*disk_plan, false), false);
        }
        if (selected) { return selected; }

        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] != nullptr) { continue; }
            ensure_lane_plan(request, lane);
            const Plan& plan          = *request->lane_plans[lane];
            const std::uint32_t reuse = plan.summary().reusable_prompt_tokens;
            if (instance_.program->can_admit_lane_after_retained_eviction(lane, plan)) {
                consider_vram(lane, reuse, true);
            }
        }
        if (ram_plan != nullptr && ram_reuse > 0) {
            consider_host(ram_reuse, ram_entry_id, 0, first_host_lane(*ram_plan, true), true);
        }
        if (disk_plan != nullptr && disk_reuse > 0) {
            consider_host(disk_reuse, 0, disk_entry_id, first_host_lane(*disk_plan, true), true);
        }
        return selected;
    }

    [[nodiscard]] AdmissionProgress remove_pending_error(const std::shared_ptr<Request>& request,
                                                         std::exception_ptr error) {
        if (!erase_pending(request)) { return AdmissionProgress::None; }
        clear_protection_if_head(request);
        complete_error(request, std::move(error));
        return AdmissionProgress::ControlProgress;
    }

    void harvest_kv_copy_seconds(const std::shared_ptr<Request>& request) noexcept {
        if (request == nullptr) { return; }
        try {
            const auto copies = instance_.program->harvest_kv_ram_copy_seconds();
            request->kv_ram_save_seconds += copies.save;
            request->kv_ram_load_seconds += copies.load;
        } catch (...) {}
        try {
            const auto copies = instance_.program->harvest_kv_disk_copy_seconds();
            request->kv_disk_save_seconds += copies.save;
            request->kv_disk_load_seconds += copies.load;
            request->kv_disk_h2d_seconds += copies.h2d;
        } catch (...) {}
    }

    [[nodiscard]] static bool is_request_local_admission_error(std::exception_ptr error) {
        try {
            if (error) { std::rethrow_exception(error); }
        } catch (const RequestError&) {
            return true;
        } catch (...) {}
        return false;
    }

    void drain_copy_hold_before_abort() noexcept {
        if (!copy_hold_) { return; }
        const std::uint32_t lane = copy_hold_->lane;
        try {
            instance_.program->cancel_disk_restore();
        } catch (...) {}
        try {
            instance_.program->synchronize_all();
        } catch (...) {}
        harvest_kv_copy_seconds(copy_hold_->request);
        if (copy_hold_->ram_claimed && !copy_hold_->ram_consumed) {
            try {
                instance_.program->release_ram_entry(copy_hold_->ram_entry_id);
            } catch (...) {}
            copy_hold_->ram_claimed = false;
        }
        if (copy_hold_->disk_claimed && !copy_hold_->disk_consumed) {
            try {
                instance_.program->release_disk_entry(copy_hold_->disk_entry_id);
            } catch (...) {}
            copy_hold_->disk_claimed = false;
        }
        if (!copy_hold_->victims_evicted) {
            try {
                for (const std::uint32_t victim : copy_hold_->victim_lanes) {
                    instance_.program->evict_retained_lane(victim);
                    invalidate_lane_plans(victim);
                }
            } catch (...) {}
            copy_hold_->victims_evicted = true;
        }
        try {
            instance_.program->abort_lane(lane);
        } catch (...) {}
        copy_hold_.reset();
    }

    [[nodiscard]] AdmissionProgress admit_complete(bool membership_empty) {
        if (!copy_hold_) { throw std::logic_error("admit-complete requires copy-hold state"); }
        CopyHold& hold = *copy_hold_;
        const std::uint32_t lane               = hold.lane;
        const std::shared_ptr<Request> request = hold.request;
        const std::uint64_t ram_entry_id       = hold.ram_entry_id;
        bool ram_claimed                       = hold.ram_claimed;
        bool ram_consumed                      = hold.ram_consumed;
        const std::uint64_t disk_entry_id      = hold.disk_entry_id;
        bool disk_claimed                      = hold.disk_claimed;
        bool disk_consumed                     = hold.disk_consumed;
        if (request == nullptr || slots_[lane] != request) {
            throw std::logic_error("copy-hold request is not occupying its lane");
        }

        auto copies_ready = [&] { return instance_.program->kv_copies_ready(); };

        try {
            if (!hold.victims_evicted) {
                if (!instance_.program->kv_ram_copies_ready()) {
                    if (!membership_empty) { return AdmissionProgress::CopyHold; }
                    instance_.program->wait_kv_ram_copies();
                }
                for (const std::uint32_t victim : hold.victim_lanes) {
                    instance_.program->evict_retained_lane(victim);
                    invalidate_lane_plans(victim);
                }
                hold.victims_evicted = true;
            }

            if (hold.ram_hit && !hold.restored) {
                instance_.program->restore_ram_entry(lane, hold.ram_entry_id, hold.plan);
                hold.restored = true;
            }
            if (hold.disk_hit && !hold.restored) {
                instance_.program->restore_disk_entry(lane, hold.disk_entry_id, hold.plan);
                hold.disk_restore_epoch = instance_.program->pending_disk_restore_ticket();
                hold.restored = true;
            }
            instance_.program->pump_disk_restore();

            if (instance_.program->kv_disk_restore_failed()) {
                instance_.program->wait_kv_disk_copies();
            }
            if (!copies_ready() && !membership_empty) { return AdmissionProgress::CopyHold; }
            if (!copies_ready()) {
                instance_.program->wait_kv_ram_copies();
                instance_.program->wait_kv_disk_copies();
            } else {
                instance_.program->wait_kv_disk_copies();
            }

            instance_.program->wait_kv_ram_copies_on_compute();

            runtime::TransientRegion transient{};
            if (hold.needs_prefill) {
                const RequestPlanSummary summary = hold.plan.summary();
                instance_.request_memory.activate(summary.transient_bytes,
                                                  summary.transient_alignment);
                prefill_lane_ = lane;
                transient     = instance_.request_memory.region();
            }
            const PrefillStepResult first = instance_.program->start_prefill_lane(
                lane, std::move(request->prompt), std::move(hold.plan), transient);
            const auto copies = instance_.program->harvest_kv_ram_copy_seconds();
            request->kv_ram_save_seconds += copies.save;
            request->kv_ram_load_seconds += copies.load;
            const auto disk_copies = instance_.program->harvest_kv_disk_copy_seconds();
            request->kv_disk_save_seconds += disk_copies.save;
            request->kv_disk_load_seconds += disk_copies.load;
            request->kv_disk_h2d_seconds += disk_copies.h2d;
            if (hold.ram_hit) {
                instance_.program->consume_ram_entry(hold.ram_entry_id);
                hold.ram_consumed = true;
                ram_consumed      = true;
            }
            if (hold.disk_hit) {
                instance_.program->consume_disk_entry(hold.disk_entry_id);
                hold.disk_consumed = true;
                disk_consumed      = true;
            }
            if (!first.complete && (!prefill_lane_ || *prefill_lane_ != lane)) {
                throw std::logic_error("partial prefill did not retain its execution owner");
            }
            const bool cancel_at_boundary = request->cancelled.load(std::memory_order_acquire);
            copy_hold_.reset();
            (void)resolve_prefill_step(request, first, cancel_at_boundary);
            publish_runtime_stats();
            return AdmissionProgress::RanGpuUnit;
        } catch (...) {
            const std::exception_ptr error = std::current_exception();
            if (copy_hold_) {
                drain_copy_hold_before_abort();
            } else {
                if (ram_claimed && !ram_consumed) {
                    try {
                        instance_.program->release_ram_entry(ram_entry_id);
                    } catch (...) {}
                }
                if (disk_claimed && !disk_consumed) {
                    try {
                        instance_.program->release_disk_entry(disk_entry_id);
                    } catch (...) {}
                }
            }
            instance_.program->abort_lane(lane);
            if (prefill_lane_ && *prefill_lane_ == lane) {
                instance_.request_memory.deactivate();
                prefill_lane_.reset();
            }
            slots_[lane].reset();
            invalidate_lane_plans(lane);
            complete_error(request, error);
            if (is_request_local_admission_error(error)) {
                publish_runtime_stats();
                return AdmissionProgress::ControlProgress;
            }
            throw;
        }
    }

    [[nodiscard]] AdmissionProgress admit_planned_request(const std::shared_ptr<Request>& request,
                                                          LaneChoice choice,
                                                          BackfillClass backfill_class,
                                                          std::uint64_t backfill_epoch) {
        if (Clock::now() >= request->deadline) {
            return remove_pending_error(
                request, std::make_exception_ptr(RequestError(
                             RequestErrorKind::QueueTimeout,
                             "inference request expired while waiting for admission")));
        }
        if (request->cancelled.load(std::memory_order_acquire)) {
            if (!erase_pending(request)) { return AdmissionProgress::None; }
            clear_protection_if_head(request);
            complete_cancelled(request);
            return AdmissionProgress::ControlProgress;
        }

        const std::uint32_t lane = choice.lane;
        const bool ram_hit       = choice.ram_entry_id != 0;
        const bool disk_hit      = choice.disk_entry_id != 0;
        if (ram_hit && disk_hit) {
            throw std::logic_error("admission cannot claim both RAM and disk");
        }
        if (ram_hit) {
            if (!request->ram_plan ||
                request->ram_plan->summary().ram_entry_id != choice.ram_entry_id) {
                throw std::logic_error("selected RAM admission has no matching request plan");
            }
        } else if (disk_hit) {
            if (!request->disk_plan ||
                request->disk_plan->summary().disk_entry_id != choice.disk_entry_id) {
                throw std::logic_error("selected disk admission has no matching request plan");
            }
        } else if (!request->lane_plans[lane]) {
            throw std::logic_error("selected admission lane has no request plan");
        }
        Plan& winning_plan = ram_hit    ? *request->ram_plan
                             : disk_hit ? *request->disk_plan
                                        : *request->lane_plans[lane];

        bool ram_claimed  = false;
        bool disk_claimed = false;
        std::vector<std::uint64_t> captured_ram_ids;
        auto release_host_if_needed = [&]() {
            if (ram_claimed) {
                try {
                    instance_.program->release_ram_entry(choice.ram_entry_id);
                } catch (...) {}
                ram_claimed = false;
            }
            if (disk_claimed) {
                try {
                    instance_.program->release_disk_entry(choice.disk_entry_id);
                } catch (...) {}
                disk_claimed = false;
            }
        };
        auto rollback_ram_captures = [&]() {
            for (const std::uint64_t id : captured_ram_ids) {
                try {
                    instance_.program->discard_ram_capture(id);
                } catch (...) {}
            }
            captured_ram_ids.clear();
        };

        try {
            if (disk_hit) {
                const auto disk_summary = winning_plan.summary();
                if (!instance_.program->claim_disk_entry(choice.disk_entry_id,
                                                         disk_summary.disk_execution_frontier,
                                                         disk_summary.disk_hash_f_lo,
                                                         disk_summary.disk_hash_f_hi,
                                                         disk_summary.reusable_prompt_tokens,
                                                         disk_summary.disk_reuse_path)) {
                    request->disk_plan.reset();
                    release_host_if_needed();
                    return AdmissionProgress::None;
                }
                disk_claimed = true;
                instance_.program->prefetch_disk_plan(choice.disk_entry_id, winning_plan);
            }
            if (ram_hit) {
                instance_.program->claim_ram_entry(choice.ram_entry_id);
                ram_claimed = true;
            }
            std::vector<std::uint32_t> victims;
            std::vector<std::uint32_t> capture_failed;
            auto capture_fail_admit = [&]() {
                harvest_kv_copy_seconds(request);
                rollback_ram_captures();
                release_host_if_needed();
                harvest_kv_copy_seconds(request);
                return remove_pending_error(
                    request, std::make_exception_ptr(RequestError(
                                 RequestErrorKind::Overloaded,
                                 "KV RAM cannot capture a retained lane")));
            };
            if (choice.evict_retained) {
                while (!instance_.program->can_admit_lane_after_releasing(lane, winning_plan,
                                                                          victims)) {
                    std::optional<std::uint32_t> victim;
                    std::uint64_t victim_tick = 0;
                    for (std::uint32_t retained_lane = 0; retained_lane < max_concurrency_;
                         ++retained_lane) {
                        if (retained_lane == lane || slots_[retained_lane] != nullptr ||
                            !instance_.program->has_retained_lane(retained_lane)) {
                            continue;
                        }
                        if (std::find(victims.begin(), victims.end(), retained_lane) !=
                            victims.end()) {
                            continue;
                        }
                        if (std::find(capture_failed.begin(), capture_failed.end(),
                                      retained_lane) != capture_failed.end()) {
                            continue;
                        }
                        const std::uint64_t tick =
                            instance_.program->retained_use_tick(retained_lane);
                        if (!victim || tick < victim_tick) {
                            victim      = retained_lane;
                            victim_tick = tick;
                        }
                    }
                    if (!victim) { return capture_fail_admit(); }
                    std::uint64_t ram_id = 0;
                    if (!instance_.program->capture_retained_lane(*victim, &ram_id)) {
                        capture_failed.push_back(*victim);
                        continue;
                    }
                    if (ram_id != 0) { captured_ram_ids.push_back(ram_id); }
                    victims.push_back(*victim);
                }
            }

            if (instance_.program->has_retained_lane(lane) &&
                (ram_hit || disk_hit || winning_plan.summary().reusable_prompt_tokens == 0)) {
                std::uint64_t ram_id = 0;
                if (!instance_.program->capture_retained_lane(lane, &ram_id)) {
                    return capture_fail_admit();
                }
                if (ram_id != 0) { captured_ram_ids.push_back(ram_id); }
                if (std::find(victims.begin(), victims.end(), lane) == victims.end()) {
                    victims.push_back(lane);
                }
            }

            Plan selected_plan = std::move(winning_plan);
            if (ram_hit) {
                request->ram_plan.reset();
            } else if (disk_hit) {
                request->disk_plan.reset();
            } else {
                request->lane_plans[lane].reset();
            }
            if (!erase_pending(request)) {
                const auto copies = instance_.program->harvest_kv_ram_copy_seconds();
                request->kv_ram_save_seconds += copies.save;
                request->kv_ram_load_seconds += copies.load;
                const auto disk_copies = instance_.program->harvest_kv_disk_copy_seconds();
                request->kv_disk_save_seconds += disk_copies.save;
                request->kv_disk_load_seconds += disk_copies.load;
                request->kv_disk_h2d_seconds += disk_copies.h2d;
                rollback_ram_captures();
                release_host_if_needed();
                return AdmissionProgress::None;
            }
            release_planning_state(request);

            const RequestPlanSummary summary = selected_plan.summary();
            if (backfill_class == BackfillClass::Temporal) {
                if (!protection_ || protection_->epoch_id != backfill_epoch ||
                    summary.service_work_quanta > protection_->temporal_credit) {
                    throw std::logic_error("temporal backfill lost its protected credit");
                }
                protection_->temporal_credit -= summary.service_work_quanta;
            }
            clear_protection_if_head(request);

            const bool needs_prefill = summary.reusable_prompt_tokens < summary.prompt_tokens;
            request->budget.emplace(summary.effective_output_tokens,
                                    summary.effective_limit_reason);
            request->generated.reserve(summary.effective_output_tokens);
            request->lane                   = lane;
            request->admission_resources    = summary.admission;
            request->remaining_service_work = summary.service_work_quanta;
            request->backfill_epoch         = backfill_epoch;
            request->backfill_class         = backfill_class;
            slots_[lane]                    = request;
            invalidate_lane_plans(lane);

            copy_hold_.emplace(CopyHold{
                .request         = request,
                .lane            = lane,
                .plan            = std::move(selected_plan),
                .ram_hit         = ram_hit,
                .ram_entry_id    = choice.ram_entry_id,
                .ram_claimed     = ram_claimed,
                .disk_hit        = disk_hit,
                .disk_entry_id   = choice.disk_entry_id,
                .disk_claimed    = disk_claimed,
                .victim_lanes    = std::move(victims),
                .needs_prefill   = needs_prefill,
            });
            ram_claimed  = false;
            disk_claimed = false;
            publish_runtime_stats();

            const bool idle = build_round_membership().empty();
            if (idle) { return admit_complete(true); }
            return AdmissionProgress::CopyHold;
        } catch (...) {
            const std::exception_ptr error = std::current_exception();
            try {
                instance_.program->synchronize_all();
            } catch (...) {}
            try {
                const auto copies = instance_.program->harvest_kv_ram_copy_seconds();
                request->kv_ram_save_seconds += copies.save;
                request->kv_ram_load_seconds += copies.load;
            } catch (...) {}
            try {
                const auto copies = instance_.program->harvest_kv_disk_copy_seconds();
                request->kv_disk_save_seconds += copies.save;
                request->kv_disk_load_seconds += copies.load;
                request->kv_disk_h2d_seconds += copies.h2d;
            } catch (...) {}
            if (!(copy_hold_ && copy_hold_->lane == lane)) {
                rollback_ram_captures();
            }
            release_host_if_needed();
            const bool claimed = slots_[lane] == request;
            if (copy_hold_ && copy_hold_->lane == lane) {
                drain_copy_hold_before_abort();
            }
            if (claimed) {
                instance_.program->abort_lane(lane);
                if (prefill_lane_ && *prefill_lane_ == lane) {
                    instance_.request_memory.deactivate();
                    prefill_lane_.reset();
                }
                slots_[lane].reset();
                invalidate_lane_plans(lane);
            }
            complete_error(request, error);
            throw;
        }
    }

    AdmissionProgress try_admit_one() {
        bool control_progress = false;
        const std::vector<std::shared_ptr<Request>> queued = pending_snapshot();
        if (queued.empty()) {
            protection_.reset();
            return AdmissionProgress::None;
        }
        for (std::size_t head_index = 0; head_index < queued.size(); ++head_index) {
            const std::shared_ptr<Request>& head = queued[head_index];
            if (protection_ && protection_->head_request_id != head->id) { protection_.reset(); }
            if (head->cancelled.load(std::memory_order_acquire)) {
                if (erase_pending(head)) {
                    clear_protection_if_head(head);
                    complete_cancelled(head);
                    control_progress = true;
                }
                continue;
            }
            if (Clock::now() >= head->deadline) {
                (void)remove_pending_error(
                    head, std::make_exception_ptr(RequestError(
                              RequestErrorKind::QueueTimeout,
                              "inference request expired while waiting for admission")));
                control_progress = true;
                continue;
            }

            try {
                ensure_base_plan(head);
            } catch (...) {
                (void)remove_pending_error(head, std::current_exception());
                control_progress = true;
                continue;
            }
            const RequestPlanSummary& head_base = head->base_plan->summary();
            if (!admission_resources_fit(head_base.admission, admission_capacity_)) {
                (void)remove_pending_error(
                    head, std::make_exception_ptr(RequestError(
                              RequestErrorKind::ContextLengthExceeded,
                              "request reservation exceeds Engine shared KV capacity")));
                control_progress = true;
                continue;
            }

            std::optional<LaneChoice> head_lane;
            try {
                head_lane = find_admission_lane(head);
            } catch (...) {
                (void)remove_pending_error(head, std::current_exception());
                control_progress = true;
                continue;
            }
            if (head_lane) {
                return admit_planned_request(head, *head_lane, BackfillClass::None, 0);
            }

            const ActiveAdmissionSet active = active_admission_set();
            if (active.size == 0) {
                throw std::logic_error("exclusive-feasible request cannot enter an idle Engine");
            }
            if (!protection_) {
                protection_.emplace(make_admission_protection(next_protection_epoch_++, head->id,
                                                              head_base.admission, active.span(),
                                                              admission_capacity_));
            }
            if (protected_head_safe_without_temporal(*protection_, active.span(),
                                                     admission_capacity_)) {
                protection_->phase = ProtectionPhase::Drain;
            }
            if (protection_->phase == ProtectionPhase::Drain) {
                return control_progress ? AdmissionProgress::ControlProgress
                                        : AdmissionProgress::None;
            }

            const std::uint64_t frontier_distance =
                protection_frontier_distance(*protection_, active.span());
            for (std::size_t i = head_index + 1; i < queued.size(); ++i) {
                const std::shared_ptr<Request>& candidate = queued[i];
                if (candidate->cancelled.load(std::memory_order_acquire)) {
                    if (erase_pending(candidate)) {
                        complete_cancelled(candidate);
                        control_progress = true;
                    }
                    continue;
                }
                if (Clock::now() >= candidate->deadline) {
                    (void)remove_pending_error(
                        candidate, std::make_exception_ptr(RequestError(
                                       RequestErrorKind::QueueTimeout,
                                       "inference request expired while waiting for admission")));
                    control_progress = true;
                    continue;
                }

                try {
                    ensure_base_plan(candidate);
                } catch (...) {
                    (void)remove_pending_error(candidate, std::current_exception());
                    control_progress = true;
                    continue;
                }
                const RequestPlanSummary& candidate_base = candidate->base_plan->summary();
                if (!admission_resources_fit(candidate_base.admission, admission_capacity_)) {
                    (void)remove_pending_error(
                        candidate, std::make_exception_ptr(RequestError(
                                       RequestErrorKind::ContextLengthExceeded,
                                       "request reservation exceeds Engine shared KV capacity")));
                    control_progress = true;
                    continue;
                }

                std::optional<LaneChoice> candidate_lane;
                try {
                    candidate_lane = find_admission_lane(candidate);
                } catch (...) {
                    (void)remove_pending_error(candidate, std::current_exception());
                    control_progress = true;
                    continue;
                }
                if (!candidate_lane) { continue; }
                const RequestPlanSummary& candidate_plan =
                    candidate_lane->ram_entry_id != 0    ? candidate->ram_plan->summary()
                    : candidate_lane->disk_entry_id != 0 ? candidate->disk_plan->summary()
                                                         : candidate->lane_plans[candidate_lane->lane]
                                                               ->summary();

                BackfillClass backfill = BackfillClass::None;
                if (persistent_backfill_is_safe(*protection_, active.span(),
                                                candidate_plan.admission, admission_capacity_)) {
                    backfill = BackfillClass::Persistent;
                } else if (candidate_plan.service_work_quanta <= frontier_distance &&
                           candidate_plan.service_work_quanta <= protection_->temporal_credit) {
                    backfill = BackfillClass::Temporal;
                }
                if (backfill != BackfillClass::None) {
                    return admit_planned_request(candidate, *candidate_lane, backfill,
                                                 protection_->epoch_id);
                }
            }
            return control_progress ? AdmissionProgress::ControlProgress : AdmissionProgress::None;
        }
        protection_.reset();
        return control_progress ? AdmissionProgress::ControlProgress : AdmissionProgress::None;
    }

    void run_decode_round(const RoundMembership& membership) {
        RoundMembership live;
        for (std::size_t index = 0; index < membership.size; ++index) {
            const std::uint32_t lane = membership.lanes[index];
            const auto& request      = slots_[lane];
            if (request != nullptr && request->cancelled.load(std::memory_order_acquire)) {
                if (copy_hold_ && copy_hold_->lane == lane) { drain_copy_hold_before_abort(); }
                instance_.program->retain_lane(lane);
                complete_cancelled(request);
                continue;
            }
            live.lanes[live.size]   = lane;
            live.budgets[live.size] = membership.budgets[index];
            ++live.size;
        }
        if (live.empty()) { return; }

        const std::span<const std::uint32_t> lanes = live.lane_span();
        const BatchedGeneratedRound round =
            instance_.program->decode_batch(lanes, live.budget_span());

        std::array<std::uint8_t, kMaximumConcurrency> cancelled{};
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            cancelled[row] =
                slots_[lanes[row]]->cancelled.load(std::memory_order_acquire) ? 1U : 0U;
        }

        if (round.row_stride == 0 ||
            (!round.row_counts.empty() && round.row_counts.size() != lanes.size()) ||
            round.tokens.size() < static_cast<std::size_t>(round.row_stride) * lanes.size()) {
            throw std::logic_error("decode batch returned an invalid ragged layout");
        }

        std::array<std::uint32_t, kMaximumConcurrency> accepted{};
        std::array<std::uint8_t, kMaximumConcurrency> terminal{};
        std::array<std::uint8_t, kMaximumConcurrency> rejected{};
        std::array<FinishReason, kMaximumConcurrency> finish_reasons{};
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            const std::uint32_t lane = lanes[row];
            const auto& request      = slots_[lane];
            const std::uint32_t count =
                round.row_counts.empty() ? 1U : static_cast<std::uint32_t>(round.row_counts[row]);
            if (count == 0 || count > round.row_stride) {
                throw std::logic_error("decode batch returned an invalid licensed row extent");
            }
            const auto row_tokens =
                round.tokens.subspan(row * round.row_stride, static_cast<std::size_t>(count));
            if (cancelled[row]) {
                (void)request->output.preview_terminal(FinishReason::Cancelled);
                accepted[row]       = 0;
                terminal[row]       = 1;
                finish_reasons[row] = FinishReason::Cancelled;
                continue;
            }
            const OutputDecision decision = request->output.preview(
                row_tokens, request->budget->remaining(), request->budget->limit_reason());
            if (decision.reject_generated_round) {
                if (decision.accepted_tokens != 0 || decision.finished()) {
                    throw std::logic_error("output policy returned an invalid rejected round");
                }
                if (request->stop_suppression_active) {
                    throw std::logic_error(
                        "speculative sampler selected a suppressed model stop token");
                }
                rejected[row] = 1;
                enable_stop_suppression(request, lane);
            } else if (decision.accepted_tokens == 0 || decision.accepted_tokens > count) {
                // Non-terminal accepted_tokens < count is a structured-output prefix
                // (first completed tool call in a round sampled with model stops excluded).
                throw std::logic_error("output policy returned an invalid licensed prefix");
            }
            accepted[row]       = decision.accepted_tokens;
            terminal[row]       = decision.finished() ? 1 : 0;
            finish_reasons[row] = decision.finish_reason;
        }

        instance_.program->resolve_pending_batch(
            lanes, std::span<const std::uint32_t>(accepted.data(), lanes.size()),
            std::span<const std::uint8_t>(terminal.data(), lanes.size()),
            std::span<const std::uint8_t>(cancelled.data(), lanes.size()),
            std::span<const std::uint8_t>(rejected.data(), lanes.size()));

        ++cumulative_stats_.decode_rounds;
        cumulative_stats_.decode_row_rounds += lanes.size();
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            if (!cancelled[row] && !rejected[row]) {
                cumulative_stats_.committed_decode_tokens += accepted[row];
            }
        }
        publish_hot_runtime_counters();

        for (std::size_t row = 0; row < lanes.size(); ++row) {
            const std::uint32_t lane = lanes[row];
            const auto request       = slots_[lane];
            if (rejected[row]) {
                request->output.discard_preview();
                continue;
            }
            if (!cancelled[row]) {
                const auto row_tokens = round.tokens.subspan(
                    row * round.row_stride, static_cast<std::size_t>(accepted[row]));
                request->generated.insert(request->generated.end(), row_tokens.begin(),
                                          row_tokens.end());
                request->budget->commit(accepted[row]);
                consume_service_work(request, accepted[row]);
            }
            auto published = request->output.commit_preview();
            if (!terminal[row]) { synchronize_stop_suppression(request, lane); }
            if (!request->first_token && accepted[row] != 0) {
                request->first_token = Clock::now();
            }
            append_output(request, std::move(published), terminal[row] == 0);
            if (terminal[row]) { complete_success(request, finish_reasons[row]); }
        }
    }

    void fail_all(std::exception_ptr error) noexcept {
        std::scoped_lock execution_lock(execution_mutex_);
        std::vector<std::shared_ptr<Request>> pending;
        {
            std::lock_guard lock(queue_mutex_);
            failed_ = true;
            pending.assign(pending_.begin(), pending_.end());
            pending_.clear();
            published_waiting_requests_.store(0, std::memory_order_relaxed);
        }
        drain_copy_hold_before_abort();
        if (prefill_lane_) {
            instance_.request_memory.deactivate();
            prefill_lane_.reset();
        }
        protection_.reset();
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] != nullptr) {
                instance_.program->abort_lane(lane);
                complete_error(slots_[lane], error);
            }
        }
        for (const auto& request : pending) { complete_error(request, error); }
    }

    void worker_loop() noexcept {
        // Once a decode-ready donor exists, successful binds spend frozen free-lane debt without
        // refund. A maximal decode is the only operation that refreshes it.
        DecodeAdmissionBurst decode_admission_burst;
        bool stable_decode_epoch = false;
        for (;;) {
            const bool control_changed =
                control_dirty_.exchange(false, std::memory_order_acquire);
            if (stable_decode_epoch && !control_changed) {
                try {
                    std::scoped_lock execution_lock(execution_mutex_);
                    const RoundMembership membership = build_round_membership();
                    if (!membership.empty()) {
                        run_decode_round(membership);
                        continue;
                    }
                    stable_decode_epoch = false;
                } catch (...) {
                    fail_all(std::current_exception());
                    return;
                }
            }
            stable_decode_epoch = false;
            {
                std::unique_lock lock(queue_mutex_);
                if (!stopping_ && pending_.empty()) {
                    bool active = false;
                    for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
                        active = active || slots_[lane] != nullptr;
                    }
                    if (!active) {
                        bool copies_ready = false;
                        if (!copy_hold_) {
                            try {
                                copies_ready = instance_.program->kv_copies_ready();
                                if (copies_ready) { instance_.program->request_idle_spill(); }
                            } catch (...) {}
                        }
                        if (copies_ready) {
                            queue_cv_.wait(lock, [&] { return stopping_ || !pending_.empty(); });
                        } else {
                            queue_cv_.wait_for(lock, std::chrono::milliseconds(20),
                                              [&] { return stopping_ || !pending_.empty(); });
                        }
                    }
                }
                if (stopping_) {
                    lock.unlock();
                    {
                        std::scoped_lock execution_lock(execution_mutex_);
                        drain_copy_hold_before_abort();
                        try {
                            instance_.program->shutdown_kv_tiers(load_progress_);
                        } catch (...) {}
                    }
                    fail_all(std::make_exception_ptr(RequestError(
                        RequestErrorKind::Unavailable, "inference engine is shutting down")));
                    return;
                }
            }

            try {
                std::scoped_lock execution_lock(execution_mutex_);
                const bool have_pending          = expire_pending_requests();
                const auto cancelled_at_boundary = snapshot_cancellations();
                cancel_active_requests(cancelled_at_boundary);
                const RoundMembership membership = build_round_membership();

                std::uint32_t active_slots = 0;
                for (const auto& request : slots_) {
                    if (request != nullptr) { ++active_slots; }
                }
                decode_admission_burst.observe_membership(max_concurrency_, active_slots,
                                                          membership.size);

                const auto run_membership_decode = [&] {
                    run_decode_round(membership);
                    decode_admission_burst.complete_decode();
                };

                if (copy_hold_) {
                    const bool held_in_membership =
                        membership_contains(membership, copy_hold_->lane);
                    if (held_in_membership) {
                        throw std::logic_error(
                            "copy-hold lane must not join decode membership before admit-complete");
                    }
                    if (!membership.empty() && !held_in_membership &&
                        !instance_.program->kv_ram_copies_ready()) {
                        run_membership_decode();
                    } else {
                        const AdmissionProgress progress = admit_complete(membership.empty());
                        if (progress == AdmissionProgress::CopyHold &&
                            !membership.empty() &&
                            !membership_contains(membership, copy_hold_->lane)) {
                            run_membership_decode();
                        }
                    }
                    continue;
                }

                if (prefill_lane_) {
                    run_prefill_step();
                    continue;
                }

                if (have_pending &&
                    (membership.empty() || decode_admission_burst.allows_admission())) {
                    const AdmissionProgress progress = try_admit_one();
                    if (progress == AdmissionProgress::RanGpuUnit ||
                        progress == AdmissionProgress::CopyHold) {
                        if (!membership.empty()) {
                            decode_admission_burst.consume_admission();
                        }
                        continue;
                    }
                    if (progress == AdmissionProgress::ControlProgress && membership.empty()) {
                        continue;
                    }
                }

                if (!membership.empty()) {
                    run_membership_decode();
                    stable_decode_epoch = !have_pending;
                    continue;
                }
            } catch (...) {
                {
                    std::scoped_lock execution_lock(execution_mutex_);
                    drain_copy_hold_before_abort();
                    try {
                        instance_.program->shutdown_kv_tiers(load_progress_);
                    } catch (...) {}
                }
                fail_all(std::current_exception());
                return;
            }
        }
    }

    Instance& instance_;
    const std::uint32_t max_concurrency_;
    const std::size_t max_outstanding_;
    const std::chrono::milliseconds pending_timeout_;
    const AdmissionResources admission_capacity_;
    LoadProgress load_progress_;

    mutable std::mutex execution_mutex_;
    mutable std::mutex queue_mutex_;
    mutable std::mutex stats_mutex_;
    std::condition_variable queue_cv_;
    std::deque<std::shared_ptr<Request>> pending_;
    std::size_t outstanding_       = 0;
    std::uint64_t next_request_id_ = 1;
    std::array<std::shared_ptr<Request>, kMaximumConcurrency> slots_{};
    std::optional<std::uint32_t> prefill_lane_;
    std::optional<CopyHold> copy_hold_;
    std::array<std::uint64_t, kMaximumConcurrency> lane_plan_versions_{};
    std::optional<AdmissionProtection> protection_;
    std::uint64_t next_protection_epoch_ = 1;
    RuntimeStats cumulative_stats_;
    RuntimeStats published_stats_;
    static_assert(std::atomic<std::uint64_t>::is_always_lock_free);
    static_assert(std::atomic<bool>::is_always_lock_free);
    std::atomic<std::uint64_t> published_computed_prefill_tokens_{0};
    std::atomic<std::uint64_t> published_committed_decode_tokens_{0};
    std::atomic<std::uint64_t> published_decode_rounds_{0};
    std::atomic<std::uint64_t> published_decode_row_rounds_{0};
    std::atomic<std::uint32_t> published_waiting_requests_{0};
    std::atomic<bool> control_dirty_{true};
    bool stopping_ = false;
    bool failed_   = false;
    std::thread worker_;
};

} // namespace ninfer::runtime
