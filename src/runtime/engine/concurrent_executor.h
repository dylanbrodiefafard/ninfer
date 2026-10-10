#pragma once

// Small fixed-capacity request scheduling and batched decode execution for every backend.

#include "ninfer/types.h"
#include "runtime/contract/types.h"
#include "runtime/engine/admission_policy.h"
#include "runtime/engine/request_memory.h"
#include "runtime/generation/generation_budget.h"
#include "targets/qwen3_6/export/ninfer/targets/qwen3_6/frontend.h"
#include "targets/qwen3_6/export/ninfer/targets/qwen3_6/generation_recovery.h"
#include "targets/qwen3_6/export/ninfer/targets/qwen3_6/prepared_prompt.h"
#include "runtime/contract/reasoning_recovery.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <exception>
#include <future>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace ninfer::runtime {

namespace detail {

// Runs one step of failure or shutdown cleanup and discards the step's own exception. Callers use
// it only where an outcome is already decided: a primary exception is propagating to its owner,
// or the caller is noexcept (abort, shutdown, worker teardown). A secondary failure there must
// not replace the primary error or skip the later steps that release claims, lanes, and waiting
// requests.
template <class Step>
void run_secondary_cleanup(Step&& step) noexcept {
    try {
        std::forward<Step>(step)();
        // NOLINTNEXTLINE(bugprone-empty-catch): secondary cleanup failure; see above.
    } catch (...) {}
}

} // namespace detail

template <class Instance>
class ConcurrentExecutor;

// Plants one queued retry and lets the worker run start_generation_recovery for
// the script selected on the probe program. Production has no other caller.
template <class ProbeInstance>
int drive_scripted_recovery(ConcurrentExecutor<ProbeInstance>& executor);

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
          context_capacity_(options.max_context), load_progress_(options.load_progress),
          generation_recovery_(options.generation_recovery) {
        if (max_concurrency_ == 0 || max_concurrency_ > kMaximumConcurrency ||
            options.max_pending_requests == 0 || pending_timeout_.count() <= 0) {
            throw std::invalid_argument("concurrent executor bounds are invalid");
        }
        if (admission_capacity_.active_lanes != max_concurrency_ ||
            admission_capacity_.main_kv_pages == 0) {
            throw std::logic_error("target admission capacity does not match the Engine");
        }
        // The worker issues all model CUDA work; bind it to the Engine device before the first
        // request so a non-zero --device never falls back to the thread-default device 0.
        std::promise<void> startup;
        std::future<void> started = startup.get_future();
        worker_                   = std::thread([this, device = options.device,
                                                 startup = std::move(startup)]() mutable {
            try {
                const cudaError_t bound = cudaSetDevice(device);
                if (bound != cudaSuccess) {
                    throw std::runtime_error(std::string("Engine worker cudaSetDevice failed: ") +
                                             cudaGetErrorString(bound));
                }
                startup.set_value();
            } catch (...) {
                startup.set_exception(std::current_exception());
                return;
            }
            worker_loop();
        });
        try {
            started.get();
        } catch (...) {
            worker_.join();
            throw;
        }
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
            ConcurrentExecutor* owner = std::exchange(owner_, nullptr);
            return owner->wait_for_request(std::exchange(request_, nullptr), sink, cancellation);
        }

    private:
        Submission(ConcurrentExecutor& owner, std::shared_ptr<Request> request) noexcept
            : owner_(&owner), request_(std::move(request)) {}

        void reset() noexcept {
            if (owner_ != nullptr && request_ != nullptr) { owner_->abandon_request(request_); }
            request_.reset();
            owner_ = nullptr;
        }

        ConcurrentExecutor* owner_ = nullptr;
        std::shared_ptr<Request> request_;

        friend class ConcurrentExecutor;
    };

    Submission submit(targets::qwen3_6::PreparedPrompt prompt, PromptSummary prompt_summary,
                      double prepare_seconds, ResolvedRequestOptions options,
                      OutputDelivery delivery, Clock::time_point pending_deadline = {},
                      HostInputLease host_input = {}) {
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
                                                std::move(host_input), generation_recovery_);
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
            request->queue_order = next_queue_order_++;
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
        Plan plan = instance_.program->plan_request_for_lane(0, prompt, base);
        const RequestPlanSummary summary = plan.summary();
        instance_.request_memory.activate(summary.transient_bytes, summary.transient_alignment);
        try {
            ScoreResult result = instance_.program->score(
                std::move(prompt), std::move(plan), instance_.request_memory.region(), options);
            instance_.request_memory.deactivate();
            return result;
        } catch (...) {
            instance_.request_memory.deactivate();
            throw;
        }
    }

    [[nodiscard]] RuntimeStats runtime_stats() const {
        const auto disk = instance_.program->try_kv_disk_snapshot();
        RuntimeStats out;
        {
            std::lock_guard lock(stats_mutex_);
            if (disk) { note_kv_disk_snapshot_locked(*disk); }
            out = published_stats_;
            copy_kv_disk_stats(out, latest_disk_stats_);
        }
        out.waiting_requests = published_waiting_requests_.load(std::memory_order_relaxed);
        out.computed_prefill_tokens =
            published_computed_prefill_tokens_.load(std::memory_order_relaxed);
        out.committed_decode_tokens =
            published_committed_decode_tokens_.load(std::memory_order_relaxed);
        out.decode_rounds     = published_decode_rounds_.load(std::memory_order_relaxed);
        out.decode_row_rounds = published_decode_row_rounds_.load(std::memory_order_relaxed);
        // Background spill changes tier counters while the scheduler sleeps idle, so observers
        // read the RAM tier live and the disk tier whenever its index lock is free.
        assign_kv_ram_stats(out, instance_.program->kv_ram_snapshot());
        return out;
    }

    void reset_memory_peaks() noexcept {
        // Only acquiring the execution mutex can throw (std::system_error); a peak reset is a
        // diagnostic request that is skipped rather than allowed to terminate the caller.
        try {
            std::scoped_lock lock(execution_mutex_);
            instance_.program->reset_memory_peaks();
            instance_.request_memory.reset_peak();
            // NOLINTNEXTLINE(bugprone-empty-catch): a skipped diagnostic reset; see above.
        } catch (const std::system_error&) {}
    }

private:
    void publish_hot_runtime_counters() noexcept {
        published_computed_prefill_tokens_.store(cumulative_stats_.computed_prefill_tokens,
                                                 std::memory_order_relaxed);
        published_committed_decode_tokens_.store(cumulative_stats_.committed_decode_tokens,
                                                 std::memory_order_relaxed);
        published_decode_rounds_.store(cumulative_stats_.decode_rounds, std::memory_order_relaxed);
        published_decode_row_rounds_.store(cumulative_stats_.decode_row_rounds,
                                           std::memory_order_relaxed);
    }

    static void assign_kv_ram_stats(RuntimeStats& out, const auto& ram) noexcept {
        out.kv_ram_captures       = ram.captures;
        out.kv_ram_restores       = ram.restores;
        out.kv_ram_evictions      = ram.evictions;
        out.kv_ram_drops          = ram.drops;
        out.kv_ram_save_seconds   = ram.save_seconds;
        out.kv_ram_load_seconds   = ram.load_seconds;
        out.kv_ram_capacity_bytes = ram.capacity_bytes;
        out.kv_ram_used_bytes     = ram.used_bytes;
        out.kv_ram_entry_count    = ram.entry_count;
    }

    static void assign_kv_disk_stats(RuntimeStats& out, const auto& disk) noexcept {
        out.kv_disk_captures       = disk.captures;
        out.kv_disk_restores       = disk.restores;
        out.kv_disk_evictions      = disk.evictions;
        out.kv_disk_drops          = disk.drops;
        out.kv_disk_drop_reasons   = disk.drop_reasons;
        out.kv_disk_save_seconds   = disk.save_seconds;
        out.kv_disk_load_seconds   = disk.load_seconds;
        out.kv_disk_capacity_bytes = disk.capacity_bytes;
        out.kv_disk_used_bytes     = disk.used_bytes;
        out.kv_disk_entry_count    = disk.entry_count;
    }

    static void copy_kv_disk_stats(RuntimeStats& out, const RuntimeStats& in) noexcept {
        out.kv_disk_captures       = in.kv_disk_captures;
        out.kv_disk_restores       = in.kv_disk_restores;
        out.kv_disk_evictions      = in.kv_disk_evictions;
        out.kv_disk_drops          = in.kv_disk_drops;
        out.kv_disk_drop_reasons   = in.kv_disk_drop_reasons;
        out.kv_disk_save_seconds   = in.kv_disk_save_seconds;
        out.kv_disk_load_seconds   = in.kv_disk_load_seconds;
        out.kv_disk_capacity_bytes = in.kv_disk_capacity_bytes;
        out.kv_disk_used_bytes     = in.kv_disk_used_bytes;
        out.kv_disk_entry_count    = in.kv_disk_entry_count;
    }

    // Keeps the newest disk snapshot seen from either the scheduler's blocking read or an
    // observer's non-blocking one, so disk counters never step backwards between reads.
    void note_kv_disk_snapshot_locked(const auto& disk) const noexcept {
        if (disk.sequence <= latest_disk_sequence_) { return; }
        latest_disk_sequence_ = disk.sequence;
        assign_kv_disk_stats(latest_disk_stats_, disk);
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
        assign_kv_ram_stats(snapshot, instance_.program->kv_ram_snapshot());
        // The disk tier's mutex can be held by its worker for file I/O; a
        // decode-loop stats publication keeps the last snapshot instead of waiting.
        const auto disk                     = instance_.program->try_kv_disk_snapshot();
        const auto gpu                      = instance_.program->kv_gpu_snapshot();
        snapshot.gpu_kv_main_capacity_pages = gpu.main.page_group_count;
        snapshot.gpu_kv_main_entitled_pages = gpu.main.entitled_pages;
        snapshot.gpu_kv_main_mapped_pages   = gpu.main.mapped_pages;
        snapshot.gpu_kv_main_free_pages     = gpu.main.free_pages;
        snapshot.gpu_kv_spec_capacity_pages = gpu.spec.page_group_count;
        snapshot.gpu_kv_spec_entitled_pages = gpu.spec.entitled_pages;
        snapshot.gpu_kv_spec_mapped_pages   = gpu.spec.mapped_pages;
        snapshot.gpu_kv_spec_free_pages     = gpu.spec.free_pages;
        std::lock_guard lock(stats_mutex_);
        if (disk) { note_kv_disk_snapshot_locked(*disk); }
        copy_kv_disk_stats(snapshot, latest_disk_stats_);
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
        std::vector<RecoveryEvent> recovery_events;
        for (;;) {
            events.clear();
            recovery_events.clear();
            bool done = false;
            {
                std::unique_lock lock(request->mutex);
                if (cancellation.armed()) {
                    request->cv.wait_for(lock, std::chrono::milliseconds(10), [&] {
                        return request->done || !request->events.empty() ||
                               !request->recovery_events.empty();
                    });
                } else {
                    request->cv.wait(lock, [&] {
                        return request->done || !request->events.empty() ||
                               !request->recovery_events.empty();
                    });
                }
                events.swap(request->events);
                recovery_events.swap(request->recovery_events);
                done = request->done;
            }

            if (caller_error == nullptr && sink != nullptr) {
                try {
                    for (const auto& event : recovery_events) { sink->recovery_event(event); }
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
                Clock::time_point submit_time, HostInputLease input_lease, bool generation_recovery)
            : id(request_identity), host_input(std::move(input_lease)), prompt(std::move(input)),
              output(std::move(output_session)), prompt_summary(summary),
              resident_prompt_tokens(summary.prompt_tokens), prepare_seconds(frontend_seconds),
              options(std::move(request_options)), delivery(output_delivery), deadline(limit),
              submitted(submit_time), pending_since(submit_time),
              stop_suppression_active(options.execution.suppressed_token_count != 0) {
            if (generation_recovery && options.execution.sampling.p_less && !options.output.raw) {
                recovery_context = output.generation_recovery_context();
            }
        }

        const std::uint64_t id;
        std::uint64_t queue_order = 0;
        HostInputLease host_input;
        targets::qwen3_6::PreparedPrompt prompt;
        targets::qwen3_6::OutputSession output;
        PromptSummary prompt_summary;
        // Length of the prompt prefix the next recovery splice copies. The admitted
        // prompt_summary.prompt_tokens value stays unchanged for usage reporting.
        std::uint32_t resident_prompt_tokens = 0;
        double prepare_seconds               = 0.0;
        ResolvedRequestOptions options;
        OutputDelivery delivery = OutputDelivery::TerminalOnly;
        Clock::time_point deadline;
        bool cache_fallback = false;
        Clock::time_point submitted;
        double queued_seconds    = 0;
        double copy_hold_seconds = 0;
        std::optional<Clock::time_point> copy_hold_started;
        Clock::time_point pending_since;
        std::optional<Clock::time_point> first_token;
        std::optional<GenerationBudget> budget;
        std::optional<BeginSummary> begin;
        std::vector<TokenId> generated;
        std::string content;
        std::string reasoning;
        std::vector<TokenLogprob> content_logprobs;
        std::vector<TokenLogprob> reasoning_logprobs;
        std::optional<std::uint32_t> lane;
        std::atomic<bool> cancelled{false};
        std::array<bool, kMaximumConcurrency> preservation_unavailable{};
        std::array<std::uint64_t, kMaximumConcurrency> preservation_ram_versions{};
        std::array<std::uint64_t, kMaximumConcurrency> preservation_lane_versions{};
        bool decode_ready            = false;
        bool stop_suppression_active = false;
        std::shared_ptr<const targets::qwen3_6::GenerationRecoveryContext> recovery_context;
        GenerationRecoveryStats recovery;
        std::uint32_t cycle_exclusions = 0;
        std::string recovery_cause;
        bool recovery_pending                = false;
        std::size_t recovery_reasoning_begin = 0;
        std::size_t recovery_generated_begin = 0;
        RepeatedReasoningSpan reasoning_cycle;
        std::uint32_t previous_reasoning_tokens = 0;
        GenerationTimings initial_timings;
        double previous_decode_seconds = 0.0;
        SpeculativeStats previous_speculative;

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
        std::vector<RecoveryEvent> recovery_events;
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
        std::uint32_t lane          = 0;
        bool evict_retained         = false;
        std::uint64_t ram_entry_id  = 0;
        std::uint64_t disk_entry_id = 0;
        std::optional<std::uint32_t> fork_source;
        bool preserve_source = false;
        std::array<std::uint8_t, kMaximumConcurrency> protected_lanes{};
    };

    struct CopyHold {
        std::shared_ptr<Request> request;
        std::uint32_t lane = 0;
        Plan plan;
        std::optional<std::uint32_t> fork_source;
        std::uint64_t fork_source_version = 0;
        bool ram_hit                      = false;
        std::uint64_t ram_entry_id        = 0;
        bool ram_claimed                  = false;
        bool ram_consumed                 = false;
        bool disk_hit                     = false;
        std::uint64_t disk_entry_id       = 0;
        bool disk_claimed                 = false;
        bool disk_consumed                = false;
        std::uint64_t disk_restore_epoch  = 0;
        std::array<std::uint32_t, kMaximumConcurrency> victim_lanes{};
        std::size_t victim_count = 0;
        bool victims_evicted     = false;
        bool restored            = false;
        bool needs_prefill       = false;
        // Set for a generation-recovery retry: its lane and output stay with the retry,
        // so a failed host restore replans or cold-prefills this lane from the base plan.
        std::optional<BasePlan> recovery_base;
        std::uint8_t recovery_stale_retries = 0;
        // Cancelled while other lanes decode: restore work has stopped and the copies already
        // queued are fenced; the hold drains once they settle.
        bool cancel_pending = false;
        // The host restore failed. While other lanes decode, its copies are fenced the same
        // way; cleanup discards only sources classified invalid.
        bool restore_failed        = false;
        bool discard_failed_source = true;
        bool replan_restore        = false;
    };

    [[nodiscard]] static bool membership_contains(const RoundMembership& membership,
                                                  std::uint32_t lane) noexcept {
        for (std::size_t i = 0; i < membership.size; ++i) {
            if (membership.lanes[i] == lane) { return true; }
        }
        return false;
    }

    // The records of one committed round, split by the channel their tokens publish to.
    struct ChannelLogprobs {
        std::vector<TokenLogprob> content;
        std::vector<TokenLogprob> reasoning;
    };

    // Builds the records of the tokens the pending preview accepted. `tokens` is the accepted
    // prefix and `first_slot` its first slot in the round's logprob storage. Must run between
    // preview() and commit_preview(), while the session still reports the token channels.
    [[nodiscard]] static ChannelLogprobs collect_token_logprobs(const Request& request,
                                                                const RoundLogprobs& logprobs,
                                                                std::size_t first_slot,
                                                                std::span<const TokenId> tokens) {
        ChannelLogprobs records;
        if (!request.options.output.top_logprobs || tokens.empty()) { return records; }
        if (logprobs.empty()) {
            throw std::logic_error("target round carries no token logprobs for a request row");
        }
        const auto channels = request.output.preview_token_channels();
        if (channels.size() != tokens.size()) {
            throw std::logic_error("output preview does not cover the accepted tokens");
        }
        const std::uint32_t top_count = *request.options.output.top_logprobs;
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            const std::optional<OutputChannel> channel = channels[i];
            if (!channel) { continue; }
            auto& channel_records =
                *channel == OutputChannel::Reasoning ? records.reasoning : records.content;
            channel_records.push_back(logprobs.record(first_slot + i, tokens[i], top_count));
        }
        return records;
    }

    // Moves each channel's records onto that channel's delta. A channel whose tokens published no
    // text this round (text held for a stop-string or UTF-8 boundary) gets a textless delta.
    static void attach_token_logprobs(targets::qwen3_6::PublishedOutput& output,
                                      ChannelLogprobs records) {
        const auto attach = [&output](OutputChannel channel, std::vector<TokenLogprob>& tokens) {
            if (tokens.empty()) { return; }
            for (OutputDelta& delta : output) {
                if (delta.channel == channel) {
                    delta.logprobs = std::move(tokens);
                    return;
                }
            }
            output.push_back(
                OutputDelta{.channel = channel, .text = {}, .logprobs = std::move(tokens)});
        };
        attach(OutputChannel::Reasoning, records.reasoning);
        attach(OutputChannel::Content, records.content);
    }

    void append_output(const std::shared_ptr<Request>& request,
                       targets::qwen3_6::PublishedOutput output, bool notify = true) {
        if (output.empty()) { return; }
        for (OutputDelta& delta : output) {
            const bool reasoning = delta.channel == OutputChannel::Reasoning;
            std::string& full    = reasoning ? request->reasoning : request->content;
            full += delta.text;
            auto& logprobs = reasoning ? request->reasoning_logprobs : request->content_logprobs;
            logprobs.insert(logprobs.end(), delta.logprobs.begin(), delta.logprobs.end());
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

    void abandon_request(const std::shared_ptr<Request>& request) noexcept {
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

    // The request is taken by value: callers may pass slots_[lane] itself, which
    // retire_request() resets before the result is published through the request.
    // NOLINTNEXTLINE(performance-unnecessary-value-param): the copy keeps the request alive.
    void complete_error(std::shared_ptr<Request> request, std::exception_ptr error) {
        end_copy_hold(request);
        request->recovery.cycle_exclusions = request->cycle_exclusions;
        error = attach_recovery_stats(std::move(error), request->recovery);
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

    void publish_recovery(const std::shared_ptr<Request>& request, RecoveryEventKind kind,
                          std::string_view cause) {
        RecoveryEvent event{.kind                 = kind,
                            .cause                = std::string(cause),
                            .attempts             = request->recovery.attempts,
                            .cycle_exclusions     = request->cycle_exclusions,
                            .discarded_tool_calls = request->recovery.discarded_tool_calls,
                            .discarded_reasoning_tokens =
                                request->recovery.discarded_reasoning_tokens,
                            .generated_tokens = request->generated.size(),
                            .remaining_tokens = request->budget ? request->budget->remaining() : 0};
        {
            std::lock_guard lock(request->mutex);
            request->recovery_events.push_back(std::move(event));
        }
        request->cv.notify_one();
    }

    // The request is taken by value for the same slot-aliasing reason as complete_error().
    // NOLINTNEXTLINE(performance-unnecessary-value-param): the copy keeps the request alive.
    void complete_success(std::shared_ptr<Request> request, FinishReason reason) {
        if (request->cycle_exclusions != 0 || !request->recovery_cause.empty()) {
            publish_recovery(request, RecoveryEventKind::Finished,
                             reason == FinishReason::Cancelled         ? "cancelled"
                             : !request->output.tool_calls().empty()   ? "tool_calls"
                             : reason == FinishReason::OutputLimit     ? "output_limit"
                             : reason == FinishReason::ContextCapacity ? "context_capacity"
                                                                       : "stop");
        }
        release_planning_state(request);
        request->prompt = {};
        request->host_input.reset();
        GenerationResult result;
        result.prompt              = request->prompt_summary;
        result.generated_token_ids = std::move(request->generated);
        result.content             = std::move(request->content);
        result.reasoning           = std::move(request->reasoning);
        result.content_logprobs    = std::move(request->content_logprobs);
        result.reasoning_logprobs  = std::move(request->reasoning_logprobs);
        const auto calls           = request->output.tool_calls();
        if (reason != FinishReason::Cancelled) {
            result.tool_calls.assign(calls.begin(), calls.end());
        }
        // A turn that stopped without a tool call is not continued from its frontier.
        if (request->lane && calls.empty() &&
            (reason == FinishReason::StopToken || reason == FinishReason::StopString)) {
            instance_.program->mark_turn_closed(*request->lane);
        }
        result.recovery                  = request->recovery;
        result.recovery.cycle_exclusions = request->cycle_exclusions;
        if (!request->output.has_token_grammar()) {
            result.undeclared_tool_call_names =
                targets::qwen3_6::unconstrained_tool_call_names(result.content, 128);
        }
        for (std::size_t index = 0; index < result.tool_calls.size(); ++index) {
            result.tool_calls[index].id =
                "call_" + std::to_string(request->id) + "_" + std::to_string(index);
        }
        result.reasoning_tokens =
            request->previous_reasoning_tokens + request->output.reasoning_tokens();
        result.finish_reason           = reason;
        result.timings.prepare_seconds = request->prepare_seconds;
        if (request->begin) {
            result.reused_prompt_tokens = request->begin->reused_prompt_tokens;
            result.prefix_reuse_path    = request->begin->prefix_reuse_path;
            result.prefix_reuse_source  = request->begin->prefix_reuse_source;
        }
        if (request->lane) {
            result.captured_context_checkpoint_tokens =
                instance_.program->captured_context_checkpoint_tokens_lane(*request->lane);
            result.restored_context_checkpoint_tokens =
                instance_.program->restored_context_checkpoint_tokens_lane(*request->lane);
        }
        result.kv_ram_save_seconds  = request->kv_ram_save_seconds;
        result.kv_ram_load_seconds  = request->kv_ram_load_seconds;
        result.kv_disk_save_seconds = request->kv_disk_save_seconds;
        result.kv_disk_load_seconds = request->kv_disk_load_seconds;
        result.kv_disk_h2d_seconds  = request->kv_disk_h2d_seconds;
        if (request->lane) {
            result.timings = instance_.program->generation_timings_lane(*request->lane);
            result.timings.prepare_seconds = request->prepare_seconds;
            result.speculative = instance_.program->speculative_stats_lane(*request->lane);
        }
        if (request->recovery.attempts != 0) {
            result.timings.prefill_seconds       = request->initial_timings.prefill_seconds;
            result.timings.vision_seconds        = request->initial_timings.vision_seconds;
            result.timings.prefill_tail_tok_s    = request->initial_timings.prefill_tail_tok_s;
            result.timings.prefill_tail_window_s = request->initial_timings.prefill_tail_window_s;
            result.timings.decode_seconds += request->previous_decode_seconds;
            add_speculative(result.speculative, request->previous_speculative);
        }
        if (request->first_token) {
            result.timings.first_token_seconds =
                request->prepare_seconds +
                std::chrono::duration<double>(*request->first_token - request->submitted).count();
        }
        result.timings.total_seconds =
            request->prepare_seconds +
            std::chrono::duration<double>(Clock::now() - request->submitted).count();
        end_copy_hold(request);
        result.timings.queued_seconds    = request->queued_seconds;
        result.timings.copy_hold_seconds = request->copy_hold_seconds;
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

    void complete_cancelled(const std::shared_ptr<Request>& request) {
        if (!request->output.terminal()) {
            (void)request->output.preview_terminal(FinishReason::Cancelled);
            append_output(request, request->output.commit_preview(), false);
        }
        complete_success(request, FinishReason::Cancelled);
    }

    void recovery_exhausted(const std::shared_ptr<Request>& request, const std::string& detail) {
        publish_recovery(request, RecoveryEventKind::Exhausted, detail);
        const auto lane = *request->lane;
        instance_.program->abort_lane(lane);
        if (prefill_lane_ && *prefill_lane_ == lane) {
            instance_.request_memory.deactivate();
            prefill_lane_.reset();
        }
        request->recovery_pending = false;
        complete_error(
            request,
            std::make_exception_ptr(RequestError(
                RequestErrorKind::RecoveryExhausted,
                "NInfer detected persistent repeated generation; bounded recovery exhausted: " +
                    detail)));
    }

    static void add_speculative(SpeculativeStats& total, const SpeculativeStats& part) {
        if (!part.enabled) { return; }
        total.backend      = part.backend;
        total.enabled      = true;
        total.draft_window = part.draft_window;
        total.rounds += part.rounds;
        total.drafted_tokens += part.drafted_tokens;
        total.accepted_tokens += part.accepted_tokens;
        total.fallback_steps += part.fallback_steps;
        if (part.p_less_draft_temperature > 0.0F) {
            total.p_less_draft_temperature = part.p_less_draft_temperature;
        }
        auto add = [](auto& into, const auto& values) {
            if (into.size() < values.size()) { into.resize(values.size()); }
            for (std::size_t i = 0; i < values.size(); ++i) { into[i] += values[i]; }
        };
        add(total.accepted_per_position, part.accepted_per_position);
        add(total.rounds_per_draft, part.rounds_per_draft);
    }

    void release_recovery_lane(std::uint32_t lane) noexcept {
        instance_.program->abort_lane(lane);
        if (prefill_lane_ && *prefill_lane_ == lane) {
            instance_.request_memory.deactivate();
            prefill_lane_.reset();
        }
    }

    [[nodiscard]] PrefillStepResult begin_recovery_prefill(const std::shared_ptr<Request>& request,
                                                           std::uint32_t lane, Plan plan) {
        const RequestPlanSummary summary = plan.summary();
        instance_.request_memory.activate(summary.transient_bytes, summary.transient_alignment);
        prefill_lane_                 = lane;
        const auto prefill_started    = Clock::now();
        const PrefillStepResult first = instance_.program->start_prefill_lane(
            lane, std::move(request->prompt), std::move(plan), instance_.request_memory.region(),
            &request->output);
        request->recovery.prefill_seconds +=
            std::chrono::duration<double>(Clock::now() - prefill_started).count();
        return first;
    }

    void resolve_recovery_prefill(const std::shared_ptr<Request>& request,
                                  const PrefillStepResult& first) {
        (void)resolve_prefill_step(request, first,
                                   request->cancelled.load(std::memory_order_acquire));
        publish_runtime_stats();
    }

    void recovery_cold_prefill(const std::shared_ptr<Request>& request, std::uint32_t lane,
                               const BasePlan& base) {
        release_recovery_lane(lane);
        const auto first = begin_recovery_prefill(
            request, lane, instance_.program->plan_request_for_lane(lane, request->prompt, base));
        resolve_recovery_prefill(request, first);
    }

    // Claims the host entry and parks the retry in copy-hold, so decode-ready lanes keep
    // running while its RAM or disk prefix restores; admit_complete starts the prefill.
    // Returns false on a disk claim miss, before anything is held.
    [[nodiscard]] bool hold_recovery_restore(const std::shared_ptr<Request>& request,
                                             std::uint32_t lane, bool ram, Plan host_plan,
                                             BasePlan& base, std::uint8_t stale_retries = 0) {
        const RequestPlanSummary summary = host_plan.summary();
        if (ram) {
            instance_.program->claim_ram_entry(summary.ram_entry_id);
        } else {
            if (!instance_.program->claim_disk_entry(
                    summary.disk_entry_id, summary.disk_execution_frontier, summary.disk_hash_f_lo,
                    summary.disk_hash_f_hi, summary.reusable_prompt_tokens, summary.disk_reuse_path,
                    summary.disk_committed_generation)) {
                return false;
            }
            try {
                instance_.program->prefetch_disk_plan(summary.disk_entry_id, host_plan);
            } catch (...) {
                detail::run_secondary_cleanup(
                    [&] { instance_.program->release_disk_entry(summary.disk_entry_id); });
                throw;
            }
        }
        copy_hold_.emplace(CopyHold{
            .request                = request,
            .lane                   = lane,
            .plan                   = std::move(host_plan),
            .ram_hit                = ram,
            .ram_entry_id           = ram ? summary.ram_entry_id : 0,
            .ram_claimed            = ram,
            .disk_hit               = !ram,
            .disk_entry_id          = ram ? 0 : summary.disk_entry_id,
            .disk_claimed           = !ram,
            .victims_evicted        = true,
            .needs_prefill          = summary.reusable_prompt_tokens < summary.prompt_tokens,
            .recovery_base          = std::move(base),
            .recovery_stale_retries = stale_retries,
        });
        begin_copy_hold(request);
        if (build_round_membership().empty()) { (void)admit_complete(true); }
        return true;
    }

    // Recovery starts only after the generated round is committed. Tool calls
    // remain unpublished; reasoning retries pause an otherwise active lane.
    // The retry keeps this lane and splices onto the resident prompt.
    [[nodiscard]] bool start_generation_recovery() {
        while (!recovery_queue_.empty()) {
            auto request = std::move(recovery_queue_.front());
            recovery_queue_.pop_front();
            if (!request->lane || slots_[*request->lane] != request || !request->recovery_pending) {
                continue;
            }
            const auto lane = *request->lane;
            try {
                start_recovery_attempt(request, lane);
            } catch (...) {
                // A nested admit_complete or failed-restore fallback may already have
                // failed the request before rethrowing; only classify it then.
                const std::exception_ptr error = std::current_exception();
                if (request_done(*request) ? !is_request_local_error(error)
                                           : !fail_admitted_request(request, lane, error)) {
                    throw;
                }
            }
            return true;
        }
        return false;
    }

    [[nodiscard]] static bool request_done(Request& request) {
        std::lock_guard lock(request.mutex);
        return request.done;
    }

    // One retry from its committed lane. The lane, prompt, and output session are mutated
    // in stages, so any exception leaves cleanup to the caller's per-request boundary.
    void start_recovery_attempt(const std::shared_ptr<Request>& request, std::uint32_t lane) {
        const auto attempt                = request->recovery.attempts + 1;
        auto execution                    = request->options.execution;
        execution.requested_output_tokens = request->budget->remaining();
        // Keep the request's reuse setting. capture stays off so a retry does not
        // pin its frontier as a turn rollback.
        execution.capture_context_checkpoint = false;
        execution.suppressed_token_count     = 0;

        const bool reuse_enabled = execution.allow_prefix_reuse && !execution.force_cold_prefill;
        const bool lane_retained = reuse_enabled && instance_.program->retain_reusable_lane(lane);
        invalidate_lane_plans(lane);
        release_planning_state(request);

        const auto started = Clock::now();
        std::vector<TokenId> prefix;
        std::uint32_t ignored_frontier = 0;
        const bool copied              = instance_.program->copy_reusable_prompt(
            lane, request->resident_prompt_tokens, prefix, ignored_frontier);
        (void)ignored_frontier;
        std::vector<TokenId> original_prefix;
        if (!copied) {
            auto rebuilt = instance_.loaded->frontend.prepare(request->recovery_context->input());
            request->resident_prompt_tokens = rebuilt.summary().prompt_tokens;
            auto taken      = targets::qwen3_6::PreparedPromptAccess::take(std::move(rebuilt));
            original_prefix = std::move(taken.token_ids);
        }
        const auto selected = targets::qwen3_6::recovery_splice_prefix(
            copied ? std::span<const TokenId>(prefix) : std::span<const TokenId>{},
            original_prefix);
        if (selected.data() != prefix.data()) { prefix = std::move(original_prefix); }
        const auto insert =
            request->recovery_context->recovery_insert(request->output.tool_calls(), attempt);
        auto spliced = instance_.loaded->frontend.splice_recovery_prompt(
            std::move(prefix), request->recovery_context->input(), insert,
            request->recovery_context);
        request->recovery.prepare_seconds +=
            std::chrono::duration<double>(Clock::now() - started).count();

        const auto route_of = [&](bool splice_accepted, bool budget_ok, bool pages_ok,
                                  std::uint32_t resident, bool can_admit, std::uint32_t ram,
                                  std::uint32_t disk) {
            return targets::qwen3_6::route_recovery_prefill(targets::qwen3_6::RecoveryPrefillInput{
                .splice_accepted          = splice_accepted,
                .allow_prefix_reuse       = execution.allow_prefix_reuse,
                .force_cold_prefill       = execution.force_cold_prefill,
                .lane_retained            = lane_retained,
                .resident_reusable_tokens = resident,
                .can_admit_lane           = can_admit,
                .output_budget_preserved  = budget_ok,
                .pages_fit                = pages_ok,
                .ram_reusable_tokens      = ram,
                .disk_reusable_tokens     = disk,
            });
        };
        if (!spliced) {
            const auto decision = route_of(false, true, true, 0, true, 0, 0);
            recovery_exhausted(request, std::string(decision.detail));
            return;
        }

        const std::uint32_t spliced_tokens = spliced->summary().prompt_tokens;
        const bool budget_ok               = targets::qwen3_6::recovery_output_budget_preserved(
            spliced_tokens, context_capacity_, request->budget->remaining());
        if (!budget_ok) {
            const auto decision = route_of(true, false, true, 0, true, 0, 0);
            recovery_exhausted(request, std::string(decision.detail));
            return;
        }

        auto session = instance_.loaded->frontend.make_output_session(
            *spliced, request->options.stop, request->options.output);
        if (!session.model_stop_tokens_allowed()) {
            const auto& stops = instance_.loaded->frontend.default_stop_policy().token_ids;
            execution.suppressed_token_count = static_cast<std::uint32_t>(stops.size());
            std::copy(stops.begin(), stops.end(), execution.suppressed_token_ids.begin());
        }
        auto base = instance_.program->plan_request_base(*spliced, execution);
        const bool pages_ok =
            admission_resources_fit(base.summary().admission, request->admission_resources) &&
            base.summary().effective_output_tokens == request->budget->remaining();
        if (!pages_ok) {
            const auto decision = route_of(true, true, false, 0, true, 0, 0);
            recovery_exhausted(request, std::string(decision.detail));
            return;
        }

        std::uint32_t resident = 0;
        bool can_admit         = true;
        std::optional<Plan> lane_plan;
        if (lane_retained) {
            lane_plan = instance_.program->plan_request_for_lane(lane, *spliced, base);
            resident  = lane_plan->summary().reusable_prompt_tokens;
            can_admit = instance_.program->can_admit_lane(lane, *lane_plan);
        }
        auto decision = route_of(true, true, true, resident, can_admit, 0, 0);
        std::optional<Plan> ram_plan;
        std::optional<Plan> disk_plan;
        if (decision.route == targets::qwen3_6::RecoveryPrefillRoute::Cold && reuse_enabled) {
            ram_plan = optional_host_reuse_plan(
                [&] { return instance_.program->plan_ram_reuse(*spliced, base); });
            disk_plan = optional_host_reuse_plan(
                [&] { return instance_.program->plan_disk_reuse(*spliced, base); });
            const auto host_tokens = [](const std::optional<Plan>& plan, PrefixReuseSource source,
                                        bool ram_source) -> std::uint32_t {
                if (!plan) { return 0; }
                const RequestPlanSummary& summary = plan->summary();
                if (summary.reusable_prompt_tokens == 0 || summary.reuse_source != source) {
                    return 0;
                }
                if ((ram_source ? summary.ram_entry_id : summary.disk_entry_id) == 0) { return 0; }
                return summary.reusable_prompt_tokens;
            };
            decision = route_of(true, true, true, resident, can_admit,
                                host_tokens(ram_plan, PrefixReuseSource::HostRam, true),
                                host_tokens(disk_plan, PrefixReuseSource::HostDisk, false));
        }
        if (decision.route == targets::qwen3_6::RecoveryPrefillRoute::Exhaust) {
            recovery_exhausted(request, std::string(decision.detail));
            return;
        }
        if (request->cancelled.load(std::memory_order_acquire)) {
            release_recovery_lane(lane);
            complete_cancelled(request);
            return;
        }

        const auto previous = instance_.program->generation_timings_lane(lane);
        if (request->recovery.attempts == 0) { request->initial_timings = previous; }
        request->previous_decode_seconds += previous.decode_seconds;
        add_speculative(request->previous_speculative,
                        instance_.program->speculative_stats_lane(lane));
        request->previous_reasoning_tokens += request->output.reasoning_tokens();
        request->prompt                 = std::move(*spliced);
        request->output                 = std::move(session);
        request->resident_prompt_tokens = spliced_tokens;
        request->recovery.attempts      = attempt;
        request->recovery_pending       = false;
        publish_recovery(request, RecoveryEventKind::RetryStarted, request->recovery_cause);
        // The failed attempt may end mid-word. Keep it visible, but do not
        // concatenate the new attempt onto that unfinished word in SSE/JSON.
        if (!request->reasoning.empty()) {
            targets::qwen3_6::PublishedOutput separator;
            separator.push_back(OutputDelta{OutputChannel::Reasoning, "\n\n"});
            append_output(request, std::move(separator));
        }
        request->recovery_reasoning_begin = request->reasoning.size();
        request->recovery_generated_begin = request->generated.size();
        request->reasoning_cycle          = {};
        request->stop_suppression_active  = execution.suppressed_token_count != 0;

        using Route = targets::qwen3_6::RecoveryPrefillRoute;
        switch (decision.route) {
        case Route::ResidentSuffix: {
            const auto first = begin_recovery_prefill(request, lane, std::move(lane_plan.value()));
            resolve_recovery_prefill(request, first);
            return;
        }
        case Route::HostRam:
        case Route::HostDisk: {
            release_recovery_lane(lane);
            const bool ram = decision.route == Route::HostRam;
            if (!hold_recovery_restore(
                    request, lane, ram,
                    ram ? std::move(ram_plan.value()) : std::move(disk_plan.value()), base)) {
                recovery_cold_prefill(request, lane, base);
            }
            return;
        }
        case Route::Cold:
            recovery_cold_prefill(request, lane, base);
            return;
        case Route::Exhaust:
            throw std::logic_error("recovery exhaust must stop before prefill");
        }
    }

    bool recover_persistent_reasoning(const std::shared_ptr<Request>& request) {
        if (!request->recovery_context || !request->output.in_reasoning() ||
            !request->content.empty() || request->options.execution.sampling.temperature <= 0) {
            return false;
        }
        const auto current =
            std::span<const TokenId>(request->generated).subspan(request->recovery_generated_begin);
        if (!request->reasoning_cycle.observe(current)) { return false; }
        request->recovery.discarded_reasoning_tokens += request->output.reasoning_tokens();
        request->recovery_cause = "repeated_reasoning";
        publish_recovery(request, RecoveryEventKind::RetryTriggered, request->recovery_cause);
        if (request->recovery.attempts >=
                targets::qwen3_6::GenerationRecoveryContext::maximum_attempts ||
            request->budget->remaining() == 0) {
            recovery_exhausted(request,
                               "persistent reasoning exhausted its retry or output-token budget");
            return true;
        }
        request->decode_ready     = false;
        request->recovery_pending = true;
        recovery_queue_.push_back(request);
        signal_control();
        return true;
    }

    void finish_generation(const std::shared_ptr<Request>& request, FinishReason reason) {
        if (!generation_recovery_allowed_at_finish(reason, request->generated,
                                                   request->options.stop)) {
            complete_success(request, reason);
            return;
        }
        // An explicit caller stop still wins. A budget boundary containing the
        // third repeated passage must not silently disguise recovery exhaustion
        // as a normal length finish merely because it is the last licensed round.
        if ((reason == FinishReason::OutputLimit || reason == FinishReason::ContextCapacity) &&
            recover_persistent_reasoning(request)) {
            return;
        }
        const auto calls = request->output.tool_calls();
        const auto reasoning =
            std::string_view(request->reasoning).substr(request->recovery_reasoning_begin);
        if (request->recovery_context && request->recovery_context->repeats(calls, reasoning)) {
            request->recovery.discarded_tool_calls += static_cast<std::uint32_t>(calls.size());
            request->recovery.discarded_reasoning_tokens += request->output.reasoning_tokens();
            request->recovery_cause = "duplicate_tool_call";
            publish_recovery(request, RecoveryEventKind::RetryTriggered, request->recovery_cause);
            if (request->recovery.attempts >=
                    targets::qwen3_6::GenerationRecoveryContext::maximum_attempts ||
                request->budget->remaining() == 0) {
                recovery_exhausted(request, "no retry or output-token budget remains");
                return;
            }
            request->decode_ready     = false;
            request->recovery_pending = true;
            recovery_queue_.push_back(request);
            signal_control();
            return;
        }
        complete_success(request, reason);
    }

    bool resolve_round(const std::shared_ptr<Request>& request, const GeneratedRound& round,
                       bool cancel_at_boundary) {
        const TokenId token      = round.tokens.front();
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
            throw std::logic_error(
                "non-speculative sampler selected a suppressed model stop token");
        }
        if (decision.accepted_tokens != 1) {
            throw std::logic_error("prefill output policy did not accept its licensed token");
        }
        request->generated.push_back(token);
        instance_.program->resolve_prefill_lane(lane, decision.finished());
        request->budget->commit(1);
        ChannelLogprobs token_logprobs =
            collect_token_logprobs(*request, round.logprobs, 0, tokens);
        auto published = request->output.commit_preview();
        attach_token_logprobs(published, std::move(token_logprobs));
        if (!decision.finished()) { synchronize_stop_suppression(request, lane); }
        if (!request->first_token) { request->first_token = Clock::now(); }
        append_output(request, std::move(published), !decision.finished());
        if (decision.finished()) {
            finish_generation(request, decision.finish_reason);
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

    void synchronize_stop_suppression(const std::shared_ptr<Request>& request, std::uint32_t lane) {
        const bool suppress = !request->output.model_stop_tokens_allowed();
        if (suppress) {
            enable_stop_suppression(request, lane);
        } else if (request->stop_suppression_active) {
            instance_.program->clear_suppressed_tokens_lane(lane);
            request->stop_suppression_active = false;
        }
        instance_.program->set_typical_cycle_reasoning_lane(
            lane, generation_recovery_ && request->output.in_reasoning());
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
            if (copy_hold_ && copy_hold_->lane == lane && !build_round_membership().empty()) {
                // Draining here would wait for the hold's in-flight disk reads and copies
                // while other lanes decode; the worker loop finishes it once they settle.
                if (!copy_hold_->cancel_pending) {
                    instance_.program->begin_copy_hold_cancel();
                    copy_hold_->cancel_pending = true;
                }
                continue;
            }
            cancel_lane(lane);
        }
    }

    void cancel_lane(std::uint32_t lane) {
        const std::shared_ptr<Request> request = slots_[lane];
        if (copy_hold_ && copy_hold_->lane == lane) { drain_copy_hold_before_abort(); }
        if (request->decode_ready) {
            if (!instance_.program->retain_reusable_lane(lane)) {
                instance_.program->abort_lane(lane);
            }
        } else if (!instance_.program->revert_cancelled_prefill_lane(lane)) {
            instance_.program->abort_lane(lane);
        }
        if (prefill_lane_ && *prefill_lane_ == lane) {
            instance_.request_memory.deactivate();
            prefill_lane_.reset();
        }
        complete_cancelled(request);
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
                    account_left_pending(*it, now);
                    cancelled.push_back(*it);
                    it = pending_.erase(it);
                } else if (!(*it)->cache_fallback && now >= (*it)->deadline) {
                    account_left_pending(*it, now);
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
        if (request->recovery.attempts != 0) {
            request->recovery.prefill_tokens += step.processed_prompt_tokens;
        }
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
        if (request->recovery.attempts == 0) {
            request->begin = step.summary;
        } else {
            ++request->recovery.prefill_samples;
            publish_recovery(request, RecoveryEventKind::RetryPrefillComplete,
                             request->recovery_cause);
        }
        if (step.round.tokens.size() != 1) {
            throw std::logic_error("prefill did not license exactly one token");
        }
        if (!resolve_round(request, step.round, false)) { request->decode_ready = true; }
        return true;
    }

    void run_prefill_step() {
        if (!prefill_lane_) { throw std::logic_error("no request owns staged prefill"); }
        const std::uint32_t lane = *prefill_lane_;
        const auto request       = slots_[lane];
        if (request == nullptr || request->decode_ready) {
            throw std::logic_error("staged prefill lane has invalid request state");
        }
        const auto started           = Clock::now();
        const PrefillStepResult step = instance_.program->advance_prefill_lane(lane);
        if (request->recovery.attempts != 0) {
            request->recovery.prefill_seconds +=
                std::chrono::duration<double>(Clock::now() - started).count();
        }
        const bool cancel_at_boundary = request->cancelled.load(std::memory_order_acquire);
        if (resolve_prefill_step(request, step, cancel_at_boundary)) { publish_runtime_stats(); }
    }

    [[nodiscard]] std::vector<std::shared_ptr<Request>> pending_snapshot() const {
        std::lock_guard lock(queue_mutex_);
        return {pending_.begin(), pending_.end()};
    }

    void account_left_pending(const std::shared_ptr<Request>& request,
                              Clock::time_point now) noexcept {
        request->queued_seconds +=
            std::chrono::duration<double>(now - request->pending_since).count();
        request->pending_since = now;
    }

    void begin_copy_hold(const std::shared_ptr<Request>& request) noexcept {
        if (!request->copy_hold_started) { request->copy_hold_started = Clock::now(); }
    }

    void end_copy_hold(const std::shared_ptr<Request>& request) noexcept {
        if (!request || !request->copy_hold_started) { return; }
        request->copy_hold_seconds +=
            std::chrono::duration<double>(Clock::now() - *request->copy_hold_started).count();
        request->copy_hold_started.reset();
    }

    [[nodiscard]] static std::exception_ptr attach_recovery_stats(std::exception_ptr error,
                                                                  GenerationRecoveryStats stats) {
        if (!error) { return error; }
        try {
            std::rethrow_exception(error);
        } catch (const RequestError& request_error) {
            return std::make_exception_ptr(
                RequestError(request_error.kind(), request_error.what(), stats));
        } catch (...) { return error; }
    }

    [[nodiscard]] bool erase_pending(const std::shared_ptr<Request>& request) {
        std::lock_guard lock(queue_mutex_);
        const auto it = std::find(pending_.begin(), pending_.end(), request);
        if (it == pending_.end()) { return false; }
        pending_.erase(it);
        account_left_pending(request, Clock::now());
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

    // Host-tier prefix lookup is optional: when planning it runs out of host memory, the request
    // keeps lane and cold admission instead of failing. Other planning errors propagate.
    template <class Lookup>
    [[nodiscard]] static std::optional<Plan> optional_host_reuse_plan(Lookup&& lookup) {
        try {
            return std::optional<Plan>(std::forward<Lookup>(lookup)());
        } catch (const std::bad_alloc&) { return std::nullopt; }
    }

    void ensure_ram_candidate(const std::shared_ptr<Request>& request) {
        if (!request->options.execution.allow_prefix_reuse) {
            request->ram_plan.reset();
            return;
        }
        const std::uint64_t version = instance_.program->kv_ram_index_version();
        if (request->ram_index_version == version) { return; }
        request->ram_plan.reset();
        // Recording the index version even when the lookup ran out of host memory keeps
        // allocation pressure from re-planning the same index every scheduler pass.
        std::optional<Plan> plan = optional_host_reuse_plan([&] {
            return instance_.program->plan_ram_reuse(request->prompt, *request->base_plan);
        });
        if (plan && plan->summary().reusable_prompt_tokens > 0 &&
            plan->summary().ram_entry_id != 0 &&
            plan->summary().reuse_source == PrefixReuseSource::HostRam) {
            request->ram_plan = std::move(plan);
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
        // Recording the index version even when the lookup ran out of host memory keeps
        // allocation pressure from re-planning the same index every scheduler pass.
        std::optional<Plan> plan = optional_host_reuse_plan([&] {
            return instance_.program->plan_disk_reuse(request->prompt, *request->base_plan);
        });
        if (plan && plan->summary().reusable_prompt_tokens > 0 &&
            plan->summary().disk_entry_id != 0 &&
            plan->summary().reuse_source == PrefixReuseSource::HostDisk) {
            request->disk_plan = std::move(plan);
        }
        request->disk_index_version = version;
    }

    [[nodiscard]] std::optional<LaneChoice>
    find_fork_lane(const std::shared_ptr<Request>& request) {
        std::array<std::uint8_t, kMaximumConcurrency> protected_lanes{};
        bool any_active = false;
        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane]) {
                any_active = true;
                continue;
            }
            ensure_lane_plan(request, lane);
            if (request->preservation_ram_versions[lane] !=
                    instance_.program->kv_ram_index_version() ||
                request->preservation_lane_versions[lane] != lane_plan_versions_[lane]) {
                request->preservation_unavailable[lane] = false;
            }
            protected_lanes[lane] =
                request->lane_plans[lane].value().summary().reusable_prompt_tokens != 0;
        }
        ensure_ram_candidate(request);
        ensure_disk_candidate(request);
        std::optional<LaneChoice> selected;
        std::uint32_t selected_reuse = 0;
        unsigned selected_rank       = 0;
        auto consider                = [&](LaneChoice choice, const Plan& plan, unsigned rank) {
            if (choice.fork_source) { choice.protected_lanes[*choice.fork_source] = 1; }
            std::array<std::uint32_t, kMaximumConcurrency> victims{};
            std::size_t count = 0;
            for (std::uint32_t other = 0; other < max_concurrency_; ++other) {
                if (other != choice.lane && !slots_[other] && !choice.protected_lanes[other] &&
                    instance_.program->has_retained_lane(other)) {
                    victims[count++] = other;
                }
            }
            if (!instance_.program->can_admit_lane_after_releasing(
                    choice.lane, plan, std::span(victims).first(count))) {
                return;
            }
            choice.evict_retained = !instance_.program->can_admit_lane(choice.lane, plan);
            const auto reuse      = plan.summary().reusable_prompt_tokens;
            if (selected) {
                if (reuse < selected_reuse || (reuse == selected_reuse && rank > selected_rank)) {
                    return;
                }
                if (reuse == selected_reuse && rank == selected_rank) {
                    const bool dirty = instance_.program->has_retained_lane(choice.lane);
                    const bool selected_dirty =
                        instance_.program->has_retained_lane(selected->lane);
                    if (dirty && !selected_dirty) { return; }
                    if (dirty == selected_dirty &&
                        !instance_.program->retained_lane_precedes(choice.lane, selected->lane)) {
                        return;
                    }
                }
            }
            selected       = choice;
            selected_reuse = reuse;
            selected_rank  = rank;
        };
        for (std::uint32_t source = 0; source < max_concurrency_; ++source) {
            if (!protected_lanes[source]) { continue; }
            const Plan& plan = request->lane_plans[source].value();
            for (std::uint32_t destination = 0; destination < max_concurrency_; ++destination) {
                if (slots_[destination] || destination == source) { continue; }
                consider(LaneChoice{.lane = destination, .fork_source = source}, plan, 0);
            }
            if (!request->preservation_unavailable[source] &&
                instance_.program->retained_lane_cache_writable(source)) {
                consider(LaneChoice{.lane = source, .fork_source = source, .preserve_source = true},
                         plan, 3);
            }
        }
        for (std::uint32_t destination = 0; destination < max_concurrency_; ++destination) {
            if (slots_[destination]) { continue; }
            if (request->ram_plan) {
                consider(LaneChoice{.lane         = destination,
                                    .ram_entry_id = request->ram_plan->summary().ram_entry_id},
                         *request->ram_plan, 1);
            }
            if (request->disk_plan) {
                consider(LaneChoice{.lane          = destination,
                                    .disk_entry_id = request->disk_plan->summary().disk_entry_id},
                         *request->disk_plan, 2);
            }
            if (!protected_lanes[destination]) {
                consider(LaneChoice{.lane = destination, .protected_lanes = protected_lanes},
                         request->lane_plans[destination].value(), 4);
            }
        }
        if (!selected && !any_active) {
            throw RequestError(RequestErrorKind::Overloaded,
                               "preserving the cached source requires another lane or sufficient "
                               "RAM cache capacity");
        }
        return selected;
    }

    [[nodiscard]] std::optional<LaneChoice>
    find_admission_lane(const std::shared_ptr<Request>& request) {
        if (!request->options.execution.cache_write) { return find_fork_lane(request); }
        // Every plan of a request carries its base plan's KV entitlements, so lane feasibility
        // does not depend on the reuse source. Plan the optional host tiers only when some free
        // lane can take the request at all.
        bool lane_feasible = false;
        for (std::uint32_t lane = 0; lane < max_concurrency_ && !lane_feasible; ++lane) {
            if (slots_[lane] != nullptr) { continue; }
            ensure_lane_plan(request, lane);
            const Plan& plan = request->lane_plans[lane].value();
            lane_feasible = instance_.program->can_admit_lane(lane, plan) ||
                            instance_.program->can_admit_lane_after_retained_eviction(lane, plan);
        }
        if (!lane_feasible) { return std::nullopt; }
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

            for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
                if (slots_[lane] != nullptr) { continue; }
                const bool feasible =
                    after_eviction
                        ? instance_.program->can_admit_lane_after_retained_eviction(lane, host_plan)
                        : instance_.program->can_admit_lane(lane, host_plan);
                if (!feasible) { continue; }
                if (!instance_.program->has_retained_lane(lane)) { return lane; }
                if (!dirty || instance_.program->retained_lane_precedes(lane, *dirty)) {
                    dirty = lane;
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
                    if (!instance_.program->retained_lane_precedes(lane, selected->lane)) {
                        return;
                    }
                }
            }
            selected = LaneChoice{
                .lane = lane, .evict_retained = evict, .ram_entry_id = 0, .disk_entry_id = 0};
            selected_reuse = reuse;
            selected_dirty = dirty;
        };
        auto consider_host = [&](std::uint32_t reuse, std::uint64_t ram_id, std::uint64_t disk_id,
                                 std::optional<std::uint32_t> lane, bool evict) {
            if (!lane || reuse == 0) { return; }
            if (selected && reuse <= selected_reuse) { return; }
            selected       = LaneChoice{.lane           = *lane,
                                        .evict_retained = evict,
                                        .ram_entry_id   = ram_id,
                                        .disk_entry_id  = disk_id};
            selected_reuse = reuse;
            selected_dirty = instance_.program->has_retained_lane(*lane);
        };

        for (std::uint32_t lane = 0; lane < max_concurrency_; ++lane) {
            if (slots_[lane] != nullptr) { continue; }
            ensure_lane_plan(request, lane);
            const Plan& plan          = request->lane_plans[lane].value();
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
            const Plan& plan          = request->lane_plans[lane].value();
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
        detail::run_secondary_cleanup([&] {
            const auto copies = instance_.program->harvest_kv_ram_copy_seconds();
            request->kv_ram_save_seconds += copies.save;
            request->kv_ram_load_seconds += copies.load;
        });
        detail::run_secondary_cleanup([&] {
            const auto copies = instance_.program->harvest_kv_disk_copy_seconds();
            request->kv_disk_save_seconds += copies.save;
            request->kv_disk_load_seconds += copies.load;
            request->kv_disk_h2d_seconds += copies.h2d;
            cumulative_stats_.kv_disk_h2d_seconds += copies.h2d;
        });
    }

    // A RequestError rejects one request, and a CacheRestoreFailure is confined to the optional
    // host bytes one request tried to reuse. Anything else is an Engine invariant or device
    // failure.
    [[nodiscard]] static bool is_request_local_error(const std::exception_ptr& error) {
        if (!error) { return false; }
        try {
            std::rethrow_exception(error);
        } catch (const RequestError&) { return true; } catch (const CacheRestoreFailure&) {
            return true;
        } catch (...) { return false; }
    }

    // Settles the hold's copies and returns its host claims; a failed restore also drops its
    // entry so planning stops offering it. Every step is best-effort.
    void release_copy_hold_host(CopyHold& hold) noexcept {
        detail::run_secondary_cleanup([&] { instance_.program->cancel_disk_restore(); });
        detail::run_secondary_cleanup([&] { instance_.program->synchronize_all(); });
        harvest_kv_copy_seconds(hold.request);
        if (hold.ram_claimed && !hold.ram_consumed) {
            detail::run_secondary_cleanup(
                [&] { instance_.program->release_ram_entry(hold.ram_entry_id); });
            hold.ram_claimed = false;
        }
        if (hold.restore_failed && hold.discard_failed_source && hold.ram_hit &&
            !hold.ram_consumed) {
            detail::run_secondary_cleanup(
                [&] { instance_.program->discard_ram_capture(hold.ram_entry_id); });
        }
        if (hold.disk_claimed && !hold.disk_consumed) {
            detail::run_secondary_cleanup(
                [&] { instance_.program->release_disk_entry(hold.disk_entry_id); });
            hold.disk_claimed = false;
        }
        if (hold.restore_failed && hold.discard_failed_source && hold.disk_hit &&
            !hold.disk_consumed) {
            detail::run_secondary_cleanup(
                [&] { instance_.program->invalidate_disk_entry(hold.disk_entry_id); });
        }
    }

    // Victims captured for the hold leave VRAM even when the hold ends without its restore:
    // their bundles already live in the RAM tier. Best-effort, after the hold's copies settled.
    void evict_copy_hold_victims(CopyHold& hold) noexcept {
        if (hold.victims_evicted) { return; }
        detail::run_secondary_cleanup([&] {
            for (const std::uint32_t victim :
                 std::span(hold.victim_lanes).first(hold.victim_count)) {
                instance_.program->evict_retained_lane(victim);
                invalidate_lane_plans(victim);
            }
        });
        hold.victims_evicted = true;
    }

    void drain_copy_hold_before_abort() noexcept {
        if (!copy_hold_) { return; }
        CopyHold& hold           = *copy_hold_;
        const std::uint32_t lane = hold.lane;
        release_copy_hold_host(hold);
        end_copy_hold(hold.request);
        evict_copy_hold_victims(hold);
        instance_.program->abort_lane(lane);
        copy_hold_.reset();
    }

    // A request that owns a lane failed: release its copy-hold, lane, staged prefill, and slot,
    // then fail it. Returns whether the failure is request-local, so the Engine continues.
    [[nodiscard]] bool fail_admitted_request(const std::shared_ptr<Request>& request,
                                             std::uint32_t lane, const std::exception_ptr& error) {
        if (copy_hold_ && copy_hold_->request == request) { drain_copy_hold_before_abort(); }
        instance_.program->abort_lane(lane);
        if (prefill_lane_ && *prefill_lane_ == lane) {
            instance_.request_memory.deactivate();
            prefill_lane_.reset();
        }
        if (slots_[lane] == request) {
            slots_[lane].reset();
            invalidate_lane_plans(lane);
        }
        request->decode_ready     = false;
        request->recovery_pending = false;
        complete_error(request, error);
        return is_request_local_error(error);
    }

    static void mark_restore_failure(CopyHold& hold, const std::exception_ptr& error) {
        hold.restore_failed = true;
        try {
            std::rethrow_exception(error);
        } catch (const CacheRestoreFailure& failure) {
            hold.discard_failed_source = failure.kind() == CacheRestoreFailureKind::InvalidSource;
            hold.replan_restore        = failure.kind() == CacheRestoreFailureKind::StalePlan;
        }
    }

    // A host restore that failed before prefill consumed the prompt falls back to computing it.
    // A new request must not have produced anything yet; a retry keeps its lane and output.
    [[nodiscard]] static bool is_cache_restore_failure(const std::exception_ptr& error) {
        if (!error) { return false; }
        try {
            std::rethrow_exception(error);
        } catch (const CacheRestoreFailure&) { return true; } catch (...) {
            return false;
        }
    }

    [[nodiscard]] bool restore_failure_falls_back(const std::exception_ptr& error,
                                                  const Request& request) const {
        if (!is_cache_restore_failure(error)) { return false; }
        if (!copy_hold_ || (!copy_hold_->ram_hit && !copy_hold_->disk_hit) || !request.prompt ||
            prefill_lane_) {
            return false;
        }
        return copy_hold_->recovery_base ||
               (!request.cache_fallback && !request.begin && request.generated.empty());
    }

    // Fence queued readers before source cleanup or replanning. Invalid source bytes are
    // dropped; stale plans and destination failures keep the healthy source available.
    // Other lanes keep decoding while the failed restore's copies settle.
    [[nodiscard]] AdmissionProgress fail_host_restore(bool membership_empty,
                                                      const std::exception_ptr& error) {
        mark_restore_failure(copy_hold_.value(), error);
        if (membership_empty) { return finish_failed_restore(); }
        instance_.program->begin_copy_hold_cancel();
        return AdmissionProgress::CopyHold;
    }

    [[nodiscard]] AdmissionProgress finish_failed_restore() {
        CopyHold& hold                         = copy_hold_.value();
        const std::shared_ptr<Request> request = hold.request;
        const std::uint32_t lane               = hold.lane;
        release_copy_hold_host(hold);
        end_copy_hold(request);
        evict_copy_hold_victims(hold);
        if (hold.recovery_base) {
            // Refresh a stale host lookup once without restarting the output session or
            // its recovery attempt. Bound synchronous restore failures before cold fallback.
            const bool replan = hold.replan_restore && hold.recovery_stale_retries == 0;
            BasePlan base     = std::move(*hold.recovery_base);
            copy_hold_.reset();
            try {
                if (replan) {
                    release_recovery_lane(lane);
                    auto ram_plan = optional_host_reuse_plan(
                        [&] { return instance_.program->plan_ram_reuse(request->prompt, base); });
                    auto disk_plan = optional_host_reuse_plan(
                        [&] { return instance_.program->plan_disk_reuse(request->prompt, base); });
                    const auto tokens = [](const std::optional<Plan>& plan, bool ram) {
                        if (!plan) { return std::uint32_t{0}; }
                        const auto& summary = plan->summary();
                        const auto source =
                            ram ? PrefixReuseSource::HostRam : PrefixReuseSource::HostDisk;
                        return summary.reuse_source == source &&
                                       (ram ? summary.ram_entry_id : summary.disk_entry_id) != 0
                                   ? summary.reusable_prompt_tokens
                                   : 0;
                    };
                    const auto ram_tokens  = tokens(ram_plan, true);
                    const auto disk_tokens = tokens(disk_plan, false);
                    const bool ram         = ram_tokens >= disk_tokens;
                    if ((ram_tokens != 0 || disk_tokens != 0) &&
                        hold_recovery_restore(request, lane, ram,
                                              ram ? std::move(ram_plan.value())
                                                  : std::move(disk_plan.value()),
                                              base, 1)) {
                        return AdmissionProgress::CopyHold;
                    }
                }
                recovery_cold_prefill(request, lane, base);
            } catch (...) {
                if (!fail_admitted_request(request, lane, std::current_exception())) { throw; }
                return AdmissionProgress::ControlProgress;
            }
            return AdmissionProgress::RanGpuUnit;
        }
        // A new request relinquishes its reservation, then normal FIFO admission rechecks the
        // cold reservation against its peers. It re-enters the queue before anything else
        // changes, so a failed insert still fails it through its lane.
        try {
            std::lock_guard lock(queue_mutex_);
            const auto position =
                std::lower_bound(pending_.begin(), pending_.end(), request->queue_order,
                                 [](const auto& queued, std::uint64_t order) {
                                     return queued->queue_order < order;
                                 });
            pending_.insert(position, request);
            request->pending_since = Clock::now();
            published_waiting_requests_.store(static_cast<std::uint32_t>(pending_.size()),
                                              std::memory_order_relaxed);
        } catch (...) {
            if (!fail_admitted_request(request, lane, std::current_exception())) { throw; }
            return AdmissionProgress::ControlProgress;
        }
        const bool replan = hold.replan_restore;
        instance_.program->abort_lane(lane);
        copy_hold_.reset();
        slots_[lane].reset();
        invalidate_lane_plans(lane);
        release_planning_state(request);
        request->lane.reset();
        request->budget.reset();
        request->admission_resources    = {};
        request->remaining_service_work = 0;
        request->backfill_epoch         = 0;
        request->backfill_class         = BackfillClass::None;
        if (!replan) {
            request->cache_fallback = true;
            ++cumulative_stats_.kv_cache_fallbacks;
            request->options.execution.force_cold_prefill = true;
        }
        publish_runtime_stats();
        return AdmissionProgress::ControlProgress;
    }

    [[nodiscard]] AdmissionProgress admit_complete(bool membership_empty) {
        if (!copy_hold_) { throw std::logic_error("admit-complete requires copy-hold state"); }
        CopyHold& hold                         = *copy_hold_;
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

        std::exception_ptr error;
        try {
            if (!hold.victims_evicted) {
                if (!instance_.program->kv_ram_copies_ready()) {
                    if (!membership_empty) { return AdmissionProgress::CopyHold; }
                    instance_.program->wait_kv_ram_copies();
                }
                for (const std::uint32_t victim :
                     std::span(hold.victim_lanes).first(hold.victim_count)) {
                    instance_.program->evict_retained_lane(victim);
                    invalidate_lane_plans(victim);
                }
                hold.victims_evicted = true;
            }

            if (hold.fork_source && !hold.restored) {
                if (lane_plan_versions_[*hold.fork_source] != hold.fork_source_version) {
                    throw std::logic_error(
                        "resident fork source changed while transfer admission was held");
                }
                instance_.program->fork_retained_lane(lane, *hold.fork_source, hold.plan);
                hold.restored = true;
            }
            if (hold.ram_hit && !hold.restored) {
                // The restore reads the entry's block, which a capture may still be filling;
                // keep decoding members running until its copies have landed.
                if (!membership_empty && !instance_.program->ram_restore_ready(hold.ram_entry_id)) {
                    return AdmissionProgress::CopyHold;
                }
                instance_.program->restore_ram_entry(lane, hold.ram_entry_id, hold.plan);
                hold.restored = true;
            }
            if (hold.disk_hit && !hold.restored) {
                // Setup waits for other entries' window reads; keep decoding members
                // running until it can start without blocking.
                if (!membership_empty &&
                    !instance_.program->disk_restore_ready(hold.disk_entry_id)) {
                    return AdmissionProgress::CopyHold;
                }
                instance_.program->restore_disk_entry(lane, hold.disk_entry_id, hold.plan);
                hold.disk_restore_epoch = instance_.program->pending_disk_restore_ticket();
                hold.restored           = true;
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
            end_copy_hold(request);
            const auto prefill_started    = Clock::now();
            const PrefillStepResult first = instance_.program->start_prefill_lane(
                lane, std::move(request->prompt), std::move(hold.plan), transient,
                &request->output);
            if (hold.recovery_base) {
                request->recovery.prefill_seconds +=
                    std::chrono::duration<double>(Clock::now() - prefill_started).count();
            }
            const auto copies = instance_.program->harvest_kv_ram_copy_seconds();
            request->kv_ram_save_seconds += copies.save;
            request->kv_ram_load_seconds += copies.load;
            const auto disk_copies = instance_.program->harvest_kv_disk_copy_seconds();
            request->kv_disk_save_seconds += disk_copies.save;
            request->kv_disk_load_seconds += disk_copies.load;
            request->kv_disk_h2d_seconds += disk_copies.h2d;
            cumulative_stats_.kv_disk_h2d_seconds += disk_copies.h2d;
            if (hold.ram_hit) {
                instance_.program->finish_ram_restore(hold.ram_entry_id,
                                                      request->options.execution.cache_write
                                                          ? CacheReadIntent::Consume
                                                          : CacheReadIntent::Fork);
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
        } catch (...) { error = std::current_exception(); }
        if (restore_failure_falls_back(error, *request)) {
            return fail_host_restore(membership_empty, error);
        }
        if (copy_hold_ && is_cache_restore_failure(error)) {
            // The drain applies the source disposition after all readers finish.
            mark_restore_failure(copy_hold_.value(), error);
        }
        if (!copy_hold_) {
            if (ram_claimed && !ram_consumed) {
                detail::run_secondary_cleanup(
                    [&] { instance_.program->release_ram_entry(ram_entry_id); });
            }
            if (disk_claimed && !disk_consumed) {
                detail::run_secondary_cleanup(
                    [&] { instance_.program->release_disk_entry(disk_entry_id); });
            }
        }
        if (!fail_admitted_request(request, lane, error)) { std::rethrow_exception(error); }
        return AdmissionProgress::ControlProgress;
    }

    [[nodiscard]] AdmissionProgress admit_planned_request(const std::shared_ptr<Request>& request,
                                                          LaneChoice choice,
                                                          BackfillClass backfill_class,
                                                          std::uint64_t backfill_epoch) {
        if (!request->cache_fallback && Clock::now() >= request->deadline) {
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
        bool ram_hit             = choice.ram_entry_id != 0;
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
        } else if (!request->lane_plans[choice.fork_source.value_or(lane)]) {
            throw std::logic_error("selected admission lane has no request plan");
        }
        Plan& winning_plan = ram_hit ? request->ram_plan.value()
                             : disk_hit
                                 ? request->disk_plan.value()
                                 : request->lane_plans[choice.fork_source.value_or(lane)].value();

        bool ram_claimed  = false;
        bool disk_claimed = false;
        std::array<std::uint64_t, kMaximumConcurrency> captured_ram_ids{};
        std::size_t captured_ram_count = 0;
        auto release_host_if_needed    = [&]() {
            if (ram_claimed) {
                detail::run_secondary_cleanup(
                    [&] { instance_.program->release_ram_entry(choice.ram_entry_id); });
                ram_claimed = false;
            }
            if (disk_claimed) {
                detail::run_secondary_cleanup(
                    [&] { instance_.program->release_disk_entry(choice.disk_entry_id); });
                disk_claimed = false;
            }
        };
        auto rollback_ram_captures = [&]() {
            for (const std::uint64_t id : std::span(captured_ram_ids).first(captured_ram_count)) {
                detail::run_secondary_cleanup([&] { instance_.program->discard_ram_capture(id); });
            }
            captured_ram_count = 0;
        };

        // While other lanes decode, freeing host RAM for these captures must
        // not wait on a synchronous disk spill.
        bool others_decoding = false;
        for (std::uint32_t other = 0; other < max_concurrency_; ++other) {
            others_decoding = others_decoding || (other != lane && slots_[other] != nullptr);
        }
        const bool captures_lane = instance_.program->has_retained_lane(lane) &&
                                   (ram_hit || disk_hit || choice.fork_source ||
                                    winning_plan.summary().reusable_prompt_tokens == 0);
        // A full RAM tier is spilling an entry on the disk worker so that this
        // admission's captures can evict it instead of dropping it unsaved.
        // Keep decoding until the spill lands; nothing is claimed or captured yet.
        if (others_decoding && (choice.evict_retained || captures_lane) &&
            instance_.program->kv_ram_reclaim_pending()) {
            return AdmissionProgress::None;
        }
        auto defer_for_reclaim = [&]() {
            harvest_kv_copy_seconds(request);
            release_host_if_needed();
            rollback_ram_captures();
            return AdmissionProgress::None;
        };

        try {
            if (choice.preserve_source) {
                const auto source_lane        = choice.fork_source.value();
                bool deferred                 = false;
                std::uint64_t preserved_id    = 0;
                const auto ram_version_before = instance_.program->kv_ram_index_version();
                const bool preserved          = instance_.program->preserve_retained_lane(
                    source_lane, preserved_id, !others_decoding, deferred);
                if (deferred) { return AdmissionProgress::None; }
                if (!preserved || preserved_id == 0) {
                    const auto ram_version_after = instance_.program->kv_ram_index_version();
                    // Failed captures update drop counters and the RAM index. Those mutations
                    // must not make this request alternate indefinitely between failed sources.
                    for (std::uint32_t source = 0; source < max_concurrency_; ++source) {
                        if (request->preservation_unavailable[source] &&
                            request->preservation_ram_versions[source] == ram_version_before) {
                            request->preservation_ram_versions[source] = ram_version_after;
                        }
                    }
                    request->preservation_unavailable[source_lane]  = true;
                    request->preservation_ram_versions[source_lane] = ram_version_after;
                    request->preservation_lane_versions[source_lane] =
                        lane_plan_versions_[source_lane];
                    return AdmissionProgress::ControlProgress;
                }
                captured_ram_ids[captured_ram_count++] = preserved_id;
                choice.ram_entry_id                    = preserved_id;
                ram_hit                                = true;
            }
            if (disk_hit) {
                const auto disk_summary = winning_plan.summary();
                if (!instance_.program->claim_disk_entry(
                        choice.disk_entry_id, disk_summary.disk_execution_frontier,
                        disk_summary.disk_hash_f_lo, disk_summary.disk_hash_f_hi,
                        disk_summary.reusable_prompt_tokens, disk_summary.disk_reuse_path,
                        disk_summary.disk_committed_generation)) {
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
            std::array<std::uint32_t, kMaximumConcurrency> victims{};
            std::size_t victim_count = 0;
            if (choice.evict_retained) {
                while (!instance_.program->can_admit_lane_after_releasing(
                    lane, winning_plan, std::span(victims).first(victim_count))) {
                    std::optional<std::uint32_t> victim;

                    for (std::uint32_t retained_lane = 0; retained_lane < max_concurrency_;
                         ++retained_lane) {
                        if (retained_lane == lane || choice.protected_lanes[retained_lane] ||
                            slots_[retained_lane] != nullptr ||
                            !instance_.program->has_retained_lane(retained_lane)) {
                            continue;
                        }
                        if (std::find(victims.begin(), victims.begin() + victim_count,
                                      retained_lane) != victims.begin() + victim_count) {
                            continue;
                        }
                        if (!victim ||
                            instance_.program->retained_lane_precedes(retained_lane, *victim)) {
                            victim = retained_lane;
                        }
                    }
                    if (!victim) {
                        throw std::logic_error(
                            "selected admission cannot reclaim enough retained KV");
                    }
                    std::uint64_t ram_id = 0;
                    // Saving a completed prefix is optional. Even if its host
                    // image is dropped, this free lane can release its GPU pages.
                    // A capture held back by a pending disk spill is retried
                    // instead, keeping the victim's GPU pages until it lands.
                    bool deferred = false;
                    (void)instance_.program->capture_retained_lane(
                        *victim, &ram_id, !others_decoding, &deferred,
                        std::span(captured_ram_ids).first(captured_ram_count));
                    if (deferred) { return defer_for_reclaim(); }
                    if (ram_id != 0) { captured_ram_ids[captured_ram_count++] = ram_id; }
                    victims[victim_count++] = *victim;
                }
            }

            if (captures_lane && !choice.preserve_source) {
                std::uint64_t ram_id = 0;
                bool deferred        = false;
                (void)instance_.program->capture_retained_lane(
                    lane, &ram_id, !others_decoding, &deferred,
                    std::span(captured_ram_ids).first(captured_ram_count));
                if (deferred) { return defer_for_reclaim(); }
                if (ram_id != 0) { captured_ram_ids[captured_ram_count++] = ram_id; }
                if (std::find(victims.begin(), victims.begin() + victim_count, lane) ==
                    victims.begin() + victim_count) {
                    victims[victim_count++] = lane;
                }
            }

            if (choice.preserve_source) {
                victims[victim_count++] = lane;
                instance_.program->retarget_plan_to_ram(winning_plan, choice.ram_entry_id);
            }
            Plan selected_plan = std::move(winning_plan);
            if (ram_hit) {
                request->ram_plan.reset();
            } else if (disk_hit) {
                request->disk_plan.reset();
            } else {
                request->lane_plans[choice.fork_source.value_or(lane)].reset();
            }
            if (!erase_pending(request)) {
                const auto copies = instance_.program->harvest_kv_ram_copy_seconds();
                request->kv_ram_save_seconds += copies.save;
                request->kv_ram_load_seconds += copies.load;
                const auto disk_copies = instance_.program->harvest_kv_disk_copy_seconds();
                request->kv_disk_save_seconds += disk_copies.save;
                request->kv_disk_load_seconds += disk_copies.load;
                request->kv_disk_h2d_seconds += disk_copies.h2d;
                cumulative_stats_.kv_disk_h2d_seconds += disk_copies.h2d;
                release_host_if_needed();
                rollback_ram_captures();
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
                .request     = request,
                .lane        = lane,
                .plan        = std::move(selected_plan),
                .fork_source = choice.preserve_source ? std::nullopt : choice.fork_source,
                .fork_source_version =
                    choice.fork_source ? lane_plan_versions_[*choice.fork_source] : 0,
                .ram_hit       = ram_hit,
                .ram_entry_id  = choice.ram_entry_id,
                .ram_claimed   = ram_claimed,
                .disk_hit      = disk_hit,
                .disk_entry_id = choice.disk_entry_id,
                .disk_claimed  = disk_claimed,
                .victim_lanes  = victims,
                .victim_count  = victim_count,
                .needs_prefill = needs_prefill,
            });
            begin_copy_hold(request);
            ram_claimed  = false;
            disk_claimed = false;
            publish_runtime_stats();
        } catch (...) {
            const std::exception_ptr error = std::current_exception();
            detail::run_secondary_cleanup([&] { instance_.program->synchronize_all(); });
            harvest_kv_copy_seconds(request);
            release_host_if_needed();
            if (!(copy_hold_ && copy_hold_->lane == lane)) { rollback_ram_captures(); }
            const bool claimed = slots_[lane] == request;
            if (copy_hold_ && copy_hold_->lane == lane) { drain_copy_hold_before_abort(); }
            if (claimed) {
                instance_.program->abort_lane(lane);
                if (prefill_lane_ && *prefill_lane_ == lane) {
                    instance_.request_memory.deactivate();
                    prefill_lane_.reset();
                }
                slots_[lane].reset();
                invalidate_lane_plans(lane);
            } else if (erase_pending(request)) {
                clear_protection_if_head(request);
            }
            complete_error(request, error);
            if (!is_request_local_error(error)) { throw; }
            return AdmissionProgress::ControlProgress;
        }
        // The copy-hold owns the admission from here; admit_complete cleans up its own failures.
        if (build_round_membership().empty()) { return admit_complete(true); }
        return AdmissionProgress::CopyHold;
    }

    AdmissionProgress try_admit_one() {
        bool control_progress                              = false;
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
            if (!head->cache_fallback && Clock::now() >= head->deadline) {
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
                if (!candidate->cache_fallback && Clock::now() >= candidate->deadline) {
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
                const auto& selected_plan =
                    candidate_lane->ram_entry_id != 0 ? candidate->ram_plan
                    : candidate_lane->disk_entry_id != 0
                        ? candidate->disk_plan
                        : candidate->lane_plans[candidate_lane->fork_source.value_or(
                              candidate_lane->lane)];
                if (!selected_plan) {
                    throw std::logic_error("admission lane has no selected reuse plan");
                }
                const RequestPlanSummary& candidate_plan = selected_plan.value().summary();

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
                if (!instance_.program->retain_reusable_lane(lane)) {
                    instance_.program->abort_lane(lane);
                }
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
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            if (!round.cycle_exclusions[row]) { continue; }
            const auto& request = slots_[lanes[row]];
            const auto count    = ++request->cycle_exclusions;
            // Bound log traffic in a persistent cycle: first and powers of two;
            // the terminal event always reports the exact total.
            if ((count & (count - 1)) == 0) {
                publish_recovery(request, RecoveryEventKind::CycleExclusion, "reasoning_cycle");
            }
        }

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
            ChannelLogprobs token_logprobs;
            if (!cancelled[row]) {
                const std::size_t first_slot = row * round.row_stride;
                const auto row_tokens =
                    round.tokens.subspan(first_slot, static_cast<std::size_t>(accepted[row]));
                request->generated.insert(request->generated.end(), row_tokens.begin(),
                                          row_tokens.end());
                request->budget->commit(accepted[row]);
                consume_service_work(request, accepted[row]);
                token_logprobs =
                    collect_token_logprobs(*request, round.logprobs, first_slot, row_tokens);
            }
            auto published = request->output.commit_preview();
            attach_token_logprobs(published, std::move(token_logprobs));
            if (!terminal[row]) { synchronize_stop_suppression(request, lane); }
            if (!request->first_token && accepted[row] != 0) {
                request->first_token = Clock::now();
            }
            append_output(request, std::move(published), terminal[row] == 0);
            if (terminal[row]) {
                finish_generation(request, finish_reasons[row]);
            } else {
                recover_persistent_reasoning(request);
            }
        }
    }

    void fail_all(const std::exception_ptr& error) noexcept {
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
            const bool control_changed = control_dirty_.exchange(false, std::memory_order_acquire);
            if (stable_decode_epoch && !control_changed && recovery_queue_.empty()) {
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
            bool idle           = false;
            {
                std::scoped_lock lock(queue_mutex_);
                idle = !stopping_ && pending_.empty();
            }
            for (std::uint32_t lane = 0; idle && lane < max_concurrency_; ++lane) {
                idle = slots_[lane] == nullptr;
            }
            // Cache-tier calls take the tiers' own locks and may wait on disk
            // maintenance; submit() must never queue behind them on queue_mutex_.
            // Idle spill is opportunistic: if the readiness query or spill request fails, the
            // worker polls again below, and a broken device fails the next scheduled request.
            bool copies_ready = false;
            if (idle && !copy_hold_) {
                try {
                    copies_ready = instance_.program->kv_copies_ready();
                    if (copies_ready) { instance_.program->request_idle_spill(); }
                    // NOLINTNEXTLINE(bugprone-empty-catch): opportunistic spill; see above.
                } catch (...) {}
            }
            {
                std::unique_lock lock(queue_mutex_);
                if (idle && copies_ready) {
                    queue_cv_.wait(lock, [&] { return stopping_ || !pending_.empty(); });
                } else if (idle) {
                    queue_cv_.wait_for(lock, std::chrono::milliseconds(20),
                                       [&] { return stopping_ || !pending_.empty(); });
                }
                if (stopping_) {
                    lock.unlock();
                    {
                        std::scoped_lock execution_lock(execution_mutex_);
                        // A hold the caller already cancelled finishes as cancelled.
                        detail::run_secondary_cleanup([&] {
                            if (copy_hold_ && copy_hold_->cancel_pending) {
                                cancel_lane(copy_hold_->lane);
                            }
                        });
                        drain_copy_hold_before_abort();
                        detail::run_secondary_cleanup(
                            [&] { instance_.program->shutdown_kv_tiers(load_progress_); });
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

                // A cancelled or failed hold's fenced copies settle while the others decode.
                if (copy_hold_ && (copy_hold_->cancel_pending || copy_hold_->restore_failed)) {
                    if (!membership.empty() && !instance_.program->copy_hold_cancel_settled()) {
                        run_membership_decode();
                    } else if (copy_hold_->cancel_pending) {
                        cancel_lane(copy_hold_->lane);
                    } else {
                        (void)finish_failed_restore();
                    }
                    continue;
                }

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
                        if (progress == AdmissionProgress::CopyHold && !membership.empty() &&
                            !membership_contains(membership, copy_hold_.value().lane)) {
                            run_membership_decode();
                        }
                    }
                    continue;
                }

                if (prefill_lane_) {
                    run_prefill_step();
                    continue;
                }

                if (start_generation_recovery()) { continue; }

                if (have_pending &&
                    (membership.empty() || decode_admission_burst.allows_admission())) {
                    const AdmissionProgress progress = try_admit_one();
                    if (progress == AdmissionProgress::RanGpuUnit ||
                        progress == AdmissionProgress::CopyHold) {
                        if (!membership.empty()) { decode_admission_burst.consume_admission(); }
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
                    detail::run_secondary_cleanup(
                        [&] { instance_.program->shutdown_kv_tiers(load_progress_); });
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
    const std::uint32_t context_capacity_;
    LoadProgress load_progress_;
    const bool generation_recovery_;

    mutable std::mutex execution_mutex_;
    mutable std::mutex queue_mutex_;
    mutable std::mutex stats_mutex_;
    std::condition_variable queue_cv_;
    std::deque<std::shared_ptr<Request>> pending_;
    std::deque<std::shared_ptr<Request>> recovery_queue_;
    std::size_t outstanding_        = 0;
    std::uint64_t next_request_id_  = 1;
    std::uint64_t next_queue_order_ = 1;
    std::array<std::shared_ptr<Request>, kMaximumConcurrency> slots_{};
    std::optional<std::uint32_t> prefill_lane_;
    std::optional<CopyHold> copy_hold_;
    std::array<std::uint64_t, kMaximumConcurrency> lane_plan_versions_{};
    std::optional<AdmissionProtection> protection_;
    std::uint64_t next_protection_epoch_ = 1;
    RuntimeStats cumulative_stats_;
    RuntimeStats published_stats_;
    mutable RuntimeStats latest_disk_stats_; // disk fields only; guarded by stats_mutex_
    mutable std::uint64_t latest_disk_sequence_ = 0;
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

    friend int drive_scripted_recovery<Instance>(ConcurrentExecutor& executor);
};

} // namespace ninfer::runtime
