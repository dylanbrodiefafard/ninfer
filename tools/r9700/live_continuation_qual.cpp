#include "core/device.h"
#include "runtime/contract/types.h"
#include "targets/registry.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <exception>
#include <cstdlib>
#include <new>
#include <utility>
#include "targets/qwen3/impl/runtime/kv_ram_cache.h"
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// Fault only the test caller's next C++ allocation; HIP and host-copy workers keep their
// allocators. Each fault is armed after model startup at a settled transaction boundary.
thread_local bool fail_next_caller_allocation = false;
thread_local bool caller_allocation_failed    = false;

void* operator new(std::size_t bytes) {
    if (std::exchange(fail_next_caller_allocation, false)) {
        caller_allocation_failed = true;
        throw std::bad_alloc();
    }
    if (void* memory = std::malloc(bytes == 0 ? 1 : bytes)) { return memory; }
    throw std::bad_alloc();
}

void* operator new[](std::size_t bytes) { return ::operator new(bytes); }

// The replacement operator new uses malloc, so free is the matching release. GCC 13
// diagnoses the inlined std::allocator path despite that pairing.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"

void operator delete(void* memory) noexcept { std::free(memory); }

void operator delete(void* memory, std::size_t) noexcept { std::free(memory); }

void operator delete[](void* memory) noexcept { std::free(memory); }

void operator delete[](void* memory, std::size_t) noexcept { std::free(memory); }

#pragma GCC diagnostic pop

namespace {
using namespace ninfer;
constexpr std::uint32_t kCap    = 32;
constexpr std::uint32_t kOutput = 160;

enum class Boundary { BeforeCap, PendingClose, AfterClose };
enum class CancelTransfer { None, CaptureStarted, Parked, RestoreStarted };

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

bool at_boundary(const targets::qwen3::OutputSession& output, Boundary boundary,
                 std::uint32_t cap = kCap) {
    const auto stats = output.reasoning_limit_stats();
    switch (boundary) {
    case Boundary::BeforeCap:
        return stats.used_tokens == cap - 1 && output.in_reasoning();
    case Boundary::PendingClose:
        return stats.used_tokens == cap && output.reasoning_limit_forced_close_token().has_value();
    case Boundary::AfterClose:
        return stats.forced_closes == 1 && !output.in_reasoning();
    }
    throw std::logic_error("unknown qualification boundary");
}

std::vector<TokenId> suppressed_tokens(const targets::qwen3::OutputSession& output,
                                       const targets::qwen3::Frontend& frontend) {
    std::vector<TokenId> result;
    if (!output.model_stop_tokens_allowed()) {
        const auto& stops = frontend.default_stop_policy().token_ids;
        result.insert(result.end(), stops.begin(), stops.end());
    }
    if (const auto close = output.suppressed_reasoning_end_token()) { result.push_back(*close); }
    require(result.size() <= runtime::ResolvedExecutionOptions::kMaximumSuppressedTokens,
            "qualifier stop exclusions exceed the production sampling domain");
    return result;
}

struct Result {
    std::vector<TokenId> tokens;
    std::vector<std::uint32_t> logprob_bits;
    std::string reasoning;
    std::string content;
    ReasoningLimitStats policy;
    std::uint32_t boundary_output        = 0;
    std::uint32_t transfers              = 0;
    FinishReason finish                  = FinishReason::None;
    bool cancelled                       = false;
    std::uint32_t released_main_pages    = 0;
    std::uint32_t released_backend_pages = 0;
};

void publish(Result& result, targets::qwen3::PublishedOutput output) {
    for (const auto& delta : output) {
        require(!delta.tool_call, "generic continuation unexpectedly generated a tool event");
        (delta.channel == OutputChannel::Reasoning ? result.reasoning : result.content) +=
            delta.text;
    }
}

Result run(targets::Qwen3_8_27BInstance& instance, DeviceContext& device, Boundary boundary,
           bool transfer, std::uint64_t owner, bool sampled,
           CancelTransfer cancel = CancelTransfer::None, bool shrink_before_transfer = false,
           bool allocation_faults = false) {
    auto& frontend = instance.loaded->frontend;
    auto& program  = *instance.program;
    PromptInput input;
    input.options.enable_thinking = true;
    ChatMessage message;
    message.parts.push_back(MessagePart{
        .text =
            "Compute the sum of the squares of integers 1 through 100 using two independent "
            "methods, "
            "checking the intermediate arithmetic carefully. Then enumerate the first one hundred "
            "squares explicitly, in increasing order, with a short explanation of each group."});
    input.messages.push_back(std::move(message));
    auto prompt        = frontend.prepare(std::move(input));
    constexpr auto cap = kCap;
    StopPolicy stop;
    stop.include_model_defaults = false;
    OutputOptions output_options;
    output_options.thinking_token_budget = cap;
    output_options.top_logprobs          = 0;
    auto output = frontend.make_output_session(prompt, stop, output_options, false);
    runtime::ResolvedExecutionOptions execution;
    execution.requested_output_tokens    = kOutput;
    execution.reservation_window_tokens  = kOutput;
    execution.sampling.seed              = 17;
    execution.sampling.temperature       = sampled ? 0.7F : 0.0F;
    execution.sampling.presence_penalty  = sampled ? 0.2F : 0.0F;
    execution.sampling.frequency_penalty = sampled ? 0.1F : 0.0F;
    execution.sampling.p_less            = false;
    const auto initial_suppression       = suppressed_tokens(output, frontend);
    execution.suppressed_token_count     = static_cast<std::uint32_t>(initial_suppression.size());
    std::copy(initial_suppression.begin(), initial_suppression.end(),
              execution.suppressed_token_ids.begin());
    execution.allow_prefix_reuse = false;
    execution.force_cold_prefill = true;
    execution.cache_write        = false;
    execution.token_logprobs     = true;
    auto base                    = program.plan_request_base(prompt, execution);
    auto plan                    = program.plan_request_for_lane(0, prompt, base);
    const auto summary           = plan.summary();
    require(summary.effective_output_tokens == kOutput, "qualifier completion room was reduced");
    require(program.can_admit_lane(0, plan), "cold qualifier lane cannot be admitted");
    instance.request_memory.activate(summary.transient_bytes, summary.transient_alignment);
    auto step = program.start_prefill_lane(0, std::move(prompt), std::move(plan),
                                           instance.request_memory.region(), &output);
    while (!step.complete) { step = program.advance_prefill_lane(0); }
    require(step.round.tokens.size() == 1, "prefill did not produce one initial token");
    Result result;
    const auto first = output.preview(step.round.tokens, kOutput, FinishReason::OutputLimit);
    require(first.accepted_tokens == 1 && !first.reject_generated_round && !first.finished(),
            "initial qualifier token did not enter a live reasoning continuation");
    require(!step.round.logprobs.empty(), "prefill raw target logprob unavailable");
    const auto first_logprob = step.round.logprobs.token_logprobs[0];
    require(std::isfinite(first_logprob), "prefill raw target logprob is nonfinite");
    result.tokens.push_back(step.round.tokens[0]);
    result.logprob_bits.push_back(std::bit_cast<std::uint32_t>(first_logprob));
    program.resolve_prefill_lane(0, false);
    publish(result, output.commit_preview());
    const auto sync_suppression = [&](std::uint32_t current_lane) {
        const auto tokens = suppressed_tokens(output, frontend);
        if (tokens.empty()) {
            program.clear_suppressed_tokens_lane(current_lane);
        } else {
            program.set_suppressed_tokens_lane(current_lane, tokens);
        }
    };
    sync_suppression(0);
    instance.request_memory.deactivate();

    std::uint32_t lane = 0;
    bool observed      = false;
    while (result.tokens.size() < kOutput) {
        if (!observed && at_boundary(output, boundary, cap)) {
            observed               = true;
            result.boundary_output = static_cast<std::uint32_t>(result.tokens.size());
            if (transfer) {
                const auto before     = output.reasoning_limit_stats();
                const auto ram_before = program.kv_ram_snapshot().used_bytes;
                auto window           = program.plan_window(lane, kOutput);
                if (shrink_before_transfer) {
                    const auto full       = window;
                    const auto before_gpu = program.kv_gpu_snapshot();
                    const auto useful     = static_cast<std::uint32_t>(result.tokens.size()) + 1;
                    window                = program.plan_window(lane, useful);
                    require(window.admission.main_kv_pages <= full.admission.main_kv_pages &&
                                window.admission.backend_kv_pages <=
                                    full.admission.backend_kv_pages,
                            "smaller target window enlarged its entitlement");
                    require(window.admission.main_kv_pages >= before_gpu.main.mapped_pages &&
                                window.admission.backend_kv_pages >= before_gpu.spec.mapped_pages,
                            "target shrink crossed the represented physical frontier");
                    require(program.shrink_window(lane, window),
                            "unused target window did not shrink");
                    const auto after_gpu = program.kv_gpu_snapshot();
                    result.released_main_pages =
                        full.admission.main_kv_pages - window.admission.main_kv_pages;
                    result.released_backend_pages =
                        full.admission.backend_kv_pages - window.admission.backend_kv_pages;
                    require(before_gpu.main.entitled_pages ==
                                    after_gpu.main.entitled_pages + result.released_main_pages &&
                                before_gpu.spec.entitled_pages ==
                                    after_gpu.spec.entitled_pages + result.released_backend_pages,
                            "physical pool entitlement did not match the target shrink transition");
                    require(before_gpu.main.mapped_pages == after_gpu.main.mapped_pages &&
                                before_gpu.spec.mapped_pages == after_gpu.spec.mapped_pages,
                            "unused entitlement shrink discarded represented KV pages");
                    require(result.released_main_pages + result.released_backend_pages > 0,
                            "shrink qualifier fixture did not expose reclaimable future pages");
                }
                const auto cancel_transfer = [&](bool abort_source) {
                    // Exercise the asynchronous API boundary directly. discard waits for
                    // actual copies; this does not claim a copy was still physically in flight.
                    program.discard_suspended(owner);
                    if (abort_source) { program.abort_lane(lane); }
                    program.release_suspension_credit(owner);
                    device.synchronize_all();
                    require(program.kv_ram_snapshot().used_bytes == ram_before,
                            "cancelled live image leaked RAM or credit ownership");
                    require(program.reserve_suspension_credit(owner, window.suspension_bytes),
                            "cancelled transfer credit cannot be reused");
                    program.release_suspension_credit(owner);
                    require(program.kv_ram_snapshot().used_bytes == ram_before,
                            "reused transfer credit did not return to its original footprint");
                    result.policy    = output.reasoning_limit_stats();
                    result.cancelled = true;
                    return result;
                };
                require(program.reserve_suspension_credit(owner, window.suspension_bytes),
                        "could not fund live image credit");
                if (allocation_faults) {
                    using Ram             = targets::qwen3::detail::KVRamCache;
                    const auto gpu_before = program.kv_gpu_snapshot();
                    Ram::test_fail_next_capture_metadata_allocation();
                    require(!program.start_suspend(lane, owner) &&
                                !Ram::test_capture_metadata_failure_pending(),
                            "suspend capture allocation failure escaped or was not exercised");
                    const auto gpu_after = program.kv_gpu_snapshot();
                    require(gpu_before.main.entitled_pages == gpu_after.main.entitled_pages &&
                                gpu_before.spec.entitled_pages == gpu_after.spec.entitled_pages &&
                                gpu_before.main.mapped_pages == gpu_after.main.mapped_pages &&
                                gpu_before.spec.mapped_pages == gpu_after.spec.mapped_pages,
                            "failed suspension changed resident KV ownership");
                }
                require(program.start_suspend(lane, owner), "live capture did not start");
                if (cancel == CancelTransfer::CaptureStarted) { return cancel_transfer(true); }
                device.synchronize_all();
                require(program.suspension_ready(owner), "live capture did not settle");
                const auto captured_gpu = program.kv_gpu_snapshot();
                program.finish_suspend(lane, owner);
                if (cancel == CancelTransfer::Parked) { return cancel_transfer(false); }
                if (allocation_faults) {
                    fail_next_caller_allocation = true;
                    caller_allocation_failed    = false;
                }
                const auto restored_window = program.plan_suspended_window(
                    owner, shrink_before_transfer ? window.output_tokens : kOutput);
                if (allocation_faults) {
                    fail_next_caller_allocation = false;
                    require(!caller_allocation_failed, "suspended window planning allocated");
                    using Ram             = targets::qwen3::detail::KVRamCache;
                    const auto gpu_before = program.kv_gpu_snapshot();
                    Ram::test_fail_next_restore_metadata_allocation();
                    require(!program.start_resume(1, owner, restored_window) &&
                                !Ram::test_restore_metadata_failure_pending(),
                            "resume allocation failure escaped or retained the destination");
                    const auto gpu_after = program.kv_gpu_snapshot();
                    require(gpu_before.main.entitled_pages == gpu_after.main.entitled_pages &&
                                gpu_before.spec.entitled_pages == gpu_after.spec.entitled_pages &&
                                gpu_before.main.mapped_pages == gpu_after.main.mapped_pages &&
                                gpu_before.spec.mapped_pages == gpu_after.spec.mapped_pages,
                            "failed resume leaked partial destination KV ownership");
                }
                if (shrink_before_transfer) {
                    require(restored_window.admission.main_kv_pages >=
                                    captured_gpu.main.mapped_pages &&
                                restored_window.admission.backend_kv_pages >=
                                    captured_gpu.spec.mapped_pages,
                            "small suspended window dropped the captured mapped-page floor");
                }
                require(program.start_resume(1, owner, restored_window),
                        "cross-lane restore did not start");
                if (cancel == CancelTransfer::RestoreStarted) { return cancel_transfer(false); }
                device.synchronize_all();
                require(program.resume_ready(owner), "cross-lane restore did not settle");
                program.finish_resume(1, owner);
                if (shrink_before_transfer) {
                    const auto restored_gpu = program.kv_gpu_snapshot();
                    require(restored_gpu.main.mapped_pages == captured_gpu.main.mapped_pages &&
                                restored_gpu.spec.mapped_pages == captured_gpu.spec.mapped_pages,
                            "small restore changed the captured represented page set");
                    const auto full_window = program.plan_window(1, kOutput);
                    if (allocation_faults) {
                        const auto gpu_before       = program.kv_gpu_snapshot();
                        fail_next_caller_allocation = true;
                        caller_allocation_failed    = false;
                        const bool granted          = program.grant_window(1, full_window);
                        fail_next_caller_allocation = false;
                        require(caller_allocation_failed && !granted,
                                "window allocation fault escaped or committed a partial grant");
                        const auto gpu_after = program.kv_gpu_snapshot();
                        require(gpu_before.main.entitled_pages == gpu_after.main.entitled_pages &&
                                    gpu_before.spec.entitled_pages ==
                                        gpu_after.spec.entitled_pages &&
                                    gpu_before.main.mapped_pages == gpu_after.main.mapped_pages &&
                                    gpu_before.spec.mapped_pages == gpu_after.spec.mapped_pages,
                                "failed window grant changed current KV ownership");
                    }
                    require(
                        program.grant_window(1, full_window),
                        "small restored window could not regrow to its original output allowance");
                }
                program.release_suspension_credit(owner);
                lane = 1;
                ++result.transfers;
                const auto after = output.reasoning_limit_stats();
                require(before.used_tokens == after.used_tokens &&
                            before.forced_closes == after.forced_closes &&
                            before.limit_reached == after.limit_reached &&
                            at_boundary(output, boundary, cap),
                        "policy state changed during capture/restore");
            }
        }
        std::uint32_t remaining = kOutput - static_cast<std::uint32_t>(result.tokens.size());
        // Identical baseline/transfer cuts expose each exact settled semantic boundary while
        // retaining ordinary DFlash proposal rounds away from the boundary.
        if (!observed) {
            const auto used   = output.reasoning_limit_stats().used_tokens;
            const auto target = boundary == Boundary::BeforeCap ? cap - 1 : cap;
            remaining         = std::min(remaining, used < target ? target - used : 1U);
        }
        const std::array lanes{lane};
        const std::array budgets{runtime::RoundBudget{remaining}};
        const auto round = program.decode_batch(lanes, budgets);
        require(round.row_counts.empty() ||
                    (round.row_counts.size() == 1 && round.row_counts[0] > 0),
                "decode did not produce a licensed target prefix");
        const auto count = round.row_counts.empty() ? std::size_t{1}
                                                    : static_cast<std::size_t>(round.row_counts[0]);
        const auto tokens = round.tokens.first(count);
        const auto decision =
            output.preview(tokens, kOutput - static_cast<std::uint32_t>(result.tokens.size()),
                           FinishReason::OutputLimit);
        require(!decision.reject_generated_round && decision.accepted_tokens > 0,
                "qualifier encountered a rejected target round");
        require(!round.logprobs.empty(), "decode raw target logprob unavailable");
        for (std::uint32_t index = 0; index < decision.accepted_tokens; ++index) {
            const auto logprob = round.logprobs.token_logprobs[index];
            require(std::isfinite(logprob), "decode raw target logprob is nonfinite");
            result.tokens.push_back(tokens[index]);
            result.logprob_bits.push_back(std::bit_cast<std::uint32_t>(logprob));
        }
        const std::array accepted{decision.accepted_tokens};
        const std::array<std::uint8_t, 1> terminal{decision.finished() ? std::uint8_t{1}
                                                                       : std::uint8_t{0}};
        const std::array<std::uint8_t, 1> cancelled{0};
        program.resolve_pending_batch(lanes, accepted, terminal, cancelled);
        publish(result, output.commit_preview());
        sync_suppression(lane);
        if (decision.finished()) {
            result.finish = decision.finish_reason;
            break;
        }
    }
    result.policy = output.reasoning_limit_stats();
    program.abort_lane(lane);
    device.synchronize_all();
    require(observed, "model closed reasoning before required intervention boundary");
    require(result.tokens.size() == kOutput && result.finish == FinishReason::OutputLimit,
            "continuation did not reach its exact original output ceiling");
    require(result.tokens.size() - result.boundary_output >= 64,
            "fewer than 64 target tokens verified beyond the settled boundary");
    require(result.policy.used_tokens == cap && result.policy.forced_closes == 1 &&
                result.policy.retries == 0 && result.policy.limit_reached,
            "canonical close or cumulative policy ledger is incorrect");
    return result;
}

void compare(const Result& baseline, const Result& resumed, std::uint32_t expected_transfers = 1) {
    require(baseline.tokens == resumed.tokens,
            "cross-lane continuation changed sampled target token IDs");
    require(baseline.logprob_bits == resumed.logprob_bits,
            "cross-lane continuation changed raw target logprob bits");
    require(baseline.reasoning == resumed.reasoning && baseline.content == resumed.content &&
                baseline.finish == resumed.finish &&
                baseline.boundary_output == resumed.boundary_output,
            "cross-lane continuation changed publication or semantic boundary");
    require(resumed.transfers == expected_transfers && baseline.transfers == 0,
            "qualifier did not distinguish actual live transfer from the cold control");
}
} // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 3 || argc == 4,
                "usage: live_continuation_qual ARTIFACT.ninfer none|dflash|mtp "
                "[--cancel-transfers|--shrink-before-transfer|--allocation-failures]");
        const bool cancel_transfers =
            argc == 4 && std::string_view(argv[3]) == "--cancel-transfers";
        const bool allocation_faults =
            argc == 4 && std::string_view(argv[3]) == "--allocation-failures";
        const bool shrink_transfers =
            allocation_faults ||
            (argc == 4 && std::string_view(argv[3]) == "--shrink-before-transfer");
        require(argc == 3 || cancel_transfers || shrink_transfers, "unknown qualifier mode");
        const std::string_view backend = argv[2];
        require(backend == "none" || backend == "dflash" || backend == "mtp",
                "backend must be none, dflash, or mtp");
        require(backend != "mtp" || shrink_transfers,
                "MTP is qualified here only with --shrink-before-transfer");
        EngineOptions options;
        options.artifact_path         = argv[1];
        options.max_context           = 2048;
        options.kv_capacity           = KvCapacityPolicy::explicit_capacity(4096);
        options.max_concurrency       = 2;
        options.prefill_chunk         = 512;
        options.kv_ram_capacity_bytes = 2ULL << 30;
        options.generation_recovery   = false;
        options.speculative.backend   = backend == "dflash" ? SpeculativeBackend::DFlash
                                        : backend == "mtp"  ? SpeculativeBackend::Mtp
                                                            : SpeculativeBackend::None;
        if (backend != "none") {
            options.speculative.draft_tokens   = backend == "mtp" ? 3 : 7;
            options.speculative.adaptive_draft = false;
            options.speculative.proposal_head  = ProposalHead::Optimized;
        }
        DeviceContext device(options.device);
        auto target         = targets::construct_target(options, device);
        std::uint64_t owner = 1;
        if (cancel_transfers) {
            const auto baseline =
                run(*target.active, device, Boundary::PendingClose, false, owner++, true);
            for (const auto phase : {CancelTransfer::CaptureStarted, CancelTransfer::Parked,
                                     CancelTransfer::RestoreStarted}) {
                const auto cancelled =
                    run(*target.active, device, Boundary::PendingClose, true, owner++, true, phase);
                require(cancelled.cancelled && cancelled.policy.used_tokens == kCap &&
                            cancelled.policy.forced_closes == 0,
                        "transfer cancellation did not occur at the pending close");
                const auto fresh =
                    run(*target.active, device, Boundary::PendingClose, false, owner++, true);
                compare(baseline, fresh, 0);
                std::cout << "live_transfer_cancel: PASS backend=" << backend
                          << " phase=" << static_cast<int>(phase)
                          << " sampling=standard-t0.7-presence0.2-frequency0.1"
                          << " fresh_exact_ids_logprobs=true ram_credit_reusable=true"
                          << " physical_copy_inflight_not_asserted=true\n";
            }
        } else {
            for (const bool sampled : {false, true}) {
                for (const auto boundary :
                     {Boundary::BeforeCap, Boundary::PendingClose, Boundary::AfterClose}) {
                    const auto baseline = run(*target.active, device, boundary, false, owner++,
                                              sampled, CancelTransfer::None, false);
                    const auto resumed =
                        run(*target.active, device, boundary, true, owner++, sampled,
                            CancelTransfer::None, shrink_transfers, allocation_faults);
                    compare(baseline, resumed);
                    std::cout << "live_continuation: PASS backend=" << backend << " sampling="
                              << (sampled ? "standard-t0.7-presence0.2-frequency0.1" : "greedy")
                              << " boundary=" << static_cast<int>(boundary)
                              << " committed=" << resumed.tokens.size() << " after_boundary="
                              << resumed.tokens.size() - resumed.boundary_output
                              << " allocation_faults=" << allocation_faults
                              << " shrink_before_transfer=" << shrink_transfers
                              << " restore_shrunken_then_regrow=" << shrink_transfers
                              << " released_main_pages=" << resumed.released_main_pages
                              << " released_backend_pages=" << resumed.released_backend_pages
                              << " exact_sampled_ids=true exact_raw_logprob_bits=true "
                                 "lane0_to_lane1=true\n";
                }
            }
        }
        target.active->program->shutdown_kv_tiers();
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "live_continuation: FAIL " << error.what() << '\n';
        return 1;
    }
}
