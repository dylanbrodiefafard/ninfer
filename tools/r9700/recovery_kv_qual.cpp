// Manual real-artifact qualification of generation recovery on resident KV. It drives the exact
// production Program of the explicit Qwen3.8-27B artifact: prefill a thinking prompt, retain the
// lane, copy the prompt without its sampled token, and prefill only the recovery suffix from the
// resident checkpoint. It then captures the stacked prompt, aborts the lane three times, restores
// the RAM image onto the empty lane, and prefills without recomputing the restored prefix. Two
// decoded retries restore the prompt checkpoint after speculative decode overwrote the lane and
// must reproduce the committed GDN/hidden state and greedy continuation of an undecoded control.
#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "targets/qwen3_8_27b/impl/load/bindings.h"
#include "core/arena.h"
#include "core/device.h"
#include "targets/qwen3/impl/runtime/kv_ram_cache.h"
#include "targets/qwen3_8_27b/impl/variant.h"

#include <ninfer/targets/qwen3/frontend.h>
#include <ninfer/targets/qwen3/generation_recovery.h>
#include <ninfer/targets/qwen3/prepared_prompt.h>
#include <ninfer/targets/qwen3/startup_features.h>
#include <ninfer/targets/qwen3_8_27b/package.h>

#define NINFER_QWEN3_VARIANT    ::ninfer::targets::qwen3_8_27b::detail::Variant
#define NINFER_QWEN3_RUNTIME_NS qwen3_8_27b_r9700
#include "targets/qwen3/impl/runtime/linear_state_slots.h"
#include "targets/qwen3/impl/runtime/program.h"
#undef NINFER_QWEN3_RUNTIME_NS
#undef NINFER_QWEN3_VARIANT

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace target    = ninfer::targets::qwen3_8_27b::detail;
namespace family    = ninfer::targets::qwen3;
namespace execution = family::detail::qwen3_8_27b_r9700;
using Package       = ninfer::targets::qwen3_8_27b::Package;

constexpr std::size_t kRamBytes = 1024ULL * 1024ULL * 1024ULL;

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

std::string plan_text(const execution::RequestPlan& plan) {
    return "path=" + std::to_string(static_cast<int>(plan.impl_->reuse)) +
           " base=" + std::to_string(plan.impl_->reuse_base) +
           " reusable=" + std::to_string(plan.summary().reusable_prompt_tokens) +
           " source=" + std::to_string(static_cast<int>(plan.summary().reuse_source));
}

struct PrefillRun {
    ninfer::runtime::BeginSummary summary;
    std::uint32_t processed = 0;
};

struct CommittedState {
    std::vector<std::byte> conv;
    std::vector<std::byte> recurrent;
    std::vector<std::byte> hidden;

    bool operator==(const CommittedState&) const = default;
};

CommittedState committed_state(execution::ProgramImplCore& program) {
    const auto& state  = program.decoder->linear_attention;
    const auto& hidden = program.sequences[0].tail_hidden;
    CommittedState result{std::vector<std::byte>(state.conv_host_image_bytes()),
                          std::vector<std::byte>(state.recurrent_host_image_bytes()),
                          std::vector<std::byte>(hidden.bytes())};
    state.pack_slot_to_host(
        execution::LinearStateSlots::current_state_slot(0, program.max_concurrency),
        result.conv.data(), result.recurrent.data(), program.device.stream);
    HIP_CHECK(hipMemcpyAsync(result.hidden.data(), hidden.data, hidden.bytes(),
                             hipMemcpyDeviceToHost, program.device.stream));
    program.device.synchronize();
    return result;
}

std::vector<ninfer::TokenId> decode_rounds(execution::ProgramImplCore& program, int rounds) {
    const std::array<std::uint32_t, 1> lanes{0};
    const std::array<ninfer::runtime::RoundBudget, 1> budgets{{{16}}};
    const std::array<std::uint8_t, 1> flags{0};
    std::vector<ninfer::TokenId> tokens;
    for (int i = 0; i < rounds; ++i) {
        const auto round = program.decode_batch(lanes, budgets);
        // Ordinary decode commits exactly one token per row and reports no row counts.
        const std::int32_t committed = round.row_counts.empty() ? 1 : round.row_counts[0];
        require(committed > 0 && !round.tokens.empty(),
                "decode did not produce a committed candidate");
        const std::array<std::uint32_t, 1> accepted{static_cast<std::uint32_t>(committed)};
        tokens.insert(tokens.end(), round.tokens.begin(), round.tokens.begin() + accepted[0]);
        program.resolve_pending_batch(lanes, accepted, flags, flags);
    }
    return tokens;
}

PrefillRun finish_prefill(execution::ProgramImplCore& program, family::PreparedPromptData prompt,
                          execution::RequestPlan plan) {
    PrefillRun run;
    auto step     = program.start_prefill_lane(0, std::move(prompt), std::move(plan), {});
    run.summary   = step.summary;
    run.processed = step.processed_prompt_tokens;
    while (!step.complete) {
        step = program.advance_prefill_lane(0);
        run.processed += step.processed_prompt_tokens;
    }
    require(!step.round.tokens.empty(), "prefill completed without a sampled token");
    program.resolve_prefill_lane(0, false);
    return run;
}

bool prefix_equals(const std::vector<ninfer::TokenId>& tokens,
                   const std::vector<ninfer::TokenId>& prefix) {
    return tokens.size() >= prefix.size() &&
           std::equal(prefix.begin(), prefix.end(), tokens.begin());
}

void exercise_decoded_retries(execution::ProgramImplCore& program, family::Frontend& frontend,
                              const ninfer::PromptInput& input,
                              ninfer::runtime::ResolvedExecutionOptions options) {
    options.requested_output_tokens = 32;
    auto prompt                     = family::PreparedPromptAccess::take(frontend.prepare(input));
    const auto recovery             = family::GenerationRecoveryContext::analyze(input);
    for (std::uint32_t attempt = 1; attempt <= 2; ++attempt) {
        const auto prompt_tokens = static_cast<std::uint32_t>(prompt.token_ids.size());
        const auto insert        = recovery->recovery_insert({}, attempt);
        auto prepared_retry =
            frontend.splice_recovery_prompt(prompt.token_ids, input, insert, recovery, true);
        require(prepared_retry.has_value(), "decoded retry splice was rejected");
        auto retry = family::PreparedPromptAccess::take(std::move(prepared_retry.value()));

        // The control appends the same suffix to an unmodified prompt state. Both routes run
        // identical prefill chunks; only the second restores the checkpoint after speculative
        // decode overwrote the lane's KV, GDN, and proposal state.
        CommittedState expected;
        std::vector<ninfer::TokenId> expected_continuation;
        for (const bool failed_decode : {false, true}) {
            program.abort_lane(0);
            auto base = program.plan_request_base(prompt, options);
            auto plan = program.plan_request_for_lane(0, prompt, base);
            (void)finish_prefill(program, family::PreparedPromptData(prompt), std::move(plan));
            if (failed_decode) {
                (void)decode_rounds(program, 4);
                require(program.sequences[0].execution_frontier > prompt_tokens,
                        "failed decode did not advance beyond the recovery checkpoint");
            }
            require(program.retain_reusable_lane(0), "decoded lane could not be retained");
            auto retry_base = program.plan_request_base(retry, options);
            auto retry_plan = program.plan_request_for_lane(0, retry, retry_base);
            require(retry_plan.summary().reusable_prompt_tokens == prompt_tokens,
                    "decoded retry lost its prompt checkpoint: " + plan_text(retry_plan));
            require(failed_decode
                        ? execution::is_complete_checkpoint_restore(retry_plan.impl_->reuse)
                        : retry_plan.impl_->reuse == execution::ReusePath::AppendAtFrontier,
                    "fixture did not exercise append and checkpoint restore separately: " +
                        plan_text(retry_plan));
            const auto run =
                finish_prefill(program, family::PreparedPromptData(retry), std::move(retry_plan));
            require(run.processed == retry.token_ids.size() - prompt_tokens,
                    "decoded retry recomputed tokens before its checkpoint");
            const auto actual = committed_state(program);
            auto continuation = decode_rounds(program, 2);
            if (!failed_decode) {
                expected              = actual;
                expected_continuation = std::move(continuation);
            } else {
                require(actual == expected,
                        "checkpoint retry changed the committed GDN or hidden state");
                require(continuation == expected_continuation,
                        "checkpoint retry changed the greedy speculative continuation");
            }
        }
        prompt = std::move(retry);
        std::cout << "decoded retry attempt=" << attempt << " restored=" << prompt_tokens
                  << " state and continuation matched\n";
    }
}

void require_empty_lane(const execution::ProgramImplCore& program, const char* when) {
    const execution::SequenceState& sequence = program.sequences[0];
    const execution::RequestControl& request = program.requests[0];
    require(request.lifecycle == execution::Lifecycle::Empty, std::string(when) + " lifecycle");
    require(!sequence.retained && !sequence.kv && sequence.ledger.empty() &&
                sequence.execution_frontier == 0,
            std::string(when) + " still holds prompt or KV");
    std::vector<ninfer::TokenId> copied;
    std::uint32_t frontier = 0;
    require(!program.copy_reusable_prompt(0, 1, copied, frontier) && copied.empty(),
            std::string(when) + " copy still returned a prompt");
}

// A resident-only fault: change one output-head row scale, leaving every Text/MTP KV input
// finite. The artifact bytes remain untouched and the original scale is restored on every exit.
class OutputHeadFault {
public:
    explicit OutputHeadFault(const ninfer::Weight& head) {
        require(head.qtype == ninfer::QType::LQ4 && head.layout == ninfer::QuantLayout::Lq4N16K64 &&
                    head.scale_dtype == ninfer::DType::FP32 && head.n > 7 &&
                    head.scales != nullptr && head.scale_bytes >= 8U * sizeof(std::uint32_t),
                "nonfinite publication fixture requires the production LQ4/FP32-scale head");
        row_scale_ = const_cast<std::uint32_t*>(static_cast<const std::uint32_t*>(head.scales)) + 7;
        HIP_CHECK(hipMemcpy(&original_, row_scale_, sizeof(original_), hipMemcpyDeviceToHost));
    }

    OutputHeadFault(const OutputHeadFault&)            = delete;
    OutputHeadFault& operator=(const OutputHeadFault&) = delete;

    ~OutputHeadFault() {
        if (dirty_) {
            // Preserve the failure being unwound; the normal path checks restoration below.
            (void)hipMemcpy(row_scale_, &original_, sizeof(original_), hipMemcpyHostToDevice);
        }
    }

    void inject() {
        constexpr std::uint32_t nan = 0x7fc00000U;
        dirty_                      = true;
        HIP_CHECK(hipMemcpy(row_scale_, &nan, sizeof(nan), hipMemcpyHostToDevice));
    }

    void restore() {
        HIP_CHECK(hipMemcpy(row_scale_, &original_, sizeof(original_), hipMemcpyHostToDevice));
        dirty_ = false;
    }

private:
    std::uint32_t* row_scale_ = nullptr;
    std::uint32_t original_   = 0;
    bool dirty_               = false;
};

std::size_t require_graph_inventory(const execution::ProgramImplCore& program) {
    const std::array<const execution::DecodeGraphFamily*, 3> families{
        &program.ordinary_graphs, &program.mtp_graphs, &program.dflash_graphs};
    std::size_t definitions = 0;
    std::size_t executables = 0;
    for (const auto* family : families) {
        for (const auto& profile : family->profiles) {
            definitions += profile.definition.ready();
            executables += profile.executable.ready();
        }
    }
    require(definitions == program.expected_graph_definition_count &&
                executables == program.expected_graph_executable_count &&
                (program.use_device_graph ? executables > 0 : executables == 0),
            "decode graph inventory is not the complete startup-fixed family");
    return executables;
}

void exercise_nonfinite_publication(execution::ProgramImplCore& program, family::Frontend& frontend,
                                    const ninfer::PromptInput& input,
                                    ninfer::runtime::ResolvedExecutionOptions options,
                                    const ninfer::Weight& head) {
    constexpr std::string_view expected =
        "target returned nonfinite eligible logits before token publication";
    const auto graph_executables = require_graph_inventory(program);
    for (const int mode : {0, 1, 2}) {
        options.sampling.temperature = mode == 0 ? 0.0F : 1.0F;
        options.sampling.top_k       = 20;
        options.sampling.p_less      = mode == 2;
        options.allow_prefix_reuse   = false;
        options.force_cold_prefill   = true;
        for (const bool decode : {false, true}) {
            program.abort_lane(0);
            options.requested_output_tokens = decode ? 2 : 1;
            auto prompt = family::PreparedPromptAccess::take(frontend.prepare(input));
            if (decode) {
                auto base = program.plan_request_base(prompt, options);
                auto plan = program.plan_request_for_lane(0, prompt, base);
                (void)finish_prefill(program, family::PreparedPromptData(prompt), std::move(plan));
            }
            OutputHeadFault fault(head);
            fault.inject();
            bool rejected = false;
            try {
                if (decode) {
                    const std::array<std::uint32_t, 1> lanes{0};
                    const std::array<ninfer::runtime::RoundBudget, 1> budgets{{{1}}};
                    (void)program.decode_batch(lanes, budgets);
                } else {
                    auto base = program.plan_request_base(prompt, options);
                    auto plan = program.plan_request_for_lane(0, prompt, base);
                    (void)finish_prefill(program, std::move(prompt), std::move(plan));
                }
            } catch (const std::runtime_error& error) {
                if (std::string_view(error.what()) != expected) { throw; }
                rejected = true;
            }
            fault.restore();
            require(rejected, "nonfinite eligible logits returned a generated round");
            require_empty_lane(program, decode ? "nonfinite decode" : "nonfinite final prefill");

            // Re-admit on the same Program after restoring the weight. A fresh prefill and final
            // decode must succeed: this catches a sticky status surviving lane reuse/replay.
            options.requested_output_tokens = 2;
            auto clean      = family::PreparedPromptAccess::take(frontend.prepare(input));
            auto clean_base = program.plan_request_base(clean, options);
            auto clean_plan = program.plan_request_for_lane(0, clean, clean_base);
            (void)finish_prefill(program, std::move(clean), std::move(clean_plan));
            const std::array<std::uint32_t, 1> lanes{0};
            const std::array<ninfer::runtime::RoundBudget, 1> budgets{{{1}}};
            // In graph mode Program selects an immutable executable or throws; it has no eager
            // fallback. Assert startup inventory before and after the fault and clean replay.
            require(require_graph_inventory(program) == graph_executables,
                    "a fault changed the prepared graph inventory");
            const auto round = program.decode_batch(lanes, budgets);
            const auto count = round.row_counts.empty() ? 1 : round.row_counts[0];
            require(count == 1 && !round.tokens.empty(),
                    "clean final decode after a nonfinite fault did not return one token");
            const std::array<std::uint32_t, 1> accepted{1};
            const std::array<std::uint8_t, 1> terminal{1}, cancelled{0};
            program.resolve_pending_batch(lanes, accepted, terminal, cancelled);
            require(require_graph_inventory(program) == graph_executables,
                    "clean replay changed the prepared graph inventory");
        }
    }
    std::cout
        << "nonfinite output-head fault rejected before final-prefill/decode token publication "
           "for greedy/truncated/p-less sampling\n";
}

void exercise(const std::filesystem::path& artifact, ninfer::SpeculativeBackend backend,
              bool device_graph) {
    ninfer::DeviceContext device(0);
    ninfer::EngineOptions options;
    options.artifact_path         = artifact;
    options.max_context           = 4096;
    options.max_concurrency       = 1;
    options.kv_capacity           = ninfer::KvCapacityPolicy::explicit_capacity(4096);
    options.prefill_chunk         = 1024;
    options.kv_ram_capacity_bytes = kRamBytes;
    options.enable_vision         = false;
    // Eager is the default; the explicit graph run exercises the same recovery/fault sequence
    // through the production immutable executable inventory.
    options.use_device_graph          = device_graph;
    options.speculative.backend       = backend;
    options.speculative.draft_tokens  = backend == ninfer::SpeculativeBackend::None ? 0 : 3;
    options.speculative.proposal_head = ninfer::ProposalHead::Full;
    // A fixed draft width keeps the control independent of round-time estimates learned during
    // the deliberately discarded decode.
    options.speculative.adaptive_draft = false;

    ninfer::artifact::Reader reader(artifact);
    ninfer::artifact::Binder binder(reader);
    options.model_id                 = reader.identity().model_id;
    options.weights_id               = reader.identity().weights_id;
    options.artifact_tensor_identity = reader.tensor_content_identity(
        ninfer::targets::qwen3_8_27b::detail::cache_tensor_selections(
            reader, ninfer::targets::qwen3::startup_features(options)));
    const auto profile = Package::resolve_weights(reader.identity());
    auto load          = target::bind_artifact(binder, profile, family::startup_features(options));
    // Match Package::plan_load: these supported linear widths also back graph preparation.
    load.bindings.linear_widths = target::Variant::ExecutionState::eager_widths(
        std::min(options.prefill_chunk, options.max_context), options.max_concurrency,
        family::startup_verify_widths<target::DFlashConfig>(options));
    auto materialized = ninfer::artifact::materialize(reader, load.materialization, device);
    target::LoadedModelData model(std::move(load.bindings), std::move(materialized));
    auto frontend = family::make_frontend(model.frontend, false);
    device.synchronize();

    auto planner       = Package::make_sequence_planner(device, options, profile);
    const auto pages   = planner.capacity_curve().minimum_main_page_groups;
    auto sequence_plan = std::move(planner).finalize(pages);
    execution::ProgramImplCore program(model.runtime, *sequence_plan.impl_, device,
                                       std::make_unique<ninfer::HostPinnedArena>(kRamBytes));
    device.synchronize();

    ninfer::PromptInput input;
    ninfer::ChatMessage message;
    message.role = ninfer::ChatRole::User;
    message.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = "Say hello in one sentence.", .media = {}});
    input.messages.push_back(std::move(message));
    input.options.enable_thinking   = true;
    input.options.preserve_thinking = true;

    ninfer::runtime::ResolvedExecutionOptions request;
    request.requested_output_tokens    = 8;
    request.sampling.temperature       = 0.0F;
    request.allow_prefix_reuse         = true;
    request.force_cold_prefill         = false;
    request.capture_context_checkpoint = false;

    auto prepared = family::PreparedPromptAccess::take(frontend.prepare(input));
    const std::vector<ninfer::TokenId> prompt_ids = prepared.token_ids;
    const auto prompt_tokens                      = static_cast<std::uint32_t>(prompt_ids.size());
    require(prompt_tokens > 1, "thinking prompt was empty");

    auto base       = program.plan_request_base(prepared, request);
    auto first_plan = program.plan_request_for_lane(0, prepared, base);
    require(first_plan.summary().transient_bytes == 0, "text prompt requested transient storage");
    const PrefillRun first = finish_prefill(program, std::move(prepared), std::move(first_plan));
    require(first.processed == prompt_tokens, "first prefill did not consume the prompt");
    require(program.requests[0].lifecycle == execution::Lifecycle::Active,
            "committed prefill did not leave the lane active");
    require(prefix_equals(program.sequences[0].ledger, prompt_ids) &&
                program.sequences[0].ledger.size() > prompt_ids.size(),
            "ledger lost the prompt or the sampled token");
    require(program.sequences[0].rewrite_checkpoint.valid &&
                program.sequences[0].rewrite_checkpoint.frontier > 0 &&
                program.sequences[0].rewrite_checkpoint.frontier <= prompt_tokens,
            "prefill did not leave a rewrite checkpoint inside the prompt");

    require(program.retain_reusable_lane(0), "active lane was not retained");
    require(program.requests[0].lifecycle == execution::Lifecycle::Complete &&
                program.sequences[0].retained && program.sequences[0].kv.has_value(),
            "retain dropped the resident bundle");
    std::vector<ninfer::TokenId> copied;
    std::uint32_t copied_frontier = 0;
    require(program.copy_reusable_prompt(0, prompt_tokens, copied, copied_frontier),
            "retain left no prompt to copy");
    require(copied == prompt_ids,
            "copied prefix included generated tokens or dropped prompt tokens");
    require(copied_frontier == program.sequences[0].rewrite_checkpoint.frontier,
            "copied rewrite frontier does not match the resident checkpoint");

    const auto recovery = family::GenerationRecoveryContext::analyze(input);
    const auto insert   = recovery->recovery_insert({}, 1);
    auto spliced_prompt =
        frontend.splice_recovery_prompt(prompt_ids, input, insert, recovery, true);
    require(spliced_prompt.has_value(), "live thinking prompt refused the recovery splice");
    auto spliced = family::PreparedPromptAccess::take(std::move(spliced_prompt.value()));
    const std::vector<ninfer::TokenId> spliced_ids = spliced.token_ids;
    const auto spliced_tokens                      = static_cast<std::uint32_t>(spliced_ids.size());
    require(spliced_tokens > prompt_tokens && prefix_equals(spliced_ids, prompt_ids),
            "splice did not append to the copied prompt");

    auto retry_base    = program.plan_request_base(spliced, request);
    auto resident_plan = program.plan_request_for_lane(0, spliced, retry_base);
    // The live frontier is the prompt and the splice starts at the next token, so the whole
    // prompt is the hit. A shorter rewrite checkpoint must not win.
    require(resident_plan.impl_->reuse == execution::ReusePath::AppendAtFrontier &&
                resident_plan.impl_->reuse_base == prompt_tokens &&
                resident_plan.summary().reusable_prompt_tokens == prompt_tokens &&
                resident_plan.summary().reuse_source == ninfer::PrefixReuseSource::VramResident,
            "resident retry missed the retained prompt: prompt=" + std::to_string(prompt_tokens) +
                " rewrite=" + std::to_string(copied_frontier) + " " + plan_text(resident_plan));
    require(program.can_admit_lane(0, resident_plan), "retained lane cannot admit the suffix");
    const PrefillRun suffix =
        finish_prefill(program, family::PreparedPromptData(spliced), std::move(resident_plan));
    require(suffix.summary.reused_prompt_tokens == prompt_tokens &&
                suffix.summary.prefix_reuse_source == ninfer::PrefixReuseSource::VramResident &&
                suffix.summary.prefix_reuse_path == ninfer::PrefixReusePath::AppendAtFrontier &&
                suffix.processed == spliced_tokens - prompt_tokens,
            "suffix prefill recomputed the resident prefix");
    require(prefix_equals(program.sequences[0].ledger, prompt_ids),
            "suffix prefill replaced the resident prompt tokens");
    std::cout << "resident hit reused=" << prompt_tokens << " suffix=" << suffix.processed
              << " rewrite=" << copied_frontier << '\n';

    require(program.retain_reusable_lane(0), "suffix prefill was not retained for capture");
    std::vector<ninfer::TokenId> stacked;
    std::uint32_t stacked_frontier = 0;
    require(program.copy_reusable_prompt(0, spliced_tokens, stacked, stacked_frontier) &&
                stacked == spliced_ids,
            "attempt-2 copy lost the spliced prompt");
    const std::vector<ninfer::TokenId> captured_ledger = program.sequences[0].ledger;
    const std::uint32_t captured_frontier              = program.sequences[0].execution_frontier;
    const auto captured_rewrite                        = program.sequences[0].rewrite_checkpoint;
    // Fill the optional tier until capture must enumerate eviction candidates, then fault
    // exactly that bookkeeping. The retained GPU continuation must remain reusable.
    const auto initial_evictions = program.kv_ram_snapshot().evictions;
    for (unsigned attempt = 0;
         attempt < 64 && program.kv_ram_snapshot().evictions == initial_evictions; ++attempt) {
        require(program.capture_retained_lane(0), "capture-allocation fixture could not fill RAM");
        program.wait_kv_ram_copies();
    }
    require(program.kv_ram_snapshot().evictions > initial_evictions,
            "capture-allocation fixture never required reclaim");
    const auto before_allocation = program.kv_ram_snapshot();
    family::detail::KVRamCache::test_fail_next_eviction_snapshot_allocation();
    require(!program.capture_retained_lane(0) &&
                !family::detail::KVRamCache::test_eviction_snapshot_allocation_pending(),
            "optional capture bookkeeping allocation failure escaped or was not exercised");
    const auto after_allocation = program.kv_ram_snapshot();
    require(after_allocation.drops == before_allocation.drops + 1 &&
                after_allocation.captures == before_allocation.captures &&
                after_allocation.evictions == before_allocation.evictions &&
                program.sequences[0].retained && program.sequences[0].kv.has_value() &&
                program.sequences[0].ledger == captured_ledger,
            "capture allocation failure changed resident continuation or retained RAM sources");
    std::uint64_t entry_id = 0;
    require(program.capture_retained_lane(0, &entry_id) && entry_id != 0,
            "retained lane did not capture a RAM checkpoint");
    program.wait_kv_ram_copies();

    program.abort_lane(0);
    require_empty_lane(program, "first abort");
    program.abort_lane(0);
    program.abort_lane(0);
    require_empty_lane(program, "third abort");

    auto host_base = program.plan_request_base(spliced, request);
    auto host_plan = program.plan_ram_reuse(spliced, host_base);
    require(host_plan.summary().reuse_source == ninfer::PrefixReuseSource::HostRam &&
                host_plan.summary().ram_entry_id == entry_id &&
                host_plan.impl_->reuse != execution::ReusePath::FullReset &&
                host_plan.impl_->reuse_base > 0,
            "aborted lane's RAM image was not reusable: " + plan_text(host_plan));
    program.claim_ram_entry(entry_id);
    program.restore_ram_entry(0, entry_id, host_plan);
    program.wait_kv_ram_copies();
    program.wait_kv_ram_copies_on_compute();
    require(program.sequences[0].retained && program.sequences[0].kv.has_value() &&
                program.sequences[0].ledger == captured_ledger &&
                program.sequences[0].execution_frontier == captured_frontier &&
                program.sequences[0].rewrite_checkpoint.valid == captured_rewrite.valid &&
                program.sequences[0].rewrite_checkpoint.frontier == captured_rewrite.frontier,
            "RAM restore did not return the captured ledger and checkpoint");

    auto restored_base = program.plan_request_base(spliced, request);
    auto restored_plan = program.plan_request_for_lane(0, spliced, restored_base);
    require(restored_plan.impl_->reuse != execution::ReusePath::FullReset &&
                restored_plan.summary().reusable_prompt_tokens >= prompt_tokens &&
                restored_plan.summary().reuse_source == ninfer::PrefixReuseSource::VramResident,
            "restored bundle missed the spliced prefix: " + plan_text(restored_plan));
    require(program.can_admit_lane(0, restored_plan), "restored lane cannot admit");
    const std::uint32_t restored_reuse = restored_plan.summary().reusable_prompt_tokens;
    const PrefillRun restored =
        finish_prefill(program, std::move(spliced), std::move(restored_plan));
    require(restored.summary.prefix_reuse_path != ninfer::PrefixReusePath::FullReset &&
                restored.summary.reused_prompt_tokens == restored_reuse &&
                restored.processed == spliced_tokens - restored_reuse &&
                restored_reuse >= prompt_tokens,
            "prefill after restore recomputed the restored prefix");
    require(prefix_equals(program.sequences[0].ledger, spliced_ids),
            "prefill after restore lost the spliced prompt");
    program.finish_ram_restore(entry_id, ninfer::runtime::CacheReadIntent::Consume);
    std::cout << "restored hit reused=" << restored.summary.reused_prompt_tokens
              << " suffix=" << restored.processed << '\n';
    exercise_decoded_retries(program, frontend, input, request);
    exercise_nonfinite_publication(program, frontend, input, request, model.runtime.output_head);
}

ninfer::SpeculativeBackend parse_backend(std::string_view value) {
    if (value == "none") { return ninfer::SpeculativeBackend::None; }
    if (value == "mtp") { return ninfer::SpeculativeBackend::Mtp; }
    if (value == "dflash") { return ninfer::SpeculativeBackend::DFlash; }
    throw std::invalid_argument("backend must be none, mtp, or dflash");
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 3 && argc != 4) {
        std::cerr << "usage: ninfer_r9700_recovery_kv_qual ARTIFACT.ninfer none|mtp|dflash "
                     "[--device-graph]\n";
        return 2;
    }
    try {
        const std::filesystem::path artifact = argv[1];
        if (artifact.extension() != ".ninfer" || !std::filesystem::is_regular_file(artifact)) {
            throw std::invalid_argument("an explicit existing .ninfer artifact is required");
        }
        const bool device_graph = argc == 4;
        if (device_graph && std::string_view(argv[3]) != "--device-graph") {
            throw std::invalid_argument("the only optional argument is --device-graph");
        }
        exercise(artifact, parse_backend(argv[2]), device_graph);
        std::cout << "recovery KV qualification passed device_graph=" << device_graph << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ninfer_r9700_recovery_kv_qual: " << error.what() << '\n';
        return 1;
    }
}
