#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "ninfer/ops/gdn_history.h"
#include "targets/qwen3_6_27b/impl/variant.h"

#define NINFER_QWEN36_VARIANT    ::ninfer::targets::qwen3_6_27b::detail::Variant
#define NINFER_QWEN36_RUNTIME_NS qwen3_6_27b_runtime
#include "targets/qwen3_6/impl/runtime/program.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
namespace target    = ninfer::targets::qwen3_6_27b::detail;
namespace family    = ninfer::targets::qwen3_6;
namespace execution = family::detail::qwen3_6_27b_runtime;
using Package       = ninfer::targets::qwen3_6_27b::Package;

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

CommittedState committed_state(execution::ProgramImplCore& program, std::uint32_t lane = 0) {
    if (program.gdn_history && program.gdn_history_lengths[lane] != 0) {
        require(program.replay_records.has_value(), "GDN history has no replay records");
        const auto slot =
            execution::LinearStateSlots::current_state_slot(lane, program.max_concurrency);
        ninfer::ops::gdn_history_materialize(
            program.replay_records.value(), program.gdn_history.value(),
            program.decoder->linear_attention.all_layers_view(),
            std::span<const std::int32_t>(&slot, 1), program.device.stream);
        program.gdn_history_lengths[lane] = 0;
    }
    const auto& state  = program.decoder->linear_attention;
    const auto& hidden = program.sequences[lane].tail_hidden;
    CommittedState result{std::vector<std::byte>(state.conv_host_image_bytes()),
                          std::vector<std::byte>(state.recurrent_host_image_bytes()),
                          std::vector<std::byte>(hidden.bytes())};
    state.pack_slot_to_host(static_cast<std::int32_t>(lane), result.conv.data(),
                            result.recurrent.data(), program.device.stream);
    CUDA_CHECK(cudaMemcpyAsync(result.hidden.data(), hidden.data, hidden.bytes(),
                               cudaMemcpyDeviceToHost, program.device.stream));
    program.device.synchronize();
    return result;
}

std::vector<std::byte> kv_prefix_image(execution::ProgramImplCore& program,
                                       const ninfer::PagedKVAllocation& allocation,
                                       const ninfer::PagedKVPool& pool, std::uint32_t pages) {
    const auto page_bytes = ninfer::paged_kv_logical_page_bytes(pool);
    std::vector<std::byte> image(page_bytes * pages);
    for (std::uint32_t page = 0; page < pages; ++page) {
        ninfer::pack_paged_kv_logical_page_to_host(
            allocation, pool, page, image.data() + page_bytes * page, program.device.stream);
    }
    program.device.synchronize();
    return image;
}

void exercise_exact_fork(execution::ProgramImplCore& program, family::Frontend& frontend,
                         ninfer::runtime::ResolvedExecutionOptions options, bool rewrite) {
    auto& source                 = program.sequences[0];
    const auto source_state      = committed_state(program);
    const auto source_ledger     = source.ledger;
    const auto source_frontier   = source.execution_frontier;
    const auto source_checkpoint = source.rewrite_checkpoint;
    const auto frontier          = rewrite ? source_checkpoint.frontier : source_frontier;
    require(frontier > 0 && source_ledger.size() > frontier,
            "exact fork source has no continuation");
    std::vector<ninfer::TokenId> branch(source_ledger.begin(), source_ledger.begin() + frontier);
    branch.push_back((source_ledger[frontier] + 1) % 248077);
    auto prompt = family::PreparedPromptAccess::take(frontend.prepare_tokens(std::move(branch)));
    options.cache_write = false;
    auto base           = program.plan_request_base(prompt, options);
    auto plan           = program.plan_request_for_lane(0, prompt, base);
    require(plan.impl_->reuse_base == frontier &&
                plan.impl_->reuse == (rewrite ? execution::restore_path(source_checkpoint.kind)
                                              : execution::ReusePath::AppendAtFrontier),
            "exact fork did not select its source state: rewrite=" + std::to_string(rewrite) +
                " frontier=" + std::to_string(frontier) +
                " dflash=" + std::to_string(source.dflash_context_frontier) +
                " tail=" + std::to_string(source.tail_hidden_valid) + " matches=" +
                std::to_string(family::detail::prefix_matches(prompt, source.ledger,
                                                              source.prefix_identity, frontier)) +
                " " + plan_text(plan));
    const auto& source_kv = source.kv.value();
    const auto text_pages = 1U + (frontier - 1U) / 64U;
    const auto text_image =
        kv_prefix_image(program, source_kv.text, program.decoder->text_kv.pool(), text_pages);
    auto* backend_cache =
        program.speculative_backend == ninfer::SpeculativeBackend::Mtp
            ? program.decoder->mtp_cache()
            : (program.dflash && program.dflash->full ? &*program.dflash->full : nullptr);
    std::vector<std::byte> backend_image;
    const auto backend_tokens =
        program.speculative_backend == ninfer::SpeculativeBackend::Mtp ? frontier - 1 : frontier;
    const auto backend_pages = backend_tokens == 0 ? 0 : 1U + (backend_tokens - 1U) / 64U;
    if (source_kv.backend) {
        backend_image =
            kv_prefix_image(program, *source_kv.backend, backend_cache->pool(), backend_pages);
    }
    CommittedState expected = source_state;
    std::vector<std::byte> cyclic;
    if (program.dflash) {
        cyclic.resize(program.dflash->local.lane_host_bytes());
        program.dflash->local.copy_lane_to_host(0, cyclic.data(), program.device.stream);
        program.device.synchronize();
    }
    if (rewrite) {
        CUDA_CHECK(cudaEventSynchronize(source.rewrite_image.copies_done));
        std::memcpy(expected.conv.data(), source.rewrite_image.conv.data(), expected.conv.size());
        std::memcpy(expected.recurrent.data(), source.rewrite_image.recurrent.data(),
                    expected.recurrent.size());
        CUDA_CHECK(cudaMemcpy(expected.hidden.data(), source.rewrite_checkpoint_hidden.data,
                              expected.hidden.size(), cudaMemcpyDeviceToHost));
        if (program.dflash) {
            std::memcpy(cyclic.data(), source.rewrite_image.dflash.data(), cyclic.size());
        }
    }
    program.fork_retained_lane(1, 0, plan);
    program.device.synchronize_all();
    auto& destination          = program.sequences[1];
    const auto& destination_kv = destination.kv.value();
    require(committed_state(program, 1) == expected,
            "fork did not copy exact GDN convolution, FP32 recurrence, and hidden state");
    require(kv_prefix_image(program, destination_kv.text, program.decoder->text_kv.pool(),
                            text_pages) == text_image,
            "fork changed represented main KV bytes");
    for (const auto source_page : source_kv.text.page_ids()) {
        require(std::find(destination_kv.text.page_ids().begin(),
                          destination_kv.text.page_ids().end(),
                          source_page) == destination_kv.text.page_ids().end(),
                "fork shared a physical main KV page");
    }
    if (destination_kv.backend) {
        require(kv_prefix_image(program, *destination_kv.backend, backend_cache->pool(),
                                backend_pages) == backend_image,
                "fork changed represented backend KV bytes");
    }
    if (program.dflash) {
        std::vector<std::byte> actual(cyclic.size());
        program.dflash->local.copy_lane_to_host(1, actual.data(), program.device.stream);
        program.device.synchronize();
        require(actual == cyclic, "fork changed the DFlash cyclic local image");
    }
    require(source.ledger == source_ledger && source.execution_frontier == source_frontier &&
                source.rewrite_checkpoint.valid == source_checkpoint.valid &&
                source.rewrite_checkpoint.frontier == source_checkpoint.frontier &&
                committed_state(program) == source_state,
            "fork modified source state or checkpoint ownership");
    program.abort_lane(1);
}

std::vector<ninfer::TokenId> decode_rounds(execution::ProgramImplCore& program, int rounds,
                                           bool terminal = false) {
    const std::array<std::uint32_t, 1> lanes{0};
    const std::array<ninfer::runtime::RoundBudget, 1> budgets{{{16}}};
    const std::array<std::uint8_t, 1> flags{0};
    std::vector<ninfer::TokenId> tokens;
    for (int i = 0; i < rounds; ++i) {
        const auto round = program.decode_batch(lanes, budgets);
        require(round.row_counts[0] > 0, "decode did not produce a committed candidate");
        const std::array<std::uint32_t, 1> accepted{
            static_cast<std::uint32_t>(round.row_counts[0])};
        tokens.insert(tokens.end(), round.tokens.begin(), round.tokens.begin() + accepted[0]);
        const std::array<std::uint8_t, 1> final_flags{
            static_cast<std::uint8_t>(terminal && i + 1 == rounds)};
        program.resolve_pending_batch(lanes, accepted, final_flags, flags);
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
            frontend.splice_recovery_prompt(prompt.token_ids, input, insert, recovery);
        require(prepared_retry.has_value(), "decoded retry splice was rejected");
        auto retry = family::PreparedPromptAccess::take(std::move(prepared_retry).value());

        // The control run appends the exact same suffix to an unmodified prompt
        // state. Both routes execute identical prefill chunks; the only difference
        // is restoring the saved state after speculative decode has overwritten it.
        CommittedState expected;
        std::vector<ninfer::TokenId> expected_continuation;
        for (const bool failed_decode : {false, true}) {
            program.abort_lane(0);
            auto base = program.plan_request_base(prompt, options);
            auto plan = program.plan_request_for_lane(0, prompt, base);
            (void)finish_prefill(program, family::PreparedPromptData(prompt), std::move(plan));
            if (failed_decode) {
                (void)decode_rounds(program, 4);
                require(program.sequences[0].text_kv_valid > prompt_tokens,
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
                    "fixture did not exercise append and checkpoint restore separately");
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
                        "checkpoint retry changed greedy speculative continuation");
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
                sequence.text_kv_valid == 0,
            std::string(when) + " still holds prompt or KV");
    std::vector<ninfer::TokenId> copied;
    std::uint32_t frontier = 0;
    require(!program.copy_reusable_prompt(0, 1, copied, frontier) && copied.empty(),
            std::string(when) + " copy still returned a prompt");
}

void exercise_full_original_preservation(execution::ProgramImplCore& program,
                                         family::Frontend& frontend, ninfer::PromptInput input,
                                         ninfer::runtime::ResolvedExecutionOptions options) {
    program.abort_lane(0);
    input.options.preserve_thinking = false;
    auto prompt                     = family::PreparedPromptAccess::take(frontend.prepare(input));
    auto base                       = program.plan_request_base(prompt, options);
    auto plan                       = program.plan_request_for_lane(0, prompt, base);
    (void)finish_prefill(program, std::move(prompt), std::move(plan));
    (void)decode_rounds(program, 2, true);
    require(program.retain_reusable_lane(0), "closed original could not be retained");
    program.mark_turn_closed(0);
    exercise_exact_fork(program, frontend, options, false);
    exercise_exact_fork(program, frontend, options, true);
    const auto& source    = program.sequences[0];
    const auto ledger     = source.ledger;
    const auto frontier   = source.execution_frontier;
    const auto checkpoint = source.rewrite_checkpoint;
    const auto state      = committed_state(program);
    require(checkpoint.valid && checkpoint.frontier < frontier &&
                source.closure_frontier == checkpoint.frontier && source.turn_closed,
            "closed original fixture would not use rewrite-cut capture");
    std::uint64_t cut_id = 0;
    require(program.capture_retained_lane(0, &cut_id) && cut_id != 0,
            "normal closed original did not capture");
    std::uint64_t full_id = 0;
    bool deferred         = false;
    require(program.preserve_retained_lane(0, full_id, true, deferred) && full_id != 0 && !deferred,
            "full original preservation did not capture");
    program.wait_kv_ram_copies();
    auto branch         = family::PreparedPromptAccess::take(frontend.prepare_tokens(ledger));
    options.cache_write = false;
    auto fork_base      = program.plan_request_base(branch, options);
    auto fork_plan      = program.plan_ram_reuse(branch, fork_base);
    require(fork_plan.summary().ram_entry_id == full_id && fork_plan.impl_->reuse_base == frontier,
            "full preservation was shortened to the normal closed-turn cut");
    program.claim_ram_entry(full_id);
    program.restore_ram_entry(1, full_id, fork_plan);
    program.wait_kv_ram_copies();
    program.wait_kv_ram_copies_on_compute();
    program.device.synchronize_all();
    require(program.sequences[1].ledger == ledger &&
                program.sequences[1].execution_frontier == frontier &&
                !program.sequences[1].rewrite_checkpoint.valid &&
                committed_state(program, 1) == state,
            "full original preservation lost current state or its existing checkpoint");
    require(program.kv_ram_cache_.has_value(), "full original has no RAM cache");
    const auto saved_host = program.kv_ram_cache_.value().load_host(full_id);
    require(saved_host.rewrite_valid && saved_host.rewrite_frontier == checkpoint.frontier,
            "full original image lost its rewrite checkpoint");
    program.finish_ram_restore(full_id, ninfer::runtime::CacheReadIntent::Fork);
    auto still_saved = program.plan_ram_reuse(branch, fork_base);
    require(still_saved.summary().ram_entry_id == full_id, "fork consumed the full original image");
    program.abort_lane(1);
    program.discard_ram_capture(cut_id);
    program.discard_ram_capture(full_id);
}

void exercise(const char* artifact, ninfer::SpeculativeBackend backend, bool forks_only) {
    ninfer::DeviceContext device;
    ninfer::EngineOptions options;
    options.artifact_path         = artifact;
    options.max_context           = 4096;
    options.max_concurrency       = 2;
    options.kv_capacity           = ninfer::KvCapacityPolicy::explicit_capacity(4096);
    options.prefill_chunk         = 1024;
    options.kv_cache              = ninfer::KvCacheStorage::Nvfp4;
    options.kv_ram_capacity_bytes = kRamBytes;
    options.enable_vision         = false;
    // Decode-graph capture is measured by its own real tests. This run is the
    // retain/copy/abort/restore sequence, and the graph allowance check is a
    // device-wide free-memory delta that moves when another allocation lands.
    options.use_cuda_graph            = false;
    options.speculative.backend       = backend;
    options.speculative.draft_tokens  = 3;
    options.speculative.proposal_head = ninfer::ProposalHead::Optimized;
    // Fixed draft width keeps the restoration control independent of round-time
    // estimates learned during the deliberately discarded decode.
    options.speculative.adaptive_draft = false;

    ninfer::artifact::Reader reader(artifact);
    ninfer::artifact::Binder binder(reader);
    options.model_id               = reader.identity().model_id;
    options.weights_id             = reader.identity().weights_id;
    options.artifact_file_identity = reader.file_identity();
    const auto profile             = Package::resolve_weights(reader.identity(), binder);
    auto load         = target::bind_artifact(binder, profile, family::startup_features(options));
    auto materialized = ninfer::artifact::materialize(reader, load.materialization, device);
    target::LoadedModelData model(load.bindings, std::move(materialized));
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

    ninfer::runtime::ResolvedExecutionOptions execution;
    execution.requested_output_tokens    = 8;
    execution.sampling.temperature       = 0.0F;
    execution.allow_prefix_reuse         = true;
    execution.force_cold_prefill         = false;
    execution.capture_context_checkpoint = false;

    auto prepared = family::PreparedPromptAccess::take(frontend.prepare(input));
    const std::vector<ninfer::TokenId> prompt_ids = prepared.token_ids;
    const auto prompt_tokens                      = static_cast<std::uint32_t>(prompt_ids.size());
    require(prompt_tokens > 1, "thinking prompt was empty");

    auto base       = program.plan_request_base(prepared, execution);
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
    exercise_exact_fork(program, frontend, execution, false);
    if (forks_only) {
        exercise_full_original_preservation(program, frontend, input, execution);
        return;
    }
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
    auto spliced_prompt = frontend.splice_recovery_prompt(prompt_ids, input, insert, recovery);
    require(spliced_prompt.has_value(), "live thinking prompt refused the recovery splice");
    auto spliced = family::PreparedPromptAccess::take(std::move(spliced_prompt).value());
    const std::vector<ninfer::TokenId> spliced_ids = spliced.token_ids;
    const auto spliced_tokens                      = static_cast<std::uint32_t>(spliced_ids.size());
    require(spliced_tokens > prompt_tokens && prefix_equals(spliced_ids, prompt_ids),
            "splice did not append to the copied prompt");

    auto retry_base    = program.plan_request_base(spliced, execution);
    auto resident_plan = program.plan_request_for_lane(0, spliced, retry_base);
    // The live frontier is the prompt. The splice starts at the next token, so the
    // whole prompt is the hit. A shorter rewrite checkpoint must not win.
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
    const std::uint32_t captured_kv                    = program.sequences[0].text_kv_valid;
    const auto captured_rewrite                        = program.sequences[0].rewrite_checkpoint;
    std::uint64_t entry_id                             = 0;
    require(program.capture_retained_lane(0, &entry_id) && entry_id != 0,
            "retained lane did not capture a RAM checkpoint");
    program.wait_kv_ram_copies();

    program.abort_lane(0);
    require_empty_lane(program, "first abort");
    program.abort_lane(0);
    program.abort_lane(0);
    require_empty_lane(program, "third abort");

    auto host_base = program.plan_request_base(spliced, execution);
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
                program.sequences[0].text_kv_valid == captured_kv &&
                program.sequences[0].rewrite_checkpoint.valid == captured_rewrite.valid &&
                program.sequences[0].rewrite_checkpoint.frontier == captured_rewrite.frontier,
            "RAM restore did not return the captured ledger and checkpoint");

    auto restored_base = program.plan_request_base(spliced, execution);
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
    exercise_decoded_retries(program, frontend, input, execution);
    exercise_full_original_preservation(program, frontend, input, execution);
}

} // namespace

int main(int argc, char** argv) {
    const bool forks_only = argc == 2 && std::string(argv[1]) == "--fork-state";
    if (argc != 1 && !forks_only) { return 1; }
    const char* nvfp4                  = std::getenv("NINFER_QWEN3_6_27B_NVFP4_WEIGHTS");
    const char* group                  = std::getenv("NINFER_QWEN3_6_27B_WEIGHTS");
    const char* dflash                 = std::getenv("NINFER_QWEN3_8_27B_NVFP4_DFLASH_WEIGHTS");
    const char* mtp                    = std::getenv("NINFER_QWEN3_8_27B_NVFP4_MTP_WEIGHTS");
    const char* artifact               = nullptr;
    ninfer::SpeculativeBackend backend = ninfer::SpeculativeBackend::Mtp;
    if (mtp != nullptr && *mtp != '\0') {
        artifact = mtp;
    } else if (nvfp4 != nullptr && *nvfp4 != '\0') {
        artifact = nvfp4;
    } else if (group != nullptr && *group != '\0') {
        artifact = group;
    } else if (dflash != nullptr && *dflash != '\0') {
        artifact = dflash;
        backend  = ninfer::SpeculativeBackend::DFlash;
    } else {
        std::cout << "skip: set NINFER_QWEN3_6_27B_NVFP4_WEIGHTS, "
                     "NINFER_QWEN3_6_27B_WEIGHTS, or "
                     "NINFER_QWEN3_8_27B_NVFP4_DFLASH_WEIGHTS or "
                     "NINFER_QWEN3_8_27B_NVFP4_MTP_WEIGHTS\n";
        return 77;
    }
    try {
        exercise(artifact, backend, forks_only);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
