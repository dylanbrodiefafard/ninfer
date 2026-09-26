#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "targets/qwen3_6_27b/impl/variant.h"

#define NINFER_QWEN36_VARIANT ::ninfer::targets::qwen3_6_27b::detail::Variant
#define NINFER_QWEN36_RUNTIME_NS qwen3_6_27b_runtime
#include "targets/qwen3_6/impl/runtime/program.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
namespace target = ninfer::targets::qwen3_6_27b::detail;
namespace family = ninfer::targets::qwen3_6;
namespace execution = family::detail::qwen3_6_27b_runtime;
using Package = ninfer::targets::qwen3_6_27b::Package;

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

PrefillRun finish_prefill(execution::ProgramImplCore& program, family::PreparedPromptData prompt,
                          execution::RequestPlan plan) {
    PrefillRun run;
    auto step = program.start_prefill_lane(0, std::move(prompt), std::move(plan), {});
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

void exercise(const char* artifact, ninfer::SpeculativeBackend backend) {
    ninfer::DeviceContext device;
    ninfer::EngineOptions options;
    options.artifact_path              = artifact;
    options.max_context                = 4096;
    options.max_concurrency            = 1;
    options.kv_capacity                = ninfer::KvCapacityPolicy::explicit_capacity(4096);
    options.prefill_chunk              = 1024;
    options.kv_cache                   = ninfer::KvCacheStorage::Nvfp4;
    options.kv_ram_capacity_bytes      = kRamBytes;
    options.enable_vision              = false;
    // Decode-graph capture is measured by its own real tests. This run is the
    // retain/copy/abort/restore sequence, and the graph allowance check is a
    // device-wide free-memory delta that moves when another allocation lands.
    options.use_cuda_graph             = false;
    options.speculative.backend        = backend;
    options.speculative.draft_tokens   = 3;
    options.speculative.proposal_head  = ninfer::ProposalHead::Optimized;
    options.speculative.adaptive_draft = true;

    ninfer::artifact::Reader reader(artifact);
    ninfer::artifact::Binder binder(reader);
    options.model_id                = reader.identity().model_id;
    options.weights_id              = reader.identity().weights_id;
    options.artifact_file_identity  = reader.file_identity();
    const auto profile = Package::resolve_weights(reader.identity(), binder);
    auto load = target::bind_artifact(binder, profile, family::startup_features(options));
    auto materialized = ninfer::artifact::materialize(reader, load.materialization, device);
    target::LoadedModelData model(std::move(load.bindings), std::move(materialized));
    auto frontend = family::make_frontend(model.frontend, false);
    device.synchronize();

    auto planner = Package::make_sequence_planner(device, options, profile);
    const auto pages = planner.capacity_curve().minimum_main_page_groups;
    auto sequence_plan = std::move(planner).finalize(pages);
    execution::ProgramImplCore program(
        model.runtime, *sequence_plan.impl_, device, std::make_unique<ninfer::HostPinnedArena>(kRamBytes));
    device.synchronize();

    ninfer::PromptInput input;
    ninfer::ChatMessage message;
    message.role = ninfer::ChatRole::User;
    message.parts.push_back(ninfer::MessagePart{
        .kind = ninfer::MessagePartKind::Text, .text = "Say hello in one sentence.", .media = {}});
    input.messages.push_back(std::move(message));
    input.options.enable_thinking = true;

    ninfer::runtime::ResolvedExecutionOptions execution;
    execution.requested_output_tokens       = 8;
    execution.sampling.temperature          = 0.0F;
    execution.allow_prefix_reuse            = true;
    execution.force_cold_prefill            = false;
    execution.capture_context_checkpoint    = false;

    auto prepared = family::PreparedPromptAccess::take(frontend.prepare(input));
    const std::vector<ninfer::TokenId> prompt_ids = prepared.token_ids;
    const auto prompt_tokens = static_cast<std::uint32_t>(prompt_ids.size());
    require(prompt_tokens > 1, "thinking prompt was empty");

    auto base = program.plan_request_base(prepared, execution);
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
    require(copied == prompt_ids, "copied prefix included generated tokens or dropped prompt tokens");
    require(copied_frontier == program.sequences[0].rewrite_checkpoint.frontier,
            "copied rewrite frontier does not match the resident checkpoint");

    const auto recovery = family::GenerationRecoveryContext::analyze(input);
    const auto insert   = recovery->recovery_insert({}, 1);
    auto spliced_prompt = frontend.splice_recovery_prompt(prompt_ids, input, insert, recovery);
    require(spliced_prompt.has_value(), "live thinking prompt refused the recovery splice");
    auto spliced = family::PreparedPromptAccess::take(std::move(*spliced_prompt));
    const std::vector<ninfer::TokenId> spliced_ids = spliced.token_ids;
    const auto spliced_tokens = static_cast<std::uint32_t>(spliced_ids.size());
    require(spliced_tokens > prompt_tokens && prefix_equals(spliced_ids, prompt_ids),
            "splice did not append to the copied prompt");

    auto retry_base = program.plan_request_base(spliced, execution);
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
    program.consume_ram_entry(entry_id);
    std::cout << "restored hit reused=" << restored.summary.reused_prompt_tokens
              << " suffix=" << restored.processed << '\n';
}

} // namespace

int main() {
    const char* nvfp4  = std::getenv("NINFER_QWEN3_6_27B_NVFP4_WEIGHTS");
    const char* group  = std::getenv("NINFER_QWEN3_6_27B_WEIGHTS");
    const char* dflash = std::getenv("NINFER_QWEN3_8_27B_NVFP4_DFLASH_WEIGHTS");
    const char* artifact = nullptr;
    ninfer::SpeculativeBackend backend = ninfer::SpeculativeBackend::Mtp;
    if (nvfp4 != nullptr && *nvfp4 != '\0') {
        artifact = nvfp4;
    } else if (group != nullptr && *group != '\0') {
        artifact = group;
    } else if (dflash != nullptr && *dflash != '\0') {
        artifact = dflash;
        backend  = ninfer::SpeculativeBackend::DFlash;
    } else {
        std::cout << "skip: set NINFER_QWEN3_6_27B_NVFP4_WEIGHTS, "
                     "NINFER_QWEN3_6_27B_WEIGHTS, or "
                     "NINFER_QWEN3_8_27B_NVFP4_DFLASH_WEIGHTS\n";
        return 77;
    }
    try {
        exercise(artifact, backend);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
