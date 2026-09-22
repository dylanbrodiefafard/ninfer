#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "runtime/engine/kv_capacity.h"
#include "targets/qwen3_6_27b/impl/variant.h"

#define NINFER_QWEN36_VARIANT ::ninfer::targets::qwen3_6_27b::detail::Variant
#define NINFER_QWEN36_RUNTIME_NS qwen3_6_27b_runtime
#include "targets/qwen3_6/impl/runtime/program.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
namespace target = ninfer::targets::qwen3_6_27b::detail;
namespace family = ninfer::targets::qwen3_6;
namespace execution = family::detail::qwen3_6_27b_runtime;
using Package = ninfer::targets::qwen3_6_27b::Package;
constexpr std::size_t kMiB = 1024ULL * 1024ULL;

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

std::size_t free_bytes() {
    std::size_t free = 0, total = 0;
    CUDA_CHECK(cudaMemGetInfo(&free, &total));
    return free;
}

void exercise(const char* artifact) {
    ninfer::DeviceContext device;
    ninfer::EngineOptions options;
    options.artifact_path = artifact;
    options.max_context = 260000;
    options.max_concurrency = 4;
    options.kv_capacity = ninfer::KvCapacityPolicy::automatic(1024 * kMiB);
    options.prefill_chunk = 4096;
    options.kv_cache = ninfer::KvCacheStorage::Nvfp4;
    options.enable_vision = false;
    options.use_cuda_graph = true;
    options.speculative.backend = ninfer::SpeculativeBackend::DFlash;
    options.speculative.draft_tokens = 5;
    options.speculative.adaptive_draft = true;
    options.speculative.proposal_head = ninfer::ProposalHead::Optimized;

    ninfer::artifact::Reader reader(artifact);
    ninfer::artifact::Binder binder(reader);
    options.model_id = reader.identity().model_id;
    options.weights_id = reader.identity().weights_id;
    options.artifact_file_identity = reader.file_identity();
    const auto profile = Package::resolve_weights(reader.identity(), binder);
    auto load = target::bind_artifact(binder, profile, family::startup_features(options));
    auto materialized = ninfer::artifact::materialize(reader, load.materialization, device);
    target::LoadedModelData model(std::move(load.bindings), std::move(materialized));
    auto frontend = family::make_frontend(model.frontend, false);
    device.synchronize();

    // Use the production candidate builder and resolver at the real post-weight free memory.
    auto planner = Package::make_sequence_planner(device, options, profile);
    const auto curve = planner.capacity_curve();
    const auto resolution = ninfer::runtime::resolve_kv_capacity(
        options.kv_capacity, curve, free_bytes());
    auto plan = std::move(planner).finalize(resolution.main_page_groups);
    require(plan.device_reservation_bytes() == resolution.runtime_reservation_bytes,
            "finalized capacity differs from the solver reservation");
    require(resolution.main_page_groups < curve.maximum_main_page_groups,
            "fixture did not exercise memory-limited automatic capacity");
    require(resolution.planned_slack_bytes >= options.kv_capacity.automatic_headroom_bytes &&
                resolution.planned_slack_bytes < options.kv_capacity.automatic_headroom_bytes +
                                                    curve.bytes_per_additional_main_page_group,
            "automatic capacity did not consume the available page budget");

    // Exercise the real Program directly so timing-dependent policy decisions can be set at
    // round boundaries without a product test flag. All graph capture, prefill, decode, KV,
    // workspace, and commit operations below are the production implementation.
    execution::ProgramImplCore program(model.runtime, *plan.impl_, device, nullptr);
    device.synchronize();
    const auto startup = program.memory_summary();
    const auto startup_free = free_bytes();
    require(startup.cuda_graph_observed_bytes > 0 &&
                startup.cuda_graph_observed_bytes <= startup.cuda_graph_allowance_bytes,
            "DFlash graph allocation exceeded its allowance");
    require(startup.cuda_graph_allowance_bytes - startup.cuda_graph_observed_bytes <= 128 * kMiB,
            "automatic capacity still strands an oversized DFlash graph allowance");
    require(startup_free >= 768 * kMiB, "startup consumed automatic headroom");
    std::cout << "startup tokens=" << resolution.resolved_tokens
              << " graphs=" << startup.cuda_graph_observed_bytes / kMiB << '/'
              << startup.cuda_graph_allowance_bytes / kMiB
              << " MiB free=" << startup_free / kMiB << " MiB\n" << std::flush;

    // Full 4096-token chunks, partial tails, and decode across graph-profile boundaries.
    const std::array<std::size_t, 4> prompt_lengths{4097, 8191, 16383, 260000 - 256};
    ninfer::runtime::ResolvedExecutionOptions request;
    request.requested_output_tokens = 256;
    request.allow_prefix_reuse = false;
    request.sampling.temperature = 1.5F;
    request.sampling.seed = 42;
    request.sampling.p_less = true;
    std::size_t full_chunks = 0;
    for (std::uint32_t lane = 0; lane < 4; ++lane) {
        std::vector<ninfer::TokenId> tokens(prompt_lengths[lane]);
        constexpr std::array<ninfer::TokenId, 8> text{9707, 11, 358, 1079, 264, 2182, 13, 198};
        for (std::size_t i = 0; i < tokens.size(); ++i) { tokens[i] = text[i % text.size()]; }
        auto prompt = family::PreparedPromptAccess::take(frontend.prepare_tokens(std::move(tokens)));
        auto base = program.plan_request_base(prompt, request);
        auto lane_plan = program.plan_request_for_lane(lane, prompt, base);
        require(lane_plan.summary().transient_bytes == 0,
                "text fixture unexpectedly needs request transient storage");
        auto step = program.start_prefill_lane(lane, std::move(prompt), std::move(lane_plan), {});
        std::size_t processed = step.processed_prompt_tokens;
        full_chunks += step.processed_prompt_tokens == 4096;
        while (!step.complete) {
            step = program.advance_prefill_lane(lane);
            processed += step.processed_prompt_tokens;
            full_chunks += step.processed_prompt_tokens == 4096;
            if (processed % 65536 == 0) {
                std::cout << "prefill lane=" << lane << " progress=" << processed << '\n'
                          << std::flush;
            }
        }
        require(processed == prompt_lengths[lane] && step.round.tokens.size() == 1,
                "chunked prefill did not consume the prompt and produce its first token");
        program.resolve_prefill_lane(lane, false);
        std::cout << "prefill lane=" << lane << " tokens=" << processed << '\n' << std::flush;
    }
    require(full_chunks >= 68, "fixture failed to execute full 4096-token prefill chunks");

    const std::array<std::uint32_t, 4> lanes{0, 1, 2, 3};
    const std::array<ninfer::runtime::RoundBudget, 4> budgets{{{128}, {128}, {128}, {128}}};
    const std::array<std::uint8_t, 4> no_flags{};
    std::array<std::uint32_t, 4> accepted{};
    std::array<std::uint32_t, 6> widths_seen{};
    // Every exact B must change K in both directions on the same live state, including 4<->5.
    for (const std::size_t batch : {4U, 3U, 2U, 1U}) {
        for (const std::uint32_t k : {3U, 4U, 5U, 4U, 5U, 3U}) {
            for (std::size_t row = 0; row < batch; ++row) {
                program.requests[lanes[row]].adaptive.live_k = k;
            }
            program.adaptive_batch_k_by_c[batch - 1].live_k = k;
            const auto round = program.decode_batch(std::span(lanes).first(batch),
                                                    std::span(budgets).first(batch));
            for (std::size_t row = 0; row < batch; ++row) {
                require(program.requests[lanes[row]].pending.round_k == k,
                        "decode did not execute the selected adaptive K");
                require(round.row_counts[row] > 0 && round.row_counts[row] <= int(k + 1),
                        "adaptive round returned an invalid token count");
                accepted[row] = static_cast<std::uint32_t>(round.row_counts[row]);
                for (std::size_t col = 0; col < accepted[row]; ++col) {
                    const auto token = round.tokens[row * round.row_stride + col];
                    require(token >= 0 && token < target::TextConfig::token_domain,
                            "adaptive round returned an invalid token");
                }
            }
            program.resolve_pending_batch(std::span(lanes).first(batch),
                                          std::span(accepted).first(batch),
                                          std::span(no_flags).first(batch),
                                          std::span(no_flags).first(batch));
            ++widths_seen[k];
        }
        std::cout << "batch=" << batch << " widths=3,4,5,4,5,3 passed\n" << std::flush;
        // Cancel the retiring lane at the round boundary before shrinking the active batch.
        program.abort_lane(lanes[batch - 1]);
    }
    device.synchronize();
    const auto final = program.memory_summary();
    require(final.workspace_logical_peak_bytes <= final.workspace.capacity_bytes &&
                final.workspace.peak_used_bytes <= final.workspace.capacity_bytes,
            "prefill or adaptive decode exceeded the reserved workspace");
    const auto final_free = free_bytes();
    require(final_free + 64 * kMiB >= startup_free,
            "prefill or adaptive graph switching allocated unexpected device memory");
    for (const auto k : {3U, 4U, 5U}) {
        require(widths_seen[k] == 8, "adaptive width coverage is incomplete");
    }
    std::cout << "ok full_chunks=" << full_chunks
              << " workspace_peak=" << final.workspace.peak_used_bytes / kMiB << '/'
              << final.workspace.capacity_bytes / kMiB
              << " MiB final_free=" << final_free / kMiB << " MiB\n";
}
} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_QWEN3_8_27B_NVFP4_DFLASH_WEIGHTS");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "SKIP: set NINFER_QWEN3_8_27B_NVFP4_DFLASH_WEIGHTS\n";
        return 77;
    }
    try {
        exercise(artifact);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
