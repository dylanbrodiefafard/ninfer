#include "targets/qwen3_6/impl/runtime/dflash_candidate_stats.h"
#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/program.h"
#include "runtime/contract/typical_cycle.h"

#include <functional>

#include "targets/qwen3_6/impl/runtime/schedule.h"
#include "ninfer/ops/gdn_replay.h"
#include "ninfer/ops/gdn_history.h"
#include "ninfer/ops/gqa_attention.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/nll_from_logits.h"
#include "targets/qwen3_6/impl/runtime/score_index.h"
#include "ninfer/ops/prepare_ragged_prefix.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/speculative_round.h"

#include "core/arena.h"
#include "core/device.h"
#include "core/dtype.h"
#include "core/host_copy.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

// MemAvailable from /proc/meminfo, or nullopt when the kernel does not report it.
std::optional<std::size_t> available_host_memory_bytes() {
    std::ifstream meminfo("/proc/meminfo");
    std::string key;
    std::size_t kib = 0;
    std::string unit;
    while (meminfo >> key >> kib >> unit) {
        if (key == "MemAvailable:") { return kib * 1024U; }
    }
    return std::nullopt;
}

// Pinned, prefaulted checkpoint images cannot be reclaimed or swapped, so startup leaves this
// much host memory available after carving them instead of courting the OOM killer.
inline constexpr std::size_t kCheckpointSlabHostHeadroomBytes = std::size_t{4} << 30;

using Clock = std::chrono::steady_clock;

// Waits for every Program stream before a failure handler releases lane-owned KV pages and state.
// The handler is already reporting a failure, by rethrowing it or returning its documented
// failure result. A synchronization failure here is secondary to that failure and must not
// replace it.
void synchronize_for_failure_cleanup(const DeviceContext& device) noexcept {
    try {
        device.synchronize_all();
    } catch (...) { // NOLINT(bugprone-empty-catch): the handler's own failure is being reported.
    }
}

// The host view of a frame's logprob records, or empty spans when no row of the round asked.
template <std::size_t Slots>
runtime::RoundLogprobs round_logprob_spans(const qwen3_6::RoundLogprobRecords<Slots>& records,
                                           bool requested) {
    if (!requested) { return {}; }
    return runtime::RoundLogprobs{.token_logprobs = records.token_logprobs,
                                  .top_ids        = records.top_ids,
                                  .top_logprobs   = records.top_logprobs};
}

// Wall time of in-flight compute after host ingress is already filled. Graph
// select, host packing, and KV materialize stay outside decode.ms so tok_s
// matches the GPU round the engine log times, not the CPU setup around it.
double synchronize_round_seconds(DeviceContext& device, Clock::time_point started) {
    device.synchronize();
    return std::chrono::duration<double>(Clock::now() - started).count();
}

std::int32_t checked_i32(std::uint32_t value, const char* label) {
    if (value > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error(label);
    }
    return static_cast<std::int32_t>(value);
}

void rollback_speculative_stats(RequestControl& request, const PendingCandidate& pending) {
    SpeculativeStats& stats = request.speculative_stats;
    if (pending.drafted == 0) {
        if (stats.fallback_steps == 0) {
            throw std::logic_error("rejected speculative fallback stats underflow");
        }
        --stats.fallback_steps;
    } else {
        const std::uint32_t accepted = pending.produced - 1U;
        if (stats.rounds == 0 || stats.drafted_tokens < pending.drafted ||
            stats.accepted_tokens < accepted || accepted > stats.accepted_per_position.size() ||
            pending.round_k >= stats.rounds_per_draft.size() ||
            stats.rounds_per_draft[pending.round_k] == 0) {
            throw std::logic_error("rejected speculative stats do not match pending round");
        }
        --stats.rounds;
        stats.drafted_tokens -= pending.drafted;
        stats.accepted_tokens -= accepted;
        for (std::uint32_t index = 0; index < accepted; ++index) {
            if (stats.accepted_per_position[index] == 0) {
                throw std::logic_error("rejected speculative position stats underflow");
            }
            --stats.accepted_per_position[index];
        }
        --stats.rounds_per_draft[pending.round_k];
    }
    request.adaptive        = pending.adaptive_before;
    stats.live_draft_tokens = request.adaptive.live_k;
}

void arm_typical_exclude(RequestControl& request, const SequenceState& sequence) {
    request.sampling_host.typical_exclude = -1;
    if (!request.typical_cycle_reasoning) { return; }
    const ops::SamplingConfig& cfg = request.sampling_host;
    if (cfg.p_less == 0 || !(cfg.temperature > 0.0f)) { return; }
    if (sequence.ledger.size() < request.prompt_tokens) { return; }
    const std::span<const TokenId> generated(sequence.ledger.data() + request.prompt_tokens,
                                             sequence.ledger.size() - request.prompt_tokens);
    request.sampling_host.typical_exclude = ninfer::runtime::typical_exclude_token(generated);
    if (request.output &&
        !request.output->reasoning_cycle_exclusion_allowed(request.sampling_host.typical_exclude)) {
        request.sampling_host.typical_exclude = -1;
    }
}

void trim_speculative_stats_to_commit(RequestControl& request, const PendingCandidate& pending,
                                      std::uint32_t committed) {
    if (committed == pending.produced || pending.drafted == 0) { return; }
    SpeculativeStats& stats              = request.speculative_stats;
    const std::uint32_t counted_accepted = pending.produced - 1U;
    // Licensed tokens are A drafts plus one correction/bonus. A committed prefix
    // shorter than `produced` has not reached that last token, so all `committed`
    // tokens are drafts.
    const std::uint32_t keep_accepted = committed;
    if (keep_accepted >= counted_accepted) { return; }
    const std::uint32_t drop = counted_accepted - keep_accepted;
    if (stats.accepted_tokens < drop || counted_accepted > stats.accepted_per_position.size()) {
        throw std::logic_error("partial speculative commit stats do not match pending round");
    }
    stats.accepted_tokens -= drop;
    for (std::uint32_t index = keep_accepted; index < counted_accepted; ++index) {
        if (stats.accepted_per_position[index] == 0) {
            throw std::logic_error("partial speculative position stats underflow");
        }
        --stats.accepted_per_position[index];
    }
}

// The counts are read and written on the compute stream so the rollback is ordered against the
// speculative tail that produced them and the next round that consumes them. A pageable
// cudaMemcpyAsync H2D copies the source before returning, so the stack value is safe.
void rollback_sampling_counts(const DeviceContext& device, const ops::SamplingConfig& sampling,
                              std::span<const TokenId> tokens) {
    if (sampling.temperature <= 0.0F || sampling.token_counts == nullptr) { return; }
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        if (std::find(tokens.begin(), tokens.begin() + static_cast<std::ptrdiff_t>(index),
                      tokens[index]) != tokens.begin() + static_cast<std::ptrdiff_t>(index)) {
            continue;
        }
        const auto occurrences = static_cast<std::int32_t>(std::count(
            tokens.begin() + static_cast<std::ptrdiff_t>(index), tokens.end(), tokens[index]));
        std::int32_t count     = 0;
        CUDA_CHECK(cudaMemcpyAsync(&count, sampling.token_counts + tokens[index], sizeof(count),
                                   cudaMemcpyDeviceToHost, device.stream));
        device.synchronize();
        if (count < occurrences) {
            throw std::logic_error("rejected sampling token count underflow");
        }
        count -= occurrences;
        CUDA_CHECK(cudaMemcpyAsync(sampling.token_counts + tokens[index], &count, sizeof(count),
                                   cudaMemcpyHostToDevice, device.stream));
    }
}

std::array<std::int32_t, 3> prompt_rope_position(const PreparedPromptData& prompt,
                                                 std::uint32_t token) {
    const std::size_t tokens = prompt.token_ids.size();
    if (token >= tokens || prompt.positions.size() != 3 * tokens) {
        throw std::invalid_argument("MTP bridge position is outside prepared prompt metadata");
    }
    return {prompt.positions[token], prompt.positions[tokens + token],
            prompt.positions[2 * tokens + token]};
}

schedule::MtpGqaEnvelopes mtp_gqa_envelopes(std::uint32_t max_frontier, std::uint32_t k,
                                            std::uint32_t capacity) {
    const auto visible = [capacity](std::uint64_t value) {
        return static_cast<std::uint32_t>(std::min<std::uint64_t>(capacity, value));
    };
    schedule::MtpGqaEnvelopes out;
    out.target_verify = {1, visible(static_cast<std::uint64_t>(max_frontier) + k + 1ULL)};
    out.batch         = out.target_verify;
    for (std::uint32_t step = 0; step + 1 < k; ++step) {
        out.ar[step] = {1, visible(static_cast<std::uint64_t>(max_frontier) + k + step + 2ULL)};
    }
    return out;
}

// The context append commits the previous round's accepted prefix, which an adaptive switch from
// a longer k leaves wider than the current W(k); every round therefore admits up to W_ceil.
schedule::DFlashEnvelopes dflash_envelopes(std::uint32_t min_frontier, std::uint32_t max_frontier,
                                           std::uint32_t verify_width_ceiling) {
    (void)min_frontier;
    return schedule::DFlashEnvelopes{
        .local  = {0, max_frontier},
        .full   = {0, max_frontier},
        .append = {0, verify_width_ceiling},
    };
}

DecodeGraphProfile& select_graph_profile(DecodeGraphFamily& family, std::uint32_t batch_size,
                                         std::uint32_t frontier, const char* label,
                                         std::uint32_t draft_k = 0, bool grammar_exchange = false,
                                         std::uint32_t verify_width = 0) {
    const auto it = std::find_if(
        family.profiles.begin(), family.profiles.end(), [&](const DecodeGraphProfile& profile) {
            return profile.batch_size == batch_size && profile.min_execution_frontier <= frontier &&
                   frontier <= profile.max_execution_frontier && profile.draft_k == draft_k &&
                   profile.grammar_exchange == grammar_exchange &&
                   (verify_width == 0 || profile.verify_width == verify_width);
        });
    if (it == family.profiles.end()) {
        throw std::logic_error(std::string(label) + " CUDA Graph coverage is incomplete");
    }
    return *it;
}

void validate_graph_profiles(const std::vector<GraphExecutionProfile>& profiles,
                             std::uint32_t max_frontier, const char* label) {
    if (profiles.empty() || profiles.front().min != 0 || profiles.back().max != max_frontier) {
        throw std::logic_error(std::string(label) + " CUDA Graph coverage has invalid endpoints");
    }
    for (std::size_t i = 0; i < profiles.size(); ++i) {
        if (profiles[i].min > profiles[i].max ||
            (i != 0 && profiles[i].min != profiles[i - 1].max + 1)) {
            throw std::logic_error(std::string(label) + " CUDA Graph coverage has a gap");
        }
    }
}

DecodeGraphTopology& select_graph_topology(DecodeGraphFamily& family, std::uint32_t topology_class,
                                           const char* label) {
    const auto it = std::find_if(family.topologies.begin(), family.topologies.end(),
                                 [topology_class](const DecodeGraphTopology& topology) {
                                     return topology.topology_class == topology_class;
                                 });
    if (it == family.topologies.end()) {
        throw std::logic_error(std::string(label) + " CUDA Graph topology is unavailable");
    }
    return *it;
}

void update_graph_profile(DecodeGraphFamily& family, DecodeGraphTopology& topology,
                          std::size_t profile_index, const char* label) {
    const auto describe = [](const DecodeGraphProfile& profile) {
        return "B=" + std::to_string(profile.batch_size) + " k=" + std::to_string(profile.draft_k) +
               " frontier=" + std::to_string(profile.min_execution_frontier) + ".." +
               std::to_string(profile.max_execution_frontier) +
               " topology=" + std::to_string(profile.topology_class);
    };
    try {
        topology.executable.update(family.profiles[profile_index].definition);
    } catch (const std::exception& error) {
        throw std::runtime_error(
            std::string(label) + " graph profile {" +
            (topology.installed_profile ? describe(family.profiles[*topology.installed_profile])
                                        : "unknown") +
            "} -> {" + describe(family.profiles[profile_index]) + "}: " + error.what());
    }
    topology.installed_profile = profile_index;
}

DecodeGraphExecutable& install_graph_profile(DecodeGraphFamily& family, DecodeGraphProfile& profile,
                                             const char* label) {
    DecodeGraphTopology& topology   = select_graph_topology(family, profile.topology_class, label);
    const std::size_t profile_index = static_cast<std::size_t>(&profile - family.profiles.data());
    if (topology.installed_profile != profile_index) {
        update_graph_profile(family, topology, profile_index, label);
    }
    return topology.executable;
}

template <class Prepare>
void instantiate_graph_family(DecodeGraphFamily& family, const char* label, DeviceContext& device,
                              Prepare&& prepare) {
    if (family.profiles.empty()) {
        throw std::logic_error(std::string(label) + " CUDA Graph family has no profiles");
    }

    for (std::size_t i = 0; i < family.profiles.size(); ++i) {
        DecodeGraphProfile& profile = family.profiles[i];
        if (!profile.definition.ready()) {
            throw std::logic_error(std::string(label) + " CUDA Graph definition is empty");
        }
        const auto existing =
            std::find_if(family.topologies.begin(), family.topologies.end(),
                         [&](const DecodeGraphTopology& topology) {
                             return topology.topology_class == profile.topology_class;
                         });
        if (existing != family.topologies.end()) { continue; }

        family.topologies.emplace_back();
        DecodeGraphTopology& topology = family.topologies.back();
        topology.topology_class       = profile.topology_class;
        topology.executable.instantiate(profile.definition);
        topology.installed_profile = i;
    }

    const auto install_and_upload = [&](DecodeGraphTopology& topology, std::size_t profile_index) {
        if (topology.installed_profile != profile_index) {
            update_graph_profile(family, topology, profile_index, label);
        }
        topology.executable.upload(device.stream);
        device.synchronize();
    };

    for (DecodeGraphTopology& topology : family.topologies) {
        std::optional<std::size_t> first_profile;
        for (std::size_t i = 0; i < family.profiles.size(); ++i) {
            if (family.profiles[i].topology_class == topology.topology_class) {
                if (!first_profile) {
                    first_profile = i;
                    install_and_upload(topology, i);

                    DecodeGraphProfile& profile = family.profiles[i];
                    prepare(profile.min_execution_frontier, profile.batch_size, profile.draft_k);
                    device.synchronize();
                    topology.executable.launch(device.stream);
                    device.synchronize();
                    continue;
                }
                install_and_upload(topology, i);
            }
        }
        if (!first_profile) {
            throw std::logic_error(std::string(label) + " CUDA Graph topology has no definitions");
        }
        if (topology.installed_profile != *first_profile) {
            install_and_upload(topology, *first_profile);
        }
    }
}

ProgramImplCore::ProgramImplCore(const LoadedModelData& model_in, const SequencePlanImpl& plan,
                                 DeviceContext& device_in,
                                 std::unique_ptr<HostPinnedArena> kv_ram_arena)
    : model(model_in), device(device_in), capacity(plan.capacity), kv_capacity(plan.kv_capacity),
      max_concurrency(plan.max_concurrency), prefill_chunk(plan.prefill_chunk),
      draft_window(plan.draft_window), dflash_verify_width(plan.dflash_verify_width),
      adaptive_draft(plan.adaptive_draft), p_less_draft_temperature(plan.p_less_draft_temperature),
      captured_ks(plan.captured_ks), dflash_shapes(plan.dflash_shapes), dflash_arms_by_batch([&] {
          std::array<std::vector<std::uint32_t>, kMaximumConcurrency> arms;
          for (std::uint32_t b = 1; b <= kMaximumConcurrency; ++b) {
              arms[b - 1] = dflash_batch_arms(plan.dflash_shapes, b);
          }
          return arms;
      }()),
      speculative_backend(plan.speculative_backend), context_marks(plan.context_checkpoint_marks),
      kv_dtype(plan.kv_dtype), kv_quant_group(plan.kv_quant_group),
      proposal_head(plan.proposal_head), keep_frac(plan.keep_frac), xattn_tau(plan.xattn_tau),
      xattn_min_len(plan.xattn_min_len), vision_enabled(plan.features.vision),
      use_cuda_graph(plan.use_cuda_graph), kv_payload_bytes(plan.persistent.kv_payload_bytes),
      kv_ram_capacity_bytes(plan.kv_ram_capacity_bytes),
      kv_disk_capacity_bytes(plan.kv_disk_capacity_bytes),
      graph_allowance_bytes(plan.graph_allowance_bytes), workspace_plan(plan.workspace),
      persistent(plan.persistent.bytes), workspace_storage(plan.workspace.capacity),
      work(DeviceSpan{workspace_storage.base(), workspace_storage.capacity()}),
      round_host(sizeof(PrefillRoundHost)),
      ordinary_host(
          plan.speculative_backend == SpeculativeBackend::None
              ? std::make_optional<PinnedHostBuffer>(sizeof(qwen3_6::OrdinaryDecodeIngress) +
                                                     sizeof(qwen3_6::OrdinaryDecodeEgress))
              : std::nullopt),
      mtp_host(plan.speculative_backend == SpeculativeBackend::Mtp
                   ? std::make_optional<PinnedHostBuffer>(sizeof(qwen3_6::MtpDecodeIngress) +
                                                          sizeof(qwen3_6::MtpDecodeEgress))
                   : std::nullopt),
      dflash_host(plan.speculative_backend == SpeculativeBackend::DFlash
                      ? std::make_optional<PinnedHostBuffer>(sizeof(qwen3_6::DFlashDecodeIngress) +
                                                             sizeof(qwen3_6::DFlashDecodeEgress))
                      : std::nullopt) {
    if (model.weights_arena == nullptr) {
        throw std::invalid_argument("Qwen3.6 model view has no owning weight arena");
    }
    if (model.features != plan.features || model.mtp.has_value() != plan.features.mtp() ||
        model.dflash.has_value() != plan.features.dflash() ||
        model.optimized_proposal.has_value() != plan.features.optimized_proposal() ||
        model.vision.has_value() != plan.features.vision) {
        throw std::invalid_argument(
            "Qwen3.6 loaded weights do not match the frozen startup features");
    }
    if (model.mtp.has_value() && model.dflash.has_value()) {
        throw std::invalid_argument("MTP and DFlash model views are mutually exclusive");
    }
    const DeviceSpan backing = persistent.alloc_bytes(plan.persistent.bytes, 256);
    decoder = std::make_unique<qwen3_6::DecoderState>(backing, plan.persistent.decoder);
    if (plan.persistent.replay_records) {
        replay_records.emplace(backing, *plan.persistent.replay_records);
    }
    if (plan.persistent.gdn_history) {
        gdn_history.emplace(backing, *plan.persistent.gdn_history);
        ops::gdn_history_reset_all(*gdn_history, device.stream);
    }
    if (replay_records.has_value() != (speculative_backend != SpeculativeBackend::None)) {
        throw std::logic_error("ReplaySSM records do not match the sequence plan");
    }
    if (plan.persistent.dflash) { dflash.emplace(backing, *plan.persistent.dflash); }
    if (dflash.has_value() != plan.features.dflash()) {
        throw std::logic_error("DFlash state does not match the frozen sequence plan");
    }

    io = qwen3_6::RoundState(backing, plan.persistent.round);
    if (io.mtp.has_value() != (speculative_backend == SpeculativeBackend::Mtp)) {
        throw std::logic_error("round-state MTP extension does not match the sequence plan");
    }
    if (io.mtp_decode.has_value() != (speculative_backend == SpeculativeBackend::Mtp)) {
        throw std::logic_error("MTP decode frame does not match the sequence plan");
    }
    if (io.ordinary.has_value() != (speculative_backend == SpeculativeBackend::None)) {
        throw std::logic_error("ordinary decode frame does not match the sequence plan");
    }
    if (io.dflash_prefill.has_value() != (speculative_backend == SpeculativeBackend::DFlash)) {
        throw std::logic_error("DFlash prefill scratch does not match the sequence plan");
    }
    if (io.dflash_decode.has_value() != (speculative_backend == SpeculativeBackend::DFlash)) {
        throw std::logic_error("DFlash decode frame does not match the sequence plan");
    }
    prefill_hidden  = plan.persistent.prefill_hidden.bind(backing);
    token_counts    = plan.persistent.token_counts.bind(backing);
    sampling_config = plan.persistent.sampling_config.bind(backing);
    tool_masks      = std::make_unique<qwen3_6::ToolMaskExchange>(
        plan.persistent.tool_token_masks.bind(backing),
        plan.persistent.tool_sampling_config.bind(backing),
        plan.persistent.tool_nodes.bind(backing));
    tail_hidden_store               = plan.persistent.tail_hidden.bind(backing);
    rewrite_checkpoint_hidden_store = plan.persistent.rewrite_checkpoint_hidden.bind(backing);
    if (plan.persistent.staging_hidden) {
        staging_hidden = plan.persistent.staging_hidden->bind(backing);
    }
    {
        // Every pinned checkpoint image is carved here from one prefaulted, registered slab, so
        // no serving path allocates or frees pinned memory for checkpoints.
        const std::size_t conv_bytes      = decoder->linear_attention.conv_host_image_bytes();
        const std::size_t recurrent_bytes = decoder->linear_attention.recurrent_host_image_bytes();
        const std::size_t dflash_bytes    = dflash ? dflash->local.lane_host_bytes() : 0;
        const std::size_t pool_heads =
            captures_context_checkpoints() && staging_hidden.data != nullptr
                ? qwen3_6::detail::context_checkpoint_image_pool_capacity(
                      max_concurrency, context_marks, capacity, kv_capacity)
                : 0;
        const auto region = [](std::size_t bytes) {
            constexpr std::size_t alignment = qwen3_6::detail::kCheckpointImageAlignment;
            return (bytes + alignment - 1) / alignment * alignment;
        };
        const std::size_t rewrite_image_bytes =
            region(conv_bytes) + region(recurrent_bytes) + region(dflash_bytes);
        const std::size_t head_bytes = rewrite_image_bytes + region(staging_hidden.bytes());
        const std::size_t slab_bytes =
            max_concurrency * rewrite_image_bytes + pool_heads * head_bytes;
        if (const std::optional<std::size_t> available = available_host_memory_bytes();
            available && slab_bytes + kCheckpointSlabHostHeadroomBytes > *available) {
            throw std::runtime_error(
                "checkpoint images need " + std::to_string(slab_bytes >> 20) +
                " MiB of pinned host memory (" + std::to_string(pool_heads) +
                " context-checkpoint heads), but only " + std::to_string(*available >> 20) +
                " MiB is available and 4096 MiB must stay free; lower --max-concurrency or "
                "--max-context, or use fewer --context-checkpoints marks");
        }
        checkpoint_image_slab_.emplace(slab_bytes);
        context_checkpoint_pool_.reserve(pool_heads);
        for (std::size_t i = 0; i < pool_heads; ++i) {
            ContextCheckpointHead head;
            head.prepare_copy_event();
            head.conv      = carve_checkpoint_image(conv_bytes);
            head.recurrent = carve_checkpoint_image(recurrent_bytes);
            head.hidden    = carve_checkpoint_image(staging_hidden.bytes());
            head.dflash    = carve_checkpoint_image(dflash_bytes);
            context_checkpoint_pool_.push_back(std::move(head));
        }
    }
    for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
        SequenceState& sequence = sequences[lane];
        sequence.lane           = lane;
        sequence.tail_hidden    = tail_hidden_store.slice(1, static_cast<std::int32_t>(lane), 1);
        sequence.rewrite_checkpoint_hidden =
            rewrite_checkpoint_hidden_store.slice(1, static_cast<std::int32_t>(lane), 1);
        sequence.ledger.reserve(static_cast<std::size_t>(capacity) + 1ULL);
        sequence.prefix_identity.reserve(static_cast<std::size_t>(capacity) + 1ULL);
        sequence.prefix_hashes.reserve(static_cast<std::size_t>(capacity) + 1ULL);
        sequence.next_context_mark = qwen3_6::detail::first_prefill_context_mark(context_marks);
        allocate_rewrite_image(sequence);
    }

    set_device_i32(io.text_kv_table_row, 0);
    set_device_i32(io.backend_kv_table_row, 0);

    static_assert(std::is_standard_layout_v<PrefillRoundHost>);
    auto* prefill_round_host = static_cast<PrefillRoundHost*>(round_host.data());
    *prefill_round_host      = {};
    host_tokens              = &prefill_round_host->token;
    host_prefill_logprobs    = &prefill_round_host->logprobs;
    if (ordinary_host) {
        ordinary_host_ingress = static_cast<qwen3_6::OrdinaryDecodeIngress*>(ordinary_host->data());
        ordinary_host_egress  = reinterpret_cast<qwen3_6::OrdinaryDecodeEgress*>(
            static_cast<unsigned char*>(ordinary_host->data()) +
            sizeof(qwen3_6::OrdinaryDecodeIngress));
        *ordinary_host_ingress = {};
        *ordinary_host_egress  = {};
    }
    if (mtp_host) {
        mtp_host_ingress = static_cast<qwen3_6::MtpDecodeIngress*>(mtp_host->data());
        mtp_host_egress  = reinterpret_cast<qwen3_6::MtpDecodeEgress*>(
            static_cast<unsigned char*>(mtp_host->data()) + sizeof(qwen3_6::MtpDecodeIngress));
        *mtp_host_ingress = {};
        *mtp_host_egress  = {};
    }
    if (dflash_host) {
        dflash_host_ingress = static_cast<qwen3_6::DFlashDecodeIngress*>(dflash_host->data());
        dflash_host_egress  = reinterpret_cast<qwen3_6::DFlashDecodeEgress*>(
            static_cast<unsigned char*>(dflash_host->data()) +
            sizeof(qwen3_6::DFlashDecodeIngress));
        *dflash_host_ingress = {};
        *dflash_host_egress  = {};
    }
    if (io.dflash_prefill) {
        CUDA_CHECK(cudaMemsetAsync(io.dflash_prefill->produced_count.data, 0,
                                   io.dflash_prefill->produced_count.bytes(), device.stream));
    }
    CUDA_CHECK(cudaMemsetAsync(io.rope_delta.data, 0, io.rope_delta.bytes(), device.stream));
    if (io.mtp) {
        CUDA_CHECK(
            cudaMemsetAsync(io.mtp->position.data, 0, io.mtp->position.bytes(), device.stream));
    }
    CUDA_CHECK(cudaMemsetAsync(token_counts.data, 0, token_counts.bytes(), device.stream));
    CUDA_CHECK(cudaMemsetAsync(sampling_config.data, 0, sampling_config.bytes(), device.stream));
    device.synchronize();
    prepare_graphs();
    work.reset();
    work.reset_peak();
    workspace_logical_peak_bytes = 0;
    if (kv_ram_capacity_bytes == 0) {
        if (kv_ram_arena != nullptr) {
            throw std::invalid_argument("disabled KV RAM cache received a host arena");
        }
    } else {
        if (kv_ram_arena == nullptr || kv_ram_arena->capacity() != kv_ram_capacity_bytes) {
            throw std::invalid_argument("KV RAM cache host arena does not match the sequence plan");
        }
        kv_ram_cache_.emplace(std::move(*kv_ram_arena));
    }
    if (kv_disk_capacity_bytes != 0) {
        const PagedKVPool& text_pool         = decoder->text_kv.pool();
        const qwen3_6::PagedKVCache* backend = backend_kv_cache();
        const PagedKVPool* backend_pool      = backend != nullptr ? &backend->pool() : nullptr;
        std::size_t page_bytes               = paged_kv_logical_page_bytes(text_pool);
        if (backend_pool != nullptr) {
            page_bytes = std::max(page_bytes, paged_kv_logical_page_bytes(*backend_pool));
        }
        const KvCacheStorage kv_cache = kv_dtype == DType::BF16 ? KvCacheStorage::BFloat16
                                        : kv_dtype == DType::U8 ? KvCacheStorage::Nvfp4
                                                                : KvCacheStorage::Int8Group64;
        qwen3_6::detail::DiskOpenConfig disk;
        disk.location       = plan.kv_disk_location;
        disk.capacity_bytes = kv_disk_capacity_bytes;
        disk.compress       = plan.kv_disk_compress;
        disk.max_context    = capacity;
        disk.ram            = kv_ram_cache_ ? &*kv_ram_cache_ : nullptr;
        disk.fingerprint    = qwen3_6::detail::make_disk_fingerprint(
            plan.model_id, plan.weights_id, plan.artifact_file_identity, kv_cache,
            speculative_backend, text_pool, backend_pool, &decoder->linear_attention,
            dflash ? &dflash->local : nullptr);
        disk.text_pool          = &text_pool;
        disk.backend_pool       = backend_pool;
        disk.logical_page_bytes = page_bytes;
        disk.hidden_bytes       = sequences[0].tail_hidden.bytes();
        disk.restore_io_threads = 16;
        kv_disk_cache_.emplace(std::move(disk));
    }
    // Rewrite captures always pass through staging, so its events exist from startup.
    if (staging_hidden.data != nullptr) {
        qwen3_6::detail::create_cache_cuda_event(&staging_.d2d_done, cudaEventDisableTiming);
        try {
            qwen3_6::detail::create_cache_cuda_event(&staging_.copies_done, cudaEventDisableTiming);
        } catch (...) {
            (void)cudaEventDestroy(staging_.d2d_done);
            staging_.d2d_done = nullptr;
            throw;
        }
    }
    try {
        qwen3_6::detail::create_cache_cuda_event(&copy_hold_cancel_fence_, cudaEventDisableTiming);
    } catch (...) {
        if (staging_.copies_done != nullptr) {
            (void)cudaEventDestroy(staging_.copies_done);
            staging_.copies_done = nullptr;
        }
        if (staging_.d2d_done != nullptr) {
            (void)cudaEventDestroy(staging_.d2d_done);
            staging_.d2d_done = nullptr;
        }
        throw;
    }
}

ProgramImplCore::~ProgramImplCore() noexcept {
    // A round can fail after forking the host callback but before its compute
    // join. Drain before destroying requests (the callback borrows sessions).
    if (device.stream != nullptr) { (void)cudaStreamSynchronize(device.stream); }
    if (device.host_stream != nullptr) { (void)cudaStreamSynchronize(device.host_stream); }
    try {
        shutdown_kv_tiers();
    } catch (...) { // NOLINT(bugprone-empty-catch): a destructor cannot report shutdown failure.
    }
    fence_staging_copies();
    if (staging_.d2d_done != nullptr) {
        (void)cudaEventDestroy(staging_.d2d_done);
        staging_.d2d_done = nullptr;
    }
    if (staging_.copies_done != nullptr) {
        (void)cudaEventDestroy(staging_.copies_done);
        staging_.copies_done = nullptr;
    }
    if (copy_hold_cancel_fence_ != nullptr) {
        (void)cudaEventSynchronize(copy_hold_cancel_fence_);
        (void)cudaEventDestroy(copy_hold_cancel_fence_);
        copy_hold_cancel_fence_ = nullptr;
    }
    for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
        clear_context_checkpoints(sequences[lane]);
    }
    if (device.stream != nullptr) { (void)cudaStreamSynchronize(device.stream); }
    if (device.copy_stream != nullptr) { (void)cudaStreamSynchronize(device.copy_stream); }
}

bool ProgramImplCore::can_admit_lane(std::uint32_t lane, const RequestPlan& plan) const noexcept {
    if (lane >= max_concurrency || plan.impl_ == nullptr) { return false; }
    const RequestControl& request = requests[lane];
    if (request.lifecycle == Lifecycle::Prefilling || request.lifecycle == Lifecycle::Active ||
        request.lifecycle == Lifecycle::Pending) {
        return false;
    }
    const SequenceState& sequence = sequences[lane];
    const auto can_replace        = [](const PagedKVPool& pool, std::uint32_t old_pages,
                                       std::uint32_t new_pages) {
        return old_pages <= pool.entitled_pages() && new_pages <= pool.logical_page_capacity() &&
               new_pages <= pool.page_group_count() - (pool.entitled_pages() - old_pages);
    };
    const std::uint32_t old_text = sequence.kv ? sequence.kv->text.page_entitlement() : 0;
    if (!can_replace(decoder->text_kv.pool(), old_text, plan.impl_->text_kv_page_entitlement)) {
        return false;
    }
    const qwen3_6::PagedKVCache* backend = backend_kv_cache();
    if (backend == nullptr) { return plan.impl_->backend_kv_page_entitlement == 0; }
    const std::uint32_t old_backend =
        sequence.kv && sequence.kv->backend ? sequence.kv->backend->page_entitlement() : 0;
    return can_replace(backend->pool(), old_backend, plan.impl_->backend_kv_page_entitlement);
}

bool ProgramImplCore::can_admit_lane_after_retained_eviction(
    std::uint32_t lane, const RequestPlan& plan) const noexcept {
    if (lane >= max_concurrency || plan.impl_ == nullptr) { return false; }
    const RequestControl& request = requests[lane];
    if (request.lifecycle == Lifecycle::Prefilling || request.lifecycle == Lifecycle::Active ||
        request.lifecycle == Lifecycle::Pending) {
        return false;
    }

    std::uint32_t reclaimable_text    = 0;
    std::uint32_t reclaimable_backend = 0;
    for (std::uint32_t other = 0; other < max_concurrency; ++other) {
        const SequenceState& candidate = sequences[other];
        if (other == lane || !candidate.retained || !candidate.kv) { continue; }
        const auto& candidate_kv = *candidate.kv;
        reclaimable_text += candidate_kv.text.page_entitlement();
        if (candidate_kv.backend) {
            reclaimable_backend += candidate_kv.backend->page_entitlement();
        }
    }

    const auto can_replace = [](const PagedKVPool& pool, std::uint32_t old_pages,
                                std::uint32_t reclaimable_pages, std::uint32_t new_pages) {
        if (old_pages > pool.entitled_pages() ||
            reclaimable_pages > pool.entitled_pages() - old_pages ||
            new_pages > pool.logical_page_capacity()) {
            return false;
        }
        const std::uint32_t committed = pool.entitled_pages() - old_pages - reclaimable_pages;
        return new_pages <= pool.page_group_count() - committed;
    };

    const SequenceState& sequence = sequences[lane];
    const std::uint32_t old_text  = sequence.kv ? sequence.kv->text.page_entitlement() : 0;
    if (!can_replace(decoder->text_kv.pool(), old_text, reclaimable_text,
                     plan.impl_->text_kv_page_entitlement)) {
        return false;
    }

    const qwen3_6::PagedKVCache* backend = backend_kv_cache();
    if (backend == nullptr) { return plan.impl_->backend_kv_page_entitlement == 0; }
    const std::uint32_t old_backend =
        sequence.kv && sequence.kv->backend ? sequence.kv->backend->page_entitlement() : 0;
    return can_replace(backend->pool(), old_backend, reclaimable_backend,
                       plan.impl_->backend_kv_page_entitlement);
}

bool ProgramImplCore::can_admit_lane_after_releasing(
    std::uint32_t lane, const RequestPlan& plan,
    std::span<const std::uint32_t> release_lanes) const noexcept {
    if (lane >= max_concurrency || plan.impl_ == nullptr) { return false; }
    const RequestControl& request = requests[lane];
    if (request.lifecycle == Lifecycle::Prefilling || request.lifecycle == Lifecycle::Active ||
        request.lifecycle == Lifecycle::Pending) {
        return false;
    }

    std::uint32_t reclaimable_text    = 0;
    std::uint32_t reclaimable_backend = 0;
    for (const std::uint32_t other : release_lanes) {
        if (other == lane || other >= max_concurrency) { continue; }
        const SequenceState& candidate = sequences[other];
        if (!candidate.retained || !candidate.kv) { continue; }
        const auto& candidate_kv = *candidate.kv;
        reclaimable_text += candidate_kv.text.page_entitlement();
        if (candidate_kv.backend) {
            reclaimable_backend += candidate_kv.backend->page_entitlement();
        }
    }

    const auto can_replace = [](const PagedKVPool& pool, std::uint32_t old_pages,
                                std::uint32_t reclaimable_pages, std::uint32_t new_pages) {
        if (old_pages > pool.entitled_pages() ||
            reclaimable_pages > pool.entitled_pages() - old_pages ||
            new_pages > pool.logical_page_capacity()) {
            return false;
        }
        const std::uint32_t committed = pool.entitled_pages() - old_pages - reclaimable_pages;
        return new_pages <= pool.page_group_count() - committed;
    };

    const SequenceState& sequence = sequences[lane];
    const std::uint32_t old_text  = sequence.kv ? sequence.kv->text.page_entitlement() : 0;
    if (!can_replace(decoder->text_kv.pool(), old_text, reclaimable_text,
                     plan.impl_->text_kv_page_entitlement)) {
        return false;
    }

    const qwen3_6::PagedKVCache* backend = backend_kv_cache();
    if (backend == nullptr) { return plan.impl_->backend_kv_page_entitlement == 0; }
    const std::uint32_t old_backend =
        sequence.kv && sequence.kv->backend ? sequence.kv->backend->page_entitlement() : 0;
    return can_replace(backend->pool(), old_backend, reclaimable_backend,
                       plan.impl_->backend_kv_page_entitlement);
}

runtime::AdmissionResources ProgramImplCore::admission_capacity() const noexcept {
    const qwen3_6::PagedKVCache* backend = backend_kv_cache();
    return runtime::AdmissionResources{
        .active_lanes     = max_concurrency,
        .main_kv_pages    = decoder->text_kv.pool().page_group_count(),
        .backend_kv_pages = backend != nullptr ? backend->pool().page_group_count() : 0U,
    };
}

runtime::PrefillStepResult
ProgramImplCore::start_prefill_lane(std::uint32_t lane, PreparedPromptData&& prompt,
                                    RequestPlan&& plan, runtime::TransientRegion transient,
                                    const qwen3_6::OutputSession* output) {
    if (lane >= max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    SequenceState& sequence                    = sequences[lane];
    RequestControl& request                    = requests[lane];
    request.captured_context_checkpoint_tokens = 0;
    request.restored_context_checkpoint_tokens = 0;
    const RequestPlan consumed                 = std::move(plan);
    if (consumed.impl_ == nullptr) { throw std::invalid_argument("request plan is empty"); }
    RequestPlanImpl& request_plan = *consumed.impl_;
    if (request.lifecycle == Lifecycle::Prefilling || request.lifecycle == Lifecycle::Active ||
        request.lifecycle == Lifecycle::Pending) {
        throw std::logic_error("staged prefill requires a free request lane");
    }

    const std::uint32_t prompt_tokens = static_cast<std::uint32_t>(prompt.token_ids.size());
    if (prompt_tokens != request_plan.summary.prompt_tokens ||
        (request_plan.vision.has_value() && !prompt.has_media())) {
        throw std::invalid_argument("request plan does not describe the prepared prompt");
    }
    if (prompt.identity.rewrite_checkpoint &&
        (prompt.identity.rewrite_checkpoint->frontier == 0 ||
         prompt.identity.rewrite_checkpoint->frontier > prompt_tokens)) {
        throw std::invalid_argument("prepared prompt has an invalid rewrite checkpoint");
    }
    const bool suffix_has_visual = std::any_of(
        prompt.token_types.begin() + static_cast<std::ptrdiff_t>(request_plan.reuse_base),
        prompt.token_types.end(), [](std::uint8_t type) { return type != 0; });
    if (suffix_has_visual != request_plan.vision.has_value()) {
        throw std::invalid_argument("request plan does not describe the prompt suffix modality");
    }
    if (request_plan.summary.transient_bytes != 0 &&
        (transient.data == nullptr || transient.size < request_plan.summary.transient_bytes ||
         transient.alignment < request_plan.summary.transient_alignment)) {
        throw std::invalid_argument("request transient region does not satisfy the plan");
    }
    if (request_plan.reuse != ReusePath::FullReset &&
        (!sequence.retained ||
         !qwen3_6::detail::prefix_matches(prompt, sequence.ledger, sequence.prefix_identity,
                                          request_plan.reuse_base))) {
        throw std::logic_error("planned resident prefix is no longer reusable");
    }
    if (is_rewrite_checkpoint_restore(request_plan.reuse) &&
        (!sequence.rewrite_checkpoint.valid ||
         sequence.rewrite_checkpoint.frontier != request_plan.reuse_base ||
         request_plan.reuse != restore_path(sequence.rewrite_checkpoint.kind))) {
        throw std::logic_error("planned rewrite checkpoint is unavailable");
    }
    if (qwen3_6::detail::is_staged_checkpoint_restore(request_plan.reuse)) {
        const auto head =
            std::find_if(sequence.context_checkpoints.begin(), sequence.context_checkpoints.end(),
                         [&](const ContextCheckpointHead& candidate) {
                             return candidate.frontier == request_plan.reuse_base;
                         });
        const bool want_rollback = request_plan.reuse == ReusePath::RestoreTurnRollback;
        if (head == sequence.context_checkpoints.end() ||
            (head->kind == qwen3_6::detail::ContextCheckpointKind::TurnRollback) != want_rollback ||
            !qwen3_6::detail::prefix_matches(prompt, sequence.ledger, sequence.prefix_identity,
                                             request_plan.reuse_base)) {
            throw std::logic_error("planned context checkpoint is unavailable");
        }
    }
    if (request_plan.rewrite_checkpoint_action == RewriteCheckpointAction::KeepExisting &&
        (!prompt.identity.rewrite_checkpoint || !sequence.rewrite_checkpoint.valid ||
         sequence.rewrite_checkpoint.kind != prompt.identity.rewrite_checkpoint->kind ||
         sequence.rewrite_checkpoint.frontier != prompt.identity.rewrite_checkpoint->frontier ||
         request_plan.reuse == ReusePath::FullReset ||
         !qwen3_6::detail::prefix_matches(prompt, sequence.ledger, sequence.prefix_identity,
                                          sequence.rewrite_checkpoint.frontier))) {
        throw std::logic_error("planned rewrite checkpoint retention is unavailable");
    }
    if (request_plan.rewrite_checkpoint_action == RewriteCheckpointAction::ReclassifyExisting &&
        (!prompt.identity.rewrite_checkpoint || !sequence.rewrite_checkpoint.valid ||
         sequence.rewrite_checkpoint.kind == prompt.identity.rewrite_checkpoint->kind ||
         sequence.rewrite_checkpoint.frontier != prompt.identity.rewrite_checkpoint->frontier ||
         request_plan.reuse == ReusePath::FullReset ||
         !qwen3_6::detail::prefix_matches(prompt, sequence.ledger, sequence.prefix_identity,
                                          sequence.rewrite_checkpoint.frontier))) {
        throw std::logic_error("planned rewrite checkpoint reclassification is unavailable");
    }
    if (request_plan.rewrite_checkpoint_action == RewriteCheckpointAction::CaptureNew &&
        (!request_plan.rewrite_checkpoint_capture || !prompt.identity.rewrite_checkpoint ||
         request_plan.rewrite_checkpoint_capture->kind !=
             prompt.identity.rewrite_checkpoint->kind ||
         request_plan.rewrite_checkpoint_capture->frontier !=
             prompt.identity.rewrite_checkpoint->frontier ||
         request_plan.rewrite_checkpoint_capture->frontier <= request_plan.reuse_base ||
         request_plan.rewrite_checkpoint_capture->frontier > prompt_tokens)) {
        throw std::logic_error("planned rewrite checkpoint capture is invalid");
    }
    if (request_plan.rewrite_checkpoint_action == RewriteCheckpointAction::Drop &&
        prompt.identity.rewrite_checkpoint) {
        throw std::logic_error("planned rewrite checkpoint drop does not describe the prompt");
    }
    if (request_plan.rewrite_checkpoint_action == RewriteCheckpointAction::DeferCapture &&
        (!prompt.identity.rewrite_checkpoint || request_plan.reuse == ReusePath::FullReset ||
         prompt.identity.rewrite_checkpoint->frontier > request_plan.reuse_base)) {
        throw std::logic_error("planned rewrite checkpoint deferral is invalid");
    }

    const auto started       = Clock::now();
    const std::uint32_t base = request_plan.reuse_base;
    const std::uint32_t initial_mtp_extent =
        speculative_backend == SpeculativeBackend::Mtp
            ? std::min({draft_window,
                        request_plan.summary.effective_output_tokens > 1
                            ? request_plan.summary.effective_output_tokens - 2
                            : 0U,
                        capacity - prompt_tokens > 0 ? capacity - prompt_tokens - 1 : 0U})
            : 0U;
    request.lifecycle    = Lifecycle::Empty;
    sequence.retained    = false;
    sequence.turn_closed = false;
    sequence.closure_frontier =
        prompt.identity.rewrite_checkpoint &&
                prompt.identity.rewrite_checkpoint->kind == RewriteCheckpointKind::TurnClosure &&
                prompt.identity.rewrite_checkpoint->generation_opener
            ? prompt.identity.rewrite_checkpoint->frontier
            : 0;
    try {
        if (request_plan.reuse == ReusePath::FullReset) {
            sequence.kv.reset();
            clear_context_checkpoints(sequence);
            ordered_reset(sequence);
            sequence.ledger.clear();
            sequence.prefix_hashes.clear();
            sequence.text_kv_valid = 0;
            sequence.mtp_kv_valid  = 0;
            reserve_sequence_kv(sequence, request_plan.text_kv_page_entitlement,
                                request_plan.backend_kv_page_entitlement);
        } else if (request_plan.reuse == ReusePath::AppendAtFrontier) {
            if (!sequence.kv) {
                throw std::logic_error("resident prefix has no KV allocation bundle");
            }
            if (sequence.text_kv_valid < base) {
                throw std::logic_error("resident Text KV is shorter than the append frontier");
            }
            if (speculative_backend == SpeculativeBackend::Mtp) {
                const std::uint32_t mtp_base = base == 0 ? 0 : base - 1;
                if (!request_plan.prepare_mtp || sequence.mtp_kv_valid < mtp_base) {
                    throw std::logic_error("resident MTP KV is shorter than the bridge frontier");
                }
                sequence.mtp_kv_valid = mtp_base;
            } else if (speculative_backend == SpeculativeBackend::DFlash &&
                       sequence.dflash_context_frontier != base) {
                throw std::logic_error("resident DFlash context is not at the append frontier");
            }
            trim_sequence_kv(sequence, base, backend_kv_valid(sequence));
            resize_sequence_kv_entitlement(sequence, request_plan.text_kv_page_entitlement,
                                           request_plan.backend_kv_page_entitlement);
            sequence.text_kv_valid = base;
            sequence.ledger.resize(base);
            sequence.prefix_hashes.truncate(base);
            drop_context_checkpoints_after(sequence, base);
            // A tier entry cut at its turn checkpoint is restored without a rewrite checkpoint at
            // its frontier. So is one whose request drops the checkpoint, but such a request has
            // no closure frontier, so the cut-restore pin skip never applies to it.
            const bool cut_restore = (request_plan.reuse_source == PrefixReuseSource::HostRam ||
                                      request_plan.reuse_source == PrefixReuseSource::HostDisk) &&
                                     !sequence.rewrite_checkpoint.valid;
            maybe_capture_turn_rollback(sequence, request, prompt, request_plan.reuse, base,
                                        prompt_tokens, request_plan.capture_context_checkpoints,
                                        request_plan.capture_context_checkpoint, cut_restore);
        } else if (is_rewrite_checkpoint_restore(request_plan.reuse)) {
            if (!sequence.kv || sequence.text_kv_valid < base) {
                throw std::logic_error("resident rewrite checkpoint has no complete KV allocation");
            }
            sequence.text_kv_valid = base;
            if (speculative_backend == SpeculativeBackend::Mtp) {
                const std::uint32_t mtp_base = base == 0 ? 0 : base - 1;
                if (!request_plan.prepare_mtp || sequence.mtp_kv_valid < mtp_base) {
                    throw std::logic_error(
                        "rewrite-checkpoint MTP KV is shorter than the bridge frontier");
                }
                sequence.mtp_kv_valid = mtp_base;
            } else if (speculative_backend == SpeculativeBackend::DFlash) {
                if (!dflash || sequence.dflash_context_frontier < base) {
                    throw std::logic_error("planned DFlash rewrite checkpoint is unavailable");
                }
                if constexpr (DFlashConfig::full_layers > 0) {
                    if (!sequence.kv->backend) {
                        throw std::logic_error("planned DFlash rewrite checkpoint is unavailable");
                    }
                }
                sequence.dflash_context_frontier = base;
            }
            trim_sequence_kv(sequence, base, backend_kv_valid(sequence));
            resize_sequence_kv_entitlement(sequence, request_plan.text_kv_page_entitlement,
                                           request_plan.backend_kv_page_entitlement);
            restore_rewrite_checkpoint_state(sequence);
            if (base == prompt_tokens) { copy_tail(sequence, sequence.rewrite_checkpoint_hidden); }
            sequence.ledger.resize(base);
            sequence.prefix_hashes.truncate(base);
            drop_context_checkpoints_after(sequence, base);
            maybe_capture_turn_rollback(sequence, request, prompt, request_plan.reuse, base,
                                        prompt_tokens, request_plan.capture_context_checkpoints,
                                        request_plan.capture_context_checkpoint, false);
        } else if (qwen3_6::detail::is_staged_checkpoint_restore(request_plan.reuse)) {
            if (!sequence.kv || sequence.text_kv_valid < base) {
                throw std::logic_error("resident context checkpoint has no complete KV allocation");
            }
            sequence.text_kv_valid = base;
            if (speculative_backend == SpeculativeBackend::Mtp) {
                const std::uint32_t mtp_base = base == 0 ? 0 : base - 1;
                if (!request_plan.prepare_mtp || sequence.mtp_kv_valid < mtp_base) {
                    throw std::logic_error(
                        "context-checkpoint MTP KV is shorter than the bridge frontier");
                }
                sequence.mtp_kv_valid = mtp_base;
            } else if (speculative_backend == SpeculativeBackend::DFlash) {
                if (!dflash || sequence.dflash_context_frontier < base) {
                    throw std::logic_error("planned DFlash context checkpoint is unavailable");
                }
                if constexpr (DFlashConfig::full_layers > 0) {
                    if (!sequence.kv->backend) {
                        throw std::logic_error("planned DFlash context checkpoint is unavailable");
                    }
                }
                sequence.dflash_context_frontier = base;
            }
            trim_sequence_kv(sequence, base, backend_kv_valid(sequence));
            resize_sequence_kv_entitlement(sequence, request_plan.text_kv_page_entitlement,
                                           request_plan.backend_kv_page_entitlement);
            restore_context_checkpoint_state(sequence, base);
            if (qwen3_6::detail::occupy_drops_rewrite_ahead_of_restore(
                    request_plan.reuse, sequence.rewrite_checkpoint.valid,
                    sequence.rewrite_checkpoint.frontier, base)) {
                sequence.rewrite_checkpoint = {};
            }
            sequence.ledger.resize(base);
            sequence.prefix_hashes.truncate(base);
            drop_context_checkpoints_after(sequence, base);
            maybe_capture_turn_rollback(sequence, request, prompt, request_plan.reuse, base,
                                        prompt_tokens, request_plan.capture_context_checkpoints,
                                        request_plan.capture_context_checkpoint, false);
        } else {
            throw std::logic_error("request plan has an invalid prefix reuse path");
        }

        trim_sequence_kv(sequence, base, backend_kv_valid(sequence));
        bind_sequence_kv(sequence);
        const std::uint32_t backend_materialized =
            speculative_backend == SpeculativeBackend::Mtp
                ? std::min(capacity,
                           prompt_tokens + (initial_mtp_extent == 0 ? 0U : initial_mtp_extent - 1U))
            : speculative_backend == SpeculativeBackend::DFlash && DFlashConfig::full_layers > 0
                ? prompt_tokens
                : 0U;
        materialize_sequence_kv(sequence, prompt_tokens, backend_materialized);
        request.output         = output;
        request.token_logprobs = request_plan.token_logprobs;
        set_device_i32(io.prefill_logprobs.row_enabled, request.token_logprobs ? 1 : 0);
        install_sampling(sequence, request, request_plan.sampling);
        sequence.rope_delta = prompt.rope_delta;
        set_device_i32(io.rope_delta, sequence.rope_delta);

        if (request_plan.rewrite_checkpoint_action == RewriteCheckpointAction::Drop) {
            sequence.rewrite_checkpoint = {};
        } else if (request_plan.rewrite_checkpoint_action ==
                   RewriteCheckpointAction::ReclassifyExisting) {
            sequence.rewrite_checkpoint.kind = prompt.identity.rewrite_checkpoint.value().kind;
        }
        request.timings = {};
        request.pending = {};
        request.restored_context_checkpoint_tokens =
            qwen3_6::detail::is_staged_checkpoint_restore(request_plan.reuse) ? base : 0;
        sequence.tier_unpacked_context_base = 0;
        sequence.tier_unpacked_context_hash = {};
        sequence.mtp_draft_count            = 0;
        sequence.tail_hidden_valid          = base == prompt_tokens && sequence.tail_hidden_valid;
        sequence.prefix_hashes.clear();
        sequence.ledger.assign(prompt.token_ids.begin(), prompt.token_ids.end());
        sequence.prefix_identity.assign(prompt);
        if (request_plan.prompt_hashes) {
            if (request_plan.prompt_hashes->size() != prompt.token_ids.size() + 1) {
                throw std::logic_error("planned prompt hash chain does not describe the prompt");
            }
            sequence.prefix_hashes.assign(*request_plan.prompt_hashes);
        }

        if (speculative_backend == SpeculativeBackend::DFlash) {
            if (!dflash || !io.dflash_decode || !sequence.kv) {
                throw std::logic_error("DFlash prefill state is incomplete");
            }
            if constexpr (DFlashConfig::full_layers > 0) {
                if (!sequence.kv->backend) {
                    throw std::logic_error("DFlash prefill state is incomplete");
                }
            }
            *dflash_host_ingress          = {};
            dflash_host_ingress->lanes[0] = static_cast<std::int32_t>(sequence.lane);
            dflash_host_ingress->dflash_kv_table_rows[0] =
                sequence.kv->backend ? sequence.kv->backend->bound_row() : 0;
            CUDA_CHECK(cudaMemcpyAsync(io.dflash_decode->ingress.data, dflash_host_ingress,
                                       sizeof(qwen3_6::DFlashDecodeIngress), cudaMemcpyHostToDevice,
                                       device.stream));
        }

        const bool host_input_consumed = prompt.has_media() && !request_plan.vision;
        if (host_input_consumed) { prompt.release_media_payload(); }

        RequestControl::Prefill prefill{
            .prompt                      = std::move(prompt),
            .vision_plan                 = std::move(request_plan.vision),
            .vision                      = nullptr,
            .transient                   = transient,
            .rewrite_checkpoint_capture  = request_plan.rewrite_checkpoint_capture,
            .base                        = base,
            .cursor                      = base,
            .prompt_tokens               = prompt_tokens,
            .initial_mtp_extent          = initial_mtp_extent,
            .elapsed_seconds             = 0.0,
            .host_input_consumed_pending = host_input_consumed,
            .prepare_mtp                 = request_plan.prepare_mtp,
            .reuse                       = request_plan.reuse,
            .reuse_source                = request_plan.reuse_source,
            .mtp_bridge                  = request_plan.mtp_bridge,
            .capture_context_checkpoints = request_plan.capture_context_checkpoints,
        };
        request.prefill.emplace(std::move(prefill));
        request.prompt_tokens = prompt_tokens;
        auto& staged          = *request.prefill;
        if (staged.vision_plan) {
            staged.vision = std::make_unique<schedule::VisionPrefillSession>(
                device, model, work, staged.prompt, *staged.vision_plan, staged.transient);
        }
        staged.elapsed_seconds = std::chrono::duration<double>(Clock::now() - started).count();
        request.lifecycle      = Lifecycle::Prefilling;
        const runtime::PrefillStepResult first = advance_prefill(sequence, request);
        sequence.use_tick                      = next_use_tick_++;
        return first;
    } catch (...) {
        synchronize_for_failure_cleanup(device);
        clear_lane(sequence, request);
        throw;
    }
}

runtime::PrefillStepResult ProgramImplCore::advance_prefill_lane(std::uint32_t lane) {
    if (lane >= max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    return advance_prefill(sequences[lane], requests[lane]);
}

void ProgramImplCore::resolve_prefill_lane(std::uint32_t lane, bool terminal) {
    if (lane >= max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    if (requests[lane].pending.kind != PendingKind::Begin) {
        throw std::logic_error("resolve_prefill_lane requires a pending prefill token");
    }
    resolve_non_speculative_pending(sequences[lane], requests[lane], 1, terminal);
}

void ProgramImplCore::resolve_pending_batch(std::span<const std::uint32_t> lanes,
                                            std::span<const std::uint32_t> accepted_tokens,
                                            std::span<const std::uint8_t> terminal,
                                            std::span<const std::uint8_t> cancelled,
                                            std::span<const std::uint8_t> rejected) {
    if (lanes.empty() || lanes.size() > max_concurrency || accepted_tokens.size() != lanes.size() ||
        terminal.size() != lanes.size() || cancelled.size() != lanes.size() ||
        (!rejected.empty() && rejected.size() != lanes.size())) {
        throw std::invalid_argument("pending batch resolution has inconsistent membership");
    }

    const auto row_rejected = [&](std::size_t row) {
        return !rejected.empty() && rejected[row] != 0;
    };

    if (speculative_backend == SpeculativeBackend::None) {
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            const std::uint32_t lane = lanes[row];
            if (lane >= max_concurrency || requests[lane].lifecycle != Lifecycle::Pending ||
                requests[lane].pending.kind != PendingKind::Ordinary) {
                throw std::logic_error("ordinary pending batch no longer matches Program state");
            }
            if (cancelled[row]) {
                SequenceState& sequence        = sequences[lane];
                RequestControl& request        = requests[lane];
                const PendingCandidate pending = request.pending;
                if (pending.produced != 1 || sequence.ledger.size() != pending.base_S + 1 ||
                    sequence.ledger_frontier != pending.base_S ||
                    sequence.execution_frontier != pending.base_E) {
                    throw std::logic_error(
                        "cancelled ordinary pending round is not a single staged token");
                }
                // Ordinary decode overwrote current GDN and tail-hidden state.
                // No snapshot exists for pending.base_E, so truncating the ledger
                // cannot roll back this provisional round. Release this lane;
                // published output remains unchanged and peers retain their state.
                clear_lane(sequence, request);
            } else if (row_rejected(row)) {
                throw std::logic_error("ordinary pending rounds cannot be rejected");
            } else {
                resolve_non_speculative_pending(sequences[lane], requests[lane],
                                                accepted_tokens[row], terminal[row] != 0);
            }
        }
        return;
    }

    if (!replay_records) {
        throw std::logic_error("speculative pending batch has no ReplaySSM records");
    }

    std::array<ops::GdnReplayFoldRow, kMaximumConcurrency> fold_rows{};
    std::array<std::int32_t, kMaximumConcurrency> hidden_selectors{};
    bool needs_hidden_correction = false;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency || requests[lane].lifecycle != Lifecycle::Pending ||
            requests[lane].pending.kind != PendingKind::Speculative) {
            throw std::logic_error("speculative pending batch no longer matches Program state");
        }
        const PendingCandidate& pending = requests[lane].pending;
        const SequenceState& sequence   = sequences[lane];
        if (sequence.execution_frontier != pending.base_E ||
            sequence.ledger_frontier != pending.base_S ||
            sequence.ledger.size() != pending.base_S ||
            sequence.prefix_identity.size() != pending.base_S ||
            sequence.text_kv_valid != pending.base_E ||
            (speculative_backend == SpeculativeBackend::Mtp &&
             sequence.mtp_kv_valid != pending.base_E) ||
            (speculative_backend == SpeculativeBackend::DFlash &&
             sequence.dflash_context_frontier != pending.base_E)) {
            throw std::logic_error("speculative pending row is not at its recorded base");
        }
        const bool retry              = row_rejected(row);
        const std::uint32_t committed = (cancelled[row] || retry) ? 0U : accepted_tokens[row];
        if ((cancelled[row] && accepted_tokens[row] != 0) ||
            (retry && (cancelled[row] || terminal[row] || accepted_tokens[row] != 0)) ||
            (!cancelled[row] && !retry && (committed == 0 || committed > pending.produced))) {
            throw std::logic_error("speculative pending row has an invalid committed prefix");
        }
        fold_rows[row] = ops::GdnReplayFoldRow{
            .linear_state_slot = LinearStateSlots::current_state_slot(lane, max_concurrency),
            .commit_columns    = static_cast<std::int32_t>(committed),
            // A negative path length selects the ordinary chain prefix. Zero is a
            // valid tree path length, so value-initialization would otherwise make
            // every non-tree DFlash commit fold no GDN columns at all.
            .path_length = -1,
        };
        const bool tree_fold = pending.tree_verify;
        if (tree_fold && committed > 0) {
            fold_rows[row].path_length = static_cast<std::int32_t>(committed);
            // LLD Capture/run: host fold_path stays packed at row * W_ceil + i.
            for (std::uint32_t i = 0; i < committed; ++i) {
                fold_rows[row].path[i] =
                    dflash_host_egress->fold_path[row * dflash_verify_width + i];
            }
        }
        const bool partial_commit =
            !cancelled[row] && !retry && committed > 0 && committed < pending.produced;
        if (tree_fold && committed > 0) {
            hidden_selectors[row] = fold_rows[row].path[committed - 1U];
        } else {
            hidden_selectors[row] =
                static_cast<std::int32_t>(partial_commit ? committed - 1U : pending.produced - 1U);
        }
        needs_hidden_correction = needs_hidden_correction || partial_commit;
    }

    const auto tail_started = Clock::now();
    // The fold, compaction and hidden correction are ordered before the next round's work on the
    // compute stream, so the host prepares that round while they run. A context append's ingress
    // copy still reads the pinned staging the next round rewrites.
    bool ingress_copy_pending = false;
    try {
        const auto committed_rows =
            std::span<const ops::GdnReplayFoldRow>(fold_rows.data(), lanes.size());
        if (gdn_history) {
            ops::gdn_history_commit(*replay_records, *gdn_history,
                                    decoder->linear_attention.all_layers_view(), committed_rows,
                                    device.stream);
            for (const auto& row : committed_rows) {
                const auto lane     = static_cast<std::uint32_t>(row.linear_state_slot);
                const auto accepted = static_cast<std::uint32_t>(
                    row.path_length < 0 ? row.commit_columns : row.path_length);
                const auto length         = gdn_history_lengths[lane] + accepted;
                gdn_history_lengths[lane] = length >= 4U ? 0U : length;
            }
        } else {
            ops::gdn_replay_fold(*replay_records, decoder->linear_attention.all_layers_view(),
                                 committed_rows, device.stream);
        }

        if (needs_hidden_correction) {
            const auto batch = static_cast<std::int32_t>(lanes.size());
            Tensor selector_tensor;
            Tensor hidden;
            Tensor selected;
            Tensor destinations;
            if (speculative_backend == SpeculativeBackend::Mtp && io.mtp_decode) {
                qwen3_6::MtpDecodeState& frame = *io.mtp_decode;
                selector_tensor                = frame.current_extents.slice(0, 0, batch);
                hidden                         = frame.target_hidden.slice(2, 0, batch);
                selected     = frame.target_continuation_hidden.slice(1, 0, batch);
                destinations = frame.lanes.slice(0, 0, batch);
            } else if (speculative_backend == SpeculativeBackend::DFlash && io.dflash_decode) {
                qwen3_6::DFlashDecodeState& frame = *io.dflash_decode;
                selector_tensor                   = frame.proposal_extents.slice(0, 0, batch);
                hidden                            = frame.target_hidden.slice(2, 0, batch);
                selected     = frame.target_continuation_hidden.slice(1, 0, batch);
                destinations = frame.lanes.slice(0, 0, batch);
            } else {
                throw std::logic_error("partial speculative commit has no target frame");
            }
            CUDA_CHECK(cudaMemcpyAsync(selector_tensor.data, hidden_selectors.data(),
                                       lanes.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice,
                                       device.stream));
            ops::speculative_select_accepted_hidden(hidden, selector_tensor, selected,
                                                    device.stream);
            ops::scatter(selected, destinations, tail_hidden_store, device.stream);
        }

        if (speculative_backend == SpeculativeBackend::DFlash && io.dflash_decode) {
            // LLD PendingCandidate: tree vs chain from this round's pending, not Program N.
            bool tree_fold_batch = false;
            for (std::size_t row = 0; row < lanes.size(); ++row) {
                tree_fold_batch = tree_fold_batch || requests[lanes[row]].pending.tree_verify;
            }
            if (tree_fold_batch) {
                qwen3_6::DFlashDecodeState& frame = *io.dflash_decode;
                const auto batch                  = static_cast<std::int32_t>(lanes.size());
                Tensor compact_counts             = frame.append_counts.slice(0, 0, batch);
                std::array<std::int32_t, kMaximumConcurrency> host_counts{};
                for (std::size_t row = 0; row < lanes.size(); ++row) {
                    host_counts[row] =
                        cancelled[row] ? 0 : static_cast<std::int32_t>(accepted_tokens[row]);
                }
                CUDA_CHECK(cudaMemcpyAsync(compact_counts.data, host_counts.data(),
                                           lanes.size() * sizeof(std::int32_t),
                                           cudaMemcpyHostToDevice, device.stream));
                Tensor kv_rows = frame.text_kv_table_rows.slice(0, 0, batch);
                Tensor prefix  = frame.execution_frontiers.slice(0, 0, batch);
                // Tree accept mutates `execution_frontiers` as sampler `lengths`
                // (E += committed). Compact copies packed slots E+path[i] → E+i,
                // so the prefix must stay the verify-time window base.
                std::array<std::int32_t, kMaximumConcurrency> host_prefix{};
                for (std::size_t row = 0; row < lanes.size(); ++row) {
                    host_prefix[row] =
                        checked_i32(requests[lanes[row]].pending.base_E, "DFlash compact prefix");
                }
                CUDA_CHECK(cudaMemcpyAsync(prefix.data, host_prefix.data(),
                                           lanes.size() * sizeof(std::int32_t),
                                           cudaMemcpyHostToDevice, device.stream));
                const std::int32_t verify_w = static_cast<std::int32_t>(dflash_verify_width);
                bool identity_path          = true;
                for (std::size_t row = 0; row < lanes.size() && identity_path; ++row) {
                    if (cancelled[row] || accepted_tokens[row] == 0) { continue; }
                    const std::int32_t committed = static_cast<std::int32_t>(accepted_tokens[row]);
                    for (std::int32_t i = 0; i < committed; ++i) {
                        if (fold_rows[row].path[i] != i) {
                            identity_path = false;
                            break;
                        }
                    }
                }
                if (!identity_path) {
                    Tensor path   = frame.fold_path.slice(0, 0, verify_w).slice(1, 0, batch);
                    Tensor lane_t = frame.lanes.slice(0, 0, batch);
                    for (std::uint32_t layer = 0; layer < decoder->text_kv.layers(); ++layer) {
                        ops::gqa_kv_compact_path(decoder->text_kv.batch_layer_view(layer), kv_rows,
                                                 prefix, path, compact_counts, device.stream);
                    }
                    ops::gather_bf16_path(dflash->pending_features, lane_t, path, compact_counts,
                                          device.stream);
                }
            }
            std::array<std::uint32_t, kMaximumConcurrency> append_lanes{};
            std::array<std::uint32_t, kMaximumConcurrency> append_starts{};
            std::array<std::uint32_t, kMaximumConcurrency> append_counts{};
            std::size_t append_size = 0;
            for (std::size_t row = 0; row < lanes.size(); ++row) {
                if (!cancelled[row] && !row_rejected(row) && terminal[row]) {
                    append_lanes[append_size]  = lanes[row];
                    append_starts[append_size] = requests[lanes[row]].pending.base_E;
                    append_counts[append_size] = accepted_tokens[row];
                    ++append_size;
                }
            }
            if (append_size != 0) {
                ingress_copy_pending = true;
                enqueue_dflash_context_append(
                    std::span<const std::uint32_t>(append_lanes.data(), append_size),
                    std::span<const std::uint32_t>(append_starts.data(), append_size),
                    std::span<const std::uint32_t>(append_counts.data(), append_size));
                // Terminal retention exposes a dense checkpoint after the append tail's fence.
                for (std::size_t index = 0; index < append_size; ++index) {
                    materialize_gdn_history(append_lanes[index]);
                }
            }
        }

        if (ingress_copy_pending) { device.synchronize(); }
        work.reset();
    } catch (...) {
        synchronize_for_failure_cleanup(device);
        work.reset();
        for (const std::uint32_t lane : lanes) {
            if (lane < max_concurrency) { clear_lane(sequences[lane], requests[lane]); }
        }
        throw;
    }

    const double tail_seconds = std::chrono::duration<double>(Clock::now() - tail_started).count();
    const std::uint32_t width =
        speculative_backend == SpeculativeBackend::DFlash ? dflash_verify_width : draft_window + 1U;
    try {
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence = sequences[lanes[row]];
            RequestControl& request = requests[lanes[row]];
            if (cancelled[row]) {
                sequence.tail_hidden_valid = false;
                retain_committed_sequence(sequence, request);
                continue;
            }
            if (row_rejected(row)) {
                const PendingCandidate pending = request.pending;
                const TokenId* token_base =
                    speculative_backend == SpeculativeBackend::Mtp
                        ? mtp_host_egress->licensed_tokens.data() + row * width
                        : dflash_host_egress->licensed_tokens.data() + row * width;
                rollback_sampling_counts(device, request.sampling_host,
                                         std::span<const TokenId>(token_base, pending.produced));
                rollback_speculative_stats(request, pending);
                sequence.tail_hidden_valid = false;
                request.lifecycle          = Lifecycle::Active;
                request.pending            = {};
                request.timings.decode_seconds += tail_seconds;
                continue;
            }

            const PendingCandidate pending = request.pending;
            const std::uint32_t committed  = accepted_tokens[row];
            const TokenId* token_base =
                speculative_backend == SpeculativeBackend::Mtp
                    ? mtp_host_egress->licensed_tokens.data() + row * width
                    : dflash_host_egress->licensed_tokens.data() + row * width;
            sequence.ledger.insert(sequence.ledger.end(), token_base, token_base + committed);
            sequence.prefix_identity.append_generated(committed, sequence.rope_delta);
            sequence.execution_frontier = pending.base_E + committed;
            sequence.ledger_frontier    = pending.base_S + committed;
            sequence.text_kv_valid      = sequence.execution_frontier;
            sequence.tail_hidden_valid  = true;
            if (committed < pending.produced) {
                trim_speculative_stats_to_commit(request, pending, committed);
            }

            if (speculative_backend == SpeculativeBackend::Mtp) {
                sequence.mtp_kv_valid = sequence.execution_frontier;
                if (terminal[row] || committed < pending.produced) {
                    sequence.mtp_draft_count = 0;
                } else {
                    const std::int32_t next  = mtp_host_egress->next_extents[row];
                    sequence.mtp_draft_count = static_cast<std::uint32_t>(next);
                    for (std::uint32_t step = 0; step < sequence.mtp_draft_count; ++step) {
                        // Step-major egress: all rows of draft step 0, then step 1, and so on.
                        const std::size_t draft_index =
                            (static_cast<std::size_t>(step) * max_concurrency) + row;
                        sequence.mtp_drafts[step] = mtp_host_egress->next_drafts[draft_index];
                    }
                }
            } else {
                sequence.dflash_context_frontier =
                    terminal[row] ? sequence.execution_frontier : pending.base_E;
            }

            trim_sequence_kv(sequence, sequence.text_kv_valid, backend_kv_valid(sequence));
            if (terminal[row]) {
                release_sequence_growth_entitlement(sequence);
                unbind_sequence_kv(sequence);
                sequence.retained = true;
                request.lifecycle = Lifecycle::Complete;
            } else {
                request.lifecycle = Lifecycle::Active;
            }
            request.pending = {};
            request.timings.decode_seconds += tail_seconds;
        }
    } catch (...) {
        synchronize_for_failure_cleanup(device);
        for (const std::uint32_t lane : lanes) {
            if (lane < max_concurrency) { clear_lane(sequences[lane], requests[lane]); }
        }
        throw;
    }
}

void ProgramImplCore::abort_lane(std::uint32_t lane) noexcept {
    if (lane >= max_concurrency) { return; }
    clear_lane(sequences[lane], requests[lane]);
}

void ProgramImplCore::retain_committed_sequence(SequenceState& sequence, RequestControl& request) {
    if (!sequence.kv) {
        throw std::logic_error("cannot retain a lane with no KV allocation bundle");
    }
    materialize_gdn_history(sequence.lane);
    request.prefill.reset();
    request.pending          = {};
    sequence.mtp_draft_count = 0;
    trim_sequence_kv(sequence, sequence.text_kv_valid, backend_kv_valid(sequence));
    release_sequence_growth_entitlement(sequence);
    unbind_sequence_kv(sequence);
    sequence.retained = true;
    request.lifecycle = Lifecycle::Complete;
}

bool ProgramImplCore::retain_reusable_lane(std::uint32_t lane) {
    if (lane >= max_concurrency) { return false; }
    SequenceState& sequence = sequences[lane];
    RequestControl& request = requests[lane];
    if (request.lifecycle == Lifecycle::Active) {
        if (!sequence.kv) { return false; }
        retain_committed_sequence(sequence, request);
        return true;
    }
    return sequence.retained && sequence.kv && request.lifecycle == Lifecycle::Complete;
}

bool ProgramImplCore::copy_reusable_prompt(std::uint32_t lane, std::uint32_t prompt_tokens,
                                           std::vector<TokenId>& tokens,
                                           std::uint32_t& rewrite_frontier) const {
    tokens.clear();
    rewrite_frontier = 0;
    if (lane >= max_concurrency || prompt_tokens == 0) { return false; }
    const SequenceState& sequence = sequences[lane];
    const RequestControl& request = requests[lane];
    const bool held = sequence.kv && sequence.ledger.size() >= prompt_tokens &&
                      (request.lifecycle == Lifecycle::Active ||
                       (sequence.retained && request.lifecycle == Lifecycle::Complete));
    if (!held) { return false; }
    tokens.assign(sequence.ledger.begin(),
                  sequence.ledger.begin() + static_cast<std::ptrdiff_t>(prompt_tokens));
    if (sequence.rewrite_checkpoint.valid && sequence.rewrite_checkpoint.frontier != 0 &&
        sequence.rewrite_checkpoint.frontier <= prompt_tokens) {
        rewrite_frontier = sequence.rewrite_checkpoint.frontier;
    }
    return true;
}

bool ProgramImplCore::revert_cancelled_prefill_lane(std::uint32_t lane) {
    if (lane >= max_concurrency) { return false; }
    SequenceState& sequence = sequences[lane];
    RequestControl& request = requests[lane];
    if (request.lifecycle != Lifecycle::Prefilling || !sequence.kv) { return false; }

    const auto in_bounds = [&](std::uint32_t frontier) {
        return frontier != 0 && frontier <= sequence.ledger.size() &&
               frontier <= sequence.prefix_identity.size();
    };
    const auto head_at = [&](std::uint32_t frontier) {
        return std::find_if(
            sequence.context_checkpoints.begin(), sequence.context_checkpoints.end(),
            [frontier](const ContextCheckpointHead& head) { return head.frontier == frontier; });
    };
    const bool rewrite_ok =
        sequence.rewrite_checkpoint.valid && in_bounds(sequence.rewrite_checkpoint.frontier);

    std::uint32_t frontier              = 0;
    bool restore_staged                 = false;
    ninfer::PrefixReusePath staged_path = ninfer::PrefixReusePath::RestoreTurnRollback;
    if (request.prefill) {
        const std::uint32_t base = request.prefill->base;
        if (in_bounds(base)) {
            if (rewrite_ok && sequence.rewrite_checkpoint.frontier == base) {
                frontier = base;
            } else if (const auto head = head_at(base);
                       head != sequence.context_checkpoints.end()) {
                frontier       = base;
                restore_staged = true;
                staged_path = qwen3_6::detail::reuse_path_for_context_checkpoint_kind(head->kind);
            }
        }
    }
    if (frontier == 0 && rewrite_ok) { frontier = sequence.rewrite_checkpoint.frontier; }
    if (frontier == 0) { return false; }

    try {
        if (restore_staged) {
            restore_context_checkpoint_state(sequence, frontier);
            if (qwen3_6::detail::occupy_drops_rewrite_ahead_of_restore(
                    staged_path, sequence.rewrite_checkpoint.valid,
                    sequence.rewrite_checkpoint.frontier, frontier)) {
                sequence.rewrite_checkpoint = {};
            }
        } else {
            restore_rewrite_checkpoint_state(sequence);
            copy_tail(sequence, sequence.rewrite_checkpoint_hidden);
            if (speculative_backend == SpeculativeBackend::DFlash) {
                sequence.dflash_context_frontier = frontier;
            }
        }
        sequence.text_kv_valid = frontier;
        if (speculative_backend == SpeculativeBackend::Mtp) {
            sequence.mtp_kv_valid = frontier == 0 ? 0 : frontier - 1;
        }
        // Checkpoint frontiers are execution frontiers (text_kv_valid). RAM capture requires
        // ledger_frontier == execution_frontier + 1 == ledger.size(), matching a committed
        // Active sequence — keep one trailing ledger slot past the restored frontier.
        if (sequence.ledger.size() > frontier) {
            sequence.ledger.resize(frontier + 1);
            sequence.prefix_identity.truncate(frontier + 1);
            sequence.prefix_hashes.truncate(frontier + 1);
        } else {
            sequence.prefix_hashes.truncate(frontier);
            const TokenId pad = frontier == 0 ? TokenId{0} : sequence.ledger[frontier - 1];
            sequence.ledger.push_back(pad);
            sequence.prefix_identity.truncate(frontier);
            sequence.prefix_identity.append_generated(1, sequence.rope_delta);
        }
        sequence.execution_frontier = frontier;
        sequence.ledger_frontier    = frontier + 1;
        drop_context_checkpoints_after(sequence, frontier);
        if (staging_.occupied && staging_.lane == sequence.lane) { unoccupy_staging(); }
        device.synchronize();
        retain_committed_sequence(sequence, request);
        return true;
    } catch (...) {
        synchronize_for_failure_cleanup(device);
        clear_lane(sequence, request);
        return false;
    }
}

bool ProgramImplCore::has_retained_lane(std::uint32_t lane) const noexcept {
    return lane < max_concurrency && sequences[lane].retained;
}

std::uint64_t ProgramImplCore::retained_use_tick(std::uint32_t lane) const noexcept {
    return has_retained_lane(lane) ? sequences[lane].use_tick : 0;
}

void ProgramImplCore::mark_turn_closed(std::uint32_t lane) noexcept {
    if (has_retained_lane(lane)) { sequences[lane].turn_closed = true; }
}

void ProgramImplCore::evict_retained_lane(std::uint32_t lane) noexcept {
    if (!has_retained_lane(lane)) { return; }
    clear_lane(sequences[lane], requests[lane]);
}

bool ProgramImplCore::capture_cuts_at_rewrite(const SequenceState& sequence) const noexcept {
    const RewriteCheckpoint& checkpoint = sequence.rewrite_checkpoint;
    return sequence.turn_closed && checkpoint.valid &&
           checkpoint.kind == RewriteCheckpointKind::TurnClosure && checkpoint.frontier != 0 &&
           checkpoint.frontier == sequence.closure_frontier &&
           checkpoint.frontier < sequence.execution_frontier &&
           checkpoint.frontier <= sequence.text_kv_valid &&
           checkpoint.frontier < sequence.ledger.size() &&
           checkpoint.frontier < sequence.prefix_identity.size();
}

qwen3_6::detail::RamCaptureSource
ProgramImplCore::ram_capture_source(SequenceState& sequence,
                                    qwen3_6::detail::ResidentPrefixIdentity& cut_identity) {
    if (!sequence.kv || !sequence.retained) {
        throw std::logic_error("RAM capture requires a retained sequence bundle");
    }
    if (capture_cuts_at_rewrite(sequence)) {
        return cut_ram_capture_source(sequence, cut_identity);
    }
    qwen3_6::detail::RamCaptureSource source;
    source.execution_frontier      = sequence.execution_frontier;
    source.ledger_frontier         = sequence.ledger_frontier;
    source.rope_delta              = sequence.rope_delta;
    source.text_kv_valid           = sequence.text_kv_valid;
    source.mtp_kv_valid            = sequence.mtp_kv_valid;
    source.dflash_context_frontier = sequence.dflash_context_frontier;
    source.tail_hidden_valid       = sequence.tail_hidden_valid;
    source.rewrite_valid           = sequence.rewrite_checkpoint.valid;
    source.rewrite_kind            = sequence.rewrite_checkpoint.kind;
    source.rewrite_frontier        = sequence.rewrite_checkpoint.frontier;
    source.ledger                  = sequence.ledger;
    source.identity                = &sequence.prefix_identity;
    source.hash_f = sequence.prefix_hashes.at(sequence.ledger, sequence.prefix_identity,
                                              sequence.execution_frontier);
    if (sequence.rewrite_checkpoint.valid && sequence.rewrite_checkpoint.frontier != 0) {
        source.hash_c       = sequence.prefix_hashes.at(sequence.ledger, sequence.prefix_identity,
                                                        sequence.rewrite_checkpoint.frontier);
        source.hash_c_valid = true;
    }
    source.text      = &sequence.kv->text;
    source.text_pool = &decoder->text_kv.pool();
    if (sequence.kv->backend) {
        source.backend      = &*sequence.kv->backend;
        source.backend_pool = &backend_kv_cache()->pool();
    }
    source.gdn              = &decoder->linear_attention;
    source.gdn_current_slot = LinearStateSlots::current_state_slot(sequence.lane, max_concurrency);
    source.tail_hidden      = &sequence.tail_hidden;
    if (sequence.rewrite_checkpoint.valid) {
        // The host tier reads the image on the copy stream after its fence.
        const ContextCheckpointHead& image = sequence.rewrite_image;
        source.rewrite_checkpoint_hidden   = &sequence.rewrite_checkpoint_hidden;
        source.rewrite_state               = qwen3_6::detail::RewriteStateHostSource{
            .conv        = image.conv.data(),
            .recurrent   = image.recurrent.data(),
            .dflash      = image.dflash ? image.dflash.data() : nullptr,
            .copies_done = image.copies_done,
        };
    }
    source.ladder_heads.reserve(sequence.context_checkpoints.size());
    for (const ContextCheckpointHead& head : sequence.context_checkpoints) {
        RamLadderHead view;
        view.frontier    = head.frontier;
        view.hash        = head.hash;
        view.kind        = head.kind;
        view.copies_done = head.copies_done;
        if (head.conv) {
            view.conv       = head.conv.data();
            view.conv_bytes = head.conv.size();
        }
        if (head.recurrent) {
            view.recurrent       = head.recurrent.data();
            view.recurrent_bytes = head.recurrent.size();
        }
        if (head.hidden) {
            view.hidden       = head.hidden.data();
            view.hidden_bytes = head.hidden.size();
        }
        if (head.dflash) {
            view.dflash       = head.dflash.data();
            view.dflash_bytes = head.dflash.size();
        }
        source.ladder_heads.push_back(view);
    }
    if (dflash) {
        source.dflash_local = &dflash->local;
        source.dflash_lane  = static_cast<std::int32_t>(sequence.lane);
    }
    source.disk_entry_id = sequence.disk_entry_id;
    source.stream        = device.copy_stream;
    return source;
}

// The lane as of its turn checkpoint F: ledger, KV and identity end at F (plus the one ledger
// slot a committed lane keeps past its execution frontier), the checkpoint state and hidden are
// the current set, and no rewrite set or later head is stored.
qwen3_6::detail::RamCaptureSource
ProgramImplCore::cut_ram_capture_source(SequenceState& sequence,
                                        qwen3_6::detail::ResidentPrefixIdentity& cut_identity) {
    // The caller has established the retained bundle; value() throws if that contract is broken.
    const auto& kv               = sequence.kv.value();
    const std::uint32_t frontier = sequence.rewrite_checkpoint.frontier;
    cut_identity                 = sequence.prefix_identity;
    cut_identity.truncate(frontier + 1);

    qwen3_6::detail::RamCaptureSource source;
    source.execution_frontier = frontier;
    source.ledger_frontier    = frontier + 1;
    source.rope_delta         = sequence.rope_delta;
    source.text_kv_valid      = frontier;
    source.mtp_kv_valid       = speculative_backend == SpeculativeBackend::Mtp ? frontier - 1 : 0;
    source.dflash_context_frontier =
        speculative_backend == SpeculativeBackend::DFlash ? frontier : 0;
    source.tail_hidden_valid = true;
    source.ledger            = std::span<const TokenId>(sequence.ledger).first(frontier + 1);
    source.identity          = &cut_identity;
    source.hash_f = sequence.prefix_hashes.at(sequence.ledger, sequence.prefix_identity, frontier);
    source.text   = &kv.text;
    source.text_pool  = &decoder->text_kv.pool();
    source.text_pages = std::min(ninfer::pages_for_tokens(frontier), kv.text.mapped_page_count());
    if (kv.backend) {
        source.backend                     = &*kv.backend;
        source.backend_pool                = &backend_kv_cache()->pool();
        const std::uint32_t backend_tokens = speculative_backend == SpeculativeBackend::Mtp
                                                 ? source.mtp_kv_valid
                                                 : source.dflash_context_frontier;
        source.backend_pages =
            std::min(ninfer::pages_for_tokens(backend_tokens), kv.backend->mapped_page_count());
    }
    // The checkpoint image is the current state; the host tier reads it, like every head below,
    // on its host-copy stream after the image's own fence.
    const ContextCheckpointHead& image = sequence.rewrite_image;
    source.gdn                         = &decoder->linear_attention;
    source.gdn_current_slot = LinearStateSlots::current_state_slot(sequence.lane, max_concurrency);
    source.current_state    = qwen3_6::detail::RewriteStateHostSource{
        .conv        = image.conv.data(),
        .recurrent   = image.recurrent.data(),
        .dflash      = image.dflash ? image.dflash.data() : nullptr,
        .copies_done = image.copies_done,
    };
    source.tail_hidden = &sequence.rewrite_checkpoint_hidden;
    for (const ContextCheckpointHead& head : sequence.context_checkpoints) {
        if (head.frontier > frontier) { continue; }
        RamLadderHead view;
        view.frontier    = head.frontier;
        view.hash        = head.hash;
        view.kind        = head.kind;
        view.copies_done = head.copies_done;
        if (head.conv) {
            view.conv       = head.conv.data();
            view.conv_bytes = head.conv.size();
        }
        if (head.recurrent) {
            view.recurrent       = head.recurrent.data();
            view.recurrent_bytes = head.recurrent.size();
        }
        if (head.hidden) {
            view.hidden       = head.hidden.data();
            view.hidden_bytes = head.hidden.size();
        }
        if (head.dflash) {
            view.dflash       = head.dflash.data();
            view.dflash_bytes = head.dflash.size();
        }
        source.ladder_heads.push_back(view);
    }
    if (dflash) {
        source.dflash_local = &dflash->local;
        source.dflash_lane  = static_cast<std::int32_t>(sequence.lane);
    }
    source.disk_entry_id = sequence.disk_entry_id;
    source.stream        = device.copy_stream;
    return source;
}

bool ProgramImplCore::capture_retained_lane(std::uint32_t lane, std::uint64_t* ram_entry_id,
                                            bool may_block, bool* deferred,
                                            std::span<const std::uint64_t> attempt_ram_ids) {
    if (ram_entry_id != nullptr) { *ram_entry_id = 0; }
    if (deferred != nullptr) { *deferred = false; }
    if (!kv_ram_cache_ || !has_retained_lane(lane)) { return true; }
    qwen3_6::detail::KVRamCache& ram   = *kv_ram_cache_;
    qwen3_6::detail::KVDiskCache* disk = kv_disk_cache_ ? &*kv_disk_cache_ : nullptr;
    device.order_copy_after_compute();
    qwen3_6::detail::RamCaptureSource source;
    qwen3_6::detail::ResidentPrefixIdentity cut_identity;
    try {
        source = ram_capture_source(sequences[lane], cut_identity);
    } catch (const std::bad_alloc&) {
        ram.record_drop();
        return false;
    }
    for (;;) {
        const auto result = ram.capture(source);
        if (result.status == qwen3_6::detail::RamCaptureStatus::Captured) {
            if (disk != nullptr) {
                disk->note_ram_resident(result.entry_id, source.disk_entry_id);
                // Write the new entry behind on the disk worker so that a later
                // RAM eviction usually finds it durable instead of spilling it
                // synchronously on this thread. Shutdown flushes in order instead.
                if (!kv_tiers_shutdown_) { disk->request_idle_spill(); }
            }
            if (ram_entry_id != nullptr) { *ram_entry_id = result.entry_id; }
            return true;
        }
        if (result.status == qwen3_6::detail::RamCaptureStatus::Dropped) { return false; }
        if (disk != nullptr) {
            const auto reclaimed = disk->reclaim_ram_entry(may_block, attempt_ram_ids);
            if (reclaimed == qwen3_6::detail::RamReclaim::Evicted ||
                reclaimed == qwen3_6::detail::RamReclaim::Retry) {
                continue;
            }
            // Not a drop: the caller retries once the disk worker's spill lands.
            if (reclaimed == qwen3_6::detail::RamReclaim::Pending) {
                if (deferred != nullptr) { *deferred = true; }
                return false;
            }
            ram.record_drop();
            return false;
        }
        // Without a disk tier the oldest unclaimed entry is dropped. Retired blocks waiting
        // only for their copy fence free before any live entry is dropped: a non-blocking
        // caller defers, a blocking one waits for them.
        if (ram.retired_fence_pending()) {
            if (!may_block && deferred != nullptr) {
                *deferred = true;
                return false;
            }
            if (may_block && ram.wait_retired_copies()) { continue; }
        }
        // This admission's own earlier captures are never victims: a deferral rolls them back,
        // so the retry would recapture and target them again.
        std::optional<std::uint64_t> victim;
        for (const std::uint64_t id : ram.unpinned_ids()) {
            if (std::find(attempt_ram_ids.begin(), attempt_ram_ids.end(), id) ==
                attempt_ram_ids.end()) {
                victim = id;
                break;
            }
        }
        if (!victim) {
            ram.record_drop();
            return false;
        }
        if (!may_block && !ram.copies_ready(*victim)) {
            if (deferred != nullptr) { *deferred = true; }
            return false;
        }
        if (may_block) { ram.wait_entry_copies(*victim); }
        (void)ram.evict_one_unpinned(*victim);
    }
}

void ProgramImplCore::fence_staging_copies() noexcept {
    if (staging_.d2d_done != nullptr) { (void)cudaEventSynchronize(staging_.d2d_done); }
    if (staging_.copies_done != nullptr) { (void)cudaEventSynchronize(staging_.copies_done); }
}

void ProgramImplCore::unoccupy_staging() noexcept {
    staging_.rewrite            = false;
    staging_.rewrite_generation = 0;
    staging_.occupied           = false;
    staging_.lane               = 0;
    staging_.frontier           = 0;
    staging_.hash               = {};
    staging_.kind               = qwen3_6::detail::ContextCheckpointKind::Ladder;
}

void ProgramImplCore::reload_turn_rollback_into_staging(std::uint32_t lane,
                                                        qwen3_6::detail::PrefixHash128 hash,
                                                        std::uint32_t frontier) {
    if (lane >= max_concurrency || staging_hidden.data == nullptr) {
        unoccupy_staging();
        return;
    }
    SequenceState& owner = sequences[lane];
    const auto head      = std::find_if(
        owner.context_checkpoints.begin(), owner.context_checkpoints.end(),
        [&](const ContextCheckpointHead& candidate) {
            return candidate.kind == qwen3_6::detail::ContextCheckpointKind::TurnRollback &&
                   candidate.frontier == frontier && candidate.hash == hash;
        });
    if (head == owner.context_checkpoints.end() || !head->conv || !head->recurrent ||
        !head->hidden) {
        unoccupy_staging();
        return;
    }
    CUDA_CHECK(cudaStreamWaitEvent(device.copy_stream, head->copies_done, 0));
    if (staging_.d2d_done != nullptr) {
        CUDA_CHECK(cudaStreamWaitEvent(device.copy_stream, staging_.d2d_done, 0));
    }
    const std::int32_t staging = LinearStateSlots::staging_state_slot(max_concurrency);
    decoder->linear_attention.unpack_slot_from_host(staging, head->conv.data(),
                                                    head->recurrent.data(), device.copy_stream);
    CUDA_CHECK(cudaMemcpyAsync(staging_hidden.data, head->hidden.data(), staging_hidden.bytes(),
                               cudaMemcpyHostToDevice, device.copy_stream));
    CUDA_CHECK(cudaEventRecord(staging_.copies_done, device.copy_stream));
    record_context_checkpoint_head_use(*head, device.copy_stream);
    staging_.rewrite  = false;
    staging_.occupied = true;
    staging_.lane     = lane;
    staging_.frontier = frontier;
    staging_.hash     = hash;
    staging_.kind     = qwen3_6::detail::ContextCheckpointKind::TurnRollback;
}

bool ProgramImplCore::staging_holds(std::uint32_t lane, qwen3_6::detail::PrefixHash128 hash,
                                    std::uint32_t frontier) const noexcept {
    return qwen3_6::detail::staging_holds_restore_identity(
        staging_.occupied, staging_.lane, staging_.hash, staging_.frontier, lane, hash, frontier);
}

PinnedImage ProgramImplCore::carve_checkpoint_image(std::size_t bytes) {
    if (bytes == 0) { return {}; }
    if (!checkpoint_image_slab_) { throw std::logic_error("checkpoint image slab is missing"); }
    void* data =
        checkpoint_image_slab_->try_alloc(bytes, qwen3_6::detail::kCheckpointImageAlignment);
    if (data == nullptr) { throw std::logic_error("checkpoint image slab is undersized"); }
    return {data, bytes};
}

std::optional<ContextCheckpointHead> ProgramImplCore::acquire_context_checkpoint_head() noexcept {
    if (context_checkpoint_pool_.empty()) { return std::nullopt; }
    std::optional<ContextCheckpointHead> head(std::move(context_checkpoint_pool_.back()));
    context_checkpoint_pool_.pop_back();
    return head;
}

void ProgramImplCore::record_context_checkpoint_head_use(ContextCheckpointHead& head,
                                                         cudaStream_t stream) {
    // Every published head owns its fence before any transfer can borrow it.
    CUDA_CHECK(cudaEventRecord(head.copies_done, stream));
}

void ProgramImplCore::recycle_context_checkpoint_head(ContextCheckpointHead&& head) noexcept {
    qwen3_6::detail::recycle_checkpoint_image(context_checkpoint_pool_, std::move(head));
}

void ProgramImplCore::clear_context_checkpoints(SequenceState& sequence) noexcept {
    for (ContextCheckpointHead& head : sequence.context_checkpoints) {
        recycle_context_checkpoint_head(std::move(head));
    }
    sequence.context_checkpoints.clear();
    sequence.next_context_mark = qwen3_6::detail::first_prefill_context_mark(context_marks);
}

void ProgramImplCore::drop_context_checkpoints_after(SequenceState& sequence,
                                                     std::uint32_t frontier) noexcept {
    auto& heads = sequence.context_checkpoints;
    for (auto it = heads.begin(); it != heads.end();) {
        if (qwen3_6::detail::retain_context_checkpoint_head(it->frontier, frontier)) {
            ++it;
            continue;
        }
        recycle_context_checkpoint_head(std::move(*it));
        it = heads.erase(it);
    }
    const auto next = qwen3_6::detail::next_prefill_context_mark(frontier, context_marks);
    sequence.next_context_mark = next.value_or(0);
}

void ProgramImplCore::install_ram_context_checkpoints(SequenceState& sequence,
                                                      const qwen3_6::detail::RamRestoredHost& host,
                                                      std::uint64_t entry_id,
                                                      std::uint32_t reuse_base) {
    if (!kv_ram_cache_) { throw std::logic_error("RAM checkpoint install requires a RAM tier"); }
    qwen3_6::detail::KVRamCache& ram = *kv_ram_cache_;
    std::vector<ContextCheckpointHead> heads;
    heads.reserve(host.ladder_images.size());
    std::vector<HostCopy> copies;
    copies.reserve(4 * host.ladder_images.size());
    std::vector<cudaEvent_t> fences;
    fences.reserve(host.ladder_images.size());
    try {
        for (const qwen3_6::detail::RamLadderImage& image : host.ladder_images) {
            // Occupy keeps only heads at or before the restored base.
            if (image.frontier > reuse_base) { continue; }
            const auto matches = [](const void* data, std::size_t bytes, const PinnedImage& slot) {
                return (data != nullptr ? bytes : 0) == slot.size();
            };
            if (context_checkpoint_pool_.empty()) { break; }
            const ContextCheckpointHead& layout = context_checkpoint_pool_.back();
            if (!matches(image.conv, image.conv_bytes, layout.conv) ||
                !matches(image.recurrent, image.recurrent_bytes, layout.recurrent) ||
                !matches(image.hidden, image.hidden_bytes, layout.hidden) ||
                !matches(image.dflash, image.dflash_bytes, layout.dflash)) {
                continue;
            }
            std::optional<ContextCheckpointHead> head = acquire_context_checkpoint_head();
            if (!head) { break; }
            head->frontier = image.frontier;
            head->hash     = image.hash;
            head->kind     = image.kind;
            if (head->conv) { copies.push_back({head->conv.data(), image.conv, image.conv_bytes}); }
            if (head->recurrent) {
                copies.push_back({head->recurrent.data(), image.recurrent, image.recurrent_bytes});
            }
            if (head->hidden) {
                copies.push_back({head->hidden.data(), image.hidden, image.hidden_bytes});
            }
            if (head->dflash) {
                copies.push_back({head->dflash.data(), image.dflash, image.dflash_bytes});
            }
            // A recycled head's last reader or writer finishes before the copies refill it.
            fences.push_back(head->copies_done);
            heads.push_back(std::move(*head));
        }
        // The copies run on the RAM tier's host-copy stream beside the entry's H2D, and only
        // each head's fence follows them: the restored lane's prefill does not wait for heads
        // it does not read. The entry's block stays alive until they land.
        ram.copy_from_entry(entry_id, copies, fences);
    } catch (...) {
        for (ContextCheckpointHead& head : heads) {
            recycle_context_checkpoint_head(std::move(head));
        }
        throw;
    }
    clear_context_checkpoints(sequence);
    sequence.context_checkpoints = std::move(heads);
}

void ProgramImplCore::recycle_pending_disk_checkpoint_heads() noexcept {
    for (ContextCheckpointHead& head : pending_disk_checkpoint_heads_) {
        recycle_context_checkpoint_head(std::move(head));
    }
    pending_disk_checkpoint_heads_.clear();
}

bool ProgramImplCore::captures_context_checkpoints() const noexcept {
    return speculative_backend == SpeculativeBackend::Mtp ||
           speculative_backend == SpeculativeBackend::DFlash;
}

void ProgramImplCore::snapshot_dflash_cyclic_to_staging(std::int32_t lane) {
    if (speculative_backend != SpeculativeBackend::DFlash) { return; }
    if (!dflash) { throw std::logic_error("context checkpoint DFlash cyclic image is incomplete"); }
    dflash->staging_local.copy_lane_from(dflash->local, lane, 0, device.stream);
}

void ProgramImplCore::pack_dflash_cyclic_to_head(ContextCheckpointHead& head) {
    if (speculative_backend != SpeculativeBackend::DFlash) { return; }
    if (!dflash || !head.dflash) {
        throw std::logic_error("context checkpoint DFlash cyclic image is incomplete");
    }
    dflash->staging_local.copy_lane_to_host(0, head.dflash.data(), device.copy_stream);
}

void ProgramImplCore::restore_dflash_cyclic_from_head(SequenceState& sequence,
                                                      const ContextCheckpointHead& head) {
    if (speculative_backend != SpeculativeBackend::DFlash) { return; }
    if (!dflash) { throw std::logic_error("context checkpoint restore requires DFlash state"); }
    CUDA_CHECK(cudaStreamWaitEvent(device.stream, head.copies_done, 0));
    if (!head.dflash || head.dflash.size() != dflash->local.lane_host_bytes()) {
        throw std::logic_error("context checkpoint DFlash cyclic image is incomplete");
    }
    dflash->local.copy_lane_from_host(head.dflash.data(), static_cast<std::int32_t>(sequence.lane),
                                      device.stream);
    sequence.dflash_context_frontier = head.frontier;
}

void ProgramImplCore::restore_context_checkpoint_state(SequenceState& sequence,
                                                       std::uint32_t base) {
    const auto head = std::find_if(
        sequence.context_checkpoints.begin(), sequence.context_checkpoints.end(),
        [base](const ContextCheckpointHead& candidate) { return candidate.frontier == base; });
    // A host-tier restore already unpacked the planned head into current; the installed head,
    // when the pool had one for it, only remains a later reuse candidate.
    if (sequence.tier_unpacked_context_base == base &&
        (head == sequence.context_checkpoints.end() ||
         sequence.tier_unpacked_context_hash == head->hash)) {
        reset_gdn_history(sequence.lane);
        return;
    }
    if (head == sequence.context_checkpoints.end()) {
        throw std::logic_error("context checkpoint head is missing at restore");
    }
    const std::int32_t current =
        LinearStateSlots::current_state_slot(sequence.lane, max_concurrency);
    if (qwen3_6::detail::restore_may_d2d_staging(staging_holds(sequence.lane, head->hash, base),
                                                 static_cast<bool>(head->conv) &&
                                                     static_cast<bool>(head->recurrent)) &&
        staging_.kind == head->kind && staging_hidden.data != nullptr) {
        if (staging_.copies_done != nullptr) {
            CUDA_CHECK(cudaStreamWaitEvent(device.stream, staging_.copies_done, 0));
        }
        decoder->linear_attention.copy_slot_2d(
            LinearStateSlots::staging_state_slot(max_concurrency), current, device.stream);
        reset_gdn_history(sequence.lane);
        copy_tail(sequence, staging_hidden);
        restore_dflash_cyclic_from_head(sequence, *head);
        record_context_checkpoint_head_use(*head, device.stream);
        return;
    }
    CUDA_CHECK(cudaStreamWaitEvent(device.stream, head->copies_done, 0));
    if (!head->conv || !head->recurrent) {
        throw std::logic_error("context checkpoint GDN image is incomplete");
    }
    decoder->linear_attention.unpack_slot_from_host(current, head->conv.data(),
                                                    head->recurrent.data(), device.stream);
    reset_gdn_history(sequence.lane);
    if (sequence.tail_hidden.bytes() != 0 && !head->hidden) {
        throw std::logic_error("context checkpoint hidden image is incomplete");
    }
    if (head->hidden) {
        if (head->hidden.size() != sequence.tail_hidden.bytes()) {
            throw std::logic_error("context checkpoint hidden geometry mismatch");
        }
        CUDA_CHECK(cudaMemcpyAsync(sequence.tail_hidden.data, head->hidden.data(),
                                   sequence.tail_hidden.bytes(), cudaMemcpyHostToDevice,
                                   device.stream));
        sequence.tail_hidden_valid = true;
    }
    restore_dflash_cyclic_from_head(sequence, *head);
    record_context_checkpoint_head_use(*head, device.stream);
}

// Startup-carved like the device slots it replaces: no prefill or restore path allocates it.
void ProgramImplCore::allocate_rewrite_image(SequenceState& sequence) {
    ContextCheckpointHead& image = sequence.rewrite_image;
    image.prepare_copy_event();
    image.conv = carve_checkpoint_image(decoder->linear_attention.conv_host_image_bytes());
    image.recurrent =
        carve_checkpoint_image(decoder->linear_attention.recurrent_host_image_bytes());
    image.dflash = carve_checkpoint_image(dflash ? dflash->local.lane_host_bytes() : 0);
}

// Snapshots the current GDN slot (and DFlash local lane) into the staging slot on the compute
// stream, then drains staging into the lane's pinned image on copy_stream behind later work.
// Staging keeps the image until another staging writer runs, so a prompt rewrite soon after
// restores it device-to-device.
void ProgramImplCore::capture_rewrite_image(SequenceState& sequence) {
    ContextCheckpointHead& image = sequence.rewrite_image;
    ++sequence.rewrite_image_generation;
    const std::int32_t current =
        LinearStateSlots::current_state_slot(sequence.lane, max_concurrency);
    const auto lane = static_cast<std::int32_t>(sequence.lane);
    if (staging_hidden.data == nullptr) {
        // No staging slot without MTP/DFlash: copy to the image on the compute stream.
        if (dflash) { throw std::logic_error("DFlash rewrite capture requires the staging lane"); }
        CUDA_CHECK(cudaStreamWaitEvent(device.stream, image.copies_done, 0));
        materialize_gdn_history(sequence.lane);
        decoder->linear_attention.pack_slot_to_host(current, image.conv.data(),
                                                    image.recurrent.data(), device.stream);
        record_context_checkpoint_head_use(image, device.stream);
        return;
    }
    unoccupy_staging();
    // Earlier copy_stream reads of staging (a previous drain or checkpoint pack) finish first.
    CUDA_CHECK(cudaStreamWaitEvent(device.stream, staging_.copies_done, 0));
    const std::int32_t staging = LinearStateSlots::staging_state_slot(max_concurrency);
    materialize_gdn_history(sequence.lane);
    decoder->linear_attention.copy_slot_2d(current, staging, device.stream);
    if (dflash) { dflash->staging_local.copy_lane_from(dflash->local, lane, 0, device.stream); }
    CUDA_CHECK(cudaEventRecord(staging_.d2d_done, device.stream));
    CUDA_CHECK(cudaStreamWaitEvent(device.copy_stream, staging_.d2d_done, 0));
    // A staging-miss restore may still be reading the previous image.
    CUDA_CHECK(cudaStreamWaitEvent(device.copy_stream, image.copies_done, 0));
    decoder->linear_attention.pack_slot_to_host(staging, image.conv.data(), image.recurrent.data(),
                                                device.copy_stream);
    if (dflash) {
        dflash->staging_local.copy_lane_to_host(0, image.dflash.data(), device.copy_stream);
    }
    CUDA_CHECK(cudaEventRecord(image.copies_done, device.copy_stream));
    CUDA_CHECK(cudaEventRecord(staging_.copies_done, device.copy_stream));
    staging_.rewrite            = true;
    staging_.rewrite_generation = sequence.rewrite_image_generation;
    staging_.lane               = sequence.lane;
}

void ProgramImplCore::restore_rewrite_checkpoint_state(SequenceState& sequence) {
    ContextCheckpointHead& image = sequence.rewrite_image;
    const std::int32_t current =
        LinearStateSlots::current_state_slot(sequence.lane, max_concurrency);
    const auto lane = static_cast<std::int32_t>(sequence.lane);
    if (staging_.rewrite && staging_.lane == sequence.lane &&
        staging_.rewrite_generation == sequence.rewrite_image_generation) {
        // Staging was written on this stream; the drain only reads it.
        decoder->linear_attention.copy_slot_2d(
            LinearStateSlots::staging_state_slot(max_concurrency), current, device.stream);
        reset_gdn_history(sequence.lane);
        if (dflash) { dflash->local.copy_lane_from(dflash->staging_local, 0, lane, device.stream); }
        return;
    }
    CUDA_CHECK(cudaStreamWaitEvent(device.stream, image.copies_done, 0));
    decoder->linear_attention.unpack_slot_from_host(current, image.conv.data(),
                                                    image.recurrent.data(), device.stream);
    reset_gdn_history(sequence.lane);
    if (dflash) { dflash->local.copy_lane_from_host(image.dflash.data(), lane, device.stream); }
    record_context_checkpoint_head_use(image, device.stream);
}

qwen3_6::detail::RewriteStateHostTarget
ProgramImplCore::rewrite_state_host_target(SequenceState& sequence) {
    // Tier restores overwrite the image by stream-ordered host copies after its fence and
    // re-record the fence after them, so staging no longer matches it.
    ContextCheckpointHead& image = sequence.rewrite_image;
    ++sequence.rewrite_image_generation;
    return qwen3_6::detail::RewriteStateHostTarget{
        .conv        = image.conv.data(),
        .recurrent   = image.recurrent.data(),
        .dflash      = image.dflash ? image.dflash.data() : nullptr,
        .copies_done = image.copies_done,
    };
}

void ProgramImplCore::maybe_capture_turn_rollback(SequenceState& sequence, RequestControl& request,
                                                  const PreparedPromptData& prompt, ReusePath reuse,
                                                  std::uint32_t base, std::uint32_t prompt_tokens,
                                                  bool capture_enabled, bool request_pin,
                                                  bool cut_restore) {
    const bool enabled =
        capture_enabled && captures_context_checkpoints() && staging_hidden.data != nullptr;
    const bool already =
        std::any_of(sequence.context_checkpoints.begin(), sequence.context_checkpoints.end(),
                    [base](const ContextCheckpointHead& head) { return head.frontier == base; });
    const bool complete = qwen3_6::detail::prefix_items_complete_at(prompt.vision_items, base);
    // A preserve-off turn (own TurnClosure checkpoint) appending to a cut tier entry is not
    // pinned automatically: that entry is the previous turn cut at its checkpoint, the restore
    // point a pre-cut entry offered through restore_turn_checkpoint without a pin, and tiers
    // store this turn cut at its own checkpoint without later heads.
    const bool auto_rollback = !(cut_restore && sequence.closure_frontier != 0);
    if (!(auto_rollback && qwen3_6::detail::should_capture_turn_rollback(
                               reuse, base, prompt_tokens, enabled, sequence.tail_hidden_valid,
                               already, complete)) &&
        !qwen3_6::detail::should_capture_exact_hit_pin(request_pin, base, prompt_tokens, enabled,
                                                       sequence.tail_hidden_valid, already,
                                                       complete)) {
        return;
    }
    if (sequence.ledger.size() < base || sequence.tail_hidden.data == nullptr) { return; }

    const qwen3_6::detail::PrefixHash128 hash =
        sequence.prefix_hashes.at(sequence.ledger, sequence.prefix_identity, base);
    auto& heads = sequence.context_checkpoints;
    ContextCheckpointHead head;
    const auto existing_rollback =
        std::find_if(heads.begin(), heads.end(), [](const ContextCheckpointHead& existing) {
            return existing.kind == qwen3_6::detail::ContextCheckpointKind::TurnRollback;
        });
    if (existing_rollback != heads.end()) {
        head = std::move(*existing_rollback);
        heads.erase(existing_rollback);
    } else {
        try {
            heads.reserve(heads.size() + 1);
        } catch (const std::bad_alloc&) { return; }
        std::optional<ContextCheckpointHead> acquired = acquire_context_checkpoint_head();
        if (!acquired) { return; }
        head = std::move(*acquired);
    }
    head.frontier = base;
    head.hash     = hash;
    head.kind     = qwen3_6::detail::ContextCheckpointKind::TurnRollback;
    // Earlier copy-stream reads of staging (a drain or rollback reload) finish before it is
    // overwritten; the compute stream itself wrote d2d_done.
    CUDA_CHECK(cudaStreamWaitEvent(device.stream, staging_.copies_done, 0));
    unoccupy_staging();
    const std::int32_t current =
        LinearStateSlots::current_state_slot(sequence.lane, max_concurrency);
    const std::int32_t staging = LinearStateSlots::staging_state_slot(max_concurrency);
    materialize_gdn_history(sequence.lane);
    decoder->linear_attention.copy_slot_2d(current, staging, device.stream);
    CUDA_CHECK(cudaMemcpyAsync(staging_hidden.data, sequence.tail_hidden.data,
                               staging_hidden.bytes(), cudaMemcpyDeviceToDevice, device.stream));
    snapshot_dflash_cyclic_to_staging(static_cast<std::int32_t>(sequence.lane));
    CUDA_CHECK(cudaEventRecord(staging_.d2d_done, device.stream));
    staging_.occupied = true;
    staging_.lane     = sequence.lane;
    staging_.frontier = base;
    staging_.hash     = hash;
    staging_.kind     = qwen3_6::detail::ContextCheckpointKind::TurnRollback;
    CUDA_CHECK(cudaStreamWaitEvent(device.copy_stream, staging_.d2d_done, 0));
    // The head's previous reader or writer (a tier copy or restore) finishes before the refill.
    CUDA_CHECK(cudaStreamWaitEvent(device.copy_stream, head.copies_done, 0));
    decoder->linear_attention.pack_slot_to_host(staging, head.conv.data(), head.recurrent.data(),
                                                device.copy_stream);
    CUDA_CHECK(cudaMemcpyAsync(head.hidden.data(), staging_hidden.data, staging_hidden.bytes(),
                               cudaMemcpyDeviceToHost, device.copy_stream));
    pack_dflash_cyclic_to_head(head);
    CUDA_CHECK(cudaEventRecord(head.copies_done, device.copy_stream));
    CUDA_CHECK(cudaEventRecord(staging_.copies_done, device.copy_stream));
    heads.push_back(std::move(head));
    request.captured_context_checkpoint_tokens = base;
}

void ProgramImplCore::maybe_freeze_context_checkpoint(SequenceState& sequence,
                                                      RequestControl& request,
                                                      std::uint32_t chunk_tokens,
                                                      bool rewrite_capture_follows) {
    if (!request.prefill) { return; }
    RequestControl::Prefill& staged = *request.prefill;
    const std::uint32_t frontier    = staged.cursor;
    const bool capture_enabled = staged.capture_context_checkpoints &&
                                 captures_context_checkpoints() && staging_hidden.data != nullptr &&
                                 chunk_tokens != 0;
    const bool already         = std::any_of(
        sequence.context_checkpoints.begin(), sequence.context_checkpoints.end(),
        [frontier](const ContextCheckpointHead& head) { return head.frontier == frontier; });
    if (!qwen3_6::detail::should_freeze_prefill_context_checkpoint(
            true, capture_enabled, frontier, sequence.next_context_mark, already,
            qwen3_6::detail::prefix_items_complete_at(staged.prompt.vision_items, frontier))) {
        return;
    }
    if (sequence.ledger.size() < frontier) { return; }

    const qwen3_6::detail::PrefixHash128 hash =
        sequence.prefix_hashes.at(sequence.ledger, sequence.prefix_identity, frontier);
    try {
        sequence.context_checkpoints.reserve(sequence.context_checkpoints.size() + 1);
    } catch (const std::bad_alloc&) { return; }
    std::optional<ContextCheckpointHead> acquired = acquire_context_checkpoint_head();
    if (!acquired) { return; }
    ContextCheckpointHead head = std::move(*acquired);
    head.frontier              = qwen3_6::detail::advertised_context_checkpoint_frontier(frontier);
    head.hash                  = hash;
    head.kind                  = qwen3_6::detail::ContextCheckpointKind::Ladder;

    // Earlier copy-stream reads of staging finish before the compute stream overwrites it.
    CUDA_CHECK(cudaStreamWaitEvent(device.stream, staging_.copies_done, 0));
    // A rewrite capture in the same step overwrites staging next; reloading the rollback
    // would only make the compute stream wait for a copy that is discarded.
    const bool reload_rollback =
        !rewrite_capture_follows && staging_.occupied &&
        staging_.kind == qwen3_6::detail::ContextCheckpointKind::TurnRollback;
    const std::uint32_t saved_lane                  = staging_.lane;
    const std::uint32_t saved_frontier              = staging_.frontier;
    const qwen3_6::detail::PrefixHash128 saved_hash = staging_.hash;
    unoccupy_staging();

    const std::int32_t current =
        LinearStateSlots::current_state_slot(sequence.lane, max_concurrency);
    const std::int32_t staging = LinearStateSlots::staging_state_slot(max_concurrency);
    materialize_gdn_history(sequence.lane);
    decoder->linear_attention.copy_slot_2d(current, staging, device.stream);
    const Tensor last_hidden =
        prefill_hidden.slice(1, static_cast<std::int32_t>(chunk_tokens) - 1, 1);
    CUDA_CHECK(cudaMemcpyAsync(staging_hidden.data, last_hidden.data, staging_hidden.bytes(),
                               cudaMemcpyDeviceToDevice, device.stream));
    snapshot_dflash_cyclic_to_staging(static_cast<std::int32_t>(sequence.lane));
    CUDA_CHECK(cudaEventRecord(staging_.d2d_done, device.stream));
    if (!reload_rollback) {
        staging_.occupied = true;
        staging_.lane     = sequence.lane;
        staging_.frontier = head.frontier;
        staging_.hash     = hash;
        staging_.kind     = qwen3_6::detail::ContextCheckpointKind::Ladder;
    }
    CUDA_CHECK(cudaStreamWaitEvent(device.copy_stream, staging_.d2d_done, 0));
    // The recycled head's previous reader or writer finishes before the refill.
    CUDA_CHECK(cudaStreamWaitEvent(device.copy_stream, head.copies_done, 0));
    decoder->linear_attention.pack_slot_to_host(staging, head.conv.data(), head.recurrent.data(),
                                                device.copy_stream);
    CUDA_CHECK(cudaMemcpyAsync(head.hidden.data(), staging_hidden.data, staging_hidden.bytes(),
                               cudaMemcpyDeviceToHost, device.copy_stream));
    pack_dflash_cyclic_to_head(head);
    CUDA_CHECK(cudaEventRecord(head.copies_done, device.copy_stream));
    CUDA_CHECK(cudaEventRecord(staging_.copies_done, device.copy_stream));
    sequence.context_checkpoints.push_back(std::move(head));
    request.captured_context_checkpoint_tokens = sequence.context_checkpoints.back().frontier;
    sequence.next_context_mark =
        qwen3_6::detail::next_prefill_context_mark(frontier, context_marks).value_or(0);
    if (reload_rollback) {
        reload_turn_rollback_into_staging(saved_lane, saved_hash, saved_frontier);
    }
}

// The entry's rewrite checkpoint is read only when the request keeps it or restores from it.
// Otherwise start_prefill_lane drops it, so a tier restore neither reads its image nor installs
// metadata that would let a cancelled prefill revert to an image it never loaded.
[[nodiscard]] bool tier_restore_keeps_rewrite(const RequestPlanImpl& plan) noexcept {
    return plan.rewrite_checkpoint_action != RewriteCheckpointAction::Drop ||
           is_rewrite_checkpoint_restore(plan.reuse);
}

void ProgramImplCore::restore_ram_entry(std::uint32_t lane, std::uint64_t entry_id,
                                        const RequestPlan& plan) {
    if (lane >= max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    if (!kv_ram_cache_) { throw std::logic_error("RAM restore requires an enabled RAM tier"); }
    qwen3_6::detail::KVRamCache& ram = *kv_ram_cache_;
    if (plan.impl_ == nullptr) { throw std::invalid_argument("request plan is empty"); }
    const RequestPlanImpl& request_plan = *plan.impl_;
    if (request_plan.reuse == ReusePath::FullReset || request_plan.ram_entry_id != entry_id ||
        request_plan.reuse_source != PrefixReuseSource::HostRam || request_plan.reuse_base == 0) {
        throw std::logic_error("RAM restore requires a winning host-RAM reuse plan");
    }
    SequenceState& sequence = sequences[lane];
    RequestControl& request = requests[lane];
    if (request.lifecycle == Lifecycle::Prefilling || request.lifecycle == Lifecycle::Active ||
        request.lifecycle == Lifecycle::Pending) {
        throw std::logic_error("RAM restore requires a free request lane");
    }
    try {
        if (has_retained_lane(lane)) {
            throw std::logic_error(
                "RAM restore requires an empty lane; copy-hold must evict after D2H");
        }
        sequence.kv.reset();
        sequence.retained           = false;
        sequence.turn_closed        = false;
        sequence.closure_frontier   = 0;
        sequence.mtp_draft_count    = 0;
        sequence.rewrite_checkpoint = {};

        device.order_copy_after_compute();
        SequenceKVBundle& kv = reserve_sequence_kv(sequence, request_plan.text_kv_page_entitlement,
                                                   request_plan.backend_kv_page_entitlement);
        const std::uint32_t text_pages = ninfer::pages_for_tokens(request_plan.reuse_base);
        std::uint32_t backend_pages    = 0;
        if (speculative_backend == SpeculativeBackend::Mtp) {
            backend_pages = ninfer::pages_for_tokens(
                request_plan.reuse_base == 0 ? 0U : request_plan.reuse_base - 1U);
        } else if (speculative_backend == SpeculativeBackend::DFlash) {
            backend_pages = ninfer::pages_for_tokens(request_plan.reuse_base);
        }
        kv.text.materialize_pages(text_pages, device.copy_stream);
        if (kv.backend) { kv.backend->materialize_pages(backend_pages, device.copy_stream); }

        qwen3_6::detail::RamRestoreTarget target;
        target.text_dst_pages    = text_pages;
        target.backend_dst_pages = backend_pages;
        target.text              = &kv.text;
        target.text_pool         = &decoder->text_kv.pool();
        if (kv.backend) {
            target.backend      = &*kv.backend;
            target.backend_pool = &backend_kv_cache()->pool();
        }
        target.gdn = &decoder->linear_attention;
        target.gdn_current_slot =
            LinearStateSlots::current_state_slot(sequence.lane, max_concurrency);
        target.rewrite_state             = rewrite_state_host_target(sequence);
        target.tail_hidden               = &sequence.tail_hidden;
        target.rewrite_checkpoint_hidden = &sequence.rewrite_checkpoint_hidden;
        target.reuse                     = request_plan.reuse;
        target.reuse_base                = request_plan.reuse_base;
        target.keep_rewrite_checkpoint   = tier_restore_keeps_rewrite(request_plan);
        if (dflash) {
            target.dflash_local = &dflash->local;
            target.dflash_lane  = static_cast<std::int32_t>(sequence.lane);
        }
        target.stream = device.copy_stream;

        qwen3_6::detail::RamRestoredHost host = ram.unpack_device(entry_id, target);
        reset_gdn_history(sequence.lane);
        sequence.execution_frontier      = host.execution_frontier;
        sequence.ledger_frontier         = host.ledger_frontier;
        sequence.rope_delta              = host.rope_delta;
        sequence.text_kv_valid           = host.text_kv_valid;
        sequence.mtp_kv_valid            = host.mtp_kv_valid;
        sequence.dflash_context_frontier = host.dflash_context_frontier;
        sequence.tail_hidden_valid       = host.tail_hidden_valid;
        sequence.rewrite_checkpoint      = RewriteCheckpoint{
            .valid    = host.rewrite_valid,
            .kind     = host.rewrite_kind,
            .frontier = host.rewrite_frontier,
        };
        if (!target.keep_rewrite_checkpoint) { sequence.rewrite_checkpoint = {}; }
        // The frontier tail hidden is not restored; start_prefill_lane copies the checkpoint's.
        if (is_rewrite_checkpoint_restore(request_plan.reuse)) {
            sequence.tail_hidden_valid = false;
        }
        sequence.prefix_hashes.clear();
        sequence.ledger          = std::move(host.ledger);
        sequence.prefix_identity = std::move(host.identity);
        sequence.disk_entry_id   = host.disk_entry_id;
        sequence.mtp_draft_count = 0;
        sequence.retained        = true;
        install_ram_context_checkpoints(sequence, host, entry_id, request_plan.reuse_base);
        if (qwen3_6::detail::is_staged_checkpoint_restore(request_plan.reuse)) {
            // unpack_device restored the matched head into current with the entry's KV.
            if (!request_plan.prompt_hashes ||
                request_plan.reuse_base >= request_plan.prompt_hashes->size()) {
                throw std::logic_error("RAM staged restore plan has no prompt hash chain");
            }
            sequence.tier_unpacked_context_base = request_plan.reuse_base;
            sequence.tier_unpacked_context_hash =
                (*request_plan.prompt_hashes)[request_plan.reuse_base];
            sequence.tail_hidden_valid = true;
            if (speculative_backend == SpeculativeBackend::DFlash) {
                sequence.dflash_context_frontier = request_plan.reuse_base;
            }
            if (qwen3_6::detail::occupy_drops_rewrite_ahead_of_restore(
                    request_plan.reuse, sequence.rewrite_checkpoint.valid,
                    sequence.rewrite_checkpoint.frontier, request_plan.reuse_base)) {
                sequence.rewrite_checkpoint = {};
            }
        }
    } catch (const std::bad_alloc&) {
        // Partial cache H2D must finish before its destination pages are released.
        // A CUDA failure remains fatal; only optional host metadata is retried.
        device.synchronize_all();
        clear_lane(sequence, request);
        throw runtime::CacheRestoreFailure("RAM cache restore metadata allocation failed");
    } catch (...) {
        synchronize_for_failure_cleanup(device);
        clear_lane(sequence, request);
        throw;
    }
}

void ProgramImplCore::claim_ram_entry(std::uint64_t entry_id) {
    if (!kv_ram_cache_) { throw std::logic_error("RAM claim requires an enabled RAM tier"); }
    // A restore only reads the entry, like a disk spill of it: neither waits for the other,
    // and the spill's I/O pin keeps the block alive past consume. A spill that commits after
    // the restore read the image's disk ticket leaves the lane with the older ticket: an
    // Extend/Refresh keeps that id, and after a Create/Branch the lane's next spill re-plans
    // from the older ticket (extra disk copies until FIFO eviction, never wrong data).
    kv_ram_cache_->claim(entry_id);
}

void ProgramImplCore::release_ram_entry(std::uint64_t entry_id) {
    if (!kv_ram_cache_) { throw std::logic_error("RAM release requires an enabled RAM tier"); }
    kv_ram_cache_->release(entry_id);
}

void ProgramImplCore::consume_ram_entry(std::uint64_t entry_id) {
    if (!kv_ram_cache_) { throw std::logic_error("RAM consume requires an enabled RAM tier"); }
    kv_ram_cache_->consume(entry_id);
    if (kv_disk_cache_) { kv_disk_cache_->forget_ram_resident(entry_id); }
}

qwen3_6::detail::KvRamSnapshot ProgramImplCore::kv_ram_snapshot() const noexcept {
    return kv_ram_cache_ ? kv_ram_cache_->snapshot() : qwen3_6::detail::KvRamSnapshot{};
}

qwen3_6::detail::KvRamCopySeconds ProgramImplCore::harvest_kv_ram_copy_seconds() {
    return kv_ram_cache_ ? kv_ram_cache_->harvest_copy_seconds()
                         : qwen3_6::detail::KvRamCopySeconds{};
}

void disk_reuse_pages(SpeculativeBackend backend, bool growing_backend, std::uint32_t reuse_base,
                      std::uint32_t& text_pages, std::uint32_t& backend_pages) {
    text_pages    = ninfer::pages_for_tokens(reuse_base);
    backend_pages = 0;
    if (!growing_backend) { return; }
    if (backend == SpeculativeBackend::Mtp) {
        backend_pages = ninfer::pages_for_tokens(reuse_base == 0 ? 0U : reuse_base - 1U);
    } else if (backend == SpeculativeBackend::DFlash) {
        backend_pages = ninfer::pages_for_tokens(reuse_base);
    }
}

bool ProgramImplCore::ram_restore_ready(std::uint64_t entry_id) const {
    return !kv_ram_cache_ || kv_ram_cache_->copies_ready(entry_id);
}

bool ProgramImplCore::disk_restore_ready(std::uint64_t entry_id) const {
    return !kv_disk_cache_ || kv_disk_cache_->restore_setup_ready(entry_id);
}

bool ProgramImplCore::kv_ram_reclaim_pending() const {
    return kv_disk_cache_ && kv_disk_cache_->ram_reclaim_pending();
}

void ProgramImplCore::restore_disk_entry(std::uint32_t lane, std::uint64_t entry_id,
                                         const RequestPlan& plan) {
    if (lane >= max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    if (!kv_disk_cache_) { throw std::logic_error("disk restore requires an enabled disk tier"); }
    qwen3_6::detail::KVDiskCache& disk = *kv_disk_cache_;
    if (plan.impl_ == nullptr) { throw std::invalid_argument("request plan is empty"); }
    const RequestPlanImpl& request_plan = *plan.impl_;
    if (request_plan.reuse == ReusePath::FullReset || request_plan.disk_entry_id != entry_id ||
        request_plan.reuse_source != PrefixReuseSource::HostDisk || request_plan.reuse_base == 0) {
        throw std::logic_error("disk restore requires a winning host-disk reuse plan");
    }
    SequenceState& sequence = sequences[lane];
    RequestControl& request = requests[lane];
    if (request.lifecycle == Lifecycle::Prefilling || request.lifecycle == Lifecycle::Active ||
        request.lifecycle == Lifecycle::Pending) {
        throw std::logic_error("disk restore requires a free request lane");
    }
    try {
        if (has_retained_lane(lane)) {
            throw std::logic_error(
                "disk restore requires an empty lane; copy-hold must evict after D2H");
        }
        sequence.kv.reset();
        sequence.retained           = false;
        sequence.turn_closed        = false;
        sequence.closure_frontier   = 0;
        sequence.mtp_draft_count    = 0;
        sequence.rewrite_checkpoint = {};

        device.order_copy_after_compute();
        SequenceKVBundle& kv = reserve_sequence_kv(sequence, request_plan.text_kv_page_entitlement,
                                                   request_plan.backend_kv_page_entitlement);
        std::uint32_t text_pages    = 0;
        std::uint32_t backend_pages = 0;
        disk_reuse_pages(speculative_backend, kv.backend.has_value(), request_plan.reuse_base,
                         text_pages, backend_pages);
        kv.text.materialize_pages(text_pages, device.copy_stream);
        if (kv.backend) { kv.backend->materialize_pages(backend_pages, device.copy_stream); }

        qwen3_6::detail::DiskRestoreTarget target;
        target.text_dst_pages    = text_pages;
        target.backend_dst_pages = backend_pages;
        target.text              = &kv.text;
        target.text_pool         = &decoder->text_kv.pool();
        if (kv.backend) {
            target.backend      = &*kv.backend;
            target.backend_pool = &backend_kv_cache()->pool();
        }
        target.gdn = &decoder->linear_attention;
        target.gdn_current_slot =
            LinearStateSlots::current_state_slot(sequence.lane, max_concurrency);
        target.rewrite_state             = rewrite_state_host_target(sequence);
        target.tail_hidden               = &sequence.tail_hidden;
        target.rewrite_checkpoint_hidden = &sequence.rewrite_checkpoint_hidden;
        target.reuse                     = request_plan.reuse;
        target.reuse_base                = request_plan.reuse_base;
        target.keep_rewrite_checkpoint   = tier_restore_keeps_rewrite(request_plan);
        if (dflash) {
            target.dflash_local = &dflash->local;
            target.dflash_lane  = static_cast<std::int32_t>(sequence.lane);
        }
        target.stream               = device.copy_stream;
        target.committed_generation = request_plan.disk_committed_generation;

        std::optional<qwen3_6::detail::DiskRestoredHost> loaded =
            disk.load_host(entry_id, request_plan.disk_committed_generation);
        if (!loaded) { throw std::runtime_error("KV disk restore lost its claimed entry"); }
        qwen3_6::detail::DiskRestoredHost host = std::move(*loaded);
        // The restore workers decode the entry's checkpoint heads straight into pool heads.
        // Occupy keeps only heads at or before the base, so later ones are never decoded; the
        // planned head goes first so pool exhaustion cannot drop it.
        // Heads stay pending when an earlier cancellation failed before it could prove its
        // workers drained; drain again before they return to the pool.
        if (!pending_disk_checkpoint_heads_.empty()) {
            disk.cancel_restore();
            recycle_pending_disk_checkpoint_heads();
        }
        pending_disk_checkpoint_heads_.reserve(host.ladders.size());
        target.checkpoint_images.reserve(host.ladders.size());
        std::vector<qwen3_6::detail::RamLadderIndex> wanted;
        wanted.reserve(host.ladders.size());
        for (const qwen3_6::detail::RamLadderIndex& ladder : host.ladders) {
            if (ladder.frontier == request_plan.reuse_base) { wanted.push_back(ladder); }
        }
        for (const qwen3_6::detail::RamLadderIndex& ladder : host.ladders) {
            if (ladder.frontier < request_plan.reuse_base) { wanted.push_back(ladder); }
        }
        for (const qwen3_6::detail::RamLadderIndex& ladder : wanted) {
            std::optional<ContextCheckpointHead> head = acquire_context_checkpoint_head();
            if (!head) { break; }
            head->frontier = ladder.frontier;
            head->hash     = ladder.hash;
            head->kind     = ladder.kind;
            target.checkpoint_images.push_back(qwen3_6::detail::DiskCheckpointImageTarget{
                .frontier        = head->frontier,
                .hash            = head->hash,
                .kind            = head->kind,
                .conv            = head->conv.data(),
                .conv_bytes      = head->conv.size(),
                .recurrent       = head->recurrent.data(),
                .recurrent_bytes = head->recurrent.size(),
                .hidden          = head->hidden.data(),
                .hidden_bytes    = head->hidden.size(),
                .dflash          = head->dflash.data(),
                .dflash_bytes    = head->dflash.size(),
                .writable        = head->copies_done,
            });
            pending_disk_checkpoint_heads_.push_back(std::move(*head));
        }
        pending_disk_restore_ticket_ = disk.restore_device(entry_id, target);
        reset_gdn_history(sequence.lane);
        pending_disk_checkpoint_lane_    = lane;
        sequence.execution_frontier      = host.execution_frontier;
        sequence.ledger_frontier         = host.ledger_frontier;
        sequence.rope_delta              = host.rope_delta;
        sequence.text_kv_valid           = host.text_kv_valid;
        sequence.mtp_kv_valid            = host.mtp_kv_valid;
        sequence.dflash_context_frontier = host.dflash_context_frontier;
        sequence.tail_hidden_valid       = host.tail_hidden_valid;
        sequence.rewrite_checkpoint      = RewriteCheckpoint{
            .valid    = host.rewrite_valid,
            .kind     = host.rewrite_kind,
            .frontier = host.rewrite_frontier,
        };
        if (!target.keep_rewrite_checkpoint) { sequence.rewrite_checkpoint = {}; }
        // The frontier tail hidden is not restored; start_prefill_lane copies the checkpoint's.
        if (is_rewrite_checkpoint_restore(request_plan.reuse)) {
            sequence.tail_hidden_valid = false;
        }
        sequence.prefix_hashes.clear();
        sequence.ledger          = std::move(host.ledger);
        sequence.prefix_identity = std::move(host.identity);
        sequence.disk_entry_id   = entry_id;
        sequence.mtp_draft_count = 0;
        sequence.retained        = true;
        if (qwen3_6::detail::is_staged_checkpoint_restore(request_plan.reuse)) {
            // The restore unpacks the planned head into current with the entry's KV.
            sequence.tier_unpacked_context_base = request_plan.reuse_base;
            // The plan selected this head by its hash in the prompt's chain.
            if (!request_plan.prompt_hashes ||
                request_plan.reuse_base >= request_plan.prompt_hashes->size()) {
                throw std::logic_error("disk staged restore plan has no prompt hash chain");
            }
            sequence.tier_unpacked_context_hash =
                (*request_plan.prompt_hashes)[request_plan.reuse_base];
            sequence.tail_hidden_valid = true;
            if (speculative_backend == SpeculativeBackend::DFlash) {
                sequence.dflash_context_frontier = request_plan.reuse_base;
            }
            if (qwen3_6::detail::occupy_drops_rewrite_ahead_of_restore(
                    request_plan.reuse, sequence.rewrite_checkpoint.valid,
                    sequence.rewrite_checkpoint.frontier, request_plan.reuse_base)) {
                sequence.rewrite_checkpoint = {};
            }
        }
    } catch (const std::bad_alloc&) {
        pending_disk_checkpoint_lane_.reset();
        // restore_device drains any unreturned ticket on setup failure. A live
        // returned ticket and prefetched readers still belong to this Program.
        cancel_disk_restore();
        device.synchronize_all();
        clear_lane(sequence, request);
        throw runtime::CacheRestoreFailure("disk cache restore metadata allocation failed");
    } catch (...) {
        pending_disk_checkpoint_lane_.reset();
        synchronize_for_failure_cleanup(device);
        if (kv_disk_cache_) {
            try {
                kv_disk_cache_->cancel_restore();
                // No restore worker writes the heads once cancellation has drained them. A
                // failed cancellation leaves them pending for the next restore to drain.
                recycle_pending_disk_checkpoint_heads();
            } catch (...) { // NOLINT(bugprone-empty-catch): the restore's own failure is rethrown.
            }
        }
        clear_lane(sequence, request);
        throw;
    }
}

bool ProgramImplCore::claim_disk_entry(std::uint64_t entry_id, std::uint32_t expected_frontier,
                                       std::uint64_t hash_lo, std::uint64_t hash_hi,
                                       std::uint32_t expected_reuse_base,
                                       PrefixReusePath expected_reuse,
                                       std::uint64_t expected_committed_generation) {
    if (!kv_disk_cache_) { throw std::logic_error("disk claim requires an enabled disk tier"); }
    qwen3_6::detail::PrefixHash128 hash;
    hash.lo = hash_lo;
    hash.hi = hash_hi;
    return kv_disk_cache_->claim(entry_id, hash, expected_frontier, expected_reuse_base,
                                 expected_reuse, expected_committed_generation);
}

void ProgramImplCore::release_disk_entry(std::uint64_t entry_id) {
    if (!kv_disk_cache_) { throw std::logic_error("disk release requires an enabled disk tier"); }
    kv_disk_cache_->release(entry_id);
}

void ProgramImplCore::invalidate_disk_entry(std::uint64_t entry_id) {
    // A spill built on the entry sees the quarantine and abandons its own commit.
    if (kv_disk_cache_) { kv_disk_cache_->invalidate_entry(entry_id); }
}

void ProgramImplCore::consume_disk_entry(std::uint64_t entry_id) {
    if (!kv_disk_cache_) { throw std::logic_error("disk consume requires an enabled disk tier"); }
    kv_disk_cache_->consume(entry_id);
}

void ProgramImplCore::prefetch_disk_window(std::uint64_t entry_id, std::uint32_t text_pages,
                                           std::uint32_t backend_pages) {
    if (!kv_disk_cache_) { return; }
    kv_disk_cache_->prefetch_window(entry_id, text_pages, backend_pages);
}

void ProgramImplCore::prefetch_disk_plan(std::uint64_t entry_id, const RequestPlan& plan) {
    if (!kv_disk_cache_ || plan.impl_ == nullptr) { return; }
    std::uint32_t text_pages    = 0;
    std::uint32_t backend_pages = 0;
    disk_reuse_pages(speculative_backend, backend_kv_cache() != nullptr, plan.impl_->reuse_base,
                     text_pages, backend_pages);
    if (text_pages > 2) {
        text_pages    = 2;
        backend_pages = 0;
    } else if (text_pages + backend_pages > 2) {
        backend_pages = 2 - text_pages;
    }
    prefetch_disk_window(entry_id, text_pages, backend_pages);
}

void ProgramImplCore::pump_disk_restore() {
    try {
        if (kv_disk_cache_) { kv_disk_cache_->pump_restore(device.copy_stream); }
    } catch (const std::bad_alloc&) {
        cancel_disk_restore();
        device.synchronize_all();
        throw runtime::CacheRestoreFailure("disk cache copy metadata allocation failed");
    }
}

void ProgramImplCore::begin_copy_hold_cancel() {
    if (kv_disk_cache_) { kv_disk_cache_->begin_cancel_restore(); }
    CUDA_CHECK(cudaEventRecord(copy_hold_cancel_fence_, device.copy_stream));
}

bool ProgramImplCore::copy_hold_cancel_settled() const {
    const cudaError_t ready = cudaEventQuery(copy_hold_cancel_fence_);
    if (ready == cudaErrorNotReady) { return false; }
    CUDA_CHECK(ready);
    return !kv_disk_cache_ || kv_disk_cache_->restore_cancel_settled();
}

void ProgramImplCore::cancel_disk_restore() {
    if (!kv_disk_cache_) { return; }
    qwen3_6::detail::KVDiskCache& disk = *kv_disk_cache_;
    const std::uint64_t ticket         = pending_disk_restore_ticket_;
    disk.cancel_restore();
    // No restore worker writes the heads once cancellation has drained them.
    recycle_pending_disk_checkpoint_heads();
    if (ticket != 0) { disk.release_restore_ticket(ticket); }
    pending_disk_restore_ticket_ = 0;
    pending_disk_checkpoint_lane_.reset();
}

void ProgramImplCore::discard_ram_capture(std::uint64_t ram_id) {
    if (ram_id == 0 || !kv_ram_cache_) { return; }
    // Neither call waits: a spill of the capture is cancelled at its next cancellation point,
    // and the block is freed once its D2H fence and the spill's I/O pin are gone.
    if (kv_disk_cache_) { kv_disk_cache_->abandon_ram_spill(ram_id); }
    kv_ram_cache_->discard(ram_id);
}

void ProgramImplCore::shutdown_kv_tiers(LoadProgress progress) {
    if (kv_tiers_shutdown_) { return; }
    kv_tiers_shutdown_ = true;
    auto report        = [&](std::string_view phase, std::uint64_t done, std::uint64_t total) {
        if (!progress.callback) { return; }
        try {
            progress.callback(phase, done, total);
            // NOLINTNEXTLINE(bugprone-empty-catch): a progress observer must not abort shutdown.
        } catch (...) {}
    };
    const bool report_disk = kv_disk_cache_.has_value() && static_cast<bool>(progress.callback);
    if (kv_disk_cache_) {
        kv_disk_cache_->cancel_restore();
        recycle_pending_disk_checkpoint_heads();
    }
    if (kv_disk_cache_) { kv_disk_cache_->cancel_idle_spill(); }
    std::array<std::uint32_t, kMaximumConcurrency> failed_lanes{};
    std::size_t failed_lane_count = 0;
    std::uint64_t retained        = 0;
    for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
        if (has_retained_lane(lane)) { ++retained; }
    }
    if (report_disk) { report("kv-disk copy active chats", 0, retained); }
    std::uint64_t captured = 0;
    for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
        if (!has_retained_lane(lane)) { continue; }
        if (!capture_retained_lane(lane)) { failed_lanes[failed_lane_count++] = lane; }
        ++captured;
        if (report_disk) { report("kv-disk copy active chats", captured, retained); }
    }
    wait_kv_ram_copies();
    auto spill_progress = report_disk ? std::function<void(std::uint64_t, std::uint64_t)>(
                                            [&](std::uint64_t done, std::uint64_t total) {
                                                report("kv-disk save cache entries", done, total);
                                            })
                                      : std::function<void(std::uint64_t, std::uint64_t)>{};
    if (kv_disk_cache_) { kv_disk_cache_->flush_not_durable_ram(spill_progress); }
    for (std::uint32_t lane : std::span(failed_lanes).first(failed_lane_count)) {
        if (has_retained_lane(lane)) { (void)capture_retained_lane(lane); }
    }
    wait_kv_ram_copies();
    if (kv_disk_cache_) {
        kv_disk_cache_->flush_not_durable_ram(spill_progress);
        if (report_disk) { report("kv-disk finish disk writes", 0, 1); }
        kv_disk_cache_->wait_idle_and_fsync();
        if (report_disk) { report("kv-disk finish disk writes", 1, 1); }
    }
}

void ProgramImplCore::request_idle_spill() {
    if (kv_disk_cache_) { kv_disk_cache_->request_idle_spill(); }
}

std::optional<qwen3_6::detail::KvDiskSnapshot>
ProgramImplCore::try_kv_disk_snapshot() const noexcept {
    return kv_disk_cache_ ? kv_disk_cache_->try_snapshot()
                          : std::optional{qwen3_6::detail::KvDiskSnapshot{}};
}

qwen3_6::detail::KvDiskCopySeconds ProgramImplCore::harvest_kv_disk_copy_seconds() {
    return kv_disk_cache_ ? kv_disk_cache_->harvest_copy_seconds()
                          : qwen3_6::detail::KvDiskCopySeconds{};
}

qwen3_6::detail::KvGpuSnapshot ProgramImplCore::kv_gpu_snapshot() const noexcept {
    qwen3_6::detail::KvGpuSnapshot out;
    if (decoder) {
        const PagedKVPool& pool   = decoder->text_kv.pool();
        out.main.page_group_count = pool.page_group_count();
        out.main.entitled_pages   = pool.entitled_pages();
        out.main.mapped_pages     = pool.mapped_pages();
        out.main.free_pages       = pool.free_pages();
    }
    if (const qwen3_6::PagedKVCache* backend = backend_kv_cache()) {
        const PagedKVPool& pool   = backend->pool();
        out.spec.page_group_count = pool.page_group_count();
        out.spec.entitled_pages   = pool.entitled_pages();
        out.spec.mapped_pages     = pool.mapped_pages();
        out.spec.free_pages       = pool.free_pages();
    }
    return out;
}

bool ProgramImplCore::kv_ram_copies_ready() const {
    return !kv_ram_cache_ || kv_ram_cache_->pending_copies_ready();
}

bool ProgramImplCore::kv_disk_copies_ready() const {
    return !kv_disk_cache_ || kv_disk_cache_->copies_ready();
}

bool ProgramImplCore::kv_disk_restore_failed() const {
    return kv_disk_cache_ && kv_disk_cache_->restore_failed();
}

bool ProgramImplCore::kv_copies_ready() const {
    return kv_ram_copies_ready() && kv_disk_copies_ready();
}

void ProgramImplCore::wait_kv_ram_copies_on_compute() {
    if (kv_ram_cache_) { kv_ram_cache_->wait_pending_copies_on_stream(device.stream); }
    if (kv_disk_cache_) {
        kv_disk_cache_->wait_copies_on_stream(device.stream, pending_disk_restore_ticket_);
        if (pending_disk_restore_ticket_ != 0) {
            kv_disk_cache_->release_restore_ticket(pending_disk_restore_ticket_);
            pending_disk_restore_ticket_ = 0;
        }
    }
}

void ProgramImplCore::wait_kv_ram_copies() {
    if (kv_ram_cache_) { kv_ram_cache_->wait_pending_copies(); }
}

void ProgramImplCore::wait_kv_disk_copies() {
    try {
        if (kv_disk_cache_) { kv_disk_cache_->wait_copies(pending_disk_restore_ticket_); }
    } catch (const std::bad_alloc&) {
        cancel_disk_restore();
        device.synchronize_all();
        throw runtime::CacheRestoreFailure("disk cache copy metadata allocation failed");
    } catch (...) {
        pending_disk_checkpoint_lane_.reset();
        if (pending_disk_restore_ticket_ != 0 && kv_disk_cache_) {
            kv_disk_cache_->release_restore_ticket(pending_disk_restore_ticket_);
            pending_disk_restore_ticket_ = 0;
        }
        throw;
    }
    install_pending_disk_restore_checkpoints();
}

void ProgramImplCore::install_pending_disk_restore_checkpoints() {
    if (!pending_disk_checkpoint_lane_ || !kv_disk_cache_) { return; }
    const std::uint32_t lane = *pending_disk_checkpoint_lane_;
    pending_disk_checkpoint_lane_.reset();
    // A loaded restore decoded every requested head and its workers no longer write them.
    // Otherwise the restore never loaded (stopping or cancelled): drain any worker first.
    if (!kv_disk_cache_->take_restore_checkpoints()) {
        kv_disk_cache_->cancel_restore();
        recycle_pending_disk_checkpoint_heads();
        return;
    }
    SequenceState& sequence = sequences[lane];
    clear_context_checkpoints(sequence);
    sequence.context_checkpoints = std::move(pending_disk_checkpoint_heads_);
    pending_disk_checkpoint_heads_.clear();
}

std::uint64_t ProgramImplCore::pending_disk_restore_ticket() const noexcept {
    return pending_disk_restore_ticket_;
}

void ProgramImplCore::synchronize_all() { device.synchronize_all(); }

std::uint64_t ProgramImplCore::kv_ram_index_version() const noexcept {
    return kv_ram_cache_ ? kv_ram_cache_->index_version() : 0;
}

std::uint64_t ProgramImplCore::kv_disk_index_version() const noexcept {
    return kv_disk_cache_ ? kv_disk_cache_->index_version() : 0;
}

GenerationTimings ProgramImplCore::generation_timings_lane(std::uint32_t lane) const noexcept {
    return lane < max_concurrency ? requests[lane].timings : GenerationTimings{};
}

SpeculativeStats ProgramImplCore::speculative_stats_lane(std::uint32_t lane) const noexcept {
    return lane < max_concurrency ? requests[lane].speculative_stats : SpeculativeStats{};
}

std::uint32_t
ProgramImplCore::captured_context_checkpoint_tokens_lane(std::uint32_t lane) const noexcept {
    return lane < max_concurrency ? requests[lane].captured_context_checkpoint_tokens : 0;
}

std::uint32_t
ProgramImplCore::restored_context_checkpoint_tokens_lane(std::uint32_t lane) const noexcept {
    return lane < max_concurrency ? requests[lane].restored_context_checkpoint_tokens : 0;
}

void ProgramImplCore::clear_lane(SequenceState& sequence, RequestControl& request) noexcept {
    gdn_history_lengths[sequence.lane] = 0;
    request.output                     = nullptr;
    if (staging_.occupied && staging_.lane == sequence.lane) { unoccupy_staging(); }
    request.prefill.reset();
    sequence.kv.reset();
    request.lifecycle           = Lifecycle::Empty;
    sequence.execution_frontier = 0;
    sequence.ledger_frontier    = 0;
    sequence.ledger.clear();
    sequence.prefix_identity.clear();
    sequence.prefix_hashes.clear();
    sequence.text_kv_valid              = 0;
    sequence.mtp_kv_valid               = 0;
    sequence.dflash_context_frontier    = 0;
    sequence.mtp_draft_count            = 0;
    sequence.tail_hidden_valid          = false;
    sequence.retained                   = false;
    sequence.turn_closed                = false;
    sequence.closure_frontier           = 0;
    sequence.use_tick                   = 0;
    sequence.disk_entry_id              = 0;
    sequence.rewrite_checkpoint         = {};
    sequence.tier_unpacked_context_base = 0;
    sequence.tier_unpacked_context_hash = {};
    clear_context_checkpoints(sequence);
    request.pending                 = {};
    request.adaptive                = {};
    request.typical_cycle_reasoning = false;
    request.prompt_tokens           = 0;
}

qwen3_6::PagedKVCache* ProgramImplCore::backend_kv_cache() noexcept {
    if (speculative_backend == SpeculativeBackend::Mtp) { return decoder->mtp_cache(); }
    if (speculative_backend == SpeculativeBackend::DFlash && dflash) {
        if constexpr (DFlashConfig::full_layers > 0) {
            return dflash->full ? &*dflash->full : nullptr;
        }
        return nullptr;
    }
    return nullptr;
}

const qwen3_6::PagedKVCache* ProgramImplCore::backend_kv_cache() const noexcept {
    if (speculative_backend == SpeculativeBackend::Mtp) { return decoder->mtp_cache(); }
    if (speculative_backend == SpeculativeBackend::DFlash && dflash) {
        if constexpr (DFlashConfig::full_layers > 0) {
            return dflash->full ? &*dflash->full : nullptr;
        }
        return nullptr;
    }
    return nullptr;
}

std::uint32_t ProgramImplCore::backend_kv_valid(const SequenceState& sequence) const noexcept {
    if (speculative_backend == SpeculativeBackend::Mtp) { return sequence.mtp_kv_valid; }
    if (speculative_backend == SpeculativeBackend::DFlash) {
        if constexpr (DFlashConfig::full_layers > 0) { return sequence.dflash_context_frontier; }
        return 0;
    }
    return 0;
}

SequenceKVBundle& ProgramImplCore::reserve_sequence_kv(SequenceState& sequence,
                                                       std::uint32_t text_pages,
                                                       std::uint32_t backend_pages) {
    if (sequence.kv) { throw std::logic_error("sequence already owns a KV allocation bundle"); }
    if (text_pages == 0 || (backend_kv_cache() == nullptr) != (backend_pages == 0)) {
        throw std::invalid_argument("KV allocation entitlement does not match the active backend");
    }

    std::array<PagedKVReservation, 2> reservations{};
    std::size_t count     = 0;
    reservations[count++] = PagedKVReservation{
        .pool             = &decoder->text_kv.pool(),
        .page_entitlement = text_pages,
    };
    if (qwen3_6::PagedKVCache* backend = backend_kv_cache(); backend != nullptr) {
        reservations[count++] = PagedKVReservation{
            .pool             = &backend->pool(),
            .page_entitlement = backend_pages,
        };
    }

    std::vector<PagedKVAllocation> allocations =
        reserve_paged_kv_bundle(std::span<const PagedKVReservation>(reservations.data(), count));
    SequenceKVBundle bundle;
    bundle.text = std::move(allocations[0]);
    if (count == 2) { bundle.backend.emplace(std::move(allocations[1])); }
    return sequence.kv.emplace(std::move(bundle));
}

void ProgramImplCore::resize_sequence_kv_entitlement(SequenceState& sequence,
                                                     std::uint32_t text_pages,
                                                     std::uint32_t backend_pages) {
    if (!sequence.kv || text_pages == 0 ||
        (sequence.kv->backend.has_value() != (backend_pages != 0))) {
        throw std::invalid_argument("KV resize entitlement does not match the sequence bundle");
    }
    std::array<PagedKVResize, 2> changes{};
    std::size_t count = 0;
    changes[count++]  = PagedKVResize{
        .allocation       = &sequence.kv->text,
        .mapped_pages     = sequence.kv->text.mapped_page_count(),
        .page_entitlement = text_pages,
    };
    if (sequence.kv->backend) {
        changes[count++] = PagedKVResize{
            .allocation       = &*sequence.kv->backend,
            .mapped_pages     = sequence.kv->backend->mapped_page_count(),
            .page_entitlement = backend_pages,
        };
    }
    resize_paged_kv_bundle(std::span<PagedKVResize>(changes.data(), count));
}

void ProgramImplCore::bind_sequence_kv(SequenceState& sequence) {
    if (!sequence.kv || sequence.kv->text.bound_row() >= 0 ||
        (sequence.kv->backend && sequence.kv->backend->bound_row() >= 0)) {
        throw std::logic_error("KV allocation bundle is unavailable or already bound");
    }
    const std::int32_t row = static_cast<std::int32_t>(sequence.lane);
    sequence.kv->text.bind_row(row, device.stream);
    try {
        if (sequence.kv->backend) { sequence.kv->backend->bind_row(row, device.stream); }
        set_device_i32(io.text_kv_table_row, sequence.kv->text.bound_row());
        set_device_i32(io.backend_kv_table_row,
                       sequence.kv->backend ? sequence.kv->backend->bound_row() : 0);
    } catch (...) {
        if (sequence.kv->backend && sequence.kv->backend->bound_row() >= 0) {
            sequence.kv->backend->unbind_row();
        }
        sequence.kv->text.unbind_row();
        throw;
    }
}

void ProgramImplCore::unbind_sequence_kv(SequenceState& sequence) noexcept {
    if (!sequence.kv) { return; }
    if (sequence.kv->backend) { sequence.kv->backend->unbind_row(); }
    sequence.kv->text.unbind_row();
}

void ProgramImplCore::materialize_sequence_kv(SequenceState& sequence, std::uint32_t main_tokens,
                                              std::uint32_t backend_tokens) {
    if (!sequence.kv || main_tokens > capacity || backend_tokens > capacity) {
        throw std::logic_error("KV materialization request is outside the sequence bundle");
    }
    SequenceKVBundle& kv = *sequence.kv;
    auto* const backend  = kv.backend ? &*kv.backend : nullptr;
    if (backend_tokens != 0 && backend == nullptr) {
        throw std::logic_error("backend KV materialization requested without an allocation");
    }
    if (main_tokens > kv.text.mapped_token_capacity()) {
        kv.text.materialize_tokens(main_tokens, device.stream);
    }
    if (backend_tokens != 0 && backend_tokens > backend->mapped_token_capacity()) {
        backend->materialize_tokens(backend_tokens, device.stream);
    }
}

void ProgramImplCore::trim_sequence_kv(SequenceState& sequence, std::uint32_t main_tokens,
                                       std::uint32_t backend_tokens) {
    if (!sequence.kv || main_tokens > capacity || backend_tokens > main_tokens) {
        throw std::logic_error("KV trim request is outside the sequence bundle");
    }
    if (backend_tokens != 0 && !sequence.kv->backend) {
        throw std::logic_error("backend KV trim requested without an allocation");
    }
    sequence.kv->text.trim_tokens(main_tokens);
    if (sequence.kv->backend) { sequence.kv->backend->trim_tokens(backend_tokens); }
}

void ProgramImplCore::release_sequence_growth_entitlement(SequenceState& sequence) noexcept {
    if (!sequence.kv) { return; }
    sequence.kv->text.cancel_unmapped_entitlement();
    if (sequence.kv->backend) { sequence.kv->backend->cancel_unmapped_entitlement(); }
}

qwen3_6::PagedKVCacheView ProgramImplCore::text_kv_view(const SequenceState& sequence) const {
    if (!sequence.kv) { throw std::logic_error("sequence has no KV allocation bundle"); }
    return decoder->text_kv.execution_view(sequence.kv->text);
}

qwen3_6::PagedKVCacheView ProgramImplCore::mtp_kv_view(const SequenceState& sequence) const {
    if (speculative_backend != SpeculativeBackend::Mtp) { return {}; }
    if (decoder->mtp_cache() == nullptr || !sequence.kv || !sequence.kv->backend) {
        throw std::logic_error("sequence has no MTP KV allocation");
    }
    return decoder->mtp_cache()->execution_view(*sequence.kv->backend);
}

void ProgramImplCore::set_device_i32(Tensor& tensor, std::int32_t value) {
    CUDA_CHECK(
        cudaMemcpyAsync(tensor.data, &value, sizeof(value), cudaMemcpyHostToDevice, device.stream));
}

void ProgramImplCore::materialize_gdn_history(std::uint32_t lane) {
    if (!gdn_history || gdn_history_lengths[lane] == 0) { return; }
    if (!replay_records) {
        throw std::logic_error("GDN history materialization requires bound replay records");
    }
    const std::int32_t slot = LinearStateSlots::current_state_slot(lane, max_concurrency);
    ops::gdn_history_materialize(*replay_records, *gdn_history,
                                 decoder->linear_attention.all_layers_view(),
                                 std::span<const std::int32_t>(&slot, 1), device.stream);
    gdn_history_lengths[lane] = 0;
}

void ProgramImplCore::reset_gdn_history(std::uint32_t lane) {
    if (!gdn_history) { return; }
    ops::gdn_history_reset_slot(
        *gdn_history, LinearStateSlots::current_state_slot(lane, max_concurrency), device.stream);
    gdn_history_lengths[lane] = 0;
}

void ProgramImplCore::ordered_reset(SequenceState& sequence) {
    decoder->linear_attention.zero_slot(
        LinearStateSlots::current_state_slot(sequence.lane, max_concurrency), device.stream);
    reset_gdn_history(sequence.lane);
    work.reset();
    set_device_i32(io.pos, 0);
    set_device_i32(io.rope_pos, 0);
    set_device_i32(io.rope_delta, 0);
    if (io.mtp) { set_device_i32(io.mtp->position, 0); }
    sequence.text_kv_valid           = 0;
    sequence.mtp_kv_valid            = 0;
    sequence.dflash_context_frontier = 0;
}

void ProgramImplCore::prepare_graphs() {
    if (!use_cuda_graph) { return; }
    SequenceState& sequence = sequences[0];

    std::vector<PagedKVAllocation> text_capture_allocations;
    std::vector<PagedKVAllocation> mtp_capture_allocations;
    std::vector<PagedKVAllocation> dflash_capture_allocations;
    const auto reserve_capture_rows = [&](qwen3_6::PagedKVCache& cache,
                                          std::vector<PagedKVAllocation>& allocations,
                                          const char* label) {
        PagedKVPool& pool = cache.pool();
        if (pool.page_group_count() < max_concurrency) {
            throw std::invalid_argument(std::string(label) +
                                        " cannot provide one Paged KV page per concurrent request");
        }
        allocations.reserve(max_concurrency);
        for (std::uint32_t row = 0; row < max_concurrency; ++row) {
            allocations.push_back(pool.reserve(1));
            PagedKVAllocation& allocation = allocations.back();
            allocation.bind_row(static_cast<std::int32_t>(row), device.stream);
            allocation.materialize_pages(1, device.stream);

            // Capture profiles exercise arbitrary context envelopes. Repeating each row's private
            // page across its temporary table keeps every dummy read/write address valid without
            // reserving C full contexts solely for graph construction.
            const std::int32_t page = allocation.page_ids().front();
            std::vector<std::int32_t> repeated(pool.logical_page_capacity(), page);
            Tensor table = pool.block_table_row(static_cast<std::int32_t>(row));
            CUDA_CHECK(cudaMemcpyAsync(table.data, repeated.data(), table.bytes(),
                                       cudaMemcpyHostToDevice, device.stream));
        }
    };
    reserve_capture_rows(decoder->text_kv, text_capture_allocations, "target KV cache");
    if (speculative_backend == SpeculativeBackend::Mtp) {
        reserve_capture_rows(*decoder->mtp_cache(), mtp_capture_allocations, "MTP KV cache");
    } else if (speculative_backend == SpeculativeBackend::DFlash) {
        if constexpr (DFlashConfig::full_layers > 0) {
            reserve_capture_rows(dflash.value().full.value(), dflash_capture_allocations,
                                 "DFlash Full KV cache");
        }
    }
    device.synchronize();

    std::size_t free_before = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_before, &total_bytes));

    const auto clear_stable_controls = [&] {
        std::vector<Tensor> controls{
            io.token,
            io.pos,
            io.rope_pos,
            io.rope_delta,
        };
        if (io.mtp) {
            controls.push_back(io.mtp->position);
            controls.push_back(io.mtp->draft_tokens);
            controls.push_back(io.mtp->target_input_ids);
            controls.push_back(io.mtp->target_positions);
        }
        if (io.dflash_prefill) { controls.push_back(io.dflash_prefill->produced_count); }
        for (const Tensor& tensor : controls) {
            CUDA_CHECK(cudaMemsetAsync(tensor.data, 0, tensor.bytes(), device.stream));
        }
    };
    const auto zero_capture_pages = [&](qwen3_6::PagedKVCache& cache,
                                        const std::vector<PagedKVAllocation>& allocations,
                                        std::uint32_t batch_size) {
        std::vector<std::int32_t> pages;
        pages.reserve(batch_size);
        for (std::uint32_t row = 0; row < batch_size; ++row) {
            pages.push_back(allocations[row].page_ids().front());
        }
        cache.pool().zero_pages(pages, device.stream);
    };
    const auto zero_cyclic_lane = [&](CyclicKVCache& cache, std::uint32_t lane) {
        for (std::uint32_t layer = 0; layer < cache.layer_count(); ++layer) {
            const CyclicKVCacheLayerView view = cache.layer_view(layer);
            const Tensor k                    = view.k.slice(3, static_cast<std::int32_t>(lane), 1);
            const Tensor v                    = view.v.slice(3, static_cast<std::int32_t>(lane), 1);
            CUDA_CHECK(cudaMemsetAsync(k.data, 0, k.bytes(), device.stream));
            CUDA_CHECK(cudaMemsetAsync(v.data, 0, v.bytes(), device.stream));
        }
    };

    // `k` is the draft length the graph was captured for: an adaptive family holds graphs for
    // every captured k, and a representative sized for the configured window overruns a
    // smaller-k graph's verify rows.
    const auto prepare_representative = [&](std::uint32_t frontier, std::uint32_t batch_size,
                                            std::uint32_t k) {
        if (batch_size == 0 || batch_size > max_concurrency) {
            throw std::logic_error("CUDA Graph representative batch is invalid");
        }
        work.reset();
        clear_stable_controls();
        zero_capture_pages(decoder->text_kv, text_capture_allocations, batch_size);
        if (decoder->mtp_cache() != nullptr) {
            zero_capture_pages(*decoder->mtp_cache(), mtp_capture_allocations, batch_size);
        }
        if (dflash && dflash->full) {
            zero_capture_pages(*dflash->full, dflash_capture_allocations, batch_size);
        }
        for (std::uint32_t row = 0; row < batch_size; ++row) {
            decoder->linear_attention.zero_slot(
                LinearStateSlots::current_state_slot(row, max_concurrency), device.stream);
            reset_gdn_history(row);
            if (dflash) {
                zero_cyclic_lane(dflash->local, row);
                const Tensor pending =
                    dflash->pending_features.slice(2, static_cast<std::int32_t>(row), 1);
                CUDA_CHECK(cudaMemsetAsync(pending.data, 0, pending.bytes(), device.stream));
            }
        }
        set_device_i32(io.pos, checked_i32(frontier, "graph representative position"));
        set_device_i32(io.rope_pos, checked_i32(frontier, "graph representative rope position"));
        if (io.mtp) {
            set_device_i32(io.mtp->position,
                           checked_i32(frontier, "graph representative MTP position"));
        }
        if (io.dflash_decode) {
            *dflash_host_ingress       = {};
            *dflash_host_egress        = {};
            const std::uint32_t extent = std::min(k, capacity - frontier - 1U);
            for (std::uint32_t row = 0; row < batch_size; ++row) {
                dflash_host_ingress->anchors[row] = 0;
                dflash_host_ingress->execution_frontiers[row] =
                    checked_i32(frontier, "graph representative DFlash frontier");
                dflash_host_ingress->context_frontiers[row] =
                    checked_i32(frontier, "graph representative DFlash context frontier");
                dflash_host_ingress->proposal_extents[row] = static_cast<std::int32_t>(extent);
                dflash_host_ingress->target_valid_columns[row] =
                    static_cast<std::int32_t>(extent + 1U);
                dflash_host_ingress->text_kv_table_rows[row]   = static_cast<std::int32_t>(row);
                dflash_host_ingress->dflash_kv_table_rows[row] = static_cast<std::int32_t>(row);
                dflash_host_ingress->lanes[row]                = static_cast<std::int32_t>(row);
                dflash_host_ingress->sampling[row]             = {};
            }
        }
        if (io.mtp_decode) {
            *mtp_host_ingress          = {};
            *mtp_host_egress           = {};
            const std::uint32_t extent = std::min(k, capacity - frontier - 1U);
            const std::uint32_t width  = draft_window + 1U;
            for (std::uint32_t row = 0; row < batch_size; ++row) {
                mtp_host_ingress->anchors[row] = 0;
                mtp_host_ingress->base_frontiers[row] =
                    checked_i32(frontier, "graph representative MTP frontier");
                mtp_host_ingress->remaining_budgets[row] =
                    checked_i32(capacity, "graph representative MTP budget");
                mtp_host_ingress->current_extents[row] = static_cast<std::int32_t>(extent);
                mtp_host_ingress->target_valid_columns[row] =
                    static_cast<std::int32_t>(extent + 1U);
                for (std::uint32_t step = 0; step < draft_window; ++step) {
                    mtp_host_ingress->current_drafts[row * draft_window + step] = 0;
                }
                for (std::uint32_t column = 0; column < width; ++column) {
                    mtp_host_ingress->target_rope_positions[row * width + column] =
                        checked_i32(frontier + std::min(column, extent),
                                    "graph representative MTP RoPE position");
                }
                mtp_host_ingress->text_kv_table_rows[row] = static_cast<std::int32_t>(row);
                mtp_host_ingress->mtp_kv_table_rows[row]  = static_cast<std::int32_t>(row);
                mtp_host_ingress->lanes[row]              = static_cast<std::int32_t>(row);
                mtp_host_ingress->rope_deltas[row]        = 0;
                mtp_host_ingress->sampling[row]           = {};
            }
        }
        if (io.ordinary) {
            *ordinary_host_ingress = {};
            *ordinary_host_egress  = {};
            for (std::uint32_t row = 0; row < batch_size; ++row) {
                ordinary_host_ingress->tokens[row] = 0;
                ordinary_host_ingress->cache_positions[row] =
                    checked_i32(frontier, "graph representative ordinary position");
                ordinary_host_ingress->rope_positions[row] =
                    checked_i32(frontier, "graph representative ordinary RoPE position");
                ordinary_host_ingress->text_kv_table_rows[row] = static_cast<std::int32_t>(row);
                ordinary_host_ingress->lanes[row]              = static_cast<std::int32_t>(row);
                ordinary_host_ingress->sampling[row]           = {};
            }
        }
    };
    const auto execution_core = [&] {
        return schedule::ExecutionCore{device,
                                       model,
                                       work,
                                       decoder->linear_attention,
                                       replay_records ? &*replay_records : nullptr,
                                       io,
                                       prefill_hidden,
                                       prefill_chunk,
                                       proposal_head,
                                       keep_frac,
                                       xattn_tau,
                                       xattn_min_len,
                                       gdn_history ? &*gdn_history : nullptr};
    };

    if (speculative_backend == SpeculativeBackend::None) {
        const auto ordinary_profiles = ordinary_graph_profiles(capacity);
        validate_graph_profiles(ordinary_profiles, capacity - 1, "ordinary");
        const std::uint32_t ordinary_batch_limit = max_concurrency;
        schedule::OrdinaryBatchContext ordinary_state{execution_core(),      decoder->text_kv,
                                                      io.ordinary.value(),   *ordinary_host_ingress,
                                                      *ordinary_host_egress, tail_hidden_store};
        const GraphExecutionProfile code_warm = ordinary_profiles.front();
        prepare_representative(code_warm.min, 1, draft_window);
        device.synchronize();
        schedule::ordinary_decode_batch(ordinary_state, 1, {code_warm.min + 1, code_warm.max + 1},
                                        nullptr);
        device.synchronize();

        ordinary_graphs.profiles.reserve(ordinary_profiles.size() * ordinary_batch_limit);
        for (std::uint32_t batch_size = 1; batch_size <= ordinary_batch_limit; ++batch_size) {
            for (const GraphExecutionProfile planned : ordinary_profiles) {
                ordinary_graphs.profiles.emplace_back();
                DecodeGraphProfile& profile    = ordinary_graphs.profiles.back();
                profile.batch_size             = batch_size;
                profile.min_execution_frontier = planned.min;
                profile.max_execution_frontier = planned.max;
                profile.topology_class =
                    planned.topology_class * ordinary_batch_limit + (batch_size - 1U);
                const ops::GqaExecutionEnvelope envelope{planned.min + 1, planned.max + 1};
                schedule::capture_ordinary_decode_batch(ordinary_state,
                                                        static_cast<std::int32_t>(batch_size),
                                                        envelope, profile.definition);
            }
        }
    }

    if (speculative_backend == SpeculativeBackend::Mtp) {
        std::uint32_t max_planned = 0;
        for (const std::uint32_t k : captured_ks) {
            const auto planned_profiles = mtp_graph_profiles(capacity, k);
            validate_graph_profiles(planned_profiles, capacity - 1, "MTP");
            for (const GraphExecutionProfile planned : planned_profiles) {
                max_planned = std::max(max_planned, planned.topology_class);
            }
        }
        const std::uint32_t k_stride = qwen3_6::adaptive_k_stride(max_concurrency, max_planned);
        const std::uint32_t warm_k   = captured_ks.empty() ? draft_window : captured_ks.back();
        schedule::MtpBatchContext mtp_state{
            execution_core(),  decoder->text_kv, *decoder->mtp_cache(), io.mtp_decode.value(),
            *mtp_host_ingress, *mtp_host_egress, tail_hidden_store,     tool_masks.get()};
        const auto warm_profiles              = mtp_graph_profiles(capacity, warm_k);
        const GraphExecutionProfile code_warm = warm_profiles.front();
        prepare_representative(code_warm.min, 1, warm_k);
        device.synchronize();
        schedule::mtp_decode_batch(mtp_state, 1, warm_k,
                                   mtp_gqa_envelopes(code_warm.max, warm_k, capacity), nullptr);
        device.synchronize();

        std::size_t profile_count = 0;
        for (const std::uint32_t k : captured_ks) {
            profile_count += mtp_graph_profiles(capacity, k).size() * max_concurrency *
                             qwen3_6::kGrammarExchangeVariants;
        }
        mtp_graphs.profiles.reserve(profile_count);
        for (std::uint32_t k_index = 0; k_index < captured_ks.size(); ++k_index) {
            const std::uint32_t k       = captured_ks[k_index];
            const auto planned_profiles = mtp_graph_profiles(capacity, k);
            for (std::uint32_t batch_size = 1; batch_size <= max_concurrency; ++batch_size) {
                for (const bool exchange : {false, true}) {
                    mtp_state.tool_masks = exchange ? tool_masks.get() : nullptr;
                    for (const GraphExecutionProfile planned : planned_profiles) {
                        mtp_graphs.profiles.emplace_back();
                        DecodeGraphProfile& profile    = mtp_graphs.profiles.back();
                        profile.batch_size             = batch_size;
                        profile.min_execution_frontier = planned.min;
                        profile.max_execution_frontier = planned.max;
                        profile.draft_k                = k;
                        profile.verify_width           = k + 1U;
                        profile.grammar_exchange       = exchange;
                        profile.topology_class         = qwen3_6::grammar_exchange_topology(
                            qwen3_6::adaptive_topology_class(k_index, k_stride,
                                                             planned.topology_class,
                                                             max_concurrency, batch_size),
                            exchange);
                        schedule::capture_mtp_decode_batch(
                            mtp_state, static_cast<std::int32_t>(batch_size), k,
                            mtp_gqa_envelopes(planned.max, k, capacity), profile.definition);
                    }
                }
            }
        }
    }
    if (speculative_backend == SpeculativeBackend::DFlash) {
        std::uint32_t max_planned = 0;
        for (const DFlashRoundShape& shape : dflash_shapes) {
            for (std::uint32_t batch_size = 1; batch_size <= shape.max_batch; ++batch_size) {
                const auto planned_profiles =
                    dflash_graph_profiles(capacity, shape.k, batch_size, shape.verify_width);
                if (batch_size == 1) {
                    validate_graph_profiles(planned_profiles, capacity - 1, "DFlash");
                }
                for (const GraphExecutionProfile planned : planned_profiles) {
                    max_planned = std::max(max_planned, planned.topology_class);
                }
            }
        }
        const std::uint32_t k_stride = qwen3_6::adaptive_k_stride(max_concurrency, max_planned);
        const std::uint32_t warm_k   = dflash_shapes.back().k;
        const std::uint32_t warm_w   = dflash_shapes.back().verify_width;
        schedule::DFlashBatchContext dflash_state{
            execution_core(),     decoder->text_kv,    dflash.value(),    io.dflash_decode.value(),
            *dflash_host_ingress, *dflash_host_egress, tail_hidden_store, tool_masks.get()};
        const auto batch_one_profiles         = dflash_graph_profiles(capacity, warm_k, 1, warm_w);
        const GraphExecutionProfile code_warm = batch_one_profiles.front();
        const ops::GqaExecutionEnvelope code_warm_target{
            1, static_cast<std::uint32_t>(std::min<std::uint64_t>(
                   capacity, static_cast<std::uint64_t>(code_warm.max) + warm_w))};
        prepare_representative(code_warm.min, 1, warm_k);
        device.synchronize();
        schedule::dflash_decode_batch(
            dflash_state, 1, warm_k, warm_w,
            dflash_envelopes(code_warm.min, code_warm.max, dflash_verify_width), code_warm_target,
            false, nullptr);
        device.synchronize();

        std::size_t dflash_profile_count = 0;
        for (const DFlashRoundShape& shape : dflash_shapes) {
            dflash_profile_count +=
                dflash_graph_profiles(capacity, shape.k, 1, shape.verify_width).size() *
                shape.max_batch * kDFlashExchangeVariants;
        }
        dflash_graphs.profiles.reserve(dflash_profile_count);
        for (std::uint32_t shape_index = 0; shape_index < dflash_shapes.size(); ++shape_index) {
            const std::uint32_t k  = dflash_shapes[shape_index].k;
            const std::uint32_t wk = dflash_shapes[shape_index].verify_width;
            for (std::uint32_t batch_size = 1; batch_size <= dflash_shapes[shape_index].max_batch;
                 ++batch_size) {
                const auto planned_profiles = dflash_graph_profiles(capacity, k, batch_size, wk);
                validate_graph_profiles(planned_profiles, capacity - 1, "DFlash");
                for (std::uint32_t variant = 0; variant < kDFlashExchangeVariants; ++variant) {
                    const bool exchange     = kDFlashExchangeVariants == 1 || variant == 1;
                    dflash_state.tool_masks = exchange ? tool_masks.get() : nullptr;
                    for (const GraphExecutionProfile planned : planned_profiles) {
                        dflash_graphs.profiles.emplace_back();
                        DecodeGraphProfile& profile    = dflash_graphs.profiles.back();
                        profile.batch_size             = batch_size;
                        profile.min_execution_frontier = planned.min;
                        profile.max_execution_frontier = planned.max;
                        profile.draft_k                = k;
                        profile.verify_width           = wk;
                        profile.grammar_exchange       = exchange;
                        const std::uint32_t topology   = qwen3_6::adaptive_topology_class(
                            shape_index, k_stride, planned.topology_class, max_concurrency,
                            batch_size);
                        profile.topology_class =
                            kDFlashExchangeVariants == 1
                                ? topology
                                : qwen3_6::grammar_exchange_topology(topology, exchange);
                        const ops::GqaExecutionEnvelope target_envelope{
                            1, static_cast<std::uint32_t>(std::min<std::uint64_t>(
                                   capacity, static_cast<std::uint64_t>(planned.max) + wk))};
                        prepare_representative(planned.min, batch_size, k);
                        device.synchronize();
                        schedule::capture_dflash_decode_batch(
                            dflash_state, static_cast<std::int32_t>(batch_size), k, wk,
                            dflash_envelopes(planned.min, planned.max, dflash_verify_width),
                            target_envelope, profile.definition);
                    }
                }
            }
        }
    }

    if (!ordinary_graphs.profiles.empty()) {
        instantiate_graph_family(ordinary_graphs, "ordinary", device, prepare_representative);
    }
    if (speculative_backend == SpeculativeBackend::Mtp) {
        instantiate_graph_family(mtp_graphs, "MTP", device, prepare_representative);
    }
    if (speculative_backend == SpeculativeBackend::DFlash) {
        instantiate_graph_family(dflash_graphs, "DFlash", device, prepare_representative);
    }

    ordered_reset(sequence);
    clear_stable_controls();
    for (Tensor& tensor : decoder->linear_attention.conv) {
        CUDA_CHECK(cudaMemsetAsync(tensor.data, 0, tensor.bytes(), device.stream));
    }
    for (Tensor& tensor : decoder->linear_attention.recurrent) {
        CUDA_CHECK(cudaMemsetAsync(tensor.data, 0, tensor.bytes(), device.stream));
    }
    if (gdn_history) {
        ops::gdn_history_reset_all(*gdn_history, device.stream);
        gdn_history_lengths.fill(0);
    }

    if (dflash) {
        const auto zero_cyclic_cache = [&](CyclicKVCache& cache) {
            for (std::uint32_t layer = 0; layer < cache.layer_count(); ++layer) {
                const CyclicKVCacheLayerView view = cache.layer_view(layer);
                CUDA_CHECK(cudaMemsetAsync(view.k.data, 0, view.k.bytes(), device.stream));
                CUDA_CHECK(cudaMemsetAsync(view.v.data, 0, view.v.bytes(), device.stream));
            }
        };
        zero_cyclic_cache(dflash->local);
        CUDA_CHECK(cudaMemsetAsync(dflash->prefill_features.data, 0,
                                   dflash->prefill_features.bytes(), device.stream));
        CUDA_CHECK(cudaMemsetAsync(dflash->prefill_positions.data, 0,
                                   dflash->prefill_positions.bytes(), device.stream));
        CUDA_CHECK(cudaMemsetAsync(dflash->pending_features.data, 0,
                                   dflash->pending_features.bytes(), device.stream));
    }
    CUDA_CHECK(cudaMemsetAsync(token_counts.data, 0, token_counts.bytes(), device.stream));
    device.synchronize();

    // Device-wide free memory also moves with other processes' allocations, so the delta is an
    // observation reported against the allowance, not a startup gate.
    std::size_t free_after = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_after, &total_bytes));
    graph_observed_bytes = free_before > free_after ? free_before - free_after : 0;
    for (PagedKVAllocation& allocation : dflash_capture_allocations) { allocation.unbind_row(); }
    dflash_capture_allocations.clear();
    for (PagedKVAllocation& allocation : mtp_capture_allocations) { allocation.unbind_row(); }
    mtp_capture_allocations.clear();
    for (PagedKVAllocation& allocation : text_capture_allocations) { allocation.unbind_row(); }
    text_capture_allocations.clear();
}

void ProgramImplCore::install_sampling(SequenceState& sequence, RequestControl& request,
                                       const ops::SamplingConfig& config) {
    Tensor counts = token_counts.slice(1, static_cast<std::int32_t>(sequence.lane), 1)
                        .view({TextConfig::token_domain});
    CUDA_CHECK(cudaMemsetAsync(counts.data, 0, counts.bytes(), device.stream));
    request.sampling_host     = config;
    request.speculative_stats = SpeculativeStats{
        .backend               = speculative_backend,
        .enabled               = speculative_backend != SpeculativeBackend::None,
        .draft_window          = draft_window,
        .accepted_per_position = std::vector<std::uint64_t>(draft_window, 0),
        .rounds_per_draft      = std::vector<std::uint64_t>(draft_window + 1U, 0),
    };
    qwen3_6::seed_adaptive_draft_state(
        request.adaptive,
        adaptive_draft ? 0U : qwen3_6::adaptive_seed_k(captured_ks, speculative_backend),
        config.temperature <= 0.0f ? qwen3_6::AdaptiveDraftLaw::Greedy
        : config.p_less != 0       ? qwen3_6::AdaptiveDraftLaw::PLess
                                   : qwen3_6::AdaptiveDraftLaw::Sampled);
    const bool penalties = request.sampling_host.presence_penalty != 0.0F ||
                           request.sampling_host.frequency_penalty != 0.0F;
    request.sampling_host.token_counts =
        penalties ? static_cast<std::int32_t*>(counts.data) : nullptr;
    const std::array<const qwen3_6::OutputSession*, 1> outputs{request.output};
    const std::array<ops::SamplingConfig, 1> configs{request.sampling_host};
    tool_masks->bind(outputs, configs);
    // The prefill owner is exclusive. Root storage can be shared with later
    // compact rounds; request sampling itself keeps no compact-row pointer.
    request.prefill_sampling_host = tool_masks->root(0, device.stream);
    Tensor config_lane = sampling_config.slice(1, static_cast<std::int32_t>(sequence.lane), 1);
    CUDA_CHECK(cudaMemcpyAsync(config_lane.data, &request.prefill_sampling_host,
                               sizeof(request.prefill_sampling_host), cudaMemcpyHostToDevice,
                               device.stream));
}

bool ProgramImplCore::any_tool_grammar(std::span<const std::uint32_t> lanes) const {
    return std::any_of(lanes.begin(), lanes.end(), [&](std::uint32_t lane) {
        return requests[lane].output && requests[lane].output->has_tool_grammar();
    });
}

bool ProgramImplCore::any_token_logprobs(std::span<const std::uint32_t> lanes) const {
    return std::any_of(lanes.begin(), lanes.end(),
                       [&](std::uint32_t lane) { return requests[lane].token_logprobs; });
}

void ProgramImplCore::bind_tool_mask_batch(std::span<const std::uint32_t> lanes,
                                           std::span<const ops::SamplingConfig> configs) {
    std::array<const qwen3_6::OutputSession*, kMaximumConcurrency> outputs{};
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        outputs[row] = requests[lanes[row]].output;
    }
    tool_masks->bind({outputs.data(), lanes.size()}, configs.first(lanes.size()));
}

void ProgramImplCore::copy_tail(SequenceState& sequence, const Tensor& source) {
    if (source.dtype != DType::BF16 || source.ne[0] != TextConfig::hidden || source.ne[1] != 1) {
        throw std::logic_error("target tail hidden has an invalid shape");
    }
    CUDA_CHECK(cudaMemcpyAsync(sequence.tail_hidden.data, source.data, sequence.tail_hidden.bytes(),
                               cudaMemcpyDeviceToDevice, device.stream));
    sequence.tail_hidden_valid = true;
}

void ProgramImplCore::copy_round_token() {
    CUDA_CHECK(cudaMemcpyAsync(host_tokens, io.token.data, sizeof(TokenId), cudaMemcpyDeviceToHost,
                               device.stream));
}

void ProgramImplCore::copy_prefill_logprobs() {
    CUDA_CHECK(cudaMemcpyAsync(host_prefill_logprobs,
                               static_cast<const unsigned char*>(io.prefill_logprob_frame.data) +
                                   offsetof(qwen3_6::PrefillLogprobFrame, records),
                               sizeof(*host_prefill_logprobs), cudaMemcpyDeviceToHost,
                               device.stream));
}

void ProgramImplCore::mark_workspace_usage(std::size_t phase_bytes) noexcept {
    workspace_logical_peak_bytes = std::max(workspace_logical_peak_bytes, phase_bytes);
}

void ProgramImplCore::enqueue_dflash_context_append(std::span<const std::uint32_t> lanes,
                                                    std::span<const std::uint32_t> starts,
                                                    std::span<const std::uint32_t> counts) {
    if (speculative_backend != SpeculativeBackend::DFlash || !dflash || !io.dflash_decode ||
        lanes.empty() || lanes.size() > max_concurrency || starts.size() != lanes.size() ||
        counts.size() != lanes.size()) {
        throw std::logic_error("DFlash context append has invalid membership");
    }
    DFlashPersistentState& dflash_persistent = *dflash;
    qwen3_6::DFlashDecodeState& frame        = *io.dflash_decode;

    std::uint32_t minimum_count = draft_window + 1U;
    std::uint32_t maximum_count = 0;
    *dflash_host_ingress        = {};
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency || counts[row] == 0 || counts[row] > draft_window + 1U ||
            std::find(lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(row), lane) !=
                lanes.begin() + static_cast<std::ptrdiff_t>(row)) {
            throw std::logic_error("DFlash context append contains an invalid row");
        }
        SequenceState& sequence   = sequences[lane];
        const std::uint32_t start = starts[row];
        const std::uint64_t end64 = static_cast<std::uint64_t>(start) + counts[row];
        const std::uint32_t end   = static_cast<std::uint32_t>(end64);
        if (!sequence.kv || sequence.kv->text.bound_row() < 0 || end64 > capacity) {
            throw std::logic_error("DFlash context append is outside retained target storage");
        }
        if constexpr (DFlashConfig::full_layers > 0) {
            if (!sequence.kv->backend || sequence.kv->backend->bound_row() < 0) {
                throw std::logic_error("DFlash context append is outside retained target storage");
            }
        }
        dflash_host_ingress->context_frontiers[row] =
            checked_i32(start, "DFlash append context frontier");
        dflash_host_ingress->execution_frontiers[row] =
            checked_i32(end, "DFlash append target frontier");
        dflash_host_ingress->dflash_kv_table_rows[row] =
            sequence.kv->backend ? sequence.kv->backend->bound_row() : 0;
        dflash_host_ingress->lanes[row] = static_cast<std::int32_t>(lane);
        const std::uint32_t backend_end = DFlashConfig::full_layers > 0 ? end : 0U;
        materialize_sequence_kv(sequence, std::max(sequence.text_kv_valid, end), backend_end);
        minimum_count = std::min(minimum_count, counts[row]);
        maximum_count = std::max(maximum_count, counts[row]);
    }

    CUDA_CHECK(cudaMemcpyAsync(frame.ingress.data, dflash_host_ingress,
                               sizeof(qwen3_6::DFlashDecodeIngress), cudaMemcpyHostToDevice,
                               device.stream));
    const auto batch     = static_cast<std::int32_t>(lanes.size());
    Tensor lane_tensor   = frame.lanes.slice(0, 0, batch);
    Tensor device_starts = frame.context_frontiers.slice(0, 0, batch);
    Tensor device_ends   = frame.execution_frontiers.slice(0, 0, batch);
    Tensor table_rows    = frame.dflash_kv_table_rows.slice(0, 0, batch);
    Tensor positions     = frame.append_positions.slice(1, 0, batch);
    Tensor device_counts = frame.append_counts.slice(0, 0, batch);

    work.reset();
    Tensor features =
        work.alloc(DType::BF16, {DFlashConfig::feature_rows,
                                 static_cast<std::int32_t>(dflash_verify_width), batch});
    ops::prepare_ragged_prefix(dflash_persistent.pending_features, lane_tensor, device_starts,
                               device_ends, features, positions, device_counts, device.stream);

    schedule::DFlashAppendContext state{
        {device, model, work, decoder->linear_attention,
         replay_records ? &*replay_records : nullptr, io, prefill_hidden, prefill_chunk,
         proposal_head, keep_frac, xattn_tau, xattn_min_len, gdn_history ? &*gdn_history : nullptr},
        dflash_persistent};
    mark_workspace_usage(workspace_plan.dflash_context);
    schedule::dflash_append_context(state, features, positions, device_counts, lane_tensor,
                                    table_rows, {minimum_count, maximum_count});
}

void ProgramImplCore::validate_licensed_tokens(std::span<const TokenId> tokens) const {
    for (const TokenId token : tokens) {
        if (token < 0 || token >= TextConfig::token_domain) {
            throw std::runtime_error("target returned a token outside the 248077-token domain");
        }
    }
}

// Throughput over the trailing window (<= 1s) of staged prefill step records: the
// steady-state prefill rate once warm. When the whole prefill is shorter than the window,
// the window degenerates to the full prefill (the overall average). Zero-token steps
// (fully reused prefixes) contribute time but no tokens.
static void prefill_tail_rate(const std::vector<std::uint32_t>& step_tokens,
                              const std::vector<double>& step_seconds, double& tail_tok_s,
                              double& tail_window_s) {
    tail_tok_s    = 0.0;
    tail_window_s = 0.0;
    if (step_tokens.empty() || step_tokens.size() != step_seconds.size()) { return; }
    double total_seconds = 0.0;
    for (const double seconds : step_seconds) { total_seconds += seconds; }
    if (total_seconds <= 0.0) { return; }
    const double window         = std::min(1.0, total_seconds);
    double window_elapsed       = 0.0;
    std::uint64_t window_tokens = 0;
    for (std::size_t i = step_tokens.size(); i-- > 0;) {
        const double seconds = step_seconds[i];
        const double take    = std::min(seconds, window - window_elapsed);
        if (take <= 0.0) { break; }
        const double fraction = seconds > 0.0 ? take / seconds : 0.0;
        window_tokens += static_cast<std::uint64_t>(
            std::llround(static_cast<double>(step_tokens[i]) * fraction));
        window_elapsed += take;
    }
    tail_window_s = window_elapsed;
    tail_tok_s = window_elapsed > 0.0 ? static_cast<double>(window_tokens) / window_elapsed : 0.0;
}

runtime::PrefillStepResult ProgramImplCore::advance_prefill(SequenceState& sequence,
                                                            RequestControl& request) {
    if (request.lifecycle != Lifecycle::Prefilling || !request.prefill) {
        throw std::logic_error("staged prefill step requires an active concurrent request");
    }

    RequestControl::Prefill& staged = *request.prefill;
    const runtime::BeginSummary summary{.prompt_tokens        = staged.prompt_tokens,
                                        .reused_prompt_tokens = staged.base,
                                        .prefix_reuse_path    = staged.reuse,
                                        .prefix_reuse_source  = staged.reuse_source};
    bool host_input_consumed              = staged.host_input_consumed_pending;
    staged.host_input_consumed_pending    = false;
    std::uint32_t processed_prompt_tokens = 0;
    const auto started                    = Clock::now();
    try {
        materialize_gdn_history(sequence.lane);
        const bool rewrite_capture_pending =
            staged.rewrite_checkpoint_capture &&
            staged.cursor < staged.rewrite_checkpoint_capture->frontier;
        schedule::PrefillContext schedule_state{
            {device, model, work, decoder->linear_attention,
             replay_records ? &*replay_records : nullptr, io, prefill_hidden, prefill_chunk,
             proposal_head, keep_frac, xattn_tau, xattn_min_len,
             gdn_history ? &*gdn_history : nullptr},
            text_kv_view(sequence),
            mtp_kv_view(sequence),
            decoder->text_kv,
            decoder->mtp_cache(),
            dflash ? &*dflash : nullptr,
            staged.cursor,
            static_cast<const ops::SamplingConfig*>(
                sampling_config.slice(1, static_cast<std::int32_t>(sequence.lane), 1).data),
            &sequence.rewrite_checkpoint_hidden,
            LinearStateSlots::current_state_slot(sequence.lane, max_concurrency),
            staged.initial_mtp_extent,
            dflash_host_ingress};

        if (staged.mtp_bridge == MtpBridgeMode::BeforeSuffix) {
            if (staged.cursor != staged.base || staged.base == 0 ||
                staged.cursor >= staged.prompt_tokens) {
                throw std::logic_error("staged MTP bridge is outside the reusable suffix");
            }
            mark_workspace_usage(workspace_plan.mtp_prefill);
            const Tensor& previous_hidden =
                qwen3_6::detail::mtp_bridge_reads_rewrite_hidden(staged.reuse)
                    ? sequence.rewrite_checkpoint_hidden
                    : sequence.tail_hidden;
            const schedule::MtpBridgeInput bridge{
                .previous_hidden = &previous_hidden,
                .position        = checked_i32(staged.base - 1, "MTP bridge position"),
                .rope_position   = prompt_rope_position(staged.prompt, staged.base - 1),
            };
            if (staged.vision) {
                schedule::mtp_bridge_multimodal(schedule_state, staged.prompt, *staged.vision,
                                                bridge);
            } else {
                Tensor bridge_token = io.mtp.value().target_input_ids.slice(0, 0, 1);
                const TokenId token = staged.prompt.token_ids[staged.base];
                CUDA_CHECK(cudaMemcpyAsync(bridge_token.data, &token, sizeof(token),
                                           cudaMemcpyHostToDevice, device.stream));
                schedule::mtp_bridge_and_propose(schedule_state, bridge_token, previous_hidden,
                                                 bridge.position, bridge.rope_position, false);
            }
            sequence.mtp_kv_valid = staged.base;
            staged.mtp_bridge     = MtpBridgeMode::None;
        }

        if (staged.cursor < staged.prompt_tokens) {
            const std::uint32_t nominal =
                schedule::select_prefill_chunk(staged.prompt_tokens - staged.cursor, prefill_chunk);
            const bool final_candidate = staged.cursor + nominal == staged.prompt_tokens;
            mark_workspace_usage(staged.prepare_mtp ? workspace_plan.mtp_prefill
                                                    : workspace_plan.text_prefill);
            if (speculative_backend == SpeculativeBackend::DFlash) {
                mark_workspace_usage(workspace_plan.dflash_context);
            }
            schedule::PrefillChunkResult result;
            const std::optional<std::uint32_t> rewrite_checkpoint_capture_frontier =
                staged.rewrite_checkpoint_capture
                    ? std::optional<std::uint32_t>(staged.rewrite_checkpoint_capture->frontier)
                    : std::nullopt;
            // A vision-free suffix of a multimodal prompt still has 3-axis positions. The text
            // prefill path would continue with 1-D RoPE and a zero delta, so a later cold prefill
            // of the same tokens would not match the checkpoint this suffix captures.
            if (staged.vision) {
                mark_workspace_usage(workspace_plan.vision_encode);
                result = schedule::prefill_multimodal_chunk(
                    schedule_state, staged.prompt, *staged.vision, nominal,
                    rewrite_checkpoint_capture_frontier, final_candidate);
            } else if (staged.prompt.has_media()) {
                result = schedule::prefill_mrope_text_chunk(schedule_state, staged.prompt, nominal,
                                                            rewrite_checkpoint_capture_frontier,
                                                            final_candidate);
            } else {
                result = schedule::prefill_text_chunk(
                    schedule_state, std::span<const TokenId>(staged.prompt.token_ids), nominal,
                    rewrite_checkpoint_capture_frontier, final_candidate);
            }
            if (result.processed_tokens == 0 || result.processed_tokens > nominal) {
                throw std::logic_error("ordinary prefill chunk made invalid progress");
            }
            processed_prompt_tokens = result.processed_tokens;
            if (staged.vision && staged.vision->release_consumed_media_payload()) {
                host_input_consumed = true;
            }
            staged.cursor += result.processed_tokens;
            sequence.text_kv_valid = staged.cursor;
            if (staged.prepare_mtp) { sequence.mtp_kv_valid = staged.cursor; }
            if (speculative_backend == SpeculativeBackend::DFlash) {
                sequence.dflash_context_frontier = staged.cursor;
            }
            // A freeze host-fences staging and overwrites it, so it runs before the rewrite
            // capture: the capture's drain then overlaps later work and staging keeps the image.
            const bool capture_rewrite =
                rewrite_capture_pending &&
                staged.cursor >= staged.rewrite_checkpoint_capture->frontier;
            maybe_freeze_context_checkpoint(sequence, request, result.processed_tokens,
                                            capture_rewrite);
            if (capture_rewrite) {
                if (staged.cursor != staged.rewrite_checkpoint_capture->frontier) {
                    throw std::logic_error("prefill chunk crossed the rewrite checkpoint frontier");
                }
                capture_rewrite_image(sequence);
            }
            if (staged.rewrite_checkpoint_capture &&
                staged.cursor >= staged.rewrite_checkpoint_capture->frontier) {
                sequence.rewrite_checkpoint = RewriteCheckpoint{
                    .valid    = true,
                    .kind     = staged.rewrite_checkpoint_capture->kind,
                    .frontier = staged.rewrite_checkpoint_capture->frontier,
                };
            }

            if (!result.finalized) {
                if (staged.cursor == staged.prompt_tokens) {
                    throw std::logic_error("staged prefill reached the prompt without sampling");
                }
                const double step_seconds =
                    std::chrono::duration<double>(Clock::now() - started).count();
                staged.elapsed_seconds += step_seconds;
                staged.step_tokens.push_back(processed_prompt_tokens);
                staged.step_seconds.push_back(step_seconds);
                return runtime::PrefillStepResult{.summary = summary,
                                                  .processed_prompt_tokens =
                                                      processed_prompt_tokens,
                                                  .host_input_consumed = host_input_consumed};
            }
            if (staged.cursor != staged.prompt_tokens) {
                throw std::logic_error("staged prefill sampled before the prompt frontier");
            }
            copy_tail(sequence, prefill_hidden.slice(
                                    1, static_cast<std::int32_t>(result.processed_tokens) - 1, 1));
        } else {
            mark_workspace_usage(workspace_plan.ordinary_round);
            if (!sequence.tail_hidden_valid) {
                throw std::logic_error("zero-suffix reuse has no target tail hidden");
            }
            schedule::sample_from_hidden(schedule_state, sequence.tail_hidden,
                                         checked_i32(staged.prompt_tokens, "sample position"),
                                         ops::kSamplePurposePrefill);
            set_device_i32(io.rope_pos, checked_i32(staged.prompt_tokens, "rope position") +
                                            sequence.rope_delta);
            if (staged.prepare_mtp) {
                if (staged.mtp_bridge != MtpBridgeMode::AfterExactHit) {
                    throw std::logic_error("zero-suffix MTP reuse has no exact-hit bridge");
                }
                mark_workspace_usage(workspace_plan.mtp_prefill);
                const auto bridge_rope =
                    prompt_rope_position(staged.prompt, staged.prompt_tokens - 1);
                schedule::mtp_bridge_and_propose(
                    schedule_state, io.token, sequence.tail_hidden,
                    checked_i32(staged.prompt_tokens - 1, "MTP full-prefix bridge position"),
                    bridge_rope, staged.initial_mtp_extent != 0);
                sequence.mtp_kv_valid = staged.prompt_tokens;
                staged.mtp_bridge     = MtpBridgeMode::None;
            }
        }

        copy_round_token();
        const bool token_logprobs = request.token_logprobs;
        if (token_logprobs) { copy_prefill_logprobs(); }
        std::array<TokenId, qwen3_6::kMtpDecodeMaximumDrafts> initial_drafts{};
        if (staged.prepare_mtp && staged.initial_mtp_extent != 0) {
            CUDA_CHECK(cudaMemcpyAsync(initial_drafts.data(), io.mtp.value().draft_tokens.data,
                                       staged.initial_mtp_extent * sizeof(TokenId),
                                       cudaMemcpyDeviceToHost, device.stream));
        }
        device.synchronize();
        const double final_step_seconds =
            std::chrono::duration<double>(Clock::now() - started).count();
        staged.elapsed_seconds += final_step_seconds;
        staged.step_tokens.push_back(processed_prompt_tokens);
        staged.step_seconds.push_back(final_step_seconds);
        const double vision_seconds = staged.vision ? staged.vision->elapsed_seconds() : 0.0;
        const std::optional<RewriteCheckpointSpec> rewrite_checkpoint_capture =
            staged.rewrite_checkpoint_capture;
        const std::uint32_t prompt_tokens = staged.prompt_tokens;

        validate_licensed_tokens(std::span<const TokenId>(host_tokens, 1));
        if (sequence.ledger.size() != prompt_tokens) {
            throw std::logic_error("candidate token ledger does not match prompt length");
        }
        sequence.ledger.push_back(host_tokens[0]);
        sequence.prefix_identity.append_generated(1, sequence.rope_delta);
        sequence.text_kv_valid = prompt_tokens;
        if (staged.prepare_mtp) {
            if (sequence.mtp_kv_valid != prompt_tokens) {
                throw std::logic_error("staged MTP prefill did not reach the prompt frontier");
            }
            sequence.mtp_draft_count = staged.initial_mtp_extent;
            std::copy_n(initial_drafts.begin(), staged.initial_mtp_extent,
                        sequence.mtp_drafts.begin());
        } else if (speculative_backend == SpeculativeBackend::DFlash &&
                   sequence.dflash_context_frontier != prompt_tokens) {
            throw std::logic_error("staged DFlash prefill did not reach the prompt frontier");
        }
        sequence.tail_hidden_valid      = true;
        request.timings.vision_seconds  = vision_seconds;
        request.timings.prefill_seconds = std::max(0.0, staged.elapsed_seconds - vision_seconds);
        // Drain vision encode from the earliest step windows so tail_tok_s matches
        // overall prefill.tok_s on the common image-first path. Media later in the
        // prompt still biases the tail on short requests; prefill.ms / tok_s stay
        // exact (elapsed - vision). Without this, a short image request reports a
        // diluted "steady-state" prefill rate.
        if (vision_seconds > 0.0) {
            double remaining = vision_seconds;
            for (double& seconds : staged.step_seconds) {
                const double take = std::min(seconds, remaining);
                seconds -= take;
                remaining -= take;
                if (remaining <= 0.0) { break; }
            }
        }
        prefill_tail_rate(staged.step_tokens, staged.step_seconds,
                          request.timings.prefill_tail_tok_s,
                          request.timings.prefill_tail_window_s);
        if (rewrite_checkpoint_capture) {
            const std::uint32_t frontier = rewrite_checkpoint_capture->frontier;
            if (frontier == 0 || frontier > prompt_tokens || sequence.text_kv_valid < frontier) {
                throw std::logic_error("rewrite checkpoint was not materialized by Text prefill");
            }
            if (speculative_backend == SpeculativeBackend::Mtp &&
                (!staged.prepare_mtp || sequence.mtp_kv_valid < frontier - 1)) {
                throw std::logic_error("rewrite checkpoint has no complete MTP prefix");
            }
            if (speculative_backend == SpeculativeBackend::DFlash &&
                (!dflash || !sequence.kv || sequence.dflash_context_frontier < frontier)) {
                throw std::logic_error("rewrite checkpoint has no complete DFlash prefix");
            }
            if constexpr (DFlashConfig::full_layers > 0) {
                if (speculative_backend == SpeculativeBackend::DFlash &&
                    (!sequence.kv || !sequence.kv->backend)) {
                    throw std::logic_error("rewrite checkpoint has no complete DFlash prefix");
                }
            }
            sequence.rewrite_checkpoint = RewriteCheckpoint{
                .valid = true, .kind = rewrite_checkpoint_capture->kind, .frontier = frontier};
        }

        if (!staged.prompt.patches.empty()) {
            staged.prompt.release_media_payload();
            host_input_consumed = true;
        }

        request.prefill.reset();
        request.pending   = PendingCandidate{.kind          = PendingKind::Begin,
                                             .base_E        = 0,
                                             .base_S        = 0,
                                             .prompt_tokens = prompt_tokens,
                                             .produced      = 1};
        request.lifecycle = Lifecycle::Pending;
        return runtime::PrefillStepResult{
            .summary = summary,
            .round = runtime::GeneratedRound{.tokens   = std::span<const TokenId>(host_tokens, 1),
                                             .logprobs = round_logprob_spans(*host_prefill_logprobs,
                                                                             token_logprobs)},
            .processed_prompt_tokens = processed_prompt_tokens,
            .complete                = true,
            .host_input_consumed     = host_input_consumed,
        };
    } catch (...) {
        synchronize_for_failure_cleanup(device);
        clear_lane(sequence, request);
        throw;
    }
}

runtime::BatchedGeneratedRound
ProgramImplCore::decode_ordinary_batch(std::span<const std::uint32_t> lanes,
                                       std::span<const runtime::RoundBudget> budgets) {
    std::array<bool, kMaximumConcurrency> cycle_exclusions{};
    if (speculative_backend != SpeculativeBackend::None) {
        throw std::logic_error("ordinary batch execution requires the ordinary backend");
    }
    if (lanes.empty() || lanes.size() > max_concurrency || budgets.size() != lanes.size()) {
        throw std::invalid_argument("ordinary batch membership is invalid");
    }

    std::uint32_t maximum_frontier = 0;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency ||
            std::find(lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(row), lane) !=
                lanes.begin() + static_cast<std::ptrdiff_t>(row)) {
            throw std::invalid_argument("ordinary batch contains an invalid or duplicate lane");
        }
        const SequenceState& sequence = sequences[lane];
        const RequestControl& request = requests[lane];
        if (request.lifecycle != Lifecycle::Active ||
            budgets[row].generated_tokens_remaining == 0 || !sequence.kv ||
            sequence.kv->text.bound_row() < 0 || sequence.execution_frontier >= capacity ||
            sequence.ledger_frontier != sequence.execution_frontier + 1 ||
            sequence.ledger.size() != sequence.ledger_frontier ||
            sequence.prefix_identity.size() != sequence.ledger_frontier) {
            throw std::logic_error("ordinary batch row is not decode-ready");
        }
        maximum_frontier = std::max(maximum_frontier, sequence.execution_frontier);
    }

    try {
        DecodeGraphExecutable* executable = nullptr;
        ops::GqaExecutionEnvelope envelope{maximum_frontier + 1, maximum_frontier + 1};
        if (use_cuda_graph) {
            DecodeGraphProfile& profile =
                select_graph_profile(ordinary_graphs, static_cast<std::uint32_t>(lanes.size()),
                                     maximum_frontier, "ordinary batch");
            executable = &install_graph_profile(ordinary_graphs, profile, "ordinary batch");
            envelope   = {profile.min_execution_frontier + 1, profile.max_execution_frontier + 1};
        }

        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence = sequences[lanes[row]];
            RequestControl& request = requests[lanes[row]];
            arm_typical_exclude(request, sequence);
            cycle_exclusions[row]              = request.sampling_host.typical_exclude >= 0;
            const std::uint32_t frontier       = sequence.execution_frontier;
            ordinary_host_ingress->tokens[row] = sequence.ledger.back();
            ordinary_host_ingress->cache_positions[row] =
                checked_i32(frontier, "ordinary batch position");
            ordinary_host_ingress->rope_positions[row] =
                checked_i32(frontier, "ordinary batch RoPE position") + sequence.rope_delta;
            ordinary_host_ingress->text_kv_table_rows[row] = sequence.kv.value().text.bound_row();
            ordinary_host_ingress->lanes[row]        = static_cast<std::int32_t>(sequence.lane);
            ordinary_host_ingress->logprob_rows[row] = request.token_logprobs ? 1 : 0;
            ordinary_host_ingress->sampling[row]     = request.sampling_host;
            materialize_sequence_kv(sequence, frontier + 1, 0);
        }

        bind_tool_mask_batch(lanes, ordinary_host_ingress->sampling);
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            ordinary_host_ingress->sampling[row] = tool_masks->root(row, device.stream);
        }
        schedule::OrdinaryBatchContext schedule_state{
            {device, model, work, decoder->linear_attention,
             replay_records ? &*replay_records : nullptr, io, prefill_hidden, prefill_chunk,
             proposal_head, keep_frac, xattn_tau, xattn_min_len,
             gdn_history ? &*gdn_history : nullptr},
            decoder->text_kv,
            io.ordinary.value(),
            *ordinary_host_ingress,
            *ordinary_host_egress,
            tail_hidden_store};

        mark_workspace_usage(workspace_plan.ordinary_round);
        const auto start = Clock::now();
        schedule::ordinary_decode_batch(schedule_state, static_cast<std::int32_t>(lanes.size()),
                                        envelope, executable);
        const double seconds = synchronize_round_seconds(device, start);
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence    = sequences[lanes[row]];
            RequestControl& request    = requests[lanes[row]];
            const std::uint32_t base_E = sequence.execution_frontier;
            const std::uint32_t base_S = sequence.ledger_frontier;
            const TokenId token        = ordinary_host_egress->sampled_tokens[row];
            validate_licensed_tokens(std::span<const TokenId>(&token, 1));
            sequence.text_kv_valid     = base_E + 1;
            sequence.tail_hidden_valid = true;
            sequence.ledger.push_back(token);
            sequence.prefix_identity.append_generated(1, sequence.rope_delta);
            request.pending   = PendingCandidate{.kind          = PendingKind::Ordinary,
                                                 .base_E        = base_E,
                                                 .base_S        = base_S,
                                                 .prompt_tokens = 0,
                                                 .produced      = 1};
            request.lifecycle = Lifecycle::Pending;
            request.timings.decode_seconds += seconds;
        }
        return runtime::BatchedGeneratedRound{
            .tokens =
                std::span<const TokenId>(ordinary_host_egress->sampled_tokens.data(), lanes.size()),
            .logprobs =
                round_logprob_spans(ordinary_host_egress->logprobs, any_token_logprobs(lanes)),
            .cycle_exclusions = cycle_exclusions};
    } catch (...) {
        synchronize_for_failure_cleanup(device);
        for (const std::uint32_t lane : lanes) {
            if (lane < max_concurrency) { clear_lane(sequences[lane], requests[lane]); }
        }
        throw;
    }
}

runtime::BatchedGeneratedRound
ProgramImplCore::decode_mtp_batch(std::span<const std::uint32_t> lanes,
                                  std::span<const runtime::RoundBudget> budgets) {
    std::array<bool, kMaximumConcurrency> cycle_exclusions{};
    if (speculative_backend != SpeculativeBackend::Mtp || !io.mtp_decode ||
        decoder->mtp_cache() == nullptr) {
        throw std::logic_error("MTP batch execution requires the MTP backend");
    }
    if (lanes.empty() || lanes.size() > max_concurrency || budgets.size() != lanes.size()) {
        throw std::invalid_argument("MTP batch membership is invalid");
    }

    const std::uint32_t width      = draft_window + 1;
    std::uint32_t maximum_frontier = 0;
    std::array<std::uint32_t, kMaximumConcurrency> row_ks{};
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency ||
            std::find(lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(row), lane) !=
                lanes.begin() + static_cast<std::ptrdiff_t>(row)) {
            throw std::invalid_argument("MTP batch contains an invalid or duplicate lane");
        }
        const SequenceState& sequence = sequences[lane];
        const RequestControl& request = requests[lane];
        if (request.lifecycle != Lifecycle::Active ||
            budgets[row].generated_tokens_remaining == 0 || !sequence.kv || !sequence.kv->backend ||
            sequence.kv->text.bound_row() < 0 || sequence.kv->backend->bound_row() < 0 ||
            sequence.execution_frontier >= capacity ||
            sequence.mtp_kv_valid != sequence.execution_frontier ||
            sequence.ledger_frontier != sequence.execution_frontier + 1 ||
            sequence.ledger.size() != sequence.ledger_frontier ||
            sequence.prefix_identity.size() != sequence.ledger_frontier ||
            sequence.mtp_draft_count > draft_window) {
            throw std::logic_error("MTP batch row is not decode-ready");
        }
        maximum_frontier                  = std::max(maximum_frontier, sequence.execution_frontier);
        const std::uint32_t budget_extent = budgets[row].generated_tokens_remaining > 1
                                                ? budgets[row].generated_tokens_remaining - 1
                                                : 0;
        const std::uint32_t cap_extent    = capacity > sequence.execution_frontier + 1
                                                ? capacity - sequence.execution_frontier - 1
                                                : 0;
        const std::uint32_t afford        = std::min(budget_extent, cap_extent);
        row_ks[row] = adaptive_draft ? afford : std::min({draft_window, budget_extent, cap_extent});
        if (adaptive_draft && lanes.size() == 1 && request.adaptive.live_k != 0) {
            row_ks[row] = std::min(request.adaptive.live_k, afford);
        }
    }
    std::array<const qwen3_6::AdaptiveDraftState*, kMaximumConcurrency> row_states{};
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        row_states[row] = &requests[lanes[row]].adaptive;
    }
    const std::uint32_t batch_idx = static_cast<std::uint32_t>(lanes.size()) - 1U;
    std::uint32_t batch_k         = qwen3_6::adaptive_batch_k(
        std::span<const std::uint32_t>(row_ks.data(), lanes.size()), captured_ks);
    if (adaptive_draft) {
        qwen3_6::AdaptiveRoundTimeState& t_state = adaptive_t_by_batch[batch_idx];
        qwen3_6::AdaptiveBatchKState& batch_st   = adaptive_batch_k_by_c[batch_idx];
        const auto states =
            std::span<const qwen3_6::AdaptiveDraftState* const>(row_states.data(), lanes.size());
        const auto caps = std::span<const std::uint32_t>(row_ks.data(), lanes.size());
        if (lanes.size() == 1) {
            qwen3_6::AdaptiveDraftState& ad = requests[lanes[0]].adaptive;
            if (ad.live_k == 0) {
                qwen3_6::AdaptiveDraftConfig cfg;
                cfg.hop_rates                          = hop_rates();
                cfg.captured_ks                        = captured_ks;
                cfg.round_time                         = &t_state;
                cfg.length_tokens                      = maximum_frontier;
                const qwen3_6::AdaptiveDraftState* ptr = &ad;
                const std::uint32_t row_cap[]          = {row_ks[0]};
                ad.live_k                              = qwen3_6::adaptive_select_k(
                    cfg, std::span<const qwen3_6::AdaptiveDraftState* const>(&ptr, 1),
                    std::span<const std::uint32_t>(row_cap, 1), row_ks[0], 0);
            }
            row_ks[0] = std::min(ad.live_k, row_ks[0]);
            batch_k   = qwen3_6::adaptive_batch_k(
                std::span<const std::uint32_t>(row_ks.data(), lanes.size()), captured_ks);
        } else if (batch_st.live_k == 0) {
            batch_k = qwen3_6::adaptive_batch_next(batch_st, states, caps, captured_ks, &t_state,
                                                   maximum_frontier, hop_rates());
        } else {
            batch_k = std::min(batch_st.live_k, batch_k);
        }
    }

    const bool grammar_exchange = any_tool_grammar(lanes);

    try {
        DecodeGraphExecutable* executable = nullptr;
        schedule::MtpGqaEnvelopes envelopes =
            mtp_gqa_envelopes(maximum_frontier, batch_k, capacity);
        if (use_cuda_graph) {
            DecodeGraphProfile& profile =
                select_graph_profile(mtp_graphs, static_cast<std::uint32_t>(lanes.size()),
                                     maximum_frontier, "MTP batch", batch_k, grammar_exchange);
            executable = &install_graph_profile(mtp_graphs, profile, "MTP batch");
            envelopes  = mtp_gqa_envelopes(profile.max_execution_frontier, batch_k, capacity);
        }

        std::uint32_t realized_extent = 0;
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence = sequences[lanes[row]];
            RequestControl& request = requests[lanes[row]];
            arm_typical_exclude(request, sequence);
            cycle_exclusions[row]             = request.sampling_host.typical_exclude >= 0;
            const std::uint32_t frontier      = sequence.execution_frontier;
            const std::uint32_t max_by_budget = budgets[row].generated_tokens_remaining > 1
                                                    ? budgets[row].generated_tokens_remaining - 1
                                                    : 0;
            const std::uint32_t extent = std::min({sequence.mtp_draft_count, batch_k, max_by_budget,
                                                   capacity - sequence.execution_frontier - 1});
            mtp_host_ingress->anchors[row]        = sequence.ledger.back();
            mtp_host_ingress->base_frontiers[row] = checked_i32(frontier, "MTP batch frontier");
            mtp_host_ingress->remaining_budgets[row] =
                checked_i32(budgets[row].generated_tokens_remaining, "MTP batch remaining budget");
            mtp_host_ingress->current_extents[row]      = static_cast<std::int32_t>(extent);
            mtp_host_ingress->target_valid_columns[row] = static_cast<std::int32_t>(extent + 1);
            for (std::uint32_t j = 0; j < draft_window; ++j) {
                mtp_host_ingress->current_drafts[row * draft_window + j] =
                    j < extent ? sequence.mtp_drafts[j] : sequence.ledger.back();
            }
            for (std::uint32_t j = 0; j < width; ++j) {
                const std::uint32_t position = frontier + std::min(j, extent);
                mtp_host_ingress->target_rope_positions[row * width + j] =
                    checked_i32(position, "MTP batch RoPE position") + sequence.rope_delta;
            }
            const SequenceKVBundle& kv                = sequence.kv.value();
            mtp_host_ingress->text_kv_table_rows[row] = kv.text.bound_row();
            mtp_host_ingress->mtp_kv_table_rows[row]  = kv.backend.value().bound_row();
            mtp_host_ingress->lanes[row]              = static_cast<std::int32_t>(sequence.lane);
            mtp_host_ingress->rope_deltas[row]        = sequence.rope_delta;
            mtp_host_ingress->logprob_rows[row]       = request.token_logprobs ? 1 : 0;
            mtp_host_ingress->sampling[row]           = request.sampling_host;
            materialize_sequence_kv(sequence, frontier + extent + 1,
                                    std::min(capacity, frontier + extent + draft_window));
            realized_extent = std::max(realized_extent, extent);
        }

        schedule::MtpBatchContext schedule_state{{device, model, work, decoder->linear_attention,
                                                  replay_records ? &*replay_records : nullptr, io,
                                                  prefill_hidden, prefill_chunk, proposal_head,
                                                  keep_frac, xattn_tau, xattn_min_len,
                                                  gdn_history ? &*gdn_history : nullptr},
                                                 decoder->text_kv,
                                                 *decoder->mtp_cache(),
                                                 *io.mtp_decode,
                                                 *mtp_host_ingress,
                                                 *mtp_host_egress,
                                                 tail_hidden_store,
                                                 grammar_exchange ? tool_masks.get() : nullptr};

        bind_tool_mask_batch(lanes, mtp_host_ingress->sampling);
        mark_workspace_usage(workspace_plan.mtp_round);
        const auto started = Clock::now();
        schedule::mtp_decode_batch(schedule_state, static_cast<std::int32_t>(lanes.size()), batch_k,
                                   envelopes, executable);
        const double seconds = synchronize_round_seconds(device, started);
        tool_masks->rethrow_error();
        // Fallback (extent 0) is not a k-draft round; do not fit T(batch_k) from it.
        if (adaptive_draft && realized_extent > 0) {
            qwen3_6::adaptive_observe_round_time(adaptive_t_by_batch[batch_idx], batch_k,
                                                 static_cast<float>(seconds), maximum_frontier);
        }
        std::array<std::uint32_t, kMaximumConcurrency> next_caps{};
        std::uint32_t next_length = 0;
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence                           = sequences[lanes[row]];
            RequestControl& request                           = requests[lanes[row]];
            const qwen3_6::AdaptiveDraftState adaptive_before = request.adaptive;
            const std::uint32_t base_E                        = sequence.execution_frontier;
            const std::uint32_t base_S                        = sequence.ledger_frontier;
            const std::int32_t count_i    = mtp_host_egress->licensed_counts[row];
            const std::int32_t accepted_i = mtp_host_egress->accepted_drafts[row];
            const std::int32_t next_i     = mtp_host_egress->next_extents[row];
            if (count_i <= 0 || count_i > static_cast<std::int32_t>(width) || accepted_i < 0 ||
                accepted_i + 1 != count_i || next_i < 0 ||
                next_i > static_cast<std::int32_t>(draft_window) ||
                static_cast<std::uint32_t>(count_i) > budgets[row].generated_tokens_remaining ||
                static_cast<std::uint64_t>(base_E) + static_cast<std::uint32_t>(count_i) >
                    capacity) {
                throw std::runtime_error("MTP batch returned invalid row metadata");
            }
            const std::span<const TokenId> row_tokens(mtp_host_egress->licensed_tokens.data() +
                                                          row * width,
                                                      static_cast<std::size_t>(count_i));
            validate_licensed_tokens(row_tokens);
            const std::uint32_t pcur =
                static_cast<std::uint32_t>(mtp_host_ingress->current_extents[row]);
            if (pcur == 0) {
                request.speculative_stats.fallback_steps += 1;
            } else {
                request.speculative_stats.rounds += 1;
                request.speculative_stats.drafted_tokens += pcur;
                request.speculative_stats.accepted_tokens += static_cast<std::uint32_t>(accepted_i);
                for (std::int32_t i = 0; i < accepted_i; ++i) {
                    request.speculative_stats.accepted_per_position[static_cast<std::size_t>(i)] +=
                        1;
                }
                if (batch_k < request.speculative_stats.rounds_per_draft.size()) {
                    request.speculative_stats.rounds_per_draft[batch_k] += 1;
                }
            }
            if (adaptive_draft) {
                const std::uint32_t remaining_after =
                    budgets[row].generated_tokens_remaining > static_cast<std::uint32_t>(count_i)
                        ? budgets[row].generated_tokens_remaining -
                              static_cast<std::uint32_t>(count_i)
                        : 0;
                const std::uint32_t next_budget = remaining_after > 1 ? remaining_after - 1 : 0;
                const std::uint32_t next_cap =
                    capacity > static_cast<std::uint32_t>(base_E) +
                                   static_cast<std::uint32_t>(count_i) + 1
                        ? capacity - (base_E + static_cast<std::uint32_t>(count_i)) - 1
                        : 0;
                const std::uint32_t budget_extent = std::min(next_budget, next_cap);
                next_caps[row]                    = budget_extent;
                next_length = std::max(next_length, base_E + static_cast<std::uint32_t>(count_i));
                if (pcur > 0) {
                    if (lanes.size() == 1) {
                        if (budget_extent > 0) {
                            qwen3_6::AdaptiveDraftConfig cfg;
                            cfg.hop_rates     = hop_rates();
                            cfg.captured_ks   = captured_ks;
                            cfg.round_time    = &adaptive_t_by_batch[batch_idx];
                            cfg.length_tokens = base_E + static_cast<std::uint32_t>(count_i);
                            (void)qwen3_6::adaptive_draft_next(
                                cfg, request.adaptive, static_cast<std::uint32_t>(accepted_i), pcur,
                                budget_extent, batch_k, false);
                        }
                    } else {
                        qwen3_6::adaptive_record_round(request.adaptive,
                                                       static_cast<std::uint32_t>(accepted_i), pcur,
                                                       batch_k, hop_rates(), false);
                    }
                }
            }
            request.speculative_stats.live_draft_tokens = request.adaptive.live_k;
            request.pending                             = PendingCandidate{
                .kind            = PendingKind::Speculative,
                .base_E          = base_E,
                .base_S          = base_S,
                .prompt_tokens   = 0,
                .produced        = static_cast<std::uint32_t>(count_i),
                .drafted         = pcur,
                .round_k         = batch_k,
                .verify_width    = batch_k + 1U,
                .tree_verify     = false,
                .adaptive_before = adaptive_before,
            };
            request.lifecycle = Lifecycle::Pending;
            request.timings.decode_seconds += seconds;
        }
        if (adaptive_draft && lanes.size() > 1) {
            std::array<qwen3_6::AdaptiveDraftState*, kMaximumConcurrency> mut_states{};
            for (std::size_t row = 0; row < lanes.size(); ++row) {
                mut_states[row] = &requests[lanes[row]].adaptive;
                row_states[row] = mut_states[row];
            }
            const std::uint32_t next = qwen3_6::adaptive_batch_next(
                adaptive_batch_k_by_c[batch_idx],
                std::span<const qwen3_6::AdaptiveDraftState* const>(row_states.data(),
                                                                    lanes.size()),
                std::span<const std::uint32_t>(next_caps.data(), lanes.size()), captured_ks,
                &adaptive_t_by_batch[batch_idx], next_length, hop_rates());
            qwen3_6::adaptive_assign_live_k(
                std::span<qwen3_6::AdaptiveDraftState*>(mut_states.data(), lanes.size()), next);
            for (std::size_t row = 0; row < lanes.size(); ++row) {
                requests[lanes[row]].speculative_stats.live_draft_tokens = next;
            }
        }
        return runtime::BatchedGeneratedRound{
            .tokens     = std::span<const TokenId>(mtp_host_egress->licensed_tokens.data(),
                                                   lanes.size() * width),
            .logprobs   = round_logprob_spans(mtp_host_egress->logprobs, any_token_logprobs(lanes)),
            .row_counts = std::span<const std::int32_t>(mtp_host_egress->licensed_counts.data(),
                                                        lanes.size()),
            .row_stride = width,
            .cycle_exclusions = cycle_exclusions};
    } catch (...) {
        synchronize_for_failure_cleanup(device);
        for (const std::uint32_t lane : lanes) {
            if (lane < max_concurrency) { clear_lane(sequences[lane], requests[lane]); }
        }
        throw;
    }
}

runtime::BatchedGeneratedRound
ProgramImplCore::decode_dflash_batch(std::span<const std::uint32_t> lanes,
                                     std::span<const runtime::RoundBudget> budgets) {
    std::array<bool, kMaximumConcurrency> cycle_exclusions{};
    if (speculative_backend != SpeculativeBackend::DFlash || !io.dflash_decode || !dflash) {
        throw std::logic_error("DFlash batch execution requires the DFlash backend");
    }
    if (lanes.empty() || lanes.size() > max_concurrency || budgets.size() != lanes.size()) {
        throw std::invalid_argument("DFlash batch membership is invalid");
    }

    const std::uint32_t width      = dflash_verify_width;
    std::uint32_t maximum_frontier = 0;
    std::array<std::uint32_t, kMaximumConcurrency> row_ks{};
    std::array<std::uint32_t, kMaximumConcurrency> row_affords{};
    std::uint32_t explore_k = 0;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency ||
            std::find(lanes.begin(), lanes.begin() + static_cast<std::ptrdiff_t>(row), lane) !=
                lanes.begin() + static_cast<std::ptrdiff_t>(row)) {
            throw std::invalid_argument("DFlash batch contains an invalid or duplicate lane");
        }
        const SequenceState& sequence = sequences[lane];
        const RequestControl& request = requests[lane];
        const bool backend_ready =
            DFlashConfig::full_layers == 0 ||
            (sequence.kv && sequence.kv->backend && sequence.kv->backend->bound_row() >= 0);
        if (request.lifecycle != Lifecycle::Active ||
            budgets[row].generated_tokens_remaining == 0 || !sequence.kv ||
            sequence.kv->text.bound_row() < 0 || !backend_ready ||
            sequence.execution_frontier >= capacity ||
            sequence.text_kv_valid != sequence.execution_frontier ||
            sequence.dflash_context_frontier > sequence.execution_frontier ||
            sequence.execution_frontier - sequence.dflash_context_frontier > width ||
            sequence.ledger_frontier != sequence.execution_frontier + 1 ||
            sequence.ledger.size() != sequence.ledger_frontier ||
            sequence.prefix_identity.size() != sequence.ledger_frontier) {
            throw std::logic_error("DFlash batch row is not decode-ready");
        }
        maximum_frontier                  = std::max(maximum_frontier, sequence.execution_frontier);
        const std::uint32_t budget_extent = budgets[row].generated_tokens_remaining > 1
                                                ? budgets[row].generated_tokens_remaining - 1U
                                                : 0U;
        const std::uint32_t cap_extent    = capacity > sequence.execution_frontier + 1
                                                ? capacity - sequence.execution_frontier - 1U
                                                : 0U;
        const std::uint32_t afford        = std::min(budget_extent, cap_extent);
        row_affords[row]                  = afford;
        row_ks[row] = adaptive_draft ? afford : std::min({draft_window, budget_extent, cap_extent});
    }
    std::array<const qwen3_6::AdaptiveDraftState*, kMaximumConcurrency> dflash_row_states{};
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        dflash_row_states[row] = &requests[lanes[row]].adaptive;
    }
    const std::uint32_t batch_idx = static_cast<std::uint32_t>(lanes.size()) - 1U;
    // Arms this batch size can run: chain (or pinned) shapes, plus the packed-tree arm in adaptive
    // mode while the packed verify fits one A8 tile. The picker learns which wins per batch size.
    const std::span<const std::uint32_t> arms = dflash_arms_by_batch[batch_idx];
    const std::uint32_t cap_k                 = *std::max_element(
        row_ks.begin(), row_ks.begin() + static_cast<std::ptrdiff_t>(lanes.size()));
    std::uint32_t batch_arm = qwen3_6::adaptive_batch_k(
        std::span<const std::uint32_t>(row_ks.data(), lanes.size()), arms);
    if (adaptive_draft) {
        qwen3_6::AdaptiveRoundTimeState& t_state = adaptive_t_by_batch[batch_idx];
        qwen3_6::AdaptiveBatchKState& batch_st   = adaptive_batch_k_by_c[batch_idx];
        const auto states = std::span<const qwen3_6::AdaptiveDraftState* const>(
            dflash_row_states.data(), lanes.size());
        const auto caps = std::span<const std::uint32_t>(row_ks.data(), lanes.size());
        if (lanes.size() == 1) {
            qwen3_6::AdaptiveDraftState& ad = requests[lanes[0]].adaptive;
            if (ad.live_k == 0) {
                qwen3_6::AdaptiveDraftConfig cfg;
                cfg.hop_rates                          = hop_rates();
                cfg.captured_ks                        = arms;
                cfg.round_time                         = &t_state;
                cfg.length_tokens                      = maximum_frontier;
                const qwen3_6::AdaptiveDraftState* ptr = &ad;
                const std::uint32_t row_cap[]          = {row_ks[0]};
                ad.live_k                              = qwen3_6::adaptive_select_k(
                    cfg, std::span<const qwen3_6::AdaptiveDraftState* const>(&ptr, 1),
                    std::span<const std::uint32_t>(row_cap, 1), row_ks[0], 0);
            }
            batch_arm = qwen3_6::adaptive_clamp_arm(arms, ad.live_k, row_ks[0]);
        } else if (batch_st.live_k == 0) {
            batch_arm = qwen3_6::adaptive_batch_next(batch_st, states, caps, arms, &t_state,
                                                     maximum_frontier, hop_rates());
        } else {
            batch_arm = qwen3_6::adaptive_clamp_arm(arms, batch_st.live_k, cap_k);
        }
        // Exploration: a content-independent one-round override that leaves the incumbent arm.
        explore_k = qwen3_6::adaptive_explore_k(adaptive_hop_rates, arms, t_state, cap_k);
        if (explore_k != 0) { batch_arm = explore_k; }
    }
    const DFlashRoundShape& round_shape = dflash_arm_shape(dflash_shapes, batch_arm);
    const std::uint32_t batch_k         = round_shape.k;
    const std::uint32_t live_w          = round_shape.verify_width;
    std::uint32_t maximum_target_tokens = 1;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        maximum_target_tokens =
            std::max(maximum_target_tokens, sequences[lanes[row]].execution_frontier + live_w);
    }

    const bool dflash_exchange = kDFlashExchangeVariants == 1 || any_tool_grammar(lanes);
    // Packed-tree rounds draw at the Variant tree temperature and produce no calibration.
    const bool calibrate_p_less =
        calibrates_p_less_drafts() && !dflash_uses_tree_verify(batch_k, live_w);

    try {
        DecodeGraphExecutable* executable   = nullptr;
        schedule::DFlashEnvelopes envelopes = dflash_envelopes(0, maximum_frontier, width);
        ops::GqaExecutionEnvelope target_envelope{maximum_frontier + 1, maximum_target_tokens};
        if (use_cuda_graph) {
            DecodeGraphProfile& profile = select_graph_profile(
                dflash_graphs, static_cast<std::uint32_t>(lanes.size()), maximum_frontier,
                "DFlash batch", batch_k, dflash_exchange, live_w);
            executable      = &install_graph_profile(dflash_graphs, profile, "DFlash batch");
            envelopes       = dflash_envelopes(profile.min_execution_frontier,
                                               profile.max_execution_frontier, width);
            target_envelope = {
                1, static_cast<std::uint32_t>(std::min<std::uint64_t>(
                       capacity,
                       static_cast<std::uint64_t>(profile.max_execution_frontier) + live_w))};
        }

        std::uint32_t realized_extent = 0;
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence = sequences[lanes[row]];
            RequestControl& request = requests[lanes[row]];
            arm_typical_exclude(request, sequence);
            cycle_exclusions[row]             = request.sampling_host.typical_exclude >= 0;
            const std::uint32_t frontier      = sequence.execution_frontier;
            const std::uint32_t max_by_budget = budgets[row].generated_tokens_remaining > 1
                                                    ? budgets[row].generated_tokens_remaining - 1U
                                                    : 0U;
            const std::uint32_t extent =
                std::min({batch_k, max_by_budget, capacity - frontier - 1U});
            if (row == 0) { *dflash_host_ingress = {}; }
            dflash_host_ingress->anchors[row] = sequence.ledger.back();
            dflash_host_ingress->execution_frontiers[row] =
                checked_i32(frontier, "DFlash batch frontier");
            dflash_host_ingress->context_frontiers[row] =
                checked_i32(sequence.dflash_context_frontier, "DFlash context frontier");
            dflash_host_ingress->proposal_extents[row]     = static_cast<std::int32_t>(extent);
            dflash_host_ingress->target_valid_columns[row] = static_cast<std::int32_t>(extent + 1U);
            dflash_host_ingress->text_kv_table_rows[row]   = sequence.kv->text.bound_row();
            dflash_host_ingress->dflash_kv_table_rows[row] =
                sequence.kv->backend ? sequence.kv->backend->bound_row() : 0;
            dflash_host_ingress->lanes[row]        = static_cast<std::int32_t>(sequence.lane);
            dflash_host_ingress->rope_deltas[row]  = sequence.rope_delta;
            dflash_host_ingress->logprob_rows[row] = request.token_logprobs ? 1 : 0;
            dflash_host_ingress->sampling[row]     = request.sampling_host;
            ops::SamplingConfig& row_sampling      = dflash_host_ingress->sampling[row];
            if (calibrate_p_less && row_sampling.draft_temperature > 0.0f) {
                row_sampling.draft_temperature = qwen3_6::p_less_calibrated_draft_temperature(
                    p_less_calibration, row_sampling.temperature, batch_k,
                    row_sampling.draft_temperature);
            }
            materialize_sequence_kv(sequence, std::min(capacity, frontier + dflash_verify_width),
                                    DFlashConfig::full_layers > 0 ? frontier : 0U);
            realized_extent = std::max(realized_extent, extent);
        }

        schedule::DFlashBatchContext schedule_state{
            {device, model, work, decoder->linear_attention,
             replay_records ? &*replay_records : nullptr, io, prefill_hidden, prefill_chunk,
             proposal_head, keep_frac, xattn_tau, xattn_min_len,
             gdn_history ? &*gdn_history : nullptr},
            decoder->text_kv,
            *dflash,
            *io.dflash_decode,
            *dflash_host_ingress,
            *dflash_host_egress,
            tail_hidden_store,
            dflash_exchange ? tool_masks.get() : nullptr,
            calibrate_p_less};

        bind_tool_mask_batch(lanes, dflash_host_ingress->sampling);
        mark_workspace_usage(workspace_plan.dflash_round);
        const auto started = Clock::now();
        schedule::dflash_decode_batch(schedule_state, static_cast<std::int32_t>(lanes.size()),
                                      batch_k, live_w, envelopes, target_envelope, !use_cuda_graph,
                                      executable);
        const double seconds = synchronize_round_seconds(device, started);
        tool_masks->rethrow_error();
        // Fallback (extent 0) is not a k-draft round; do not fit T(batch_k) from it.
        if (adaptive_draft && realized_extent > 0) {
            qwen3_6::adaptive_observe_round_time(adaptive_t_by_batch[batch_idx], batch_arm,
                                                 static_cast<float>(seconds), maximum_frontier);
        }
        if (ninfer::targets::qwen3_6::detail::dflash_candidate_stats_enabled() &&
            io.dflash_decode.has_value()) {
            qwen3_6::DFlashDecodeState& frame = *io.dflash_decode;
            const int w                       = static_cast<int>(width);
            const int b                       = static_cast<int>(lanes.size());
            std::vector<std::int32_t> ids(static_cast<std::size_t>(w) *
                                          static_cast<std::size_t>(b));
            std::vector<std::int32_t> pars(static_cast<std::size_t>(w) *
                                           static_cast<std::size_t>(b));
            CUDA_CHECK(cudaMemcpy(ids.data(), frame.verify_ids.data,
                                  ids.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(pars.data(), frame.parent_index.data,
                                  pars.size() * sizeof(std::int32_t), cudaMemcpyDeviceToHost));
            if (b == 1) {
                ninfer::targets::qwen3_6::detail::dflash_candidate_stats::capture_target_logits(
                    frame.target_logits, w);
            }
            for (int row = 0; row < b; ++row) {
                const int count =
                    dflash_host_egress->licensed_counts[static_cast<std::size_t>(row)];
                ninfer::targets::qwen3_6::detail::dflash_candidate_stats::record_round(
                    ids.data() + static_cast<std::size_t>(row) * static_cast<std::size_t>(w),
                    pars.data() + static_cast<std::size_t>(row) * static_cast<std::size_t>(w),
                    dflash_host_egress->licensed_tokens.data() +
                        static_cast<std::size_t>(row) * static_cast<std::size_t>(w),
                    count, w, static_cast<int>(draft_window));
            }
        }
        if (calibrate_p_less) {
            const std::size_t row_values =
                static_cast<std::size_t>(ops::kPLessProposalCalibrationTemperatureCount) * batch_k;
            for (std::size_t row = 0; row < lanes.size(); ++row) {
                const ops::SamplingConfig& row_sampling = dflash_host_ingress->sampling[row];
                if (row_sampling.p_less == 0 || !(row_sampling.temperature > 0.0f) ||
                    !(row_sampling.draft_temperature > 0.0f)) {
                    continue;
                }
                qwen3_6::p_less_calibration_observe(
                    p_less_calibration, row_sampling.temperature,
                    std::span<const float>(dflash_host_egress->proposal_calibration.data() +
                                               row * row_values,
                                           row_values),
                    static_cast<std::uint32_t>(dflash_host_ingress->proposal_extents[row]));
            }
        }
        std::array<std::uint32_t, kMaximumConcurrency> next_caps{};
        std::uint32_t next_length = 0;
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence                           = sequences[lanes[row]];
            RequestControl& request                           = requests[lanes[row]];
            const qwen3_6::AdaptiveDraftState adaptive_before = request.adaptive;
            const std::uint32_t base_E                        = sequence.execution_frontier;
            const std::uint32_t base_S                        = sequence.ledger_frontier;
            const std::int32_t count_i    = dflash_host_egress->licensed_counts[row];
            const std::int32_t accepted_i = dflash_host_egress->accepted_drafts[row];
            const std::uint32_t extent =
                static_cast<std::uint32_t>(dflash_host_ingress->proposal_extents[row]);
            if (count_i <= 0 || count_i > static_cast<std::int32_t>(width) || accepted_i < 0 ||
                accepted_i + 1 != count_i || accepted_i > static_cast<std::int32_t>(extent) ||
                static_cast<std::uint32_t>(count_i) > budgets[row].generated_tokens_remaining ||
                static_cast<std::uint64_t>(base_E) + static_cast<std::uint32_t>(count_i) >
                    capacity) {
                throw std::runtime_error("DFlash batch returned invalid row metadata");
            }
            const std::span<const TokenId> row_tokens(dflash_host_egress->licensed_tokens.data() +
                                                          row * width,
                                                      static_cast<std::size_t>(count_i));
            validate_licensed_tokens(row_tokens);
            if (extent == 0) {
                request.speculative_stats.fallback_steps += 1;
            } else {
                request.speculative_stats.rounds += 1;
                request.speculative_stats.drafted_tokens += extent;
                request.speculative_stats.accepted_tokens += static_cast<std::uint32_t>(accepted_i);
                for (std::int32_t i = 0; i < accepted_i; ++i) {
                    request.speculative_stats.accepted_per_position[static_cast<std::size_t>(i)] +=
                        1;
                }
                if (batch_k < request.speculative_stats.rounds_per_draft.size()) {
                    request.speculative_stats.rounds_per_draft[batch_k] += 1;
                }
            }
            if (adaptive_draft) {
                const std::uint32_t remaining_after =
                    budgets[row].generated_tokens_remaining > static_cast<std::uint32_t>(count_i)
                        ? budgets[row].generated_tokens_remaining -
                              static_cast<std::uint32_t>(count_i)
                        : 0;
                const std::uint32_t next_budget = remaining_after > 1 ? remaining_after - 1 : 0;
                const std::uint32_t next_cap =
                    capacity > base_E + static_cast<std::uint32_t>(count_i) + 1
                        ? capacity - (base_E + static_cast<std::uint32_t>(count_i)) - 1
                        : 0;
                const std::uint32_t budget_extent = std::min(next_budget, next_cap);
                next_caps[row]                    = budget_extent;
                next_length = std::max(next_length, base_E + static_cast<std::uint32_t>(count_i));
                if (extent > 0) {
                    if (lanes.size() == 1) {
                        if (budget_extent > 0) {
                            qwen3_6::AdaptiveDraftConfig cfg;
                            cfg.hop_rates     = hop_rates();
                            cfg.captured_ks   = arms;
                            cfg.round_time    = &adaptive_t_by_batch[batch_idx];
                            cfg.length_tokens = base_E + static_cast<std::uint32_t>(count_i);
                            (void)qwen3_6::adaptive_draft_next(
                                cfg, request.adaptive, static_cast<std::uint32_t>(accepted_i),
                                extent, budget_extent, batch_arm, explore_k != 0);
                        }
                    } else {
                        qwen3_6::adaptive_record_round(
                            request.adaptive, static_cast<std::uint32_t>(accepted_i), extent,
                            batch_arm, hop_rates(), explore_k != 0);
                    }
                }
            }
            request.speculative_stats.live_draft_tokens = request.adaptive.live_k;
            sequence.dflash_context_frontier            = base_E;
            request.pending                             = PendingCandidate{
                .kind            = PendingKind::Speculative,
                .base_E          = base_E,
                .base_S          = base_S,
                .prompt_tokens   = 0,
                .produced        = static_cast<std::uint32_t>(count_i),
                .drafted         = extent,
                .round_k         = batch_k,
                .verify_width    = live_w,
                .tree_verify     = dflash_uses_tree_verify(batch_k, live_w),
                .adaptive_before = adaptive_before,
            };
            request.lifecycle = Lifecycle::Pending;
            request.timings.decode_seconds += seconds;
        }
        if (adaptive_draft && lanes.size() > 1) {
            std::array<qwen3_6::AdaptiveDraftState*, kMaximumConcurrency> mut_states{};
            for (std::size_t row = 0; row < lanes.size(); ++row) {
                mut_states[row]        = &requests[lanes[row]].adaptive;
                dflash_row_states[row] = mut_states[row];
            }
            const std::uint32_t next = qwen3_6::adaptive_batch_next(
                adaptive_batch_k_by_c[batch_idx],
                std::span<const qwen3_6::AdaptiveDraftState* const>(dflash_row_states.data(),
                                                                    lanes.size()),
                std::span<const std::uint32_t>(next_caps.data(), lanes.size()), arms,
                &adaptive_t_by_batch[batch_idx], next_length, hop_rates());
            qwen3_6::adaptive_assign_live_k(
                std::span<qwen3_6::AdaptiveDraftState*>(mut_states.data(), lanes.size()), next);
            for (std::size_t row = 0; row < lanes.size(); ++row) {
                requests[lanes[row]].speculative_stats.live_draft_tokens = next;
            }
        }
        return runtime::BatchedGeneratedRound{
            .tokens = std::span<const TokenId>(dflash_host_egress->licensed_tokens.data(),
                                               lanes.size() * width),
            .logprobs =
                round_logprob_spans(dflash_host_egress->logprobs, any_token_logprobs(lanes)),
            .row_counts = std::span<const std::int32_t>(dflash_host_egress->licensed_counts.data(),
                                                        lanes.size()),
            .row_stride = width,
            .cycle_exclusions = cycle_exclusions};
    } catch (...) {
        synchronize_for_failure_cleanup(device);
        for (const std::uint32_t lane : lanes) {
            if (lane < max_concurrency) { clear_lane(sequences[lane], requests[lane]); }
        }
        throw;
    }
}

runtime::BatchedGeneratedRound
ProgramImplCore::decode_batch(std::span<const std::uint32_t> lanes,
                              std::span<const runtime::RoundBudget> budgets) {
    if (speculative_backend == SpeculativeBackend::None) {
        return decode_ordinary_batch(lanes, budgets);
    }
    if (speculative_backend == SpeculativeBackend::Mtp) { return decode_mtp_batch(lanes, budgets); }
    return decode_dflash_batch(lanes, budgets);
}

void ProgramImplCore::clear_suppressed_tokens_lane(std::uint32_t lane) {
    if (lane >= max_concurrency) { throw std::out_of_range("sampling lane is out of range"); }
    RequestControl& request = requests[lane];
    if (request.lifecycle == Lifecycle::Empty) {
        throw std::logic_error("cannot update sampling for an idle lane");
    }
    request.sampling_host.suppressed_token_count = 0;
}

void ProgramImplCore::set_typical_cycle_reasoning_lane(std::uint32_t lane, bool enabled) {
    if (lane >= max_concurrency) { throw std::out_of_range("sampling lane is out of range"); }
    RequestControl& request = requests[lane];
    if (request.lifecycle == Lifecycle::Empty) {
        throw std::logic_error("cannot update sampling for an idle lane");
    }
    request.typical_cycle_reasoning = enabled;
}

void ProgramImplCore::set_suppressed_tokens_lane(std::uint32_t lane,
                                                 std::span<const TokenId> tokens) {
    if (lane >= max_concurrency) { throw std::out_of_range("sampling lane is out of range"); }
    RequestControl& request = requests[lane];
    if (request.lifecycle == Lifecycle::Empty) {
        throw std::logic_error("cannot update sampling for an idle lane");
    }
    if (tokens.size() > static_cast<std::size_t>(request.sampling_host.kMaximumSuppressedTokens)) {
        throw std::invalid_argument("too many suppressed sampling tokens");
    }
    request.sampling_host.suppressed_token_count = static_cast<std::int32_t>(tokens.size());
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        request.sampling_host.suppressed_tokens[index] = tokens[index];
    }
}

void ProgramImplCore::resolve_non_speculative_pending(SequenceState& sequence,
                                                      RequestControl& request,
                                                      std::uint32_t accepted_tokens,
                                                      bool terminal) {
    if (request.lifecycle != Lifecycle::Pending) {
        throw std::logic_error("pending resolution requires a pending generated round");
    }
    if ((request.pending.kind != PendingKind::Begin &&
         request.pending.kind != PendingKind::Ordinary) ||
        request.pending.produced != 1 || accepted_tokens != 1) {
        throw std::logic_error("non-speculative pending round must commit its single token");
    }

    switch (request.pending.kind) {
    case PendingKind::Begin:
        sequence.execution_frontier = request.pending.prompt_tokens;
        sequence.ledger_frontier    = request.pending.prompt_tokens + 1;
        break;
    case PendingKind::Ordinary:
        sequence.execution_frontier = request.pending.base_E + request.pending.produced;
        sequence.ledger_frontier    = request.pending.base_S + request.pending.produced;
        break;
    case PendingKind::Speculative:
    case PendingKind::None:
        throw std::logic_error("non-speculative pending round has an invalid kind");
    }
    if (sequence.ledger_frontier != sequence.execution_frontier + 1 ||
        sequence.ledger.size() != sequence.ledger_frontier ||
        sequence.prefix_identity.size() != sequence.ledger_frontier) {
        throw std::logic_error("resolved round did not establish a valid frontier");
    }
    trim_sequence_kv(sequence, sequence.text_kv_valid, backend_kv_valid(sequence));
    if (terminal) {
        sequence.mtp_draft_count = 0;
        release_sequence_growth_entitlement(sequence);
        unbind_sequence_kv(sequence);
        sequence.retained = true;
    }
    request.lifecycle = terminal ? Lifecycle::Complete : Lifecycle::Active;
    request.pending   = {};
}

MemorySummary ProgramImplCore::memory_summary() const noexcept {
    MemorySummary out;
    out.device           = device.device;
    out.max_context      = capacity;
    out.kv_capacity      = kv_capacity;
    out.kv_cache         = kv_dtype == DType::BF16 ? KvCacheStorage::BFloat16
                           : kv_dtype == DType::U8 ? KvCacheStorage::Nvfp4
                                                   : KvCacheStorage::Int8Group64;
    DeviceArena& weights = *model.weights_arena;
    out.weights = ArenaMemorySummary{weights.capacity(), weights.used(), weights.peak_used()};
    out.sequence =
        ArenaMemorySummary{persistent.capacity(), persistent.used(), persistent.peak_used()};
    out.workspace = ArenaMemorySummary{workspace_storage.capacity(), work.used(), work.peak_used()};
    out.workspace_logical_peak_bytes = workspace_logical_peak_bytes;
    out.cuda_graph_allowance_bytes   = graph_allowance_bytes;
    out.cuda_graph_observed_bytes    = graph_observed_bytes;
    out.kv_payload_bytes             = kv_payload_bytes;
    out.checkpoint_image_host_bytes =
        checkpoint_image_slab_ ? checkpoint_image_slab_->capacity() : 0;
    out.checkpoint_image_pool_heads = context_checkpoint_pool_.capacity();
    const qwen3_6::detail::KvRamSnapshot ram =
        kv_ram_cache_ ? kv_ram_cache_->snapshot() : qwen3_6::detail::KvRamSnapshot{};
    out.kv_ram_capacity_bytes = ram.capacity_bytes;
    out.kv_ram_used_bytes     = ram.used_bytes;
    out.kv_ram_entry_count    = ram.entry_count;
    const qwen3_6::detail::KvDiskSnapshot disk =
        kv_disk_cache_ ? kv_disk_cache_->snapshot() : qwen3_6::detail::KvDiskSnapshot{};
    out.kv_disk_capacity_bytes = disk.capacity_bytes;
    out.kv_disk_used_bytes     = disk.used_bytes;
    out.kv_disk_entry_count    = disk.entry_count;
    return out;
}

void ProgramImplCore::reset_memory_peaks() noexcept {
    model.weights_arena->reset_peak();
    persistent.reset_peak();
    work.reset_peak();
    workspace_logical_peak_bytes = 0;
}

void ProgramImplCore::accumulate_prefill_nll(std::span<const TokenId> ids,
                                             std::uint32_t chunk_begin, std::uint32_t chunk_tokens,
                                             std::uint32_t skip, ScoreResult& result) {
    if (chunk_tokens == 0) { return; }
    const std::uint32_t prompt_tokens = static_cast<std::uint32_t>(ids.size());
    if (chunk_begin >= prompt_tokens) { return; }
    const std::uint32_t available = std::min(chunk_tokens, prompt_tokens - chunk_begin);
    if (prefill_hidden.dtype != DType::BF16 || prefill_hidden.ne[0] != TextConfig::hidden ||
        prefill_hidden.ne[1] < static_cast<std::int32_t>(available)) {
        throw std::logic_error("score prefill hidden does not match the chunk");
    }

    const std::vector<std::int32_t> targets =
        prefill_chunk_targets(ids, chunk_begin, chunk_tokens, skip);
    for (const std::int32_t target : targets) {
        if (target < 0 || target >= TextConfig::token_domain) {
            throw std::out_of_range("score target token is outside the checkpoint vocabulary");
        }
    }
    if (targets.empty()) { return; }

    const std::uint32_t first_scored = std::max(chunk_begin, skip);
    const std::int32_t hidden_origin = static_cast<std::int32_t>(first_scored - chunk_begin);

    constexpr std::int32_t kSlice = 64;
    const auto scored             = static_cast<std::int32_t>(targets.size());
    DeviceArena score_workspace(
        static_cast<std::size_t>(TextConfig::output_rows) * static_cast<std::size_t>(kSlice) *
            dtype_size(DType::BF16) +
        static_cast<std::size_t>(kSlice) * (sizeof(std::int32_t) + sizeof(float)) + 4096);
    std::vector<float> host_nll(static_cast<std::size_t>(kSlice));

    for (std::int32_t begin = 0; begin < scored; begin += kSlice) {
        const std::int32_t width = std::min(kSlice, scored - begin);
        score_workspace.reset();
        Tensor hidden        = prefill_hidden.slice(1, hidden_origin + begin, width);
        Tensor logits        = score_workspace.alloc(DType::BF16, {TextConfig::output_rows, width});
        Tensor target_tensor = score_workspace.alloc(DType::I32, {width});
        Tensor nll           = score_workspace.alloc(DType::FP32, {width});
        CUDA_CHECK(cudaMemcpyAsync(target_tensor.data, targets.data() + begin,
                                   static_cast<std::size_t>(width) * sizeof(std::int32_t),
                                   cudaMemcpyHostToDevice, device.stream));
        ops::linear(hidden, model.output_head, logits, device.stream);
        ops::nll_from_logits(logits, target_tensor, nll, TextConfig::token_domain, device.stream);
        CUDA_CHECK(cudaMemcpyAsync(host_nll.data(), nll.data,
                                   static_cast<std::size_t>(width) * sizeof(float),
                                   cudaMemcpyDeviceToHost, device.stream));
        CUDA_CHECK(cudaStreamSynchronize(device.stream));
        for (std::int32_t i = 0; i < width; ++i) {
            record_score_nll(result, static_cast<double>(host_nll[static_cast<std::size_t>(i)]));
        }
    }
}

void ProgramImplCore::accumulate_decode_nll(const Tensor& logits, TokenId target,
                                            ScoreResult& result, DeviceArena& score_workspace) {
    if (target < 0 || target >= TextConfig::token_domain) {
        throw std::out_of_range("score target token is outside the checkpoint vocabulary");
    }
    score_workspace.reset();
    Tensor target_tensor           = score_workspace.alloc(DType::I32, {1});
    Tensor nll                     = score_workspace.alloc(DType::FP32, {1});
    const std::int32_t host_target = target;
    CUDA_CHECK(cudaMemcpyAsync(target_tensor.data, &host_target, sizeof(host_target),
                               cudaMemcpyHostToDevice, device.stream));
    ops::nll_from_logits(logits, target_tensor, nll, TextConfig::token_domain, device.stream);
    float host_nll = 0.0f;
    CUDA_CHECK(cudaMemcpyAsync(&host_nll, nll.data, sizeof(host_nll), cudaMemcpyDeviceToHost,
                               device.stream));
    CUDA_CHECK(cudaStreamSynchronize(device.stream));
    record_score_nll(result, static_cast<double>(host_nll));
}

void ProgramImplCore::run_prefill_score(PreparedPromptData&& prompt, RequestPlan&& plan,
                                        runtime::TransientRegion transient,
                                        std::span<const TokenId> ids, std::uint32_t skip,
                                        ScoreResult& result) {
    std::uint32_t cursor = 0;
    try {
        runtime::PrefillStepResult step =
            start_prefill_lane(0, std::move(prompt), std::move(plan), transient);
        if (step.processed_prompt_tokens > 0) {
            accumulate_prefill_nll(ids, cursor, step.processed_prompt_tokens, skip, result);
            cursor += step.processed_prompt_tokens;
        }
        while (!step.complete) {
            step = advance_prefill_lane(0);
            if (step.processed_prompt_tokens > 0) {
                accumulate_prefill_nll(ids, cursor, step.processed_prompt_tokens, skip, result);
                cursor += step.processed_prompt_tokens;
            }
        }
        abort_lane(0);
    } catch (...) {
        abort_lane(0);
        throw;
    }
}

void ProgramImplCore::run_decode_score(PreparedPromptData&& prompt,
                                       runtime::TransientRegion transient,
                                       std::span<const TokenId> ids, std::uint32_t prefix,
                                       ScoreResult& result) {
    if (speculative_backend == SpeculativeBackend::DFlash) {
        throw std::invalid_argument(
            "decode score does not yet teacher-force DFlash; use ordinary or MTP");
    }
    if (prompt.has_media()) { throw std::invalid_argument("decode score is text-only"); }
    const auto n = static_cast<std::uint32_t>(ids.size());
    if (prefix == 0 || prefix + 1 >= n ||
        prefix > static_cast<std::uint32_t>(prompt.token_ids.size())) {
        throw std::invalid_argument("decode score prefix is outside the prompt");
    }

    const std::size_t old_n = prompt.token_ids.size();
    if (prompt.token_types.size() != old_n || prompt.positions.size() != 3 * old_n) {
        throw std::invalid_argument("prepared prompt metadata does not match token count");
    }
    std::vector<std::int32_t> prefix_positions(static_cast<std::size_t>(prefix) * 3);
    for (int axis = 0; axis < 3; ++axis) {
        for (std::uint32_t i = 0; i < prefix; ++i) {
            prefix_positions[static_cast<std::size_t>(axis) * prefix + i] =
                prompt.positions[static_cast<std::size_t>(axis) * old_n + i];
        }
    }
    prompt.token_ids.resize(prefix);
    prompt.token_types.resize(prefix);
    prompt.positions = std::move(prefix_positions);
    prompt.identity.rewrite_checkpoint.reset();

    DeviceArena score_workspace(4096);
    runtime::ResolvedExecutionOptions execution;
    execution.sampling.temperature    = 0.0F;
    execution.requested_output_tokens = n - prefix;
    execution.allow_prefix_reuse      = false;
    RequestBasePlan base              = plan_request_base(prompt, execution);
    RequestPlan plan                  = plan_request_for_lane(0, prompt, base);

    try {
        runtime::PrefillStepResult step =
            start_prefill_lane(0, std::move(prompt), std::move(plan), transient);
        while (!step.complete) { step = advance_prefill_lane(0); }
        SequenceState& sequence = sequences[0];
        if (sequence.ledger.size() != prefix + 1) {
            throw std::logic_error("decode score prefix ledger is not prompt plus sampled token");
        }
        sequence.ledger.back() = ids[prefix];
        sequence.prefix_hashes.truncate(sequence.ledger.size() - 1);
        if (speculative_backend == SpeculativeBackend::Mtp) { sequence.mtp_draft_count = 0; }
        resolve_prefill_lane(0, false);

        const std::array<std::uint32_t, 1> lanes{0};
        const std::array<runtime::RoundBudget, 1> budgets{
            runtime::RoundBudget{.generated_tokens_remaining = 1}};
        const std::array<std::uint32_t, 1> accepted{1};
        const std::array<std::uint8_t, 1> terminal{0};
        const std::array<std::uint8_t, 1> cancelled{0};
        for (std::uint32_t position = prefix; position + 1 < n; ++position) {
            if (sequence.ledger.back() != ids[position]) {
                throw std::logic_error("decode score ledger lost teacher-forced alignment");
            }
            Tensor logits;
            if (speculative_backend == SpeculativeBackend::Mtp) {
                (void)decode_mtp_batch(lanes, budgets);
                if (!io.mtp_decode) {
                    throw std::logic_error("MTP decode score has no target logits");
                }
                // T=1 target-verify column of [vocab, draft_window+1, batch].
                logits = io.mtp_decode->target_logits.slice(2, 0, 1).slice(1, 0, 1);
            } else {
                (void)decode_ordinary_batch(lanes, budgets);
                if (!io.ordinary) { throw std::logic_error("ordinary decode score has no logits"); }
                logits = io.ordinary->logits.slice(1, 0, 1);
            }
            accumulate_decode_nll(logits, ids[position + 1], result, score_workspace);
            if (speculative_backend == SpeculativeBackend::Mtp) {
                // MTP resolve inserts licensed tokens; replace the committed token with gold
                // and drop the model's next drafts so the following step stays T=1 verify.
                resolve_pending_batch(lanes, accepted, terminal, cancelled);
                sequence.ledger.back() = ids[position + 1];
                sequence.prefix_hashes.truncate(sequence.ledger.size() - 1);
                sequence.mtp_draft_count = 0;
            } else {
                sequence.ledger.back() = ids[position + 1];
                sequence.prefix_hashes.truncate(sequence.ledger.size() - 1);
                resolve_pending_batch(lanes, accepted, terminal, cancelled);
            }
        }
        abort_lane(0);
    } catch (...) {
        abort_lane(0);
        throw;
    }
}

ScoreResult ProgramImplCore::score(PreparedPromptData&& prompt, RequestPlan&& plan,
                                   runtime::TransientRegion transient, ScoreOptions options) {
    const auto started                   = Clock::now();
    const std::uint32_t prompt_tokens    = static_cast<std::uint32_t>(prompt.token_ids.size());
    const std::vector<TokenId> token_ids = prompt.token_ids;
    if (prompt_tokens < 2) {
        throw std::invalid_argument("score requires at least two prompt tokens");
    }
    const std::uint32_t skip = resolve_score_skip(prompt_tokens, options.skip_tokens);

    ScoreResult result;
    result.schedule      = options.schedule;
    result.prompt_tokens = prompt_tokens;
    result.skip_tokens   = skip;
    if (prompt_tokens > skip + 1) { result.token_nlls.reserve(prompt_tokens - skip - 1); }
    if (options.schedule == ScoreSchedule::Decode) {
        const std::uint32_t prefix = resolve_decode_prefix(prompt_tokens, skip);
        result.skip_tokens         = prefix;
        run_decode_score(std::move(prompt), transient, token_ids, prefix, result);
        (void)plan;
    } else {
        run_prefill_score(std::move(prompt), std::move(plan), transient, token_ids, skip, result);
    }

    result.score_seconds = std::chrono::duration<double>(Clock::now() - started).count();
    if (result.tokens_scored > 0) {
        result.mean_nll   = result.sum_nll / static_cast<double>(result.tokens_scored);
        result.perplexity = std::exp(result.mean_nll);
    }
    return result;
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
