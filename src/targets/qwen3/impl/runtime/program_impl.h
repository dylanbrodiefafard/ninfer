#include "targets/qwen3/impl/runtime/dflash_candidate_stats.h"
#include "targets/qwen3/impl/runtime/decision_trace.h"
#include "targets/qwen3/impl/runtime/layer_boundary_trace.h"
#include "targets/qwen3/impl/runtime/instance.h"
#include "targets/qwen3/impl/runtime/program.h"
#include "targets/qwen3/impl/runtime/speculative_stats.h"
#include "runtime/contract/typical_cycle.h"

#include <functional>

#include "targets/qwen3/impl/runtime/schedule.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/gdn_replay.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/nll_from_logits.h"
#include "targets/qwen3/impl/runtime/score_index.h"
#include "ninfer/ops/prepare_ragged_prefix.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/speculative_round.h"

#include "core/arena.h"
#include "core/device.h"
#include "core/dtype.h"
#include "core/roctx.h"
#include "core/host_copy.h"

#include <hip/hip_runtime.h>

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

namespace ninfer::targets::qwen3::detail::NINFER_QWEN3_RUNTIME_NS {
// NOLINTNEXTLINE(misc-anonymous-namespace-in-header): single-TU fragment (runtime.hip)
namespace {

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

// Driver-backed pinned checkpoint images cannot be reclaimed or swapped, so startup leaves this
// much host memory available after carving them instead of courting the OOM killer.
inline constexpr std::size_t kCheckpointSlabHostHeadroomBytes = std::size_t{4} << 30;

using Clock = std::chrono::steady_clock;

// One stable owner spans every device-controlled append segment in a decode round. Device status
// and cumulative cursors are resolved with one stream synchronization, then each row publishes
// only its caller-licensed frontier.
struct SegmentedKvTransactionBatch {
    std::array<std::optional<qwen3::PagedKVTransaction>, kMaximumConcurrency> transactions{};
    std::array<qwen3::PagedKVTransaction*, kMaximumConcurrency> ordered{};
    std::uint32_t* status = nullptr;
    std::uint32_t* cursor = nullptr;
    std::size_t size      = 0;
    std::size_t opened    = 0;

    SegmentedKvTransactionBatch(std::size_t count, KvResolutionWords words)
        : status(words.status), cursor(words.cursor), size(count) {
        if (count == 0 || count > ordered.size() || status == nullptr || cursor == nullptr) {
            throw std::invalid_argument("MTP segmented KV transaction batch size is invalid");
        }
    }

    void append(qwen3::PagedKVTransaction&& transaction) {
        if (opened >= size) {
            throw std::logic_error("MTP segmented KV transaction batch received too many rows");
        }
        ordered[opened] = &transactions[opened].emplace(std::move(transaction));
        ++opened;
    }

    [[nodiscard]] std::span<qwen3::PagedKVTransaction* const> binding() {
        if (opened != size) {
            throw std::logic_error("MTP segmented KV transaction batch is incomplete");
        }
        return {ordered.data(), size};
    }

    void mark_graph_replay(hipStream_t stream) {
        for (auto* transaction : binding()) { transaction->mark_graph_replay(stream); }
    }

    void enqueue_resolution() {
        for (std::size_t row = 0; row < size; ++row) {
            ordered[row]->enqueue_segmented_resolution(&status[row], &cursor[row]);
        }
    }

    void finish_resolution(std::span<const std::uint32_t> retained_frontiers) {
        if (retained_frontiers.size() != size) {
            throw std::invalid_argument("MTP segmented KV resolution has invalid frontiers");
        }
        for (std::size_t row = 0; row < size; ++row) {
            ordered[row]->finish_segmented_resolution(status[row], cursor[row],
                                                      retained_frontiers[row]);
        }
    }

    void close_captured() {
        for (auto* transaction : binding()) { transaction->close_captured_segmented(); }
    }
};

// The host view of a frame's logprob records, or empty spans when no row of the round asked.
template <std::size_t Slots>
runtime::RoundLogprobs round_logprob_spans(const qwen3::RoundLogprobRecords<Slots>& records,
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

// Drains in-flight device work on an error path before lane state is released. The original
// exception is the one the caller rethrows or reports, so a second device failure here must not
// replace it.
void synchronize_all_while_unwinding(DeviceContext& device) noexcept {
    try {
        device.synchronize_all();
        // NOLINTNEXTLINE(bugprone-empty-catch): the caller's original error takes precedence
    } catch (...) {}
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

// The counts are read and written on the compute stream, ordered against the speculative tail that
// produced them and the next round that consumes them. Each write is settled before `count` is
// reused, so a pageable host source is never read after it changes.
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
        HIP_CHECK(hipMemcpyAsync(&count, sampling.token_counts + tokens[index], sizeof(count),
                                 hipMemcpyDeviceToHost, device.stream));
        device.synchronize();
        if (count < occurrences) {
            throw std::logic_error("rejected sampling token count underflow");
        }
        count -= occurrences;
        HIP_CHECK(hipMemcpyAsync(sampling.token_counts + tokens[index], &count, sizeof(count),
                                 hipMemcpyHostToDevice, device.stream));
        device.synchronize();
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

schedule::DFlashEnvelopes dflash_envelopes(std::uint32_t max_frontier) {
    return schedule::DFlashEnvelopes{
        .local = {0, max_frontier},
        .full  = {0, max_frontier},
    };
}

DecodeGraphProfile& select_graph_profile(DecodeGraphFamily& family, std::uint32_t batch_size,
                                         std::uint32_t frontier, const char* label,
                                         std::uint32_t draft_tokens = 0) {
    const auto it = std::find_if(
        family.profiles.begin(), family.profiles.end(), [&](const DecodeGraphProfile& profile) {
            return profile.batch_size == batch_size && profile.draft_tokens == draft_tokens &&
                   profile.min_execution_frontier <= frontier &&
                   frontier <= profile.max_execution_frontier;
        });
    if (it == family.profiles.end()) {
        throw std::logic_error(std::string(label) + " Device Graph coverage is incomplete");
    }
    return *it;
}

// The planned execution profile containing `frontier`. Graph and eager decode both execute a
// transaction under its maximum frontier: attention split sizes and envelopes derive from that
// bound, so eager execution is exactly the computation a captured graph replays.
GraphExecutionProfile planned_execution_profile(const std::vector<GraphExecutionProfile>& profiles,
                                                std::uint32_t frontier, const char* label) {
    for (const GraphExecutionProfile& profile : profiles) {
        if (profile.min <= frontier && frontier <= profile.max) { return profile; }
    }
    throw std::logic_error(std::string(label) + " execution profile coverage is incomplete");
}

void validate_graph_profiles(const std::vector<GraphExecutionProfile>& profiles,
                             std::uint32_t max_frontier, const char* label) {
    if (profiles.empty() || profiles.front().min != 0 || profiles.back().max != max_frontier) {
        throw std::logic_error(std::string(label) + " Device Graph coverage has invalid endpoints");
    }
    for (std::size_t i = 0; i < profiles.size(); ++i) {
        if (profiles[i].min > profiles[i].max ||
            (i != 0 && profiles[i].min != profiles[i - 1].max + 1)) {
            throw std::logic_error(std::string(label) + " Device Graph coverage has a gap");
        }
    }
}

// Keeps the tool-mask matcher armed exactly while one speculative round can execute.
struct TokenMaskRoundGuard {
    TokenMaskRoundGuard(qwen3::TokenMaskExchange& exchange, DeviceContext& device)
        : exchange(exchange), stream(device.stream) {
        exchange.arm();
    }

    ~TokenMaskRoundGuard() {
        // Enqueue failures can unwind with a publish/acquire still in flight. Keep the
        // matcher alive until that work drains, including during Program construction.
        if (!finished) { (void)hipStreamSynchronize(stream); }
        exchange.disarm();
    }

    // Called only after round synchronization. Avoid another HIP wait on the normal path.
    void finish() {
        finished = true;
        exchange.finish_round();
    }

    TokenMaskRoundGuard(const TokenMaskRoundGuard&)            = delete;
    TokenMaskRoundGuard& operator=(const TokenMaskRoundGuard&) = delete;
    qwen3::TokenMaskExchange& exchange;
    hipStream_t stream;
    bool finished = false;
};

template <class Prepare>
void instantiate_graph_family(DecodeGraphFamily& family, const char* label, DeviceContext& device,
                              Prepare&& prepare, qwen3::TokenMaskExchange& token_masks) {
    if (family.profiles.empty()) {
        throw std::logic_error(std::string(label) + " Device Graph family has no profiles");
    }
    for (DecodeGraphProfile& profile : family.profiles) {
        // HIP executable updates retain fresh kernel-argument storage on every
        // profile transition. Immutable executables make residency startup-fixed.
        profile.executable.instantiate(profile.definition);
        profile.executable.upload(device.stream);
    }
    for (DecodeGraphProfile& profile : family.profiles) {
        // B, K and the target structural class fix the kernel routes. Warm the
        // largest frontier in each class after uploading every immutable graph.
        const bool has_larger_representative = std::any_of(
            family.profiles.begin(), family.profiles.end(), [&](const DecodeGraphProfile& other) {
                return other.batch_size == profile.batch_size &&
                       other.draft_tokens == profile.draft_tokens &&
                       other.topology_class == profile.topology_class &&
                       other.max_execution_frontier > profile.max_execution_frontier;
            });
        if (has_larger_representative) { continue; }
        prepare(profile.min_execution_frontier, profile.batch_size, profile.draft_tokens);
        device.synchronize();
        TokenMaskRoundGuard token_mask_round(token_masks, device);
        profile.executable.launch(device.stream);
        device.synchronize();
        token_mask_round.finish();
    }
}

} // namespace

ProgramImplCore::ProgramImplCore(const LoadedModelData& model_in, const SequencePlanImpl& plan,
                                 DeviceContext& device_in,
                                 std::unique_ptr<HostPinnedArena> kv_ram_arena)
    : model(model_in), device(device_in), weights_profile(plan.weights_profile),
      capacity(plan.capacity), kv_capacity(plan.kv_capacity), max_concurrency(plan.max_concurrency),
      prefill_chunk(plan.prefill_chunk), mixed_forward(plan.mixed_forward),
      draft_window(plan.draft_window), dflash_verify_width(plan.dflash_verify_width),
      adaptive_draft(plan.adaptive_draft), p_less_draft_temperature(plan.p_less_draft_temperature),
      captured_ks(plan.captured_ks), speculative_backend(plan.speculative_backend),
      context_marks(plan.context_checkpoint_marks), proposal_head(plan.proposal_head),
      vision_enabled(plan.features.vision), use_device_graph(plan.use_device_graph),
      kv_payload_bytes(plan.persistent.kv_payload_bytes),
      kv_ram_capacity_bytes(plan.kv_ram_capacity_bytes),
      kv_ram_system_prompts(plan.kv_ram_system_prompts),
      expected_graph_definition_count(plan.graph_definition_count),
      expected_graph_executable_count(plan.graph_executable_count),
      kv_disk_capacity_bytes(plan.kv_disk_capacity_bytes),
      graph_allowance_bytes(plan.graph_allowance_bytes), workspace_plan(plan.workspace),
      persistent(plan.persistent.bytes), workspace_storage(plan.workspace.capacity),
      work(DeviceSpan{workspace_storage.base(), workspace_storage.capacity()}),
      round_host(sizeof(PrefillRoundHost)),
      kv_resolution_host(4U * kMaximumConcurrency * sizeof(std::uint32_t)),
      ordinary_host(
          plan.speculative_backend == SpeculativeBackend::None
              ? std::make_optional<PinnedHostBuffer>(sizeof(qwen3::OrdinaryDecodeIngress) +
                                                     sizeof(qwen3::OrdinaryDecodeEgress))
              : std::nullopt),
      mtp_host(plan.speculative_backend == SpeculativeBackend::Mtp
                   ? std::make_optional<PinnedHostBuffer>(sizeof(qwen3::MtpDecodeIngress) +
                                                          sizeof(qwen3::MtpDecodeEgress))
                   : std::nullopt),
      dflash_host(plan.speculative_backend == SpeculativeBackend::DFlash
                      ? std::make_optional<PinnedHostBuffer>(sizeof(qwen3::DFlashDecodeIngress) +
                                                             sizeof(qwen3::DFlashDecodeEgress))
                      : std::nullopt) {
    if (model.weights_arena == nullptr) {
        throw std::invalid_argument("Qwen3 model view has no owning weight arena");
    }
    if (model.features != plan.features || model.mtp.has_value() != plan.features.mtp() ||
        model.dflash.has_value() != plan.features.dflash() ||
        model.optimized_proposal.has_value() != plan.features.optimized_proposal() ||
        model.vision.has_value() != plan.features.vision) {
        throw std::invalid_argument(
            "Qwen3 loaded weights do not match the frozen startup features");
    }
    if (model.mtp.has_value() && model.dflash.has_value()) {
        throw std::invalid_argument("MTP and DFlash model views are mutually exclusive");
    }
    const DeviceSpan backing = persistent.alloc_bytes(plan.persistent.bytes, 256);
    decoder = std::make_unique<qwen3::DecoderState>(backing, plan.persistent.decoder);
    if (plan.persistent.replay_records) {
        replay_records.emplace(backing, *plan.persistent.replay_records);
    }
    if (replay_records.has_value() != (speculative_backend != SpeculativeBackend::None)) {
        throw std::logic_error("ReplaySSM records do not match the sequence plan");
    }
    if (plan.persistent.dflash) { dflash.emplace(backing, *plan.persistent.dflash); }
    if (dflash.has_value() != plan.features.dflash()) {
        throw std::logic_error("DFlash state does not match the frozen sequence plan");
    }
    if (plan.persistent.linear_execution) {
        const auto verify_widths = qwen3::captured_verify_widths<DFlashConfig>(
            plan.speculative_backend, plan.captured_ks, plan.dflash_verify_width);
        linear_execution = std::make_unique<typename Variant::ExecutionState>(
            model, plan.persistent.linear_execution->bind(backing),
            std::min(plan.prefill_chunk, plan.capacity), plan.max_concurrency, verify_widths);
    }
    // Tier images and forks copy whole physical pages and complete padded cyclic images.
    // Define unused payload positions before prefill or graph warmup writes the live ranges.
    const auto zero_paged_payload = [&](const PagedKVPool& pool) {
        for (std::size_t plane = 0; plane < pool.plane_count(); ++plane) {
            const Tensor& payload = pool.plane(plane);
            HIP_CHECK(hipMemsetAsync(payload.data, 0, payload.bytes(), device.stream));
        }
    };
    zero_paged_payload(decoder->text_kv.pool());
    if (const PagedKVPool* backend = backend_kv_pool(); backend != nullptr) {
        zero_paged_payload(*backend);
    }
    if (dflash) { zero_dflash_local_cache(dflash->local); }

    io = qwen3::RoundState(backing, plan.persistent.round);
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
    // Egress transfers include the entire fixed frame; producers update only active rows and
    // requested records. Define inactive entries and struct padding before the first transfer.
    if (io.ordinary) {
        const DeviceSpan egress = io.ordinary->egress;
        HIP_CHECK(hipMemsetAsync(egress.data, 0, egress.bytes, device.stream));
    }
    if (io.mtp_decode) {
        const DeviceSpan egress = io.mtp_decode->egress;
        HIP_CHECK(hipMemsetAsync(egress.data, 0, egress.bytes, device.stream));
    }
    if (io.dflash_decode) {
        const DeviceSpan egress = io.dflash_decode->egress;
        HIP_CHECK(hipMemsetAsync(egress.data, 0, egress.bytes, device.stream));
    }
    prefill_hidden  = plan.persistent.prefill_hidden.bind(backing);
    token_counts    = plan.persistent.token_counts.bind(backing);
    sampling_config = plan.persistent.sampling_config.bind(backing);
    token_masks     = std::make_unique<qwen3::TokenMaskExchange>(
        plan.persistent.grammar_masks.bind(backing),
        plan.persistent.grammar_sampling_config.bind(backing));
    tail_hidden_store               = plan.persistent.tail_hidden.bind(backing);
    rewrite_checkpoint_hidden_store = plan.persistent.rewrite_checkpoint_hidden.bind(backing);
    if (plan.persistent.staging_hidden) {
        staging_hidden = plan.persistent.staging_hidden->bind(backing);
    }
    prompt_embedding = std::make_unique<PromptEmbeddingStaging>(
        device, model.token_embedding, plan.persistent.prompt_embedding_image.bind(backing),
        plan.persistent.prompt_embedding_ids);
    {
        // Every pinned checkpoint image is carved here from one driver-backed HIP/GTT slab, so
        // no serving path allocates or frees pinned memory for checkpoints.
        const std::size_t conv_bytes      = decoder->linear_attention.conv_host_image_bytes();
        const std::size_t recurrent_bytes = decoder->linear_attention.recurrent_host_image_bytes();
        const std::size_t dflash_bytes    = dflash ? dflash->local.lane_host_bytes() : 0;
        const std::size_t pool_heads =
            captures_context_checkpoints() && staging_hidden.data != nullptr
                ? qwen3::detail::context_checkpoint_image_pool_capacity(
                      max_concurrency, context_marks, capacity, kv_capacity)
                : 0;
        const auto region = [](std::size_t bytes) {
            constexpr std::size_t alignment = qwen3::detail::kCheckpointImageAlignment;
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
        sequence.next_context_mark = qwen3::detail::first_prefill_context_mark(context_marks);
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
        ordinary_host_ingress = static_cast<qwen3::OrdinaryDecodeIngress*>(ordinary_host->data());
        ordinary_host_egress  = reinterpret_cast<qwen3::OrdinaryDecodeEgress*>(
            static_cast<unsigned char*>(ordinary_host->data()) +
            sizeof(qwen3::OrdinaryDecodeIngress));
        *ordinary_host_ingress = {};
        *ordinary_host_egress  = {};
    }
    if (mtp_host) {
        mtp_host_ingress = static_cast<qwen3::MtpDecodeIngress*>(mtp_host->data());
        mtp_host_egress  = reinterpret_cast<qwen3::MtpDecodeEgress*>(
            static_cast<unsigned char*>(mtp_host->data()) + sizeof(qwen3::MtpDecodeIngress));
        *mtp_host_ingress = {};
        *mtp_host_egress  = {};
    }
    if (dflash_host) {
        dflash_host_ingress = static_cast<qwen3::DFlashDecodeIngress*>(dflash_host->data());
        dflash_host_egress  = reinterpret_cast<qwen3::DFlashDecodeEgress*>(
            static_cast<unsigned char*>(dflash_host->data()) + sizeof(qwen3::DFlashDecodeIngress));
        *dflash_host_ingress = {};
        *dflash_host_egress  = {};
    }
    if (io.dflash_prefill) {
        HIP_CHECK(hipMemsetAsync(io.dflash_prefill->produced_count.data, 0,
                                 io.dflash_prefill->produced_count.bytes(), device.stream));
    }
    HIP_CHECK(hipMemsetAsync(io.rope_delta.data, 0, io.rope_delta.bytes(), device.stream));
    if (io.mtp) {
        HIP_CHECK(
            hipMemsetAsync(io.mtp->position.data, 0, io.mtp->position.bytes(), device.stream));
    }
    HIP_CHECK(hipMemsetAsync(token_counts.data, 0, token_counts.bytes(), device.stream));
    HIP_CHECK(hipMemsetAsync(sampling_config.data, 0, sampling_config.bytes(), device.stream));
    HIP_CHECK(
        hipMemsetAsync(io.nonfinite_logits.data, 0, io.nonfinite_logits.bytes(), device.stream));
    device.synchronize();
    prepare_graphs();
    if (calibrates_p_less_drafts()) {
        // Poison the calibration egress with NaN. Every chain round's scoring launch rewrites
        // the entries the host reads, so a NaN reaching p_less_calibration_observe proves a round
        // ran without the scoring kernel, and that fails loudly instead of calibrating on stale
        // values.
        Tensor& calibration = io.dflash_decode.value().proposal_calibration;
        HIP_CHECK(hipMemsetAsync(calibration.data, 0xFF, calibration.bytes(), device.stream));
        device.synchronize();
    }
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
        kv_ram_cache_.emplace(std::move(*kv_ram_arena), kv_ram_system_prompts);
    }
    if (kv_disk_capacity_bytes != 0) {
        const PagedKVPool& text_pool    = decoder->text_kv.pool();
        const auto* backend             = decoder->mtp_cache();
        const PagedKVPool* backend_pool = backend_kv_pool();
        std::size_t page_bytes          = paged_kv_logical_page_bytes(text_pool);
        if (backend_pool != nullptr) {
            page_bytes = std::max(page_bytes, paged_kv_logical_page_bytes(*backend_pool));
        }
        qwen3::detail::DiskOpenConfig disk;
        disk.location       = plan.kv_disk_location;
        disk.capacity_bytes = kv_disk_capacity_bytes;
        disk.compress       = plan.kv_disk_compress;
        disk.max_context    = capacity;
        disk.ram            = kv_ram_cache_ ? &*kv_ram_cache_ : nullptr;
        disk.fingerprint    = qwen3::detail::make_disk_fingerprint(
            plan.model_id, plan.weights_id, plan.artifact_tensor_identity, speculative_backend,
            text_pool, backend_pool, decoder->text_kv.fingerprint(),
            backend != nullptr ? std::optional(backend->fingerprint()) : std::nullopt,
            &decoder->linear_attention, dflash ? &dflash->local : nullptr);
        disk.text_pool          = &text_pool;
        disk.backend_pool       = backend_pool;
        disk.logical_page_bytes = page_bytes;
        disk.hidden_bytes       = sequences[0].tail_hidden.bytes();
        disk.restore_io_threads = qwen3::detail::kKvDiskRestoreIoThreads;
        kv_disk_cache_.emplace(std::move(disk));
    }
    // Last: the destructor does not run for a failed construction, so a failure here destroys
    // the events already created.
    try {
        if (staging_hidden.data != nullptr) {
            qwen3::detail::create_cache_hip_event(&staging_.d2d_done, hipEventDisableTiming);
            qwen3::detail::create_cache_hip_event(&staging_.copies_done, hipEventDisableTiming);
        }
        qwen3::detail::create_cache_hip_event(&copy_hold_cancel_fence_, hipEventDisableTiming);
    } catch (...) {
        if (staging_.copies_done != nullptr) {
            (void)hipEventDestroy(staging_.copies_done);
            staging_.copies_done = nullptr;
        }
        if (staging_.d2d_done != nullptr) {
            (void)hipEventDestroy(staging_.d2d_done);
            staging_.d2d_done = nullptr;
        }
        throw;
    }
}

ProgramImplCore::~ProgramImplCore() noexcept {
    try {
        shutdown_kv_tiers();
        // NOLINTNEXTLINE(bugprone-empty-catch): persistence is best-effort at teardown
    } catch (...) {}
    fence_staging_copies();
    if (staging_.d2d_done != nullptr) {
        (void)hipEventDestroy(staging_.d2d_done);
        staging_.d2d_done = nullptr;
    }
    if (staging_.copies_done != nullptr) {
        (void)hipEventDestroy(staging_.copies_done);
        staging_.copies_done = nullptr;
    }
    if (copy_hold_cancel_fence_ != nullptr) {
        (void)hipEventSynchronize(copy_hold_cancel_fence_);
        (void)hipEventDestroy(copy_hold_cancel_fence_);
        copy_hold_cancel_fence_ = nullptr;
    }
    for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
        clear_context_checkpoints(sequences[lane]);
    }
    if (device.stream != nullptr) { (void)hipStreamSynchronize(device.stream); }
    if (device.copy_stream != nullptr) { (void)hipStreamSynchronize(device.copy_stream); }
}

bool ProgramImplCore::chunked_execution_enabled() const noexcept {
#if defined(NINFER_QUALIFY_FULL_KV_RESERVATION)
    return false;
#else
    return kv_ram_cache_.has_value();
#endif
}

runtime::ExecutionWindow
ProgramImplCore::execution_window(std::uint32_t prompt_tokens, std::uint32_t output_tokens,
                                  std::size_t identity_extra_bytes, std::size_t checkpoint_count,
                                  runtime::AdmissionResources minimum) const {
    const std::uint32_t frontier = prompt_tokens + (output_tokens ? output_tokens - 1 : 0);
    const auto extent            = [&](std::uint32_t extra) {
        return static_cast<std::uint32_t>(
            std::min<std::uint64_t>(capacity, static_cast<std::uint64_t>(frontier) + extra));
    };
    runtime::ExecutionWindow result;
    result.output_tokens           = output_tokens;
    result.admission.active_lanes  = 1;
    result.admission.main_kv_pages = pages_for_tokens(
        extent(speculative_backend == SpeculativeBackend::DFlash ? dflash_verify_width : 0));
    if (speculative_backend == SpeculativeBackend::Mtp) {
        result.admission.backend_kv_pages = pages_for_tokens(extent(draft_window - 1));
    } else if (speculative_backend == SpeculativeBackend::DFlash) {
        if constexpr (DFlashConfig::full_layers > 0) {
            result.admission.backend_kv_pages = pages_for_tokens(frontier);
        }
    }
    // A settled speculative round may leave mapped tail pages beyond a smaller future
    // output window. Shrinking entitlement never discards those represented current pages.
    result.admission.main_kv_pages =
        std::max(result.admission.main_kv_pages, minimum.main_kv_pages);
    result.admission.backend_kv_pages =
        std::max(result.admission.backend_kv_pages, minimum.backend_kv_pages);
    if (kv_ram_cache_) {
        result.suspension_bytes = qwen3::detail::KVRamCache::live_image_budget(
            decoder->text_kv.pool(), backend_kv_pool(), result.admission.main_kv_pages,
            result.admission.backend_kv_pages, frontier + 1ULL,
            identity_extra_bytes + 13ULL * (frontier + 1ULL),
            decoder->linear_attention.conv_host_image_bytes(),
            decoder->linear_attention.recurrent_host_image_bytes(),
            tail_hidden_store.bytes() / max_concurrency,
            dflash ? dflash->local.lane_host_bytes() : 0, checkpoint_count);
        result.suspension_bytes +=
            static_cast<std::size_t>(TextConfig::token_domain) * sizeof(std::int32_t);
        if (dflash) {
            result.suspension_bytes += dflash->pending_features.bytes() / max_concurrency;
        }
    }
    return result;
}

runtime::AdmissionResources
ProgramImplCore::completion_resources(const RequestBasePlan& base,
                                      std::uint32_t prompt_token_bound) const {
    if (!base.impl_ || prompt_token_bound < base.impl_->summary.prompt_tokens) {
        throw std::invalid_argument("completion bound is shorter than its prepared prompt");
    }
    const auto output   = base.impl_->summary.effective_output_tokens;
    const auto frontier = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(capacity, static_cast<std::uint64_t>(prompt_token_bound) +
                                              (output ? output - 1ULL : 0ULL)));
    return execution_window(frontier, 1, 0, 0).admission;
}

bool ProgramImplCore::shrink_window(std::uint32_t lane, const runtime::ExecutionWindow& window) {
    if (lane >= max_concurrency) { throw std::out_of_range("window lane out of range"); }
    auto& sequence = sequences[lane];
    auto& request  = requests[lane];
    if (request.lifecycle != Lifecycle::Active || request.pending.kind != PendingKind::None ||
        request.prefill || !sequence.kv || window.output_tokens >= request.window_output_tokens ||
        window.output_tokens <= sequence.ledger_frontier - request.prompt_tokens) {
        return false;
    }
    const auto expected = plan_window(lane, window.output_tokens);
    if (expected.admission.main_kv_pages != window.admission.main_kv_pages ||
        expected.admission.backend_kv_pages != window.admission.backend_kv_pages) {
        throw std::logic_error("invalid target window shrink");
    }
    resize_sequence_kv_entitlement(sequence, window.admission.main_kv_pages,
                                   window.admission.backend_kv_pages);
    request.window_output_tokens = window.output_tokens;
    return true;
}

runtime::ExecutionWindow ProgramImplCore::plan_window(std::uint32_t lane,
                                                      std::uint32_t output_tokens) const {
    if (lane >= max_concurrency) { throw std::out_of_range("window lane out of range"); }
    const auto& sequence = sequences[lane];
    const auto& request  = requests[lane];
    if (!sequence.kv || output_tokens > request.effective_output_tokens) {
        throw std::logic_error("window exceeds live request lifetime");
    }
    return execution_window(request.prompt_tokens, output_tokens,
                            sequence.prefix_identity.packed_bytes() -
                                13ULL * sequence.prefix_identity.size(),
                            sequence.context_checkpoints.size(),
                            {1, sequence.kv->text.mapped_page_count(),
                             sequence.kv->backend ? sequence.kv->backend->mapped_page_count() : 0});
}

runtime::ExecutionWindow ProgramImplCore::plan_suspended_window(std::uint64_t owner,
                                                                std::uint32_t output_tokens) const {
    const auto& parked = suspended_requests_.at(owner);
    if (parked.phase != SuspendedRequest::Phase::Parked ||
        output_tokens > parked.request.effective_output_tokens) {
        throw std::logic_error("window exceeds suspended request lifetime");
    }
    return execution_window(parked.request.prompt_tokens, output_tokens,
                            parked.sequence.prefix_identity.packed_bytes() -
                                13ULL * parked.sequence.prefix_identity.size(),
                            parked.checkpoint_count, {1, parked.text_pages, parked.backend_pages});
}

bool ProgramImplCore::grant_window(std::uint32_t lane, const runtime::ExecutionWindow& window) {
    if (lane >= max_concurrency) { throw std::out_of_range("window lane out of range"); }
    auto& sequence      = sequences[lane];
    auto& request       = requests[lane];
    const auto expected = plan_window(lane, window.output_tokens);
    if (expected.admission.main_kv_pages != window.admission.main_kv_pages ||
        expected.admission.backend_kv_pages != window.admission.backend_kv_pages ||
        window.output_tokens < request.window_output_tokens) {
        throw std::logic_error("invalid target window grant");
    }
    const auto can_replace = [](const PagedKVPool& pool, std::uint32_t old_pages,
                                std::uint32_t new_pages) {
        return old_pages <= pool.entitled_pages() && new_pages <= pool.logical_page_capacity() &&
               new_pages <= pool.page_group_count() - (pool.entitled_pages() - old_pages);
    };
    if (!can_replace(decoder->text_kv.pool(), sequence.kv.value().text.page_entitlement(),
                     window.admission.main_kv_pages)) {
        return false;
    }
    if (const auto* backend = backend_kv_pool();
        backend && !can_replace(*backend, sequence.kv.value().backend.value().page_entitlement(),
                                window.admission.backend_kv_pages)) {
        return false;
    }
    try {
        resize_sequence_kv_entitlement(sequence, window.admission.main_kv_pages,
                                       window.admission.backend_kv_pages);
    } catch (const std::bad_alloc&) {
        // The bundle reserves every host page vector before committing any entitlement.
        return false;
    }
    request.window_output_tokens = window.output_tokens;
    return true;
}

bool ProgramImplCore::reserve_suspension_credit(std::uint64_t owner, std::size_t bytes) {
    if (!kv_ram_cache_) { return false; }
    return kv_ram_cache_.value().reserve_live_credit(owner, bytes);
}

void ProgramImplCore::release_suspension_credit(std::uint64_t owner) {
    if (kv_ram_cache_) { kv_ram_cache_.value().release_live_credit(owner); }
}

bool ProgramImplCore::start_suspend(std::uint32_t lane, std::uint64_t owner) {
    if (!kv_ram_cache_ || lane >= max_concurrency || owner == 0) { return false; }
    auto& sequence = sequences[lane];
    auto& request  = requests[lane];
    if ((request.lifecycle != Lifecycle::Active && request.lifecycle != Lifecycle::Complete) ||
        request.prefill || request.pending.kind != PendingKind::None || !sequence.kv ||
        suspended_requests_.contains(owner)) {
        throw std::logic_error("suspension requires a settled active request");
    }
    flush_deferred_gdn_fold(lane);
    device.order_copy_after_compute();
    auto it = suspended_requests_.end();
    try {
        it = suspended_requests_.try_emplace(owner).first;
    } catch (const std::bad_alloc&) { return false; }
    it->second.lane             = lane;
    it->second.checkpoint_count = sequence.context_checkpoints.size();
    const bool was_retained     = sequence.retained;
    try {
        sequence.retained = true;
        qwen3::detail::ResidentPrefixIdentity unused;
        auto source = ram_capture_source(sequence, unused, CacheCaptureMode::CompleteOriginal);
        sequence.retained     = was_retained;
        source.live_owner     = owner;
        it->second.text_pages = source.text_pages.value_or(source.text->mapped_page_count());
        it->second.backend_pages =
            source.backend ? source.backend_pages.value_or(source.backend->mapped_page_count()) : 0;
        const auto [block, bytes] = kv_ram_cache_.value().live_credit_view(owner);
        std::size_t tail_bytes    = 0;
        if (request.sampling_host.token_counts) {
            const std::size_t count_bytes =
                static_cast<std::size_t>(TextConfig::token_domain) * sizeof(std::int32_t);
            tail_bytes += count_bytes;
            if (bytes < tail_bytes) {
                throw std::logic_error("sampling count credit is too small");
            }
            it->second.sampling_counts_present = true;
            it->second.sampling_counts_offset  = bytes - tail_bytes;
            HIP_CHECK(
                hipMemcpyAsync(static_cast<std::byte*>(block) + it->second.sampling_counts_offset,
                               request.sampling_host.token_counts, count_bytes,
                               hipMemcpyDeviceToHost, device.copy_stream));
        }
        if (dflash) {
            const auto features =
                dflash->pending_features.slice(2, static_cast<std::int32_t>(lane), 1);
            tail_bytes += features.bytes();
            if (bytes < tail_bytes) {
                throw std::logic_error("pending DFlash feature credit is too small");
            }
            it->second.pending_features_offset = bytes - tail_bytes;
            HIP_CHECK(hipMemcpyAsync(
                static_cast<std::byte*>(block) + it->second.pending_features_offset, features.data,
                features.bytes(), hipMemcpyDeviceToHost, device.copy_stream));
        }
        source.live_reserved_tail_bytes = tail_bytes;
        auto result                     = kv_ram_cache_.value().capture(source);
        if (result.status != qwen3::detail::RamCaptureStatus::Captured) {
            HIP_CHECK(hipStreamSynchronize(device.copy_stream));
            suspended_requests_.erase(it);
            return false;
        }
        it->second.entry_id = result.entry_id;
        return true;
    } catch (const std::bad_alloc&) {
        device.synchronize_all();
        sequence.retained = was_retained;
        suspended_requests_.erase(it);
        return false;
    } catch (...) {
        (void)hipStreamSynchronize(device.copy_stream);
        sequence.retained = was_retained;
        suspended_requests_.erase(it);
        throw;
    }
}

bool ProgramImplCore::suspension_ready(std::uint64_t owner) const {
    const auto& saved = suspended_requests_.at(owner);
    return kv_ram_cache_.value().live_image_ready(saved.entry_id);
}

void ProgramImplCore::finish_suspend(std::uint32_t lane, std::uint64_t owner) {
    auto& saved = suspended_requests_.at(owner);
    if (saved.phase != SuspendedRequest::Phase::Capturing || saved.lane != lane ||
        !suspension_ready(owner)) {
        throw std::logic_error("suspension capture is not settled");
    }
    auto& sequence = sequences[lane];
    auto& request  = requests[lane];
    unbind_sequence_kv(sequence);
    sequence.kv.reset();
    if ((staging_.occupied || staging_.rewrite) && staging_.lane == lane) { unoccupy_staging(); }
    clear_context_checkpoints(sequence);
    saved.sequence = std::move(sequence);
    saved.request  = std::move(request);
    // Physical tensor slices and rewrite image slots belong to the vacated lane, not the
    // logical parked request. Their contents are already secured in its live RAM image.
    sequence             = SequenceState{};
    sequence.lane        = lane;
    sequence.tail_hidden = std::exchange(saved.sequence.tail_hidden, {});
    sequence.rewrite_checkpoint_hidden =
        std::exchange(saved.sequence.rewrite_checkpoint_hidden, {});
    sequence.rewrite_image = std::move(saved.sequence.rewrite_image);
    request                = RequestControl{};
    saved.phase            = SuspendedRequest::Phase::Parked;
}

bool ProgramImplCore::start_resume(std::uint32_t lane, std::uint64_t owner,
                                   const runtime::ExecutionWindow& window) {
    auto& saved = suspended_requests_.at(owner);
    if (saved.phase != SuspendedRequest::Phase::Parked || lane >= max_concurrency ||
        requests[lane].lifecycle != Lifecycle::Empty || sequences[lane].kv) {
        throw std::logic_error("resume requires a parked request and an empty lane");
    }
    const auto expected = plan_suspended_window(owner, window.output_tokens);
    if (expected.admission.main_kv_pages != window.admission.main_kv_pages ||
        expected.admission.backend_kv_pages != window.admission.backend_kv_pages ||
        window.output_tokens < saved.request.window_output_tokens) {
        throw std::logic_error("invalid suspended window grant");
    }
    if (!decoder->text_kv.pool().can_reserve(window.admission.main_kv_pages) ||
        (backend_kv_pool() && !backend_kv_pool()->can_reserve(window.admission.backend_kv_pages))) {
        return false;
    }
    auto& sequence = sequences[lane];
    // A reused physical lane may still have a rewrite staging identity from its previous
    // logical owner. Its generation can coincide with the parked request's generation.
    if ((staging_.occupied || staging_.rewrite) && staging_.lane == lane) { unoccupy_staging(); }
    try {
        const auto host_view = kv_ram_cache_.value().host_kv(saved.entry_id);
        if (context_checkpoint_pool_.size() < host_view.ladder_images.size()) { return false; }
        reserve_sequence_kv(sequence, window.admission.main_kv_pages,
                            window.admission.backend_kv_pages);
        device.order_copy_after_compute();
        auto& kv = sequence.kv.value();
        kv.text.materialize_pages(host_view.text_pages, device.copy_stream);
        if (kv.backend) {
            kv.backend->materialize_pages(host_view.backend_pages, device.copy_stream);
        }
        qwen3::detail::RamRestoreTarget target;
        target.text_dst_pages    = host_view.text_pages;
        target.backend_dst_pages = host_view.backend_pages;
        target.text              = &kv.text;
        target.text_pool         = &decoder->text_kv.pool();
        target.text_semantics    = decoder->text_kv.fingerprint();
        if (kv.backend) {
            target.backend      = &*kv.backend;
            target.backend_pool = backend_kv_pool();
            if (speculative_backend == SpeculativeBackend::Mtp) {
                target.backend_semantics = decoder->mtp_cache()->fingerprint();
            }
        }
        target.gdn              = &decoder->linear_attention;
        target.gdn_current_slot = LinearStateSlots::current_state_slot(lane, max_concurrency);
        target.rewrite_state    = rewrite_state_host_target(sequence);
        target.tail_hidden      = &sequence.tail_hidden;
        target.rewrite_checkpoint_hidden = &sequence.rewrite_checkpoint_hidden;
        target.reuse                     = ReusePath::AppendAtFrontier;
        target.reuse_base                = saved.sequence.execution_frontier;
        if (dflash) {
            target.dflash_local = &dflash->local;
            target.dflash_lane  = static_cast<std::int32_t>(lane);
        }
        target.stream = device.copy_stream;
        if (saved.sampling_counts_present) {
            const auto [block, bytes] = kv_ram_cache_.value().live_credit_view(owner);
            const std::size_t count_bytes =
                static_cast<std::size_t>(TextConfig::token_domain) * sizeof(std::int32_t);
            if (saved.sampling_counts_offset + count_bytes > bytes) {
                throw std::logic_error("suspended sampling counts exceed their RAM credit");
            }
            Tensor counts = token_counts.slice(1, static_cast<std::int32_t>(lane), 1);
            HIP_CHECK(hipMemcpyAsync(
                counts.data, static_cast<const std::byte*>(block) + saved.sampling_counts_offset,
                count_bytes, hipMemcpyHostToDevice, device.copy_stream));
        }
        if (dflash) {
            const auto [block, bytes] = kv_ram_cache_.value().live_credit_view(owner);
            const auto features =
                dflash->pending_features.slice(2, static_cast<std::int32_t>(lane), 1);
            if (saved.pending_features_offset + features.bytes() > bytes) {
                throw std::logic_error("suspended DFlash features exceed their RAM credit");
            }
            HIP_CHECK(hipMemcpyAsync(
                features.data, static_cast<const std::byte*>(block) + saved.pending_features_offset,
                features.bytes(), hipMemcpyHostToDevice, device.copy_stream));
        }
        auto host = kv_ram_cache_.value().unpack_device(saved.entry_id, target);
        install_ram_context_checkpoints(sequence, host, saved.entry_id,
                                        saved.sequence.execution_frontier, true);
        if (sequence.context_checkpoints.size() != host.ladder_images.size()) {
            throw std::logic_error("live resume must restore every checkpoint image");
        }
        saved.request.window_output_tokens = window.output_tokens;
        saved.lane                         = lane;
        saved.phase                        = SuspendedRequest::Phase::Restoring;
        return true;
    } catch (const std::bad_alloc&) {
        // Fence readers before releasing a partial destination; the source remains parked.
        device.synchronize_all();
        clear_lane(sequence, requests[lane]);
        return false;
    } catch (...) {
        synchronize_all_while_unwinding(device);
        clear_lane(sequence, requests[lane]);
        throw;
    }
}

bool ProgramImplCore::resume_ready(std::uint64_t owner) const {
    const auto& saved = suspended_requests_.at(owner);
    if (!kv_ram_cache_.value().live_image_ready(saved.entry_id)) { return false; }
    for (const auto& head : sequences[saved.lane].context_checkpoints) {
        const auto status = hipEventQuery(head.copies_done);
        if (status == hipErrorNotReady) { return false; }
        HIP_CHECK(status);
    }
    return true;
}

void ProgramImplCore::finish_resume(std::uint32_t lane, std::uint64_t owner) {
    auto& saved = suspended_requests_.at(owner);
    if (saved.phase != SuspendedRequest::Phase::Restoring || saved.lane != lane ||
        !resume_ready(owner)) {
        throw std::logic_error("resume is not settled");
    }
    auto& sequence                     = sequences[lane];
    auto physical                      = std::move(sequence);
    sequence                           = std::move(saved.sequence);
    sequence.lane                      = lane;
    sequence.kv                        = std::move(physical.kv);
    sequence.tail_hidden               = physical.tail_hidden;
    sequence.rewrite_checkpoint_hidden = physical.rewrite_checkpoint_hidden;
    sequence.rewrite_image             = std::move(physical.rewrite_image);
    sequence.context_checkpoints       = std::move(physical.context_checkpoints);
    if (saved.request.lifecycle == Lifecycle::Active) { bind_sequence_kv(sequence); }
    requests[lane]            = std::move(saved.request);
    auto& sampling            = requests[lane].sampling_host;
    sampling.nonfinite_logits = static_cast<std::uint32_t*>(io.nonfinite_logits.data) +
                                lane * qwen3::kDFlashDecodeMaximumWidth;
    if (saved.sampling_counts_present) {
        sampling.token_counts = static_cast<std::int32_t*>(
            token_counts.slice(1, static_cast<std::int32_t>(lane), 1).data);
    }
    requests[lane].prefill_sampling_host = sampling;
    kv_ram_cache_.value().retire_live_image(saved.entry_id);
    suspended_requests_.erase(owner);
}

void ProgramImplCore::discard_suspended(std::uint64_t owner) {
    auto it = suspended_requests_.find(owner);
    if (it == suspended_requests_.end()) { return; }
    auto& saved = it->second;
    kv_ram_cache_.value().wait_live_image(saved.entry_id);
    if (saved.phase == SuspendedRequest::Phase::Restoring) {
        device.synchronize_all();
        clear_lane(sequences[saved.lane], requests[saved.lane]);
    }
    kv_ram_cache_.value().retire_live_image(saved.entry_id);
    suspended_requests_.erase(it);
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
    const PagedKVPool* backend = backend_kv_pool();
    if (backend == nullptr) { return plan.impl_->backend_kv_page_entitlement == 0; }
    const std::uint32_t old_backend =
        sequence.kv && sequence.kv->backend ? sequence.kv->backend->page_entitlement() : 0;
    return can_replace(*backend, old_backend, plan.impl_->backend_kv_page_entitlement);
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
        reclaimable_text += candidate.kv->text.page_entitlement();
        if (candidate.kv->backend) {
            reclaimable_backend += candidate.kv->backend->page_entitlement();
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

    const PagedKVPool* backend = backend_kv_pool();
    if (backend == nullptr) { return plan.impl_->backend_kv_page_entitlement == 0; }
    const std::uint32_t old_backend =
        sequence.kv && sequence.kv->backend ? sequence.kv->backend->page_entitlement() : 0;
    return can_replace(*backend, old_backend, reclaimable_backend,
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
        reclaimable_text += candidate.kv->text.page_entitlement();
        if (candidate.kv->backend) {
            reclaimable_backend += candidate.kv->backend->page_entitlement();
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

    const PagedKVPool* backend = backend_kv_pool();
    if (backend == nullptr) { return plan.impl_->backend_kv_page_entitlement == 0; }
    const std::uint32_t old_backend =
        sequence.kv && sequence.kv->backend ? sequence.kv->backend->page_entitlement() : 0;
    return can_replace(*backend, old_backend, reclaimable_backend,
                       plan.impl_->backend_kv_page_entitlement);
}

runtime::AdmissionResources ProgramImplCore::admission_capacity() const noexcept {
    const PagedKVPool* backend = backend_kv_pool();
    return runtime::AdmissionResources{
        .active_lanes     = max_concurrency,
        .main_kv_pages    = decoder->text_kv.pool().page_group_count(),
        .backend_kv_pages = backend != nullptr ? backend->page_group_count() : 0U,
    };
}

// The plan is consumed through plan.impl_: its vision schedule moves into the staged prefill.
runtime::PrefillStepResult ProgramImplCore::start_prefill_lane(
    std::uint32_t lane, PreparedPromptData&& prompt,
    // NOLINTNEXTLINE(cppcoreguidelines-rvalue-reference-param-not-moved): see above
    RequestPlan&& plan, runtime::TransientRegion transient, const qwen3::OutputSession* output,
    bool decode_waiting) {
    layer_boundary_trace::require_eager(use_device_graph);
    if (lane >= max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    // Admission may reuse or snapshot any lane's state.
    flush_deferred_gdn_folds();
    SequenceState& sequence                    = sequences[lane];
    RequestControl& request                    = requests[lane];
    request.captured_context_checkpoint_tokens = 0;
    request.restored_context_checkpoint_tokens = 0;
    if (plan.impl_ == nullptr) { throw std::invalid_argument("request plan is empty"); }
    RequestPlanImpl& request_plan = *plan.impl_;
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
        ((!sequence.retained && !sequence.fork_initialized) ||
         !qwen3::detail::prefix_matches(prompt, sequence.ledger, sequence.prefix_identity,
                                        request_plan.reuse_base))) {
        throw std::logic_error("planned resident prefix is no longer reusable");
    }
    if (sequence.fork_initialized &&
        (request_plan.cache_read_intent != runtime::CacheReadIntent::Fork ||
         request_plan.reuse_source != PrefixReuseSource::VramResident ||
         sequence.execution_frontier != request_plan.reuse_base ||
         sequence.text_kv_publication.valid_frontier != request_plan.reuse_base ||
         !sequence.tail_hidden_valid || sequence.disk_entry_id != 0)) {
        throw std::logic_error("initialized fork does not describe the selected current state");
    }
    if (!sequence.fork_initialized && is_rewrite_checkpoint_restore(request_plan.reuse) &&
        (!sequence.rewrite_checkpoint.valid ||
         sequence.rewrite_checkpoint.frontier != request_plan.reuse_base ||
         request_plan.reuse != restore_path(sequence.rewrite_checkpoint.kind))) {
        throw std::logic_error("planned rewrite checkpoint is unavailable");
    }
    if (!sequence.fork_initialized &&
        qwen3::detail::is_staged_checkpoint_restore(request_plan.reuse)) {
        const auto head =
            std::find_if(sequence.context_checkpoints.begin(), sequence.context_checkpoints.end(),
                         [&](const ContextCheckpointHead& candidate) {
                             return candidate.frontier == request_plan.reuse_base;
                         });
        const bool want_rollback = request_plan.reuse == ReusePath::RestoreTurnRollback;
        if (head == sequence.context_checkpoints.end() ||
            (head->kind == qwen3::detail::ContextCheckpointKind::TurnRollback) != want_rollback ||
            !qwen3::detail::prefix_matches(prompt, sequence.ledger, sequence.prefix_identity,
                                           request_plan.reuse_base)) {
            throw std::logic_error("planned context checkpoint is unavailable");
        }
    }
    if (request_plan.rewrite_checkpoint_action == RewriteCheckpointAction::KeepExisting &&
        (!prompt.identity.rewrite_checkpoint || !sequence.rewrite_checkpoint.valid ||
         sequence.rewrite_checkpoint.kind != prompt.identity.rewrite_checkpoint->kind ||
         sequence.rewrite_checkpoint.frontier != prompt.identity.rewrite_checkpoint->frontier ||
         request_plan.reuse == ReusePath::FullReset ||
         !qwen3::detail::prefix_matches(prompt, sequence.ledger, sequence.prefix_identity,
                                        sequence.rewrite_checkpoint.frontier))) {
        throw std::logic_error("planned rewrite checkpoint retention is unavailable");
    }
    if (request_plan.rewrite_checkpoint_action == RewriteCheckpointAction::ReclassifyExisting &&
        (!prompt.identity.rewrite_checkpoint || !sequence.rewrite_checkpoint.valid ||
         sequence.rewrite_checkpoint.kind == prompt.identity.rewrite_checkpoint->kind ||
         sequence.rewrite_checkpoint.frontier != prompt.identity.rewrite_checkpoint->frontier ||
         request_plan.reuse == ReusePath::FullReset ||
         !qwen3::detail::prefix_matches(prompt, sequence.ledger, sequence.prefix_identity,
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
        request_plan.cache_read_intent == runtime::CacheReadIntent::Consume &&
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
    request.lifecycle        = Lifecycle::Empty;
    sequence.retained        = false;
    sequence.turn_closed     = false;
    sequence.cache_retention = request_plan.cache_read_intent == runtime::CacheReadIntent::Fork
                                   ? runtime::CacheRetention::Disposable
                                   : runtime::CacheRetention::Ordinary;
    if (sequence.cache_retention == runtime::CacheRetention::Disposable) {
        sequence.disk_entry_id = 0;
    }
    sequence.closure_frontier =
        request_plan.cache_read_intent == runtime::CacheReadIntent::Consume &&
                prompt.identity.rewrite_checkpoint &&
                prompt.identity.rewrite_checkpoint->kind == RewriteCheckpointKind::TurnClosure &&
                prompt.identity.rewrite_checkpoint->generation_opener
            ? prompt.identity.rewrite_checkpoint->frontier
            : 0;
    try {
        if (sequence.fork_initialized) {
            sequence.fork_initialized = false;
            resize_sequence_kv_entitlement(sequence, request_plan.text_kv_page_entitlement,
                                           request_plan.backend_kv_page_entitlement);
        } else if (request_plan.reuse == ReusePath::FullReset) {
            sequence.kv.reset();
            clear_context_checkpoints(sequence);
            ordered_reset(sequence);
            sequence.ledger.clear();
            sequence.prefix_hashes.clear();
            sequence.text_kv_publication = {};
            sequence.mtp_kv_publication  = {};
            reserve_sequence_kv(sequence, request_plan.text_kv_page_entitlement,
                                request_plan.backend_kv_page_entitlement);
        } else if (request_plan.reuse == ReusePath::AppendAtFrontier) {
            if (!sequence.kv) {
                throw std::logic_error("resident prefix has no KV allocation bundle");
            }
            if (sequence.text_kv_publication.valid_frontier < base) {
                throw std::logic_error("resident Text KV is shorter than the append frontier");
            }
            if (speculative_backend == SpeculativeBackend::Mtp) {
                const std::uint32_t mtp_base = base == 0 ? 0 : base - 1;
                if (!request_plan.prepare_mtp ||
                    sequence.mtp_kv_publication.valid_frontier < mtp_base) {
                    throw std::logic_error("resident MTP KV is shorter than the bridge frontier");
                }
                truncate_sequence_mtp_publication(sequence, mtp_base);
            } else if (speculative_backend == SpeculativeBackend::DFlash &&
                       sequence.dflash_context_frontier != base) {
                throw std::logic_error("resident DFlash context is not at the append frontier");
            }
            trim_sequence_kv(sequence, base, backend_kv_valid(sequence));
            resize_sequence_kv_entitlement(sequence, request_plan.text_kv_page_entitlement,
                                           request_plan.backend_kv_page_entitlement);
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
            if (!sequence.kv || sequence.text_kv_publication.valid_frontier < base) {
                throw std::logic_error("resident rewrite checkpoint has no complete KV allocation");
            }
            if (speculative_backend == SpeculativeBackend::Mtp) {
                const std::uint32_t mtp_base = base == 0 ? 0 : base - 1;
                if (!request_plan.prepare_mtp ||
                    sequence.mtp_kv_publication.valid_frontier < mtp_base) {
                    throw std::logic_error(
                        "rewrite-checkpoint MTP KV is shorter than the bridge frontier");
                }
                truncate_sequence_mtp_publication(sequence, mtp_base);
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
        } else if (qwen3::detail::is_staged_checkpoint_restore(request_plan.reuse)) {
            if (!sequence.kv || sequence.text_kv_publication.valid_frontier < base) {
                throw std::logic_error("resident context checkpoint has no complete KV allocation");
            }
            if (speculative_backend == SpeculativeBackend::Mtp) {
                const std::uint32_t mtp_base = base == 0 ? 0 : base - 1;
                if (!request_plan.prepare_mtp ||
                    sequence.mtp_kv_publication.valid_frontier < mtp_base) {
                    throw std::logic_error(
                        "context-checkpoint MTP KV is shorter than the bridge frontier");
                }
                truncate_sequence_mtp_publication(sequence, mtp_base);
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
            if (qwen3::detail::occupy_drops_rewrite_ahead_of_restore(
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

        if (request_plan.cache_read_intent == runtime::CacheReadIntent::Fork) {
            clear_context_checkpoints(sequence);
            sequence.rewrite_checkpoint = {};
        }
        trim_sequence_kv(sequence, base, backend_kv_valid(sequence));
        bind_sequence_kv(sequence);
        decoder->text_kv.truncate_publication(sequence.kv.value().text,
                                              sequence.text_kv_publication, base);
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
            qwen3::detail::is_staged_checkpoint_restore(request_plan.reuse) ? base : 0;
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
        }

        const bool host_input_consumed = prompt.has_media() && !request_plan.vision;
        if (host_input_consumed) { prompt.release_media_payload(); }

        RequestControl::Prefill prefill{
            .prompt                         = std::move(prompt),
            .vision_plan                    = std::move(request_plan.vision),
            .vision                         = nullptr,
            .transient                      = transient,
            .rewrite_checkpoint_capture     = request_plan.rewrite_checkpoint_capture,
            .base                           = base,
            .cursor                         = base,
            .prompt_tokens                  = prompt_tokens,
            .initial_mtp_extent             = initial_mtp_extent,
            .elapsed_seconds                = 0.0,
            .host_input_consumed_pending    = host_input_consumed,
            .prepare_mtp                    = request_plan.prepare_mtp,
            .reuse                          = request_plan.reuse,
            .reuse_source                   = request_plan.reuse_source,
            .system_prompt_cache_hit_tokens = request_plan.system_prompt_cache_hit_tokens,
            .mtp_bridge                     = request_plan.mtp_bridge,
            .capture_context_checkpoints    = request_plan.capture_context_checkpoints,
        };
        request.prefill.emplace(std::move(prefill));
        request.prompt_tokens           = prompt_tokens;
        request.window_output_tokens    = request_plan.summary.window_output_tokens;
        request.effective_output_tokens = request_plan.summary.effective_output_tokens;
        auto& staged                    = *request.prefill;
        if (request_plan.capture_system_prompt && staged.prompt.system_prompt_frontier &&
            *staged.prompt.system_prompt_frontier > base) {
            staged.system_prompt_capture = staged.prompt.system_prompt_frontier;
        }
        if (staged.vision_plan) {
            staged.vision = std::make_unique<schedule::VisionPrefillSession>(
                device, model, work, staged.prompt, *staged.vision_plan, staged.transient);
        }
        staged.elapsed_seconds = std::chrono::duration<double>(Clock::now() - started).count();
        request.lifecycle      = Lifecycle::Prefilling;
        if (decode_waiting && mixed_forward != 0 && prefill_mixable(lane)) {
            // The first slice joins the next decode round's target forward (mixed round)
            // instead of running alone while the decode lanes wait.
            sequence.use_tick = next_use_tick_++;
            return runtime::PrefillStepResult{};
        }
        const runtime::PrefillStepResult first = advance_prefill(sequence, request, decode_waiting);
        sequence.use_tick                      = next_use_tick_++;
        return first;
    } catch (...) {
        synchronize_all_while_unwinding(device);
        clear_lane(sequence, request);
        throw;
    }
}

runtime::PrefillStepResult ProgramImplCore::advance_prefill_lane(std::uint32_t lane,
                                                                 bool decode_waiting) {
    if (lane >= max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    flush_deferred_gdn_fold(lane);
    return advance_prefill(sequences[lane], requests[lane], decode_waiting);
}

void ProgramImplCore::resolve_prefill_lane(std::uint32_t lane, bool terminal) {
    if (lane >= max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    flush_deferred_gdn_fold(lane);
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
    std::array<std::uint32_t, kMaximumConcurrency> defer_rows{};
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
            sequence.prefix_identity.size() != pending.base_S || pending.text_kv_appended == 0 ||
            sequence.text_kv_publication.valid_frontier !=
                pending.base_E + pending.text_kv_appended ||
            (speculative_backend == SpeculativeBackend::Mtp &&
             sequence.mtp_kv_publication.valid_frontier != pending.base_E + pending.produced) ||
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
        // A continuing chain row folds inside its next verification forward instead
        // (DFlashDecodeIngress::gdn_fold); every other row folds here.
        if (speculative_backend == SpeculativeBackend::DFlash && !tree_fold && committed > 0 &&
            !terminal[row]) {
            fold_rows[row].commit_columns = 0;
            defer_rows[row]               = committed;
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
    // same stream, so the host prepares that round while they run. Only a context append's
    // ingress copy still reads the pinned staging the next round rewrites.
    bool ingress_copy_pending = false;
    try {
        // Target verification publishes every represented candidate column before tokens leave
        // Program. Resolve that temporary suffix through the typed codec before advancing the
        // logical sequence: compact an accepted chain/tree path, or truncate a cancelled suffix.
        auto* transaction_status = static_cast<std::uint32_t*>(io.text_kv_status.data);
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence         = sequences[lanes[row]];
            const PendingCandidate& pending = requests[lanes[row]].pending;
            const std::uint32_t committed   = cancelled[row] ? 0U : accepted_tokens[row];
            const std::uint32_t desired     = pending.base_E + committed;
            if (!sequence.kv) {
                throw std::logic_error("speculative Text KV resolution has no allocation");
            }
            if (speculative_backend == SpeculativeBackend::Mtp) {
                if (decoder->mtp_cache() == nullptr || !sequence.kv->backend) {
                    throw std::logic_error("speculative MTP KV resolution has no allocation");
                }
                decoder->mtp_cache()->truncate_publication(*sequence.kv->backend,
                                                           sequence.mtp_kv_publication, desired);
            }
            if (committed == 0U) {
                decoder->text_kv.truncate_publication(sequence.kv->text,
                                                      sequence.text_kv_publication, desired);
                continue;
            }

            if (committed > qwen3::kDFlashDecodeMaximumWidth) {
                throw std::logic_error("speculative Text KV path exceeds fixed staging capacity");
            }
            std::array<std::uint32_t, qwen3::kDFlashDecodeMaximumWidth> selected_path{};
            bool identity_path = true;
            for (std::uint32_t i = 0; i < committed; ++i) {
                const std::int32_t selected =
                    pending.tree_verify ? fold_rows[row].path[i] : static_cast<std::int32_t>(i);
                if (selected < 0) {
                    throw std::runtime_error("speculative Text KV path contains a negative column");
                }
                selected_path[i] = static_cast<std::uint32_t>(selected);
                identity_path    = identity_path && selected_path[i] == i;
            }
            if (identity_path) {
                decoder->text_kv.truncate_publication(sequence.kv->text,
                                                      sequence.text_kv_publication, desired);
                continue;
            }

            std::uint32_t* position_staging = nullptr;
            std::size_t position_capacity   = 0;
            if (speculative_backend == SpeculativeBackend::Mtp && io.mtp_decode) {
                position_capacity = draft_window + 1U;
                position_staging =
                    static_cast<std::uint32_t*>(io.mtp_decode->target_positions.data) +
                    row * position_capacity;
            } else if (speculative_backend == SpeculativeBackend::DFlash && io.dflash_decode) {
                position_capacity = dflash_verify_width;
                position_staging =
                    static_cast<std::uint32_t*>(io.dflash_decode->append_positions.data) +
                    row * position_capacity;
            } else {
                throw std::logic_error("speculative Text KV resolution has no staging frame");
            }
            qwen3::PagedKVTransaction transaction =
                decoder->text_kv.begin_compact(sequence.kv->text, sequence.text_kv_publication,
                                               {.positions         = position_staging,
                                                .position_capacity = position_capacity,
                                                .status            = transaction_status + row});
            for (std::uint32_t layer = 0; layer < decoder->text_kv.layers(); ++layer) {
                transaction.launch_compact_layer(
                    layer, pending.base_E,
                    std::span<const std::uint32_t>(selected_path.data(), committed), device.stream);
            }
            if (transaction.commit() != desired) {
                throw std::logic_error(
                    "speculative Text KV compaction published an unexpected frontier");
            }
        }

        const bool eager_fold = std::any_of(
            fold_rows.begin(), fold_rows.begin() + static_cast<std::ptrdiff_t>(lanes.size()),
            [](const ops::GdnReplayFoldRow& row) {
                return row.path_length > 0 || (row.path_length < 0 && row.commit_columns > 0);
            });
        if (eager_fold) {
            ops::gdn_replay_fold(
                *replay_records, decoder->linear_attention.all_layers_view(),
                std::span<const ops::GdnReplayFoldRow>(fold_rows.data(), lanes.size()),
                device.stream);
        }
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence        = sequences[lanes[row]];
            sequence.deferred_fold_columns = defer_rows[row];
            sequence.deferred_fold_row     = static_cast<std::uint32_t>(row);
        }

        if (needs_hidden_correction) {
            const auto batch = static_cast<std::int32_t>(lanes.size());
            Tensor selector_tensor;
            Tensor hidden;
            Tensor selected;
            Tensor destinations;
            if (speculative_backend == SpeculativeBackend::Mtp && io.mtp_decode) {
                qwen3::MtpDecodeState& frame = *io.mtp_decode;
                selector_tensor              = frame.current_extents.slice(0, 0, batch);
                hidden                       = frame.target_hidden.slice(2, 0, batch);
                selected                     = frame.target_continuation_hidden.slice(1, 0, batch);
                destinations                 = frame.lanes.slice(0, 0, batch);
            } else if (speculative_backend == SpeculativeBackend::DFlash && io.dflash_decode) {
                qwen3::DFlashDecodeState& frame = *io.dflash_decode;
                selector_tensor                 = frame.proposal_extents.slice(0, 0, batch);
                hidden                          = frame.target_hidden.slice(2, 0, batch);
                selected     = frame.target_continuation_hidden.slice(1, 0, batch);
                destinations = frame.lanes.slice(0, 0, batch);
            } else {
                throw std::logic_error("partial speculative commit has no target frame");
            }
            HIP_CHECK(hipMemcpyAsync(selector_tensor.data, hidden_selectors.data(),
                                     lanes.size() * sizeof(std::int32_t), hipMemcpyHostToDevice,
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
                qwen3::DFlashDecodeState& frame = *io.dflash_decode;
                const auto batch                = static_cast<std::int32_t>(lanes.size());
                Tensor compact_counts           = frame.append_counts.slice(0, 0, batch);
                std::array<std::int32_t, kMaximumConcurrency> host_counts{};
                for (std::size_t row = 0; row < lanes.size(); ++row) {
                    host_counts[row] =
                        cancelled[row] ? 0 : static_cast<std::int32_t>(accepted_tokens[row]);
                }
                HIP_CHECK(hipMemcpyAsync(compact_counts.data, host_counts.data(),
                                         lanes.size() * sizeof(std::int32_t), hipMemcpyHostToDevice,
                                         device.stream));
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
            }
        }

        if (ingress_copy_pending) device.synchronize();
        work.reset();
    } catch (...) {
        synchronize_all_while_unwinding(device);
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
            if (sequence.text_kv_publication.valid_frontier != sequence.execution_frontier) {
                throw std::logic_error(
                    "speculative Text KV resolution did not publish the committed frontier");
            }
            sequence.tail_hidden_valid = true;

            trim_speculative_stats_to_commit(request.speculative_stats, pending.produced,
                                             committed);

            if (speculative_backend == SpeculativeBackend::Mtp) {
                if (sequence.mtp_kv_publication.valid_frontier != sequence.execution_frontier) {
                    throw std::logic_error(
                        "speculative MTP KV resolution did not publish the committed frontier");
                }
                if (terminal[row] || committed < pending.produced) {
                    sequence.mtp_draft_count = 0;
                } else {
                    const std::int32_t next  = mtp_host_egress->next_extents[row];
                    sequence.mtp_draft_count = static_cast<std::uint32_t>(next);
                    for (std::uint32_t step = 0; step < sequence.mtp_draft_count; ++step) {
                        sequence.mtp_drafts[step] =
                            mtp_host_egress->next_drafts[step * max_concurrency + row];
                    }
                }
            } else {
                sequence.dflash_context_frontier =
                    terminal[row] ? sequence.execution_frontier : pending.base_E;
            }

            trim_sequence_kv(sequence, sequence.text_kv_publication.valid_frontier,
                             backend_kv_valid(sequence));
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
        synchronize_all_while_unwinding(device);
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
    request.prefill.reset();
    request.pending          = {};
    sequence.mtp_draft_count = 0;
    trim_sequence_kv(sequence, sequence.text_kv_publication.valid_frontier,
                     backend_kv_valid(sequence));
    release_sequence_growth_entitlement(sequence);
    unbind_sequence_kv(sequence);
    sequence.retained = true;
    request.lifecycle = Lifecycle::Complete;
}

bool ProgramImplCore::retain_reusable_lane(std::uint32_t lane) {
    if (lane >= max_concurrency) { return false; }
    flush_deferred_gdn_fold(lane);
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
    flush_deferred_gdn_fold(lane);
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
                staged_path    = qwen3::detail::reuse_path_for_context_checkpoint_kind(head->kind);
            }
        }
    }
    if (frontier == 0 && rewrite_ok) { frontier = sequence.rewrite_checkpoint.frontier; }
    if (frontier == 0) { return false; }

    try {
        if (restore_staged) {
            restore_context_checkpoint_state(sequence, frontier);
            if (qwen3::detail::occupy_drops_rewrite_ahead_of_restore(
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
        decoder->text_kv.truncate_publication(sequence.kv->text, sequence.text_kv_publication,
                                              frontier);
        if (speculative_backend == SpeculativeBackend::Mtp) {
            truncate_sequence_mtp_publication(sequence, frontier == 0 ? 0 : frontier - 1);
        }
        // Checkpoint frontiers are committed Text publication frontiers. RAM capture requires
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
        synchronize_all_while_unwinding(device);
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

bool ProgramImplCore::retained_lane_cache_writable(std::uint32_t lane) const noexcept {
    return has_retained_lane(lane) &&
           sequences[lane].cache_retention == runtime::CacheRetention::Ordinary;
}

bool ProgramImplCore::retained_lane_precedes(std::uint32_t left,
                                             std::uint32_t right) const noexcept {
    const auto& a = sequences[left];
    const auto& b = sequences[right];
    if (a.cache_retention != b.cache_retention) {
        return a.cache_retention == runtime::CacheRetention::Disposable;
    }
    return a.use_tick < b.use_tick || (a.use_tick == b.use_tick && left < right);
}

bool ProgramImplCore::preserve_retained_lane(std::uint32_t lane, std::uint64_t& entry_id,
                                             bool may_block, bool& deferred) {
    if (!retained_lane_cache_writable(lane)) {
        throw std::logic_error("source preservation requires an ordinary retained lane");
    }
    return capture_retained_lane(lane, &entry_id, may_block, &deferred, {},
                                 CacheCaptureMode::CompleteOriginal);
}

void ProgramImplCore::retarget_plan_to_ram(RequestPlan& plan, std::uint64_t entry_id) {
    if (!plan.impl_ || entry_id == 0 ||
        plan.impl_->cache_read_intent != runtime::CacheReadIntent::Fork ||
        plan.impl_->reuse_source != PrefixReuseSource::VramResident) {
        throw std::logic_error(
            "source migration requires a resident fork plan and complete RAM image");
    }
    plan.impl_->reuse_source         = PrefixReuseSource::HostRam;
    plan.impl_->ram_entry_id         = entry_id;
    plan.impl_->summary.reuse_source = PrefixReuseSource::HostRam;
    plan.impl_->summary.ram_entry_id = entry_id;
}

void ProgramImplCore::fork_retained_lane(std::uint32_t destination, std::uint32_t source,
                                         const RequestPlan& plan) {
    if (destination >= max_concurrency || source >= max_concurrency || destination == source ||
        !has_retained_lane(source) || !plan.impl_ || sequences[destination].kv ||
        requests[destination].lifecycle != Lifecycle::Empty) {
        throw std::logic_error(
            "resident fork requires a retained source and empty distinct destination");
    }
    const RequestPlanImpl& selected = *plan.impl_;
    if (selected.cache_read_intent != runtime::CacheReadIntent::Fork || selected.reuse_base == 0 ||
        selected.reuse_source != PrefixReuseSource::VramResident) {
        throw std::logic_error("resident fork requires a winning resident reuse plan");
    }
    auto& from                   = sequences[source];
    auto& to                     = sequences[destination];
    const std::uint32_t frontier = selected.reuse_base;
    if (!from.kv || frontier > from.text_kv_publication.valid_frontier) {
        throw std::logic_error("resident fork exceeds source KV frontier");
    }
    try {
        to.ledger                             = from.ledger;
        to.prefix_identity                    = from.prefix_identity;
        to.ledger_frontier                    = from.ledger_frontier;
        to.rope_delta                         = from.rope_delta;
        to.execution_frontier                 = frontier;
        to.text_kv_publication.valid_frontier = frontier;
        to.mtp_kv_publication.valid_frontier =
            speculative_backend == SpeculativeBackend::Mtp ? frontier - 1 : 0;
        to.dflash_context_frontier =
            speculative_backend == SpeculativeBackend::DFlash ? frontier : 0;
        to.tail_hidden_valid = true;
        to.disk_entry_id     = 0;
        to.cache_retention   = runtime::CacheRetention::Disposable;
        reserve_sequence_kv(to, selected.text_kv_page_entitlement,
                            selected.backend_kv_page_entitlement);
        auto& kv                          = to.kv.value();
        const std::uint32_t text_pages    = ninfer::pages_for_tokens(frontier);
        const std::uint32_t backend_pages = ninfer::pages_for_tokens(backend_kv_valid(to));
        kv.text.materialize_pages(text_pages, device.stream);
        copy_paged_kv_allocation(from.kv->text, kv.text, decoder->text_kv.pool(), text_pages,
                                 device.stream);
        if (kv.backend) {
            if (!from.kv->backend) {
                throw std::logic_error("resident fork backend image is missing");
            }
            kv.backend->materialize_pages(backend_pages, device.stream);
            copy_paged_kv_allocation(*from.kv->backend, *kv.backend, *backend_kv_pool(),
                                     backend_pages, device.stream);
        }
        const std::int32_t current =
            LinearStateSlots::current_state_slot(destination, max_concurrency);
        ContextCheckpointHead* head = nullptr;
        const Tensor* hidden        = &from.tail_hidden;
        if (is_rewrite_checkpoint_restore(selected.reuse)) {
            if (!from.rewrite_checkpoint.valid || from.rewrite_checkpoint.frontier != frontier) {
                throw std::logic_error("resident fork rewrite checkpoint is unavailable");
            }
            head   = &from.rewrite_image;
            hidden = &from.rewrite_checkpoint_hidden;
        } else if (qwen3::detail::is_staged_checkpoint_restore(selected.reuse)) {
            const auto it =
                std::find_if(from.context_checkpoints.begin(), from.context_checkpoints.end(),
                             [frontier](const ContextCheckpointHead& candidate) {
                                 return candidate.frontier == frontier;
                             });
            if (it == from.context_checkpoints.end()) {
                throw std::logic_error("resident fork context checkpoint is unavailable");
            }
            head   = &*it;
            hidden = nullptr;
        } else if (selected.reuse != ReusePath::AppendAtFrontier ||
                   from.execution_frontier != frontier) {
            throw std::logic_error("resident fork current state is not at the selected frontier");
        }
        if (head != nullptr) {
            HIP_CHECK(hipStreamWaitEvent(device.stream, head->copies_done, 0));
            decoder->linear_attention.unpack_slot_from_host(current, head->conv.data(),
                                                            head->recurrent.data(), device.stream);
            if (dflash) {
                dflash->local.copy_lane_from_host(
                    head->dflash.data(), static_cast<std::int32_t>(destination), device.stream);
            }
            if (hidden == nullptr) {
                HIP_CHECK(hipMemcpyAsync(to.tail_hidden.data, head->hidden.data(),
                                         to.tail_hidden.bytes(), hipMemcpyHostToDevice,
                                         device.stream));
            }
            // The executor holds the source until this stream completes. Re-recording also
            // protects the image against a later reuse of its staging/head storage.
            record_context_checkpoint_head_use(*head, device.stream);
        } else {
            flush_deferred_gdn_fold(source);
            decoder->linear_attention.copy_slot_2d(
                LinearStateSlots::current_state_slot(source, max_concurrency), current,
                device.stream);
            if (dflash) {
                dflash->local.copy_lane_from(dflash->local, static_cast<std::int32_t>(source),
                                             static_cast<std::int32_t>(destination), device.stream);
            }
        }
        if (hidden != nullptr) {
            HIP_CHECK(hipMemcpyAsync(to.tail_hidden.data, hidden->data, to.tail_hidden.bytes(),
                                     hipMemcpyDeviceToDevice, device.stream));
        }
        to.deferred_fold_columns = 0;
        to.fork_initialized      = true;
    } catch (const std::bad_alloc&) {
        device.synchronize_all();
        clear_lane(to, requests[destination]);
        throw runtime::CacheRestoreFailure(
            "resident fork destination metadata allocation failed",
            runtime::CacheRestoreFailureKind::DestinationUnavailable);
    } catch (...) {
        synchronize_all_while_unwinding(device);
        clear_lane(to, requests[destination]);
        throw;
    }
}

void ProgramImplCore::evict_retained_lane(std::uint32_t lane) noexcept {
    if (!has_retained_lane(lane)) { return; }
    clear_lane(sequences[lane], requests[lane]);
}

void ProgramImplCore::capture_system_prompt(SequenceState& sequence,
                                            RequestControl::Prefill& staged,
                                            std::uint32_t processed_tokens) {
    const auto frontier = staged.cursor;
    if (!kv_ram_cache_ || frontier == 0 || frontier >= staged.prompt_tokens) { return; }
    if (!sequence.kv) { throw std::logic_error("system capture requires mapped sequence KV"); }
    auto& kv  = *sequence.kv;
    auto& ram = *kv_ram_cache_;
    try {
        qwen3::detail::ResidentPrefixIdentity identity = sequence.prefix_identity;
        identity.truncate(frontier + 1);
        qwen3::detail::RamCaptureSource source;
        source.system_prompt      = true;
        source.execution_frontier = frontier;
        source.ledger_frontier    = frontier + 1;
        source.rope_delta         = sequence.rope_delta;
        source.text_kv_valid      = frontier;
        // MTP column F-1 was prepared with lookahead token F, which belongs to the new user.
        // Keep only the independent columns; BeforeSuffix regenerates the bridge column.
        source.mtp_kv_valid            = staged.prepare_mtp ? frontier - 1 : 0;
        source.dflash_context_frontier = dflash ? frontier : 0;
        source.tail_hidden_valid       = staged.prepare_mtp;
        Tensor mtp_hidden;
        if (staged.prepare_mtp) {
            mtp_hidden =
                prefill_hidden.slice(1, static_cast<std::int32_t>(processed_tokens - 1), 1);
            source.tail_hidden = &mtp_hidden;
        }
        source.ledger   = std::span<const TokenId>(sequence.ledger).first(frontier + 1);
        source.identity = &identity;
        source.hash_f =
            sequence.prefix_hashes.at(sequence.ledger, sequence.prefix_identity, frontier);
        if (ram.has_system_prompt(staged.prompt, source.hash_f)) { return; }
        source.text           = &kv.text;
        source.text_pool      = &decoder->text_kv.pool();
        source.text_semantics = decoder->text_kv.fingerprint();
        source.text_pages     = ninfer::pages_for_tokens(frontier);
        if (kv.backend) {
            source.backend      = &*kv.backend;
            source.backend_pool = backend_kv_pool();
            source.backend_pages =
                ninfer::pages_for_tokens(dflash ? frontier : source.mtp_kv_valid);
            if (speculative_backend == SpeculativeBackend::Mtp) {
                source.backend_semantics = decoder->mtp_cache()->fingerprint();
            }
        }
        source.gdn = &decoder->linear_attention;
        source.gdn_current_slot =
            LinearStateSlots::current_state_slot(sequence.lane, max_concurrency);
        if (dflash) {
            source.dflash_local = &dflash->local;
            source.dflash_lane  = static_cast<std::int32_t>(sequence.lane);
        }
        device.order_copy_after_compute();
        source.stream = device.copy_stream;
        for (;;) {
            const auto captured = ram.capture(source);
            if (captured.status == qwen3::detail::RamCaptureStatus::Captured) {
                // The next compute may overwrite GDN/cyclic state or the last copied KV page.
                // The copy stream owns any H2D readers; compute waits before reusing those bytes.
                ram.wait_entry_copies_on_stream(captured.entry_id, device.stream);
                return;
            }
            if (captured.status == qwen3::detail::RamCaptureStatus::Dropped) { return; }
            if (ram.retired_fence_pending()) {
                ram.record_drop();
                return;
            }
            if (kv_disk_cache_) {
                const auto reclaimed = kv_disk_cache_->reclaim_ram_entry(false, {});
                if (reclaimed == qwen3::detail::RamReclaim::Evicted ||
                    reclaimed == qwen3::detail::RamReclaim::Retry) {
                    continue;
                }
            } else {
                const auto victims = ram.unpinned_ids();
                if (!victims.empty() && ram.evict_one_unpinned(victims.front())) { continue; }
            }
            if (ram.evict_system_prompt()) { continue; }
            ram.record_drop();
            return;
        }
    } catch (const std::bad_alloc&) { ram.record_drop(); }
}

bool ProgramImplCore::capture_cuts_at_rewrite(const SequenceState& sequence) const noexcept {
    const RewriteCheckpoint& checkpoint = sequence.rewrite_checkpoint;
    return sequence.turn_closed && checkpoint.valid &&
           checkpoint.kind == RewriteCheckpointKind::TurnClosure && checkpoint.frontier != 0 &&
           checkpoint.frontier == sequence.closure_frontier &&
           checkpoint.frontier < sequence.execution_frontier &&
           checkpoint.frontier <= sequence.text_kv_publication.valid_frontier &&
           checkpoint.frontier < sequence.ledger.size() &&
           checkpoint.frontier < sequence.prefix_identity.size();
}

qwen3::detail::RamCaptureSource
ProgramImplCore::ram_capture_source(SequenceState& sequence,
                                    qwen3::detail::ResidentPrefixIdentity& cut_identity,
                                    CacheCaptureMode mode) {
    if (!sequence.kv || !sequence.retained) {
        throw std::logic_error("RAM capture requires a retained sequence bundle");
    }
    if (mode == CacheCaptureMode::ReachableTurn && capture_cuts_at_rewrite(sequence)) {
        return cut_ram_capture_source(sequence, cut_identity);
    }
    qwen3::detail::RamCaptureSource source;
    source.execution_frontier      = sequence.execution_frontier;
    source.ledger_frontier         = sequence.ledger_frontier;
    source.rope_delta              = sequence.rope_delta;
    source.text_kv_valid           = sequence.text_kv_publication.valid_frontier;
    source.mtp_kv_valid            = sequence.mtp_kv_publication.valid_frontier;
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
    source.text           = &sequence.kv->text;
    source.text_pool      = &decoder->text_kv.pool();
    source.text_semantics = decoder->text_kv.fingerprint();
    if (sequence.kv->backend) {
        source.backend      = &*sequence.kv->backend;
        source.backend_pool = backend_kv_pool();
        if (speculative_backend == SpeculativeBackend::Mtp) {
            source.backend_semantics = decoder->mtp_cache()->fingerprint();
        }
    }
    source.gdn              = &decoder->linear_attention;
    source.gdn_current_slot = LinearStateSlots::current_state_slot(sequence.lane, max_concurrency);
    source.tail_hidden      = &sequence.tail_hidden;
    if (sequence.rewrite_checkpoint.valid) {
        // The host tier reads the image on the copy stream after its fence.
        const ContextCheckpointHead& image = sequence.rewrite_image;
        source.rewrite_checkpoint_hidden   = &sequence.rewrite_checkpoint_hidden;
        source.rewrite_state               = qwen3::detail::RewriteStateHostSource{
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
qwen3::detail::RamCaptureSource
ProgramImplCore::cut_ram_capture_source(SequenceState& sequence,
                                        qwen3::detail::ResidentPrefixIdentity& cut_identity) {
    const std::uint32_t frontier = sequence.rewrite_checkpoint.frontier;
    cut_identity                 = sequence.prefix_identity;
    cut_identity.truncate(frontier + 1);

    qwen3::detail::RamCaptureSource source;
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
    const SequenceKVBundle& kv = sequence.kv.value();
    source.text                = &kv.text;
    source.text_pool           = &decoder->text_kv.pool();
    source.text_semantics      = decoder->text_kv.fingerprint();
    source.text_pages = std::min(ninfer::pages_for_tokens(frontier), kv.text.mapped_page_count());
    if (kv.backend) {
        source.backend      = &*kv.backend;
        source.backend_pool = backend_kv_pool();
        if (speculative_backend == SpeculativeBackend::Mtp) {
            source.backend_semantics = decoder->mtp_cache()->fingerprint();
        }
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
    source.current_state    = qwen3::detail::RewriteStateHostSource{
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
                                            std::span<const std::uint64_t> attempt_ram_ids,
                                            CacheCaptureMode mode) {
    if (ram_entry_id != nullptr) { *ram_entry_id = 0; }
    if (deferred != nullptr) { *deferred = false; }
    if (!kv_ram_cache_ || !retained_lane_cache_writable(lane)) { return true; }
    qwen3::detail::KVRamCache& ram_cache = *kv_ram_cache_;
    flush_deferred_gdn_fold(lane);
    device.order_copy_after_compute();
    qwen3::detail::RamCaptureSource source;
    qwen3::detail::ResidentPrefixIdentity cut_identity;
    try {
        source = ram_capture_source(sequences[lane], cut_identity, mode);
        for (;;) {
            const auto result = ram_cache.capture(source);
            if (result.status == qwen3::detail::RamCaptureStatus::Captured) {
                if (kv_disk_cache_) {
                    qwen3::detail::KVDiskCache& disk_cache = *kv_disk_cache_;
                    disk_cache.note_ram_resident(result.entry_id, source.disk_entry_id);
                    // Write the new entry behind on the disk worker so that a later
                    // RAM eviction usually finds it durable instead of spilling it
                    // synchronously on this thread. Shutdown flushes in order instead.
                    if (!kv_tiers_shutdown_) { disk_cache.request_idle_spill(); }
                }
                if (ram_entry_id != nullptr) { *ram_entry_id = result.entry_id; }
                return true;
            }
            if (result.status == qwen3::detail::RamCaptureStatus::Dropped) { return false; }
            if (kv_disk_cache_) {
                const auto reclaimed =
                    kv_disk_cache_->reclaim_ram_entry(may_block, attempt_ram_ids);
                if (reclaimed == qwen3::detail::RamReclaim::Evicted ||
                    reclaimed == qwen3::detail::RamReclaim::Retry) {
                    continue;
                }
                // Not a drop: the caller retries once the disk worker's spill lands.
                if (reclaimed == qwen3::detail::RamReclaim::Pending) {
                    if (deferred != nullptr) { *deferred = true; }
                    return false;
                }
                ram_cache.record_drop();
                return false;
            }
            // Without a disk tier the oldest unclaimed entry is dropped. Retired blocks waiting
            // only for their copy fence free before any live entry is dropped: a non-blocking
            // caller defers, a blocking one waits for them.
            if (ram_cache.retired_fence_pending()) {
                if (!may_block && deferred != nullptr) {
                    *deferred = true;
                    return false;
                }
                if (may_block && ram_cache.wait_retired_copies()) { continue; }
            }
            // This admission's own earlier captures are never victims: a deferral rolls them back,
            // so the retry would recapture and target them again.
            std::optional<std::uint64_t> victim;
            for (const std::uint64_t id : ram_cache.unpinned_ids()) {
                if (std::find(attempt_ram_ids.begin(), attempt_ram_ids.end(), id) ==
                    attempt_ram_ids.end()) {
                    victim = id;
                    break;
                }
            }
            if (!victim) {
                ram_cache.record_drop();
                return false;
            }
            if (!may_block && !ram_cache.copies_ready(*victim)) {
                if (deferred != nullptr) { *deferred = true; }
                return false;
            }
            if (may_block) { ram_cache.wait_entry_copies(*victim); }
            (void)ram_cache.evict_one_unpinned(*victim);
        }
    } catch (const std::bad_alloc&) {
        // Optional capture includes the bookkeeping required to reclaim room for it.
        ram_cache.record_drop();
        return false;
    }
}

void ProgramImplCore::fence_staging_copies() noexcept {
    if (staging_.d2d_done != nullptr) { (void)hipEventSynchronize(staging_.d2d_done); }
    if (staging_.copies_done != nullptr) { (void)hipEventSynchronize(staging_.copies_done); }
}

void ProgramImplCore::unoccupy_staging() noexcept {
    staging_.occupied           = false;
    staging_.rewrite            = false;
    staging_.rewrite_generation = 0;
    staging_.lane               = 0;
    staging_.frontier           = 0;
    staging_.hash               = {};
    staging_.kind               = qwen3::detail::ContextCheckpointKind::Ladder;
}

void ProgramImplCore::reload_turn_rollback_into_staging(std::uint32_t lane,
                                                        qwen3::detail::PrefixHash128 hash,
                                                        std::uint32_t frontier) {
    if (lane >= max_concurrency || staging_hidden.data == nullptr) {
        unoccupy_staging();
        return;
    }
    SequenceState& owner = sequences[lane];
    const auto head      = std::find_if(
        owner.context_checkpoints.begin(), owner.context_checkpoints.end(),
        [&](const ContextCheckpointHead& candidate) {
            return candidate.kind == qwen3::detail::ContextCheckpointKind::TurnRollback &&
                   candidate.frontier == frontier && candidate.hash == hash;
        });
    if (head == owner.context_checkpoints.end() || !head->conv || !head->recurrent ||
        !head->hidden) {
        unoccupy_staging();
        return;
    }
    HIP_CHECK(hipStreamWaitEvent(device.copy_stream, head->copies_done, 0));
    if (staging_.d2d_done != nullptr) {
        HIP_CHECK(hipStreamWaitEvent(device.copy_stream, staging_.d2d_done, 0));
    }
    const std::int32_t staging = LinearStateSlots::staging_state_slot(max_concurrency);
    decoder->linear_attention.unpack_slot_from_host(staging, head->conv.data(),
                                                    head->recurrent.data(), device.copy_stream);
    HIP_CHECK(hipMemcpyAsync(staging_hidden.data, head->hidden.data(), staging_hidden.bytes(),
                             hipMemcpyHostToDevice, device.copy_stream));
    HIP_CHECK(hipEventRecord(staging_.copies_done, device.copy_stream));
    record_context_checkpoint_head_use(*head, device.copy_stream);
    staging_.occupied = true;
    staging_.lane     = lane;
    staging_.frontier = frontier;
    staging_.hash     = hash;
    staging_.kind     = qwen3::detail::ContextCheckpointKind::TurnRollback;
}

bool ProgramImplCore::staging_holds(std::uint32_t lane, qwen3::detail::PrefixHash128 hash,
                                    std::uint32_t frontier) const noexcept {
    return qwen3::detail::staging_holds_restore_identity(
        staging_.occupied, staging_.lane, staging_.hash, staging_.frontier, lane, hash, frontier);
}

PinnedImage ProgramImplCore::carve_checkpoint_image(std::size_t bytes) {
    if (bytes == 0) { return {}; }
    if (!checkpoint_image_slab_) { throw std::logic_error("checkpoint image slab is missing"); }
    void* data = checkpoint_image_slab_->try_alloc(bytes, qwen3::detail::kCheckpointImageAlignment);
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
                                                         hipStream_t stream) {
    // Every published head owns its fence before any transfer can borrow it.
    HIP_CHECK(hipEventRecord(head.copies_done, stream));
}

void ProgramImplCore::recycle_context_checkpoint_head(ContextCheckpointHead&& head) noexcept {
    qwen3::detail::recycle_checkpoint_image(context_checkpoint_pool_, std::move(head));
}

void ProgramImplCore::clear_context_checkpoints(SequenceState& sequence) noexcept {
    for (ContextCheckpointHead& head : sequence.context_checkpoints) {
        recycle_context_checkpoint_head(std::move(head));
    }
    sequence.context_checkpoints.clear();
    sequence.next_context_mark = qwen3::detail::first_prefill_context_mark(context_marks);
}

void ProgramImplCore::drop_context_checkpoints_after(SequenceState& sequence,
                                                     std::uint32_t frontier) noexcept {
    auto& heads = sequence.context_checkpoints;
    for (auto it = heads.begin(); it != heads.end();) {
        if (qwen3::detail::retain_context_checkpoint_head(it->frontier, frontier)) {
            ++it;
            continue;
        }
        recycle_context_checkpoint_head(std::move(*it));
        it = heads.erase(it);
    }
    const auto next            = qwen3::detail::next_prefill_context_mark(frontier, context_marks);
    sequence.next_context_mark = next.value_or(0);
}

bool ProgramImplCore::select_imported_checkpoint(std::uint32_t frontier,
                                                 qwen3::detail::ContextCheckpointKind kind,
                                                 std::span<std::uint8_t> selected) const {
    std::size_t index = context_marks.size();
    if (kind == qwen3::detail::ContextCheckpointKind::Ladder) {
        const auto upper = std::upper_bound(context_marks.begin(), context_marks.end(), frontier);
        if (upper == context_marks.begin()) { return false; }
        index = static_cast<std::size_t>(upper - context_marks.begin() - 1);
    }
    if (selected[index]) { return false; }
    selected[index] = 1;
    return true;
}

void ProgramImplCore::install_ram_context_checkpoints(SequenceState& sequence,
                                                      const qwen3::detail::RamRestoredHost& host,
                                                      std::uint64_t entry_id,
                                                      std::uint32_t reuse_base, bool live_restore) {
    if (!kv_ram_cache_) { throw std::logic_error("RAM checkpoint install requires a RAM tier"); }
    qwen3::detail::KVRamCache& ram = *kv_ram_cache_;
    std::vector<ContextCheckpointHead> heads;
    heads.reserve(host.ladder_images.size());
    std::vector<std::uint8_t> selected(context_marks.size() + 1, 0);
    std::vector<HostCopy> copies;
    copies.reserve(4 * host.ladder_images.size());
    std::vector<hipEvent_t> fences;
    fences.reserve(host.ladder_images.size());
    try {
        for (const qwen3::detail::RamLadderImage& image : host.ladder_images) {
            // Occupy keeps only heads at or before the restored base.
            if (image.frontier > reuse_base) { continue; }
            if (!live_restore &&
                !select_imported_checkpoint(image.frontier, image.kind, selected)) {
                continue;
            }
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
    HIP_CHECK(hipStreamWaitEvent(device.stream, head.copies_done, 0));
    if (!head.dflash || head.dflash.size() != dflash->local.lane_host_bytes()) {
        throw std::logic_error("context checkpoint DFlash cyclic image is incomplete");
    }
    dflash->local.copy_lane_from_host(head.dflash.data(), static_cast<std::int32_t>(sequence.lane),
                                      device.stream);
    HIP_CHECK(hipEventRecord(head.copies_done, device.stream));
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
        return;
    }
    if (head == sequence.context_checkpoints.end()) {
        throw std::logic_error("context checkpoint head is missing at restore");
    }
    const std::int32_t current =
        LinearStateSlots::current_state_slot(sequence.lane, max_concurrency);
    if (qwen3::detail::restore_may_d2d_staging(staging_holds(sequence.lane, head->hash, base),
                                               static_cast<bool>(head->conv) &&
                                                   static_cast<bool>(head->recurrent)) &&
        staging_.kind == head->kind && staging_hidden.data != nullptr) {
        if (staging_.copies_done != nullptr) {
            HIP_CHECK(hipStreamWaitEvent(device.stream, staging_.copies_done, 0));
        }
        decoder->linear_attention.copy_slot_2d(
            LinearStateSlots::staging_state_slot(max_concurrency), current, device.stream);
        copy_tail(sequence, staging_hidden);
        restore_dflash_cyclic_from_head(sequence, *head);
        record_context_checkpoint_head_use(*head, device.stream);
        return;
    }
    HIP_CHECK(hipStreamWaitEvent(device.stream, head->copies_done, 0));
    if (!head->conv || !head->recurrent) {
        throw std::logic_error("context checkpoint GDN image is incomplete");
    }
    decoder->linear_attention.unpack_slot_from_host(current, head->conv.data(),
                                                    head->recurrent.data(), device.stream);
    if (sequence.tail_hidden.bytes() != 0 && !head->hidden) {
        throw std::logic_error("context checkpoint hidden image is incomplete");
    }
    if (head->hidden) {
        if (head->hidden.size() != sequence.tail_hidden.bytes()) {
            throw std::logic_error("context checkpoint hidden geometry mismatch");
        }
        HIP_CHECK(hipMemcpyAsync(sequence.tail_hidden.data, head->hidden.data(),
                                 sequence.tail_hidden.bytes(), hipMemcpyHostToDevice,
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

// Called once the prefill chunk that ends exactly at the rewrite frontier has run, so the lane's
// current GDN slot and DFlash cyclic lane hold the checkpoint. MTP/DFlash engines snapshot them
// into the Engine-wide staging slot and lane on the compute stream (a device-local copy), and
// the copy stream drains staging into the lane's pinned image while later prefill and decode
// run. Ordinary engines have no staging slot; their D2H stays on the compute stream ahead of the
// next update of the live slot.
void ProgramImplCore::capture_rewrite_image(SequenceState& sequence) {
    ContextCheckpointHead& image = sequence.rewrite_image;
    const auto lane              = static_cast<std::int32_t>(sequence.lane);
    const std::int32_t current =
        LinearStateSlots::current_state_slot(sequence.lane, max_concurrency);
    ++sequence.rewrite_image_generation;
    if (staging_hidden.data == nullptr) {
        if (dflash) { throw std::logic_error("DFlash rewrite capture requires the staging lane"); }
        // A host-tier callback on the copy stream may still be reading the previous contents.
        HIP_CHECK(hipStreamWaitEvent(device.stream, image.copies_done, 0));
        decoder->linear_attention.pack_slot_to_host(current, image.conv.data(),
                                                    image.recurrent.data(), device.stream);
        record_context_checkpoint_head_use(image, device.stream);
        return;
    }
    unoccupy_staging();
    // Earlier copy-stream readers and writers of staging finish before the snapshot replaces it.
    HIP_CHECK(hipStreamWaitEvent(device.stream, staging_.copies_done, 0));
    const std::int32_t staging = LinearStateSlots::staging_state_slot(max_concurrency);
    decoder->linear_attention.copy_slot(current, staging, device.stream);
    if (dflash) { dflash->staging_local.copy_lane_from(dflash->local, lane, 0, device.stream); }
    HIP_CHECK(hipEventRecord(staging_.d2d_done, device.stream));
    HIP_CHECK(hipStreamWaitEvent(device.copy_stream, staging_.d2d_done, 0));
    // A restore H2D still reading the previous image contents runs on the compute stream.
    HIP_CHECK(hipStreamWaitEvent(device.copy_stream, image.copies_done, 0));
    decoder->linear_attention.pack_slot_to_host(staging, image.conv.data(), image.recurrent.data(),
                                                device.copy_stream);
    if (dflash) {
        dflash->staging_local.copy_lane_to_host(0, image.dflash.data(), device.copy_stream);
    }
    HIP_CHECK(hipEventRecord(image.copies_done, device.copy_stream));
    HIP_CHECK(hipEventRecord(staging_.copies_done, device.copy_stream));
    staging_.rewrite            = true;
    staging_.rewrite_generation = sequence.rewrite_image_generation;
    staging_.lane               = sequence.lane;
}

void ProgramImplCore::restore_rewrite_checkpoint_state(SequenceState& sequence) {
    ContextCheckpointHead& image = sequence.rewrite_image;
    const auto lane              = static_cast<std::int32_t>(sequence.lane);
    const std::int32_t current =
        LinearStateSlots::current_state_slot(sequence.lane, max_concurrency);
    if (staging_.rewrite && staging_.lane == sequence.lane &&
        staging_.rewrite_generation == sequence.rewrite_image_generation) {
        // Staging still holds this image's snapshot (written on this stream): copy it on device.
        decoder->linear_attention.copy_slot(LinearStateSlots::staging_state_slot(max_concurrency),
                                            current, device.stream);
        if (dflash) { dflash->local.copy_lane_from(dflash->staging_local, 0, lane, device.stream); }
        return;
    }
    // Stream-ordered after the capture D2H or host tier write that filled the image.
    HIP_CHECK(hipStreamWaitEvent(device.stream, image.copies_done, 0));
    decoder->linear_attention.unpack_slot_from_host(current, image.conv.data(),
                                                    image.recurrent.data(), device.stream);
    if (dflash) { dflash->local.copy_lane_from_host(image.dflash.data(), lane, device.stream); }
    record_context_checkpoint_head_use(image, device.stream);
}

qwen3::detail::RewriteStateHostTarget
ProgramImplCore::rewrite_state_host_target(SequenceState& sequence) {
    // Tier restores overwrite the image by stream-ordered host copies after its fence and
    // re-record the fence after them, so staging no longer matches it.
    ContextCheckpointHead& image = sequence.rewrite_image;
    ++sequence.rewrite_image_generation;
    return qwen3::detail::RewriteStateHostTarget{
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
    const bool complete = qwen3::detail::prefix_items_complete_at(prompt.vision_items, base);
    // A preserve-off turn (own TurnClosure checkpoint) appending to a cut tier entry is not
    // pinned automatically: that entry is the previous turn cut at its checkpoint, the restore
    // point a pre-cut entry offered through restore_turn_checkpoint without a pin, and tiers
    // store this turn cut at its own checkpoint without later heads.
    const bool auto_rollback = !(cut_restore && sequence.closure_frontier != 0);
    if (!(auto_rollback && qwen3::detail::should_capture_turn_rollback(
                               reuse, base, prompt_tokens, enabled, sequence.tail_hidden_valid,
                               already, complete)) &&
        !qwen3::detail::should_capture_exact_hit_pin(request_pin, base, prompt_tokens, enabled,
                                                     sequence.tail_hidden_valid, already,
                                                     complete)) {
        return;
    }
    if (sequence.ledger.size() < base || sequence.tail_hidden.data == nullptr) { return; }

    const qwen3::detail::PrefixHash128 hash =
        sequence.prefix_hashes.at(sequence.ledger, sequence.prefix_identity, base);
    auto& heads = sequence.context_checkpoints;
    ContextCheckpointHead head;
    const auto existing_rollback =
        std::find_if(heads.begin(), heads.end(), [](const ContextCheckpointHead& existing) {
            return existing.kind == qwen3::detail::ContextCheckpointKind::TurnRollback;
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
    head.kind     = qwen3::detail::ContextCheckpointKind::TurnRollback;
    // Earlier copy-stream reads of staging (a drain or rollback reload) finish before it is
    // overwritten; the compute stream itself wrote d2d_done.
    HIP_CHECK(hipStreamWaitEvent(device.stream, staging_.copies_done, 0));
    unoccupy_staging();
    const std::int32_t current =
        LinearStateSlots::current_state_slot(sequence.lane, max_concurrency);
    const std::int32_t staging = LinearStateSlots::staging_state_slot(max_concurrency);
    decoder->linear_attention.copy_slot_2d(current, staging, device.stream);
    HIP_CHECK(hipMemcpyAsync(staging_hidden.data, sequence.tail_hidden.data, staging_hidden.bytes(),
                             hipMemcpyDeviceToDevice, device.stream));
    snapshot_dflash_cyclic_to_staging(static_cast<std::int32_t>(sequence.lane));
    HIP_CHECK(hipEventRecord(staging_.d2d_done, device.stream));
    staging_.occupied = true;
    staging_.lane     = sequence.lane;
    staging_.frontier = base;
    staging_.hash     = hash;
    staging_.kind     = qwen3::detail::ContextCheckpointKind::TurnRollback;
    HIP_CHECK(hipStreamWaitEvent(device.copy_stream, staging_.d2d_done, 0));
    // The head's previous reader or writer (a tier copy or restore) finishes before the refill.
    HIP_CHECK(hipStreamWaitEvent(device.copy_stream, head.copies_done, 0));
    decoder->linear_attention.pack_slot_to_host(staging, head.conv.data(), head.recurrent.data(),
                                                device.copy_stream);
    HIP_CHECK(hipMemcpyAsync(head.hidden.data(), staging_hidden.data, staging_hidden.bytes(),
                             hipMemcpyDeviceToHost, device.copy_stream));
    pack_dflash_cyclic_to_head(head);
    HIP_CHECK(hipEventRecord(head.copies_done, device.copy_stream));
    HIP_CHECK(hipEventRecord(staging_.copies_done, device.copy_stream));
    heads.push_back(std::move(head));
    request.captured_context_checkpoint_tokens = base;
}

void ProgramImplCore::maybe_freeze_context_checkpoint(SequenceState& sequence,
                                                      RequestControl& request,
                                                      std::uint32_t chunk_tokens) {
    if (!request.prefill) { return; }
    RequestControl::Prefill& staged = *request.prefill;
    const std::uint32_t frontier    = staged.cursor;
    const bool capture_enabled = staged.capture_context_checkpoints &&
                                 captures_context_checkpoints() && staging_hidden.data != nullptr &&
                                 chunk_tokens != 0;
    const bool already         = std::any_of(
        sequence.context_checkpoints.begin(), sequence.context_checkpoints.end(),
        [frontier](const ContextCheckpointHead& head) { return head.frontier == frontier; });
    if (!qwen3::detail::should_freeze_prefill_context_checkpoint(
            true, capture_enabled, frontier, sequence.next_context_mark, already,
            qwen3::detail::prefix_items_complete_at(staged.prompt.vision_items, frontier))) {
        return;
    }
    if (sequence.ledger.size() < frontier) { return; }

    const qwen3::detail::PrefixHash128 hash =
        sequence.prefix_hashes.at(sequence.ledger, sequence.prefix_identity, frontier);
    try {
        sequence.context_checkpoints.reserve(sequence.context_checkpoints.size() + 1);
    } catch (const std::bad_alloc&) { return; }
    std::optional<ContextCheckpointHead> acquired = acquire_context_checkpoint_head();
    if (!acquired) { return; }
    ContextCheckpointHead head = std::move(*acquired);
    head.frontier              = qwen3::detail::advertised_context_checkpoint_frontier(frontier);
    head.hash                  = hash;
    head.kind                  = qwen3::detail::ContextCheckpointKind::Ladder;

    // Earlier copy-stream reads of staging finish before the compute stream overwrites it.
    HIP_CHECK(hipStreamWaitEvent(device.stream, staging_.copies_done, 0));
    const bool reload_rollback =
        staging_.occupied && staging_.kind == qwen3::detail::ContextCheckpointKind::TurnRollback;
    const std::uint32_t saved_lane                = staging_.lane;
    const std::uint32_t saved_frontier            = staging_.frontier;
    const qwen3::detail::PrefixHash128 saved_hash = staging_.hash;
    unoccupy_staging();

    const std::int32_t current =
        LinearStateSlots::current_state_slot(sequence.lane, max_concurrency);
    const std::int32_t staging = LinearStateSlots::staging_state_slot(max_concurrency);
    decoder->linear_attention.copy_slot_2d(current, staging, device.stream);
    const Tensor last_hidden =
        prefill_hidden.slice(1, static_cast<std::int32_t>(chunk_tokens) - 1, 1);
    HIP_CHECK(hipMemcpyAsync(staging_hidden.data, last_hidden.data, staging_hidden.bytes(),
                             hipMemcpyDeviceToDevice, device.stream));
    snapshot_dflash_cyclic_to_staging(static_cast<std::int32_t>(sequence.lane));
    HIP_CHECK(hipEventRecord(staging_.d2d_done, device.stream));
    if (!reload_rollback) {
        staging_.occupied = true;
        staging_.lane     = sequence.lane;
        staging_.frontier = head.frontier;
        staging_.hash     = hash;
        staging_.kind     = qwen3::detail::ContextCheckpointKind::Ladder;
    }
    HIP_CHECK(hipStreamWaitEvent(device.copy_stream, staging_.d2d_done, 0));
    // The recycled head's previous reader or writer finishes before the refill.
    HIP_CHECK(hipStreamWaitEvent(device.copy_stream, head.copies_done, 0));
    decoder->linear_attention.pack_slot_to_host(staging, head.conv.data(), head.recurrent.data(),
                                                device.copy_stream);
    HIP_CHECK(hipMemcpyAsync(head.hidden.data(), staging_hidden.data, staging_hidden.bytes(),
                             hipMemcpyDeviceToHost, device.copy_stream));
    pack_dflash_cyclic_to_head(head);
    HIP_CHECK(hipEventRecord(head.copies_done, device.copy_stream));
    HIP_CHECK(hipEventRecord(staging_.copies_done, device.copy_stream));
    sequence.context_checkpoints.push_back(std::move(head));
    request.captured_context_checkpoint_tokens = sequence.context_checkpoints.back().frontier;
    sequence.next_context_mark =
        qwen3::detail::next_prefill_context_mark(frontier, context_marks).value_or(0);
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
    qwen3::detail::KVRamCache& ram_cache = *kv_ram_cache_;
    flush_deferred_gdn_fold(lane);
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
        reserve_sequence_kv(sequence, request_plan.text_kv_page_entitlement,
                            request_plan.backend_kv_page_entitlement);
        SequenceKVBundle& kv           = sequence.kv.value();
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

        qwen3::detail::RamRestoreTarget target;
        target.text_dst_pages    = text_pages;
        target.backend_dst_pages = backend_pages;
        target.text              = &kv.text;
        target.text_pool         = &decoder->text_kv.pool();
        target.text_semantics    = decoder->text_kv.fingerprint();
        if (kv.backend) {
            target.backend      = &*kv.backend;
            target.backend_pool = backend_kv_pool();
            if (speculative_backend == SpeculativeBackend::Mtp) {
                target.backend_semantics = decoder->mtp_cache()->fingerprint();
            }
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

        qwen3::detail::RamRestoredHost host = ram_cache.unpack_device(entry_id, target);
        sequence.execution_frontier         = host.execution_frontier;
        sequence.ledger_frontier            = host.ledger_frontier;
        sequence.rope_delta                 = host.rope_delta;
        sequence.text_kv_publication        = {.valid_frontier = host.text_kv_valid};
        sequence.mtp_kv_publication         = {.valid_frontier = host.mtp_kv_valid};
        sequence.dflash_context_frontier    = host.dflash_context_frontier;
        sequence.tail_hidden_valid          = host.tail_hidden_valid;
        sequence.rewrite_checkpoint         = RewriteCheckpoint{
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
        if (qwen3::detail::is_staged_checkpoint_restore(request_plan.reuse)) {
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
            if (qwen3::detail::occupy_drops_rewrite_ahead_of_restore(
                    request_plan.reuse, sequence.rewrite_checkpoint.valid,
                    sequence.rewrite_checkpoint.frontier, request_plan.reuse_base)) {
                sequence.rewrite_checkpoint = {};
            }
        }
    } catch (const std::bad_alloc&) {
        // Partial cache H2D must finish before its destination pages are released.
        // A HIP failure remains fatal; only optional host metadata is retried.
        device.synchronize_all();
        clear_lane(sequence, request);
        throw runtime::CacheRestoreFailure(
            "RAM cache restore metadata allocation failed",
            runtime::CacheRestoreFailureKind::DestinationUnavailable);
    } catch (...) {
        synchronize_all_while_unwinding(device);
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

void ProgramImplCore::finish_ram_restore(std::uint64_t entry_id, runtime::CacheReadIntent intent) {
    if (!kv_ram_cache_) {
        throw std::logic_error("RAM restore completion requires an enabled RAM tier");
    }
    if (kv_ram_cache_->is_system_prompt(entry_id)) { intent = runtime::CacheReadIntent::Fork; }
    if (intent == runtime::CacheReadIntent::Consume) {
        kv_ram_cache_->consume(entry_id);
    } else {
        kv_ram_cache_->release_restored(entry_id);
    }
    if (intent == runtime::CacheReadIntent::Consume && kv_disk_cache_) {
        kv_disk_cache_->forget_ram_resident(entry_id);
    }
}

qwen3::detail::KvRamSnapshot ProgramImplCore::kv_ram_snapshot() const noexcept {
    return kv_ram_cache_ ? kv_ram_cache_->snapshot() : qwen3::detail::KvRamSnapshot{};
}

qwen3::detail::KvRamCopySeconds ProgramImplCore::harvest_kv_ram_copy_seconds() {
    return kv_ram_cache_ ? kv_ram_cache_->harvest_copy_seconds()
                         : qwen3::detail::KvRamCopySeconds{};
}

// NOLINTNEXTLINE(misc-anonymous-namespace-in-header): single-TU fragment (runtime.hip)
namespace {

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

} // namespace

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
    qwen3::detail::KVDiskCache& disk_cache = *kv_disk_cache_;
    flush_deferred_gdn_fold(lane);
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
        reserve_sequence_kv(sequence, request_plan.text_kv_page_entitlement,
                            request_plan.backend_kv_page_entitlement);
        SequenceKVBundle& kv        = sequence.kv.value();
        std::uint32_t text_pages    = 0;
        std::uint32_t backend_pages = 0;
        disk_reuse_pages(speculative_backend, kv.backend.has_value(), request_plan.reuse_base,
                         text_pages, backend_pages);
        kv.text.materialize_pages(text_pages, device.copy_stream);
        if (kv.backend) { kv.backend->materialize_pages(backend_pages, device.copy_stream); }

        qwen3::detail::DiskRestoreTarget target;
        target.text_semantics = decoder->text_kv.fingerprint();
        if (decoder->mtp_cache() != nullptr) {
            target.backend_semantics = decoder->mtp_cache()->fingerprint();
        }
        target.text_dst_pages    = text_pages;
        target.backend_dst_pages = backend_pages;
        target.text              = &kv.text;
        target.text_pool         = &decoder->text_kv.pool();
        if (kv.backend) {
            target.backend      = &*kv.backend;
            target.backend_pool = backend_kv_pool();
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

        std::optional<qwen3::detail::DiskRestoredHost> loaded =
            disk_cache.load_host(entry_id, request_plan.disk_committed_generation);
        if (!loaded) { throw std::runtime_error("KV disk restore lost its claimed entry"); }
        qwen3::detail::DiskRestoredHost host = std::move(*loaded);
        // The restore workers decode the entry's checkpoint heads straight into pool heads.
        // Occupy keeps only heads at or before the base, so later ones are never decoded; the
        // planned head goes first so pool exhaustion cannot drop it.
        // Heads stay pending when an earlier cancellation failed before it could prove its
        // workers drained; drain again before they return to the pool.
        if (!pending_disk_checkpoint_heads_.empty()) {
            disk_cache.cancel_restore();
            recycle_pending_disk_checkpoint_heads();
        }
        pending_disk_checkpoint_heads_.reserve(host.ladders.size());
        target.checkpoint_images.reserve(host.ladders.size());
        std::vector<qwen3::detail::RamLadderIndex> wanted;
        wanted.reserve(host.ladders.size());
        for (const qwen3::detail::RamLadderIndex& ladder : host.ladders) {
            if (ladder.frontier == request_plan.reuse_base) { wanted.push_back(ladder); }
        }
        for (const qwen3::detail::RamLadderIndex& ladder : host.ladders) {
            if (ladder.frontier < request_plan.reuse_base) { wanted.push_back(ladder); }
        }
        std::vector<std::uint8_t> selected(context_marks.size() + 1, 0);
        for (const qwen3::detail::RamLadderIndex& ladder : wanted) {
            if (!select_imported_checkpoint(ladder.frontier, ladder.kind, selected)) { continue; }
            std::optional<ContextCheckpointHead> head = acquire_context_checkpoint_head();
            if (!head) { break; }
            head->frontier = ladder.frontier;
            head->hash     = ladder.hash;
            head->kind     = ladder.kind;
            target.checkpoint_images.push_back(qwen3::detail::DiskCheckpointImageTarget{
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
        pending_disk_restore_ticket_     = disk_cache.restore_device(entry_id, target);
        pending_disk_checkpoint_lane_    = lane;
        sequence.execution_frontier      = host.execution_frontier;
        sequence.ledger_frontier         = host.ledger_frontier;
        sequence.rope_delta              = host.rope_delta;
        sequence.text_kv_publication     = {.valid_frontier = host.text_kv_valid};
        sequence.mtp_kv_publication      = {.valid_frontier = host.mtp_kv_valid};
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
        if (qwen3::detail::is_staged_checkpoint_restore(request_plan.reuse)) {
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
            if (qwen3::detail::occupy_drops_rewrite_ahead_of_restore(
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
        throw runtime::CacheRestoreFailure(
            "disk cache restore metadata allocation failed",
            runtime::CacheRestoreFailureKind::DestinationUnavailable);
    } catch (...) {
        pending_disk_checkpoint_lane_.reset();
        synchronize_all_while_unwinding(device);
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
    qwen3::detail::PrefixHash128 hash;
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
    disk_reuse_pages(speculative_backend, backend_kv_pool() != nullptr, plan.impl_->reuse_base,
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
        throw runtime::CacheRestoreFailure(
            "disk cache copy metadata allocation failed",
            runtime::CacheRestoreFailureKind::DestinationUnavailable);
    }
}

void ProgramImplCore::begin_copy_hold_cancel() {
    if (kv_disk_cache_) { kv_disk_cache_->begin_cancel_restore(); }
    HIP_CHECK(hipEventRecord(copy_hold_cancel_fence_, device.copy_stream));
}

bool ProgramImplCore::copy_hold_cancel_settled() const {
    const hipError_t ready = hipEventQuery(copy_hold_cancel_fence_);
    if (ready == hipErrorNotReady) { return false; }
    HIP_CHECK(ready);
    return !kv_disk_cache_ || kv_disk_cache_->restore_cancel_settled();
}

void ProgramImplCore::cancel_disk_restore() {
    if (!kv_disk_cache_) { return; }
    qwen3::detail::KVDiskCache& disk = *kv_disk_cache_;
    const std::uint64_t ticket       = pending_disk_restore_ticket_;
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
    while (!suspended_requests_.empty()) {
        const auto owner = suspended_requests_.begin()->first;
        discard_suspended(owner);
        release_suspension_credit(owner);
    }
    flush_deferred_gdn_folds();
    kv_tiers_shutdown_ = true;
    auto report        = [&](std::string_view phase, std::uint64_t done, std::uint64_t total) {
        if (!progress.callback) { return; }
        try {
            progress.callback(phase, done, total);
            // NOLINTNEXTLINE(bugprone-empty-catch): a failing observer must not abort the flush
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
        if (retained_lane_cache_writable(lane)) { ++retained; }
    }
    if (report_disk) { report("kv-disk copy active chats", 0, retained); }
    std::uint64_t captured = 0;
    for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
        if (!retained_lane_cache_writable(lane)) { continue; }
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

void ProgramImplCore::set_cache_idle(bool idle) {
    if (kv_disk_cache_) { kv_disk_cache_->set_cache_idle(idle); }
}

std::optional<qwen3::detail::KvDiskSnapshot>
ProgramImplCore::try_kv_disk_snapshot() const noexcept {
    return kv_disk_cache_ ? kv_disk_cache_->try_snapshot()
                          : std::optional{qwen3::detail::KvDiskSnapshot{}};
}

qwen3::detail::KvDiskCopySeconds ProgramImplCore::harvest_kv_disk_copy_seconds() {
    return kv_disk_cache_ ? kv_disk_cache_->harvest_copy_seconds()
                          : qwen3::detail::KvDiskCopySeconds{};
}

qwen3::detail::KvGpuSnapshot ProgramImplCore::kv_gpu_snapshot() const noexcept {
    const auto pool_snapshot = [](const PagedKVPool& pool) {
        return qwen3::detail::KvGpuPoolSnapshot{.page_group_count = pool.page_group_count(),
                                                .entitled_pages   = pool.entitled_pages(),
                                                .mapped_pages     = pool.mapped_pages(),
                                                .free_pages       = pool.free_pages()};
    };
    qwen3::detail::KvGpuSnapshot out;
    if (decoder) { out.main = pool_snapshot(decoder->text_kv.pool()); }
    if (const PagedKVPool* backend = backend_kv_pool()) { out.spec = pool_snapshot(*backend); }
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
        throw runtime::CacheRestoreFailure(
            "disk cache copy metadata allocation failed",
            runtime::CacheRestoreFailureKind::DestinationUnavailable);
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
    request.output = nullptr;
    if (staging_.occupied && staging_.lane == sequence.lane) { unoccupy_staging(); }
    request.prefill.reset();
    sequence.kv.reset();
    request.lifecycle           = Lifecycle::Empty;
    sequence.execution_frontier = 0;
    sequence.ledger_frontier    = 0;
    sequence.ledger.clear();
    sequence.prefix_identity.clear();
    sequence.prefix_hashes.clear();
    sequence.text_kv_publication        = {};
    sequence.mtp_kv_publication         = {};
    sequence.dflash_context_frontier    = 0;
    sequence.mtp_draft_count            = 0;
    sequence.tail_hidden_valid          = false;
    sequence.retained                   = false;
    sequence.cache_retention            = runtime::CacheRetention::Ordinary;
    sequence.fork_initialized           = false;
    sequence.turn_closed                = false;
    sequence.closure_frontier           = 0;
    sequence.use_tick                   = 0;
    sequence.disk_entry_id              = 0;
    sequence.rewrite_checkpoint         = {};
    sequence.tier_unpacked_context_base = 0;
    sequence.tier_unpacked_context_hash = {};
    clear_context_checkpoints(sequence);
    sequence.deferred_fold_columns  = 0;
    request.pending                 = {};
    request.adaptive                = {};
    request.typical_cycle_reasoning = false;
    request.prompt_tokens           = 0;
    request.window_output_tokens    = 0;
    request.effective_output_tokens = 0;
}

void ProgramImplCore::flush_deferred_gdn_folds(std::span<const std::uint32_t> lanes) {
    std::array<ops::GdnReplayFoldRow, kMaximumConcurrency> rows{};
    for (ops::GdnReplayFoldRow& row : rows) {
        row = {.linear_state_slot = LinearStateSlots::current_state_slot(0, max_concurrency),
               .commit_columns    = 0};
    }
    std::size_t row_count = 0;
    const auto take       = [&](std::uint32_t lane) {
        if (lane >= max_concurrency) { throw std::out_of_range("request lane is out of range"); }
        SequenceState& sequence = sequences[lane];
        if (sequence.deferred_fold_columns == 0) { return; }
        const std::uint32_t row = sequence.deferred_fold_row;
        if (!replay_records || row >= kMaximumConcurrency || rows[row].commit_columns != 0) {
            throw std::logic_error("deferred GDN fold row is invalid");
        }
        rows[row] = {.linear_state_slot =
                         LinearStateSlots::current_state_slot(lane, max_concurrency),
                     .commit_columns = static_cast<std::int32_t>(sequence.deferred_fold_columns)};
        row_count = std::max<std::size_t>(row_count, row + 1U);
        sequence.deferred_fold_columns = 0;
    };
    if (lanes.empty()) {
        for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) { take(lane); }
    } else {
        for (const std::uint32_t lane : lanes) { take(lane); }
    }
    if (row_count == 0) { return; }
    ops::gdn_replay_fold(*replay_records, decoder->linear_attention.all_layers_view(),
                         std::span<const ops::GdnReplayFoldRow>(rows.data(), row_count),
                         device.stream);
}

void ProgramImplCore::flush_deferred_gdn_fold(std::uint32_t lane) {
    flush_deferred_gdn_folds(std::span<const std::uint32_t>(&lane, 1));
}

PagedKVPool* ProgramImplCore::backend_kv_pool() noexcept {
    if (speculative_backend == SpeculativeBackend::Mtp) {
        qwen3::PagedKVCache* cache = decoder->mtp_cache();
        return cache != nullptr ? &cache->pool() : nullptr;
    }
    if (speculative_backend == SpeculativeBackend::DFlash && dflash) {
        if constexpr (DFlashConfig::full_layers > 0) {
            return dflash->full ? &dflash->full->pool() : nullptr;
        }
        return nullptr;
    }
    return nullptr;
}

const PagedKVPool* ProgramImplCore::backend_kv_pool() const noexcept {
    if (speculative_backend == SpeculativeBackend::Mtp) {
        const qwen3::PagedKVCache* cache = decoder->mtp_cache();
        return cache != nullptr ? &cache->pool() : nullptr;
    }
    if (speculative_backend == SpeculativeBackend::DFlash && dflash) {
        if constexpr (DFlashConfig::full_layers > 0) {
            return dflash->full ? &dflash->full->pool() : nullptr;
        }
        return nullptr;
    }
    return nullptr;
}

std::uint32_t ProgramImplCore::backend_kv_valid(const SequenceState& sequence) const noexcept {
    if (speculative_backend == SpeculativeBackend::Mtp) {
        return sequence.mtp_kv_publication.valid_frontier;
    }
    if (speculative_backend == SpeculativeBackend::DFlash) {
        if constexpr (DFlashConfig::full_layers > 0) { return sequence.dflash_context_frontier; }
        return 0;
    }
    return 0;
}

void ProgramImplCore::reserve_sequence_kv(SequenceState& sequence, std::uint32_t text_pages,
                                          std::uint32_t backend_pages) {
    if (sequence.kv) { throw std::logic_error("sequence already owns a KV allocation bundle"); }
    if (text_pages == 0 || (backend_kv_pool() == nullptr) != (backend_pages == 0)) {
        throw std::invalid_argument("KV allocation entitlement does not match the active backend");
    }

    std::array<PagedKVReservation, 2> reservations{};
    std::size_t count     = 0;
    reservations[count++] = PagedKVReservation{
        .pool             = &decoder->text_kv.pool(),
        .page_entitlement = text_pages,
    };
    if (PagedKVPool* backend = backend_kv_pool(); backend != nullptr) {
        reservations[count++] = PagedKVReservation{
            .pool             = backend,
            .page_entitlement = backend_pages,
        };
    }

    std::vector<PagedKVAllocation> allocations =
        reserve_paged_kv_bundle(std::span<const PagedKVReservation>(reservations.data(), count));
    SequenceKVBundle bundle;
    bundle.text = std::move(allocations[0]);
    if (count == 2) { bundle.backend.emplace(std::move(allocations[1])); }
    sequence.kv.emplace(std::move(bundle));
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

void ProgramImplCore::truncate_sequence_mtp_publication(SequenceState& sequence,
                                                        std::uint32_t retained_frontier) {
    qwen3::PagedKVCache* cache = decoder->mtp_cache();
    if (speculative_backend != SpeculativeBackend::Mtp || cache == nullptr || !sequence.kv ||
        !sequence.kv->backend) {
        throw std::logic_error("MTP publication truncation has no typed cache allocation");
    }

    const bool text_bound    = sequence.kv->text.bound_row() >= 0;
    const bool backend_bound = sequence.kv->backend->bound_row() >= 0;
    if (text_bound != backend_bound) {
        throw std::logic_error("MTP publication truncation found a partially bound KV bundle");
    }
    if (text_bound &&
        (sequence.kv->text.bound_row() != static_cast<std::int32_t>(sequence.lane) ||
         sequence.kv->backend->bound_row() != static_cast<std::int32_t>(sequence.lane))) {
        throw std::logic_error("MTP publication truncation found a foreign KV table row");
    }

    const bool bind_here = !text_bound;
    if (bind_here) { bind_sequence_kv(sequence); }
    try {
        cache->truncate_publication(*sequence.kv->backend, sequence.mtp_kv_publication,
                                    retained_frontier);
    } catch (...) {
        if (bind_here) { unbind_sequence_kv(sequence); }
        throw;
    }
    if (bind_here) { unbind_sequence_kv(sequence); }
}

void ProgramImplCore::materialize_sequence_kv(SequenceState& sequence, std::uint32_t main_tokens,
                                              std::uint32_t backend_tokens) {
    if (!sequence.kv || main_tokens > capacity || backend_tokens > capacity) {
        throw std::logic_error("KV materialization request is outside the sequence bundle");
    }
    if (backend_tokens != 0 && !sequence.kv->backend) {
        throw std::logic_error("backend KV materialization requested without an allocation");
    }
    if (main_tokens > sequence.kv->text.mapped_token_capacity()) {
        sequence.kv->text.materialize_tokens(main_tokens, device.stream);
    }
    if (backend_tokens != 0) {
        PagedKVAllocation& backend = sequence.kv->backend.value();
        if (backend_tokens > backend.mapped_token_capacity()) {
            backend.materialize_tokens(backend_tokens, device.stream);
        }
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

qwen3::PagedKVCacheView ProgramImplCore::text_kv_view(const SequenceState& sequence) const {
    if (!sequence.kv) { throw std::logic_error("sequence has no KV allocation bundle"); }
    if (!sequence.text_kv_publication.healthy || sequence.text_kv_publication.transaction_open) {
        throw std::logic_error("sequence Text KV publication is unavailable");
    }
    return decoder->text_kv.execution_view(sequence.kv->text);
}

qwen3::PagedKVCacheView ProgramImplCore::mtp_kv_view(const SequenceState& sequence) const {
    if (speculative_backend != SpeculativeBackend::Mtp) { return {}; }
    if (decoder->mtp_cache() == nullptr || !sequence.kv || !sequence.kv->backend) {
        throw std::logic_error("sequence has no MTP KV allocation");
    }
    if (!sequence.mtp_kv_publication.healthy || sequence.mtp_kv_publication.transaction_open) {
        throw std::logic_error("sequence MTP KV publication is unavailable");
    }
    return decoder->mtp_cache()->execution_view(*sequence.kv->backend);
}

void ProgramImplCore::set_device_i32(Tensor& tensor, std::int32_t value) {
    HIP_CHECK(
        hipMemcpyAsync(tensor.data, &value, sizeof(value), hipMemcpyHostToDevice, device.stream));
}

void ProgramImplCore::ordered_reset(SequenceState& sequence) {
    decoder->linear_attention.zero_slot(
        LinearStateSlots::current_state_slot(sequence.lane, max_concurrency), device.stream);
    work.reset();
    set_device_i32(io.pos, 0);
    set_device_i32(io.rope_pos, 0);
    set_device_i32(io.rope_delta, 0);
    if (io.mtp) { set_device_i32(io.mtp->position, 0); }
    sequence.text_kv_publication     = {};
    sequence.mtp_kv_publication      = {};
    sequence.dflash_context_frontier = 0;
}

void ProgramImplCore::prepare_graphs() {
    if (!use_device_graph) { return; }
    SequenceState& sequence = sequences[0];
    ops::PersistentDecodeStorage persistent_storage;

    std::vector<PagedKVAllocation> text_capture_allocations;
    std::vector<PagedKVAllocation> mtp_capture_allocations;
    std::vector<PagedKVAllocation> dflash_capture_allocations;
    const auto reserve_capture_rows = [&](PagedKVPool& pool,
                                          std::vector<PagedKVAllocation>& allocations,
                                          const char* label) {
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
            HIP_CHECK(hipMemcpyAsync(table.data, repeated.data(), table.bytes(),
                                     hipMemcpyHostToDevice, device.stream));
        }
    };
    reserve_capture_rows(decoder->text_kv.pool(), text_capture_allocations, "target KV cache");
    if (speculative_backend == SpeculativeBackend::Mtp) {
        if (decoder->mtp_cache() == nullptr) {
            throw std::logic_error("MTP Device Graph has no typed backend cache");
        }
        reserve_capture_rows(decoder->mtp_cache()->pool(), mtp_capture_allocations, "MTP KV cache");
    }
    if (speculative_backend == SpeculativeBackend::DFlash) {
        if constexpr (DFlashConfig::full_layers > 0) {
            reserve_capture_rows(dflash->full->pool(), dflash_capture_allocations,
                                 "DFlash Full KV cache");
        }
    }
    device.synchronize();

    std::size_t free_before = 0;
    std::size_t total_bytes = 0;
    HIP_CHECK(hipMemGetInfo(&free_before, &total_bytes));

    const auto clear_stable_controls = [&] {
        std::vector<Tensor> controls{
            io.token,
            io.pos,
            io.rope_pos,
            io.rope_delta,
        };
        if (io.dflash_prefill) { controls.push_back(io.dflash_prefill->produced_count); }
        for (const Tensor& tensor : controls) {
            HIP_CHECK(hipMemsetAsync(tensor.data, 0, tensor.bytes(), device.stream));
        }
    };
    const auto zero_capture_pages = [&](PagedKVPool& pool,
                                        const std::vector<PagedKVAllocation>& allocations,
                                        std::uint32_t batch_size) {
        std::vector<std::int32_t> pages;
        pages.reserve(batch_size);
        for (std::uint32_t row = 0; row < batch_size; ++row) {
            pages.push_back(allocations[row].page_ids().front());
        }
        pool.zero_pages(pages, device.stream);
    };
    const auto zero_cyclic_lane = [&](CyclicKVCache& cache, std::uint32_t lane) {
        for (std::uint32_t layer = 0; layer < cache.layer_count(); ++layer) {
            const CyclicKVCacheLayerView view = cache.layer_view(layer);
            const Tensor k                    = view.k.slice(3, static_cast<std::int32_t>(lane), 1);
            const Tensor v                    = view.v.slice(3, static_cast<std::int32_t>(lane), 1);
            HIP_CHECK(hipMemsetAsync(k.data, 0, k.bytes(), device.stream));
            HIP_CHECK(hipMemsetAsync(v.data, 0, v.bytes(), device.stream));
        }
    };

    const auto prepare_representative = [&](std::uint32_t frontier, std::uint32_t batch_size,
                                            std::uint32_t representative_k = 0) {
        if (batch_size == 0 || batch_size > max_concurrency) {
            throw std::logic_error("Device Graph representative batch is invalid");
        }
        work.reset();
        clear_stable_controls();
        zero_capture_pages(decoder->text_kv.pool(), text_capture_allocations, batch_size);
        if (!mtp_capture_allocations.empty()) {
            zero_capture_pages(decoder->mtp_cache()->pool(), mtp_capture_allocations, batch_size);
        }
        if (dflash && dflash->full) {
            zero_capture_pages(dflash->full->pool(), dflash_capture_allocations, batch_size);
        }
        for (std::uint32_t row = 0; row < batch_size; ++row) {
            decoder->linear_attention.zero_slot(
                LinearStateSlots::current_state_slot(row, max_concurrency), device.stream);
            if (dflash) {
                zero_cyclic_lane(dflash->local, row);
                const Tensor pending =
                    dflash->pending_features.slice(2, static_cast<std::int32_t>(row), 1);
                HIP_CHECK(hipMemsetAsync(pending.data, 0, pending.bytes(), device.stream));
            }
        }
        set_device_i32(io.pos, checked_i32(frontier, "graph representative position"));
        set_device_i32(io.rope_pos, checked_i32(frontier, "graph representative rope position"));
        if (io.dflash_decode) {
            *dflash_host_ingress         = {};
            *dflash_host_egress          = {};
            const std::uint32_t active_k = representative_k == 0U ? draft_window : representative_k;
            const std::uint32_t extent   = std::min(active_k, capacity - frontier - 1U);
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
        if (io.mtp_decode) {
            *mtp_host_ingress            = {};
            *mtp_host_egress             = {};
            const std::uint32_t active_k = representative_k == 0U ? draft_window : representative_k;
            const std::uint32_t extent   = std::min(active_k, capacity - frontier - 1U);
            const std::uint32_t width    = draft_window + 1U;
            for (std::uint32_t row = 0; row < batch_size; ++row) {
                mtp_host_ingress->anchors[row] = 0;
                mtp_host_ingress->base_frontiers[row] =
                    checked_i32(frontier, "graph representative MTP frontier");
                mtp_host_ingress->remaining_budgets[row] =
                    checked_i32(capacity - frontier, "graph representative MTP budget");
                mtp_host_ingress->current_extents[row] = static_cast<std::int32_t>(extent);
                mtp_host_ingress->target_valid_columns[row] =
                    static_cast<std::int32_t>(extent + 1U);
                for (std::uint32_t j = 0; j < draft_window; ++j) {
                    mtp_host_ingress->current_drafts[row * draft_window + j] = 0;
                }
                for (std::uint32_t j = 0; j < width; ++j) {
                    mtp_host_ingress->target_rope_positions[row * width + j] = checked_i32(
                        frontier + std::min(j, extent), "graph representative MTP RoPE position");
                }
                mtp_host_ingress->text_kv_table_rows[row] = static_cast<std::int32_t>(row);
                mtp_host_ingress->mtp_kv_table_rows[row]  = static_cast<std::int32_t>(row);
                mtp_host_ingress->lanes[row]              = static_cast<std::int32_t>(row);
                mtp_host_ingress->sampling[row]           = {};
            }
        }
    };
    const auto execution_core = [&] {
        return schedule::ExecutionCore{device,
                                       model,
                                       linear_execution.get(),
                                       work,
                                       decoder->linear_attention,
                                       replay_records ? &*replay_records : nullptr,
                                       io,
                                       prefill_hidden,
                                       prefill_chunk,
                                       proposal_head};
    };

    if (speculative_backend == SpeculativeBackend::None) {
        const auto ordinary_profiles = ordinary_graph_profiles(capacity);
        validate_graph_profiles(ordinary_profiles, capacity - 1, "ordinary");
        const std::uint32_t ordinary_batch_limit = max_concurrency;
        const auto run_representative = [&](std::uint32_t frontier, std::uint32_t maximum_frontier,
                                            std::uint32_t batch_size,
                                            DecodeGraphDefinition* definition) {
            prepare_representative(frontier, batch_size);
            device.synchronize();

            std::array<qwen3::PagedKVPublication, kMaximumConcurrency> publications{};
            SegmentedKvTransactionBatch transactions(batch_size, kv_resolution_words(false));
            const auto* positions =
                static_cast<const std::int32_t*>(io.ordinary->cache_positions.data);
            const auto* table_rows =
                static_cast<const std::int32_t*>(io.ordinary->text_kv_table_rows.data);
            auto* status                      = static_cast<std::uint32_t*>(io.text_kv_status.data);
            auto* cursor                      = static_cast<std::uint32_t*>(io.text_kv_cursor.data);
            const std::uint32_t visible_limit = maximum_frontier + 1U;
            for (std::uint32_t row = 0; row < batch_size; ++row) {
                publications[row].valid_frontier = frontier;
                transactions.append(decoder->text_kv.begin_device_segmented_append(
                    text_capture_allocations[row], publications[row], positions + row, 0U,
                    maximum_frontier, visible_limit,
                    {.status = status + row, .cursor = cursor + row}, table_rows + row));
            }

            schedule::OrdinaryBatchContext ordinary_state{
                execution_core(),       decoder->text_kv,      *io.ordinary,
                *ordinary_host_ingress, *ordinary_host_egress, tail_hidden_store,
                transactions.binding()};
            if (definition != nullptr) {
                schedule::capture_ordinary_decode_batch(ordinary_state,
                                                        static_cast<std::int32_t>(batch_size),
                                                        *definition, persistent_storage);
                transactions.close_captured();
                return;
            }
            schedule::ordinary_decode_batch(ordinary_state, static_cast<std::int32_t>(batch_size),
                                            nullptr);
            transactions.enqueue_resolution();
            device.synchronize();
            std::array<std::uint32_t, kMaximumConcurrency> retained_frontiers{};
            for (std::uint32_t row = 0; row < batch_size; ++row) {
                retained_frontiers[row] = frontier + 1U;
            }
            transactions.finish_resolution({retained_frontiers.data(), batch_size});
        };

        const GraphExecutionProfile code_warm = ordinary_profiles.front();
        run_representative(code_warm.min, code_warm.max, 1, nullptr);

        ordinary_graphs.profiles.reserve(ordinary_profiles.size() * ordinary_batch_limit);
        for (std::uint32_t batch_size = 1; batch_size <= ordinary_batch_limit; ++batch_size) {
            for (const GraphExecutionProfile planned : ordinary_profiles) {
                ordinary_graphs.profiles.emplace_back();
                DecodeGraphProfile& profile    = ordinary_graphs.profiles.back();
                profile.batch_size             = batch_size;
                profile.min_execution_frontier = planned.min;
                profile.max_execution_frontier = planned.max;
                profile.topology_class         = planned.topology_class;
                run_representative(planned.min, planned.max, batch_size, &profile.definition);
            }
        }
    }

    if (speculative_backend == SpeculativeBackend::Mtp) {
        for (const auto capture_k : captured_ks) {

            if (!io.mtp_decode || !decoder->mtp_cache()) {
                throw std::logic_error("MTP Device Graph state is incomplete");
            }
            const auto run_representative = [&](std::uint32_t frontier,
                                                std::uint32_t maximum_frontier,
                                                std::uint32_t batch_size, std::uint32_t k,
                                                DecodeGraphDefinition* definition) {
                prepare_representative(frontier, batch_size, capture_k);
                device.synchronize();

                std::array<qwen3::PagedKVPublication, kMaximumConcurrency> text_publications{};
                std::array<qwen3::PagedKVPublication, kMaximumConcurrency> mtp_publications{};
                SegmentedKvTransactionBatch text_transactions(batch_size,
                                                              kv_resolution_words(false));
                SegmentedKvTransactionBatch mtp_transactions(batch_size, kv_resolution_words(true));
                auto* text_status = static_cast<std::uint32_t*>(io.text_kv_status.data);
                auto* text_cursor = static_cast<std::uint32_t*>(io.text_kv_cursor.data);
                auto* mtp_status  = static_cast<std::uint32_t*>(io.backend_kv_status.data);
                auto* mtp_cursor  = static_cast<std::uint32_t*>(io.backend_kv_cursor.data);
                const auto* base =
                    static_cast<const std::int32_t*>(io.mtp_decode->base_frontiers.data);
                const auto* text_rows =
                    static_cast<const std::int32_t*>(io.mtp_decode->text_kv_table_rows.data);
                const auto* mtp_rows =
                    static_cast<const std::int32_t*>(io.mtp_decode->mtp_kv_table_rows.data);
                const std::uint32_t text_visible =
                    std::min(capacity, maximum_frontier + std::min(k + 1U, capacity));
                const std::uint32_t mtp_visible =
                    std::min(capacity, maximum_frontier + std::min(2U * k, capacity));
                for (std::uint32_t row = 0; row < batch_size; ++row) {
                    text_publications[row].valid_frontier = frontier;
                    mtp_publications[row].valid_frontier  = frontier;
                    text_transactions.append(decoder->text_kv.begin_device_segmented_append(
                        text_capture_allocations[row], text_publications[row], base + row, 0U,
                        maximum_frontier, text_visible,
                        {.status = text_status + row, .cursor = text_cursor + row},
                        text_rows + row));
                    mtp_transactions.append(decoder->mtp_cache()->begin_device_segmented_append(
                        mtp_capture_allocations[row], mtp_publications[row], base + row, 0U,
                        maximum_frontier, mtp_visible,
                        {.status = mtp_status + row, .cursor = mtp_cursor + row}, mtp_rows + row));
                }

                schedule::MtpBatchContext mtp_state{
                    execution_core(),           decoder->text_kv,
                    *decoder->mtp_cache(),      *io.mtp_decode,
                    *mtp_host_ingress,          *mtp_host_egress,
                    tail_hidden_store,          text_transactions.binding(),
                    mtp_transactions.binding(), token_masks.get()};
                if (definition != nullptr) {
                    schedule::capture_mtp_decode_batch(mtp_state,
                                                       static_cast<std::int32_t>(batch_size), k,
                                                       *definition, persistent_storage);
                    text_transactions.close_captured();
                    mtp_transactions.close_captured();
                    return;
                }
                TokenMaskRoundGuard token_mask_round(*token_masks, device);
                schedule::mtp_decode_batch(mtp_state, static_cast<std::int32_t>(batch_size), k,
                                           nullptr);
                text_transactions.enqueue_resolution();
                mtp_transactions.enqueue_resolution();
                device.synchronize();
                token_mask_round.finish();
                text_transactions.finish_resolution({text_transactions.cursor, batch_size});
                mtp_transactions.finish_resolution({mtp_transactions.cursor, batch_size});
            };

            const auto planned_profiles = mtp_graph_profiles(capacity, capture_k);
            if (planned_profiles.empty()) {
                throw std::logic_error("MTP Device Graph has no execution profiles");
            }
            run_representative(planned_profiles.front().min, planned_profiles.front().max, 1,
                               capture_k, nullptr);

            validate_graph_profiles(planned_profiles, capacity - 1U, "MTP");
            mtp_graphs.profiles.reserve(planned_profiles.size() * max_concurrency);
            for (std::uint32_t batch_size = 1; batch_size <= max_concurrency; ++batch_size) {
                for (const GraphExecutionProfile planned : planned_profiles) {
                    mtp_graphs.profiles.emplace_back();
                    DecodeGraphProfile& profile    = mtp_graphs.profiles.back();
                    profile.batch_size             = batch_size;
                    profile.draft_tokens           = capture_k;
                    profile.min_execution_frontier = planned.min;
                    profile.max_execution_frontier = planned.max;
                    profile.topology_class         = planned.topology_class;
                    run_representative(planned.min, planned.max, batch_size, capture_k,
                                       &profile.definition);
                }
            }
        }
    }

    if (speculative_backend == SpeculativeBackend::DFlash) {
        for (const auto capture_k : captured_ks) {

            const std::uint32_t fixed_k = capture_k;
            const std::uint32_t fixed_w =
                dflash_captured_verify_width(capture_k, dflash_verify_width);
            const auto run_representative = [&](std::uint32_t frontier,
                                                std::uint32_t maximum_frontier,
                                                std::uint32_t batch_size,
                                                DecodeGraphDefinition* definition) {
                prepare_representative(frontier, batch_size, capture_k);
                device.synchronize();

                std::array<qwen3::PagedKVPublication, kMaximumConcurrency> publications{};
                SegmentedKvTransactionBatch transactions(batch_size, kv_resolution_words(false));
                const auto* base_frontiers =
                    static_cast<const std::int32_t*>(io.dflash_decode->execution_frontiers.data);
                const auto* table_rows =
                    static_cast<const std::int32_t*>(io.dflash_decode->text_kv_table_rows.data);
                auto* status = static_cast<std::uint32_t*>(io.text_kv_status.data);
                auto* cursor = static_cast<std::uint32_t*>(io.text_kv_cursor.data);
                const std::uint32_t visible_limit =
                    std::min(capacity, maximum_frontier + std::min(fixed_w, capacity));
                for (std::uint32_t row = 0; row < batch_size; ++row) {
                    publications[row].valid_frontier = frontier;
                    transactions.append(decoder->text_kv.begin_device_segmented_append(
                        text_capture_allocations[row], publications[row], base_frontiers + row, 0U,
                        maximum_frontier, visible_limit,
                        {.status = status + row, .cursor = cursor + row}, table_rows + row));
                }

                // The captured round must score the calibration grid exactly when replays of
                // this shape read it back (decode_dflash_batch).
                schedule::DFlashBatchContext dflash_state{
                    execution_core(),          decoder->text_kv,       *dflash,
                    *io.dflash_decode,         *dflash_host_ingress,   *dflash_host_egress,
                    tail_hidden_store,         transactions.binding(), token_masks.get(),
                    calibrates_p_less_drafts()};
                const schedule::DFlashEnvelopes envelopes = dflash_envelopes(maximum_frontier);
                if (definition != nullptr) {
                    schedule::capture_dflash_decode_batch(
                        dflash_state, static_cast<std::int32_t>(batch_size), fixed_k, fixed_w,
                        envelopes, *definition, persistent_storage);
                    transactions.close_captured();
                    return;
                }
                TokenMaskRoundGuard token_mask_round(*token_masks, device);
                schedule::dflash_decode_batch(dflash_state, static_cast<std::int32_t>(batch_size),
                                              fixed_k, fixed_w, envelopes, nullptr);
                transactions.enqueue_resolution();
                device.synchronize();
                token_mask_round.finish();
                transactions.finish_resolution({transactions.cursor, batch_size});
            };

            const auto batch_one_profiles = dflash_graph_profiles(capacity, fixed_k, 1, fixed_w);
            const GraphExecutionProfile code_warm = batch_one_profiles.front();
            run_representative(code_warm.min, code_warm.max, 1, nullptr);

            std::size_t dflash_profile_count = 0;
            for (std::uint32_t batch_size = 1; batch_size <= max_concurrency; ++batch_size) {
                dflash_profile_count +=
                    dflash_graph_profiles(capacity, fixed_k, batch_size, fixed_w).size();
            }
            dflash_graphs.profiles.reserve(dflash_profile_count);
            for (std::uint32_t batch_size = 1; batch_size <= max_concurrency; ++batch_size) {
                const auto planned_profiles =
                    dflash_graph_profiles(capacity, fixed_k, batch_size, fixed_w);
                validate_graph_profiles(planned_profiles, capacity - 1, "DFlash");
                for (const GraphExecutionProfile planned : planned_profiles) {
                    dflash_graphs.profiles.emplace_back();
                    DecodeGraphProfile& profile    = dflash_graphs.profiles.back();
                    profile.batch_size             = batch_size;
                    profile.draft_tokens           = capture_k;
                    profile.min_execution_frontier = planned.min;
                    profile.max_execution_frontier = planned.max;
                    profile.topology_class         = planned.topology_class;
                    run_representative(planned.min, planned.max, batch_size, &profile.definition);
                }
            }
        }
    }

    if (!ordinary_graphs.profiles.empty()) {
        instantiate_graph_family(ordinary_graphs, "ordinary", device, prepare_representative,
                                 *token_masks);
    }
    if (speculative_backend == SpeculativeBackend::DFlash) {
        instantiate_graph_family(dflash_graphs, "DFlash", device, prepare_representative,
                                 *token_masks);
    }
    if (speculative_backend == SpeculativeBackend::Mtp) {
        instantiate_graph_family(mtp_graphs, "MTP", device, prepare_representative, *token_masks);
    }

    const std::size_t captured_definitions     = ordinary_graphs.profiles.size() +
                                                 mtp_graphs.profiles.size() +
                                                 dflash_graphs.profiles.size();
    const std::size_t instantiated_executables = ordinary_graphs.profiles.size() +
                                                 mtp_graphs.profiles.size() +
                                                 dflash_graphs.profiles.size();
    if (captured_definitions != expected_graph_definition_count) {
        throw std::logic_error("Device Graph definition inventory disagrees with its memory plan");
    }
    if (instantiated_executables != expected_graph_executable_count) {
        throw std::logic_error("Device Graph executable inventory disagrees with its memory plan");
    }

    ordered_reset(sequence);
    clear_stable_controls();
    for (Tensor& tensor : decoder->linear_attention.conv) {
        HIP_CHECK(hipMemsetAsync(tensor.data, 0, tensor.bytes(), device.stream));
    }
    for (Tensor& tensor : decoder->linear_attention.recurrent) {
        HIP_CHECK(hipMemsetAsync(tensor.data, 0, tensor.bytes(), device.stream));
    }
    if (dflash) {
        auto& state = *dflash;
        zero_dflash_local_cache(state.local);
        HIP_CHECK(hipMemsetAsync(state.prefill_features.data, 0, state.prefill_features.bytes(),
                                 device.stream));
        HIP_CHECK(hipMemsetAsync(state.prefill_positions.data, 0, state.prefill_positions.bytes(),
                                 device.stream));
        HIP_CHECK(hipMemsetAsync(state.pending_features.data, 0, state.pending_features.bytes(),
                                 device.stream));
    }
    HIP_CHECK(hipMemsetAsync(token_counts.data, 0, token_counts.bytes(), device.stream));
    device.synchronize();

    std::size_t free_after = 0;
    HIP_CHECK(hipMemGetInfo(&free_after, &total_bytes));
    const std::size_t consumed = free_before > free_after ? free_before - free_after : 0;
    graph_observed_bytes       = consumed;
    if (consumed > graph_allowance_bytes) {
        throw std::runtime_error("Device Graph preparation consumed " + std::to_string(consumed) +
                                 " bytes, exceeding the planned allowance of " +
                                 std::to_string(graph_allowance_bytes) + " bytes");
    }
    for (PagedKVAllocation& allocation : dflash_capture_allocations) { allocation.unbind_row(); }
    dflash_capture_allocations.clear();
    for (PagedKVAllocation& allocation : mtp_capture_allocations) { allocation.unbind_row(); }
    mtp_capture_allocations.clear();
    for (PagedKVAllocation& allocation : text_capture_allocations) { allocation.unbind_row(); }
    text_capture_allocations.clear();
}

void ProgramImplCore::zero_dflash_local_cache(CyclicKVCache& cache) {
    for (std::uint32_t layer = 0; layer < cache.layer_count(); ++layer) {
        const CyclicKVCacheLayerView view = cache.layer_view(layer);
        HIP_CHECK(hipMemsetAsync(view.k.data, 0, view.k.bytes(), device.stream));
        HIP_CHECK(hipMemsetAsync(view.v.data, 0, view.v.bytes(), device.stream));
    }
}

void ProgramImplCore::install_sampling(SequenceState& sequence, RequestControl& request,
                                       const ops::SamplingConfig& config) {
    Tensor counts = token_counts.slice(1, static_cast<std::int32_t>(sequence.lane), 1)
                        .view({TextConfig::token_domain});
    HIP_CHECK(hipMemsetAsync(counts.data, 0, counts.bytes(), device.stream));
    request.sampling_host                  = config;
    request.sampling_host.nonfinite_logits = static_cast<std::uint32_t*>(io.nonfinite_logits.data) +
                                             sequence.lane * qwen3::kDFlashDecodeMaximumWidth;
    request.speculative_stats              = SpeculativeStats{
        .backend               = speculative_backend,
        .enabled               = speculative_backend != SpeculativeBackend::None,
        .draft_window          = draft_window,
        .accepted_per_position = std::vector<std::uint64_t>(draft_window, 0),
        .rounds_per_draft      = std::vector<std::uint64_t>(draft_window + 1U, 0),
    };
    qwen3::seed_adaptive_draft_state(
        request.adaptive,
        adaptive_draft ? 0U : qwen3::adaptive_seed_k(captured_ks, speculative_backend));
    const bool penalties = request.sampling_host.presence_penalty != 0.0F ||
                           request.sampling_host.frequency_penalty != 0.0F;
    request.sampling_host.token_counts =
        penalties ? static_cast<std::int32_t*>(counts.data) : nullptr;
    bind_prefill_sampling(sequence, request);
}

void ProgramImplCore::bind_prefill_sampling(SequenceState& sequence, RequestControl& request) {
    // The root mask row is shared by every prefilling request; request sampling itself keeps
    // no compact-row pointer.
    request.prefill_sampling_host =
        token_masks->prefill_root(request.output, request.sampling_host, device.stream);
    Tensor config_lane = sampling_config.slice(1, static_cast<std::int32_t>(sequence.lane), 1);
    HIP_CHECK(hipMemcpyAsync(config_lane.data, &request.prefill_sampling_host,
                             sizeof(request.prefill_sampling_host), hipMemcpyHostToDevice,
                             device.stream));
}

bool ProgramImplCore::any_token_logprobs(std::span<const std::uint32_t> lanes) const {
    return std::any_of(lanes.begin(), lanes.end(),
                       [&](std::uint32_t lane) { return requests[lane].token_logprobs; });
}

void ProgramImplCore::bind_token_mask_batch(std::span<const std::uint32_t> lanes,
                                            std::span<const ops::SamplingConfig> configs) {
    std::array<const qwen3::OutputSession*, kMaximumConcurrency> outputs{};
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        outputs[row] = requests[lanes[row]].output;
    }
    token_masks->bind({outputs.data(), lanes.size()}, configs.first(lanes.size()));
}

void ProgramImplCore::copy_tail(SequenceState& sequence, const Tensor& source) {
    if (source.dtype != DType::BF16 || source.ne[0] != TextConfig::hidden || source.ne[1] != 1) {
        throw std::logic_error("target tail hidden has an invalid shape");
    }
    HIP_CHECK(hipMemcpyAsync(sequence.tail_hidden.data, source.data, sequence.tail_hidden.bytes(),
                             hipMemcpyDeviceToDevice, device.stream));
    sequence.tail_hidden_valid = true;
}

void ProgramImplCore::copy_round_token() {
    HIP_CHECK(hipMemcpyAsync(host_tokens, io.token.data, sizeof(TokenId), hipMemcpyDeviceToHost,
                             device.stream));
}

void ProgramImplCore::copy_prefill_logprobs() {
    HIP_CHECK(hipMemcpyAsync(host_prefill_logprobs,
                             static_cast<const unsigned char*>(io.prefill_logprob_frame.data) +
                                 offsetof(qwen3::PrefillLogprobFrame, records),
                             sizeof(*host_prefill_logprobs), hipMemcpyDeviceToHost, device.stream));
}

void ProgramImplCore::mark_workspace_usage(std::size_t phase_bytes) noexcept {
    workspace_logical_peak_bytes = std::max(workspace_logical_peak_bytes, phase_bytes);
}

void ProgramImplCore::order_persistent_decode_after_transfers(
    std::size_t batch_size, const DecodeGraphExecutable* executable) {
    if (batch_size != 1U || executable == nullptr) { return; }
    // A resident grid must not compete with previously submitted transfer kernels. The worker
    // alone pumps GPU cache work: these fixed prior-work edges cannot include a future round.
    if (kv_disk_cache_) { kv_disk_cache_->join_submitted_restore(device.copy_stream); }
    device.order_compute_after_transfers();
}

void ProgramImplCore::enqueue_dflash_context_append(std::span<const std::uint32_t> lanes,
                                                    std::span<const std::uint32_t> starts,
                                                    std::span<const std::uint32_t> counts) {
    if (speculative_backend != SpeculativeBackend::DFlash || !dflash || !io.dflash_decode ||
        lanes.empty() || lanes.size() > max_concurrency || starts.size() != lanes.size() ||
        counts.size() != lanes.size()) {
        throw std::logic_error("DFlash context append has invalid membership");
    }
    DFlashPersistentState& dflash_state = *dflash;

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
        materialize_sequence_kv(
            sequence, std::max(sequence.text_kv_publication.valid_frontier, end), backend_end);
        minimum_count = std::min(minimum_count, counts[row]);
        maximum_count = std::max(maximum_count, counts[row]);
    }

    qwen3::DFlashDecodeState& frame = *io.dflash_decode;
    HIP_CHECK(hipMemcpyAsync(frame.ingress.data, dflash_host_ingress,
                             sizeof(qwen3::DFlashDecodeIngress), hipMemcpyHostToDevice,
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
    ops::prepare_ragged_prefix(dflash_state.pending_features, lane_tensor, device_starts,
                               device_ends, features, positions, device_counts, device.stream);

    schedule::DFlashAppendContext state{{device, model, linear_execution.get(), work,
                                         decoder->linear_attention,
                                         replay_records ? &*replay_records : nullptr, io,
                                         prefill_hidden, prefill_chunk, proposal_head},
                                        dflash_state};
    mark_workspace_usage(workspace_plan.dflash_context);
    schedule::dflash_append_context(state, features, positions, device_counts, lane_tensor,
                                    table_rows, {minimum_count, maximum_count});
}

void ProgramImplCore::clear_nonfinite_logits(std::uint32_t lane) {
    Tensor status = io.nonfinite_logits.slice(1, static_cast<std::int32_t>(lane), 1);
    HIP_CHECK(hipMemsetAsync(status.data, 0, status.bytes(), device.stream));
}

void ProgramImplCore::copy_nonfinite_logits() {
    auto* host = static_cast<PrefillRoundHost*>(round_host.data());
    HIP_CHECK(hipMemcpyAsync(host->nonfinite_logits.data(), io.nonfinite_logits.data,
                             io.nonfinite_logits.bytes(), hipMemcpyDeviceToHost, device.stream));
}

void ProgramImplCore::validate_finite_logits(std::uint32_t lane, std::uint32_t columns) const {
    if (lane >= max_concurrency || columns == 0 || columns > qwen3::kDFlashDecodeMaximumWidth) {
        throw std::logic_error("target logit validation has an invalid live domain");
    }
    const auto* host   = static_cast<const PrefillRoundHost*>(round_host.data());
    const auto* status = host->nonfinite_logits.data() + lane * qwen3::kDFlashDecodeMaximumWidth;
    for (std::uint32_t column = 0; column < columns; ++column) {
        if (status[column] != 0U) {
            throw std::runtime_error(
                "target returned nonfinite eligible logits before token publication");
        }
    }
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
                                                            RequestControl& request,
                                                            bool decode_waiting,
                                                            const MixedChunkRunner* mixed,
                                                            std::uint32_t step_cap) {
    if (request.lifecycle != Lifecycle::Prefilling || !request.prefill) {
        throw std::logic_error("staged prefill step requires an active concurrent request");
    }

    RequestControl::Prefill& staged = *request.prefill;
    const runtime::BeginSummary summary{.prompt_tokens        = staged.prompt_tokens,
                                        .reused_prompt_tokens = staged.base,
                                        .prefix_reuse_path    = staged.reuse,
                                        .prefix_reuse_source  = staged.reuse_source,
                                        .system_prompt_cache_hit_tokens =
                                            staged.system_prompt_cache_hit_tokens};
    bool host_input_consumed              = staged.host_input_consumed_pending;
    staged.host_input_consumed_pending    = false;
    std::uint32_t processed_prompt_tokens = 0;
    const auto started                    = Clock::now();
    try {
        clear_nonfinite_logits(sequence.lane);
        schedule::PrefillContext schedule_state{
            {device, model, linear_execution.get(), work, decoder->linear_attention,
             replay_records ? &*replay_records : nullptr, io, prefill_hidden, prefill_chunk,
             proposal_head},
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
            dflash_host_ingress,
            sequence.kv ? &sequence.kv->text : nullptr,
            &sequence.text_kv_publication,
            static_cast<std::uint32_t*>(io.text_kv_status.data) + sequence.lane,
            speculative_backend == SpeculativeBackend::Mtp && sequence.kv && sequence.kv->backend
                ? &*sequence.kv->backend
                : nullptr,
            speculative_backend == SpeculativeBackend::Mtp ? &sequence.mtp_kv_publication : nullptr,
            speculative_backend == SpeculativeBackend::Mtp
                ? static_cast<std::uint32_t*>(io.backend_kv_status.data) + sequence.lane
                : nullptr,
            prompt_embedding.get()};

        if (staged.mtp_bridge == MtpBridgeMode::BeforeSuffix) {
            if (staged.cursor != staged.base || staged.base == 0 ||
                staged.cursor >= staged.prompt_tokens) {
                throw std::logic_error("staged MTP bridge is outside the reusable suffix");
            }
            mark_workspace_usage(workspace_plan.mtp_prefill);
            const Tensor& previous_hidden =
                qwen3::detail::mtp_bridge_reads_rewrite_hidden(staged.reuse)
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
                HIP_CHECK(hipMemcpyAsync(bridge_token.data, &token, sizeof(token),
                                         hipMemcpyHostToDevice, device.stream));
                schedule::mtp_bridge_and_propose(schedule_state, bridge_token, previous_hidden,
                                                 bridge.position, bridge.rope_position, false);
            }
            if (sequence.mtp_kv_publication.valid_frontier != staged.base) {
                throw std::logic_error("MTP bridge did not publish the reusable frontier");
            }
            staged.mtp_bridge = MtpBridgeMode::None;
        }

        if (staged.cursor < staged.prompt_tokens) {
            if (speculative_backend == SpeculativeBackend::DFlash) {
                // The owner's DFlash context append reads its own ingress slot. Decode rounds
                // between this owner's steps rewrite the shared ingress, so stage it per step;
                // a mixed round copies the ingress holding its decode rows and this slot.
                const std::size_t slot = max_concurrency - 1U;
                if (mixed == nullptr) { *dflash_host_ingress = {}; }
                const SequenceKVBundle& kv       = sequence.kv.value();
                dflash_host_ingress->lanes[slot] = static_cast<std::int32_t>(sequence.lane);
                dflash_host_ingress->dflash_kv_table_rows[slot] =
                    kv.backend ? kv.backend->bound_row() : 0;
                if (mixed == nullptr) {
                    HIP_CHECK(hipMemcpyAsync(
                        io.dflash_decode.value().ingress.data, dflash_host_ingress,
                        sizeof(qwen3::DFlashDecodeIngress), hipMemcpyHostToDevice, device.stream));
                }
            }
            // A mixed step fills the runner's owner width exactly (the forward width is the
            // aligned unit); other steps take the slice or chunk under the irregular-split rule.
            std::uint32_t step_tokens = prefill_step_tokens(decode_waiting);
            if (step_cap != 0) { step_tokens = std::min(step_tokens, step_cap); }
            const std::uint32_t remaining = staged.prompt_tokens - staged.cursor;
            std::uint32_t nominal = mixed != nullptr
                                        ? std::min(remaining, step_cap)
                                        : schedule::select_prefill_chunk(remaining, step_tokens);
            if (staged.system_prompt_capture && staged.cursor < *staged.system_prompt_capture) {
                nominal = std::min(nominal, *staged.system_prompt_capture - staged.cursor);
            }
            const bool final_candidate = staged.cursor + nominal == staged.prompt_tokens;
            if (final_candidate && request.output != nullptr && request.output->has_token_masks()) {
                // Another prefilling request may have written the shared root row since this
                // request's admission. A mixed round's matcher answers only after the runner
                // publishes, so this host fill does not overlap it.
                bind_prefill_sampling(sequence, request);
            }
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
            if (mixed != nullptr) {
                if (staged.vision || staged.prompt.has_media()) {
                    throw std::logic_error("mixed prefill requires a text-only prompt");
                }
                result = (*mixed)(schedule_state, nominal, final_candidate,
                                  rewrite_checkpoint_capture_frontier);
            } else if (staged.vision) {
                mark_workspace_usage(workspace_plan.vision_encode);
                // While decode rows wait, a Vision encode runs as its own step, so decode rounds
                // resume before the chunk that consumes it.
                if (decode_waiting && staged.vision->encode_ahead(staged.cursor, nominal)) {
                    const double step_seconds =
                        std::chrono::duration<double>(Clock::now() - started).count();
                    staged.elapsed_seconds += step_seconds;
                    return runtime::PrefillStepResult{.summary             = summary,
                                                      .host_input_consumed = host_input_consumed,
                                                      .encoded_only        = true};
                }
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
            if (sequence.text_kv_publication.valid_frontier != staged.cursor) {
                throw std::logic_error(
                    "Text prefill transaction did not publish the realized chunk frontier");
            }
            if (staged.prepare_mtp && sequence.mtp_kv_publication.valid_frontier != staged.cursor) {
                throw std::logic_error(
                    "MTP prefill transaction did not publish the realized chunk frontier");
            }
            if (speculative_backend == SpeculativeBackend::DFlash) {
                sequence.dflash_context_frontier = staged.cursor;
            }
            if (staged.system_prompt_capture && staged.cursor == *staged.system_prompt_capture) {
                capture_system_prompt(sequence, staged, result.processed_tokens);
                staged.system_prompt_capture.reset();
            }
            maybe_freeze_context_checkpoint(sequence, request, result.processed_tokens);
            if (staged.rewrite_checkpoint_capture &&
                staged.cursor - result.processed_tokens <
                    staged.rewrite_checkpoint_capture->frontier &&
                staged.cursor >= staged.rewrite_checkpoint_capture->frontier) {
                // Text prefill ends a chunk exactly at the frontier; the lane state is the
                // checkpoint only there.
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
                if (sequence.mtp_kv_publication.valid_frontier != staged.prompt_tokens) {
                    throw std::logic_error(
                        "MTP exact-hit bridge did not publish the prompt frontier");
                }
                staged.mtp_bridge = MtpBridgeMode::None;
            }
        }

        copy_round_token();
        copy_nonfinite_logits();
        const bool token_logprobs = request.token_logprobs;
        if (token_logprobs) { copy_prefill_logprobs(); }
        std::array<TokenId, qwen3::kMtpDecodeMaximumDrafts> initial_drafts{};
        if (staged.prepare_mtp && staged.initial_mtp_extent != 0) {
            HIP_CHECK(hipMemcpyAsync(initial_drafts.data(), io.mtp.value().draft_tokens.data,
                                     staged.initial_mtp_extent * sizeof(TokenId),
                                     hipMemcpyDeviceToHost, device.stream));
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

        validate_finite_logits(sequence.lane, 1);
        validate_licensed_tokens(std::span<const TokenId>(host_tokens, 1));
        if (sequence.ledger.size() != prompt_tokens) {
            throw std::logic_error("candidate token ledger does not match prompt length");
        }
        sequence.ledger.push_back(host_tokens[0]);
        sequence.prefix_identity.append_generated(1, sequence.rope_delta);
        if (sequence.text_kv_publication.valid_frontier != prompt_tokens) {
            throw std::logic_error("Text prefill did not publish the prompt frontier");
        }
        if (staged.prepare_mtp) {
            if (sequence.mtp_kv_publication.valid_frontier != prompt_tokens) {
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
            if (frontier == 0 || frontier > prompt_tokens ||
                sequence.text_kv_publication.valid_frontier < frontier) {
                throw std::logic_error("rewrite checkpoint was not materialized by Text prefill");
            }
            if (speculative_backend == SpeculativeBackend::Mtp &&
                (!staged.prepare_mtp ||
                 sequence.mtp_kv_publication.valid_frontier < frontier - 1)) {
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
        synchronize_all_while_unwinding(device);
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
    decision_trace::require_eager(use_device_graph);
    layer_boundary_trace::require_eager(use_device_graph);

    auto& ordinary_frame           = io.ordinary.value();
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
            sequence.kv.value().text.bound_row() < 0 || sequence.execution_frontier >= capacity ||
            sequence.text_kv_publication.valid_frontier != sequence.execution_frontier ||
            sequence.ledger_frontier != sequence.execution_frontier + 1 ||
            sequence.ledger.size() != sequence.ledger_frontier ||
            sequence.prefix_identity.size() != sequence.ledger_frontier) {
            throw std::logic_error("ordinary batch row is not decode-ready");
        }
        maximum_frontier = std::max(maximum_frontier, sequence.execution_frontier);
    }

    try {
        DecodeGraphExecutable* executable = nullptr;
        const std::uint32_t transaction_maximum_frontier =
            planned_execution_profile(ordinary_graph_profiles(capacity), maximum_frontier,
                                      "ordinary batch")
                .max;
        if (use_device_graph) {
            DecodeGraphProfile& profile =
                select_graph_profile(ordinary_graphs, static_cast<std::uint32_t>(lanes.size()),
                                     maximum_frontier, "ordinary batch");
            executable = &profile.executable;
        }

        HIP_CHECK(hipMemsetAsync(io.nonfinite_logits.data, 0, io.nonfinite_logits.bytes(),
                                 device.stream));
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

        SegmentedKvTransactionBatch text_transactions(lanes.size(), kv_resolution_words(false));
        const auto* positions =
            static_cast<const std::int32_t*>(ordinary_frame.cache_positions.data);
        const auto* table_rows =
            static_cast<const std::int32_t*>(ordinary_frame.text_kv_table_rows.data);
        auto* status                      = static_cast<std::uint32_t*>(io.text_kv_status.data);
        auto* cursor                      = static_cast<std::uint32_t*>(io.text_kv_cursor.data);
        const std::uint32_t visible_limit = transaction_maximum_frontier + 1U;
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence = sequences[lanes[row]];
            text_transactions.append(decoder->text_kv.begin_device_segmented_append(
                sequence.kv.value().text, sequence.text_kv_publication, positions + row, 0U,
                transaction_maximum_frontier, visible_limit,
                {.status = status + row, .cursor = cursor + row}, table_rows + row));
        }

        bind_token_mask_batch(lanes, ordinary_host_ingress->sampling);
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            ordinary_host_ingress->sampling[row] = token_masks->root(row, device.stream);
        }
        schedule::OrdinaryBatchContext schedule_state{
            {device, model, linear_execution.get(), work, decoder->linear_attention,
             replay_records ? &*replay_records : nullptr, io, prefill_hidden, prefill_chunk,
             proposal_head},
            decoder->text_kv,
            ordinary_frame,
            *ordinary_host_ingress,
            *ordinary_host_egress,
            tail_hidden_store,
            text_transactions.binding()};

        mark_workspace_usage(workspace_plan.ordinary_round);
        const auto start = Clock::now();
        roctx::ScopedRange ordinary_round_range(roctx::Name::DecodeOrdinaryRound,
                                                roctx::Category::Decode, maximum_frontier);
        if (executable != nullptr) { text_transactions.mark_graph_replay(device.stream); }
        order_persistent_decode_after_transfers(lanes.size(), executable);
        schedule::ordinary_decode_batch(schedule_state, static_cast<std::int32_t>(lanes.size()),
                                        executable);
        text_transactions.enqueue_resolution();
        copy_nonfinite_logits();
        const double seconds = synchronize_round_seconds(device, start);
        std::array<std::uint32_t, kMaximumConcurrency> retained_frontiers{};
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            retained_frontiers[row] = sequences[lanes[row]].execution_frontier + 1U;
        }
        for (const std::uint32_t lane : lanes) { validate_finite_logits(lane, 1); }
        text_transactions.finish_resolution({retained_frontiers.data(), lanes.size()});
        decision_trace::record_ordinary(
            ordinary_frame.logits, *ordinary_host_ingress, *ordinary_host_egress,
            static_cast<std::int32_t>(lanes.size()), TextConfig::token_domain);
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence    = sequences[lanes[row]];
            RequestControl& request    = requests[lanes[row]];
            const std::uint32_t base_E = sequence.execution_frontier;
            const std::uint32_t base_S = sequence.ledger_frontier;
            const TokenId token        = ordinary_host_egress->sampled_tokens[row];
            validate_licensed_tokens(std::span<const TokenId>(&token, 1));
            if (sequence.text_kv_publication.valid_frontier != base_E + 1U) {
                throw std::logic_error(
                    "ordinary Text KV transaction did not publish one decode token");
            }
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
        synchronize_all_while_unwinding(device);
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

    const std::uint32_t width = draft_window + 1;
    std::array<std::uint32_t, kMaximumConcurrency> row_ks{};
    std::uint32_t maximum_frontier = 0;
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
            sequence.text_kv_publication.valid_frontier != sequence.execution_frontier ||
            sequence.mtp_kv_publication.valid_frontier != sequence.execution_frontier ||
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
    std::array<const qwen3::AdaptiveDraftState*, kMaximumConcurrency> row_states{};
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        row_states[row] = &requests[lanes[row]].adaptive;
    }
    const std::uint32_t batch_idx = static_cast<std::uint32_t>(lanes.size()) - 1U;
    std::uint32_t batch_k         = qwen3::adaptive_batch_k(
        std::span<const std::uint32_t>(row_ks.data(), lanes.size()), captured_ks);
    if (adaptive_draft) {
        qwen3::AdaptiveRoundTimeState& t_state = adaptive_t_by_batch[batch_idx];
        qwen3::AdaptiveBatchKState& batch_st   = adaptive_batch_k_by_c[batch_idx];
        const auto states =
            std::span<const qwen3::AdaptiveDraftState* const>(row_states.data(), lanes.size());
        const auto caps = std::span<const std::uint32_t>(row_ks.data(), lanes.size());
        if (lanes.size() == 1) {
            qwen3::AdaptiveDraftState& ad = requests[lanes[0]].adaptive;
            if (ad.live_k == 0) {
                qwen3::AdaptiveDraftConfig cfg;
                cfg.captured_ks                      = captured_ks;
                cfg.round_time                       = &t_state;
                cfg.length_tokens                    = maximum_frontier;
                const qwen3::AdaptiveDraftState* ptr = &ad;
                const std::uint32_t row_cap[]        = {row_ks[0]};
                ad.live_k                            = qwen3::adaptive_select_k(
                    cfg, std::span<const qwen3::AdaptiveDraftState* const>(&ptr, 1),
                    std::span<const std::uint32_t>(row_cap, 1), row_ks[0], 0);
            }
            row_ks[0] = std::min(ad.live_k, row_ks[0]);
            batch_k   = qwen3::adaptive_batch_k(
                std::span<const std::uint32_t>(row_ks.data(), lanes.size()), captured_ks);
        } else if (batch_st.live_k == 0) {
            batch_k = qwen3::adaptive_batch_next(batch_st, states, caps, captured_ks, &t_state,
                                                 maximum_frontier);
        } else {
            batch_k = std::min(batch_st.live_k, batch_k);
        }
    }

    try {
        DecodeGraphExecutable* executable = nullptr;
        const std::uint32_t transaction_maximum_frontier =
            planned_execution_profile(mtp_graph_profiles(capacity, batch_k), maximum_frontier,
                                      "MTP batch")
                .max;
        if (use_device_graph) {
            DecodeGraphProfile& profile =
                select_graph_profile(mtp_graphs, static_cast<std::uint32_t>(lanes.size()),
                                     maximum_frontier, "MTP batch", batch_k);
            executable = &profile.executable;
        }

        std::uint32_t realized_extent = 0;
        HIP_CHECK(hipMemsetAsync(io.nonfinite_logits.data, 0, io.nonfinite_logits.bytes(),
                                 device.stream));
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

        SegmentedKvTransactionBatch text_transactions(lanes.size(), kv_resolution_words(false));
        SegmentedKvTransactionBatch mtp_transactions(lanes.size(), kv_resolution_words(true));
        const std::uint32_t target_width = batch_k + 1U;
        const auto* base_frontiers =
            static_cast<const std::int32_t*>(io.mtp_decode->base_frontiers.data);
        const auto* text_table_rows =
            static_cast<const std::int32_t*>(io.mtp_decode->text_kv_table_rows.data);
        const auto* mtp_table_rows =
            static_cast<const std::int32_t*>(io.mtp_decode->mtp_kv_table_rows.data);
        auto* status      = static_cast<std::uint32_t*>(io.text_kv_status.data);
        auto* text_cursor = static_cast<std::uint32_t*>(io.text_kv_cursor.data);
        auto* mtp_status  = static_cast<std::uint32_t*>(io.backend_kv_status.data);
        auto* mtp_cursor  = static_cast<std::uint32_t*>(io.backend_kv_cursor.data);
        const std::uint32_t visible_limit =
            std::min(capacity, transaction_maximum_frontier + std::min(2U * batch_k, capacity));
        const std::uint32_t text_visible_limit =
            std::min(capacity, transaction_maximum_frontier + std::min(target_width, capacity));
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence = sequences[lanes[row]];
            SequenceKVBundle& kv    = sequence.kv.value();
            text_transactions.append(decoder->text_kv.begin_device_segmented_append(
                kv.text, sequence.text_kv_publication, base_frontiers + row, 0U,
                transaction_maximum_frontier, text_visible_limit,
                {.status = status + row, .cursor = text_cursor + row}, text_table_rows + row));
            mtp_transactions.append(decoder->mtp_cache()->begin_device_segmented_append(
                kv.backend.value(), sequence.mtp_kv_publication, base_frontiers + row, 0U,
                transaction_maximum_frontier, visible_limit,
                {.status = mtp_status + row, .cursor = mtp_cursor + row}, mtp_table_rows + row));
        }

        schedule::MtpBatchContext schedule_state{{device, model, linear_execution.get(), work,
                                                  decoder->linear_attention,
                                                  replay_records ? &*replay_records : nullptr, io,
                                                  prefill_hidden, prefill_chunk, proposal_head},
                                                 decoder->text_kv,
                                                 *decoder->mtp_cache(),
                                                 *io.mtp_decode,
                                                 *mtp_host_ingress,
                                                 *mtp_host_egress,
                                                 tail_hidden_store,
                                                 text_transactions.binding(),
                                                 mtp_transactions.binding(),
                                                 token_masks.get()};

        bind_token_mask_batch(lanes, mtp_host_ingress->sampling);
        TokenMaskRoundGuard token_mask_round(*token_masks, device);
        mark_workspace_usage(workspace_plan.mtp_round);
        const auto started = Clock::now();
        if (executable != nullptr) {
            text_transactions.mark_graph_replay(device.stream);
            mtp_transactions.mark_graph_replay(device.stream);
        }
        order_persistent_decode_after_transfers(lanes.size(), executable);
        schedule::mtp_decode_batch(schedule_state, static_cast<std::int32_t>(lanes.size()), batch_k,
                                   executable);
        text_transactions.enqueue_resolution();
        mtp_transactions.enqueue_resolution();
        copy_nonfinite_logits();
        const double seconds = synchronize_round_seconds(device, started);
        std::array<std::uint32_t, kMaximumConcurrency> text_retained_frontiers{};
        std::array<std::uint32_t, kMaximumConcurrency> mtp_retained_frontiers{};
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            const std::int32_t count = mtp_host_egress->licensed_counts[row];
            if (count <= 0 || count > static_cast<std::int32_t>(width)) {
                throw std::runtime_error("MTP batch returned an invalid licensed count");
            }
            const std::uint32_t base = sequences[lanes[row]].execution_frontier;
            text_retained_frontiers[row] =
                base + static_cast<std::uint32_t>(mtp_host_ingress->target_valid_columns[row]);
            mtp_retained_frontiers[row] = base + static_cast<std::uint32_t>(count);
        }
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            validate_finite_logits(lanes[row], static_cast<std::uint32_t>(
                                                   mtp_host_ingress->target_valid_columns[row]));
        }
        text_transactions.finish_resolution({text_retained_frontiers.data(), lanes.size()});
        mtp_transactions.finish_resolution({mtp_retained_frontiers.data(), lanes.size()});
        token_mask_round.finish();
        // Fallback (extent 0) is not a k-draft round; do not fit T(batch_k) from it.
        if (adaptive_draft && realized_extent > 0) {
            qwen3::adaptive_observe_round_time(adaptive_t_by_batch[batch_idx], batch_k,
                                               static_cast<float>(seconds), maximum_frontier);
        }
        std::array<std::uint32_t, kMaximumConcurrency> next_caps{};
        std::uint32_t next_length = 0;
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence                         = sequences[lanes[row]];
            RequestControl& request                         = requests[lanes[row]];
            const qwen3::AdaptiveDraftState adaptive_before = request.adaptive;
            const std::uint32_t base_E                      = sequence.execution_frontier;
            const std::uint32_t base_S                      = sequence.ledger_frontier;
            const std::int32_t count_i                      = mtp_host_egress->licensed_counts[row];
            const std::int32_t accepted_i                   = mtp_host_egress->accepted_drafts[row];
            const std::int32_t next_i                       = mtp_host_egress->next_extents[row];
            if (count_i <= 0 || count_i > static_cast<std::int32_t>(width) || accepted_i < 0 ||
                accepted_i + 1 != count_i || next_i < 0 ||
                next_i > static_cast<std::int32_t>(draft_window) ||
                static_cast<std::uint32_t>(count_i) > budgets[row].generated_tokens_remaining ||
                static_cast<std::uint64_t>(base_E) + static_cast<std::uint32_t>(count_i) >
                    capacity) {
                throw std::runtime_error("MTP batch returned invalid row metadata");
            }
            if (sequence.mtp_kv_publication.valid_frontier !=
                base_E + static_cast<std::uint32_t>(count_i)) {
                throw std::runtime_error(
                    "MTP cache transaction did not publish the licensed frontier");
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
                            qwen3::AdaptiveDraftConfig cfg;
                            cfg.captured_ks   = captured_ks;
                            cfg.round_time    = &adaptive_t_by_batch[batch_idx];
                            cfg.length_tokens = base_E + static_cast<std::uint32_t>(count_i);
                            (void)qwen3::adaptive_draft_next(cfg, request.adaptive,
                                                             static_cast<std::uint32_t>(accepted_i),
                                                             pcur, budget_extent, batch_k);
                        }
                    } else {
                        qwen3::adaptive_record_round(request.adaptive,
                                                     static_cast<std::uint32_t>(accepted_i), pcur,
                                                     batch_k);
                    }
                }
            }
            request.speculative_stats.live_draft_tokens = request.adaptive.live_k;
            request.pending                             = PendingCandidate{
                .kind          = PendingKind::Speculative,
                .base_E        = base_E,
                .base_S        = base_S,
                .prompt_tokens = 0,
                .produced      = static_cast<std::uint32_t>(count_i),
                .text_kv_appended =
                    static_cast<std::uint32_t>(mtp_host_ingress->target_valid_columns[row]),
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
            std::array<qwen3::AdaptiveDraftState*, kMaximumConcurrency> mut_states{};
            for (std::size_t row = 0; row < lanes.size(); ++row) {
                mut_states[row] = &requests[lanes[row]].adaptive;
                row_states[row] = mut_states[row];
            }
            const std::uint32_t next = qwen3::adaptive_batch_next(
                adaptive_batch_k_by_c[batch_idx],
                std::span<const qwen3::AdaptiveDraftState* const>(row_states.data(), lanes.size()),
                std::span<const std::uint32_t>(next_caps.data(), lanes.size()), captured_ks,
                &adaptive_t_by_batch[batch_idx], next_length);
            qwen3::adaptive_assign_live_k(
                std::span<qwen3::AdaptiveDraftState*>(mut_states.data(), lanes.size()), next);
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
        synchronize_all_while_unwinding(device);
        for (const std::uint32_t lane : lanes) {
            if (lane < max_concurrency) { clear_lane(sequences[lane], requests[lane]); }
        }
        throw;
    }
}

runtime::BatchedGeneratedRound ProgramImplCore::decode_dflash_batch(
    std::span<const std::uint32_t> lanes, std::span<const runtime::RoundBudget> budgets,
    std::optional<std::uint32_t> prefill_lane, runtime::PrefillStepResult* prefill_step) {
    std::array<bool, kMaximumConcurrency> cycle_exclusions{};
    if (speculative_backend != SpeculativeBackend::DFlash || !io.dflash_decode || !dflash) {
        throw std::logic_error("DFlash batch execution requires the DFlash backend");
    }
    if (lanes.empty() || lanes.size() > max_concurrency || budgets.size() != lanes.size()) {
        throw std::invalid_argument("DFlash batch membership is invalid");
    }
    decision_trace::require_eager(use_device_graph);
    layer_boundary_trace::require_eager(use_device_graph);
    // This round rewrites every record row: lanes outside it apply their deferred fold first.
    {
        std::array<std::uint32_t, kMaximumConcurrency> outside{};
        std::size_t outside_count = 0;
        for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
            if (sequences[lane].deferred_fold_columns != 0 &&
                std::find(lanes.begin(), lanes.end(), lane) == lanes.end()) {
                outside[outside_count++] = lane;
            }
        }
        if (outside_count != 0) {
            flush_deferred_gdn_folds(std::span<const std::uint32_t>(outside.data(), outside_count));
        }
    }

    const std::uint32_t width = dflash_verify_width;
    std::array<std::uint32_t, kMaximumConcurrency> row_ks{};
    std::uint32_t physical_context_extent = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t maximum_frontier        = 0;
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
            sequence.text_kv_publication.valid_frontier != sequence.execution_frontier ||
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
        physical_context_extent           = std::min(physical_context_extent, cap_extent);
        row_ks[row] = adaptive_draft ? afford : std::min({draft_window, budget_extent, cap_extent});
    }
    std::array<const qwen3::AdaptiveDraftState*, kMaximumConcurrency> dflash_row_states{};
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        dflash_row_states[row] = &requests[lanes[row]].adaptive;
    }
    const std::uint32_t batch_idx = static_cast<std::uint32_t>(lanes.size()) - 1U;
    std::uint32_t batch_k         = qwen3::adaptive_batch_k(
        std::span<const std::uint32_t>(row_ks.data(), lanes.size()), captured_ks);
    if (adaptive_draft) {
        qwen3::AdaptiveRoundTimeState& t_state = adaptive_t_by_batch[batch_idx];
        qwen3::AdaptiveBatchKState& batch_st   = adaptive_batch_k_by_c[batch_idx];
        const auto states                      = std::span<const qwen3::AdaptiveDraftState* const>(
            dflash_row_states.data(), lanes.size());
        const auto caps = std::span<const std::uint32_t>(row_ks.data(), lanes.size());
        qwen3::AdaptiveDraftConfig cfg;
        cfg.captured_ks   = captured_ks;
        cfg.round_time    = &t_state;
        cfg.length_tokens = maximum_frontier;
        const auto previous_k =
            lanes.size() == 1 ? requests[lanes[0]].adaptive.live_k : batch_st.live_k;
        batch_k = qwen3::adaptive_dflash_physical_k(cfg, states, caps, physical_context_extent,
                                                    previous_k);
        if (batch_st.live_k != batch_k) {
            batch_st.rounds_at_k = 0;
        } else {
            ++batch_st.rounds_at_k;
        }
        batch_st.live_k = batch_k;
        std::array<qwen3::AdaptiveDraftState*, kMaximumConcurrency> mut_states{};
        for (std::size_t row = 0; row < lanes.size(); ++row)
            mut_states[row] = &requests[lanes[row]].adaptive;
        qwen3::adaptive_assign_live_k(
            std::span<qwen3::AdaptiveDraftState*>(mut_states.data(), lanes.size()), batch_k);
    }
    const std::uint32_t live_w = dflash_captured_verify_width(batch_k, dflash_verify_width);
    std::uint32_t maximum_target_tokens = 1;
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        maximum_target_tokens =
            std::max(maximum_target_tokens, sequences[lanes[row]].execution_frontier + live_w);
    }
    try {
        DecodeGraphExecutable* executable   = nullptr;
        const GraphExecutionProfile planned = planned_execution_profile(
            dflash_graph_profiles(capacity, batch_k, static_cast<std::uint32_t>(lanes.size()),
                                  live_w),
            maximum_frontier, "DFlash batch");
        const std::uint32_t transaction_maximum_frontier = planned.max;
        const schedule::DFlashEnvelopes envelopes        = dflash_envelopes(planned.max);
        if (prefill_lane && (prefill_step == nullptr || !prefill_mixable(*prefill_lane) ||
                             dflash_uses_tree_verify(batch_k, live_w) ||
                             std::find(lanes.begin(), lanes.end(), *prefill_lane) != lanes.end())) {
            throw std::logic_error("mixed DFlash round has an ineligible prefill owner");
        }
        // A mixed round's aggregate target width differs from every captured profile.
        if (use_device_graph && !prefill_lane) {
            DecodeGraphProfile& profile =
                select_graph_profile(dflash_graphs, static_cast<std::uint32_t>(lanes.size()),
                                     maximum_frontier, "DFlash batch", batch_k);
            executable = &profile.executable;
        }

        // Packed-tree rounds draw deterministic drafts and produce no calibration.
        const bool chain_round      = !dflash_uses_tree_verify(batch_k, live_w);
        const bool calibrate_p_less = calibrates_p_less_drafts() && chain_round;
        std::array<std::uint32_t, kMaximumConcurrency> text_target_columns{};
        std::uint32_t realized_extent = 0;
        HIP_CHECK(hipMemsetAsync(io.nonfinite_logits.data, 0, io.nonfinite_logits.bytes(),
                                 device.stream));
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
            dflash_host_ingress->proposal_extents[row] = static_cast<std::int32_t>(extent);
            // Proposal layers consume only the actual chain prefix. Tree selection later
            // replaces the device count with its real packed width, which is independently
            // predictable from k and W and owns the target-cache transaction extent.
            dflash_host_ingress->target_valid_columns[row] = static_cast<std::int32_t>(extent + 1U);
            text_target_columns[row]                     = dflash_uses_tree_verify(batch_k, live_w)
                                                               ? std::min(live_w, 1U + 2U * batch_k)
                                                               : extent + 1U;
            dflash_host_ingress->text_kv_table_rows[row] = sequence.kv->text.bound_row();
            dflash_host_ingress->dflash_kv_table_rows[row] =
                sequence.kv->backend ? sequence.kv->backend->bound_row() : 0;
            dflash_host_ingress->lanes[row]        = static_cast<std::int32_t>(sequence.lane);
            dflash_host_ingress->rope_deltas[row]  = sequence.rope_delta;
            dflash_host_ingress->logprob_rows[row] = request.token_logprobs ? 1 : 0;
            dflash_host_ingress->sampling[row]     = request.sampling_host;
            ops::SamplingConfig& row_sampling      = dflash_host_ingress->sampling[row];
            if (calibrate_p_less && row_sampling.draft_temperature > 0.0f) {
                row_sampling.draft_temperature = qwen3::p_less_calibrated_draft_temperature(
                    p_less_calibration, row_sampling.temperature, batch_k,
                    row_sampling.draft_temperature);
            }
            if (chain_round && row_sampling.p_less != 0 && row_sampling.temperature > 0.0f &&
                extent > 0) {
                request.speculative_stats.p_less_draft_temperature = row_sampling.draft_temperature;
            }
            materialize_sequence_kv(sequence, std::min(capacity, frontier + dflash_verify_width),
                                    DFlashConfig::full_layers > 0 ? frontier : 0U);
            realized_extent = std::max(realized_extent, extent);
        }
        // Tree verification folds through a path, so its rows take no deferred chain fold.
        if (dflash_uses_tree_verify(batch_k, live_w)) { flush_deferred_gdn_folds(lanes); }
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence = sequences[lanes[row]];
            if (sequence.deferred_fold_columns == 0) { continue; }
            dflash_host_ingress->gdn_fold.row[row] = ops::GdnDeferredFoldRow{
                .linear_state_slot =
                    LinearStateSlots::current_state_slot(lanes[row], max_concurrency),
                .commit_columns = static_cast<std::int32_t>(sequence.deferred_fold_columns),
                .record_row     = static_cast<std::int32_t>(sequence.deferred_fold_row),
            };
            sequence.deferred_fold_columns = 0;
        }

        SegmentedKvTransactionBatch text_transactions(lanes.size(), kv_resolution_words(false));
        const auto* base_frontiers =
            static_cast<const std::int32_t*>(io.dflash_decode->execution_frontiers.data);
        const auto* table_rows =
            static_cast<const std::int32_t*>(io.dflash_decode->text_kv_table_rows.data);
        auto* status = static_cast<std::uint32_t*>(io.text_kv_status.data);
        auto* cursor = static_cast<std::uint32_t*>(io.text_kv_cursor.data);
        const std::uint32_t visible_limit =
            std::min(capacity, transaction_maximum_frontier + std::min(live_w, capacity));
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence = sequences[lanes[row]];
            text_transactions.append(decoder->text_kv.begin_device_segmented_append(
                sequence.kv->text, sequence.text_kv_publication, base_frontiers + row, 0U,
                transaction_maximum_frontier, visible_limit,
                {.status = status + (prefill_lane ? lanes[row] : row), .cursor = cursor + row},
                table_rows + row));
        }

        schedule::DFlashBatchContext schedule_state{
            {device, model, linear_execution.get(), work, decoder->linear_attention,
             replay_records ? &*replay_records : nullptr, io, prefill_hidden, prefill_chunk,
             proposal_head},
            decoder->text_kv,
            *dflash,
            *io.dflash_decode,
            *dflash_host_ingress,
            *dflash_host_egress,
            tail_hidden_store,
            text_transactions.binding(),
            token_masks.get(),
            calibrates_p_less_drafts()};

        bind_token_mask_batch(lanes, dflash_host_ingress->sampling);
        TokenMaskRoundGuard token_mask_round(*token_masks, device);
        mark_workspace_usage(prefill_lane ? workspace_plan.dflash_mixed
                                          : workspace_plan.dflash_round);
        const auto started = Clock::now();
        if (executable != nullptr) { text_transactions.mark_graph_replay(device.stream); }
        if (prefill_lane) {
            // The owner's chunk and the verify batch share one target forward; the prefill
            // workspace bounds their aggregate width.
            const RequestControl& owner = requests[*prefill_lane];
            const std::span<const int> owner_prompt(owner.prefill->prompt.token_ids.data(),
                                                    owner.prefill->prompt.token_ids.size());
            // The owner fills the mixed forward beside the live verify columns, so rounds with
            // fewer decode rows or shorter drafts keep the configured forward width.
            const std::uint32_t verify_columns = static_cast<std::uint32_t>(lanes.size()) * live_w;
            const std::uint32_t forward_tokens = mixed_forward_tokens();
            if (forward_tokens > prefill_chunk || forward_tokens <= verify_columns) {
                throw std::logic_error("prefill chunk cannot hold a mixed DFlash round");
            }
            const MixedChunkRunner runner = [&](schedule::PrefillContext& prefill,
                                                std::uint32_t nominal, bool final_candidate,
                                                std::optional<std::uint32_t> rewrite_frontier) {
                schedule::DFlashMixedOwner mixed_owner{
                    .prefill                             = &prefill,
                    .slice                               = {.prompt          = owner_prompt,
                                                            .nominal_length  = nominal,
                                                            .finalize_at_end = final_candidate,
                                                            .sampling        = prefill.sampling},
                    .rewrite_checkpoint_capture_frontier = rewrite_frontier,
                };
                schedule::dflash_mixed_batch(schedule_state, mixed_owner,
                                             static_cast<std::int32_t>(lanes.size()), batch_k,
                                             live_w, envelopes);
                return mixed_owner.result;
            };
            *prefill_step = advance_prefill(sequences[*prefill_lane], requests[*prefill_lane], true,
                                            &runner, forward_tokens - verify_columns);
        } else {
            order_persistent_decode_after_transfers(lanes.size(), executable);
            schedule::dflash_decode_batch(schedule_state, static_cast<std::int32_t>(lanes.size()),
                                          batch_k, live_w, envelopes, executable);
        }
        text_transactions.enqueue_resolution();
        copy_nonfinite_logits();
        const double seconds = synchronize_round_seconds(device, started);
        std::array<std::uint32_t, kMaximumConcurrency> retained_frontiers{};
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            retained_frontiers[row] =
                sequences[lanes[row]].execution_frontier + text_target_columns[row];
        }
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            validate_finite_logits(lanes[row], text_target_columns[row]);
        }
        text_transactions.finish_resolution({retained_frontiers.data(), lanes.size()});
        decision_trace::record_dflash(
            io.dflash_decode->target_logits, io.dflash_decode->target_argmax,
            io.dflash_decode->cache_positions, io.dflash_decode->verify_ids, *dflash_host_ingress,
            *dflash_host_egress,
            std::span<const std::uint32_t>(text_target_columns.data(), lanes.size()),
            static_cast<std::int32_t>(lanes.size()), static_cast<std::int32_t>(width),
            static_cast<std::int32_t>(live_w), TextConfig::token_domain,
            dflash_uses_tree_verify(batch_k, live_w), token_masks->published_sampling(),
            io.dflash_decode->selector_ids, io.dflash_decode->selector_q);
        token_mask_round.finish();
        if (calibrate_p_less) {
            const std::size_t row_values =
                static_cast<std::size_t>(ops::kPLessProposalCalibrationTemperatureCount) * batch_k;
            for (std::size_t row = 0; row < lanes.size(); ++row) {
                const ops::SamplingConfig& row_sampling = dflash_host_ingress->sampling[row];
                if (row_sampling.p_less == 0 || !(row_sampling.temperature > 0.0f) ||
                    !(row_sampling.draft_temperature > 0.0f)) {
                    continue;
                }
                qwen3::p_less_calibration_observe(
                    p_less_calibration, row_sampling.temperature,
                    std::span<const float>(dflash_host_egress->proposal_calibration.data() +
                                               row * row_values,
                                           row_values),
                    static_cast<std::uint32_t>(dflash_host_ingress->proposal_extents[row]));
            }
        }
        // Fallback (extent 0) is not a k-draft round; do not fit T(batch_k) from it, nor from
        // a mixed round whose time includes the owner's chunk.
        if (adaptive_draft && realized_extent > 0 && !prefill_lane) {
            qwen3::adaptive_observe_round_time(adaptive_t_by_batch[batch_idx], batch_k,
                                               static_cast<float>(seconds), maximum_frontier);
        }
        if (ninfer::targets::qwen3::detail::dflash_candidate_stats_enabled() &&
            io.dflash_decode.has_value()) {
            qwen3::DFlashDecodeState& frame = *io.dflash_decode;
            const int w                     = static_cast<int>(width);
            const int b                     = static_cast<int>(lanes.size());
            std::vector<std::int32_t> ids(static_cast<std::size_t>(w) *
                                          static_cast<std::size_t>(b));
            std::vector<std::int32_t> pars(static_cast<std::size_t>(w) *
                                           static_cast<std::size_t>(b));
            HIP_CHECK(hipMemcpy(ids.data(), frame.verify_ids.data,
                                ids.size() * sizeof(std::int32_t), hipMemcpyDeviceToHost));
            HIP_CHECK(hipMemcpy(pars.data(), frame.parent_index.data,
                                pars.size() * sizeof(std::int32_t), hipMemcpyDeviceToHost));
            for (int row = 0; row < b; ++row) {
                const int count =
                    dflash_host_egress->licensed_counts[static_cast<std::size_t>(row)];
                ninfer::targets::qwen3::detail::dflash_candidate_stats::record_round(
                    ids.data() + static_cast<std::size_t>(row) * static_cast<std::size_t>(w),
                    pars.data() + static_cast<std::size_t>(row) * static_cast<std::size_t>(w),
                    dflash_host_egress->licensed_tokens.data() +
                        static_cast<std::size_t>(row) * static_cast<std::size_t>(w),
                    count, w, static_cast<int>(draft_window));
            }
        }
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            SequenceState& sequence                         = sequences[lanes[row]];
            RequestControl& request                         = requests[lanes[row]];
            const qwen3::AdaptiveDraftState adaptive_before = request.adaptive;
            const std::uint32_t base_E                      = sequence.execution_frontier;
            const std::uint32_t base_S                      = sequence.ledger_frontier;
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
            // Observe only actual proposals. Select the next physical route at
            // its round boundary, after publication determines the surviving
            // compact batch and its real per-row budgets.
            if (adaptive_draft && extent > 0) {
                qwen3::adaptive_record_round(request.adaptive,
                                             static_cast<std::uint32_t>(accepted_i), extent,
                                             batch_k, true);
            }
            sequence.dflash_context_frontier            = base_E;
            request.speculative_stats.live_draft_tokens = request.adaptive.live_k;
            request.pending                             = PendingCandidate{
                .kind             = PendingKind::Speculative,
                .base_E           = base_E,
                .base_S           = base_S,
                .prompt_tokens    = 0,
                .produced         = static_cast<std::uint32_t>(count_i),
                .text_kv_appended = text_target_columns[row],
                .drafted          = extent,
                .round_k          = batch_k,
                .verify_width     = live_w,
                .tree_verify      = dflash_uses_tree_verify(batch_k, live_w),
                .adaptive_before  = adaptive_before,
            };
            request.lifecycle = Lifecycle::Pending;
            request.timings.decode_seconds += seconds;
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
        synchronize_all_while_unwinding(device);
        for (const std::uint32_t lane : lanes) {
            if (lane < max_concurrency) { clear_lane(sequences[lane], requests[lane]); }
        }
        throw;
    }
}

bool ProgramImplCore::prefill_mixable(std::uint32_t lane) const noexcept {
    if (lane >= max_concurrency || speculative_backend != SpeculativeBackend::DFlash) {
        return false;
    }
    const RequestControl& request = requests[lane];
    if (request.lifecycle != Lifecycle::Prefilling || !request.prefill) { return false; }
    const RequestControl::Prefill& staged = *request.prefill;
    if (staged.vision || staged.prompt.has_media() || staged.prepare_mtp ||
        staged.mtp_bridge != MtpBridgeMode::None || staged.cursor >= staged.prompt_tokens) {
        return false;
    }
    return true;
}

runtime::MixedGeneratedRound
ProgramImplCore::decode_batch_with_prefill(std::span<const std::uint32_t> lanes,
                                           std::span<const runtime::RoundBudget> budgets,
                                           std::uint32_t prefill_lane) {
    runtime::MixedGeneratedRound mixed;
    mixed.round = decode_dflash_batch(lanes, budgets, prefill_lane, &mixed.prefill);
    return mixed;
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
    trim_sequence_kv(sequence, sequence.text_kv_publication.valid_frontier,
                     backend_kv_valid(sequence));
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
    DeviceArena& weights = *model.weights_arena;
    out.weights = ArenaMemorySummary{weights.capacity(), weights.used(), weights.peak_used()};
    out.sequence =
        ArenaMemorySummary{persistent.capacity(), persistent.used(), persistent.peak_used()};
    out.workspace = ArenaMemorySummary{workspace_storage.capacity(), work.used(), work.peak_used()};
    out.workspace_logical_peak_bytes = workspace_logical_peak_bytes;
    out.device_graph_allowance_bytes = graph_allowance_bytes;
    out.device_graph_observed_bytes  = graph_observed_bytes;
    out.kv_payload_bytes             = kv_payload_bytes;
    out.checkpoint_image_host_bytes =
        checkpoint_image_slab_ ? checkpoint_image_slab_->capacity() : 0;
    out.checkpoint_image_pool_heads = context_checkpoint_pool_.capacity();
    const qwen3::detail::KvRamSnapshot ram =
        kv_ram_cache_ ? kv_ram_cache_->snapshot() : qwen3::detail::KvRamSnapshot{};
    out.kv_ram_capacity_bytes = ram.capacity_bytes;
    out.kv_ram_used_bytes     = ram.used_bytes;
    out.kv_ram_entry_count    = ram.entry_count;
    const qwen3::detail::KvDiskSnapshot disk =
        kv_disk_cache_ ? kv_disk_cache_->snapshot() : qwen3::detail::KvDiskSnapshot{};
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
    const std::size_t score_tensor_bytes =
        static_cast<std::size_t>(TextConfig::output_rows) * static_cast<std::size_t>(kSlice) *
            dtype_size(DType::BF16) +
        static_cast<std::size_t>(kSlice) * (2 * sizeof(std::int32_t) + sizeof(float)) + 4096;
    DeviceArena score_workspace(score_tensor_bytes);
    std::vector<float> host_nll(static_cast<std::size_t>(kSlice));
    std::vector<std::int32_t> host_argmax(static_cast<std::size_t>(kSlice));

    for (std::int32_t begin = 0; begin < scored; begin += kSlice) {
        const std::int32_t width = std::min(kSlice, scored - begin);
        score_workspace.reset();
        Tensor hidden        = prefill_hidden.slice(1, hidden_origin + begin, width);
        Tensor logits        = score_workspace.alloc(DType::BF16, {TextConfig::output_rows, width});
        Tensor target_tensor = score_workspace.alloc(DType::I32, {width});
        Tensor nll           = score_workspace.alloc(DType::FP32, {width});
        Tensor argmax        = score_workspace.alloc(DType::I32, {width});
        HIP_CHECK(hipMemcpyAsync(target_tensor.data, targets.data() + begin,
                                 static_cast<std::size_t>(width) * sizeof(std::int32_t),
                                 hipMemcpyHostToDevice, device.stream));
        if (linear_execution != nullptr) {
            linear_execution->linear(hidden, model.output_head, logits, score_workspace,
                                     device.stream);
        } else {
            ops::linear(hidden, model.output_head, logits, score_workspace, device.stream);
        }
        ops::nll_from_logits(logits, target_tensor, nll, TextConfig::token_domain, device.stream);
        ops::argmax(logits, argmax, TextConfig::token_domain, device.stream);
        HIP_CHECK(hipMemcpyAsync(host_nll.data(), nll.data,
                                 static_cast<std::size_t>(width) * sizeof(float),
                                 hipMemcpyDeviceToHost, device.stream));
        HIP_CHECK(hipMemcpyAsync(host_argmax.data(), argmax.data,
                                 static_cast<std::size_t>(width) * sizeof(std::int32_t),
                                 hipMemcpyDeviceToHost, device.stream));
        HIP_CHECK(hipStreamSynchronize(device.stream));
        for (std::int32_t i = 0; i < width; ++i) {
            const double value = static_cast<double>(host_nll[static_cast<std::size_t>(i)]);
            record_score_nll(result, value);
            if (std::isfinite(value)) {
                result.argmax_token_ids.push_back(host_argmax[static_cast<std::size_t>(i)]);
            }
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
    Tensor argmax                  = score_workspace.alloc(DType::I32, {1});
    const std::int32_t host_target = target;
    HIP_CHECK(hipMemcpyAsync(target_tensor.data, &host_target, sizeof(host_target),
                             hipMemcpyHostToDevice, device.stream));
    ops::nll_from_logits(logits, target_tensor, nll, TextConfig::token_domain, device.stream);
    ops::argmax(logits, argmax, TextConfig::token_domain, device.stream);
    float host_nll           = 0.0f;
    std::int32_t host_argmax = 0;
    HIP_CHECK(hipMemcpyAsync(&host_nll, nll.data, sizeof(host_nll), hipMemcpyDeviceToHost,
                             device.stream));
    HIP_CHECK(hipMemcpyAsync(&host_argmax, argmax.data, sizeof(host_argmax), hipMemcpyDeviceToHost,
                             device.stream));
    HIP_CHECK(hipStreamSynchronize(device.stream));
    const double value = static_cast<double>(host_nll);
    record_score_nll(result, value);
    if (std::isfinite(value)) { result.argmax_token_ids.push_back(host_argmax); }
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
    // Teacher forcing owns one idle, complete scoring reservation and has no live scheduler
    // to grant generation windows. Keep its complete represented token extent funded.
    base.impl_->summary.admission            = base.impl_->summary.lifetime_admission;
    base.impl_->summary.window_output_tokens = base.impl_->summary.effective_output_tokens;
    base.impl_->text_kv_page_entitlement     = base.impl_->summary.admission.main_kv_pages;
    base.impl_->backend_kv_page_entitlement  = base.impl_->summary.admission.backend_kv_pages;
    RequestPlan plan                         = plan_request_for_lane(0, prompt, base);

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
    flush_deferred_gdn_folds();
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
    if (prompt_tokens > skip + 1) {
        result.token_nlls.reserve(prompt_tokens - skip - 1);
        result.argmax_token_ids.reserve(prompt_tokens - skip - 1);
    }
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
    if (result.argmax_token_ids.size() != result.tokens_scored) {
        throw std::logic_error("score argmax sequence lost teacher-forced alignment");
    }
    return result;
}

} // namespace ninfer::targets::qwen3::detail::NINFER_QWEN3_RUNTIME_NS
