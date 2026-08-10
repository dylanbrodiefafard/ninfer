#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/program.h"

#include "core/nvtx.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/speculative_round.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {
namespace {

using Clock = std::chrono::steady_clock;

std::int32_t checked_i32(std::uint32_t value, const char* label) {
    if (value > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error(label);
    }
    return static_cast<std::int32_t>(value);
}

std::uint32_t final_prefill_chunk_length(std::uint32_t base, std::uint32_t end, std::uint32_t chunk,
                                         std::optional<std::uint32_t> boundary) {
    std::uint32_t cursor = base;
    std::uint32_t last   = 0;
    while (cursor < end) {
        last = std::min(chunk, end - cursor);
        if (boundary && *boundary > cursor && *boundary < cursor + last) {
            last = *boundary - cursor;
        }
        cursor += last;
    }
    if (last == 0) { throw std::logic_error("prefill suffix is empty"); }
    return last;
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

schedule::MtpGqaEnvelopes mtp_gqa_envelopes(std::uint32_t min_frontier, std::uint32_t max_frontier,
                                            std::uint32_t k, std::uint32_t capacity) {
    const auto visible = [capacity](std::uint64_t value) {
        return static_cast<std::uint32_t>(std::min<std::uint64_t>(capacity, value));
    };
    schedule::MtpGqaEnvelopes out;
    out.target_verify = {visible(static_cast<std::uint64_t>(min_frontier) + 1ULL),
                         visible(static_cast<std::uint64_t>(max_frontier) + k + 1ULL)};
    out.batch         = out.target_verify;
    for (std::uint32_t step = 0; step + 1 < k; ++step) {
        out.ar[step] = {visible(static_cast<std::uint64_t>(min_frontier) + step + 2ULL),
                        visible(static_cast<std::uint64_t>(max_frontier) + k + step + 2ULL)};
    }
    return out;
}

schedule::DFlashEnvelopes dflash_envelopes(std::uint32_t min_frontier, std::uint32_t max_frontier,
                                           std::uint32_t k) {
    return schedule::DFlashEnvelopes{
        .local  = {min_frontier, max_frontier},
        .full   = {min_frontier, max_frontier},
        .append = {1, k + 1},
    };
}

DecodeGraphProfile& select_graph_profile(DecodeGraphFamily& family, std::uint32_t batch_size,
                                         std::uint32_t frontier, const char* label) {
    const auto it = std::find_if(
        family.profiles.begin(), family.profiles.end(), [&](const DecodeGraphProfile& profile) {
            return profile.batch_size == batch_size && profile.min_execution_frontier <= frontier &&
                   frontier <= profile.max_execution_frontier;
        });
    if (it == family.profiles.end()) {
        throw std::logic_error(std::string(label) + " CUDA Graph coverage is incomplete");
    }
    return *it;
}

DecodeGraphProfile& select_graph_profile(DecodeGraphFamily& family, std::uint32_t frontier,
                                         const char* label) {
    return select_graph_profile(family, 1, frontier, label);
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

DecodeGraphExecutable& install_graph_profile(DecodeGraphFamily& family, DecodeGraphProfile& profile,
                                             const char* label) {
    DecodeGraphTopology& topology   = select_graph_topology(family, profile.topology_class, label);
    const std::size_t profile_index = static_cast<std::size_t>(&profile - family.profiles.data());
    if (topology.installed_profile != profile_index) {
        topology.executable.update(profile.definition);
        topology.installed_profile = profile_index;
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

    const auto replay_profile = [&](DecodeGraphTopology& topology, std::size_t profile_index) {
        DecodeGraphProfile& profile = family.profiles[profile_index];
        prepare(profile.min_execution_frontier);
        device.synchronize();
        if (topology.installed_profile != profile_index) {
            topology.executable.update(profile.definition);
            topology.installed_profile = profile_index;
        }
        topology.executable.launch(device.stream);
        device.synchronize();
    };

    for (DecodeGraphTopology& topology : family.topologies) {
        for (std::size_t i = 0; i < family.profiles.size(); ++i) {
            if (family.profiles[i].topology_class == topology.topology_class) {
                replay_profile(topology, i);
            }
        }
        for (std::size_t i = family.profiles.size(); i-- > 0;) {
            if (family.profiles[i].topology_class == topology.topology_class) {
                replay_profile(topology, i);
            }
        }
    }
}

} // namespace

ProgramImplCore::ProgramImplCore(const LoadedModelData& model_in, const SequencePlanImpl& plan,
                                 DeviceContext& device_in)
    : model(model_in), device(device_in), capacity(plan.capacity),
      max_concurrency(plan.max_concurrency), prefill_chunk(plan.prefill_chunk),
      draft_window(plan.draft_window), speculative_backend(plan.speculative_backend),
      kv_dtype(plan.kv_dtype), kv_quant_group(plan.kv_quant_group),
      proposal_head(plan.proposal_head), vision_enabled(plan.features.vision),
      use_cuda_graph(plan.use_cuda_graph), kv_payload_bytes(plan.persistent.kv_payload_bytes),
      graph_allowance_bytes(plan.graph_allowance_bytes), workspace_plan(plan.workspace),
      persistent(plan.persistent.bytes), workspace_storage(plan.workspace.capacity),
      work(DeviceSpan{workspace_storage.base(), workspace_storage.capacity()}),
      round_host((static_cast<std::size_t>(draft_window) + 2ULL) * sizeof(std::int32_t)),
      ordinary_host(sizeof(qwen3_6::OrdinaryDecodeIngress) + sizeof(qwen3_6::OrdinaryDecodeEgress)),
      mtp_host(sizeof(qwen3_6::MtpDecodeIngress) + sizeof(qwen3_6::MtpDecodeEgress)) {
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
    if (model.dflash.has_value() && model.vision.has_value()) {
        throw std::invalid_argument("DFlash and Vision model views are mutually exclusive");
    }
    const DeviceSpan backing = persistent.alloc_bytes(plan.persistent.bytes, 256);
    decoder = std::make_unique<qwen3_6::DecoderState>(backing, plan.persistent.decoder);
    if (plan.persistent.dflash) {
        active_sequence().dflash.emplace(backing, *plan.persistent.dflash);
    }
    if (active_sequence().dflash.has_value() != plan.features.dflash()) {
        throw std::logic_error("DFlash state does not match the frozen sequence plan");
    }

    io = qwen3_6::RoundState(backing, plan.persistent.round);
    if (io.mtp.has_value() != (speculative_backend == SpeculativeBackend::Mtp)) {
        throw std::logic_error("round-state MTP extension does not match the sequence plan");
    }
    if (io.mtp_decode.has_value() != (speculative_backend == SpeculativeBackend::Mtp)) {
        throw std::logic_error("MTP decode frame does not match the sequence plan");
    }
    prefill_hidden        = plan.persistent.prefill_hidden.bind(backing);
    token_counts          = plan.persistent.token_counts.bind(backing);
    sampling_config       = plan.persistent.sampling_config.bind(backing);
    tail_hidden_store     = plan.persistent.tail_hidden.bind(backing);
    boundary_hidden_store = plan.persistent.boundary_hidden.bind(backing);
    const std::int32_t linear_slots_per_sequence =
        LinearStateSlots::required_slot_count(draft_window);
    for (std::uint32_t lane = 0; lane < max_concurrency; ++lane) {
        SequenceState& sequence    = sequences[lane];
        sequence.lane              = lane;
        sequence.linear_state_base = static_cast<std::int32_t>(lane) * linear_slots_per_sequence;
        sequence.linear_state_capacity = linear_slots_per_sequence;
        sequence.current_linear_state_slot =
            LinearStateSlots::prefill_working_slot(sequence.linear_state_base);
        sequence.tail_hidden = tail_hidden_store.slice(1, static_cast<std::int32_t>(lane), 1);
        sequence.boundary_hidden =
            boundary_hidden_store.slice(1, static_cast<std::int32_t>(lane), 1);
        sequence.ledger.reserve(static_cast<std::size_t>(capacity) + 1ULL);
        sequence.prefix_identity.reserve(static_cast<std::size_t>(capacity) + 1ULL);
    }

    set_device_i32(io.text_kv_table_row, 0);
    set_device_i32(io.backend_kv_table_row, 0);

    host_count            = static_cast<std::int32_t*>(round_host.data());
    host_tokens           = reinterpret_cast<TokenId*>(host_count + 1);
    ordinary_host_ingress = static_cast<qwen3_6::OrdinaryDecodeIngress*>(ordinary_host.data());
    ordinary_host_egress  = reinterpret_cast<qwen3_6::OrdinaryDecodeEgress*>(
        static_cast<unsigned char*>(ordinary_host.data()) + sizeof(qwen3_6::OrdinaryDecodeIngress));
    *ordinary_host_ingress = {};
    *ordinary_host_egress  = {};
    mtp_host_ingress       = static_cast<qwen3_6::MtpDecodeIngress*>(mtp_host.data());
    mtp_host_egress        = reinterpret_cast<qwen3_6::MtpDecodeEgress*>(
        static_cast<unsigned char*>(mtp_host.data()) + sizeof(qwen3_6::MtpDecodeIngress));
    *mtp_host_ingress = {};
    *mtp_host_egress  = {};
    CUDA_CHECK(cudaMemsetAsync(io.speculative.produced_count.data, 0,
                               io.speculative.produced_count.bytes(), device.stream));
    CUDA_CHECK(cudaMemsetAsync(io.speculative.current_proposal_extent.data, 0,
                               io.speculative.current_proposal_extent.bytes(), device.stream));
    CUDA_CHECK(cudaMemsetAsync(io.rope_delta.data, 0, io.rope_delta.bytes(), device.stream));
    CUDA_CHECK(cudaMemsetAsync(io.speculative.accepted_drafts.data, 0,
                               io.speculative.accepted_drafts.bytes(), device.stream));
    CUDA_CHECK(cudaMemsetAsync(io.linear_state_read_slot.data, 0, io.linear_state_read_slot.bytes(),
                               device.stream));
    CUDA_CHECK(cudaMemsetAsync(io.linear_state_snapshot_base_slot.data, 0,
                               io.linear_state_snapshot_base_slot.bytes(), device.stream));
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
}

ProgramImplCore::~ProgramImplCore() noexcept {
    if (device.stream != nullptr) { (void)cudaStreamSynchronize(device.stream); }
}

RequestPlan ProgramImplCore::plan_request_for_lane(std::uint32_t lane,
                                                   const PreparedPromptData& prompt,
                                                   const ExecutionOptions& options) {
    if (lane >= max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    active_lane_ = lane;
    return plan_request(prompt, options);
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
        if (other == lane || requests[other].lifecycle != Lifecycle::Resident ||
            !sequences[other].kv) {
            continue;
        }
        reclaimable_text += sequences[other].kv->text.page_entitlement();
        if (sequences[other].kv->backend) {
            reclaimable_backend += sequences[other].kv->backend->page_entitlement();
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

runtime::PrefillStepResult ProgramImplCore::start_prefill_lane(std::uint32_t lane,
                                                               PreparedPromptData&& prompt,
                                                               RequestPlan&& plan,
                                                               runtime::TransientRegion transient) {
    if (lane >= max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    active_lane_ = lane;
    if (speculative_backend == SpeculativeBackend::DFlash) {
        throw std::logic_error("DFlash does not support staged concurrent prefill");
    }
    if (plan.impl_ == nullptr) { throw std::invalid_argument("request plan is empty"); }
    RequestPlanImpl& request_plan = *plan.impl_;
    if (active_request().lifecycle == Lifecycle::Prefilling ||
        active_request().lifecycle == Lifecycle::Active ||
        active_request().lifecycle == Lifecycle::Pending) {
        throw std::logic_error("staged prefill requires a free request lane");
    }

    const std::uint32_t prompt_tokens = static_cast<std::uint32_t>(prompt.token_ids.size());
    if (prompt_tokens != request_plan.summary.prompt_tokens ||
        (request_plan.vision.has_value() && !prompt.has_media())) {
        throw std::invalid_argument("request plan does not describe the prepared prompt");
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
        (active_request().lifecycle != Lifecycle::Resident ||
         !qwen3_6::detail::prefix_matches(prompt, active_sequence().ledger,
                                          active_sequence().prefix_identity,
                                          request_plan.reuse_base))) {
        throw std::logic_error("planned resident prefix is no longer reusable");
    }
    if (request_plan.reuse == ReusePath::RestoreBoundary &&
        (!active_sequence().boundary.valid ||
         active_sequence().boundary.boundary != request_plan.reuse_base)) {
        throw std::logic_error("planned sequence boundary checkpoint is unavailable");
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
    active_request().lifecycle = Lifecycle::Invalid;
    try {
        if (request_plan.reuse == ReusePath::FullReset) {
            active_sequence().kv.reset();
            ordered_reset();
            active_sequence().ledger.clear();
            active_sequence().text_kv_valid = 0;
            active_sequence().mtp_kv_valid  = 0;
            reserve_sequence_kv(request_plan.text_kv_page_entitlement,
                                request_plan.backend_kv_page_entitlement);
        } else if (request_plan.reuse == ReusePath::AppendAtFrontier) {
            if (!active_sequence().kv) {
                throw std::logic_error("resident prefix has no KV allocation bundle");
            }
            if (active_sequence().text_kv_valid < base) {
                throw std::logic_error("resident Text KV is shorter than the append frontier");
            }
            if (speculative_backend == SpeculativeBackend::Mtp) {
                const std::uint32_t mtp_base = base == 0 ? 0 : base - 1;
                if (!request_plan.prepare_mtp || active_sequence().mtp_kv_valid < mtp_base) {
                    throw std::logic_error("resident MTP KV is shorter than the bridge frontier");
                }
                active_sequence().mtp_kv_valid = mtp_base;
            }
            trim_sequence_kv(base, backend_kv_valid());
            resize_sequence_kv_entitlement(request_plan.text_kv_page_entitlement,
                                           request_plan.backend_kv_page_entitlement);
            active_sequence().text_kv_valid = base;
            active_sequence().ledger.resize(base);
            set_device_i32(io.linear_state_read_slot, active_sequence().current_linear_state_slot);
        } else {
            if (!active_sequence().kv || active_sequence().text_kv_valid < base) {
                throw std::logic_error("resident boundary has no complete KV allocation");
            }
            active_sequence().text_kv_valid = base;
            if (speculative_backend == SpeculativeBackend::Mtp) {
                const std::uint32_t mtp_base = base == 0 ? 0 : base - 1;
                if (!request_plan.prepare_mtp || !active_sequence().boundary.mtp_prefix_valid ||
                    active_sequence().mtp_kv_valid < mtp_base) {
                    throw std::logic_error("boundary MTP KV is shorter than the bridge frontier");
                }
                active_sequence().mtp_kv_valid = mtp_base;
            }
            trim_sequence_kv(base, backend_kv_valid());
            resize_sequence_kv_entitlement(request_plan.text_kv_page_entitlement,
                                           request_plan.backend_kv_page_entitlement);
            decoder->linear_attention.copy_slot(
                LinearStateSlots::prefix_boundary_slot(active_sequence().linear_state_base,
                                                       active_sequence().linear_state_capacity),
                LinearStateSlots::prefill_working_slot(active_sequence().linear_state_base),
                device.stream);
            active_sequence().current_linear_state_slot =
                LinearStateSlots::prefill_working_slot(active_sequence().linear_state_base);
            set_device_i32(io.linear_state_read_slot, active_sequence().current_linear_state_slot);
            active_sequence().ledger.resize(base);
        }

        trim_sequence_kv(base, backend_kv_valid());
        bind_sequence_kv();
        const std::uint32_t backend_materialized =
            speculative_backend == SpeculativeBackend::Mtp
                ? std::min(capacity,
                           prompt_tokens + (initial_mtp_extent == 0 ? 0U : initial_mtp_extent - 1U))
                : 0U;
        materialize_sequence_kv(prompt_tokens, backend_materialized);
        install_sampling(request_plan.sampling);
        active_sequence().rope_delta = prompt.rope_delta;
        set_device_i32(io.rope_delta, active_sequence().rope_delta);

        active_sequence().boundary = {};
        active_request().timings   = {};
        active_request().pending   = {};
        ++active_sequence().request_epoch;
        active_sequence().dflash_proposal_ready = false;
        active_sequence().mtp_draft_count       = 0;
        active_sequence().pending_context_valid = false;
        active_sequence().ordinary_tail         = false;
        active_sequence().tail_hidden_valid =
            base == prompt_tokens && active_sequence().tail_hidden_valid;
        active_sequence().ledger.assign(prompt.token_ids.begin(), prompt.token_ids.end());
        active_sequence().prefix_identity.assign(prompt);

        const bool host_input_consumed = prompt.has_media() && !request_plan.vision;
        if (host_input_consumed) { prompt.release_media_payload(); }

        RequestControl::Prefill prefill{
            .prompt                      = std::move(prompt),
            .vision_plan                 = std::move(request_plan.vision),
            .vision                      = nullptr,
            .transient                   = transient,
            .snapshot_boundary           = request_plan.snapshot_boundary,
            .base                        = base,
            .cursor                      = base,
            .prompt_tokens               = prompt_tokens,
            .initial_mtp_extent          = initial_mtp_extent,
            .elapsed_seconds             = 0.0,
            .host_input_consumed_pending = host_input_consumed,
            .prepare_mtp                 = request_plan.prepare_mtp,
            .needs_mtp_bridge            = request_plan.needs_mtp_bridge,
            .mtp_bridge_complete         = !request_plan.needs_mtp_bridge,
            .mtp_bridge_uses_boundary    = request_plan.reuse == ReusePath::RestoreBoundary,
        };
        active_request().prefill.emplace(std::move(prefill));
        auto& staged = *active_request().prefill;
        if (staged.vision_plan) {
            staged.vision = std::make_unique<schedule::VisionPrefillSession>(
                device, model, work, staged.prompt, *staged.vision_plan, staged.transient);
        }
        staged.elapsed_seconds     = std::chrono::duration<double>(Clock::now() - started).count();
        active_request().lifecycle = Lifecycle::Prefilling;
        return advance_prefill();
    } catch (...) {
        try {
            device.synchronize();
        } catch (...) {}
        make_invalid();
        throw;
    }
}

runtime::PrefillStepResult ProgramImplCore::advance_prefill_lane(std::uint32_t lane) {
    if (lane >= max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    active_lane_ = lane;
    return advance_prefill();
}

void ProgramImplCore::resolve_pending_lane(std::uint32_t lane, std::uint32_t accepted_tokens,
                                           bool terminal) {
    if (lane >= max_concurrency) { throw std::out_of_range("request lane is out of range"); }
    active_lane_ = lane;
    resolve_pending(accepted_tokens, terminal);
}

void ProgramImplCore::resolve_pending_batch(std::span<const std::uint32_t> lanes,
                                            std::span<const std::uint32_t> accepted_tokens,
                                            std::span<const std::uint8_t> terminal,
                                            std::span<const std::uint8_t> cancelled) {
    if (lanes.empty() || lanes.size() > max_concurrency || accepted_tokens.size() != lanes.size() ||
        terminal.size() != lanes.size() || cancelled.size() != lanes.size()) {
        throw std::invalid_argument("pending batch resolution has inconsistent membership");
    }

    bool needs_hidden_correction = false;
    std::array<std::int32_t, kMaximumConcurrency> selectors{};
    for (std::size_t row = 0; row < lanes.size(); ++row) {
        const std::uint32_t lane = lanes[row];
        if (lane >= max_concurrency || requests[lane].lifecycle != Lifecycle::Pending) {
            throw std::logic_error("pending batch membership no longer matches Program state");
        }
        const PendingCandidate& pending = requests[lane].pending;
        if ((speculative_backend == SpeculativeBackend::Mtp &&
             (!pending.batch_frame || pending.frame_row != row)) ||
            (speculative_backend == SpeculativeBackend::None &&
             (pending.batch_frame || pending.kind != PendingKind::Ordinary))) {
            throw std::logic_error("pending batch metadata does not match its decode backend");
        }
        if (!cancelled[row] && terminal[row] && accepted_tokens[row] < pending.produced) {
            if (accepted_tokens[row] == 0) {
                throw std::logic_error("terminal MTP row cannot commit an empty licensed prefix");
            }
            selectors[row]          = static_cast<std::int32_t>(accepted_tokens[row] - 1);
            needs_hidden_correction = true;
        } else {
            selectors[row] = static_cast<std::int32_t>(pending.accepted_drafts);
        }
    }

    if (needs_hidden_correction) {
        if (speculative_backend != SpeculativeBackend::Mtp || !io.mtp_decode) {
            throw std::logic_error("partial batch correction has no MTP frame");
        }
        qwen3_6::MtpDecodeState& frame = *io.mtp_decode;
        const auto batch               = static_cast<std::int32_t>(lanes.size());
        Tensor selector_tensor         = frame.current_extents.slice(0, 0, batch);
        CUDA_CHECK(cudaMemcpyAsync(selector_tensor.data, selectors.data(), lanes.size_bytes(),
                                   cudaMemcpyHostToDevice, device.stream));
        Tensor hidden       = frame.target_hidden.slice(2, 0, batch);
        Tensor selected     = frame.target_continuation_hidden.slice(1, 0, batch);
        Tensor destinations = frame.continuation_slots.slice(0, 0, batch);
        ops::speculative_select_accepted_hidden(hidden, selector_tensor, selected, device.stream);
        ops::scatter(selected, destinations, tail_hidden_store, device.stream);
        device.synchronize();
    }

    for (std::size_t row = 0; row < lanes.size(); ++row) {
        active_lane_ = lanes[row];
        if (cancelled[row]) {
            abort_request();
        } else {
            resolve_pending_impl(accepted_tokens[row], terminal[row] != 0,
                                 !needs_hidden_correction);
        }
    }
}

void ProgramImplCore::abort_lane(std::uint32_t lane) noexcept {
    if (lane >= max_concurrency) { return; }
    active_lane_ = lane;
    abort_request();
}

bool ProgramImplCore::has_retained_lane(std::uint32_t lane) const noexcept {
    return lane < max_concurrency && requests[lane].lifecycle == Lifecycle::Resident;
}

void ProgramImplCore::evict_retained_lane(std::uint32_t lane) noexcept {
    if (!has_retained_lane(lane)) { return; }
    active_lane_ = lane;
    make_invalid();
}

GenerationTimings ProgramImplCore::generation_timings_lane(std::uint32_t lane) const noexcept {
    return lane < max_concurrency ? requests[lane].timings : GenerationTimings{};
}

SpeculativeStats ProgramImplCore::speculative_stats_lane(std::uint32_t lane) const noexcept {
    return lane < max_concurrency ? requests[lane].speculative_stats : SpeculativeStats{};
}

void ProgramImplCore::make_invalid() noexcept {
    active_request().prefill.reset();
    active_sequence().kv.reset();
    active_request().lifecycle           = Lifecycle::Invalid;
    active_sequence().execution_frontier = 0;
    active_sequence().ledger_frontier    = 0;
    active_sequence().ledger.clear();
    active_sequence().prefix_identity.clear();
    active_sequence().current_linear_state_slot =
        LinearStateSlots::prefill_working_slot(active_sequence().linear_state_base);
    active_sequence().text_kv_valid            = 0;
    active_sequence().mtp_kv_valid             = 0;
    active_sequence().dflash_context_frontier  = 0;
    active_sequence().dflash_proposal_frontier = 0;
    active_sequence().dflash_proposal_anchor   = 0;
    active_sequence().dflash_proposal_epoch    = 0;
    active_sequence().pending_context_base     = 0;
    active_sequence().pending_context_count    = 0;
    active_sequence().pending_context_valid    = false;
    active_sequence().dflash_boundary_valid    = false;
    active_sequence().dflash_boundary_frontier = 0;
    active_sequence().ordinary_tail            = false;
    active_sequence().dflash_proposal_ready    = false;
    active_sequence().mtp_draft_count          = 0;
    active_sequence().tail_hidden_valid        = false;
    active_sequence().boundary                 = {};
    active_request().pending                   = {};
}

qwen3_6::PagedKVCache* ProgramImplCore::backend_kv_cache() noexcept {
    if (speculative_backend == SpeculativeBackend::Mtp) { return decoder->mtp_cache(); }
    if (speculative_backend == SpeculativeBackend::DFlash && active_sequence().dflash) {
        return &active_sequence().dflash->full;
    }
    return nullptr;
}

const qwen3_6::PagedKVCache* ProgramImplCore::backend_kv_cache() const noexcept {
    if (speculative_backend == SpeculativeBackend::Mtp) { return decoder->mtp_cache(); }
    if (speculative_backend == SpeculativeBackend::DFlash && active_sequence().dflash) {
        return &active_sequence().dflash->full;
    }
    return nullptr;
}

std::uint32_t ProgramImplCore::backend_kv_valid() const noexcept {
    if (speculative_backend == SpeculativeBackend::Mtp) { return active_sequence().mtp_kv_valid; }
    if (speculative_backend == SpeculativeBackend::DFlash) {
        return active_sequence().dflash_context_frontier;
    }
    return 0;
}

void ProgramImplCore::reserve_sequence_kv(std::uint32_t text_pages, std::uint32_t backend_pages) {
    if (active_sequence().kv) {
        throw std::logic_error("sequence already owns a KV allocation bundle");
    }
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
    active_sequence().kv.emplace(std::move(bundle));
}

void ProgramImplCore::resize_sequence_kv_entitlement(std::uint32_t text_pages,
                                                     std::uint32_t backend_pages) {
    if (!active_sequence().kv || text_pages == 0 ||
        (active_sequence().kv->backend.has_value() != (backend_pages != 0))) {
        throw std::invalid_argument("KV resize entitlement does not match the sequence bundle");
    }
    std::array<PagedKVResize, 2> changes{};
    std::size_t count = 0;
    changes[count++]  = PagedKVResize{
         .allocation       = &active_sequence().kv->text,
         .mapped_pages     = active_sequence().kv->text.mapped_page_count(),
         .page_entitlement = text_pages,
    };
    if (active_sequence().kv->backend) {
        changes[count++] = PagedKVResize{
            .allocation       = &*active_sequence().kv->backend,
            .mapped_pages     = active_sequence().kv->backend->mapped_page_count(),
            .page_entitlement = backend_pages,
        };
    }
    resize_paged_kv_bundle(std::span<PagedKVResize>(changes.data(), count));
}

void ProgramImplCore::bind_sequence_kv() {
    if (!active_sequence().kv || active_sequence().kv->text.bound_row() >= 0 ||
        (active_sequence().kv->backend && active_sequence().kv->backend->bound_row() >= 0)) {
        throw std::logic_error("KV allocation bundle is unavailable or already bound");
    }
    const std::int32_t row = static_cast<std::int32_t>(active_sequence().lane);
    active_sequence().kv->text.bind_row(row, device.stream);
    try {
        if (active_sequence().kv->backend) {
            active_sequence().kv->backend->bind_row(row, device.stream);
        }
        set_device_i32(io.text_kv_table_row, active_sequence().kv->text.bound_row());
        set_device_i32(io.backend_kv_table_row, active_sequence().kv->backend
                                                    ? active_sequence().kv->backend->bound_row()
                                                    : 0);
    } catch (...) {
        if (active_sequence().kv->backend && active_sequence().kv->backend->bound_row() >= 0) {
            active_sequence().kv->backend->unbind_row();
        }
        active_sequence().kv->text.unbind_row();
        throw;
    }
}

void ProgramImplCore::unbind_sequence_kv() noexcept {
    if (!active_sequence().kv) { return; }
    if (active_sequence().kv->backend) { active_sequence().kv->backend->unbind_row(); }
    active_sequence().kv->text.unbind_row();
}

void ProgramImplCore::materialize_sequence_kv(std::uint32_t main_tokens,
                                              std::uint32_t backend_tokens) {
    if (!active_sequence().kv || main_tokens > capacity || backend_tokens > capacity) {
        throw std::logic_error("KV materialization request is outside the sequence bundle");
    }
    if (backend_tokens != 0 && !active_sequence().kv->backend) {
        throw std::logic_error("backend KV materialization requested without an allocation");
    }
    if (main_tokens > active_sequence().kv->text.mapped_token_capacity()) {
        active_sequence().kv->text.materialize_tokens(main_tokens, device.stream);
    }
    if (backend_tokens != 0 &&
        backend_tokens > active_sequence().kv->backend->mapped_token_capacity()) {
        active_sequence().kv->backend->materialize_tokens(backend_tokens, device.stream);
    }
}

void ProgramImplCore::trim_sequence_kv(std::uint32_t main_tokens, std::uint32_t backend_tokens) {
    if (!active_sequence().kv || main_tokens > capacity || backend_tokens > main_tokens) {
        throw std::logic_error("KV trim request is outside the sequence bundle");
    }
    if (backend_tokens != 0 && !active_sequence().kv->backend) {
        throw std::logic_error("backend KV trim requested without an allocation");
    }
    active_sequence().kv->text.trim_tokens(main_tokens);
    if (active_sequence().kv->backend) {
        active_sequence().kv->backend->trim_tokens(backend_tokens);
    }
}

void ProgramImplCore::release_sequence_growth_entitlement() noexcept {
    if (!active_sequence().kv) { return; }
    active_sequence().kv->text.cancel_unmapped_entitlement();
    if (active_sequence().kv->backend) {
        active_sequence().kv->backend->cancel_unmapped_entitlement();
    }
}

qwen3_6::PagedKVCacheView ProgramImplCore::text_kv_view() const {
    if (!active_sequence().kv) { throw std::logic_error("sequence has no KV allocation bundle"); }
    return decoder->text_kv.execution_view(active_sequence().kv->text);
}

qwen3_6::PagedKVCacheView ProgramImplCore::mtp_kv_view() const {
    if (speculative_backend != SpeculativeBackend::Mtp) { return {}; }
    if (decoder->mtp_cache() == nullptr || !active_sequence().kv ||
        !active_sequence().kv->backend) {
        throw std::logic_error("sequence has no MTP KV allocation");
    }
    return decoder->mtp_cache()->execution_view(*active_sequence().kv->backend);
}

qwen3_6::PagedKVCacheView ProgramImplCore::dflash_full_kv_view() const {
    if (speculative_backend != SpeculativeBackend::DFlash) { return {}; }
    if (!active_sequence().dflash || !active_sequence().kv || !active_sequence().kv->backend) {
        throw std::logic_error("sequence has no DFlash Full KV allocation");
    }
    return active_sequence().dflash->full_view(*active_sequence().kv->backend);
}

void ProgramImplCore::set_device_i32(Tensor& tensor, std::int32_t value) {
    CUDA_CHECK(
        cudaMemcpyAsync(tensor.data, &value, sizeof(value), cudaMemcpyHostToDevice, device.stream));
}

void ProgramImplCore::ordered_reset() {
    decoder->linear_attention.zero_slot(
        LinearStateSlots::prefill_working_slot(active_sequence().linear_state_base), device.stream);
    work.reset();
    set_device_i32(io.pos, 0);
    set_device_i32(io.rope_pos, 0);
    set_device_i32(io.rope_delta, 0);
    set_device_i32(io.speculative.accepted_drafts, 0);
    set_device_i32(io.linear_state_read_slot,
                   LinearStateSlots::prefill_working_slot(active_sequence().linear_state_base));
    set_device_i32(io.linear_state_snapshot_base_slot, LinearStateSlots::verify_snapshot_base_slot(
                                                           active_sequence().linear_state_base));
    if (io.mtp) { set_device_i32(io.mtp->position, 0); }
    active_sequence().current_linear_state_slot =
        LinearStateSlots::prefill_working_slot(active_sequence().linear_state_base);
    active_sequence().text_kv_valid            = 0;
    active_sequence().mtp_kv_valid             = 0;
    active_sequence().dflash_context_frontier  = 0;
    active_sequence().dflash_proposal_frontier = 0;
    active_sequence().dflash_proposal_anchor   = 0;
    active_sequence().dflash_proposal_epoch    = 0;
    active_sequence().pending_context_base     = 0;
    active_sequence().pending_context_count    = 0;
    active_sequence().pending_context_valid    = false;
    active_sequence().dflash_boundary_valid    = false;
    active_sequence().dflash_boundary_frontier = 0;
    active_sequence().ordinary_tail            = false;
    active_sequence().dflash_proposal_ready    = false;
    if (active_sequence().dflash) { set_device_i32(active_sequence().dflash->commit_count, 0); }
}

void ProgramImplCore::prepare_graphs() {
    if (!use_cuda_graph) { return; }

    std::vector<PagedKVAllocation> text_capture_allocations;
    std::vector<PagedKVAllocation> mtp_capture_allocations;
    qwen3_6::PagedKVCacheView capture_text_kv;
    qwen3_6::PagedKVCacheView capture_mtp_kv;
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
        return cache.execution_view(allocations.front());
    };
    if (speculative_backend != SpeculativeBackend::DFlash) {
        capture_text_kv =
            reserve_capture_rows(decoder->text_kv, text_capture_allocations, "target KV cache");
        if (speculative_backend == SpeculativeBackend::Mtp) {
            capture_mtp_kv = reserve_capture_rows(*decoder->mtp_cache(), mtp_capture_allocations,
                                                  "MTP KV cache");
        }
    } else {
        reserve_sequence_kv(
            decoder->text_kv.pool().logical_page_capacity(),
            backend_kv_cache() != nullptr ? backend_kv_cache()->pool().logical_page_capacity() : 0);
        bind_sequence_kv();
        materialize_sequence_kv(capacity, backend_kv_cache() != nullptr ? capacity : 0);
        capture_text_kv = text_kv_view();
    }

    std::size_t free_before = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_before, &total_bytes));

    const auto clear_stable_controls = [&] {
        std::vector<Tensor> controls{
            io.token,
            io.pos,
            io.rope_pos,
            io.rope_delta,
            io.speculative.target_argmax,
            io.speculative.draft_tokens,
            io.speculative.current_proposal_extent,
            io.speculative.round_tokens,
            io.speculative.produced_count,
            io.speculative.target_input_ids,
            io.speculative.target_positions,
            io.speculative.accepted_drafts,
            io.linear_state_read_slot,
            io.linear_state_snapshot_base_slot,
        };
        if (io.mtp) { controls.push_back(io.mtp->position); }
        for (const Tensor& tensor : controls) {
            CUDA_CHECK(cudaMemsetAsync(tensor.data, 0, tensor.bytes(), device.stream));
        }
    };
    const auto initialize_paged_cache = [&](qwen3_6::PagedKVCache& cache) {
        for (std::size_t plane = 0; plane < cache.pool().plane_count(); ++plane) {
            const Tensor& tensor = cache.pool().plane(plane);
            CUDA_CHECK(cudaMemsetAsync(tensor.data, 0, tensor.bytes(), device.stream));
        }
    };
    const auto initialize_cyclic_cache = [&](CyclicKVCache& cache) {
        for (std::uint32_t layer = 0; layer < cache.layer_count(); ++layer) {
            const CyclicKVCacheLayerView view = cache.layer_view(layer);
            CUDA_CHECK(cudaMemsetAsync(view.k.data, 0, view.k.bytes(), device.stream));
            CUDA_CHECK(cudaMemsetAsync(view.v.data, 0, view.v.bytes(), device.stream));
        }
    };
    initialize_paged_cache(decoder->text_kv);
    if (decoder->mtp_cache() != nullptr) { initialize_paged_cache(*decoder->mtp_cache()); }
    if (active_sequence().dflash) {
        initialize_cyclic_cache(active_sequence().dflash->local);
        initialize_paged_cache(active_sequence().dflash->full);
        CUDA_CHECK(cudaMemsetAsync(active_sequence().dflash->target_features.data, 0,
                                   active_sequence().dflash->target_features.bytes(),
                                   device.stream));
        CUDA_CHECK(cudaMemsetAsync(active_sequence().dflash->feature_positions.data, 0,
                                   active_sequence().dflash->feature_positions.bytes(),
                                   device.stream));
    }
    device.synchronize();

    const auto prepare_representative = [&](std::uint32_t frontier) {
        work.reset();
        clear_stable_controls();
        const std::int32_t slots_per_sequence = LinearStateSlots::required_slot_count(draft_window);
        for (std::uint32_t row = 0; row < max_concurrency; ++row) {
            decoder->linear_attention.zero_slot(static_cast<std::int32_t>(row) * slots_per_sequence,
                                                device.stream);
        }
        set_device_i32(io.pos, checked_i32(frontier, "graph representative position"));
        set_device_i32(io.rope_pos, checked_i32(frontier, "graph representative rope position"));
        if (io.mtp) {
            set_device_i32(io.mtp->position,
                           checked_i32(frontier, "graph representative MTP position"));
        }
        if (active_sequence().dflash) {
            set_device_i32(io.speculative.produced_count, 1);
            set_device_i32(io.speculative.target_positions,
                           checked_i32(frontier == 0 ? 0 : frontier - 1,
                                       "graph representative DFlash append position"));
        }
        if (io.mtp_decode) {
            *mtp_host_ingress          = {};
            *mtp_host_egress           = {};
            const std::uint32_t extent = std::min(draft_window, capacity - frontier - 1U);
            const std::uint32_t width  = draft_window + 1U;
            for (std::uint32_t row = 0; row < max_concurrency; ++row) {
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
                const std::int32_t linear_base =
                    static_cast<std::int32_t>(row) * slots_per_sequence;
                mtp_host_ingress->linear_state_read_slots[row]          = linear_base;
                mtp_host_ingress->linear_state_snapshot_base_slots[row] = linear_base;
                mtp_host_ingress->continuation_slots[row] = static_cast<std::int32_t>(row);
                mtp_host_ingress->rope_deltas[row]        = 0;
                mtp_host_ingress->sampling[row]           = {};
            }
        }
        for (std::uint32_t row = 0; row < max_concurrency; ++row) {
            ordinary_host_ingress->tokens[row] = 0;
            ordinary_host_ingress->cache_positions[row] =
                checked_i32(frontier, "graph representative ordinary position");
            ordinary_host_ingress->rope_positions[row] =
                checked_i32(frontier, "graph representative ordinary RoPE position");
            ordinary_host_ingress->text_kv_table_rows[row] = static_cast<std::int32_t>(row);
            const std::int32_t linear_base = static_cast<std::int32_t>(row) * slots_per_sequence;
            ordinary_host_ingress->linear_state_read_slots[row]          = linear_base;
            ordinary_host_ingress->linear_state_snapshot_base_slots[row] = linear_base;
            ordinary_host_ingress->continuation_slots[row] = static_cast<std::int32_t>(row);
            ordinary_host_ingress->sampling[row]           = {};
        }
    };
    const auto state = [&](std::uint32_t frontier) {
        return schedule::State{
            device,
            model,
            work,
            capture_text_kv,
            capture_mtp_kv,
            dflash_full_kv_view(),
            active_sequence().dflash ? &*active_sequence().dflash : nullptr,
            decoder->linear_attention,
            io,
            prefill_hidden,
            prefill_chunk,
            frontier,
            static_cast<const ops::SamplingConfig*>(
                sampling_config.slice(1, static_cast<std::int32_t>(active_sequence().lane), 1)
                    .data),
            proposal_head,
            &active_sequence().tail_hidden,
            &active_sequence().boundary_hidden,
            active_sequence().linear_state_base,
            active_sequence().linear_state_capacity,
            ordinary_host_ingress,
            ordinary_host_egress,
            &tail_hidden_store,
            mtp_host_ingress,
            mtp_host_egress};
    };

    if (speculative_backend != SpeculativeBackend::Mtp) {
        const auto ordinary_profiles = ordinary_graph_profiles(capacity);
        validate_graph_profiles(ordinary_profiles, capacity - 1, "ordinary");
        const std::uint32_t ordinary_batch_limit =
            speculative_backend == SpeculativeBackend::None ? max_concurrency : 1U;
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
                const std::uint32_t representative = planned.min;
                const ops::GqaExecutionEnvelope envelope{planned.min + 1, planned.max + 1};
                const auto prepare = [&, representative] {
                    prepare_representative(representative);
                };

                auto ordinary_state = state(representative);
                if (speculative_backend == SpeculativeBackend::None) {
                    schedule::warm_capture_ordinary_decode_batch(
                        ordinary_state, static_cast<std::int32_t>(batch_size), envelope, prepare,
                        profile.definition);
                } else {
                    schedule::warm_capture_ordinary_round(ordinary_state, envelope, prepare,
                                                          profile.definition);
                }
            }
        }
    }

    if (speculative_backend == SpeculativeBackend::Mtp) {
        const auto planned_profiles = mtp_graph_profiles(capacity, draft_window);
        validate_graph_profiles(planned_profiles, capacity - 1, "MTP");
        mtp_graphs.profiles.reserve(planned_profiles.size() * max_concurrency);
        for (std::uint32_t batch_size = 1; batch_size <= max_concurrency; ++batch_size) {
            for (const GraphExecutionProfile planned : planned_profiles) {
                mtp_graphs.profiles.emplace_back();
                DecodeGraphProfile& profile    = mtp_graphs.profiles.back();
                profile.batch_size             = batch_size;
                profile.min_execution_frontier = planned.min;
                profile.max_execution_frontier = planned.max;
                profile.topology_class =
                    planned.topology_class * max_concurrency + (batch_size - 1U);
                const std::uint32_t representative = planned.min;
                const auto prepare                 = [&, representative] {
                    prepare_representative(representative);
                };

                auto mtp_state = state(representative);
                schedule::warm_capture_mtp_decode_batch(
                    mtp_state, static_cast<std::int32_t>(batch_size), draft_window,
                    mtp_gqa_envelopes(planned.min, planned.max, draft_window, capacity), prepare,
                    profile.definition);
            }
        }
    }
    if (speculative_backend == SpeculativeBackend::DFlash) {
        const auto initial_profiles = dflash_initial_graph_profiles(capacity, draft_window);
        validate_graph_profiles(initial_profiles, capacity - draft_window - 1, "DFlash initial");
        dflash_initial_graphs.profiles.reserve(initial_profiles.size());
        for (const GraphExecutionProfile planned : initial_profiles) {
            dflash_initial_graphs.profiles.emplace_back();
            DecodeGraphProfile& profile        = dflash_initial_graphs.profiles.back();
            profile.min_execution_frontier     = planned.min;
            profile.max_execution_frontier     = planned.max;
            profile.topology_class             = planned.topology_class;
            const std::uint32_t representative = planned.min;
            const auto prepare = [&, representative] { prepare_representative(representative); };
            const ops::GqaExecutionEnvelope target_envelope{planned.min + draft_window + 1,
                                                            planned.max + draft_window + 1};

            auto initial_state = state(representative);
            schedule::warm_capture_dflash_initial_round(
                initial_state, draft_window, target_envelope, prepare, profile.definition);
        }

        const auto steady_profiles = dflash_steady_graph_profiles(capacity, draft_window);
        validate_graph_profiles(steady_profiles, capacity - draft_window - 1, "DFlash steady");
        dflash_steady_graphs.profiles.reserve(steady_profiles.size());
        for (const GraphExecutionProfile planned : steady_profiles) {
            dflash_steady_graphs.profiles.emplace_back();
            DecodeGraphProfile& profile        = dflash_steady_graphs.profiles.back();
            profile.min_execution_frontier     = planned.min;
            profile.max_execution_frontier     = planned.max;
            profile.topology_class             = planned.topology_class;
            const std::uint32_t representative = planned.min;
            const auto prepare = [&, representative] { prepare_representative(representative); };
            const ops::GqaExecutionEnvelope target_envelope{planned.min + draft_window + 1,
                                                            planned.max + draft_window + 1};

            auto steady_state = state(representative);
            schedule::warm_capture_dflash_steady_round(
                steady_state, draft_window,
                dflash_envelopes(planned.min, planned.max, draft_window), target_envelope, prepare,
                profile.definition);
        }
    }

    if (!ordinary_graphs.profiles.empty()) {
        instantiate_graph_family(ordinary_graphs, "ordinary", device, prepare_representative);
    }
    if (speculative_backend == SpeculativeBackend::Mtp) {
        instantiate_graph_family(mtp_graphs, "MTP", device, prepare_representative);
    }
    if (speculative_backend == SpeculativeBackend::DFlash) {
        instantiate_graph_family(dflash_initial_graphs, "DFlash initial", device,
                                 prepare_representative);
        instantiate_graph_family(dflash_steady_graphs, "DFlash steady", device,
                                 prepare_representative);
    }

    ordered_reset();
    clear_stable_controls();
    for (Tensor& tensor : decoder->linear_attention.conv) {
        CUDA_CHECK(cudaMemsetAsync(tensor.data, 0, tensor.bytes(), device.stream));
    }
    for (Tensor& tensor : decoder->linear_attention.recurrent) {
        CUDA_CHECK(cudaMemsetAsync(tensor.data, 0, tensor.bytes(), device.stream));
    }
    CUDA_CHECK(cudaMemsetAsync(token_counts.data, 0, token_counts.bytes(), device.stream));
    device.synchronize();

    std::size_t free_after = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_after, &total_bytes));
    const std::size_t consumed = free_before > free_after ? free_before - free_after : 0;
    if (consumed > graph_allowance_bytes) {
        throw std::runtime_error("CUDA Graph warm/capture consumed " + std::to_string(consumed) +
                                 " bytes, exceeding the planned allowance of " +
                                 std::to_string(graph_allowance_bytes) + " bytes");
    }
    if (speculative_backend != SpeculativeBackend::DFlash) {
        for (PagedKVAllocation& allocation : mtp_capture_allocations) { allocation.unbind_row(); }
        mtp_capture_allocations.clear();
        for (PagedKVAllocation& allocation : text_capture_allocations) { allocation.unbind_row(); }
        text_capture_allocations.clear();
    } else {
        unbind_sequence_kv();
        active_sequence().kv.reset();
    }
}

void ProgramImplCore::install_sampling(const ops::SamplingConfig& config) {
    Tensor counts = token_counts.slice(1, static_cast<std::int32_t>(active_sequence().lane), 1)
                        .view({TextConfig::token_domain});
    CUDA_CHECK(cudaMemsetAsync(counts.data, 0, counts.bytes(), device.stream));
    active_request().sampling_host     = config;
    active_request().speculative_stats = SpeculativeStats{
        .backend               = speculative_backend,
        .enabled               = speculative_backend != SpeculativeBackend::None,
        .draft_window          = draft_window,
        .accepted_per_position = std::vector<std::uint64_t>(draft_window, 0),
    };
    const bool penalties = active_request().sampling_host.presence_penalty != 0.0F ||
                           active_request().sampling_host.frequency_penalty != 0.0F;
    active_request().sampling_host.token_counts =
        penalties ? static_cast<std::int32_t*>(counts.data) : nullptr;
    Tensor config_lane =
        sampling_config.slice(1, static_cast<std::int32_t>(active_sequence().lane), 1);
    CUDA_CHECK(cudaMemcpyAsync(config_lane.data, &active_request().sampling_host,
                               sizeof(active_request().sampling_host), cudaMemcpyHostToDevice,
                               device.stream));
}

void ProgramImplCore::copy_tail(const Tensor& source) {
    if (source.dtype != DType::BF16 || source.ne[0] != TextConfig::hidden || source.ne[1] != 1) {
        throw std::logic_error("target tail hidden has an invalid shape");
    }
    CUDA_CHECK(cudaMemcpyAsync(active_sequence().tail_hidden.data, source.data,
                               active_sequence().tail_hidden.bytes(), cudaMemcpyDeviceToDevice,
                               device.stream));
    active_sequence().tail_hidden_valid = true;
}

void ProgramImplCore::copy_round_token() {
    CUDA_CHECK(cudaMemcpyAsync(host_tokens, io.token.data, sizeof(TokenId), cudaMemcpyDeviceToHost,
                               device.stream));
}

void ProgramImplCore::mark_workspace_usage(std::size_t phase_bytes) noexcept {
    workspace_logical_peak_bytes = std::max(workspace_logical_peak_bytes, phase_bytes);
}

void ProgramImplCore::flush_dflash_context_prefix(std::uint32_t count) {
    if (!active_sequence().dflash || !active_sequence().pending_context_valid ||
        active_sequence().dflash_context_frontier != active_sequence().pending_context_base ||
        count == 0 || count > active_sequence().pending_context_count) {
        throw std::logic_error("DFlash context flush does not match the pending target features");
    }
    const std::uint32_t context_end = active_sequence().pending_context_base + count;
    materialize_sequence_kv(std::max(active_sequence().text_kv_valid, context_end), context_end);
    schedule::State state{
        device,
        model,
        work,
        text_kv_view(),
        mtp_kv_view(),
        dflash_full_kv_view(),
        &*active_sequence().dflash,
        decoder->linear_attention,
        io,
        prefill_hidden,
        prefill_chunk,
        active_sequence().pending_context_base,
        static_cast<const ops::SamplingConfig*>(
            sampling_config.slice(1, static_cast<std::int32_t>(active_sequence().lane), 1).data),
        proposal_head,
        &active_sequence().tail_hidden,
        &active_sequence().boundary_hidden,
        active_sequence().linear_state_base,
        active_sequence().linear_state_capacity};
    set_device_i32(active_sequence().dflash->commit_count,
                   checked_i32(count, "DFlash context commit count"));
    Tensor features =
        active_sequence().dflash->target_features.slice(1, 0, static_cast<std::int32_t>(count));
    Tensor positions =
        active_sequence().dflash->feature_positions.slice(0, 0, static_cast<std::int32_t>(count));
    mark_workspace_usage(workspace_plan.dflash_context);
    schedule::dflash_append_context(state, features, positions,
                                    active_sequence().dflash->commit_count, {count, count});
    device.synchronize();
    active_sequence().dflash_context_frontier = active_sequence().pending_context_base + count;
    active_sequence().pending_context_base    = 0;
    active_sequence().pending_context_count   = 0;
    active_sequence().pending_context_valid   = false;
}

void ProgramImplCore::validate_licensed_tokens(std::span<const TokenId> tokens) const {
    for (const TokenId token : tokens) {
        if (token < 0 || token >= TextConfig::token_domain) {
            throw std::runtime_error("target returned a token outside the 248077-token domain");
        }
    }
}

runtime::PrefillStepResult ProgramImplCore::advance_prefill() {
    if (speculative_backend == SpeculativeBackend::DFlash ||
        active_request().lifecycle != Lifecycle::Prefilling || !active_request().prefill) {
        throw std::logic_error("staged prefill step requires an active concurrent request");
    }

    RequestControl::Prefill& staged = *active_request().prefill;
    const runtime::BeginSummary summary{.prompt_tokens        = staged.prompt_tokens,
                                        .reused_prompt_tokens = staged.base};
    bool host_input_consumed           = staged.host_input_consumed_pending;
    staged.host_input_consumed_pending = false;
    const auto started                 = Clock::now();
    try {
        schedule::State schedule_state{
            device,
            model,
            work,
            text_kv_view(),
            mtp_kv_view(),
            {},
            nullptr,
            decoder->linear_attention,
            io,
            prefill_hidden,
            prefill_chunk,
            staged.cursor,
            static_cast<const ops::SamplingConfig*>(
                sampling_config.slice(1, static_cast<std::int32_t>(active_sequence().lane), 1)
                    .data),
            proposal_head,
            &active_sequence().tail_hidden,
            &active_sequence().boundary_hidden,
            active_sequence().linear_state_base,
            active_sequence().linear_state_capacity,
            ordinary_host_ingress,
            ordinary_host_egress,
            &tail_hidden_store,
            mtp_host_ingress,
            mtp_host_egress,
            staged.initial_mtp_extent};

        if (staged.prepare_mtp && staged.needs_mtp_bridge && !staged.mtp_bridge_complete) {
            if (staged.cursor != staged.base || staged.base == 0 ||
                staged.cursor >= staged.prompt_tokens) {
                throw std::logic_error("staged MTP bridge is outside the reusable suffix");
            }
            mark_workspace_usage(workspace_plan.mtp_prefill);
            const Tensor& previous_hidden = staged.mtp_bridge_uses_boundary
                                                ? active_sequence().boundary_hidden
                                                : active_sequence().tail_hidden;
            const schedule::MtpBridgeInput bridge{
                .previous_hidden = &previous_hidden,
                .position        = checked_i32(staged.base - 1, "MTP bridge position"),
                .rope_position   = prompt_rope_position(staged.prompt, staged.base - 1),
            };
            if (staged.vision) {
                schedule::mtp_bridge_multimodal(schedule_state, staged.prompt, *staged.vision,
                                                bridge);
            } else {
                Tensor bridge_token = io.speculative.target_input_ids.slice(0, 0, 1);
                const TokenId token = staged.prompt.token_ids[staged.base];
                CUDA_CHECK(cudaMemcpyAsync(bridge_token.data, &token, sizeof(token),
                                           cudaMemcpyHostToDevice, device.stream));
                schedule::mtp_bridge_and_propose(schedule_state, bridge_token, previous_hidden,
                                                 bridge.position, bridge.rope_position, false);
            }
            active_sequence().mtp_kv_valid = staged.base;
            staged.mtp_bridge_complete     = true;
        }

        if (staged.cursor < staged.prompt_tokens) {
            const std::uint32_t nominal =
                std::min(prefill_chunk, staged.prompt_tokens - staged.cursor);
            const bool final_candidate = staged.cursor + nominal == staged.prompt_tokens;
            mark_workspace_usage(staged.prepare_mtp ? workspace_plan.mtp_prefill
                                                    : workspace_plan.text_prefill);
            schedule::PrefillChunkResult result;
            if (staged.vision) {
                mark_workspace_usage(workspace_plan.vision_encode);
                result = schedule::prefill_multimodal_chunk(
                    schedule_state, staged.prompt, *staged.vision, nominal,
                    staged.snapshot_boundary, final_candidate);
            } else {
                result = schedule::prefill_text_chunk(
                    schedule_state, std::span<const TokenId>(staged.prompt.token_ids), nominal,
                    staged.snapshot_boundary, final_candidate);
            }
            if (result.processed_tokens == 0 || result.processed_tokens > nominal) {
                throw std::logic_error("ordinary prefill chunk made invalid progress");
            }
            if (staged.vision && staged.vision->release_consumed_media_payload()) {
                host_input_consumed = true;
            }
            staged.cursor += result.processed_tokens;
            active_sequence().text_kv_valid = staged.cursor;
            if (staged.prepare_mtp) { active_sequence().mtp_kv_valid = staged.cursor; }
            active_sequence().current_linear_state_slot =
                LinearStateSlots::prefill_working_slot(active_sequence().linear_state_base);

            if (!result.finalized) {
                if (staged.cursor == staged.prompt_tokens) {
                    throw std::logic_error("staged prefill reached the prompt without sampling");
                }
                staged.elapsed_seconds +=
                    std::chrono::duration<double>(Clock::now() - started).count();
                return runtime::PrefillStepResult{.summary             = summary,
                                                  .host_input_consumed = host_input_consumed};
            }
            if (staged.cursor != staged.prompt_tokens) {
                throw std::logic_error("staged prefill sampled before the prompt frontier");
            }
            copy_tail(
                prefill_hidden.slice(1, static_cast<std::int32_t>(result.processed_tokens) - 1, 1));
        } else {
            mark_workspace_usage(workspace_plan.ordinary_round);
            if (!active_sequence().tail_hidden_valid) {
                throw std::logic_error("zero-suffix reuse has no target tail hidden");
            }
            schedule::sample_from_hidden(schedule_state, active_sequence().tail_hidden,
                                         checked_i32(staged.prompt_tokens, "sample position"),
                                         ops::kSamplePurposePrefill);
            set_device_i32(io.rope_pos, checked_i32(staged.prompt_tokens, "rope position") +
                                            active_sequence().rope_delta);
            if (staged.prepare_mtp) {
                mark_workspace_usage(workspace_plan.mtp_prefill);
                const auto bridge_rope =
                    prompt_rope_position(staged.prompt, staged.prompt_tokens - 1);
                schedule::mtp_bridge_and_propose(
                    schedule_state, io.token, active_sequence().tail_hidden,
                    checked_i32(staged.prompt_tokens - 1, "MTP full-prefix bridge position"),
                    bridge_rope, staged.initial_mtp_extent != 0);
                active_sequence().mtp_kv_valid = staged.prompt_tokens;
            }
        }

        copy_round_token();
        std::array<TokenId, qwen3_6::kMtpDecodeMaximumDrafts> initial_drafts{};
        if (staged.prepare_mtp && staged.initial_mtp_extent != 0) {
            CUDA_CHECK(cudaMemcpyAsync(initial_drafts.data(), io.speculative.draft_tokens.data,
                                       staged.initial_mtp_extent * sizeof(TokenId),
                                       cudaMemcpyDeviceToHost, device.stream));
        }
        device.synchronize();
        staged.elapsed_seconds += std::chrono::duration<double>(Clock::now() - started).count();
        const double vision_seconds = staged.vision ? staged.vision->elapsed_seconds() : 0.0;
        const std::optional<std::uint32_t> snapshot_boundary = staged.snapshot_boundary;
        const std::uint32_t prompt_tokens                    = staged.prompt_tokens;

        validate_licensed_tokens(std::span<const TokenId>(host_tokens, 1));
        if (active_sequence().ledger.size() != prompt_tokens) {
            throw std::logic_error("candidate token ledger does not match prompt length");
        }
        active_sequence().ledger.push_back(host_tokens[0]);
        active_sequence().prefix_identity.append_generated(1, active_sequence().rope_delta);
        active_sequence().text_kv_valid = prompt_tokens;
        if (staged.prepare_mtp) {
            if (active_sequence().mtp_kv_valid != prompt_tokens) {
                throw std::logic_error("staged MTP prefill did not reach the prompt frontier");
            }
            active_sequence().mtp_draft_count = staged.initial_mtp_extent;
            std::copy_n(initial_drafts.begin(), staged.initial_mtp_extent,
                        active_sequence().mtp_drafts.begin());
        }
        active_sequence().tail_hidden_valid     = true;
        active_request().timings.vision_seconds = vision_seconds;
        active_request().timings.prefill_seconds =
            std::max(0.0, staged.elapsed_seconds - vision_seconds);
        if (snapshot_boundary) {
            active_sequence().boundary.valid            = true;
            active_sequence().boundary.boundary         = *snapshot_boundary;
            active_sequence().boundary.hidden_valid     = true;
            active_sequence().boundary.mtp_prefix_valid = staged.prepare_mtp;
        }

        if (!staged.prompt.patches.empty()) {
            staged.prompt.release_media_payload();
            host_input_consumed = true;
        }

        active_request().prefill.reset();
        active_request().pending   = PendingCandidate{.kind          = PendingKind::Begin,
                                                      .base_E        = 0,
                                                      .base_S        = 0,
                                                      .prompt_tokens = prompt_tokens,
                                                      .produced      = 1};
        active_request().lifecycle = Lifecycle::Pending;
        return runtime::PrefillStepResult{
            .summary  = summary,
            .round    = runtime::GeneratedRound{.tokens = std::span<const TokenId>(host_tokens, 1)},
            .complete = true,
            .host_input_consumed = host_input_consumed,
        };
    } catch (...) {
        try {
            device.synchronize();
        } catch (...) {}
        make_invalid();
        throw;
    }
}

runtime::BeginResult ProgramImplCore::begin(PreparedPromptData&& prompt, RequestPlan&& request_plan,
                                            runtime::TransientRegion transient) {
    if (request_plan.impl_ == nullptr) { throw std::invalid_argument("request plan is empty"); }
    if (speculative_backend == SpeculativeBackend::None) {
        throw std::logic_error("ordinary requests require the staged Engine execution path");
    }
    const RequestPlanImpl& plan = *request_plan.impl_;
    if (active_request().lifecycle == Lifecycle::Prefilling ||
        active_request().lifecycle == Lifecycle::Active ||
        active_request().lifecycle == Lifecycle::Pending) {
        throw std::logic_error("begin requires Empty, Resident, or Invalid Program state");
    }
    const std::uint32_t prompt_tokens = static_cast<std::uint32_t>(prompt.token_ids.size());
    if (prompt_tokens != plan.summary.prompt_tokens ||
        (plan.vision.has_value() && !prompt.has_media())) {
        throw std::invalid_argument("request plan does not describe the prepared prompt");
    }
    const bool suffix_has_visual =
        std::any_of(prompt.token_types.begin() + static_cast<std::ptrdiff_t>(plan.reuse_base),
                    prompt.token_types.end(), [](std::uint8_t type) { return type != 0; });
    if (suffix_has_visual != plan.vision.has_value()) {
        throw std::invalid_argument("request plan does not describe the prompt suffix modality");
    }
    if (plan.summary.transient_bytes != 0 &&
        (transient.data == nullptr || transient.size < plan.summary.transient_bytes ||
         transient.alignment < plan.summary.transient_alignment)) {
        throw std::invalid_argument("request transient region does not satisfy the plan");
    }
    if (plan.reuse != ReusePath::FullReset) {
        if (active_request().lifecycle != Lifecycle::Resident ||
            !qwen3_6::detail::prefix_matches(prompt, active_sequence().ledger,
                                             active_sequence().prefix_identity, plan.reuse_base)) {
            throw std::logic_error("planned resident prefix is no longer reusable");
        }
        if (plan.reuse == ReusePath::RestoreBoundary &&
            (!active_sequence().boundary.valid ||
             active_sequence().boundary.boundary != plan.reuse_base)) {
            throw std::logic_error("planned sequence-boundary checkpoint is unavailable");
        }
    }

    const std::uint32_t base              = plan.reuse_base;
    const bool had_suffix                 = prompt_tokens > base;
    const std::int32_t request_rope_delta = prompt.rope_delta;
    const auto snapshot_boundary          = plan.snapshot_boundary;
    const auto begin_start                = Clock::now();

    // From here on, the old identity is deliberately unreachable. Any failure takes the Program
    // to Invalid rather than attempting a mixed restore/reset fallback.
    active_request().lifecycle = Lifecycle::Invalid;
    try {
        if (plan.reuse == ReusePath::FullReset) {
            active_sequence().kv.reset();
            ordered_reset();
            active_sequence().ledger.clear();
            reserve_sequence_kv(plan.text_kv_page_entitlement, plan.backend_kv_page_entitlement);
        } else if (plan.reuse == ReusePath::AppendAtFrontier) {
            if (!active_sequence().kv) {
                throw std::logic_error("resident prefix has no KV allocation bundle");
            }
            trim_sequence_kv(base, backend_kv_valid());
            resize_sequence_kv_entitlement(plan.text_kv_page_entitlement,
                                           plan.backend_kv_page_entitlement);
            if (active_sequence().text_kv_valid < base) {
                throw std::logic_error("resident Text KV is shorter than the execution frontier");
            }
            active_sequence().text_kv_valid = base;
            active_sequence().ledger.resize(base);
            set_device_i32(io.linear_state_read_slot, active_sequence().current_linear_state_slot);
            if (active_sequence().dflash && (active_sequence().pending_context_valid ||
                                             active_sequence().dflash_context_frontier != base)) {
                throw std::logic_error("resident DFlash context is not at the append frontier");
            }
        } else {
            if (!active_sequence().kv) {
                throw std::logic_error("resident boundary has no KV allocation bundle");
            }
            if (active_sequence().text_kv_valid < base) {
                throw std::logic_error("resident Text KV is shorter than the boundary");
            }
            active_sequence().text_kv_valid = base;
            trim_sequence_kv(base, backend_kv_valid());
            resize_sequence_kv_entitlement(plan.text_kv_page_entitlement,
                                           plan.backend_kv_page_entitlement);
            decoder->linear_attention.copy_slot(
                LinearStateSlots::prefix_boundary_slot(active_sequence().linear_state_base,
                                                       active_sequence().linear_state_capacity),
                LinearStateSlots::prefill_working_slot(active_sequence().linear_state_base),
                device.stream);
            active_sequence().current_linear_state_slot =
                LinearStateSlots::prefill_working_slot(active_sequence().linear_state_base);
            set_device_i32(io.linear_state_read_slot, active_sequence().current_linear_state_slot);
            active_sequence().ledger.resize(base);
            if (active_sequence().dflash) {
                if (!active_sequence().dflash_boundary_valid ||
                    active_sequence().dflash_boundary_frontier != base) {
                    throw std::logic_error("planned DFlash boundary checkpoint is unavailable");
                }
                active_sequence().dflash->restore_boundary(device.stream);
                active_sequence().dflash_context_frontier = base;
            }
        }

        if (plan.prepare_mtp && base != 0) {
            const std::uint32_t mtp_base = base - 1;
            if (decoder->mtp_cache() == nullptr || active_sequence().mtp_kv_valid < mtp_base) {
                throw std::logic_error("reusable MTP prefix is shorter than its bridge position");
            }
            active_sequence().mtp_kv_valid = mtp_base;
        } else if (speculative_backend == SpeculativeBackend::Mtp &&
                   active_sequence().kv->backend) {
            active_sequence().mtp_kv_valid = 0;
        }

        trim_sequence_kv(base, backend_kv_valid());
        bind_sequence_kv();
        const std::uint32_t initial_mtp_extent =
            speculative_backend == SpeculativeBackend::Mtp
                ? std::min({draft_window,
                            plan.summary.effective_output_tokens > 1
                                ? plan.summary.effective_output_tokens - 2
                                : 0U,
                            capacity - prompt_tokens > 0 ? capacity - prompt_tokens - 1 : 0U})
                : 0U;
        std::uint32_t backend_materialized = 0;
        if (speculative_backend == SpeculativeBackend::Mtp && plan.prepare_mtp) {
            backend_materialized = std::min(
                capacity, prompt_tokens + (initial_mtp_extent == 0 ? 0U : initial_mtp_extent - 1U));
        } else if (speculative_backend == SpeculativeBackend::DFlash) {
            backend_materialized = prompt_tokens;
        }
        materialize_sequence_kv(prompt_tokens, backend_materialized);

        install_sampling(plan.sampling);
        active_sequence().rope_delta = request_rope_delta;
        set_device_i32(io.rope_delta, active_sequence().rope_delta);
        // Invalidate the old checkpoint identity now that execution has started. The separately
        // allocated boundary-hidden tensor is deliberately left untouched until a restore bridge
        // consumes h[B-1] below.
        active_sequence().boundary                 = {};
        active_sequence().dflash_boundary_valid    = false;
        active_sequence().dflash_boundary_frontier = 0;
        active_request().timings                   = {};
        ++active_sequence().request_epoch;
        active_sequence().dflash_proposal_ready = false;
        active_sequence().pending_context_valid = false;
        active_sequence().ordinary_tail         = false;

        active_sequence().ledger.assign(prompt.token_ids.begin(), prompt.token_ids.end());
        active_sequence().prefix_identity.assign(prompt);

        schedule::State schedule_state{
            device,
            model,
            work,
            text_kv_view(),
            mtp_kv_view(),
            dflash_full_kv_view(),
            active_sequence().dflash ? &*active_sequence().dflash : nullptr,
            decoder->linear_attention,
            io,
            prefill_hidden,
            prefill_chunk,
            base,
            static_cast<const ops::SamplingConfig*>(
                sampling_config.slice(1, static_cast<std::int32_t>(active_sequence().lane), 1)
                    .data),
            proposal_head,
            &active_sequence().tail_hidden,
            &active_sequence().boundary_hidden,
            active_sequence().linear_state_base,
            active_sequence().linear_state_capacity,
            ordinary_host_ingress,
            ordinary_host_egress,
            &tail_hidden_store};
        schedule_state.mtp_proposal_extent = initial_mtp_extent;
        bool mtp_prepared                  = false;
        bool dflash_prepared               = false;

        if (had_suffix && plan.needs_mtp_bridge && !plan.vision) {
            mark_workspace_usage(workspace_plan.mtp_prefill);
            Tensor bridge_token = io.speculative.target_input_ids.slice(0, 0, 1);
            const TokenId token = prompt.token_ids[base];
            CUDA_CHECK(cudaMemcpyAsync(bridge_token.data, &token, sizeof(token),
                                       cudaMemcpyHostToDevice, device.stream));
            const Tensor& bridge_hidden = plan.reuse == ReusePath::RestoreBoundary
                                              ? active_sequence().boundary_hidden
                                              : active_sequence().tail_hidden;
            const auto bridge_rope      = prompt_rope_position(prompt, base - 1);
            schedule::mtp_bridge_and_propose(schedule_state, bridge_token, bridge_hidden,
                                             checked_i32(base - 1, "bridge position"), bridge_rope,
                                             false);
            active_sequence().mtp_kv_valid = base;
        }

        if (plan.vision) {
            mark_workspace_usage(workspace_plan.vision_encode);
            mark_workspace_usage(plan.prepare_mtp ? workspace_plan.mtp_prefill
                                                  : workspace_plan.text_prefill);
            std::optional<schedule::MtpBridgeInput> mtp_bridge;
            if (plan.needs_mtp_bridge) {
                const Tensor& bridge_hidden = plan.reuse == ReusePath::RestoreBoundary
                                                  ? active_sequence().boundary_hidden
                                                  : active_sequence().tail_hidden;
                mtp_bridge                  = schedule::MtpBridgeInput{
                                     .previous_hidden = &bridge_hidden,
                                     .position        = checked_i32(base - 1, "bridge position"),
                                     .rope_position   = prompt_rope_position(prompt, base - 1),
                };
            }
            const auto multimodal_start                    = Clock::now();
            const schedule::MultimodalPrefillResult result = schedule::prefill_multimodal(
                schedule_state, prompt, *plan.vision, transient, snapshot_boundary,
                plan.prepare_mtp, mtp_bridge ? &*mtp_bridge : nullptr);
            if (mtp_bridge) { active_sequence().mtp_kv_valid = base; }
            mtp_prepared = result.mtp_prepared;
            copy_tail(prefill_hidden.slice(1, static_cast<int>(result.final_chunk_tokens) - 1, 1));
            copy_round_token();
            device.synchronize();
            const double combined_seconds =
                std::chrono::duration<double>(Clock::now() - multimodal_start).count();
            active_request().timings.vision_seconds = result.vision_seconds;
            active_request().timings.prefill_seconds =
                std::max(0.0, combined_seconds - result.vision_seconds);
        } else {
            const auto text_start = Clock::now();
            if (had_suffix) {
                mark_workspace_usage(plan.prepare_mtp ? workspace_plan.mtp_prefill
                                                      : workspace_plan.text_prefill);
                if (active_sequence().dflash) {
                    mark_workspace_usage(workspace_plan.dflash_context);
                }
                mtp_prepared = schedule::prefill_text(
                    schedule_state, std::span<const TokenId>(prompt.token_ids).subspan(base),
                    snapshot_boundary, plan.prepare_mtp);
                const std::uint32_t final_length = final_prefill_chunk_length(
                    base, prompt_tokens, prefill_chunk, snapshot_boundary);
                copy_tail(prefill_hidden.slice(1, static_cast<int>(final_length) - 1, 1));
            } else {
                mark_workspace_usage(workspace_plan.ordinary_round);
                if (!active_sequence().tail_hidden_valid) {
                    throw std::logic_error("zero-suffix reuse has no target tail hidden");
                }
                schedule::sample_from_hidden(schedule_state, active_sequence().tail_hidden,
                                             checked_i32(prompt_tokens, "sample position"),
                                             ops::kSamplePurposePrefill);
                set_device_i32(io.rope_pos, checked_i32(prompt_tokens, "rope position") +
                                                active_sequence().rope_delta);
                if (plan.prepare_mtp) {
                    const auto bridge_rope = prompt_rope_position(prompt, prompt_tokens - 1);
                    schedule::mtp_bridge_and_propose(
                        schedule_state, io.token, active_sequence().tail_hidden,
                        checked_i32(prompt_tokens - 1, "bridge position"), bridge_rope, true);
                    mtp_prepared = true;
                }
            }
            if (active_sequence().dflash) {
                active_sequence().dflash_context_frontier = prompt_tokens;
                if (plan.prepare_dflash) {
                    mark_workspace_usage(workspace_plan.dflash_proposal);
                    schedule::dflash_propose(
                        schedule_state, draft_window,
                        dflash_envelopes(prompt_tokens, prompt_tokens, draft_window));
                    dflash_prepared = true;
                }
            }
            copy_round_token();
            device.synchronize();
            active_request().timings.prefill_seconds =
                std::chrono::duration<double>(Clock::now() - text_start).count();
        }

        if (prompt.has_media()) { prompt.release_media_payload(); }

        validate_licensed_tokens(std::span<const TokenId>(host_tokens, 1));
        if (active_sequence().ledger.size() != prompt_tokens) {
            throw std::logic_error("candidate token ledger does not match prompt length");
        }
        active_sequence().ledger.push_back(host_tokens[0]);
        active_sequence().prefix_identity.append_generated(1, active_sequence().rope_delta);
        active_sequence().text_kv_valid = prompt_tokens;
        // Target prefill leaves its recurrent state in slot 0. Exact-frontier reuse performs no
        // target work, so it must retain the MTP snapshot that was committed at the old frontier.
        if (had_suffix) {
            active_sequence().current_linear_state_slot =
                LinearStateSlots::prefill_working_slot(active_sequence().linear_state_base);
        }
        active_sequence().mtp_kv_valid          = mtp_prepared ? prompt_tokens : 0;
        active_sequence().dflash_proposal_ready = dflash_prepared;
        if (active_sequence().dflash) {
            if (active_sequence().dflash_context_frontier != prompt_tokens) {
                throw std::logic_error("DFlash prefill did not reach the prompt frontier");
            }
            active_sequence().dflash_proposal_frontier = dflash_prepared ? prompt_tokens : 0;
            active_sequence().dflash_proposal_anchor   = dflash_prepared ? host_tokens[0] : 0;
            active_sequence().dflash_proposal_epoch =
                dflash_prepared ? active_sequence().request_epoch : 0;
            active_sequence().ordinary_tail = !dflash_prepared;
        }
        active_sequence().tail_hidden_valid = true;
        if (snapshot_boundary) {
            active_sequence().boundary.valid            = true;
            active_sequence().boundary.boundary         = *snapshot_boundary;
            active_sequence().boundary.hidden_valid     = true;
            active_sequence().boundary.mtp_prefix_valid = mtp_prepared;
            if (active_sequence().dflash) {
                active_sequence().dflash_boundary_valid    = true;
                active_sequence().dflash_boundary_frontier = *snapshot_boundary;
            }
        }

        active_request().pending   = PendingCandidate{.kind          = PendingKind::Begin,
                                                      .base_E        = 0,
                                                      .base_S        = 0,
                                                      .prompt_tokens = prompt_tokens,
                                                      .produced      = 1};
        active_request().lifecycle = Lifecycle::Pending;
        active_request().timings.prefill_seconds =
            std::max(active_request().timings.prefill_seconds,
                     std::chrono::duration<double>(Clock::now() - begin_start).count() -
                         active_request().timings.vision_seconds);
        return runtime::BeginResult{
            .summary =
                runtime::BeginSummary{.prompt_tokens = prompt_tokens, .reused_prompt_tokens = base},
            .round = runtime::GeneratedRound{.tokens = std::span<const TokenId>(host_tokens, 1)},
        };
    } catch (...) {
        try {
            device.synchronize();
        } catch (...) {}
        make_invalid();
        throw;
    }
}

runtime::BatchedGeneratedRound
ProgramImplCore::decode_ordinary_batch(std::span<const std::uint32_t> lanes,
                                       std::span<const runtime::RoundBudget> budgets) {
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

    const auto start = Clock::now();
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
            active_lane_                       = lanes[row];
            SequenceState& sequence            = active_sequence();
            const std::uint32_t frontier       = sequence.execution_frontier;
            ordinary_host_ingress->tokens[row] = sequence.ledger.back();
            ordinary_host_ingress->cache_positions[row] =
                checked_i32(frontier, "ordinary batch position");
            ordinary_host_ingress->rope_positions[row] =
                checked_i32(frontier, "ordinary batch RoPE position") + sequence.rope_delta;
            ordinary_host_ingress->text_kv_table_rows[row] = sequence.kv->text.bound_row();
            ordinary_host_ingress->linear_state_read_slots[row] =
                sequence.current_linear_state_slot;
            ordinary_host_ingress->linear_state_snapshot_base_slots[row] =
                LinearStateSlots::verify_snapshot_base_slot(sequence.linear_state_base);
            ordinary_host_ingress->continuation_slots[row] =
                static_cast<std::int32_t>(sequence.lane);
            ordinary_host_ingress->sampling[row] = active_request().sampling_host;
            materialize_sequence_kv(frontier + 1, 0);
        }

        active_lane_                  = lanes.front();
        SequenceState& first_sequence = active_sequence();
        schedule::State schedule_state{device,
                                       model,
                                       work,
                                       decoder->text_kv.execution_view(first_sequence.kv->text),
                                       {},
                                       {},
                                       nullptr,
                                       decoder->linear_attention,
                                       io,
                                       prefill_hidden,
                                       prefill_chunk,
                                       0,
                                       nullptr,
                                       proposal_head,
                                       &first_sequence.tail_hidden,
                                       &first_sequence.boundary_hidden,
                                       first_sequence.linear_state_base,
                                       first_sequence.linear_state_capacity,
                                       ordinary_host_ingress,
                                       ordinary_host_egress,
                                       &tail_hidden_store};

        mark_workspace_usage(workspace_plan.ordinary_round);
        schedule::ordinary_decode_batch(schedule_state, static_cast<std::int32_t>(lanes.size()),
                                        envelope, executable);
        device.synchronize();

        const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            active_lane_               = lanes[row];
            SequenceState& sequence    = active_sequence();
            RequestControl& request    = active_request();
            const std::uint32_t base_E = sequence.execution_frontier;
            const std::uint32_t base_S = sequence.ledger_frontier;
            const TokenId token        = ordinary_host_egress->sampled_tokens[row];
            validate_licensed_tokens(std::span<const TokenId>(&token, 1));
            sequence.text_kv_valid             = base_E + 1;
            sequence.current_linear_state_slot = LinearStateSlots::committed_snapshot_slot(
                1, sequence.linear_state_base, sequence.linear_state_capacity);
            sequence.dflash_proposal_ready = false;
            sequence.tail_hidden_valid     = true;
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
            .tokens = std::span<const TokenId>(ordinary_host_egress->sampled_tokens.data(),
                                               lanes.size())};
    } catch (...) {
        try {
            device.synchronize();
        } catch (...) {}
        for (const std::uint32_t lane : lanes) {
            if (lane < max_concurrency) {
                active_lane_ = lane;
                make_invalid();
            }
        }
        throw;
    }
}

runtime::BatchedGeneratedRound
ProgramImplCore::decode_mtp_batch(std::span<const std::uint32_t> lanes,
                                  std::span<const runtime::RoundBudget> budgets) {
    if (speculative_backend != SpeculativeBackend::Mtp || !io.mtp_decode ||
        decoder->mtp_cache() == nullptr) {
        throw std::logic_error("MTP batch execution requires the MTP backend");
    }
    if (lanes.empty() || lanes.size() > max_concurrency || budgets.size() != lanes.size()) {
        throw std::invalid_argument("MTP batch membership is invalid");
    }

    const std::uint32_t width            = draft_window + 1;
    std::uint32_t maximum_frontier       = 0;
    std::uint32_t maximum_target_tokens  = 1;
    std::uint32_t maximum_backend_tokens = 1;
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
        const std::uint32_t max_by_budget  = budgets[row].generated_tokens_remaining > 1
                                                 ? budgets[row].generated_tokens_remaining - 1
                                                 : 0;
        const std::uint32_t max_by_context = capacity - sequence.execution_frontier - 1;
        const std::uint32_t extent =
            std::min({sequence.mtp_draft_count, draft_window, max_by_budget, max_by_context});
        maximum_frontier = std::max(maximum_frontier, sequence.execution_frontier);
        maximum_target_tokens =
            std::max(maximum_target_tokens, sequence.execution_frontier + extent + 1);
        maximum_backend_tokens =
            std::max(maximum_backend_tokens,
                     std::min(capacity, sequence.execution_frontier + extent + draft_window));
    }

    const auto started = Clock::now();
    try {
        DecodeGraphExecutable* executable = nullptr;
        schedule::MtpGqaEnvelopes envelopes;
        envelopes.target_verify = {maximum_target_tokens, maximum_target_tokens};
        envelopes.batch         = envelopes.target_verify;
        for (std::uint32_t step = 0; step + 1 < draft_window; ++step) {
            envelopes.ar[step] = {maximum_backend_tokens, maximum_backend_tokens};
        }
        if (use_cuda_graph) {
            DecodeGraphProfile& profile =
                select_graph_profile(mtp_graphs, static_cast<std::uint32_t>(lanes.size()),
                                     maximum_frontier, "MTP batch");
            executable = &install_graph_profile(mtp_graphs, profile, "MTP batch");
            envelopes  = mtp_gqa_envelopes(profile.min_execution_frontier,
                                           profile.max_execution_frontier, draft_window, capacity);
        }

        for (std::size_t row = 0; row < lanes.size(); ++row) {
            active_lane_                      = lanes[row];
            SequenceState& sequence           = active_sequence();
            const std::uint32_t frontier      = sequence.execution_frontier;
            const std::uint32_t max_by_budget = budgets[row].generated_tokens_remaining > 1
                                                    ? budgets[row].generated_tokens_remaining - 1
                                                    : 0;
            const std::uint32_t extent =
                std::min({sequence.mtp_draft_count, draft_window, max_by_budget,
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
            mtp_host_ingress->text_kv_table_rows[row]      = sequence.kv->text.bound_row();
            mtp_host_ingress->mtp_kv_table_rows[row]       = sequence.kv->backend->bound_row();
            mtp_host_ingress->linear_state_read_slots[row] = sequence.current_linear_state_slot;
            mtp_host_ingress->linear_state_snapshot_base_slots[row] =
                LinearStateSlots::verify_snapshot_base_slot(sequence.linear_state_base);
            mtp_host_ingress->continuation_slots[row] = static_cast<std::int32_t>(sequence.lane);
            mtp_host_ingress->rope_deltas[row]        = sequence.rope_delta;
            mtp_host_ingress->sampling[row]           = active_request().sampling_host;
            materialize_sequence_kv(frontier + extent + 1,
                                    std::min(capacity, frontier + extent + draft_window));
        }

        active_lane_                  = lanes.front();
        SequenceState& first_sequence = active_sequence();
        schedule::State schedule_state{
            device,
            model,
            work,
            decoder->text_kv.execution_view(first_sequence.kv->text),
            decoder->mtp_cache()->execution_view(*first_sequence.kv->backend),
            {},
            nullptr,
            decoder->linear_attention,
            io,
            prefill_hidden,
            prefill_chunk,
            0,
            nullptr,
            proposal_head,
            &first_sequence.tail_hidden,
            &first_sequence.boundary_hidden,
            first_sequence.linear_state_base,
            first_sequence.linear_state_capacity,
            ordinary_host_ingress,
            ordinary_host_egress,
            &tail_hidden_store,
            mtp_host_ingress,
            mtp_host_egress};

        mark_workspace_usage(workspace_plan.mtp_round);
        schedule::mtp_decode_batch(schedule_state, static_cast<std::int32_t>(lanes.size()),
                                   draft_window, envelopes, executable);
        device.synchronize();

        const double seconds = std::chrono::duration<double>(Clock::now() - started).count();
        for (std::size_t row = 0; row < lanes.size(); ++row) {
            active_lane_                  = lanes[row];
            SequenceState& sequence       = active_sequence();
            RequestControl& request       = active_request();
            const std::uint32_t base_E    = sequence.execution_frontier;
            const std::uint32_t base_S    = sequence.ledger_frontier;
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
            }
            sequence.text_kv_valid             = base_E + static_cast<std::uint32_t>(count_i);
            sequence.mtp_kv_valid              = sequence.text_kv_valid;
            sequence.current_linear_state_slot = LinearStateSlots::committed_snapshot_slot(
                static_cast<std::uint32_t>(accepted_i + 1), sequence.linear_state_base,
                sequence.linear_state_capacity);
            sequence.mtp_draft_count = static_cast<std::uint32_t>(next_i);
            for (std::uint32_t step = 0; step < sequence.mtp_draft_count; ++step) {
                sequence.mtp_drafts[step] =
                    mtp_host_egress->next_drafts[step * max_concurrency + row];
            }
            sequence.tail_hidden_valid = true;
            sequence.ledger.insert(sequence.ledger.end(), row_tokens.begin(), row_tokens.end());
            sequence.prefix_identity.append_generated(static_cast<std::uint32_t>(count_i),
                                                      sequence.rope_delta);
            request.pending = PendingCandidate{
                .kind                 = PendingKind::Speculative,
                .base_E               = base_E,
                .base_S               = base_S,
                .prompt_tokens        = 0,
                .produced             = static_cast<std::uint32_t>(count_i),
                .proposal_extent      = pcur,
                .accepted_drafts      = static_cast<std::uint32_t>(accepted_i),
                .next_proposal_extent = static_cast<std::uint32_t>(next_i),
                .frame_row            = static_cast<std::uint32_t>(row),
                .batch_frame          = true,
            };
            request.lifecycle = Lifecycle::Pending;
            request.timings.decode_seconds += seconds;
        }
        return runtime::BatchedGeneratedRound{
            .tokens     = std::span<const TokenId>(mtp_host_egress->licensed_tokens.data(),
                                                   lanes.size() * width),
            .row_counts = std::span<const std::int32_t>(mtp_host_egress->licensed_counts.data(),
                                                        lanes.size()),
            .row_stride = width};
    } catch (...) {
        try {
            device.synchronize();
        } catch (...) {}
        for (const std::uint32_t lane : lanes) {
            if (lane < max_concurrency) {
                active_lane_ = lane;
                make_invalid();
            }
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
    throw std::logic_error("DFlash does not support concurrent batch execution");
}

runtime::GeneratedRound ProgramImplCore::decode_round(runtime::RoundBudget budget) {
    if (speculative_backend != SpeculativeBackend::DFlash) {
        throw std::logic_error("None and MTP requests require batched Engine execution");
    }
    if (active_request().lifecycle != Lifecycle::Active) {
        throw std::logic_error("decode_round requires Active Program state");
    }
    if (budget.generated_tokens_remaining == 0) {
        throw std::invalid_argument("decode round budget must be nonzero");
    }
    if (active_sequence().execution_frontier >= capacity) {
        throw std::out_of_range("Text execution context is full");
    }
    if (active_sequence().ledger_frontier != active_sequence().execution_frontier + 1 ||
        active_sequence().ledger.size() != active_sequence().ledger_frontier ||
        active_sequence().prefix_identity.size() != active_sequence().ledger_frontier) {
        throw std::logic_error("Active frontier is inconsistent");
    }

    if (active_sequence().dflash_proposal_ready &&
        (active_sequence().dflash_context_frontier != active_sequence().execution_frontier ||
         active_sequence().dflash_proposal_frontier != active_sequence().execution_frontier ||
         active_sequence().dflash_proposal_anchor != active_sequence().ledger.back() ||
         active_sequence().dflash_proposal_epoch != active_sequence().request_epoch)) {
        throw std::logic_error("DFlash proposal does not match the Active execution frontier");
    }
    if (active_sequence().pending_context_valid &&
        (active_sequence().dflash_context_frontier != active_sequence().pending_context_base ||
         active_sequence().pending_context_base + active_sequence().pending_context_count !=
             active_sequence().execution_frontier)) {
        throw std::logic_error("pending DFlash context update has an invalid frontier");
    }
    const bool use_dflash =
        !active_sequence().ordinary_tail &&
        (active_sequence().dflash_proposal_ready || active_sequence().pending_context_valid) &&
        budget.generated_tokens_remaining >= draft_window + 1 &&
        static_cast<std::uint64_t>(active_sequence().execution_frontier) + draft_window + 1ULL <=
            capacity;
    const std::uint32_t base_E = active_sequence().execution_frontier;
    const std::uint32_t base_S = active_sequence().ledger_frontier;
    const nvtx::Name round_name =
        use_dflash ? nvtx::Name::DecodeDFlashRound : nvtx::Name::DecodeOrdinaryRound;
    const nvtx::Category round_category =
        use_dflash ? nvtx::Category::DFlash : nvtx::Category::Decode;
    nvtx::ScopedRange round_range(round_name, round_category, base_E);
    try {
        set_device_i32(io.linear_state_read_slot, active_sequence().current_linear_state_slot);
        set_device_i32(
            io.linear_state_snapshot_base_slot,
            LinearStateSlots::verify_snapshot_base_slot(active_sequence().linear_state_base));
        schedule::State schedule_state{
            device,
            model,
            work,
            text_kv_view(),
            mtp_kv_view(),
            dflash_full_kv_view(),
            active_sequence().dflash ? &*active_sequence().dflash : nullptr,
            decoder->linear_attention,
            io,
            prefill_hidden,
            prefill_chunk,
            base_E,
            static_cast<const ops::SamplingConfig*>(
                sampling_config.slice(1, static_cast<std::int32_t>(active_sequence().lane), 1)
                    .data),
            proposal_head,
            &active_sequence().tail_hidden,
            &active_sequence().boundary_hidden,
            active_sequence().linear_state_base,
            active_sequence().linear_state_capacity,
            ordinary_host_ingress,
            ordinary_host_egress,
            &tail_hidden_store};

        std::uint32_t produced = 1;
        std::uint32_t accepted = 0;
        PendingKind kind       = PendingKind::Ordinary;
        if (use_dflash) {
            mark_workspace_usage(workspace_plan.dflash_round);
            const bool steady                 = active_sequence().pending_context_valid;
            DecodeGraphExecutable* executable = nullptr;
            auto envelopes                    = dflash_envelopes(base_E, base_E, draft_window);
            ops::GqaExecutionEnvelope target_envelope{base_E + draft_window + 1,
                                                      base_E + draft_window + 1};
            if (use_cuda_graph) {
                DecodeGraphFamily& family   = steady ? dflash_steady_graphs : dflash_initial_graphs;
                const char* label           = steady ? "DFlash steady" : "DFlash initial";
                DecodeGraphProfile& profile = select_graph_profile(family, base_E, label);
                executable                  = &install_graph_profile(family, profile, label);
                envelopes                   = dflash_envelopes(profile.min_execution_frontier,
                                                               profile.max_execution_frontier, draft_window);
                target_envelope             = {profile.min_execution_frontier + draft_window + 1,
                                               profile.max_execution_frontier + draft_window + 1};
            }
            materialize_sequence_kv(target_envelope.max_visible_keys, envelopes.full.max_context);
            {
                nvtx::ScopedRange submit_range(nvtx::Name::DecodeDFlashSubmit,
                                               nvtx::Category::DFlash, base_E);
                if (steady) {
                    schedule::dflash_steady_round(schedule_state, draft_window, envelopes,
                                                  target_envelope, executable);
                } else {
                    schedule::dflash_initial_round(schedule_state, draft_window, target_envelope,
                                                   executable);
                }
                CUDA_CHECK(cudaMemcpyAsync(host_count, io.speculative.produced_count.data,
                                           sizeof(std::int32_t), cudaMemcpyDeviceToHost,
                                           device.stream));
                CUDA_CHECK(cudaMemcpyAsync(host_tokens, io.speculative.round_tokens.data,
                                           (draft_window + 1ULL) * sizeof(TokenId),
                                           cudaMemcpyDeviceToHost, device.stream));
            }
            {
                nvtx::ScopedRange wait_range(nvtx::Name::DecodeDFlashWait, nvtx::Category::Control,
                                             base_E);
                device.synchronize();
            }
            if (*host_count <= 0 || *host_count > static_cast<std::int32_t>(draft_window + 1)) {
                throw std::runtime_error("DFlash returned an invalid licensed-token count");
            }
            produced = static_cast<std::uint32_t>(*host_count);
            if (produced > budget.generated_tokens_remaining ||
                static_cast<std::uint64_t>(base_E) + produced > capacity) {
                throw std::runtime_error("DFlash round exceeded its budget or context capacity");
            }
            accepted = produced - 1;
            active_request().speculative_stats.rounds += 1;
            active_request().speculative_stats.drafted_tokens += draft_window;
            active_request().speculative_stats.accepted_tokens += accepted;
            for (std::uint32_t i = 0; i < accepted; ++i) {
                active_request().speculative_stats.accepted_per_position[i] += 1;
            }
            kind                                        = PendingKind::Speculative;
            active_sequence().text_kv_valid             = base_E + produced;
            active_sequence().current_linear_state_slot = LinearStateSlots::committed_snapshot_slot(
                produced, active_sequence().linear_state_base,
                active_sequence().linear_state_capacity);
            active_sequence().dflash_context_frontier  = base_E;
            active_sequence().pending_context_base     = base_E;
            active_sequence().pending_context_count    = produced;
            active_sequence().pending_context_valid    = true;
            active_sequence().dflash_proposal_frontier = 0;
            active_sequence().dflash_proposal_anchor   = 0;
            active_sequence().dflash_proposal_epoch    = 0;
            active_sequence().dflash_proposal_ready    = false;
            active_sequence().tail_hidden_valid        = true;
        } else {
            active_request().speculative_stats.fallback_steps += 1;
            if (active_sequence().dflash && active_sequence().pending_context_valid) {
                materialize_sequence_kv(base_E, base_E);
                mark_workspace_usage(workspace_plan.dflash_context);
                Tensor features = active_sequence().dflash->target_features.slice(
                    1, 0, static_cast<std::int32_t>(draft_window + 1));
                Tensor positions = active_sequence().dflash->feature_positions.slice(
                    0, 0, static_cast<std::int32_t>(draft_window + 1));
                schedule::dflash_append_context(schedule_state, features, positions,
                                                io.speculative.produced_count,
                                                {1, draft_window + 1});
                active_sequence().dflash_context_frontier = base_E;
                active_sequence().pending_context_base    = 0;
                active_sequence().pending_context_count   = 0;
                active_sequence().pending_context_valid   = false;
            }
            mark_workspace_usage(workspace_plan.ordinary_round);
            DecodeGraphExecutable* executable = nullptr;
            ops::GqaExecutionEnvelope envelope{base_E + 1, base_E + 1};
            if (use_cuda_graph) {
                DecodeGraphProfile& profile =
                    select_graph_profile(ordinary_graphs, 1, base_E, "ordinary");
                executable = &install_graph_profile(ordinary_graphs, profile, "ordinary");
                envelope = {profile.min_execution_frontier + 1, profile.max_execution_frontier + 1};
            }
            materialize_sequence_kv(envelope.max_visible_keys, envelope.max_visible_keys);
            {
                nvtx::ScopedRange submit_range(nvtx::Name::DecodeOrdinarySubmit,
                                               nvtx::Category::Decode, base_E);
                schedule::ordinary_round(schedule_state, envelope, executable);
                copy_tail(io.verify_hidden.slice(1, 0, 1));
                copy_round_token();
            }
            {
                nvtx::ScopedRange wait_range(nvtx::Name::DecodeOrdinaryWait,
                                             nvtx::Category::Control, base_E);
                device.synchronize();
            }
            active_sequence().text_kv_valid             = base_E + 1;
            active_sequence().current_linear_state_slot = LinearStateSlots::committed_snapshot_slot(
                1, active_sequence().linear_state_base, active_sequence().linear_state_capacity);
            if (active_sequence().dflash) {
                active_sequence().dflash_context_frontier  = base_E + 1;
                active_sequence().dflash_proposal_frontier = 0;
                active_sequence().dflash_proposal_anchor   = 0;
                active_sequence().dflash_proposal_epoch    = 0;
                active_sequence().ordinary_tail            = true;
            }
            active_sequence().dflash_proposal_ready = false;
            active_sequence().tail_hidden_valid     = true;
        }

        validate_licensed_tokens(std::span<const TokenId>(host_tokens, produced));
        active_sequence().ledger.insert(active_sequence().ledger.end(), host_tokens,
                                        host_tokens + produced);
        active_sequence().prefix_identity.append_generated(produced, active_sequence().rope_delta);
        active_request().pending   = PendingCandidate{.kind          = kind,
                                                      .base_E        = base_E,
                                                      .base_S        = base_S,
                                                      .prompt_tokens = 0,
                                                      .produced      = produced};
        active_request().lifecycle = Lifecycle::Pending;
        return runtime::GeneratedRound{.tokens = std::span<const TokenId>(host_tokens, produced)};
    } catch (...) {
        try {
            device.synchronize();
        } catch (...) {}
        make_invalid();
        throw;
    }
}

void ProgramImplCore::resolve_pending(std::uint32_t accepted_tokens, bool terminal) {
    resolve_pending_impl(accepted_tokens, terminal, true);
}

void ProgramImplCore::resolve_pending_impl(std::uint32_t accepted_tokens, bool terminal,
                                           bool correct_partial_hidden) {
    if (active_request().lifecycle != Lifecycle::Pending) {
        throw std::logic_error("resolve_pending requires a pending generated round");
    }
    if (accepted_tokens == 0 || accepted_tokens > active_request().pending.produced) {
        throw std::out_of_range("accepted prefix is outside the pending generated round");
    }
    if (!terminal && accepted_tokens != active_request().pending.produced) {
        throw std::logic_error("a continuing round must accept every licensed token");
    }
    if (terminal && speculative_backend == SpeculativeBackend::DFlash &&
        active_request().pending.kind == PendingKind::Speculative) {
        try {
            flush_dflash_context_prefix(accepted_tokens);
        } catch (...) {
            make_invalid();
            throw;
        }
        active_sequence().dflash_proposal_ready    = false;
        active_sequence().dflash_proposal_frontier = 0;
        active_sequence().dflash_proposal_anchor   = 0;
        active_sequence().dflash_proposal_epoch    = 0;
    }
    if (terminal && active_request().pending.kind == PendingKind::Speculative &&
        accepted_tokens < active_request().pending.produced) {
        // The output policy may stop inside a target-licensed speculative batch. Target
        // verification has already materialized KV, hidden, and one GDN snapshot for every returned
        // prefix, so commit the exact externally accepted frontier instead of discarding the
        // resident sequence. The next request lets the active drafter rebuild proposals from this
        // target state.
        const std::uint32_t committed_E = active_request().pending.base_E + accepted_tokens;
        const std::uint32_t committed_S = active_request().pending.base_S + accepted_tokens;
        if (committed_S > active_sequence().ledger.size() ||
            committed_S > active_sequence().prefix_identity.size()) {
            throw std::logic_error(
                "partial speculative terminal exceeds the provisional sequence ledger");
        }
        if (correct_partial_hidden) {
            if (active_request().pending.batch_frame && io.mtp_decode &&
                active_request().pending.frame_row < max_concurrency) {
                Tensor row_hidden = io.mtp_decode->target_hidden.slice(
                    2, static_cast<std::int32_t>(active_request().pending.frame_row), 1);
                copy_tail(row_hidden.slice(1, static_cast<int>(accepted_tokens) - 1, 1));
            } else {
                copy_tail(io.verify_hidden.slice(1, static_cast<int>(accepted_tokens) - 1, 1));
            }
        }
        active_sequence().ledger.resize(committed_S);
        active_sequence().prefix_identity.truncate(committed_S);
        active_sequence().execution_frontier        = committed_E;
        active_sequence().ledger_frontier           = committed_S;
        active_sequence().current_linear_state_slot = LinearStateSlots::committed_snapshot_slot(
            accepted_tokens, active_sequence().linear_state_base,
            active_sequence().linear_state_capacity);
        active_sequence().text_kv_valid = committed_E;
        if (speculative_backend == SpeculativeBackend::Mtp) {
            active_sequence().mtp_kv_valid = committed_E;
        }
        set_device_i32(io.pos, checked_i32(committed_E, "partial speculative frontier"));
        set_device_i32(io.rope_pos, checked_i32(committed_E, "partial speculative RoPE frontier") +
                                        active_sequence().rope_delta);
        device.synchronize();
        trim_sequence_kv(active_sequence().text_kv_valid, backend_kv_valid());
        release_sequence_growth_entitlement();
        unbind_sequence_kv();
        active_sequence().dflash_proposal_ready = false;
        active_sequence().mtp_draft_count       = 0;
        active_request().lifecycle              = Lifecycle::Resident;
        active_request().pending                = {};
        return;
    }
    if (accepted_tokens != active_request().pending.produced) {
        throw std::logic_error("a non-speculative terminal round must accept its only token");
    }

    switch (active_request().pending.kind) {
    case PendingKind::Begin:
        active_sequence().execution_frontier = active_request().pending.prompt_tokens;
        active_sequence().ledger_frontier    = active_request().pending.prompt_tokens + 1;
        break;
    case PendingKind::Ordinary:
    case PendingKind::Speculative:
        active_sequence().execution_frontier =
            active_request().pending.base_E + active_request().pending.produced;
        active_sequence().ledger_frontier =
            active_request().pending.base_S + active_request().pending.produced;
        break;
    case PendingKind::None:
        throw std::logic_error("pending generated round has no candidate");
    }
    if (active_sequence().ledger_frontier != active_sequence().execution_frontier + 1 ||
        active_sequence().ledger.size() != active_sequence().ledger_frontier ||
        active_sequence().prefix_identity.size() != active_sequence().ledger_frontier) {
        throw std::logic_error("resolved round did not establish a valid frontier");
    }
    trim_sequence_kv(active_sequence().text_kv_valid, backend_kv_valid());
    if (terminal) {
        active_sequence().dflash_proposal_ready = false;
        active_sequence().mtp_draft_count       = 0;
        release_sequence_growth_entitlement();
        unbind_sequence_kv();
    }
    active_request().lifecycle = terminal ? Lifecycle::Resident : Lifecycle::Active;
    active_request().pending   = {};
}

void ProgramImplCore::finish_active() {
    if (active_request().lifecycle != Lifecycle::Active) {
        throw std::logic_error("finish_active requires Active Program state");
    }
    if (speculative_backend == SpeculativeBackend::DFlash &&
        active_sequence().pending_context_valid) {
        try {
            flush_dflash_context_prefix(active_sequence().pending_context_count);
        } catch (...) {
            make_invalid();
            throw;
        }
    }
    trim_sequence_kv(active_sequence().text_kv_valid, backend_kv_valid());
    release_sequence_growth_entitlement();
    unbind_sequence_kv();
    active_request().lifecycle = Lifecycle::Resident;
}

void ProgramImplCore::abort_request() noexcept {
    if (active_request().lifecycle == Lifecycle::Empty ||
        active_request().lifecycle == Lifecycle::Invalid) {
        return;
    }
    make_invalid();
}

std::uint32_t ProgramImplCore::materialized_tokens() const noexcept {
    return active_request().lifecycle == Lifecycle::Active ||
                   active_request().lifecycle == Lifecycle::Resident
               ? active_sequence().execution_frontier
               : 0;
}

MemorySummary ProgramImplCore::memory_summary() const noexcept {
    MemorySummary out;
    out.device      = device.device;
    out.max_context = capacity;
    out.kv_cache = kv_dtype == DType::BF16 ? KvCacheStorage::BFloat16 : KvCacheStorage::Int8Group64;
    DeviceArena& weights = *model.weights_arena;
    out.weights = ArenaMemorySummary{weights.capacity(), weights.used(), weights.peak_used()};
    out.sequence =
        ArenaMemorySummary{persistent.capacity(), persistent.used(), persistent.peak_used()};
    out.workspace = ArenaMemorySummary{workspace_storage.capacity(), work.used(), work.peak_used()};
    out.workspace_logical_peak_bytes = workspace_logical_peak_bytes;
    out.cuda_graph_allowance_bytes   = graph_allowance_bytes;
    out.kv_payload_bytes             = kv_payload_bytes;
    return out;
}

SpeculativeStats ProgramImplCore::speculative_stats() const {
    return active_request().speculative_stats;
}

void ProgramImplCore::reset_memory_peaks() noexcept {
    model.weights_arena->reset_peak();
    persistent.reset_peak();
    work.reset_peak();
    workspace_logical_peak_bytes = 0;
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
