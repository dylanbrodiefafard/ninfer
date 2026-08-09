#include "targets/qwen3_6/impl/runtime/instance.h"
#include "targets/qwen3_6/impl/runtime/program.h"

#include "core/nvtx.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"

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
                                            std::uint32_t k) {
    schedule::MtpGqaEnvelopes out;
    out.target_verify = {min_frontier + k + 1, max_frontier + k + 1};
    out.batch         = out.target_verify;
    for (std::uint32_t i = 1; i < k; ++i) {
        out.ar[i - 1] = {min_frontier + i + 1, max_frontier + k + i + 1};
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

DecodeGraphProfile& select_graph_profile(DecodeGraphFamily& family, std::uint32_t frontier,
                                         const char* label) {
    const auto it = std::lower_bound(family.profiles.begin(), family.profiles.end(), frontier,
                                     [](const DecodeGraphProfile& profile, std::uint32_t value) {
                                         return profile.max_execution_frontier < value;
                                     });
    if (it == family.profiles.end() || frontier < it->min_execution_frontier) {
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
      prefill_chunk(plan.prefill_chunk), draft_window(plan.draft_window),
      speculative_backend(plan.speculative_backend), kv_dtype(plan.kv_dtype),
      kv_quant_group(plan.kv_quant_group), proposal_head(plan.proposal_head),
      vision_enabled(plan.features.vision), use_cuda_graph(plan.use_cuda_graph),
      kv_payload_bytes(plan.persistent.kv_payload_bytes),
      graph_allowance_bytes(plan.graph_allowance_bytes), workspace_plan(plan.workspace),
      persistent(plan.persistent.bytes), workspace_storage(plan.workspace.capacity),
      work(DeviceSpan{workspace_storage.base(), workspace_storage.capacity()}),
      round_host((static_cast<std::size_t>(draft_window) + 2ULL) * sizeof(std::int32_t)) {
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
    if (plan.persistent.dflash) { dflash.emplace(backing, *plan.persistent.dflash); }
    if (dflash.has_value() != plan.features.dflash()) {
        throw std::logic_error("DFlash state does not match the frozen sequence plan");
    }

    io = qwen3_6::RoundState(backing, plan.persistent.round);
    if (io.mtp.has_value() != (speculative_backend == SpeculativeBackend::Mtp)) {
        throw std::logic_error("round-state MTP extension does not match the sequence plan");
    }
    prefill_hidden  = plan.persistent.prefill_hidden.bind(backing);
    token_counts    = plan.persistent.token_counts.bind(backing);
    sampling_config = plan.persistent.sampling_config.bind(backing);
    tail_hidden     = plan.persistent.tail_hidden.bind(backing);
    boundary_hidden = plan.persistent.boundary_hidden.bind(backing);

    host_count  = static_cast<std::int32_t*>(round_host.data());
    host_tokens = reinterpret_cast<TokenId*>(host_count + 1);
    ledger.reserve(static_cast<std::size_t>(capacity) + 1ULL);
    prefix_identity.reserve(static_cast<std::size_t>(capacity) + 1ULL);

    CUDA_CHECK(cudaMemsetAsync(io.speculative.produced_count.data, 0,
                               io.speculative.produced_count.bytes(), device.stream));
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
    CUDA_CHECK(
        cudaMemsetAsync(io.speculative.stats.data, 0, io.speculative.stats.bytes(), device.stream));
    CUDA_CHECK(cudaMemsetAsync(token_counts.data, 0, token_counts.bytes(), device.stream));
    sampling_host = {};
    CUDA_CHECK(cudaMemcpyAsync(sampling_config.data, &sampling_host, sizeof(sampling_host),
                               cudaMemcpyHostToDevice, device.stream));
    device.synchronize();
    prepare_graphs();
    work.reset();
    work.reset_peak();
    workspace_logical_peak_bytes = 0;
}

ProgramImplCore::~ProgramImplCore() noexcept {
    if (device.stream != nullptr) { (void)cudaStreamSynchronize(device.stream); }
}

void ProgramImplCore::make_invalid() noexcept {
    sequence_kv.reset();
    lifecycle = Lifecycle::Invalid;
    E         = 0;
    S         = 0;
    ledger.clear();
    prefix_identity.clear();
    current_linear_state_slot = LinearStateSlots::prefill_working_slot();
    text_kv_valid             = 0;
    mtp_kv_valid              = 0;
    dflash_context_frontier   = 0;
    dflash_proposal_frontier  = 0;
    dflash_proposal_anchor    = 0;
    dflash_proposal_epoch     = 0;
    pending_context_base      = 0;
    pending_context_count     = 0;
    pending_context_valid     = false;
    dflash_boundary_valid     = false;
    dflash_boundary_frontier  = 0;
    ordinary_tail             = false;
    drafts_ready              = false;
    tail_hidden_valid         = false;
    boundary                  = {};
    pending                   = {};
}

qwen3_6::PagedKVCache* ProgramImplCore::backend_kv_cache() noexcept {
    if (speculative_backend == SpeculativeBackend::Mtp) { return decoder->mtp_cache(); }
    if (speculative_backend == SpeculativeBackend::DFlash && dflash) { return &dflash->full; }
    return nullptr;
}

const qwen3_6::PagedKVCache* ProgramImplCore::backend_kv_cache() const noexcept {
    if (speculative_backend == SpeculativeBackend::Mtp) { return decoder->mtp_cache(); }
    if (speculative_backend == SpeculativeBackend::DFlash && dflash) { return &dflash->full; }
    return nullptr;
}

std::uint32_t ProgramImplCore::backend_kv_valid() const noexcept {
    if (speculative_backend == SpeculativeBackend::Mtp) { return mtp_kv_valid; }
    if (speculative_backend == SpeculativeBackend::DFlash) { return dflash_context_frontier; }
    return 0;
}

void ProgramImplCore::reserve_sequence_kv() {
    if (sequence_kv) { throw std::logic_error("sequence already owns a KV allocation bundle"); }

    std::array<PagedKVReservation, 2> reservations{};
    std::size_t count     = 0;
    reservations[count++] = PagedKVReservation{
        .pool             = &decoder->text_kv.pool(),
        .page_entitlement = decoder->text_kv.pool().logical_page_capacity(),
    };
    if (qwen3_6::PagedKVCache* backend = backend_kv_cache(); backend != nullptr) {
        reservations[count++] = PagedKVReservation{
            .pool             = &backend->pool(),
            .page_entitlement = backend->pool().logical_page_capacity(),
        };
    }

    std::vector<PagedKVAllocation> allocations =
        reserve_paged_kv_bundle(std::span<const PagedKVReservation>(reservations.data(), count));
    SequenceKVBundle bundle;
    bundle.text = std::move(allocations[0]);
    if (count == 2) { bundle.backend.emplace(std::move(allocations[1])); }
    sequence_kv.emplace(std::move(bundle));
}

void ProgramImplCore::bind_sequence_kv() {
    if (!sequence_kv || sequence_kv->text.bound_row() >= 0 ||
        (sequence_kv->backend && sequence_kv->backend->bound_row() >= 0)) {
        throw std::logic_error("KV allocation bundle is unavailable or already bound");
    }
    sequence_kv->text.bind_row(0, device.stream);
    try {
        if (sequence_kv->backend) { sequence_kv->backend->bind_row(0, device.stream); }
    } catch (...) {
        sequence_kv->text.unbind_row();
        throw;
    }
}

void ProgramImplCore::unbind_sequence_kv() noexcept {
    if (!sequence_kv) { return; }
    if (sequence_kv->backend) { sequence_kv->backend->unbind_row(); }
    sequence_kv->text.unbind_row();
}

void ProgramImplCore::materialize_sequence_kv(std::uint32_t main_tokens,
                                              std::uint32_t backend_tokens) {
    if (!sequence_kv || main_tokens > capacity || backend_tokens > capacity) {
        throw std::logic_error("KV materialization request is outside the sequence bundle");
    }
    if (backend_tokens != 0 && !sequence_kv->backend) {
        throw std::logic_error("backend KV materialization requested without an allocation");
    }
    if (main_tokens > sequence_kv->text.mapped_token_capacity()) {
        sequence_kv->text.materialize_tokens(main_tokens, device.stream);
    }
    if (backend_tokens != 0 && backend_tokens > sequence_kv->backend->mapped_token_capacity()) {
        sequence_kv->backend->materialize_tokens(backend_tokens, device.stream);
    }
}

void ProgramImplCore::trim_sequence_kv(std::uint32_t main_tokens, std::uint32_t backend_tokens) {
    if (!sequence_kv || main_tokens > capacity || backend_tokens > main_tokens) {
        throw std::logic_error("KV trim request is outside the sequence bundle");
    }
    if (backend_tokens != 0 && !sequence_kv->backend) {
        throw std::logic_error("backend KV trim requested without an allocation");
    }
    sequence_kv->text.trim_tokens(main_tokens);
    if (sequence_kv->backend) { sequence_kv->backend->trim_tokens(backend_tokens); }
}

qwen3_6::PagedKVCacheView ProgramImplCore::text_kv_view() const {
    if (!sequence_kv) { throw std::logic_error("sequence has no KV allocation bundle"); }
    return decoder->text_kv.execution_view(sequence_kv->text);
}

qwen3_6::PagedKVCacheView ProgramImplCore::mtp_kv_view() const {
    if (speculative_backend != SpeculativeBackend::Mtp) { return {}; }
    if (decoder->mtp_cache() == nullptr || !sequence_kv || !sequence_kv->backend) {
        throw std::logic_error("sequence has no MTP KV allocation");
    }
    return decoder->mtp_cache()->execution_view(*sequence_kv->backend);
}

qwen3_6::PagedKVCacheView ProgramImplCore::dflash_full_kv_view() const {
    if (speculative_backend != SpeculativeBackend::DFlash) { return {}; }
    if (!dflash || !sequence_kv || !sequence_kv->backend) {
        throw std::logic_error("sequence has no DFlash Full KV allocation");
    }
    return dflash->full_view(*sequence_kv->backend);
}

void ProgramImplCore::set_device_i32(Tensor& tensor, std::int32_t value) {
    CUDA_CHECK(
        cudaMemcpyAsync(tensor.data, &value, sizeof(value), cudaMemcpyHostToDevice, device.stream));
}

void ProgramImplCore::ordered_reset() {
    decoder->linear_attention.zero_slot(LinearStateSlots::prefill_working_slot(), device.stream);
    work.reset();
    set_device_i32(io.pos, 0);
    set_device_i32(io.rope_pos, 0);
    set_device_i32(io.rope_delta, 0);
    set_device_i32(io.speculative.accepted_drafts, 0);
    set_device_i32(io.linear_state_read_slot, LinearStateSlots::prefill_working_slot());
    set_device_i32(io.linear_state_snapshot_base_slot,
                   LinearStateSlots::verify_snapshot_base_slot());
    if (io.mtp) { set_device_i32(io.mtp->position, 0); }
    current_linear_state_slot = LinearStateSlots::prefill_working_slot();
    text_kv_valid             = 0;
    mtp_kv_valid              = 0;
    dflash_context_frontier   = 0;
    dflash_proposal_frontier  = 0;
    dflash_proposal_anchor    = 0;
    dflash_proposal_epoch     = 0;
    pending_context_base      = 0;
    pending_context_count     = 0;
    pending_context_valid     = false;
    dflash_boundary_valid     = false;
    dflash_boundary_frontier  = 0;
    ordinary_tail             = false;
    drafts_ready              = false;
    if (dflash) { set_device_i32(dflash->commit_count, 0); }
}

void ProgramImplCore::prepare_graphs() {
    if (!use_cuda_graph) { return; }

    reserve_sequence_kv();
    bind_sequence_kv();
    materialize_sequence_kv(capacity, backend_kv_cache() != nullptr ? capacity : 0);

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
            io.speculative.round_tokens,
            io.speculative.produced_count,
            io.speculative.target_input_ids,
            io.speculative.target_positions,
            io.speculative.accepted_drafts,
            io.linear_state_read_slot,
            io.linear_state_snapshot_base_slot,
            io.speculative.stats,
        };
        if (io.mtp) {
            controls.push_back(io.mtp->alignment_ids);
            controls.push_back(io.mtp->position);
        }
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
    if (dflash) {
        initialize_cyclic_cache(dflash->local);
        initialize_paged_cache(dflash->full);
        CUDA_CHECK(cudaMemsetAsync(dflash->target_features.data, 0, dflash->target_features.bytes(),
                                   device.stream));
        CUDA_CHECK(cudaMemsetAsync(dflash->feature_positions.data, 0,
                                   dflash->feature_positions.bytes(), device.stream));
    }
    device.synchronize();

    const auto prepare_representative = [&](std::uint32_t frontier) {
        work.reset();
        clear_stable_controls();
        decoder->linear_attention.zero_slot(LinearStateSlots::prefill_working_slot(),
                                            device.stream);
        set_device_i32(io.pos, checked_i32(frontier, "graph representative position"));
        set_device_i32(io.rope_pos, checked_i32(frontier, "graph representative rope position"));
        if (io.mtp) {
            set_device_i32(io.mtp->position,
                           checked_i32(frontier, "graph representative MTP position"));
        }
        if (dflash) {
            set_device_i32(io.speculative.produced_count, 1);
            set_device_i32(io.speculative.target_positions,
                           checked_i32(frontier == 0 ? 0 : frontier - 1,
                                       "graph representative DFlash append position"));
        }
    };
    const auto state = [&](std::uint32_t frontier) {
        return schedule::State{device,
                               model,
                               work,
                               text_kv_view(),
                               mtp_kv_view(),
                               dflash_full_kv_view(),
                               dflash ? &*dflash : nullptr,
                               decoder->linear_attention,
                               io,
                               prefill_hidden,
                               prefill_chunk,
                               frontier,
                               static_cast<const ops::SamplingConfig*>(sampling_config.data),
                               proposal_head,
                               &tail_hidden,
                               &boundary_hidden};
    };

    const auto ordinary_profiles = ordinary_graph_profiles(capacity);
    validate_graph_profiles(ordinary_profiles, capacity - 1, "ordinary");
    ordinary_graphs.profiles.reserve(ordinary_profiles.size());
    if (decoder->mtp_cache() != nullptr) {
        ordinary_aligned_graphs.profiles.reserve(ordinary_profiles.size());
    }
    for (const GraphExecutionProfile planned : ordinary_profiles) {
        ordinary_graphs.profiles.emplace_back();
        DecodeGraphProfile& profile        = ordinary_graphs.profiles.back();
        profile.min_execution_frontier     = planned.min;
        profile.max_execution_frontier     = planned.max;
        profile.topology_class             = planned.topology_class;
        const std::uint32_t representative = planned.min;
        const ops::GqaExecutionEnvelope envelope{planned.min + 1, planned.max + 1};
        const auto prepare = [&, representative] { prepare_representative(representative); };

        auto ordinary_state = state(representative);
        schedule::warm_capture_ordinary_round(ordinary_state, false, envelope, prepare,
                                              profile.definition);
        if (decoder->mtp_cache() != nullptr) {
            ordinary_aligned_graphs.profiles.emplace_back();
            DecodeGraphProfile& aligned_profile    = ordinary_aligned_graphs.profiles.back();
            aligned_profile.min_execution_frontier = planned.min;
            aligned_profile.max_execution_frontier = planned.max;
            aligned_profile.topology_class         = planned.topology_class;
            auto aligned_state                     = state(representative);
            schedule::warm_capture_ordinary_round(aligned_state, true, envelope, prepare,
                                                  aligned_profile.definition);
        }
    }

    if (speculative_backend == SpeculativeBackend::Mtp) {
        const auto planned_profiles = mtp_graph_profiles(capacity, draft_window);
        validate_graph_profiles(planned_profiles, capacity - 2 * draft_window, "MTP");
        mtp_graphs.profiles.reserve(planned_profiles.size());
        for (const GraphExecutionProfile planned : planned_profiles) {
            mtp_graphs.profiles.emplace_back();
            DecodeGraphProfile& profile        = mtp_graphs.profiles.back();
            profile.min_execution_frontier     = planned.min;
            profile.max_execution_frontier     = planned.max;
            profile.topology_class             = planned.topology_class;
            const std::uint32_t representative = planned.min;
            const auto prepare = [&, representative] { prepare_representative(representative); };

            auto mtp_state = state(representative);
            schedule::warm_capture_mtp_round(
                mtp_state, draft_window, mtp_gqa_envelopes(planned.min, planned.max, draft_window),
                prepare, profile.definition);
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

    instantiate_graph_family(ordinary_graphs, "ordinary", device, prepare_representative);
    if (decoder->mtp_cache() != nullptr) {
        instantiate_graph_family(ordinary_aligned_graphs, "ordinary aligned", device,
                                 prepare_representative);
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
    unbind_sequence_kv();
    sequence_kv.reset();
}

void ProgramImplCore::install_sampling(const ops::SamplingConfig& config) {
    CUDA_CHECK(cudaMemsetAsync(token_counts.data, 0, token_counts.bytes(), device.stream));
    CUDA_CHECK(
        cudaMemsetAsync(io.speculative.stats.data, 0, io.speculative.stats.bytes(), device.stream));
    sampling_host = config;
    const bool penalties =
        sampling_host.presence_penalty != 0.0F || sampling_host.frequency_penalty != 0.0F;
    sampling_host.token_counts =
        penalties ? static_cast<std::int32_t*>(token_counts.data) : nullptr;
    CUDA_CHECK(cudaMemcpyAsync(sampling_config.data, &sampling_host, sizeof(sampling_host),
                               cudaMemcpyHostToDevice, device.stream));
}

void ProgramImplCore::copy_tail(const Tensor& source) {
    if (source.dtype != DType::BF16 || source.ne[0] != TextConfig::hidden || source.ne[1] != 1) {
        throw std::logic_error("target tail hidden has an invalid shape");
    }
    CUDA_CHECK(cudaMemcpyAsync(tail_hidden.data, source.data, tail_hidden.bytes(),
                               cudaMemcpyDeviceToDevice, device.stream));
    tail_hidden_valid = true;
}

void ProgramImplCore::copy_round_token() {
    CUDA_CHECK(cudaMemcpyAsync(host_tokens, io.token.data, sizeof(TokenId), cudaMemcpyDeviceToHost,
                               device.stream));
}

void ProgramImplCore::mark_workspace_usage(std::size_t phase_bytes) noexcept {
    workspace_logical_peak_bytes = std::max(workspace_logical_peak_bytes, phase_bytes);
}

void ProgramImplCore::flush_dflash_context_prefix(std::uint32_t count) {
    if (!dflash || !pending_context_valid || dflash_context_frontier != pending_context_base ||
        count == 0 || count > pending_context_count) {
        throw std::logic_error("DFlash context flush does not match the pending target features");
    }
    const std::uint32_t context_end = pending_context_base + count;
    materialize_sequence_kv(std::max(text_kv_valid, context_end), context_end);
    schedule::State state{device,
                          model,
                          work,
                          text_kv_view(),
                          mtp_kv_view(),
                          dflash_full_kv_view(),
                          &*dflash,
                          decoder->linear_attention,
                          io,
                          prefill_hidden,
                          prefill_chunk,
                          pending_context_base,
                          static_cast<const ops::SamplingConfig*>(sampling_config.data),
                          proposal_head,
                          &tail_hidden,
                          &boundary_hidden};
    set_device_i32(dflash->commit_count, checked_i32(count, "DFlash context commit count"));
    Tensor features  = dflash->target_features.slice(1, 0, static_cast<std::int32_t>(count));
    Tensor positions = dflash->feature_positions.slice(0, 0, static_cast<std::int32_t>(count));
    mark_workspace_usage(workspace_plan.dflash_context);
    schedule::dflash_append_context(state, features, positions, dflash->commit_count,
                                    {count, count});
    device.synchronize();
    dflash_context_frontier = pending_context_base + count;
    pending_context_base    = 0;
    pending_context_count   = 0;
    pending_context_valid   = false;
}

void ProgramImplCore::validate_licensed_tokens(std::span<const TokenId> tokens) const {
    for (const TokenId token : tokens) {
        if (token < 0 || token >= TextConfig::token_domain) {
            throw std::runtime_error("target returned a token outside the 248077-token domain");
        }
    }
}

runtime::BeginResult ProgramImplCore::begin(PreparedPromptData&& prompt, RequestPlan&& request_plan,
                                            runtime::TransientRegion transient) {
    if (request_plan.impl_ == nullptr) { throw std::invalid_argument("request plan is empty"); }
    const RequestPlanImpl& plan = *request_plan.impl_;
    if (lifecycle == Lifecycle::Active || lifecycle == Lifecycle::Pending) {
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
        if (lifecycle != Lifecycle::Resident ||
            !qwen3_6::detail::prefix_matches(prompt, ledger, prefix_identity, plan.reuse_base)) {
            throw std::logic_error("planned resident prefix is no longer reusable");
        }
        if (plan.reuse == ReusePath::RestoreBoundary &&
            (!boundary.valid || boundary.boundary != plan.reuse_base)) {
            throw std::logic_error("planned boundary checkpoint is unavailable");
        }
    }

    const std::uint32_t base              = plan.reuse_base;
    const bool had_suffix                 = prompt_tokens > base;
    const std::int32_t request_rope_delta = prompt.rope_delta;
    const auto snapshot_boundary          = plan.snapshot_boundary;
    const auto begin_start                = Clock::now();

    // From here on, the old identity is deliberately unreachable. Any failure takes the Program
    // to Invalid rather than attempting a mixed restore/reset fallback.
    lifecycle = Lifecycle::Invalid;
    try {
        if (plan.reuse == ReusePath::FullReset) {
            sequence_kv.reset();
            ordered_reset();
            ledger.clear();
            reserve_sequence_kv();
        } else if (plan.reuse == ReusePath::AppendAtFrontier) {
            if (!sequence_kv) {
                throw std::logic_error("resident prefix has no KV allocation bundle");
            }
            if (text_kv_valid < base) {
                throw std::logic_error("resident Text KV is shorter than E");
            }
            text_kv_valid = base;
            ledger.resize(base);
            set_device_i32(io.linear_state_read_slot, current_linear_state_slot);
            if (dflash && (pending_context_valid || dflash_context_frontier != base)) {
                throw std::logic_error("resident DFlash context is not at the append frontier");
            }
        } else {
            if (!sequence_kv) {
                throw std::logic_error("resident boundary has no KV allocation bundle");
            }
            if (text_kv_valid < base) {
                throw std::logic_error("resident Text KV is shorter than boundary");
            }
            text_kv_valid = base;
            decoder->linear_attention.copy_slot(
                LinearStateSlots::prefix_boundary_slot(decoder->linear_attention.slot_count()),
                LinearStateSlots::prefill_working_slot(), device.stream);
            current_linear_state_slot = LinearStateSlots::prefill_working_slot();
            set_device_i32(io.linear_state_read_slot, current_linear_state_slot);
            ledger.resize(base);
            if (dflash) {
                if (!dflash_boundary_valid || dflash_boundary_frontier != base) {
                    throw std::logic_error("planned DFlash boundary checkpoint is unavailable");
                }
                dflash->restore_boundary(device.stream);
                dflash_context_frontier = base;
            }
        }

        if (plan.prepare_mtp && base != 0) {
            const std::uint32_t mtp_base = base - 1;
            if (decoder->mtp_cache() == nullptr || mtp_kv_valid < mtp_base) {
                throw std::logic_error("reusable MTP prefix is shorter than its bridge position");
            }
            mtp_kv_valid = mtp_base;
        } else if (speculative_backend == SpeculativeBackend::Mtp && sequence_kv->backend) {
            mtp_kv_valid = 0;
        }

        trim_sequence_kv(base, backend_kv_valid());
        bind_sequence_kv();
        std::uint32_t backend_materialized = 0;
        if (speculative_backend == SpeculativeBackend::Mtp && plan.prepare_mtp) {
            backend_materialized = prompt_tokens + draft_window - 1U;
        } else if (speculative_backend == SpeculativeBackend::DFlash) {
            backend_materialized = prompt_tokens;
        }
        materialize_sequence_kv(prompt_tokens, backend_materialized);

        install_sampling(plan.sampling);
        rope_delta = request_rope_delta;
        set_device_i32(io.rope_delta, rope_delta);
        // Invalidate the old checkpoint identity now that execution has started. The separately
        // allocated boundary_hidden tensor is deliberately left untouched until a restore bridge
        // consumes h[B-1] below.
        boundary                 = {};
        dflash_boundary_valid    = false;
        dflash_boundary_frontier = 0;
        timings                  = {};
        ++request_epoch;
        drafts_ready          = false;
        pending_context_valid = false;
        ordinary_tail         = false;

        ledger.assign(prompt.token_ids.begin(), prompt.token_ids.end());
        prefix_identity.assign(prompt);

        schedule::State schedule_state{
            device,
            model,
            work,
            text_kv_view(),
            mtp_kv_view(),
            dflash_full_kv_view(),
            dflash ? &*dflash : nullptr,
            decoder->linear_attention,
            io,
            prefill_hidden,
            prefill_chunk,
            base,
            static_cast<const ops::SamplingConfig*>(sampling_config.data),
            proposal_head,
            &tail_hidden,
            &boundary_hidden};
        bool mtp_prepared    = false;
        bool dflash_prepared = false;

        if (had_suffix && plan.needs_mtp_bridge && !plan.vision) {
            mark_workspace_usage(workspace_plan.mtp_prefill);
            Tensor bridge_token = io.speculative.target_input_ids.slice(0, 0, 1);
            const TokenId token = prompt.token_ids[base];
            CUDA_CHECK(cudaMemcpyAsync(bridge_token.data, &token, sizeof(token),
                                       cudaMemcpyHostToDevice, device.stream));
            const Tensor& bridge_hidden =
                plan.reuse == ReusePath::RestoreBoundary ? boundary_hidden : tail_hidden;
            const auto bridge_rope = prompt_rope_position(prompt, base - 1);
            schedule::mtp_bridge_and_propose(schedule_state, bridge_token, bridge_hidden,
                                             checked_i32(base - 1, "bridge position"), bridge_rope,
                                             false);
            mtp_kv_valid = base;
        }

        if (plan.vision) {
            mark_workspace_usage(workspace_plan.vision_encode);
            mark_workspace_usage(plan.prepare_mtp ? workspace_plan.mtp_prefill
                                                  : workspace_plan.text_prefill);
            std::optional<schedule::MtpBridgeInput> mtp_bridge;
            if (plan.needs_mtp_bridge) {
                const Tensor& bridge_hidden =
                    plan.reuse == ReusePath::RestoreBoundary ? boundary_hidden : tail_hidden;
                mtp_bridge = schedule::MtpBridgeInput{
                    .previous_hidden = &bridge_hidden,
                    .position        = checked_i32(base - 1, "bridge position"),
                    .rope_position   = prompt_rope_position(prompt, base - 1),
                };
            }
            const auto multimodal_start                    = Clock::now();
            const schedule::MultimodalPrefillResult result = schedule::prefill_multimodal(
                schedule_state, prompt, *plan.vision, transient, snapshot_boundary,
                plan.prepare_mtp, mtp_bridge ? &*mtp_bridge : nullptr);
            if (mtp_bridge) { mtp_kv_valid = base; }
            mtp_prepared = result.mtp_prepared;
            copy_tail(prefill_hidden.slice(1, static_cast<int>(result.final_chunk_tokens) - 1, 1));
            copy_round_token();
            device.synchronize();
            const double combined_seconds =
                std::chrono::duration<double>(Clock::now() - multimodal_start).count();
            timings.vision_seconds  = result.vision_seconds;
            timings.prefill_seconds = std::max(0.0, combined_seconds - result.vision_seconds);
        } else {
            const auto text_start = Clock::now();
            if (had_suffix) {
                mark_workspace_usage(plan.prepare_mtp ? workspace_plan.mtp_prefill
                                                      : workspace_plan.text_prefill);
                if (dflash) { mark_workspace_usage(workspace_plan.dflash_context); }
                mtp_prepared = schedule::prefill_text(
                    schedule_state, std::span<const TokenId>(prompt.token_ids).subspan(base),
                    snapshot_boundary, plan.prepare_mtp);
                const std::uint32_t final_length = final_prefill_chunk_length(
                    base, prompt_tokens, prefill_chunk, snapshot_boundary);
                copy_tail(prefill_hidden.slice(1, static_cast<int>(final_length) - 1, 1));
            } else {
                mark_workspace_usage(workspace_plan.ordinary_round);
                if (!tail_hidden_valid) {
                    throw std::logic_error("zero-suffix reuse has no target tail hidden");
                }
                schedule::sample_from_hidden(schedule_state, tail_hidden,
                                             checked_i32(prompt_tokens, "sample position"),
                                             ops::kSamplePurposePrefill);
                set_device_i32(io.rope_pos,
                               checked_i32(prompt_tokens, "rope position") + rope_delta);
                if (plan.prepare_mtp) {
                    const auto bridge_rope = prompt_rope_position(prompt, prompt_tokens - 1);
                    schedule::mtp_bridge_and_propose(
                        schedule_state, io.token, tail_hidden,
                        checked_i32(prompt_tokens - 1, "bridge position"), bridge_rope, true);
                    mtp_prepared = true;
                }
            }
            if (dflash) {
                dflash_context_frontier = prompt_tokens;
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
            timings.prefill_seconds =
                std::chrono::duration<double>(Clock::now() - text_start).count();
        }

        validate_licensed_tokens(std::span<const TokenId>(host_tokens, 1));
        if (ledger.size() != prompt_tokens) {
            throw std::logic_error("candidate token ledger does not match prompt length");
        }
        ledger.push_back(host_tokens[0]);
        prefix_identity.append_generated(1, rope_delta);
        text_kv_valid = prompt_tokens;
        // Target prefill leaves its recurrent state in slot 0. Exact-frontier reuse performs no
        // target work, so it must retain the MTP snapshot that was committed at the old frontier.
        if (had_suffix) { current_linear_state_slot = LinearStateSlots::prefill_working_slot(); }
        mtp_kv_valid = mtp_prepared ? prompt_tokens : 0;
        drafts_ready = mtp_prepared || dflash_prepared;
        if (dflash) {
            if (dflash_context_frontier != prompt_tokens) {
                throw std::logic_error("DFlash prefill did not reach the prompt frontier");
            }
            dflash_proposal_frontier = dflash_prepared ? prompt_tokens : 0;
            dflash_proposal_anchor   = dflash_prepared ? host_tokens[0] : 0;
            dflash_proposal_epoch    = dflash_prepared ? request_epoch : 0;
            ordinary_tail            = !dflash_prepared;
        }
        tail_hidden_valid = true;
        if (snapshot_boundary) {
            boundary.valid            = true;
            boundary.boundary         = *snapshot_boundary;
            boundary.hidden_valid     = true;
            boundary.mtp_prefix_valid = mtp_prepared;
            if (dflash) {
                dflash_boundary_valid    = true;
                dflash_boundary_frontier = *snapshot_boundary;
            }
        }

        pending   = PendingCandidate{.kind          = PendingKind::Begin,
                                     .base_E        = 0,
                                     .base_S        = 0,
                                     .prompt_tokens = prompt_tokens,
                                     .produced      = 1};
        lifecycle = Lifecycle::Pending;
        timings.prefill_seconds =
            std::max(timings.prefill_seconds,
                     std::chrono::duration<double>(Clock::now() - begin_start).count() -
                         timings.vision_seconds);
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

runtime::GeneratedRound ProgramImplCore::decode_round(runtime::RoundBudget budget) {
    if (lifecycle != Lifecycle::Active) {
        throw std::logic_error("decode_round requires Active Program state");
    }
    if (budget.generated_tokens_remaining == 0) {
        throw std::invalid_argument("decode round budget must be nonzero");
    }
    if (E >= capacity) { throw std::out_of_range("Text execution context is full"); }
    if (S != E + 1 || ledger.size() != S || prefix_identity.size() != S) {
        throw std::logic_error("Active frontier is inconsistent");
    }

    if (speculative_backend == SpeculativeBackend::Mtp && drafts_ready &&
        (decoder->mtp_cache() == nullptr || mtp_kv_valid != E)) {
        throw std::logic_error("MTP proposal does not match the Active execution frontier");
    }
    if (speculative_backend == SpeculativeBackend::DFlash) {
        if (drafts_ready &&
            (dflash_context_frontier != E || dflash_proposal_frontier != E ||
             dflash_proposal_anchor != ledger.back() || dflash_proposal_epoch != request_epoch)) {
            throw std::logic_error("DFlash proposal does not match the Active execution frontier");
        }
        if (pending_context_valid && (dflash_context_frontier != pending_context_base ||
                                      pending_context_base + pending_context_count != E)) {
            throw std::logic_error("pending DFlash context update has an invalid frontier");
        }
    }
    const bool use_mtp = speculative_backend == SpeculativeBackend::Mtp && drafts_ready &&
                         mtp_kv_valid == E &&
                         budget.generated_tokens_remaining >= draft_window + 1 &&
                         static_cast<std::uint64_t>(E) + 2ULL * draft_window <= capacity;
    const bool use_dflash = speculative_backend == SpeculativeBackend::DFlash && !ordinary_tail &&
                            (drafts_ready || pending_context_valid) &&
                            budget.generated_tokens_remaining >= draft_window + 1 &&
                            static_cast<std::uint64_t>(E) + draft_window + 1ULL <= capacity;
    const std::uint32_t base_E = E;
    const std::uint32_t base_S = S;
    const nvtx::Name round_name =
        use_dflash ? nvtx::Name::DecodeDFlashRound
                   : (use_mtp ? nvtx::Name::DecodeMtpRound : nvtx::Name::DecodeOrdinaryRound);
    const nvtx::Category round_category =
        use_dflash ? nvtx::Category::DFlash
                   : (use_mtp ? nvtx::Category::Mtp : nvtx::Category::Decode);
    nvtx::ScopedRange round_range(round_name, round_category, base_E);
    try {
        set_device_i32(io.linear_state_read_slot, current_linear_state_slot);
        set_device_i32(io.linear_state_snapshot_base_slot,
                       LinearStateSlots::verify_snapshot_base_slot());
        schedule::State schedule_state{
            device,
            model,
            work,
            text_kv_view(),
            mtp_kv_view(),
            dflash_full_kv_view(),
            dflash ? &*dflash : nullptr,
            decoder->linear_attention,
            io,
            prefill_hidden,
            prefill_chunk,
            base_E,
            static_cast<const ops::SamplingConfig*>(sampling_config.data),
            proposal_head,
            &tail_hidden,
            &boundary_hidden};

        std::uint32_t produced = 1;
        std::uint32_t accepted = 0;
        PendingKind kind       = PendingKind::Ordinary;
        if (use_mtp) {
            mark_workspace_usage(workspace_plan.mtp_round);
            DecodeGraphExecutable* executable = nullptr;
            auto envelopes                    = mtp_gqa_envelopes(base_E, base_E, draft_window);
            if (use_cuda_graph) {
                DecodeGraphProfile& profile = select_graph_profile(mtp_graphs, base_E, "MTP");
                executable                  = &install_graph_profile(mtp_graphs, profile, "MTP");
                envelopes                   = mtp_gqa_envelopes(profile.min_execution_frontier,
                                                                profile.max_execution_frontier, draft_window);
            }
            const std::uint32_t main_materialized = envelopes.target_verify.max_visible_keys;
            const std::uint32_t backend_materialized =
                draft_window > 1
                    ? envelopes.ar[static_cast<std::size_t>(draft_window - 2)].max_visible_keys
                    : envelopes.target_verify.max_visible_keys;
            materialize_sequence_kv(main_materialized, backend_materialized);
            {
                nvtx::ScopedRange submit_range(nvtx::Name::DecodeMtpSubmit, nvtx::Category::Mtp,
                                               base_E);
                schedule::mtp_round(schedule_state, draft_window, envelopes, executable);
                CUDA_CHECK(cudaMemcpyAsync(host_count, io.speculative.produced_count.data,
                                           sizeof(std::int32_t), cudaMemcpyDeviceToHost,
                                           device.stream));
                CUDA_CHECK(cudaMemcpyAsync(host_tokens, io.speculative.round_tokens.data,
                                           (draft_window + 1ULL) * sizeof(TokenId),
                                           cudaMemcpyDeviceToHost, device.stream));
            }
            {
                nvtx::ScopedRange wait_range(nvtx::Name::DecodeMtpWait, nvtx::Category::Control,
                                             base_E);
                device.synchronize();
            }
            if (*host_count <= 0 || *host_count > static_cast<std::int32_t>(draft_window + 1)) {
                throw std::runtime_error("MTP returned an invalid licensed-token count");
            }
            produced = static_cast<std::uint32_t>(*host_count);
            if (produced > budget.generated_tokens_remaining ||
                static_cast<std::uint64_t>(base_E) + produced > capacity) {
                throw std::runtime_error("MTP round exceeded its budget or context capacity");
            }
            accepted                  = produced - 1;
            kind                      = PendingKind::Speculative;
            text_kv_valid             = base_E + produced;
            current_linear_state_slot = LinearStateSlots::committed_snapshot_slot(
                produced, decoder->linear_attention.slot_count());
            mtp_kv_valid      = base_E + produced;
            drafts_ready      = true;
            tail_hidden_valid = true;
        } else if (use_dflash) {
            mark_workspace_usage(workspace_plan.dflash_round);
            const bool steady                 = pending_context_valid;
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
            accepted                  = produced - 1;
            kind                      = PendingKind::Speculative;
            text_kv_valid             = base_E + produced;
            current_linear_state_slot = LinearStateSlots::committed_snapshot_slot(
                produced, decoder->linear_attention.slot_count());
            dflash_context_frontier  = base_E;
            pending_context_base     = base_E;
            pending_context_count    = produced;
            pending_context_valid    = true;
            dflash_proposal_frontier = 0;
            dflash_proposal_anchor   = 0;
            dflash_proposal_epoch    = 0;
            drafts_ready             = false;
            tail_hidden_valid        = true;
        } else {
            if (dflash && pending_context_valid) {
                materialize_sequence_kv(base_E, base_E);
                mark_workspace_usage(workspace_plan.dflash_context);
                Tensor features = dflash->target_features.slice(
                    1, 0, static_cast<std::int32_t>(draft_window + 1));
                Tensor positions = dflash->feature_positions.slice(
                    0, 0, static_cast<std::int32_t>(draft_window + 1));
                schedule::dflash_append_context(schedule_state, features, positions,
                                                io.speculative.produced_count,
                                                {1, draft_window + 1});
                dflash_context_frontier = base_E;
                pending_context_base    = 0;
                pending_context_count   = 0;
                pending_context_valid   = false;
            }
            const bool align_mtp = decoder->mtp_cache() != nullptr && mtp_kv_valid == base_E;
            mark_workspace_usage(workspace_plan.ordinary_round);
            DecodeGraphExecutable* executable = nullptr;
            ops::GqaExecutionEnvelope envelope{base_E + 1, base_E + 1};
            if (use_cuda_graph) {
                DecodeGraphFamily& family   = align_mtp ? ordinary_aligned_graphs : ordinary_graphs;
                const char* label           = align_mtp ? "ordinary aligned" : "ordinary";
                DecodeGraphProfile& profile = select_graph_profile(family, base_E, label);
                executable                  = &install_graph_profile(family, profile, label);
                envelope = {profile.min_execution_frontier + 1, profile.max_execution_frontier + 1};
            }
            const std::uint32_t backend_materialized =
                align_mtp || speculative_backend == SpeculativeBackend::DFlash
                    ? envelope.max_visible_keys
                    : 0;
            materialize_sequence_kv(envelope.max_visible_keys, backend_materialized);
            {
                nvtx::ScopedRange submit_range(nvtx::Name::DecodeOrdinarySubmit,
                                               nvtx::Category::Decode, base_E);
                schedule::ordinary_round(schedule_state, align_mtp, envelope, executable);
                copy_tail(io.verify_hidden.slice(1, 0, 1));
                copy_round_token();
            }
            {
                nvtx::ScopedRange wait_range(nvtx::Name::DecodeOrdinaryWait,
                                             nvtx::Category::Control, base_E);
                device.synchronize();
            }
            text_kv_valid             = base_E + 1;
            current_linear_state_slot = LinearStateSlots::committed_snapshot_slot(
                1, decoder->linear_attention.slot_count());
            if (align_mtp) { mtp_kv_valid = base_E + 1; }
            if (dflash) {
                dflash_context_frontier  = base_E + 1;
                dflash_proposal_frontier = 0;
                dflash_proposal_anchor   = 0;
                dflash_proposal_epoch    = 0;
                ordinary_tail            = true;
            }
            drafts_ready      = false;
            tail_hidden_valid = true;
        }

        validate_licensed_tokens(std::span<const TokenId>(host_tokens, produced));
        ledger.insert(ledger.end(), host_tokens, host_tokens + produced);
        prefix_identity.append_generated(produced, rope_delta);
        pending   = PendingCandidate{.kind          = kind,
                                     .base_E        = base_E,
                                     .base_S        = base_S,
                                     .prompt_tokens = 0,
                                     .produced      = produced};
        lifecycle = Lifecycle::Pending;
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
    if (lifecycle != Lifecycle::Pending) {
        throw std::logic_error("resolve_pending requires a pending generated round");
    }
    if (accepted_tokens == 0 || accepted_tokens > pending.produced) {
        throw std::out_of_range("accepted prefix is outside the pending generated round");
    }
    if (!terminal && accepted_tokens != pending.produced) {
        throw std::logic_error("a continuing round must accept every licensed token");
    }
    if (terminal && speculative_backend == SpeculativeBackend::DFlash &&
        pending.kind == PendingKind::Speculative) {
        try {
            flush_dflash_context_prefix(accepted_tokens);
        } catch (...) {
            make_invalid();
            throw;
        }
        drafts_ready             = false;
        dflash_proposal_frontier = 0;
        dflash_proposal_anchor   = 0;
        dflash_proposal_epoch    = 0;
    }
    if (terminal && pending.kind == PendingKind::Speculative &&
        accepted_tokens < pending.produced) {
        // The output policy may stop inside a target-licensed speculative batch. Target
        // verification has already materialized KV, hidden, and one GDN snapshot for every returned
        // prefix, so commit the exact externally accepted frontier instead of discarding the
        // resident sequence. The next request lets the active drafter rebuild proposals from this
        // target state.
        const std::uint32_t committed_E = pending.base_E + accepted_tokens;
        const std::uint32_t committed_S = pending.base_S + accepted_tokens;
        if (committed_S > ledger.size() || committed_S > prefix_identity.size()) {
            throw std::logic_error("partial speculative terminal exceeds the provisional ledger");
        }
        copy_tail(io.verify_hidden.slice(1, static_cast<int>(accepted_tokens) - 1, 1));
        ledger.resize(committed_S);
        prefix_identity.truncate(committed_S);
        E                         = committed_E;
        S                         = committed_S;
        current_linear_state_slot = LinearStateSlots::committed_snapshot_slot(
            accepted_tokens, decoder->linear_attention.slot_count());
        text_kv_valid = committed_E;
        if (speculative_backend == SpeculativeBackend::Mtp) { mtp_kv_valid = committed_E; }
        set_device_i32(io.pos, checked_i32(committed_E, "partial speculative frontier"));
        set_device_i32(io.rope_pos,
                       checked_i32(committed_E, "partial speculative RoPE frontier") + rope_delta);
        device.synchronize();
        trim_sequence_kv(text_kv_valid, backend_kv_valid());
        unbind_sequence_kv();
        drafts_ready = false;
        lifecycle    = Lifecycle::Resident;
        pending      = {};
        return;
    }
    if (accepted_tokens != pending.produced) {
        throw std::logic_error("a non-speculative terminal round must accept its only token");
    }

    switch (pending.kind) {
    case PendingKind::Begin:
        E = pending.prompt_tokens;
        S = pending.prompt_tokens + 1;
        break;
    case PendingKind::Ordinary:
    case PendingKind::Speculative:
        E = pending.base_E + pending.produced;
        S = pending.base_S + pending.produced;
        break;
    case PendingKind::None:
        throw std::logic_error("pending generated round has no candidate");
    }
    if (S != E + 1 || ledger.size() != S || prefix_identity.size() != S) {
        throw std::logic_error("resolved round did not establish a valid frontier");
    }
    trim_sequence_kv(text_kv_valid, backend_kv_valid());
    if (terminal) { unbind_sequence_kv(); }
    lifecycle = terminal ? Lifecycle::Resident : Lifecycle::Active;
    pending   = {};
}

void ProgramImplCore::finish_active() {
    if (lifecycle != Lifecycle::Active) {
        throw std::logic_error("finish_active requires Active Program state");
    }
    if (speculative_backend == SpeculativeBackend::DFlash && pending_context_valid) {
        try {
            flush_dflash_context_prefix(pending_context_count);
        } catch (...) {
            make_invalid();
            throw;
        }
    }
    trim_sequence_kv(text_kv_valid, backend_kv_valid());
    unbind_sequence_kv();
    lifecycle = Lifecycle::Resident;
}

void ProgramImplCore::abort_request() noexcept {
    if (lifecycle == Lifecycle::Empty || lifecycle == Lifecycle::Invalid) { return; }
    make_invalid();
}

std::uint32_t ProgramImplCore::materialized_tokens() const noexcept {
    return lifecycle == Lifecycle::Active || lifecycle == Lifecycle::Resident ? E : 0;
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
    SpeculativeStats out;
    out.backend      = speculative_backend;
    out.enabled      = speculative_backend != SpeculativeBackend::None;
    out.draft_window = draft_window;
    if (draft_window == 0) { return out; }
    std::vector<std::int64_t> values(static_cast<std::size_t>(io.speculative.stats.ne[0]));
    CUDA_CHECK(cudaMemcpyAsync(values.data(), io.speculative.stats.data,
                               io.speculative.stats.bytes(), cudaMemcpyDeviceToHost,
                               device.stream));
    device.synchronize();
    out.drafted_tokens  = static_cast<std::uint64_t>(std::max<std::int64_t>(0, values[0]));
    out.accepted_tokens = static_cast<std::uint64_t>(std::max<std::int64_t>(0, values[1]));
    out.rounds          = static_cast<std::uint64_t>(std::max<std::int64_t>(0, values[2]));
    out.fallback_steps  = static_cast<std::uint64_t>(std::max<std::int64_t>(0, values[3]));
    out.accepted_per_position.resize(draft_window);
    for (std::uint32_t i = 0; i < draft_window; ++i) {
        out.accepted_per_position[i] =
            static_cast<std::uint64_t>(std::max<std::int64_t>(0, values[4 + i]));
    }
    return out;
}

void ProgramImplCore::reset_memory_peaks() noexcept {
    model.weights_arena->reset_peak();
    persistent.reset_peak();
    work.reset_peak();
    workspace_logical_peak_bytes = 0;
}

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
