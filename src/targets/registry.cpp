#include "targets/registry.h"

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "core/arena.h"
#include "core/device.h"
#include "runtime/engine/kv_capacity.h"

#include "load/binder.h"
#include "ninfer/ops/qsa.h"

#include <chrono>
#include <cstdint>
#include <exception>
#include <future>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace ninfer::targets {
namespace {

using Clock = std::chrono::steady_clock;

void validate_options(const EngineOptions& options) {
    validate_speculative_backend(options.speculative.backend);
    if (options.artifact_path.empty()) {
        throw std::invalid_argument("Engine artifact_path must not be empty");
    }
    if (options.artifact_path.extension() != ".ninfer") {
        throw std::invalid_argument("NInfer accepts only .ninfer artifacts");
    }
    if (options.max_context == 0) {
        throw std::invalid_argument("Engine max_context must be nonzero");
    }
    switch (options.kv_capacity.mode) {
    case KvCapacityMode::Explicit:
        if (options.kv_capacity.explicit_tokens == 0) {
            throw std::invalid_argument("Engine explicit kv_capacity must be nonzero");
        }
        if (options.kv_capacity.automatic_headroom_bytes != 0) {
            throw std::invalid_argument(
                "Engine explicit kv_capacity must not carry automatic headroom");
        }
        break;
    case KvCapacityMode::Automatic:
        if (options.kv_capacity.explicit_tokens != 0) {
            throw std::invalid_argument(
                "Engine automatic kv_capacity must not carry explicit tokens");
        }
        break;
    default:
        throw std::invalid_argument("Engine kv_capacity mode is invalid");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("Engine max_concurrency must be in [1,12]");
    }
    if (options.max_pending_requests == 0 || options.pending_timeout_ms == 0) {
        throw std::invalid_argument("Engine pending request capacity and timeout must be nonzero");
    }
    validate_kv_system_prompt_options(options.kv_ram_capacity_bytes, options.kv_ram_system_prompts);
    validate_kv_disk_options(options.kv_ram_capacity_bytes, options.kv_disk_capacity_bytes,
                             options.kv_disk_location);
}

artifact::LoadProgress artifact_progress(const LoadProgress& progress) {
    return artifact::LoadProgress{.callback = progress.callback};
}

std::size_t runtime_bytes_after_planned_weights(std::uint64_t weight_bytes) {
    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    if (weight_bytes > free_bytes) {
        throw std::invalid_argument("model weights require " + std::to_string(weight_bytes) +
                                    " bytes of device memory, but only " +
                                    std::to_string(free_bytes) +
                                    " bytes are free before loading weights");
    }
    return free_bytes - static_cast<std::size_t>(weight_bytes);
}

std::size_t current_free_device_bytes() {
    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    return free_bytes;
}

// Automatic sizing claims nearly all free device memory, so memory taken by another process
// between sizing and allocation (or before startup) surfaces here. Name the knob that fixes it.
template <class Fn>
decltype(auto) with_automatic_headroom_hint(const KvCapacityPolicy& policy, Fn&& fn) {
    if (policy.mode != KvCapacityMode::Automatic) { return fn(); }
    try {
        return fn();
    } catch (const std::exception& error) {
        throw std::runtime_error(
            std::string(error.what()) + " (automatic KV capacity left " +
            std::to_string(policy.automatic_headroom_bytes / (1024ULL * 1024ULL)) +
            " MiB of device memory free; if a desktop or another process also uses this GPU, "
            "raise --kv-capacity-headroom or set --kv-capacity N)");
    }
}

// Engine options only qwen3.8-flash-next implements.
void reject_flash_next_options(const EngineOptions& options, std::string_view target_key) {
    if (!options.ngram_table_path.empty() || options.qsa_select_budget.has_value()) {
        throw std::invalid_argument("--ngram-table and --qsa-select-budget apply only to "
                                    "qwen3.8-flash-next, not to target '" +
                                    std::string(target_key) + "'");
    }
}

template <class Target, class Loaded, class Instance>
ConstructedTarget construct_registered(const EngineOptions& options, DeviceContext& device,
                                       artifact::Reader& reader, Clock::time_point load_start,
                                       std::string_view target_key) {
    reject_flash_next_options(options, target_key);
    const auto& identity = reader.identity();
    artifact::Binder binder(reader);
    const auto weights_profile                    = Target::resolve_weights(identity, binder);
    const ModelSamplingDefaults sampling_defaults = Target::sampling_defaults(identity.model_id);

    EngineOptions planned = options;
    planned.model_id      = identity.model_id;
    planned.weights_id    = identity.weights_id;
    auto load_plan        = Target::plan_load(binder, planned, weights_profile);
    if (options.kv_disk_capacity_bytes != 0) {
        planned.artifact_tensor_identity = load_plan.cache_tensor_identity(reader);
    }

    auto sequence_planner = Target::make_sequence_planner(device, planned, weights_profile);
    const runtime::SequenceCapacityCurve curve = sequence_planner.capacity_curve();
    const std::size_t preflight_runtime_bytes =
        runtime_bytes_after_planned_weights(load_plan.materialization().device_capacity_bytes);
    with_automatic_headroom_hint(options.kv_capacity, [&] {
        (void)runtime::resolve_kv_capacity(options.kv_capacity, curve, preflight_runtime_bytes);
    });

    std::future<std::unique_ptr<HostPinnedArena>> kv_ram_future;
    if (options.kv_ram_capacity_bytes != 0) {
        const int device_index      = options.device;
        const std::size_t ram_bytes = options.kv_ram_capacity_bytes;
        kv_ram_future               = std::async(std::launch::async, [device_index, ram_bytes] {
            CUDA_CHECK(cudaSetDevice(device_index));
            return std::make_unique<HostPinnedArena>(ram_bytes);
        });
    }

    auto progress     = artifact_progress(options.load_progress);
    auto materialized = artifact::materialize(reader, load_plan.materialization(), device,
                                              progress.callback ? &progress : nullptr);
    const artifact::MaterializationStats stats = materialized.stats();

    auto model = Target::construct_loaded_model(std::move(load_plan), std::move(materialized));
    device.synchronize();
    runtime::KvCapacityResolution capacity_resolution =
        with_automatic_headroom_hint(options.kv_capacity, [&] {
            return runtime::resolve_kv_capacity(options.kv_capacity, curve,
                                                current_free_device_bytes());
        });
    auto sequence_plan = std::move(sequence_planner).finalize(capacity_resolution.main_page_groups);
    if (sequence_plan.device_reservation_bytes() != capacity_resolution.runtime_reservation_bytes ||
        sequence_plan.kv_capacity() != capacity_resolution.resolved_tokens) {
        throw std::logic_error("resolved KV capacity does not match the finalized target plan");
    }
    std::unique_ptr<HostPinnedArena> kv_ram_arena;
    if (kv_ram_future.valid()) { kv_ram_arena = kv_ram_future.get(); }
    auto loaded   = std::make_unique<Loaded>(std::move(model));
    auto instance = with_automatic_headroom_hint(options.kv_capacity, [&] {
        auto constructed =
            std::make_unique<Instance>(std::move(loaded), capacity_resolution,
                                       std::move(sequence_plan), device, std::move(kv_ram_arena));
        device.synchronize();
        return constructed;
    });
    instance->kv_capacity_resolution.available_after_startup_bytes = current_free_device_bytes();

    LoadSummary summary;
    summary.target               = std::string(target_key);
    summary.model_id             = identity.model_id;
    summary.weights_id           = identity.weights_id;
    summary.load_seconds         = std::chrono::duration<double>(Clock::now() - load_start).count();
    summary.upload_seconds       = stats.upload_seconds;
    summary.artifact_bytes_read  = stats.file_bytes;
    summary.host_to_device_bytes = stats.h2d_bytes;
    summary.mapped_host_bytes    = stats.mapped_host_bytes;
    summary.peak_staging_bytes   = stats.peak_staging_bytes;
    summary.tensor_count         = stats.tensor_count;
    summary.resource_count       = stats.resource_count;
    return ConstructedTarget{.active            = std::move(instance),
                             .load              = std::move(summary),
                             .sampling_defaults = sampling_defaults};
}

// qwen3.8-flash-next loads its dense weights and routed experts, then builds its Program over
// the device memory left: QSA state for the requested capacity, then the expert cache. The KV
// capacity is not traded against device memory by a sizing curve; Automatic reserves
// max_concurrency * max_context tokens.
ConstructedTarget construct_flash_next(const EngineOptions& options, DeviceContext& device,
                                       const artifact::Reader& reader,
                                       Clock::time_point load_start) {
    Qwen3_8FlashNext::validate_options(options);
    const auto& identity = reader.identity();
    if (identity.weights_id != Qwen3_8FlashNext::weights_id) {
        throw std::runtime_error("artifact identity '" + identity.model_id + "/" +
                                 identity.weights_id + "' is not supported by target '" +
                                 std::string(Qwen3_8FlashNext::target_key) + "'");
    }
    const ModelSamplingDefaults sampling_defaults =
        Qwen3_8FlashNext::sampling_defaults(identity.model_id);
    auto progress = artifact_progress(options.load_progress);
    auto model    = Qwen3_8FlashNext::load(reader, device, progress.callback ? &progress : nullptr);
    device.synchronize();
    const std::size_t available_after_weights  = current_free_device_bytes();
    const artifact::MaterializationStats stats = model->materialization_stats();
    auto loaded   = std::make_unique<LoadedQwen3_8FlashNext>(std::move(model));
    auto instance = std::make_unique<Qwen3_8FlashNextInstance>(std::move(loaded), options, device);
    device.synchronize();

    const MemorySummary memory = instance->program->memory_summary();
    const std::uint32_t pages =
        memory.kv_capacity / static_cast<std::uint32_t>(ops::kQsaPageTokens);
    runtime::KvCapacityResolution& resolved    = instance->kv_capacity_resolution;
    resolved.mode                              = options.kv_capacity.mode;
    resolved.main_page_groups                  = pages;
    resolved.maximum_main_page_groups          = pages;
    resolved.resolved_tokens                   = memory.kv_capacity;
    resolved.minimum_runtime_reservation_bytes = memory.sequence.capacity_bytes;
    resolved.runtime_reservation_bytes         = memory.sequence.capacity_bytes;
    resolved.available_after_weights_bytes     = available_after_weights;
    resolved.available_after_startup_bytes     = current_free_device_bytes();

    LoadSummary summary;
    summary.target               = std::string(Qwen3_8FlashNext::target_key);
    summary.model_id             = identity.model_id;
    summary.weights_id           = identity.weights_id;
    summary.load_seconds         = std::chrono::duration<double>(Clock::now() - load_start).count();
    summary.upload_seconds       = stats.upload_seconds;
    summary.artifact_bytes_read  = stats.file_bytes;
    summary.host_to_device_bytes = stats.h2d_bytes;
    summary.mapped_host_bytes    = stats.mapped_host_bytes;
    summary.peak_staging_bytes   = stats.peak_staging_bytes;
    summary.tensor_count         = stats.tensor_count;
    summary.resource_count       = stats.resource_count;
    return ConstructedTarget{.active            = std::move(instance),
                             .load              = std::move(summary),
                             .sampling_defaults = sampling_defaults};
}

} // namespace

LoadedQwen3_8FlashNext::LoadedQwen3_8FlashNext(
    std::unique_ptr<Qwen3_8FlashNext::LoadedModel> stable_model)
    : model(std::move(stable_model)), frontend(Qwen3_8FlashNext::make_frontend(*model)) {}

LoadedQwen3_8FlashNext::~LoadedQwen3_8FlashNext() = default;

Qwen3_8FlashNextInstance::Qwen3_8FlashNextInstance(
    std::unique_ptr<LoadedQwen3_8FlashNext> stable_loaded, const EngineOptions& options,
    DeviceContext& device)
    : loaded(std::move(stable_loaded)), request_memory(device, 0), capacity(options.max_context),
      program(Qwen3_8FlashNext::create_program(*loaded->model, options, device)) {}

Qwen3_8FlashNextInstance::~Qwen3_8FlashNextInstance() = default;

LoadedQwen3_6_27B::LoadedQwen3_6_27B(std::unique_ptr<Qwen3_6_27B::LoadedModel> stable_model)
    : model(std::move(stable_model)), frontend(Qwen3_6_27B::make_frontend(*model)) {}

LoadedQwen3_6_27B::~LoadedQwen3_6_27B() = default;

Qwen3_6_27BInstance::Qwen3_6_27BInstance(std::unique_ptr<LoadedQwen3_6_27B> stable_loaded,
                                         runtime::KvCapacityResolution resolution,
                                         Qwen3_6_27B::SequencePlan sequence_plan,
                                         DeviceContext& device,
                                         std::unique_ptr<HostPinnedArena> kv_ram_arena)
    : loaded(std::move(stable_loaded)), kv_capacity_resolution(resolution),
      request_memory(device, sequence_plan.request_transient_capacity_bytes()),
      capacity(sequence_plan.capacity()),
      program(Qwen3_6_27B::create_program(*loaded->model, std::move(sequence_plan), device,
                                          std::move(kv_ram_arena))) {}

Qwen3_6_27BInstance::~Qwen3_6_27BInstance() = default;

ConstructedTarget construct_target(const EngineOptions& options, DeviceContext& device) {
    validate_options(options);
    const auto load_start = Clock::now();

    artifact::Reader reader(options.artifact_path);
    const auto& identity = reader.identity();
    if (identity.model_id == Qwen3_6_27B::model_id) {
        return construct_registered<Qwen3_6_27B, LoadedQwen3_6_27B, Qwen3_6_27BInstance>(
            options, device, reader, load_start, Qwen3_6_27B::target_key);
    }
    if (identity.model_id == Qwen3_8FlashNext::model_id) {
        return construct_flash_next(options, device, reader, load_start);
    }
    throw std::runtime_error("artifact identity '" + identity.model_id + "/" + identity.weights_id +
                             "' has no registered target for this device");
}

} // namespace ninfer::targets
