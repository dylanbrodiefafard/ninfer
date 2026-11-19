#include <ninfer/targets/qwen3_8_flash_next/package.h>

#include "load/binder.h"
#include "ninfer/ops/qsa.h"
#include "runtime/forward.h"
#include "runtime/program.h"
#include "targets/startup_options.h"

#include "artifact/materializer.h"
#include "artifact/reader.h"

#include <memory>
#include <stdexcept>
#include <string>

namespace ninfer::targets::qwen3_8_flash_next {
namespace {

// The checkpoint's published presets (Thinking: temperature 1.0, top-p 0.95, top-k 20; Instruct:
// temperature 0.7, top-p 0.80, top-k 20, presence penalty 1.5).
constexpr ModelSamplingDefaults kFlashNextDefaults{
    .thinking     = {.temperature       = 1.0F,
                     .top_k             = 20,
                     .top_p             = 0.95F,
                     .min_p             = 0.0F,
                     .presence_penalty  = 0.0F,
                     .frequency_penalty = 0.0F},
    .non_thinking = {.temperature       = 0.7F,
                     .top_k             = 20,
                     .top_p             = 0.80F,
                     .min_p             = 0.0F,
                     .presence_penalty  = 1.5F,
                     .frequency_penalty = 0.0F},
};

void reject(bool unsupported, const char* what) {
    if (unsupported) {
        throw std::invalid_argument(std::string("qwen3.8-flash-next does not support ") + what);
    }
}

} // namespace

ModelSamplingDefaults Package::sampling_defaults(std::string_view model) {
    if (model == model_id) { return kFlashNextDefaults; }
    throw std::runtime_error("model '" + std::string(model) +
                             "' has no sampling defaults in target package '" +
                             std::string(target_key) + "'");
}

void Package::validate_options(const EngineOptions& options) {
    validate_expert_helper_option(options.expert_helper_pci, model_id);
    validate_speculative_options(options.speculative);
    (void)resolve_kv_idle_timeout(options.kv_idle_timeout, options.kv_ram_capacity_bytes, model_id);
    (void)resolve_kv_idle_headroom(options.kv_idle_headroom, options.kv_idle_timeout,
                                   options.kv_ram_capacity_bytes, model_id);
    if (options.kv_ram_system_prompts != 0 && options.kv_ram_capacity_bytes == 0) {
        throw std::invalid_argument("--kv-ram-system-prompts requires --kv-ram-capacity");
    }
    if (options.kv_disk_capacity_bytes != 0 &&
        (options.kv_ram_capacity_bytes == 0 || options.kv_disk_location.empty())) {
        throw std::invalid_argument("the disk KV tier requires RAM capacity and a disk location");
    }
    if (options.kv_disk_capacity_bytes == 0 && !options.kv_disk_location.empty()) {
        throw std::invalid_argument("a disk KV location requires disk capacity");
    }
    reject(!options.context_checkpoint_marks.empty(),
           "context checkpoints (--context-checkpoints)");
    reject(options.enable_vision, "Vision (--vision)");
    reject(options.kv_cache == KvCacheStorage::Int8Group64, "INT8 KV (--kv-dtype int8)");
    reject(options.sage_attn || options.keep_frac < 1.0F || options.xattn_tau < 1.0F,
           "sparse or Sage attention (--sage, --keep-frac, --xattn-tau)");
    reject(!options.use_cuda_graph, "eager decode (--no-cuda-graph)");
    if (options.ngram_table_path.empty()) {
        throw std::invalid_argument("qwen3.8-flash-next requires --ngram-table");
    }
    // A sub-chunk never exceeds its super-chunk, the target's prefill and interruption unit.
    const auto super_chunk = static_cast<std::uint32_t>(ForwardOptions{}.super_chunk_tokens);
    if (options.prefill_chunk == 0 || options.prefill_chunk > super_chunk) {
        throw std::invalid_argument("qwen3.8-flash-next --prefill-chunk must be in [1, " +
                                    std::to_string(super_chunk) + "]");
    }
    if (options.max_context > static_cast<std::uint32_t>(ops::kQsaMaximumCapacity)) {
        throw std::invalid_argument("qwen3.8-flash-next max_context must be at most " +
                                    std::to_string(ops::kQsaMaximumCapacity));
    }
}

std::unique_ptr<LoadedModel> Package::load(const artifact::Reader& reader, DeviceContext& device,
                                           artifact::LoadProgress* progress) {
    return std::make_unique<LoadedModel>(reader, device, progress);
}

Package::Frontend Package::make_frontend(const LoadedModel& model) {
    return qwen3_6::make_frontend(model.frontend_resources(), false);
}

std::unique_ptr<Package::Program> Package::create_program(const LoadedModel& model,
                                                          const EngineOptions& options,
                                                          DeviceContext& device) {
    return std::make_unique<Program>(model, options, device);
}

} // namespace ninfer::targets::qwen3_8_flash_next
