#include "core/device.h"

#include <ninfer/engine.h>
#include <ninfer/targets/qwen3_6_27b/package.h>

#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using Package27 = ninfer::targets::qwen3_6_27b::Package;

ninfer::EngineOptions dflash_vision_options(std::string model_id, std::string weights_id) {
    ninfer::EngineOptions options;
    options.max_context              = 128;
    options.kv_capacity              = ninfer::KvCapacityPolicy::explicit_capacity(128);
    options.prefill_chunk            = 128;
    options.mixed_forward            = 0;
    options.max_concurrency          = 1;
    options.speculative.backend      = ninfer::SpeculativeBackend::DFlash;
    options.speculative.draft_tokens = 4;
    options.enable_vision            = true;
    options.use_cuda_graph           = false;
    options.model_id                 = std::move(model_id);
    options.weights_id               = std::move(weights_id);
    return options;
}

} // namespace

int main() {
    ninfer::DeviceContext device(0);
    try {
        auto options = dflash_vision_options("qwen3.8-27b", "nvfp4");
        (void)Package27::make_sequence_planner(device, options, Package27::WeightsProfile::Nvfp4);
    } catch (const std::exception& error) {
        std::cerr << "qwen3.8-27b/nvfp4 rejected DFlash with Vision: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
