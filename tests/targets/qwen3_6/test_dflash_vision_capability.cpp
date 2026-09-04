#include "core/device.h"

#include <ninfer/engine.h>
#include <ninfer/targets/qwen3_6_27b/package.h>
#include <ninfer/targets/qwen3_6_35b_a3b/package.h>

#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

using Package27 = ninfer::targets::qwen3_6_27b::Package;
using Package35 = ninfer::targets::qwen3_6_35b_a3b::Package;

ninfer::EngineOptions dflash_vision_options(std::string model_id, std::string weights_id) {
    ninfer::EngineOptions options;
    options.max_context              = 128;
    options.kv_capacity              = ninfer::KvCapacityPolicy::explicit_capacity(128);
    options.prefill_chunk            = 128;
    options.max_concurrency          = 1;
    options.speculative.backend      = ninfer::SpeculativeBackend::DFlash;
    options.speculative.draft_tokens = 4;
    options.enable_vision           = true;
    options.use_cuda_graph          = false;
    options.model_id                = std::move(model_id);
    options.weights_id              = std::move(weights_id);
    return options;
}

template <class MakePlanner>
int expect_rejection(std::string_view label, MakePlanner&& make_planner) {
    try {
        (void)make_planner();
    } catch (const std::invalid_argument& error) {
        if (std::string(error.what()).find("DFlash and Vision") != std::string::npos) { return 0; }
        std::cerr << label << " rejected for the wrong reason: " << error.what() << '\n';
        return 1;
    }
    std::cerr << label << " unexpectedly accepted DFlash with Vision\n";
    return 1;
}

} // namespace

int main() {
    ninfer::DeviceContext device(0);
    try {
        auto options = dflash_vision_options("qwen3.8-27b", "nvfp4");
        (void)Package27::make_sequence_planner(device, options,
                                               Package27::WeightsProfile::Nvfp4);
    } catch (const std::exception& error) {
        std::cerr << "qwen3.8-27b/nvfp4 rejected DFlash with Vision: " << error.what() << '\n';
        return 1;
    }

    int failures = 0;
    failures += expect_rejection("qwen3.6-27b/nvfp4", [&] {
        auto options = dflash_vision_options("qwen3.6-27b", "nvfp4");
        return Package27::make_sequence_planner(device, options,
                                                Package27::WeightsProfile::Nvfp4);
    });
    failures += expect_rejection("qwen3.8-27b/groupwise-int", [&] {
        auto options = dflash_vision_options("qwen3.8-27b", "groupwise-int");
        return Package27::make_sequence_planner(
            device, options, Package27::WeightsProfile::GroupwiseIntW8Endpoints);
    });
    failures += expect_rejection("qwen3.6-35b-a3b/groupwise-int", [&] {
        auto options = dflash_vision_options("qwen3.6-35b-a3b", "groupwise-int");
        return Package35::make_sequence_planner(device, options,
                                                Package35::WeightsProfile::GroupwiseInt);
    });
    return failures == 0 ? 0 : 1;
}
