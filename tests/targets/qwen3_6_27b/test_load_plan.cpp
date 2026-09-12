#include "artifact/binder.h"
#include "artifact/reader.h"
#include "targets/qwen3_6_27b/impl/load/bindings.h"
#include "targets/qwen3_6_27b/impl/variant.h"

#include <ninfer/targets/qwen3_6_27b/package.h>

#include <bit>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <variant>

namespace {

using ninfer::artifact::NumericFormat;
using ninfer::targets::qwen3_6_27b::Package;
using namespace ninfer::targets::qwen3_6_27b::detail;

std::filesystem::path artifact_path(const char* environment, const char* filename) {
    if (const char* value = std::getenv(environment); value != nullptr && *value != '\0') {
        return value;
    }
    return std::filesystem::path(NINFER_SOURCE_DIR) / "out" / filename;
}

ninfer::targets::qwen3_6::StartupFeatures all_features() {
    return {
        .vision        = true,
        .speculative   = ninfer::SpeculativeBackend::Mtp,
        .proposal_head = ninfer::ProposalHead::Optimized,
    };
}

bool valid_divisors(const WeightPlan& weight) {
    if (weight.format != NumericFormat::NVFP4) { return false; }
    const float weight_divisor = std::bit_cast<float>(weight.weight_scale_divisor_bits);
    const float input_divisor  = std::bit_cast<float>(weight.input_scale_divisor_bits);
    return std::isfinite(weight_divisor) && weight_divisor > 0.0F && std::isfinite(input_divisor) &&
           input_divisor > 0.0F;
}

int verify_groupwise(const std::filesystem::path& path) {
    ninfer::artifact::Reader reader(path);
    ninfer::artifact::Binder binder(reader);
    if (Package::resolve_weights(reader.identity(), binder) != WeightsProfile::GroupwiseInt) {
        std::cerr << "groupwise identity resolved to the wrong profile\n";
        return 1;
    }
    const ArtifactLoadPlan plan =
        bind_artifact(binder, WeightsProfile::GroupwiseInt, all_features());
    if (plan.materialization.object_count != 1124 ||
        plan.materialization.device_objects.size() != 1118 ||
        plan.materialization.host_objects.size() != 6 ||
        plan.materialization.device_capacity_bytes == 0) {
        std::cerr << "groupwise materialization plan is incomplete\n";
        return 1;
    }
    if (plan.bindings.token_embedding.format != NumericFormat::Q6G64_F16S ||
        plan.bindings.output_head.format != NumericFormat::Q6G64_F16S) {
        std::cerr << "groupwise vocabulary endpoints have the wrong storage profile\n";
        return 1;
    }
    for (const TextLayerPlan& layer : plan.bindings.text_layers) {
        if (layer.is_full_attention) {
            if (!std::holds_alternative<SplitAttentionProjectionPlan>(layer.attention.projection)) {
                std::cerr << "groupwise attention parent boundary changed\n";
                return 1;
            }
        } else if (!std::holds_alternative<SplitGdnInputProjectionPlan>(
                       layer.gdn.input_projection)) {
            std::cerr << "groupwise GDN parent boundary changed\n";
            return 1;
        }
        if (layer.mlp.gate_up.format != NumericFormat::Q4G64_F16S ||
            layer.mlp.down.format != NumericFormat::Q5G64_F16S) {
            std::cerr << "groupwise MLP storage profile changed\n";
            return 1;
        }
    }
    return 0;
}

int verify_nvfp4(const std::filesystem::path& path) {
    ninfer::artifact::Reader reader(path);
    ninfer::artifact::Binder binder(reader);
    if (Package::resolve_weights(reader.identity(), binder) != WeightsProfile::Nvfp4) {
        std::cerr << "NVFP4 identity resolved to the wrong profile\n";
        return 1;
    }
    const ArtifactLoadPlan plan = bind_artifact(binder, WeightsProfile::Nvfp4, all_features());
    if (plan.materialization.object_count != 1307 ||
        plan.materialization.device_objects.size() != 1054 ||
        plan.materialization.host_objects.size() != 6 ||
        plan.materialization.object_count - plan.materialization.device_objects.size() -
                plan.materialization.host_objects.size() !=
            247 ||
        plan.materialization.device_capacity_bytes == 0) {
        std::cerr << "NVFP4 materialization plan is incomplete: objects="
                  << plan.materialization.object_count
                  << " device=" << plan.materialization.device_objects.size()
                  << " host=" << plan.materialization.host_objects.size() << '\n';
        return 1;
    }
    if (plan.bindings.token_embedding.format != NumericFormat::W8G32_F16S ||
        plan.bindings.output_head.format != NumericFormat::W8G32_F16S) {
        std::cerr << "NVFP4 vocabulary endpoints have the wrong storage profile\n";
        return 1;
    }

    std::size_t nvfp4_weights          = 0;
    std::size_t bf16_attention_inputs  = 0;
    std::size_t bf16_attention_outputs = 0;
    std::size_t bf16_gdn_outputs       = 0;
    const auto count_weight            = [&](const WeightPlan& weight) {
        if (weight.format == NumericFormat::NVFP4) {
            ++nvfp4_weights;
            return valid_divisors(weight);
        }
        return true;
    };
    for (const TextLayerPlan& layer : plan.bindings.text_layers) {
        if (!count_weight(layer.mlp.gate_up) || !count_weight(layer.mlp.down)) {
            std::cerr << "NVFP4 MLP divisor is invalid\n";
            return 1;
        }
        if (layer.is_full_attention) {
            const auto* fused =
                std::get_if<FusedAttentionProjectionPlan>(&layer.attention.projection);
            if (fused == nullptr || !count_weight(fused->query_key_gate_value) ||
                !count_weight(layer.attention.output)) {
                std::cerr << "NVFP4 attention binding is invalid\n";
                return 1;
            }
            bf16_attention_inputs +=
                fused->query_key_gate_value.format == NumericFormat::BF16 ? 1 : 0;
            bf16_attention_outputs += layer.attention.output.format == NumericFormat::BF16 ? 1 : 0;
        } else {
            const auto* fused =
                std::get_if<FusedGdnInputProjectionPlan>(&layer.gdn.input_projection);
            if (fused == nullptr || !count_weight(fused->query_key_value_z) ||
                !count_weight(layer.gdn.output)) {
                std::cerr << "NVFP4 GDN binding is invalid\n";
                return 1;
            }
            bf16_gdn_outputs += layer.gdn.output.format == NumericFormat::BF16 ? 1 : 0;
        }
    }
    if (nvfp4_weights != 247 || bf16_attention_inputs != 6 || bf16_attention_outputs != 2 ||
        bf16_gdn_outputs != 1) {
        std::cerr << "NVFP4 Text inventory has the wrong storage profile: nvfp4=" << nvfp4_weights
                  << " bf16_attention_input=" << bf16_attention_inputs
                  << " bf16_attention_output=" << bf16_attention_outputs
                  << " bf16_gdn_output=" << bf16_gdn_outputs << '\n';
        return 1;
    }
    return 0;
}

int verify_rejection(const std::filesystem::path& path) {
    ninfer::artifact::Reader reader(path);
    ninfer::artifact::Binder binder(reader);
    try {
        (void)Package::resolve_weights({"qwen3.6-27b", "unknown"}, binder);
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        if (message.find("qwen3.6-27b/unknown") != std::string::npos) { return 0; }
    }
    std::cerr << "unknown weights identity was not rejected with the full identity\n";
    return 1;
}

int verify_profile_mismatch_rejection() {
    ninfer::DeviceContext device(0);
    ninfer::EngineOptions options;
    options.max_context    = 128;
    options.kv_capacity    = ninfer::KvCapacityPolicy::explicit_capacity(128);
    options.prefill_chunk  = 128;
    options.use_cuda_graph = false;
    auto planner = Package::make_sequence_planner(device, options, WeightsProfile::GroupwiseInt);
    const std::uint32_t pages = planner.capacity_curve().minimum_main_page_groups;
    auto sequence             = std::move(planner).finalize(pages);
    RuntimeModelView empty_model;
    try {
        (void)ninfer::targets::qwen3_6::create_program<Variant>(empty_model, WeightsProfile::Nvfp4,
                                                                std::move(sequence), device,
                                                                nullptr);
    } catch (const std::invalid_argument& error) {
        if (std::string(error.what()).find("weights profile") != std::string::npos) { return 0; }
    }
    std::cerr << "mismatched load/sequence weights profiles were not rejected\n";
    return 1;
}

int verify_selective(const std::filesystem::path& path) {
    ninfer::artifact::Reader reader(path);
    ninfer::artifact::Binder binder(reader);
    if (Package::resolve_weights(reader.identity(), binder) != WeightsProfile::SelectiveFp8Nvfp4) {
        throw std::runtime_error("selective FP8 profile was not identified");
    }
    const auto plan = bind_artifact(binder, WeightsProfile::SelectiveFp8Nvfp4, all_features());
    if (plan.bindings.token_embedding.format != NumericFormat::W8G32_F16S ||
        plan.bindings.output_head.format != NumericFormat::W8G32_F16S) {
        throw std::runtime_error("selective FP8 changed vocabulary storage");
    }
    int fp8 = 0;
    for (std::size_t i = 0; i < plan.bindings.text_layers.size(); ++i) {
        const auto& layer = plan.bindings.text_layers[i];
        const auto count = [&](const WeightPlan& w) {
            if (w.format == NumericFormat::FP8_E4M3FN_ROW_BF16S) ++fp8;
            if (w.format == NumericFormat::NVFP4 && !valid_divisors(w)) {
                throw std::runtime_error("selective profile lost NVFP4 divisor");
            }
        };
        count(layer.mlp.gate_up); count(layer.mlp.down);
        if (layer.is_full_attention) {
            const auto& input = std::get<FusedAttentionProjectionPlan>(layer.attention.projection).query_key_gate_value;
            count(input); count(layer.attention.output);
            if ((i <= 23 && input.format != NumericFormat::BF16) ||
                (i <= 7 && layer.attention.output.format != NumericFormat::BF16)) {
                throw std::runtime_error("selective profile changed protected attention weight");
            }
        } else {
            count(std::get<FusedGdnInputProjectionPlan>(layer.gdn.input_projection).query_key_value_z);
            count(layer.gdn.output);
            if (i == 4 && layer.gdn.output.format != NumericFormat::BF16) {
                throw std::runtime_error("selective profile changed protected GDN weight");
            }
        }
    }
    if (fp8 == 0) throw std::runtime_error("selective profile contains no FP8 matrices");
    std::cout << "selective FP8 binding verified: " << fp8 << " matrices\n";
    return 0;
}

} // namespace

int main() {
    if (const char* selected = std::getenv("NINFER_SELECTIVE_FP8_WEIGHTS")) {
        return verify_selective(selected);
    }
    const std::filesystem::path groupwise =
        artifact_path("NINFER_QWEN3_6_27B_WEIGHTS", "qwen3_6_27b.ninfer");
    const std::filesystem::path nvfp4 =
        artifact_path("NINFER_QWEN3_6_27B_NVFP4_WEIGHTS", "qwen3_6_27b_nvfp4.ninfer");
    if (!std::filesystem::is_regular_file(groupwise) || !std::filesystem::is_regular_file(nvfp4)) {
        std::cerr << "skip: both real 27B artifacts are required: groupwise=" << groupwise
                  << " nvfp4=" << nvfp4 << '\n';
        return 77;
    }
    if (const int result = verify_rejection(groupwise); result != 0) { return result; }
    if (const int result = verify_profile_mismatch_rejection(); result != 0) { return result; }
    if (const int result = verify_groupwise(groupwise); result != 0) { return result; }
    if (const int result = verify_nvfp4(nvfp4); result != 0) { return result; }
    return 0;
}
