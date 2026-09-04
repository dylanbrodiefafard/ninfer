#include "targets/qwen4/verifier.h"

#include "artifact/typed_binding.h"
#include "ninfer/ops/gated_residual.h"
#include "ninfer/ops/ggml_embedding.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::targets::qwen4::verifier {
namespace {

using artifact::NumericFormat;
using artifact::TensorPlacement;

constexpr std::string_view kModelId = "qwen4/verification";
constexpr std::string_view kWeightsId = "unsloth-ud-iq1-s-host-staged";

bool is_qsa_layer(std::size_t layer) { return layer >= 3 && (layer - 3) % 4 == 0; }
bool is_iq2_layer(std::size_t layer) {
    constexpr std::array<std::size_t, 14> layers = {1, 2, 4, 14, 16, 25, 30,
                                                    32, 37, 39, 42, 45, 46, 47};
    return std::find(layers.begin(), layers.end(), layer) != layers.end();
}

artifact::ObjectHandle bind(artifact::Binder& binder, const std::string& name,
                            NumericFormat format, std::initializer_list<std::uint64_t> shape,
                            TensorPlacement placement = TensorPlacement::Device) {
    return artifact::bind_tensor(binder, name, format, shape, placement);
}

GrHandles bind_gr(artifact::Binder& binder, const std::string& prefix, bool inject) {
    GrHandles out{
        .norm = bind(binder, prefix + "norm.weight", NumericFormat::FP32, {10240}),
        .down = bind(binder, prefix + "down.weight", NumericFormat::Q8_0, {320, 10240}),
        .up = bind(binder, prefix + "up.weight", NumericFormat::Q8_0, {10240, 320}),
    };
    if (inject) {
        out.inject = bind(binder, prefix + "inject.weight", NumericFormat::FP32, {4, 10240});
    }
    return out;
}

MoeHandles bind_moe(artifact::Binder& binder, std::size_t layer, const std::string& prefix) {
    const NumericFormat routed = is_iq2_layer(layer) ? NumericFormat::IQ2_XXS
                                                      : NumericFormat::IQ1_S;
    const NumericFormat shared = layer == 2 ? NumericFormat::Q6_K : NumericFormat::Q5_K;
    return {
        .routed_down = bind(binder, prefix + "ffn_down_exps.weight", NumericFormat::IQ4_NL,
                            {512, 2560, 640}),
        .routed_gate = bind(binder, prefix + "ffn_gate_exps.weight", routed,
                            {512, 640, 2560}, TensorPlacement::MappedHost),
        .router = bind(binder, prefix + "ffn_gate_inp.weight", NumericFormat::FP32, {512, 2560}),
        .shared_gate = bind(binder, prefix + "ffn_gate_inp_shexp.weight", NumericFormat::FP32,
                            {2560}),
        .shared_gate_proj = bind(binder, prefix + "ffn_gate_shexp.weight", shared, {640, 2560}),
        .routed_up = bind(binder, prefix + "ffn_up_exps.weight", routed, {512, 640, 2560},
                          TensorPlacement::MappedHost),
        .shared_up = bind(binder, prefix + "ffn_up_shexp.weight", shared, {640, 2560}),
        .shared_down = bind(binder, prefix + "ffn_down_shexp.weight", NumericFormat::Q8_0,
                            {2560, 640}),
    };
}

GdnHandles bind_gdn(artifact::Binder& binder, std::size_t layer, const std::string& prefix) {
    const NumericFormat input = layer == 2 ? NumericFormat::Q6_K : NumericFormat::Q5_K;
    return {
        .z = bind(binder, prefix + "attn_gate.weight", input, {6144, 2560}),
        .qkv = bind(binder, prefix + "attn_qkv.weight", input, {10240, 2560}),
        .ssm_a = bind(binder, prefix + "ssm_a", NumericFormat::FP32, {48}),
        .a = bind(binder, prefix + "ssm_alpha.weight", NumericFormat::FP32, {48, 2560}),
        .b = bind(binder, prefix + "ssm_beta.weight", NumericFormat::FP32, {48, 2560}),
        .conv = bind(binder, prefix + "ssm_conv1d.weight", NumericFormat::FP32, {10240, 4}),
        .dt_bias = bind(binder, prefix + "ssm_dt.bias", NumericFormat::FP32, {48}),
        .norm = bind(binder, prefix + "ssm_norm.weight", NumericFormat::FP32, {128}),
        .output = bind(binder, prefix + "ssm_out.weight", NumericFormat::Q6_K, {2560, 6144}),
    };
}

QsaHandles bind_qsa(artifact::Binder& binder, const std::string& prefix) {
    return {
        .key = bind(binder, prefix + "attn_k.weight", NumericFormat::Q5_K, {512, 2560}),
        .key_norm = bind(binder, prefix + "attn_k_norm.weight", NumericFormat::FP32, {256}),
        .output = bind(binder, prefix + "attn_output.weight", NumericFormat::Q5_K, {2560, 6144}),
        .query_gate = bind(binder, prefix + "attn_q.weight", NumericFormat::Q5_K, {12288, 2560}),
        .query_norm = bind(binder, prefix + "attn_q_norm.weight", NumericFormat::FP32, {256}),
        .value = bind(binder, prefix + "attn_v.weight", NumericFormat::Q5_K, {512, 2560}),
        .index_key_norm = bind(binder, prefix + "indexer.k_norm.weight", NumericFormat::FP32,
                               {128}),
        .index_key = bind(binder, prefix + "indexer.k_proj.weight", NumericFormat::BF16,
                          {128, 2560}),
        .index_query_norm = bind(binder, prefix + "indexer.q_norm.weight", NumericFormat::FP32,
                                 {128}),
        .index_query = bind(binder, prefix + "indexer.q_proj.weight", NumericFormat::BF16,
                            {512, 2560}),
    };
}

GrWeights load_gr(const GrHandles& handles, const artifact::MaterializedArtifact& materialized,
                  bool inject) {
    GrWeights out{
        .norm = artifact::materialized_tensor(materialized, handles.norm, NumericFormat::FP32,
                                               {10240}),
        .down = artifact::materialized_weight(materialized, handles.down, NumericFormat::Q8_0,
                                               320, 10240),
        .up = artifact::materialized_weight(materialized, handles.up, NumericFormat::Q8_0,
                                             10240, 320),
    };
    if (inject) {
        out.inject = artifact::materialized_tensor(materialized, handles.inject,
                                                    NumericFormat::FP32, {10240, 4});
    }
    return out;
}

ops::Qwen4MappedRoutedGateUp load_mapped(const MoeHandles& handles,
                                         const artifact::MaterializedArtifact& materialized,
                                         std::size_t layer) {
    return {.gate = materialized.mapped_tensor_bytes(handles.routed_gate),
            .up = materialized.mapped_tensor_bytes(handles.routed_up),
            .qtype = is_iq2_layer(layer) ? QType::GGML_IQ2_XXS : QType::GGML_IQ1_S};
}

ops::Qwen4SparseMoeWeights load_moe(const MoeHandles& handles,
                                    const artifact::MaterializedArtifact& materialized,
                                    std::size_t layer) {
    const NumericFormat shared = layer == 2 ? NumericFormat::Q6_K : NumericFormat::Q5_K;
    return {
        .router = artifact::materialized_weight(materialized, handles.router, NumericFormat::FP32,
                                                512, 2560),
        .routed_gate_up = load_mapped(handles, materialized, layer),
        .routed_down = artifact::materialized_ggml_block_weight(
            materialized, handles.routed_down, NumericFormat::IQ4_NL, {512, 2560, 640}),
        .shared_gate = artifact::materialized_tensor(materialized, handles.shared_gate,
                                                     NumericFormat::FP32, {2560}),
        .shared_gate_proj = artifact::materialized_weight(materialized, handles.shared_gate_proj,
                                                          shared, 640, 2560),
        .shared_up = artifact::materialized_weight(materialized, handles.shared_up, shared, 640,
                                                   2560),
        .shared_down = artifact::materialized_weight(materialized, handles.shared_down,
                                                     NumericFormat::Q8_0, 2560, 640),
    };
}

ops::GatedDeltaNetLayerWeights load_gdn(const GdnHandles& handles,
                                        const artifact::MaterializedArtifact& materialized,
                                        std::size_t layer) {
    const NumericFormat input = layer == 2 ? NumericFormat::Q6_K : NumericFormat::Q5_K;
    return {
        .qkv = artifact::materialized_weight(materialized, handles.qkv, input, 10240, 2560),
        .z = artifact::materialized_weight(materialized, handles.z, input, 6144, 2560),
        .a = artifact::materialized_tensor(materialized, handles.a, NumericFormat::FP32, {2560, 48}),
        .b = artifact::materialized_tensor(materialized, handles.b, NumericFormat::FP32, {2560, 48}),
        .conv = artifact::materialized_tensor(materialized, handles.conv, NumericFormat::FP32,
                                              {4, 10240}),
        .ssm_a = artifact::materialized_tensor(materialized, handles.ssm_a, NumericFormat::FP32,
                                                {48}),
        .dt_bias = artifact::materialized_tensor(materialized, handles.dt_bias,
                                                 NumericFormat::FP32, {48}),
        .norm = artifact::materialized_tensor(materialized, handles.norm, NumericFormat::FP32,
                                              {128}),
        .output = artifact::materialized_weight(materialized, handles.output, NumericFormat::Q6_K,
                                                2560, 6144),
    };
}

ops::QsaVerifierWeights load_qsa(const QsaHandles& handles,
                                 const artifact::MaterializedArtifact& materialized) {
    return {
        .index_query = artifact::materialized_weight(materialized, handles.index_query,
                                                     NumericFormat::BF16, 512, 2560),
        .index_key = artifact::materialized_weight(materialized, handles.index_key,
                                                   NumericFormat::BF16, 128, 2560),
        .core_query_gate = artifact::materialized_weight(materialized, handles.query_gate,
                                                         NumericFormat::Q5_K, 12288, 2560),
        .core_key = artifact::materialized_weight(materialized, handles.key, NumericFormat::Q5_K,
                                                  512, 2560),
        .core_value = artifact::materialized_weight(materialized, handles.value,
                                                    NumericFormat::Q5_K, 512, 2560),
        .output = artifact::materialized_weight(materialized, handles.output, NumericFormat::Q5_K,
                                                2560, 6144),
        .index_query_norm = artifact::materialized_tensor(
            materialized, handles.index_query_norm, NumericFormat::FP32, {128}),
        .index_key_norm = artifact::materialized_tensor(materialized, handles.index_key_norm,
                                                        NumericFormat::FP32, {128}),
        .core_query_norm = artifact::materialized_tensor(materialized, handles.query_norm,
                                                         NumericFormat::FP32, {256}),
        .core_key_norm = artifact::materialized_tensor(materialized, handles.key_norm,
                                                       NumericFormat::FP32, {256}),
    };
}

ModelView load_view(const BindingPlan& bindings,
                    const artifact::MaterializedArtifact& materialized) {
    ModelView view;
    view.token_embedding = artifact::materialized_weight(materialized, bindings.token_embedding,
                                                         NumericFormat::Q4_K, 248320, 2560);
    for (std::size_t layer = 0; layer < kLayerCount; ++layer) {
        const LayerHandles& source = bindings.layers[layer];
        LayerWeights& target = view.layers[layer];
        target.attention_gr = load_gr(source.attention_gr, materialized, true);
        target.ffn_gr = load_gr(source.ffn_gr, materialized, true);
        target.moe = load_moe(source.moe, materialized, layer);
        if (source.gdn) { target.gdn = load_gdn(*source.gdn, materialized, layer); }
        if (source.qsa) { target.qsa = load_qsa(*source.qsa, materialized); }
    }
    const auto table = materialized.mapped_tensor_bytes(bindings.ple.table);
    view.ple = {
        .table = {.data = reinterpret_cast<const std::uint8_t*>(table.data()),
                  .rows = 320001536,
                  .bytes = table.size()},
        .conv = artifact::materialized_tensor(materialized, bindings.ple.conv,
                                              NumericFormat::FP32, {4, 10240}),
        .key = artifact::materialized_weight(materialized, bindings.ple.key,
                                             NumericFormat::Q8_0, 10240, 2560),
        .conv_norm = artifact::materialized_tensor(materialized, bindings.ple.conv_norm,
                                                   NumericFormat::FP32, {10240}),
        .key_norm = artifact::materialized_tensor(materialized, bindings.ple.key_norm,
                                                  NumericFormat::FP32, {10240}),
        .query_norm = artifact::materialized_tensor(materialized, bindings.ple.query_norm,
                                                    NumericFormat::FP32, {10240}),
        .value = artifact::materialized_weight(materialized, bindings.ple.value,
                                               NumericFormat::Q8_0, 2560, 2560),
    };
    view.final_gr = load_gr(bindings.final_gr, materialized, false);
    view.output_head = artifact::materialized_weight(materialized, bindings.output_head,
                                                     NumericFormat::Q4_K, 248320, 2560);
    return view;
}

std::size_t verifier_workspace_bytes() {
    return std::max({ops::gated_residual_workspace_capacity_bytes(),
                     ops::gated_delta_net_layer_workspace_capacity_bytes(),
                     ops::qsa_verifier_workspace_bytes(), ops::ple_workspace_capacity_bytes(1),
                     ops::qwen4_sparse_moe_workspace_capacity_bytes()});
}

} // namespace

ArtifactLoadPlan bind_artifact(const artifact::Reader& reader) {
    if (reader.identity().model_id != kModelId || reader.identity().weights_id != kWeightsId ||
        reader.objects().size() != kObjectCount) {
        throw artifact::ArtifactError("Qwen4 verifier identity or object count mismatch");
    }
    artifact::Binder binder(reader);
    BindingPlan out;
    out.output_head = bind(binder, "output.weight", NumericFormat::Q4_K, {248320, 2560});
    out.final_gr.norm = bind(binder, "output_hc_norm.weight", NumericFormat::FP32, {10240});
    out.final_gr.down = bind(binder, "output_hc_down.weight", NumericFormat::Q8_0, {320, 10240});
    out.final_gr.up = bind(binder, "output_hc_up.weight", NumericFormat::Q8_0, {10240, 320});
    out.ple.table = bind(binder, "per_layer_token_embd.weight", NumericFormat::IQ4_NL,
                         {320001536, 160}, TensorPlacement::MappedHost);
    out.token_embedding = bind(binder, "token_embd.weight", NumericFormat::Q4_K,
                               {248320, 2560});

    for (std::size_t layer = 0; layer < kLayerCount; ++layer) {
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        LayerHandles& target = out.layers[layer];
        target.moe = bind_moe(binder, layer, prefix);
        target.attention_gr = bind_gr(binder, prefix + "hc_attn_", true);
        target.ffn_gr = bind_gr(binder, prefix + "hc_ffn_", true);
        if (is_qsa_layer(layer)) {
            target.qsa = bind_qsa(binder, prefix);
        } else {
            target.gdn = bind_gdn(binder, layer, prefix);
        }
        if (layer == 1) {
            out.ple.conv = bind(binder, prefix + "ple_conv1d.weight", NumericFormat::FP32,
                                {10240, 4});
            out.ple.key = bind(binder, prefix + "ple_key.weight", NumericFormat::Q8_0,
                               {10240, 2560});
            out.ple.conv_norm = bind(binder, prefix + "ple_norm_conv.weight", NumericFormat::FP32,
                                     {10240});
            out.ple.key_norm = bind(binder, prefix + "ple_norm_key.weight", NumericFormat::FP32,
                                    {10240});
            out.ple.query_norm = bind(binder, prefix + "ple_norm_query.weight", NumericFormat::FP32,
                                      {10240});
            out.ple.value = bind(binder, prefix + "ple_value.weight", NumericFormat::Q8_0,
                                 {2560, 2560});
        }
    }
    artifact::MaterializationPlan materialization = binder.finish();
    std::uint64_t mapped_bytes = 0;
    for (const auto& object : materialization.mapped_tensor_objects) { mapped_bytes += object.bytes; }
    std::uint64_t device_bytes = 0;
    for (const auto& object : materialization.device_objects) { device_bytes += object.bytes; }
    if (materialization.mapped_tensor_objects.size() != kMappedTensorCount ||
        materialization.device_objects.size() != kDeviceTensorCount ||
        mapped_bytes != kMappedTensorBytes || device_bytes != kDevicePayloadBytes ||
        materialization.device_capacity_bytes < kDevicePayloadBytes ||
        materialization.device_capacity_bytes >= kDevicePayloadBytes + kDeviceTensorCount * 256ULL) {
        throw artifact::ArtifactError("Qwen4 verifier placement totals changed");
    }
    return {std::move(out), std::move(materialization)};
}

std::unique_ptr<LoadedModel> LoadedModel::load(const std::filesystem::path& path,
                                               DeviceContext& device) {
    artifact::Reader reader(path);
    ArtifactLoadPlan plan = bind_artifact(reader);
    artifact::MaterializedArtifact materialized =
        artifact::materialize(reader, plan.materialization, device);
    return std::unique_ptr<LoadedModel>(
        new LoadedModel(std::move(plan.bindings), std::move(materialized)));
}

LoadedModel::LoadedModel(BindingPlan bindings, artifact::MaterializedArtifact materialized)
    : backing_(std::move(materialized)), view_(load_view(bindings, backing_)) {}

LoadedModel::~LoadedModel() = default;

State::State()
    : persistent_(kPersistentStateBytes), residual_storage_(10240 * 2),
      workspace_(verifier_workspace_bytes()),
      round_transient_storage_(10240 + 8),
      ple_device_rows_storage_(ops::kPleStagedBytes),
      expert_device_stage_storage_(kExpertStageBytes), ple_host_rows_(ops::kPleStagedBytes),
      expert_host_stage_(kExpertStageBytes) {
    auto* base = static_cast<std::byte*>(persistent_.p);
    std::size_t offset = 0;
    auto take = [&](std::size_t bytes) {
        void* pointer = base + offset;
        offset += bytes;
        return pointer;
    };
    for (std::size_t layer = 0; layer < kLayerCount; ++layer) {
        if (is_qsa_layer(layer)) { continue; }
        Tensor conv(take(10240 * 3 * 2), DType::BF16, {10240, 3});
        Tensor recurrence(take(48 * 128 * 128 * 4), DType::FP32, {128, 128, 48});
        gdn_[layer] = GdnStateView{conv, recurrence};
    }
    for (std::size_t layer = 0; layer < kLayerCount; ++layer) {
        if (!is_qsa_layer(layer)) { continue; }
        ops::QsaStateView state{
            .k_codes = Tensor(take(128 * kQsaCapacity * 2), DType::U8,
                              {128, kQsaCapacity, 2}),
            .v_codes = Tensor(take(128 * kQsaCapacity * 2), DType::U8,
                              {128, kQsaCapacity, 2}),
            .k_scales = Tensor(take(16 * kQsaCapacity * 2), DType::FP8_E4M3FN,
                               {16, kQsaCapacity, 2}),
            .v_scales = Tensor(take(16 * kQsaCapacity * 2), DType::FP8_E4M3FN,
                               {16, kQsaCapacity, 2}),
            .raw_index_keys = Tensor(take(128 * kQsaCapacity * 2), DType::BF16,
                                     {128, kQsaCapacity}),
            .positions = Tensor(take(3 * kQsaCapacity * 4), DType::I32,
                                {3, kQsaCapacity}),
        };
        qsa_[layer] = state;
    }
    ple_conv_state_ = Tensor(take(10240 * 9 * 2), DType::BF16, {10240, 9});
    ple_token_history_ = Tensor(take(2 * 4), DType::I32, {2});
    if (offset != kPersistentStateBytes) {
        throw std::logic_error("Qwen4 verifier persistent-state capacity mismatch");
    }
    residual_ = Tensor(residual_storage_.p, DType::BF16, {2560, 4});
    auto* transient = static_cast<std::byte*>(round_transient_storage_.p);
    mixed_x_ = Tensor(transient, DType::BF16, {2560});
    block_output_ = Tensor(transient + 5120, DType::BF16, {2560});
    write_scale_ = Tensor(transient + 10240, DType::BF16, {4});
    ple_device_rows_ = Tensor(ple_device_rows_storage_.p, DType::U8,
                              {ops::kPleIq4NlRowBytes, ops::kPleHeads});
    expert_device_stage_ = Tensor(expert_device_stage_storage_.p, DType::U8,
                                  {static_cast<std::int32_t>(kExpertStageBytes)});
}

void State::reset(cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(persistent_.p, 0, persistent_.bytes, stream));
    static constexpr std::array<std::int32_t, 2> history = {kPleResetToken, kPleResetToken};
    CUDA_CHECK(cudaMemcpyAsync(ple_token_history_.data, history.data(), sizeof(history),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemsetAsync(residual_storage_.p, 0, residual_storage_.bytes, stream));
    CUDA_CHECK(cudaMemsetAsync(round_transient_storage_.p, 0, round_transient_storage_.bytes,
                               stream));
    CUDA_CHECK(cudaMemsetAsync(ple_device_rows_storage_.p, 0, ple_device_rows_storage_.bytes,
                               stream));
    CUDA_CHECK(cudaMemsetAsync(expert_device_stage_storage_.p, 0,
                               expert_device_stage_storage_.bytes, stream));
}

void State::embed_token(const Weight& token_embedding, std::int32_t token_id,
                        cudaStream_t stream) {
    Tensor first = residual_.slice(1, 0, 1).reshape({2560});
    ops::ggml_q4_k_embedding_row(token_embedding, token_id, first, stream);
    for (std::int32_t branch = 1; branch < 4; ++branch) {
        Tensor destination = residual_.slice(1, branch, 1).reshape({2560});
        CUDA_CHECK(cudaMemcpyAsync(destination.data, first.data, first.bytes(),
                                   cudaMemcpyDeviceToDevice, stream));
    }
}

} // namespace ninfer::targets::qwen4::verifier
