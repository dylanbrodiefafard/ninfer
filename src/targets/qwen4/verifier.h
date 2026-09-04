#pragma once

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "core/arena.h"
#include "core/device.h"
#include "ninfer/ops/gated_delta_net_layer.h"
#include "ninfer/ops/ngram_embedding.h"
#include "ninfer/ops/ple.h"
#include "ninfer/ops/qsa.h"
#include "ninfer/ops/qwen4_sparse_moe.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>

namespace ninfer::targets::qwen4::verifier {

inline constexpr std::size_t kLayerCount = 48;
inline constexpr std::size_t kGdnLayerCount = 36;
inline constexpr std::size_t kQsaLayerCount = 12;
inline constexpr std::size_t kObjectCount = 1224;
inline constexpr std::size_t kMappedTensorCount = 97;
inline constexpr std::size_t kDeviceTensorCount = 1127;
inline constexpr std::uint64_t kMappedTensorBytes = 45'996'784'640ULL;
inline constexpr std::uint64_t kDevicePayloadBytes = 26'538'652'160ULL;
inline constexpr std::size_t kPersistentStateBytes = 157'126'664ULL;
inline constexpr std::size_t kExpertStageBytes = ops::kQwen4SparseMoeStageBytes;
inline constexpr std::int32_t kQsaCapacity = 4096;
inline constexpr std::int32_t kPleResetToken = 248044;

struct GrHandles {
    artifact::ObjectHandle norm;
    artifact::ObjectHandle down;
    artifact::ObjectHandle up;
    artifact::ObjectHandle inject;
};

struct GdnHandles {
    artifact::ObjectHandle z;
    artifact::ObjectHandle qkv;
    artifact::ObjectHandle ssm_a;
    artifact::ObjectHandle a;
    artifact::ObjectHandle b;
    artifact::ObjectHandle conv;
    artifact::ObjectHandle dt_bias;
    artifact::ObjectHandle norm;
    artifact::ObjectHandle output;
};

struct QsaHandles {
    artifact::ObjectHandle key;
    artifact::ObjectHandle key_norm;
    artifact::ObjectHandle output;
    artifact::ObjectHandle query_gate;
    artifact::ObjectHandle query_norm;
    artifact::ObjectHandle value;
    artifact::ObjectHandle index_key_norm;
    artifact::ObjectHandle index_key;
    artifact::ObjectHandle index_query_norm;
    artifact::ObjectHandle index_query;
};

struct MoeHandles {
    artifact::ObjectHandle routed_down;
    artifact::ObjectHandle routed_gate;
    artifact::ObjectHandle router;
    artifact::ObjectHandle shared_gate;
    artifact::ObjectHandle shared_gate_proj;
    artifact::ObjectHandle routed_up;
    artifact::ObjectHandle shared_up;
    artifact::ObjectHandle shared_down;
};

struct PleHandles {
    artifact::ObjectHandle table;
    artifact::ObjectHandle conv;
    artifact::ObjectHandle key;
    artifact::ObjectHandle conv_norm;
    artifact::ObjectHandle key_norm;
    artifact::ObjectHandle query_norm;
    artifact::ObjectHandle value;
};

struct LayerHandles {
    GrHandles attention_gr;
    GrHandles ffn_gr;
    MoeHandles moe;
    std::optional<GdnHandles> gdn;
    std::optional<QsaHandles> qsa;
};

struct BindingPlan {
    artifact::ObjectHandle output_head;
    GrHandles final_gr;
    artifact::ObjectHandle token_embedding;
    PleHandles ple;
    std::array<LayerHandles, kLayerCount> layers;
};

struct ArtifactLoadPlan {
    BindingPlan bindings;
    artifact::MaterializationPlan materialization;
};

struct GrWeights {
    Tensor norm;
    Weight down;
    Weight up;
    Tensor inject;
};

struct PleWeights {
    ops::PleMappedIq4NlTable table;
    Tensor conv;
    Weight key;
    Tensor conv_norm;
    Tensor key_norm;
    Tensor query_norm;
    Weight value;
};

struct LayerWeights {
    GrWeights attention_gr;
    GrWeights ffn_gr;
    ops::Qwen4SparseMoeWeights moe;
    std::optional<ops::GatedDeltaNetLayerWeights> gdn;
    std::optional<ops::QsaVerifierWeights> qsa;
};

struct ModelView {
    Weight token_embedding;
    std::array<LayerWeights, kLayerCount> layers;
    PleWeights ple;
    GrWeights final_gr;
    Weight output_head;
};

[[nodiscard]] ArtifactLoadPlan bind_artifact(const artifact::Reader& reader);

class LoadedModel {
public:
    static std::unique_ptr<LoadedModel> load(const std::filesystem::path& path,
                                             DeviceContext& device);
    ~LoadedModel();

    LoadedModel(const LoadedModel&) = delete;
    LoadedModel& operator=(const LoadedModel&) = delete;

    const ModelView& view() const noexcept { return view_; }
    const artifact::MaterializedArtifact& backing() const noexcept { return backing_; }

private:
    LoadedModel(BindingPlan bindings, artifact::MaterializedArtifact materialized);

    artifact::MaterializedArtifact backing_;
    ModelView view_;
};

struct GdnStateView {
    Tensor conv;
    Tensor recurrence;
};

class State {
public:
    State();

    State(const State&) = delete;
    State& operator=(const State&) = delete;

    /** Reset every continuation record. This does not materialize a token embedding. */
    void reset(cudaStream_t stream);

    /** Replace all four residual branches with one token row without changing continuation. */
    void embed_token(const Weight& token_embedding, std::int32_t token_id,
                     cudaStream_t stream);

    const std::array<std::optional<GdnStateView>, kLayerCount>& gdn() const noexcept {
        return gdn_;
    }
    const std::array<std::optional<ops::QsaStateView>, kLayerCount>& qsa() const noexcept {
        return qsa_;
    }
    Tensor ple_conv_state() const noexcept { return ple_conv_state_; }
    Tensor ple_token_history() const noexcept { return ple_token_history_; }
    Tensor residual() const noexcept { return residual_; }
    Tensor mixed_x() const noexcept { return mixed_x_; }
    Tensor block_output() const noexcept { return block_output_; }
    Tensor write_scale() const noexcept { return write_scale_; }
    Tensor ple_device_rows() const noexcept { return ple_device_rows_; }
    void* ple_host_rows() const noexcept { return ple_host_rows_.data(); }
    std::size_t ple_host_rows_bytes() const noexcept { return ple_host_rows_.size(); }
    Tensor expert_device_stage() const noexcept { return expert_device_stage_; }
    void* expert_host_stage() const noexcept { return expert_host_stage_.data(); }
    std::size_t expert_host_stage_bytes() const noexcept { return expert_host_stage_.size(); }
    WorkspaceArena workspace() { return WorkspaceArena(DeviceSpan{workspace_.p, workspace_.bytes}); }
    std::size_t workspace_bytes() const noexcept { return workspace_.bytes; }

private:
    DeviceBuffer persistent_;
    DeviceBuffer residual_storage_;
    DeviceBuffer workspace_;
    DeviceBuffer round_transient_storage_;
    DeviceBuffer ple_device_rows_storage_;
    DeviceBuffer expert_device_stage_storage_;
    PinnedHostBuffer ple_host_rows_;
    PinnedHostBuffer expert_host_stage_;
    std::array<std::optional<GdnStateView>, kLayerCount> gdn_;
    std::array<std::optional<ops::QsaStateView>, kLayerCount> qsa_;
    Tensor ple_conv_state_;
    Tensor ple_token_history_;
    Tensor residual_;
    Tensor mixed_x_;
    Tensor block_output_;
    Tensor write_scale_;
    Tensor ple_device_rows_;
    Tensor expert_device_stage_;
};

struct QsaDiagnosticView {
    std::size_t layer = 0;
    Tensor selected_ids;
    Tensor selected_count;
};

struct RouterDiagnosticView {
    std::size_t layer = 0;
    Tensor selected_ids;
    Tensor selected_weights;
};

struct GrDiagnosticView {
    std::size_t layer = 0;
    Tensor attention_residual;
    Tensor ffn_residual;
};

/**
 * Views produced by one completed eager token. Storage is owned by Program and remains valid until
 * the next execute_token or reset call. token_index is the zero-based QSA append/frontier id.
 */
struct TokenResultView {
    std::int32_t token_index = -1;
    Tensor logits;
    Tensor nll;
    Tensor final_hidden;
    Tensor ple_row_ids;
    std::span<const QsaDiagnosticView> qsa;
    std::span<const RouterDiagnosticView> routers;
    std::span<const GrDiagnosticView> gr;
};

/**
 * Complete unregistered C=1 eager Text verifier over the bound UD-IQ1_S artifact profile.
 *
 * Program owns one 4096-token continuation, every integer control/visibility/PLE staging buffer,
 * BF16 logits, and per-layer QSA/router diagnostic outputs. reset starts a new sequence. Each
 * execute_token refreshes all four residual branches from the numeric token id, executes PLE at
 * zero-based layer 1, then all 48 attention and MoE sublayers, and advances exactly one frontier.
 * Calls are deliberately synchronous because PLE and every routed expert layer have host-address
 * dependencies; no work remains on stream when a public method returns.
 *
 * The caller owns model and stream and must keep both alive for Program's lifetime. This class is
 * not a registry, Engine, CLI, serving, Vision, MTP, batching, or CUDA-Graph entry.
 */
class Program {
public:
    Program(const LoadedModel& model, cudaStream_t stream);

    Program(const Program&) = delete;
    Program& operator=(const Program&) = delete;

    void reset();
    [[nodiscard]] TokenResultView execute_token(std::int32_t token_id,
                                                std::int32_t target_id);

    [[nodiscard]] std::int32_t frontier() const noexcept { return frontier_; }
    [[nodiscard]] const State& state() const noexcept { return state_; }

private:
    const LoadedModel& model_;
    cudaStream_t stream_ = nullptr;
    State state_;
    DeviceBuffer controls_;
    DeviceBuffer ngram_rows_;
    DeviceBuffer ple_embedding_;
    DeviceBuffer final_hidden_;
    DeviceBuffer logits_;
    DeviceBuffer nll_;
    DeviceBuffer gr_trace_;
    DeviceBuffer qsa_selected_ids_;
    DeviceBuffer qsa_selected_counts_;
    DeviceBuffer router_ids_;
    DeviceBuffer router_weights_;
    PinnedHostBuffer host_controls_;
    PinnedHostBuffer host_ngram_rows_;
    ops::PreparedNgramRowConfig ple_ngram_config_;
    std::array<std::int32_t, 2> ple_history_{kPleResetToken, kPleResetToken};
    std::array<QsaDiagnosticView, kQsaLayerCount> qsa_diagnostics_;
    std::array<RouterDiagnosticView, kLayerCount> router_diagnostics_;
    std::array<GrDiagnosticView, kLayerCount> gr_diagnostics_;
    std::int32_t frontier_ = 0;
    bool reset_ = false;
};

} // namespace ninfer::targets::qwen4::verifier
