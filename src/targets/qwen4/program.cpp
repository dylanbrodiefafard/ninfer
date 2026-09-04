#include "targets/qwen4/verifier.h"

#include "targets/qwen4/profile_range.h"

#include "core/device.h"
#include "ninfer/ops/gated_residual.h"
#include "ninfer/ops/ggml_block_linear.h"
#include "ninfer/ops/ngram_embedding.h"
#include "ninfer/ops/nll_from_logits.h"

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace ninfer::targets::qwen4::verifier {
namespace {

constexpr std::int32_t kVocabulary = 248320;
constexpr std::int32_t kHidden = 2560;
constexpr std::int32_t kBranches = 4;
constexpr std::size_t kResidualBytes =
    static_cast<std::size_t>(kHidden) * kBranches * sizeof(std::uint16_t);

constexpr std::size_t kTokenOffset = 0;
constexpr std::size_t kValidOffset = 4;
constexpr std::size_t kAppendOffset = 8;
constexpr std::size_t kTargetOffset = 12;
constexpr std::size_t kPositionOffset = 16;
constexpr std::size_t kVisibleOffsetsOffset = 32;
constexpr std::size_t kHostPleHistoryOffset = 64;
constexpr std::size_t kVisibleIdsOffset = 256;
constexpr std::size_t kControlBytes =
    kVisibleIdsOffset + static_cast<std::size_t>(kQsaCapacity) * sizeof(std::int32_t);

constexpr std::size_t kQsaIdsPerLayerBytes =
    static_cast<std::size_t>(ops::kQsaSelectedCapacity) * sizeof(std::int32_t);
constexpr std::size_t kRouterStride = 64;

static_assert(kControlBytes <= static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()));
static_assert(kHostPleHistoryOffset + 2 * sizeof(std::int32_t) <= kVisibleIdsOffset);

void validate_topology(const ModelView& view) {
    if (view.token_embedding.n != kVocabulary || view.output_head.n != kVocabulary) {
        throw std::invalid_argument("Qwen4 Program requires the exact 248320-row embedding/head");
    }
    std::size_t qsa_count = 0;
    for (std::size_t layer = 0; layer < kLayerCount; ++layer) {
        const bool qsa_layer = layer >= 3 && (layer - 3) % 4 == 0;
        const LayerWeights& weights = view.layers[layer];
        if (weights.qsa.has_value() != qsa_layer || weights.gdn.has_value() == qsa_layer) {
            throw std::invalid_argument("Qwen4 Program layer schedule is incomplete");
        }
        qsa_count += weights.qsa.has_value();
    }
    if (qsa_count != kQsaLayerCount || view.ple.table.rows != 320001536 ||
        view.ple.table.bytes != 28'800'138'240ULL) {
        throw std::invalid_argument("Qwen4 Program requires the exact QSA/PLE profile");
    }
}

Tensor tensor_at(DeviceBuffer& storage, std::size_t offset, DType dtype,
                 std::initializer_list<std::int32_t> shape) {
    return Tensor(static_cast<std::byte*>(storage.p) + offset, dtype, shape);
}

class TokenExecutionGuard {
public:
    explicit TokenExecutionGuard(bool& ready) : ready_(ready) {}
    ~TokenExecutionGuard() {
        if (!complete_) { ready_ = false; }
    }
    void complete() noexcept { complete_ = true; }

private:
    bool& ready_;
    bool complete_ = false;
};

} // namespace

Program::Program(const LoadedModel& model, DeviceContext& device,
                 DiagnosticSnapshots diagnostic_snapshots)
    : model_(model), stream_(device.stream), transfer_stream_(device.copy_stream),
      controls_(kControlBytes),
      ngram_rows_(ops::kPleHeads * sizeof(std::int32_t)),
      ple_embedding_(ops::kPleEmbeddingWidth * sizeof(std::uint16_t)),
      final_hidden_(kHidden * sizeof(std::uint16_t)),
      logits_(static_cast<std::size_t>(kVocabulary) * sizeof(std::uint16_t)),
      nll_(sizeof(float)),
      gr_trace_(diagnostic_snapshots == DiagnosticSnapshots::Enabled
                    ? 2 * kLayerCount * kResidualBytes
                    : 0),
      qsa_selected_ids_(kQsaLayerCount * kQsaIdsPerLayerBytes),
      qsa_selected_counts_(kQsaLayerCount * sizeof(std::int32_t)),
      router_ids_(kLayerCount * kRouterStride),
      router_weights_(kLayerCount * kRouterStride), host_controls_(kControlBytes),
      host_ngram_rows_(ops::kPleHeads * sizeof(std::int32_t)),
      ple_ngram_config_(ops::prepare_ngram_row_config({
          .vocabulary_size = kVocabulary,
          .eos_token_id = kPleResetToken,
          .ple_layer_index = 0,
          .seed = 1234,
          .vocab_base = 20'000'000,
      })),
      diagnostic_snapshots_(diagnostic_snapshots) {
    validate_topology(model_.view());
    try {
        CUDA_CHECK(cudaEventCreateWithFlags(&moe_route_ready_, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(
            &moe_ids_ready_, cudaEventDisableTiming | cudaEventBlockingSync));
        for (std::size_t slot = 0; slot < ops::kQwen4SparseMoePipelineSlots; ++slot) {
            CUDA_CHECK(cudaEventCreateWithFlags(
                &moe_transfer_ready_[slot], cudaEventDisableTiming | cudaEventBlockingSync));
            CUDA_CHECK(cudaEventCreateWithFlags(&moe_consumer_complete_[slot],
                                                cudaEventDisableTiming));
        }
        std::size_t qsa = 0;
        for (std::size_t layer = 0; layer < kLayerCount; ++layer) {
            if (model_.view().layers[layer].qsa) {
                qsa_diagnostics_[qsa] = {
                    .layer = layer,
                    .selected_ids = tensor_at(qsa_selected_ids_, qsa * kQsaIdsPerLayerBytes,
                                              DType::I32, {ops::kQsaSelectedCapacity}),
                    .selected_count = tensor_at(qsa_selected_counts_,
                                                qsa * sizeof(std::int32_t), DType::I32, {1}),
                };
                ++qsa;
            }
            router_diagnostics_[layer] = {
                .layer = layer,
                .selected_ids = tensor_at(router_ids_, layer * kRouterStride, DType::I32,
                                          {ops::kQwen4SparseMoeTopK}),
                .selected_weights = tensor_at(router_weights_, layer * kRouterStride,
                                              DType::FP32, {ops::kQwen4SparseMoeTopK}),
            };
            if (diagnostic_snapshots_ == DiagnosticSnapshots::Enabled) {
                gr_diagnostics_[layer] = {
                    .layer = layer,
                    .attention_residual = tensor_at(gr_trace_, (2 * layer) * kResidualBytes,
                                                    DType::BF16, {kHidden, kBranches}),
                    .ffn_residual = tensor_at(gr_trace_, (2 * layer + 1) * kResidualBytes,
                                              DType::BF16, {kHidden, kBranches}),
                };
            }
        }
        if (qsa != kQsaLayerCount) {
            throw std::logic_error("Qwen4 Program QSA count changed");
        }
    } catch (...) {
        destroy_pipeline_events();
        throw;
    }
}

Program::~Program() {
    if (stream_ != nullptr) { (void)cudaStreamSynchronize(stream_); }
    if (transfer_stream_ != nullptr) { (void)cudaStreamSynchronize(transfer_stream_); }
    destroy_pipeline_events();
}

void Program::destroy_pipeline_events() noexcept {
    for (cudaEvent_t& event : moe_consumer_complete_) {
        if (event != nullptr) {
            (void)cudaEventDestroy(event);
            event = nullptr;
        }
    }
    for (cudaEvent_t& event : moe_transfer_ready_) {
        if (event != nullptr) {
            (void)cudaEventDestroy(event);
            event = nullptr;
        }
    }
    if (moe_ids_ready_ != nullptr) {
        (void)cudaEventDestroy(moe_ids_ready_);
        moe_ids_ready_ = nullptr;
    }
    if (moe_route_ready_ != nullptr) {
        (void)cudaEventDestroy(moe_route_ready_);
        moe_route_ready_ = nullptr;
    }
}

void Program::reset() {
    reset_ = false;
    state_.reset(stream_);
    CUDA_CHECK(cudaMemsetAsync(controls_.p, 0, controls_.bytes, stream_));
    CUDA_CHECK(cudaMemsetAsync(ngram_rows_.p, 0xff, ngram_rows_.bytes, stream_));
    CUDA_CHECK(cudaMemsetAsync(ple_embedding_.p, 0, ple_embedding_.bytes, stream_));
    CUDA_CHECK(cudaMemsetAsync(final_hidden_.p, 0, final_hidden_.bytes, stream_));
    CUDA_CHECK(cudaMemsetAsync(logits_.p, 0, logits_.bytes, stream_));
    CUDA_CHECK(cudaMemsetAsync(nll_.p, 0, nll_.bytes, stream_));
    if (gr_trace_.bytes != 0) {
        CUDA_CHECK(cudaMemsetAsync(gr_trace_.p, 0, gr_trace_.bytes, stream_));
    }
    CUDA_CHECK(cudaMemsetAsync(qsa_selected_ids_.p, 0xff, qsa_selected_ids_.bytes, stream_));
    CUDA_CHECK(cudaMemsetAsync(qsa_selected_counts_.p, 0, qsa_selected_counts_.bytes, stream_));
    CUDA_CHECK(cudaMemsetAsync(router_ids_.p, 0xff, router_ids_.bytes, stream_));
    CUDA_CHECK(cudaMemsetAsync(router_weights_.p, 0, router_weights_.bytes, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    ple_history_ = {kPleResetToken, kPleResetToken};
    frontier_ = 0;
    reset_ = true;
}

TokenResultView Program::execute_token(std::int32_t token_id, std::int32_t target_id) {
    if (!reset_) { throw std::logic_error("Qwen4 Program must be reset before execution"); }
    if (token_id < 0 || token_id >= kVocabulary || target_id < 0 ||
        target_id >= kVocabulary) {
        throw std::invalid_argument("Qwen4 Program token/target is outside the vocabulary");
    }
    if (frontier_ < 0 || frontier_ >= kQsaCapacity) {
        throw std::length_error("Qwen4 Program reached its 4096-token frontier capacity");
    }

    TokenExecutionGuard execution_guard(reset_);
    const std::int32_t current = frontier_;
    profile::ScopedRange token_range(profile::Phase::Token,
                                     static_cast<std::uint64_t>(current));
    auto* host = static_cast<std::byte*>(host_controls_.data());
    std::memcpy(host + kTokenOffset, &token_id, sizeof(token_id));
    const std::int32_t valid = 1;
    std::memcpy(host + kValidOffset, &valid, sizeof(valid));
    std::memcpy(host + kAppendOffset, &current, sizeof(current));
    std::memcpy(host + kTargetOffset, &target_id, sizeof(target_id));
    const std::array<std::int32_t, 3> position = {current, current, current};
    std::memcpy(host + kPositionOffset, position.data(), sizeof(position));
    const std::array<std::int32_t, 2> offsets = {0, current + 1};
    std::memcpy(host + kVisibleOffsetsOffset, offsets.data(), sizeof(offsets));
    const ops::NgramRowHostStep ple_step =
        ops::ngram_row_ids_host_step(token_id, ple_history_, ple_ngram_config_);
    ple_history_ = ple_step.new_history;
    std::memcpy(host + kHostPleHistoryOffset, ple_history_.data(), sizeof(ple_history_));
    std::memcpy(host_ngram_rows_.data(), ple_step.row_ids.data(),
                ple_step.row_ids.size() * sizeof(std::int32_t));
    auto* visible = reinterpret_cast<std::int32_t*>(host + kVisibleIdsOffset);
    for (std::int32_t id = 0; id <= current; ++id) { visible[id] = id; }
    const std::size_t control_copy_bytes =
        kVisibleIdsOffset + static_cast<std::size_t>(current + 1) * sizeof(std::int32_t);
    CUDA_CHECK(cudaMemcpyAsync(controls_.p, host_controls_.data(), control_copy_bytes,
                               cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(ngram_rows_.p, host_ngram_rows_.data(), ngram_rows_.bytes,
                               cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(state_.ple_token_history().data,
                               host + kHostPleHistoryOffset, sizeof(ple_history_),
                               cudaMemcpyHostToDevice, stream_));

    Tensor append_id = tensor_at(controls_, kAppendOffset, DType::I32, {1});
    Tensor target = tensor_at(controls_, kTargetOffset, DType::I32, {1});
    Tensor position_ids = tensor_at(controls_, kPositionOffset, DType::I32, {3});
    Tensor visible_offsets = tensor_at(controls_, kVisibleOffsetsOffset, DType::I32, {2});
    Tensor visible_ids = tensor_at(controls_, kVisibleIdsOffset, DType::I32, {current + 1});
    Tensor row_ids(ngram_rows_.p, DType::I32, {ops::kPleHeads, 1, 1});
    Tensor ple_embedding(ple_embedding_.p, DType::BF16,
                         {ops::kPleRowWidth, ops::kPleHeads});
    Tensor final_hidden(final_hidden_.p, DType::BF16, {kHidden});
    Tensor logits(logits_.p, DType::BF16, {kVocabulary});
    Tensor nll(nll_.p, DType::FP32, {1});

    state_.embed_token(model_.view().token_embedding, token_id, stream_);
    std::size_t qsa_index = 0;
    for (std::size_t layer = 0; layer < kLayerCount; ++layer) {
        profile::ScopedRange layer_range(profile::Phase::Layer,
                                         static_cast<std::uint64_t>(layer));
        if (layer == 1) {
            Tensor ple_device_rows = state_.ple_device_rows();
            ops::ple_iq4_nl_decode_rows(ple_device_rows, ple_embedding, stream_);
            WorkspaceArena workspace = state_.workspace();
            Tensor residual = state_.residual();
            Tensor conv_state = state_.ple_conv_state();
            ops::ple_inject(residual, ple_embedding.reshape({kHidden}), model_.view().ple.key,
                            model_.view().ple.value, model_.view().ple.key_norm,
                            model_.view().ple.query_norm, model_.view().ple.conv_norm,
                            model_.view().ple.conv, conv_state, conv_state, residual, workspace,
                            stream_);
        }

        const LayerWeights& weights = model_.view().layers[layer];
        Tensor residual = state_.residual();
        Tensor mixed = state_.mixed_x();
        Tensor block = state_.block_output();
        Tensor write_scale = state_.write_scale();
        {
            WorkspaceArena workspace = state_.workspace();
            ops::gated_residual_read_write(
                residual, weights.attention_gr.norm, weights.attention_gr.down,
                weights.attention_gr.up, weights.attention_gr.inject, mixed, write_scale,
                workspace, stream_);
        }
        if (weights.gdn) {
            const GdnStateView& gdn = *state_.gdn()[layer];
            WorkspaceArena workspace = state_.workspace();
            Tensor conv = gdn.conv;
            Tensor recurrence = gdn.recurrence;
            ops::gated_delta_net_layer(mixed, *weights.gdn, conv, conv, recurrence, recurrence,
                                       block, workspace, stream_);
        } else {
            if (qsa_index >= kQsaLayerCount || !state_.qsa()[layer]) {
                throw std::logic_error("Qwen4 Program QSA schedule changed after validation");
            }
            profile::ScopedRange qsa_range(profile::Phase::Qsa,
                                           static_cast<std::uint64_t>(layer));
            Tensor qsa_workspace(state_.workspace().base(), DType::U8,
                                 {static_cast<std::int32_t>(ops::qsa_verifier_workspace_bytes())});
            ops::qsa_verifier_token(
                mixed, append_id, position_ids, visible_ids, visible_offsets, *weights.qsa,
                *state_.qsa()[layer], qsa_diagnostics_[qsa_index].selected_ids,
                qsa_diagnostics_[qsa_index].selected_count, block, qsa_workspace, stream_);
            ++qsa_index;
        }
        if (layer == 0) {
            // Layer-0 mixer work is already queued. Faulting/copying the mapped rows here lets the
            // host gather overlap that GPU work; the same-stream H2D remains ordered before the
            // layer-1 decode and injection.
            Tensor ple_device_rows = state_.ple_device_rows();
            ops::ple_iq4_nl_stage_rows(model_.view().ple.table, ple_step.row_ids,
                                       state_.ple_host_rows(), state_.ple_host_rows_bytes(),
                                       ple_device_rows, stream_);
        }
        ops::gated_residual_inject(residual, block, write_scale, residual, stream_);
        if (diagnostic_snapshots_ == DiagnosticSnapshots::Enabled) {
            CUDA_CHECK(cudaMemcpyAsync(gr_diagnostics_[layer].attention_residual.data,
                                       residual.data, kResidualBytes, cudaMemcpyDeviceToDevice,
                                       stream_));
        }

        {
            WorkspaceArena workspace = state_.workspace();
            ops::gated_residual_read_write(residual, weights.ffn_gr.norm, weights.ffn_gr.down,
                                           weights.ffn_gr.up, weights.ffn_gr.inject, mixed,
                                           write_scale, workspace, stream_);
        }
        {
            profile::ScopedRange moe_range(profile::Phase::SparseMoe,
                                           static_cast<std::uint64_t>(layer));
            WorkspaceArena workspace = state_.workspace();
            ops::Qwen4SparseMoePipeline pipeline{
                .pinned_stage = state_.expert_host_stage(),
                .pinned_stage_bytes = state_.expert_host_stage_bytes(),
                .device_stage = state_.expert_device_stage(),
                .transfer_stream = transfer_stream_,
                .route_ready = moe_route_ready_,
                .ids_ready = moe_ids_ready_,
                .transfer_ready = {moe_transfer_ready_[0], moe_transfer_ready_[1]},
                .consumer_complete = {moe_consumer_complete_[0], moe_consumer_complete_[1]},
            };
            ops::qwen4_sparse_moe(
                mixed, weights.moe, pipeline, router_diagnostics_[layer].selected_ids,
                router_diagnostics_[layer].selected_weights, block, workspace, stream_);
        }
        ops::gated_residual_inject(residual, block, write_scale, residual, stream_);
        if (diagnostic_snapshots_ == DiagnosticSnapshots::Enabled) {
            CUDA_CHECK(cudaMemcpyAsync(gr_diagnostics_[layer].ffn_residual.data, residual.data,
                                       kResidualBytes, cudaMemcpyDeviceToDevice, stream_));
        }
    }
    if (qsa_index != kQsaLayerCount) {
        throw std::logic_error("Qwen4 Program did not execute all QSA layers");
    }

    {
        WorkspaceArena workspace = state_.workspace();
        ops::gated_residual_read(state_.residual(), model_.view().final_gr.norm,
                                 model_.view().final_gr.down, model_.view().final_gr.up,
                                 final_hidden, workspace, stream_);
    }
    ops::ggml_block_linear(final_hidden, model_.view().output_head, logits, stream_);
    ops::nll_from_logits(logits.reshape({kVocabulary, 1}), target, nll, kVocabulary, stream_);
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    frontier_ = current + 1;
    execution_guard.complete();

    const std::span<const GrDiagnosticView> gr =
        diagnostic_snapshots_ == DiagnosticSnapshots::Enabled
            ? std::span<const GrDiagnosticView>(gr_diagnostics_)
            : std::span<const GrDiagnosticView>();
    return {
        .token_index = current,
        .logits = logits,
        .nll = nll,
        .final_hidden = final_hidden,
        .ple_row_ids = row_ids.reshape({ops::kPleHeads}),
        .qsa = qsa_diagnostics_,
        .routers = router_diagnostics_,
        .gr = gr,
    };
}

} // namespace ninfer::targets::qwen4::verifier
