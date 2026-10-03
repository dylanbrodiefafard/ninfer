#pragma once

#include "targets/qwen3_6_35b_a3b/impl/config.h"
#include "targets/qwen3_6_35b_a3b/impl/load/bindings.h"
#include "ninfer/ops/a8_activation.h"
#include <ninfer/targets/qwen3_6/runtime.h>

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace ninfer::targets::qwen3_6_35b_a3b::detail {

using GraphExecutionProfile = qwen3_6::GraphExecutionProfile;

struct Variant {
    using WeightsProfile                 = detail::WeightsProfile;
    using TextConfig                     = detail::TextConfig;
    using VisionConfig                   = detail::VisionConfig;
    using DFlashConfig                   = detail::DFlashConfig;
    using ModelView                      = detail::RuntimeModelView;
    using FullAttentionProjectionWeights = detail::AttentionProjectionPayload;
    using GdnProjectionWeights           = detail::GdnProjectionPayload;
    using PostMixerWeights               = detail::SparseMoePayload;
    using MtpAttentionProjectionWeights  = detail::AttentionProjectionPayload;
    using MtpPostMixerWeights            = detail::SparseMoePayload;
    using VisionWeights                  = qwen3_6::VisionWeights;
    using GraphExecutionProfile          = detail::GraphExecutionProfile;

    static constexpr float attention_scale                     = kAttentionScale;
    static constexpr float gdn_scale                           = kGdnScale;
    static constexpr std::uint32_t prefill_chunk_alignment     = kPrefillChunkAlignment;
    static constexpr std::uint32_t maximum_mtp_draft_tokens    = kMaximumMtpDraftTokens;
    static constexpr std::uint32_t maximum_dflash_draft_tokens = kMaximumDFlashDraftTokens;
    // Adaptive DFlash stays within the W<=6 small-T verify routes of the 35B geometry.
    static constexpr std::uint32_t maximum_adaptive_dflash_draft_tokens = 5;

    // DFlash v1 drafts without a selector proposal, so no p-less draft temperature applies.
    [[nodiscard]] static constexpr float dflash_p_less_draft_temperature_prior(float) {
        return 0.0f;
    }

    // DFlash v1 has no tree route; the selector temperature is unused.
    static constexpr float dflash_p_less_tree_temperature = 1.0f;

    static constexpr std::uint32_t maximum_context = kNativeContext;
    static constexpr bool supports_dflash          = DFlashConfig::supported;
    static constexpr std::int32_t draft_head_rows  = 131072;

    [[nodiscard]] static constexpr bool supports_dflash_vision(std::string_view, std::string_view) {
        return false;
    }

    [[nodiscard]] static std::vector<GraphExecutionProfile>
    ordinary_graph_profiles(std::uint32_t capacity);
    [[nodiscard]] static std::vector<GraphExecutionProfile>
    mtp_graph_profiles(std::uint32_t capacity, std::uint32_t draft_window);
    [[nodiscard]] static std::vector<GraphExecutionProfile>
    dflash_graph_profiles(std::uint32_t capacity, std::uint32_t draft_window,
                          std::uint32_t batch_size, std::uint32_t verify_width);

    // route_tokens is the C=1 width of a packed verify round (0 when not batched packed
    // verify). The 35B-A3B package is all-A16: every route below is T-family-stable, so the
    // parameter is accepted for the family interface and ignored.
    // Attention input: normalizes residual with norm_weight/eps (hidden is caller-owned BF16
    // scratch that a route without a BF16 consumer may leave unwritten), then projects.
    static void attention_projection(const Tensor& residual, const Tensor& norm_weight, float eps,
                                     Tensor& hidden, const FullAttentionProjectionWeights& weights,
                                     Tensor& query, Tensor& gate, Tensor& key, Tensor& value,
                                     qwen3_6::TextPhase phase, WorkspaceArena& workspace,
                                     cudaStream_t stream, std::int32_t route_tokens = 0);
    // Attention output: residual += W * (attention * sigmoid(gate)). `attention` is caller-owned
    // scratch that may be overwritten with the gated values.
    static void attention_output_projection(const Tensor& gate, Tensor& attention,
                                            const Weight& weight, Tensor& residual,
                                            qwen3_6::TextPhase phase, WorkspaceArena& workspace,
                                            cudaStream_t stream, std::int32_t route_tokens = 0);
    static void mtp_attention_projection(const Tensor& hidden,
                                         const MtpAttentionProjectionWeights& weights,
                                         Tensor& query, Tensor& gate, Tensor& key, Tensor& value,
                                         WorkspaceArena& workspace, cudaStream_t stream);
    static void mtp_kv_projection(const Tensor& hidden,
                                  const MtpAttentionProjectionWeights& weights, Tensor& key,
                                  Tensor& value, WorkspaceArena& workspace, cudaStream_t stream);
    static void mtp_q_gate_projection(const Tensor& hidden,
                                      const MtpAttentionProjectionWeights& weights, Tensor& query,
                                      Tensor& gate, WorkspaceArena& workspace, cudaStream_t stream);
    static void mtp_fc(const Tensor& embedding_norm, const Tensor& hidden_norm,
                       const Weight& weight, Tensor& residual, WorkspaceArena& workspace,
                       cudaStream_t stream, std::int32_t route_tokens = 0);
    static void mtp_attention_output(const Tensor& attention, const Weight& weight,
                                     Tensor& residual, WorkspaceArena& workspace,
                                     cudaStream_t stream, std::int32_t route_tokens = 0);
    static void gdn_input_projection(const Tensor& hidden, const GdnProjectionWeights& weights,
                                     Tensor& qkv, Tensor& output_gate, qwen3_6::TextPhase phase,
                                     WorkspaceArena& workspace, cudaStream_t stream);
    static void
    gdn_input_projection_snapshot(const Tensor& hidden, const GdnProjectionWeights& weights,
                                  const Tensor& conv_weight, Tensor& conv_states,
                                  const Tensor& valid_columns, const Tensor& initial_slot,
                                  const Tensor& snapshot_base_slot, Tensor& query, Tensor& key,
                                  Tensor& value, Tensor& output_gate, qwen3_6::TextPhase phase,
                                  WorkspaceArena& workspace, cudaStream_t stream);
    // hidden_activation is null or the A8 activation published by gdn_norm_control_projection
    // for this hidden; the leaf consumes it exactly when it would quantize hidden itself.
    static void gdn_input_projection_record(
        const Tensor& hidden, const ops::A8Activation* hidden_activation,
        const GdnProjectionWeights& weights, const Tensor& conv_weight, const Tensor& conv_states,
        const Tensor& valid_columns, const Tensor& initial_slots, Tensor& conv_record,
        Tensor& query, Tensor& key, Tensor& value, Tensor& output_gate, qwen3_6::TextPhase phase,
        WorkspaceArena& workspace, cudaStream_t stream, const Tensor* parent_index = nullptr);
    // GDN output: residual += W * gated_rmsnorm(output, norm_weight, gate, eps) per head.
    // normalized is caller-owned BF16 scratch that a route without a BF16 consumer may leave
    // unwritten.
    static void gdn_output_projection(const Tensor& output, const Tensor& norm_weight,
                                      const Tensor& gate, float eps, Tensor& normalized,
                                      const Weight& weight, Tensor& residual,
                                      qwen3_6::TextPhase phase, WorkspaceArena& workspace,
                                      cudaStream_t stream, std::int32_t route_tokens = 0);
    // Writes the normalized BF16 hidden and the control g/beta. When hidden_activation is
    // non-null and the record leaf consumes an A8 activation in `phase`, also publishes it there.
    static void gdn_norm_control_projection(const Tensor& residual, const Tensor& norm_weight,
                                            float eps, const GdnProjectionWeights& weights,
                                            Tensor& hidden, ops::A8Activation* hidden_activation,
                                            Tensor& g, Tensor& beta, qwen3_6::TextPhase phase,
                                            WorkspaceArena& workspace, cudaStream_t stream,
                                            std::int32_t route_tokens = 0);
    // Normalize the raw residual for the post-mixer; hidden is caller-owned scratch.
    static void post_mixer(const Tensor& norm_weight, float norm_eps, Tensor& hidden,
                           const PostMixerWeights& weights, Tensor& residual,
                           qwen3_6::TextPhase phase, WorkspaceArena& workspace, cudaStream_t stream,
                           std::int32_t route_tokens = 0);
    static void mtp_post_mixer(const Tensor& hidden, const MtpPostMixerWeights& weights,
                               Tensor& residual, WorkspaceArena& workspace, cudaStream_t stream,
                               std::int32_t route_tokens = 0);

    [[nodiscard]] static std::size_t
    mtp_attention_projection_workspace_capacity_bytes(std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t mtp_kv_projection_workspace_capacity_bytes(std::int32_t first,
                                                                                std::int32_t last);
    [[nodiscard]] static std::size_t
    mtp_q_gate_projection_workspace_capacity_bytes(std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t mtp_fc_workspace_capacity_bytes(std::int32_t first,
                                                                     std::int32_t last);
    [[nodiscard]] static std::size_t
    mtp_attention_output_workspace_capacity_bytes(std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t
    attention_projection_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                  qwen3_6::TextPhase phase, std::int32_t first,
                                                  std::int32_t last);
    [[nodiscard]] static std::size_t
    attention_output_projection_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                         qwen3_6::TextPhase phase,
                                                         std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t
    gdn_input_projection_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                  qwen3_6::TextPhase phase, std::int32_t first,
                                                  std::int32_t last);
    [[nodiscard]] static std::size_t gdn_input_projection_snapshot_workspace_capacity_bytes(
        WeightsProfile weights_profile, qwen3_6::TextPhase phase, std::int32_t batch_size,
        std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t gdn_input_projection_record_workspace_capacity_bytes(
        WeightsProfile weights_profile, qwen3_6::TextPhase phase, std::int32_t batch_size,
        std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t
    gdn_output_projection_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                   qwen3_6::TextPhase phase, std::int32_t first,
                                                   std::int32_t last);
    [[nodiscard]] static std::size_t
    gdn_norm_control_projection_workspace_capacity_bytes(std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t
    post_mixer_workspace_capacity_bytes(WeightsProfile weights_profile, qwen3_6::TextPhase phase,
                                        std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t mtp_post_mixer_workspace_capacity_bytes(std::int32_t first,
                                                                             std::int32_t last);
};

} // namespace ninfer::targets::qwen3_6_35b_a3b::detail
