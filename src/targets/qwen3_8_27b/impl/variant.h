#pragma once

#include "core/cache_warm.h"
#include "ninfer/ops/gdn_replay.h"
#include "ops/r9700/linear/a8q4_small_batch_projection.h"
#include "ops/r9700/linear/fp8_activation.h"
#include "targets/qwen3_8_27b/impl/config.h"
#include "targets/qwen3_8_27b/impl/load/bindings.h"
#include <array>

#include <ninfer/targets/qwen3/decoder_state.h>
#include <ninfer/targets/qwen3/runtime.h>

#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::ops::r9700::linear {
struct FusedSiluA8Q4G64DownArgs;
}

namespace ninfer::targets::qwen3_8_27b::detail {

using GraphExecutionProfile = qwen3::GraphExecutionProfile;

// Compile-time data and the three closed execution leaves supplied to the Qwen3 family runtime.
// It owns no request state, execution phase, graph object, or schedule callback.
struct Variant {
    using WeightsProfile                 = detail::WeightsProfile;
    using TextConfig                     = detail::TextConfig;
    using VisionConfig                   = detail::VisionConfig;
    using DFlashConfig                   = detail::DFlashConfig;
    using ModelView                      = detail::RuntimeModelView;
    using FullAttentionProjectionWeights = detail::FullAttentionProjectionPayload;
    using GdnProjectionWeights           = detail::GdnProjectionPayload;
    using PostMixerWeights               = detail::DensePostMixerPayload;
    using MtpAttentionProjectionWeights  = detail::MtpAttentionPayload;
    using MtpPostMixerWeights            = detail::DensePostMixerPayload;
    using VisionWeights                  = qwen3::VisionWeights;
    using GraphExecutionProfile          = detail::GraphExecutionProfile;

    enum class SelectedLinearRole : std::uint8_t {
        AttentionQueryKey,
        AttentionGateValue,
        GdnQueryKey,
        MlpGateUp,
        AttentionOutput,
        MlpDown,
        GdnOutput,
    };

    // Program-owned bindings for loaded-target-prepared row-scaled-E4M3 Text projections.
    // Borrows per-weight descriptors and the loaded target's shared library context.
    // Explicit activation/matmul storage remains in ProgramImplCore's serialized region;
    // opaque library resources are already resident at the final capacity snapshot.
    class ExecutionState final {
    public:
        ExecutionState(const ModelView& model, DeviceSpan serialized_storage,
                       std::uint32_t prefill_tokens, std::uint32_t maximum_concurrency,
                       std::span<const std::uint32_t> verify_widths);
        ~ExecutionState();

        ExecutionState(const ExecutionState&)            = delete;
        ExecutionState& operator=(const ExecutionState&) = delete;
        ExecutionState(ExecutionState&&)                 = delete;
        ExecutionState& operator=(ExecutionState&&)      = delete;

        [[nodiscard]] bool run(SelectedLinearRole role, std::int32_t text_layer,
                               const Tensor& input, const Weight& weight, Tensor& output,
                               hipStream_t stream);
        // Two row-scaled E4M3 projections of one input sharing a single activation quantization;
        // false unless both roles are FP8 and bound to the same activation storage.
        [[nodiscard]] bool run_shared(SelectedLinearRole first_role, SelectedLinearRole second_role,
                                      std::int32_t text_layer, const Tensor& input,
                                      const Weight& first, Tensor& first_output,
                                      const Weight& second, Tensor& second_output,
                                      hipStream_t stream);
        void linear(const Tensor& input, const Weight& weight, Tensor& output,
                    WorkspaceArena& fallback_workspace, hipStream_t stream);
        // The serialized activation region every projection of this state binds at its base; a
        // producer may publish a Q4 activation image there for the next projection.
        [[nodiscard]] DeviceSpan activation_storage() const noexcept;

        // One LQ4 projection target: rows [0, leading rows) publish to `leading`, the rest to
        // `trailing` (nullptr: every row to `leading`); `accumulate` publishes
        // BF16(output + BF16(projection)); `silu_pair` publishes the SiLU-gated activation of a
        // gate/up-interleaved weight to `leading` [rows / 2, T].
        struct Lq4Target {
            const Weight& weight;
            Tensor& leading;
            Tensor* trailing;
            bool accumulate;
            bool silu_pair = false;
        };

        // LQ4 projections of one per-token E4M3 image; each returns false unless every target
        // weight is LQ4. The image is produced from
        // - the BF16 rows `input` [K, T];
        [[nodiscard]] bool lq4_projections(const Tensor& input, std::span<const Lq4Target> targets,
                                           hipStream_t stream);
        // - RMSNorm (unit offset) of `residual` [5120, T], also published to a non-null
        //   `normalized` [5120, T];
        [[nodiscard]] bool lq4_normalized_projections(const Tensor& residual, const Tensor& norm,
                                                      float eps, std::span<const Lq4Target> targets,
                                                      hipStream_t stream,
                                                      Tensor* normalized = nullptr);
        // - the attention output gate BF16(BF16(attention) * sigmoid(gate)) [6144, T];
        [[nodiscard]] bool lq4_gated_projections(const Tensor& gate, const Tensor& attention_fp32,
                                                 std::span<const Lq4Target> targets,
                                                 hipStream_t stream);
        // - the GDN gated per-head RMSNorm of `recurrent_output` with gate `z` [6144, T].
        [[nodiscard]] bool lq4_gated_rmsnorm_projections(const Tensor& recurrent_output,
                                                         const Tensor& norm, const Tensor& z,
                                                         float eps,
                                                         std::span<const Lq4Target> targets,
                                                         hipStream_t stream);
        // SiLU-gated down projection added to the residual in place.
        // Small verification widths: normalized gate/up plus fused SiLU down as one Op whose
        // prepare also clears the down status (no reset launch). gate_up is BF16 scratch.
        [[nodiscard]] bool normalized_mlp(Tensor& residual, const Tensor& norm, float eps,
                                          const Weight& gate_up_weight, const Weight& down,
                                          Tensor& gate_up, std::int32_t text_layer,
                                          hipStream_t stream);
        void fused_mlp_down(const Tensor& gate_up, const Weight& down, Tensor& residual,
                            hipStream_t stream);
        [[nodiscard]] bool gdn_q4_pair_t1(const Tensor& input, const Weight& weight0,
                                          const Weight& weight1, Tensor& output0, Tensor& output1,
                                          hipStream_t stream);
        [[nodiscard]] bool attention_q4_pair_t1(const Tensor& hidden, const Weight& query_key,
                                                const Weight& gate_value, Tensor& query,
                                                Tensor& key, Tensor& gate, Tensor& value,
                                                hipStream_t stream);
        // One A8 activation quantization shared by both GDN projections (any width the Q4 routes
        // serve).
        [[nodiscard]] bool gdn_q4_pair_shared(const Tensor& input, const Weight& query_key,
                                              const Weight& value_z, Tensor& query_key_output,
                                              Tensor& value_z_output, hipStream_t stream);
        // Ordinary prefill GDN front: the input RMSNorm of `residual` feeds the shared codec
        // directly and is published to `normalized`; the query-key, value and output-gate rows
        // (the value-z matrix split at value_dim) are projected from the same planes.
        // The output-gate projection runs on a private side stream, overlapping the
        // convolution and recurrence; join_gdn_gate orders `stream` after it.
        // Both Q4 attention input projections from one A8G64 quantization of the hidden rows.
        [[nodiscard]] bool attention_q4_shared(const Tensor& hidden, const Weight& query_key,
                                               const Weight& gate_value, Tensor& query_key_output,
                                               Tensor& gate_value_output, hipStream_t stream);

        // Operands of the recorded GDN convolution (projection_conv_record_bf16).
        struct GdnConvRecord {
            const Tensor& conv_weight;
            const Tensor& conv_states;
            const Tensor& valid_columns;
            const Tensor& initial_slots;
            const Tensor* parent_index;
            Tensor& conv_record;
            Tensor& query;
            Tensor& key;
            Tensor& value;
            Tensor& output_gate;
        };

        // Verification-width GDN record front: the input RMSNorm of `residual`, the a/b controls
        // into g/beta and the A8G64 planes from one kernel, then both GDN input projections from
        // those planes and the recorded convolution (fused into the projection epilogue for one
        // sequence of width 5..6). The normalized rows are not materialized.
        // One sequence at T1: the same front kernel, then the T1 GDN pair projection of the
        // prepared planes into query_key [4096] and value_z [12288].
        [[nodiscard]] bool
        gdn_q4_normalized_front_t1(const Tensor& residual, const Tensor& norm, float eps,
                                   const GdnProjectionWeights& weights, Tensor& g, Tensor& beta,
                                   Tensor& query_key_output, Tensor& value_z_output,
                                   std::int32_t text_layer, hipStream_t stream);
        // The front kernel shared by both routes: validates the operands, launches it into the
        // activation region's (T, 5120) planes and records the gated-status handoff.
        [[nodiscard]] bool gdn_q4_front(const Tensor& residual, const Tensor& norm, float eps,
                                        const GdnProjectionWeights& weights, Tensor& g,
                                        Tensor& beta, std::int32_t text_layer, hipStream_t stream,
                                        ops::r9700::linear::A8G64ActivationWorkspace* planes_out,
                                        std::size_t* required_out);
        [[nodiscard]] bool
        gdn_q4_normalized_front_record(const Tensor& residual, const Tensor& norm, float eps,
                                       const GdnProjectionWeights& weights,
                                       const GdnConvRecord& record, Tensor& g, Tensor& beta,
                                       WorkspaceArena& workspace, std::int32_t text_layer,
                                       hipStream_t stream, const ops::GdnLayerFold* fold);
        [[nodiscard]] bool gdn_q4_normalized_prefill(const Tensor& residual, const Tensor& norm,
                                                     float eps, const Weight& query_key,
                                                     const Weight& value_z, Tensor& normalized,
                                                     Tensor& query_key_output, Tensor& value_output,
                                                     Tensor& gate_output, hipStream_t stream);
        void join_gdn_gate(hipStream_t stream);
        [[nodiscard]] bool gdn_q4_pair_c2c4(const Tensor& input, const Weight& query_key,
                                            const Weight& value_z, Tensor& query_key_output,
                                            Tensor& value_z_output, hipStream_t stream);
        [[nodiscard]] bool attention_q4_pair_c2c4(const Tensor& hidden, const Weight& query_key,
                                                  const Weight& gate_value, Tensor& query,
                                                  Tensor& key, Tensor& gate, Tensor& value,
                                                  hipStream_t stream);
        // Small widths (T <= 16) of the FP8 attention input: the input RMSNorm of `residual` feeds
        // the E4M3 codec in one launch, then both projections run as one launch publishing
        // query/key and gate/value directly.
        [[nodiscard]] bool
        attention_fp8_normalized_projection(const Tensor& residual, const Tensor& norm, float eps,
                                            const Weight& query_key, const Weight& gate_value,
                                            Tensor& query, Tensor& key, Tensor& gate, Tensor& value,
                                            std::int32_t text_layer, hipStream_t stream);
        // Small widths of the FP8 attention output: the output gate feeds the codec and the
        // projection epilogue adds the rounded result to the residual.
        [[nodiscard]] bool attention_fp8_gated_output(const Tensor& gate,
                                                      const Tensor& attention_fp32,
                                                      const Weight& weight, Tensor& residual,
                                                      std::int32_t text_layer, hipStream_t stream);
        // Q4 forms at the split small-batch pair widths (T5/T6) and small-batch output widths:
        // the input RMSNorm feeds the A8G64 codec and the pair publishes query/key and
        // gate/value directly; the output gate feeds the codec and the projection is added to
        // the residual in the epilogue.
        [[nodiscard]] bool attention_q4_normalized_split(const Tensor& residual, const Tensor& norm,
                                                         float eps, const Weight& query_key,
                                                         const Weight& gate_value, Tensor& query,
                                                         Tensor& key, Tensor& gate, Tensor& value,
                                                         hipStream_t stream);
        [[nodiscard]] bool attention_q4_gated_output(const Tensor& gate,
                                                     const Tensor& attention_fp32,
                                                     const Weight& weight, Tensor& residual,
                                                     hipStream_t stream);
        [[nodiscard]] bool projected_residual_t1(const Tensor& input, const Weight& weight,
                                                 Tensor& residual, qwen3::TextPhase phase,
                                                 bool ordinary_decode, hipStream_t stream);
        // A8 widths T >= 2 (verification and prefill) of a Q4 projection whose BF16 result is
        // added to the residual in the GEMM epilogue.
        [[nodiscard]] bool projected_residual_batched(const Tensor& input, const Weight& weight,
                                                      Tensor& residual, qwen3::TextPhase phase,
                                                      std::int32_t text_layer, hipStream_t stream);
        [[nodiscard]] bool normalized_linear_t1(const Tensor& input, const Tensor& norm, float eps,
                                                const Weight& weight, Tensor& output,
                                                qwen3::TextPhase phase, bool ordinary_decode,
                                                std::int32_t text_layer, hipStream_t stream);

        // These local Q4 boundaries do not depend on neighboring matrix formats. Keep explicit
        // ordinary-decode intent: speculative target verification can also have width one.
        [[nodiscard]] static constexpr bool
        normalized_linear_t1_selected(std::uint32_t activation_bits, qwen3::TextPhase phase,
                                      bool ordinary_decode, std::int32_t text_layer,
                                      std::uint32_t tokens, std::uint32_t rows,
                                      std::uint32_t columns, QType weight) noexcept {
            return activation_bits == 8U && ordinary_decode && phase == qwen3::TextPhase::Verify &&
                   text_layer >= 0 && text_layer < TextConfig::layers && tokens == 1U &&
                   rows == 2U * TextConfig::intermediate && columns == TextConfig::hidden &&
                   weight == QType::Q4G64_F16S;
        }

        // GDN output projection at A8 small-batch widths (T1..8) and full prefill chunks: the
        // gated RMSNorm feeds the codec directly and the GEMM epilogue adds the projection to the
        // residual.
        [[nodiscard]] bool gated_normalized_output(const Tensor& recurrent_output,
                                                   const Tensor& norm, const Tensor& gate,
                                                   float eps, const Weight& weight,
                                                   Tensor& residual, std::int32_t text_layer,
                                                   hipStream_t stream);
        // A8 widths T >= 2 (verification and prefill) of the same gate/up boundary normalize
        // straight into the codec.
        [[nodiscard]] bool normalized_linear_batched(const Tensor& input, const Tensor& norm,
                                                     float eps, const Weight& weight,
                                                     Tensor& output, std::int32_t text_layer,
                                                     hipStream_t stream);

        [[nodiscard]] static constexpr bool
        projected_residual_t1_selected(std::uint32_t activation_bits, qwen3::TextPhase phase,
                                       bool ordinary_decode, std::uint32_t tokens,
                                       std::uint32_t rows, std::uint32_t columns,
                                       QType weight) noexcept {
            return activation_bits == 8U && phase == qwen3::TextPhase::Verify && ordinary_decode &&
                   tokens == 1U && rows == TextConfig::hidden &&
                   (columns == TextConfig::query_size || columns == TextConfig::intermediate) &&
                   weight == QType::Q4G64_F16S;
        }

        [[nodiscard]] static constexpr bool
        attention_q4_pair_t1_selected(std::uint32_t activation_bits, std::uint32_t tokens,
                                      QType query_key, QType gate_value) noexcept {
            return activation_bits == 8U && tokens == 1U && query_key == QType::Q4G64_F16S &&
                   gate_value == QType::Q4G64_F16S;
        }

        [[nodiscard]] static constexpr bool gdn_q4_pair_t1_selected(std::uint32_t activation_bits,
                                                                    std::uint32_t tokens,
                                                                    QType weight0,
                                                                    QType weight1) noexcept {
            return activation_bits == 8U && tokens == 1U && weight0 == QType::Q4G64_F16S &&
                   weight1 == QType::Q4G64_F16S;
        }

        [[nodiscard]] static constexpr bool q4_pair_c2c4_selected(std::uint32_t activation_bits,
                                                                  std::uint32_t tokens,
                                                                  QType weight0,
                                                                  QType weight1) noexcept {
            return activation_bits == 8U && tokens >= 2U && tokens <= 4U &&
                   weight0 == QType::Q4G64_F16S && weight1 == QType::Q4G64_F16S;
        }

        [[nodiscard]] static constexpr bool
        fused_mlp_down_selected(std::uint32_t activation_bits, QType down, std::uint32_t tokens,
                                std::int32_t text_layer) noexcept {
            // The fused Op consumes represented BF16 gate/up values, independently
            // of the producer's weight or activation precision.
            // The full prefill chunk (M128) and every small-batch N5120/K17408 width own the
            // in-place residual epilogue.
            return activation_bits == 8U && down == QType::Q4G64_F16S &&
                   (tokens == 2048U || ops::r9700::linear::detail::use_a8q4_small_batch_projection(
                                           tokens, TextConfig::hidden, TextConfig::intermediate,
                                           TextConfig::intermediate)) &&
                   text_layer >= 0 && text_layer < TextConfig::layers;
        }

        [[nodiscard]] std::size_t selected_count() const noexcept;
        // Complete set of token widths that graph-captured selected projections can observe,
        // plus the fixed full prefill chunk prepared before graph definition begins.
        [[nodiscard]] static std::vector<std::uint32_t>
        eager_widths(std::uint32_t prefill_tokens, std::uint32_t maximum_concurrency,
                     std::span<const std::uint32_t> verify_widths);

        // Whether every target runs on the LQ4 projection route (LQ4 weights, or
        // row-scaled E4M3 protections at prefill widths) over a [columns, tokens] image.
        [[nodiscard]] bool lq4_targets_supported(std::uint32_t tokens, std::uint32_t columns,
                                                 std::span<const Lq4Target> targets) const noexcept;

    private:
        struct Impl;
        [[nodiscard]] ops::r9700::linear::FusedSiluA8Q4G64DownArgs
        fused_down_args(const Tensor& gate_up, const Weight& down, Tensor& residual) const;
        // LQ4 GDN front: residual RMSNorm, a/b controls into g/beta and the per-token E4M3
        // image of the seam (bound in the serialized region) from one kernel; false unless both GDN
        // input projections are LQ4. A non-null `fold` (that layer's deferred replay fold)
        // runs in the same launch.
        [[nodiscard]] bool gdn_lq4_front(const Tensor& residual, const Tensor& norm, float eps,
                                         const GdnProjectionWeights& weights, Tensor& g,
                                         Tensor& beta, hipStream_t stream,
                                         ops::r9700::linear::Fp8ActivationWorkspace* image,
                                         const ops::GdnLayerFold* fold = nullptr);
        [[nodiscard]] ops::r9700::linear::Fp8ActivationWorkspace
        lq4_image(std::uint32_t tokens, std::uint32_t columns) const;
        void lq4_project(const ops::r9700::linear::Fp8ActivationWorkspace& image,
                         std::span<const Lq4Target> targets, hipStream_t stream) const;
        [[nodiscard]] std::optional<ops::r9700::linear::Fp8ActivationWorkspace>
        fp8_small_activation(SelectedLinearRole role, std::int32_t text_layer, const Weight& weight,
                             std::uint32_t tokens) const;
        std::unique_ptr<Impl> impl_;
    };

    static constexpr float attention_scale                     = kAttentionScale;
    static constexpr float gdn_scale                           = kGdnScale;
    static constexpr std::uint32_t prefill_chunk_alignment     = kPrefillChunkAlignment;
    static constexpr std::uint32_t maximum_mtp_draft_tokens    = kMaximumMtpDraftTokens;
    static constexpr std::uint32_t maximum_dflash_draft_tokens = kMaximumDFlashDraftTokens;
    static constexpr std::uint32_t maximum_context             = kNativeContext;
    static constexpr bool supports_dflash                      = DFlashConfig::supported;
    static constexpr std::int32_t draft_head_rows              = 131072;

    // Cold-start DFlash2 p-less proposal temperature for a p-less target temperature T, used
    // until the online proposal calibration has observed rounds at T. Upstream offline replay of
    // p-less selector dumps put the effective optimum near 0.3-0.45 at T=1.5 and 0.6-0.8 at T=2,
    // and the R9700 measured 0.4 best at T=1.5; 0.8*(T-1) passes through both, clamped to the
    // calibration grid.
    [[nodiscard]] static constexpr float dflash_p_less_draft_temperature_prior(float temperature) {
        const float prior = 0.8f * (temperature - 1.0f);
        return prior < 0.2f ? 0.2f : (prior > 1.25f ? 1.25f : prior);
    }

    // The target-verify leaves receive route_tokens: the C=1 width of a packed verify round (0
    // when the round is not batched packed verify). It selects the C=1 quantization family for
    // the single aggregate launch; see the pinned policy in variant.cpp.
    static void attention_projection(const Tensor& hidden,
                                     const FullAttentionProjectionWeights& weights, Tensor& query,
                                     Tensor& gate, Tensor& key, Tensor& value,
                                     qwen3::TextPhase phase, WorkspaceArena& workspace,
                                     hipStream_t stream, std::int32_t route_tokens = 0,
                                     ExecutionState* execution = nullptr,
                                     std::int32_t text_layer   = -1);
    // Fused small-width FP8 attention boundaries. False when not selected: the caller then runs
    // the input RMSNorm and attention_projection, or the output gate and
    // attention_output_projection.
    [[nodiscard]] static bool
    attention_normalized_projection(const Tensor& residual, const Tensor& norm, float eps,
                                    const FullAttentionProjectionWeights& weights, Tensor& query,
                                    Tensor& gate, Tensor& key, Tensor& value, hipStream_t stream,
                                    ExecutionState* execution, std::int32_t text_layer);
    [[nodiscard]] static bool
    attention_gated_output_projection(const Tensor& gate, const Tensor& attention_fp32,
                                      const Weight& weight, Tensor& residual, hipStream_t stream,
                                      ExecutionState* execution, std::int32_t text_layer);
    // The family schedule owns append/publication ordering and supplies the resulting typed read
    // capability. This target leaf owns the fixed Qwen3.8 full-attention geometry and R9700
    // implementation; it cannot manufacture a cache view or publish a frontier.
    static void full_attention(const Tensor& normalized_query,
                               const qwen3::PagedKVLayerRead& cache_read,
                               const Tensor& cache_positions, Tensor& attention_fp32,
                               WorkspaceArena& workspace, hipStream_t stream,
                               const Tensor* ancestor_masks    = nullptr,
                               const Tensor* prefix_lengths    = nullptr,
                               const Tensor* active_query_rows = nullptr);
    [[nodiscard]] static std::size_t
    full_attention_workspace_capacity_bytes(std::int32_t maximum_query_rows,
                                            std::uint32_t maximum_visible_context,
                                            bool tree_or_device_count);

    // One causal sequence of a compact decode batch: its query/position/output panels and the
    // read capability its append transaction produced.
    struct FullAttentionSequence {
        Tensor query;
        qwen3::PagedKVLayerRead cache_read;
        Tensor cache_positions;
        Tensor output;
    };

    // Attends every sequence of the batch in one launch when all of them take the batched route,
    // with each sequence's output bytes equal to its full_attention call. False, launching
    // nothing, when some sequence is outside that route; the caller then attends one by one.
    [[nodiscard]] static bool
    full_attention_sequences(std::span<const FullAttentionSequence> sequences,
                             WorkspaceArena& workspace, hipStream_t stream);
    [[nodiscard]] static std::size_t
    full_attention_sequences_workspace_capacity_bytes(std::int32_t sequences,
                                                      std::int32_t maximum_query_rows,
                                                      std::uint32_t maximum_visible_context);
#if defined(NINFER_R9700_XATTENTION_QUALIFICATION)
    // Compile-isolated model-gate leaf. The ordinary target library has neither this declaration
    // nor its sparse Op object, so no runtime selector or disabled-path branch reaches production.
    static void text_prefill_attention(const Tensor& normalized_query,
                                       const qwen3::PagedKVLayerRead& cache_read,
                                       const Tensor& cache_positions, Tensor& attention_fp32,
                                       WorkspaceArena& workspace, hipStream_t stream);
    [[nodiscard]] static std::size_t
    text_prefill_attention_workspace_capacity_bytes(std::int32_t maximum_query_rows,
                                                    std::uint32_t maximum_visible_context);
#endif
    static void attention_output_projection(const Tensor& attention, const Weight& weight,
                                            Tensor& residual, qwen3::TextPhase phase,
                                            WorkspaceArena& workspace, hipStream_t stream,
                                            std::int32_t text_layer,
                                            ExecutionState* execution = nullptr,
                                            bool ordinary_decode      = false);
    static void mtp_attention_projection(const Tensor& hidden,
                                         const MtpAttentionProjectionWeights& weights,
                                         Tensor& query, Tensor& gate, Tensor& key, Tensor& value,
                                         WorkspaceArena& workspace, hipStream_t stream,
                                         ExecutionState* execution = nullptr);
    static void mtp_kv_projection(const Tensor& hidden,
                                  const MtpAttentionProjectionWeights& weights, Tensor& key,
                                  Tensor& value, WorkspaceArena& workspace, hipStream_t stream,
                                  ExecutionState* execution = nullptr);
    static void mtp_q_gate_projection(const Tensor& hidden,
                                      const MtpAttentionProjectionWeights& weights, Tensor& query,
                                      Tensor& gate, WorkspaceArena& workspace, hipStream_t stream,
                                      ExecutionState* execution = nullptr);
    static void mtp_fc(const Tensor& embedding_norm, const Tensor& hidden_norm,
                       const Weight& weight, Tensor& residual, WorkspaceArena& workspace,
                       hipStream_t stream, std::int32_t route_tokens = 0,
                       ExecutionState* execution = nullptr);
    static void mtp_attention_output(const Tensor& attention, const Weight& weight,
                                     Tensor& residual, WorkspaceArena& workspace,
                                     hipStream_t stream, std::int32_t route_tokens = 0,
                                     ExecutionState* execution = nullptr);
    static void gdn_input_projection(const Tensor& hidden, const GdnProjectionWeights& weights,
                                     Tensor& qkv, Tensor& output_gate, qwen3::TextPhase phase,
                                     WorkspaceArena& workspace, hipStream_t stream,
                                     ExecutionState* execution = nullptr,
                                     std::int32_t text_layer   = -1);

    [[nodiscard]] static constexpr bool
    gdn_input_projection_prefill_selected(qwen3::TextPhase phase) noexcept {
        return phase == qwen3::TextPhase::Prefill;
    }

    // Ordinary one-sequence prefill (any width) owns the whole GDN front: the input RMSNorm of
    // `residual` (published to `hidden`), the a/b control projection into g/beta, the
    // query-key/value-z projections and the convolution scatter. The A8 route normalizes straight
    // into the shared codec.
    static void gdn_input_projection_prefill(
        const Tensor& residual, const Tensor& norm_weight, float eps,
        const GdnProjectionWeights& weights, const Tensor& conv_weight, Tensor& conv_state,
        Tensor& hidden, Tensor& g, Tensor& beta, Tensor& query, Tensor& key, Tensor& value,
        Tensor& output_gate, qwen3::TextPhase phase, WorkspaceArena& workspace, hipStream_t stream,
        ExecutionState* execution = nullptr, std::int32_t text_layer = -1);
    static void gdn_input_projection_snapshot(
        const Tensor& hidden, const GdnProjectionWeights& weights, const Tensor& conv_weight,
        Tensor& conv_states, const Tensor& valid_columns, const Tensor& initial_slots,
        const Tensor& snapshot_base_slots, Tensor& query, Tensor& key, Tensor& value,
        Tensor& output_gate, qwen3::TextPhase phase, WorkspaceArena& workspace, hipStream_t stream,
        ExecutionState* execution = nullptr, std::int32_t text_layer = -1);
    static void gdn_input_projection_record(
        const Tensor& hidden, const GdnProjectionWeights& weights, const Tensor& conv_weight,
        const Tensor& conv_states, const Tensor& valid_columns, const Tensor& initial_slots,
        Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value, Tensor& output_gate,
        qwen3::TextPhase phase, WorkspaceArena& workspace, hipStream_t stream,
        const Tensor* parent_index = nullptr, ExecutionState* execution = nullptr,
        std::int32_t text_layer = -1);
    // Verification record front: owns the GDN input RMSNorm of `residual` [hidden,width,batch],
    // the a/b controls into g/beta, the query-key/value-z projections and the convolution record
    // (gdn_input_projection_record's outputs). `hidden` [hidden,width,batch] is scratch for the
    // unfused composition; the fused A8 route does not materialize it. A non-null `fold` (this
    // layer's deferred replay fold of the previous round) is applied before the convolution
    // record reads the layer's state, fused into the FP8 front when that route runs.
    static void gdn_front_record(const Tensor& residual, const Tensor& norm_weight, float eps,
                                 const GdnProjectionWeights& weights, const Tensor& conv_weight,
                                 const Tensor& conv_states, const Tensor& valid_columns,
                                 const Tensor& initial_slots, Tensor& hidden, Tensor& g,
                                 Tensor& beta, Tensor& conv_record, Tensor& query, Tensor& key,
                                 Tensor& value, Tensor& output_gate, qwen3::TextPhase phase,
                                 WorkspaceArena& workspace, hipStream_t stream,
                                 const Tensor* parent_index = nullptr,
                                 ExecutionState* execution = nullptr, std::int32_t text_layer = -1,
                                 const ops::GdnLayerFold* fold = nullptr);
    // Mixed front: one prefill owner's leading `prefill_columns` columns of `residual`
    // [hidden,T] followed by a record-verify batch of `conv_record`'s [conv,width,batch] columns.
    // Normalization, a/b controls and the input projections run once over all T columns; the
    // owner convolves through its single `prefill_conv_state` and the batch through the record
    // path. query/key/value/output_gate are the aggregate [rows,T] outputs.
    static void gdn_front_mixed(const Tensor& residual, const Tensor& norm_weight, float eps,
                                const GdnProjectionWeights& weights, const Tensor& conv_weight,
                                Tensor& prefill_conv_state, const Tensor& conv_states,
                                const Tensor& valid_columns, const Tensor& initial_slots,
                                std::int32_t prefill_columns, Tensor& hidden, Tensor& g,
                                Tensor& beta, Tensor& conv_record, Tensor& query, Tensor& key,
                                Tensor& value, Tensor& output_gate, WorkspaceArena& workspace,
                                hipStream_t stream, ExecutionState* execution,
                                std::int32_t text_layer);
    // Snapshot front for one T1 sequence: owns the GDN input RMSNorm of `residual`, the a/b
    // controls into g/beta, the input projections and the snapshot convolution
    // (gdn_input_projection_snapshot's outputs); `hidden` is scratch for the unfused composition.
    static void gdn_front_snapshot(const Tensor& residual, const Tensor& norm_weight, float eps,
                                   const GdnProjectionWeights& weights, const Tensor& conv_weight,
                                   Tensor& conv_states, const Tensor& valid_columns,
                                   const Tensor& initial_slots, const Tensor& snapshot_base_slots,
                                   Tensor& hidden, Tensor& g, Tensor& beta, Tensor& query,
                                   Tensor& key, Tensor& value, Tensor& output_gate,
                                   qwen3::TextPhase phase, WorkspaceArena& workspace,
                                   hipStream_t stream, ExecutionState* execution = nullptr,
                                   std::int32_t text_layer = -1);
    // The part of the GDN output projection's stream the recurrence warms ahead of it, past the
    // head its activation producer warms (empty at prefill widths).
    [[nodiscard]] static CacheWarm gdn_recurrence_warm(const Weight& output_projection,
                                                       std::int32_t tokens);
    // The family provides the gated-RMSNorm parameters and the BF16 [128,48,T] normalized
    // scratch; the leaf owns whether the normalized output is materialized (always when
    // `materialize_normalized`) or fused into the output projection's activation codec.
    static void gdn_output_projection(const Tensor& recurrent_output, const Tensor& norm,
                                      const Tensor& gate, float eps, Tensor& normalized,
                                      bool materialize_normalized, const Weight& weight,
                                      Tensor& residual, qwen3::TextPhase phase,
                                      WorkspaceArena& workspace, hipStream_t stream,
                                      std::int32_t route_tokens = 0,
                                      ExecutionState* execution = nullptr,
                                      std::int32_t text_layer = -1, bool ordinary_decode = false);
    static void gdn_norm_control_projection(const Tensor& residual, const Tensor& norm_weight,
                                            float eps, const GdnProjectionWeights& weights,
                                            Tensor& hidden, Tensor& g, Tensor& beta,
                                            hipStream_t stream);
    // The family provides the normalization parameters and BF16 scratch; the leaf owns whether
    // normalization is materialized or fused into the gate/up projection. ordinary_decode is
    // supplied only by the family ordinary-decode entry point, never inferred from Verify/T1.
    static void post_mixer(const Tensor& norm, float eps, const Tensor& hidden,
                           const PostMixerWeights& weights, Tensor& residual,
                           qwen3::TextPhase phase, WorkspaceArena& workspace, hipStream_t stream,
                           ExecutionState* execution = nullptr, std::int32_t text_layer = -1,
                           bool ordinary_decode = false);
    static void mtp_post_mixer(const Tensor& hidden, const MtpPostMixerWeights& weights,
                               Tensor& residual, WorkspaceArena& workspace, hipStream_t stream,
                               std::int32_t route_tokens = 0, ExecutionState* execution = nullptr);
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
    attention_projection_workspace_capacity_bytes(WeightsProfile profile, qwen3::TextPhase phase,
                                                  std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t attention_output_projection_workspace_capacity_bytes(
        WeightsProfile profile, qwen3::TextPhase phase, std::int32_t first, std::int32_t last);

    // At <=32 aggregate tokens the front already hosts the deferred fold. Above that
    // boundary, fold/record fusion removes a recurrent-state read on batched DFlash chains.
    [[nodiscard]] static constexpr bool gdn_record_fold_selected(std::int32_t tokens) noexcept {
        return tokens > 32;
    }

    [[nodiscard]] static std::size_t
    gdn_input_projection_workspace_capacity_bytes(WeightsProfile profile, qwen3::TextPhase phase,
                                                  std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t gdn_input_projection_snapshot_workspace_capacity_bytes(
        WeightsProfile profile, qwen3::TextPhase phase, std::int32_t batch, std::int32_t first,
        std::int32_t last);
    [[nodiscard]] static std::size_t
    gdn_input_projection_record_workspace_capacity_bytes(WeightsProfile profile,
                                                         qwen3::TextPhase phase, std::int32_t batch,
                                                         std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t
    gdn_output_projection_workspace_capacity_bytes(WeightsProfile profile, qwen3::TextPhase phase,
                                                   std::int32_t first, std::int32_t last);
    [[nodiscard]] static std::size_t post_mixer_workspace_capacity_bytes(WeightsProfile profile,
                                                                         qwen3::TextPhase phase,
                                                                         std::int32_t first,
                                                                         std::int32_t last);
    [[nodiscard]] static std::size_t mtp_post_mixer_workspace_capacity_bytes(std::int32_t first,
                                                                             std::int32_t last);
    // Maximum caller-owned compile-selected activation scratch needed by any Q4 or W8 matrix in
    // the selected profile. The family schedule adds this reserve to every phase whose live
    // workspace may contain a candidate linear call.
    [[nodiscard]] static QType dflash_matrix_qtype(WeightsProfile profile);
    // Encoding of the pinned-host token embedding the profile binds; sizes prompt staging.
    [[nodiscard]] static QType token_embedding_qtype(WeightsProfile profile);
    [[nodiscard]] static std::size_t linear_workspace_capacity_bytes(WeightsProfile profile,
                                                                     std::int32_t tokens);
    [[nodiscard]] static std::size_t vision_linear_workspace_capacity_bytes(WeightsProfile profile,
                                                                            std::int32_t tokens);

    // One serialized Program-owned activation region replaces the former per-call arena reserve.
    // The hybrid profile also uses its prefix for selected FP8 activation quantization.
    [[nodiscard]] static std::size_t
    execution_state_capacity_bytes(WeightsProfile profile, std::uint32_t prefill_tokens,
                                   std::uint32_t maximum_graph_tokens);

    // gfx1201/ROCm hipMalloc rounds the two Program arenas to at most 2 MiB
    // units. Keep this physical-allocation bound separate from logical arenas
    // and graph allowance; it is independent of KV pages, preserving affinity.
    [[nodiscard]] static constexpr std::size_t
    runtime_allocation_overhead_bound(WeightsProfile profile) noexcept {
        return profile == WeightsProfile::R9700Q4G64Fp8FourRoleN16K16Evaluation ||
                       is_fp8_capped_profile(profile) || is_selective_protected_profile(profile) ||
                       profile == WeightsProfile::R9700Q4G64Fp8FourRoleDFlash2Q4Evaluation ||
                       profile == WeightsProfile::R9700Q4G64Fp8FourRoleDFlash2Q4MseEvaluation ||
                       profile == WeightsProfile::R9700Q4G64Fp8FourRoleDFlash2W8MseEvaluation
                   ? std::size_t{4} << 20U
                   : 0U;
    }

    [[nodiscard]] static std::vector<GraphExecutionProfile>
    ordinary_graph_profiles(std::uint32_t capacity);

    // NInfer-owned runtime overhead beyond logical arenas, HIP graph residency and the
    // physical allocation bound. The C-specific bound includes lazy kernel residency and
    // the measured physical-counter residual; SSD scatter adds its own residency.
    // Keep the 2-MiB startup component separate from the main-stream bound.
    [[nodiscard]] static constexpr std::size_t
    runtime_driver_reservation_bytes(WeightsProfile profile, std::uint32_t concurrency,
                                     bool disk_cache_enabled) noexcept {
        if (!is_lq4_text_profile(profile) || concurrency == 0U || concurrency > kMaximumConcurrency)
            return 0U;
        constexpr std::array<std::size_t, kMaximumConcurrency> main_mib{
            6, 6, 4, 4, 2, 4, 4, 6, 32, 32, 32, 34, 34, 34, 34, 34};
        return (2U + main_mib[concurrency - 1U] + (disk_cache_enabled ? 2U : 0U)) << 20U;
    }

    [[nodiscard]] static std::vector<GraphExecutionProfile>
    mtp_graph_profiles(std::uint32_t capacity, std::uint32_t draft_window);
    [[nodiscard]] static std::vector<GraphExecutionProfile>
    dflash_graph_profiles(std::uint32_t capacity, std::uint32_t draft_window,
                          std::uint32_t batch_size, std::uint32_t verify_width);

    [[nodiscard]] static std::optional<std::size_t>
    calibrated_dflash_graph_allowance_bytes(WeightsProfile profile, qwen3::StartupFeatures features,
                                            std::uint32_t capacity, std::uint32_t concurrency,
                                            std::uint32_t prefill_chunk, std::uint32_t draft_window,
                                            bool adaptive_draft);
};

} // namespace ninfer::targets::qwen3_8_27b::detail
