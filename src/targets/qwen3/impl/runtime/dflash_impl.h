#include "targets/qwen3/impl/runtime/dflash_candidate_stats.h"
#include "targets/qwen3/impl/runtime/graph_impl.h"
#include "targets/qwen3/impl/runtime/instance.h"
#include "targets/qwen3/impl/runtime/panel_copy.h"
#include "targets/qwen3/impl/runtime/schedule.h"
#include "targets/qwen3/impl/runtime/workspace_recipe.h"

#include "ninfer/ops/dflash2_path_select.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/grouped_dynamic_conv.h"
#include "ninfer/ops/kv_cache_append_prefix.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/normalized_rope_kv_append.h"
#include "ninfer/ops/position.h"
#include "ninfer/ops/prepare_masked_block.h"
#include "ninfer/ops/prepare_ragged_prefix.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/scalar.h"
#include "ninfer/ops/silu_mul.h"
#include "ninfer/ops/speculative_round.h"
#include "ninfer/ops/swa.h"

#include <hip/hip_runtime.h>

#include <array>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <utility>

namespace ninfer::targets::qwen3::detail::NINFER_QWEN3_RUNTIME_NS::schedule {
// NOLINTNEXTLINE(misc-anonymous-namespace-in-header): single-TU fragment (runtime.hip)
namespace {

void require_dflash_state(const PrefillContext& state) {
    if (state.dflash == nullptr || !state.execution.model.dflash.has_value()) {
        throw std::logic_error("DFlash schedule requires DFlash weights and state");
    }
}

DFlashPersistentState& dflash_state(PrefillContext& state) {
    require_dflash_state(state);
    return *state.dflash;
}

DFlashPersistentState& dflash_state(DFlashBatchContext& state) { return state.dflash; }

DFlashPersistentState& dflash_state(DFlashAppendContext& state) { return state.dflash; }

void serialized_linear(ExecutionCore& execution, const Tensor& input, const Weight& weight,
                       Tensor& output) {
    if (execution.linear_execution != nullptr) {
        execution.linear_execution->linear(input, weight, output, execution.work,
                                           execution.device.stream);
    } else {
        ops::linear(input, weight, output, execution.work, execution.device.stream);
    }
}

// [C,K,B] with C fastest. Copies src hops into dst starting at hop0.
void copy_selector_hops(Tensor& dst, const Tensor& src, std::int32_t hop0, hipStream_t stream) {
    if (src.ne[0] != dst.ne[0] || src.ne[2] != dst.ne[2] || hop0 < 0 ||
        hop0 + src.ne[1] > dst.ne[1]) {
        throw std::logic_error("DFlash selector copy shape mismatch");
    }
    if (dst.data == src.data && hop0 == 0) { return; }
    const std::size_t elem = dtype_size(src.dtype);
    auto* dst_bytes        = static_cast<std::byte*>(dst.data) +
                             static_cast<std::size_t>(hop0) * static_cast<std::size_t>(dst.nb[1]);
    HIP_CHECK(hipMemcpy2DAsync(
        dst_bytes, static_cast<std::size_t>(dst.nb[2]), src.data,
        static_cast<std::size_t>(src.nb[2]),
        static_cast<std::size_t>(src.ne[0]) * static_cast<std::size_t>(src.ne[1]) * elem,
        static_cast<std::size_t>(src.ne[2]), hipMemcpyDeviceToDevice, stream));
}

template <class V>
DFlashFeatureSink prefill_feature_sink_impl(PrefillContext& state,
                                            DFlashFeatureSink::PrefillConsumer consume_prefill) {
    if constexpr (!V::supports_dflash) {
        throw std::logic_error("DFlash feature capture is unavailable for this target");
    } else {
        require_dflash_state(state);
        using Config = typename V::DFlashConfig;
        return DFlashFeatureSink{
            .features        = &dflash_state(state).prefill_features,
            .positions       = &dflash_state(state).prefill_positions,
            .layers          = std::span<const int>(Config::target_feature_layers),
            .consume_prefill = std::move(consume_prefill),
        };
    }
}

template <class V>
DFlashFeatureSink batch_feature_sink_impl(DFlashBatchContext& state, const Tensor& lanes,
                                          const Tensor& valid_columns, std::int32_t width,
                                          std::int32_t batch_size) {
    if constexpr (!V::supports_dflash) {
        throw std::logic_error("DFlash feature capture is unavailable for this target");
    } else {
        using Config = typename V::DFlashConfig;
        return DFlashFeatureSink{
            .batch_features      = &dflash_state(state).pending_features,
            .batch_lanes         = &lanes,
            .batch_valid_columns = &valid_columns,
            .batch_width         = width,
            .batch_size          = batch_size,
            .layers              = std::span<const int>(Config::target_feature_layers),
        };
    }
}

template <class V, class Context>
void append_context_impl(Context& state, const Tensor& features, const Tensor& positions,
                         const Tensor& commit_counts, const Tensor& lanes, const Tensor& table_rows,
                         ops::KVCacheAppendPrefixExecutionEnvelope envelope) {
    if constexpr (!V::supports_dflash) {
        throw std::logic_error("DFlash context append is unavailable for this target");
    } else {
        using Config               = typename V::DFlashConfig;
        const std::int32_t width   = features.ne[1];
        const std::int32_t batch   = features.ne[2];
        const std::int32_t columns = width * batch;
        if (width <= 0 || batch <= 0 || features.dtype != DType::BF16 ||
            features.ne[0] != Config::feature_rows || features.ne[3] != 1 ||
            positions.dtype != DType::I32 || positions.ne[0] != width || positions.ne[1] != batch ||
            commit_counts.dtype != DType::I32 || commit_counts.ne[0] != batch ||
            lanes.dtype != DType::I32 || lanes.ne[0] != batch || table_rows.dtype != DType::I32 ||
            table_rows.ne[0] != batch) {
            throw std::invalid_argument("DFlash context append inputs are invalid");
        }
        const bool replace_local_window = batch == 1 && width > Config::local_capacity;
        if (replace_local_window && (envelope.min_count != static_cast<std::uint32_t>(width) ||
                                     envelope.max_count != static_cast<std::uint32_t>(width))) {
            throw std::invalid_argument(
                "DFlash oversized local append requires an exact full-prefix commit");
        }
        const int local_offset = replace_local_window ? width - Config::local_capacity : 0;
        const int local_width  = replace_local_window ? Config::local_capacity : width;
        const ops::KVCacheAppendPrefixExecutionEnvelope local_envelope{
            replace_local_window ? static_cast<std::uint32_t>(Config::local_capacity)
                                 : envelope.min_count,
            replace_local_window ? static_cast<std::uint32_t>(Config::local_capacity)
                                 : envelope.max_count,
        };
        Tensor local_counts = commit_counts;
        if (replace_local_window) {
            if (!state.execution.io.dflash_prefill) {
                throw std::logic_error("DFlash prefill count storage is unavailable");
            }
            local_counts = state.execution.io.dflash_prefill->produced_count;
            ops::set_i32_scalar(local_counts, Config::local_capacity,
                                state.execution.device.stream);
        }

        const auto context_roots =
            workspace_recipe::dflash_context<Config>(state.execution.work, columns);
        Tensor projected = context_roots.projected;
        serialized_linear(state.execution, features.view({Config::feature_rows, columns}),
                          state.execution.model.dflash.value().feature_projection, projected);
        Tensor context = context_roots.normalized;
        ops::rmsnorm(projected, state.execution.model.dflash.value().context_norm,
                     Config::rms_epsilon, false, context, state.execution.device.stream);

        static_assert(Config::local_layers == Config::layers,
                      "the sole DFlash2 route stores every layer in the cyclic local cache");
        static_assert(Config::head_dim == 128 && Config::query_heads == 32 &&
                          Config::kv_heads == 8 && Config::rope_theta == 1.0e7F,
                      "dflash_qkv_norm_rope serves the DFlash2 attention geometry");
        const int layer_width   = local_width;
        const int layer_columns = layer_width * batch;
        Tensor layer_context =
            replace_local_window ? context.slice(1, local_offset, local_width) : context;
        Tensor layer_positions =
            replace_local_window ? positions.slice(0, local_offset, local_width) : positions;
        // Every layer projects the same context: with Q4 projections it is quantized once into the
        // serialized activation region, which nothing else writes before the last projection.
        const auto& layers      = state.execution.model.dflash.value().layers;
        const DeviceSpan region = state.execution.linear_execution != nullptr
                                      ? state.execution.linear_execution->activation_storage()
                                      : DeviceSpan{};
        bool shared_image =
            region.data != nullptr &&
            region.bytes >= ops::q4_activation_image_bytes(layer_columns, Config::hidden);
        for (const auto& weight : layers) {
            shared_image = shared_image && weight.context_key_value.qtype == QType::Q4G64_F16S;
        }
        if (shared_image) {
            ops::quantize_q4_activation_image(layer_context, region, state.execution.device.stream);
        }
        for (int layer = 0; layer < Config::layers; ++layer) {
            auto layer_scope   = state.execution.work.scope();
            const auto& weight = layers.at(static_cast<std::size_t>(layer));
            auto layer_roots =
                workspace_recipe::dflash_context_layer<Config>(state.execution.work, layer_columns);
            if (shared_image) {
                ops::linear_q4_activation_image(weight.context_key_value, layer_roots.fused_kv,
                                                region, state.execution.device.stream);
            } else {
                serialized_linear(state.execution, layer_context, weight.context_key_value,
                                  layer_roots.fused_kv);
            }
            Tensor position_batch = layer_positions.view({layer_width, batch});
            ops::normalized_rope_kv_append(
                layer_roots.fused_kv, weight.key_norm, position_batch, local_counts, lanes,
                Config::rms_epsilon, local_envelope,
                dflash_state(state).local_layer(static_cast<std::uint32_t>(layer)),
                state.execution.device.stream);
        }
        (void)table_rows;
    }
}

template <class V>
void propose_batch_impl(DFlashBatchContext& state, qwen3::DFlashDecodeState& frame,
                        std::int32_t batch_size, std::uint32_t k, DFlashEnvelopes envelopes,
                        std::uint32_t verify_width, bool exact_sequence_envelope,
                        std::int32_t row_begin = 0) {
    if constexpr (!V::supports_dflash) {
        throw std::logic_error("DFlash proposal is unavailable for this target");
    } else {
        using Config = typename V::DFlashConfig;
        // Chain rounds draft every row in one batched pass; tree verify keeps per-row proposals.
        const bool batched_chain = verify_width == k + 1;
        if (batch_size > 1 && !batched_chain) {
            for (std::int32_t row = 0; row < batch_size; ++row) {
                propose_batch_impl<V>(state, frame, 1, k, envelopes, verify_width,
                                      exact_sequence_envelope, row_begin + row);
            }
            return;
        }
        // Eager callers provide the batch maximum for workspace safety, but SWA's direct/split
        // route is selected from max_context. Use the host ingress frontier so an isolated row
        // takes the same route as an eager C=1 call. Captured graphs retain their fixed profile
        // envelope because launch topology cannot vary across replays.
        if (exact_sequence_envelope && batch_size == 1) {
            const std::int32_t frontier =
                state.host_ingress.execution_frontiers.at(static_cast<std::size_t>(row_begin));
            if (frontier < 0) {
                throw std::logic_error("DFlash proposal frontier must be non-negative");
            }
            const auto visible = static_cast<std::uint32_t>(frontier);
            envelopes.local    = {0, visible};
            envelopes.full     = {0, visible};
        }
        const std::int32_t full_width = static_cast<std::int32_t>(k) + 1;
        std::int32_t width            = full_width;
        std::int32_t columns          = width * batch_size;
        Tensor anchors                = frame.anchors.slice(0, row_begin, batch_size);
        Tensor frontiers              = frame.execution_frontiers.slice(0, row_begin, batch_size);
        Tensor valid_full             = frame.target_valid_columns.slice(0, row_begin, batch_size);
        Tensor valid_columns          = valid_full;
        Tensor lanes                  = frame.lanes.slice(0, row_begin, batch_size);
        Tensor ids_view =
            frame.proposal_ids.slice(0, 0, full_width).slice(1, row_begin, batch_size);
        Tensor pos_view =
            frame.proposal_positions.slice(0, 0, full_width).slice(1, row_begin, batch_size);
        Tensor drafts_view  = frame.draft_tokens.slice(0, 0, static_cast<std::int32_t>(k))
                                  .slice(1, row_begin, batch_size);
        constexpr int kSel  = ops::kDflash2PathSelectTopK;
        Tensor sel_ids_view = frame.selector_ids.slice(1, 0, static_cast<std::int32_t>(k))
                                  .slice(2, row_begin, batch_size);
        Tensor sel_q_view   = frame.selector_q.slice(1, 0, static_cast<std::int32_t>(k))
                                  .slice(2, row_begin, batch_size);
        const bool two_block =
            Config::two_block_first > 0 && static_cast<int>(k) > Config::two_block_first;

        state.execution.work.reset();
        // Tree verify allocates W=12; k=7 propose uses W=8. The width prefix is not
        // contiguous at B>1, and prepare_masked_block / embedding / path_select require it.
        Tensor ids_full  = ids_view.is_contiguous()
                               ? ids_view
                               : state.execution.work.alloc(DType::I32, {full_width, batch_size});
        Tensor pos_full  = pos_view.is_contiguous()
                               ? pos_view
                               : state.execution.work.alloc(DType::I32, {full_width, batch_size});
        Tensor drafts    = drafts_view.is_contiguous()
                               ? drafts_view
                               : state.execution.work.alloc(
                                     DType::I32, {static_cast<std::int32_t>(k), batch_size});
        Tensor ids       = ids_full;
        Tensor positions = pos_full;
        ops::prepare_masked_block(anchors, frontiers, valid_full, Config::mask_token, ids_full,
                                  pos_full, state.execution.device.stream);
        Tensor residual_full =
            state.execution.work.alloc(DType::BF16, {Config::hidden, full_width * batch_size});
        Tensor residual = residual_full;
        // A producer feeding a Q4 projection publishes its activation image straight into the
        // serialized activation region, so the projection reads it without its own quantization.
        const DeviceSpan activation_region =
            state.execution.linear_execution != nullptr
                ? state.execution.linear_execution->activation_storage()
                : DeviceSpan{};
        const auto image_for = [&](const Weight& consumer,
                                   std::int32_t input_rows) -> const DeviceSpan* {
            if (consumer.qtype != QType::Q4G64_F16S || activation_region.data == nullptr ||
                activation_region.bytes < ops::q4_activation_image_bytes(columns, input_rows)) {
                return nullptr;
            }
            return &activation_region;
        };
        // Multi-block producers publish the image status word through the drafter's completion
        // words instead of a reset launch.
        const auto image_target =
            [&](const DeviceSpan* image) -> std::optional<ops::Q4ActivationImageTarget> {
            if (image == nullptr) return std::nullopt;
            return ops::Q4ActivationImageTarget{
                *image, static_cast<std::uint32_t*>(dflash_state(state).image_completion.data)};
        };
        const auto project = [&](const Tensor& input, const DeviceSpan* image, const Weight& weight,
                                 Tensor& output) {
            if (image != nullptr) {
                ops::linear_q4_activation_image(weight, output, *image,
                                                state.execution.device.stream);
            } else {
                serialized_linear(state.execution, input, weight, output);
            }
        };
        const auto run_embed_layers = [&] {
            auto layers_scope = state.execution.work.scope();
            auto carry =
                workspace_recipe::dflash_layer_carry<Config>(state.execution.work, columns);
            // Every sublayer but the first publishes the finished residual out of place; with an
            // odd count of those, the embedding starts in the partner buffer so the stream ends in
            // `residual`.
            constexpr bool kStartInPartner = (2 * Config::layers - 1) % 2 == 1;
            Tensor current                 = kStartInPartner ? carry.residual_alt : residual;
            Tensor partner                 = kStartInPartner ? residual : carry.residual_alt;
            ops::embedding(ids.view({columns}), state.execution.model.token_embedding, current,
                           state.execution.device.stream);
            const auto batched = [&](const Tensor& matrix) {
                return matrix.view({matrix.ne[0], width, batch_size});
            };
            Tensor finish_dynamic =
                carry.finish_dynamic.view({ops::kGroupedDynamicConvGroups, 2, width, batch_size});
            // Finishes the previous sublayer (its conv applied to carry.delta) into the partner
            // residual and normalizes it for the next one.
            const auto finish_normalized = [&](const Tensor& previous_base, const Tensor& norm,
                                               Tensor& normalized, const DeviceSpan* image) {
                Tensor delta_batch      = batched(carry.delta);
                Tensor current_batch    = batched(current);
                Tensor partner_batch    = batched(partner);
                Tensor normalized_batch = batched(normalized);
                const auto target       = image_target(image);
                ops::grouped_dynamic_conv_finish_normalized(
                    delta_batch, previous_base, finish_dynamic, current_batch, partner_batch, norm,
                    Config::rms_epsilon, normalized_batch, target ? &*target : nullptr,
                    state.execution.device.stream);
                std::swap(current, partner);
            };
            // The grouped dynamic conv of `hidden`, published as `prepared` or the image.
            const auto conv_prepare = [&](const Tensor& hidden, const Weight& kernel_projection,
                                          const Tensor& base_kernel, Tensor& projection,
                                          const DeviceSpan* hidden_image, Tensor& prepared,
                                          const DeviceSpan* prepared_image) {
                project(hidden, hidden_image, kernel_projection, projection);
                Tensor prepared_batch = batched(prepared);
                const auto target     = image_target(prepared_image);
                ops::grouped_dynamic_conv_prepare(
                    batched(hidden), base_kernel, batched(projection),
                    prepared_image != nullptr ? nullptr : &prepared_batch, finish_dynamic,
                    target ? &*target : nullptr, state.execution.device.stream);
            };

            [&](const auto& dflash) {
                for (int layer = 0; layer < Config::layers; ++layer) {
                    const auto& weight = dflash.layers.at(static_cast<std::size_t>(layer));
                    {
                        auto attention_scope = state.execution.work.scope();
                        auto roots           = workspace_recipe::dflash_attention<Config>(
                            state.execution.work, columns);
                        const DeviceSpan* hidden_image =
                            image_for(weight.attention_conv.kernel_projection, Config::hidden);
                        if (layer == 0) {
                            if (hidden_image != nullptr) {
                                ops::rmsnorm_q4_activation_image(
                                    current, weight.input_norm, Config::rms_epsilon, false,
                                    roots.hidden, *hidden_image, state.execution.device.stream);
                            } else {
                                ops::rmsnorm(current, weight.input_norm, Config::rms_epsilon, false,
                                             roots.hidden, state.execution.device.stream);
                            }
                        } else {
                            const auto& previous =
                                dflash.layers.at(static_cast<std::size_t>(layer - 1));
                            finish_normalized(previous.mlp_conv.base_kernel, weight.input_norm,
                                              roots.hidden, hidden_image);
                        }
                        conv_prepare(roots.hidden, weight.attention_conv.kernel_projection,
                                     weight.attention_conv.base_kernel, roots.projection,
                                     hidden_image, roots.prepared,
                                     image_for(weight.query_key_value, Config::hidden));
                        project(roots.prepared, image_for(weight.query_key_value, Config::hidden),
                                weight.query_key_value, roots.fused_qkv);
                        Tensor value =
                            roots.value.view({Config::head_dim, Config::kv_heads, columns});
                        Tensor query =
                            roots.query.view({Config::head_dim, Config::query_heads, columns});
                        Tensor key = roots.key.view({Config::head_dim, Config::kv_heads, columns});
                        ops::dflash_qkv_norm_rope(positions.view({columns}), roots.fused_qkv,
                                                  weight.query_norm, weight.key_norm,
                                                  Config::rms_epsilon, &query, key, value,
                                                  state.execution.device.stream);
                        Tensor query_batch =
                            query.view({Config::head_dim, Config::query_heads, width, batch_size});
                        Tensor key_batch =
                            key.view({Config::head_dim, Config::kv_heads, width, batch_size});
                        Tensor value_batch =
                            value.view({Config::head_dim, Config::kv_heads, width, batch_size});
                        Tensor attention_batch = roots.attention.view(
                            {Config::head_dim, Config::query_heads, width, batch_size});
                        ops::swa(query_batch, key_batch, value_batch, positions, valid_columns,
                                 lanes, Config::attention_scale,
                                 dflash_state(state).local_layer(static_cast<std::uint32_t>(layer)),
                                 envelopes.local, state.execution.work, attention_batch,
                                 state.execution.device.stream);
                        serialized_linear(state.execution,
                                          roots.attention.view({Config::query_size, columns}),
                                          weight.attention_output, carry.delta);
                    }
                    {
                        auto mlp_scope = state.execution.work.scope();
                        auto roots =
                            workspace_recipe::dflash_mlp<Config>(state.execution.work, columns);
                        const DeviceSpan* hidden_image =
                            image_for(weight.mlp_conv.kernel_projection, Config::hidden);
                        finish_normalized(weight.attention_conv.base_kernel,
                                          weight.post_attention_norm, roots.hidden, hidden_image);
                        const DeviceSpan* prepared_image =
                            image_for(weight.gate_up, Config::hidden);
                        conv_prepare(roots.hidden, weight.mlp_conv.kernel_projection,
                                     weight.mlp_conv.base_kernel, roots.projection, hidden_image,
                                     roots.prepared, prepared_image);
                        project(roots.prepared, prepared_image, weight.gate_up, roots.gate_up);
                        const DeviceSpan* intermediate_image =
                            image_for(weight.down, Config::intermediate);
                        if (intermediate_image != nullptr) {
                            ops::silu_mul_q4_activation_image(roots.gate_up,
                                                              *image_target(intermediate_image),
                                                              state.execution.device.stream);
                        } else {
                            ops::silu_mul(
                                roots.gate_up.slice(0, 0, Config::intermediate),
                                roots.gate_up.slice(0, Config::intermediate, Config::intermediate),
                                roots.intermediate, state.execution.device.stream);
                        }
                        project(roots.intermediate, intermediate_image, weight.down, carry.delta);
                    }
                }
                Tensor delta_batch    = batched(carry.delta);
                Tensor residual_batch = batched(current);
                ops::grouped_dynamic_conv_finish(
                    delta_batch,
                    dflash.layers.at(static_cast<std::size_t>(Config::layers - 1))
                        .mlp_conv.base_kernel,
                    finish_dynamic, residual_batch, state.execution.device.stream);
            }(*state.execution.model.dflash);
            if (current.data != residual.data) {
                throw std::logic_error("DFlash drafter residual ended outside its buffer");
            }
        };
        if (two_block) {
            const std::int32_t split       = Config::two_block_first;
            const std::int32_t first_width = split + 1;
            width                          = first_width;
            columns                        = first_width * batch_size;
            Tensor ids_first   = state.execution.work.alloc(DType::I32, {first_width, batch_size});
            Tensor pos_first   = state.execution.work.alloc(DType::I32, {first_width, batch_size});
            Tensor valid_first = state.execution.work.alloc(DType::I32, {batch_size});
            for (std::int32_t b = 0; b < batch_size; ++b) {
                Tensor slot = valid_first.slice(0, b, 1);
                ops::set_i32_scalar(slot, first_width, state.execution.device.stream);
            }
            HIP_CHECK(hipMemcpy2DAsync(
                ids_first.data, static_cast<std::size_t>(first_width) * sizeof(std::int32_t),
                ids_full.data, static_cast<std::size_t>(full_width) * sizeof(std::int32_t),
                static_cast<std::size_t>(first_width) * sizeof(std::int32_t),
                static_cast<std::size_t>(batch_size), hipMemcpyDeviceToDevice,
                state.execution.device.stream));
            HIP_CHECK(hipMemcpy2DAsync(
                pos_first.data, static_cast<std::size_t>(first_width) * sizeof(std::int32_t),
                pos_full.data, static_cast<std::size_t>(full_width) * sizeof(std::int32_t),
                static_cast<std::size_t>(first_width) * sizeof(std::int32_t),
                static_cast<std::size_t>(batch_size), hipMemcpyDeviceToDevice,
                state.execution.device.stream));
            ids           = ids_first;
            positions     = pos_first;
            valid_columns = valid_first;
            residual      = residual_full.slice(1, 0, columns);
        }
        run_embed_layers();

        if (two_block) {
            [&](const auto& dflash) {
                const std::int32_t split        = Config::two_block_first;
                const std::int32_t suffix       = static_cast<std::int32_t>(k) - split;
                const std::size_t element_bytes = dtype_size(DType::BF16);
                const auto emit_path = [&](const Tensor& residual_in, std::int32_t src_width,
                                           std::int32_t pack_k, std::int32_t col0,
                                           const Tensor& path_anchors, std::int32_t position_offset,
                                           unsigned long long seed_xor, Tensor& path_out,
                                           Tensor* selector_ids, Tensor* selector_q) {
                    Tensor packed = state.execution.work.alloc(
                        DType::BF16, {Config::hidden, pack_k * batch_size});
                    Tensor proposal_hidden = state.execution.work.alloc(
                        DType::BF16, {Config::hidden, pack_k * batch_size});
                    const std::size_t row_bytes = static_cast<std::size_t>(Config::hidden) *
                                                  static_cast<std::size_t>(pack_k) * element_bytes;
                    const std::size_t source_pitch = static_cast<std::size_t>(Config::hidden) *
                                                     static_cast<std::size_t>(src_width) *
                                                     element_bytes;
                    const auto* src = static_cast<const std::byte*>(residual_in.data) +
                                      static_cast<std::size_t>(Config::hidden) *
                                          static_cast<std::size_t>(col0) * element_bytes;
                    HIP_CHECK(hipMemcpy2DAsync(packed.data, row_bytes, src, source_pitch, row_bytes,
                                               static_cast<std::size_t>(batch_size),
                                               hipMemcpyDeviceToDevice,
                                               state.execution.device.stream));
                    ops::rmsnorm(packed, state.execution.model.dflash->final_norm,
                                 Config::rms_epsilon, false, proposal_hidden,
                                 state.execution.device.stream);
                    Tensor hidden_batch =
                        proposal_hidden.view({Config::hidden, pack_k, batch_size});
                    const auto run_select = [&](const Tensor& logits_batch,
                                                const Tensor* logit_token_ids) {
                        ops::dflash2_path_select(
                            logits_batch, hidden_batch, dflash.hidden_projection,
                            dflash.predecessor_codebook, dflash.successor_codebook, path_anchors,
                            frontiers, frame.sampling + row_begin, path_out, state.execution.work,
                            state.execution.device.stream, logit_token_ids, selector_ids,
                            selector_q, seed_xor, position_offset);
                    };
                    if (state.execution.proposal_head == ProposalHead::Full) {
                        Tensor logits = state.execution.work.alloc(
                            DType::BF16, {TextConfig::output_rows, pack_k * batch_size});
                        serialized_linear(state.execution, proposal_hidden,
                                          state.execution.model.output_head, logits);
                        Tensor logits_batch =
                            logits.view({TextConfig::output_rows, pack_k, batch_size});
                        run_select(logits_batch, nullptr);
                    } else {
                        if (!state.execution.model.optimized_proposal.has_value()) {
                            throw std::logic_error("optimized DFlash proposal head is unavailable");
                        }
                        const auto& proposal = *state.execution.model.optimized_proposal;
                        Tensor logits        = state.execution.work.alloc(
                            DType::BF16, {V::draft_head_rows, pack_k * batch_size});
                        serialized_linear(state.execution, proposal_hidden, proposal.head, logits);
                        Tensor logits_batch = logits.view({V::draft_head_rows, pack_k, batch_size});
                        run_select(logits_batch, &proposal.token_ids);
                    }
                };
                Tensor path_first = state.execution.work.alloc(DType::I32, {split, batch_size});
                Tensor sel_ids_first =
                    state.execution.work.alloc(DType::I32, {kSel, split, batch_size});
                Tensor sel_q_first =
                    state.execution.work.alloc(DType::FP32, {kSel, split, batch_size});
                emit_path(residual, width, split, 1, anchors, 0, 0ull, path_first, &sel_ids_first,
                          &sel_q_first);
                HIP_CHECK(hipMemcpy2DAsync(
                    drafts.data, static_cast<std::size_t>(k) * sizeof(std::int32_t),
                    path_first.data, static_cast<std::size_t>(split) * sizeof(std::int32_t),
                    static_cast<std::size_t>(split) * sizeof(std::int32_t),
                    static_cast<std::size_t>(batch_size), hipMemcpyDeviceToDevice,
                    state.execution.device.stream));
                copy_selector_hops(sel_ids_view, sel_ids_first, 0, state.execution.device.stream);
                copy_selector_hops(sel_q_view, sel_q_first, 0, state.execution.device.stream);
                auto* idp = static_cast<std::int32_t*>(ids_full.data);
                auto* pth = static_cast<std::int32_t*>(path_first.data);
                for (std::int32_t b = 0; b < batch_size; ++b) {
                    for (std::int32_t u = 0; u < split; ++u) {
                        HIP_CHECK(hipMemcpyAsync(idp + (1 + u) + full_width * b,
                                                 pth + u + split * b, sizeof(std::int32_t),
                                                 hipMemcpyDeviceToDevice,
                                                 state.execution.device.stream));
                    }
                }
                width         = full_width;
                columns       = full_width * batch_size;
                ids           = ids_full;
                positions     = pos_full;
                valid_columns = valid_full;
                residual      = residual_full;
                run_embed_layers();
                Tensor path_suffix = state.execution.work.alloc(DType::I32, {suffix, batch_size});
                Tensor sel_ids_suffix =
                    state.execution.work.alloc(DType::I32, {kSel, suffix, batch_size});
                Tensor sel_q_suffix =
                    state.execution.work.alloc(DType::FP32, {kSel, suffix, batch_size});
                Tensor anchors2 = state.execution.work.alloc(DType::I32, {batch_size});
                for (std::int32_t b = 0; b < batch_size; ++b) {
                    HIP_CHECK(hipMemcpyAsync(static_cast<std::int32_t*>(anchors2.data) + b,
                                             pth + (split - 1) + split * b, sizeof(std::int32_t),
                                             hipMemcpyDeviceToDevice,
                                             state.execution.device.stream));
                }
                emit_path(residual, width, suffix, 1 + split, anchors2, split,
                          0x9E3779B97F4A7C15ull, path_suffix, &sel_ids_suffix, &sel_q_suffix);
                HIP_CHECK(hipMemcpy2DAsync(static_cast<std::int32_t*>(drafts.data) + split,
                                           static_cast<std::size_t>(k) * sizeof(std::int32_t),
                                           path_suffix.data,
                                           static_cast<std::size_t>(suffix) * sizeof(std::int32_t),
                                           static_cast<std::size_t>(suffix) * sizeof(std::int32_t),
                                           static_cast<std::size_t>(batch_size),
                                           hipMemcpyDeviceToDevice, state.execution.device.stream));
                copy_selector_hops(sel_ids_view, sel_ids_suffix, split,
                                   state.execution.device.stream);
                copy_selector_hops(sel_q_view, sel_q_suffix, split, state.execution.device.stream);
            }(*state.execution.model.dflash);
        } else {
            Tensor packed = state.execution.work.alloc(
                DType::BF16, {Config::hidden, static_cast<std::int32_t>(k) * batch_size});
            const std::size_t element_bytes = dtype_size(DType::BF16);
            const std::size_t row_bytes     = static_cast<std::size_t>(Config::hidden) *
                                              static_cast<std::size_t>(k) * element_bytes;
            const std::size_t source_pitch =
                static_cast<std::size_t>(Config::hidden) * width * element_bytes;
            const auto* source     = static_cast<const std::byte*>(residual.data) +
                                     static_cast<std::size_t>(Config::hidden) * element_bytes;
            Tensor proposal_hidden = state.execution.work.alloc(
                DType::BF16, {Config::hidden, static_cast<std::int32_t>(k) * batch_size});
            const auto pack_proposal = [&] {
                HIP_CHECK(hipMemcpy2DAsync(packed.data, row_bytes, source, source_pitch, row_bytes,
                                           static_cast<std::size_t>(batch_size),
                                           hipMemcpyDeviceToDevice, state.execution.device.stream));
                ops::rmsnorm(packed, state.execution.model.dflash->final_norm, Config::rms_epsilon,
                             false, proposal_hidden, state.execution.device.stream);
            };
            pack_proposal();
            ninfer::targets::qwen3::detail::dflash_candidate_stats::capture_activation(
                "proposal_hidden", proposal_hidden, state.execution.device.stream);
            ninfer::targets::qwen3::detail::dflash_candidate_stats::capture_activation(
                "dflash_residual", residual, state.execution.device.stream);
            [&](const auto& dflash) {
                Tensor hidden_batch = proposal_hidden.view(
                    {Config::hidden, static_cast<std::int32_t>(k), batch_size});
                Tensor sel_ids =
                    sel_ids_view.is_contiguous()
                        ? sel_ids_view
                        : state.execution.work.alloc(
                              DType::I32, {kSel, static_cast<std::int32_t>(k), batch_size});
                Tensor sel_q =
                    sel_q_view.is_contiguous()
                        ? sel_q_view
                        : state.execution.work.alloc(
                              DType::FP32, {kSel, static_cast<std::int32_t>(k), batch_size});
                const auto select = [&](const Tensor& logits_batch, const Tensor* logit_token_ids) {
                    if (dflash_uses_tree_verify(k, verify_width)) {
                        const auto verify_w = static_cast<std::int32_t>(verify_width);
                        Tensor verify_ids =
                            frame.verify_ids.slice(0, 0, verify_w).slice(1, row_begin, batch_size);
                        Tensor parent_index    = frame.parent_index.slice(0, 0, verify_w)
                                                     .slice(1, row_begin, batch_size);
                        Tensor cache_positions = frame.cache_positions.slice(0, 0, verify_w)
                                                     .slice(1, row_begin, batch_size);
                        Tensor rope_positions  = frame.proposal_positions.slice(0, 0, verify_w)
                                                     .slice(1, row_begin, batch_size);
                        Tensor ancestor_mask   = frame.ancestor_mask.slice(0, 0, verify_w)
                                                     .slice(1, row_begin, batch_size);
                        Tensor live_columns =
                            frame.target_valid_columns.slice(0, row_begin, batch_size);
                        ops::dflash2_tree_select(
                            logits_batch, hidden_batch, dflash.hidden_projection,
                            dflash.predecessor_codebook, dflash.successor_codebook, anchors,
                            frontiers, verify_ids, parent_index, cache_positions, rope_positions,
                            ancestor_mask, live_columns, state.execution.work,
                            state.execution.device.stream, logit_token_ids);
                    } else {
                        ops::dflash2_path_select(
                            logits_batch, hidden_batch, dflash.hidden_projection,
                            dflash.predecessor_codebook, dflash.successor_codebook, anchors,
                            frontiers, frame.sampling + row_begin, drafts, state.execution.work,
                            state.execution.device.stream, logit_token_ids, &sel_ids, &sel_q);
                    }
                };
                const auto refine_unmask = [&](Tensor& logits, const auto& head,
                                               const Tensor* logit_token_ids) {
                    if constexpr (Config::unmask_refine <= 0) { return; }
                    constexpr int prefix = Config::unmask_refine;
                    if (prefix <= 0 || prefix >= static_cast<int>(k)) { return; }
                    Tensor stash_logits =
                        state.execution.work.alloc(DType::BF16, {logits.ne[0], logits.ne[1]});
                    Tensor stash_hidden = state.execution.work.alloc(
                        DType::BF16, {proposal_hidden.ne[0], proposal_hidden.ne[1]});
                    HIP_CHECK(hipMemcpyAsync(stash_logits.data, logits.data, logits.bytes(),
                                             hipMemcpyDeviceToDevice,
                                             state.execution.device.stream));
                    HIP_CHECK(hipMemcpyAsync(stash_hidden.data, proposal_hidden.data,
                                             proposal_hidden.bytes(), hipMemcpyDeviceToDevice,
                                             state.execution.device.stream));
                    Tensor logits_for_path =
                        logits.view({logits.ne[0], static_cast<std::int32_t>(k), batch_size});
                    ops::dflash2_path_select(logits_for_path, hidden_batch,
                                             dflash.hidden_projection, dflash.predecessor_codebook,
                                             dflash.successor_codebook, anchors, frontiers,
                                             frame.sampling + row_begin, drafts,
                                             state.execution.work, state.execution.device.stream,
                                             logit_token_ids, nullptr, nullptr, 0, 0, true);
                    auto* idp         = static_cast<std::int32_t*>(ids.data);
                    auto* drp         = static_cast<std::int32_t*>(drafts.data);
                    const int width_i = width;
                    const int k_i     = static_cast<int>(k);
                    for (int b = 0; b < batch_size; ++b) {
                        for (int u = 0; u < prefix; ++u) {
                            HIP_CHECK(hipMemcpyAsync(idp + (1 + u) + width_i * b, drp + u + k_i * b,
                                                     sizeof(std::int32_t), hipMemcpyDeviceToDevice,
                                                     state.execution.device.stream));
                        }
                    }
                    run_embed_layers();
                    pack_proposal();
                    serialized_linear(state.execution, proposal_hidden, head, logits);
                    const std::size_t logit_col =
                        static_cast<std::size_t>(logits.ne[0]) * sizeof(std::uint16_t);
                    const std::size_t hid_col =
                        static_cast<std::size_t>(Config::hidden) * sizeof(std::uint16_t);
                    for (int b = 0; b < batch_size; ++b) {
                        for (int u = 0; u < prefix; ++u) {
                            const std::size_t col = static_cast<std::size_t>(u + k_i * b);
                            HIP_CHECK(hipMemcpyAsync(
                                static_cast<std::byte*>(logits.data) + col * logit_col,
                                static_cast<std::byte*>(stash_logits.data) + col * logit_col,
                                logit_col, hipMemcpyDeviceToDevice, state.execution.device.stream));
                            HIP_CHECK(hipMemcpyAsync(
                                static_cast<std::byte*>(proposal_hidden.data) + col * hid_col,
                                static_cast<std::byte*>(stash_hidden.data) + col * hid_col, hid_col,
                                hipMemcpyDeviceToDevice, state.execution.device.stream));
                        }
                    }
                };
                if (state.execution.proposal_head == ProposalHead::Full) {
                    Tensor logits = state.execution.work.alloc(
                        DType::BF16,
                        {TextConfig::output_rows, static_cast<std::int32_t>(k) * batch_size});
                    serialized_linear(state.execution, proposal_hidden,
                                      state.execution.model.output_head, logits);
                    Tensor logits_batch = logits.view(
                        {TextConfig::output_rows, static_cast<std::int32_t>(k), batch_size});
                    refine_unmask(logits, state.execution.model.output_head, nullptr);
                    ninfer::targets::qwen3::detail::dflash_candidate_stats::capture_logits(
                        logits, nullptr, static_cast<int>(k), state.execution.device.stream);
                    select(logits_batch, nullptr);
                } else {
                    if (!state.execution.model.optimized_proposal.has_value()) {
                        throw std::logic_error("optimized DFlash proposal head is unavailable");
                    }
                    const auto& proposal = *state.execution.model.optimized_proposal;
                    Tensor logits        = state.execution.work.alloc(
                        DType::BF16,
                        {V::draft_head_rows, static_cast<std::int32_t>(k) * batch_size});
                    serialized_linear(state.execution, proposal_hidden, proposal.head, logits);
                    Tensor logits_batch =
                        logits.view({V::draft_head_rows, static_cast<std::int32_t>(k), batch_size});
                    refine_unmask(logits, proposal.head, &proposal.token_ids);
                    ninfer::targets::qwen3::detail::dflash_candidate_stats::capture_logits(
                        logits, &proposal.token_ids, static_cast<int>(k),
                        state.execution.device.stream);
                    select(logits_batch, &proposal.token_ids);
                }
                copy_selector_hops(sel_ids_view, sel_ids, 0, state.execution.device.stream);
                copy_selector_hops(sel_q_view, sel_q, 0, state.execution.device.stream);
            }(state.execution.model.dflash.value());
        }
        qwen3::copy_i32_panel(ids_view, ids_full, state.execution.device.stream);
        qwen3::copy_i32_panel(pos_view, pos_full, state.execution.device.stream);
        qwen3::copy_i32_panel(drafts_view, drafts, state.execution.device.stream);
        state.execution.work.reset();
    }
}

auto dflash_decode_batch_body(DFlashBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                              std::uint32_t verify_width, DFlashEnvelopes envelopes,
                              bool exact_sequence_envelopes, DFlashMixedOwner* owner = nullptr) {
    return [&state, batch_size, k, verify_width, envelopes, exact_sequence_envelopes, owner] {
        if (batch_size <= 0 || batch_size > static_cast<std::int32_t>(kDFlashMaximumConcurrency) ||
            k == 0 || k > kDFlashDecodeMaximumDrafts || verify_width < 2) {
            throw std::logic_error("DFlash decode batch state is incomplete");
        }
        qwen3::DFlashDecodeState& frame = state.frame;
        const auto vw                   = static_cast<std::int32_t>(verify_width);
        HIP_CHECK(hipMemcpyAsync(frame.ingress.data, &state.host_ingress,
                                 sizeof(qwen3::DFlashDecodeIngress), hipMemcpyHostToDevice,
                                 state.execution.device.stream));

        Tensor anchors        = frame.anchors.slice(0, 0, batch_size);
        Tensor frontiers      = frame.execution_frontiers.slice(0, 0, batch_size);
        Tensor context_starts = frame.context_frontiers.slice(0, 0, batch_size);
        Tensor extents        = frame.proposal_extents.slice(0, 0, batch_size);
        Tensor valid_columns  = frame.target_valid_columns.slice(0, 0, batch_size);
        Tensor text_rows      = frame.text_kv_table_rows.slice(0, 0, batch_size);
        Tensor dflash_rows    = frame.dflash_kv_table_rows.slice(0, 0, batch_size);
        Tensor lanes          = frame.lanes.slice(0, 0, batch_size);
        Tensor rope_deltas    = frame.rope_deltas.slice(0, 0, batch_size);
        Tensor append_counts  = frame.append_counts.slice(0, 0, batch_size);
        Tensor drafts =
            frame.draft_tokens.slice(0, 0, static_cast<std::int32_t>(k)).slice(1, 0, batch_size);
        Tensor selector_ids =
            frame.selector_ids.slice(1, 0, static_cast<std::int32_t>(k)).slice(2, 0, batch_size);
        Tensor selector_q =
            frame.selector_q.slice(1, 0, static_cast<std::int32_t>(k)).slice(2, 0, batch_size);
        Tensor verify_ids = frame.verify_ids.slice(0, 0, vw).slice(1, 0, batch_size);
        [[maybe_unused]] Tensor parent_index =
            frame.parent_index.slice(0, 0, vw).slice(1, 0, batch_size);
        [[maybe_unused]] Tensor ancestor_mask =
            frame.ancestor_mask.slice(0, 0, vw).slice(1, 0, batch_size);
        Tensor cache_positions = frame.cache_positions.slice(0, 0, vw).slice(1, 0, batch_size);
        Tensor rope_positions = frame.target_rope_positions.slice(0, 0, vw).slice(1, 0, batch_size);
        Tensor target_tokens  = frame.target_argmax.slice(0, 0, vw).slice(1, 0, batch_size);
        Tensor target_logits  = frame.target_logits.slice(1, 0, vw).slice(2, 0, batch_size);
        Tensor target_hidden  = frame.target_hidden.slice(1, 0, vw).slice(2, 0, batch_size);
        Tensor selected_hidden = frame.target_continuation_hidden.slice(1, 0, batch_size);
        Tensor licensed_tokens = frame.licensed_tokens.slice(0, 0, vw).slice(1, 0, batch_size);
        Tensor licensed_counts = frame.licensed_counts.slice(0, 0, batch_size);
        Tensor accepted        = frame.accepted_drafts.slice(0, 0, batch_size);
        [[maybe_unused]] Tensor accepted_column = frame.accepted_column.slice(0, 0, batch_size);
        [[maybe_unused]] Tensor fold_path = frame.fold_path.slice(0, 0, vw).slice(1, 0, batch_size);

        state.execution.work.reset();
        const std::int32_t storage_width = frame.verify_ids.ne[0];
        const bool compact               = storage_width != vw;
        Tensor append_positions =
            frame.append_positions.slice(0, 0, storage_width).slice(1, 0, batch_size);
        Tensor append_features = state.execution.work.alloc(
            DType::BF16, {Variant::DFlashConfig::feature_rows, storage_width, batch_size});
        ops::prepare_ragged_prefix(dflash_state(state).pending_features, lanes, context_starts,
                                   frontiers, append_features, append_positions, append_counts,
                                   state.execution.device.stream);
        append_context_impl<Variant>(state, append_features, append_positions, append_counts, lanes,
                                     dflash_rows, {0, static_cast<std::uint32_t>(storage_width)});

        propose_batch_impl<Variant>(state, frame, batch_size, k, envelopes, verify_width,
                                    exact_sequence_envelopes);
        drafts =
            frame.draft_tokens.slice(0, 0, static_cast<std::int32_t>(k)).slice(1, 0, batch_size);
        verify_ids      = frame.verify_ids.slice(0, 0, vw).slice(1, 0, batch_size);
        parent_index    = frame.parent_index.slice(0, 0, vw).slice(1, 0, batch_size);
        ancestor_mask   = frame.ancestor_mask.slice(0, 0, vw).slice(1, 0, batch_size);
        cache_positions = frame.cache_positions.slice(0, 0, vw).slice(1, 0, batch_size);
        Tensor proposal_positions =
            frame.proposal_positions.slice(0, 0, vw).slice(1, 0, batch_size);
        const bool vision_positions = state.execution.model.vision.has_value();
        if (vision_positions && !compact) {
            rope_positions = frame.target_rope_positions.slice(0, 0, vw).slice(1, 0, batch_size);
            Tensor deltas  = rope_deltas.slice(0, 0, batch_size);
            ops::offset_i32_position_rows(proposal_positions, deltas, rope_positions,
                                          state.execution.device.stream);
        } else {
            // Preserve the original text-only graph: its MRoPE delta is always zero, so no
            // position materialization or extra graph node is needed.
            rope_positions = proposal_positions;
        }
        target_tokens   = frame.target_argmax.slice(0, 0, vw).slice(1, 0, batch_size);
        licensed_tokens = frame.licensed_tokens.slice(0, 0, vw).slice(1, 0, batch_size);
        fold_path       = frame.fold_path.slice(0, 0, vw).slice(1, 0, batch_size);
        if (compact) {
            // Overlay prefix/propose scratch. Compact verify panels stay live through
            // accept (reset_workspace=false) without stacking on the proposal peak.
            state.execution.work.reset();
            auto copy_panel = [&](const Tensor& src, std::int32_t rows) {
                Tensor dst = state.execution.work.alloc(DType::I32, {rows, batch_size});
                qwen3::copy_i32_panel(dst, src, state.execution.device.stream);
                return dst;
            };
            drafts          = copy_panel(drafts, static_cast<std::int32_t>(k));
            verify_ids      = copy_panel(verify_ids, vw);
            parent_index    = copy_panel(parent_index, vw);
            ancestor_mask   = copy_panel(ancestor_mask, vw);
            cache_positions = copy_panel(cache_positions, vw);
            rope_positions  = copy_panel(rope_positions, vw);
            if (vision_positions) {
                Tensor deltas = rope_deltas.slice(0, 0, batch_size);
                ops::offset_i32_position_rows(rope_positions, deltas, rope_positions,
                                              state.execution.device.stream);
            }
            target_tokens   = state.execution.work.alloc(DType::I32, {vw, batch_size});
            licensed_tokens = state.execution.work.alloc(DType::I32, {vw, batch_size});
            fold_path       = state.execution.work.alloc(DType::I32, {vw, batch_size});
            target_hidden =
                state.execution.work.alloc(DType::BF16, {TextConfig::hidden, vw, batch_size});
            target_logits =
                state.execution.work.alloc(DType::BF16, {TextConfig::output_rows, vw, batch_size});
        }
        const bool use_tree = dflash_uses_tree_verify(k, verify_width);
        if (!use_tree) {
            // Chain verify writes ids only unless positions are filled here. Tree select
            // writes cache_positions = E+j; without this, GQA attends/writes at garbage
            // slots and column 0 diverges from ordinary decode.
            ops::speculative_prepare_verify_inputs(anchors, drafts, frontiers, extents, verify_ids,
                                                   cache_positions, state.execution.device.stream);
        }

        // A mixed round's card carries the prefill owner's KV authority, base and state slot;
        // the verify batch binds through its transactions and frame exactly as alone.
        std::optional<TextContext> card_storage;
        if (owner != nullptr) {
            if (use_tree || owner->prefill == nullptr) {
                throw std::logic_error("mixed DFlash rounds require chain verification");
            }
            PrefillContext& prefill = *owner->prefill;
            card_storage.emplace(
                state.execution.device, state.execution.model, state.execution.linear_execution,
                state.execution.work, prefill.text_kv, state.execution.linear_attention,
                state.execution.io, state.execution.prefill_hidden, state.execution.prefill_chunk,
                prefill.text_kv_base, prefill.mtp_kv, prefill.mtp_cache);
            attach_prefill_state(*card_storage, prefill,
                                 owner->rewrite_checkpoint_capture_frontier);
        } else {
            card_storage.emplace(state.execution.device, state.execution.model,
                                 state.execution.linear_execution, state.execution.work,
                                 qwen3::PagedKVCacheView{}, state.execution.linear_attention,
                                 state.execution.io, state.execution.prefill_hidden,
                                 state.execution.prefill_chunk, 0U);
        }
        TextContext& card = *card_storage;
        if (!state.text_kv_transactions.empty()) {
            card.set_text_kv_transactions(state.text_kv_transactions);
        }
        DFlashFeatureSink sink =
            batch_feature_sink_impl<Variant>(state, lanes, valid_columns, vw, batch_size);
        TargetVerifyFrameView verify_frame{
            .ids             = verify_ids,
            .cache_positions = cache_positions,
            .rope_positions  = rope_positions,
            .valid_columns   = valid_columns,
            .kv_table_rows   = text_rows,
            .lanes           = lanes,
            .target_hidden   = target_hidden,
            .target_logits   = target_logits,
            .target_tokens   = target_tokens,
            .drafts          = drafts,
            .current_extents = extents,
            .frontiers       = frontiers,
            .anchors         = anchors,
            .licensed_tokens = licensed_tokens,
            .licensed_counts = licensed_counts,
            .accepted_drafts = accepted,
            .selected_hidden = selected_hidden,
            .replay_records  = state.execution.replay_records,
            .gdn_fold        = use_tree ? nullptr : frame.gdn_fold,
            .sampling        = frame.sampling,
            .feature_sink    = &sink,
            .token_masks     = state.token_masks,
        };
        if (use_tree) {
            verify_frame.parent_index    = parent_index;
            verify_frame.ancestor_mask   = ancestor_mask;
            verify_frame.prefix_lengths  = frontiers;
            verify_frame.accepted_column = accepted_column;
            verify_frame.fold_path       = fold_path;
            verify_frame.tree_verify     = true;
        } else {
            if (!selector_ids.is_contiguous() || !selector_q.is_contiguous()) {
                Tensor compact_ids = state.execution.work.alloc(
                    DType::I32,
                    {ops::kDflash2PathSelectTopK, static_cast<std::int32_t>(k), batch_size});
                Tensor compact_q = state.execution.work.alloc(
                    DType::FP32,
                    {ops::kDflash2PathSelectTopK, static_cast<std::int32_t>(k), batch_size});
                copy_selector_hops(compact_ids, selector_ids, 0, state.execution.device.stream);
                copy_selector_hops(compact_q, selector_q, 0, state.execution.device.stream);
                selector_ids = compact_ids;
                selector_q   = compact_q;
            }
            verify_frame.draft_selector_ids = selector_ids;
            verify_frame.draft_selector_q   = selector_q;
            if (state.calibrate_p_less_drafts) {
                verify_frame.proposal_calibration =
                    Tensor(frame.proposal_calibration.data, DType::FP32,
                           {ops::kPLessProposalCalibrationTemperatureCount,
                            static_cast<std::int32_t>(k), batch_size});
            }
        }
        if (state.text_kv_transactions.size() != static_cast<std::size_t>(batch_size)) {
            throw std::logic_error("DFlash decode batch has incomplete Text KV authorities");
        }
        const auto* all_positions = static_cast<const std::int32_t*>(cache_positions.data);
        const auto* all_counts    = static_cast<const std::int32_t*>(valid_columns.data);
        for (std::int32_t row = 0; row < batch_size; ++row) {
            state.text_kv_transactions[static_cast<std::size_t>(row)]->begin_device_segment(
                all_positions + static_cast<std::size_t>(row) * verify_width, verify_width,
                all_counts + row);
        }
        std::optional<DFlashFeatureSink> owner_sink;
        std::optional<qwen3::PagedKVTransaction> owner_transaction;
        std::uint32_t owner_frontier = 0;
        if (owner != nullptr) {
            verify_frame = target_verify_prepare(state.execution, card, verify_frame);
            if (!compact) { state.execution.work.reset(); }
            owner_sink.emplace(make_dflash_prefill_sink(*owner->prefill));
            owner->result = card.mixed_prefill_verify(
                owner->slice, *owner_sink, verify_frame.ids, verify_frame.cache_positions,
                verify_frame.rope_positions, verify_frame.valid_columns, verify_frame.kv_table_rows,
                verify_frame.lanes, verify_frame.target_hidden, verify_frame.target_logits,
                verify_frame.target_tokens, sink, owner_transaction, owner_frontier);
            target_verify_resolve(state.execution, state.continuation_hidden_store, card,
                                  verify_frame);
        } else {
            target_verify_accept(state.execution, state.continuation_hidden_store, card,
                                 verify_frame, !compact);
        }
        // Tree output slot i belongs to target column fold_path[i]; chain slots use column i.
        qwen3::record_round_logprobs(frame.logprobs, target_logits, licensed_tokens,
                                     &licensed_counts, use_tree ? &fold_path : nullptr,
                                     TextConfig::token_domain, state.execution.device.stream);
        for (qwen3::PagedKVTransaction* transaction : state.text_kv_transactions) {
            transaction->end_device_segment(state.execution.device.stream);
        }
        if (compact) {
            qwen3::copy_i32_panel(frame.licensed_tokens.slice(0, 0, vw).slice(1, 0, batch_size),
                                  licensed_tokens, state.execution.device.stream);
            qwen3::copy_i32_panel(frame.fold_path.slice(0, 0, vw).slice(1, 0, batch_size),
                                  fold_path, state.execution.device.stream);
            qwen3::copy_strided_width_panel(frame.target_hidden.slice(2, 0, batch_size),
                                            target_hidden, state.execution.device.stream);
        }
        HIP_CHECK(hipMemcpyAsync(&state.host_egress, frame.egress.data,
                                 sizeof(qwen3::DFlashDecodeEgress), hipMemcpyDeviceToHost,
                                 state.execution.device.stream));
        if (owner_sink) {
            // The owner's chunk joins its drafter context after the batch no longer reads
            // workspace: accept, compaction and the egress copy are ordered before it.
            state.execution.work.reset();
            owner_sink->consume_prefill_chunk(
                static_cast<std::int32_t>(owner->result.processed_tokens));
        }
        if (owner_transaction) {
            // Last host action of the round: the commit's status read waits for the whole
            // round instead of stalling submission between the last layer and the tails.
            if (owner_transaction->commit() != owner_frontier) {
                throw std::logic_error("mixed prefill KV transaction published a wrong frontier");
            }
        }
    };
}

} // namespace

DFlashFeatureSink dflash_feature_sink(PrefillContext& state,
                                      DFlashFeatureSink::PrefillConsumer consume_prefill) {
    return prefill_feature_sink_impl<Variant>(state, std::move(consume_prefill));
}

void dflash_append_context(DFlashAppendContext& state, const Tensor& features,
                           const Tensor& positions, const Tensor& commit_counts,
                           const Tensor& lanes, const Tensor& table_rows,
                           ops::KVCacheAppendPrefixExecutionEnvelope envelope) {
    append_context_impl<Variant>(state, features, positions, commit_counts, lanes, table_rows,
                                 envelope);
}

void dflash_append_context(PrefillContext& state, const Tensor& features, const Tensor& positions,
                           const Tensor& commit_counts, const Tensor& lanes,
                           const Tensor& table_rows,
                           ops::KVCacheAppendPrefixExecutionEnvelope envelope) {
    append_context_impl<Variant>(state, features, positions, commit_counts, lanes, table_rows,
                                 envelope);
}

void capture_dflash_decode_batch(DFlashBatchContext& state, std::int32_t batch_size,
                                 std::uint32_t k, std::uint32_t verify_width,
                                 DFlashEnvelopes envelopes, DecodeGraphDefinition& definition,
                                 ops::PersistentDecodeStorage& storage) {
    auto body = dflash_decode_batch_body(state, batch_size, k, verify_width, envelopes, false);
    capture_graph(state, definition, batch_size, storage, body);
}

void dflash_mixed_batch(DFlashBatchContext& state, DFlashMixedOwner& owner, std::int32_t batch_size,
                        std::uint32_t k, std::uint32_t verify_width, DFlashEnvelopes envelopes) {
    auto body =
        dflash_decode_batch_body(state, batch_size, k, verify_width, envelopes, true, &owner);
    run_prepared(state, nullptr, body);
}

void dflash_decode_batch(DFlashBatchContext& state, std::int32_t batch_size, std::uint32_t k,
                         std::uint32_t verify_width, DFlashEnvelopes envelopes,
                         DecodeGraphExecutable* executable) {
    auto body = dflash_decode_batch_body(state, batch_size, k, verify_width, envelopes,
                                         executable == nullptr);
    run_prepared(state, executable, body);
}

} // namespace ninfer::targets::qwen3::detail::NINFER_QWEN3_RUNTIME_NS::schedule
