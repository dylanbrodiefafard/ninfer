#include "targets/qwen3_6_27b/impl/variant.h"
#include "targets/qwen3_6/impl/runtime/adaptive_draft.h"

#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_pair.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/mtp_fc.h"
#include "ninfer/ops/mtp_pack.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/silu_mul.h"

#include <algorithm>
#include <stdexcept>

#define NINFER_QWEN36_VARIANT    ::ninfer::targets::qwen3_6_27b::detail::Variant
#define NINFER_QWEN36_RUNTIME_NS qwen3_6_27b_runtime
#include "targets/qwen3_6/impl/runtime/instantiate.h"

namespace ninfer::targets::qwen3_6_27b::detail {
namespace {

std::vector<GraphExecutionProfile>
graph_profiles_through(std::uint32_t max_frontier,
                       const std::vector<std::uint32_t>& preferred_ends) {
    std::vector<GraphExecutionProfile> out;
    std::uint32_t begin = 0;
    for (const std::uint32_t preferred_end : preferred_ends) {
        if (begin > max_frontier) { break; }
        const std::uint32_t end = std::min(preferred_end, max_frontier);
        out.push_back({begin, end});
        if (end == max_frontier) { return out; }
        begin = end + 1;
    }
    if (begin <= max_frontier) { out.push_back({begin, max_frontier}); }
    return out;
}

void validate_token_interval(std::int32_t first, std::int32_t last) {
    if (first <= 0 || last < first) {
        throw std::invalid_argument("invalid target leaf token interval");
    }
}

constexpr ops::LinearPolicy kNvfp4TextPolicy = ops::LinearPolicy::AllowA4;

ops::LinearPolicy text_policy(const Weight& weight,
                              qwen3_6::TextPhase phase = qwen3_6::TextPhase::Prefill,
                              std::int32_t aggregate_tokens = 0) {
    // P-less is sensitive to small target-logit perturbations at its collision-probability
    // boundary. Keep target verification on the same A16 matrix route as ordinary decode.
    // Wide concurrent verification is presented to these leaves as per-request panels below.
    if (phase == qwen3_6::TextPhase::Verify && aggregate_tokens > 0 &&
        aggregate_tokens <= 16) {
        return ops::LinearPolicy::A16Only;
    }
    return weight.qtype == QType::NVFP4 ? kNvfp4TextPolicy : ops::LinearPolicy::A16Only;
}

bool split_verify_panels(qwen3_6::TextPhase phase, std::int32_t route_tokens,
                         std::int32_t aggregate_tokens) {
    return phase == qwen3_6::TextPhase::Verify && route_tokens > 0 &&
           route_tokens < aggregate_tokens;
}

// Packed verify launches Linear at T=width*B. Pin the C=1 width's NVFP4 family so a
// C>1 aggregate does not flip A16↔W4A4 vs the sequential round. Thresholds match
// ops/linear/nvfp4/nvfp4_config.h (attn-in 4, residual-6144 5, residual-17408 3).
ops::LinearPolicy attn_input_packed_policy(const Weight& weight, qwen3_6::TextPhase phase,
                                           std::int32_t route_tokens,
                                           std::int32_t aggregate_tokens) {
    const ops::LinearPolicy policy = text_policy(weight, phase, aggregate_tokens);
    if (route_tokens > 0 && route_tokens < 4 && weight.qtype == QType::NVFP4 &&
        policy == ops::LinearPolicy::AllowA4) {
        return ops::LinearPolicy::A16Only;
    }
    return policy;
}

ops::LinearPolicy residual_packed_policy(const Weight& weight, qwen3_6::TextPhase phase,
                                         std::int32_t route_tokens,
                                         std::int32_t aggregate_tokens) {
    const ops::LinearPolicy policy = text_policy(weight, phase, aggregate_tokens);
    if (route_tokens <= 0 || weight.qtype != QType::NVFP4 ||
        policy != ops::LinearPolicy::AllowA4) {
        return policy;
    }
    if (weight.n == TextConfig::hidden &&
        (weight.k == TextConfig::query_size || weight.k == TextConfig::value_dim) &&
        route_tokens < 5) {
        return ops::LinearPolicy::A16Only;
    }
    if (weight.n == TextConfig::hidden && weight.k == TextConfig::intermediate &&
        route_tokens < 3) {
        return ops::LinearPolicy::A16Only;
    }
    return policy;
}

constexpr std::size_t kMinimumLeafWorkspaceBytes = 1;

// NVFP4 packed verify T=2..16 is fused T=1 GEMV+FP32 conv (one launch per sequence).
std::size_t nvfp4_gdn_record_leaf_bytes(std::int32_t batch, std::int32_t min_width,
                                        std::int32_t max_width) {
    return std::max(kMinimumLeafWorkspaceBytes,
                    ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                        QType::NVFP4, 16384, TextConfig::hidden, kNvfp4TextPolicy, batch, min_width,
                        max_width));
}

std::size_t gdn_record_workspace_bytes(const Tensor& hidden,
                                       const Variant::GdnProjectionWeights& weights) {
    const std::int32_t batch = hidden.ne[2];
    const std::int32_t width = hidden.ne[1];
    if (std::holds_alternative<SplitGdnInputProjectionPayload>(weights.input_projection)) {
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                            TextConfig::key_dim, TextConfig::key_dim, TextConfig::value_dim, batch,
                            width, width));
    }
    return nvfp4_gdn_record_leaf_bytes(batch, width, width);
}

std::size_t gdn_snapshot_workspace_bytes(const Tensor& hidden,
                                         const Variant::GdnProjectionWeights& weights) {
    const std::int32_t batch = hidden.ne[2];
    const std::int32_t width = hidden.ne[1];
    if (std::holds_alternative<SplitGdnInputProjectionPayload>(weights.input_projection)) {
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                            TextConfig::key_dim, TextConfig::key_dim, TextConfig::value_dim, batch,
                            width, width));
    }
    return std::max(
        kMinimumLeafWorkspaceBytes,
        ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
            QType::NVFP4, 16384, TextConfig::hidden, kNvfp4TextPolicy, batch, width, width));
}

void mtp_split_packed_attention(const Tensor& hidden, const Weight& packed, Tensor& query,
                                Tensor& gate, Tensor& key, Tensor& value,
                                WorkspaceArena& workspace, cudaStream_t stream) {
    if (packed.qtype == QType::NVFP4) {
        ops::attn_input_proj(hidden, packed, query, gate, key, value, text_policy(packed),
                             workspace, stream);
        return;
    }
    auto scope     = workspace.scope();
    const int cols = hidden.ne[1];
    Tensor attn_in = workspace.alloc(DType::BF16, {TextConfig::mtp_attention_input_rows, cols});
    ops::linear(hidden, packed, attn_in, stream);
    Tensor query_heads = query.view({TextConfig::head_dim, TextConfig::query_heads, cols});
    Tensor key_heads   = key.view({TextConfig::head_dim, TextConfig::kv_heads, cols});
    Tensor gate_heads  = gate.view({TextConfig::head_dim, TextConfig::query_heads, cols});
    Tensor value_heads = value.view({TextConfig::head_dim, TextConfig::kv_heads, cols});
    ops::mtp_split_attn_in(attn_in, query_heads, key_heads, gate_heads, value_heads, stream);
}

std::size_t mtp_packed_attention_workspace_bytes(std::int32_t first, std::int32_t last) {
    WorkspaceLayoutBuilder layout;
    {
        auto w8 = layout.scope();
        (void)layout.alloc(DType::BF16, {TextConfig::mtp_attention_input_rows, last});
    }
    {
        auto nvfp4 = layout.scope();
        (void)layout.alloc_bytes(ops::attn_input_proj_workspace_capacity_bytes(
            QType::NVFP4, TextConfig::mtp_attention_input_rows, TextConfig::hidden,
            kNvfp4TextPolicy, first, last));
    }
    return layout.peak_bytes(1);
}

} // namespace

std::vector<GraphExecutionProfile> Variant::ordinary_graph_profiles(std::uint32_t capacity) {
    // E+1 is the one-token visible window. Early ranges limit empty producer CTAs; later ranges
    // follow measured split-policy transitions until the producer grid reaches its fixed cap.
    return graph_profiles_through(capacity - 1, {127, 511, 2047, 4095, 8197, 16389, 32767});
}

std::vector<GraphExecutionProfile> Variant::mtp_graph_profiles(std::uint32_t capacity,
                                                               std::uint32_t draft_window) {
    if (draft_window == 0 || capacity == 0) { return {}; }
    // Bound the final AR window E+2K at split-policy transitions until the grid reaches its cap.
    std::vector<std::uint32_t> ends;
    const auto add_shifted = [&](std::uint32_t visible_end, std::uint32_t offset) {
        if (visible_end >= offset) { ends.push_back(visible_end - offset); }
    };
    for (const std::uint32_t visible_end : {128U, 512U, 2048U, 4096U, 8198U, 16390U, 32768U}) {
        add_shifted(visible_end, 2 * draft_window);
    }
    // Target verify and MTP batch both have T=K+1 and W=E+K+1. Preserve one concrete INT8
    // implementation per range at the T=4/5/6 launch boundaries.
    if (draft_window == 3) {
        add_shifted(1029, draft_window + 1);
    } else if (draft_window == 4) {
        for (const std::uint32_t visible_end : {128U, 512U, 1029U}) {
            add_shifted(visible_end, draft_window + 1);
        }
    } else if (draft_window == 5) {
        for (const std::uint32_t visible_end : {128U, 160U, 2054U, 8198U}) {
            add_shifted(visible_end, draft_window + 1);
        }
    }
    std::sort(ends.begin(), ends.end());
    ends.erase(std::unique(ends.begin(), ends.end()), ends.end());
    return graph_profiles_through(capacity - 1, ends);
}

std::vector<GraphExecutionProfile> Variant::dflash_graph_profiles(std::uint32_t capacity,
                                                                  std::uint32_t draft_window,
                                                                  std::uint32_t,
                                                                  std::uint32_t verify_width) {
    if (draft_window == 0 || capacity == 0) { return {}; }
    const std::uint32_t block = verify_width != 0 ? verify_width : draft_window + 1;
    std::vector<std::uint32_t> ends;
    const auto add_shifted = [&](std::uint32_t visible_end, std::uint32_t offset) {
        if (visible_end >= offset) { ends.push_back(visible_end - offset); }
    };
    for (const std::uint32_t visible_end : {128U, 512U, 2048U, 4096U, 8198U, 16390U, 32768U}) {
        add_shifted(visible_end, block);
    }
    for (const std::uint32_t ordinary_end : {127U, 511U, 2047U, 4095U, 8197U, 16389U, 32767U}) {
        ends.push_back(ordinary_end);
    }
    std::sort(ends.begin(), ends.end());
    ends.erase(std::unique(ends.begin(), ends.end()), ends.end());
    return graph_profiles_through(capacity - 1, ends);
}

void Variant::attention_projection(const Tensor& hidden,
                                   const FullAttentionProjectionWeights& weights, Tensor& query,
                                   Tensor& gate, Tensor& key, Tensor& value, qwen3_6::TextPhase phase,
                                   WorkspaceArena& workspace, cudaStream_t stream,
                                   std::int32_t route_tokens) {
    if (const auto* split = std::get_if<SplitAttentionProjectionPayload>(&weights)) {
        ops::attn_input_proj(hidden, split->query_key, split->gate_value, query, gate, key, value,
                             stream);
        return;
    }
    const Weight& fused = std::get<FusedAttentionProjectionPayload>(weights).query_key_gate_value;
    if (split_verify_panels(phase, route_tokens, hidden.ne[1])) {
        for (std::int32_t offset = 0; offset < hidden.ne[1]; offset += route_tokens) {
            Tensor query_panel = query.slice(1, offset, route_tokens);
            Tensor gate_panel  = gate.slice(1, offset, route_tokens);
            Tensor key_panel   = key.slice(1, offset, route_tokens);
            Tensor value_panel = value.slice(1, offset, route_tokens);
            ops::attn_input_proj(hidden.slice(1, offset, route_tokens), fused, query_panel,
                                 gate_panel, key_panel, value_panel,
                                 text_policy(fused, phase, route_tokens), workspace, stream);
        }
        return;
    }
    ops::attn_input_proj(hidden, fused, query, gate, key, value,
                         attn_input_packed_policy(fused, phase, route_tokens, hidden.ne[1]),
                         workspace, stream);
}

void Variant::attention_output_projection(const Tensor& attention, const Weight& weight,
                                          Tensor& residual, qwen3_6::TextPhase phase,
                                          WorkspaceArena& workspace, cudaStream_t stream,
                                          std::int32_t route_tokens) {
    if (split_verify_panels(phase, route_tokens, attention.ne[1])) {
        for (std::int32_t offset = 0; offset < attention.ne[1]; offset += route_tokens) {
            Tensor residual_panel = residual.slice(1, offset, route_tokens);
            ops::linear_add(attention.slice(1, offset, route_tokens), weight,
                            residual_panel, text_policy(weight, phase, route_tokens), workspace,
                            stream);
        }
        return;
    }
    ops::linear_add(attention, weight, residual,
                    residual_packed_policy(weight, phase, route_tokens, attention.ne[1]), workspace,
                    stream);
}

void Variant::mtp_attention_projection(const Tensor& hidden,
                                       const MtpAttentionProjectionWeights& weights, Tensor& query,
                                       Tensor& gate, Tensor& key, Tensor& value,
                                       WorkspaceArena& workspace, cudaStream_t stream) {
    mtp_split_packed_attention(hidden, weights.packed, query, gate, key, value, workspace, stream);
}

void Variant::mtp_kv_projection(const Tensor& hidden, const MtpAttentionProjectionWeights& weights,
                                Tensor& key, Tensor& value, WorkspaceArena& workspace,
                                cudaStream_t stream) {
    if (weights.packed.layout == QuantLayout::RowSplit) {
        ops::linear_pair(hidden, weights.key, weights.value, key, value, stream);
        return;
    }
    auto scope         = workspace.scope();
    const int cols     = hidden.ne[1];
    Tensor query       = workspace.alloc(DType::BF16, {TextConfig::query_size, cols});
    Tensor output_gate = workspace.alloc(DType::BF16, {TextConfig::query_size, cols});
    mtp_split_packed_attention(hidden, weights.packed, query, output_gate, key, value, workspace,
                               stream);
}

void Variant::mtp_q_gate_projection(const Tensor& hidden,
                                    const MtpAttentionProjectionWeights& weights, Tensor& query,
                                    Tensor& gate, WorkspaceArena& workspace, cudaStream_t stream) {
    if (weights.packed.layout == QuantLayout::RowSplit) {
        ops::linear(hidden, weights.query, query, stream);
        ops::linear(hidden, weights.output_gate, gate, stream);
        return;
    }
    auto scope     = workspace.scope();
    const int cols = hidden.ne[1];
    Tensor key     = workspace.alloc(DType::BF16, {TextConfig::kv_size, cols});
    Tensor value   = workspace.alloc(DType::BF16, {TextConfig::kv_size, cols});
    mtp_split_packed_attention(hidden, weights.packed, query, gate, key, value, workspace, stream);
}

void Variant::mtp_fc(const Tensor& embedding_norm, const Tensor& hidden_norm, const Weight& weight,
                     Tensor& residual, WorkspaceArena& workspace, cudaStream_t stream,
                     std::int32_t route_tokens) {
    const int cols = embedding_norm.ne[1];
    const std::int32_t family_tokens = route_tokens > 0 ? route_tokens : cols;
    // Packed C>1 alignment is T=width*B. Pin the C=1 width so fused A16 vs pack+W4A4
    // does not flip vs C=1 (same contract as packed target verify).
    if (weight.qtype == QType::NVFP4 && family_tokens < 8) {
        ops::mtp_fc(embedding_norm, hidden_norm, weight, residual, stream);
        return;
    }
    auto scope          = workspace.scope();
    Tensor packed_input = workspace.alloc(DType::BF16, {TextConfig::mtp_input_rows, cols});
    ops::mtp_pack_fc_input(embedding_norm, hidden_norm, packed_input, stream);
    if (weight.qtype == QType::NVFP4) {
        ops::linear(packed_input, weight, residual, text_policy(weight), workspace, stream);
        return;
    }
    ops::linear(packed_input, weight, residual, stream);
}

void Variant::mtp_attention_output(const Tensor& attention, const Weight& weight, Tensor& residual,
                                   WorkspaceArena& workspace, cudaStream_t stream,
                                   std::int32_t route_tokens) {
    if (weight.qtype == QType::NVFP4) {
        ops::linear_add(attention, weight, residual,
                        residual_packed_policy(weight, qwen3_6::TextPhase::Verify, route_tokens,
                                               attention.ne[1]),
                        workspace, stream);
        return;
    }
    auto scope     = workspace.scope();
    const int cols = attention.ne[1];
    Tensor delta   = workspace.alloc(DType::BF16, {TextConfig::hidden, cols});
    ops::linear(attention, weight, delta, stream);
    ops::residual_add(delta, residual, stream);
}

void Variant::gdn_input_projection(const Tensor& hidden, const GdnProjectionWeights& weights,
                                   Tensor& qkv, Tensor& output_gate, qwen3_6::TextPhase phase,
                                   WorkspaceArena& workspace, cudaStream_t stream) {
    Tensor output_gate_flat =
        output_gate.view({TextConfig::value_dim, static_cast<int>(hidden.ne[1])});
    if (const auto* split =
            std::get_if<SplitGdnInputProjectionPayload>(&weights.input_projection)) {
        ops::gdn_input_proj(hidden, split->query_key, split->value_z, qkv, output_gate_flat,
                            stream);
        return;
    }
    const Weight& fused =
        std::get<FusedGdnInputProjectionPayload>(weights.input_projection).query_key_value_z;
    ops::gdn_input_proj(hidden, fused, qkv, output_gate_flat,
                        text_policy(fused, phase, hidden.ne[1]), workspace, stream);
}

void Variant::gdn_input_projection_snapshot(
    const Tensor& hidden, const GdnProjectionWeights& weights, const Tensor& conv_weight,
    Tensor& conv_states, const Tensor& valid_columns, const Tensor& initial_slot,
    const Tensor& snapshot_base_slot, Tensor& query, Tensor& key, Tensor& value,
    Tensor& output_gate, qwen3_6::TextPhase phase, WorkspaceArena& workspace, cudaStream_t stream) {
    auto workspace_scope     = workspace.scope();
    const DeviceSpan storage = workspace.alloc_bytes(gdn_snapshot_workspace_bytes(hidden, weights));
    WorkspaceArena leaf_workspace(storage);
    Tensor output_gate_view = output_gate.view({TextConfig::value_dim, hidden.ne[1], hidden.ne[2]});
    if (const auto* split =
            std::get_if<SplitGdnInputProjectionPayload>(&weights.input_projection)) {
        ops::gdn_input_proj_conv_snapshot(hidden, split->query_key, split->value_z, conv_weight,
                                          conv_states, valid_columns, initial_slot,
                                          snapshot_base_slot, query, key, value, output_gate_view,
                                          leaf_workspace, stream);
        return;
    }
    const Weight& fused =
        std::get<FusedGdnInputProjectionPayload>(weights.input_projection).query_key_value_z;
    ops::gdn_input_proj_conv_snapshot(hidden, fused, conv_weight, conv_states, valid_columns,
                                      initial_slot, snapshot_base_slot, query, key, value,
                                      output_gate_view,
                                      text_policy(fused, phase, hidden.ne[1]),
                                      leaf_workspace, stream);
}

void Variant::gdn_input_projection_record(const Tensor& hidden, const GdnProjectionWeights& weights,
                                          const Tensor& conv_weight, const Tensor& conv_states,
                                          const Tensor& valid_columns, const Tensor& initial_slots,
                                          Tensor& conv_record, Tensor& query, Tensor& key,
                                          Tensor& value, Tensor& output_gate, qwen3_6::TextPhase phase,
                                          WorkspaceArena& workspace, cudaStream_t stream,
                                          const Tensor* parent_index) {
    auto workspace_scope     = workspace.scope();
    const DeviceSpan storage = workspace.alloc_bytes(gdn_record_workspace_bytes(hidden, weights));
    WorkspaceArena leaf_workspace(storage);
    Tensor output_gate_view = output_gate.view({TextConfig::value_dim, hidden.ne[1], hidden.ne[2]});
    if (const auto* split =
            std::get_if<SplitGdnInputProjectionPayload>(&weights.input_projection)) {
        ops::gdn_input_proj_conv_record(hidden, split->query_key, split->value_z, conv_weight,
                                        conv_states, valid_columns, initial_slots, conv_record,
                                        query, key, value, output_gate_view, leaf_workspace,
                                        stream, parent_index);
        return;
    }
    const Weight& fused =
        std::get<FusedGdnInputProjectionPayload>(weights.input_projection).query_key_value_z;
    ops::gdn_input_proj_conv_record(hidden, fused, conv_weight, conv_states, valid_columns,
                                    initial_slots, conv_record, query, key, value, output_gate_view,
                                    text_policy(fused, phase, hidden.ne[1]), leaf_workspace, stream,
                                    parent_index);
}

void Variant::gdn_output_projection(const Tensor& hidden, const Weight& weight, Tensor& residual,
                                    qwen3_6::TextPhase phase, WorkspaceArena& workspace,
                                    cudaStream_t stream, std::int32_t route_tokens) {
    if (split_verify_panels(phase, route_tokens, hidden.ne[1])) {
        for (std::int32_t offset = 0; offset < hidden.ne[1]; offset += route_tokens) {
            Tensor residual_panel = residual.slice(1, offset, route_tokens);
            ops::linear_add(hidden.slice(1, offset, route_tokens), weight,
                            residual_panel, text_policy(weight, phase, route_tokens), workspace,
                            stream);
        }
        return;
    }
    ops::linear_add(hidden, weight, residual,
                    residual_packed_policy(weight, phase, route_tokens, hidden.ne[1]), workspace,
                    stream);
}

void Variant::gdn_norm_control_projection(const Tensor& residual, const Tensor& norm_weight,
                                          float eps, const GdnProjectionWeights& weights,
                                          Tensor& hidden, Tensor& g, Tensor& beta,
                                          WorkspaceArena& workspace, cudaStream_t stream,
                                          std::int32_t route_tokens) {
    // Packed C>1 verify is T=width*B. 27B GDN control keeps SmallT GEMV through T=16 and
    // switches to MMA at T>=17, so C=2 W=12 (T=24) would not match C=1 W=12 (T=12).
    if (route_tokens > 0 && route_tokens < residual.ne[1]) {
        for (std::int32_t offset = 0; offset < residual.ne[1]; offset += route_tokens) {
            Tensor hidden_panel = hidden.slice(1, offset, route_tokens);
            Tensor g_panel      = g.slice(1, offset, route_tokens);
            Tensor beta_panel   = beta.slice(1, offset, route_tokens);
            ops::gdn_norm_gating_proj(residual.slice(1, offset, route_tokens), norm_weight, eps,
                                      weights.a_projection, weights.b_projection, weights.a_log,
                                      weights.dt_bias, workspace, hidden_panel, g_panel,
                                      beta_panel, stream);
        }
        return;
    }
    ops::gdn_norm_gating_proj(residual, norm_weight, eps, weights.a_projection,
                              weights.b_projection, weights.a_log, weights.dt_bias, workspace,
                              hidden, g, beta, stream);
}

void Variant::post_mixer(const Tensor& hidden, const PostMixerWeights& weights, Tensor& residual,
                         qwen3_6::TextPhase phase, WorkspaceArena& workspace, cudaStream_t stream,
                         std::int32_t route_tokens) {
    auto scope        = workspace.scope();
    Tensor activation = workspace.alloc(DType::BF16, {TextConfig::intermediate, hidden.ne[1]});
    if (split_verify_panels(phase, route_tokens, hidden.ne[1])) {
        for (std::int32_t offset = 0; offset < hidden.ne[1]; offset += route_tokens) {
            const Tensor hidden_panel = hidden.slice(1, offset, route_tokens);
            Tensor activation_panel   = activation.slice(1, offset, route_tokens);
            Tensor residual_panel     = residual.slice(1, offset, route_tokens);
            ops::linear_swiglu(hidden_panel, weights.gate_up, activation_panel,
                               text_policy(weights.gate_up, phase, route_tokens), workspace, stream);
            ops::linear_add(activation_panel, weights.down, residual_panel,
                            text_policy(weights.down, phase, route_tokens), workspace, stream);
        }
        return;
    }
    ops::linear_swiglu(hidden, weights.gate_up, activation,
                       text_policy(weights.gate_up, phase, hidden.ne[1]), workspace, stream);
    ops::linear_add(activation, weights.down, residual,
                    residual_packed_policy(weights.down, phase, route_tokens, hidden.ne[1]),
                    workspace, stream);
}

void Variant::mtp_post_mixer(const Tensor& hidden, const MtpPostMixerWeights& weights,
                             Tensor& residual, WorkspaceArena& workspace, cudaStream_t stream,
                             std::int32_t route_tokens) {
    auto scope     = workspace.scope();
    const int cols = hidden.ne[1];
    if (weights.gate_up.qtype == QType::NVFP4) {
        Tensor activation = workspace.alloc(DType::BF16, {TextConfig::intermediate, cols});
        ops::linear_swiglu(hidden, weights.gate_up, activation, text_policy(weights.gate_up),
                           workspace, stream);
        ops::linear_add(activation, weights.down, residual,
                        residual_packed_policy(weights.down, qwen3_6::TextPhase::Verify,
                                               route_tokens, hidden.ne[1]),
                        workspace, stream);
        return;
    }
    Tensor gate_up = workspace.alloc(DType::BF16, {TextConfig::mtp_mlp_gate_up_rows, cols});
    ops::linear(hidden, weights.gate_up, gate_up, stream);
    Tensor activation = workspace.alloc(DType::BF16, {TextConfig::intermediate, cols});
    ops::silu_mul(gate_up.slice(0, 0, TextConfig::intermediate),
                  gate_up.slice(0, TextConfig::intermediate, TextConfig::intermediate), activation,
                  stream);
    Tensor delta = workspace.alloc(DType::BF16, {TextConfig::hidden, cols});
    ops::linear(activation, weights.down, delta, stream);
    ops::residual_add(delta, residual, stream);
}

std::size_t Variant::mtp_attention_projection_workspace_capacity_bytes(std::int32_t first,
                                                                       std::int32_t last) {
    validate_token_interval(first, last);
    return mtp_packed_attention_workspace_bytes(first, last);
}

std::size_t Variant::mtp_kv_projection_workspace_capacity_bytes(std::int32_t first,
                                                                std::int32_t last) {
    validate_token_interval(first, last);
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {TextConfig::query_size, last});
    (void)layout.alloc(DType::BF16, {TextConfig::query_size, last});
    (void)layout.alloc_bytes(mtp_packed_attention_workspace_bytes(first, last));
    return layout.peak_bytes(1);
}

std::size_t Variant::mtp_q_gate_projection_workspace_capacity_bytes(std::int32_t first,
                                                                    std::int32_t last) {
    validate_token_interval(first, last);
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {TextConfig::kv_size, last});
    (void)layout.alloc(DType::BF16, {TextConfig::kv_size, last});
    (void)layout.alloc_bytes(mtp_packed_attention_workspace_bytes(first, last));
    return layout.peak_bytes(1);
}

std::size_t Variant::mtp_fc_workspace_capacity_bytes(std::int32_t first, std::int32_t last) {
    validate_token_interval(first, last);
    // W8 and NVFP4 T≥8 pack into [10240,T] then Linear. NVFP4 T<8 is fused and needs no
    // scratch; oversizing that interval with packed_input is required so W8 MTP (always pack)
    // is not captured against a 0-byte arena.
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {TextConfig::mtp_input_rows, last});
    if (last >= 8) {
        (void)layout.alloc_bytes(ops::linear_workspace_capacity_bytes(
            QType::NVFP4, TextConfig::hidden, TextConfig::mtp_input_rows, kNvfp4TextPolicy, first,
            last));
    }
    return layout.peak_bytes(1);
}

std::size_t Variant::mtp_attention_output_workspace_capacity_bytes(std::int32_t first,
                                                                   std::int32_t last) {
    validate_token_interval(first, last);
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {TextConfig::hidden, last});
    return std::max(layout.peak_bytes(1),
                    ops::linear_add_workspace_capacity_bytes(
                        QType::NVFP4, TextConfig::hidden, TextConfig::query_size, kNvfp4TextPolicy,
                        first, last));
}

std::size_t Variant::attention_projection_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                                   qwen3_6::TextPhase,
                                                                   std::int32_t first,
                                                                   std::int32_t last) {
    validate_token_interval(first, last);
    switch (weights_profile) {
    case WeightsProfile::GroupwiseInt:
    case WeightsProfile::GroupwiseIntW8Endpoints:
        return 0;
    case WeightsProfile::Nvfp4:
        return ops::attn_input_proj_workspace_capacity_bytes(
            QType::NVFP4, 14336, TextConfig::hidden, kNvfp4TextPolicy, first, last);
    }
    throw std::logic_error("invalid 27B weights profile");
}

std::size_t Variant::attention_output_projection_workspace_capacity_bytes(
    WeightsProfile weights_profile, qwen3_6::TextPhase, std::int32_t first, std::int32_t last) {
    validate_token_interval(first, last);
    switch (weights_profile) {
    case WeightsProfile::GroupwiseInt:
    case WeightsProfile::GroupwiseIntW8Endpoints:
        return ops::linear_add_workspace_capacity_bytes(QType::Q5G64_F16S, TextConfig::hidden,
                                                        TextConfig::query_size,
                                                        ops::LinearPolicy::A16Only, first, last);
    case WeightsProfile::Nvfp4:
        return ops::linear_add_workspace_capacity_bytes(QType::NVFP4, TextConfig::hidden,
                                                        TextConfig::query_size, kNvfp4TextPolicy,
                                                        first, last);
    }
    throw std::logic_error("invalid 27B weights profile");
}

std::size_t Variant::gdn_input_projection_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                                   qwen3_6::TextPhase,
                                                                   std::int32_t first,
                                                                   std::int32_t last) {
    validate_token_interval(first, last);
    switch (weights_profile) {
    case WeightsProfile::GroupwiseInt:
    case WeightsProfile::GroupwiseIntW8Endpoints:
        return 0;
    case WeightsProfile::Nvfp4:
        return ops::gdn_input_proj_workspace_capacity_bytes(QType::NVFP4, 16384, TextConfig::hidden,
                                                            kNvfp4TextPolicy, first, last);
    }
    throw std::logic_error("invalid 27B weights profile");
}

std::size_t Variant::gdn_input_projection_snapshot_workspace_capacity_bytes(
    WeightsProfile weights_profile, qwen3_6::TextPhase, std::int32_t batch_size, std::int32_t first,
    std::int32_t last) {
    validate_token_interval(first, last);
    switch (weights_profile) {
    case WeightsProfile::GroupwiseInt:
    case WeightsProfile::GroupwiseIntW8Endpoints:
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                            TextConfig::key_dim, TextConfig::key_dim, TextConfig::value_dim,
                            batch_size, first, last));
    case WeightsProfile::Nvfp4:
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                            QType::NVFP4, 16384, TextConfig::hidden, kNvfp4TextPolicy, batch_size,
                            first, last));
    }
    throw std::logic_error("invalid 27B weights profile");
}

std::size_t Variant::gdn_input_projection_record_workspace_capacity_bytes(
    WeightsProfile weights_profile, qwen3_6::TextPhase, std::int32_t batch_size, std::int32_t first,
    std::int32_t last) {
    validate_token_interval(first, last);
    switch (weights_profile) {
    case WeightsProfile::GroupwiseInt:
    case WeightsProfile::GroupwiseIntW8Endpoints:
        return std::max(kMinimumLeafWorkspaceBytes,
                        ops::gdn_input_proj_conv_record_workspace_capacity_bytes(
                            TextConfig::key_dim, TextConfig::key_dim, TextConfig::value_dim,
                            batch_size, first, last));
    case WeightsProfile::Nvfp4:
        return nvfp4_gdn_record_leaf_bytes(batch_size, first, last);
    }
    throw std::logic_error("invalid 27B weights profile");
}

std::size_t Variant::gdn_output_projection_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                                    qwen3_6::TextPhase,
                                                                    std::int32_t first,
                                                                    std::int32_t last) {
    validate_token_interval(first, last);
    switch (weights_profile) {
    case WeightsProfile::GroupwiseInt:
    case WeightsProfile::GroupwiseIntW8Endpoints:
        return ops::linear_add_workspace_capacity_bytes(QType::Q5G64_F16S, TextConfig::hidden,
                                                        TextConfig::value_dim,
                                                        ops::LinearPolicy::A16Only, first, last);
    case WeightsProfile::Nvfp4:
        return ops::linear_add_workspace_capacity_bytes(
            QType::NVFP4, TextConfig::hidden, TextConfig::value_dim, kNvfp4TextPolicy, first, last);
    }
    throw std::logic_error("invalid 27B weights profile");
}

std::size_t Variant::gdn_norm_control_projection_workspace_capacity_bytes(std::int32_t first,
                                                                          std::int32_t last) {
    return ops::gdn_norm_gating_proj_workspace_capacity_bytes(TextConfig::gdn_value_heads,
                                                              TextConfig::hidden, first, last);
}

std::size_t Variant::post_mixer_workspace_capacity_bytes(WeightsProfile weights_profile,
                                                         qwen3_6::TextPhase, std::int32_t first,
                                                         std::int32_t last) {
    validate_token_interval(first, last);
    QType gate_up_qtype;
    QType down_qtype;
    ops::LinearPolicy policy;
    switch (weights_profile) {
    case WeightsProfile::GroupwiseInt:
    case WeightsProfile::GroupwiseIntW8Endpoints:
        gate_up_qtype = QType::Q4G64_F16S;
        down_qtype    = QType::Q5G64_F16S;
        policy        = ops::LinearPolicy::A16Only;
        break;
    case WeightsProfile::Nvfp4:
        gate_up_qtype = QType::NVFP4;
        down_qtype    = QType::NVFP4;
        policy        = kNvfp4TextPolicy;
        break;
    default:
        throw std::invalid_argument("qwen3_6_27b: invalid weights profile");
    }
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {TextConfig::intermediate, last});
    {
        auto scope = layout.scope();
        (void)layout.alloc_bytes(ops::linear_swiglu_workspace_capacity_bytes(
            gate_up_qtype, 2 * TextConfig::intermediate, TextConfig::hidden, policy, first, last));
    }
    {
        auto scope = layout.scope();
        (void)layout.alloc_bytes(ops::linear_add_workspace_capacity_bytes(
            down_qtype, TextConfig::hidden, TextConfig::intermediate, policy, first, last));
    }
    return layout.peak_bytes(1);
}

std::size_t Variant::mtp_post_mixer_workspace_capacity_bytes(std::int32_t first,
                                                             std::int32_t last) {
    validate_token_interval(first, last);
    WorkspaceLayoutBuilder layout;
    {
        auto w8 = layout.scope();
        (void)layout.alloc(DType::BF16, {TextConfig::mtp_mlp_gate_up_rows, last});
        (void)layout.alloc(DType::BF16, {TextConfig::intermediate, last});
        (void)layout.alloc(DType::BF16, {TextConfig::hidden, last});
    }
    {
        auto nvfp4 = layout.scope();
        (void)layout.alloc(DType::BF16, {TextConfig::intermediate, last});
        {
            auto swiglu = layout.scope();
            (void)layout.alloc_bytes(ops::linear_swiglu_workspace_capacity_bytes(
                QType::NVFP4, TextConfig::mtp_mlp_gate_up_rows, TextConfig::hidden,
                kNvfp4TextPolicy, first, last));
        }
        {
            auto down = layout.scope();
            (void)layout.alloc_bytes(ops::linear_add_workspace_capacity_bytes(
                QType::NVFP4, TextConfig::hidden, TextConfig::intermediate, kNvfp4TextPolicy, first,
                last));
        }
    }
    return layout.peak_bytes(1);
}

float Variant::adaptive_draft_round_time(SpeculativeBackend backend, std::uint32_t k) {
    if (backend == SpeculativeBackend::Mtp) {
        return k < 6 ? qwen3_6::kAdaptiveMtpT[k] : 0.0f;
    }
    if (backend == SpeculativeBackend::DFlash) {
        return k < 8 ? qwen3_6::kAdaptiveDflashT[k] : 0.0f;
    }
    return 0.0f;
}

} // namespace ninfer::targets::qwen3_6_27b::detail
