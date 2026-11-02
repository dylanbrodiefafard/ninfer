#include "targets/qwen3_8_27b/impl/variant.h"

#include "core/device.h"
#include "core/layout.h"
#include "ninfer/ops/attention_projection.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/gated_rmsnorm.h"
#include "ninfer/ops/gdn_gating.h"
#include "ninfer/ops/gdn_projection.h"
#include "ninfer/ops/gdn_replay.h"
#include "ninfer/ops/mtp_pack.h"
#include "ninfer/ops/normalized_linear.h"
#include "ninfer/ops/projected_residual.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/silu_mul.h"
#include "ninfer/types.h"
#include "ops/r9700/gdn/gdn_ops.h"
#include "ops/r9700/gdn_recurrence/gdn_verify_profile.h"
#include "ops/r9700/kv/r9700_attention_profile.h"
#include "ops/r9700/linear/lq4_linear.h"
#include "ops/r9700/linear/fp8_small_t_linear.h"
#include "ops/r9700/linear/linear_execution.h"
#include "ops/r9700/linear/r9700_linear.h"
#include "ops/r9700/linear/r9700_q4_activation_profile.h"
#include "ops/r9700/linear/r9700_w8_activation_profile.h"
#include "targets/qwen3_8_27b/impl/r9700_full_attention.h"
#include "targets/qwen3/impl/runtime/r9700_cache_profile.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::targets::qwen3_8_27b::detail {
namespace {

constexpr std::size_t kExecutionAlignment = 256U;
constexpr std::size_t kSelectedRoleCount  = 7U;
constexpr std::size_t kSelectedProjectionCount =
    3U * static_cast<std::size_t>(TextConfig::full_attention_layers()) +
    2U * static_cast<std::size_t>(TextConfig::gdn_layers());

std::size_t checked_add(std::size_t lhs, std::size_t rhs, const char* label) {
    if (lhs > std::numeric_limits<std::size_t>::max() - rhs) { throw std::overflow_error(label); }
    return lhs + rhs;
}

std::size_t align_up(std::size_t value, std::size_t alignment, const char* label) {
    return checked_add(value, alignment - 1U, label) / alignment * alignment;
}

std::size_t execution_activation_bytes(std::uint32_t prefill_tokens,
                                       std::uint32_t maximum_graph_tokens,
                                       std::uint32_t columns = TextConfig::hidden) {
    const std::uint32_t tokens = std::max(prefill_tokens, maximum_graph_tokens);
    if (tokens == 0U) return 0U;
    const std::size_t bytes =
        ops::LinearExecution::activation_workspace_capacity_bytes(tokens, columns);
    if (bytes == 0U) {
        throw std::overflow_error("R9700 FP8 execution activation capacity overflows");
    }
    return bytes;
}

std::size_t execution_storage_bytes(std::uint32_t prefill_tokens,
                                    std::uint32_t maximum_graph_tokens,
                                    std::uint32_t columns = TextConfig::hidden) {
    return align_up(execution_activation_bytes(prefill_tokens, maximum_graph_tokens, columns),
                    kExecutionAlignment, "R9700 FP8 execution activation alignment overflows");
}

void validate_token_interval(std::int32_t first, std::int32_t last) {
    if (first <= 0 || last < first) {
        throw std::invalid_argument("invalid R9700 target-leaf token interval");
    }
}

void validate_profile(WeightsProfile profile) {
    if (is_fp8_capped_profile(profile)) return;
    if (profile != WeightsProfile::R9700W8G32Candidate &&
        profile != WeightsProfile::R9700W8Bf16EmbeddingEvaluation &&
        profile != WeightsProfile::R9700W8Bf16AttentionQueryKeyEvaluation &&
        profile != WeightsProfile::R9700W8Bf16AttentionValueOutputEvaluation &&
        profile != WeightsProfile::R9700W8Bf16GdnQueryKeyEvaluation &&
        profile != WeightsProfile::R9700Q4G64Evaluation &&
        profile != WeightsProfile::R9700Q4SelectiveProtectedN16K16Evaluation &&
        profile != WeightsProfile::R9700Q4SelectiveProtectedDFlash2Q4Evaluation &&
        profile != WeightsProfile::R9700Q4G64Fp8FourRoleN16K16Evaluation &&
        profile != WeightsProfile::R9700Q4W8Evaluation &&
        profile != WeightsProfile::R9700Q4G64DFlash2Q4MseEvaluation &&
        profile != WeightsProfile::R9700Q4G64DFlash2W8MseEvaluation &&
        profile != WeightsProfile::R9700Q4G64DFlash2Q4Evaluation &&
        profile != WeightsProfile::R9700Q4W8MseDFlash2Q4MseEvaluation &&
        profile != WeightsProfile::R9700Q4W8MseDFlash2W8MseEvaluation &&
        profile != WeightsProfile::R9700Q4W8MseDFlash2Q4Evaluation &&
        profile != WeightsProfile::R9700Q4G64Fp8FourRoleDFlash2Q4MseEvaluation &&
        profile != WeightsProfile::R9700Q4G64Fp8FourRoleDFlash2W8MseEvaluation &&
        profile != WeightsProfile::R9700Q4G64Fp8FourRoleDFlash2Q4Evaluation) {
        throw std::invalid_argument("invalid R9700 target weight profile");
    }
}

std::vector<GraphExecutionProfile>
graph_profiles_through(std::uint32_t max_frontier,
                       const std::vector<std::uint32_t>& preferred_ends) {
    std::vector<GraphExecutionProfile> output;
    std::uint32_t begin = 0;
    for (const std::uint32_t preferred_end : preferred_ends) {
        if (begin > max_frontier) { break; }
        const std::uint32_t end = std::min(preferred_end, max_frontier);
        output.push_back({begin, end});
        if (end == max_frontier) { return output; }
        begin = end + 1;
    }
    if (begin <= max_frontier) { output.push_back({begin, max_frontier}); }
    return output;
}

void copy_bf16_rows(const Tensor& source, std::int32_t source_row, Tensor& destination,
                    std::int32_t destination_row, std::int32_t rows, hipStream_t stream) {
    if (source.dtype != DType::BF16 || destination.dtype != DType::BF16 ||
        !source.is_contiguous() || !destination.is_contiguous() || source.data == nullptr ||
        destination.data == nullptr || source.ne[1] != destination.ne[1] || source.ne[2] != 1 ||
        source.ne[3] != 1 || destination.ne[2] != 1 || destination.ne[3] != 1 || source_row < 0 ||
        destination_row < 0 || rows <= 0 || source_row > source.ne[0] - rows ||
        destination_row > destination.ne[0] - rows) {
        throw std::invalid_argument("R9700 target row copy has invalid geometry");
    }
    const std::size_t element_bytes = sizeof(hip_bfloat16);
    const std::size_t width         = static_cast<std::size_t>(rows) * element_bytes;
    HIP_CHECK(hipMemcpy2DAsync(static_cast<std::byte*>(destination.data) +
                                   static_cast<std::size_t>(destination_row) * element_bytes,
                               static_cast<std::size_t>(destination.ne[0]) * element_bytes,
                               static_cast<const std::byte*>(source.data) +
                                   static_cast<std::size_t>(source_row) * element_bytes,
                               static_cast<std::size_t>(source.ne[0]) * element_bytes, width,
                               static_cast<std::size_t>(source.ne[1]), hipMemcpyDeviceToDevice,
                               stream));
}

std::size_t one_matrix_bytes(std::int32_t rows, std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {rows, tokens});
    return layout.peak_bytes(1);
}

std::size_t two_matrix_bytes(std::int32_t first_rows, std::int32_t second_rows,
                             std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {first_rows, tokens});
    (void)layout.alloc(DType::BF16, {second_rows, tokens});
    return layout.peak_bytes(1);
}

void selected_linear(Variant::ExecutionState* execution, Variant::SelectedLinearRole role,
                     std::int32_t text_layer, const Tensor& input, const Weight& weight,
                     Tensor& output, WorkspaceArena& workspace, hipStream_t stream) {
    if (execution != nullptr && execution->run(role, text_layer, input, weight, output, stream)) {
        return;
    }
    if (weight.qtype == QType::F8E4M3_ROW_F32S) {
        throw std::logic_error("R9700 row-scaled E4M3 projection has no Program execution state");
    }
    if (execution != nullptr) {
        execution->linear(input, weight, output, workspace, stream);
    } else {
        ops::linear(input, weight, output, workspace, stream);
    }
}

void serialized_linear(Variant::ExecutionState* execution, const Tensor& input,
                       const Weight& weight, Tensor& output, WorkspaceArena& workspace,
                       hipStream_t stream) {
    if (execution != nullptr) {
        execution->linear(input, weight, output, workspace, stream);
    } else {
        ops::linear(input, weight, output, workspace, stream);
    }
}

void project_gdn_inputs(const Tensor& hidden, const Variant::GdnProjectionWeights& weights,
                        Tensor& query_key, Tensor& value_z, WorkspaceArena& workspace,
                        hipStream_t stream, Variant::ExecutionState* execution,
                        std::int32_t text_layer) {
    if (hidden.ne[1] <= 0 || hidden.ne[2] <= 0 ||
        hidden.ne[1] > std::numeric_limits<std::int32_t>::max() / hidden.ne[2]) {
        throw std::invalid_argument("R9700 GDN projection has invalid token geometry");
    }
    const std::int32_t tokens = hidden.ne[1] * hidden.ne[2];
    Tensor hidden_flat        = hidden.view({hidden.ne[0], tokens});
    if (execution != nullptr) {
        const Variant::ExecutionState::Lq4Target targets[] = {
            {weights.input_projection.query_key, query_key, nullptr, false},
            {weights.input_projection.value_z, value_z, nullptr, false}};
        if (execution->lq4_projections(hidden_flat, targets, stream)) return;
    }
    if (execution != nullptr &&
        execution->gdn_q4_pair_t1(hidden_flat, weights.input_projection.query_key,
                                  weights.input_projection.value_z, query_key, value_z, stream)) {
        return;
    }
    if (execution != nullptr &&
        execution->gdn_q4_pair_c2c4(hidden_flat, weights.input_projection.query_key,
                                    weights.input_projection.value_z, query_key, value_z, stream)) {
        return;
    }
    if (execution != nullptr && execution->gdn_q4_pair_shared(
                                    hidden_flat, weights.input_projection.query_key,
                                    weights.input_projection.value_z, query_key, value_z, stream)) {
        return;
    }
    selected_linear(execution, Variant::SelectedLinearRole::GdnQueryKey, text_layer, hidden_flat,
                    weights.input_projection.query_key, query_key, workspace, stream);
    serialized_linear(execution, hidden_flat, weights.input_projection.value_z, value_z, workspace,
                      stream);
}

void require_bf16_shape(const Tensor& tensor, std::int32_t rows, std::int32_t width,
                        std::int32_t batch, const char* label) {
    if (tensor.dtype != DType::BF16 || tensor.data == nullptr || !tensor.is_contiguous() ||
        tensor.ne[0] != rows || tensor.ne[1] != width || tensor.ne[2] != batch ||
        tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string("R9700 GDN invalid ") + label);
    }
}

void require_i32_selector(const Tensor& tensor, std::int32_t rows, std::int32_t columns,
                          bool optional, const char* label) {
    if (optional && tensor.data == nullptr) { return; }
    if (tensor.dtype != DType::I32 || tensor.data == nullptr || !tensor.is_contiguous() ||
        tensor.ne[0] != rows || tensor.ne[1] != columns || tensor.ne[2] != 1 || tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string("R9700 GDN invalid ") + label);
    }
}

void require_gdn_conv_operands(const Tensor& hidden, const Tensor& conv_weight,
                               const Tensor& conv_states, const Tensor& valid_columns,
                               const Tensor& initial_slots, std::int32_t minimum_width,
                               std::int32_t maximum_width) {
    const std::int32_t width = hidden.ne[1];
    const std::int32_t batch = hidden.ne[2];
    if (width < minimum_width || width > maximum_width || batch <= 0 ||
        batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("R9700 GDN invalid verification geometry");
    }
    require_bf16_shape(hidden, TextConfig::hidden, width, batch, "hidden");
    require_bf16_shape(conv_weight, TextConfig::convolution_dim, TextConfig::gdn_conv_kernel, 1,
                       "convolution weight");
    if (conv_states.dtype != DType::BF16 || conv_states.data == nullptr ||
        !conv_states.is_contiguous() || conv_states.ne[0] != TextConfig::convolution_dim ||
        conv_states.ne[1] != TextConfig::gdn_conv_state_width || conv_states.ne[2] <= 0 ||
        conv_states.ne[3] != 1) {
        throw std::invalid_argument("R9700 GDN invalid convolution state pool");
    }
    require_i32_selector(valid_columns, batch, 1, true, "valid columns");
    require_i32_selector(initial_slots, batch, 1, false, "initial state slots");
}

} // namespace

struct Variant::ExecutionState::Impl {
    struct Slot {
        const Weight* weight            = nullptr;
        ops::LinearExecution* execution = nullptr;
    };

    std::array<Slot, static_cast<std::size_t>(TextConfig::layers) * kSelectedRoleCount> slots{};
    std::size_t selected = 0;
    DeviceSpan activation{};
    // Ordinary-prefill GDN output-gate projection concurrency (see gdn_q4_normalized_prefill).
    hipStream_t side_stream = nullptr;
    hipEvent_t gate_fork    = nullptr;
    hipEvent_t gate_join    = nullptr;
    bool gate_pending       = false;
    // Layer and token count whose (T, 6144) gated-output status word the GDN front of that layer
    // already zeroed (tokens 0: none).
    std::int32_t gated_status_cleared_layer  = -1;
    std::int32_t gated_status_cleared_tokens = 0;

    Impl() {
        if (hipStreamCreateWithFlags(&side_stream, hipStreamNonBlocking) != hipSuccess ||
            hipEventCreateWithFlags(&gate_fork, hipEventDisableTiming) != hipSuccess ||
            hipEventCreateWithFlags(&gate_join, hipEventDisableTiming) != hipSuccess) {
            release();
            throw std::runtime_error("R9700 GDN side stream creation failed");
        }
    }

    ~Impl() { release(); }

    Impl(const Impl&)            = delete;
    Impl& operator=(const Impl&) = delete;

    void release() noexcept {
        if (gate_join != nullptr) (void)hipEventDestroy(gate_join);
        if (gate_fork != nullptr) (void)hipEventDestroy(gate_fork);
        if (side_stream != nullptr) (void)hipStreamDestroy(side_stream);
        gate_join = gate_fork = nullptr;
        side_stream           = nullptr;
    }

    [[nodiscard]] static std::size_t index(SelectedLinearRole role, std::int32_t text_layer) {
        if (text_layer < 0 || text_layer >= TextConfig::layers) {
            throw std::invalid_argument("R9700 FP8 selected projection has invalid Text layer");
        }
        return static_cast<std::size_t>(text_layer) * kSelectedRoleCount +
               static_cast<std::size_t>(role);
    }
};

Variant::ExecutionState::ExecutionState(const ModelView& model, DeviceSpan serialized_storage,
                                        std::uint32_t prefill_tokens,
                                        std::uint32_t maximum_concurrency,
                                        std::span<const std::uint32_t> verify_widths)
    : impl_(std::make_unique<Impl>()) {
    std::uint32_t maximum_verify_width = 1U;
    for (const auto width : verify_widths)
        maximum_verify_width = std::max(maximum_verify_width, width);
    const std::uint64_t maximum_graph_tokens64 =
        static_cast<std::uint64_t>(maximum_concurrency) * maximum_verify_width;
    if (maximum_graph_tokens64 > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("R9700 FP8 execution graph width overflows");
    }
    const auto maximum_graph_tokens   = static_cast<std::uint32_t>(maximum_graph_tokens64);
    std::uint32_t maximum_fp8_columns = TextConfig::hidden;
    const auto include_fp8_columns    = [&](const Weight& weight) {
        if (weight.qtype == QType::F8E4M3_ROW_F32S)
            maximum_fp8_columns =
                std::max(maximum_fp8_columns, static_cast<std::uint32_t>(weight.k));
    };
    for (const auto& layer : model.full_layers) {
        include_fp8_columns(layer.output);
        include_fp8_columns(layer.post_mixer.down);
    }
    for (const auto& layer : model.gdn_layers) {
        include_fp8_columns(layer.output);
        include_fp8_columns(layer.post_mixer.down);
    }
    const std::size_t activation_bytes =
        execution_activation_bytes(prefill_tokens, maximum_graph_tokens, maximum_fp8_columns);
    const std::size_t required =
        execution_storage_bytes(prefill_tokens, maximum_graph_tokens, maximum_fp8_columns);
    if (serialized_storage.data == nullptr || serialized_storage.bytes < required ||
        reinterpret_cast<std::uintptr_t>(serialized_storage.data) % kExecutionAlignment != 0U) {
        throw std::invalid_argument("R9700 FP8 execution serialized region is invalid");
    }
    auto* base        = static_cast<std::byte*>(serialized_storage.data);
    impl_->activation = serialized_storage;

    const auto install = [&](SelectedLinearRole role, std::int32_t layer, const Weight& weight,
                             ops::LinearExecution* execution) {
        if (weight.qtype != QType::F8E4M3_ROW_F32S) return;
        if (execution == nullptr) {
            throw std::invalid_argument("R9700 FP8 selected projection lacks loaded preparation");
        }
        Impl::Slot& slot = impl_->slots[Impl::index(role, layer)];
        if (slot.execution != nullptr) {
            throw std::logic_error("R9700 FP8 selected projection slot is duplicated");
        }
        slot.weight    = &weight;
        slot.execution = execution;
        slot.execution->bind_storage(base, activation_bytes);
        ++impl_->selected;
    };

    for (std::int32_t layer = 0; layer < TextConfig::layers; ++layer) {
        if (TextConfig::is_full_attention(layer)) {
            const auto& weights =
                model
                    .full_layers[static_cast<std::size_t>(TextConfig::full_attention_index(layer))];
            install(SelectedLinearRole::AttentionQueryKey, layer, weights.projection.query_key,
                    weights.projection.query_key_execution);
            install(SelectedLinearRole::AttentionGateValue, layer, weights.projection.gate_value,
                    weights.projection.gate_value_execution);
            install(SelectedLinearRole::MlpGateUp, layer, weights.post_mixer.gate_up,
                    weights.post_mixer.gate_up_execution);
            install(SelectedLinearRole::AttentionOutput, layer, weights.output,
                    weights.projection.output_execution);
            install(SelectedLinearRole::MlpDown, layer, weights.post_mixer.down,
                    weights.post_mixer.down_execution);
        } else {
            const auto& weights =
                model.gdn_layers[static_cast<std::size_t>(TextConfig::gdn_index(layer))];
            install(SelectedLinearRole::GdnQueryKey, layer,
                    weights.projection.input_projection.query_key,
                    weights.projection.input_projection.query_key_execution);
            install(SelectedLinearRole::GdnOutput, layer, weights.output,
                    weights.projection.output_execution);
            install(SelectedLinearRole::MlpGateUp, layer, weights.post_mixer.gate_up,
                    weights.post_mixer.gate_up_execution);
            install(SelectedLinearRole::MlpDown, layer, weights.post_mixer.down,
                    weights.post_mixer.down_execution);
        }
    }
    if (impl_->selected != 0U && impl_->selected != kSelectedProjectionCount &&
        impl_->selected != 3U && impl_->selected != 22U && impl_->selected != 11U &&
        impl_->selected != 12U && impl_->selected != 15U && impl_->selected != 21U &&
        impl_->selected != 32U && impl_->selected != 80U && impl_->selected != 26U) {
        throw std::invalid_argument(
            "R9700 FP8 execution state has an incomplete selected inventory");
    }
}

void Variant::ExecutionState::linear(const Tensor& input, const Weight& weight, Tensor& output,
                                     WorkspaceArena& fallback_workspace, hipStream_t stream) {
    if (impl_ == nullptr) throw std::logic_error("R9700 linear execution state is empty");
    const std::size_t required =
        ops::linear_workspace_capacity_bytes(weight.qtype, input.ne[1], input.ne[0]);
    if (required == 0U) {
        ops::linear(input, weight, output, stream);
        return;
    }
    if (required > impl_->activation.bytes) {
        throw std::invalid_argument("R9700 serialized linear activation region is too small");
    }
    (void)fallback_workspace;
    ops::linear(input, weight, output, DeviceSpan{impl_->activation.data, required}, stream);
}

DeviceSpan Variant::ExecutionState::activation_storage() const noexcept {
    return impl_ == nullptr ? DeviceSpan{} : impl_->activation;
}

bool Variant::ExecutionState::lq4_targets_supported(
    std::uint32_t tokens, std::uint32_t columns,
    std::span<const Lq4Target> targets) const noexcept {
    if (impl_ == nullptr || targets.empty() || tokens == 0U) return false;
    for (const Lq4Target& target : targets) {
        const Weight& weight        = target.weight;
        const std::int32_t leading  = target.leading.ne[0];
        const std::int32_t trailing = target.trailing == nullptr ? 0 : target.trailing->ne[0];
        const auto valid_output     = [&](const Tensor& output) {
            return output.dtype == DType::BF16 && output.data != nullptr &&
                   output.is_contiguous() && output.ne[0] > 0 &&
                   static_cast<std::int64_t>(output.ne[1]) * output.ne[2] * output.ne[3] ==
                       static_cast<std::int64_t>(tokens);
        };
        // Row-scaled E4M3 protections share the prefill CTA (raw staging) at prefill widths.
        const bool row_scaled = weight.qtype == QType::F8E4M3_ROW_F32S && !target.silu_pair &&
                                ops::r9700::linear::fp8_row_scaled_projection_supported(
                                    tokens, static_cast<std::uint32_t>(weight.n), columns);
        const bool lq4 = weight.qtype == QType::LQ4 && weight.layout == QuantLayout::Lq4N16K64 &&
                         weight.padded_shape[1] == static_cast<std::int32_t>(columns);
        if (!(row_scaled || lq4) || weight.qdata == nullptr || weight.scales == nullptr ||
            weight.k != static_cast<std::int32_t>(columns) || !valid_output(target.leading) ||
            (target.trailing != nullptr &&
             (!valid_output(*target.trailing) || target.accumulate || leading % 128 != 0)) ||
            (target.silu_pair
                 ? (target.trailing != nullptr || target.accumulate || 2 * leading != weight.n)
                 : leading + trailing != weight.n) ||
            !ops::r9700::linear::lq4_linear_supported(tokens, static_cast<std::uint32_t>(weight.n),
                                                      columns)) {
            return false;
        }
    }
    return true;
}

ops::r9700::linear::Fp8ActivationWorkspace
Variant::ExecutionState::lq4_image(std::uint32_t tokens, std::uint32_t columns) const {
    namespace linear        = ops::r9700::linear;
    const std::size_t bytes = linear::fp8_activation_workspace_capacity_bytes(tokens, columns);
    if (bytes == 0U || bytes > impl_->activation.bytes) {
        throw std::invalid_argument("R9700 LQ4 activation image exceeds the serialized region");
    }
    linear::Fp8ActivationWorkspace image{};
    HIP_CHECK(linear::fp8_bind_activation_workspace(impl_->activation.data, bytes, tokens, columns,
                                                    &image));
    return image;
}

void Variant::ExecutionState::lq4_project(const ops::r9700::linear::Fp8ActivationWorkspace& image,
                                          std::span<const Lq4Target> targets,
                                          hipStream_t stream) const {
    namespace linear = ops::r9700::linear;
    const auto view  = [&](const Lq4Target& target) {
        const Weight& weight = target.weight;
        return linear::Lq4Weight{static_cast<const std::uint8_t*>(weight.qdata),
                                 weight.qtype == QType::F8E4M3_ROW_F32S
                                     ? nullptr
                                     : static_cast<const std::uint8_t*>(weight.qhigh),
                                 static_cast<const float*>(weight.scales),
                                 static_cast<std::uint32_t>(weight.n), image.padded_columns};
    };
    const auto output = [](const Lq4Target& target) {
        linear::Lq4Output out{.leading    = static_cast<hip_bfloat16*>(target.leading.data),
                              .accumulate = target.accumulate,
                              .silu_pair  = target.silu_pair};
        if (target.trailing != nullptr) {
            out.trailing = static_cast<hip_bfloat16*>(target.trailing->data);
            out.split    = static_cast<std::uint32_t>(target.leading.ne[0]);
        }
        return out;
    };
    std::size_t index = 0;
    // Consecutive targets of equal epilogue kind share one launch.
    for (; index + 1U < targets.size(); index += 2U) {
        const Lq4Target& first  = targets[index];
        const Lq4Target& second = targets[index + 1U];
        if (first.accumulate != second.accumulate || first.silu_pair || second.silu_pair) break;
        HIP_CHECK(linear::lq4_linear_pair(view(first), output(first), view(second), output(second),
                                          image, stream));
    }
    for (; index < targets.size(); ++index)
        HIP_CHECK(linear::lq4_linear(view(targets[index]), image, output(targets[index]), stream));
}

namespace {

// Cache warming (core/cache_warm.h) at weight-streaming widths only: a prefill GEMM reuses its
// weight across token tiles. Sizes were swept on the C1 DFlash K7 round (2026-10-01): a
// producer warms the first ~2 MiB, about its own duration of DRAM time; the GDN front (9 us)
// and recurrence (13 us) windows warm 4 MiB of the GDN pair projection's head and of the output
// projection after its producer's share.
constexpr std::uint32_t kWarmMaximumTokens    = 128U;
constexpr std::size_t kProducerWarmBytes      = std::size_t{2} << 20U;
constexpr std::size_t kGdnFrontWarmBytes      = std::size_t{4} << 20U;
constexpr std::size_t kGdnRecurrenceWarmBytes = std::size_t{4} << 20U;

// The head of a projection's stream, warmed by the activation producer launched just before it.
CacheWarm producer_warm(const Weight& weight, std::uint32_t tokens) {
    if (tokens > kWarmMaximumTokens) return {};
    return weight_warm(weight, 0U, kProducerWarmBytes);
}

CacheWarm lq4_warm(std::uint32_t tokens,
                   std::span<const Variant::ExecutionState::Lq4Target> targets) {
    return targets.empty() ? CacheWarm{} : producer_warm(targets.front().weight, tokens);
}

std::uint32_t tensor_columns(const Tensor& tensor) {
    return static_cast<std::uint32_t>(static_cast<std::int64_t>(tensor.ne[1]) * tensor.ne[2] *
                                      tensor.ne[3]);
}

} // namespace

bool Variant::ExecutionState::lq4_projections(const Tensor& input,
                                              std::span<const Lq4Target> targets,
                                              hipStream_t stream) {
    if (input.dtype != DType::BF16 || input.data == nullptr || !input.is_contiguous() ||
        input.ne[0] <= 0 || input.ne[1] <= 0)
        return false;
    const std::uint32_t tokens = tensor_columns(input);
    const auto columns         = static_cast<std::uint32_t>(input.ne[0]);
    if (!lq4_targets_supported(tokens, columns, targets)) return false;
    const auto image = lq4_image(tokens, columns);
    HIP_CHECK(ops::r9700::linear::fp8_quantize_activation(
        {static_cast<const hip_bfloat16*>(input.data), image, lq4_warm(tokens, targets)}, stream));
    lq4_project(image, targets, stream);
    return true;
}

bool Variant::ExecutionState::lq4_normalized_projections(const Tensor& residual, const Tensor& norm,
                                                         float eps,
                                                         std::span<const Lq4Target> targets,
                                                         hipStream_t stream, Tensor* normalized) {
    if (residual.dtype != DType::BF16 || residual.ne[0] != TextConfig::hidden ||
        !residual.is_contiguous() || norm.dtype != DType::BF16 || norm.ne[0] != TextConfig::hidden)
        return false;
    const std::uint32_t tokens = tensor_columns(residual);
    if (normalized != nullptr &&
        (normalized->dtype != DType::BF16 || !normalized->is_contiguous() ||
         normalized->numel() != residual.numel()))
        return false;
    if (!lq4_targets_supported(tokens, TextConfig::hidden, targets)) return false;
    const auto image = lq4_image(tokens, TextConfig::hidden);
    HIP_CHECK(ops::r9700::linear::fp8_quantize_normalized_activation(
        {.input       = static_cast<const hip_bfloat16*>(residual.data),
         .weight      = static_cast<const hip_bfloat16*>(norm.data),
         .eps         = eps,
         .unit_offset = true,
         .workspace   = image,
         .normalized =
             normalized == nullptr ? nullptr : static_cast<hip_bfloat16*>(normalized->data),
         .warm = lq4_warm(tokens, targets)},
        stream));
    lq4_project(image, targets, stream);
    return true;
}

bool Variant::ExecutionState::lq4_gated_projections(const Tensor& gate,
                                                    const Tensor& attention_fp32,
                                                    std::span<const Lq4Target> targets,
                                                    hipStream_t stream) {
    if (gate.dtype != DType::BF16 || attention_fp32.dtype != DType::FP32 || !gate.is_contiguous() ||
        !attention_fp32.is_contiguous() || gate.numel() != attention_fp32.numel() ||
        gate.numel() % TextConfig::query_size != 0)
        return false;
    const auto tokens = static_cast<std::uint32_t>(gate.numel() / TextConfig::query_size);
    if (!lq4_targets_supported(tokens, TextConfig::query_size, targets)) return false;
    const auto image = lq4_image(tokens, TextConfig::query_size);
    HIP_CHECK(ops::r9700::linear::fp8_quantize_gated_activation(
        {static_cast<const hip_bfloat16*>(gate.data),
         static_cast<const float*>(attention_fp32.data), image, lq4_warm(tokens, targets)},
        stream));
    lq4_project(image, targets, stream);
    return true;
}

bool Variant::ExecutionState::lq4_gated_rmsnorm_projections(const Tensor& recurrent_output,
                                                            const Tensor& norm, const Tensor& z,
                                                            float eps,
                                                            std::span<const Lq4Target> targets,
                                                            hipStream_t stream) {
    if (recurrent_output.dtype != DType::BF16 || z.dtype != DType::BF16 ||
        norm.dtype != DType::BF16 || norm.numel() != 128 || !recurrent_output.is_contiguous() ||
        !z.is_contiguous() || recurrent_output.numel() != z.numel() ||
        recurrent_output.numel() % TextConfig::value_dim != 0)
        return false;
    const auto tokens =
        static_cast<std::uint32_t>(recurrent_output.numel() / TextConfig::value_dim);
    // Row-scaled protections keep the standalone gated RMSNorm (its per-head reduction order)
    // ahead of their projection; the fused producer serves LQ4 weights.
    for (const Lq4Target& target : targets)
        if (target.weight.qtype == QType::F8E4M3_ROW_F32S) return false;
    if (!lq4_targets_supported(tokens, TextConfig::value_dim, targets)) return false;
    const auto image = lq4_image(tokens, TextConfig::value_dim);
    HIP_CHECK(ops::r9700::linear::fp8_quantize_gated_rmsnorm_activation(
        {.input     = static_cast<const hip_bfloat16*>(recurrent_output.data),
         .gate      = static_cast<const hip_bfloat16*>(z.data),
         .weight    = static_cast<const hip_bfloat16*>(norm.data),
         .eps       = eps,
         .workspace = image,
         .warm      = lq4_warm(tokens, targets)},
        stream));
    lq4_project(image, targets, stream);
    return true;
}

CacheWarm Variant::gdn_recurrence_warm(const Weight& output_projection, std::int32_t tokens) {
    if (tokens <= 0 || static_cast<std::uint32_t>(tokens) > kWarmMaximumTokens) return {};
    return weight_warm(output_projection, kProducerWarmBytes, kGdnRecurrenceWarmBytes);
}

ops::r9700::linear::FusedSiluA8Q4G64DownArgs
Variant::ExecutionState::fused_down_args(const Tensor& gate_up, const Weight& down,
                                         Tensor& residual) const {
    const auto tokens               = static_cast<std::uint32_t>(gate_up.ne[1]);
    constexpr std::uint32_t rows    = TextConfig::hidden;
    constexpr std::uint32_t columns = TextConfig::intermediate;
    if (impl_ == nullptr || gate_up.dtype != DType::BF16 || residual.dtype != DType::BF16 ||
        gate_up.data == nullptr || residual.data == nullptr || !gate_up.is_contiguous() ||
        !residual.is_contiguous() || gate_up.ne[0] != 2 * TextConfig::intermediate ||
        gate_up.ne[1] != static_cast<std::int32_t>(tokens) || gate_up.ne[2] != 1 ||
        gate_up.ne[3] != 1 || residual.ne[0] != TextConfig::hidden ||
        residual.ne[1] != static_cast<std::int32_t>(tokens) || residual.ne[2] != 1 ||
        residual.ne[3] != 1 || down.qtype != QType::Q4G64_F16S || down.ndim != 2 ||
        down.n != static_cast<std::int32_t>(rows) || down.k != static_cast<std::int32_t>(columns) ||
        down.shape[0] != static_cast<std::int32_t>(rows) ||
        down.shape[1] != static_cast<std::int32_t>(columns) ||
        down.padded_shape[0] != static_cast<std::int32_t>(rows) ||
        down.padded_shape[1] != static_cast<std::int32_t>(columns) ||
        down.layout != QuantLayout::Q4N16K16 || down.group != 64 || down.group_size != 64 ||
        down.scale_dtype != DType::FP16 || down.qdata == nullptr || down.scales == nullptr ||
        down.qhigh != nullptr || down.high_plane_bytes != 0U) {
        throw std::invalid_argument("R9700 fused MLP-down binding differs from qualified shape");
    }
    const std::size_t required =
        ops::r9700::linear::a8q4g64_activation_workspace_capacity_bytes(tokens, columns);
    if (required == 0U || impl_->activation.data == nullptr || impl_->activation.bytes < required) {
        throw std::invalid_argument("R9700 fused MLP-down activation region is too small");
    }
    return {.gate_up                    = static_cast<const hip_bfloat16*>(gate_up.data),
            .weight_codes               = static_cast<const std::uint8_t*>(down.qdata),
            .weight_code_bytes          = static_cast<std::size_t>(down.qdata_bytes),
            .weight_scales              = static_cast<const std::uint16_t*>(down.scales),
            .weight_scale_bytes         = static_cast<std::size_t>(down.scale_bytes),
            .activation_workspace       = impl_->activation.data,
            .activation_workspace_bytes = required,
            .residual                   = static_cast<hip_bfloat16*>(residual.data),
            .tokens                     = tokens,
            .rows                       = rows,
            .columns                    = columns,
            .padded_columns             = columns};
}

void Variant::ExecutionState::fused_mlp_down(const Tensor& gate_up, const Weight& down,
                                             Tensor& residual, hipStream_t stream) {
    HIP_CHECK(ops::r9700::linear::fused_silu_a8q4g64_down(fused_down_args(gate_up, down, residual),
                                                          stream));
}

bool Variant::ExecutionState::normalized_mlp(Tensor& residual, const Tensor& norm, float eps,
                                             const Weight& gate_up_weight, const Weight& down,
                                             Tensor& gate_up, std::int32_t text_layer,
                                             hipStream_t stream) {
    const std::int32_t tokens = residual.ne[1];
    if (impl_ == nullptr || ops::r9700::linear::kQ4ActivationBits != 8 || text_layer < 0 ||
        text_layer >= TextConfig::layers || gate_up_weight.qtype != QType::Q4G64_F16S ||
        tokens <= 0 ||
        !ops::r9700::linear::a8q4g64_normalized_mlp_supported(static_cast<std::uint32_t>(tokens)) ||
        residual.dtype != DType::BF16 || !residual.is_contiguous() ||
        residual.ne[0] != TextConfig::hidden || residual.ne[2] != 1 || residual.ne[3] != 1 ||
        gate_up.ne[0] != 2 * TextConfig::intermediate || gate_up.ne[1] != tokens ||
        reinterpret_cast<std::uintptr_t>(residual.data) % 16U != 0U ||
        reinterpret_cast<std::uintptr_t>(norm.data) % 16U != 0U) {
        return false;
    }
    const std::size_t required = ops::r9700::linear::a8q4g64_activation_workspace_capacity_bytes(
        static_cast<std::uint32_t>(tokens), TextConfig::hidden);
    if (required == 0U || impl_->activation.data == nullptr || impl_->activation.bytes < required) {
        throw std::invalid_argument("R9700 normalized MLP activation region is too small");
    }
    HIP_CHECK(ops::r9700::linear::a8q4g64_normalized_mlp(
        {.input                      = static_cast<const hip_bfloat16*>(residual.data),
         .weight_codes               = static_cast<const std::uint8_t*>(gate_up_weight.qdata),
         .weight_code_bytes          = static_cast<std::size_t>(gate_up_weight.qdata_bytes),
         .weight_scales              = static_cast<const std::uint16_t*>(gate_up_weight.scales),
         .weight_scale_bytes         = static_cast<std::size_t>(gate_up_weight.scale_bytes),
         .activation_workspace       = impl_->activation.data,
         .activation_workspace_bytes = required,
         .output                     = static_cast<hip_bfloat16*>(gate_up.data),
         .tokens                     = static_cast<std::uint32_t>(tokens),
         .rows                       = 2 * TextConfig::intermediate,
         .columns                    = TextConfig::hidden,
         .padded_columns             = TextConfig::hidden},
        static_cast<const hip_bfloat16*>(norm.data), eps, true,
        fused_down_args(gate_up, down, residual), stream));
    return true;
}

bool Variant::ExecutionState::gdn_q4_pair_t1(const Tensor& input, const Weight& weight0,
                                             const Weight& weight1, Tensor& output0,
                                             Tensor& output1, hipStream_t stream) {
    if (input.ne[1] <= 0 || input.ne[2] <= 0 || input.ne[3] != 1 ||
        input.ne[1] > std::numeric_limits<std::int32_t>::max() / input.ne[2] ||
        !gdn_q4_pair_t1_selected(ops::r9700::linear::kQ4ActivationBits,
                                 static_cast<std::uint32_t>(input.ne[1] * input.ne[2]),
                                 weight0.qtype, weight1.qtype)) {
        return false;
    }
    constexpr std::int32_t kColumns = TextConfig::hidden;
    constexpr std::int32_t kRows0   = 2 * TextConfig::key_dim;
    constexpr std::int32_t kRows1   = 2 * TextConfig::value_dim;
    const auto valid_tensor         = [](const Tensor& tensor, std::int32_t rows) {
        return tensor.dtype == DType::BF16 && tensor.data != nullptr && tensor.is_contiguous() &&
               tensor.ne[0] == rows && tensor.ne[1] == 1 && tensor.ne[2] == 1 && tensor.ne[3] == 1;
    };
    const auto valid_weight = [](const Weight& weight, std::int32_t rows) {
        return weight.qtype == QType::Q4G64_F16S && weight.ndim == 2U && weight.n == rows &&
               weight.k == kColumns && weight.shape[0] == rows && weight.shape[1] == kColumns &&
               weight.padded_shape[0] == rows && weight.padded_shape[1] == kColumns &&
               weight.layout == QuantLayout::Q4N16K16 && weight.group == 64 &&
               weight.group_size == 64U && weight.scale_dtype == DType::FP16 &&
               weight.qdata != nullptr && weight.scales != nullptr && weight.qhigh == nullptr &&
               weight.high_plane_bytes == 0U;
    };
    if (impl_ == nullptr || input.dtype != DType::BF16 || input.data == nullptr ||
        !input.is_contiguous() || input.ne[0] != kColumns || !valid_tensor(output0, kRows0) ||
        !valid_tensor(output1, kRows1) || !valid_weight(weight0, kRows0) ||
        !valid_weight(weight1, kRows1)) {
        throw std::invalid_argument("R9700 paired GDN T1 binding differs from qualified shape");
    }
    const std::size_t required =
        ops::r9700::linear::a8q4g64_activation_workspace_capacity_bytes(1U, kColumns);
    if (required == 0U || impl_->activation.data == nullptr || impl_->activation.bytes < required) {
        throw std::invalid_argument("R9700 paired GDN T1 activation region is too small");
    }
    HIP_CHECK(ops::r9700::linear::a8q4g64_gdn_pair_t1(
        {.input                      = static_cast<const hip_bfloat16*>(input.data),
         .weight0_codes              = static_cast<const std::uint8_t*>(weight0.qdata),
         .weight0_code_bytes         = static_cast<std::size_t>(weight0.qdata_bytes),
         .weight0_scales             = static_cast<const std::uint16_t*>(weight0.scales),
         .weight0_scale_bytes        = static_cast<std::size_t>(weight0.scale_bytes),
         .output0                    = static_cast<hip_bfloat16*>(output0.data),
         .weight1_codes              = static_cast<const std::uint8_t*>(weight1.qdata),
         .weight1_code_bytes         = static_cast<std::size_t>(weight1.qdata_bytes),
         .weight1_scales             = static_cast<const std::uint16_t*>(weight1.scales),
         .weight1_scale_bytes        = static_cast<std::size_t>(weight1.scale_bytes),
         .output1                    = static_cast<hip_bfloat16*>(output1.data),
         .activation_workspace       = impl_->activation.data,
         .activation_workspace_bytes = required,
         .tokens                     = 1U,
         .columns                    = static_cast<std::uint32_t>(kColumns)},
        stream));
    return true;
}

bool Variant::ExecutionState::attention_q4_pair_t1(const Tensor& hidden, const Weight& query_key,
                                                   const Weight& gate_value, Tensor& query,
                                                   Tensor& key, Tensor& gate, Tensor& value,
                                                   hipStream_t stream) {
    if (hidden.ne[1] <= 0 || hidden.ne[2] <= 0 || hidden.ne[3] != 1 ||
        hidden.ne[1] > std::numeric_limits<std::int32_t>::max() / hidden.ne[2] ||
        !attention_q4_pair_t1_selected(ops::r9700::linear::kQ4ActivationBits,
                                       static_cast<std::uint32_t>(hidden.ne[1] * hidden.ne[2]),
                                       query_key.qtype, gate_value.qtype)) {
        return false;
    }
    if (impl_ == nullptr) {
        throw std::invalid_argument("R9700 attention projection execution state is absent");
    }
    const std::size_t required =
        ops::linear_workspace_capacity_bytes(QType::Q4G64_F16S, 1, TextConfig::hidden);
    if (required == 0U || impl_->activation.data == nullptr || impl_->activation.bytes < required) {
        throw std::invalid_argument("R9700 attention projection activation region is too small");
    }
    ops::full_attention_projection_t1(hidden, query_key, gate_value, query, key, gate, value,
                                      {impl_->activation.data, required}, stream);
    return true;
}

namespace {

bool gdn_q4_input_weight(const Weight& weight, std::int32_t rows) {
    return weight.qtype == QType::Q4G64_F16S && weight.ndim == 2U && weight.n == rows &&
           weight.k == TextConfig::hidden && weight.padded_shape[0] == rows &&
           weight.padded_shape[1] == TextConfig::hidden && weight.layout == QuantLayout::Q4N16K16 &&
           weight.group == 64 && weight.scale_dtype == DType::FP16 && weight.qhigh == nullptr;
}

bool gdn_prefill_rows(const Tensor& tensor, std::int32_t rows, std::int32_t tokens) {
    return tensor.dtype == DType::BF16 && tensor.data != nullptr && tensor.is_contiguous() &&
           tensor.ne[0] == rows && tensor.ne[1] == tokens && tensor.ne[2] == 1 && tensor.ne[3] == 1;
}

// Rows [first, first + rows) of a Q4N16K16 K5120 matrix; row blocks of sixteen are contiguous in
// both planes.
ops::r9700::linear::A8Q4G64SharedProjection gdn_q4_projection(const Weight& weight,
                                                              std::int32_t first, std::int32_t rows,
                                                              Tensor& destination) {
    constexpr std::size_t kCodeBytesPerRow = TextConfig::hidden / 2;
    constexpr std::size_t kScalesPerRow    = TextConfig::hidden / 64;
    return ops::r9700::linear::A8Q4G64SharedProjection{
        static_cast<const std::uint8_t*>(weight.qdata) + first * kCodeBytesPerRow,
        static_cast<std::size_t>(rows) * kCodeBytesPerRow,
        static_cast<const std::uint16_t*>(weight.scales) + first * kScalesPerRow,
        static_cast<std::size_t>(rows) * kScalesPerRow * sizeof(std::uint16_t),
        static_cast<hip_bfloat16*>(destination.data),
        static_cast<std::uint32_t>(rows)};
}

} // namespace

bool Variant::ExecutionState::gdn_q4_pair_shared(const Tensor& input, const Weight& query_key,
                                                 const Weight& value_z, Tensor& query_key_output,
                                                 Tensor& value_z_output, hipStream_t stream) {
    constexpr std::int32_t kRows0 = 2 * TextConfig::key_dim;
    constexpr std::int32_t kRows1 = 2 * TextConfig::value_dim;
    const std::int32_t tokens     = input.ne[1];
    const auto shared             = [&](std::int32_t rows) {
        return ops::r9700::linear::a8q4g64_shared_activation_supported(
            static_cast<std::uint32_t>(tokens), TextConfig::hidden,
            static_cast<std::uint32_t>(rows));
    };
    if (impl_ == nullptr || ops::r9700::linear::kQ4ActivationBits != 8 || tokens <= 0 ||
        input.dtype != DType::BF16 || !input.is_contiguous() || input.ne[0] != TextConfig::hidden ||
        input.ne[2] != 1 || input.ne[3] != 1 || !gdn_q4_input_weight(query_key, kRows0) ||
        !gdn_q4_input_weight(value_z, kRows1) ||
        !gdn_prefill_rows(query_key_output, kRows0, tokens) ||
        !gdn_prefill_rows(value_z_output, kRows1, tokens) || !shared(kRows0) || !shared(kRows1)) {
        return false;
    }
    const std::size_t required = ops::r9700::linear::a8q4g64_activation_workspace_capacity_bytes(
        static_cast<std::uint32_t>(tokens), TextConfig::hidden);
    if (required == 0U || impl_->activation.data == nullptr || impl_->activation.bytes < required) {
        throw std::invalid_argument("R9700 GDN prefill pair activation region is too small");
    }
    HIP_CHECK(ops::r9700::linear::a8q4g64_shared_activation_linear(
        {.input                      = static_cast<const hip_bfloat16*>(input.data),
         .activation_workspace       = impl_->activation.data,
         .activation_workspace_bytes = required,
         .projections                = {gdn_q4_projection(query_key, 0, kRows0, query_key_output),
                                        gdn_q4_projection(value_z, 0, kRows1, value_z_output)},
         .projection_count           = 2,
         .tokens                     = static_cast<std::uint32_t>(tokens),
         .columns                    = TextConfig::hidden},
        stream));
    return true;
}

bool Variant::ExecutionState::gdn_q4_front(const Tensor& residual, const Tensor& norm, float eps,
                                           const GdnProjectionWeights& weights, Tensor& g,
                                           Tensor& beta, std::int32_t text_layer,
                                           hipStream_t stream,
                                           ops::r9700::linear::A8G64ActivationWorkspace* planes_out,
                                           std::size_t* required_out) {
    constexpr std::int32_t kRows0 = 2 * TextConfig::key_dim;
    constexpr std::int32_t kRows1 = 2 * TextConfig::value_dim;
    constexpr std::int32_t kHeads = TextConfig::gdn_value_heads;
    const std::int32_t tokens     = residual.ne[1];
    const auto aligned            = [](const Tensor& tensor) {
        return reinterpret_cast<std::uintptr_t>(tensor.data) % 16U == 0U;
    };
    const auto control_weight = [](const Weight& weight) {
        return weight.qtype == QType::BF16_CTRL && weight.qdata != nullptr && weight.n == kHeads &&
               weight.k == TextConfig::hidden &&
               weight.qdata_bytes ==
                   static_cast<std::uint64_t>(kHeads) * TextConfig::hidden * sizeof(hip_bfloat16) &&
               reinterpret_cast<std::uintptr_t>(weight.qdata) % 16U == 0U;
    };
    const auto fp32_rows = [](const Tensor& tensor, std::int32_t columns) {
        return tensor.dtype == DType::FP32 && tensor.data != nullptr && tensor.is_contiguous() &&
               tensor.ne[0] == kHeads && tensor.ne[1] == columns && tensor.ne[2] == 1 &&
               tensor.ne[3] == 1;
    };
    if (impl_ == nullptr || ops::r9700::linear::kQ4ActivationBits != 8 || tokens <= 0 ||
        !ops::r9700::gdn::bf16_gdn_normalized_front_supported(static_cast<std::uint32_t>(tokens)) ||
        !gdn_prefill_rows(residual, TextConfig::hidden, tokens) || !aligned(residual) ||
        !gdn_prefill_rows(norm, TextConfig::hidden, 1) || !aligned(norm) ||
        !gdn_q4_input_weight(weights.input_projection.query_key, kRows0) ||
        !gdn_q4_input_weight(weights.input_projection.value_z, kRows1) ||
        !control_weight(weights.a_projection) || !control_weight(weights.b_projection) ||
        !fp32_rows(weights.a_log, 1) || !fp32_rows(weights.dt_bias, 1) || !fp32_rows(g, tokens) ||
        !fp32_rows(beta, tokens)) {
        return false;
    }
    const std::size_t required = ops::r9700::linear::a8q4g64_activation_workspace_capacity_bytes(
        static_cast<std::uint32_t>(tokens), TextConfig::hidden);
    if (required == 0U || impl_->activation.data == nullptr || impl_->activation.bytes < required) {
        throw std::invalid_argument("R9700 GDN verification front activation region is too small");
    }
    ops::r9700::linear::A8G64ActivationWorkspace planes{};
    HIP_CHECK(ops::r9700::linear::a8q4g64_bind_activation_workspace(
        impl_->activation.data, required, static_cast<std::uint32_t>(tokens), TextConfig::hidden,
        &planes));
    // The same layer's gated output projection binds this region for (T, 6144); its status word
    // lies beyond every (T, 5120) plane the input projections read, so the front zeroes it and
    // the gated route skips its reset launch.
    std::uint32_t* gated_status = nullptr;
    const std::size_t gated_required =
        ops::r9700::linear::a8q4g64_activation_workspace_capacity_bytes(
            static_cast<std::uint32_t>(tokens), TextConfig::value_dim);
    ops::r9700::linear::A8G64ActivationWorkspace gated_planes{};
    if (gated_required != 0U && impl_->activation.bytes >= gated_required &&
        ops::r9700::linear::a8q4g64_bind_activation_workspace(
            impl_->activation.data, gated_required, static_cast<std::uint32_t>(tokens),
            TextConfig::value_dim, &gated_planes) == hipSuccess &&
        reinterpret_cast<std::uintptr_t>(gated_planes.status) >=
            reinterpret_cast<std::uintptr_t>(planes.status + 1)) {
        gated_status = gated_planes.status;
    }
    HIP_CHECK(ops::r9700::gdn::bf16_gdn_normalized_front(
        static_cast<const hip_bfloat16*>(residual.data),
        static_cast<const hip_bfloat16*>(norm.data), eps, true,
        static_cast<const hip_bfloat16*>(weights.a_projection.qdata),
        static_cast<const hip_bfloat16*>(weights.b_projection.qdata),
        static_cast<const float*>(weights.a_log.data),
        static_cast<const float*>(weights.dt_bias.data), static_cast<float*>(g.data),
        static_cast<float*>(beta.data), planes, nullptr, gated_status, stream));
    impl_->gated_status_cleared_layer  = text_layer;
    impl_->gated_status_cleared_tokens = gated_status != nullptr ? tokens : 0;
    *planes_out                        = planes;
    *required_out                      = required;
    return true;
}

namespace {

ops::r9700::linear::Lq4Weight lq4_view(const Weight& weight) {
    return {static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.qhigh),
            static_cast<const float*>(weight.scales), static_cast<std::uint32_t>(weight.n),
            static_cast<std::uint32_t>(weight.padded_shape[1])};
}

} // namespace

bool Variant::ExecutionState::gdn_q4_normalized_front_t1(
    const Tensor& residual, const Tensor& norm, float eps, const GdnProjectionWeights& weights,
    Tensor& g, Tensor& beta, Tensor& query_key_output, Tensor& value_z_output,
    std::int32_t text_layer, hipStream_t stream) {
    constexpr std::int32_t kRows0 = 2 * TextConfig::key_dim;
    constexpr std::int32_t kRows1 = 2 * TextConfig::value_dim;
    const Weight& query_key       = weights.input_projection.query_key;
    const Weight& value_z         = weights.input_projection.value_z;
    ops::r9700::linear::Fp8ActivationWorkspace image{};
    if (residual.ne[1] == 1 && gdn_prefill_rows(query_key_output, kRows0, 1) &&
        gdn_prefill_rows(value_z_output, kRows1, 1) &&
        gdn_lq4_front(residual, norm, eps, weights, g, beta, stream, &image)) {
        HIP_CHECK(ops::r9700::linear::lq4_linear_pair(
            lq4_view(query_key), {.leading = static_cast<hip_bfloat16*>(query_key_output.data)},
            lq4_view(value_z), {.leading = static_cast<hip_bfloat16*>(value_z_output.data)}, image,
            stream));
        return true;
    }
    ops::r9700::linear::A8G64ActivationWorkspace planes{};
    std::size_t required = 0U;
    if (residual.ne[1] != 1 ||
        !gdn_q4_pair_t1_selected(ops::r9700::linear::kQ4ActivationBits, 1U, query_key.qtype,
                                 value_z.qtype) ||
        !gdn_prefill_rows(query_key_output, kRows0, 1) ||
        !gdn_prefill_rows(value_z_output, kRows1, 1) ||
        !gdn_q4_front(residual, norm, eps, weights, g, beta, text_layer, stream, &planes,
                      &required)) {
        return false;
    }
    HIP_CHECK(ops::r9700::linear::a8q4g64_gdn_pair_t1(
        {.input                      = nullptr,
         .weight0_codes              = static_cast<const std::uint8_t*>(query_key.qdata),
         .weight0_code_bytes         = static_cast<std::size_t>(query_key.qdata_bytes),
         .weight0_scales             = static_cast<const std::uint16_t*>(query_key.scales),
         .weight0_scale_bytes        = static_cast<std::size_t>(query_key.scale_bytes),
         .output0                    = static_cast<hip_bfloat16*>(query_key_output.data),
         .weight1_codes              = static_cast<const std::uint8_t*>(value_z.qdata),
         .weight1_code_bytes         = static_cast<std::size_t>(value_z.qdata_bytes),
         .weight1_scales             = static_cast<const std::uint16_t*>(value_z.scales),
         .weight1_scale_bytes        = static_cast<std::size_t>(value_z.scale_bytes),
         .output1                    = static_cast<hip_bfloat16*>(value_z_output.data),
         .activation_workspace       = impl_->activation.data,
         .activation_workspace_bytes = required,
         .tokens                     = 1U,
         .columns                    = TextConfig::hidden},
        stream));
    return true;
}

bool Variant::ExecutionState::gdn_lq4_front(const Tensor& residual, const Tensor& norm, float eps,
                                            const GdnProjectionWeights& weights, Tensor& g,
                                            Tensor& beta, hipStream_t stream,
                                            ops::r9700::linear::Fp8ActivationWorkspace* image,
                                            const ops::GdnLayerFold* fold) {
    constexpr std::int32_t kHeads = TextConfig::gdn_value_heads;
    const std::int32_t tokens     = residual.ne[1];
    const auto control_weight     = [](const Weight& weight) {
        return weight.qtype == QType::BF16_CTRL && weight.qdata != nullptr && weight.n == kHeads &&
               weight.k == TextConfig::hidden &&
               reinterpret_cast<std::uintptr_t>(weight.qdata) % 16U == 0U;
    };
    const auto fp32_rows = [](const Tensor& tensor, std::int32_t columns) {
        return tensor.dtype == DType::FP32 && tensor.data != nullptr && tensor.is_contiguous() &&
               tensor.ne[0] == kHeads && tensor.ne[1] == columns && tensor.ne[2] == 1 &&
               tensor.ne[3] == 1;
    };
    const Weight& query_key = weights.input_projection.query_key;
    const Weight& value_z   = weights.input_projection.value_z;
    if (impl_ == nullptr || tokens <= 0 || query_key.qtype != QType::LQ4 ||
        value_z.qtype != QType::LQ4 ||
        !ops::r9700::gdn::bf16_gdn_normalized_front_supported(static_cast<std::uint32_t>(tokens)) ||
        !gdn_prefill_rows(residual, TextConfig::hidden, tokens) ||
        !gdn_prefill_rows(norm, TextConfig::hidden, 1) || !control_weight(weights.a_projection) ||
        !control_weight(weights.b_projection) || !fp32_rows(weights.a_log, 1) ||
        !fp32_rows(weights.dt_bias, 1) || !fp32_rows(g, tokens) || !fp32_rows(beta, tokens)) {
        return false;
    }
    *image = lq4_image(static_cast<std::uint32_t>(tokens), TextConfig::hidden);
    HIP_CHECK(ops::r9700::gdn::fp8_gdn_normalized_front(
        static_cast<const hip_bfloat16*>(residual.data),
        static_cast<const hip_bfloat16*>(norm.data), eps, true,
        static_cast<const hip_bfloat16*>(weights.a_projection.qdata),
        static_cast<const hip_bfloat16*>(weights.b_projection.qdata),
        static_cast<const float*>(weights.a_log.data),
        static_cast<const float*>(weights.dt_bias.data), static_cast<float*>(g.data),
        static_cast<float*>(beta.data), *image, stream,
        static_cast<std::uint32_t>(tokens) <= kWarmMaximumTokens
            ? weight_warm(query_key, 0U, kGdnFrontWarmBytes)
            : CacheWarm{},
        fold));
    return true;
}

bool Variant::ExecutionState::gdn_q4_normalized_front_record(
    const Tensor& residual, const Tensor& norm, float eps, const GdnProjectionWeights& weights,
    const GdnConvRecord& record, Tensor& g, Tensor& beta, WorkspaceArena& workspace,
    std::int32_t text_layer, hipStream_t stream, const ops::GdnLayerFold* fold) {
    constexpr std::int32_t kRows0 = 2 * TextConfig::key_dim;
    constexpr std::int32_t kRows1 = 2 * TextConfig::value_dim;
    const std::int32_t tokens     = residual.ne[1];
    const auto record_i32         = [](const Tensor* tensor) {
        return tensor == nullptr || tensor->data == nullptr
                   ? nullptr
                   : static_cast<const std::int32_t*>(tensor->data);
    };
    const auto record_bf16 = [](const Tensor& tensor) {
        return static_cast<hip_bfloat16*>(tensor.data);
    };
    ops::r9700::linear::Fp8ActivationWorkspace image{};
    if (gdn_lq4_front(residual, norm, eps, weights, g, beta, stream, &image, fold)) {
        const auto width       = static_cast<std::uint32_t>(record.query.ne[1]);
        const auto batch       = static_cast<std::uint32_t>(record.query.ne[2]);
        const auto state_slots = static_cast<std::uint32_t>(record.conv_states.ne[2]);
        const auto query_key   = lq4_view(weights.input_projection.query_key);
        const auto value_z     = lq4_view(weights.input_projection.value_z);
        if (ops::r9700::gdn::gdn_lq4_pair_conv_record_supported(width, batch)) {
            HIP_CHECK(ops::r9700::gdn::gdn_lq4_pair_conv_record_bf16(
                image, query_key, value_z,
                static_cast<const hip_bfloat16*>(record.conv_weight.data),
                static_cast<const hip_bfloat16*>(record.conv_states.data),
                record_i32(&record.valid_columns), record_i32(&record.initial_slots),
                record_i32(record.parent_index), record_bf16(record.conv_record),
                record_bf16(record.query), record_bf16(record.key), record_bf16(record.value),
                record_bf16(record.output_gate), width, state_slots, stream));
            return true;
        }
        auto scope              = workspace.scope();
        Tensor query_key_output = workspace.alloc(DType::BF16, {kRows0, tokens});
        Tensor value_z_output   = workspace.alloc(DType::BF16, {kRows1, tokens});
        HIP_CHECK(ops::r9700::linear::lq4_linear_pair(
            query_key, {.leading = static_cast<hip_bfloat16*>(query_key_output.data)}, value_z,
            {.leading = static_cast<hip_bfloat16*>(value_z_output.data)}, image, stream));
        HIP_CHECK(ops::r9700::gdn::projection_conv_record_bf16(
            static_cast<const hip_bfloat16*>(query_key_output.data),
            static_cast<const hip_bfloat16*>(value_z_output.data),
            static_cast<const hip_bfloat16*>(record.conv_weight.data),
            static_cast<const hip_bfloat16*>(record.conv_states.data),
            record_i32(&record.valid_columns), record_i32(&record.initial_slots),
            record_i32(record.parent_index), record_bf16(record.conv_record),
            record_bf16(record.query), record_bf16(record.key), record_bf16(record.value),
            record_bf16(record.output_gate), width, batch, state_slots, stream));
        return true;
    }
    const auto shared = [&](std::int32_t rows) {
        return ops::r9700::linear::a8q4g64_shared_activation_supported(
            static_cast<std::uint32_t>(tokens), TextConfig::hidden,
            static_cast<std::uint32_t>(rows));
    };
    ops::r9700::linear::A8G64ActivationWorkspace planes{};
    std::size_t required = 0U;
    if (tokens <= 0 || !shared(kRows0) || !shared(kRows1) ||
        !gdn_q4_front(residual, norm, eps, weights, g, beta, text_layer, stream, &planes,
                      &required)) {
        return false;
    }
    if (fold != nullptr) ops::gdn_replay_fold_layer(*fold, stream);
    const auto i32 = [](const Tensor* tensor) {
        return tensor == nullptr || tensor->data == nullptr
                   ? nullptr
                   : static_cast<const std::int32_t*>(tensor->data);
    };
    const auto bf16  = [](const Tensor& tensor) { return static_cast<hip_bfloat16*>(tensor.data); };
    const auto width = static_cast<std::uint32_t>(record.query.ne[1]);
    const auto batch = static_cast<std::uint32_t>(record.query.ne[2]);
    const auto state_slots  = static_cast<std::uint32_t>(record.conv_states.ne[2]);
    const Weight& query_key = weights.input_projection.query_key;
    const Weight& value_z   = weights.input_projection.value_z;
    if (ops::r9700::gdn::gdn_pair_conv_record_supported(width, batch)) {
        HIP_CHECK(ops::r9700::gdn::gdn_pair_conv_record_bf16(
            planes, static_cast<const std::uint8_t*>(query_key.qdata),
            static_cast<const std::uint16_t*>(query_key.scales),
            static_cast<const std::uint8_t*>(value_z.qdata),
            static_cast<const std::uint16_t*>(value_z.scales),
            static_cast<const hip_bfloat16*>(record.conv_weight.data),
            static_cast<const hip_bfloat16*>(record.conv_states.data), i32(&record.valid_columns),
            i32(&record.initial_slots), i32(record.parent_index), bf16(record.conv_record),
            bf16(record.query), bf16(record.key), bf16(record.value), bf16(record.output_gate),
            width, state_slots, stream));
        return true;
    }
    auto scope              = workspace.scope();
    Tensor query_key_output = workspace.alloc(DType::BF16, {kRows0, tokens});
    Tensor value_z_output   = workspace.alloc(DType::BF16, {kRows1, tokens});
    HIP_CHECK(ops::r9700::linear::a8q4g64_prepared_shared_activation_linear(
        {.input                      = nullptr,
         .activation_workspace       = impl_->activation.data,
         .activation_workspace_bytes = required,
         .projections                = {gdn_q4_projection(query_key, 0, kRows0, query_key_output),
                                        gdn_q4_projection(value_z, 0, kRows1, value_z_output)},
         .projection_count           = 2,
         .tokens                     = static_cast<std::uint32_t>(tokens),
         .columns                    = TextConfig::hidden},
        stream));
    HIP_CHECK(ops::r9700::gdn::projection_conv_record_bf16(
        static_cast<const hip_bfloat16*>(query_key_output.data),
        static_cast<const hip_bfloat16*>(value_z_output.data),
        static_cast<const hip_bfloat16*>(record.conv_weight.data),
        static_cast<const hip_bfloat16*>(record.conv_states.data), i32(&record.valid_columns),
        i32(&record.initial_slots), i32(record.parent_index), bf16(record.conv_record),
        bf16(record.query), bf16(record.key), bf16(record.value), bf16(record.output_gate), width,
        batch, state_slots, stream));
    return true;
}

bool Variant::ExecutionState::attention_q4_normalized_split(const Tensor& residual,
                                                            const Tensor& norm, float eps,
                                                            const Weight& query_key,
                                                            const Weight& gate_value, Tensor& query,
                                                            Tensor& key, Tensor& gate,
                                                            Tensor& value, hipStream_t stream) {
    constexpr std::int32_t kRows = TextConfig::query_size + TextConfig::kv_size;
    const std::int32_t tokens    = residual.ne[1];
    const auto aligned           = [](const Tensor& tensor) {
        return reinterpret_cast<std::uintptr_t>(tensor.data) % 16U == 0U;
    };
    if (impl_ == nullptr || ops::r9700::linear::kQ4ActivationBits != 8 ||
        (tokens != 5 && tokens != 6) || !gdn_q4_input_weight(query_key, kRows) ||
        !gdn_q4_input_weight(gate_value, kRows) ||
        !gdn_prefill_rows(residual, TextConfig::hidden, tokens) || !aligned(residual) ||
        norm.dtype != DType::BF16 || norm.data == nullptr || norm.numel() != TextConfig::hidden ||
        !aligned(norm) || !gdn_prefill_rows(query, TextConfig::query_size, tokens) ||
        !gdn_prefill_rows(gate, TextConfig::query_size, tokens) ||
        !gdn_prefill_rows(key, TextConfig::kv_size, tokens) ||
        !gdn_prefill_rows(value, TextConfig::kv_size, tokens)) {
        return false;
    }
    const std::size_t required = ops::r9700::linear::a8q4g64_activation_workspace_capacity_bytes(
        static_cast<std::uint32_t>(tokens), TextConfig::hidden);
    if (required == 0U || impl_->activation.data == nullptr || impl_->activation.bytes < required) {
        throw std::invalid_argument("R9700 attention normalized activation region is too small");
    }
    auto projection = [&](const Weight& weight, Tensor& leading, Tensor& trailing) {
        ops::r9700::linear::A8Q4G64SharedProjection result =
            gdn_q4_projection(weight, 0, kRows, leading);
        result.split    = TextConfig::query_size;
        result.trailing = static_cast<hip_bfloat16*>(trailing.data);
        return result;
    };
    HIP_CHECK(ops::r9700::linear::a8q4g64_normalized_shared_activation_linear(
        {.input                      = static_cast<const hip_bfloat16*>(residual.data),
         .activation_workspace       = impl_->activation.data,
         .activation_workspace_bytes = required,
         .projections = {projection(query_key, query, key), projection(gate_value, gate, value)},
         .projection_count = 2,
         .tokens           = static_cast<std::uint32_t>(tokens),
         .columns          = TextConfig::hidden},
        static_cast<const hip_bfloat16*>(norm.data), eps, true, nullptr, stream));
    return true;
}

bool Variant::ExecutionState::attention_q4_gated_output(const Tensor& gate,
                                                        const Tensor& attention_fp32,
                                                        const Weight& weight, Tensor& residual,
                                                        hipStream_t stream) {
    const std::int32_t tokens = residual.ne[1];
    if (impl_ == nullptr || ops::r9700::linear::kQ4ActivationBits != 8 || tokens <= 0 ||
        !ops::r9700::linear::a8q4g64_gated_output_supported(static_cast<std::uint32_t>(tokens)) ||
        weight.qtype != QType::Q4G64_F16S || weight.layout != QuantLayout::Q4N16K16 ||
        weight.group != 64 || weight.scale_dtype != DType::FP16 || weight.qhigh != nullptr ||
        weight.ndim != 2U || weight.n != TextConfig::hidden || weight.k != TextConfig::query_size ||
        weight.padded_shape[1] != weight.k ||
        !gdn_prefill_rows(residual, TextConfig::hidden, tokens) || gate.dtype != DType::BF16 ||
        gate.data == nullptr || !gate.is_contiguous() ||
        gate.numel() != static_cast<std::int64_t>(TextConfig::query_size) * tokens ||
        attention_fp32.dtype != DType::FP32 || attention_fp32.data == nullptr ||
        !attention_fp32.is_contiguous() ||
        attention_fp32.numel() != static_cast<std::int64_t>(TextConfig::query_size) * tokens) {
        return false;
    }
    const std::size_t required = ops::r9700::linear::a8q4g64_activation_workspace_capacity_bytes(
        static_cast<std::uint32_t>(tokens), TextConfig::query_size);
    if (required == 0U || impl_->activation.data == nullptr || impl_->activation.bytes < required) {
        throw std::invalid_argument("R9700 attention gated-output activation region is too small");
    }
    HIP_CHECK(ops::r9700::linear::a8q4g64_gated_output(
        {.gate                       = static_cast<const hip_bfloat16*>(gate.data),
         .attention                  = static_cast<const float*>(attention_fp32.data),
         .weight_codes               = static_cast<const std::uint8_t*>(weight.qdata),
         .weight_code_bytes          = static_cast<std::size_t>(weight.qdata_bytes),
         .weight_scales              = static_cast<const std::uint16_t*>(weight.scales),
         .weight_scale_bytes         = static_cast<std::size_t>(weight.scale_bytes),
         .activation_workspace       = impl_->activation.data,
         .activation_workspace_bytes = required,
         .residual                   = static_cast<hip_bfloat16*>(residual.data),
         .tokens                     = static_cast<std::uint32_t>(tokens)},
        stream));
    return true;
}

bool Variant::ExecutionState::attention_q4_shared(const Tensor& hidden, const Weight& query_key,
                                                  const Weight& gate_value,
                                                  Tensor& query_key_output,
                                                  Tensor& gate_value_output, hipStream_t stream) {
    constexpr std::int32_t kRows = 7168;
    const std::int32_t tokens    = hidden.ne[1];
    if (impl_ == nullptr || ops::r9700::linear::kQ4ActivationBits != 8 || tokens <= 0 ||
        hidden.dtype != DType::BF16 || !hidden.is_contiguous() ||
        hidden.ne[0] != TextConfig::hidden || hidden.ne[2] != 1 || hidden.ne[3] != 1 ||
        !gdn_q4_input_weight(query_key, kRows) || !gdn_q4_input_weight(gate_value, kRows) ||
        !gdn_prefill_rows(query_key_output, kRows, tokens) ||
        !gdn_prefill_rows(gate_value_output, kRows, tokens) ||
        !ops::r9700::linear::a8q4g64_shared_activation_supported(static_cast<std::uint32_t>(tokens),
                                                                 TextConfig::hidden, kRows)) {
        return false;
    }
    const std::size_t required = ops::r9700::linear::a8q4g64_activation_workspace_capacity_bytes(
        static_cast<std::uint32_t>(tokens), TextConfig::hidden);
    if (required == 0U || impl_->activation.data == nullptr || impl_->activation.bytes < required) {
        throw std::invalid_argument("R9700 attention shared activation region is too small");
    }
    HIP_CHECK(ops::r9700::linear::a8q4g64_shared_activation_linear(
        {.input                      = static_cast<const hip_bfloat16*>(hidden.data),
         .activation_workspace       = impl_->activation.data,
         .activation_workspace_bytes = required,
         .projections                = {gdn_q4_projection(query_key, 0, kRows, query_key_output),
                                        gdn_q4_projection(gate_value, 0, kRows, gate_value_output)},
         .projection_count           = 2,
         .tokens                     = static_cast<std::uint32_t>(tokens),
         .columns                    = TextConfig::hidden},
        stream));
    return true;
}

bool Variant::ExecutionState::gdn_q4_normalized_prefill(const Tensor& residual, const Tensor& norm,
                                                        float eps, const Weight& query_key,
                                                        const Weight& value_z, Tensor& normalized,
                                                        Tensor& query_key_output,
                                                        Tensor& value_output, Tensor& gate_output,
                                                        hipStream_t stream) {
    constexpr std::int32_t kQueryKeyRows = 2 * TextConfig::key_dim;
    constexpr std::int32_t kValueRows    = TextConfig::value_dim;
    const std::int32_t tokens            = residual.ne[1];
    const auto aligned = [&](const Tensor& tensor, std::int32_t rows, std::int32_t columns) {
        return gdn_prefill_rows(tensor, rows, columns) &&
               reinterpret_cast<std::uintptr_t>(tensor.data) % 16U == 0U;
    };
    const auto shared = [&](std::int32_t rows) {
        return ops::r9700::linear::a8q4g64_shared_activation_supported(
            static_cast<std::uint32_t>(tokens), TextConfig::hidden,
            static_cast<std::uint32_t>(rows));
    };
    if (impl_ == nullptr || ops::r9700::linear::kQ4ActivationBits != 8 || tokens <= 0 ||
        !gdn_q4_input_weight(query_key, kQueryKeyRows) ||
        !gdn_q4_input_weight(value_z, 2 * kValueRows) ||
        !aligned(residual, TextConfig::hidden, tokens) || !aligned(norm, TextConfig::hidden, 1) ||
        !aligned(normalized, TextConfig::hidden, tokens) ||
        !gdn_prefill_rows(query_key_output, kQueryKeyRows, tokens) ||
        !gdn_prefill_rows(value_output, kValueRows, tokens) ||
        !gdn_prefill_rows(gate_output, kValueRows, tokens) || !shared(kQueryKeyRows) ||
        !shared(kValueRows)) {
        return false;
    }
    const std::size_t required = ops::r9700::linear::a8q4g64_activation_workspace_capacity_bytes(
        static_cast<std::uint32_t>(tokens), TextConfig::hidden);
    if (required == 0U || impl_->activation.data == nullptr || impl_->activation.bytes < required) {
        throw std::invalid_argument("R9700 GDN normalized prefill activation region is too small");
    }
    // Stream capture keeps the whole front on the captured stream.
    hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
    HIP_CHECK(hipStreamIsCapturing(stream, &capture));
    const ops::r9700::linear::A8Q4G64SideProjection side{impl_->side_stream, impl_->gate_fork,
                                                         impl_->gate_join};
    const bool concurrent = capture == hipStreamCaptureStatusNone;
    join_gdn_gate(stream);
    HIP_CHECK(ops::r9700::linear::a8q4g64_normalized_shared_activation_linear(
        {.input                      = static_cast<const hip_bfloat16*>(residual.data),
         .activation_workspace       = impl_->activation.data,
         .activation_workspace_bytes = required,
         .projections      = {gdn_q4_projection(query_key, 0, kQueryKeyRows, query_key_output),
                              gdn_q4_projection(value_z, 0, kValueRows, value_output),
                              gdn_q4_projection(value_z, kValueRows, kValueRows, gate_output)},
         .projection_count = 3,
         .tokens           = static_cast<std::uint32_t>(tokens),
         .columns          = TextConfig::hidden},
        static_cast<const hip_bfloat16*>(norm.data), eps, true,
        static_cast<hip_bfloat16*>(normalized.data), stream, concurrent ? &side : nullptr));
    impl_->gate_pending = concurrent;
    return true;
}

void Variant::ExecutionState::join_gdn_gate(hipStream_t stream) {
    if (impl_ == nullptr || !impl_->gate_pending) return;
    HIP_CHECK(hipStreamWaitEvent(stream, impl_->gate_join, 0));
    impl_->gate_pending = false;
}

bool Variant::ExecutionState::gdn_q4_pair_c2c4(const Tensor& input, const Weight& query_key,
                                               const Weight& value_z, Tensor& query_key_output,
                                               Tensor& value_z_output, hipStream_t stream) {
    if (input.ne[1] <= 0 || input.ne[2] <= 0 || input.ne[3] != 1 ||
        input.ne[1] > std::numeric_limits<std::int32_t>::max() / input.ne[2] ||
        !q4_pair_c2c4_selected(ops::r9700::linear::kQ4ActivationBits,
                               static_cast<std::uint32_t>(input.ne[1] * input.ne[2]),
                               query_key.qtype, value_z.qtype)) {
        return false;
    }
    if (impl_ == nullptr) {
        throw std::invalid_argument("R9700 GDN paired projection execution state is absent");
    }
    const auto tokens          = static_cast<std::uint32_t>(input.ne[1] * input.ne[2]);
    const std::size_t required = ops::linear_workspace_capacity_bytes(
        QType::Q4G64_F16S, static_cast<std::int32_t>(tokens), TextConfig::hidden);
    if (required == 0U || impl_->activation.data == nullptr || impl_->activation.bytes < required) {
        throw std::invalid_argument("R9700 GDN paired projection activation region is too small");
    }
    ops::gdn_input_projection_decode(input, query_key, value_z, query_key_output, value_z_output,
                                     {impl_->activation.data, required}, stream);
    return true;
}

bool Variant::ExecutionState::attention_q4_pair_c2c4(const Tensor& hidden, const Weight& query_key,
                                                     const Weight& gate_value, Tensor& query,
                                                     Tensor& key, Tensor& gate, Tensor& value,
                                                     hipStream_t stream) {
    if (hidden.ne[1] <= 0 || hidden.ne[2] <= 0 || hidden.ne[3] != 1 ||
        hidden.ne[1] > std::numeric_limits<std::int32_t>::max() / hidden.ne[2] ||
        !q4_pair_c2c4_selected(ops::r9700::linear::kQ4ActivationBits,
                               static_cast<std::uint32_t>(hidden.ne[1] * hidden.ne[2]),
                               query_key.qtype, gate_value.qtype)) {
        return false;
    }
    if (impl_ == nullptr) {
        throw std::invalid_argument("R9700 attention paired projection execution state is absent");
    }
    const auto tokens          = static_cast<std::uint32_t>(hidden.ne[1] * hidden.ne[2]);
    const std::size_t required = ops::linear_workspace_capacity_bytes(
        QType::Q4G64_F16S, static_cast<std::int32_t>(tokens), TextConfig::hidden);
    if (required == 0U || impl_->activation.data == nullptr || impl_->activation.bytes < required) {
        throw std::invalid_argument(
            "R9700 attention paired projection activation region is too small");
    }
    ops::full_attention_projection_decode(hidden, query_key, gate_value, query, key, gate, value,
                                          {impl_->activation.data, required}, stream);
    return true;
}

bool Variant::ExecutionState::projected_residual_t1(const Tensor& input, const Weight& weight,
                                                    Tensor& residual, qwen3::TextPhase phase,
                                                    bool ordinary_decode, hipStream_t stream) {
    constexpr std::int32_t kRows = TextConfig::hidden;
    const bool exact_shape =
        input.ne[0] == weight.k && input.ne[1] == 1 && input.ne[2] == 1 && input.ne[3] == 1 &&
        residual.ne[0] == kRows && residual.ne[1] == 1 && residual.ne[2] == 1 &&
        residual.ne[3] == 1 && weight.n == kRows &&
        (weight.k == TextConfig::query_size || weight.k == TextConfig::intermediate);
    if (impl_ == nullptr || !exact_shape ||
        !projected_residual_t1_selected(ops::r9700::linear::kQ4ActivationBits, phase,
                                        ordinary_decode, 1U, static_cast<std::uint32_t>(weight.n),
                                        static_cast<std::uint32_t>(weight.k), weight.qtype)) {
        return false;
    }
    const std::size_t required = ops::projected_residual_t1_workspace_capacity_bytes(weight.k);
    if (required == 0U || impl_->activation.data == nullptr || impl_->activation.bytes < required) {
        throw std::invalid_argument("R9700 projected-residual T1 activation region is too small");
    }
    ops::projected_residual_t1(input, weight, residual, {impl_->activation.data, required}, stream);
    return true;
}

bool Variant::ExecutionState::normalized_linear_t1(const Tensor& input, const Tensor& norm,
                                                   float eps, const Weight& weight, Tensor& output,
                                                   qwen3::TextPhase phase, bool ordinary_decode,
                                                   std::int32_t text_layer, hipStream_t stream) {
    const bool exact_shape = input.ne[0] == TextConfig::hidden && input.ne[1] == 1 &&
                             input.ne[2] == 1 && input.ne[3] == 1 &&
                             output.ne[0] == 2 * TextConfig::intermediate && output.ne[1] == 1 &&
                             output.ne[2] == 1 && output.ne[3] == 1;
    if (impl_ == nullptr || !exact_shape ||
        !normalized_linear_t1_selected(ops::r9700::linear::kQ4ActivationBits, phase,
                                       ordinary_decode, text_layer, 1U,
                                       static_cast<std::uint32_t>(weight.n),
                                       static_cast<std::uint32_t>(weight.k), weight.qtype)) {
        return false;
    }
    const std::size_t required = ops::normalized_linear_workspace_capacity_bytes(
        1, TextConfig::hidden, 2 * TextConfig::intermediate);
    if (required == 0U || impl_->activation.data == nullptr || impl_->activation.bytes < required) {
        throw std::invalid_argument("R9700 normalized-linear T1 activation region is too small");
    }
    ops::normalized_linear(input, norm, eps, true, weight, output,
                           {impl_->activation.data, required}, stream);
    return true;
}

bool Variant::ExecutionState::gated_normalized_output(const Tensor& recurrent_output,
                                                      const Tensor& norm, const Tensor& gate,
                                                      float eps, const Weight& weight,
                                                      Tensor& residual, std::int32_t text_layer,
                                                      hipStream_t stream) {
    const std::int32_t tokens = recurrent_output.ne[2];
    // The fused prepare reads rows, gates, and gains as 16-byte vectors.
    const auto contiguous_bf16 = [](const Tensor& tensor) {
        return tensor.dtype == DType::BF16 && tensor.data != nullptr && tensor.is_contiguous() &&
               reinterpret_cast<std::uintptr_t>(tensor.data) % 16U == 0U;
    };
    if (impl_ == nullptr || ops::r9700::linear::kQ4ActivationBits != 8 || text_layer < 0 ||
        text_layer >= TextConfig::layers || tokens <= 0 ||
        !ops::r9700::linear::a8q4g64_gated_normalized_linear_supported(
            static_cast<std::uint32_t>(tokens)) ||
        weight.qtype != QType::Q4G64_F16S || weight.layout != QuantLayout::Q4N16K16 ||
        weight.n != TextConfig::hidden || weight.k != TextConfig::value_dim ||
        weight.padded_shape[1] != TextConfig::value_dim || weight.group != 64 ||
        weight.scale_dtype != DType::FP16 || weight.qhigh != nullptr ||
        !contiguous_bf16(recurrent_output) || !contiguous_bf16(gate) || !contiguous_bf16(norm) ||
        recurrent_output.ne[0] != TextConfig::gdn_value_head_dim ||
        recurrent_output.ne[1] != TextConfig::gdn_value_heads || recurrent_output.ne[3] != 1 ||
        gate.ne[0] * gate.ne[1] != TextConfig::value_dim || gate.ne[2] != tokens ||
        norm.ne[0] != TextConfig::gdn_value_head_dim || !contiguous_bf16(residual) ||
        residual.ne[0] != TextConfig::hidden || residual.ne[1] != tokens) {
        return false;
    }
    const std::size_t required = ops::r9700::linear::a8q4g64_activation_workspace_capacity_bytes(
        static_cast<std::uint32_t>(tokens), TextConfig::value_dim);
    if (required == 0U || impl_->activation.data == nullptr || impl_->activation.bytes < required) {
        throw std::invalid_argument("R9700 gated output activation region is too small");
    }
    HIP_CHECK(ops::r9700::linear::a8q4g64_gated_normalized_linear(
        {.input                      = static_cast<const hip_bfloat16*>(recurrent_output.data),
         .weight_codes               = static_cast<const std::uint8_t*>(weight.qdata),
         .weight_code_bytes          = static_cast<std::size_t>(weight.qdata_bytes),
         .weight_scales              = static_cast<const std::uint16_t*>(weight.scales),
         .weight_scale_bytes         = static_cast<std::size_t>(weight.scale_bytes),
         .activation_workspace       = impl_->activation.data,
         .activation_workspace_bytes = required,
         .output                     = static_cast<hip_bfloat16*>(residual.data),
         .tokens                     = static_cast<std::uint32_t>(tokens),
         .rows                       = TextConfig::hidden,
         .columns                    = TextConfig::value_dim,
         .padded_columns             = TextConfig::value_dim},
        static_cast<const hip_bfloat16*>(norm.data), static_cast<const hip_bfloat16*>(gate.data),
        eps, stream,
        impl_->gated_status_cleared_layer == text_layer &&
            impl_->gated_status_cleared_tokens == tokens));
    impl_->gated_status_cleared_tokens = 0;
    return true;
}

bool Variant::ExecutionState::projected_residual_batched(const Tensor& input, const Weight& weight,
                                                         Tensor& residual, qwen3::TextPhase phase,
                                                         std::int32_t text_layer,
                                                         hipStream_t stream) {
    const std::int32_t tokens = input.ne[1];
    (void)phase;
    if (impl_ == nullptr || ops::r9700::linear::kQ4ActivationBits != 8 || text_layer < 0 ||
        text_layer >= TextConfig::layers || tokens < 2 || weight.qtype != QType::Q4G64_F16S ||
        weight.layout != QuantLayout::Q4N16K16 || weight.group != 64 ||
        weight.scale_dtype != DType::FP16 || weight.qhigh != nullptr || weight.ndim != 2 ||
        weight.n != TextConfig::hidden || weight.padded_shape[1] != weight.k ||
        input.dtype != DType::BF16 || !input.is_contiguous() || input.ne[0] != weight.k ||
        input.ne[2] != 1 || input.ne[3] != 1 || residual.dtype != DType::BF16 ||
        !residual.is_contiguous() || residual.ne[0] != TextConfig::hidden ||
        residual.ne[1] != tokens || residual.ne[2] != 1 || residual.ne[3] != 1 ||
        !ops::r9700::linear::a8q4g64_projected_residual_supported(
            static_cast<std::uint32_t>(tokens), TextConfig::hidden,
            static_cast<std::uint32_t>(weight.k))) {
        return false;
    }
    const std::size_t required = ops::r9700::linear::a8q4g64_activation_workspace_capacity_bytes(
        static_cast<std::uint32_t>(tokens), static_cast<std::uint32_t>(weight.k));
    if (required == 0U || impl_->activation.data == nullptr || impl_->activation.bytes < required) {
        throw std::invalid_argument("R9700 projected-residual activation region is too small");
    }
    HIP_CHECK(ops::r9700::linear::a8q4g64_projected_residual(
        {.input                      = static_cast<const hip_bfloat16*>(input.data),
         .weight_codes               = static_cast<const std::uint8_t*>(weight.qdata),
         .weight_code_bytes          = static_cast<std::size_t>(weight.qdata_bytes),
         .weight_scales              = static_cast<const std::uint16_t*>(weight.scales),
         .weight_scale_bytes         = static_cast<std::size_t>(weight.scale_bytes),
         .activation_workspace       = impl_->activation.data,
         .activation_workspace_bytes = required,
         .output                     = static_cast<hip_bfloat16*>(residual.data),
         .tokens                     = static_cast<std::uint32_t>(tokens),
         .rows                       = TextConfig::hidden,
         .columns                    = static_cast<std::uint32_t>(weight.k),
         .padded_columns             = static_cast<std::uint32_t>(weight.k)},
        stream));
    return true;
}

bool Variant::ExecutionState::normalized_linear_batched(const Tensor& input, const Tensor& norm,
                                                        float eps, const Weight& weight,
                                                        Tensor& output, std::int32_t text_layer,
                                                        hipStream_t stream) {
    const std::int32_t tokens = input.ne[1];
    if (impl_ == nullptr || ops::r9700::linear::kQ4ActivationBits != 8 || text_layer < 0 ||
        text_layer >= TextConfig::layers || weight.qtype != QType::Q4G64_F16S || tokens <= 0 ||
        !ops::r9700::linear::a8q4g64_normalized_linear_batched_supported(
            static_cast<std::uint32_t>(tokens)) ||
        input.ne[0] != TextConfig::hidden || input.ne[2] != 1 || input.ne[3] != 1 ||
        output.ne[0] != 2 * TextConfig::intermediate || output.ne[1] != tokens ||
        reinterpret_cast<std::uintptr_t>(input.data) % 16U != 0U ||
        reinterpret_cast<std::uintptr_t>(norm.data) % 16U != 0U) {
        return false;
    }
    const std::size_t required = ops::normalized_linear_workspace_capacity_bytes(
        tokens, TextConfig::hidden, 2 * TextConfig::intermediate);
    if (required == 0U || impl_->activation.data == nullptr || impl_->activation.bytes < required) {
        throw std::invalid_argument(
            "R9700 batched normalized-linear activation region is too small");
    }
    ops::normalized_linear(input, norm, eps, true, weight, output,
                           {impl_->activation.data, required}, stream);
    return true;
}

Variant::ExecutionState::~ExecutionState() = default;

std::vector<std::uint32_t>
Variant::ExecutionState::eager_widths(std::uint32_t prefill_tokens,
                                      std::uint32_t maximum_concurrency,
                                      std::span<const std::uint32_t> verify_widths) {
    if (prefill_tokens == 0U || maximum_concurrency == 0U) {
        throw std::invalid_argument("R9700 linear execution widths must be positive");
    }
    std::vector<std::uint32_t> widths;
    widths.reserve(static_cast<std::size_t>(maximum_concurrency) * (verify_widths.size() + 1U) +
                   1U);
    const auto append_product = [&](std::uint32_t batch, std::uint32_t width, const char* label) {
        if (width == 0U) return;
        const std::uint64_t product = static_cast<std::uint64_t>(batch) * width;
        if (product > std::numeric_limits<std::uint32_t>::max()) {
            throw std::overflow_error(label);
        }
        widths.push_back(static_cast<std::uint32_t>(product));
    };
    for (std::uint32_t batch = 1; batch <= maximum_concurrency; ++batch) {
        widths.push_back(batch); // Ordinary decode and one-column speculative substeps.
        for (const auto width : verify_widths)
            append_product(batch, width, "R9700 speculative linear width overflows");
    }
    widths.push_back(prefill_tokens);
    std::sort(widths.begin(), widths.end());
    widths.erase(std::unique(widths.begin(), widths.end()), widths.end());
    return widths;
}

bool Variant::ExecutionState::run_shared(SelectedLinearRole first_role,
                                         SelectedLinearRole second_role, std::int32_t text_layer,
                                         const Tensor& input, const Weight& first,
                                         Tensor& first_output, const Weight& second,
                                         Tensor& second_output, hipStream_t stream) {
    if (first.qtype != QType::F8E4M3_ROW_F32S || second.qtype != QType::F8E4M3_ROW_F32S ||
        impl_ == nullptr || first.k != second.k || input.ne[1] <= 0) {
        return false;
    }
    Impl::Slot& second_slot = impl_->slots[Impl::index(second_role, text_layer)];
    Impl::Slot& first_slot  = impl_->slots[Impl::index(first_role, text_layer)];
    if (second_slot.execution == nullptr || second_slot.weight != &second ||
        first_slot.execution == nullptr) {
        throw std::invalid_argument(
            "R9700 FP8 selected projection binding differs from Program state");
    }
    const auto tokens = static_cast<std::uint32_t>(input.ne[1]);
    if (!run(first_role, text_layer, input, first, first_output, stream)) return false;
    const auto first_activation  = first_slot.execution->activation_workspace(tokens);
    const auto second_activation = second_slot.execution->activation_workspace(tokens);
    if (second_output.dtype != DType::BF16 || second_output.data == nullptr ||
        !second_output.is_contiguous() || second_output.ne[0] != second.n ||
        second_output.ne[1] != input.ne[1] || !first_activation || !second_activation ||
        first_activation->codes != second_activation->codes ||
        first_activation->scales != second_activation->scales ||
        first_activation->status != second_activation->status) {
        // Not provably the same activation image: quantize again.
        return run(second_role, text_layer, input, second, second_output, stream);
    }
    HIP_CHECK(second_slot.execution->run_quantized(
        tokens, static_cast<hip_bfloat16*>(second_output.data), stream));
    return true;
}

bool Variant::ExecutionState::run(SelectedLinearRole role, std::int32_t text_layer,
                                  const Tensor& input, const Weight& weight, Tensor& output,
                                  hipStream_t stream) {
    if (weight.qtype != QType::F8E4M3_ROW_F32S) return false;
    if (impl_ == nullptr) { throw std::logic_error("R9700 FP8 execution state is empty"); }
    Impl::Slot& slot = impl_->slots[Impl::index(role, text_layer)];
    // LinearExecution retains the exact qdata/scale pointers and geometry of the materialized
    // Weight object installed with this role.  Sharing only its enclosing payload span is not a
    // sufficient binding identity: a copied or altered view could otherwise select an execution
    // prepared for different offsets or dimensions.
    if (slot.execution == nullptr || slot.weight != &weight) {
        throw std::invalid_argument(
            "R9700 FP8 selected projection binding differs from Program state");
    }
    if (input.dtype != DType::BF16 || output.dtype != DType::BF16 || input.data == nullptr ||
        output.data == nullptr || !input.is_contiguous() || !output.is_contiguous() ||
        input.ne[0] != weight.k || output.ne[0] != weight.n || input.ne[1] <= 0 ||
        output.ne[1] != input.ne[1] || input.ne[2] != 1 || input.ne[3] != 1 || output.ne[2] != 1 ||
        output.ne[3] != 1) {
        throw std::invalid_argument("R9700 FP8 selected projection tensor geometry differs");
    }
    const auto tokens = static_cast<std::uint32_t>(input.ne[1]);
    HIP_CHECK(slot.execution->run(tokens, static_cast<const hip_bfloat16*>(input.data),
                                  static_cast<hip_bfloat16*>(output.data), stream));
    return true;
}

std::optional<ops::r9700::linear::Fp8ActivationWorkspace>
Variant::ExecutionState::fp8_small_activation(SelectedLinearRole role, std::int32_t text_layer,
                                              const Weight& weight, std::uint32_t tokens) const {
    if (impl_ == nullptr || weight.qtype != QType::F8E4M3_ROW_F32S || tokens == 0U ||
        tokens > ops::r9700::linear::kFp8SmallTokenLimit ||
        !ops::r9700::linear::fp8_small_t_supported(
            tokens, static_cast<std::uint32_t>(weight.n), static_cast<std::uint32_t>(weight.k),
            static_cast<std::uint32_t>(weight.padded_shape[1]))) {
        return std::nullopt;
    }
    const Impl::Slot& slot = impl_->slots[Impl::index(role, text_layer)];
    if (slot.execution == nullptr || slot.weight != &weight) {
        throw std::invalid_argument(
            "R9700 FP8 selected projection binding differs from Program state");
    }
    return slot.execution->activation_workspace(tokens);
}

namespace {

ops::r9700::linear::Fp8RowScaledWeight fp8_rows(const Weight& weight) {
    return {static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const float*>(weight.scales), static_cast<std::uint32_t>(weight.n)};
}

bool bf16_tokens(const Tensor& tensor, std::int32_t features, std::int32_t tokens) {
    return tensor.dtype == DType::BF16 && tensor.data != nullptr && tensor.is_contiguous() &&
           tensor.numel() == static_cast<std::int64_t>(features) * tokens;
}

} // namespace

bool Variant::ExecutionState::attention_fp8_normalized_projection(
    const Tensor& residual, const Tensor& norm, float eps, const Weight& query_key,
    const Weight& gate_value, Tensor& query, Tensor& key, Tensor& gate, Tensor& value,
    std::int32_t text_layer, hipStream_t stream) {
    const std::int32_t tokens = residual.ne[1];
    if (tokens <= 0 || residual.ne[0] != TextConfig::hidden || residual.ne[2] != 1 ||
        residual.ne[3] != 1 || !bf16_tokens(residual, TextConfig::hidden, tokens) ||
        norm.dtype != DType::BF16 || norm.data == nullptr || norm.numel() != TextConfig::hidden ||
        query_key.k != TextConfig::hidden || gate_value.k != TextConfig::hidden ||
        query_key.n != TextConfig::query_size + TextConfig::kv_size ||
        gate_value.n != query_key.n || !bf16_tokens(query, TextConfig::query_size, tokens) ||
        !bf16_tokens(gate, TextConfig::query_size, tokens) ||
        !bf16_tokens(key, TextConfig::kv_size, tokens) ||
        !bf16_tokens(value, TextConfig::kv_size, tokens)) {
        return false;
    }
    const auto width = static_cast<std::uint32_t>(tokens);
    const auto first =
        fp8_small_activation(SelectedLinearRole::AttentionQueryKey, text_layer, query_key, width);
    const auto second = first ? fp8_small_activation(SelectedLinearRole::AttentionGateValue,
                                                     text_layer, gate_value, width)
                              : std::nullopt;
    if (!second || first->codes != second->codes || first->scales != second->scales ||
        first->status != second->status) {
        return false;
    }
    HIP_CHECK(ops::r9700::linear::fp8_quantize_normalized_activation(
        {.input       = static_cast<const hip_bfloat16*>(residual.data),
         .weight      = static_cast<const hip_bfloat16*>(norm.data),
         .eps         = eps,
         .unit_offset = true,
         .workspace   = *first,
         .warm        = producer_warm(query_key, width)},
        stream));
    HIP_CHECK(ops::r9700::linear::fp8_small_t_pair_split(
        {.first           = fp8_rows(query_key),
         .second          = fp8_rows(gate_value),
         .split           = static_cast<std::uint32_t>(TextConfig::query_size),
         .first_leading   = static_cast<hip_bfloat16*>(query.data),
         .first_trailing  = static_cast<hip_bfloat16*>(key.data),
         .second_leading  = static_cast<hip_bfloat16*>(gate.data),
         .second_trailing = static_cast<hip_bfloat16*>(value.data)},
        *first, stream));
    return true;
}

bool Variant::ExecutionState::attention_fp8_gated_output(const Tensor& gate,
                                                         const Tensor& attention_fp32,
                                                         const Weight& weight, Tensor& residual,
                                                         std::int32_t text_layer,
                                                         hipStream_t stream) {
    const std::int32_t tokens = residual.ne[1];
    if (tokens <= 0 || residual.ne[0] != TextConfig::hidden || residual.ne[2] != 1 ||
        residual.ne[3] != 1 || !bf16_tokens(residual, TextConfig::hidden, tokens) ||
        !bf16_tokens(gate, TextConfig::query_size, tokens) || attention_fp32.dtype != DType::FP32 ||
        attention_fp32.data == nullptr || !attention_fp32.is_contiguous() ||
        attention_fp32.numel() != static_cast<std::int64_t>(TextConfig::query_size) * tokens ||
        weight.k != TextConfig::query_size || weight.n != TextConfig::hidden) {
        return false;
    }
    const auto activation = fp8_small_activation(SelectedLinearRole::AttentionOutput, text_layer,
                                                 weight, static_cast<std::uint32_t>(tokens));
    if (!activation) return false;
    HIP_CHECK(ops::r9700::linear::fp8_quantize_gated_activation(
        {.gate      = static_cast<const hip_bfloat16*>(gate.data),
         .attention = static_cast<const float*>(attention_fp32.data),
         .workspace = *activation,
         .warm      = producer_warm(weight, static_cast<std::uint32_t>(tokens))},
        stream));
    HIP_CHECK(ops::r9700::linear::fp8_small_t_residual(
        fp8_rows(weight), *activation, static_cast<hip_bfloat16*>(residual.data), stream));
    return true;
}

std::size_t Variant::ExecutionState::selected_count() const noexcept {
    return impl_ == nullptr ? 0U : impl_->selected;
}

std::vector<GraphExecutionProfile> Variant::ordinary_graph_profiles(std::uint32_t capacity) {
    if (capacity == 0) { return {}; }
    std::vector<GraphExecutionProfile> profiles =
        graph_profiles_through(capacity - 1, {127, 511, 2047, 4095, 8190, 8191, 16383, 32767});
    for (GraphExecutionProfile& profile : profiles) {
        const std::size_t maximum_visible = static_cast<std::size_t>(profile.max) + 1U;
        profile.topology_class =
            ops::r9700::kv::use_packed_decode_attention(1U, maximum_visible, false) ? 3U : 0U;
    }
    return profiles;
}

std::vector<GraphExecutionProfile> Variant::mtp_graph_profiles(std::uint32_t capacity,
                                                               std::uint32_t draft_window) {
    if (draft_window == 0 || capacity == 0) { return {}; }
    std::vector<std::uint32_t> ends;
    for (const std::uint32_t visible_end : {128U, 512U, 2048U, 4096U, 8192U, 16384U, 32768U}) {
        const std::uint32_t mtp_offset = 2U * draft_window;
        if (visible_end == ops::r9700::kv::kSplit512MinimumContext && draft_window == 3U) {
            // One captured MTP round contains two fixed-width T=4 attention leaves. The MTP-cache
            // leaf (device-selected rows) can see max+2K keys and splits at 8192; target
            // verification sees max+(K+1) on the packed route. Their transitions require distinct
            // profile boundaries and topology bits.
            const std::uint32_t text_offset = draft_window + 1U;
            ends.push_back(visible_end - 1U - mtp_offset);
            ends.push_back(visible_end - mtp_offset);
            ends.push_back(visible_end - 1U - text_offset);
            ends.push_back(visible_end - text_offset);
            continue;
        }
        if (visible_end >= mtp_offset) { ends.push_back(visible_end - mtp_offset); }
    }
    std::vector<GraphExecutionProfile> profiles = graph_profiles_through(capacity - 1, ends);
    const std::uint32_t width                   = draft_window + 1U;
    for (GraphExecutionProfile& profile : profiles) {
        const std::size_t maximum_visible =
            static_cast<std::size_t>(profile.max) + std::size_t{2} * draft_window;
        const std::size_t maximum_text_visible =
            static_cast<std::size_t>(profile.max) + draft_window + 1U;
        profile.topology_class =
            (ops::r9700::kv::use_split512_attention(width, maximum_visible, true) ? 1U : 0U) |
            (ops::r9700::kv::use_packed_decode_attention(width, maximum_text_visible, false) ? 2U
                                                                                             : 0U);
    }
    return profiles;
}

std::vector<GraphExecutionProfile> Variant::dflash_graph_profiles(std::uint32_t capacity,
                                                                  std::uint32_t draft_window,
                                                                  std::uint32_t,
                                                                  std::uint32_t verify_width) {
    if (draft_window == 0 || capacity == 0) { return {}; }
    const std::uint32_t block = verify_width != 0 ? verify_width : draft_window + 1;
    std::vector<std::uint32_t> ends;
    for (const std::uint32_t visible_end :
         {128U, 512U, 2048U, 4096U, 8192U, 16384U, 32768U, 65536U, 131072U, 262144U}) {
        if ((visible_end == 4096U || visible_end == 8192U) && visible_end > block) {
            ends.push_back(visible_end - 1U - block);
        }
        if (visible_end >= block) { ends.push_back(visible_end - block); }
    }
    std::vector<GraphExecutionProfile> profiles = graph_profiles_through(capacity - 1, ends);
    for (GraphExecutionProfile& profile : profiles) {
        const std::size_t maximum_visible =
            std::min<std::size_t>(capacity, static_cast<std::size_t>(profile.max) + block);
        // Match the launcher's packed-first precedence: the packed decode route (chunk kernel and
        // merge) is its own executable topology; split512 serves tree/device-count T=4.
        constexpr auto planes = qwen3::detail::kR9700TextKVPlaneLayouts;
        const bool packed =
            qwen3::detail::kR9700TextKVValueGroup == 16 &&
            planes.key == Fp8KInt4VPlaneLayout::TokenFastestHeadMajor &&
            planes.value == Fp8KInt4VPlaneLayout::FeatureFastestPageMajor &&
            planes.value_scale == Fp8KInt4VPlaneLayout::FeatureFastestPageMajor &&
            ops::r9700::kv::use_packed_decode_attention(block, maximum_visible, false);
        profile.topology_class =
            packed
                ? 3U
                : (ops::r9700::kv::use_split512_attention(block, maximum_visible, true) ? 2U : 0U);
    }
    return profiles;
}

std::optional<std::size_t> Variant::calibrated_dflash_graph_allowance_bytes(
    WeightsProfile profile, qwen3::StartupFeatures features, std::uint32_t capacity,
    std::uint32_t concurrency, std::uint32_t prefill_chunk, std::uint32_t draft_window,
    bool adaptive_draft) {
#if defined(NINFER_R9700_XATTENTION_QUALIFICATION)
    constexpr bool production_attention = false;
#else
    constexpr bool production_attention = true;
#endif
    constexpr auto planes = qwen3::detail::kR9700TextKVPlaneLayouts;
    const bool qualified_build =
        production_attention && qwen3::detail::kR9700TextKVValueGroup == 16 &&
        planes.key == Fp8KInt4VPlaneLayout::TokenFastestHeadMajor &&
        planes.value == Fp8KInt4VPlaneLayout::FeatureFastestPageMajor &&
        planes.value_scale == Fp8KInt4VPlaneLayout::FeatureFastestPageMajor &&
        ops::r9700::linear::kQ4ActivationBits == 8U &&
        ops::r9700::linear::kW8ActivationBits == 8U &&
        ops::r9700::linear::kQ4PrefillA4Families == 0U &&
        !ops::r9700::linear::kDFlashSmallTCandidateEnabled &&
        !ops::r9700::linear::kDFlashMlpDownT5CandidateEnabled &&
        !ops::r9700::gdn_recurrence::kVerifyWaveQkCandidate && ops::r9700::kv::kFp8QkWmmaDecode &&
        !ops::r9700::kv::kTextP129WmmaTailCandidate;
    if (!qualified_build || profile != WeightsProfile::R9700Lq4 || features.vision ||
        !features.dflash() || !features.optimized_proposal() || capacity != maximum_context ||
        concurrency == 0U || concurrency > 8U || prefill_chunk != 2048U || draft_window != 7U ||
        !adaptive_draft)
        return std::nullopt;

    // Packet replay keeps immutable kernel arguments in pinned host pools when the runtime
    // does not use large BAR. This avoids per-executable VRAM pools and per-node submission.
    const auto setting = [](const char* name, const char* expected) {
        const char* value = std::getenv(name);
        return value != nullptr && std::strcmp(value, expected) == 0;
    };
    if (setting("DEBUG_HIP_GRAPH_CLASSIC_PATH", "0") && setting("ROC_ENABLE_LARGE_BAR", "0") &&
        setting("ROC_SKIP_KERNEL_ARG_COPY", "0") &&
        setting("HSA_DISABLE_FRAGMENT_ALLOCATOR", "1")) {
        // Shared run images and the complete C1..8 inventory on HIP 7.15.26333.
        // C1 remeasured after decode hosting and host-allocation changes.
        // profiles/bench/readme-refresh-20261017/
        constexpr std::array<std::size_t, 8> host_argument_graph_mib{72, 56, 60, 60,
                                                                     62, 62, 62, 62};
        return host_argument_graph_mib[concurrency - 1U] << 20U;
    }

    // Complete immutable inventory, measured separately at every C on gfx1201.
    // profiles/bench/r9700-vram-runtime-budget-20261015/
    constexpr std::array<std::size_t, 8> graph_mib{318, 578, 842, 1102, 1364, 1624, 1884, 2144};
    return graph_mib[concurrency - 1U] << 20U;
}

void Variant::attention_projection(const Tensor& hidden,
                                   const FullAttentionProjectionWeights& weights, Tensor& query,
                                   Tensor& gate, Tensor& key, Tensor& value, qwen3::TextPhase,
                                   WorkspaceArena& workspace, hipStream_t stream, std::int32_t,
                                   ExecutionState* execution, std::int32_t text_layer) {
    if (execution != nullptr &&
        execution->attention_q4_pair_t1(hidden, weights.query_key, weights.gate_value, query, key,
                                        gate, value, stream)) {
        return;
    }
    if (execution != nullptr &&
        execution->attention_q4_pair_c2c4(hidden, weights.query_key, weights.gate_value, query, key,
                                          gate, value, stream)) {
        return;
    }
    if (execution != nullptr) {
        const ExecutionState::Lq4Target targets[] = {{weights.query_key, query, &key, false},
                                                     {weights.gate_value, gate, &value, false}};
        if (execution->lq4_projections(hidden, targets, stream)) return;
    }
    if (execution != nullptr && weights.query_key.qtype == QType::F8E4M3_ROW_F32S &&
        weights.gate_value.qtype == QType::F8E4M3_ROW_F32S) {
        auto scope        = workspace.scope();
        Tensor query_key  = workspace.alloc(DType::BF16, {7168, hidden.ne[1]});
        Tensor gate_value = workspace.alloc(DType::BF16, {7168, hidden.ne[1]});
        if (execution->run_shared(SelectedLinearRole::AttentionQueryKey,
                                  SelectedLinearRole::AttentionGateValue, text_layer, hidden,
                                  weights.query_key, query_key, weights.gate_value, gate_value,
                                  stream)) {
            ops::split_bf16_columns(query_key, query, key, stream);
            ops::split_bf16_columns(gate_value, gate, value, stream);
            return;
        }
    }
    if (execution != nullptr) {
        auto scope        = workspace.scope();
        Tensor query_key  = workspace.alloc(DType::BF16, {7168, hidden.ne[1]});
        Tensor gate_value = workspace.alloc(DType::BF16, {7168, hidden.ne[1]});
        if (execution->attention_q4_shared(hidden, weights.query_key, weights.gate_value, query_key,
                                           gate_value, stream)) {
            ops::split_bf16_columns(query_key, query, key, stream);
            ops::split_bf16_columns(gate_value, gate, value, stream);
            return;
        }
    }
    {
        auto query_scope = workspace.scope();
        Tensor query_key = workspace.alloc(DType::BF16, {7168, hidden.ne[1]});
        selected_linear(execution, SelectedLinearRole::AttentionQueryKey, text_layer, hidden,
                        weights.query_key, query_key, workspace, stream);
        ops::split_bf16_columns(query_key, query, key, stream);
    }
    {
        auto gate_scope   = workspace.scope();
        Tensor gate_value = workspace.alloc(DType::BF16, {7168, hidden.ne[1]});
        selected_linear(execution, SelectedLinearRole::AttentionGateValue, text_layer, hidden,
                        weights.gate_value, gate_value, workspace, stream);
        ops::split_bf16_columns(gate_value, gate, value, stream);
    }
}

bool Variant::attention_normalized_projection(const Tensor& residual, const Tensor& norm, float eps,
                                              const FullAttentionProjectionWeights& weights,
                                              Tensor& query, Tensor& gate, Tensor& key,
                                              Tensor& value, hipStream_t stream,
                                              ExecutionState* execution, std::int32_t text_layer) {
    if (execution != nullptr) {
        const ExecutionState::Lq4Target targets[] = {{weights.query_key, query, &key, false},
                                                     {weights.gate_value, gate, &value, false}};
        if (execution->lq4_normalized_projections(residual, norm, eps, targets, stream))
            return true;
    }
    return execution != nullptr &&
           (execution->attention_fp8_normalized_projection(residual, norm, eps, weights.query_key,
                                                           weights.gate_value, query, key, gate,
                                                           value, text_layer, stream) ||
            execution->attention_q4_normalized_split(residual, norm, eps, weights.query_key,
                                                     weights.gate_value, query, key, gate, value,
                                                     stream));
}

bool Variant::attention_gated_output_projection(const Tensor& gate, const Tensor& attention_fp32,
                                                const Weight& weight, Tensor& residual,
                                                hipStream_t stream, ExecutionState* execution,
                                                std::int32_t text_layer) {
    if (execution != nullptr) {
        const ExecutionState::Lq4Target targets[] = {{weight, residual, nullptr, true}};
        if (execution->lq4_gated_projections(gate, attention_fp32, targets, stream)) return true;
    }
    return execution != nullptr &&
           (execution->attention_fp8_gated_output(gate, attention_fp32, weight, residual,
                                                  text_layer, stream) ||
            execution->attention_q4_gated_output(gate, attention_fp32, weight, residual, stream));
}

void Variant::full_attention(const Tensor& normalized_query,
                             const qwen3::PagedKVLayerRead& cache_read,
                             const Tensor& cache_positions, Tensor& attention_fp32,
                             WorkspaceArena& workspace, hipStream_t stream,
                             const Tensor* ancestor_masks, const Tensor* prefix_lengths,
                             const Tensor* active_query_rows) {
    if (cache_positions.dtype != DType::I32 || cache_positions.data == nullptr ||
        !cache_positions.is_contiguous() || cache_positions.ne[0] != normalized_query.ne[2] ||
        cache_positions.ne[1] != 1 || cache_positions.ne[2] != 1 || cache_positions.ne[3] != 1) {
        throw std::invalid_argument("R9700 full attention has invalid causal positions");
    }
    if ((ancestor_masks == nullptr) != (prefix_lengths == nullptr)) {
        throw std::invalid_argument("R9700 full attention has an incomplete tree mask pair");
    }
    const auto valid_tree_vector = [](const Tensor* tensor) {
        return tensor != nullptr && tensor->dtype == DType::I32 && tensor->data != nullptr &&
               tensor->is_contiguous() && tensor->ne[1] == 1 && tensor->ne[2] == 1 &&
               tensor->ne[3] == 1;
    };
    if (ancestor_masks != nullptr &&
        (!valid_tree_vector(ancestor_masks) || ancestor_masks->ne[0] != normalized_query.ne[2] ||
         !valid_tree_vector(prefix_lengths) ||
         (prefix_lengths->ne[0] != 1 && prefix_lengths->ne[0] != normalized_query.ne[2]))) {
        throw std::invalid_argument("R9700 full attention has invalid tree mask geometry");
    }
    if (active_query_rows != nullptr &&
        (active_query_rows->dtype != DType::I32 || active_query_rows->data == nullptr ||
         !active_query_rows->is_contiguous() || active_query_rows->ne[0] != 1 ||
         active_query_rows->ne[1] != 1 || active_query_rows->ne[2] != 1 ||
         active_query_rows->ne[3] != 1)) {
        throw std::invalid_argument("R9700 full attention has invalid active-row scalar");
    }
    auto attention_scope                        = workspace.scope();
    const std::size_t attention_workspace_bytes = r9700_full_attention_workspace_capacity_bytes(
        static_cast<std::uint32_t>(normalized_query.ne[2]), cache_read.visible_frontier(),
        ancestor_masks != nullptr);
    DeviceSpan attention_workspace{};
    if (attention_workspace_bytes != 0U) {
        attention_workspace = workspace.alloc_bytes(attention_workspace_bytes);
    }
    HIP_CHECK(r9700_qwen3_8_27b_full_attention(
        R9700FullAttentionArgs{
            .query          = normalized_query,
            .cache_read     = cache_read,
            .row_positions  = static_cast<const std::int32_t*>(cache_positions.data),
            .ancestor_masks = ancestor_masks == nullptr
                                  ? nullptr
                                  : static_cast<const std::int32_t*>(ancestor_masks->data),
            .prefix_lengths = prefix_lengths == nullptr
                                  ? nullptr
                                  : static_cast<const std::int32_t*>(prefix_lengths->data),
            .prefix_length_stride =
                prefix_lengths != nullptr && prefix_lengths->ne[0] == 1 ? 0U : 1U,
            .active_query_rows = active_query_rows == nullptr
                                     ? nullptr
                                     : static_cast<const std::int32_t*>(active_query_rows->data),
            .workspace         = attention_workspace.data,
            .workspace_bytes   = attention_workspace.bytes,
            .output            = attention_fp32,
        },
        stream));
}

std::size_t Variant::full_attention_workspace_capacity_bytes(std::int32_t maximum_query_rows,
                                                             std::uint32_t maximum_visible_context,
                                                             bool tree_or_device_count) {
    if (maximum_query_rows <= 0) { return 0U; }
    return r9700_full_attention_workspace_capacity_bytes(
        static_cast<std::uint32_t>(maximum_query_rows), maximum_visible_context,
        tree_or_device_count);
}

bool Variant::full_attention_sequences(std::span<const FullAttentionSequence> sequences,
                                       WorkspaceArena& workspace, hipStream_t stream) {
    std::array<R9700FullAttentionArgs, kMaximumConcurrency> args{};
    if (sequences.empty() || sequences.size() > args.size()) {
        throw std::invalid_argument("R9700 batched full attention has an invalid sequence count");
    }
    std::size_t stride = 0U;
    for (std::size_t i = 0; i < sequences.size(); ++i) {
        const FullAttentionSequence& sequence = sequences[i];
        const Tensor& positions               = sequence.cache_positions;
        if (positions.dtype != DType::I32 || positions.data == nullptr ||
            !positions.is_contiguous() || positions.ne[0] != sequence.query.ne[2] ||
            positions.ne[1] != 1 || positions.ne[2] != 1 || positions.ne[3] != 1) {
            throw std::invalid_argument("R9700 full attention has invalid causal positions");
        }
        args[i] = R9700FullAttentionArgs{
            .query         = sequence.query,
            .cache_read    = sequence.cache_read,
            .row_positions = static_cast<const std::int32_t*>(positions.data),
            .output        = sequence.output,
        };
        stride = std::max(stride, r9700_full_attention_sequence_workspace_stride_bytes(
                                      static_cast<std::uint32_t>(sequence.query.ne[2]),
                                      sequence.cache_read.visible_frontier()));
    }
    const std::span<const R9700FullAttentionArgs> batch(args.data(), sequences.size());
    if (!r9700_full_attention_sequences_supported(batch, stream)) return false;
    // One stable arena span: sequence i owns [i * stride, +stride).
    auto attention_scope                 = workspace.scope();
    const DeviceSpan attention_workspace = workspace.alloc_bytes(stride * sequences.size());
    for (std::size_t i = 0; i < sequences.size(); ++i) {
        args[i].workspace       = static_cast<std::byte*>(attention_workspace.data) + i * stride;
        args[i].workspace_bytes = stride;
    }
    HIP_CHECK(r9700_qwen3_8_27b_full_attention_sequences(batch, stream));
    return true;
}

std::size_t
Variant::full_attention_sequences_workspace_capacity_bytes(std::int32_t sequences,
                                                           std::int32_t maximum_query_rows,
                                                           std::uint32_t maximum_visible_context) {
    if (sequences <= 0 || maximum_query_rows <= 0) return 0U;
    return static_cast<std::size_t>(sequences) *
           r9700_full_attention_sequence_workspace_stride_bytes(
               static_cast<std::uint32_t>(maximum_query_rows), maximum_visible_context);
}

#if defined(NINFER_R9700_XATTENTION_QUALIFICATION)
void Variant::text_prefill_attention(const Tensor& normalized_query,
                                     const qwen3::PagedKVLayerRead& cache_read,
                                     const Tensor& cache_positions, Tensor& attention_fp32,
                                     WorkspaceArena& workspace, hipStream_t stream) {
    if (normalized_query.ne[2] < 2) {
        full_attention(normalized_query, cache_read, cache_positions, attention_fp32, workspace,
                       stream);
        return;
    }
    if (cache_positions.dtype != DType::I32 || cache_positions.data == nullptr ||
        !cache_positions.is_contiguous() || cache_positions.ne[0] != normalized_query.ne[2] ||
        cache_positions.ne[1] != 1 || cache_positions.ne[2] != 1 || cache_positions.ne[3] != 1) {
        throw std::invalid_argument("R9700 XAttention prefill has invalid causal positions");
    }
    auto sparse_scope       = workspace.scope();
    const std::size_t bytes = r9700_qwen3_8_27b_text_prefill_attention_workspace_capacity_bytes(
        static_cast<std::uint32_t>(normalized_query.ne[2]), cache_read.visible_frontier());
    if (bytes == 0U) {
        throw std::invalid_argument("R9700 XAttention prefill has invalid workspace geometry");
    }
    DeviceSpan sparse_workspace = workspace.alloc_bytes(bytes);
    HIP_CHECK(r9700_qwen3_8_27b_text_prefill_attention(
        R9700FullAttentionArgs{
            .query         = normalized_query,
            .cache_read    = cache_read,
            .row_positions = static_cast<const std::int32_t*>(cache_positions.data),
            .output        = attention_fp32,
        },
        sparse_workspace.data, sparse_workspace.bytes, stream));
}

std::size_t
Variant::text_prefill_attention_workspace_capacity_bytes(std::int32_t maximum_query_rows,
                                                         std::uint32_t maximum_visible_context) {
    if (maximum_query_rows <= 0) return 0U;
    // A nominal prefill envelope can still end in a one-row remainder, which follows the
    // ordinary dense T=1 leaf.  Size that possible leaf independently of the envelope's maximum
    // width; size the one-row remainder independently of any candidate-only wider route.
    const std::size_t dense =
        full_attention_workspace_capacity_bytes(1, maximum_visible_context, false);
    const std::size_t sparse = r9700_qwen3_8_27b_text_prefill_attention_workspace_capacity_bytes(
        static_cast<std::uint32_t>(maximum_query_rows), maximum_visible_context);
    return std::max(dense, sparse);
}
#endif

void Variant::attention_output_projection(const Tensor& attention, const Weight& weight,
                                          Tensor& residual, qwen3::TextPhase phase,
                                          WorkspaceArena& workspace, hipStream_t stream,
                                          std::int32_t text_layer, ExecutionState* execution,
                                          bool ordinary_decode) {
    if (execution != nullptr && (execution->projected_residual_t1(attention, weight, residual,
                                                                  phase, ordinary_decode, stream) ||
                                 execution->projected_residual_batched(
                                     attention, weight, residual, phase, text_layer, stream))) {
        return;
    }
    if (execution != nullptr) {
        const ExecutionState::Lq4Target targets[] = {{weight, residual, nullptr, true}};
        if (execution->lq4_projections(attention, targets, stream)) return;
    }
    auto scope   = workspace.scope();
    Tensor delta = workspace.alloc(DType::BF16, {TextConfig::hidden, attention.ne[1]});
    if (weight.qtype == QType::F8E4M3_ROW_F32S) {
        selected_linear(execution, SelectedLinearRole::AttentionOutput, text_layer, attention,
                        weight, delta, workspace, stream);
    } else {
        serialized_linear(execution, attention, weight, delta, workspace, stream);
    }
    ops::residual_add(delta, residual, stream);
}

void Variant::mtp_attention_projection(const Tensor& hidden,
                                       const MtpAttentionProjectionWeights& weights, Tensor& query,
                                       Tensor& gate, Tensor& key, Tensor& value,
                                       WorkspaceArena& workspace, hipStream_t stream,
                                       ExecutionState* execution) {
    serialized_linear(execution, hidden, weights.query, query, workspace, stream);
    serialized_linear(execution, hidden, weights.key, key, workspace, stream);
    serialized_linear(execution, hidden, weights.output_gate, gate, workspace, stream);
    serialized_linear(execution, hidden, weights.value, value, workspace, stream);
}

void Variant::mtp_kv_projection(const Tensor& hidden, const MtpAttentionProjectionWeights& weights,
                                Tensor& key, Tensor& value, WorkspaceArena& workspace,
                                hipStream_t stream, ExecutionState* execution) {
    serialized_linear(execution, hidden, weights.key, key, workspace, stream);
    serialized_linear(execution, hidden, weights.value, value, workspace, stream);
}

void Variant::mtp_q_gate_projection(const Tensor& hidden,
                                    const MtpAttentionProjectionWeights& weights, Tensor& query,
                                    Tensor& gate, WorkspaceArena& workspace, hipStream_t stream,
                                    ExecutionState* execution) {
    serialized_linear(execution, hidden, weights.query, query, workspace, stream);
    serialized_linear(execution, hidden, weights.output_gate, gate, workspace, stream);
}

void Variant::mtp_fc(const Tensor& embedding_norm, const Tensor& hidden_norm, const Weight& weight,
                     Tensor& residual, WorkspaceArena& workspace, hipStream_t stream, std::int32_t,
                     ExecutionState* execution) {
    auto scope = workspace.scope();
    Tensor packed =
        workspace.alloc(DType::BF16, {TextConfig::mtp_input_rows, embedding_norm.ne[1]});
    ops::mtp_pack_fc_input(embedding_norm, hidden_norm, packed, stream);
    serialized_linear(execution, packed, weight, residual, workspace, stream);
}

void Variant::mtp_attention_output(const Tensor& attention, const Weight& weight, Tensor& residual,
                                   WorkspaceArena& workspace, hipStream_t stream, std::int32_t,
                                   ExecutionState* execution) {
    auto scope   = workspace.scope();
    Tensor delta = workspace.alloc(DType::BF16, {TextConfig::hidden, attention.ne[1]});
    serialized_linear(execution, attention, weight, delta, workspace, stream);
    ops::residual_add(delta, residual, stream);
}

void Variant::gdn_input_projection(const Tensor& hidden, const GdnProjectionWeights& weights,
                                   Tensor& qkv, Tensor& output_gate, qwen3::TextPhase,
                                   WorkspaceArena& workspace, hipStream_t stream,
                                   ExecutionState* execution, std::int32_t text_layer) {
    auto scope       = workspace.scope();
    Tensor query_key = workspace.alloc(DType::BF16, {2 * TextConfig::key_dim, hidden.ne[1]});
    Tensor value_z   = workspace.alloc(DType::BF16, {2 * TextConfig::value_dim, hidden.ne[1]});
    project_gdn_inputs(hidden, weights, query_key, value_z, workspace, stream, execution,
                       text_layer);
    copy_bf16_rows(query_key, 0, qkv, 0, 2 * TextConfig::key_dim, stream);
    copy_bf16_rows(value_z, 0, qkv, 2 * TextConfig::key_dim, TextConfig::value_dim, stream);
    copy_bf16_rows(value_z, TextConfig::value_dim, output_gate, 0, TextConfig::value_dim, stream);
}

void Variant::gdn_input_projection_prefill(const Tensor& residual, const Tensor& norm_weight,
                                           float eps, const GdnProjectionWeights& weights,
                                           const Tensor& conv_weight, Tensor& conv_state,
                                           Tensor& hidden, Tensor& g, Tensor& beta, Tensor& query,
                                           Tensor& key, Tensor& value, Tensor& output_gate,
                                           qwen3::TextPhase phase, WorkspaceArena& workspace,
                                           hipStream_t stream, ExecutionState* execution,
                                           std::int32_t text_layer) {
    const std::int32_t tokens = residual.ne[1];
    if (!gdn_input_projection_prefill_selected(phase) || tokens <= 0 || residual.ne[2] != 1 ||
        residual.ne[3] != 1) {
        throw std::invalid_argument("R9700 GDN direct-scatter requires ordinary B1 prefill");
    }
    require_bf16_shape(residual, TextConfig::hidden, tokens, 1, "direct-scatter residual");
    require_bf16_shape(hidden, TextConfig::hidden, tokens, 1, "direct-scatter hidden");
    require_bf16_shape(conv_weight, TextConfig::convolution_dim, TextConfig::gdn_conv_kernel, 1,
                       "direct-scatter convolution weight");
    require_bf16_shape(conv_state, TextConfig::convolution_dim, TextConfig::gdn_conv_state_width, 1,
                       "direct-scatter convolution state");
    require_bf16_shape(query, TextConfig::key_dim, tokens, 1, "direct-scatter query");
    require_bf16_shape(key, TextConfig::key_dim, tokens, 1, "direct-scatter key");
    require_bf16_shape(value, TextConfig::value_dim, tokens, 1, "direct-scatter value");
    require_bf16_shape(output_gate, TextConfig::value_dim, tokens, 1, "direct-scatter output gate");

    auto scope             = workspace.scope();
    Tensor query_key       = workspace.alloc(DType::BF16, {2 * TextConfig::key_dim, tokens});
    Tensor projected_value = workspace.alloc(DType::BF16, {TextConfig::value_dim, tokens});
    bool projected         = false;
    if (execution != nullptr && weights.input_projection.query_key.qtype == QType::LQ4) {
        // One kernel normalizes, quantizes and publishes the BF16 rows the controls read.
        Tensor output_gate_flat = output_gate.view({TextConfig::value_dim, tokens});
        const ExecutionState::Lq4Target targets[] = {
            {weights.input_projection.query_key, query_key, nullptr, false},
            {weights.input_projection.value_z, projected_value, &output_gate_flat, false}};
        projected = execution->lq4_normalized_projections(residual, norm_weight, eps, targets,
                                                          stream, &hidden);
    }
    if (!projected &&
        (execution == nullptr || !execution->gdn_q4_normalized_prefill(
                                     residual, norm_weight, eps, weights.input_projection.query_key,
                                     weights.input_projection.value_z, hidden, query_key,
                                     projected_value, output_gate, stream))) {
        ops::rmsnorm(residual, norm_weight, eps, true, hidden, stream);
        auto value_scope = workspace.scope();
        Tensor value_z   = workspace.alloc(DType::BF16, {2 * TextConfig::value_dim, tokens});
        project_gdn_inputs(hidden, weights, query_key, value_z, workspace, stream, execution,
                           text_layer);
        copy_bf16_rows(value_z, 0, projected_value, 0, TextConfig::value_dim, stream);
        copy_bf16_rows(value_z, TextConfig::value_dim, output_gate, 0, TextConfig::value_dim,
                       stream);
    }
    ops::bf16_gdn_projected_gating(hidden, weights.a_projection, weights.b_projection,
                                   weights.a_log, weights.dt_bias, g, beta, stream);
    HIP_CHECK(ops::r9700::gdn::projection_conv_prefill_direct_scatter_bf16(
        static_cast<const hip_bfloat16*>(query_key.data),
        static_cast<const hip_bfloat16*>(projected_value.data), TextConfig::value_dim,
        static_cast<const hip_bfloat16*>(conv_weight.data),
        static_cast<const hip_bfloat16*>(conv_state.data),
        static_cast<hip_bfloat16*>(conv_state.data), static_cast<hip_bfloat16*>(query.data),
        static_cast<hip_bfloat16*>(key.data), static_cast<hip_bfloat16*>(value.data),
        static_cast<std::uint32_t>(tokens), stream));
}

void Variant::gdn_input_projection_snapshot(
    const Tensor& hidden, const GdnProjectionWeights& weights, const Tensor& conv_weight,
    Tensor& conv_states, const Tensor& valid_columns, const Tensor& initial_slots,
    const Tensor& snapshot_base_slots, Tensor& query, Tensor& key, Tensor& value,
    Tensor& output_gate, qwen3::TextPhase, WorkspaceArena& workspace, hipStream_t stream,
    ExecutionState* execution, std::int32_t text_layer) {
    require_gdn_conv_operands(hidden, conv_weight, conv_states, valid_columns, initial_slots, 1,
                              hidden.ne[2] == 1 ? hidden.ne[1] : 16);
    const std::int32_t width = hidden.ne[1];
    const std::int32_t batch = hidden.ne[2];
    require_i32_selector(snapshot_base_slots, batch, 1, false, "snapshot base slots");
    require_bf16_shape(query, TextConfig::key_dim, width, batch, "snapshot query");
    require_bf16_shape(key, TextConfig::key_dim, width, batch, "snapshot key");
    require_bf16_shape(value, TextConfig::value_dim, width, batch, "snapshot value");
    require_bf16_shape(output_gate, TextConfig::value_dim, width, batch, "snapshot output gate");

    auto scope                = workspace.scope();
    const std::int32_t tokens = width * batch;
    Tensor query_key          = workspace.alloc(DType::BF16, {2 * TextConfig::key_dim, tokens});
    Tensor value_z            = workspace.alloc(DType::BF16, {2 * TextConfig::value_dim, tokens});
    project_gdn_inputs(hidden, weights, query_key, value_z, workspace, stream, execution,
                       text_layer);
    HIP_CHECK(ops::r9700::gdn::projection_conv_snapshot_bf16(
        static_cast<const hip_bfloat16*>(query_key.data),
        static_cast<const hip_bfloat16*>(value_z.data),
        static_cast<const hip_bfloat16*>(conv_weight.data),
        static_cast<hip_bfloat16*>(conv_states.data),
        valid_columns.data == nullptr ? nullptr
                                      : static_cast<const std::int32_t*>(valid_columns.data),
        static_cast<const std::int32_t*>(initial_slots.data),
        static_cast<const std::int32_t*>(snapshot_base_slots.data),
        static_cast<hip_bfloat16*>(query.data), static_cast<hip_bfloat16*>(key.data),
        static_cast<hip_bfloat16*>(value.data), static_cast<hip_bfloat16*>(output_gate.data),
        static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(batch),
        static_cast<std::uint32_t>(conv_states.ne[2]), stream));
}

void Variant::gdn_front_snapshot(const Tensor& residual, const Tensor& norm_weight, float eps,
                                 const GdnProjectionWeights& weights, const Tensor& conv_weight,
                                 Tensor& conv_states, const Tensor& valid_columns,
                                 const Tensor& initial_slots, const Tensor& snapshot_base_slots,
                                 Tensor& hidden, Tensor& g, Tensor& beta, Tensor& query,
                                 Tensor& key, Tensor& value, Tensor& output_gate,
                                 qwen3::TextPhase phase, WorkspaceArena& workspace,
                                 hipStream_t stream, ExecutionState* execution,
                                 std::int32_t text_layer) {
    const std::int32_t width   = hidden.ne[1];
    const std::int32_t batch   = hidden.ne[2];
    const std::int32_t tokens  = width * batch;
    const Tensor residual_flat = residual.view({TextConfig::hidden, tokens});
    if (execution != nullptr && tokens == 1) {
        require_gdn_conv_operands(hidden, conv_weight, conv_states, valid_columns, initial_slots, 1,
                                  1);
        require_i32_selector(snapshot_base_slots, batch, 1, false, "snapshot base slots");
        require_bf16_shape(query, TextConfig::key_dim, width, batch, "snapshot query");
        require_bf16_shape(key, TextConfig::key_dim, width, batch, "snapshot key");
        require_bf16_shape(value, TextConfig::value_dim, width, batch, "snapshot value");
        require_bf16_shape(output_gate, TextConfig::value_dim, width, batch,
                           "snapshot output gate");
        auto scope       = workspace.scope();
        Tensor query_key = workspace.alloc(DType::BF16, {2 * TextConfig::key_dim, tokens});
        Tensor value_z   = workspace.alloc(DType::BF16, {2 * TextConfig::value_dim, tokens});
        if (execution->gdn_q4_normalized_front_t1(residual_flat, norm_weight, eps, weights, g, beta,
                                                  query_key, value_z, text_layer, stream)) {
            HIP_CHECK(ops::r9700::gdn::projection_conv_snapshot_bf16(
                static_cast<const hip_bfloat16*>(query_key.data),
                static_cast<const hip_bfloat16*>(value_z.data),
                static_cast<const hip_bfloat16*>(conv_weight.data),
                static_cast<hip_bfloat16*>(conv_states.data),
                valid_columns.data == nullptr
                    ? nullptr
                    : static_cast<const std::int32_t*>(valid_columns.data),
                static_cast<const std::int32_t*>(initial_slots.data),
                static_cast<const std::int32_t*>(snapshot_base_slots.data),
                static_cast<hip_bfloat16*>(query.data), static_cast<hip_bfloat16*>(key.data),
                static_cast<hip_bfloat16*>(value.data),
                static_cast<hip_bfloat16*>(output_gate.data), static_cast<std::uint32_t>(width),
                static_cast<std::uint32_t>(batch), static_cast<std::uint32_t>(conv_states.ne[2]),
                stream));
            return;
        }
    }
    Tensor hidden_flat = hidden.view({TextConfig::hidden, tokens});
    gdn_norm_control_projection(residual_flat, norm_weight, eps, weights, hidden_flat, g, beta,
                                stream);
    gdn_input_projection_snapshot(hidden, weights, conv_weight, conv_states, valid_columns,
                                  initial_slots, snapshot_base_slots, query, key, value,
                                  output_gate, phase, workspace, stream, execution, text_layer);
}

void Variant::gdn_input_projection_record(const Tensor& hidden, const GdnProjectionWeights& weights,
                                          const Tensor& conv_weight, const Tensor& conv_states,
                                          const Tensor& valid_columns, const Tensor& initial_slots,
                                          Tensor& conv_record, Tensor& query, Tensor& key,
                                          Tensor& value, Tensor& output_gate, qwen3::TextPhase,
                                          WorkspaceArena& workspace, hipStream_t stream,
                                          const Tensor* parent_index, ExecutionState* execution,
                                          std::int32_t text_layer) {
    require_gdn_conv_operands(hidden, conv_weight, conv_states, valid_columns, initial_slots, 2,
                              16);
    const std::int32_t width = hidden.ne[1];
    const std::int32_t batch = hidden.ne[2];
    require_bf16_shape(conv_record, TextConfig::convolution_dim, width, batch, "conv record");
    require_bf16_shape(query, TextConfig::key_dim, width, batch, "record query");
    require_bf16_shape(key, TextConfig::key_dim, width, batch, "record key");
    require_bf16_shape(value, TextConfig::value_dim, width, batch, "record value");
    require_bf16_shape(output_gate, TextConfig::value_dim, width, batch, "record output gate");
    if (parent_index != nullptr && parent_index->data != nullptr) {
        require_i32_selector(*parent_index, width, batch, false, "parent index");
    }

    auto scope                = workspace.scope();
    const std::int32_t tokens = width * batch;
    Tensor query_key          = workspace.alloc(DType::BF16, {2 * TextConfig::key_dim, tokens});
    Tensor value_z            = workspace.alloc(DType::BF16, {2 * TextConfig::value_dim, tokens});
    project_gdn_inputs(hidden, weights, query_key, value_z, workspace, stream, execution,
                       text_layer);
    HIP_CHECK(ops::r9700::gdn::projection_conv_record_bf16(
        static_cast<const hip_bfloat16*>(query_key.data),
        static_cast<const hip_bfloat16*>(value_z.data),
        static_cast<const hip_bfloat16*>(conv_weight.data),
        static_cast<const hip_bfloat16*>(conv_states.data),
        valid_columns.data == nullptr ? nullptr
                                      : static_cast<const std::int32_t*>(valid_columns.data),
        static_cast<const std::int32_t*>(initial_slots.data),
        parent_index == nullptr || parent_index->data == nullptr
            ? nullptr
            : static_cast<const std::int32_t*>(parent_index->data),
        static_cast<hip_bfloat16*>(conv_record.data), static_cast<hip_bfloat16*>(query.data),
        static_cast<hip_bfloat16*>(key.data), static_cast<hip_bfloat16*>(value.data),
        static_cast<hip_bfloat16*>(output_gate.data), static_cast<std::uint32_t>(width),
        static_cast<std::uint32_t>(batch), static_cast<std::uint32_t>(conv_states.ne[2]), stream));
}

void Variant::gdn_front_mixed(const Tensor& residual, const Tensor& norm_weight, float eps,
                              const GdnProjectionWeights& weights, const Tensor& conv_weight,
                              Tensor& prefill_conv_state, const Tensor& conv_states,
                              const Tensor& valid_columns, const Tensor& initial_slots,
                              std::int32_t prefill_columns, Tensor& hidden, Tensor& g, Tensor& beta,
                              Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value,
                              Tensor& output_gate, WorkspaceArena& workspace, hipStream_t stream,
                              ExecutionState* execution, std::int32_t text_layer) {
    const std::int32_t tokens = residual.ne[1];
    const std::int32_t width  = conv_record.ne[1];
    const std::int32_t batch  = conv_record.ne[2];
    const std::int32_t owner  = prefill_columns;
    if (owner <= 0 || width <= 0 || batch <= 0 || owner + width * batch != tokens) {
        throw std::invalid_argument("R9700 GDN mixed front has invalid column geometry");
    }
    require_bf16_shape(residual, TextConfig::hidden, tokens, 1, "mixed residual");
    require_bf16_shape(hidden, TextConfig::hidden, tokens, 1, "mixed hidden");
    require_bf16_shape(conv_record, TextConfig::convolution_dim, width, batch, "mixed record");
    require_bf16_shape(query, TextConfig::key_dim, tokens, 1, "mixed query");
    require_bf16_shape(key, TextConfig::key_dim, tokens, 1, "mixed key");
    require_bf16_shape(value, TextConfig::value_dim, tokens, 1, "mixed value");
    require_bf16_shape(output_gate, TextConfig::value_dim, tokens, 1, "mixed output gate");
    require_bf16_shape(prefill_conv_state, TextConfig::convolution_dim,
                       TextConfig::gdn_conv_state_width, 1, "mixed prefill convolution state");
    const Tensor verify_hidden =
        hidden.slice(1, owner, width * batch).view({TextConfig::hidden, width, batch});
    require_gdn_conv_operands(verify_hidden, conv_weight, conv_states, valid_columns, initial_slots,
                              2, 16);

    auto scope        = workspace.scope();
    const auto column = [](const Tensor& tensor, std::int32_t first) {
        return static_cast<hip_bfloat16*>(tensor.data) +
               static_cast<std::size_t>(tensor.ne[0]) * static_cast<std::size_t>(first);
    };
    Tensor query_key = workspace.alloc(DType::BF16, {2 * TextConfig::key_dim, tokens});
    // LQ4: value and output gate project as separate outputs, z straight into output_gate
    // for every column. Otherwise one combined value-z projection whose z rows are copied out.
    Tensor projected_value  = workspace.alloc(DType::BF16, {TextConfig::value_dim, tokens});
    Tensor output_gate_flat = output_gate.view({TextConfig::value_dim, tokens});
    bool split              = false;
    if (execution != nullptr && weights.input_projection.query_key.qtype == QType::LQ4) {
        // One kernel normalizes, quantizes and publishes the BF16 rows the controls read.
        const ExecutionState::Lq4Target targets[] = {
            {weights.input_projection.query_key, query_key, nullptr, false},
            {weights.input_projection.value_z, projected_value, &output_gate_flat, false}};
        split = execution->lq4_normalized_projections(residual, norm_weight, eps, targets, stream,
                                                      &hidden);
    }
    if (split) {
        ops::bf16_gdn_projected_gating(hidden, weights.a_projection, weights.b_projection,
                                       weights.a_log, weights.dt_bias, g, beta, stream);
    } else {
        gdn_norm_control_projection(residual, norm_weight, eps, weights, hidden, g, beta, stream);
    }
    Tensor value_z;
    if (!split) {
        value_z = workspace.alloc(DType::BF16, {2 * TextConfig::value_dim, tokens});
        project_gdn_inputs(hidden, weights, query_key, value_z, workspace, stream, execution,
                           text_layer);
        Tensor owner_gate = output_gate.slice(1, 0, owner);
        copy_bf16_rows(value_z.slice(1, 0, owner), TextConfig::value_dim, owner_gate, 0,
                       TextConfig::value_dim, stream);
    }

    const std::int32_t* valid = valid_columns.data == nullptr
                                    ? nullptr
                                    : static_cast<const std::int32_t*>(valid_columns.data);
    if (split) {
        // One launch: the owner's prefill convolution over its leading columns, scattered
        // straight from the projection rows into query/key/value, and the verify batch's record
        // convolution over the trailing columns.
        HIP_CHECK(ops::r9700::gdn::projection_conv_mixed_split_bf16(
            static_cast<const hip_bfloat16*>(query_key.data),
            static_cast<const hip_bfloat16*>(projected_value.data),
            static_cast<const hip_bfloat16*>(conv_weight.data),
            static_cast<const hip_bfloat16*>(prefill_conv_state.data),
            static_cast<hip_bfloat16*>(prefill_conv_state.data),
            static_cast<hip_bfloat16*>(query.data), static_cast<hip_bfloat16*>(key.data),
            static_cast<hip_bfloat16*>(value.data), static_cast<std::uint32_t>(owner),
            static_cast<const hip_bfloat16*>(conv_states.data), valid,
            static_cast<const std::int32_t*>(initial_slots.data),
            static_cast<hip_bfloat16*>(conv_record.data), static_cast<std::uint32_t>(width),
            static_cast<std::uint32_t>(batch), static_cast<std::uint32_t>(conv_states.ne[2]),
            stream));
    } else {
        HIP_CHECK(ops::r9700::gdn::projection_conv_prefill_direct_scatter_bf16(
            static_cast<const hip_bfloat16*>(query_key.data),
            static_cast<const hip_bfloat16*>(value_z.data), 2 * TextConfig::value_dim,
            static_cast<const hip_bfloat16*>(conv_weight.data),
            static_cast<const hip_bfloat16*>(prefill_conv_state.data),
            static_cast<hip_bfloat16*>(prefill_conv_state.data),
            static_cast<hip_bfloat16*>(query.data), static_cast<hip_bfloat16*>(key.data),
            static_cast<hip_bfloat16*>(value.data), static_cast<std::uint32_t>(owner), stream));
        HIP_CHECK(ops::r9700::gdn::projection_conv_record_bf16(
            column(query_key, owner), column(value_z, owner),
            static_cast<const hip_bfloat16*>(conv_weight.data),
            static_cast<const hip_bfloat16*>(conv_states.data), valid,
            static_cast<const std::int32_t*>(initial_slots.data), nullptr,
            static_cast<hip_bfloat16*>(conv_record.data), column(query, owner), column(key, owner),
            column(value, owner), column(output_gate, owner), static_cast<std::uint32_t>(width),
            static_cast<std::uint32_t>(batch), static_cast<std::uint32_t>(conv_states.ne[2]),
            stream));
    }
}

void Variant::gdn_front_record(const Tensor& residual, const Tensor& norm_weight, float eps,
                               const GdnProjectionWeights& weights, const Tensor& conv_weight,
                               const Tensor& conv_states, const Tensor& valid_columns,
                               const Tensor& initial_slots, Tensor& hidden, Tensor& g, Tensor& beta,
                               Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value,
                               Tensor& output_gate, qwen3::TextPhase phase,
                               WorkspaceArena& workspace, hipStream_t stream,
                               const Tensor* parent_index, ExecutionState* execution,
                               std::int32_t text_layer, const ops::GdnLayerFold* fold) {
    const std::int32_t width   = hidden.ne[1];
    const std::int32_t batch   = hidden.ne[2];
    const std::int32_t tokens  = width * batch;
    const Tensor residual_flat = residual.view({TextConfig::hidden, tokens});
    if (execution != nullptr) {
        require_gdn_conv_operands(hidden, conv_weight, conv_states, valid_columns, initial_slots, 2,
                                  16);
        require_bf16_shape(conv_record, TextConfig::convolution_dim, width, batch, "conv record");
        require_bf16_shape(query, TextConfig::key_dim, width, batch, "record query");
        require_bf16_shape(key, TextConfig::key_dim, width, batch, "record key");
        require_bf16_shape(value, TextConfig::value_dim, width, batch, "record value");
        require_bf16_shape(output_gate, TextConfig::value_dim, width, batch, "record output gate");
        if (parent_index != nullptr && parent_index->data != nullptr) {
            require_i32_selector(*parent_index, width, batch, false, "parent index");
        }
        const ExecutionState::GdnConvRecord record{
            conv_weight, conv_states, valid_columns, initial_slots, parent_index,
            conv_record, query,       key,           value,         output_gate};
        if (execution->gdn_q4_normalized_front_record(residual_flat, norm_weight, eps, weights,
                                                      record, g, beta, workspace, text_layer,
                                                      stream, fold)) {
            return;
        }
    }
    if (fold != nullptr) ops::gdn_replay_fold_layer(*fold, stream);
    Tensor hidden_flat = hidden.view({TextConfig::hidden, tokens});
    gdn_norm_control_projection(residual_flat, norm_weight, eps, weights, hidden_flat, g, beta,
                                stream);
    gdn_input_projection_record(hidden, weights, conv_weight, conv_states, valid_columns,
                                initial_slots, conv_record, query, key, value, output_gate, phase,
                                workspace, stream, parent_index, execution, text_layer);
}

void Variant::gdn_output_projection(const Tensor& recurrent_output, const Tensor& norm,
                                    const Tensor& gate, float eps, Tensor& normalized,
                                    bool materialize_normalized, const Weight& weight,
                                    Tensor& residual, qwen3::TextPhase phase,
                                    WorkspaceArena& workspace, hipStream_t stream, std::int32_t,
                                    ExecutionState* execution, std::int32_t text_layer,
                                    bool ordinary_decode) {
    // The output gate and the activation workspace may still belong to the side projection.
    if (execution != nullptr) execution->join_gdn_gate(stream);
    if (!materialize_normalized && execution != nullptr) {
        const ExecutionState::Lq4Target targets[] = {{weight, residual, nullptr, true}};
        if (execution->lq4_gated_rmsnorm_projections(recurrent_output, norm, gate, eps, targets,
                                                     stream))
            return;
    }
    if (!materialize_normalized && execution != nullptr &&
        execution->gated_normalized_output(recurrent_output, norm, gate, eps, weight, residual,
                                           text_layer, stream)) {
        return;
    }
    ops::gated_rmsnorm(recurrent_output, norm, gate, eps, normalized, stream);
    const Tensor hidden = normalized.view({TextConfig::value_dim, normalized.ne[2]});
    if (execution != nullptr) {
        const ExecutionState::Lq4Target targets[] = {{weight, residual, nullptr, true}};
        if (execution->lq4_projections(hidden, targets, stream)) return;
    }
    if (execution != nullptr && (execution->projected_residual_t1(hidden, weight, residual, phase,
                                                                  ordinary_decode, stream) ||
                                 execution->projected_residual_batched(
                                     hidden, weight, residual, phase, text_layer, stream))) {
        return;
    }
    auto scope   = workspace.scope();
    Tensor delta = workspace.alloc(DType::BF16, {TextConfig::hidden, hidden.ne[1]});
    selected_linear(execution, SelectedLinearRole::GdnOutput, text_layer, hidden, weight, delta,
                    workspace, stream);
    ops::residual_add(delta, residual, stream);
}

void Variant::gdn_norm_control_projection(const Tensor& residual, const Tensor& norm_weight,
                                          float eps, const GdnProjectionWeights& weights,
                                          Tensor& hidden, Tensor& g, Tensor& beta,
                                          hipStream_t stream) {
    ops::rmsnorm(residual, norm_weight, eps, true, hidden, stream);
    ops::bf16_gdn_projected_gating(hidden, weights.a_projection, weights.b_projection,
                                   weights.a_log, weights.dt_bias, g, beta, stream);
}

namespace {

void post_mixer_body(const Tensor& hidden, const Variant::PostMixerWeights& weights,
                     Tensor& residual, qwen3::TextPhase phase, WorkspaceArena& workspace,
                     hipStream_t stream, Variant::ExecutionState* execution,
                     std::int32_t text_layer, const Tensor* norm, float eps, bool ordinary_decode) {
    auto outer = workspace.scope();
    if (execution != nullptr && weights.gate_up.qtype == QType::LQ4 &&
        weights.down.qtype == QType::LQ4) {
        // gate/up rows are interleaved: the projection publishes the SiLU-gated activation.
        Tensor activation = workspace.alloc(DType::BF16, {TextConfig::intermediate, hidden.ne[1]});
        const Variant::ExecutionState::Lq4Target up[] = {
            {weights.gate_up, activation, nullptr, false, true}};
        const bool projected =
            norm != nullptr
                ? execution->lq4_normalized_projections(residual, *norm, eps, up, stream)
                : execution->lq4_projections(hidden, up, stream);
        const Variant::ExecutionState::Lq4Target down[] = {{weights.down, residual, nullptr, true}};
        if (projected && execution->lq4_projections(activation, down, stream)) return;
        throw std::logic_error("R9700 LQ4 MLP projection is unsupported at this width");
    }
    // Row-scaled E4M3 protected layers at prefill widths: the same image producers and down
    // accumulate epilogue; a row-scaled gate/up (not interleaved) publishes both halves for the
    // separate SiLU-gated product.
    const auto fp8_route = [](const Weight& weight) {
        return weight.qtype == QType::LQ4 || weight.qtype == QType::F8E4M3_ROW_F32S;
    };
    if (execution != nullptr && fp8_route(weights.gate_up) && fp8_route(weights.down)) {
        const auto tokens = static_cast<std::uint32_t>(hidden.ne[1]);
        const bool paired = weights.gate_up.qtype == QType::LQ4;
        auto mlp_scope    = workspace.scope();
        Tensor activation = workspace.alloc(DType::BF16, {TextConfig::intermediate, hidden.ne[1]});
        Tensor gate_up =
            paired ? activation
                   : workspace.alloc(DType::BF16, {2 * TextConfig::intermediate, hidden.ne[1]});
        const Variant::ExecutionState::Lq4Target up[] = {
            {weights.gate_up, paired ? activation : gate_up, nullptr, false, paired}};
        const Variant::ExecutionState::Lq4Target down[] = {{weights.down, residual, nullptr, true}};
        if (execution->lq4_targets_supported(tokens, TextConfig::hidden, up) &&
            execution->lq4_targets_supported(tokens, TextConfig::intermediate, down)) {
            const bool projected =
                norm != nullptr
                    ? execution->lq4_normalized_projections(residual, *norm, eps, up, stream)
                    : execution->lq4_projections(hidden, up, stream);
            if (!projected) throw std::logic_error("R9700 FP8 MLP gate/up projection failed");
            if (!paired) {
                ops::silu_mul(gate_up.slice(0, 0, TextConfig::intermediate),
                              gate_up.slice(0, TextConfig::intermediate, TextConfig::intermediate),
                              activation, stream);
            }
            if (!execution->lq4_projections(activation, down, stream)) {
                throw std::logic_error("R9700 FP8 MLP down projection failed");
            }
            return;
        }
    }
    const auto project_gate_up = [&](Tensor& gate_up) {
        if (norm != nullptr) {
            if (execution != nullptr &&
                execution->normalized_linear_t1(residual, *norm, eps, weights.gate_up, gate_up,
                                                phase, ordinary_decode, text_layer, stream)) {
                return;
            }
            if (execution != nullptr &&
                execution->normalized_linear_batched(residual, *norm, eps, weights.gate_up, gate_up,
                                                     text_layer, stream)) {
                return;
            }
            Tensor normalized_hidden = hidden;
            ops::rmsnorm(residual, *norm, eps, true, normalized_hidden, stream);
        }
        selected_linear(execution, Variant::SelectedLinearRole::MlpGateUp, text_layer, hidden,
                        weights.gate_up, gate_up, workspace, stream);
    };
    const bool fused_down = execution != nullptr && hidden.ne[1] > 0 &&
                            ops::r9700::linear::q4_linear_activation_bits(
                                static_cast<std::uint32_t>(hidden.ne[1]), TextConfig::hidden,
                                TextConfig::intermediate, TextConfig::intermediate) == 8U &&
                            Variant::ExecutionState::fused_mlp_down_selected(
                                ops::r9700::linear::kQ4ActivationBits, weights.down.qtype,
                                static_cast<std::uint32_t>(hidden.ne[1]), text_layer);
    if (fused_down) {
        Tensor gate_up = workspace.alloc(DType::BF16, {2 * TextConfig::intermediate, hidden.ne[1]});
        if (norm != nullptr &&
            execution->normalized_mlp(residual, *norm, eps, weights.gate_up, weights.down, gate_up,
                                      text_layer, stream)) {
            return;
        }
        project_gate_up(gate_up);
        execution->fused_mlp_down(gate_up, weights.down, residual, stream);
        return;
    }
    Tensor activation = workspace.alloc(DType::BF16, {TextConfig::intermediate, hidden.ne[1]});
    {
        auto gate_scope = workspace.scope();
        Tensor gate_up = workspace.alloc(DType::BF16, {2 * TextConfig::intermediate, hidden.ne[1]});
        project_gate_up(gate_up);
        ops::silu_mul(gate_up.slice(0, 0, TextConfig::intermediate),
                      gate_up.slice(0, TextConfig::intermediate, TextConfig::intermediate),
                      activation, stream);
    }
    {
        auto delta_scope = workspace.scope();
        if (execution != nullptr && text_layer >= 0 &&
            (execution->projected_residual_t1(activation, weights.down, residual, phase,
                                              ordinary_decode, stream) ||
             execution->projected_residual_batched(activation, weights.down, residual, phase,
                                                   text_layer, stream))) {
            return;
        }
        Tensor delta = workspace.alloc(DType::BF16, {TextConfig::hidden, hidden.ne[1]});
        if (weights.down.qtype == QType::F8E4M3_ROW_F32S) {
            selected_linear(execution, Variant::SelectedLinearRole::MlpDown, text_layer, activation,
                            weights.down, delta, workspace, stream);
        } else {
            serialized_linear(execution, activation, weights.down, delta, workspace, stream);
        }
        ops::residual_add(delta, residual, stream);
    }
}

} // namespace

void Variant::post_mixer(const Tensor& norm, float eps, const Tensor& hidden,
                         const PostMixerWeights& weights, Tensor& residual, qwen3::TextPhase phase,
                         WorkspaceArena& workspace, hipStream_t stream, ExecutionState* execution,
                         std::int32_t text_layer, bool ordinary_decode) {
    post_mixer_body(hidden, weights, residual, phase, workspace, stream, execution, text_layer,
                    &norm, eps, ordinary_decode);
}

void Variant::mtp_post_mixer(const Tensor& hidden, const MtpPostMixerWeights& weights,
                             Tensor& residual, WorkspaceArena& workspace, hipStream_t stream,
                             std::int32_t, ExecutionState* execution) {
    post_mixer_body(hidden, weights, residual, qwen3::TextPhase::Verify, workspace, stream,
                    execution, -1, nullptr, 0.0F, false);
}

std::size_t Variant::mtp_attention_projection_workspace_capacity_bytes(std::int32_t first,
                                                                       std::int32_t last) {
    validate_token_interval(first, last);
    return 0;
}

std::size_t Variant::mtp_kv_projection_workspace_capacity_bytes(std::int32_t first,
                                                                std::int32_t last) {
    validate_token_interval(first, last);
    return 0;
}

std::size_t Variant::mtp_q_gate_projection_workspace_capacity_bytes(std::int32_t first,
                                                                    std::int32_t last) {
    validate_token_interval(first, last);
    return 0;
}

std::size_t Variant::mtp_fc_workspace_capacity_bytes(std::int32_t first, std::int32_t last) {
    validate_token_interval(first, last);
    return one_matrix_bytes(TextConfig::mtp_input_rows, last);
}

std::size_t Variant::mtp_attention_output_workspace_capacity_bytes(std::int32_t first,
                                                                   std::int32_t last) {
    validate_token_interval(first, last);
    return one_matrix_bytes(TextConfig::hidden, last);
}

std::size_t Variant::attention_projection_workspace_capacity_bytes(WeightsProfile profile,
                                                                   qwen3::TextPhase,
                                                                   std::int32_t first,
                                                                   std::int32_t last) {
    validate_profile(profile);
    validate_token_interval(first, last);
    // The shared-quantization FP8 route holds both projections at once.
    return two_matrix_bytes(7168, 7168, last);
}

std::size_t Variant::attention_output_projection_workspace_capacity_bytes(WeightsProfile profile,
                                                                          qwen3::TextPhase,
                                                                          std::int32_t first,
                                                                          std::int32_t last) {
    validate_profile(profile);
    validate_token_interval(first, last);
    return one_matrix_bytes(TextConfig::hidden, last);
}

std::size_t Variant::gdn_input_projection_workspace_capacity_bytes(WeightsProfile profile,
                                                                   qwen3::TextPhase,
                                                                   std::int32_t first,
                                                                   std::int32_t last) {
    validate_profile(profile);
    validate_token_interval(first, last);
    return two_matrix_bytes(2 * TextConfig::key_dim, 2 * TextConfig::value_dim, last);
}

std::size_t Variant::gdn_input_projection_snapshot_workspace_capacity_bytes(WeightsProfile profile,
                                                                            qwen3::TextPhase,
                                                                            std::int32_t batch,
                                                                            std::int32_t first,
                                                                            std::int32_t last) {
    validate_profile(profile);
    validate_token_interval(first, last);
    if (batch <= 0 || batch > static_cast<std::int32_t>(kMaximumConcurrency) ||
        (batch > 1 && last > 16)) {
        throw std::invalid_argument("invalid R9700 GDN snapshot workspace geometry");
    }
    return two_matrix_bytes(2 * TextConfig::key_dim, 2 * TextConfig::value_dim, batch * last);
}

std::size_t Variant::gdn_input_projection_record_workspace_capacity_bytes(WeightsProfile profile,
                                                                          qwen3::TextPhase,
                                                                          std::int32_t batch,
                                                                          std::int32_t first,
                                                                          std::int32_t last) {
    validate_profile(profile);
    validate_token_interval(first, last);
    if (batch <= 0 || batch > static_cast<std::int32_t>(kMaximumConcurrency) || first < 2 ||
        last > 16) {
        throw std::invalid_argument("invalid R9700 GDN record workspace geometry");
    }
    return two_matrix_bytes(2 * TextConfig::key_dim, 2 * TextConfig::value_dim, batch * last);
}

std::size_t Variant::gdn_output_projection_workspace_capacity_bytes(WeightsProfile profile,
                                                                    qwen3::TextPhase,
                                                                    std::int32_t first,
                                                                    std::int32_t last) {
    validate_profile(profile);
    validate_token_interval(first, last);
    return one_matrix_bytes(TextConfig::hidden, last);
}

std::size_t Variant::post_mixer_workspace_capacity_bytes(WeightsProfile profile, qwen3::TextPhase,
                                                         std::int32_t first, std::int32_t last) {
    validate_profile(profile);
    validate_token_interval(first, last);
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {TextConfig::intermediate, last});
    {
        auto scope = layout.scope();
        (void)layout.alloc(DType::BF16, {2 * TextConfig::intermediate, last});
    }
    {
        auto scope = layout.scope();
        (void)layout.alloc(DType::BF16, {TextConfig::hidden, last});
    }
    return layout.peak_bytes(1);
}

std::size_t Variant::mtp_post_mixer_workspace_capacity_bytes(std::int32_t first,
                                                             std::int32_t last) {
    return post_mixer_workspace_capacity_bytes(WeightsProfile::R9700W8G32Candidate,
                                               qwen3::TextPhase::Verify, first, last);
}

QType Variant::dflash_matrix_qtype(WeightsProfile profile) {
    validate_profile(profile);
    profile = fp8_capped_base_profile(profile);
    switch (profile) {
    case WeightsProfile::R9700Q4Fp8EarlyAttentionEvaluation:
    case WeightsProfile::R9700Q4Fp8DefaultProtectedEvaluation:
    case WeightsProfile::R9700Q4Fp8OutputOnlyEvaluation:
    case WeightsProfile::R9700Q4Fp8SelectiveNoLateMlpEvaluation:
    case WeightsProfile::R9700Q4Fp8AllAttentionEvaluation:
    case WeightsProfile::R9700Q4Fp8AttentionGdnEvaluation:
    case WeightsProfile::R9700Q4Fp8SelectiveCapEvaluation:
        throw std::invalid_argument("FP8 capped base has no DFlash companion");
    case WeightsProfile::R9700Q4Fp8SelectiveCapDFlash2Q4Evaluation:
    case WeightsProfile::R9700Q4SelectiveProtectedDFlash2Q4Evaluation:
    case WeightsProfile::R9700Q4G64DFlash2Q4Evaluation:
    case WeightsProfile::R9700Q4G64DFlash2Q4MseEvaluation:
    case WeightsProfile::R9700Q4W8MseDFlash2Q4Evaluation:
    case WeightsProfile::R9700Q4W8MseDFlash2Q4MseEvaluation:
    case WeightsProfile::R9700Q4G64Fp8FourRoleDFlash2Q4Evaluation:
    case WeightsProfile::R9700Q4G64Fp8FourRoleDFlash2Q4MseEvaluation:
        return QType::Q4G64_F16S;
    case WeightsProfile::R9700Q4G64DFlash2W8MseEvaluation:
    case WeightsProfile::R9700Q4W8MseDFlash2W8MseEvaluation:
    case WeightsProfile::R9700Q4G64Fp8FourRoleDFlash2W8MseEvaluation:
    case WeightsProfile::R9700W8G32Candidate:
    case WeightsProfile::R9700W8Bf16EmbeddingEvaluation:
    case WeightsProfile::R9700W8Bf16AttentionQueryKeyEvaluation:
    case WeightsProfile::R9700W8Bf16AttentionValueOutputEvaluation:
    case WeightsProfile::R9700W8Bf16GdnQueryKeyEvaluation:
    case WeightsProfile::R9700Q4G64Evaluation:
    case WeightsProfile::R9700Q4W8Evaluation:
        return QType::W8G32_F16S;
    case WeightsProfile::R9700Lq4:
#define NINFER_QWEN38_FP8_ENDPOINT(symbol, id, base, embed, head) case WeightsProfile::symbol:
#include "targets/qwen3_8_27b/impl/load/fp8_endpoint_selection.inc"
#undef NINFER_QWEN38_FP8_ENDPOINT
    case WeightsProfile::R9700Q4G64Fp8FourRoleN16K16Evaluation:
    case WeightsProfile::R9700Q4SelectiveProtectedN16K16Evaluation:
        // fp8_capped_base_profile() mapped the FP8-capped profiles to their base recipe; the two
        // N16K16 profiles carry no DFlash weights.
        break;
    }
    throw std::invalid_argument("invalid R9700 target DFlash matrix profile");
}

QType Variant::token_embedding_qtype(WeightsProfile profile) {
    validate_profile(profile);
    switch (token_embedding_format(profile)) {
    case artifact::NumericFormat::BF16:
        return QType::BF16_CTRL;
    case artifact::NumericFormat::Q4G64_F16S:
        return QType::Q4G64_F16S;
    case artifact::NumericFormat::W8G32_F16S:
        return QType::W8G32_F16S;
    default:
        throw std::logic_error("R9700 token embedding format has no staging encoding");
    }
}

std::size_t Variant::linear_workspace_capacity_bytes(WeightsProfile profile, std::int32_t tokens) {
    validate_profile(profile);
    const auto capped_base = fp8_capped_base_profile(profile);
    if (capped_base != profile) {
        const auto base = linear_workspace_capacity_bytes(capped_base, tokens);
        if (is_lq4_text_profile(profile)) {
            return std::max(
                base, ops::linear_workspace_capacity_bytes(QType::LQ4, tokens, TextConfig::hidden));
        }
        return fp8_capped_w8_head(profile)
                   ? std::max(base, ops::linear_workspace_capacity_bytes(QType::W8G32_F16S, tokens,
                                                                         TextConfig::hidden))
                   : base;
    }
    // Companion activation storage is independent of the unchanged base recipe.
    switch (profile) {
    case WeightsProfile::R9700Q4Fp8SelectiveCapDFlash2Q4Evaluation:
        return std::max(linear_workspace_capacity_bytes(
                            WeightsProfile::R9700Q4Fp8SelectiveCapEvaluation, tokens),
                        ops::linear_workspace_capacity_bytes(QType::Q4G64_F16S, tokens,
                                                             DFlashConfig::feature_rows));
    case WeightsProfile::R9700Q4SelectiveProtectedDFlash2Q4Evaluation:
        return std::max(linear_workspace_capacity_bytes(
                            WeightsProfile::R9700Q4SelectiveProtectedN16K16Evaluation, tokens),
                        ops::linear_workspace_capacity_bytes(QType::Q4G64_F16S, tokens,
                                                             DFlashConfig::feature_rows));
    case WeightsProfile::R9700Q4SelectiveProtectedN16K16Evaluation:
        return std::max(
            ops::linear_workspace_capacity_bytes(QType::Q4G64_F16S, tokens,
                                                 TextConfig::intermediate),
            ops::linear_workspace_capacity_bytes(QType::W8G32_F16S, tokens, TextConfig::hidden));
    case WeightsProfile::R9700Q4G64DFlash2W8MseEvaluation:
    case WeightsProfile::R9700Q4W8MseDFlash2W8MseEvaluation:
    case WeightsProfile::R9700Q4G64Fp8FourRoleDFlash2W8MseEvaluation: {
        const auto base = profile == WeightsProfile::R9700Q4W8MseDFlash2W8MseEvaluation
                              ? WeightsProfile::R9700Q4W8Evaluation
                              : WeightsProfile::R9700Q4G64Evaluation;
        return std::max(linear_workspace_capacity_bytes(base, tokens),
                        ops::linear_workspace_capacity_bytes(QType::W8G32_F16S, tokens,
                                                             DFlashConfig::feature_rows));
    }
    case WeightsProfile::R9700W8G32Candidate:
    case WeightsProfile::R9700W8Bf16EmbeddingEvaluation:
    case WeightsProfile::R9700W8Bf16AttentionQueryKeyEvaluation:
    case WeightsProfile::R9700W8Bf16AttentionValueOutputEvaluation:
    case WeightsProfile::R9700W8Bf16GdnQueryKeyEvaluation:
        return ops::linear_workspace_capacity_bytes(QType::W8G32_F16S, tokens,
                                                    TextConfig::intermediate);
    case WeightsProfile::R9700Q4W8MseDFlash2Q4Evaluation:
    case WeightsProfile::R9700Q4W8MseDFlash2Q4MseEvaluation:
        // DFlash feature projection consumes [5120,25600], which is the largest K in the
        // companion or mixed base. Reserve its A8G64 image for every DFlash schedule.
        return ops::linear_workspace_capacity_bytes(QType::Q4G64_F16S, tokens,
                                                    DFlashConfig::feature_rows);
    case WeightsProfile::R9700Q4W8Evaluation:
        // The mixed recipe's Q4 roles top out at hidden K, while its W8 MLP-down roles consume
        // intermediate K. The compile-time W8 profile is workspace-free when exact and owns an
        // A8G32 image only in the separately built A8 evaluator.
        return std::max(
            ops::linear_workspace_capacity_bytes(QType::Q4G64_F16S, tokens, TextConfig::hidden),
            ops::linear_workspace_capacity_bytes(QType::W8G32_F16S, tokens,
                                                 TextConfig::intermediate));
    case WeightsProfile::R9700Q4G64DFlash2Q4Evaluation:
    case WeightsProfile::R9700Q4G64DFlash2Q4MseEvaluation:
    case WeightsProfile::R9700Q4G64Fp8FourRoleDFlash2Q4Evaluation:
    case WeightsProfile::R9700Q4G64Fp8FourRoleDFlash2Q4MseEvaluation:
        return ops::linear_workspace_capacity_bytes(QType::Q4G64_F16S, tokens,
                                                    DFlashConfig::feature_rows);
    case WeightsProfile::R9700Q4G64Evaluation:
    case WeightsProfile::R9700Q4G64Fp8FourRoleN16K16Evaluation:
    case WeightsProfile::R9700Q4Fp8EarlyAttentionEvaluation:
    case WeightsProfile::R9700Q4Fp8DefaultProtectedEvaluation:
    case WeightsProfile::R9700Q4Fp8OutputOnlyEvaluation:
    case WeightsProfile::R9700Q4Fp8SelectiveNoLateMlpEvaluation:
    case WeightsProfile::R9700Q4Fp8AllAttentionEvaluation:
    case WeightsProfile::R9700Q4Fp8AttentionGdnEvaluation:
    case WeightsProfile::R9700Q4Fp8SelectiveCapEvaluation:
        // The all-Q4 recipe also quantizes MLP down, whose K is the largest Text matrix input.
        return ops::linear_workspace_capacity_bytes(QType::Q4G64_F16S, tokens,
                                                    TextConfig::intermediate);
    case WeightsProfile::R9700Lq4:
#define NINFER_QWEN38_FP8_ENDPOINT(symbol, id, base, embed, head) case WeightsProfile::symbol:
#include "targets/qwen3_8_27b/impl/load/fp8_endpoint_selection.inc"
#undef NINFER_QWEN38_FP8_ENDPOINT
        break; // returned through their base recipe above
    }
    throw std::invalid_argument("invalid R9700 target weight profile");
}

std::size_t Variant::vision_linear_workspace_capacity_bytes(WeightsProfile profile,
                                                            std::int32_t tokens) {
    validate_profile(profile);
    profile = fp8_capped_base_profile(profile);
    switch (profile) {
    case WeightsProfile::R9700Q4Fp8SelectiveCapDFlash2Q4Evaluation:
    case WeightsProfile::R9700Q4SelectiveProtectedDFlash2Q4Evaluation:
    case WeightsProfile::R9700Q4SelectiveProtectedN16K16Evaluation:
        return vision_linear_workspace_capacity_bytes(WeightsProfile::R9700Q4G64Evaluation, tokens);
    case WeightsProfile::R9700W8G32Candidate:
    case WeightsProfile::R9700W8Bf16EmbeddingEvaluation:
    case WeightsProfile::R9700W8Bf16AttentionQueryKeyEvaluation:
    case WeightsProfile::R9700W8Bf16AttentionValueOutputEvaluation:
    case WeightsProfile::R9700W8Bf16GdnQueryKeyEvaluation:
        return ops::linear_workspace_capacity_bytes(QType::W8G32_F16S, tokens,
                                                    VisionConfig::merger_hidden);
    case WeightsProfile::R9700Q4W8Evaluation:
    case WeightsProfile::R9700Q4W8MseDFlash2Q4Evaluation:
    case WeightsProfile::R9700Q4W8MseDFlash2Q4MseEvaluation:
    case WeightsProfile::R9700Q4W8MseDFlash2W8MseEvaluation:
        // Mixed Vision Q4 tops out at backbone hidden K; its W8 merger consumes the larger
        // merger-hidden K and therefore owns an A8 workspace in the A8 evaluator.
        return std::max(
            ops::linear_workspace_capacity_bytes(QType::Q4G64_F16S, tokens, VisionConfig::hidden),
            ops::linear_workspace_capacity_bytes(QType::W8G32_F16S, tokens,
                                                 VisionConfig::merger_hidden));
    case WeightsProfile::R9700Q4G64Evaluation:
    case WeightsProfile::R9700Q4G64Fp8FourRoleN16K16Evaluation:
    case WeightsProfile::R9700Q4Fp8EarlyAttentionEvaluation:
    case WeightsProfile::R9700Q4Fp8DefaultProtectedEvaluation:
    case WeightsProfile::R9700Q4Fp8OutputOnlyEvaluation:
    case WeightsProfile::R9700Q4Fp8SelectiveNoLateMlpEvaluation:
    case WeightsProfile::R9700Q4Fp8AllAttentionEvaluation:
    case WeightsProfile::R9700Q4Fp8AttentionGdnEvaluation:
    case WeightsProfile::R9700Q4Fp8SelectiveCapEvaluation:
    case WeightsProfile::R9700Q4G64DFlash2Q4Evaluation:
    case WeightsProfile::R9700Q4G64DFlash2Q4MseEvaluation:
    case WeightsProfile::R9700Q4G64DFlash2W8MseEvaluation:
    case WeightsProfile::R9700Q4G64Fp8FourRoleDFlash2Q4Evaluation:
    case WeightsProfile::R9700Q4G64Fp8FourRoleDFlash2Q4MseEvaluation:
    case WeightsProfile::R9700Q4G64Fp8FourRoleDFlash2W8MseEvaluation:
        // All-Q4 includes the merger matrices, whose represented BF16 input is the largest
        // Vision K (4608, larger than backbone FC2's 4304).
        return ops::linear_workspace_capacity_bytes(QType::Q4G64_F16S, tokens,
                                                    VisionConfig::merger_hidden);
    case WeightsProfile::R9700Lq4:
#define NINFER_QWEN38_FP8_ENDPOINT(symbol, id, base, embed, head) case WeightsProfile::symbol:
#include "targets/qwen3_8_27b/impl/load/fp8_endpoint_selection.inc"
#undef NINFER_QWEN38_FP8_ENDPOINT
        break; // fp8_capped_base_profile() mapped these to their base recipe
    }
    throw std::invalid_argument("invalid R9700 target weight profile");
}

std::size_t Variant::execution_state_capacity_bytes(WeightsProfile profile,
                                                    std::uint32_t prefill_tokens,
                                                    std::uint32_t maximum_graph_tokens) {
    validate_profile(profile);
    if (prefill_tokens == 0U || maximum_graph_tokens == 0U) {
        throw std::invalid_argument("R9700 linear execution profile widths must be positive");
    }
    const std::uint32_t tokens = std::max(prefill_tokens, maximum_graph_tokens);
    std::size_t bytes = linear_workspace_capacity_bytes(profile, static_cast<std::int32_t>(tokens));
    profile           = fp8_capped_base_profile(profile);
    if (is_selective_protected_profile(profile) ||
        profile == WeightsProfile::R9700Q4Fp8SelectiveCapEvaluation ||
        profile == WeightsProfile::R9700Q4Fp8SelectiveCapDFlash2Q4Evaluation) {
        bytes = std::max(bytes, execution_storage_bytes(prefill_tokens, maximum_graph_tokens,
                                                        TextConfig::intermediate));
    }
    if (is_fp8_capped_profile(profile) &&
        profile != WeightsProfile::R9700Q4Fp8SelectiveCapEvaluation &&
        profile != WeightsProfile::R9700Q4Fp8SelectiveCapDFlash2Q4Evaluation) {
        const bool protected_outputs =
            profile == WeightsProfile::R9700Q4Fp8DefaultProtectedEvaluation ||
            profile == WeightsProfile::R9700Q4Fp8OutputOnlyEvaluation ||
            profile == WeightsProfile::R9700Q4Fp8SelectiveNoLateMlpEvaluation;
        const auto columns = protected_outputs
                                 ? std::max(TextConfig::query_size, TextConfig::value_dim)
                                 : TextConfig::hidden;
        bytes =
            std::max(bytes, execution_storage_bytes(prefill_tokens, maximum_graph_tokens, columns));
    }
    if (profile == WeightsProfile::R9700Q4G64Fp8FourRoleN16K16Evaluation ||
        profile == WeightsProfile::R9700Q4G64Fp8FourRoleDFlash2Q4MseEvaluation ||
        profile == WeightsProfile::R9700Q4G64Fp8FourRoleDFlash2W8MseEvaluation ||
        profile == WeightsProfile::R9700Q4G64Fp8FourRoleDFlash2Q4Evaluation) {
        bytes = std::max(bytes, execution_storage_bytes(prefill_tokens, maximum_graph_tokens));
    }
    return bytes == 0U ? 0U
                       : align_up(bytes, kExecutionAlignment,
                                  "R9700 serialized linear activation alignment overflows");
}

} // namespace ninfer::targets::qwen3_8_27b::detail
