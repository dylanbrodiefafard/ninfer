#include "ninfer/ops/gdn_input_proj.h"

#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/scatter.h"

#include "core/layout.h"
#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_kernels.h"
#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.h"
#include "ops/gdn_input_proj/w8/w8_gdn_input_kernels.h"
#include "ops/gdn_input_proj/w8/w8_gdn_input_plan.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void require_matrix(const Tensor& tensor, std::int32_t rows, std::int32_t cols, const char* label) {
    if (tensor.dtype != DType::BF16 || tensor.ne[0] != rows || tensor.ne[1] != cols ||
        tensor.ne[2] != 1 || tensor.ne[3] != 1 || !tensor.is_contiguous() ||
        !aligned_to(tensor.data, 16)) {
        throw std::invalid_argument(std::string("gdn_input_proj: invalid ") + label);
    }
}

void require_snapshot_operands(const Tensor& conv_weight, const Tensor& conv_states,
                               const Tensor& initial_slot, std::int32_t channels,
                               std::int32_t tokens) {
    require_matrix(conv_weight, channels, 4, "conv weight");
    if (conv_states.dtype != DType::BF16 || conv_states.ne[0] != channels ||
        conv_states.ne[1] != 3 || conv_states.ne[2] < tokens || conv_states.ne[3] != 1 ||
        !conv_states.is_contiguous() || !aligned_to(conv_states.data, 16)) {
        throw std::invalid_argument(
            "gdn_input_proj_conv_snapshot: invalid convolution snapshot state");
    }
    if (initial_slot.dtype != DType::I32 || initial_slot.ne[0] != 1 || initial_slot.ne[1] != 1 ||
        initial_slot.ne[2] != 1 || initial_slot.ne[3] != 1 || !initial_slot.is_contiguous() ||
        initial_slot.data == nullptr) {
        throw std::invalid_argument("gdn_input_proj_conv_snapshot: invalid initial slot");
    }
}

void require_rowsplit(const Weight& weight, QType qtype, std::int32_t rows, const char* label) {
    const bool q4_planes =
        qtype != QType::Q4G64_F16S || (weight.qhigh == nullptr && weight.high_plane_bytes == 0);
    const bool q5_planes =
        qtype != QType::Q5G64_F16S || (weight.qhigh != nullptr && weight.high_plane_bytes != 0);
    if (weight.qtype != qtype || weight.layout != QuantLayout::RowSplit ||
        weight.scale_dtype != DType::FP16 || weight.group_size != 64 || weight.group != 64 ||
        weight.ndim != 2 || weight.n != rows || weight.k != 5120 || weight.shape[0] != rows ||
        weight.shape[1] != 5120 || weight.padded_shape[0] != rows ||
        weight.padded_shape[1] != 5120 || !q4_planes || !q5_planes ||
        !aligned_to(weight.qdata, 16) || !aligned_to(weight.scales, 4) ||
        (qtype == QType::Q5G64_F16S && !aligned_to(weight.qhigh, 16))) {
        throw std::invalid_argument(std::string("gdn_input_proj: invalid ") + label);
    }
}

void require_w8_rowsplit(const Weight& weight, std::int32_t rows, const char* label) {
    if (weight.qtype != QType::W8G32_F16S || weight.layout != QuantLayout::RowSplit ||
        weight.scale_dtype != DType::FP16 || weight.group_size != 32 || weight.group != 32 ||
        weight.ndim != 2 || weight.n != rows || weight.k != 2048 || weight.shape[0] != rows ||
        weight.shape[1] != 2048 || weight.padded_shape[0] != rows ||
        weight.padded_shape[1] != 2048 || weight.qhigh != nullptr || weight.high_plane_bytes != 0 ||
        !aligned_to(weight.qdata, 16) || !aligned_to(weight.scales, 16)) {
        throw std::invalid_argument(std::string("gdn_input_proj: invalid ") + label);
    }
}

enum class SnapshotWorkspaceKind {
    None,
    Projected,
    ProjectedAndConvolved,
};

struct SnapshotRoute {
    SnapshotWorkspaceKind workspace;
    detail::W8GdnInputSnapshotScheduleId w8_schedule =
        detail::W8GdnInputSnapshotScheduleId::Composed;
};

SnapshotRoute resolve_snapshot_route(bool q4_q5, std::int32_t tokens) {
    if (tokens <= 0) {
        throw std::invalid_argument("gdn_input_proj_conv_snapshot: T must be positive");
    }
    if (q4_q5) {
        if (tokens == 4) { return {SnapshotWorkspaceKind::Projected}; }
        if (tokens <= 6) { return {SnapshotWorkspaceKind::None}; }
        return {SnapshotWorkspaceKind::ProjectedAndConvolved};
    }

    constexpr std::int32_t kHidden   = 2048;
    constexpr std::int32_t kChannels = 8192;
    constexpr std::int32_t kZRows    = 4096;
    const auto plan                  = detail::w8_gdn_input_snapshot_resolve_plan(
        {kHidden, kChannels, kZRows, kChannels + kZRows, kHidden, tokens});
    return {
        plan.schedule == detail::W8GdnInputSnapshotScheduleId::Composed
            ? SnapshotWorkspaceKind::ProjectedAndConvolved
            : SnapshotWorkspaceKind::None,
        plan.schedule,
    };
}

struct SnapshotWorkspace {
    Tensor projected;
    Tensor convolved;
};

template <class Allocator>
SnapshotWorkspace allocate_snapshot_workspace(Allocator& allocator, std::int32_t channels,
                                              std::int32_t tokens, SnapshotWorkspaceKind kind) {
    SnapshotWorkspace out;
    if (kind == SnapshotWorkspaceKind::None) { return out; }
    out.projected = allocator.alloc(DType::BF16, {channels, tokens});
    if (kind == SnapshotWorkspaceKind::ProjectedAndConvolved) {
        out.convolved = allocator.alloc(DType::BF16, {channels, tokens});
    }
    return out;
}

} // namespace

void gdn_input_proj(const Tensor& x, const Weight& qk_weight, const Weight& v_weight, Tensor& qkv,
                    cudaStream_t stream) {
    constexpr std::int32_t kHidden = 5120;
    constexpr std::int32_t kQkRows = 4096;
    constexpr std::int32_t kVRows  = 6144;
    constexpr std::int32_t kRows   = kQkRows + kVRows;
    const std::int32_t cols        = x.ne[1];
    require_matrix(x, kHidden, cols, "x");
    require_matrix(qkv, kRows, cols, "qkv");
    require_rowsplit(qk_weight, QType::Q4G64_F16S, kQkRows, "qk weight");
    require_rowsplit(v_weight, QType::Q5G64_F16S, kVRows, "value weight");

    detail::q4_q5_gdn_input_dispatch(x, qk_weight, v_weight, qkv, stream);
}

void gdn_input_proj(const Tensor& x, const Weight& query_key_value_z_weight, Tensor& qkv, Tensor& z,
                    cudaStream_t stream) {
    constexpr std::int32_t kHidden  = 2048;
    constexpr std::int32_t kQkvRows = 8192;
    constexpr std::int32_t kZRows   = 4096;
    constexpr std::int32_t kRows    = kQkvRows + kZRows;
    const std::int32_t cols         = x.ne[1];
    if (cols <= 0) { throw std::invalid_argument("gdn_input_proj: T must be positive"); }
    require_matrix(x, kHidden, cols, "x");
    require_matrix(qkv, kQkvRows, cols, "qkv");
    require_matrix(z, kZRows, cols, "z");
    require_w8_rowsplit(query_key_value_z_weight, kRows, "query/key/value/z weight");

    detail::w8_gdn_input_dispatch(x, query_key_value_z_weight, qkv, z, stream);
}

std::size_t gdn_input_proj_conv_snapshot_workspace_capacity_bytes(std::int32_t query_rows,
                                                                  std::int32_t key_rows,
                                                                  std::int32_t value_rows,
                                                                  std::int32_t min_tokens,
                                                                  std::int32_t max_tokens) {
    const bool q4_q5 = query_rows == 2048 && key_rows == 2048 && value_rows == 6144;
    const bool w8    = query_rows == 2048 && key_rows == 2048 && value_rows == 4096;
    if ((!q4_q5 && !w8) || min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument(
            "gdn_input_proj_conv_snapshot: unregistered shape or invalid token interval");
    }
    const std::int32_t channels = query_rows + key_rows + value_rows;
    const auto exact_capacity   = [&](std::int32_t tokens) {
        WorkspaceLayoutBuilder layout;
        const SnapshotRoute route = resolve_snapshot_route(q4_q5, tokens);
        (void)allocate_snapshot_workspace(layout, channels, tokens, route.workspace);
        return layout.peak_bytes(1);
    };

    std::size_t maximum = 0;
    if (q4_q5 && min_tokens <= 4 && max_tokens >= 4) { maximum = exact_capacity(4); }
    const std::int32_t composed_first = q4_q5 ? 7 : 17;
    if (max_tokens >= composed_first) { maximum = std::max(maximum, exact_capacity(max_tokens)); }
    (void)resolve_snapshot_route(q4_q5, min_tokens);
    (void)resolve_snapshot_route(q4_q5, max_tokens);
    return maximum;
}

void gdn_input_proj_conv_snapshot(const Tensor& x, const Weight& qk_weight,
                                  const Weight& value_weight, const Tensor& conv_weight,
                                  Tensor& conv_states, const Tensor& initial_slot, Tensor& query,
                                  Tensor& key, Tensor& value, WorkspaceArena& ws,
                                  cudaStream_t stream) {
    constexpr std::int32_t kHidden    = 5120;
    constexpr std::int32_t kQueryRows = 2048;
    constexpr std::int32_t kKeyRows   = 2048;
    constexpr std::int32_t kValueRows = 6144;
    constexpr std::int32_t kChannels  = kQueryRows + kKeyRows + kValueRows;
    const std::int32_t tokens         = x.ne[1];
    if (tokens <= 0) {
        throw std::invalid_argument("gdn_input_proj_conv_snapshot: T must be positive");
    }
    require_matrix(x, kHidden, tokens, "x");
    require_rowsplit(qk_weight, QType::Q4G64_F16S, kQueryRows + kKeyRows, "qk weight");
    require_rowsplit(value_weight, QType::Q5G64_F16S, kValueRows, "value weight");
    require_snapshot_operands(conv_weight, conv_states, initial_slot, kChannels, tokens);
    require_matrix(query, kQueryRows, tokens, "query");
    require_matrix(key, kKeyRows, tokens, "key");
    require_matrix(value, kValueRows, tokens, "value");

    const SnapshotRoute route = resolve_snapshot_route(true, tokens);
    if (route.workspace == SnapshotWorkspaceKind::None) {
        detail::q4_q5_gdn_input_conv_snapshot_launch(x, qk_weight, value_weight, conv_weight,
                                                     conv_states, initial_slot, query, key, value,
                                                     stream);
        return;
    }

    auto scope                = ws.scope();
    SnapshotWorkspace scratch = allocate_snapshot_workspace(ws, kChannels, tokens, route.workspace);
    gdn_input_proj(x, qk_weight, value_weight, scratch.projected, stream);
    if (route.workspace == SnapshotWorkspaceKind::Projected) {
        detail::q4_q5_gdn_input_t4_post_snapshot_launch(scratch.projected, conv_weight, conv_states,
                                                        initial_slot, query, key, value, stream);
    } else {
        causal_conv1d_silu_snapshot(scratch.projected, conv_weight, conv_states, initial_slot,
                                    scratch.convolved, stream);
        extract_bf16_columns(scratch.convolved, 0, query, stream);
        extract_bf16_columns(scratch.convolved, kQueryRows, key, stream);
        extract_bf16_columns(scratch.convolved, kQueryRows + kKeyRows, value, stream);
    }
}

void gdn_input_proj_conv_snapshot(const Tensor& x, const Weight& query_key_value_z_weight,
                                  const Tensor& conv_weight, Tensor& conv_states,
                                  const Tensor& initial_slot, Tensor& query, Tensor& key,
                                  Tensor& value, Tensor& z, WorkspaceArena& ws,
                                  cudaStream_t stream) {
    constexpr std::int32_t kHidden    = 2048;
    constexpr std::int32_t kQueryRows = 2048;
    constexpr std::int32_t kKeyRows   = 2048;
    constexpr std::int32_t kValueRows = 4096;
    constexpr std::int32_t kZRows     = 4096;
    constexpr std::int32_t kChannels  = kQueryRows + kKeyRows + kValueRows;
    const std::int32_t tokens         = x.ne[1];
    if (tokens <= 0) {
        throw std::invalid_argument("gdn_input_proj_conv_snapshot: T must be positive");
    }
    require_matrix(x, kHidden, tokens, "x");
    require_w8_rowsplit(query_key_value_z_weight, kChannels + kZRows, "query/key/value/z weight");
    require_snapshot_operands(conv_weight, conv_states, initial_slot, kChannels, tokens);
    require_matrix(query, kQueryRows, tokens, "query");
    require_matrix(key, kKeyRows, tokens, "key");
    require_matrix(value, kValueRows, tokens, "value");
    require_matrix(z, kZRows, tokens, "z");

    const SnapshotRoute route = resolve_snapshot_route(false, tokens);
    if (route.w8_schedule == detail::W8GdnInputSnapshotScheduleId::DecodeFused) {
        detail::w8_gdn_input_decode_conv_snapshot_launch(x, query_key_value_z_weight, conv_weight,
                                                         conv_states, initial_slot, query, key,
                                                         value, z, stream);
        return;
    }
    if (route.w8_schedule == detail::W8GdnInputSnapshotScheduleId::SplitKMmaFused) {
        detail::w8_gdn_input_splitk_conv_snapshot_launch(x, query_key_value_z_weight, conv_weight,
                                                         conv_states, initial_slot, query, key,
                                                         value, z, stream);
        return;
    }

    auto scope                = ws.scope();
    SnapshotWorkspace scratch = allocate_snapshot_workspace(ws, kChannels, tokens, route.workspace);
    gdn_input_proj(x, query_key_value_z_weight, scratch.projected, z, stream);
    causal_conv1d_silu_snapshot(scratch.projected, conv_weight, conv_states, initial_slot,
                                scratch.convolved, stream);
    extract_bf16_columns(scratch.convolved, 0, query, stream);
    extract_bf16_columns(scratch.convolved, kQueryRows, key, stream);
    extract_bf16_columns(scratch.convolved, kQueryRows + kKeyRows, value, stream);
}

} // namespace ninfer::ops
