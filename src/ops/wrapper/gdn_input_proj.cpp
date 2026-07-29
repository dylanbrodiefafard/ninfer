#include "ninfer/ops/gdn_input_proj.h"

#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/scatter.h"

#include "core/layout.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.h"
#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_kernels.h"
#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.h"
#include "ops/gdn_input_proj/w8/w8_gdn_input_kernels.h"
#include "ops/gdn_input_proj/w8/w8_gdn_input_plan.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_format.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"

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

bool overlaps(const Tensor& lhs, const Tensor& rhs) {
    const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs.data);
    const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs.data);
    return lhs_begin < rhs_begin + rhs.bytes() && rhs_begin < lhs_begin + lhs.bytes();
}

void require_single_parent_nonoverlap(const Tensor& x, const Tensor& qkv, const Tensor& z) {
    if (overlaps(x, qkv) || overlaps(x, z) || overlaps(qkv, z)) {
        throw std::invalid_argument("gdn_input_proj: x, qkv, and z must not overlap");
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

void validate_policy(LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA4:
        return;
    }
    throw std::invalid_argument("gdn_input_proj: invalid compute policy");
}

void dispatch_single_parent(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                            LinearPolicy policy, WorkspaceArena* workspace, cudaStream_t stream) {
    validate_policy(policy);
    const std::int32_t cols = x.ne[1];
    if (cols <= 0) { throw std::invalid_argument("gdn_input_proj: T must be positive"); }

    if (weight.qtype == QType::NVFP4) {
        constexpr std::int32_t kHidden  = 5120;
        constexpr std::int32_t kQkvRows = 10240;
        constexpr std::int32_t kZRows   = 6144;
        constexpr std::int32_t kRows    = kQkvRows + kZRows;
        if (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA4) {
            throw std::invalid_argument("NVFP4 gdn_input_proj admits only A16 or A4");
        }
        require_matrix(x, kHidden, cols, "x");
        require_matrix(qkv, kQkvRows, cols, "qkv");
        require_matrix(z, kZRows, cols, "z");
        require_single_parent_nonoverlap(x, qkv, z);
        detail::validate_nvfp4_weight(weight, "nvfp4 gdn_input_proj");
        if (weight.n != kRows || weight.k != kHidden) {
            throw std::invalid_argument("nvfp4 gdn_input_proj: unsupported weight shape");
        }
        detail::nvfp4_gdn_input_dispatch(x, weight, qkv, z, policy, workspace, stream);
        return;
    }

    constexpr std::int32_t kHidden  = 2048;
    constexpr std::int32_t kQkvRows = 8192;
    constexpr std::int32_t kZRows   = 4096;
    constexpr std::int32_t kRows    = kQkvRows + kZRows;
    if (policy != LinearPolicy::A16Only) {
        throw std::invalid_argument("W8 gdn_input_proj admits only A16");
    }
    require_matrix(x, kHidden, cols, "x");
    require_matrix(qkv, kQkvRows, cols, "qkv");
    require_matrix(z, kZRows, cols, "z");
    require_single_parent_nonoverlap(x, qkv, z);
    require_w8_rowsplit(weight, kRows, "query/key/value/z weight");
    detail::w8_gdn_input_dispatch(x, weight, qkv, z, stream);
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

void dispatch_single_parent_snapshot(const Tensor& x, const Weight& weight,
                                     const Tensor& conv_weight, Tensor& conv_states,
                                     const Tensor& initial_slot, Tensor& query, Tensor& key,
                                     Tensor& value, Tensor& z, LinearPolicy policy,
                                     WorkspaceArena& workspace, cudaStream_t stream) {
    validate_policy(policy);
    const std::int32_t tokens = x.ne[1];
    if (tokens <= 0) {
        throw std::invalid_argument("gdn_input_proj_conv_snapshot: T must be positive");
    }

    if (weight.qtype == QType::NVFP4) {
        constexpr std::int32_t kHidden     = 5120;
        constexpr std::int32_t kQueryRows  = 2048;
        constexpr std::int32_t kKeyRows    = 2048;
        constexpr std::int32_t kValueRows  = 6144;
        constexpr std::int32_t kZRows      = 6144;
        constexpr std::int32_t kChannels   = kQueryRows + kKeyRows + kValueRows;
        constexpr std::int32_t kParentRows = kChannels + kZRows;
        require_matrix(x, kHidden, tokens, "x");
        detail::validate_nvfp4_weight(weight, "nvfp4 gdn_input_proj_conv_snapshot");
        if (weight.n != kParentRows || weight.k != kHidden) {
            throw std::invalid_argument(
                "nvfp4 gdn_input_proj_conv_snapshot: unsupported weight shape");
        }
        require_snapshot_operands(conv_weight, conv_states, initial_slot, kChannels, tokens);
        require_matrix(query, kQueryRows, tokens, "query");
        require_matrix(key, kKeyRows, tokens, "key");
        require_matrix(value, kValueRows, tokens, "value");
        require_matrix(z, kZRows, tokens, "z");
        detail::nvfp4_gdn_snapshot_dispatch(x, weight, conv_weight, conv_states, initial_slot,
                                            query, key, value, z, policy, workspace, stream);
        return;
    }

    constexpr std::int32_t kHidden    = 2048;
    constexpr std::int32_t kQueryRows = 2048;
    constexpr std::int32_t kKeyRows   = 2048;
    constexpr std::int32_t kValueRows = 4096;
    constexpr std::int32_t kZRows     = 4096;
    constexpr std::int32_t kChannels  = kQueryRows + kKeyRows + kValueRows;
    if (policy != LinearPolicy::A16Only) {
        throw std::invalid_argument("W8 gdn_input_proj_conv_snapshot admits only A16");
    }
    require_matrix(x, kHidden, tokens, "x");
    require_w8_rowsplit(weight, kChannels + kZRows, "query/key/value/z weight");
    require_snapshot_operands(conv_weight, conv_states, initial_slot, kChannels, tokens);
    require_matrix(query, kQueryRows, tokens, "query");
    require_matrix(key, kKeyRows, tokens, "key");
    require_matrix(value, kValueRows, tokens, "value");
    require_matrix(z, kZRows, tokens, "z");

    const SnapshotRoute route = resolve_snapshot_route(false, tokens);
    if (route.w8_schedule == detail::W8GdnInputSnapshotScheduleId::DecodeFused) {
        detail::w8_gdn_input_decode_conv_snapshot_launch(
            x, weight, conv_weight, conv_states, initial_slot, query, key, value, z, stream);
        return;
    }
    if (route.w8_schedule == detail::W8GdnInputSnapshotScheduleId::SplitKMmaFused) {
        detail::w8_gdn_input_splitk_conv_snapshot_launch(
            x, weight, conv_weight, conv_states, initial_slot, query, key, value, z, stream);
        return;
    }

    auto scope = workspace.scope();
    SnapshotWorkspace scratch =
        allocate_snapshot_workspace(workspace, kChannels, tokens, route.workspace);
    gdn_input_proj(x, weight, scratch.projected, z, stream);
    causal_conv1d_silu_snapshot(scratch.projected, conv_weight, conv_states, initial_slot,
                                scratch.convolved, stream);
    extract_bf16_columns(scratch.convolved, 0, query, stream);
    extract_bf16_columns(scratch.convolved, kQueryRows, key, stream);
    extract_bf16_columns(scratch.convolved, kQueryRows + kKeyRows, value, stream);
}

} // namespace

void gdn_input_proj(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                    Tensor& qkv, Tensor& z, cudaStream_t stream) {
    constexpr std::int32_t kHidden     = 5120;
    constexpr std::int32_t kQkRows     = 4096;
    constexpr std::int32_t kValueRows  = 6144;
    constexpr std::int32_t kZRows      = 6144;
    constexpr std::int32_t kQkvRows    = kQkRows + kValueRows;
    constexpr std::int32_t kParentRows = kValueRows + kZRows;
    const std::int32_t cols            = x.ne[1];
    if (cols <= 0) { throw std::invalid_argument("gdn_input_proj: T must be positive"); }
    require_matrix(x, kHidden, cols, "x");
    require_matrix(qkv, kQkvRows, cols, "qkv");
    require_matrix(z, kZRows, cols, "z");
    require_rowsplit(qk_weight, QType::Q4G64_F16S, kQkRows, "qk weight");
    require_rowsplit(value_z_weight, QType::Q5G64_F16S, kParentRows, "value/z weight");

    detail::q4_q5_gdn_input_dispatch(x, qk_weight, value_z_weight, qkv, z, stream);
}

std::size_t gdn_input_proj_workspace_capacity_bytes(QType parent_qtype, std::int32_t parent_rows,
                                                    std::int32_t input_rows, LinearPolicy policy,
                                                    std::int32_t min_tokens,
                                                    std::int32_t max_tokens) {
    validate_policy(policy);
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("gdn_input_proj workspace: invalid token interval");
    }
    if (parent_qtype == QType::NVFP4) {
        if (parent_rows != detail::Nvfp4GdnInputGeometry::kOutputRows ||
            input_rows != detail::Nvfp4GdnInputGeometry::kInputRows ||
            (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA4)) {
            throw std::invalid_argument("gdn_input_proj workspace: unsupported NVFP4 profile");
        }
        if (policy == LinearPolicy::A16Only || max_tokens < detail::kNvfp4FirstA4T) { return 0; }
        return detail::nvfp4_w4a4_workspace_capacity_bytes(max_tokens, input_rows);
    }
    if (parent_qtype == QType::W8G32_F16S && parent_rows == 12288 && input_rows == 2048 &&
        policy == LinearPolicy::A16Only) {
        (void)detail::w8_gdn_input_resolve_plan(
            {input_rows, 8192, 4096, parent_rows, input_rows, min_tokens});
        (void)detail::w8_gdn_input_resolve_plan(
            {input_rows, 8192, 4096, parent_rows, input_rows, max_tokens});
        return 0;
    }
    throw std::invalid_argument("gdn_input_proj workspace: unsupported parent profile");
}

void gdn_input_proj(const Tensor& x, const Weight& query_key_value_z_weight, Tensor& qkv, Tensor& z,
                    LinearPolicy policy, WorkspaceArena& workspace, cudaStream_t stream) {
    dispatch_single_parent(x, query_key_value_z_weight, qkv, z, policy, &workspace, stream);
}

void gdn_input_proj(const Tensor& x, const Weight& query_key_value_z_weight, Tensor& qkv, Tensor& z,
                    cudaStream_t stream) {
    dispatch_single_parent(x, query_key_value_z_weight, qkv, z, LinearPolicy::A16Only, nullptr,
                           stream);
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

std::size_t gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
    QType parent_qtype, std::int32_t parent_rows, std::int32_t input_rows, LinearPolicy policy,
    std::int32_t min_tokens, std::int32_t max_tokens) {
    validate_policy(policy);
    if (parent_qtype != QType::NVFP4 || parent_rows != detail::Nvfp4GdnInputGeometry::kOutputRows ||
        input_rows != detail::Nvfp4GdnInputGeometry::kInputRows) {
        throw std::invalid_argument(
            "gdn_input_proj_conv_snapshot workspace: unsupported single-parent profile");
    }
    return detail::nvfp4_gdn_snapshot_workspace_capacity_bytes(policy, min_tokens, max_tokens);
}

void gdn_input_proj_conv_snapshot(const Tensor& x, const Weight& qk_weight,
                                  const Weight& value_z_weight, const Tensor& conv_weight,
                                  Tensor& conv_states, const Tensor& initial_slot, Tensor& query,
                                  Tensor& key, Tensor& value, Tensor& z, WorkspaceArena& ws,
                                  cudaStream_t stream) {
    constexpr std::int32_t kHidden     = 5120;
    constexpr std::int32_t kQueryRows  = 2048;
    constexpr std::int32_t kKeyRows    = 2048;
    constexpr std::int32_t kValueRows  = 6144;
    constexpr std::int32_t kZRows      = 6144;
    constexpr std::int32_t kChannels   = kQueryRows + kKeyRows + kValueRows;
    constexpr std::int32_t kParentRows = kValueRows + kZRows;
    const std::int32_t tokens          = x.ne[1];
    if (tokens <= 0) {
        throw std::invalid_argument("gdn_input_proj_conv_snapshot: T must be positive");
    }
    require_matrix(x, kHidden, tokens, "x");
    require_rowsplit(qk_weight, QType::Q4G64_F16S, kQueryRows + kKeyRows, "qk weight");
    require_rowsplit(value_z_weight, QType::Q5G64_F16S, kParentRows, "value/z weight");
    require_snapshot_operands(conv_weight, conv_states, initial_slot, kChannels, tokens);
    require_matrix(query, kQueryRows, tokens, "query");
    require_matrix(key, kKeyRows, tokens, "key");
    require_matrix(value, kValueRows, tokens, "value");
    require_matrix(z, kZRows, tokens, "z");

    const SnapshotRoute route = resolve_snapshot_route(true, tokens);
    if (route.workspace == SnapshotWorkspaceKind::None) {
        detail::q4_q5_gdn_input_conv_snapshot_launch(x, qk_weight, value_z_weight, conv_weight,
                                                     conv_states, initial_slot, query, key, value,
                                                     z, stream);
        return;
    }

    auto scope                = ws.scope();
    SnapshotWorkspace scratch = allocate_snapshot_workspace(ws, kChannels, tokens, route.workspace);
    gdn_input_proj(x, qk_weight, value_z_weight, scratch.projected, z, stream);
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
                                  Tensor& value, Tensor& z, LinearPolicy policy, WorkspaceArena& ws,
                                  cudaStream_t stream) {
    dispatch_single_parent_snapshot(x, query_key_value_z_weight, conv_weight, conv_states,
                                    initial_slot, query, key, value, z, policy, ws, stream);
}

void gdn_input_proj_conv_snapshot(const Tensor& x, const Weight& query_key_value_z_weight,
                                  const Tensor& conv_weight, Tensor& conv_states,
                                  const Tensor& initial_slot, Tensor& query, Tensor& key,
                                  Tensor& value, Tensor& z, WorkspaceArena& ws,
                                  cudaStream_t stream) {
    dispatch_single_parent_snapshot(x, query_key_value_z_weight, conv_weight, conv_states,
                                    initial_slot, query, key, value, z, LinearPolicy::A16Only, ws,
                                    stream);
}

} // namespace ninfer::ops
