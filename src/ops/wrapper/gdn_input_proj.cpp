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

void require_snapshot_tensor(const Tensor& tensor, std::int32_t rows, std::int32_t width,
                             std::int32_t batch, const char* label) {
    if (tensor.dtype != DType::BF16 || tensor.ne[0] != rows || tensor.ne[1] != width ||
        tensor.ne[2] != batch || tensor.ne[3] != 1 || !tensor.is_contiguous() ||
        !aligned_to(tensor.data, 16)) {
        throw std::invalid_argument(std::string("gdn_input_proj_conv_snapshot: invalid ") + label);
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

struct SnapshotGeometry {
    std::int32_t width;
    std::int32_t batch;
    std::int32_t aggregate_columns;
};

SnapshotGeometry require_snapshot_input(const Tensor& x, std::int32_t hidden) {
    constexpr std::int32_t kMaximumBatch = 8;
    constexpr std::int32_t kMaximumWidth = 16;
    const std::int32_t width             = x.ne[1];
    const std::int32_t batch             = x.ne[2];
    if (width <= 0 || batch <= 0 || batch > kMaximumBatch || (batch > 1 && width > kMaximumWidth)) {
        throw std::invalid_argument("gdn_input_proj_conv_snapshot: unsupported B/W domain");
    }
    require_snapshot_tensor(x, hidden, width, batch, "x");
    return {width, batch, width * batch};
}

void require_snapshot_operands(const Tensor& conv_weight, const Tensor& conv_states,
                               const Tensor& valid_columns, const Tensor& initial_state_slots,
                               const Tensor& snapshot_base_slots, std::int32_t channels,
                               SnapshotGeometry geometry) {
    require_matrix(conv_weight, channels, 4, "conv weight");
    if (conv_states.dtype != DType::BF16 || conv_states.ne[0] != channels ||
        conv_states.ne[1] != 3 || conv_states.ne[2] < geometry.aggregate_columns ||
        conv_states.ne[3] != 1 || !conv_states.is_contiguous() ||
        !aligned_to(conv_states.data, 16)) {
        throw std::invalid_argument(
            "gdn_input_proj_conv_snapshot: invalid convolution snapshot state");
    }
    const auto valid_selector = [batch = geometry.batch](const Tensor& selector) {
        return selector.dtype == DType::I32 && selector.ne[0] == batch && selector.ne[1] == 1 &&
               selector.ne[2] == 1 && selector.ne[3] == 1 && selector.is_contiguous() &&
               selector.data != nullptr;
    };
    if (!valid_selector(initial_state_slots) || !valid_selector(snapshot_base_slots)) {
        throw std::invalid_argument("gdn_input_proj_conv_snapshot: invalid state selector");
    }
    if (valid_columns.data != nullptr) {
        if (geometry.batch == 1 || !valid_selector(valid_columns)) {
            throw std::invalid_argument("gdn_input_proj_conv_snapshot: invalid valid columns");
        }
    }
}

Tensor flatten_snapshot_tensor(const Tensor& tensor, std::int32_t rows, SnapshotGeometry geometry) {
    return Tensor(tensor.data, tensor.dtype, {rows, geometry.aggregate_columns});
}

void require_snapshot_capacity_domain(std::int32_t batch_size, std::int32_t min_width,
                                      std::int32_t max_width) {
    constexpr std::int32_t kMaximumBatch = 8;
    constexpr std::int32_t kMaximumWidth = 16;
    if (batch_size <= 0 || batch_size > kMaximumBatch || min_width <= 0 || max_width < min_width ||
        (batch_size > 1 && max_width > kMaximumWidth)) {
        throw std::invalid_argument("gdn_input_proj_conv_snapshot workspace: invalid B/W domain");
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

std::size_t composed_snapshot_capacity(std::int32_t channels, std::int32_t aggregate_columns,
                                       std::size_t projection_workspace_bytes) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_snapshot_workspace(layout, channels, aggregate_columns,
                                      SnapshotWorkspaceKind::ProjectedAndConvolved);
    if (projection_workspace_bytes != 0) { (void)layout.alloc_bytes(projection_workspace_bytes); }
    return layout.peak_bytes(1);
}

template <class Project>
void compose_batched_snapshot(const Tensor& x, const Tensor& conv_weight, Tensor& conv_states,
                              const Tensor& valid_columns, const Tensor& initial_state_slots,
                              const Tensor& snapshot_base_slots, Tensor& query, Tensor& key,
                              Tensor& value, Tensor& z, std::int32_t query_rows,
                              std::int32_t key_rows, std::int32_t value_rows,
                              SnapshotGeometry geometry, WorkspaceArena& workspace,
                              cudaStream_t stream, Project&& project) {
    const std::int32_t channels = query_rows + key_rows + value_rows;
    auto scope                  = workspace.scope();
    SnapshotWorkspace scratch =
        allocate_snapshot_workspace(workspace, channels, geometry.aggregate_columns,
                                    SnapshotWorkspaceKind::ProjectedAndConvolved);

    Tensor x_flat     = flatten_snapshot_tensor(x, x.ne[0], geometry);
    Tensor z_flat     = flatten_snapshot_tensor(z, z.ne[0], geometry);
    Tensor query_flat = flatten_snapshot_tensor(query, query_rows, geometry);
    Tensor key_flat   = flatten_snapshot_tensor(key, key_rows, geometry);
    Tensor value_flat = flatten_snapshot_tensor(value, value_rows, geometry);
    project(x_flat, scratch.projected, z_flat);

    Tensor projected(scratch.projected.data, DType::BF16,
                     {channels, geometry.width, geometry.batch});
    Tensor convolved(scratch.convolved.data, DType::BF16,
                     {channels, geometry.width, geometry.batch});
    causal_conv1d_silu_snapshot(projected, conv_weight, conv_states, valid_columns,
                                initial_state_slots, snapshot_base_slots, convolved, stream);
    extract_bf16_columns(scratch.convolved, 0, query_flat, stream);
    extract_bf16_columns(scratch.convolved, query_rows, key_flat, stream);
    extract_bf16_columns(scratch.convolved, query_rows + key_rows, value_flat, stream);
}

void dispatch_single_parent_snapshot(const Tensor& x, const Weight& weight,
                                     const Tensor& conv_weight, Tensor& conv_states,
                                     const Tensor& valid_columns, const Tensor& initial_state_slots,
                                     const Tensor& snapshot_base_slots, Tensor& query, Tensor& key,
                                     Tensor& value, Tensor& z, LinearPolicy policy,
                                     WorkspaceArena& workspace, cudaStream_t stream) {
    validate_policy(policy);

    if (weight.qtype == QType::NVFP4) {
        constexpr std::int32_t kHidden     = 5120;
        constexpr std::int32_t kQueryRows  = 2048;
        constexpr std::int32_t kKeyRows    = 2048;
        constexpr std::int32_t kValueRows  = 6144;
        constexpr std::int32_t kZRows      = 6144;
        constexpr std::int32_t kChannels   = kQueryRows + kKeyRows + kValueRows;
        constexpr std::int32_t kParentRows = kChannels + kZRows;
        const SnapshotGeometry geometry    = require_snapshot_input(x, kHidden);
        detail::validate_nvfp4_weight(weight, "nvfp4 gdn_input_proj_conv_snapshot");
        if (weight.n != kParentRows || weight.k != kHidden) {
            throw std::invalid_argument(
                "nvfp4 gdn_input_proj_conv_snapshot: unsupported weight shape");
        }
        require_snapshot_operands(conv_weight, conv_states, valid_columns, initial_state_slots,
                                  snapshot_base_slots, kChannels, geometry);
        require_snapshot_tensor(query, kQueryRows, geometry.width, geometry.batch, "query");
        require_snapshot_tensor(key, kKeyRows, geometry.width, geometry.batch, "key");
        require_snapshot_tensor(value, kValueRows, geometry.width, geometry.batch, "value");
        require_snapshot_tensor(z, kZRows, geometry.width, geometry.batch, "z");
        if (geometry.batch > 1) {
            compose_batched_snapshot(x, conv_weight, conv_states, valid_columns,
                                     initial_state_slots, snapshot_base_slots, query, key, value, z,
                                     kQueryRows, kKeyRows, kValueRows, geometry, workspace, stream,
                                     [&](const Tensor& x_flat, Tensor& projected, Tensor& z_flat) {
                                         gdn_input_proj(x_flat, weight, projected, z_flat, policy,
                                                        workspace, stream);
                                     });
            return;
        }
        detail::nvfp4_gdn_snapshot_dispatch(x, weight, conv_weight, conv_states,
                                            initial_state_slots, snapshot_base_slots, query, key,
                                            value, z, policy, workspace, stream);
        return;
    }

    constexpr std::int32_t kHidden    = 2048;
    constexpr std::int32_t kQueryRows = 2048;
    constexpr std::int32_t kKeyRows   = 2048;
    constexpr std::int32_t kValueRows = 4096;
    constexpr std::int32_t kZRows     = 4096;
    constexpr std::int32_t kChannels  = kQueryRows + kKeyRows + kValueRows;
    const SnapshotGeometry geometry   = require_snapshot_input(x, kHidden);
    if (policy != LinearPolicy::A16Only) {
        throw std::invalid_argument("W8 gdn_input_proj_conv_snapshot admits only A16");
    }
    require_w8_rowsplit(weight, kChannels + kZRows, "query/key/value/z weight");
    require_snapshot_operands(conv_weight, conv_states, valid_columns, initial_state_slots,
                              snapshot_base_slots, kChannels, geometry);
    require_snapshot_tensor(query, kQueryRows, geometry.width, geometry.batch, "query");
    require_snapshot_tensor(key, kKeyRows, geometry.width, geometry.batch, "key");
    require_snapshot_tensor(value, kValueRows, geometry.width, geometry.batch, "value");
    require_snapshot_tensor(z, kZRows, geometry.width, geometry.batch, "z");
    if (geometry.batch > 1) {
        compose_batched_snapshot(x, conv_weight, conv_states, valid_columns, initial_state_slots,
                                 snapshot_base_slots, query, key, value, z, kQueryRows, kKeyRows,
                                 kValueRows, geometry, workspace, stream,
                                 [&](const Tensor& x_flat, Tensor& projected, Tensor& z_flat) {
                                     gdn_input_proj(x_flat, weight, projected, z_flat, stream);
                                 });
        return;
    }

    const SnapshotRoute route = resolve_snapshot_route(false, geometry.width);
    if (route.w8_schedule == detail::W8GdnInputSnapshotScheduleId::DecodeFused) {
        detail::w8_gdn_input_decode_conv_snapshot_launch(x, weight, conv_weight, conv_states,
                                                         initial_state_slots, snapshot_base_slots,
                                                         query, key, value, z, stream);
        return;
    }
    if (route.w8_schedule == detail::W8GdnInputSnapshotScheduleId::SplitKMmaFused) {
        detail::w8_gdn_input_splitk_conv_snapshot_launch(x, weight, conv_weight, conv_states,
                                                         initial_state_slots, snapshot_base_slots,
                                                         query, key, value, z, stream);
        return;
    }

    auto scope = workspace.scope();
    SnapshotWorkspace scratch =
        allocate_snapshot_workspace(workspace, kChannels, geometry.width, route.workspace);
    gdn_input_proj(x, weight, scratch.projected, z, stream);
    causal_conv1d_silu_snapshot(scratch.projected, conv_weight, conv_states, Tensor{},
                                initial_state_slots, snapshot_base_slots, scratch.convolved,
                                stream);
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
        return detail::nvfp4_gdn_input_workspace_capacity_bytes(policy, min_tokens, max_tokens);
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

std::size_t gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
    std::int32_t query_rows, std::int32_t key_rows, std::int32_t value_rows,
    std::int32_t batch_size, std::int32_t min_width, std::int32_t max_width) {
    const bool q4_q5 = query_rows == 2048 && key_rows == 2048 && value_rows == 6144;
    const bool w8    = query_rows == 2048 && key_rows == 2048 && value_rows == 4096;
    if (!q4_q5 && !w8) {
        throw std::invalid_argument("gdn_input_proj_conv_snapshot workspace: unregistered shape");
    }
    require_snapshot_capacity_domain(batch_size, min_width, max_width);
    const std::int32_t channels = query_rows + key_rows + value_rows;
    if (batch_size > 1) {
        std::size_t maximum = 0;
        for (std::int32_t width = min_width; width <= max_width; ++width) {
            maximum =
                std::max(maximum, composed_snapshot_capacity(channels, batch_size * width, 0));
        }
        return maximum;
    }

    const auto exact_capacity = [&](std::int32_t tokens) {
        WorkspaceLayoutBuilder layout;
        const SnapshotRoute route = resolve_snapshot_route(q4_q5, tokens);
        (void)allocate_snapshot_workspace(layout, channels, tokens, route.workspace);
        return layout.peak_bytes(1);
    };

    std::size_t maximum = 0;
    if (q4_q5 && min_width <= 4 && max_width >= 4) { maximum = exact_capacity(4); }
    const std::int32_t composed_first = q4_q5 ? 7 : 17;
    if (max_width >= composed_first) { maximum = std::max(maximum, exact_capacity(max_width)); }
    (void)resolve_snapshot_route(q4_q5, min_width);
    (void)resolve_snapshot_route(q4_q5, max_width);
    return maximum;
}

std::size_t gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
    QType parent_qtype, std::int32_t parent_rows, std::int32_t input_rows, LinearPolicy policy,
    std::int32_t batch_size, std::int32_t min_width, std::int32_t max_width) {
    validate_policy(policy);
    if (parent_qtype != QType::NVFP4 || parent_rows != detail::Nvfp4GdnInputGeometry::kOutputRows ||
        input_rows != detail::Nvfp4GdnInputGeometry::kInputRows) {
        throw std::invalid_argument(
            "gdn_input_proj_conv_snapshot workspace: unsupported single-parent profile");
    }
    require_snapshot_capacity_domain(batch_size, min_width, max_width);
    if (batch_size == 1) {
        return detail::nvfp4_gdn_snapshot_workspace_capacity_bytes(policy, min_width, max_width);
    }

    constexpr std::int32_t kChannels = 10240;
    std::size_t maximum              = 0;
    for (std::int32_t width = min_width; width <= max_width; ++width) {
        const std::int32_t aggregate_columns   = batch_size * width;
        const std::size_t projection_workspace = gdn_input_proj_workspace_capacity_bytes(
            parent_qtype, parent_rows, input_rows, policy, aggregate_columns, aggregate_columns);
        maximum = std::max(maximum, composed_snapshot_capacity(kChannels, aggregate_columns,
                                                               projection_workspace));
    }
    return maximum;
}

void gdn_input_proj_conv_snapshot(const Tensor& x, const Weight& qk_weight,
                                  const Weight& value_z_weight, const Tensor& conv_weight,
                                  Tensor& conv_states, const Tensor& valid_columns,
                                  const Tensor& initial_state_slots,
                                  const Tensor& snapshot_base_slots, Tensor& query, Tensor& key,
                                  Tensor& value, Tensor& z, WorkspaceArena& ws,
                                  cudaStream_t stream) {
    constexpr std::int32_t kHidden     = 5120;
    constexpr std::int32_t kQueryRows  = 2048;
    constexpr std::int32_t kKeyRows    = 2048;
    constexpr std::int32_t kValueRows  = 6144;
    constexpr std::int32_t kZRows      = 6144;
    constexpr std::int32_t kChannels   = kQueryRows + kKeyRows + kValueRows;
    constexpr std::int32_t kParentRows = kValueRows + kZRows;
    const SnapshotGeometry geometry    = require_snapshot_input(x, kHidden);
    require_rowsplit(qk_weight, QType::Q4G64_F16S, kQueryRows + kKeyRows, "qk weight");
    require_rowsplit(value_z_weight, QType::Q5G64_F16S, kParentRows, "value/z weight");
    require_snapshot_operands(conv_weight, conv_states, valid_columns, initial_state_slots,
                              snapshot_base_slots, kChannels, geometry);
    require_snapshot_tensor(query, kQueryRows, geometry.width, geometry.batch, "query");
    require_snapshot_tensor(key, kKeyRows, geometry.width, geometry.batch, "key");
    require_snapshot_tensor(value, kValueRows, geometry.width, geometry.batch, "value");
    require_snapshot_tensor(z, kZRows, geometry.width, geometry.batch, "z");

    if (geometry.batch > 1) {
        compose_batched_snapshot(
            x, conv_weight, conv_states, valid_columns, initial_state_slots, snapshot_base_slots,
            query, key, value, z, kQueryRows, kKeyRows, kValueRows, geometry, ws, stream,
            [&](const Tensor& x_flat, Tensor& projected, Tensor& z_flat) {
                gdn_input_proj(x_flat, qk_weight, value_z_weight, projected, z_flat, stream);
            });
        return;
    }

    const SnapshotRoute route = resolve_snapshot_route(true, geometry.width);
    if (route.workspace == SnapshotWorkspaceKind::None) {
        detail::q4_q5_gdn_input_conv_snapshot_launch(
            x, qk_weight, value_z_weight, conv_weight, conv_states, initial_state_slots,
            snapshot_base_slots, query, key, value, z, stream);
        return;
    }

    auto scope = ws.scope();
    SnapshotWorkspace scratch =
        allocate_snapshot_workspace(ws, kChannels, geometry.width, route.workspace);
    gdn_input_proj(x, qk_weight, value_z_weight, scratch.projected, z, stream);
    if (route.workspace == SnapshotWorkspaceKind::Projected) {
        detail::q4_q5_gdn_input_t4_post_snapshot_launch(scratch.projected, conv_weight, conv_states,
                                                        initial_state_slots, snapshot_base_slots,
                                                        query, key, value, stream);
    } else {
        causal_conv1d_silu_snapshot(scratch.projected, conv_weight, conv_states, Tensor{},
                                    initial_state_slots, snapshot_base_slots, scratch.convolved,
                                    stream);
        extract_bf16_columns(scratch.convolved, 0, query, stream);
        extract_bf16_columns(scratch.convolved, kQueryRows, key, stream);
        extract_bf16_columns(scratch.convolved, kQueryRows + kKeyRows, value, stream);
    }
}

void gdn_input_proj_conv_snapshot(const Tensor& x, const Weight& query_key_value_z_weight,
                                  const Tensor& conv_weight, Tensor& conv_states,
                                  const Tensor& valid_columns, const Tensor& initial_state_slots,
                                  const Tensor& snapshot_base_slots, Tensor& query, Tensor& key,
                                  Tensor& value, Tensor& z, LinearPolicy policy, WorkspaceArena& ws,
                                  cudaStream_t stream) {
    dispatch_single_parent_snapshot(x, query_key_value_z_weight, conv_weight, conv_states,
                                    valid_columns, initial_state_slots, snapshot_base_slots, query,
                                    key, value, z, policy, ws, stream);
}

void gdn_input_proj_conv_snapshot(const Tensor& x, const Weight& query_key_value_z_weight,
                                  const Tensor& conv_weight, Tensor& conv_states,
                                  const Tensor& valid_columns, const Tensor& initial_state_slots,
                                  const Tensor& snapshot_base_slots, Tensor& query, Tensor& key,
                                  Tensor& value, Tensor& z, WorkspaceArena& ws,
                                  cudaStream_t stream) {
    dispatch_single_parent_snapshot(x, query_key_value_z_weight, conv_weight, conv_states,
                                    valid_columns, initial_state_slots, snapshot_base_slots, query,
                                    key, value, z, LinearPolicy::A16Only, ws, stream);
}

} // namespace ninfer::ops
