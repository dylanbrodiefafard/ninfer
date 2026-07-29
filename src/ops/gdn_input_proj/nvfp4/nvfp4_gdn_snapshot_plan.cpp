#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.h"

#include "core/layout.h"
#include "ops/linear/nvfp4/nvfp4_config.h"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

enum class Nvfp4GdnSnapshotRoute {
    DecodeFusedA16,
    SmallTFusedA16,
    LinearW4A4Post,
};

Nvfp4GdnSnapshotRoute resolve_route(LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0) { throw std::invalid_argument("nvfp4 gdn snapshot: T must be positive"); }
    if (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA4) {
        throw std::invalid_argument("nvfp4 gdn snapshot admits only A16 or A4");
    }
    if (tokens == 1) { return Nvfp4GdnSnapshotRoute::DecodeFusedA16; }
    if (tokens <= 16) { return Nvfp4GdnSnapshotRoute::SmallTFusedA16; }
    if (policy == LinearPolicy::A16Only) {
        throw std::invalid_argument("nvfp4 gdn snapshot A16 is registered only through T=16");
    }
    return Nvfp4GdnSnapshotRoute::LinearW4A4Post;
}

struct Nvfp4GdnSnapshotWorkspace {
    Tensor projected;
    DeviceSpan linear;
};

template <class Allocator>
Nvfp4GdnSnapshotWorkspace allocate_workspace(Allocator& allocator, std::int32_t tokens) {
    Nvfp4GdnSnapshotWorkspace out;
    out.projected = allocator.alloc(DType::BF16, {Nvfp4GdnInputGeometry::kOutputRows, tokens}, 256);
    const std::size_t linear_bytes = linear_workspace_capacity_bytes(
        QType::NVFP4, Nvfp4GdnInputGeometry::kOutputRows, Nvfp4GdnInputGeometry::kInputRows,
        LinearPolicy::AllowA4, tokens, tokens);
    out.linear = allocator.alloc_bytes(linear_bytes, 256);
    return out;
}

} // namespace

std::size_t nvfp4_gdn_snapshot_workspace_capacity_bytes(LinearPolicy policy,
                                                        std::int32_t min_tokens,
                                                        std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("nvfp4 gdn snapshot workspace: invalid token interval");
    }
    (void)resolve_route(policy, min_tokens);
    const Nvfp4GdnSnapshotRoute maximum_route = resolve_route(policy, max_tokens);
    if (maximum_route != Nvfp4GdnSnapshotRoute::LinearW4A4Post) { return 0; }

    WorkspaceLayoutBuilder layout;
    (void)allocate_workspace(layout, max_tokens);
    return layout.peak_bytes(1);
}

void nvfp4_gdn_snapshot_dispatch(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                                 Tensor& conv_states, const Tensor& initial_slot, Tensor& query,
                                 Tensor& key, Tensor& value, Tensor& z, LinearPolicy policy,
                                 WorkspaceArena& workspace, cudaStream_t stream) {
    switch (resolve_route(policy, x.ne[1])) {
    case Nvfp4GdnSnapshotRoute::DecodeFusedA16:
        nvfp4_gdn_snapshot_decode_launch(x, weight, conv_weight, conv_states, initial_slot, query,
                                         key, value, z, stream);
        return;
    case Nvfp4GdnSnapshotRoute::SmallTFusedA16:
        nvfp4_gdn_snapshot_small_t_launch(x, weight, conv_weight, conv_states, initial_slot, query,
                                          key, value, z, stream);
        return;
    case Nvfp4GdnSnapshotRoute::LinearW4A4Post:
        break;
    }

    auto scope                        = workspace.scope();
    Nvfp4GdnSnapshotWorkspace scratch = allocate_workspace(workspace, x.ne[1]);
    WorkspaceArena linear_workspace(scratch.linear);
    linear(x, weight, scratch.projected, LinearPolicy::AllowA4, linear_workspace, stream);
    nvfp4_gdn_snapshot_post_launch(scratch.projected, conv_weight, conv_states, initial_slot, query,
                                   key, value, z, stream);
}

} // namespace ninfer::ops::detail
