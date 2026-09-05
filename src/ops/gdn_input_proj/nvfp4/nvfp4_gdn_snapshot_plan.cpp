#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.h"

#include "core/layout.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"
#include "ops/linear/nvfp4/nvfp4_config.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

inline constexpr std::int32_t kNvfp4RecordChannels = 10240;

struct Nvfp4GdnProjectedWorkspace {
    Tensor projected;
    DeviceSpan projection;
};

template <class Allocator>
Nvfp4GdnProjectedWorkspace allocate_workspace(Allocator& allocator, std::int32_t tokens) {
    Nvfp4GdnProjectedWorkspace out;
    out.projected = allocator.alloc(DType::BF16, {10240, tokens}, 256);
    const std::size_t projection_bytes =
        nvfp4_gdn_input_workspace_capacity_bytes(LinearPolicy::AllowA4, tokens, tokens);
    out.projection = allocator.alloc_bytes(projection_bytes, 256);
    return out;
}

} // namespace

Nvfp4GdnConvPlan nvfp4_gdn_conv_resolve_plan(LinearPolicy policy, std::int32_t tokens,
                                             std::int32_t batch_size) {
    if (tokens <= 0 || batch_size <= 0 || batch_size > 8) {
        throw std::invalid_argument("nvfp4 gdn conv: invalid B/T domain");
    }
    if (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA4) {
        throw std::invalid_argument("nvfp4 gdn conv admits only A16 or A4");
    }
    // Width selects the fused family. Snapshot T=2..16 is SmallT GEMM+FP32 conv. Record
    // B=1 retains fused T=1 GEMV+FP32 conv. Qualified B>1 W=2/5 shapes group requests
    // while retaining the W-local reduction and FP32 conv input; other widths use
    // request-indexed SmallT CTAs. Do not flatten to B*W W4A4 compose.
    if (tokens == 1) { return {Nvfp4GdnConvScheduleId::DecodeFusedA16}; }
    if (tokens <= kNvfp4LastPackedGdnConvSmallT) { return {Nvfp4GdnConvScheduleId::SmallTFusedA16}; }
    if (policy == LinearPolicy::A16Only) {
        throw std::invalid_argument("nvfp4 gdn conv A16 is registered only through T=16");
    }
    return {Nvfp4GdnConvScheduleId::Materialized};
}

std::size_t nvfp4_gdn_decode_columns_workspace_bytes() {
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::BF16, {2048 + 2048 + 6144, 3, 1});
    (void)layout.alloc(DType::I32, {1});
    (void)layout.alloc(DType::I32, {1});
    return layout.peak_bytes(1);
}

std::size_t nvfp4_gdn_snapshot_workspace_capacity_bytes(LinearPolicy policy,
                                                        std::int32_t min_tokens,
                                                        std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("nvfp4 gdn snapshot workspace: invalid token interval");
    }
    (void)nvfp4_gdn_conv_resolve_plan(policy, min_tokens, 1);
    const Nvfp4GdnConvPlan maximum_plan = nvfp4_gdn_conv_resolve_plan(policy, max_tokens, 1);
    if (maximum_plan.schedule != Nvfp4GdnConvScheduleId::Materialized) { return 0; }

    WorkspaceLayoutBuilder layout;
    (void)allocate_workspace(layout, max_tokens);
    return layout.peak_bytes(1);
}

std::size_t nvfp4_gdn_record_workspace_capacity_bytes(LinearPolicy policy, std::int32_t batch_size,
                                                      std::int32_t min_tokens,
                                                      std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens || batch_size <= 0 || batch_size > 4) {
        throw std::invalid_argument("nvfp4 gdn record workspace: invalid B/T domain");
    }
    const Nvfp4GdnConvPlan minimum_plan =
        nvfp4_gdn_conv_resolve_plan(policy, min_tokens, batch_size);
    const Nvfp4GdnConvPlan maximum_plan =
        nvfp4_gdn_conv_resolve_plan(policy, max_tokens, batch_size);
    if (minimum_plan.schedule == Nvfp4GdnConvScheduleId::DecodeFusedA16) {
        throw std::logic_error("ReplaySSM record planner admitted NVFP4 decode");
    }
    if (maximum_plan.schedule == Nvfp4GdnConvScheduleId::SmallTFusedA16) {
        std::int32_t maximum_grouped_width = 0;
        if (min_tokens <= 2 && max_tokens >= 2 &&
            nvfp4_gdn_record_uses_grouped_replay(2, batch_size)) {
            maximum_grouped_width = 2;
        }
        if (min_tokens <= 5 && max_tokens >= 5 &&
            nvfp4_gdn_record_uses_grouped_replay(5, batch_size)) {
            maximum_grouped_width = 5;
        }
        if (maximum_grouped_width == 0) { return 0; }
        WorkspaceLayoutBuilder layout;
        (void)layout.alloc(DType::FP32,
                           {kNvfp4RecordChannels, maximum_grouped_width, batch_size}, 256);
        return layout.peak_bytes(1);
    }
    if (batch_size > 1) {
        throw std::logic_error("batched NVFP4 conv-record has no Materialized route");
    }
    return nvfp4_gdn_input_workspace_capacity_bytes(LinearPolicy::AllowA4, std::max(min_tokens, 4),
                                                    max_tokens);
}

void nvfp4_gdn_snapshot_dispatch(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                                 Tensor& conv_states, const Tensor& valid_columns,
                                 const Tensor& initial_slot, const Tensor& snapshot_base_slot,
                                 Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                                 LinearPolicy policy, WorkspaceArena& workspace,
                                 cudaStream_t stream) {
    switch (nvfp4_gdn_conv_resolve_plan(policy, x.ne[1], 1).schedule) {
    case Nvfp4GdnConvScheduleId::DecodeFusedA16:
        nvfp4_gdn_snapshot_decode_launch(x, weight, conv_weight, conv_states, valid_columns,
                                         initial_slot, snapshot_base_slot, query, key, value, z,
                                         stream);
        return;
    case Nvfp4GdnConvScheduleId::SmallTFusedA16:
        nvfp4_gdn_snapshot_small_t_launch(x, weight, conv_weight, conv_states, valid_columns,
                                          initial_slot, snapshot_base_slot, query, key, value, z,
                                          stream);
        return;
    case Nvfp4GdnConvScheduleId::Materialized:
        break;
    }

    auto scope                         = workspace.scope();
    Nvfp4GdnProjectedWorkspace scratch = allocate_workspace(workspace, x.ne[1]);
    WorkspaceArena projection_workspace(scratch.projection);
    nvfp4_gdn_input_dispatch(x, weight, scratch.projected, z, LinearPolicy::AllowA4,
                             &projection_workspace, stream);
    nvfp4_gdn_snapshot_post_launch(scratch.projected, conv_weight, conv_states, valid_columns,
                                   initial_slot, snapshot_base_slot, query, key, value, stream);
}

} // namespace ninfer::ops::detail
