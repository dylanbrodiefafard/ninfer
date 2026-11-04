#include "ninfer/ops/swa.h"
#include "ninfer/types.h"

#include "ops/r9700/dflash/swa_launch.h"

#include "core/layout.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kHeadDim             = 128;
constexpr std::int32_t kQHeads              = 32;
constexpr std::int32_t kKvHeads             = 8;
constexpr float kExpectedScale              = 0.08838834764831844055F;
constexpr std::uint32_t kDirectContextLimit = 96;
constexpr std::int32_t kMaximumSplits       = 32;

void require_shape(const Tensor& tensor, std::int32_t n0, std::int32_t n1, std::int32_t n2,
                   std::int32_t n3, const char* name) {
    if (tensor.ne[0] != n0 || tensor.ne[1] != n1 || tensor.ne[2] != n2 || tensor.ne[3] != n3) {
        throw std::invalid_argument(std::string("swa: invalid shape for ") + name);
    }
}

void require_tensor(const Tensor& tensor, DType dtype, const char* name) {
    if (tensor.dtype != dtype || !tensor.is_contiguous() || tensor.data == nullptr) {
        throw std::invalid_argument(std::string("swa: invalid ") + name);
    }
}

void validate_profile(SwaContextExecutionEnvelope envelope, std::int32_t min_tokens,
                      std::int32_t max_tokens, std::int32_t batch_size) {
    if (min_tokens < 1 || max_tokens < min_tokens || max_tokens > 16 || batch_size < 1 ||
        batch_size > static_cast<std::int32_t>(kDFlashMaximumConcurrency) ||
        envelope.min_context > envelope.max_context ||
        envelope.max_context >
            static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("swa: invalid envelope, token interval, or batch");
    }
}

void validate_context(const CyclicKVCacheLayerView& context) {
    if (context.num_kv_heads != kKvHeads || context.head_dim != kHeadDim ||
        (context.capacity != 2048 && context.capacity != 4096) ||
        context.padded_capacity < context.capacity || context.lane_capacity < 1 ||
        context.padded_capacity >
            static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("swa: invalid cyclic context geometry");
    }
    const auto padded = static_cast<std::int32_t>(context.padded_capacity);
    require_tensor(context.k, DType::BF16, "context K");
    require_tensor(context.v, DType::BF16, "context V");
    require_shape(context.k, kHeadDim, padded, kKvHeads, context.lane_capacity, "context K");
    require_shape(context.v, kHeadDim, padded, kKvHeads, context.lane_capacity, "context V");
}

detail::SwaLaunchPlan resolve_plan(SwaContextExecutionEnvelope envelope, std::int32_t tokens,
                                   std::int32_t batch) {
    // Long windows always split, at every compact-batch width: at 56-64 columns (C8 K6/K7) the
    // direct kernel took 3.2-3.3 ms against 0.36 ms split (4096-token windows, swa_qual).
    const std::int32_t columns = tokens * batch;
    if (envelope.max_context <= kDirectContextLimit) { return {.direct = true, .splits = 1}; }
    std::int32_t splits       = 1;
    const std::int32_t target = (96 + columns - 1) / columns;
    while (splits < target && splits < kMaximumSplits) splits *= 2;
    return {.direct = false, .splits = splits};
}

struct PartialWorkspace {
    Tensor acc;
    Tensor m;
    Tensor l;
};

template <class Allocator>
PartialWorkspace allocate_workspace(Allocator& workspace, std::int32_t tokens, std::int32_t batch,
                                    detail::SwaLaunchPlan plan) {
    if (plan.direct) return {};
    return {
        workspace.alloc(DType::BF16, {kHeadDim, kQHeads, tokens, plan.splits * batch}),
        workspace.alloc(DType::FP32, {kQHeads, tokens, plan.splits * batch}),
        workspace.alloc(DType::FP32, {kQHeads, tokens, plan.splits * batch}),
    };
}

} // namespace

std::size_t swa_workspace_capacity_bytes(SwaContextExecutionEnvelope envelope,
                                         std::int32_t min_tokens, std::int32_t max_tokens,
                                         std::int32_t batch_size) {
    validate_profile(envelope, min_tokens, max_tokens, batch_size);
    std::size_t capacity = 0;
    for (std::int32_t tokens = min_tokens; tokens <= max_tokens; ++tokens) {
        WorkspaceLayoutBuilder layout;
        (void)allocate_workspace(layout, tokens, batch_size,
                                 resolve_plan(envelope, tokens, batch_size));
        capacity = std::max(capacity, layout.peak_bytes(1));
    }
    return capacity;
}

void swa(const Tensor& q, const Tensor& query_k, const Tensor& query_v, const Tensor& positions,
         const Tensor& valid_columns, const Tensor& lanes, float scale,
         const CyclicKVCacheLayerView& context, SwaContextExecutionEnvelope envelope,
         WorkspaceArena& workspace, Tensor& out, hipStream_t stream) {
    const std::int32_t tokens = q.ne[2];
    const std::int32_t batch  = q.ne[3];
    validate_profile(envelope, tokens, tokens, batch);
    require_tensor(q, DType::BF16, "q");
    require_tensor(query_k, DType::BF16, "query K");
    require_tensor(query_v, DType::BF16, "query V");
    require_tensor(positions, DType::I32, "positions");
    require_tensor(valid_columns, DType::I32, "valid columns");
    require_tensor(lanes, DType::I32, "lanes");
    require_tensor(out, DType::BF16, "output");
    require_shape(q, kHeadDim, kQHeads, tokens, batch, "q");
    require_shape(query_k, kHeadDim, kKvHeads, tokens, batch, "query K");
    require_shape(query_v, kHeadDim, kKvHeads, tokens, batch, "query V");
    require_shape(positions, tokens, batch, 1, 1, "positions");
    require_shape(valid_columns, batch, 1, 1, 1, "valid columns");
    require_shape(lanes, batch, 1, 1, 1, "lanes");
    require_shape(out, kHeadDim, kQHeads, tokens, batch, "output");
    validate_context(context);
    if (!std::isfinite(scale) || std::abs(scale - kExpectedScale) > 1.0e-7F) {
        throw std::invalid_argument("swa: scale must be 1/sqrt(128)");
    }
    if (out.data == q.data || out.data == query_k.data || out.data == query_v.data ||
        out.data == positions.data || out.data == valid_columns.data || out.data == lanes.data ||
        out.data == context.k.data || out.data == context.v.data) {
        throw std::invalid_argument("swa: output aliases an input");
    }

    const detail::SwaLaunchPlan plan = resolve_plan(envelope, tokens, batch);
    auto scope                       = workspace.scope();
    const PartialWorkspace partial   = allocate_workspace(workspace, tokens, batch, plan);
    detail::swa_launch(q, query_k, query_v, positions, valid_columns, lanes, scale, context,
                       envelope, plan, partial.acc, partial.m, partial.l, out, stream);
}

} // namespace ninfer::ops
