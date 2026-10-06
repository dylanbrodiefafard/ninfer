#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"

#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"
#include "ops/linear/fp8/fp8_a8_plan.h"

#include <cstddef>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

enum class Nvfp4LinearSwiGluRoute {
    DecodeFusedA16,
    SmallTFusedA16,
    FusedW4A4,
    FusedW4A8,
};

Nvfp4LinearSwiGluRoute resolve_route(LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0) { throw std::invalid_argument("nvfp4 linear_swiglu: T must be positive"); }
    if (policy == LinearPolicy::AllowA8) {
        if (tokens >= kNvfp4FirstA8) { return Nvfp4LinearSwiGluRoute::FusedW4A8; }
        return tokens == 1 ? Nvfp4LinearSwiGluRoute::DecodeFusedA16
                           : Nvfp4LinearSwiGluRoute::SmallTFusedA16;
    }
    if (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA4) {
        throw std::invalid_argument("nvfp4 linear_swiglu admits only A16 or A4");
    }
    if (policy == LinearPolicy::A16Only) {
        if (tokens == 1) { return Nvfp4LinearSwiGluRoute::DecodeFusedA16; }
        if (tokens <= 20) { return Nvfp4LinearSwiGluRoute::SmallTFusedA16; }
        throw std::invalid_argument("nvfp4 linear_swiglu A16 is registered only through T=20");
    }
    return tokens == 1 ? Nvfp4LinearSwiGluRoute::DecodeFusedA16 : Nvfp4LinearSwiGluRoute::FusedW4A4;
}

std::size_t fused_workspace_bytes(std::int32_t tokens) {
    return nvfp4_w4a4_workspace_capacity_bytes(tokens, Nvfp4MlpGateUpGeometry::kInputRows);
}

} // namespace

std::size_t nvfp4_linear_swiglu_workspace_capacity_bytes(LinearPolicy policy,
                                                         std::int32_t min_tokens,
                                                         std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("nvfp4 linear_swiglu workspace: invalid token interval");
    }
    (void)resolve_route(policy, min_tokens);
    (void)resolve_route(policy, max_tokens);
    if (policy == LinearPolicy::AllowA8) {
        return max_tokens >= kNvfp4FirstA8 ? fp8_a8_workspace_capacity_bytes(max_tokens, 5120) : 0;
    }
    if (policy == LinearPolicy::A16Only || max_tokens < kNvfp4FirstW4a4MlpGateUp) { return 0; }

    // The W4A4 workspace grows with T on both the MMA and the TMA route.
    return fused_workspace_bytes(max_tokens);
}

void nvfp4_linear_swiglu_dispatch(const Tensor& x, const Weight& weight, Tensor& out,
                                  LinearPolicy policy, WorkspaceArena& workspace,
                                  cudaStream_t stream) {
    switch (resolve_route(policy, x.ne[1])) {
    case Nvfp4LinearSwiGluRoute::FusedW4A8:
        nvfp4_linear_swiglu_w4a8_launch(x, weight, out, workspace, stream);
        return;
    case Nvfp4LinearSwiGluRoute::DecodeFusedA16:
        nvfp4_linear_swiglu_decode_launch(x, weight, out, stream);
        return;
    case Nvfp4LinearSwiGluRoute::SmallTFusedA16:
        nvfp4_linear_swiglu_small_t_launch(x, weight, out, stream);
        return;
    case Nvfp4LinearSwiGluRoute::FusedW4A4:
        nvfp4_linear_swiglu_w4a4_launch(x, weight, out, workspace, stream);
        return;
    }
}

} // namespace ninfer::ops::detail
