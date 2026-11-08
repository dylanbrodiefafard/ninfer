#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"

#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

enum class Nvfp4AttnInputRoute : std::uint8_t {
    A16,
    W4A4,
    W4A8,
};

// The A16 route serves T=1 and the prefill widths below the W4A4 cutover; verify takes W4A8 from
// T=2 under AllowA8.
Nvfp4AttnInputRoute resolve_route(LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0) { throw std::invalid_argument("nvfp4 attn_input_proj: T must be positive"); }
    if (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA4 &&
        policy != LinearPolicy::AllowA8) {
        throw std::invalid_argument("nvfp4 attn_input_proj: unsupported policy");
    }
    if (policy == LinearPolicy::AllowA8 && tokens >= kNvfp4FirstA8) {
        return Nvfp4AttnInputRoute::W4A8;
    }
    if (tokens < kNvfp4FirstW4a4AttnInput) { return Nvfp4AttnInputRoute::A16; }
    if (policy != LinearPolicy::AllowA4) {
        throw std::invalid_argument(
            "nvfp4 attn_input_proj: A16 is registered only below the W4A4 cutover");
    }
    return Nvfp4AttnInputRoute::W4A4;
}

void launch_a16(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate, Tensor& k,
                Tensor& v, cudaStream_t stream) {
    if (x.ne[1] == 1) {
        nvfp4_attn_input_decode_launch(x, weight, q, gate, k, v, stream);
        return;
    }
    nvfp4_attn_input_small_t_launch(x, weight, q, gate, k, v, stream);
}

} // namespace

std::size_t nvfp4_attn_input_workspace_capacity_bytes(LinearPolicy policy, std::int32_t min_tokens,
                                                      std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("nvfp4 attn_input_proj workspace: invalid token interval");
    }
    (void)resolve_route(policy, min_tokens);
    if (resolve_route(policy, max_tokens) == Nvfp4AttnInputRoute::W4A8) {
        return fp8_a8_workspace_capacity_bytes(max_tokens, 5120);
    }
    return resolve_route(policy, max_tokens) == Nvfp4AttnInputRoute::W4A4
               ? nvfp4_w4a4_workspace_capacity_bytes(max_tokens, Nvfp4AttnInputGeometry::kInputRows)
               : 0;
}

void nvfp4_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                               Tensor& k, Tensor& v, LinearPolicy policy, WorkspaceArena* workspace,
                               cudaStream_t stream) {
    if (resolve_route(policy, x.ne[1]) == Nvfp4AttnInputRoute::A16) {
        launch_a16(x, weight, q, gate, k, v, stream);
        return;
    }
    if (workspace == nullptr) {
        throw std::invalid_argument("nvfp4 W4A4 attn_input_proj requires caller workspace");
    }
    auto scope = workspace->scope();
    if (resolve_route(policy, x.ne[1]) == Nvfp4AttnInputRoute::W4A8) {
        const auto scratch = allocate_fp8_a8_workspace(*workspace, x.ne[1], weight.k);
        nvfp4_attn_input_w4a8_launch(x, weight, q, gate, k, v, scratch, stream);
        return;
    }
    const Nvfp4W4a4Workspace scratch = allocate_nvfp4_w4a4_workspace(*workspace, x.ne[1], weight.k);
    nvfp4_attn_input_w4a4_launch(x, weight, q, gate, k, v, scratch, stream);
}

} // namespace ninfer::ops::detail
