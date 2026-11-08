#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"

#include "ops/linear/nvfp4/nvfp4_config.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

enum class Nvfp4GdnInputRoute : std::uint8_t {
    A16,
    W4A4,
};

// The prefill GDN input projection runs A16 below the W4A4 cutover under either policy and W4A4
// from the cutover under AllowA4.
Nvfp4GdnInputRoute resolve_route(LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0) { throw std::invalid_argument("nvfp4 gdn_input_proj: T must be positive"); }
    if (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA4) {
        throw std::invalid_argument("nvfp4 gdn_input_proj: unsupported policy");
    }
    if (tokens < kNvfp4FirstW4a4GdnInput) { return Nvfp4GdnInputRoute::A16; }
    if (policy != LinearPolicy::AllowA4) {
        throw std::invalid_argument(
            "nvfp4 gdn_input_proj: A16 is registered only below the W4A4 cutover");
    }
    return Nvfp4GdnInputRoute::W4A4;
}

void launch_a16(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                cudaStream_t stream) {
    if (x.ne[1] == 1) {
        nvfp4_gdn_input_decode_launch(x, weight, qkv, z, stream);
        return;
    }
    nvfp4_gdn_input_small_t_launch(x, weight, qkv, z, stream);
}

} // namespace

std::size_t nvfp4_gdn_input_workspace_capacity_bytes(LinearPolicy policy, std::int32_t min_tokens,
                                                     std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("nvfp4 gdn_input_proj workspace: invalid token interval");
    }
    (void)resolve_route(policy, min_tokens);
    return resolve_route(policy, max_tokens) == Nvfp4GdnInputRoute::W4A4
               ? nvfp4_w4a4_workspace_capacity_bytes(max_tokens, Nvfp4GdnInputGeometry::kInputRows)
               : 0;
}

void nvfp4_gdn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                              LinearPolicy policy, WorkspaceArena* workspace, cudaStream_t stream) {
    if (resolve_route(policy, x.ne[1]) == Nvfp4GdnInputRoute::A16) {
        launch_a16(x, weight, qkv, z, stream);
        return;
    }
    if (workspace == nullptr) {
        throw std::invalid_argument("nvfp4 W4A4 gdn_input_proj requires caller workspace");
    }
    auto scope                       = workspace->scope();
    const Nvfp4W4a4Workspace scratch = allocate_nvfp4_w4a4_workspace(*workspace, x.ne[1], weight.k);
    nvfp4_gdn_input_w4a4_launch(x, weight, qkv, z, scratch, stream);
}

} // namespace ninfer::ops::detail
