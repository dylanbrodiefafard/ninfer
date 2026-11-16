#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"

#include "ops/linear/nvfp4/nvfp4_config.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

enum class Nvfp4GdnInputRoute : std::uint8_t {
    Decode,
    A16SmallT,
    A16Mma,
    W4A4,
};

// The [16384,5120] parent runs A16 below the W4A4 cutover under either policy and W4A4 from the
// cutover under AllowA4. The [16384,2560] parent admits only A16Only and runs A16 at every T: the
// T=1 GEMV and the BF16-activation MMA from T=2.
Nvfp4GdnInputRoute resolve_route(std::int32_t input_rows, LinearPolicy policy,
                                 std::int32_t tokens) {
    if (tokens <= 0) { throw std::invalid_argument("nvfp4 gdn_input_proj: T must be positive"); }
    if (input_rows == Nvfp4FlashNextGdnInputGeometry::kInputRows) {
        if (policy != LinearPolicy::A16Only) {
            throw std::invalid_argument("nvfp4 gdn_input_proj: K=2560 admits only A16Only");
        }
        return tokens == 1 ? Nvfp4GdnInputRoute::Decode : Nvfp4GdnInputRoute::A16Mma;
    }
    if (input_rows != Nvfp4GdnInputGeometry::kInputRows) {
        throw std::invalid_argument("nvfp4 gdn_input_proj: unsupported input rows");
    }
    if (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA4) {
        throw std::invalid_argument("nvfp4 gdn_input_proj: unsupported policy");
    }
    if (tokens == 1) { return Nvfp4GdnInputRoute::Decode; }
    if (tokens < kNvfp4FirstW4a4GdnInput) { return Nvfp4GdnInputRoute::A16SmallT; }
    if (policy != LinearPolicy::AllowA4) {
        throw std::invalid_argument(
            "nvfp4 gdn_input_proj: A16 is registered only below the W4A4 cutover");
    }
    return Nvfp4GdnInputRoute::W4A4;
}

} // namespace

std::size_t nvfp4_gdn_input_workspace_capacity_bytes(std::int32_t input_rows, LinearPolicy policy,
                                                     std::int32_t min_tokens,
                                                     std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("nvfp4 gdn_input_proj workspace: invalid token interval");
    }
    (void)resolve_route(input_rows, policy, min_tokens);
    return resolve_route(input_rows, policy, max_tokens) == Nvfp4GdnInputRoute::W4A4
               ? nvfp4_w4a4_workspace_capacity_bytes(max_tokens, input_rows)
               : 0;
}

void nvfp4_gdn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                              LinearPolicy policy, WorkspaceArena* workspace, cudaStream_t stream) {
    switch (resolve_route(weight.k, policy, x.ne[1])) {
    case Nvfp4GdnInputRoute::Decode:
        nvfp4_gdn_input_decode_launch(x, weight, qkv, z, stream);
        return;
    case Nvfp4GdnInputRoute::A16SmallT:
        nvfp4_gdn_input_small_t_launch(x, weight, qkv, z, stream);
        return;
    case Nvfp4GdnInputRoute::A16Mma:
        nvfp4_gdn_input_a16_mma_launch(x, weight, qkv, z, stream);
        return;
    case Nvfp4GdnInputRoute::W4A4:
        break;
    }
    if (workspace == nullptr) {
        throw std::invalid_argument("nvfp4 W4A4 gdn_input_proj requires caller workspace");
    }
    auto scope                       = workspace->scope();
    const Nvfp4W4a4Workspace scratch = allocate_nvfp4_w4a4_workspace(*workspace, x.ne[1], weight.k);
    nvfp4_gdn_input_w4a4_launch(x, weight, qkv, z, scratch, stream);
}

} // namespace ninfer::ops::detail
