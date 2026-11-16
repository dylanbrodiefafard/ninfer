#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"

#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

enum class Nvfp4AttnInputRoute : std::uint8_t {
    Decode,
    A16SmallT,
    A16Mma,
    W4A4,
    W4A8,
};

// The [14336,5120] parent serves T=1 and the prefill widths below the W4A4 cutover on A16, AllowA4
// prefill from the cutover on W4A4, and AllowA8 verify from T=2 on W4A8. The [13312,2560] parent
// admits only A16Only and runs A16 at every T: the T=1 GEMV and the BF16-activation MMA from T=2.
Nvfp4AttnInputRoute resolve_route(std::int32_t input_rows, LinearPolicy policy,
                                  std::int32_t tokens) {
    if (tokens <= 0) { throw std::invalid_argument("nvfp4 attn_input_proj: T must be positive"); }
    if (input_rows == Nvfp4FlashNextAttnInputGeometry::kInputRows) {
        if (policy != LinearPolicy::A16Only) {
            throw std::invalid_argument("nvfp4 attn_input_proj: K=2560 admits only A16Only");
        }
        return tokens == 1 ? Nvfp4AttnInputRoute::Decode : Nvfp4AttnInputRoute::A16Mma;
    }
    if (input_rows != Nvfp4AttnInputGeometry::kInputRows) {
        throw std::invalid_argument("nvfp4 attn_input_proj: unsupported input rows");
    }
    if (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA4 &&
        policy != LinearPolicy::AllowA8) {
        throw std::invalid_argument("nvfp4 attn_input_proj: unsupported policy");
    }
    if (policy == LinearPolicy::AllowA8 && tokens >= kNvfp4FirstA8) {
        return Nvfp4AttnInputRoute::W4A8;
    }
    if (tokens == 1) { return Nvfp4AttnInputRoute::Decode; }
    if (tokens < kNvfp4FirstW4a4AttnInput) { return Nvfp4AttnInputRoute::A16SmallT; }
    if (policy != LinearPolicy::AllowA4) {
        throw std::invalid_argument(
            "nvfp4 attn_input_proj: A16 is registered only below the W4A4 cutover");
    }
    return Nvfp4AttnInputRoute::W4A4;
}

} // namespace

std::size_t nvfp4_attn_input_workspace_capacity_bytes(std::int32_t input_rows, LinearPolicy policy,
                                                      std::int32_t min_tokens,
                                                      std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("nvfp4 attn_input_proj workspace: invalid token interval");
    }
    (void)resolve_route(input_rows, policy, min_tokens);
    switch (resolve_route(input_rows, policy, max_tokens)) {
    case Nvfp4AttnInputRoute::W4A8:
        return fp8_a8_workspace_capacity_bytes(max_tokens, input_rows);
    case Nvfp4AttnInputRoute::W4A4:
        return nvfp4_w4a4_workspace_capacity_bytes(max_tokens, input_rows);
    case Nvfp4AttnInputRoute::Decode:
    case Nvfp4AttnInputRoute::A16SmallT:
    case Nvfp4AttnInputRoute::A16Mma:
        break;
    }
    return 0;
}

void nvfp4_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                               Tensor& k, Tensor& v, LinearPolicy policy, WorkspaceArena* workspace,
                               cudaStream_t stream) {
    const Nvfp4AttnInputRoute route = resolve_route(weight.k, policy, x.ne[1]);
    switch (route) {
    case Nvfp4AttnInputRoute::Decode:
        nvfp4_attn_input_decode_launch(x, weight, q, gate, k, v, stream);
        return;
    case Nvfp4AttnInputRoute::A16SmallT:
        nvfp4_attn_input_small_t_launch(x, weight, q, gate, k, v, stream);
        return;
    case Nvfp4AttnInputRoute::A16Mma:
        nvfp4_attn_input_a16_mma_launch(x, weight, q, gate, k, v, stream);
        return;
    case Nvfp4AttnInputRoute::W4A4:
    case Nvfp4AttnInputRoute::W4A8:
        break;
    }
    if (workspace == nullptr) {
        throw std::invalid_argument("nvfp4 W4A4 attn_input_proj requires caller workspace");
    }
    auto scope = workspace->scope();
    if (route == Nvfp4AttnInputRoute::W4A8) {
        const auto scratch = allocate_fp8_a8_workspace(*workspace, x.ne[1], weight.k);
        nvfp4_attn_input_w4a8_launch(x, weight, q, gate, k, v, scratch, stream);
        return;
    }
    const Nvfp4W4a4Workspace scratch = allocate_nvfp4_w4a4_workspace(*workspace, x.ne[1], weight.k);
    nvfp4_attn_input_w4a4_launch(x, weight, q, gate, k, v, scratch, stream);
}

} // namespace ninfer::ops::detail
