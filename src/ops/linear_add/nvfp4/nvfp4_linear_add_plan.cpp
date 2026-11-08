#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"

#include "ops/linear/nvfp4/nvfp4_config.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

enum class Nvfp4LinearAddRoute : std::uint8_t {
    A16,
    W4A4,
    W4A8,
};

// The A16 route serves T=1 and the prefill widths below the W4A4 cutover; verify takes W4A8 from
// T=2 under AllowA8.
Nvfp4LinearAddRoute resolve_route(std::int32_t output_rows, std::int32_t input_rows,
                                  LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0 || output_rows != 5120 || (input_rows != 6144 && input_rows != 17408)) {
        throw std::invalid_argument("nvfp4 linear_add: unsupported shape");
    }
    if (policy != LinearPolicy::A16Only && policy != LinearPolicy::AllowA4 &&
        policy != LinearPolicy::AllowA8) {
        throw std::invalid_argument("nvfp4 linear_add: unsupported policy");
    }
    if (policy == LinearPolicy::AllowA8 && tokens >= kNvfp4FirstA8) {
        return Nvfp4LinearAddRoute::W4A8;
    }
    const std::int32_t first_w4a4 =
        input_rows == 6144 ? kNvfp4FirstW4a4Residual6144 : kNvfp4FirstW4a4Residual17408;
    if (tokens < first_w4a4) { return Nvfp4LinearAddRoute::A16; }
    if (policy != LinearPolicy::AllowA4) {
        throw std::invalid_argument("nvfp4 linear_add: A16 is registered only below the W4A4 "
                                    "cutover");
    }
    return Nvfp4LinearAddRoute::W4A4;
}

void launch_a16(const Tensor& x, const Weight& weight, Tensor& residual, cudaStream_t stream) {
    if (x.ne[1] == 1) {
        nvfp4_linear_add_decode_launch(x, weight, residual, stream);
        return;
    }
    nvfp4_linear_add_small_t_launch(x, weight, residual, stream);
}

} // namespace

std::size_t nvfp4_linear_add_workspace_capacity_bytes(std::int32_t output_rows,
                                                      std::int32_t input_rows, LinearPolicy policy,
                                                      std::int32_t min_tokens,
                                                      std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("nvfp4 linear_add workspace: invalid token interval");
    }
    (void)resolve_route(output_rows, input_rows, policy, min_tokens);
    if (resolve_route(output_rows, input_rows, policy, max_tokens) == Nvfp4LinearAddRoute::W4A8) {
        return fp8_a8_workspace_capacity_bytes(max_tokens, input_rows);
    }
    return resolve_route(output_rows, input_rows, policy, max_tokens) == Nvfp4LinearAddRoute::W4A4
               ? nvfp4_w4a4_workspace_capacity_bytes(max_tokens, input_rows)
               : 0;
}

void nvfp4_linear_add_dispatch(const Tensor& x, const Weight& weight, Tensor& residual,
                               LinearPolicy policy, WorkspaceArena& workspace,
                               cudaStream_t stream) {
    if (resolve_route(weight.n, weight.k, policy, x.ne[1]) == Nvfp4LinearAddRoute::A16) {
        launch_a16(x, weight, residual, stream);
        return;
    }
    auto scope = workspace.scope();
    if (resolve_route(weight.n, weight.k, policy, x.ne[1]) == Nvfp4LinearAddRoute::W4A8) {
        const auto scratch = allocate_fp8_a8_workspace(workspace, x.ne[1], weight.k);
        nvfp4_linear_add_w4a8_launch(x, weight, residual, scratch, stream);
        return;
    }
    const Nvfp4W4a4Workspace scratch = allocate_nvfp4_w4a4_workspace(workspace, x.ne[1], weight.k);
    nvfp4_linear_add_w4a4_launch(x, weight, residual, scratch, stream);
}

} // namespace ninfer::ops::detail
