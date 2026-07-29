#include "ops/linear/nvfp4/nvfp4_dispatch.h"

#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_format.h"
#include "ops/linear/nvfp4/nvfp4_launch.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    constexpr std::int32_t kChunk = kNvfp4LastSmallT;
    for (std::int32_t token_begin = 0; token_begin < x.ne[1]; token_begin += kChunk) {
        const std::int32_t active = std::min(kChunk, x.ne[1] - token_begin);
        auto* input               = static_cast<std::uint8_t*>(x.data) +
                      static_cast<std::int64_t>(token_begin) * weight.k * sizeof(std::uint16_t);
        auto* output = static_cast<std::uint8_t*>(out.data) +
                       static_cast<std::int64_t>(token_begin) * weight.n * sizeof(std::uint16_t);
        Tensor input_chunk(input, DType::BF16, {weight.k, active});
        Tensor output_chunk(output, DType::BF16, {weight.n, active});
        if (active == 1) {
            launch_nvfp4_decode(input_chunk, weight, output_chunk, stream);
        } else {
            launch_nvfp4_small_t(input_chunk, weight, output_chunk, stream);
        }
    }
}

} // namespace

void nvfp4_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                    WorkspaceArena* workspace, cudaStream_t stream) {
    validate_nvfp4_weight(weight, "nvfp4 linear");
    if (!is_nvfp4_linear_problem(weight.n, weight.k) || x.ne[1] <= 0) {
        throw std::invalid_argument("nvfp4 linear: unsupported shape");
    }

    if (policy == LinearPolicy::A16Only ||
        (policy == LinearPolicy::AllowA4 && x.ne[1] < kNvfp4FirstA4T)) {
        launch_a16(x, weight, out, stream);
        return;
    }
    if (policy != LinearPolicy::AllowA4) {
        throw std::invalid_argument("nvfp4 linear: unsupported policy");
    }
    if (workspace == nullptr) {
        throw std::invalid_argument("nvfp4 W4A4 linear requires caller workspace");
    }
    auto scope                       = workspace->scope();
    const Nvfp4W4a4Workspace scratch = allocate_nvfp4_w4a4_workspace(*workspace, x.ne[1], weight.k);
    launch_nvfp4_w4a4(x, weight, out, scratch, stream);
}

} // namespace ninfer::ops::detail
