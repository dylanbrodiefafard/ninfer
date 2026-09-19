#include "ops/linear/bf16/bf16_dispatch.h"

#include "ops/linear/bf16/bf16_config.h"
#include "ops/linear/bf16/bf16_launch.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {

Bf16Launch select_bf16_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t) {
    const bool vision_problem = (n == 1152 && (k == 1536 || k == 1152 || k == 4304)) ||
        (k == 1152 && (n == 3456 || n == 4304));
    const bool supported_problem = vision_problem || (n == 14336 && k == 5120) || (n == 5120 && k == 6144) ||
        (k == 2560 && (n == 10240 || n == 6144 || n == 12288 || n == 512 ||
                       n == 640 || n == 2560 || n == 7680 || n == 248320)) ||
        (n == 2560 && (k == 6144 || k == 640 || k == 4608 || k == 7680 || k == 12800)) ||
        (n == 4608 && k == 4608) ||
        (n == 320 && k == 10240) || (n == 10240 && k == 320);
    if (!supported_problem || t <= 0) {
        throw std::invalid_argument("bf16 linear: unsupported shape or T");
    }
    // These exact preview widths do not fit the existing vector GEMV K phase.
    // The existing 64-wide BF16 MMA tile represents them without padding/repacking.
    if (vision_problem || k == 320 || k == 640) { return launch_bf16_mma; }
    if (t == 1) { return launch_bf16_decode; }
    const std::int32_t small_t_end =
        n == 5120 ? kBf16SmallTMaxTokens : kBf16LinearSmallTDispatchEnd;
    if (t <= small_t_end) { return launch_bf16_small_t; }
    return launch_bf16_mma;
}

Bf16Launch select_bf16_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
        return select_bf16_a16_launch(n, k, t);
    case LinearPolicy::AllowA8:
    case LinearPolicy::AllowA4:
        break;
    }
    throw std::invalid_argument("bf16 linear: unsupported policy");
}

void bf16_dispatch(const Tensor& x, const Weight& weight, Tensor& out, LinearPolicy policy,
                   cudaStream_t stream) {
    const Bf16Launch launch = select_bf16_launch(weight.n, weight.k, x.ne[1], policy);
    launch(x, weight, out, stream);
}

} // namespace ninfer::ops::detail
