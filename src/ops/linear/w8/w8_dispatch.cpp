#include "ops/linear/w8/w8_dispatch.h"

#include <stdexcept>

namespace ninfer::ops::detail {

bool is_w8_vocabulary_problem(std::int32_t n, std::int32_t k) noexcept {
    return n == 248320 && k == 5120;
}

W8Launch select_w8_a16_launch(std::int32_t n, std::int32_t k, std::int32_t t) {
    if (t <= 0) { throw std::invalid_argument("w8 linear: unsupported shape or T"); }

    switch (k) {
    case 10240:
    case 6144:
    case 17408:
        if (n == 5120) {
            if (t <= 48) { return launch_w8_small_t; }
            return launch_w8_mma_r64_c128;
        }
        break;
    case 5120:
        switch (n) {
        case 256:
        case 1024:
        case 1280:
            if (t <= 4) { return launch_w8_simt_r8_c4; }
            if (t <= 16) { return launch_w8_simt_r8_c8; }
            return launch_w8_mma_r32_c128;
        case 6144:
            if (t <= 4) { return launch_w8_simt_r8_c4; }
            if (t <= 16) { return launch_w8_simt_r8_c8; }
            return launch_w8_mma_r64_c128;
        case 14336:
            if (t <= 48) { return launch_w8_small_t; }
            return launch_w8_mma_r64_c128;
        case 34816:
            if (t <= 40) { return launch_w8_small_t; }
            if (t <= 48) { return launch_w8_mma_r64x16_c48_k128_a1; }
            return launch_w8_mma_r64_c128;
        case 248320:
            if (t <= 48) { return launch_w8_small_t; }
            if (t <= 64) { return launch_w8_mma_r32_c64; }
            return launch_w8_mma_r64_c128;
        default:
            break;
        }
        break;
    case 25600:
        if (n == 5120) {
            if (t <= 4) { return launch_w8_simt_r8_c4; }
            if (t <= 16) { return launch_w8_simt_r8_c8; }
            return launch_w8_mma_r64_c128;
        }
        break;
    case 4096:
        if (n == 5120) {
            if (t <= 4) { return launch_w8_simt_r8_c4; }
            if (t <= 16) { return launch_w8_simt_r8_c8; }
            return launch_w8_mma_r64_c128;
        }
        break;
    case 4608:
        if (t > 32768) { break; }
        switch (n) {
        case 4608:
            if (t <= 8 || t == 12) { return launch_w8_simt_r8_c4; }
            if (t <= 256) { return launch_w8_mma_r32_c128; }
            return launch_w8_mma_r64_c128;
        case 5120:
            if (t <= 4) { return launch_w8_simt_r8_c4; }
            if (t == 5) { return launch_w8_simt_r8_c8; }
            return launch_w8_mma_r64_c128;
        default:
            break;
        }
        break;
    default:
        break;
    }

    throw std::invalid_argument("w8 linear: unsupported shape or T");
}

W8Launch select_w8_launch(std::int32_t n, std::int32_t k, std::int32_t t, LinearPolicy policy) {
    switch (policy) {
    case LinearPolicy::A16Only:
    case LinearPolicy::AllowA8:
        return select_w8_a16_launch(n, k, t);
    case LinearPolicy::AllowA4:
        break;
    }
    throw std::invalid_argument("w8 linear: unsupported policy");
}

void w8_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                 cudaStream_t stream) {
    const W8Launch launch = select_w8_launch(w.n, w.k, x.ne[1], policy);
    launch(x, w, out, stream);
}

} // namespace ninfer::ops::detail
