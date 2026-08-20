// ninfer::ops::detail - residual_rmsnorm launcher: one launch replaces the residual_add +
// rmsnorm pair (the per-layer unit_offset RMSNorm). The D=5120 (per-layer hidden) path uses the
// CTA kernel (Block=512, MaxPairsPerThread=8), mirroring the standalone rmsnorm's D=5120 CTA
// geometry + reduce, so the result is bit-exact with the unfused pair.
#include "ops/launcher/residual_rmsnorm.h"

#include "ops/kernel/residual_rmsnorm.cuh"
#include "core/device.h"

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {

void residual_rmsnorm_launch(const Tensor& y, Tensor& x, const Tensor& weight, float eps,
                             Tensor& out, cudaStream_t stream) {
    const std::int32_t d = x.ne[0];
    if (d <= 0) { throw std::invalid_argument("residual_rmsnorm: ne[0] must be positive"); }
    const std::int64_t rows = out.numel() / static_cast<std::int64_t>(d);
    if (rows > std::numeric_limits<int>::max()) {
        throw std::overflow_error("residual_rmsnorm: row count exceeds CUDA grid limit");
    }

    const auto* y_bf16 = static_cast<const __nv_bfloat16*>(y.data);
    auto* x_bf16       = static_cast<__nv_bfloat16*>(x.data);
    const auto* w_bf16 = static_cast<const __nv_bfloat16*>(weight.data);
    auto* out_bf16     = static_cast<__nv_bfloat16*>(out.data);

    const auto y_addr = reinterpret_cast<std::uintptr_t>(y.data);
    const auto x_addr = reinterpret_cast<std::uintptr_t>(x.data);
    const auto w_addr = reinterpret_cast<std::uintptr_t>(weight.data);
    const auto o_addr = reinterpret_cast<std::uintptr_t>(out.data);
    const bool aligned2 =
        ((y_addr | x_addr | w_addr | o_addr) & (alignof(__nv_bfloat162) - 1)) == 0;

    // The per-layer hidden (D=5120) is the target: the CTA kernel, mirroring the standalone
    // rmsnorm's D=5120 CTA geometry (Block=512, MaxPairsPerThread=8) for bit-exactness.
    if (aligned2 && d > 3072 && d <= 8192 && d % 1024 == 0) {
        residual_rmsnorm_cta_bf16x2_kernel<512, 8>
            <<<static_cast<unsigned int>(rows), 512, 0, stream>>>(
                reinterpret_cast<const __nv_bfloat162*>(y_bf16),
                reinterpret_cast<__nv_bfloat162*>(x_bf16),
                reinterpret_cast<const __nv_bfloat162*>(w_bf16),
                reinterpret_cast<__nv_bfloat162*>(out_bf16), d, rows, eps);
    } else {
        // Non-D=5120 (e.g. the MTP 1792) is out of scope for the CTA kernel; the per-layer
        // (19,867-launch) path is D=5120. Guard loudly rather than silently mis-dispatch.
        throw std::invalid_argument(
            "residual_rmsnorm: supported D is the aligned BF16x2 per-layer hidden (5120)");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail