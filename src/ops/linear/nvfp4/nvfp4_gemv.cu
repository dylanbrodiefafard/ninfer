#include "ops/linear/nvfp4/nvfp4_launch.h"

#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_gemv.cuh"

#include <cuda_bf16.h>

#include <stdexcept>

namespace ninfer::ops::detail {

void launch_nvfp4_decode(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Geometry = Nvfp4LinearDecodeGeometry;
    using Schedule = Nvfp4LinearDecodeSchedule;
    if (x.ne[0] != Geometry::kInputRows || x.ne[1] != 1 || out.ne[0] != Geometry::kOutputRows ||
        out.ne[1] != 1 || weight.n != Geometry::kOutputRows || weight.k != Geometry::kInputRows) {
        throw std::invalid_argument("nvfp4 linear decode requires [14336,5120] and T=1");
    }

    const Nvfp4ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data)};
    constexpr int kBlocks              = Geometry::kOutputRows / Schedule::kRowsPerCta;
    const float inverse_weight_divisor = 1.0F / weight.weight_scale_divisor;
    nvfp4_gemv_kernel<Geometry, Schedule><<<kBlocks, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), inverse_weight_divisor, output);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
