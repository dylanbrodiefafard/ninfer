#include "ops/linear/bf16/bf16_launch.h"

#include "core/device.h"
#include "ops/linear/bf16/bf16_gemv.cuh"

#include <cuda_bf16.h>

#include <stdexcept>

namespace ninfer::ops::detail {

void launch_bf16_decode(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Geometry = Bf16LinearDecodeGeometry;
    using Schedule = Bf16LinearDecodeSchedule;
    if (x.ne[0] != Geometry::kInputRows || x.ne[1] != 1 || out.ne[0] != Geometry::kOutputRows ||
        out.ne[1] != 1 || weight.n != Geometry::kOutputRows || weight.k != Geometry::kInputRows) {
        throw std::invalid_argument("bf16 linear decode requires [14336,5120] and T=1");
    }

    const Bf16ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data)};
    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    bf16_gemv_kernel<Geometry, Schedule><<<kBlocks, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const __nv_bfloat16*>(weight.qdata),
        output);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
