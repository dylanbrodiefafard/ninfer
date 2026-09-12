#include "ops/linear/nvfp4/nvfp4_launch.h"

#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_gemv.cuh"

#include <cuda_bf16.h>

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

template <class Geometry>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Schedule = typename Nvfp4LinearDecodeProductionSchedule<Geometry>::Type;
    if (x.ne[0] != Geometry::kInputRows || x.ne[1] < 1 || out.ne[0] != Geometry::kOutputRows ||
        out.ne[1] != x.ne[1] || weight.n != Geometry::kOutputRows ||
        weight.k != Geometry::kInputRows) {
        throw std::invalid_argument("nvfp4 exact-geometry decode: invalid problem");
    }

    const Nvfp4ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data),
                                       Geometry::kOutputRows};
    constexpr int kBlocks              = Geometry::kOutputRows / Schedule::kRowsPerCta;
    const float inverse_weight_divisor = 1.0F / weight.weight_scale_divisor;
    nvfp4_gemv_kernel<Geometry, Schedule>
        <<<dim3(kBlocks, x.ne[1]), Schedule::kThreads, 0, stream>>>(
            Nvfp4PackedActivation<Geometry>{static_cast<const __nv_bfloat16*>(x.data)},
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), inverse_weight_divisor,
            Nvfp4IdentityEpilogue{}, output);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_nvfp4_exact_geometry_decode(const Tensor& x, const Weight& weight, Tensor& out,
                                        cudaStream_t stream) {
    switch (resolve_nvfp4_problem(weight.n, weight.k)) {
    case Nvfp4Problem::N10240K2560:
        return launch_exact<Nvfp4N10240K2560Geometry>(x, weight, out, stream);
    case Nvfp4Problem::N6144K2560:
        return launch_exact<Nvfp4N6144K2560Geometry>(x, weight, out, stream);
    case Nvfp4Problem::N12288K2560:
        return launch_exact<Nvfp4N12288K2560Geometry>(x, weight, out, stream);
    case Nvfp4Problem::N512K2560:
        return launch_exact<Nvfp4N512K2560Geometry>(x, weight, out, stream);
    case Nvfp4Problem::N2560K6144:
        return launch_exact<Nvfp4N2560K6144Geometry>(x, weight, out, stream);
    case Nvfp4Problem::N640K2560:
        return launch_exact<Nvfp4N640K2560Geometry>(x, weight, out, stream);
    case Nvfp4Problem::N1280K2560:
        return launch_exact<Nvfp4N1280K2560Geometry>(x, weight, out, stream);
    case Nvfp4Problem::N2560K640:
        return launch_exact<Nvfp4N2560K640Geometry>(x, weight, out, stream);
    case Nvfp4Problem::N10240K320:
        return launch_exact<Nvfp4N10240K320Geometry>(x, weight, out, stream);
    case Nvfp4Problem::N2560K2560:
        return launch_exact<Nvfp4N2560K2560Geometry>(x, weight, out, stream);
    case Nvfp4Problem::N248320K2560:
        return launch_exact<Nvfp4N248320K2560Geometry>(x, weight, out, stream);
    default:
        throw std::invalid_argument("nvfp4 exact-geometry decode: unsupported problem");
    }
}

} // namespace ninfer::ops::detail
