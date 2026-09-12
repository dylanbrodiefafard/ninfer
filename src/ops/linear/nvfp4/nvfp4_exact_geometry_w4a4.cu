#include "ops/linear/nvfp4/nvfp4_launch.h"

#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"

#include <cuda_bf16.h>

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// Conservative instances of the existing SM120 warp-MMA family. Exact-shape measurements own
// any later tile or crossover change. The short-K schedules avoid padding or runtime repacking.
using M32N64K256S4 = Nvfp4W4a4MmaSchedule<32, 64, 256, 2, 4, 4, 1>;
using M32N64K128S4 = Nvfp4W4a4MmaSchedule<32, 64, 128, 2, 4, 4, 1>;
using M32N64K64S4  = Nvfp4W4a4MmaSchedule<32, 64, 64, 2, 4, 4, 1>;

template <class Geometry, class Schedule>
void launch_exact(const Weight& weight, Tensor& out, Nvfp4W4a4Workspace workspace,
                  cudaStream_t stream) {
    const std::int32_t tokens = out.ne[1];
    const dim3 grid(Geometry::kOutputRows / Schedule::kBlockN,
                    (tokens + Schedule::kBlockM - 1) / Schedule::kBlockM);
    const Nvfp4W4a4MaterializedActivation activation{workspace.codes, workspace.scales};
    const Nvfp4ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data),
                                       Geometry::kOutputRows};
    const float alpha = 1.0F / (weight.input_scale_divisor * weight.weight_scale_divisor);
    nvfp4_w4a4_mma_kernel<Geometry, Schedule><<<grid, Schedule::kThreads, 0, stream>>>(
        activation, static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), tokens, alpha, Nvfp4IdentityEpilogue{},
        output);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry>
void launch_standard(const Weight& weight, Tensor& out, Nvfp4W4a4Workspace workspace,
                     cudaStream_t stream) {
    launch_exact<Geometry, M32N64K256S4>(weight, out, workspace, stream);
}

} // namespace

void launch_nvfp4_exact_geometry_w4a4(const Weight& weight, Tensor& out,
                                      Nvfp4W4a4Workspace workspace, cudaStream_t stream) {
    switch (resolve_nvfp4_problem(weight.n, weight.k)) {
    case Nvfp4Problem::N10240K2560:
        return launch_standard<Nvfp4N10240K2560Geometry>(weight, out, workspace, stream);
    case Nvfp4Problem::N6144K2560:
        return launch_standard<Nvfp4N6144K2560Geometry>(weight, out, workspace, stream);
    case Nvfp4Problem::N12288K2560:
        return launch_standard<Nvfp4N12288K2560Geometry>(weight, out, workspace, stream);
    case Nvfp4Problem::N512K2560:
        return launch_standard<Nvfp4N512K2560Geometry>(weight, out, workspace, stream);
    case Nvfp4Problem::N2560K6144:
        return launch_standard<Nvfp4N2560K6144Geometry>(weight, out, workspace, stream);
    case Nvfp4Problem::N640K2560:
        return launch_standard<Nvfp4N640K2560Geometry>(weight, out, workspace, stream);
    case Nvfp4Problem::N1280K2560:
        return launch_standard<Nvfp4N1280K2560Geometry>(weight, out, workspace, stream);
    case Nvfp4Problem::N2560K640:
        return launch_exact<Nvfp4N2560K640Geometry, M32N64K128S4>(weight, out, workspace,
                                                                  stream);
    case Nvfp4Problem::N10240K320:
        return launch_exact<Nvfp4N10240K320Geometry, M32N64K64S4>(weight, out, workspace, stream);
    case Nvfp4Problem::N2560K2560:
        return launch_standard<Nvfp4N2560K2560Geometry>(weight, out, workspace, stream);
    default:
        throw std::invalid_argument("nvfp4 exact-geometry W4A4: unsupported problem");
    }
}

} // namespace ninfer::ops::detail
