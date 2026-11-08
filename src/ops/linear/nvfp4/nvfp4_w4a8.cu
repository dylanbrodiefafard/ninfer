#include "ops/linear/nvfp4/nvfp4_w4a8_plan.h"
#include "ops/linear/nvfp4/nvfp4_w4a8_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_config.h"

#include <stdexcept>

namespace ninfer::ops::detail {

void launch_nvfp4_a16_mma(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const Nvfp4Bf16Activation activation{static_cast<const __nv_bfloat16*>(x.data)};
    const auto run = [&]<class Geometry>() {
        launch_nvfp4_w4a8_mma<Geometry>(
            weight, x.ne[1], activation, Nvfp4IdentityEpilogue{},
            Nvfp4ContiguousOutput{static_cast<__nv_bfloat16*>(out.data), weight.n}, stream);
    };
    switch (resolve_nvfp4_problem(weight.n, weight.k)) {
    case Nvfp4Problem::DflashFeature:
        run.template operator()<Nvfp4DflashFeatureGeometry>();
        return;
    case Nvfp4Problem::DflashQkv:
        run.template operator()<Nvfp4DflashQkvGeometry>();
        return;
    case Nvfp4Problem::DflashAttnOut:
        run.template operator()<Nvfp4DflashAttnOutGeometry>();
        return;
    default:
        throw std::invalid_argument("NVFP4 A16 MMA: unsupported problem");
    }
}
} // namespace ninfer::ops::detail
