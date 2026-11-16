#include "ops/linear/nvfp4/nvfp4_w4a8_plan.h"
#include "ops/linear/nvfp4/nvfp4_w4a8_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_a16_small_t_mma.cuh"

#include <stdexcept>

namespace ninfer::ops::detail {

void launch_nvfp4_a16_mma(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const Nvfp4Bf16Activation activation{static_cast<const __nv_bfloat16*>(x.data)};
    const auto run = [&]<class Geometry>() {
        // Narrow drafter projections split K across warps through sixteen tokens. Wider output
        // grids and the 27B geometries retain their independently measured streaming schedules.
        if constexpr (nvfp4_a16_k_split_problem(Geometry::kOutputRows, Geometry::kInputRows)) {
            if (x.ne[1] <= kNvfp4A16KSplitLastTokens) {
                launch_nvfp4_a16_small_t_mma<Geometry>(x, weight, out, stream);
                return;
            }
        }
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
    case Nvfp4Problem::FlashNextMixerOut:
        run.template operator()<Nvfp4FlashNextMixerOutGeometry>();
        return;
    case Nvfp4Problem::FlashNextDflashFeature:
        run.template operator()<Nvfp4FlashNextDflashFeatureGeometry>();
        return;
    case Nvfp4Problem::FlashNextDflashQkv:
        run.template operator()<Nvfp4FlashNextDflashQkvGeometry>();
        return;
    case Nvfp4Problem::FlashNextDflashAttnOut:
        run.template operator()<Nvfp4FlashNextDflashAttnOutGeometry>();
        return;
    case Nvfp4Problem::FlashNextDflashConvProj:
        run.template operator()<Nvfp4FlashNextDflashConvProjGeometry>();
        return;
    case Nvfp4Problem::FlashNextDflashGateUp:
        run.template operator()<Nvfp4FlashNextDflashGateUpGeometry>();
        return;
    case Nvfp4Problem::FlashNextDflashDown:
        run.template operator()<Nvfp4FlashNextDflashDownGeometry>();
        return;
    default:
        throw std::invalid_argument("NVFP4 A16 MMA: unsupported problem");
    }
}
} // namespace ninfer::ops::detail
