#include "ops/linear/nvfp4/nvfp4_w4a8_plan.h"
#include "ops/linear/nvfp4/nvfp4_w4a8_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_config.h"

namespace ninfer::ops::detail {
namespace {
template <class Geometry>
void launch(const Tensor& x, const Weight& weight, Tensor& out,
            Fp8A8Workspace workspace, cudaStream_t stream) {
    launch_nvfp4_w4a8_mma<Geometry>(weight, x.ne[1], workspace, Nvfp4IdentityEpilogue{},
        Nvfp4ContiguousOutput{static_cast<__nv_bfloat16*>(out.data), weight.n}, stream);
}
}

void launch_nvfp4_w4a8(const Tensor& x, const Weight& weight, Tensor& out,
                      Fp8A8Workspace workspace, cudaStream_t stream) {
    launch_fp8_a8_quantize(x, weight, workspace, stream);
    switch (resolve_nvfp4_problem(weight.n, weight.k)) {
    case Nvfp4Problem::AttnInput: launch<Nvfp4AttnInputGeometry>(x, weight, out, workspace, stream); return;
    case Nvfp4Problem::GdnInput: launch<Nvfp4GdnInputGeometry>(x, weight, out, workspace, stream); return;
    case Nvfp4Problem::MlpGateUp: launch<Nvfp4MlpGateUpGeometry>(x, weight, out, workspace, stream); return;
    case Nvfp4Problem::Residual6144: launch<Nvfp4Residual6144Geometry>(x, weight, out, workspace, stream); return;
    case Nvfp4Problem::Residual17408: launch<Nvfp4Residual17408Geometry>(x, weight, out, workspace, stream); return;
    default: throw std::invalid_argument("NVFP4 A8: unsupported problem");
    }
}
} // namespace ninfer::ops::detail
