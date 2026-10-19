#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"

#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_w4a4_tma_launch.h"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

template <class Geometry, class Schedule>
void launch_gemm(const Weight& weight, Tensor& out, Nvfp4W4a4Workspace workspace,
                 std::int32_t tokens, cudaStream_t stream) {
    const dim3 grid(Geometry::kOutputRows / Schedule::kBlockN,
                    (tokens + Schedule::kBlockM - 1) / Schedule::kBlockM);
    const Nvfp4W4a4MaterializedActivation activation{workspace.codes, workspace.scales};
    const Nvfp4ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data),
                                       Geometry::kOutputRows};
    const float alpha = 1.0F / (weight.input_scale_divisor * weight.weight_scale_divisor);
    launch_nvfp4_w4a4_mma<Geometry, Schedule, Nvfp4IdentityEpilogue, Nvfp4ContiguousOutput>(
        grid, stream, activation, static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), tokens, alpha, Nvfp4IdentityEpilogue{},
        output, Nvfp4W4a4IdentityRows{});
}

template <class ActivationGeometry>
void launch_quantize_exact(const Tensor& x, const Weight& weight, Nvfp4W4a4Workspace workspace,
                           cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    constexpr int kThreads    = 256;
    const std::int32_t tasks  = (tokens + 15) / 16 * 16 * ActivationGeometry::kGroupsPerRow;
    CUDA_CHECK(pdl::launch_dependent(
        pdl::LaunchConfig{dim3((tasks + kThreads - 1) / kThreads), dim3(kThreads), 0, stream},
        nvfp4_w4a4_quantize_kernel<ActivationGeometry, kThreads>,
        static_cast<const __nv_bfloat16*>(x.data), workspace.codes, workspace.scales, tokens,
        weight.input_scale_divisor));
}

template <Nvfp4Problem Problem, class Geometry>
void launch_problem(Nvfp4W4a4Route route, const Weight& weight, Tensor& out,
                    Nvfp4W4a4Workspace workspace, std::int32_t tokens, cudaStream_t stream) {
    if (route == Nvfp4W4a4Route::Tma) {
        const float alpha = 1.0F / (weight.input_scale_divisor * weight.weight_scale_divisor);
        launch_nvfp4_w4a4_tma_linear(Problem, workspace.codes, workspace.scales,
                                     static_cast<const std::uint8_t*>(weight.qdata),
                                     static_cast<const std::uint8_t*>(weight.scales),
                                     static_cast<__nv_bfloat16*>(out.data), tokens, alpha, stream);
        return;
    }
    visit_nvfp4_w4a4_mma_schedule<Problem>(route, [&]<class Schedule>() {
        launch_gemm<Geometry, Schedule>(weight, out, workspace, tokens, stream);
    });
}

} // namespace

void launch_nvfp4_w4a4_quantize(const Tensor& x, const Weight& weight, Nvfp4W4a4Workspace workspace,
                                cudaStream_t stream) {
    if (workspace.codes == nullptr || workspace.scales == nullptr) {
        throw std::invalid_argument("nvfp4 W4A4 requires caller workspace");
    }
    switch (weight.k) {
    case Nvfp4Activation5120Geometry::kInputRows:
        launch_quantize_exact<Nvfp4Activation5120Geometry>(x, weight, workspace, stream);
        return;
    case Nvfp4Activation6144Geometry::kInputRows:
        launch_quantize_exact<Nvfp4Activation6144Geometry>(x, weight, workspace, stream);
        return;
    case Nvfp4Activation17408Geometry::kInputRows:
        launch_quantize_exact<Nvfp4Activation17408Geometry>(x, weight, workspace, stream);
        return;
    case Nvfp4Activation10240Geometry::kInputRows:
        launch_quantize_exact<Nvfp4Activation10240Geometry>(x, weight, workspace, stream);
        return;
    default:
        throw std::invalid_argument("nvfp4 W4A4 quantize: unsupported K");
    }
}

void launch_nvfp4_w4a4(const Tensor& x, const Weight& weight, Tensor& out,
                       Nvfp4W4a4Workspace workspace, cudaStream_t stream) {
    const std::int32_t tokens  = x.ne[1];
    const Nvfp4Problem problem = resolve_nvfp4_problem(weight.n, weight.k);
    const Nvfp4W4a4Route route = nvfp4_w4a4_route(problem, tokens);
    launch_nvfp4_w4a4_quantize(x, weight, workspace, stream);
    switch (problem) {
    case Nvfp4Problem::AttnInput:
        launch_problem<Nvfp4Problem::AttnInput, Nvfp4AttnInputGeometry>(route, weight, out,
                                                                        workspace, tokens, stream);
        return;
    case Nvfp4Problem::GdnInput:
        launch_problem<Nvfp4Problem::GdnInput, Nvfp4GdnInputGeometry>(route, weight, out, workspace,
                                                                      tokens, stream);
        return;
    case Nvfp4Problem::MlpGateUp:
        launch_problem<Nvfp4Problem::MlpGateUp, Nvfp4MlpGateUpGeometry>(route, weight, out,
                                                                        workspace, tokens, stream);
        return;
    case Nvfp4Problem::Residual6144:
        launch_problem<Nvfp4Problem::Residual6144, Nvfp4Residual6144Geometry>(
            route, weight, out, workspace, tokens, stream);
        return;
    case Nvfp4Problem::Residual17408:
        launch_problem<Nvfp4Problem::Residual17408, Nvfp4Residual17408Geometry>(
            route, weight, out, workspace, tokens, stream);
        return;
    case Nvfp4Problem::MtpFc:
        launch_problem<Nvfp4Problem::MtpFc, Nvfp4MtpFcGeometry>(route, weight, out, workspace,
                                                                tokens, stream);
        return;
    case Nvfp4Problem::DflashFeature:
    case Nvfp4Problem::DflashQkv:
    case Nvfp4Problem::DflashAttnOut:
    case Nvfp4Problem::DflashConvProj:
    case Nvfp4Problem::DflashSelector:
        break;
    }
    throw std::invalid_argument("nvfp4 W4A4 linear: DFlash2 problems are A16-only");
}

} // namespace ninfer::ops::detail
