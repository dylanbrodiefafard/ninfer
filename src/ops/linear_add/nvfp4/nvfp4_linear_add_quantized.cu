#include "ops/linear_add/nvfp4/nvfp4_linear_add_plan.h"

// A4/A8 projection routes share LinearAdd's residual epilogue.

#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_w4a8_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_w4a4_tma_launch.h"
#include "ops/linear_add/nvfp4/nvfp4_linear_add_epilogue.cuh"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

template <class Geometry, class Schedule>
void launch_gemm(const Weight& weight, Tensor& residual, Nvfp4W4a4Workspace workspace,
                 std::int32_t tokens, cudaStream_t stream) {
    const dim3 grid(Geometry::kOutputRows / Schedule::kBlockN,
                    (tokens + Schedule::kBlockM - 1) / Schedule::kBlockM);
    const Nvfp4W4a4MaterializedActivation activation{workspace.codes, workspace.scales};
    auto* output      = static_cast<__nv_bfloat16*>(residual.data);
    const float alpha = 1.0F / (weight.input_scale_divisor * weight.weight_scale_divisor);
    const Nvfp4AddResidualEpilogue epilogue{output, Geometry::kOutputRows};
    const Nvfp4ContiguousOutput out{output, Geometry::kOutputRows};
    launch_nvfp4_w4a4_mma<Geometry, Schedule, Nvfp4AddResidualEpilogue, Nvfp4ContiguousOutput>(
        grid, stream, activation, static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), tokens, alpha, epilogue, out,
        Nvfp4W4a4IdentityRows{});
}

template <Nvfp4Problem Problem, class Geometry>
void launch_problem(Nvfp4W4a4Route route, const Weight& weight, Tensor& residual,
                    Nvfp4W4a4Workspace workspace, std::int32_t tokens, cudaStream_t stream) {
    visit_nvfp4_w4a4_mma_schedule<Problem>(route, [&]<class Schedule>() {
        launch_gemm<Geometry, Schedule>(weight, residual, workspace, tokens, stream);
    });
}

} // namespace

void nvfp4_linear_add_w4a8_project(const Weight& weight, std::int32_t tokens,
                                   Fp8A8Workspace activation, Tensor& residual,
                                   cudaStream_t stream) {
    auto* data = static_cast<__nv_bfloat16*>(residual.data);
    const Nvfp4AddResidualEpilogue epilogue{data, weight.n};
    const Nvfp4ContiguousOutput out{data, weight.n};
    if (weight.k == 6144) {
        launch_nvfp4_w4a8_mma<Nvfp4Residual6144Geometry>(weight, tokens, activation, epilogue, out,
                                                         stream);
    } else {
        launch_nvfp4_w4a8_mma<Nvfp4Residual17408Geometry>(weight, tokens, activation, epilogue, out,
                                                          stream);
    }
}

void nvfp4_linear_add_w4a8_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                  Fp8A8Workspace workspace, cudaStream_t stream) {
    launch_fp8_a8_quantize(x, weight, workspace, stream);
    nvfp4_linear_add_w4a8_project(weight, x.ne[1], workspace, residual, stream);
}

void nvfp4_linear_add_w4a4_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                  Nvfp4W4a4Workspace workspace, cudaStream_t stream) {
    launch_nvfp4_w4a4_quantize(x, weight, workspace, stream);
    nvfp4_linear_add_w4a4_project(weight, x.ne[1], workspace, residual, stream);
}

void nvfp4_linear_add_w4a4_project(const Weight& weight, std::int32_t tokens,
                                   Nvfp4W4a4Workspace activation, Tensor& residual,
                                   cudaStream_t stream) {
    const Nvfp4Problem problem = resolve_nvfp4_problem(weight.n, weight.k);
    const Nvfp4W4a4Route route = nvfp4_w4a4_route(problem, tokens);
    if (route == Nvfp4W4a4Route::Tma) {
        const float alpha = 1.0F / (weight.input_scale_divisor * weight.weight_scale_divisor);
        launch_nvfp4_w4a4_tma_linear_add(problem, activation.codes, activation.scales,
                                         static_cast<const std::uint8_t*>(weight.qdata),
                                         static_cast<const std::uint8_t*>(weight.scales),
                                         static_cast<__nv_bfloat16*>(residual.data), tokens, alpha,
                                         stream);
        return;
    }
    switch (problem) {
    case Nvfp4Problem::Residual6144:
        launch_problem<Nvfp4Problem::Residual6144, Nvfp4Residual6144Geometry>(
            route, weight, residual, activation, tokens, stream);
        return;
    case Nvfp4Problem::Residual17408:
        launch_problem<Nvfp4Problem::Residual17408, Nvfp4Residual17408Geometry>(
            route, weight, residual, activation, tokens, stream);
        return;
    case Nvfp4Problem::AttnInput:
    case Nvfp4Problem::GdnInput:
    case Nvfp4Problem::MlpGateUp:
    case Nvfp4Problem::DflashFeature:
    case Nvfp4Problem::DflashQkv:
    case Nvfp4Problem::DflashAttnOut:
    case Nvfp4Problem::DflashConvProj:
    case Nvfp4Problem::DflashSelector:
    case Nvfp4Problem::MtpFc:
        break;
    }
    throw std::invalid_argument("nvfp4 linear_add: unsupported problem");
}

} // namespace ninfer::ops::detail
