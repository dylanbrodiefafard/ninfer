#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"

// Both quantized activation profiles preserve the FP32 current projection for record replay.

#include "core/device.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_output.cuh"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_w4a8_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_w4a4_tma_launch.h"

namespace ninfer::ops::detail {
namespace {

using Geometry = Nvfp4GdnInputGeometry;

struct GdnFp32ProjectionOutput {
    float* qkv;
    __nv_bfloat16* z;

    __device__ __forceinline__ void store_fp32(int row, int token, float value) const {
        if (row < 10240) {
            qkv[static_cast<std::int64_t>(token) * 10240 + row] = value;
        } else {
            z[static_cast<std::int64_t>(token) * 6144 + row - 10240] = __float2bfloat16_rn(value);
        }
    }
};

template <class Schedule>
void launch_gemm(const Weight& weight, Tensor& qkv, Tensor& z, Nvfp4W4a4Workspace workspace,
                 std::int32_t tokens, cudaStream_t stream) {
    const dim3 grid(Geometry::kOutputRows / Schedule::kBlockN,
                    (tokens + Schedule::kBlockM - 1) / Schedule::kBlockM);
    const Nvfp4W4a4MaterializedActivation activation{workspace.codes, workspace.scales};
    const float alpha = 1.0F / (weight.input_scale_divisor * weight.weight_scale_divisor);
    const Nvfp4GdnInputOutput output{static_cast<__nv_bfloat16*>(qkv.data),
                                     static_cast<__nv_bfloat16*>(z.data)};
    launch_nvfp4_w4a4_mma<Geometry, Schedule, Nvfp4IdentityEpilogue, Nvfp4GdnInputOutput>(
        grid, stream, activation, static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), tokens, alpha, Nvfp4IdentityEpilogue{},
        output, Nvfp4W4a4IdentityRows{});
}

} // namespace

void nvfp4_gdn_input_w4a8_fp32_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                      Fp8A8Workspace workspace, cudaStream_t stream) {
    launch_fp8_a8_quantize(x, weight, workspace, stream);
    nvfp4_gdn_input_w4a8_fp32_project(weight, x.ne[1], workspace, qkv, z, stream);
}

void nvfp4_gdn_input_w4a8_fp32_project(const Weight& weight, std::int32_t tokens,
                                       Fp8A8Workspace activation, Tensor& qkv, Tensor& z,
                                       cudaStream_t stream) {
    launch_nvfp4_w4a8_mma<Geometry>(
        weight, tokens, activation, Nvfp4IdentityEpilogue{},
        GdnFp32ProjectionOutput{static_cast<float*>(qkv.data), static_cast<__nv_bfloat16*>(z.data)},
        stream);
}

void nvfp4_gdn_input_w4a4_fp32_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                      Nvfp4W4a4Workspace workspace, cudaStream_t stream) {
    launch_nvfp4_w4a4_quantize(x, weight, workspace, stream);
    using Schedule   = Nvfp4W4a4M32N64S4;
    const int tokens = x.ne[1];
    const dim3 grid(Geometry::kOutputRows / Schedule::kBlockN,
                    (tokens + Schedule::kBlockM - 1) / Schedule::kBlockM);
    const float alpha = 1.0F / (weight.input_scale_divisor * weight.weight_scale_divisor);
    launch_nvfp4_w4a4_mma<Geometry, Schedule, Nvfp4IdentityEpilogue, GdnFp32ProjectionOutput>(
        grid, stream, Nvfp4W4a4MaterializedActivation{workspace.codes, workspace.scales},
        static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), tokens, alpha, Nvfp4IdentityEpilogue{},
        GdnFp32ProjectionOutput{static_cast<float*>(qkv.data), static_cast<__nv_bfloat16*>(z.data)},
        Nvfp4W4a4IdentityRows{});
}

void nvfp4_gdn_input_w4a4_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                 Nvfp4W4a4Workspace workspace, cudaStream_t stream) {
    launch_nvfp4_w4a4_quantize(x, weight, workspace, stream);
    nvfp4_gdn_input_w4a4_project(weight, x.ne[1], workspace, qkv, z, stream);
}

void nvfp4_gdn_input_w4a4_project(const Weight& weight, std::int32_t tokens,
                                  Nvfp4W4a4Workspace activation, Tensor& qkv, Tensor& z,
                                  cudaStream_t stream) {
    const Nvfp4W4a4Route route = nvfp4_w4a4_route(Nvfp4Problem::GdnInput, tokens);
    if (route == Nvfp4W4a4Route::Tma) {
        const float alpha = 1.0F / (weight.input_scale_divisor * weight.weight_scale_divisor);
        launch_nvfp4_w4a4_tma_gdn(
            activation.codes, activation.scales, static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(qkv.data),
            static_cast<__nv_bfloat16*>(z.data), tokens, alpha, stream);
        return;
    }
    visit_nvfp4_w4a4_mma_schedule<Nvfp4Problem::GdnInput>(route, [&]<class Schedule>() {
        launch_gemm<Schedule>(weight, qkv, z, activation, tokens, stream);
    });
}

} // namespace ninfer::ops::detail
