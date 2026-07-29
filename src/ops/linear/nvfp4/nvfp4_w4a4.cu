#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"

#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_w4a4_tma_launch.h"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

using Geometry = Nvfp4LinearDecodeGeometry;

using M32N64                      = Nvfp4W4a4MmaSchedule<32, 64, 256, 2, 4, 2, 2>;
using M32N128                     = Nvfp4W4a4MmaSchedule<32, 128, 256, 2, 4, 2, 1>;
using M64N128                     = Nvfp4W4a4MmaSchedule<64, 128, 256, 4, 2, 2, 1>;
using M128N128Pipelined           = Nvfp4W4a4MmaSchedule<128, 128, 256, 4, 2, 2, 1>;
using M128N128Resident            = Nvfp4W4a4MmaSchedule<128, 128, 256, 4, 2, 1, 2>;
constexpr std::int32_t kTmaBlockM = 256;

template <class Schedule>
void launch_gemm(const Weight& weight, Tensor& out, Nvfp4W4a4Workspace workspace,
                 std::int32_t tokens, cudaStream_t stream) {
    const dim3 grid(Geometry::kOutputRows / Schedule::kBlockN,
                    (tokens + Schedule::kBlockM - 1) / Schedule::kBlockM);
    const Nvfp4W4a4MaterializedActivation activation{workspace.codes, workspace.scales};
    const Nvfp4W4a4ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data),
                                           Geometry::kOutputRows};
    const float alpha = 1.0F / (weight.input_scale_divisor * weight.weight_scale_divisor);
    nvfp4_w4a4_mma_kernel<Geometry, Schedule><<<grid, Schedule::kThreads, 0, stream>>>(
        activation, static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), tokens, alpha, output);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_nvfp4_w4a4_quantize(const Tensor& x, const Weight& weight, Nvfp4W4a4Workspace workspace,
                                cudaStream_t stream) {
    if (workspace.codes == nullptr || workspace.scales == nullptr) {
        throw std::invalid_argument("nvfp4 W4A4 requires caller workspace");
    }
    const std::int32_t tokens = x.ne[1];
    constexpr int kThreads    = 256;
    const std::int32_t tasks  = tokens * Geometry::kGroupsPerRow;
    nvfp4_w4a4_quantize_kernel<Geometry>
        <<<(tasks + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data), workspace.codes, workspace.scales, tokens,
            weight.input_scale_divisor);
    CUDA_CHECK(cudaGetLastError());
}

void launch_nvfp4_w4a4(const Tensor& x, const Weight& weight, Tensor& out,
                       Nvfp4W4a4Workspace workspace, cudaStream_t stream) {
    launch_nvfp4_w4a4_quantize(x, weight, workspace, stream);
    const std::int32_t tokens = x.ne[1];
    if (tokens >= 1024 && (tokens % kTmaBlockM) == 0) {
        const float alpha = 1.0F / (weight.input_scale_divisor * weight.weight_scale_divisor);
        launch_nvfp4_w4a4_tma_linear(workspace.codes, workspace.scales,
                                     static_cast<const std::uint8_t*>(weight.qdata),
                                     static_cast<const std::uint8_t*>(weight.scales),
                                     static_cast<__nv_bfloat16*>(out.data), tokens, alpha, stream);
    } else if (tokens <= 64) {
        launch_gemm<M32N64>(weight, out, workspace, tokens, stream);
    } else if (tokens <= 96) {
        launch_gemm<M32N128>(weight, out, workspace, tokens, stream);
    } else if (tokens <= 128) {
        launch_gemm<M128N128Pipelined>(weight, out, workspace, tokens, stream);
    } else if (tokens <= 192) {
        launch_gemm<M64N128>(weight, out, workspace, tokens, stream);
    } else if (tokens <= 384) {
        launch_gemm<M128N128Resident>(weight, out, workspace, tokens, stream);
    } else if (tokens <= 512) {
        launch_gemm<M128N128Pipelined>(weight, out, workspace, tokens, stream);
    } else {
        launch_gemm<M128N128Resident>(weight, out, workspace, tokens, stream);
    }
}

} // namespace ninfer::ops::detail
