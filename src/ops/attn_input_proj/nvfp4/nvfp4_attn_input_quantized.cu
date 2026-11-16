#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"

// A4 and A8 share the semantic split-output policy; their MMA arithmetic remains separate.

#include "core/device.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_output.cuh"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_format.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_w4a4_tma_launch.h"
#include "ops/linear/nvfp4/nvfp4_w4a8_mma.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

using Geometry = Nvfp4AttnInputGeometry;

static_assert(Geometry::kOutputRows == Nvfp4AttentionInput5120Output::kParentRows);

template <class Schedule>
void launch_gemm(const Weight& weight, Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                 Nvfp4W4a4Workspace workspace, std::int32_t tokens, cudaStream_t stream) {
    const dim3 grid(Geometry::kOutputRows / Schedule::kBlockN,
                    (tokens + Schedule::kBlockM - 1) / Schedule::kBlockM);
    const Nvfp4W4a4MaterializedActivation activation{workspace.codes, workspace.scales};
    const Nvfp4AttentionInput5120Output output{
        static_cast<__nv_bfloat16*>(q.data),
        static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(gate.data),
        static_cast<__nv_bfloat16*>(v.data),
    };
    const float alpha = nvfp4_w4a4_alpha(weight, "nvfp4 attn_input_proj");
    launch_nvfp4_w4a4_mma<Geometry, Schedule, Nvfp4IdentityEpilogue, Nvfp4AttentionInput5120Output>(
        grid, stream, activation, static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), tokens, alpha, Nvfp4IdentityEpilogue{},
        output, Nvfp4W4a4IdentityRows{});
}

} // namespace

void nvfp4_attn_input_w4a8_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                  Tensor& k, Tensor& v, Fp8A8Workspace workspace,
                                  cudaStream_t stream) {
    launch_fp8_a8_quantize(x, weight, workspace, stream);
    nvfp4_attn_input_w4a8_project(weight, x.ne[1], workspace, q, gate, k, v, stream);
}

void nvfp4_attn_input_w4a8_project(const Weight& weight, std::int32_t tokens,
                                   Fp8A8Workspace activation, Tensor& q, Tensor& gate, Tensor& k,
                                   Tensor& v, cudaStream_t stream) {
    launch_nvfp4_w4a8_mma<Geometry>(
        weight, tokens, activation, Nvfp4IdentityEpilogue{},
        Nvfp4AttentionInput5120Output{
            static_cast<__nv_bfloat16*>(q.data), static_cast<__nv_bfloat16*>(k.data),
            static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(v.data)},
        stream);
}

void nvfp4_attn_input_w4a4_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                  Tensor& k, Tensor& v, Nvfp4W4a4Workspace workspace,
                                  cudaStream_t stream) {
    launch_nvfp4_w4a4_quantize(x, weight, workspace, stream);
    nvfp4_attn_input_w4a4_project(weight, x.ne[1], workspace, q, gate, k, v, stream);
}

void nvfp4_attn_input_w4a4_project(const Weight& weight, std::int32_t tokens,
                                   Nvfp4W4a4Workspace activation, Tensor& q, Tensor& gate,
                                   Tensor& k, Tensor& v, cudaStream_t stream) {
    const Nvfp4W4a4Route route = nvfp4_w4a4_route(Nvfp4Problem::AttnInput, tokens);
    if (route == Nvfp4W4a4Route::Tma) {
        const float alpha = nvfp4_w4a4_alpha(weight, "nvfp4 attn_input_proj");
        launch_nvfp4_w4a4_tma_attention(
            activation.codes, activation.scales, static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(q.data),
            static_cast<__nv_bfloat16*>(gate.data), static_cast<__nv_bfloat16*>(k.data),
            static_cast<__nv_bfloat16*>(v.data), tokens, alpha, stream);
        return;
    }
    visit_nvfp4_w4a4_mma_schedule<Nvfp4Problem::AttnInput>(route, [&]<class Schedule>() {
        launch_gemm<Schedule>(weight, q, gate, k, v, activation, tokens, stream);
    });
}

} // namespace ninfer::ops::detail
