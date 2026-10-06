#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"

// A4 and A8 share the semantic split-output policy; their MMA arithmetic remains separate.

#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_w4a8_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_w4a4_tma_launch.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

using Geometry = Nvfp4AttnInputGeometry;

constexpr std::int32_t kQueryRows  = 6144;
constexpr std::int32_t kKeyRows    = 1024;
constexpr std::int32_t kGateRows   = 6144;
constexpr std::int32_t kKeyBegin   = kQueryRows;
constexpr std::int32_t kGateBegin  = kKeyBegin + kKeyRows;
constexpr std::int32_t kValueBegin = kGateBegin + kGateRows;

static_assert((kQueryRows % 128) == 0);
static_assert((kKeyRows % 128) == 0);
static_assert((kGateRows % 128) == 0);

struct Nvfp4W4a4AttentionOutput {
    __nv_bfloat16* query;
    __nv_bfloat16* key;
    __nv_bfloat16* gate;
    __nv_bfloat16* value;

    __device__ __forceinline__ __nv_bfloat16* destination(std::int32_t parent_row,
                                                          std::int32_t token) const {
        if (parent_row < kKeyBegin) {
            return query + static_cast<std::int64_t>(token) * kQueryRows + parent_row;
        }
        if (parent_row < kGateBegin) {
            return key + static_cast<std::int64_t>(token) * kKeyRows + parent_row - kKeyBegin;
        }
        if (parent_row < kValueBegin) {
            return gate + static_cast<std::int64_t>(token) * kGateRows + parent_row - kGateBegin;
        }
        return value + static_cast<std::int64_t>(token) * kKeyRows + parent_row - kValueBegin;
    }

    __device__ __forceinline__ void store_vector(std::int32_t parent_row, std::int32_t token,
                                                 uint4 values) const {
        store_vec(destination(parent_row, token), values);
    }
};

template <class Schedule>
void launch_gemm(const Weight& weight, Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                 Nvfp4W4a4Workspace workspace, std::int32_t tokens, cudaStream_t stream) {
    const dim3 grid(Geometry::kOutputRows / Schedule::kBlockN,
                    (tokens + Schedule::kBlockM - 1) / Schedule::kBlockM);
    const Nvfp4W4a4MaterializedActivation activation{workspace.codes, workspace.scales};
    const Nvfp4W4a4AttentionOutput output{
        static_cast<__nv_bfloat16*>(q.data),
        static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(gate.data),
        static_cast<__nv_bfloat16*>(v.data),
    };
    const float alpha = 1.0F / (weight.input_scale_divisor * weight.weight_scale_divisor);
    launch_nvfp4_w4a4_mma<Geometry, Schedule, Nvfp4IdentityEpilogue, Nvfp4W4a4AttentionOutput>(
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
    launch_nvfp4_w4a8_mma<Geometry>(weight, tokens, activation, Nvfp4IdentityEpilogue{},
                                    Nvfp4W4a4AttentionOutput{static_cast<__nv_bfloat16*>(q.data),
                                                             static_cast<__nv_bfloat16*>(k.data),
                                                             static_cast<__nv_bfloat16*>(gate.data),
                                                             static_cast<__nv_bfloat16*>(v.data)},
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
        const float alpha = 1.0F / (weight.input_scale_divisor * weight.weight_scale_divisor);
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
