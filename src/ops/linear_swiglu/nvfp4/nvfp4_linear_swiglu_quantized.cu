#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"
#include "ops/kernel/rmsnorm.cuh"

// A4/A8 routes share the paired gate/up epilogue and its private BF16 projection staging.

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/kernel/a4_activation.cuh"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_output.cuh"
#include "ops/linear/nvfp4/nvfp4_w4a4_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_w4a8_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma_launch.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

using Geometry = Nvfp4MlpGateUpGeometry;

constexpr int kIntermediate = Geometry::kOutputRows / 2;

template <class Schedule>
struct Nvfp4SwiGluRows {
    static constexpr bool kContiguous   = false;
    static constexpr int kRowsPerBranch = Schedule::kBlockN / 2;

    __device__ __forceinline__ int weight_row(int row_begin, int local_row) const {
        return row_begin + (local_row & (kRowsPerBranch - 1)) +
               (local_row >= kRowsPerBranch ? kIntermediate : 0);
    }
};

union Nvfp4SwiGluBf16Pair {
    unsigned bits;
    __nv_bfloat162 values;
};

// BF16 silu(gate) * up of two staged BF16x2 words.
__device__ __forceinline__ unsigned nvfp4_swiglu_combine(unsigned gate_bits, unsigned up_bits) {
    Nvfp4SwiGluBf16Pair gate{gate_bits};
    Nvfp4SwiGluBf16Pair up{up_bits};
    const float2 gate_values = __bfloat1622float2(gate.values);
    const float2 up_values   = __bfloat1622float2(up.values);
    Nvfp4SwiGluBf16Pair result;
    result.values =
        __floats2bfloat162_rn(silu(gate_values.x) * up_values.x, silu(gate_values.y) * up_values.y);
    return result.bits;
}

__device__ __forceinline__ uint4 nvfp4_swiglu_combine(uint4 gate, uint4 up) {
    return make_uint4(nvfp4_swiglu_combine(gate.x, up.x), nvfp4_swiglu_combine(gate.y, up.y),
                      nvfp4_swiglu_combine(gate.z, up.z), nvfp4_swiglu_combine(gate.w, up.w));
}

struct Nvfp4SwiGluOutput {
    __nv_bfloat16* data;

    __device__ __forceinline__ void store_pair_vector(std::int32_t row, std::int32_t token,
                                                      uint4 gate, uint4 up) const {
        store_vec(data + static_cast<std::int64_t>(token) * kIntermediate + row,
                  nvfp4_swiglu_combine(gate, up));
    }
};

// The NVFP4 activation (ninfer/ops/a4_activation.h) of the BF16 SwiGLU output for the down
// projection, published one 16-row group at a time by the W4A4 MMA kernel's group epilogue.
struct Nvfp4SwiGluA4Output {
    static constexpr int kGroupRows = 16;
    std::uint8_t* codes;
    std::uint8_t* scales;
    float input_scale_divisor;

    __device__ __forceinline__ void store_pair_group(std::int32_t row, std::int32_t token,
                                                     uint4 gate_lo, uint4 gate_hi, uint4 up_lo,
                                                     uint4 up_hi) const {
        a4_publish_group16<kIntermediate>(codes, scales, input_scale_divisor, token, row / 16,
                                          nvfp4_swiglu_combine(gate_lo, up_lo),
                                          nvfp4_swiglu_combine(gate_hi, up_hi));
    }

    __device__ __forceinline__ void store_padding_group(std::int32_t row,
                                                        std::int32_t token) const {
        scales[a4_scale_offset<kIntermediate / 16>(token, row / 16)] = 0;
    }
};

template <class Schedule, class Output>
void launch_gemm(const Weight& weight, Output output, Nvfp4W4a4Workspace workspace,
                 std::int32_t tokens, cudaStream_t stream) {
    constexpr int kPairRows = Schedule::kBlockN / 2;
    const dim3 grid(kIntermediate / kPairRows,
                    (tokens + Schedule::kBlockM - 1) / Schedule::kBlockM);
    const Nvfp4W4a4MaterializedActivation activation{workspace.codes, workspace.scales};
    const Nvfp4SwiGluRows<Schedule> row_policy{};
    const float alpha = 1.0F / (weight.input_scale_divisor * weight.weight_scale_divisor);
    launch_nvfp4_w4a4_mma<Geometry, Schedule, Nvfp4IdentityEpilogue, Output,
                          Nvfp4SwiGluRows<Schedule>, true, Cache::cg>(
        grid, stream, activation, static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), tokens, alpha, Nvfp4IdentityEpilogue{},
        output, row_policy);
}

} // namespace

void nvfp4_linear_swiglu_a8_project(const Weight& weight, Tensor& out, Fp8A8Workspace scratch,
                                    cudaStream_t stream) {
    using Schedule = Nvfp4W4a4MmaSchedule<32, 64, 128, 1, 4, 2, 1>;
    launch_nvfp4_w4a8_mma<Geometry, Nvfp4IdentityEpilogue, Nvfp4SwiGluOutput,
                          Nvfp4SwiGluRows<Schedule>, true>(
        weight, out.ne[1], scratch, Nvfp4IdentityEpilogue{},
        Nvfp4SwiGluOutput{static_cast<__nv_bfloat16*>(out.data)}, stream);
}

void nvfp4_linear_swiglu_w4a8_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                     WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope         = workspace.scope();
    const auto scratch = allocate_fp8_a8_workspace(workspace, x.ne[1], weight.k);
    launch_fp8_a8_quantize(x, weight, scratch, stream);
    nvfp4_linear_swiglu_a8_project(weight, out, scratch, stream);
}

void nvfp4_linear_swiglu_w4a4_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                     WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope = workspace.scope();
    const Nvfp4W4a4Workspace scratch =
        allocate_nvfp4_w4a4_workspace(workspace, x.ne[1], Geometry::kInputRows);
    launch_nvfp4_w4a4_quantize(x, weight, scratch, stream);
    nvfp4_linear_swiglu_w4a4_project(weight, x.ne[1], scratch, out, stream);
}

void nvfp4_linear_swiglu_w4a4_project(const Weight& weight, std::int32_t tokens,
                                      Nvfp4W4a4Workspace activation, Tensor& out,
                                      cudaStream_t stream) {
    const Nvfp4W4a4Route route = nvfp4_w4a4_route(Nvfp4Problem::MlpGateUp, tokens);
    if (route == Nvfp4W4a4Route::Tma) {
        const float alpha = 1.0F / (weight.input_scale_divisor * weight.weight_scale_divisor);
        launch_nvfp4_linear_swiglu_w4a4_tma(
            activation.codes, activation.scales, static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(out.data),
            tokens, alpha, stream);
        return;
    }
    visit_nvfp4_w4a4_mma_schedule<Nvfp4Problem::MlpGateUp>(route, [&]<class Schedule>() {
        launch_gemm<Schedule>(weight, Nvfp4SwiGluOutput{static_cast<__nv_bfloat16*>(out.data)},
                              activation, tokens, stream);
    });
}

void nvfp4_linear_swiglu_w4a4_project_a4(const Weight& weight, std::int32_t tokens,
                                         Nvfp4W4a4Workspace activation, Nvfp4W4a4Workspace output,
                                         float output_input_scale_divisor, cudaStream_t stream) {
    const Nvfp4W4a4Route route = nvfp4_w4a4_route(Nvfp4Problem::MlpGateUp, tokens);
    if (route == Nvfp4W4a4Route::Tma) {
        const float alpha = 1.0F / (weight.input_scale_divisor * weight.weight_scale_divisor);
        launch_nvfp4_linear_swiglu_w4a4_tma_a4(
            activation.codes, activation.scales, static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), output.codes, output.scales,
            output_input_scale_divisor, tokens, alpha, stream);
        return;
    }
    visit_nvfp4_w4a4_mma_schedule<Nvfp4Problem::MlpGateUp>(route, [&]<class Schedule>() {
        launch_gemm<Schedule>(
            weight, Nvfp4SwiGluA4Output{output.codes, output.scales, output_input_scale_divisor},
            activation, tokens, stream);
    });
}

void nvfp4_rmsnorm_linear_swiglu_launch(const Tensor& x, const Tensor& norm_weight, float eps,
                                        const Weight& gate_up, Tensor& out,
                                        WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope         = workspace.scope();
    const auto scratch = detail::allocate_fp8_a8_workspace(workspace, x.ne[1], 5120);
    rmsnorm_cta_bf16x2_kernel<RmsEpilogue::Offset, 512, 8, RmsOutput::A8>
        <<<x.ne[1], 512, 0, stream>>>(static_cast<const __nv_bfloat162*>(x.data),
                                      static_cast<const __nv_bfloat162*>(norm_weight.data), nullptr,
                                      nullptr, 5120, x.ne[1], eps, scratch.codes, scratch.scales);
    CUDA_CHECK(cudaGetLastError());
    detail::nvfp4_linear_swiglu_a8_project(gate_up, out, scratch, stream);
}

} // namespace ninfer::ops::detail
