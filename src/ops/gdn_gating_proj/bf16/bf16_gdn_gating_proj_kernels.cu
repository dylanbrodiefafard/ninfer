#include "ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_kernels.h"

#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_gemm_mma.cuh"
#include "ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.h"

#include "core/device.h" // CUDA_CHECK

#include <cuda_bf16.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ninfer::ops::detail {
namespace {

constexpr int kN = 48;
constexpr int kK = 5120;

template <int Columns>
struct Bf16Gdn27SmallGeometry {
    static constexpr int kHeads  = 48;
    static constexpr int kHidden = 5120;
    static constexpr int kBlockN = Columns;
};

void require_shape(const Weight& w, const char* name) {
    if (w.n != kN || w.k != kK || w.shape[0] != kN || w.shape[1] != kK) {
        throw std::invalid_argument(std::string("gdn_gating_proj: ") + name +
                                    " requires contiguous BF16 [48,5120]");
    }
}

template <class Geometry, int SplitK, int Warps = kBf16GdnWarps>
void launch_bf16_prefill_mma(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                             const Weight& a_weight, const Weight& b_weight, const Tensor& A_log,
                             const Tensor& dt_bias, void* workspace, Tensor& g, Tensor& beta,
                             cudaStream_t stream) {
    const std::int32_t t     = x.ne[1];
    constexpr int kBlockN    = Geometry::kBlockN;
    constexpr int kSmemBytes = kBf16GdnSmemBytes<kBlockN>;
    const dim3 block(Warps * 32);
    const dim3 grid(static_cast<unsigned>(div_up(t, kBlockN)),
                    static_cast<unsigned>(Geometry::kHeads / kBf16GdnBlockM),
                    static_cast<unsigned>(SplitK));
    auto launch = [&](auto full_tokens) {
        constexpr bool FullTokens     = decltype(full_tokens)::value;
        static const cudaError_t attr = cudaFuncSetAttribute(
            bf16_gdn_gating_proj_gemm_mma_kernel<Geometry, SplitK, FullTokens, Warps>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);
        CUDA_CHECK(attr);
        if constexpr (SplitK > 1) {
            cudaLaunchConfig_t config{};
            config.gridDim          = grid;
            config.blockDim         = block;
            config.dynamicSmemBytes = kSmemBytes;
            config.stream           = stream;
            cudaLaunchAttribute cooperative{};
            cooperative.id              = cudaLaunchAttributeCooperative;
            cooperative.val.cooperative = 1;
            config.attrs                = &cooperative;
            config.numAttrs             = 1;
            CUDA_CHECK(cudaLaunchKernelEx(
                &config, bf16_gdn_gating_proj_gemm_mma_kernel<Geometry, SplitK, FullTokens, Warps>,
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const __nv_bfloat16*>(a_weight.qdata),
                static_cast<const __nv_bfloat16*>(b_weight.qdata),
                static_cast<const float*>(A_log.data), static_cast<const float*>(dt_bias.data),
                static_cast<float*>(workspace), static_cast<float*>(g.data),
                static_cast<float*>(beta.data), t));
        } else {
            bf16_gdn_gating_proj_gemm_mma_kernel<Geometry, SplitK, FullTokens, Warps>
                <<<grid, block, kSmemBytes, stream>>>(
                    static_cast<const __nv_bfloat16*>(x.data),
                    static_cast<const __nv_bfloat16*>(a_weight.qdata),
                    static_cast<const __nv_bfloat16*>(b_weight.qdata),
                    static_cast<const float*>(A_log.data), static_cast<const float*>(dt_bias.data),
                    static_cast<float*>(workspace), static_cast<float*>(g.data),
                    static_cast<float*>(beta.data), t);
        }
    };
    if (variant == Bf16GdnGatingTokenVariant::Full) {
        launch(std::true_type{});
    } else {
        launch(std::false_type{});
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void bf16_gdn_gating_proj_mma_split40_launch(const Tensor& x, const Weight& a_weight,
                                             const Weight& b_weight, const Tensor& A_log,
                                             const Tensor& dt_bias, void* workspace, Tensor& g,
                                             Tensor& beta, cudaStream_t stream) {
    require_shape(a_weight, "a_weight");
    require_shape(b_weight, "b_weight");
    if (x.ne[1] < 1 || x.ne[1] > 48) {
        throw std::invalid_argument("gdn_gating_proj: narrow split-40 admits T=1..48");
    }
    const auto launch = [&]<int Columns>() {
        launch_bf16_prefill_mma<Bf16Gdn27SmallGeometry<Columns>, 40, Columns / 8>(
            Bf16GdnGatingTokenVariant::Predicated, x, a_weight, b_weight, A_log, dt_bias, workspace,
            g, beta, stream);
    };
    if (x.ne[1] <= 8) {
        launch.template operator()<8>();
    } else if (x.ne[1] <= 16) {
        launch.template operator()<16>();
    } else if (x.ne[1] <= 32) {
        launch.template operator()<32>();
    } else {
        launch.template operator()<48>();
    }
}

void bf16_gdn_gating_proj_mma_split8_launch(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                                            const Weight& a_weight, const Weight& b_weight,
                                            const Tensor& A_log, const Tensor& dt_bias,
                                            void* workspace, Tensor& g, Tensor& beta,
                                            cudaStream_t stream) {
    launch_bf16_prefill_mma<Bf16Gdn27Geometry, 8, 8>(variant, x, a_weight, b_weight, A_log, dt_bias,
                                                     workspace, g, beta, stream);
}

void bf16_gdn_gating_proj_mma_split4_launch(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                                            const Weight& a_weight, const Weight& b_weight,
                                            const Tensor& A_log, const Tensor& dt_bias,
                                            void* workspace, Tensor& g, Tensor& beta,
                                            cudaStream_t stream) {
    launch_bf16_prefill_mma<Bf16Gdn27Geometry, 4>(variant, x, a_weight, b_weight, A_log, dt_bias,
                                                  workspace, g, beta, stream);
}

void bf16_gdn_gating_proj_mma_split2_launch(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                                            const Weight& a_weight, const Weight& b_weight,
                                            const Tensor& A_log, const Tensor& dt_bias,
                                            void* workspace, Tensor& g, Tensor& beta,
                                            cudaStream_t stream) {
    launch_bf16_prefill_mma<Bf16Gdn27Geometry, 2>(variant, x, a_weight, b_weight, A_log, dt_bias,
                                                  workspace, g, beta, stream);
}

void bf16_gdn_gating_proj_mma_unsplit_launch(Bf16GdnGatingTokenVariant variant, const Tensor& x,
                                             const Weight& a_weight, const Weight& b_weight,
                                             const Tensor& A_log, const Tensor& dt_bias, Tensor& g,
                                             Tensor& beta, cudaStream_t stream) {
    launch_bf16_prefill_mma<Bf16Gdn27Geometry, 1, 8>(variant, x, a_weight, b_weight, A_log, dt_bias,
                                                     nullptr, g, beta, stream);
}

} // namespace ninfer::ops::detail
