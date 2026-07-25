#include "ops/linear_swiglu/q4/q4_linear_swiglu_kernels.h"

#include "ops/linear_swiglu/q4/q4_linear_swiglu_gemm_mma.cuh"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/token_slices.h"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

using GateUpCfg = GemmCfg<64, 128, 64, 64, 16, 2, 1, false, true, true>;

template <bool Full>
void launch_folded(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    constexpr int PM = GateUpCfg::BM / 2;
    const int t      = x.ne[1];
    const dim3 grid(static_cast<unsigned>(div_up(out.ne[0], PM)),
                    static_cast<unsigned>(div_up(t, GateUpCfg::BN)));
    if constexpr (Full) {
        q4_linear_swiglu_mma_split_half_pair_kernel<GateUpCfg, true>
            <<<grid, GateUpCfg::THREADS, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const std::uint8_t*>(weight.qdata),
                static_cast<const std::uint8_t*>(weight.scales),
                static_cast<__nv_bfloat16*>(out.data), out.ne[0], x.ne[0], t,
                weight.padded_shape[1]);
    } else {
        q4_linear_swiglu_mma_split_half_pair_kernel<GateUpCfg, false>
            <<<grid, GateUpCfg::THREADS, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const std::uint8_t*>(weight.qdata),
                static_cast<const std::uint8_t*>(weight.scales),
                static_cast<__nv_bfloat16*>(out.data), out.ne[0], x.ne[0], t,
                weight.padded_shape[1]);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void q4_linear_swiglu_mma_split_half_pair_r32_c128_launch(const Tensor& x, const Weight& weight,
                                                          Tensor& out, cudaStream_t stream) {
    constexpr std::int32_t kTileCols = 128;
    const bool full                  = (x.ne[1] % kTileCols) == 0;
    for_each_token_slice(x.ne[1], kTileCols, [&](std::int32_t offset, std::int32_t count) {
        const Tensor x_slice = x.slice(1, offset, count);
        Tensor out_slice     = out.slice(1, offset, count);
        if (full) {
            launch_folded<true>(x_slice, weight, out_slice, stream);
        } else {
            launch_folded<false>(x_slice, weight, out_slice, stream);
        }
    });
}

} // namespace ninfer::ops::detail
