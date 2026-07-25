#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_kernels.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/q4/q4_rowsplit_gemm_simt.cuh"
#include "ops/linear/q4/q4_rowsplit_gemv.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kQkRows = 4096;
constexpr std::int32_t kHidden = 5120;

using Q4GdnSimtR8C4Schedule = Q4RowSplitSimtGemmSchedule<8, 4, 16, 2, Cache::ca, 1>;
using Q4GdnSimtR8C8Schedule = Q4RowSplitSimtGemmSchedule<8, 8, 16, 2, Cache::ca, 1>;

void launch_q4_gemv(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Schedule = Q4GemvR1W8DirectSchedule;
    const dim3 grid(static_cast<unsigned>(div_up(kQkRows, Schedule::kRowsPerCta)), 1u, 1u);
    constexpr dim3 block(static_cast<unsigned>(Schedule::kThreads), 1u, 1u);
    q4_rowsplit_gemv_kernel<Schedule><<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(out.data),
        nullptr, kQkRows, kHidden);
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule, bool Full>
void launch_q4_simt(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const std::int32_t cols   = x.ne[1];
    const std::int32_t out_ld = static_cast<std::int32_t>(out.nb[1] / sizeof(__nv_bfloat16));
    const dim3 grid(static_cast<unsigned>(div_up(kQkRows, Schedule::kRowsPerCta)),
                    static_cast<unsigned>(div_up(cols, Schedule::kColsPerTile)), 1u);
    q4_rowsplit_gemm_simt_kernel<Schedule, Full><<<grid, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(out.data),
        nullptr, out_ld, 0, kQkRows, kHidden, cols, weight.padded_shape[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule>
void launch_q4_simt_route(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const bool full = (kQkRows % Schedule::kRowsPerCta) == 0 &&
                      ((kHidden / Q4RowSplitStorage::kGroupK) % Schedule::kGroupsPerStage) == 0 &&
                      (x.ne[1] % Schedule::kColsPerTile) == 0;
    if (full) {
        launch_q4_simt<Schedule, true>(x, weight, out, stream);
    } else {
        launch_q4_simt<Schedule, false>(x, weight, out, stream);
    }
}

void launch_q4(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    if (x.ne[1] == 1) {
        launch_q4_gemv(x, weight, out, stream);
        return;
    }
    if (x.ne[1] <= 4) {
        launch_q4_simt_route<Q4GdnSimtR8C4Schedule>(x, weight, out, stream);
        return;
    }
    if (x.ne[1] <= 16) {
        launch_q4_simt_route<Q4GdnSimtR8C8Schedule>(x, weight, out, stream);
        return;
    }
    throw std::invalid_argument("Q4/Q5 GDN independent launch requires T in [1,16]");
}

} // namespace

void q4_q5_gdn_input_independent_launch(Q5Plan value_plan, const Tensor& x, const Weight& qk_weight,
                                        const Weight& v_weight, Tensor& qk, Tensor& value,
                                        cudaStream_t stream) {
    launch_q4(x, qk_weight, qk, stream);
    q5_rowsplit_launch_fixed_pitched(value_plan, x, v_weight, value, stream);
}

} // namespace ninfer::ops::detail
