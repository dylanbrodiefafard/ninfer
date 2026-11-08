#include "ops/linear/w8/w8_rowsplit_gemm_mma.cuh"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/linear/w8/w8_launch.h"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

template <class Schedule, bool Full>
void launch_slice(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const std::int32_t rows     = w.n;
    const std::int32_t k        = w.k;
    const std::int32_t cols     = x.ne[1];
    const std::int32_t padded_k = w.padded_shape[1];
    const dim3 grid(static_cast<unsigned>(div_up(rows, Schedule::BM)),
                    static_cast<unsigned>(div_up(cols, Schedule::BN)), 1u);
    const W8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), rows};
    w8_rowsplit_gemm_mma_kernel<Schedule, Full><<<grid, Schedule::THREADS, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<const std::uint8_t*>(w.scales), output, rows, k, cols, padded_k);
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule>
void launch_route(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const bool full = (w.n % Schedule::BM) == 0 && (x.ne[1] % Schedule::BN) == 0 &&
                      w.k == w.padded_shape[1] && (w.k % Schedule::BK) == 0;
    for_each_token_slice(x.ne[1], Schedule::BN, [&](std::int32_t offset, std::int32_t count) {
        const Tensor x_slice = x.slice(1, offset, count);
        Tensor out_slice     = out.slice(1, offset, count);
        if (full) {
            launch_slice<Schedule, true>(x_slice, w, out_slice, stream);
        } else {
            launch_slice<Schedule, false>(x_slice, w, out_slice, stream);
        }
    });
}

using MmaR32C64          = W8RowSplitMmaGemmSchedule<32, 64, 32, 16, 3>;
using MmaR32C128         = W8RowSplitMmaGemmSchedule<32, 128, 32, 16, 2>;
using MmaR64C128         = W8RowSplitMmaGemmSchedule<64, 128, 64, 16, 2, 2>;
using MmaR64x16C48K128A1 = W8RowSplitMmaGemmSchedule<64, 48, 16, 24, 2, 2, 128, 1>;

} // namespace

#define NINFER_W8_MMA_LAUNCHER(Name, Schedule)                                                     \
    void Name(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {                \
        launch_route<Schedule>(x, w, out, stream);                                                 \
    }

NINFER_W8_MMA_LAUNCHER(launch_w8_mma_r32_c64, MmaR32C64)
NINFER_W8_MMA_LAUNCHER(launch_w8_mma_r32_c128, MmaR32C128)
NINFER_W8_MMA_LAUNCHER(launch_w8_mma_r64_c128, MmaR64C128)
NINFER_W8_MMA_LAUNCHER(launch_w8_mma_r64x16_c48_k128_a1, MmaR64x16C48K128A1)

#undef NINFER_W8_MMA_LAUNCHER

} // namespace ninfer::ops::detail
