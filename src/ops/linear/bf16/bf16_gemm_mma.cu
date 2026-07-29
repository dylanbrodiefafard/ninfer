#include "ops/linear/bf16/bf16_launch.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/bf16/bf16_config.h"
#include "ops/linear/bf16/bf16_gemm_mma_config.h"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {
namespace {

template <class Schedule, bool FullTokens>
void launch_variant(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Geometry = Bf16LinearControlGeometry;
    static_assert((Geometry::kOutputRows % Schedule::kBlockRows) == 0);
    static_assert((Geometry::kInputRows % Schedule::kBlockK) == 0);

    constexpr int tiles_m = Geometry::kOutputRows / Schedule::kBlockRows;
    const int tiles_n     = div_up(x.ne[1], Schedule::kBlockCols);
    const int blocks      = tiles_m * tiles_n;
    const Bf16MmaContiguousOutput output{static_cast<__nv_bfloat16*>(out.data),
                                         Geometry::kOutputRows};

    if constexpr (Schedule::kSharedBytes > 48 * 1024) {
        static const cudaError_t attr = cudaFuncSetAttribute(
            bf16_gemm_mma_kernel<Geometry, Schedule, FullTokens, Bf16MmaContiguousOutput>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, Schedule::kSharedBytes);
        CUDA_CHECK(attr);
    }
    bf16_gemm_mma_kernel<Geometry, Schedule, FullTokens>
        <<<blocks, Schedule::kThreads, Schedule::kSharedBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(weight.qdata), output, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule>
void launch_schedule(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    if ((x.ne[1] % Schedule::kBlockCols) == 0) {
        launch_variant<Schedule, true>(x, weight, out, stream);
    } else {
        launch_variant<Schedule, false>(x, weight, out, stream);
    }
}

} // namespace

void launch_bf16_mma(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    launch_schedule<Bf16MmaProductionSchedule>(x, weight, out, stream);
}

} // namespace ninfer::ops::detail
