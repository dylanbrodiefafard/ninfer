#include "ops/linear/bf16/bf16_launch.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/linear/bf16/bf16_config.h"
#include "ops/linear/bf16/bf16_gemm_mma_config.h"

#include <cuda_bf16.h>

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

template <class Geometry, class Schedule, bool FullTokens>
void launch_variant(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    constexpr int tiles_m = (Geometry::kOutputRows + Schedule::kBlockRows - 1) / Schedule::kBlockRows;
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

template <class Geometry>
void launch_geometry(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Schedule = Bf16MmaProductionSchedule<Geometry>;
    if ((x.ne[1] % Schedule::kBlockCols) == 0) {
        launch_variant<Geometry, Schedule, true>(x, weight, out, stream);
    } else {
        launch_variant<Geometry, Schedule, false>(x, weight, out, stream);
    }
}

} // namespace

void launch_bf16_mma(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    if (weight.n == 1152 && weight.k == 1536) {
        launch_geometry<Bf16GemvGeometry<1152,1536>>(x,weight,out,stream); return;
    }
    if (weight.n == 3456 && weight.k == 1152) {
        launch_geometry<Bf16GemvGeometry<3456,1152>>(x,weight,out,stream); return;
    }
    if (weight.n == 1152 && weight.k == 1152) {
        launch_geometry<Bf16GemvGeometry<1152,1152>>(x,weight,out,stream); return;
    }
    if (weight.n == 4304 && weight.k == 1152) {
        launch_geometry<Bf16GemvGeometry<4304,1152>>(x,weight,out,stream); return;
    }
    if (weight.n == 1152 && weight.k == 4304) {
        launch_geometry<Bf16GemvGeometry<1152,4304>>(x,weight,out,stream); return;
    }
    if (weight.n == 4608 && weight.k == 4608) {
        launch_geometry<Bf16GemvGeometry<4608,4608>>(x,weight,out,stream);return;
    }
    if (weight.n == 2560 && weight.k == 4608) {
        launch_geometry<Bf16GemvGeometry<2560,4608>>(x,weight,out,stream);return;
    }
    if (weight.n == 248320 && weight.k == 2560) {
        launch_geometry<Bf16GemvGeometry<248320, 2560>>(x, weight, out, stream);
        return;
    }
    if (weight.n == 14336 && weight.k == 5120) {
        launch_geometry<Bf16GemvGeometry<14336, 5120>>(x, weight, out, stream);
        return;
    }
    if (weight.n == 5120 && weight.k == 6144) {
        launch_geometry<Bf16GemvGeometry<5120, 6144>>(x, weight, out, stream);
        return;
    }
    if (weight.n == 10240 && weight.k == 2560) {
        launch_geometry<Bf16GemvGeometry<10240, 2560>>(x, weight, out, stream);
        return;
    }
    if (weight.n == 6144 && weight.k == 2560) {
        launch_geometry<Bf16GemvGeometry<6144, 2560>>(x, weight, out, stream);
        return;
    }
    if (weight.n == 12288 && weight.k == 2560) {
        launch_geometry<Bf16GemvGeometry<12288, 2560>>(x, weight, out, stream);
        return;
    }
    if (weight.n == 512 && weight.k == 2560) {
        launch_geometry<Bf16GemvGeometry<512, 2560>>(x, weight, out, stream);
        return;
    }
    if (weight.n == 640 && weight.k == 2560) {
        launch_geometry<Bf16GemvGeometry<640, 2560>>(x, weight, out, stream);
        return;
    }
    if (weight.n == 2560 && weight.k == 2560) {
        launch_geometry<Bf16GemvGeometry<2560, 2560>>(x, weight, out, stream);
        return;
    }
    if (weight.n == 2560 && weight.k == 6144) {
        launch_geometry<Bf16GemvGeometry<2560, 6144>>(x, weight, out, stream);
        return;
    }
    if (weight.n == 320 && weight.k == 10240) {
        launch_geometry<Bf16GemvGeometry<320, 10240>>(x, weight, out, stream);
        return;
    }
    if (weight.n == 2560 && weight.k == 640) {
        launch_geometry<Bf16GemvGeometry<2560, 640>>(x, weight, out, stream);
        return;
    }
    if (weight.n == 10240 && weight.k == 320) {
        launch_geometry<Bf16GemvGeometry<10240, 320>>(x, weight, out, stream);
        return;
    }
    throw std::invalid_argument("bf16 linear MMA: unsupported exact problem");
}

} // namespace ninfer::ops::detail
