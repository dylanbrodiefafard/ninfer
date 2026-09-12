#include "ops/linear/fp8/fp8_launch.h"

#include "core/device.h"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_gemv.cuh"
#include "ops/linear/fp8/fp8_output.cuh"

#include <cuda_bf16.h>

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

template <class Geometry>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Schedule = typename Fp8LinearDecodeProductionSchedule<Geometry>::Type;
    if (x.ne[0] != Geometry::kInputRows || x.ne[1] != 1 || out.ne[0] != Geometry::kOutputRows ||
        out.ne[1] != 1 || weight.n != Geometry::kOutputRows || weight.k != Geometry::kInputRows) {
        throw std::invalid_argument("fp8 linear decode: invalid exact problem");
    }

    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    const Fp8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), Geometry::kOutputRows};
    fp8_gemv_kernel<Geometry, Schedule><<<kBlocks, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const __nv_bfloat16*>(weight.scales), output);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_fp8_decode(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    switch (resolve_fp8_problem(weight.n, weight.k)) {
    case Fp8Problem::AttnInput:
        launch_exact<Fp8AttnInputGeometry>(x, weight, out, stream);
        return;
    case Fp8Problem::GdnInput:
        launch_exact<Fp8GdnInputGeometry>(x, weight, out, stream);
        return;
    case Fp8Problem::MlpGateUp:
        launch_exact<Fp8MlpGateUpGeometry>(x, weight, out, stream);
        return;
    case Fp8Problem::Vocabulary:
        break;
    case Fp8Problem::Residual6144:
        launch_exact<Fp8Residual6144Geometry>(x, weight, out, stream);
        return;
    case Fp8Problem::Residual17408:
        launch_exact<Fp8Residual17408Geometry>(x, weight, out, stream);
        return;
    case Fp8Problem::Rows10240K2560:
        launch_exact<Fp8Rows10240K2560Geometry>(x, weight, out, stream);
        return;
    case Fp8Problem::Rows6144K2560:
        launch_exact<Fp8Rows6144K2560Geometry>(x, weight, out, stream);
        return;
    case Fp8Problem::Rows12288K2560:
        launch_exact<Fp8Rows12288K2560Geometry>(x, weight, out, stream);
        return;
    case Fp8Problem::Rows512K2560:
        launch_exact<Fp8Rows512K2560Geometry>(x, weight, out, stream);
        return;
    case Fp8Problem::Rows2560K6144:
        launch_exact<Fp8Rows2560K6144Geometry>(x, weight, out, stream);
        return;
    case Fp8Problem::Rows640K2560:
        launch_exact<Fp8Rows640K2560Geometry>(x, weight, out, stream);
        return;
    case Fp8Problem::Rows1280K2560:
        launch_exact<Fp8Rows1280K2560Geometry>(x, weight, out, stream);
        return;
    case Fp8Problem::Rows2560K640:
        launch_exact<Fp8Rows2560K640Geometry>(x, weight, out, stream);
        return;
    case Fp8Problem::Rows320K10240:
        launch_exact<Fp8Rows320K10240Geometry>(x, weight, out, stream);
        return;
    case Fp8Problem::Rows10240K320:
        launch_exact<Fp8Rows10240K320Geometry>(x, weight, out, stream);
        return;
    case Fp8Problem::Rows2560K2560:
        launch_exact<Fp8Rows2560K2560Geometry>(x, weight, out, stream);
        return;
    case Fp8Problem::Vocabulary2560:
        break;
    }
    throw std::logic_error("FP8 vocabulary decode uses its A16 MMA route");
}

} // namespace ninfer::ops::detail
