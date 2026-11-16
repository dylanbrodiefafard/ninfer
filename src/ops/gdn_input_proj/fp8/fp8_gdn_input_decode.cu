#include "ops/gdn_input_proj/fp8/fp8_gdn_input_plan.h"

#include "core/device.h"
#include "ops/gdn_input_proj/fp8/fp8_gdn_input_output.cuh"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_gemv.cuh"

#include <cuda_bf16.h>

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

template <class Geometry>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                  cudaStream_t stream) {
    using Schedule = typename Fp8LinearDecodeProductionSchedule<Geometry>::Type;
    static_assert(Geometry::kOutputRows == Fp8GdnInputOutput::kRows);
    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    const Fp8GdnInputOutput output{static_cast<__nv_bfloat16*>(qkv.data),
                                   static_cast<__nv_bfloat16*>(z.data)};
    fp8_gemv_kernel<Geometry, Schedule><<<kBlocks, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const __nv_bfloat16*>(weight.scales), output);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void fp8_gdn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                 cudaStream_t stream) {
    switch (weight.k) {
    case Fp8GdnInputGeometry::kInputRows:
        launch_exact<Fp8GdnInputGeometry>(x, weight, qkv, z, stream);
        return;
    case Fp8FlashNextGdnInputGeometry::kInputRows:
        launch_exact<Fp8FlashNextGdnInputGeometry>(x, weight, qkv, z, stream);
        return;
    default:
        throw std::invalid_argument("fp8 gdn_input_proj decode: unsupported input rows");
    }
}

} // namespace ninfer::ops::detail
