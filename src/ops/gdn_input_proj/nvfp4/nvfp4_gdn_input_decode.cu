#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"

#include "core/device.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_output.cuh"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_gemv.cuh"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

template <class Geometry>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                  cudaStream_t stream) {
    using Schedule = typename Nvfp4LinearDecodeProductionSchedule<Geometry>::Type;
    static_assert(Geometry::kOutputRows ==
                  Nvfp4GdnInputOutput::kQkvRows + Nvfp4GdnInputOutput::kZRows);

    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    const float inverse   = 1.0F / weight.weight_scale_divisor;
    nvfp4_gemv_kernel<Geometry, Schedule>
        <<<dim3(kBlocks, x.ne[1]), Schedule::kThreads, 0, stream>>>(
            Nvfp4PackedActivation<Geometry>{static_cast<const __nv_bfloat16*>(x.data)},
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), inverse, Nvfp4IdentityEpilogue{},
            Nvfp4GdnInputOutput{static_cast<__nv_bfloat16*>(qkv.data),
                                static_cast<__nv_bfloat16*>(z.data)});
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void nvfp4_gdn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                   cudaStream_t stream) {
    switch (weight.k) {
    case Nvfp4GdnInputGeometry::kInputRows:
        launch_exact<Nvfp4GdnInputGeometry>(x, weight, qkv, z, stream);
        return;
    case Nvfp4FlashNextGdnInputGeometry::kInputRows:
        launch_exact<Nvfp4FlashNextGdnInputGeometry>(x, weight, qkv, z, stream);
        return;
    default:
        throw std::invalid_argument("nvfp4 gdn_input_proj decode: unsupported input rows");
    }
}

} // namespace ninfer::ops::detail
