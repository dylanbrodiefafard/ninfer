#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"

#include "core/device.h"
#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_output.cuh"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_gemv.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

template <class Geometry, class Output>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate, Tensor& k,
                  Tensor& v, cudaStream_t stream) {
    using Schedule = typename Nvfp4LinearDecodeProductionSchedule<Geometry>::Type;
    static_assert(Geometry::kOutputRows == Output::kParentRows);

    const Output output{
        static_cast<__nv_bfloat16*>(q.data),
        static_cast<__nv_bfloat16*>(k.data),
        static_cast<__nv_bfloat16*>(gate.data),
        static_cast<__nv_bfloat16*>(v.data),
    };
    constexpr int kBlocks              = Geometry::kOutputRows / Schedule::kRowsPerCta;
    const float inverse_weight_divisor = 1.0F / weight.weight_scale_divisor;
    nvfp4_gemv_kernel<Geometry, Schedule>
        <<<dim3(kBlocks, x.ne[1]), Schedule::kThreads, 0, stream>>>(
            Nvfp4PackedActivation<Geometry>{static_cast<const __nv_bfloat16*>(x.data)},
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), inverse_weight_divisor,
            Nvfp4IdentityEpilogue{}, output);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void nvfp4_attn_input_decode_launch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                    Tensor& k, Tensor& v, cudaStream_t stream) {
    switch (weight.k) {
    case Nvfp4AttnInputGeometry::kInputRows:
        launch_exact<Nvfp4AttnInputGeometry, Nvfp4AttentionInput5120Output>(x, weight, q, gate, k,
                                                                            v, stream);
        return;
    case Nvfp4FlashNextAttnInputGeometry::kInputRows:
        launch_exact<Nvfp4FlashNextAttnInputGeometry, Nvfp4AttentionInput2560Output>(
            x, weight, q, gate, k, v, stream);
        return;
    default:
        throw std::invalid_argument("nvfp4 attn_input_proj decode: unsupported input rows");
    }
}

} // namespace ninfer::ops::detail
