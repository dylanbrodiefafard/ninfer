#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.h"

#include "core/device.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_conv_output.cuh"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_gemv.cuh"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {

void nvfp4_gdn_snapshot_decode_launch(const Tensor& x, const Weight& weight,
                                      const Tensor& conv_weight, Tensor& conv_states,
                                      const Tensor& valid_columns, const Tensor& initial_slot,
                                      const Tensor& snapshot_base_slot, Tensor& query, Tensor& key,
                                      Tensor& value, Tensor& z, cudaStream_t stream) {
    using Geometry        = Nvfp4GdnInputGeometry;
    using Schedule        = typename Nvfp4LinearDecodeProductionSchedule<Geometry>::Type;
    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    const float inverse   = 1.0F / weight.weight_scale_divisor;
    const SnapshotHistoryPublish publish{static_cast<__nv_bfloat16*>(conv_states.data),
                                         static_cast<const std::int32_t*>(snapshot_base_slot.data),
                                         kNvfp4GdnChannels};
    nvfp4_gemv_kernel<Geometry, Schedule><<<kBlocks, Schedule::kThreads, 0, stream>>>(
        Nvfp4PackedActivation<Geometry>{static_cast<const __nv_bfloat16*>(x.data)},
        static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), inverse, Nvfp4IdentityEpilogue{},
        make_nvfp4_gdn_conv_output<1>(conv_weight, conv_states, valid_columns, initial_slot, query,
                                      key, value, z, publish));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
