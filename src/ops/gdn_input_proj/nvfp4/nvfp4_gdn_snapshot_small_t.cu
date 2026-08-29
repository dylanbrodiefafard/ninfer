#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.h"

#include "core/device.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_conv_output.cuh"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_small_t.cuh"

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace ninfer::ops::detail {
namespace {

using Launch       = void (*)(const Tensor&, const Weight&, const Tensor&, Tensor&, const Tensor&,
                        const Tensor&, const Tensor&, Tensor&, Tensor&, Tensor&, Tensor&,
                        cudaStream_t);
using RecordLaunch = void (*)(const Tensor&, const Weight&, const Tensor&, const Tensor&,
                              const Tensor&, const Tensor&, Tensor&, Tensor&, Tensor&, Tensor&,
                              Tensor&, const std::int32_t*, cudaStream_t);

// Packed T=2..16 GDN conv uses the SmallT production schedule (token-parallel GEMM) with
// FP32 projected-conv in the epilogue. B>1 is one launch (grid.x = B, T=W per row), not a
// host loop and not flatten-to-B*W compose: the latter's W4A4+BF16 conv flips greedy col 0.
template <int ActiveTokens, bool Tree, class Publish>
void launch_exact(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                  const Tensor& conv_states, const Tensor& valid_columns,
                  const Tensor& initial_slot, Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                  Publish publish, cudaStream_t stream,
                  const std::int32_t* parent_index = nullptr) {
    using Geometry = Nvfp4GdnInputGeometry;
    using Schedule = typename Nvfp4LinearSmallTProductionSchedule<Geometry, ActiveTokens>::Type;
    static_assert(Schedule::kTokenTile == ActiveTokens);
    static_assert(Schedule::kWarpsPerRow == 1);

    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    const float inverse   = 1.0F / weight.weight_scale_divisor;
    const int batch       = x.ne[2] > 1 ? x.ne[2] : 1;
    const auto output     = make_nvfp4_gdn_conv_output<ActiveTokens, Tree>(
        conv_weight, conv_states, valid_columns, initial_slot, query, key, value, z, publish,
        parent_index);
    if (batch > 1) {
        nvfp4_small_t_kernel<Geometry, ActiveTokens, Schedule, Nvfp4IdentityEpilogue,
                             Nvfp4GdnConvOutput<ActiveTokens, Publish, Tree>,
                             Nvfp4SmallTFinalization::RowVector,
                             Nvfp4BatchedPackedActivation<Geometry>>
            <<<dim3(batch, kBlocks), Schedule::kThreads, 0, stream>>>(
                Nvfp4BatchedPackedActivation<Geometry>{static_cast<const __nv_bfloat16*>(x.data),
                                                       ActiveTokens},
                static_cast<const std::uint8_t*>(weight.qdata),
                static_cast<const std::uint8_t*>(weight.scales), inverse, Nvfp4IdentityEpilogue{},
                output);
    } else {
        nvfp4_small_t_kernel<Geometry, ActiveTokens, Schedule, Nvfp4IdentityEpilogue,
                             Nvfp4GdnConvOutput<ActiveTokens, Publish, Tree>,
                             Nvfp4SmallTFinalization::RowVector>
            <<<kBlocks, Schedule::kThreads, 0, stream>>>(
                Nvfp4PackedActivation<Geometry>{static_cast<const __nv_bfloat16*>(x.data)},
                static_cast<const std::uint8_t*>(weight.qdata),
                static_cast<const std::uint8_t*>(weight.scales), inverse, Nvfp4IdentityEpilogue{},
                output);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <int ActiveTokens>
void launch_snapshot_exact(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                           Tensor& conv_states, const Tensor& valid_columns,
                           const Tensor& initial_slot, const Tensor& snapshot_base_slot,
                           Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                           cudaStream_t stream) {
    launch_exact<ActiveTokens, false>(
        x, weight, conv_weight, conv_states, valid_columns, initial_slot, query, key, value, z,
        SnapshotHistoryPublish{static_cast<__nv_bfloat16*>(conv_states.data),
                               static_cast<const std::int32_t*>(snapshot_base_slot.data),
                               kNvfp4GdnChannels},
        stream);
}

template <int ActiveTokens>
void launch_record_exact(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                         const Tensor& conv_states, const Tensor& valid_columns,
                         const Tensor& initial_slot, Tensor& conv_record, Tensor& query,
                         Tensor& key, Tensor& value, Tensor& z, const std::int32_t* parent_index,
                         cudaStream_t stream) {
    const RecordColumnPublish publish{static_cast<__nv_bfloat16*>(conv_record.data),
                                      kNvfp4GdnChannels, ActiveTokens};
    if (parent_index == nullptr) {
        launch_exact<ActiveTokens, false>(x, weight, conv_weight, conv_states, valid_columns,
                                          initial_slot, query, key, value, z, publish, stream);
        return;
    }
    launch_exact<ActiveTokens, true>(x, weight, conv_weight, conv_states, valid_columns,
                                     initial_slot, query, key, value, z, publish, stream,
                                     parent_index);
}

template <std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Launch, sizeof...(Offsets)>{
        &launch_snapshot_exact<kNvfp4FirstSmallT + static_cast<int>(Offsets)>...};
}

template <std::size_t... Offsets>
constexpr auto make_record_launchers(std::index_sequence<Offsets...>) {
    return std::array<RecordLaunch, sizeof...(Offsets)>{
        &launch_record_exact<kNvfp4FirstSmallT + static_cast<int>(Offsets)>...};
}

constexpr auto kLaunchers = make_launchers(std::make_index_sequence<16 - kNvfp4FirstSmallT + 1>{});
constexpr auto kRecordLaunchers =
    make_record_launchers(std::make_index_sequence<16 - kNvfp4FirstSmallT + 1>{});

} // namespace

void nvfp4_gdn_snapshot_small_t_launch(const Tensor& x, const Weight& weight,
                                       const Tensor& conv_weight, Tensor& conv_states,
                                       const Tensor& valid_columns, const Tensor& initial_slot,
                                       const Tensor& snapshot_base_slot, Tensor& query, Tensor& key,
                                       Tensor& value, Tensor& z, cudaStream_t stream) {
    const std::size_t index = static_cast<std::size_t>(x.ne[1] - kNvfp4FirstSmallT);
    kLaunchers[index](x, weight, conv_weight, conv_states, valid_columns, initial_slot,
                      snapshot_base_slot, query, key, value, z, stream);
}

void nvfp4_gdn_record_small_t_launch(const Tensor& x, const Weight& weight,
                                     const Tensor& conv_weight, const Tensor& conv_states,
                                     const Tensor& valid_columns, const Tensor& initial_slot,
                                     Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value,
                                     Tensor& z, cudaStream_t stream,
                                     const std::int32_t* parent_index) {
    const std::size_t index = static_cast<std::size_t>(x.ne[1] - kNvfp4FirstSmallT);
    kRecordLaunchers[index](x, weight, conv_weight, conv_states, valid_columns, initial_slot,
                            conv_record, query, key, value, z, parent_index, stream);
}

} // namespace ninfer::ops::detail
