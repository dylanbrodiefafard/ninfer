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

// Packed snapshot T=2..16 GDN conv uses the SmallT production schedule (token-parallel
// GEMM) with FP32 projected-conv in the epilogue. Record T=2..16 is fused T=1 GEMV+conv.
// B>1 runs the C=1 kernel once per row so verify does not take a MultiBatch
// specialization; do not flatten to B*W compose (W4A4+BF16 conv).
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
    const auto output     = make_nvfp4_gdn_conv_output<ActiveTokens, Tree>(
        conv_weight, conv_states, valid_columns, initial_slot, query, key, value, z, publish,
        parent_index);
    nvfp4_small_t_kernel<Geometry, ActiveTokens, Schedule, Nvfp4IdentityEpilogue,
                         Nvfp4GdnConvOutput<ActiveTokens, Publish, Tree>,
                         Nvfp4SmallTFinalization::RowVector>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(
            Nvfp4PackedActivation<Geometry>{static_cast<const __nv_bfloat16*>(x.data)},
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), inverse, Nvfp4IdentityEpilogue{},
            output);
    CUDA_CHECK(cudaGetLastError());
}

template <int ActiveTokens>
void launch_snapshot_exact(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                           Tensor& conv_states, const Tensor& valid_columns,
                           const Tensor& initial_slot, const Tensor& snapshot_base_slot,
                           Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                           cudaStream_t stream) {
    if (x.ne[2] > 1) {
        const int batch = x.ne[2];
        for (int b = 0; b < batch; ++b) {
            const Tensor valid_b =
                valid_columns.data == nullptr ? Tensor{} : valid_columns.slice(0, b, 1);
            Tensor query_b = query.slice(2, b, 1);
            Tensor key_b   = key.slice(2, b, 1);
            Tensor value_b = value.slice(2, b, 1);
            Tensor z_b     = z.slice(2, b, 1);
            launch_snapshot_exact<ActiveTokens>(
                x.slice(2, b, 1), weight, conv_weight, conv_states, valid_b,
                initial_slot.slice(0, b, 1), snapshot_base_slot.slice(0, b, 1), query_b, key_b,
                value_b, z_b, stream);
        }
        return;
    }
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
    if (x.ne[2] > 1) {
        const int batch = x.ne[2];
        const int width = x.ne[1];
        for (int b = 0; b < batch; ++b) {
            const Tensor valid_b =
                valid_columns.data == nullptr ? Tensor{} : valid_columns.slice(0, b, 1);
            Tensor record_b = conv_record.slice(2, b, 1);
            Tensor query_b  = query.slice(2, b, 1);
            Tensor key_b    = key.slice(2, b, 1);
            Tensor value_b  = value.slice(2, b, 1);
            Tensor z_b      = z.slice(2, b, 1);
            const std::int32_t* parent_b =
                parent_index == nullptr
                    ? nullptr
                    : parent_index + static_cast<std::int64_t>(b) * width;
            launch_record_exact<ActiveTokens>(x.slice(2, b, 1), weight, conv_weight, conv_states,
                                              valid_b, initial_slot.slice(0, b, 1), record_b,
                                              query_b, key_b, value_b, z_b, parent_b, stream);
        }
        return;
    }
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
