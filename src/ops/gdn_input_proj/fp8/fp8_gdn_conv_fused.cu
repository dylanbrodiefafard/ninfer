#include "ops/gdn_input_proj/fp8/fp8_gdn_conv_plan.h"

#include "core/device.h"
#include "ops/gdn_input_proj/gdn_conv_output.cuh"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_gemv.cuh"
#include "ops/linear/fp8/fp8_small_t.cuh"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

using Geometry = Fp8GdnInputGeometry;

using RecordLaunch = void (*)(const Tensor&, const Weight&, const Tensor&, const Tensor&,
                              const Tensor&, const Tensor&, Tensor&, Tensor&, Tensor&, Tensor&,
                              Tensor&, cudaStream_t, const std::int32_t*);

template <int ActiveTokens, bool Tree = false, class Publish>
void launch_small_t(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                    const Tensor& conv_states, const Tensor& valid_columns,
                    const Tensor& initial_slot, Tensor& query, Tensor& key, Tensor& value,
                    Tensor& z, Publish publish, cudaStream_t stream,
                    const std::int32_t* parent_index = nullptr) {
    using Schedule = typename Fp8LinearSmallTProductionSchedule<Geometry, ActiveTokens>::Type;
    static_assert(Schedule::kTokenTile == ActiveTokens);
    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    using Output          = GdnConvOutput<ActiveTokens, Publish, Tree>;
    fp8_small_t_kernel<Geometry, ActiveTokens, Schedule, Output, Fp8IdentityEpilogue,
                       Fp8GemvIdentityRows, false, Fp8SmallTFinalization::RowVector>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales),
            make_gdn_conv_output<ActiveTokens, Publish, Tree>(
                conv_weight, conv_states, valid_columns, initial_slot, query, key, value, z,
                publish, parent_index));
    CUDA_CHECK(cudaGetLastError());
}

template <int ActiveTokens, bool Tree>
void launch_record_small_t(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                           const Tensor& conv_states, const Tensor& valid_columns,
                           const Tensor& initial_slot, Tensor& conv_record, Tensor& query,
                           Tensor& key, Tensor& value, Tensor& z, cudaStream_t stream,
                           const std::int32_t* parent_index) {
    launch_small_t<ActiveTokens, Tree>(
        x, weight, conv_weight, conv_states, valid_columns, initial_slot, query, key, value, z,
        RecordColumnPublish{static_cast<__nv_bfloat16*>(conv_record.data), kGdnChannels,
                            ActiveTokens},
        stream, parent_index);
}

// The launcher of a fused record width, nullptr elsewhere, so only those widths instantiate.
template <int ActiveTokens, bool Tree>
constexpr RecordLaunch fused_record_launcher() {
    if constexpr (fp8_gdn_fused_record_width(ActiveTokens, Tree)) {
        return &launch_record_small_t<ActiveTokens, Tree>;
    } else {
        return nullptr;
    }
}

template <bool Tree, std::size_t... Offsets>
constexpr auto make_record_launchers(std::index_sequence<Offsets...>) {
    return std::array<RecordLaunch, sizeof...(Offsets)>{
        fused_record_launcher<kFp8FirstSmallT + static_cast<int>(Offsets), Tree>()...};
}

constexpr auto kTreeRecordLaunchers = make_record_launchers<true>(
    std::make_index_sequence<kFp8LinearSmallTMax<Geometry> - kFp8FirstSmallT + 1>{});
constexpr auto kRecordLaunchers = make_record_launchers<false>(
    std::make_index_sequence<kFp8LinearSmallTMax<Geometry> - kFp8FirstSmallT + 1>{});

} // namespace

void fp8_gdn_snapshot_decode_launch(const Tensor& x, const Weight& weight,
                                    const Tensor& conv_weight, Tensor& conv_states,
                                    const Tensor& valid_columns, const Tensor& initial_slot,
                                    const Tensor& snapshot_base_slot, Tensor& query, Tensor& key,
                                    Tensor& value, Tensor& z, cudaStream_t stream) {
    using Schedule        = typename Fp8LinearDecodeProductionSchedule<Geometry>::Type;
    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    fp8_gemv_kernel<Geometry, Schedule><<<kBlocks, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const __nv_bfloat16*>(weight.scales),
        make_gdn_conv_output<1>(
            conv_weight, conv_states, valid_columns, initial_slot, query, key, value, z,
            SnapshotHistoryPublish{static_cast<__nv_bfloat16*>(conv_states.data),
                                   static_cast<const std::int32_t*>(snapshot_base_slot.data),
                                   kGdnChannels}));
    CUDA_CHECK(cudaGetLastError());
}

void fp8_gdn_record_fused_launch(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                                 const Tensor& conv_states, const Tensor& valid_columns,
                                 const Tensor& initial_slot, Tensor& conv_record, Tensor& query,
                                 Tensor& key, Tensor& value, Tensor& z, cudaStream_t stream,
                                 const Tensor* parent_index) {
    const auto* parents =
        parent_index == nullptr ? nullptr : static_cast<const std::int32_t*>(parent_index->data);
    if (x.ne[2] != 1 || !fp8_gdn_fused_record_width(x.ne[1], parents != nullptr)) {
        throw std::invalid_argument("fp8 GDN record fused: unregistered B/W/tree shape");
    }
    const auto& launchers = parents == nullptr ? kRecordLaunchers : kTreeRecordLaunchers;
    launchers[static_cast<std::size_t>(x.ne[1] - kFp8FirstSmallT)](
        x, weight, conv_weight, conv_states, valid_columns, initial_slot, conv_record, query, key,
        value, z, stream, parents);
}

} // namespace ninfer::ops::detail
