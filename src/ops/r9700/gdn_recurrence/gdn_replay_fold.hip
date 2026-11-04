// Native wave32 implementation of the all-layer ReplaySSM fold boundary.
#include "ops/r9700/gdn_recurrence/gdn_replay_fold_internal.h"
#include "ops/r9700/gdn_recurrence/gdn_replay_fold_impl.h"
#include "ops/r9700/persistent/persistent_cta.h"

#include "core/device.h"

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace ninfer::ops::detail::gated_delta_net {
namespace {

constexpr int kStateDim       = fold::kStateDim;
constexpr int kWaveSize       = fold::kWaveSize;
constexpr int kWavesPerBlock  = fold::kItemThreads / fold::kWaveSize;
constexpr int kValuesPerBlock = fold::kValuesPerItem;

static_assert(sizeof(GdnReplayFoldKernelRow) == 80);
static_assert(alignof(GdnReplayFoldKernelRow) == 8);
static_assert(sizeof(GdnReplayFoldKernelRows) == 80U * kMaximumConcurrency);
static_assert(alignof(GdnReplayFoldKernelRows) == 16);
static_assert(std::is_trivially_copyable_v<GdnReplayFoldKernelRows>);

template <int Layers, int QkHeads, int ValueHeads, int ConvChannels>
struct FoldGeometry {
    static constexpr int kLayers       = Layers;
    static constexpr int kQkHeads      = QkHeads;
    static constexpr int kValueHeads   = ValueHeads;
    static constexpr int kConvChannels = ConvChannels;
    static_assert(QkHeads == fold::kQkHeads && ValueHeads == fold::kValueHeads &&
                  ConvChannels == fold::kConvChannels);
};

using FoldGeometry48x48 = FoldGeometry<48, 16, 48, 10240>;

struct FoldAccess {
    fold::Planes planes;
    float* recurrent_layer0;
    hip_bfloat16* conv_layer0;
    std::int64_t recurrent_layer_stride;
    std::int64_t conv_layer_stride;
    GdnReplayFoldKernelRows rows;
};

static_assert(sizeof(FoldAccess) <= 4096U, "replay fold arguments must fit the kernarg segment");

// One CTA per 128-channel block. Small immutable record staging and convolution publication
// precede projection; the large recurrent state stays untouched until the record consumer.
__global__ __launch_bounds__(128) void replay_prepare_layer_kernel(fold::LayerArgs args,
                                                                   fold::Planes staged) {
    constexpr int blocks_per_row = fold::kConvChannels / fold::kStateDim;
    const int batch              = static_cast<int>(blockIdx.x) / blocks_per_row;
    const int head               = static_cast<int>(blockIdx.x) % blocks_per_row;
    const auto source            = args.rows[batch];
    if (source.columns <= 0) return;
    const fold::Row row{
        .record_row = source.record_row, .slot = source.slot, .columns = source.columns};
    const int tid = static_cast<int>(threadIdx.x);
    // publish_conv_history maps (value_head,state_tile) to this channel block.
    fold::publish_conv_history(args.planes, args.layer, row, args.conv_layer,
                               head / fold::kStateTiles, head % fold::kStateTiles, tid);
    if (head >= fold::kValueHeads) return;
    for (int token = 0; token < source.columns; ++token) {
        const auto input     = fold::record_column(args.planes, args.layer, row, token);
        const auto output    = static_cast<std::int64_t>(batch) * staged.width + token;
        const auto in_value  = (input * fold::kValueHeads + head) * fold::kStateDim + tid;
        const auto out_value = (output * fold::kValueHeads + head) * fold::kStateDim + tid;
        const_cast<hip_bfloat16*>(staged.value)[out_value] = args.planes.value[in_value];
        if (head % 3 == 0) {
            const auto in_key  = (input * fold::kQkHeads + head / 3) * fold::kStateDim + tid;
            const auto out_key = (output * fold::kQkHeads + head / 3) * fold::kStateDim + tid;
            const_cast<hip_bfloat16*>(staged.key)[out_key] = args.planes.key[in_key];
        }
        if (tid < 2)
            const_cast<float*>(staged.gate)[(output * fold::kValueHeads + head) * 2 + tid] =
                args.planes.gate[(input * fold::kValueHeads + head) * 2 + tid];
    }
}

template <class Geometry>
__global__ __launch_bounds__(kWaveSize* kWavesPerBlock,
                             2) void replay_fold_kernel(FoldAccess access) {
    const std::int32_t batch             = static_cast<std::int32_t>(blockIdx.y);
    const std::int32_t layer_tile        = static_cast<std::int32_t>(blockIdx.z);
    const std::int32_t layer             = layer_tile / fold::kStateTiles;
    const std::int32_t state_tile        = layer_tile % fold::kStateTiles;
    const std::int32_t value_head        = static_cast<std::int32_t>(blockIdx.x);
    const GdnReplayFoldKernelRow& source = access.rows.row[batch];
    const std::int32_t columns =
        source.path_length >= 0 ? source.path_length : source.commit_columns;
    if (columns == 0) { return; }
    const fold::Row row{.record_row  = batch,
                        .slot        = source.linear_state_slot,
                        .columns     = columns,
                        .path_length = source.path_length,
                        .path        = source.path};
    __shared__ float key[kStateDim];
    __shared__ float reduction[kStateDim];
    fold::fold_item(
        access.planes, layer, row,
        access.recurrent_layer0 + static_cast<std::int64_t>(layer) * access.recurrent_layer_stride,
        access.conv_layer0 + static_cast<std::int64_t>(layer) * access.conv_layer_stride,
        value_head, state_tile,
        static_cast<int>(threadIdx.y) * kWaveSize + static_cast<int>(threadIdx.x), true, key,
        reduction, [] { __syncthreads(); });
}

// One layer's deferred fold (ops::gdn_replay_fold_layer): one 128-thread item per CTA.
__global__
__launch_bounds__(fold::kItemThreads) void replay_fold_layer_kernel(fold::LayerArgs args) {
    __shared__ float key[1][kStateDim];
    __shared__ float reduction[1][kStateDim];
    fold::fold_layer_cta<fold::kItemThreads>(r9700::persistent::LaunchCta{}, args, blockIdx.x, key,
                                             reduction);
}

template <class Geometry>
void launch_fixed(const GdnReplayRecords& records, LinearAttentionStateAllLayersView states,
                  const GdnReplayFoldKernelRows& rows, std::int32_t active_rows,
                  hipStream_t stream) {
    const FoldAccess access{
        {static_cast<const hip_bfloat16*>(records.key.data),
         static_cast<const hip_bfloat16*>(records.value.data),
         static_cast<const float*>(records.gate.data),
         static_cast<const hip_bfloat16*>(records.conv.data), records.spec.record_capacity,
         records.spec.width},
        static_cast<float*>(states.recurrent_layer0.data),
        static_cast<hip_bfloat16*>(states.conv_layer0.data),
        states.recurrent_layer_stride_bytes / static_cast<std::int64_t>(sizeof(float)),
        states.conv_layer_stride_bytes / static_cast<std::int64_t>(sizeof(hip_bfloat16)),
        rows,
    };
    const dim3 grid(static_cast<unsigned>(Geometry::kValueHeads),
                    static_cast<unsigned>(active_rows),
                    static_cast<unsigned>(Geometry::kLayers * (kStateDim / kValuesPerBlock)));
    const dim3 block(kWaveSize, kWavesPerBlock, 1);
    hipLaunchKernelGGL((replay_fold_kernel<Geometry>), grid, block, 0, stream, access);
    HIP_CHECK(hipGetLastError());
}

} // namespace

void launch_replay_fold(const GdnReplayRecords& records, LinearAttentionStateAllLayersView states,
                        const GdnReplayFoldKernelRows& rows, std::int32_t active_rows,
                        hipStream_t stream) {
    if (records.spec.layers == FoldGeometry48x48::kLayers &&
        records.spec.qk_heads == FoldGeometry48x48::kQkHeads &&
        records.spec.value_heads == FoldGeometry48x48::kValueHeads &&
        records.spec.conv_channels == FoldGeometry48x48::kConvChannels) {
        launch_fixed<FoldGeometry48x48>(records, states, rows, active_rows, stream);
        return;
    }
    throw std::invalid_argument("GDN replay fold launcher received an unregistered geometry");
}

void launch_replay_fold_layer(const fold::LayerArgs& args, hipStream_t stream) {
    hipLaunchKernelGGL(
        replay_fold_layer_kernel,
        dim3(static_cast<unsigned>(args.batch) * fold::layer_ctas_per_row(fold::kItemThreads)),
        dim3(fold::kItemThreads), 0, stream, args);
    HIP_CHECK(hipGetLastError());
}

void launch_replay_prepare_layer(const fold::LayerArgs& args, const fold::Planes& staged,
                                 hipStream_t stream) {
    hipLaunchKernelGGL(
        replay_prepare_layer_kernel,
        dim3(static_cast<unsigned>(args.batch) * (fold::kConvChannels / fold::kStateDim)),
        dim3(128), 0, stream, args, staged);
    HIP_CHECK(hipGetLastError());
}

} // namespace ninfer::ops::detail::gated_delta_net
