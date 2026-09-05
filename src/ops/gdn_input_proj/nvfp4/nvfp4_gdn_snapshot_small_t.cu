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
                              Tensor&, WorkspaceArena&, const std::int32_t*, cudaStream_t);

template <class Geometry>
struct Nvfp4GroupedPackedActivation {
    const __nv_bfloat16* x;
    int group_width;
    int columns;

    __device__ __forceinline__ const __nv_bfloat16* values(int token, int value_begin) const {
        int column = static_cast<int>(blockIdx.x) * group_width + token;
        column     = column < columns ? column : columns - 1;
        return x + static_cast<std::int64_t>(column) * Geometry::kInputRows + value_begin;
    }
};

template <int Width, int RequestsPerGroup>
struct Nvfp4GroupedGdnProjectionOutput {
    float* projected;
    __nv_bfloat16* z;
    int columns;

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t token,
                                          float value) const {
        const int column = static_cast<int>(blockIdx.x) * (RequestsPerGroup * Width) + token;
        if (column >= columns) { return; }
        if (parent_row < kNvfp4GdnChannels) {
            projected[static_cast<std::int64_t>(column) * kNvfp4GdnChannels + parent_row] = value;
        } else {
            z[static_cast<std::int64_t>(column) * kNvfp4GdnZRows + parent_row - kNvfp4GdnChannels] =
                __float2bfloat16_rn(value);
        }
    }
};

template <class Geometry>
struct Nvfp4TriplePackedActivation {
    const __nv_bfloat16* x;

    __device__ __forceinline__ const __nv_bfloat16* values(int token, int value_begin) const {
        return x + static_cast<std::int64_t>(token) * Geometry::kInputRows + value_begin;
    }
};

template <int Width>
struct Nvfp4TripleGdnProjectionOutput {
    float* projected;
    __nv_bfloat16* z;

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t token,
                                          float value) const {
        if (parent_row < kNvfp4GdnChannels) {
            projected[static_cast<std::int64_t>(token) * kNvfp4GdnChannels + parent_row] = value;
        } else {
            z[static_cast<std::int64_t>(token) * kNvfp4GdnZRows + parent_row -
              kNvfp4GdnChannels] = __float2bfloat16_rn(value);
        }
    }
};

template <int Width, bool Tree, class Publish>
__global__ __launch_bounds__(256) void nvfp4_grouped_gdn_conv_kernel(
    const float* __restrict__ projected, GdnConvEpilogue<Publish, Tree> conv, int batch) {
    const int batch_row = static_cast<int>(blockIdx.x);
    const int row =
        static_cast<int>(blockIdx.y) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (batch_row >= batch || row >= kNvfp4GdnChannels) { return; }

    float row_projection[Width];
#pragma unroll
    for (int token = 0; token < Width; ++token) {
        const int column = batch_row * Width + token;
        row_projection[token] =
            projected[static_cast<std::int64_t>(column) * kNvfp4GdnChannels + row];
    }
    conv.batch_row = batch_row;
    conv.store(row, row_projection);
}

template <int Width, int RequestsPerGroup, bool Tree, class Publish>
void launch_grouped_record(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                           const Tensor& conv_states, const Tensor& valid_columns,
                           const Tensor& initial_slot, Tensor& conv_record, Tensor& query,
                           Tensor& key, Tensor& value, Tensor& z, Publish publish,
                           WorkspaceArena& workspace, cudaStream_t stream,
                           const std::int32_t* parent_index) {
    using Geometry                = Nvfp4GdnInputGeometry;
    constexpr int kGroupedTokens  = RequestsPerGroup * Width;
    constexpr int kRowsPerWarp    = 2;
    // Match the W-local GDN reduction exactly: 8 warps, 16 values/lane, four
    // accumulator chains. Only the token register panel widens to replay weights.
    using Schedule =
        Nvfp4SmallTSchedule<8, 1, kRowsPerWarp, 16, kGroupedTokens, 4,
                            Nvfp4SmallTActivationAccess::TokenPacked,
                            Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 1,
                            Nvfp4SmallTBlockOrder::RowsContiguous, 1>;
    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    const int batch       = x.ne[2];
    const int columns     = batch * Width;
    const int group_count = (batch + RequestsPerGroup - 1) / RequestsPerGroup;
    const float inverse   = 1.0F / weight.weight_scale_divisor;

    auto scope       = workspace.scope();
    Tensor projected = workspace.alloc(DType::FP32, {kNvfp4GdnChannels, Width, batch}, 256);
    nvfp4_small_t_kernel<Geometry, kGroupedTokens, Schedule, Nvfp4IdentityEpilogue,
                         Nvfp4GroupedGdnProjectionOutput<Width, RequestsPerGroup>,
                         Nvfp4SmallTFinalization::Elementwise,
                         Nvfp4GroupedPackedActivation<Geometry>>
        <<<dim3(group_count, kBlocks), Schedule::kThreads, 0, stream>>>(
            Nvfp4GroupedPackedActivation<Geometry>{static_cast<const __nv_bfloat16*>(x.data),
                                                   kGroupedTokens, columns},
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), inverse, Nvfp4IdentityEpilogue{},
            Nvfp4GroupedGdnProjectionOutput<Width, RequestsPerGroup>{
                static_cast<float*>(projected.data), static_cast<__nv_bfloat16*>(z.data), columns});
    CUDA_CHECK(cudaGetLastError());

    auto conv = make_nvfp4_gdn_conv_output<Width, Tree>(conv_weight, conv_states, valid_columns,
                                                        initial_slot, query, key, value, z, publish,
                                                        parent_index)
                    .conv;
    constexpr int kPostBlocks = (kNvfp4GdnChannels + 255) / 256;
    nvfp4_grouped_gdn_conv_kernel<Width, Tree><<<dim3(batch, kPostBlocks), 256, 0, stream>>>(
        static_cast<const float*>(projected.data), conv, batch);
    CUDA_CHECK(cudaGetLastError());
}

template <int Width, bool Tree, class Publish>
void launch_triple_record(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                          const Tensor& conv_states, const Tensor& valid_columns,
                          const Tensor& initial_slot, Tensor& conv_record, Tensor& query,
                          Tensor& key, Tensor& value, Tensor& z, Publish publish,
                          WorkspaceArena& workspace, cudaStream_t stream,
                          const std::int32_t* parent_index) {
    // C=3 owns one exact request-major 3*W panel, avoiding the half-empty second pair.
    using Geometry              = Nvfp4GdnInputGeometry;
    constexpr int kTripleTokens = 3 * Width;
    using Schedule =
        Nvfp4SmallTSchedule<8, 1, 2, 16, kTripleTokens, 4,
                            Nvfp4SmallTActivationAccess::TokenPacked,
                            Nvfp4ScaleAccess::StagedRaw, Nvfp4CodeCache::Default, 1,
                            Nvfp4SmallTBlockOrder::RowsContiguous, 1>;
    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    const float inverse   = 1.0F / weight.weight_scale_divisor;

    auto scope       = workspace.scope();
    Tensor projected = workspace.alloc(DType::FP32, {kNvfp4GdnChannels, Width, 3}, 256);
    nvfp4_small_t_kernel<Geometry, kTripleTokens, Schedule, Nvfp4IdentityEpilogue,
                         Nvfp4TripleGdnProjectionOutput<Width>,
                         Nvfp4SmallTFinalization::Elementwise,
                         Nvfp4TriplePackedActivation<Geometry>>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(
            Nvfp4TriplePackedActivation<Geometry>{static_cast<const __nv_bfloat16*>(x.data)},
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), inverse, Nvfp4IdentityEpilogue{},
            Nvfp4TripleGdnProjectionOutput<Width>{static_cast<float*>(projected.data),
                                                  static_cast<__nv_bfloat16*>(z.data)});
    CUDA_CHECK(cudaGetLastError());

    auto conv = make_nvfp4_gdn_conv_output<Width, Tree>(conv_weight, conv_states, valid_columns,
                                                        initial_slot, query, key, value, z, publish,
                                                        parent_index)
                    .conv;
    constexpr int kPostBlocks = (kNvfp4GdnChannels + 255) / 256;
    nvfp4_grouped_gdn_conv_kernel<Width, Tree><<<dim3(3, kPostBlocks), 256, 0, stream>>>(
        static_cast<const float*>(projected.data), conv, 3);
    CUDA_CHECK(cudaGetLastError());
}

// Packed snapshot T=2..16 GDN conv uses the SmallT production schedule (token-parallel
// GEMM) with FP32 projected-conv in the epilogue. Record B=1 retains the fused T=1
// GEMV+conv route; qualified B>1 W=2/5 shapes use grouped weight-replay projection
// (pairs, plus a direct W=5 triple at B=3) and separate FP32 conv. Convolution history
// rounds through BF16 at every column, matching the T=1 route.
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
                Nvfp4BatchedPackedActivation<Geometry>{
                    static_cast<const __nv_bfloat16*>(x.data), ActiveTokens},
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
                         Tensor& key, Tensor& value, Tensor& z, WorkspaceArena& workspace,
                         const std::int32_t* parent_index, cudaStream_t stream) {
    const RecordColumnPublish publish{static_cast<__nv_bfloat16*>(conv_record.data),
                                      kNvfp4GdnChannels, ActiveTokens};
    if (nvfp4_gdn_record_uses_grouped_replay(ActiveTokens, x.ne[2])) {
        if (parent_index == nullptr) {
            if constexpr (ActiveTokens == 5) {
                if (x.ne[2] == 3) {
                    launch_triple_record<ActiveTokens, false>(
                        x, weight, conv_weight, conv_states, valid_columns, initial_slot,
                        conv_record, query, key, value, z, publish, workspace, stream, nullptr);
                    return;
                }
            }
            {
                launch_grouped_record<ActiveTokens, 2, false>(
                    x, weight, conv_weight, conv_states, valid_columns, initial_slot, conv_record,
                    query, key, value, z, publish, workspace, stream, nullptr);
            }
        } else {
            if constexpr (ActiveTokens == 5) {
                if (x.ne[2] == 3) {
                    launch_triple_record<ActiveTokens, true>(
                        x, weight, conv_weight, conv_states, valid_columns, initial_slot,
                        conv_record, query, key, value, z, publish, workspace, stream,
                        parent_index);
                    return;
                }
            }
            {
                launch_grouped_record<ActiveTokens, 2, true>(
                    x, weight, conv_weight, conv_states, valid_columns, initial_slot, conv_record,
                    query, key, value, z, publish, workspace, stream, parent_index);
            }
        }
        return;
    }
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
                                     Tensor& z, WorkspaceArena& workspace, cudaStream_t stream,
                                     const std::int32_t* parent_index) {
    const std::size_t index = static_cast<std::size_t>(x.ne[1] - kNvfp4FirstSmallT);
    kRecordLaunchers[index](x, weight, conv_weight, conv_states, valid_columns, initial_slot,
                            conv_record, query, key, value, z, workspace, parent_index, stream);
}

} // namespace ninfer::ops::detail
