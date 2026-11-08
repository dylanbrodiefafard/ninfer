#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.h"

#include "core/device.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_conv_output.cuh"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_w4a8_mma.cuh"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

// Convolution and record publication of a staged FP32 [kNvfp4GdnChannels, Width, batch]
// projection. Grid (batch, ceil(kNvfp4GdnChannels / 256)), 256 threads: each thread owns one
// channel row of one request, loads that row's Width projected columns and runs the conv
// epilogue over them. No shared memory or block synchronization.
template <int Width, bool Tree, class Publish>
__global__ __launch_bounds__(256) void nvfp4_staged_gdn_conv_kernel(
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

template <int Width, bool Tree>
struct A8RecordOutput {
    static_assert(nvfp4_gdn_record_width_registered(Width, Tree));
    // W*B<=48 fits the launcher's single M16/M32/M48 tile, so every request's temporal columns
    // are present here; larger launches retain global staging.
    Nvfp4GdnConvOutput<Width, RecordColumnPublish, Tree> output;

    // The wide-path panel counts the fused launches of this width select.
    static constexpr bool reachable_panels(int panels) {
        return nvfp4_gdn_a8_fused_record_panels(Width, Tree, panels);
    }

    template <class Schedule>
    __device__ void store_tile(const float* tile, int row_begin, int tokens) const {
        constexpr int BN = Schedule::kBlockN, stride = BN + 8;
        for (int task = threadIdx.x; task < BN * (tokens / Width); task += Schedule::kThreads) {
            const int row = task % BN, batch = task / BN;
            float projected[Width];
#pragma unroll
            for (int t = 0; t < Width; ++t) {
                projected[t] = tile[(batch * Width + t) * stride + row];
            }
            auto lane           = output;
            lane.conv.batch_row = batch;
            lane.store_row(row_begin + row, projected);
        }
    }
};

// Grouped convolution/record publication from a staged FP32 [channels, Width, batch] projection.
template <int Width, bool Tree>
void launch_record_conv(const Tensor& projected, const Tensor& conv_weight,
                        const Tensor& conv_states, const Tensor& valid_columns,
                        const Tensor& initial_slot, Tensor& conv_record, Tensor& query, Tensor& key,
                        Tensor& value, Tensor& z, int batch, const std::int32_t* parent_index,
                        cudaStream_t stream) {
    const RecordColumnPublish publish{static_cast<__nv_bfloat16*>(conv_record.data),
                                      kNvfp4GdnChannels, Width};
    auto conv = make_nvfp4_gdn_conv_output<Width, Tree>(conv_weight, conv_states, valid_columns,
                                                        initial_slot, query, key, value, z, publish,
                                                        parent_index)
                    .conv;
    nvfp4_staged_gdn_conv_kernel<Width, Tree>
        <<<dim3(batch, (kNvfp4GdnChannels + 255) / 256), 256, 0, stream>>>(
            static_cast<const float*>(projected.data), conv, batch);
    CUDA_CHECK(cudaGetLastError());
}

// W4A8 record from an already quantized [5120, Width*batch] activation. Launches that fit the
// single fused tile (Width*batch <= kNvfp4GdnA8FusedRecordMaxTokens) publish from the MMA
// epilogue; the others stage FP32. Only widths whose largest batch exceeds the fused tile
// compile the staged route.
template <int Width, bool Tree>
void project_a8_record(Fp8A8Workspace activation, int batch, const Weight& weight,
                       const Tensor& conv_weight, const Tensor& conv_states,
                       const Tensor& valid_columns, const Tensor& initial_slot, Tensor& conv_record,
                       Tensor& query, Tensor& key, Tensor& value, Tensor& z,
                       WorkspaceArena& workspace, const std::int32_t* parent_index,
                       cudaStream_t stream) {
    if (Width * batch <= kNvfp4GdnA8FusedRecordMaxTokens) {
        const RecordColumnPublish publish{static_cast<__nv_bfloat16*>(conv_record.data),
                                          kNvfp4GdnChannels, Width};
        const auto output = make_nvfp4_gdn_conv_output<Width, Tree>(
            conv_weight, conv_states, valid_columns, initial_slot, query, key, value, z, publish,
            parent_index);
        launch_nvfp4_w4a8_mma<Nvfp4GdnInputGeometry>(weight, Width * batch, activation,
                                                     Nvfp4IdentityEpilogue{},
                                                     A8RecordOutput<Width, Tree>{output}, stream);
        return;
    }
    if constexpr (Width * kDflash2MaxBatch > kNvfp4GdnA8FusedRecordMaxTokens) {
        auto scope       = workspace.scope();
        Tensor projected = workspace.alloc(DType::FP32, {kNvfp4GdnChannels, Width, batch}, 256);
        nvfp4_gdn_input_w4a8_fp32_project(weight, Width * batch, activation, projected, z, stream);
        launch_record_conv<Width, Tree>(projected, conv_weight, conv_states, valid_columns,
                                        initial_slot, conv_record, query, key, value, z, batch,
                                        parent_index, stream);
    } else {
        throw std::logic_error("nvfp4 gdn A8 record: batch exceeds the registered domain");
    }
}

using A8InputRecordLaunch = void (*)(Fp8A8Workspace, int, const Weight&, const Tensor&,
                                     const Tensor&, const Tensor&, const Tensor&, Tensor&, Tensor&,
                                     Tensor&, Tensor&, Tensor&, WorkspaceArena&,
                                     const std::int32_t*, cudaStream_t);

constexpr std::size_t kA8RecordWidths = kNvfp4GdnA8FusedRecordMaxWidth - kNvfp4FirstA8 + 1;

// The launcher of a registered width/tree pair, nullptr elsewhere, so only registered pairs
// instantiate.
template <int Width, bool Tree>
constexpr A8InputRecordLaunch registered_launcher() {
    if constexpr (nvfp4_gdn_record_width_registered(Width, Tree)) {
        return &project_a8_record<Width, Tree>;
    } else {
        return nullptr;
    }
}

template <std::size_t... Offsets>
constexpr auto make_a8_input_record_launchers(std::index_sequence<Offsets...>) {
    return std::array<A8InputRecordLaunch, sizeof...(Offsets) * 2>{
        registered_launcher<kNvfp4FirstA8 + static_cast<int>(Offsets), false>()...,
        registered_launcher<kNvfp4FirstA8 + static_cast<int>(Offsets), true>()...};
}

constexpr auto kA8InputRecordLaunchers =
    make_a8_input_record_launchers(std::make_index_sequence<kA8RecordWidths>{});

void require_registered_record(std::int32_t width, std::int32_t batch, bool tree) {
    if (!nvfp4_gdn_record_width_registered(width, tree) || batch <= 0 || batch > kDflash2MaxBatch) {
        throw std::invalid_argument("nvfp4 gdn A8 record: unregistered W/B/tree shape");
    }
}

} // namespace

void nvfp4_gdn_record_a8_input_launch(Fp8A8Workspace activation, std::int32_t width,
                                      std::int32_t batch, const Weight& weight,
                                      const Tensor& conv_weight, const Tensor& conv_states,
                                      const Tensor& valid_columns, const Tensor& initial_slot,
                                      Tensor& conv_record, Tensor& query, Tensor& key,
                                      Tensor& value, Tensor& z, WorkspaceArena& workspace,
                                      cudaStream_t stream, const std::int32_t* parent_index) {
    const bool tree = parent_index != nullptr;
    require_registered_record(width, batch, tree);
    const auto launch = kA8InputRecordLaunchers[static_cast<std::size_t>(width - kNvfp4FirstA8) +
                                                (tree ? kA8RecordWidths : 0)];
    launch(activation, batch, weight, conv_weight, conv_states, valid_columns, initial_slot,
           conv_record, query, key, value, z, workspace, parent_index, stream);
}

void nvfp4_gdn_record_quantized_launch(const Tensor& x, const Weight& weight,
                                       const Tensor& conv_weight, const Tensor& conv_states,
                                       const Tensor& valid_columns, const Tensor& initial_slot,
                                       Tensor& conv_record, Tensor& query, Tensor& key,
                                       Tensor& value, Tensor& z, WorkspaceArena& workspace,
                                       cudaStream_t stream, const std::int32_t* parent_index) {
    const std::int32_t width = x.ne[1];
    const std::int32_t batch = x.ne[2];
    require_registered_record(width, batch, parent_index != nullptr);
    auto scope           = workspace.scope();
    const auto quantized = allocate_fp8_a8_workspace(workspace, width * batch, weight.k);
    launch_fp8_a8_quantize(x.view({weight.k, width * batch}), weight, quantized, stream);
    nvfp4_gdn_record_a8_input_launch(quantized, width, batch, weight, conv_weight, conv_states,
                                     valid_columns, initial_slot, conv_record, query, key, value, z,
                                     workspace, stream, parent_index);
}

} // namespace ninfer::ops::detail
