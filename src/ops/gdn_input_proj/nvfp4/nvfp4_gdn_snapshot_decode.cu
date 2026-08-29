#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.h"

#include "core/device.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_conv_output.cuh"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_gemv.cuh"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {
namespace {

Tensor bf16_token_column(const Tensor& tensor, std::int32_t token, std::int32_t rows) {
    auto* data = static_cast<std::uint8_t*>(tensor.data) +
                 static_cast<std::int64_t>(token) * rows * sizeof(std::uint16_t);
    return Tensor(data, DType::BF16, {rows, 1});
}

__global__ void nvfp4_gdn_column_valid_kernel(const std::int32_t* valid_columns,
                                              std::int32_t token, std::int32_t width,
                                              std::int32_t* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        const std::int32_t valid = valid_columns == nullptr ? width : valid_columns[0];
        out[0]                   = token < valid ? 1 : 0;
    }
}

__global__ void nvfp4_gdn_copy_conv_slot_kernel(const __nv_bfloat16* src, __nv_bfloat16* dst,
                                                const std::int32_t* slot, std::int32_t channels) {
    const std::int32_t n        = channels * 3;
    const std::int64_t src_base = static_cast<std::int64_t>(slot[0]) * n;
    for (std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x); i < n;
         i += static_cast<std::int32_t>(blockDim.x * gridDim.x)) {
        dst[i] = src[src_base + i];
    }
}

__global__ void nvfp4_gdn_store_conv_slot_kernel(const __nv_bfloat16* src, __nv_bfloat16* dst,
                                                 const std::int32_t* base, std::int32_t token,
                                                 const std::int32_t* valid_flag,
                                                 std::int32_t channels) {
    if (valid_flag[0] == 0) { return; }
    const std::int32_t n        = channels * 3;
    const std::int64_t dst_base = static_cast<std::int64_t>(base[0] + token) * n;
    for (std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x); i < n;
         i += static_cast<std::int32_t>(blockDim.x * gridDim.x)) {
        dst[dst_base + i] = src[i];
    }
}

template <class Publish>
void launch_decode_column(const Tensor& x, const Weight& weight, const Tensor& conv_weight,
                          const Tensor& conv_states, const Tensor& valid_columns,
                          const Tensor& initial_slot, Tensor& query, Tensor& key, Tensor& value,
                          Tensor& z, Publish publish, cudaStream_t stream) {
    using Geometry = Nvfp4GdnInputGeometry;
    using Schedule = typename Nvfp4LinearDecodeProductionSchedule<Geometry>::Type;
    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    const float inverse   = 1.0F / weight.weight_scale_divisor;
    nvfp4_gemv_kernel<Geometry, Schedule><<<kBlocks, Schedule::kThreads, 0, stream>>>(
        Nvfp4PackedActivation<Geometry>{static_cast<const __nv_bfloat16*>(x.data)},
        static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), inverse, Nvfp4IdentityEpilogue{},
        make_nvfp4_gdn_conv_output<1>(conv_weight, conv_states, valid_columns, initial_slot, query,
                                      key, value, z, publish));
    CUDA_CHECK(cudaGetLastError());
}

void prepare_scratch(const Tensor& conv_states, const Tensor& initial_slot, Tensor& scratch,
                     Tensor& scratch_slot, cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(scratch_slot.data, 0, sizeof(std::int32_t), stream));
    nvfp4_gdn_copy_conv_slot_kernel<<<(kNvfp4GdnChannels * 3 + 255) / 256, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(conv_states.data),
        static_cast<__nv_bfloat16*>(scratch.data),
        static_cast<const std::int32_t*>(initial_slot.data), kNvfp4GdnChannels);
    CUDA_CHECK(cudaGetLastError());
}

void fill_column_valid(const Tensor& valid_columns, std::int32_t token, std::int32_t width,
                       Tensor& valid_t, cudaStream_t stream) {
    nvfp4_gdn_column_valid_kernel<<<1, 1, 0, stream>>>(
        valid_columns.data == nullptr ? nullptr
                                      : static_cast<const std::int32_t*>(valid_columns.data),
        token, width, static_cast<std::int32_t*>(valid_t.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void nvfp4_gdn_snapshot_decode_launch(const Tensor& x, const Weight& weight,
                                      const Tensor& conv_weight, Tensor& conv_states,
                                      const Tensor& valid_columns, const Tensor& initial_slot,
                                      const Tensor& snapshot_base_slot, Tensor& query, Tensor& key,
                                      Tensor& value, Tensor& z, cudaStream_t stream) {
    launch_decode_column(
        x, weight, conv_weight, conv_states, valid_columns, initial_slot, query, key, value, z,
        SnapshotHistoryPublish{static_cast<__nv_bfloat16*>(conv_states.data),
                               static_cast<const std::int32_t*>(snapshot_base_slot.data),
                               kNvfp4GdnChannels},
        stream);
}

void nvfp4_gdn_record_decode_launch(const Tensor& x, const Weight& weight,
                                    const Tensor& conv_weight, const Tensor& conv_states,
                                    const Tensor& valid_columns, const Tensor& initial_slot,
                                    Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value,
                                    Tensor& z, cudaStream_t stream) {
    launch_decode_column(
        x, weight, conv_weight, conv_states, valid_columns, initial_slot, query, key, value, z,
        RecordColumnPublish{static_cast<__nv_bfloat16*>(conv_record.data), kNvfp4GdnChannels,
                            x.ne[1]},
        stream);
}

void nvfp4_gdn_snapshot_decode_columns(const Tensor& x, const Weight& weight,
                                       const Tensor& conv_weight, Tensor& conv_states,
                                       const Tensor& valid_columns, const Tensor& initial_slot,
                                       const Tensor& snapshot_base_slot, Tensor& query, Tensor& key,
                                       Tensor& value, Tensor& z, WorkspaceArena& workspace,
                                       cudaStream_t stream) {
    const std::int32_t width = x.ne[1];
    if (width == 1) {
        nvfp4_gdn_snapshot_decode_launch(x, weight, conv_weight, conv_states, valid_columns,
                                         initial_slot, snapshot_base_slot, query, key, value, z,
                                         stream);
        return;
    }
    auto scope           = workspace.scope();
    Tensor scratch       = workspace.alloc(DType::BF16, {kNvfp4GdnChannels, 3, 1});
    Tensor scratch_slot  = workspace.alloc(DType::I32, {1});
    Tensor valid_t       = workspace.alloc(DType::I32, {1});
    prepare_scratch(conv_states, initial_slot, scratch, scratch_slot, stream);
    for (std::int32_t token = 0; token < width; ++token) {
        fill_column_valid(valid_columns, token, width, valid_t, stream);
        Tensor x_t     = bf16_token_column(x, token, Nvfp4GdnInputGeometry::kInputRows);
        Tensor query_t = bf16_token_column(query, token, kNvfp4GdnQueryRows);
        Tensor key_t   = bf16_token_column(key, token, kNvfp4GdnKeyRows);
        Tensor value_t = bf16_token_column(value, token, kNvfp4GdnValueRows);
        Tensor z_t     = bf16_token_column(z, token, kNvfp4GdnZRows);
        launch_decode_column(
            x_t, weight, conv_weight, scratch, valid_t, scratch_slot, query_t, key_t, value_t, z_t,
            SnapshotHistoryPublish{static_cast<__nv_bfloat16*>(scratch.data),
                                   static_cast<const std::int32_t*>(scratch_slot.data),
                                   kNvfp4GdnChannels},
            stream);
        nvfp4_gdn_store_conv_slot_kernel<<<(kNvfp4GdnChannels * 3 + 255) / 256, 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(scratch.data),
            static_cast<__nv_bfloat16*>(conv_states.data),
            static_cast<const std::int32_t*>(snapshot_base_slot.data), token,
            static_cast<const std::int32_t*>(valid_t.data), kNvfp4GdnChannels);
        CUDA_CHECK(cudaGetLastError());
    }
}

void nvfp4_gdn_record_decode_columns(const Tensor& x, const Weight& weight,
                                     const Tensor& conv_weight, const Tensor& conv_states,
                                     const Tensor& valid_columns, const Tensor& initial_slot,
                                     Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value,
                                     Tensor& z, WorkspaceArena& workspace, cudaStream_t stream) {
    const std::int32_t width = x.ne[1];
    auto scope               = workspace.scope();
    Tensor scratch           = workspace.alloc(DType::BF16, {kNvfp4GdnChannels, 3, 1});
    Tensor scratch_slot      = workspace.alloc(DType::I32, {1});
    Tensor valid_t           = workspace.alloc(DType::I32, {1});
    prepare_scratch(conv_states, initial_slot, scratch, scratch_slot, stream);
    for (std::int32_t token = 0; token < width; ++token) {
        fill_column_valid(valid_columns, token, width, valid_t, stream);
        Tensor x_t      = bf16_token_column(x, token, Nvfp4GdnInputGeometry::kInputRows);
        Tensor query_t  = bf16_token_column(query, token, kNvfp4GdnQueryRows);
        Tensor key_t    = bf16_token_column(key, token, kNvfp4GdnKeyRows);
        Tensor value_t  = bf16_token_column(value, token, kNvfp4GdnValueRows);
        Tensor z_t      = bf16_token_column(z, token, kNvfp4GdnZRows);
        Tensor record_t = bf16_token_column(conv_record, token, kNvfp4GdnChannels);
        launch_decode_column(
            x_t, weight, conv_weight, scratch, valid_t, scratch_slot, query_t, key_t, value_t, z_t,
            SnapshotAndRecordPublish{
                SnapshotHistoryPublish{static_cast<__nv_bfloat16*>(scratch.data),
                                       static_cast<const std::int32_t*>(scratch_slot.data),
                                       kNvfp4GdnChannels},
                RecordColumnPublish{static_cast<__nv_bfloat16*>(record_t.data), kNvfp4GdnChannels,
                                    1}},
            stream);
    }
}

} // namespace ninfer::ops::detail
