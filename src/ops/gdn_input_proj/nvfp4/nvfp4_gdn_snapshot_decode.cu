#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.h"

#include "core/device.h"
#include "ops/common/math.cuh"
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

__device__ __forceinline__ float nvfp4_gdn_bf16_roundtrip(float x) {
    return __bfloat162float(__float2bfloat16_rn(x));
}

__device__ __forceinline__ void nvfp4_gdn_store_qkv(std::int32_t row, std::int32_t token,
                                                    std::int32_t /*width*/, __nv_bfloat16* query,
                                                    __nv_bfloat16* key, __nv_bfloat16* value,
                                                    __nv_bfloat16 out) {
    const std::int64_t column = token;
    if (row < kNvfp4GdnQueryRows) {
        query[column * kNvfp4GdnQueryRows + row] = out;
    } else if (row < kNvfp4GdnQueryRows + kNvfp4GdnKeyRows) {
        key[column * kNvfp4GdnKeyRows + row - kNvfp4GdnQueryRows] = out;
    } else {
        value[column * kNvfp4GdnValueRows + row - kNvfp4GdnQueryRows - kNvfp4GdnKeyRows] = out;
    }
}

template <class Geometry, class Schedule, bool Tree>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void
nvfp4_gdn_record_t1_kernel(Nvfp4PackedActivation<Geometry> activation,
                           const std::uint8_t* __restrict__ codes,
                           const std::uint8_t* __restrict__ scales, float inverse_weight_divisor,
                           const __nv_bfloat16* __restrict__ conv_weight,
                           const __nv_bfloat16* __restrict__ state_read,
                           const std::int32_t* __restrict__ initial_slots,
                           const std::int32_t* __restrict__ valid_columns,
                           const std::int32_t* __restrict__ parent_index,
                           __nv_bfloat16* __restrict__ query, __nv_bfloat16* __restrict__ key,
                           __nv_bfloat16* __restrict__ value, __nv_bfloat16* __restrict__ z,
                           __nv_bfloat16* __restrict__ record, std::int32_t width) {
    static_assert((Geometry::kOutputRows % 128) == 0);
    __shared__ Nvfp4GemvSharedStorage<Geometry, Schedule> shared;
    constexpr int kCtasPerM128 = 128 / Schedule::kRowsPerCta;
    const int m_tile           = static_cast<int>(blockIdx.x) / kCtasPerM128;
    const int cta_in_tile      = static_cast<int>(blockIdx.x) - m_tile * kCtasPerM128;
    const int rmod_base        = cta_in_tile * (Schedule::kRowsPerCta / 4);
    stage_nvfp4_scales<Geometry, Schedule>(scales, shared, m_tile, rmod_base);

    const int lane      = static_cast<int>(threadIdx.x) & 31;
    const int warp      = static_cast<int>(threadIdx.x) >> 5;
    const int flat_row0 = warp * Schedule::kRowsPerWarp;
    int parent_rows[Schedule::kRowsPerWarp];
#pragma unroll
    for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
        const int flat_row     = flat_row0 + local_row;
        const int rmod         = rmod_base + flat_row / 4;
        const int quartile     = flat_row & 3;
        parent_rows[local_row] = m_tile * 128 + rmod + quartile * 32;
    }

    const std::int32_t valid =
        valid_columns == nullptr ? width : valid_columns[0];
    const std::int64_t slot_stride = static_cast<std::int64_t>(kNvfp4GdnChannels) * 3;
    const std::int64_t initial_base =
        static_cast<std::int64_t>(initial_slots[0]) * slot_stride;

    float s0[Schedule::kRowsPerWarp];
    float s1[Schedule::kRowsPerWarp];
    float s2[Schedule::kRowsPerWarp];
    float saved0[Schedule::kRowsPerWarp][16];
    float saved1[Schedule::kRowsPerWarp][16];
    float saved2[Schedule::kRowsPerWarp][16];
    if (lane == 0) {
#pragma unroll
        for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
            const int row = parent_rows[local_row];
            if (row < kNvfp4GdnChannels) {
                s0[local_row] = __bfloat162float(state_read[initial_base + row]);
                s1[local_row] = __bfloat162float(state_read[initial_base + kNvfp4GdnChannels + row]);
                s2[local_row] =
                    __bfloat162float(state_read[initial_base + 2LL * kNvfp4GdnChannels + row]);
            }
        }
    }

    for (std::int32_t token = 0; token < width; ++token) {
        float accumulators[Schedule::kRowsPerWarp][Schedule::kAccumulatorChains] = {};
        compute_nvfp4_rows<Geometry, Schedule>(activation, codes, scales, shared,
                                               inverse_weight_divisor, parent_rows, flat_row0, lane,
                                               token, accumulators);
#pragma unroll
        for (int local_row = 0; local_row < Schedule::kRowsPerWarp; ++local_row) {
            float total = 0.0F;
#pragma unroll
            for (int chain = 0; chain < Schedule::kAccumulatorChains; ++chain) {
                total += accumulators[local_row][chain];
            }
            total = warp_reduce_sum(total);
            if (lane != 0) { continue; }
            const int row = parent_rows[local_row];
            if (row >= kNvfp4GdnChannels) {
                z[static_cast<std::int64_t>(token) * kNvfp4GdnZRows + row - kNvfp4GdnChannels] =
                    __float2bfloat16_rn(total);
                continue;
            }
            if (token >= valid) {
                nvfp4_gdn_store_qkv(row, token, width, query, key, value,
                                    __float2bfloat16_rn(0.0F));
                continue;
            }

            float h0 = s0[local_row];
            float h1 = s1[local_row];
            float h2 = s2[local_row];
            if constexpr (Tree) {
                const std::int32_t parent = parent_index[token];
                if (parent >= 0) {
                    h0 = saved0[local_row][parent];
                    h1 = saved1[local_row][parent];
                    h2 = saved2[local_row][parent];
                }
            }
            const float w0 = __bfloat162float(conv_weight[row]);
            const float w1 = __bfloat162float(conv_weight[kNvfp4GdnChannels + row]);
            const float w2 = __bfloat162float(conv_weight[2LL * kNvfp4GdnChannels + row]);
            const float w3 = __bfloat162float(conv_weight[3LL * kNvfp4GdnChannels + row]);
            float conv     = fmaf(w0, h0, 0.0F);
            conv           = fmaf(w1, h1, conv);
            conv           = fmaf(w2, h2, conv);
            conv           = fmaf(w3, total, conv);
            nvfp4_gdn_store_qkv(row, token, width, query, key, value, __float2bfloat16_rn(silu(conv)));
            record[static_cast<std::int64_t>(token) * kNvfp4GdnChannels + row] =
                __float2bfloat16_rn(total);
            if constexpr (Tree) {
                saved0[local_row][token] = nvfp4_gdn_bf16_roundtrip(h1);
                saved1[local_row][token] = nvfp4_gdn_bf16_roundtrip(h2);
                saved2[local_row][token] = nvfp4_gdn_bf16_roundtrip(total);
            } else {
                s0[local_row] = nvfp4_gdn_bf16_roundtrip(h1);
                s1[local_row] = nvfp4_gdn_bf16_roundtrip(h2);
                s2[local_row] = nvfp4_gdn_bf16_roundtrip(total);
            }
        }
    }
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

void nvfp4_gdn_record_t1_fused_launch(const Tensor& x, const Weight& weight,
                                      const Tensor& conv_weight, const Tensor& conv_states,
                                      const Tensor& valid_columns, const Tensor& initial_slot,
                                      Tensor& conv_record, Tensor& query, Tensor& key, Tensor& value,
                                      Tensor& z, cudaStream_t stream,
                                      const std::int32_t* parent_index) {
    if (x.ne[2] > 1) {
        const std::int32_t batch = x.ne[2];
        const std::int32_t width = x.ne[1];
        for (std::int32_t b = 0; b < batch; ++b) {
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
            nvfp4_gdn_record_t1_fused_launch(x.slice(2, b, 1), weight, conv_weight, conv_states,
                                             valid_b, initial_slot.slice(0, b, 1), record_b, query_b,
                                             key_b, value_b, z_b, stream, parent_b);
        }
        return;
    }

    using Geometry = Nvfp4GdnInputGeometry;
    using Schedule = typename Nvfp4LinearDecodeProductionSchedule<Geometry>::Type;
    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    const float inverse   = 1.0F / weight.weight_scale_divisor;
    const std::int32_t width = x.ne[1];
    const Nvfp4PackedActivation<Geometry> activation{static_cast<const __nv_bfloat16*>(x.data)};
    if (parent_index == nullptr) {
        nvfp4_gdn_record_t1_kernel<Geometry, Schedule, false>
            <<<kBlocks, Schedule::kThreads, 0, stream>>>(
                activation, static_cast<const std::uint8_t*>(weight.qdata),
                static_cast<const std::uint8_t*>(weight.scales), inverse,
                static_cast<const __nv_bfloat16*>(conv_weight.data),
                static_cast<const __nv_bfloat16*>(conv_states.data),
                static_cast<const std::int32_t*>(initial_slot.data),
                valid_columns.data == nullptr ? nullptr
                                              : static_cast<const std::int32_t*>(valid_columns.data),
                nullptr, static_cast<__nv_bfloat16*>(query.data),
                static_cast<__nv_bfloat16*>(key.data), static_cast<__nv_bfloat16*>(value.data),
                static_cast<__nv_bfloat16*>(z.data), static_cast<__nv_bfloat16*>(conv_record.data),
                width);
    } else {
        nvfp4_gdn_record_t1_kernel<Geometry, Schedule, true>
            <<<kBlocks, Schedule::kThreads, 0, stream>>>(
                activation, static_cast<const std::uint8_t*>(weight.qdata),
                static_cast<const std::uint8_t*>(weight.scales), inverse,
                static_cast<const __nv_bfloat16*>(conv_weight.data),
                static_cast<const __nv_bfloat16*>(conv_states.data),
                static_cast<const std::int32_t*>(initial_slot.data),
                valid_columns.data == nullptr ? nullptr
                                              : static_cast<const std::int32_t*>(valid_columns.data),
                parent_index, static_cast<__nv_bfloat16*>(query.data),
                static_cast<__nv_bfloat16*>(key.data), static_cast<__nv_bfloat16*>(value.data),
                static_cast<__nv_bfloat16*>(z.data), static_cast<__nv_bfloat16*>(conv_record.data),
                width);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
