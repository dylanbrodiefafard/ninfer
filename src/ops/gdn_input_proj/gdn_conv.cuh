#pragma once

#include "ops/common/math.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

struct SnapshotHistoryPublish {
    __nv_bfloat16* state_write;
    const std::int32_t* snapshot_base_slots;
    std::int32_t channels;

    __device__ __forceinline__ void publish(std::int32_t token, std::int32_t batch,
                                            std::int32_t row, float s1, float s2, float p) const {
        const std::int64_t slot_stride = static_cast<std::int64_t>(channels) * 3;
        const std::int64_t base =
            static_cast<std::int64_t>(snapshot_base_slots[batch] + token) * slot_stride;
        state_write[base + row]                  = __float2bfloat16_rn(s1);
        state_write[base + channels + row]       = __float2bfloat16_rn(s2);
        state_write[base + 2LL * channels + row] = __float2bfloat16_rn(p);
    }
};

struct RecordColumnPublish {
    __nv_bfloat16* record;
    std::int32_t channels;
    std::int32_t width;

    __device__ __forceinline__ void publish(std::int32_t token, std::int32_t batch,
                                            std::int32_t row, float, float, float p) const {
        const std::int64_t column       = static_cast<std::int64_t>(batch) * width + token;
        record[column * channels + row] = __float2bfloat16_rn(p);
    }
};

struct SnapshotAndRecordPublish {
    SnapshotHistoryPublish snapshot;
    RecordColumnPublish record;

    __device__ __forceinline__ void publish(std::int32_t token, std::int32_t batch,
                                            std::int32_t row, float s1, float s2, float p) const {
        snapshot.publish(token, batch, row, s1, s2, p);
        record.publish(token, batch, row, s1, s2, p);
    }
};

struct NoHistoryPublish {
    __device__ __forceinline__ void publish(std::int32_t, std::int32_t, std::int32_t, float, float,
                                            float) const {}
};

// Device-side implementation detail shared by exact packed projection kernels. Projection
// accumulators stay in the route's existing private precision; Publish changes only the side
// effect after the convolution has consumed that accumulator.
//
// Tree=true is a separate instantiation so the sequential SmallT fused kernel does not grow a
// parent-index branch. Packed-tree verify (parent_index[0]=-1) must use the same fused GEMV+conv
// as T=1 / chain SmallT; compose-record's SmallT GEMV + BF16 projected-conv flips greedy tokens.
template <class Publish, bool Tree = false>
struct GdnConvEpilogue {
    const __nv_bfloat16* conv_weight;
    const __nv_bfloat16* state_read;
    const std::int32_t* initial_slots;
    const std::int32_t* valid_columns;
    __nv_bfloat16* query;
    __nv_bfloat16* key;
    __nv_bfloat16* value;
    std::int32_t channels;
    std::int32_t query_rows;
    std::int32_t key_rows;
    std::int32_t value_rows;
    std::int32_t global_row_offset;
    std::int32_t width;
    std::int32_t batch_row;
    Publish publish;

    template <int Tokens>
    __device__ __forceinline__ void store(std::int32_t local_row,
                                          const float (&projected)[Tokens]) const {
        static_assert(Tokens >= 1);
        const std::int32_t row         = global_row_offset + local_row;
        const std::int64_t slot_stride = static_cast<std::int64_t>(channels) * 3;
        const std::int64_t initial_base =
            static_cast<std::int64_t>(initial_slots[batch_row]) * slot_stride;
        std::int32_t valid = valid_columns == nullptr ? Tokens : valid_columns[batch_row];
        valid              = valid < 0 ? 0 : (valid > Tokens ? Tokens : valid);

        float s0       = __bfloat162float(state_read[initial_base + row]);
        float s1       = __bfloat162float(state_read[initial_base + channels + row]);
        float s2       = __bfloat162float(state_read[initial_base + 2LL * channels + row]);
        const float w0 = __bfloat162float(conv_weight[row]);
        const float w1 = __bfloat162float(conv_weight[channels + row]);
        const float w2 = __bfloat162float(conv_weight[2LL * channels + row]);
        const float w3 = __bfloat162float(conv_weight[3LL * channels + row]);

#pragma unroll
        for (int token = 0; token < Tokens; ++token) {
            const std::int64_t column = static_cast<std::int64_t>(batch_row) * width + token;
            if (token >= valid) {
                if (row < query_rows) {
                    query[column * query_rows + row] = __float2bfloat16_rn(0.0F);
                } else if (row < query_rows + key_rows) {
                    key[column * key_rows + row - query_rows] = __float2bfloat16_rn(0.0F);
                } else {
                    value[column * value_rows + row - query_rows - key_rows] =
                        __float2bfloat16_rn(0.0F);
                }
                continue;
            }

            const float p              = projected[token];
            float conv                 = fmaf(w0, s0, 0.0F);
            conv                       = fmaf(w1, s1, conv);
            conv                       = fmaf(w2, s2, conv);
            conv                       = fmaf(w3, p, conv);
            const __nv_bfloat16 output = __float2bfloat16_rn(silu(conv));
            if (row < query_rows) {
                query[column * query_rows + row] = output;
            } else if (row < query_rows + key_rows) {
                key[column * key_rows + row - query_rows] = output;
            } else {
                value[column * value_rows + row - query_rows - key_rows] = output;
            }

            publish.publish(token, batch_row, row, s1, s2, p);
            s0 = __bfloat162float(__float2bfloat16_rn(s1));
            s1 = __bfloat162float(__float2bfloat16_rn(s2));
            s2 = __bfloat162float(__float2bfloat16_rn(p));
        }
    }
};

template <class Publish>
struct GdnConvEpilogue<Publish, true> {
    const __nv_bfloat16* conv_weight;
    const __nv_bfloat16* state_read;
    const std::int32_t* initial_slots;
    const std::int32_t* valid_columns;
    __nv_bfloat16* query;
    __nv_bfloat16* key;
    __nv_bfloat16* value;
    std::int32_t channels;
    std::int32_t query_rows;
    std::int32_t key_rows;
    std::int32_t value_rows;
    std::int32_t global_row_offset;
    std::int32_t width;
    std::int32_t batch_row;
    Publish publish;
    const std::int32_t* parent_index;

    template <int Tokens>
    __device__ __forceinline__ void store(std::int32_t local_row,
                                          const float (&projected)[Tokens]) const {
        static_assert(Tokens >= 1);
        static_assert(Tokens <= 16);
        const std::int32_t row         = global_row_offset + local_row;
        const std::int64_t slot_stride = static_cast<std::int64_t>(channels) * 3;
        const std::int64_t initial_base =
            static_cast<std::int64_t>(initial_slots[batch_row]) * slot_stride;
        std::int32_t valid = valid_columns == nullptr ? Tokens : valid_columns[batch_row];
        valid              = valid < 0 ? 0 : (valid > Tokens ? Tokens : valid);

        const float checkpoint0 = __bfloat162float(state_read[initial_base + row]);
        const float checkpoint1 = __bfloat162float(state_read[initial_base + channels + row]);
        const float checkpoint2 =
            __bfloat162float(state_read[initial_base + 2LL * channels + row]);
        const float w0 = __bfloat162float(conv_weight[row]);
        const float w1 = __bfloat162float(conv_weight[channels + row]);
        const float w2 = __bfloat162float(conv_weight[2LL * channels + row]);
        const float w3 = __bfloat162float(conv_weight[3LL * channels + row]);

        float saved0[Tokens];
        float saved1[Tokens];
        float saved2[Tokens];
        for (int token = 0; token < Tokens; ++token) {
            const std::int64_t column = static_cast<std::int64_t>(batch_row) * width + token;
            if (token >= valid) {
                if (row < query_rows) {
                    query[column * query_rows + row] = __float2bfloat16_rn(0.0F);
                } else if (row < query_rows + key_rows) {
                    key[column * key_rows + row - query_rows] = __float2bfloat16_rn(0.0F);
                } else {
                    value[column * value_rows + row - query_rows - key_rows] =
                        __float2bfloat16_rn(0.0F);
                }
                continue;
            }

            const std::int32_t parent = parent_index[column];
            float s0;
            float s1;
            float s2;
            if (parent < 0) {
                s0 = checkpoint0;
                s1 = checkpoint1;
                s2 = checkpoint2;
            } else {
                s0 = saved0[parent];
                s1 = saved1[parent];
                s2 = saved2[parent];
            }

            const float p              = projected[token];
            float conv                 = fmaf(w0, s0, 0.0F);
            conv                       = fmaf(w1, s1, conv);
            conv                       = fmaf(w2, s2, conv);
            conv                       = fmaf(w3, p, conv);
            const __nv_bfloat16 output = __float2bfloat16_rn(silu(conv));
            if (row < query_rows) {
                query[column * query_rows + row] = output;
            } else if (row < query_rows + key_rows) {
                key[column * key_rows + row - query_rows] = output;
            } else {
                value[column * value_rows + row - query_rows - key_rows] = output;
            }

            publish.publish(token, batch_row, row, s1, s2, p);
            saved0[token] = __bfloat162float(__float2bfloat16_rn(s1));
            saved1[token] = __bfloat162float(__float2bfloat16_rn(s2));
            saved2[token] = __bfloat162float(__float2bfloat16_rn(p));
        }
    }
};

} // namespace ninfer::ops::detail
