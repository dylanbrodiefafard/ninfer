#pragma once

#include "ops/common/bf16_vector.cuh"
#include "ops/linear_attention/gated_delta_net/common.cuh"
#include "ninfer/ops/gated_delta_net.h"
#include "ops/linear_attention/gated_delta_net/launch.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail::gated_delta_net {

inline constexpr int kDvPerWarp = 4;
inline constexpr int kNumWarps  = 4;
inline constexpr int kBlockDv   = kNumWarps * kDvPerWarp;
inline constexpr int kQkPerLane = kStateDim / kWarpSize;

static_assert(kStateDim % kWarpSize == 0);
static_assert(kQkPerLane == 4);
static_assert(kStateDim % kBlockDv == 0);

__device__ __forceinline__ void load_qk_lane(float (&reg)[kQkPerLane], const float* base,
                                             std::uint32_t dqk_base) {
    store_vec(reg, load_vec<float4>(base + dqk_base));
}

__device__ __forceinline__ void store_qk_lane(const float (&reg)[kQkPerLane], float* base,
                                              std::uint32_t dqk_base) {
    store_vec(base + dqk_base, load_vec<float4>(reg));
}

inline constexpr float kQkL2NormEps = 1.0e-6f;

struct RawQkLane {
    Bf16x4Pack bits;
    float value[kQkPerLane];
};

struct RawValueLane {
    __nv_bfloat16 bits;
    float value;
};

struct RawGatePair {
    uint2 bits;
    float g;
    float beta;
};

__device__ __forceinline__ RawQkLane load_raw_qk_lane(const __nv_bfloat16* base,
                                                      std::uint32_t dqk_base) {
    RawQkLane out;
    out.bits        = load_vec<Bf16x4Pack>(base + dqk_base);
    const float2 lo = bf16x2_to_float2(out.bits.pair[0]);
    const float2 hi = bf16x2_to_float2(out.bits.pair[1]);
    out.value[0]    = lo.x;
    out.value[1]    = lo.y;
    out.value[2]    = hi.x;
    out.value[3]    = hi.y;
    return out;
}

template <bool Normalize>
__device__ __forceinline__ void normalize_qk_lane(float (&value)[kQkPerLane], int lane) {
    if constexpr (Normalize) {
        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < kQkPerLane; ++i) { sum += value[i] * value[i]; }
        sum       = warp_reduce_sum(sum);
        float inv = lane == 0 ? rsqrtf(sum + kQkL2NormEps) : 0.0f;
        inv       = __shfl_sync(kFullWarpMask, inv, 0);
#pragma unroll
        for (int i = 0; i < kQkPerLane; ++i) { value[i] *= inv; }
    }
}

__device__ __forceinline__ RawValueLane load_value_lane(const __nv_bfloat16* base, int lane,
                                                        std::uint32_t dv_base) {
    RawValueLane out{__float2bfloat16(0.0f), 0.0f};
    if (lane < kDvPerWarp) {
        out.bits  = base[dv_base + lane];
        out.value = __bfloat162float(out.bits);
    }
    return out;
}

__device__ __forceinline__ RawGatePair load_source_gate(const float* g, const float* beta,
                                                        std::int64_t offset) {
    const float g_value    = g[offset];
    const float beta_value = beta[offset];
    return {make_uint2(__float_as_uint(g_value), __float_as_uint(beta_value)), g_value, beta_value};
}

__device__ __forceinline__ RawGatePair load_record_gate(const uint2* gate, std::int64_t offset) {
    const uint2 bits = load_vec<uint2>(gate + offset);
    return {bits, __uint_as_float(bits.x), __uint_as_float(bits.y)};
}

struct GdnInnovationStore {
    float* key             = nullptr;
    float* innovation      = nullptr;
    float* alpha           = nullptr;
    std::uint32_t dqk_base = 0;
    std::uint32_t dv_base  = 0;
    int lane               = 0;
};

template <bool StoreInnovation = false>
__device__ __forceinline__ void
apply_gdn_transition(float (&state)[kDvPerWarp][kQkPerLane], const float (&key)[kQkPerLane],
                     float v_local, float g, float beta, GdnInnovationStore store = {}) {
    const float alpha = expf(g);
    if constexpr (StoreInnovation) {
        if (store.key != nullptr) { store_qk_lane(key, store.key, store.dqk_base); }
        if (store.alpha != nullptr && store.lane == 0) { *store.alpha = alpha; }
    }

#pragma unroll
    for (int r = 0; r < kDvPerWarp; ++r) {
        float partial = 0.0f;
#pragma unroll
        for (int c = 0; c < kQkPerLane; ++c) { partial += state[r][c] * key[c]; }
        partial = warp_sum<kWarpSize>(partial);

        const float v_r   = __shfl_sync(0xffffffff, v_local, r, kWarpSize);
        const float delta = beta * (v_r - alpha * partial);
        if constexpr (StoreInnovation) {
            if (store.lane == r) { store.innovation[store.dv_base + r] = delta; }
        }

#pragma unroll
        for (int c = 0; c < kQkPerLane; ++c) { state[r][c] = alpha * state[r][c] + delta * key[c]; }
    }
}

template <bool Normalize>
__device__ __forceinline__ void readout_and_store(float (&state)[kDvPerWarp][kQkPerLane],
                                                  const __nv_bfloat16* query, __nv_bfloat16* output,
                                                  std::uint32_t dqk_base, std::uint32_t dv_base,
                                                  int lane, float scale) {
    RawQkLane q = load_raw_qk_lane(query, dqk_base);
    normalize_qk_lane<Normalize>(q.value, lane);

    float attn_val = 0.0f;
#pragma unroll
    for (int r = 0; r < kDvPerWarp; ++r) {
        float partial = 0.0f;
#pragma unroll
        for (int c = 0; c < kQkPerLane; ++c) { partial += state[r][c] * q.value[c]; }
        partial = warp_sum<kWarpSize>(partial);
        if (lane == r) { attn_val = partial; }
    }
    if (lane < kDvPerWarp) { output[dv_base + lane] = __float2bfloat16(attn_val * scale); }
}

template <bool NormalizeQK>
__global__ void __launch_bounds__(kWarpSize* kNumWarps, 2)
    recurrent_bf16_direct_kernel(const __nv_bfloat16* __restrict__ q,
                                 const __nv_bfloat16* __restrict__ k,
                                 const __nv_bfloat16* __restrict__ v, const float* __restrict__ g,
                                 const float* __restrict__ beta,
                                 const float* __restrict__ state_read,
                                 float* __restrict__ state_write, __nv_bfloat16* __restrict__ out,
                                 std::int32_t width, head_map heads, float scale) {
    const int lane           = threadIdx.x;
    const int warp_id        = threadIdx.y;
    const std::uint32_t h_v  = static_cast<std::uint32_t>(blockIdx.x);
    const std::uint32_t h_qk = static_cast<std::uint32_t>(heads.qk_head(static_cast<int>(h_v)));
    const std::uint32_t dv_base =
        static_cast<std::uint32_t>(blockIdx.z * kBlockDv + warp_id * kDvPerWarp);
    const std::uint32_t dqk_base = static_cast<std::uint32_t>(lane * kQkPerLane);
    const float* read_h = state_read + static_cast<std::int64_t>(h_v) * kStateDim * kStateDim;

    __align__(16) float state[kDvPerWarp][kQkPerLane];
#pragma unroll
    for (int r = 0; r < kDvPerWarp; ++r) {
        load_qk_lane(state[r], read_h + static_cast<std::int64_t>(dv_base + r) * kStateDim,
                     dqk_base);
    }

    RawQkLane key = load_raw_qk_lane(k + static_cast<std::int64_t>(h_qk) * kStateDim, dqk_base);
    normalize_qk_lane<NormalizeQK>(key.value, lane);
    for (std::int32_t token = 0; token < width; ++token) {
        const std::int64_t column = token;
        const RawGatePair gate    = load_source_gate(g, beta, column * heads.H_v + h_v);
        const RawValueLane value =
            load_value_lane(v + (column * heads.H_v + h_v) * kStateDim, lane, dv_base);
        apply_gdn_transition(state, key.value, value.value, gate.g, gate.beta);

        if (token + 1 < width) {
            key = load_raw_qk_lane(k + ((column + 1) * heads.H_qk + h_qk) * kStateDim, dqk_base);
            normalize_qk_lane<NormalizeQK>(key.value, lane);
        }

        readout_and_store<NormalizeQK>(state, q + (column * heads.H_qk + h_qk) * kStateDim,
                                       out + (column * heads.H_v + h_v) * kStateDim, dqk_base,
                                       dv_base, lane, scale);
    }

    float* write_h = state_write + static_cast<std::int64_t>(h_v) * kStateDim * kStateDim;
#pragma unroll
    for (int r = 0; r < kDvPerWarp; ++r) {
        store_qk_lane(state[r], write_h + static_cast<std::int64_t>(dv_base + r) * kStateDim,
                      dqk_base);
    }
}

enum class RecurrentMode {
    Snapshot,
    Record,
    Fold,
};

struct RecurrentCoordinates {
    int lane;
    int warp;
    std::int32_t batch;
    std::int32_t layer;
    std::int32_t state_tile;
    std::uint32_t value_head;
    std::uint32_t qk_head;
    std::uint32_t dv_base;
    std::uint32_t dqk_base;
};

__device__ __forceinline__ RecurrentCoordinates make_coordinates(std::int32_t batch,
                                                                 std::int32_t layer,
                                                                 std::int32_t state_tile,
                                                                 head_map heads,
                                                                 int block_dv = kBlockDv) {
    const int lane                 = threadIdx.x;
    const int warp                 = threadIdx.y;
    const std::uint32_t value_head = static_cast<std::uint32_t>(blockIdx.x);
    const std::uint32_t qk_head =
        static_cast<std::uint32_t>(heads.qk_head(static_cast<int>(value_head)));
    const std::uint32_t dv_base =
        static_cast<std::uint32_t>(state_tile * block_dv + warp * kDvPerWarp);
    return {lane,    warp,       batch,
            layer,   state_tile, value_head,
            qk_head, dv_base,    static_cast<std::uint32_t>(lane * kQkPerLane)};
}

template <bool Batched, bool Masked>
struct SnapshotAccess {
    static constexpr bool kParentIndexed = false;

    const __nv_bfloat16* q;
    const __nv_bfloat16* k;
    const __nv_bfloat16* v;
    const float* g;
    const float* beta;
    float* states;
    const std::int32_t* valid_columns;
    const std::int32_t* initial_slots;
    const std::int32_t* snapshot_bases;
    __nv_bfloat16* out;
    head_map heads;
    std::int32_t width;
    std::int64_t state_slot_stride;
    float scale;

    __device__ __forceinline__ RecurrentCoordinates coordinates() const {
        const std::int32_t batch = Batched ? static_cast<std::int32_t>(blockIdx.y) : 0;
        return make_coordinates(batch, 0, static_cast<std::int32_t>(blockIdx.z), heads);
    }

    __device__ __forceinline__ std::int32_t
    active_columns(const RecurrentCoordinates& coord) const {
        if constexpr (Masked) { return valid_columns[coord.batch]; }
        return width;
    }

    __device__ __forceinline__ std::int64_t column(const RecurrentCoordinates& coord,
                                                   std::int32_t token) const {
        return static_cast<std::int64_t>(coord.batch) * width + token;
    }

    __device__ __forceinline__ const float*
    state_read_base(const RecurrentCoordinates& coord) const {
        return states + static_cast<std::int64_t>(initial_slots[coord.batch]) * state_slot_stride +
               static_cast<std::int64_t>(coord.value_head) * kStateDim * kStateDim;
    }

    __device__ __forceinline__ const __nv_bfloat16* key_ptr(const RecurrentCoordinates& coord,
                                                            std::int32_t token) const {
        return k + (column(coord, token) * heads.H_qk + coord.qk_head) * kStateDim;
    }

    __device__ __forceinline__ const __nv_bfloat16* value_ptr(const RecurrentCoordinates& coord,
                                                              std::int32_t token) const {
        return v + (column(coord, token) * heads.H_v + coord.value_head) * kStateDim;
    }

    __device__ __forceinline__ RawGatePair load_gate(const RecurrentCoordinates& coord,
                                                     std::int32_t token) const {
        return load_source_gate(g, beta, column(coord, token) * heads.H_v + coord.value_head);
    }

    __device__ __forceinline__ const __nv_bfloat16* query_ptr(const RecurrentCoordinates& coord,
                                                              std::int32_t token) const {
        return q + (column(coord, token) * heads.H_qk + coord.qk_head) * kStateDim;
    }

    __device__ __forceinline__ __nv_bfloat16* output_ptr(const RecurrentCoordinates& coord,
                                                         std::int32_t token) const {
        return out + (column(coord, token) * heads.H_v + coord.value_head) * kStateDim;
    }

    __device__ __forceinline__ void
    store_snapshot(const RecurrentCoordinates& coord, std::int32_t token,
                   const float (&state)[kDvPerWarp][kQkPerLane]) const {
        float* snapshot =
            states +
            static_cast<std::int64_t>(snapshot_bases[coord.batch] + token) * state_slot_stride +
            static_cast<std::int64_t>(coord.value_head) * kStateDim * kStateDim;
#pragma unroll
        for (int r = 0; r < kDvPerWarp; ++r) {
            store_qk_lane(state[r],
                          snapshot + static_cast<std::int64_t>(coord.dv_base + r) * kStateDim,
                          coord.dqk_base);
        }
    }
};

template <bool Masked, bool ParentIndexed = false, int NumWarps = kNumWarps>
struct RecordAccess {
    static constexpr bool kParentIndexed = ParentIndexed;
    static constexpr int kWarps          = NumWarps;
    static constexpr int kDv             = NumWarps * kDvPerWarp;

    const __nv_bfloat16* q;
    const __nv_bfloat16* k;
    const __nv_bfloat16* v;
    const float* g;
    const float* beta;
    const float* states;
    const std::int32_t* valid_columns;
    const std::int32_t* initial_slots;
    // Non-null exactly when ParentIndexed: gated_delta_net_tree_schedule output.
    const std::int32_t* tree_schedule;
    __nv_bfloat16* key_record;
    __nv_bfloat16* value_record;
    uint2* gate_record;
    __nv_bfloat16* out;
    head_map heads;
    std::int32_t width;
    std::int64_t state_slot_stride;
    float scale;

    __device__ __forceinline__ RecurrentCoordinates coordinates() const {
        return make_coordinates(static_cast<std::int32_t>(blockIdx.y), 0,
                                static_cast<std::int32_t>(blockIdx.z), heads, kDv);
    }

    __device__ __forceinline__ std::int32_t
    active_columns(const RecurrentCoordinates& coord) const {
        if constexpr (Masked) { return valid_columns[coord.batch]; }
        return width;
    }

    __device__ __forceinline__ std::int64_t column(const RecurrentCoordinates& coord,
                                                   std::int32_t token) const {
        return static_cast<std::int64_t>(coord.batch) * width + token;
    }

    __device__ __forceinline__ const float*
    state_read_base(const RecurrentCoordinates& coord) const {
        return states + static_cast<std::int64_t>(initial_slots[coord.batch]) * state_slot_stride +
               static_cast<std::int64_t>(coord.value_head) * kStateDim * kStateDim;
    }

    __device__ __forceinline__ const __nv_bfloat16* key_ptr(const RecurrentCoordinates& coord,
                                                            std::int32_t token) const {
        return k + (column(coord, token) * heads.H_qk + coord.qk_head) * kStateDim;
    }

    __device__ __forceinline__ const __nv_bfloat16* value_ptr(const RecurrentCoordinates& coord,
                                                              std::int32_t token) const {
        return v + (column(coord, token) * heads.H_v + coord.value_head) * kStateDim;
    }

    __device__ __forceinline__ RawGatePair load_gate(const RecurrentCoordinates& coord,
                                                     std::int32_t token) const {
        return load_source_gate(g, beta, column(coord, token) * heads.H_v + coord.value_head);
    }

    __device__ __forceinline__ const __nv_bfloat16* query_ptr(const RecurrentCoordinates& coord,
                                                              std::int32_t token) const {
        return q + (column(coord, token) * heads.H_qk + coord.qk_head) * kStateDim;
    }

    __device__ __forceinline__ __nv_bfloat16* output_ptr(const RecurrentCoordinates& coord,
                                                         std::int32_t token) const {
        return out + (column(coord, token) * heads.H_v + coord.value_head) * kStateDim;
    }

    __device__ __forceinline__ void store_key(const RecurrentCoordinates& coord, std::int32_t token,
                                              const RawQkLane& raw) const {
        if (coord.state_tile == 0 && coord.warp == 0 &&
            static_cast<int>(coord.value_head) % heads.group_size() == 0) {
            __nv_bfloat16* destination =
                key_record + (column(coord, token) * heads.H_qk + coord.qk_head) * kStateDim;
            store_vec(destination + coord.dqk_base, raw.bits);
        }
    }

    __device__ __forceinline__ void store_value(const RecurrentCoordinates& coord,
                                                std::int32_t token, const RawValueLane& raw) const {
        if (coord.lane < kDvPerWarp) {
            __nv_bfloat16* destination =
                value_record + (column(coord, token) * heads.H_v + coord.value_head) * kStateDim;
            destination[coord.dv_base + coord.lane] = raw.bits;
        }
    }

    __device__ __forceinline__ void store_gate(const RecurrentCoordinates& coord,
                                               std::int32_t token, const RawGatePair& raw) const {
        if (coord.state_tile == 0 && coord.warp == 0 && coord.lane == 0) {
            gate_record[column(coord, token) * heads.H_v + coord.value_head] = raw.bits;
        }
    }
};

template <int Layers, int QkHeads, int ValueHeads, int ConvChannels>
struct FoldGeometry {
    static constexpr int kLayers       = Layers;
    static constexpr int kQkHeads      = QkHeads;
    static constexpr int kValueHeads   = ValueHeads;
    static constexpr int kConvChannels = ConvChannels;
    static_assert(ValueHeads % QkHeads == 0);
    static_assert(ConvChannels % 128 == 0);
};

using FoldGeometry48x48 = FoldGeometry<48, 16, 48, 10240>;
using FoldGeometry30x32 = FoldGeometry<30, 16, 32, 8192>;

template <class Geometry>
struct FoldAccess {
    static constexpr bool kParentIndexed = false;

    const __nv_bfloat16* key_record;
    const __nv_bfloat16* value_record;
    const uint2* gate_record;
    const __nv_bfloat16* conv_record;
    float* recurrent_layer0;
    __nv_bfloat16* conv_layer0;
    std::int64_t recurrent_layer_stride;
    std::int64_t conv_layer_stride;
    std::int32_t record_capacity;
    std::int32_t width;
    GdnReplayFoldKernelRows rows;

    __device__ __forceinline__ RecurrentCoordinates coordinates() const {
        const std::int32_t batch       = static_cast<std::int32_t>(blockIdx.y);
        const std::int32_t layer_tile  = static_cast<std::int32_t>(blockIdx.z);
        const int lane                 = threadIdx.x;
        const int warp                 = threadIdx.y;
        const std::int32_t state_tile  = layer_tile & 7;
        const std::uint32_t value_head = static_cast<std::uint32_t>(blockIdx.x);
        constexpr std::uint32_t kGroup = Geometry::kValueHeads / Geometry::kQkHeads;
        const std::uint32_t qk_head    = value_head / kGroup;
        const std::uint32_t dv_base =
            static_cast<std::uint32_t>(state_tile * kBlockDv + warp * kDvPerWarp);
        return {lane,
                warp,
                batch,
                layer_tile >> 3,
                state_tile,
                value_head,
                qk_head,
                dv_base,
                static_cast<std::uint32_t>(lane * kQkPerLane)};
    }

    __device__ __forceinline__ std::int32_t sequence_column(const RecurrentCoordinates& coord,
                                                            std::int32_t step) const {
        const std::int32_t path_length = rows.row[coord.batch].path_length;
        if (path_length > 0) { return rows.row[coord.batch].path[step]; }
        return step;
    }

    __device__ __forceinline__ std::int32_t
    active_columns(const RecurrentCoordinates& coord) const {
        const std::int32_t path_length = rows.row[coord.batch].path_length;
        if (path_length >= 0) { return path_length; }
        return rows.row[coord.batch].commit_columns;
    }

    __device__ __forceinline__ std::int64_t record_outer(const RecurrentCoordinates& coord) const {
        return static_cast<std::int64_t>(coord.layer) * record_capacity + coord.batch;
    }

    __device__ __forceinline__ float* state_read_base(const RecurrentCoordinates& coord) const {
        const std::int64_t slot_stride =
            static_cast<std::int64_t>(Geometry::kValueHeads) * kStateDim * kStateDim;
        return recurrent_layer0 + static_cast<std::int64_t>(coord.layer) * recurrent_layer_stride +
               static_cast<std::int64_t>(rows.row[coord.batch].linear_state_slot) * slot_stride +
               static_cast<std::int64_t>(coord.value_head) * kStateDim * kStateDim;
    }

    __device__ __forceinline__ const __nv_bfloat16* key_ptr(const RecurrentCoordinates& coord,
                                                            std::int32_t token) const {
        const std::int64_t column = record_outer(coord) * width + sequence_column(coord, token);
        return key_record + (column * Geometry::kQkHeads + coord.qk_head) * kStateDim;
    }

    __device__ __forceinline__ const __nv_bfloat16* value_ptr(const RecurrentCoordinates& coord,
                                                              std::int32_t token) const {
        const std::int64_t column = record_outer(coord) * width + sequence_column(coord, token);
        return value_record + (column * Geometry::kValueHeads + coord.value_head) * kStateDim;
    }

    __device__ __forceinline__ RawGatePair load_gate(const RecurrentCoordinates& coord,
                                                     std::int32_t token) const {
        const std::int64_t column = record_outer(coord) * width + sequence_column(coord, token);
        return load_record_gate(gate_record, column * Geometry::kValueHeads + coord.value_head);
    }

    __device__ __forceinline__ void
    store_final_state(const RecurrentCoordinates& coord,
                      const float (&state)[kDvPerWarp][kQkPerLane]) const {
        float* destination = state_read_base(coord);
#pragma unroll
        for (int r = 0; r < kDvPerWarp; ++r) {
            store_qk_lane(state[r],
                          destination + static_cast<std::int64_t>(coord.dv_base + r) * kStateDim,
                          coord.dqk_base);
        }
    }

    __device__ __forceinline__ void publish_final_conv_history(const RecurrentCoordinates& coord,
                                                               std::int32_t commit) const {
        const std::int32_t tile_block =
            static_cast<std::int32_t>(coord.value_head) * 8 + coord.state_tile;
        if (tile_block >= Geometry::kConvChannels / 128) { return; }

        const std::int32_t tid     = coord.warp * kWarpSize + coord.lane;
        const std::int32_t channel = tile_block * 128 + tid;
        __nv_bfloat16* history =
            conv_layer0 + static_cast<std::int64_t>(coord.layer) * conv_layer_stride +
            static_cast<std::int64_t>(rows.row[coord.batch].linear_state_slot) *
                (3LL * Geometry::kConvChannels) +
            channel;
        const __nv_bfloat16* record =
            conv_record + record_outer(coord) * width * Geometry::kConvChannels + channel;

        const std::int32_t i0 =
            sequence_column(coord, commit == 1 ? 0 : (commit == 2 ? 0 : commit - 3));
        const std::int32_t i1 =
            sequence_column(coord, commit == 1 ? 0 : (commit == 2 ? 0 : commit - 2));
        const std::int32_t i2 =
            sequence_column(coord, commit == 1 ? 0 : (commit == 2 ? 1 : commit - 1));

        __nv_bfloat16 h0;
        __nv_bfloat16 h1;
        __nv_bfloat16 h2;
        if (commit == 1) {
            h0 = history[Geometry::kConvChannels];
            h1 = history[2LL * Geometry::kConvChannels];
            h2 = record[static_cast<std::int64_t>(i2) * Geometry::kConvChannels];
        } else if (commit == 2) {
            h0 = history[2LL * Geometry::kConvChannels];
            h1 = record[static_cast<std::int64_t>(i1) * Geometry::kConvChannels];
            h2 = record[static_cast<std::int64_t>(i2) * Geometry::kConvChannels];
        } else {
            h0 = record[static_cast<std::int64_t>(i0) * Geometry::kConvChannels];
            h1 = record[static_cast<std::int64_t>(i1) * Geometry::kConvChannels];
            h2 = record[static_cast<std::int64_t>(i2) * Geometry::kConvChannels];
        }
        history[0]                             = h0;
        history[Geometry::kConvChannels]       = h1;
        history[2LL * Geometry::kConvChannels] = h2;
    }
};

template <RecurrentMode Mode, bool NormalizeInputs, bool History = false, class Access>
__device__ __forceinline__ void recurrent_bf16_body(const Access& access,
                                                    const RecurrentCoordinates& coord,
                                                    std::int32_t width, std::int32_t valid) {
    if constexpr (Mode == RecurrentMode::Record && Access::kParentIndexed) {
        // Execute the row's precompiled step list (see gated_delta_net_tree_schedule). Each
        // step resumes from the live registers, the checkpoint, or a branch slot, applies one
        // column's transition, optionally saves the state to a branch slot, and, when it emits,
        // publishes that column's records and output. Each thread reads back only the slot words
        // it wrote, so the loop needs no barrier.
        constexpr int kBranchSlots = 3;
        constexpr int kStateWords  = kDvPerWarp * kQkPerLane;
        constexpr int kThreads     = kWarpSize * kNumWarps;
        constexpr int kResumeRoot  = 1;
        constexpr int kEmit        = 1 << 6;
        __shared__ float saved[kBranchSlots][kStateWords][kThreads];
        __shared__ std::int32_t steps_sm[kGdnTreeScheduleWords];
        const std::int32_t* schedule =
            access.tree_schedule + static_cast<std::int64_t>(coord.batch) * kGdnTreeScheduleWords;
        const int thread =
            static_cast<int>(threadIdx.y) * kWarpSize + static_cast<int>(threadIdx.x);
        const int steps = schedule[0];
        for (int i = thread; i < steps; i += kThreads) { steps_sm[i] = schedule[1 + i]; }
        __syncthreads();
        const float* initial = access.state_read_base(coord);
        __align__(16) float root[kDvPerWarp][kQkPerLane];
        __align__(16) float state[kDvPerWarp][kQkPerLane];
#pragma unroll
        for (int r = 0; r < kDvPerWarp; ++r) {
            load_qk_lane(root[r],
                         initial + static_cast<std::int64_t>(coord.dv_base + r) * kStateDim,
                         coord.dqk_base);
        }
        if constexpr (History) { access.initialize_history(root, coord); }
        // Column inputs do not depend on the recurrent state, so the next step's key, gate, and
        // value loads are issued before the current step's transition.
        std::int32_t column   = steps_sm[0] & 0xff;
        RawQkLane next_key    = load_raw_qk_lane(access.key_ptr(coord, column), coord.dqk_base);
        RawGatePair next_gate = access.load_gate(coord, column);
        RawValueLane next_value =
            load_value_lane(access.value_ptr(coord, column), coord.lane, coord.dv_base);
        for (int step = 0; step < steps; ++step) {
            RawQkLane key            = next_key;
            const RawGatePair gate   = next_gate;
            const RawValueLane value = next_value;
            const int code           = steps_sm[step] >> 8;
            const std::int32_t token = column;
            if (step + 1 < steps) {
                column    = steps_sm[step + 1] & 0xff;
                next_key  = load_raw_qk_lane(access.key_ptr(coord, column), coord.dqk_base);
                next_gate = access.load_gate(coord, column);
                next_value =
                    load_value_lane(access.value_ptr(coord, column), coord.lane, coord.dv_base);
            }
            const int resume = code & 7;
            if (resume == kResumeRoot) {
#pragma unroll
                for (int r = 0; r < kDvPerWarp; ++r) {
#pragma unroll
                    for (int c = 0; c < kQkPerLane; ++c) { state[r][c] = root[r][c]; }
                }
            } else if (resume >= 2) {
#pragma unroll
                for (int r = 0; r < kDvPerWarp; ++r) {
#pragma unroll
                    for (int c = 0; c < kQkPerLane; ++c) {
                        state[r][c] = saved[resume - 2][r * kQkPerLane + c][thread];
                    }
                }
            }
            const bool emit = (code & kEmit) != 0;
            if (emit) {
                access.store_key(coord, token, key);
                access.store_value(coord, token, value);
                access.store_gate(coord, token, gate);
            }
            normalize_qk_lane<NormalizeInputs>(key.value, coord.lane);
            if constexpr (History) {
                apply_gdn_transition<true>(state, key.value, value.value, gate.g, gate.beta,
                                           access.innovation_store(coord, token));
            } else {
                apply_gdn_transition(state, key.value, value.value, gate.g, gate.beta);
            }
            const int save = (code >> 3) & 7;
            if (save != 0) {
#pragma unroll
                for (int r = 0; r < kDvPerWarp; ++r) {
#pragma unroll
                    for (int c = 0; c < kQkPerLane; ++c) {
                        saved[save - 1][r * kQkPerLane + c][thread] = state[r][c];
                    }
                }
            }
            if (emit) {
                readout_and_store<NormalizeInputs>(state, access.query_ptr(coord, token),
                                                   access.output_ptr(coord, token), coord.dqk_base,
                                                   coord.dv_base, coord.lane, access.scale);
            }
        }
        if (coord.lane < kDvPerWarp) {
            for (std::int32_t token = valid; token < width; ++token) {
                access.output_ptr(coord, token)[coord.dv_base + coord.lane] =
                    __float2bfloat16(0.0f);
            }
        }
        return;
    }

    if constexpr (Mode == RecurrentMode::Fold) {
        if (valid == 0) { return; }
    }

    const float* initial = access.state_read_base(coord);
    __align__(16) float state[kDvPerWarp][kQkPerLane];
#pragma unroll
    for (int r = 0; r < kDvPerWarp; ++r) {
        load_qk_lane(state[r], initial + static_cast<std::int64_t>(coord.dv_base + r) * kStateDim,
                     coord.dqk_base);
    }

    if constexpr (History) { access.initialize_history(state, coord); }

    RawQkLane key = load_raw_qk_lane(access.key_ptr(coord, 0), coord.dqk_base);
    if constexpr (Mode == RecurrentMode::Record) { access.store_key(coord, 0, key); }
    normalize_qk_lane<NormalizeInputs>(key.value, coord.lane);

    for (std::int32_t token = 0; token < valid; ++token) {
        const RawGatePair gate = access.load_gate(coord, token);
        const RawValueLane value =
            load_value_lane(access.value_ptr(coord, token), coord.lane, coord.dv_base);
        if constexpr (Mode == RecurrentMode::Record) {
            access.store_value(coord, token, value);
            access.store_gate(coord, token, gate);
        }

        if constexpr (History) {
            apply_gdn_transition<true>(state, key.value, value.value, gate.g, gate.beta,
                                       access.innovation_store(coord, token));
        } else {
            apply_gdn_transition(state, key.value, value.value, gate.g, gate.beta);
        }

        if (token + 1 < valid) {
            key = load_raw_qk_lane(access.key_ptr(coord, token + 1), coord.dqk_base);
            if constexpr (Mode == RecurrentMode::Record) {
                access.store_key(coord, token + 1, key);
            }
            normalize_qk_lane<NormalizeInputs>(key.value, coord.lane);
        }

        if constexpr (Mode != RecurrentMode::Fold) {
            readout_and_store<NormalizeInputs>(state, access.query_ptr(coord, token),
                                               access.output_ptr(coord, token), coord.dqk_base,
                                               coord.dv_base, coord.lane, access.scale);
        }
        if constexpr (Mode == RecurrentMode::Snapshot) {
            access.store_snapshot(coord, token, state);
        }
    }

    if constexpr (Mode == RecurrentMode::Fold) {
        access.store_final_state(coord, state);
        access.publish_final_conv_history(coord, valid);
    } else {
        if (coord.lane < kDvPerWarp) {
            for (std::int32_t token = valid; token < width; ++token) {
                access.output_ptr(coord, token)[coord.dv_base + coord.lane] =
                    __float2bfloat16(0.0f);
            }
        }
    }
}

template <bool NormalizeInputs, bool Batched, bool Masked>
__global__ void __launch_bounds__(kWarpSize* kNumWarps, 2)
    recurrent_snapshot_kernel(SnapshotAccess<Batched, Masked> access) {
    static_assert(!Masked || Batched);
    const RecurrentCoordinates coord = access.coordinates();
    recurrent_bf16_body<RecurrentMode::Snapshot, NormalizeInputs>(access, coord, access.width,
                                                                  access.active_columns(coord));
}

template <bool Masked, bool ParentIndexed, int NumWarps>
__global__ void __launch_bounds__(kWarpSize* NumWarps, 2)
    recurrent_record_kernel(RecordAccess<Masked, ParentIndexed, NumWarps> access) {
    const RecurrentCoordinates coord = access.coordinates();
    recurrent_bf16_body<RecurrentMode::Record, true>(access, coord, access.width,
                                                     access.active_columns(coord));
}

template <class Geometry>
__global__ void __launch_bounds__(kWarpSize* kNumWarps, 2)
    recurrent_fold_kernel(const __grid_constant__ FoldAccess<Geometry> access) {
    const RecurrentCoordinates coord = access.coordinates();
    recurrent_bf16_body<RecurrentMode::Fold, true>(access, coord, access.width,
                                                   access.active_columns(coord));
}



} // namespace ninfer::ops::detail::gated_delta_net
