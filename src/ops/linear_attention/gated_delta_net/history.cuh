#pragma once

#include "ops/linear_attention/gated_delta_net/recurrent.cuh"

namespace ninfer::ops::detail::gated_delta_net {

struct HistoryPlanes {
    float* key;
    float* innovation;
    float* alpha;
    std::int32_t* counts;
    float* provisional_key;
    float* provisional_innovation;
    float* provisional_alpha;
    std::int32_t slots;
    std::int32_t width;

    __device__ __forceinline__ std::int64_t head(int layer, int slot, int entry,
                                                 int value_head) const {
        return ((static_cast<std::int64_t>(layer) * slots + slot) * 4 + entry) * 48 + value_head;
    }

    __device__ __forceinline__ std::int64_t key_head(int layer, int slot, int entry,
                                                     int qk_head) const {
        return ((static_cast<std::int64_t>(layer) * slots + slot) * 4 + entry) * 16 + qk_head;
    }

    __device__ __forceinline__ std::int64_t provisional_head(int layer, int row, int token,
                                                             int value_head) const {
        return ((static_cast<std::int64_t>(layer) * slots + row) * width + token) * 48 + value_head;
    }
};

// Every thread owns four key columns of four value rows. Retained length is shared across all
// layers and immutable during record/commit; a later ordered kernel publishes the new length.
// Sequential alpha*S+u*k^T handles represented zero alpha without division or dropped terms.
__device__ __forceinline__ void apply_history(float (&state)[kDvPerWarp][kQkPerLane],
                                              const RecurrentCoordinates& coord,
                                              HistoryPlanes planes, int slot) {
    const int count = planes.counts[slot];
    for (int entry = 0; entry < count; ++entry) {
        const auto head = planes.head(coord.layer, slot, entry, coord.value_head);
        float key[kQkPerLane];
        load_qk_lane(key,
                     planes.key + planes.key_head(coord.layer, slot, entry, coord.qk_head) * 128,
                     coord.dqk_base);
        const float alpha = planes.alpha[head];
#pragma unroll
        for (int row = 0; row < kDvPerWarp; ++row) {
            const float innovation = planes.innovation[head * 128 + coord.dv_base + row];
#pragma unroll
            for (int column = 0; column < kQkPerLane; ++column) {
                state[row][column] = alpha * state[row][column] + innovation * key[column];
            }
        }
    }
}

template <bool Masked, bool ParentIndexed>
struct HistoryRecordAccess : RecordAccess<Masked, ParentIndexed, kNumWarps> {
    HistoryPlanes history;

    __device__ __forceinline__ void initialize_history(float (&state)[kDvPerWarp][kQkPerLane],
                                                       const RecurrentCoordinates& coord) const {
        apply_history(state, coord, history, this->initial_slots[coord.batch]);
    }

    __device__ __forceinline__ GdnInnovationStore
    innovation_store(const RecurrentCoordinates& coord, int token) const {
        const auto head        = history.provisional_head(0, coord.batch, token, coord.value_head);
        const bool first_strip = coord.state_tile == 0 && coord.warp == 0;
        return {
            .key        = first_strip && coord.value_head % 3 == 0
                              ? history.provisional_key + (head / 48 * 16 + coord.qk_head) * 128
                              : nullptr,
            .innovation = history.provisional_innovation + head * 128,
            .alpha      = first_strip ? history.provisional_alpha + head : nullptr,
            .dqk_base   = coord.dqk_base,
            .dv_base    = coord.dv_base,
            .lane       = coord.lane,
        };
    }
};

// Native grid [Hv,B,8] and128threads retain the current chain/tree traversal and raw publication.
// Only the first value head/strip0/warp0 writes each shared normalized key, and lane0 writes its
// alpha; innovation value tiles have disjoint writers. Tree traversal uses the native branch
// storage mapping.
template <bool Masked, bool ParentIndexed>
__global__
__launch_bounds__(128,
                  2) void history_record_kernel(HistoryRecordAccess<Masked, ParentIndexed> access) {
    const auto coord = access.coordinates();
    recurrent_bf16_body<RecurrentMode::Record, true, true>(access, coord, access.width,
                                                           access.active_columns(coord));
}

using HistoryFold = FoldAccess<FoldGeometry<48, 16, 48, 10240>>;

// Grid [Hv,B,48*8], 128 threads owns disjoint layer/state/value strips. Append copies derived
// records into the owning slot; flush applies retained and selected ancestor innovations and
// publishes the complete FP32 checkpoint. Conv3 uses the unchanged raw-record path transition.
// Counts are read-only until the following one-CTA publication kernel finishes all-layer writes.
__global__ __launch_bounds__(128, 2) void history_commit_kernel(HistoryFold fold,
                                                                HistoryPlanes history,
                                                                bool force_flush) {
    const auto coord   = fold.coordinates();
    const auto& row    = fold.rows.row[coord.batch];
    const int accepted = fold.active_columns(coord);
    const int slot     = row.linear_state_slot;
    const int retained = history.counts[slot];
    if (accepted == 0 && (!force_flush || retained == 0)) { return; }
    if (accepted > 0) { fold.publish_final_conv_history(coord, accepted); }
    if (force_flush || retained + accepted >= 4) {
        __align__(16) float state[kDvPerWarp][kQkPerLane];
        const float* checkpoint = fold.state_read_base(coord);
#pragma unroll
        for (int value = 0; value < kDvPerWarp; ++value) {
            load_qk_lane(state[value],
                         checkpoint + static_cast<std::int64_t>(coord.dv_base + value) * 128,
                         coord.dqk_base);
        }
        apply_history(state, coord, history, slot);
        for (int entry = 0; entry < accepted; ++entry) {
            const int token = fold.sequence_column(coord, entry);
            const auto source =
                history.provisional_head(coord.layer, coord.batch, token, coord.value_head);
            float key[kQkPerLane];
            load_qk_lane(key, history.provisional_key + (source / 48 * 16 + coord.qk_head) * 128,
                         coord.dqk_base);
            const float alpha = history.provisional_alpha[source];
#pragma unroll
            for (int value = 0; value < kDvPerWarp; ++value) {
                const float innovation =
                    history.provisional_innovation[source * 128 + coord.dv_base + value];
#pragma unroll
                for (int column = 0; column < kQkPerLane; ++column) {
                    state[value][column] = alpha * state[value][column] + innovation * key[column];
                }
            }
        }
        fold.store_final_state(coord, state);
    } else {
        for (int entry = 0; entry < accepted; ++entry) {
            const int token = fold.sequence_column(coord, entry);
            const auto source =
                history.provisional_head(coord.layer, coord.batch, token, coord.value_head);
            const auto destination =
                history.head(coord.layer, slot, retained + entry, coord.value_head);
            if (coord.state_tile == 0 && coord.warp == 0) {
                if (coord.value_head % 3 == 0) {
                    float key[kQkPerLane];
                    load_qk_lane(key,
                                 history.provisional_key + (source / 48 * 16 + coord.qk_head) * 128,
                                 coord.dqk_base);
                    store_qk_lane(key,
                                  history.key + history.key_head(coord.layer, slot,
                                                                 retained + entry, coord.qk_head) *
                                                    128,
                                  coord.dqk_base);
                }
                if (coord.lane == 0) {
                    history.alpha[destination] = history.provisional_alpha[source];
                }
            }
            if (coord.lane < kDvPerWarp) {
                history.innovation[destination * 128 + coord.dv_base + coord.lane] =
                    history.provisional_innovation[source * 128 + coord.dv_base + coord.lane];
            }
        }
    }
}

// One thread per distinct active slot; the preceding kernel is the global publication barrier.
__global__ void history_count_kernel(HistoryPlanes history, GdnReplayFoldKernelRows rows, int batch,
                                     bool force_flush) {
    const int row = static_cast<int>(threadIdx.x);
    if (row >= batch) { return; }
    const auto& control  = rows.row[row];
    const int accepted   = control.path_length < 0 ? control.commit_columns : control.path_length;
    const int slot       = control.linear_state_slot;
    const int total      = history.counts[slot] + accepted;
    history.counts[slot] = force_flush || total >= 4 ? 0 : total;
}

} // namespace ninfer::ops::detail::gated_delta_net
