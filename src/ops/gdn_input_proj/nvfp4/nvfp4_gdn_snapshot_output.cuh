#pragma once

#include "core/tensor.h"
#include "ops/gdn_input_proj/gdn_conv_snapshot.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr std::int32_t kNvfp4GdnSnapshotQueryRows = 2048;
inline constexpr std::int32_t kNvfp4GdnSnapshotKeyRows   = 2048;
inline constexpr std::int32_t kNvfp4GdnSnapshotValueRows = 6144;
inline constexpr std::int32_t kNvfp4GdnSnapshotChannels =
    kNvfp4GdnSnapshotQueryRows + kNvfp4GdnSnapshotKeyRows + kNvfp4GdnSnapshotValueRows;
inline constexpr std::int32_t kNvfp4GdnSnapshotZRows = 6144;
inline constexpr std::int32_t kNvfp4GdnSnapshotParentRows =
    kNvfp4GdnSnapshotChannels + kNvfp4GdnSnapshotZRows;

template <int Tokens>
struct Nvfp4GdnSnapshotOutput {
    GdnConvSnapshotEpilogue conv;
    __nv_bfloat16* z;

    __device__ __forceinline__ void store_row(std::int32_t parent_row,
                                              const float (&projected)[Tokens]) const {
        if (parent_row < kNvfp4GdnSnapshotChannels) {
            conv.store(parent_row, projected);
            return;
        }
#pragma unroll
        for (int token = 0; token < Tokens; ++token) {
            z[static_cast<std::int64_t>(token) * kNvfp4GdnSnapshotZRows + parent_row -
              kNvfp4GdnSnapshotChannels] = __float2bfloat16_rn(projected[token]);
        }
    }

    __device__ __forceinline__ void store(std::int32_t parent_row, std::int32_t token,
                                          float projected) const {
        static_assert(Tokens == 1);
        (void)token;
        const float row[1]{projected};
        store_row(parent_row, row);
    }
};

template <int Tokens>
Nvfp4GdnSnapshotOutput<Tokens>
make_nvfp4_gdn_snapshot_output(const Tensor& conv_weight, Tensor& conv_states,
                               const Tensor& valid_columns, const Tensor& initial_slot,
                               const Tensor& snapshot_base_slot, Tensor& query, Tensor& key,
                               Tensor& value, Tensor& z) {
    return {
        {
            static_cast<const __nv_bfloat16*>(conv_weight.data),
            static_cast<__nv_bfloat16*>(conv_states.data),
            static_cast<const std::int32_t*>(initial_slot.data),
            static_cast<const std::int32_t*>(snapshot_base_slot.data),
            valid_columns.data == nullptr ? nullptr
                                          : static_cast<const std::int32_t*>(valid_columns.data),
            static_cast<__nv_bfloat16*>(query.data),
            static_cast<__nv_bfloat16*>(key.data),
            static_cast<__nv_bfloat16*>(value.data),
            kNvfp4GdnSnapshotChannels,
            kNvfp4GdnSnapshotQueryRows,
            kNvfp4GdnSnapshotKeyRows,
            kNvfp4GdnSnapshotValueRows,
            0,
        },
        static_cast<__nv_bfloat16*>(z.data),
    };
}

} // namespace ninfer::ops::detail
