#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_kernels.h"

#include "core/device.h"
#include "core/pdl.cuh"
#include "ops/common/math.h"
#include "ops/gdn_input_proj/gdn_conv_snapshot.cuh"
#include "ops/linear/q4/q4_rowsplit_gemm_simt.cuh"
#include "ops/linear/q4/q4_rowsplit_gemv.cuh"
#include "ops/linear/q5/q5_rowsplit_gemm_simt.cuh"
#include "ops/linear/q5/q5_rowsplit_gemv.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kHidden      = 5120;
constexpr int kQueryRows   = 2048;
constexpr int kKeyRows     = 2048;
constexpr int kValueRows   = 6144;
constexpr int kZRows       = 6144;
constexpr int kValueZRows  = kValueRows + kZRows;
constexpr int kQkRows      = kQueryRows + kKeyRows;
constexpr int kChannels    = kQkRows + kValueRows;
constexpr int kValueOffset = kQkRows;

using Q4ScheduleC4 = Q4RowSplitSimtGemmSchedule<8, 4, 16, 2, Cache::ca, 1>;
using Q4ScheduleC8 = Q4RowSplitSimtGemmSchedule<8, 8, 16, 2, Cache::ca, 1>;

enum class PdlOrder {
    Q4ThenQ5,
    Q5ThenQ4,
};

GdnConvSnapshotEpilogue make_epilogue(const Tensor& conv_weight, Tensor& conv_states,
                                      const Tensor& valid_columns, const Tensor& initial_slot,
                                      const Tensor& snapshot_base_slot, Tensor& query, Tensor& key,
                                      Tensor& value, int global_row_offset) {
    return {
        static_cast<const __nv_bfloat16*>(conv_weight.data),
        static_cast<__nv_bfloat16*>(conv_states.data),
        static_cast<const std::int32_t*>(initial_slot.data),
        static_cast<const std::int32_t*>(snapshot_base_slot.data),
        valid_columns.data == nullptr ? nullptr
                                      : static_cast<const std::int32_t*>(valid_columns.data),
        static_cast<__nv_bfloat16*>(query.data),
        static_cast<__nv_bfloat16*>(key.data),
        static_cast<__nv_bfloat16*>(value.data),
        kChannels,
        kQueryRows,
        kKeyRows,
        kValueRows,
        global_row_offset,
    };
}

struct Q4GdnDecodeEpilogue {
    GdnConvSnapshotEpilogue conv;

    template <bool, int>
    __device__ __forceinline__ void operator()(__nv_bfloat16*, __nv_bfloat16*, int row,
                                               float value) const {
        const float projected[1]{value};
        conv.store(row, projected);
    }
};

template <int Tokens>
struct Q4GdnSmallTEpilogue {
    GdnConvSnapshotEpilogue conv;

    template <bool, int, int TileCols>
    __device__ __forceinline__ void
    operator()(__nv_bfloat16*, __nv_bfloat16*, std::int32_t, std::int32_t, std::int32_t row,
               std::int32_t, std::int32_t active_cols, const float (&values)[TileCols]) const {
        float projected[Tokens];
#pragma unroll
        for (int token = 0; token < Tokens; ++token) { projected[token] = values[token]; }
        if (active_cols == Tokens) { conv.store(row, projected); }
    }
};

struct Q5GdnDecodeEpilogue {
    GdnConvSnapshotEpilogue conv;
    __nv_bfloat16* z;

    template <bool, int>
    __device__ __forceinline__ void operator()(__nv_bfloat16*, __nv_bfloat16*, int row,
                                               float value) const {
        if (row < kValueRows) {
            const float projected[1]{value};
            conv.store(row, projected);
        } else {
            z[row - kValueRows] = __float2bfloat16_rn(value);
        }
    }
};

template <int Tokens>
struct Q5GdnSmallTEpilogue {
    GdnConvSnapshotEpilogue conv;
    __nv_bfloat16* z;

    template <bool, int, int ProducedTokens>
    __device__ __forceinline__ void operator()(__nv_bfloat16*, __nv_bfloat16*, std::int32_t,
                                               std::int32_t, std::int32_t row,
                                               const float (&values)[ProducedTokens]) const {
        static_assert(ProducedTokens == Tokens);
        if (row < kValueRows) {
            conv.store(row, values);
        } else {
#pragma unroll
            for (int token = 0; token < Tokens; ++token) {
                z[static_cast<std::int64_t>(token) * kZRows + row - kValueRows] =
                    __float2bfloat16_rn(values[token]);
            }
        }
    }
};

template <bool TriggerPdl, bool JoinPdl, bool Dependent>
void launch_q4_t1(const Tensor& x, const Weight& qk_weight,
                  const GdnConvSnapshotEpilogue& qk_epilogue, Tensor& query, cudaStream_t stream) {
    constexpr int q4_threads = Q4GemvR1W8DirectSchedule::kThreads;
    constexpr int q4_blocks  = kQkRows / Q4GemvR1W8DirectSchedule::kRowsPerCta;
    if constexpr (Dependent) {
        CUDA_CHECK(
            pdl::launch_dependent({dim3(q4_blocks), dim3(q4_threads), 0, stream},
                                  q4_rowsplit_gemv_kernel<Q4GemvR1W8DirectSchedule, false, 0,
                                                          Q4GdnDecodeEpilogue, TriggerPdl, JoinPdl>,
                                  static_cast<const __nv_bfloat16*>(x.data),
                                  static_cast<const std::uint8_t*>(qk_weight.qdata),
                                  static_cast<const std::uint8_t*>(qk_weight.scales),
                                  static_cast<__nv_bfloat16*>(query.data), nullptr, kQkRows,
                                  kHidden, Q4GdnDecodeEpilogue{qk_epilogue}));
    } else {
        q4_rowsplit_gemv_kernel<Q4GemvR1W8DirectSchedule, false, 0, Q4GdnDecodeEpilogue, TriggerPdl,
                                JoinPdl><<<q4_blocks, q4_threads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(qk_weight.qdata),
            static_cast<const std::uint8_t*>(qk_weight.scales),
            static_cast<__nv_bfloat16*>(query.data), nullptr, kQkRows, kHidden,
            Q4GdnDecodeEpilogue{qk_epilogue});
    }
}

template <bool TriggerPdl, bool JoinPdl, bool Dependent>
void launch_q5_t1(const Tensor& x, const Weight& value_z_weight,
                  const GdnConvSnapshotEpilogue& value_epilogue, Tensor& value, Tensor& z,
                  cudaStream_t stream) {
    constexpr int q5_rows_per_block = 16;
    constexpr int q5_threads        = q5_rows_per_block * 32;
    constexpr int q5_blocks         = kValueZRows / q5_rows_per_block;
    if constexpr (Dependent) {
        CUDA_CHECK(pdl::launch_dependent(
            {dim3(q5_blocks), dim3(q5_threads), 0, stream},
            q5_rowsplit_gemv_kernel<kValueZRows, kHidden, q5_rows_per_block, 2, true, false, true,
                                    kValueRows, Q5GdnDecodeEpilogue, TriggerPdl, JoinPdl>,
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(value_z_weight.qdata),
            static_cast<const std::uint8_t*>(value_z_weight.qhigh),
            static_cast<const std::uint8_t*>(value_z_weight.scales),
            static_cast<__nv_bfloat16*>(value.data), static_cast<__nv_bfloat16*>(z.data),
            Q5GdnDecodeEpilogue{value_epilogue, static_cast<__nv_bfloat16*>(z.data)}));
    } else {
        q5_rowsplit_gemv_kernel<kValueZRows, kHidden, q5_rows_per_block, 2, true, false, true,
                                kValueRows, Q5GdnDecodeEpilogue, TriggerPdl, JoinPdl>
            <<<q5_blocks, q5_threads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const std::uint8_t*>(value_z_weight.qdata),
                static_cast<const std::uint8_t*>(value_z_weight.qhigh),
                static_cast<const std::uint8_t*>(value_z_weight.scales),
                static_cast<__nv_bfloat16*>(value.data), static_cast<__nv_bfloat16*>(z.data),
                Q5GdnDecodeEpilogue{value_epilogue, static_cast<__nv_bfloat16*>(z.data)});
    }
}

template <int Tokens, class Q4Schedule, bool TriggerPdl, bool JoinPdl, bool Dependent>
void launch_q4_small_t(const Tensor& x, const Weight& qk_weight,
                       const GdnConvSnapshotEpilogue& qk_epilogue, Tensor& query,
                       cudaStream_t stream) {
    const dim3 q4_grid(kQkRows / Q4Schedule::kRowsPerCta, 1u, 1u);
    if constexpr (Dependent) {
        CUDA_CHECK(pdl::launch_dependent(
            {q4_grid, dim3(Q4Schedule::kThreads), 0, stream},
            q4_rowsplit_gemm_simt_kernel<Q4Schedule, false, false, 0, Q4GdnSmallTEpilogue<Tokens>,
                                         TriggerPdl, JoinPdl>,
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(qk_weight.qdata),
            static_cast<const std::uint8_t*>(qk_weight.scales),
            static_cast<__nv_bfloat16*>(query.data), nullptr, kQueryRows, 0, kQkRows, kHidden,
            Tokens, kHidden, Q4GdnSmallTEpilogue<Tokens>{qk_epilogue}));
    } else {
        q4_rowsplit_gemm_simt_kernel<Q4Schedule, false, false, 0, Q4GdnSmallTEpilogue<Tokens>,
                                     TriggerPdl, JoinPdl>
            <<<q4_grid, Q4Schedule::kThreads, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(x.data),
                static_cast<const std::uint8_t*>(qk_weight.qdata),
                static_cast<const std::uint8_t*>(qk_weight.scales),
                static_cast<__nv_bfloat16*>(query.data), nullptr, kQueryRows, 0, kQkRows, kHidden,
                Tokens, kHidden, Q4GdnSmallTEpilogue<Tokens>{qk_epilogue});
    }
}

template <int Tokens, bool TriggerPdl, bool JoinPdl, bool Dependent>
void launch_q5_small_t(const Tensor& x, const Weight& value_z_weight,
                       const GdnConvSnapshotEpilogue& value_epilogue, Tensor& value, Tensor& z,
                       cudaStream_t stream) {
    constexpr int q5_threads = 4 * 32;
    const dim3 q5_grid(kValueZRows, 1u, 1u);
    if constexpr (Dependent) {
        CUDA_CHECK(pdl::launch_dependent(
            {q5_grid, dim3(q5_threads), 0, stream},
            q5_rowsplit_gemm_simt_split4_kernel<Q5RowSplitSimtSchedule, Tokens, 5, kHidden, true,
                                                kValueRows, Q5GdnSmallTEpilogue<Tokens>, TriggerPdl,
                                                JoinPdl>,
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(value_z_weight.qdata),
            static_cast<const std::uint8_t*>(value_z_weight.qhigh),
            static_cast<const std::uint8_t*>(value_z_weight.scales),
            static_cast<__nv_bfloat16*>(value.data), static_cast<__nv_bfloat16*>(z.data),
            kValueZRows, kValueRows, kHidden, Tokens, kHidden, 5,
            Q5GdnSmallTEpilogue<Tokens>{
                value_epilogue,
                static_cast<__nv_bfloat16*>(z.data),
            }));
    } else {
        q5_rowsplit_gemm_simt_split4_kernel<Q5RowSplitSimtSchedule, Tokens, 5, kHidden, true,
                                            kValueRows, Q5GdnSmallTEpilogue<Tokens>, TriggerPdl,
                                            JoinPdl><<<q5_grid, q5_threads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(value_z_weight.qdata),
            static_cast<const std::uint8_t*>(value_z_weight.qhigh),
            static_cast<const std::uint8_t*>(value_z_weight.scales),
            static_cast<__nv_bfloat16*>(value.data), static_cast<__nv_bfloat16*>(z.data),
            kValueZRows, kValueRows, kHidden, Tokens, kHidden, 5,
            Q5GdnSmallTEpilogue<Tokens>{
                value_epilogue,
                static_cast<__nv_bfloat16*>(z.data),
            });
    }
}

template <PdlOrder Order>
void launch_t1(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
               const GdnConvSnapshotEpilogue& qk_epilogue,
               const GdnConvSnapshotEpilogue& value_epilogue, Tensor& query, Tensor& value,
               Tensor& z, cudaStream_t stream) {
    // The Q4 and Q5 sides read the same activation but write disjoint output/state rows. The
    // dependent side therefore computes before waiting, then joins the producer at kernel exit.
    if constexpr (Order == PdlOrder::Q5ThenQ4) {
        launch_q5_t1<true, false, false>(x, value_z_weight, value_epilogue, value, z, stream);
        launch_q4_t1<false, true, true>(x, qk_weight, qk_epilogue, query, stream);
    } else {
        launch_q4_t1<true, false, false>(x, qk_weight, qk_epilogue, query, stream);
        launch_q5_t1<false, true, true>(x, value_z_weight, value_epilogue, value, z, stream);
    }
}

template <int Tokens, class Q4Schedule, PdlOrder Order>
void launch_small_t_schedule(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                             const GdnConvSnapshotEpilogue& qk_epilogue,
                             const GdnConvSnapshotEpilogue& value_epilogue, Tensor& query,
                             Tensor& value, Tensor& z, cudaStream_t stream) {
    if constexpr (Order == PdlOrder::Q5ThenQ4) {
        launch_q5_small_t<Tokens, true, false, false>(x, value_z_weight, value_epilogue, value, z,
                                                      stream);
        launch_q4_small_t<Tokens, Q4Schedule, false, true, true>(x, qk_weight, qk_epilogue, query,
                                                                 stream);
    } else {
        launch_q4_small_t<Tokens, Q4Schedule, true, false, false>(x, qk_weight, qk_epilogue, query,
                                                                  stream);
        launch_q5_small_t<Tokens, false, true, true>(x, value_z_weight, value_epilogue, value, z,
                                                     stream);
    }
}

template <int Tokens, PdlOrder Order>
void launch_small_t(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                    const GdnConvSnapshotEpilogue& qk_epilogue,
                    const GdnConvSnapshotEpilogue& value_epilogue, Tensor& query, Tensor& value,
                    Tensor& z, cudaStream_t stream) {
    if constexpr (Tokens <= 4) {
        launch_small_t_schedule<Tokens, Q4ScheduleC4, Order>(
            x, qk_weight, value_z_weight, qk_epilogue, value_epilogue, query, value, z, stream);
    } else {
        launch_small_t_schedule<Tokens, Q4ScheduleC8, Order>(
            x, qk_weight, value_z_weight, qk_epilogue, value_epilogue, query, value, z, stream);
    }
}

template <PdlOrder Order>
void launch_snapshot(const Tensor& x, const Weight& qk_weight, const Weight& value_z_weight,
                     const Tensor& conv_weight, Tensor& conv_states, const Tensor& valid_columns,
                     const Tensor& initial_slot, const Tensor& snapshot_base_slot, Tensor& query,
                     Tensor& key, Tensor& value, Tensor& z, cudaStream_t stream) {
    const GdnConvSnapshotEpilogue qk_epilogue =
        make_epilogue(conv_weight, conv_states, valid_columns, initial_slot, snapshot_base_slot,
                      query, key, value, 0);
    const GdnConvSnapshotEpilogue value_epilogue =
        make_epilogue(conv_weight, conv_states, valid_columns, initial_slot, snapshot_base_slot,
                      query, key, value, kValueOffset);

    switch (x.ne[1]) {
    case 1:
        launch_t1<Order>(x, qk_weight, value_z_weight, qk_epilogue, value_epilogue, query, value, z,
                         stream);
        break;
    case 2:
        launch_small_t<2, Order>(x, qk_weight, value_z_weight, qk_epilogue, value_epilogue, query,
                                 value, z, stream);
        break;
    case 3:
        launch_small_t<3, Order>(x, qk_weight, value_z_weight, qk_epilogue, value_epilogue, query,
                                 value, z, stream);
        break;
    case 5:
        launch_small_t<5, Order>(x, qk_weight, value_z_weight, qk_epilogue, value_epilogue, query,
                                 value, z, stream);
        break;
    case 6:
        launch_small_t<6, Order>(x, qk_weight, value_z_weight, qk_epilogue, value_epilogue, query,
                                 value, z, stream);
        break;
    default:
        throw std::invalid_argument(
            "Q4/Q5 projection-epilogue GDN snapshot requires T=1..3 or 5..6");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void q4_q5_gdn_input_conv_snapshot_launch(const Tensor& x, const Weight& qk_weight,
                                          const Weight& value_z_weight, const Tensor& conv_weight,
                                          Tensor& conv_states, const Tensor& valid_columns,
                                          const Tensor& initial_slot,
                                          const Tensor& snapshot_base_slot, Tensor& query,
                                          Tensor& key, Tensor& value, Tensor& z,
                                          cudaStream_t stream) {
    if (x.ne[1] == 2) {
        launch_snapshot<PdlOrder::Q4ThenQ5>(x, qk_weight, value_z_weight, conv_weight, conv_states,
                                            valid_columns, initial_slot, snapshot_base_slot, query,
                                            key, value, z, stream);
    } else {
        launch_snapshot<PdlOrder::Q5ThenQ4>(x, qk_weight, value_z_weight, conv_weight, conv_states,
                                            valid_columns, initial_slot, snapshot_base_slot, query,
                                            key, value, z, stream);
    }
}

void q4_q5_gdn_input_t4_post_snapshot_launch(const Tensor& projected, const Tensor& conv_weight,
                                             Tensor& conv_states, const Tensor& valid_columns,
                                             const Tensor& initial_slot,
                                             const Tensor& snapshot_base_slot, Tensor& query,
                                             Tensor& key, Tensor& value, cudaStream_t stream) {
    if (projected.ne[1] != 4) {
        throw std::invalid_argument("Q4/Q5 staged GDN post projection requires T=4");
    }
    constexpr int threads = 64;
    constexpr int blocks  = (kChannels + threads - 1) / threads;
    gdn_projected_conv_snapshot_kernel<kChannels, kQueryRows, kKeyRows, kValueRows, 4>
        <<<blocks, threads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(projected.data),
            static_cast<const __nv_bfloat16*>(conv_weight.data),
            static_cast<__nv_bfloat16*>(conv_states.data),
            static_cast<const std::int32_t*>(initial_slot.data),
            static_cast<const std::int32_t*>(snapshot_base_slot.data),
            valid_columns.data == nullptr ? nullptr
                                          : static_cast<const std::int32_t*>(valid_columns.data),
            static_cast<__nv_bfloat16*>(query.data), static_cast<__nv_bfloat16*>(key.data),
            static_cast<__nv_bfloat16*>(value.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
