#include "ops/linear/q4/q4_launch.h"

#include "core/device.h"
#include "ops/linear/q4/q4_small_t_mma.cuh"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

constexpr int kFirstSmallT    = 2;
constexpr int kLastFullT      = 32;
constexpr int kLastOptimizedT = 20;
using FullGeometry            = Q4DraftHeadGeometry<5120>;
using OptimizedGeometry       = Q4DraftHeadGeometry<2048>;

// Each warp owns sixteen weight rows over the full K, and the CTA's warps share one staged
// activation tile, so activation traffic per weight byte stays small as T grows. Every output
// accumulates ascending 64-wide scale groups, independent of T and of the tile schedule.
template <class Geometry, int TileCols, int ActiveCols, int Stages, int kRows, int KC>
__global__ __launch_bounds__(kRows * 2) void q4_draft_head_rows_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, __nv_bfloat16* __restrict__ out) {
    constexpr int K = Geometry::kInputRows, kNt = TileCols / 8, kThreads = kRows * 2;
    constexpr int KT = K / KC, kCodeStride = KC / 2 + 16;
    static_assert(K % KC == 0 && KT >= Stages);
    struct Stage {
        alignas(16) std::uint8_t codes[kRows][kCodeStride];
        alignas(16) std::uint16_t scales[kRows][KC / 64];
        alignas(16) __nv_bfloat16 x[TileCols][KC];
    };
    extern __shared__ __align__(16) std::uint8_t q4_draft_head_shared[];
    auto* stages = reinterpret_cast<Stage*>(q4_draft_head_shared);
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gid = lane >> 2, lid = lane & 3;
    const int row0 = blockIdx.x * kRows;
    const auto swizzle = [](int col, int k) { return (((k >> 3) ^ (col & 7)) << 3) | (k & 7); };
    const auto load = [&](int stage, int kt) {
        Stage& st = stages[stage];
        for (int task = tid; task < kRows * KC / 32; task += kThreads) {
            const int row = task / (KC / 32), chunk = task % (KC / 32);
            cp_async<16, Cache::cg>(&st.codes[row][chunk * 16],
                codes + static_cast<std::int64_t>(row0 + row) * (K / 2) + kt * (KC / 2) + chunk * 16);
        }
        for (int row = tid; row < kRows; row += kThreads) {
            cp_async<(KC / 64) * 2>(&st.scales[row][0],
                scales + (static_cast<std::int64_t>(row0 + row) * Geometry::kGroupsPerRow + kt * (KC / 64)) * 2);
        }
        for (int task = tid; task < ActiveCols * (KC / 8); task += kThreads) {
            const int col = task / (KC / 8), k8 = task % (KC / 8);
            cp_async<16>(&st.x[col][swizzle(col, k8 * 8)], x + static_cast<std::int64_t>(col) * K + kt * KC + k8 * 8);
        }
        cp_commit();
    };
    for (int stage = 0; stage < Stages; ++stage) { load(stage, stage); }
    const int b_rin = lane & 7, b_koff = ((lane >> 3) & 1) << 3;
    const int top = warp * 16 + gid, bottom = top + 8;
    float acc[kNt][4] = {};
    for (int kt = 0; kt < KT; ++kt) {
        const int stage = kt % Stages;
        if (kt + Stages <= KT) { cp_wait<Stages - 1>(); } else { cp_wait<0>(); }
        __syncthreads();
        const Stage& st = stages[stage];
#pragma unroll
        for (int g = 0; g < KC / 64; ++g) {
            float group_acc[kNt][4] = {};
#pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                const int byte = g * 32 + ks * 8 + lid;
                const unsigned a0 = q4_small_t_bf16_pair(st.codes[top][byte]);
                const unsigned a1 = q4_small_t_bf16_pair(st.codes[bottom][byte]);
                const unsigned a2 = q4_small_t_bf16_pair(st.codes[top][byte + 4]);
                const unsigned a3 = q4_small_t_bf16_pair(st.codes[bottom][byte + 4]);
#pragma unroll
                for (int nt = 0; nt < kNt; ++nt) {
                    unsigned b0, b1;
                    const int col = nt * 8 + b_rin;
                    ldmatrix_x2(b0, b1, smem_addr(&st.x[col][swizzle(col, g * 64 + ks * 16 + b_koff)]));
                    mma_bf16(group_acc[nt][0], group_acc[nt][1], group_acc[nt][2], group_acc[nt][3],
                             a0, a1, a2, a3, b0, b1);
                }
            }
            const float top_scale = __half2float(__ushort_as_half(st.scales[top][g]));
            const float bottom_scale = __half2float(__ushort_as_half(st.scales[bottom][g]));
#pragma unroll
            for (int nt = 0; nt < kNt; ++nt) {
                acc[nt][0] = fmaf(group_acc[nt][0], top_scale, acc[nt][0]);
                acc[nt][1] = fmaf(group_acc[nt][1], top_scale, acc[nt][1]);
                acc[nt][2] = fmaf(group_acc[nt][2], bottom_scale, acc[nt][2]);
                acc[nt][3] = fmaf(group_acc[nt][3], bottom_scale, acc[nt][3]);
            }
        }
        __syncthreads();
        if (kt + Stages < KT) { load(stage, kt + Stages); }
    }
#pragma unroll
    for (int nt = 0; nt < kNt; ++nt) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int col = nt * 8 + 2 * lid + (j & 1);
            if (col < ActiveCols) {
                out[static_cast<std::int64_t>(col) * Geometry::kOutputRows + row0 + (j < 2 ? top : bottom)] =
                    __float2bfloat16_rn(acc[nt][j]);
            }
        }
    }
}

template <class Geometry, int TileTokens, int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    // K256 stages stream 128 contiguous code bytes per row; above 24 tokens the activation tile
    // grows, so three K128 stages keep the same staged bytes. Both keep every output's reduction.
    constexpr int kStages = TileTokens <= 24 ? 2 : 3, kChunk = TileTokens <= 24 ? 256 : 128;
    constexpr auto kernel = q4_draft_head_rows_kernel<Geometry, TileTokens, ActiveTokens, kStages, 128, kChunk>;
    constexpr int smem = kStages * (128 * (kChunk / 2 + 16) + 128 * (kChunk / 64) * 2 + TileTokens * kChunk * 2);
    static const bool configured = [] {
        CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        return true;
    }();
    (void)configured;
    kernel<<<Geometry::kOutputRows / 128, 256, smem, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, int First, std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Q4Launch, sizeof...(Offsets)>{
        &launch_exact<Geometry, ((First + static_cast<int>(Offsets) + 7) / 8) * 8,
                      First + static_cast<int>(Offsets)>...};
}

constexpr auto kFullLaunchers = make_launchers<FullGeometry, kFirstSmallT>(
    std::make_index_sequence<kLastFullT - kFirstSmallT + 1>{});
constexpr auto kOptimizedLaunchers = make_launchers<OptimizedGeometry, kFirstSmallT>(
    std::make_index_sequence<kLastOptimizedT - kFirstSmallT + 1>{});

template <class Geometry>
bool matches(const Tensor& x, const Weight& weight) {
    return weight.n == Geometry::kOutputRows && weight.k == Geometry::kInputRows &&
           weight.padded_shape[1] == Geometry::kInputRows && x.ne[1] >= kFirstSmallT;
}

} // namespace

void launch_q4_draft_head_small_t(const Tensor& x, const Weight& weight, Tensor& out,
                                  cudaStream_t stream) {
    if (matches<FullGeometry>(x, weight) && x.ne[1] <= kLastFullT) {
        kFullLaunchers[static_cast<std::size_t>(x.ne[1] - kFirstSmallT)](x, weight, out, stream);
        return;
    }
    if (matches<OptimizedGeometry>(x, weight) && x.ne[1] <= kLastOptimizedT) {
        kOptimizedLaunchers[static_cast<std::size_t>(x.ne[1] - kFirstSmallT)](x, weight, out,
                                                                              stream);
        return;
    }
    throw std::invalid_argument("Q4 Linear draft-head small-T: unsupported exact problem");
}

} // namespace ninfer::ops::detail
