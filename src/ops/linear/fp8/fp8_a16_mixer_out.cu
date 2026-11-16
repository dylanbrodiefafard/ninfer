#include "ops/linear/fp8/fp8_launch.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/linear/fp8/fp8_a16_gemm_mma.cuh"
#include "ops/linear/fp8/fp8_a16_small_t_mma.cuh"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_gemv.cuh"
#include "ops/linear/fp8/fp8_output.cuh"
#include "ops/linear/fp8/fp8_small_t.cuh"

#include <cuda_bf16.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

using Geometry = Fp8FlashNextMixerOutGeometry;
using Launch   = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

Fp8ContiguousOutput contiguous_output(Tensor& out) {
    return {static_cast<__nv_bfloat16*>(out.data), Geometry::kOutputRows};
}

void launch_decode(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Schedule        = typename Fp8LinearDecodeProductionSchedule<Geometry>::Type;
    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    fp8_gemv_kernel<Geometry, Schedule><<<kBlocks, Schedule::kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const __nv_bfloat16*>(weight.scales), contiguous_output(out));
    CUDA_CHECK(cudaGetLastError());
}

template <int ActiveTokens>
void launch_simt(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Schedule = typename Fp8LinearSmallTProductionSchedule<Geometry, ActiveTokens>::Type;
    static_assert(Schedule::kTokenTile == ActiveTokens);
    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    fp8_small_t_kernel<Geometry, ActiveTokens, Schedule>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales), contiguous_output(out));
    CUDA_CHECK(cudaGetLastError());
}

template <std::size_t... Offsets>
constexpr auto make_simt_launchers(std::index_sequence<Offsets...>) {
    return std::array<Launch, sizeof...(Offsets)>{
        &launch_simt<kFp8FirstSmallT + static_cast<int>(Offsets)>...};
}

constexpr auto kSimtLaunchers = make_simt_launchers(
    std::make_index_sequence<kFp8LinearSmallTMax<Geometry> - kFp8FirstSmallT + 1>{});

// One CTA per 16 output rows; the kernel masks the columns of the tile beyond T.
template <int TileTokens>
void launch_small_mma(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Schedule =
        typename Fp8FlashNextA16SmallTMmaProductionSchedule<Geometry, TileTokens>::Type;
    fp8_a16_small_t_mma_kernel<Geometry, TileTokens, Schedule, Fp8ContiguousOutput, true>
        <<<Geometry::kOutputRows / Schedule::kRowsPerCta, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales), contiguous_output(out), x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule, bool FullTokens>
void launch_gemm_slice(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    static_assert((Geometry::kOutputRows % Schedule::kBlockRows) == 0);
    static_assert((Geometry::kInputRows % Schedule::kBlockK) == 0);
    static_assert(Schedule::kSharedBytes <= 48 * 1024);
    constexpr int kRowTiles = Geometry::kOutputRows / Schedule::kBlockRows;
    const int token_tiles   = div_up(x.ne[1], Schedule::kBlockTokens);
    const dim3 grid(static_cast<unsigned>(kRowTiles), static_cast<unsigned>(token_tiles), 1U);
    fp8_a16_gemm_mma_kernel<Geometry, Schedule, FullTokens>
        <<<grid, Schedule::kThreads, Schedule::kSharedBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales), contiguous_output(out), x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

// Splits T only where grid.y would exceed CUDA's limit; a slice that is a whole number of token
// tiles takes the unmasked kernel.
template <class Schedule>
void launch_gemm(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    for_each_token_slice(x.ne[1], Schedule::kBlockTokens,
                         [&](std::int32_t offset, std::int32_t count) {
                             const Tensor input = x.slice(1, offset, count);
                             Tensor output      = out.slice(1, offset, count);
                             if ((count % Schedule::kBlockTokens) == 0) {
                                 launch_gemm_slice<Schedule, true>(input, weight, output, stream);
                             } else {
                                 launch_gemm_slice<Schedule, false>(input, weight, output, stream);
                             }
                         });
}

} // namespace

// Width tiers measured on the RTX 5090 with cold weights. N=2560 gives few output-row CTAs, so the
// K-serial MMA and GEMM mainloops are latency-bound below a few hundred columns: the T=1 GEMV and
// the T=2..8 SIMT kernel stream the 15.7 MB matrix in the same time, the small-T MMA tiers hold
// that time through 16 columns, and its 48-column tile (29 us) remains faster than the first GEMM
// launch (48 us at T=49). The 32x64 GEMM tile leads through 384 columns and the 64x64 tile beyond.
void launch_fp8_flash_next_mixer_out(const Tensor& x, const Weight& weight, Tensor& out,
                                     cudaStream_t stream) {
    if (weight.n != Geometry::kOutputRows || weight.k != Geometry::kInputRows || x.ne[1] <= 0) {
        throw std::invalid_argument("fp8 Flash-Next mixer output: invalid exact problem");
    }
    using Gemm32x64           = Fp8A16GemmSchedule<32, 64, 128, 16, 16, 1, 3>;
    using Gemm64x64           = Fp8A16GemmSchedule<64, 64, 64, 32, 16, 2, 2>;
    const std::int32_t tokens = x.ne[1];
    if (tokens == 1) {
        launch_decode(x, weight, out, stream);
    } else if (tokens <= kFp8LinearSmallTMax<Geometry>) {
        kSimtLaunchers[static_cast<std::size_t>(tokens - kFp8FirstSmallT)](x, weight, out, stream);
    } else if (tokens <= 16) {
        launch_small_mma<16>(x, weight, out, stream);
    } else if (tokens <= 24) {
        launch_small_mma<24>(x, weight, out, stream);
    } else if (tokens <= 32) {
        launch_small_mma<32>(x, weight, out, stream);
    } else if (tokens <= 48) {
        launch_small_mma<48>(x, weight, out, stream);
    } else if (tokens <= 384) {
        launch_gemm<Gemm32x64>(x, weight, out, stream);
    } else {
        launch_gemm<Gemm64x64>(x, weight, out, stream);
    }
}

} // namespace ninfer::ops::detail
