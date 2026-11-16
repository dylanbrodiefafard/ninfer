#include "ops/gdn_input_proj/fp8/fp8_gdn_input_plan.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/gdn_input_proj/fp8/fp8_gdn_input_output.cuh"
#include "ops/linear/fp8/fp8_a16_gemm_mma.cuh"
#include "ops/linear/fp8/fp8_a16_small_t_mma.cuh"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_small_t.cuh"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

template <class Geometry, int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                  cudaStream_t stream) {
    using Schedule = typename Fp8LinearSmallTProductionSchedule<Geometry, ActiveTokens>::Type;
    constexpr int kTokenTiles = (ActiveTokens + Schedule::kTokenTile - 1) / Schedule::kTokenTile;
    constexpr int kBlocks     = (Geometry::kOutputRows / Schedule::kRowsPerCta) * kTokenTiles;
    const Fp8GdnInputOutput output{static_cast<__nv_bfloat16*>(qkv.data),
                                   static_cast<__nv_bfloat16*>(z.data)};
    fp8_small_t_kernel<Geometry, ActiveTokens, Schedule>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales), output);
    CUDA_CHECK(cudaGetLastError());
}

// One CTA per 16 parent rows; the kernel masks the columns of the tile beyond T.
template <class Geometry, int Capacity, class Schedule>
void launch_small_mma(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                      cudaStream_t stream) {
    static_assert(Schedule::kTileTokens == Capacity);
    const Fp8GdnInputOutput output{static_cast<__nv_bfloat16*>(qkv.data),
                                   static_cast<__nv_bfloat16*>(z.data)};
    fp8_a16_small_t_mma_kernel<Geometry, Capacity, Schedule, Fp8GdnInputOutput, true>
        <<<Geometry::kOutputRows / Schedule::kRowsPerCta, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales), output, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

// [16384,5120]: sixteen K warps (1024-column groups) through 8 columns, eight through 24, four
// from 32.
template <int Capacity>
void launch_small_mma_5120(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                           cudaStream_t stream) {
    constexpr int kKWarps = Capacity <= 8 ? 16 : Capacity <= 24 ? 8 : 4;
    using Schedule        = Fp8A16SmallTMmaSchedule<kKWarps, Capacity, kKWarps == 16 ? 1 : 2>;
    launch_small_mma<Fp8GdnInputGeometry, Capacity, Schedule>(x, weight, qkv, z, stream);
}

template <int Capacity>
void launch_small_mma_2560(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                           cudaStream_t stream) {
    using Geometry = Fp8FlashNextGdnInputGeometry;
    using Schedule = typename Fp8FlashNextA16SmallTMmaProductionSchedule<Geometry, Capacity>::Type;
    launch_small_mma<Geometry, Capacity, Schedule>(x, weight, qkv, z, stream);
}

// Every GEMM row tile lies inside the q|k|v or z range, so whole-tile stores never straddle the
// two destinations. The final token tile is partial whenever T is not a tile multiple. T is split
// only where grid.y would exceed CUDA's limit; each slice offsets x, qkv, and z by its first
// column.
template <class Geometry, class Schedule>
void launch_gemm(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                 cudaStream_t stream) {
    static_assert(Fp8GdnInputOutput::kQkvRows % Schedule::kBlockRows == 0);
    static_assert(Fp8GdnInputOutput::kZRows % Schedule::kBlockRows == 0);
    static_assert(Schedule::kSharedBytes <= 48 * 1024);
    for_each_token_slice(
        x.ne[1], Schedule::kBlockTokens, [&](std::int32_t offset, std::int32_t count) {
            const Tensor input = x.slice(1, offset, count);
            Tensor qkv_slice   = qkv.slice(1, offset, count);
            Tensor z_slice     = z.slice(1, offset, count);
            const dim3 grid(Geometry::kOutputRows / Schedule::kBlockRows,
                            static_cast<unsigned>(div_up(count, Schedule::kBlockTokens)));
            const Fp8GdnInputOutput output{static_cast<__nv_bfloat16*>(qkv_slice.data),
                                           static_cast<__nv_bfloat16*>(z_slice.data)};
            fp8_a16_gemm_mma_kernel<Geometry, Schedule, false>
                <<<grid, Schedule::kThreads, Schedule::kSharedBytes, stream>>>(
                    static_cast<const __nv_bfloat16*>(input.data),
                    static_cast<const std::uint8_t*>(weight.qdata),
                    static_cast<const __nv_bfloat16*>(weight.scales), output, count);
            CUDA_CHECK(cudaGetLastError());
        });
}

// [16384,5120], measured on the RTX 5090: SIMT for the latency regime, bounded MMA column
// capacities, then amortized weight decode.
void launch_matrix_5120(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                        cudaStream_t stream) {
    using Geometry    = Fp8GdnInputGeometry;
    const int columns = x.ne[1];
    if (columns == 2) return launch_exact<Geometry, 2>(x, weight, qkv, z, stream);
    if (columns == 3) return launch_exact<Geometry, 3>(x, weight, qkv, z, stream);
    if (columns == 4) return launch_exact<Geometry, 4>(x, weight, qkv, z, stream);
    if (columns <= 8) return launch_small_mma_5120<8>(x, weight, qkv, z, stream);
    if (columns <= 16) return launch_small_mma_5120<16>(x, weight, qkv, z, stream);
    if (columns <= 24) return launch_small_mma_5120<24>(x, weight, qkv, z, stream);
    if (columns <= 32) return launch_small_mma_5120<32>(x, weight, qkv, z, stream);
    if (columns <= 64)
        return launch_gemm<Geometry, Fp8A16GemmSchedule<32, 64, 128, 16, 16, 1, 3>>(x, weight, qkv,
                                                                                    z, stream);
    if (columns <= 96)
        return launch_gemm<Geometry, Fp8A16GemmSchedule<64, 96, 128, 64, 16, 1, 2>>(x, weight, qkv,
                                                                                    z, stream);
    return launch_gemm<Geometry, Fp8A16GemmSchedule<64, 128, 64, 32, 16, 2, 2>>(x, weight, qkv, z,
                                                                                stream);
}

// [16384,2560], measured on the RTX 5090 with cold weights: SIMT through 8 columns, the small-T
// MMA through 32, then the GEMM. The 64x64 tile is the GEMM core; for 65..128 columns one
// 128-column token tile beats two 64-column tiles by 9%, while from 129 the 64x64 tile's finer
// token quantization wins again (within 4% of the 128-column tile at T=2048 and 4096).
void launch_matrix_2560(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                        cudaStream_t stream) {
    using Geometry = Fp8FlashNextGdnInputGeometry;
    using Gemm64   = Fp8A16GemmSchedule<64, 64, 64, 32, 16, 2, 2>;
    using Gemm128  = Fp8A16GemmSchedule<64, 128, 64, 64, 16, 2, 2>;
    static_assert(kFp8LinearSmallTMax<Geometry> == 8);
    const int columns = x.ne[1];
    if (columns == 2) return launch_exact<Geometry, 2>(x, weight, qkv, z, stream);
    if (columns == 3) return launch_exact<Geometry, 3>(x, weight, qkv, z, stream);
    if (columns == 4) return launch_exact<Geometry, 4>(x, weight, qkv, z, stream);
    if (columns == 5) return launch_exact<Geometry, 5>(x, weight, qkv, z, stream);
    if (columns == 6) return launch_exact<Geometry, 6>(x, weight, qkv, z, stream);
    if (columns == 7) return launch_exact<Geometry, 7>(x, weight, qkv, z, stream);
    if (columns == 8) return launch_exact<Geometry, 8>(x, weight, qkv, z, stream);
    if (columns <= 16) return launch_small_mma_2560<16>(x, weight, qkv, z, stream);
    if (columns <= 24) return launch_small_mma_2560<24>(x, weight, qkv, z, stream);
    if (columns <= 32) return launch_small_mma_2560<32>(x, weight, qkv, z, stream);
    if (columns <= 64) return launch_gemm<Geometry, Gemm64>(x, weight, qkv, z, stream);
    if (columns <= 128) return launch_gemm<Geometry, Gemm128>(x, weight, qkv, z, stream);
    return launch_gemm<Geometry, Gemm64>(x, weight, qkv, z, stream);
}

} // namespace

void fp8_gdn_input_matrix_launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
                                 cudaStream_t stream) {
    if (x.ne[1] < 2) {
        throw std::invalid_argument("fp8 gdn_input_proj matrix: T must be at least two");
    }
    switch (weight.k) {
    case Fp8GdnInputGeometry::kInputRows:
        launch_matrix_5120(x, weight, qkv, z, stream);
        return;
    case Fp8FlashNextGdnInputGeometry::kInputRows:
        launch_matrix_2560(x, weight, qkv, z, stream);
        return;
    default:
        throw std::invalid_argument("fp8 gdn_input_proj matrix: unsupported input rows");
    }
}

} // namespace ninfer::ops::detail
