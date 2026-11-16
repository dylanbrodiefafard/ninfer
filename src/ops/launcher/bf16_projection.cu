#include "ops/launcher/bf16_projection.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/bf16/bf16_config.h"
#include "ops/linear/bf16/bf16_gemm_mma_config.h"
#include "ops/linear/bf16/bf16_gemv.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace ninfer::ops::detail {
namespace {

// Routes, by output rows N and tokens T (public-Op measurements on RTX 5090 in bench/README.md):
//   - small-T MMA (below) for T <= 48 at N in {96, 128, 512, 544}, T = 1 included, and for
//     T = 2..48 at N = 2560;
//   - the GEMV for T = 1 at N in {2560, 10240}, where it reads the weight at 73% and 89% of the
//     sustained DRAM roof;
//   - the tensor-core GEMM for T > 48, and for every T >= 2 at N = 10240, where it reads the
//     50 MiB weight at 74% of the roof.
constexpr std::int32_t kSmallTMaxTokens = 48;

// T=1 at N in {2560, 10240}: one GEMV CTA per four output rows. Eight warps, two per row
// interleaving 256-value halves of each 512-value K phase (2560 = 5 phases).
using DecodeSchedule =
    Bf16GemvSchedule<8, 2, 1, 8, 4, Bf16ActivationAccess::Direct, Bf16WeightCache::Default,
                     Bf16PhaseOrder::RowSwizzled, 1, 1, 1, 2>;

// GEMM: the production BF16 GEMM tile (64 rows x 128 tokens) where N is a multiple of 64 (every
// projection but two), and a 32-row tile for the 96-row gate and 544-row MoE control projections.
template <class Geometry>
using PrefillSchedule = std::conditional_t<
    (Geometry::kOutputRows % 64) == 0, Bf16MmaProductionSchedule<Geometry>,
    Bf16MmaSchedule<32, 128, 64, 32, 32, 2, 2, Cache::cg, Cache::cg,
                    Bf16MmaFragmentPipeline::PingPong, Bf16MmaRaster::TokenFast>>;

// Small-T route: register-direct BF16 tensor-core MMA with the whole K reduction inside one CTA.
// The tokens sit on MMA-M and the output rows on MMA-N, so one m16n8k16 atom covers a 16-token
// tile and an 8-row tile, and the row granularity of a CTA is 8.
constexpr int kSmallTTokenTile = 16;
// A lane stages 16 K values per row or token and chunk, so a chunk spans 64 K values across the
// four lanes of a quad: four k16 atoms.
constexpr int kSmallTChunkK = 64;

enum class SmallTTokenSplit : std::uint8_t {
    // One CTA per 16-token tile and row block: the grid grows with T while each CTA reads one
    // tile of x; row blocks re-read their weight from L2 once per further tile.
    PerTile,
    // One CTA per row block for all tiles: the weight is read once while every CTA reads all of x.
    Whole,
};

// RowTiles 8-row tiles per CTA, Warps warps per CTA, each reducing a contiguous 1/Warps of K.
template <int RowTiles, int Warps, SmallTTokenSplit Split>
struct SmallTSchedule {
    static_assert(RowTiles >= 1 && Warps >= 1 && Warps <= 32);
    static constexpr int kRowTiles                = RowTiles;
    static constexpr int kWarps                   = Warps;
    static constexpr SmallTTokenSplit kTokenSplit = Split;
    static constexpr int kThreads                 = Warps * kWarpSize;
    static constexpr int kRowsPerCta              = 8 * RowTiles;
};

// Measured schedules. Each keeps its launch within one wave of the 170 SMs: Narrow launches
// N/8 * div_up(T,16) CTAs (N in {96, 128} for every T <= 48, N in {512, 544} for T <= 32), Wide
// N/16 * div_up(T,16) (N in {512, 544}, T = 33..48), and Whole N/16 = 160 (N = 2560, T = 2..48).
using SmallTNarrow = SmallTSchedule<1, 8, SmallTTokenSplit::PerTile>;
using SmallTWide   = SmallTSchedule<2, 8, SmallTTokenSplit::PerTile>;
using SmallTWhole  = SmallTSchedule<2, 8, SmallTTokenSplit::Whole>;

// Output element conversion: FP32 stores the accumulator unrounded, BF16 rounds it to nearest even.
template <class Element>
__device__ __forceinline__ Element convert_output(float accumulator) {
    if constexpr (std::is_same_v<Element, float>) {
        return accumulator;
    } else {
        return __float2bfloat16_rn(accumulator);
    }
}

// Stores two adjacent output rows of one token: 8 bytes for FP32, 4 bytes for BF16.
template <class Element>
__device__ __forceinline__ void store_row_pair(Element* destination, float first, float second) {
    if constexpr (std::is_same_v<Element, float>) {
        *reinterpret_cast<float2*>(destination) = make_float2(first, second);
    } else {
        *reinterpret_cast<__nv_bfloat162*>(destination) = __floats2bfloat162_rn(first, second);
    }
}

// How a lane reads its 16 values of a chunk: one 32-byte load when x and the weight are both
// 32-byte aligned (each warp load then reads 128 contiguous bytes of each of eight rows or tokens),
// otherwise two 16-byte loads of the same values (each warp load reads every other 16 bytes of
// those 128; the second load finds the rest in L2 or L1).
enum class SmallTLoadWidth : std::uint8_t {
    Bytes32,
    Bytes16,
};

enum class SmallTOperand : std::uint8_t {
    // Read once per call: not allocated in L1.
    Weight,
    // Read by every row block from L2, and allocated in L1 for CTAs sharing an SM.
    Activation,
};

// The four load forms of load_small_t_chunk. Both operands are read-only while the kernel runs
// (the contract's stream ordering and y's disjointness), so they use the non-coherent path; the
// asm is volatile so that a load predicated off for tokens past T is never speculated. The v8
// forms require a 32-byte-aligned address, the v4 forms a 16-byte-aligned one.
__device__ __forceinline__ void load_32_bytes_no_allocate(const __nv_bfloat16* p,
                                                          std::uint32_t (&w)[8]) {
    asm volatile("ld.global.nc.L1::no_allocate.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                 : "=r"(w[0]), "=r"(w[1]), "=r"(w[2]), "=r"(w[3]), "=r"(w[4]), "=r"(w[5]),
                   "=r"(w[6]), "=r"(w[7])
                 : "l"(p));
}

__device__ __forceinline__ void load_32_bytes(const __nv_bfloat16* p, std::uint32_t (&w)[8]) {
    asm volatile("ld.global.nc.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                 : "=r"(w[0]), "=r"(w[1]), "=r"(w[2]), "=r"(w[3]), "=r"(w[4]), "=r"(w[5]),
                   "=r"(w[6]), "=r"(w[7])
                 : "l"(p));
}

__device__ __forceinline__ void load_2x16_bytes_no_allocate(const __nv_bfloat16* p,
                                                            std::uint32_t (&w)[8]) {
    asm volatile("ld.global.nc.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%8];\n\t"
                 "ld.global.nc.L1::no_allocate.v4.b32 {%4, %5, %6, %7}, [%8 + 16];"
                 : "=r"(w[0]), "=r"(w[1]), "=r"(w[2]), "=r"(w[3]), "=r"(w[4]), "=r"(w[5]),
                   "=r"(w[6]), "=r"(w[7])
                 : "l"(p));
}

__device__ __forceinline__ void load_2x16_bytes(const __nv_bfloat16* p, std::uint32_t (&w)[8]) {
    asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%8];\n\t"
                 "ld.global.nc.v4.b32 {%4, %5, %6, %7}, [%8 + 16];"
                 : "=r"(w[0]), "=r"(w[1]), "=r"(w[2]), "=r"(w[3]), "=r"(w[4]), "=r"(w[5]),
                   "=r"(w[6]), "=r"(w[7])
                 : "l"(p));
}

// Lane q's 16 values of a 64-value chunk, K offsets [16q, 16q + 16), as eight 32-bit words.
template <SmallTLoadWidth Width, SmallTOperand Operand>
__device__ __forceinline__ Bf16GemvPack<16> load_small_t_chunk(const __nv_bfloat16* lane_chunk) {
    Bf16GemvPack<16> values;
    if constexpr (Width == SmallTLoadWidth::Bytes32) {
        if constexpr (Operand == SmallTOperand::Weight) {
            load_32_bytes_no_allocate(lane_chunk, values.words);
        } else {
            load_32_bytes(lane_chunk, values.words);
        }
    } else {
        if constexpr (Operand == SmallTOperand::Weight) {
            load_2x16_bytes_no_allocate(lane_chunk, values.words);
        } else {
            load_2x16_bytes(lane_chunk, values.words);
        }
    }
    return values;
}

// Kernel: y[n,t] = sum_k weight[n,k] x[k,t] over the CTA's rows and up to 16 * TokenTiles tokens.
//
// Grid: (N / (8 * RowTiles), div_up(tokens, 16 * TokenTiles)). CTA (b, c) owns output rows
// [8 * RowTiles * b, +8 * RowTiles) and tokens [16 * TokenTiles * c, +16 * TokenTiles). Block:
// Warps warps; warp w reduces the K slice [w * K / Warps, +K / Warps) for every row and token of
// the CTA. Lane (g, q) = (lane / 4, lane % 4).
//
// K permutation: within each 64-value chunk, lane (g, q) loads the 16 K values of
// load_small_t_chunk for output row 8r + g (the B operand of 8-row tile r) and for tokens
// 16m + g and 16m + 8 + g (the A operand of 16-token tile m), as eight 32-bit words each. Atom j
// of the chunk takes words 2j and 2j + 1 as its k slots {2q, 2q + 1} and {2q + 8, 2q + 9}. A and B
// apply the same permutation, so every atom pairs each weight value with the activation of the
// same logical k, and the chunk's four atoms cover its 64 values once. The fragments come straight
// from global memory with no shared-memory staging; tokens at or beyond `tokens` feed zero
// activations and are not stored.
//
// Reduction order, fixed and independent of T, of the token split and of the load width: per
// warp, its chunks in K order and atoms j = 0..3 within a chunk on the MMA accumulator; then the
// warps' partial sums in warp order 0..Warps-1, in IEEE FP32 through shared memory.
//
// Shared memory: partials[warp][m][r][c][lane], accumulator c of each warp's fragment,
// Warps * TokenTiles * RowTiles * 512 bytes, written once and read once around one __syncthreads.
//
// Preconditions (the launcher's): N % (8 * RowTiles) == 0, K % (64 * Warps) == 0,
// 1 <= tokens <= 16 * TokenTiles * gridDim.y, x and weight 32-byte aligned for Bytes32 and 16-byte
// aligned for Bytes16, y 8-byte aligned (FP32) or 4-byte aligned (BF16). Element offsets are
// 64-bit products.
template <class Geometry, class Schedule, int TokenTiles, SmallTLoadWidth Width, class Element>
__global__ __launch_bounds__(Schedule::kThreads) void bf16_projection_small_t_kernel(
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ weight,
    Element* __restrict__ y, std::int32_t tokens) {
    constexpr int N      = Geometry::kOutputRows;
    constexpr int K      = Geometry::kInputRows;
    constexpr int R      = Schedule::kRowTiles;
    constexpr int W      = Schedule::kWarps;
    constexpr int M      = TokenTiles;
    constexpr int Chunks = K / (W * kSmallTChunkK);
    static_assert(N % Schedule::kRowsPerCta == 0);
    static_assert(K % (W * kSmallTChunkK) == 0);
    static_assert(M >= 1);
    static_assert(sizeof(float) * W * M * R * 4 * kWarpSize <= 48 * 1024);
    using Pack = Bf16GemvPack<16>;

    __shared__ float partials[W][M][R][4][kWarpSize];

    const int lane                    = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int warp                    = static_cast<int>(threadIdx.x) / kWarpSize;
    const int g                       = lane >> 2;
    const int q                       = lane & 3;
    const int row0                    = static_cast<int>(blockIdx.x) * Schedule::kRowsPerCta;
    const int token0                  = static_cast<int>(blockIdx.y) * kSmallTTokenTile * M;
    const int local_tokens            = tokens - token0;
    const int k0                      = warp * Chunks * kSmallTChunkK + 16 * q;
    const __nv_bfloat16* const x_tile = x + static_cast<std::int64_t>(token0) * K;

    // Every weight load of the slice is issued before the first atom: the weight is the DRAM
    // stream, while x was just written and is read from L2.
    Pack w[Chunks][R];
#pragma unroll
    for (int chunk = 0; chunk < Chunks; ++chunk) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            w[chunk][r] = load_small_t_chunk<Width, SmallTOperand::Weight>(
                weight + static_cast<std::int64_t>(row0 + 8 * r + g) * K + k0 +
                chunk * kSmallTChunkK);
        }
    }

    float accumulators[M][R][4] = {};
#pragma unroll
    for (int chunk = 0; chunk < Chunks; ++chunk) {
        Pack a[M][2];
#pragma unroll
        for (int m = 0; m < M; ++m) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int token = kSmallTTokenTile * m + 8 * half + g;
                a[m][half]      = token < local_tokens
                                      ? load_small_t_chunk<Width, SmallTOperand::Activation>(
                                            x_tile + static_cast<std::int64_t>(token) * K + k0 +
                                            chunk * kSmallTChunkK)
                                      : Pack{};
            }
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
#pragma unroll
            for (int m = 0; m < M; ++m) {
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    float (&c)[4] = accumulators[m][r];
                    mma_bf16(c[0], c[1], c[2], c[3], a[m][0].words[2 * j], a[m][1].words[2 * j],
                             a[m][0].words[2 * j + 1], a[m][1].words[2 * j + 1],
                             w[chunk][r].words[2 * j], w[chunk][r].words[2 * j + 1]);
                }
            }
        }
    }

#pragma unroll
    for (int m = 0; m < M; ++m) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
#pragma unroll
            for (int c = 0; c < 4; ++c) { partials[warp][m][r][c][lane] = accumulators[m][r][c]; }
        }
    }
    __syncthreads();

    // Item (m, r, half, lane) sums accumulators 2 half and 2 half + 1 of that lane's fragments:
    // token 16m + 8 half + lane / 4 of the CTA, rows 8r + 2 (lane % 4) and the next one.
    constexpr int kItems  = M * R * 2 * kWarpSize;
    Element* const y_tile = y + static_cast<std::int64_t>(token0) * N;
    for (int item = static_cast<int>(threadIdx.x); item < kItems; item += Schedule::kThreads) {
        const int source_lane = item & (kWarpSize - 1);
        const int half        = (item / kWarpSize) & 1;
        const int r           = (item / (2 * kWarpSize)) % R;
        const int m           = item / (2 * kWarpSize * R);
        const int token       = kSmallTTokenTile * m + 8 * half + (source_lane >> 2);
        if (token >= local_tokens) { continue; }
        float first  = partials[0][m][r][2 * half][source_lane];
        float second = partials[0][m][r][2 * half + 1][source_lane];
#pragma unroll
        for (int source_warp = 1; source_warp < W; ++source_warp) {
            first += partials[source_warp][m][r][2 * half][source_lane];
            second += partials[source_warp][m][r][2 * half + 1][source_lane];
        }
        const int row = row0 + 8 * r + 2 * (source_lane & 3);
        store_row_pair(y_tile + static_cast<std::int64_t>(token) * N + row, first, second);
    }
}

template <class Element>
struct GemvOutput {
    Element* data;

    __device__ __forceinline__ void store(std::int32_t row, float value) const {
        data[row] = convert_output<Element>(value);
    }
};

struct GemvEpilogue {
    template <class Output>
    __device__ __forceinline__ void operator()(const Output& output, std::int32_t row,
                                               float accumulator) const {
        output.store(row, accumulator);
    }
};

template <class Element>
struct MmaOutputTile {
    Element* data;
    std::int32_t leading_dim;

    __device__ __forceinline__ void store(std::int32_t row, std::int32_t token,
                                          float accumulator) const {
        data[static_cast<std::int64_t>(token) * leading_dim + row] =
            convert_output<Element>(accumulator);
    }
};

template <class Element>
struct MmaOutput {
    Element* data;
    std::int32_t leading_dim;

    __device__ __forceinline__ MmaOutputTile<Element> tile(std::int32_t) const {
        return {data, leading_dim};
    }
};

template <class Geometry, class Element>
void launch_decode(const __nv_bfloat16* x, const __nv_bfloat16* weight, Element* y,
                   cudaStream_t stream) {
    static_assert((Geometry::kOutputRows % DecodeSchedule::kRowsPerCta) == 0);
    constexpr int kBlocks = Geometry::kOutputRows / DecodeSchedule::kRowsPerCta;
    bf16_gemv_kernel<Geometry, DecodeSchedule, GemvOutput<Element>, GemvEpilogue>
        <<<kBlocks, DecodeSchedule::kThreads, 0, stream>>>(x, weight, GemvOutput<Element>{y},
                                                           GemvEpilogue{});
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, class Schedule, int TokenTiles, class Element>
void launch_small_t_tiles(const __nv_bfloat16* x, const __nv_bfloat16* weight, Element* y,
                          std::int32_t tokens, cudaStream_t stream) {
    const dim3 grid(Geometry::kOutputRows / Schedule::kRowsPerCta,
                    div_up(tokens, kSmallTTokenTile * TokenTiles));
    // Row and token strides are 5120 bytes, so the bases decide every lane's alignment.
    static_assert((Geometry::kInputRows * sizeof(__nv_bfloat16)) % 32 == 0);
    const bool aligned32 =
        ((reinterpret_cast<std::uintptr_t>(x) | reinterpret_cast<std::uintptr_t>(weight)) % 32) ==
        0;
    if (aligned32) {
        bf16_projection_small_t_kernel<Geometry, Schedule, TokenTiles, SmallTLoadWidth::Bytes32,
                                       Element>
            <<<grid, Schedule::kThreads, 0, stream>>>(x, weight, y, tokens);
    } else {
        bf16_projection_small_t_kernel<Geometry, Schedule, TokenTiles, SmallTLoadWidth::Bytes16,
                                       Element>
            <<<grid, Schedule::kThreads, 0, stream>>>(x, weight, y, tokens);
    }
    CUDA_CHECK(cudaGetLastError());
}

// Precondition: 1 <= tokens <= kSmallTMaxTokens.
template <class Geometry, class Schedule, class Element>
void launch_small_t(const __nv_bfloat16* x, const __nv_bfloat16* weight, Element* y,
                    std::int32_t tokens, cudaStream_t stream) {
    if constexpr (Schedule::kTokenSplit == SmallTTokenSplit::PerTile) {
        launch_small_t_tiles<Geometry, Schedule, 1>(x, weight, y, tokens, stream);
    } else {
        static_assert(kSmallTMaxTokens == 3 * kSmallTTokenTile);
        if (tokens <= kSmallTTokenTile) {
            launch_small_t_tiles<Geometry, Schedule, 1>(x, weight, y, tokens, stream);
        } else if (tokens <= 2 * kSmallTTokenTile) {
            launch_small_t_tiles<Geometry, Schedule, 2>(x, weight, y, tokens, stream);
        } else {
            launch_small_t_tiles<Geometry, Schedule, 3>(x, weight, y, tokens, stream);
        }
    }
}

template <class Geometry, bool FullTokens, class Element>
void launch_mma(const __nv_bfloat16* x, const __nv_bfloat16* weight, Element* y,
                std::int32_t tokens, cudaStream_t stream) {
    using Schedule = PrefillSchedule<Geometry>;
    static_assert((Geometry::kOutputRows % Schedule::kBlockRows) == 0);
    static_assert((Geometry::kInputRows % Schedule::kBlockK) == 0);
    constexpr auto kernel =
        bf16_gemm_mma_kernel<Geometry, Schedule, FullTokens, MmaOutput<Element>>;
    if constexpr (Schedule::kSharedBytes > 48 * 1024) {
        static const cudaError_t attribute = cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Schedule::kSharedBytes);
        CUDA_CHECK(attribute);
    }
    constexpr int tiles_m = Geometry::kOutputRows / Schedule::kBlockRows;
    const int blocks      = tiles_m * div_up(tokens, Schedule::kBlockCols);
    kernel<<<blocks, Schedule::kThreads, Schedule::kSharedBytes, stream>>>(
        x, weight, MmaOutput<Element>{y, Geometry::kOutputRows}, tokens);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, class Element>
void launch_gemm(const __nv_bfloat16* x, const __nv_bfloat16* weight, Element* y,
                 std::int32_t tokens, cudaStream_t stream) {
    if ((tokens % PrefillSchedule<Geometry>::kBlockCols) == 0) {
        launch_mma<Geometry, true>(x, weight, y, tokens, stream);
    } else {
        launch_mma<Geometry, false>(x, weight, y, tokens, stream);
    }
}

template <class Geometry, class Element>
void launch_problem(const __nv_bfloat16* x, const __nv_bfloat16* weight, Element* y,
                    std::int32_t tokens, cudaStream_t stream) {
    constexpr int N = Geometry::kOutputRows;
    if constexpr (N == 10240) {
        if (tokens == 1) {
            launch_decode<Geometry>(x, weight, y, stream);
        } else {
            launch_gemm<Geometry>(x, weight, y, tokens, stream);
        }
    } else if constexpr (N == 2560) {
        if (tokens == 1) {
            launch_decode<Geometry>(x, weight, y, stream);
        } else if (tokens <= kSmallTMaxTokens) {
            launch_small_t<Geometry, SmallTWhole>(x, weight, y, tokens, stream);
        } else {
            launch_gemm<Geometry>(x, weight, y, tokens, stream);
        }
    } else if constexpr (N < 512) {
        if (tokens <= kSmallTMaxTokens) {
            launch_small_t<Geometry, SmallTNarrow>(x, weight, y, tokens, stream);
        } else {
            launch_gemm<Geometry>(x, weight, y, tokens, stream);
        }
    } else {
        if (tokens <= 2 * kSmallTTokenTile) {
            launch_small_t<Geometry, SmallTNarrow>(x, weight, y, tokens, stream);
        } else if (tokens <= kSmallTMaxTokens) {
            launch_small_t<Geometry, SmallTWide>(x, weight, y, tokens, stream);
        } else {
            launch_gemm<Geometry>(x, weight, y, tokens, stream);
        }
    }
}

} // namespace

void bf16_projection_launch(const std::uint16_t* x, const std::uint16_t* weight, float* y,
                            std::int32_t output_rows, std::int32_t input_rows, std::int32_t tokens,
                            cudaStream_t stream) {
    const auto* activation = reinterpret_cast<const __nv_bfloat16*>(x);
    const auto* matrix     = reinterpret_cast<const __nv_bfloat16*>(weight);
    if (input_rows != 2560) {
        throw std::logic_error("bf16_projection: unadmitted input width reached the launcher");
    }
    switch (output_rows) {
    case 96:
        launch_problem<Bf16GemvGeometry<96, 2560>>(activation, matrix, y, tokens, stream);
        return;
    case 544:
        launch_problem<Bf16GemvGeometry<544, 2560>>(activation, matrix, y, tokens, stream);
        return;
    case 2560:
        launch_problem<Bf16GemvGeometry<2560, 2560>>(activation, matrix, y, tokens, stream);
        return;
    case 10240:
        launch_problem<Bf16GemvGeometry<10240, 2560>>(activation, matrix, y, tokens, stream);
        return;
    default:
        throw std::logic_error(
            "bf16_projection: unadmitted FP32-output height reached the launcher");
    }
}

void bf16_projection_launch(const std::uint16_t* x, const std::uint16_t* weight, std::uint16_t* y,
                            std::int32_t output_rows, std::int32_t input_rows, std::int32_t tokens,
                            cudaStream_t stream) {
    const auto* activation = reinterpret_cast<const __nv_bfloat16*>(x);
    const auto* matrix     = reinterpret_cast<const __nv_bfloat16*>(weight);
    auto* out              = reinterpret_cast<__nv_bfloat16*>(y);
    if (input_rows != 2560) {
        throw std::logic_error("bf16_projection: unadmitted input width reached the launcher");
    }
    switch (output_rows) {
    case 128:
        launch_problem<Bf16GemvGeometry<128, 2560>>(activation, matrix, out, tokens, stream);
        return;
    case 512:
        launch_problem<Bf16GemvGeometry<512, 2560>>(activation, matrix, out, tokens, stream);
        return;
    default:
        throw std::logic_error(
            "bf16_projection: unadmitted BF16-output height reached the launcher");
    }
}

} // namespace ninfer::ops::detail
