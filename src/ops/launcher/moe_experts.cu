#include "ops/launcher/moe_experts.h"

#include "core/device.h"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"
#include "ops/linear/nvfp4/nvfp4_w4a8_mma.cuh"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// --- Shared geometry ---------------------------------------------------------------------------
//
// Both projections run the same tile computation: a work item is one job (up to C assignments of
// one expert) and RW tiles of 32 weight rows, multiplied over the full K by BF16 m16n8k16
// MMAs with FP32 accumulators. K is processed in chunks of 128, the unit of the record layout
// (ninfer/ops/moe_experts.h). In each chunk, lane L = 4 g + c of a warp owns weight rows
// rho_k = g + 8 k (k = 0..3; MMA fragment f covers units 2f and 2f+1, i.e. rows 16f + g and
// 16f + g + 8) and K offsets 32 c .. 32 c + 31, consumed by eight MMA steps: step s maps the
// fragment's k slots {2c, 2c+1, 2c+8, 2c+9} to chunk K offsets 32 c + 4 s + {0,1,2,3}. The B
// operand (activations) uses the same permutation, so every product pairs a weight with the
// activation of its own K and each dot product is unchanged; the permutation lets one F3 unit or
// two NVFP4 scale groups belong to a single lane.

constexpr int kChunkK        = 128;
constexpr int kChunkUnits    = kChunkK / 8; // 16-byte BF16 activation units per chunk
constexpr int kStepsPerChunk = 8;           // m16n8k16 steps of one lane's 32 K offsets
constexpr int kTileRows      = 32;
constexpr int kGateUpRows    = 2 * kMoeIntermediate;

enum class Matrix : std::uint8_t {
    GateUp, // gated = SiLU(G x) * U x, K = 2560
    Down,   // contribution = weight * D gated, K = 640
};

template <Matrix M>
struct MatrixShape;

template <>
struct MatrixShape<Matrix::GateUp> {
    static constexpr int kK          = kMoeHidden;
    static constexpr int kChunks     = kMoeHidden / kChunkK;    // 20
    static constexpr int kTiles      = kGateUpRows / kTileRows; // 40
    static constexpr int kFirstChunk = 0;
};

template <>
struct MatrixShape<Matrix::Down> {
    static constexpr int kK          = kMoeIntermediate;
    static constexpr int kChunks     = kMoeIntermediate / kChunkK; // 5
    static constexpr int kTiles      = kMoeHidden / kTileRows;     // 80
    static constexpr int kFirstChunk = static_cast<int>(kMoeGateUpChunks);
};

static_assert(MatrixShape<Matrix::GateUp>::kTiles * MatrixShape<Matrix::GateUp>::kChunks ==
              kMoeGateUpChunks);
static_assert(MatrixShape<Matrix::Down>::kTiles * MatrixShape<Matrix::Down>::kChunks ==
              kMoeDownChunks);

// Warp arrangement of one CTA: KW warps split each stage's KW chunks of K (their partial sums are
// added in ascending warp order), RW warps own distinct 32-row tiles and share the staged
// activations. Stages counts the activation buffers of the cp.async pipeline. Small token tiles
// split K so a single token still spreads one expert over 40 (gate|up) or 80 (down) CTAs of
// enough warps; wide tiles give each warp whole tiles so a staged activation feeds four warps.
// MinCtas is the CTAs per SM the register budget must admit (__launch_bounds__): the down
// projection at C = 16 has a single unbuffered stage per CTA and hides its latency with three.
template <Matrix M, int C>
struct Schedule;

template <int KW, int RW, int Stages, int MinCtas = 1>
struct ScheduleShape {
    static constexpr int kWarpsK    = KW;
    static constexpr int kWarpsRows = RW;
    static constexpr int kStages    = Stages;
    static constexpr int kMinCtas   = MinCtas;
};

template <>
struct Schedule<Matrix::GateUp, 16> : ScheduleShape<4, 1, 3> {};

template <>
struct Schedule<Matrix::GateUp, 32> : ScheduleShape<1, 4, 4> {};

template <>
struct Schedule<Matrix::GateUp, 64> : ScheduleShape<1, 4, 3> {};

template <>
struct Schedule<Matrix::Down, 16> : ScheduleShape<5, 1, 1, 3> {};

template <>
struct Schedule<Matrix::Down, 32> : ScheduleShape<1, 4, 3> {};

template <>
struct Schedule<Matrix::Down, 64> : ScheduleShape<1, 4, 3> {};

// Derived launch facts of one (matrix, token tile) instantiation.
template <Matrix M, int C>
struct Plan {
    using Shape                           = MatrixShape<M>;
    static constexpr int kWarpsK          = Schedule<M, C>::kWarpsK;
    static constexpr int kWarpsRows       = Schedule<M, C>::kWarpsRows;
    static constexpr int kThreads         = kWarpsK * kWarpsRows * kWarpSize;
    static constexpr int kStages          = Shape::kChunks / kWarpsK;
    static constexpr int kBuffers         = std::min(Schedule<M, C>::kStages, kStages);
    static constexpr int kRowBytes        = kWarpsK * kChunkK * 2; // one token's staged K
    static constexpr int kStageBytes      = C * kRowBytes;
    static constexpr int kTileGroups      = Shape::kTiles / kWarpsRows;
    static constexpr int kReductionStride = kWarpsRows * kTileRows + 4; // floats per token
    static constexpr int kReductionBytes =
        kWarpsK * C * kReductionStride * static_cast<int>(sizeof(float));
    static constexpr int kStagingBytes = std::max(kBuffers * kStageBytes, kReductionBytes);
    static constexpr int kSharedBytes  = kStagingBytes + C * static_cast<int>(sizeof(int));

    static_assert(Shape::kChunks % kWarpsK == 0 && Shape::kTiles % kWarpsRows == 0);
    static_assert(kStages == 1 || kBuffers >= 2);
    static_assert(C % 8 == 0 && kSharedBytes <= 99 * 1024);
};

// --- Weight codecs ---------------------------------------------------------------------------
//
// A codec loads one lane's part of a 32-row x 128-K chunk directly into registers (each warp-wide
// load is 512 contiguous bytes of the chunk record) and expands the A fragment of one MMA step:
// rows rho_{2f} (a0, a2) and rho_{2f+1} (a1, a3), K offsets 32 c + 4 s + {0,1} (a0, a1) and + {2,3}
// (a2, a3). Every expanded value is the exact BF16 product of a code and its E4M3 scale (at most
// six significant bits, see nvfp4_codes_to_bf16x2).

// Byte-permute tables of one F3 codebook: byte v of {lo[1]:lo[0]} and {hi[1]:hi[0]} is the low and
// high byte of the BF16 value of book entry v.
struct F3Lut {
    unsigned lo[2];
    unsigned hi[2];
};

// Four 3-bit book indices (bits 0..2 of nibbles 0..3 of selectors; the nibble high bits and bits
// 16..31 are ignored) to BF16 pairs {index 0, 1} and {2, 3}, each scaled by the E4M3 scale.
__device__ __forceinline__ void f3_codes_to_bf16x2(unsigned selectors, const F3Lut& lut,
                                                   const __nv_bfloat162& scale, unsigned& low,
                                                   unsigned& high) {
    const unsigned index         = selectors & 0x7777U;
    const unsigned lo            = __byte_perm(lut.lo[0], lut.lo[1], index);
    const unsigned hi            = __byte_perm(lut.hi[0], lut.hi[1], index);
    unsigned pair0               = __byte_perm(lo, hi, 0x5140U);
    unsigned pair1               = __byte_perm(lo, hi, 0x7362U);
    const __nv_bfloat162 scaled0 = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&pair0), scale);
    const __nv_bfloat162 scaled1 = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&pair1), scale);
    low                          = *reinterpret_cast<const unsigned*>(&scaled0);
    high                         = *reinterpret_cast<const unsigned*>(&scaled1);
}

__device__ __forceinline__ void unpack(const uint4& value, unsigned (&words)[4]) {
    words[0] = value.x;
    words[1] = value.y;
    words[2] = value.z;
    words[3] = value.w;
}

// F3G32 record chunks: per lane one scale byte and three code words per unit (see the header).
struct F3Codec {
    static constexpr int kChunkBytes = static_cast<int>(kMoeF3ChunkBytes);
    static constexpr int kCodeOffset = 128;
    const std::uint8_t* tile_chunks  = nullptr; // chunk 0 of this warp's tile
    int lane                         = 0;
    F3Lut books[2]{}; // fragment 0 and fragment 1 rows

    struct Chunk {
        unsigned scales;
        unsigned words[3][4]; // [code word j][unit k]
        __nv_bfloat162 scale[4];
        unsigned high[4]; // codes 24..31 of each unit, gathered from the nibble high bits
    };

    __device__ __forceinline__ void load(Chunk& chunk, int q) const {
        const std::uint8_t* source = tile_chunks + static_cast<std::int64_t>(q) * kChunkBytes;
        chunk.scales               = __ldcg(reinterpret_cast<const unsigned*>(source + 4 * lane));
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            unpack(__ldcg(reinterpret_cast<const uint4*>(source + kCodeOffset +
                                                         16 * (kWarpSize * j + lane))),
                   chunk.words[j]);
        }
    }

    __device__ __forceinline__ void prepare(Chunk& chunk) const {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            chunk.scale[k] = nvfp4_scale_bf16x2(static_cast<std::uint8_t>(chunk.scales >> (8 * k)));
            chunk.high[k]  = ((chunk.words[0][k] >> 3) & 0x11111111U) |
                             ((chunk.words[1][k] >> 2) & 0x22222222U) |
                             ((chunk.words[2][k] >> 1) & 0x44444444U);
        }
    }

    __device__ __forceinline__ void fragment(const Chunk& chunk, int step, int frag,
                                             unsigned (&a)[4]) const {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int unit      = 2 * frag + r;
            const unsigned word = step < 6 ? chunk.words[step / 2][unit] : chunk.high[unit];
            f3_codes_to_bf16x2(word >> (16 * (step & 1)), books[frag], chunk.scale[unit], a[r],
                               a[r + 2]);
        }
    }
};

// NVFP4G16 record chunks: per lane two scale bytes and four code words per row (see the header).
struct Nvfp4Codec {
    static constexpr int kChunkBytes = static_cast<int>(kMoeNvfp4ChunkBytes);
    static constexpr int kCodeOffset = 256;
    const std::uint8_t* tile_chunks  = nullptr;
    int lane                         = 0;

    struct Chunk {
        uint2 scales;
        unsigned words[4][4]; // [code word j][unit k]
        __nv_bfloat162 scale[4][2];
    };

    __device__ __forceinline__ void load(Chunk& chunk, int q) const {
        const std::uint8_t* source = tile_chunks + static_cast<std::int64_t>(q) * kChunkBytes;
        chunk.scales               = __ldcg(reinterpret_cast<const uint2*>(source + 8 * lane));
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            unpack(__ldcg(reinterpret_cast<const uint4*>(source + kCodeOffset +
                                                         16 * (kWarpSize * j + lane))),
                   chunk.words[j]);
        }
    }

    __device__ __forceinline__ void prepare(Chunk& chunk) const {
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const unsigned word = k < 2 ? chunk.scales.x : chunk.scales.y;
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                chunk.scale[k][h] =
                    nvfp4_scale_bf16x2(static_cast<std::uint8_t>(word >> (16 * (k & 1) + 8 * h)));
            }
        }
    }

    __device__ __forceinline__ void fragment(const Chunk& chunk, int step, int frag,
                                             unsigned (&a)[4]) const {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int unit = 2 * frag + r;
            nvfp4_codes_to_bf16x2(chunk.words[step / 2][unit] >> (16 * (step & 1)),
                                  chunk.scale[unit][step / 4], a[r], a[r + 2]);
        }
    }
};

// The expert table holds record addresses as integers by contract.
__device__ __forceinline__ std::uint8_t* table_address(std::uint64_t bits) {
    return reinterpret_cast<std::uint8_t*>(bits); // NOLINT(performance-no-int-to-ptr)
}

// --- Expert matmul kernel ----------------------------------------------------------------------

// The jobs one matmul launch serves, as ranges of the job table: every job, the jobs before the
// deferred ones (job_count[0]), or the deferred ones [job_count[0], job_count[1]).
enum class JobRange : std::uint8_t { All, Resident, Deferred };

struct MatmulParams {
    const std::uint16_t* activations  = nullptr; // GateUp: x [T][2560]; Down: gated [11T][640]
    const std::uint8_t* shared_record = nullptr; // NVFP4G16 record, 256-byte aligned
    const float* shared_matrix_scales = nullptr; // [3]
    const std::uint64_t* expert_table = nullptr;
    const std::uint64_t* deferred     = nullptr; // Deferred range: untagged slot addresses
    JobRange range                    = JobRange::All;
    const float* matrix_scales        = nullptr;
    const std::int32_t* sorted        = nullptr;
    const MoeJob* jobs                = nullptr;
    MoeJobSchedule* schedule          = nullptr;
    int launch                        = 0;       // index of this launch's next_item counter
    const float* route_weights        = nullptr; // Down only
    const float* shared_scale         = nullptr; // Down only
    std::uint16_t* gated              = nullptr; // GateUp output
    float* contributions              = nullptr; // Down output
    F3Lut books[2]{};                            // GateUp: gate, up; Down: down twice
};

// Activation unit u (8 BF16 at chunk K offset 8u) of token slot n is stored at unit position
// u ^ ((u >> 2) & 2) ^ (n & 1) of its chunk row. A B-fragment load (below) has the eight lanes of
// one shared-memory phase read units 4c + h (c = 0..3) of two adjacent slots; the swizzle maps
// them to eight distinct 16-byte bank groups.
__device__ __forceinline__ int staged_unit(int unit, int slot) {
    return unit ^ ((unit >> 2) & 2) ^ (slot & 1);
}

// Runs the K loop of one CTA for one codec over the first Blocks eight-token column blocks and
// leaves each warp's partial sums in acc[.][0, Blocks) (the other blocks stay zero).
//
// Pipeline: the CTA's threads stage the activations of KW chunks for the first 8 * Blocks token
// slots per stage with 16-byte cp.async copies into kBuffers rotating buffers (zero-filled for
// empty slots); each iteration waits for its stage, then a block barrier publishes it and retires
// the buffer the previous iteration read before the next stage is issued into it. Warp (kw, rw)
// computes chunk stage * KW + kw of tile tile_group * RW + rw; its weights come straight from
// global memory into registers, one chunk ahead. With more than one stage, thread 0 claims the
// CTA's next work item into `ticket` at the last stage (moe_expert_matmul_kernel), so the atomic's
// latency overlaps that stage's MMAs; a single-stage plan claims after the K loop instead, so that
// a CTA stalled on a slow record does not hold its next item.
template <Matrix M, int C, int Blocks, class Codec>
__device__ __forceinline__ void accumulate(const Codec& codec, const MatmulParams& p,
                                           std::uint8_t* stages, const int* columns,
                                           float (&acc)[2][C / 8][4], int& ticket) {
    using P        = Plan<M, C>;
    const int tid  = static_cast<int>(threadIdx.x);
    const int lane = tid % kWarpSize;
    const int kw   = (tid / kWarpSize) % P::kWarpsK;
    const int g    = lane / 4;
    const int c    = lane % 4;
    static_assert(Blocks >= 1 && Blocks <= C / 8);
    // Only the first 8 * Blocks token slots are staged and multiplied.
    constexpr int kStageTasks = Blocks * 8 * P::kWarpsK * kChunkUnits;

    const auto load_stage = [&](int buffer, int stage) {
        std::uint8_t* base = stages + buffer * P::kStageBytes;
#pragma unroll
        for (int task = tid; task < kStageTasks; task += P::kThreads) {
            const int slot   = task / (P::kWarpsK * kChunkUnits);
            const int within = task % (P::kWarpsK * kChunkUnits);
            const int chunk  = within / kChunkUnits;
            const int unit   = within % kChunkUnits;
            const int column = columns[slot];
            const std::uint16_t* source =
                p.activations + static_cast<std::int64_t>(column < 0 ? 0 : column) * P::Shape::kK +
                (stage * P::kWarpsK + chunk) * kChunkK + unit * 8;
            cp_async_zfill<16, Cache::cg>(base + slot * P::kRowBytes + chunk * (kChunkK * 2) +
                                              staged_unit(unit, slot) * 16,
                                          source, column < 0 ? 0 : 16);
        }
    };

    typename Codec::Chunk current;
    typename Codec::Chunk next;
    codec.load(next, kw);
    if constexpr (P::kBuffers == 1) {
        load_stage(0, 0);
        cp_commit();
    } else {
#pragma unroll
        for (int stage = 0; stage < P::kBuffers - 1; ++stage) {
            load_stage(stage, stage);
            cp_commit();
        }
    }

#pragma unroll 1
    for (int stage = 0; stage < P::kStages; ++stage) {
        const int q = stage * P::kWarpsK + kw;
        if constexpr (P::kBuffers == 1) {
            cp_wait<0>();
            __syncthreads();
        } else {
            cp_wait<P::kBuffers - 2>();
            __syncthreads();
            const int ahead = stage + P::kBuffers - 1;
            if (ahead < P::kStages) { load_stage(ahead % P::kBuffers, ahead); }
            cp_commit();
        }
        current = next;
        if (stage + 1 < P::kStages) {
            codec.load(next, q + P::kWarpsK);
        } else if (P::kStages > 1 && tid == 0) {
            ticket = atomicAdd(&p.schedule->next_item[p.launch], 1);
        }
        codec.prepare(current);

        const std::uint8_t* buffer =
            stages + (stage % P::kBuffers) * P::kStageBytes + kw * (kChunkK * 2);
#pragma unroll
        for (int pair = 0; pair < kStepsPerChunk / 2; ++pair) {
            uint4 b[Blocks];
#pragma unroll
            for (int n = 0; n < Blocks; ++n) {
                const int slot = n * 8 + g;
                b[n]           = load_vec<uint4>(buffer + slot * P::kRowBytes +
                                                 staged_unit(4 * c + pair, slot) * 16);
            }
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int step = 2 * pair + half;
#pragma unroll
                for (int frag = 0; frag < 2; ++frag) {
                    unsigned a[4];
                    codec.fragment(current, step, frag, a);
#pragma unroll
                    for (int n = 0; n < Blocks; ++n) {
                        mma_bf16(acc[frag][n][0], acc[frag][n][1], acc[frag][n][2], acc[frag][n][3],
                                 a[0], a[1], a[2], a[3], half != 0 ? b[n].z : b[n].x,
                                 half != 0 ? b[n].w : b[n].y);
                    }
                }
            }
        }
    }
    cp_wait<0>();
    if constexpr (P::kStages == 1) {
        if (tid == 0) { ticket = atomicAdd(&p.schedule->next_item[p.launch], 1); }
    }
}

// Runs body.template operator()<Blocks>() with the fewest eight-token column blocks that cover
// count tokens, a power of two up to C / 8: MMAs and activation staging of empty blocks are
// skipped.
template <int C, class Body>
__device__ __forceinline__ void with_column_blocks(int count, Body&& body) {
    const int blocks = (count + 7) / 8;
    if constexpr (C >= 64) {
        if (blocks > 4) {
            body.template operator()<8>();
            return;
        }
    }
    if constexpr (C >= 32) {
        if (blocks > 2) {
            body.template operator()<4>();
            return;
        }
    }
    if (blocks > 1) {
        body.template operator()<2>();
        return;
    }
    body.template operator()<1>();
}

// One work item: job job_index (up to C assignments of one expert) and the RW 32-row tiles of tile
// group tile_group, on a CTA of KW * RW warps (P::kThreads). A deferred job reads its record at
// deferred[expert] with the codec of its table entry. Dynamic shared memory: kBuffers activation
// stages of C slots x KW chunks x 256 bytes (reused after the K loop as the FP32 reduction buffer
// [KW][C][RW*32 + 4]), then C ints of source columns (-1 for an empty slot). Barriers: after the
// column table, one per stage, and two around the reduction. The next item may start without a
// barrier: before its column-table barrier it writes only the column table, which this item last
// reads in the K loop's final activation staging, before its first reduction barrier. Each output
// of the job is written by exactly one thread.
//
// GateUp writes gated[begin + slot][16 tile + i] = BF16(SiLU(s_g g) * (s_u u)) for the 16
// intermediate rows of each tile; Down writes contributions[a][32 tile + row] =
// (s_d * acc) * weight for sorted assignment a (s = the expert's matrix scales, the shared ones for
// the shared expert; weight = route_weights or shared_scale of a's token and slot).
//
// Thread 0 claims the CTA's next work item in the K loop's last stage and stores it to *claimed
// before the first reduction barrier, which publishes it.
template <Matrix M, int C>
__device__ __forceinline__ void matmul_item(const MatmulParams& p, int job_index, int tile_group,
                                            std::uint8_t* shared, int* claimed) {
    using P              = Plan<M, C>;
    const MoeJob job     = p.jobs[job_index];
    std::uint8_t* stages = shared;
    int* columns         = reinterpret_cast<int*>(shared + P::kStagingBytes);
    const int tid        = static_cast<int>(threadIdx.x);
    const int lane       = tid % kWarpSize;
    const int warp       = tid / kWarpSize;
    const int kw         = warp % P::kWarpsK;
    const int rw         = warp / P::kWarpsK;
    const int tile       = tile_group * P::kWarpsRows + rw;
    const int g          = lane / 4;
    const int c          = lane % 4;

    for (int slot = tid; slot < C; slot += P::kThreads) {
        int column = -1;
        if (slot < job.count) {
            const int position = job.begin + slot;
            column             = M == Matrix::GateUp ? p.sorted[position] / kMoeSlots : position;
        }
        columns[slot] = column;
    }
    __syncthreads();

    float acc[2][C / 8][4] = {};
    int ticket             = 0; // thread 0's claim
    // The shared expert is an NVFP4G16 record outside the table; a deferred record keeps the codec
    // tag of its table entry.
    const bool shared_expert = job.expert == kMoeSharedExpert;
    std::uint64_t entry      = reinterpret_cast<std::uintptr_t>(p.shared_record);
    if (!shared_expert) {
        entry = p.expert_table[job.expert];
        if (p.range == JobRange::Deferred) { entry = p.deferred[job.expert] | (entry & 1U); }
    }
    const bool f3            = (entry & 1U) != 0;
    const std::int64_t chunk = f3 ? kMoeF3ChunkBytes : kMoeNvfp4ChunkBytes;
    const std::int64_t first =
        (P::Shape::kFirstChunk + static_cast<std::int64_t>(tile) * P::Shape::kChunks) * chunk;
    const std::uint8_t* source = table_address(entry & ~std::uint64_t{1}) + first;
    if (f3) {
        F3Codec codec;
        codec.tile_chunks = source;
        codec.lane        = lane;
        codec.books[0]    = p.books[0];
        codec.books[1]    = p.books[1];
        with_column_blocks<C>(job.count, [&]<int Blocks>() {
            accumulate<M, C, Blocks>(codec, p, stages, columns, acc, ticket);
        });
    } else {
        Nvfp4Codec codec;
        codec.tile_chunks = source;
        codec.lane        = lane;
        with_column_blocks<C>(job.count, [&]<int Blocks>() {
            accumulate<M, C, Blocks>(codec, p, stages, columns, acc, ticket);
        });
    }

    if (tid == 0) { *claimed = ticket; }

    // Partial sums to the reduction buffer: thread (g, c) holds rows 16 frag + g (+8) and slots
    // 8 n + 2 c (+1). The 4-float row padding keeps these stores conflict-free.
    __syncthreads();
    auto* reduction = reinterpret_cast<float*>(stages);
#pragma unroll
    for (int frag = 0; frag < 2; ++frag) {
#pragma unroll
        for (int n = 0; n < C / 8; ++n) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int row  = 16 * frag + g + 8 * (j / 2);
                const int slot = 8 * n + 2 * c + (j & 1);
                reduction[(kw * C + slot) * P::kReductionStride + rw * kTileRows + row] =
                    acc[frag][n][j];
            }
        }
    }
    __syncthreads();

    const auto reduced = [&](int slot, int local_row) {
        float sum = reduction[slot * P::kReductionStride + local_row];
#pragma unroll
        for (int part = 1; part < P::kWarpsK; ++part) {
            sum += reduction[(part * C + slot) * P::kReductionStride + local_row];
        }
        return sum;
    };
    const float* scales = shared_expert
                              ? p.shared_matrix_scales
                              : p.matrix_scales + static_cast<std::int64_t>(job.expert) * 3;
    if constexpr (M == Matrix::GateUp) {
        const float gate_scale = scales[0];
        const float up_scale   = scales[1];
        constexpr int kRows    = P::kWarpsRows * 16;
        for (int task = tid; task < C * kRows; task += P::kThreads) {
            const int slot = task / kRows;
            if (slot >= job.count) { break; }
            const int row       = task % kRows;
            const int local_row = (row / 16) * kTileRows + row % 16;
            const float gate    = reduced(slot, local_row) * gate_scale;
            const float up      = reduced(slot, local_row + 16) * up_scale;
            p.gated[static_cast<std::int64_t>(job.begin + slot) * kMoeIntermediate +
                    tile_group * kRows + row] =
                __bfloat16_as_ushort(__float2bfloat16_rn(silu(gate) * up));
        }
    } else {
        const float down_scale = scales[2];
        constexpr int kRows    = P::kWarpsRows * kTileRows;
        for (int task = tid; task < C * kRows; task += P::kThreads) {
            const int slot = task / kRows;
            if (slot >= job.count) { break; }
            const int row        = task % kRows;
            const int assignment = p.sorted[job.begin + slot];
            const int token      = assignment / kMoeSlots;
            const int token_slot = assignment % kMoeSlots;
            const float weight   = token_slot == kMoeTopK
                                       ? p.shared_scale[token]
                                       : p.route_weights[token * kMoeTopK + token_slot];
            p.contributions[static_cast<std::int64_t>(assignment) * kMoeHidden +
                            tile_group * kRows + row] = reduced(slot, row) * down_scale * weight;
        }
    }
}

// Grid: at most the CTAs resident at once (launch_matmul), block P::kThreads. The launch's work
// items are (job, tile group) pairs of its job range, w = (job - first) * kTileGroups + tile group
// for first = job_count[0] in the deferred range, else 0. Every item is claimed from the launch's
// next_item counter (a claim returning w < items takes item w): a CTA claims its first item on
// entry and each next one in its current item's last K stage (after the K loop for a single-stage
// plan), and exits once a claim passes the range's end, or on entry, before any claim or barrier,
// when the range is empty. Claims follow the K work, so a CTA held up by a slow record (mapped
// host memory) takes fewer items, and a CTA that starts
// late, behind a concurrent kernel's CTAs, finds the items already taken instead of owning any, as
// the hardware scheduler balances a grid of one CTA per item. The claim is broadcast through the
// shared `claimed`: thread 0 writes it before a barrier that precedes every thread's read, and the
// next write follows the next item's column-table barrier, after every thread's read.
template <Matrix M, int C>
__global__ void __launch_bounds__(Plan<M, C>::kThreads, Schedule<M, C>::kMinCtas)
    moe_expert_matmul_kernel(const __grid_constant__ MatmulParams p) {
    using P                       = Plan<M, C>;
    const std::int32_t* job_count = p.schedule->job_count;
    const int first_job           = p.range == JobRange::Deferred ? job_count[0] : 0;
    const int end_job             = p.range == JobRange::Resident ? job_count[0] : job_count[1];
    const int items               = (end_job - first_job) * P::kTileGroups;
    if (items == 0) { return; }
    extern __shared__ __align__(16) std::uint8_t moe_shared[];
    __shared__ int claimed;
    if (threadIdx.x == 0) { claimed = atomicAdd(&p.schedule->next_item[p.launch], 1); }
    __syncthreads();
    for (int item = claimed; item < items; item = claimed) {
        matmul_item<M, C>(p, first_job + item / P::kTileGroups, item % P::kTileGroups, moe_shared,
                          &claimed);
    }
}

// --- Grouping kernel ---------------------------------------------------------------------------

constexpr int kGroupThreads = kMoeExperts; // thread e owns routed expert e in the plan phase
constexpr int kGroupWarps   = kGroupThreads / kWarpSize;

// Exclusive prefix sum over the 512 threads of the block; total receives the block sum. Contains
// three block barriers; every thread must call it.
__device__ int block_exclusive_sum(int value, int* scratch, int& total) {
    const int lane = static_cast<int>(threadIdx.x) % kWarpSize;
    const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
    int inclusive  = value;
#pragma unroll
    for (int offset = 1; offset < kWarpSize; offset *= 2) {
        const int lower = __shfl_up_sync(kFullWarpMask, inclusive, offset);
        if (lane >= offset) { inclusive += lower; }
    }
    if (lane == kWarpSize - 1) { scratch[warp] = inclusive; }
    __syncthreads();
    if (warp == 0) {
        const int warp_total = lane < kGroupWarps ? scratch[lane] : 0;
        int prefix           = warp_total;
#pragma unroll
        for (int offset = 1; offset < kWarpSize; offset *= 2) {
            const int lower = __shfl_up_sync(kFullWarpMask, prefix, offset);
            if (lane >= offset) { prefix += lower; }
        }
        if (lane < kGroupWarps) { scratch[lane] = prefix - warp_total; }
        if (lane == kGroupWarps - 1) { scratch[kGroupWarps] = prefix; }
    }
    __syncthreads();
    const int exclusive = scratch[warp] + inclusive - value;
    total               = scratch[kGroupWarps];
    __syncthreads();
    return exclusive;
}

// Grid: one CTA of 512 threads. Builds the stable expert grouping and the job table.
//
// Warp w walks the routed assignments r = 10 t + j of tokens [w * ceil(T/16), (w+1) * ceil(T/16))
// in ascending r, 32 at a time; __match_any_sync groups the lanes holding one expert, so a lane's
// rank among earlier assignments of its expert in the warp is the running count plus the number of
// lower matching lanes. Shared memory counts[16][512] (32 KiB) holds per-warp counts after the
// first walk and per-warp bases (exclusive over warps) after the plan phase; the second walk
// scatters assignment 11 t + j to sorted[T + offset_e + base_{w,e} + rank]. The shared expert's
// assignments 11 t + 10 occupy sorted[0, T) in token order. Jobs: the shared expert's ceil(T/C)
// first, then each routed expert's ceil(n_e/C) in ascending expert order, those of deferred
// experts (deferred non-null and deferred[e] != 0) after all others; job_count[0] counts the jobs
// before the deferred ones and job_count[1] all jobs. The matmul launches' next_item counters are
// zeroed.
__global__ void __launch_bounds__(kGroupThreads)
    moe_group_kernel(const std::int32_t* __restrict__ expert_ids,
                     const std::uint64_t* __restrict__ deferred, int tokens, int token_tile,
                     std::int32_t* __restrict__ sorted, MoeJob* __restrict__ jobs,
                     MoeJobSchedule* __restrict__ schedule) {
    __shared__ int counts[kGroupWarps][kMoeExperts];
    __shared__ int routed_offset[kMoeExperts];
    __shared__ int scratch[kGroupWarps + 1];
    const int tid  = static_cast<int>(threadIdx.x);
    const int lane = tid % kWarpSize;
    const int warp = tid / kWarpSize;
#pragma unroll
    for (int w = 0; w < kGroupWarps; ++w) { counts[w][tid] = 0; }
    __syncthreads();

    const int warp_tokens = (tokens + kGroupWarps - 1) / kGroupWarps;
    const int first       = min(tokens, warp * warp_tokens) * kMoeTopK;
    const int last        = min(tokens, (warp + 1) * warp_tokens) * kMoeTopK;
    const unsigned lower  = (1U << lane) - 1U;
    const auto walk       = [&](bool scatter) {
        for (int base = first; base < last; base += kWarpSize) {
            const int r      = base + lane;
            const bool valid = r < last;
            // Ids are promised in [0,512); the clamp only keeps a violation inside the tables.
            const int expert     = valid ? min(max(expert_ids[r], 0), kMoeExperts - 1) : -1;
            const unsigned peers = __match_any_sync(kFullWarpMask, expert);
            if (valid && scatter) {
                const int rank = counts[warp][expert] + __popc(peers & lower);
                sorted[tokens + routed_offset[expert] + rank] =
                    (r / kMoeTopK) * kMoeSlots + r % kMoeTopK;
            }
            __syncwarp();
            if (valid && lane == kWarpSize - 1 - __clz(peers)) {
                counts[warp][expert] += __popc(peers);
            }
            __syncwarp();
        }
    };
    walk(false);
    __syncthreads();

    int assignments = 0;
#pragma unroll
    for (int w = 0; w < kGroupWarps; ++w) {
        const int count = counts[w][tid];
        counts[w][tid]  = assignments;
        assignments += count;
    }
    int unused                = 0;
    routed_offset[tid]        = block_exclusive_sum(assignments, scratch, unused);
    const int expert_jobs     = (assignments + token_tile - 1) / token_tile;
    const int shared_jobs     = (tokens + token_tile - 1) / token_tile;
    const bool late           = deferred != nullptr && expert_jobs > 0 && deferred[tid] != 0;
    int resident_jobs         = 0;
    int late_jobs             = 0;
    const int resident_offset = block_exclusive_sum(late ? 0 : expert_jobs, scratch, resident_jobs);
    const int late_offset     = block_exclusive_sum(late ? expert_jobs : 0, scratch, late_jobs);
    const int job_base = shared_jobs + (late ? resident_jobs + late_offset : resident_offset);
    for (int i = 0; i < expert_jobs; ++i) {
        jobs[job_base + i] = MoeJob{tid, tokens + routed_offset[tid] + i * token_tile,
                                    min(token_tile, assignments - i * token_tile)};
    }
    for (int i = tid; i < shared_jobs; i += kGroupThreads) {
        jobs[i] =
            MoeJob{kMoeSharedExpert, i * token_tile, min(token_tile, tokens - i * token_tile)};
    }
    for (int t = tid; t < tokens; t += kGroupThreads) { sorted[t] = t * kMoeSlots + kMoeTopK; }
    if (tid == 0) {
        schedule->job_count[0] = shared_jobs + resident_jobs;
        schedule->job_count[1] = shared_jobs + resident_jobs + late_jobs;
    }
    if (tid < kMoeMatmulLaunches) { schedule->next_item[tid] = 0; }
    __syncthreads();
    walk(true);
}

// --- Merge kernel ------------------------------------------------------------------------------

constexpr int kMergeThreads      = 128;
constexpr int kMergeRowsPerBlock = kMergeThreads * 4;

// Grid: (2560 / 512, T); block 128. Thread i of block (b, t) sums rows 512 b + 4 i .. +3 of the
// eleven contributions of token t in slot order (float4 loads; rows 16-byte aligned) and rounds
// each sum once to BF16.
__global__ void __launch_bounds__(kMergeThreads)
    moe_merge_kernel(const float* __restrict__ contributions, std::uint16_t* __restrict__ y) {
    const int token = static_cast<int>(blockIdx.y);
    const int row =
        static_cast<int>(blockIdx.x) * kMergeRowsPerBlock + static_cast<int>(threadIdx.x) * 4;
    const float* source =
        contributions + static_cast<std::int64_t>(token) * kMoeSlots * kMoeHidden + row;
    float4 sum = load_vec<float4>(source);
#pragma unroll
    for (int slot = 1; slot < kMoeSlots; ++slot) {
        const float4 value = load_vec<float4>(source + slot * kMoeHidden);
        sum.x += value.x;
        sum.y += value.y;
        sum.z += value.z;
        sum.w += value.w;
    }
    const __nv_bfloat162 low  = __floats2bfloat162_rn(sum.x, sum.y);
    const __nv_bfloat162 high = __floats2bfloat162_rn(sum.z, sum.w);
    store_vec(y + static_cast<std::int64_t>(token) * kMoeHidden + row,
              make_uint2(*reinterpret_cast<const unsigned*>(&low),
                         *reinterpret_cast<const unsigned*>(&high)));
}

static_assert(kMoeHidden % kMergeRowsPerBlock == 0);

// --- Launch ------------------------------------------------------------------------------------

F3Lut make_f3_lut(std::uint32_t book) {
    // BF16 bits of the E2M1 magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6.
    constexpr std::uint16_t kMagnitudes[8] = {0x0000, 0x3F00, 0x3F80, 0x3FC0,
                                              0x4000, 0x4040, 0x4080, 0x40C0};
    F3Lut lut{};
    for (int entry = 0; entry < 8; ++entry) {
        const std::uint32_t nibble = (book >> (4 * entry)) & 0xFU;
        const std::uint32_t bits =
            kMagnitudes[nibble & 0x7U] | ((nibble & 0x8U) != 0 ? 0x8000U : 0U);
        lut.lo[entry / 4] |= (bits & 0xFFU) << (8 * (entry % 4));
        lut.hi[entry / 4] |= (bits >> 8) << (8 * (entry % 4));
    }
    return lut;
}

// CTAs of one instantiation resident on the device at once, queried once (after raising its
// dynamic shared-memory limit where it needs more than the default 48 KiB). The product runs on one
// device, the current one at the first call.
template <Matrix M, int C>
std::int64_t one_wave_ctas() {
    using P                        = Plan<M, C>;
    constexpr auto kernel          = moe_expert_matmul_kernel<M, C>;
    static const std::int64_t ctas = [] {
        if constexpr (P::kSharedBytes > 48 * 1024) {
            CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            P::kSharedBytes));
        }
        int per_sm = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, P::kThreads,
                                                                 P::kSharedBytes));
        if (per_sm < 1) { throw std::logic_error("moe_experts: a matmul CTA does not fit an SM"); }
        int device          = 0;
        int multiprocessors = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(
            cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device));
        return static_cast<std::int64_t>(per_sm) * multiprocessors;
    }();
    return ctas;
}

// Launches one matmul over a job range of at most max_jobs jobs. The grid is the work-item bound
// or one wave of resident CTAs, whichever is smaller: CTAs find no item past the range's real
// ones and exit, and finished CTAs claim the items past the grid (moe_expert_matmul_kernel), so
// the cost follows the real job count rather than the bound.
template <Matrix M, int C>
void launch_matmul(const MatmulParams& params, std::int32_t max_jobs, cudaStream_t stream) {
    using P                  = Plan<M, C>;
    const std::int64_t bound = static_cast<std::int64_t>(max_jobs) * P::kTileGroups;
    const auto blocks        = static_cast<unsigned>(std::min(bound, one_wave_ctas<M, C>()));
    moe_expert_matmul_kernel<M, C><<<blocks, P::kThreads, P::kSharedBytes, stream>>>(params);
    CUDA_CHECK(cudaGetLastError());
}

template <Matrix M>
void launch_matmul(const MatmulParams& params, std::int32_t token_tile, std::int32_t max_jobs,
                   cudaStream_t stream) {
    switch (token_tile) {
    case 16:
        launch_matmul<M, 16>(params, max_jobs, stream);
        return;
    case 32:
        launch_matmul<M, 32>(params, max_jobs, stream);
        return;
    case 64:
        launch_matmul<M, 64>(params, max_jobs, stream);
        return;
    default:
        throw std::logic_error("moe_experts: unsupported token tile");
    }
}

} // namespace

void moe_experts_launch(const MoeExpertsOperands& operands, const MoeExpertsWorkspace& workspace,
                        cudaStream_t stream) {
    const std::int32_t tokens     = operands.tokens;
    const std::int32_t token_tile = moe_experts_token_tile(tokens);
    const std::int32_t max_jobs   = moe_experts_max_jobs(tokens);

    moe_group_kernel<<<1, kGroupThreads, 0, stream>>>(operands.expert_ids, operands.deferred_slots,
                                                      tokens, token_tile, workspace.sorted,
                                                      workspace.jobs, workspace.schedule);
    CUDA_CHECK(cudaGetLastError());

    MatmulParams params;
    params.expert_table         = operands.expert_table;
    params.deferred             = operands.deferred_slots;
    params.matrix_scales        = operands.matrix_scales;
    params.shared_record        = operands.shared_record;
    params.shared_matrix_scales = operands.shared_matrix_scales;
    params.sorted               = workspace.sorted;
    params.jobs                 = workspace.jobs;
    params.schedule             = workspace.schedule;
    params.route_weights        = operands.route_weights;
    params.shared_scale         = operands.shared_scale;
    params.gated                = workspace.gated;
    params.contributions        = workspace.contributions;

    // Without a deferral one gate|up and one down launch serve every job. With one, the shared and
    // resident experts' jobs run first, overlapping the copies that fill the deferred records on
    // the caller's other stream; the stream then waits for `deferred_ready` and runs the deferred
    // jobs, which are at most the routed jobs.
    const F3Lut gate_up_books[2] = {make_f3_lut(operands.f3_books[0]),
                                    make_f3_lut(operands.f3_books[1])};
    const F3Lut down_book        = make_f3_lut(operands.f3_books[2]);
    const auto run_range         = [&](JobRange range, std::int32_t jobs_bound) {
        // Counters 0 and 1 serve the first range's launches, 2 and 3 the deferred range's.
        const int launch   = range == JobRange::Deferred ? 2 : 0;
        params.range       = range;
        params.activations = operands.x;
        params.books[0]    = gate_up_books[0];
        params.books[1]    = gate_up_books[1];
        params.launch      = launch;
        launch_matmul<Matrix::GateUp>(params, token_tile, jobs_bound, stream);
        params.activations = workspace.gated;
        params.books[0]    = down_book;
        params.books[1]    = down_book;
        params.launch      = launch + 1;
        launch_matmul<Matrix::Down>(params, token_tile, jobs_bound, stream);
    };
    if (operands.deferred_slots == nullptr) {
        run_range(JobRange::All, max_jobs);
    } else {
        run_range(JobRange::Resident, max_jobs);
        CUDA_CHECK(cudaStreamWaitEvent(stream, operands.deferred_ready, 0));
        run_range(JobRange::Deferred, max_jobs - (tokens + token_tile - 1) / token_tile);
    }

    moe_merge_kernel<<<dim3(kMoeHidden / kMergeRowsPerBlock, static_cast<unsigned>(tokens)),
                       kMergeThreads, 0, stream>>>(workspace.contributions, operands.y);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
