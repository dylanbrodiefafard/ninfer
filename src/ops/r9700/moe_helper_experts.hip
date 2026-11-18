#include "ninfer/ops/moe_helper_experts.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

// --- Geometry ----------------------------------------------------------------------------------
//
// One launch serves the whole call. Its work items are (job, row block) pairs, where a job is
// one expert with every compact-token route entry naming it. A workgroup of four wave32 waves
// computes one item; each wave computes one *slice*: one 32-row tile of a record plane over 640
// consecutive K (five 128-K chunks), for the job's tokens.
//
//   gate|up item (job, tile r in [0,40)): wave w takes chunks 5w..5w+4 of gate|up tile r (gate
//     and up rows 16r..16r+15, K = 640w..640w+639); the four K quarters are summed in the
//     workgroup and the item writes gated = BF16(SiLU(s_g G x) * (s_u U x)) for its 16
//     intermediate rows.
//   down item (job, group d in [0,20)): wave w takes down tile 4d + w (rows 128d + 32w ..+31)
//     over the full K = 640 and writes each route's FP32 contribution s_d w_j (D gated).
//
// Items are claimed in order from one counter: every gate|up item (job-major), then every down
// item (job-major). A down item needs its job's 40 gate|up items complete; since those were
// claimed earlier by running workgroups, which never wait on later items, the wait cannot
// deadlock whatever the residency.
//
// A slice runs one of two paths, chosen per job by its token count:
//   SIMT (fewer than kWmmaMinTokens tokens): the wave holds all five chunks' record bytes in
//     registers and, per token, decodes each row and accumulates v_dot2 products against the
//     token's activations staged in LDS. Decode work scales with the tokens.
//   WMMA (kWmmaMinTokens or more): per 16-token block, each row is decoded once into an LDS tile
//     and multiplied with BF16 WMMA against activations read from global memory (multiply_wmma).
// A route's contribution is the FP32 down output times its weight; contributions are summed per
// token in route-slot order by the down item that completes them (no float atomics).

constexpr int kWaveSize      = 32;
constexpr int kWaves         = 4;
constexpr int kThreads       = kWaves * kWaveSize;
constexpr int kRoutes        = kMoeHelperRoutesPerToken;
constexpr int kSliceChunks   = 5;
constexpr int kSliceK        = kSliceChunks * 128; // 640
constexpr int kGateUpTiles   = 40;                 // gate|up items per job
constexpr int kDownGroups    = 20;                 // down items per job, four tiles each
constexpr int kItemsPerJob   = kGateUpTiles + kDownGroups;
constexpr int kStageRowBytes = kSliceK * 2;         // one token's staged BF16 activations
constexpr int kStageUnits    = kStageRowBytes / 16; // 16-byte units of one staged row
// Jobs of at least this many tokens take the WMMA path, fewer the SIMT path. Measured on the
// R9700 (F3 records, bench/ops/r9700/moe_helper_experts_bench.hip): WMMA wins from three tokens
// per expert when a call's bytes are small (latency-bound), but in throughput-bound calls (T = 48,
// ~240 experts) jobs of three to five tokens run faster on SIMT, whose five-chunk register
// prefetch pipelines memory better than WMMA's two-chunk ring.
constexpr int kWmmaMinTokens = 6;
constexpr int kWmmaTokens    = 16; // tokens of one WMMA block (the M extent)
// WMMA B tile of one 32-K unit: 16 rows of 32 BF16 at an 80-byte stride (16 bytes of padding
// spread the 16 rows a B-fragment read touches over distinct banks), then FP32 scales [16][2].
constexpr int kTileRowBytes    = 80;
constexpr int kTileScaleOffset = 16 * kTileRowBytes;
// Per-wave LDS scratch: the SIMT stage (one token, 1280 bytes) or the WMMA tile (1408 bytes),
// then a gate|up item's FP32 partial sums [16 tokens][32 rows] (2048 bytes) in the same bytes.
constexpr int kWaveScratchBytes = kWmmaTokens * 32 * 4;
static_assert(kStageRowBytes <= kWaveScratchBytes &&
              kTileScaleOffset + 16 * 2 * 4 <= kWaveScratchBytes);
constexpr int kDownPlaneChunk       = static_cast<int>(kMoeGateUpChunks);
constexpr std::uint32_t kInvalidKey = 0xFFFFFFFFU;

static_assert(kGateUpTiles * 20 == kMoeGateUpChunks && kDownGroups * kWaves * 5 == kMoeDownChunks);
static_assert(kWaves * kSliceK == kMoeHidden && kSliceK == kMoeIntermediate);
// Sort key: expert (9 bits) above the route entry index e = 10 t + j (9 bits).
static_assert(kMoeHelperExperts <= 512 && kMoeHelperMaxTokens * kRoutes <= 512);

// Byte-permute tables of one 8-entry BF16 book: byte v of {lo[1]:lo[0]} and {hi[1]:hi[0]} is the
// low and high byte of entry v.
struct Lut {
    std::uint32_t lo[2];
    std::uint32_t hi[2];
};

// E2M1 magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6 as BF16 bits.
constexpr std::uint16_t kE2m1Bf16[8] = {0x0000, 0x3F00, 0x3F80, 0x3FC0,
                                        0x4000, 0x4040, 0x4080, 0x40C0};

constexpr Lut make_lut(std::uint32_t book) {
    Lut lut{};
    for (int v = 0; v < 8; ++v) {
        const std::uint32_t nibble = (book >> (4 * v)) & 0xFU;
        const std::uint32_t bits   = kE2m1Bf16[nibble & 7U] | ((nibble & 8U) != 0 ? 0x8000U : 0U);
        lut.lo[v / 4] |= (bits & 0xFFU) << (8 * (v % 4));
        lut.hi[v / 4] |= (bits >> 8) << (8 * (v % 4));
    }
    return lut;
}

// The identity book (entry v = E2M1 nibble v): its first half is the E2M1 magnitude table.
constexpr Lut kE2m1Magnitudes = make_lut(0x76543210U);

// One job: an expert and its route entries, sorted positions [begin, begin + count).
struct alignas(16) Job {
    std::uint64_t entry; // expert-table entry: record address | codec
    std::int32_t expert;
    std::uint16_t begin;
    std::uint16_t count;
};

// Dynamic LDS of one workgroup, for max_tokens Tm:
//   scratch [kWaves][kWaveScratchBytes]: each wave's SIMT stage or WMMA tile, then a gate|up
//           item's partial sums. During the prologue it holds the sort keys U32 [10 Tm].
//   jobs    Job [10 Tm]
//   sorted  U16 [10 Tm]: route entry indices e = 10 t + j, grouped by expert, ascending e within
//   mask    U16 [Tm]: bit j set when entry j of token t is active
//   merges  U16 [Tm]: tokens one down item completes
struct LdsLayout {
    int jobs;
    int sorted;
    int mask;
    int merges;
    int bytes;
};

constexpr int align16(int bytes) { return (bytes + 15) / 16 * 16; }

constexpr LdsLayout lds_layout(int max_tokens) {
    const int routes = kRoutes * max_tokens;
    LdsLayout layout{};
    layout.jobs   = kWaves * kWaveScratchBytes;
    layout.sorted = layout.jobs + routes * static_cast<int>(sizeof(Job));
    layout.mask   = layout.sorted + align16(routes * 2);
    layout.merges = layout.mask + align16(max_tokens * 2);
    layout.bytes  = layout.merges + align16(max_tokens * 2);
    return layout;
}

static_assert(kRoutes * kMoeHelperMaxTokens * 4 <= kWaves * kWaveScratchBytes); // keys fit

// Device workspace for max_tokens Tm (offsets from the base, 256-byte aligned):
//   control        U32 [2]: work-item claims, workgroup exits
//   job_done       U32 [10 Tm]: completed gate|up items per job
//   merge          U32 [Tm][20]: completed route contributions per (token, down group)
//   gated          BF16 [10 Tm][640]: gate|up products by sorted position
//   contributions  FP32 [10 Tm][2560]: route contributions by entry index e
struct WorkspaceLayout {
    std::size_t job_done;
    std::size_t merge;
    std::size_t gated;
    std::size_t contributions;
    std::size_t bytes;
};

constexpr std::size_t align256(std::size_t bytes) { return (bytes + 255) / 256 * 256; }

constexpr WorkspaceLayout workspace_layout(std::int32_t max_tokens) {
    const auto routes = static_cast<std::size_t>(kRoutes) * static_cast<std::size_t>(max_tokens);
    WorkspaceLayout layout{};
    layout.job_done = 256;
    layout.merge    = layout.job_done + align256(routes * 4);
    layout.gated = layout.merge + align256(static_cast<std::size_t>(max_tokens) * kDownGroups * 4);
    layout.contributions = layout.gated + align256(routes * kMoeIntermediate * 2);
    layout.bytes         = layout.contributions + align256(routes * kMoeHidden * 4);
    return layout;
}

struct Params {
    const std::int32_t* tokens;
    const std::uint16_t* x;
    const MoeHelperRoute* routes;
    const std::uint64_t* expert_table;
    const float* matrix_scales;
    float* y;
    std::uint32_t* control;
    std::uint32_t* job_done;
    std::uint32_t* merge;
    std::uint16_t* gated;
    float* contributions;
    Lut books[3]; // gate, up, down
    std::int32_t max_tokens;
    LdsLayout lds;
};

// --- Synchronization -----------------------------------------------------------------------------

// A workgroup barrier ordering only LDS: __syncthreads' workgroup fence also waits for every
// global load in flight, which would drain a slice's weight loads.
__device__ __forceinline__ void lds_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    // NOLINTNEXTLINE(portability-no-assembler): this wave's LDS accesses; no builtin on gfx12
    asm volatile("s_wait_dscnt 0x0" ::: "memory");
    __builtin_amdgcn_s_barrier_signal(-1);
    __builtin_amdgcn_s_barrier_wait(-1);
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// Orders one wave's LDS writes before its later LDS reads by other lanes of the same wave.
__device__ __forceinline__ void wave_lds_sync() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront", "local");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront", "local");
}

__device__ __forceinline__ std::uint32_t atomic_add_agent(std::uint32_t* counter, int order) {
    return __hip_atomic_fetch_add(counter, 1U, order, __HIP_MEMORY_SCOPE_AGENT);
}

// --- Codecs ------------------------------------------------------------------------------------
//
// Lane L = 4 g + c of a wave owns, in every chunk, local rows rho_k = g + 8 k (k = 0..3) and K
// offsets 32 c .. 32 c + 31 (ninfer/ops/moe_experts.h). A codec loads the lane's bytes of one chunk
// into registers (each wave-wide load is 512 contiguous bytes) and decodes row k into sixteen
// BF16 pairs in *run order*: K offsets 32 c + 8 j + {0..7} (run j = 0..3) as the pairs
// (0,2), (4,6), (1,3), (5,7). The staged activations use the same order (stage_store), so each
// v_dot2 pairs a weight with the activation of its own K. Every decoded value is a signed E2M1
// value (at most two significant bits), exact in BF16; the E4M3 block scale is applied to the
// FP32 sum of its group's products.

// Eight nibbles (K offsets 0..7 of a run, nibble n at bits 4n..4n+3) through an 8-entry table;
// the nibble's bit 3 is ignored. Returns the high-byte lookups so a caller can set sign bits.
__device__ __forceinline__ void lookup_run(std::uint32_t word, const Lut& lut, std::uint32_t& el,
                                           std::uint32_t& eh, std::uint32_t& ol,
                                           std::uint32_t& oh) {
    const std::uint32_t even = word & 0x07070707U;        // nibbles 0, 2, 4, 6
    const std::uint32_t odd  = (word >> 4) & 0x07070707U; // nibbles 1, 3, 5, 7
    el                       = __builtin_amdgcn_perm(lut.lo[1], lut.lo[0], even);
    eh                       = __builtin_amdgcn_perm(lut.hi[1], lut.hi[0], even);
    ol                       = __builtin_amdgcn_perm(lut.lo[1], lut.lo[0], odd);
    oh                       = __builtin_amdgcn_perm(lut.hi[1], lut.hi[0], odd);
}

// Low/high-byte lookups of a run's even nibbles (el, eh: nibbles 0, 2, 4, 6) and odd nibbles
// (ol, oh: 1, 3, 5, 7) to the run's four BF16 pairs in run order.
__device__ __forceinline__ void pair_run(std::uint32_t el, std::uint32_t eh, std::uint32_t ol,
                                         std::uint32_t oh, std::uint32_t* out) {
    out[0] = __builtin_amdgcn_perm(eh, el, 0x05010400U); // nibbles 0, 2
    out[1] = __builtin_amdgcn_perm(eh, el, 0x07030602U); // nibbles 4, 6
    out[2] = __builtin_amdgcn_perm(oh, ol, 0x05010400U); // nibbles 1, 3
    out[3] = __builtin_amdgcn_perm(oh, ol, 0x07030602U); // nibbles 5, 7
}

// Exact E4M3FN (bias 7, subnormals 2^-9 m) to FP32; the NaN encodings are not valid scales.
__device__ __forceinline__ float e4m3_to_float(std::uint32_t bits) {
    const std::uint32_t magnitude = bits & 0x7FU;
    const float value = magnitude >= 8U ? __uint_as_float((magnitude << 20) + 0x3C000000U)
                                        : static_cast<float>(magnitude) * 0x1p-9F;
    return (bits & 0x80U) != 0 ? -value : value;
}

// Hides a register's value from the optimizer at this point: work derived from it cannot move
// above the point, work feeding it cannot move below, and volatile asm keeps program order
// (see Slice::multiply).
__device__ __forceinline__ void launder(std::uint32_t& value) {
    // NOLINTNEXTLINE(portability-no-assembler): an empty asm with a read-write VGPR operand
    asm volatile("" : "+v"(value));
}

__device__ __forceinline__ void launder(float& value) {
    // NOLINTNEXTLINE(portability-no-assembler): an empty asm with a read-write VGPR operand
    asm volatile("" : "+v"(value));
}

__device__ __forceinline__ void launder(int& value) {
    // NOLINTNEXTLINE(portability-no-assembler): an empty asm with a read-write VGPR operand
    asm volatile("" : "+v"(value));
}

template <class Vector>
__device__ __forceinline__ void launder_vector(Vector& value) {
    // NOLINTNEXTLINE(portability-no-assembler): an empty asm with a read-write VGPR tuple operand
    asm volatile("" : "+v"(value));
}

using Bf16x2 = __bf16 __attribute__((ext_vector_type(2)));

// Record bytes in the global address space. A record address comes from an integer table entry,
// which would otherwise compile to flat loads; flat loads also count on the LDS wait counter, so
// every LDS wait would drain the slice's in-flight record loads.
using RecordBytes = const __attribute__((address_space(1))) std::uint8_t*;

using U32x2 = std::uint32_t __attribute__((ext_vector_type(2)));
using U32x4 = std::uint32_t __attribute__((ext_vector_type(4)));

template <class T>
__device__ __forceinline__ T load_record(RecordBytes bytes) {
    return *reinterpret_cast<const __attribute__((address_space(1))) T*>(bytes);
}

__device__ __forceinline__ float dot2(std::uint32_t w, std::uint32_t x, float acc) {
    return __builtin_amdgcn_fdot2_f32_bf16(__builtin_bit_cast(Bf16x2, w),
                                           __builtin_bit_cast(Bf16x2, x), acc, false);
}

// Sum of `Pairs` BF16 pair products from index First, in two chains.
template <int First, int Pairs>
__device__ __forceinline__ float dot_pairs(const std::uint32_t (&w)[16],
                                           const std::uint32_t (&x)[16]) {
    float a = 0.0F;
    float b = 0.0F;
#pragma unroll
    for (int i = First; i < First + Pairs; i += 2) {
        a = dot2(w[i], x[i], a);
        b = dot2(w[i + 1], x[i + 1], b);
    }
    return a + b;
}

// gfx12 wave32 WMMA 16x16x16 BF16 -> FP32. Lane l holds A row l % 16 and B column l % 16 at K
// 8 (l / 16) .. + 7 (eight BF16), and D column l % 16 at rows 8 (l / 16) .. + 7.
using WmmaBf16 = __attribute__((__vector_size__(8 * sizeof(short)))) short;
using WmmaF32  = __attribute__((__vector_size__(8 * sizeof(float)))) float;

__device__ __forceinline__ WmmaF32 wmma(const WmmaBf16& a, const WmmaBf16& b, const WmmaF32& c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
}

// Reorders a decoded row from run order (pairs (0,2), (4,6), (1,3), (5,7) per run) to natural K
// order (pairs (0,1), (2,3), (4,5), (6,7)), the order the WMMA B tile holds.
__device__ __forceinline__ void natural_order(std::uint32_t (&w)[16]) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const std::uint32_t p0 = w[4 * j];
        const std::uint32_t p1 = w[4 * j + 1];
        const std::uint32_t q0 = w[4 * j + 2];
        const std::uint32_t q1 = w[4 * j + 3];
        w[4 * j]               = __builtin_amdgcn_perm(q0, p0, 0x05040100U);
        w[4 * j + 1]           = __builtin_amdgcn_perm(q0, p0, 0x07060302U);
        w[4 * j + 2]           = __builtin_amdgcn_perm(q1, p1, 0x05040100U);
        w[4 * j + 3]           = __builtin_amdgcn_perm(q1, p1, 0x07060302U);
    }
}

// F3G32 chunk: per lane one scale byte (unit k) and three code words per unit.
struct F3Codec {
    static constexpr int kChunkBytes = static_cast<int>(kMoeF3ChunkBytes);

    struct Raw {
        std::uint32_t scales;
        std::uint32_t words[3][4]; // [code word j][unit k]
    };

    __device__ __forceinline__ static Raw load(RecordBytes chunk, int lane) {
        Raw raw;
        raw.scales = load_record<std::uint32_t>(chunk + 4 * lane);
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            const auto value = load_record<U32x4>(chunk + 128 + 16 * (kWaveSize * j + lane));
            raw.words[j][0]  = value.x;
            raw.words[j][1]  = value.y;
            raw.words[j][2]  = value.z;
            raw.words[j][3]  = value.w;
        }
        return raw;
    }

    // The words row k decodes from (Slice::multiply).
    __device__ __forceinline__ static void launder_row(Raw& raw, int k) {
        if (k == 0) { launder(raw.scales); }
#pragma unroll
        for (int j = 0; j < 3; ++j) { launder(raw.words[j][k]); }
    }

    // Codes 0..23 are the low three bits of the words' nibbles; code 24 + n gathers bit j from the
    // high bit of nibble n of word j.
    __device__ __forceinline__ static void decode(const Raw& raw, int k, const Lut& lut,
                                                  std::uint32_t (&w)[16]) {
        const std::uint32_t w0 = raw.words[0][k];
        const std::uint32_t w1 = raw.words[1][k];
        const std::uint32_t w2 = raw.words[2][k];
        const std::uint32_t high =
            ((w0 >> 3) & 0x11111111U) | ((w1 >> 2) & 0x22222222U) | ((w2 >> 1) & 0x44444444U);
        const std::uint32_t runs[4] = {w0, w1, w2, high};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            std::uint32_t el, eh, ol, oh;
            lookup_run(runs[j], lut, el, eh, ol, oh);
            pair_run(el, eh, ol, oh, w + 4 * j);
        }
    }

    __device__ __forceinline__ static float scale(const Raw& raw, int k, int /*half*/) {
        return e4m3_to_float((raw.scales >> (8 * k)) & 0xFFU);
    }

    __device__ __forceinline__ static float accumulate(const std::uint32_t (&w)[16],
                                                       const std::uint32_t (&x)[16],
                                                       const float (&scales)[2], float acc) {
        return fmaf(dot_pairs<0, 16>(w, x), scales[0], acc);
    }

    // One 32-K unit (two WMMA K steps under one scale per row) into acc.
    __device__ __forceinline__ static void wmma_unit(const WmmaBf16 (&a)[2], const WmmaBf16 (&b)[2],
                                                     const float (&scales)[2], WmmaF32& acc) {
        const WmmaF32 sum = wmma(a[1], b[1], wmma(a[0], b[0], WmmaF32{}));
        acc += sum * scales[0];
    }
};

// NVFP4G16 chunk: per lane two scale bytes per unit (K halves) and four code words per unit.
struct Nvfp4Codec {
    static constexpr int kChunkBytes = static_cast<int>(kMoeNvfp4ChunkBytes);

    struct Raw {
        std::uint32_t scales[2];
        std::uint32_t words[4][4]; // [code word j][unit k]
    };

    __device__ __forceinline__ static Raw load(RecordBytes chunk, int lane) {
        Raw raw;
        const auto scales = load_record<U32x2>(chunk + 8 * lane);
        raw.scales[0]     = scales.x;
        raw.scales[1]     = scales.y;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const auto value = load_record<U32x4>(chunk + 256 + 16 * (kWaveSize * j + lane));
            raw.words[j][0]  = value.x;
            raw.words[j][1]  = value.y;
            raw.words[j][2]  = value.z;
            raw.words[j][3]  = value.w;
        }
        return raw;
    }

    // The words row k decodes from (Slice::multiply).
    __device__ __forceinline__ static void launder_row(Raw& raw, int k) {
        if (k % 2 == 0) { launder(raw.scales[k / 2]); }
#pragma unroll
        for (int j = 0; j < 4; ++j) { launder(raw.words[j][k]); }
    }

    // Word j holds K offsets 8 j + n in nibble n; nibble bit 3 is the sign, moved to bit 15 of
    // the BF16 value (bit 7 of its high byte).
    __device__ __forceinline__ static void decode(const Raw& raw, int k, const Lut& /*book*/,
                                                  std::uint32_t (&w)[16]) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const std::uint32_t word = raw.words[j][k];
            std::uint32_t el, eh, ol, oh;
            lookup_run(word, kE2m1Magnitudes, el, eh, ol, oh);
            eh |= (word << 4) & 0x80808080U;
            oh |= word & 0x80808080U;
            pair_run(el, eh, ol, oh, w + 4 * j);
        }
    }

    // Byte 2 k + h of the lane's eight scale bytes: K offsets 32 c + 16 h .. + 15 (runs 2h, 2h+1).
    __device__ __forceinline__ static float scale(const Raw& raw, int k, int half) {
        return e4m3_to_float((raw.scales[k / 2] >> (16 * (k % 2) + 8 * half)) & 0xFFU);
    }

    __device__ __forceinline__ static float accumulate(const std::uint32_t (&w)[16],
                                                       const std::uint32_t (&x)[16],
                                                       const float (&scales)[2], float acc) {
        acc = fmaf(dot_pairs<0, 8>(w, x), scales[0], acc);
        return fmaf(dot_pairs<8, 8>(w, x), scales[1], acc);
    }

    // One 32-K unit: two WMMA K steps, each under its own 16-K scale per row.
    __device__ __forceinline__ static void wmma_unit(const WmmaBf16 (&a)[2], const WmmaBf16 (&b)[2],
                                                     const float (&scales)[2], WmmaF32& acc) {
        acc += wmma(a[0], b[0], WmmaF32{}) * scales[0];
        acc += wmma(a[1], b[1], WmmaF32{}) * scales[1];
    }
};

// --- Wave slice ----------------------------------------------------------------------------------

// One token's 640 staged activations as 80 16-byte units; lane L loads units L + 32 i.
constexpr int kStageUnitsPerLane = (kStageUnits + kWaveSize - 1) / kWaveSize;

struct StageRegs {
    uint4 value[kStageUnitsPerLane];
};

// Issues the loads of one token's 640 BF16 activations starting at `row`.
__device__ __forceinline__ void stage_load(StageRegs& regs, int lane, const std::uint16_t* row) {
#pragma unroll
    for (int i = 0; i < kStageUnitsPerLane; ++i) {
        const int unit = lane + kWaveSize * i;
        if (unit < kStageUnits) { regs.value[i] = *reinterpret_cast<const uint4*>(row + 8 * unit); }
    }
}

// Stores the loaded units into the wave's stage, each eight-K run in the decode's pair order:
// source dwords (x0,x1) (x2,x3) (x4,x5) (x6,x7) become (x0,x2) (x4,x6) (x1,x3) (x5,x7).
__device__ __forceinline__ void stage_store(const StageRegs& regs, int lane, std::uint8_t* stage) {
#pragma unroll
    for (int i = 0; i < kStageUnitsPerLane; ++i) {
        const int unit = lane + kWaveSize * i;
        if (unit < kStageUnits) {
            const uint4 s = regs.value[i];
            uint4 d;
            d.x = __builtin_amdgcn_perm(s.y, s.x, 0x05040100U);
            d.y = __builtin_amdgcn_perm(s.w, s.z, 0x05040100U);
            d.z = __builtin_amdgcn_perm(s.y, s.x, 0x07060302U);
            d.w = __builtin_amdgcn_perm(s.w, s.z, 0x07060302U);
            *reinterpret_cast<uint4*>(stage + 16 * unit) = d;
        }
    }
}

// Sum over the four lanes of a quad (the lanes c = 0..3 sharing g); identical in every lane.
__device__ __forceinline__ float quad_sum(float value) {
    // DPP quad_perm [1,0,3,2] (xor 1) and [2,3,0,1] (xor 2).
    value +=
        __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, value),
                                                              0xB1, 0xF, 0xF, false));
    value +=
        __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(0, __builtin_bit_cast(int, value),
                                                              0x4E, 0xF, 0xF, false));
    return value;
}

// One wave's slice: kSliceChunks chunks of one 32-row tile held in registers, multiplied by one
// token's staged activations.
template <class Codec>
struct Slice {
    typename Codec::Raw raw[kSliceChunks];

    __device__ __forceinline__ void load(RecordBytes first_chunk, int lane) {
#pragma unroll
        for (int q = 0; q < kSliceChunks; ++q) {
            raw[q] = Codec::load(first_chunk + q * Codec::kChunkBytes, lane);
        }
    }

    // Returns the full-slice dot product of local row g + 8 c with the staged token (quad-reduced
    // over the four K quarters of each chunk). books[0] decodes local rows 0..15 (k = 0, 1),
    // books[1] rows 16..31 (k = 2, 3).
    //
    // Row order is pinned through data dependencies on empty volatile asm (launder): row k of
    // chunk q launders its raw words when it starts and its accumulator when it ends, so its
    // decode and dot products stay between those points, and chunk q's activations are loaded
    // through an offset laundered at its start. Without them the compiler hoists every row's
    // decode and activation load (also out of a caller's token loop) and sinks every dot product
    // to the end, keeping all five chunks' activations and decoded rows live and spilling.
    // Activation LDS latency is left to the other waves of the SIMD.
    __device__ __forceinline__ float multiply(const std::uint8_t* stage, int lane,
                                              const Lut (&books)[2]) {
        const int c  = lane % 4;
        float acc[4] = {};
#pragma unroll
        for (int q = 0; q < kSliceChunks; ++q) {
            Codec::launder_row(raw[q], 0);
            // The offset, not the pointer, is laundered: the pointer stays provably LDS.
            int offset = 256 * q + 64 * c;
            launder(offset);
            const auto* row = reinterpret_cast<const uint4*>(stage + offset);
            std::uint32_t x[16];
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                const uint4 value = row[v];
                x[4 * v]          = value.x;
                x[4 * v + 1]      = value.y;
                x[4 * v + 2]      = value.z;
                x[4 * v + 3]      = value.w;
            }
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                if (k > 0) { Codec::launder_row(raw[q], k); }
                std::uint32_t w[16];
                Codec::decode(raw[q], k, books[k / 2], w);
                const float scales[2] = {Codec::scale(raw[q], k, 0), Codec::scale(raw[q], k, 1)};
                acc[k]                = Codec::accumulate(w, x, scales, acc[k]);
                launder(acc[k]);
            }
        }
        float rows[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) { rows[k] = quad_sum(acc[k]); }
        return c == 0 ? rows[0] : c == 1 ? rows[1] : c == 2 ? rows[2] : rows[3];
    }
};

// WMMA path of one wave's slice (the five chunks from first_chunk): multiplies up to 16 tokens,
// whose slice activations (640 BF16 each) start at row(t), read from global memory as A
// fragments; a lane past `tokens` reads the last token and its results are unused. Returns
// acc[h][e] = dot product of local row 16 h + lane % 16 with token 8 (lane / 16) + e.
//
// The record streams through a two-chunk register ring (chunk q + 1 is requested when chunk q
// starts), which keeps this path's registers within the SIMT path's: a job of more than 16
// tokens reads its record once per 16-token block. Per chunk and row half h (local rows
// 16 h .. +15, the lanes' rows k = 2h, 2h+1), every lane decodes its two rows once into
// registers in natural K order; then, per 32-K unit c, the eight lanes holding it
// (lane % 4 == c) write a 16-row tile and the rows' scales to the wave's scratch, and the wave
// runs that unit's two K steps (Codec::wmma_unit). Wave syncs bracket each tile; laundering pins
// the decode per half and the accumulators per unit, as in Slice::multiply.
template <class Codec, class Row>
__device__ __forceinline__ void multiply_wmma(RecordBytes first_chunk, const Row& row, int tokens,
                                              std::uint8_t* tile, int lane, const Lut (&books)[2],
                                              WmmaF32 (&acc)[2]) {
    const int column                 = lane % 16; // A row (token) and B column (row of the half)
    const int k_half                 = lane / 16; // which eight K of each 16-K step
    const int g                      = lane / 4;
    const int c_lane                 = lane % 4;
    const std::uint16_t* activations = row(min(column, tokens - 1));
    auto* scales                     = reinterpret_cast<float*>(tile + kTileScaleOffset);
    typename Codec::Raw ring[2];
    ring[0] = Codec::load(first_chunk, lane);
    acc[0]  = WmmaF32{};
    acc[1]  = WmmaF32{};
#pragma unroll
    for (int q = 0; q < kSliceChunks; ++q) {
        typename Codec::Raw& raw = ring[q % 2];
        Codec::launder_row(raw, 0);
        if (q + 1 < kSliceChunks) {
            ring[(q + 1) % 2] = Codec::load(first_chunk + (q + 1) * Codec::kChunkBytes, lane);
        }
        int offset = 128 * q + 8 * k_half;
        launder(offset);
        WmmaBf16 a[8];
#pragma unroll
        for (int step = 0; step < 8; ++step) {
            a[step] = *reinterpret_cast<const WmmaBf16*>(activations + offset + 16 * step);
        }
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            if (h > 0) { Codec::launder_row(raw, 2 * h); }
            Codec::launder_row(raw, 2 * h + 1);
            std::uint32_t w[2][16];
            float row_scales[2][2];
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                Codec::decode(raw, 2 * h + r, books[h], w[r]);
                natural_order(w[r]);
                row_scales[r][0] = Codec::scale(raw, 2 * h + r, 0);
                row_scales[r][1] = Codec::scale(raw, 2 * h + r, 1);
            }
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                if (c_lane == c) {
#pragma unroll
                    for (int r = 0; r < 2; ++r) {
                        const int tile_row = g + 8 * r;
                        auto* destination =
                            reinterpret_cast<uint4*>(tile + tile_row * kTileRowBytes);
#pragma unroll
                        for (int v = 0; v < 4; ++v) {
                            destination[v] = make_uint4(w[r][4 * v], w[r][4 * v + 1],
                                                        w[r][4 * v + 2], w[r][4 * v + 3]);
                        }
                        scales[2 * tile_row]     = row_scales[r][0];
                        scales[2 * tile_row + 1] = row_scales[r][1];
                    }
                }
                wave_lds_sync();
                const WmmaBf16 b[2] = {
                    *reinterpret_cast<const WmmaBf16*>(tile + column * kTileRowBytes + 16 * k_half),
                    *reinterpret_cast<const WmmaBf16*>(tile + column * kTileRowBytes + 32 +
                                                       16 * k_half)};
                const float unit_scales[2] = {scales[2 * column], scales[2 * column + 1]};
                const WmmaBf16 unit_a[2]   = {a[2 * c], a[2 * c + 1]};
                Codec::wmma_unit(unit_a, b, unit_scales, acc[h]);
                launder_vector(acc[h]);
                wave_lds_sync();
            }
        }
    }
}

// --- Items -------------------------------------------------------------------------------------

__device__ __forceinline__ std::uint16_t bf16_round(float value) {
    const std::uint32_t bits = __float_as_uint(value);
    return static_cast<std::uint16_t>((bits + 0x7FFFU + ((bits >> 16) & 1U)) >> 16);
}

__device__ __forceinline__ float silu(float z) { return z / (1.0F + expf(-z)); }

// The job's record address and expert, uniform across the workgroup.
struct JobView {
    RecordBytes record;
    bool f3;
    int expert;
    int begin;
    int count;
};

__device__ __forceinline__ JobView view_job(const Job* jobs, int job) {
    const Job value = jobs[job];
    const auto lo   = static_cast<std::uint32_t>(
        __builtin_amdgcn_readfirstlane(static_cast<int>(value.entry & 0xFFFFFFFFU)));
    const auto hi = static_cast<std::uint32_t>(
        __builtin_amdgcn_readfirstlane(static_cast<int>(value.entry >> 32)));
    const std::uint64_t entry = (static_cast<std::uint64_t>(hi) << 32) | lo;
    JobView view;
    // The expert table holds record addresses as integers by contract.
    view.record = reinterpret_cast<RecordBytes>(
        entry & ~std::uint64_t{1}); // NOLINT(performance-no-int-to-ptr)
    view.f3     = (entry & 1U) != 0;
    view.expert = __builtin_amdgcn_readfirstlane(value.expert);
    view.begin  = __builtin_amdgcn_readfirstlane(value.begin);
    view.count  = __builtin_amdgcn_readfirstlane(value.count);
    return view;
}

// Writes gated rows 16 tile .. +15 of `tokens` tokens from the four waves' partial sums
// [token][32 local rows] in their scratch (local rows 0..15 gate, 16..31 up), published by an LDS
// barrier, and ends with another so the scratch may be reused.
__device__ __forceinline__ void write_gated(const Params& p, const std::uint8_t* lds,
                                            const JobView& job, int tile, int first, int tokens,
                                            float gate_scale, float up_scale) {
    lds_barrier();
    for (int task = static_cast<int>(threadIdx.x); task < 16 * tokens; task += kThreads) {
        const int t = task / 16;
        const int i = task % 16;
        float gate  = 0.0F;
        float up    = 0.0F;
#pragma unroll
        for (int w = 0; w < kWaves; ++w) {
            const auto* sums = reinterpret_cast<const float*>(lds + w * kWaveScratchBytes);
            gate += sums[t * 32 + i];
            up += sums[t * 32 + 16 + i];
        }
        const float product = silu(gate * gate_scale) * (up * up_scale);
        p.gated[static_cast<std::int64_t>(job.begin + first + t) * kMoeIntermediate + 16 * tile +
                i]          = bf16_round(product);
    }
    lds_barrier();
}

// gate|up item: four K-quarter slices of tile `tile`; writes the job's gated rows 16 tile .. +15.
// SIMT: one token at a time (two barriers each). WMMA: blocks of 16 tokens (two barriers each).
template <class Codec>
__device__ __forceinline__ void gate_up_item(const Params& p, std::uint8_t* lds, const JobView& job,
                                             int tile) {
    const int tid          = static_cast<int>(threadIdx.x);
    const int lane         = tid % kWaveSize;
    const int wave         = tid / kWaveSize;
    const auto* sorted     = reinterpret_cast<const std::uint16_t*>(lds + p.lds.sorted);
    std::uint8_t* scratch  = lds + wave * kWaveScratchBytes;
    auto* partial          = reinterpret_cast<float*>(scratch);
    const Lut books[2]     = {p.books[0], p.books[1]};
    const float gate_scale = p.matrix_scales[3 * job.expert];
    const float up_scale   = p.matrix_scales[3 * job.expert + 1];
    const auto x_row       = [&](int token) {
        return p.x + static_cast<std::int64_t>(sorted[job.begin + token] / kRoutes) * kMoeHidden +
               wave * kSliceK;
    };
    const RecordBytes first_chunk =
        job.record + static_cast<std::int64_t>(20 * tile + 5 * wave) * Codec::kChunkBytes;
    if (job.count >= kWmmaMinTokens) {
        for (int first = 0; first < job.count; first += kWmmaTokens) {
            const int tokens = min(kWmmaTokens, job.count - first);
            WmmaF32 acc[2];
            multiply_wmma<Codec>(
                first_chunk, [&](int t) { return x_row(first + t); }, tokens, scratch, lane, books,
                acc);
            // The partial sums replace the tile: [token][local row].
            const int token0 = 8 * (lane / 16);
#pragma unroll
            for (int h = 0; h < 2; ++h) {
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    if (token0 + e < tokens) {
                        partial[(token0 + e) * 32 + 16 * h + lane % 16] = acc[h][e];
                    }
                }
            }
            write_gated(p, lds, job, tile, first, tokens, gate_scale, up_scale);
        }
        return;
    }

    // The first token's activations are requested before the record, so waiting for them does
    // not wait for the record loads (loads return in order).
    StageRegs regs;
    stage_load(regs, lane, x_row(0));
    __builtin_amdgcn_sched_barrier(0);
    Slice<Codec> slice;
    slice.load(first_chunk, lane);
    __builtin_amdgcn_sched_barrier(0);
    for (int token = 0; token < job.count; ++token) {
        if (token > 0) { stage_load(regs, lane, x_row(token)); }
        stage_store(regs, lane, scratch);
        wave_lds_sync();
        const float sum = slice.multiply(scratch, lane, books);
        // The wave's partial sums replace its staged activations, indexed by local row g + 8 c.
        wave_lds_sync();
        partial[lane / 4 + 8 * (lane % 4)] = sum;
        write_gated(p, lds, job, tile, token, 1, gate_scale, up_scale);
    }
}

// down item: four row tiles of group `group`, each a full-K slice; writes every route's
// contribution rows 128 group .. +127 (SIMT per token or WMMA per 16-token block), then merges
// each token whose contributions for these rows are now all present.
template <class Codec>
__device__ __forceinline__ void down_item(const Params& p, std::uint8_t* lds, const JobView& job,
                                          int group, int& merge_count) {
    const int tid          = static_cast<int>(threadIdx.x);
    const int lane         = tid % kWaveSize;
    const int wave         = tid / kWaveSize;
    const auto* sorted     = reinterpret_cast<const std::uint16_t*>(lds + p.lds.sorted);
    const auto* mask       = reinterpret_cast<const std::uint16_t*>(lds + p.lds.mask);
    auto* merges           = reinterpret_cast<std::uint16_t*>(lds + p.lds.merges);
    std::uint8_t* scratch  = lds + wave * kWaveScratchBytes;
    const Lut books[2]     = {p.books[2], p.books[2]};
    const float down_scale = p.matrix_scales[3 * job.expert + 2];
    const int tile         = kWaves * group + wave;
    const auto gated_row   = [&](int token) {
        return p.gated + static_cast<std::int64_t>(job.begin + token) * kMoeIntermediate;
    };
    // Route entry e = 10 t + j of the job's token and its contribution row.
    const auto contribute = [&](int token, int row, float sum) {
        const int entry    = sorted[job.begin + token];
        const float weight = p.routes[entry].weight;
        p.contributions[static_cast<std::int64_t>(entry) * kMoeHidden + row] =
            sum * down_scale * weight;
    };
    const RecordBytes first_chunk =
        job.record + static_cast<std::int64_t>(kDownPlaneChunk + 5 * tile) * Codec::kChunkBytes;
    if (job.count >= kWmmaMinTokens) {
        const int row = 32 * tile + lane % 16;
        for (int first = 0; first < job.count; first += kWmmaTokens) {
            const int tokens = min(kWmmaTokens, job.count - first);
            WmmaF32 acc[2];
            multiply_wmma<Codec>(
                first_chunk, [&](int t) { return gated_row(first + t); }, tokens, scratch, lane,
                books, acc);
            const int token0 = 8 * (lane / 16);
#pragma unroll
            for (int h = 0; h < 2; ++h) {
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    if (token0 + e < tokens) {
                        contribute(first + token0 + e, row + 16 * h, acc[h][e]);
                    }
                }
            }
        }
    } else {
        StageRegs regs;
        stage_load(regs, lane, gated_row(0));
        __builtin_amdgcn_sched_barrier(0);
        Slice<Codec> slice;
        slice.load(first_chunk, lane);
        __builtin_amdgcn_sched_barrier(0);
        const int row = 32 * tile + lane / 4 + 8 * (lane % 4);
        for (int token = 0; token < job.count; ++token) {
            if (token > 0) { stage_load(regs, lane, gated_row(token)); }
            stage_store(regs, lane, scratch);
            wave_lds_sync();
            contribute(token, row, slice.multiply(scratch, lane, books));
        }
    }

    // Publish the contributions, then count them per (token, group). The thread that completes a
    // token's count lists it; the counter is reset by its merge below, its last use in the call.
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    lds_barrier();
    for (int i = tid; i < job.count; i += kThreads) {
        const int token = sorted[job.begin + i] / kRoutes;
        const std::uint32_t done =
            atomic_add_agent(p.merge + token * kDownGroups + group, __ATOMIC_ACQ_REL) + 1U;
        if (done == static_cast<std::uint32_t>(__builtin_popcount(mask[token]))) {
            merges[atomicAdd(&merge_count, 1)] = static_cast<std::uint16_t>(token);
        }
    }
    lds_barrier();
    const int count = merge_count;
    if (count == 0) { return; }
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    for (int m = 0; m < count; ++m) {
        const int token  = merges[m];
        const int bits   = mask[token];
        const int column = 128 * group + tid;
        float sum        = 0.0F;
        for (int j = 0; j < kRoutes; ++j) {
            if ((bits >> j) & 1) {
                sum += p.contributions[static_cast<std::int64_t>(token * kRoutes + j) * kMoeHidden +
                                       column];
            }
        }
        p.y[static_cast<std::int64_t>(token) * kMoeHidden + column] = sum;
        if (tid == 0) {
            __hip_atomic_store(p.merge + token * kDownGroups + group, 0U, __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_AGENT);
        }
    }
}

// --- Kernel ------------------------------------------------------------------------------------

// Blocks until item `item`, if it is a down item, may read its job's gated rows (thread 0 only).
__device__ __forceinline__ void wait_inputs(const Params& p, int item, int jobs) {
    const int down = item - kGateUpTiles * jobs;
    if (down < 0 || item >= kItemsPerJob * jobs) { return; }
    const std::uint32_t* done = p.job_done + down / kDownGroups;
    while (__hip_atomic_load(done, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) <
           static_cast<std::uint32_t>(kGateUpTiles)) {
        __builtin_amdgcn_s_sleep(1);
    }
}

// Grid: at most the workgroups resident at once (moe_helper_experts), block kThreads.
//
// Prologue (every workgroup, identically): read N, sort the active route entries by
// (expert, entry index) into `sorted` by rank counting, split them into jobs, load each job's
// expert-table entry, and zero the output rows of tokens without an active entry (token t by
// workgroup t mod grid).
//
// Main loop: claim items from control[0]; the first claim is issued at entry and each next one
// when an item starts, consumed when it ends, and thread 0 waits there for a down item's inputs
// before publishing it through s_item (double-buffered: a slot is rewritten only after the next
// barrier).
//
// Exit: the last workgroup to count itself on control[1] resets the counters it and the others
// advanced (control and job_done; the merge counters were reset by their merges), so the next
// call starts from zero.
__global__ void __launch_bounds__(kThreads) moe_helper_experts_kernel(const Params p) {
    extern __shared__ __attribute__((aligned(16))) std::uint8_t lds[];
    __shared__ int s_item[2];
    __shared__ int s_valid;
    __shared__ int s_scan[kWaves];
    __shared__ int s_merges;
    __shared__ int s_last;

    const int tid  = static_cast<int>(threadIdx.x);
    const int lane = tid % kWaveSize;
    const int wave = tid / kWaveSize;
    // The first claim needs no prologue result; issued here, its latency overlaps the prologue.
    int first_claim = 0;
    if (tid == 0) { first_claim = static_cast<int>(atomic_add_agent(p.control, __ATOMIC_RELAXED)); }
    const int tokens = __builtin_amdgcn_readfirstlane(min(max(*p.tokens, 0), p.max_tokens));
    const int routes = kRoutes * tokens;
    auto* keys       = reinterpret_cast<std::uint32_t*>(lds);
    auto* jobs       = reinterpret_cast<Job*>(lds + p.lds.jobs);
    auto* sorted     = reinterpret_cast<std::uint16_t*>(lds + p.lds.sorted);
    auto* mask       = reinterpret_cast<std::uint16_t*>(lds + p.lds.mask);

    // Sort keys; an entry outside [0,512) is inactive.
    for (int e = tid; e < routes; e += kThreads) {
        const int expert = p.routes[e].expert;
        keys[e] = expert >= 0 && expert < kMoeHelperExperts
                      ? (static_cast<std::uint32_t>(expert) << 9) | static_cast<std::uint32_t>(e)
                      : kInvalidKey;
    }
    if (tid == 0) { s_valid = 0; }
    lds_barrier();
    for (int t = tid; t < tokens; t += kThreads) {
        int bits = 0;
        for (int j = 0; j < kRoutes; ++j) {
            if (keys[t * kRoutes + j] != kInvalidKey) { bits |= 1 << j; }
        }
        mask[t] = static_cast<std::uint16_t>(bits);
    }
    // Rank of each active key among the distinct keys is its sorted position.
    int active = 0;
    for (int e = tid; e < routes; e += kThreads) {
        const std::uint32_t key = keys[e];
        if (key == kInvalidKey) { continue; }
        int rank = 0;
        for (int other = 0; other < routes; ++other) { rank += keys[other] < key ? 1 : 0; }
        sorted[rank] = static_cast<std::uint16_t>(e);
        ++active;
    }
    atomicAdd(&s_valid, active);
    lds_barrier();

    // Job heads: positions whose expert differs from the previous position's. Thread i owns
    // positions 4 i .. 4 i + 3 (10 * 48 <= 4 * kThreads); a block scan numbers the heads.
    const int valid = s_valid;
    static_assert(kRoutes * kMoeHelperMaxTokens <= 4 * kThreads);
    int heads = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int position = 4 * tid + i;
        if (position < valid &&
            (position == 0 || (keys[sorted[position]] >> 9) != (keys[sorted[position - 1]] >> 9))) {
            heads |= 1 << i;
        }
    }
    const int own = __builtin_popcount(heads);
    int inclusive = own;
#pragma unroll
    for (int offset = 1; offset < kWaveSize; offset *= 2) {
        const int lower = __shfl_up(inclusive, offset, kWaveSize);
        if (lane >= offset) { inclusive += lower; }
    }
    if (lane == kWaveSize - 1) { s_scan[wave] = inclusive; }
    lds_barrier();
    int job_count = 0;
    int job       = inclusive - own;
#pragma unroll
    for (int w = 0; w < kWaves; ++w) {
        job_count += s_scan[w];
        if (w < wave) { job += s_scan[w]; }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        if ((heads >> i) & 1) {
            const int position = 4 * tid + i;
            jobs[job].expert   = static_cast<std::int32_t>(keys[sorted[position]] >> 9);
            jobs[job].begin    = static_cast<std::uint16_t>(position);
            ++job;
        }
    }
    lds_barrier();
    for (int j = tid; j < job_count; j += kThreads) {
        const int end = j + 1 < job_count ? jobs[j + 1].begin : valid;
        jobs[j].count = static_cast<std::uint16_t>(end - jobs[j].begin);
        jobs[j].entry = p.expert_table[jobs[j].expert];
    }
    for (int t = static_cast<int>(blockIdx.x); t < tokens; t += static_cast<int>(gridDim.x)) {
        if (mask[t] != 0) { continue; }
        auto* row = reinterpret_cast<float4*>(p.y + static_cast<std::int64_t>(t) * kMoeHidden);
        for (int i = tid; i < kMoeHidden / 4; i += kThreads) {
            row[i] = make_float4(0.F, 0.F, 0.F, 0.F);
        }
    }

    const int items = kItemsPerJob * job_count;
    if (tid == 0) {
        wait_inputs(p, first_claim, job_count);
        s_item[0] = first_claim;
    }
    lds_barrier(); // also publishes the job table before any item reads it
    int item = s_item[0];
    for (int iteration = 0; item < items; ++iteration) {
        if (item >= kGateUpTiles * job_count) { __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent"); }
        int next = 0;
        if (tid == 0) {
            next     = static_cast<int>(atomic_add_agent(p.control, __ATOMIC_RELAXED));
            s_merges = 0;
        }
        if (item < kGateUpTiles * job_count) {
            const JobView view = view_job(jobs, item / kGateUpTiles);
            if (view.f3) {
                gate_up_item<F3Codec>(p, lds, view, item % kGateUpTiles);
            } else {
                gate_up_item<Nvfp4Codec>(p, lds, view, item % kGateUpTiles);
            }
            __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
            lds_barrier();
            if (tid == 0) { atomic_add_agent(p.job_done + item / kGateUpTiles, __ATOMIC_RELEASE); }
        } else {
            const int down     = item - kGateUpTiles * job_count;
            const JobView view = view_job(jobs, down / kDownGroups);
            if (view.f3) {
                down_item<F3Codec>(p, lds, view, down % kDownGroups, s_merges);
            } else {
                down_item<Nvfp4Codec>(p, lds, view, down % kDownGroups, s_merges);
            }
        }
        const int slot = (iteration + 1) & 1;
        if (tid == 0) {
            wait_inputs(p, next, job_count);
            s_item[slot] = next;
        }
        lds_barrier();
        item = s_item[slot];
    }

    if (tid == 0) {
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
        s_last = atomic_add_agent(p.control + 1, __ATOMIC_ACQ_REL) + 1U == gridDim.x ? 1 : 0;
    }
    lds_barrier();
    if (s_last != 0) {
        for (int j = tid; j < job_count; j += kThreads) { p.job_done[j] = 0; }
        if (tid == 0) {
            p.control[0] = 0;
            p.control[1] = 0;
        }
    }
}

// --- Host --------------------------------------------------------------------------------------

void hip_check(hipError_t status, const char* what) {
    if (status != hipSuccess) {
        throw std::runtime_error(std::string("moe_helper_experts: ") + what + ": " +
                                 hipGetErrorString(status));
    }
}

[[noreturn]] void reject(const std::string& message) {
    throw std::invalid_argument("moe_helper_experts: " + message);
}

bool aligned(const void* pointer, std::uintptr_t alignment) {
    return reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0;
}

struct Span {
    const void* begin;
    std::size_t bytes;
};

bool overlaps(const Span& a, const Span& b) {
    const auto a0 = reinterpret_cast<std::uintptr_t>(a.begin);
    const auto b0 = reinterpret_cast<std::uintptr_t>(b.begin);
    return a0 < b0 + b.bytes && b0 < a0 + a.bytes;
}

// Workgroups resident at once for a call extent, queried once per extent. The product drives one
// helper device, the current one at the first call.
int resident_workgroups(std::int32_t max_tokens) {
    static std::array<std::atomic<int>, kMoeHelperMaxTokens + 1> cache{};
    std::atomic<int>& entry = cache[static_cast<std::size_t>(max_tokens)];
    if (const int known = entry.load(std::memory_order_relaxed); known > 0) { return known; }
    int per_unit = 0;
    hip_check(hipOccupancyMaxActiveBlocksPerMultiprocessor(
                  &per_unit, reinterpret_cast<const void*>(&moe_helper_experts_kernel), kThreads,
                  lds_layout(max_tokens).bytes),
              "occupancy query");
    if (per_unit < 1) { throw std::logic_error("moe_helper_experts: a workgroup does not fit"); }
    int device = 0;
    int units  = 0;
    hip_check(hipGetDevice(&device), "hipGetDevice");
    hip_check(hipDeviceGetAttribute(&units, hipDeviceAttributeMultiprocessorCount, device),
              "multiprocessor count");
    const int workgroups = per_unit * units;
    entry.store(workgroups, std::memory_order_relaxed);
    return workgroups;
}

} // namespace

MoeHelperExpertsWorkspace::MoeHelperExpertsWorkspace(std::int32_t max_tokens)
    : max_tokens_(max_tokens) {
    if (max_tokens < 1 || max_tokens > kMoeHelperMaxTokens) {
        reject("workspace max_tokens " + std::to_string(max_tokens) + " outside [1, 48]");
    }
    hip_check(hipGetDevice(&device_), "hipGetDevice");
    const std::size_t bytes = workspace_layout(max_tokens).bytes;
    void* base              = nullptr;
    hip_check(hipMalloc(&base, bytes), "workspace hipMalloc");
    base_ = static_cast<std::byte*>(base);
    try {
        hip_check(hipMemset(base_, 0, bytes), "workspace hipMemset");
        hip_check(hipDeviceSynchronize(), "workspace zero fill");
    } catch (...) {
        (void)hipFree(base_);
        throw;
    }
}

MoeHelperExpertsWorkspace::~MoeHelperExpertsWorkspace() { (void)hipFree(base_); }

void moe_helper_experts(const MoeHelperExpertsArgs& args, MoeHelperExpertsWorkspace& workspace,
                        hipStream_t stream) {
    const std::int32_t max_tokens = args.max_tokens;
    if (max_tokens < 1 || max_tokens > kMoeHelperMaxTokens) {
        reject("max_tokens " + std::to_string(max_tokens) + " outside [1, 48]");
    }
    if (max_tokens > workspace.max_tokens()) {
        reject("max_tokens " + std::to_string(max_tokens) + " exceeds the workspace's " +
               std::to_string(workspace.max_tokens()));
    }
    if (args.tokens == nullptr || args.x == nullptr || args.routes == nullptr ||
        args.bank.expert_table == nullptr || args.bank.matrix_scales == nullptr ||
        args.y == nullptr) {
        reject("null operand");
    }
    if (!aligned(args.tokens, 4) || !aligned(args.x, 16) || !aligned(args.routes, 8) ||
        !aligned(args.bank.expert_table, 8) || !aligned(args.bank.matrix_scales, 4) ||
        !aligned(args.y, 16)) {
        reject("misaligned operand");
    }
    const auto rows = static_cast<std::size_t>(max_tokens);
    const Span y{args.y, rows * kMoeHidden * sizeof(float)};
    const Span inputs[] = {
        {args.tokens, sizeof(std::int32_t)},
        {args.x, rows * kMoeHidden * sizeof(std::uint16_t)},
        {args.routes, rows * kRoutes * sizeof(MoeHelperRoute)},
        {args.bank.expert_table, kMoeHelperExperts * sizeof(std::uint64_t)},
        {args.bank.matrix_scales, kMoeHelperExperts * 3 * sizeof(float)},
    };
    for (const Span& input : inputs) {
        if (overlaps(y, input)) { reject("y overlaps an input"); }
    }
    int device = 0;
    hip_check(hipGetDevice(&device), "hipGetDevice");
    if (device != workspace.device()) { reject("workspace belongs to another device"); }

    const WorkspaceLayout layout = workspace_layout(workspace.max_tokens());
    std::byte* base              = workspace.base();
    Params params{};
    params.tokens        = args.tokens;
    params.x             = args.x;
    params.routes        = args.routes;
    params.expert_table  = args.bank.expert_table;
    params.matrix_scales = args.bank.matrix_scales;
    params.y             = args.y;
    params.control       = reinterpret_cast<std::uint32_t*>(base);
    params.job_done      = reinterpret_cast<std::uint32_t*>(base + layout.job_done);
    params.merge         = reinterpret_cast<std::uint32_t*>(base + layout.merge);
    params.gated         = reinterpret_cast<std::uint16_t*>(base + layout.gated);
    params.contributions = reinterpret_cast<float*>(base + layout.contributions);
    for (int m = 0; m < 3; ++m) { params.books[m] = make_lut(args.bank.f3_books[m]); }
    params.max_tokens = max_tokens;
    params.lds        = lds_layout(max_tokens);

    // Items are bounded by 60 per active route entry; extra workgroups find none and exit.
    const int grid = std::min(resident_workgroups(max_tokens), kItemsPerJob * kRoutes * max_tokens);
    const auto lds_bytes = static_cast<unsigned>(params.lds.bytes);
    moe_helper_experts_kernel<<<grid, kThreads, lds_bytes, stream>>>(params);
    hip_check(hipGetLastError(), "kernel launch");
}

} // namespace ninfer::ops
