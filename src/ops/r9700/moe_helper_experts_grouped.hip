#include "ops/r9700/moe_helper_experts_grouped.h"

#include "ops/r9700/moe_helper_common.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

// --- Design --------------------------------------------------------------------------------------
//
// Five kernels on one stream, all with fixed grids, reading N and the routes on the device:
//
//   count    per 4096-entry block of the N x 10 route entries, the active entries per expert.
//   scatter  a stable counting sort of the active entries by expert (ascending entry index
//            e = 10 t + j within an expert): for sorted position p, the token, the route weight
//            and, for entry e, its position (-1 when inactive). Block 0 also splits every expert's
//            entries into jobs of at most 128 consecutive positions, full jobs first, then the
//            partial ones in descending size, and resets the GEMM kernels' item counters.
//   gate|up  item (job, group g in [0,5)): eight waves, wave w owning record tile 8 g + w (gate
//            and up rows 16 (8 g + w) .. +15), compute the job's gate/up products over K = 2560
//            and write gated[p][i] = BF16(SiLU(s_g G x) * (s_u U x)) for their 16 rows i.
//   down     item (job, group g in [0,10)): wave w owns down tile 8 g + w (rows 32 (8 g + w)
//            .. +31) over K = 640 and writes each route's FP32 contribution
//            s_d w_j (D gated)[row] to contributions[p][row].
//   combine  per token t < N: y[t,:] = sum of its active entries' contributions in route-slot
//            order j = 0..9, starting from +0.
//
// The GEMM kernels are persistent: each workgroup claims items from a counter in order, so the
// largest jobs start first and the partial ones fill the tail. Every contribution is computed by
// exactly one item and summed by the combine in slot order: no float atomics, and a token's
// output does not depend on its sorted position or on which workgroup computed it. The combine is
// a separate bandwidth-bound pass: merging each token in the down item that completes it leaves
// that item waiting on scattered reads (measured 3x slower at N = 16384).
//
// GEMM item. M = the job's tokens (up to 128, eight 16-token m-blocks, only the occupied ones
// computed), N = a wave's 32 record rows as two 16-row n-blocks, K in stages of 64 (half a
// 128-K record chunk). Per stage:
//   activations  the workgroup stages the job's tokens' 64 K into LDS (row stride 144 bytes, so
//                the 16 rows one A-fragment read touches fall in distinct banks), each eight-K
//                run in pair order (K offsets 0, 2, 4, 6, 1, 3, 5, 7);
//   weights      each lane loads the record words of its B-fragment positions straight from
//                global memory and decodes them in registers through a scaled table: the stage
//                is double-buffered in registers and in LDS, so the next stage's loads are in
//                flight while this one computes;
//   WMMA         per 16-K step, every occupied m-block multiplies its LDS A-fragment with the two
//                decoded B-fragments (BF16 x BF16, FP32 accumulation).
//
// K order. A WMMA sums its 16 K in any consistent order of A and B, so a step needs no natural
// order: in stage t (chunk q = t / 2, half hc = t % 2), step s, lane half h takes the eight-K run
// (unit c = 2 hc + h, run j = s) of the record's lane layout (ninfer/ops/moe_experts.h), in pair
// order on both sides.
//
// B-fragment lane mapping. Lane l = 16 h + n holds column n: local rows rho = n (n-block 0) and
// 16 + n (n-block 1), both in record lane L = 4 (n % 8) + c at units k = n / 8 and 2 + n / 8.
// In the gate|up plane they are gate and up row 16 r + n, so SiLU and the product happen in the
// lane's registers; in the down plane they are down rows 32 r + n and 32 r + 16 + n.
//
// Scaled decode. A decoded weight is code value times its E4M3 block scale: at most two and four
// significant bits, so the product is exact in BF16 and the scale needs no FP32 pass after the
// WMMA. Per workgroup, LDS holds for each codebook (gate, up or down F3 book, and the E2M1 table
// of NVFP4) and each of the 256 scale bytes the eight products as BF16 low and high byte planes;
// one 16-byte read per (row, scale group) gives the byte-permute tables of the run decode.

constexpr int kWaveSize = 32;
constexpr int kRoutes   = kMoeHelperRoutesPerToken;

// Grouping kernels: 256 threads, 16 entries each.
constexpr int kGroupThreads     = 256;
constexpr int kGroupWaves       = kGroupThreads / kWaveSize;
constexpr int kEntriesPerThread = 16;
constexpr int kGroupEntries     = kGroupThreads * kEntriesPerThread; // 4096
constexpr int kWaveEntries      = kGroupEntries / kGroupWaves;       // 512

// GEMM kernels.
constexpr int kJobTokens          = 128;
constexpr int kMBlocks            = kJobTokens / 16;
constexpr int kGemmWaves          = 8;
constexpr int kGemmThreads        = kGemmWaves * kWaveSize;
constexpr int kStageK             = 64;
constexpr int kStageRowBytes      = 144; // 64 BF16 and 16 bytes of padding
constexpr int kStageBytes         = kJobTokens * kStageRowBytes;
constexpr int kStageUnits         = kStageK / 8; // 16-byte units of one staged row
constexpr int kUnitsPerThread     = kJobTokens * kStageUnits / kGemmThreads;
constexpr int kScaleCodes         = 256;
constexpr int kTableBytes         = kScaleCodes * 16;
constexpr std::uint32_t kE2m1Book = 0x76543210U; // entry v = E2M1 code v (magnitudes)

static_assert(kUnitsPerThread * kGemmThreads == kJobTokens * kStageUnits);
static_assert(kStageRowBytes % 16 == 0 && kStageRowBytes / 4 % 32 == 4);

// Combine kernel.
constexpr int kCombineThreads = 128;
constexpr int kRowVectors     = kMoeHidden / 4;
static_assert(kRowVectors % kCombineThreads == 0);

// Control words of the workspace region.
constexpr int kClaimGateUp = 0;
constexpr int kClaimDown   = 1;
constexpr int kJobCount    = 2;

// One job: an expert and up to 128 consecutive sorted positions of its entries.
struct alignas(16) Job {
    std::uint64_t entry; // expert-table entry: record address | codec
    std::int32_t first;  // sorted position of the job's first entry
    std::uint32_t shape; // token count (bits 0..15) | expert << 16
};

// 32-bit indices are exact: every product below is at most 10 * 16384 * 2560 < 2^31.
static_assert(static_cast<std::int64_t>(kRoutes) * kMoeHelperExpertsMaxTokens * kMoeHidden <
              (std::int64_t{1} << 31));

[[nodiscard]] constexpr std::int64_t ceil_div(std::int64_t a, std::int64_t b) {
    return (a + b - 1) / b;
}

constexpr std::size_t align256(std::size_t bytes) { return (bytes + 255) / 256 * 256; }

// Workspace region for capacity Tm, R = 10 Tm route entries (offsets 256-byte aligned):
//   control        U32 [64]: item claims of the gate|up and down kernels, job count
//   counts         U32 [ceil(R / 4096)][512]: active entries per (entry block, expert)
//   jobs           Job [ceil(R / 128) + 512]
//   token_of       U32 [R]: token of sorted position p
//   weight_of      FP32 [R]: route weight of sorted position p
//   position       I32 [R]: sorted position of entry e, -1 when inactive
//   gated          BF16 [R][640] by sorted position
//   contributions  FP32 [R][2560] by sorted position
struct Layout {
    std::size_t counts;
    std::size_t jobs;
    std::size_t token_of;
    std::size_t weight_of;
    std::size_t position;
    std::size_t gated;
    std::size_t contributions;
    std::size_t bytes;
};

constexpr std::int64_t max_jobs(std::int64_t routes) {
    // sum over experts of ceil(n_e / 128) <= R / 128 + 512 (at most 512 experts are occupied).
    return ceil_div(routes, kJobTokens) + kMoeHelperExperts;
}

constexpr Layout layout_of(std::int32_t capacity) {
    const auto routes = static_cast<std::size_t>(kRoutes) * static_cast<std::size_t>(capacity);
    const auto blocks =
        static_cast<std::size_t>(ceil_div(static_cast<std::int64_t>(routes), kGroupEntries));
    Layout layout{};
    layout.counts = 256;
    layout.jobs   = layout.counts + align256(blocks * kMoeHelperExperts * 4);
    layout.token_of =
        layout.jobs +
        align256(static_cast<std::size_t>(max_jobs(static_cast<std::int64_t>(routes))) *
                 sizeof(Job));
    layout.weight_of     = layout.token_of + align256(routes * 4);
    layout.position      = layout.weight_of + align256(routes * 4);
    layout.gated         = layout.position + align256(routes * 4);
    layout.contributions = layout.gated + align256(routes * kMoeIntermediate * 2);
    layout.bytes         = layout.contributions + align256(routes * kMoeHidden * 4);
    return layout;
}

struct GroupParams {
    const std::int32_t* tokens;
    const MoeHelperRoute* routes;
    const std::uint64_t* expert_table;
    std::uint32_t* control;
    std::uint32_t* counts;
    Job* jobs;
    std::uint32_t* token_of;
    float* weight_of;
    std::int32_t* position;
    std::int32_t max_tokens;
};

struct GemmParams {
    const std::uint32_t* control;
    std::uint32_t* claims;
    const Job* jobs;
    const std::uint16_t* activations; // x (gate|up) or gated (down)
    const std::uint32_t* token_of;
    const float* weight_of;
    const float* matrix_scales;
    std::uint16_t* gated;
    float* contributions;
    std::uint32_t books[3]; // per scaled table: the F3 book, or kE2m1Book for NVFP4
};

struct CombineParams {
    const std::int32_t* tokens;
    const std::int32_t* position;
    const float* contributions;
    float* y;
    std::int32_t max_tokens;
};

__device__ __forceinline__ int token_count(const std::int32_t* tokens, std::int32_t max_tokens) {
    return __builtin_amdgcn_readfirstlane(min(max(*tokens, 0), max_tokens));
}

__device__ __forceinline__ bool active_expert(std::int32_t expert) {
    return expert >= 0 && expert < kMoeHelperExperts;
}

// --- Grouping ------------------------------------------------------------------------------------

// Grid: one workgroup per 4096-entry block of max_tokens x 10 (blocks past the call's entries
// exit), kGroupThreads threads. Writes counts[block][expert]; block 0 resets the item claims.
__global__ void __launch_bounds__(kGroupThreads) count_routes_kernel(const GroupParams p) {
    __shared__ std::uint32_t histogram[kMoeHelperExperts];
    const int tid = static_cast<int>(threadIdx.x);
    if (blockIdx.x == 0 && tid < 2) { p.control[tid] = 0; }
    const int entries = kRoutes * token_count(p.tokens, p.max_tokens);
    const int begin   = static_cast<int>(blockIdx.x) * kGroupEntries;
    if (begin >= entries) { return; }
    for (int e = tid; e < kMoeHelperExperts; e += kGroupThreads) { histogram[e] = 0; }
    std::int32_t experts[kEntriesPerThread];
#pragma unroll
    for (int i = 0; i < kEntriesPerThread; ++i) {
        const int e = begin + tid + kGroupThreads * i;
        experts[i]  = e < entries ? p.routes[e].expert : kMoeHelperInactiveExpert;
    }
    lds_barrier();
#pragma unroll
    for (int i = 0; i < kEntriesPerThread; ++i) {
        if (active_expert(experts[i])) { atomicAdd(&histogram[experts[i]], 1U); }
    }
    lds_barrier();
    for (int e = tid; e < kMoeHelperExperts; e += kGroupThreads) {
        p.counts[static_cast<int>(blockIdx.x) * kMoeHelperExperts + e] = histogram[e];
    }
}

// Exclusive prefix over the workgroup of one value per thread (in thread order); `total`
// receives the sum. Uses `partials` (kGroupWaves words) and ends with a barrier.
__device__ __forceinline__ std::uint32_t
block_exclusive_scan(std::uint32_t value, std::uint32_t* partials, std::uint32_t& total) {
    const int lane    = static_cast<int>(threadIdx.x) % kWaveSize;
    const int wave    = static_cast<int>(threadIdx.x) / kWaveSize;
    std::uint32_t inc = value;
#pragma unroll
    for (int offset = 1; offset < kWaveSize; offset *= 2) {
        const std::uint32_t lower = __shfl_up(inc, offset, kWaveSize);
        if (lane >= offset) { inc += lower; }
    }
    if (lane == kWaveSize - 1) { partials[wave] = inc; }
    lds_barrier();
    std::uint32_t before = inc - value;
    total                = 0;
#pragma unroll
    for (int w = 0; w < kGroupWaves; ++w) {
        const std::uint32_t part = partials[w];
        if (w < wave) { before += part; }
        total += part;
    }
    lds_barrier();
    return before;
}

// Grid and block as count_routes_kernel. Every block recomputes, for each expert, its entries in
// earlier blocks and in total and the expert offsets, then ranks its own entries stably: wave w
// owns entries 512 w .. +511 of the block in 16 rounds of 32, ranked among equal experts with
// ballots and placed after the wave's earlier entries of that expert. Block 0 writes the jobs.
__global__ void __launch_bounds__(kGroupThreads) scatter_routes_kernel(const GroupParams p) {
    __shared__ std::uint32_t bases[kGroupWaves][kMoeHelperExperts];
    __shared__ std::uint32_t partials[kGroupWaves];
    __shared__ std::uint32_t bins[kJobTokens]; // partial jobs per size, then the size's cursor
    const int tid     = static_cast<int>(threadIdx.x);
    const int lane    = tid % kWaveSize;
    const int wave    = tid / kWaveSize;
    const int block   = static_cast<int>(blockIdx.x);
    const int entries = kRoutes * token_count(p.tokens, p.max_tokens);
    const int blocks  = static_cast<int>(ceil_div(entries, kGroupEntries));
    if (block >= max(blocks, 1)) { return; }

    // Thread i owns experts 2 i and 2 i + 1.
    std::uint32_t before[2] = {0U, 0U};
    std::uint32_t total[2]  = {0U, 0U};
    for (int b = 0; b < blocks; ++b) {
        const auto pair =
            *reinterpret_cast<const U32x2*>(p.counts + b * kMoeHelperExperts + 2 * tid);
        if (b < block) {
            before[0] += pair.x;
            before[1] += pair.y;
        }
        total[0] += pair.x;
        total[1] += pair.y;
    }
    std::uint32_t entry_total = 0;
    const std::uint32_t pair_offset =
        block_exclusive_scan(total[0] + total[1], partials, entry_total);
    const std::uint32_t offset[2] = {pair_offset, pair_offset + total[0]};

    if (block == 0) {
        // Full jobs in expert order, then partial jobs by descending size (any order within a
        // size: it affects only scheduling).
        const std::uint32_t full[2] = {total[0] / kJobTokens, total[1] / kJobTokens};
        std::uint32_t full_count    = 0;
        const std::uint32_t full_first =
            block_exclusive_scan(full[0] + full[1], partials, full_count);
        for (int i = tid; i < kJobTokens; i += kGroupThreads) { bins[i] = 0; }
        lds_barrier();
        for (int x = 0; x < 2; ++x) {
            const std::uint32_t rest = total[x] % kJobTokens;
            if (rest != 0) { atomicAdd(&bins[rest], 1U); }
        }
        lds_barrier();
        if (tid == 0) {
            // First job of each size: full jobs, then sizes 127 down to 1.
            std::uint32_t cursor = full_count;
            for (int size = kJobTokens - 1; size >= 1; --size) {
                const std::uint32_t jobs = bins[size];
                bins[size]               = cursor;
                cursor += jobs;
            }
            p.control[kJobCount] = cursor;
        }
        lds_barrier();
        std::uint32_t job = full_first;
        for (int x = 0; x < 2; ++x) {
            const int expert = 2 * tid + x;
            if (total[x] == 0) { continue; }
            const std::uint64_t entry = p.expert_table[expert];
            const auto shape          = [&](std::uint32_t count) {
                return count | (static_cast<std::uint32_t>(expert) << 16);
            };
            for (std::uint32_t b = 0; b < full[x]; ++b) {
                p.jobs[job++] = Job{entry, static_cast<std::int32_t>(offset[x] + kJobTokens * b),
                                    shape(kJobTokens)};
            }
            const std::uint32_t rest = total[x] % kJobTokens;
            if (rest != 0) {
                const std::uint32_t slot = atomicAdd(&bins[rest], 1U);
                p.jobs[slot] =
                    Job{entry, static_cast<std::int32_t>(offset[x] + kJobTokens * full[x]),
                        shape(rest)};
            }
        }
    }
    if (entries == 0) { return; }

    // This block's entries: wave w, round i, lane l holds entry e = 4096 block + 512 w + 32 i + l.
    const int wave_begin = block * kGroupEntries + wave * kWaveEntries;
    MoeHelperRoute routes[kEntriesPerThread];
#pragma unroll
    for (int i = 0; i < kEntriesPerThread; ++i) {
        const int e = wave_begin + kWaveSize * i + lane;
        routes[i]   = e < entries ? p.routes[e] : MoeHelperRoute{kMoeHelperInactiveExpert, 0.0F};
    }
    for (int e = tid; e < kGroupWaves * kMoeHelperExperts; e += kGroupThreads) {
        bases[e / kMoeHelperExperts][e % kMoeHelperExperts] = 0;
    }
    lds_barrier();
#pragma unroll
    for (int i = 0; i < kEntriesPerThread; ++i) {
        if (active_expert(routes[i].expert)) { atomicAdd(&bases[wave][routes[i].expert], 1U); }
    }
    lds_barrier();
    for (int x = 0; x < 2; ++x) {
        std::uint32_t run = offset[x] + before[x];
        for (int w = 0; w < kGroupWaves; ++w) {
            const std::uint32_t count = bases[w][2 * tid + x];
            bases[w][2 * tid + x]     = run;
            run += count;
        }
    }
    lds_barrier();
    const std::uint32_t lower_lanes = (1U << lane) - 1U;
#pragma unroll
    for (int i = 0; i < kEntriesPerThread; ++i) {
        const int e         = wave_begin + kWaveSize * i + lane;
        const int expert    = routes[i].expert;
        const bool valid    = active_expert(expert);
        std::uint32_t peers = __builtin_amdgcn_ballot_w32(valid);
#pragma unroll
        for (int bit = 0; bit < 9; ++bit) {
            const bool set           = ((expert >> bit) & 1) != 0;
            const std::uint32_t same = __builtin_amdgcn_ballot_w32(set);
            peers &= set ? same : ~same;
        }
        if (e < entries) {
            std::int32_t position = -1;
            if (valid) {
                const std::uint32_t at =
                    bases[wave][expert] +
                    static_cast<std::uint32_t>(__builtin_popcount(peers & lower_lanes));
                p.token_of[at]  = static_cast<std::uint32_t>(e / kRoutes);
                p.weight_of[at] = routes[i].weight;
                position        = static_cast<std::int32_t>(at);
            }
            p.position[e] = position;
        }
        wave_lds_sync();
        if (valid && (peers & lower_lanes) == 0) {
            bases[wave][expert] += static_cast<std::uint32_t>(__builtin_popcount(peers));
        }
        wave_lds_sync();
    }
}

// --- Codecs --------------------------------------------------------------------------------------

// Eight codes of a run (nibble n at bits 4n..4n+3, low three bits used) through a scaled table
// {lo[0], lo[1], hi[0], hi[1]} to four BF16 pairs in pair order (0,2), (4,6), (1,3), (5,7). With
// `Signed`, nibble bit 3 flips the value's sign (NVFP4).
template <bool Signed>
__device__ __forceinline__ WmmaBf16 decode_run(std::uint32_t word, const U32x4& table) {
    const std::uint32_t even = word & 0x07070707U;
    const std::uint32_t odd  = (word >> 4) & 0x07070707U;
    const std::uint32_t el   = __builtin_amdgcn_perm(table.y, table.x, even);
    std::uint32_t eh         = __builtin_amdgcn_perm(table.w, table.z, even);
    const std::uint32_t ol   = __builtin_amdgcn_perm(table.y, table.x, odd);
    std::uint32_t oh         = __builtin_amdgcn_perm(table.w, table.z, odd);
    if constexpr (Signed) {
        eh ^= (word << 4) & 0x80808080U;
        oh ^= word & 0x80808080U;
    }
    const U32x4 pairs = {
        __builtin_amdgcn_perm(eh, el, 0x05010400U), __builtin_amdgcn_perm(eh, el, 0x07030602U),
        __builtin_amdgcn_perm(oh, ol, 0x05010400U), __builtin_amdgcn_perm(oh, ol, 0x07030602U)};
    return __builtin_bit_cast(WmmaBf16, pairs);
}

__device__ __forceinline__ const U32x4& scaled_table(const std::uint8_t* tables, int table,
                                                     std::uint32_t scale) {
    return *reinterpret_cast<const U32x4*>(tables + table * kTableBytes +
                                           16 * static_cast<int>(scale));
}

// A lane's record words of one stage: for both of its rows (r = 0, 1 at units k0 and k0 + 2) the
// code words of lane L, and the scale bytes. Loads are 4- or 8-byte reads of the record's lane
// layout ("Routed-expert record", ninfer/ops/moe_experts.h).
struct F3Codec {
    static constexpr int kChunkBytes = static_cast<int>(kMoeF3ChunkBytes);

    struct Raw {
        std::uint32_t words[2][3];
        std::uint32_t scales; // bytes k = 0..3 of lane L
    };

    struct Tables {
        U32x4 rows[2];
    };

    __device__ __forceinline__ static Raw load(RecordBytes chunk, std::uint32_t unit_lane,
                                               std::uint32_t k0) {
        Raw raw;
        const std::uint32_t words = 128 + 16 * unit_lane + 4 * k0;
#pragma unroll
        for (int r = 0; r < 2; ++r) {
#pragma unroll
            for (int j = 0; j < 3; ++j) {
                raw.words[r][j] =
                    load_record<std::uint32_t>(chunk + (words + 16 * kWaveSize * j + 8 * r));
            }
        }
        raw.scales = load_record<std::uint32_t>(chunk + 4 * unit_lane);
        return raw;
    }

    // One 32-K scale per row: table F3 book `row_table[r]`.
    __device__ __forceinline__ static Tables tables(const Raw& raw, const std::uint8_t* lds,
                                                    const int (&row_table)[2], int /*e2m1*/,
                                                    int k0) {
        Tables t;
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            t.rows[r] = scaled_table(lds, row_table[r], (raw.scales >> (8 * (k0 + 2 * r))) & 0xFFU);
        }
        return t;
    }

    // Run j of row r: word j < 3, or for j = 3 the codes gathered from the words' high bits.
    template <int J>
    __device__ __forceinline__ static WmmaBf16 decode(const Raw& raw, const Tables& t, int r) {
        std::uint32_t word = 0;
        if constexpr (J < 3) {
            word = raw.words[r][J];
        } else {
            word = ((raw.words[r][0] >> 3) & 0x11111111U) | ((raw.words[r][1] >> 2) & 0x22222222U) |
                   ((raw.words[r][2] >> 1) & 0x44444444U);
        }
        return decode_run<false>(word, t.rows[r]);
    }
};

struct Nvfp4Codec {
    static constexpr int kChunkBytes = static_cast<int>(kMoeNvfp4ChunkBytes);

    struct Raw {
        std::uint32_t words[2][4];
        std::uint32_t scales[2]; // [r]: bytes 2 (k0 + 2 r - 2 r) + h, 16-K halves h
    };

    struct Tables {
        U32x4 rows[2][2]; // [row][16-K half]
    };

    __device__ __forceinline__ static Raw load(RecordBytes chunk, std::uint32_t unit_lane,
                                               std::uint32_t k0) {
        Raw raw;
        const std::uint32_t words = 256 + 16 * unit_lane + 4 * k0;
#pragma unroll
        for (int r = 0; r < 2; ++r) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                raw.words[r][j] =
                    load_record<std::uint32_t>(chunk + (words + 16 * kWaveSize * j + 8 * r));
            }
        }
        const auto scales = load_record<U32x2>(chunk + 8 * unit_lane);
        raw.scales[0]     = scales.x; // units 0, 1
        raw.scales[1]     = scales.y; // units 2, 3
        return raw;
    }

    // Unit k0 + 2 r holds byte 2 k0 + h of scales[r] for its 16-K half h; every row reads the
    // E2M1 table.
    __device__ __forceinline__ static Tables tables(const Raw& raw, const std::uint8_t* lds,
                                                    const int (& /*row_table*/)[2], int e2m1,
                                                    int k0) {
        Tables t;
#pragma unroll
        for (int r = 0; r < 2; ++r) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                t.rows[r][h] =
                    scaled_table(lds, e2m1, (raw.scales[r] >> (8 * (2 * k0 + h))) & 0xFFU);
            }
        }
        return t;
    }

    template <int J>
    __device__ __forceinline__ static WmmaBf16 decode(const Raw& raw, const Tables& t, int r) {
        return decode_run<true>(raw.words[r][J], t.rows[r][J / 2]);
    }
};

// --- GEMM ----------------------------------------------------------------------------------------

// gate|up plane: K = 2560 over 40 stages, A rows are x rows of the entries' tokens.
struct GateUpPlane {
    static constexpr int kStages     = kMoeHidden / kStageK;
    static constexpr int kGroups     = 5;
    static constexpr int kTables     = 3; // gate book, up book, E2M1
    static constexpr int kClaim      = kClaimGateUp;
    static constexpr int kChunkFirst = 0;
    static constexpr int kTileChunks = 20;
    static constexpr int kRowLength  = kMoeHidden;

    __device__ __forceinline__ static int a_row(const GemmParams& p, int position) {
        return static_cast<int>(p.token_of[position]);
    }

    // Gate row 16 tile + n and up row 16 tile + n.
    __device__ __forceinline__ static void finish(const GemmParams& p, const Job& job, int expert,
                                                  int count, int m_blocks, int tile, int lane,
                                                  const WmmaF32 (&acc)[2][kMBlocks]) {
        const float gate_scale = p.matrix_scales[3 * expert];
        const float up_scale   = p.matrix_scales[3 * expert + 1];
        const int column       = 16 * tile + lane % 16;
        const int token0       = 8 * (lane / 16);
#pragma unroll
        for (int mb = 0; mb < kMBlocks; ++mb) {
            if (mb >= m_blocks) { break; }
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int token = 16 * mb + token0 + e;
                if (token < count) {
                    const float product =
                        silu(acc[0][mb][e] * gate_scale) * (acc[1][mb][e] * up_scale);
                    p.gated[(job.first + token) * kMoeIntermediate + column] = bf16_round(product);
                }
            }
        }
    }
};

// down plane: K = 640 over 10 stages, A rows are the entries' gated rows.
struct DownPlane {
    static constexpr int kStages     = kMoeIntermediate / kStageK;
    static constexpr int kGroups     = 10;
    static constexpr int kTables     = 2; // down book, E2M1
    static constexpr int kClaim      = kClaimDown;
    static constexpr int kChunkFirst = static_cast<int>(kMoeGateUpChunks);
    static constexpr int kTileChunks = 5;
    static constexpr int kRowLength  = kMoeIntermediate;

    __device__ __forceinline__ static int a_row(const GemmParams& /*p*/, int position) {
        return position;
    }

    // Down rows 32 tile + n and 32 tile + 16 + n of each entry's FP32 contribution.
    __device__ __forceinline__ static void finish(const GemmParams& p, const Job& job, int expert,
                                                  int count, int m_blocks, int tile, int lane,
                                                  const WmmaF32 (&acc)[2][kMBlocks]) {
        const float down_scale = p.matrix_scales[3 * expert + 2];
        const int column       = 32 * tile + lane % 16;
        const int token0       = 8 * (lane / 16);
#pragma unroll
        for (int mb = 0; mb < kMBlocks; ++mb) {
            if (mb >= m_blocks) { break; }
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int token = 16 * mb + token0 + e;
                if (token < count) {
                    const int position = job.first + token;
                    const float weight = p.weight_of[position];
                    float* row         = p.contributions + position * kMoeHidden + column;
                    row[0]             = acc[0][mb][e] * down_scale * weight;
                    row[16]            = acc[1][mb][e] * down_scale * weight;
                }
            }
        }
    }
};

// Builds the workgroup's scaled tables: table i, scale byte s, entry v = BF16(book_i[v] * s).
template <int Tables>
__device__ __forceinline__ void build_tables(std::uint8_t* lds, const std::uint32_t (&books)[3]) {
    for (int index = static_cast<int>(threadIdx.x); index < Tables * kScaleCodes;
         index += kGemmThreads) {
        const int table          = index / kScaleCodes;
        const std::uint32_t book = books[table];
        const float scale        = e4m3_to_float(static_cast<std::uint32_t>(index % kScaleCodes));
        U32x4 bytes              = {0U, 0U, 0U, 0U};
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            const std::uint32_t nibble = (book >> (4 * v)) & 0xFU;
            const std::uint32_t code   = nibble & 7U;
            // E2M1 magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6.
            const float magnitude =
                code == 0   ? 0.0F
                : code == 1 ? 0.5F
                            : static_cast<float>((2U + (code & 1U)) << (code >> 1)) * 0.25F;
            const float value = (nibble & 8U) != 0 ? -magnitude : magnitude;
            // Exact: the product has at most six significant bits.
            const std::uint32_t bits = __float_as_uint(value * scale) >> 16;
            const int word           = v / 4;
            const int shift          = 8 * (v % 4);
            bytes[word] |= (bits & 0xFFU) << shift;
            bytes[2 + word] |= (bits >> 8) << shift;
        }
        *reinterpret_cast<U32x4*>(lds + index * 16) = bytes;
    }
}

// One item: the job's tokens times the wave's record tile, double-buffered over Plane::kStages.
template <class Plane, class Codec>
__device__ __forceinline__ void run_item(const GemmParams& p, std::uint8_t* lds, const Job& job,
                                         int group) {
    const int tid       = static_cast<int>(threadIdx.x);
    const int lane      = tid % kWaveSize;
    const int wave      = tid / kWaveSize;
    const int expert    = __builtin_amdgcn_readfirstlane(static_cast<int>(job.shape >> 16));
    const int count     = __builtin_amdgcn_readfirstlane(static_cast<int>(job.shape & 0xFFFFU));
    const int first     = __builtin_amdgcn_readfirstlane(job.first);
    const int m_blocks  = (count + 15) / 16;
    const int tile      = kGemmWaves * group + wave;
    const int n         = lane % 16;
    const int h         = lane / 16;
    const int k0        = n / 8;
    const auto entry_lo = static_cast<std::uint32_t>(
        __builtin_amdgcn_readfirstlane(static_cast<int>(job.entry & 0xFFFFFFFFU)));
    const auto entry_hi = static_cast<std::uint32_t>(
        __builtin_amdgcn_readfirstlane(static_cast<int>(job.entry >> 32)));
    const RecordBytes chunks =
        record_address((static_cast<std::uint64_t>(entry_hi) << 32) | entry_lo) +
        static_cast<std::int64_t>(Plane::kChunkFirst + Plane::kTileChunks * tile) *
            Codec::kChunkBytes;
    const std::uint8_t* tables = lds;
    std::uint8_t* stages       = lds + Plane::kTables * kTableBytes;
    // F3 rows read their matrix's book table; NVFP4 rows read the E2M1 table (the last).
    const int row_table[2] = {0, Plane::kTables == 3 ? 1 : 0};
    const int e2m1_table   = Plane::kTables - 1;

    // Staging: thread tid loads units idx = tid + 256 i: token idx / 8, unit idx % 8 (eight K).
    // Rows past the job's tokens repeat its last token and are never read; staging every row
    // keeps the loads unconditional, which keeps the compiler's load waits from draining the
    // next stage's loads.
    std::uint32_t sources[kUnitsPerThread]; // byte offsets in the activation matrix
    int targets[kUnitsPerThread];
#pragma unroll
    for (int i = 0; i < kUnitsPerThread; ++i) {
        const int index = tid + kGemmThreads * i;
        const int token = index / kStageUnits;
        const int unit  = index % kStageUnits;
        targets[i]      = token * kStageRowBytes + 16 * unit;
        sources[i]      = static_cast<std::uint32_t>(
            2 * (Plane::a_row(p, first + min(token, count - 1)) * Plane::kRowLength + 8 * unit));
    }
    uint4 staged[kUnitsPerThread];
    const auto stage_load = [&](int stage) {
#pragma unroll
        for (int i = 0; i < kUnitsPerThread; ++i) {
            staged[i] = *reinterpret_cast<const uint4*>(
                reinterpret_cast<const std::uint8_t*>(p.activations) + sources[i] +
                2 * kStageK * stage);
        }
    };
    // Each eight-K run in pair order: dwords (x0,x1) (x2,x3) (x4,x5) (x6,x7) -> (x0,x2) (x4,x6)
    // (x1,x3) (x5,x7).
    const auto stage_store = [&](int stage) {
        std::uint8_t* buffer = stages + (stage % 2) * kStageBytes;
#pragma unroll
        for (int i = 0; i < kUnitsPerThread; ++i) {
            const uint4 s = staged[i];
            *reinterpret_cast<uint4*>(buffer + targets[i]) =
                make_uint4(__builtin_amdgcn_perm(s.y, s.x, 0x05040100U),
                           __builtin_amdgcn_perm(s.w, s.z, 0x05040100U),
                           __builtin_amdgcn_perm(s.y, s.x, 0x07060302U),
                           __builtin_amdgcn_perm(s.w, s.z, 0x07060302U));
        }
    };
    // Lane L of the record unit this lane's half takes in a stage of half hc.
    const auto load_weights = [&](int stage) {
        const auto unit_lane = static_cast<std::uint32_t>(4 * (n % 8) + 2 * (stage % 2) + h);
        return Codec::load(chunks + (stage / 2) * Codec::kChunkBytes, unit_lane,
                           static_cast<std::uint32_t>(k0));
    };

    WmmaF32 acc[2][kMBlocks];
#pragma unroll
    for (int mb = 0; mb < kMBlocks; ++mb) {
        acc[0][mb] = WmmaF32{};
        acc[1][mb] = WmmaF32{};
    }
    const auto compute = [&](int stage, const typename Codec::Raw& raw) {
        const std::uint8_t* buffer          = stages + (stage % 2) * kStageBytes;
        const typename Codec::Tables scaled = Codec::tables(raw, tables, row_table, e2m1_table, k0);
        const auto step                     = [&]<int S>() {
            const WmmaBf16 b0 = Codec::template decode<S>(raw, scaled, 0);
            const WmmaBf16 b1 = Codec::template decode<S>(raw, scaled, 1);
#pragma unroll
            for (int mb = 0; mb < kMBlocks; ++mb) {
                if (mb < m_blocks) {
                    const WmmaBf16 a = *reinterpret_cast<const WmmaBf16*>(
                        buffer + (16 * mb + n) * kStageRowBytes + 16 * (4 * h + S));
                    acc[0][mb] = wmma(a, b0, acc[0][mb]);
                    acc[1][mb] = wmma(a, b1, acc[1][mb]);
                }
            }
        };
        step.template operator()<0>();
        __builtin_amdgcn_sched_barrier(0);
        step.template operator()<1>();
        __builtin_amdgcn_sched_barrier(0);
        step.template operator()<2>();
        __builtin_amdgcn_sched_barrier(0);
        step.template operator()<3>();
    };

    // Stage t computes from `current` and LDS buffer t % 2 while stage t + 1 loads. The loads of
    // stage t + 1 reach the next iteration only through `current`, so they cannot sink below this
    // iteration's compute; the scheduling barriers keep the compute between loads and stores.
    typename Codec::Raw current = load_weights(0);
    stage_load(0);
    stage_store(0);
    lds_barrier();
    for (int stage = 0; stage + 1 < Plane::kStages; ++stage) {
        const typename Codec::Raw next = load_weights(stage + 1);
        stage_load(stage + 1);
        __builtin_amdgcn_sched_barrier(0);
        compute(stage, current);
        __builtin_amdgcn_sched_barrier(0);
        stage_store(stage + 1);
        lds_barrier();
        current = next;
    }
    compute(Plane::kStages - 1, current);
    Plane::finish(p, job, expert, count, m_blocks, tile, lane, acc);
}

// Grid: kGemmBlocksPerUnit per WGP (persistent), block kGemmThreads, dynamic LDS
// gemm_lds_bytes<Plane>(): the scaled tables, then two activation stages [128][144 bytes].
template <class Plane>
__global__ void __launch_bounds__(kGemmThreads) gemm_kernel(const GemmParams p) {
    extern __shared__ __attribute__((aligned(16))) std::uint8_t lds[];
    __shared__ int s_item;
    build_tables<Plane::kTables>(lds, p.books);
    const int items = static_cast<int>(p.control[kJobCount]) * Plane::kGroups;
    for (;;) {
        // The previous item's last barrier ordered every read of s_item before this write.
        if (threadIdx.x == 0) {
            s_item = static_cast<int>(atomicAdd(p.claims + Plane::kClaim, 1U));
        }
        lds_barrier(); // also publishes the tables before the first item
        const int item = s_item;
        if (item >= items) { return; }
        const Job job = p.jobs[item / Plane::kGroups];
        if ((job.entry & 1U) != 0) {
            run_item<Plane, F3Codec>(p, lds, job, item % Plane::kGroups);
        } else {
            run_item<Plane, Nvfp4Codec>(p, lds, job, item % Plane::kGroups);
        }
    }
}

template <class Plane>
constexpr int gemm_lds_bytes() {
    return Plane::kTables * kTableBytes + 2 * kStageBytes;
}

// --- Combine -------------------------------------------------------------------------------------

// Grid: min(max_tokens, kCombineBlocksPerUnit per WGP), block kCombineThreads; token t by
// workgroup t mod grid. Rows t >= N are not written.
__global__ void __launch_bounds__(kCombineThreads) combine_kernel(const CombineParams p) {
    const int tokens          = token_count(p.tokens, p.max_tokens);
    const auto* contributions = reinterpret_cast<const float4*>(p.contributions);
    auto* y                   = reinterpret_cast<float4*>(p.y);
    for (int t = static_cast<int>(blockIdx.x); t < tokens; t += static_cast<int>(gridDim.x)) {
        int positions[kRoutes];
#pragma unroll
        for (int j = 0; j < kRoutes; ++j) {
            positions[j] = __builtin_amdgcn_readfirstlane(p.position[t * kRoutes + j]);
        }
#pragma unroll
        for (int c = 0; c < kRowVectors / kCombineThreads; ++c) {
            const int column = static_cast<int>(threadIdx.x) + kCombineThreads * c;
            float4 sum       = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
#pragma unroll
            for (int j = 0; j < kRoutes; ++j) {
                if (positions[j] >= 0) {
                    const float4 v = contributions[positions[j] * kRowVectors + column];
                    sum.x += v.x;
                    sum.y += v.y;
                    sum.z += v.z;
                    sum.w += v.w;
                }
            }
            y[t * kRowVectors + column] = sum;
        }
    }
}

// Persistent GEMM workgroups per WGP: two fit the LDS of a WGP (128 KiB) at 48 KiB each.
constexpr int kGemmBlocksPerUnit    = 2;
constexpr int kCombineBlocksPerUnit = 16;

template <class Kernel, class Params>
void launch(Kernel kernel, int grid, int threads, int lds_bytes, const Params& params,
            hipStream_t stream, const char* what) {
    Params copy         = params;
    void* kernel_args[] = {&copy};
    // hipLaunchKernel returns this launch's status, not an earlier error still latched in the
    // runtime's last-error state.
    hip_check(hipLaunchKernel(reinterpret_cast<const void*>(kernel),
                              dim3(static_cast<unsigned>(grid)),
                              dim3(static_cast<unsigned>(threads)), kernel_args,
                              static_cast<std::size_t>(lds_bytes), stream),
              what);
}

} // namespace

std::size_t moe_helper_grouped_workspace_bytes(std::int32_t max_tokens) {
    return layout_of(max_tokens).bytes;
}

void moe_helper_experts_grouped(const MoeHelperExpertsArgs& args, std::byte* region,
                                std::int32_t capacity, int device, hipStream_t stream) {
    const Layout layout = layout_of(capacity);
    auto* control       = reinterpret_cast<std::uint32_t*>(region);
    auto* jobs          = reinterpret_cast<Job*>(region + layout.jobs);
    auto* token_of      = reinterpret_cast<std::uint32_t*>(region + layout.token_of);
    auto* weight_of     = reinterpret_cast<float*>(region + layout.weight_of);
    auto* position      = reinterpret_cast<std::int32_t*>(region + layout.position);
    auto* gated         = reinterpret_cast<std::uint16_t*>(region + layout.gated);
    auto* contributions = reinterpret_cast<float*>(region + layout.contributions);

    int units = 0;
    hip_check(hipDeviceGetAttribute(&units, hipDeviceAttributeMultiprocessorCount, device),
              "multiprocessor count");

    const GroupParams group{args.tokens,
                            args.routes,
                            args.bank.expert_table,
                            control,
                            reinterpret_cast<std::uint32_t*>(region + layout.counts),
                            jobs,
                            token_of,
                            weight_of,
                            position,
                            args.max_tokens};
    const int group_grid = static_cast<int>(
        ceil_div(static_cast<std::int64_t>(kRoutes) * args.max_tokens, kGroupEntries));
    launch(&count_routes_kernel, group_grid, kGroupThreads, 0, group, stream, "count launch");
    launch(&scatter_routes_kernel, group_grid, kGroupThreads, 0, group, stream, "scatter launch");

    GemmParams gemm{};
    gemm.control        = control;
    gemm.claims         = control;
    gemm.jobs           = jobs;
    gemm.token_of       = token_of;
    gemm.weight_of      = weight_of;
    gemm.matrix_scales  = args.bank.matrix_scales;
    gemm.gated          = gated;
    gemm.contributions  = contributions;
    const int gemm_grid = kGemmBlocksPerUnit * units;

    gemm.activations = args.x;
    gemm.books[0]    = args.bank.f3_books[0];
    gemm.books[1]    = args.bank.f3_books[1];
    gemm.books[2]    = kE2m1Book;
    launch(&gemm_kernel<GateUpPlane>, gemm_grid, kGemmThreads, gemm_lds_bytes<GateUpPlane>(), gemm,
           stream, "gate/up launch");

    gemm.activations = gated;
    gemm.books[0]    = args.bank.f3_books[2];
    gemm.books[1]    = kE2m1Book;
    gemm.books[2]    = 0;
    launch(&gemm_kernel<DownPlane>, gemm_grid, kGemmThreads, gemm_lds_bytes<DownPlane>(), gemm,
           stream, "down launch");

    const CombineParams combine{args.tokens, position, contributions, args.y, args.max_tokens};
    launch(&combine_kernel, std::min(args.max_tokens, kCombineBlocksPerUnit * units),
           kCombineThreads, 0, combine, stream, "combine launch");
}

} // namespace ninfer::ops::detail
