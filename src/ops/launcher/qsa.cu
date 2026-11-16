#include "ops/launcher/qsa.h"

#include "core/device.h"
#include "ops/common/memory.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"

#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kHeadDim       = kQsaHeadDim;
constexpr int kKvHeads       = kQsaKvHeads;
constexpr int kQueryHeads    = kQsaQueryHeads;
constexpr int kGroupHeads    = kQueryHeads / kKvHeads;
constexpr int kCodeBytes     = kHeadDim / 2;  // NVFP4 code bytes of one token's KV head
constexpr int kScaleGroups   = kHeadDim / 16; // NVFP4 scales of one token's KV head
constexpr int kIndexDim      = kQsaIndexHeadDim;
constexpr int kIndexHeads    = kQsaIndexQueryHeads;
constexpr int kBlockTokens   = kQsaBlockTokens;
constexpr float kNormEpsilon = 1.0e-6F;
constexpr int kLaneFeatures  = kIndexDim / kWarpSize; // 4: lane l holds features l + 32i
constexpr int kPageTokens    = kQsaPageTokens;
constexpr int kPageBlocks    = kQsaPageBlocks;
static_assert(kQueryHeads % kKvHeads == 0);
static_assert(kPageTokens % kBlockTokens == 0);
static_assert(kIndexDim == 4 * kWarpSize);

// The rotary embedding rotates the first 64 index features as 32 rotate-half pairs (l, l + 32),
// exactly one pair per warp lane. inv_frequency[i] = 1e7^(-2i/64) in FP64.
constexpr int kRotaryPairs = 32;
static_assert(kRotaryPairs == kWarpSize);
__constant__ double kIndexRopeInvFrequency[kRotaryPairs] = {
    1.00000000000000000e+00, 6.04296390238132863e-01, 3.65174127254837722e-01,
    2.20673406908458991e-01, 1.33352143216332403e-01, 8.05842187761481865e-02,
    4.86967525165863113e-02, 2.94272717620928173e-02, 1.77827941003892293e-02,
    1.07460782832131743e-02, 6.49381631576211298e-03, 3.92418975848453627e-03,
    2.37137370566165538e-03, 1.43301257023696268e-03, 8.65964323360065387e-04,
    5.23299114681494734e-04, 3.16227766016837939e-04, 1.91095297497044048e-04,
    1.15478198468945822e-04, 6.97830584859866353e-05, 4.21696503428582224e-05,
    2.54829674797934641e-05, 1.53992652605949185e-05, 9.30572040929699043e-06,
    5.62341325190349121e-06, 3.39820832894255927e-06, 2.05352502645714607e-06,
    1.24093776075171953e-06, 7.49894209332455848e-07, 4.53158363760081793e-07,
    2.73841963426436139e-07, 1.65481709994318135e-07,
};

__device__ __forceinline__ float bf16_bits_to_float(std::uint16_t bits) {
    return __uint_as_float(static_cast<std::uint32_t>(bits) << 16);
}

__device__ __forceinline__ void unpack_bf16x8(const uint4& packed, float* out) {
    const std::uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        out[2 * i]     = __uint_as_float(words[i] << 16);
        out[2 * i + 1] = __uint_as_float(words[i] & 0xFFFF0000U);
    }
}

// --- paged state addressing --------------------------------------------------------------------

// The block-table row of packed column t, or -1 when the row's lane is outside the table.
__device__ __forceinline__ int column_lane(const QsaColumns& columns, const QsaStatePlanes& state,
                                           int token) {
    const int lane = columns.lanes[token / columns.width];
    return lane >= 0 && lane < state.table_rows ? lane : -1;
}

// The logical slot of packed column t, or -1 for padding, an out-of-range lane, or a slot past
// the lane capacity 64 * logical_pages.
__device__ __forceinline__ int column_slot(const QsaColumns& columns, const QsaStatePlanes& state,
                                           int token) {
    const int slot = columns.slots[token];
    if (slot < 0 || slot >= kPageTokens * state.logical_pages) { return -1; }
    return column_lane(columns, state, token) >= 0 ? slot : -1;
}

// `slot` of a column whose row reads lane `lane` (-1 for an out-of-range lane), or -1 for padding
// or a slot past the lane capacity. Columns of one row share the lane, so score groups and tiles
// resolve it once instead of per column.
__device__ __forceinline__ int lane_slot(const QsaStatePlanes& state, int lane, int slot) {
    return lane >= 0 && slot >= 0 && slot < kPageTokens * state.logical_pages ? slot : -1;
}

// The physical page holding logical slot `slot` (in [0, capacity)) of lane `lane`, or -1 when the
// table maps it outside the pool.
__device__ __forceinline__ int slot_page(const QsaStatePlanes& state, int lane, int slot) {
    const int page = state.block_tables[static_cast<std::int64_t>(lane) * state.logical_pages +
                                        slot / kPageTokens];
    return page >= 0 && page < state.physical_pages ? page : -1;
}

// Element offsets of one slot or block of a page (see QsaPagedState).
__device__ __forceinline__ std::int64_t kv_row(int page, int head, int in_page) {
    return (static_cast<std::int64_t>(page) * kKvHeads + head) * kPageTokens + in_page;
}

__device__ __forceinline__ std::int64_t token_row(int page, int in_page) {
    return static_cast<std::int64_t>(page) * kPageTokens + in_page;
}

__device__ __forceinline__ const std::int32_t* slot_position(const QsaStatePlanes& state, int page,
                                                             int in_page) {
    return state.positions + static_cast<std::int64_t>(page) * 3 * kPageTokens + in_page;
}

__device__ __forceinline__ std::int64_t block_key_offset(int page, int in_page_block) {
    return (static_cast<std::int64_t>(page) * kPageBlocks + in_page_block) * kIndexDim;
}

// Sine and cosine of rotary pair `pair` at the MRoPE position of one state slot, whose axis a
// coordinate is position[a * 64]; the pair's axis is pair % 3 (temporal, height, width
// interleaved). The angle is formed and reduced modulo 2*pi in FP64, so its error stays at FP32
// rounding of the reduced angle for every position up to the 262144-token capacity instead of
// growing with the position.
__device__ __forceinline__ void index_rope_sincos(const std::int32_t* position, int pair,
                                                  float* sine, float* cosine) {
    constexpr double kInvTwoPi = 1.59154943091895336e-01;
    constexpr double kTwoPi    = 6.28318530717958648e+00;
    const int axis             = pair % 3;
    const double angle =
        static_cast<double>(position[axis * kPageTokens]) * kIndexRopeInvFrequency[pair];
    const auto reduced = static_cast<float>(angle - rint(angle * kInvTwoPi) * kTwoPi);
    sincosf(reduced, sine, cosine);
}

// One warp's 128-wide index vector, lane l holding features l + 32i in x[i]. RMS-normalizes it with
// gamma 1 + weight (FP32 sum of squares, warp butterfly), then rotates rotary pair l (features l
// and l + 32, i.e. x[0] and x[1]) at the slot position `position` (see index_rope_sincos). Every
// lane of the warp must call it.
__device__ __forceinline__ void normalize_rotate(float (&x)[kLaneFeatures],
                                                 const std::uint16_t* weight,
                                                 const std::int32_t* position, int lane) {
    float sum = 0.0F;
#pragma unroll
    for (int i = 0; i < kLaneFeatures; ++i) { sum = fmaf(x[i], x[i], sum); }
    sum                 = warp_sum(sum);
    const float inv_rms = rsqrtf(sum * (1.0F / static_cast<float>(kIndexDim)) + kNormEpsilon);
#pragma unroll
    for (int i = 0; i < kLaneFeatures; ++i) {
        x[i] = x[i] * inv_rms * (1.0F + bf16_bits_to_float(weight[lane + kWarpSize * i]));
    }
    float sine;
    float cosine;
    index_rope_sincos(position, lane, &sine, &cosine);
    const float first  = x[0];
    const float second = x[1];
    x[0]               = first * cosine - second * sine;
    x[1]               = second * cosine + first * sine;
}

// --- qsa_state_append --------------------------------------------------------------------------

constexpr int kAppendThreads = 128;

// Grid (W*B), block 128; CTA t appends packed column t into its lane's page unless its slot is
// padding, past the capacity, or in an unmapped page.
// BF16 state: thread j copies 16-byte word j % 32 of KV head (j % 64) / 32 of K (j < 64) or V.
// NVFP4 state: thread j < 64 encodes 16-feature group j % 16 of KV head (j % 32) / 16 of K (j < 32)
// or V, storing its eight code bytes as one 8-byte word and its scale byte. Threads 0..15 copy the
// raw index key's 16-byte words and threads 0..2 the three position axes. No shared memory.
template <QsaKvFormat Format>
__global__ void __launch_bounds__(kAppendThreads)
    append_kernel(const std::uint16_t* __restrict__ k, const std::uint16_t* __restrict__ v,
                  const std::uint16_t* __restrict__ index_keys,
                  const std::int32_t* __restrict__ positions, QsaColumns columns,
                  QsaStatePlanes state) {
    const int token = static_cast<int>(blockIdx.x);
    const int slot  = column_slot(columns, state, token);
    if (slot < 0) { return; }
    const int page = slot_page(state, column_lane(columns, state, token), slot);
    if (page < 0) { return; }
    const int in_page = slot % kPageTokens;
    const int thread  = static_cast<int>(threadIdx.x);
    if constexpr (Format == QsaKvFormat::BF16) {
        constexpr int kWords = kHeadDim / 8;
        static_assert(kAppendThreads == 2 * kKvHeads * kWords);
        const bool value  = thread >= kKvHeads * kWords;
        const int head    = (thread % (kKvHeads * kWords)) / kWords;
        const int word    = thread % kWords;
        const auto* input = reinterpret_cast<const uint4*>(value ? v : k);
        auto* output      = static_cast<uint4*>(value ? state.v : state.k);
        output[kv_row(page, head, in_page) * kWords + word] =
            input[(static_cast<std::int64_t>(token) * kKvHeads + head) * kWords + word];
    } else {
        constexpr int kGroups = kKvHeads * kScaleGroups;
        if (thread < 2 * kGroups) {
            const bool value = thread >= kGroups;
            const int head   = (thread % kGroups) / kScaleGroups;
            const int group  = thread % kScaleGroups;
            const std::uint16_t* input =
                (value ? v : k) + (static_cast<std::int64_t>(token) * kKvHeads + head) * kHeadDim +
                16 * group;
            const Nvfp4QuantizedK16 encoded =
                quantize_nvfp4_k16(reinterpret_cast<const __nv_bfloat16*>(input), 1.0F);
            const std::int64_t row = kv_row(page, head, in_page);
            auto* codes            = static_cast<std::uint8_t*>(value ? state.v : state.k);
            *reinterpret_cast<uint2*>(codes + row * kCodeBytes + 8 * group) =
                make_uint2(encoded.codes_lo, encoded.codes_hi);
            (value ? state.v_scales : state.k_scales)[row * kScaleGroups + group] = encoded.scale;
        }
    }
    constexpr int kIndexWords = kIndexDim / 8;
    if (thread < kIndexWords) {
        reinterpret_cast<uint4*>(
            state.index_keys)[token_row(page, in_page) * kIndexWords + thread] =
            reinterpret_cast<const uint4*>(
                index_keys)[static_cast<std::int64_t>(token) * kIndexWords + thread];
    }
    if (thread < 3) {
        // Input positions are [W,3,B]: axis a of column j of row b is element 3Wb + Wa + j.
        const int row    = token / columns.width;
        const int column = token - row * columns.width;
        state.positions[static_cast<std::int64_t>(page) * 3 * kPageTokens + thread * kPageTokens +
                        in_page] =
            positions[(static_cast<std::int64_t>(row) * 3 + thread) * columns.width + column];
    }
}

constexpr int kSealWarps = 8;

// Grid (ceil(W*B / 8)), block 256: warp w seals the block whose last slot packed column
// t = 8*blockIdx.x + w appended, after append_kernel wrote every column of the call. A block lies
// in one page. Lane l sums features l + 32i of the block's four raw state keys in slot order in
// FP32, scales by 0.25 and rounds the mean once to BF16 (the semantic pooling cast), then
// normalizes and rotates at the position of the block's first slot. No shared memory or barrier.
__global__ void __launch_bounds__(kSealWarps* kWarpSize)
    seal_kernel(QsaColumns columns, const std::uint16_t* __restrict__ norm_weight, int tokens,
                QsaStatePlanes state) {
    const int token =
        static_cast<int>(blockIdx.x) * kSealWarps + static_cast<int>(threadIdx.x) / kWarpSize;
    const int lane = static_cast<int>(threadIdx.x) % kWarpSize;
    if (token >= tokens) { return; }
    const int slot = column_slot(columns, state, token);
    if (slot < 0 || slot % kBlockTokens != kBlockTokens - 1) { return; }
    const int page = slot_page(state, column_lane(columns, state, token), slot);
    if (page < 0) { return; }
    const int in_page        = slot % kPageTokens;
    const int first_in_page  = in_page - (kBlockTokens - 1);
    const std::int64_t first = token_row(page, first_in_page);
    float x[kLaneFeatures];
#pragma unroll
    for (int i = 0; i < kLaneFeatures; ++i) {
        const int feature = lane + kWarpSize * i;
        float sum         = 0.0F;
#pragma unroll
        for (int r = 0; r < kBlockTokens; ++r) {
            sum += bf16_bits_to_float(state.index_keys[(first + r) * kIndexDim + feature]);
        }
        x[i] = __bfloat162float(__float2bfloat16_rn(sum * 0.25F));
    }
    normalize_rotate(x, norm_weight, slot_position(state, page, first_in_page), lane);
    float* key = state.block_keys + block_key_offset(page, in_page / kBlockTokens);
#pragma unroll
    for (int i = 0; i < kLaneFeatures; ++i) { key[lane + kWarpSize * i] = x[i]; }
}

// --- qsa_index_select --------------------------------------------------------------------------

constexpr int kQueryThreads = kIndexHeads * kWarpSize;

// Grid (W*B), block 128: warp h normalizes and rotates index-query head h of packed column
// blockIdx.x at the column's own slot position in its lane into query[t][h][128]; lane l holds
// features l + 32i. Padding, out-of-range, and unmapped columns store zeros so the score kernel
// never stages uninitialized workspace. No shared memory or barrier.
__global__ void __launch_bounds__(kQueryThreads)
    index_query_kernel(const std::uint16_t* __restrict__ index_query, QsaColumns columns,
                       const std::uint16_t* __restrict__ norm_weight, QsaStatePlanes state,
                       int max_visible_tokens, float* __restrict__ query) {
    const int token   = static_cast<int>(blockIdx.x);
    const int head    = static_cast<int>(threadIdx.x) / kWarpSize;
    const int lane    = static_cast<int>(threadIdx.x) % kWarpSize;
    const int slot    = column_slot(columns, state, token);
    const int page    = slot >= 0 ? slot_page(state, column_lane(columns, state, token), slot) : -1;
    const auto vector = (static_cast<std::int64_t>(token) * kIndexHeads + head) * kIndexDim;
    float* output     = query + vector;
    if (slot < 0 || slot >= max_visible_tokens || page < 0) {
#pragma unroll
        for (int i = 0; i < kLaneFeatures; ++i) { output[lane + kWarpSize * i] = 0.0F; }
        return;
    }
    float x[kLaneFeatures];
#pragma unroll
    for (int i = 0; i < kLaneFeatures; ++i) {
        x[i] = bf16_bits_to_float(index_query[vector + lane + kWarpSize * i]);
    }
    normalize_rotate(x, norm_weight, slot_position(state, page, slot % kPageTokens), lane);
#pragma unroll
    for (int i = 0; i < kLaneFeatures; ++i) { output[lane + kWarpSize * i] = x[i]; }
}

// A selection chunk: rows [row_begin, row_begin + row_count) and their columns
// [column_begin, column_begin + column_count), with row_count == 1 or whole rows. Chunk-local query
// i is row row_begin + i / column_count, column column_begin + i % column_count, and owns row i of
// the scores workspace. Score groups and tiles never span two rows, so they share one lane.
struct SelectChunk {
    int row_begin;
    int row_count;
    int column_begin;
    int column_count;
};

__device__ __forceinline__ int chunk_token(const QsaColumns& columns, const SelectChunk& chunk,
                                           int local) {
    const int row = local / chunk.column_count;
    return (chunk.row_begin + row) * columns.width + chunk.column_begin + local -
           row * chunk.column_count;
}

// The query's complete-block count when it must rank its blocks, otherwise zero.
__device__ __forceinline__ int ranked_blocks(int slot, int max_visible_tokens, int budget) {
    if (slot < 0 || slot >= max_visible_tokens) { return 0; }
    const int complete = (slot + 1) / kBlockTokens;
    return complete > budget ? complete : 0;
}

// Streaming scoring of short query chunks (fewer than 32 queries), where the block-key read is
// the cost: a CTA of 8 warps scores a group of 4 queries (16 query-head rows) against 32 * steps
// blocks, warp w taking 4 * steps consecutive blocks of them, four at a time; steps (1, 2, 4 or
// 8, so a warp's blocks never straddle more than two pages) is chosen per launch so short
// prefixes still spread over the GPU. Lane l holds features 4l..4l+3 of
// all 16 rows in registers and loads the same features of each block key (one coalesced 512-byte
// row per warp load), forms 16 partial dots, and a transposing butterfly (8 + 4 + 2 + 1 + 1
// shuffles) leaves lane l with row 8*b4 + 4*b3 + 2*b2 + b1 (b_i = bit i of l) summed over all
// features; two more shuffles sum the ReLU over the four heads of query 2*b4 + b3, which lanes
// 0, 8, 16, 24 store.
constexpr int kStreamQueries    = 4;
constexpr int kStreamRows       = kStreamQueries * kIndexHeads;
constexpr int kStreamWarps      = 8;
constexpr int kStreamMaxSteps   = 8;   // steps of four blocks per warp
constexpr int kStreamTargetCtas = 340; // two per SM of the 170-SM RTX 5090
static_assert(kStreamRows == 16 && kIndexDim == 4 * kWarpSize);
static_assert(4 * kStreamMaxSteps == 2 * kPageBlocks);

// Reduces v (16 partial sums per lane) across the warp; returns the full sum of row
// 8*b4 + 4*b3 + 2*b2 + b1 of the lane (lanes l and l^1 hold the same row).
__device__ __forceinline__ float transpose_reduce16(float (&v)[kStreamRows], int lane) {
#pragma unroll
    for (int width = 8, offset = 16; width >= 1; width /= 2, offset /= 2) {
        const bool upper = (lane & offset) != 0;
#pragma unroll
        for (int i = 0; i < width; ++i) {
            const float keep = upper ? v[width + i] : v[i];
            const float send = upper ? v[i] : v[width + i];
            v[i]             = keep + __shfl_xor_sync(kFullWarpMask, send, offset);
        }
    }
    return v[0] + __shfl_xor_sync(kFullWarpMask, v[0], 1);
}

// Grid (ceil(blocks / (32 * steps)), row_count * ceil(column_count / 4)), block 256: CTA row y
// scores the four columns 4g.. of chunk row r (y = r * groups + g) and writes scores[query][block]
// = sum_h max(dot(query[h], block_key), 0) (unscaled) for every block below each query's ranked
// count, reading the lane's block keys through its block table (four blocks of a step share a
// page). Blocks at or past the group's largest ranked count are not read (zero keys). No shared
// memory or barrier.
__global__ void __launch_bounds__(kStreamWarps* kWarpSize)
    stream_score_kernel(const float* __restrict__ query, const float* __restrict__ block_keys,
                        QsaStatePlanes state, QsaColumns columns, SelectChunk chunk, int blocks,
                        int budget, int max_visible_tokens, int steps, float* __restrict__ scores) {
    const int warp         = static_cast<int>(threadIdx.x) / kWarpSize;
    const int lane         = static_cast<int>(threadIdx.x) % kWarpSize;
    const int groups       = (chunk.column_count + kStreamQueries - 1) / kStreamQueries;
    const int chunk_row    = static_cast<int>(blockIdx.y) / groups;
    const int group_column = (static_cast<int>(blockIdx.y) - chunk_row * groups) * kStreamQueries;
    // Chunk-local index of the group's first query.
    const int group_query = chunk_row * chunk.column_count + group_column;
    const int first       = (static_cast<int>(blockIdx.x) * kStreamWarps + warp) * 4 * steps;
    // steps is a power of two, so the warp's 4 * steps blocks start at a multiple of 4 * steps and
    // span one page (steps <= 4) or two (steps == 8, the pages of blocks first and first + 16).
    // Lanes 0 and 1 look them up before the slots are ranked, so the table load overlaps the
    // ranking instead of delaying the first key load. A page past the launch's blocks (whose
    // slots are within the lane capacity) is not looked up; blocks past the limit are never read.
    // The group's columns are consecutive columns of one row: packed index group_token + q.
    const int group_token = chunk_token(columns, chunk, group_query);
    const int key_lane    = column_lane(columns, state, group_token);
    int warp_page         = -1;
    if (key_lane >= 0 && lane < 2 && first + kPageBlocks * lane < blocks &&
        (lane == 0 || 4 * steps > kPageBlocks)) {
        warp_page = slot_page(state, key_lane, (first + kPageBlocks * lane) * kBlockTokens);
    }
    int ranked[kStreamQueries];
    int tokens[kStreamQueries];
    int limit = 0;
#pragma unroll
    for (int q = 0; q < kStreamQueries; ++q) {
        const bool present = group_column + q < chunk.column_count;
        tokens[q]          = present ? group_token + q : -1;
        ranked[q] = present
                        ? ranked_blocks(lane_slot(state, key_lane, columns.slots[group_token + q]),
                                        max_visible_tokens, budget)
                        : 0;
        limit     = max(limit, ranked[q]);
    }
    if (first >= limit) { return; }

    float4 rows[kStreamRows];
#pragma unroll
    for (int r = 0; r < kStreamRows; ++r) {
        const int q = r / kIndexHeads;
        rows[r] = tokens[q] >= 0
                      ? *reinterpret_cast<const float4*>(
                            query +
                            (static_cast<std::int64_t>(tokens[q]) * kIndexHeads + r % kIndexHeads) *
                                kIndexDim +
                            4 * lane)
                      : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
    }
    const int row_query = 2 * ((lane >> 4) & 1) + ((lane >> 3) & 1);
    const int row_limit = ranked[0] * (row_query == 0) + ranked[1] * (row_query == 1) +
                          ranked[2] * (row_query == 2) + ranked[3] * (row_query == 3);
    // Lane l's slice of the keys of block first (and first + 16), or null for an unmapped page.
    const int pages[2] = {__shfl_sync(kFullWarpMask, warp_page, 0),
                          __shfl_sync(kFullWarpMask, warp_page, 1)};
    const float* page_keys[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        page_keys[i] = pages[i] >= 0
                           ? block_keys + block_key_offset(pages[i], first % kPageBlocks) + 4 * lane
                           : nullptr;
    }
    for (int step = 0; step < steps; ++step) {
        const int base = first + 4 * step;
        if (base >= limit) { break; }
        // Blocks base..base+3 are blocks 4 * step .. of the warp's range; page 1 from block 16.
        const float* step_keys = page_keys[step >= kPageBlocks / 4 ? 1 : 0];
        const float* keys_at =
            step_keys != nullptr ? step_keys + (4 * step % kPageBlocks) * kIndexDim : nullptr;
        float4 keys[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            keys[u] = base + u < limit && keys_at != nullptr
                          ? *reinterpret_cast<const float4*>(keys_at + u * kIndexDim)
                          : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
        }
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            float partial[kStreamRows];
#pragma unroll
            for (int r = 0; r < kStreamRows; ++r) {
                float dot  = rows[r].x * keys[u].x;
                dot        = fmaf(rows[r].y, keys[u].y, dot);
                dot        = fmaf(rows[r].z, keys[u].z, dot);
                partial[r] = fmaf(rows[r].w, keys[u].w, dot);
            }
            float total = fmaxf(transpose_reduce16(partial, lane), 0.0F);
            total += __shfl_xor_sync(kFullWarpMask, total, 2);
            total += __shfl_xor_sync(kFullWarpMask, total, 4);
            const int block = base + u;
            if (lane % 8 == 0 && block < row_limit) {
                scores[static_cast<std::int64_t>(group_query + row_query) * blocks + block] = total;
            }
        }
    }
}

// Register-tiled scoring of wide chunk rows: a CTA scores 32 columns of one chunk row against 64
// blocks and thread (ty, tx) = (thread / 16, thread % 16) owns columns {ty, ty + 16} x blocks
// {tx + 16j} (j < 4) x 4 heads, 32 FP32 accumulators. The 128-feature reduction runs in four
// stages of 32 features through shared memory: query_tile [32][4][32] FP32 (16 KiB; LDS.128 reads
// are broadcasts within each half-warp) and key_tile [64][36] FP32 (9 KiB; the four-float row pad
// makes the 16-byte reads of eight rows conflict-free). Each four-feature step issues 12 LDS.128
// for 128 FMAs, so the loop is FMA-bound. A tile's 64 blocks are four whole pages of its lane.
constexpr int kTiledQueries  = 32;
constexpr int kTiledBlocks   = 64;
constexpr int kTiledThreads  = 256;
constexpr int kTiledStage    = 32; // features per shared-memory stage
constexpr int kTiledKeyPitch = kTiledStage + 4;
constexpr int kTiledRows     = kTiledThreads / 16;
static_assert(kTiledQueries == 2 * kTiledRows && kTiledBlocks == 4 * 16);
static_assert(kTiledBlocks % kPageBlocks == 0);

// Grid (ceil(blocks / 64), row_count * ceil(column_count / 32)), block 256: CTA row y covers
// columns 32c.. of chunk row r (y = r * tiles + c) and writes scores[query][block] =
// sum_h max(dot(query[h], block_key), 0) (unscaled) for every block below each query's ranked
// count, like stream_score_kernel. Block keys are staged only below the CTA's largest ranked
// count (zero rows past it), so no unsealed key is read. Barriers: the shared limit is cleared and
// the tile's four page ids are published, then the limit is raised; per stage, one barrier
// publishes the staged tiles and one retires them.
__global__ void __launch_bounds__(kTiledThreads)
    tiled_score_kernel(const float* __restrict__ query, const float* __restrict__ block_keys,
                       QsaStatePlanes state, QsaColumns columns, SelectChunk chunk, int blocks,
                       int budget, int max_visible_tokens, float* __restrict__ scores) {
    __shared__ __align__(16) float query_tile[kTiledQueries][kIndexHeads][kTiledStage];
    __shared__ __align__(16) float key_tile[kTiledBlocks][kTiledKeyPitch];
    __shared__ int tile_pages[kTiledBlocks / kPageBlocks];
    __shared__ int ranked_limit;
    const int thread      = static_cast<int>(threadIdx.x);
    const int ty          = thread / 16;
    const int tx          = thread % 16;
    const int tiles       = (chunk.column_count + kTiledQueries - 1) / kTiledQueries;
    const int chunk_row   = static_cast<int>(blockIdx.y) / tiles;
    const int tile_column = (static_cast<int>(blockIdx.y) - chunk_row * tiles) * kTiledQueries;
    // Chunk-local index of the tile's first query.
    const int tile_query = chunk_row * chunk.column_count + tile_column;
    const int tile_block = static_cast<int>(blockIdx.x) * kTiledBlocks;
    // The tile's columns are consecutive columns of one row: packed index tile_token + i.
    const int tile_token = chunk_token(columns, chunk, tile_query);
    const int key_lane   = column_lane(columns, state, tile_token);

    int ranked[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int column = tile_column + ty + kTiledRows * i;
        ranked[i] = column < chunk.column_count
                        ? ranked_blocks(lane_slot(state, key_lane,
                                                  columns.slots[tile_token + ty + kTiledRows * i]),
                                        max_visible_tokens, budget)
                        : 0;
    }
    if (thread == 0) { ranked_limit = 0; }
    // The physical pages of the tile's four pages of blocks, published by the next barrier.
    if (thread < kTiledBlocks / kPageBlocks) {
        const int first_slot = (tile_block + kPageBlocks * thread) * kBlockTokens;
        tile_pages[thread]   = key_lane >= 0 && first_slot < kPageTokens * state.logical_pages
                                   ? slot_page(state, key_lane, first_slot)
                                   : -1;
    }
    __syncthreads();
    // Threads with tx == 0 cover every query row of the tile.
    if (tx == 0) {
        const int most = max(ranked[0], ranked[1]);
        if (most > tile_block) { atomicMax(&ranked_limit, most); }
    }
    __syncthreads();
    const int limit = ranked_limit;
    if (limit == 0) { return; }

    float accumulator[2][4][kIndexHeads] = {};
    for (int stage = 0; stage < kIndexDim; stage += kTiledStage) {
        constexpr int kQueryWords = kTiledQueries * kIndexHeads * kTiledStage / 4;
        for (int index = thread; index < kQueryWords; index += kTiledThreads) {
            const int row  = index / (kIndexHeads * kTiledStage / 4);
            const int head = (index / (kTiledStage / 4)) % kIndexHeads;
            const int word = index % (kTiledStage / 4);
            float4 value   = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
            if (tile_column + row < chunk.column_count) {
                const int token = tile_token + row;
                value           = *reinterpret_cast<const float4*>(
                    query + (static_cast<std::int64_t>(token) * kIndexHeads + head) * kIndexDim +
                    stage + 4 * word);
            }
            *reinterpret_cast<float4*>(&query_tile[row][head][4 * word]) = value;
        }
        constexpr int kKeyWords = kTiledBlocks * kTiledStage / 4;
        for (int index = thread; index < kKeyWords; index += kTiledThreads) {
            const int row   = index / (kTiledStage / 4);
            const int word  = index % (kTiledStage / 4);
            const int block = tile_block + row;
            float4 value    = make_float4(0.0F, 0.0F, 0.0F, 0.0F);
            if (block < limit) {
                const int page = tile_pages[row / kPageBlocks];
                if (page >= 0) {
                    value = *reinterpret_cast<const float4*>(
                        block_keys + block_key_offset(page, block % kPageBlocks) + stage +
                        4 * word);
                }
            }
            *reinterpret_cast<float4*>(&key_tile[row][4 * word]) = value;
        }
        __syncthreads();
#pragma unroll 2
        for (int feature = 0; feature < kTiledStage; feature += 4) {
            float4 keys[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                keys[j] = *reinterpret_cast<const float4*>(&key_tile[tx + 16 * j][feature]);
            }
#pragma unroll
            for (int i = 0; i < 2; ++i) {
#pragma unroll
                for (int head = 0; head < kIndexHeads; ++head) {
                    const float4 q = *reinterpret_cast<const float4*>(
                        &query_tile[ty + kTiledRows * i][head][feature]);
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        float dot               = accumulator[i][j][head];
                        dot                     = fmaf(q.x, keys[j].x, dot);
                        dot                     = fmaf(q.y, keys[j].y, dot);
                        dot                     = fmaf(q.z, keys[j].z, dot);
                        dot                     = fmaf(q.w, keys[j].w, dot);
                        accumulator[i][j][head] = dot;
                    }
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int local = tile_query + ty + kTiledRows * i;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int block = tile_block + tx + 16 * j;
            if (block >= ranked[i]) { continue; }
            float total = 0.0F;
#pragma unroll
            for (int head = 0; head < kIndexHeads; ++head) {
                total += fmaxf(accumulator[i][j][head], 0.0F);
            }
            scores[static_cast<std::int64_t>(local) * blocks + block] = total;
        }
    }
}

constexpr int kSelectThreads = 1024;
constexpr int kSelectWarps   = kSelectThreads / kWarpSize;
constexpr int kSelectTile    = 8 * kSelectThreads; // listing tile: eight blocks per thread
constexpr int kRadixBins     = 256;
// Chunks of at most kSelectClusterQueries queries whose rows exceed one listing tile are selected
// by a cluster of kSelectClusterCtas CTAs per query; larger chunks fill the GPU with one CTA per
// query, and a short row is cheaper on one CTA than across a cluster.
constexpr int kSelectClusterCtas    = 8;
constexpr int kSelectClusterQueries = 16;
static_assert(kSelectWarps == kWarpSize && kRadixBins == 8 * kWarpSize);
// Packed 16-bit tile counts: a tile holds fewer than 2^16 blocks.
static_assert(kSelectTile < (1 << 16));

// Unsigned comparison of the keys orders finite FP32 scores like FP32 comparison.
__device__ __forceinline__ std::uint32_t score_key(float score) {
    const std::uint32_t bits = __float_as_uint(score);
    return (bits & 0x80000000U) != 0 ? ~bits : bits | 0x80000000U;
}

// Inclusive warp prefix sum.
__device__ __forceinline__ int warp_inclusive_scan(int value, int lane) {
#pragma unroll
    for (int offset = 1; offset < kWarpSize; offset <<= 1) {
        const int other = __shfl_up_sync(kFullWarpMask, value, offset);
        if (lane >= offset) { value += other; }
    }
    return value;
}

// Exclusive prefix sum of `value` over the 1024-thread CTA in thread order; *total receives the
// CTA sum. warp_totals is shared [32]. Every thread must call it. Three barriers: the warp totals
// are published, scanned by warp 0, and kept intact until every thread has read them.
__device__ __forceinline__ int block_exclusive_scan(int value, int* warp_totals, int* total) {
    const int lane      = static_cast<int>(threadIdx.x) % kWarpSize;
    const int warp      = static_cast<int>(threadIdx.x) / kWarpSize;
    const int inclusive = warp_inclusive_scan(value, lane);
    if (lane == kWarpSize - 1) { warp_totals[warp] = inclusive; }
    __syncthreads();
    if (warp == 0) { warp_totals[lane] = warp_inclusive_scan(warp_totals[lane], lane); }
    __syncthreads();
    const int before = warp == 0 ? 0 : warp_totals[warp - 1];
    *total           = warp_totals[kSelectWarps - 1];
    __syncthreads();
    return before + inclusive - value;
}

// Thread block cluster primitives (PTX ISA 7.8, sm_90+).
__device__ __forceinline__ int cluster_cta_rank() {
    std::uint32_t rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(rank));
    return static_cast<int>(rank);
}

// Cluster-wide barrier: release orders this thread's shared-memory writes before every other
// CTA's acquire, so distributed shared-memory reads after it observe them.
__device__ __forceinline__ void cluster_sync() {
    asm volatile("barrier.cluster.arrive.release.aligned;\n"
                 "barrier.cluster.wait.acquire.aligned;\n" ::
                     : "memory");
}

// Adds `value` to the int at `local` (an address in this CTA's shared memory) in the shared memory
// of cluster CTA `rank`. The add is a fire-and-forget reduction; cluster_sync's release/acquire
// makes it visible to the target CTA.
__device__ __forceinline__ void cluster_shared_add(int* local, int rank, int value) {
    const auto address = static_cast<std::uint32_t>(__cvta_generic_to_shared(local));
    std::uint32_t remote;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(remote) : "r"(address), "r"(rank));
    asm volatile("red.relaxed.cluster.shared::cluster.add.s32 [%0], %1;\n" ::"r"(remote), "r"(value)
                 : "memory");
}

// Lists the visible prefix [0, visible) into the output column (count = visible) or, with
// visible = 0, an empty column; CTA `rank` of a cluster of `ctas` writes the 1024-entry stripes
// rank, rank + ctas, ... so the cluster shares the column.
__device__ __forceinline__ void list_prefix(std::int32_t* output, std::int32_t* count, int visible,
                                            int selected_slots, int rank, int ctas) {
    for (int i = rank * kSelectThreads + static_cast<int>(threadIdx.x); i < selected_slots;
         i += ctas * kSelectThreads) {
        output[i] = i < visible ? i : -1;
    }
    if (rank == 0 && threadIdx.x == 0) { *count = visible; }
}

// Adds one to histogram[digit] for every calling lane, one shared atomic per distinct digit of
// the warp (scores of nearby magnitude share their leading digits, so lanes collide heavily).
__device__ __forceinline__ void histogram_add(int* histogram, unsigned digit, unsigned active,
                                              int lane) {
    const unsigned peers = __match_any_sync(active, digit);
    if (lane == __ffs(peers) - 1) { atomicAdd(&histogram[digit], __popc(peers)); }
}

// Grid (chunk queries * Ctas), block 1024, cluster (Ctas): the Ctas CTAs of cluster i select for
// chunk-local query i and CTA r owns complete blocks [complete*r/Ctas, complete*(r+1)/Ctas).
// A query with at most `budget` complete blocks lists its whole prefix. Otherwise:
//  1. A four-pass most-significant-digit radix select over the orderable score keys finds the key
//     T of the budget-th best block and how many blocks with key T to take (remaining). Each pass
//     histograms the 8-bit digit of the keys matching the decided prefix (shared histogram [256],
//     warp-aggregated shared atomics); in a cluster every CTA adds its nonzero bins into every
//     CTA's `combined` sums (distributed shared-memory reductions, two buffers alternating per
//     pass, each cleared after its pass's decision and before the next cluster barrier), so no CTA
//     waits on a remote load. Warp 0 scans the sums from the highest digit (lane l owns digits
//     255-8l .. 248-8l) and publishes the next prefix and remaining rank to its CTA.
//  2. Each CTA counts its blocks with key > T (above) and key == T (ties) and adds both into the
//     `before` counts of every higher-ranked CTA; after a cluster barrier every CTA knows the
//     blocks of each kind before its range. No remote access follows, so CTAs may exit freely.
//  3. Tiles of 8192 blocks, eight consecutive per thread: one CTA-wide exclusive scan of the packed
//     per-thread counts (ties << 16 | above; a tile holds at most 8192 of each) gives every block
//     its global count of earlier above blocks A and ties G; an above block lands at entry
//     A + min(G, remaining) and a tie with G < remaining at A + G.
// Cluster barriers: the initial clear, one per pass, and one after the range counts. CTA 0 lists
// the tail and the count. Each CTA reads its range of the scores row from the chunk workspace
// (L2-resident) six times: four radix passes, the count, and the listing.
template <int Ctas>
__global__ void __launch_bounds__(kSelectThreads)
    select_kernel(const float* __restrict__ scores, QsaColumns columns, QsaStatePlanes state,
                  SelectChunk chunk, int blocks, int budget, int selected_slots,
                  int max_visible_tokens, std::int32_t* __restrict__ selected_ids,
                  std::int32_t* __restrict__ selected_count) {
    __shared__ int histogram[kRadixBins];   // this CTA's digit counts of the current pass
    __shared__ int combined[2][kRadixBins]; // the cluster's digit counts, alternating per pass
    __shared__ int before[2];               // above blocks and ties of lower-ranked CTAs
    __shared__ int warp_totals[kSelectWarps];
    __shared__ std::uint32_t pass_prefix;
    __shared__ int pass_remaining;
    const int rank       = Ctas > 1 ? cluster_cta_rank() : 0;
    const int thread     = static_cast<int>(threadIdx.x);
    const int lane       = thread % kWarpSize;
    const int warp       = thread / kWarpSize;
    const int query      = static_cast<int>(blockIdx.x) / Ctas;
    const int token      = chunk_token(columns, chunk, query);
    const int slot       = column_slot(columns, state, token);
    std::int32_t* output = selected_ids + static_cast<std::int64_t>(token) * selected_slots;
    // Both early exits depend only on the query, so they are uniform across the cluster.
    if (slot < 0 || slot >= max_visible_tokens) {
        list_prefix(output, selected_count + token, 0, selected_slots, rank, Ctas);
        return;
    }
    const int visible  = slot + 1;
    const int complete = visible / kBlockTokens;
    if (complete <= budget) {
        list_prefix(output, selected_count + token, visible, selected_slots, rank, Ctas);
        return;
    }

    const float* row      = scores + static_cast<std::int64_t>(query) * blocks;
    const int first       = complete * rank / Ctas;
    const int last        = complete * (rank + 1) / Ctas;
    std::uint32_t prefix  = 0;
    std::uint32_t decided = 0;
    int remaining         = budget;
    if constexpr (Ctas > 1) {
        // Every CTA's sums are cleared before any CTA adds into them.
        for (int i = thread; i < 2 * kRadixBins; i += kSelectThreads) { (&combined[0][0])[i] = 0; }
        if (thread < 2) { before[thread] = 0; }
        cluster_sync();
    }
    for (int pass = 0; pass < 4; ++pass) {
        const int shift = 24 - 8 * pass;
        if (thread < kRadixBins) { histogram[thread] = 0; }
        __syncthreads();
        for (int base = first; base < last; base += kSelectThreads) {
            const int block         = base + thread;
            const std::uint32_t key = block < last ? score_key(row[block]) : 0U;
            const bool counted      = block < last && (key & decided) == prefix;
            const unsigned active   = __ballot_sync(kFullWarpMask, counted);
            if (counted) { histogram_add(histogram, (key >> shift) & 0xFFU, active, lane); }
        }
        __syncthreads();
        const int* totals = histogram;
        if constexpr (Ctas > 1) {
            int* sums = combined[pass % 2];
            for (int i = thread; i < Ctas * kRadixBins; i += kSelectThreads) {
                const int bin = i % kRadixBins;
                if (histogram[bin] != 0) {
                    cluster_shared_add(&sums[bin], i / kRadixBins, histogram[bin]);
                }
            }
            cluster_sync();
            totals = sums;
        }
        if (warp == 0) {
            int counts[8];
            int lane_total = 0;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                counts[j] = totals[kRadixBins - 1 - 8 * lane - j];
                lane_total += counts[j];
            }
            const int through = warp_inclusive_scan(lane_total, lane);
            int above         = through - lane_total;
            if (above < remaining && remaining <= through) {
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    if (above + counts[j] >= remaining) {
                        const auto digit =
                            static_cast<std::uint32_t>(kRadixBins - 1 - 8 * lane - j);
                        pass_prefix    = prefix | (digit << shift);
                        pass_remaining = remaining - above;
                        break;
                    }
                    above += counts[j];
                }
            }
        }
        __syncthreads();
        prefix    = pass_prefix;
        remaining = pass_remaining;
        decided |= 0xFFU << shift;
        if constexpr (Ctas > 1) {
            // Cleared for pass + 2: every CTA adds into it only after the next pass's cluster
            // barrier, which this CTA reaches after clearing.
            if (thread < kRadixBins) { combined[pass % 2][thread] = 0; }
        }
    }

    // Blocks above the threshold key and ties in this CTA's range, added into every higher-ranked
    // CTA's `before` counts.
    int above_count = 0;
    int tie_count   = 0;
    for (int block = first + thread; block < last; block += kSelectThreads) {
        const std::uint32_t key = score_key(row[block]);
        above_count += key > prefix ? 1 : 0;
        tie_count += key == prefix ? 1 : 0;
    }
    // Separate sums: a range may hold up to 65536 ties (fewer than `budget` blocks are above).
    int above_total = 0;
    int tie_total   = 0;
    (void)block_exclusive_scan(above_count, warp_totals, &above_total);
    (void)block_exclusive_scan(tie_count, warp_totals, &tie_total);
    int above_before = 0;
    int ties_before  = 0;
    if constexpr (Ctas > 1) {
        if (thread > rank && thread < Ctas) {
            cluster_shared_add(&before[0], thread, above_total);
            cluster_shared_add(&before[1], thread, tie_total);
        }
        cluster_sync();
        above_before = before[0];
        ties_before  = before[1];
    }

    constexpr int kItems = kSelectTile / kSelectThreads;
    for (int tile = first; tile < last; tile += kSelectTile) {
        const int base = tile + kItems * thread;
        std::uint32_t keys[kItems];
        int packed = 0;
#pragma unroll
        for (int j = 0; j < kItems; ++j) {
            keys[j] = base + j < last ? score_key(row[base + j]) : 0U;
            packed += base + j < last && keys[j] > prefix ? 1 : 0;
            packed += base + j < last && keys[j] == prefix ? 1 << 16 : 0;
        }
        int tile_total   = 0;
        const int before = block_exclusive_scan(packed, warp_totals, &tile_total);
        int above_rank   = above_before + (before & 0xFFFF);
        int tie_rank     = ties_before + (before >> 16);
#pragma unroll
        for (int j = 0; j < kItems; ++j) {
            if (base + j >= last) { continue; }
            int position = -1;
            if (keys[j] > prefix) {
                position = above_rank + min(tie_rank, remaining);
                ++above_rank;
            } else if (keys[j] == prefix) {
                position = tie_rank < remaining ? above_rank + tie_rank : -1;
                ++tie_rank;
            }
            if (position >= 0) {
#pragma unroll
                for (int token_in_block = 0; token_in_block < kBlockTokens; ++token_in_block) {
                    output[kBlockTokens * position + token_in_block] =
                        kBlockTokens * (base + j) + token_in_block;
                }
            }
        }
        above_before += tile_total & 0xFFFF;
        ties_before += tile_total >> 16;
    }

    if (rank == 0) {
        const int listed     = kBlockTokens * budget;
        const int tail_begin = kBlockTokens * complete;
        for (int i = listed + thread; i < selected_slots; i += kSelectThreads) {
            const int id = tail_begin + i - listed;
            output[i]    = id < visible ? id : -1;
        }
        if (thread == 0) { selected_count[token] = listed + visible - tail_begin; }
    }
}

// --- qsa_selected_attention --------------------------------------------------------------------

// One attention CTA serves one token, one KV group (its 12 query heads as the rows of a 16-row MMA
// tile, rows 12..15 zero) and one split of the selected ranks, in tiles of 64 ranks: warp w forms
// the logits of ranks [16w, 16w+16) of the tile and accumulates features [64w, 64w+64) of the
// output over all 64 ranks.
constexpr int kAttentionWarps   = 4;
constexpr int kAttentionThreads = kAttentionWarps * kWarpSize;
constexpr int kKeyTile          = kQsaAttentionKeyTile;
constexpr int kWarpKeys         = kKeyTile / kAttentionWarps;
constexpr int kWarpFeatures     = kHeadDim / kAttentionWarps;
constexpr int kHeadRows         = 16;
constexpr int kQRowStride   = kHeadDim + 8;  // BF16 elements: 528-byte rows, ldmatrix conflict-free
constexpr int kPRowStride   = kKeyTile + 8;  // BF16 elements: 144-byte rows, ldmatrix conflict-free
constexpr int kOutRowStride = kHeadDim + 2;  // FP32 elements
constexpr int kKeySteps     = kHeadDim / 16; // MMA k-steps of one logit
constexpr int kValueTiles   = kWarpFeatures / 8;
// Logits are kept in the base-2 domain: dot / 16 * log2(e), so exp2 of their differences is the
// softmax weight.
constexpr float kLogitScaleLog2 = 0.0625F * 1.44269504088896341F;
static_assert(kWarpKeys == 16 && kWarpFeatures == 64 && kGroupHeads <= kHeadRows);
static_assert(kQsaAttentionMinSplitSlots % kKeyTile == 0);

struct AttentionShared {
    union {
        // The KV group's query rows in the logical k order of the logit MMA (see
        // logical_query_column).
        std::uint16_t q[kHeadRows][kQRowStride];
        // The output rows after the rank loop: normalized, or the split's unnormalized partial.
        // The two-float row pad limits the accumulator stores to two-way bank conflicts.
        float out[kHeadRows][kOutRowStride];
    };

    // The tile's softmax weights split as p = hi + lo, two BF16 values each, rows = heads.
    std::uint16_t p_hi[kHeadRows][kPRowStride];
    std::uint16_t p_lo[kHeadRows][kPRowStride];
    float warp_max[kAttentionWarps][kHeadRows];
    float warp_sum[kAttentionWarps][kHeadRows];
    // The listed in-range slot of each rank of a tile, -1 for none; two buffers alternate so the
    // next tile's slots are staged while the current tile runs.
    int tile_slots[2][kKeyTile];
};

__device__ __forceinline__ std::uint32_t bf16x2_bits(float lo, float hi) {
    const __nv_bfloat162 packed = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const std::uint32_t*>(&packed);
}

// The NVFP4 code byte (features 2i low nibble, 2i+1 high nibble) times its E4M3 scale as BF16x2:
// exact, since an E2M1 value times an E4M3 scale has at most six significant bits.
__device__ __forceinline__ std::uint32_t decode_nvfp4_bf16x2(std::uint32_t code, float scale) {
    const float2 pair = decode_nvfp4_e2m1x2(static_cast<std::uint8_t>(code));
    return bf16x2_bits(pair.x * scale, pair.y * scale);
}

// The logit MMA uses a permuted reduction order so each lane's key fragment is one contiguous
// run of its key row: in k-step s, logical columns 2c+e and 8+2c+e (c < 4, e < 2) hold features
// 64c+4s+e and 64c+4s+2+e. Lane (g = lane/4, c = lane%4) therefore needs features [64c, 64c+64) of
// its key. The query rows are stored in that logical order (logical column 16s + k), so ldmatrix
// yields matching A fragments.
__device__ __forceinline__ int logical_query_column(int feature) {
    const int owner = feature / 64;
    const int rest  = feature % 64;
    return 16 * (rest / 4) + 8 * ((rest % 4) / 2) + 2 * owner + rest % 2;
}

// B fragments (K^T, 16 k-steps x {b0, b1}) of one key row for lane quarter c, zero when !valid.
template <QsaKvFormat Format>
__device__ __forceinline__ void load_key_fragments(const QsaStatePlanes& state, std::int64_t row,
                                                   bool valid, int c,
                                                   std::uint32_t (&b)[kKeySteps][2]) {
    if constexpr (Format == QsaKvFormat::BF16) {
        const auto* source = reinterpret_cast<const uint4*>(
            static_cast<const std::uint16_t*>(state.k) + row * kHeadDim + 64 * c);
#pragma unroll
        for (int m = 0; m < kKeySteps / 2; ++m) {
            const uint4 words = valid ? __ldg(source + m) : make_uint4(0U, 0U, 0U, 0U);
            b[2 * m][0]       = words.x;
            b[2 * m][1]       = words.y;
            b[2 * m + 1][0]   = words.z;
            b[2 * m + 1][1]   = words.w;
        }
    } else {
        // Features [64c, 64c+64) are code bytes [32c, 32c+32) and scale groups [4c, 4c+4).
        const auto* codes = reinterpret_cast<const uint4*>(
            static_cast<const std::uint8_t*>(state.k) + row * kCodeBytes + 32 * c);
        const uint4 low              = valid ? __ldg(codes) : make_uint4(0U, 0U, 0U, 0U);
        const uint4 high             = valid ? __ldg(codes + 1) : make_uint4(0U, 0U, 0U, 0U);
        const std::uint32_t scales   = valid ? __ldg(reinterpret_cast<const std::uint32_t*>(
                                                   state.k_scales + row * kScaleGroups + 4 * c))
                                             : 0U;
        const std::uint32_t words[8] = {low.x, low.y, low.z, low.w, high.x, high.y, high.z, high.w};
#pragma unroll
        for (int step = 0; step < kKeySteps; ++step) {
            const float scale =
                decode_nvfp4_e4m3(static_cast<std::uint8_t>(scales >> (8 * (step / 4))));
            const std::uint32_t word = words[step / 2];
            const int shift          = 16 * (step % 2);
            b[step][0]               = decode_nvfp4_bf16x2(word >> shift, scale);
            b[step][1]               = decode_nvfp4_bf16x2(word >> (shift + 8), scale);
        }
    }
}

// The stored bits of eight features [feature, feature+8) of one value row: one 16-byte BF16 run,
// or four NVFP4 code bytes and their E4M3 scale byte.
template <QsaKvFormat Format>
struct ValueRun;

template <>
struct ValueRun<QsaKvFormat::BF16> {
    uint4 words;
};

template <>
struct ValueRun<QsaKvFormat::NVFP4G16> {
    std::uint32_t codes;
    std::uint32_t scale;
};

// Loads the run of `row` (zero when slot < 0) without decoding, so a tile's runs are in flight
// together with its key rows.
template <QsaKvFormat Format>
__device__ __forceinline__ ValueRun<Format>
fetch_value_run(const QsaStatePlanes& state, std::int64_t row, bool valid, int feature) {
    ValueRun<Format> run{};
    if (!valid) { return run; }
    if constexpr (Format == QsaKvFormat::BF16) {
        run.words = __ldg(reinterpret_cast<const uint4*>(
            static_cast<const std::uint16_t*>(state.v) + row * kHeadDim + feature));
    } else {
        run.codes = __ldg(reinterpret_cast<const std::uint32_t*>(
            static_cast<const std::uint8_t*>(state.v) + row * kCodeBytes + feature / 2));
        run.scale = state.v_scales[row * kScaleGroups + feature / 16];
    }
    return run;
}

// The eight features of a fetched run as floats (exact).
template <QsaKvFormat Format>
__device__ __forceinline__ void decode_value_run(const ValueRun<Format>& run, float (&values)[8]) {
    if constexpr (Format == QsaKvFormat::BF16) {
        unpack_bf16x8(run.words, values);
    } else {
        const float scale = decode_nvfp4_e4m3(static_cast<std::uint8_t>(run.scale));
#pragma unroll
        for (int byte = 0; byte < 4; ++byte) {
            const float2 pair =
                decode_nvfp4_e2m1x2(static_cast<std::uint8_t>(run.codes >> (8 * byte)));
            values[2 * byte]     = pair.x * scale;
            values[2 * byte + 1] = pair.y * scale;
        }
    }
}

// Grid (splits, 2 KV heads, W*B), block 128 (4 warps). The CTA owns selected ranks
// [split * split_slots, min(count, (split + 1) * split_slots)) of packed column blockIdx.z and KV
// head g = blockIdx.y; its MMA rows are query heads 12g + r (r < 12; rows 12..15 are zero). Lane
// (g' = lane/4, c = lane%4) holds accumulator rows g' and g'+8 throughout. Each listed logical
// slot is translated once, when its tile is staged, to its physical K/V row in the column's lane.
//
// Shared memory (AttentionShared, about 23 KiB): the query rows in the permuted logit order (after
// the rank loop, the output rows [16][258] FP32), the tile's weights p = hi + lo as BF16 [16][72]
// each, per-warp row maxima and sums [4][16], and the listed slots of two tiles [2][64] (-1 for an
// unlisted, out-of-range, or unmapped rank; otherwise the physical row (page * 2 + g) * 64 + slot %
// 64). Per tile of 64 ranks:
//  0. Threads 0..63 load the next tile's listed slots into registers, and every lane issues the
//     loads of its 16 value runs for step 4, so they overlap the logits.
//  1. Warp w loads the K rows of ranks 16w + n (n < 16) straight into B fragments (unlisted
//     slots are zero and masked) and forms S = Q K^T with 32 BF16 m16n8k16 MMAs (FP32
//     accumulation; products of BF16 values are exact), then the base-2 logits.
//  2. Row maxima over the warp's 16 ranks go to warp_max; barrier 1. Threads 0..63 then store
//     the next tile's slots into the other buffer, which barrier 2 publishes.
//  3. Every warp combines the four maxima into the same running maximum m, forms each of its
//     weights exactly once, p = exp2(logit - m), stores p as hi = BF16_rn(p) and lo = BF16_rn(p -
//     hi), and its row sums of p to warp_sum; barrier 2.
//  4. Every warp rescales its accumulator and denominator by exp2(m_old - m) and adds P V for its
//     64 features: the fetched eight-feature V runs are decoded, transposed into key-pair B
//     fragments in registers, and multiplied by the hi and lo weight fragments (ldmatrix), 64
//     MMAs.
// Barrier 1 of the next tile orders this tile's weight and slot reads before the next writes. After
// the loop a barrier retires the query rows and the output rows are staged through the same
// shared memory: with one split the normalized BF16 output (exact zero when nothing is listed),
// otherwise the unnormalized accumulators[t][h][split][256] and stats[t][h][split] = (m, l) in the
// base-2 domain for merge_kernel.
template <QsaKvFormat Format>
__global__ void __launch_bounds__(kAttentionThreads)
    attention_kernel(const std::uint16_t* __restrict__ q,
                     const std::int32_t* __restrict__ selected_ids,
                     const std::int32_t* __restrict__ selected_count, QsaColumns columns,
                     QsaStatePlanes state, int selected_slots, int split_slots,
                     std::uint16_t* __restrict__ out, float* __restrict__ accumulators,
                     float* __restrict__ stats) {
    __shared__ __align__(16) AttentionShared shared;
    const int split   = static_cast<int>(blockIdx.x);
    const int kv_head = static_cast<int>(blockIdx.y);
    const int token   = static_cast<int>(blockIdx.z);
    const int thread  = static_cast<int>(threadIdx.x);
    const int warp    = thread / kWarpSize;
    const int lane    = thread % kWarpSize;
    const int group   = lane / 4;
    const int quarter = lane % 4;

    const int lane_row = column_lane(columns, state, token);
    int count          = lane_row >= 0 ? selected_count[token] : 0;
    if (count < 0 || count > selected_slots) { count = 0; }
    const int begin         = split * split_slots;
    const int end           = min(count, begin + split_slots);
    const std::int32_t* ids = selected_ids + static_cast<std::int64_t>(token) * selected_slots;
    const int capacity      = kPageTokens * state.logical_pages;
    const std::uint16_t* query =
        q + (static_cast<std::int64_t>(token) * kQueryHeads + kv_head * kGroupHeads) * kHeadDim;

    for (int index = thread; index < kHeadRows * kHeadDim / 2; index += kAttentionThreads) {
        const int row     = index / (kHeadDim / 2);
        const int feature = 2 * (index % (kHeadDim / 2));
        const std::uint32_t pair =
            row < kGroupHeads
                ? *reinterpret_cast<const std::uint32_t*>(query + row * kHeadDim + feature)
                : 0U;
        *reinterpret_cast<std::uint32_t*>(&shared.q[row][logical_query_column(feature)]) = pair;
    }
    // The physical K/V row of rank `rank`'s listed slot, or -1 when it is unlisted, out of range,
    // or unmapped.
    const auto listed_slot = [&](int rank) {
        const int id = rank < end ? ids[rank] : -1;
        if (id < 0 || id >= capacity) { return -1; }
        const int page = slot_page(state, lane_row, id);
        return page >= 0 ? static_cast<int>(kv_row(page, kv_head, id % kPageTokens)) : -1;
    };
    if (thread < kKeyTile) { shared.tile_slots[0][thread] = listed_slot(begin + thread); }
    __syncthreads();

    // ldmatrix row addresses: lanes 8i..8i+7 address rows of matrix i (rows +8 for odd i, columns
    // +8 for i >= 2).
    const int matrix_row        = lane % 8 + 8 * ((lane / 8) % 2);
    const int matrix_column     = 8 * (lane / 16);
    const unsigned q_address    = smem_addr(&shared.q[matrix_row][matrix_column]);
    const unsigned p_hi_address = smem_addr(&shared.p_hi[matrix_row][matrix_column]);
    const unsigned p_lo_address = smem_addr(&shared.p_lo[matrix_row][matrix_column]);

    float accumulator[kValueTiles][4] = {};
    float maximum[2]                  = {-INFINITY, -INFINITY}; // rows group, group + 8
    float denominator[2]              = {0.0F, 0.0F};

    for (int tile = begin, buffer = 0; tile < end; tile += kKeyTile, buffer ^= 1) {
        const int* slots    = shared.tile_slots[buffer];
        const int next_slot = thread < kKeyTile ? listed_slot(tile + kKeyTile + thread) : -1;
        // The tile's value runs: lane (group, quarter) needs ranks 16*step + 2*quarter + {0, 1, 8,
        // 9} and features 64*warp + 8*group + i of value tile i (see the P V step below).
        const int feature = kWarpFeatures * warp + 8 * group;
        ValueRun<Format> runs[kKeyTile / 16][4];
#pragma unroll
        for (int step = 0; step < kKeyTile / 16; ++step) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int slot = slots[16 * step + 2 * quarter + (j % 2) + 8 * (j / 2)];
                runs[step][j]  = fetch_value_run<Format>(state, max(slot, 0), slot >= 0, feature);
            }
        }
        float logits[2][4];
#pragma unroll
        for (int n = 0; n < 2; ++n) {
            const int slot   = slots[kWarpKeys * warp + 8 * n + group];
            const bool valid = slot >= 0;
            const int id     = max(slot, 0);
            std::uint32_t b[kKeySteps][2];
            load_key_fragments<Format>(state, id, valid, quarter, b);
            float* d = logits[n];
            d[0] = d[1] = d[2] = d[3] = 0.0F;
#pragma unroll
            for (int step = 0; step < kKeySteps; ++step) {
                unsigned a0, a1, a2, a3;
                ldmatrix_x4(a0, a1, a2, a3, q_address + 32U * static_cast<unsigned>(step));
                mma_bf16(d[0], d[1], d[2], d[3], a0, a1, a2, a3, b[step][0], b[step][1]);
            }
        }
        // Lane (group, quarter) holds tile ranks 16*warp + 8n + 2*quarter + {0,1} of rows group and
        // group + 8.
        float row_maximum[2] = {-INFINITY, -INFINITY};
#pragma unroll
        for (int n = 0; n < 2; ++n) {
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const bool valid = slots[kWarpKeys * warp + 8 * n + 2 * quarter + e] >= 0;
                logits[n][e]     = valid ? logits[n][e] * kLogitScaleLog2 : -INFINITY;
                logits[n][e + 2] = valid ? logits[n][e + 2] * kLogitScaleLog2 : -INFINITY;
                row_maximum[0]   = fmaxf(row_maximum[0], logits[n][e]);
                row_maximum[1]   = fmaxf(row_maximum[1], logits[n][e + 2]);
            }
        }
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            row_maximum[r] =
                fmaxf(row_maximum[r], __shfl_xor_sync(kFullWarpMask, row_maximum[r], 1));
            row_maximum[r] =
                fmaxf(row_maximum[r], __shfl_xor_sync(kFullWarpMask, row_maximum[r], 2));
        }
        if (quarter == 0) {
            shared.warp_max[warp][group]     = row_maximum[0];
            shared.warp_max[warp][group + 8] = row_maximum[1];
        }
        __syncthreads();
        // Every warp has finished the previous tile, the last reader of the other buffer.
        if (thread < kKeyTile) { shared.tile_slots[buffer ^ 1][thread] = next_slot; }

        float correction[2];
        float next_maximum[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            float tile_maximum = shared.warp_max[0][group + 8 * r];
#pragma unroll
            for (int w = 1; w < kAttentionWarps; ++w) {
                tile_maximum = fmaxf(tile_maximum, shared.warp_max[w][group + 8 * r]);
            }
            next_maximum[r] = fmaxf(maximum[r], tile_maximum);
            // While nothing is listed the maximum stays -inf; subtracting 0 then keeps every weight
            // and correction an exact zero instead of NaN.
            const float base = next_maximum[r] == -INFINITY ? 0.0F : next_maximum[r];
            correction[r]    = exp2f(maximum[r] - base);
            float row_sum    = 0.0F;
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                const float p0 = exp2f(logits[n][2 * r] - base);
                const float p1 = exp2f(logits[n][2 * r + 1] - base);
                row_sum += p0 + p1;
                const __nv_bfloat162 hi = __floats2bfloat162_rn(p0, p1);
                const float2 rounded    = __bfloat1622float2(hi);
                const int column        = kWarpKeys * warp + 8 * n + 2 * quarter;
                *reinterpret_cast<__nv_bfloat162*>(&shared.p_hi[group + 8 * r][column]) = hi;
                *reinterpret_cast<std::uint32_t*>(&shared.p_lo[group + 8 * r][column]) =
                    bf16x2_bits(p0 - rounded.x, p1 - rounded.y);
            }
            row_sum += __shfl_xor_sync(kFullWarpMask, row_sum, 1);
            row_sum += __shfl_xor_sync(kFullWarpMask, row_sum, 2);
            if (quarter == 0) { shared.warp_sum[warp][group + 8 * r] = row_sum; }
        }
        __syncthreads();

#pragma unroll
        for (int r = 0; r < 2; ++r) {
            float tile_sum = 0.0F;
#pragma unroll
            for (int w = 0; w < kAttentionWarps; ++w) {
                tile_sum += shared.warp_sum[w][group + 8 * r];
            }
            denominator[r] = denominator[r] * correction[r] + tile_sum;
            maximum[r]     = next_maximum[r];
#pragma unroll
            for (int i = 0; i < kValueTiles; ++i) {
                accumulator[i][2 * r] *= correction[r];
                accumulator[i][2 * r + 1] *= correction[r];
            }
        }

        // P V over the tile's 64 ranks in k-steps of 16 with the fetched value runs.
#pragma unroll
        for (int step = 0; step < kKeyTile / 16; ++step) {
            float values[4][8];
#pragma unroll
            for (int j = 0; j < 4; ++j) { decode_value_run<Format>(runs[step][j], values[j]); }
            unsigned hi0, hi1, hi2, hi3, lo0, lo1, lo2, lo3;
            ldmatrix_x4(hi0, hi1, hi2, hi3, p_hi_address + 32U * static_cast<unsigned>(step));
            ldmatrix_x4(lo0, lo1, lo2, lo3, p_lo_address + 32U * static_cast<unsigned>(step));
#pragma unroll
            for (int i = 0; i < kValueTiles; ++i) {
                const std::uint32_t b0 = bf16x2_bits(values[0][i], values[1][i]);
                const std::uint32_t b1 = bf16x2_bits(values[2][i], values[3][i]);
                float* d               = accumulator[i];
                mma_bf16(d[0], d[1], d[2], d[3], hi0, hi1, hi2, hi3, b0, b1);
                mma_bf16(d[0], d[1], d[2], d[3], lo0, lo1, lo2, lo3, b0, b1);
            }
        }
    }

    // Every warp is past its last query ldmatrix before the output rows overwrite them.
    __syncthreads();
    const bool single = gridDim.x == 1;
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        // The denominator is zero only when nothing is listed, and then every accumulator is zero.
        const float scale = single ? (denominator[r] > 0.0F ? 1.0F / denominator[r] : 0.0F) : 1.0F;
#pragma unroll
        for (int i = 0; i < kValueTiles; ++i) {
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                shared.out[group + 8 * r][kWarpFeatures * warp + 8 * (2 * quarter + e) + i] =
                    accumulator[i][2 * r + e] * scale;
            }
        }
    }
    __syncthreads();

    const std::int64_t first_head =
        static_cast<std::int64_t>(token) * kQueryHeads + kv_head * kGroupHeads;
    if (single) {
        for (int index = thread; index < kGroupHeads * kHeadDim / 2; index += kAttentionThreads) {
            const int row     = index / (kHeadDim / 2);
            const int feature = 2 * (index % (kHeadDim / 2));
            *reinterpret_cast<std::uint32_t*>(out + (first_head + row) * kHeadDim + feature) =
                bf16x2_bits(shared.out[row][feature], shared.out[row][feature + 1]);
        }
        return;
    }
    const int splits = static_cast<int>(gridDim.x);
    for (int index = thread; index < kGroupHeads * kHeadDim / 2; index += kAttentionThreads) {
        const int row           = index / (kHeadDim / 2);
        const int feature       = 2 * (index % (kHeadDim / 2));
        const std::int64_t part = (first_head + row) * splits + split;
        *reinterpret_cast<float2*>(accumulators + part * kHeadDim + feature) =
            *reinterpret_cast<const float2*>(&shared.out[row][feature]);
    }
    if (warp == 0 && quarter == 0) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int row = group + 8 * r;
            if (row < kGroupHeads) {
                const std::int64_t part = (first_head + row) * splits + split;
                stats[2 * part]         = maximum[r];
                stats[2 * part + 1]     = denominator[r];
            }
        }
    }
}

constexpr int kMergeThreads = kHeadDim;
static_assert(qsa_attention_splits(1, qsa_selected_capacity(QsaSelectBudget::Blocks2048)) <=
              kMergeThreads);

// Grid (24 query heads, W), block 256: merges one query head's split partial softmax states. With M
// the largest split maximum (base 2), split s weighs w_s = 2^(m_s - M); thread s < splits stages
// (m_s, l_s) (maxima, sums [256]), then forms w_s once into weights [256], and thread d writes
// feature d: out = sum_s w_s acc_s[d] / sum_s w_s l_s, its independent accumulator loads unrolled
// eight deep. A split without listed slots has m = -inf and weight zero; when every split is empty
// the output is exact zero. Barriers: stats staged, weights published.
__global__ void __launch_bounds__(kMergeThreads)
    merge_kernel(const float* __restrict__ accumulators, const float* __restrict__ stats,
                 int splits, std::uint16_t* __restrict__ out) {
    __shared__ float maxima[kMergeThreads];
    __shared__ float sums[kMergeThreads];
    __shared__ float weights[kMergeThreads];
    const int head   = static_cast<int>(blockIdx.x);
    const int token  = static_cast<int>(blockIdx.y);
    const int d      = static_cast<int>(threadIdx.x);
    const auto first = (static_cast<std::int64_t>(token) * kQueryHeads + head) * splits;
    if (d < splits) {
        const float2 stat = *reinterpret_cast<const float2*>(stats + 2 * (first + d));
        maxima[d]         = stat.x;
        sums[d]           = stat.y;
    }
    __syncthreads();
    float maximum = -INFINITY;
    for (int split = 0; split < splits; ++split) { maximum = fmaxf(maximum, maxima[split]); }
    if (d < splits) { weights[d] = maximum == -INFINITY ? 0.0F : exp2f(maxima[d] - maximum); }
    __syncthreads();
    float value       = 0.0F;
    float denominator = 0.0F;
#pragma unroll 8
    for (int split = 0; split < splits; ++split) {
        const float weight = weights[split];
        denominator        = fmaf(weight, sums[split], denominator);
        value              = fmaf(weight, accumulators[(first + split) * kHeadDim + d], value);
    }
    out[(static_cast<std::int64_t>(token) * kQueryHeads + head) * kHeadDim + d] =
        __bfloat16_as_ushort(__float2bfloat16_rn(denominator > 0.0F ? value / denominator : 0.0F));
}

template <QsaKvFormat Format>
void launch_attention(const std::uint16_t* q, const std::int32_t* selected_ids,
                      const std::int32_t* selected_count, const QsaColumns& columns,
                      const QsaStatePlanes& state, std::int32_t selected_slots, std::uint16_t* out,
                      const QsaAttentionWorkspace& workspace, cudaStream_t stream) {
    const std::int32_t width       = columns.width * columns.rows;
    const std::int32_t split_slots = qsa_attention_split_slots(width, selected_slots);
    const std::int32_t splits      = qsa_attention_splits(width, selected_slots);
    attention_kernel<Format><<<dim3(splits, kKvHeads, width), kAttentionThreads, 0, stream>>>(
        q, selected_ids, selected_count, columns, state, selected_slots, split_slots, out,
        workspace.accumulators, workspace.stats);
    CUDA_CHECK(cudaGetLastError());
    if (splits > 1) {
        merge_kernel<<<dim3(kQueryHeads, width), kMergeThreads, 0, stream>>>(
            workspace.accumulators, workspace.stats, splits, out);
        CUDA_CHECK(cudaGetLastError());
    }
}

// Launches select_kernel for one query chunk: clusters of kSelectClusterCtas CTAs per query for
// short chunks of long rows, one CTA per query otherwise.
void launch_select(const float* scores, const QsaColumns& columns, const QsaStatePlanes& state,
                   const SelectChunk& chunk, int blocks, int budget, int selected_slots,
                   int max_visible_tokens, std::int32_t* selected_ids, std::int32_t* selected_count,
                   cudaStream_t stream) {
    const int chunk_queries = chunk.row_count * chunk.column_count;
    if (chunk_queries > kSelectClusterQueries || blocks <= kSelectTile) {
        select_kernel<1><<<chunk_queries, kSelectThreads, 0, stream>>>(
            scores, columns, state, chunk, blocks, budget, selected_slots, max_visible_tokens,
            selected_ids, selected_count);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    cudaLaunchConfig_t config{};
    config.gridDim  = dim3(static_cast<unsigned>(chunk_queries * kSelectClusterCtas));
    config.blockDim = dim3(kSelectThreads);
    config.stream   = stream;
    cudaLaunchAttribute cluster{};
    cluster.id               = cudaLaunchAttributeClusterDimension;
    cluster.val.clusterDim.x = kSelectClusterCtas;
    cluster.val.clusterDim.y = 1;
    cluster.val.clusterDim.z = 1;
    config.attrs             = &cluster;
    config.numAttrs          = 1;
    CUDA_CHECK(cudaLaunchKernelEx(&config, select_kernel<kSelectClusterCtas>, scores, columns,
                                  state, chunk, blocks, budget, selected_slots, max_visible_tokens,
                                  selected_ids, selected_count));
}

// The selection chunks of a call: runs of whole rows while a row fits the score workspace
// (at most `chunk_queries` queries per chunk), otherwise column ranges of one row whose starts are
// multiples of the tiled kernel's 32 columns. Either way a row's columns keep the score-group and
// tile positions of a single-row call, so a query's scores do not depend on the other rows.
template <class Visit>
void for_each_select_chunk(const QsaColumns& columns, std::int32_t chunk_queries, Visit&& visit) {
    if (columns.width <= chunk_queries) {
        const std::int32_t rows_per_chunk = chunk_queries / columns.width;
        for (std::int32_t row = 0; row < columns.rows; row += rows_per_chunk) {
            visit(SelectChunk{row, std::min(rows_per_chunk, columns.rows - row), 0, columns.width});
        }
        return;
    }
    const std::int32_t segment = chunk_queries >= kTiledQueries
                                     ? chunk_queries / kTiledQueries * kTiledQueries
                                     : chunk_queries;
    for (std::int32_t row = 0; row < columns.rows; ++row) {
        for (std::int32_t column = 0; column < columns.width; column += segment) {
            visit(SelectChunk{row, 1, column, std::min(segment, columns.width - column)});
        }
    }
}

} // namespace

void qsa_state_append_launch(QsaKvFormat format, const std::uint16_t* k, const std::uint16_t* v,
                             const std::uint16_t* index_keys, const std::int32_t* positions,
                             const QsaColumns& columns, const std::uint16_t* key_norm_weight,
                             const QsaStatePlanes& state, cudaStream_t stream) {
    const std::int32_t tokens = columns.width * columns.rows;
    if (format == QsaKvFormat::BF16) {
        append_kernel<QsaKvFormat::BF16>
            <<<tokens, kAppendThreads, 0, stream>>>(k, v, index_keys, positions, columns, state);
    } else {
        append_kernel<QsaKvFormat::NVFP4G16>
            <<<tokens, kAppendThreads, 0, stream>>>(k, v, index_keys, positions, columns, state);
    }
    CUDA_CHECK(cudaGetLastError());
    // Sealing reads raw keys other columns of this call wrote, so it follows the append in stream
    // order.
    seal_kernel<<<(tokens + kSealWarps - 1) / kSealWarps, kSealWarps * kWarpSize, 0, stream>>>(
        columns, key_norm_weight, tokens, state);
    CUDA_CHECK(cudaGetLastError());
}

void qsa_index_select_launch(const std::uint16_t* index_query, const QsaStatePlanes& state,
                             const QsaColumns& columns, const std::uint16_t* query_norm_weight,
                             std::int32_t budget_blocks, std::int32_t max_visible_tokens,
                             std::int32_t* selected_ids, std::int32_t* selected_count,
                             const QsaSelectWorkspace& workspace, cudaStream_t stream) {
    const std::int32_t queries        = columns.width * columns.rows;
    const std::int32_t blocks         = max_visible_tokens / kBlockTokens;
    const std::int32_t selected_slots = kBlockTokens * budget_blocks + kBlockTokens - 1;
    const bool ranks                  = blocks > budget_blocks;
    if (ranks) {
        index_query_kernel<<<queries, kQueryThreads, 0, stream>>>(
            index_query, columns, query_norm_weight, state, max_visible_tokens, workspace.query);
        CUDA_CHECK(cudaGetLastError());
    }
    const std::int32_t chunk_queries = ranks ? qsa_score_chunk_queries(queries, blocks) : queries;
    for_each_select_chunk(columns, chunk_queries, [&](const SelectChunk& chunk) {
        if (ranks && chunk.column_count >= kTiledQueries) {
            const int tiles = (chunk.column_count + kTiledQueries - 1) / kTiledQueries;
            tiled_score_kernel<<<dim3((blocks + kTiledBlocks - 1) / kTiledBlocks,
                                      static_cast<unsigned>(chunk.row_count * tiles)),
                                 kTiledThreads, 0, stream>>>(
                workspace.query, state.block_keys, state, columns, chunk, blocks, budget_blocks,
                max_visible_tokens, workspace.scores);
            CUDA_CHECK(cudaGetLastError());
        } else if (ranks) {
            // A power of two, so a warp's blocks span at most two whole-aligned pages.
            const int wanted = std::clamp((blocks + kStreamTargetCtas * kStreamWarps * 4 - 1) /
                                              (kStreamTargetCtas * kStreamWarps * 4),
                                          1, kStreamMaxSteps);
            int steps        = 1;
            while (steps < wanted) { steps *= 2; }
            const int cta_blocks = kStreamWarps * 4 * steps;
            const int groups     = (chunk.column_count + kStreamQueries - 1) / kStreamQueries;
            stream_score_kernel<<<dim3((blocks + cta_blocks - 1) / cta_blocks,
                                       static_cast<unsigned>(chunk.row_count * groups)),
                                  kStreamWarps * kWarpSize, 0, stream>>>(
                workspace.query, state.block_keys, state, columns, chunk, blocks, budget_blocks,
                max_visible_tokens, steps, workspace.scores);
            CUDA_CHECK(cudaGetLastError());
        }
        launch_select(workspace.scores, columns, state, chunk, blocks, budget_blocks,
                      selected_slots, max_visible_tokens, selected_ids, selected_count, stream);
    });
}

void qsa_selected_attention_launch(const std::uint16_t* q, const std::int32_t* selected_ids,
                                   const std::int32_t* selected_count, const QsaColumns& columns,
                                   QsaKvFormat format, const QsaStatePlanes& state,
                                   std::int32_t selected_slots, std::uint16_t* out,
                                   const QsaAttentionWorkspace& workspace, cudaStream_t stream) {
    if (format == QsaKvFormat::BF16) {
        launch_attention<QsaKvFormat::BF16>(q, selected_ids, selected_count, columns, state,
                                            selected_slots, out, workspace, stream);
    } else {
        launch_attention<QsaKvFormat::NVFP4G16>(q, selected_ids, selected_count, columns, state,
                                                selected_slots, out, workspace, stream);
    }
}

} // namespace ninfer::ops::detail
