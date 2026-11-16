#include "ops/launcher/hyper_connection.h"

#include "core/device.h"
#include "core/pdl.cuh"
#include "ninfer/ops/hyper_connection.h"
#include "ops/common/math.cuh"
#include "ops/common/mma.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/fp8/fp8_a16_codec.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kHidden    = kHyperConnectionHidden;
constexpr int kStreams   = kHyperConnectionStreams;
constexpr int kFlat      = kHyperConnectionFlat;
constexpr int kRank      = kHyperConnectionRank;
constexpr int kRows      = kHyperConnectionProjectedRows;
constexpr int kSplits    = kHyperConnectionKSplits;
constexpr int kTokenTile = kHyperConnectionTokenTile;
constexpr int kThreads   = 256;
constexpr int kWarps     = kThreads / kWarpSize;

// One 16-byte word holds 16 E4M3FN codes or 8 BF16 values.
constexpr int kCodesPerWord = 16;
constexpr int kBf16PerWord  = 8;
static_assert(kRows == kRank + kStreams);
// Split s is exactly residual stream s, so its partial dot products share stream s's inverse RMS.
static_assert(kSplits == kStreams && kFlat / kSplits == kHidden);
static_assert(kTokenTile <= kWarpSize);

// Down/write projection: a warp-wide step covers kChunk inputs, one 16-code word per lane.
constexpr int kChunk  = kWarpSize * kCodesPerWord;
constexpr int kChunks = kHidden / kChunk;
static_assert(kHidden % kChunk == 0);
// Activation staging: each tile token's 2560 split inputs are kSplitWords BF16 words, staged raw
// in shared memory once, together with the split's norm words.
constexpr int kSplitWords = kHidden / kBf16PerWord;
static_assert(kTokenTile <= kWarps);
// Prefetched weight words per lane: a down warp holds kRowsPerWarp <= 2 rows of kChunks code
// words, a write warp one BF16 row of 2 * kChunks words.
constexpr int kPrefetchWords = 2 * kChunks;
// The write rows are projected by warps 0..3 of one extra row group.
static_assert(kStreams <= kWarps);

// Up projection: four lanes cover one 320-code row with kUpWordsPerLane words each, so a warp
// pass projects kUpRowsPerPass rows. Two warps serve each stream.
constexpr int kUpLanesPerRow  = 4;
constexpr int kUpWordsPerLane = kRank / (kCodesPerWord * kUpLanesPerRow);
constexpr int kUpRowsPerPass  = kWarpSize / kUpLanesPerRow;
constexpr int kWarpsPerStream = kWarps / kStreams;
static_assert(kRank % (kCodesPerWord * kUpLanesPerRow) == 0 && kWarps % kStreams == 0);

// Route selection. A single token tile streams every down row once with one row per warp (164
// CTAs); wider launches keep two rows per warp so each staged activation serves both rows. The up
// projection keeps 16 hidden indices per CTA (160 CTAs per tile) through kUpNarrowMaxTokens and
// otherwise 64, which amortizes the per-CTA gate prologue over four times the weight bytes.
constexpr int kDownNarrowMaxTokens = kTokenTile;
constexpr int kUpNarrowMaxTokens   = 4 * kTokenTile;
constexpr int kUpNarrowHidden      = 16;
constexpr int kUpWideHidden        = 64;

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

// Bytes of the up code plane [10240,320]; a multiple of 16.
constexpr int kUpCodeBytes = kFlat * kRank;
static_assert(kUpCodeBytes % 16 == 0);

// Asynchronous bulk prefetch of [address, address + bytes) into L2 (no shared memory, no
// completion tracking). address is 16-byte aligned and bytes a positive multiple of 16.
__device__ __forceinline__ void prefetch_l2(const void* address, int bytes) {
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(address), "r"(bytes)
                 : "memory");
}

// Exact decode of 16 E4M3FN codes (byte b of the word is code b): every E4M3FN value, subnormals
// and signed zeros included, is representable in FP16 and FP32.
__device__ __forceinline__ void decode_e4m3x16(const uint4& packed, float* out) {
    const std::uint32_t words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const __half2_raw pair = __nv_cvt_fp8x2_to_halfraw2(
                static_cast<__nv_fp8x2_storage_t>(words[i] >> (16 * half)), __NV_E4M3);
            const float2 values       = __half22float2(__half2(pair));
            out[4 * i + 2 * half]     = values.x;
            out[4 * i + 2 * half + 1] = values.y;
        }
    }
}

// Shared position of BF16 word w = 2l + h (inputs 8w .. 8w+7) of chunk c within one staged
// 2560-input row: (2c + h) * 32 + l. Lane l's two words of a chunk are then the l-th 16-byte slot
// of two 512-byte runs, so a warp's loads are free of bank conflicts.
__device__ __forceinline__ int staged_word(int word) {
    const int chunk = word / (kChunk / kBf16PerWord);
    const int local = word % (kChunk / kBf16PerWord);
    return (2 * chunk + local % 2) * kWarpSize + local / 2;
}

// Raw split activations and norm of one CTA: residual words [token][kSplitWords] and norm words
// [kSplitWords], both in staged_word order. 46080 bytes.
struct DownStage {
    uint4 residual[kTokenTile][kSplitWords];
    uint4 norm[kSplitWords];
};

// weights[row][i] *= (1 + norm) for the lane's 16 inputs of one chunk; folding the gain into the
// weights once serves every token.
template <int kRows, int kAccRows>
__device__ __forceinline__ void apply_gain(const DownStage& stage, int chunk, int lane,
                                           float (&weights)[kAccRows][kCodesPerWord]) {
    float gain[kCodesPerWord];
    unpack_bf16x8(stage.norm[2 * chunk * kWarpSize + lane], gain);
    unpack_bf16x8(stage.norm[(2 * chunk + 1) * kWarpSize + lane], gain + kBf16PerWord);
#pragma unroll
    for (int row = 0; row < kRows; ++row) {
#pragma unroll
        for (int i = 0; i < kCodesPerWord; ++i) {
            weights[row][i] = fmaf(weights[row][i], gain[i], weights[row][i]);
        }
    }
}

// acc[row][token] += sum over the lane's 16 inputs of weights[row][i] * R[token][i] for the tile's
// tokens; kAccRows >= kRows lets a one-row write warp share a two-row accumulator.
template <int kRows, int kAccRows>
__device__ __forceinline__ void
accumulate_chunk(const float (&weights)[kAccRows][kCodesPerWord], const DownStage& stage, int chunk,
                 int lane, int tile_tokens, float (&acc)[kAccRows][kTokenTile]) {
    static_assert(kRows <= kAccRows);
#pragma unroll
    for (int token = 0; token < kTokenTile; ++token) {
        if (token >= tile_tokens) { break; }
        float a[kCodesPerWord];
        unpack_bf16x8(stage.residual[token][2 * chunk * kWarpSize + lane], a);
        unpack_bf16x8(stage.residual[token][(2 * chunk + 1) * kWarpSize + lane], a + kBf16PerWord);
#pragma unroll
        for (int row = 0; row < kRows; ++row) {
            float sum = acc[row][token];
#pragma unroll
            for (int i = 0; i < kCodesPerWord; ++i) { sum = fmaf(weights[row][i], a[i], sum); }
            acc[row][token] = sum;
        }
    }
}

// Grid (kRank / (8 kRowsPerWarp) down row groups + 1 write group when write is present, 4 splits,
// ceil(T/8) token tiles), block 256 (8 warps), 46080 bytes of dynamic shared memory (DownStage).
// Split s reduces the inputs of residual stream s.
//
// Warp w of a down group owns rows 8 kRowsPerWarp blockIdx.x + kRowsPerWarp w + r; warp w < 4 of
// the write group owns write row w (its other warps only stage). Before waiting on the producer of
// residual, every lane loads all of its rows' weight words for the split (one 16-code word per row
// and chunk; a BF16 write row needs two), so the whole matrix is in flight at once, and the CTA
// triggers the dependent up kernel's launch. The norm words are copied with cp.async at the same
// time.
//
// After the wait, the tile's residual rows of the split are copied raw with cp.async (one round
// trip); a cp.async wait and one barrier publish the stage. Lane l then multiplies inputs
// 16l .. 16l+15 of each of the five 512-input chunks: it decodes its code words, folds (1 + norm)
// into them, and accumulates every tile token, with no further barrier. FP8 rows accumulate the
// decoded codes, so a row's multiplier is applied once to its finished sum. Lane sums are
// warp-reduced and lane 0 writes partials[s][t][row] = multiplier * sum_k W[row,k] R[t,k]
// (1 + norm[k]); the inverse RMS is applied by the up kernel. These unnormalized FP32 sums require
// the |R (1 + norm) W| dot products to stay within FP32 range, as the stream sums of squares
// already must. In CTAs of row group 0, warp t < T_tile also sums R[t,k]^2 over the stream from the
// stage and publishes square_sums[s][t].
template <int kRowsPerWarp>
__global__ void __launch_bounds__(kThreads)
    down_write_kernel(const std::uint16_t* __restrict__ residual,
                      const uint4* __restrict__ norm_weight, const uint4* __restrict__ down_codes,
                      const std::uint16_t* __restrict__ down_scales,
                      const uint4* __restrict__ write_weight, const std::uint8_t* up_codes,
                      float* __restrict__ partials, float* __restrict__ square_sums, int tokens) {
    static_assert(kRowsPerWarp == 1 || kRowsPerWarp == 2);
    static_assert(kRank % (kWarps * kRowsPerWarp) == 0);
    constexpr int kDownGroups = kRank / (kWarps * kRowsPerWarp);
    extern __shared__ uint4 dynamic_shared[];
    DownStage& stage = *reinterpret_cast<DownStage*>(dynamic_shared);

    const int warp        = static_cast<int>(threadIdx.x) / kWarpSize;
    const int lane        = static_cast<int>(threadIdx.x) % kWarpSize;
    const int split       = static_cast<int>(blockIdx.y);
    const int token_begin = static_cast<int>(blockIdx.z) * kTokenTile;
    const int tile_tokens = min(kTokenTile, tokens - token_begin);
    const bool down_warp  = static_cast<int>(blockIdx.x) < kDownGroups;
    const bool write_warp = !down_warp && warp < kStreams;
    const int first_row =
        down_warp ? static_cast<int>(blockIdx.x) * kWarps * kRowsPerWarp + warp * kRowsPerWarp
                  : kRank + warp;
    const int split_input = split * kHidden;

    uint4 words[kPrefetchWords];
    float multiplier[kRowsPerWarp];
    if (down_warp) {
#pragma unroll
        for (int row = 0; row < kRowsPerWarp; ++row) {
            const uint4* codes =
                down_codes + ((first_row + row) * kFlat + split_input) / kCodesPerWord;
#pragma unroll
            for (int chunk = 0; chunk < kChunks; ++chunk) {
                words[row * kChunks + chunk] = codes[chunk * kWarpSize + lane];
            }
            multiplier[row] = bf16_bits_to_float(down_scales[first_row + row]);
        }
    } else if (write_warp) {
        const uint4* values = write_weight + (warp * kFlat + split_input) / kBf16PerWord;
#pragma unroll
        for (int chunk = 0; chunk < kChunks; ++chunk) {
            words[2 * chunk] = values[(chunk * kChunk + lane * kCodesPerWord) / kBf16PerWord];
            words[2 * chunk + 1] =
                values[(chunk * kChunk + lane * kCodesPerWord) / kBf16PerWord + 1];
        }
    }
    for (int word = static_cast<int>(threadIdx.x); word < kSplitWords; word += kThreads) {
        cp_async<16>(&stage.norm[staged_word(word)],
                     norm_weight + split_input / kBf16PerWord + word);
    }
    if (threadIdx.x == 0) {
        pdl::trigger_dependents();
        // The up kernel launches late after this trigger (measured: about 3.6 us after this grid
        // starts at T=1), so the first token tile's CTAs pull its code plane into L2 now and both
        // weight streams overlap; the up kernel then reads its codes from L2.
        if (blockIdx.z == 0) {
            const int cta    = static_cast<int>(blockIdx.y * gridDim.x + blockIdx.x);
            const int ctas   = static_cast<int>(gridDim.x * gridDim.y);
            const int slice  = (kUpCodeBytes / ctas + 15) / 16 * 16;
            const int offset = cta * slice;
            if (offset < kUpCodeBytes) {
                prefetch_l2(up_codes + offset, min(slice, kUpCodeBytes - offset));
            }
        }
    }
    pdl::wait_for_dependencies();

    const auto* residual_words = reinterpret_cast<const uint4*>(residual);
    for (int index = static_cast<int>(threadIdx.x); index < tile_tokens * kSplitWords;
         index += kThreads) {
        const int token = index / kSplitWords;
        const int word  = index % kSplitWords;
        cp_async<16, Cache::cg>(
            &stage.residual[token][staged_word(word)],
            residual_words +
                (static_cast<std::int64_t>(token_begin + token) * kFlat + split_input) /
                    kBf16PerWord +
                word);
    }
    cp_commit();
    cp_wait<0>();
    __syncthreads();

    float acc[kRowsPerWarp][kTokenTile] = {};
#pragma unroll
    for (int chunk = 0; chunk < kChunks; ++chunk) {
        float weights[kRowsPerWarp][kCodesPerWord];
        if (down_warp) {
#pragma unroll
            for (int row = 0; row < kRowsPerWarp; ++row) {
                decode_e4m3x16(words[row * kChunks + chunk], weights[row]);
            }
            apply_gain<kRowsPerWarp>(stage, chunk, lane, weights);
            accumulate_chunk<kRowsPerWarp>(weights, stage, chunk, lane, tile_tokens, acc);
        } else if (write_warp) {
            unpack_bf16x8(words[2 * chunk], weights[0]);
            unpack_bf16x8(words[2 * chunk + 1], weights[0] + kBf16PerWord);
            apply_gain<1>(stage, chunk, lane, weights);
            accumulate_chunk<1>(weights, stage, chunk, lane, tile_tokens, acc);
        }
    }

    if (down_warp || write_warp) {
        const int rows = down_warp ? kRowsPerWarp : 1;
#pragma unroll
        for (int row = 0; row < kRowsPerWarp; ++row) {
            if (row >= rows) { break; }
            const float scale = down_warp ? multiplier[row] : 1.0F;
#pragma unroll
            for (int token = 0; token < kTokenTile; ++token) {
                if (token >= tile_tokens) { break; }
                const float sum = warp_reduce_sum(acc[row][token]);
                if (lane == 0) {
                    partials[(static_cast<std::int64_t>(split) * tokens + token_begin + token) *
                                 kRows +
                             first_row + row] = sum * scale;
                }
            }
        }
    }

    if (blockIdx.x == 0 && warp < tile_tokens) {
        float sum = 0.0F;
        for (int word = lane; word < kSplitWords; word += kWarpSize) {
            float r[kBf16PerWord];
            unpack_bf16x8(stage.residual[warp][word], r);
#pragma unroll
            for (int i = 0; i < kBf16PerWord; ++i) { sum = fmaf(r[i], r[i], sum); }
        }
        sum = warp_reduce_sum(sum);
        if (lane == 0) {
            square_sums[static_cast<std::int64_t>(split) * tokens + token_begin + warp] = sum;
        }
    }
}

// Grid (2560 / kHiddenPerCta hidden groups, ceil(T/8) token tiles), block 256 (8 warps).
//
// Warp w serves stream i = w / 2 and lane l row group g = l / 4: in pass p it projects up row
// i*2560 + d for hidden index d = kHiddenPerCta blockIdx.x + (w % 2) kHiddenPerCta / 2 + 8p + g.
// Lane l holds the row's code words 4s + j (s = 0..4, j = l % 4), i.e. inputs 16(4s+j) .. +15, so
// one warp load reads 64 consecutive bytes of each of eight consecutive rows. Pass 0's words are
// copied into shared memory (cp.async, 20 KiB) and its multiplier and the (1 + norm) gains of the
// thread's mix items are loaded before waiting on down_write; later passes load one pass ahead.
//
// After the wait every producer-dependent load of the prologue and the mix is issued at once (the
// thread's partials, the square sums, its mix residual values). Prologue, two barriers:
// inverse_rms[t][s] = rsqrt(square_sums[s][t] / 2560 + 1e-6), then u[t][r] = SiLU(sum_s
// inverse_rms[t][s] partials[s][t][r] / 4) for the tile's tokens. Input 16(4s+j) + 4q + e is stored
// as element e of u float4 (4s+q)*4 + j: for one (s, q) the four lanes of a row read 64 consecutive
// bytes, which the eight row groups share as a broadcast. CTA x = 0 also finishes write_scale = 2
// sigmoid(sum_s inverse_rms[t][s] partials[s][t][320 + i] / 4).
//
// Each pass accumulates the decoded codes against u, reduces over the row's four lanes, and lane
// j = 0 stores sigmoid(multiplier * sum) to gates[t][i][d]. After one barrier mix item
// (tid + 256m) of each thread forms x[t][d] = (1/4) sum_i gate_i * R[t,i,d] inverse_rms[t][i] (1 +
// norm[i*2560 + d]).
template <int kHiddenPerCta>
__global__ void __launch_bounds__(kThreads)
    up_mix_kernel(const std::uint16_t* __restrict__ residual,
                  const std::uint16_t* __restrict__ norm_weight, const uint4* __restrict__ up_codes,
                  const std::uint16_t* __restrict__ up_scales, const float* __restrict__ partials,
                  const float* __restrict__ square_sums, std::uint16_t* __restrict__ x,
                  float* __restrict__ write_scale, int tokens) {
    constexpr int kHiddenPerWarp = kHiddenPerCta / kWarpsPerStream;
    constexpr int kPasses        = kHiddenPerWarp / kUpRowsPerPass;
    static_assert(kHidden % kHiddenPerCta == 0 && kHiddenPerWarp % kUpRowsPerPass == 0);
    __shared__ float4 u[kTokenTile][kRank / 4];
    __shared__ float inverse_rms[kTokenTile][kStreams];
    __shared__ float gates[kTokenTile][kStreams][kHiddenPerCta];
    // Pass 0's code words, slot [word][tid]: staged with cp.async before the PDL wait.
    __shared__ uint4 first_pass_words[kUpWordsPerLane][kThreads];

    const int warp          = static_cast<int>(threadIdx.x) / kWarpSize;
    const int lane          = static_cast<int>(threadIdx.x) % kWarpSize;
    const int stream        = warp / kWarpsPerStream;
    const int part          = lane % kUpLanesPerRow;
    const int token_begin   = static_cast<int>(blockIdx.y) * kTokenTile;
    const int tile_tokens   = min(kTokenTile, tokens - token_begin);
    const int hidden_base   = static_cast<int>(blockIdx.x) * kHiddenPerCta;
    const auto hidden_local = [&](int pass) {
        return (warp % kWarpsPerStream) * kHiddenPerWarp + pass * kUpRowsPerPass +
               lane / kUpLanesPerRow;
    };
    const auto row_codes = [&](int pass) {
        return up_codes +
               (stream * kHidden + hidden_base + hidden_local(pass)) * (kRank / kCodesPerWord) +
               part;
    };
    const auto row_multiplier = [&](int pass) {
        return bf16_bits_to_float(up_scales[stream * kHidden + hidden_base + hidden_local(pass)]);
    };
    const auto load_row = [&](int pass, uint4(&words)[kUpWordsPerLane], float& multiplier) {
        const uint4* codes = row_codes(pass);
#pragma unroll
        for (int word = 0; word < kUpWordsPerLane; ++word) {
            words[word] = codes[word * kUpLanesPerRow];
        }
        multiplier = row_multiplier(pass);
    };

    // Mix item m of this thread: token (tid + 256m) / kHiddenPerCta, hidden index
    // (tid + 256m) % kHiddenPerCta of the CTA.
    constexpr int kMixItems = (kTokenTile * kHiddenPerCta + kThreads - 1) / kThreads;
    const auto mix_token    = [&](int item) {
        return (static_cast<int>(threadIdx.x) + item * kThreads) / kHiddenPerCta;
    };
    const auto mix_hidden = [&](int item) {
        return hidden_base + (static_cast<int>(threadIdx.x) + item * kThreads) % kHiddenPerCta;
    };

    // Register loads of invariant data may legally be sunk below griddepcontrol.wait (and are,
    // by NVVM and ptxas), which would serialize pass 0's DRAM round trips after the producer; the
    // asynchronous copies into shared memory are issued here, so they stream while down_write runs.
    {
        const uint4* codes = row_codes(0);
#pragma unroll
        for (int word = 0; word < kUpWordsPerLane; ++word) {
            cp_async<16>(&first_pass_words[word][threadIdx.x], codes + word * kUpLanesPerRow);
        }
        cp_commit();
    }
    float multiplier = row_multiplier(0);
    float gain[kMixItems][kStreams];
#pragma unroll
    for (int item = 0; item < kMixItems; ++item) {
        if (mix_token(item) < tile_tokens) {
#pragma unroll
            for (int i = 0; i < kStreams; ++i) {
                gain[item][i] =
                    1.0F + bf16_bits_to_float(norm_weight[i * kHidden + mix_hidden(item)]);
            }
        }
    }
    pdl::wait_for_dependencies();

    // Every producer-dependent load of the prologue and the mix is issued before the first
    // barrier: the stream square sums, this thread's partials, and its mix residual values.
    constexpr int kRankSteps = kTokenTile * kRank / kThreads;
    static_assert(kTokenTile * kRank % kThreads == 0);
    const auto partial = [&](int split, int token, int row) {
        return __ldcg(partials +
                      (static_cast<std::int64_t>(split) * tokens + token_begin + token) * kRows +
                      row);
    };
    float rank_partials[kRankSteps][kSplits];
#pragma unroll
    for (int step = 0; step < kRankSteps; ++step) {
        const int index = static_cast<int>(threadIdx.x) + step * kThreads;
        if (index / kRank < tile_tokens) {
#pragma unroll
            for (int split = 0; split < kSplits; ++split) {
                rank_partials[step][split] = partial(split, index / kRank, index % kRank);
            }
        }
    }
    // Thread tid < 32 owns (token tid / 4, stream or write row tid % 4).
    const int head_token = static_cast<int>(threadIdx.x) / kStreams;
    const int head_index = static_cast<int>(threadIdx.x) % kStreams;
    const bool head =
        static_cast<int>(threadIdx.x) < kTokenTile * kStreams && head_token < tile_tokens;
    const bool write_head   = head && write_scale != nullptr && blockIdx.x == 0;
    float stream_square_sum = 0.0F;
    float write_partials[kSplits];
    if (head) {
        stream_square_sum = __ldcg(square_sums + static_cast<std::int64_t>(head_index) * tokens +
                                   token_begin + head_token);
    }
    if (write_head) {
#pragma unroll
        for (int split = 0; split < kSplits; ++split) {
            write_partials[split] = partial(split, head_token, kRank + head_index);
        }
    }
    std::uint16_t mix_residual[kMixItems][kStreams];
#pragma unroll
    for (int item = 0; item < kMixItems; ++item) {
        if (mix_token(item) < tile_tokens) {
            const std::uint16_t* token_base =
                residual + static_cast<std::int64_t>(token_begin + mix_token(item)) * kFlat;
#pragma unroll
            for (int i = 0; i < kStreams; ++i) {
                mix_residual[item][i] = __ldcg(token_base + i * kHidden + mix_hidden(item));
            }
        }
    }

    if (head) {
        inverse_rms[head_token][head_index] =
            rsqrtf(stream_square_sum * (1.0F / static_cast<float>(kHidden)) + 1.0e-6F);
    }
    __syncthreads();
    float* const u_values = reinterpret_cast<float*>(u);
#pragma unroll
    for (int step = 0; step < kRankSteps; ++step) {
        const int index = static_cast<int>(threadIdx.x) + step * kThreads;
        const int token = index / kRank;
        const int rank  = index % kRank;
        if (token < tile_tokens) {
            float sum = 0.0F;
#pragma unroll
            for (int split = 0; split < kSplits; ++split) {
                sum = fmaf(inverse_rms[token][split], rank_partials[step][split], sum);
            }
            // rank = 16(4s+j) + 4q + e  ->  float (((4s+q)*4 + j)*4 + e).
            const int word = rank / (kCodesPerWord * kUpLanesPerRow);
            const int j    = (rank / kCodesPerWord) % kUpLanesPerRow;
            const int quad = (rank / 4) % 4;
            u_values[token * kRank + ((word * 4 + quad) * kUpLanesPerRow + j) * 4 + rank % 4] =
                silu(sum * 0.25F);
        }
    }
    if (write_head) {
        float sum = 0.0F;
#pragma unroll
        for (int split = 0; split < kSplits; ++split) {
            sum = fmaf(inverse_rms[head_token][split], write_partials[split], sum);
        }
        write_scale[static_cast<std::int64_t>(token_begin + head_token) * kStreams + head_index] =
            2.0F * sigmoid(sum * 0.25F);
    }
    // Each thread reads back only its own staged words, so completing its own copies suffices; the
    // barrier after the wait also keeps the compiler from hoisting those shared loads above it
    // (cp_wait is volatile asm without a memory clobber).
    cp_wait<0>();
    __syncthreads();

    uint4 words[kUpWordsPerLane];
#pragma unroll
    for (int word = 0; word < kUpWordsPerLane; ++word) {
        words[word] = first_pass_words[word][threadIdx.x];
    }
#pragma unroll
    for (int pass = 0; pass < kPasses; ++pass) {
        uint4 next_words[kUpWordsPerLane];
        float next_multiplier = 0.0F;
        if (pass + 1 < kPasses) { load_row(pass + 1, next_words, next_multiplier); }
        float acc[kTokenTile] = {};
#pragma unroll
        for (int word = 0; word < kUpWordsPerLane; ++word) {
            float weights[kCodesPerWord];
            decode_e4m3x16(words[word], weights);
#pragma unroll
            for (int token = 0; token < kTokenTile; ++token) {
                if (token >= tile_tokens) { break; }
#pragma unroll
                for (int quad = 0; quad < 4; ++quad) {
                    const float4 a = u[token][(word * 4 + quad) * kUpLanesPerRow + part];
                    float sum      = acc[token];
                    sum            = fmaf(weights[4 * quad], a.x, sum);
                    sum            = fmaf(weights[4 * quad + 1], a.y, sum);
                    sum            = fmaf(weights[4 * quad + 2], a.z, sum);
                    sum            = fmaf(weights[4 * quad + 3], a.w, sum);
                    acc[token]     = sum;
                }
            }
        }
#pragma unroll
        for (int token = 0; token < kTokenTile; ++token) {
            if (token >= tile_tokens) { break; }
            const float logit = warp_sum<kUpLanesPerRow>(acc[token]);
            if (part == 0) {
                gates[token][stream][hidden_local(pass)] = sigmoid(multiplier * logit);
            }
        }
        if (pass + 1 < kPasses) {
#pragma unroll
            for (int word = 0; word < kUpWordsPerLane; ++word) { words[word] = next_words[word]; }
            multiplier = next_multiplier;
        }
    }
    __syncthreads();

#pragma unroll
    for (int item = 0; item < kMixItems; ++item) {
        const int token = mix_token(item);
        if (token < tile_tokens) {
            const int local = mix_hidden(item) - hidden_base;
            float mixed     = 0.0F;
#pragma unroll
            for (int i = 0; i < kStreams; ++i) {
                const float normalized = bf16_bits_to_float(mix_residual[item][i]) *
                                         inverse_rms[token][i] * gain[item][i];
                mixed                  = fmaf(gates[token][i][local], normalized, mixed);
            }
            x[static_cast<std::int64_t>(token_begin + token) * kHidden + mix_hidden(item)] =
                __bfloat16_as_ushort(__float2bfloat16_rn(mixed * 0.25F));
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Prefill Tensor Core route (tokens >= kHyperConnectionMmaMinTokens).
//
// Both projections run as BF16 m16n8k16 MMAs with FP32 accumulation. E4M3 codes and BF16 write
// weights are exact BF16 operands; every FP32 operand v is entered as the BF16 pair
// hi = bf16(v), lo = bf16(v - hi) (|v - hi - lo| <= 2^-17 |v|), so each product is formed twice
// and both land in the same FP32 accumulator. Down: the weight operand is code * (1 + norm[k]) in
// FP32 split hi/lo and the activation the exact BF16 residual. Up: the weight operand is the exact
// code and the activation u split hi/lo. Row multipliers are applied once to finished sums. The
// split requires every operand below the BF16 maximum (about 3.4e38), which bounded model weights
// (|code| <= 448 times a BF16 multiplier, |1 + norm| small) and SiLU outputs satisfy. The down
// projection writes the SIMT route's split partials and square sums, mma_finish_kernel turns them
// into the gate input u (as a hi/lo BF16 pair), the inverse RMS and the write scales, and
// mma_up_kernel projects u and mixes.

// Down/write rows padded to whole 16-row MMA tiles: 320 down rows, 4 write rows, 12 zero rows.
constexpr int kMmaRows            = 336;
constexpr int kMmaBlockK          = 64;
constexpr int kMmaStreamKTiles    = kHidden / kMmaBlockK;
constexpr int kMmaDownBlockRows   = 48;
constexpr int kMmaDownBlockTokens = 128;
constexpr int kMmaDownThreads     = 256;
constexpr int kMmaDownWarpTokens  = kMmaDownBlockTokens / (kMmaDownThreads / kWarpSize);
constexpr int kMmaDownRowMmas     = kMmaDownBlockRows / 16;
constexpr int kMmaDownTokenMmas   = kMmaDownWarpTokens / 8;
static_assert(kMmaRows % kMmaDownBlockRows == 0 && kMmaRows >= kRows && kMmaRows % 16 == 0);
static_assert(kFlat % kMmaBlockK == 0 && kHidden % kMmaBlockK == 0 && kMmaBlockK == 64);
static_assert(kMmaDownWarpTokens % 8 == 0 && kMmaDownBlockTokens * 2 == kMmaDownThreads);

constexpr int kMmaUpHidden      = 32;
constexpr int kMmaUpBlockRows   = kStreams * kMmaUpHidden;
constexpr int kMmaUpBlockTokens = 64;
constexpr int kMmaUpThreads     = 256;
constexpr int kMmaUpWarpRows    = 32;
constexpr int kMmaUpWarpTokens  = 32;
constexpr int kMmaUpWarpsTokens = kMmaUpBlockTokens / kMmaUpWarpTokens;
constexpr int kMmaUpRowMmas     = kMmaUpWarpRows / 16;
constexpr int kMmaUpTokenMmas   = kMmaUpWarpTokens / 8;
constexpr int kMmaUpKTiles      = kRank / kMmaBlockK;
static_assert(kRank % kMmaBlockK == 0 && kHidden % kMmaUpHidden == 0);
static_assert((kMmaUpBlockRows / kMmaUpWarpRows) * kMmaUpWarpsTokens * kWarpSize == kMmaUpThreads);

__device__ __forceinline__ std::uint32_t bf16x2_bits(__nv_bfloat162 pair) {
    return *reinterpret_cast<const std::uint32_t*>(&pair);
}

// Splits v0, v1 into BF16 hi and lo pairs with hi + lo = v to about 2^-17 relative.
__device__ __forceinline__ void split_bf16x2(float v0, float v1, std::uint32_t& hi,
                                             std::uint32_t& lo) {
    const __nv_bfloat162 high = __floats2bfloat162_rn(v0, v1);
    const float2 rounded      = __bfloat1622float2(high);
    hi                        = bf16x2_bits(high);
    lo                        = bf16x2_bits(__floats2bfloat162_rn(v0 - rounded.x, v1 - rounded.y));
}

// Shared memory of mma_down_kernel (47.6 KiB, dynamic). The BF16 operand tiles are [row or
// token][64] with fp8_a16_shared_col_64 swizzling of their 16-byte groups, which keeps ldmatrix
// conflict-free.
struct MmaDownShared {
    std::uint16_t weight_hi[kMmaDownBlockRows * kMmaBlockK];
    std::uint16_t weight_lo[kMmaDownBlockRows * kMmaBlockK];
    std::uint16_t activation[2][kMmaDownBlockTokens * kMmaBlockK];
    std::uint8_t codes[kMmaDownBlockRows * kMmaBlockK];
    std::uint16_t write_rows[kStreams * kMmaBlockK];
    std::uint16_t norm[kMmaBlockK];
};

// Grid (336 / 48 row tiles, ceil(T / 128) token tiles, 4 splits), block 256 (8 warps), dynamic
// shared memory MmaDownShared. Split s reduces the 2560 inputs of residual stream s for 48
// projected rows (row tile 6 holds the write rows 320..323 and 12 zero rows) and 128 tokens; warp w
// owns tokens 16w .. 16w+15 of every row (3 x 2 m16n8 tiles). It produces the SIMT route's
// workspace: partials[s][t][row] = multiplier * sum_k W[row,k] R[t,k] (1 + norm[k]) and, in row
// tile 0, square_sums[s][t] = sum_k R[t,k]^2.
//
// Per 64-input k tile, one cp.async group stages the codes, write-row and norm words (single
// buffer) and the residual tile (double buffer). After the group completes and a barrier, every
// thread widens 6 code pairs to w = code * (1 + norm) and stores their hi/lo BF16 pairs (row tile 0
// also sums its token's squared residual values: thread t takes token t / 2, half t % 2 of the
// 64 inputs); a second barrier publishes the weight tiles and frees the code buffer for the next
// tile's copies, issued before the 4 x 12 MMAs of this tile. The first barrier of the next tile
// retires this tile's operand reads.
__global__ void __launch_bounds__(kMmaDownThreads)
    mma_down_kernel(const std::uint16_t* __restrict__ residual,
                    const std::uint16_t* __restrict__ norm_weight,
                    const std::uint8_t* __restrict__ down_codes,
                    const std::uint16_t* __restrict__ down_scales,
                    const std::uint16_t* __restrict__ write_weight, float* __restrict__ partials,
                    float* __restrict__ square_sums, int tokens) {
    extern __shared__ uint4 dynamic_shared[];
    MmaDownShared& shared = *reinterpret_cast<MmaDownShared*>(dynamic_shared);

    const int tid         = static_cast<int>(threadIdx.x);
    const int warp        = tid / kWarpSize;
    const int lane        = tid % kWarpSize;
    const int gid         = lane >> 2;
    const int lid         = lane & 3;
    const int row_begin   = static_cast<int>(blockIdx.x) * kMmaDownBlockRows;
    const int token_begin = static_cast<int>(blockIdx.y) * kMmaDownBlockTokens;
    const int split       = static_cast<int>(blockIdx.z);
    const int split_input = split * kHidden;
    const int tile_tokens = min(kMmaDownBlockTokens, tokens - token_begin);
    const bool write_tile = write_weight != nullptr && row_begin + kMmaDownBlockRows > kRank;
    const bool squares    = blockIdx.x == 0;

    const auto stage_weights = [&](int k_tile) {
        const int k_begin = split_input + k_tile * kMmaBlockK;
        for (int item = tid; item < kMmaDownBlockRows * (kMmaBlockK / 16);
             item += kMmaDownThreads) {
            const int local = item / (kMmaBlockK / 16);
            const int chunk = item % (kMmaBlockK / 16);
            const int row   = row_begin + local;
            if (row < kRank) {
                cp_async<16>(&shared.codes[local * kMmaBlockK + chunk * 16],
                             down_codes + static_cast<std::int64_t>(row) * kFlat + k_begin +
                                 chunk * 16);
            }
        }
        if (write_tile && threadIdx.x < kStreams * (kMmaBlockK / 8)) {
            const unsigned row   = threadIdx.x / (kMmaBlockK / 8);
            const unsigned group = threadIdx.x % (kMmaBlockK / 8);
            cp_async<16>(&shared.write_rows[row * kMmaBlockK + group * 8],
                         write_weight + row * kFlat + k_begin + group * 8);
        }
        if (threadIdx.x < kMmaBlockK / 8) {
            cp_async<16>(&shared.norm[threadIdx.x * 8], norm_weight + k_begin + threadIdx.x * 8);
        }
    };
    const auto stage_activation = [&](int k_tile, int stage) {
        const int k_begin = split_input + k_tile * kMmaBlockK;
        for (int item = tid; item < kMmaDownBlockTokens * (kMmaBlockK / 8);
             item += kMmaDownThreads) {
            const int token  = item / (kMmaBlockK / 8);
            const int group  = item % (kMmaBlockK / 8);
            const bool valid = token < tile_tokens;
            cp_async_zfill<16, Cache::cg>(
                &shared.activation[stage]
                                  [token * kMmaBlockK + fp8_a16_shared_col_64(token, group * 8)],
                residual + static_cast<std::int64_t>(token_begin + (valid ? token : 0)) * kFlat +
                    k_begin + group * 8,
                valid ? 16 : 0);
        }
    };

    stage_weights(0);
    if (tid == 0) { pdl::trigger_dependents(); }
    pdl::wait_for_dependencies();
    stage_activation(0, 0);
    cp_commit();

    float acc[kMmaDownRowMmas][kMmaDownTokenMmas][4] = {};
    float square_sum                                 = 0.0F;
    const int sq_token                               = tid / 2;
    const int sq_half                                = tid % 2;
    const int a_row                                  = (lane & 7) + ((lane >> 3) & 1) * 8;
    const int a_col                                  = (lane >> 4) * 8;
    const int b_row                                  = lane & 7;
    const int b_col                                  = ((lane >> 3) & 1) * 8;

#pragma unroll 1
    for (int k_tile = 0; k_tile < kMmaStreamKTiles; ++k_tile) {
        const int stage = k_tile % 2;
        cp_wait<0>();
        __syncthreads();

        // Widen 48 x 64 weights: pair p = tid + 256 i covers row p / 32, inputs 2 (p % 32) and +1.
#pragma unroll
        for (int i = 0; i < kMmaDownBlockRows * kMmaBlockK / 2 / kMmaDownThreads; ++i) {
            const int pair  = tid + i * kMmaDownThreads;
            const int local = pair / (kMmaBlockK / 2);
            const int col   = (pair % (kMmaBlockK / 2)) * 2;
            const int row   = row_begin + local;
            float w0        = 0.0F;
            float w1        = 0.0F;
            if (row < kRank) {
                const __half2_raw codes =
                    __nv_cvt_fp8x2_to_halfraw2(*reinterpret_cast<const __nv_fp8x2_storage_t*>(
                                                   &shared.codes[local * kMmaBlockK + col]),
                                               __NV_E4M3);
                const float2 values = __half22float2(__half2(codes));
                w0                  = values.x;
                w1                  = values.y;
            } else if (write_tile && row < kRows) {
                w0 = bf16_bits_to_float(shared.write_rows[(row - kRank) * kMmaBlockK + col]);
                w1 = bf16_bits_to_float(shared.write_rows[(row - kRank) * kMmaBlockK + col + 1]);
            }
            std::uint32_t hi = 0;
            std::uint32_t lo = 0;
            split_bf16x2(fmaf(w0, bf16_bits_to_float(shared.norm[col]), w0),
                         fmaf(w1, bf16_bits_to_float(shared.norm[col + 1]), w1), hi, lo);
            const int position = local * kMmaBlockK + fp8_a16_shared_col_64(local, col);
            store_vec(&shared.weight_hi[position], hi);
            store_vec(&shared.weight_lo[position], lo);
        }
        if (squares) {
#pragma unroll
            for (int group = 0; group < 4; ++group) {
                const int col = sq_half * 32 + group * 8;
                float values[8];
                unpack_bf16x8(*reinterpret_cast<const uint4*>(
                                  &shared.activation[stage][sq_token * kMmaBlockK +
                                                            fp8_a16_shared_col_64(sq_token, col)]),
                              values);
#pragma unroll
                for (int e = 0; e < 8; ++e) { square_sum = fmaf(values[e], values[e], square_sum); }
            }
        }
        __syncthreads();

        if (k_tile + 1 < kMmaStreamKTiles) {
            stage_weights(k_tile + 1);
            stage_activation(k_tile + 1, stage ^ 1);
            cp_commit();
        }

#pragma unroll
        for (int k_step = 0; k_step < kMmaBlockK / 16; ++k_step) {
            unsigned a_hi[kMmaDownRowMmas][4];
            unsigned a_lo[kMmaDownRowMmas][4];
            unsigned b[kMmaDownTokenMmas][2];
#pragma unroll
            for (int mt = 0; mt < kMmaDownRowMmas; ++mt) {
                const int row = mt * 16 + a_row;
                const int position =
                    row * kMmaBlockK + fp8_a16_shared_col_64(row, k_step * 16 + a_col);
                ldmatrix_x4(a_hi[mt][0], a_hi[mt][1], a_hi[mt][2], a_hi[mt][3],
                            smem_addr(&shared.weight_hi[position]));
                ldmatrix_x4(a_lo[mt][0], a_lo[mt][1], a_lo[mt][2], a_lo[mt][3],
                            smem_addr(&shared.weight_lo[position]));
            }
#pragma unroll
            for (int nt = 0; nt < kMmaDownTokenMmas; ++nt) {
                const int token = warp * kMmaDownWarpTokens + nt * 8 + b_row;
                ldmatrix_x2(b[nt][0], b[nt][1],
                            smem_addr(&shared.activation[stage][token * kMmaBlockK +
                                                                fp8_a16_shared_col_64(
                                                                    token, k_step * 16 + b_col)]));
            }
#pragma unroll
            for (int mt = 0; mt < kMmaDownRowMmas; ++mt) {
#pragma unroll
                for (int nt = 0; nt < kMmaDownTokenMmas; ++nt) {
                    float* c = acc[mt][nt];
                    mma_bf16(c[0], c[1], c[2], c[3], a_hi[mt][0], a_hi[mt][1], a_hi[mt][2],
                             a_hi[mt][3], b[nt][0], b[nt][1]);
                    mma_bf16(c[0], c[1], c[2], c[3], a_lo[mt][0], a_lo[mt][1], a_lo[mt][2],
                             a_lo[mt][3], b[nt][0], b[nt][1]);
                }
            }
        }
    }

    if (squares) {
        const float sum = square_sum + __shfl_xor_sync(kFullWarpMask, square_sum, 1);
        if (sq_half == 0 && sq_token < tile_tokens) {
            square_sums[static_cast<std::int64_t>(split) * tokens + token_begin + sq_token] = sum;
        }
    }
#pragma unroll
    for (int mt = 0; mt < kMmaDownRowMmas; ++mt) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int row = row_begin + mt * 16 + gid + 8 * half;
            if (row >= kRank && !(write_tile && row < kRows)) { continue; }
            const float multiplier = row < kRank ? bf16_bits_to_float(down_scales[row]) : 1.0F;
#pragma unroll
            for (int nt = 0; nt < kMmaDownTokenMmas; ++nt) {
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int token = warp * kMmaDownWarpTokens + nt * 8 + 2 * lid + e;
                    if (token >= tile_tokens) { continue; }
                    partials[(static_cast<std::int64_t>(split) * tokens + token_begin + token) *
                                 kRows +
                             row] = multiplier * acc[mt][nt][2 * half + e];
                }
            }
        }
    }
}

// Grid (T), block 352 (11 warps). Finishes the Tensor Core route's gate input from the split
// partials: thread r < 320 of token t stores u = SiLU(sum_s inverse_rms[t][s] partials[s][t][r] /
// 4) as its hi/lo BF16 pair to u_hi/u_lo[t][r]; thread 320 + i stores write_scale[t][i] =
// 2 sigmoid(sum_s inverse_rms[t][s] partials[s][t][320 + i] / 4) when write is present; thread
// i < 4 stores inverse_rms[t][i] = rsqrt(square_sums[i][t] / 2560 + 1e-6) for the up kernel.
// No shared memory is used. Every CTA triggers at entry so the dependent up kernel can stage
// immutable weights before waiting for the finished activations.
constexpr int kMmaFinishThreads = 352;
static_assert(kMmaFinishThreads >= kRows && kMmaFinishThreads % kWarpSize == 0);

__global__ void __launch_bounds__(kMmaFinishThreads)
    mma_finish_kernel(const float* partials, const float* square_sums,
                      std::uint16_t* __restrict__ u_hi, std::uint16_t* __restrict__ u_lo,
                      float* __restrict__ inverse_rms_out, float* __restrict__ write_scale,
                      int tokens) {
    pdl::trigger_dependents();
    const int row        = static_cast<int>(threadIdx.x);
    const std::int64_t t = blockIdx.x;
    pdl::wait_for_dependencies();
    if (row >= kRows || (row >= kRank && write_scale == nullptr)) { return; }
    float pre = 0.0F;
#pragma unroll
    for (int split = 0; split < kSplits; ++split) {
        const float inverse =
            rsqrtf(__ldcg(square_sums + split * tokens + t) * (1.0F / static_cast<float>(kHidden)) +
                   1.0e-6F);
        if (row == split) { inverse_rms_out[t * kStreams + split] = inverse; }
        pre = fmaf(inverse, __ldcg(partials + (split * tokens + t) * kRows + row), pre);
    }
    if (row < kRank) {
        const float value        = silu(pre * 0.25F);
        const __nv_bfloat16 high = __float2bfloat16_rn(value);
        u_hi[t * kRank + row]    = __bfloat16_as_ushort(high);
        u_lo[t * kRank + row] =
            __bfloat16_as_ushort(__float2bfloat16_rn(value - __bfloat162float(high)));
    } else {
        write_scale[t * kStreams + row - kRank] = 2.0F * sigmoid(pre * 0.25F);
    }
}

// Shared memory of mma_up_kernel (56 KiB, dynamic): code tile, widened BF16 weight tile, and two
// stages of u_hi/u_lo tiles, [row or token][64] swizzled as in mma_down_kernel. After the main
// loop the u stages are reused for gates[token][128 local rows] (FP32, 32 KiB).
struct MmaUpShared {
    std::uint8_t codes[kMmaUpBlockRows * kMmaBlockK];
    std::uint16_t weight[kMmaUpBlockRows * kMmaBlockK];
    std::uint16_t u_hi[2][kMmaUpBlockTokens * kMmaBlockK];
    std::uint16_t u_lo[2][kMmaUpBlockTokens * kMmaBlockK];
};

static_assert(sizeof(float) * kMmaUpBlockTokens * kMmaUpBlockRows <=
              sizeof(MmaUpShared::u_hi) + sizeof(MmaUpShared::u_lo));

// Grid (2560 / 32 hidden groups, ceil(T / 64) token tiles), block 256 (8 warps: 4 row x 2 token),
// dynamic shared memory MmaUpShared. Local row m = 32 i + h of the CTA is up row
// i*2560 + 32 blockIdx.x + h, so the CTA holds all four gates of its 32 hidden indices; warp w owns
// local rows 32 (w / 2) .. +31 and tokens 32 (w % 2) .. +31 (2 x 4 m16n8 tiles).
//
// Five 64-input k tiles: one cp.async group per tile stages the codes (single buffer) and the
// u_hi/u_lo tiles (double buffer); the first tile's codes are copied before the PDL wait. After the
// group completes and a barrier, the codes are widened exactly to BF16; a second barrier publishes
// them and frees the code buffer for the next copies, issued before the 4 x 16 MMAs of the tile.
// Epilogue: gates[t][m] = sigmoid(multiplier * acc) to shared memory, a barrier, then thread
// (h = tid % 32, token tid / 32 + 8j) forms x[t][d] = (1/4) sum_i gates[t][32 i + h] *
// R[t,i,d] inverse_rms[t][i] (1 + norm[i*2560 + d]) for d = 32 blockIdx.x + h.
__global__ void __launch_bounds__(kMmaUpThreads)
    mma_up_kernel(const std::uint16_t* __restrict__ residual,
                  const std::uint16_t* __restrict__ norm_weight,
                  const std::uint8_t* __restrict__ up_codes,
                  const std::uint16_t* __restrict__ up_scales, const std::uint16_t* u_hi,
                  const std::uint16_t* u_lo, const float* inverse_rms,
                  std::uint16_t* __restrict__ x, int tokens) {
    extern __shared__ uint4 dynamic_shared[];
    MmaUpShared& shared = *reinterpret_cast<MmaUpShared*>(dynamic_shared);

    const int tid         = static_cast<int>(threadIdx.x);
    const int warp        = tid / kWarpSize;
    const int lane        = tid % kWarpSize;
    const int gid         = lane >> 2;
    const int lid         = lane & 3;
    const int wm          = warp / kMmaUpWarpsTokens;
    const int wn          = warp % kMmaUpWarpsTokens;
    const int hidden_base = static_cast<int>(blockIdx.x) * kMmaUpHidden;
    const int token_begin = static_cast<int>(blockIdx.y) * kMmaUpBlockTokens;
    const int tile_tokens = min(kMmaUpBlockTokens, tokens - token_begin);
    const auto global_row = [&](int local) {
        return (local / kMmaUpHidden) * kHidden + hidden_base + local % kMmaUpHidden;
    };

    const auto stage_codes = [&](int k_tile) {
        const int k_begin = k_tile * kMmaBlockK;
        for (int item = tid; item < kMmaUpBlockRows * (kMmaBlockK / 16); item += kMmaUpThreads) {
            const int local = item / (kMmaBlockK / 16);
            const int chunk = item % (kMmaBlockK / 16);
            cp_async<16>(&shared.codes[local * kMmaBlockK + chunk * 16],
                         up_codes + static_cast<std::int64_t>(global_row(local)) * kRank + k_begin +
                             chunk * 16);
        }
    };
    const auto stage_u = [&](int k_tile, int stage) {
        const int k_begin = k_tile * kMmaBlockK;
        for (int item = tid; item < kMmaUpBlockTokens * (kMmaBlockK / 8); item += kMmaUpThreads) {
            const int token    = item / (kMmaBlockK / 8);
            const int group    = item % (kMmaBlockK / 8);
            const bool valid   = token < tile_tokens;
            const int position = token * kMmaBlockK + fp8_a16_shared_col_64(token, group * 8);
            const std::int64_t source =
                static_cast<std::int64_t>(token_begin + (valid ? token : 0)) * kRank + k_begin +
                group * 8;
            cp_async_zfill<16, Cache::cg>(&shared.u_hi[stage][position], u_hi + source,
                                          valid ? 16 : 0);
            cp_async_zfill<16, Cache::cg>(&shared.u_lo[stage][position], u_lo + source,
                                          valid ? 16 : 0);
        }
    };

    stage_codes(0);
    pdl::wait_for_dependencies();
    stage_u(0, 0);
    cp_commit();

    float multiplier[kMmaUpRowMmas][2];
#pragma unroll
    for (int mt = 0; mt < kMmaUpRowMmas; ++mt) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            multiplier[mt][half] = bf16_bits_to_float(
                up_scales[global_row(wm * kMmaUpWarpRows + mt * 16 + gid + 8 * half)]);
        }
    }

    float acc[kMmaUpRowMmas][kMmaUpTokenMmas][4] = {};
    const int a_row                              = (lane & 7) + ((lane >> 3) & 1) * 8;
    const int a_col                              = (lane >> 4) * 8;
    const int b_row                              = lane & 7;
    const int b_col                              = ((lane >> 3) & 1) * 8;

#pragma unroll 1
    for (int k_tile = 0; k_tile < kMmaUpKTiles; ++k_tile) {
        const int stage = k_tile % 2;
        cp_wait<0>();
        __syncthreads();
        for (int pair = tid; pair < kMmaUpBlockRows * kMmaBlockK / 2; pair += kMmaUpThreads) {
            const int local = pair / (kMmaBlockK / 2);
            const int col   = (pair % (kMmaBlockK / 2)) * 2;
            const unsigned packed =
                *reinterpret_cast<const std::uint16_t*>(&shared.codes[local * kMmaBlockK + col]);
            store_vec(&shared.weight[local * kMmaBlockK + fp8_a16_shared_col_64(local, col)],
                      fp8_e4m3x2_to_bf16x2_bits(packed));
        }
        __syncthreads();
        if (k_tile + 1 < kMmaUpKTiles) {
            stage_codes(k_tile + 1);
            stage_u(k_tile + 1, stage ^ 1);
            cp_commit();
        }
#pragma unroll
        for (int k_step = 0; k_step < kMmaBlockK / 16; ++k_step) {
            unsigned a[kMmaUpRowMmas][4];
            unsigned b_hi[kMmaUpTokenMmas][2];
            unsigned b_lo[kMmaUpTokenMmas][2];
#pragma unroll
            for (int mt = 0; mt < kMmaUpRowMmas; ++mt) {
                const int row = wm * kMmaUpWarpRows + mt * 16 + a_row;
                ldmatrix_x4(
                    a[mt][0], a[mt][1], a[mt][2], a[mt][3],
                    smem_addr(&shared.weight[row * kMmaBlockK +
                                             fp8_a16_shared_col_64(row, k_step * 16 + a_col)]));
            }
#pragma unroll
            for (int nt = 0; nt < kMmaUpTokenMmas; ++nt) {
                const int token = wn * kMmaUpWarpTokens + nt * 8 + b_row;
                const int position =
                    token * kMmaBlockK + fp8_a16_shared_col_64(token, k_step * 16 + b_col);
                ldmatrix_x2(b_hi[nt][0], b_hi[nt][1], smem_addr(&shared.u_hi[stage][position]));
                ldmatrix_x2(b_lo[nt][0], b_lo[nt][1], smem_addr(&shared.u_lo[stage][position]));
            }
#pragma unroll
            for (int mt = 0; mt < kMmaUpRowMmas; ++mt) {
#pragma unroll
                for (int nt = 0; nt < kMmaUpTokenMmas; ++nt) {
                    float* c = acc[mt][nt];
                    mma_bf16(c[0], c[1], c[2], c[3], a[mt][0], a[mt][1], a[mt][2], a[mt][3],
                             b_hi[nt][0], b_hi[nt][1]);
                    mma_bf16(c[0], c[1], c[2], c[3], a[mt][0], a[mt][1], a[mt][2], a[mt][3],
                             b_lo[nt][0], b_lo[nt][1]);
                }
            }
        }
    }
    // Every warp has finished its last MMAs before the u stages are reused for the gates.
    __syncthreads();
    float* const gates = reinterpret_cast<float*>(shared.u_hi);
#pragma unroll
    for (int mt = 0; mt < kMmaUpRowMmas; ++mt) {
#pragma unroll
        for (int nt = 0; nt < kMmaUpTokenMmas; ++nt) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int local = wm * kMmaUpWarpRows + mt * 16 + gid + 8 * (e >> 1);
                const int token = wn * kMmaUpWarpTokens + nt * 8 + 2 * lid + (e & 1);
                gates[token * kMmaUpBlockRows + local] =
                    sigmoid(multiplier[mt][e >> 1] * acc[mt][nt][e]);
            }
        }
    }
    __syncthreads();
    const int h = tid % kMmaUpHidden;
    const int d = hidden_base + h;
    float gain[kStreams];
#pragma unroll
    for (int i = 0; i < kStreams; ++i) {
        gain[i] = 1.0F + bf16_bits_to_float(norm_weight[i * kHidden + d]);
    }
    for (int token = tid / kMmaUpHidden; token < tile_tokens;
         token += kMmaUpThreads / kMmaUpHidden) {
        const std::int64_t t = token_begin + token;
        float mixed          = 0.0F;
#pragma unroll
        for (int i = 0; i < kStreams; ++i) {
            const float normalized =
                bf16_bits_to_float(__ldcg(residual + t * kFlat + i * kHidden + d)) *
                __ldcg(inverse_rms + t * kStreams + i) * gain[i];
            mixed = fmaf(gates[token * kMmaUpBlockRows + i * kMmaUpHidden + h], normalized, mixed);
        }
        x[t * kHidden + d] = __bfloat16_as_ushort(__float2bfloat16_rn(mixed * 0.25F));
    }
}

// ---------------------------------------------------------------------------------------------
// Small-T Tensor Core route. The up projection runs here for kSmallUpMinTokens <= T <=
// kSmallMaxTokens and the down/write projection for kSmallDownMinTokens <= T <= kSmallMaxTokens,
// where each measured faster than its SIMT kernel in the 16-read chain; T = 1, the down projection
// at T = 2, and T = 33..159 stay on SIMT. Both kernels read and write the SIMT workspace layout
// (partials, square_sums), so either pairs with the other route's kernel.
//
// The weight rows sit on MMA-M and the tokens, padded to kTokenMmas 8-token MMA tiles, on MMA-N.
// Weight operands are exact: E4M3 codes widen exactly to BF16 and the write rows are BF16. Each
// FP32 activation operand v is entered as the BF16 pair hi = bf16(v), lo = bf16(v - hi), as on the
// prefill route: the down activation is R (1 + norm) (one FMA of the BF16 residual and gain) and
// the up activation the gate input u. Row multipliers are applied once to finished sums.
//
// Fragments are loaded straight from 16-byte words, not through ldmatrix: lane (g, q) = (lane / 4,
// lane % 4) owns the 16 consecutive inputs 16q .. 16q+15 of each 64-input chunk, and in k step j
// (0..3) of the chunk its MMA k slots {2q, 2q+1} carry inputs 16q + 4j + {0, 1} and its slots
// {2q+8, 2q+9} carry inputs 16q + 4j + {2, 3}. A and B fragments use the same slot-to-input map, so
// every product pairs matching inputs; the dot product only depends on the set of inputs, which
// the four k steps of the four lane quads cover exactly once.
constexpr int kSmallUpMinTokens   = 2;
constexpr int kSmallDownMinTokens = 3;
constexpr int kSmallTileTokens    = 2 * kTokenTile;
constexpr int kSmallMaxTokens     = kHyperConnectionSmallMaxTokens;
static_assert(kSmallMaxTokens == 2 * kSmallTileTokens);
static_assert(kHyperConnectionFinishMinTokens == kTokenTile + 1);
constexpr int kSmallChunkK     = 64;
constexpr int kSmallLaneInputs = kSmallChunkK / 4;
static_assert(kSmallLaneInputs == kCodesPerWord);

// Down/write: one CTA per 16-row MMA tile and split; its 20 warps each reduce 128 of the split's
// 2560 inputs (two chunks).
constexpr int kSmallDownWarps   = 20;
constexpr int kSmallDownThreads = kSmallDownWarps * kWarpSize;
constexpr int kSmallDownChunks  = kHidden / (kSmallDownWarps * kSmallChunkK);
constexpr int kSmallWriteTile   = kRank / 16;
static_assert(kHidden % (kSmallDownWarps * kSmallChunkK) == 0 && kRank % 16 == 0);
static_assert(kStreams <= 8);

// Up: one CTA per 16 hidden indices, i.e. one 16-row MMA tile per stream; warp w serves stream
// w / 5 and chunk w % 5 of the 320 rank inputs.
constexpr int kSmallUpHidden  = 16;
constexpr int kSmallUpChunks  = kRank / kSmallChunkK;
constexpr int kSmallUpWarps   = kStreams * kSmallUpChunks;
constexpr int kSmallUpThreads = kSmallUpWarps * kWarpSize;
// BF16 elements per staged u row: a 656-byte stride puts the 16-byte words a phase of eight lanes
// reads ((g, q) for two consecutive g) in eight distinct 4-bank groups.
constexpr int kSmallUStride = kRank + 8;
static_assert(kRank % kSmallChunkK == 0 && kHidden % kSmallUpHidden == 0);
static_assert((kSmallUStride * 2 / 16) % 2 == 1);

// Widens the four E4M3 codes of word to the A-fragment pairs of one k step: low holds codes 0, 1
// and high codes 2, 3.
__device__ __forceinline__ void widen_code_word(std::uint32_t word, std::uint32_t& low,
                                                std::uint32_t& high) {
    low  = fp8_e4m3x2_to_bf16x2_bits(word & 0xFFFFU);
    high = fp8_e4m3x2_to_bf16x2_bits(word >> 16);
}

__device__ __forceinline__ void words_of(const uint4& value, std::uint32_t* out) {
    out[0] = value.x;
    out[1] = value.y;
    out[2] = value.z;
    out[3] = value.w;
}

// Shared memory of small_down_kernel (dynamic): weights holds each lane's own staged words (codes
// of rows g and g + 8, or the two BF16 words of write row g); sums and squares carry the per-warp
// partial results to the cross-warp reduction.
template <int kTokenMmas>
struct SmallDownShared {
    uint4 weights[kSmallDownWarps][kSmallDownChunks][2][kWarpSize];
    float sums[kSmallDownWarps][kTokenMmas * 4][kWarpSize];
    float squares[kSmallDownWarps][kTokenMmas * 8];
};

// Grid (20 down tiles + 1 write tile when write is present, 4 splits, ceil(T / 16)),
// block 640 (20 warps),
// dynamic shared memory SmallDownShared. CTA (m, s) projects rows 16m .. 16m+15 (tile 20: write
// rows 320..323 and 12 zero rows) over the inputs of residual stream s and token tile 16
// blockIdx.z. Warp w reduces split inputs 128w .. 128w+127; lane (g, q) handles rows 16m + g and
// 16m + g + 8, token 8n + g of MMA tile n, and inputs 64c + 16q .. +15 of the warp's chunk c.
//
// Before waiting on the producer of residual, every lane copies its own weight words into shared
// memory with cp.async (register loads may be sunk below the wait) and loads its norm words; the
// CTA triggers the dependent launch and pulls a slice of the up code plane into L2, as
// down_write_kernel does. After the wait each lane loads its tokens' residual words (L2-coherent),
// waits for its own copies (each lane reads back only its own words), and runs 4 k steps x
// kTokenMmas x 2 (hi, lo) MMAs per chunk. Row tile 0 also sums its lanes' squared residual
// values. The per-warp accumulators go to shared memory, one barrier, and thread i < 128
// kTokenMmas sums one accumulator element over the 20 warps and stores partials[s][t][row] =
// multiplier * sum_k W[row,k] R[t,k] (1 + norm[k]); row tile 0 also stores square_sums[s][t].
template <int kTokenMmas>
__global__ void __launch_bounds__(kSmallDownThreads)
    small_down_kernel(const std::uint16_t* __restrict__ residual,
                      const uint4* __restrict__ norm_weight,
                      const std::uint8_t* __restrict__ down_codes,
                      const std::uint16_t* __restrict__ down_scales,
                      const std::uint16_t* __restrict__ write_weight, const std::uint8_t* up_codes,
                      float* __restrict__ partials, float* __restrict__ square_sums, int tokens) {
    extern __shared__ uint4 dynamic_shared[];
    auto& shared = *reinterpret_cast<SmallDownShared<kTokenMmas>*>(dynamic_shared);

    const int tid         = static_cast<int>(threadIdx.x);
    const int warp        = tid / kWarpSize;
    const int lane        = tid % kWarpSize;
    const int gid         = lane >> 2;
    const int lid         = lane & 3;
    const int tile        = static_cast<int>(blockIdx.x);
    const int split       = static_cast<int>(blockIdx.y);
    const int token_begin = static_cast<int>(blockIdx.z) * kSmallTileTokens;
    const int tile_tokens = min(kTokenMmas * 8, tokens - token_begin);
    const bool write_tile = tile == kSmallWriteTile;
    const bool squares    = tile == 0;
    const int row_a       = tile * 16 + gid;
    const auto input      = [&](int chunk) {
        return split * kHidden + (warp * kSmallDownChunks + chunk) * kSmallChunkK +
               lid * kSmallLaneInputs;
    };

    uint4 gain_words[kSmallDownChunks][2];
#pragma unroll
    for (int chunk = 0; chunk < kSmallDownChunks; ++chunk) {
        const int k = input(chunk);
        if (!write_tile) {
            cp_async<16>(&shared.weights[warp][chunk][0][lane],
                         down_codes + static_cast<std::int64_t>(row_a) * kFlat + k);
            cp_async<16>(&shared.weights[warp][chunk][1][lane],
                         down_codes + static_cast<std::int64_t>(row_a + 8) * kFlat + k);
        } else if (gid < kStreams) {
            cp_async<16>(&shared.weights[warp][chunk][0][lane], write_weight + gid * kFlat + k);
            cp_async<16>(&shared.weights[warp][chunk][1][lane],
                         write_weight + gid * kFlat + k + kBf16PerWord);
        }
        gain_words[chunk][0] = norm_weight[k / kBf16PerWord];
        gain_words[chunk][1] = norm_weight[k / kBf16PerWord + 1];
    }
    cp_commit();
    if (tid == 0) {
        pdl::trigger_dependents();
        const int cta =
            static_cast<int>((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x);
        const int ctas   = static_cast<int>(gridDim.x * gridDim.y * gridDim.z);
        const int slice  = (kUpCodeBytes / ctas + 15) / 16 * 16;
        const int offset = cta * slice;
        if (offset < kUpCodeBytes) {
            prefetch_l2(up_codes + offset, min(slice, kUpCodeBytes - offset));
        }
    }
    pdl::wait_for_dependencies();

    uint4 residual_words[kSmallDownChunks][kTokenMmas][2];
#pragma unroll
    for (int chunk = 0; chunk < kSmallDownChunks; ++chunk) {
#pragma unroll
        for (int nt = 0; nt < kTokenMmas; ++nt) {
            const int token = nt * 8 + gid;
            if (token < tile_tokens) {
                const auto* source = reinterpret_cast<const uint4*>(
                    residual + static_cast<std::int64_t>(token_begin + token) * kFlat +
                    input(chunk));
                residual_words[chunk][nt][0] = __ldcg(source);
                residual_words[chunk][nt][1] = __ldcg(source + 1);
            } else {
                residual_words[chunk][nt][0] = make_uint4(0, 0, 0, 0);
                residual_words[chunk][nt][1] = make_uint4(0, 0, 0, 0);
            }
        }
    }
    cp_wait<0>();
    __syncwarp();

    float acc[kTokenMmas][4] = {};
    float square[kTokenMmas] = {};
#pragma unroll
    for (int chunk = 0; chunk < kSmallDownChunks; ++chunk) {
        float gain[kSmallLaneInputs];
        unpack_bf16x8(gain_words[chunk][0], gain);
        unpack_bf16x8(gain_words[chunk][1], gain + kBf16PerWord);
        std::uint32_t b_hi[kTokenMmas][kSmallLaneInputs / 2];
        std::uint32_t b_lo[kTokenMmas][kSmallLaneInputs / 2];
#pragma unroll
        for (int nt = 0; nt < kTokenMmas; ++nt) {
            float r[kSmallLaneInputs];
            unpack_bf16x8(residual_words[chunk][nt][0], r);
            unpack_bf16x8(residual_words[chunk][nt][1], r + kBf16PerWord);
#pragma unroll
            for (int i = 0; i < kSmallLaneInputs; i += 2) {
                if (squares) {
                    square[nt] = fmaf(r[i], r[i], square[nt]);
                    square[nt] = fmaf(r[i + 1], r[i + 1], square[nt]);
                }
                split_bf16x2(fmaf(r[i], gain[i], r[i]), fmaf(r[i + 1], gain[i + 1], r[i + 1]),
                             b_hi[nt][i / 2], b_lo[nt][i / 2]);
            }
        }
        std::uint32_t row_low[4]  = {};
        std::uint32_t row_high[4] = {};
        if (!write_tile || gid < kStreams) {
            words_of(shared.weights[warp][chunk][0][lane], row_low);
            words_of(shared.weights[warp][chunk][1][lane], row_high);
        }
#pragma unroll
        for (int step = 0; step < 4; ++step) {
            std::uint32_t a[4];
            if (!write_tile) {
                widen_code_word(row_low[step], a[0], a[2]);
                widen_code_word(row_high[step], a[1], a[3]);
            } else {
                // BF16 write row g: inputs 4j, 4j+1 are word 2j of the two staged words.
                const std::uint32_t* values = step < 2 ? row_low : row_high;
                a[0]                        = values[(2 * step) % 4];
                a[2]                        = values[(2 * step) % 4 + 1];
                a[1]                        = 0;
                a[3]                        = 0;
            }
#pragma unroll
            for (int nt = 0; nt < kTokenMmas; ++nt) {
                float* c = acc[nt];
                mma_bf16(c[0], c[1], c[2], c[3], a[0], a[1], a[2], a[3], b_hi[nt][2 * step],
                         b_hi[nt][2 * step + 1]);
                mma_bf16(c[0], c[1], c[2], c[3], a[0], a[1], a[2], a[3], b_lo[nt][2 * step],
                         b_lo[nt][2 * step + 1]);
            }
        }
    }

#pragma unroll
    for (int nt = 0; nt < kTokenMmas; ++nt) {
#pragma unroll
        for (int e = 0; e < 4; ++e) { shared.sums[warp][nt * 4 + e][lane] = acc[nt][e]; }
    }
    if (squares) {
#pragma unroll
        for (int nt = 0; nt < kTokenMmas; ++nt) {
            float sum = square[nt];
            sum += __shfl_xor_sync(kFullWarpMask, sum, 1);
            sum += __shfl_xor_sync(kFullWarpMask, sum, 2);
            if (lid == 0) { shared.squares[warp][nt * 8 + gid] = sum; }
        }
    }
    __syncthreads();

    if (tid < kTokenMmas * 4 * kWarpSize) {
        const int element = tid / kWarpSize;
        const int source  = tid % kWarpSize;
        float sum         = 0.0F;
#pragma unroll
        for (int w = 0; w < kSmallDownWarps; ++w) { sum += shared.sums[w][element][source]; }
        const int e     = element % 4;
        const int row   = tile * 16 + (source >> 2) + 8 * (e >> 1);
        const int token = (element / 4) * 8 + 2 * (source & 3) + (e & 1);
        if (token < tile_tokens && row < (write_tile ? kRows : kRank)) {
            const float multiplier = write_tile ? 1.0F : bf16_bits_to_float(down_scales[row]);
            partials[(static_cast<std::int64_t>(split) * tokens + token_begin + token) * kRows +
                     row]          = multiplier * sum;
        }
    } else if (squares) {
        const int token = tid - kTokenMmas * 4 * kWarpSize;
        if (token < tile_tokens) {
            float sum = 0.0F;
#pragma unroll
            for (int w = 0; w < kSmallDownWarps; ++w) { sum += shared.squares[w][token]; }
            square_sums[static_cast<std::int64_t>(split) * tokens + token_begin + token] = sum;
        }
    }
}

// Shared memory of small_up_kernel (dynamic): each lane's own staged code words (rows g and g + 8),
// the gate input u as BF16 hi/lo [token][kSmallUStride], the per-warp accumulators, the gates
// [token][stream][hidden], and the inverse RMS [token][stream].
template <int kTokenMmas>
struct SmallUpShared {
    uint4 codes[kSmallUpWarps][2][kWarpSize];
    std::uint16_t u_hi[kTokenMmas * 8][kSmallUStride];
    std::uint16_t u_lo[kTokenMmas * 8][kSmallUStride];
    float sums[kSmallUpWarps][kTokenMmas * 4][kWarpSize];
    float gates[kTokenMmas * 8][kStreams][kSmallUpHidden];
    float inverse_rms[kTokenMmas * 8][kStreams];
};

// Grid (2560 / 16 hidden groups, ceil(T / 16)), block 640 (20 warps), dynamic shared memory
// SmallUpShared. Token indices in shared memory are local to the 16-token tile. CTA b
// serves hidden indices d = 16b .. 16b+15; its MMA tile i holds up rows i*2560 + d, so the CTA has
// all four gates of its indices. Warp w = 5i + c multiplies tile i with rank inputs 64c .. 64c+63;
// lane (g, q) handles rows d = 16b + g and 16b + g + 8 and local token 8n + g of MMA tile n.
// Global tokens add 16 blockIdx.y; invalid tail-token operands are zero.
//
// Before the PDL wait every lane copies its two code words into shared memory with cp.async and
// mix thread (t, h) = (tid / 16, tid % 16) loads its (1 + norm) gains. After the wait every
// producer-dependent load is issued at once: the stream square sums (thread 4t + s), the split
// partials of 4 consecutive ranks per item (item tid + 640m covers token item / 80, ranks
// 4 (item % 80) .. +3), the write partials (CTA 0), and the mix residual values. Barrier 1
// publishes inverse_rms; the items then store u = SiLU(sum_s inverse_rms[t][s] partials[s][t][r] /
// 4) as hi/lo BF16, CTA 0 stores write_scale, and barrier 2 publishes u (and orders each lane's
// read-back of its own staged codes after its cp.async wait). With kFinished, the finish kernel
// owns the gate-input and write-scale calculation: this CTA instead loads the global token tile's
// u_hi/u_lo and inverse RMS with L2-coherent vector loads after the dependency wait, then publishes
// them at the code-copy completion barrier; the fused prologue's first barrier is unnecessary.
// Each warp runs 4 k steps x kTokenMmas
// x 2 (hi, lo) MMAs; barrier 3 publishes the accumulators, every gate is the sum over the tile's
// five chunk warps, gates[t][i][h] = sigmoid(multiplier * sum), and after barrier 4 mix thread
// (t, h) forms x[t][d] = (1/4) sum_i gate_i * R[t,i,d] inverse_rms[t][i] (1 + norm[i*2560 + d]).
template <int kTokenMmas, bool kFinished>
__global__ void __launch_bounds__(kSmallUpThreads)
    small_up_kernel(const std::uint16_t* __restrict__ residual,
                    const std::uint16_t* __restrict__ norm_weight,
                    const std::uint8_t* __restrict__ up_codes,
                    const std::uint16_t* __restrict__ up_scales, const float* __restrict__ partials,
                    const float* __restrict__ square_sums, std::uint16_t* __restrict__ x,
                    float* __restrict__ write_scale, HyperConnectionMmaWorkspace finished,
                    int tokens) {
    constexpr int kTileTokens = kTokenMmas * 8;
    constexpr int kRankQuads  = kRank / 4;
    constexpr int kUItems     = (kTileTokens * kRankQuads + kSmallUpThreads - 1) / kSmallUpThreads;
    static_assert(kTileTokens * kSmallUpHidden <= kSmallUpThreads &&
                  kTileTokens * kStreams <= kSmallUpThreads);
    extern __shared__ uint4 dynamic_shared[];
    auto& shared = *reinterpret_cast<SmallUpShared<kTokenMmas>*>(dynamic_shared);

    const int tid         = static_cast<int>(threadIdx.x);
    const int warp        = tid / kWarpSize;
    const int lane        = tid % kWarpSize;
    const int gid         = lane >> 2;
    const int lid         = lane & 3;
    const int stream      = warp / kSmallUpChunks;
    const int chunk       = warp % kSmallUpChunks;
    const int hidden_base = static_cast<int>(blockIdx.x) * kSmallUpHidden;
    const int token_begin = static_cast<int>(blockIdx.y) * kSmallTileTokens;
    const int tile_tokens = min(kTileTokens, tokens - token_begin);
    const int row_a       = stream * kHidden + hidden_base + gid;
    const int rank_input  = chunk * kSmallChunkK + lid * kSmallLaneInputs;

    cp_async<16>(&shared.codes[warp][0][lane],
                 up_codes + static_cast<std::int64_t>(row_a) * kRank + rank_input);
    cp_async<16>(&shared.codes[warp][1][lane],
                 up_codes + static_cast<std::int64_t>(row_a + 8) * kRank + rank_input);
    cp_commit();
    const int mix_token  = tid / kSmallUpHidden;
    const int mix_hidden = hidden_base + tid % kSmallUpHidden;
    const bool mixer     = mix_token < tile_tokens;
    float gain[kStreams];
    if (mixer) {
#pragma unroll
        for (int i = 0; i < kStreams; ++i) {
            gain[i] = 1.0F + bf16_bits_to_float(norm_weight[i * kHidden + mix_hidden]);
        }
    }
    pdl::wait_for_dependencies();

    std::uint16_t mix_residual[kStreams];
    if (mixer) {
#pragma unroll
        for (int i = 0; i < kStreams; ++i) {
            mix_residual[i] =
                __ldcg(residual + static_cast<std::int64_t>(token_begin + mix_token) * kFlat +
                       i * kHidden + mix_hidden);
        }
    }
    if constexpr (kFinished) {
        for (int item = tid; item < tile_tokens * (kRank / 8); item += kSmallUpThreads) {
            const int token = item / (kRank / 8);
            const int rank  = (item % (kRank / 8)) * 8;
            const std::int64_t source =
                static_cast<std::int64_t>(token_begin + token) * kRank + rank;
            store_vec(&shared.u_hi[token][rank],
                      __ldcg(reinterpret_cast<const uint4*>(finished.u_hi + source)));
            store_vec(&shared.u_lo[token][rank],
                      __ldcg(reinterpret_cast<const uint4*>(finished.u_lo + source)));
        }
        if (tid < tile_tokens * kStreams) {
            shared.inverse_rms[tid / kStreams][tid % kStreams] = __ldcg(
                finished.inverse_rms + static_cast<std::int64_t>(token_begin) * kStreams + tid);
        }
    } else {
        const int head_token    = tid / kStreams;
        const int head_index    = tid % kStreams;
        const bool head         = head_token < tile_tokens;
        const bool write_head   = head && write_scale != nullptr && blockIdx.x == 0;
        float stream_square_sum = 0.0F;
        float write_partials[kSplits];
        if (head) {
            stream_square_sum =
                __ldcg(square_sums + static_cast<std::int64_t>(head_index) * tokens + token_begin +
                       head_token);
        }
        if (write_head) {
#pragma unroll
            for (int split = 0; split < kSplits; ++split) {
                write_partials[split] = __ldcg(
                    partials +
                    (static_cast<std::int64_t>(split) * tokens + token_begin + head_token) * kRows +
                    kRank + head_index);
            }
        }
        float4 rank_partials[kUItems][kSplits];
#pragma unroll
        for (int item = 0; item < kUItems; ++item) {
            const int index = tid + item * kSmallUpThreads;
            if (index < tile_tokens * kRankQuads) {
#pragma unroll
                for (int split = 0; split < kSplits; ++split) {
                    rank_partials[item][split] = __ldcg(
                        reinterpret_cast<const float4*>(partials +
                                                        (static_cast<std::int64_t>(split) * tokens +
                                                         token_begin + index / kRankQuads) *
                                                            kRows +
                                                        4 * (index % kRankQuads)));
                }
            }
        }
        if (head) {
            shared.inverse_rms[head_token][head_index] =
                rsqrtf(stream_square_sum * (1.0F / static_cast<float>(kHidden)) + 1.0e-6F);
        }
        __syncthreads();
#pragma unroll
        for (int item = 0; item < kUItems; ++item) {
            const int index = tid + item * kSmallUpThreads;
            if (index < tile_tokens * kRankQuads) {
                const int token = index / kRankQuads;
                const int rank  = 4 * (index % kRankQuads);
                float u[4];
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    float pre = 0.0F;
#pragma unroll
                    for (int split = 0; split < kSplits; ++split) {
                        const float4& p   = rank_partials[item][split];
                        const float value = e == 0 ? p.x : (e == 1 ? p.y : (e == 2 ? p.z : p.w));
                        pre               = fmaf(shared.inverse_rms[token][split], value, pre);
                    }
                    u[e] = silu(pre * 0.25F);
                }
                std::uint32_t hi[2];
                std::uint32_t lo[2];
                split_bf16x2(u[0], u[1], hi[0], lo[0]);
                split_bf16x2(u[2], u[3], hi[1], lo[1]);
                store_vec(&shared.u_hi[token][rank], make_uint2(hi[0], hi[1]));
                store_vec(&shared.u_lo[token][rank], make_uint2(lo[0], lo[1]));
            }
        }
        if (write_head) {
            float sum = 0.0F;
#pragma unroll
            for (int split = 0; split < kSplits; ++split) {
                sum = fmaf(shared.inverse_rms[head_token][split], write_partials[split], sum);
            }
            write_scale[static_cast<std::int64_t>(token_begin + head_token) * kStreams +
                        head_index] = 2.0F * sigmoid(sum * 0.25F);
        }
    }
    cp_wait<0>();
    __syncthreads();

    std::uint32_t row_low[4];
    std::uint32_t row_high[4];
    words_of(shared.codes[warp][0][lane], row_low);
    words_of(shared.codes[warp][1][lane], row_high);
    std::uint32_t b_hi[kTokenMmas][kSmallLaneInputs / 2];
    std::uint32_t b_lo[kTokenMmas][kSmallLaneInputs / 2];
#pragma unroll
    for (int nt = 0; nt < kTokenMmas; ++nt) {
        const int token = nt * 8 + gid;
        uint4 words[4]  = {};
        if (token < tile_tokens) {
            const auto* hi = reinterpret_cast<const uint4*>(&shared.u_hi[token][rank_input]);
            const auto* lo = reinterpret_cast<const uint4*>(&shared.u_lo[token][rank_input]);
            words[0]       = hi[0];
            words[1]       = hi[1];
            words[2]       = lo[0];
            words[3]       = lo[1];
        }
        words_of(words[0], b_hi[nt]);
        words_of(words[1], b_hi[nt] + 4);
        words_of(words[2], b_lo[nt]);
        words_of(words[3], b_lo[nt] + 4);
    }
    float acc[kTokenMmas][4] = {};
#pragma unroll
    for (int step = 0; step < 4; ++step) {
        std::uint32_t a[4];
        widen_code_word(row_low[step], a[0], a[2]);
        widen_code_word(row_high[step], a[1], a[3]);
#pragma unroll
        for (int nt = 0; nt < kTokenMmas; ++nt) {
            float* c = acc[nt];
            mma_bf16(c[0], c[1], c[2], c[3], a[0], a[1], a[2], a[3], b_hi[nt][2 * step],
                     b_hi[nt][2 * step + 1]);
            mma_bf16(c[0], c[1], c[2], c[3], a[0], a[1], a[2], a[3], b_lo[nt][2 * step],
                     b_lo[nt][2 * step + 1]);
        }
    }
#pragma unroll
    for (int nt = 0; nt < kTokenMmas; ++nt) {
#pragma unroll
        for (int e = 0; e < 4; ++e) { shared.sums[warp][nt * 4 + e][lane] = acc[nt][e]; }
    }
    __syncthreads();

    for (int index = tid; index < kStreams * kTokenMmas * 4 * kWarpSize; index += kSmallUpThreads) {
        const int gate_stream = index / (kTokenMmas * 4 * kWarpSize);
        const int element     = (index / kWarpSize) % (kTokenMmas * 4);
        const int source      = index % kWarpSize;
        const int e           = element % 4;
        const int token       = (element / 4) * 8 + 2 * (source & 3) + (e & 1);
        if (token < tile_tokens) {
            const int hidden = (source >> 2) + 8 * (e >> 1);
            float sum        = 0.0F;
#pragma unroll
            for (int c = 0; c < kSmallUpChunks; ++c) {
                sum += shared.sums[gate_stream * kSmallUpChunks + c][element][source];
            }
            const float multiplier =
                bf16_bits_to_float(up_scales[gate_stream * kHidden + hidden_base + hidden]);
            shared.gates[token][gate_stream][hidden] = sigmoid(multiplier * sum);
        }
    }
    __syncthreads();

    if (mixer) {
        const int local = mix_hidden - hidden_base;
        float mixed     = 0.0F;
#pragma unroll
        for (int i = 0; i < kStreams; ++i) {
            const float normalized =
                bf16_bits_to_float(mix_residual[i]) * shared.inverse_rms[mix_token][i] * gain[i];
            mixed = fmaf(shared.gates[mix_token][i][local], normalized, mixed);
        }
        x[static_cast<std::int64_t>(token_begin + mix_token) * kHidden + mix_hidden] =
            __bfloat16_as_ushort(__float2bfloat16_rn(mixed * 0.25F));
    }
}

// Grid (T), block 320: thread j copies the j-th 16-byte word of the token's embedding into all four
// streams.
__global__ void __launch_bounds__(kHidden / 8)
    broadcast_kernel(const uint4* __restrict__ embedding, uint4* __restrict__ residual) {
    const int token      = static_cast<int>(blockIdx.x);
    const int word       = static_cast<int>(threadIdx.x);
    constexpr int kWords = kHidden / 8;
    const uint4 value    = embedding[static_cast<std::int64_t>(token) * kWords + word];
#pragma unroll
    for (int stream = 0; stream < kStreams; ++stream) {
        residual[(static_cast<std::int64_t>(token) * kStreams + stream) * kWords + word] = value;
    }
}

// Grid (T, 5), block 256: thread j of tile b updates the token's 16-byte word 256b + j.
// residual and residual_out may alias exactly; each word has one owning thread and is read before
// its store, so neither pointer is __restrict__. Triggering at entry lets the next read stage
// immutable weights; that read waits for this grid before accessing residual. No shared memory
// or cross-thread synchronization is required.
__global__ void __launch_bounds__(kThreads)
    inject_kernel(const uint4* residual, const uint4* __restrict__ block_output,
                  const float* __restrict__ write_scale, uint4* residual_out) {
    constexpr int kWords       = kFlat / 8;
    constexpr int kStreamWords = kHidden / 8;
    const std::int64_t token   = blockIdx.x;
    pdl::trigger_dependents();
    static_assert(kWords % kThreads == 0);
    const int word    = static_cast<int>(blockIdx.y) * kThreads + static_cast<int>(threadIdx.x);
    const float scale = write_scale[token * kStreams + word / kStreamWords];
    float r[8];
    float y[8];
    unpack_bf16x8(residual[token * kWords + word], r);
    unpack_bf16x8(block_output[token * kStreamWords + word % kStreamWords], y);
    std::uint32_t packed[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const __nv_bfloat162 pair = __floats2bfloat162_rn(fmaf(scale, y[2 * i], r[2 * i]),
                                                          fmaf(scale, y[2 * i + 1], r[2 * i + 1]));
        packed[i]                 = *reinterpret_cast<const std::uint32_t*>(&pair);
    }
    residual_out[token * kWords + word] = make_uint4(packed[0], packed[1], packed[2], packed[3]);
}

template <int kRowsPerWarp>
void launch_down_write(const std::uint16_t* residual, const std::uint16_t* norm_weight,
                       HyperConnectionFp8Planes down, const std::uint16_t* write_weight,
                       const std::uint8_t* up_codes, float* partials, float* square_sums,
                       std::int32_t tokens, int token_tiles, cudaStream_t stream) {
    static_assert(sizeof(DownStage) <= 48 * 1024);
    const int groups = kRank / (kWarps * kRowsPerWarp) + (write_weight != nullptr ? 1 : 0);
    CUDA_CHECK(pdl::launch_dependent(
        pdl::LaunchConfig{dim3(groups, kSplits, token_tiles), dim3(kThreads), sizeof(DownStage),
                          stream},
        down_write_kernel<kRowsPerWarp>, residual, reinterpret_cast<const uint4*>(norm_weight),
        reinterpret_cast<const uint4*>(down.codes), down.scales,
        reinterpret_cast<const uint4*>(write_weight), up_codes, partials, square_sums, tokens));
}

template <int kHiddenPerCta>
void launch_up_mix(const std::uint16_t* residual, const std::uint16_t* norm_weight,
                   HyperConnectionFp8Planes up, const float* partials, const float* square_sums,
                   std::uint16_t* x, float* write_scale, std::int32_t tokens, int token_tiles,
                   cudaStream_t stream) {
    CUDA_CHECK(pdl::launch_dependent(
        pdl::LaunchConfig{dim3(kHidden / kHiddenPerCta, token_tiles), dim3(kThreads), 0, stream},
        up_mix_kernel<kHiddenPerCta>, residual, norm_weight,
        reinterpret_cast<const uint4*>(up.codes), up.scales, partials, square_sums, x, write_scale,
        tokens));
}

template <int kTokenMmas>
void launch_small_down(const std::uint16_t* residual, const std::uint16_t* norm_weight,
                       HyperConnectionFp8Planes down, const std::uint16_t* write_weight,
                       const std::uint8_t* up_codes, float* partials, float* square_sums,
                       std::int32_t tokens, cudaStream_t stream) {
    constexpr int kBytes = sizeof(SmallDownShared<kTokenMmas>);
    static_assert(kBytes <= 99 * 1024);
    static const cudaError_t attribute = cudaFuncSetAttribute(
        small_down_kernel<kTokenMmas>, cudaFuncAttributeMaxDynamicSharedMemorySize, kBytes);
    CUDA_CHECK(attribute);
    const int tiles = kSmallWriteTile + (write_weight != nullptr ? 1 : 0);
    CUDA_CHECK(pdl::launch_dependent(
        pdl::LaunchConfig{dim3(tiles, kSplits, (tokens + kSmallTileTokens - 1) / kSmallTileTokens),
                          dim3(kSmallDownThreads), kBytes, stream},
        small_down_kernel<kTokenMmas>, residual, reinterpret_cast<const uint4*>(norm_weight),
        down.codes, down.scales, write_weight, up_codes, partials, square_sums, tokens));
}

template <int kTokenMmas, bool kFinished>
void launch_small_up(const std::uint16_t* residual, const std::uint16_t* norm_weight,
                     HyperConnectionFp8Planes up, const float* partials, const float* square_sums,
                     std::uint16_t* x, float* write_scale, HyperConnectionMmaWorkspace finished,
                     std::int32_t tokens, cudaStream_t stream) {
    constexpr int kBytes = sizeof(SmallUpShared<kTokenMmas>);
    static_assert(kBytes <= 99 * 1024);
    static const cudaError_t attribute =
        cudaFuncSetAttribute(small_up_kernel<kTokenMmas, kFinished>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, kBytes);
    CUDA_CHECK(attribute);
    CUDA_CHECK(pdl::launch_dependent(
        pdl::LaunchConfig{
            dim3(kHidden / kSmallUpHidden, (tokens + kSmallTileTokens - 1) / kSmallTileTokens),
            dim3(kSmallUpThreads), kBytes, stream},
        small_up_kernel<kTokenMmas, kFinished>, residual, norm_weight, up.codes, up.scales,
        partials, square_sums, x, write_scale, finished, tokens));
}

} // namespace

void hyper_connection_broadcast_launch(const std::uint16_t* embedding, std::uint16_t* residual,
                                       std::int32_t tokens, cudaStream_t stream) {
    broadcast_kernel<<<tokens, kHidden / 8, 0, stream>>>(reinterpret_cast<const uint4*>(embedding),
                                                         reinterpret_cast<uint4*>(residual));
    CUDA_CHECK(cudaGetLastError());
}

// Every read kernel is a programmatic dependent: the down/write kernel (down_write or
// small_down) stages or loads its weights while its predecessor drains and waits before reading
// residual or writing the workspace, and the up kernel (up_mix or small_up) stages its first up
// rows while the down kernel runs and waits before reading the workspace, residual, or writing x
// and write_scale. When a separate finish kernel produces u and inverse RMS, it triggers at entry
// and the up kernel waits for that finish before loading its results. A dependent grid can be
// resident while its producers still write, so it reads producer data (residual, partials,
// square_sums) only through L2-coherent loads (ld.global.cg, cp.async.cg), never through the
// non-coherent read-only path, whose cached lines may predate the producer's writes. Only immutable
// weights use that path.
void hyper_connection_read_launch(const std::uint16_t* residual, const std::uint16_t* norm_weight,
                                  HyperConnectionFp8Planes down, HyperConnectionFp8Planes up,
                                  const std::uint16_t* write_weight, std::uint16_t* x,
                                  float* write_scale, float* partials, float* square_sums,
                                  HyperConnectionMmaWorkspace mma, std::int32_t tokens,
                                  cudaStream_t stream) {
    if (tokens >= kHyperConnectionMmaMinTokens) {
        static const cudaError_t down_attribute =
            cudaFuncSetAttribute(mma_down_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(sizeof(MmaDownShared)));
        CUDA_CHECK(down_attribute);
        static const cudaError_t up_attribute =
            cudaFuncSetAttribute(mma_up_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(sizeof(MmaUpShared)));
        CUDA_CHECK(up_attribute);
        CUDA_CHECK(pdl::launch_dependent(
            pdl::LaunchConfig{dim3(kMmaRows / kMmaDownBlockRows,
                                   (tokens + kMmaDownBlockTokens - 1) / kMmaDownBlockTokens,
                                   kSplits),
                              dim3(kMmaDownThreads), sizeof(MmaDownShared), stream},
            mma_down_kernel, residual, norm_weight, down.codes, down.scales, write_weight, partials,
            square_sums, tokens));
        CUDA_CHECK(pdl::launch_dependent(
            pdl::LaunchConfig{dim3(tokens), dim3(kMmaFinishThreads), 0, stream}, mma_finish_kernel,
            static_cast<const float*>(partials), static_cast<const float*>(square_sums), mma.u_hi,
            mma.u_lo, mma.inverse_rms, write_scale, tokens));
        CUDA_CHECK(pdl::launch_dependent(
            pdl::LaunchConfig{
                dim3(kHidden / kMmaUpHidden, (tokens + kMmaUpBlockTokens - 1) / kMmaUpBlockTokens),
                dim3(kMmaUpThreads), sizeof(MmaUpShared), stream},
            mma_up_kernel, residual, norm_weight, up.codes, up.scales,
            static_cast<const std::uint16_t*>(mma.u_hi),
            static_cast<const std::uint16_t*>(mma.u_lo), static_cast<const float*>(mma.inverse_rms),
            x, tokens));
        return;
    }
    const int token_tiles = (tokens + kTokenTile - 1) / kTokenTile;
    if (tokens >= kSmallDownMinTokens && tokens <= kSmallMaxTokens) {
        if (tokens <= kTokenTile) {
            launch_small_down<1>(residual, norm_weight, down, write_weight, up.codes, partials,
                                 square_sums, tokens, stream);
        } else {
            launch_small_down<2>(residual, norm_weight, down, write_weight, up.codes, partials,
                                 square_sums, tokens, stream);
        }
    } else if (tokens <= kDownNarrowMaxTokens) {
        launch_down_write<1>(residual, norm_weight, down, write_weight, up.codes, partials,
                             square_sums, tokens, token_tiles, stream);
    } else {
        launch_down_write<2>(residual, norm_weight, down, write_weight, up.codes, partials,
                             square_sums, tokens, token_tiles, stream);
    }
    if (tokens >= kSmallUpMinTokens && tokens <= kSmallMaxTokens) {
        if (tokens >= kHyperConnectionFinishMinTokens) {
            CUDA_CHECK(pdl::launch_dependent(
                pdl::LaunchConfig{dim3(tokens), dim3(kMmaFinishThreads), 0, stream},
                mma_finish_kernel, static_cast<const float*>(partials),
                static_cast<const float*>(square_sums), mma.u_hi, mma.u_lo, mma.inverse_rms,
                write_scale, tokens));
            launch_small_up<2, true>(residual, norm_weight, up, partials, square_sums, x,
                                     write_scale, mma, tokens, stream);
        } else {
            launch_small_up<1, false>(residual, norm_weight, up, partials, square_sums, x,
                                      write_scale, mma, tokens, stream);
        }
    } else if (tokens <= kUpNarrowMaxTokens) {
        launch_up_mix<kUpNarrowHidden>(residual, norm_weight, up, partials, square_sums, x,
                                       write_scale, tokens, token_tiles, stream);
    } else {
        launch_up_mix<kUpWideHidden>(residual, norm_weight, up, partials, square_sums, x,
                                     write_scale, tokens, token_tiles, stream);
    }
}

void hyper_connection_inject_launch(const std::uint16_t* residual,
                                    const std::uint16_t* block_output, const float* write_scale,
                                    std::uint16_t* residual_out, std::int32_t tokens,
                                    cudaStream_t stream) {
    inject_kernel<<<dim3(tokens, kFlat / (8 * kThreads)), kThreads, 0, stream>>>(
        reinterpret_cast<const uint4*>(residual), reinterpret_cast<const uint4*>(block_output),
        write_scale, reinterpret_cast<uint4*>(residual_out));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
