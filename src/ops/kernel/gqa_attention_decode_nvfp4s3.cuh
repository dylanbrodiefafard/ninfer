#pragma once

#include <cstdio> // device debug printf (temporary)

// Sage3-style NVFP4 GQA decode kernel (split-KV partials). QK is unchanged from
// gqa_attention_decode_nvfp4.cuh (E2M1 K + NVFP4 QK Tensor Core mma). The PV GEMM
// is FP4xFP4: P is quantized to E2M1 codes with per-16-key UE4M3 block scales, V
// stays FP4 (transposed to d-major smem, no BF16 dequant), and PV runs the
// block-scaled m16n8k64 NVFP4 instruction (one per n-tile instead of PVKs BF16
// matmuls).
//
// StrictPV: same QK + sage cache layout, but P stays BF16 (no S3 e2m1 grid /
// amplification) and V is dequantized to BF16 in smem, so PV runs the plain
// m16n8k16 BF16 mma chain (the exact-NVFP4 decode path's numerics on the sage
// cache). The FP4-PV rate win is dropped in exchange for removing the FP4-P-quant
// floor; decode attention is memory-bound, so the rate loss is not the wall.
//
// Cache writes (writes_cache): V is quantized in the S3 d-major layout (scale per
// (d, 16-key block) instead of per (key, d-16-group)). A new key joins an existing
// 16-key block, so the block scale is a running max: when the new key raises the
// max, the earlier codes of the block are rescaled (code * S_old/S_new) and
// re-quantized to E2M1.
//
// Numerics follow S3 (arXiv 2505.11594): P is amplified by 448*6 inside the
// softmax (constant folded into the exp2), the per-16-key block scale maps the
// amplified block max into e4m3 (max 448), codes land in [0, 6] (full e2m1 range),
// and the amplification cancels in out = acc / L.

#include "ops/kernel/gqa_attention_decode.cuh"
#include "ops/kernel/gqa_attention_kv_nvfp4.cuh"
#include "ops/kernel/gqa_attention_nvfp4s3_common.cuh"

#include <math_constants.h>

#include <cfloat>

namespace ninfer::ops {

// Dequantize 8 V values (4 e2m1 code bytes, low nibble = even d) with the S3 d-major
// per-(d, 16-key-block) scales (s[2i], s[2i+1] for the two nibbles of code byte i).
// The e2m1 x e4m3 product is exact in bf16 (2-bit x 4-bit mantissa within the 8-bit
// bf16 mantissa), so the strict-PV dequant introduces no extra rounding.
__device__ __forceinline__ int4 gqa_s3_dequant_v_bf16x8(const std::uint8_t* codes4,
                                                        const float* s) {
    unsigned packed[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float lo = gqa_s3_e2m1_value(codes4[i] & 0x0fu);
        const float hi = gqa_s3_e2m1_value((codes4[i] >> 4) & 0x0fu);
        packed[i] = pack_bf16x2(lo * s[2 * i], hi * s[2 * i + 1]);
    }
    return make_int4(static_cast<int>(packed[0]), static_cast<int>(packed[1]),
                     static_cast<int>(packed[2]), static_cast<int>(packed[3]));
}

// Hot-loop e2m1 decode without the const-memory table (the prefill header's
// gqa_s3_e2m1_value indexes an 8-float table; per-lane indices serialize in
// const memory, which the consumer-side PV dequant hits per fragment). Same
// exact values: {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}.
__device__ __forceinline__ float gqa_s3_e2m1_value_alu(std::uint8_t code) {
    const int e = (code >> 1) & 0x03;
    const float m = static_cast<float>(code & 0x01u);
    // e2m1: sign|exp(2)|mant(1). exp=0 -> subnormal 0.m (0, 0.5);
    // exp>=1 -> (1 + 0.5*m) * 2^(e-1) = {1,1.5,2,3,4,6}. Matches the host
    // decode_e2m1_word table (mantissa weight is 0.5, not 1).
    const float mag = (e == 0) ? 0.5f * m
                               : static_cast<float>(1 << (e - 1)) * (1.f + 0.5f * m);
    return (code & 0x08u) ? -mag : mag;
}

// Sparge-decode tile-skip rank kernel: one CTA per (kv_head, batch row) for a
// T=1 step. Ranks the window's Bc-key tiles with the meansim proxy
//
//   score(kb) = sum_d q_mean[d] * k_mean[page(kb), kv_head, d]
//
// (q_mean = the average of the kv group's BF16 query heads for this step;
// k_mean = the per-page dequantized K-mean plane the fill kernel writes - the
// same plane the prefill s3 rank uses), and emits a per-(kv_head, batch) keep
// list plus per-split prefix offsets so the partial kernel fetches only the
// kept tiles (K/V DRAM traffic cut by ~(1 - keep_frac)). The keep set is the
// top keep_frac fraction by (score desc, tile asc) plus attention sinks (the
// first tiles) and a recency window (the last tiles) - the prefill rank's
// coefficients. The selection is a 64-bin histogram with a deterministic
// tile-ascending tie-break inside the threshold bin, so a host oracle can
// rebuild the exact keep set from a dump.
template <typename Geometry, int Bc, bool MultiBatch>
__launch_bounds__(256) __global__
void gqa_attention_decode_rank_nvfp4s3_kernel(
    const __nv_bfloat16* q, const std::int32_t* pos, const std::int32_t* block_tables,
    const std::int32_t* table_rows, std::int32_t table_stride, std::int32_t full_width,
    std::int32_t column_begin, std::int32_t splits, float keep_frac, const float* k_mean,
    std::int32_t* keep_tiles, std::int32_t* keep_count, std::int32_t* split_off,
    std::int32_t n_pow2, std::int32_t keep_stride) {
    constexpr int D         = kGqaHeadDim;
    constexpr int Warps     = 8;
    constexpr int MemberCap = 256; // threshold-bin members beyond this keep the whole bin
    const int kv_head       = static_cast<int>(blockIdx.x);
    const int batch         = MultiBatch ? static_cast<int>(blockIdx.z) : 0;
    const int tid           = static_cast<int>(threadIdx.x);
    const int warp          = tid >> 5;
    const int lane          = tid & 31;

    const std::int32_t* block_table = block_tables;
    std::int64_t column_base        = column_begin;
    if constexpr (MultiBatch) {
        block_table =
            block_tables + static_cast<std::int64_t>(table_rows[batch]) * table_stride;
        column_base += static_cast<std::int64_t>(batch) * full_width;
    }
    q += static_cast<std::int64_t>(D) * Geometry::QHeads * column_base;

    const int row = batch * Geometry::KVHeads + kv_head;
    std::int32_t* keep_list_row = keep_tiles + static_cast<std::int64_t>(row) * keep_stride;
    std::int32_t* off_row       = split_off + static_cast<std::int64_t>(row) * (splits + 1);

    const int window     = (pos[column_base] >= 0) ? (static_cast<int>(pos[column_base]) + 1) : 0;
    const int active     = gqa_small_t_active_splits<Geometry, true>(window, splits, 1);
    const int tiles      = div_up(window, Bc);
    const int flag_words = (n_pow2 + 31) / 32;
    const int per_warp   = (n_pow2 + Warps - 1) / Warps;

    // Dynamic smem (launcher-sized, all below the consumer 99KB cap): per-warp
    // tile scores, per-warp 64-bin histograms, keep flags, per-32-tile
    // popcount prefix, q_mean, bin edges, the threshold-bin member list, and
    // the small cross-warp scratch.
    extern __shared__ std::uint8_t rank_smem[];
    float* warp_score = reinterpret_cast<float*>(rank_smem);
    int* warp_hist    = reinterpret_cast<int*>(warp_score + std::size_t(Warps) * per_warp);
    std::uint32_t* flags     = reinterpret_cast<std::uint32_t*>(warp_hist + Warps * 64);
    std::uint32_t* word_pref = reinterpret_cast<std::uint32_t*>(flags + flag_words);
    float* q_mean_s          = reinterpret_cast<float*>(word_pref + flag_words);
    float* bin_edge          = q_mean_s + D;
    int* member_list         = reinterpret_cast<int*>(bin_edge + 65);
    __shared__ int warp_cnt[Warps];
    __shared__ float warp_min[Warps], warp_max[Warps];
    __shared__ int global_hist[64];
    __shared__ int thr_bin_s, needed_s;
    __shared__ int rank_total_s;

    const auto write_neutral = [&] {
        keep_count[row] = 0;
        for (std::int32_t s = tid; s <= splits; s += 256) { off_row[s] = 0; }
    };
    if (window <= 0 || tiles == 0) { write_neutral(); return; }

    // q_mean[d] = the average of the group's BF16 query heads (T=1 step).
    if (tid < D) {
        float acc = 0.0f;
        for (int g = 0; g < Geometry::GroupSize; ++g) {
            acc += __bfloat162float(q[gqa_q_index<Geometry>(kv_head * Geometry::GroupSize + g,
                                                            tid, 0)]);
        }
        q_mean_s[tid] = acc / static_cast<float>(Geometry::GroupSize);
    }
    if (tid < Warps * 64) { warp_hist[tid] = 0; }
    if (tid < Warps) { warp_min[tid] = FLT_MAX; warp_max[tid] = -FLT_MAX; }
    __syncthreads();

    // Phase 1: per-tile proxy scores, one warp per tile, 8 d's per lane
    // (warp-local dot + shfl reduce; each warp only touches its own smem
    // slots, so no cross-warp barrier is needed inside the loop).
    for (int it = 0; it < per_warp; ++it) {
        const int kb = warp + it * Warps;
        if (kb >= tiles) { break; }
        const std::int64_t base = paged_kv_element_offset<4, Geometry::KVHeads>(
            block_table[(kb * Bc) >> kPagedKVPageShift], kv_head, 0, 0);
        const float* km = k_mean + base;
        float acc       = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int d = lane * 8 + j;
            acc += q_mean_s[d] * km[(d >> 2) * 4 + (d & 3)];
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) { acc += __shfl_down_sync(0xffffffffu, acc, o); }
        if (lane == 0) {
            warp_score[static_cast<std::size_t>(warp) * per_warp + it] = acc;
            warp_min[warp] = fminf(warp_min[warp], acc);
            warp_max[warp] = fmaxf(warp_max[warp], acc);
        }
    }
    __syncthreads();

    // CTA-wide score range -> 64 deterministic bin edges.
    if (warp == 0) {
        float lo = (lane < Warps) ? warp_min[lane] : 0.0f;
        float hi = (lane < Warps) ? warp_max[lane] : 0.0f;
#pragma unroll
        for (int o = 4; o > 0; o >>= 1) {
            lo = fmin(lo, __shfl_xor_sync(0xffffffffu, lo, o));
            hi = fmax(hi, __shfl_xor_sync(0xffffffffu, hi, o));
        }
        if (lane == 0) { warp_min[0] = lo; warp_max[0] = hi; }
    }
    __syncthreads();
    const float lo_all = warp_min[0];
    const float range  = warp_max[0] - lo_all;
    if (tid <= 64) {
        bin_edge[tid] = (range > 0.0f)
                            ? (lo_all + range * static_cast<float>(tid) / 64.0f)
                            : (lo_all + static_cast<float>(tid));
    }
    __syncthreads();

    // Per-tile coarse bin (deterministic; re-derived from the stored score so
    // no k_mean bytes are re-read).
    const auto tile_bin = [&](int kb) {
        const float s =
            warp_score[static_cast<std::size_t>(kb % Warps) * per_warp + kb / Warps];
        if (range <= 0.0f) { return 0; }
        const int b = static_cast<int>((s - lo_all) / range * 64.0f);
        return (b > 63) ? 63 : b;
    };

    // Phase 2: per-warp histogram (warp-local atomics) + threshold bin.
    for (int it = 0; it < per_warp; ++it) {
        const int kb = warp + it * Warps;
        if (kb >= tiles) { break; }
        atomicAdd(&warp_hist[warp * 64 + tile_bin(kb)], 1);
    }
    __syncthreads();
    if (tid < 64) {
        int acc = 0;
        for (int w = 0; w < Warps; ++w) { acc += warp_hist[w * 64 + tid]; }
        global_hist[tid] = acc;
    }
    __syncthreads();
    const int topk =
        (static_cast<int>(static_cast<std::size_t>(tiles) * keep_frac) < 1) ? 1
                                                                           : (static_cast<int>(static_cast<std::size_t>(tiles) *
                                                                                          keep_frac));
    if (tid == 0) {
        int cum = 0;
        thr_bin_s = 0;
        needed_s  = 0;
        for (int b = 63; b >= 0; --b) {
            const int next = cum + global_hist[b];
            if (next >= topk) { thr_bin_s = b; needed_s = topk - cum; break; }
            cum = next;
        }
    }
    __syncthreads();
    const int thr_bin = thr_bin_s;
    const int needed  = needed_s;

    // Phase 3: keep flags - every tile strictly above the threshold bin, plus
    // the first `needed` members of the threshold bin in tile-ascending order
    // (the lane-strided pass preserves the global tile order), plus the forced
    // sinks (first tiles) and recency window (last tiles).
    for (std::size_t i = 0; i < flag_words; ++i) { flags[i] = 0u; }
    __syncthreads();
    for (int kb = tid; kb < tiles; kb += 256) {
        if (tile_bin(kb) > thr_bin) { atomicOr(&flags[kb >> 5], 1u << (kb & 31)); }
    }
    __syncthreads();
    if (warp == 0) {
        int member_rank = 0;
        for (int kb = lane; kb < tiles; kb += 32) {
            if (tile_bin(kb) != thr_bin) { continue; }
            if (member_rank < MemberCap) { member_list[member_rank] = kb; }
            if (member_rank < needed) { atomicOr(&flags[kb >> 5], 1u << (kb & 31)); }
            ++member_rank;
        }
        // Beyond the cap the whole bin is kept (a deterministic over-keep in
        // the degenerate near-tie case; keeping more only moves toward exact).
        if (member_rank > MemberCap) {
            for (int kb = lane; kb < tiles; kb += 32) {
                if (tile_bin(kb) == thr_bin) { atomicOr(&flags[kb >> 5], 1u << (kb & 31)); }
            }
        }
    }
    __syncthreads();
    const int sinks =
        (static_cast<int>(static_cast<std::size_t>(tiles) * keep_frac * 0.2f) < 1) ? 1
                                                                                  : (static_cast<int>(static_cast<std::size_t>(tiles) *
                                                                                              keep_frac * 0.2f));
    const int win =
        (static_cast<int>(static_cast<std::size_t>(tiles) * keep_frac * 0.4f) < 1) ? 1
                                                                                  : (static_cast<int>(static_cast<std::size_t>(tiles) *
                                                                                              keep_frac * 0.4f));
    if (warp == 0) {
        for (int kb = lane; kb < tiles; kb += 32) {
            if (kb < sinks || kb >= tiles - win) {
                atomicOr(&flags[kb >> 5], 1u << (kb & 31));
            }
        }
    }
    __syncthreads();

    // Phase 4: keep count + keep list (thread-major deterministic order; the
    // partial pipeline only needs a stable per-CTA order).
    // Contiguous tile chunks per thread so the keep list is in ascending-tile
    // order (the split_off boundaries below are tile-index based, so the list
    // must be tile-sorted for each split's slice to hold exactly its kept tiles).
    const int chunk = (tiles + 255) / 256;
    const int kb_lo = tid * chunk;
    const int kb_hi = (kb_lo + chunk < tiles) ? (kb_lo + chunk) : tiles;
    int local_n = 0;
    for (int kb = kb_lo; kb < kb_hi; ++kb) {
        if ((flags[kb >> 5] >> (kb & 31)) & 1u) { ++local_n; }
    }
    int excl = 0;
    {
        int inc = local_n;
#pragma unroll
        for (int o = 1; o <= 16; o <<= 1) {
            const int t = __shfl_up_sync(0xffffffffu, inc, o);
            inc += (lane >= o) ? t : 0;
        }
        excl = inc - local_n;
    }
    {
        int wtot = local_n;
#pragma unroll
        for (int o = 16; o; o >>= 1) { wtot += __shfl_down_sync(0xffffffffu, wtot, o); }
        if (lane == 0) { warp_cnt[warp] = wtot; }
    }
    __syncthreads();
    int base = 0;
#pragma unroll
    for (int w = 0; w < warp; ++w) { base += warp_cnt[w]; }
    {
        int i2 = 0;
        for (int kb = kb_lo; kb < kb_hi; ++kb) {
            if ((flags[kb >> 5] >> (kb & 31)) & 1u) {
                keep_list_row[base + excl + i2++] = kb;
            }
        }
    }
    if (tid == 0) {
        int total = 0;
        for (int w = 0; w < Warps; ++w) { total += warp_cnt[w]; }
        keep_count[row] = total;
    }
    __syncthreads();

    // Phase 5: per-split keep prefixes. The split layout mirrors the partial
    // kernel (gqa_small_t_active_splits + its units_per_split choice): split s
    // processes the whole tiles covering its key range [s*units, s*units+units)
    // (tile-aligned in tile-split mode, floor-to-tile otherwise).
    const bool tile_split      = tiles >= active;
    const int units_per_split  = tile_split ? div_up(tiles, active) : div_up(window, active);
    for (std::size_t i = 0; i < flag_words; ++i) {
        word_pref[i] = (i * 32 < tiles) ? __popc(flags[i]) : 0u;
    }
    if (tid == 0) {
        std::uint32_t run = 0;
        const int words = (tiles + 31) / 32;
        for (int i = 0; i < words; ++i) {
            const std::uint32_t p = word_pref[i];
            word_pref[i]         = run;
            run += p;
        }
        for (int i = words; i < flag_words; ++i) { word_pref[i] = run; }
        rank_total_s = static_cast<int>(run);
    }
    __syncthreads();

    if (warp == 0) {
        for (std::int32_t s = lane; s <= active; s += 32) {
            const int first_tile =
                tile_split ? (s * units_per_split) : (s * units_per_split) / Bc;
            int kept_before = 0;
            if (first_tile < tiles) {
                const int w   = first_tile >> 5;
                const int rem = first_tile & 31;
                kept_before   = static_cast<int>(word_pref[w]);
                if (rem != 0) { kept_before += __popc(flags[w] & ((1u << rem) - 1u)); }
            } else {
                kept_before = rank_total_s;
            }
            off_row[s] = kept_before;
        }
        if (lane == 0) {
            off_row[active] = rank_total_s;
        }
    }
}

// Dynamic-smem size for the rank kernel (must match its extern __shared__ layout).
inline std::size_t gqa_s3_decode_rank_smem_bytes(std::int32_t n_pow2) {
    const std::size_t flag_words = (static_cast<std::size_t>(n_pow2) + 31) / 32;
    const std::size_t per_warp  = (static_cast<std::size_t>(n_pow2) + 7) / 8;
    return 8 * per_warp * 4 /*warp_score*/ + 8 * 64 * 4 /*warp_hist*/ +
           flag_words * 4 /*flags*/ + flag_words * 4 /*word_pref*/ + 256 * 4 /*q_mean*/ +
           65 * 4 /*bin_edge*/ + 256 * 4 /*member_list*/;
}

template <typename Geometry, int TokenTile, int WarpsPerCta, int MinBlocksPerSm, int KeyBlock,
          bool DynamicArena, bool MultiBatch, bool Masked, typename CacheInput,
          bool FusedFill = true, bool StrictPV = false, bool TileSkip = false>
__launch_bounds__(WarpsPerCta * 32, MinBlocksPerSm) __global__
    void gqa_attention_decode_nvfp4s3_tiled_kernel(
        const __nv_bfloat16* q, CacheInput input, const std::int32_t* pos,
        std::uint8_t* cache_k, std::uint8_t* cache_v, std::uint8_t* cache_k_scale,
        std::uint8_t* cache_v_scale, const std::int32_t* block_tables,
        const std::int32_t* valid_columns, const std::int32_t* table_rows,
        std::int32_t table_stride, std::int32_t full_width, std::int32_t column_begin,
        std::int32_t logical_capacity, float scale, __nv_bfloat16* partial_acc, float* partial_m,
        float* partial_l,
        const std::int32_t* keep_tiles, const std::int32_t* split_off, std::int32_t max_keep) {
    constexpr int Wc                   = WarpsPerCta;
    constexpr int RowCount             = TokenTile * Geometry::GroupSize;
    constexpr int RowTiles             = (RowCount + 15) / 16;
    constexpr int Br                   = RowTiles * 16;
    constexpr int Bc                   = KeyBlock;
    constexpr int D                    = kGqaHeadDim;
    constexpr int Threads              = Wc * 32;
    constexpr int Groups               = kGqaNvfp4Groups;
    constexpr int CodeW                = kGqaNvfp4CodeWidth;
    constexpr int QKNt                 = Bc / 8;
    constexpr int K64s                 = kGqaNvfp4K64;
    constexpr int PVKs                 = Bc / 16;
    constexpr int ConsumerWarpsPerTile = Wc / RowTiles;
    constexpr int PVNtPerWarp          = D / (ConsumerWarpsPerTile * 8);
    constexpr int PageIds              = 64;
    constexpr int ProducerThreads      = RowTiles * 32;
    constexpr int VLoaderThreads       = Threads - ProducerThreads;
    constexpr float Log2E              = 1.4426950408889634074f;
    constexpr unsigned FullMask        = 0xffffffffu;

    // S3 FP4-PV smem: P codes (32B rows for 64 keys, padded to 48; 16B rows for
    // 32 keys, padded to 32), P per-16-key scales, d-major V codes, d-major V
    // per-(d, 16-key-block) scales.
    // Strict-PV smem (packed V, double-buffered): V stays packed in smem (no
    // BF16 V arena); the consumer dequants each k16 PV fragment on the fly
    // from the codes + the staged e4m3 [kb][d] scale plane. The V code region
    // holds two buffers (tile parity): the next tile's cp.async targets the
    // other buffer, so the in-flight copy can never clobber the tile whose PV
    // mma is still reading its codes. K codes stay single-buffered exactly as
    // the FP4 path has always run them (the QK ldmatrix finishes before the
    // DRAM bytes land - the FP4-PV oracle tier proves it).
    constexpr int P4Row        = (Bc == 64) ? 48 : 32;
    constexpr int PBlocks      = Bc / 16;
    constexpr std::size_t ArenaBytes =
        StrictPV
            ? std::size_t(4 * Bc * CodeW + 2 * D * 4 + Bc * Groups + Br * Bc * 2)
            : std::size_t(2 * Bc * CodeW + Br * P4Row + Br * 4 + D * P4Row + D * 4 +
                          Bc * Groups);

    static_assert(TokenTile >= 1 && TokenTile <= 6);
    static_assert(Bc == 32 || Bc == 64);
    static_assert(RowTiles >= 1 && RowTiles <= 3);
    static_assert(Wc % RowTiles == 0);
    static_assert(PVNtPerWarp == 2 || PVNtPerWarp == 4 || PVNtPerWarp == 8 || PVNtPerWarp == 16);

    __shared__ __align__(16) std::uint8_t q_s[Br * CodeW];
    __shared__ __align__(16) std::uint8_t
        static_r_s[DynamicArena ? 16 : ArenaBytes];
    extern __shared__ __align__(16) std::uint8_t dynamic_r_s_nvfp4s3[];
    std::uint8_t* r_s        = DynamicArena ? dynamic_r_s_nvfp4s3 : static_r_s;
    std::uint8_t* q_codes    = q_s;
    std::uint8_t* k_codes    = r_s;
    std::uint8_t* v_codes    = r_s + Bc * CodeW;
    std::uint8_t* p4         = v_codes + Bc * CodeW;
    std::uint8_t* psf        = p4 + Br * P4Row;
    std::uint8_t* v_t        = psf + Br * 4;
    std::uint8_t* v_scales   = v_t + D * P4Row;
    std::uint8_t* k_scale_s  = v_scales + D * 4;
    std::uint8_t* v_scales_e4m3t = nullptr;
    __nv_bfloat16* p_s       = nullptr;
    if constexpr (StrictPV) {
        // v_codes[buf]: two V code buffers indexed by tile parity (kb & 1). The
        // staged e4m3 scale plane is double-buffered with the same parity: the
        // staging runs inside issue_kv_tile (one tile ahead), so a single plane
        // would be overwritten with the next tile's scales before the current
        // tile's PV dequant reads it (adjacent-tile scales are close, which is
        // why the error was a mild rel_L2 ~0.06, not a blow-up; single-tile CTAs
        // never saw it at all).
        v_scales_e4m3t =
            reinterpret_cast<std::uint8_t*>(v_codes + 2 * Bc * CodeW);
        k_scale_s     = reinterpret_cast<std::uint8_t*>(v_scales_e4m3t + 2 * D * 4);
        p_s           = reinterpret_cast<__nv_bfloat16*>(k_scale_s + Bc * Groups);
        p4            = nullptr;
        psf           = nullptr;
        v_t           = nullptr;
        v_scales      = nullptr;
    }
    __shared__ __align__(16) float alpha_s[Br];
    __shared__ __align__(16) std::uint8_t q_scale_s[Br * Groups];
    __shared__ std::int32_t physical_pages_s[PageIds];

    const int kv_head     = static_cast<int>(blockIdx.x);
    const int split       = static_cast<int>(blockIdx.y);
    const int batch       = MultiBatch ? static_cast<int>(blockIdx.z) : 0;
    const int split_count = static_cast<int>(gridDim.y);
    const int tid         = static_cast<int>(threadIdx.x);
    const int warp        = tid >> 5;
    const int lane        = tid & 31;

    int valid_tokens = TokenTile;
    if constexpr (Masked) {
        const int remaining = valid_columns[batch] - column_begin;
        valid_tokens        = remaining <= 0 ? 0 : (remaining < TokenTile ? remaining : TokenTile);
    }
    std::int64_t column_base = column_begin;
    if constexpr (MultiBatch) { column_base += static_cast<std::int64_t>(batch) * full_width; }
    q += static_cast<std::int64_t>(kGqaHeadDim) * Geometry::QHeads * column_base;
    pos += column_base;
    if constexpr (CacheInput::writes_cache) {
        input.k += static_cast<std::int64_t>(kGqaHeadDim) * Geometry::KVHeads * column_base;
        input.v += static_cast<std::int64_t>(kGqaHeadDim) * Geometry::KVHeads * column_base;
    }
    const int table_row = table_rows == nullptr ? 0 : table_rows[batch];
    const std::int32_t* block_table =
        block_tables + static_cast<std::int64_t>(table_row) * table_stride;
    if constexpr (MultiBatch) {
        partial_acc += static_cast<std::int64_t>(batch) * kGqaHeadDim * Geometry::QHeads *
                       TokenTile * split_count;
        partial_m += static_cast<std::int64_t>(batch) * Geometry::QHeads * TokenTile * split_count;
        partial_l += static_cast<std::int64_t>(batch) * Geometry::QHeads * TokenTile * split_count;
    }

    auto write_neutral = [&]() {
        for (int row = tid; row < RowCount; row += Threads) {
            int q_head = 0;
            int token  = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row, TokenTile, kv_head, q_head, token);
            if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] =
                    -CUDART_INF_F;
                partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] = 0.0f;
            }
        }
        for (int idx = tid; idx < RowCount * D; idx += Threads) {
            const int row = idx / D;
            const int d   = idx - row * D;
            int q_head    = 0;
            int token     = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row, TokenTile, kv_head, q_head, token);
            if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_acc[gqa_partial_acc_index<Geometry>(q_head, d, token, split, TokenTile)] =
                    __float2bfloat16(0.0f);
            }
        }
    };

    if (kv_head < 0 || kv_head >= Geometry::KVHeads || split_count <= 0) { return; }
    if (valid_tokens == 0) {
        write_neutral();
        return;
    }

    const std::int32_t first_pos = pos[0];
    const std::int32_t last_pos  = pos[TokenTile - 1];
    if (first_pos < 0 || last_pos < 0 || last_pos >= logical_capacity) {
        write_neutral();
        return;
    }

    const int window = last_pos + 1;
    const int active_split_count =
        gqa_small_t_active_splits<Geometry, true>(window, split_count, TokenTile);
    if (split >= active_split_count) { return; }

    // Sparge decode (TileSkip): the rank kernel has already computed the
    // per-(kv_head, batch) keep set; this split walks only its kept tiles
    // (list-order slice [off_cur, off_next)). A split with no kept tiles
    // writes a neutral partial and returns.
    std::int64_t skip_base = 0;
    int off_cur = 0, off_next = 0;
    if constexpr (TileSkip) {
        skip_base = static_cast<std::int64_t>(batch * Geometry::KVHeads + kv_head);
        const std::int32_t* off_row = split_off + skip_base * (split_count + 1);
        off_cur  = off_row[split];
        off_next = off_row[split + 1];
        if (off_cur == off_next) {
            write_neutral();
            return;
        }
    }

    const int logical_tiles = div_up(window, Bc);
    const bool tile_split   = logical_tiles >= active_split_count;
    const int units_per_split =
        tile_split ? div_up(logical_tiles, active_split_count) : div_up(window, active_split_count);
    const int split_start = split * units_per_split * (tile_split ? Bc : 1);
    const int split_limit = split_start + units_per_split * (tile_split ? Bc : 1);
    const int split_end   = (split_limit < window) ? split_limit : window;
    if (split_start >= split_end) {
        write_neutral();
        return;
    }
    const int first_tile = (split_start / Bc) * Bc;
    const int key_blocks = div_up(split_end - first_tile, Bc);
    const int first_page = first_tile >> kPagedKVPageShift;
    const int page_count = ((split_end - 1) >> kPagedKVPageShift) - first_page + 1;
    for (int page = tid; page < page_count; page += Threads) {
        physical_pages_s[page] = block_table[first_page + page];
    }

    if constexpr (CacheInput::writes_cache && FusedFill) {
        const int k_units = valid_tokens * Groups;
        for (int unit = tid; unit < k_units; unit += Threads) {
            const int token  = unit / Groups;
            const int grp    = unit - token * Groups;
            const int position = pos[token];
            if (position < split_start || position >= split_end) { continue; }
            const int physical_page = paged_kv_physical_page(block_table, position);
            const int page_offset   = position & kPagedKVPageMask;
            const int d0            = grp * kGqaNvfp4Group;
            std::uint32_t k_lo = 0, k_hi = 0;
            std::uint8_t k_sc = 0;
            gqa_nvfp4_quantize_bf16x16(&input.k[gqa_kv_new_index<Geometry>(kv_head, d0, token)],
                                       k_lo, k_hi, k_sc);
            const std::int64_t code =
                gqa_nvfp4_code_index<Geometry>(physical_page, kv_head, grp * 8, page_offset);
            *reinterpret_cast<std::uint32_t*>(cache_k + code)     = k_lo;
            *reinterpret_cast<std::uint32_t*>(cache_k + code + 4) = k_hi;
            cache_k_scale[gqa_nvfp4_scale_index<Geometry>(physical_page, kv_head, grp,
                                                           page_offset)] = k_sc;
        }
        // V side (S3 d-major layout): the 16-key block scale is cross-key state, so
        // tokens are processed in order (a new key joins an existing block and may
        // raise its max). Each launch's new keys fall inside one split's tail range,
        // so the per-token sync below keeps the rescale reads settled.
        for (int token = 0; token < valid_tokens; ++token) {
            const int position = pos[token];
            if (position < split_start || position >= split_end) { continue; }
            const int physical_page = paged_kv_physical_page(block_table, position);
            const int page_offset   = position & kPagedKVPageMask;
            const int kb            = page_offset >> 4;
            for (int dp = tid; dp < D / 2; dp += Threads) {
                const int d0            = dp * 2;
                const std::int64_t vidx  = gqa_kv_new_index<Geometry>(kv_head, d0, token);
                const float v0          = __bfloat162float(input.v[vidx]);
                const float v1          = __bfloat162float(input.v[vidx + 1]);
                // The code byte holds (d0 low, d1 high) and each d carries its own running
                // block scale, so rescale once per d-pair with per-nibble factors.
                const std::int64_t sc0_off =
                    gqa_s3_v_scale_index<Geometry>(physical_page, kv_head, d0, kb);
                const std::int64_t sc1_off =
                    gqa_s3_v_scale_index<Geometry>(physical_page, kv_head, d0 + 1, kb);
                const float s_cur0       = detail::decode_nvfp4_e4m3(cache_v_scale[sc0_off]);
                const float s_cur1       = detail::decode_nvfp4_e4m3(cache_v_scale[sc1_off]);
                const bool bumped0       = fmaxf(fabsf(v0), s_cur0 * 6.0f) > s_cur0 * 6.0f;
                const bool bumped1       = fmaxf(fabsf(v1), s_cur1 * 6.0f) > s_cur1 * 6.0f;
                if (bumped0) {
                    cache_v_scale[sc0_off] =
                        __nv_cvt_float_to_fp8(fmaxf(fabsf(v0), s_cur0 * 6.0f) / 6.0f,
                                              __NV_SATFINITE, __NV_E4M3);
                }
                if (bumped1) {
                    cache_v_scale[sc1_off] =
                        __nv_cvt_float_to_fp8(fmaxf(fabsf(v1), s_cur1 * 6.0f) / 6.0f,
                                              __NV_SATFINITE, __NV_E4M3);
                }
                const float s0_final     = detail::decode_nvfp4_e4m3(cache_v_scale[sc0_off]);
                const float s1_final     = detail::decode_nvfp4_e4m3(cache_v_scale[sc1_off]);
                const float rescale0     = (bumped0 && s_cur0 > 0.0f && s0_final > 0.0f)
                                               ? s_cur0 / s0_final
                                               : 1.0f;
                const float rescale1     = (bumped1 && s_cur1 > 0.0f && s1_final > 0.0f)
                                               ? s_cur1 / s1_final
                                               : 1.0f;
                if (rescale0 != 1.0f || rescale1 != 1.0f) {
                    const int page_base = position - page_offset;
                    for (int i = page_base + kb * 16; i < position; ++i) {
                        const std::int64_t co = gqa_nvfp4_code_index<Geometry>(
                            physical_page, kv_head, dp, i & kPagedKVPageMask);
                        const std::uint8_t byte = cache_v[co];
                        const float d0          = gqa_s3_e2m1_value(byte & 0x0fu);
                        const float d1          = gqa_s3_e2m1_value((byte >> 4) & 0x0fu);
                        cache_v[co] = gqa_s3_cvt_e2m1x2(d0 * rescale0, d1 * rescale1);
                    }
                }
                const float c0      = (s0_final > 0.0f) ? v0 / s0_final : 0.0f;
                const float c1      = (s1_final > 0.0f) ? v1 / s1_final : 0.0f;
                cache_v[gqa_nvfp4_code_index<Geometry>(physical_page, kv_head, dp, page_offset)] =
                    gqa_s3_cvt_e2m1x2(c0, c1);
            }
            __syncthreads();
        }
    }

    for (int i = tid; i < Br * CodeW; i += Threads) { q_codes[i] = 0; }
    for (int i = tid; i < Br * Groups; i += Threads) { q_scale_s[i] = 0; }
    if constexpr (!StrictPV) {
        // The k64 PV mma reads 32B per P/V row; zero the unused halves (Bc=32) once.
        for (int i = tid; i < Br * P4Row; i += Threads) { p4[i] = 0; }
        for (int i = tid; i < D * P4Row; i += Threads) { v_t[i] = 0; }
    }
    __syncthreads();

    for (int unit = tid; unit < RowCount * Groups; unit += Threads) {
        const int row = unit / Groups;
        const int grp = unit - row * Groups;
        int q_head    = 0;
        int token     = 0;
        gqa_small_t_tc_row_to_qt<Geometry>(row, TokenTile, kv_head, q_head, token);
        std::uint32_t lo = 0, hi = 0;
        std::uint8_t sc = 0;
        if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
            gqa_nvfp4_quantize_bf16x16(&q[gqa_q_index<Geometry>(q_head, grp * kGqaNvfp4Group, token)],
                                       lo, hi, sc);
        }
        const int logical = grp * 8;
        const int phys    = gqa_nvfp4_swizzle_byte(row, logical);
        *reinterpret_cast<std::uint32_t*>(q_codes + row * CodeW + phys)     = lo;
        *reinterpret_cast<std::uint32_t*>(q_codes + row * CodeW + phys + 4) = hi;
        q_scale_s[row * Groups + grp]                                        = sc;
    }
    __syncthreads();

    const int gid = lane >> 2;
    const int lid = lane & 3;

    const int a_matrix      = lane >> 3;
    const int a_row_offset  = (lane & 7) + ((a_matrix & 1) << 3);
    const int a_column_byte = (a_matrix >> 1) * 16;
    const int b_row_offset  = lane & 7;
    const int b_column_byte = ((lane >> 3) & 1) * 16;
    const int sfa_row       = ((lane & 1) << 3) | (lane >> 2);
    const int sfb_row       = lane >> 2;
    // BF16 PV mma lane offsets (strict path only).
    [[maybe_unused]] const int a_coloff = (a_matrix >> 1) << 3;
    [[maybe_unused]] const int b_rin    = lane & 7;
    [[maybe_unused]] const int b_koff   = ((lane >> 3) & 1) << 3;

    float acc[PVNtPerWarp][4];
#pragma unroll
    for (int n = 0; n < PVNtPerWarp; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[n][i] = 0.0f; }
    }

    float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F;
    float l0 = 0.0f, l1 = 0.0f;

    auto issue_kv_tile = [&](int tile_k0, int physical_page, int buf) {
        for (int key_l = tid; key_l < Bc; key_l += Threads) {
            const int key = tile_k0 + key_l;
            if ((TileSkip || (key >= split_start && key < split_end))) {
                const std::int64_t off = gqa_nvfp4_scale_index<Geometry>(
                    physical_page, kv_head, 0, key & kPagedKVPageMask);
                ninfer::ops::cp_async<16>(&k_scale_s[key_l * Groups], &cache_k_scale[off]);
            } else {
                store_vec(&k_scale_s[key_l * Groups], make_int4(0, 0, 0, 0));
            }
        }
        for (int chunk = tid; chunk < Bc * (CodeW / 16); chunk += Threads) {
            const int key_l       = chunk / (CodeW / 16);
            const int seg         = chunk - key_l * (CodeW / 16);
            const int logical     = seg * 16;
            const int key         = tile_k0 + key_l;
            const int phys        = gqa_nvfp4_swizzle_byte(key_l, logical);
            std::uint8_t* k_dst   = &k_codes[key_l * CodeW + phys];
            // Double-buffered V (strict) vs single-buffered (FP4): the buffer
            // offset is a compile-time zero on the FP4 path.
            const std::size_t vbuf_off = StrictPV ? static_cast<std::size_t>(buf) * Bc * CodeW : 0;
            std::uint8_t* v_dst   = &v_codes[vbuf_off + key_l * CodeW + logical];
            if ((TileSkip || (key >= split_start && key < split_end))) {
                const std::int64_t off = gqa_nvfp4_code_index<Geometry>(
                    physical_page, kv_head, logical, key & kPagedKVPageMask);
                ninfer::ops::cp_async<16>(k_dst, &cache_k[off]);
                ninfer::ops::cp_async<16>(v_dst, &cache_v[off]);
            } else {
                store_vec(k_dst, make_int4(0, 0, 0, 0));
                store_vec(v_dst, make_int4(0, 0, 0, 0));
            }
        }
        if constexpr (StrictPV) {
            // Transposed e4m3 staging [kb 4][d 256] (1 KB, 3/4 smaller than the
            // f32 plane): a thread takes one 16B DRAM chunk (d0..d0+3 x in-page
            // block 0..3; the plane is d-major [d][4]) and scatters the tile's
            // in-page blocks into tile-local rows so the loader reads 4B-aligned
            // words with no bank spread.
            for (int c = tid; c < D / 4; c += Threads) {
                const int d0 = c * 4;
                std::uint32_t w[4];
                const std::int64_t off = gqa_s3_v_scale_index<Geometry>(
                    physical_page, kv_head, d0, 0);
                *reinterpret_cast<int4*>(w) =
                    *reinterpret_cast<const int4*>(&cache_v_scale[off]);
                // Plane parity matches the V code buffer: tile kb's scales
                // land in plane (buf & 1), so the current tile's PV dequant
                // reads its own scales even though staging runs ahead.
                std::uint8_t* plane = v_scales_e4m3t + (buf ? D * 4 : 0u);
#pragma unroll
                for (int kb = 0; kb < 4; ++kb) {
                    const int key = tile_k0 + kb * 16;
                    std::uint8_t* row = plane + kb * D + d0;
                    if (kb < PBlocks && (TileSkip || (key + 16 > split_start && key < split_end))) {
                        // Tile-local kb -> in-page 16-key block: a Bc=32 tile may
                        // start at in-page offset 32, where its kb 0,1 hold
                        // in-page blocks 2,3 (the plane is written per in-page
                        // block by the fill kernels).
                        const int key_block = (key & kPagedKVPageMask) >> 4;
                        // Word w[i] = [in-page kb 0..3] of d = d0+i: repack the
                        // tile's 4 in-page blocks into one 4B row per d (4 STS.32
                        // with no bank spread, vs 16 strided byte stores).
                        std::uint32_t r[4];
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            r[i] = (w[i] >> (8 * key_block)) & 0xFFu;
                            // The 4 d-bytes of row i come from words 0..3 (d0..d0+3
                            // at this in-page block); pack low-to-high.
                        }
                        std::uint32_t packed = r[0] | (r[1] << 8) | (r[2] << 16) | (r[3] << 24);
                        *reinterpret_cast<std::uint32_t*>(row) = packed;
                    } else {
#pragma unroll
                        for (int i = 0; i < 4; ++i) { row[i] = 0; }
                    }
                }
            }
        } else {
            for (int d = tid; d < D; d += Threads) {
#pragma unroll
                for (int kb = 0; kb < 4; ++kb) {
                    const int key = tile_k0 + kb * 16;
                    if (kb < PBlocks && (TileSkip || (key + 16 > split_start && key < split_end))) {
                        const int key_block = (key & kPagedKVPageMask) >> 4;
                        v_scales[d * 4 + kb] =
                            cache_v_scale[gqa_s3_v_scale_index<Geometry>(
                                physical_page, kv_head, d, key_block)];
                    } else {
                        v_scales[d * 4 + kb] = 0;
                    }
                }
            }
        }
        ninfer::ops::cp_commit();
    };

    int physical_page;
    if constexpr (TileSkip) {
        const int gkb = keep_tiles[skip_base * max_keep + off_cur];
        physical_page = physical_pages_s[(gkb * Bc >> kPagedKVPageShift) - first_page];
    } else {
        physical_page = physical_pages_s[0];
    }
    issue_kv_tile(TileSkip ? (keep_tiles[skip_base * max_keep + off_cur] * Bc) : first_tile,
                  physical_page, 0);
    ninfer::ops::cp_wait<0>();
    __syncthreads();

    const int tile_last = TileSkip ? (off_next - off_cur - 1) : (key_blocks - 1);
    for (int kb = 0; kb <= tile_last; ++kb) {
        const int k0 = TileSkip
                           ? (keep_tiles[skip_base * max_keep + off_cur + kb] * Bc)
                           : (first_tile + kb * Bc);

        if (warp < RowTiles) {
            const int producer_row_base = warp * 16;
            float score[QKNt][4];
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                score[nt][0] = 0.0f;
                score[nt][1] = 0.0f;
                score[nt][2] = 0.0f;
                score[nt][3] = 0.0f;
            }

#pragma unroll
            for (int k64 = 0; k64 < K64s; ++k64) {
                const int row           = producer_row_base + a_row_offset;
                const int logical_byte  = k64 * 32 + a_column_byte;
                const int physical_byte = gqa_nvfp4_swizzle_byte(row, logical_byte);
                unsigned af[4];
                ldmatrix_x4(af[0], af[1], af[2], af[3],
                            smem_addr(q_codes + row * CodeW + physical_byte));
                const int scale_row = producer_row_base + sfa_row;
                unsigned sfa        = 0;
                if ((lane & 2) == 0 && scale_row < Br) {
                    sfa = *reinterpret_cast<const unsigned*>(
                        &q_scale_s[scale_row * Groups + k64 * 4]);
                }
#pragma unroll
                for (int nt = 0; nt < QKNt; ++nt) {
                    const int brow          = nt * 8 + b_row_offset;
                    const int b_logical     = k64 * 32 + b_column_byte;
                    const int b_physical    = gqa_nvfp4_swizzle_byte(brow, b_logical);
                    unsigned bf[2];
                    ldmatrix_x2(bf[0], bf[1], smem_addr(k_codes + brow * CodeW + b_physical));
                    const int b_scale_row = nt * 8 + sfb_row;
                    unsigned sfb          = 0;
                    if ((lane & 3) == 0) {
                        sfb = *reinterpret_cast<const unsigned*>(
                            &k_scale_s[b_scale_row * Groups + k64 * 4]);
                    }
                    mma_nvfp4_e4m3(score[nt][0], score[nt][1], score[nt][2], score[nt][3], af[0],
                                   af[1], af[2], af[3], bf[0], bf[1], sfa, sfb);
                }
            }

            const int row0 = producer_row_base + gid;
            const int row1 = row0 + 8;
            int q_head0 = 0, token0 = 0, q_head1 = 0, token1 = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row0, TokenTile, kv_head, q_head0, token0);
            gqa_small_t_tc_row_to_qt<Geometry>(row1, TokenTile, kv_head, q_head1, token1);
            const int qabs0 = (row0 < RowCount) ? pos[token0] : -1;
            const int qabs1 = (row1 < RowCount) ? pos[token1] : -1;

            if constexpr (StrictPV) {
                // Strict PV: tile-wide max, BF16 P, plain exp2 (no S3 e2m1 grid or
                // amplification); the PV mma below consumes p_s + dequantized BF16 V.
                float bm0 = -CUDART_INF_F, bm1 = -CUDART_INF_F;
#pragma unroll
                for (int nt = 0; nt < QKNt; ++nt) {
                    const int col0 = nt * 8 + 2 * lid;
                    const int key0 = k0 + col0;
                    const int key1 = key0 + 1;
                    score[nt][0] = (row0 < RowCount && (TileSkip || (key0 >= split_start && key0 < split_end)) &&
                                    key0 <= qabs0)
                                       ? score[nt][0] * scale
                                       : -CUDART_INF_F;
                    score[nt][1] = (row0 < RowCount && (TileSkip || (key1 >= split_start && key1 < split_end)) &&
                                    key1 <= qabs0)
                                       ? score[nt][1] * scale
                                       : -CUDART_INF_F;
                    score[nt][2] = (row1 < RowCount && (TileSkip || (key0 >= split_start && key0 < split_end)) &&
                                    key0 <= qabs1)
                                       ? score[nt][2] * scale
                                       : -CUDART_INF_F;
                    score[nt][3] = (row1 < RowCount && (TileSkip || (key1 >= split_start && key1 < split_end)) &&
                                    key1 <= qabs1)
                                       ? score[nt][3] * scale
                                       : -CUDART_INF_F;
                    bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
                    bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
                }
                bm0 = warp_max<4>(bm0, FullMask);
                bm1 = warp_max<4>(bm1, FullMask);

                const float nm0    = fmaxf(m0, bm0);
                const float nm1    = fmaxf(m1, bm1);
                const float alpha0 = (m0 == -CUDART_INF_F) ? 0.0f : exp2_approx((m0 - nm0) * Log2E);
                const float alpha1 = (m1 == -CUDART_INF_F) ? 0.0f : exp2_approx((m1 - nm1) * Log2E);

                float bl0 = 0.0f, bl1 = 0.0f;
                __nv_bfloat16* p_sw = &p_s[producer_row_base * Bc];
#pragma unroll
                for (int nt = 0; nt < QKNt; ++nt) {
                    const int col0 = nt * 8 + 2 * lid;
                    const int col1 = col0 + 1;
                    const float p00 = (nm0 > -CUDART_INF_F && score[nt][0] > -CUDART_INF_F)
                                          ? exp2_approx((score[nt][0] - nm0) * Log2E)
                                          : 0.0f;
                    const float p01 = (nm0 > -CUDART_INF_F && score[nt][1] > -CUDART_INF_F)
                                          ? exp2_approx((score[nt][1] - nm0) * Log2E)
                                          : 0.0f;
                    const float p10 = (nm1 > -CUDART_INF_F && score[nt][2] > -CUDART_INF_F)
                                          ? exp2_approx((score[nt][2] - nm1) * Log2E)
                                          : 0.0f;
                    const float p11 = (nm1 > -CUDART_INF_F && score[nt][3] > -CUDART_INF_F)
                                          ? exp2_approx((score[nt][3] - nm1) * Log2E)
                                          : 0.0f;
                    bl0 += p00 + p01;
                    bl1 += p10 + p11;
                    p_sw[gid * Bc + gqa_small_t_tc_swz32(gid, col0)]           = __float2bfloat16(p00);
                    p_sw[gid * Bc + gqa_small_t_tc_swz32(gid, col1)]           = __float2bfloat16(p01);
                    p_sw[(gid + 8) * Bc + gqa_small_t_tc_swz32(gid + 8, col0)] =
                        __float2bfloat16(p10);
                    p_sw[(gid + 8) * Bc + gqa_small_t_tc_swz32(gid + 8, col1)] =
                        __float2bfloat16(p11);
                }
                bl0 = warp_sum<4>(bl0, FullMask);
                bl1 = warp_sum<4>(bl1, FullMask);

                l0 = l0 * alpha0 + bl0;
                l1 = l1 * alpha1 + bl1;
                m0 = nm0;
                m1 = nm1;
                if (lid == 0) {
                    alpha_s[row0] = alpha0;
                    alpha_s[row1] = alpha1;
                }
            } else {
            // Per-16-key raw-score maxima (after masking). Each 16-key block spans
            // two nt tiles; each lane holds 4 of the 16 keys per row.
            float bm0_blk[PBlocks], bm1_blk[PBlocks];
#pragma unroll
            for (int nb = 0; nb < PBlocks; ++nb) {
                const int nt = 2 * nb;
                const int col0 = nt * 8 + 2 * lid;
                const int key0 = k0 + col0;
                const int key1 = key0 + 1;
                const int key2 = k0 + (nt + 1) * 8 + 2 * lid;
                const int key3 = key2 + 1;
                score[nt][0]   = (row0 < RowCount && (TileSkip || (key0 >= split_start && key0 < split_end)) &&
                                  key0 <= qabs0)
                                     ? score[nt][0] * scale
                                     : -CUDART_INF_F;
                score[nt][1]   = (row0 < RowCount && (TileSkip || (key1 >= split_start && key1 < split_end)) &&
                                  key1 <= qabs0)
                                     ? score[nt][1] * scale
                                     : -CUDART_INF_F;
                score[nt + 1][0] = (row0 < RowCount && (TileSkip || (key2 >= split_start && key2 < split_end)) &&
                                    key2 <= qabs0)
                                        ? score[nt + 1][0] * scale
                                        : -CUDART_INF_F;
                score[nt + 1][1] = (row0 < RowCount && (TileSkip || (key3 >= split_start && key3 < split_end)) &&
                                    key3 <= qabs0)
                                        ? score[nt + 1][1] * scale
                                        : -CUDART_INF_F;
                score[nt][2]   = (row1 < RowCount && (TileSkip || (key0 >= split_start && key0 < split_end)) &&
                                  key0 <= qabs1)
                                     ? score[nt][2] * scale
                                     : -CUDART_INF_F;
                score[nt][3]   = (row1 < RowCount && (TileSkip || (key1 >= split_start && key1 < split_end)) &&
                                  key1 <= qabs1)
                                     ? score[nt][3] * scale
                                     : -CUDART_INF_F;
                score[nt + 1][2] = (row1 < RowCount && (TileSkip || (key2 >= split_start && key2 < split_end)) &&
                                    key2 <= qabs1)
                                        ? score[nt + 1][2] * scale
                                        : -CUDART_INF_F;
                score[nt + 1][3] = (row1 < RowCount && (TileSkip || (key3 >= split_start && key3 < split_end)) &&
                                    key3 <= qabs1)
                                        ? score[nt + 1][3] * scale
                                        : -CUDART_INF_F;
                bm0_blk[nb] = fmaxf(fmaxf(score[nt][0], score[nt][1]),
                                    fmaxf(score[nt + 1][0], score[nt + 1][1]));
                bm1_blk[nb] = fmaxf(fmaxf(score[nt][2], score[nt][3]),
                                    fmaxf(score[nt + 1][2], score[nt + 1][3]));
            }
#pragma unroll
            for (int nb = 0; nb < PBlocks; ++nb) {
                bm0_blk[nb] = warp_max<4>(bm0_blk[nb], FullMask);
                bm1_blk[nb] = warp_max<4>(bm1_blk[nb], FullMask);
            }
            float bm0 = bm0_blk[0], bm1 = bm1_blk[0];
#pragma unroll
            for (int nb = 1; nb < PBlocks; ++nb) {
                bm0 = fmaxf(bm0, bm0_blk[nb]);
                bm1 = fmaxf(bm1, bm1_blk[nb]);
            }

            const float nm0    = fmaxf(m0, bm0);
            const float nm1    = fmaxf(m1, bm1);
            const float alpha0 = (m0 == -CUDART_INF_F) ? 0.0f : exp2_approx((m0 - nm0) * Log2E);
            const float alpha1 = (m1 == -CUDART_INF_F) ? 0.0f : exp2_approx((m1 - nm1) * Log2E);
            const float nm0_scaled = nm0 * Log2E;
            const float nm1_scaled = nm1 * Log2E;
            const float amp        = -kGqaS3Fp8ScaleLog2; // log2(448 * 6)
            const float sbl        = -kGqaS3Fp8ScaleLog2 + kGqaS3Fp4ScaleLog2; // log2(448)

            float bl0 = 0.0f, bl1 = 0.0f;
#pragma unroll
            for (int nb = 0; nb < PBlocks; ++nb) {
                const float sb0f = (bm0_blk[nb] == -CUDART_INF_F)
                                       ? 0.0f
                                       : exp2_approx(
                                             __fmaf_rn(bm0_blk[nb], Log2E, -nm0_scaled) + sbl);
                const float sb1f = (bm1_blk[nb] == -CUDART_INF_F)
                                       ? 0.0f
                                       : exp2_approx(
                                             __fmaf_rn(bm1_blk[nb], Log2E, -nm1_scaled) + sbl);
                const std::uint8_t sc0 =
                    (sb0f == 0.0f) ? 0 : __nv_cvt_float_to_fp8(sb0f, __NV_SATFINITE, __NV_E4M3);
                const std::uint8_t sc1 =
                    (sb1f == 0.0f) ? 0 : __nv_cvt_float_to_fp8(sb1f, __NV_SATFINITE, __NV_E4M3);
                const float dec0 = detail::decode_nvfp4_e4m3(sc0);
                const float dec1 = detail::decode_nvfp4_e4m3(sc1);
                const int nt     = 2 * nb;
                const float pa0  = (nm0 > -CUDART_INF_F && score[nt][0] > -CUDART_INF_F)
                                        ? exp2_approx(__fmaf_rn(score[nt][0], Log2E, -nm0_scaled) +
                                                      amp)
                                        : 0.0f;
                const float pa1  = (nm0 > -CUDART_INF_F && score[nt][1] > -CUDART_INF_F)
                                        ? exp2_approx(__fmaf_rn(score[nt][1], Log2E, -nm0_scaled) +
                                                      amp)
                                        : 0.0f;
                const float pa2  = (nm0 > -CUDART_INF_F && score[nt + 1][0] > -CUDART_INF_F)
                                        ? exp2_approx(__fmaf_rn(score[nt + 1][0], Log2E,
                                                                -nm0_scaled) +
                                                      amp)
                                        : 0.0f;
                const float pa3  = (nm0 > -CUDART_INF_F && score[nt + 1][1] > -CUDART_INF_F)
                                        ? exp2_approx(__fmaf_rn(score[nt + 1][1], Log2E,
                                                                -nm0_scaled) +
                                                      amp)
                                        : 0.0f;
                const float pb0  = (nm1 > -CUDART_INF_F && score[nt][2] > -CUDART_INF_F)
                                        ? exp2_approx(__fmaf_rn(score[nt][2], Log2E, -nm1_scaled) +
                                                      amp)
                                        : 0.0f;
                const float pb1  = (nm1 > -CUDART_INF_F && score[nt][3] > -CUDART_INF_F)
                                        ? exp2_approx(__fmaf_rn(score[nt][3], Log2E, -nm1_scaled) +
                                                      amp)
                                        : 0.0f;
                const float pb2  = (nm1 > -CUDART_INF_F && score[nt + 1][2] > -CUDART_INF_F)
                                        ? exp2_approx(__fmaf_rn(score[nt + 1][2], Log2E,
                                                                -nm1_scaled) +
                                                      amp)
                                        : 0.0f;
                const float pb3  = (nm1 > -CUDART_INF_F && score[nt + 1][3] > -CUDART_INF_F)
                                        ? exp2_approx(__fmaf_rn(score[nt + 1][3], Log2E,
                                                                -nm1_scaled) +
                                                      amp)
                                        : 0.0f;
                const float qa0 = (dec0 > 0.0f) ? pa0 / dec0 : 0.0f;
                const float qa1 = (dec0 > 0.0f) ? pa1 / dec0 : 0.0f;
                const float qa2 = (dec0 > 0.0f) ? pa2 / dec0 : 0.0f;
                const float qa3 = (dec0 > 0.0f) ? pa3 / dec0 : 0.0f;
                const float qb0 = (dec1 > 0.0f) ? pb0 / dec1 : 0.0f;
                const float qb1 = (dec1 > 0.0f) ? pb1 / dec1 : 0.0f;
                const float qb2 = (dec1 > 0.0f) ? pb2 / dec1 : 0.0f;
                const float qb3 = (dec1 > 0.0f) ? pb3 / dec1 : 0.0f;
                p4[row0 * P4Row + nb * 8 + lid]   = gqa_s3_cvt_e2m1x2(qa0, qa1);
                p4[row0 * P4Row + nb * 8 + 4 + lid] = gqa_s3_cvt_e2m1x2(qa2, qa3);
                p4[row1 * P4Row + nb * 8 + lid]   = gqa_s3_cvt_e2m1x2(qb0, qb1);
                p4[row1 * P4Row + nb * 8 + 4 + lid] = gqa_s3_cvt_e2m1x2(qb2, qb3);
                if (lid == 0) {
                    psf[row0 * 4 + nb] = sc0;
                    psf[row1 * 4 + nb] = sc1;
                    // Bc=32: the k64 PV mma still reads four e4m3 SFA per row; the
                    // slots past PBlocks belong to zero-padded keys and must be 0.
                    for (int pb = PBlocks; pb < 4; ++pb) {
                        psf[row0 * 4 + pb] = 0;
                        psf[row1 * 4 + pb] = 0;
                    }
                }
                bl0 += pa0 + pa1 + pa2 + pa3;
                bl1 += pb0 + pb1 + pb2 + pb3;
            }
            bl0 = warp_sum<4>(bl0, FullMask);
            bl1 = warp_sum<4>(bl1, FullMask);

            l0 = l0 * alpha0 + bl0;
            l1 = l1 * alpha1 + bl1;
            m0 = nm0;
            m1 = nm1;
            if (lid == 0) {
                alpha_s[row0] = alpha0;
                alpha_s[row1] = alpha1;
            }
            } // else (S3 FP4-PV path)
        } else {
            if constexpr (StrictPV) {
                // Packed-V strict: the cp.async'd V codes stay packed in v_codes;
                // the consumer dequants each k16 PV fragment on the fly (no
                // producer dequant phase, no BF16 V arena in the smem layout).
            } else {
                const int loader_tid = tid - ProducerThreads;
                // One loader thread per d-row: write a contiguous 32-byte v_t row so
                // the store hits distinct SMEM banks. The original unit-strided store
                // placed every lane's write on the same bank (32-way conflict).
                for (int d = loader_tid; d < D; d += VLoaderThreads) {
                const int dp = d >> 1;
                const int sh = (d & 1) * 4;
#pragma unroll 1
                for (int kp = 0; kp < Bc / 2; ++kp) {
                    const std::uint8_t b0 = v_codes[2 * kp * CodeW + dp];
                    const std::uint8_t b1 = v_codes[(2 * kp + 1) * CodeW + dp];
                    v_t[d * P4Row + kp]   = (static_cast<std::uint8_t>(((b1 >> sh) & 0x0Fu) << 4) |
                                           static_cast<std::uint8_t>((b0 >> sh) & 0x0Fu));
                }
            }
            }
        }
        __syncthreads();

        const bool has_next =
            TileSkip ? (off_cur + kb + 1 < off_next) : (kb + 1 < key_blocks);
        if (has_next) {
            const int next_k0 =
                TileSkip ? (keep_tiles[skip_base * max_keep + off_cur + kb + 1] * Bc) : (k0 + Bc);
            if constexpr (TileSkip) {
                // The keep-list order is not monotonic in key index, so the page
                // is looked up per entry (the contiguous path only updates it on
                // page boundaries).
                physical_page =
                    physical_pages_s[(next_k0 >> kPagedKVPageShift) - first_page];
            } else if ((next_k0 & kPagedKVPageMask) == 0) {
                physical_page =
                    physical_pages_s[(next_k0 >> kPagedKVPageShift) - first_page];
            }
            issue_kv_tile(next_k0, physical_page, (kb + 1) & 1);
        }

        const int consumer_tile     = warp % RowTiles;
        const int consumer_slice    = warp / RowTiles;
        const int consumer_row_base = consumer_tile * 16;
        const float alpha0          = alpha_s[consumer_row_base + gid];
        const float alpha1          = alpha_s[consumer_row_base + gid + 8];
#pragma unroll
        for (int n = 0; n < PVNtPerWarp; ++n) {
            acc[n][0] *= alpha0;
            acc[n][1] *= alpha0;
            acc[n][2] *= alpha1;
            acc[n][3] *= alpha1;
        }

        if constexpr (StrictPV) {
            // BF16 PV: p_s (swizzled BF16 rows) x V dequantized on the fly from
            // the packed FP4 codes + the staged e4m3 [kb][d] scale plane. The
            // m16n8k16 B fragment per lane holds keys k*16 + 2*(lane&3) +
            // {0,1,8,9} at d = global_n*8 + lane/4; each element is the exact
            // e2m1 x e4m3 product rounded to BF16 (bit-identical to the old
            // producer-side dequant), so numerics are unchanged.
            __nv_bfloat16* p_consumer = &p_s[consumer_row_base * Bc];
            const int d_key0 = 2 * (lane & 3);
#pragma unroll
            for (int n = 0; n < PVNtPerWarp; ++n) {
                const int global_n = consumer_slice * PVNtPerWarp + n;
                const int d        = global_n * 8 + (lane >> 2);
                const int dbyte    = d >> 1;
                const int nib      = (d & 1) << 2;
#pragma unroll
                for (int k = 0; k < PVKs; ++k) {
                    const float sv = detail::decode_nvfp4_e4m3(
                        v_scales_e4m3t[(kb & 1) * D * 4 + k * D + d]);
                    unsigned pf[4];
                    const int pcol = k * 16 + a_coloff;
                    ldmatrix_x4(pf[0], pf[1], pf[2], pf[3],
                                smem_addr(&p_consumer[a_row_offset * Bc +
                                                  gqa_small_t_tc_swz32(a_row_offset, pcol)]));
                    const std::uint8_t* v_base = &v_codes[(kb & 1) * Bc * CodeW];
                    const std::uint8_t* ck0 = &v_base[(k * 16 + d_key0) * CodeW + dbyte];
                    const std::uint8_t* ck1  = &v_base[(k * 16 + d_key0 + 1) * CodeW + dbyte];
                    const std::uint8_t* ck8  = &v_base[(k * 16 + d_key0 + 8) * CodeW + dbyte];
                    const std::uint8_t* ck9  = &v_base[(k * 16 + d_key0 + 9) * CodeW + dbyte];
                    const unsigned vf0 =
                        pack_bf16x2(gqa_s3_e2m1_value_alu(((*ck0 >> nib) & 0x0Fu)) * sv,
                                    gqa_s3_e2m1_value_alu(((*ck1 >> nib) & 0x0Fu)) * sv);
                    const unsigned vf1 =
                        pack_bf16x2(gqa_s3_e2m1_value_alu(((*ck8 >> nib) & 0x0Fu)) * sv,
                                    gqa_s3_e2m1_value_alu(((*ck9 >> nib) & 0x0Fu)) * sv);
                    mma_bf16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], pf[0], pf[1], pf[2], pf[3],
                             vf0, vf1);
                }
            }
        } else {
        // A: P codes (16 rows x 32B for the k64 mma); SFA: per-16-key e4m3 row.
        unsigned pf[4];
        ldmatrix_x4(pf[0], pf[1], pf[2], pf[3],
                    smem_addr(p4 + (consumer_row_base + a_row_offset) * P4Row + a_column_byte));
        unsigned sfa = 0;
        if ((lane & 2) == 0) {
                sfa = *reinterpret_cast<const unsigned*>(
                    &psf[(consumer_row_base + sfa_row) * 4]);
        }
#pragma unroll
        for (int n = 0; n < PVNtPerWarp; ++n) {
            const int global_n = consumer_slice * PVNtPerWarp + n;
            const int vrow     = global_n * 8 + b_row_offset;
            unsigned vf[2];
            ldmatrix_x2(vf[0], vf[1],
                        smem_addr(v_t + vrow * P4Row + b_column_byte));
            const int vsf_row = global_n * 8 + sfb_row;
            unsigned sfb      = 0;
            if ((lane & 3) == 0) {
                sfb = *reinterpret_cast<const unsigned*>(&v_scales[vsf_row * 4]);
            }
            mma_nvfp4_e4m3(acc[n][0], acc[n][1], acc[n][2], acc[n][3], pf[0], pf[1], pf[2], pf[3],
                           vf[0], vf[1], sfa, sfb);
        }
        }
        if (has_next) { ninfer::ops::cp_wait<0>(); }
        __syncthreads();
    }

    if (warp < RowTiles && lid == 0) {
        const int row0 = warp * 16 + gid;
        const int row1 = row0 + 8;
        if (row0 < RowCount) {
            int q_head = 0;
            int token  = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row0, TokenTile, kv_head, q_head, token);
            partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] = m0;
            partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] = l0;
        }
        if (row1 < RowCount) {
            int q_head = 0;
            int token  = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row1, TokenTile, kv_head, q_head, token);
            partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] = m1;
            partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, TokenTile)] = l1;
        }
    }

#pragma unroll
    for (int n = 0; n < PVNtPerWarp; ++n) {
        const int consumer_tile     = warp % RowTiles;
        const int consumer_slice    = warp / RowTiles;
        const int consumer_row_base = consumer_tile * 16;
        const int d0                = (consumer_slice * PVNtPerWarp + n) * 8 + 2 * lid;
        const int row0              = consumer_row_base + gid;
        const int row1              = row0 + 8;
        if (row0 < RowCount) {
            int q_head = 0;
            int token  = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row0, TokenTile, kv_head, q_head, token);
            const std::int64_t dst =
                gqa_partial_acc_index<Geometry>(q_head, d0, token, split, TokenTile);
            *reinterpret_cast<unsigned*>(&partial_acc[dst]) = pack_bf16x2(acc[n][0], acc[n][1]);
        }
        if (row1 < RowCount) {
            int q_head = 0;
            int token  = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row1, TokenTile, kv_head, q_head, token);
            const std::int64_t dst =
                gqa_partial_acc_index<Geometry>(q_head, d0, token, split, TokenTile);
            *reinterpret_cast<unsigned*>(&partial_acc[dst]) = pack_bf16x2(acc[n][2], acc[n][3]);
        }
    }
}

// Standalone S3 fill (K + V), launched stream-ordered BEFORE the tiled kernel when the
// append path writes the cache. The tiled kernel's fused V fill is split-racy: a Sage
// 16-key block scale is cross-key state, and a block straddling a split boundary lets
// two CTAs race the running max / rescale. Here each (16-key block, kv_head, d pair) is
// owned by a single thread and its block's tokens are processed in position order, so
// the state is single-CTA; the tc kernel then reads the settled codes/scales.
template <typename Geometry, int TokenTile, bool MultiBatch, bool Masked>
__launch_bounds__(256) __global__ void gqa_attention_decode_fill_nvfp4s3_kernel(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    const std::int32_t* __restrict__ pos, const std::int32_t* __restrict__ block_tables,
    const std::int32_t* __restrict__ table_rows, std::int32_t table_stride,
    std::uint8_t* __restrict__ cache_k, std::uint8_t* __restrict__ cache_v,
    std::uint8_t* __restrict__ scale_k, std::uint8_t* __restrict__ scale_v,
    std::int32_t full_width, std::int32_t column_begin, std::int32_t batch_size,
    const std::int32_t* __restrict__ valid_columns) {
    constexpr int D       = kGqaHeadDim;
    constexpr int Groups  = kGqaNvfp4Groups;
    constexpr int DpPairs = D / 2;
    const int tid         = static_cast<int>(threadIdx.x);
    const int unit        = static_cast<int>(blockIdx.x) * 256 + tid;

    const int v_blocks = (TokenTile + 15) / 16 + 1;
    const int k_max    = TokenTile * Geometry::KVHeads * Groups;
    const int per_batch =
        k_max + v_blocks * Geometry::KVHeads * DpPairs;
    const int batch = MultiBatch ? unit / per_batch : 0;
    if (MultiBatch && batch >= batch_size) { return; }
    const int rest = MultiBatch ? unit - batch * per_batch : unit;

    const int column_base = column_begin + (MultiBatch ? batch * full_width : 0);
    const int valid_tokens =
        Masked
            ? (valid_columns[batch] <= column_base
                   ? 0
                   : (valid_columns[batch] - column_base < TokenTile
                          ? valid_columns[batch] - column_base
                          : TokenTile))
            : TokenTile;
    if (valid_tokens <= 0) { return; }
    const std::int32_t* block_table = block_tables +
                                      (table_rows == nullptr ? 0 : table_rows[batch]) * table_stride;
    const int base_pos = pos[column_base];
    const __nv_bfloat16* kb =
        k + static_cast<std::int64_t>(column_base) * D * Geometry::KVHeads;
    const __nv_bfloat16* vb =
        v + static_cast<std::int64_t>(column_base) * D * Geometry::KVHeads;

    if (rest < k_max) {
        // K: per-(token, d-16-group), stateless.
        const int token   = rest / (Geometry::KVHeads * Groups);
        if (token >= valid_tokens) { return; }
        const int tmp     = rest % (Geometry::KVHeads * Groups);
        const int kv_head = tmp / Groups;
        const int grp     = tmp % Groups;
        const int position      = base_pos + token;
        const int physical_page = paged_kv_physical_page(block_table, position);
        const int page_off      = position & kPagedKVPageMask;
        const int d0            = grp * kGqaNvfp4Group;
        std::uint32_t lo = 0, hi = 0;
        std::uint8_t sc = 0;
        gqa_nvfp4_quantize_bf16x16(&kb[gqa_kv_new_index<Geometry>(kv_head, d0, token)], lo, hi, sc);
        const std::int64_t code =
            gqa_nvfp4_code_index<Geometry>(physical_page, kv_head, grp * 8, page_off);
        *reinterpret_cast<std::uint32_t*>(cache_k + code)     = lo;
        *reinterpret_cast<std::uint32_t*>(cache_k + code + 4) = hi;
        scale_k[gqa_nvfp4_scale_index<Geometry>(physical_page, kv_head, grp, page_off)] = sc;
        return;
    }

    // V: one thread per (cache 16-key block, kv_head, d pair). Single-shot contract:
    // the new keys' unquantized values are loaded into registers, the per-d block max
    // is reduced once, the block scale is computed once, and the codes are written in
    // a single pass (a scale bump moves the block's prefix codes by a single-ratio
    // one-shot pass). A 16-key block never straddles a page (64 keys/page), so its
    // whole state lives in one physical page.
    const int v_unit  = rest - k_max;
    const int blk     = v_unit / (Geometry::KVHeads * DpPairs);
    const int tmp     = v_unit % (Geometry::KVHeads * DpPairs);
    const int kv_head = tmp / DpPairs;
    const int dp      = tmp % DpPairs;
    const int block_key0 = (base_pos / 16 + blk) * 16;
    if (block_key0 + 16 <= base_pos || block_key0 >= base_pos + valid_tokens) { return; }
    const int ib_begin = base_pos > block_key0 ? base_pos - block_key0 : 0;
    const int ib_end =
        (base_pos + valid_tokens < block_key0 + 16 ? base_pos + valid_tokens - block_key0 : 16);
    const int physical_page = block_table[block_key0 >> kPagedKVPageShift];
    const int page_off      = block_key0 & kPagedKVPageMask;
    const int key_block     = page_off >> 4;
    const int d0            = dp * 2;
    const std::int64_t sc0_off =
        gqa_s3_v_scale_index<Geometry>(physical_page, kv_head, d0, key_block);
    const std::int64_t sc1_off =
        gqa_s3_v_scale_index<Geometry>(physical_page, kv_head, d0 + 1, key_block);
    const std::uint8_t scale_byte0 = scale_v[sc0_off];
    const std::uint8_t scale_byte1 = scale_v[sc1_off];
    const float s_cur0 = detail::decode_nvfp4_e4m3(scale_byte0);
    const float s_cur1 = detail::decode_nvfp4_e4m3(scale_byte1);
    // New (in-range) keys: register-resident; the per-d block max reduced once.
    float v0s[16], v1s[16];
    float m0 = 0.0f, m1 = 0.0f;
    for (int ki = ib_begin; ki < ib_end; ++ki) {
        const int t = ki + block_key0 - base_pos;
        const std::int64_t src = gqa_kv_new_index<Geometry>(kv_head, d0, t);
        v0s[ki] = __bfloat162float(vb[src]);
        v1s[ki] = __bfloat162float(vb[src + 1]);
        m0 = fmaxf(m0, fabsf(v0s[ki]));
        m1 = fmaxf(m1, fabsf(v1s[ki]));
    }
    // Final block scale: new keys' max vs stored implied max, encoded once.
    const float fin_max0 = fmaxf(m0, s_cur0 * 6.0f);
    const float fin_max1 = fmaxf(m1, s_cur1 * 6.0f);
    const std::uint8_t sc_new0 =
        __nv_cvt_float_to_fp8(__fdiv_rn(fin_max0, 6.0f), __NV_SATFINITE, __NV_E4M3);
    const std::uint8_t sc_new1 =
        __nv_cvt_float_to_fp8(__fdiv_rn(fin_max1, 6.0f), __NV_SATFINITE, __NV_E4M3);
    const float s_new0 = detail::decode_nvfp4_e4m3(sc_new0);
    const float s_new1 = detail::decode_nvfp4_e4m3(sc_new1);
    // A bump: the in-block prefix codes (written under the stored scale by an earlier
    // step) move to the new scale by a single s_cur/s_new ratio, one-shot pass.
    const bool bump0 = sc_new0 != scale_byte0;
    const bool bump1 = sc_new1 != scale_byte1;
    const float rescale0 =
        (bump0 && s_cur0 > 0.0f && s_new0 > 0.0f) ? __fdiv_rn(s_cur0, s_new0) : 1.0f;
    const float rescale1 =
        (bump1 && s_cur1 > 0.0f && s_new1 > 0.0f) ? __fdiv_rn(s_cur1, s_new1) : 1.0f;
    if (rescale0 != 1.0f || rescale1 != 1.0f) {
        for (int ki = 0; ki < ib_begin; ++ki) {
            const std::int64_t co = gqa_nvfp4_code_index<Geometry>(
                physical_page, kv_head, dp, page_off + ki);
            const std::uint8_t byte = cache_v[co];
            const float d0v        = gqa_s3_e2m1_value(byte & 0x0fu);
            const float d1          = gqa_s3_e2m1_value((byte >> 4) & 0x0fu);
            cache_v[co] = gqa_s3_cvt_e2m1x2(d0v * rescale0, d1 * rescale1);
        }
    }
    // New key codes under the final scale: single rounding, one coalesced pass.
    for (int ki = ib_begin; ki < ib_end; ++ki) {
        const float c0 = s_new0 > 0.0f ? __fdiv_rn(v0s[ki], s_new0) : 0.0f;
        const float c1 = s_new1 > 0.0f ? __fdiv_rn(v1s[ki], s_new1) : 0.0f;
        cache_v[gqa_nvfp4_code_index<Geometry>(physical_page, kv_head, dp, page_off + ki)] =
            gqa_s3_cvt_e2m1x2(c0, c1);
    }
    if (bump0) { scale_v[sc0_off] = sc_new0; }
    if (bump1) { scale_v[sc1_off] = sc_new1; }
}

} // namespace ninfer::ops