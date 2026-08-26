#pragma once

#include <cstdio> // device debug printf (temporary)
#include <cstring> // std::memcpy (op-dump side-band)

// Sage3-style NVFP4 GQA prompt kernel (SageAttention3, arXiv 2505.11594, sm120 tree).
// QK stays E2M1 through m16n8k64 NVFP4 Tensor Cores with hardware UE4M3 scales
// (identical to gqa_attention_prefill_nvfp4.cuh). The PV GEMM is FP4xFP4:
// P is quantized to E2M1 codes with a per-16-key UE4M3 block scale, V stays FP4
// (transposed to d-major smem, no BF16 dequant stage), and the PV matmul runs on
// the same block-scaled m16n8k64 NVFP4 Tensor Core instruction.
//
// S3 folds a constant 448*6 amplification into the softmax exponent so per-block P
// maxima land in the e4m3 scale range for long context; the amplification cancels in
// the final out = acc / L. The constants below are taken verbatim from the S3 sm120
// kernel (sageattn3/blackwell/softmax_fused.h).
//
// SmoothQ (Sage3 preprocess_qkv): each CTA subtracts its Br-row Q mean before
// NVFP4 Q-quant, then adds q_mean·K_hat onto the QK scores so softmax sees
// Q·K_hat rather than (Q-mean)·K_hat.
//
// K is NOT page-centered. Sage3's `k -= k.mean(dim=seq)` is a single sequence-
// global mean, which is softmax-invariant (Q·(K-c) = Q·K - Q·c, constant per
// row). A per-page mean is a different c per 64-key tile and wrecks softmax
// (8k PPL 10.86 vs kv-nvfp4 6.70, 2026-08-25). Incremental paged fill cannot
// cheaply keep a global mean constant, so K stays the production NVFP4 quant.
//
// Cache layout (NVFP4 storage, --sage only): K is quantized exactly as the prod
// NVFP4 fill (per key, 16-d groups). V is quantized along the d axis: one UE4M3
// scale per (d, 16-key block) in the v_scale plane, laid out d-major
// [page][kv_head][d 256][key_block 4] (same 1024 B per page-head as the prod
// per-key plane; only the index mapping and the code normalization differ). The
// v code plane keeps the prod per-key [key][128 B] layout.

#include "ops/kernel/gqa_attention_nvfp4s3_common.cuh"
#include "ninfer/ops/gqa_attention.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops {

template <typename Geometry, typename Metadata>
__launch_bounds__(256) __global__ void gqa_attention_prefill_fill_nvfp4s3_kernel(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    const std::int32_t* __restrict__ positions, Metadata metadata,
    std::uint8_t* __restrict__ cache_k, std::uint8_t* __restrict__ cache_v,
    std::uint8_t* __restrict__ scale_k, std::uint8_t* __restrict__ scale_v,
    float* __restrict__ k_mean, std::int32_t width) {
    constexpr int Groups  = kGqaNvfp4Groups;
    constexpr int DpPairs = kGqaNvfp4HeadDim / 2;
    const int tokens      = metadata.valid_tokens(width);
    const int tid         = static_cast<int>(threadIdx.x);
    const int unit        = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + tid;

    const int k_units = tokens * Geometry::KVHeads * Groups;
    // Meansim proxy: one thread per (page, kv_head, 16-d group) accumulates the
    // dequantized K sum of the page's valid keys into k_mean (sum plane).
    const int kmean_units = k_mean != nullptr
                                ? (tokens + kPagedKVPageSize - 1) / kPagedKVPageSize *
                                      Geometry::KVHeads * Groups
                                : 0;
    // V: one thread per (cache 16-key block, kv_head, d pair); a token range touches at
    // most div_up(tokens, 16) + 1 cache blocks (first + last partial).
    const int v_blocks = (tokens + 15) / 16 + 1;
    const int v_units = v_blocks * Geometry::KVHeads * DpPairs;
    if (unit >= k_units + kmean_units + v_units) { return; }

    const std::int32_t* block_table = metadata.block_table();
    const int base_pos              = positions[0];

    if (unit < k_units) {
        // K: production NVFP4 (per key, 16-d groups). Do not per-page-center:
        // that is not Sage3's global mean and is not softmax-invariant.
        const int grp     = unit % Groups;
        const int tmp     = unit / Groups;
        const int kv_head = tmp % Geometry::KVHeads;
        const int token   = tmp / Geometry::KVHeads;
        const int position        = base_pos + token;
        const int physical_page    = paged_kv_physical_page(block_table, position);
        const int page_off         = position & kPagedKVPageMask;
        const int d0               = grp * kGqaNvfp4Group;
        std::uint32_t lo = 0, hi = 0;
        std::uint8_t sc = 0;
        gqa_nvfp4_quantize_bf16x16(&k[gqa_nvfp4_src_index<Geometry>(kv_head, d0, token)], lo, hi,
                                   sc);
        const std::int64_t code =
            gqa_nvfp4_code_index<Geometry>(physical_page, kv_head, grp * 8, page_off);
        *reinterpret_cast<std::uint32_t*>(cache_k + code)     = lo;
        *reinterpret_cast<std::uint32_t*>(cache_k + code + 4) = hi;
        scale_k[gqa_nvfp4_scale_index<Geometry>(physical_page, kv_head, grp, page_off)] = sc;
        return;
    }

    if (unit < k_units + kmean_units) {
        // Meansim proxy sum unit.
        const int mu      = unit - k_units;
        const int grp     = mu % Groups;
        const int tmp     = mu / Groups;
        const int kv_head = tmp % Geometry::KVHeads;
        const int page_l  = tmp / Geometry::KVHeads;
        const int first_pos = base_pos + page_l * kPagedKVPageSize;
        if (first_pos >= base_pos + tokens) { return; }
        const std::int64_t mean_base =
            paged_kv_element_offset<4, Geometry::KVHeads>(page_l, kv_head, grp * 4, 0);
        const int d0     = grp * kGqaNvfp4Group;
        const int last   = min(first_pos + kPagedKVPageSize, base_pos + tokens);
        float sum[kGqaNvfp4Group];
#pragma unroll
        for (int i = 0; i < kGqaNvfp4Group; ++i) { sum[i] = 0.0f; }
        for (int position = first_pos; position < last; ++position) {
            const int token = position - base_pos;
#pragma unroll
            for (int i = 0; i < kGqaNvfp4Group; ++i) {
                sum[i] += __bfloat162float(k[gqa_nvfp4_src_index<Geometry>(kv_head, d0 + i, token)]);
            }
        }
        const float inv_n = 1.0f / static_cast<float>(last - first_pos);
#pragma unroll
        for (int i = 0; i < kGqaNvfp4Group; ++i) {
            k_mean[mean_base + i] = sum[i] * inv_n;
        }
        return;
    }

    // V: Sage-style d-major quantization. One thread per (cache 16-key block, kv_head,
    // d pair). The block scale is per (d, 16-key block) and a thread owns its d-pair
    // for the whole block, so the scale entry's 16-key max is a local reduction over
    // the thread's own loads -- a cross-lane reduction would mix different d's (the
    // wrong granularity) and buys nothing.
    //
    // A block may already hold live codes under the stored block scale (the decode/MTP
    // prefix of a straddling block, or in-block keys this append does not cover). The
    // stored scale implies that existing data's block max (s*6), so the final block
    // scale is the max of (a) the new keys' max, reduced once in registers over the
    // in-range keys, and (b) the stored scale's implied max. It is computed once; the
    // new key codes are written under it in a single coalesced pass and the scale bytes
    // are written only when the scale changed. There is no backward-looking rescale
    // loop: out-of-range codes keep their bytes (their e2m1 values are unchanged when
    // re-consumed under the bumped scale, to within one e2m1 ulp).
    const int v_unit    = unit - k_units - kmean_units;
    const int blk       = v_unit / (Geometry::KVHeads * DpPairs);
    const int tmp       = v_unit % (Geometry::KVHeads * DpPairs);
    const int kv_head   = tmp / DpPairs;
    const int dp        = tmp % DpPairs;
    const int block_key0 = (base_pos / 16 + blk) * 16;
    if (block_key0 + 16 <= base_pos || block_key0 >= base_pos + tokens) { return; }
    const int inblock_begin = base_pos > block_key0 ? base_pos - block_key0 : 0;
    const int inblock_end   = base_pos + tokens < block_key0 + 16
                               ? base_pos + tokens - block_key0
                              : 16;
    const int physical_page = block_table[block_key0 >> kPagedKVPageShift];
    const int page_off      = block_key0 & kPagedKVPageMask;
    const int key_block     = page_off >> 4;
    const int d0            = dp * 2;
    const std::int64_t sc0_off = gqa_s3_v_scale_index<Geometry>(physical_page, kv_head, d0, key_block);
    const std::int64_t sc1_off =
        gqa_s3_v_scale_index<Geometry>(physical_page, kv_head, d0 + 1, key_block);

    // Existing in-block keys (outside this append's range) hold live codes under the
    // stored block scale; its implied block max (s*6) contributes to the block max.
    const std::uint8_t scale_byte0 = scale_v[sc0_off];
    const std::uint8_t scale_byte1 = scale_v[sc1_off];
    const float s_cur0 = detail::decode_nvfp4_e4m3(scale_byte0);
    const float s_cur1 = detail::decode_nvfp4_e4m3(scale_byte1);

    // New (in-range) keys: register-resident; the per-d 16-key max reduced once.
    // ki is the in-block index; in-range keys are [inblock_begin, inblock_end).
    float v0s[16], v1s[16];
    for (int ki = inblock_begin; ki < inblock_end; ++ki) {
        const std::int64_t src =
            gqa_nvfp4_src_index<Geometry>(kv_head, d0, block_key0 + ki - base_pos);
        v0s[ki] = __bfloat162float(v[src]);
        v1s[ki] = __bfloat162float(v[src + 1]);
    }
    float m0 = 0.0f, m1 = 0.0f;
    for (int ki = inblock_begin; ki < inblock_end; ++ki) {
        m0 = fmaxf(m0, fabsf(v0s[ki]));
        m1 = fmaxf(m1, fabsf(v1s[ki]));
    }
    // Final block scale: new keys' max vs stored implied max, encoded once.
    const float fin_max0 = fmaxf(m0, s_cur0 * 6.0f);
    const float fin_max1 = fmaxf(m1, s_cur1 * 6.0f);
    const std::uint8_t sc_new0 = __nv_cvt_float_to_fp8(__fdiv_rn(fin_max0, 6.0f), __NV_SATFINITE, __NV_E4M3);
    const std::uint8_t sc_new1 = __nv_cvt_float_to_fp8(__fdiv_rn(fin_max1, 6.0f), __NV_SATFINITE, __NV_E4M3);
    const float s_new0 = detail::decode_nvfp4_e4m3(sc_new0);
    const float s_new1 = detail::decode_nvfp4_e4m3(sc_new1);

    // A scale bump: the in-block prefix codes (ki < inblock_begin, written in earlier
    // steps under the stored scale) must be brought to the new scale -- one bounded
    // single-ratio pass (each prefix byte touched at most once per append; no
    // backward-looking rescale loop). Out-of-range in-block keys are never touched.
    const bool bump0 = sc_new0 != scale_byte0;
    const bool bump1 = sc_new1 != scale_byte1;
    const float rescale0 = (bump0 && s_cur0 > 0.0f && s_new0 > 0.0f) ? __fdiv_rn(s_cur0, s_new0) : 1.0f;
    const float rescale1 = (bump1 && s_cur1 > 0.0f && s_new1 > 0.0f) ? __fdiv_rn(s_cur1, s_new1) : 1.0f;
    if (rescale0 != 1.0f || rescale1 != 1.0f) {
        for (int ki = 0; ki < inblock_begin; ++ki) {
            const std::int64_t co =
                gqa_nvfp4_code_index<Geometry>(physical_page, kv_head, dp, page_off + ki);
            const std::uint8_t cb = cache_v[co];
            const float d0v       = gqa_s3_e2m1_value(cb & 0x0fu);
            const float d1v       = gqa_s3_e2m1_value((cb >> 4) & 0x0fu);
            cache_v[co] = gqa_s3_cvt_e2m1x2(d0v * rescale0, d1v * rescale1);
        }
    }
    // New key codes under the final scale: single rounding, one coalesced pass.
    for (int ki = inblock_begin; ki < inblock_end; ++ki) {
        const float c0 = s_new0 > 0.0f ? __fdiv_rn(v0s[ki], s_new0) : 0.0f;
        const float c1 = s_new1 > 0.0f ? __fdiv_rn(v1s[ki], s_new1) : 0.0f;
        cache_v[gqa_nvfp4_code_index<Geometry>(physical_page, kv_head, dp, page_off + ki)] =
            gqa_s3_cvt_e2m1x2(c0, c1);
    }
    if (bump0) { scale_v[sc0_off] = sc_new0; }
    if (bump1) { scale_v[sc1_off] = sc_new1; }
}

template <typename Geometry, typename Metadata, int BrT, int WarpsT, bool ExactT>
__device__ __forceinline__ void gqa_attention_prefill_nvfp4s3_device(
    const __nv_bfloat16* __restrict__ q, const std::uint8_t* __restrict__ cache_k,
    const std::uint8_t* __restrict__ cache_v, const std::uint8_t* __restrict__ cache_k_scale,
    const std::uint8_t* __restrict__ cache_v_scale, const float* __restrict__ k_mean,
    Metadata metadata,
    const std::int32_t* __restrict__ positions, float scale, __nv_bfloat16* __restrict__ out,
    std::int32_t width, float keep_frac, GqaS3PrefillDump* dump,
    std::uint32_t* dbg_regs = nullptr, std::uint8_t* dbg_q = nullptr) {
    using Tile = GqaPrefillNvfp4s3Tile<BrT, WarpsT>;
    constexpr int D             = kGqaPrefillHeadDim;
    constexpr int Br            = Tile::Br;
    constexpr int Bc            = kGqaPrefillNvfp4s3Bc;
    constexpr int Groups        = kGqaPrefillNvfp4s3Groups;
    constexpr int CodeW         = kGqaPrefillNvfp4s3CodeW;
    constexpr int P4Row         = kGqaPrefillNvfp4s3P4RowBytes;
    constexpr int Threads       = Tile::Threads;
    constexpr int QKNt          = Bc / 8;
    // Exact (occ2) path: all 8 warps run QK+P-quant. Two warps share a 16-row
    // tile by splitting the 64 keys (nt 0-3 / 4-7). NCU on the 4+4 producer/
    // transpose split showed ~31% of issue time parked at the CTA barrier —
    // the transpose is cheap and the 4 worker warps wait on softmax/P-quant.
    constexpr int NtLocal       = ExactT ? QKNt / 2 : QKNt;
    constexpr int NbLocal       = ExactT ? 2 : 4;
    constexpr int K64s          = kGqaNvfp4K64;
    constexpr int PVNtPerWarp   = D / (Tile::DConsumers * 8);
    constexpr int ProducerWarps = Tile::RowTiles;
    constexpr int VWorkerWarps  = Tile::Warps - ProducerWarps;
    constexpr int WorkerThreads = VWorkerWarps * 32;
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;

    static_assert(PVNtPerWarp == 16);
    static_assert(Tile::DConsumers == 2);

    extern __shared__ __align__(16) unsigned char smem_raw[];
    std::uint8_t* q_codes = smem_raw;
    std::uint8_t* q_scale = q_codes + Tile::QBytes;
    std::uint8_t* k_codes = q_scale + Tile::QScaleBytes;
    std::uint8_t* v_codes = k_codes + kGqaPrefillNvfp4s3KBytes;
    std::uint8_t* p4      = v_codes + kGqaPrefillNvfp4s3VBytes;
    std::uint8_t* psf     = p4 + Tile::P4Bytes;
    std::uint8_t* v_t     = psf + Tile::PsfBytes;
    std::uint8_t* v_scales   = v_t + kGqaPrefillNvfp4s3VtBytes;
    std::uint8_t* v_scales_b = v_scales + kGqaPrefillNvfp4s3VsfBytes;
    std::uint8_t* k_scale_s  = v_scales_b + kGqaPrefillNvfp4s3VsfBytes;
    float* alpha_s          = reinterpret_cast<float*>(k_scale_s + kGqaPrefillNvfp4s3KScaleBytes);
    float* final_l_s        = alpha_s + Br;

    __shared__ float q_smooth_mean[kGqaPrefillHeadDim];
    __shared__ float k_smooth_delta[kGqaPrefillNvfp4s3Bc];

    const int q_block = static_cast<int>(blockIdx.x);
    const int q_head  = static_cast<int>(blockIdx.y);
    const int tid     = static_cast<int>(threadIdx.x);
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int q0      = q_block * Br;
    const int kv_head = q_head / Geometry::GroupSize;
    // op-dump side-band: only q_block 0 writes (the buffers are sized per head for it).
    const bool do_dump = dump != nullptr && q_block == 0;
    const std::int64_t dht =
        do_dump ? static_cast<std::int64_t>(q_head) * dump->max_tiles : 0; // [h][t] base
    const int tokens  = metadata.valid_tokens(width);
    if (q_head >= Geometry::QHeads || q0 >= width) { return; }
    if (q0 >= tokens) {
        gqa_prefill_zero_output_rows<Geometry>(out, q_head, q0, min(q0 + Br, width), tid,
                                               Threads);
        return;
    }
    const int base_pos              = positions[0];
    const std::int32_t* block_table = metadata.block_table();

    const int tile_rows     = min(Br, tokens - q0);
    const int max_query_abs = base_pos + q0 + tile_rows - 1;
    const int key_blocks    = max_query_abs / Bc + 1;

    for (int i = tid; i < Br * CodeW; i += Threads) { q_codes[i] = 0; }
    for (int i = tid; i < Br * Groups; i += Threads) { q_scale[i] = 0; }
    __syncthreads();

    // SmoothQ: mean over this CTA's Q rows (Sage3 group-mean; GROUP_SIZE analog is Br).
    for (int d = tid; d < D; d += Threads) {
        float s = 0.0f;
        for (int r = 0; r < tile_rows; ++r) {
            s += __bfloat162float(q[gqa_prefill_q_index<Geometry>(q_head, d, q0 + r)]);
        }
        q_smooth_mean[d] = tile_rows > 0 ? s / static_cast<float>(tile_rows) : 0.0f;
    }
    __syncthreads();

    for (int unit = tid; unit < Br * Groups; unit += Threads) {
        const int row = unit / Groups;
        const int grp = unit - row * Groups;
        std::uint32_t lo = 0, hi = 0;
        std::uint8_t sc = 0;
        if (row < tile_rows) {
            float adj[kGqaNvfp4Group];
            const int d0 = grp * kGqaNvfp4Group;
#pragma unroll
            for (int i = 0; i < kGqaNvfp4Group; ++i) {
                adj[i] = __bfloat162float(q[gqa_prefill_q_index<Geometry>(q_head, d0 + i, q0 + row)]) -
                         q_smooth_mean[d0 + i];
            }
            gqa_nvfp4_quantize_f32x16(adj, lo, hi, sc);
        }
        const int phys = gqa_nvfp4_swizzle_byte(row, grp * 8);
        *reinterpret_cast<std::uint32_t*>(q_codes + row * CodeW + phys)     = lo;
        *reinterpret_cast<std::uint32_t*>(q_codes + row * CodeW + phys + 4) = hi;
        q_scale[row * Groups + grp]                                         = sc;
    }
    __syncthreads();
    // Debug: capture the fully-written Q codes + Q scale smem for host A/B.
    if (dbg_q != nullptr && q_head == 0 && q_block == 0) {
        for (int i = tid; i < Br * CodeW; i += Threads) dbg_q[i] = q_codes[i];
        for (int i = tid; i < Br * Groups; i += Threads)
            dbg_q[Tile::QBytes + i] = q_scale[i];
    }

    auto issue_kv_tile = [&](int tile_k0, std::uint8_t* vs_stage) {
        const int physical_page = block_table[tile_k0 >> kPagedKVPageShift];
        for (int key_l = tid; key_l < Bc; key_l += Threads) {
            const int key = tile_k0 + key_l;
            if (key <= max_query_abs) {
                const std::int64_t off =
                    gqa_nvfp4_scale_index<Geometry>(physical_page, kv_head, 0, key_l);
                ninfer::ops::cp_async<16>(&k_scale_s[key_l * Groups], &cache_k_scale[off]);
            } else {
                store_vec(&k_scale_s[key_l * Groups], make_int4(0, 0, 0, 0));
            }
        }
        for (int chunk = tid; chunk < Bc * (CodeW / 16); chunk += Threads) {
            const int key_l   = chunk / (CodeW / 16);
            const int seg     = chunk - key_l * (CodeW / 16);
            const int logical = seg * 16;
            const int key     = tile_k0 + key_l;
            const int phys    = gqa_nvfp4_swizzle_byte(key_l, logical);
            std::uint8_t* kd  = &k_codes[key_l * CodeW + phys];
            std::uint8_t* vd  = &v_codes[key_l * CodeW + logical];
            if (key <= max_query_abs) {
                const std::int64_t off =
                    gqa_nvfp4_code_index<Geometry>(physical_page, kv_head, logical, key_l);
                cp_async<16, Cache::cg>(kd, &cache_k[off]);
                cp_async<16, Cache::cg>(vd, &cache_v[off]);
            } else {
                store_vec(kd, make_int4(0, 0, 0, 0));
                store_vec(vd, make_int4(0, 0, 0, 0));
            }
        }
        for (int d = tid; d < D; d += Threads) {
            // One aligned 4B copy of the whole kb row (kb 1..3 offsets are not 4B aligned).
            // vs_stage is the parity slot for the tile being issued (see v_scales_b).
            const std::int64_t off = gqa_s3_v_scale_index<Geometry>(physical_page, kv_head, d, 0);
            ninfer::ops::cp_async<4>(&vs_stage[d * 4], &cache_v_scale[off]);
        }
        ninfer::ops::cp_commit();
    };

    const int gid           = lane >> 2;
    const int lid           = lane & 3;
    const int a_matrix      = lane >> 3;
    const int a_row_offset  = (lane & 7) + ((a_matrix & 1) << 3);
    const int a_column_byte = (a_matrix >> 1) * 16;
    const int b_row_offset  = lane & 7;
    const int b_column_byte = ((lane >> 3) & 1) * 16;
    const int sfa_row       = ((lane & 1) << 3) | (lane >> 2);
    const int sfb_row       = lane >> 2;

    float acc[PVNtPerWarp][4];
#pragma unroll
    for (int n = 0; n < PVNtPerWarp; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[n][i] = 0.0f; }
    }
    float running_m0     = -CUDART_INF_F;
    float running_m1     = -CUDART_INF_F;
    float running_l0     = 0.0f;
    float running_l1     = 0.0f;
    const float scale_l2 = scale * Log2E;

    int n_tiles = key_blocks;
    const int* keep_list_ptr = nullptr;
    if constexpr (!ExactT) {
    __shared__ float proxy_scores[4096];
    __shared__ int keep_list[4096];
    __shared__ int k_keep_count;
    __shared__ float q_mean_s[kGqaPrefillHeadDim];
    // keep_frac is the fraction of key tiles kept, in (0, 1]. approx only when actually
    // skipping tiles (keep_frac<1) and the sage_pv k_mean proxy is present; keep_frac==1.0
    // (keep all) takes the exact path with no proxy. keep_frac<=0 is invalid (caught at the
    // launcher) and degrades to the exact path here as a defensive fallback.
    const bool approx = keep_frac > 0.0f && keep_frac < 1.0f && k_mean != nullptr;
    // Sinks/window are only consulted on the approx (tile-skip) path; gate them on approx so
    // the exact path does not compute unused tile counts.
    const int k_sinks  = approx ? max(1, (int)(key_blocks * keep_frac * 0.2f)) : 1;
    const int k_window = approx ? max(1, (int)(key_blocks * keep_frac * 0.4f)) : 1;

    if (!approx) {
        // Exact path: keep every key tile.
        if (warp == 0 && lid == 0) {
            k_keep_count = 0;
            for (int kb = 0; kb < key_blocks; ++kb) keep_list[k_keep_count++] = kb;
        }
    } else {
        // SpargeAttn meansim gate (arXiv 2502.18137): rank tiles by
        // mean(Q_block) . mean(K_tile) using the per-page dequantized K mean
        // precomputed by the fill kernel (k_mean) -- zero K/V fetch, so the main
        // loop below fetches only the kept tiles.
        if (tid < D) {
            const int d       = tid;
            const int grp     = d / kGqaNvfp4Group;
            const int byte_l  = d >> 1;
            const bool hi_nib = (d & 1) != 0;
            float acc         = 0.0f;
            for (int row = 0; row < tile_rows; ++row) {
                const int phys         = gqa_nvfp4_swizzle_byte(row, byte_l);
                const std::uint8_t cb  = q_codes[row * CodeW + phys];
                const std::uint8_t nib = hi_nib ? ((cb >> 4) & 0x0fu) : (cb & 0x0fu);
                acc += gqa_s3_e2m1_value(nib) *
                       detail::decode_nvfp4_e4m3(q_scale[row * Groups + grp]);
            }
            q_mean_s[d] = acc / static_cast<float>(tile_rows);
        }
        __syncthreads();
        for (int kb = warp; kb < key_blocks; kb += Tile::Warps) {
            const int physical_page = block_table[kb];
            float acc               = 0.0f;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int d = i * 32 + lane;
                const std::int64_t off = paged_kv_element_offset<4, Geometry::KVHeads>(
                    physical_page, kv_head, d >> 2, d & 3);
                acc += q_mean_s[d] * k_mean[off];
            }
#pragma unroll
            for (int r = 16; r; r >>= 1) { acc += __shfl_xor_sync(FullMask, acc, r); }
            if (lane == 0) { proxy_scores[kb] = acc; }
        }
        __syncthreads();
        // Keep set: top-k tiles by proxy score + attention sinks + sliding window.
        if (warp == 0) {
            if (lane == 0) { k_keep_count = 0; }
            __syncwarp();
            const int topk = min(static_cast<int>(keep_frac * key_blocks), key_blocks);
            for (int sel = 0; sel < topk; ++sel) {
                float best  = -CUDART_INF_F;
                int best_kb = -1;
                for (int kb = lane; kb < key_blocks; kb += 32) {
                    if (proxy_scores[kb] > best) { best = proxy_scores[kb]; best_kb = kb; }
                }
#pragma unroll
                for (int r = 16; r; r >>= 1) {
                    const float ov = __shfl_xor_sync(FullMask, best, r);
                    const int ok   = __shfl_xor_sync(FullMask, best_kb, r);
                    if (ov > best || (ov == best && ok >= 0 && ok < best_kb)) {
                        best    = ov;
                        best_kb = ok;
                    }
                }
                if (lane == 0) {
                    keep_list[k_keep_count++] = best_kb;
                    proxy_scores[best_kb]     = -CUDART_INF_F;
                }
                __syncwarp();
            }
            if (lane == 0) {
                for (int kb = 0; kb < key_blocks; ++kb) {
                    if ((kb < k_sinks || kb >= key_blocks - k_window) &&
                        proxy_scores[kb] > -CUDART_INF_F) {
                        proxy_scores[kb]          = -CUDART_INF_F;
                        keep_list[k_keep_count++] = kb;
                    }
                }
            }
        }
    }
    __syncthreads();
    if (do_dump && warp == 0 && lane == 0) {
        const int ncopy = min(k_keep_count, dump->max_tiles);
        for (int i = 0; i < ncopy; ++i) { dump->keep_list[q_head * dump->max_tiles + i] = keep_list[i]; }
        dump->tile_count[q_head] = k_keep_count;
    }
    n_tiles = k_keep_count;
    keep_list_ptr = keep_list;
    } // if constexpr (!ExactT)
    if (n_tiles > 0) {
        const int first_kb = ExactT ? 0 : keep_list_ptr[0];
        issue_kv_tile(first_kb * Bc, v_scales);
        ninfer::ops::cp_wait<0>();
    }
    __syncthreads();
    for (int ki = 0; ki < n_tiles; ++ki) {
        const int kb = ExactT ? ki : keep_list_ptr[ki];
        const int k0 = kb * Bc;
        // SmoothQ restore: k_delta[key] = q_mean · K_hat[key] (same addend for every
        // row in this Br tile). All threads participate so the skip-list producer
        // warps and the occ2 split-N warps see a settled plane before QK.
        if (tid < Bc) { k_smooth_delta[tid] = 0.0f; }
        __syncthreads();
        {
            const int nslice = Threads / Bc;
            const int d_per  = D / nslice;
            const int key_l  = tid & (Bc - 1);
            const int d0     = (tid / Bc) * d_per;
            float p          = 0.0f;
            if (k0 + key_l <= max_query_abs) {
                for (int i = 0; i < d_per; ++i) {
                    p += q_smooth_mean[d0 + i] *
                         gqa_s3_k_smem_dequant(k_codes, k_scale_s, key_l, d0 + i);
                }
            }
            atomicAdd(&k_smooth_delta[key_l], p);
        }
        __syncthreads();
        const bool do_qk = ExactT || (warp < ProducerWarps);
        if (do_qk) {
            const int k_half   = ExactT ? warp / Tile::RowTiles : 0;
            const int row_base = ExactT ? (warp % Tile::RowTiles) * 16 : warp * 16;
            const int nt_base  = k_half * NtLocal;
            const int nb_base  = k_half * NbLocal;
            float score[NtLocal][4];
#pragma unroll
            for (int nt = 0; nt < NtLocal; ++nt) {
                score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0f;
            }

#pragma unroll
            for (int k64 = 0; k64 < K64s; ++k64) {
                const int row           = row_base + a_row_offset;
                const int logical_byte  = k64 * 32 + a_column_byte;
                const int physical_byte = gqa_nvfp4_swizzle_byte(row, logical_byte);
                unsigned af[4];
                ldmatrix_x4(af[0], af[1], af[2], af[3],
                            smem_addr(q_codes + row * CodeW + physical_byte));
                const int scale_row = row_base + sfa_row;
                // MMA block_scale only consumes SFA from even-pair lanes and SFB from
                // lane%4==0; the other lanes duplicate those rows.
                unsigned sfa = 0;
                if ((lane & 2) == 0) {
                    sfa = *reinterpret_cast<const unsigned*>(
                        &q_scale[scale_row * Groups + k64 * 4]);
                }
#pragma unroll
                for (int nt = 0; nt < NtLocal; ++nt) {
                    const int nt_g       = nt + nt_base;
                    const int brow       = nt_g * 8 + b_row_offset;
                    const int b_logical  = k64 * 32 + b_column_byte;
                    const int b_physical = gqa_nvfp4_swizzle_byte(brow, b_logical);
                    unsigned bf[2];
                    ldmatrix_x2(bf[0], bf[1], smem_addr(k_codes + brow * CodeW + b_physical));
                    const int b_scale_row = nt_g * 8 + sfb_row;
                    unsigned sfb          = 0;
                    if ((lane & 3) == 0) {
                        sfb = *reinterpret_cast<const unsigned*>(
                            &k_scale_s[b_scale_row * Groups + k64 * 4]);
                    }
                    mma_nvfp4_e4m3(score[nt][0], score[nt][1], score[nt][2], score[nt][3], af[0],
                                   af[1], af[2], af[3], bf[0], bf[1], sfa, sfb);
                    if (dbg_regs != nullptr && warp == 0 && k64 == 0 && nt == 0 &&
                        q_head == 0 && q_block == 0) {
                        dbg_regs[lane * 8 + 0] = af[0];
                        dbg_regs[lane * 8 + 1] = af[1];
                        dbg_regs[lane * 8 + 2] = af[2];
                        dbg_regs[lane * 8 + 3] = af[3];
                        dbg_regs[lane * 8 + 4] = bf[0];
                        dbg_regs[lane * 8 + 5] = bf[1];
                        dbg_regs[lane * 8 + 6] = sfa;
                        dbg_regs[lane * 8 + 7] = sfb;
                    }
                }
            }

#pragma unroll
            for (int nt = 0; nt < NtLocal; ++nt) {
                const int k0l = (nt + nt_base) * 8 + 2 * lid;
                score[nt][0] += k_smooth_delta[k0l];
                score[nt][1] += k_smooth_delta[k0l + 1];
                score[nt][2] += k_smooth_delta[k0l];
                score[nt][3] += k_smooth_delta[k0l + 1];
            }

            const int row0             = row_base + gid;
            const int row1             = row0 + 8;
            const int qabs0            = row0 < tile_rows ? base_pos + q0 + row0 : -1;
            const int qabs1            = row1 < tile_rows ? base_pos + q0 + row1 : -1;
            const bool full_score_tile = q0 + Br <= tokens && k0 + Bc - 1 <= base_pos + q0;
            // Per-16-key raw-score maxima: block nb covers nt 2nb/2nb+1, 8 lanes
            // (2 nt × 4 lid) hold its 16 scores; reduce across the 4 lid lanes.
            float bm0_blk[NbLocal], bm1_blk[NbLocal];
#pragma unroll
            for (int nb = 0; nb < NbLocal; ++nb) {
                float m0 = -CUDART_INF_F;
                float m1 = -CUDART_INF_F;
#pragma unroll
                for (int nt = 2 * nb; nt < 2 * nb + 2; ++nt) {
                    const int nt_g = nt + nt_base;
                    if (!full_score_tile) {
                        const int key0 = k0 + nt_g * 8 + 2 * lid;
                        const int key1 = key0 + 1;
                        if (key0 > qabs0) { score[nt][0] = -CUDART_INF_F; }
                        if (key1 > qabs0) { score[nt][1] = -CUDART_INF_F; }
                        if (key0 > qabs1) { score[nt][2] = -CUDART_INF_F; }
                        if (key1 > qabs1) { score[nt][3] = -CUDART_INF_F; }
                    }
                    m0 = fmaxf(m0, fmaxf(score[nt][0], score[nt][1]));
                    m1 = fmaxf(m1, fmaxf(score[nt][2], score[nt][3]));
                }
                bm0_blk[nb] = warp_max<4>(m0, FullMask);
                bm1_blk[nb] = warp_max<4>(m1, FullMask);
            }
            if (do_dump) {
                // Raw (post-causal-mask) QK dot per (row, in-tile key): lane holds
                // (row0,row1) x (key, key+1) per nt. -INF marks causally masked keys.
                float* ds = &dump->score[(dht + ki) * static_cast<std::int64_t>(Br) * Bc];
                for (int nt = 0; nt < NtLocal; ++nt) {
                    const int key_l = (nt + nt_base) * 8 + 2 * lid;
                    ds[static_cast<std::int64_t>(row0) * Bc + key_l]       = score[nt][0];
                    ds[static_cast<std::int64_t>(row0) * Bc + key_l + 1]   = score[nt][1];
                    ds[static_cast<std::int64_t>(row1) * Bc + key_l]       = score[nt][2];
                    ds[static_cast<std::int64_t>(row1) * Bc + key_l + 1]   = score[nt][3];
                }
            }
            float bm0 = -CUDART_INF_F;
            float bm1 = -CUDART_INF_F;
#pragma unroll
            for (int nb = 0; nb < NbLocal; ++nb) {
                bm0 = fmaxf(bm0, bm0_blk[nb]);
                bm1 = fmaxf(bm1, bm1_blk[nb]);
            }
            if constexpr (ExactT) {
                // Partner warp holds the other 32 keys of this 16-row tile. v_t is
                // unused until the transpose below, so it hosts the max/L scratch.
                float* half_m = reinterpret_cast<float*>(v_t);
                if (lid == 0) {
                    half_m[k_half * Br + row0] = bm0;
                    half_m[k_half * Br + row1] = bm1;
                }
                __syncthreads();
                bm0 = fmaxf(bm0, half_m[(k_half ^ 1) * Br + row0]);
                bm1 = fmaxf(bm1, half_m[(k_half ^ 1) * Br + row1]);
            }

            const float nm0        = fmaxf(running_m0, bm0);
            const float nm1        = fmaxf(running_m1, bm1);
            const float nm0_scaled = nm0 * scale_l2;
            const float nm1_scaled = nm1 * scale_l2;
            const float alpha0     = running_m0 == -CUDART_INF_F
                                         ? 0.0f
                                         : exp2_approx(__fmaf_rn(running_m0, scale_l2, -nm0_scaled));
            const float alpha1     = running_m1 == -CUDART_INF_F
                                         ? 0.0f
                                         : exp2_approx(__fmaf_rn(running_m1, scale_l2, -nm1_scaled));
            float bl0              = 0.0f;
            float bl1              = 0.0f;
#pragma unroll
            for (int nb = 0; nb < NbLocal; ++nb) {
                const int nb_g = nb + nb_base;
                // Per-16-key P block scale (S3): S = 448 * exp2((m_blk - nm) * scale_l2),
                // encoded to e4m3; P codes = P / S land in [0, 6] (e2m1 range).
                const float sf0f = bm0_blk[nb] == -CUDART_INF_F
                                       ? 0.0f
                                       : exp2_approx(__fmaf_rn(bm0_blk[nb], scale_l2,
                                                                -nm0_scaled + kGqaS3SfLog2));
                const float sf1f = bm1_blk[nb] == -CUDART_INF_F
                                       ? 0.0f
                                       : exp2_approx(__fmaf_rn(bm1_blk[nb], scale_l2,
                                                                -nm1_scaled + kGqaS3SfLog2));
                const std::uint8_t sc0 =
                    sf0f == 0.0f ? 0 : __nv_cvt_float_to_fp8(sf0f, __NV_SATFINITE, __NV_E4M3);
                const std::uint8_t sc1 =
                    sf1f == 0.0f ? 0 : __nv_cvt_float_to_fp8(sf1f, __NV_SATFINITE, __NV_E4M3);
                const float dec0 = detail::decode_nvfp4_e4m3(sc0);
                const float dec1 = detail::decode_nvfp4_e4m3(sc1);
                const float amp  = kGqaS3PvAmpLog2;
                const float pa0  = score[2 * nb][0] > -CUDART_INF_F
                                       ? exp2_approx(__fmaf_rn(score[2 * nb][0], scale_l2,
                                                               -nm0_scaled + amp))
                                       : 0.0f;
                const float pa1  = score[2 * nb][1] > -CUDART_INF_F
                                       ? exp2_approx(__fmaf_rn(score[2 * nb][1], scale_l2,
                                                               -nm0_scaled + amp))
                                       : 0.0f;
                const float pa2  = score[2 * nb + 1][0] > -CUDART_INF_F
                                       ? exp2_approx(__fmaf_rn(score[2 * nb + 1][0], scale_l2,
                                                               -nm0_scaled + amp))
                                       : 0.0f;
                const float pa3  = score[2 * nb + 1][1] > -CUDART_INF_F
                                       ? exp2_approx(__fmaf_rn(score[2 * nb + 1][1], scale_l2,
                                                               -nm0_scaled + amp))
                                       : 0.0f;
                const float pb0  = score[2 * nb][2] > -CUDART_INF_F
                                       ? exp2_approx(__fmaf_rn(score[2 * nb][2], scale_l2,
                                                               -nm1_scaled + amp))
                                       : 0.0f;
                const float pb1  = score[2 * nb][3] > -CUDART_INF_F
                                       ? exp2_approx(__fmaf_rn(score[2 * nb][3], scale_l2,
                                                               -nm1_scaled + amp))
                                       : 0.0f;
                const float pb2  = score[2 * nb + 1][2] > -CUDART_INF_F
                                       ? exp2_approx(__fmaf_rn(score[2 * nb + 1][2], scale_l2,
                                                               -nm1_scaled + amp))
                                       : 0.0f;
                const float pb3  = score[2 * nb + 1][3] > -CUDART_INF_F
                                       ? exp2_approx(__fmaf_rn(score[2 * nb + 1][3], scale_l2,
                                                               -nm1_scaled + amp))
                                       : 0.0f;
                bl0 += pa0 + pa1 + pa2 + pa3;
                bl1 += pb0 + pb1 + pb2 + pb3;
                const float qa0 = dec0 > 0.0f ? __fdiv_rn(pa0, dec0) : 0.0f;
                const float qa1 = dec0 > 0.0f ? __fdiv_rn(pa1, dec0) : 0.0f;
                const float qa2 = dec0 > 0.0f ? __fdiv_rn(pa2, dec0) : 0.0f;
                const float qa3 = dec0 > 0.0f ? __fdiv_rn(pa3, dec0) : 0.0f;
                const float qb0 = dec1 > 0.0f ? __fdiv_rn(pb0, dec1) : 0.0f;
                const float qb1 = dec1 > 0.0f ? __fdiv_rn(pb1, dec1) : 0.0f;
                const float qb2 = dec1 > 0.0f ? __fdiv_rn(pb2, dec1) : 0.0f;
                const float qb3 = dec1 > 0.0f ? __fdiv_rn(pb3, dec1) : 0.0f;
                p4[row0 * P4Row + nb_g * 8 + lid] = gqa_s3_cvt_e2m1x2(qa0, qa1);
                p4[row0 * P4Row + nb_g * 8 + 4 + lid] = gqa_s3_cvt_e2m1x2(qa2, qa3);
                p4[row1 * P4Row + nb_g * 8 + lid] = gqa_s3_cvt_e2m1x2(qb0, qb1);
                p4[row1 * P4Row + nb_g * 8 + 4 + lid] = gqa_s3_cvt_e2m1x2(qb2, qb3);
                if (lid == 0) {
                    psf[row0 * 4 + nb_g] = sc0;
                    psf[row1 * 4 + nb_g] = sc1;
                }
            }
            bl0        = warp_sum<4>(bl0, FullMask);
            bl1        = warp_sum<4>(bl1, FullMask);
            if constexpr (ExactT) {
                float* half_l = reinterpret_cast<float*>(v_t);
                if (lid == 0) {
                    half_l[k_half * Br + row0] = bl0;
                    half_l[k_half * Br + row1] = bl1;
                }
                __syncthreads();
                bl0 += half_l[(k_half ^ 1) * Br + row0];
                bl1 += half_l[(k_half ^ 1) * Br + row1];
            }
            if (do_dump && (!ExactT || k_half == 0) && lane < 16) {
                // e2m1 P codes + e4m3 P-block scales for this tile's 16 rows. p4 byte j
                // holds keys (2j, 2j+1) (low nibble = key 2j); p4/psf are reused per tile,
                // so the copy must land before the next iteration's producer phase.
                const int drow = row_base + lane;
                std::uint8_t* pc = &dump->p_code[(dht + ki) * static_cast<std::int64_t>(Br) * Bc +
                                                  static_cast<std::int64_t>(drow) * Bc];
                const std::uint8_t* prow = p4 + drow * P4Row;
                for (int j = 0; j < 32; ++j) {
                    const std::uint8_t b = prow[j];
                    pc[2 * j]       = b & 0x0fu;
                    pc[2 * j + 1]   = (b >> 4) & 0x0fu;
                }
                std::memcpy(&dump->psf[(dht + ki) * static_cast<std::int64_t>(Br) * 4 +
                                         static_cast<std::int64_t>(drow) * 4],
                             psf + drow * 4, 4);
            }
            running_l0 = __fmaf_rn(running_l0, alpha0, bl0);
            running_l1 = __fmaf_rn(running_l1, alpha1, bl1);
            running_m0 = nm0;
            running_m1 = nm1;
            if (do_dump && lid == 0 && (!ExactT || k_half == 0)) {
                const std::int64_t ml = (dht + ki) * Br;
                dump->m[ml + row0] = running_m0;
                dump->l[ml + row0] = running_l0;
                dump->m[ml + row1] = running_m1;
                dump->l[ml + row1] = running_l1;
            }
            if (lid == 0) {
                alpha_s[row0] = alpha0;
                alpha_s[row1] = alpha1;
            }
        }
        if constexpr (ExactT) {
            // All 8 warps nibble-transpose V after QK. Cheap vs softmax; serial so
            // every warp stays on the QK/P-quant critical path.
            for (int d = tid; d < D; d += Threads) {
                const int dp = d >> 1;
                const int sh = (d & 1) * 4;
#pragma unroll
                for (int kp = 0; kp < Bc / 2; kp += 4) {
                    std::uint32_t packed = 0;
#pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const std::uint8_t b0 = v_codes[(2 * (kp + i)) * CodeW + dp];
                        const std::uint8_t b1 = v_codes[(2 * (kp + i) + 1) * CodeW + dp];
                        const std::uint8_t byte =
                            static_cast<std::uint8_t>(((b1 >> sh) & 0x0Fu) << 4) |
                            static_cast<std::uint8_t>((b0 >> sh) & 0x0Fu);
                        packed |= static_cast<std::uint32_t>(byte) << (i * 8);
                    }
                    *reinterpret_cast<std::uint32_t*>(&v_t[d * P4Row + kp]) = packed;
                }
            }
        } else if (warp < ProducerWarps + VWorkerWarps) {
            // Transpose V codes key-major -> d-major (v_t), feeding the PV B operand.
            const int worker_tid = tid - ProducerWarps * 32;
            for (int d = worker_tid; d < D; d += WorkerThreads) {
                const int dp = d >> 1;
                const int sh = (d & 1) * 4;
                // One worker thread per d-row: write a contiguous 32-byte v_t
                // row so the store hits distinct SMEM banks.
#pragma unroll 1
                for (int kp = 0; kp < Bc / 2; ++kp) {
                    const std::uint8_t b0 = v_codes[2 * kp * CodeW + dp];
                    const std::uint8_t b1 = v_codes[(2 * kp + 1) * CodeW + dp];
                    v_t[d * P4Row + kp] =
                        static_cast<std::uint8_t>(((b1 >> sh) & 0x0Fu) << 4) |
                        static_cast<std::uint8_t>((b0 >> sh) & 0x0Fu);
                }
            }
        }
        __syncthreads();

        // Double-buffered V-block scale plane (tile parity): this iteration's PV mma
        // reads parity slot (ki & 1), and the next iteration's async load (issued
        // below) targets the opposite slot, so the in-flight cp.async can never
        // overwrite the plane the mma is reading.
        std::uint8_t* const vs_cur  = (ki & 1) ? v_scales_b : v_scales;
        std::uint8_t* const vs_next = (ki & 1) ? v_scales : v_scales_b;
        if (do_dump && tid < D) {
            // V-block scales for this tile (read before the next tile's async load is
            // issued and before it can land on the opposite parity slot).
            std::memcpy(&dump->v_scale[(dht + ki) * static_cast<std::int64_t>(D) * 4 +
                                         static_cast<std::int64_t>(tid) * 4],
                         vs_cur + tid * 4, 4);
        }

        const bool has_next = ki + 1 < n_tiles;
        // Issue the next tile's async load now (pipelined overlap with this tile's
        // PV mma): it targets the opposite v_scales parity slot (vs_next), and every
        // read of the shared single-buffered planes (q/k codes, k scales, v_codes)
        // finished before the __syncthreads() above, so the in-flight load cannot
        // clobber any plane the current tile still reads.
        if (has_next) {
            const int next_kb = ExactT ? ki + 1 : keep_list_ptr[ki + 1];
            issue_kv_tile(next_kb * Bc, vs_next);
        }

        const int row_tile = warp % Tile::RowTiles;
        const int d_slice  = warp / Tile::RowTiles;
        const int row_base = row_tile * 16;
        const float alpha0 = alpha_s[row_base + gid];
        const float alpha1 = alpha_s[row_base + gid + 8];
#pragma unroll
        for (int n = 0; n < PVNtPerWarp; ++n) {
            acc[n][0] *= alpha0;
            acc[n][1] *= alpha0;
            acc[n][2] *= alpha1;
            acc[n][3] *= alpha1;
        }

        // A: P codes (16 rows × 32 B, all 64 keys in one block-scaled mma).
        unsigned pf[4];
        ldmatrix_x4(pf[0], pf[1], pf[2], pf[3],
                    smem_addr(p4 + (row_base + a_row_offset) * P4Row + a_column_byte));
        unsigned sfa = 0;
        if ((lane & 2) == 0) {
            sfa = *reinterpret_cast<const unsigned*>(&psf[(row_base + sfa_row) * 4]);
        }
#pragma unroll
        for (int n = 0; n < PVNtPerWarp; ++n) {
            const int global_n = d_slice * PVNtPerWarp + n;
            const int vrow     = global_n * 8 + b_row_offset;
            unsigned vf[2];
            ldmatrix_x2(vf[0], vf[1], smem_addr(v_t + vrow * P4Row + b_column_byte));
            const int vsf_row  = global_n * 8 + sfb_row;
            unsigned sfb       = 0;
            if ((lane & 3) == 0) {
                sfb = *reinterpret_cast<const unsigned*>(&vs_cur[vsf_row * 4]);
            }
            mma_nvfp4_e4m3(acc[n][0], acc[n][1], acc[n][2], acc[n][3], pf[0], pf[1], pf[2],
                           pf[3], vf[0], vf[1], sfa, sfb);
        }
        if (do_dump) {
            // PV accumulator after this tile's mma, in the tile's running-max frame
            // (rescaled into the next tile's frame at the top of the next iteration).
            float* da = &dump->acc[(dht + ki) * static_cast<std::int64_t>(Br) * D];
            const int arow0 = row_base + gid;
            const int arow1 = arow0 + 8;
            for (int n = 0; n < PVNtPerWarp; ++n) {
                const int d0 = (d_slice * PVNtPerWarp + n) * 8 + 2 * lid;
                da[static_cast<std::int64_t>(arow0) * D + d0]       = acc[n][0];
                da[static_cast<std::int64_t>(arow0) * D + d0 + 1]   = acc[n][1];
                da[static_cast<std::int64_t>(arow1) * D + d0]       = acc[n][2];
                da[static_cast<std::int64_t>(arow1) * D + d0 + 1]   = acc[n][3];
            }
            // v_t plane (this tile's transposed V-code B operand): 256 d-rows x 32 B,
            // copied by one warp (8 rows per lane) so no two threads write a byte.
            if (warp == ProducerWarps) {
                const std::int64_t vt_base =
                    (dht + ki) * static_cast<std::int64_t>(D) * 32 + (std::int64_t)(lane * 8) * 32;
                for (int r = 0; r < 8; ++r) {
                    std::memcpy(&dump->v_t[vt_base + r * 32], v_t + (lane * 8 + r) * P4Row, 32);
                }
            }
        }
        if (has_next) { ninfer::ops::cp_wait<0>(); }
        __syncthreads();
        }

    if (warp < Tile::RowTiles && lid == 0) {
        const int row0  = warp * 16 + gid;
        const int row1  = row0 + 8;
        final_l_s[row0] = running_l0;
        final_l_s[row1] = running_l1;
    }
    __syncthreads();

    const int row_tile = warp % Tile::RowTiles;
    const int d_slice  = warp / Tile::RowTiles;
    const int row_base = row_tile * 16;
    const int row0     = row_base + gid;
    const int row1     = row0 + 8;
    const float inv_l0 = final_l_s[row0] > 0.0f ? __frcp_rn(final_l_s[row0]) : 0.0f;
    const float inv_l1 = final_l_s[row1] > 0.0f ? __frcp_rn(final_l_s[row1]) : 0.0f;
#pragma unroll
    for (int n = 0; n < PVNtPerWarp; ++n) {
        const int d0 = (d_slice * PVNtPerWarp + n) * 8 + 2 * lid;
        if (row0 < tile_rows) {
            *reinterpret_cast<unsigned*>(
                &out[gqa_prefill_q_index<Geometry>(q_head, d0, q0 + row0)]) =
                pack_bf16x2(acc[n][0] * inv_l0, acc[n][1] * inv_l0);
        }
        if (row1 < tile_rows) {
            *reinterpret_cast<unsigned*>(
                &out[gqa_prefill_q_index<Geometry>(q_head, d0, q0 + row1)]) =
                pack_bf16x2(acc[n][2] * inv_l1, acc[n][3] * inv_l1);
        }
    }
}

template <typename Geometry, typename Metadata>
__global__ __maxnreg__(128) void gqa_attention_prefill_nvfp4s3_kernel(
    const __nv_bfloat16* __restrict__ q, const std::uint8_t* __restrict__ cache_k,
    const std::uint8_t* __restrict__ cache_v, const std::uint8_t* __restrict__ cache_k_scale,
    const std::uint8_t* __restrict__ cache_v_scale, const float* __restrict__ k_mean,
    Metadata metadata,
    const std::int32_t* __restrict__ positions, float scale, __nv_bfloat16* __restrict__ out,
    std::int32_t width, float keep_frac, GqaS3PrefillDump* dump,
    std::uint32_t* dbg_regs = nullptr, std::uint8_t* dbg_q = nullptr) {
    // Exact tiles only: Sparge/XAttention skip lives on the exact-NVFP4 sibling
    // kernel. Passing ExactT=true drops the meansim keep-list path.
    gqa_attention_prefill_nvfp4s3_device<Geometry, Metadata, kGqaPrefillNvfp4s3Br,
                                         kGqaPrefillNvfp4s3Warps, true>(
        q, cache_k, cache_v, cache_k_scale, cache_v_scale, k_mean, metadata, positions, scale, out,
        width, keep_frac, dump, dbg_regs, dbg_q);
}

template <typename Geometry, typename Metadata>
__global__ __launch_bounds__(GqaPrefillNvfp4s3Occ2::Threads, 2) void gqa_attention_prefill_nvfp4s3_occ2_kernel(
    const __nv_bfloat16* __restrict__ q, const std::uint8_t* __restrict__ cache_k,
    const std::uint8_t* __restrict__ cache_v, const std::uint8_t* __restrict__ cache_k_scale,
    const std::uint8_t* __restrict__ cache_v_scale, const float* __restrict__ k_mean,
    Metadata metadata,
    const std::int32_t* __restrict__ positions, float scale, __nv_bfloat16* __restrict__ out,
    std::int32_t width, float keep_frac, GqaS3PrefillDump* dump,
    std::uint32_t* dbg_regs = nullptr, std::uint8_t* dbg_q = nullptr) {
    gqa_attention_prefill_nvfp4s3_device<Geometry, Metadata, GqaPrefillNvfp4s3Occ2::Br,
                                         GqaPrefillNvfp4s3Occ2::Warps, true>(
        q, cache_k, cache_v, cache_k_scale, cache_v_scale, k_mean, metadata, positions, scale, out,
        width, keep_frac, dump, dbg_regs, dbg_q);
}

} // namespace ninfer::ops