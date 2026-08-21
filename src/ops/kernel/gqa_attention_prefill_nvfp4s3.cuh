#pragma once

#include <cstdio> // device debug printf (temporary)

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
// Cache layout (NVFP4 storage, --sage only): K is quantized exactly as the prod
// NVFP4 fill (per key, 16-d groups). V is quantized along the d axis: one UE4M3
// scale per (d, 16-key block) in the v_scale plane, laid out d-major
// [page][kv_head][d 256][key_block 4] (same 1024 B per page-head as the prod
// per-key plane; only the index mapping and the code normalization differ). The
// v code plane keeps the prod per-key [key][128 B] layout.

#include "ops/kernel/gqa_attention_kv_nvfp4.cuh"
#include "ops/kernel/gqa_attention_prefill_common.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops {

// S3 constant folding: P is amplified by 448*6 inside the exp2 (the constant is
// added to the -max term in log2 domain); the per-16-key block scale then maps the
// amplified block max into e4m3, and P/scale lands in [0, 6] (e2m1 range).
inline constexpr float kGqaS3Fp8ScaleLog2 = -11.392317422778762f; // log2f(1.f / (448 * 6))
inline constexpr float kGqaS3Fp4ScaleLog2 = -2.584962500721156f;  // log2f(1.f / 6.f)
inline constexpr float kGqaS3PvAmpLog2    = -kGqaS3Fp8ScaleLog2;  // log2f(448 * 6)
inline constexpr float kGqaS3SfLog2 = -kGqaS3Fp8ScaleLog2 + kGqaS3Fp4ScaleLog2; // log2f(448)

inline constexpr int kGqaPrefillNvfp4s3Warps     = 16;
inline constexpr int kGqaPrefillNvfp4s3Threads    = kGqaPrefillNvfp4s3Warps * 32;
inline constexpr int kGqaPrefillNvfp4s3Br         = 128;
inline constexpr int kGqaPrefillNvfp4s3Bc         = 64;
inline constexpr int kGqaPrefillNvfp4s3Groups     = kGqaNvfp4Groups;
inline constexpr int kGqaPrefillNvfp4s3CodeW      = kGqaNvfp4CodeWidth;
inline constexpr int kGqaPrefillNvfp4s3RowTiles   = kGqaPrefillNvfp4s3Br / 16;
inline constexpr int kGqaPrefillNvfp4s3DConsumers =
    kGqaPrefillNvfp4s3Warps / kGqaPrefillNvfp4s3RowTiles;
// P codes: [128 rows][48 B rows, 32 B used]. 48 B stride keeps 16 B ldmatrix
// windows aligned with no swizzle.
inline constexpr int kGqaPrefillNvfp4s3P4RowBytes = 48;
inline constexpr int kGqaPrefillNvfp4s3P4Bytes    = kGqaPrefillNvfp4s3Br * kGqaPrefillNvfp4s3P4RowBytes;
// P per-16-key block scales: [128 rows][4 e4m3].
inline constexpr int kGqaPrefillNvfp4s3PsfBytes   = kGqaPrefillNvfp4s3Br * 4;
// V codes transposed to d-major: [256 d rows][48 B rows, 32 B used].
inline constexpr int kGqaPrefillNvfp4s3VtBytes     = kGqaPrefillHeadDim * kGqaPrefillNvfp4s3P4RowBytes;
// V per-(d, 16-key) block scales (d-major PV SFB operand): [256 d rows][4 e4m3].
inline constexpr int kGqaPrefillNvfp4s3VsfBytes    = kGqaPrefillHeadDim * 4;

inline constexpr int kGqaPrefillNvfp4s3QBytes      = kGqaPrefillNvfp4s3Br * kGqaPrefillNvfp4s3CodeW;
inline constexpr int kGqaPrefillNvfp4s3QScaleBytes = kGqaPrefillNvfp4s3Br * kGqaPrefillNvfp4s3Groups;
inline constexpr int kGqaPrefillNvfp4s3KBytes      = kGqaPrefillNvfp4s3Bc * kGqaPrefillNvfp4s3CodeW;
inline constexpr int kGqaPrefillNvfp4s3VBytes      = kGqaPrefillNvfp4s3Bc * kGqaPrefillNvfp4s3CodeW;
inline constexpr int kGqaPrefillNvfp4s3KScaleBytes = kGqaPrefillNvfp4s3Bc * kGqaPrefillNvfp4s3Groups;
inline constexpr int kGqaPrefillNvfp4s3StatsBytes  =
    2 * kGqaPrefillNvfp4s3Br * static_cast<int>(sizeof(float));
inline constexpr int kGqaPrefillNvfp4s3SmemBytes =
    kGqaPrefillNvfp4s3QBytes + kGqaPrefillNvfp4s3QScaleBytes + kGqaPrefillNvfp4s3KBytes +
    kGqaPrefillNvfp4s3VBytes + kGqaPrefillNvfp4s3P4Bytes + kGqaPrefillNvfp4s3PsfBytes +
    kGqaPrefillNvfp4s3VtBytes + kGqaPrefillNvfp4s3VsfBytes +
    kGqaPrefillNvfp4s3KScaleBytes + kGqaPrefillNvfp4s3VsfBytes + kGqaPrefillNvfp4s3StatsBytes;

static_assert(kGqaPrefillNvfp4s3Groups == 16);
static_assert(kGqaPrefillNvfp4s3DConsumers == 2);
static_assert(kGqaPrefillNvfp4s3SmemBytes == 57856);

// Sage V scale plane: d-major [page][kv_head][d 256][key_block 4] (1024 B per
// page-head, same size as the prod per-key plane).
template <typename Geometry>
__device__ __forceinline__ std::int64_t gqa_s3_v_scale_index(int physical_page, int kv_head,
                                                             int d, int key_block) {
    return paged_kv_element_offset<kGqaNvfp4Groups, Geometry::KVHeads>(physical_page, kv_head, 0, 0) +
           static_cast<std::int64_t>(d) * 4 + key_block;
}

// Pack two floats into one e2m1x2 byte (low nibble = first operand).
__device__ __forceinline__ std::uint8_t gqa_s3_cvt_e2m1x2(float lo, float hi) {
    unsigned tmp;
    asm volatile(
        "{\n"
        " .reg .b8 b;\n"
        " cvt.rn.satfinite.e2m1x2.f32 b, %2, %1;\n"
        " mov.b32 %0, {b, b, b, b};\n"
        "}"
        : "=r"(tmp)
        : "f"(lo), "f"(hi));
    return static_cast<std::uint8_t>(tmp);
}

// Per-nibble e2m1 decode (software). Avoids the hardware cvt.rn.f16x2.e2m1x2 path
// (__nv_fp4x2_e2m1 -> float2), which on sm_120a swaps the two halves when both
// nibbles are negative. Matches the host decode_e2m1_word table exactly.
__device__ __forceinline__ float gqa_s3_e2m1_value(std::uint8_t code) {
    static const float kGqaS3E2m1Mags[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float mag = kGqaS3E2m1Mags[code & 0x07u];
    return (code & 0x08u) ? -mag : mag;
}

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
        // K: prod NVFP4 quantization (per key, 16-d groups), unchanged.
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
        gqa_nvfp4_quantize_bf16x16(
            &k[gqa_nvfp4_src_index<Geometry>(kv_head, d0, token)], lo, hi, sc);
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
    // d pair); the block's scale is running state (a new key either fits the block's
    // current max or bumps it, rescaling the block's earlier codes). Tokens of the
    // block are processed in order, so rescale reads are settled.
    const int v_unit    = unit - k_units - kmean_units;
    const int blk       = v_unit / (Geometry::KVHeads * DpPairs);
    const int tmp       = v_unit % (Geometry::KVHeads * DpPairs);
    const int kv_head   = tmp / DpPairs;
    const int dp        = tmp % DpPairs;
    const int block_key0 = (base_pos / 16 + blk) * 16;
    if (block_key0 + 16 <= base_pos || block_key0 >= base_pos + tokens) { return; }
    const int tok_begin = (base_pos > block_key0 ? base_pos : block_key0) - base_pos;
    const int tok_end   = (base_pos + tokens < block_key0 + 16
                               ? base_pos + tokens
                               : block_key0 + 16) -
                          base_pos;
    const int physical_page = block_table[block_key0 >> kPagedKVPageShift];
    const int page_off      = block_key0 & kPagedKVPageMask;
    const int key_block     = page_off >> 4;
    const int d0            = dp * 2;
    const std::int64_t sc0_off = gqa_s3_v_scale_index<Geometry>(physical_page, kv_head, d0, key_block);
    const std::int64_t sc1_off =
        gqa_s3_v_scale_index<Geometry>(physical_page, kv_head, d0 + 1, key_block);
    float s_cur0 = detail::decode_nvfp4_e4m3(scale_v[sc0_off]);
    float s_cur1 = detail::decode_nvfp4_e4m3(scale_v[sc1_off]);
    for (int t = tok_begin; t < tok_end; ++t) {
        const int position = base_pos + t;
        const std::int64_t src = gqa_nvfp4_src_index<Geometry>(kv_head, d0, t);
        const float v0        = __bfloat162float(v[src]);
        const float v1        = __bfloat162float(v[src + 1]);
        const float prev_max0 = s_cur0 * 6.0f;
        const float prev_max1 = s_cur1 * 6.0f;
        const float new_max0  = fmaxf(fabsf(v0), prev_max0);
        const float new_max1  = fmaxf(fabsf(v1), prev_max1);
        const float s_prev0   = s_cur0;
        const float s_prev1   = s_cur1;
        if (new_max0 > prev_max0) {
            scale_v[sc0_off] =
                __nv_cvt_float_to_fp8(__fdiv_rn(new_max0, 6.0f), __NV_SATFINITE, __NV_E4M3);
            s_cur0 = detail::decode_nvfp4_e4m3(scale_v[sc0_off]);
        }
        if (new_max1 > prev_max1) {
            scale_v[sc1_off] =
                __nv_cvt_float_to_fp8(__fdiv_rn(new_max1, 6.0f), __NV_SATFINITE, __NV_E4M3);
            s_cur1 = detail::decode_nvfp4_e4m3(scale_v[sc1_off]);
        }
        const float rescale0 =
            (new_max0 > prev_max0 && s_prev0 > 0.0f && s_cur0 > 0.0f) ? s_prev0 / s_cur0 : 1.0f;
        const float rescale1 =
            (new_max1 > prev_max1 && s_prev1 > 0.0f && s_cur1 > 0.0f) ? s_prev1 / s_cur1 : 1.0f;
        if (rescale0 != 1.0f || rescale1 != 1.0f) {
            for (int i = block_key0; i < position; ++i) {
                const std::int64_t co = gqa_nvfp4_code_index<Geometry>(
                    physical_page, kv_head, dp, i & kPagedKVPageMask);
                const std::uint8_t byte = cache_v[co];
                const float d0          = gqa_s3_e2m1_value(byte & 0x0fu);
                const float d1          = gqa_s3_e2m1_value((byte >> 4) & 0x0fu);
                cache_v[co] = gqa_s3_cvt_e2m1x2(d0 * rescale0, d1 * rescale1);
            }
        }
        const float c0 = s_cur0 > 0.0f ? __fdiv_rn(v0, s_cur0) : 0.0f;
        const float c1 = s_cur1 > 0.0f ? __fdiv_rn(v1, s_cur1) : 0.0f;
        cache_v[gqa_nvfp4_code_index<Geometry>(physical_page, kv_head, dp,
                                               page_off + (position - block_key0))] =
            gqa_s3_cvt_e2m1x2(c0, c1);
    }
}

template <typename Geometry, typename Metadata>
// Occupancy-1: 16 warps × 128 regs = 65536. 512-thread CTA cannot exceed 128.
__global__ __maxnreg__(128) void gqa_attention_prefill_nvfp4s3_kernel(
    const __nv_bfloat16* __restrict__ q, const std::uint8_t* __restrict__ cache_k,
    const std::uint8_t* __restrict__ cache_v, const std::uint8_t* __restrict__ cache_k_scale,
    const std::uint8_t* __restrict__ cache_v_scale, const float* __restrict__ k_mean,
    Metadata metadata,
    const std::int32_t* __restrict__ positions, float scale, __nv_bfloat16* __restrict__ out,
    std::int32_t width, float keep_frac) {
    constexpr int D             = kGqaPrefillHeadDim;
    constexpr int Br            = kGqaPrefillNvfp4s3Br;
    constexpr int Bc            = kGqaPrefillNvfp4s3Bc;
    constexpr int Groups        = kGqaPrefillNvfp4s3Groups;
    constexpr int CodeW         = kGqaPrefillNvfp4s3CodeW;
    constexpr int P4Row         = kGqaPrefillNvfp4s3P4RowBytes;
    constexpr int QKNt          = Bc / 8;
    constexpr int K64s          = kGqaNvfp4K64;
    constexpr int PVNtPerWarp   = D / (kGqaPrefillNvfp4s3DConsumers * 8);
    constexpr int ProducerWarps = kGqaPrefillNvfp4s3RowTiles;
    constexpr int VWorkerWarps  = kGqaPrefillNvfp4s3Warps - ProducerWarps;
    constexpr int WorkerThreads = VWorkerWarps * 32;
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;

    static_assert(PVNtPerWarp == 16);

    extern __shared__ __align__(16) unsigned char smem_raw[];
    std::uint8_t* q_codes = smem_raw;
    std::uint8_t* q_scale = q_codes + kGqaPrefillNvfp4s3QBytes;
    std::uint8_t* k_codes = q_scale + kGqaPrefillNvfp4s3QScaleBytes;
    std::uint8_t* v_codes = k_codes + kGqaPrefillNvfp4s3KBytes;
    std::uint8_t* p4      = v_codes + kGqaPrefillNvfp4s3VBytes;
    std::uint8_t* psf     = p4 + kGqaPrefillNvfp4s3P4Bytes;
    std::uint8_t* v_t     = psf + kGqaPrefillNvfp4s3PsfBytes;
    std::uint8_t* v_scales = v_t + kGqaPrefillNvfp4s3VtBytes;
    std::uint8_t* k_scale_s = v_scales + kGqaPrefillNvfp4s3VsfBytes;
    float* alpha_s          = reinterpret_cast<float*>(k_scale_s + kGqaPrefillNvfp4s3KScaleBytes);
    float* final_l_s        = alpha_s + Br;

    const int q_block = static_cast<int>(blockIdx.x);
    const int q_head  = static_cast<int>(blockIdx.y);
    const int tid     = static_cast<int>(threadIdx.x);
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int q0      = q_block * Br;
    const int kv_head = q_head / Geometry::GroupSize;
    const int tokens  = metadata.valid_tokens(width);
    if (q_head >= Geometry::QHeads || q0 >= width) { return; }
    if (q0 >= tokens) {
        gqa_prefill_zero_output_rows<Geometry>(out, q_head, q0, min(q0 + Br, width), tid,
                                               kGqaPrefillNvfp4s3Threads);
        return;
    }
    const int base_pos              = positions[0];
    const std::int32_t* block_table = metadata.block_table();

    const int tile_rows     = min(Br, tokens - q0);
    const int max_query_abs = base_pos + q0 + tile_rows - 1;
    const int key_blocks    = max_query_abs / Bc + 1;

    for (int i = tid; i < Br * CodeW; i += kGqaPrefillNvfp4s3Threads) { q_codes[i] = 0; }
    for (int i = tid; i < Br * Groups; i += kGqaPrefillNvfp4s3Threads) { q_scale[i] = 0; }
    __syncthreads();

    for (int unit = tid; unit < Br * Groups; unit += kGqaPrefillNvfp4s3Threads) {
        const int row = unit / Groups;
        const int grp = unit - row * Groups;
        std::uint32_t lo = 0, hi = 0;
        std::uint8_t sc = 0;
        if (row < tile_rows) {
            gqa_nvfp4_quantize_bf16x16(
                &q[gqa_prefill_q_index<Geometry>(q_head, grp * kGqaNvfp4Group, q0 + row)], lo, hi,
                sc);
        }
        const int phys = gqa_nvfp4_swizzle_byte(row, grp * 8);
        *reinterpret_cast<std::uint32_t*>(q_codes + row * CodeW + phys)     = lo;
        *reinterpret_cast<std::uint32_t*>(q_codes + row * CodeW + phys + 4) = hi;
        q_scale[row * Groups + grp]                                         = sc;
    }
    __syncthreads();

    auto issue_kv_tile = [&](int tile_k0) {
        const int physical_page = block_table[tile_k0 >> kPagedKVPageShift];
        for (int key_l = tid; key_l < Bc; key_l += kGqaPrefillNvfp4s3Threads) {
            const int key = tile_k0 + key_l;
            if (key <= max_query_abs) {
                const std::int64_t off =
                    gqa_nvfp4_scale_index<Geometry>(physical_page, kv_head, 0, key_l);
                ninfer::ops::cp_async<16>(&k_scale_s[key_l * Groups], &cache_k_scale[off]);
            } else {
                store_vec(&k_scale_s[key_l * Groups], make_int4(0, 0, 0, 0));
            }
        }
        for (int chunk = tid; chunk < Bc * (CodeW / 16); chunk += kGqaPrefillNvfp4s3Threads) {
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
        for (int d = tid; d < D; d += kGqaPrefillNvfp4s3Threads) {
            // One aligned 4B copy of the whole kb row (kb 1..3 offsets are not 4B aligned).
            const std::int64_t off = gqa_s3_v_scale_index<Geometry>(physical_page, kv_head, d, 0);
            ninfer::ops::cp_async<4>(&v_scales[d * 4], &cache_v_scale[off]);
        }
        ninfer::ops::cp_commit();
    };

    issue_kv_tile(0);
    ninfer::ops::cp_wait<0>();
    __syncthreads();

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
        for (int kb = warp; kb < key_blocks; kb += kGqaPrefillNvfp4s3Warps) {
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
    if (k_keep_count > 0) { issue_kv_tile(keep_list[0] * Bc); ninfer::ops::cp_wait<0>(); }
    __syncthreads();
    for (int ki = 0; ki < k_keep_count; ++ki) {
        const int kb = keep_list[ki];
        const int k0 = kb * Bc;
        if (warp < ProducerWarps) {
            const int row_base = warp * 16;
            float score[QKNt][4];
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
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
                for (int nt = 0; nt < QKNt; ++nt) {
                    const int brow       = nt * 8 + b_row_offset;
                    const int b_logical  = k64 * 32 + b_column_byte;
                    const int b_physical = gqa_nvfp4_swizzle_byte(brow, b_logical);
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

            const int row0             = row_base + gid;
            const int row1             = row0 + 8;
            const int qabs0            = row0 < tile_rows ? base_pos + q0 + row0 : -1;
            const int qabs1            = row1 < tile_rows ? base_pos + q0 + row1 : -1;
            const bool full_score_tile = q0 + Br <= tokens && k0 + Bc - 1 <= base_pos + q0;
            // Per-16-key raw-score maxima: block nb covers nt 2nb/2nb+1, 8 lanes
            // (2 nt × 4 lid) hold its 16 scores; reduce across the 4 lid lanes.
            float bm0_blk[4], bm1_blk[4];
#pragma unroll
            for (int nb = 0; nb < 4; ++nb) {
                float m0 = -CUDART_INF_F;
                float m1 = -CUDART_INF_F;
#pragma unroll
                for (int nt = 2 * nb; nt < 2 * nb + 2; ++nt) {
                    if (!full_score_tile) {
                        const int key0 = k0 + nt * 8 + 2 * lid;
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
            float bm0 = -CUDART_INF_F;
            float bm1 = -CUDART_INF_F;
#pragma unroll
            for (int nb = 0; nb < 4; ++nb) {
                bm0 = fmaxf(bm0, bm0_blk[nb]);
                bm1 = fmaxf(bm1, bm1_blk[nb]);
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
            for (int nb = 0; nb < 4; ++nb) {
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
                p4[row0 * P4Row + nb * 8 + lid] = gqa_s3_cvt_e2m1x2(qa0, qa1);
                p4[row0 * P4Row + nb * 8 + 4 + lid] = gqa_s3_cvt_e2m1x2(qa2, qa3);
                p4[row1 * P4Row + nb * 8 + lid] = gqa_s3_cvt_e2m1x2(qb0, qb1);
                p4[row1 * P4Row + nb * 8 + 4 + lid] = gqa_s3_cvt_e2m1x2(qb2, qb3);
                if (lid == 0) {
                    psf[row0 * 4 + nb] = sc0;
                    psf[row1 * 4 + nb] = sc1;
                }
            }
            bl0        = warp_sum<4>(bl0, FullMask);
            bl1        = warp_sum<4>(bl1, FullMask);
            running_l0 = __fmaf_rn(running_l0, alpha0, bl0);
            running_l1 = __fmaf_rn(running_l1, alpha1, bl1);
            running_m0 = nm0;
            running_m1 = nm1;
            if (lid == 0) {
                alpha_s[row0] = alpha0;
                alpha_s[row1] = alpha1;
            }
        } else if (warp < ProducerWarps + VWorkerWarps) {
            // Transpose V codes key-major -> d-major (v_t), feeding the PV B operand.
            const int worker_tid = tid - ProducerWarps * 32;
            // One worker thread per d-row: write a contiguous 32-byte v_t row so the
            // store hits distinct SMEM banks. The original unit-strided store placed
            // every lane's write on the same bank (32-way conflict).
            for (int d = worker_tid; d < D; d += WorkerThreads) {
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
        __syncthreads();

        const bool has_next = ki + 1 < k_keep_count;
        if (has_next) { issue_kv_tile(keep_list[ki + 1] * Bc); }

        const int row_tile = warp % kGqaPrefillNvfp4s3RowTiles;
        const int d_slice  = warp / kGqaPrefillNvfp4s3RowTiles;
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
                sfb = *reinterpret_cast<const unsigned*>(&v_scales[vsf_row * 4]);
            }
            mma_nvfp4_e4m3(acc[n][0], acc[n][1], acc[n][2], acc[n][3], pf[0], pf[1], pf[2],
                           pf[3], vf[0], vf[1], sfa, sfb);
        }
        if (has_next) { ninfer::ops::cp_wait<0>(); }
        __syncthreads();
        }

    if (warp < ProducerWarps && lid == 0) {
        const int row0  = warp * 16 + gid;
        const int row1  = row0 + 8;
        final_l_s[row0] = running_l0;
        final_l_s[row1] = running_l1;
    }
    __syncthreads();

    const int row_tile = warp % kGqaPrefillNvfp4s3RowTiles;
    const int d_slice  = warp / kGqaPrefillNvfp4s3RowTiles;
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

} // namespace ninfer::ops