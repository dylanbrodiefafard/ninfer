#pragma once

// NVFP4-native GQA prompt kernel. QK stays E2M1 through m16n8k64 NVFP4 Tensor Cores
// with hardware UE4M3 scales; V is dequantized to BF16 while producer warps execute QK.

#include "ops/kernel/gqa_attention_kv_nvfp4.cuh"
#include "ops/kernel/gqa_attention_prefill_common.cuh"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops {

inline constexpr int kGqaPrefillNvfp4Warps    = 16;
inline constexpr int kGqaPrefillNvfp4Threads  = kGqaPrefillNvfp4Warps * 32;
// 128 query rows halves K/V reread versus Br=64: each CTA still streams the
// full key history once, but the grid has half as many tiles.
inline constexpr int kGqaPrefillNvfp4Br       = 128;
inline constexpr int kGqaPrefillNvfp4Bc       = 64;
inline constexpr int kGqaPrefillNvfp4Groups   = kGqaNvfp4Groups;
inline constexpr int kGqaPrefillNvfp4CodeW    = kGqaNvfp4CodeWidth;
inline constexpr int kGqaPrefillNvfp4RowTiles = kGqaPrefillNvfp4Br / 16;
inline constexpr int kGqaPrefillNvfp4DConsumers =
    kGqaPrefillNvfp4Warps / kGqaPrefillNvfp4RowTiles;

inline constexpr int kGqaPrefillNvfp4QBytes     = kGqaPrefillNvfp4Br * kGqaPrefillNvfp4CodeW;
inline constexpr int kGqaPrefillNvfp4QScaleBytes = kGqaPrefillNvfp4Br * kGqaPrefillNvfp4Groups;
inline constexpr int kGqaPrefillNvfp4KBytes     = kGqaPrefillNvfp4Bc * kGqaPrefillNvfp4CodeW;
inline constexpr int kGqaPrefillNvfp4VBytes     = kGqaPrefillNvfp4Bc * kGqaPrefillNvfp4CodeW;
inline constexpr int kGqaPrefillNvfp4VStageBytes =
    kGqaPrefillNvfp4Bc * kGqaPrefillHeadDim * static_cast<int>(sizeof(__nv_bfloat16));
inline constexpr int kGqaPrefillNvfp4PBytes =
    kGqaPrefillNvfp4Br * kGqaPrefillNvfp4Bc * static_cast<int>(sizeof(__nv_bfloat16));
inline constexpr int kGqaPrefillNvfp4ScaleBytes = 2 * kGqaPrefillNvfp4Bc * kGqaPrefillNvfp4Groups;
inline constexpr int kGqaPrefillNvfp4StatsBytes =
    2 * kGqaPrefillNvfp4Br * static_cast<int>(sizeof(float));
inline constexpr int kGqaPrefillNvfp4SmemBytes =
    kGqaPrefillNvfp4QBytes + kGqaPrefillNvfp4QScaleBytes + kGqaPrefillNvfp4KBytes +
    kGqaPrefillNvfp4VBytes + kGqaPrefillNvfp4VStageBytes + kGqaPrefillNvfp4PBytes +
    kGqaPrefillNvfp4ScaleBytes + kGqaPrefillNvfp4StatsBytes;

static_assert(kGqaPrefillNvfp4Groups == 16);
static_assert(kGqaPrefillNvfp4DConsumers == 2);
static_assert(kGqaPrefillNvfp4SmemBytes == 87040);

template <typename Geometry, typename Metadata>
__launch_bounds__(256) __global__
    void gqa_attention_prefill_fill_nvfp4_kernel(const __nv_bfloat16* __restrict__ k,
                                                 const __nv_bfloat16* __restrict__ v,
                                                 const std::int32_t* __restrict__ positions,
                                                 Metadata metadata,
                                                 std::uint8_t* __restrict__ cache_k,
                                                 std::uint8_t* __restrict__ cache_v,
                                                 std::uint8_t* __restrict__ scale_k,
                                                 std::uint8_t* __restrict__ scale_v,
                                                 std::int32_t width) {
    const int tokens = metadata.valid_tokens(width);
    const int tid    = static_cast<int>(threadIdx.x);
    const int unit   = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + tid;
    const int units  = tokens * Geometry::KVHeads * kGqaPrefillNvfp4Groups;
    if (unit >= units) { return; }

    const int grp     = unit % kGqaPrefillNvfp4Groups;
    const int tmp     = unit / kGqaPrefillNvfp4Groups;
    const int kv_head = tmp % Geometry::KVHeads;
    const int token   = tmp / Geometry::KVHeads;
    const int position              = positions[0] + token;
    const std::int32_t* block_table = metadata.block_table();
    const int physical_page         = paged_kv_physical_page(block_table, position);
    const int page_off              = position & kPagedKVPageMask;
    const int d0                    = grp * kGqaNvfp4Group;

    std::uint32_t k_lo = 0, k_hi = 0, v_lo = 0, v_hi = 0;
    std::uint8_t k_sc = 0, v_sc = 0;
    gqa_nvfp4_quantize_bf16x16(&k[gqa_nvfp4_src_index<Geometry>(kv_head, d0, token)], k_lo, k_hi,
                               k_sc);
    gqa_nvfp4_quantize_bf16x16(&v[gqa_nvfp4_src_index<Geometry>(kv_head, d0, token)], v_lo, v_hi,
                               v_sc);
    const std::int64_t code =
        gqa_nvfp4_code_index<Geometry>(physical_page, kv_head, grp * 8, page_off);
    *reinterpret_cast<std::uint32_t*>(cache_k + code)     = k_lo;
    *reinterpret_cast<std::uint32_t*>(cache_k + code + 4) = k_hi;
    *reinterpret_cast<std::uint32_t*>(cache_v + code)     = v_lo;
    *reinterpret_cast<std::uint32_t*>(cache_v + code + 4) = v_hi;
    scale_k[gqa_nvfp4_scale_index<Geometry>(physical_page, kv_head, grp, page_off)] = k_sc;
    scale_v[gqa_nvfp4_scale_index<Geometry>(physical_page, kv_head, grp, page_off)] = v_sc;
}

template <typename Geometry, typename Metadata>
__launch_bounds__(256) __global__ void gqa_attention_prefill_fill_nvfp4_page_kernel(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    const std::int32_t* __restrict__ positions, Metadata metadata,
    std::uint8_t* __restrict__ cache_k, std::uint8_t* __restrict__ cache_v,
    std::uint8_t* __restrict__ scale_k, std::uint8_t* __restrict__ scale_v, std::int32_t width) {
    constexpr int TokensPerTile = 8;
    const int tokens            = metadata.valid_tokens(width);
    const int warp              = static_cast<int>(threadIdx.x) >> 5;
    const int lane              = static_cast<int>(threadIdx.x) & 31;
    const int kv_head           = static_cast<int>(blockIdx.y);
    const int tile_delta        = static_cast<int>(blockIdx.x);
    const int base_position     = positions[0];
    const int tile_position     = (base_position / TokensPerTile + tile_delta) * TokensPerTile;
    const int logical_page      = tile_position >> kPagedKVPageShift;
    const int token_begin       = max(0, tile_position - base_position);
    const int token_end         = min(tokens, tile_position + TokensPerTile - base_position);
    if (token_begin >= token_end) { return; }

    const std::int32_t* block_table = metadata.block_table();
    int physical_page               = lane == 0 ? block_table[logical_page] : 0;
    physical_page                   = __shfl_sync(0xffffffffu, physical_page, 0);

    const int token  = token_begin + warp;
    const bool valid = token < token_end && lane < kGqaPrefillNvfp4Groups;
    if (!valid) { return; }

    const int grp      = lane;
    const int position = base_position + token;
    const int page_off = position & kPagedKVPageMask;
    const int d0       = grp * kGqaNvfp4Group;
    std::uint32_t k_lo = 0, k_hi = 0, v_lo = 0, v_hi = 0;
    std::uint8_t k_sc = 0, v_sc = 0;
    gqa_nvfp4_quantize_bf16x16(&k[gqa_nvfp4_src_index<Geometry>(kv_head, d0, token)], k_lo, k_hi,
                               k_sc);
    gqa_nvfp4_quantize_bf16x16(&v[gqa_nvfp4_src_index<Geometry>(kv_head, d0, token)], v_lo, v_hi,
                               v_sc);
    const std::int64_t code =
        gqa_nvfp4_code_index<Geometry>(physical_page, kv_head, grp * 8, page_off);
    *reinterpret_cast<std::uint32_t*>(cache_k + code)     = k_lo;
    *reinterpret_cast<std::uint32_t*>(cache_k + code + 4) = k_hi;
    *reinterpret_cast<std::uint32_t*>(cache_v + code)     = v_lo;
    *reinterpret_cast<std::uint32_t*>(cache_v + code + 4) = v_hi;
    scale_k[gqa_nvfp4_scale_index<Geometry>(physical_page, kv_head, grp, page_off)] = k_sc;
    scale_v[gqa_nvfp4_scale_index<Geometry>(physical_page, kv_head, grp, page_off)] = v_sc;
}

// Per-page dequantized K mean for Sparge meansim. Indexed by physical page:
// k_mean[d] lives at paged_kv_element_offset<4,KVHeads>(page, kv_head, d>>2, d&3).
template <typename Geometry, typename Metadata>
__launch_bounds__(256) __global__ void gqa_attention_prefill_kmean_nvfp4_kernel(
    const std::uint8_t* __restrict__ cache_k, const std::uint8_t* __restrict__ scale_k,
    Metadata metadata, const std::int32_t* __restrict__ positions, float* __restrict__ k_mean,
    std::int32_t width) {
    const int tokens = metadata.valid_tokens(width);
    const int d      = static_cast<int>(threadIdx.x);
    if (d >= kGqaPrefillHeadDim) { return; }
    const int kv_head  = static_cast<int>(blockIdx.y);
    const int page_l   = static_cast<int>(blockIdx.x);
    const int base_pos = positions[0];
    const int first_page = base_pos >> kPagedKVPageShift;
    const int page       = first_page + page_l;
    const int page_lo    = page << kPagedKVPageShift;
    const int lo         = max(page_lo, base_pos);
    const int hi         = min(page_lo + kPagedKVPageSize, base_pos + tokens);
    if (lo >= hi) { return; }
    const std::int32_t* block_table = metadata.block_table();
    const int physical_page         = paged_kv_physical_page(block_table, page_lo);
    float sum                       = 0.0f;
    for (int pos = lo; pos < hi; ++pos) {
        const int page_off = pos & kPagedKVPageMask;
        const std::uint8_t packed =
            cache_k[gqa_nvfp4_code_index<Geometry>(physical_page, kv_head, d >> 1, page_off)];
        const float2 pair = detail::decode_nvfp4_e2m1x2(packed);
        const float s     = detail::decode_nvfp4_e4m3(
            scale_k[gqa_nvfp4_scale_index<Geometry>(physical_page, kv_head, d >> 4, page_off)]);
        sum += ((d & 1) != 0 ? pair.y : pair.x) * s;
    }
    k_mean[paged_kv_element_offset<4, Geometry::KVHeads>(physical_page, kv_head, d >> 2, d & 3)] =
        sum / static_cast<float>(hi - lo);
}

template <typename Geometry, typename Metadata>
// Occupancy-1: 16 warps × 128 regs = 65536. 512-thread CTA cannot exceed 128.
__global__ __maxnreg__(128) void gqa_attention_prefill_nvfp4_kernel(
    const __nv_bfloat16* __restrict__ q, const std::uint8_t* __restrict__ cache_k,
    const std::uint8_t* __restrict__ cache_v, const std::uint8_t* __restrict__ cache_k_scale,
    const std::uint8_t* __restrict__ cache_v_scale, Metadata metadata,
    const std::int32_t* __restrict__ positions, float scale, __nv_bfloat16* __restrict__ out,
    std::int32_t width) {
    constexpr int D             = kGqaPrefillHeadDim;
    constexpr int Br            = kGqaPrefillNvfp4Br;
    constexpr int Bc            = kGqaPrefillNvfp4Bc;
    constexpr int Groups        = kGqaPrefillNvfp4Groups;
    constexpr int CodeW         = kGqaPrefillNvfp4CodeW;
    constexpr int QKNt          = Bc / 8;
    constexpr int K64s          = kGqaNvfp4K64;
    constexpr int PVNtPerWarp   = D / (kGqaPrefillNvfp4DConsumers * 8);
    constexpr int PVKs          = Bc / 16;
    constexpr int ProducerWarps = kGqaPrefillNvfp4RowTiles;
    constexpr int VWorkerWarps  = kGqaPrefillNvfp4Warps - ProducerWarps;
    constexpr int WorkerThreads = VWorkerWarps * 32;
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;

    static_assert(PVNtPerWarp == 16);

    extern __shared__ __align__(16) unsigned char smem_raw[];
    std::uint8_t* q_codes = smem_raw;
    std::uint8_t* q_scale = q_codes + kGqaPrefillNvfp4QBytes;
    std::uint8_t* k_codes = q_scale + kGqaPrefillNvfp4QScaleBytes;
    std::uint8_t* v_codes = k_codes + kGqaPrefillNvfp4KBytes;
    __nv_bfloat16* v_bf16 = reinterpret_cast<__nv_bfloat16*>(v_codes + kGqaPrefillNvfp4VBytes);
    __nv_bfloat16* p_s    = reinterpret_cast<__nv_bfloat16*>(
        reinterpret_cast<unsigned char*>(v_bf16) + kGqaPrefillNvfp4VStageBytes);
    std::uint8_t* k_scale_s =
        reinterpret_cast<std::uint8_t*>(reinterpret_cast<unsigned char*>(p_s) +
                                        kGqaPrefillNvfp4PBytes);
    std::uint8_t* v_scale_s = k_scale_s + Bc * Groups;
    float* alpha_s          = reinterpret_cast<float*>(v_scale_s + Bc * Groups);
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
                                               kGqaPrefillNvfp4Threads);
        return;
    }
    const int base_pos              = positions[0];
    const std::int32_t* block_table = metadata.block_table();

    const int tile_rows     = min(Br, tokens - q0);
    const int max_query_abs = base_pos + q0 + tile_rows - 1;
    const int key_blocks    = max_query_abs / Bc + 1;

    for (int i = tid; i < Br * CodeW; i += kGqaPrefillNvfp4Threads) { q_codes[i] = 0; }
    for (int i = tid; i < Br * Groups; i += kGqaPrefillNvfp4Threads) { q_scale[i] = 0; }
    __syncthreads();

    for (int unit = tid; unit < Br * Groups; unit += kGqaPrefillNvfp4Threads) {
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
        for (int key_l = tid; key_l < Bc; key_l += kGqaPrefillNvfp4Threads) {
            const int key = tile_k0 + key_l;
            if (key <= max_query_abs) {
                const std::int64_t off =
                    gqa_nvfp4_scale_index<Geometry>(physical_page, kv_head, 0, key_l);
                ninfer::ops::cp_async<16>(&k_scale_s[key_l * Groups], &cache_k_scale[off]);
                ninfer::ops::cp_async<16>(&v_scale_s[key_l * Groups], &cache_v_scale[off]);
            } else {
                store_vec(&k_scale_s[key_l * Groups], make_int4(0, 0, 0, 0));
                store_vec(&v_scale_s[key_l * Groups], make_int4(0, 0, 0, 0));
            }
        }
        for (int chunk = tid; chunk < Bc * (CodeW / 16); chunk += kGqaPrefillNvfp4Threads) {
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
    const int a_coloff      = (a_matrix >> 1) << 3;
    const int b_rin         = lane & 7;
    const int b_koff        = ((lane >> 3) & 1) << 3;

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

    for (int kb = 0; kb < key_blocks; ++kb) {
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
            float bm0                  = -CUDART_INF_F;
            float bm1                  = -CUDART_INF_F;
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int key0 = k0 + nt * 8 + 2 * lid;
                const int key1 = key0 + 1;
                if (!full_score_tile) {
                    score[nt][0] = key0 <= qabs0 ? score[nt][0] : -CUDART_INF_F;
                    score[nt][1] = key1 <= qabs0 ? score[nt][1] : -CUDART_INF_F;
                    score[nt][2] = key0 <= qabs1 ? score[nt][2] : -CUDART_INF_F;
                    score[nt][3] = key1 <= qabs1 ? score[nt][3] : -CUDART_INF_F;
                }
                bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
                bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
            }
            bm0 = warp_max<4>(bm0, FullMask);
            bm1 = warp_max<4>(bm1, FullMask);

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
            for (int nt = 0; nt < QKNt; ++nt) {
                const int col0  = nt * 8 + 2 * lid;
                const int col1  = col0 + 1;
                const float p00 = score[nt][0] > -CUDART_INF_F
                                      ? exp2_approx(__fmaf_rn(score[nt][0], scale_l2, -nm0_scaled))
                                      : 0.0f;
                const float p01 = score[nt][1] > -CUDART_INF_F
                                      ? exp2_approx(__fmaf_rn(score[nt][1], scale_l2, -nm0_scaled))
                                      : 0.0f;
                const float p10 = score[nt][2] > -CUDART_INF_F
                                      ? exp2_approx(__fmaf_rn(score[nt][2], scale_l2, -nm1_scaled))
                                      : 0.0f;
                const float p11 = score[nt][3] > -CUDART_INF_F
                                      ? exp2_approx(__fmaf_rn(score[nt][3], scale_l2, -nm1_scaled))
                                      : 0.0f;
                bl0 += p00 + p01;
                bl1 += p10 + p11;
                p_s[row0 * Bc + gqa_prefill_swz(row0, col0)] = __float2bfloat16(p00);
                p_s[row0 * Bc + gqa_prefill_swz(row0, col1)] = __float2bfloat16(p01);
                p_s[row1 * Bc + gqa_prefill_swz(row1, col0)] = __float2bfloat16(p10);
                p_s[row1 * Bc + gqa_prefill_swz(row1, col1)] = __float2bfloat16(p11);
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
            const int worker_tid = tid - ProducerWarps * 32;
#pragma unroll 1
            for (int chunk = worker_tid; chunk < Bc * (D / 8); chunk += WorkerThreads) {
                const int key_l    = chunk / (D / 8);
                const int dc       = chunk - key_l * (D / 8);
                const int d        = dc * 8;
                const int key      = k0 + key_l;
                __nv_bfloat16* dst = &v_bf16[key_l * D + gqa_prefill_swz(key_l, d)];
                if (key <= max_query_abs) {
                    const int grp = d >> 4;
                    const float vs =
                        detail::decode_nvfp4_e4m3(v_scale_s[key_l * Groups + grp]);
                    store_vec(dst, gqa_nvfp4_dequant_bf16x8(&v_codes[key_l * CodeW + d / 2], vs));
                } else {
                    store_vec(dst, make_int4(0, 0, 0, 0));
                }
            }
        }
        __syncthreads();

        const bool has_next = kb + 1 < key_blocks;
        if (has_next) { issue_kv_tile((kb + 1) * Bc); }

        const int row_tile = warp % kGqaPrefillNvfp4RowTiles;
        const int d_slice  = warp / kGqaPrefillNvfp4RowTiles;
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

#pragma unroll
        for (int k = 0; k < PVKs; ++k) {
            unsigned pf[4];
            const int pcol = k * 16 + a_coloff;
            ldmatrix_x4(pf[0], pf[1], pf[2], pf[3],
                        smem_addr(&p_s[(row_base + a_row_offset) * Bc +
                                       gqa_prefill_swz(row_base + a_row_offset, pcol)]));
#pragma unroll
            for (int n = 0; n < PVNtPerWarp; ++n) {
                const int global_n = d_slice * PVNtPerWarp + n;
                unsigned vf[2];
                const int vrow = k * 16 + b_koff + b_rin;
                const int vcol = global_n * 8;
                ldmatrix_x2_t(vf[0], vf[1],
                              smem_addr(&v_bf16[vrow * D + gqa_prefill_swz(vrow, vcol)]));
                mma_bf16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], pf[0], pf[1], pf[2], pf[3],
                         vf[0], vf[1]);
            }
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

    const int row_tile = warp % kGqaPrefillNvfp4RowTiles;
    const int d_slice  = warp / kGqaPrefillNvfp4RowTiles;
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
