#pragma once

#include "ops/kernel/gqa_attention_decode.cuh"
#include "ops/kernel/gqa_attention_kv_nvfp4.cuh"

#include <math_constants.h>

namespace ninfer::ops {

template <typename Geometry, int TokenTile, int WarpsPerCta, int MinBlocksPerSm, int KeyBlock,
          bool DynamicArena, bool MultiBatch, bool Masked, bool TreeMasked, typename CacheInput>
__launch_bounds__(WarpsPerCta * 32, MinBlocksPerSm) __global__
    void gqa_attention_decode_nvfp4_tiled_kernel(
        const __nv_bfloat16* q, CacheInput input, const std::int32_t* pos,
        std::uint8_t* cache_k, std::uint8_t* cache_v, std::uint8_t* cache_k_scale,
        std::uint8_t* cache_v_scale, const std::int32_t* block_tables,
        const std::int32_t* valid_columns, const std::int32_t* ancestor_mask,
        const std::int32_t* prefix_lengths, const std::int32_t* table_rows,
        std::int32_t table_stride, std::int32_t full_width, std::int32_t column_begin,
        std::int32_t logical_capacity, float scale, __nv_bfloat16* partial_acc, float* partial_m,
        float* partial_l) {
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
    constexpr int ConsumerWarpsPerTile = Wc / RowTiles;
    constexpr int PVNtPerWarp          = D / (ConsumerWarpsPerTile * 8);
    constexpr int PVKs                 = Bc / 16;
    constexpr int PageIds              = 64;
    constexpr int ProducerThreads      = RowTiles * 32;
    constexpr int VLoaderThreads       = Threads - ProducerThreads;
    constexpr float Log2E              = 1.4426950408889634074f;
    constexpr unsigned FullMask        = 0xffffffffu;

    static_assert(TokenTile >= 1 && TokenTile <= 6);
    static_assert(Bc == 32 || Bc == 64);
    static_assert(RowTiles >= 1 && RowTiles <= 3);
    static_assert(Wc % RowTiles == 0);
    static_assert(PVNtPerWarp == 2 || PVNtPerWarp == 4 || PVNtPerWarp == 8 || PVNtPerWarp == 16);

    __shared__ __align__(16) std::uint8_t q_s[Br * CodeW];
    __shared__ __align__(16) std::uint8_t
        static_r_s[DynamicArena ? 16 : (Bc * CodeW + Bc * CodeW + 2 * Bc * D)];
    extern __shared__ __align__(16) std::uint8_t dynamic_r_s_nvfp4[];
    std::uint8_t* r_s        = DynamicArena ? dynamic_r_s_nvfp4 : static_r_s;
    std::uint8_t* q_codes    = q_s;
    std::uint8_t* k_codes    = r_s;
    std::uint8_t* v_codes    = r_s + Bc * CodeW;
    __nv_bfloat16* v_bf16    = reinterpret_cast<__nv_bfloat16*>(r_s + 2 * Bc * CodeW);
    __shared__ __align__(16) __nv_bfloat16 p_s[Br * Bc];
    __shared__ float alpha_s[Br];
    __shared__ __align__(16) std::uint8_t q_scale_s[Br * Groups];
    __shared__ __align__(16) std::uint8_t k_scale_s[Bc * Groups];
    __shared__ __align__(16) std::uint8_t v_scale_s[Bc * Groups];
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
    if constexpr (TreeMasked) {
        ancestor_mask += column_base;
    } else {
        (void)ancestor_mask;
        (void)prefix_lengths;
    }
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

    if constexpr (CacheInput::writes_cache) {
        for (int unit = tid; unit < valid_tokens * Groups; unit += Threads) {
            const int token    = unit / Groups;
            const int grp      = unit - token * Groups;
            const int position = pos[token];
            if (position < split_start || position >= split_end) { continue; }
            const int physical_page = paged_kv_physical_page(block_table, position);
            const int page_offset   = position & kPagedKVPageMask;
            const int d0            = grp * kGqaNvfp4Group;
            std::uint32_t k_lo = 0, k_hi = 0, v_lo = 0, v_hi = 0;
            std::uint8_t k_sc = 0, v_sc = 0;
            gqa_nvfp4_quantize_bf16x16(&input.k[gqa_kv_new_index<Geometry>(kv_head, d0, token)],
                                       k_lo, k_hi, k_sc);
            gqa_nvfp4_quantize_bf16x16(&input.v[gqa_kv_new_index<Geometry>(kv_head, d0, token)],
                                       v_lo, v_hi, v_sc);
            const std::int64_t code =
                gqa_nvfp4_code_index<Geometry>(physical_page, kv_head, grp * 8, page_offset);
            *reinterpret_cast<std::uint32_t*>(cache_k + code)     = k_lo;
            *reinterpret_cast<std::uint32_t*>(cache_k + code + 4) = k_hi;
            *reinterpret_cast<std::uint32_t*>(cache_v + code)     = v_lo;
            *reinterpret_cast<std::uint32_t*>(cache_v + code + 4) = v_hi;
            cache_k_scale[gqa_nvfp4_scale_index<Geometry>(physical_page, kv_head, grp,
                                                          page_offset)] = k_sc;
            cache_v_scale[gqa_nvfp4_scale_index<Geometry>(physical_page, kv_head, grp,
                                                          page_offset)] = v_sc;
        }
        __syncthreads();
    }

    for (int i = tid; i < Br * CodeW; i += Threads) { q_codes[i] = 0; }
    for (int i = tid; i < Br * Groups; i += Threads) { q_scale_s[i] = 0; }
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
    const int a_coloff      = (a_matrix >> 1) << 3;
    const int b_rin         = lane & 7;
    const int b_koff        = ((lane >> 3) & 1) << 3;

    float acc[PVNtPerWarp][4];
#pragma unroll
    for (int n = 0; n < PVNtPerWarp; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[n][i] = 0.0f; }
    }

    float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F;
    float l0 = 0.0f, l1 = 0.0f;

    auto issue_kv_tile = [&](int tile_k0, int physical_page) {
        for (int key_l = tid; key_l < Bc; key_l += Threads) {
            const int key = tile_k0 + key_l;
            if (key >= split_start && key < split_end) {
                const std::int64_t off = gqa_nvfp4_scale_index<Geometry>(
                    physical_page, kv_head, 0, key & kPagedKVPageMask);
                ninfer::ops::cp_async<16>(&k_scale_s[key_l * Groups], &cache_k_scale[off]);
                ninfer::ops::cp_async<16>(&v_scale_s[key_l * Groups], &cache_v_scale[off]);
            } else {
                store_vec(&k_scale_s[key_l * Groups], make_int4(0, 0, 0, 0));
                store_vec(&v_scale_s[key_l * Groups], make_int4(0, 0, 0, 0));
            }
        }
        for (int chunk = tid; chunk < Bc * (CodeW / 16); chunk += Threads) {
            const int key_l       = chunk / (CodeW / 16);
            const int seg         = chunk - key_l * (CodeW / 16);
            const int logical     = seg * 16;
            const int key         = tile_k0 + key_l;
            const int phys        = gqa_nvfp4_swizzle_byte(key_l, logical);
            std::uint8_t* k_dst   = &k_codes[key_l * CodeW + phys];
            std::uint8_t* v_dst   = &v_codes[key_l * CodeW + logical];
            if (key >= split_start && key < split_end) {
                const std::int64_t off = gqa_nvfp4_code_index<Geometry>(
                    physical_page, kv_head, logical, key & kPagedKVPageMask);
                ninfer::ops::cp_async<16>(k_dst, &cache_k[off]);
                ninfer::ops::cp_async<16>(v_dst, &cache_v[off]);
            } else {
                store_vec(k_dst, make_int4(0, 0, 0, 0));
                store_vec(v_dst, make_int4(0, 0, 0, 0));
            }
        }
        ninfer::ops::cp_commit();
    };

    int physical_page = physical_pages_s[0];
    issue_kv_tile(first_tile, physical_page);
    ninfer::ops::cp_wait<0>();
    __syncthreads();

    for (int kb = 0; kb < key_blocks; ++kb) {
        const int k0 = first_tile + kb * Bc;

        if (warp < RowTiles) {
            const int producer_row_base = warp * 16;
            __nv_bfloat16* p_sw         = &p_s[producer_row_base * Bc];
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
            // Packed-tree verify writes unique cache slots E+j; causal-only
            // (key <= qabs) therefore includes siblings. Ancestor bits hide them,
            // matching the I8/BF16 small-T kernels.
            int prefix_length = 0;
            int bits0         = 0;
            int bits1         = 0;
            if constexpr (TreeMasked) {
                prefix_length = prefix_lengths[batch];
                bits0         = (row0 < RowCount) ? ancestor_mask[token0] : 0;
                bits1         = (row1 < RowCount) ? ancestor_mask[token1] : 0;
            }
            float bm0 = -CUDART_INF_F, bm1 = -CUDART_INF_F;
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int col0 = nt * 8 + 2 * lid;
                const int col1 = col0 + 1;
                const int key0 = k0 + col0;
                const int key1 = k0 + col1;
                score[nt][0] =
                    (gqa_key_in_causal_split(row0, RowCount, key0, split_start, split_end, qabs0) &&
                     gqa_tree_allows_key<TreeMasked>(key0, prefix_length, full_width, bits0))
                        ? score[nt][0] * scale
                        : -CUDART_INF_F;
                score[nt][1] =
                    (gqa_key_in_causal_split(row0, RowCount, key1, split_start, split_end, qabs0) &&
                     gqa_tree_allows_key<TreeMasked>(key1, prefix_length, full_width, bits0))
                        ? score[nt][1] * scale
                        : -CUDART_INF_F;
                score[nt][2] =
                    (gqa_key_in_causal_split(row1, RowCount, key0, split_start, split_end, qabs1) &&
                     gqa_tree_allows_key<TreeMasked>(key0, prefix_length, full_width, bits1))
                        ? score[nt][2] * scale
                        : -CUDART_INF_F;
                score[nt][3] =
                    (gqa_key_in_causal_split(row1, RowCount, key1, split_start, split_end, qabs1) &&
                     gqa_tree_allows_key<TreeMasked>(key1, prefix_length, full_width, bits1))
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
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int col0  = nt * 8 + 2 * lid;
                const int col1  = col0 + 1;
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
                p_sw[(gid + 8) * Bc + gqa_small_t_tc_swz32(gid + 8, col0)] = __float2bfloat16(p10);
                p_sw[(gid + 8) * Bc + gqa_small_t_tc_swz32(gid + 8, col1)] = __float2bfloat16(p11);
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
            const int loader_tid = tid - ProducerThreads;
#pragma unroll 1
            for (int chunk = loader_tid; chunk < Bc * (D / 8); chunk += VLoaderThreads) {
                const int key_l    = chunk / (D / 8);
                const int dc       = chunk - key_l * (D / 8);
                const int d        = dc * 8;
                const int key      = k0 + key_l;
                __nv_bfloat16* dst = &v_bf16[key_l * D + gqa_small_t_tc_swz(key_l, d)];
                if (key >= split_start && key < split_end) {
                    const int grp   = d >> 4;
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
        if (has_next) {
            const int next_k0 = k0 + Bc;
            if ((next_k0 & kPagedKVPageMask) == 0) {
                physical_page = physical_pages_s[(next_k0 >> kPagedKVPageShift) - first_page];
            }
            issue_kv_tile(next_k0, physical_page);
        }

        const int consumer_tile     = warp % RowTiles;
        const int consumer_slice    = warp / RowTiles;
        const int consumer_row_base = consumer_tile * 16;
        __nv_bfloat16* p_consumer   = &p_s[consumer_row_base * Bc];
        const float alpha0          = alpha_s[consumer_row_base + gid];
        const float alpha1          = alpha_s[consumer_row_base + gid + 8];
#pragma unroll
        for (int n = 0; n < PVNtPerWarp; ++n) {
            acc[n][0] *= alpha0;
            acc[n][1] *= alpha0;
            acc[n][2] *= alpha1;
            acc[n][3] *= alpha1;
        }

#pragma unroll
        for (int n = 0; n < PVNtPerWarp; ++n) {
            const int global_n = consumer_slice * PVNtPerWarp + n;
#pragma unroll
            for (int k = 0; k < PVKs; ++k) {
                unsigned pf[4];
                const int pcol = k * 16 + a_coloff;
                ldmatrix_x4(
                    pf[0], pf[1], pf[2], pf[3],
                    smem_addr(&p_consumer[a_row_offset * Bc +
                                          gqa_small_t_tc_swz32(a_row_offset, pcol)]));
                unsigned vf[2];
                const int vrow = k * 16 + b_koff + b_rin;
                const int vcol = global_n * 8;
                ldmatrix_x2_t(vf[0], vf[1],
                              smem_addr(&v_bf16[vrow * D + gqa_small_t_tc_swz(vrow, vcol)]));
                mma_bf16(acc[n][0], acc[n][1], acc[n][2], acc[n][3], pf[0], pf[1], pf[2], pf[3],
                         vf[0], vf[1]);
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

} // namespace ninfer::ops
