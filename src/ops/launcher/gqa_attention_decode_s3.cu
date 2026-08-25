#define NINFER_GQA_DECODE_SKIP_REDUCE_KERNEL
#include "ops/launcher/gqa_attention_s3_launch.h"

#include "ops/common/math.h"
#include "ops/kernel/gqa_attention_decode_nvfp4s3.cuh"
#include "core/device.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

namespace ninfer::ops::detail {

template <typename Geometry, int TokenTile, bool MultiBatch, bool Masked, typename CacheInput>
void launch_tc_partial_nvfp4s3(const Tensor& q, CacheInput input, const Tensor& pos, float scale,
                               PagedKVBatchLayerView cache, const GqaSmallTInvocation& invocation,
                               std::int32_t logical_capacity, std::int32_t implementation_window,
                               std::int32_t splits, Tensor& partial_acc, Tensor& partial_m,
                               Tensor& partial_l, cudaStream_t stream, float keep_frac,
                               const Tensor& keep_tiles, const Tensor& keep_count,
                               const Tensor& split_off, std::int32_t keep_stride) {
    Tensor& cache_k       = cache.k_pages;
    Tensor& cache_v       = cache.v_pages;
    Tensor& cache_k_scale = cache.k_scale_pages;
    Tensor& cache_v_scale = cache.v_scale_pages;
    if constexpr (CacheInput::writes_cache) {
        // The Sage 16-key V block scale is cross-key state: when a block straddles a
        // split boundary, the fused per-split fill races across CTAs. Run the
        // block-aligned fill first (stream-ordered), so the tc kernel reads settled
        // codes/scales.
        const int fill_units = invocation.batch_size *
                               (TokenTile * Geometry::KVHeads * kGqaNvfp4Groups +
                                (div_up(TokenTile, 16) + 1) * Geometry::KVHeads *
                                    (kGqaHeadDim / 2));
        gqa_attention_decode_fill_nvfp4s3_kernel<Geometry, TokenTile, MultiBatch, Masked>
            <<<div_up(fill_units, 256), 256, 0, stream>>>(
                input.k, input.v, static_cast<const std::int32_t*>(pos.data),
                static_cast<const std::int32_t*>(cache.block_tables.data),
                invocation.table_rows == nullptr
                    ? nullptr
                    : static_cast<const std::int32_t*>(invocation.table_rows->data),
                cache.block_tables.ne[0], static_cast<std::uint8_t*>(cache_k.data),
                static_cast<std::uint8_t*>(cache_v.data),
                static_cast<std::uint8_t*>(cache_k_scale.data),
                static_cast<std::uint8_t*>(cache_v_scale.data), invocation.full_width,
                invocation.column_begin, invocation.batch_size,
                invocation.valid_columns == nullptr
                    ? nullptr
                    : static_cast<const std::int32_t*>(invocation.valid_columns->data));
        CUDA_CHECK(cudaGetLastError());
    }
    // Prefill-only skip (exact NVFP4). Sage decode is always exact.
    (void)keep_frac;
    const bool tile_skip = false;
    // Sage decode defaults to strict (BF16) P/PV on the sage cache (QK stays
    // NVFP4). FP4-P is the quality floor (+0.03–0.06 NLL, MTP accept collapse);
    // at T=1 the BF16-PV kernel is speed-parity with FP4-PV because decode is
    // DRAM/latency-bound, not MMA-rate-bound. Prefill keeps FP4-PV (O(n²) wall).
    // NINFER_S3_STRICT_PV=0 restores the old FP4-P recipe.
    static const bool strict_pv = [] {
        const char* e = std::getenv("NINFER_S3_STRICT_PV");
        return e == nullptr || e[0] != '0';
    }();
    auto launch = [&]<int WarpsPerCta, int MinBlocksPerSm, int KeyBlock, bool DynamicArena,
                      bool Strict, bool TileSkip>() {
        const dim3 grid(Geometry::KVHeads, splits, invocation.batch_size);
        constexpr int kBr    = ((TokenTile * Geometry::GroupSize + 15) / 16) * 16;
        constexpr int kP4Row = (KeyBlock == 64) ? 48 : 32;
        constexpr int kPBlk  = KeyBlock / 16;
        constexpr std::size_t kDynamicBytes =
            DynamicArena
                ? (Strict
                        ? static_cast<std::size_t>(4 * KeyBlock * kGqaNvfp4CodeWidth +
                                                    2 * kGqaHeadDim * 4 + KeyBlock * kGqaNvfp4Groups +
                                                    kBr * KeyBlock * 2)
                        : static_cast<std::size_t>(2 * KeyBlock * kGqaNvfp4CodeWidth +
                                                   kBr * kP4Row + kBr * kPBlk + kGqaHeadDim * kP4Row +
                                                   kGqaHeadDim * 4 + KeyBlock * kGqaNvfp4Groups))
                : 0u;
        if constexpr (DynamicArena) {
            static const cudaError_t attr = cudaFuncSetAttribute(
                gqa_attention_decode_nvfp4s3_tiled_kernel<Geometry, TokenTile, WarpsPerCta,
                                                           MinBlocksPerSm, KeyBlock, DynamicArena,
                                                           MultiBatch, Masked, CacheInput,
                                                           false, Strict, TileSkip>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kDynamicBytes));
            CUDA_CHECK(attr);
        }
        if constexpr (TileSkip) {
            // Rank the tier's Bc-key tiles before the partial kernel; the partial
            // walks its per-split keep-list slice (keep_tiles/split_off).
            std::int32_t rank_tiles_pow2 = 1;
            const std::int32_t rank_tiles = div_up(implementation_window, KeyBlock);
            while (rank_tiles_pow2 < rank_tiles) { rank_tiles_pow2 <<= 1; }
            gqa_attention_decode_rank_nvfp4s3_kernel<Geometry, KeyBlock, MultiBatch>
                <<<dim3(Geometry::KVHeads, 1, MultiBatch ? invocation.batch_size : 1), 256,
                    gqa_s3_decode_rank_smem_bytes(rank_tiles_pow2), stream>>>(
                    static_cast<const __nv_bfloat16*>(q.data),
                    static_cast<const std::int32_t*>(pos.data),
                    static_cast<const std::int32_t*>(cache.block_tables.data),
                    invocation.table_rows == nullptr
                        ? nullptr
                        : static_cast<const std::int32_t*>(invocation.table_rows->data),
                    cache.block_tables.ne[0], invocation.full_width, invocation.column_begin,
                    splits, keep_frac, static_cast<const float*>(cache.k_mean_pages.data),
                    static_cast<std::int32_t*>(keep_tiles.data),
                    static_cast<std::int32_t*>(keep_count.data),
                    static_cast<std::int32_t*>(split_off.data), rank_tiles_pow2,
                    keep_stride);
            CUDA_CHECK(cudaGetLastError());
        }
        gqa_attention_decode_nvfp4s3_tiled_kernel<Geometry, TokenTile, WarpsPerCta, MinBlocksPerSm,
                                                   KeyBlock, DynamicArena, MultiBatch, Masked,
                                                   CacheInput, false, Strict, TileSkip>
            <<<grid, WarpsPerCta * 32, kDynamicBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data), input,
                static_cast<const std::int32_t*>(pos.data),
                static_cast<std::uint8_t*>(cache_k.data), static_cast<std::uint8_t*>(cache_v.data),
                static_cast<std::uint8_t*>(cache_k_scale.data),
                static_cast<std::uint8_t*>(cache_v_scale.data),
                static_cast<const std::int32_t*>(cache.block_tables.data),
                invocation.valid_columns == nullptr
                    ? nullptr
                    : static_cast<const std::int32_t*>(invocation.valid_columns->data),
                invocation.table_rows == nullptr
                    ? nullptr
                    : static_cast<const std::int32_t*>(invocation.table_rows->data),
                cache.block_tables.ne[0], invocation.full_width, invocation.column_begin,
                logical_capacity, scale, static_cast<__nv_bfloat16*>(partial_acc.data),
                static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data),
                TileSkip ? static_cast<const std::int32_t*>(keep_tiles.data) : nullptr,
                TileSkip ? static_cast<const std::int32_t*>(split_off.data) : nullptr, keep_stride);
    };
    auto launch_tier = [&]<int WarpsPerCta, int MinBlocksPerSm, int KeyBlock, bool DynamicArena>() {
        if (tile_skip) {
            if (strict_pv) {
                launch.template operator()<WarpsPerCta, MinBlocksPerSm, KeyBlock, DynamicArena,
                                           true, true>();
            } else {
                launch.template operator()<WarpsPerCta, MinBlocksPerSm, KeyBlock, DynamicArena,
                                           false, true>();
            }
        } else {
            if (strict_pv) {
                launch.template operator()<WarpsPerCta, MinBlocksPerSm, KeyBlock, DynamicArena,
                                           true, false>();
            } else {
                launch.template operator()<WarpsPerCta, MinBlocksPerSm, KeyBlock, DynamicArena,
                                           false, false>();
            }
        }
    };
    if constexpr (TokenTile == 6) {
        if (implementation_window > 128 && implementation_window <= 160) {
            launch_tier.template operator()<24, 1, 32, false>();
        } else if (implementation_window <= 2054) {
            launch_tier.template operator()<12, 1, 32, false>();
        } else if (implementation_window <= 8198) {
            launch_tier.template operator()<12, 1, 64, true>();
        } else {
            launch_tier.template operator()<6, 2, 32, false>();
        }
    } else if constexpr (TokenTile == 5) {
        if constexpr (Geometry::GroupSize == 6) {
            if (implementation_window > 128 && implementation_window <= 512) {
                launch_tier.template operator()<32, 1, 32, false>();
            } else if (implementation_window <= 1029) {
                launch_tier.template operator()<16, 1, 32, false>();
            } else {
                launch_tier.template operator()<8, 2, 32, false>();
            }
        } else {
            if (implementation_window > 128 && implementation_window <= 512) {
                launch_tier.template operator()<24, 1, 32, false>();
            } else if (implementation_window <= 1029) {
                launch_tier.template operator()<24, 1, 32, false>();
            } else if (implementation_window <= 4096) {
                launch_tier.template operator()<12, 1, 32, false>();
            } else {
                launch_tier.template operator()<6, 2, 32, false>();
            }
        }
    } else if constexpr (TokenTile == 4) {
        if (implementation_window <= 1029) {
            launch_tier.template operator()<16, 1, 32, false>();
        } else {
            launch_tier.template operator()<8, 2, 32, false>();
        }
    } else {
        launch_tier.template operator()<8, 2, 32, false>();
    }
    CUDA_CHECK(cudaGetLastError());
}

#define NINFER_S3_DECODE_INSTANTIATE_IN(Geom, T, MB, Mask, In)                                     \
    template void launch_tc_partial_nvfp4s3<Geom, T, MB, Mask, In>(                                \
        const Tensor&, In, const Tensor&, float, PagedKVBatchLayerView,                            \
        const GqaSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&,     \
        Tensor&, cudaStream_t, float, const Tensor&, const Tensor&, const Tensor&, std::int32_t)

#define NINFER_S3_DECODE_INSTANTIATE_T(Geom, T)                                                    \
    NINFER_S3_DECODE_INSTANTIATE_IN(Geom, T, false, false, GqaAppendInput);                         \
    NINFER_S3_DECODE_INSTANTIATE_IN(Geom, T, false, true, GqaAppendInput);                          \
    NINFER_S3_DECODE_INSTANTIATE_IN(Geom, T, true, false, GqaAppendInput);                          \
    NINFER_S3_DECODE_INSTANTIATE_IN(Geom, T, true, true, GqaAppendInput);                           \
    NINFER_S3_DECODE_INSTANTIATE_IN(Geom, T, false, false, GqaCachedInput);                         \
    NINFER_S3_DECODE_INSTANTIATE_IN(Geom, T, false, true, GqaCachedInput);                          \
    NINFER_S3_DECODE_INSTANTIATE_IN(Geom, T, true, false, GqaCachedInput);                          \
    NINFER_S3_DECODE_INSTANTIATE_IN(Geom, T, true, true, GqaCachedInput)

#define NINFER_S3_DECODE_INSTANTIATE(Geom)                                                         \
    NINFER_S3_DECODE_INSTANTIATE_T(Geom, 1);                                                       \
    NINFER_S3_DECODE_INSTANTIATE_T(Geom, 2);                                                       \
    NINFER_S3_DECODE_INSTANTIATE_T(Geom, 3);                                                       \
    NINFER_S3_DECODE_INSTANTIATE_T(Geom, 4);                                                       \
    NINFER_S3_DECODE_INSTANTIATE_T(Geom, 5);                                                       \
    NINFER_S3_DECODE_INSTANTIATE_T(Geom, 6)

NINFER_S3_DECODE_INSTANTIATE(Gqa27Geometry);
NINFER_S3_DECODE_INSTANTIATE(Gqa35Geometry);

#undef NINFER_S3_DECODE_INSTANTIATE
#undef NINFER_S3_DECODE_INSTANTIATE_T
#undef NINFER_S3_DECODE_INSTANTIATE_IN

} // namespace ninfer::ops::detail
