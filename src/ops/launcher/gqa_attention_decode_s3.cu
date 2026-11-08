#define NINFER_GQA_DECODE_SKIP_REDUCE_KERNEL
#include "ops/launcher/gqa_attention_s3_launch.h"

#include "ops/common/math.h"
#include "ops/kernel/gqa_attention_decode_nvfp4s3.cuh"
#include "core/device.h"

#include <cstdint>
#include <cuda_runtime.h>

namespace ninfer::ops::detail {

template <typename Geometry, int TokenTile, bool MultiBatch, bool Masked, typename CacheInput>
void launch_tc_partial_nvfp4s3(const Tensor& q, CacheInput input, const Tensor& pos, float scale,
                               PagedKVBatchLayerView cache, const GqaSmallTInvocation& invocation,
                               std::int32_t logical_capacity, std::int32_t implementation_window,
                               std::int32_t splits, Tensor& partial_acc, Tensor& partial_m,
                               Tensor& partial_l, cudaStream_t stream) {
    Tensor& cache_k       = cache.k_pages;
    Tensor& cache_v       = cache.v_pages;
    Tensor& cache_k_scale = cache.k_scale_pages;
    Tensor& cache_v_scale = cache.v_scale_pages;
    if constexpr (CacheInput::writes_cache) {
        // The Sage 16-key V block scale is cross-key state, so a fill inside the split-KV
        // partial would race across CTAs on a block that straddles a split boundary. The
        // block-aligned fill runs first (stream-ordered) and the partial reads settled codes
        // and scales.
        const int fill_units =
            invocation.batch_size *
            (TokenTile * Geometry::KVHeads * kGqaNvfp4Groups +
             (div_up(TokenTile, 16) + 1) * Geometry::KVHeads * (kGqaHeadDim / 2));
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
    auto launch_tier = [&]<int WarpsPerCta, int MinBlocksPerSm, int KeyBlock, bool DynamicArena>() {
        const dim3 grid(Geometry::KVHeads, splits, invocation.batch_size);
        constexpr int kBr = ((TokenTile * Geometry::GroupSize + 15) / 16) * 16;
        // Must match the kernel's ArenaBytes.
        constexpr std::size_t kDynamicBytes =
            DynamicArena
                ? static_cast<std::size_t>(4 * KeyBlock * kGqaNvfp4CodeWidth + 2 * kGqaHeadDim * 4 +
                                           KeyBlock * kGqaNvfp4Groups + kBr * KeyBlock * 2)
                : 0u;
        const auto kernel =
            gqa_attention_decode_nvfp4s3_tiled_kernel<Geometry, TokenTile, WarpsPerCta,
                                                      MinBlocksPerSm, KeyBlock, DynamicArena,
                                                      MultiBatch, Masked>;
        if constexpr (DynamicArena) {
            static const cudaError_t attr =
                cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     static_cast<int>(kDynamicBytes));
            CUDA_CHECK(attr);
        }
        kernel<<<grid, WarpsPerCta * 32, kDynamicBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data), static_cast<const std::int32_t*>(pos.data),
            static_cast<const std::uint8_t*>(cache_k.data),
            static_cast<const std::uint8_t*>(cache_v.data),
            static_cast<const std::uint8_t*>(cache_k_scale.data),
            static_cast<const std::uint8_t*>(cache_v_scale.data),
            static_cast<const std::int32_t*>(cache.block_tables.data),
            invocation.valid_columns == nullptr
                ? nullptr
                : static_cast<const std::int32_t*>(invocation.valid_columns->data),
            invocation.table_rows == nullptr
                ? nullptr
                : static_cast<const std::int32_t*>(invocation.table_rows->data),
            cache.block_tables.ne[0], invocation.full_width, invocation.column_begin,
            logical_capacity, scale, static_cast<__nv_bfloat16*>(partial_acc.data),
            static_cast<float*>(partial_m.data), static_cast<float*>(partial_l.data));
    };
    if constexpr (TokenTile >= 6) {
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
        static_assert(Geometry::GroupSize == 6, "T=5 decode tiles assume the group of six");
        if (implementation_window > 128 && implementation_window <= 512) {
            launch_tier.template operator()<32, 1, 32, false>();
        } else if (implementation_window <= 1029) {
            launch_tier.template operator()<16, 1, 32, false>();
        } else {
            launch_tier.template operator()<8, 2, 32, false>();
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
        const GqaSmallTInvocation&, std::int32_t, std::int32_t, std::int32_t, Tensor&, Tensor&,    \
        Tensor&, cudaStream_t)

#define NINFER_S3_DECODE_INSTANTIATE_T(Geom, T)                                                    \
    NINFER_S3_DECODE_INSTANTIATE_IN(Geom, T, false, false, GqaAppendInput);                        \
    NINFER_S3_DECODE_INSTANTIATE_IN(Geom, T, false, true, GqaAppendInput);                         \
    NINFER_S3_DECODE_INSTANTIATE_IN(Geom, T, true, false, GqaAppendInput);                         \
    NINFER_S3_DECODE_INSTANTIATE_IN(Geom, T, true, true, GqaAppendInput);                          \
    NINFER_S3_DECODE_INSTANTIATE_IN(Geom, T, false, false, GqaCachedInput)

#define NINFER_S3_DECODE_INSTANTIATE(Geom)                                                         \
    NINFER_S3_DECODE_INSTANTIATE_T(Geom, 1);                                                       \
    NINFER_S3_DECODE_INSTANTIATE_T(Geom, 2);                                                       \
    NINFER_S3_DECODE_INSTANTIATE_T(Geom, 3);                                                       \
    NINFER_S3_DECODE_INSTANTIATE_T(Geom, 4);                                                       \
    NINFER_S3_DECODE_INSTANTIATE_T(Geom, 5);                                                       \
    NINFER_S3_DECODE_INSTANTIATE_T(Geom, 6);                                                       \
    NINFER_S3_DECODE_INSTANTIATE_T(Geom, 7);                                                       \
    NINFER_S3_DECODE_INSTANTIATE_T(Geom, 8)

NINFER_S3_DECODE_INSTANTIATE(Gqa27Geometry);

#undef NINFER_S3_DECODE_INSTANTIATE
#undef NINFER_S3_DECODE_INSTANTIATE_T
#undef NINFER_S3_DECODE_INSTANTIATE_IN

} // namespace ninfer::ops::detail
