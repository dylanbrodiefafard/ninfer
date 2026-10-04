// ninfer::ops - gqa_attention prompt-scale launcher: fill k/v at device
// positions then launch causal attention over absolute cached history.
#include "ops/launcher/gqa_attention.h"
#include "ops/launcher/gqa_attention_s3_launch.h"
#include "ops/launcher/gqa_attention_sparse_launch.h"

#include "ops/common/math.h"
#include "ops/kernel/gqa_attention_prefill_bf16.cuh"
#include "ops/kernel/gqa_attention_prefill_i8.cuh"
#include "ops/kernel/gqa_attention_prefill_nvfp4.cuh"
#include "ops/kernel/gqa_kv_compact.cuh"
#include "core/device.h" // CUDA_CHECK

#include <algorithm>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {


template <typename Geometry, typename CacheView, typename Metadata>
void gqa_attention_prompt_attention_launch_for(
    const Tensor& q, const Tensor& positions, float scale, const CacheView& cache,
    Metadata metadata, Tensor& out, cudaStream_t stream, float keep_frac = 1.0f,
    float xattn_tau = 1.0f, std::int32_t xattn_min_len = 8192, GqaS3PrefillDump* dump = nullptr,
    std::uint32_t* dbg_regs = nullptr, std::uint8_t* dbg_q = nullptr, void* xattn_scratch = nullptr,
    GqaExecutionEnvelope envelope = {1, kGqaAttentionMaximumVisibleKeys},
    const GqaPromptSplit& split   = {}) {
    const Tensor& cache_k = cache.k_pages;
    const Tensor& cache_v = cache.v_pages;
    // Both dtype-specialized kernels exceed the default 48 KiB dynamic-smem ceiling.
    static const cudaError_t attr_bf16 =
        cudaFuncSetAttribute(gqa_attention_prefill_bf16_kernel<Geometry, Metadata>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, kGqaPrefillSmemBytes);
    CUDA_CHECK(attr_bf16);
    static const cudaError_t attr_i8 =
        cudaFuncSetAttribute(gqa_attention_prefill_i8_kernel<Geometry, Metadata>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, kGqaPrefillI8SmemBytes);
    CUDA_CHECK(attr_i8);
    static const cudaError_t attr_nvfp4 = cudaFuncSetAttribute(
        gqa_attention_prefill_nvfp4_kernel<Geometry, Metadata, false>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kGqaPrefillNvfp4SmemBytes);
    CUDA_CHECK(attr_nvfp4);
    static const cudaError_t attr_nvfp4_split = cudaFuncSetAttribute(
        gqa_attention_prefill_nvfp4_kernel<Geometry, Metadata, true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kGqaPrefillNvfp4SmemBytes);
    CUDA_CHECK(attr_nvfp4_split);

    const auto tokens = static_cast<std::int32_t>(q.ne[2]);
    if (cache.dtype == DType::U8 && cache.sage_pv) {
        if (keep_frac != 1.0f || xattn_tau != 1.0f) {
            throw std::invalid_argument(
                "gqa_attention: --sage is exact-S3 only; --keep-frac / --xattn-tau require "
                "--kv-dtype nvfp4 without --sage");
        }
        if (gqa_s3_prefill_tma_try_launch<Geometry>(q, positions, scale, cache, metadata, out,
                                                    stream, keep_frac, dump, dbg_regs, dbg_q)) {
            return;
        }
        gqa_s3_prefill_attention_launch<Geometry>(q, positions, scale, cache, metadata, out, stream,
                                                  keep_frac, dump, dbg_regs, dbg_q);
        return;
    }
    if (cache.dtype == DType::U8 && (keep_frac < 1.0f || xattn_tau < 1.0f || dump != nullptr)) {
        const bool skip_xattn_to_dense =
            xattn_tau < 1.0f && !(keep_frac < 1.0f) &&
            envelope.max_visible_keys <= static_cast<std::uint32_t>(xattn_min_len);
        if (!skip_xattn_to_dense) {
            gqa_sparse_prefill_attention_launch<Geometry>(
                q, positions, scale, cache, metadata, out, stream, keep_frac, xattn_tau,
                xattn_min_len, dump, xattn_scratch, envelope);
            return;
        }
    }
    if (cache.dtype == DType::I8) {
        const dim3 attention_grid(static_cast<unsigned>(div_up(tokens, kGqaPrefillI8Br)),
                                  static_cast<unsigned>(Geometry::QHeads), 1u);
        const Tensor& cache_k_scale = cache.k_scale_pages;
        const Tensor& cache_v_scale = cache.v_scale_pages;
        gqa_attention_prefill_i8_kernel<Geometry, Metadata>
            <<<attention_grid, kGqaPrefillI8Threads, kGqaPrefillI8SmemBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data),
                static_cast<const std::int8_t*>(cache_k.data),
                static_cast<const std::int8_t*>(cache_v.data),
                static_cast<const __half*>(cache_k_scale.data),
                static_cast<const __half*>(cache_v_scale.data), metadata,
                static_cast<const std::int32_t*>(positions.data), scale,
                static_cast<__nv_bfloat16*>(out.data), tokens);
    } else if (cache.dtype == DType::U8) {
        const Tensor& cache_k_scale = cache.k_scale_pages;
        const Tensor& cache_v_scale = cache.v_scale_pages;
        const auto launch_attention =
            [&]<bool Split, int QueryRows = 128>(std::int32_t splits,
                                                 GqaPrefillNvfp4SplitPartials partials) {
                using Tile = GqaPrefillNvfp4Tile<QueryRows>;
                if constexpr (QueryRows == 64) {
                    static const cudaError_t attr = cudaFuncSetAttribute(
                        gqa_attention_prefill_nvfp4_kernel<Geometry, Metadata, Split, QueryRows>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, Tile::SmemBytes);
                    CUDA_CHECK(attr);
                }
                const dim3 attention_grid(static_cast<unsigned>(div_up(tokens, Tile::Br)),
                                          static_cast<unsigned>(Geometry::QHeads),
                                          static_cast<unsigned>(splits));
                gqa_attention_prefill_nvfp4_kernel<Geometry, Metadata, Split, QueryRows>
                    <<<attention_grid, Tile::Threads, Tile::SmemBytes, stream>>>(
                        static_cast<const __nv_bfloat16*>(q.data),
                        static_cast<const std::uint8_t*>(cache_k.data),
                        static_cast<const std::uint8_t*>(cache_v.data),
                        static_cast<const std::uint8_t*>(cache_k_scale.data),
                        static_cast<const std::uint8_t*>(cache_v_scale.data), metadata,
                        static_cast<const std::int32_t*>(positions.data), scale,
                        static_cast<__nv_bfloat16*>(out.data), tokens, partials);
            };
        if (split.splits > 1) {
            const GqaPrefillNvfp4SplitPartials partials{static_cast<float*>(split.acc.data),
                                                        static_cast<float*>(split.m.data),
                                                        static_cast<float*>(split.l.data)};
            launch_attention.template operator()<true>(split.splits, partials);
            CUDA_CHECK(cudaGetLastError());
            gqa_attention_prefill_nvfp4_merge_kernel<Geometry, Metadata>
                <<<dim3(static_cast<unsigned>(tokens), static_cast<unsigned>(Geometry::QHeads)),
                   kGqaPrefillHeadDim, 0, stream>>>(partials, metadata,
                                                    static_cast<__nv_bfloat16*>(out.data), tokens,
                                                    split.splits);
        } else {
            // Short native query batches need more independent CTAs to fill the GPU.
            if (Geometry::QHeads == 24 && tokens <= 256) {
                launch_attention.template operator()<false, 64>(1, {});
            } else {
                launch_attention.template operator()<false>(1, {});
            }
        }
    } else {
        const dim3 attention_grid(static_cast<unsigned>(div_up(tokens, kGqaPrefillBr)),
                                  static_cast<unsigned>(Geometry::QHeads), 1u);
        gqa_attention_prefill_bf16_kernel<Geometry, Metadata>
            <<<attention_grid, kGqaPrefillThreads, kGqaPrefillSmemBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data),
                static_cast<const __nv_bfloat16*>(cache_k.data),
                static_cast<const __nv_bfloat16*>(cache_v.data), metadata,
                static_cast<const std::int32_t*>(positions.data), scale,
                static_cast<__nv_bfloat16*>(out.data), tokens);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <typename Geometry, typename CacheView, typename Metadata>
void gqa_kv_append_launch_for(const Tensor& k, const Tensor& v, const Tensor& positions,
                              CacheView cache, Metadata metadata, cudaStream_t stream) {
    const auto tokens = static_cast<std::int32_t>(k.ne[2]);
    Tensor& cache_k   = cache.k_pages;
    Tensor& cache_v   = cache.v_pages;
    if (cache.dtype == DType::I8) {
        Tensor& cache_k_scale    = cache.k_scale_pages;
        Tensor& cache_v_scale    = cache.v_scale_pages;
        constexpr int kFillBlock = 256;
        // Page-tiled fill is Geometry-templated (Hkv in blockIdx.y). Enable for both
        // Gqa35 (Hkv=2) and Gqa27/Qwen3.8 (Hkv=4); the prior Hkv==2 gate left 27B on the
        // slower per-token warp fill for every INT8 prefill chunk.
        if (tokens >= 128 && (Geometry::KVHeads == 2 || Geometry::KVHeads == 4)) {
            constexpr int kPageBlock     = 256;
            constexpr int kTokensPerTile = 8;
            const int max_tiles          = div_up(tokens + kTokensPerTile - 1, kTokensPerTile);
            const dim3 fill_grid(static_cast<unsigned>(max_tiles),
                                 static_cast<unsigned>(Geometry::KVHeads),
                                 static_cast<unsigned>(kGqaKvQuantGroups));
            gqa_attention_prefill_fill_i8_page_kernel<Geometry, Metadata>
                <<<fill_grid, kPageBlock, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(k.data),
                    static_cast<const __nv_bfloat16*>(v.data),
                    static_cast<const std::int32_t*>(positions.data), metadata,
                    static_cast<std::int8_t*>(cache_k.data),
                    static_cast<std::int8_t*>(cache_v.data),
                    static_cast<__half*>(cache_k_scale.data),
                    static_cast<__half*>(cache_v_scale.data), tokens);
        } else {
            constexpr int kFillWarps = kFillBlock / 32;
            const std::int64_t fill_units =
                static_cast<std::int64_t>(tokens) * Geometry::KVHeads * kGqaKvQuantGroups;
            const int fill_grid =
                static_cast<int>(div_up(fill_units, static_cast<std::int64_t>(kFillWarps)));
            gqa_attention_prefill_fill_i8_kernel<Geometry, Metadata>
                <<<fill_grid, kFillBlock, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(k.data),
                    static_cast<const __nv_bfloat16*>(v.data),
                    static_cast<const std::int32_t*>(positions.data), metadata,
                    static_cast<std::int8_t*>(cache_k.data),
                    static_cast<std::int8_t*>(cache_v.data),
                    static_cast<__half*>(cache_k_scale.data),
                    static_cast<__half*>(cache_v_scale.data), tokens);
        }
        CUDA_CHECK(cudaGetLastError());
    } else if (cache.dtype == DType::U8) {
        Tensor& cache_k_scale = cache.k_scale_pages;
        Tensor& cache_v_scale = cache.v_scale_pages;
        if (cache.sage_pv) {
            gqa_s3_prefill_fill_launch<Geometry>(k, v, positions, cache, metadata, stream);
            return;
        }
        constexpr int kFillBlock = 256;
        if (tokens >= 128 && (Geometry::KVHeads == 2 || Geometry::KVHeads == 4)) {
            constexpr int kPageBlock     = 256;
            constexpr int kTokensPerTile = 8;
            const int max_tiles          = div_up(tokens + kTokensPerTile - 1, kTokensPerTile);
            const dim3 fill_grid(static_cast<unsigned>(max_tiles),
                                 static_cast<unsigned>(Geometry::KVHeads), 1u);
            gqa_attention_prefill_fill_nvfp4_page_kernel<Geometry, Metadata>
                <<<fill_grid, kPageBlock, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(k.data),
                    static_cast<const __nv_bfloat16*>(v.data),
                    static_cast<const std::int32_t*>(positions.data), metadata,
                    static_cast<std::uint8_t*>(cache_k.data),
                    static_cast<std::uint8_t*>(cache_v.data),
                    static_cast<std::uint8_t*>(cache_k_scale.data),
                    static_cast<std::uint8_t*>(cache_v_scale.data), tokens);
        } else {
            const std::int64_t fill_units =
                static_cast<std::int64_t>(tokens) * Geometry::KVHeads * kGqaNvfp4Groups;
            const int fill_grid =
                static_cast<int>(div_up(fill_units, static_cast<std::int64_t>(kFillBlock)));
            gqa_attention_prefill_fill_nvfp4_kernel<Geometry, Metadata>
                <<<fill_grid, kFillBlock, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(k.data),
                    static_cast<const __nv_bfloat16*>(v.data),
                    static_cast<const std::int32_t*>(positions.data), metadata,
                    static_cast<std::uint8_t*>(cache_k.data),
                    static_cast<std::uint8_t*>(cache_v.data),
                    static_cast<std::uint8_t*>(cache_k_scale.data),
                    static_cast<std::uint8_t*>(cache_v_scale.data), tokens);
        }
        CUDA_CHECK(cudaGetLastError());
        if (cache.k_mean_pages.data != nullptr) {
            // +1 covers a fill window that straddles one extra logical page.
            const int pages = div_up(tokens, kPagedKVPageSize) + 1;
            gqa_attention_prefill_kmean_nvfp4_kernel<Geometry, Metadata>
                <<<dim3(static_cast<unsigned>(pages), static_cast<unsigned>(Geometry::KVHeads)),
                   256, 0, stream>>>(static_cast<const std::uint8_t*>(cache_k.data),
                                     static_cast<const std::uint8_t*>(cache_k_scale.data), metadata,
                                     static_cast<const std::int32_t*>(positions.data),
                                     static_cast<float*>(cache.k_mean_pages.data), tokens);
            CUDA_CHECK(cudaGetLastError());
        }
    } else {
        constexpr int kBlock           = Geometry::KVHeads == 4 ? 128 : 96;
        constexpr int kFillVecElems    = 8;
        const std::int64_t kv_elements = static_cast<std::int64_t>(tokens) * Geometry::KVHeads *
                                         (kGqaPrefillHeadDim / kFillVecElems);
        const int fill_grid =
            static_cast<int>(div_up(kv_elements, static_cast<std::int64_t>(kBlock)));
        gqa_attention_prefill_fill_bf16_kernel<Geometry, Metadata>
            <<<fill_grid, kBlock, 0, stream>>>(static_cast<const __nv_bfloat16*>(k.data),
                                               static_cast<const __nv_bfloat16*>(v.data),
                                               static_cast<const std::int32_t*>(positions.data),
                                               metadata, static_cast<__nv_bfloat16*>(cache_k.data),
                                               static_cast<__nv_bfloat16*>(cache_v.data), tokens);
        CUDA_CHECK(cudaGetLastError());
    }
}

// One 512-thread, 87 KiB CTA per SM on the RTX 5090's 170 SMs.
constexpr std::int32_t kPromptSplitSms     = 170;
constexpr std::int32_t kPromptSplitMaximum = 16;
// A split streams at least this many 64-key tiles, amortizing its Q staging and merge traffic.
constexpr std::int32_t kPromptSplitMinimumTiles = 8;
// W * splits bound on the FP32 partial rows (1 KiB per row and q head).
constexpr std::int32_t kPromptSplitRowBudget = 2 * kGqaPromptSplitMaximumWidth;
// Per-CTA fixed cost (Q quantization, first tile fill) in key-tile units.
constexpr std::int32_t kPromptSplitFixedTiles = 1;

std::int32_t prompt_split_ctas(std::int32_t q_heads, std::int32_t width) {
    return div_up(width, kGqaPrefillNvfp4Br) * q_heads;
}

std::int32_t prompt_key_tiles(std::uint32_t visible_keys) {
    return static_cast<std::int32_t>(
        div_up(visible_keys, static_cast<std::uint32_t>(kGqaPrefillNvfp4Bc)));
}

} // namespace

std::int32_t gqa_attention_prompt_split_capacity(std::int32_t q_heads, std::int32_t width,
                                                 std::uint32_t visible_keys) {
    if (width <= 0) { return 1; }
    const std::int32_t capacity =
        std::min({kPromptSplitMaximum, prompt_key_tiles(visible_keys) / kPromptSplitMinimumTiles,
                  kPromptSplitRowBudget / width});
    // Reserve nothing unless some admitted split count needs fewer waves per key tile than the
    // unsplit grid; otherwise no history length makes splitting pay.
    const std::int64_t ctas  = prompt_split_ctas(q_heads, width);
    const std::int64_t waves = div_up(ctas, std::int64_t{kPromptSplitSms});
    for (std::int32_t splits = 2; splits <= capacity; ++splits) {
        if (div_up(ctas * splits, std::int64_t{kPromptSplitSms}) < waves * splits) {
            return capacity;
        }
    }
    return 1;
}

std::int32_t gqa_attention_prompt_splits(std::int32_t q_heads, std::int32_t width,
                                         std::uint32_t visible_keys) {
    const std::int32_t capacity = gqa_attention_prompt_split_capacity(q_heads, width, visible_keys);
    const std::int64_t ctas     = prompt_split_ctas(q_heads, width);
    const std::int64_t tiles    = prompt_key_tiles(visible_keys);
    const auto cost             = [&](std::int64_t splits) {
        const std::int64_t waves = div_up(ctas * splits, std::int64_t{kPromptSplitSms});
        return waves * (div_up(tiles, splits) + kPromptSplitFixedTiles);
    };
    std::int32_t best = 1;
    for (std::int32_t splits = 2; splits <= capacity; ++splits) {
        if (cost(splits) < cost(best)) { best = splits; }
    }
    return best;
}

void gqa_attention_prompt_attention_launch(const Tensor& q, const Tensor& positions, float scale,
                                           const PagedKVLayerView& cache, Tensor& out,
                                           cudaStream_t stream, float keep_frac, float xattn_tau,
                                           std::int32_t xattn_min_len, GqaS3PrefillDump* dump,
                                           std::uint32_t* dbg_regs, std::uint8_t* dbg_q,
                                           void* xattn_scratch, GqaExecutionEnvelope envelope,
                                           const GqaPromptSplit& split) {
    const GqaPrefillDirectMetadata metadata{
        static_cast<const std::int32_t*>(cache.block_table.data)};
    if (q.ne[1] == Gqa27Geometry::QHeads) {
        gqa_attention_prompt_attention_launch_for<Gqa27Geometry>(
            q, positions, scale, cache, metadata, out, stream, keep_frac, xattn_tau, xattn_min_len,
            dump, dbg_regs, dbg_q, xattn_scratch, envelope, split);
        return;
    }
    gqa_attention_prompt_attention_launch_for<Gqa35Geometry>(
        q, positions, scale, cache, metadata, out, stream, keep_frac, xattn_tau, xattn_min_len,
        dump, dbg_regs, dbg_q, xattn_scratch, envelope, split);
}

void gqa_kv_append_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                          PagedKVLayerView cache, cudaStream_t stream) {
    const GqaPrefillDirectMetadata metadata{
        static_cast<const std::int32_t*>(cache.block_table.data)};
    if (k.ne[1] == Gqa27Geometry::KVHeads) {
        gqa_kv_append_launch_for<Gqa27Geometry>(k, v, positions, cache, metadata, stream);
        return;
    }
    gqa_kv_append_launch_for<Gqa35Geometry>(k, v, positions, cache, metadata, stream);
}

void gqa_attention_prompt_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                 const Tensor& positions, const Tensor& valid_columns,
                                 const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
                                 Tensor& out, cudaStream_t stream, float keep_frac, float xattn_tau,
                                 std::int32_t xattn_min_len, GqaS3PrefillDump* dump,
                                 void* xattn_scratch, GqaExecutionEnvelope envelope,
                                 const GqaPromptSplit& split) {
    const auto launch = [&]<bool Masked>() {
        const GqaPrefillBatchMetadata<Masked> metadata{
            .tables = static_cast<const std::int32_t*>(cache.block_tables.data),
            .valid_columns =
                Masked ? static_cast<const std::int32_t*>(valid_columns.data) : nullptr,
            .table_rows   = static_cast<const std::int32_t*>(table_rows.data),
            .table_stride = cache.block_tables.ne[0],
        };
        if (q.ne[1] == Gqa27Geometry::QHeads) {
            gqa_kv_append_launch_for<Gqa27Geometry>(k, v, positions, cache, metadata, stream);
            gqa_attention_prompt_attention_launch_for<Gqa27Geometry>(
                q, positions, scale, cache, metadata, out, stream, keep_frac, xattn_tau,
                xattn_min_len, dump, nullptr, nullptr, xattn_scratch, envelope, split);
            return;
        }
        gqa_kv_append_launch_for<Gqa35Geometry>(k, v, positions, cache, metadata, stream);
        gqa_attention_prompt_attention_launch_for<Gqa35Geometry>(
            q, positions, scale, cache, metadata, out, stream, keep_frac, xattn_tau, xattn_min_len,
            dump, nullptr, nullptr, xattn_scratch, envelope, split);
    };
    if (valid_columns.data == nullptr) {
        launch.template operator()<false>();
    } else {
        launch.template operator()<true>();
    }
}

void gqa_kv_compact_path_launch(PagedKVBatchLayerView cache, const Tensor& kv_table_rows,
                                const Tensor& prefix_lengths, const Tensor& path,
                                const Tensor& counts, cudaStream_t stream) {
    const std::int32_t batch         = counts.ne[0];
    const std::int32_t logical_pages = cache.block_tables.ne[0];
    const std::int32_t width         = path.ne[0];
    if (cache.dtype != DType::BF16 && cache.dtype != DType::I8 && cache.dtype != DType::U8) {
        throw std::invalid_argument("gqa_kv_compact_path: unsupported cache dtype");
    }
    const auto launch = [&]<typename Geometry, typename Code, int CodeLeading, typename Scale,
                            int ScaleLeading, bool HasScale>() {
        gqa_kv_compact_path_kernel<Geometry, Code, CodeLeading, Scale, ScaleLeading, HasScale>
            <<<batch, 256, 0, stream>>>(
                static_cast<Code*>(cache.k_pages.data), static_cast<Code*>(cache.v_pages.data),
                HasScale ? static_cast<Scale*>(cache.k_scale_pages.data) : nullptr,
                HasScale ? static_cast<Scale*>(cache.v_scale_pages.data) : nullptr,
                static_cast<const std::int32_t*>(cache.block_tables.data),
                static_cast<const std::int32_t*>(kv_table_rows.data),
                static_cast<const std::int32_t*>(prefix_lengths.data),
                static_cast<const std::int32_t*>(path.data),
                static_cast<const std::int32_t*>(counts.data), logical_pages, width);
    };
    const auto launch_geometry = [&]<typename Geometry>() {
        if (cache.dtype == DType::U8) {
            // NVFP4: 128 code bytes + 16 e4m3 groups per token, not the BF16/I8 256-wide layout.
            launch.template operator()<Geometry, std::uint8_t, kGqaNvfp4CodeWidth, std::uint8_t,
                                       kGqaNvfp4Groups, true>();
        } else if (cache.dtype == DType::I8) {
            launch.template operator()<Geometry, std::int8_t, kGqaCompactHeadDim, __half,
                                       kGqaKvQuantGroups, true>();
        } else {
            launch.template
            operator()<Geometry, __nv_bfloat16, kGqaCompactHeadDim, std::uint8_t, 1, false>();
        }
    };
    if (cache.num_kv_heads == Gqa27Geometry::KVHeads) {
        launch_geometry.template operator()<Gqa27Geometry>();
    } else if (cache.num_kv_heads == Gqa35Geometry::KVHeads) {
        launch_geometry.template operator()<Gqa35Geometry>();
    } else {
        throw std::invalid_argument("gqa_kv_compact_path: unsupported KV head geometry");
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
