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

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdexcept>


namespace ninfer::ops::detail {
namespace {


template <typename Geometry, typename CacheView, typename Metadata>
void gqa_attention_prompt_attention_launch_for(const Tensor& q, const Tensor& positions,
                                                float scale, const CacheView& cache,
                                                Metadata metadata, Tensor& out,
                                                cudaStream_t stream, float keep_frac = 1.0f,
                                                float xattn_tau = 1.0f,
                                                std::int32_t xattn_min_len = 8192,
                                                GqaS3PrefillDump* dump = nullptr,
                                                std::uint32_t* dbg_regs = nullptr,
                                                std::uint8_t* dbg_q = nullptr,
                                                void* xattn_scratch = nullptr,
                                                GqaExecutionEnvelope envelope = {
                                                    1, kGqaAttentionMaximumVisibleKeys}) {
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
        gqa_attention_prefill_nvfp4_kernel<Geometry, Metadata>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kGqaPrefillNvfp4SmemBytes);
    CUDA_CHECK(attr_nvfp4);

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
    if (cache.dtype == DType::U8 &&
        (keep_frac < 1.0f || xattn_tau < 1.0f || dump != nullptr)) {
        const bool skip_xattn_to_dense =
            xattn_tau < 1.0f && !(keep_frac < 1.0f) &&
            envelope.max_visible_keys < static_cast<std::uint32_t>(xattn_min_len);
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
        const dim3 attention_grid(static_cast<unsigned>(div_up(tokens, kGqaPrefillNvfp4Br)),
                                  static_cast<unsigned>(Geometry::QHeads), 1u);
        const Tensor& cache_k_scale = cache.k_scale_pages;
        const Tensor& cache_v_scale = cache.v_scale_pages;
        gqa_attention_prefill_nvfp4_kernel<Geometry, Metadata>
            <<<attention_grid, kGqaPrefillNvfp4Threads, kGqaPrefillNvfp4SmemBytes, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data),
                static_cast<const std::uint8_t*>(cache_k.data),
                static_cast<const std::uint8_t*>(cache_v.data),
                static_cast<const std::uint8_t*>(cache_k_scale.data),
                static_cast<const std::uint8_t*>(cache_v_scale.data), metadata,
                static_cast<const std::int32_t*>(positions.data), scale,
                static_cast<__nv_bfloat16*>(out.data), tokens);
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
        Tensor& cache_k_scale    = cache.k_scale_pages;
        Tensor& cache_v_scale    = cache.v_scale_pages;
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

} // namespace

void gqa_attention_prompt_attention_launch(const Tensor& q, const Tensor& positions, float scale,
                                           const PagedKVLayerView& cache, Tensor& out,
                                           cudaStream_t stream, float keep_frac, float xattn_tau,
                                           std::int32_t xattn_min_len, GqaS3PrefillDump* dump,
                                           std::uint32_t* dbg_regs, std::uint8_t* dbg_q,
                                           void* xattn_scratch, GqaExecutionEnvelope envelope) {
    const GqaPrefillDirectMetadata metadata{
        static_cast<const std::int32_t*>(cache.block_table.data)};
    if (q.ne[1] == Gqa27Geometry::QHeads) {
        gqa_attention_prompt_attention_launch_for<Gqa27Geometry>(
            q, positions, scale, cache, metadata, out, stream, keep_frac, xattn_tau, xattn_min_len,
            dump, dbg_regs, dbg_q, xattn_scratch, envelope);
        return;
    }
    gqa_attention_prompt_attention_launch_for<Gqa35Geometry>(
        q, positions, scale, cache, metadata, out, stream, keep_frac, xattn_tau, xattn_min_len,
        dump, dbg_regs, dbg_q, xattn_scratch, envelope);
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
                                 Tensor& out, cudaStream_t stream, float keep_frac,
                                 float xattn_tau, std::int32_t xattn_min_len,
                                 GqaS3PrefillDump* dump, void* xattn_scratch,
                                 GqaExecutionEnvelope envelope) {
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
                xattn_min_len, dump, nullptr, nullptr, xattn_scratch, envelope);
            return;
        }
        gqa_kv_append_launch_for<Gqa35Geometry>(k, v, positions, cache, metadata, stream);
        gqa_attention_prompt_attention_launch_for<Gqa35Geometry>(
            q, positions, scale, cache, metadata, out, stream, keep_frac, xattn_tau, xattn_min_len,
            dump, nullptr, nullptr, xattn_scratch, envelope);
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
    const auto launch = [&]<typename Geometry, typename Code, bool HasScale>() {
        gqa_kv_compact_path_kernel<Geometry, Code, HasScale><<<batch, 256, 0, stream>>>(
            static_cast<Code*>(cache.k_pages.data), static_cast<Code*>(cache.v_pages.data),
            HasScale ? static_cast<__half*>(cache.k_scale_pages.data) : nullptr,
            HasScale ? static_cast<__half*>(cache.v_scale_pages.data) : nullptr,
            static_cast<const std::int32_t*>(cache.block_tables.data),
            static_cast<const std::int32_t*>(kv_table_rows.data),
            static_cast<const std::int32_t*>(prefix_lengths.data),
            static_cast<const std::int32_t*>(path.data),
            static_cast<const std::int32_t*>(counts.data), logical_pages, width);
    };
    if (cache.num_kv_heads == Gqa27Geometry::KVHeads) {
        if (cache.dtype == DType::I8) {
            launch.template operator()<Gqa27Geometry, std::int8_t, true>();
        } else {
            launch.template operator()<Gqa27Geometry, __nv_bfloat16, false>();
        }
    } else if (cache.dtype == DType::I8) {
        launch.template operator()<Gqa35Geometry, std::int8_t, true>();
    } else {
        launch.template operator()<Gqa35Geometry, __nv_bfloat16, false>();
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
