// Sage3 NVFP4 prefill attention + fill, compiled as its own translation unit so
// cicc can run in parallel with the BF16/INT8/NVFP4 and TMA launchers.
#include "ops/launcher/gqa_attention_s3_launch.h"

#include "ops/common/math.h"
#include "ops/kernel/gqa_attention_prefill_nvfp4s3.cuh"
#include "core/device.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <stdexcept>

namespace ninfer::ops::detail {

namespace {

inline bool gqa_s3_occ2_enabled() {
    static const bool enabled = [] {
        const char* e = std::getenv("NINFER_S3_OCC2");
        return e == nullptr || e[0] != '0';
    }();
    return enabled;
}

} // namespace

template <typename Geometry, typename CacheView, typename Metadata>
void gqa_s3_prefill_attention_launch(const Tensor& q, const Tensor& positions, float scale,
                                     const CacheView& cache, Metadata metadata, Tensor& out,
                                     cudaStream_t stream, float keep_frac, GqaS3PrefillDump* dump,
                                     std::uint32_t* dbg_regs, std::uint8_t* dbg_q) {
    if (keep_frac != 1.0f) {
        throw std::invalid_argument(
            "gqa_attention: --sage is exact-S3 only; --keep-frac / --xattn-tau require "
            "--kv-dtype nvfp4 without --sage");
    }
    const Tensor& cache_k       = cache.k_pages;
    const Tensor& cache_v       = cache.v_pages;
    const Tensor& cache_k_scale = cache.k_scale_pages;
    const Tensor& cache_v_scale = cache.v_scale_pages;
    const Tensor& cache_k_mean  = cache.k_mean_pages;
    const auto tokens           = static_cast<std::int32_t>(q.ne[2]);

    static const cudaError_t attr_nvfp4s3 = cudaFuncSetAttribute(
        gqa_attention_prefill_nvfp4s3_kernel<Geometry, Metadata>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kGqaPrefillNvfp4s3SmemBytes);
    CUDA_CHECK(attr_nvfp4s3);

    const bool exact_occ2 = gqa_s3_occ2_enabled() && keep_frac == 1.0f && dump == nullptr &&
                            dbg_regs == nullptr && dbg_q == nullptr;
    if (exact_occ2) {
        using Occ2          = GqaPrefillNvfp4s3Occ2;
        auto* const occ2_fn = gqa_attention_prefill_nvfp4s3_occ2_kernel<Geometry, Metadata>;
        static const cudaError_t attr_occ2 = cudaFuncSetAttribute(
            occ2_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, Occ2::SmemBytes);
        CUDA_CHECK(attr_occ2);
        static bool logged_occ2 = false;
        if (!logged_occ2) {
            logged_occ2     = true;
            const char* log = std::getenv("NINFER_S3_OCC_LOG");
            if (log != nullptr && log[0] == '1') {
                int occ = 0;
                CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &occ, occ2_fn, Occ2::Threads, Occ2::SmemBytes));
                int occ_legacy = 0;
                CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    &occ_legacy, gqa_attention_prefill_nvfp4s3_kernel<Geometry, Metadata>,
                    kGqaPrefillNvfp4s3Threads, kGqaPrefillNvfp4s3SmemBytes));
                std::fprintf(stderr,
                             "[s3-prefill] occ2 Br=%d warps=%d smem=%d occ/SM=%d | legacy occ/SM=%d "
                             "smem=%d\n",
                             Occ2::Br, Occ2::Warps, Occ2::SmemBytes, occ, occ_legacy,
                             kGqaPrefillNvfp4s3SmemBytes);
            }
        }
        const dim3 occ2_grid(static_cast<unsigned>(div_up(tokens, Occ2::Br)),
                             static_cast<unsigned>(Geometry::QHeads), 1u);
        occ2_fn<<<occ2_grid, Occ2::Threads, Occ2::SmemBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data),
            static_cast<const std::uint8_t*>(cache_k.data),
            static_cast<const std::uint8_t*>(cache_v.data),
            static_cast<const std::uint8_t*>(cache_k_scale.data),
            static_cast<const std::uint8_t*>(cache_v_scale.data),
            static_cast<const float*>(cache_k_mean.data), metadata,
            static_cast<const std::int32_t*>(positions.data), scale,
            static_cast<__nv_bfloat16*>(out.data), tokens, keep_frac, dump, dbg_regs, dbg_q);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    const dim3 attention_grid(static_cast<unsigned>(div_up(tokens, kGqaPrefillNvfp4s3Br)),
                              static_cast<unsigned>(Geometry::QHeads), 1u);
    gqa_attention_prefill_nvfp4s3_kernel<Geometry, Metadata>
        <<<attention_grid, kGqaPrefillNvfp4s3Threads, kGqaPrefillNvfp4s3SmemBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data),
            static_cast<const std::uint8_t*>(cache_k.data),
            static_cast<const std::uint8_t*>(cache_v.data),
            static_cast<const std::uint8_t*>(cache_k_scale.data),
            static_cast<const std::uint8_t*>(cache_v_scale.data),
            static_cast<const float*>(cache_k_mean.data), metadata,
            static_cast<const std::int32_t*>(positions.data), scale,
            static_cast<__nv_bfloat16*>(out.data), tokens, keep_frac, dump, dbg_regs, dbg_q);
    CUDA_CHECK(cudaGetLastError());
}

template <typename Geometry, typename CacheView, typename Metadata>
void gqa_s3_prefill_fill_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                CacheView cache, Metadata metadata, cudaStream_t stream) {
    const auto tokens     = static_cast<std::int32_t>(k.ne[2]);
    Tensor& cache_k       = cache.k_pages;
    Tensor& cache_v       = cache.v_pages;
    Tensor& cache_k_scale = cache.k_scale_pages;
    Tensor& cache_v_scale = cache.v_scale_pages;
    const std::int64_t kmean_units = cache.k_mean_pages.data != nullptr
                                         ? div_up(tokens, kPagedKVPageSize) *
                                               static_cast<std::int64_t>(Geometry::KVHeads) *
                                               kGqaNvfp4Groups
                                         : 0;
    const std::int64_t fill_units =
        static_cast<std::int64_t>(tokens) * Geometry::KVHeads * kGqaNvfp4Groups +
        (div_up(tokens, 16) + 1) * static_cast<std::int64_t>(Geometry::KVHeads) *
            (kGqaNvfp4HeadDim / 2) +
        kmean_units;
    const int fill_grid = static_cast<int>(div_up(fill_units, static_cast<std::int64_t>(256)));
    gqa_attention_prefill_fill_nvfp4s3_kernel<Geometry, Metadata>
        <<<fill_grid, 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(k.data), static_cast<const __nv_bfloat16*>(v.data),
            static_cast<const std::int32_t*>(positions.data), metadata,
            static_cast<std::uint8_t*>(cache_k.data), static_cast<std::uint8_t*>(cache_v.data),
            static_cast<std::uint8_t*>(cache_k_scale.data),
            static_cast<std::uint8_t*>(cache_v_scale.data),
            static_cast<float*>(cache.k_mean_pages.data), tokens);
    CUDA_CHECK(cudaGetLastError());
}

#define NINFER_S3_PREFILL_INSTANTIATE(Geom, View, Meta)                                            \
    template void gqa_s3_prefill_attention_launch<Geom, View, Meta>(                               \
        const Tensor&, const Tensor&, float, const View&, Meta, Tensor&, cudaStream_t, float,      \
        GqaS3PrefillDump*, std::uint32_t*, std::uint8_t*);                                         \
    template void gqa_s3_prefill_fill_launch<Geom, View, Meta>(                                    \
        const Tensor&, const Tensor&, const Tensor&, View, Meta, cudaStream_t)

NINFER_S3_PREFILL_INSTANTIATE(Gqa27Geometry, PagedKVLayerView, GqaPrefillDirectMetadata);
NINFER_S3_PREFILL_INSTANTIATE(Gqa35Geometry, PagedKVLayerView, GqaPrefillDirectMetadata);
NINFER_S3_PREFILL_INSTANTIATE(Gqa27Geometry, PagedKVBatchLayerView, GqaPrefillBatchMetadata<false>);
NINFER_S3_PREFILL_INSTANTIATE(Gqa27Geometry, PagedKVBatchLayerView, GqaPrefillBatchMetadata<true>);
NINFER_S3_PREFILL_INSTANTIATE(Gqa35Geometry, PagedKVBatchLayerView, GqaPrefillBatchMetadata<false>);
NINFER_S3_PREFILL_INSTANTIATE(Gqa35Geometry, PagedKVBatchLayerView, GqaPrefillBatchMetadata<true>);

#undef NINFER_S3_PREFILL_INSTANTIATE

} // namespace ninfer::ops::detail
