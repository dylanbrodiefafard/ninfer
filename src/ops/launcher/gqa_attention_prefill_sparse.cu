// Exact-NVFP4 skip-list prefill (Sparge / XAttention). Own translation unit so
// cicc can run in parallel with the dense NVFP4 and S3 launchers.
#include "ops/launcher/gqa_attention_sparse_launch.h"

#include "ops/common/math.h"
#include "ops/kernel/gqa_attention_prefill_nvfp4_sparse.cuh"
#include "core/device.h"
#include "core/paged_kv_cache.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

namespace ninfer::ops::detail {
namespace {

int gqa_xattn_logical_pages(const PagedKVLayerView& cache) {
    return static_cast<int>(cache.block_table.ne[0]);
}

int gqa_xattn_logical_pages(const PagedKVBatchLayerView& cache) {
    return static_cast<int>(cache.block_tables.ne[0]);
}

struct XattnScratchPool {
    void* ptr         = nullptr;
    std::size_t bytes = 0;
};

XattnScratchPool& xattn_scratch_pool() {
    static XattnScratchPool pool;
    return pool;
}

void* xattn_ensure_scratch(std::size_t bytes) {
    XattnScratchPool& pool = xattn_scratch_pool();
    if (bytes > pool.bytes) {
        if (pool.ptr != nullptr) { CUDA_CHECK(cudaFree(pool.ptr)); }
        CUDA_CHECK(cudaMalloc(&pool.ptr, bytes));
        pool.bytes = bytes;
    }
    return pool.ptr;
}

bool xattn_keep_log_enabled() {
    const char* ev = std::getenv("NINFER_XATTN_KEEP_LOG");
    return ev != nullptr && ev[0] != '\0' && ev[0] != '0';
}

// One line per attention launch: keep% vs identity, mass peak, ranker vs MMA µs.
void xattn_log_keep(const GqaXattnScratchView& scratch, const Tensor& positions, int tokens,
                    int xattn_min_len, float rank_us, float mma_us) {
    std::int32_t base_pos = 0;
    CUDA_CHECK(cudaMemcpy(&base_pos, positions.data, sizeof(base_pos), cudaMemcpyDeviceToHost));
    const int q_heads    = scratch.q_heads;
    const int n_br       = scratch.n_br;
    const int n_kb_cap   = scratch.n_kb;
    const int n_slots    = q_heads * n_br;
    const int max_abs    = base_pos + tokens - 1;
    const bool identity  = (max_abs + 1) < xattn_min_len;
    std::vector<int> counts(static_cast<std::size_t>(n_slots));
    CUDA_CHECK(cudaMemcpy(counts.data(), scratch.count, counts.size() * sizeof(int),
                          cudaMemcpyDeviceToHost));

    double keep_sum = 0.0;
    int keep_min    = 1000;
    int keep_max    = 0;
    int full_n      = 0;
    int ident_n     = 0;
    int denom_n     = 0;
    for (int br = 0; br < n_br; ++br) {
        const int q0         = br * kGqaPrefillNvfp4Br;
        if (q0 >= tokens) { continue; }
        const int tile_rows  = std::min(kGqaPrefillNvfp4Br, tokens - q0);
        const int key_blocks = std::min(kGqaPrefillNvfp4RankTiles,
                                        (base_pos + q0 + tile_rows - 1) / kGqaPrefillNvfp4Bc + 1);
        if (key_blocks <= 0) { continue; }
        for (int h = 0; h < q_heads; ++h) {
            const int n = std::min(std::max(counts[static_cast<std::size_t>(h * n_br + br)], 0),
                                   key_blocks);
            const int pct = (n * 100) / key_blocks;
            keep_sum += pct;
            keep_min = std::min(keep_min, pct);
            keep_max = std::max(keep_max, pct);
            if (n == key_blocks) { ++full_n; }
            ++denom_n;
        }
    }

    const int slot = q_heads * n_br - n_br; // head 0, last br with work
    int last_br    = 0;
    for (int br = n_br - 1; br >= 0; --br) {
        if (br * kGqaPrefillNvfp4Br < tokens) {
            last_br = br;
            break;
        }
    }
    const int inspect = last_br; // head 0
    const int q0      = inspect * kGqaPrefillNvfp4Br;
    const int tile_rows =
        std::min(kGqaPrefillNvfp4Br, std::max(0, tokens - q0));
    const int key_blocks = std::min(kGqaPrefillNvfp4RankTiles,
                                    (base_pos + q0 + std::max(tile_rows, 1) - 1) / kGqaPrefillNvfp4Bc + 1);
    const int nkeep      = (inspect < n_br) ? std::min(std::max(counts[static_cast<std::size_t>(inspect)], 0),
                                                  key_blocks)
                                           : 0;
    std::vector<std::uint16_t> keep(static_cast<std::size_t>(std::max(nkeep, 1)));
    if (nkeep > 0) {
        CUDA_CHECK(cudaMemcpy(keep.data(), scratch.keep + static_cast<std::int64_t>(inspect) * n_kb_cap,
                              static_cast<std::size_t>(nkeep) * sizeof(std::uint16_t),
                              cudaMemcpyDeviceToHost));
        bool ident = nkeep == key_blocks;
        for (int i = 0; ident && i < nkeep; ++i) {
            ident = keep[static_cast<std::size_t>(i)] == static_cast<std::uint16_t>(i);
        }
        ident_n = ident ? 1 : 0;
    }

    float top1 = 0.0f;
    const int br_last = last_br;
    const int kb_lim = std::min(n_kb_cap, std::max(1, max_abs / kGqaPrefillNvfp4Bc + 1));
    if (!identity && kb_lim > 0 && scratch.mass != nullptr) {
        std::vector<float> mass(static_cast<std::size_t>(kb_lim));
        const int mass_off = (0 * scratch.n_br + br_last) * n_kb_cap;
        CUDA_CHECK(cudaMemcpy(mass.data(), scratch.mass + mass_off,
                              mass.size() * sizeof(float), cudaMemcpyDeviceToHost));
        float z = 0.0f;
        float m = 0.0f;
        for (float x : mass) {
            z += x;
            m = std::max(m, x);
        }
        top1 = z > 0.0f ? m / z : 0.0f;
    }

    const double keep_mean = denom_n > 0 ? keep_sum / static_cast<double>(denom_n) : 0.0;
    const double full_pct  = denom_n > 0 ? 100.0 * full_n / static_cast<double>(denom_n) : 0.0;
    std::fprintf(stderr,
                 "[xattn] pos=%d T=%d kb=%d ident=%d keep%% mean/min/max=%.1f/%d/%d full_slots=%.0f%% "
                 "h0_last %d/%d%s top1=%.3f rank=%.1fus mma=%.1fus\n",
                 base_pos, tokens, kb_lim, identity ? 1 : 0, keep_mean, keep_min == 1000 ? 0 : keep_min,
                 keep_max, full_pct, nkeep, key_blocks, ident_n ? " consecutive" : "", top1, rank_us,
                 mma_us);
    (void)slot;
}

} // namespace

template <typename Geometry, typename CacheView, typename Metadata>
void gqa_sparse_prefill_attention_launch(const Tensor& q, const Tensor& positions, float scale,
                                          const CacheView& cache, Metadata metadata, Tensor& out,
                                          cudaStream_t stream, float keep_frac, float xattn_tau,
                                          std::int32_t xattn_min_len, GqaS3PrefillDump* dump,
                                          void* xattn_scratch, GqaExecutionEnvelope envelope) {
    const Tensor& cache_k       = cache.k_pages;
    const Tensor& cache_v       = cache.v_pages;
    const Tensor& cache_k_scale = cache.k_scale_pages;
    const Tensor& cache_v_scale = cache.v_scale_pages;
    const auto tokens           = static_cast<std::int32_t>(q.ne[2]);

    static const cudaError_t attr = cudaFuncSetAttribute(
        gqa_attention_prefill_nvfp4_sparse_kernel<Geometry, Metadata>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kGqaPrefillNvfp4SmemBytes);
    CUDA_CHECK(attr);

    const std::uint16_t* xattn_keep = nullptr;
    const int* xattn_count          = nullptr;
    int xattn_keep_stride           = 0;
    int xattn_n_br                  = 0;

    const bool want_xattn = xattn_tau > 0.0f && xattn_tau < 1.0f && tokens > 0;
    const bool log_keep   = want_xattn && xattn_keep_log_enabled();
    GqaXattnScratchView scratch{};
    cudaEvent_t ev_rank0{}, ev_rank1{}, ev_mma{};
    if (log_keep) {
        CUDA_CHECK(cudaEventCreate(&ev_rank0));
        CUDA_CHECK(cudaEventCreate(&ev_rank1));
        CUDA_CHECK(cudaEventCreate(&ev_mma));
        CUDA_CHECK(cudaEventRecord(ev_rank0, stream));
    }
    if (want_xattn) {
        const int q_heads  = Geometry::QHeads;
        const int kv_heads = Geometry::KVHeads;
        const int n_br     = div_up(tokens, kGqaPrefillNvfp4Br);
        const int n_kb =
            gqa_xattn_n_kb(gqa_xattn_logical_pages(cache), envelope.max_visible_keys);
        const int n_split  = std::max(1, std::min(kXAttnSplits, n_kb));
        const int score_smem = gqa_xattn_score_smem_bytes();
        static const cudaError_t score_attr = cudaFuncSetAttribute(
            gqa_xattn_score_kernel<Geometry, Metadata>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            std::max(score_smem, kGqaPrefillNvfp4SmemBytes));
        CUDA_CHECK(score_attr);
        static const cudaError_t finish_attr = cudaFuncSetAttribute(
            gqa_xattn_finish_kernel<Geometry, Metadata>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, kXAttnFinishSmemBytes);
        CUDA_CHECK(finish_attr);

        const std::size_t bytes = gqa_xattn_scratch_bytes(q_heads, kv_heads, n_br, n_kb);
        void* scratch_mem = xattn_scratch != nullptr ? xattn_scratch : xattn_ensure_scratch(bytes);
        scratch = gqa_xattn_bind_scratch(scratch_mem, q_heads, kv_heads, n_br, n_kb);

        const dim3 pack_grid(static_cast<unsigned>(kv_heads), static_cast<unsigned>(n_kb), 1u);
        gqa_xattn_pack_kernel<Geometry, Metadata>
            <<<pack_grid, kXAttnScoreThreads, 0, stream>>>(
                static_cast<const std::uint8_t*>(cache_k.data),
                static_cast<const std::uint8_t*>(cache_k_scale.data), metadata,
                static_cast<const std::int32_t*>(positions.data), tokens, xattn_min_len, n_kb,
                scratch.n_j, scratch.packed);
        CUDA_CHECK(cudaGetLastError());

        const dim3 score_grid(static_cast<unsigned>(q_heads), static_cast<unsigned>(n_br),
                              static_cast<unsigned>(n_split));
        gqa_xattn_score_kernel<Geometry, Metadata>
            <<<score_grid, kXAttnScoreThreads, static_cast<std::size_t>(score_smem), stream>>>(
                static_cast<const __nv_bfloat16*>(q.data), scratch.packed, metadata,
                static_cast<const std::int32_t*>(positions.data), scale, tokens, xattn_min_len,
                n_br, scratch.n_j, n_split, scratch.logits);
        CUDA_CHECK(cudaGetLastError());

        const dim3 mass_grid(static_cast<unsigned>(q_heads), static_cast<unsigned>(n_br), 1u);
        gqa_xattn_softmax_mass_kernel<Geometry, Metadata>
            <<<mass_grid, kGqaPrefillNvfp4Threads, 0, stream>>>(
                metadata, static_cast<const std::int32_t*>(positions.data), tokens, xattn_min_len,
                n_br, scratch.n_j, n_kb, scratch.logits, scratch.mass);
        CUDA_CHECK(cudaGetLastError());

        const dim3 finish_grid(static_cast<unsigned>(q_heads), static_cast<unsigned>(n_br), 1u);
        gqa_xattn_finish_kernel<Geometry, Metadata>
            <<<finish_grid, kGqaPrefillNvfp4Threads, kXAttnFinishSmemBytes, stream>>>(
                metadata, static_cast<const std::int32_t*>(positions.data), tokens, xattn_tau,
                xattn_min_len, n_br, n_kb, scratch.mass, scratch.keep, scratch.count);
        CUDA_CHECK(cudaGetLastError());

        xattn_keep        = scratch.keep;
        xattn_count       = scratch.count;
        xattn_keep_stride = scratch.keep_stride;
        xattn_n_br        = n_br;
    }
    if (log_keep) { CUDA_CHECK(cudaEventRecord(ev_rank1, stream)); }

    const dim3 attention_grid(static_cast<unsigned>(div_up(tokens, kGqaPrefillNvfp4Br)),
                              static_cast<unsigned>(Geometry::QHeads), 1u);
    gqa_attention_prefill_nvfp4_sparse_kernel<Geometry, Metadata>
        <<<attention_grid, kGqaPrefillNvfp4Threads, kGqaPrefillNvfp4SmemBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data),
            static_cast<const std::uint8_t*>(cache_k.data),
            static_cast<const std::uint8_t*>(cache_v.data),
            static_cast<const std::uint8_t*>(cache_k_scale.data),
            static_cast<const std::uint8_t*>(cache_v_scale.data),
            static_cast<const float*>(cache.k_mean_pages.data), metadata,
            static_cast<const std::int32_t*>(positions.data), scale,
            static_cast<__nv_bfloat16*>(out.data), tokens, keep_frac, xattn_tau, xattn_min_len,
            dump, xattn_keep, xattn_count, xattn_keep_stride, xattn_n_br);
    CUDA_CHECK(cudaGetLastError());
    if (log_keep) {
        CUDA_CHECK(cudaEventRecord(ev_mma, stream));
        CUDA_CHECK(cudaEventSynchronize(ev_mma));
        float rank_ms = 0.0f;
        float mma_ms  = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&rank_ms, ev_rank0, ev_rank1));
        CUDA_CHECK(cudaEventElapsedTime(&mma_ms, ev_rank1, ev_mma));
        xattn_log_keep(scratch, positions, tokens, xattn_min_len, rank_ms * 1000.0f,
                       mma_ms * 1000.0f);
        CUDA_CHECK(cudaEventDestroy(ev_rank0));
        CUDA_CHECK(cudaEventDestroy(ev_rank1));
        CUDA_CHECK(cudaEventDestroy(ev_mma));
    }
}

#define NINFER_SPARSE_PREFILL_INSTANTIATE(Geom, View, Meta)                                        \
    template void gqa_sparse_prefill_attention_launch<Geom, View, Meta>(                           \
        const Tensor&, const Tensor&, float, const View&, Meta, Tensor&, cudaStream_t, float,      \
        float, std::int32_t, GqaS3PrefillDump*, void*, GqaExecutionEnvelope)

NINFER_SPARSE_PREFILL_INSTANTIATE(Gqa27Geometry, PagedKVLayerView, GqaPrefillDirectMetadata);
NINFER_SPARSE_PREFILL_INSTANTIATE(Gqa35Geometry, PagedKVLayerView, GqaPrefillDirectMetadata);
NINFER_SPARSE_PREFILL_INSTANTIATE(Gqa27Geometry, PagedKVBatchLayerView,
                                  GqaPrefillBatchMetadata<false>);
NINFER_SPARSE_PREFILL_INSTANTIATE(Gqa27Geometry, PagedKVBatchLayerView,
                                  GqaPrefillBatchMetadata<true>);
NINFER_SPARSE_PREFILL_INSTANTIATE(Gqa35Geometry, PagedKVBatchLayerView,
                                  GqaPrefillBatchMetadata<false>);
NINFER_SPARSE_PREFILL_INSTANTIATE(Gqa35Geometry, PagedKVBatchLayerView,
                                  GqaPrefillBatchMetadata<true>);

#undef NINFER_SPARSE_PREFILL_INSTANTIATE

} // namespace ninfer::ops::detail
