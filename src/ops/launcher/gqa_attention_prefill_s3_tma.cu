// Sage3 TMA prefill path, compiled separately from the cp.async occupancy-2
// kernel so editing SmoothQ/K-centering does not re-cicc this TU.
#include "ops/launcher/gqa_attention_s3_launch.h"

#include "ops/common/math.h"
#include "ops/kernel/gqa_attention_nvfp4s3_tma.cuh"
#include "core/device.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <mutex>
#include <thread>
#include <unordered_map>

namespace ninfer::ops::detail {

namespace {

struct GqaS3TmaStuckWatch {
    std::uint32_t* host   = nullptr;
    std::uint32_t* device = nullptr;
    bool active() const { return device != nullptr; }
};

__host__ GqaS3TmaStuckWatch& gqa_s3_tma_stuck_watch() {
    static GqaS3TmaStuckWatch w = [] {
        GqaS3TmaStuckWatch out;
        if (std::getenv("NINFER_S3_TMA_WATCH") == nullptr) { return out; }
        void* p = nullptr;
        if (cudaHostAlloc(&p, sizeof(std::uint32_t), cudaHostAllocMapped) != cudaSuccess ||
            p == nullptr) {
            std::fprintf(stderr, "[s3-tma-watch] mapped host alloc failed; watch disabled\n");
            return out;
        }
        void* d = nullptr;
        if (cudaHostGetDevicePointer(&d, p, 0) != cudaSuccess) {
            cudaFreeHost(p);
            std::fprintf(stderr, "[s3-tma-watch] mapped device pointer failed; watch disabled\n");
            return out;
        }
        out.host   = static_cast<std::uint32_t*>(p);
        out.device = static_cast<std::uint32_t*>(d);
        *out.host  = 0x7FFFFFFFu;
        std::thread([watch = &out] {
            using namespace std::chrono;
            const steady_clock::time_point start = steady_clock::now();
            std::uint32_t last                   = *watch->host;
            steady_clock::time_point last_change = start;
            auto ms_since = [&start](const steady_clock::time_point& t) {
                return static_cast<long long>(duration_cast<milliseconds>(t - start).count());
            };
            for (;;) {
                std::this_thread::sleep_for(milliseconds(250));
                const std::uint32_t v                = *watch->host;
                const steady_clock::time_point now   = steady_clock::now();
                if (v != last) {
                    last        = v;
                    last_change = now;
                    if (v != 0x7FFFFFFFu) {
                        const int ki     = v & 0xFFFFF;
                        const int head   = (v >> 20) & 0x1F;
                        const int qblock = (v >> 25) & 0x1F;
                        const bool empty = (v >> 30) & 1u;
                        std::fprintf(stderr,
                                     "[s3-tma-watch] spin: %s ki=%d head=%d qblock=%d (t=%lld ms)\n",
                                     empty ? "empty-wait" : "full-wait", ki, head, qblock,
                                     ms_since(now));
                        std::fflush(stderr);
                    }
                } else if (v != 0x7FFFFFFFu && now - last_change > seconds(2)) {
                    last_change = now;
                    std::fprintf(stderr,
                                 "[s3-tma-watch] still spinning, last value 0x%08x (t=%lld ms)\n", v,
                                 ms_since(now));
                    std::fflush(stderr);
                }
            }
        }).detach();
        return out;
    }();
    return w;
}

__host__ inline bool gqa_s3_tma_trace() {
    static const bool on = std::getenv("NINFER_S3_TMA_TRACE") != nullptr;
    return on;
}

inline bool gqa_s3_tma_enabled() {
    static const bool enabled = [] {
        const char* e = std::getenv("NINFER_S3_TMA");
        return e != nullptr && e[0] == '1';
    }();
    return enabled;
}

inline int gqa_s3_tma_stages() {
    static const int stages = [] {
        const char* e = std::getenv("NINFER_TMA_STAGES");
        const int v   = e ? std::strtol(e, nullptr, 10) : 2;
        return v == 3 ? 3 : 2;
    }();
    return stages;
}

struct GqaS3TmaDescCache {
    std::mutex* mutex                                     = nullptr;
    std::unordered_map<const void*, void*>* entries       = nullptr;
};

__host__ inline GqaS3TmaDescCache& gqa_s3_tma_desc_cache() {
    static GqaS3TmaDescCache cache = [] {
        return GqaS3TmaDescCache{new std::mutex, new std::unordered_map<const void*, void*>};
    }();
    return cache;
}

template <typename Geometry, typename CacheView>
__host__ const GqaNvfp4s3TmaDesc* gqa_s3_tma_descriptor(const CacheView& cache) {
    const Tensor& cache_k           = cache.k_pages;
    const std::int64_t per_page     = static_cast<std::int64_t>(kGqaNvfp4CodeWidth) *
                                  kPagedKVPageSize * Geometry::KVHeads;
    std::int64_t numel = 1;
    for (int i = 0; i < 4; ++i) { numel *= cache_k.ne[i]; }
    if (numel <= 0 || per_page <= 0 || numel % per_page != 0) { return nullptr; }
    const std::int32_t pages = static_cast<std::int32_t>(numel / per_page);
    if (pages <= 0) { return nullptr; }

    GqaS3TmaDescCache& cache_store = gqa_s3_tma_desc_cache();
    std::lock_guard<std::mutex> lock(*cache_store.mutex);
    const void* key = cache_k.data;
    {
        const auto found = cache_store.entries->find(key);
        if (found != cache_store.entries->end()) {
            return static_cast<const GqaNvfp4s3TmaDesc*>(found->second);
        }
    }
    GqaNvfp4s3TmaDesc desc;
    if (!GqaNvfp4s3TmaDesc::build<Geometry, kGqaPrefillNvfp4s3Bc>(
            static_cast<const std::uint8_t*>(cache_k.data),
            static_cast<const std::uint8_t*>(cache.v_pages.data),
            static_cast<const std::uint8_t*>(cache.k_scale_pages.data), pages, &desc)) {
        return nullptr;
    }
    void* device = nullptr;
    if (cudaMalloc(&device, sizeof(GqaNvfp4s3TmaDesc)) != cudaSuccess ||
        cudaMemcpy(device, &desc, sizeof(desc), cudaMemcpyHostToDevice) != cudaSuccess) {
        if (device != nullptr) { cudaFree(device); }
        return nullptr;
    }
    cache_store.entries->emplace(key, device);
    return static_cast<const GqaNvfp4s3TmaDesc*>(device);
}

} // namespace

template <typename Geometry, typename CacheView, typename Metadata>
bool gqa_s3_prefill_tma_try_launch(const Tensor& q, const Tensor& positions, float scale,
                                   const CacheView& cache, Metadata metadata, Tensor& out,
                                   cudaStream_t stream, float keep_frac, GqaS3PrefillDump* dump,
                                   std::uint32_t* dbg_regs, std::uint8_t* dbg_q) {
    if (!gqa_s3_tma_enabled() || keep_frac != 1.0f) { return false; }
    const GqaNvfp4s3TmaDesc* desc_dev = gqa_s3_tma_descriptor<Geometry, CacheView>(cache);
    if (desc_dev == nullptr) { return false; }

    const Tensor& cache_v_scale = cache.v_scale_pages;
    const Tensor& cache_k_mean  = cache.k_mean_pages;
    const auto tokens           = static_cast<std::int32_t>(q.ne[2]);
    if (gqa_s3_tma_trace()) {
        std::fprintf(stderr, "[s3-tma] desc=%p desc_cache=%d tokens=%d\n", (const void*)desc_dev,
                     (int)gqa_s3_tma_desc_cache().entries->size(), (int)tokens);
        std::fflush(stderr);
    }
    auto launch_tma = [&]<int Stages>() {
        constexpr int kSmem = static_cast<int>(sizeof(GqaNvfp4s3TmaScratch<Stages>));
        static const cudaError_t attr = cudaFuncSetAttribute(
            gqa_attention_prefill_nvfp4s3_tma_kernel<Geometry, Metadata, Stages>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem);
        CUDA_CHECK(attr);
        const dim3 attention_grid(static_cast<unsigned>(div_up(tokens, kGqaPrefillNvfp4s3Br)),
                                  static_cast<unsigned>(Geometry::QHeads), 1u);
        GqaS3TmaStuckWatch& watch = gqa_s3_tma_stuck_watch();
        gqa_attention_prefill_nvfp4s3_tma_kernel<Geometry, Metadata, Stages>
            <<<attention_grid, kGqaPrefillNvfp4s3Threads, kSmem, stream>>>(
                static_cast<const __nv_bfloat16*>(q.data),
                static_cast<const std::uint8_t*>(cache_v_scale.data),
                static_cast<const float*>(cache_k_mean.data), metadata,
                static_cast<const std::int32_t*>(positions.data), scale,
                static_cast<__nv_bfloat16*>(out.data), tokens, keep_frac, desc_dev, dump, nullptr,
                dbg_regs, dbg_q, watch.active() ? watch.device : nullptr);
        CUDA_CHECK(cudaGetLastError());
        if (gqa_s3_tma_trace()) {
            std::fprintf(stderr, "[s3-tma] kernel enqueued (Stages=%d)\n", Stages);
            std::fflush(stderr);
        }
    };
    if (gqa_s3_tma_stages() == 3) { launch_tma.template operator()<3>(); }
    else { launch_tma.template operator()<2>(); }
    return true;
}

#define NINFER_S3_TMA_INSTANTIATE(Geom, View, Meta)                                                \
    template bool gqa_s3_prefill_tma_try_launch<Geom, View, Meta>(                                 \
        const Tensor&, const Tensor&, float, const View&, Meta, Tensor&, cudaStream_t, float,      \
        GqaS3PrefillDump*, std::uint32_t*, std::uint8_t*)

NINFER_S3_TMA_INSTANTIATE(Gqa27Geometry, PagedKVLayerView, GqaPrefillDirectMetadata);
NINFER_S3_TMA_INSTANTIATE(Gqa35Geometry, PagedKVLayerView, GqaPrefillDirectMetadata);
NINFER_S3_TMA_INSTANTIATE(Gqa27Geometry, PagedKVBatchLayerView, GqaPrefillBatchMetadata<false>);
NINFER_S3_TMA_INSTANTIATE(Gqa27Geometry, PagedKVBatchLayerView, GqaPrefillBatchMetadata<true>);
NINFER_S3_TMA_INSTANTIATE(Gqa35Geometry, PagedKVBatchLayerView, GqaPrefillBatchMetadata<false>);
NINFER_S3_TMA_INSTANTIATE(Gqa35Geometry, PagedKVBatchLayerView, GqaPrefillBatchMetadata<true>);

#undef NINFER_S3_TMA_INSTANTIATE

} // namespace ninfer::ops::detail
