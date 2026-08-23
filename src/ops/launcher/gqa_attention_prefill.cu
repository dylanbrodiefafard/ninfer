// ninfer::ops - gqa_attention prompt-scale launcher: fill k/v at device
// positions then launch causal attention over absolute cached history.
#include "ops/launcher/gqa_attention.h"

#include "ops/common/math.h"
#include "ops/kernel/gqa_attention_nvfp4s3_tma.cuh"
#include "ops/kernel/gqa_attention_prefill_bf16.cuh"
#include "ops/kernel/gqa_attention_prefill_i8.cuh"
#include "ops/kernel/gqa_attention_prefill_nvfp4.cuh"
#include "ops/kernel/gqa_attention_prefill_nvfp4s3.cuh"
#include "core/device.h" // CUDA_CHECK

#include <chrono>
#include <cstdint>
#include <cstdio> // NINFER_S3_TMA_TRACE, NINFER_S3_TMA_WATCH
#include <cstdlib> // NINFER_S3_TMA, NINFER_TMA_STAGES
#include <cuda_runtime.h>
#include <mutex>
#include <thread>
#include <unordered_map>


namespace ninfer::ops::detail {
namespace {

// NINFER_S3_TMA=1: route the sage (nvfp4s3) prompt attention through the TMA + mbarrier
// pipeline kernel (gqa_attention_prefill_nvfp4s3_tma) instead of the cp.async baseline.
// The TMA kernel carries the sparge tile-skip proxy arrays stripped (inert at keep_frac<1
// would be wrong), so it only engages for exact attention (keep_frac == 1.0); any other
// keep_frac falls through to the cp.async kernel, which owns the tile-skip path.
// NINFER_TMA_STAGES selects the pipeline depth (2 or 3; default 2).
// Read once per process, mirroring the NINFER_S3_STRICT_PV precedent in the decode launcher.
// Stuck-probe watch: a mapped-host 32-bit word the TMA kernel's spin loops write (only
// when the launcher hands a non-null pointer), plus a host watchdog thread that decodes
// and logs the live spin location while the kernel is still alive. Enabled by
// NINFER_S3_TMA_WATCH=1 (zero cost when unset: the kernel gets a null probe pointer).
struct GqaS3TmaStuckWatch {
    std::uint32_t* host = nullptr;   // mapped host view
    std::uint32_t* device = nullptr; // device view (kernel argument)
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
        out.host = static_cast<std::uint32_t*>(p);
        out.device = static_cast<std::uint32_t*>(d);
        *out.host = 0x7FFFFFFFu;  // sentinel: kernel has not probed yet
        std::thread([watch = &out] {
            using namespace std::chrono;
            const steady_clock::time_point start = steady_clock::now();
            std::uint32_t last = *watch->host;
            steady_clock::time_point last_change = start;
            auto ms_since = [&start](const steady_clock::time_point& t) {
                return static_cast<long long>(duration_cast<milliseconds>(t - start).count());
            };
            for (;;) {
                std::this_thread::sleep_for(milliseconds(250));
                const std::uint32_t v = *watch->host;
                const steady_clock::time_point now = steady_clock::now();
                if (v != last) {
                    last = v;
                    last_change = now;
                    if (v != 0x7FFFFFFFu) {
                        const int ki      = v & 0xFFFFF;
                        const int head    = (v >> 20) & 0x1F;
                        const int qblock  = (v >> 25) & 0x1F;
                        const bool empty  = (v >> 30) & 1u;
                        std::fprintf(stderr, "[s3-tma-watch] spin: %s ki=%d head=%d qblock=%d (t=%lld ms)\n",
                                     empty ? "empty-wait" : "full-wait", ki, head, qblock,
                                     ms_since(now));
                        std::fflush(stderr);
                    }
                } else if (v != 0x7FFFFFFFu && now - last_change > seconds(2)) {
                    last_change = now;
                    std::fprintf(stderr, "[s3-tma-watch] still spinning, last value 0x%08x (t=%lld ms)\n",
                                 v, ms_since(now));
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
        const int v = e ? std::strtol(e, nullptr, 10) : 2;
        return v == 3 ? 3 : 2;
    }();
    return stages;
}

// Host-side TMA descriptor cache. The descriptor encodes the (stable) paged KV plane
// pointers once per plane and is staged in device memory: SM120 + driver 580 traps on
// by-value CUtensorMap kernel parameters, so the kernel takes a device pointer. The
// buffers are small (4 x 128 B) and intentionally never freed: the prompt route is eager
// (never graph-captured), and per-launch (re)allocation would be pure overhead.
//
// nvcc compiles unmarked namespace-scope types in this .cu for the device pass, so the
// cache type itself only holds pointers (device-considerable); the std state is heap-
// allocated behind __host__-marked accessors, keeping it out of the device pass.
struct GqaS3TmaDescCache {
    std::mutex* mutex = nullptr;
    // Keyed on the K-plane pointer alone: a paged pool's plane is a unique allocation,
    // so the pointer identifies the (pool, page count) pair; std::pair has no std::hash.
    std::unordered_map<const void*, void*>* entries = nullptr;
};

__host__ inline GqaS3TmaDescCache& gqa_s3_tma_desc_cache() {
    static GqaS3TmaDescCache cache = [] {
        return GqaS3TmaDescCache{new std::mutex, new std::unordered_map<const void*, void*>};
    }();
    return cache;
}

// Builds (once) and returns the device-staged TMA descriptor for the sage cache's paged
// planes, or nullptr when the plane layout does not match the rank-4 descriptor assumption
// (the caller then falls back to the cp.async kernel). Works for both the single-sequence
// and batched layer views (shared plane fields).
template <typename Geometry, typename CacheView>
__host__ const GqaNvfp4s3TmaDesc* gqa_s3_tma_descriptor(const CacheView& cache) {
    const Tensor& cache_k = cache.k_pages;
    const std::int64_t per_page = static_cast<std::int64_t>(kGqaNvfp4CodeWidth) *
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
    {
        std::lock_guard<std::mutex> lock(*cache_store.mutex);
        cache_store.entries->emplace(key, device);
    }
    return static_cast<const GqaNvfp4s3TmaDesc*>(device);
}

template <typename Geometry, typename CacheView, typename Metadata>
void gqa_attention_prompt_attention_launch_for(const Tensor& q, const Tensor& positions,
                                                float scale, const CacheView& cache,
                                                Metadata metadata, Tensor& out,
                                                cudaStream_t stream, float keep_frac = 1.0f,
                                                GqaS3PrefillDump* dump = nullptr,
                                                std::uint32_t* dbg_regs = nullptr,
                                                std::uint8_t* dbg_q = nullptr) {
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
    static const cudaError_t attr_nvfp4s3 = cudaFuncSetAttribute(
        gqa_attention_prefill_nvfp4s3_kernel<Geometry, Metadata>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kGqaPrefillNvfp4s3SmemBytes);
    CUDA_CHECK(attr_nvfp4s3);

    const auto tokens = static_cast<std::int32_t>(q.ne[2]);
    if (cache.dtype == DType::U8 && cache.sage_pv) {
        const Tensor& cache_k_scale = cache.k_scale_pages;
        const Tensor& cache_v_scale = cache.v_scale_pages;
        const Tensor& cache_k_mean  = cache.k_mean_pages;
        if (gqa_s3_tma_enabled() && keep_frac == 1.0f) {
            if (const GqaNvfp4s3TmaDesc* desc_dev =
                    gqa_s3_tma_descriptor<Geometry, CacheView>(cache)) {
                if (gqa_s3_tma_trace()) {
                    std::fprintf(stderr, "[s3-tma] desc=%p desc_cache=%d tokens=%d\n",
                                 (const void*)desc_dev, (int)gqa_s3_tma_desc_cache().entries->size(),
                                 (int)tokens);
                    std::fflush(stderr);
                }
                auto launch_tma = [&]<int Stages>() {
                    constexpr int kSmem = static_cast<int>(sizeof(GqaNvfp4s3TmaScratch<Stages>));
                    static const cudaError_t attr = cudaFuncSetAttribute(
                        gqa_attention_prefill_nvfp4s3_tma_kernel<Geometry, Metadata, Stages>,
                        cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem);
                    CUDA_CHECK(attr);
                    const dim3 attention_grid(
                        static_cast<unsigned>(div_up(tokens, kGqaPrefillNvfp4s3Br)),
                        static_cast<unsigned>(Geometry::QHeads), 1u);
                    GqaS3TmaStuckWatch& watch = gqa_s3_tma_stuck_watch();
                    gqa_attention_prefill_nvfp4s3_tma_kernel<Geometry, Metadata, Stages>
                        <<<attention_grid, kGqaPrefillNvfp4s3Threads, kSmem, stream>>>(
                            static_cast<const __nv_bfloat16*>(q.data),
                            static_cast<const std::uint8_t*>(cache_v_scale.data),
                            static_cast<const float*>(cache_k_mean.data), metadata,
                            static_cast<const std::int32_t*>(positions.data), scale,
                            static_cast<__nv_bfloat16*>(out.data), tokens, keep_frac, desc_dev,
                            dump, nullptr, dbg_regs, dbg_q,
                            watch.active() ? watch.device : nullptr);
                    CUDA_CHECK(cudaGetLastError());
                    if (gqa_s3_tma_trace()) {
                        std::fprintf(stderr, "[s3-tma] kernel enqueued (Stages=%d)\n", Stages);
                        std::fflush(stderr);
                    }
                };
                if (gqa_s3_tma_stages() == 3) { launch_tma.template operator()<3>(); }
                else { launch_tma.template operator()<2>(); }
                return;
            }
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
        return;
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
            // The fill kernel also runs a meansim unit per (page, kv_head, group) when a k_mean
            // plane is present, offset into [k_units, k_units+kmean_units); size the grid to cover it
             // so the V units (offset past the kmean range) are all launched.
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
            const int fill_grid =
                static_cast<int>(div_up(fill_units, static_cast<std::int64_t>(256)));
            gqa_attention_prefill_fill_nvfp4s3_kernel<Geometry, Metadata>
                <<<fill_grid, 256, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(k.data),
                    static_cast<const __nv_bfloat16*>(v.data),
                    static_cast<const std::int32_t*>(positions.data), metadata,
                    static_cast<std::uint8_t*>(cache_k.data),
                    static_cast<std::uint8_t*>(cache_v.data),
                    static_cast<std::uint8_t*>(cache_k_scale.data),
                    static_cast<std::uint8_t*>(cache_v_scale.data),
                    static_cast<float*>(cache.k_mean_pages.data), tokens);
            CUDA_CHECK(cudaGetLastError());
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
                                           cudaStream_t stream, float keep_frac,
                                           GqaS3PrefillDump* dump, std::uint32_t* dbg_regs,
                                           std::uint8_t* dbg_q) {
    const GqaPrefillDirectMetadata metadata{
        static_cast<const std::int32_t*>(cache.block_table.data)};
    if (q.ne[1] == Gqa27Geometry::QHeads) {
        gqa_attention_prompt_attention_launch_for<Gqa27Geometry>(q, positions, scale, cache,
                                                                  metadata, out, stream, keep_frac,
                                                                  dump, dbg_regs, dbg_q);
        return;
    }
    gqa_attention_prompt_attention_launch_for<Gqa35Geometry>(q, positions, scale, cache, metadata,
                                                              out, stream, keep_frac, dump, dbg_regs,
                                                              dbg_q);
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
                                 GqaS3PrefillDump* dump) {
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
            gqa_attention_prompt_attention_launch_for<Gqa27Geometry>(q, positions, scale, cache,
                                                                      metadata, out, stream,
                                                                      keep_frac, dump);
            return;
        }
        gqa_kv_append_launch_for<Gqa35Geometry>(k, v, positions, cache, metadata, stream);
        gqa_attention_prompt_attention_launch_for<Gqa35Geometry>(q, positions, scale, cache,
                                                                  metadata, out, stream,
                                                                  keep_frac, dump);
    };
    if (valid_columns.data == nullptr) {
        launch.template operator()<false>();
    } else {
        launch.template operator()<true>();
    }
}

} // namespace ninfer::ops::detail
