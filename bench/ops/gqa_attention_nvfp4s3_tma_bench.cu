// ninfer_gqa_attention_nvfp4s3_tma_bench.cu
//
// TMA + mbarrier pipeline prefill kernel A/B vs the cp.async baseline, on the
// 5090. The TMA kernel (gqa_attention_prefill_nvfp4s3_tma_kernel) stages the K/V
// code+scale tiles with a dedicated producer running `Stages` ahead (cp.async.bulk
// 4-D tensor loads + a 1-D bulk v_scale copy), all 512 threads are the consumer,
// and the K/V smem is a `Stages`-deep circular mbarrier pipeline. The cp.async
// baseline (gqa_attention_prefill_nvfp4s3_kernel) is the production prefill kernel.
// Both run on identical cache data; only the attention kernel is timed (the fill
// is a one-shot setup).
//
// Usage: ninfer_gqa_attention_nvfp4s3_tma_bench
// Env: NINFER_TMA_STAGES (default 2). S3 is exact-only; tile-skip is not on this kernel.

#include "ops/launcher/gqa_attention.h"
#include "core/device.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"
#include "ninfer_bench_common.h"

#include "ops/kernel/gqa_attention_geometry.cuh"
#include "ops/kernel/gqa_attention_kv_nvfp4.cuh"
#include "ops/kernel/gqa_attention_nvfp4s3_tma.cuh"

#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <limits>
#include <vector>

using namespace ninfer;
using namespace ninfer::ops;
using namespace ninfer::bench;

namespace {
using Geom = Gqa27Geometry;
constexpr int kQHeads    = Geom::QHeads;         // 24
constexpr int kKVHeads   = Geom::KVHeads;         // 4
constexpr int kHeadDim   = kGqaNvfp4HeadDim;      // 256
constexpr int kCodeW     = kGqaNvfp4CodeWidth;    // 128 bytes/key
constexpr int kGroups    = kGqaNvfp4Groups;       // 16 e4m3 scales/key
constexpr int kPageSize  = kPagedKVPageSize;      // 64
constexpr int kQueryWin  = 4096;                  // = production NINFER_PREFILL_CHUNK
constexpr float kScale   = 0.0625f;               // 1/sqrt(head_dim)

// Owns the U8 cache planes + the PagedKVLayerView the launchers consume.
struct Nvfp4s3Cache {
    DeviceBuffer k_pages, v_pages, k_scale, v_scale, k_mean, block_table;
    PagedKVLayerView view;
    int logical_pages  = 0;
    int physical_pages = 0;
    int capacity       = 0;
};

Nvfp4s3Cache make_cache(int context) {
    const int cap = ((context + 127) / 128) * 128;  // pad to whole 128-key (2-page) units
    const int logical_pages  = cap / kPageSize;
    const int physical_pages = (cap + kPageSize - 1) / kPageSize;  // identity page map
    const std::size_t code_bytes  = static_cast<std::size_t>(kCodeW) * kPageSize * kKVHeads * physical_pages;
    const std::size_t scale_bytes = static_cast<std::size_t>(kGroups) * kPageSize * kKVHeads * physical_pages;
    const std::size_t table_bytes = static_cast<std::size_t>(logical_pages) * sizeof(std::int32_t);

    Nvfp4s3Cache cache;
    cache.logical_pages  = logical_pages;
    cache.physical_pages = physical_pages;
    cache.capacity       = cap;
    cache.k_pages     = DeviceBuffer(code_bytes);
    cache.v_pages     = DeviceBuffer(code_bytes);
    cache.k_scale     = DeviceBuffer(scale_bytes);
    cache.v_scale     = DeviceBuffer(scale_bytes);
    cache.k_mean      = DeviceBuffer(static_cast<std::size_t>(4) * kPageSize * kKVHeads * physical_pages * sizeof(float));
    cache.block_table = DeviceBuffer(table_bytes);
    cache.k_pages.fill(0);
    cache.v_pages.fill(0);
    cache.k_scale.fill(0);
    cache.v_scale.fill(0);
    cache.k_mean.fill(0);

    std::vector<std::int32_t> h_table(logical_pages);
    for (int p = 0; p < logical_pages; ++p) h_table[p] = p;
    cache.block_table.copy_from_host(h_table.data(), table_bytes);

    cache.view = PagedKVLayerView{};
    cache.view.k_pages       = Tensor(cache.k_pages.p, DType::U8, {kCodeW, kPageSize, kKVHeads, physical_pages});
    cache.view.v_pages       = Tensor(cache.v_pages.p, DType::U8, {kCodeW, kPageSize, kKVHeads, physical_pages});
    cache.view.k_scale_pages = Tensor(cache.k_scale.p, DType::FP8_E4M3FN, {kGroups, kPageSize, kKVHeads, physical_pages});
    cache.view.v_scale_pages = Tensor(cache.v_scale.p, DType::FP8_E4M3FN, {kGroups, kPageSize, kKVHeads, physical_pages});
    cache.view.k_mean_pages  = Tensor(cache.k_mean.p, DType::FP32, {4, kPageSize, kKVHeads, physical_pages});
    cache.view.block_table   = Tensor(cache.block_table.p, DType::I32, {logical_pages});
    cache.view.head_dim      = kHeadDim;
    cache.view.num_kv_heads = kKVHeads;
    cache.view.dtype        = DType::U8;
    cache.view.quant_group  = kGroups;
    cache.view.sage_pv      = true;
    return cache;
}

namespace s3oracle {
constexpr std::int32_t kHeadDim    = 256;
constexpr std::int32_t kQHeads     = 24;
constexpr std::int32_t kKVHeads    = 4;
constexpr std::int32_t kNvfp4Group = 16;
constexpr std::int32_t kPageSize   = 64;
constexpr float kAttentionScale    = 0.0625f;
constexpr double kLog2E            = 1.4426950408889634;
constexpr double kAmp              = 2688.0;  // 448 * 6
constexpr double kSMax             = 448.0;   // e4m3 max finite

// kAttentionNvfp4s3Criterion (test_gqa_attention.cpp:68).
struct Criterion {
    double relative_l2                       = 9.0e-2;
    double gross_absolute                    = 8.0e-2;
    double gross_relative_to_max_reference   = 1.6e-1;
};

inline double decode_e2m1_word(std::uint8_t n) {
    static constexpr double mag[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
    return (n & 0x08u) == 0 ? mag[n & 0x07u] : -mag[n & 0x07u];
}
inline double decode_e4m3_word(std::uint8_t w) {
    __nv_fp8_e4m3 v;
    v.__x = w;
    return static_cast<double>(static_cast<float>(v));
}
inline std::uint8_t encode_e4m3_rne(float x) {
    return static_cast<std::uint8_t>(__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3));
}
inline std::uint8_t encode_e2m1_rne(float x) {
    return static_cast<std::uint8_t>(__nv_cvt_float_to_fp4(x, __NV_E2M1, cudaRoundNearest)) &
           0x0fu;
}

// Emulate the kernel's in-kernel NVFP4 Q quantization (per-16-d e4m3 scale + e2m1 codes).
// q layout: [d][q_head][token] (the bench Tensor layout).
inline void q_nvfp4_log(const std::vector<float>& q, std::int32_t q_head, std::int32_t token,
                        std::vector<double>& q_log) {
    q_log.assign(kHeadDim, 0.0);
    const auto qi = [&](std::int32_t d) {
        return static_cast<std::size_t>(d) +
               static_cast<std::size_t>(kHeadDim) *
                   (static_cast<std::size_t>(q_head) +
                    static_cast<std::size_t>(kQHeads) * static_cast<std::size_t>(token));
    };
    for (std::int32_t grp = 0; grp < kHeadDim / kNvfp4Group; ++grp) {
        float absmax = 0.0f;
        for (std::int32_t i = 0; i < kNvfp4Group; ++i) {
            const std::int32_t d = grp * kNvfp4Group + i;
            absmax = std::max(absmax, std::abs(q[qi(d)]));
        }
        const std::uint8_t sb = encode_e4m3_rne(absmax / 6.0f);
        const float sc        = static_cast<float>(decode_e4m3_word(sb));
        for (std::int32_t pair = 0; pair < kNvfp4Group / 2; ++pair) {
            const std::int32_t d0 = grp * kNvfp4Group + 2 * pair;
            const std::int32_t d1  = d0 + 1;
            float x = 0.0f;
            float y = 0.0f;
            if (sc != 0.0f) {
                x = q[qi(d0)] / sc;
                y = q[qi(d1)] / sc;
            }
            const float2 packed  = {x, y};
            const std::uint8_t code =
                static_cast<std::uint8_t>(
                    __nv_cvt_float2_to_fp4x2(packed, __NV_E2M1, cudaRoundNearest));
            q_log[static_cast<std::size_t>(d0)] =
                decode_e2m1_word(code & 0x0Fu) * static_cast<double>(sc);
            q_log[static_cast<std::size_t>(d1)] =
                decode_e2m1_word((code >> 4) & 0x0Fu) * static_cast<double>(sc);
        }
    }
}

// Exact S3 reference attention (host FP64). Output layout [d][head][token].
// GPU plane layouts (paged_kv_element_offset ground truth, head-major within page):
//   k/v codes: ((key/64)*4 + head) * 8192 + (key%64) * 128 + d/2
//   k scale:   ((key/64)*4 + head) * 1024 + (key%64) * 16 + d/16
//   v scale:   ((key/64)*4 + head) * 1024 + d*4 + (key%64)/16
inline std::vector<double> sage_ideal_attention(
    const std::vector<float>& q, const std::vector<std::int32_t>& positions,
    const std::uint8_t* k_codes, const std::uint8_t* v_codes, const std::uint8_t* k_scale,
    const std::uint8_t* v_scale) {
    auto kv_code = [&](std::int32_t head, std::int32_t key, std::int32_t d, bool is_k)
        -> std::uint8_t {
        const std::size_t off =
            static_cast<std::size_t>((key / kPageSize) * kKVHeads + head) * (kPageSize * 128u) +
            static_cast<std::size_t>(key % kPageSize) * 128u + static_cast<std::size_t>(d / 2);
        const std::uint8_t* plane = is_k ? k_codes : v_codes;
        const std::uint8_t byte    = plane[off];
        return (d & 1) == 0 ? (byte & 0x0fu) : (byte >> 4);
    };
    auto k_scale_of = [&](std::int32_t head, std::int32_t key, std::int32_t d) -> std::uint8_t {
        const std::size_t off =
            static_cast<std::size_t>((key / kPageSize) * kKVHeads + head) * (kPageSize * 16u) +
            static_cast<std::size_t>(key % kPageSize) * 16u + static_cast<std::size_t>(d / 16);
        return k_scale[off];
    };
    auto v_scale_of = [&](std::int32_t head, std::int32_t key, std::int32_t d) -> std::uint8_t {
        const std::size_t off =
            static_cast<std::size_t>((key / kPageSize) * kKVHeads + head) * (kPageSize * 16u) +
            static_cast<std::size_t>(d) * 4u + static_cast<std::size_t>(key % kPageSize) / 16;
        return v_scale[off];
    };

    const std::int32_t tokens = static_cast<std::int32_t>(positions.size());
    std::vector<double> output(
        static_cast<std::size_t>(kHeadDim) * kQHeads * static_cast<std::size_t>(tokens));
    std::vector<double> q_log;
    for (std::int32_t token = 0; token < tokens; ++token) {
        const std::int32_t visible = positions[static_cast<std::size_t>(token)] + 1;
        for (std::int32_t q_head = 0; q_head < kQHeads; ++q_head) {
            const std::int32_t kv_head = q_head / (kQHeads / kKVHeads);
            q_nvfp4_log(q, q_head, token, q_log);
            std::vector<double> p(visible, 0.0);
            double max_score = -std::numeric_limits<double>::infinity();
            for (std::int32_t position = 0; position < visible; ++position) {
                double dot = 0.0;
                for (std::int32_t d = 0; d < kHeadDim; ++d)
                    dot += q_log[static_cast<std::size_t>(d)] *
                           decode_e2m1_word(kv_code(kv_head, position, d, true)) *
                           decode_e4m3_word(k_scale_of(kv_head, position, d));
                const double score = dot * static_cast<double>(kAttentionScale);
                p[static_cast<std::size_t>(position)] = score;
                max_score = std::max(max_score, score);
            }
            double l_sum = 0.0;
            for (std::int32_t position = 0; position < visible; ++position) {
                p[static_cast<std::size_t>(position)] =
                    std::exp2(kLog2E * (p[static_cast<std::size_t>(position)] - max_score));
                l_sum += p[static_cast<std::size_t>(position)];
            }
            std::vector<double> value(kHeadDim, 0.0);
            for (std::int32_t d = 0; d < kHeadDim; ++d) {
                double acc = 0.0;
                for (std::int32_t kb = 0; kb < visible; kb += 16) {
                    const std::int32_t k1 = std::min(kb + 16, visible);
                    double block_max = 0.0;
                    for (std::int32_t k = kb; k < k1; ++k)
                        block_max = std::max(block_max, p[static_cast<std::size_t>(k)]);
                    const std::uint8_t sc =
                        block_max == 0.0
                            ? 0
                            : encode_e4m3_rne(
                                  static_cast<float>(std::min(kSMax, kSMax * block_max)));
                    const double s_dec = decode_e4m3_word(sc);
                    const double v_dec = decode_e4m3_word(v_scale_of(kv_head, kb, d));
                    double pv          = 0.0;
                    for (std::int32_t k = kb; k < k1; ++k) {
                        const double p_p =
                            s_dec == 0.0
                                ? 0.0
                                : std::min(6.0, kAmp * p[static_cast<std::size_t>(k)] / s_dec);
                        const std::uint8_t p_code = encode_e2m1_rne(static_cast<float>(p_p));
                        pv += decode_e2m1_word(p_code) *
                              decode_e2m1_word(kv_code(kv_head, k, d, false));
                    }
                    acc += pv * s_dec * v_dec;
                }
                value[static_cast<std::size_t>(d)] = acc;
            }
            for (std::int32_t d = 0; d < kHeadDim; ++d)
                output[static_cast<std::size_t>(d) +
                       static_cast<std::size_t>(kHeadDim) *
                           (static_cast<std::size_t>(q_head) +
                            static_cast<std::size_t>(kQHeads) * static_cast<std::size_t>(token))]
                    = l_sum > 0.0 ? value[static_cast<std::size_t>(d)] / (kAmp * l_sum) : 0.0;
        }
    }
    return output;
}
}  // namespace s3oracle

// FP64 S3 oracle gate: both kernels vs the host reference (see s3oracle above).

template <int Stages>
void launch_tma_once(const Nvfp4s3Cache& cache, DeviceBuffer& q_src, DeviceBuffer& pos_q,
                     int window, float keep_frac, const GqaNvfp4s3TmaDesc* desc_dev, void* out,
                     cudaStream_t stream, GqaS3PrefillDump* dump = nullptr,
                     std::uint8_t* dbg_stage = nullptr, std::uint32_t* dbg_regs = nullptr,
                     std::uint8_t* dbg_q = nullptr) {
    const GqaPrefillDirectMetadata metadata{
        static_cast<const std::int32_t*>(cache.block_table.p)};
    const dim3 grid(div_up(window, kGqaPrefillNvfp4s3Br), kQHeads, 1u);
    constexpr int kSmem = static_cast<int>(sizeof(GqaNvfp4s3TmaScratch<Stages>));
    gqa_attention_prefill_nvfp4s3_tma_kernel<Geom, GqaPrefillDirectMetadata, Stages>
        <<<grid, kGqaPrefillNvfp4s3Threads, kSmem, stream>>>(
            static_cast<const __nv_bfloat16*>(q_src.p),
            static_cast<const std::uint8_t*>(cache.v_scale.p),
            static_cast<const float*>(cache.k_mean.p), metadata,
            static_cast<const std::int32_t*>(pos_q.p), kScale,
            static_cast<__nv_bfloat16*>(out), window, keep_frac, desc_dev, dump, dbg_stage,
            dbg_regs, dbg_q, nullptr);
    CUDA_CHECK(cudaGetLastError());
}

// Device-side storage for a GqaS3PrefillDump (fields live in cudaMalloc buffers).
struct S3Dump {
    GqaS3PrefillDump d{};
    std::vector<void*> raw;
    void alloc(std::int32_t heads, std::int32_t max_tiles) {
        d.max_tiles = max_tiles;
        const std::size_t h = static_cast<std::size_t>(heads);
        const std::size_t t = static_cast<std::size_t>(max_tiles);
        const std::size_t Br = kGqaPrefillNvfp4s3Br;
        const std::size_t Bc = kGqaPrefillNvfp4s3Bc;
        const std::size_t Dd = kGqaPrefillHeadDim;
        auto mk = [&](std::size_t n) -> void* {
            void* p = nullptr;
            CUDA_CHECK(cudaMalloc(&p, n));
            raw.push_back(p);
            return p;
        };
        d.score    = static_cast<float*>(mk(h * t * Br * Bc * sizeof(float)));
        d.p_code   = static_cast<std::uint8_t*>(mk(h * t * Br * Bc));
        d.psf      = static_cast<std::uint8_t*>(mk(h * t * Br * 4));
        d.v_scale  = static_cast<std::uint8_t*>(mk(h * t * Dd * 4));
        d.v_t      = static_cast<std::uint8_t*>(mk(h * t * Dd * 32));
        d.m        = static_cast<float*>(mk(h * t * Br * sizeof(float)));
        d.l        = static_cast<float*>(mk(h * t * Br * sizeof(float)));
        d.acc      = static_cast<float*>(mk(h * t * Br * Dd * sizeof(float)));
        d.keep_list = static_cast<std::int32_t*>(mk(h * t * sizeof(std::int32_t)));
        d.tile_count = static_cast<std::int32_t*>(mk(h * sizeof(std::int32_t)));
    }
    ~S3Dump() {
        for (void* p : raw) cudaFree(p);
    }
};

// Max |a-b| over two bf16 buffers (per-block reduction + atomicMax).
__global__ void gqa_bf16_max_abs_diff(const __nv_bfloat16* a, const __nv_bfloat16* b, long n,
                                      float* out_max) {
    float m = 0.0f;
    const long stride = static_cast<long>(gridDim.x) * blockDim.x;
    for (long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
        m = fmaxf(m, fabsf(__bfloat162float(a[i]) - __bfloat162float(b[i])));
    }
    for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
    __shared__ float sm[32];
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;
    if (lane == 0) sm[wid] = m;
    __syncthreads();
    if (wid == 0) {
        m = (lane < static_cast<int>(blockDim.x >> 5)) ? sm[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
        if (lane == 0) atomicMax(reinterpret_cast<unsigned*>(out_max), __float_as_uint(m));
    }
}

// Bucketed diff map for the debug output: max |a-b| per (head, row/128, dim/32) bucket.
// a/b layout: [dim][head][row]; grid = (kQHeads, div_up(window,128), 8), block = 256.
__global__ void gqa_s3_bucket_diff(const __nv_bfloat16* a, const __nv_bfloat16* b, float* diffmap,
                                   int window) {
    const int head   = blockIdx.x;
    const int row0   = blockIdx.y * 128;
    const int d0     = static_cast<int>(blockIdx.z) * 32;
    float m = 0.0f;
    for (int t = static_cast<int>(threadIdx.x); t < 128 * 32; t += 256) {
        const int row = row0 + t / 32;
        if (row >= window) break;
        const std::size_t ia = static_cast<std::size_t>(d0 + t % 32) * kQHeads * window +
                               static_cast<std::size_t>(head) * window + row;
        m = fmaxf(m, fabsf(__bfloat162float(a[ia]) - __bfloat162float(b[ia])));
    }
    for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
    __shared__ float sm[8];
    const int lane = threadIdx.x & 31;
    const int wid  = threadIdx.x >> 5;
    if (lane == 0) sm[wid] = m;
    __syncthreads();
    if (wid == 0) {
        m = (lane < 8) ? sm[lane] : 0.0f;
        for (int off = 4; off > 0; off >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
        if (lane == 0) {
            float* bucket =
                diffmap + (static_cast<std::size_t>(head) * gridDim.y + blockIdx.y) * 8 + blockIdx.z;
            atomicMax(reinterpret_cast<unsigned*>(bucket), __float_as_uint(m));
        }
    }
}

template <int Stages>
ColdTiming run_tma(Nvfp4s3Cache& cache, DeviceBuffer& q_src, DeviceBuffer& out_buf,
                    DeviceBuffer& pos_q, int window, float keep_frac, cudaStream_t stream) {
    // Host-side rank-4 TMA descriptor for the paged KV planes.
    GqaNvfp4s3TmaDesc desc;
    if (!GqaNvfp4s3TmaDesc::build<Geom, kGqaPrefillNvfp4s3Bc>(
            static_cast<const std::uint8_t*>(cache.k_pages.p),
            static_cast<const std::uint8_t*>(cache.v_pages.p),
            static_cast<const std::uint8_t*>(cache.k_scale.p), cache.physical_pages, &desc)) {
        std::fprintf(stderr, "TMA descriptor build failed\n");
        std::exit(1);
    }
    constexpr int kSmem = static_cast<int>(sizeof(GqaNvfp4s3TmaScratch<Stages>));
    const cudaError_t attr = cudaFuncSetAttribute(
        gqa_attention_prefill_nvfp4s3_tma_kernel<Geom, GqaPrefillDirectMetadata, Stages>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kSmem);
    if (attr != cudaSuccess) {
        std::fprintf(stderr, "cudaFuncSetAttribute failed (stages=%d, smem=%d): %s\n", Stages,
                     kSmem, cudaGetErrorString(attr));
        std::exit(1);
    }

    // Stage the descriptor in device memory (SM120 forbids by-value CUtensorMap params).
    DeviceBuffer desc_dev(sizeof(GqaNvfp4s3TmaDesc));
    desc_dev.copy_from_host(&desc, sizeof(desc));
    const GqaNvfp4s3TmaDesc* desc_ptr = static_cast<const GqaNvfp4s3TmaDesc*>(desc_dev.p);

    const dim3 grid(div_up(window, kGqaPrefillNvfp4s3Br), kQHeads, 1u);
    auto launch = [&](cudaStream_t s) {
        launch_tma_once<Stages>(cache, q_src, pos_q, window, keep_frac, desc_ptr, out_buf.p, s);
    };
    launch(stream);
    const cudaError_t warmup_err = cudaStreamSynchronize(stream);
    const cudaError_t launch_err  = cudaGetLastError();
    if (warmup_err != cudaSuccess || launch_err != cudaSuccess) {
        std::fprintf(stderr,
                     "TMA kernel launch failed (stages=%d, smem=%d, grid=%dx%d): launch=%s sync=%s\n",
                     Stages, kSmem, grid.x, grid.y, cudaGetErrorString(launch_err),
                     cudaGetErrorString(warmup_err));
        std::exit(1);
    }
    ColdTiming t = measure_launch(launch, stream, 8, 64);
    if (cudaGetLastError() != cudaSuccess) {
        std::fprintf(stderr, "TMA kernel launch failed (stages=%d): %s\n", Stages,
                     cudaGetErrorString(cudaGetLastError()));
        std::exit(1);
    }
    return t;
}

} // namespace

int main() {
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    const bool debug_map = std::getenv("NINFER_TMA_DEBUG") != nullptr;

    constexpr float keep_frac = 1.0f;
    int stages = 2;
    if (const char* e = std::getenv("NINFER_TMA_STAGES")) stages = std::strtol(e, nullptr, 10);
    std::printf("keep_frac=%.3f tma_stages=%d\n", keep_frac, stages);

    std::printf("ninfer gqa-attention nvfp4s3 TMA-vs-cpasync bench (geom=27B q=%d kv=%d hdim=%d window=%d)\n",
                 kQHeads, kKVHeads, kHeadDim, kQueryWin);
    print_device_caps("gqa-nvfp4s3-tma");
    {
        int optin = 0,
            default_smem = 0;
        cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
        cudaDeviceGetAttribute(&default_smem, cudaDevAttrMaxSharedMemoryPerBlock, 0);
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        std::printf("[gqa-nvfp4s3-tma] smem limits: default/blk=%d optin/blk=%d | %s sm_%d%d\n", default_smem,
                    optin, prop.name, prop.major, prop.minor);
    }
    std::printf("%-8s %8s %14s %14s %12s %10s %10s\n", "context", "window", "cpasync(us)",
                "tma(us)", "speedup", "tma GB/s", "maxdiff");
    std::printf("------------------------------------------------------------------------------\n");

    int max_ctx = 153600;
    if (const char* e = std::getenv("NINFER_MAX_CTX")) max_ctx = std::strtol(e, nullptr, 10);
    const std::vector<int> contexts = {64, 128, 256, 512, 1024, 4096, 8192, 16384, 32768, 65536, 153600};
    const char* ncu_ctx_env = std::getenv("NINFER_NCU_CTX");
    const int ncu_ctx = ncu_ctx_env ? std::strtol(ncu_ctx_env, nullptr, 10) : -1;
    for (int context : contexts) {
        if (context > max_ctx) break;
        if (ncu_ctx > 0 && context != ncu_ctx) continue;
        Nvfp4s3Cache cache = make_cache(context);

        const std::size_t kv_elems = static_cast<std::size_t>(context) * kKVHeads * kHeadDim;
        DeviceBuffer k_src = make_bf16(static_cast<std::size_t>(context) * kKVHeads * kHeadDim);
        DeviceBuffer v_src = make_bf16(kv_elems);
        std::vector<std::int32_t> h_pos(context);
        for (int i = 0; i < context; ++i) h_pos[i] = i;
        DeviceBuffer pos_fill(static_cast<std::size_t>(context) * sizeof(std::int32_t));
        pos_fill.copy_from_host(h_pos.data(), static_cast<std::size_t>(context) * sizeof(std::int32_t));

        Tensor k_fill = Tensor(k_src.p, DType::BF16, {kHeadDim, kKVHeads, context});
        Tensor v_fill = Tensor(v_src.p, DType::BF16, {kHeadDim, kKVHeads, context});
        Tensor pos_fill_t = Tensor(pos_fill.p, DType::I32, {context});
        auto fill_launch = [&](cudaStream_t s) {
            detail::gqa_kv_append_launch(k_fill, v_fill, pos_fill_t, cache.view, s);
        };
        measure_launch(fill_launch, stream, 2, 8);  // one-shot fill

        const int window = std::min(kQueryWin, context);
        const std::size_t q_elems = static_cast<std::size_t>(window) * kQHeads * kHeadDim;
        DeviceBuffer q_src = make_bf16(q_elems);
        DeviceBuffer out_buf = DeviceBuffer(static_cast<std::size_t>(q_elems) * 2);
        std::vector<std::int32_t> h_pos_q(window);
        for (int i = 0; i < window; ++i) h_pos_q[i] = (context - window) + i;
        DeviceBuffer pos_q(static_cast<std::size_t>(window) * sizeof(std::int32_t));
        pos_q.copy_from_host(h_pos_q.data(), static_cast<std::size_t>(window) * sizeof(std::int32_t));

        Tensor q_t = Tensor(q_src.p, DType::BF16, {kHeadDim, kQHeads, window});
        Tensor pos_q_t = Tensor(pos_q.p, DType::I32, {window});
        Tensor out_t = Tensor(out_buf.p, DType::BF16, {kHeadDim, kQHeads, window});
        // Dump A/B (small context): capture both kernels' named intermediates and diff them.
        const bool dump_on = (context == 64 && keep_frac >= 1.0f);
        S3Dump dump_base, dump_tma;
        DeviceBuffer dbg_stage(dump_on ? (4u * (64u * 128u + kGqaPrefillNvfp4s3KScaleBytes)) : 1u);
        const std::size_t dump_tiles =
            dump_on ? ((static_cast<std::size_t>(context) + 63) / 64 + 1) : 1;
        if (dump_on) {
            dump_base.alloc(kQHeads, static_cast<std::int32_t>(dump_tiles));
            dump_tma.alloc(kQHeads, static_cast<std::int32_t>(dump_tiles));
        }
        auto attn_launch = [&](cudaStream_t s) {
            detail::gqa_attention_prompt_attention_launch(q_t, pos_q_t, kScale, cache.view, out_t, s, keep_frac);
        };
        if (ncu_ctx == context) {
            // NCU single-shot: one cp.async launch + one TMA launch, no timing loops.
            detail::gqa_attention_prompt_attention_launch(q_t, pos_q_t, kScale, cache.view, out_t,
                                                           stream, keep_frac);
            GqaNvfp4s3TmaDesc ncu_desc;
            GqaNvfp4s3TmaDesc::build<Geom, kGqaPrefillNvfp4s3Bc>(
                static_cast<const std::uint8_t*>(cache.k_pages.p),
                static_cast<const std::uint8_t*>(cache.v_pages.p),
                static_cast<const std::uint8_t*>(cache.k_scale.p), cache.physical_pages, &ncu_desc);
            DeviceBuffer ncu_desc_dev(sizeof(GqaNvfp4s3TmaDesc));
            ncu_desc_dev.copy_from_host(&ncu_desc, sizeof(GqaNvfp4s3TmaDesc));
            const auto* ncu_ptr = static_cast<const GqaNvfp4s3TmaDesc*>(ncu_desc_dev.p);
            constexpr int ncu_smem = static_cast<int>(sizeof(GqaNvfp4s3TmaScratch<2>));
            constexpr int ncu_smem3 = static_cast<int>(sizeof(GqaNvfp4s3TmaScratch<3>));
            if (stages == 3) {
                CUDA_CHECK(cudaFuncSetAttribute(
                    gqa_attention_prefill_nvfp4s3_tma_kernel<Geom, GqaPrefillDirectMetadata, 3>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, ncu_smem3));
                launch_tma_once<3>(cache, q_src, pos_q, window, keep_frac, ncu_ptr, out_buf.p, stream);
            } else {
                CUDA_CHECK(cudaFuncSetAttribute(
                    gqa_attention_prefill_nvfp4s3_tma_kernel<Geom, GqaPrefillDirectMetadata, 2>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, ncu_smem));
                launch_tma_once<2>(cache, q_src, pos_q, window, keep_frac, ncu_ptr, out_buf.p, stream);
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
            std::printf("[ncu] ctx=%d single-shot cpasync+tma done (stages=%d)\n", context, stages);
            continue;
        }
        const ColdTiming cpasync = measure_launch(attn_launch, stream, 8, 64);

        ColdTiming tma;
        if (stages == 2) tma = run_tma<2>(cache, q_src, out_buf, pos_q, window, keep_frac, stream);
        else if (stages == 3) tma = run_tma<3>(cache, q_src, out_buf, pos_q, window, keep_frac, stream);
        else { tma = run_tma<2>(cache, q_src, out_buf, pos_q, window, keep_frac, stream); }

        // Numeric cross-check: cp.async baseline result vs TMA result (max abs diff).
        DeviceBuffer out_ref(static_cast<std::size_t>(q_elems) * 2);
        Tensor out_ref_t = Tensor(out_ref.p, DType::BF16, {kHeadDim, kQHeads, window});
        detail::gqa_attention_prompt_attention_launch(q_t, pos_q_t, kScale, cache.view, out_ref_t,
                                                      stream, keep_frac);
        GqaNvfp4s3TmaDesc check_desc;
        GqaNvfp4s3TmaDesc::build<Geom, kGqaPrefillNvfp4s3Bc>(
            static_cast<const std::uint8_t*>(cache.k_pages.p),
            static_cast<const std::uint8_t*>(cache.v_pages.p),
            static_cast<const std::uint8_t*>(cache.k_scale.p), cache.physical_pages,
            &check_desc);
        DeviceBuffer check_desc_dev(sizeof(GqaNvfp4s3TmaDesc));
        check_desc_dev.copy_from_host(&check_desc, sizeof(check_desc));
        const GqaNvfp4s3TmaDesc* check_desc_ptr =
            static_cast<const GqaNvfp4s3TmaDesc*>(check_desc_dev.p);
        if (stages == 3) {
            launch_tma_once<3>(cache, q_src, pos_q, window, keep_frac, check_desc_ptr, out_buf.p,
                               stream);
        } else {
            launch_tma_once<2>(cache, q_src, pos_q, window, keep_frac, check_desc_ptr, out_buf.p,
                               stream);
        }
        float h_max_diff = -1.0f;
        {
            DeviceBuffer d_diff(sizeof(float));
            gqa_bf16_max_abs_diff<<<(q_elems + 1023) / 1024, 256, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(out_ref.p),
                static_cast<const __nv_bfloat16*>(out_buf.p), static_cast<long>(q_elems),
                static_cast<float*>(d_diff.p));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            d_diff.copy_to_host(&h_max_diff, sizeof(float));
        }

        // Intermediate A/B: re-launch both kernels with the dump side-band and diff each
        // named stage (score -> p_code/psf -> m/l -> v_scale/v_t -> acc) to localize the
        // first divergent stage between the cp.async and TMA kernels.
        if (dump_on) {
            // Optional pattern fill (NINFER_TMA_PATTERN=1) for staged-layout inversion.
            if (std::getenv("NINFER_TMA_PATTERN") != nullptr) {
                const std::size_t page_bytes2 = (std::size_t)kPageSize * kKVHeads * 128;
                const std::size_t kplane_bytes2 = (std::size_t)cache.physical_pages * page_bytes2;
                std::vector<std::uint8_t> pat(kplane_bytes2, 0);
                for (int r = 0; r < kPageSize; ++r)
                    for (int c = 0; c < 64; ++c) {
                        const std::uint16_t v = (std::uint16_t)(r * 128 + 2 * c);
                        pat[r * 128 + 2 * c] = (std::uint8_t)(v & 0xff);
                        pat[r * 128 + 2 * c + 1] = (std::uint8_t)((v >> 8) & 0xff);
                    }
                CUDA_CHECK(cudaMemcpy(cache.k_pages.p, pat.data(), kplane_bytes2,
                                      cudaMemcpyHostToDevice));
            }
            DeviceBuffer regs_base(256 * sizeof(std::uint32_t));
            DeviceBuffer regs_tma(273 * sizeof(std::uint32_t));
            constexpr int kQDumpBytes = kGqaPrefillNvfp4s3QBytes + kGqaPrefillNvfp4s3QScaleBytes;
            DeviceBuffer qdump_base(kQDumpBytes);
            DeviceBuffer qdump_tma(kQDumpBytes);
            detail::gqa_attention_prompt_attention_launch(q_t, pos_q_t, kScale, cache.view,
                                                          out_ref_t, stream, keep_frac, 1.0f, 8192,
                                                          &dump_base.d,
                                                          static_cast<std::uint32_t*>(regs_base.p),
                                                          static_cast<std::uint8_t*>(qdump_base.p));
            if (stages == 3) {
                launch_tma_once<3>(cache, q_src, pos_q, window, keep_frac, check_desc_ptr,
                                   out_buf.p, stream, &dump_tma.d,
                                   static_cast<std::uint8_t*>(dbg_stage.p),
                                   static_cast<std::uint32_t*>(regs_tma.p),
                                   static_cast<std::uint8_t*>(qdump_tma.p));
            } else {
                launch_tma_once<2>(cache, q_src, pos_q, window, keep_frac, check_desc_ptr,
                                   out_buf.p, stream, &dump_tma.d,
                                   static_cast<std::uint8_t*>(dbg_stage.p),
                                   static_cast<std::uint32_t*>(regs_tma.p),
                                   static_cast<std::uint8_t*>(qdump_tma.p));
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
            // MMA operand A/B (warp 0, k64=0, nt=0): af[4], bf[2], sfa, sfb per lane.
            {
                std::vector<std::uint32_t> hb(257), ht(273);
                regs_base.copy_to_host(hb.data(), 256 * sizeof(std::uint32_t));
                regs_tma.copy_to_host(ht.data(), 273 * sizeof(std::uint32_t));
                std::printf("  [dump] tma kbase8=%u  b_physical(lane0..7)=%u %u %u %u %u %u %u %u",
                            ht[256], ht[257], ht[258], ht[259], ht[260], ht[261], ht[262],
                            ht[263], ht[264]);
                std::printf("\n");
                const char* fname[8] = {"af0", "af1", "af2", "af3", "bf0", "bf1", "sfa", "sfb"};
                std::printf("  [dump] mma operands (warp0,k64=0,nt=0) per-field mismatches:");
                 for (int f = 0; f < 8; ++f) {
                     int bad = 0;
                     int flane = -1;
                     for (int lane_i = 0; lane_i < 32; ++lane_i)
                         if (hb[lane_i * 8 + f] != ht[lane_i * 8 + f]) {
                             if (flane < 0) flane = lane_i;
                             ++bad;
                         }
                     if (bad)
                         std::printf(" %s=%d(l%d:b=0x%08x,t=0x%08x)", fname[f], bad, flane,
                                     hb[flane * 8 + f], ht[flane * 8 + f]);
                 }
                 std::printf("\n");
                 std::printf("  [dump] mma operand dump done\n");
            }
            // Raw Q smem A/B: q_codes (16 KiB) + q_scale (2 KiB), written by the kernel's
            // own quantize loop before the main loop.
            {
                std::vector<std::uint8_t> qb(kQDumpBytes), qt(kQDumpBytes);
                qdump_base.copy_to_host(qb.data(), kQDumpBytes);
                qdump_tma.copy_to_host(qt.data(), kQDumpBytes);
                int bad = 0;
                int fb = -1;
                for (int i = 0; i < kQDumpBytes; ++i)
                    if (qb[i] != qt[i]) {
                        if (fb < 0) fb = i;
                        ++bad;
                    }
                std::printf("  [dump] q smem (codes+scale): %d/%d mismatch%s\n", bad,
                            kQDumpBytes, fb >= 0 ? "  first@" : "");
                if (fb >= 0)
                    std::printf("  [dump]   byte%d %s idx=0x%02x tma=0x%02x base=0x%02x\n", fb,
                                fb >= kGqaPrefillNvfp4s3QBytes ? "q_scale" : "q_codes",
                                fb >= kGqaPrefillNvfp4s3QBytes ? (fb - kGqaPrefillNvfp4s3QBytes)
                                                               : fb,
                                qt[fb], qb[fb]);
                if (fb >= 0 && fb < kGqaPrefillNvfp4s3QBytes) {
                    const int row = fb / kGqaPrefillNvfp4s3CodeW;
                    const int col = fb % kGqaPrefillNvfp4s3CodeW;
                    std::printf("  [dump]   -> q_codes row%d byte%d (logical col, 16B chunks\n"
                                "     of d-dim)\n",
                                row, col);
                }
            }
            // Verify the TMA K staging in isolation: the raw staged smem bytes (head 0,
            // tiles 0..3) must equal the global K plane bytes permuted by the measured
            // swizzle table. Identity page map: tile ki -> page ki.
            {
                // Measured TMA SWIZZLE_128B mapping (rank-4 probe + pattern inversion,
                // with and without static smem): 16 B chunk c of key row r lands in
                // physical chunk c ^ ((T8 + r) & 7) where T8 = (align128(static smem) +
                // offsetof(k_codes)) >> 7 & 7 (the tile's CTA-smem 128 B-row index; the
                // stage stride is 72 rows, 0 mod 8). The static smem size comes from
                // the kernel's cudaFuncAttributes (0 after the proxy-array strip; the
                // cp.async-era static arrays made T8 differ by 1 row).
                cudaFuncAttributes tfa{};
                if (stages == 3)
                    CUDA_CHECK(cudaFuncGetAttributes(&tfa, (const void*)
                                                       gqa_attention_prefill_nvfp4s3_tma_kernel<Geom, GqaPrefillDirectMetadata, 3>));
                else
                    CUDA_CHECK(cudaFuncGetAttributes(&tfa, (const void*)
                                                       gqa_attention_prefill_nvfp4s3_tma_kernel<Geom, GqaPrefillDirectMetadata, 2>));
                const std::size_t t_row = ((tfa.sharedSizeBytes + 127) & ~std::size_t(127)) +
                                          offsetof(GqaNvfp4s3TmaScratch<2>, k_codes);
                const int T8 = static_cast<int>(t_row >> 7) & 7;
                auto phys_of = [T8](int r, int b) -> std::size_t {
                    return static_cast<std::size_t>(r) * 128 +
                           16u * static_cast<unsigned>((b >> 4) ^ ((T8 + r) & 7)) + (b & 15);
                };
                std::printf("  [dump] swizzle base row T8=%d (static smem=%zu B)\n", T8,
                            tfa.sharedSizeBytes);
                 const std::size_t tile_bytes = 64 * 128;
                 const std::size_t tile_stride = tile_bytes + kGqaPrefillNvfp4s3KScaleBytes;
                 const std::size_t n_tiles_check =
                     (static_cast<std::size_t>(context) + 63) / 64 < 4
                         ? (static_cast<std::size_t>(context) + 63) / 64
                         : 4;
                 std::vector<std::uint8_t> staged(4 * tile_stride);
                 CUDA_CHECK(cudaMemcpy(staged.data(), dbg_stage.p, 4 * tile_stride,
                                       cudaMemcpyDeviceToHost));
                // Invert the kernel's ACTUAL staging layout via the pattern plane:
                // each physical 16B slot's first uint16 identifies (key, 16B chunk).
                if (std::getenv("NINFER_TMA_PATTERN") != nullptr) {
                    const std::uint8_t* st0 = staged.data();
                    int order[64][8];
                    for (int r = 0; r < 64; ++r)
                        for (int c = 0; c < 8; ++c) order[r][c] = -1;
                    int resolved = 0;
                    for (int pr = 0; pr < 64; ++pr)
                        for (int pc = 0; pc < 8; ++pc) {
                            const std::size_t p = static_cast<std::size_t>(pr) * 128 + pc * 16;
                            const std::uint16_t v =
                                static_cast<std::uint16_t>(st0[p] | (static_cast<unsigned>(st0[p + 1]) << 8));
                            if (v % 2 != 0 || v / 128 >= 64) continue;
                            const int key = v / 128;
                            const int c16 = (v % 128) / 2;
                            if (c16 % 8 != 0) continue;
                            order[key][c16 / 8] = pc;
                            ++resolved;
                        }
                    std::printf("  [dump] kernel-staged layout (resolved %d/512), 16B order vs expected c^((T8+r)&7):\n",
                                resolved);
                    int agree = 0;
                    for (int r = 0; r < 64; ++r) {
                        char line[160] = "";
                        char cell[24];
                        std::snprintf(line, sizeof(line), "    row %2d: ", r);
                        for (int c = 0; c < 8; ++c) {
                            const int expect = c ^ ((T8 + r) & 7);
                            const int actual = order[r][c];
                            if (actual == expect) ++agree;
                            std::snprintf(cell, sizeof(cell), "%d ", actual);
                            std::strcat(line, cell);
                        }
                        std::printf("%s\n", line);
                    }
                    std::printf("  [dump] cells matching expected formula: %d/512\n", agree);
                }
                const std::size_t page_bytes = static_cast<std::size_t>(kPageSize) * kKVHeads * 128;
                const std::size_t kplane_bytes =
                    static_cast<std::size_t>(cache.physical_pages) * page_bytes;
                std::vector<std::uint8_t> kplane(kplane_bytes);
                CUDA_CHECK(cudaMemcpy(kplane.data(), cache.k_pages.p, kplane_bytes,
                                      cudaMemcpyDeviceToHost));
                int bad_total = 0;
                int f_ki = -1, f_r = -1, f_b = -1;
                unsigned f_st = 0, f_gp = 0;
                for (std::size_t ki = 0; ki < n_tiles_check; ++ki) {
                    const std::uint8_t* st = staged.data() + ki * tile_stride;
                    const std::uint8_t* gp = kplane.data() + ki * page_bytes;  // head 0
                    for (int r = 0; r < 64; ++r)
                        for (int b = 0; b < 128; ++b) {
                            const std::size_t phys = phys_of(r, b);
                            const std::size_t glo = static_cast<std::size_t>(r) * 128 + b;
                            if (st[phys] != gp[glo]) {
                                if (f_ki < 0) {
                                    f_ki = static_cast<int>(ki);
                                    f_r  = r;
                                    f_b  = b;
                                    f_st = st[phys];
                                    f_gp = gp[glo];
                                }
                                ++bad_total;
                            }
                        }
                }
                const std::size_t total_checks = n_tiles_check * 64 * 128;
                std::printf("  [dump] staged-K vs global plane: %d/%zu mismatch",
                            bad_total, total_checks);
                if (f_ki >= 0)
                    std::printf("  first: tile%d row%d byte%d phys=0x%02x plane=0x%02x",
                                f_ki, f_r, f_b, f_st, f_gp);
                 std::printf("\n");
                 // k_scale staging check: SWIZZLE_NONE 4-D load -> linear 1024 B per tile;
                 // global layout [page][head][key][16 groups] (same as the fill/reader).
                 const std::size_t kscale_bytes =
                     static_cast<std::size_t>(kGroups) * kPageSize * kKVHeads * cache.physical_pages;
                 std::vector<std::uint8_t> kscale(kscale_bytes);
                 CUDA_CHECK(cudaMemcpy(kscale.data(), cache.k_scale.p, kscale_bytes,
                                       cudaMemcpyDeviceToHost));
                 int bad_ks = 0;
                 int f_ks_ki = -1, f_ks_o = -1;
                 unsigned f_ks_st = 0, f_ks_g = 0;
                 for (std::size_t ki = 0; ki < n_tiles_check; ++ki) {
                     const std::uint8_t* st = staged.data() + ki * tile_stride + tile_bytes;
                     const std::uint8_t* gp =
                         kscale.data() + ki * kPageSize * kKVHeads * kGroups;  // head 0
                     for (int r = 0; r < kPageSize; ++r)
                         for (int g = 0; g < kGroups; ++g) {
                             const std::size_t o = static_cast<std::size_t>(r) * kGroups + g;
                             if (st[o] != gp[o]) {
                                 if (f_ks_ki < 0) {
                                     f_ks_ki = static_cast<int>(ki);
                                     f_ks_o = static_cast<int>(o);
                                     f_ks_st = st[o];
                                     f_ks_g = gp[o];
                                 }
                                 ++bad_ks;
                             }
                         }
                 }
                 std::printf("  [dump] staged-k_scale vs global: %d/%zu mismatch",
                             bad_ks, n_tiles_check * kPageSize * kGroups);
                 if (f_ks_ki >= 0)
                     std::printf("  first: tile%d off%d staged=0x%02x global=0x%02x", f_ks_ki,
                                 f_ks_o, f_ks_st, f_ks_g);
                 std::printf("\n");
                // 0-mismatch candidate = the true mapping; all-far-from-zero means the
                // staged bytes are not a permutation of the plane tile (coordinate bug).
                const std::uint8_t* st0 = staged.data();
                const std::uint8_t* gp0 = kplane.data();
                auto cand_mismatch = [&](const char* name, auto phys_of) -> int {
                    int bad = 0;
                    for (int r = 0; r < 64; ++r)
                        for (int b = 0; b < 128; ++b)
                            if (st0[phys_of(r, b)] != gp0[static_cast<std::size_t>(r) * 128 + b])
                                ++bad;
                    (void)name;
                    return bad;
                };
                std::printf("  [dump] candidate layouts (tile0, head0): ");
                std::printf("id=%d ", cand_mismatch("id", [](int r, int b) -> std::size_t {
                    return static_cast<std::size_t>(r) * 128 + b;
                }));
                std::printf("sw16=%d ", cand_mismatch("sw16", [&](int r, int b) -> std::size_t {
                    return static_cast<std::size_t>(r) * 128 +
                           16u * static_cast<unsigned>((b >> 4) ^ ((T8 + r) & 7)) + (b & 15);
                }));
                std::printf("xorr1=%d ", cand_mismatch("xorr1", [](int r, int b) -> std::size_t {
                    return static_cast<std::size_t>(r) * 128 +
                           32u * static_cast<unsigned>(
                               ((b >> 5) ^ (((r + 1) >> 1) & 3))) +
                           (b & 31);
                }));
                std::printf("xorr=%d ", cand_mismatch("xorr", [](int r, int b) -> std::size_t {
                    return static_cast<std::size_t>(r) * 128 +
                           32u * static_cast<unsigned>(
                               ((b >> 5) & 3) ^ (r & 3)) +
                           (b & 31);
                }));
                std::printf("old16=%d", cand_mismatch("old16", [](int r, int b) -> std::size_t {
                    // cp.async-era helper: 16B segments, XOR (row & 7)
                    const int seg = b >> 4;        // 16B segment within the 128B row
                    const int off = b & 15;
                    return static_cast<std::size_t>(r) * 128 +
                           16u * static_cast<unsigned>((seg & 7) ^ (r & 7)) + off;
                }));
                std::printf("\n");
                // Which global (page, head) location does the staged tile actually
                // correspond to? Try identity + table mapping over all heads.
                for (int h = 0; h < 4; ++h) {
                    int bad_id = 0;
                    const std::uint8_t* gh = gp0 + h * 8192;
                    for (int r = 0; r < 64; ++r)
                        for (int b = 0; b < 128; ++b)
                            if (st0[r * 128 + b] != gh[r * 128 + b]) ++bad_id;
                    int bad_tab = 0;
                    for (int r = 0; r < 64; ++r)
                        for (int b = 0; b < 128; ++b)
                            if (st0[phys_of(r, b)] != gh[r * 128 + b]) ++bad_tab;
                    std::printf("  [dump] head%d: id=%d sw16=%d\n", h, bad_id, bad_tab);
                }
            }
            const std::size_t H  = kQHeads;
            const std::size_t T  = dump_tiles;
            const std::size_t Br = kGqaPrefillNvfp4s3Br;
            const std::size_t Bc = kGqaPrefillNvfp4s3Bc;
            const std::size_t Dd = kGqaPrefillHeadDim;
            auto cmp_field = [&](const char* name, const void* a, const void* b, std::size_t n,
                                 bool is_float) {
                const std::size_t bytes = is_float ? n * 4 : n;
                std::vector<char> ha(bytes);
                std::vector<char> hb(bytes);
                CUDA_CHECK(cudaMemcpy(ha.data(), a, bytes, cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(hb.data(), b, bytes, cudaMemcpyDeviceToHost));
                std::size_t bad  = 0;
                std::size_t first = n;
                for (std::size_t i = 0; i < n; ++i) {
                    bool eq;
                    if (is_float) {
                        const float fa = *reinterpret_cast<const float*>(ha.data() + i * 4);
                        const float fb = *reinterpret_cast<const float*>(hb.data() + i * 4);
                        eq = (fa == fb);
                    } else {
                        eq = (ha[i] == hb[i]);
                    }
                    if (!eq && bad == 0) first = i;
                    if (!eq) ++bad;
                }
                std::printf("  [dump] %-8s: %zu/%zu mismatch", name, bad, n);
                if (first < n) {
                    const char* sa = ha.data() + first * (is_float ? 4 : 1);
                    const char* sb = hb.data() + first * (is_float ? 4 : 1);
                    if (is_float) {
                        std::printf("  first@%zu base=%.8g tma=%.8g", first,
                                    *reinterpret_cast<const float*>(sa),
                                    *reinterpret_cast<const float*>(sb));
                    } else {
                        std::printf("  first@%zu base=0x%02x tma=0x%02x", first,
                                    static_cast<unsigned>(static_cast<unsigned char>(*sa)),
                                    static_cast<unsigned>(static_cast<unsigned char>(*sb)));
                    }
                }
                std::printf("\n");
            };
            cmp_field("score", dump_base.d.score, dump_tma.d.score, H * T * Br * Bc, true);
            // Localize the score divergence: 8 x 4 grid of (16-row tile, 16-key block)
            // mismatch fractions for the first (head, tile) pair.
            {
                const float* sb = dump_base.d.score;  // [h][t][128][64]; head0 tile0 first
                const float* st = dump_tma.d.score;
                std::vector<char> ha(128 * 64 * 4);
                std::vector<char> hb(128 * 64 * 4);
                CUDA_CHECK(cudaMemcpy(ha.data(), sb, ha.size(), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(hb.data(), st, hb.size(), cudaMemcpyDeviceToHost));
                std::printf("  [dump] score grid (rows 0-127 x keys 0-63, head0 tile0):\n");
                for (int rt = 0; rt < 8; ++rt) {
                    char line[128] = "    rt=%d: ";
                    std::snprintf(line, sizeof(line), "    rt=%d: ", rt);
                    for (int kb = 0; kb < 4; ++kb) {
                        int bad = 0;
                        for (int r = rt * 16; r < rt * 16 + 16; ++r)
                            for (int k = kb * 16; k < kb * 16 + 16; ++k) {
                                const float a =
                                    *reinterpret_cast<const float*>(ha.data() + (r * 64 + k) * 4);
                                const float b =
                                    *reinterpret_cast<const float*>(hb.data() + (r * 64 + k) * 4);
                                if (a != b) ++bad;
                            }
                        char cell[24];
                        std::snprintf(cell, sizeof(cell), "%4d/256   ", bad);
                        std::strcat(line, cell);
                    }
                    std::printf("%s\n", line);
                }
            }
            cmp_field("p_code", dump_base.d.p_code, dump_tma.d.p_code, H * T * Br * Bc, false);
            cmp_field("psf", dump_base.d.psf, dump_tma.d.psf, H * T * Br * 4, false);
            cmp_field("v_scale", dump_base.d.v_scale, dump_tma.d.v_scale, H * T * Dd * 4, false);
            cmp_field("v_t", dump_base.d.v_t, dump_tma.d.v_t, H * T * Dd * 32, false);
            cmp_field("m", dump_base.d.m, dump_tma.d.m, H * T * Br, true);
            cmp_field("l", dump_base.d.l, dump_tma.d.l, H * T * Br, true);
            cmp_field("acc", dump_base.d.acc, dump_tma.d.acc, H * T * Br * Dd, true);
            {
                std::vector<std::int32_t> ka(H * T), kb(H * T), ta(H), tb(H);
                CUDA_CHECK(cudaMemcpy(ka.data(), dump_base.d.keep_list, H * T * 4,
                                      cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(kb.data(), dump_tma.d.keep_list, H * T * 4,
                                     cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(ta.data(), dump_base.d.tile_count, H * 4,
                                      cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(tb.data(), dump_tma.d.tile_count, H * 4,
                                      cudaMemcpyDeviceToHost));
                bool same = true;
                for (std::size_t i = 0; i < H && same; ++i) same = (ta[i] == tb[i]);
                for (std::size_t i = 0; i < H * T && same; ++i) same = (ka[i] == kb[i]);
                std::printf("  [dump] keep/tiles: %s\n",
                            same ? "identical" : "DIVERGENT");
            }
        }

        // Oracle gate (small contexts only): score BOTH kernels against the host FP64 S3
        // reference instead of trusting the cross-kernel diff (independent e4m3/e2m1
        // rounding legitimately separates two correct implementations).
        if (context <= 128 && keep_frac >= 1.0f) {
            std::vector<__nv_bfloat16> h_qb(static_cast<std::size_t>(q_elems));
            q_src.copy_to_host(h_qb.data(), static_cast<std::size_t>(q_elems) * 2);
            std::vector<float> h_q(static_cast<std::size_t>(q_elems));
            for (std::size_t i = 0; i < q_elems; ++i) h_q[i] = __bfloat162float(h_qb[i]);
            const std::size_t code_bytes  =
                static_cast<std::size_t>(kCodeW) * kPageSize * kKVHeads * cache.physical_pages;
            const std::size_t scale_bytes =
                static_cast<std::size_t>(kGroups) * kPageSize * kKVHeads * cache.physical_pages;
            std::vector<std::uint8_t> h_kc(code_bytes), h_vc(code_bytes);
            std::vector<std::uint8_t> h_ks(scale_bytes), h_vs(scale_bytes);
            cache.k_pages.copy_to_host(h_kc.data(), code_bytes);
            cache.v_pages.copy_to_host(h_vc.data(), code_bytes);
            cache.k_scale.copy_to_host(h_ks.data(), scale_bytes);
            cache.v_scale.copy_to_host(h_vs.data(), scale_bytes);
            const std::vector<double> ref =
                s3oracle::sage_ideal_attention(h_q, h_pos_q, h_kc.data(), h_vc.data(),
                                               h_ks.data(), h_vs.data());
            const s3oracle::Criterion crit;
            const auto score = [&](const DeviceBuffer& out, const char* label) {
                std::vector<__nv_bfloat16> h_out(static_cast<std::size_t>(q_elems));
                out.copy_to_host(h_out.data(), static_cast<std::size_t>(q_elems) * 2);
                double num = 0.0, den = 0.0, gmax = 0.0, rmax = 0.0;
                for (std::size_t i = 0; i < q_elems; ++i) {
                    const double o = __bfloat162float(h_out[i]);
                    const double r = ref[i];
                    const double df = std::fabs(o - r);
                    num += df * df;
                    den += r * r;
                    gmax = std::max(gmax, df);
                    rmax = std::max(rmax, std::fabs(r));
                }
                const double rel_l2   = std::sqrt(num / den);
                const double gross_rel = gmax / std::max(1e-12, rmax);
                const bool pass = rel_l2 <= crit.relative_l2 &&
                                  gross_rel <= crit.gross_relative_to_max_reference;
                std::printf(
                    "[oracle %s] ctx=%d rel_L2=%.4f (crit %.2f) gross=%.4f (abs crit %.2f) "
                    "gross_rel=%.3f (crit %.2f) -> %s\n",
                    label, context, rel_l2, crit.relative_l2, gmax, crit.gross_absolute,
                    gross_rel, crit.gross_relative_to_max_reference, pass ? "PASS" : "FAIL");
            };
            score(out_ref, "cpasync");
            score(out_buf, "tma");
        }

        if (debug_map && context == 4096) {
            // Bucketed diff map: max |cpasync - tma| per (head, row/128, dim/32).
            const int rblks = div_up(window, 128);
            DeviceBuffer d_map(static_cast<std::size_t>(kQHeads) * rblks * 8 * sizeof(float));
            float* h_map = static_cast<float*>(std::calloc(
                static_cast<std::size_t>(kQHeads) * rblks * 8, sizeof(float)));
            {
                const dim3 g(kQHeads, rblks, 8);
                gqa_s3_bucket_diff<<<g, 256, 0, stream>>>(
                    static_cast<const __nv_bfloat16*>(out_ref.p),
                    static_cast<const __nv_bfloat16*>(out_buf.p),
                    static_cast<float*>(d_map.p), window);
                CUDA_CHECK(cudaStreamSynchronize(stream));
                d_map.copy_to_host(h_map, static_cast<std::size_t>(kQHeads) * rblks * 8 * sizeof(float));
            }
            auto hval = [](float m) -> char {
                if (m < 1e-4f) return '.';
                const int e = static_cast<int>(std::log10(m));
                return static_cast<char>('0' + (e < 0 ? 0 : e > 9 ? 9 : e + 1));
            };
            int worst_head = 0;
            for (int h = 1; h < kQHeads; ++h) {
                for (int i = 0; i < rblks * 8; ++i)
                    if (h_map[h * rblks * 8 + i] > h_map[worst_head * rblks * 8 + i]) worst_head = h;
            }
            std::printf("[debug-map] window=%d rblks=%d (char = 1+log10(maxdiff), '.' < 1e-4)\n", window,
                         rblks);
            std::printf("[debug-map] per-head max:\n");
            for (int h = 0; h < kQHeads; ++h) {
                float m = 0.0f;
                for (int i = 0; i < rblks * 8; ++i) m = std::max(m, h_map[h * rblks * 8 + i]);
                std::printf(" %d:%.1e", h, m);
            }
            std::printf("\n");
            std::printf("[debug-map] rowblocks(128) x dchunks(32) for worst head %d:\n", worst_head);
            std::printf("        ");
            for (int db = 0; db < 8; ++db) std::printf("%4d", db * 32);
            std::printf("\n");
            for (int rb = 0; rb < rblks; ++rb) {
                std::printf("  %4d ", rb * 128);
                for (int db = 0; db < 8; ++db)
                    std::printf("  %c", hval(h_map[worst_head * rblks * 8 + rb * 8 + db]));
                std::printf("\n");
            }
            std::free(h_map);
        }

        const double bytes_moved =
            static_cast<double>(context) * (2 * (kCodeW + kGroups)) + 2 * static_cast<double>(q_elems) * 2.0;
        const double speedup = cpasync.median_us / tma.median_us;
        std::printf("%-8d %8d %14.1f %14.1f %12.2fx %10.1f %10.2e\n",
                    context, window, cpasync.median_us, tma.median_us, speedup,
                    bytes_moved / (tma.median_us * 1e-6) / 1e9, h_max_diff);
    }
    return 0;
}