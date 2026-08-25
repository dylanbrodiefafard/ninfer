// Isolated perf bench for exact-NVFP4 prefill with optional Sparge / XAttention
// skip-list kernels (gqa_attention_prefill_nvfp4.cuh dense, _sparse.cuh skip).
//
// Usage:
//   ninfer_gqa_attention_nvfp4_sparse_bench [--keep-frac F] [--xattn-tau F]
// Default: dense exact NVFP4 (keep_frac=1, tau=1). Contexts 8k/32k/64k/128k.

#include "ops/launcher/gqa_attention.h"
#include "core/device.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"
#include "ninfer_bench_common.h"
#include "ninfer/types.h"

#include "ops/kernel/gqa_attention_geometry.cuh"
#include "ops/kernel/gqa_attention_kv_nvfp4.cuh"

#include <cuda_bf16.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::ops;
using namespace ninfer::bench;

namespace {
using Geom = Gqa27Geometry;
constexpr int kQHeads   = Geom::QHeads;
constexpr int kKVHeads  = Geom::KVHeads;
constexpr int kHeadDim  = kGqaNvfp4HeadDim;
constexpr int kCodeW    = kGqaNvfp4CodeWidth;
constexpr int kGroups   = kGqaNvfp4Groups;
constexpr int kPageSize = kPagedKVPageSize;
constexpr int kQueryWin = 4096;
constexpr float kScale  = 0.0625f;

struct Nvfp4Cache {
    DeviceBuffer k_pages, v_pages, k_scale, v_scale, k_mean, block_table;
    PagedKVLayerView view;
    int logical_pages  = 0;
    int physical_pages = 0;
    int capacity       = 0;
};

Nvfp4Cache make_cache(int context, bool want_k_mean) {
    const int cap            = ((context + 127) / 128) * 128;
    const int logical_pages  = cap / kPageSize;
    const int physical_pages = (cap + kPageSize - 1) / kPageSize;
    const std::size_t code_bytes =
        static_cast<std::size_t>(kCodeW) * kPageSize * kKVHeads * physical_pages;
    const std::size_t scale_bytes =
        static_cast<std::size_t>(kGroups) * kPageSize * kKVHeads * physical_pages;
    const std::size_t table_bytes = static_cast<std::size_t>(logical_pages) * sizeof(std::int32_t);

    Nvfp4Cache cache;
    cache.logical_pages  = logical_pages;
    cache.physical_pages = physical_pages;
    cache.capacity       = cap;
    cache.k_pages        = DeviceBuffer(code_bytes);
    cache.v_pages        = DeviceBuffer(code_bytes);
    cache.k_scale        = DeviceBuffer(scale_bytes);
    cache.v_scale        = DeviceBuffer(scale_bytes);
    cache.block_table    = DeviceBuffer(table_bytes);
    cache.k_pages.fill(0);
    cache.v_pages.fill(0);
    cache.k_scale.fill(0);
    cache.v_scale.fill(0);
    if (want_k_mean) {
        cache.k_mean = DeviceBuffer(static_cast<std::size_t>(4) * kPageSize * kKVHeads *
                                    physical_pages * sizeof(float));
        cache.k_mean.fill(0);
    }

    std::vector<std::int32_t> h_table(logical_pages);
    for (int p = 0; p < logical_pages; ++p) { h_table[p] = p; }
    cache.block_table.copy_from_host(h_table.data(), table_bytes);

    cache.view                 = PagedKVLayerView{};
    cache.view.k_pages         = Tensor(cache.k_pages.p, DType::U8,
                                {kCodeW, kPageSize, kKVHeads, physical_pages});
    cache.view.v_pages         = Tensor(cache.v_pages.p, DType::U8,
                                {kCodeW, kPageSize, kKVHeads, physical_pages});
    cache.view.k_scale_pages   = Tensor(cache.k_scale.p, DType::FP8_E4M3FN,
                                      {kGroups, kPageSize, kKVHeads, physical_pages});
    cache.view.v_scale_pages   = Tensor(cache.v_scale.p, DType::FP8_E4M3FN,
                                      {kGroups, kPageSize, kKVHeads, physical_pages});
    if (want_k_mean) {
        cache.view.k_mean_pages =
            Tensor(cache.k_mean.p, DType::FP32, {4, kPageSize, kKVHeads, physical_pages});
    }
    cache.view.block_table = Tensor(cache.block_table.p, DType::I32, {logical_pages});
    cache.view.head_dim    = kHeadDim;
    cache.view.num_kv_heads = kKVHeads;
    cache.view.dtype        = DType::U8;
    cache.view.quant_group  = kGroups;
    cache.view.sage_pv      = false;
    return cache;
}

} // namespace

int main(int argc, char** argv) {
    try {
        float keep_frac = 1.0f;
        float xattn_tau = 1.0f;
        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);
            auto value = [&](const char* flag) -> const char* {
                if (i + 1 >= argc) {
                    throw std::invalid_argument(std::string(flag) + " requires a value");
                }
                return argv[++i];
            };
            if (arg == "--keep-frac") {
                keep_frac = parse_unit_interval_flag(value("--keep-frac"), "--keep-frac");
            } else if (arg == "--xattn-tau") {
                xattn_tau = parse_unit_interval_flag(value("--xattn-tau"), "--xattn-tau");
            } else if (arg == "-h" || arg == "--help") {
                std::printf(
                    "Usage: ninfer_gqa_attention_nvfp4_sparse_bench [--keep-frac F] [--xattn-tau F]\n");
                return 0;
            } else {
                throw std::invalid_argument("unknown argument: " + arg);
            }
        }
        validate_sparse_attn_flags(KvCacheStorage::Nvfp4, false, keep_frac, xattn_tau);
        const bool want_k_mean = keep_frac < 1.0f;

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreate(&stream));

        std::printf("keep_frac=%.3f xattn_tau=%.3f\n", keep_frac, xattn_tau);
        std::printf("ninfer gqa-attention nvfp4-sparse bench (geom=27B q=%d kv=%d hdim=%d window=%d)\n",
                    kQHeads, kKVHeads, kHeadDim, kQueryWin);
        print_device_caps("gqa-nvfp4-sparse-prefill");
        std::printf("%-8s %8s %12s %14s %12s %10s %10s\n", "context", "window", "fill(us)",
                    "attn median(us)", "attn p95(us)", "attn GB/s", "attn TF/s");
        std::printf("------------------------------------------------------------------------------\n");

        std::vector<int> contexts = {8192, 32768, 65536, 131072};
        if (const char* e = std::getenv("NINFER_BENCH_MAX_CTX")) {
            const int cap = std::atoi(e);
            contexts.erase(std::remove_if(contexts.begin(), contexts.end(),
                                          [cap](int c) { return c > cap; }),
                           contexts.end());
        }
        if (const char* e = std::getenv("NINFER_BENCH_MIN_CTX")) {
            const int floor = std::atoi(e);
            contexts.erase(std::remove_if(contexts.begin(), contexts.end(),
                                          [floor](int c) { return c < floor; }),
                           contexts.end());
        }
        for (int context : contexts) {
            Nvfp4Cache cache = make_cache(context, want_k_mean);

            const std::size_t kv_elems = static_cast<std::size_t>(context) * kKVHeads * kHeadDim;
            DeviceBuffer k_src         = make_bf16(kv_elems);
            DeviceBuffer v_src         = make_bf16(kv_elems);
            std::vector<std::int32_t> h_pos(context);
            for (int i = 0; i < context; ++i) { h_pos[i] = i; }
            DeviceBuffer pos_fill(static_cast<std::size_t>(context) * sizeof(std::int32_t));
            pos_fill.copy_from_host(h_pos.data(),
                                    static_cast<std::size_t>(context) * sizeof(std::int32_t));

            Tensor k_fill     = Tensor(k_src.p, DType::BF16, {kHeadDim, kKVHeads, context});
            Tensor v_fill     = Tensor(v_src.p, DType::BF16, {kHeadDim, kKVHeads, context});
            Tensor pos_fill_t = Tensor(pos_fill.p, DType::I32, {context});
            auto fill_launch  = [&](cudaStream_t s) {
                detail::gqa_kv_append_launch(k_fill, v_fill, pos_fill_t, cache.view, s);
            };
            const ColdTiming fill_timing = measure_launch(fill_launch, stream, 2, 8);

            const int window          = std::min(kQueryWin, context);
            const std::size_t q_elems = static_cast<std::size_t>(window) * kQHeads * kHeadDim;
            DeviceBuffer q_src        = make_bf16(q_elems);
            DeviceBuffer out_buf      = DeviceBuffer(q_elems * 2);
            std::vector<std::int32_t> h_pos_q(window);
            for (int i = 0; i < window; ++i) { h_pos_q[i] = (context - window) + i; }
            DeviceBuffer pos_q(static_cast<std::size_t>(window) * sizeof(std::int32_t));
            pos_q.copy_from_host(h_pos_q.data(), static_cast<std::size_t>(window) * sizeof(std::int32_t));

            Tensor q_t     = Tensor(q_src.p, DType::BF16, {kHeadDim, kQHeads, window});
            Tensor pos_q_t = Tensor(pos_q.p, DType::I32, {window});
            Tensor out_t   = Tensor(out_buf.p, DType::BF16, {kHeadDim, kQHeads, window});
            auto attn_launch = [&](cudaStream_t s) {
                detail::gqa_attention_prompt_attention_launch(q_t, pos_q_t, kScale, cache.view, out_t, s,
                                                              keep_frac, xattn_tau, 0);
            };
            const ColdTiming attn = measure_launch(attn_launch, stream, 8, 64);

            int keep_n = 0;
            int keep_d = 0;
            if (keep_frac < 1.0f || xattn_tau < 1.0f) {
                const int max_tiles = (context + kPageSize - 1) / kPageSize;
                DeviceBuffer dkeep(static_cast<std::size_t>(kQHeads) * max_tiles *
                                   sizeof(std::int32_t));
                DeviceBuffer dcount(static_cast<std::size_t>(kQHeads) * sizeof(std::int32_t));
                GqaS3PrefillDump dump{};
                dump.max_tiles  = max_tiles;
                dump.keep_list  = static_cast<std::int32_t*>(dkeep.p);
                dump.tile_count = static_cast<std::int32_t*>(dcount.p);
                detail::gqa_attention_prompt_attention_launch(q_t, pos_q_t, kScale, cache.view,
                                                              out_t, stream, keep_frac, xattn_tau,
                                                              0, &dump);
                CUDA_CHECK(cudaStreamSynchronize(stream));
                std::vector<std::int32_t> hcount(kQHeads);
                dcount.copy_to_host(hcount.data(), hcount.size() * sizeof(std::int32_t));
                const int qabs = (context - window) + std::min(window, 128) - 1;
                keep_d         = qabs / kPageSize + 1;
                for (int h = 0; h < kQHeads; ++h) { keep_n += hcount[static_cast<std::size_t>(h)]; }
                keep_d *= kQHeads;
            }

            const double bytes_moved =
                static_cast<double>(context) * (2 * (kCodeW + kGroups)) +
                2 * static_cast<double>(q_elems) * 2.0;
            const double useful_flops =
                4.0 * static_cast<double>(kQHeads) * window * context * kHeadDim;
            const double tflops = useful_flops / (attn.median_us * 1e-6) / 1.0e12;
            std::printf("%-8d %8d %12.1f %12.1f %14.1f %10.1f %10.1f", context, window,
                        fill_timing.median_us, attn.median_us, attn.p95_us,
                        bytes_moved / (attn.median_us * 1e-6) / 1e9, tflops);
            if (keep_d > 0) {
                std::printf("  keep q0 %d/%d (%.1f%%)", keep_n, keep_d,
                            100.0 * static_cast<double>(keep_n) / static_cast<double>(keep_d));
            }
            std::printf("\n");
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
