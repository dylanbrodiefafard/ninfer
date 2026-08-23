// ninfer_gqa_attention_decode_nvfp4s3_bench.cu
//
// Isolated perf bench for the SageAttention3-style NVFP4 DECODE attention kernel
// (the `--sage` / --kv-dtype nvfp4 route: NVFP4 KV cache + FP4 P-quant), which the
// live serve runs with `--kv-dtype nvfp4 --sage`:
//   decode: gqa_attention_cached -> gqa_attention_cached_small_t_launch
//                                   -> gqa_attention_decode_nvfp4s3_tiled_kernel + reduce
//
// The KV cache is filled by the on-device nvfp4s3 fill kernel (gqa_kv_append_launch)
// so the cache layout is exactly the production one; the fill is a one-shot setup
// (not in the timed loop). Only the decode attention kernel (split-KV partials +
// reduction) is timed. We sweep context length (visible keys) AND number of query
// tokens, since the production decode/verify path issues 1..6 token windows.
//
// Usage: ninfer_gqa_attention_decode_nvfp4s3_bench
// Reports the decode attention kernel's timing + GB/s so kernel-level speedups can
// be attributed cleanly without touching the live engine.
//
// NOTE: This routes through the public ninfer::ops::gqa_attention_cached entry, which
// owns the partial-buffer workspace internally (sized via
// gqa_attention_workspace_capacity_bytes). That is exactly the production decode call
// path, so the numbers reflect real per-decode-token attention latency.

#include "ninfer/ops/gqa_attention.h"
#include "ops/launcher/gqa_attention.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"
#include "ninfer_bench_common.h"

#include "ops/kernel/gqa_attention_geometry.cuh"
#include "ops/kernel/gqa_attention_kv_nvfp4.cuh"

#include <cuda_bf16.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

using namespace ninfer;
using namespace ninfer::ops;
using namespace ninfer::bench;

namespace {
using Geom = Gqa27Geometry;
constexpr int kQHeads    = Geom::QHeads;         // 24
constexpr int kKVHeads   = Geom::KVHeads;       // 4
constexpr int kHeadDim   = kGqaNvfp4HeadDim;    // 256
constexpr int kCodeW     = kGqaNvfp4CodeWidth;  // 128 bytes/key
constexpr int kGroups    = kGqaNvfp4Groups;     // 16 e4m3 scales/key
constexpr int kPageSize  = kPagedKVPageSize;    // 64
constexpr float kScale   = 0.0625f;  // 1/sqrt(head_dim) = 1/16

// Owns the U8 cache planes + the PagedKVLayerView that the launchers consume.
struct Nvfp4s3Cache {
    DeviceBuffer k_pages, v_pages, k_scale, v_scale, block_table;
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
    cache.v_pages = DeviceBuffer(code_bytes);
    cache.k_scale = DeviceBuffer(scale_bytes);
    cache.v_scale = DeviceBuffer(scale_bytes);
    cache.block_table = DeviceBuffer(table_bytes);
    cache.k_pages.fill(0);
    cache.v_pages.fill(0);
    cache.k_scale.fill(0);
    cache.v_scale.fill(0);

    std::vector<std::int32_t> h_table(logical_pages);
    for (int p = 0; p < logical_pages; ++p) h_table[p] = p;
    cache.block_table.copy_from_host(h_table.data(), table_bytes);

    cache.view = PagedKVLayerView{};
    cache.view.k_pages       = Tensor(cache.k_pages.p, DType::U8, {kCodeW, kPageSize, kKVHeads, physical_pages});
    cache.view.v_pages       = Tensor(cache.v_pages.p, DType::U8, {kCodeW, kPageSize, kKVHeads, physical_pages});
    cache.view.k_scale_pages = Tensor(cache.k_scale.p, DType::FP8_E4M3FN, {kGroups, kPageSize, kKVHeads, physical_pages});
    cache.view.v_scale_pages = Tensor(cache.v_scale.p, DType::FP8_E4M3FN, {kGroups, kPageSize, kKVHeads, physical_pages});
    cache.view.block_table   = Tensor(cache.block_table.p, DType::I32, {logical_pages});
    cache.view.head_dim      = kHeadDim;
    cache.view.num_kv_heads  = kKVHeads;
    cache.view.dtype         = DType::U8;
    cache.view.quant_group   = kGroups;
    cache.view.sage_pv       = true;
    return cache;
}

} // namespace

int main(int argc, char** argv) {
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Optional one-shot mode: `bench <context> <tokens>` profiles a single cell
    // (for NCU); default is the full sweep.
    const std::vector<int> contexts =
        (argc > 1) ? std::vector<int>{std::atoi(argv[1])}
                   : std::vector<int>{128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768,
                                      65536, 98304, 153600};
    const std::vector<int> tokens_list =
        (argc > 2) ? std::vector<int>{std::atoi(argv[2])} : std::vector<int>{1, 2, 4, 6};
    std::printf("ninfer gqa-decode nvfp4s3 bench (geom=27B q=%d kv=%d hdim=%d code=%d groups=%d)\n",
                 kQHeads, kKVHeads, kHeadDim, kCodeW, kGroups);
    print_device_caps("gqa-nvfp4s3-decode");
    std::printf("%-8s %6s %12s %14s %12s %10s %10s\n", "context", "tokens", "fill(us)", "dec median(us)",
                 "dec p95(us)", "dec GB/s", "dec TF/s");
    std::printf("--------------------------------------------------------------------------------\n");

    for (int context : contexts) {
        Nvfp4s3Cache cache = make_cache(context);

        // One-shot fill of the whole cache from bf16 k/v (production nvfp4s3 fill kernel).
        DeviceBuffer k_src = make_bf16(static_cast<std::size_t>(context) * kKVHeads * kHeadDim);
        DeviceBuffer v_src = make_bf16(static_cast<std::size_t>(context) * kKVHeads * kHeadDim);
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
        const ColdTiming fill_timing = measure_launch(fill_launch, stream, 2, 8);

        for (int tokens : tokens_list) {
            // Query token(s) at the tail of the context (the decode attention window).
            DeviceBuffer q_src = make_bf16(static_cast<std::size_t>(tokens) * kQHeads * kHeadDim);
            DeviceBuffer out_buf = DeviceBuffer(static_cast<std::size_t>(tokens) * kQHeads * kHeadDim * 2);
            std::vector<std::int32_t> h_pos_q(tokens);
            for (int i = 0; i < tokens; ++i) h_pos_q[i] = (context - tokens) + i;
            DeviceBuffer pos_q(static_cast<std::size_t>(tokens) * sizeof(std::int32_t));
            pos_q.copy_from_host(h_pos_q.data(), static_cast<std::size_t>(tokens) * sizeof(std::int32_t));

            Tensor q_t = Tensor(q_src.p, DType::BF16, {kHeadDim, kQHeads, tokens});
            Tensor pos_q_t = Tensor(pos_q.p, DType::I32, {tokens});
            Tensor out_t = Tensor(out_buf.p, DType::BF16, {kHeadDim, kQHeads, tokens});

            const GqaExecutionEnvelope envelope{static_cast<std::uint32_t>(context),
                                                 static_cast<std::uint32_t>(context)};
            const std::size_t ws_bytes = gqa_attention_workspace_capacity_bytes(
                kQHeads, DType::U8, envelope, 1, tokens, tokens);
            DeviceBuffer ws(std::max<std::size_t>(ws_bytes, 256));
            WorkspaceArena arena{DeviceSpan{ws.p, ws.bytes}};

            auto dec_launch = [&](cudaStream_t s) {
                gqa_attention_cached(q_t, pos_q_t, kScale, cache.view, envelope, arena, out_t, s);
            };
            const ColdTiming dec = measure_launch(dec_launch, stream, 8, 64);

            // GB/s moved by the decode kernel: read the K+V codes + scales for `context` keys
            // (per query token) and the tiny q/out traffic. (Informational; ncu is the real gate.)
        const double bytes_moved =
            static_cast<double>(context) * tokens * (2 * (kCodeW + kGroups)) +
            2 * static_cast<double>(tokens) * kQHeads * kHeadDim * 2.0;  // q in + out (bf16)
        // Useful tensor-core FLOPs: QK^T + PV mma (2 mma x 2 FLOP/MAC x QHeads x tokens x context x hdim).
        const double useful_flops = 4.0 * static_cast<double>(kQHeads) * tokens * context * kHeadDim;
        const double tflops = useful_flops / (dec.median_us * 1e-6) / 1.0e12;
        std::printf("%-8d %6d %12.1f %12.1f %14.1f %10.1f %10.1f\n",
                     context, tokens, fill_timing.median_us,
                     dec.median_us, dec.p95_us, bytes_moved / (dec.median_us * 1e-6) / 1e9, tflops);
        }
    }
    return 0;
}
