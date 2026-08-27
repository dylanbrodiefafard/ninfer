// Isolated perf bench for exact-NVFP4 prefill with optional Sparge / XAttention
// skip-list kernels (gqa_attention_prefill_nvfp4.cuh dense, _sparse.cuh skip).
//
// Usage:
//   ninfer_gqa_attention_nvfp4_sparse_bench [--keep-frac F] [--xattn-tau F]
//     [--tokens LIST] [--contexts LIST] [--xattn-min-len N] [--warmup N] [--repeat N]
// Default: dense exact NVFP4 (keep_frac=1, tau=1), T=4096, contexts 8k/32k/64k/128k.

#include "core/device.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"
#include "ninfer_bench_common.h"
#include "ninfer/ops/gqa_attention.h"
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

PagedKVBatchLayerView batch_view(Nvfp4Cache& cache) {
    return {
        .k_pages       = cache.view.k_pages,
        .v_pages       = cache.view.v_pages,
        .k_scale_pages = cache.view.k_scale_pages,
        .v_scale_pages = cache.view.v_scale_pages,
        .k_mean_pages  = cache.view.k_mean_pages,
        .block_tables  = cache.view.block_table.view({cache.logical_pages, 1}),
        .head_dim      = cache.view.head_dim,
        .num_kv_heads  = cache.view.num_kv_heads,
        .dtype         = cache.view.dtype,
        .quant_group   = cache.view.quant_group,
        .sage_pv       = cache.view.sage_pv,
    };
}

std::vector<int> parse_positive_list(const char* text, const char* flag) {
    std::vector<int> values;
    const std::string list(text);
    std::size_t begin = 0;
    while (begin <= list.size()) {
        const std::size_t comma = list.find(',', begin);
        const std::string item  = list.substr(begin, comma - begin);
        if (item.empty()) { throw std::invalid_argument(std::string(flag) + " has an empty item"); }
        const int value = std::stoi(item);
        if (value <= 0) {
            throw std::invalid_argument(std::string(flag) + " values must be positive");
        }
        values.push_back(value);
        if (comma == std::string::npos) { break; }
        begin = comma + 1;
    }
    return values;
}

int parse_positive(const char* text, const char* flag) {
    const int value = std::stoi(text);
    if (value <= 0) { throw std::invalid_argument(std::string(flag) + " must be positive"); }
    return value;
}

} // namespace

int main(int argc, char** argv) {
    try {
        float keep_frac         = 1.0f;
        float xattn_tau         = 1.0f;
        int xattn_min_len       = kDefaultXattnMinLen;
        int warmup              = 8;
        int repeat              = 64;
        std::vector<int> tokens = {4096};
        std::vector<int> contexts = {8192, 32768, 65536, 131072};
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
            } else if (arg == "--tokens") {
                tokens = parse_positive_list(value("--tokens"), "--tokens");
            } else if (arg == "--contexts") {
                contexts = parse_positive_list(value("--contexts"), "--contexts");
            } else if (arg == "--xattn-min-len") {
                xattn_min_len = std::stoi(value("--xattn-min-len"));
                if (xattn_min_len < 0) {
                    throw std::invalid_argument("--xattn-min-len must be non-negative");
                }
            } else if (arg == "--warmup") {
                warmup = parse_positive(value("--warmup"), "--warmup");
            } else if (arg == "--repeat") {
                repeat = parse_positive(value("--repeat"), "--repeat");
            } else if (arg == "-h" || arg == "--help") {
                std::printf(
                    "Usage: ninfer_gqa_attention_nvfp4_sparse_bench [--keep-frac F] "
                    "[--xattn-tau F] [--tokens LIST] [--contexts LIST] "
                    "[--xattn-min-len N] [--warmup N] [--repeat N]\n");
                return 0;
            } else {
                throw std::invalid_argument("unknown argument: " + arg);
            }
        }
        validate_sparse_attn_flags(KvCacheStorage::Nvfp4, false, keep_frac, xattn_tau);
        const bool want_k_mean = keep_frac < 1.0f;

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreate(&stream));

        std::printf("keep_frac=%.3f xattn_tau=%.3f xattn_min_len=%d\n", keep_frac, xattn_tau,
                    xattn_min_len);
        std::printf("ninfer public gqa_attention NVFP4 prefill "
                    "(geom=27B q=%d kv=%d hdim=%d)\n",
                    kQHeads, kKVHeads, kHeadDim);
        print_device_caps("gqa-nvfp4-sparse-prefill");
        std::printf("%-8s %8s %12s %14s %12s %12s %10s\n", "context", "T", "fill(us)",
                    "op median(us)", "op p95(us)", "workspace", "attn TF/s");
        std::printf("--------------------------------------------------------------------------------\n");

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
                gqa_kv_append(k_fill, v_fill, pos_fill_t, cache.view, s);
            };
            const ColdTiming fill_timing = measure_launch(fill_launch, stream, 2, 8);

            for (const int width : tokens) {
                if (width > context) {
                    throw std::invalid_argument("every --tokens value must be <= every context");
                }
                const std::size_t q_elems =
                    static_cast<std::size_t>(width) * kQHeads * kHeadDim;
                const std::size_t kv_window_elems =
                    static_cast<std::size_t>(width) * kKVHeads * kHeadDim;
                DeviceBuffer q_query   = make_bf16(q_elems);
                DeviceBuffer k_query   = make_bf16(kv_window_elems);
                DeviceBuffer v_query   = make_bf16(kv_window_elems);
                DeviceBuffer out_buf(q_elems * sizeof(std::uint16_t));
                std::vector<std::int32_t> h_pos_q(width);
                for (int i = 0; i < width; ++i) { h_pos_q[i] = (context - width) + i; }
                DeviceBuffer pos_q(static_cast<std::size_t>(width) * sizeof(std::int32_t));
                pos_q.copy_from_host(h_pos_q.data(), h_pos_q.size() * sizeof(std::int32_t));
                const std::int32_t row_zero = 0;
                DeviceBuffer table_row(sizeof(row_zero));
                table_row.copy_from_host(&row_zero, sizeof(row_zero));

                Tensor q_t(q_query.p, DType::BF16, {kHeadDim, kQHeads, width, 1});
                Tensor k_t(k_query.p, DType::BF16, {kHeadDim, kKVHeads, width, 1});
                Tensor v_t(v_query.p, DType::BF16, {kHeadDim, kKVHeads, width, 1});
                Tensor pos_q_t(pos_q.p, DType::I32, {width, 1});
                Tensor table_rows(table_row.p, DType::I32, {1});
                Tensor out_t(out_buf.p, DType::BF16, {kHeadDim, kQHeads, width, 1});
                const Tensor empty;
                const GqaExecutionEnvelope envelope{static_cast<std::uint32_t>(context),
                                                    static_cast<std::uint32_t>(context)};
                const std::size_t workspace_bytes = gqa_attention_workspace_capacity_bytes(
                    kQHeads, DType::U8, envelope, 1, width, width, keep_frac, false, xattn_tau);
                WorkspaceArena workspace(std::max<std::size_t>(workspace_bytes, 1));
                auto op_launch = [&](cudaStream_t s) {
                    gqa_attention(q_t, k_t, v_t, pos_q_t, empty, table_rows, kScale,
                                  batch_view(cache), envelope, workspace, out_t, s, keep_frac,
                                  xattn_tau, xattn_min_len);
                };
                const ColdTiming op = measure_launch(op_launch, stream, warmup, repeat);
                if (workspace.peak_used() != workspace_bytes) {
                    throw std::runtime_error("gqa_attention workspace query/high-water mismatch");
                }

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
                    gqa_attention(q_t, k_t, v_t, pos_q_t, empty, table_rows, kScale,
                                  batch_view(cache), envelope, workspace, out_t, stream, keep_frac,
                                  xattn_tau, xattn_min_len, &dump);
                    CUDA_CHECK(cudaStreamSynchronize(stream));
                    std::vector<std::int32_t> hcount(kQHeads);
                    dcount.copy_to_host(hcount.data(), hcount.size() * sizeof(std::int32_t));
                    const int qabs = (context - width) + std::min(width, 128) - 1;
                    keep_d         = qabs / kPageSize + 1;
                    for (int h = 0; h < kQHeads; ++h) {
                        keep_n += hcount[static_cast<std::size_t>(h)];
                    }
                    keep_d *= kQHeads;
                }

                const double useful_flops =
                    4.0 * static_cast<double>(kQHeads) * width * context * kHeadDim;
                const double tflops = useful_flops / (op.median_us * 1e-6) / 1.0e12;
                std::printf("%-8d %8d %12.1f %14.1f %12.1f %12zu %10.1f", context, width,
                            fill_timing.median_us, op.median_us, op.p95_us, workspace_bytes,
                            tflops);
                if (keep_d > 0) {
                    std::printf("  keep q0 %d/%d (%.1f%%)", keep_n, keep_d,
                                100.0 * static_cast<double>(keep_n) /
                                    static_cast<double>(keep_d));
                }
                std::printf("\n");
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
