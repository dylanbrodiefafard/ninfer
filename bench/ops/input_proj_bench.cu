// Fixed-shape production/control benchmark for the 27B Attention and GDN input projections.
// Controls reproduce only the superseded Small-T compositions and are intentionally benchmark-
// local: production dispatch has no fallback to them.

#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/scatter.h"

#include "core/device.h"
#include "direct_bf16_weight.cuh"
#include "ninfer_bench_common.h"
#include "quantized_weight.cuh"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace ninfer;

namespace {

constexpr std::int32_t kHidden            = 5120;
constexpr std::int32_t kQueryRows         = 6144;
constexpr std::int32_t kKvRows            = 1024;
constexpr std::int32_t kParentRows        = kQueryRows + kKvRows;
constexpr std::int32_t kGdnQkRows         = 4096;
constexpr std::int32_t kGdnValueRows      = 6144;
constexpr std::int32_t kGdnZRows          = 6144;
constexpr std::int32_t kGdnValueZRows     = kGdnValueRows + kGdnZRows;
constexpr std::int32_t kGdnRows           = kGdnQkRows + kGdnValueRows;
constexpr std::int32_t kGdnKeyRows        = 2048;
constexpr std::int32_t kGdnSlots          = 7;
constexpr std::int32_t kMaxBenchTokens    = 16384;
constexpr std::size_t kFlushBytes         = 256ULL << 20;
constexpr double kRtx5090DramGBs          = 1792.0;
constexpr double kRtx5090SustainedReadGBs = 1674.5;
constexpr double kRtx5090Bf16Tflops       = 209.5;

enum class OpSelection { All, Attention, Gdn };
enum class AttentionWeightType { Q4Q5, Bf16 };

struct Options {
    OpSelection op                            = OpSelection::All;
    AttentionWeightType attention_weight_type = AttentionWeightType::Q4Q5;
    bool profile                              = false;
    std::vector<std::int32_t> t_sweep{1,  2,  3,  4,  5,  6,  7,  8,   9,   10,
                                      11, 12, 13, 14, 15, 16, 17, 128, 129, 1024};
    int warmup = 5;
    int repeat = 50;
    std::string csv_out;
};

struct Result {
    std::string op;
    std::string path;
    std::int32_t t = 0;
    bench::ColdTiming timing;
};

std::vector<std::int32_t> parse_t_sweep(std::string_view raw) {
    std::vector<std::int32_t> result;
    std::size_t begin = 0;
    while (begin < raw.size()) {
        const std::size_t end = raw.find(',', begin);
        const std::string token(
            raw.substr(begin, end == std::string_view::npos ? raw.size() - begin : end - begin));
        if (token.empty()) { throw std::invalid_argument("empty --t-sweep element"); }
        const long value = std::stol(token);
        if (value <= 0 || value > std::numeric_limits<std::int32_t>::max()) {
            throw std::invalid_argument("--t-sweep values must be positive int32");
        }
        result.push_back(static_cast<std::int32_t>(value));
        if (end == std::string_view::npos) { break; }
        begin = end + 1;
    }
    if (result.empty()) { throw std::invalid_argument("--t-sweep must not be empty"); }
    return result;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view arg(argv[i]);
        const auto next = [&](const char* name) -> std::string_view {
            if (++i >= argc) { throw std::invalid_argument(std::string("missing ") + name); }
            return argv[i];
        };
        if (arg == "--op") {
            const std::string_view value = next("--op value");
            if (value == "all") {
                options.op = OpSelection::All;
            } else if (value == "attention") {
                options.op = OpSelection::Attention;
            } else if (value == "gdn") {
                options.op = OpSelection::Gdn;
            } else {
                throw std::invalid_argument("--op must be all, attention, or gdn");
            }
        } else if (arg == "--t-sweep") {
            options.t_sweep = parse_t_sweep(next("--t-sweep value"));
        } else if (arg == "--weight-type") {
            const std::string_view value = next("--weight-type value");
            if (value == "q4q5") {
                options.attention_weight_type = AttentionWeightType::Q4Q5;
            } else if (value == "bf16") {
                options.attention_weight_type = AttentionWeightType::Bf16;
            } else {
                throw std::invalid_argument("--weight-type must be q4q5 or bf16");
            }
        } else if (arg == "--profile") {
            options.profile = true;
        } else if (arg == "--warmup") {
            options.warmup = std::stoi(std::string(next("--warmup value")));
        } else if (arg == "--repeat") {
            options.repeat = std::stoi(std::string(next("--repeat value")));
        } else if (arg == "--csv-out") {
            options.csv_out = next("--csv-out path");
        } else if (arg == "--help" || arg == "-h") {
            std::printf("Usage: %s [--op all|attention|gdn] [--weight-type q4q5|bf16] "
                        "[--t-sweep 1,2,...] [--profile] [--warmup N] [--repeat N] "
                        "[--csv-out PATH]\n",
                        argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(arg));
        }
    }
    if (options.warmup < 0 || options.repeat <= 0) {
        throw std::invalid_argument("--warmup must be nonnegative and --repeat positive");
    }
    if (options.attention_weight_type == AttentionWeightType::Bf16) {
        if (options.op != OpSelection::Attention) {
            throw std::invalid_argument("BF16 weight mode requires --op attention");
        }
        if (options.t_sweep.size() != 1 || options.t_sweep.front() != 1) {
            throw std::invalid_argument("BF16 attention currently requires --t-sweep 1");
        }
    }
    if (options.profile) {
        if (options.attention_weight_type != AttentionWeightType::Bf16 ||
            options.op != OpSelection::Attention || options.t_sweep.size() != 1) {
            throw std::invalid_argument("--profile requires one BF16 Attention point");
        }
        if (!options.csv_out.empty()) {
            throw std::invalid_argument("--profile does not write timing CSV");
        }
    }
    return options;
}

void append_result(std::vector<Result>& results, std::string op, std::string path, std::int32_t t,
                   bench::ColdTiming timing) {
    std::printf("%-9s T=%-3d %-31s median=%8.3f us min=%8.3f us p95=%8.3f us\n", op.c_str(), t,
                path.c_str(), timing.median_us, timing.min_us, timing.p95_us);
    results.push_back({std::move(op), std::move(path), t, timing});
}

void write_csv(const std::string& path, const std::vector<Result>& results,
               const Options& options) {
    if (path.empty()) { return; }
    const std::filesystem::path output(path);
    if (!output.parent_path().empty()) {
        std::filesystem::create_directories(output.parent_path());
    }
    std::ofstream stream(output);
    if (!stream) { throw std::runtime_error("failed to open CSV: " + path); }
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    int runtime = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
    stream << "op,path,T,median_us,min_us,p95_us,warmup,repeat,build_type,gpu,cuda_runtime\n";
    for (const Result& result : results) {
        stream << result.op << ',' << result.path << ',' << result.t << ','
               << result.timing.median_us << ',' << result.timing.min_us << ','
               << result.timing.p95_us << ',' << options.warmup << ',' << options.repeat << ','
#ifdef NDEBUG
               << "Release"
#else
               << "Debug"
#endif
               << ',' << properties.name << ',' << runtime << '\n';
    }
}

struct Bf16AttentionResult {
    bench::ColdTiming timing;
    std::uint64_t weight_bytes = 0;
    std::uint64_t x_bytes      = 0;
    std::uint64_t out_bytes    = 0;
    std::uint64_t model_bytes  = 0;
    double useful_flops        = 0.0;
    double effective_gbs       = 0.0;
    double dram_spec_pct       = 0.0;
    double sustained_read_pct  = 0.0;
    double useful_tflops       = 0.0;
    double bf16_spec_pct       = 0.0;
    double memory_floor_us     = 0.0;
    double compute_floor_us    = 0.0;
    double roofline_floor_us   = 0.0;
    double roofline_pct        = 0.0;
    const char* bound          = "";
};

Bf16AttentionResult make_bf16_attention_result(const bench::ColdTiming& timing,
                                               std::uint64_t weight_bytes) {
    constexpr std::uint64_t kXBytes   = static_cast<std::uint64_t>(kHidden) * 2;
    constexpr std::uint64_t kOutBytes = static_cast<std::uint64_t>(14336) * 2;
    const std::uint64_t model_bytes   = weight_bytes + kXBytes + kOutBytes;
    const double useful_flops = 2.0 * static_cast<double>(14336) * static_cast<double>(kHidden);
    const double seconds      = timing.median_us * 1.0e-6;
    const double memory_floor_us =
        static_cast<double>(model_bytes) / (kRtx5090DramGBs * 1.0e9) * 1.0e6;
    const double compute_floor_us  = useful_flops / (kRtx5090Bf16Tflops * 1.0e12) * 1.0e6;
    const double roofline_floor_us = std::max(memory_floor_us, compute_floor_us);

    Bf16AttentionResult result;
    result.timing             = timing;
    result.weight_bytes       = weight_bytes;
    result.x_bytes            = kXBytes;
    result.out_bytes          = kOutBytes;
    result.model_bytes        = model_bytes;
    result.useful_flops       = useful_flops;
    result.effective_gbs      = static_cast<double>(model_bytes) / seconds / 1.0e9;
    result.dram_spec_pct      = result.effective_gbs / kRtx5090DramGBs * 100.0;
    result.sustained_read_pct = result.effective_gbs / kRtx5090SustainedReadGBs * 100.0;
    result.useful_tflops      = useful_flops / seconds / 1.0e12;
    result.bf16_spec_pct      = result.useful_tflops / kRtx5090Bf16Tflops * 100.0;
    result.memory_floor_us    = memory_floor_us;
    result.compute_floor_us   = compute_floor_us;
    result.roofline_floor_us  = roofline_floor_us;
    result.roofline_pct       = roofline_floor_us / timing.median_us * 100.0;
    result.bound              = memory_floor_us >= compute_floor_us ? "memory" : "compute";
    return result;
}

void print_bf16_attention_result(const Bf16AttentionResult& result) {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    std::printf("# actual_gpu=%s sm=%d%d reference_gpu=RTX_5090\n", properties.name,
                properties.major, properties.minor);
    std::printf("# dram_spec_gbs=%.1f sustained_read_gbs=%.1f bf16_dense_tc_spec_tflops=%.1f "
                "cache=cold\n",
                kRtx5090DramGBs, kRtx5090SustainedReadGBs, kRtx5090Bf16Tflops);
    std::printf("%-10s %-22s %5s %8s %8s %6s %11s %11s %11s %10s %7s %7s %10s %7s %7s %9s\n", "op",
                "path", "type", "N", "K", "T", "median_us", "min_us", "p95_us", "eff_GB/s",
                "DRAM_%", "READ_%", "TFLOP/s", "TC_%", "bound", "roof_%");
    std::printf("%-10s %-22s %5s %8d %8d %6d %11.3f %11.3f %11.3f %10.1f %7.2f "
                "%7.2f %10.2f %7.2f %7s %9.2f\n",
                "attention", "production_parent", "BF16", 14336, kHidden, 1,
                result.timing.median_us, result.timing.min_us, result.timing.p95_us,
                result.effective_gbs, result.dram_spec_pct, result.sustained_read_pct,
                result.useful_tflops, result.bf16_spec_pct, result.bound, result.roofline_pct);
}

void write_bf16_attention_csv(const std::string& path, const Bf16AttentionResult& result,
                              const Options& options) {
    if (path.empty()) { return; }
    const std::filesystem::path output(path);
    if (!output.parent_path().empty()) {
        std::filesystem::create_directories(output.parent_path());
    }
    std::ofstream stream(output);
    if (!stream) { throw std::runtime_error("failed to open CSV: " + path); }
    stream << "op,path,weight_type,N,K,T,weight_bytes,x_bytes,out_bytes,model_bytes,useful_flops,"
              "median_us,min_us,p95_us,effective_gbs,dram_spec_gbs,dram_spec_pct,"
              "sustained_read_gbs,sustained_read_pct,useful_tflops,"
              "bf16_dense_tc_spec_tflops,bf16_tc_spec_pct,memory_floor_us,compute_floor_us,"
              "roofline_floor_us,bound,roofline_pct,warmup,repeat,flush_bytes\n";
    stream << "attention,production_parent,BF16,14336," << kHidden << ",1," << result.weight_bytes
           << ',' << result.x_bytes << ',' << result.out_bytes << ',' << result.model_bytes << ','
           << result.useful_flops << ',' << result.timing.median_us << ',' << result.timing.min_us
           << ',' << result.timing.p95_us << ',' << result.effective_gbs << ',' << kRtx5090DramGBs
           << ',' << result.dram_spec_pct << ',' << kRtx5090SustainedReadGBs << ','
           << result.sustained_read_pct << ',' << result.useful_tflops << ',' << kRtx5090Bf16Tflops
           << ',' << result.bf16_spec_pct << ',' << result.memory_floor_us << ','
           << result.compute_floor_us << ',' << result.roofline_floor_us << ',' << result.bound
           << ',' << result.roofline_pct << ',' << options.warmup << ',' << options.repeat << ','
           << kFlushBytes << '\n';
}

void run_bf16_attention(const Options& options, DeviceBuffer& flush, cudaStream_t stream) {
    constexpr std::int32_t kRows   = 14336;
    bench::DirectBf16Weight weight = bench::make_direct_bf16_weight(kRows, kHidden, 0x61U);
    DeviceBuffer input             = bench::make_bf16(static_cast<std::size_t>(kHidden));
    DeviceBuffer query(static_cast<std::size_t>(kQueryRows) * 2);
    DeviceBuffer gate(static_cast<std::size_t>(kQueryRows) * 2);
    DeviceBuffer key(static_cast<std::size_t>(kKvRows) * 2);
    DeviceBuffer value(static_cast<std::size_t>(kKvRows) * 2);
    Tensor x(input.p, DType::BF16, {kHidden, 1});
    Tensor q(query.p, DType::BF16, {kQueryRows, 1});
    Tensor g(gate.p, DType::BF16, {kQueryRows, 1});
    Tensor k(key.p, DType::BF16, {kKvRows, 1});
    Tensor v(value.p, DType::BF16, {kKvRows, 1});
    const auto launch = [&](cudaStream_t launch_stream) {
        ops::attn_input_proj(x, weight.weight, q, g, k, v, launch_stream);
    };

    if (options.profile) {
        for (int iteration = 0; iteration < options.warmup; ++iteration) {
            bench::flush_l2(flush, stream);
            launch(stream);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        bench::flush_l2(flush, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const std::uint64_t model_bytes =
            weight.model_weight_bytes() + static_cast<std::uint64_t>(kHidden + kRows) * 2;
        const double useful_flops = 2.0 * static_cast<double>(kRows) * static_cast<double>(kHidden);
        std::printf("PROFILE attn_input_proj weight_type=BF16 N=%d K=%d T=1 model_bytes=%llu "
                    "useful_flops=%.0f\n",
                    kRows, kHidden, static_cast<unsigned long long>(model_bytes), useful_flops);
        std::fflush(stdout);
        CUDA_CHECK(cudaProfilerStart());
        launch(stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaProfilerStop());
        return;
    }

    const bench::ColdTiming timing =
        bench::measure_cold_launch(launch, flush, stream, options.warmup, options.repeat);
    const Bf16AttentionResult result =
        make_bf16_attention_result(timing, weight.model_weight_bytes());
    print_bf16_attention_result(result);
    write_bf16_attention_csv(options.csv_out, result, options);
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const std::int32_t max_t =
            *std::max_element(options.t_sweep.begin(), options.t_sweep.end());
        if (max_t > kMaxBenchTokens) {
            throw std::invalid_argument("input-projection benchmark allocation limit is T<=16384");
        }

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        ninfer::DeviceBuffer flush(kFlushBytes);
        if (options.attention_weight_type == AttentionWeightType::Bf16) {
            run_bf16_attention(options, flush, stream);
            CUDA_CHECK(cudaStreamDestroy(stream));
            return 0;
        }

        ninfer::DeviceBuffer input = bench::make_bf16(static_cast<std::size_t>(kHidden) * max_t);
        const std::size_t workspace_bytes =
            options.op == OpSelection::Attention
                ? 1
                : std::max<std::size_t>(1,
                                        ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                                            kGdnKeyRows, kGdnKeyRows, kGdnValueRows, 1, max_t));
        WorkspaceArena workspace(workspace_bytes);
        std::vector<Result> results;

        if (options.op != OpSelection::Gdn) {
            bench::PackedQuantizedWeight query_key = bench::make_row_split_weight(
                QType::Q4G64_F16S, kParentRows, kHidden, kHidden, {0x53, 0x53, 0x3c00});
            bench::PackedQuantizedWeight gate_value = bench::make_row_split_weight(
                QType::Q5G64_F16S, kParentRows, kHidden, kHidden, {0x53, 0x53, 0x3c00});
            const Weight query = bench::row_view(query_key.weight, 0, kQueryRows);
            const Weight key   = bench::row_view(query_key.weight, kQueryRows, kKvRows);
            const Weight gate  = bench::row_view(gate_value.weight, 0, kQueryRows);
            const Weight value = bench::row_view(gate_value.weight, kQueryRows, kKvRows);
            ninfer::DeviceBuffer q(static_cast<std::size_t>(kQueryRows) * max_t * 2);
            ninfer::DeviceBuffer g(static_cast<std::size_t>(kQueryRows) * max_t * 2);
            ninfer::DeviceBuffer k(static_cast<std::size_t>(kKvRows) * max_t * 2);
            ninfer::DeviceBuffer v(static_cast<std::size_t>(kKvRows) * max_t * 2);

            for (const std::int32_t t : options.t_sweep) {
                Tensor x(input.p, DType::BF16, {kHidden, t});
                Tensor tq(q.p, DType::BF16, {kQueryRows, t});
                Tensor tg(g.p, DType::BF16, {kQueryRows, t});
                Tensor tk(k.p, DType::BF16, {kKvRows, t});
                Tensor tv(v.p, DType::BF16, {kKvRows, t});
                const auto production = [&](cudaStream_t launch_stream) {
                    ops::attn_input_proj(x, query_key.weight, gate_value.weight, tq, tg, tk, tv,
                                         launch_stream);
                };
                append_result(results, "attention", "production_parent_split", t,
                              bench::measure_cold_launch(production, flush, stream, options.warmup,
                                                         options.repeat));
                if (t <= 16) {
                    const auto control = [&](cudaStream_t launch_stream) {
                        ops::linear(x, query, tq, launch_stream);
                        ops::linear(x, gate, tg, launch_stream);
                        ops::linear(x, key, tk, launch_stream);
                        ops::linear(x, value, tv, launch_stream);
                    };
                    append_result(results, "attention", "control_four_projection", t,
                                  bench::measure_cold_launch(control, flush, stream, options.warmup,
                                                             options.repeat));
                }
            }
        }

        if (options.op != OpSelection::Attention) {
            bench::PackedQuantizedWeight qk_weight = bench::make_row_split_weight(
                QType::Q4G64_F16S, kGdnQkRows, kHidden, kHidden, {0x53, 0x53, 0x3c00});
            bench::PackedQuantizedWeight value_z_weight = bench::make_row_split_weight(
                QType::Q5G64_F16S, kGdnValueZRows, kHidden, kHidden, {0x53, 0x53, 0x3c00});
            const Weight value_weight = bench::row_view(value_z_weight.weight, 0, kGdnValueRows);
            const Weight z_weight =
                bench::row_view(value_z_weight.weight, kGdnValueRows, kGdnZRows);
            ninfer::DeviceBuffer qkv(static_cast<std::size_t>(kGdnRows) * max_t * 2);
            ninfer::DeviceBuffer qkv_conv(static_cast<std::size_t>(kGdnRows) * max_t * 2);
            ninfer::DeviceBuffer qk_tmp(static_cast<std::size_t>(kGdnQkRows) * max_t * 2);
            ninfer::DeviceBuffer value_tmp(static_cast<std::size_t>(kGdnValueRows) * max_t * 2);
            ninfer::DeviceBuffer z_out(static_cast<std::size_t>(kGdnZRows) * max_t * 2);
            ninfer::DeviceBuffer query(static_cast<std::size_t>(kGdnKeyRows) * max_t * 2);
            ninfer::DeviceBuffer key(static_cast<std::size_t>(kGdnKeyRows) * max_t * 2);
            ninfer::DeviceBuffer value_out(static_cast<std::size_t>(kGdnValueRows) * max_t * 2);
            ninfer::DeviceBuffer conv_weight =
                bench::make_bf16(static_cast<std::size_t>(kGdnRows) * 4);
            ninfer::DeviceBuffer conv_states =
                bench::make_zeros(static_cast<std::size_t>(kGdnRows) * 3 * kGdnSlots * 2);
            ninfer::DeviceBuffer initial_slot(sizeof(std::int32_t));
            constexpr std::int32_t kInitialSlot = 6;
            CUDA_CHECK(cudaMemcpy(initial_slot.p, &kInitialSlot, sizeof(kInitialSlot),
                                  cudaMemcpyHostToDevice));

            for (const std::int32_t t : options.t_sweep) {
                Tensor x(input.p, DType::BF16, {kHidden, t});
                Tensor out(qkv.p, DType::BF16, {kGdnRows, t});
                Tensor convolved(qkv_conv.p, DType::BF16, {kGdnRows, t});
                Tensor qk(qk_tmp.p, DType::BF16, {kGdnQkRows, t});
                Tensor value(value_tmp.p, DType::BF16, {kGdnValueRows, t});
                Tensor z(z_out.p, DType::BF16, {kGdnZRows, t});
                Tensor tq(query.p, DType::BF16, {kGdnKeyRows, t});
                Tensor tk(key.p, DType::BF16, {kGdnKeyRows, t});
                Tensor tv(value_out.p, DType::BF16, {kGdnValueRows, t});
                Tensor conv_w(conv_weight.p, DType::BF16, {kGdnRows, 4});
                Tensor states(conv_states.p, DType::BF16, {kGdnRows, 3, kGdnSlots});
                Tensor initial(initial_slot.p, DType::I32, {1});
                const auto production = [&](cudaStream_t launch_stream) {
                    ops::gdn_input_proj(x, qk_weight.weight, value_z_weight.weight, out, z,
                                        launch_stream);
                };
                append_result(results, "gdn", "production_direct", t,
                              bench::measure_cold_launch(production, flush, stream, options.warmup,
                                                         options.repeat));
                if (t <= 6) {
                    const auto fused_snapshot = [&](cudaStream_t launch_stream) {
                        ops::gdn_input_proj_conv_snapshot(
                            x, qk_weight.weight, value_z_weight.weight, conv_w, states, initial, tq,
                            tk, tv, z, workspace, launch_stream);
                    };
                    append_result(results, "gdn", "fused_projection_conv_snapshot", t,
                                  bench::measure_cold_launch(fused_snapshot, flush, stream,
                                                             options.warmup, options.repeat));
                    const auto composed_snapshot = [&](cudaStream_t launch_stream) {
                        ops::gdn_input_proj(x, qk_weight.weight, value_z_weight.weight, out, z,
                                            launch_stream);
                        ops::causal_conv1d_silu_snapshot(out, conv_w, states, initial, convolved,
                                                         launch_stream);
                        ops::extract_bf16_columns(convolved, 0, tq, launch_stream);
                        ops::extract_bf16_columns(convolved, kGdnKeyRows, tk, launch_stream);
                        ops::extract_bf16_columns(convolved, 2 * kGdnKeyRows, tv, launch_stream);
                    };
                    append_result(results, "gdn", "composed_projection_conv_snapshot", t,
                                  bench::measure_cold_launch(composed_snapshot, flush, stream,
                                                             options.warmup, options.repeat));
                }
                if (t <= 16) {
                    const auto projections = [&](cudaStream_t launch_stream) {
                        ops::linear(x, qk_weight.weight, qk, launch_stream);
                        ops::linear(x, value_weight, value, launch_stream);
                        ops::linear(x, z_weight, z, launch_stream);
                    };
                    append_result(results, "gdn", "control_projection_only", t,
                                  bench::measure_cold_launch(projections, flush, stream,
                                                             options.warmup, options.repeat));
                    const auto materialize_copy = [&](cudaStream_t launch_stream) {
                        projections(launch_stream);
                        CUDA_CHECK(cudaMemcpy2DAsync(out.data, out.nb[1], qk.data, qk.nb[1],
                                                     static_cast<std::size_t>(kGdnQkRows) * 2, t,
                                                     cudaMemcpyDeviceToDevice, launch_stream));
                        CUDA_CHECK(cudaMemcpy2DAsync(static_cast<std::uint8_t*>(out.data) +
                                                         static_cast<std::size_t>(kGdnQkRows) * 2,
                                                     out.nb[1], value.data, value.nb[1],
                                                     static_cast<std::size_t>(kGdnValueRows) * 2, t,
                                                     cudaMemcpyDeviceToDevice, launch_stream));
                    };
                    append_result(results, "gdn", "control_materialize_copy", t,
                                  bench::measure_cold_launch(materialize_copy, flush, stream,
                                                             options.warmup, options.repeat));
                }
            }
        }

        CUDA_CHECK(cudaStreamSynchronize(stream));
        write_csv(options.csv_out, results, options);
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_input_proj_bench: %s\n", error.what());
        return 1;
    }
}
