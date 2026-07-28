// Fixed-shape production/control benchmark for the 27B Attention and GDN input projections.
// Controls reproduce only the superseded Small-T compositions and are intentionally benchmark-
// local: production dispatch has no fallback to them.

#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/scatter.h"

#include "core/device.h"
#include "ninfer_bench_common.h"
#include "quantized_weight.cuh"

#include <cuda_runtime.h>

#include <algorithm>
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

constexpr std::int32_t kHidden         = 5120;
constexpr std::int32_t kQueryRows      = 6144;
constexpr std::int32_t kKvRows         = 1024;
constexpr std::int32_t kParentRows     = kQueryRows + kKvRows;
constexpr std::int32_t kGdnQkRows      = 4096;
constexpr std::int32_t kGdnValueRows   = 6144;
constexpr std::int32_t kGdnRows        = kGdnQkRows + kGdnValueRows;
constexpr std::int32_t kGdnKeyRows     = 2048;
constexpr std::int32_t kGdnSlots       = 7;
constexpr std::int32_t kMaxBenchTokens = 16384;
constexpr std::size_t kFlushBytes      = 256ULL << 20;

enum class OpSelection { All, Attention, Gdn };

struct Options {
    OpSelection op = OpSelection::All;
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
        } else if (arg == "--warmup") {
            options.warmup = std::stoi(std::string(next("--warmup value")));
        } else if (arg == "--repeat") {
            options.repeat = std::stoi(std::string(next("--repeat value")));
        } else if (arg == "--csv-out") {
            options.csv_out = next("--csv-out path");
        } else if (arg == "--help" || arg == "-h") {
            std::printf("Usage: %s [--op all|attention|gdn] [--t-sweep 1,2,...] "
                        "[--warmup N] [--repeat N] [--csv-out PATH]\n",
                        argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(arg));
        }
    }
    if (options.warmup < 0 || options.repeat <= 0) {
        throw std::invalid_argument("--warmup must be nonnegative and --repeat positive");
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
            bench::PackedQuantizedWeight value_weight = bench::make_row_split_weight(
                QType::Q5G64_F16S, kGdnValueRows, kHidden, kHidden, {0x53, 0x53, 0x3c00});
            ninfer::DeviceBuffer qkv(static_cast<std::size_t>(kGdnRows) * max_t * 2);
            ninfer::DeviceBuffer qkv_conv(static_cast<std::size_t>(kGdnRows) * max_t * 2);
            ninfer::DeviceBuffer qk_tmp(static_cast<std::size_t>(kGdnQkRows) * max_t * 2);
            ninfer::DeviceBuffer value_tmp(static_cast<std::size_t>(kGdnValueRows) * max_t * 2);
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
                Tensor tq(query.p, DType::BF16, {kGdnKeyRows, t});
                Tensor tk(key.p, DType::BF16, {kGdnKeyRows, t});
                Tensor tv(value_out.p, DType::BF16, {kGdnValueRows, t});
                Tensor conv_w(conv_weight.p, DType::BF16, {kGdnRows, 4});
                Tensor states(conv_states.p, DType::BF16, {kGdnRows, 3, kGdnSlots});
                Tensor initial(initial_slot.p, DType::I32, {1});
                const auto production = [&](cudaStream_t launch_stream) {
                    ops::gdn_input_proj(x, qk_weight.weight, value_weight.weight, out,
                                        launch_stream);
                };
                append_result(results, "gdn", "production_direct", t,
                              bench::measure_cold_launch(production, flush, stream, options.warmup,
                                                         options.repeat));
                if (t <= 6) {
                    const auto fused_snapshot = [&](cudaStream_t launch_stream) {
                        ops::gdn_input_proj_conv_snapshot(x, qk_weight.weight, value_weight.weight,
                                                          conv_w, states, initial, tq, tk, tv,
                                                          workspace, launch_stream);
                    };
                    append_result(results, "gdn", "fused_projection_conv_snapshot", t,
                                  bench::measure_cold_launch(fused_snapshot, flush, stream,
                                                             options.warmup, options.repeat));
                    const auto composed_snapshot = [&](cudaStream_t launch_stream) {
                        ops::gdn_input_proj(x, qk_weight.weight, value_weight.weight, out,
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
                        ops::linear(x, value_weight.weight, value, launch_stream);
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
