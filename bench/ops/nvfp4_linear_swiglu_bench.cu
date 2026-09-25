#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rmsnorm_linear_swiglu.h"

#include "core/device.h"
#include "ninfer_bench_common.h"
#include "quantized_weight.cuh"

#include <cuda_profiler_api.h>
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
#include <vector>

using namespace ninfer;

namespace {

constexpr std::int32_t kGateUpRows = 34816;
constexpr std::int32_t kOutputRows = 17408;
constexpr std::int32_t kHidden     = 5120;
constexpr std::size_t kFlushBytes  = 256ULL << 20;

struct Options {
    ops::LinearPolicy policy = ops::LinearPolicy::A16Only;
    std::vector<std::int32_t> t_sweep{1, 4, 8, 16};
    int warmup   = 5;
    int repeat   = 30;
    bool profile = false;
    bool rmsnorm = false;
    bool rmsnorm_fused = false;
    std::int32_t panel_width = 0;
    std::string csv_out;
};

struct Result {
    std::int32_t tokens;
    bench::ColdTiming timing;
    double effective_gbs;
    double useful_tflops;
};

std::vector<std::int32_t> parse_t_sweep(std::string_view raw) {
    std::vector<std::int32_t> result;
    std::size_t begin = 0;
    while (begin < raw.size()) {
        const std::size_t end = raw.find(',', begin);
        const std::string token(
            raw.substr(begin, end == std::string_view::npos ? raw.size() - begin : end - begin));
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
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto next = [&](const char* label) -> std::string_view {
            if (++index >= argc) { throw std::invalid_argument(std::string("missing ") + label); }
            return argv[index];
        };
        if (argument == "--policy") {
            const std::string_view value = next("--policy");
            if (value == "a16") {
                options.policy  = ops::LinearPolicy::A16Only;
                options.t_sweep = {1, 4, 8, 16};
            } else if (value == "a4") {
                options.policy  = ops::LinearPolicy::AllowA4;
                options.t_sweep = {17, 128, 1024};
            } else if (value == "a8") {
                options.policy = ops::LinearPolicy::AllowA8;
                options.t_sweep = {4, 5, 6, 16, 20, 24};
            } else {
                throw std::invalid_argument("--policy must be a16, a4 or a8");
            }
        } else if (argument == "--t-sweep") {
            options.t_sweep = parse_t_sweep(next("--t-sweep"));
        } else if (argument == "--warmup") {
            options.warmup = std::stoi(std::string(next("--warmup")));
        } else if (argument == "--repeat") {
            options.repeat = std::stoi(std::string(next("--repeat")));
        } else if (argument == "--profile") {
            options.profile = true;
        } else if (argument == "--rmsnorm") {
            options.rmsnorm = true;
        } else if (argument == "--rmsnorm-fused") {
            options.rmsnorm_fused = true;
        } else if (argument == "--panel-width") {
            options.panel_width = std::stoi(std::string(next("--panel-width")));
        } else if (argument == "--csv-out") {
            options.csv_out = next("--csv-out");
        } else if (argument == "--help" || argument == "-h") {
            std::printf("Usage: %s --policy a16|a4|a8 [--rmsnorm|--rmsnorm-fused] [--t-sweep 1,4,...] [--panel-width N] "
                        "[--warmup N] [--repeat N] [--profile] [--csv-out PATH]\n",
                        argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + std::string(argument));
        }
    }
    if (options.warmup < 0 || options.repeat <= 0) {
        throw std::invalid_argument("--warmup must be nonnegative and --repeat positive");
    }
    if (options.rmsnorm_fused && (options.rmsnorm || options.policy != ops::LinearPolicy::AllowA8 || options.panel_width != 0)) {
        throw std::invalid_argument("--rmsnorm-fused requires a8 without panel splitting or --rmsnorm");
    }
    if (options.panel_width < 0) {
        throw std::invalid_argument("--panel-width must be nonnegative");
    }
    if (options.profile && options.t_sweep.size() != 1) {
        throw std::invalid_argument("--profile requires exactly one T");
    }
    if (options.policy == ops::LinearPolicy::A16Only &&
        *std::max_element(options.t_sweep.begin(), options.t_sweep.end()) > 20) {
        throw std::invalid_argument("A16 policy is registered only through T=20");
    }
    return options;
}

const char* policy_name(ops::LinearPolicy policy) {
    return policy == ops::LinearPolicy::AllowA8 ? "A8" : policy == ops::LinearPolicy::AllowA4 ? "A4" : "A16";
}

std::uint64_t modeled_bytes(const Options& options, std::int32_t tokens, std::uint64_t weight_bytes) {
    const std::int32_t reads = options.panel_width == 0 ? 1 : tokens / options.panel_width;
    const std::uint64_t normalization =
        (options.rmsnorm || options.rmsnorm_fused ? 2ULL * kHidden : 0) +
        (options.rmsnorm ? 4ULL * kHidden * tokens : 0);
    return weight_bytes * reads + 2ULL * (kHidden + kOutputRows) * tokens + normalization;
}

void write_csv(const Options& options, const std::vector<Result>& results,
               std::uint64_t weight_bytes) {
    if (options.csv_out.empty()) { return; }
    const std::filesystem::path path(options.csv_out);
    if (!path.parent_path().empty()) { std::filesystem::create_directories(path.parent_path()); }
    std::ofstream out(path);
    if (!out) { throw std::runtime_error("failed to open CSV: " + options.csv_out); }
    out << "op,weight_type,policy,N,K,T,panel_width,weight_reads,weight_bytes,modeled_bytes,"
           "median_us,min_us,p95_us,effective_gbs,useful_tflops,warmup,repeat,flush_bytes\n";
    for (const Result& result : results) {
        const std::int32_t weight_reads =
            options.panel_width == 0 ? 1 : result.tokens / options.panel_width;
        out << (options.rmsnorm_fused ? "rmsnorm_swiglu_fused" : options.rmsnorm ? "rmsnorm_swiglu" : options.panel_width ? "linear_swiglu_panels" : "linear_swiglu")
            << ",NVFP4," << policy_name(options.policy) << ',' << kGateUpRows << ','
            << kHidden << ',' << result.tokens << ',' << options.panel_width << ',' << weight_reads
            << ',' << weight_bytes << ',' << modeled_bytes(options, result.tokens, weight_bytes) << ','
            << result.timing.median_us << ',' << result.timing.min_us << ',' << result.timing.p95_us
            << ',' << result.effective_gbs << ',' << result.useful_tflops << ',' << options.warmup
            << ',' << options.repeat << ',' << kFlushBytes << '\n';
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const std::int32_t max_t =
            *std::max_element(options.t_sweep.begin(), options.t_sweep.end());
        const std::int32_t min_t =
            *std::min_element(options.t_sweep.begin(), options.t_sweep.end());
        if (options.panel_width > 0) {
            for (const std::int32_t tokens : options.t_sweep) {
                if (tokens % options.panel_width != 0) {
                    throw std::invalid_argument("every T must be divisible by --panel-width");
                }
            }
        }
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        DeviceBuffer flush(kFlushBytes);
        DeviceBuffer input = bench::make_bf16(static_cast<std::size_t>(kHidden) * max_t);
        DeviceBuffer norm_weight = bench::make_bf16(kHidden);
        DeviceBuffer normalized(static_cast<std::size_t>(kHidden) * max_t * sizeof(std::uint16_t));
        DeviceBuffer output(static_cast<std::size_t>(kOutputRows) * max_t * sizeof(std::uint16_t));
        bench::PackedQuantizedWeight packed  = bench::make_nvfp4_weight(kGateUpRows, kHidden);
        const std::size_t workspace_capacity = options.rmsnorm_fused
            ? ops::rmsnorm_linear_swiglu_workspace_capacity_bytes(max_t)
            : ops::linear_swiglu_workspace_capacity_bytes(
                QType::NVFP4, kGateUpRows, kHidden, options.policy, min_t, max_t);
        WorkspaceArena workspace(std::max<std::size_t>(workspace_capacity, 256));

        const auto make_launch = [&](std::int32_t tokens) {
            return [&, tokens](cudaStream_t launch_stream) {
                Tensor x(input.p, DType::BF16, {kHidden, tokens});
                if (options.rmsnorm) {
                    Tensor h(normalized.p, DType::BF16, {kHidden, tokens});
                    const Tensor norm(norm_weight.p, DType::BF16, {kHidden});
                    ops::rmsnorm(x, norm, 1e-6F, true, h, launch_stream);
                    x = h;
                }
                Tensor out(output.p, DType::BF16, {kOutputRows, tokens});
                if (options.rmsnorm_fused) {
                    const Tensor norm(norm_weight.p, DType::BF16, {kHidden});
                    ops::rmsnorm_linear_swiglu(x, norm, 1e-6F, packed.weight, out, workspace, launch_stream);
                    return;
                }
                if (options.panel_width == 0) {
                    ops::linear_swiglu(x, packed.weight, out, options.policy, workspace,
                                       launch_stream);
                    return;
                }
                for (std::int32_t offset = 0; offset < tokens; offset += options.panel_width) {
                    Tensor panel_x = x.slice(1, offset, options.panel_width);
                    Tensor panel_out = out.slice(1, offset, options.panel_width);
                    ops::linear_swiglu(panel_x, packed.weight, panel_out, options.policy, workspace,
                                       launch_stream);
                }
            };
        };

        if (options.profile) {
            const std::int32_t tokens = options.t_sweep.front();
            const auto launch         = make_launch(tokens);
            for (int iteration = 0; iteration < options.warmup; ++iteration) {
                bench::flush_l2(flush, stream);
                launch(stream);
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));
            bench::flush_l2(flush, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            std::printf("PROFILE linear_swiglu weight_type=NVFP4 policy=%s T=%d\n",
                        policy_name(options.policy), tokens);
            std::fflush(stdout);
            CUDA_CHECK(cudaProfilerStart());
            launch(stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaProfilerStop());
            CUDA_CHECK(cudaStreamDestroy(stream));
            return 0;
        }

        std::vector<Result> results;
        results.reserve(options.t_sweep.size());
        std::printf("%-14s %3s %8s %8s %6s %11s %11s %11s %10s %10s\n", "op", "pol", "N", "K", "T",
                    "median_us", "min_us", "p95_us", "eff_GB/s", "TFLOP/s");
        for (const std::int32_t tokens : options.t_sweep) {
            const auto launch = make_launch(tokens);
            const bench::ColdTiming timing =
                bench::measure_cold_launch(launch, flush, stream, options.warmup, options.repeat);
            const double seconds      = timing.median_us * 1.0e-6;
            const double useful_flops = 2.0 * static_cast<double>(kGateUpRows) * kHidden * tokens;
            const double model_bytes = static_cast<double>(modeled_bytes(options, tokens, packed.model_weight_bytes()));
            const double useful_tflops = useful_flops / seconds / 1.0e12;
            const double effective_gbs = model_bytes / seconds / 1.0e9;
            std::printf("%-14s %3s %8d %8d %6d %11.3f %11.3f %11.3f %10.1f %10.2f\n",
                        options.rmsnorm_fused ? "rmsnorm_fused" : options.rmsnorm ? "rmsnorm_swiglu" : options.panel_width ? "swiglu_panels" : "linear_swiglu",
                        policy_name(options.policy), kGateUpRows, kHidden, tokens,
                        timing.median_us, timing.min_us, timing.p95_us, effective_gbs,
                        useful_tflops);
            results.push_back({tokens, timing, effective_gbs, useful_tflops});
        }
        write_csv(options, results, packed.model_weight_bytes());
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_nvfp4_linear_swiglu_bench: %s\n", error.what());
        return 1;
    }
}
