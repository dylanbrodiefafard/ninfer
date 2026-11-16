// Public-Op benchmark for bf16_projection at the admitted Qwen3.8-Flash-Next shapes: BF16 weights
// [N,2560] for N in {96, 128, 512, 544, 2560, 10240}, BF16 x [2560,T], and the projection's
// fixed output dtype (BF16 for 128 and 512 rows, FP32 otherwise).
//
// One CUDA graph captures --layers consecutive calls (default 16), each over its own weight and
// output, as the model's layers do, and all over one x, which the model has just written; a
// 256 MiB L2 flush precedes every replay, so each weight is read cold while x stays L2-resident
// after the first call. The chain resolves call times below the event timer's granularity and
// includes the inter-kernel gap a graph pays between consecutive calls. Reported time is the
// replay median divided by the chain length. Bandwidth counts the contract's compulsory bytes once
// per call (the weight, x and y) against the 1674.5 GB/s sustained-read roof.

#include "ninfer/ops/bf16_projection.h"

#include "core/device.h"
#include "core/tensor.h"
#include "ninfer_bench_common.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;

namespace {

constexpr std::int32_t kInputRows = 2560;
constexpr std::size_t kFlushBytes = std::size_t{256} << 20;
constexpr double kDramRoofGBs     = 1674.5;

struct Options {
    std::vector<std::int32_t> rows{96, 128, 512, 544, 2560, 10240};
    std::vector<std::int32_t> tokens{1, 2, 3, 4, 8, 12, 16, 20, 24, 32, 40, 48};
    int layers = 16;
    int warmup = 5;
    int repeat = 41;
};

std::vector<std::int32_t> parse_list(std::string_view text, std::string_view what) {
    std::vector<std::int32_t> values;
    while (!text.empty()) {
        const std::size_t comma = text.find(',');
        values.push_back(bench::parse_number<std::int32_t>(text.substr(0, comma), what));
        if (values.back() < 1) {
            throw std::invalid_argument(std::string(what) + " values must be positive");
        }
        text = comma == std::string_view::npos ? std::string_view{} : text.substr(comma + 1);
    }
    if (values.empty()) { throw std::invalid_argument(std::string(what) + " must not be empty"); }
    return values;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto value = [&]() -> std::string_view {
            if (index + 1 >= argc) {
                throw std::invalid_argument(std::string(argument) + " requires a value");
            }
            return argv[++index];
        };
        if (argument == "--rows") {
            options.rows = parse_list(value(), "--rows");
        } else if (argument == "--tokens") {
            options.tokens = parse_list(value(), "--tokens");
        } else if (argument == "--layers") {
            options.layers = bench::parse_number<int>(value(), "--layers");
        } else if (argument == "--warmup") {
            options.warmup = bench::parse_number<int>(value(), "--warmup");
        } else if (argument == "--repeat") {
            options.repeat = bench::parse_number<int>(value(), "--repeat");
        } else {
            throw std::invalid_argument("unknown argument " + std::string(argument));
        }
    }
    if (options.warmup < 0 || options.repeat < 1 || options.layers < 1) {
        throw std::invalid_argument(
            "--warmup must be nonnegative and --repeat and --layers positive");
    }
    for (const std::int32_t rows : options.rows) {
        if (!ops::bf16_projection_admits(rows, kInputRows)) {
            throw std::invalid_argument("--rows " + std::to_string(rows) +
                                        " is not an admitted bf16_projection height");
        }
    }
    return options;
}

std::size_t dtype_bytes(DType dtype) { return dtype == DType::BF16 ? 2 : 4; }

void run_rows(const Options& options, std::int32_t rows, std::int32_t max_tokens,
              const DeviceBuffer& x, DeviceBuffer& flush, cudaStream_t stream) {
    const DType output_dtype          = ops::bf16_projection_output_dtype(rows);
    const std::size_t weight_elements = static_cast<std::size_t>(rows) * kInputRows;
    std::vector<DeviceBuffer> weights;
    std::vector<DeviceBuffer> outputs;
    weights.reserve(static_cast<std::size_t>(options.layers));
    outputs.reserve(static_cast<std::size_t>(options.layers));
    for (int layer = 0; layer < options.layers; ++layer) {
        weights.push_back(bench::make_bf16(weight_elements));
        outputs.emplace_back(static_cast<std::size_t>(rows) * max_tokens *
                             dtype_bytes(output_dtype));
    }
    for (const std::int32_t tokens : options.tokens) {
        const Tensor x_tensor(x.p, DType::BF16, {kInputRows, tokens});
        bench::TimedGraph graph;
        graph.capture(stream, [&](cudaStream_t capture_stream) {
            for (int layer = 0; layer < options.layers; ++layer) {
                const Tensor weight(weights[static_cast<std::size_t>(layer)].p, DType::BF16,
                                    {kInputRows, rows});
                Tensor y(outputs[static_cast<std::size_t>(layer)].p, output_dtype, {rows, tokens});
                ops::bf16_projection(x_tensor, weight, y, capture_stream);
            }
        });
        const bench::ColdTiming chain =
            bench::measure_cold_graph(graph, flush, stream, options.warmup, options.repeat);
        const double median_us = chain.median_us / options.layers;
        const double p95_us    = chain.p95_us / options.layers;
        const double bytes =
            static_cast<double>(weight_elements) * 2.0 +
            static_cast<double>(tokens) *
                (kInputRows * 2.0 + static_cast<double>(rows * dtype_bytes(output_dtype)));
        const double roof_us = bytes / (kDramRoofGBs * 1.0e3);
        const double gbs     = bytes / (median_us * 1.0e3);
        std::printf("N=%5d T=%4d  median=%8.3f us  p95=%8.3f us  roof=%7.3f us  %7.1f GB/s  "
                    "%6.2f%% of %.1f\n",
                    rows, tokens, median_us, p95_us, roof_us, gbs, gbs / kDramRoofGBs * 100.0,
                    kDramRoofGBs);
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
            std::printf("SKIP: no usable CUDA device\n");
            return 0;
        }
        const Options options   = parse_options(argc, argv);
        std::int32_t max_tokens = 1;
        for (const std::int32_t tokens : options.tokens) {
            max_tokens = tokens > max_tokens ? tokens : max_tokens;
        }
        const DeviceBuffer x = bench::make_bf16(static_cast<std::size_t>(max_tokens) * kInputRows);
        DeviceBuffer flush(kFlushBytes);
        // Graph capture needs a non-default stream; the context owns one.
        const DeviceContext context;

        std::printf("bf16_projection [N,2560], cold weights, graph of %d calls\n", options.layers);
        for (const std::int32_t rows : options.rows) {
            run_rows(options, rows, max_tokens, x, flush, context.stream);
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_bf16_projection_bench: %s\n", error.what());
        return 1;
    }
}
