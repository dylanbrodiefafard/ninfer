// Closed MTP stem normalization boundary: two independent BF16 [5120,T] panels.
#include "ninfer/ops/rmsnorm.h"
#include "ninfer_bench_common.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string_view>

namespace {

struct Options {
    int tokens = 0;
    int warmup = 50;
    int repeat = 200;
};

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view option(argv[index]);
        if (option == "--help") {
            std::puts("ninfer_dual_rmsnorm_bench [--tokens T] [--warmup N] [--repeat N]\n"
                      "Measures the public dual offset RMSNorm Op, eager and CUDA Graph replay.\n"
                      "Default extents cover B1/2/6 with packed widths 1..6 (T up to 36).\n"
                      "Inputs and outputs remain resident; timing uses warmed CUDA events.");
            std::exit(0);
        }
        if (index + 1 == argc) { throw std::invalid_argument("missing option value"); }
        const int value = ninfer::bench::parse_number<int>(argv[++index], option);
        if (option == "--tokens") {
            options.tokens = value;
            if (value <= 0) { throw std::invalid_argument("--tokens must be positive"); }
        } else if (option == "--warmup") {
            options.warmup = value;
        } else if (option == "--repeat") {
            options.repeat = value;
        } else {
            throw std::invalid_argument("unknown option");
        }
    }
    if (options.warmup < 0 || options.repeat <= 0) {
        throw std::invalid_argument("warmup must be nonnegative and repeat positive");
    }
    return options;
}

void run_case(int tokens, const Options& options, const ninfer::DeviceContext& context) {
    constexpr int kWidth = 5120;
    const auto elements  = static_cast<std::size_t>(tokens) * kWidth;
    auto input0          = ninfer::bench::make_bf16(elements);
    auto input1          = ninfer::bench::make_bf16(elements);
    auto weight0         = ninfer::bench::make_bf16(kWidth);
    auto weight1         = ninfer::bench::make_bf16(kWidth);
    ninfer::DeviceBuffer output0(elements * sizeof(std::uint16_t));
    ninfer::DeviceBuffer output1(elements * sizeof(std::uint16_t));
    ninfer::Tensor x0(input0.p, ninfer::DType::BF16, {kWidth, tokens});
    ninfer::Tensor x1(input1.p, ninfer::DType::BF16, {kWidth, tokens});
    ninfer::Tensor w0(weight0.p, ninfer::DType::BF16, {kWidth});
    ninfer::Tensor w1(weight1.p, ninfer::DType::BF16, {kWidth});
    ninfer::Tensor y0(output0.p, ninfer::DType::BF16, {kWidth, tokens});
    ninfer::Tensor y1(output1.p, ninfer::DType::BF16, {kWidth, tokens});
    const auto launch = [&](cudaStream_t stream) {
        ninfer::ops::dual_offset_rmsnorm(x0, w0, x1, w1, 1.0e-6F, y0, y1, stream);
    };
    const auto eager =
        ninfer::bench::measure_launch(launch, context.stream, options.warmup, options.repeat);
    ninfer::bench::TimedGraph graph;
    graph.capture(context.stream, launch);
    const auto replay =
        ninfer::bench::measure_graph(graph, context.stream, options.warmup, options.repeat);
    std::printf("route=dual_norm T=%d mode=eager kernels=1 median_us=%.6f min_us=%.6f "
                "p95_us=%.6f\n",
                tokens, eager.median_us, eager.min_us, eager.p95_us);
    std::printf("route=dual_norm T=%d mode=graph nodes=%zu median_us=%.6f min_us=%.6f "
                "p95_us=%.6f\n",
                tokens, graph.nodes(), replay.median_us, replay.min_us, replay.p95_us);
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const ninfer::DeviceContext context;
        if (options.tokens > 0) {
            run_case(options.tokens, options, context);
        } else {
            constexpr std::array extents{1, 2, 3, 4, 5, 6, 8, 10, 12, 18, 24, 30, 36};
            for (const int tokens : extents) { run_case(tokens, options, context); }
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "dual RMSNorm benchmark: %s\n", error.what());
        return 1;
    }
}
