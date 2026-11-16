// Public stream_mean benchmark: the four-stream hyper-connection mean written into one 2560-row
// slot of a [12800,T] feature matrix, timed as a captured graph with warm and flushed L2.
#include "ninfer/ops/stream_mean.h"
#include "ninfer_bench_common.h"

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {
using namespace ninfer;

constexpr double kSustainedReadGbs = 1674.5;

void run(int tokens, int repeat) {
    constexpr int kHidden   = ops::kStreamMeanHidden;
    constexpr int kStreams  = ops::kStreamMeanStreams;
    constexpr int kFeatures = 5 * kHidden;
    std::vector<std::uint16_t> residual_bits(static_cast<std::size_t>(kHidden) * kStreams * tokens);
    for (std::size_t i = 0; i < residual_bits.size(); ++i) {
        residual_bits[i] = bench::f32_to_bf16(static_cast<float>(i % 97) * 0.03125F - 1.5F);
    }
    DeviceBuffer residual(residual_bits.size() * sizeof(std::uint16_t));
    residual.copy_from_host(residual_bits.data(), residual.bytes);
    DeviceBuffer features(static_cast<std::size_t>(kFeatures) * tokens * sizeof(std::uint16_t));
    DeviceBuffer flush(256ULL << 20);
    Tensor residual_t(residual.p, DType::BF16, {kHidden, kStreams, tokens});
    Tensor matrix(features.p, DType::BF16, {kFeatures, tokens});
    Tensor out_t = matrix.slice(0, 2 * kHidden, kHidden);
    DeviceContext context;
    CUDA_CHECK(cudaDeviceSynchronize());
    bench::TimedGraph graph;
    graph.capture(context.stream,
                  [&](cudaStream_t stream) { ops::stream_mean(residual_t, out_t, stream); });
    const auto warm = bench::measure_graph(graph, context.stream, 20, repeat);
    const auto cold = bench::measure_cold_graph(graph, flush, context.stream, 5, repeat);
    // Model bytes: four BF16 stream reads and one BF16 write per channel and token.
    const double bytes   = static_cast<double>(tokens) * kHidden * (kStreams + 1) * 2.0;
    const double roof_us = bytes / (kSustainedReadGbs * 1.0e3);
    std::printf("stream_mean T=%d bytes=%.0f roof_us=%.3f warm_median_us=%.3f warm_min_us=%.3f "
                "cold_median_us=%.3f cold_GBs=%.1f\n",
                tokens, bytes, roof_us, warm.median_us, warm.min_us, cold.median_us,
                bytes / (cold.median_us * 1.0e3));
}
} // namespace

int main(int argc, char** argv) {
    try {
        int tokens = 8;
        int repeat = 200;
        for (int index = 1; index < argc; ++index) {
            const std::string_view argument(argv[index]);
            if (argument == "--help") {
                std::printf("usage: %s [--tokens T] [--repeat N]\n", argv[0]);
                return 0;
            }
            if (index + 1 == argc) { throw std::invalid_argument("missing option value"); }
            const int value = bench::parse_number<int>(argv[++index], argument);
            if (argument == "--tokens") {
                tokens = value;
            } else if (argument == "--repeat") {
                repeat = value;
            } else {
                throw std::invalid_argument("unknown option");
            }
        }
        if (tokens < 1 || tokens > 65536 || repeat < 1 || repeat > 100000) {
            throw std::invalid_argument("requires T=1..65536, repeat=1..100000");
        }
        run(tokens, repeat);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
