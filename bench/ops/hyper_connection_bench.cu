// Public-Op benchmark for hyper_connection_read at the Qwen3.8-Flash-Next geometry: residual
// BF16 [2560,4,T], BF16 norm [10240] and write [4,10240], and FP8_E4M3FN_ROW_BF16S down
// [320,10240] and up [10240,320] projections. --final drops the write projection, as the model's
// final read does. --inject follows each read with an in-place inject of a fixed independent
// BF16 block output, measuring microseconds per read/inject pair and its next-read dependency.
//
// One CUDA graph captures --layers consecutive Op calls (default 16), each over its own weight set
// (16 x 6.7 MB exceeds the L2), as the model's layers do; a 256 MiB L2 flush precedes every
// replay. A chain resolves single-read times below the event timer's granularity; consecutive
// reads may overlap at their launch boundaries as reads separated by other Ops cannot, so the
// read-only figures are a lower bound for an isolated read. Reported time is the replay median
// divided by the chain length. Bandwidth counts the contract's compulsory bytes once per read
// (every weight plane, the residual read, x and the write scales written) against the 1674.5 GB/s
// sustained-read roof; --inject also counts the inject's residual read/write, block read and scale
// read once. The FP32 workspace round trip is private and is not counted.

#include "ninfer/ops/hyper_connection.h"

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "ninfer_bench_common.h"
#include "quantized_weight.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>
#include <string_view>
#include <vector>

using namespace ninfer;

namespace {

constexpr std::int32_t kHidden    = ops::kHyperConnectionHidden;
constexpr std::int32_t kStreams   = ops::kHyperConnectionStreams;
constexpr std::int32_t kFlat      = ops::kHyperConnectionFlat;
constexpr std::int32_t kRank      = ops::kHyperConnectionRank;
constexpr std::size_t kFlushBytes = std::size_t{256} << 20;
constexpr double kDramRoofGBs     = 1674.5;

struct Options {
    std::vector<std::int32_t> tokens{1, 12, 512, 4096};
    bool with_write = true;
    bool inject     = false;
    int layers      = 16;
    int warmup      = 5;
    int repeat      = 41;
};

std::vector<std::int32_t> parse_list(std::string_view text) {
    std::vector<std::int32_t> values;
    while (!text.empty()) {
        const std::size_t comma = text.find(',');
        values.push_back(bench::parse_number<std::int32_t>(text.substr(0, comma), "--tokens"));
        if (values.back() < 1) { throw std::invalid_argument("--tokens values must be positive"); }
        text = comma == std::string_view::npos ? std::string_view{} : text.substr(comma + 1);
    }
    if (values.empty()) { throw std::invalid_argument("--tokens must not be empty"); }
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
        if (argument == "--tokens") {
            options.tokens = parse_list(value());
        } else if (argument == "--layers") {
            options.layers = bench::parse_number<int>(value(), "--layers");
        } else if (argument == "--inject") {
            options.inject = true;
        } else if (argument == "--final") {
            options.with_write = false;
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
    if (options.inject && !options.with_write) {
        throw std::invalid_argument("--inject requires the write projection");
    }
    return options;
}

struct Fixture {
    DeviceBuffer norm                 = bench::make_bf16(kFlat);
    bench::PackedQuantizedWeight down = bench::make_fp8_weight(kRank, kFlat);
    bench::PackedQuantizedWeight up   = bench::make_fp8_weight(kFlat, kRank);
    DeviceBuffer write = bench::make_bf16(static_cast<std::size_t>(kStreams) * kFlat);

    [[nodiscard]] ops::HyperConnectionWeights weights(bool with_write) const {
        return {Tensor(norm.p, DType::BF16, {kFlat}), down.weight, up.weight,
                with_write ? Tensor(write.p, DType::BF16, {kFlat, kStreams}) : Tensor{}};
    }

    // Code and multiplier planes once, without the payload's alignment padding.
    [[nodiscard]] double weight_bytes(bool with_write) const {
        return static_cast<double>(norm.bytes + down.model_weight_bytes() +
                                   up.model_weight_bytes() + (with_write ? write.bytes : 0));
    }
};

} // namespace

int main(int argc, char** argv) {
    try {
        int count = 0;
        if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
            std::printf("SKIP: no usable CUDA device\n");
            return 0;
        }
        const Options options = parse_options(argc, argv);
        const std::int32_t max_tokens =
            *std::max_element(options.tokens.begin(), options.tokens.end());

        const std::vector<Fixture> fixtures(static_cast<std::size_t>(options.layers));
        DeviceBuffer residual    = bench::make_bf16(static_cast<std::size_t>(max_tokens) * kFlat);
        const DeviceBuffer block = bench::make_bf16(static_cast<std::size_t>(max_tokens) * kHidden);
        DeviceBuffer x(static_cast<std::size_t>(max_tokens) * kHidden * sizeof(std::uint16_t));
        DeviceBuffer scale(static_cast<std::size_t>(max_tokens) * kStreams * sizeof(float));
        DeviceBuffer flush(kFlushBytes);
        WorkspaceArena workspace(ops::hyper_connection_read_workspace_bytes(max_tokens));
        // Graph capture needs a non-default stream.
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        std::printf("hyper_connection_read %s%s, cold L2, graph of %d reads\n",
                    options.with_write ? "with write" : "final (no write)",
                    options.inject ? " + inject" : "", options.layers);
        for (const std::int32_t tokens : options.tokens) {
            Tensor residual_tensor(residual.p, DType::BF16, {kHidden, kStreams, tokens});
            Tensor x_tensor(x.p, DType::BF16, {kHidden, tokens});
            Tensor scale_tensor(scale.p, DType::FP32, {kStreams, tokens});
            Tensor block_tensor(block.p, DType::BF16, {kHidden, tokens});
            bench::TimedGraph graph;
            graph.capture(stream, [&](cudaStream_t capture_stream) {
                for (const Fixture& fixture : fixtures) {
                    ops::hyper_connection_read(residual_tensor, fixture.weights(options.with_write),
                                               x_tensor, scale_tensor, workspace, capture_stream);
                    if (options.inject) {
                        ops::hyper_connection_inject(residual_tensor, block_tensor, scale_tensor,
                                                     residual_tensor, capture_stream);
                    }
                }
            });
            const bench::ColdTiming chain =
                bench::measure_cold_graph(graph, flush, stream, options.warmup, options.repeat);
            const double median_us = chain.median_us / options.layers;
            const double p95_us    = chain.p95_us / options.layers;
            const double activation_bytes =
                static_cast<double>(tokens) *
                (kFlat * 2.0 + kHidden * 2.0 + (options.with_write ? kStreams * 4.0 : 0.0));
            const double inject_bytes =
                options.inject
                    ? static_cast<double>(tokens) * (kFlat * 4.0 + kHidden * 2.0 + kStreams * 4.0)
                    : 0.0;
            const double bytes =
                fixtures.front().weight_bytes(options.with_write) + activation_bytes + inject_bytes;
            const double gbs = bytes / (median_us * 1.0e-6) / 1.0e9;
            std::printf("T=%5d  median=%9.3f us  p95=%9.3f us  %8.1f GB/s  %6.2f%% of %.1f\n",
                        tokens, median_us, p95_us, gbs, gbs / kDramRoofGBs * 100.0, kDramRoofGBs);
        }
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_hyper_connection_bench: %s\n", error.what());
        return 1;
    }
}
