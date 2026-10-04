// Measures the public two-output Linear projection with original immutable weight descriptors.

#include "ninfer/ops/linear.h"

#include "core/device.h"
#include "ninfer_bench_common.h"
#include "quantized_weight.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
using namespace ninfer;

constexpr std::int32_t kHidden     = 5120;
constexpr std::int32_t kParentRows = 6144;
constexpr std::int32_t kKvRows     = 1024;

struct Options {
    std::vector<std::int32_t> widths{1, 2, 3, 4, 5, 6, 7, 8, 128, 512, 2048};
    std::int32_t batch = 1;
    int warmup         = 10;
    int repeat         = 50;
    int layers         = 1;
    bool graph         = true;
    bool cold          = false;
};

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        const auto next = [&]() -> std::string_view {
            if (++index == argc) { throw std::invalid_argument("missing option value"); }
            return argv[index];
        };
        if (argument == "--widths") {
            options.widths.clear();
            std::string_view text = next();
            while (!text.empty()) {
                const auto end   = text.find(',');
                const auto width = bench::parse_number<std::int32_t>(text.substr(0, end), "width");
                if (width <= 0 || width > 2048) {
                    throw std::invalid_argument("width must be in [1,2048]");
                }
                options.widths.push_back(width);
                if (end == std::string_view::npos) { break; }
                text.remove_prefix(end + 1);
                if (text.empty()) { throw std::invalid_argument("empty width"); }
            }
            if (options.widths.empty()) { throw std::invalid_argument("empty widths"); }
        } else if (argument == "--batch") {
            options.batch = bench::parse_number<std::int32_t>(next(), "batch");
        } else if (argument == "--warmup") {
            options.warmup = bench::parse_number<int>(next(), "warmup");
        } else if (argument == "--repeat") {
            options.repeat = bench::parse_number<int>(next(), "repeat");
        } else if (argument == "--layers") {
            options.layers = bench::parse_number<int>(next(), "layers");
        } else if (argument == "--mode") {
            const auto mode = next();
            if (mode != "graph" && mode != "eager") {
                throw std::invalid_argument("mode must be graph or eager");
            }
            options.graph = mode == "graph";
        } else if (argument == "--cold") {
            options.cold = true;
        } else if (argument == "--help") {
            std::printf("Usage: %s [--widths 1,2,3,4,5,6,7,8,128,512,2048] [--batch 1..6] "
                        "[--mode graph|eager] [--cold] [--layers 1|5] [--warmup N] [--repeat N]\n",
                        argv[0]);
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown option: " + std::string(argument));
        }
    }
    if (options.batch < 1 || options.batch > 6 || options.warmup < 0 || options.repeat <= 0 ||
        (options.layers != 1 && options.layers != 5)) {
        throw std::invalid_argument("invalid batch, layer or repetition count");
    }
    return options;
}
} // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        const auto max_tokens =
            *std::max_element(options.widths.begin(), options.widths.end()) * options.batch;
        DeviceContext device;
        DeviceBuffer input = bench::make_bf16(static_cast<std::size_t>(kHidden) * max_tokens);
        DeviceBuffer key_storage(static_cast<std::size_t>(kKvRows) * max_tokens * 2);
        DeviceBuffer value_storage(static_cast<std::size_t>(kKvRows) * max_tokens * 2);
        DeviceBuffer flush(options.cold ? 256ULL << 20 : 0);
        std::vector<bench::PackedQuantizedWeight> parents;
        parents.reserve(static_cast<std::size_t>(options.layers));
        for (int layer = 0; layer < options.layers; ++layer) {
            parents.push_back(bench::make_nvfp4_weight(kParentRows, kHidden));
        }
        std::printf("# gpu=%s public=linear_kv_projection original_weight=[6144,5120] "
                    "outputs=two_[1024,T] policy=a16 mode=%s cache=%s layers=%d\n",
                    device.props.name, options.graph ? "graph" : "eager",
                    options.cold ? "cold" : "warm", options.layers);
        std::printf("width,batch,t,median_us,min_us,p95_us,nodes\n");
        for (const auto width : options.widths) {
            const auto tokens = width * options.batch;
            Tensor x(input.p, DType::BF16, {kHidden, tokens});
            Tensor key(key_storage.p, DType::BF16, {kKvRows, tokens});
            Tensor value(value_storage.p, DType::BF16, {kKvRows, tokens});
            const auto launch = [&](cudaStream_t stream) {
                for (const auto& parent : parents) {
                    ops::linear_kv_projection(x, parent.weight, key, value, stream, width);
                }
            };
            launch(device.stream);
            device.synchronize();
            bench::TimedGraph graph;
            if (options.graph) { graph.capture(device.stream, launch); }
            const auto timed_launch = [&](cudaStream_t stream) {
                if (options.graph) {
                    graph.launch(stream);
                } else {
                    launch(stream);
                }
            };
            const auto timing = options.cold
                                    ? bench::measure_cold_launch(timed_launch, flush, device.stream,
                                                                 options.warmup, options.repeat)
                                    : bench::measure_launch(timed_launch, device.stream,
                                                            options.warmup, options.repeat);
            std::printf("%d,%d,%d,%.6f,%.6f,%.6f,%zu\n", width, options.batch, tokens,
                        timing.median_us, timing.min_us, timing.p95_us,
                        options.graph ? graph.nodes() : 0);
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_linear_kv_projection_bench: %s\n", error.what());
        return 1;
    }
}
