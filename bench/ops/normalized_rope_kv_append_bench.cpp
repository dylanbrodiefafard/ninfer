#include "ninfer_bench_common.h"
#include "ops/normalized_rope_kv_append_reference.h"

#include <array>
#include <cstdio>
#include <exception>
#include <string_view>
#include <vector>

using namespace ninfer;
namespace ref = ninfer::test::normalized_append;

int main(int argc, char** argv) {
    try {
        int warmup         = 20;
        int repeat         = 100;
        int selected_batch = 0;
        int selected_width = 0;
        int selected_count = -2;
        for (int i = 1; i < argc; ++i) {
            const std::string_view option(argv[i]);
            if ((option == "--warmup" || option == "--repeat" || option == "--batch" ||
                 option == "--width" || option == "--count") &&
                i + 1 < argc) {
                const int value = bench::parse_number<int>(argv[++i], option);
                if (option == "--warmup") {
                    warmup = value;
                } else if (option == "--repeat") {
                    repeat = value;
                } else if (option == "--batch") {
                    if (value < 1 || value > 6) {
                        throw std::invalid_argument("batch must be 1..6");
                    }
                    selected_batch = value;
                } else if (option == "--width") {
                    if (value < 1 || value > ref::kCapacity) {
                        throw std::invalid_argument("width must be 1..2048");
                    }
                    selected_width = value;
                } else {
                    if (value < -1) {
                        throw std::invalid_argument("count must be -1 or nonnegative");
                    }
                    selected_count = value;
                }
            } else if (option == "--help") {
                std::puts(
                    "normalized_rope_kv_append_bench [--warmup N] [--repeat N] "
                    "[--batch 1..6] [--width 1..2048] [--count -1..W]\n"
                    "Defaults: B=1,2,6; W=3..8; counts=0,1,mixed,full. Count -1 selects mixed.");
                return 0;
            } else {
                throw std::invalid_argument("unknown or incomplete option");
            }
        }
        if (warmup < 0 || repeat <= 0) {
            throw std::invalid_argument("warmup must be nonnegative and repeat positive");
        }
        if (test::require_cuda() != 0) { return 1; }
        DeviceContext device;
        const cudaStream_t stream = device.stream;
        std::puts("B,W,C,execution,median_us,min_us,p95_us,graph_nodes");
        const std::vector<int> batches =
            selected_batch == 0 ? std::vector<int>{1, 2, 6} : std::vector<int>{selected_batch};
        const int first_width = selected_width == 0 ? 3 : selected_width;
        const int last_width  = selected_width == 0 ? 8 : selected_width;
        for (const int batch : batches) {
            for (int tokens = first_width; tokens <= last_width; ++tokens) {
                ref::Fixture data(tokens, batch, 4093);
                bench::TimedGraph graph;
                graph.capture(stream, [&](cudaStream_t s) { data.launch(s); });
                const std::vector<int> counts = selected_count == -2
                                                    ? std::vector<int>{0, 1, -1, tokens}
                                                    : std::vector<int>{selected_count};
                for (const int count : counts) {
                    if (count > tokens) { throw std::invalid_argument("count exceeds width"); }
                    data.reset(count, false);
                    for (const bool captured : std::array{false, true}) {
                        const auto timing =
                            captured
                                ? bench::measure_graph(graph, stream, warmup, repeat)
                                : bench::measure_launch([&](cudaStream_t s) { data.launch(s); },
                                                        stream, warmup, repeat);
                        std::printf("%d,%d,%d,%s,%.6f,%.6f,%.6f,%zu\n", batch, tokens, count,
                                    captured ? "graph" : "eager", timing.median_us, timing.min_us,
                                    timing.p95_us, captured ? graph.nodes() : 0);
                    }
                }
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "normalized_rope_kv_append_bench: %s\n", error.what());
        return 1;
    }
}
