// Public convolution-finish/residual/plain-RMS benchmark with warm-L2 graph timing.
#include "ninfer/ops/grouped_dynamic_conv.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer_bench_common.h"

#include <cstdio>
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {
using namespace ninfer;

std::vector<std::uint16_t> fixture(std::size_t count, unsigned seed, float scale) {
    std::vector<std::uint16_t> values(count);
    for (auto& value : values) {
        seed  = seed * 1664525U + 1013904223U;
        value = bench::f32_to_bf16(scale * (static_cast<float>(seed >> 8) / 8388608.0F - 1.0F));
    }
    return values;
}

DeviceBuffer upload(const std::vector<std::uint16_t>& values) {
    DeviceBuffer result(values.size() * sizeof(std::uint16_t));
    result.copy_from_host(values.data(), result.bytes);
    return result;
}

void run(int batch, int width, int repeat) {
    constexpr int kD            = ops::kGroupedDynamicConvHidden;
    constexpr int kG            = ops::kGroupedDynamicConvGroups;
    const auto columns          = static_cast<std::size_t>(batch) * static_cast<std::size_t>(width);
    const auto elems            = columns * kD;
    const auto initial_residual = fixture(elems, 67U, 1.0F);
    auto hidden                 = upload(fixture(elems, 17U, 1.0F));
    auto base                   = upload(fixture(4 * kD, 31U, 0.5F));
    auto dynamic                = upload(fixture(columns * 2 * kG, 47U, 0.25F));
    auto residual               = upload(initial_residual);
    auto weight                 = upload(fixture(kD, 79U, 1.0F));
    DeviceBuffer finished(elems * 2), normalized(elems * 2);
    Tensor hidden_t(hidden.p, DType::BF16, {kD, width, batch});
    Tensor base_t(base.p, DType::BF16, {kD, 2, 2});
    Tensor dynamic_t(dynamic.p, DType::BF16, {kG, 2, width, batch});
    Tensor residual_t(residual.p, DType::BF16, {kD, width, batch});
    Tensor weight_t(weight.p, DType::BF16, {kD});
    Tensor finished_t(finished.p, DType::BF16, {kD, width, batch});
    Tensor normalized_t(normalized.p, DType::BF16, {kD, width, batch});
    DeviceContext context;
    CUDA_CHECK(cudaDeviceSynchronize());
    for (const bool composite : {false, true}) {
        bench::TimedGraph graph;
        graph.capture(context.stream, [&](cudaStream_t stream) {
            if (composite) {
                ops::grouped_dynamic_conv_finish_residual_rmsnorm(hidden_t, base_t, dynamic_t,
                                                                  residual_t, weight_t, 1.0e-6F,
                                                                  normalized_t, stream);
            } else {
                ops::grouped_dynamic_conv_finish(hidden_t, base_t, dynamic_t, finished_t, stream);
                ops::residual_add(finished_t, residual_t, stream);
                ops::rmsnorm(residual_t, weight_t, 1.0e-6F, false, normalized_t, stream);
            }
        });
        std::vector<double> samples;
        samples.reserve(static_cast<std::size_t>(repeat));
        for (int iteration = 0; iteration < repeat + 20; ++iteration) {
            // Restore observable state outside timing and fence default-stream fixture transfer.
            residual.copy_from_host(initial_residual.data(), residual.bytes);
            CUDA_CHECK(cudaDeviceSynchronize());
            const double time = graph.launch_timed(context.stream);
            if (iteration >= 20) { samples.push_back(time); }
        }
        const auto timing = bench::summarize_timings(std::move(samples));
        std::printf("%s B=%d T=%d nodes=%zu median_us=%.6f min_us=%.6f p95_us=%.6f "
                    "cache=warm-L2 reset=excluded\n",
                    composite ? "finish_residual_rmsnorm" : "finish+residual+plain_rmsnorm", batch,
                    width, graph.nodes(), timing.median_us, timing.min_us, timing.p95_us);
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        int batch  = 6;
        int width  = 8;
        int repeat = 100;
        for (int index = 1; index < argc; ++index) {
            const std::string_view argument(argv[index]);
            if (argument == "--help") {
                std::printf("usage: %s [--batch B] [--width T] [--repeat N]\n", argv[0]);
                return 0;
            }
            if (index + 1 == argc) { throw std::invalid_argument("missing option value"); }
            const int value = bench::parse_number<int>(argv[++index], argument);
            if (argument == "--batch") {
                batch = value;
            } else if (argument == "--width") {
                width = value;
            } else if (argument == "--repeat") {
                repeat = value;
            } else {
                throw std::invalid_argument("unknown option");
            }
        }
        if (batch < 1 || batch > 8 || width < 1 || width > 16 || repeat < 1 || repeat > 100000) {
            throw std::invalid_argument("requires B=1..8, T=1..16, repeat=1..100000");
        }
        run(batch, width, repeat);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
