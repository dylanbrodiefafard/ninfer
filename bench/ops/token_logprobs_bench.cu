// Benchmark for the public token_logprobs Op at the qwen3.8-27b output-head geometry.
//
//   ./ninfer_token_logprobs_bench
//
// Each shape is timed with every slot active and with every row disabled; the disabled time is
// what a round pays when no request asked for logprobs.
#include "ninfer/ops/token_logprobs.h"
#include "ninfer_bench_common.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

constexpr std::int32_t kPhysicalRows = 248320;
constexpr std::int32_t kTokenDomain  = 248077;

// Low floor with a handful of dominant tokens per column, like a decoded next-token distribution.
DeviceBuffer make_logits(std::int32_t columns) {
    std::vector<std::uint16_t> host(static_cast<std::size_t>(kPhysicalRows) *
                                    static_cast<std::size_t>(columns));
    std::uint32_t state = 0x1234567u;
    const auto next     = [&state] {
        state = state * 1664525u + 1013904223u;
        return state >> 8;
    };
    for (std::int32_t column = 0; column < columns; ++column) {
        std::uint16_t* base = host.data() + static_cast<std::size_t>(column) * kPhysicalRows;
        for (std::int32_t v = 0; v < kPhysicalRows; ++v) {
            base[v] = f32_to_bf16(-12.0f + 8.0f * static_cast<float>(next() & 0xffffu) / 65536.0f);
        }
        for (int peak = 0; peak < 32; ++peak) {
            base[next() % static_cast<std::uint32_t>(kTokenDomain)] =
                f32_to_bf16(18.0f - 0.5f * static_cast<float>(peak));
        }
    }
    DeviceBuffer device(host.size() * sizeof(std::uint16_t));
    device.copy_from_host(host.data(), device.bytes);
    return device;
}

void run_shape(std::int32_t width, std::int32_t batch, bool enabled) {
    constexpr std::int32_t kTop = ops::kMaximumTopLogprobs;
    const std::size_t slots     = static_cast<std::size_t>(width) * static_cast<std::size_t>(batch);
    DeviceBuffer logits         = make_logits(width * batch);
    DeviceBuffer tokens         = make_zeros(slots * sizeof(std::int32_t));
    std::vector<std::int32_t> host_enabled(static_cast<std::size_t>(batch), enabled ? 1 : 0);
    DeviceBuffer row_enabled(host_enabled.size() * sizeof(std::int32_t));
    row_enabled.copy_from_host(host_enabled.data(), row_enabled.bytes);
    DeviceBuffer token_logprob = make_zeros(slots * sizeof(float));
    DeviceBuffer top_ids       = make_zeros(slots * kTop * sizeof(std::int32_t));
    DeviceBuffer top_logprobs  = make_zeros(slots * kTop * sizeof(float));

    const Tensor logits_tensor(logits.p, DType::BF16, {kPhysicalRows, width, batch});
    const Tensor tokens_tensor(tokens.p, DType::I32, {width, batch});
    const Tensor enabled_tensor(row_enabled.p, DType::I32, {batch});
    Tensor token_logprob_tensor(token_logprob.p, DType::FP32, {width, batch});
    Tensor top_ids_tensor(top_ids.p, DType::I32, {kTop, width, batch});
    Tensor top_logprobs_tensor(top_logprobs.p, DType::FP32, {kTop, width, batch});

    const double bytes =
        enabled ? static_cast<double>(kTokenDomain) * 2.0 * static_cast<double>(slots) : 0.0;
    const Result result = bench_loop(
        [&](cudaStream_t stream) {
            ops::token_logprobs(logits_tensor, tokens_tensor, enabled_tensor, nullptr, nullptr,
                                kTokenDomain, token_logprob_tensor, top_ids_tensor,
                                top_logprobs_tensor, stream);
        },
        bytes);
    const std::string label = "token_logprobs W=" + std::to_string(width) +
                              " B=" + std::to_string(batch) + (enabled ? " on" : " off");
    print_result(label.c_str(), result);
}

} // namespace

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }

    struct Shape {
        std::int32_t width;
        std::int32_t batch;
    };

    // Ordinary decode, a typical accepted chain, and the widest verify frame.
    for (const Shape shape :
         {Shape{1, 1}, Shape{1, 6}, Shape{4, 1}, Shape{8, 1}, Shape{8, 6}, Shape{16, 6}}) {
        run_shape(shape.width, shape.batch, true);
        run_shape(shape.width, shape.batch, false);
    }
    return 0;
}
