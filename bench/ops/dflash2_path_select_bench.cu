// Public benchmark for dflash2_path_select at the registered drafter shapes.
//
//   ./ninfer_dflash2_path_select_bench                       # Flash-Next and 27B sweep
//   ./ninfer_dflash2_path_select_bench --vocab 248320 --hidden 2560 --tokens 3 --batch 5
//   ./ninfer_dflash2_path_select_bench --vocab 131072 --hidden 5120 --tokens 7 --token-ids
//
// One captured graph holds one Op call (hidden projection, column top-16, Markov walk with
// selector q), replayed with its logits resident in L2 as they are after the drafter head.
#include "core/device.h"
#include "core/tensor.h"
#include "ninfer/ops/dflash2_path_select.h"
#include "ninfer/ops/sampling.h"
#include "direct_bf16_weight.cuh"
#include "ninfer_bench_common.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace ninfer;
using namespace ninfer::bench;

namespace {

constexpr std::int32_t kCodebookRows = ops::kDflash2PathSelectCodebookRows;
constexpr std::int32_t kRank         = ops::kDflash2PathSelectRank;
constexpr std::int32_t kTopK         = ops::kDflash2PathSelectTopK;

struct Point {
    std::int32_t vocab  = 248320;
    std::int32_t hidden = 2560;
    std::int32_t tokens = 3;
    std::int32_t batch  = 1;
    bool token_ids      = false;
};

// Drafter-like logits: a hashed bulk in roughly [-8, 8) with every 4099th row lifted by 12, so
// each column has a few dozen clear leaders above a dense tail, as a trained head produces.
__global__ void fill_logits_kernel(__nv_bfloat16* logits, std::uint64_t count, std::int32_t vocab) {
    const std::uint64_t begin  = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t stride = static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
    for (std::uint64_t index = begin; index < count; index += stride) {
        std::uint32_t hash = static_cast<std::uint32_t>(index) * 0x9e3779b9U + 0x7f4a7c15U;
        hash ^= hash >> 15;
        hash *= 0x2c1b3c6dU;
        hash ^= hash >> 12;
        const float bulk = static_cast<float>(hash & 0xffffU) * (16.0F / 65536.0F) - 8.0F;
        const auto row   = static_cast<std::int32_t>(index % static_cast<std::uint64_t>(vocab));
        logits[index]    = __float2bfloat16_rn(row % 4099 == 0 ? bulk + 12.0F : bulk);
    }
}

void run(const Point& point) {
    const std::size_t columns        = static_cast<std::size_t>(point.tokens) * point.batch;
    const std::size_t logit_elements = static_cast<std::size_t>(point.vocab) * columns;
    DeviceBuffer logits(logit_elements * sizeof(std::uint16_t));
    fill_logits_kernel<<<1024, 256, 0, cudaStreamLegacy>>>(static_cast<__nv_bfloat16*>(logits.p),
                                                           logit_elements, point.vocab);
    CUDA_CHECK(cudaGetLastError());
    DeviceBuffer hidden               = make_bf16(static_cast<std::size_t>(point.hidden) * columns);
    DeviceBuffer pred                 = make_bf16(static_cast<std::size_t>(kRank) * kCodebookRows);
    DeviceBuffer succ                 = make_bf16(static_cast<std::size_t>(kRank) * kCodebookRows);
    const DirectBf16Weight projection = make_direct_bf16_weight(kRank, point.hidden);

    std::vector<std::int32_t> host_anchors(static_cast<std::size_t>(point.batch));
    std::vector<std::int32_t> host_positions(static_cast<std::size_t>(point.batch));
    std::vector<ops::SamplingConfig> host_configs(static_cast<std::size_t>(point.batch));
    for (std::int32_t b = 0; b < point.batch; ++b) {
        host_anchors[static_cast<std::size_t>(b)]             = 1000 + 37 * b;
        host_positions[static_cast<std::size_t>(b)]           = 4096 + b;
        host_configs[static_cast<std::size_t>(b)].temperature = 0.6F;
        host_configs[static_cast<std::size_t>(b)].seed = 0x1234ULL + static_cast<unsigned>(b);
    }
    DeviceBuffer anchors(host_anchors.size() * sizeof(std::int32_t));
    DeviceBuffer positions(host_positions.size() * sizeof(std::int32_t));
    DeviceBuffer configs(host_configs.size() * sizeof(ops::SamplingConfig));
    anchors.copy_from_host(host_anchors.data(), anchors.bytes);
    positions.copy_from_host(host_positions.data(), positions.bytes);
    configs.copy_from_host(host_configs.data(), configs.bytes);

    // The 27B optimized proposal head maps shortlist row r to token id 2r.
    DeviceBuffer token_ids;
    if (point.token_ids) {
        std::vector<std::int32_t> host_ids(static_cast<std::size_t>(point.vocab));
        for (std::int32_t r = 0; r < point.vocab; ++r) {
            host_ids[static_cast<std::size_t>(r)] = (2 * r) % kCodebookRows;
        }
        token_ids = DeviceBuffer(host_ids.size() * sizeof(std::int32_t));
        token_ids.copy_from_host(host_ids.data(), token_ids.bytes);
    }

    DeviceBuffer path(columns * sizeof(std::int32_t));
    DeviceBuffer selector_ids(columns * kTopK * sizeof(std::int32_t));
    DeviceBuffer selector_q(columns * kTopK * sizeof(float));
    Tensor logits_t(logits.p, DType::BF16, {point.vocab, point.tokens, point.batch});
    Tensor hidden_t(hidden.p, DType::BF16, {point.hidden, point.tokens, point.batch});
    Tensor pred_t(pred.p, DType::BF16, {kRank, kCodebookRows});
    Tensor succ_t(succ.p, DType::BF16, {kRank, kCodebookRows});
    Tensor anchors_t(anchors.p, DType::I32, {point.batch});
    Tensor positions_t(positions.p, DType::I32, {point.batch});
    Tensor path_t(path.p, DType::I32, {point.tokens, point.batch});
    Tensor ids_t(selector_ids.p, DType::I32, {kTopK, point.tokens, point.batch});
    Tensor q_t(selector_q.p, DType::FP32, {kTopK, point.tokens, point.batch});
    Tensor token_ids_t(token_ids.p, DType::I32, {point.vocab});
    WorkspaceArena workspace(ops::dflash2_path_select_workspace_capacity_bytes(
        QType::BF16_CTRL, point.hidden, point.tokens, point.tokens, point.batch));

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    TimedGraph graph;
    graph.capture(stream, [&](cudaStream_t capture) {
        ops::dflash2_path_select(
            logits_t, hidden_t, projection.weight, pred_t, succ_t, anchors_t, positions_t,
            static_cast<const ops::SamplingConfig*>(configs.p), path_t, workspace, capture,
            point.token_ids ? &token_ids_t : nullptr, nullptr, nullptr, &ids_t, &q_t);
    });
    const ColdTiming timing = measure_graph(graph, stream, 50, 400);
    CUDA_CHECK(cudaStreamDestroy(stream));
    std::printf("dflash2_path_select V=%d D=%d T=%d B=%d%s  median=%8.2f us  min=%8.2f us  "
                "p95=%8.2f us\n",
                point.vocab, point.hidden, point.tokens, point.batch,
                point.token_ids ? " ids" : "    ", timing.median_us, timing.min_us, timing.p95_us);
}

std::int32_t parse_extent(std::string_view text, const char* option) {
    return parse_number<std::int32_t>(text, option);
}

} // namespace

int main(int argc, char** argv) {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::printf("SKIP: no usable CUDA device\n");
        return 0;
    }
    try {
        if (argc == 1) {
            // Flash-Next: full head, D=2560, T=K drafts in 1..3 for C=1..12 requests.
            for (const auto& [tokens, batch] : std::vector<std::pair<int, int>>{
                     {1, 1}, {3, 1}, {2, 4}, {3, 4}, {3, 5}, {3, 8}, {3, 12}}) {
                run({248320, 2560, tokens, batch, false});
            }
            // Qwen3.8-27B: optimized shortlist head and the full head at D=5120.
            for (const auto& [tokens, batch] :
                 std::vector<std::pair<int, int>>{{7, 1}, {7, 2}, {5, 4}, {3, 12}}) {
                run({131072, 5120, tokens, batch, true});
            }
            run({248320, 5120, 7, 1, false});
            return 0;
        }
        Point point;
        for (int i = 1; i < argc; ++i) {
            const std::string_view arg = argv[i];
            const auto value           = [&]() -> std::string_view {
                if (i + 1 >= argc) {
                    throw std::invalid_argument(std::string(arg) + " needs a value");
                }
                return argv[++i];
            };
            if (arg == "--vocab") {
                point.vocab = parse_extent(value(), "--vocab");
            } else if (arg == "--hidden") {
                point.hidden = parse_extent(value(), "--hidden");
            } else if (arg == "--tokens") {
                point.tokens = parse_extent(value(), "--tokens");
            } else if (arg == "--batch") {
                point.batch = parse_extent(value(), "--batch");
            } else if (arg == "--token-ids") {
                point.token_ids = true;
            } else {
                throw std::invalid_argument(
                    "usage: ninfer_dflash2_path_select_bench [--vocab V] [--hidden D] "
                    "[--tokens T] [--batch B] [--token-ids]");
            }
        }
        run(point);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_dflash2_path_select_bench: %s\n", error.what());
        return 2;
    }
}
