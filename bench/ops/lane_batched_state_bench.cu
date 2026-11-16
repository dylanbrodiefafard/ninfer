// Public-Op benchmark of the lane-batched stateful forms at the Qwen3.8-Flash-Next geometry: B
// lanes (rows) of W columns each, every lane on its own state slot.
//
//   gdn_snapshot   gdn_input_proj_conv_snapshot, [16384,2560] parent, A16, in place for W=1
//   gdn_record     gdn_input_proj_conv_record, [16384,2560] parent, A16
//   gdn_per_lane   the same snapshot as B one-row calls (the per-request launch structure)
//   gdn_unfused    gdn_input_proj over W*B columns, then causal_conv1d_silu_snapshot (whose
//                  unsplit [10240,W,B] output still needs a q/k/v split before the recurrence)
//   gdn_recurrence gated_delta_net_snapshot, 16 q/k and 48 value heads, normalized q/k
//   ple_inject / ple_inject_snapshot
//
// Each case is one CUDA graph of the named public call(s), replayed cold after a 256 MiB L2 flush.

#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/ple.h"

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "ninfer_bench_common.h"
#include "quantized_weight.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <functional>
#include <string>
#include <vector>

using namespace ninfer;

namespace {

constexpr std::int32_t kHidden    = 2560;
constexpr std::int32_t kQueryRows = 2048;
constexpr std::int32_t kKeyRows   = 2048;
constexpr std::int32_t kValueRows = 6144;
constexpr std::int32_t kZRows     = 6144;
constexpr std::int32_t kChannels  = kQueryRows + kKeyRows + kValueRows;
constexpr std::int32_t kRows      = kChannels + kZRows;
constexpr std::int32_t kQkHeads   = 16;
constexpr std::int32_t kVHeads    = 48;
constexpr std::int32_t kHeadDim   = 128;
constexpr int kWarmup             = 5;
constexpr int kRepeat             = 50;

// Graph capture needs a non-default stream.
class Stream {
public:
    Stream() { CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking)); }

    ~Stream() { cudaStreamDestroy(stream_); }

    Stream(const Stream&)            = delete;
    Stream& operator=(const Stream&) = delete;

    [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }

private:
    cudaStream_t stream_ = nullptr;
};

DeviceBuffer upload_i32(const std::vector<std::int32_t>& values) {
    DeviceBuffer buffer(values.size() * sizeof(std::int32_t));
    buffer.copy_from_host(values.data(), buffer.bytes);
    return buffer;
}

void report(const char* op, const char* format, std::int32_t batch, std::int32_t width,
            const std::function<void(cudaStream_t)>& body, DeviceBuffer& flush,
            cudaStream_t stream) {
    bench::TimedGraph graph;
    graph.capture(stream, body);
    const bench::ColdTiming timing =
        bench::measure_cold_graph(graph, flush, stream, kWarmup, kRepeat);
    std::printf("%-20s %-5s B=%2d W=%2d  median=%9.2f us  min=%9.2f  p95=%9.2f\n", op, format,
                batch, width, timing.median_us, timing.min_us, timing.p95_us);
}

// Lane b uses state slot b (and, for a W>1 snapshot, slots B + b*W .. B + b*W + W-1).
std::vector<std::int32_t> iota_slots(std::int32_t batch, std::int32_t first, std::int32_t step) {
    std::vector<std::int32_t> slots(static_cast<std::size_t>(batch));
    for (std::int32_t b = 0; b < batch; ++b) {
        slots[static_cast<std::size_t>(b)] = first + b * step;
    }
    return slots;
}

void run_gdn(QType qtype, std::int32_t batch, std::int32_t width, DeviceBuffer& flush,
             cudaStream_t stream) {
    const char* format                  = qtype == QType::NVFP4 ? "nvfp4" : "fp8";
    bench::PackedQuantizedWeight parent = qtype == QType::NVFP4
                                              ? bench::make_nvfp4_weight(kRows, kHidden)
                                              : bench::make_fp8_weight(kRows, kHidden);
    const std::int32_t columns          = width * batch;
    const std::int32_t slots            = batch + columns;
    DeviceBuffer x       = bench::make_bf16(static_cast<std::size_t>(kHidden) * columns);
    DeviceBuffer conv_w  = bench::make_bf16(static_cast<std::size_t>(kChannels) * 4);
    DeviceBuffer states  = bench::make_bf16(static_cast<std::size_t>(kChannels) * 3 * slots);
    DeviceBuffer query   = bench::make_bf16(static_cast<std::size_t>(kQueryRows) * columns);
    DeviceBuffer key     = bench::make_bf16(static_cast<std::size_t>(kKeyRows) * columns);
    DeviceBuffer value   = bench::make_bf16(static_cast<std::size_t>(kValueRows) * columns);
    DeviceBuffer z       = bench::make_bf16(static_cast<std::size_t>(kZRows) * columns);
    DeviceBuffer record  = bench::make_bf16(static_cast<std::size_t>(kChannels) * columns);
    DeviceBuffer unsplit = bench::make_bf16(static_cast<std::size_t>(kChannels) * columns);
    // W=1 publishes in place; W>1 publishes W snapshots per lane after the B home slots.
    const DeviceBuffer initial = upload_i32(iota_slots(batch, 0, 1));
    const DeviceBuffer base =
        upload_i32(width == 1 ? iota_slots(batch, 0, 1) : iota_slots(batch, batch, width));

    const Tensor tx(x.p, DType::BF16, {kHidden, width, batch});
    const Tensor tconv(conv_w.p, DType::BF16, {kChannels, 4});
    Tensor tstates(states.p, DType::BF16, {kChannels, 3, slots});
    const Tensor tinitial(initial.p, DType::I32, {batch});
    const Tensor tbase(base.p, DType::I32, {batch});
    Tensor tq(query.p, DType::BF16, {kQueryRows, width, batch});
    Tensor tk(key.p, DType::BF16, {kKeyRows, width, batch});
    Tensor tv(value.p, DType::BF16, {kValueRows, width, batch});
    Tensor tz(z.p, DType::BF16, {kZRows, width, batch});
    Tensor trecord(record.p, DType::BF16, {kChannels, width, batch});
    WorkspaceArena workspace(std::max<std::size_t>(
        256, ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                 qtype, kRows, kHidden, ops::LinearPolicy::A16Only, batch, width, width)));
    WorkspaceArena lane_workspace(std::max<std::size_t>(
        256, ops::gdn_input_proj_conv_snapshot_workspace_capacity_bytes(
                 qtype, kRows, kHidden, ops::LinearPolicy::A16Only, 1, width, width)));

    report(
        "gdn_snapshot", format, batch, width,
        [&](cudaStream_t s) {
            ops::gdn_input_proj_conv_snapshot(tx, parent.weight, tconv, tstates, Tensor{}, tinitial,
                                              tbase, tq, tk, tv, tz, workspace, s);
        },
        flush, stream);
    report(
        "gdn_per_lane", format, batch, width,
        [&](cudaStream_t s) {
            for (std::int32_t b = 0; b < batch; ++b) {
                Tensor q_b = tq.slice(2, b, 1), k_b = tk.slice(2, b, 1), v_b = tv.slice(2, b, 1),
                       z_b = tz.slice(2, b, 1);
                ops::gdn_input_proj_conv_snapshot(tx.slice(2, b, 1), parent.weight, tconv, tstates,
                                                  Tensor{}, tinitial.slice(0, b, 1),
                                                  tbase.slice(0, b, 1), q_b, k_b, v_b, z_b,
                                                  ops::LinearPolicy::A16Only, lane_workspace, s);
            }
        },
        flush, stream);
    report(
        "gdn_unfused", format, batch, width,
        [&](cudaStream_t s) {
            const Tensor x_flat(x.p, DType::BF16, {kHidden, columns});
            Tensor qkv(record.p, DType::BF16, {kChannels, columns});
            Tensor z_flat(z.p, DType::BF16, {kZRows, columns});
            ops::gdn_input_proj(x_flat, parent.weight, qkv, z_flat, s);
            const Tensor projected(record.p, DType::BF16, {kChannels, width, batch});
            Tensor out(unsplit.p, DType::BF16, {kChannels, width, batch});
            ops::causal_conv1d_silu_snapshot(projected, tconv, tstates, Tensor{}, tinitial, tbase,
                                             out, s);
        },
        flush, stream);
    if (width >= 2) {
        report(
            "gdn_record", format, batch, width,
            [&](cudaStream_t s) {
                ops::gdn_input_proj_conv_record(tx, parent.weight, tconv, tstates, Tensor{},
                                                tinitial, trecord, tq, tk, tv, tz, workspace, s);
            },
            flush, stream);
    }
}

void run_recurrence(std::int32_t batch, std::int32_t width, DeviceBuffer& flush,
                    cudaStream_t stream) {
    const std::int32_t columns = width * batch;
    const std::int32_t slots   = batch + columns;
    DeviceBuffer q = bench::make_bf16(static_cast<std::size_t>(kHeadDim) * kQkHeads * columns);
    DeviceBuffer k = bench::make_bf16(static_cast<std::size_t>(kHeadDim) * kQkHeads * columns);
    DeviceBuffer v = bench::make_bf16(static_cast<std::size_t>(kHeadDim) * kVHeads * columns);
    DeviceBuffer out(static_cast<std::size_t>(kHeadDim) * kVHeads * columns * 2);
    DeviceBuffer g     = bench::make_zeros(static_cast<std::size_t>(kVHeads) * columns * 4);
    DeviceBuffer beta  = bench::make_zeros(static_cast<std::size_t>(kVHeads) * columns * 4);
    DeviceBuffer state = bench::make_zeros(static_cast<std::size_t>(kHeadDim) * kHeadDim * kVHeads *
                                           slots * sizeof(float));
    const DeviceBuffer initial = upload_i32(iota_slots(batch, 0, 1));
    const DeviceBuffer base =
        upload_i32(width == 1 ? iota_slots(batch, 0, 1) : iota_slots(batch, batch, width));
    const Tensor tq(q.p, DType::BF16, {kHeadDim, kQkHeads, width, batch});
    const Tensor tk(k.p, DType::BF16, {kHeadDim, kQkHeads, width, batch});
    const Tensor tv(v.p, DType::BF16, {kHeadDim, kVHeads, width, batch});
    const Tensor tg(g.p, DType::FP32, {kVHeads, width, batch});
    const Tensor tbeta(beta.p, DType::FP32, {kVHeads, width, batch});
    Tensor tstate(state.p, DType::FP32, {kHeadDim, kHeadDim, kVHeads, slots});
    Tensor tout(out.p, DType::BF16, {kHeadDim, kVHeads, width, batch});
    const Tensor tinitial(initial.p, DType::I32, {batch});
    const Tensor tbase(base.p, DType::I32, {batch});
    const float scale = 1.0F / std::sqrt(static_cast<float>(kHeadDim));
    report(
        "gdn_recurrence", "fp32", batch, width,
        [&](cudaStream_t s) {
            ops::gated_delta_net_snapshot(tq, tk, tv, tg, tbeta, scale, true, tstate, Tensor{},
                                          tinitial, tbase, tout, s);
        },
        flush, stream);
}

void run_ple(std::int32_t batch, std::int32_t width, DeviceBuffer& flush, cudaStream_t stream) {
    const std::int32_t columns = width * batch;
    const std::int32_t slots   = batch + columns;
    DeviceBuffer residual = bench::make_bf16(static_cast<std::size_t>(ops::kPleChannels) * columns);
    DeviceBuffer key = bench::make_zeros(static_cast<std::size_t>(ops::kPleChannels) * columns * 4);
    DeviceBuffer value = bench::make_zeros(static_cast<std::size_t>(ops::kPleHidden) * columns * 4);
    DeviceBuffer norms = bench::make_bf16(static_cast<std::size_t>(ops::kPleChannels) * 3);
    DeviceBuffer conv  = bench::make_bf16(static_cast<std::size_t>(ops::kPleChannels) * 4);
    DeviceBuffer states        = bench::make_bf16(static_cast<std::size_t>(ops::kPleConvHistory) *
                                                  ops::kPleChannels * slots);
    const DeviceBuffer initial = upload_i32(iota_slots(batch, 0, 1));
    const DeviceBuffer base    = upload_i32(iota_slots(batch, batch, width));
    Tensor tresidual(residual.p, DType::BF16, {ops::kPleHidden, ops::kPleStreams, width, batch});
    const Tensor tkey(key.p, DType::FP32, {ops::kPleChannels, width, batch});
    const Tensor tvalue(value.p, DType::FP32, {ops::kPleHidden, width, batch});
    auto* norm_base = static_cast<std::uint16_t*>(norms.p);
    const ops::PleWeights weights{
        Tensor(norm_base, DType::BF16, {ops::kPleChannels}),
        Tensor(norm_base + ops::kPleChannels, DType::BF16, {ops::kPleChannels}),
        Tensor(norm_base + 2 * ops::kPleChannels, DType::BF16, {ops::kPleChannels}),
        Tensor(conv.p, DType::BF16, {ops::kPleConvWidth, ops::kPleChannels})};
    Tensor tstates(states.p, DType::BF16, {ops::kPleConvHistory, ops::kPleChannels, slots});
    const Tensor tinitial(initial.p, DType::I32, {batch});
    const Tensor tbase(base.p, DType::I32, {batch});
    WorkspaceArena workspace(ops::ple_inject_workspace_bytes(columns));
    report(
        "ple_inject", "bf16", batch, width,
        [&](cudaStream_t s) {
            ops::ple_inject(tresidual, tkey, tvalue, weights, tstates, Tensor{}, tinitial,
                            workspace, s);
        },
        flush, stream);
    report(
        "ple_inject_snapshot", "bf16", batch, width,
        [&](cudaStream_t s) {
            ops::ple_inject_snapshot(tresidual, tkey, tvalue, weights, tstates, Tensor{}, tinitial,
                                     tbase, workspace, s);
        },
        flush, stream);
}

} // namespace

int main() {
    try {
        bench::print_device_caps("ninfer_lane_batched_state_bench");
        DeviceBuffer flush(std::size_t{256} << 20);
        const Stream stream;
        for (const std::int32_t width : {1, 4}) {
            for (const std::int32_t batch : {1, 4, 12}) {
                run_gdn(QType::FP8_E4M3FN_ROW_BF16S, batch, width, flush, stream.get());
                run_gdn(QType::NVFP4, batch, width, flush, stream.get());
                run_recurrence(batch, width, flush, stream.get());
                run_ple(batch, width, flush, stream.get());
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ninfer_lane_batched_state_bench: %s\n", error.what());
        return 1;
    }
}
