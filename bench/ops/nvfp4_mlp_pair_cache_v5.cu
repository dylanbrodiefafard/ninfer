// Dual-gate / pair harness for NVFP4 SwiGLU->MLP-down cache hints (T=4).
// See ninfer/bench/PERF_LANDS.md — Streaming on SwiGLU is gated on *pair* warm, not solo.
// Build: ninfer_nvfp4_mlp_pair_cache_v5 (NINFER_BUILD_BENCHMARKS=ON)

#include "core/device.h"
#include "ninfer_bench_common.h"
#include "ops/common/math.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_output.cuh"
#include "ops/linear/nvfp4/nvfp4_small_t.cuh"
#include "quantized_weight.cuh"

#include <cuda_bf16.h>

#include <cstdio>
#include <cstring>
#include <string_view>
#include <vector>

using namespace ninfer;
using namespace ninfer::ops;
using namespace ninfer::ops::detail;

namespace {

constexpr int kT = 4;
constexpr std::size_t kFlushBytes = 256ULL << 20;

using GateUp = Nvfp4MlpGateUpGeometry;
using Down   = Nvfp4Residual17408Geometry;
constexpr int kInter = GateUp::kOutputRows / 2;

using SwigluDef =
    Nvfp4SmallTSchedule<8, 1, 2, 16, kT, 1, Nvfp4SmallTActivationAccess::SharedPhase,
                        Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                        Nvfp4SmallTBlockOrder::RowsContiguous, 1>;
using SwigluCs =
    Nvfp4SmallTSchedule<8, 1, 2, 16, kT, 1, Nvfp4SmallTActivationAccess::SharedPhase,
                        Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Streaming, 1,
                        Nvfp4SmallTBlockOrder::RowsContiguous, 1>;
using SwigluCsU4 =
    Nvfp4SmallTSchedule<8, 1, 2, 16, kT, 1, Nvfp4SmallTActivationAccess::SharedPhase,
                        Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Streaming, 4,
                        Nvfp4SmallTBlockOrder::RowsContiguous, 1>;
using DownDef = typename Nvfp4LinearSmallTProductionSchedule<Down, kT>::Type;
using DownCs =
    Nvfp4SmallTSchedule<4, 1, 2, 16, kT, 1, Nvfp4SmallTActivationAccess::TokenPacked,
                        Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Streaming, 4,
                        Nvfp4SmallTBlockOrder::RowsContiguous, 1>;

template <class Schedule>
__global__ __launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void swiglu_k(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, float inverse, __nv_bfloat16* __restrict__ out) {
    __shared__ Nvfp4SmallTSharedStorage<GateUp, kT, Schedule> shared;
    constexpr int kCtasPerM128 = 128 / Schedule::kWarpsPerCta;
    const int block            = static_cast<int>(blockIdx.x);
    const int m_tile           = block / kCtasPerM128;
    const int cta_in_tile      = block - m_tile * kCtasPerM128;
    const int lane             = static_cast<int>(threadIdx.x) & 31;
    const int warp             = static_cast<int>(threadIdx.x) >> 5;
    const int flat_pair        = cta_in_tile * Schedule::kWarpsPerCta + warp;
    const int gate_row =
        m_tile * 128 + (flat_pair >> 2) + (flat_pair & 3) * 32;
    const int parent_rows[2] = {gate_row, gate_row + kInter};
    float accumulators[2][kT][1] = {};
    compute_nvfp4_small_t_rows<GateUp, kT, Schedule>(
        Nvfp4PackedActivation<GateUp>{x}, codes, scales, shared, inverse, parent_rows, warp * 2, 0,
        0, lane, accumulators);
#pragma unroll
    for (int token = 0; token < kT; ++token) {
        float gate = accumulators[0][token][0];
        float up   = accumulators[1][token][0];
        gate       = ops::warp_reduce_sum(gate);
        up         = ops::warp_reduce_sum(up);
        if (lane == 0) {
            out[static_cast<std::int64_t>(token) * kInter + gate_row] =
                __float2bfloat16_rn(ops::silu(gate) * up);
        }
    }
}

template <class Schedule>
void launch_swiglu(const __nv_bfloat16* x, const std::uint8_t* c, const std::uint8_t* s, float inv,
                   __nv_bfloat16* o, cudaStream_t stream) {
    constexpr int kBlocks = kInter / Schedule::kWarpsPerCta;
    swiglu_k<Schedule><<<kBlocks, Schedule::kThreads, 0, stream>>>(x, c, s, inv, o);
    CUDA_CHECK(cudaGetLastError());
}

template <class Schedule>
void launch_down(const __nv_bfloat16* x, const std::uint8_t* c, const std::uint8_t* s, float inv,
                 __nv_bfloat16* o, cudaStream_t stream) {
    constexpr int kTokenTiles = (kT + Schedule::kTokenTile - 1) / Schedule::kTokenTile;
    constexpr int kBlocks     = (Down::kOutputRows / Schedule::kRowsPerCta) * kTokenTiles;
    nvfp4_small_t_kernel<Down, kT, Schedule>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(Nvfp4PackedActivation<Down>{x}, c, s, inv,
                                                    Nvfp4IdentityEpilogue{},
                                                    Nvfp4ContiguousOutput{o, Down::kOutputRows});
    CUDA_CHECK(cudaGetLastError());
}

using PairFn = void (*)(const __nv_bfloat16*, const std::uint8_t*, const std::uint8_t*, float,
                        const std::uint8_t*, const std::uint8_t*, float, __nv_bfloat16*,
                        __nv_bfloat16*, cudaStream_t);

template <class Sw, class Dn>
void pair(const __nv_bfloat16* x, const std::uint8_t* gc, const std::uint8_t* gs, float gi,
          const std::uint8_t* dc, const std::uint8_t* ds, float di, __nv_bfloat16* mid,
          __nv_bfloat16* out, cudaStream_t stream) {
    launch_swiglu<Sw>(x, gc, gs, gi, mid, stream);
    launch_down<Dn>(mid, dc, ds, di, out, stream);
}

double median_warm(PairFn fn, const __nv_bfloat16* x, const std::uint8_t* gc, const std::uint8_t* gs,
                   float gi, const std::uint8_t* dc, const std::uint8_t* ds, float di,
                   __nv_bfloat16* mid, __nv_bfloat16* out, cudaStream_t stream) {
    for (int i = 0; i < 20; ++i) fn(x, gc, gs, gi, dc, ds, di, mid, out, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaEvent_t a = nullptr, b = nullptr;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    std::vector<double> samples;
    for (int i = 0; i < 80; ++i) {
        CUDA_CHECK(cudaEventRecord(a, stream));
        for (int j = 0; j < 8; ++j) fn(x, gc, gs, gi, dc, ds, di, mid, out, stream);
        CUDA_CHECK(cudaEventRecord(b, stream));
        CUDA_CHECK(cudaEventSynchronize(b));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
        samples.push_back(ms * 1000.0 / 8.0);
    }
    std::sort(samples.begin(), samples.end());
    CUDA_CHECK(cudaEventDestroy(a));
    CUDA_CHECK(cudaEventDestroy(b));
    return samples[samples.size() / 2];
}

} // namespace

int main() {
    try {
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        DeviceBuffer flush(kFlushBytes);
        DeviceBuffer xin = bench::make_bf16(static_cast<std::size_t>(GateUp::kInputRows) * kT);
        DeviceBuffer mid(static_cast<std::size_t>(kInter) * kT * 2);
        DeviceBuffer out(static_cast<std::size_t>(Down::kOutputRows) * kT * 2);
        auto gate = bench::make_nvfp4_weight(GateUp::kOutputRows, GateUp::kInputRows);
        auto down = bench::make_nvfp4_weight(Down::kOutputRows, Down::kInputRows);
        const float gi = 1.0F / gate.weight.weight_scale_divisor;
        const float di = 1.0F / down.weight.weight_scale_divisor;
        const auto* x  = static_cast<const __nv_bfloat16*>(xin.p);
        const auto* gc = static_cast<const std::uint8_t*>(gate.weight.qdata);
        const auto* gs = static_cast<const std::uint8_t*>(gate.weight.scales);
        const auto* dc = static_cast<const std::uint8_t*>(down.weight.qdata);
        const auto* ds = static_cast<const std::uint8_t*>(down.weight.scales);
        auto* m        = static_cast<__nv_bfloat16*>(mid.p);
        auto* o        = static_cast<__nv_bfloat16*>(out.p);

        struct Cand {
            const char* name;
            PairFn fn;
        };
        const Cand cands[] = {
            {"def+def", &pair<SwigluDef, DownDef>},
            {"cs+def", &pair<SwigluCs, DownDef>},
            {"csu4+def", &pair<SwigluCsU4, DownDef>},
            {"def+cs", &pair<SwigluDef, DownCs>},
            {"cs+cs", &pair<SwigluCs, DownCs>},
        };

        std::printf("SwiGLU→down T=4 pair (cache hints)\n");
        std::printf("%-10s %10s %10s %8s %8s\n", "variant", "cold_us", "warm_us", "cold%",
                    "warm%");

        double base_cold = 0, base_warm = 0;
        for (const auto& c : cands) {
            const auto run = [&](cudaStream_t s) {
                c.fn(x, gc, gs, gi, dc, ds, di, m, o, s);
            };
            const double cold =
                bench::measure_cold_launch(run, flush, stream, 5, 30).median_us;
            const double warm = median_warm(c.fn, x, gc, gs, gi, dc, ds, di, m, o, stream);
            if (std::string_view(c.name) == "def+def") {
                base_cold = cold;
                base_warm = warm;
            }
            const double cold_vs = 100.0 * (base_cold - cold) / base_cold;
            const double warm_vs = 100.0 * (base_warm - warm) / base_warm;
            std::printf("%-10s %10.3f %10.3f %+7.2f %+7.2f%s\n", c.name, cold, warm, cold_vs,
                        warm_vs,
                        (cold_vs >= 1.0 && warm_vs >= 1.0) ? "  <<WIN"
                        : (warm_vs >= 1.0)                 ? "  (warm)"
                                                           : "");
        }

        // L2-bridge upper bound: flush mid between kernels on baseline path.
        {
            auto flush_pair = [&](cudaStream_t s) {
                launch_swiglu<SwigluDef>(x, gc, gs, gi, m, s);
                CUDA_CHECK(cudaMemsetAsync(flush.p, 0, flush.bytes, s));
                launch_down<DownDef>(m, dc, ds, di, o, s);
            };
            for (int i = 0; i < 20; ++i) flush_pair(stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            cudaEvent_t a = nullptr, b = nullptr;
            CUDA_CHECK(cudaEventCreate(&a));
            CUDA_CHECK(cudaEventCreate(&b));
            std::vector<double> samples;
            for (int i = 0; i < 40; ++i) {
                CUDA_CHECK(cudaEventRecord(a, stream));
                for (int j = 0; j < 4; ++j) flush_pair(stream);
                CUDA_CHECK(cudaEventRecord(b, stream));
                CUDA_CHECK(cudaEventSynchronize(b));
                float ms = 0;
                CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
                samples.push_back(ms * 1000.0 / 4.0);
            }
            std::sort(samples.begin(), samples.end());
            const double flushed = samples[samples.size() / 2];
            std::printf("\nL2-bridge: warm pair=%.3f  flushed_mid=%.3f  gap=%.3f us "
                        "(%.1f%% of pair)\n",
                        base_warm, flushed, flushed - base_warm,
                        100.0 * (flushed - base_warm) / base_warm);
            CUDA_CHECK(cudaEventDestroy(a));
            CUDA_CHECK(cudaEventDestroy(b));
        }

        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
