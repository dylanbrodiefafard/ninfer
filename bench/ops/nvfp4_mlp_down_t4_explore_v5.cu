// MLP-down [5120,17408] T=4 dual-gate explore (cold + warm, bit-exact).

#include "core/device.h"
#include "ninfer_bench_common.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_output.cuh"
#include "ops/linear/nvfp4/nvfp4_small_t.cuh"
#include "quantized_weight.cuh"

#include <cstdio>
#include <cstring>
#include <string_view>
#include <vector>

using namespace ninfer;
using namespace ninfer::ops::detail;

namespace {

constexpr int kT = 4;
using Geometry   = Nvfp4Residual17408Geometry;
constexpr std::size_t kFlushBytes = 256ULL << 20;

using Prod = typename Nvfp4LinearSmallTProductionSchedule<Geometry, kT>::Type;

template <class Schedule>
void launch_sched(const __nv_bfloat16* x, const std::uint8_t* codes, const std::uint8_t* scales,
                  float inverse, __nv_bfloat16* out, cudaStream_t stream) {
    constexpr int kTokenTiles = (kT + Schedule::kTokenTile - 1) / Schedule::kTokenTile;
    constexpr int kBlocks     = (Geometry::kOutputRows / Schedule::kRowsPerCta) * kTokenTiles;
    nvfp4_small_t_kernel<Geometry, kT, Schedule>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(x, codes, scales, inverse,
                                                    Nvfp4IdentityEpilogue{},
                                                    Nvfp4ContiguousOutput{out, Geometry::kOutputRows});
    CUDA_CHECK(cudaGetLastError());
}

using ProdMb2 =
    Nvfp4SmallTSchedule<4, 1, 2, 16, kT, 1, Nvfp4SmallTActivationAccess::TokenPacked,
                        Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                        Nvfp4SmallTBlockOrder::RowsContiguous, 2>;
using ProdStream =
    Nvfp4SmallTSchedule<4, 1, 2, 16, kT, 1, Nvfp4SmallTActivationAccess::TokenPacked,
                        Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Streaming, 1,
                        Nvfp4SmallTBlockOrder::RowsContiguous, 1>;
using ProdUnroll4 =
    Nvfp4SmallTSchedule<4, 1, 2, 16, kT, 1, Nvfp4SmallTActivationAccess::TokenPacked,
                        Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 4,
                        Nvfp4SmallTBlockOrder::RowsContiguous, 1>;
using Warps8 =
    Nvfp4SmallTSchedule<8, 1, 2, 16, kT, 1, Nvfp4SmallTActivationAccess::TokenPacked,
                        Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                        Nvfp4SmallTBlockOrder::RowsContiguous, 1>;
using Chains2 =
    Nvfp4SmallTSchedule<4, 1, 2, 16, kT, 2, Nvfp4SmallTActivationAccess::TokenPacked,
                        Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                        Nvfp4SmallTBlockOrder::RowsContiguous, 1>;
using SharedPh =
    Nvfp4SmallTSchedule<4, 1, 2, 16, kT, 1, Nvfp4SmallTActivationAccess::SharedPhase,
                        Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                        Nvfp4SmallTBlockOrder::RowsContiguous, 1>;
// Register-cap sweep on the production schedule (PhaseUnroll=4, 128 thr): MinBlocksPerSm
// caps regs at 65536/(128*mb) -> 4/5/6/8/10/12 blocks per SM (33/42/50/67/83/100% warp occ).
using ProdMb4  =
    Nvfp4SmallTSchedule<4, 1, 2, 16, kT, 1, Nvfp4SmallTActivationAccess::TokenPacked,
                        Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 4,
                        Nvfp4SmallTBlockOrder::RowsContiguous, 4>;
using ProdMb5  =
    Nvfp4SmallTSchedule<4, 1, 2, 16, kT, 1, Nvfp4SmallTActivationAccess::TokenPacked,
                        Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 4,
                        Nvfp4SmallTBlockOrder::RowsContiguous, 5>;
using ProdMb6  =
    Nvfp4SmallTSchedule<4, 1, 2, 16, kT, 1, Nvfp4SmallTActivationAccess::TokenPacked,
                        Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 4,
                        Nvfp4SmallTBlockOrder::RowsContiguous, 6>;
using ProdMb8  =
    Nvfp4SmallTSchedule<4, 1, 2, 16, kT, 1, Nvfp4SmallTActivationAccess::TokenPacked,
                        Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 4,
                        Nvfp4SmallTBlockOrder::RowsContiguous, 8>;
using ProdMb10 =
    Nvfp4SmallTSchedule<4, 1, 2, 16, kT, 1, Nvfp4SmallTActivationAccess::TokenPacked,
                        Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 4,
                        Nvfp4SmallTBlockOrder::RowsContiguous, 10>;
using ProdMb12 =
    Nvfp4SmallTSchedule<4, 1, 2, 16, kT, 1, Nvfp4SmallTActivationAccess::TokenPacked,
                        Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 4,
                        Nvfp4SmallTBlockOrder::RowsContiguous, 12>;

using LaunchFn = void (*)(const __nv_bfloat16*, const std::uint8_t*, const std::uint8_t*, float,
                          __nv_bfloat16*, cudaStream_t);

struct Candidate {
    const char* name;
    LaunchFn fn;
};

using Chains2Unroll4 =
    Nvfp4SmallTSchedule<4, 1, 2, 16, kT, 2, Nvfp4SmallTActivationAccess::TokenPacked,
                        Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 4,
                        Nvfp4SmallTBlockOrder::RowsContiguous, 1>;

const Candidate kCandidates[] = {
    {"prod", &launch_sched<Prod>},
    {"mb2", &launch_sched<ProdMb2>},
    {"stream", &launch_sched<ProdStream>},
    {"unroll4", &launch_sched<ProdUnroll4>},
    {"warps8", &launch_sched<Warps8>},
    {"chains2", &launch_sched<Chains2>},
    {"shared", &launch_sched<SharedPh>},
    {"ch2_u4", &launch_sched<Chains2Unroll4>},
    {"mb4", &launch_sched<ProdMb4>},
    {"mb5", &launch_sched<ProdMb5>},
    {"mb6", &launch_sched<ProdMb6>},
    {"mb8", &launch_sched<ProdMb8>},
    {"mb10", &launch_sched<ProdMb10>},
    {"mb12", &launch_sched<ProdMb12>},
};

struct Cmp {
    int bit_mism = 0;
    float max_abs = 0.f;
};

Cmp compare(const std::uint16_t* a, const std::uint16_t* b, std::size_t n) {
    Cmp r;
    for (std::size_t i = 0; i < n; ++i) {
        if (a[i] != b[i]) {
            ++r.bit_mism;
            auto to_f = [](std::uint16_t v) {
                std::uint32_t u = static_cast<std::uint32_t>(v) << 16;
                float f;
                std::memcpy(&f, &u, 4);
                return f;
            };
            r.max_abs = fmaxf(r.max_abs, fabsf(to_f(a[i]) - to_f(b[i])));
        }
    }
    return r;
}

bench::ColdTiming measure_warm(LaunchFn fn, const __nv_bfloat16* x, const std::uint8_t* c,
                               const std::uint8_t* s, float inv, __nv_bfloat16* o,
                               cudaStream_t stream) {
    for (int i = 0; i < 20; ++i) fn(x, c, s, inv, o, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaEvent_t start = nullptr, stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<double> samples;
    constexpr int kBatch = 10;
    for (int i = 0; i < 100; ++i) {
        CUDA_CHECK(cudaEventRecord(start, stream));
        for (int j = 0; j < kBatch; ++j) fn(x, c, s, inv, o, stream);
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        samples.push_back(static_cast<double>(ms) * 1000.0 / kBatch);
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::sort(samples.begin(), samples.end());
    return {samples[samples.size() / 2], samples.front(), samples.back()};
}

} // namespace

int main() {
    try {
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        DeviceBuffer flush(kFlushBytes);
        DeviceBuffer input = bench::make_bf16(static_cast<std::size_t>(Geometry::kInputRows) * kT);
        DeviceBuffer output(static_cast<std::size_t>(Geometry::kOutputRows) * kT * 2);
        DeviceBuffer output_ref(output.bytes);
        auto packed         = bench::make_nvfp4_weight(Geometry::kOutputRows, Geometry::kInputRows);
        const float inverse = 1.0F / packed.weight.weight_scale_divisor;
        const auto* x       = static_cast<const __nv_bfloat16*>(input.p);
        const auto* codes   = static_cast<const std::uint8_t*>(packed.weight.qdata);
        const auto* scales  = static_cast<const std::uint8_t*>(packed.weight.scales);
        auto* out           = static_cast<__nv_bfloat16*>(output.p);
        auto* out_ref       = static_cast<__nv_bfloat16*>(output_ref.p);
        const std::size_t n_out = static_cast<std::size_t>(Geometry::kOutputRows) * kT;
        std::vector<std::uint16_t> href(n_out), hgot(n_out);

        std::printf("MLP-down T=4 dual-gate\n");
        std::printf("%-10s %6s %10s %10s %8s %8s\n", "variant", "exact", "cold_us", "warm_us",
                    "cold%", "warm%");

        double prod_cold = 0, prod_warm = 0;
        const Candidate* best = nullptr;
        double best_score     = -1e9;

        for (const auto& cand : kCandidates) {
            launch_sched<Prod>(x, codes, scales, inverse, out_ref, stream);
            cand.fn(x, codes, scales, inverse, out, stream);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaMemcpy(href.data(), out_ref, n_out * 2, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(hgot.data(), out, n_out * 2, cudaMemcpyDeviceToHost));
            const Cmp cmp = compare(href.data(), hgot.data(), n_out);
            const bool ok = cmp.bit_mism == 0;

            double cold_us = 0, warm_us = 0, cold_vs = 0, warm_vs = 0;
            if (ok) {
                const auto run = [&](cudaStream_t s) {
                    cand.fn(x, codes, scales, inverse, out, s);
                };
                cold_us =
                    bench::measure_cold_launch(run, flush, stream, 8, 40).median_us;
                warm_us = measure_warm(cand.fn, x, codes, scales, inverse, out, stream).median_us;
                if (std::string_view(cand.name) == "prod") {
                    prod_cold = cold_us;
                    prod_warm = warm_us;
                }
                cold_vs = prod_cold > 0 ? 100.0 * (prod_cold - cold_us) / prod_cold : 0;
                warm_vs = prod_warm > 0 ? 100.0 * (prod_warm - warm_us) / prod_warm : 0;
            } else {
                std::printf("%-10s FAIL mism=%d max_abs=%.4f\n", cand.name, cmp.bit_mism,
                            cmp.max_abs);
                continue;
            }
            std::printf("%-10s %6s %10.3f %10.3f %+7.2f %+7.2f%s\n", cand.name, "PASS", cold_us,
                        warm_us, cold_vs, warm_vs,
                        (cold_vs >= 1.0 && warm_vs >= 1.0) ? "  <<WIN"
                        : (cold_vs >= 1.0)                 ? "  (cold-only)"
                                                           : "");
            if (cold_vs >= 1.0 && warm_vs >= 1.0) {
                const double score = std::min(cold_vs, warm_vs);
                if (score > best_score) {
                    best_score = score;
                    best       = &cand;
                }
            }
        }

        if (best)
            std::printf("\nWINNER: %s\n", best->name);
        else
            std::printf("\nNO dual-gate winner\n");

        CUDA_CHECK(cudaStreamDestroy(stream));
        return best ? 0 : 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
