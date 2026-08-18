// Dual-gate T=4 W4A4 SwiGLU tile/stage sweep vs production M48N64/s2/EvictFirst.
// Question: can we close the 82µs → ~56µs DRAM-floor gap (≈10% e2e if fully closed)?
#include "core/device.h"
#include "ninfer/ops/linear.h"
#include "ninfer_bench_common.h"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_output.cuh"
#include "ops/linear/nvfp4/nvfp4_w4a4_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"
#include "ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.h"
#include "quantized_weight.cuh"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdio>
#include <string_view>
#include <vector>

using namespace ninfer;
using namespace ninfer::ops;
using namespace ninfer::ops::detail;

namespace {

constexpr int kT                   = 4;
constexpr std::int32_t kGateUp     = 34816;
constexpr std::int32_t kInter      = 17408;
constexpr std::int32_t kHidden     = 5120;
constexpr std::int32_t kDownOut    = 5120;
constexpr std::size_t kFlushBytes  = 256ULL << 20;
constexpr double kWeightBytes      = 34816.0 * 5120.0 * 0.5625;
constexpr double kDramFloorUs      = kWeightBytes / (bench::kRooflineGBs * 1.0e3);

using Geometry = Nvfp4MlpGateUpGeometry;
using Prod     = Nvfp4W4a4MmaSchedule<48, 64, 256, 3, 4, 2, 2>;
using Stages3  = Nvfp4W4a4MmaSchedule<48, 64, 256, 3, 4, 3, 2>;
using Stages4  = Nvfp4W4a4MmaSchedule<48, 64, 256, 3, 4, 4, 2>;
using M16N64   = Nvfp4W4a4MmaSchedule<16, 64, 256, 1, 4, 2, 4>;
using M16N32   = Nvfp4W4a4MmaSchedule<16, 32, 256, 1, 4, 2, 4>;
using M32N64   = Nvfp4W4a4MmaSchedule<32, 64, 256, 2, 4, 2, 2>;
// MinBlocksPerSm register-cap sweep on the production tile (384 thr/block):
// 65536/(384*mb) -> m1 no cap, m3 caps ~56 regs (3 blocks/SM), m4 caps ~42 regs (4 blocks/SM).
using M48N64m1 = Nvfp4W4a4MmaSchedule<48, 64, 256, 3, 4, 2, 1>;
using M48N64m3 = Nvfp4W4a4MmaSchedule<48, 64, 256, 3, 4, 2, 3>;
using M48N64m4 = Nvfp4W4a4MmaSchedule<48, 64, 256, 3, 4, 2, 4>;

template <class Schedule>
struct SwiGluRows {
    static constexpr bool kContiguous   = false;
    static constexpr int kRowsPerBranch = Schedule::kBlockN / 2;

    __device__ __forceinline__ int weight_row(int row_begin, int local_row) const {
        return row_begin + (local_row & (kRowsPerBranch - 1)) +
               (local_row >= kRowsPerBranch ? kInter : 0);
    }
};

union SwiGluBf16Pair {
    unsigned bits;
    __nv_bfloat162 values;
};

struct SwiGluOutput {
    __nv_bfloat16* data;

    __device__ __forceinline__ unsigned combine(unsigned gate_bits, unsigned up_bits) const {
        SwiGluBf16Pair gate{gate_bits};
        SwiGluBf16Pair up{up_bits};
        const float2 gate_values = __bfloat1622float2(gate.values);
        const float2 up_values   = __bfloat1622float2(up.values);
        SwiGluBf16Pair result;
        result.values = __floats2bfloat162_rn(silu(gate_values.x) * up_values.x,
                                              silu(gate_values.y) * up_values.y);
        return result.bits;
    }

    __device__ __forceinline__ void store_pair_vector(std::int32_t row, std::int32_t token,
                                                      uint4 gate, uint4 up) const {
        const uint4 values = make_uint4(combine(gate.x, up.x), combine(gate.y, up.y),
                                        combine(gate.z, up.z), combine(gate.w, up.w));
        store_vec(data + static_cast<std::int64_t>(token) * kInter + row, values);
    }
};

template <class Schedule, Cache WeightCache = Cache::cg>
void launch_gemm(const Weight& weight, Tensor& out, Nvfp4W4a4Workspace workspace,
                 cudaStream_t stream) {
    constexpr int kPairRows = Schedule::kBlockN / 2;
    const dim3 grid(kInter / kPairRows, (kT + Schedule::kBlockM - 1) / Schedule::kBlockM);
    const Nvfp4W4a4MaterializedActivation activation{workspace.codes, workspace.scales};
    const SwiGluRows<Schedule> row_policy{};
    const SwiGluOutput output{static_cast<__nv_bfloat16*>(out.data)};
    const float alpha = 1.0F / (weight.input_scale_divisor * weight.weight_scale_divisor);
    nvfp4_w4a4_mma_kernel<Geometry, Schedule, Nvfp4IdentityEpilogue, SwiGluOutput,
                          SwiGluRows<Schedule>, true, WeightCache>
        <<<grid, Schedule::kThreads, 0, stream>>>(activation,
                                                   static_cast<const std::uint8_t*>(weight.qdata),
                                                   static_cast<const std::uint8_t*>(weight.scales),
                                                   kT, alpha, Nvfp4IdentityEpilogue{}, output,
                                                   row_policy);
    CUDA_CHECK(cudaGetLastError());
}

using LaunchFn = void (*)(const Tensor&, const Weight&, Tensor&, Nvfp4W4a4Workspace, cudaStream_t);

template <class Schedule>
void launch_named(const Tensor& x, const Weight& weight, Tensor& out, Nvfp4W4a4Workspace scratch,
                  cudaStream_t stream) {
    (void)x;
    launch_gemm<Schedule>(weight, out, scratch, stream);
}

double median_warm(auto&& fn, cudaStream_t stream) {
    for (int i = 0; i < 20; ++i) fn(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaEvent_t a = nullptr, b = nullptr;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    std::vector<double> samples;
    for (int i = 0; i < 40; ++i) {
        CUDA_CHECK(cudaEventRecord(a, stream));
        for (int j = 0; j < 8; ++j) fn(stream);
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
        cudaDeviceReset();
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        DeviceBuffer flush(kFlushBytes);
        DeviceBuffer xin = bench::make_bf16(static_cast<std::size_t>(kHidden) * kT);
        DeviceBuffer mid(static_cast<std::size_t>(kInter) * kT * 2);
        DeviceBuffer down_out(static_cast<std::size_t>(kDownOut) * kT * 2);
        auto gate = bench::make_nvfp4_weight(kGateUp, kHidden);
        auto down = bench::make_nvfp4_weight(kDownOut, kInter);
        WorkspaceArena ws_w4(std::max(nvfp4_w4a4_workspace_capacity_bytes(kT, kHidden),
                                      static_cast<std::size_t>(256)));
        WorkspaceArena ws_prod(std::max(
            nvfp4_linear_swiglu_workspace_capacity_bytes(ops::LinearPolicy::AllowA4, kT, kT),
            static_cast<std::size_t>(256)));
        WorkspaceArena ws_dn(std::max(
            ops::linear_workspace_capacity_bytes(QType::NVFP4, kDownOut, kInter,
                                                 ops::LinearPolicy::A16Only, kT, kT),
            static_cast<std::size_t>(256)));

        Tensor x(xin.p, DType::BF16, {kHidden, kT});
        Tensor m(mid.p, DType::BF16, {kInter, kT});
        Tensor o(down_out.p, DType::BF16, {kDownOut, kT});

        struct Cand {
            const char* name;
            bool production_ef;
            LaunchFn gemm;
        };
        const Cand cands[] = {
            {"prod_M48N64_s2_ef", true, nullptr},
            {"M48N64_s2_cg", false, &launch_named<Prod>},
            {"M48N64_s3_cg", false, &launch_named<Stages3>},
            {"M48N64_s4_cg", false, &launch_named<Stages4>},
            {"M16N64_s2_cg", false, &launch_named<M16N64>},
            {"M16N32_s2_cg", false, &launch_named<M16N32>},
            {"M32N64_s2_cg", false, &launch_named<M32N64>},
            {"M48N64_m1_cg", false, &launch_named<M48N64m1>},
            {"M48N64_m3_cg", false, &launch_named<M48N64m3>},
            {"M48N64_m4_cg", false, &launch_named<M48N64m4>},
        };

        std::printf("W4A4 SwiGLU T=4 tile/stage sweep  (DRAM floor %.1f us, %.0f MB)\n",
                    kDramFloorUs, kWeightBytes / 1.0e6);
        std::printf("%-20s %9s %9s %8s %8s %9s %9s %8s %8s\n", "variant", "sw_c", "sw_w",
                    "sw_c%", "sw_w%", "pair_c", "pair_w", "p_c%", "p_w%");

        double base_sw_c = 0, base_sw_w = 0, base_p_c = 0, base_p_w = 0;
        for (const auto& c : cands) {
            auto sw = [&](cudaStream_t s) {
                if (c.production_ef) {
                    nvfp4_linear_swiglu_w4a4_launch(x, gate.weight, m, ws_prod, s);
                    return;
                }
                auto scope         = ws_w4.scope();
                const auto scratch = allocate_nvfp4_w4a4_workspace(ws_w4, kT, kHidden);
                launch_nvfp4_w4a4_quantize(x, gate.weight, scratch, s);
                c.gemm(x, gate.weight, m, scratch, s);
            };
            auto pair = [&](cudaStream_t s) {
                sw(s);
                ops::linear(m, down.weight, o, ops::LinearPolicy::A16Only, ws_dn, s);
            };

            const double sw_c = bench::measure_cold_launch(sw, flush, stream, 4, 20).median_us;
            const double sw_w = median_warm(sw, stream);
            const double p_c  = bench::measure_cold_launch(pair, flush, stream, 4, 20).median_us;
            const double p_w  = median_warm(pair, stream);
            if (std::string_view(c.name) == "prod_M48N64_s2_ef") {
                base_sw_c = sw_c;
                base_sw_w = sw_w;
                base_p_c  = p_c;
                base_p_w  = p_w;
            }
            const auto pct = [](double base, double v) { return 100.0 * (base - v) / base; };
            const bool win = pct(base_p_c, p_c) >= 1.0 && pct(base_p_w, p_w) >= 1.0;
            std::printf("%-20s %9.2f %9.2f %+7.1f %+7.1f %9.2f %9.2f %+7.1f %+7.1f%s\n", c.name,
                        sw_c, sw_w, pct(base_sw_c, sw_c), pct(base_sw_w, sw_w), p_c, p_w,
                        pct(base_p_c, p_c), pct(base_p_w, p_w), win ? "  <<WIN" : "");
        }
        std::printf("10%% e2e needs pair ~-20%% (SwiGLU 82→56 us). Dual-gate: both pair cold+warm "
                    ">= +1%%.\n");
        CUDA_CHECK(cudaStreamDestroy(stream));
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
