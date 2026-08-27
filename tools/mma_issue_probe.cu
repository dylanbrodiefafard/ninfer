// Isolated warp-MMA issue-rate probe for the production atoms in ops/common/mma.cuh.
//
// This is not an Op. It times register-only mma.sync loops so Layer 0 can set t_issue.
//
// Build (CMake, with NINFER_BUILD_BENCHMARKS=ON):
//   cmake --build build --target ninfer_mma_issue_probe
// Standalone:
//   nvcc -O3 -std=c++20 -arch=sm_120a -I src tools/mma_issue_probe.cu -o mma_issue_probe
//
//   ./mma_issue_probe
//   ./mma_issue_probe --atom nvfp4 --json

#include "ops/common/mma.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CUDA_CHECK(expr)                                                                           \
    do {                                                                                           \
        const cudaError_t error__ = (expr);                                                        \
        if (error__ != cudaSuccess) {                                                              \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,                  \
                         cudaGetErrorString(error__));                                             \
            std::exit(EXIT_FAILURE);                                                               \
        }                                                                                          \
    } while (false)

namespace {

constexpr int kInner    = 16;
constexpr int kDefaultIters = 8192;
constexpr int kWarmup   = 5;
constexpr int kTrials   = 7;
constexpr double kDenseFp4TflopS = 1676.0;

enum class Atom : int { Nvfp4, Bf16, S8 };

struct Options {
    Atom atom      = Atom::Nvfp4;
    bool all       = true;
    bool json      = false;
    int iters      = kDefaultIters;
    int warps      = 8;
    int blocks_per_sm = 2;
};

__device__ __forceinline__ void issue_nvfp4(int iters, float& c0, float& c1, float& c2, float& c3) {
    unsigned a0 = 0x11111111u, a1 = 0x22222222u, a2 = 0x33333333u, a3 = 0x44444444u;
    unsigned b0 = 0x55555555u, b1 = 0x66666666u;
    unsigned sfa = 0x38383838u, sfb = 0x38383838u;
    for (int i = 0; i < iters; ++i) {
#pragma unroll
        for (int u = 0; u < kInner; ++u) {
            ninfer::ops::mma_nvfp4_e4m3(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1, sfa, sfb);
        }
    }
}

__device__ __forceinline__ void issue_bf16(int iters, float& c0, float& c1, float& c2, float& c3) {
    unsigned a0 = 0x3c003c00u, a1 = 0x3c003c00u, a2 = 0x3c003c00u, a3 = 0x3c003c00u;
    unsigned b0 = 0x3c003c00u, b1 = 0x3c003c00u;
    for (int i = 0; i < iters; ++i) {
#pragma unroll
        for (int u = 0; u < kInner; ++u) { ninfer::ops::mma_bf16(c0, c1, c2, c3, a0, a1, a2, a3, b0, b1); }
    }
}

__device__ __forceinline__ void issue_s8(int iters, float& c0, float& c1, float& c2, float& c3) {
    int ic0 = 0, ic1 = 0, ic2 = 0, ic3 = 0;
    unsigned a0 = 0x11111111u, a1 = 0x22222222u, a2 = 0x33333333u, a3 = 0x44444444u;
    unsigned b0 = 0x55555555u, b1 = 0x66666666u;
    for (int i = 0; i < iters; ++i) {
#pragma unroll
        for (int u = 0; u < kInner; ++u) { ninfer::ops::mma_s8(ic0, ic1, ic2, ic3, a0, a1, a2, a3, b0, b1); }
    }
    c0 = static_cast<float>(ic0);
    c1 = static_cast<float>(ic1);
    c2 = static_cast<float>(ic2);
    c3 = static_cast<float>(ic3);
}

template <Atom kAtom>
__global__ void mma_issue_kernel(float* sink, int iters) {
    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
    if constexpr (kAtom == Atom::Nvfp4) {
        issue_nvfp4(iters, c0, c1, c2, c3);
    } else if constexpr (kAtom == Atom::Bf16) {
        issue_bf16(iters, c0, c1, c2, c3);
    } else {
        issue_s8(iters, c0, c1, c2, c3);
    }
    if ((threadIdx.x & 31) == 0) { atomicAdd(sink, c0 + c1 + c2 + c3); }
}

struct Result {
    const char* name;
    int m;
    int n;
    int k;
    double mma_per_s;
    double tflop_s;
    double median_ms;
    int warps;
    int blocks;
};

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

template <Atom kAtom>
Result run_atom(const char* name, int m, int n, int k, const Options& opt, int sm_count) {
    const int threads = opt.warps * 32;
    const int blocks  = sm_count * opt.blocks_per_sm;
    const int warps   = blocks * opt.warps;
    float* sink       = nullptr;
    CUDA_CHECK(cudaMalloc(&sink, sizeof(float)));

    auto launch = [&]() {
        CUDA_CHECK(cudaMemset(sink, 0, sizeof(float)));
        mma_issue_kernel<kAtom><<<blocks, threads>>>(sink, opt.iters);
        CUDA_CHECK(cudaGetLastError());
    };

    for (int i = 0; i < kWarmup; ++i) { launch(); }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<double> ms;
    ms.reserve(kTrials);
    for (int trial = 0; trial < kTrials; ++trial) {
        CUDA_CHECK(cudaEventRecord(start));
        launch();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
        ms.push_back(static_cast<double>(elapsed));
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(sink));

    const double median_ms     = median(ms);
    const double seconds       = median_ms * 1e-3;
    const double mma_total     = static_cast<double>(warps) * opt.iters * kInner;
    const double mma_per_s     = mma_total / seconds;
    const double flops_per_mma = 2.0 * m * n * k;
    const double tflop_s       = mma_per_s * flops_per_mma / 1e12;
    return Result{name, m, n, k, mma_per_s, tflop_s, median_ms, opt.warps, blocks};
}

void print_usage(const char* argv0) {
    std::fprintf(stderr,
                 "usage: %s [--atom nvfp4|bf16|s8|all] [--iters N] [--warps W] "
                 "[--blocks-per-sm B] [--json]\n",
                 argv0);
}

Options parse(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next             = [&]() -> const char* {
            if (i + 1 >= argc) {
                print_usage(argv[0]);
                std::exit(2);
            }
            return argv[++i];
        };
        if (arg == "--atom") {
            const std::string value = next();
            opt.all                 = false;
            if (value == "nvfp4") {
                opt.atom = Atom::Nvfp4;
            } else if (value == "bf16") {
                opt.atom = Atom::Bf16;
            } else if (value == "s8") {
                opt.atom = Atom::S8;
            } else if (value == "all") {
                opt.all = true;
            } else {
                print_usage(argv[0]);
                std::exit(2);
            }
        } else if (arg == "--iters") {
            opt.iters = std::atoi(next());
        } else if (arg == "--warps") {
            opt.warps = std::atoi(next());
        } else if (arg == "--blocks-per-sm") {
            opt.blocks_per_sm = std::atoi(next());
        } else if (arg == "--json") {
            opt.json = true;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            print_usage(argv[0]);
            std::exit(2);
        }
    }
    if (opt.iters <= 0 || opt.warps <= 0 || opt.blocks_per_sm <= 0) {
        std::fprintf(stderr, "iters, warps, and blocks-per-sm must be positive\n");
        std::exit(2);
    }
    return opt;
}

} // namespace

int main(int argc, char** argv) {
    const Options opt = parse(argc, argv);
    int sm_count      = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::vector<Result> results;
    auto want = [&](Atom atom) { return opt.all || opt.atom == atom; };
    if (want(Atom::Nvfp4)) {
        results.push_back(run_atom<Atom::Nvfp4>("nvfp4", 16, 8, 64, opt, sm_count));
    }
    if (want(Atom::Bf16)) {
        results.push_back(run_atom<Atom::Bf16>("bf16", 16, 8, 16, opt, sm_count));
    }
    if (want(Atom::S8)) { results.push_back(run_atom<Atom::S8>("s8", 16, 8, 32, opt, sm_count)); }

    if (opt.json) {
        std::printf("{\n");
        std::printf("  \"device\": \"%s\",\n", prop.name);
        std::printf("  \"sm_count\": %d,\n", sm_count);
        std::printf("  \"dense_fp4_tflop_s\": %.1f,\n", kDenseFp4TflopS);
        std::printf("  \"iters\": %d,\n", opt.iters);
        std::printf("  \"inner\": %d,\n", kInner);
        for (std::size_t i = 0; i < results.size(); ++i) {
            const Result& r = results[i];
            std::printf("  \"%s\": {\"m\": %d, \"n\": %d, \"k\": %d, \"mma_per_s\": %.6e, "
                        "\"tflop_s\": %.3f, \"median_ms\": %.4f, \"warps_per_block\": %d, "
                        "\"blocks\": %d}%s\n",
                        r.name, r.m, r.n, r.k, r.mma_per_s, r.tflop_s, r.median_ms, r.warps, r.blocks,
                        i + 1 == results.size() ? "" : ",");
        }
        std::printf("}\n");
        return 0;
    }

    std::printf("mma_issue_probe  device=%s  SMs=%d  warps/block=%d  blocks/SM=%d  iters=%d*%d\n",
                prop.name, sm_count, opt.warps, opt.blocks_per_sm, opt.iters, kInner);
    std::printf("%-8s  %6s  %12s  %10s  %8s  %s\n", "atom", "shape", "MMA/s", "TFLOP/s", "%FP4",
                "median");
    for (const Result& r : results) {
        const double pct = 100.0 * r.tflop_s / kDenseFp4TflopS;
        std::printf("%-8s  m%dn%dk%d  %12.4e  %10.2f  %7.1f  %.3f ms\n", r.name, r.m, r.n, r.k,
                    r.mma_per_s, r.tflop_s, pct, r.median_ms);
    }
    std::printf("Use NVFP4 mma_per_s as --mma-per-s / profiles/kdev/mma_issue.json for t_issue.\n");
    return 0;
}
