// Isolated A/B bench for the TMA W4A4 (prefill) GEMM schedule variants.
//
// Drives nvfp4_w4a4_tma_kernel DIRECTLY (bypassing the production dispatch) so every
// schedule variant runs in one process on the same deterministic input/weight. Each variant
// is conformance-gated against the production schedule (m256s3) and timed (warm: L2-hot
// weight). The A/B question: does a BlockM=128 / higher-occupancy / deeper-pipeline schedule
// beat the production m256s3 (occ=1, ~79 KB smem) at the production M=4096 prefill chunk?

#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_w4a4_tma.cuh"
#include "ops/linear/nvfp4/nvfp4_output.cuh"

#include "core/device.h"
#include "ninfer_bench_common.h"
#include "quantized_weight.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace ninfer;
using namespace ninfer::ops::detail;

namespace {

// GDN in-proj shape (N=16384, K=5120) — the clean single GEMM that plateaued at ~1100 TF/s.
using Geom    = Nvfp4GdnInputGeometry;
using Act5120 = Nvfp4Activation5120Geometry;

// Schedule variants (Nvfp4W4a4TmaSchedule<BlockM, Stages, MinBlocksPerSm>):
using M256S3   = Nvfp4W4a4TmaSchedule<256, 3, 1>;  // production
using M256S2   = Nvfp4W4a4TmaSchedule<256, 2, 1>;
using M128S2O1 = Nvfp4W4a4TmaSchedule<128, 2, 1>;
using M128S3O1 = Nvfp4W4a4TmaSchedule<128, 3, 1>;
using M128S4O1 = Nvfp4W4a4TmaSchedule<128, 4, 1>;
using M128S2O2 = Nvfp4W4a4TmaSchedule<128, 2, 2>;
// Consumer-warp-count variant (WarpsN=4 -> 16 consumer warps): shrinks the per-warp
// accumulator (kMmaN 8->4). (32 consumer warps would need 1152 threads/CTA > the 1024 limit.)
using M256C16  = Nvfp4W4a4TmaSchedule<256, 3, 1, 4, 4>;  // 16 consumer warps (64 accum)
using M128C16  = Nvfp4W4a4TmaSchedule<128, 3, 1, 4, 4>;  // 16 consumer warps (32 accum)

// Deterministic bf16 fill (same index -> same value across processes).
__global__ void fill_input_bf16(__nv_bfloat16* x, std::int64_t n) {
  const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) { return; }
  const std::uint32_t s = static_cast<std::uint32_t>(i * 2654435761ULL + 0x9E3779B9u);
  const float v         = (static_cast<float>(static_cast<std::uint32_t>(s & 0xFFFFFFu)) /
                     16777215.0F -
                     0.5F) *
                     0.2F;  // ~[-0.1, 0.1]
  x[i] = __float2bfloat16(v);
}

struct Ctx {
  const std::uint8_t* a_codes;
  const std::uint8_t* a_scales;
  const std::uint8_t* w_codes;
  const std::uint8_t* w_scales;
  __nv_bfloat16* out;
  std::int32_t tokens;
  float alpha;
  cudaStream_t stream;
};

// One TMA launch for a given schedule (descriptor build + smem attr + kernel launch).
template <class Schedule>
void run_once(const Ctx& c) {
  using Epi = Nvfp4IdentityEpilogue;
  using Out = Nvfp4ContiguousOutput;
  const auto descriptors = make_nvfp4_w4a4_tma_descriptors<Geom, Schedule::kBlockM>(
      c.a_codes, c.a_scales, c.w_codes, c.w_scales, c.tokens);
  constexpr std::size_t kShared = sizeof(Nvfp4W4a4TmaSharedStorage<Schedule>);
  auto* kernel                  = &nvfp4_w4a4_tma_kernel<Geom, Schedule, Epi, Out>;
  static const bool configured  =
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                           static_cast<int>(kShared)) == cudaSuccess;
  (void)configured;
  dim3 grid(Geom::kOutputRows / Schedule::kBlockN, c.tokens / Schedule::kBlockM);
  kernel<<<grid, Schedule::kThreads, kShared, c.stream>>>(descriptors, c.alpha, Epi{},
                                                          Out{c.out, Geom::kOutputRows});
  CUDA_CHECK(cudaGetLastError());  // a silently-failed launch (e.g. too many threads) would otherwise leave stale output -> false maxdiff=0
}

// Time a schedule (warm, L2-hot) via the shared measure_launch (warmup + median of repeat).
template <class Schedule>
bench::ColdTiming time_variant(const Ctx& c, int warmup, int repeat) {
  auto launch = [&](cudaStream_t s) {
    Ctx cs = c;
    cs.stream = s;
    run_once<Schedule>(cs);
  };
  return bench::measure_launch(launch, c.stream, warmup, repeat);
}

// In-GPU max abs diff (two-stage: per-block max -> global) so we avoid a 100s-of-MB D2H per
// variant. `per_block` is a >=1024-float device scratch; `out_diff` is one float.
__global__ void block_max_abs_diff(const __nv_bfloat16* a, const __nv_bfloat16* b, std::size_t n,
                                   float* per_block) {
  const std::size_t stride = (std::size_t)gridDim.x * blockDim.x;
  float local = 0.f;
  for (std::size_t i = (std::size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    local = fmaxf(local, std::fabs(__bfloat162float(a[i]) - __bfloat162float(b[i])));
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) { local = fmaxf(local, __shfl_xor_sync(0xffffffffu, local, o)); }
  __shared__ float red[32];
  if ((threadIdx.x & 31) == 0) { red[threadIdx.x >> 5] = local; }
  __syncthreads();
  if (threadIdx.x == 0) {
    float m = 0.f;
    for (int b = 0; b < (int)(blockDim.x >> 5); ++b) { m = fmaxf(m, red[b]); }
    per_block[blockIdx.x] = m;
  }
}
__global__ void reduce_block_max(const float* per_block, int nb, float* out) {
  float local = 0.f;
  for (int i = 0; i < nb; ++i) { local = fmaxf(local, per_block[i]); }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) { local = fmaxf(local, __shfl_xor_sync(0xffffffffu, local, o)); }
  __shared__ float red[32];
  if ((threadIdx.x & 31) == 0) { red[threadIdx.x >> 5] = local; }
  __syncthreads();
  if (threadIdx.x == 0) {
    float m = 0.f;
    for (int b = 0; b < (int)(blockDim.x >> 5); ++b) { m = fmaxf(m, red[b]); }
    *out = m;
  }
}
double maxdiff_gpu(const __nv_bfloat16* ref, const __nv_bfloat16* out, std::size_t count,
                   float* per_block, float* out_diff, cudaStream_t stream) {
  const int nb = (int)std::min<std::size_t>(1024, (count + 255) / 256);
  block_max_abs_diff<<<nb, 256, 0, stream>>>(ref, out, count, per_block);
  reduce_block_max<<<1, 256, 0, stream>>>(per_block, nb, out_diff);
  float d = 0.f;
  CUDA_CHECK(cudaMemcpyAsync(&d, out_diff, sizeof(float), cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return static_cast<double>(d);
}

void print_variant(const char* name, bench::ColdTiming t, double max_diff, bool is_prod,
                   std::int32_t tokens, std::int32_t rows, std::int32_t cols) {
  const double seconds = t.median_us * 1.0e-6;
  const double flops   = 2.0 * static_cast<double>(rows) * cols * tokens;
  std::printf("%-12s %s median=%10.3f us min=%10.3f us p95=%10.3f us math=%8.2f TFLOP/s "
              "maxdiff=%g\n",
              name, is_prod ? "(prod)" : "", t.median_us, t.min_us, t.p95_us,
              flops / seconds / 1.0e12, max_diff);
  std::fflush(stdout);
}

// Time + conformance-check one schedule against the prod reference (in-GPU maxdiff).
template <class Schedule>
double run_variant(const Ctx& c, const char* name, const __nv_bfloat16* ref_dev, float* per_block,
                   float* out_diff, int warmup, int repeat, std::int32_t rows, std::int32_t cols) {
  const bench::ColdTiming t = time_variant<Schedule>(c, warmup, repeat);
  // Sentinel-invalidate the output so a no-op/failed launch cannot fake a maxdiff==0 pass.
  CUDA_CHECK(cudaMemsetAsync(c.out, 0xff, static_cast<std::size_t>(rows) * c.tokens * 2, c.stream));
  run_once<Schedule>(c);  // final launch leaves `out` holding this schedule's result
  const double d =
      maxdiff_gpu(ref_dev, c.out, static_cast<std::size_t>(rows) * c.tokens, per_block, out_diff,
                  c.stream);
  print_variant(name, t, d, false, c.tokens, rows, cols);
  return t.median_us;
}

} // namespace

int main(int argc, char** argv) {
  try {
    std::int32_t tokens = 4096;
    int warmup          = 5;
    int repeat          = 30;
    std::string only;
    for (int i = 1; i < argc; ++i) {
      if (std::strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
        tokens = static_cast<std::int32_t>(std::strtol(argv[++i], nullptr, 10));
      } else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
        warmup = std::strtol(argv[++i], nullptr, 10);
      } else if (std::strcmp(argv[i], "--repeat") == 0 && i + 1 < argc) {
        repeat = std::strtol(argv[++i], nullptr, 10);
      } else if (std::strcmp(argv[i], "--only") == 0 && i + 1 < argc) {
        only = argv[++i];
      }
    }
    if (tokens % 256 != 0) {
      std::fprintf(stderr, "tokens must be a multiple of 256 (TMA path); got %d\n", tokens);
      return 2;
    }

    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
      std::printf("SKIP: no usable CUDA device\n");
      return 0;
    }
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    const std::int32_t rows = Geom::kOutputRows;  // 16384
    const std::int32_t cols = Geom::kInputRows;   // 5120

    bench::PackedQuantizedWeight packed = bench::make_nvfp4_weight(rows, cols);
    Weight& weight = packed.weight;
    const float input_div  = weight.input_scale_divisor;   // 3.5
    const float weight_div = weight.weight_scale_divisor;  // 0.125
    const float alpha      = 1.0F / (input_div * weight_div);

    DeviceBuffer x(static_cast<std::size_t>(cols) * tokens * 2);
    fill_input_bf16<<<(static_cast<std::size_t>(cols) * tokens + 255) / 256, 256, 0, stream>>>(
        (__nv_bfloat16*)x.p, static_cast<std::int64_t>(cols) * tokens);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    DeviceBuffer a_codes(static_cast<std::size_t>(tokens) * (cols / 2));
    DeviceBuffer a_scales(static_cast<std::size_t>(tokens) * (cols / 16));
    {
      constexpr int kThreads = 256;
      const std::int32_t kGroups = cols / 16;
      const std::int32_t grid_x  = (tokens * kGroups + kThreads - 1) / kThreads;
      nvfp4_w4a4_quantize_kernel<Act5120, 256><<<grid_x, kThreads, 0, stream>>>(
          (__nv_bfloat16*)x.p, (std::uint8_t*)a_codes.p, (std::uint8_t*)a_scales.p, tokens,
          input_div);
      CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    DeviceBuffer out(static_cast<std::size_t>(rows) * tokens * 2);
    Ctx ctx{(const std::uint8_t*)a_codes.p, (const std::uint8_t*)a_scales.p,
            (const std::uint8_t*)weight.qdata, (const std::uint8_t*)weight.scales,
            (__nv_bfloat16*)out.p, tokens, alpha, stream};

    // Device-side reference + scratch for in-GPU maxdiff.
    DeviceBuffer ref_dev(static_cast<std::size_t>(rows) * tokens * 2);
    DeviceBuffer per_block(1024 * 4);
    DeviceBuffer out_diff(4);

    // Production reference (m256s3) — the conformance gate.
    const bench::ColdTiming prod = time_variant<M256S3>(ctx, warmup, repeat);
    run_once<M256S3>(ctx);
    CUDA_CHECK(cudaMemcpy(ref_dev.p, out.p, static_cast<std::size_t>(rows) * tokens * 2,
                          cudaMemcpyDeviceToDevice));
    std::printf("[nvfp4-tma-ab] tokens=%d rows=%d cols=%d alpha=%g\n", tokens, rows, cols, alpha);
    print_variant("m256s3", prod, 0.0, true, tokens, rows, cols);

    // A/B variants (conformance vs prod + timing). maxdiff==0 => bit-identical => correct.
    const __nv_bfloat16* refp = static_cast<const __nv_bfloat16*>(ref_dev.p);
    float* pb                 = static_cast<float*>(per_block.p);
    float* od                 = static_cast<float*>(out_diff.p);
    if (only.empty() || only == "m256s2")
      run_variant<M256S2>(ctx, "m256s2", refp, pb, od, warmup, repeat, rows, cols);
    if (only.empty() || only == "m128s2o1")
      run_variant<M128S2O1>(ctx, "m128s2o1", refp, pb, od, warmup, repeat, rows, cols);
    if (only.empty() || only == "m128s3o1")
      run_variant<M128S3O1>(ctx, "m128s3o1", refp, pb, od, warmup, repeat, rows, cols);
    if (only.empty() || only == "m128s4o1")
      run_variant<M128S4O1>(ctx, "m128s4o1", refp, pb, od, warmup, repeat, rows, cols);
    if (only.empty() || only == "m128s2o2")
      run_variant<M128S2O2>(ctx, "m128s2o2", refp, pb, od, warmup, repeat, rows, cols);
    if (only.empty() || only == "m256c16")
      run_variant<M256C16>(ctx, "m256c16", refp, pb, od, warmup, repeat, rows, cols);
    if (only.empty() || only == "m128c16")
      run_variant<M128C16>(ctx, "m128c16", refp, pb, od, warmup, repeat, rows, cols);
    std::fflush(stdout);

    std::printf("[nvfp4-tma-ab] done (maxdiff==0 = bit-identical to prod = correct).\n");
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "ninfer_nvfp4_tma_ab_bench: %s\n", error.what());
    return 1;
  }
}