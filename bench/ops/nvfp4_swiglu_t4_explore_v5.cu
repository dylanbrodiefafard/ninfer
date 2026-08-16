// SwiGLU T=4 explore v5: bit-exact + cold AND warm(+graph) gates.
// A candidate only "wins" if bit-exact and >= +1% on BOTH cold and warm.
// Cold-only wins have already failed e2e (reg_pipe2 / full_unroll).

#include "core/device.h"
#include "ninfer_bench_common.h"
#include "ops/common/math.cuh"
#include "ops/common/memory.cuh"
#include "ops/common/warp.cuh"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_gemv.cuh"
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

constexpr int kT            = 4;
constexpr int kGateUpRows   = Nvfp4MlpGateUpGeometry::kOutputRows;
constexpr int kIntermediate = kGateUpRows / 2;
constexpr int kHidden       = Nvfp4MlpGateUpGeometry::kInputRows;
constexpr std::size_t kFlushBytes = 256ULL << 20;
constexpr int kColdWarmup = 8;
constexpr int kColdRepeat = 40;
constexpr int kWarmWarmup = 20;
constexpr int kWarmRepeat = 200;

using Geometry = Nvfp4MlpGateUpGeometry;
using Schedule = Nvfp4SmallTSchedule<8, 1, 2, 16, kT, 1, Nvfp4SmallTActivationAccess::SharedPhase,
                                     Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                                     Nvfp4SmallTBlockOrder::RowsContiguous, 1>;
using ScheduleTail =
    Nvfp4SmallTSchedule<8, 1, 2, 16, kT, 1, Nvfp4SmallTActivationAccess::SharedPhase,
                        Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 1,
                        Nvfp4SmallTBlockOrder::RowsContiguous, 1>;

constexpr int kValuesPerPhase = 512;
constexpr int kPhases         = 10;
constexpr int kPacksPerToken  = kValuesPerPhase / 8;
constexpr int kStagePacks     = kT * kPacksPerToken;
constexpr int kBlocks         = kIntermediate / 8;

__device__ __forceinline__ void
resolve_rows(int& gate_row, int (&parent_rows)[2], int& lane, int& warp) {
    constexpr int kCtasPerM128 = 128 / Schedule::kWarpsPerCta;
    const int block            = static_cast<int>(blockIdx.x);
    const int m_tile           = block / kCtasPerM128;
    const int cta_in_tile      = block - m_tile * kCtasPerM128;
    lane                       = static_cast<int>(threadIdx.x) & 31;
    warp                       = static_cast<int>(threadIdx.x) >> 5;
    const int flat_pair        = cta_in_tile * Schedule::kWarpsPerCta + warp;
    gate_row                   = m_tile * 128 + (flat_pair >> 2) + (flat_pair & 3) * 32;
    parent_rows[0]             = gate_row;
    parent_rows[1]             = gate_row + kIntermediate;
}

__device__ __forceinline__ std::int64_t code_off(int parent_row, int phase, int lane) {
    return static_cast<std::int64_t>(parent_row) * Geometry::kCodeBytesPerRow +
           phase * (kValuesPerPhase / 2) + lane * 8;
}

__device__ __forceinline__ void stage_act(const __nv_bfloat16* x, int phase, __nv_bfloat16* smem) {
    auto* dst = reinterpret_cast<uint4*>(smem);
    for (int task = static_cast<int>(threadIdx.x); task < kStagePacks; task += 256) {
        const int tok  = task / kPacksPerToken;
        const int pack = task - tok * kPacksPerToken;
        dst[task] =
            load_vec<uint4>(x + static_cast<std::int64_t>(tok) * kHidden + phase * kValuesPerPhase +
                            pack * 8);
    }
}

__device__ __forceinline__ void
fma_phase(const __nv_bfloat16* act, const Nvfp4CodePack<16> (&codes)[2], const float (&coeff)[2],
          int lane, float (&acc)[2][kT]) {
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
        float2 rw[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const std::uint8_t packed =
                static_cast<std::uint8_t>(codes[r].words[pair / 4] >> (8 * (pair & 3)));
            const float2 code = decode_nvfp4_e2m1x2(packed);
            rw[r]             = make_float2(code.x * coeff[r], code.y * coeff[r]);
        }
        const auto* ap = reinterpret_cast<const std::uint32_t*>(act);
#pragma unroll
        for (int t = 0; t < kT; ++t) {
            const float2 a = bf16x2_bits_to_float2(ap[t * 256 + lane * 8 + pair]);
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                acc[r][t] = fmaf(rw[r].x, a.x, acc[r][t]);
                acc[r][t] = fmaf(rw[r].y, a.y, acc[r][t]);
            }
        }
    }
}

__device__ __forceinline__ void load_coeff(const std::uint8_t* scales, const int (&rows)[2],
                                            int phase, int lane, float inverse, float (&coeff)[2]) {
    const int group = (phase * kValuesPerPhase + lane * 16) / 16;
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        coeff[r] =
            decode_nvfp4_e4m3(scales[nvfp4_scale_offset<Geometry>(rows[r], group)]) * inverse;
    }
}

__device__ __forceinline__ void epilogue(float (&acc)[2][kT], int lane, int gate_row,
                                          __nv_bfloat16* out) {
#pragma unroll
    for (int t = 0; t < kT; ++t) {
        float g = warp_reduce_sum(acc[0][t]);
        float u = warp_reduce_sum(acc[1][t]);
        if (lane == 0) {
            out[static_cast<std::int64_t>(t) * kIntermediate + gate_row] =
                __float2bfloat16_rn(silu(g) * u);
        }
    }
}

__global__ __launch_bounds__(256, 1) void kernel_prod(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, float inverse, __nv_bfloat16* __restrict__ out) {
    __shared__ Nvfp4SmallTSharedStorage<Geometry, kT, Schedule> shared;
    int gate_row = 0, lane = 0, warp = 0, rows[2];
    resolve_rows(gate_row, rows, lane, warp);
    float acc[2][kT][1] = {};
    compute_nvfp4_small_t_rows<Geometry, kT, Schedule>(x, codes, scales, shared, inverse, rows,
                                                       warp * 2, 0, 0, lane, acc);
    float flat[2][kT];
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
        for (int t = 0; t < kT; ++t) flat[r][t] = acc[r][t][0];
    epilogue(flat, lane, gate_row, out);
}

// Manual SharedPhase with no trailing sync (matches patched helper).
__global__ __launch_bounds__(256, 1) void kernel_no_tail(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, float inverse, __nv_bfloat16* __restrict__ out) {
    __shared__ __nv_bfloat16 act[kT * kValuesPerPhase];
    int gate_row = 0, lane = 0, warp = 0, rows[2];
    resolve_rows(gate_row, rows, lane, warp);
    float acc[2][kT] = {};
#pragma unroll 1
    for (int phase = 0; phase < kPhases; ++phase) {
        stage_act(x, phase, act);
        __syncthreads();
        Nvfp4CodePack<16> c[2];
        float coeff[2];
        c[0] = load_nvfp4_codes<Nvfp4CodeCache::Default, 16>(codes + code_off(rows[0], phase, lane));
        c[1] = load_nvfp4_codes<Nvfp4CodeCache::Default, 16>(codes + code_off(rows[1], phase, lane));
        load_coeff(scales, rows, phase, lane, inverse, coeff);
        fma_phase(act, c, coeff, lane, acc);
        if (phase + 1 < kPhases) { __syncthreads(); }
    }
    epilogue(acc, lane, gate_row, out);
}

__global__ __launch_bounds__(256, 1) void kernel_reg_pipe2(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, float inverse, __nv_bfloat16* __restrict__ out) {
    __shared__ __nv_bfloat16 act_bufs[2][kT * kValuesPerPhase];
    int gate_row = 0, lane = 0, warp = 0, rows[2];
    resolve_rows(gate_row, rows, lane, warp);
    float acc[2][kT] = {};
    stage_act(x, 0, act_bufs[0]);
    __syncthreads();
    Nvfp4CodePack<16> c0[2], c1[2];
    c0[0] = load_nvfp4_codes<Nvfp4CodeCache::Default, 16>(codes + code_off(rows[0], 0, lane));
    c0[1] = load_nvfp4_codes<Nvfp4CodeCache::Default, 16>(codes + code_off(rows[1], 0, lane));
    c1[0] = load_nvfp4_codes<Nvfp4CodeCache::Default, 16>(codes + code_off(rows[0], 1, lane));
    c1[1] = load_nvfp4_codes<Nvfp4CodeCache::Default, 16>(codes + code_off(rows[1], 1, lane));
#pragma unroll
    for (int phase = 0; phase < kPhases; ++phase) {
        float coeff[2];
        load_coeff(scales, rows, phase, lane, inverse, coeff);
        if ((phase & 1) == 0) fma_phase(act_bufs[0], c0, coeff, lane, acc);
        else fma_phase(act_bufs[1], c1, coeff, lane, acc);
        if (phase + 1 < kPhases) stage_act(x, phase + 1, act_bufs[(phase + 1) & 1]);
        if (phase + 2 < kPhases) {
            if ((phase & 1) == 0) {
                c0[0] = load_nvfp4_codes<Nvfp4CodeCache::Default, 16>(
                    codes + code_off(rows[0], phase + 2, lane));
                c0[1] = load_nvfp4_codes<Nvfp4CodeCache::Default, 16>(
                    codes + code_off(rows[1], phase + 2, lane));
            } else {
                c1[0] = load_nvfp4_codes<Nvfp4CodeCache::Default, 16>(
                    codes + code_off(rows[0], phase + 2, lane));
                c1[1] = load_nvfp4_codes<Nvfp4CodeCache::Default, 16>(
                    codes + code_off(rows[1], phase + 2, lane));
            }
        }
        if (phase + 1 < kPhases) __syncthreads();
    }
    epilogue(acc, lane, gate_row, out);
}

__global__ __launch_bounds__(256, 1) void kernel_full_unroll(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, float inverse, __nv_bfloat16* __restrict__ out) {
    __shared__ __nv_bfloat16 act[kT * kValuesPerPhase];
    int gate_row = 0, lane = 0, warp = 0, rows[2];
    resolve_rows(gate_row, rows, lane, warp);
    float acc[2][kT] = {};
#define PHASE(P, TS)                                                                               \
    do {                                                                                           \
        stage_act(x, (P), act);                                                                    \
        __syncthreads();                                                                           \
        Nvfp4CodePack<16> c[2];                                                                    \
        float coeff[2];                                                                            \
        c[0] = load_nvfp4_codes<Nvfp4CodeCache::Default, 16>(codes + code_off(rows[0], (P), lane)); \
        c[1] = load_nvfp4_codes<Nvfp4CodeCache::Default, 16>(codes + code_off(rows[1], (P), lane)); \
        load_coeff(scales, rows, (P), lane, inverse, coeff);                                       \
        fma_phase(act, c, coeff, lane, acc);                                                       \
        if (TS) __syncthreads();                                                                   \
    } while (0)
    PHASE(0, 1); PHASE(1, 1); PHASE(2, 1); PHASE(3, 1); PHASE(4, 1);
    PHASE(5, 1); PHASE(6, 1); PHASE(7, 1); PHASE(8, 1); PHASE(9, 0);
#undef PHASE
    epilogue(acc, lane, gate_row, out);
}

// Lean smem: only activation buffer via helper path is already tested.
// Try TokenPacked with Streaming codes (no smem act).
using SchedStream =
    Nvfp4SmallTSchedule<8, 1, 2, 16, kT, 1, Nvfp4SmallTActivationAccess::TokenPacked,
                        Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Streaming, 1,
                        Nvfp4SmallTBlockOrder::RowsContiguous, 1>;
using SchedSharedCs =
    Nvfp4SmallTSchedule<8, 1, 2, 16, kT, 1, Nvfp4SmallTActivationAccess::SharedPhase,
                        Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Streaming, 1,
                        Nvfp4SmallTBlockOrder::RowsContiguous, 1>;
using SchedSharedCsU4 =
    Nvfp4SmallTSchedule<8, 1, 2, 16, kT, 1, Nvfp4SmallTActivationAccess::SharedPhase,
                        Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Streaming, 4,
                        Nvfp4SmallTBlockOrder::RowsContiguous, 1>;
using SchedSharedU4 =
    Nvfp4SmallTSchedule<8, 1, 2, 16, kT, 1, Nvfp4SmallTActivationAccess::SharedPhase,
                        Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 4,
                        Nvfp4SmallTBlockOrder::RowsContiguous, 1>;
using SchedSharedCsU10 =
    Nvfp4SmallTSchedule<8, 1, 2, 16, kT, 1, Nvfp4SmallTActivationAccess::SharedPhase,
                        Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Streaming, 10,
                        Nvfp4SmallTBlockOrder::RowsContiguous, 1>;
using SchedSharedU10 =
    Nvfp4SmallTSchedule<8, 1, 2, 16, kT, 1, Nvfp4SmallTActivationAccess::SharedPhase,
                        Nvfp4ScaleAccess::Direct, Nvfp4CodeCache::Default, 10,
                        Nvfp4SmallTBlockOrder::RowsContiguous, 1>;

__global__ __launch_bounds__(256, 1) void kernel_shared_cs_u10(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, float inverse, __nv_bfloat16* __restrict__ out) {
    __shared__ Nvfp4SmallTSharedStorage<Geometry, kT, SchedSharedCsU10> shared;
    int gate_row = 0, lane = 0, warp = 0, rows[2];
    resolve_rows(gate_row, rows, lane, warp);
    float acc[2][kT][1] = {};
    compute_nvfp4_small_t_rows<Geometry, kT, SchedSharedCsU10>(x, codes, scales, shared, inverse,
                                                               rows, warp * 2, 0, 0, lane, acc);
    float flat[2][kT];
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
        for (int t = 0; t < kT; ++t) flat[r][t] = acc[r][t][0];
    epilogue(flat, lane, gate_row, out);
}

__global__ __launch_bounds__(256, 1) void kernel_shared_u10(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, float inverse, __nv_bfloat16* __restrict__ out) {
    __shared__ Nvfp4SmallTSharedStorage<Geometry, kT, SchedSharedU10> shared;
    int gate_row = 0, lane = 0, warp = 0, rows[2];
    resolve_rows(gate_row, rows, lane, warp);
    float acc[2][kT][1] = {};
    compute_nvfp4_small_t_rows<Geometry, kT, SchedSharedU10>(x, codes, scales, shared, inverse,
                                                             rows, warp * 2, 0, 0, lane, acc);
    float flat[2][kT];
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
        for (int t = 0; t < kT; ++t) flat[r][t] = acc[r][t][0];
    epilogue(flat, lane, gate_row, out);
}

// full_unroll body but with Streaming weight codes (matches production L2 policy).
__global__ __launch_bounds__(256, 1) void kernel_full_unroll_cs(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, float inverse, __nv_bfloat16* __restrict__ out) {
    __shared__ __nv_bfloat16 act[kT * kValuesPerPhase];
    int gate_row = 0, lane = 0, warp = 0, rows[2];
    resolve_rows(gate_row, rows, lane, warp);
    float acc[2][kT] = {};
#define PHASE(P, TS)                                                                               \
    do {                                                                                           \
        stage_act(x, (P), act);                                                                    \
        __syncthreads();                                                                           \
        Nvfp4CodePack<16> c[2];                                                                    \
        float coeff[2];                                                                            \
        c[0] = load_nvfp4_codes<Nvfp4CodeCache::Streaming, 16>(codes + code_off(rows[0], (P), lane)); \
        c[1] = load_nvfp4_codes<Nvfp4CodeCache::Streaming, 16>(codes + code_off(rows[1], (P), lane)); \
        load_coeff(scales, rows, (P), lane, inverse, coeff);                                       \
        fma_phase(act, c, coeff, lane, acc);                                                       \
        if (TS) __syncthreads();                                                                   \
    } while (0)
    PHASE(0, 1); PHASE(1, 1); PHASE(2, 1); PHASE(3, 1); PHASE(4, 1);
    PHASE(5, 1); PHASE(6, 1); PHASE(7, 1); PHASE(8, 1); PHASE(9, 0);
#undef PHASE
    epilogue(acc, lane, gate_row, out);
}

__global__ __launch_bounds__(256, 1) void kernel_tok_stream(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, float inverse, __nv_bfloat16* __restrict__ out) {
    __shared__ Nvfp4SmallTSharedStorage<Geometry, kT, SchedStream> shared;
    int gate_row = 0, lane = 0, warp = 0, rows[2];
    resolve_rows(gate_row, rows, lane, warp);
    float acc[2][kT][1] = {};
    compute_nvfp4_small_t_rows<Geometry, kT, SchedStream>(x, codes, scales, shared, inverse, rows,
                                                          warp * 2, 0, 0, lane, acc);
    float flat[2][kT];
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
        for (int t = 0; t < kT; ++t) flat[r][t] = acc[r][t][0];
    epilogue(flat, lane, gate_row, out);
}

__global__ __launch_bounds__(256, 1) void kernel_shared_cs(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, float inverse, __nv_bfloat16* __restrict__ out) {
    __shared__ Nvfp4SmallTSharedStorage<Geometry, kT, SchedSharedCs> shared;
    int gate_row = 0, lane = 0, warp = 0, rows[2];
    resolve_rows(gate_row, rows, lane, warp);
    float acc[2][kT][1] = {};
    compute_nvfp4_small_t_rows<Geometry, kT, SchedSharedCs>(x, codes, scales, shared, inverse, rows,
                                                            warp * 2, 0, 0, lane, acc);
    float flat[2][kT];
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
        for (int t = 0; t < kT; ++t) flat[r][t] = acc[r][t][0];
    epilogue(flat, lane, gate_row, out);
}

__global__ __launch_bounds__(256, 1) void kernel_shared_cs_u4(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, float inverse, __nv_bfloat16* __restrict__ out) {
    __shared__ Nvfp4SmallTSharedStorage<Geometry, kT, SchedSharedCsU4> shared;
    int gate_row = 0, lane = 0, warp = 0, rows[2];
    resolve_rows(gate_row, rows, lane, warp);
    float acc[2][kT][1] = {};
    compute_nvfp4_small_t_rows<Geometry, kT, SchedSharedCsU4>(x, codes, scales, shared, inverse,
                                                              rows, warp * 2, 0, 0, lane, acc);
    float flat[2][kT];
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
        for (int t = 0; t < kT; ++t) flat[r][t] = acc[r][t][0];
    epilogue(flat, lane, gate_row, out);
}

__global__ __launch_bounds__(256, 1) void kernel_shared_u4(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, float inverse, __nv_bfloat16* __restrict__ out) {
    __shared__ Nvfp4SmallTSharedStorage<Geometry, kT, SchedSharedU4> shared;
    int gate_row = 0, lane = 0, warp = 0, rows[2];
    resolve_rows(gate_row, rows, lane, warp);
    float acc[2][kT][1] = {};
    compute_nvfp4_small_t_rows<Geometry, kT, SchedSharedU4>(x, codes, scales, shared, inverse, rows,
                                                            warp * 2, 0, 0, lane, acc);
    float flat[2][kT];
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
        for (int t = 0; t < kT; ++t) flat[r][t] = acc[r][t][0];
    epilogue(flat, lane, gate_row, out);
}

using LaunchFn = void (*)(const __nv_bfloat16*, const std::uint8_t*, const std::uint8_t*, float,
                          __nv_bfloat16*, cudaStream_t);

template <auto* Kernel>
void launch(const __nv_bfloat16* x, const std::uint8_t* c, const std::uint8_t* s, float inv,
            __nv_bfloat16* o, cudaStream_t stream) {
    Kernel<<<kBlocks, 256, 0, stream>>>(x, c, s, inv, o);
    CUDA_CHECK(cudaGetLastError());
}

struct Candidate {
    const char* name;
    LaunchFn fn;
};

__global__ __launch_bounds__(256, 2) void kernel_prod_occ2(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, float inverse, __nv_bfloat16* __restrict__ out) {
    __shared__ Nvfp4SmallTSharedStorage<Geometry, kT, Schedule> shared;
    int gate_row = 0, lane = 0, warp = 0, rows[2];
    resolve_rows(gate_row, rows, lane, warp);
    float acc[2][kT][1] = {};
    compute_nvfp4_small_t_rows<Geometry, kT, Schedule>(x, codes, scales, shared, inverse, rows,
                                                       warp * 2, 0, 0, lane, acc);
    float flat[2][kT];
#pragma unroll
    for (int r = 0; r < 2; ++r)
#pragma unroll
        for (int t = 0; t < kT; ++t) flat[r][t] = acc[r][t][0];
    epilogue(flat, lane, gate_row, out);
}

// Soft pipeline: prefetch next codes before FMA; single act buffer; no dblbuf smem.
__global__ __launch_bounds__(256, 1) void kernel_soft_prefetch(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, float inverse, __nv_bfloat16* __restrict__ out) {
    __shared__ __nv_bfloat16 act[kT * kValuesPerPhase];
    int gate_row = 0, lane = 0, warp = 0, rows[2];
    resolve_rows(gate_row, rows, lane, warp);
    float acc[2][kT] = {};
    stage_act(x, 0, act);
    __syncthreads();
    Nvfp4CodePack<16> c[2];
    c[0] = load_nvfp4_codes<Nvfp4CodeCache::Default, 16>(codes + code_off(rows[0], 0, lane));
    c[1] = load_nvfp4_codes<Nvfp4CodeCache::Default, 16>(codes + code_off(rows[1], 0, lane));
#pragma unroll 1
    for (int phase = 0; phase < kPhases; ++phase) {
        float coeff[2];
        load_coeff(scales, rows, phase, lane, inverse, coeff);
        Nvfp4CodePack<16> c_next[2];
        if (phase + 1 < kPhases) {
            // Issue next codes before FMA so they overlap with math when L2-warm.
            c_next[0] = load_nvfp4_codes<Nvfp4CodeCache::Default, 16>(
                codes + code_off(rows[0], phase + 1, lane));
            c_next[1] = load_nvfp4_codes<Nvfp4CodeCache::Default, 16>(
                codes + code_off(rows[1], phase + 1, lane));
        }
        fma_phase(act, c, coeff, lane, acc);
        if (phase + 1 < kPhases) {
            __syncthreads();
            stage_act(x, phase + 1, act);
            c[0] = c_next[0];
            c[1] = c_next[1];
            __syncthreads();
        }
    }
    epilogue(acc, lane, gate_row, out);
}

// Full unroll + occ2 launch bounds.
__global__ __launch_bounds__(256, 2) void kernel_full_unroll_occ2(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, float inverse, __nv_bfloat16* __restrict__ out) {
    __shared__ __nv_bfloat16 act[kT * kValuesPerPhase];
    int gate_row = 0, lane = 0, warp = 0, rows[2];
    resolve_rows(gate_row, rows, lane, warp);
    float acc[2][kT] = {};
#define PHASE(P, TS)                                                                               \
    do {                                                                                           \
        stage_act(x, (P), act);                                                                    \
        __syncthreads();                                                                           \
        Nvfp4CodePack<16> c[2];                                                                    \
        float coeff[2];                                                                            \
        c[0] = load_nvfp4_codes<Nvfp4CodeCache::Default, 16>(codes + code_off(rows[0], (P), lane)); \
        c[1] = load_nvfp4_codes<Nvfp4CodeCache::Default, 16>(codes + code_off(rows[1], (P), lane)); \
        load_coeff(scales, rows, (P), lane, inverse, coeff);                                       \
        fma_phase(act, c, coeff, lane, acc);                                                       \
        if (TS) __syncthreads();                                                                   \
    } while (0)
    PHASE(0, 1); PHASE(1, 1); PHASE(2, 1); PHASE(3, 1); PHASE(4, 1);
    PHASE(5, 1); PHASE(6, 1); PHASE(7, 1); PHASE(8, 1); PHASE(9, 0);
#undef PHASE
    epilogue(acc, lane, gate_row, out);
}

const Candidate kCandidates[] = {
    {"prod", &launch<kernel_prod>},
    {"no_tail", &launch<kernel_no_tail>},
    {"reg_pipe2", &launch<kernel_reg_pipe2>},
    {"full_unroll", &launch<kernel_full_unroll>},
    {"tok_stream", &launch<kernel_tok_stream>},
    {"shared_cs", &launch<kernel_shared_cs>},
    {"shared_cs_u4", &launch<kernel_shared_cs_u4>},
    {"shared_u4", &launch<kernel_shared_u4>},
    {"shared_cs_u10", &launch<kernel_shared_cs_u10>},
    {"shared_u10", &launch<kernel_shared_u10>},
    {"full_unroll_cs", &launch<kernel_full_unroll_cs>},
    {"prod_occ2", &launch<kernel_prod_occ2>},
    {"soft_prefetch", &launch<kernel_soft_prefetch>},
    {"full_unroll_o2", &launch<kernel_full_unroll_occ2>},
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
    for (int i = 0; i < kWarmWarmup; ++i) fn(x, c, s, inv, o, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaEvent_t start = nullptr, stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<double> samples;
    samples.reserve(kWarmRepeat);
    // Batch 10 launches per timed sample to reduce event noise.
    constexpr int kBatch = 10;
    for (int i = 0; i < kWarmRepeat; ++i) {
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
    return {samples[samples.size() / 2], samples.front(),
            samples[std::min(samples.size() - 1, static_cast<std::size_t>(0.95 * samples.size()))]};
}

} // namespace

int main() {
    try {
        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        DeviceBuffer flush(kFlushBytes);
        DeviceBuffer input = bench::make_bf16(static_cast<std::size_t>(kHidden) * kT);
        DeviceBuffer output(static_cast<std::size_t>(kIntermediate) * kT * 2);
        DeviceBuffer output_ref(output.bytes);
        auto packed         = bench::make_nvfp4_weight(kGateUpRows, kHidden);
        const float inverse = 1.0F / packed.weight.weight_scale_divisor;
        const auto* x       = static_cast<const __nv_bfloat16*>(input.p);
        const auto* codes   = static_cast<const std::uint8_t*>(packed.weight.qdata);
        const auto* scales  = static_cast<const std::uint8_t*>(packed.weight.scales);
        auto* out           = static_cast<__nv_bfloat16*>(output.p);
        auto* out_ref       = static_cast<__nv_bfloat16*>(output_ref.p);
        const std::size_t n_out = static_cast<std::size_t>(kIntermediate) * kT;
        std::vector<std::uint16_t> href(n_out), hgot(n_out);

        std::printf("%-12s %6s %10s %10s %8s %8s\n", "variant", "exact", "cold_us", "warm_us",
                    "cold%", "warm%");

        double prod_cold = 0, prod_warm = 0;
        const Candidate* best = nullptr;
        double best_min_vs    = -1e9;

        for (const auto& cand : kCandidates) {
            launch<kernel_prod>(x, codes, scales, inverse, out_ref, stream);
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
                cold_us = bench::measure_cold_launch(run, flush, stream, kColdWarmup, kColdRepeat)
                              .median_us;
                warm_us = measure_warm(cand.fn, x, codes, scales, inverse, out, stream).median_us;
                if (std::string_view(cand.name) == "prod") {
                    prod_cold = cold_us;
                    prod_warm = warm_us;
                }
                cold_vs = prod_cold > 0 ? 100.0 * (prod_cold - cold_us) / prod_cold : 0;
                warm_vs = prod_warm > 0 ? 100.0 * (prod_warm - warm_us) / prod_warm : 0;
            }
            std::printf("%-12s %6s %10.3f %10.3f %+7.2f %+7.2f%s\n", cand.name,
                        ok ? "PASS" : "FAIL", cold_us, warm_us, cold_vs, warm_vs,
                        (!ok) ? ""
                        : (cold_vs >= 1.0 && warm_vs >= 1.0)
                            ? "  <<WIN"
                            : (cold_vs >= 1.0 && warm_vs < 1.0) ? "  (cold-only)"
                                                                : "");
            if (ok && cold_vs >= 1.0 && warm_vs >= 1.0) {
                const double score = std::min(cold_vs, warm_vs);
                if (score > best_min_vs) {
                    best_min_vs = score;
                    best        = &cand;
                }
            }
        }

        if (best)
            std::printf("\nWINNER (cold&warm): %s\n", best->name);
        else
            std::printf("\nNO dual-gate winner (do not land cold-only)\n");

        CUDA_CHECK(cudaStreamDestroy(stream));
        return best ? 0 : 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
