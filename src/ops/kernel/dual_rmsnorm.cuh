#pragma once

#include "ops/kernel/rmsnorm.cuh"

namespace ninfer::ops {

// Implements the independent-panel contract in include/ninfer/ops/rmsnorm.h. Grid (T,2)
// assigns one 512-thread CTA to a row of the selected panel. Each lane retains five BF16 pairs;
// 16 warp sums and one inverse RMS occupy shared memory. The shared row body synchronizes its
// warp sums before the inverse and synchronizes the inverse before output. All pointers must
// be BF16x2-aligned, with contiguous width 5120; panels remain separate allocations.
__launch_bounds__(512) __global__
    void dual_offset_rmsnorm_bf16x2_kernel(const __nv_bfloat162* x0, const __nv_bfloat162* weight0,
                                           const __nv_bfloat162* x1, const __nv_bfloat162* weight1,
                                           __nv_bfloat162* out0, __nv_bfloat162* out1,
                                           std::int64_t rows, float eps) {
    const bool second = blockIdx.y == 1;
    rmsnorm_cta_bf16x2_body<RmsEpilogue::Offset, 512, 8>(second ? x1 : x0,
                                                         second ? weight1 : weight0, nullptr,
                                                         second ? out1 : out0, 5120, rows, eps);
}

// Grid (T,2) assigns one 256-thread CTA per independent width 5120 panel row. Scalar BF16
// pointers need only two-byte alignment. A 256-float shared tree reduces the row, with a block
// barrier after each tree level; no shared data or statistic crosses panel CTAs.
__launch_bounds__(256) __global__
    void dual_offset_rmsnorm_generic_kernel(const __nv_bfloat16* x0, const __nv_bfloat16* weight0,
                                            const __nv_bfloat16* x1, const __nv_bfloat16* weight1,
                                            __nv_bfloat16* out0, __nv_bfloat16* out1,
                                            std::int64_t rows, float eps) {
    const bool second = blockIdx.y == 1;
    rmsnorm_generic_body<RmsEpilogue::Offset>(second ? x1 : x0, second ? weight1 : weight0, nullptr,
                                              second ? out1 : out0, 5120, rows, eps);
}

} // namespace ninfer::ops
