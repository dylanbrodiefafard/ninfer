#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

using Q4Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

void launch_q4_gemv_r4_w1_direct(const Tensor& x, const Weight& w, Tensor& out,
                                 cudaStream_t stream);
void launch_q4_gemv_r1_w8_direct(const Tensor& x, const Weight& w, Tensor& out,
                                 cudaStream_t stream);
void launch_q4_simt_r8_c4(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q4_simt_r8_c8(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q4_mma_r64_c64(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);
void launch_q4_mma_r64_c128(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
