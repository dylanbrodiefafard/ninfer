#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

using W8Launch = void (*)(const Tensor&, const Weight&, Tensor&, WorkspaceArena&, cudaStream_t);

void launch_w8_decode_r4(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                         cudaStream_t stream);
void launch_w8_exact_t_splitk(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                              cudaStream_t stream);
void launch_w8_exact_t_composite(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                                 cudaStream_t stream);
void launch_w8_medium_splitk_c96(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                                 cudaStream_t stream);
void launch_w8_medium_splitk_c128(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                                  cudaStream_t stream);
void launch_w8_medium_splitk_c144(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                                  cudaStream_t stream);

void launch_w8_simt_r8_c4(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                          cudaStream_t stream);
void launch_w8_simt_r8_c8(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                          cudaStream_t stream);

void launch_w8_mma_r32_c64(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                           cudaStream_t stream);
void launch_w8_mma_r32_c96(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                           cudaStream_t stream);
void launch_w8_mma_r32_c128(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                            cudaStream_t stream);
void launch_w8_mma_r48_c64(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                           cudaStream_t stream);
void launch_w8_mma_r48_c96(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                           cudaStream_t stream);
void launch_w8_mma_r48_c112(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                            cudaStream_t stream);
void launch_w8_mma_r48_c128(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                            cudaStream_t stream);
void launch_w8_mma_r64_c96(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                           cudaStream_t stream);
void launch_w8_mma_r64_c112(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                            cudaStream_t stream);
void launch_w8_mma_r64_c128(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                            cudaStream_t stream);
void launch_w8_mma_r96_c96(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                           cudaStream_t stream);
void launch_w8_mma_r128_c64(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                            cudaStream_t stream);
void launch_w8_mma_r128_c80(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                            cudaStream_t stream);

void launch_w8_exact_mma_r32_c96(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                                 cudaStream_t stream);
void launch_w8_exact_mma_r32_c128(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                                  cudaStream_t stream);
void launch_w8_exact_mma_r48_c96(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                                 cudaStream_t stream);
void launch_w8_exact_mma_r48_c128(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                                  cudaStream_t stream);
void launch_w8_exact_mma_r64_c96(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                                 cudaStream_t stream);
void launch_w8_exact_mma_r64_c128(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                                  cudaStream_t stream);
void launch_w8_exact_mma_r96_c96(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                                 cudaStream_t stream);
void launch_w8_exact_mma_r128_c80(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                                  cudaStream_t stream);

} // namespace ninfer::ops::detail
