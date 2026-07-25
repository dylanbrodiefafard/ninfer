#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

using Q6Launch = void (*)(const Tensor&, const Weight&, Tensor&, WorkspaceArena&, cudaStream_t);

void launch_q6_simt_r8_c4(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                          cudaStream_t stream);
void launch_q6_simt_r8_c8(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                          cudaStream_t stream);
void launch_q6_mma_r64_c64(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                           cudaStream_t stream);
void launch_q6_mma_r64_c128(const Tensor& x, const Weight& w, Tensor& out, WorkspaceArena& ws,
                            cudaStream_t stream);

} // namespace ninfer::ops::detail
