#pragma once

// ninfer::ops::detail - private launch prototypes for l2norm.

#include "core/tensor.h"
#include "ninfer/ops/l2norm.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// Shared launcher; `dump` is a non-null L2NormDump only for the dev side-band.
void l2norm_run(const Tensor& x, float eps, Tensor& out, cudaStream_t stream, L2NormDump* dump);
void l2norm_launch(const Tensor& x, float eps, Tensor& out, cudaStream_t stream);
void l2norm_dump(const Tensor& x, float eps, Tensor& out, cudaStream_t stream, L2NormDump& dump);

} // namespace ninfer::ops::detail