#pragma once
#include "core/tensor.h"
#include <cuda_runtime.h>
namespace ninfer::ops::detail {
void linear_bias_launch(const Tensor&,const Weight&,const Tensor&,Tensor&,cudaStream_t);
}
