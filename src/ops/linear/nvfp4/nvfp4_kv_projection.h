#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// The public Linear entry validates the original NVFP4 descriptor, distinct contiguous BF16
// outputs, and packed shape. This launcher preserves original row and scale-plane addressing.
void launch_nvfp4_kv_projection(const Tensor& x, const Weight& weight, Tensor& key, Tensor& value,
                                cudaStream_t stream, std::int32_t sequence_width);

} // namespace ninfer::ops::detail
