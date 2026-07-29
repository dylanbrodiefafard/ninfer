#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr std::int32_t kBf16FirstSmallT = 2;
inline constexpr std::int32_t kBf16LastSmallT  = 32;

using Bf16Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

void launch_bf16_decode(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);
void launch_bf16_small_t(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
