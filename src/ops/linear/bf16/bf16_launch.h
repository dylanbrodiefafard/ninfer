#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Keep the candidate domain separate from the measured production crossover so the two kernel
// families remain directly comparable in the benchmark overlap.
inline constexpr std::int32_t kBf16SmallTMinTokens         = 2;
inline constexpr std::int32_t kBf16SmallTMaxTokens         = 32;
inline constexpr std::int32_t kBf16LinearSmallTDispatchEnd = 27;

using Bf16Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

void launch_bf16_decode(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);
void launch_bf16_small_t(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);
void launch_bf16_mma(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);
// Private QSA key / MoE shared gate-up precision route; not public Linear FP32 admission.
// Exact [512,2560] and [640,2560], canonical schedules with FP32 output storage.
void launch_bf16_f32(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops::detail
