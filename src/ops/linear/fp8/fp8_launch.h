#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// The registered FP8 vocabulary heads ([248320,5120] and [248320,2560]). The small-T MMA serves
// T in [kFp8VocabularyFirstA16SmallTMmaT, kFp8VocabularyLastA16SmallTMmaT] and the GEMM every
// T >= kFp8VocabularyFirstA16GemmT; other problems or widths throw std::invalid_argument.
void launch_fp8_vocabulary_a16_small_t(const Tensor& x, const Weight& weight, Tensor& out,
                                       cudaStream_t stream);
void launch_fp8_vocabulary_a16_gemm(const Tensor& x, const Weight& weight, Tensor& out,
                                    cudaStream_t stream);

// The Qwen3.8-Flash-Next GDN/attention output [2560,6144] at every positive T with BF16
// activations; another problem or a non-positive T throws std::invalid_argument.
void launch_fp8_flash_next_mixer_out(const Tensor& x, const Weight& weight, Tensor& out,
                                     cudaStream_t stream);

} // namespace ninfer::ops::detail
