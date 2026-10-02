#pragma once

// ninfer::ops::detail - private launch prototypes for causal_conv1d.

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

inline constexpr std::int32_t kCausalConvSequenceMaxTokens = 64;
// The split (direct q/k/v store) form switches to the token-parallel pairs kernel above this
// width: at C=10240 the serial kernel costs 6.1/8.2/14.4 us at T=16/32/64 against 4.1/6.1/6.2 us,
// and the two tie at T=8.
inline constexpr std::int32_t kCausalConvSplitSequenceMaxTokens = 8;
inline constexpr std::int32_t kCausalConvParallelMaxTokens      = 16;

void causal_conv1d_prefill_launch(const Tensor& x, const Tensor& weight,
                                  const Tensor& conv_state_in, Tensor& conv_state_out, Tensor& out,
                                  cudaStream_t stream);
void causal_conv1d_split_launch(const Tensor& x, const Tensor& weight, Tensor& conv_state,
                                Tensor& query, Tensor& key, Tensor& value, cudaStream_t stream);
void causal_conv1d_sequence_launch(const Tensor& x, const Tensor& weight,
                                   const Tensor& conv_state_in, Tensor& conv_state_out, Tensor& out,
                                   cudaStream_t stream);
void causal_conv1d_smallt_launch(const Tensor& x, const Tensor& weight, const Tensor& conv_state_in,
                                 Tensor& conv_state_out, Tensor& out, cudaStream_t stream);
void causal_conv1d_decode_launch(const Tensor& x, const Tensor& weight, const Tensor& conv_state_in,
                                 Tensor& conv_state_out, Tensor& out, cudaStream_t stream);
void causal_conv1d_snapshot_launch(const Tensor& x, const Tensor& weight, Tensor& conv_states,
                                   const Tensor& valid_columns, const Tensor& initial_state_slots,
                                   const Tensor& snapshot_base_slots, Tensor& out,
                                   cudaStream_t stream);

} // namespace ninfer::ops::detail
