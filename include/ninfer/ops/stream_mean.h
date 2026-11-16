#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

// Registered stream-mean geometry: four residual streams of a 2560-wide hidden state.
inline constexpr std::int32_t kStreamMeanHidden  = 2560;
inline constexpr std::int32_t kStreamMeanStreams = 4;

/**
 * Op: stream_mean
 *
 * Math / indexing:
 *   out[d,t] = (1/4) * sum_{i=0}^{3} residual[d,i,t]   for d in [0,2560), t in [0,T).
 *
 * Logical shapes:
 *   residual is BF16 [2560,4,T] (channel fastest, then stream, then token), contiguous. out is
 *   BF16 [2560,T] with contiguous rows (nb[0] = 2) and any column stride nb[1] >= 5120 bytes that
 *   is a multiple of 16, so it may be a 2560-row slice of a wider feature matrix. T >= 1.
 *
 * Supported domain:
 *   BF16 only, at the registered geometry. residual.data and out.data are 16-byte aligned.
 *
 * Numeric:
 *   The oracle evaluates the mean in FP64 from the represented BF16 inputs. out holds one BF16
 *   rounding of the mean; private accumulation precision is implementation-defined and covered
 *   by the suite criterion. Non-finite inputs propagate as IEEE arithmetic gives.
 *
 * Effects:
 *   Writes out[0:2560, 0:T] only; bytes between columns of a strided out are unchanged.
 *   residual is unchanged and must not overlap the byte span of out.
 *
 * Workspace:
 *   None.
 *
 * Execution:
 *   Asynchronous on stream and graph-capturable for fixed T and addresses. Invalid operands throw
 *   std::invalid_argument before any work is submitted.
 */
void stream_mean(const Tensor& residual, Tensor& out, cudaStream_t stream);

} // namespace ninfer::ops
