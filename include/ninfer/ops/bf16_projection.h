#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

#include <cstdint>

namespace ninfer::ops {

/** Whether bf16_projection admits an [output_rows, input_rows] weight. */
[[nodiscard]] bool bf16_projection_admits(std::int32_t output_rows, std::int32_t input_rows);

/** The output dtype of an admitted projection with output_rows rows: BF16 or FP32. */
[[nodiscard]] DType bf16_projection_output_dtype(std::int32_t output_rows);

/**
 * Op: bf16_projection
 *
 *   y[n,t] = sum_k weight[n,k] * x[k,t]
 *
 * A BF16 projection for the Qwen3.8-Flash-Next projections that do not run through NVFP4. Most
 * feed sensitive FP32 control math and produce FP32: the GDN a|b gates [96,2560], the MoE control
 * rows [544,2560] (router, shared-expert gate, zero padding; see moe_route), and the PLE key
 * [10240,2560] and value [2560,2560] projections. The QSA indexer query [512,2560] and key
 * [128,2560] projections feed the BF16 inputs of the QSA Ops and produce BF16. Only these exact
 * shapes, each with its fixed output dtype (bf16_projection_output_dtype), are admitted.
 *
 * x is BF16 [K,T]; weight is BF16 [N,K] stored row-major (Tensor ne {K, N}); y is [N,T]. All three
 * are contiguous and 16-byte aligned, y is disjoint from x and weight, and T >= 1; x and weight are
 * not written while the call runs. The oracle evaluates the sum in FP64 from the represented BF16
 * inputs. Accumulation is FP32: T=1 at the 2560- and 10240-row shapes uses IEEE FP32 fused
 * multiply-adds (a GEMV); every other point uses tensor-core MMA accumulators (a small-T route for
 * T <= 48, a GEMM beyond and for T >= 2 at 10240 rows), in route-specific reduction orders. FP32
 * outputs store the accumulator unrounded; BF16 outputs round it once to nearest even.
 * No workspace or persistent state. Invalid arguments throw std::invalid_argument. Asynchronous
 * on stream and graph-capturable for a fixed T.
 */
void bf16_projection(const Tensor& x, const Tensor& weight, Tensor& y, cudaStream_t stream);

} // namespace ninfer::ops
