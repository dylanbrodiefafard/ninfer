#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// K splits of the down/write projection: split s reduces the 2560 inputs of residual stream s, so
// one split's partial dot products share one inverse RMS.
inline constexpr std::int32_t kHyperConnectionKSplits = 4;
// Down rows (320) followed by the four write rows.
inline constexpr std::int32_t kHyperConnectionProjectedRows = 324;
// Tokens one CTA carries through a SIMT projection: every weight byte a CTA loads serves this many
// tokens.
inline constexpr std::int32_t kHyperConnectionTokenTile = 8;
// Launches of at least kHyperConnectionMmaMinTokens tokens run the prefill Tensor Core route (the
// projections are Tensor Core bound from T = 146; the route is measured faster from T = 160); the
// read wrapper then passes chunks of up to kHyperConnectionMmaChunkTokens, and otherwise chunks of
// up to kHyperConnectionSimtChunkTokens.
inline constexpr std::int32_t kHyperConnectionMmaMinTokens    = 160;
inline constexpr std::int32_t kHyperConnectionMmaChunkTokens  = 2048;
inline constexpr std::int32_t kHyperConnectionSimtChunkTokens = 512;
// Small-T reads from this crossover through 32 tokens finish gate inputs once per token.
inline constexpr std::int32_t kHyperConnectionFinishMinTokens = 9;
inline constexpr std::int32_t kHyperConnectionSmallMaxTokens  = 32;

// One FP8_E4M3FN_ROW_BF16S matrix [rows, cols]: row-major E4M3FN codes and one BF16 multiplier per
// row. Both planes are 16-byte aligned device addresses.
struct HyperConnectionFp8Planes {
    const std::uint8_t* codes;
    const std::uint16_t* scales;
};

// Workspace for separately finished Tensor Core gate inputs, outermost dimension first:
// u_hi and u_lo are BF16
// [tokens][320] (the gate input u = u_hi + u_lo to about 2^-17 relative) and inverse_rms is FP32
// [tokens][4].
struct HyperConnectionMmaWorkspace {
    std::uint16_t* u_hi;
    std::uint16_t* u_lo;
    float* inverse_rms;
};

// The launchers trust the wrapper's validation (shapes, formats, alignment, disjointness).
// Pointers are 16-byte aligned device addresses. Every launch takes 1 <= tokens <= 65535; the read
// wrapper passes chunks of at most kHyperConnectionMmaChunkTokens tokens, and of at most
// kHyperConnectionSimtChunkTokens below kHyperConnectionMmaMinTokens.

void hyper_connection_broadcast_launch(const std::uint16_t* embedding, std::uint16_t* residual,
                                       std::int32_t tokens, cudaStream_t stream);

// Workspace, outermost dimension first: partials is FP32
// [kHyperConnectionKSplits][tokens][kHyperConnectionProjectedRows] and square_sums is FP32
// [kHyperConnectionKSplits][tokens] on every route. Prefill Tensor Core reads and small-T reads
// from kHyperConnectionFinishMinTokens through kHyperConnectionSmallMaxTokens also use mma.
// write_weight and write_scale are null for a read without write.
void hyper_connection_read_launch(const std::uint16_t* residual, const std::uint16_t* norm_weight,
                                  HyperConnectionFp8Planes down, HyperConnectionFp8Planes up,
                                  const std::uint16_t* write_weight, std::uint16_t* x,
                                  float* write_scale, float* partials, float* square_sums,
                                  HyperConnectionMmaWorkspace mma, std::int32_t tokens,
                                  cudaStream_t stream);

void hyper_connection_inject_launch(const std::uint16_t* residual,
                                    const std::uint16_t* block_output, const float* write_scale,
                                    std::uint16_t* residual_out, std::int32_t tokens,
                                    cudaStream_t stream);

} // namespace ninfer::ops::detail
