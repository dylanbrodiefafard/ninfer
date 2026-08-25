#pragma once

// Host launch entry points for Sage3 kernels compiled in their own TUs so
// nvcc/cicc can use multiple cores. Declarations only; definitions and
// explicit instantiations live in gqa_attention_{prefill,decode}_s3*.cu.

#include "ops/launcher/gqa_attention.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

template <typename Geometry, typename CacheView, typename Metadata>
bool gqa_s3_prefill_tma_try_launch(const Tensor& q, const Tensor& positions, float scale,
                                   const CacheView& cache, Metadata metadata, Tensor& out,
                                   cudaStream_t stream, float keep_frac, GqaS3PrefillDump* dump,
                                   std::uint32_t* dbg_regs, std::uint8_t* dbg_q);

template <typename Geometry, typename CacheView, typename Metadata>
void gqa_s3_prefill_attention_launch(const Tensor& q, const Tensor& positions, float scale,
                                     const CacheView& cache, Metadata metadata, Tensor& out,
                                     cudaStream_t stream, float keep_frac, GqaS3PrefillDump* dump,
                                     std::uint32_t* dbg_regs, std::uint8_t* dbg_q);

template <typename Geometry, typename CacheView, typename Metadata>
void gqa_s3_prefill_fill_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                CacheView cache, Metadata metadata, cudaStream_t stream);

template <typename Geometry, int TokenTile, bool MultiBatch, bool Masked, typename CacheInput>
void launch_tc_partial_nvfp4s3(const Tensor& q, CacheInput input, const Tensor& pos, float scale,
                               PagedKVBatchLayerView cache, const GqaSmallTInvocation& invocation,
                               std::int32_t logical_capacity, std::int32_t implementation_window,
                               std::int32_t splits, Tensor& partial_acc, Tensor& partial_m,
                               Tensor& partial_l, cudaStream_t stream, float keep_frac,
                               const Tensor& keep_tiles, const Tensor& keep_count,
                               const Tensor& split_off, std::int32_t keep_stride);

} // namespace ninfer::ops::detail
