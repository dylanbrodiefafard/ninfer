#pragma once

// Exact-NVFP4 skip-list prefill, compiled in its own TU so cicc can run in
// parallel with the dense NVFP4 / S3 launchers.

#include "ops/launcher/gqa_attention.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

template <typename Geometry, typename CacheView, typename Metadata>
void gqa_sparse_prefill_attention_launch(const Tensor& q, const Tensor& positions, float scale,
                                          const CacheView& cache, Metadata metadata, Tensor& out,
                                          cudaStream_t stream, float keep_frac, float xattn_tau,
                                          std::int32_t xattn_min_len, GqaS3PrefillDump* dump,
                                          void* xattn_scratch = nullptr,
                                          GqaExecutionEnvelope envelope = {
                                              1, kGqaAttentionMaximumVisibleKeys});

} // namespace ninfer::ops::detail
