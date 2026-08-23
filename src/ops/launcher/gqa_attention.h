#pragma once

// ninfer::ops::detail - private launch prototypes for gqa_attention policies.

#include "core/paged_kv_cache.h"
#include "core/tensor.h"
#include "ninfer/ops/gqa_attention.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

enum class GqaAttentionRoute { SmallT, ChunkedSmallT, Prompt };

struct GqaSmallTInvocation {
    const Tensor* valid_columns = nullptr;
    const Tensor* table_rows    = nullptr;
    std::int32_t full_width     = 0;
    std::int32_t column_begin   = 0;
    std::int32_t width          = 0;
    std::int32_t batch_size     = 1;
};

std::int32_t gqa_attention_split_capacity(std::int32_t q_heads, std::int32_t tokens,
                                          DType cache_dtype, GqaExecutionEnvelope envelope);

bool gqa_attention_uses_small_t(std::int32_t tokens);

GqaAttentionRoute gqa_attention_resolve_route(std::int32_t q_heads, std::int32_t width,
                                              std::int32_t batch_size,
                                              GqaExecutionEnvelope envelope);

const char* gqa_attention_route_name(GqaAttentionRoute route);

struct GqaSmallTKeepScratch {
    // Sparge-decode tile-skip scratch (all empty unless the sage tile-skip is
    // engaged: keep_frac < 1 + the k_mean plane + a T=1 step).
    Tensor keep_tiles; // [batch*KVHeads][max_keep] kept-tile index (i32)
    Tensor keep_count; // [batch*KVHeads] kept-tile count (i32)
    Tensor split_off;  // [batch*KVHeads][splits+1] per-split keep prefix (i32)
    std::int32_t max_keep = 0; // per-(kv,batch) keep-list capacity
};

void gqa_attention_small_t_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                   const Tensor& positions, const Tensor& valid_columns,
                                   const Tensor& table_rows, float scale,
                                   PagedKVBatchLayerView cache, GqaExecutionEnvelope envelope,
                                   std::int32_t column_begin, std::int32_t width,
                                   Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l,
                                   Tensor& out, cudaStream_t stream, float keep_frac = 1.0f,
                                   const GqaSmallTKeepScratch& keep = {});

void gqa_attention_cached_small_t_launch(const Tensor& q, const Tensor& positions, float scale,
                                          const PagedKVLayerView& cache,
                                          GqaExecutionEnvelope envelope, Tensor& partial_acc,
                                          Tensor& partial_m, Tensor& partial_l, Tensor& out,
                                          cudaStream_t stream, float keep_frac = 1.0f,
                                          const GqaSmallTKeepScratch& keep = {},
                                          GqaS3DecodeRankDump* rank_dump = nullptr);

void gqa_attention_prompt_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                 const Tensor& positions, const Tensor& valid_columns,
                                 const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
                                 Tensor& out, cudaStream_t stream, float keep_frac = 1.0f,
                                 GqaS3PrefillDump* dump = nullptr);

void gqa_kv_append_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                          PagedKVLayerView cache, cudaStream_t stream);

void gqa_attention_prompt_attention_launch(const Tensor& q, const Tensor& positions, float scale,
                                           const PagedKVLayerView& cache, Tensor& out,
                                           cudaStream_t stream, float keep_frac = 1.0f,
                                           GqaS3PrefillDump* dump = nullptr,
                                           std::uint32_t* dbg_regs = nullptr,
                                           std::uint8_t* dbg_q = nullptr);

} // namespace ninfer::ops::detail
