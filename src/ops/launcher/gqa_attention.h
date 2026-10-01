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
    const Tensor* valid_columns  = nullptr;
    const Tensor* table_rows     = nullptr;
    const Tensor* ancestor_mask  = nullptr;
    const Tensor* prefix_lengths = nullptr;
    std::int32_t full_width      = 0;
    std::int32_t column_begin    = 0;
    std::int32_t width           = 0;
    std::int32_t batch_size      = 1;
};

std::int32_t gqa_attention_split_capacity(std::int32_t q_heads, std::int32_t tokens,
                                          DType cache_dtype, GqaExecutionEnvelope envelope);

bool gqa_attention_uses_small_t(std::int32_t tokens);

// dense_nvfp4: the call would run the dense NVFP4 Prompt kernel (U8 cache without S3, no
// Sparge/XAttention skip, no dump). Only such 27B calls move short appends to six-row chunks;
// every skip, S3, and dump profile keeps its Prompt-route semantics.
GqaAttentionRoute gqa_attention_resolve_route(std::int32_t q_heads, std::int32_t width,
                                              std::int32_t batch_size,
                                              GqaExecutionEnvelope envelope,
                                              bool tree_verify = false, bool dense_nvfp4 = false);

const char* gqa_attention_route_name(GqaAttentionRoute route);

// Context-split execution of the dense NVFP4 Prompt kernel. splits <= 1 selects the unsplit
// kernel; otherwise acc is FP32 [256, q_heads, W, splits] and m/l are FP32 [q_heads, W, splits].
struct GqaPromptSplit {
    std::int32_t splits = 1;
    Tensor acc;
    Tensor m;
    Tensor l;
};

// Widest Prompt call that can split (W * splits within the partial-row budget).
inline constexpr std::int32_t kGqaPromptSplitMaximumWidth = 1024;

// Partial-buffer split capacity for one dense NVFP4 Prompt call over at most visible_keys keys.
// It is non-decreasing in visible_keys, so the capacity at the envelope maximum covers every
// call a workspace query admits. 1 means the call never splits, including every W above
// kGqaPromptSplitMaximumWidth and every grid that already fills whole waves.
std::int32_t gqa_attention_prompt_split_capacity(std::int32_t q_heads, std::int32_t width,
                                                 std::uint32_t visible_keys);

// Launch split count in [1, capacity]: the count that minimizes whole-wave key-tile work.
std::int32_t gqa_attention_prompt_splits(std::int32_t q_heads, std::int32_t width,
                                         std::uint32_t visible_keys);

struct GqaSmallTKeepScratch {
    // Sparge-decode tile-skip scratch (all empty unless the sage tile-skip is
    // engaged: keep_frac < 1 + the k_mean plane + a T=1 step).
    Tensor keep_tiles;         // [batch*KVHeads][max_keep] kept-tile index (i32)
    Tensor keep_count;         // [batch*KVHeads] kept-tile count (i32)
    Tensor split_off;          // [batch*KVHeads][splits+1] per-split keep prefix (i32)
    std::int32_t max_keep = 0; // per-(kv,batch) keep-list capacity
};

void gqa_attention_small_t_launch(
    const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
    const Tensor& valid_columns, const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
    GqaExecutionEnvelope envelope, std::int32_t column_begin, std::int32_t width,
    Tensor& partial_acc, Tensor& partial_m, Tensor& partial_l, Tensor& out, cudaStream_t stream,
    const Tensor& ancestor_mask = {}, const Tensor& prefix_lengths = {}, float keep_frac = 1.0f,
    const GqaSmallTKeepScratch& keep = {});

void gqa_attention_cached_small_t_launch(const Tensor& q, const Tensor& positions, float scale,
                                         const PagedKVLayerView& cache,
                                         GqaExecutionEnvelope envelope, Tensor& partial_acc,
                                         Tensor& partial_m, Tensor& partial_l, Tensor& out,
                                         cudaStream_t stream, float keep_frac = 1.0f,
                                         const GqaSmallTKeepScratch& keep = {},
                                         GqaS3DecodeRankDump* rank_dump   = nullptr);

void gqa_attention_prompt_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                 const Tensor& positions, const Tensor& valid_columns,
                                 const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
                                 Tensor& out, cudaStream_t stream, float keep_frac = 1.0f,
                                 float xattn_tau = 1.0f, std::int32_t xattn_min_len = 8192,
                                 GqaS3PrefillDump* dump = nullptr, void* xattn_scratch = nullptr,
                                 GqaExecutionEnvelope envelope = {1,
                                                                  kGqaAttentionMaximumVisibleKeys},
                                 const GqaPromptSplit& split   = {});

void gqa_kv_append_launch(const Tensor& k, const Tensor& v, const Tensor& positions,
                          PagedKVLayerView cache, cudaStream_t stream);

void gqa_kv_compact_path_launch(PagedKVBatchLayerView cache, const Tensor& kv_table_rows,
                                const Tensor& prefix_lengths, const Tensor& path,
                                const Tensor& counts, cudaStream_t stream);

void gqa_attention_prompt_attention_launch(
    const Tensor& q, const Tensor& positions, float scale, const PagedKVLayerView& cache,
    Tensor& out, cudaStream_t stream, float keep_frac = 1.0f, float xattn_tau = 1.0f,
    std::int32_t xattn_min_len = 8192, GqaS3PrefillDump* dump = nullptr,
    std::uint32_t* dbg_regs = nullptr, std::uint8_t* dbg_q = nullptr, void* xattn_scratch = nullptr,
    GqaExecutionEnvelope envelope = {1, kGqaAttentionMaximumVisibleKeys},
    const GqaPromptSplit& split   = {});

} // namespace ninfer::ops::detail
