#pragma once

#include "core/arena.h"
#include "core/paged_kv_cache.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Host execution-resource promise for bidirectional_gqa_attention.
 *
 * Device context_lengths define the exact per-row mathematical contexts. This envelope bounds
 * every row so a fixed launch can be captured and replayed without a host read.
 */
struct GqaContextExecutionEnvelope {
    std::uint32_t min_context = 0;
    std::uint32_t max_context = 0;
};

/**
 * Op: bidirectional grouped-query attention over persistent context and one query block
 *
 * For either (D,Hq,Hkv)=(128,32,8) or (256,24,2), query row i, query head h,
 * and kvh=floor(h/(Hq/Hkv)):
 *
 *   keys   = context K rows [0,L) followed logically by every live query K row [0,V)
 *   score  = scale * dot(q[:,h,i], key[:,kvh,j])
 *   prob   = softmax over the complete logical key set
 *   ideal[:,h,i] = sum_j prob[j] * value[:,kvh,j]
 *
 * q/out are contiguous BF16 [D,Hq,W,B]. query_k/query_v are contiguous BF16 [D,Hkv,W,B].
 * context_lengths, valid_columns, and table_rows are contiguous device I32 [B]. Row b has
 * V=valid_columns[b] live query columns and reads logical context [0,context_lengths[b]) through
 * table row table_rows[b]. Columns i>=V are an inert physical tail and produce zero output.
 * context is a read-only paged BF16 cache with head-major page planes [D,64,Nphysical,Hkv].
 * scale is 1/sqrt(D).
 *
 * There is no causal triangle: every live query row attends every other live query K/V row in the
 * same batch row. Context and query K/V remain separate physical segments and every input/cache
 * byte is unchanged. The oracle evaluates `ideal` naively in FP64 from represented inputs. The
 * BF16 out is promoted and compared directly with that result; output storage rounding belongs to
 * the Op's numerical criterion, not the oracle. out is the only observable mutation and is
 * completely overwritten. The current optimized implementation domain is W=1..16 on sm_120a.
 *
 * The caller guarantees min_context <= L <= max_context and that every logical page intersecting
 * [0,L) is materialized. The execution envelope may affect finite launch selection and workspace
 * capacity, never the admitted key set or numerical result.
 */
void bidirectional_gqa_attention(const Tensor& q, const Tensor& query_k, const Tensor& query_v,
                                 const Tensor& context_lengths, const Tensor& valid_columns,
                                 const Tensor& table_rows, float scale,
                                 const PagedKVBatchLayerView& context,
                                 GqaContextExecutionEnvelope envelope, WorkspaceArena& workspace,
                                 Tensor& out, cudaStream_t stream);

/**
 * Returns the transient arena capacity required for every T in the inclusive optimized interval.
 * The execution envelope is the fixed profile; invalid profiles or intervals throw. head_dim
 * selects one of the two exact geometries above (128 or 256).
 */
[[nodiscard]] std::size_t bidirectional_gqa_attention_workspace_capacity_bytes(
    GqaContextExecutionEnvelope envelope, std::int32_t min_tokens, std::int32_t max_tokens,
    std::int32_t batch_size, std::int32_t head_dim);

} // namespace ninfer::ops
