#pragma once

#include "core/paged_kv_cache.h"
#include "core/tensor.h"

#include <cuda_runtime.h> // cudaStream_t

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

inline constexpr std::uint32_t kGqaAttentionMaximumVisibleKeys = 262144;

struct GqaExecutionEnvelope {
    std::uint32_t min_visible_keys = 0;
    std::uint32_t max_visible_keys = 0;
};

struct GqaS3PrefillDump;

/**
 * Shared numerical contract for A1/A2/A3.
 *
 * Public q/k/v inputs and BF16 cache values are interpreted after their BF16 storage boundary.
 * INT8-G64 cache rows use one FP16 scale for each contiguous 64-element group. For BF16 source
 * values x, their exact observable encoding is:
 *
 *   a          = max_i abs(FP32(x[i]))
 *   scale_bits = FP16_RNE(a / 127)
 *   s          = FP32(scale_bits)
 *   inv        = s == 0 ? 0 : FP32(1 / s)
 *   code[i]    = s == 0 ? 0 : I8(clamp(RNE_even(FP32(x[i]) * inv), -127, 127))
 *   decode[i]  = FP32(code[i]) * s
 *
 * NVFP4-G16 cache rows pack two E2M1 values per U8 along D (leading extent 128) and store one
 * UE4M3 scale per contiguous 16-element group (leading extent 16). For BF16 source values x:
 *
 *   a          = max_i abs(FP32(x[i])) over the 16-wide group
 *   scale_bits = E4M3_SATFINITE_RNE(a / 6)
 *   s          = FP32(decode_e4m3(scale_bits))
 *   code[i]    = s == 0 ? 0 : E2M1_SATFINITE_RNE(FP32(x[i]) / s)
 *   decode[i]  = FP32(decode_e2m1(code[i])) * s
 *
 * A1 and A2 produce identical code and scale bits. The common ideal attention oracle uses BF16 Q
 * and logical cache values (BF16 values for a BF16 cache, FP32 decode above for INT8-G64, and
 * decode_e2m1(code)*decode_e4m3(scale) for NVFP4-G16), then evaluates score dot products, stable
 * softmax, and value reduction in FP64. The BF16 Op output is promoted to FP64 for comparison with
 * that result.
 *
 * The registered INT8 implementation defines Q8-G64, paired with INT8-G64 K, as its native query
 * compute profile. The registered NVFP4 implementation defines Q-NVFP4-G16, paired with NVFP4-G16 K,
 * as its native query compute profile using m16n8k64 hardware block scales. Those profile-defined
 * query quantizations and any narrower staging do not replace BF16 Q in the ideal oracle. BF16-cache,
 * INT8-cache, and NVFP4-cache compute profiles therefore have separate named numerical criteria
 * owned by the GQA conformance test. Those envelopes apply to the registered geometries, tested
 * token extents, conformance matrix, and target-representative activation range; they are not a
 * universal error bound for arbitrary adversarial BF16 tensors. A1 and A3 are each qualified
 * directly against the ideal oracle. A1-versus-A3 parity is only an additional consistency check.
 */

/**
 * Returns the transient arena capacity required for every W in the inclusive interval at one
 * exact logical batch size. Head geometry, cache dtype, and execution envelope are the fixed
 * implementation profile. Invalid profiles or intervals throw; a legal B=1 prompt route may
 * return zero. Prefill Sparge keep-lists live in kernel smem. XAttention ranker
 * scratch (packed-K / logits / mass / keep) is sized here when xattn_tau < 1 on
 * a U8 Prompt route; it is allocated from the caller workspace. keep_frac < 1.0
 * on a sage U8 cache still sizes the sparge-decode tile-skip scratch for W=1
 * cached SmallT; the default 1.0 is exact (zero added). tree_verify=true reserves decode
 * small-T / chunked-small-T scratch for packed-tree A1, including B=1 W=7..16 where ordinary
 * causal A1 would use the Prompt route.
 */
[[nodiscard]] std::size_t
gqa_attention_workspace_capacity_bytes(std::int32_t q_heads, DType cache_dtype,
                                       GqaExecutionEnvelope envelope, std::int32_t batch_size,
                                       std::int32_t min_width, std::int32_t max_width,
                                       float keep_frac = 1.0f, bool tree_verify = false,
                                       float xattn_tau = 1.0f);

/**
 * A1: append K/V for B independent sequences and compute causal grouped-query attention. Let
 * Vb=W when valid_columns is empty and Vb=valid_columns[b] otherwise. For row b, query head h,
 * kvh=floor(h/group), 0<=j<Vb, p=positions[j,b], and that row's populated cache history [0,p]:
 *
 *   score[x]      = scale * dot(q[:,h,j,b], K_cache[b][:,x,kvh]),  0 <= x <= p
 *   probability   = softmax_x(score)
 *   ideal[:,h,j,b] = sum_x probability[x] * V_cache[b][:,x,kvh].
 *
 * The registered geometries are `[256,24|4,W,B]` group 6 and `[256,16|2,W,B]` group 8.
 * q/k/v/out are contiguous BF16 in request-major order, positions is contiguous I32 [W,B], and
 * kv_table_rows is contiguous I32 [B]. valid_columns is either contiguous I32 [B], or an empty
 * Tensor meaning every row has exactly W valid columns. This dense/masked choice is part of the
 * call topology; it is not inferred by copying device metadata to the host. B=1 accepts every
 * positive W in the current prefill/decode domain; B=2..8 accepts W=1..16. Cache storage is BF16,
 * INT8-G64, or NVFP4-G16 under the shared numerical contract above. PagedKVBatchLayerView supplies shared
 * planes and the complete block-table matrix; kv_table_rows[b] selects one row for sequence b.
 *
 * In masked form, every row's valid columns are the prefix [0,valid_columns[b]); positions in that
 * prefix are sequential and address populated causal histories. Each nonempty row repeats its
 * final valid position through the invalid tail; an empty row uses zero positions. Other
 * invalid-tail inputs contain safe dummy values. A1 does not modify cache for invalid columns and
 * writes exact BF16 zero to their output. The caller guarantees that the maximum final valid
 * position plus one over nonempty rows lies in the declared execution envelope. The envelope is a
 * host launch-resource promise over that batch maximum; it does not alter any row's causal mask.
 *
 * q/k/v/positions/valid_columns/kv_table_rows/out, every cache plane/table, and live workspace
 * suballocations are pairwise non-overlapping. The Op overwrites every addressed cache row but
 * owns no persistent frontier, allocation, request identity, or commit authority.
 *
 * Prefill tile-skip on exact NVFP4 (Prompt route, T>6): keep_frac in (0,1] is Sparge meansim
 * (1.0 = dense). xattn_tau in (0,1] is XAttention mass threshold (1.0 = dense). The two are
 * mutually exclusive when both are < 1. Both require NVFP4 without sage_pv. SmallT/cached
 * routes ignore skip flags. dump is a test side-band (keep_list / tile_count); production
 * passes nullptr.
 *
 * Packed-tree verify is off when ancestor_mask and prefix_lengths are empty. When both are
 * populated, A1 still writes K/V at positions[j,b] but query j attends key x iff x <=
 * positions[j,b] and either x < prefix_lengths[b] or packed = x - prefix_lengths[b] is in [0, W)
 * and bit packed of ancestor_mask[j,b] is set. ancestor_mask is contiguous I32 [W,B];
 * prefix_lengths is contiguous I32 [B]. Bit i of ancestor_mask[j] means packed column i is an
 * ancestor of j, including j itself. The caller assigns unique cache slots positions[j,b] = E + j
 * rather than E + depth; RoPE remains the caller's pre-Op responsibility. Tree verify uses the
 * decode small-T route at W=1..16 (chunked for W>6), including B=1; it is not a Prompt-kernel
 * path. gqa_attention_workspace_capacity_bytes(..., tree_verify=true) reserves that decode
 * scratch. Ordinary MTP/causal calls omit both tensors.
 */
void gqa_attention(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& positions,
                   const Tensor& valid_columns, const Tensor& kv_table_rows, float scale,
                   PagedKVBatchLayerView cache, GqaExecutionEnvelope envelope,
                   WorkspaceArena& workspace, Tensor& out, cudaStream_t stream,
                   float keep_frac = 1.0f, float xattn_tau = 1.0f,
                   std::int32_t xattn_min_len = 8192, GqaS3PrefillDump* dump = nullptr,
                   const Tensor& ancestor_mask = {}, const Tensor& prefix_lengths = {});

/**
 * A2: perform only the cache-write part of A1. k/v are contiguous BF16 `[256,4|2,T]`, positions is
 * contiguous sequential I32 [T], and every addressed code and quantized scale is overwritten. It reads
 * no unrelated cache row, receives no execution envelope, and owns no persistent frontier.
 */
void gqa_kv_append(const Tensor& k, const Tensor& v, const Tensor& positions,
                   PagedKVLayerView cache, cudaStream_t stream);

/**
 * Compact packed-tree KV slots onto a sequential prefix. For each row b and i < counts[b], token
 * prefix_lengths[b] + path[i,b] is copied onto prefix_lengths[b] + i. path is strictly increasing
 * packed indices, so the copy is safe in increasing i. kv_table_rows, prefix_lengths, and counts
 * are I32 [B]; path is I32 [W,B]. The cache view is the same batched GQA layer consumed by A1.
 */
void gqa_kv_compact_path(PagedKVBatchLayerView cache, const Tensor& kv_table_rows,
                         const Tensor& prefix_lengths, const Tensor& path, const Tensor& counts,
                         cudaStream_t stream);

/**
 * Dev/test side-band for the sparge-decode tile-skip rank (the tools/kdev
 * op-dump tooling). Not part of the production API: the production path never
 * sees a non-null dump. The caller supplies device arrays sized [batch*KVHeads]
 * (keep_count) and [batch*KVHeads][max_tiles] (keep_tiles, max_tiles = the
 * rank's tile capacity div_up(window, 32)); after a gqa_attention_cached call
 * with keep_frac < 1 the rank kernel's keep set is copied into them, so a host
 * oracle can rebuild the exact keep set and score the kept-tile reference
 * attention.
 */
struct GqaS3DecodeRankDump {
    std::int32_t max_tiles; // [kv-row] keep-list stride (>= div_up(window, 32))
    std::int32_t* keep_tiles; // [batch*KVHeads][max_tiles] kept key-tile index
    std::int32_t* keep_count; // [batch*KVHeads] kept key-tile count
    std::int32_t splits;      // launch split count (0 when skip did not engage)
    std::int32_t* split_off;  // [batch*KVHeads][splits+1] per-split keep-slice offset
};

// Runs the cached (A3) attention with an optional sparge-decode tile-skip
// keep_frac (see gqa_attention's keep_frac docs; 1.0 = exact). With keep_frac
// < 1 on a sage cache, the T=1 step ranks the window's key tiles with the
// k_mean meansim proxy and computes attention over the kept tiles only
// (sinks + recency window + top fraction). The cached route is the PPL
// decode lane; width > 1 (MTP verify) never engages the skip.
/**
 * A3: compute causal attention from an already populated cache without accepting new K/V or
 * mutating any cache plane. q/out are contiguous BF16 `[256,24|16,T]`, positions is contiguous
 * sequential I32 [T], and the mathematical formula and execution-envelope contract are identical
 * to A1. Caller workspace is reported by gqa_attention_workspace_capacity_bytes().
 */
void gqa_attention_cached(const Tensor& q, const Tensor& positions, float scale,
                          const PagedKVLayerView& cache, GqaExecutionEnvelope envelope,
                          WorkspaceArena& workspace, Tensor& out, cudaStream_t stream,
                          float keep_frac = 1.0f, GqaS3DecodeRankDump* rank_dump = nullptr);

/**
 * Dev/test side-band for the SageAttention3 (nvfp4s3) prefill kernel (the tools/kdev
 * op-dump tooling). Not part of the production API: the production path never sees a
 * non-null dump, so the pointers below are never dereferenced on it.
 *
 * The caller supplies device arrays sized for grid q_block 0 (q_block > 0 is not
 * dumped), indexed [q_head][tile][...], with `max_tiles` = div_up(envelope_max, 64)
 * (key tile Bc=64). All values are the kernel's named intermediates, captured at the
 * stage where the kernel computes them, so a diff against the FP64 step-exact
 * reference localizes the first divergent stage (QK score vs P-quant codes vs V
 * scale vs online-softmax m/l vs PV acc) without editing the kernel.
 */
struct GqaS3PrefillDump {
    int max_tiles; // [h][t] stride (>= the actual key-tile count)
    float* score; // [h][t][128 rows][64 keys] raw QK dot (pre-scale); -INFINITY = causally masked key
    std::uint8_t* p_code; // [h][t][128][64] e2m1 P-quant nibble (0..7); 0xFF = masked key
    std::uint8_t* psf; // [h][t][128][4] e4m3 P-block scale byte (per 16-key block)
    std::uint8_t* v_scale; // [h][t][256 d][4] e4m3 V-block scale byte (per (d, 16-key block))
    std::uint8_t* v_t; // [h][t][256 d][32] transposed V-code B operand (per tile)
    float* m; // [h][t][128] running max after the tile (raw score domain)
    float* l; // [h][t][128] running L after the tile (amplified, tile frame)
    float* acc; // [h][t][128][256] PV accumulator after the tile (tile frame, pre-normalize)
    std::int32_t* keep_list; // [h][max_tiles] kept key-tile index (valid up to tile_count)
    std::int32_t* tile_count; // [h] tiles actually processed
};

// Runs the same prompt fill + attention as gqa_attention (A1 Prompt route) and
// writes the named s3 intermediates into `dump` on the Sage (U8 + sage_pv) cache.
// Throws on the SmallT/ChunkedSmallT routes or on a non-sage cache, where the s3
// prefill kernel does not run and the dump would silently stay empty.
void gqa_attention_s3_dump(const Tensor& q, const Tensor& k, const Tensor& v,
                           const Tensor& positions, const Tensor& valid_columns,
                           const Tensor& kv_table_rows, float scale,
                           PagedKVBatchLayerView cache, GqaExecutionEnvelope envelope,
                           WorkspaceArena& workspace, Tensor& out, cudaStream_t stream,
                           float keep_frac, GqaS3PrefillDump& dump);

} // namespace ninfer::ops
