#pragma once

#include "ninfer/ops/kv_cache_append_prefix.h"

namespace ninfer::ops {

/**
 * Op: normalize and rotate K, then append device-selected K/V prefixes to a BF16 cyclic cache.
 *
 * For represented BF16 K[d,h,t,b], BF16 gamma[d], and absolute I32 position p[t,b]:
 *   inv = 1 / sqrt(sum_d K[d,h,t,b]^2 / 128 + epsilon)
 *   z[d] = K[d,h,t,b] * inv * gamma[d]
 *   phi[j] = p[t,b] * theta^(-2*j/128), 0 <= j < 64
 *   rotated[j] = z[j]*cos(phi[j]) - z[j+64]*sin(phi[j])
 *   rotated[j+64] = z[j+64]*cos(phi[j]) + z[j]*sin(phi[j]).
 *
 * k/v are contiguous BF16 [128,8,W,B], gamma is contiguous BF16 [128], positions is
 * contiguous I32 [W,B], and counts/lanes are contiguous I32 [B]. W is positive, B is 1..6,
 * and epsilon/theta are positive and finite. The cache is contiguous BF16
 * [128,padded_capacity,8,lane_capacity] with capacity 2048, padded_capacity >= capacity,
 * and B <= lane_capacity <= 6.
 * All tensor storage is pairwise non-overlapping and at least four-byte aligned.
 *
 * For each t < counts[b], write the final represented BF16 rotated K to cache slot
 * [d,p[t,b] mod 2048,h,lanes[b]] and copy V[d,h,t,b] bit-for-bit to the same V slot.
 * Every other cache byte and all input storage remain unchanged. No intermediate normalized
 * tensor is observable, and its precision and kernel reduction order are implementation choices.
 * The independent oracle evaluates the complete norm/rotation formula naively in FP64 from
 * represented inputs; the final BF16 cache K is promoted and compared directly with that result.
 *
 * The caller guarantees valid distinct destination lanes, sequential nonnegative positions,
 * 0 <= counts[b] <= W within the execution envelope, and envelope.max_count <= 2048.
 * Each row's live interval ends immediately before positions[0,b]; advancing it by counts[b]
 * makes every overwritten old slot dead. The envelope fixes launch capacity on every captured
 * replay. It does not select accepted tokens. The Op never decides or publishes a frontier.
 *
 * No workspace or host/device synchronization is required. Invalid host-visible geometry,
 * scalar parameters, alignment, aliasing, or envelope raises std::invalid_argument before launch;
 * launch failure is reported through the common CUDA error mechanism. Device metadata validity
 * remains the caller's execution promise.
 */
void normalized_rope_kv_append(const Tensor& k, const Tensor& v, const Tensor& gamma,
                               const Tensor& positions, const Tensor& counts, const Tensor& lanes,
                               float epsilon, float theta,
                               KVCacheAppendPrefixExecutionEnvelope envelope,
                               CyclicKVCacheLayerView cache, cudaStream_t stream);

} // namespace ninfer::ops
