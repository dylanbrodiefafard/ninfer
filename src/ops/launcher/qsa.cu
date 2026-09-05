#include "ops/launcher/qsa.h"

#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_codec.cuh"

#include <cuda_bf16.h>

#include <cfloat>
#include <cmath>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kCodeWidth = kQsaHeadDim / 2;
constexpr int kGroups    = kQsaHeadDim / 16;
constexpr int kKeep      = 512;
constexpr int kMaximumBlocks = kQsaMaximumTokens / 4;
constexpr int kSelectorScratchFloats = kQsaIndexHeadDim * kQsaIndexQueryHeads + kMaximumBlocks;
constexpr int kAttentionHeadsPerGroup = 4;
constexpr int kAttentionHeadGroups = kQsaQueryHeads / kAttentionHeadsPerGroup;
constexpr int kAttentionGroupsPerKvHead =
    kQsaQueryHeads / kQsaKvHeads / kAttentionHeadsPerGroup;
constexpr int kAttentionScoreRanksPerBlock = 16;
constexpr int kAttentionScoreRanksPerStep = 2;
constexpr int kAttentionValueTile = 64;
constexpr int kAttentionScoreFloats =
    ((kQsaQueryHeads * kQsaSelectedCapacity + 63) / 64) * 64;
constexpr float kEpsilon = 1.0e-6F;

__device__ __forceinline__ std::int64_t code_index(int byte, int token, int head, int capacity) {
    return static_cast<std::int64_t>(byte) + static_cast<std::int64_t>(kCodeWidth) *
               (static_cast<std::int64_t>(token) + static_cast<std::int64_t>(capacity) * head);
}

__device__ __forceinline__ std::int64_t scale_index(int group, int token, int head, int capacity) {
    return static_cast<std::int64_t>(group) + static_cast<std::int64_t>(kGroups) *
               (static_cast<std::int64_t>(token) + static_cast<std::int64_t>(capacity) * head);
}

__device__ __forceinline__ float decode_value(const std::uint8_t* codes,
                                               const std::uint8_t* scales, int d, int token,
                                               int head, int capacity) {
    const std::uint8_t packed = codes[code_index(d >> 1, token, head, capacity)];
    const float2 pair         = decode_nvfp4_e2m1x2(packed);
    const float scale         = decode_nvfp4_e4m3(scales[scale_index(d >> 4, token, head, capacity)]);
    return ((d & 1) == 0 ? pair.x : pair.y) * scale;
}

__global__ void qsa_append_nvfp4_kernel(const __nv_bfloat16* k, const __nv_bfloat16* v,
                                         const std::int32_t* ids, std::uint8_t* k_codes,
                                         std::uint8_t* v_codes, std::uint8_t* k_scales,
                                         std::uint8_t* v_scales, int width, int capacity) {
    const int group = blockIdx.x;
    const int token = blockIdx.y;
    const int head  = blockIdx.z;
    const int id    = ids[token];
    if (group >= kGroups || token >= width || head >= kQsaKvHeads || id < 0 || id >= capacity) {
        return;
    }
    const std::int64_t source = static_cast<std::int64_t>(group) * 16 +
                                static_cast<std::int64_t>(kQsaHeadDim) *
                                    (head + static_cast<std::int64_t>(kQsaKvHeads) * token);
    const Nvfp4QuantizedK16 kq = quantize_nvfp4_k16(k + source, 1.0F);
    const Nvfp4QuantizedK16 vq = quantize_nvfp4_k16(v + source, 1.0F);
    const std::int64_t dst = code_index(group * 8, id, head, capacity);
    *reinterpret_cast<std::uint32_t*>(k_codes + dst)     = kq.codes_lo;
    *reinterpret_cast<std::uint32_t*>(k_codes + dst + 4) = kq.codes_hi;
    *reinterpret_cast<std::uint32_t*>(v_codes + dst)     = vq.codes_lo;
    *reinterpret_cast<std::uint32_t*>(v_codes + dst + 4) = vq.codes_hi;
    k_scales[scale_index(group, id, head, capacity)]      = kq.scale;
    v_scales[scale_index(group, id, head, capacity)]      = vq.scale;
}

__global__ void qsa_append_index_kernel(const __nv_bfloat16* raw_keys,
                                         const std::int32_t* positions,
                                         const std::int32_t* ids, __nv_bfloat16* state_keys,
                                         std::int32_t* state_positions, int width, int capacity) {
    const int token = blockIdx.x;
    const int lane  = threadIdx.x;
    if (token >= width) { return; }
    const int id = ids[token];
    if (id < 0 || id >= capacity) { return; }
    if (lane < kQsaIndexHeadDim) {
        state_keys[lane + static_cast<std::int64_t>(kQsaIndexHeadDim) * id] =
            raw_keys[lane + static_cast<std::int64_t>(kQsaIndexHeadDim) * token];
    }
    if (lane < 3) {
        state_positions[lane + 3LL * id] = positions[lane + 3LL * token];
    }
}

__device__ __forceinline__ float rope_component(float x0, float x1, int pair, bool upper,
                                                 const std::int32_t* position) {
    const float inv_frequency = powf(1.0e7F, -static_cast<float>(pair) / 32.0F);
    const float phase = static_cast<float>(position[pair % 3]) * inv_frequency;
    float sine;
    float cosine;
    sincosf(phase, &sine, &cosine);
    return upper ? x1 * cosine + x0 * sine : x0 * cosine - x1 * sine;
}

__device__ __forceinline__ float normalized_query(const __nv_bfloat16* query,
                                                   const float* weight, int d, float inv_rms) {
    return __bfloat162float(query[d]) * inv_rms * weight[d];
}

__global__ void qsa_select_prepare_kernel(const __nv_bfloat16* raw_query,
                                          const std::int32_t* state_positions,
                                          const std::int32_t* query_ids,
                                          const std::int32_t* visible_ids,
                                          const std::int32_t* visible_offsets,
                                          const float* query_weight, std::int32_t* selected,
                                          std::int32_t* selected_count, float* scratch, int width,
                                          int visible_extent, int capacity) {
    __shared__ int shared_query_id;
    __shared__ int shared_begin;
    __shared__ int shared_end;
    __shared__ int valid;
    __shared__ float query_inv_rms[kQsaIndexQueryHeads];
    const int token = blockIdx.x;
    const int lane = threadIdx.x;
    if (token >= width) { return; }
    std::int32_t* output = selected + static_cast<std::int64_t>(kQsaSelectedCapacity) * token;
    for (int i = lane; i < kQsaSelectedCapacity; i += blockDim.x) { output[i] = -1; }
    if (lane == 0) {
        selected_count[token] = 0;
        shared_query_id = query_ids[token];
        shared_begin = visible_offsets[token];
        shared_end = visible_offsets[token + 1];
        valid = shared_query_id >= 0 && shared_query_id < capacity && shared_begin >= 0 &&
                shared_end > shared_begin && shared_end <= visible_extent &&
                shared_end - shared_begin <= capacity;
    }
    __syncthreads();
    if (!valid) { return; }
    for (int i = shared_begin + lane; i < shared_end; i += blockDim.x) {
        if (visible_ids[i] < 0 || visible_ids[i] >= capacity ||
            (i > shared_begin && visible_ids[i] <= visible_ids[i - 1])) {
            atomicExch(&valid, 0);
        }
    }
    __syncthreads();
    if (!valid) { return; }

    float* q_rot = scratch + static_cast<std::int64_t>(token) * kSelectorScratchFloats;
    if (lane == 0) {
        for (int head = 0; head < kQsaIndexQueryHeads; ++head) {
            const __nv_bfloat16* q =
                raw_query + static_cast<std::int64_t>(kQsaIndexHeadDim) *
                                (head + kQsaIndexQueryHeads * token);
            float sum2 = 0.0F;
            for (int d = 0; d < kQsaIndexHeadDim; ++d) {
                const float value = __bfloat162float(q[d]);
                sum2 += value * value;
            }
            query_inv_rms[head] = rsqrtf(sum2 / kQsaIndexHeadDim + kEpsilon);
        }
    }
    __syncthreads();
    const std::int32_t* query_position = state_positions + 3LL * shared_query_id;
    if (lane < kQsaIndexHeadDim) {
        for (int head = 0; head < kQsaIndexQueryHeads; ++head) {
            const __nv_bfloat16* q =
                raw_query + static_cast<std::int64_t>(kQsaIndexHeadDim) *
                                (head + kQsaIndexQueryHeads * token);
            float value = normalized_query(q, query_weight, lane, query_inv_rms[head]);
            if (lane < 64) {
                const int pair = lane & 31;
                const float lo = normalized_query(q, query_weight, pair, query_inv_rms[head]);
                const float hi =
                    normalized_query(q, query_weight, pair + 32, query_inv_rms[head]);
                value = rope_component(lo, hi, pair, lane >= 32, query_position);
            }
            q_rot[lane + kQsaIndexHeadDim * head] = value;
        }
    }
    __syncthreads();
    // A negative count is private inter-kernel readiness state on this stream. Malformed and
    // invalid columns retain the public empty result initialized above.
    if (lane == 0) { selected_count[token] = -1; }
}

__global__ void qsa_select_score_kernel(const __nv_bfloat16* raw_keys,
                                        const std::int32_t* state_positions,
                                        const std::int32_t* visible_ids,
                                        const std::int32_t* visible_offsets,
                                        const float* key_weight, const std::int32_t* selected_count,
                                        float* scratch, int width) {
    const int block = blockIdx.x;
    const int token = blockIdx.y;
    if (threadIdx.x != 0 || token >= width || selected_count[token] != -1) { return; }
    const int begin = visible_offsets[token];
    const int end   = visible_offsets[token + 1];
    const int complete = (end - begin) / 4;
    if (block >= complete) { return; }
    float* q_rot = scratch + static_cast<std::int64_t>(token) * kSelectorScratchFloats;
    float* block_scores = q_rot + kQsaIndexHeadDim * kQsaIndexQueryHeads;

    float key_sum2 = 0.0F;
    for (int d = 0; d < kQsaIndexHeadDim; ++d) {
        float pooled = 0.0F;
        for (int r = 0; r < 4; ++r) {
            const int id = visible_ids[begin + block * 4 + r];
            pooled += __bfloat162float(
                raw_keys[d + static_cast<std::int64_t>(kQsaIndexHeadDim) * id]);
        }
        const float represented = __bfloat162float(__float2bfloat16_rn(pooled * 0.25F));
        key_sum2 += represented * represented;
    }
    const float key_inv = rsqrtf(key_sum2 / kQsaIndexHeadDim + kEpsilon);
    const int block_id = visible_ids[begin + block * 4];
    const std::int32_t* block_position = state_positions + 3LL * block_id;
    float score = 0.0F;
    for (int head = 0; head < kQsaIndexQueryHeads; ++head) {
        float dot = 0.0F;
        for (int d = 0; d < kQsaIndexHeadDim; ++d) {
            auto pooled_at = [&](int kd) {
                float pooled = 0.0F;
                for (int r = 0; r < 4; ++r) {
                    const int id = visible_ids[begin + block * 4 + r];
                    pooled += __bfloat162float(
                        raw_keys[kd + static_cast<std::int64_t>(kQsaIndexHeadDim) * id]);
                }
                const float represented =
                    __bfloat162float(__float2bfloat16_rn(pooled * 0.25F));
                return represented * key_inv * key_weight[kd];
            };
            float key_value = pooled_at(d);
            if (d < 64) {
                const int pair = d & 31;
                key_value = rope_component(pooled_at(pair), pooled_at(pair + 32), pair, d >= 32,
                                           block_position);
            }
            dot += q_rot[d + kQsaIndexHeadDim * head] * key_value;
        }
        score += fmaxf(dot, 0.0F);
    }
    score *= 0.08838834764831845F; // 1/sqrt(128)
    block_scores[block] = score;
}

__device__ __forceinline__ bool qsa_block_is_better(float lhs_score, int lhs_block,
                                                    float rhs_score, int rhs_block) {
    return lhs_score > rhs_score || (lhs_score == rhs_score && lhs_block < rhs_block);
}

__global__ void qsa_select_topk_short_kernel(const std::int32_t* visible_ids,
                                             const std::int32_t* visible_offsets,
                                             std::int32_t* selected,
                                             std::int32_t* selected_count, float* scratch,
                                             int width, int sort_blocks) {
    __shared__ float scores[kKeep];
    __shared__ std::int32_t blocks[kKeep];
    const int token = blockIdx.x;
    const int lane = threadIdx.x;
    if (token >= width || selected_count[token] != -1) { return; }
    const int begin = visible_offsets[token];
    const int end = visible_offsets[token + 1];
    const int complete = (end - begin) / 4;
    const float* block_scores =
        scratch + static_cast<std::int64_t>(token) * kSelectorScratchFloats +
        kQsaIndexHeadDim * kQsaIndexQueryHeads;

    for (int index = lane; index < sort_blocks; index += blockDim.x) {
        if (index < complete) {
            scores[index] = block_scores[index];
            blocks[index] = index;
        } else {
            scores[index] = -FLT_MAX;
            blocks[index] = sort_blocks;
        }
    }
    __syncthreads();

    const int comparators = sort_blocks / 2;
    for (int size = 2; size <= sort_blocks; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            if (lane < comparators) {
                const int left = (lane / stride) * (stride << 1) + lane % stride;
                const int right = left + stride;
                const bool ascending_better = (left & size) == 0;
                const bool right_better = qsa_block_is_better(
                    scores[right], blocks[right], scores[left], blocks[left]);
                const bool left_better = qsa_block_is_better(
                    scores[left], blocks[left], scores[right], blocks[right]);
                if ((ascending_better && right_better) ||
                    (!ascending_better && left_better)) {
                    const float score = scores[left];
                    scores[left] = scores[right];
                    scores[right] = score;
                    const int block = blocks[left];
                    blocks[left] = blocks[right];
                    blocks[right] = block;
                }
            }
            __syncthreads();
        }
    }

    std::int32_t* output =
        selected + static_cast<std::int64_t>(kQsaSelectedCapacity) * token;
    const int kept = complete < kKeep ? complete : kKeep;
    for (int rank = lane; rank < kept; rank += blockDim.x) {
        const int source = begin + 4 * blocks[rank];
        const int destination = 4 * rank;
        for (int r = 0; r < 4; ++r) { output[destination + r] = visible_ids[source + r]; }
    }
    if (lane == 0) {
        int count = 4 * kept;
        for (int i = begin + 4 * complete; i < end; ++i) { output[count++] = visible_ids[i]; }
        selected_count[token] = count;
    }
}

__global__ void qsa_select_topk_kernel(const std::int32_t* visible_ids,
                                       const std::int32_t* visible_offsets,
                                       std::int32_t* selected, std::int32_t* selected_count,
                                       float* scratch, int width) {
    __shared__ float scores[kMaximumBlocks];
    __shared__ std::int32_t blocks[kMaximumBlocks];
    const int token = blockIdx.x;
    const int lane = threadIdx.x;
    if (token >= width || selected_count[token] != -1) { return; }
    const int begin = visible_offsets[token];
    const int end = visible_offsets[token + 1];
    const int complete = (end - begin) / 4;
    const float* block_scores =
        scratch + static_cast<std::int64_t>(token) * kSelectorScratchFloats +
        kQsaIndexHeadDim * kQsaIndexQueryHeads;

    for (int offset = 0; offset < kMaximumBlocks; offset += kKeep) {
        const int index = lane + offset;
        if (index < complete) {
            scores[index] = block_scores[index];
            blocks[index] = index;
        } else {
            scores[index] = -FLT_MAX;
            blocks[index] = kMaximumBlocks;
        }
    }
    __syncthreads();

    // Fixed-size bitonic network. The final order is score descending and logical block rank
    // ascending, so exact ties retain the semantic lower-rank winner at the 512-block boundary.
    for (int size = 2; size <= kMaximumBlocks; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            const int left = (lane / stride) * (stride << 1) + lane % stride;
            const int right = left + stride;
            const bool ascending_better = (left & size) == 0;
            const bool right_better = qsa_block_is_better(scores[right], blocks[right],
                                                          scores[left], blocks[left]);
            const bool left_better = qsa_block_is_better(scores[left], blocks[left],
                                                         scores[right], blocks[right]);
            if ((ascending_better && right_better) || (!ascending_better && left_better)) {
                const float score = scores[left];
                scores[left] = scores[right];
                scores[right] = score;
                const int block = blocks[left];
                blocks[left] = blocks[right];
                blocks[right] = block;
            }
            __syncthreads();
        }
    }

    std::int32_t* output = selected + static_cast<std::int64_t>(kQsaSelectedCapacity) * token;
    const int kept = complete < kKeep ? complete : kKeep;
    if (lane < kept) {
        const int source = begin + 4 * blocks[lane];
        const int destination = 4 * lane;
        for (int r = 0; r < 4; ++r) { output[destination + r] = visible_ids[source + r]; }
    }
    if (lane == 0) {
        int count = 4 * kept;
        for (int i = begin + 4 * complete; i < end; ++i) { output[count++] = visible_ids[i]; }
        selected_count[token] = count;
    }
}

__global__ void qsa_attention_short_kernel(const __nv_bfloat16* q,
                                           const std::int32_t* selected,
                                           const std::int32_t* counts,
                                           const std::uint8_t* k_codes,
                                           const std::uint8_t* v_codes,
                                           const std::uint8_t* k_scales,
                                           const std::uint8_t* v_scales,
                                           __nv_bfloat16* out, int selected_bound,
                                           int capacity, int width) {
    __shared__ float scores[kQsaSelectedCapacity];
    __shared__ float reduction[kQsaHeadDim];
    const int d     = threadIdx.x;
    const int lane  = d & 31;
    const int warp  = d >> 5;
    const int head  = blockIdx.x;
    const int token = blockIdx.y;
    if (token >= width) { return; }
    const int count = counts[token];
    const auto* token_selected =
        selected + static_cast<std::int64_t>(selected_bound) * token;
    const std::int64_t output_index =
        d + static_cast<std::int64_t>(kQsaHeadDim) *
                (head + static_cast<std::int64_t>(kQsaQueryHeads) * token);
    if (count <= 0 || count > selected_bound) {
        out[output_index] = __float2bfloat16_rn(0.0F);
        return;
    }
    const int kv_head = head / 12;
    for (int j = warp; j < count; j += kQsaHeadDim / 32) {
        const int id = token_selected[j];
        float dot = 0.0F;
        if (id >= 0 && id < capacity) {
            for (int feature = lane; feature < kQsaHeadDim; feature += 32) {
                const std::int64_t query_index =
                    feature + static_cast<std::int64_t>(kQsaHeadDim) *
                                  (head + static_cast<std::int64_t>(kQsaQueryHeads) * token);
                dot += __bfloat162float(q[query_index]) *
                       decode_value(k_codes, k_scales, feature, id, kv_head, capacity);
            }
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            dot += __shfl_down_sync(0xffffffffU, dot, offset);
        }
        if (lane == 0) { scores[j] = dot * 0.0625F; }
    }
    __syncthreads();

    float maximum = -FLT_MAX;
    for (int j = d; j < count; j += kQsaHeadDim) { maximum = fmaxf(maximum, scores[j]); }
    reduction[d] = maximum;
    __syncthreads();
    for (int stride = kQsaHeadDim / 2; stride > 0; stride >>= 1) {
        if (d < stride) { reduction[d] = fmaxf(reduction[d], reduction[d + stride]); }
        __syncthreads();
    }
    maximum = reduction[0];

    float sum = 0.0F;
    for (int j = d; j < count; j += kQsaHeadDim) {
        const float probability = expf(scores[j] - maximum);
        scores[j] = probability;
        sum += probability;
    }
    reduction[d] = sum;
    __syncthreads();
    for (int stride = kQsaHeadDim / 2; stride > 0; stride >>= 1) {
        if (d < stride) { reduction[d] += reduction[d + stride]; }
        __syncthreads();
    }
    const float inv = 1.0F / reduction[0];
    float value = 0.0F;
    for (int j = 0; j < count; ++j) {
        const int id = token_selected[j];
        if (id >= 0 && id < capacity) {
            value += scores[j] * inv * decode_value(v_codes, v_scales, d, id, kv_head, capacity);
        }
    }
    out[output_index] = __float2bfloat16_rn(value);
}

__global__ void qsa_attention_score_tiled_kernel(
    const __nv_bfloat16* q, const std::int32_t* selected, const std::int32_t* counts,
    const std::uint8_t* k_codes, const std::uint8_t* k_scales, float* scores,
    int selected_bound, int capacity) {
    __shared__ float keys[kAttentionScoreRanksPerStep][kQsaHeadDim];
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int head_group = static_cast<int>(blockIdx.y);
    const int head = head_group * kAttentionHeadsPerGroup + (warp & 3);
    const int local_rank = warp >> 2;
    const int kv_head = head_group / kAttentionGroupsPerKvHead;
    const int count = counts[0];
    if (count <= 0 || count > selected_bound) { return; }

    const int rank_base = static_cast<int>(blockIdx.x) * kAttentionScoreRanksPerBlock;
    for (int offset = 0; offset < kAttentionScoreRanksPerBlock;
         offset += kAttentionScoreRanksPerStep) {
#pragma unroll
        for (int rank_offset = 0; rank_offset < kAttentionScoreRanksPerStep; ++rank_offset) {
            const int rank = rank_base + offset + rank_offset;
            const int id = rank < count ? selected[rank] : -1;
            for (int d = static_cast<int>(threadIdx.x); d < kQsaHeadDim; d += blockDim.x) {
                keys[rank_offset][d] =
                    id >= 0 && id < capacity
                        ? decode_value(k_codes, k_scales, d, id, kv_head, capacity)
                        : 0.0F;
            }
        }
        __syncthreads();

        const int rank = rank_base + offset + local_rank;
        if (rank < count) {
            float dot = 0.0F;
            for (int d = lane; d < kQsaHeadDim; d += 32) {
                dot += __bfloat162float(q[d + kQsaHeadDim * head]) * keys[local_rank][d];
            }
            for (int delta = 16; delta > 0; delta >>= 1) {
                dot += __shfl_down_sync(0xffffffffU, dot, delta);
            }
            if (lane == 0) {
                scores[rank + static_cast<std::int64_t>(selected_bound) * head] = dot * 0.0625F;
            }
        }
        __syncthreads();
    }
}

__global__ void qsa_attention_softmax_kernel(float* scores, const std::int32_t* counts,
                                             int selected_bound) {
    __shared__ float reduction[kQsaHeadDim];
    const int d = static_cast<int>(threadIdx.x);
    const int head = static_cast<int>(blockIdx.x);
    const int count = counts[0];
    if (count <= 0 || count > selected_bound) { return; }
    float* head_scores = scores + static_cast<std::int64_t>(selected_bound) * head;

    float maximum = -FLT_MAX;
    for (int rank = d; rank < count; rank += kQsaHeadDim) {
        maximum = fmaxf(maximum, head_scores[rank]);
    }
    reduction[d] = maximum;
    __syncthreads();
    for (int stride = kQsaHeadDim / 2; stride > 0; stride >>= 1) {
        if (d < stride) { reduction[d] = fmaxf(reduction[d], reduction[d + stride]); }
        __syncthreads();
    }
    maximum = reduction[0];

    float denominator = 0.0F;
    for (int rank = d; rank < count; rank += kQsaHeadDim) {
        const float probability = expf(head_scores[rank] - maximum);
        head_scores[rank] = probability;
        denominator += probability;
    }
    reduction[d] = denominator;
    __syncthreads();
    for (int stride = kQsaHeadDim / 2; stride > 0; stride >>= 1) {
        if (d < stride) { reduction[d] += reduction[d + stride]; }
        __syncthreads();
    }
    const float inverse = 1.0F / reduction[0];
    for (int rank = d; rank < count; rank += kQsaHeadDim) { head_scores[rank] *= inverse; }
}

__global__ void qsa_attention_value_tiled_kernel(
    const std::int32_t* selected, const std::int32_t* counts, const std::uint8_t* v_codes,
    const std::uint8_t* v_scales, const float* probabilities, float* partials,
    int selected_bound, int capacity) {
    const int d = static_cast<int>(threadIdx.x);
    const int tile = static_cast<int>(blockIdx.x);
    const int head_group = static_cast<int>(blockIdx.y);
    const int head_base = head_group * kAttentionHeadsPerGroup;
    const int kv_head = head_group / kAttentionGroupsPerKvHead;
    const int count = counts[0];
    float values[kAttentionHeadsPerGroup]{};
    if (count > 0 && count <= selected_bound) {
        const int begin = tile * kAttentionValueTile;
        const int end = begin + kAttentionValueTile < count ? begin + kAttentionValueTile : count;
        for (int rank = begin; rank < end; ++rank) {
            const int id = selected[rank];
            const float value = id >= 0 && id < capacity
                                    ? decode_value(v_codes, v_scales, d, id, kv_head, capacity)
                                    : 0.0F;
#pragma unroll
            for (int local_head = 0; local_head < kAttentionHeadsPerGroup; ++local_head) {
                const int head = head_base + local_head;
                values[local_head] =
                    fmaf(probabilities[rank +
                                       static_cast<std::int64_t>(selected_bound) * head],
                         value, values[local_head]);
            }
        }
    }
#pragma unroll
    for (int local_head = 0; local_head < kAttentionHeadsPerGroup; ++local_head) {
        const int head = head_base + local_head;
        partials[d + static_cast<std::int64_t>(kQsaHeadDim) *
                         (head + kQsaQueryHeads * tile)] = values[local_head];
    }
}

__global__ void qsa_attention_finalize_kernel(const std::int32_t* counts,
                                              const float* partials,
                                              __nv_bfloat16* out, int selected_bound,
                                              int tile_count) {
    const int d = static_cast<int>(threadIdx.x);
    const int head = static_cast<int>(blockIdx.x);
    const int count = counts[0];
    if (count <= 0 || count > selected_bound) {
        out[d + kQsaHeadDim * head] = __float2bfloat16_rn(0.0F);
        return;
    }
    float value = 0.0F;
    for (int tile = 0; tile < tile_count; ++tile) {
        value += partials[d + static_cast<std::int64_t>(kQsaHeadDim) *
                                  (head + kQsaQueryHeads * tile)];
    }
    out[d + kQsaHeadDim * head] = __float2bfloat16_rn(value);
}

} // namespace

void qsa_state_append_launch(const Tensor& k, const Tensor& v, const Tensor& raw_index_keys,
                             const Tensor& position_ids, const Tensor& append_ids,
                             QsaStateView state, cudaStream_t stream) {
    const int width    = k.ne[2];
    const int capacity = state.raw_index_keys.ne[1];
    qsa_append_nvfp4_kernel<<<dim3(kGroups, width, kQsaKvHeads), 1, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(k.data), static_cast<const __nv_bfloat16*>(v.data),
        static_cast<const std::int32_t*>(append_ids.data),
        static_cast<std::uint8_t*>(state.k_codes.data),
        static_cast<std::uint8_t*>(state.v_codes.data),
        static_cast<std::uint8_t*>(state.k_scales.data),
        static_cast<std::uint8_t*>(state.v_scales.data), width, capacity);
    CUDA_CHECK(cudaGetLastError());
    qsa_append_index_kernel<<<width, 128, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(raw_index_keys.data),
        static_cast<const std::int32_t*>(position_ids.data),
        static_cast<const std::int32_t*>(append_ids.data),
        static_cast<__nv_bfloat16*>(state.raw_index_keys.data),
        static_cast<std::int32_t*>(state.positions.data), width, capacity);
    CUDA_CHECK(cudaGetLastError());
}

void qsa_index_select_launch(const Tensor& raw_query, const QsaStateView& state,
                             const Tensor& query_ids, const Tensor& visible_ids,
                             const Tensor& visible_offsets, const Tensor& query_norm_weight,
                             const Tensor& key_norm_weight, Tensor& selected_ids,
                             Tensor& selected_count, Tensor& workspace, cudaStream_t stream) {
    qsa_select_prepare_kernel<<<raw_query.ne[2], kQsaIndexHeadDim, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(raw_query.data),
        static_cast<const std::int32_t*>(state.positions.data),
        static_cast<const std::int32_t*>(query_ids.data),
        static_cast<const std::int32_t*>(visible_ids.data),
        static_cast<const std::int32_t*>(visible_offsets.data),
        static_cast<const float*>(query_norm_weight.data),
        static_cast<std::int32_t*>(selected_ids.data),
        static_cast<std::int32_t*>(selected_count.data), static_cast<float*>(workspace.data),
        raw_query.ne[2], visible_ids.ne[0], state.raw_index_keys.ne[1]);
    CUDA_CHECK(cudaGetLastError());
    int score_blocks = visible_ids.ne[0] / 4;
    if (score_blocks < 1) { score_blocks = 1; }
    if (score_blocks > kMaximumBlocks) { score_blocks = kMaximumBlocks; }
    qsa_select_score_kernel<<<dim3(score_blocks, raw_query.ne[2]), 1, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(state.raw_index_keys.data),
        static_cast<const std::int32_t*>(state.positions.data),
        static_cast<const std::int32_t*>(visible_ids.data),
        static_cast<const std::int32_t*>(visible_offsets.data),
        static_cast<const float*>(key_norm_weight.data),
        static_cast<const std::int32_t*>(selected_count.data), static_cast<float*>(workspace.data),
        raw_query.ne[2]);
    CUDA_CHECK(cudaGetLastError());
    if (visible_ids.ne[0] <= selected_ids.ne[0]) {
        int sort_blocks = 1;
        while (sort_blocks < score_blocks) { sort_blocks <<= 1; }
        int sort_threads = sort_blocks / 2;
        if (sort_threads < 32) { sort_threads = 32; }
        qsa_select_topk_short_kernel<<<raw_query.ne[2], sort_threads, 0, stream>>>(
            static_cast<const std::int32_t*>(visible_ids.data),
            static_cast<const std::int32_t*>(visible_offsets.data),
            static_cast<std::int32_t*>(selected_ids.data),
            static_cast<std::int32_t*>(selected_count.data),
            static_cast<float*>(workspace.data), raw_query.ne[2], sort_blocks);
    } else {
        qsa_select_topk_kernel<<<raw_query.ne[2], kKeep, 0, stream>>>(
            static_cast<const std::int32_t*>(visible_ids.data),
            static_cast<const std::int32_t*>(visible_offsets.data),
            static_cast<std::int32_t*>(selected_ids.data),
            static_cast<std::int32_t*>(selected_count.data),
            static_cast<float*>(workspace.data), raw_query.ne[2]);
    }
    CUDA_CHECK(cudaGetLastError());
}

void qsa_selected_attention_launch(const Tensor& q, const Tensor& selected_ids,
                                   const Tensor& selected_count, const QsaStateView& state,
                                   Tensor& out, Tensor& workspace, cudaStream_t stream) {
    const int selected_bound = selected_ids.ne[0];
    const int width = q.ne[2];
    if (width > 1 || selected_bound <= kAttentionValueTile) {
        qsa_attention_short_kernel<<<dim3(kQsaQueryHeads, width), kQsaHeadDim, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data),
            static_cast<const std::int32_t*>(selected_ids.data),
            static_cast<const std::int32_t*>(selected_count.data),
            static_cast<const std::uint8_t*>(state.k_codes.data),
            static_cast<const std::uint8_t*>(state.v_codes.data),
            static_cast<const std::uint8_t*>(state.k_scales.data),
            static_cast<const std::uint8_t*>(state.v_scales.data),
            static_cast<__nv_bfloat16*>(out.data), selected_bound,
            state.raw_index_keys.ne[1], width);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    auto* scores = static_cast<float*>(workspace.data);
    auto* partials = scores + kAttentionScoreFloats;
    const int score_blocks =
        (selected_bound + kAttentionScoreRanksPerBlock - 1) /
        kAttentionScoreRanksPerBlock;
    const int value_tiles =
        (selected_bound + kAttentionValueTile - 1) / kAttentionValueTile;
    qsa_attention_score_tiled_kernel<<<dim3(score_blocks, kAttentionHeadGroups), kQsaHeadDim,
                                       0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data),
        static_cast<const std::int32_t*>(selected_ids.data),
        static_cast<const std::int32_t*>(selected_count.data),
        static_cast<const std::uint8_t*>(state.k_codes.data),
        static_cast<const std::uint8_t*>(state.k_scales.data),
        scores, selected_bound, state.raw_index_keys.ne[1]);
    CUDA_CHECK(cudaGetLastError());
    qsa_attention_softmax_kernel<<<kQsaQueryHeads, kQsaHeadDim, 0, stream>>>(
        scores, static_cast<const std::int32_t*>(selected_count.data), selected_bound);
    CUDA_CHECK(cudaGetLastError());
    qsa_attention_value_tiled_kernel<<<dim3(value_tiles, kAttentionHeadGroups), kQsaHeadDim,
                                       0, stream>>>(
        static_cast<const std::int32_t*>(selected_ids.data),
        static_cast<const std::int32_t*>(selected_count.data),
        static_cast<const std::uint8_t*>(state.v_codes.data),
        static_cast<const std::uint8_t*>(state.v_scales.data), scores, partials,
        selected_bound, state.raw_index_keys.ne[1]);
    CUDA_CHECK(cudaGetLastError());
    qsa_attention_finalize_kernel<<<kQsaQueryHeads, kQsaHeadDim, 0, stream>>>(
        static_cast<const std::int32_t*>(selected_count.data), partials,
        static_cast<__nv_bfloat16*>(out.data), selected_bound, value_tiles);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
