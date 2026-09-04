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

__global__ void qsa_select_kernel(const __nv_bfloat16* raw_query,
                                  const __nv_bfloat16* raw_keys,
                                  const std::int32_t* state_positions,
                                  const std::int32_t* query_ids,
                                  const std::int32_t* visible_ids,
                                  const std::int32_t* visible_offsets,
                                  const float* query_weight, const float* key_weight,
                                  std::int32_t* selected,
                                  std::int32_t* selected_count, float* scratch, int width,
                                  int visible_extent, int capacity) {
    const int token = blockIdx.x;
    if (threadIdx.x != 0 || token >= width) { return; }
    std::int32_t* output = selected + static_cast<std::int64_t>(kQsaSelectedCapacity) * token;
    for (int i = 0; i < kQsaSelectedCapacity; ++i) { output[i] = -1; }
    selected_count[token] = 0;

    const int query_id = query_ids[token];
    const int begin    = visible_offsets[token];
    const int end      = visible_offsets[token + 1];
    if (query_id < 0) { return; }
    if (query_id >= capacity || begin < 0 || end <= begin || end > visible_extent ||
        end - begin > capacity) {
        return;
    }
    for (int i = begin; i < end; ++i) {
        if (visible_ids[i] < 0 || visible_ids[i] >= capacity ||
            (i > begin && visible_ids[i] <= visible_ids[i - 1])) {
            return;
        }
    }

    float* q_rot       = scratch + static_cast<std::int64_t>(token) * 3 * kKeep;
    float* top_scores  = q_rot + kKeep;
    auto* top_blocks   = reinterpret_cast<std::int32_t*>(top_scores + kKeep);
    const std::int32_t* query_position = state_positions + 3LL * query_id;
    for (int head = 0; head < kQsaIndexQueryHeads; ++head) {
        const __nv_bfloat16* q = raw_query + static_cast<std::int64_t>(kQsaIndexHeadDim) *
                                                 (head + kQsaIndexQueryHeads * token);
        float sum2 = 0.0F;
        for (int d = 0; d < kQsaIndexHeadDim; ++d) {
            const float value = __bfloat162float(q[d]);
            sum2 += value * value;
        }
        const float inv_rms = rsqrtf(sum2 / kQsaIndexHeadDim + kEpsilon);
        for (int d = 0; d < kQsaIndexHeadDim; ++d) {
            float value = normalized_query(q, query_weight, d, inv_rms);
            if (d < 64) {
                const int pair = d & 31;
                const float lo = normalized_query(q, query_weight, pair, inv_rms);
                const float hi = normalized_query(q, query_weight, pair + 32, inv_rms);
                value = rope_component(lo, hi, pair, d >= 32, query_position);
            }
            q_rot[d + kQsaIndexHeadDim * head] = value;
        }
    }

    const int complete = (end - begin) / 4;
    int kept           = 0;
    for (int block = 0; block < complete; ++block) {
        float key_sum2 = 0.0F;
        for (int d = 0; d < kQsaIndexHeadDim; ++d) {
            float pooled = 0.0F;
            for (int r = 0; r < 4; ++r) {
                const int id = visible_ids[begin + block * 4 + r];
                pooled += __bfloat162float(raw_keys[d + static_cast<std::int64_t>(kQsaIndexHeadDim) * id]);
            }
            const float represented = __bfloat162float(__float2bfloat16_rn(pooled * 0.25F));
            key_sum2 += represented * represented;
        }
        const float key_inv = rsqrtf(key_sum2 / kQsaIndexHeadDim + kEpsilon);
        const int block_id  = visible_ids[begin + block * 4];
        const std::int32_t* block_position = state_positions + 3LL * block_id;
        float score = 0.0F;
        for (int head = 0; head < kQsaIndexQueryHeads; ++head) {
            float dot = 0.0F;
            for (int d = 0; d < kQsaIndexHeadDim; ++d) {
                auto pooled_at = [&](int kd) {
                    float pooled = 0.0F;
                    for (int r = 0; r < 4; ++r) {
                        const int id = visible_ids[begin + block * 4 + r];
                        pooled += __bfloat162float(raw_keys[kd + static_cast<std::int64_t>(kQsaIndexHeadDim) * id]);
                    }
                    const float represented =
                        __bfloat162float(__float2bfloat16_rn(pooled * 0.25F));
                    return represented * key_inv * key_weight[kd];
                };
                float key_value = pooled_at(d);
                if (d < 64) {
                    const int pair = d & 31;
                    key_value = rope_component(pooled_at(pair), pooled_at(pair + 32), pair,
                                               d >= 32, block_position);
                }
                dot += q_rot[d + kQsaIndexHeadDim * head] * key_value;
            }
            score += fmaxf(dot, 0.0F);
        }
        score *= 0.08838834764831845F; // 1/sqrt(128)

        int insertion = kept;
        while (insertion > 0 && score > top_scores[insertion - 1]) { --insertion; }
        if (insertion < kKeep) {
            const int limit = kept < kKeep ? kept : kKeep - 1;
            for (int i = limit; i > insertion; --i) {
                top_scores[i] = top_scores[i - 1];
                top_blocks[i] = top_blocks[i - 1];
            }
            top_scores[insertion] = score;
            top_blocks[insertion] = block;
            if (kept < kKeep) { ++kept; }
        }
    }

    int count = 0;
    for (int rank = 0; rank < kept; ++rank) {
        const int source = begin + 4 * top_blocks[rank];
        for (int r = 0; r < 4; ++r) { output[count++] = visible_ids[source + r]; }
    }
    for (int i = begin + 4 * complete; i < end; ++i) { output[count++] = visible_ids[i]; }
    selected_count[token] = count;
}

__global__ void qsa_attention_kernel(const __nv_bfloat16* q, const std::int32_t* selected,
                                     const std::int32_t* counts, const std::uint8_t* k_codes,
                                     const std::uint8_t* v_codes,
                                     const std::uint8_t* k_scales,
                                     const std::uint8_t* v_scales, __nv_bfloat16* out,
                                     int width, int capacity) {
    __shared__ float scores[kQsaSelectedCapacity];
    __shared__ float reduction[kQsaHeadDim];
    const int d     = threadIdx.x;
    const int head  = blockIdx.x;
    const int token = blockIdx.y;
    const int count = counts[token];
    const std::int64_t output_index = d + static_cast<std::int64_t>(kQsaHeadDim) *
                                             (head + kQsaQueryHeads * token);
    if (count <= 0 || count > kQsaSelectedCapacity) {
        out[output_index] = __float2bfloat16_rn(0.0F);
        return;
    }
    const int kv_head = head / 12;
    const float query = __bfloat162float(q[output_index]);
    for (int j = 0; j < count; ++j) {
        const int id = selected[j + static_cast<std::int64_t>(kQsaSelectedCapacity) * token];
        reduction[d] = id >= 0 && id < capacity
                           ? query * decode_value(k_codes, k_scales, d, id, kv_head, capacity)
                           : 0.0F;
        __syncthreads();
        for (int stride = kQsaHeadDim / 2; stride > 0; stride >>= 1) {
            if (d < stride) { reduction[d] += reduction[d + stride]; }
            __syncthreads();
        }
        if (d == 0) { scores[j] = reduction[0] * 0.0625F; }
        __syncthreads();
    }
    if (d == 0) {
        float maximum = -FLT_MAX;
        for (int j = 0; j < count; ++j) { maximum = fmaxf(maximum, scores[j]); }
        float sum = 0.0F;
        for (int j = 0; j < count; ++j) {
            scores[j] = expf(scores[j] - maximum);
            sum += scores[j];
        }
        const float inv = 1.0F / sum;
        for (int j = 0; j < count; ++j) { scores[j] *= inv; }
    }
    __syncthreads();
    float value = 0.0F;
    for (int j = 0; j < count; ++j) {
        const int id = selected[j + static_cast<std::int64_t>(kQsaSelectedCapacity) * token];
        if (id >= 0 && id < capacity) {
            value += scores[j] * decode_value(v_codes, v_scales, d, id, kv_head, capacity);
        }
    }
    out[output_index] = __float2bfloat16_rn(value);
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
    qsa_select_kernel<<<raw_query.ne[2], 1, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(raw_query.data),
        static_cast<const __nv_bfloat16*>(state.raw_index_keys.data),
        static_cast<const std::int32_t*>(state.positions.data),
        static_cast<const std::int32_t*>(query_ids.data),
        static_cast<const std::int32_t*>(visible_ids.data),
        static_cast<const std::int32_t*>(visible_offsets.data),
        static_cast<const float*>(query_norm_weight.data),
        static_cast<const float*>(key_norm_weight.data),
        static_cast<std::int32_t*>(selected_ids.data),
        static_cast<std::int32_t*>(selected_count.data), static_cast<float*>(workspace.data),
        raw_query.ne[2], visible_ids.ne[0], state.raw_index_keys.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

void qsa_selected_attention_launch(const Tensor& q, const Tensor& selected_ids,
                                   const Tensor& selected_count, const QsaStateView& state,
                                   Tensor& out, cudaStream_t stream) {
    qsa_attention_kernel<<<dim3(kQsaQueryHeads, q.ne[2]), kQsaHeadDim, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data),
        static_cast<const std::int32_t*>(selected_ids.data),
        static_cast<const std::int32_t*>(selected_count.data),
        static_cast<const std::uint8_t*>(state.k_codes.data),
        static_cast<const std::uint8_t*>(state.v_codes.data),
        static_cast<const std::uint8_t*>(state.k_scales.data),
        static_cast<const std::uint8_t*>(state.v_scales.data),
        static_cast<__nv_bfloat16*>(out.data), q.ne[2], state.raw_index_keys.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
