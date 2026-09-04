#pragma once

// Shared implementation primitives for include/ninfer/ops/sampling.h and
// include/ninfer/ops/speculative_round.h. The ordering key is exact for finite BF16
// logits (including numeric-zero ties); candidate storage is bounded by the
// semantic top-20 cap and all global staging is supplied by the caller.

#include "ops/common/math.h"
#include "ops/common/sampling_workspace.h"
#include "ninfer/ops/sampling.h"

#include <cub/block/block_merge_sort.cuh>

#include <cuda/atomic>
#include <cuda_bf16.h>
#include <climits>
#include <cstdint>
#include <math_constants.h>

namespace ninfer::ops {

using SamplingPartialSort =
    cub::BlockMergeSort<unsigned long long, kSamplerBlock, kSamplerItemsPerThread>;
using SamplingGroupSort =
    cub::BlockMergeSort<unsigned long long, kSamplerGroupBlock, kSamplerGroupItemsPerThread>;

struct SamplingKeyGreater {
    __device__ __forceinline__ bool operator()(unsigned long long a, unsigned long long b) const {
        return a > b;
    }
};

__device__ inline unsigned long long sampling_block_max_key(unsigned long long key,
                                                            unsigned long long* warp_keys) {
    constexpr unsigned int kMask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        const unsigned long long other = __shfl_down_sync(kMask, key, offset);
        if (other > key) { key = other; }
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) { warp_keys[warp] = key; }
    __syncthreads();
    key = (warp == 0 && lane < (blockDim.x >> 5)) ? warp_keys[lane] : 0ull;
    if (warp == 0) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            const unsigned long long other = __shfl_down_sync(kMask, key, offset);
            if (other > key) { key = other; }
        }
    }
    return key;
}

__device__ __forceinline__ unsigned long long sampling_splitmix64(unsigned long long x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// Uniform float in [0,1) from (seed, position, purpose, sub). Pure function of
// its inputs so it is safe under CUDA-graph replay (no mutable RNG state).
__device__ __forceinline__ float sampling_uniform(unsigned long long seed, int position,
                                                  int purpose, unsigned int sub) {
    unsigned long long key = seed;
    key                    = sampling_splitmix64(
        key ^ (static_cast<unsigned long long>(static_cast<unsigned int>(position)) *
               0xD1B54A32D192ED03ull));
    key = sampling_splitmix64(
        key ^ (static_cast<unsigned long long>(static_cast<unsigned int>(purpose)) << 21) ^
        (static_cast<unsigned long long>(sub) * 0x2545F4914F6CDD1Dull));
    const unsigned int bits = static_cast<unsigned int>(key >> 40); // 24 bits
    return static_cast<float>(bits) * (1.0f / 16777216.0f);
}

// Candidate ordering: higher value wins, ties broken by lower vocab index.
__device__ __forceinline__ bool sampling_better(float v, int i, float bv, int bi) {
    return v > bv || (v == bv && i < bi);
}

// True when (v,i) ranks strictly below pivot (pv,pi) in the ordering above.
__device__ __forceinline__ bool sampling_worse_than(float v, int i, float pv, int pi) {
    return pv > v || (pv == v && pi < i);
}

__device__ __forceinline__ unsigned int sampling_ordered_float(float v) {
    // Numeric equality, including +0 == -0, must reach the token-id tie break.
    // Canonicalizing zero prevents the IEEE sign bit from ranking +0 above -0.
    if (v == 0.0f) { v = 0.0f; }
    const unsigned int bits = __float_as_uint(v);
    return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

__device__ __forceinline__ unsigned long long sampling_sort_key(float v, int idx) {
    if (idx == INT_MAX) { return 0ull; }
    return (static_cast<unsigned long long>(sampling_ordered_float(v)) << 32) |
           static_cast<unsigned int>(0xffffffffu - static_cast<unsigned int>(idx));
}

__device__ __forceinline__ float sampling_key_float(unsigned long long key) {
    const unsigned int ordered = static_cast<unsigned int>(key >> 32);
    const unsigned int bits    = (ordered & 0x80000000u) ? (ordered ^ 0x80000000u) : ~ordered;
    return __uint_as_float(bits);
}

__device__ __forceinline__ int sampling_key_index(unsigned long long key) {
    if (key == 0ull) { return INT_MAX; }
    return static_cast<int>(0xffffffffu - static_cast<unsigned int>(key));
}

__device__ __forceinline__ bool sampling_p_less_active(const SamplingConfig& cfg) {
    return cfg.p_less != 0 && cfg.temperature > 0.0f;
}

__device__ __forceinline__ bool sampling_token_suppressed(int v, const SamplingConfig& c) {
    const int count = min(c.suppressed_token_count, SamplingConfig::kMaximumSuppressedTokens);
    for (int i = 0; i < count; ++i) {
        if (c.suppressed_tokens[i] == v) { return true; }
    }
    return false;
}

__device__ __forceinline__ bool sampling_p_less_in_domain(int v, std::int32_t vocab,
                                                         const SamplingConfig& cfg) {
    return v >= 0 && v < vocab && !sampling_token_suppressed(v, cfg);
}

// A missed inverse-CDF or a dirty workspace can yield INT_MAX / a padding row.
// Counting that id would write past token_counts and poison later rounds.
__device__ __forceinline__ int sampling_clamp_token(int token, int fallback, std::int32_t vocab) {
    if (token >= 0 && token < vocab) { return token; }
    if (fallback >= 0 && fallback < vocab) { return fallback; }
    return 0;
}

__device__ __forceinline__ void sampling_count_token(const SamplingConfig& cfg, int token,
                                                     std::int32_t vocab) {
    if (cfg.token_counts == nullptr || token < 0 || token >= vocab) { return; }
    atomicAdd(&cfg.token_counts[token], 1);
}

__device__ inline float sampling_block_sum(float value, float* shared) {
    const int tid = threadIdx.x;
    shared[tid]   = value;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) { shared[tid] += shared[tid + s]; }
        __syncthreads();
    }
    return shared[0];
}

__device__ inline float sampling_block_sum_fast(float value, float* warp_sums) {
    constexpr unsigned int kMask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(kMask, value, offset);
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) { warp_sums[warp] = value; }
    __syncthreads();
    if (warp == 0) {
        value = lane < (blockDim.x >> 5) ? warp_sums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(kMask, value, offset);
        }
        if (lane == 0) { warp_sums[0] = value; }
    }
    __syncthreads();
    return warp_sums[0];
}

struct SamplingFloatPair {
    float first  = 0.0f;
    float second = 0.0f;
};

__device__ inline SamplingFloatPair sampling_block_sum_pair(float first, float second,
                                                            float* first_warp_sums,
                                                            float* second_warp_sums) {
    constexpr unsigned int kMask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        first += __shfl_down_sync(kMask, first, offset);
        second += __shfl_down_sync(kMask, second, offset);
    }
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        first_warp_sums[warp]  = first;
        second_warp_sums[warp] = second;
    }
    __syncthreads();
    if (warp == 0) {
        first  = lane < (blockDim.x >> 5) ? first_warp_sums[lane] : 0.0f;
        second = lane < (blockDim.x >> 5) ? second_warp_sums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            first += __shfl_down_sync(kMask, first, offset);
            second += __shfl_down_sync(kMask, second, offset);
        }
        if (lane == 0) {
            first_warp_sums[0]  = first;
            second_warp_sums[0] = second;
        }
    }
    __syncthreads();
    return {first_warp_sums[0], second_warp_sums[0]};
}

__device__ inline unsigned long long
sampling_block_max_key_broadcast(unsigned long long key, unsigned long long* warp_keys) {
    key = sampling_block_max_key(key, warp_keys);
    if (threadIdx.x == 0) { warp_keys[0] = key; }
    __syncthreads();
    return warp_keys[0];
}

__device__ inline void sampling_block_inclusive_scan(float* shared) {
    const int tid = threadIdx.x;
    for (int offset = 1; offset < blockDim.x; offset <<= 1) {
        float add = 0.0f;
        if (tid >= offset) { add = shared[tid - offset]; }
        __syncthreads();
        shared[tid] += add;
        __syncthreads();
    }
}

// Full-vocabulary p-less moments from represented BF16 logits. All threads of
// the block must call. Penalties are not applied. red_idx[0] remains the min
// argmax; red_val[0] is sum_exp and red_aux[0] is sum_exp2.
struct SamplingPLessMoments {
    float m        = 0.0f;
    float sum_exp  = 0.0f;
    float sum_exp2 = 0.0f;
    int argmax     = 0;
};

// Membership is e*sum_exp >= sum_exp2 (exact p_v>=L without dividing into a
// threshold). logit_cut is a conservative prefilter: z < logit_cut cannot pass.
struct SamplingPLessGate {
    float m         = 0.0f;
    float sum_exp   = 0.0f;
    float sum_exp2  = 0.0f;
    float inv_temp  = 0.0f;
    float logit_cut = 1.0e30f;
};

__device__ __forceinline__ SamplingPLessGate sampling_p_less_gate(const SamplingPLessMoments& st,
                                                                  float inv_temp) {
    SamplingPLessGate g;
    g.m        = st.m;
    g.sum_exp  = st.sum_exp;
    g.sum_exp2 = st.sum_exp2;
    g.inv_temp = inv_temp;
    g.logit_cut = 1.0e30f;
    if (st.sum_exp > 0.0f && st.sum_exp2 > 0.0f) {
        const float x_cut = logf(st.sum_exp2 / st.sum_exp) - 1.0e-3f;
        g.logit_cut       = st.m + x_cut / inv_temp;
    }
    return g;
}

__device__ __forceinline__ bool sampling_p_less_weight_admitted(float e,
                                                               const SamplingPLessGate& g) {
    return __fmaf_rn(e, g.sum_exp, -g.sum_exp2) >= 0.0f;
}

__device__ __forceinline__ float sampling_p_less_survivor_exp(float z, const SamplingPLessGate& g) {
    if (!(z >= g.logit_cut)) { return 0.0f; }
    const float e = __expf((z - g.m) * g.inv_temp);
    return sampling_p_less_weight_admitted(e, g) ? e : 0.0f;
}

__device__ inline SamplingPLessMoments sampling_p_less_moments(const __nv_bfloat16* logits,
                                                               std::int64_t base,
                                                               std::int32_t vocab,
                                                               const SamplingConfig& cfg,
                                                               float inv_temp, float* red_val,
                                                               int* red_idx, float* red_aux) {
    const int tid = threadIdx.x;
    float bv      = -CUDART_INF_F;
    int bi        = INT_MAX;
    for (int v = tid; v < vocab; v += blockDim.x) {
        if (!sampling_p_less_in_domain(v, vocab, cfg)) { continue; }
        const float x = __bfloat162float(logits[base + v]);
        if (sampling_better(x, v, bv, bi)) {
            bv = x;
            bi = v;
        }
    }
    red_val[tid] = bv;
    red_idx[tid] = bi;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s &&
            sampling_better(red_val[tid + s], red_idx[tid + s], red_val[tid], red_idx[tid])) {
            red_val[tid] = red_val[tid + s];
            red_idx[tid] = red_idx[tid + s];
        }
        __syncthreads();
    }
    SamplingPLessMoments out;
    out.m      = red_val[0];
    out.argmax = red_idx[0];
    float local_s = 0.0f;
    float local_q = 0.0f;
    for (int v = tid; v < vocab; v += blockDim.x) {
        if (!sampling_p_less_in_domain(v, vocab, cfg)) { continue; }
        const float e = __expf((__bfloat162float(logits[base + v]) - out.m) * inv_temp);
        local_s += e;
        local_q += e * e;
    }
    red_val[tid] = local_s;
    red_aux[tid] = local_q;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            red_val[tid] += red_val[tid + s];
            red_aux[tid] += red_aux[tid + s];
        }
        __syncthreads();
    }
    out.sum_exp  = red_val[0];
    out.sum_exp2 = red_aux[0];
    return out;
}

__device__ inline float sampling_p_less_admitted_mass(const __nv_bfloat16* logits,
                                                      std::int64_t base, std::int32_t vocab,
                                                      const SamplingConfig& cfg,
                                                      const SamplingPLessGate& gate,
                                                      float* red_val) {
    const int tid = threadIdx.x;
    float local_z = 0.0f;
    for (int v = tid; v < vocab; v += blockDim.x) {
        if (!sampling_p_less_in_domain(v, vocab, cfg)) { continue; }
        local_z += sampling_p_less_survivor_exp(__bfloat162float(logits[base + v]), gate);
    }
    return sampling_block_sum(local_z, red_val);
}

// Inverse-CDF over p-less support. All threads of the block must call. `u` is
// the same counter-based uniform on every thread. argmax is the fallback.
__device__ inline int sampling_p_less_inverse_cdf(const __nv_bfloat16* logits, std::int64_t base,
                                                  std::int32_t vocab, const SamplingConfig& cfg,
                                                  const SamplingPLessGate& gate, float u,
                                                  int argmax, float* red_val, int* red_idx) {
    const int tid = threadIdx.x;
    float local_z = 0.0f;
    for (int v = tid; v < vocab; v += blockDim.x) {
        if (!sampling_p_less_in_domain(v, vocab, cfg)) { continue; }
        local_z += sampling_p_less_survivor_exp(__bfloat162float(logits[base + v]), gate);
    }
    red_val[tid] = local_z;
    __syncthreads();
    sampling_block_inclusive_scan(red_val);
    const float inclusive = red_val[tid];
    const float exclusive = inclusive - local_z;
    const float total     = red_val[blockDim.x - 1];
    if (!(total > 0.0f)) { return argmax; }
    const float goal      = u * total;
    int local_pick        = -1;
    if (local_z > 0.0f && goal >= exclusive && (goal < inclusive || inclusive == total)) {
        float acc = exclusive;
        for (int v = tid; v < vocab; v += blockDim.x) {
            if (!sampling_p_less_in_domain(v, vocab, cfg)) { continue; }
            const float e =
                sampling_p_less_survivor_exp(__bfloat162float(logits[base + v]), gate);
            if (!(e > 0.0f)) { continue; }
            acc += e;
            local_pick = v;
            if (goal < acc) { break; }
        }
    }
    red_idx[tid] = local_pick;
    __syncthreads();
    int picked = argmax;
    for (int t = 0; t < blockDim.x; ++t) {
        if (red_idx[t] >= 0) {
            picked = red_idx[t];
            break;
        }
    }
    if (tid == 0) { red_idx[0] = picked; }
    __syncthreads();
    return red_idx[0];
}

// One p-less token from a logit column. All threads of the block must call.
__device__ inline int sampling_p_less_draw(const __nv_bfloat16* logits, std::int64_t base,
                                           std::int32_t vocab, const SamplingConfig& cfg,
                                           int position, int purpose, float* red_val, int* red_idx,
                                           float* red_aux) {
    const float inv_temp          = 1.0f / cfg.temperature;
    const SamplingPLessMoments st =
        sampling_p_less_moments(logits, base, vocab, cfg, inv_temp, red_val, red_idx, red_aux);
    if (!(st.sum_exp > 0.0f)) { return st.argmax; }
    const SamplingPLessGate gate = sampling_p_less_gate(st, inv_temp);
    const float u = sampling_uniform(cfg.seed, position, purpose, 0u);
    return sampling_p_less_inverse_cdf(logits, base, vocab, cfg, gate, u, st.argmax, red_val,
                                       red_idx);
}

__device__ __forceinline__ int sampling_candidate_cap(const SamplingConfig& cfg,
                                                      std::int32_t vocab) {
    // top_k is clamped to the pipeline cap: top_k <= 0 (no explicit limit) or a
    // top_k larger than the cap both select the full kSamplerCandidateCap set.
    int cap = kSamplerCandidateCap;
    if (cfg.top_k > 0 && cfg.top_k < cap) { cap = cfg.top_k; }
    if (vocab < cap) { cap = vocab; }
    return cap;
}

__device__ __forceinline__ int sampling_partial_offset(const SamplingWorkspace& workspace, int col,
                                                       int partial, int j) {
    return ((col * workspace.partial_stride + partial) * kSamplerCandidateCap) + j;
}

__device__ __forceinline__ int sampling_dist_offset(int col, int j) {
    return col * kSamplerCandidateCap + j;
}

// Device-scope release RMWs form one release sequence; the last CTA's acquire
// load makes every participant's preceding stores visible before reduction.
__device__ __forceinline__ int sampling_completion_add(std::int32_t* ptr, int expected) {
    cuda::atomic_ref<std::int32_t, cuda::thread_scope_device> counter(*ptr);
    const int done = counter.fetch_add(1, cuda::memory_order_release) + 1;
    if (done == expected) { (void)counter.load(cuda::memory_order_acquire); }
    return done;
}

__device__ __forceinline__ void
sampling_publish_key(const SamplingWorkspace& workspace, int offset, unsigned long long value) {
    workspace.partial_keys[offset] = value;
}

__device__ __forceinline__ unsigned long long
sampling_load_published_key(const SamplingWorkspace& workspace, int offset) {
    return workspace.partial_keys[offset];
}

__device__ __forceinline__ void sampling_publish_float(float* ptr, int offset, float value) {
    ptr[offset] = value;
}

__device__ __forceinline__ float sampling_load_published_float(const float* ptr, int offset) {
    return ptr[offset];
}

__device__ __forceinline__ void
sampling_publish_i32(std::int32_t* ptr, int offset, std::int32_t value) {
    ptr[offset] = value;
}

__device__ __forceinline__ std::int32_t
sampling_load_published_i32(const std::int32_t* ptr, int offset) {
    return ptr[offset];
}

inline constexpr int kSamplingPLessMaxSlot      = 0;
inline constexpr int kSamplingPLessSumSlot      = 1;
inline constexpr int kSamplingPLessSumSqSlot    = 2;
inline constexpr int kSamplingPLessThreshSlot   = 3;
inline constexpr int kSamplingPLessAdmittedSlot = 4;
static_assert(kSamplerCandidateCap > kSamplingPLessAdmittedSlot);

__device__ __forceinline__ unsigned long long sampling_pack_float_pair(float first, float second) {
    return static_cast<unsigned long long>(__float_as_uint(first)) |
           (static_cast<unsigned long long>(__float_as_uint(second)) << 32);
}

__device__ __forceinline__ SamplingFloatPair
sampling_unpack_float_pair(unsigned long long packed) {
    return {__uint_as_float(static_cast<unsigned int>(packed)),
            __uint_as_float(static_cast<unsigned int>(packed >> 32))};
}

__device__ __forceinline__ void sampling_p_less_store_moments(
    const SamplingWorkspace& workspace, int col, int partial, const SamplingPLessMoments& moments) {
    sampling_publish_key(workspace, sampling_partial_offset(workspace, col, partial, 0),
                         sampling_sort_key(moments.m, moments.argmax));
    sampling_publish_key(workspace, sampling_partial_offset(workspace, col, partial, 1),
                         sampling_pack_float_pair(moments.sum_exp, moments.sum_exp2));
}

__device__ inline SamplingPLessMoments sampling_p_less_tile_moments(
    const __nv_bfloat16* logits, std::int64_t base, std::int32_t vocab, const SamplingConfig& cfg,
    int tile_start, float inv_temp, float* red_val, float* red_aux, unsigned long long* warp_keys) {
    float values[kSamplerItemsPerThread];
    unsigned long long best = 0ull;
#pragma unroll
    for (int item = 0; item < kSamplerItemsPerThread; ++item) {
        const int v = tile_start + item * blockDim.x + threadIdx.x;
        const bool keep = sampling_p_less_in_domain(v, vocab, cfg);
        values[item]    = keep ? __bfloat162float(logits[base + v]) : -CUDART_INF_F;
        const unsigned long long key = keep ? sampling_sort_key(values[item], v) : 0ull;
        if (key > best) { best = key; }
    }
    best          = sampling_block_max_key_broadcast(best, warp_keys);
    const float m = sampling_key_float(best);

    float local_s = 0.0f;
    float local_q = 0.0f;
#pragma unroll
    for (int item = 0; item < kSamplerItemsPerThread; ++item) {
        const int v = tile_start + item * blockDim.x + threadIdx.x;
        if (!sampling_p_less_in_domain(v, vocab, cfg)) { continue; }
        const float e = __expf((values[item] - m) * inv_temp);
        local_s += e;
        local_q += e * e;
    }
    const SamplingFloatPair sums =
        sampling_block_sum_pair(local_s, local_q, red_val, red_aux);
    return {m, sums.first, sums.second, sampling_key_index(best)};
}

__device__ inline SamplingPLessMoments sampling_p_less_merge_moments(
    const SamplingWorkspace& workspace, int col, int partial_begin, int partial_count,
    float inv_temp, float* red_val, float* red_aux, unsigned long long* warp_keys) {
    unsigned long long key = 0ull;
    SamplingFloatPair sums{};
    if (threadIdx.x < partial_count) {
        const int partial = partial_begin + threadIdx.x;
        key = sampling_load_published_key(
            workspace, sampling_partial_offset(workspace, col, partial, 0));
        sums = sampling_unpack_float_pair(sampling_load_published_key(
            workspace, sampling_partial_offset(workspace, col, partial, 1)));
    }
    const unsigned long long best = sampling_block_max_key_broadcast(key, warp_keys);
    const float m                 = sampling_key_float(best);
    float local_s                 = 0.0f;
    float local_q                 = 0.0f;
    if (threadIdx.x < partial_count) {
        const float scale = __expf((sampling_key_float(key) - m) * inv_temp);
        local_s           = sums.first * scale;
        local_q           = sums.second * scale * scale;
    }
    const SamplingFloatPair merged =
        sampling_block_sum_pair(local_s, local_q, red_val, red_aux);
    return {m, merged.first, merged.second, sampling_key_index(best)};
}

__device__ __forceinline__ void
sampling_p_less_store_global(const SamplingWorkspace& workspace, int col,
                             const SamplingPLessMoments& moments) {
    workspace.dist_prob[sampling_dist_offset(col, kSamplingPLessMaxSlot)] = moments.m;
    workspace.dist_prob[sampling_dist_offset(col, kSamplingPLessSumSlot)] = moments.sum_exp;
    workspace.dist_prob[sampling_dist_offset(col, kSamplingPLessSumSqSlot)] = moments.sum_exp2;
    workspace.dist_prob[sampling_dist_offset(col, kSamplingPLessThreshSlot)] =
        moments.sum_exp > 0.0f ? moments.sum_exp2 / moments.sum_exp : 0.0f;
    workspace.dist_idx[sampling_dist_offset(col, kSamplingPLessMaxSlot)] = moments.argmax;
}

__device__ __forceinline__ SamplingPLessMoments
sampling_p_less_load_global(const SamplingWorkspace& workspace, int col) {
    return {
        workspace.dist_prob[sampling_dist_offset(col, kSamplingPLessMaxSlot)],
        workspace.dist_prob[sampling_dist_offset(col, kSamplingPLessSumSlot)],
        workspace.dist_prob[sampling_dist_offset(col, kSamplingPLessSumSqSlot)],
        workspace.dist_idx[sampling_dist_offset(col, kSamplingPLessMaxSlot)],
    };
}

__device__ __forceinline__ float
sampling_p_less_load_threshold(const SamplingWorkspace& workspace, int col) {
    return workspace.dist_prob[sampling_dist_offset(col, kSamplingPLessThreshSlot)];
}

__device__ __forceinline__ float
sampling_p_less_load_tile_max(const SamplingWorkspace& workspace, int col, int partial) {
    const unsigned long long key = sampling_load_published_key(
        workspace, sampling_partial_offset(workspace, col, partial, 0));
    if (key == 0ull) { return -CUDART_INF_F; }
    return sampling_key_float(key);
}

__device__ __forceinline__ void
sampling_p_less_store_tile_mass(const SamplingWorkspace& workspace, int col, int partial,
                                float mass) {
    sampling_publish_key(workspace, sampling_partial_offset(workspace, col, partial, 0),
                         static_cast<unsigned long long>(__float_as_uint(mass)));
}

__device__ __forceinline__ float
sampling_p_less_load_tile_mass(const SamplingWorkspace& workspace, int col, int partial) {
    return __uint_as_float(static_cast<unsigned int>(sampling_load_published_key(
        workspace, sampling_partial_offset(workspace, col, partial, 0))));
}

__device__ __forceinline__ void
sampling_p_less_store_admitted(const SamplingWorkspace& workspace, int col, float admitted) {
    sampling_publish_float(workspace.dist_prob,
                           sampling_dist_offset(col, kSamplingPLessAdmittedSlot), admitted);
}

__device__ __forceinline__ float
sampling_p_less_load_admitted(const SamplingWorkspace& workspace, int col) {
    return sampling_load_published_float(
        workspace.dist_prob, sampling_dist_offset(col, kSamplingPLessAdmittedSlot));
}

__device__ inline float sampling_p_less_tile_admitted_mass(
    const __nv_bfloat16* logits, std::int64_t base, std::int32_t vocab, const SamplingConfig& cfg,
    int tile_start, const SamplingPLessGate& gate, float tile_max, float* warp_sums) {
    float local_z = 0.0f;
    if (tile_max >= gate.logit_cut) {
        const float e_peak = __expf((tile_max - gate.m) * gate.inv_temp);
        if (sampling_p_less_weight_admitted(e_peak, gate)) {
#pragma unroll
            for (int item = 0; item < kSamplerItemsPerThread; ++item) {
                const int v = tile_start + item * blockDim.x + threadIdx.x;
                if (!sampling_p_less_in_domain(v, vocab, cfg)) { continue; }
                local_z +=
                    sampling_p_less_survivor_exp(__bfloat162float(logits[base + v]), gate);
            }
        }
    }
    return sampling_block_sum_fast(local_z, warp_sums);
}

__device__ __forceinline__ float sampling_p_less_q_at(int v, int draft_id, const int* q_ids,
                                                      const float* q_vals, int q_n);

__device__ inline int sampling_p_less_pick_from_tile(
    const __nv_bfloat16* logits, std::int64_t base, std::int32_t vocab, const SamplingConfig& cfg,
    int tile, const SamplingPLessGate& gate, float admitted, float goal, int fallback,
    bool residual, int draft_id, const int* q_ids, const float* q_vals, int q_n, float* weights,
    float* running, int* picked, int* found) {
    if (threadIdx.x == 0) {
        *running = 0.0f;
        *picked  = fallback;
        *found   = 0;
    }
    __syncthreads();

    const int tile_start = tile * kSamplerPartialTileItems;
#pragma unroll
    for (int item = 0; item < kSamplerItemsPerThread; ++item) {
        const int v = tile_start + item * blockDim.x + threadIdx.x;
        float weight = 0.0f;
        if (sampling_p_less_in_domain(v, vocab, cfg)) {
            const float e =
                sampling_p_less_survivor_exp(__bfloat162float(logits[base + v]), gate);
            if (e > 0.0f) {
                weight = residual
                             ? fmaxf(0.0f, e / admitted -
                                              sampling_p_less_q_at(v, draft_id, q_ids, q_vals, q_n))
                             : e;
            }
        }
        weights[threadIdx.x] = weight;
        __syncthreads();
        if (threadIdx.x == 0 && *found == 0) {
            float acc = *running;
            for (int lane = 0; lane < blockDim.x; ++lane) {
                const float w = weights[lane];
                if (!(w > 0.0f)) { continue; }
                acc += w;
                *picked = tile_start + item * blockDim.x + lane;
                if (goal < acc) {
                    *found  = 1;
                    break;
                }
            }
            *running = acc;
        }
        __syncthreads();
        if (*found != 0) { break; }
    }
    return *picked;
}

// Applies presence/frequency penalties to an eligible raw logit. `overlay`/`overlay_len` carry a
// round-local count overlay: tokens already committed earlier in the
// current speculative round but not yet flushed to the global `token_counts`. For speculative
// verify column `col` the overlay is exactly drafts[0..col-1] (statically known,
// since column `col` is only consumed when every earlier draft was accepted), so
// the penalty at each column sees the same prefix a per-token sampler would.
// Non-speculative callers pass no overlay. The scan is bounded by k and
// only runs when penalties are active, so it is free on the no-penalty path.
__device__ __forceinline__ float sampling_adjusted_logit(float raw, int v, const SamplingConfig& c,
                                                         const std::int32_t* overlay = nullptr,
                                                         int overlay_len             = 0) {
    float x = raw;
    if (c.presence_penalty == 0.0f && c.frequency_penalty == 0.0f) { return x; }
    int cnt = c.token_counts != nullptr ? c.token_counts[v] : 0;
    for (int j = 0; j < overlay_len; ++j) {
        if (overlay[j] == v) { ++cnt; }
    }
    if (cnt > 0) { x -= c.presence_penalty; }
    if (c.frequency_penalty != 0.0f) { x -= c.frequency_penalty * static_cast<float>(cnt); }
    return x;
}

__device__ __forceinline__ void sampling_insert_candidate(float* vals, int* idxs, int cap, float v,
                                                          int idx) {
    if (cap <= 0 || !sampling_better(v, idx, vals[cap - 1], idxs[cap - 1])) { return; }
    int pos = cap - 1;
    while (pos > 0 && sampling_better(v, idx, vals[pos - 1], idxs[pos - 1])) {
        vals[pos] = vals[pos - 1];
        idxs[pos] = idxs[pos - 1];
        --pos;
    }
    vals[pos] = v;
    idxs[pos] = idx;
}

__device__ inline void sampling_sort_tile_desc(float* vals, int* idxs) {
    const int tid = threadIdx.x;
    for (int size = 2; size <= kSamplerTileItems; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            const int other = tid ^ stride;
            __syncthreads();
            if (other > tid) {
                const float ov        = vals[other];
                const int oi          = idxs[other];
                const bool descending = ((tid & size) == 0);
                const bool swap       = descending ? sampling_better(ov, oi, vals[tid], idxs[tid])
                                                   : sampling_better(vals[tid], idxs[tid], ov, oi);
                if (swap) {
                    const float tv = vals[tid];
                    const int ti   = idxs[tid];
                    vals[tid]      = ov;
                    idxs[tid]      = oi;
                    vals[other]    = tv;
                    idxs[other]    = ti;
                }
            }
            __syncthreads();
        }
    }
}

__device__ inline void sampling_normalize_support(const SamplingConfig& cfg, float* cand_val,
                                                  int* cand_idx, float* prob, int* n_support,
                                                  int n) {
    const int tid = threadIdx.x;
    if (tid == 0) {
        while (n > 0 && cand_idx[n - 1] == INT_MAX) { --n; }
        const float inv_temp = 1.0f / cfg.temperature;
        const float m        = cand_val[0] * inv_temp;
        float sum            = 0.0f;
        for (int j = 0; j < n; ++j) {
            const float e = __expf(cand_val[j] * inv_temp - m);
            prob[j]       = e;
            sum += e;
        }
        const float e0           = prob[0];
        const float min_p_thresh = (cfg.min_p > 0.0f) ? cfg.min_p * e0 : -1.0f;
        const bool top_p_active  = (cfg.top_p < 1.0f);
        const float top_p_target = cfg.top_p * sum;
        float cum                = 0.0f;
        int support              = 0;
        for (int j = 0; j < n; ++j) {
            if (min_p_thresh >= 0.0f && prob[j] < min_p_thresh) { break; }
            cum += prob[j];
            support = j + 1;
            if (top_p_active && cum >= top_p_target) { break; }
        }
        if (support < 1) { support = 1; }
        float ssum = 0.0f;
        for (int j = 0; j < support; ++j) { ssum += prob[j]; }
        const float inv = 1.0f / ssum;
        for (int j = 0; j < support; ++j) { prob[j] *= inv; }
        *n_support = support;
    }
    __syncthreads();
}

// All threads of the block must call. Small-column exact truncated support
// builder used by unit tests and any <=256-token column. It sorts one shared
// tile and then applies the same top-p/min-p renormalization as the large path.
__device__ inline void
sampling_build_truncated_small(const __nv_bfloat16* logits, std::int64_t base, std::int32_t vocab,
                               const SamplingConfig& cfg, float* tile_val, int* tile_idx,
                               float* cand_val, int* cand_idx, float* prob, int* n_support,
                               const std::int32_t* overlay = nullptr, int overlay_len = 0) {
    const int tid = threadIdx.x;
    const int cap = sampling_candidate_cap(cfg, vocab);
    if (tid < kSamplerTileItems) {
        if (tid < vocab) {
            if (sampling_token_suppressed(tid, cfg)) {
                tile_val[tid] = -CUDART_INF_F;
                tile_idx[tid] = INT_MAX;
            } else {
                tile_val[tid] = sampling_adjusted_logit(
                    __bfloat162float(logits[base + tid]), tid, cfg, overlay, overlay_len);
                tile_idx[tid] = tid;
            }
        } else {
            tile_val[tid] = -CUDART_INF_F;
            tile_idx[tid] = INT_MAX;
        }
    }
    __syncthreads();
    sampling_sort_tile_desc(tile_val, tile_idx);
    if (tid < cap) {
        cand_val[tid] = tile_val[tid];
        cand_idx[tid] = tile_idx[tid];
    }
    __syncthreads();
    sampling_normalize_support(cfg, cand_val, cand_idx, prob, n_support, cap);
}

// Single-block fallback for large columns when the finite multi-block workspace
// route cannot represent the launch. It still reads each vocab entry once and
// keeps a bounded per-thread top-20, so it avoids a top_k*vocab global reread.
__device__ inline void sampling_build_truncated_block_fast(
    const __nv_bfloat16* logits, std::int64_t base, std::int32_t vocab, const SamplingConfig& cfg,
    float* merge_val, int* merge_idx, float* cand_val, int* cand_idx, float* prob, int* n_support,
    const std::int32_t* overlay = nullptr, int overlay_len = 0) {
    const int tid = threadIdx.x;
    const int cap = sampling_candidate_cap(cfg, vocab); // always <= kSamplerFastCandidates

    float local_val[kSamplerFastCandidates];
    int local_idx[kSamplerFastCandidates];
#pragma unroll
    for (int j = 0; j < kSamplerFastCandidates; ++j) {
        local_val[j] = -CUDART_INF_F;
        local_idx[j] = INT_MAX;
    }

    const int fast_cap = cap;
    for (int v = tid; v < vocab; v += blockDim.x) {
        if (sampling_token_suppressed(v, cfg)) { continue; }
        const float x = sampling_adjusted_logit(__bfloat162float(logits[base + v]), v, cfg, overlay,
                                                overlay_len);
        sampling_insert_candidate(local_val, local_idx, fast_cap, x, v);
    }

    for (int j = 0; j < kSamplerFastCandidates; ++j) {
        const int off  = tid * kSamplerFastCandidates + j;
        merge_val[off] = local_val[j];
        merge_idx[off] = local_idx[j];
    }
    __syncthreads();

    if (tid == 0) {
        for (int j = 0; j < cap; ++j) {
            cand_val[j] = -CUDART_INF_F;
            cand_idx[j] = INT_MAX;
        }
        const int merge_n = blockDim.x * kSamplerFastCandidates;
        for (int p = 0; p < merge_n; ++p) {
            const int idx = merge_idx[p];
            if (idx == INT_MAX) { continue; }
            sampling_insert_candidate(cand_val, cand_idx, fast_cap, merge_val[p], idx);
        }
    }
    __syncthreads();
    sampling_normalize_support(cfg, cand_val, cand_idx, prob, n_support, fast_cap);
}

// thread-0 helper: inverse-CDF pick over a normalized `prob[0..n-1]` support,
// optionally excluding `exclude` (a rejected draft token) and renormalizing over
// the remainder. `u` is a uniform in [0,1). Returns the chosen vocab id.
__device__ __forceinline__ int sampling_pick_from_support(const int* cand_idx, const float* prob,
                                                          int n, int exclude, float u) {
    float mass = 0.0f;
    for (int j = 0; j < n; ++j) {
        if (cand_idx[j] == exclude) { continue; }
        mass += prob[j];
    }
    if (mass <= 0.0f) {
        // Degenerate: support is only the excluded token. Return it (caller only
        // reaches this when accept probability was ~1, i.e. it will not happen).
        return cand_idx[0];
    }
    const float goal = u * mass;
    float acc        = 0.0f;
    int picked       = -1;
    for (int j = 0; j < n; ++j) {
        if (cand_idx[j] == exclude) { continue; }
        acc += prob[j];
        picked = cand_idx[j];
        if (goal < acc) { return picked; }
    }
    return picked;
}

// q(token) over a shortlist of `n` (id, probability) pairs. Missing ids are 0.
// A null shortlist is the one-hot draft convention (q=1 at the drafted token).
__device__ __forceinline__ float sampling_selector_q(const int* ids, const float* q, int n,
                                                     int token) {
    if (ids == nullptr || q == nullptr || n <= 0) { return 1.0f; }
    for (int c = 0; c < n; ++c) {
        if (ids[c] == token) { return q[c]; }
    }
    return 0.0f;
}

// Leviathan residual: inverse-CDF of max(0, p - q) on the truncated target support.
// Null q is one-hot at `draft_id` and matches sampling_pick_from_support(..., exclude).
__device__ __forceinline__ int sampling_pick_from_p_minus_q(const int* cand_idx, const float* prob,
                                                            int n, const int* q_ids,
                                                            const float* q_vals, int q_n,
                                                            int draft_id, float u) {
    if (q_ids == nullptr || q_vals == nullptr || q_n <= 0) {
        return sampling_pick_from_support(cand_idx, prob, n, draft_id, u);
    }
    float mass = 0.0f;
    for (int j = 0; j < n; ++j) {
        mass += fmaxf(0.0f, prob[j] - sampling_selector_q(q_ids, q_vals, q_n, cand_idx[j]));
    }
    if (mass <= 0.0f) { return sampling_pick_from_support(cand_idx, prob, n, -1, u); }
    const float goal = u * mass;
    float acc        = 0.0f;
    int picked       = cand_idx[0];
    for (int j = 0; j < n; ++j) {
        const float r =
            fmaxf(0.0f, prob[j] - sampling_selector_q(q_ids, q_vals, q_n, cand_idx[j]));
        if (r <= 0.0f) { continue; }
        acc += r;
        picked = cand_idx[j];
        if (goal < acc) { return picked; }
    }
    return picked;
}

__device__ __forceinline__ float sampling_p_less_q_at(int v, int draft_id, const int* q_ids,
                                                      const float* q_vals, int q_n) {
    if (q_ids == nullptr || q_vals == nullptr || q_n <= 0) { return v == draft_id ? 1.0f : 0.0f; }
    return sampling_selector_q(q_ids, q_vals, q_n, v);
}

__device__ __forceinline__ float sampling_p_less_prob(const __nv_bfloat16* logits, std::int64_t base,
                                                      int token, std::int32_t vocab,
                                                      const SamplingConfig& cfg,
                                                      const SamplingPLessGate& gate,
                                                      float admitted) {
    if (!(admitted > 0.0f) || !sampling_p_less_in_domain(token, vocab, cfg)) { return 0.0f; }
    const float e = sampling_p_less_survivor_exp(__bfloat162float(logits[base + token]), gate);
    return e > 0.0f ? e / admitted : 0.0f;
}

// Leviathan residual inverse-CDF of max(0, p'-q) under p-less. All threads call.
__device__ inline int sampling_p_less_residual(const __nv_bfloat16* logits, std::int64_t base,
                                               std::int32_t vocab, const SamplingConfig& cfg,
                                               const SamplingPLessGate& gate, float admitted,
                                               float u, int draft_id, const int* q_ids,
                                               const float* q_vals, int q_n, int argmax,
                                               float* red_val, int* red_idx) {
    const int tid = threadIdx.x;
    // Degenerate p': no residual or p' inverse-CDF is possible.
    if (!(admitted > 0.0f)) { return argmax; }
    const float inv_z = 1.0f / admitted;
    float local_mass  = 0.0f;
    for (int v = tid; v < vocab; v += blockDim.x) {
        if (!sampling_p_less_in_domain(v, vocab, cfg)) { continue; }
        const float e = sampling_p_less_survivor_exp(__bfloat162float(logits[base + v]), gate);
        if (!(e > 0.0f)) { continue; }
        const float r = e * inv_z - sampling_p_less_q_at(v, draft_id, q_ids, q_vals, q_n);
        if (r > 0.0f) { local_mass += r; }
    }
    const float mass = sampling_block_sum(local_mass, red_val);
    if (!(mass > 0.0f)) {
        // Empty max(0,p'-q): q covered p' (16-way shortlist) or the draft was
        // the only survivor. Draw from p' rather than re-emitting the rejected
        // draft (that loop is independent of whether the selector was greedy).
        return sampling_p_less_inverse_cdf(logits, base, vocab, cfg, gate, u, argmax, red_val,
                                           red_idx);
    }
    red_val[tid] = local_mass;
    __syncthreads();
    sampling_block_inclusive_scan(red_val);
    const float inclusive = red_val[tid];
    const float exclusive = inclusive - local_mass;
    const float total     = red_val[blockDim.x - 1];
    const float goal      = u * total;
    int local_pick        = -1;
    if (local_mass > 0.0f && goal >= exclusive && (goal < inclusive || inclusive == total)) {
        float acc = exclusive;
        for (int v = tid; v < vocab; v += blockDim.x) {
            if (!sampling_p_less_in_domain(v, vocab, cfg)) { continue; }
            const float e =
                sampling_p_less_survivor_exp(__bfloat162float(logits[base + v]), gate);
            if (!(e > 0.0f)) { continue; }
            const float r = e * inv_z - sampling_p_less_q_at(v, draft_id, q_ids, q_vals, q_n);
            if (r <= 0.0f) { continue; }
            acc += r;
            local_pick = v;
            if (goal < acc) { break; }
        }
    }
    red_idx[tid] = local_pick;
    __syncthreads();
    int picked = argmax;
    for (int t = 0; t < blockDim.x; ++t) {
        if (red_idx[t] >= 0) {
            picked = red_idx[t];
            break;
        }
    }
    if (tid == 0) { red_idx[0] = picked; }
    __syncthreads();
    return red_idx[0];
}

} // namespace ninfer::ops
