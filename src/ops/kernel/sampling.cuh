#pragma once

// Implements: include/ninfer/ops/sampling.h
// Match: contiguous BF16 logits, physical stride >= token domain, and at most
// sixteen columns on the multi-block route.
// Algorithm assumptions: 256-thread/2-item partial tiles feed bounded top-20
// merges or full-vocabulary p-less moments through caller-owned workspace;
// unsupported finite geometries use the single-block fallback.

#include "ops/kernel/sampling_device.cuh"

namespace ninfer::ops {

__launch_bounds__(kSamplerBlock) __global__
    void sample_row_kernel(const __nv_bfloat16* logits, std::int32_t* out,
                           const SamplingConfig* configs, const std::int32_t* logical_positions,
                           std::int32_t purpose, std::int32_t token_domain,
                           std::int32_t physical_rows) {
    const int row            = static_cast<int>(blockIdx.x);
    const std::int64_t base  = static_cast<std::int64_t>(row) * physical_rows;
    const int tid            = threadIdx.x;
    const SamplingConfig cfg = configs[row];

    __shared__ float red_val[kSamplerBlock];
    __shared__ int red_idx[kSamplerBlock];
    __shared__ float red_aux[kSamplerBlock];

    const int partial_blocks = div_up(token_domain, kSamplerPartialTileItems);
    const int group_count    = sampler_group_count(partial_blocks);
    // The partial/group/mass path owns every mode on representable large shapes.
    // This kernel remains in the graph and handles only the small/fallback route.
    if (sampler_multiblock_ok(token_domain, static_cast<int>(gridDim.x), partial_blocks,
                              group_count)) {
        return;
    }

    if (sampling_p_less_active(cfg)) {
        const int picked =
            sampling_p_less_draw(logits, base, token_domain, cfg, logical_positions[row], purpose,
                                 red_val, red_idx, red_aux);
        if (tid == 0) {
            out[row] = picked;
            if (cfg.token_counts != nullptr) { atomicAdd(&cfg.token_counts[picked], 1); }
        }
        return;
    }

    // Greedy: exact argmax over raw logits. Bit-identical to argmax().
    if (!(cfg.temperature > 0.0f)) {
        float bv = -CUDART_INF_F;
        int bi   = INT_MAX;
        for (int v = tid; v < token_domain; v += blockDim.x) {
            if (sampling_token_suppressed(v, cfg)) { continue; }
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
        if (tid == 0) { out[row] = red_idx[0]; }
        return;
    }

    __shared__ float cand_val[kSamplerCandidateCap];
    __shared__ int cand_idx[kSamplerCandidateCap];
    __shared__ float prob[kSamplerCandidateCap];
    __shared__ int n_support;
    __shared__ float merge_val[kSamplerBlock * kSamplerFastCandidates];
    __shared__ int merge_idx[kSamplerBlock * kSamplerFastCandidates];

    if (token_domain <= kSamplerTileItems) {
        sampling_build_truncated_small(logits, base, token_domain, cfg, red_val, red_idx, cand_val,
                                       cand_idx, prob, &n_support);
    } else {
        sampling_build_truncated_block_fast(logits, base, token_domain, cfg, merge_val, merge_idx,
                                            cand_val, cand_idx, prob, &n_support);
    }

    if (tid != 0) { return; }
    const int support = n_support;
    const float u     = sampling_uniform(cfg.seed, logical_positions[row], purpose, 0u);
    float acc         = 0.0f;
    int picked        = cand_idx[support - 1];
    for (int j = 0; j < support; ++j) {
        acc += prob[j]; // prob is normalized: goal == u
        if (u < acc) {
            picked = cand_idx[j];
            break;
        }
    }
    out[row] = picked;
    if (cfg.token_counts != nullptr) { atomicAdd(&cfg.token_counts[picked], 1); }
}

__launch_bounds__(kSamplerBlock) __global__
    void sampling_partial_topk_kernel(const __nv_bfloat16* logits, const SamplingConfig* cfg_ptr,
                                      std::int32_t token_domain, std::int32_t physical_rows,
                                      SamplingWorkspace workspace) {
    const int col            = static_cast<int>(blockIdx.y);
    const int partial        = static_cast<int>(blockIdx.x);
    const SamplingConfig cfg = cfg_ptr[col];
    if (partial == 0 && threadIdx.x == 0) { workspace.group_done[col] = 0; }

    __shared__ typename SamplingPartialSort::TempStorage sort_storage;
    __shared__ unsigned long long greedy_warp_keys[kSamplerBlock / 32];
    __shared__ float p_less_red_val[kSamplerBlock];
    __shared__ float p_less_red_aux[kSamplerBlock];
    unsigned long long keys[kSamplerItemsPerThread];

    const std::int64_t base = static_cast<std::int64_t>(col) * physical_rows;
    const int tile_start    = partial * kSamplerPartialTileItems;
    if (sampling_p_less_active(cfg)) {
        const SamplingPLessMoments moments = sampling_p_less_tile_moments(
            logits, base, token_domain, cfg, tile_start, 1.0f / cfg.temperature, p_less_red_val,
            p_less_red_aux, greedy_warp_keys);
        if (threadIdx.x == 0) { sampling_p_less_store_moments(workspace, col, partial, moments); }
        return;
    }

    const bool greedy = !(cfg.temperature > 0.0f);
    const int cap     = greedy ? 1 : sampling_candidate_cap(cfg, token_domain);
#pragma unroll
    for (int item = 0; item < kSamplerItemsPerThread; ++item) {
        const int v = tile_start + item * blockDim.x + threadIdx.x;
        if (v < token_domain && !sampling_token_suppressed(v, cfg)) {
            const float raw = __bfloat162float(logits[base + v]);
            const float x   = greedy ? raw : sampling_adjusted_logit(raw, v, cfg);
            keys[item]      = sampling_sort_key(x, v);
        } else {
            keys[item] = 0ull;
        }
    }
    if (greedy) {
        unsigned long long best = keys[0];
#pragma unroll
        for (int item = 1; item < kSamplerItemsPerThread; ++item) {
            if (keys[item] > best) { best = keys[item]; }
        }
        best = sampling_block_max_key(best, greedy_warp_keys);
        if (threadIdx.x == 0) {
            const int off               = sampling_partial_offset(workspace, col, partial, 0);
            workspace.partial_keys[off] = best;
        }
        return;
    }
    SamplingPartialSort(sort_storage).Sort(keys, SamplingKeyGreater{});

#pragma unroll
    for (int item = 0; item < kSamplerItemsPerThread; ++item) {
        const int rank = threadIdx.x * kSamplerItemsPerThread + item;
        if (rank < cap) {
            const int off               = sampling_partial_offset(workspace, col, partial, rank);
            workspace.partial_keys[off] = keys[item];
        }
    }
}

__launch_bounds__(kSamplerGroupBlock) __global__ void sampling_group_finalize_sample_kernel(
    std::int32_t* out, const SamplingConfig* cfg_ptr, const std::int32_t* logical_positions,
    std::int32_t purpose, std::int32_t token_domain, std::int32_t partial_blocks,
    std::int32_t group_count, SamplingWorkspace workspace) {
    const int group          = static_cast<int>(blockIdx.x);
    const int col            = static_cast<int>(blockIdx.y);
    const int tid            = threadIdx.x;
    const SamplingConfig cfg = cfg_ptr[col];
    __shared__ typename SamplingGroupSort::TempStorage sort_storage;
    __shared__ float cand_val[kSamplerCandidateCap];
    __shared__ int cand_idx[kSamplerCandidateCap];
    __shared__ float prob[kSamplerCandidateCap];
    __shared__ unsigned long long published_keys[kSamplerCandidateCap];
    __shared__ int n_support;
    __shared__ int is_last;
    __shared__ unsigned long long greedy_warp_keys[kSamplerGroupBlock / 32];
    __shared__ float p_less_red_val[kSamplerGroupBlock];
    __shared__ float p_less_red_aux[kSamplerGroupBlock];
    unsigned long long keys[kSamplerGroupItemsPerThread];

    // The preceding partial launch initializes group_done[col], so caller-owned
    // workspace does not rely on prior contents or a separate memset launch.

    const int group_begin = group * kSamplerPartialsPerGroup;
    int group_partials    = partial_blocks - group_begin;
    if (group_partials < 0) { group_partials = 0; }
    if (group_partials > kSamplerPartialsPerGroup) { group_partials = kSamplerPartialsPerGroup; }

    if (sampling_p_less_active(cfg)) {
        const float inv_temp = 1.0f / cfg.temperature;
        const SamplingPLessMoments group_moments = sampling_p_less_merge_moments(
            workspace, col, group_begin, group_partials, inv_temp, p_less_red_val,
            p_less_red_aux, greedy_warp_keys);
        if (tid == 0) {
            sampling_p_less_store_moments(workspace, col, partial_blocks + group, group_moments);
            const int done = sampling_completion_add(&workspace.group_done[col], group_count);
            is_last        = done == group_count ? 1 : 0;
        }
        __syncthreads();
        if (!is_last) { return; }

        const SamplingPLessMoments global_moments = sampling_p_less_merge_moments(
            workspace, col, partial_blocks, group_count, inv_temp, p_less_red_val,
            p_less_red_aux, greedy_warp_keys);
        if (tid == 0) {
            sampling_p_less_store_global(workspace, col, global_moments);
            workspace.group_done[col] = 0;
        }
        return;
    }

    const bool greedy = !(cfg.temperature > 0.0f);
    const int cap     = greedy ? 1 : sampling_candidate_cap(cfg, token_domain);
    if (greedy) {
        unsigned long long best = 0ull;
        for (int p = tid; p < group_partials; p += blockDim.x) {
            const int off = sampling_partial_offset(workspace, col, group_begin + p, 0);
            if (workspace.partial_keys[off] > best) { best = workspace.partial_keys[off]; }
        }
        best = sampling_block_max_key(best, greedy_warp_keys);
        if (tid == 0) {
            const int out_off = sampling_partial_offset(workspace, col, partial_blocks + group, 0);
            sampling_publish_key(workspace, out_off, best);
            const int done = sampling_completion_add(&workspace.group_done[col], group_count);
            is_last        = (done == group_count) ? 1 : 0;
        }
        __syncthreads();
        if (!is_last) { return; }

        best = 0ull;
        for (int p = tid; p < group_count; p += blockDim.x) {
            const int off = sampling_partial_offset(workspace, col, partial_blocks + p, 0);
            const unsigned long long candidate = sampling_load_published_key(workspace, off);
            if (candidate > best) { best = candidate; }
        }
        best = sampling_block_max_key(best, greedy_warp_keys);
        if (tid == 0) {
            out[col]                  = sampling_key_index(best);
            workspace.group_done[col] = 0;
        }
        return;
    }

    const int group_n = group_partials * cap;
#pragma unroll
    for (int item = 0; item < kSamplerGroupItemsPerThread; ++item) {
        const int p = item * blockDim.x + tid;
        if (p < group_n) {
            const int partial = group_begin + p / cap;
            const int j       = p - (p / cap) * cap;
            const int off     = sampling_partial_offset(workspace, col, partial, j);
            keys[item]        = workspace.partial_keys[off];
        } else {
            keys[item] = 0ull;
        }
    }
    SamplingGroupSort(sort_storage).Sort(keys, SamplingKeyGreater{});

#pragma unroll
    for (int item = 0; item < kSamplerGroupItemsPerThread; ++item) {
        const int rank = tid * kSamplerGroupItemsPerThread + item;
        if (rank < cap) { published_keys[rank] = keys[item]; }
    }
    __syncthreads();

    if (tid == 0) {
        for (int rank = 0; rank < cap; ++rank) {
            const int out_off =
                sampling_partial_offset(workspace, col, partial_blocks + group, rank);
            sampling_publish_key(workspace, out_off, published_keys[rank]);
        }
        const int done = sampling_completion_add(&workspace.group_done[col], group_count);
        is_last        = (done == group_count) ? 1 : 0;
    }
    __syncthreads();
    if (!is_last) { return; }

    const int final_n = group_count * cap;
#pragma unroll
    for (int item = 0; item < kSamplerGroupItemsPerThread; ++item) {
        const int p = item * blockDim.x + tid;
        if (p < final_n) {
            const int partial = partial_blocks + p / cap;
            const int j       = p - (p / cap) * cap;
            const int off     = sampling_partial_offset(workspace, col, partial, j);
            keys[item]        = sampling_load_published_key(workspace, off);
        } else {
            keys[item] = 0ull;
        }
    }
    SamplingGroupSort(sort_storage).Sort(keys, SamplingKeyGreater{});

#pragma unroll
    for (int item = 0; item < kSamplerGroupItemsPerThread; ++item) {
        const int rank = tid * kSamplerGroupItemsPerThread + item;
        if (rank < cap) {
            cand_val[rank] = sampling_key_float(keys[item]);
            cand_idx[rank] = sampling_key_index(keys[item]);
        }
    }
    __syncthreads();

    sampling_normalize_support(cfg, cand_val, cand_idx, prob, &n_support, cap);
    if (tid == 0) {
        const int support = n_support;
        const float u     = sampling_uniform(cfg.seed, logical_positions[col], purpose, 0u);
        float acc         = 0.0f;
        int picked        = cand_idx[support - 1];
        for (int j = 0; j < support; ++j) {
            acc += prob[j];
            if (u < acc) {
                picked = cand_idx[j];
                break;
            }
        }
        out[col] = picked;
        if (cfg.token_counts != nullptr) { atomicAdd(&cfg.token_counts[picked], 1); }
        workspace.group_done[col] = 0;
    }
}

__launch_bounds__(kSamplerBlock) __global__ void sampling_p_less_mass_sample_kernel(
    const __nv_bfloat16* logits, std::int32_t* out, const SamplingConfig* cfg_ptr,
    const std::int32_t* logical_positions, std::int32_t purpose, std::int32_t token_domain,
    std::int32_t physical_rows, std::int32_t partial_blocks, SamplingWorkspace workspace) {
    const int col            = static_cast<int>(blockIdx.y);
    const SamplingConfig cfg = cfg_ptr[col];
    if (!sampling_p_less_active(cfg)) { return; }

    __shared__ float warp_sums[kSamplerBlock];
    __shared__ float weights[kSamplerBlock];
    __shared__ float admitted;
    __shared__ float selected_goal;
    __shared__ float running;
    __shared__ int is_last;
    __shared__ int selected_tile;
    __shared__ int picked;
    __shared__ int found;

    const SamplingPLessMoments moments = sampling_p_less_load_global(workspace, col);
    const SamplingPLessGate gate =
        sampling_p_less_gate(moments, 1.0f / cfg.temperature);
    const std::int64_t base            = static_cast<std::int64_t>(col) * physical_rows;
    for (int partial = static_cast<int>(blockIdx.x); partial < partial_blocks;
         partial += static_cast<int>(gridDim.x)) {
        const int tile_start = partial * kSamplerPartialTileItems;
        const float tile_max = sampling_p_less_load_tile_max(workspace, col, partial);
        const float tile_mass = sampling_p_less_tile_admitted_mass(
            logits, base, token_domain, cfg, tile_start, gate, tile_max, warp_sums);

        if (threadIdx.x == 0) {
            sampling_p_less_store_tile_mass(workspace, col, partial, tile_mass);
            const int done = sampling_completion_add(&workspace.group_done[col], partial_blocks);
            is_last        = done == partial_blocks ? 1 : 0;
        }
        __syncthreads();
        if (!is_last) { continue; }

        if (threadIdx.x == 0) {
            float total = 0.0f;
            for (int p = 0; p < partial_blocks; ++p) {
                total += sampling_p_less_load_tile_mass(workspace, col, p);
            }
            admitted = total;
            sampling_p_less_store_admitted(workspace, col, total);
            selected_tile = -1;
            selected_goal = 0.0f;
            picked        = moments.argmax;
            if (total > 0.0f) {
                const float goal =
                    sampling_uniform(cfg.seed, logical_positions[col], purpose, 0u) * total;
                float prefix      = 0.0f;
                int last_nonempty = -1;
                float last_prefix = 0.0f;
                for (int p = 0; p < partial_blocks; ++p) {
                    const float mass = sampling_p_less_load_tile_mass(workspace, col, p);
                    if (mass > 0.0f) {
                        last_nonempty = p;
                        last_prefix   = prefix;
                    }
                    if (goal < prefix + mass) {
                        selected_tile = p;
                        selected_goal = goal - prefix;
                        break;
                    }
                    prefix += mass;
                }
                if (selected_tile < 0 && last_nonempty >= 0) {
                    selected_tile = last_nonempty;
                    selected_goal = fmaxf(0.0f, goal - last_prefix);
                }
            }
        }
        __syncthreads();

        int result = moments.argmax;
        if (selected_tile >= 0) {
            result = sampling_p_less_pick_from_tile(
                logits, base, token_domain, cfg, selected_tile, gate, admitted, selected_goal,
                moments.argmax, false, -1, nullptr, nullptr, 0, weights, &running, &picked,
                &found);
        }
        if (threadIdx.x == 0) {
            out[col] = result;
            if (cfg.token_counts != nullptr) { atomicAdd(&cfg.token_counts[result], 1); }
            workspace.group_done[col] = 0;
        }
        return;
    }
}

} // namespace ninfer::ops
