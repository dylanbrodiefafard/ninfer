#pragma once

// Implements: include/ninfer/ops/speculative_round.h
// Match: contiguous request-major state and BF16 verification logits.
// Algorithm assumptions: small vocabularies use one cooperative block; the
// full-vocabulary stochastic and p-less routes use the sampling partial/group
// pipeline and caller-owned workspace, while greedy commit remains one thread.

#include "ops/kernel/sampling_device.cuh"

#include <cuda_bf16.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

__global__ void speculative_prepare_verify_inputs_kernel(const std::int32_t* anchors,
                                                         const std::int32_t* drafts,
                                                         const std::int32_t* base_positions,
                                                         const std::int32_t* current_extents,
                                                         std::int32_t* verify_ids,
                                                         std::int32_t* positions, std::int32_t k) {
    const int row = static_cast<int>(blockIdx.y);
    const int T   = k + 1;
    int extent    = current_extents[row];
    extent        = extent < 0 ? 0 : (extent > k ? k : extent);
    for (int j = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x; j < T;
         j += blockDim.x * gridDim.x) {
        const int off = row * T + j;
        verify_ids[off] =
            j == 0 ? anchors[row] : (j <= extent ? drafts[row * k + j - 1] : anchors[row]);
        if (positions != nullptr) {
            positions[off] = base_positions[row] + (j <= extent ? j : extent);
        }
    }
}

template <typename T>
__device__ inline T* speculative_workspace_offset(T* ptr, std::size_t byte_offset) {
    return ptr == nullptr
               ? nullptr
               : reinterpret_cast<T*>(reinterpret_cast<unsigned char*>(ptr) + byte_offset);
}

__device__ inline SamplingWorkspace
speculative_workspace_row(SamplingWorkspace workspace, std::size_t row_stride, std::int32_t row) {
    const std::size_t offset = static_cast<std::size_t>(row) * row_stride;
    workspace.partial_keys   = speculative_workspace_offset(workspace.partial_keys, offset);
    workspace.dist_idx       = speculative_workspace_offset(workspace.dist_idx, offset);
    workspace.dist_prob      = speculative_workspace_offset(workspace.dist_prob, offset);
    workspace.dist_support   = speculative_workspace_offset(workspace.dist_support, offset);
    workspace.group_done     = speculative_workspace_offset(workspace.group_done, offset);
    workspace.speculative_finalize_count =
        speculative_workspace_offset(workspace.speculative_finalize_count, offset);
    return workspace;
}

// Optional chain-draft q shortlist for hop `i` of row `row`. Layout is [C,K,B]
// with C fastest. Null ids means one-hot q at the drafted token.
__device__ __forceinline__ void speculative_chain_hop_q(const std::int32_t* selector_ids,
                                                        const float* selector_q, int selector_k,
                                                        int hop, int row, int k,
                                                        const int*& hop_ids, const float*& hop_q) {
    hop_ids = nullptr;
    hop_q   = nullptr;
    if (selector_ids == nullptr || selector_q == nullptr || selector_k <= 0) { return; }
    const int base = selector_k * (hop + row * k);
    hop_ids        = selector_ids + base;
    hop_q          = selector_q + base;
}

// P-less temperature parameterizes the target p', not a 16-way draft softmax.
// A recorded selector q at that T is nearly uniform and would make Leviathan
// accept almost every top-16 copy (command loops / punctuation collapse). Same
// one-hot convention as MTP: ignore the shortlist.
__device__ __forceinline__ void
speculative_chain_hop_q_for_target(const SamplingConfig& cfg, const std::int32_t* selector_ids,
                                   const float* selector_q, int selector_k, int hop, int row, int k,
                                   const int*& hop_ids, const float*& hop_q) {
    hop_ids = nullptr;
    hop_q   = nullptr;
    if (sampling_p_less_active(cfg)) { return; }
    speculative_chain_hop_q(selector_ids, selector_q, selector_k, hop, row, k, hop_ids, hop_q);
}

__device__ inline int speculative_tree_child_for_token(const std::int32_t* parent_index,
                                                       const std::int32_t* verify_ids, int node,
                                                       int token, int valid) {
    for (int c = node + 1; c < valid; ++c) {
        if (parent_index[c] == node && verify_ids[c] == token) { return c; }
    }
    return -1;
}

// Tokens already committed on the packed path from root to `node`, excluding the
// root (the old anchor). Same overlay the sequential tree walk builds in lic_sh.
__device__ inline int speculative_tree_path_overlay(const std::int32_t* ids,
                                                    const std::int32_t* pars, int node,
                                                    int* overlay) {
    int stack[kSamplerMaxColumns];
    int n = 0;
    int cur = node;
    while (cur > 0 && n < kSamplerMaxColumns) {
        stack[n++] = cur;
        cur        = pars[cur];
        if (cur < 0) { break; }
    }
    for (int i = 0; i < n; ++i) { overlay[i] = ids[stack[n - 1 - i]]; }
    return n;
}

__device__ inline void speculative_tree_greedy_commit(
    const std::int32_t* ttok, const std::int32_t* ids, const std::int32_t* pars, std::int32_t* lic,
    std::int32_t* path, int valid, int extent, int width, int row, std::int32_t* licensed_counts,
    std::int32_t* accepted, std::int32_t* accepted_column, std::int32_t* anchors,
    std::int32_t* lengths) {
    for (int i = 0; i < width; ++i) {
        lic[i]  = 0;
        path[i] = 0;
    }
    int node = 0;
    int a    = 0;
    path[0]  = 0;
    while (a < extent) {
        const int predicted = ttok[node];
        const int child = speculative_tree_child_for_token(pars, ids, node, predicted, valid);
        if (child < 0) {
            lic[a]               = predicted;
            licensed_counts[row] = a + 1;
            accepted[row]        = a;
            accepted_column[row] = node;
            anchors[row]         = predicted;
            lengths[row] += a + 1;
            return;
        }
        lic[a]      = predicted;
        node        = child;
        path[a + 1] = child;
        ++a;
    }
    lic[a]               = ttok[node];
    licensed_counts[row] = a + 1;
    accepted[row]        = a;
    accepted_column[row] = node;
    anchors[row]         = ttok[node];
    lengths[row] += a + 1;
}

__device__ inline void speculative_tree_sampling_commit(
    std::int32_t* lic, std::int32_t* path, int width, int row, const int* lic_src,
    const int* path_src, int a, int tstar, int node, std::int32_t* licensed_counts,
    std::int32_t* accepted, std::int32_t* accepted_column, std::int32_t* anchors,
    std::int32_t* lengths, int L, const SamplingConfig& cfg, std::int32_t token_domain) {
    for (int i = 0; i < width; ++i) {
        lic[i]  = 0;
        path[i] = 0;
    }
    for (int i = 0; i < a; ++i) {
        lic[i] = sampling_clamp_token(lic_src[i], 0, token_domain);
    }
    lic[a] = sampling_clamp_token(tstar, 0, token_domain);
    for (int i = 0; i <= a; ++i) { path[i] = path_src[i]; }
    const int produced   = a + 1;
    licensed_counts[row] = produced;
    accepted[row]        = a;
    accepted_column[row] = node;
    anchors[row]         = lic[a];
    lengths[row]         = L + produced;
    for (int i = 0; i < produced; ++i) { sampling_count_token(cfg, lic[i], token_domain); }
}

// Commits the round's accepted tokens plus one correction/bonus token, then
// advances the target length. The greedy branch
// (config temperature <= 0) is bit-identical to the original argmax accept: keep
// the longest draft prefix whose target argmax matches, then take the target
// argmax at the divergence column. The sampling branch (temperature > 0) runs
// Leviathan speculative rejection sampling over the verify logits: accept
// drafts[i] with probability min(1, p_i(d)/q_i(d)) under the truncated target
// distribution, resample from max(0, p-q) on the first rejection, and draw a
// bonus from the last column when every draft accepts. Null selector q is the
// one-hot (greedy draft) convention, so the accept test collapses to `u < p(d)`
// and the residual excludes d. Launch with a single block of kSamplerBlock
// threads; only thread 0 performs the sequential accept/commit while the whole
// block cooperates on the per-column truncated-distribution build.
__launch_bounds__(kSamplerBlock) __global__ void speculative_accept_greedy_drafts_kernel(
    const std::int32_t* target_tokens, const __nv_bfloat16* logits, const std::int32_t* drafts,
    const std::int32_t* current_extents, std::int32_t* lengths, std::int32_t* anchors,
    std::int32_t* licensed_tokens, std::int32_t* licensed_counts, std::int32_t* accepted,
    const SamplingConfig* configs, std::int32_t token_domain, std::int32_t physical_rows,
    std::int32_t k, const std::int32_t* selector_ids, const float* selector_q,
    std::int32_t selector_k) {
    const int tid                   = threadIdx.x;
    const int row                   = static_cast<int>(blockIdx.x);
    const int cols                  = k + 1;
    int extent                      = current_extents[row];
    extent                          = extent < 0 ? 0 : (extent > k ? k : extent);
    const SamplingConfig cfg        = configs[row];
    const std::int32_t* row_targets = target_tokens + row * cols;
    const std::int32_t* row_drafts  = drafts + row * k;
    std::int32_t* row_tokens        = licensed_tokens + row * cols;
    const __nv_bfloat16* row_logits =
        logits + static_cast<std::int64_t>(row) * cols * physical_rows;

    const int partial_blocks = div_up(token_domain, kSamplerPartialTileItems);
    const int group_count    = sampler_group_count(partial_blocks);
    if (sampler_multiblock_ok(token_domain, cols, partial_blocks, group_count)) { return; }

    if (!(cfg.temperature > 0.0f)) {
        if (tid == 0) {
            int a = 0;
            while (a < extent && row_targets[a] == row_drafts[a]) { ++a; }
            const int t_star = row_targets[a];

            for (int i = 0; i <= k; ++i) { row_tokens[i] = 0; }
            for (int i = 0; i < a; ++i) { row_tokens[i] = row_drafts[i]; }
            row_tokens[a] = t_star;

            const int produced   = a + 1;
            licensed_counts[row] = produced;
            accepted[row]        = a;
            anchors[row]         = t_star;
            lengths[row] += produced;
        }
        return;
    }

    __shared__ float red_val[kSamplerBlock];
    __shared__ int red_idx[kSamplerBlock];
    __shared__ float red_aux[kSamplerBlock];
    __shared__ float cand_val[kSamplerCandidateCap];
    __shared__ int cand_idx[kSamplerCandidateCap];
    __shared__ float prob[kSamplerCandidateCap];
    __shared__ float merge_val[kSamplerBlock * kSamplerFastCandidates];
    __shared__ int merge_idx[kSamplerBlock * kSamplerFastCandidates];
    __shared__ int n_support;
    __shared__ int a_sh;
    __shared__ int done_sh;
    __shared__ int tstar_sh;
    __shared__ int L_sh;
    __shared__ int decision_sh;

    const bool p_less = sampling_p_less_active(cfg);

    if (tid == 0) {
        a_sh     = 0;
        done_sh  = 0;
        tstar_sh = 0;
        L_sh     = lengths[row];
    }
    __syncthreads();

    for (int i = 0; i <= extent; ++i) {
        const std::int64_t base = static_cast<std::int64_t>(i) * physical_rows;
        if (p_less) {
            const float inv_temp = 1.0f / cfg.temperature;
            const SamplingPLessMoments st = sampling_p_less_moments(
                row_logits, base, token_domain, cfg, inv_temp, red_val, red_idx, red_aux);
            const SamplingPLessGate gate = sampling_p_less_gate(st, inv_temp);
            const float admitted = sampling_p_less_admitted_mass(
                row_logits, base, token_domain, cfg, gate, red_val);
            if (tid == 0) {
                decision_sh = 0;
                if (done_sh == 0) {
                    const int L = L_sh;
                    if (i == 0) {
                        if (i < extent) {
                            const int d        = row_drafts[i];
                            const float pd     = sampling_p_less_prob(
                                row_logits, base, d, token_domain, cfg, gate, admitted);
                            const int* hop_ids = nullptr;
                            const float* hop_q = nullptr;
                            speculative_chain_hop_q_for_target(cfg, selector_ids, selector_q,
                                                               selector_k, i, row, k, hop_ids,
                                                               hop_q);
                            const float qd = sampling_selector_q(
                                hop_ids, hop_q, selector_k > 0 ? selector_k : 0, d);
                            const float u = sampling_uniform(cfg.seed, L + i + 1,
                                                             kSamplePurposeSpeculativeAccept, 0u);
                            const bool take = qd > 0.0f && u < fminf(1.0f, pd / qd);
                            if (take) {
                                a_sh        = i + 1;
                                decision_sh = 0;
                            } else {
                                decision_sh = 1;
                            }
                        } else {
                            decision_sh = 2;
                        }
                    } else if (i < extent) {
                        if (row_drafts[i] == st.argmax) {
                            a_sh        = i + 1;
                            decision_sh = 0;
                        } else {
                            tstar_sh    = st.argmax;
                            decision_sh = 3;
                            done_sh     = 1;
                        }
                    } else {
                        tstar_sh    = st.argmax;
                        decision_sh = 3;
                        done_sh     = 1;
                    }
                }
            }
            __syncthreads();
            if (decision_sh == 3) { break; }
            if (decision_sh == 1) {
                const int d        = row_drafts[i];
                const int* hop_ids = nullptr;
                const float* hop_q = nullptr;
                speculative_chain_hop_q_for_target(cfg, selector_ids, selector_q, selector_k, i,
                                                   row, k, hop_ids, hop_q);
                const float ur = sampling_uniform(cfg.seed, L_sh + i + 1,
                                                  kSamplePurposeSpeculativeCorrection, 0u);
                const int tstar = sampling_p_less_residual(
                    row_logits, base, token_domain, cfg, gate, admitted, ur, d, hop_ids, hop_q,
                    selector_k, st.argmax, red_val, red_idx);
                if (tid == 0) {
                    tstar_sh = tstar;
                    done_sh  = 1;
                }
                __syncthreads();
                break;
            }
            if (decision_sh == 2) {
                const float u =
                    sampling_uniform(cfg.seed, L_sh + extent + 1, kSamplePurposeSpeculativeBonus,
                                     0u);
                const int tstar = sampling_p_less_inverse_cdf(
                    row_logits, base, token_domain, cfg, gate, u, st.argmax, red_val, red_idx);
                if (tid == 0) {
                    tstar_sh = tstar;
                    done_sh  = 1;
                }
                __syncthreads();
                break;
            }
            continue;
        }
        if (token_domain <= kSamplerTileItems) {
            sampling_build_truncated_small(row_logits, base, token_domain, cfg, red_val, red_idx,
                                           cand_val, cand_idx, prob, &n_support, row_drafts, i);
        } else {
            sampling_build_truncated_block_fast(row_logits, base, token_domain, cfg, merge_val,
                                                merge_idx, cand_val, cand_idx, prob, &n_support,
                                                row_drafts, i);
        }
        if (tid == 0 && done_sh == 0) {
            const int L = L_sh;
            if (i < extent) {
                const int d = row_drafts[i];
                float pd    = 0.0f;
                for (int j = 0; j < n_support; ++j) {
                    if (cand_idx[j] == d) {
                        pd = prob[j];
                        break;
                    }
                }
                const int* hop_ids  = nullptr;
                const float* hop_q  = nullptr;
                speculative_chain_hop_q_for_target(cfg, selector_ids, selector_q, selector_k, i,
                                                   row, k, hop_ids, hop_q);
                const float qd =
                    sampling_selector_q(hop_ids, hop_q, selector_k > 0 ? selector_k : 0, d);
                const float u =
                    sampling_uniform(cfg.seed, L + i + 1, kSamplePurposeSpeculativeAccept, 0u);
                const bool take = qd > 0.0f && u < fminf(1.0f, pd / qd);
                if (take) {
                    a_sh = i + 1;
                } else {
                    const float ur = sampling_uniform(cfg.seed, L + i + 1,
                                                      kSamplePurposeSpeculativeCorrection, 0u);
                    tstar_sh       = sampling_pick_from_p_minus_q(cand_idx, prob, n_support, hop_ids,
                                                                  hop_q, selector_k, d, ur);
                    done_sh        = 1;
                }
            } else {
                const float u =
                    sampling_uniform(cfg.seed, L + extent + 1, kSamplePurposeSpeculativeBonus, 0u);
                tstar_sh = sampling_pick_from_support(cand_idx, prob, n_support, -1, u);
                done_sh  = 1;
            }
        }
        __syncthreads();
        if (done_sh) { break; }
    }

    if (tid == 0) {
        const int a     = a_sh;
        const int tstar = tstar_sh;
        const int L     = L_sh;

        for (int i = 0; i <= k; ++i) { row_tokens[i] = 0; }
        for (int i = 0; i < a; ++i) { row_tokens[i] = row_drafts[i]; }
        row_tokens[a] = tstar;

        const int produced   = a + 1;
        licensed_counts[row] = produced;
        accepted[row]        = a;
        anchors[row]         = tstar;
        lengths[row]         = L + produced;
        for (int i = 0; i < produced; ++i) {
            row_tokens[i] = sampling_clamp_token(row_tokens[i], 0, token_domain);
            sampling_count_token(cfg, row_tokens[i], token_domain);
        }
        anchors[row] = row_tokens[a];
    }
}

__launch_bounds__(kSamplerBlock) __global__ void speculative_sampling_partial_topk_kernel(
    const __nv_bfloat16* logits, const std::int32_t* drafts, const std::int32_t* current_extents,
    const SamplingConfig* configs, std::int32_t token_domain, std::int32_t physical_rows,
    std::int32_t cols, std::int32_t k, SamplingWorkspace workspace,
    std::size_t workspace_row_stride, const std::int32_t* verify_ids,
    const std::int32_t* parent_index, const std::int32_t* valid_columns) {
    const int row     = static_cast<int>(blockIdx.z);
    const int col     = static_cast<int>(blockIdx.y);
    const int partial = static_cast<int>(blockIdx.x);
    const bool tree   = parent_index != nullptr;
    int extent        = current_extents[row];
    extent            = extent < 0 ? 0 : (extent > k ? k : extent);
    if (tree) {
        int valid = valid_columns[row];
        valid     = valid < 1 ? 1 : (valid > cols ? cols : valid);
        if (col >= valid) { return; }
    } else if (col > extent) {
        return;
    }
    const SamplingConfig cfg = configs[row];
    if (!(cfg.temperature > 0.0f) || token_domain <= kSamplerTileItems) {
        return;
    }
    workspace = speculative_workspace_row(workspace, workspace_row_stride, row);
    if (partial == 0 && threadIdx.x == 0) {
        workspace.group_done[col] = 0;
        if (col == 0) { *workspace.speculative_finalize_count = 0; }
    }

    __shared__ typename SamplingPartialSort::TempStorage sort_storage;
    __shared__ unsigned long long p_less_warp_keys[kSamplerBlock / 32];
    __shared__ float p_less_red_val[kSamplerBlock];
    __shared__ float p_less_red_aux[kSamplerBlock];
    unsigned long long keys[kSamplerItemsPerThread];

    const std::int64_t base = (static_cast<std::int64_t>(row) * cols + col) * physical_rows;
    const int tile_start    = partial * kSamplerPartialTileItems;
    if (sampling_p_less_active(cfg)) {
        const SamplingPLessMoments moments = sampling_p_less_tile_moments(
            logits, base, token_domain, cfg, tile_start, 1.0f / cfg.temperature, p_less_red_val,
            p_less_red_aux, p_less_warp_keys);
        if (threadIdx.x == 0) { sampling_p_less_store_moments(workspace, col, partial, moments); }
        return;
    }

    const int cap = sampling_candidate_cap(cfg, token_domain);
    int overlay_buf[kSamplerMaxColumns];
    const std::int32_t* overlay = nullptr;
    int overlay_len             = 0;
    if (tree) {
        overlay_len = speculative_tree_path_overlay(verify_ids + row * cols,
                                                    parent_index + row * cols, col, overlay_buf);
        overlay     = overlay_buf;
    } else {
        overlay     = drafts + row * k;
        overlay_len = col;
    }
    // Chain: overlay is drafts[0..col). Tree: overlay is the packed path tokens
    // from root to this node (excluding the anchor). Applied before top-k so
    // penalties can change the candidate set, not just post-truncation mass.
#pragma unroll
    for (int item = 0; item < kSamplerItemsPerThread; ++item) {
        const int v = tile_start + item * blockDim.x + threadIdx.x;
        if (v < token_domain && !sampling_token_suppressed(v, cfg)) {
            const float x = sampling_adjusted_logit(__bfloat162float(logits[base + v]), v, cfg,
                                                    overlay, overlay_len);
            keys[item]    = sampling_sort_key(x, v);
        } else {
            keys[item] = 0ull;
        }
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

__launch_bounds__(kSamplerGroupBlock) __global__ void speculative_sampling_group_finalize_kernel(
    const std::int32_t* target_tokens, const std::int32_t* drafts,
    const std::int32_t* current_extents, std::int32_t* lengths, std::int32_t* anchors,
    std::int32_t* licensed_tokens, std::int32_t* licensed_counts, std::int32_t* accepted,
    const SamplingConfig* configs, std::int32_t token_domain, std::int32_t cols,
    std::int32_t partial_blocks, std::int32_t group_count, SamplingWorkspace workspace,
    std::size_t workspace_row_stride, const std::int32_t* selector_ids, const float* selector_q,
    std::int32_t selector_k, const std::int32_t* verify_ids, const std::int32_t* parent_index,
    const std::int32_t* valid_columns, std::int32_t* accepted_column, std::int32_t* fold_path) {
    const int row   = static_cast<int>(blockIdx.z);
    const int group = static_cast<int>(blockIdx.x);
    const int col   = static_cast<int>(blockIdx.y);
    const int tid   = threadIdx.x;
    const int k     = cols - 1;
    const bool tree = parent_index != nullptr;
    int extent      = current_extents[row];
    extent          = extent < 0 ? 0 : (extent > k ? k : extent);
    int valid       = cols;
    if (tree) {
        valid = valid_columns[row];
        valid = valid < 1 ? 1 : (valid > cols ? cols : valid);
        if (col >= valid) { return; }
    } else if (col > extent) {
        return;
    }
    const SamplingConfig cfg        = configs[row];
    const std::int32_t* row_targets = target_tokens + row * cols;
    const std::int32_t* row_drafts  = drafts != nullptr ? drafts + row * k : nullptr;
    std::int32_t* row_tokens        = licensed_tokens + row * cols;
    const std::int32_t* ids         = tree ? verify_ids + row * cols : nullptr;
    const std::int32_t* pars        = tree ? parent_index + row * cols : nullptr;
    std::int32_t* path              = tree ? fold_path + row * cols : nullptr;
    if (token_domain <= kSamplerTileItems) { return; }

    if (!(cfg.temperature > 0.0f)) {
        if (tid == 0 && col == 0 && group == 0) {
            if (tree) {
                speculative_tree_greedy_commit(row_targets, ids, pars, row_tokens, path, valid,
                                               extent, cols, row, licensed_counts, accepted,
                                               accepted_column, anchors, lengths);
            } else {
                int a = 0;
                while (a < extent && row_targets[a] == row_drafts[a]) { ++a; }
                const int t_star = row_targets[a];
                for (int i = 0; i <= k; ++i) { row_tokens[i] = 0; }
                for (int i = 0; i < a; ++i) { row_tokens[i] = row_drafts[i]; }
                row_tokens[a]        = t_star;
                const int produced   = a + 1;
                licensed_counts[row] = produced;
                accepted[row]        = a;
                anchors[row]         = t_star;
                lengths[row] += produced;
            }
        }
        return;
    }

    workspace = speculative_workspace_row(workspace, workspace_row_stride, row);

    __shared__ typename SamplingGroupSort::TempStorage sort_storage;
    __shared__ float cand_val[kSamplerCandidateCap];
    __shared__ int cand_idx[kSamplerCandidateCap];
    __shared__ float prob[kSamplerCandidateCap];
    __shared__ unsigned long long published_keys[kSamplerCandidateCap];
    __shared__ int n_support;
    __shared__ int is_last_group;
    __shared__ unsigned long long p_less_warp_keys[kSamplerGroupBlock / 32];
    __shared__ float p_less_red_val[kSamplerGroupBlock];
    __shared__ float p_less_red_aux[kSamplerGroupBlock];
    unsigned long long keys[kSamplerGroupItemsPerThread];

    // The preceding partial launch initializes all caller-owned counters. CUDA
    // stream ordering makes those writes visible before this launch begins.

    const int group_begin = group * kSamplerPartialsPerGroup;
    int group_partials    = partial_blocks - group_begin;
    if (group_partials < 0) { group_partials = 0; }
    if (group_partials > kSamplerPartialsPerGroup) { group_partials = kSamplerPartialsPerGroup; }

    if (sampling_p_less_active(cfg)) {
        const float inv_temp = 1.0f / cfg.temperature;
        const SamplingPLessMoments group_moments = sampling_p_less_merge_moments(
            workspace, col, group_begin, group_partials, inv_temp, p_less_red_val,
            p_less_red_aux, p_less_warp_keys);
        if (tid == 0) {
            sampling_p_less_store_moments(workspace, col, partial_blocks + group, group_moments);
            const int done = sampling_completion_add(&workspace.group_done[col], group_count);
            is_last_group  = done == group_count ? 1 : 0;
        }
        __syncthreads();
        if (!is_last_group) { return; }

        const SamplingPLessMoments global_moments = sampling_p_less_merge_moments(
            workspace, col, partial_blocks, group_count, inv_temp, p_less_red_val,
            p_less_red_aux, p_less_warp_keys);
        if (tid == 0) {
            sampling_p_less_store_global(workspace, col, global_moments);
            workspace.group_done[col] = 0;
        }
        return;
    }

    const int cap = sampling_candidate_cap(cfg, token_domain);
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
        is_last_group  = (done == group_count) ? 1 : 0;
    }
    __syncthreads();
    if (!is_last_group) { return; }

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
        sampling_publish_i32(workspace.dist_support, col, n_support);
        for (int j = 0; j < n_support; ++j) {
            const int off = sampling_dist_offset(col, j);
            sampling_publish_i32(workspace.dist_idx, off, cand_idx[j]);
            sampling_publish_float(workspace.dist_prob, off, prob[j]);
        }
        workspace.group_done[col] = 0;
        const int need_cols = tree ? valid : (extent + 1);
        const int done_cols =
            sampling_completion_add(workspace.speculative_finalize_count, need_cols);
        if (done_cols == need_cols) {
            const int L = lengths[row];
            if (tree) {
                int path_local[kSamplerMaxColumns];
                int lic_local[kSamplerMaxColumns];
                int node  = 0;
                int a     = 0;
                int tstar = 0;
                path_local[0] = 0;
                for (int step = 0; step <= extent; ++step) {
                    const int n = sampling_load_published_i32(workspace.dist_support, node);
                    const int* dist_idx =
                        workspace.dist_idx + sampling_dist_offset(node, 0);
                    const float* dist_prob =
                        workspace.dist_prob + sampling_dist_offset(node, 0);
                    const float u = sampling_uniform(
                        cfg.seed, L + a + 1, kSamplePurposeSpeculativeAccept, 0u);
                    const int sampled = sampling_pick_from_support(dist_idx, dist_prob, n, -1, u);
                    const int child   = step < extent ? speculative_tree_child_for_token(
                                                          pars, ids, node, sampled, valid)
                                                      : -1;
                    if (child >= 0) {
                        lic_local[a]      = sampled;
                        path_local[a + 1] = child;
                        node              = child;
                        ++a;
                    } else {
                        tstar = sampled;
                        break;
                    }
                }
                speculative_tree_sampling_commit(row_tokens, path, cols, row, lic_local, path_local,
                                                 a, tstar, node, licensed_counts, accepted,
                                                 accepted_column, anchors, lengths, L, cfg,
                                                 token_domain);
            } else {
            int a     = 0;
            int tstar = 0;
            for (int i = 0; i <= extent; ++i) {
                const int n = sampling_load_published_i32(workspace.dist_support, i);
                const int* dist_idx =
                    workspace.dist_idx + sampling_dist_offset(i, 0);
                const float* dist_prob =
                    workspace.dist_prob + sampling_dist_offset(i, 0);
                if (i < extent) {
                    const int d = row_drafts[i];
                    float pd    = 0.0f;
                    for (int j = 0; j < n; ++j) {
                        if (dist_idx[j] == d) {
                            pd = dist_prob[j];
                            break;
                        }
                    }
                    const int* hop_ids = nullptr;
                    const float* hop_q = nullptr;
                    speculative_chain_hop_q_for_target(cfg, selector_ids, selector_q, selector_k, i,
                                                       row, k, hop_ids, hop_q);
                    const float qd =
                        sampling_selector_q(hop_ids, hop_q, selector_k > 0 ? selector_k : 0, d);
                    const float u =
                        sampling_uniform(cfg.seed, L + i + 1, kSamplePurposeSpeculativeAccept, 0u);
                    // A zero-q draft is outside the proposal support and must be corrected.
                    const bool take = qd > 0.0f && u < fminf(1.0f, pd / qd);
                    if (take) {
                        a = i + 1;
                        continue;
                    }
                    const float ur = sampling_uniform(cfg.seed, L + i + 1,
                                                      kSamplePurposeSpeculativeCorrection, 0u);
                    tstar          = sampling_pick_from_p_minus_q(dist_idx, dist_prob, n, hop_ids,
                                                                  hop_q, selector_k, d, ur);
                    break;
                }
                const float u =
                    sampling_uniform(cfg.seed, L + extent + 1, kSamplePurposeSpeculativeBonus, 0u);
                tstar = sampling_pick_from_support(dist_idx, dist_prob, n, -1, u);
            }
            for (int i = 0; i <= k; ++i) { row_tokens[i] = 0; }
            for (int i = 0; i < a; ++i) { row_tokens[i] = row_drafts[i]; }
            row_tokens[a]        = tstar;
            const int produced   = a + 1;
            licensed_counts[row] = produced;
            accepted[row]        = a;
            anchors[row]         = tstar;
            lengths[row]         = L + produced;
            for (int i = 0; i < produced; ++i) {
                row_tokens[i] = sampling_clamp_token(row_tokens[i], 0, token_domain);
                sampling_count_token(cfg, row_tokens[i], token_domain);
            }
            anchors[row] = row_tokens[a];
            }
            *workspace.speculative_finalize_count = 0;
        }
    }
}

__device__ __forceinline__ void
speculative_p_less_store_aux_mass(const SamplingWorkspace& workspace, int col, int partial,
                                  float mass) {
    // Slot 1 aliases the per-partial moments pair after it has been merged into the dist region.
    workspace.partial_keys[sampling_partial_offset(workspace, col, partial, 1)] =
        static_cast<unsigned long long>(__float_as_uint(mass));
}

__device__ __forceinline__ float
speculative_p_less_load_aux_mass(const SamplingWorkspace& workspace, int col, int partial) {
    return __uint_as_float(static_cast<unsigned int>(
        workspace.partial_keys[sampling_partial_offset(workspace, col, partial, 1)]));
}

__device__ inline float speculative_p_less_residual_tile_mass(
    const __nv_bfloat16* row_logits, std::int64_t base, std::int32_t token_domain,
    const SamplingConfig& cfg, int tile, const SamplingWorkspace& workspace, int col,
    const SamplingPLessGate& gate, float admitted, int draft_id, const int* q_ids,
    const float* q_vals, int q_n) {
    if (!(admitted > 0.0f)) { return 0.0f; }
    float correction = 0.0f;
    const int begin  = tile * kSamplerPartialTileItems;
    const int end    = begin + kSamplerPartialTileItems;
    if (q_ids == nullptr || q_vals == nullptr || q_n <= 0) {
        if (sampling_p_less_in_domain(draft_id, token_domain, cfg) && draft_id >= begin &&
            draft_id < end) {
            const float e = sampling_p_less_survivor_exp(
                __bfloat162float(row_logits[base + draft_id]), gate);
            if (e > 0.0f) { correction = fminf(e / admitted, 1.0f); }
        }
    } else {
        for (int c = 0; c < q_n; ++c) {
            const int token = q_ids[c];
            bool duplicate  = false;
            for (int j = 0; j < c; ++j) {
                if (q_ids[j] == token) {
                    duplicate = true;
                    break;
                }
            }
            if (duplicate || token < begin || token >= end ||
                !sampling_p_less_in_domain(token, token_domain, cfg)) {
                continue;
            }
            const float e = sampling_p_less_survivor_exp(
                __bfloat162float(row_logits[base + token]), gate);
            if (e > 0.0f) { correction += fminf(e / admitted, q_vals[c]); }
        }
    }
    const float p_mass = sampling_p_less_load_tile_mass(workspace, col, tile) / admitted;
    return fmaxf(0.0f, p_mass - correction);
}

// Thread-0 helper. Chooses a tile in either p' or max(0,p'-q) and leaves only
// that tile for the cooperative 512-token rescan.
__device__ inline void speculative_p_less_choose_tile(
    const __nv_bfloat16* row_logits, std::int64_t base, std::int32_t token_domain,
    const SamplingConfig& cfg, const SamplingWorkspace& workspace, int col, int partial_blocks,
    float inv_temp, bool residual, int draft_id, const int* q_ids, const float* q_vals, int q_n,
    float u, int* selected_tile, float* selected_goal, int* fallback) {
    const SamplingPLessMoments moments = sampling_p_less_load_global(workspace, col);
    const SamplingPLessGate gate       = sampling_p_less_gate(moments, inv_temp);
    const float admitted               = sampling_p_less_load_admitted(workspace, col);
    *selected_tile = -1;
    *selected_goal = 0.0f;
    *fallback      = sampling_clamp_token(moments.argmax, 0, token_domain);

    float total = admitted;
    if (residual) {
        total = 0.0f;
        for (int p = 0; p < partial_blocks; ++p) {
            const float mass = speculative_p_less_residual_tile_mass(
                row_logits, base, token_domain, cfg, p, workspace, col, gate, admitted, draft_id,
                q_ids, q_vals, q_n);
            speculative_p_less_store_aux_mass(workspace, col, p, mass);
            total += mass;
        }
    }
    if (!(total > 0.0f)) { return; }

    const float goal = u * total;
    float prefix     = 0.0f;
    int last         = -1;
    float last_begin = 0.0f;
    for (int p = 0; p < partial_blocks; ++p) {
        const float mass = residual ? speculative_p_less_load_aux_mass(workspace, col, p)
                                    : sampling_p_less_load_tile_mass(workspace, col, p);
        if (mass > 0.0f) {
            last       = p;
            last_begin = prefix;
        }
        if (goal < prefix + mass) {
            *selected_tile = p;
            *selected_goal = goal - prefix;
            return;
        }
        prefix += mass;
    }
    if (last >= 0) {
        *selected_tile = last;
        *selected_goal = fmaxf(0.0f, goal - last_begin);
    }
}

__launch_bounds__(kSamplerBlock) __global__ void speculative_sampling_p_less_mass_finalize_kernel(
    const __nv_bfloat16* logits, const std::int32_t* drafts,
    const std::int32_t* current_extents, std::int32_t* lengths, std::int32_t* anchors,
    std::int32_t* licensed_tokens, std::int32_t* licensed_counts, std::int32_t* accepted,
    const SamplingConfig* configs, std::int32_t token_domain, std::int32_t physical_rows,
    std::int32_t cols, std::int32_t partial_blocks, SamplingWorkspace workspace,
    std::size_t workspace_row_stride, const std::int32_t* selector_ids, const float* selector_q,
    std::int32_t selector_k, const std::int32_t* verify_ids, const std::int32_t* parent_index,
    const std::int32_t* valid_columns, std::int32_t* accepted_column, std::int32_t* fold_path) {
    const int row     = static_cast<int>(blockIdx.z);
    const int col     = static_cast<int>(blockIdx.y);
    const int k       = cols - 1;
    const bool tree   = parent_index != nullptr;
    int extent        = current_extents[row];
    extent            = extent < 0 ? 0 : (extent > k ? k : extent);
    int valid         = cols;
    if (tree) {
        valid = valid_columns[row];
        valid = valid < 1 ? 1 : (valid > cols ? cols : valid);
        if (col >= valid) { return; }
    } else if (col > extent) {
        return;
    }
    const SamplingConfig cfg = configs[row];
    if (!sampling_p_less_active(cfg)) { return; }

    workspace = speculative_workspace_row(workspace, workspace_row_stride, row);
    const __nv_bfloat16* row_logits =
        logits + static_cast<std::int64_t>(row) * cols * physical_rows;
    const std::int64_t base = static_cast<std::int64_t>(col) * physical_rows;

    __shared__ float warp_sums[kSamplerBlock];
    __shared__ float weights[kSamplerBlock];
    __shared__ float running;
    __shared__ float selected_goal;
    __shared__ int is_last;
    __shared__ int is_finalizer;
    __shared__ int selected_tile;
    __shared__ int fallback;
    __shared__ int picked;
    __shared__ int found;
    __shared__ int action;
    __shared__ int node_sh;
    __shared__ int a_sh;
    __shared__ int tstar_sh;
    __shared__ int L_sh;
    __shared__ int path_sh[kSamplerMaxColumns];
    __shared__ int lic_sh[kSamplerMaxColumns];

    const SamplingPLessMoments moments = sampling_p_less_load_global(workspace, col);
    const SamplingPLessGate gate       = sampling_p_less_gate(moments, 1.0f / cfg.temperature);
    if (threadIdx.x == 0) { is_finalizer = 0; }
    __syncthreads();
    for (int partial = static_cast<int>(blockIdx.x); partial < partial_blocks;
         partial += static_cast<int>(gridDim.x)) {
        const float tile_max = sampling_p_less_load_tile_max(workspace, col, partial);
        const float tile_mass = sampling_p_less_tile_admitted_mass(
            row_logits, base, token_domain, cfg, partial * kSamplerPartialTileItems, gate,
            tile_max, warp_sums);
        if (threadIdx.x == 0) {
            sampling_p_less_store_tile_mass(workspace, col, partial, tile_mass);
            const int done = sampling_completion_add(&workspace.group_done[col], partial_blocks);
            is_last        = done == partial_blocks ? 1 : 0;
        }
        __syncthreads();
        if (!is_last) { continue; }

        if (threadIdx.x == 0) {
            float admitted = 0.0f;
            for (int p = 0; p < partial_blocks; ++p) {
                admitted += sampling_p_less_load_tile_mass(workspace, col, p);
            }
            sampling_p_less_store_admitted(workspace, col, admitted);
            workspace.group_done[col] = 0;
            const int need_cols = tree ? valid : (extent + 1);
            const int done_cols =
                sampling_completion_add(workspace.speculative_finalize_count, need_cols);
            is_finalizer        = done_cols == need_cols ? 1 : 0;
        }
        __syncthreads();
        if (!is_finalizer) { return; }
        break;
    }
    if (!is_finalizer) { return; }

    const float inv_temp = 1.0f / cfg.temperature;
    const std::int32_t* row_drafts = drafts != nullptr ? drafts + row * k : nullptr;
    std::int32_t* row_tokens        = licensed_tokens + row * cols;
    const std::int32_t* ids         = tree ? verify_ids + row * cols : nullptr;
    const std::int32_t* pars        = tree ? parent_index + row * cols : nullptr;
    std::int32_t* path              = tree ? fold_path + row * cols : nullptr;
    if (threadIdx.x == 0) {
        node_sh  = 0;
        a_sh     = 0;
        tstar_sh = 0;
        L_sh     = lengths[row];
        path_sh[0] = 0;
        for (int i = 0; i < cols; ++i) { lic_sh[i] = 0; }
    }
    __syncthreads();

    if (tree) {
        for (int step = 0; step <= extent; ++step) {
            const int node = node_sh;
            const SamplingPLessMoments node_moments =
                sampling_p_less_load_global(workspace, node);
            const SamplingPLessGate node_gate = sampling_p_less_gate(node_moments, inv_temp);
            const float admitted = sampling_p_less_load_admitted(workspace, node);
            int sampled          = node_moments.argmax;
            if (step == 0) {
                if (threadIdx.x == 0) {
                    const float u = sampling_uniform(cfg.seed, L_sh + a_sh + 1,
                                                     kSamplePurposeSpeculativeAccept, 0u);
                    speculative_p_less_choose_tile(
                        row_logits, static_cast<std::int64_t>(node) * physical_rows, token_domain,
                        cfg, workspace, node, partial_blocks, inv_temp, false, -1, nullptr, nullptr,
                        0, u, &selected_tile, &selected_goal, &fallback);
                }
                __syncthreads();
                sampled = fallback;
                if (selected_tile >= 0) {
                    sampled = sampling_p_less_pick_from_tile(
                        row_logits, static_cast<std::int64_t>(node) * physical_rows, token_domain,
                        cfg, selected_tile, node_gate, admitted, selected_goal, fallback, false, -1,
                        nullptr, nullptr, 0, weights, &running, &picked, &found);
                }
            }
            if (threadIdx.x == 0) {
                const int child =
                    step < extent
                        ? speculative_tree_child_for_token(pars, ids, node, sampled, valid)
                        : -1;
                if (child >= 0) {
                    lic_sh[a_sh]      = sampled;
                    path_sh[a_sh + 1] = child;
                    node_sh           = child;
                    ++a_sh;
                    action = 0;
                } else {
                    tstar_sh = sampled;
                    action   = 1;
                }
            }
            __syncthreads();
            if (action != 0) { break; }
        }
        if (threadIdx.x == 0) {
            speculative_tree_sampling_commit(
                row_tokens, path, cols, row, lic_sh, path_sh, a_sh, tstar_sh, node_sh,
                licensed_counts, accepted, accepted_column, anchors, lengths, L_sh, cfg,
                token_domain);
            *workspace.speculative_finalize_count = 0;
        }
        return;
    }

    for (int i = 0; i <= extent; ++i) {
        const SamplingPLessMoments col_moments =
            sampling_p_less_load_global(workspace, i);
        const SamplingPLessGate col_gate = sampling_p_less_gate(col_moments, inv_temp);
        const float admitted = sampling_p_less_load_admitted(workspace, i);
        const int d = i < extent ? row_drafts[i] : -1;
        const int* hop_ids = nullptr;
        const float* hop_q = nullptr;
        if (i < extent) {
            speculative_chain_hop_q_for_target(cfg, selector_ids, selector_q, selector_k, i, row,
                                               k, hop_ids, hop_q);
        }
        if (threadIdx.x == 0) {
            if (i > 0) {
                if (i < extent) {
                    if (d == col_moments.argmax) {
                        a_sh   = i + 1;
                        action = 0;
                    } else {
                        tstar_sh = col_moments.argmax;
                        action   = 3;
                    }
                } else {
                    tstar_sh = col_moments.argmax;
                    action   = 3;
                }
            } else if (i < extent) {
                const float pd = sampling_p_less_prob(
                    row_logits, static_cast<std::int64_t>(i) * physical_rows, d, token_domain, cfg,
                    col_gate, admitted);
                const float qd =
                    sampling_selector_q(hop_ids, hop_q, selector_k > 0 ? selector_k : 0, d);
                const float u =
                    sampling_uniform(cfg.seed, L_sh + i + 1, kSamplePurposeSpeculativeAccept, 0u);
                if (qd > 0.0f && u < fminf(1.0f, pd / qd)) {
                    a_sh   = i + 1;
                    action = 0;
                } else {
                    action = 1;
                }
            } else {
                action = 2;
            }
        }
        __syncthreads();
        if (action == 0) { continue; }
        if (action == 3) { break; }

        if (threadIdx.x == 0) {
            const bool residual = action == 1;
            const float u       = sampling_uniform(
                cfg.seed, L_sh + i + 1,
                residual ? kSamplePurposeSpeculativeCorrection : kSamplePurposeSpeculativeBonus,
                0u);
            speculative_p_less_choose_tile(
                row_logits, static_cast<std::int64_t>(i) * physical_rows, token_domain, cfg,
                workspace, i, partial_blocks, inv_temp, residual, d, hop_ids, hop_q, selector_k, u,
                &selected_tile, &selected_goal, &fallback);
            if (selected_tile < 0 && residual) {
                speculative_p_less_choose_tile(
                    row_logits, static_cast<std::int64_t>(i) * physical_rows, token_domain, cfg,
                    workspace, i, partial_blocks, inv_temp, false, d, hop_ids, hop_q, selector_k, u,
                    &selected_tile, &selected_goal, &fallback);
                action = 2;
            }
        }
        __syncthreads();
        int sampled = fallback;
        if (selected_tile >= 0) {
            sampled = sampling_p_less_pick_from_tile(
                row_logits, static_cast<std::int64_t>(i) * physical_rows, token_domain, cfg,
                selected_tile, col_gate, admitted, selected_goal, fallback, action == 1, d,
                hop_ids, hop_q, selector_k, weights, &running, &picked, &found);
        }
        if (threadIdx.x == 0) { tstar_sh = sampled; }
        __syncthreads();
        break;
    }

    if (threadIdx.x == 0) {
        for (int i = 0; i <= k; ++i) { row_tokens[i] = 0; }
        for (int i = 0; i < a_sh; ++i) { row_tokens[i] = row_drafts[i]; }
        row_tokens[a_sh]        = tstar_sh;
        const int produced      = a_sh + 1;
        licensed_counts[row]    = produced;
        accepted[row]           = a_sh;
        anchors[row]            = tstar_sh;
        lengths[row]            = L_sh + produced;
        for (int i = 0; i < produced; ++i) {
            row_tokens[i] = sampling_clamp_token(row_tokens[i], 0, token_domain);
            sampling_count_token(cfg, row_tokens[i], token_domain);
        }
        anchors[row] = row_tokens[a_sh];
        *workspace.speculative_finalize_count = 0;
    }
}

__global__ void speculative_select_accepted_hidden_kernel(const __nv_bfloat16* hidden,
                                                          const std::int32_t* selectors,
                                                          __nv_bfloat16* out, std::int32_t rows,
                                                          std::int32_t cols) {
    const int batch = static_cast<int>(blockIdx.y);
    const int row   = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) { return; }
    const int col = selectors[batch];
    if (col < 0 || col >= cols) { return; }
    out[static_cast<std::int64_t>(batch) * rows + row] =
        hidden[(static_cast<std::int64_t>(batch) * cols + col) * rows + row];
}

__global__ void proposal_remap_token_ids_kernel(std::int32_t* proposal_tokens,
                                                std::int32_t proposal_count,
                                                const std::int32_t* id_map, std::int32_t n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= proposal_count) { return; }
    const int idx = proposal_tokens[i];
    if (idx >= 0 && idx < n) { proposal_tokens[i] = id_map[idx]; }
}

__launch_bounds__(kSamplerBlock) __global__ void speculative_accept_tree_drafts_kernel(
    const std::int32_t* target_tokens, const __nv_bfloat16* logits, const std::int32_t* verify_ids,
    const std::int32_t* parent_index, const std::int32_t* valid_columns,
    const std::int32_t* current_extents, std::int32_t* lengths, std::int32_t* anchors,
    std::int32_t* licensed_tokens, std::int32_t* licensed_counts, std::int32_t* accepted,
    std::int32_t* accepted_column, std::int32_t* fold_path, const SamplingConfig* configs,
    std::int32_t token_domain, std::int32_t physical_rows, std::int32_t width) {
    const int tid            = threadIdx.x;
    const int row            = static_cast<int>(blockIdx.x);
    int valid                = valid_columns[row];
    valid                    = valid < 1 ? 1 : (valid > width ? width : valid);
    int extent               = current_extents[row];
    extent                   = extent < 0 ? 0 : (extent > width - 1 ? width - 1 : extent);
    const SamplingConfig cfg = configs[row];
    const std::int32_t* ids  = verify_ids + row * width;
    const std::int32_t* pars = parent_index + row * width;
    const std::int32_t* ttok = target_tokens + row * width;
    std::int32_t* lic        = licensed_tokens + row * width;
    std::int32_t* path       = fold_path + row * width;
    const __nv_bfloat16* row_logits =
        logits + static_cast<std::int64_t>(row) * width * physical_rows;

    const int partial_blocks = div_up(token_domain, kSamplerPartialTileItems);
    const int group_count    = sampler_group_count(partial_blocks);
    if (sampler_multiblock_ok(token_domain, width, partial_blocks, group_count)) { return; }

    if (!(cfg.temperature > 0.0f)) {
        if (tid == 0) {
            speculative_tree_greedy_commit(ttok, ids, pars, lic, path, valid, extent, width, row,
                                           licensed_counts, accepted, accepted_column, anchors,
                                           lengths);
        }
        return;
    }

    const bool p_less = sampling_p_less_active(cfg);

    __shared__ float red_val[kSamplerBlock];
    __shared__ int red_idx[kSamplerBlock];
    __shared__ float red_aux[kSamplerBlock];
    __shared__ float cand_val[kSamplerCandidateCap];
    __shared__ int cand_idx[kSamplerCandidateCap];
    __shared__ float prob[kSamplerCandidateCap];
    __shared__ float merge_val[kSamplerBlock * kSamplerFastCandidates];
    __shared__ int merge_idx[kSamplerBlock * kSamplerFastCandidates];
    __shared__ int n_support;
    __shared__ int node_sh;
    __shared__ int a_sh;
    __shared__ int done_sh;
    __shared__ int tstar_sh;
    __shared__ int L_sh;
    __shared__ int path_sh[kSamplerMaxColumns];
    __shared__ int lic_sh[kSamplerMaxColumns];

    if (tid == 0) {
        node_sh  = 0;
        a_sh     = 0;
        done_sh  = 0;
        tstar_sh = 0;
        L_sh     = lengths[row];
        path_sh[0] = 0;
        for (int i = 0; i < width; ++i) {
            lic_sh[i] = 0;
        }
    }
    __syncthreads();

    for (int step = 0; step <= extent; ++step) {
        const int node            = node_sh;
        const std::int64_t base =
            static_cast<std::int64_t>(node) * physical_rows;
        int sampled = 0;
        if (p_less) {
            if (step == 0) {
                sampled = sampling_p_less_draw(row_logits, base, token_domain, cfg, L_sh + a_sh + 1,
                                               kSamplePurposeSpeculativeAccept, red_val, red_idx,
                                               red_aux);
            } else {
                const float inv_temp = 1.0f / cfg.temperature;
                const SamplingPLessMoments st = sampling_p_less_moments(
                    row_logits, base, token_domain, cfg, inv_temp, red_val, red_idx, red_aux);
                sampled = st.argmax;
            }
        } else if (token_domain <= kSamplerTileItems) {
            sampling_build_truncated_small(row_logits, base, token_domain, cfg, red_val, red_idx,
                                           cand_val, cand_idx, prob, &n_support, lic_sh, a_sh);
        } else {
            sampling_build_truncated_block_fast(row_logits, base, token_domain, cfg, merge_val,
                                                merge_idx, cand_val, cand_idx, prob, &n_support,
                                                lic_sh, a_sh);
        }
        if (tid == 0 && done_sh == 0) {
            if (!p_less) {
                const float u = sampling_uniform(cfg.seed, L_sh + a_sh + 1,
                                                 kSamplePurposeSpeculativeAccept, 0u);
                sampled       = sampling_pick_from_support(cand_idx, prob, n_support, -1, u);
            }
            const int child =
                step < extent ? speculative_tree_child_for_token(pars, ids, node, sampled, valid)
                              : -1;
            if (child >= 0) {
                lic_sh[a_sh]      = sampled;
                path_sh[a_sh + 1] = child;
                node_sh           = child;
                ++a_sh;
            } else {
                tstar_sh = sampled;
                done_sh  = 1;
            }
        }
        __syncthreads();
        if (done_sh) { break; }
    }

    if (tid == 0) {
        speculative_tree_sampling_commit(lic, path, width, row, lic_sh, path_sh, a_sh, tstar_sh,
                                         node_sh, licensed_counts, accepted, accepted_column,
                                         anchors, lengths, L_sh, cfg, token_domain);
    }
}

} // namespace ninfer::ops
