#pragma once

#include "ops/common/mma.cuh"
#include "ops/linear_attention/gated_delta_net/chunked/common.cuh"

#include <cmath>

// Stage 4: chunk_output. Builds attn_out per chunk via fused MM2 + MM3.
//
// Math + I/O layouts: see the Gated DeltaNet chunked chunk_output_config.
// Block / smem total: 4 warps (128 t) with __launch_bounds__(128, 3);
//                     ~48 KB smem at S=128 (q permanent + k_pass / h / v
//                     alias + g).

namespace ninfer::ops::detail::gated_delta_net::chunked::output {

using ninfer::ops::mma_tf32;
using ninfer::ops::exp2_approx;

static_assert(kChunkSize == 64,
              "stage_chunk_output: kChunkSize must be 64 (kernel hard-codes BT=64)");
static_assert(kStateDim == 128);

constexpr int N_WARPS = 4;
constexpr int THREADS = N_WARPS * ninfer::ops::kWarpSize; // 128

static_assert(BT == N_WARPS * MMA_M,
              "kernel assigns one 16-row strip per warp; BT must equal N_WARPS * MMA_M");

constexpr int K_TILES_BT = BT / MMA_K; // 8 K-tiles when K-axis = BT (MM3)
constexpr int N_TILES_BT = BT / MMA_N; // 8 N-tiles when N-axis = BT (MM2)

struct kernel_dims {
    static constexpr int N_TILES_TOTAL     = kStateDim / MMA_N;
    static constexpr int N_TILES_PER_CHUNK = 4;
    static constexpr int N_CHUNKS          = N_TILES_TOTAL / N_TILES_PER_CHUNK;
    static_assert(N_CHUNKS * N_TILES_PER_CHUNK == N_TILES_TOTAL,
                  "N_TILES_PER_CHUNK must divide N_TILES_TOTAL");
    static constexpr int D_CHUNK = N_TILES_PER_CHUNK * MMA_N;
    static constexpr int K_TILES = kStateDim / MMA_K;

    static constexpr int K_TILE_PASSES    = 2;
    static constexpr int K_TILES_PER_PASS = K_TILES / K_TILE_PASSES;
    static constexpr int K_PER_PASS       = kStateDim / K_TILE_PASSES;

    static constexpr int Q_FLOATS      = BT * kStateDim;
    static constexpr int K_PASS_FLOATS = BT * K_PER_PASS;
    static constexpr int H_PART_FLOATS = D_CHUNK * kStateDim;
    static constexpr int V_PART_FLOATS = BT * D_CHUNK;
    static constexpr int PART_FLOATS   = H_PART_FLOATS;
    static_assert(V_PART_FLOATS <= PART_FLOATS);

    // shared_smem aliases k_pass (Phase B) and h_part / v_part (Phase E).
    static constexpr int SHARED_FLOATS = K_PASS_FLOATS;
    static_assert(SHARED_FLOATS == PART_FLOATS);
    static constexpr int SMEM_FLOATS = Q_FLOATS + SHARED_FLOATS + BT;
};

__launch_bounds__(THREADS, 3) __global__
    void output_kernel(const __nv_bfloat16* __restrict__ q_in,
                       const __nv_bfloat16* __restrict__ k_in,
                       const __nv_bfloat16* __restrict__ v_new_in,
                       const float* __restrict__ g_cumsum_in,
                       const __nv_bfloat16* __restrict__ h_chunk_in,
                       __nv_bfloat16* __restrict__ attn_out, head_map qk_map, float scale) {
    constexpr int N_TILES_PER_CHUNK = kernel_dims::N_TILES_PER_CHUNK;
    constexpr int N_CHUNKS          = kernel_dims::N_CHUNKS;
    constexpr int D_CHUNK           = kernel_dims::D_CHUNK;
    constexpr int K_TILES           = kernel_dims::K_TILES;
    constexpr int K_TILE_PASSES     = kernel_dims::K_TILE_PASSES;
    constexpr int K_TILES_PER_PASS  = kernel_dims::K_TILES_PER_PASS;
    constexpr int K_PER_PASS        = kernel_dims::K_PER_PASS;
    constexpr int Q_FLOATS          = kernel_dims::Q_FLOATS;
    constexpr int SHARED_FLOATS     = kernel_dims::SHARED_FLOATS;

    extern __shared__ float smem[];
    float* const q_smem      = smem;
    float* const shared_smem = q_smem + Q_FLOATS; // alias: k_pass / h_part / v_part
    float* const g_smem      = q_smem + Q_FLOATS + SHARED_FLOATS;

    SmemTile<kStateDim> q_view{q_smem};
    SmemTile<K_PER_PASS> k_pass_view{shared_smem};
    SmemTile<kStateDim> h_part_view{shared_smem};
    SmemTile<D_CHUNK> v_part_view{shared_smem};

    const int tid    = static_cast<int>(threadIdx.x);
    const int lane   = tid & (kWarpSize - 1);
    const int warp   = tid / kWarpSize;
    const int lane_g = lane >> 2;
    const int lane_t = lane & 3;

    const int chunk                = static_cast<int>(blockIdx.x);
    const int h_v                  = static_cast<int>(blockIdx.y);
    const std::int64_t cs          = static_cast<std::int64_t>(chunk) * BT;
    const std::int64_t H_v         = qk_map.H_v;
    const std::int64_t qk_stride_t = static_cast<std::int64_t>(qk_map.H_qk) * kStateDim;
    const std::int64_t qk_head_idx = static_cast<std::int64_t>(qk_map.qk_head(h_v)) * kStateDim;
    const std::int64_t q_base      = cs * qk_stride_t + qk_head_idx;
    const std::int64_t k_base      = cs * qk_stride_t + qk_head_idx;
    const std::int64_t vn_base = cs * H_v * kStateDim + static_cast<std::int64_t>(h_v) * kStateDim;
    const std::int64_t hc_base =
        (static_cast<std::int64_t>(chunk) * H_v + h_v) * kStateDim * kStateDim;
    const std::int64_t out_base = vn_base;

    const int64_t v_row_stride   = H_v * kStateDim;
    const int64_t out_row_stride = v_row_stride;

    // === Phase A: bf16 q + k_pass[0] -> float smem; sync-load g_cumsum ===
    issue_load_bf16_to_float_vec4<BT, kStateDim, THREADS>(q_view, q_in + q_base, qk_stride_t, tid);
    issue_load_bf16_to_float_vec4<BT, K_PER_PASS, THREADS>(k_pass_view, k_in + k_base, qk_stride_t,
                                                           tid);

    if (tid < BT) {
        const std::int64_t goff = (cs + tid) * H_v + h_v;
        g_smem[tid]             = g_cumsum_in[goff];
    }

    __syncthreads();

    // === Phase B: MM2 = Q @ K^T, accumulate over K_TILE_PASSES passes ===
    float A_strip[N_TILES_BT][4] = {};

    const int row_g0 = warp * MMA_M + lane_g; // M-row top half
    const int row_g1 = row_g0 + 8;            // M-row bottom half

#pragma unroll
    for (int pass = 0; pass < K_TILE_PASSES; ++pass) {
#pragma unroll
        for (int kt_local = 0; kt_local < K_TILES_PER_PASS; ++kt_local) {
            const int kt_global    = pass * K_TILES_PER_PASS + kt_local;
            const int k_off_global = kt_global * MMA_K;
            const int k_off_local  = kt_local * MMA_K;
            const int col_t0_g     = k_off_global + lane_t;
            const int col_t1_g     = col_t0_g + 4;
            const int col_t0_l     = k_off_local + lane_t;
            const int col_t1_l     = col_t0_l + 4;

            const float a0 = q_view.at(row_g0, col_t0_g);
            const float a1 = q_view.at(row_g1, col_t0_g);
            const float a2 = q_view.at(row_g0, col_t1_g);
            const float a3 = q_view.at(row_g1, col_t1_g);

#pragma unroll
            for (int nt = 0; nt < N_TILES_BT; ++nt) {
                const int n_off = nt * MMA_N;
                const int row_t =
                    n_off + lane_g; // B operand: row=t, but B is K^T so rows are k cols
                const float b0 = k_pass_view.at(row_t, col_t0_l);
                const float b1 = k_pass_view.at(row_t, col_t1_l);
                mma_tf32(A_strip[nt][0], A_strip[nt][1], A_strip[nt][2], A_strip[nt][3], a0, a1, a2,
                         a3, b0, b1);
            }
        }

        // Issue + wait next pass's k_pass into the same alias region.
        if (pass + 1 < K_TILE_PASSES) {
            __syncthreads();
            issue_load_bf16_to_float_vec4<BT, K_PER_PASS, THREADS>(
                k_pass_view, k_in + k_base + (int64_t)(pass + 1) * K_PER_PASS, qk_stride_t, tid);
            __syncthreads();
        }
    }

    // === Phase C: mask + decay -> A[r,s] = (s<=r) ? A_raw * exp(g_r-g_s) : 0 ===
    // (s>r) entries: exp2 may saturate to +inf, but the conditional select
    // overwrites the bad product with 0.0f; no NaN escapes.
    {
        const int r0     = warp * MMA_M + lane_g;
        const int r1     = r0 + 8;
        const float g_r0 = g_smem[r0];
        const float g_r1 = g_smem[r1];

#pragma unroll
        for (int nt = 0; nt < N_TILES_BT; ++nt) {
            const int s0     = nt * MMA_N + 2 * lane_t;
            const int s1     = s0 + 1;
            const float g_s0 = g_smem[s0];
            const float g_s1 = g_smem[s1];

            const float dec00 = exp2_approx((g_r0 - g_s0) * kLog2E);
            const float dec01 = exp2_approx((g_r0 - g_s1) * kLog2E);
            const float dec10 = exp2_approx((g_r1 - g_s0) * kLog2E);
            const float dec11 = exp2_approx((g_r1 - g_s1) * kLog2E);

            A_strip[nt][0] = (s0 <= r0) ? A_strip[nt][0] * dec00 : 0.0f;
            A_strip[nt][1] = (s1 <= r0) ? A_strip[nt][1] * dec01 : 0.0f;
            A_strip[nt][2] = (s0 <= r1) ? A_strip[nt][2] * dec10 : 0.0f;
            A_strip[nt][3] = (s1 <= r1) ? A_strip[nt][3] * dec11 : 0.0f;
        }
    }

    // === Phase D: D-frag -> A-frag conversion via warp shuffles ===
    //
    // PTX m16n8k8.row.col fragment layouts (lane T = (g, t)):
    //   D[T][0..3] = (row g, col 2t) / (row g, col 2t+1) / (g+8, 2t) / (g+8, 2t+1)
    //   A[T][0..3] = (row g, col t)  / (row g+8, col t)  / (g, t+4)  / (g+8, t+4)
    // 8 shfls per K-tile; A_strip is dead after this and the compiler reuses
    // its registers for A_a.
    float A_a[K_TILES_BT][4];
    {
        const int src_lo        = (lane_g << 2) | (lane_t >> 1);
        const int src_hi        = src_lo + 2;
        const bool t_odd        = (lane_t & 1) != 0;
        constexpr unsigned mask = 0xFFFFFFFFu;

#pragma unroll
        for (int kt = 0; kt < K_TILES_BT; ++kt) {
            const float d0_lo = __shfl_sync(mask, A_strip[kt][0], src_lo);
            const float d1_lo = __shfl_sync(mask, A_strip[kt][1], src_lo);
            const float d2_lo = __shfl_sync(mask, A_strip[kt][2], src_lo);
            const float d3_lo = __shfl_sync(mask, A_strip[kt][3], src_lo);
            const float d0_hi = __shfl_sync(mask, A_strip[kt][0], src_hi);
            const float d1_hi = __shfl_sync(mask, A_strip[kt][1], src_hi);
            const float d2_hi = __shfl_sync(mask, A_strip[kt][2], src_hi);
            const float d3_hi = __shfl_sync(mask, A_strip[kt][3], src_hi);

            A_a[kt][0] = t_odd ? d1_lo : d0_lo;
            A_a[kt][1] = t_odd ? d3_lo : d2_lo;
            A_a[kt][2] = t_odd ? d1_hi : d0_hi;
            A_a[kt][3] = t_odd ? d3_hi : d2_hi;
        }
    }

    __syncthreads(); // gates MM2 reads of k_pass before h_part overwrite

    // === Phase E: per-chunk MM1 (Q @ h^T) + MM3 (A @ v_new) ===
    //
    // gamma_r0 / gamma_r1 in [0, 1] under the test's g distribution; underflow
    // to 0 is safe.
    const float gamma_r0 = exp2_approx(g_smem[row_g0] * kLog2E);
    const float gamma_r1 = exp2_approx(g_smem[row_g1] * kLog2E);

#pragma unroll 1
    for (int c = 0; c < N_CHUNKS; ++c) {
        const int d_chunk_off = c * D_CHUNK;

        // --- E.1: bf16 h_chunk[..., d_off:+D_CHUNK, :] -> float h_part[c] ---
        {
            const __nv_bfloat16* h_part_gmem =
                h_chunk_in + hc_base + (int64_t)d_chunk_off * kStateDim;
            issue_load_bf16_to_float_vec4<D_CHUNK, kStateDim, THREADS>(
                h_part_view, h_part_gmem, /*row_stride=*/kStateDim, tid);
        }
        __syncthreads();

        // --- E.2: MM1: D_frag = Q @ h^T (raw, gamma_r applied below) ---
        float D_frag[N_TILES_PER_CHUNK][4] = {};

#pragma unroll
        for (int kt = 0; kt < K_TILES; ++kt) {
            const int k_off  = kt * MMA_K;
            const int col_t0 = k_off + lane_t;
            const int col_t1 = col_t0 + 4;

            const float a0 = q_view.at(row_g0, col_t0);
            const float a1 = q_view.at(row_g1, col_t0);
            const float a2 = q_view.at(row_g0, col_t1);
            const float a3 = q_view.at(row_g1, col_t1);

#pragma unroll
            for (int nt = 0; nt < N_TILES_PER_CHUNK; ++nt) {
                const int n_off = nt * MMA_N;
                const int row_t = n_off + lane_g;
                const float b0  = h_part_view.at(row_t, col_t0);
                const float b1  = h_part_view.at(row_t, col_t1);
                mma_tf32(D_frag[nt][0], D_frag[nt][1], D_frag[nt][2], D_frag[nt][3], a0, a1, a2, a3,
                         b0, b1);
            }
        }

        // --- E.3: gamma_r row-scale (turns MM1 result into o_inter) ---
#pragma unroll
        for (int nt = 0; nt < N_TILES_PER_CHUNK; ++nt) {
            D_frag[nt][0] *= gamma_r0;
            D_frag[nt][1] *= gamma_r0;
            D_frag[nt][2] *= gamma_r1;
            D_frag[nt][3] *= gamma_r1;
        }

        __syncthreads(); // gates MM1 reads of h_part before v_part overwrite

        // --- E.4: bf16 v_new[..., d_off:+D_CHUNK] -> float v_part[c] ---
        {
            const __nv_bfloat16* v_part_gmem = v_new_in + vn_base + (int64_t)d_chunk_off;
            issue_load_bf16_to_float_vec4<BT, D_CHUNK, THREADS>(v_part_view, v_part_gmem,
                                                                v_row_stride, tid);
        }
        __syncthreads();

        // --- E.5: MM3: D_frag += A_a (regs) @ v_part ---
#pragma unroll
        for (int kt = 0; kt < K_TILES_BT; ++kt) {
            const int k_off = kt * MMA_K;
            const float a0  = A_a[kt][0];
            const float a1  = A_a[kt][1];
            const float a2  = A_a[kt][2];
            const float a3  = A_a[kt][3];

#pragma unroll
            for (int nt = 0; nt < N_TILES_PER_CHUNK; ++nt) {
                const int n_off  = nt * MMA_N;
                const int row_t0 = k_off + lane_t;
                const int row_t1 = k_off + lane_t + 4;
                const int col_g  = n_off + lane_g;
                const float b0   = v_part_view.at(row_t0, col_g);
                const float b1   = v_part_view.at(row_t1, col_g);
                mma_tf32(D_frag[nt][0], D_frag[nt][1], D_frag[nt][2], D_frag[nt][3], a0, a1, a2, a3,
                         b0, b1);
            }
        }

        // --- E.6: scale + float2 store -> attn_out ---
#pragma unroll
        for (int nt = 0; nt < N_TILES_PER_CHUNK; ++nt) {
            const int n_off    = nt * MMA_N;
            const int d_global = d_chunk_off + n_off + 2 * lane_t;
            const __nv_bfloat162 v0 =
                __floats2bfloat162_rn(scale * D_frag[nt][0], scale * D_frag[nt][1]);
            const __nv_bfloat162 v1 =
                __floats2bfloat162_rn(scale * D_frag[nt][2], scale * D_frag[nt][3]);
            store_vec(&attn_out[out_base + (int64_t)row_g0 * out_row_stride + d_global], v0);
            store_vec(&attn_out[out_base + (int64_t)row_g1 * out_row_stride + d_global], v1);
        }

        __syncthreads(); // gates v_part read before next h_part overwrite
    }
}

} // namespace ninfer::ops::detail::gated_delta_net::chunked::output
