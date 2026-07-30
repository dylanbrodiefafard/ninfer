#pragma once

#include "ops/common/mma.cuh"
#include "ops/linear_attention/gated_delta_net/chunked/common.cuh"

#include <cmath>

// Stage 3: chunk-sequential state passing.
//
// Math + I/O layouts: see the Gated DeltaNet chunked state_passing_config.
// Block / smem total: 8 warps (256 t), ~88 KB smem
// (W + UVD alias + K + snapshots + g).

namespace ninfer::ops::detail::gated_delta_net::chunked::state_passing {

using ninfer::ops::mma_tf32;
using ninfer::ops::mma_tf32_bits;
using ninfer::ops::ldmatrix_x4;
using ninfer::ops::ldmatrix_x2;
using ninfer::ops::smem_addr;
using ninfer::ops::exp2_approx;

static_assert(kChunkSize == 64, "stage_state_passing: kChunkSize must be 64");
static_assert(kStateDim == 128);

struct kernel_dims {
    static constexpr int N_STRIP_PER_BLOCK  = 32;
    static constexpr int D_STRIPS           = 4;
    static constexpr int DT_TILES_PER_BLOCK = 4;
    static constexpr int BT_SPLITS          = 2;
    static constexpr int N_WARPS            = DT_TILES_PER_BLOCK * BT_SPLITS; // 8
    static constexpr int THREADS            = N_WARPS * ninfer::ops::kWarpSize;
};

struct smem_layout {
    static constexpr int W_STRIDE       = kStateDim;
    static constexpr int W_LOAD_FLT     = BT * W_STRIDE;
    static constexpr int UVD_FLT        = BT * kernel_dims::N_STRIP_PER_BLOCK; // U-vd alias
    static constexpr int K_LOAD_ROWS    = BT;
    static constexpr int K_LOAD_FLT     = K_LOAD_ROWS * kStateDim;
    static constexpr int M_TILES_H_PW   = (kStateDim / kernel_dims::BT_SPLITS) / MMA_M;
    static constexpr int SNAP_K_ROWS    = MMA_M * M_TILES_H_PW;
    static constexpr int SNAP_FLT       = SNAP_K_ROWS * kernel_dims::N_STRIP_PER_BLOCK;
    static constexpr int M_TILES_H_GLOB = kStateDim / MMA_M;
    static constexpr int N_SNAP_ITERS   = M_TILES_H_GLOB / M_TILES_H_PW;
    // One buffer per sit so all warps can scatter their owned h_frag once at
    // chunk start (Phase A) and then read from any sit's buffer in the unified
    // matmul1 / coop_write paths without per-sit re-scatter+sync.
    static constexpr int N_SNAP_BUF = N_SNAP_ITERS;
    static constexpr int SMEM_FLOATS =
        W_LOAD_FLT + UVD_FLT + K_LOAD_FLT + SNAP_FLT * N_SNAP_BUF + BT;
};

// ---------------------------------------------------------------------------
// SnapView: custom swizzle for snap_smem (transposed h[d][k] layout).
// Different from SmemTile<64>'s default swizzle to keep the 4-d-row scatter
// conflict-free. Do not merge with the family-wide SmemTile.
//
// STRIDE=64 case (S=128 wide-block design): with row*64/4=row*16 mod 32, even
// rows start at bank 0, odd rows at bank 16. For the 4-d-row scatter (d takes
// 4 same-parity values per warp, e.g. {0,2,4,6} or {16,18,20,22}), all 4 rows
// land at the same bank without swizzle. The `(row & 7) << 2` swizzle splits
// d ∈ {0,2,4,6} into swz {0,8,16,24} → banks {0,8,16,24} (distinct). For odd
// d ∈ {1,3,5,7}, base bank=16 + swz {4,12,20,28} mod 32 = {20,28,4,12}, also
// distinct. The 8-d-row MM1 B-read (d ∈ {warp_d_local..warp_d_local+7}) maps
// to 8 distinct banks too.
// ---------------------------------------------------------------------------
struct SnapView {
    float* __restrict__ base;
    static constexpr int kStride = smem_layout::SNAP_K_ROWS;
    static_assert(kStride == 64);

    __device__ __forceinline__ int swz_xor(int row) const { return (row & 7) << 2; }

    __device__ __forceinline__ float& at(int row, int col) const {
        return base[row * kStride + (col ^ swz_xor(row))];
    }

    __device__ __forceinline__ float4& vec4_at(int row, int col) const {
        return *reinterpret_cast<float4*>(&base[row * kStride + (col ^ swz_xor(row))]);
    }
};

// __launch_bounds__: 256 threads + min_blocks=1 -> ~250 reg cap. With R2
// (ldmatrix.x4 for mm1 A) the compiler needs to preload multiple A frags
// concurrently to hide ldmatrix's data-ready latency (~80 cycles vs ~24 for
// LDS.32). The 128-reg cap from LB(2) blocked preload pipelining and made
// ldmatrix net-negative (R2 wall +31us at LB(2)); LB(1) lets ptxas use enough
// regs for in-flight overlap. Occupancy already only 17% in either mode, so
// dropping min_blocks costs nothing.
__launch_bounds__(kernel_dims::THREADS, 1) __global__ void state_passing_kernel(
    const __nv_bfloat16* __restrict__ W_in, const __nv_bfloat16* __restrict__ U_in,
    const __nv_bfloat16* __restrict__ k_in, const float* __restrict__ g_cumsum,
    const float* state_in, __nv_bfloat16* __restrict__ v_new, __nv_bfloat16* __restrict__ h_chunk,
    float* state_out, head_map qk_map, int chunks) {
    constexpr int N_STRIP_PER_BLOCK = kernel_dims::N_STRIP_PER_BLOCK;
    constexpr int D_STRIPS          = kernel_dims::D_STRIPS;
    constexpr int BT_SPLITS         = kernel_dims::BT_SPLITS;
    constexpr int THREADS_K         = kernel_dims::THREADS;
    constexpr int W_STRIDE          = smem_layout::W_STRIDE;

    constexpr int BT_PER_WARP    = BT / BT_SPLITS;
    constexpr int S_PER_WARP     = kStateDim / BT_SPLITS;
    constexpr int M_TILES_MM1_PW = BT_PER_WARP / MMA_M;
    constexpr int M_TILES_H_PW   = S_PER_WARP / MMA_M;
    constexpr int M_TILES_H_GLOB = kStateDim / MMA_M;
    constexpr int K_TILES_MM2    = BT / MMA_K;

    static_assert(M_TILES_H_PW >= 1, "S_PER_WARP must yield >= 1 M-tile per warp");
    static_assert(M_TILES_MM1_PW >= 1, "BT_PER_WARP must yield >= 1 M-tile per warp");
    static_assert(M_TILES_H_GLOB == BT_SPLITS * M_TILES_H_PW,
                  "BT_SPLITS partition of state dim must be exact");
    static_assert(W_STRIDE >= 16, "SmemTile<W_STRIDE> requires stride >= 16");

    constexpr int SNAP_K_ROWS           = smem_layout::SNAP_K_ROWS;
    constexpr int N_SNAP_ITERS          = smem_layout::N_SNAP_ITERS;
    constexpr int K_TILES_PER_SNAP_ITER = SNAP_K_ROWS / MMA_K;

    static_assert(N_SNAP_ITERS == BT_SPLITS, "design assumes one snap iter per s_idx");

    constexpr int W_LOAD_FLT  = smem_layout::W_LOAD_FLT;
    constexpr int UVD_FLT     = smem_layout::UVD_FLT;
    constexpr int K_LOAD_ROWS = smem_layout::K_LOAD_ROWS;
    constexpr int K_LOAD_FLT  = smem_layout::K_LOAD_FLT;
    constexpr int SNAP_FLT    = smem_layout::SNAP_FLT;
    constexpr int N_SNAP_BUF  = smem_layout::N_SNAP_BUF;

    // Smem partition. U and vd alias (`uvd_smem`) -- disjoint phases. snap is
    // per-sit (one buffer per sit) so the chunk-start unified scatter feeds
    // every sit's read without re-scatter.
    extern __shared__ float smem[];
    float* const W_smem    = smem;                              // W_LOAD_FLT
    float* const uvd_smem  = W_smem + W_LOAD_FLT;               // UVD_FLT (U & vd alias)
    float* const k_smem    = uvd_smem + UVD_FLT;                // K_LOAD_FLT
    float* const snap_smem = k_smem + K_LOAD_FLT;               // SNAP_FLT * N_SNAP_BUF
    float* const g_smem    = snap_smem + SNAP_FLT * N_SNAP_BUF; // BT

    SmemTile<W_STRIDE> W_view{W_smem};
    SmemTile<N_STRIP_PER_BLOCK> vd_view{uvd_smem};
    SmemTile<kStateDim> k_view{k_smem};
    SmemTile<N_STRIP_PER_BLOCK> U_view{uvd_smem};
    // One SnapView per sit so each owning warp scatters into a unique buffer
    // (Phase 2.a unified-scatter design). The array is sized on N_SNAP_BUF
    // (= N_SNAP_ITERS) rather than hard-coded.
    SnapView snap_views[N_SNAP_BUF];
#pragma unroll
    for (int b_ = 0; b_ < N_SNAP_BUF; ++b_) {
        snap_views[b_] = SnapView{snap_smem + b_ * SNAP_FLT};
    }

    // Block / lane indexing.
    //   grid.x = hd in [0, H_v*D_STRIPS).
    //   warp = dt_idx * BT_SPLITS + s_idx.
    const int tid    = static_cast<int>(threadIdx.x);
    const int lane   = tid & (kWarpSize - 1);
    const int warp   = tid / kWarpSize;
    const int lane_g = lane >> 2;
    const int lane_t = lane & 3;

    const int s_idx        = warp % BT_SPLITS;
    const int dt_idx       = warp / BT_SPLITS;
    const int warp_d_local = dt_idx * MMA_N;

    const int hd            = static_cast<int>(blockIdx.x);
    const int h_v           = hd / D_STRIPS;
    const int strip_idx     = hd - h_v * D_STRIPS;
    const int d_off         = strip_idx * N_STRIP_PER_BLOCK;
    const int warp_d_global = d_off + warp_d_local;
    const std::int64_t H_v  = qk_map.H_v;

    // === Phase 0: load state_in (AR-transposed) -> per-warp h_frag ===
    float h_frag[M_TILES_H_PW][4];
    {
        const int64_t st_base = static_cast<int64_t>(h_v) * kStateDim * kStateDim;
        const int row_off     = s_idx * S_PER_WARP;
#pragma unroll
        for (int m = 0; m < M_TILES_H_PW; ++m) {
            const int row_g0 = row_off + m * MMA_M + lane_g;
            const int row_g1 = row_g0 + 8;
            const int col_d0 = warp_d_global + 2 * lane_t;
            const int col_d1 = col_d0 + 1;
            h_frag[m][0] =
                load_ldg<float>(state_in + st_base + (int64_t)col_d0 * kStateDim + row_g0);
            h_frag[m][1] =
                load_ldg<float>(state_in + st_base + (int64_t)col_d1 * kStateDim + row_g0);
            h_frag[m][2] =
                load_ldg<float>(state_in + st_base + (int64_t)col_d0 * kStateDim + row_g1);
            h_frag[m][3] =
                load_ldg<float>(state_in + st_base + (int64_t)col_d1 * kStateDim + row_g1);
        }
    }

    // Cross-chunk staging. Chunk 0's W, U, and k are loaded before the loop;
    // subsequent chunks load W at the end of Phase B and U/k at the end of
    // Phase E.
    //
    // R7.1: All per-chunk gmem bases are precomputed as `*_block_base`
    // (chunk-0 value) + `*_chunk_stride` (delta), and advanced ADDITIVELY at
    // end of each iter -- saves 4 IMAD.WIDE per chunk that the old lambdas
    // ran for current/next W and current/next k bases.
    // R7.3: g_cumsum index splits into per-thread invariant base
    // (`g_thread_base`) + per-chunk delta (`g_cs_offset`), advanced by a
    // single int64 add per chunk per thread.
    const int64_t W_stride        = H_v * kStateDim;
    const int64_t k_stride        = static_cast<int64_t>(qk_map.H_qk) * kStateDim;
    const int64_t W_chunk_stride  = (int64_t)BT * W_stride;
    const int64_t k_chunk_stride  = (int64_t)BT * k_stride;
    const int64_t hc_chunk_stride = H_v * kStateDim * kStateDim;
    const int64_t vn_stride       = W_stride;
    const int64_t vn_chunk_stride = (int64_t)BT * vn_stride;
    const int64_t g_chunk_step    = (int64_t)BT * H_v;

    const int64_t W_block_base  = static_cast<int64_t>(h_v) * kStateDim;
    const int64_t k_block_base  = static_cast<int64_t>(qk_map.qk_head(h_v)) * kStateDim;
    const int64_t hc_block_base = static_cast<int64_t>(h_v) * kStateDim * kStateDim;
    const int64_t vn_block_base = static_cast<int64_t>(h_v) * kStateDim;
    const int64_t g_block_base  = h_v;
    const int64_t g_thread_base = g_block_base + (int64_t)tid * H_v;

    // W/U/k are bf16 in global workspace/boundary storage and are converted
    // into float shared memory before the math phases consume them.
    {
        issue_load_bf16_to_float_vec4<BT, W_STRIDE, THREADS_K>(W_view, W_in + W_block_base,
                                                               W_stride, tid);
        issue_load_bf16_to_float_vec4<BT, N_STRIP_PER_BLOCK, THREADS_K>(
            U_view, U_in + W_block_base + d_off, W_stride, tid);
        issue_load_bf16_to_float_vec4<K_LOAD_ROWS, kStateDim, THREADS_K>(
            k_view, k_in + k_block_base, k_stride, tid);
    }

    // === Main chunk loop ===
    int64_t W_base      = W_block_base;
    int64_t k_base      = k_block_base;
    int64_t hc_base     = hc_block_base;
    int64_t vn_base     = vn_block_base;
    int64_t g_cs_offset = 0;
    for (int chunk = 0; chunk < chunks; ++chunk) {
        const int64_t W_base_next = W_base + W_chunk_stride;
        const int64_t k_base_next = k_base + k_chunk_stride;

        // === Phase A: drain early group (W + U) + scatter h_frag to snap ===
        //
        // Phase 2.a unified-scatter: every warp scatters its owned h_frag into
        // snap_views[s_idx] BEFORE the Phase A drain sync. The single sync
        // below covers W/U/g_smem AND snap visibility, so Phase B no longer
        // needs per-sit scatter+sync (saves 2 syncs/chunk on the BT_SPLITS=2
        // fixed 128-state path).
        if (tid < BT) { g_smem[tid] = g_cumsum[g_thread_base + g_cs_offset]; }

        {
            SnapView snap = snap_views[s_idx];
#pragma unroll
            for (int m = 0; m < M_TILES_H_PW; ++m) {
                const int k_g0    = m * MMA_M + lane_g;
                const int k_g1    = k_g0 + 8;
                const int d0      = warp_d_local + 2 * lane_t;
                const int d1      = d0 + 1;
                snap.at(d0, k_g0) = h_frag[m][0];
                snap.at(d1, k_g0) = h_frag[m][1];
                snap.at(d0, k_g1) = h_frag[m][2];
                snap.at(d1, k_g1) = h_frag[m][3];
            }
        }

        __syncthreads(); // gates W+U+g_smem STS and scatter visibility

        // === Phase B: per-sit coop_write + matmul1 (no per-sit scatter sync) ===
        //
        // The unified scatter in Phase A populated snap_views[s_idx] for all
        // s_idx in one shot, so each sit just consumes snap[sit] without
        // re-scattering or syncing. We retain the per-sit interleave between
        // coop_write (LDS+STG to gmem) and mma (LDS) because that was the
        // crucial latency-hiding pattern in the per-sit baseline -- moving
        // all coop_writes ahead of all mmas costs ~15 us on L=4096.
        float vnew_frag[M_TILES_MM1_PW][4] = {};

#pragma unroll
        for (int sit = 0; sit < N_SNAP_ITERS; ++sit) {
            const int k_row_off = sit * SNAP_K_ROWS;
            SnapView snap       = snap_views[sit];

            // Coop float4 gmem write of h_chunk for this snap block.
            // No sync above: snap was populated by the Phase A
            // unified-scatter, and snap[sit] is read-only from now on.
            {
                constexpr int K_VEC_PER_D = SNAP_K_ROWS / 4;
                constexpr int N_VEC_SNAP  = SNAP_FLT / 4;
#pragma unroll
                for (int v = tid; v < N_VEC_SNAP; v += THREADS_K) {
                    const int d_local  = v / K_VEC_PER_D;
                    const int kvec     = v - d_local * K_VEC_PER_D;
                    const int k_off    = kvec * 4;
                    float4 val         = snap.vec4_at(d_local, k_off);
                    const int d_global = d_off + d_local;
                    __nv_bfloat16* out =
                        &h_chunk[hc_base + (int64_t)d_global * kStateDim + k_row_off + k_off];
                    store_vec(out, __floats2bfloat162_rn(val.x, val.y));
                    store_vec(out + 2, __floats2bfloat162_rn(val.z, val.w));
                }
            }

            // matmul1 inner: K_TILES_PER_SNAP_ITER mma K-tiles.
            //
            // A/B operand loads use ldmatrix.x4 / x2 .b16 to fold what was
            // 4+2 LDS.32 per mma into 1+1 issues going through the smem
            // crossbar. Per-lane address ownership (PTX docs Fig.70):
            //   A x4: lanes 0..7 source rows 0..7  of A-frag, col 0..3
            //         lanes 8..15        rows 0..7,           col 4..7
            //         lanes 16..23       rows 8..15,          col 0..3
            //         lanes 24..31       rows 8..15,          col 4..7
            //   B x2: lanes 0..7 source rows 0..7 of B-frag (mma-N row, =
            //         snap row d=warp_d_local..+7), col 0..3 of B-frag
            //         (mma-K col, = snap col snap_k..+3)
            //         lanes 8..15 source rows 0..7, col 4..7.
            // snap stores B as smem[N, K] which is exactly the no-trans
            // ldmatrix layout for tf32 B (b0 = snap[t/4, t%4]). W_view
            // stores A as smem[M, K] (no-trans tf32 A: a0 = W_view[t/4,
            // t%4]).
            // R3-minimal: for sit == s_idx (own-sit) the mm1 B operand
            // lives in THIS warp's h_frag. Use a per-lane register-shuffle
            // instead of ldmatrix.x2 from snap — skips MIO pipe entirely
            // for half of the B reads. snap STS is unchanged (other warps'
            // cross-sit reads still need it). Cross-sit (sit != s_idx)
            // stays on the ldmatrix.x2 path.
            //
            // Per-thread B-frag wants:
            //   b0 = h[k=snap_k+lane_t, d=warp_d_local+lane_g]
            //   b1 = h[k=snap_k+lane_t+4, d=warp_d_local+lane_g]
            // Our h_frag[m_h][i_h] for lane (lane_g_o, lane_t_o) stores:
            //   h[k=lane_g_o + m_h*16 + ((i_h>>1)&1)*8, d=warp_d_local + 2*lane_t_o + (i_h&1)]
            // Solving for own-sit:
            //   m_h         = kt >> 1
            //   i_h         = (kt&1)*2 + (lane_g & 1)
            //   src_b0_lane = (lane_t << 2) | (lane_g >> 1)
            //   src_b1_lane = src_b0_lane + 16
            // i_h picks {0,1} (kt even) or {2,3} (kt odd) of h_frag[m_h]
            // based on lane_g parity — selected branch-free via FSEL.
            // R2: matmul1 inner. A operand via ldmatrix.x4 (16x8 tf32 from
            // W_view), B operand via ldmatrix.x2 (8x8 tf32 from snap). Both
            // use the no-trans .b16 layout because W_view stores A as
            // smem[M, K] and snap stores B as smem[N, K] (mma B-frag's
            // per-lane b0 = source[t/4, t%4] matches the no-trans output).
            //
            // R3-minimal (own-sit shuffle) was tried: replace own-sit
            // ldmatrix.x2 with 4 __shfl_sync + 2 SEL per kt. Wall-time
            // 364us -> 392us (+28us, -7.7%). The shuffle EU is also
            // 1-issue/cycle, so 4 shfls cost more than 1 LDSM.x2 going
            // through MIO. Reverted; see analysis/root_cause.md.
            const int lane_in_8   = lane & 7;
            const int which_horiz = (lane >> 3) & 1;
            const int which_vert  = (lane >> 4) & 1;
#pragma unroll
            for (int kt = 0; kt < K_TILES_PER_SNAP_ITER; ++kt) {
                const int W_k_local = k_row_off + kt * MMA_K;
                const int snap_k    = kt * MMA_K;

                const int b_row       = warp_d_local + lane_in_8;
                const int b_col       = snap_k + which_horiz * 4;
                const unsigned b_addr = smem_addr(&snap.at(b_row, b_col));
                unsigned ub0, ub1;
                ldmatrix_x2(ub0, ub1, b_addr);

#pragma unroll
                for (int m_mm1 = 0; m_mm1 < M_TILES_MM1_PW; ++m_mm1) {
                    const int row_base    = s_idx * BT_PER_WARP + m_mm1 * MMA_M;
                    const int a_row       = row_base + which_vert * 8 + lane_in_8;
                    const int a_col       = W_k_local + which_horiz * 4;
                    const unsigned a_addr = smem_addr(&W_view.at(a_row, a_col));
                    unsigned ua0, ua1, ua2, ua3;
                    // The Gated DeltaNet TF32 fragment consumes ldmatrix registers in 0,2,1,3
                    // order.
                    ldmatrix_x4(ua0, ua2, ua1, ua3, a_addr);

                    mma_tf32_bits(vnew_frag[m_mm1][0], vnew_frag[m_mm1][1], vnew_frag[m_mm1][2],
                                  vnew_frag[m_mm1][3], ua0, ua1, ua2, ua3, ub0, ub1);
                }
            }
        }

        // Cross-chunk W load right after Phase B finishes consuming
        // W_smem. The __syncthreads gates the current chunk's mma reads before
        // this thread overwrites W_smem with the next chunk. Without it,
        // racecheck reports a write vs
        // f32_to_tf32-read race on W_smem at sm_120, which
        // surfaces as ~25% flaky v_new at S=128/L=256 (the chunk-end barrier
        // before next iter's Phase A only fences the *future* read of the
        // arriving W, not the *prior* read of the outgoing W).
        if (chunk + 1 < chunks) {
            __syncthreads();
            issue_load_bf16_to_float_vec4<BT, W_STRIDE, THREADS_K>(W_view, W_in + W_base_next,
                                                                   W_stride, tid);
        }

        // === Phase C: subtract U from U_smem (no global wait, U landed in A) ===
#pragma unroll
        for (int m_mm1 = 0; m_mm1 < M_TILES_MM1_PW; ++m_mm1) {
            const int row_g0    = s_idx * BT_PER_WARP + m_mm1 * MMA_M + lane_g;
            const int row_g1    = row_g0 + 8;
            const int col_d0    = warp_d_local + 2 * lane_t;
            const float2 u_top  = load_vec<float2>(&U_view.at(row_g0, col_d0));
            const float2 u_bot  = load_vec<float2>(&U_view.at(row_g1, col_d0));
            vnew_frag[m_mm1][0] = u_top.x - vnew_frag[m_mm1][0];
            vnew_frag[m_mm1][1] = u_top.y - vnew_frag[m_mm1][1];
            vnew_frag[m_mm1][2] = u_bot.x - vnew_frag[m_mm1][2];
            vnew_frag[m_mm1][3] = u_bot.y - vnew_frag[m_mm1][3];
        }

        // === Phase D: STG vnew (UNDECAYED), STS v_decay -> vd_view, scale h_frag ===
        const float g_C     = g_smem[BT - 1];
        const float gamma_C = exp2_approx(g_C * kLog2E);

#pragma unroll
        for (int m_mm1 = 0; m_mm1 < M_TILES_MM1_PW; ++m_mm1) {
            const int row_g0 = s_idx * BT_PER_WARP + m_mm1 * MMA_M + lane_g;
            const int row_g1 = row_g0 + 8;
            const int col_d0 = warp_d_global + 2 * lane_t;

            const float g_top   = g_smem[row_g0];
            const float g_bot   = g_smem[row_g1];
            const float dec_top = exp2_approx((g_C - g_top) * kLog2E);
            const float dec_bot = exp2_approx((g_C - g_bot) * kLog2E);

            const float v0 = vnew_frag[m_mm1][0];
            const float v1 = vnew_frag[m_mm1][1];
            const float v2 = vnew_frag[m_mm1][2];
            const float v3 = vnew_frag[m_mm1][3];

            const __nv_bfloat162 out0 = __floats2bfloat162_rn(v0, v1);
            const __nv_bfloat162 out1 = __floats2bfloat162_rn(v2, v3);
            store_vec(&v_new[vn_base + (int64_t)row_g0 * vn_stride + col_d0], out0);
            store_vec(&v_new[vn_base + (int64_t)row_g1 * vn_stride + col_d0], out1);

            const int row_g0_loc = s_idx * BT_PER_WARP + m_mm1 * MMA_M + lane_g;
            const int row_g1_loc = row_g0_loc + 8;
            const int col_d0_loc = warp_d_local + 2 * lane_t;
            store_vec(&vd_view.at(row_g0_loc, col_d0_loc), make_float2(v0 * dec_top, v1 * dec_top));
            store_vec(&vd_view.at(row_g1_loc, col_d0_loc), make_float2(v2 * dec_bot, v3 * dec_bot));
        }

#pragma unroll
        for (int m = 0; m < M_TILES_H_PW; ++m) {
#pragma unroll
            for (int e = 0; e < 4; ++e) { h_frag[m][e] *= gamma_C; }
        }

        // The barrier gates matmul2's reads of vd_view (Phase D write ->
        // Phase E read) and completes any next-chunk W staging.
        __syncthreads();

        // === Phase E: matmul2 over the full chunk of k ===
#pragma unroll
        for (int kt = 0; kt < K_TILES_MM2; ++kt) {
            const int k_off_local = kt * MMA_K;

            const int row_t0 = k_off_local + lane_t;
            const int row_t1 = row_t0 + 4;
            const int col_g  = warp_d_local + lane_g;
            const float b0   = vd_view.at(row_t0, col_g);
            const float b1   = vd_view.at(row_t1, col_g);

#pragma unroll
            for (int m = 0; m < M_TILES_H_PW; ++m) {
                const int row_a0    = k_off_local + lane_t;
                const int row_a1    = row_a0 + 4;
                const int col_a_top = s_idx * S_PER_WARP + m * MMA_M + lane_g;
                const int col_a_bot = col_a_top + 8;

                const float a0 = k_view.at(row_a0, col_a_top);
                const float a1 = k_view.at(row_a0, col_a_bot);
                const float a2 = k_view.at(row_a1, col_a_top);
                const float a3 = k_view.at(row_a1, col_a_bot);

                mma_tf32(h_frag[m][0], h_frag[m][1], h_frag[m][2], h_frag[m][3], a0, a1, a2, a3, b0,
                         b1);
            }
        }

        __syncthreads(); // before chunk-end loads overwrite k/U smem

        // Cross-chunk load: chunk t+1 U/k are converted synchronously into
        // shared memory.
        if (chunk + 1 < chunks) {
            issue_load_bf16_to_float_vec4<BT, N_STRIP_PER_BLOCK, THREADS_K>(
                U_view, U_in + W_base_next + d_off, W_stride, tid);
            issue_load_bf16_to_float_vec4<K_LOAD_ROWS, kStateDim, THREADS_K>(
                k_view, k_in + k_base_next, k_stride, tid);
        }

        // R7.1: advance loop-carried accumulators for next iter.
        W_base = W_base_next;
        k_base = k_base_next;
        hc_base += hc_chunk_stride;
        vn_base += vn_chunk_stride;
        g_cs_offset += g_chunk_step;
    }

    // === Phase Z: store h_frag -> state_out (AR-transposed) ===
    const int64_t st_base = static_cast<int64_t>(h_v) * kStateDim * kStateDim;

#pragma unroll
    for (int m = 0; m < M_TILES_H_PW; ++m) {
        const int k_g0 = s_idx * S_PER_WARP + m * MMA_M + lane_g;
        const int k_g1 = k_g0 + 8;
        const int d0   = warp_d_global + 2 * lane_t;
        const int d1   = d0 + 1;
        state_out[st_base + (int64_t)d0 * kStateDim + k_g0] = h_frag[m][0];
        state_out[st_base + (int64_t)d1 * kStateDim + k_g0] = h_frag[m][1];
        state_out[st_base + (int64_t)d0 * kStateDim + k_g1] = h_frag[m][2];
        state_out[st_base + (int64_t)d1 * kStateDim + k_g1] = h_frag[m][3];
    }
}

} // namespace ninfer::ops::detail::gated_delta_net::chunked::state_passing
