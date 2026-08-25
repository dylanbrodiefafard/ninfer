#pragma once

// (See top-of-file comment: TMA + mbarrier pipeline variant of the Sage3 NVFP4
// GQA attention kernels. WGMMA is NOT an SM120 instruction; the NVFP4 Tensor
// Core path on consumer Blackwell is the block-scaled mma.sync m16n8k64
// block-scaled mma.sync, which this file keeps. Only the K/V producer path is
// migrated to TMA.)

#include "ops/kernel/gqa_attention_kv_nvfp4.cuh"
#include "ops/kernel/gqa_attention_nvfp4s3_common.cuh"
#include "ops/kernel/gqa_attention_prefill_common.cuh"
#include "ninfer/ops/gqa_attention.h"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math_constants.h>

#include <cstdint>
#include <cstring>

namespace ninfer::ops {

// ---------------------------------------------------------------------------
// Device mbarrier + TMA helpers.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void nvfp4s3_mbarrier_init(std::uint64_t* barrier,
                                                    std::uint32_t arrivals) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" : : "r"(smem_addr(barrier)),
               "r"(arrivals)
               : "memory");
}

__device__ __forceinline__ void nvfp4s3_mbarrier_wait(std::uint64_t* barrier,
                                                         std::uint32_t phase,
                                                         std::uint32_t* stuck_out = nullptr,
                                                         std::uint32_t stuck_val = 0u) {
    if (stuck_out == nullptr) {
        constexpr std::uint32_t kSuspendTicks = 0x989680;
        asm volatile("{\n"
                     ".reg .pred done;\n"
                     "wait_loop:\n"
                     "mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1, %2;\n"
                     "@done bra wait_done;\n"
                     "bra wait_loop;\n"
                     "wait_done:\n"
                     "}\n"
                     :
                     : "r"(smem_addr(barrier)), "r"(phase), "r"(kSuspendTicks)
                     : "memory");
        return;
    }
    // Instrumented spin (stuck-probe build): same try_wait loop, but every 2^16 polls the
    // caller's identifier is published to a mapped-host word so a host watchdog can read
    // which (tile, head, block, which-wait) is spinning while the kernel is still alive.
    std::uint32_t polls = 0;
    for (;;) {
        std::uint32_t done = 0;
        asm volatile("{\n"
                     ".reg .pred p;\n"
                     "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
                     "@p mov.u32 %2, 1;\n"
                     "}\n"
                     : "=r"(done)
                     : "r"(smem_addr(barrier)), "r"(phase)
                     : "memory");
        if (done) { return; }
        if ((polls++ & 0xFFFFu) == 0u) { *stuck_out = stuck_val; }
    }
}

__device__ __forceinline__ void nvfp4s3_mbarrier_arrive(std::uint64_t* barrier) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" : : "r"(smem_addr(barrier))
               : "memory");
}

__device__ __forceinline__ void nvfp4s3_mbarrier_arrive_expect_tx(std::uint64_t* barrier,
                                                                   std::uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                 :
                : "r"(smem_addr(barrier)), "r"(bytes)
                : "memory");
}

// 4-D TMA tile load (rank-4 tile into smem, transaction on the mbarrier).
__device__ __forceinline__ void nvfp4s3_tma_load_4d(std::uint8_t* destination,
                                                        const CUtensorMap* descriptor,
                                                        std::int32_t coordinate0,
                                                        int coordinate1,
                                                        int coordinate2,
                                                        int coordinate3,
                                                        std::uint64_t* barrier) {
    asm volatile("cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
                 "[%0], [%1, {%2, %3, %4, %5}], [%6];"
                 :
                 : "r"(smem_addr(destination)),
                  "l"(descriptor),
                  "r"(coordinate0),
                  "r"(coordinate1),
                  "r"(coordinate2),
                  "r"(coordinate3),
                  "r"(smem_addr(barrier))
                 : "memory");
}

// 1-D bulk copy (bytes global -> smem, transaction on the mbarrier). For the
// 1024 B v_scale d-major plane (4 B inner is not a valid TMA tile).
__device__ __forceinline__ void nvfp4s3_tma_bulk_copy_1d(std::uint8_t* destination,
                                                              const std::uint8_t* source,
                                                              std::uint32_t bytes,
                                                              std::uint64_t* barrier) {
    asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
                 "[%0], [%1], %2, [%3];"
                 :
                 : "r"(smem_addr(destination)),
                  "l"(source),
                  "r"(bytes),
                  "r"(smem_addr(barrier))
                 : "memory");
}

// ---------------------------------------------------------------------------
// Host-side rank-4 descriptor builder for the paged KV planes. Templated on the
// per-tile key dimension (32 or 64).
// ---------------------------------------------------------------------------
// Host-built rank-4 TMA descriptors for the paged KV planes. Kernels must receive this
// struct BY DEVICE POINTER: passing a CUtensorMap struct by value as a kernel parameter
// traps on SM120 (RTX 5090, driver 580) with an illegal-memory-access in the TMA unit
// (verified 2026-08-23 via probe_tma2.cu: by-value -> IMA, by-device-pointer -> clean).
struct GqaNvfp4s3TmaDesc {
    CUtensorMap k_codes;  // 128B swizzle
    CUtensorMap v_codes;  // no swizzle
    CUtensorMap k_scale;  // 16 B inner, no swizzle

    template <typename Geometry, int KeysPerTile>
    static bool build(const std::uint8_t* k_plane, const std::uint8_t* v_plane,
                 const std::uint8_t* k_scale_plane, std::int32_t pages,
                 GqaNvfp4s3TmaDesc* out) {
        constexpr std::int32_t Kp = kPagedKVPageSize;  // 64
        const std::uint64_t shape_k[4] = {
            static_cast<std::uint64_t>(kGqaNvfp4CodeWidth),  // inner (code bytes)
            static_cast<std::uint64_t>(Kp),                  // keys per page
            static_cast<std::uint64_t>(Geometry::KVHeads),
            static_cast<std::uint64_t>(pages)};
        const std::uint64_t stride_k[3] = {
            static_cast<std::uint64_t>(kGqaNvfp4CodeWidth),
            static_cast<std::uint64_t>(kGqaNvfp4CodeWidth) * Kp,
            static_cast<std::uint64_t>(kGqaNvfp4CodeWidth) * Kp * Geometry::KVHeads};
        const std::uint32_t box_k[4] = {
            static_cast<std::uint32_t>(kGqaNvfp4CodeWidth),
            static_cast<std::uint32_t>(KeysPerTile),
            1,
            1};
        const std::uint32_t elem_c[4] = {1, 1, 1, 1};

        const std::uint64_t shape_s[4] = {
            static_cast<std::uint64_t>(kGqaNvfp4Groups),
            static_cast<std::uint64_t>(Kp),
            static_cast<std::uint64_t>(Geometry::KVHeads),
            static_cast<std::uint64_t>(pages)};
        const std::uint64_t stride_s[3] = {
            static_cast<std::uint64_t>(kGqaNvfp4Groups),
            static_cast<std::uint64_t>(kGqaNvfp4Groups) * Kp,
            static_cast<std::uint64_t>(kGqaNvfp4Groups) * Kp * Geometry::KVHeads};
        const std::uint32_t box_s[4] = {
            static_cast<std::uint32_t>(kGqaNvfp4Groups),
            static_cast<std::uint32_t>(KeysPerTile),
            1,
            1};

        if (cuTensorMapEncodeTiled(&out->k_codes, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
                                   const_cast<void*>(static_cast<const void*>(k_plane)),
                                   shape_k, stride_k, box_k, elem_c,
                                   CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                   CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                   CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS) {
            return false;
        }
        if (cuTensorMapEncodeTiled(&out->v_codes, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
                                   const_cast<void*>(static_cast<const void*>(v_plane)),
                                   shape_k, stride_k, box_k, elem_c,
                                   CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                                   CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                   CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS) {
            return false;
        }
        if (cuTensorMapEncodeTiled(&out->k_scale, CU_TENSOR_MAP_DATA_TYPE_UINT8, 4,
                                   const_cast<void*>(static_cast<const void*>(k_scale_plane)),
                                   shape_s, stride_s, box_s, elem_c,
                                   CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                                   CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                   CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS) {
            return false;
        }
        return true;
    }
};

// ---------------------------------------------------------------------------
// TMA + mbarrier pipeline prefill kernel. K/V code+scale tiles are staged into a
// Stages-deep circular smem pipeline; a dedicated thread-0 producer issues 4-D TMA
// loads (k_codes 128 B swizzled, v_codes un-swizzled, k_scale, and a 1-D bulk v_scale
// copy) and all 512 threads are the consumer. Q stays on the bf16-quantize path and the
// QK/PV block-scaled mma + epilogue are unchanged from the cp.async kernel.
// ---------------------------------------------------------------------------
template <int Stages>
struct alignas(128) GqaNvfp4s3TmaScratch {
    std::uint8_t q_codes[kGqaPrefillNvfp4s3QBytes];
    std::uint8_t q_scale[kGqaPrefillNvfp4s3QScaleBytes];
    std::uint8_t p4[kGqaPrefillNvfp4s3P4Bytes];
    std::uint8_t psf[kGqaPrefillNvfp4s3PsfBytes];
    // Double-buffered V transpose output when it fits (Stages<=2: 87.7 KB total), so the
    // next tile's V transpose cannot race the current tile's PV mma on a single v_t;
    // that removes one per-tile CTA barrier. Stages=3 stays single-buffered.
    static constexpr int kVtBuffs = (Stages <= 2) ? 2 : 1;
    alignas(128) std::uint8_t v_t[kVtBuffs][kGqaPrefillNvfp4s3VtBytes];
    float alpha_s[kGqaPrefillNvfp4s3Br];
    float final_l_s[kGqaPrefillNvfp4s3Br];
    alignas(128) std::uint8_t k_codes[Stages][kGqaPrefillNvfp4s3KBytes];
    alignas(128) std::uint8_t v_codes[Stages][kGqaPrefillNvfp4s3VBytes];
    std::uint8_t k_scale[Stages][kGqaPrefillNvfp4s3KScaleBytes];
    std::uint8_t v_scale[Stages][kGqaPrefillNvfp4s3VsfBytes];
    alignas(8) std::uint64_t full[Stages];
    alignas(8) std::uint64_t empty[Stages];
};

static_assert(sizeof(GqaNvfp4s3TmaScratch<2>) <= 101376);
static_assert(sizeof(GqaNvfp4s3TmaScratch<3>) <= 101376);

// TMA CU_TENSOR_MAP_SWIZZLE_128B physical byte mapping, measured on SM120 (rank-4 probe
// + bench pattern-inversion, both with and without static smem): 16 B chunk c of key
// row r lands in physical 16 B chunk c ^ ((T8 + r) & 7), where T8 is the CTA-smem
// window 128 B-row index of the tile: T8 = (align128(static smem) + offsetof(k_codes)
// + stage*stride) >> 7 & 7 (stage stride is 72 rows, 0 mod 8). This kernel has no
// static smem, so T8 = offsetof(k_codes) >> 7 & 7. This is NOT the cp.async-era
// gqa_nvfp4_swizzle_byte (16 B segments XOR row & 7) used for in-kernel-written
// buffers such as q_codes/p4.
__device__ __forceinline__ int gqa_nvfp4_tma_swizzle_byte(int row_index8, int row, int logical_byte) {
    const int chunk16 = logical_byte >> 4;
    const int off16   = logical_byte & 15;
    // NOTE: adds +1 internally; callers pass (T8 - 1) as row_index8.
    return 16 * (chunk16 ^ ((row_index8 + row + 1) & 7)) + off16;
}

template <typename Geometry, typename Metadata, int Stages>
__global__ __maxnreg__(128) void gqa_attention_prefill_nvfp4s3_tma_kernel(
    const __nv_bfloat16* __restrict__ q, const std::uint8_t* __restrict__ cache_v_scale,
    const float* __restrict__ k_mean, Metadata metadata,
    const std::int32_t* __restrict__ positions, float scale, __nv_bfloat16* __restrict__ out,
    std::int32_t width, float keep_frac, const GqaNvfp4s3TmaDesc* descriptors,
    GqaS3PrefillDump* dump, std::uint8_t* dbg_stage, std::uint32_t* dbg_regs,
    std::uint8_t* dbg_q, std::uint32_t* dbg_stuck) {
    constexpr int D             = kGqaPrefillHeadDim;
    constexpr int Br            = kGqaPrefillNvfp4s3Br;
    constexpr int Bc            = kGqaPrefillNvfp4s3Bc;
    constexpr int Groups        = kGqaPrefillNvfp4s3Groups;
    constexpr int CodeW         = kGqaPrefillNvfp4s3CodeW;
    constexpr int P4Row       = kGqaPrefillNvfp4s3P4RowBytes;
    constexpr int QKNt        = Bc / 8;
    constexpr int K64s        = kGqaNvfp4K64;
    constexpr int PVNtPerWarp = D / (kGqaPrefillNvfp4s3DConsumers * 8);
    constexpr int ProducerWarps = kGqaPrefillNvfp4s3RowTiles;
    constexpr int VWorkerWarps  = kGqaPrefillNvfp4s3Warps - ProducerWarps;
    constexpr int WorkerThreads = VWorkerWarps * 32;
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;
    constexpr std::uint32_t kTmaTileBytes =
        kGqaPrefillNvfp4s3KBytes + kGqaPrefillNvfp4s3VBytes +
        kGqaPrefillNvfp4s3KScaleBytes + kGqaPrefillNvfp4s3VsfBytes;
    static_assert(PVNtPerWarp == 16);

    extern __shared__ __align__(128) unsigned char shared_bytes[];
    GqaNvfp4s3TmaScratch<Stages>& smem = *reinterpret_cast<GqaNvfp4s3TmaScratch<Stages>*>(shared_bytes);
    std::uint8_t* q_codes = smem.q_codes;
    std::uint8_t* q_scale    = smem.q_scale;
    std::uint8_t* p4         = smem.p4;
    std::uint8_t* psf          = smem.psf;
    float* alpha_s          = smem.alpha_s;
    float* final_l_s        = smem.final_l_s;

    const int q_block = static_cast<int>(blockIdx.x);
    const int q_head  = static_cast<int>(blockIdx.y);
    const int tid     = static_cast<int>(threadIdx.x);
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int q0      = q_block * Br;
    const int kv_head = q_head / Geometry::GroupSize;
    const int tokens  = metadata.valid_tokens(width);
    if (q_head >= Geometry::QHeads || q0 >= width) { return; }
    if (q0 >= tokens) {
        gqa_prefill_zero_output_rows<Geometry>(out, q_head, q0, min(q0 + Br, width), tid,
                                                   kGqaPrefillNvfp4s3Threads);
        return;
    }
    const int base_pos              = positions[0];
    // op-dump side-band (mirrors the cp.async kernel): only q_block 0 writes.
    const bool do_dump = dump != nullptr && q_block == 0;
    const std::int64_t dht_base =
        do_dump ? static_cast<std::int64_t>(q_head) * dump->max_tiles : 0;
    const std::int32_t* block_table = metadata.block_table();

    const int tile_rows     = min(Br, tokens - q0);
    const int max_query_abs = base_pos + q0 + tile_rows - 1;
    // Causal bound (matches the cp.async kernel): only blocks up to the last query row are
    // ever scored; a `tokens/Bc + 1` count created a phantom tile whose block_table read
    // (and TMA load) ran off the end of the page table.
    const int key_blocks = max_query_abs / Bc + 1;

    for (int i = tid; i < Br * CodeW; i += kGqaPrefillNvfp4s3Threads) { smem.q_codes[i] = 0; }
    for (int i = tid; i < Br * Groups; i += kGqaPrefillNvfp4s3Threads) { smem.q_scale[i] = 0; }
    __syncthreads();

    for (int unit = tid; unit < Br * Groups; unit += kGqaPrefillNvfp4s3Threads) {
        const int row = unit / Groups;
        const int grp = unit - (unit / Groups) * Groups;
        std::uint32_t lo = 0, hi = 0;
        std::uint8_t sc = 0;
        if (row < tile_rows) {
            gqa_nvfp4_quantize_bf16x16(
                &q[gqa_prefill_q_index<Geometry>(q_head, grp * kGqaNvfp4Group, q0 + row)], lo, hi,
                sc);
        }
        const int phys = gqa_nvfp4_swizzle_byte(row, grp * 8);
        *reinterpret_cast<std::uint32_t*>(q_codes + row * CodeW + phys)     = lo;
        *reinterpret_cast<std::uint32_t*>(reinterpret_cast<std::uint8_t*>(q_codes + row * CodeW + phys) + 4) = hi;
        q_scale[row * Groups + grp]                                         = sc;
    }
        __syncthreads();
        // Debug: capture the fully-written Q codes + Q scale smem for host A/B.
        if (dbg_q != nullptr && q_head == 0 && q_block == 0) {
            for (int i = tid; i < kGqaPrefillNvfp4s3QBytes; i += kGqaPrefillNvfp4s3Threads)
                dbg_q[i] = smem.q_codes[i];
            for (int i = tid; i < kGqaPrefillNvfp4s3QScaleBytes; i += kGqaPrefillNvfp4s3Threads)
                dbg_q[kGqaPrefillNvfp4s3QBytes + i] = smem.q_scale[i];
        }

        // ------------------------------------------------------------------
        // TMA producer helper: issue one tile's four loads into a stage.
    // ------------------------------------------------------------------
    const GqaNvfp4s3TmaDesc* const tma_desc = descriptors;
    auto issue_tma_tile = [&](int kb, int stage) {
        const int physical_page = block_table[kb];
        nvfp4s3_mbarrier_arrive_expect_tx(&smem.full[stage], kTmaTileBytes);
        nvfp4s3_tma_load_4d(&smem.k_codes[stage][0], &tma_desc->k_codes, 0, 0, kv_head,
                            physical_page, &smem.full[stage]);
        nvfp4s3_tma_load_4d(&smem.v_codes[stage][0], &tma_desc->v_codes, 0, 0, kv_head,
                            physical_page, &smem.full[stage]);
        nvfp4s3_tma_load_4d(&smem.k_scale[stage][0], &tma_desc->k_scale, 0, 0, kv_head,
                            physical_page, &smem.full[stage]);
        nvfp4s3_tma_bulk_copy_1d(&smem.v_scale[stage][0],
                             cache_v_scale + gqa_s3_v_scale_index<Geometry>(physical_page, kv_head,
                                                                           0, 0),
                             kGqaPrefillNvfp4s3VsfBytes, &smem.full[stage]);
    };

    // Initialize the full/empty mbarriers, then pre-issue the first `Stages` tiles.
    if (tid == 0) {
        for (int s = 0; s < Stages; ++s) {
            nvfp4s3_mbarrier_init(&smem.full[s], 1);   // single producer arrival + tx
            nvfp4s3_mbarrier_init(&smem.empty[s], kGqaPrefillNvfp4s3Warps);
        }
    }
    __syncthreads();
    if (tid == 0) {
        for (int ki = 0; ki < Stages && ki < key_blocks; ++ki) {
            issue_tma_tile(ki, ki);  // stage == ki (ki < Stages)
        }
    }
    __syncthreads();

    const int gid           = lane >> 2;
    const int lid           = lane & 3;
    const int a_matrix      = lane >> 3;
    const int a_row_offset  = (lane & 7) + ((a_matrix & 1) << 3);
    const int a_column_byte = (a_matrix >> 1) * 16;
    const int b_row_offset  = lane & 7;
    const int b_column_byte = ((lane >> 3) & 1) * 16;
    const int sfa_row       = ((lane & 1) << 3) | (lane >> 2);
    const int sfb_row       = lane >> 2;

    float acc[PVNtPerWarp][4];
#pragma unroll
    for (int n = 0; n < PVNtPerWarp; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[n][i] = 0.0f; }
    }
    float running_m0     = -CUDART_INF_F;
    float running_m1     = -CUDART_INF_F;
    float running_l0     = 0.0f;
    float running_l1     = 0.0f;
    const float scale_l2 = scale * Log2E;

    // This standalone probe kernel is exact-path only (keep-all tiles): no proxy/tile-skip
    // state, so no static smem (the cp.async kernel's ~21 KB of static proxy arrays are
    // dropped), which lets the Stages=3 scratch (93824 B) fit the 101376 B opt-in cap.
    (void)k_mean;
    (void)keep_frac;
    __syncthreads();

    // ------------------------------------------------------------------
    // Main loop over the kept tiles (keep_list order). TMA pipeline: thread 0 is the
    // producer (issues tile ki+Stages into stage ki%Stages, running Stages ahead); all
    // 512 threads are the consumer. v2 protocol: the mbarrier full-wait is spun by
    // thread 0 alone (a 512-thread spin only added branch/L1 pressure), consumers are
    // released with one CTA barrier; the stage's empty barrier is released right after
    // the QK+V-consume barrier (before the PV mma) so the producer refills a stage one
    // tile earlier; double-buffered v_t drops the second per-tile barrier.
    // ------------------------------------------------------------------
    const int n_tiles = key_blocks;
    if (do_dump && warp == 0 && lane == 0) {
        const int ncopy = min(n_tiles, dump->max_tiles);
        for (int i = 0; i < ncopy; ++i) {
            dump->keep_list[static_cast<std::int64_t>(q_head) * dump->max_tiles + i] = i;
        }
        dump->tile_count[q_head] = n_tiles;
    }
    for (int ki = 0; ki < n_tiles; ++ki) {
        const int stage    = ki % Stages;
        const std::uint32_t full_phase = (static_cast<int>(ki) / Stages) & 1;
        // Consumer: all threads spin the mbarrier full-wait (v3: no extra CTA barrier —
        // the per-tile barrier count stays at one, matching the cp.async kernel).
        nvfp4s3_mbarrier_wait(&smem.full[stage], full_phase, dbg_stuck,
                              (static_cast<std::uint32_t>(ki) | (static_cast<std::uint32_t>(q_head) << 20) |
                               (static_cast<std::uint32_t>(q_block) << 25)));

        const int kb = ki;
        const int k0 = kb * Bc;
        const std::uint8_t* kc  = smem.k_codes[stage];
        // TMA 128 B swizzle XOR row for key row r: (T8 + r) & 7, T8 = window-row of the
        // tile (align128(static smem) + scratch offset). No static smem here, and the
        // stage stride (72 rows) is 0 mod 8, so T8 is the compile-time scratch row.
        constexpr int kT8    =
            static_cast<int>(offsetof(GqaNvfp4s3TmaScratch<Stages>, k_codes) >> 7) & 7;
        const int kbase8    = kT8 - 1;  // helper adds +1 internally
        const std::uint8_t* vc  = smem.v_codes[stage];
        const std::uint8_t* ksc = smem.k_scale[stage];
        const std::uint8_t* vsf = smem.v_scale[stage];
        std::uint8_t* vt        = smem.v_t[GqaNvfp4s3TmaScratch<Stages>::kVtBuffs > 1 ? (ki & 1)
                                                                                   : 0];

        // Debug side-band: dump the raw staged K codes + K scale (pre-consumer) for
        // the first 4 tiles of head 0, so the host can verify TMA placement.
        if (dbg_stage != nullptr && q_head == 0 && ki < 4) {
            const std::size_t stride = static_cast<std::size_t>(Bc) * 128 + kGqaPrefillNvfp4s3KScaleBytes;
            std::uint8_t* dst  = dbg_stage + static_cast<std::size_t>(ki) * stride;
            for (int i = tid; i < Bc * 128; i += kGqaPrefillNvfp4s3Threads) dst[i] = kc[i];
            std::uint8_t* dsts = dst + Bc * 128;
            for (int i = tid; i < kGqaPrefillNvfp4s3KScaleBytes; i += kGqaPrefillNvfp4s3Threads)
                dsts[i] = ksc[i];
        }

        if (warp < ProducerWarps) {
            const int row_base = warp * 16;
            float score[QKNt][4];
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0f;
            }

#pragma unroll
            for (int k64 = 0; k64 < K64s; ++k64) {
                const int row           = row_base + a_row_offset;
                const int logical_byte  = k64 * 32 + a_column_byte;
                const int physical_byte = gqa_nvfp4_swizzle_byte(row, logical_byte);
                unsigned af[4];
                ldmatrix_x4(af[0], af[1], af[2], af[3],
                             smem_addr(q_codes + row * CodeW + physical_byte));
                const int scale_row = row_base + sfa_row;
                unsigned sfa = 0;
                if ((lane & 2) == 0) {
                    sfa = *reinterpret_cast<const unsigned*>(
                        &q_scale[scale_row * Groups + k64 * 4]);
                }
#pragma unroll
                for (int nt = 0; nt < QKNt; ++nt) {
                    const int brow       = nt * 8 + b_row_offset;
                    const int b_logical  = k64 * 32 + b_column_byte;
                    const int b_physical =
                        gqa_nvfp4_tma_swizzle_byte(kbase8, brow, b_logical);
                    unsigned bf[2];
                    ldmatrix_x2(bf[0], bf[1], smem_addr(kc + brow * CodeW + b_physical));
                    const int b_scale_row = nt * 8 + sfb_row;
                    unsigned sfb          = 0;
                    if ((lane & 3) == 0) {
                        sfb = *reinterpret_cast<const unsigned*>(
                            &ksc[b_scale_row * Groups + k64 * 4]);
                    }
                    mma_nvfp4_e4m3(score[nt][0], score[nt][1], score[nt][2], score[nt][3], af[0],
                                   af[1], af[2], af[3], bf[0], bf[1], sfa, sfb);
                    if (dbg_regs != nullptr && warp == 0 && k64 == 0 && nt == 0 &&
                        q_head == 0 && q_block == 0) {
                        dbg_regs[lane * 8 + 0] = af[0];
                        dbg_regs[lane * 8 + 1] = af[1];
                        dbg_regs[lane * 8 + 2] = af[2];
                        dbg_regs[lane * 8 + 3] = af[3];
                        dbg_regs[lane * 8 + 4] = bf[0];
                        dbg_regs[lane * 8 + 5] = bf[1];
                        dbg_regs[lane * 8 + 6] = sfa;
                        dbg_regs[lane * 8 + 7] = sfb;
                        dbg_regs[256] = static_cast<std::uint32_t>(kbase8);
                        if (lane < 16)
                            dbg_regs[257 + lane] =
                                static_cast<std::uint32_t>(brow * CodeW + b_physical);
                    }
                }
            }

            const int row0             = row_base + gid;
            const int row1             = row0 + 8;
            const int qabs0            = row0 < tile_rows ? base_pos + q0 + row0 : -1;
            const int qabs1            = (row1 < tile_rows) ? base_pos + q0 + row1 : -1;
            const bool full_score_tile = q0 + Br <= tokens && k0 + Bc - 1 <= base_pos + q0;
            float bm0_blk[4], bm1_blk[4];
#pragma unroll
            for (int nb = 0; nb < 4; ++nb) {
                float m0 = -CUDART_INF_F;
                float m1 = -CUDART_INF_F;
#pragma unroll
                for (int nt = 2 * nb; nt < 2 * nb + 2; ++nt) {
                    if (!full_score_tile) {
                        const int key0 = k0 + nt * 8 + 2 * lid;
                        const int key1 = key0 + 1;
                        if (key0 > qabs0) { score[nt][0] = -CUDART_INF_F; }
                        if (key1 > qabs0) { score[nt][1] = -CUDART_INF_F; }
                        if (key0 > qabs1) { score[nt][2] = -CUDART_INF_F; }
                        if (key1 > qabs1) { score[nt][3] = -CUDART_INF_F; }
                    }
                    m0 = fmaxf(m0, fmaxf(score[nt][0], score[nt][1]));
                    m1 = fmaxf(m1, fmaxf(score[nt][2], score[nt][3]));
                }
                bm0_blk[nb] = warp_max<4>(m0, FullMask);
                bm1_blk[nb] = warp_max<4>(m1, FullMask);
            }
            const std::int64_t dht = dht_base + ki;
            float bm0 = -CUDART_INF_F;
            float bm1 = -CUDART_INF_F;
#pragma unroll
            for (int nb = 0; nb < 4; ++nb) {
                bm0 = fmaxf(bm0, bm0_blk[nb]);
                bm1 = fmaxf(bm1, bm1_blk[nb]);
            }
            if (do_dump) {
                float* ds = &dump->score[(dht) * static_cast<std::int64_t>(Br) * Bc];
                for (int nt = 0; nt < QKNt; ++nt) {
                    const int key_l = nt * 8 + 2 * lid;
                    ds[static_cast<std::int64_t>(row0) * Bc + key_l]       = score[nt][0];
                    ds[static_cast<std::int64_t>(row0) * Bc + key_l + 1]   = score[nt][1];
                    ds[static_cast<std::int64_t>(row1) * Bc + key_l]       = score[nt][2];
                    ds[static_cast<std::int64_t>(row1) * Bc + key_l + 1]   = score[nt][3];
                }
            }

            const float nm0        = fmaxf(running_m0, bm0);
            const float nm1        = fmaxf(running_m1, bm1);
            const float nm0_scaled = nm0 * scale_l2;
            const float nm1_scaled = nm1 * scale_l2;
            const float alpha0     = running_m0 == -CUDART_INF_F
                                         ? 0.0f
                                         : exp2_approx(__fmaf_rn(running_m0, scale_l2, -nm0_scaled));
            const float alpha1     = running_m1 == -CUDART_INF_F
                                         ? 0.0f
                                         : exp2_approx(__fmaf_rn(running_m1, scale_l2, -nm1_scaled));
            float bl0              = 0.0f;
            float bl1              = 0.0f;
#pragma unroll
            for (int nb = 0; nb < 4; ++nb) {
                const float sf0f = bm0_blk[nb] == -CUDART_INF_F
                                        ? 0.0f
                                        : exp2_approx(__fmaf_rn(bm0_blk[nb], scale_l2,
                                                                 -nm0_scaled + kGqaS3SfLog2));
                const float sf1f = bm1_blk[nb] == -CUDART_INF_F
                                        ? 0.0f
                                        : exp2_approx(__fmaf_rn(bm1_blk[nb], scale_l2,
                                                                 -nm1_scaled + kGqaS3SfLog2));
                const std::uint8_t sc0 =
                    sf0f == 0.0f ? 0 : __nv_cvt_float_to_fp8(sf0f, __NV_SATFINITE, __NV_E4M3);
                const std::uint8_t sc1 =
                    sf1f == 0.0f ? 0 : __nv_cvt_float_to_fp8(sf1f, __NV_SATFINITE, __NV_E4M3);
                const float dec0 = detail::decode_nvfp4_e4m3(sc0);
                const float dec1 = detail::decode_nvfp4_e4m3(sc1);
                const float amp  = kGqaS3PvAmpLog2;
                const float pa0  = score[2 * nb][0] > -CUDART_INF_F
                                        ? exp2_approx(__fmaf_rn(score[2 * nb][0], scale_l2,
                                                                -nm0_scaled + amp))
                                        : 0.0f;
                const float pa1  = score[2 * nb][1] > -CUDART_INF_F
                                        ? exp2_approx(__fmaf_rn(score[2 * nb][1], scale_l2,
                                                                -nm0_scaled + amp))
                                        : 0.0f;
                const float pa2  = score[2 * nb + 1][0] > -CUDART_INF_F
                                        ? exp2_approx(__fmaf_rn(score[2 * nb + 1][0], scale_l2,
                                                                -nm0_scaled + amp))
                                        : 0.0f;
                const float pa3  = score[2 * nb + 1][1] > -CUDART_INF_F
                                        ? exp2_approx(__fmaf_rn(score[2 * nb + 1][1], scale_l2,
                                                                -nm0_scaled + amp))
                                        : 0.0f;
                const float pb0  = score[2 * nb][2] > -CUDART_INF_F
                                        ? exp2_approx(__fmaf_rn(score[2 * nb][2], scale_l2,
                                                                -nm1_scaled + amp))
                                        : 0.0f;
                const float pb1  = score[2 * nb][3] > -CUDART_INF_F
                                        ? exp2_approx(__fmaf_rn(score[2 * nb][3], scale_l2,
                                                                -nm1_scaled + amp))
                                        : 0.0f;
                const float pb2  = score[2 * nb + 1][2] > -CUDART_INF_F
                                        ? exp2_approx(__fmaf_rn(score[2 * nb + 1][2], scale_l2,
                                                                -nm1_scaled + amp))
                                        : 0.0f;
                const float pb3  = score[2 * nb + 1][3] > -CUDART_INF_F
                                        ? exp2_approx(__fmaf_rn(score[2 * nb + 1][3], scale_l2,
                                                                -nm1_scaled + amp))
                                        : 0.0f;
                bl0 += pa0 + pa1 + pa2 + pa3;
                bl1 += pb0 + pb1 + pb2 + pb3;
                const float qa0 = dec0 > 0.0f ? __fdiv_rn(pa0, dec0) : 0.0f;
                const float qa1 = dec0 > 0.0f ? __fdiv_rn(pa1, dec0) : 0.0f;
                const float qa2 = dec0 > 0.0f ? __fdiv_rn(pa2, dec0) : 0.0f;
                const float qa3 = dec0 > 0.0f ? __fdiv_rn(pa3, dec0) : 0.0f;
                const float qb0 = dec1 > 0.0f ? __fdiv_rn(pb0, dec1) : 0.0f;
                const float qb1 = dec1 > 0.0f ? __fdiv_rn(pb1, dec1) : 0.0f;
                const float qb2 = dec1 > 0.0f ? __fdiv_rn(pb2, dec1) : 0.0f;
                const float qb3 = dec1 > 0.0f ? __fdiv_rn(pb3, dec1) : 0.0f;
                p4[row0 * P4Row + nb * 8 + lid] = gqa_s3_cvt_e2m1x2(qa0, qa1);
                p4[row0 * P4Row + nb * 8 + 4 + lid] = gqa_s3_cvt_e2m1x2(qa2, qa3);
                p4[row1 * P4Row + nb * 8 + lid] = gqa_s3_cvt_e2m1x2(qb0, qb1);
                p4[row1 * P4Row + nb * 8 + 4 + lid] = gqa_s3_cvt_e2m1x2(qb2, qb3);
                if (lid == 0) {
                    psf[row0 * 4 + nb] = sc0;
                    psf[row1 * 4 + nb] = sc1;
                }
            }
            if (do_dump && lane < 16) {
                const int drow         = row_base + lane;
                std::uint8_t* pc = &dump->p_code[(dht) * static_cast<std::int64_t>(Br) * Bc +
                                                  static_cast<std::int64_t>(drow) * Bc];
                const std::uint8_t* prow = p4 + drow * P4Row;
                for (int j = 0; j < 32; ++j) {
                    const std::uint8_t b = prow[j];
                    pc[2 * j]   = b & 0x0fu;
                    pc[2 * j + 1] = (b >> 4) & 0x0fu;
                }
                std::memcpy(&dump->psf[(dht) * static_cast<std::int64_t>(Br) * 4 +
                                         static_cast<std::int64_t>(drow) * 4],
                             psf + drow * 4, 4);
            }
            bl0        = warp_sum<4>(bl0, FullMask);
            bl1        = warp_sum<4>(bl1, FullMask);
            running_l0 = __fmaf_rn(running_l0, alpha0, bl0);
            running_l1 = __fmaf_rn(running_l1, alpha1, bl1);
            running_m0 = nm0;
            running_m1 = nm1;
            if (do_dump && lid == 0) {
                const std::int64_t ml = (dht_base + ki) * Br;
                dump->m[ml + row0] = running_m0;
                dump->l[ml + row0] = running_l0;
                dump->m[ml + row1] = running_m1;
                dump->l[ml + row1] = running_l1;
            }
            if (lid == 0) {
                alpha_s[row0] = alpha0;
                alpha_s[row1] = alpha1;
            }
        } else if (warp < ProducerWarps + VWorkerWarps) {
            const int worker_tid = tid - ProducerWarps * 32;
            for (int d = worker_tid; d < D; d += WorkerThreads) {
                const int dp = d >> 1;
                const int sh = (d & 1) * 4;
#pragma unroll 1
                for (int kp = 0; kp < Bc / 2; ++kp) {
                    const std::uint8_t b0 = vc[2 * kp * CodeW + dp];
                    const std::uint8_t b1 = vc[(2 * kp + 1) * CodeW + dp];
                    vt[d * P4Row + kp]   = (static_cast<std::uint8_t>(((b1 >> sh) & 0x0Fu) << 4) |
                                            static_cast<std::uint8_t>((b0 >> sh) & 0x0Fu));
                }
            }
        }
        __syncthreads();
        // Stage smem is now fully consumed (QK read k_codes, V workers read v_codes into
        // vt), so release this stage early: warp-leader empty arrivals let the producer
        // refill it before the PV mma below.
        if ((tid & 31) == 0) {
            nvfp4s3_mbarrier_arrive(&smem.empty[stage]);
        }
        if (tid == 0) {
            const int nki = ki + Stages;
            if (nki < n_tiles) {
                const int nstage    = ki % Stages;  // == (nki % Stages)
                const std::uint32_t nempty_phase = 1U ^ ((nki / Stages) & 1U);
                nvfp4s3_mbarrier_wait(&smem.empty[nstage], nempty_phase, dbg_stuck,
                                       0x40000000u | (static_cast<std::uint32_t>(nki) |
                                                       (static_cast<std::uint32_t>(q_head) << 20) |
                                                       (static_cast<std::uint32_t>(q_block) << 25)));
                issue_tma_tile(nki, nstage);
            }
        }

        if (do_dump && tid < D) {
            std::memcpy(&dump->v_scale[(dht_base + ki) * static_cast<std::int64_t>(D) * 4 +
                                         static_cast<std::int64_t>(tid) * 4],
                         vsf + tid * 4, 4);
        }

        const int row_tile = warp % kGqaPrefillNvfp4s3RowTiles;
        const int d_slice  = warp / kGqaPrefillNvfp4s3RowTiles;
        const int row_base = row_tile * 16;
        const float alpha0 = alpha_s[row_base + gid];
        const float alpha1 = alpha_s[row_base + gid + 8];
#pragma unroll
        for (int n = 0; n < PVNtPerWarp; ++n) {
            acc[n][0] *= alpha0;
            acc[n][1] *= alpha0;
            acc[n][2] *= alpha1;
            acc[n][3] *= alpha1;
        }

        unsigned pf[4];
        ldmatrix_x4(pf[0], pf[1], pf[2], pf[3],
                    smem_addr(p4 + (row_base + a_row_offset) * P4Row + a_column_byte));
        unsigned sfa = 0;
        if ((lane & 2) == 0) {
            sfa = *reinterpret_cast<const unsigned*>(&psf[(row_base + sfa_row) * 4]);
        }
#pragma unroll
        for (int n = 0; n < PVNtPerWarp; ++n) {
            const int global_n = d_slice * PVNtPerWarp + n;
            const int vrow     = global_n * 8 + b_row_offset;
            unsigned vf[2];
            ldmatrix_x2(vf[0], vf[1], smem_addr(vt + vrow * P4Row + b_column_byte));
            const int vsf_row  = global_n * 8 + sfb_row;
            unsigned sfb       = 0;
            if ((lane & 3) == 0) {
                sfb = *reinterpret_cast<const unsigned*>(&vsf[vsf_row * 4]);
            }
            mma_nvfp4_e4m3(acc[n][0], acc[n][1], acc[n][2], acc[n][3], pf[0], pf[1], pf[2],
                            pf[3], vf[0], vf[1], sfa, sfb);
        }
        if (do_dump) {
            float* da = &dump->acc[(dht_base + ki) * static_cast<std::int64_t>(Br) * D];
            const int arow0 = row_base + gid;
            const int arow1 = arow0 + 8;
            for (int n = 0; n < PVNtPerWarp; ++n) {
                const int d0 = (d_slice * PVNtPerWarp + n) * 8 + 2 * lid;
                da[static_cast<std::int64_t>(arow0) * D + d0]       = acc[n][0];
                da[static_cast<std::int64_t>(arow0) * D + d0 + 1]   = acc[n][1];
                da[static_cast<std::int64_t>(arow1) * D + d0]       = acc[n][2];
                da[static_cast<std::int64_t>(arow1) * D + d0 + 1]   = acc[n][3];
            }
            if (warp == ProducerWarps) {
                const std::int64_t vt_base =
                    (dht_base + ki) * static_cast<std::int64_t>(D) * 32 +
                    static_cast<std::int64_t>(lane * 8) * 32;
                for (int r = 0; r < 8; ++r) {
                    std::memcpy(&dump->v_t[vt_base + r * 32], vt + (lane * 8 + r) * P4Row, 32);
                }
            }
        }

        // Single-buffer v_t (Stages=3 only): the next tile's V transpose must not
        // clobber v_t while this tile's PV mma is still reading it, so keep the tail
        // barrier in that config only.
        if (GqaNvfp4s3TmaScratch<Stages>::kVtBuffs == 1) {
            __syncthreads();
        }
    }

    if (warp < ProducerWarps && lid == 0) {
        const int row0  = warp * 16 + gid;
        const int row1  = row0 + 8;
        final_l_s[row0] = running_l0;
        final_l_s[row1] = running_l1;
    }
    __syncthreads();

    const int row_tile = warp % kGqaPrefillNvfp4s3RowTiles;
    const int d_slice  = warp / kGqaPrefillNvfp4s3RowTiles;
    const int row_base = row_tile * 16;
    const int row0     = row_base + gid;
    const int row1     = row0 + 8;
    const float inv_l0 = final_l_s[row0] > 0.0f ? __frcp_rn(final_l_s[row0]) : 0.0f;
    const float inv_l1 = final_l_s[row1] > 0.0f ? __frcp_rn(final_l_s[row1]) : 0.0f;
#pragma unroll
    for (int n = 0; n < PVNtPerWarp; ++n) {
        const int d0 = (d_slice * PVNtPerWarp + n) * 8 + 2 * lid;
        if (row0 < tile_rows) {
            *reinterpret_cast<unsigned*>(
                &out[gqa_prefill_q_index<Geometry>(q_head, d0, q0 + row0)]) =
                pack_bf16x2(acc[n][0] * inv_l0, acc[n][1] * inv_l0);
        }
        if (row1 < tile_rows) {
            *reinterpret_cast<unsigned*>(
                &out[gqa_prefill_q_index<Geometry>(q_head, d0, q0 + row1)]) =
                pack_bf16x2(acc[n][2] * inv_l1, acc[n][3] * inv_l1);
        }
    }
}

}  // namespace ninfer::ops