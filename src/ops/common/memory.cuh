#pragma once

#include <cuda_pipeline.h>
#include <cuda_runtime.h>

namespace ninfer::ops {

// L2 cache-policy selectors for the cp.async weight/activation fills in the NVFP4 GEMV/GEMM
// kernels (see cp_async / cp_async_zfill below).
//
//   ca — plain cp.async (no L2 hint; 4/8/16-byte fills).
//   cg — cp.async.cg (16-byte fill; the default for streaming weight fills).
//   EvictFirst — cp.async.cg plus an L2::evict_first cache_hint (sm_90+ / sm_120).
//
// WHEN TO USE EvictFirst, and why: it marks the L2 lines filled by the weight stream so the
// hardware evicts them first, keeping that streaming traffic OUT of the L2 working set. That
// is the right call for ONE-SHOT weight fills: in the small-T GEMV (T<=4, the MTP verify /
// decode round) each weight byte is read exactly once (a single M-block, no cross-tile
// re-read), so retaining it in L2 is worthless — and actively harmful, because the 10s-100s
// of MB of streamed weights would otherwise displace the small working set that a *following*
// consumer re-reads in the same round (SwiGLU's BF16 mid for MLP-down, or attention KV / GDN
// state). Marking the stream evict-first frees that L2 for the data that is actually reused.
// Shipped evidence: the SwiGLU T<=4 path (nvfp4_linear_swiglu_w4a4.cu) earns a -15%
// (swiglu+down) pair win from exactly this hint.
//
// WHEN NOT TO USE it (the con, why every call site scopes it to T<=4): for large-T (prefill)
// GEMM the same weight tile is re-read across many M-blocks, so L2 retention of the weights
// IS valuable and evict-first REGRESSES. Every site therefore applies EvictFirst only when
// `tokens <= 4` and falls back to Cache::cg otherwise.
//
// Implementation note: the evict-first fill routes through cp_async_evict_first_16_noinline
// (memory_evict.cu); inlining createpolicy.fractional + the cache_hint directly into the big
// MMA kernels makes ptxas 13.1 on sm_120a emit an illegal LDGSTS-desc form.
enum class Cache {
    ca,
    cg,
    EvictFirst,
};

template <class V, class T>
__device__ __forceinline__ V load_vec(const T* ptr) {
    static_assert(sizeof(V) == 1 || sizeof(V) == 2 || sizeof(V) == 4 || sizeof(V) == 8 ||
                  sizeof(V) == 16);
    return *reinterpret_cast<const V*>(ptr);
}

template <class V, class T>
__device__ __forceinline__ V load_ldg(const T* ptr) {
    static_assert(sizeof(V) == 1 || sizeof(V) == 2 || sizeof(V) == 4 || sizeof(V) == 8 ||
                  sizeof(V) == 16);
    return __ldg(reinterpret_cast<const V*>(ptr));
}

template <class T, class V>
__device__ __forceinline__ void store_vec(T* ptr, V value) {
    static_assert(sizeof(V) == 1 || sizeof(V) == 2 || sizeof(V) == 4 || sizeof(V) == 8 ||
                  sizeof(V) == 16);
    *reinterpret_cast<V*>(ptr) = value;
}

__device__ __forceinline__ unsigned smem_addr(const void* ptr) {
    return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ unsigned long long l2_evict_first_policy() {
    unsigned long long pol;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
    return pol;
}

__device__ __forceinline__ void cp_async_evict_first_16(void* smem_dst, const void* gmem_src,
                                                       unsigned long long pol) {
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;\n"
                 :
                 : "r"(smem_addr(smem_dst)), "l"(gmem_src), "l"(pol));
}

// Defined in memory_evict.cu (single TU). Inlining createpolicy+cache_hint into large MMA
// kernels makes ptxas 13.1/sm_120a emit LDGSTS desc[URx] without R2UR (illegal insn).
__device__ void cp_async_evict_first_16_noinline(void* smem_dst, const void* gmem_src);

template <int Bytes, Cache Policy = Cache::ca>
__device__ __forceinline__ void cp_async(void* smem_dst, const void* gmem_src) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16, "cp_async supports 4, 8, or 16 bytes");
    if constexpr (Policy == Cache::cg) {
        static_assert(Bytes == 16, "cp.async.cg requires a 16-byte copy");
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src));
    } else if constexpr (Policy == Cache::EvictFirst) {
        static_assert(Bytes == 16, "cp.async L2::evict_first requires a 16-byte copy");
        cp_async_evict_first_16_noinline(smem_dst, gmem_src);
    } else {
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src), "n"(Bytes));
    }
}

// Synchronous streaming fill: ld.global.cs into regs, then plain shared store. Use when cp.async
// cannot carry an L2 eviction hint (or as an A/B against EvictFirst).
__device__ __forceinline__ void cp_sync_cs_16(void* smem_dst, const void* gmem_src) {
    unsigned x, y, z, w;
    asm volatile("ld.global.cs.v4.u32 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(x), "=r"(y), "=r"(z), "=r"(w)
                 : "l"(gmem_src));
    *reinterpret_cast<uint4*>(smem_dst) = make_uint4(x, y, z, w);
}

template <int Bytes, Cache Policy = Cache::ca>
__device__ __forceinline__ void cp_async_zfill(void* smem_dst, const void* gmem_src,
                                               int src_bytes) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16,
                  "cp_async_zfill supports 4, 8, or 16 bytes");
    if constexpr (Policy == Cache::cg) {
        static_assert(Bytes == 16, "cp.async.cg requires a 16-byte copy");
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src), "r"(src_bytes));
    } else {
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;\n"
                     :
                     : "r"(smem_addr(smem_dst)), "l"(gmem_src), "n"(Bytes), "r"(src_bytes));
    }
}

__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n"); }

template <int Groups>
__device__ __forceinline__ void cp_wait() {
    static_assert(Groups >= 0 && Groups <= 7, "cp_wait group count must fit the PTX immediate");
    asm volatile("cp.async.wait_group %0;\n" : : "n"(Groups));
}

template <int Bytes>
__device__ __forceinline__ void pipe_copy(void* smem_dst, const void* gmem_src) {
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16, "pipe_copy supports 4, 8, or 16 bytes");
    __pipeline_memcpy_async(smem_dst, gmem_src, Bytes);
}

__device__ __forceinline__ void pipe_commit() { __pipeline_commit(); }

template <int Groups>
__device__ __forceinline__ void pipe_wait() {
    static_assert(Groups >= 0 && Groups <= 7, "pipe_wait group count must fit the PTX immediate");
    __pipeline_wait_prior(Groups);
}

} // namespace ninfer::ops
