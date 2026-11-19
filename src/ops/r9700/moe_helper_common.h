#pragma once

// Private gfx1201 primitives shared by the moe_helper_experts routes
// (ninfer/ops/moe_helper_experts.h): LDS and wave synchronization, record loads, the exact E4M3
// decode, the A8 profile's E4M3 encode, BF16 rounding, SiLU, the wave32 BF16 and FP8 WMMA, and
// launch-error reporting. Included only by HIP translation units of src/ops/r9700.

#include <hip/hip_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {

// A workgroup barrier ordering only LDS: __syncthreads' workgroup fence also waits for every
// global load in flight, which would drain the loads a pipeline issued for its next stage.
__device__ __forceinline__ void lds_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    // NOLINTNEXTLINE(portability-no-assembler): this wave's LDS accesses; no builtin on gfx12
    asm volatile("s_wait_dscnt 0x0" ::: "memory");
    __builtin_amdgcn_s_barrier_signal(-1);
    __builtin_amdgcn_s_barrier_wait(-1);
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// Orders one wave's LDS writes before its later LDS reads by other lanes of the same wave.
__device__ __forceinline__ void wave_lds_sync() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront", "local");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront", "local");
}

// Exact E4M3FN (bias 7, subnormals 2^-9 m) to FP32; the NaN encodings are not valid scales.
__device__ __forceinline__ float e4m3_to_float(std::uint32_t bits) {
    const std::uint32_t magnitude = bits & 0x7FU;
    const float value = magnitude >= 8U ? __uint_as_float((magnitude << 20) + 0x3C000000U)
                                        : static_cast<float>(magnitude) * 0x1p-9F;
    return (bits & 0x80U) != 0 ? -value : value;
}

// Record bytes in the global address space. A record address comes from an integer table entry,
// which would otherwise compile to flat loads; flat loads also count on the LDS wait counter, so
// every LDS wait would drain the in-flight record loads.
using RecordBytes = const __attribute__((address_space(1))) std::uint8_t*;

using U32x2 = std::uint32_t __attribute__((ext_vector_type(2)));
using U32x4 = std::uint32_t __attribute__((ext_vector_type(4)));

template <class T>
__device__ __forceinline__ T load_record(RecordBytes bytes) {
    return *reinterpret_cast<const __attribute__((address_space(1))) T*>(bytes);
}

// The record address of an expert-table entry (bit 0 holds the codec).
__device__ __forceinline__ RecordBytes record_address(std::uint64_t entry) {
    // NOLINTNEXTLINE(performance-no-int-to-ptr): the table holds addresses as integers by contract
    return reinterpret_cast<RecordBytes>(entry & ~std::uint64_t{1});
}

// gfx12 wave32 WMMA 16x16x16 BF16 -> FP32. Lane l holds A row l % 16 and B column l % 16 at K
// 8 (l / 16) .. + 7 (eight BF16), and D column l % 16 at rows 8 (l / 16) .. + 7.
using WmmaBf16 = __attribute__((__vector_size__(8 * sizeof(short)))) short;
using WmmaF32  = __attribute__((__vector_size__(8 * sizeof(float)))) float;

__device__ __forceinline__ WmmaF32 wmma(const WmmaBf16& a, const WmmaBf16& b, const WmmaF32& c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
}

// gfx12 wave32 WMMA 16x16x16 E4M3 x E4M3 -> FP32. Lane l holds A row l % 16 and B column l % 16
// at eight K (eight bytes) of K half l / 16, and D is laid out as for BF16. Every lane's byte i of
// A and of B covers the same K, so any K permutation applied to both operands gives the same sum.
using WmmaFp8 = int __attribute__((ext_vector_type(2)));

__device__ __forceinline__ WmmaF32 wmma_fp8(const WmmaFp8& a, const WmmaFp8& b, const WmmaF32& c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c);
}

// The A8 profile's code of a finite quotient v = x / s: E4M3FN round to nearest even, saturating
// to +-448. fmed3 clamps |v| to 448 (exact in E4M3), so v_cvt_pk_fp8_f32's RNE never reaches the
// NaN encoding; the sign of zero and of underflow to zero is kept (0x80 for negative). Byte 0 of
// the result holds the code of `low`, byte 1 that of `high`.
__device__ __forceinline__ std::uint32_t e4m3_pair(float low, float high) {
    return static_cast<std::uint32_t>(__builtin_amdgcn_cvt_pk_fp8_f32(
               __builtin_amdgcn_fmed3f(low, 448.0F, -448.0F),
               __builtin_amdgcn_fmed3f(high, 448.0F, -448.0F), 0, false)) &
           0xFFFFU;
}

// Round to nearest even; the inputs are finite.
__device__ __forceinline__ std::uint16_t bf16_round(float value) {
    const std::uint32_t bits = __float_as_uint(value);
    return static_cast<std::uint16_t>((bits + 0x7FFFU + ((bits >> 16) & 1U)) >> 16);
}

__device__ __forceinline__ float silu(float z) { return z / (1.0F + expf(-z)); }

inline void hip_check(hipError_t status, const char* what) {
    if (status != hipSuccess) {
        throw std::runtime_error(std::string("moe_helper_experts: ") + what + ": " +
                                 hipGetErrorString(status));
    }
}

} // namespace ninfer::ops::detail
