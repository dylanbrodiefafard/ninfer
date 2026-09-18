#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::ops {

inline constexpr std::int32_t kPleHeads             = 16;
inline constexpr std::int32_t kPleRowWidth          = 160;
inline constexpr std::int32_t kPleEmbeddingWidth    = 2560;
inline constexpr std::int32_t kPleBranches          = 4;
inline constexpr std::int32_t kPleChannels          = 10240;
inline constexpr std::int32_t kPleConvHistory       = 9;
inline constexpr std::int32_t kPleIq4NlBlockValues  = 32;
inline constexpr std::int32_t kPleIq4NlBlockBytes   = 18;
inline constexpr std::int32_t kPleIq4NlRowBytes     = 90;
inline constexpr std::int32_t kPleStagedBytes       = 1440;
inline constexpr std::int32_t kPleMaxWidth          = 4096;
inline constexpr std::size_t kPleMaxStagedBytes =
    static_cast<std::size_t>(kPleStagedBytes) * kPleMaxWidth;

struct PleMappedIq4NlTable {
    const std::uint8_t* data = nullptr;
    std::uint64_t rows       = 0;
    std::uint64_t bytes      = 0;
};

struct PleResidentFp8Table {
    const std::uint8_t* data = nullptr;
    std::uint64_t rows = 0;
    std::uint64_t bytes = 0;
};

inline constexpr std::int32_t kPleNvfp4RowBytes = 90;
inline constexpr std::int32_t kPleNvfp4StagedRowBytes = 94;

struct PleResidentNvfp4Table {
    const std::uint8_t* data = nullptr;
    std::uint64_t partitions = 0;
    std::uint64_t rows_per_partition = 0;
    std::uint64_t bytes = 0;
};

/** Gather from an eagerly populated, OS-locked NVFP4_PARTITION_F32M [P,R,160] payload.
 * Each row has 80 adjacent-pair E2M1 code bytes then ten nonnegative finite E4M3FN scales;
 * the payload ends in P positive finite little-endian FP32 partition multipliers. Global row
 * ids address partition-major rows. Selected bytes are copied into U8 [94,16,T] records:
 * 90 source row bytes followed by the exact four bytes of that row's partition multiplier.
 * CPU work is byte copying/integer addressing only. All row ids are checked before writes.
 * Mapping and bounded CUDA-pinned/device slots remain owned until stream completion. No
 * inference-time file access, allocation, floating-point decode or table repacking occurs.
 */
void ple_nvfp4_stage_rows_batch(const PleResidentNvfp4Table& table,
                                std::span<const std::int32_t> row_ids, std::int32_t width,
                                void* pinned_rows, std::size_t pinned_bytes,
                                Tensor& device_rows, cudaStream_t stream);

/** Decode U8 [94,16,T] to BF16 [160,16,T]. T in [1,4096]. Even features use the low
 * nibble. The registered dequantization is BF16_RNE(FP32_RNE(E2M1 * E4M3FN * F32 multiplier));
 * the first product is exact. Codes include signed zero. Packed inputs promise nonnegative
 * finite block scales and positive finite partition multipliers. Source and output are
 * disjoint; all lifetimes extend through stream completion. No workspace/allocation.
 */
void ple_nvfp4_decode_rows(const Tensor& device_rows, Tensor& embedding, cudaStream_t stream);

/** Gather E4M3FN rows from an already fully populated, OS-locked host table.
 * row_ids is I32 [16,T], head-fastest, T in [1,4096]. Output is U8 [160,16,T].
 * The caller owns the resident mapping and non-overlapping CUDA-pinned/device slots of
 * 2560*T bytes until stream completion; neither slot may be reused before that completion.
 * Validates all IDs before writing. Host work is byte copies only: no decode, allocation,
 * file access or floating-point math. Residency admission belongs to artifact loading.
 */
void ple_fp8_stage_rows_batch(const PleResidentFp8Table& table,
                             std::span<const std::int32_t> row_ids, std::int32_t width,
                             void* pinned_rows, std::size_t pinned_bytes,
                             Tensor& device_rows, cudaStream_t stream);

/** Exact E4M3FN per-tensor PLE decode: output = BF16_RNE(E4M3FN(code) * BF16(scale_bits)).
 * Input U8 [160,16,T] contains finite E4M3FN codes, including signed zero; output is BF16
 * [160,16,T], T in [1,4096]. scale_bits is the exact stored positive finite BF16 multiplier,
 * not a reciprocal or per-row scale. Input/output are disjoint; no workspace or allocation.
 * Storage remains alive through stream completion. The raw FP8 codes convert exactly to BF16
 * before multiplication, preserving the source gathered-value boundary.
 */
void ple_fp8_decode_rows(const Tensor& device_rows, std::uint16_t scale_bits,
                         Tensor& embedding, cudaStream_t stream);

/**
 * Copy exactly the sixteen IQ4_NL rows named by row_ids from a read-only mapped table into the
 * caller's fixed pinned slot, then enqueue one H2D transfer into device_rows. Each logical row is
 * 160 values stored as five 18-byte IQ4_NL blocks, hence both slots are exactly 1440 bytes.
 *
 * This is the complete host boundary: it performs no decode, floating-point work, allocation,
 * file I/O, deduplication, or scheduling. pinned_rows must name cudaMallocHost/registered storage
 * of at least 1440 bytes. device_rows is contiguous U8 [90,16]. The mapped span and both staging
 * slots are non-overlapping and remain alive through completion of stream.
 */
void ple_iq4_nl_stage_rows(const PleMappedIq4NlTable& table,
                           std::span<const std::int32_t, kPleHeads> row_ids, void* pinned_rows,
                           std::size_t pinned_bytes, Tensor& device_rows, cudaStream_t stream);

/**
 * T-wide form of ple_iq4_nl_stage_rows. row_ids is the contiguous I32 logical shape [16,T],
 * head-fastest. The exact selected source bytes are packed into U8 [90,16,T], also
 * head-fastest within each token, and transferred with one H2D. T is in [1,4096]. The caller's
 * pinned and device slots require exactly 1440*T live bytes; all other ownership and host-work
 * restrictions are identical to the scalar entry point.
 */
void ple_iq4_nl_stage_rows_batch(const PleMappedIq4NlTable& table,
                                 std::span<const std::int32_t> row_ids, std::int32_t width,
                                 void* pinned_rows, std::size_t pinned_bytes,
                                 Tensor& device_rows, cudaStream_t stream);

/**
 * Exact GPU decode of staged U8 IQ4_NL [90,16,T] into BF16 [160,16,T], T in [1,4096]. In each
 * 18-byte block, bytes
 * 0..1 are a little-endian binary16 scale and byte 2+j carries output j in its low nibble and
 * output 16+j in its high nibble. The signed codebook is
 * [-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113]. The FP32 product is rounded once
 * to BF16 at the output boundary. Input and output do not overlap.
 */
void ple_iq4_nl_decode_rows(const Tensor& device_rows, Tensor& embedding, cudaStream_t stream);

enum class PleNormFormat { EffectiveFp32, ZeroCenteredBf16 };

/** Transient capacity for C=1 width W and explicit projection formats.
 * Native formats use Linear A16Only. The mapped table format is unchanged. */
[[nodiscard]] std::size_t ple_workspace_capacity_bytes(
    std::int32_t width, QType key, QType value);

/**
 * Complete preview PLE injection and convolution-state transition for C=1.
 *
 * residual/residual_out are contiguous BF16 [2560,4,W], embedding is BF16 [2560,W]. key_weight is
 * BF16, Q8_0, NVFP4 or row-scaled FP8 [10240,2560]; value_weight independently uses those formats
 * at [2560,2560]. key_norm_weight,
 * query_norm_weight, and conv_norm_weight are [10240]: effective FP32 gamma for EffectiveFp32,
 * or source zero-centered BF16 parameters for ZeroCenteredBf16 (effective gamma is FP32 1+w).
 * conv_weight independently admits BF16 or FP32 and has mathematical shape
 * [10240,4] and contiguous physical Tensor shape [4,10240] (four taps fastest per channel).
 * old_conv_state/new_conv_state are BF16 [10240,9]. The two state tensors may be disjoint or alias
 * exactly; residual_out may alias residual exactly. All other storage is non-overlapping.
 *
 * K/Q use independent branch-wise RMSNorm with effective gamma (epsilon 1e-6). For branch b,
 * z=dot(Khat_b,Qhat_b)/sqrt(2560), gate=sigmoid(sign(z)*sqrt(max(abs(z),1e-6))), and
 * G_b=gate*V. N is branch-wise RMSNorm(G) with effective gamma, rounded to BF16 as the represented
 * convolution-state boundary. The depthwise cross-correlation is
 * sum_j weight[channel,j]*N_all[t-9+3*j,channel], j=0..3. The output is
 * residual + G + SiLU(conv), rounded to BF16. new state is the last nine represented N columns of
 * old state followed by current valid columns. The Op advances no token history or model frontier.
 *
 * Projections and every subsequent floating operation execute on the GPU. W is positive and
 * at most 4096. The caller owns interval-sized workspace and all lifetimes through stream.
 */
void ple_inject(const Tensor& residual, const Tensor& embedding, const Weight& key_weight,
                const Weight& value_weight, const Tensor& key_norm_weight,
                const Tensor& query_norm_weight, const Tensor& conv_norm_weight,
                const Tensor& conv_weight, const Tensor& old_conv_state,
                Tensor& new_conv_state, Tensor& residual_out, WorkspaceArena& workspace,
                PleNormFormat norm_format, cudaStream_t stream);

} // namespace ninfer::ops
