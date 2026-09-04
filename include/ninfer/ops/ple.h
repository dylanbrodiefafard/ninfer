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

struct PleMappedIq4NlTable {
    const std::uint8_t* data = nullptr;
    std::uint64_t rows       = 0;
    std::uint64_t bytes      = 0;
};

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
 * Exact GPU decode of staged U8 IQ4_NL [90,16] into BF16 [160,16]. In each 18-byte block, bytes
 * 0..1 are a little-endian binary16 scale and byte 2+j carries output j in its low nibble and
 * output 16+j in its high nibble. The signed codebook is
 * [-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113]. The FP32 product is rounded once
 * to BF16 at the output boundary. Input and output do not overlap.
 */
void ple_iq4_nl_decode_rows(const Tensor& device_rows, Tensor& embedding, cudaStream_t stream);

/** Fixed transient device capacity for ple_inject at exact C=1 width W. */
[[nodiscard]] std::size_t ple_workspace_capacity_bytes(std::int32_t width);

/**
 * Complete preview PLE injection and convolution-state transition for C=1.
 *
 * residual/residual_out are contiguous BF16 [2560,4,W], embedding is BF16 [2560,W]. key_weight is
 * GGML Q8_0 [10240,2560], value_weight is GGML Q8_0 [2560,2560]. key_norm_weight,
 * query_norm_weight, and conv_norm_weight are FP32 [10240]; conv_weight has mathematical shape
 * [10240,4] and contiguous physical Tensor shape [4,10240] (four taps fastest per channel).
 * old_conv_state/new_conv_state are BF16 [10240,9]. The two state tensors may be disjoint or alias
 * exactly; residual_out may alias residual exactly. All other storage is non-overlapping.
 *
 * K/Q use independent branch-wise RMSNorm with converted GGUF gamma (epsilon 1e-6). GGUF has
 * already folded each source zero-centered unit offset. For branch b,
 * z=dot(Khat_b,Qhat_b)/sqrt(2560), gate=sigmoid(sign(z)*sqrt(max(abs(z),1e-6))), and
 * G_b=gate*V. N is branch-wise RMSNorm(G) with converted gamma, rounded to BF16 as the represented
 * convolution-state boundary. The depthwise cross-correlation is
 * sum_j weight[channel,j]*N_all[t-9+3*j,channel], j=0..3. The output is
 * residual + G + SiLU(conv), rounded to BF16. new state is the last nine represented N columns of
 * old state followed by current valid columns. The Op advances no token history or model frontier.
 *
 * Q8_0 projections and every subsequent floating operation execute on the GPU. W is positive and
 * at most 4096. The caller owns interval-sized workspace and all lifetimes through stream.
 */
void ple_inject(const Tensor& residual, const Tensor& embedding, const Weight& key_weight,
                const Weight& value_weight, const Tensor& key_norm_weight,
                const Tensor& query_norm_weight, const Tensor& conv_norm_weight,
                const Tensor& conv_weight, const Tensor& old_conv_state,
                Tensor& new_conv_state, Tensor& residual_out, WorkspaceArena& workspace,
                cudaStream_t stream);

} // namespace ninfer::ops
