#include "ops/launcher/ggml_embedding.h"

#include "core/device.h"
#include "ops/kernel/ggml_block_linear.cuh"

#include <cuda_bf16.h>

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

constexpr int kWidth       = 2560;
constexpr int kBlockValues = GgmlDecoder<QType::GGML_Q4_K>::block_values;
constexpr int kBlockBytes  = GgmlDecoder<QType::GGML_Q4_K>::block_bytes;
constexpr int kRowBlocks   = kWidth / kBlockValues;
constexpr int kRowBytes    = kRowBlocks * kBlockBytes;

__global__ void q4_k_embedding_row_kernel(const std::uint8_t* weights, std::int32_t token_id,
                                          __nv_bfloat16* out) {
    const int column = static_cast<int>(blockIdx.x) * kBlockValues + threadIdx.x;
    const auto* block = weights + static_cast<std::uint64_t>(token_id) * kRowBytes +
                        static_cast<std::uint64_t>(blockIdx.x) * kBlockBytes;
    out[column] = __float2bfloat16_rn(
        GgmlDecoder<QType::GGML_Q4_K>::value(block, static_cast<int>(threadIdx.x)));
}

__global__ void q4_k_embedding_kernel(const std::uint8_t* weights,
                                      const std::int32_t* token_ids, std::int32_t rows,
                                      __nv_bfloat16* out) {
    const int token = static_cast<int>(blockIdx.y);
    const int token_id = token_ids[token];
    const int column = static_cast<int>(blockIdx.x) * kBlockValues + threadIdx.x;
    if (token_id < 0 || token_id >= rows) {
        out[column + static_cast<std::int64_t>(kWidth) * token] = __float2bfloat16_rn(0.0F);
        return;
    }
    const auto* block = weights + static_cast<std::uint64_t>(token_id) * kRowBytes +
                        static_cast<std::uint64_t>(blockIdx.x) * kBlockBytes;
    out[column + static_cast<std::int64_t>(kWidth) * token] = __float2bfloat16_rn(
        GgmlDecoder<QType::GGML_Q4_K>::value(block, static_cast<int>(threadIdx.x)));
}

} // namespace

void ggml_q4_k_embedding_row_launch(const Weight& weight, std::int32_t token_id, Tensor& out,
                                    cudaStream_t stream) {
    q4_k_embedding_row_kernel<<<kRowBlocks, kBlockValues, 0, stream>>>(
        static_cast<const std::uint8_t*>(weight.qdata), token_id,
        static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

void ggml_q4_k_embedding_launch(const Weight& weight, const Tensor& token_ids, Tensor& out,
                                cudaStream_t stream) {
    const dim3 grid(kRowBlocks, static_cast<unsigned>(token_ids.ne[0]));
    q4_k_embedding_kernel<<<grid, kBlockValues, 0, stream>>>(
        static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::int32_t*>(token_ids.data), weight.n,
        static_cast<__nv_bfloat16*>(out.data));
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
