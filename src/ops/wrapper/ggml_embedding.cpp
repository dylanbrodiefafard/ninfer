#include "ninfer/ops/ggml_embedding.h"

#include "ops/launcher/ggml_embedding.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kWidth = 2560;
constexpr std::uint64_t kBlockValues = 256;
constexpr std::uint64_t kBlockBytes = 144;

bool overlaps(const void* left, std::uint64_t left_bytes, const void* right,
              std::uint64_t right_bytes) {
    const auto left_begin = reinterpret_cast<std::uintptr_t>(left);
    const auto right_begin = reinterpret_cast<std::uintptr_t>(right);
    if (left_bytes > std::numeric_limits<std::uintptr_t>::max() - left_begin ||
        right_bytes > std::numeric_limits<std::uintptr_t>::max() - right_begin) {
        throw std::invalid_argument("ggml_q4_k_embedding_row: address range overflows");
    }
    return left_begin < right_begin + right_bytes && right_begin < left_begin + left_bytes;
}

} // namespace

void ggml_q4_k_embedding_row(const Weight& weight, std::int32_t token_id, Tensor& out,
                             cudaStream_t stream) {
    if (weight.qtype != QType::GGML_Q4_K || weight.layout != QuantLayout::GgmlBlockRow ||
        weight.ndim != 2 || weight.n <= 0 || weight.n > 248320 || weight.k != kWidth ||
        weight.shape[0] != weight.n || weight.shape[1] != kWidth || weight.shape[2] != 1 ||
        weight.shape[3] != 1 || weight.padded_shape[0] != weight.n ||
        weight.padded_shape[1] != kWidth || weight.padded_shape[2] != 1 ||
        weight.padded_shape[3] != 1 || weight.group_size != kBlockValues ||
        weight.group != static_cast<std::int32_t>(kBlockValues) || weight.payload == nullptr ||
        weight.qdata != weight.payload || weight.qhigh != nullptr || weight.scales != nullptr ||
        weight.high_plane_bytes != 0) {
        throw std::invalid_argument("ggml_q4_k_embedding_row: invalid Q4_K embedding weight");
    }
    const std::uint64_t row_bytes = (kWidth / kBlockValues) * kBlockBytes;
    const std::uint64_t expected_bytes = static_cast<std::uint64_t>(weight.n) * row_bytes;
    if (weight.payload_bytes != expected_bytes || token_id < 0 || token_id >= weight.n) {
        throw std::invalid_argument("ggml_q4_k_embedding_row: invalid payload or token id");
    }
    if (out.dtype != DType::BF16 || out.ne[0] != kWidth || out.ne[1] != 1 ||
        out.ne[2] != 1 || out.ne[3] != 1 || !out.is_contiguous() || out.data == nullptr ||
        (reinterpret_cast<std::uintptr_t>(out.data) & 0xfU) != 0) {
        throw std::invalid_argument("ggml_q4_k_embedding_row: invalid output");
    }
    if (overlaps(out.data, out.bytes(), weight.payload, weight.payload_bytes)) {
        throw std::invalid_argument("ggml_q4_k_embedding_row: output overlaps weight");
    }
    detail::ggml_q4_k_embedding_row_launch(weight, token_id, out, stream);
}

} // namespace ninfer::ops
