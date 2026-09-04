#include "ninfer/ops/ggml_block_linear.h"

#include "ops/launcher/ggml_block_linear.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

struct FormatGeometry {
    std::uint32_t block_values;
    std::uint32_t block_bytes;
};

FormatGeometry format_geometry(QType qtype) {
    switch (qtype) {
    case QType::GGML_Q8_0:
        return {32, 34};
    case QType::GGML_Q4_K:
        return {256, 144};
    case QType::GGML_Q5_K:
        return {256, 176};
    case QType::GGML_Q6_K:
        return {256, 210};
    case QType::GGML_IQ1_S:
        return {256, 50};
    case QType::GGML_IQ2_XXS:
        return {256, 66};
    case QType::GGML_IQ4_NL:
        return {32, 18};
    default:
        throw std::invalid_argument("ggml_block_linear: unsupported weight qtype");
    }
}

std::uint64_t checked_mul(std::uint64_t a, std::uint64_t b) {
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) {
        throw std::invalid_argument("ggml_block_linear: weight payload size overflows");
    }
    return a * b;
}

struct ByteRange {
    std::uintptr_t begin;
    std::uintptr_t end;
};

ByteRange range(const void* data, std::uint64_t bytes) {
    const auto begin = reinterpret_cast<std::uintptr_t>(data);
    if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
        throw std::invalid_argument("ggml_block_linear: address range overflows");
    }
    return {begin, begin + static_cast<std::uintptr_t>(bytes)};
}

bool overlaps(ByteRange a, ByteRange b) { return a.begin < b.end && b.begin < a.end; }

void require_vector(const Tensor& tensor, std::int32_t size, const char* label) {
    if (tensor.dtype != DType::BF16 || tensor.ne[0] != size || tensor.ne[1] != 1 ||
        tensor.ne[2] != 1 || tensor.ne[3] != 1 || !tensor.is_contiguous() ||
        tensor.data == nullptr) {
        throw std::invalid_argument(std::string("ggml_block_linear: ") + label +
                                    " must be a non-null contiguous BF16 vector");
    }
    if ((reinterpret_cast<std::uintptr_t>(tensor.data) & 0xfU) != 0) {
        throw std::invalid_argument(std::string("ggml_block_linear: ") + label +
                                    " must be 16-byte aligned");
    }
}

} // namespace

void ggml_block_linear(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    const FormatGeometry geometry = format_geometry(w.qtype);
    if (w.layout != QuantLayout::GgmlBlockRow || w.ndim != 2 || w.n <= 0 || w.n > 248320 ||
        w.k <= 0 || w.k > 10240 ||
        w.shape[0] != w.n || w.shape[1] != w.k || w.shape[2] != 1 || w.shape[3] != 1 ||
        w.padded_shape[0] != w.n || w.padded_shape[1] != w.k ||
        w.padded_shape[2] != 1 || w.padded_shape[3] != 1 ||
        w.group_size != geometry.block_values ||
        w.group != static_cast<std::int32_t>(geometry.block_values) ||
        w.k % static_cast<std::int32_t>(geometry.block_values) != 0) {
        throw std::invalid_argument("ggml_block_linear: invalid rank-two GGML block-row weight");
    }
    const std::uint64_t row_bytes =
        checked_mul(static_cast<std::uint64_t>(w.k) / geometry.block_values,
                    geometry.block_bytes);
    const std::uint64_t expected_bytes = checked_mul(static_cast<std::uint64_t>(w.n), row_bytes);
    if (w.payload == nullptr || w.qdata != w.payload || w.payload_bytes != expected_bytes ||
        w.qhigh != nullptr || w.scales != nullptr || w.high_plane_bytes != 0) {
        throw std::invalid_argument("ggml_block_linear: invalid GGML payload metadata");
    }
    require_vector(x, w.k, "x");
    require_vector(out, w.n, "out");
    const ByteRange output_range = range(out.data, out.bytes());
    if (overlaps(output_range, range(x.data, x.bytes())) ||
        overlaps(output_range, range(w.payload, w.payload_bytes))) {
        throw std::invalid_argument("ggml_block_linear: output must not overlap input or weight");
    }
    detail::ggml_block_linear_launch(x, w, out, stream);
}

} // namespace ninfer::ops
