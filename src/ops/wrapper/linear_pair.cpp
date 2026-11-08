#include "ninfer/ops/linear_pair.h"

#include "ops/linear_pair/w8/w8_pair_plan.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kRows   = 1024;
constexpr std::int32_t kHidden = 5120;

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void require_matrix(const Tensor& tensor, std::int32_t rows, std::int32_t cols, const char* label) {
    if (tensor.dtype != DType::BF16 || tensor.ne[0] != rows || tensor.ne[1] != cols ||
        tensor.ne[2] != 1 || tensor.ne[3] != 1 || !tensor.is_contiguous() ||
        !aligned_to(tensor.data, 16)) {
        throw std::invalid_argument(std::string("linear_pair: invalid ") + label);
    }
}

void require_weight(const Weight& weight, const char* label) {
    constexpr std::uint64_t payload_bytes =
        static_cast<std::uint64_t>(kRows) * kHidden +
        static_cast<std::uint64_t>(kRows) * (kHidden / 32) * sizeof(std::uint16_t);
    if (weight.qtype != QType::W8G32_F16S || weight.layout != QuantLayout::RowSplit ||
        weight.scale_dtype != DType::FP16 || weight.group != 32 || weight.group_size != 32 ||
        weight.ndim != 2 || weight.n != kRows || weight.k != kHidden || weight.shape[0] != kRows ||
        weight.shape[1] != kHidden || weight.padded_shape[0] != kRows ||
        weight.padded_shape[1] != kHidden || weight.qhigh != nullptr ||
        weight.high_plane_bytes != 0 || weight.payload_bytes < payload_bytes ||
        !aligned_to(weight.qdata, 16) || !aligned_to(weight.scales, 4)) {
        throw std::invalid_argument(std::string("linear_pair: invalid ") + label);
    }
}

void require_nonoverlap(const Tensor& x, const Weight& first_weight, const Weight& second_weight,
                        const Tensor& first_out, const Tensor& second_out) {
    struct Range {
        const void* pointer;
        std::uint64_t bytes;
        const char* label;
    };

    const std::uint64_t cols        = static_cast<std::uint64_t>(x.ne[1]);
    const std::uint64_t k           = static_cast<std::uint64_t>(x.ne[0]);
    const std::uint64_t code_bytes  = 1024u * k;
    const std::uint64_t scale_bytes = 1024u * (k / 32u) * 2u;
    const std::array<Range, 7> ranges{{
        {x.data, k * cols * 2u, "x"},
        {first_out.data, 1024u * cols * 2u, "first output"},
        {second_out.data, 1024u * cols * 2u, "second output"},
        {first_weight.qdata, code_bytes, "first codes"},
        {first_weight.scales, scale_bytes, "first scales"},
        {second_weight.qdata, code_bytes, "second codes"},
        {second_weight.scales, scale_bytes, "second scales"},
    }};
    for (std::size_t i = 0; i < ranges.size(); ++i) {
        const auto first_begin = reinterpret_cast<std::uintptr_t>(ranges[i].pointer);
        const auto first_end   = first_begin + ranges[i].bytes;
        for (std::size_t j = i + 1; j < ranges.size(); ++j) {
            const auto second_begin = reinterpret_cast<std::uintptr_t>(ranges[j].pointer);
            const auto second_end   = second_begin + ranges[j].bytes;
            if (first_begin < second_end && second_begin < first_end) {
                throw std::invalid_argument(std::string("linear_pair: ") + ranges[i].label +
                                            " overlaps " + ranges[j].label);
            }
        }
    }
}

} // namespace

void linear_pair(const Tensor& x, const Weight& first_weight, const Weight& second_weight,
                 Tensor& first_out, Tensor& second_out, cudaStream_t stream) {
    const std::int32_t cols = x.ne[1];
    require_matrix(x, kHidden, cols, "x");
    require_matrix(first_out, kRows, cols, "first output");
    require_matrix(second_out, kRows, cols, "second output");
    require_weight(first_weight, "first weight");
    require_weight(second_weight, "second weight");
    require_nonoverlap(x, first_weight, second_weight, first_out, second_out);

    detail::w8_pair_dispatch(x, first_weight, second_weight, first_out, second_out, stream);
}

} // namespace ninfer::ops
