#include "ninfer/ops/linear.h"

#include "ops/linear/nvfp4/nvfp4_format.h"
#include "ops/linear/nvfp4/nvfp4_kv_projection.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

void require_matrix(const Tensor& matrix, std::int32_t rows, std::int32_t tokens,
                    const char* label) {
    if (matrix.dtype != DType::BF16 || matrix.ne[0] != rows || matrix.ne[1] != tokens ||
        matrix.ne[2] != 1 || matrix.ne[3] != 1 || !matrix.is_contiguous() ||
        matrix.data == nullptr || (reinterpret_cast<std::uintptr_t>(matrix.data) & 15U) != 0) {
        throw std::invalid_argument(std::string("linear_kv_projection: invalid ") + label);
    }
}

struct AddressRange {
    std::uintptr_t begin;
    std::uintptr_t end;
};

AddressRange address_range(const void* data, std::uint64_t bytes) {
    const auto begin = reinterpret_cast<std::uintptr_t>(data);
    if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
        throw std::invalid_argument("linear_kv_projection: address range overflows");
    }
    return {begin, begin + static_cast<std::uintptr_t>(bytes)};
}

bool overlaps(AddressRange first, AddressRange second) {
    return first.begin < second.end && second.begin < first.end;
}

struct RowSplitGeometry {
    std::uint64_t code_row_bytes;
    std::uint64_t scale_row_bytes;
};

RowSplitGeometry require_row_split_weight(const Weight& weight) {
    const auto group          = weight.qtype == QType::W8G32_F16S ? 32 : 64;
    const auto groups_per_row = static_cast<std::uint64_t>(weight.k / group);
    const RowSplitGeometry geometry{groups_per_row * 32, groups_per_row * 2};
    const auto codes_bytes  = static_cast<std::uint64_t>(weight.n) * geometry.code_row_bytes;
    const auto scale_offset = (codes_bytes + 255) / 256 * 256;
    const auto scales_bytes = static_cast<std::uint64_t>(weight.n) * geometry.scale_row_bytes;
    if (weight.layout != QuantLayout::RowSplit || weight.scale_dtype != DType::FP16 ||
        weight.group_size != static_cast<std::uint32_t>(group) || weight.group != group ||
        weight.ndim != 2 || weight.shape[0] != weight.n || weight.shape[1] != weight.k ||
        weight.padded_shape[0] != weight.n || weight.padded_shape[1] != weight.k ||
        weight.payload == nullptr || weight.payload_bytes < scale_offset + scales_bytes ||
        weight.qdata != weight.payload || weight.qhigh != nullptr || weight.high_plane_bytes != 0 ||
        (reinterpret_cast<std::uintptr_t>(weight.payload) & 15U) != 0) {
        throw std::invalid_argument("linear_kv_projection: invalid original RowSplit weight");
    }
    (void)address_range(weight.payload, weight.payload_bytes);
    if (weight.scales != static_cast<const std::byte*>(weight.payload) + scale_offset) {
        throw std::invalid_argument("linear_kv_projection: invalid original RowSplit scale plane");
    }
    return geometry;
}

Weight row_split_view(const Weight& parent, RowSplitGeometry geometry, std::int32_t first_row) {
    Weight view          = parent;
    view.n               = 1024;
    view.shape[0]        = 1024;
    view.padded_shape[0] = 1024;
    view.qdata           = static_cast<const std::byte*>(parent.qdata) +
                           static_cast<std::uint64_t>(first_row) * geometry.code_row_bytes;
    view.scales          = static_cast<const std::byte*>(parent.scales) +
                           static_cast<std::uint64_t>(first_row) * geometry.scale_row_bytes;
    return view;
}

} // namespace

void linear_kv_projection(const Tensor& x, const Weight& w, Tensor& key, Tensor& value,
                          cudaStream_t stream, std::int32_t sequence_width) {
    constexpr std::int32_t kHidden = 5120;
    constexpr std::int32_t kKvRows = 1024;
    const auto tokens              = x.ne[1];
    if (w.n != 6144 || w.k != kHidden || tokens <= 0 || sequence_width <= 0 ||
        tokens % sequence_width != 0 || tokens / sequence_width > 6) {
        throw std::invalid_argument("linear_kv_projection: unsupported original or packed shape");
    }
    RowSplitGeometry row_split{};
    switch (w.qtype) {
    case QType::NVFP4:
        (void)detail::validate_nvfp4_weight(w, "linear_kv_projection");
        break;
    case QType::W8G32_F16S:
    case QType::Q4G64_F16S:
        row_split = require_row_split_weight(w);
        break;
    default:
        throw std::invalid_argument("linear_kv_projection: unsupported original weight format");
    }
    require_matrix(x, kHidden, tokens, "input");
    require_matrix(key, kKvRows, tokens, "key output");
    require_matrix(value, kKvRows, tokens, "value output");

    const auto input   = address_range(x.data, static_cast<std::uint64_t>(kHidden) * tokens * 2);
    const auto payload = address_range(w.payload, w.payload_bytes);
    const auto key_range =
        address_range(key.data, static_cast<std::uint64_t>(kKvRows) * tokens * 2);
    const auto value_range =
        address_range(value.data, static_cast<std::uint64_t>(kKvRows) * tokens * 2);
    if (overlaps(key_range, value_range) || overlaps(key_range, input) ||
        overlaps(value_range, input) || overlaps(key_range, payload) ||
        overlaps(value_range, payload)) {
        throw std::invalid_argument(
            "linear_kv_projection: output aliases another public input or output");
    }
    if (w.qtype == QType::NVFP4) {
        detail::launch_nvfp4_kv_projection(x, w, key, value, stream, sequence_width);
    } else {
        const Weight key_weight   = row_split_view(w, row_split, 4096);
        const Weight value_weight = row_split_view(w, row_split, 5120);
        linear_packed_sequences(x, key_weight, key, stream, sequence_width);
        linear_packed_sequences(x, value_weight, value, stream, sequence_width);
    }
}

} // namespace ninfer::ops
