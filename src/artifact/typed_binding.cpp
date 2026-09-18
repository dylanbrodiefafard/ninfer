#include "artifact/typed_binding.h"

#include "artifact/materializer.h"

#include <array>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>

namespace ninfer::artifact {
namespace {

StorageLayout storage_layout_for(NumericFormat format) {
    switch (format) {
    case NumericFormat::BF16:
    case NumericFormat::FP32:
    case NumericFormat::I32:
        return StorageLayout::ContiguousLeV1;
    case NumericFormat::Q4G64_F16S:
    case NumericFormat::Q5G64_F16S:
    case NumericFormat::Q6G64_F16S:
    case NumericFormat::W8G32_F16S:
        return StorageLayout::RowSplitK128V1;
    case NumericFormat::NVFP4:
        return StorageLayout::BlockScaleK16M128x4V1;
    case NumericFormat::NVFP4_EXPERT_F32M:
        return StorageLayout::ExpertBlockScaleK16M128x4V1;
    case NumericFormat::NVFP4_PARTITION_F32M:
        return StorageLayout::PartitionedRowBlockScaleK16V1;
    case NumericFormat::FP8_E4M3FN_ROW_BF16S:
        return StorageLayout::RowScaleV1;
    case NumericFormat::FP8_E4M3FN_TENSOR_BF16S:
        return StorageLayout::TensorScaleV1;
    case NumericFormat::FP8_E4M3FN_TENSOR_F32M:
        return StorageLayout::TensorCalibratedV1;
    case NumericFormat::Q8_0:
    case NumericFormat::Q4_K:
    case NumericFormat::Q5_K:
    case NumericFormat::Q6_K:
    case NumericFormat::IQ1_S:
    case NumericFormat::IQ2_XXS:
    case NumericFormat::IQ4_NL:
        return StorageLayout::GgmlBlockRowV1;
    }
    throw std::logic_error("unhandled numeric format");
}

QType qtype_for(NumericFormat format) {
    switch (format) {
    case NumericFormat::BF16:
        return QType::BF16_CTRL;
    case NumericFormat::FP32:
        return QType::FP32_CTRL;
    case NumericFormat::I32:
        return QType::I32_CTRL;
    case NumericFormat::Q4G64_F16S:
        return QType::Q4G64_F16S;
    case NumericFormat::Q5G64_F16S:
        return QType::Q5G64_F16S;
    case NumericFormat::Q6G64_F16S:
        return QType::Q6G64_F16S;
    case NumericFormat::W8G32_F16S:
        return QType::W8G32_F16S;
    case NumericFormat::NVFP4:
        return QType::NVFP4;
    case NumericFormat::NVFP4_EXPERT_F32M:
        return QType::NVFP4_EXPERT_F32M;
    case NumericFormat::FP8_E4M3FN_ROW_BF16S:
        return QType::FP8_E4M3FN_ROW_BF16S;
    case NumericFormat::FP8_E4M3FN_TENSOR_F32M:
        return QType::FP8_E4M3FN_TENSOR_F32M;
    case NumericFormat::FP8_E4M3FN_TENSOR_BF16S:
    case NumericFormat::NVFP4_PARTITION_F32M:
        throw std::invalid_argument("embedding table is not a Linear Weight");
    case NumericFormat::Q8_0:
        return QType::GGML_Q8_0;
    case NumericFormat::Q4_K:
        return QType::GGML_Q4_K;
    case NumericFormat::Q5_K:
        return QType::GGML_Q5_K;
    case NumericFormat::Q6_K:
        return QType::GGML_Q6_K;
    case NumericFormat::IQ1_S:
        return QType::GGML_IQ1_S;
    case NumericFormat::IQ2_XXS:
        return QType::GGML_IQ2_XXS;
    case NumericFormat::IQ4_NL:
        return QType::GGML_IQ4_NL;
    }
    throw std::logic_error("unhandled numeric format");
}

DType dtype_for(NumericFormat format) {
    switch (format) {
    case NumericFormat::BF16:
        return DType::BF16;
    case NumericFormat::FP32:
        return DType::FP32;
    case NumericFormat::I32:
        return DType::I32;
    default:
        throw std::logic_error("quantized format has no direct dtype");
    }
}

Weight contiguous_weight(const MaterializedArtifact& materialized, ObjectHandle handle,
                         NumericFormat format, std::int32_t rows, std::int32_t columns) {
    Weight out{};
    out.payload       = materialized.device_data(handle);
    out.qdata         = out.payload;
    out.payload_bytes = static_cast<std::uint64_t>(rows) * columns * dtype_size(dtype_for(format));
    out.qtype         = qtype_for(format);
    out.layout        = QuantLayout::Contiguous;
    out.n             = rows;
    out.k             = columns;
    out.ndim          = 2;
    out.shape[0]      = rows;
    out.shape[1]      = columns;
    out.padded_shape[0] = rows;
    out.padded_shape[1] = columns;
    return out;
}

Weight row_split_weight(const MaterializedArtifact& materialized, ObjectHandle handle,
                        NumericFormat format, std::int32_t rows, std::int32_t columns) {
    const std::array<std::uint64_t, 2> shape = {static_cast<std::uint64_t>(rows),
                                                static_cast<std::uint64_t>(columns)};
    const RowSplitGeometry geometry          = row_split_geometry(format, shape);
    const auto* bytes = static_cast<const std::byte*>(materialized.device_data(handle));

    Weight out{};
    out.payload          = bytes;
    out.payload_bytes    = geometry.encoded_bytes;
    out.high_plane_bytes = geometry.high_plane_bytes;
    out.qtype            = qtype_for(format);
    out.layout           = QuantLayout::RowSplit;
    out.group_size       = static_cast<std::uint32_t>(geometry.group_size);
    out.qdata            = bytes;
    out.qhigh       = geometry.high_plane_bytes == 0 ? nullptr : bytes + geometry.high_plane_offset;
    out.scales      = bytes + geometry.scale_plane_offset;
    out.n           = rows;
    out.k           = columns;
    out.group       = static_cast<std::int32_t>(geometry.group_size);
    out.scale_dtype = DType::FP16;
    out.ndim        = 2;
    out.shape[0]    = rows;
    out.shape[1]    = columns;
    out.padded_shape[0] = rows;
    out.padded_shape[1] = static_cast<std::int32_t>(geometry.padded_columns);
    return out;
}

Weight row_scale_weight(const MaterializedArtifact& materialized, ObjectHandle handle,
                        NumericFormat format, std::int32_t rows, std::int32_t columns) {
    const std::array<std::uint64_t, 2> shape = {static_cast<std::uint64_t>(rows),
                                                static_cast<std::uint64_t>(columns)};
    const RowScaleGeometry geometry          = row_scale_geometry(format, shape);
    const auto* bytes = static_cast<const std::byte*>(materialized.device_data(handle));

    Weight out{};
    out.payload         = bytes;
    out.payload_bytes   = geometry.encoded_bytes;
    out.qtype           = qtype_for(format);
    out.layout          = QuantLayout::RowScale;
    out.group_size      = static_cast<std::uint32_t>(geometry.columns);
    out.qdata           = bytes;
    out.scales          = bytes + geometry.scale_plane_offset;
    out.n               = rows;
    out.k               = columns;
    out.group           = columns;
    out.scale_dtype     = DType::BF16;
    out.ndim            = 2;
    out.shape[0]        = rows;
    out.shape[1]        = columns;
    out.padded_shape[0] = rows;
    out.padded_shape[1] = columns;
    out.scale_ne[0]     = rows;
    out.scale_nb[0]     = 2;
    out.scale_nb[1]     = static_cast<std::int64_t>(rows) * 2;
    out.scale_nb[2]     = out.scale_nb[1];
    out.scale_nb[3]     = out.scale_nb[1];
    return out;
}

bool is_ggml_block_format(NumericFormat format) {
    return storage_layout_for(format) == StorageLayout::GgmlBlockRowV1;
}

struct GgmlQtypeGeometry {
    std::uint32_t block_values;
    std::uint32_t block_bytes;
};

GgmlQtypeGeometry ggml_qtype_geometry(QType qtype) {
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
        throw std::invalid_argument("ggml_block_matrix_view: weight is not a GGML block format");
    }
}

Weight ggml_block_weight(const MaterializedArtifact& materialized, ObjectHandle handle,
                         NumericFormat format, std::span<const std::int32_t> shape) {
    if (!is_ggml_block_format(format) || (shape.size() != 2 && shape.size() != 3)) {
        throw std::invalid_argument(
            "materialized_ggml_block_weight: expected a rank-two matrix or rank-three bank");
    }
    std::array<std::uint64_t, 3> artifact_shape{};
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] <= 0) {
            throw std::invalid_argument(
                "materialized_ggml_block_weight: dimensions must be positive");
        }
        artifact_shape[i] = static_cast<std::uint64_t>(shape[i]);
    }
    const auto geometry =
        ggml_block_geometry(format, std::span(artifact_shape.data(), shape.size()));
    const auto* bytes = static_cast<const std::byte*>(materialized.device_data(handle));

    Weight out{};
    out.payload       = bytes;
    out.payload_bytes = geometry.encoded_bytes;
    out.qtype         = qtype_for(format);
    out.layout        = QuantLayout::GgmlBlockRow;
    out.group_size    = static_cast<std::uint32_t>(geometry.block_values);
    out.qdata         = bytes;
    out.n             = static_cast<std::int32_t>(geometry.rows_per_matrix);
    out.k             = static_cast<std::int32_t>(geometry.columns);
    out.group         = static_cast<std::int32_t>(geometry.block_values);
    out.ndim          = static_cast<std::uint32_t>(shape.size());
    for (std::size_t i = 0; i < shape.size(); ++i) {
        out.shape[i]        = shape[i];
        out.padded_shape[i] = shape[i];
    }
    return out;
}

} // namespace

ObjectHandle bind_tensor(Binder& binder, std::string_view name, NumericFormat format,
                         std::initializer_list<std::uint64_t> shape, TensorPlacement placement) {
    const ObjectHandle handle =
        binder.require_tensor(name, format, storage_layout_for(format),
                              std::span<const std::uint64_t>(shape.begin(), shape.size()));
    switch (placement) {
    case TensorPlacement::Device:
        binder.materialize_on_device(handle);
        break;
    case TensorPlacement::MappedHost:
        binder.map_tensor_on_host(handle);
        break;
    case TensorPlacement::ResidentHost:
        binder.map_tensor_on_host(handle, true);
        break;
    case TensorPlacement::ValidateOnly:
        binder.validate_only(handle);
        break;
    default:
        throw std::invalid_argument("bind_tensor: invalid tensor placement");
    }
    return handle;
}

ObjectHandle bind_device_tensor(Binder& binder, std::string_view name, NumericFormat format,
                                std::initializer_list<std::uint64_t> shape) {
    return bind_tensor(binder, name, format, shape, TensorPlacement::Device);
}

ObjectHandle bind_raw_resource(Binder& binder, std::string_view name) {
    const ObjectHandle handle = binder.require_resource(name, ResourceEncoding::RawBytesV1);
    binder.retain_on_host(handle);
    return handle;
}

Tensor materialized_tensor(const MaterializedArtifact& materialized, ObjectHandle handle,
                           NumericFormat format,
                           std::initializer_list<std::int32_t> internal_shape) {
    return Tensor(materialized.device_data(handle), dtype_for(format), internal_shape);
}

Weight materialized_weight(const MaterializedArtifact& materialized, ObjectHandle handle,
                           NumericFormat format, std::int32_t rows, std::int32_t columns) {
    if (format == NumericFormat::FP8_E4M3FN_TENSOR_F32M) {
        if (rows <= 0 || columns <= 0) { throw std::invalid_argument("FP8 matrix dimensions must be positive"); }
        const std::array<std::uint64_t, 2> shape{static_cast<std::uint64_t>(rows), static_cast<std::uint64_t>(columns)};
        const auto geometry = tensor_calibrated_geometry(format, shape);
        Weight out{};
        out.payload = materialized.device_data(handle); out.qdata = out.payload;
        out.payload_bytes = geometry.encoded_bytes;
        out.scales = static_cast<const std::byte*>(out.payload) + geometry.multiplier_offset;
        out.qtype = QType::FP8_E4M3FN_TENSOR_F32M; out.layout = QuantLayout::TensorCalibrated;
        out.scale_dtype = DType::FP32; out.ndim = 2; out.n = rows; out.k = columns;
        out.shape[0] = out.padded_shape[0] = rows;
        out.shape[1] = out.padded_shape[1] = columns;
        out.scale_ne[0] = 2; out.scale_nb[0] = 4;
        for (int i = 1; i < 4; ++i) { out.scale_nb[i] = 8; }
        return out;
    }
    if (format == NumericFormat::FP8_E4M3FN_TENSOR_BF16S ||
        format == NumericFormat::NVFP4_PARTITION_F32M) {
        throw std::invalid_argument("embedding table is not a Linear Weight");
    }
    if (format == NumericFormat::NVFP4_EXPERT_F32M) {
        throw std::invalid_argument("materialized_weight: use rank-three expert bank binding");
    }
    if (format == NumericFormat::NVFP4) {
        throw std::invalid_argument(
            "materialized_weight: NVFP4 requires target-validated weight and input divisors");
    }
    if (is_ggml_block_format(format)) {
        const std::array<std::int32_t, 2> shape = {rows, columns};
        return ggml_block_weight(materialized, handle, format, shape);
    }
    if (storage_layout_for(format) == StorageLayout::ContiguousLeV1) {
        return contiguous_weight(materialized, handle, format, rows, columns);
    }
    if (storage_layout_for(format) == StorageLayout::RowScaleV1) {
        return row_scale_weight(materialized, handle, format, rows, columns);
    }
    return row_split_weight(materialized, handle, format, rows, columns);
}

Weight materialized_ggml_block_weight(const MaterializedArtifact& materialized,
                                      ObjectHandle handle, NumericFormat format,
                                      std::initializer_list<std::int32_t> shape) {
    return ggml_block_weight(materialized, handle, format,
                             std::span<const std::int32_t>(shape.begin(), shape.size()));
}

Weight materialized_nvfp4_expert_weight(const MaterializedArtifact& materialized,
                                        ObjectHandle handle, std::int32_t experts,
                                        std::int32_t rows, std::int32_t columns) {
    if (experts <= 0 || rows <= 0 || columns <= 0) {
        throw std::invalid_argument("NVFP4 expert dimensions must be positive");
    }
    const std::array<std::uint64_t, 3> shape = {
        static_cast<std::uint64_t>(experts), static_cast<std::uint64_t>(rows),
        static_cast<std::uint64_t>(columns)};
    const auto geometry = expert_block_scale_geometry(NumericFormat::NVFP4_EXPERT_F32M, shape);
    Weight out{};
    out.payload = materialized.device_data(handle);
    out.qdata = out.payload;
    out.scales = static_cast<const std::byte*>(out.payload) + geometry.scale_plane_offset;
    out.payload_bytes = geometry.encoded_bytes;
    out.qtype = QType::NVFP4_EXPERT_F32M;
    out.scale_dtype = DType::FP8_E4M3FN;
    out.layout = QuantLayout::ExpertBlockScaleK16M128x4;
    out.ndim = 3;
    out.n = rows; out.k = columns; out.group = 16; out.group_size = 16;
    out.shape[0] = out.padded_shape[0] = experts;
    out.shape[1] = out.padded_shape[1] = rows;
    out.shape[2] = out.padded_shape[2] = columns;
    return out;
}

Weight ggml_block_matrix_view(const Weight& bank, std::int32_t matrix_index) {
    const auto geometry = ggml_qtype_geometry(bank.qtype);
    if (bank.layout != QuantLayout::GgmlBlockRow || bank.ndim != 3 || bank.shape[0] <= 0 ||
        bank.shape[1] <= 0 || bank.shape[2] <= 0 || matrix_index < 0 ||
        matrix_index >= bank.shape[0] || bank.payload == nullptr || bank.qdata != bank.payload ||
        bank.qhigh != nullptr || bank.scales != nullptr || bank.high_plane_bytes != 0 ||
        bank.n != bank.shape[1] || bank.k != bank.shape[2] ||
        bank.padded_shape[0] != bank.shape[0] || bank.padded_shape[1] != bank.shape[1] ||
        bank.padded_shape[2] != bank.shape[2] || bank.group_size != geometry.block_values ||
        bank.group != static_cast<std::int32_t>(geometry.block_values) ||
        bank.shape[2] % static_cast<std::int32_t>(geometry.block_values) != 0) {
        throw std::invalid_argument("ggml_block_matrix_view: invalid GGML expert bank or index");
    }
    const std::uint64_t row_bytes =
        static_cast<std::uint64_t>(bank.shape[2]) / geometry.block_values * geometry.block_bytes;
    if (row_bytes > std::numeric_limits<std::uint64_t>::max() /
                        static_cast<std::uint64_t>(bank.shape[1])) {
        throw std::invalid_argument("ggml_block_matrix_view: GGML matrix size overflows");
    }
    const std::uint64_t matrix_bytes = row_bytes * static_cast<std::uint64_t>(bank.shape[1]);
    if (matrix_bytes > std::numeric_limits<std::uint64_t>::max() /
                           static_cast<std::uint64_t>(bank.shape[0]) ||
        bank.payload_bytes != matrix_bytes * static_cast<std::uint64_t>(bank.shape[0])) {
        throw std::invalid_argument("ggml_block_matrix_view: invalid GGML expert bank payload");
    }
    const auto offset = matrix_bytes * static_cast<std::uint64_t>(matrix_index);
    const auto* data  = static_cast<const std::byte*>(bank.payload) + offset;

    Weight out        = bank;
    out.payload       = data;
    out.payload_bytes = matrix_bytes;
    out.qdata         = data;
    out.ndim          = 2;
    out.shape[0]      = bank.shape[1];
    out.shape[1]      = bank.shape[2];
    out.shape[2]      = 1;
    out.padded_shape[0] = bank.padded_shape[1];
    out.padded_shape[1] = bank.padded_shape[2];
    out.padded_shape[2] = 1;
    out.n               = bank.shape[1];
    out.k               = bank.shape[2];
    return out;
}

} // namespace ninfer::artifact
