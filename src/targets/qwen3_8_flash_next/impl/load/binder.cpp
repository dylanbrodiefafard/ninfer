#include "load/binder.h"

#include "artifact/binder.h"
#include "artifact/reader.h"
#include "artifact/typed_binding.h"
#include "core/device.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <initializer_list>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <variant>
#include <vector>

namespace ninfer::targets::qwen3_8_flash_next {
namespace {

using artifact::NumericFormat;
using artifact::ObjectHandle;

constexpr std::array<std::string_view, 6> kFrontendResources = {
    "frontend/tokenizer.json",           "frontend/tokenizer_config.json",
    "frontend/chat_template.jinja",      "frontend/generation_config.json",
    "frontend/preprocessor_config.json", "frontend/video_preprocessor_config.json",
};

constexpr std::string_view kE2m1Records    = "text/routed_experts/e2m1_records";
constexpr std::string_view kF3Records      = "text/routed_experts/f3_records";
constexpr std::string_view kExpertCodecs   = "text/routed_experts/codecs";
constexpr std::string_view kF3Codebooks    = "text/routed_experts/f3_codebooks";
constexpr std::string_view kMatrixScales   = "text/routed_experts/matrix_scales";
constexpr std::int32_t kExpertMatrices     = 3; // gate, up, down
constexpr std::int32_t kRoutedExpertsTotal = kLayers * ops::kMoeExperts;
constexpr std::string_view kModelId        = "qwen3.8-flash-next";
// The one storage contract; it fixes every object's format (artifact doc, Section 3.1).
constexpr std::string_view kWeightsId = "f3-fp8-nvfp4";

constexpr NumericFormat kFp8 = NumericFormat::FP8_E4M3FN_ROW_BF16S;

std::uint64_t u64(std::int32_t value) { return static_cast<std::uint64_t>(value); }

std::uint32_t read_u32_le(std::span<const std::byte> bytes, std::size_t offset,
                          std::string_view label) {
    if (offset > bytes.size() || bytes.size() - offset < sizeof(std::uint32_t)) {
        throw artifact::ArtifactError(std::string(label) + ": word is outside the payload");
    }
    std::uint32_t value = 0;
    for (std::size_t i = 0; i < sizeof(std::uint32_t); ++i) {
        value |= std::to_integer<std::uint32_t>(bytes[offset + i]) << (8U * i);
    }
    return value;
}

// One quantized dense projection (NVFP4 or FP8 rows) or the BF16 embedding.
struct MatrixPlan {
    ObjectHandle object;
    NumericFormat format       = NumericFormat::BF16;
    std::int32_t rows          = 0;
    std::int32_t columns       = 0;
    float weight_scale_divisor = 0.0F;
};

// A projection of the profile's fixed `format`, resident on the device.
MatrixPlan bind_matrix(artifact::Binder& binder, const std::string& name, std::int32_t rows,
                       std::int32_t columns, NumericFormat format) {
    MatrixPlan plan{.format = format, .rows = rows, .columns = columns};
    plan.object = artifact::bind_device_tensor(binder, name, format, {u64(rows), u64(columns)});
    if (format == NumericFormat::NVFP4) {
        const std::array<std::uint64_t, 2> shape = {u64(rows), u64(columns)};
        const artifact::BlockScaleGeometry geometry =
            artifact::block_scale_geometry(NumericFormat::NVFP4, shape);
        plan.weight_scale_divisor = std::bit_cast<float>(
            read_u32_le(binder.payload(plan.object).data, geometry.weight_divisor_offset, name));
        if (!std::isfinite(plan.weight_scale_divisor) || plan.weight_scale_divisor <= 0.0F) {
            throw artifact::ArtifactError(name + ": weight divisor must be finite and positive");
        }
    }
    return plan;
}

Weight materialize_matrix(const artifact::MaterializedArtifact& materialized,
                          const MatrixPlan& plan) {
    if (plan.format != NumericFormat::NVFP4) {
        return artifact::materialized_weight(materialized, plan.object, plan.format, plan.rows,
                                             plan.columns);
    }
    const std::array<std::uint64_t, 2> shape = {u64(plan.rows), u64(plan.columns)};
    const artifact::BlockScaleGeometry geometry =
        artifact::block_scale_geometry(NumericFormat::NVFP4, shape);
    const auto* bytes = static_cast<const std::byte*>(materialized.device_data(plan.object));
    Weight out{};
    out.payload              = bytes;
    out.payload_bytes        = geometry.encoded_bytes;
    out.qtype                = QType::NVFP4;
    out.group_size           = 16;
    out.ndim                 = 2;
    out.qdata                = bytes;
    out.scales               = bytes + geometry.scale_plane_offset;
    out.n                    = plan.rows;
    out.k                    = plan.columns;
    out.group                = 16;
    out.layout               = QuantLayout::BlockScaleK16M128x4;
    out.scale_dtype          = DType::FP8_E4M3FN;
    out.shape[0]             = plan.rows;
    out.shape[1]             = plan.columns;
    out.padded_shape[0]      = plan.rows;
    out.padded_shape[1]      = plan.columns;
    out.weight_scale_divisor = plan.weight_scale_divisor;
    // The artifact stores no activation divisor: these projections run A16 routes only, which
    // never quantize the activation.
    out.input_scale_divisor = 0.0F;
    return out;
}

struct BlobPlan {
    ObjectHandle object;
    std::array<std::int32_t, 2> shape{1, 1}; // internal ne {fastest, slowest}
    NumericFormat format = NumericFormat::BF16;
};

// A contiguous BF16/FP32 tensor bound to the device. `shape` lists the artifact's rows-first
// dimensions (at most two); the internal Tensor reverses them.
BlobPlan bind_blob(artifact::Binder& binder, const std::string& name, NumericFormat format,
                   std::initializer_list<std::int32_t> shape) {
    BlobPlan plan{.format = format};
    if (shape.size() == 1) {
        plan.object = artifact::bind_device_tensor(binder, name, format, {u64(*shape.begin())});
        plan.shape  = {*shape.begin(), 1};
    } else if (shape.size() == 2) {
        const std::int32_t rows    = *shape.begin();
        const std::int32_t columns = *(shape.begin() + 1);
        plan.object = artifact::bind_device_tensor(binder, name, format, {u64(rows), u64(columns)});
        plan.shape  = {columns, rows};
    } else {
        throw std::logic_error("bind_blob: rank must be one or two");
    }
    return plan;
}

Tensor materialize_blob(const artifact::MaterializedArtifact& materialized, const BlobPlan& plan) {
    if (plan.shape[1] == 1) {
        return artifact::materialized_tensor(materialized, plan.object, plan.format,
                                             {plan.shape[0]});
    }
    return artifact::materialized_tensor(materialized, plan.object, plan.format,
                                         {plan.shape[0], plan.shape[1]});
}

struct HyperConnectionPlan {
    BlobPlan norm;
    MatrixPlan down;
    MatrixPlan up;
    BlobPlan write;
    bool has_write = false;
};

// A decoder block's read carries a write (block-inject) projection; the model's final read has
// none.
enum class HyperConnectionKind : std::uint8_t {
    BlockRead,
    FinalRead,
};

HyperConnectionPlan bind_hyper_connection(artifact::Binder& binder, const std::string& prefix,
                                          HyperConnectionKind kind) {
    const bool has_write = kind == HyperConnectionKind::BlockRead;
    HyperConnectionPlan plan{
        .norm =
            bind_blob(binder, prefix + "/norm", NumericFormat::BF16, {ops::kHyperConnectionFlat}),
        .down      = bind_matrix(binder, prefix + "/down", ops::kHyperConnectionRank,
                                 ops::kHyperConnectionFlat, kFp8),
        .up        = bind_matrix(binder, prefix + "/up", ops::kHyperConnectionFlat,
                                 ops::kHyperConnectionRank, kFp8),
        .write     = {},
        .has_write = has_write,
    };
    if (has_write) {
        plan.write = bind_blob(binder, prefix + "/write", NumericFormat::BF16,
                               {ops::kHyperConnectionStreams, ops::kHyperConnectionFlat});
    }
    return plan;
}

ops::HyperConnectionWeights
materialize_hyper_connection(const artifact::MaterializedArtifact& materialized,
                             const HyperConnectionPlan& plan) {
    ops::HyperConnectionWeights out{
        .norm  = materialize_blob(materialized, plan.norm),
        .down  = materialize_matrix(materialized, plan.down),
        .up    = materialize_matrix(materialized, plan.up),
        .write = Tensor(),
    };
    if (plan.has_write) { out.write = materialize_blob(materialized, plan.write); }
    return out;
}

struct GdnPlan {
    MatrixPlan query_key_value_z;
    BlobPlan convolution;
    BlobPlan gate_projection;
    BlobPlan a_log;
    BlobPlan dt_bias;
    BlobPlan norm;
    MatrixPlan output;
};

struct QsaPlan {
    MatrixPlan query_key_gate_value;
    BlobPlan query_norm;
    BlobPlan key_norm;
    BlobPlan index_query;
    BlobPlan index_key;
    BlobPlan index_query_norm;
    BlobPlan index_key_norm;
    MatrixPlan output;
};

struct LayerPlan {
    HyperConnectionPlan mixer_connection;
    HyperConnectionPlan moe_connection;
    BlobPlan moe_control;
    ObjectHandle shared_record;
    BlobPlan shared_matrix_scales;
};

struct PlePlan {
    BlobPlan key_projection;
    BlobPlan value_projection;
    BlobPlan key_norm;
    BlobPlan query_norm;
    BlobPlan conv_norm;
    BlobPlan convolution;
};

struct DflashLayerPlan {
    BlobPlan input_norm;
    MatrixPlan query_key_value;
    BlobPlan query_norm;
    BlobPlan key_norm;
    MatrixPlan attention_output;
    ObjectHandle attention_kernel;
    MatrixPlan attention_kernel_projection;
    BlobPlan post_attention_norm;
    MatrixPlan gate_up;
    MatrixPlan down;
    ObjectHandle mlp_kernel;
    MatrixPlan mlp_kernel_projection;
};

struct DflashPlan {
    MatrixPlan feature_projection;
    BlobPlan context_norm;
    std::array<DflashLayerPlan, 5> layers;
    BlobPlan final_norm;
    MatrixPlan hidden_projection;
    ObjectHandle predecessor_codebook;
    ObjectHandle successor_codebook;
};

struct ModelPlan {
    MatrixPlan embedding;
    MatrixPlan head;
    std::array<LayerPlan, kLayers> layers;
    std::array<GdnPlan, kGdnLayers> gdn;
    std::array<QsaPlan, kQsaLayers> qsa;
    PlePlan ple;
    HyperConnectionPlan final_connection;
    BlobPlan matrix_scales;
    // Retained raw bytes of kFrontendResources, in that order.
    std::array<ObjectHandle, kFrontendResources.size()> frontend{};
    // Host-read tables, validated and consumed during binding.
    NgramHashConstants ngram;
    std::vector<ops::MoeExpertCodec> codecs;
    std::array<std::array<std::uint32_t, kExpertMatrices>, kLayers> f3_books{};
    ObjectHandle e2m1_records;
    ObjectHandle f3_records;
    std::int64_t e2m1_count = 0;
    std::int64_t f3_count   = 0;
    std::optional<DflashPlan> dflash;
};

DflashPlan bind_dflash(artifact::Binder& binder) {
    constexpr auto fp4  = NumericFormat::NVFP4;
    constexpr auto bf16 = NumericFormat::BF16;
    DflashPlan plan;
    plan.feature_projection =
        bind_matrix(binder, "dflash/feature_projection", kHidden, 5 * kHidden, fp4);
    plan.context_norm = bind_blob(binder, "dflash/context_norm", bf16, {kHidden});
    for (std::size_t layer = 0; layer < plan.layers.size(); ++layer) {
        const auto prefix = "dflash/layers/" + std::to_string(layer) + "/";
        auto& target      = plan.layers.at(layer);
        target.input_norm = bind_blob(binder, prefix + "input_norm", bf16, {kHidden});
        target.query_key_value =
            bind_matrix(binder, prefix + "attention/query_key_value", 6144, kHidden, fp4);
        target.query_norm = bind_blob(binder, prefix + "attention/query_norm", bf16, {128});
        target.key_norm   = bind_blob(binder, prefix + "attention/key_norm", bf16, {128});
        target.attention_output =
            bind_matrix(binder, prefix + "attention/output", kHidden, 4096, fp4);
        target.attention_kernel = artifact::bind_device_tensor(
            binder, prefix + "attention_conv/base_kernel", bf16, {u64(kHidden), 2, 2});
        target.attention_kernel_projection =
            bind_matrix(binder, prefix + "attention_conv/kernel_projection", 640, kHidden, fp4);
        target.post_attention_norm =
            bind_blob(binder, prefix + "post_attention_norm", bf16, {kHidden});
        target.gate_up    = bind_matrix(binder, prefix + "mlp/gate_up", 15360, kHidden, fp4);
        target.down       = bind_matrix(binder, prefix + "mlp/down", kHidden, 7680, fp4);
        target.mlp_kernel = artifact::bind_device_tensor(binder, prefix + "mlp_conv/base_kernel",
                                                         bf16, {u64(kHidden), 2, 2});
        target.mlp_kernel_projection =
            bind_matrix(binder, prefix + "mlp_conv/kernel_projection", 640, kHidden, fp4);
    }
    plan.final_norm = bind_blob(binder, "dflash/final_norm", bf16, {kHidden});
    plan.hidden_projection =
        bind_matrix(binder, "dflash/selector/hidden_projection", 256, kHidden, fp4);
    plan.predecessor_codebook = artifact::bind_device_tensor(
        binder, "dflash/selector/predecessor_codebook", bf16, {256, u64(kVocabRows)});
    plan.successor_codebook = artifact::bind_device_tensor(
        binder, "dflash/selector/successor_codebook", bf16, {256, u64(kVocabRows)});
    return plan;
}

DflashWeights materialize_dflash(const artifact::MaterializedArtifact& backing,
                                 const DflashPlan& plan) {
    DflashWeights weights;
    weights.feature_projection = materialize_matrix(backing, plan.feature_projection);
    weights.context_norm       = materialize_blob(backing, plan.context_norm);
    for (std::size_t layer = 0; layer < plan.layers.size(); ++layer) {
        const auto& source      = plan.layers.at(layer);
        auto& target            = weights.layers.at(layer);
        target.input_norm       = materialize_blob(backing, source.input_norm);
        target.query_key_value  = materialize_matrix(backing, source.query_key_value);
        target.query_norm       = materialize_blob(backing, source.query_norm);
        target.key_norm         = materialize_blob(backing, source.key_norm);
        target.attention_output = materialize_matrix(backing, source.attention_output);
        target.attention_conv   = {artifact::materialized_tensor(backing, source.attention_kernel,
                                                                 NumericFormat::BF16,
                                                                 {kHidden, 2, 2}),
                                   materialize_matrix(backing, source.attention_kernel_projection)};
        target.post_attention_norm = materialize_blob(backing, source.post_attention_norm);
        target.gate_up             = materialize_matrix(backing, source.gate_up);
        target.down                = materialize_matrix(backing, source.down);
        target.mlp_conv = {artifact::materialized_tensor(backing, source.mlp_kernel,
                                                         NumericFormat::BF16, {kHidden, 2, 2}),
                           materialize_matrix(backing, source.mlp_kernel_projection)};
    }
    weights.final_norm           = materialize_blob(backing, plan.final_norm);
    weights.hidden_projection    = materialize_matrix(backing, plan.hidden_projection);
    weights.predecessor_codebook = artifact::materialized_tensor(
        backing, plan.predecessor_codebook, NumericFormat::BF16, {256, kVocabRows});
    weights.successor_codebook = artifact::materialized_tensor(
        backing, plan.successor_codebook, NumericFormat::BF16, {256, kVocabRows});
    return weights;
}

// I32 [n,2] (low word, high word) little-endian words of a host-read int64 [n] tensor.
template <std::size_t N>
std::array<std::int64_t, N> read_i64_words(artifact::Binder& binder, std::string_view name) {
    const ObjectHandle handle = artifact::bind_tensor(binder, name, NumericFormat::I32, {N, 2},
                                                      artifact::TensorPlacement::ValidateOnly);
    const std::span<const std::byte> bytes = binder.payload(handle).data;
    std::array<std::int64_t, N> out{};
    for (std::size_t i = 0; i < N; ++i) {
        const std::uint64_t low  = read_u32_le(bytes, 8 * i, name);
        const std::uint64_t high = read_u32_le(bytes, 8 * i + 4, name);
        out.at(i)                = static_cast<std::int64_t>(low | (high << 32U));
    }
    return out;
}

NgramHashConstants bind_ngram_constants(artifact::Binder& binder, const std::string& prefix) {
    NgramHashConstants constants;
    const std::array<std::int64_t, 3> multipliers =
        read_i64_words<3>(binder, prefix + "/ngram_hash_multipliers");
    for (std::size_t i = 0; i < multipliers.size(); ++i) {
        constants.multipliers.at(i) = static_cast<std::uint64_t>(multipliers.at(i));
    }
    constants.head_offsets = read_i64_words<kNgramHeads>(binder, prefix + "/ngram_head_offsets");
    constants.head_sizes   = read_i64_words<kNgramHeads>(binder, prefix + "/ngram_head_sizes");
    for (const std::int64_t size : constants.head_sizes) {
        if (size <= 0) {
            throw artifact::ArtifactError(prefix + ": n-gram head size must be positive");
        }
    }
    return constants;
}

void bind_routed_tables(artifact::Binder& binder, ModelPlan& plan) {
    const ObjectHandle codecs = artifact::bind_tensor(binder, kExpertCodecs, NumericFormat::I32,
                                                      {u64(kLayers), u64(ops::kMoeExperts)},
                                                      artifact::TensorPlacement::ValidateOnly);
    const std::span<const std::byte> codec_bytes = binder.payload(codecs).data;
    plan.codecs.reserve(static_cast<std::size_t>(kRoutedExpertsTotal));
    for (std::int32_t i = 0; i < kRoutedExpertsTotal; ++i) {
        const auto value = static_cast<std::int32_t>(
            read_u32_le(codec_bytes, static_cast<std::size_t>(i) * 4, kExpertCodecs));
        // The codec map stores ops::MoeExpertCodec values.
        if (value == static_cast<std::int32_t>(ops::MoeExpertCodec::Nvfp4G16)) {
            plan.codecs.push_back(ops::MoeExpertCodec::Nvfp4G16);
            ++plan.e2m1_count;
        } else if (value == static_cast<std::int32_t>(ops::MoeExpertCodec::F3G32)) {
            plan.codecs.push_back(ops::MoeExpertCodec::F3G32);
            ++plan.f3_count;
        } else {
            throw artifact::ArtifactError(std::string(kExpertCodecs) + ": unknown codec " +
                                          std::to_string(value));
        }
    }

    const ObjectHandle books = artifact::bind_tensor(binder, kF3Codebooks, NumericFormat::I32,
                                                     {u64(kLayers), u64(kExpertMatrices)},
                                                     artifact::TensorPlacement::ValidateOnly);
    const std::span<const std::byte> book_bytes = binder.payload(books).data;
    for (std::int32_t layer = 0; layer < kLayers; ++layer) {
        for (std::int32_t matrix = 0; matrix < kExpertMatrices; ++matrix) {
            plan.f3_books.at(static_cast<std::size_t>(layer)).at(static_cast<std::size_t>(matrix)) =
                read_u32_le(book_bytes,
                            static_cast<std::size_t>(layer * kExpertMatrices + matrix) * 4,
                            kF3Codebooks);
        }
    }

    const ObjectHandle scales = binder.require_tensor(
        kMatrixScales, NumericFormat::FP32, artifact::StorageLayout::ContiguousLeV1,
        std::array<std::uint64_t, 3>{u64(kLayers), u64(ops::kMoeExperts), u64(kExpertMatrices)});
    binder.materialize_on_device(scales);
    plan.matrix_scales = BlobPlan{.object = scales,
                                  .shape  = {kExpertMatrices * ops::kMoeExperts, kLayers},
                                  .format = NumericFormat::FP32};

    // Records are read from the file into the host store, not materialized.
    plan.e2m1_records = binder.require_tensor(
        kE2m1Records, NumericFormat::E2M1G16_E4M3FN, artifact::StorageLayout::MoeExpertTilesV1,
        std::array<std::uint64_t, 3>{static_cast<std::uint64_t>(plan.e2m1_count), u64(kHidden),
                                     u64(ops::kMoeIntermediate)});
    binder.validate_only(plan.e2m1_records);
    plan.f3_records = binder.require_tensor(
        kF3Records, NumericFormat::F3G32_E4M3FN, artifact::StorageLayout::MoeExpertTilesV1,
        std::array<std::uint64_t, 3>{static_cast<std::uint64_t>(plan.f3_count), u64(kHidden),
                                     u64(ops::kMoeIntermediate)});
    binder.validate_only(plan.f3_records);
}

// The BF16 token embedding stays in device-mapped pinned host memory: each token reads one 5 KiB
// row over PCIe, so the table costs no VRAM.
MatrixPlan bind_embedding(artifact::Binder& binder) {
    return MatrixPlan{.object = artifact::bind_tensor(
                          binder, "text/token_embedding", NumericFormat::BF16,
                          {u64(kVocabRows), u64(kHidden)}, artifact::TensorPlacement::MappedHost),
                      .format  = NumericFormat::BF16,
                      .rows    = kVocabRows,
                      .columns = kHidden};
}

ModelPlan bind_model(artifact::Binder& binder) {
    ModelPlan plan;
    // The frontend resources are retained on the host for the shared Qwen3.6 frontend.
    for (std::size_t i = 0; i < kFrontendResources.size(); ++i) {
        plan.frontend.at(i) = artifact::bind_raw_resource(binder, kFrontendResources.at(i));
    }
    plan.embedding = bind_embedding(binder);
    for (std::int32_t layer = 0; layer < kLayers; ++layer) {
        const std::string prefix = "text/layers/" + std::to_string(layer);
        LayerPlan& layer_plan    = plan.layers.at(static_cast<std::size_t>(layer));
        if (layer == kPleLayer) {
            const std::string ple = prefix + "/ple";
            plan.ple              = PlePlan{
                .key_projection   = bind_blob(binder, ple + "/key_projection", NumericFormat::BF16,
                                              {kPleKeyRows, kHidden}),
                .value_projection = bind_blob(binder, ple + "/value_projection",
                                              NumericFormat::BF16, {kPleValueRows, kHidden}),
                .key_norm =
                    bind_blob(binder, ple + "/key_norm", NumericFormat::BF16, {ops::kPleChannels}),
                .query_norm = bind_blob(binder, ple + "/query_norm", NumericFormat::BF16,
                                        {ops::kPleChannels}),
                .conv_norm =
                    bind_blob(binder, ple + "/conv_norm", NumericFormat::BF16, {ops::kPleChannels}),
                // Artifact [10240,4], taps fastest: Tensor ne {4,10240} as ple_inject reads it.
                .convolution = bind_blob(binder, ple + "/convolution", NumericFormat::BF16,
                                         {ops::kPleChannels, ops::kPleConvWidth}),
            };
            plan.ngram = bind_ngram_constants(binder, ple);
        }
        layer_plan.mixer_connection = bind_hyper_connection(
            binder, prefix + "/mixer_hyper_connection", HyperConnectionKind::BlockRead);
        if (is_qsa_layer(layer)) {
            const std::string a                                     = prefix + "/attention";
            plan.qsa.at(static_cast<std::size_t>(qsa_index(layer))) = QsaPlan{
                .query_key_gate_value =
                    bind_matrix(binder, a + "/query_key_gate_value", kQsaProjectionRows, kHidden,
                                NumericFormat::NVFP4),
                .query_norm =
                    bind_blob(binder, a + "/query_norm", NumericFormat::BF16, {ops::kQsaHeadDim}),
                .key_norm =
                    bind_blob(binder, a + "/key_norm", NumericFormat::BF16, {ops::kQsaHeadDim}),
                .index_query      = bind_blob(binder, a + "/index_query", NumericFormat::BF16,
                                              {kQsaIndexQueryDim, kHidden}),
                .index_key        = bind_blob(binder, a + "/index_key", NumericFormat::BF16,
                                              {ops::kQsaIndexHeadDim, kHidden}),
                .index_query_norm = bind_blob(binder, a + "/index_query_norm", NumericFormat::BF16,
                                              {ops::kQsaIndexHeadDim}),
                .index_key_norm   = bind_blob(binder, a + "/index_key_norm", NumericFormat::BF16,
                                              {ops::kQsaIndexHeadDim}),
                .output = bind_matrix(binder, a + "/output", kHidden, kMixerOutputColumns, kFp8),
            };
        } else {
            const std::string g                                     = prefix + "/gdn";
            plan.gdn.at(static_cast<std::size_t>(gdn_index(layer))) = GdnPlan{
                .query_key_value_z = bind_matrix(binder, g + "/query_key_value_z",
                                                 kGdnProjectionRows, kHidden, kFp8),
                // Artifact [4,10240], tap-major: Tensor ne {10240,4} as causal_conv1d reads it.
                .convolution     = bind_blob(binder, g + "/convolution", NumericFormat::BF16,
                                             {kGdnConvKernel, kGdnConvChannels}),
                .gate_projection = bind_blob(binder, g + "/a_b_projection", NumericFormat::BF16,
                                             {kGdnGateRows, kHidden}),
                .a_log   = bind_blob(binder, g + "/a_log", NumericFormat::FP32, {kGdnValueHeads}),
                .dt_bias = bind_blob(binder, g + "/dt_bias", NumericFormat::FP32, {kGdnValueHeads}),
                .norm    = bind_blob(binder, g + "/norm", NumericFormat::BF16, {kGdnHeadDim}),
                .output  = bind_matrix(binder, g + "/output", kHidden, kMixerOutputColumns, kFp8),
            };
        }
        layer_plan.moe_connection = bind_hyper_connection(binder, prefix + "/moe_hyper_connection",
                                                          HyperConnectionKind::BlockRead);
        const std::string moe     = prefix + "/moe";
        layer_plan.moe_control    = bind_blob(binder, moe + "/control", NumericFormat::BF16,
                                              {ops::kMoeControlRows, kHidden});
        // The shared expert is one NVFP4 record in the routed record layout, resident on the
        // device, with its three FP32 matrix multipliers.
        layer_plan.shared_record = binder.require_tensor(
            moe + "/shared_expert", NumericFormat::E2M1G16_E4M3FN,
            artifact::StorageLayout::MoeExpertTilesV1,
            std::array<std::uint64_t, 3>{1, u64(kHidden), u64(ops::kMoeIntermediate)});
        binder.materialize_on_device(layer_plan.shared_record);
        layer_plan.shared_matrix_scales = bind_blob(binder, moe + "/shared_matrix_scales",
                                                    NumericFormat::FP32, {kExpertMatrices});
    }
    plan.final_connection = bind_hyper_connection(binder, "text/final_hyper_connection",
                                                  HyperConnectionKind::FinalRead);
    plan.head             = bind_matrix(binder, "text/output_head", kVocabRows, kHidden, kFp8);
    bind_routed_tables(binder, plan);
    if (binder.contains("dflash/feature_projection")) { plan.dflash = bind_dflash(binder); }
    return plan;
}

// Copies every routed record from the artifact into the host store. A records tensor holds the
// experts of its codec in layer-major order. Reading is spread over threads because the store is
// tens of gigabytes; write_record of distinct records is thread-safe. The calling thread reports
// the record bytes written so far to `progress` (phase "experts") until every worker has finished.
void fill_expert_store(const artifact::Binder& binder, const ModelPlan& plan,
                       ExpertHostStore& store, const artifact::LoadProgress* progress) {
    struct Source {
        std::span<const std::byte> bytes;
        std::size_t record_bytes = 0;
        std::int64_t next        = 0;
    };

    std::array<Source, 2> sources{
        Source{
            binder.payload(plan.e2m1_records).data,
            static_cast<std::size_t>(ops::moe_expert_record_bytes(ops::MoeExpertCodec::Nvfp4G16))},
        Source{binder.payload(plan.f3_records).data,
               static_cast<std::size_t>(ops::moe_expert_record_bytes(ops::MoeExpertCodec::F3G32))},
    };

    struct Job {
        std::int32_t layer;
        std::int32_t expert;
        std::span<const std::byte> record;
    };

    std::vector<Job> jobs;
    jobs.reserve(plan.codecs.size());
    for (std::int32_t i = 0; i < kRoutedExpertsTotal; ++i) {
        const ops::MoeExpertCodec codec = plan.codecs.at(static_cast<std::size_t>(i));
        Source& source                  = sources.at(static_cast<std::size_t>(codec));
        const std::size_t offset = static_cast<std::size_t>(source.next) * source.record_bytes;
        if (offset + source.record_bytes > source.bytes.size()) {
            throw artifact::ArtifactError("routed-expert records are shorter than their codec map");
        }
        jobs.push_back(Job{.layer  = i / ops::kMoeExperts,
                           .expert = i % ops::kMoeExperts,
                           .record = source.bytes.subspan(offset, source.record_bytes)});
        ++source.next;
    }
    for (const Source& source : sources) {
        if (static_cast<std::size_t>(source.next) * source.record_bytes != source.bytes.size()) {
            throw artifact::ArtifactError("routed-expert record count differs from the codec map");
        }
    }

    constexpr std::size_t kThreads   = 8;
    constexpr auto kProgressInterval = std::chrono::milliseconds(250);
    const bool report                = progress != nullptr && progress->callback;
    std::atomic<std::uint64_t> written{0};
    std::mutex state_mutex;
    std::condition_variable finished_cv;
    std::size_t finished = 0;
    std::exception_ptr failure;
    {
        // jthreads join on every exit path, including a failed thread start or a throwing
        // progress callback.
        std::vector<std::jthread> threads;
        threads.reserve(kThreads);
        for (std::size_t t = 0; t < kThreads; ++t) {
            threads.emplace_back([&, t] {
                try {
                    for (std::size_t j = t; j < jobs.size(); j += kThreads) {
                        store.write_record(jobs[j].layer, jobs[j].expert, jobs[j].record);
                        written.fetch_add(jobs[j].record.size(), std::memory_order_relaxed);
                    }
                } catch (...) {
                    const std::lock_guard<std::mutex> lock(state_mutex);
                    if (!failure) { failure = std::current_exception(); }
                }
                {
                    const std::lock_guard<std::mutex> lock(state_mutex);
                    ++finished;
                }
                finished_cv.notify_one();
            });
        }
        std::uint64_t total = 0;
        for (const Job& job : jobs) { total += job.record.size(); }
        std::unique_lock<std::mutex> lock(state_mutex);
        // At least one report, the last one after every worker finished.
        do {
            (void)finished_cv.wait_for(lock, kProgressInterval,
                                       [&] { return finished == threads.size(); });
            if (report) {
                lock.unlock();
                progress->callback("experts", written.load(std::memory_order_relaxed), total);
                lock.lock();
            }
        } while (finished != threads.size());
    }
    if (failure) { std::rethrow_exception(failure); }
}

} // namespace

LoadedModel::LoadedModel(const artifact::Reader& reader, DeviceContext& device,
                         artifact::LoadProgress* progress) {
    const artifact::ArtifactIdentity& identity = reader.identity();
    if (identity.model_id != kModelId || identity.weights_id != kWeightsId) {
        throw artifact::ArtifactError("artifact is " + identity.model_id + "/" +
                                      identity.weights_id + ", not " + std::string(kModelId) + "/" +
                                      std::string(kWeightsId));
    }
    artifact::Binder binder(reader);
    const ModelPlan plan     = bind_model(binder);
    ngram_                   = plan.ngram;
    dense_                   = artifact::materialize(reader, binder.finish(), device, progress);
    const auto take_resource = [&](std::size_t index) {
        const std::vector<std::byte> bytes = dense_.take_resource_bytes(plan.frontend.at(index));
        return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    };
    // kFrontendResources order.
    frontend_ = qwen3_6::FrontendResources{
        .tokenizer_json                 = take_resource(0),
        .tokenizer_config_json          = take_resource(1),
        .chat_template_jinja            = take_resource(2),
        .generation_config_json         = take_resource(3),
        .preprocessor_config_json       = take_resource(4),
        .video_preprocessor_config_json = take_resource(5),
    };

    experts_ = std::make_unique<ExpertHostStore>(plan.codecs, kExpertHostChunkBytes);
    fill_expert_store(binder, plan, *experts_, progress);

    const auto& d   = dense_;
    view_.embedding = materialize_matrix(d, plan.embedding);
    view_.head      = materialize_matrix(d, plan.head);
    if (plan.dflash) { view_.dflash = materialize_dflash(d, *plan.dflash); }
    view_.ple = PleModelWeights{
        .key_projection   = materialize_blob(d, plan.ple.key_projection),
        .value_projection = materialize_blob(d, plan.ple.value_projection),
        .inject =
            ops::PleWeights{
                .norm_key   = materialize_blob(d, plan.ple.key_norm),
                .norm_query = materialize_blob(d, plan.ple.query_norm),
                .norm_conv  = materialize_blob(d, plan.ple.conv_norm),
                .conv       = materialize_blob(d, plan.ple.convolution),
            },
    };
    view_.final_connection     = materialize_hyper_connection(d, plan.final_connection);
    const Tensor matrix_scales = materialize_blob(d, plan.matrix_scales);
    for (std::size_t i = 0; i < view_.gdn.size(); ++i) {
        const GdnPlan& g = plan.gdn.at(i);
        view_.gdn.at(i)  = GdnLayerWeights{
            .query_key_value_z = materialize_matrix(d, g.query_key_value_z),
            .convolution       = materialize_blob(d, g.convolution),
            .gate_projection   = materialize_blob(d, g.gate_projection),
            .a_log             = materialize_blob(d, g.a_log),
            .dt_bias           = materialize_blob(d, g.dt_bias),
            .norm              = materialize_blob(d, g.norm),
            .output            = materialize_matrix(d, g.output),
        };
    }
    for (std::size_t i = 0; i < view_.qsa.size(); ++i) {
        const QsaPlan& q = plan.qsa.at(i);
        view_.qsa.at(i)  = QsaLayerWeights{
            .query_key_gate_value = materialize_matrix(d, q.query_key_gate_value),
            .query_norm           = materialize_blob(d, q.query_norm),
            .key_norm             = materialize_blob(d, q.key_norm),
            .index_query          = materialize_blob(d, q.index_query),
            .index_key            = materialize_blob(d, q.index_key),
            .index_query_norm     = materialize_blob(d, q.index_query_norm),
            .index_key_norm       = materialize_blob(d, q.index_key_norm),
            .output               = materialize_matrix(d, q.output),
        };
    }
    constexpr std::size_t kScaleLayerBytes = sizeof(float) * kExpertMatrices * ops::kMoeExperts;
    for (std::size_t layer = 0; layer < view_.layers.size(); ++layer) {
        const LayerPlan& l     = plan.layers.at(layer);
        view_.layers.at(layer) = LayerWeights{
            .mixer_connection = materialize_hyper_connection(d, l.mixer_connection),
            .moe_connection   = materialize_hyper_connection(d, l.moe_connection),
            .moe_control      = materialize_blob(d, l.moe_control),
            .experts =
                ops::MoeExpertBank{
                    // The expert-cache table column is the residency's (runtime/forward.cpp).
                    .expert_table  = Tensor(),
                    .matrix_scales = Tensor(static_cast<std::byte*>(matrix_scales.data) +
                                                layer * kScaleLayerBytes,
                                            DType::FP32, {kExpertMatrices, ops::kMoeExperts}),
                    .f3_books      = plan.f3_books.at(layer),
                    .shared_record = Tensor(d.device_data(l.shared_record), DType::U8,
                                            {static_cast<std::int32_t>(ops::kMoeNvfp4RecordBytes)}),
                    .shared_matrix_scales = materialize_blob(d, l.shared_matrix_scales),
                },
        };
    }
}

} // namespace ninfer::targets::qwen3_8_flash_next
