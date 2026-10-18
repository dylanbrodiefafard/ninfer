#include "artifact/reader.h"
#include "ninfer/engine.h"
#include "artifact_fixture.h"
#include "targets/qwen3_6_27b/impl/load/bindings.h"

#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace artifact = ninfer::artifact;
namespace target   = ninfer::targets::qwen3_6_27b::detail;
using Json         = nlohmann::json;
using Backend      = ninfer::SpeculativeBackend;

void exercise(artifact::NumericFormat format, artifact::StorageLayout layout) {
    Json directory = {
        {"identity", {{"model_id", "qwen3.8-27b"}, {"weights_id", "nvfp4"}}},
        {"objects", Json::array()},
    };
    std::uint64_t offset = 0;
    const auto add       = [&](std::string_view name, std::vector<std::uint64_t> shape,
                               artifact::NumericFormat dtype, artifact::StorageLayout storage) {
        offset           = (offset + 255) / 256 * 256;
        const auto bytes = artifact::tensor_encoded_size(storage, dtype, shape);
        directory["objects"].push_back({{"name", name},
                                        {"kind", "tensor"},
                                        {"shape", shape},
                                        {"format", artifact::format_name(dtype)},
                                        {"layout", artifact::layout_name(storage)},
                                        {"offset", offset},
                                        {"bytes", bytes}});
        offset += bytes;
    };
    const auto bf16 = [&](std::string_view name) {
        add(name, {128}, artifact::NumericFormat::BF16, artifact::StorageLayout::ContiguousLeV1);
    };
    for (const auto name :
         {"text/token_embedding", "text/final_norm", "text/output_head", "text/draft_head",
          "text/draft_head_token_ids", "vision/patch_embedding", "mtp/embedding_norm",
          "mtp/hidden_norm", "mtp/layer/input_norm", "mtp/layer/attention/key_norm",
          "mtp/layer/attention/query_norm", "mtp/layer/mlp/down", "mtp/final_norm",
          "dflash/context_norm", "dflash/layers/0/attention/key_norm",
          "dflash/layers/0/attention/query_norm", "dflash/layers/0/mlp/down", "dflash/final_norm",
          "dflash/selector/predecessor_codebook"}) {
        bf16(name);
    }
    // The output-row boundaries match the real projections; K=128 keeps the fixture small
    // while exercising complete NVFP4 scale tiles and row-split groups.
    add("mtp/input_projection", {128, 128}, format, layout);
    add("mtp/layer/attention/query_key_gate_value", {14336, 128}, format, layout);
    add("dflash/feature_projection", {128, 128}, format, layout);
    add("dflash/layers/0/attention/query_key_value", {6144, 128}, format, layout);
    auto fixture = ninfer::test::artifact_fixture::write_fixture(directory, "cache-dependencies");
    const auto digest = [&](Backend backend) {
        const artifact::Reader reader(fixture.path);
        return reader.tensor_content_identity(
            target::cache_tensor_selections(reader, {.vision = false, .speculative = backend}));
    };
    constexpr std::array backends{Backend::None, Backend::DFlash};
    const std::array baseline{digest(backends[0]), digest(backends[1])};
    const auto mutation = [&](std::string_view name, std::uint64_t relative_offset,
                              std::array<bool, 2> affects) {
        std::uint64_t absolute = 0;
        {
            const artifact::Reader reader(fixture.path);
            const auto payload = reader.payload(name);
            if (relative_offset >= payload.data.size()) {
                throw std::runtime_error("bad fixture mutation");
            }
            absolute = payload.absolute_offset + relative_offset;
        }
        char original    = 0;
        const auto write = [&](char byte) {
            std::fstream file(fixture.path, std::ios::in | std::ios::out | std::ios::binary);
            file.seekp(static_cast<std::streamoff>(absolute));
            file.put(byte);
            if (!file) { throw std::runtime_error("cannot mutate content fixture"); }
        };
        {
            std::ifstream file(fixture.path, std::ios::binary);
            file.seekg(static_cast<std::streamoff>(absolute));
            file.get(original);
            if (!file) { throw std::runtime_error("cannot read content fixture"); }
        }
        write(static_cast<char>(original ^ 1));
        for (std::size_t i = 0; i < backends.size(); ++i) {
            if ((digest(backends[i]) != baseline[i]) != affects[i]) {
                throw std::runtime_error("incorrect retained-state dependency: " +
                                         std::string(name));
            }
        }
        write(original);
    };
    constexpr std::array all{true, true};
    constexpr std::array none{false, false};
    constexpr std::array dflash{false, true};
    mutation("text/token_embedding", 0, all);
    mutation("text/final_norm", 0, all);
    mutation("vision/patch_embedding", 0, all);
    for (const auto name :
         {"text/output_head", "text/draft_head", "text/draft_head_token_ids",
          "mtp/layer/attention/query_norm", "mtp/layer/mlp/down", "mtp/final_norm",
          "dflash/layers/0/attention/query_norm", "dflash/layers/0/mlp/down", "dflash/final_norm",
          "dflash/selector/predecessor_codebook"}) {
        mutation(name, 0, none);
    }
    for (const auto name : {"mtp/input_projection", "mtp/embedding_norm", "mtp/hidden_norm",
                            "mtp/layer/input_norm", "mtp/layer/attention/key_norm"}) {
        mutation(name, 0, none);
    }
    for (const auto name : {"dflash/feature_projection", "dflash/context_norm",
                            "dflash/layers/0/attention/key_norm"}) {
        mutation(name, 0, dflash);
    }
    for (const auto name : {"mtp/layer/attention/query_key_gate_value",
                            "dflash/layers/0/attention/query_key_value"}) {
        const bool is_mtp         = std::string_view(name).starts_with("mtp/");
        const std::uint64_t key   = is_mtp ? 6144 : 4096;
        const std::uint64_t value = is_mtp ? 13312 : 5120;
        const auto active         = is_mtp ? none : dflash;
        const std::array<std::uint64_t, 2> shape{is_mtp ? 14336ULL : 6144ULL, 128};
        std::uint64_t code_stride = 0, scale_offset = 0, scale_stride = 0;
        if (format == artifact::NumericFormat::NVFP4) {
            const auto geometry = artifact::block_scale_geometry(format, shape);
            code_stride         = 64;
            scale_offset        = geometry.scale_plane_offset;
            scale_stride        = 8;
            mutation(name, geometry.weight_divisor_offset, active);
        } else {
            const auto geometry = artifact::row_split_geometry(format, shape);
            code_stride         = format == artifact::NumericFormat::W8G32_F16S ? 128 : 64;
            scale_offset        = geometry.scale_plane_offset;
            scale_stride        = format == artifact::NumericFormat::W8G32_F16S ? 8 : 4;
        }
        mutation(name, 0, none);
        mutation(name, scale_offset, none);
        mutation(name, key * code_stride, active);
        mutation(name, value * code_stride, active);
        mutation(name, scale_offset + key * scale_stride, active);
        mutation(name, scale_offset + value * scale_stride, active);
    }
}

} // namespace

int main() {
    try {
        bool disabled = false;
        try {
            ninfer::EngineOptions options;
            options.speculative.backend = Backend::Mtp;
            ninfer::Engine engine(options);
        } catch (const std::invalid_argument& error) {
            disabled = std::string_view(error.what()).find("MTP execution is disabled") !=
                       std::string_view::npos;
        }
        if (!disabled) {
            throw std::runtime_error("Engine did not reject MTP before device/artifact setup");
        }
        exercise(artifact::NumericFormat::NVFP4, artifact::StorageLayout::BlockScaleK16M128x4V1);
        exercise(artifact::NumericFormat::W8G32_F16S, artifact::StorageLayout::RowSplitK128V1);
        exercise(artifact::NumericFormat::Q4G64_F16S, artifact::StorageLayout::RowSplitK128V1);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
