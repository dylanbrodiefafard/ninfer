#pragma once

#include "artifact/reader.h"
#include "ops/native_projection_fixture.h"

#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::test::qwen4_native {

// Test-only, exact source payload owner. Derived execution views never alter the artifact.
struct Bf16Source {
    artifact::Reader reader;
    std::string prefix;

    explicit Bf16Source(const std::string& endpoint_path) : reader(endpoint_path) {
        if (reader.identity() != artifact::ArtifactIdentity{
                "qwen4/native-endpoint-qualification", "nvidia-bf16-source"}) {
            throw std::invalid_argument("expected native BF16 endpoint fixture");
        }
    }

    Bf16Source(const std::string& path, int layer)
        : reader(path), prefix("model.language_model.layers." + std::to_string(layer) + ".") {
        if (reader.identity() != artifact::ArtifactIdentity{
                "qwen4/native-layer-qualification", "nvidia-nvfp4-source"}) {
            throw std::invalid_argument("expected native source layer fixture");
        }
    }

    std::vector<std::uint16_t> bits(const std::string& name,
                                     std::vector<std::uint64_t> shape) const {
        const auto* object = reader.find(prefix + name);
        const auto* descriptor = object ? std::get_if<artifact::TensorDescriptor>(object) : nullptr;
        if (!descriptor || descriptor->format != artifact::NumericFormat::BF16 ||
            descriptor->layout != artifact::StorageLayout::ContiguousLeV1 || descriptor->shape != shape) {
            throw std::invalid_argument("invalid native BF16 source role: " + name);
        }
        const auto payload = reader.payload(*object);
        std::vector<std::uint16_t> result(payload.data.size() / 2);
        for (std::size_t i = 0; i < result.size(); ++i) {
            result[i] = std::to_integer<unsigned>(payload.data[2*i]) |
                        (std::to_integer<unsigned>(payload.data[2*i+1]) << 8);
        }
        return result;
    }

    std::vector<float> values(const std::string& name, std::vector<std::uint64_t> shape) const {
        const auto source = bits(name, std::move(shape));
        std::vector<float> result(source.size());
        std::transform(source.begin(), source.end(), result.begin(), bf16_to_f32);
        return result;
    }
};

inline quantized_weight::PackedWeight bf16_matrix(std::span<const std::uint16_t> bits,
                                                  int rows, int columns) {
    if (bits.size() != static_cast<std::size_t>(rows) * columns) {
        throw std::invalid_argument("native BF16 matrix extent");
    }
    quantized_weight::PackedWeight result;
    result.weight = direct_bf16_weight::HostWeight{rows, columns,
        std::vector<std::uint16_t>(bits.begin(), bits.end())}.device_weight(nullptr);
    result.payload.resize(bits.size() * 2);
    result.code_plane_bytes = result.payload.size();
    for (std::size_t i = 0; i < bits.size(); ++i) {
        quantized_weight::detail::store_u16_le(result.payload, i * 2, bits[i]);
    }
    return result;
}


} // namespace ninfer::test::qwen4_native
