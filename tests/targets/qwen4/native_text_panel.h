#pragma once

#include "artifact/reader.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::test::qwen4_sequence {

// Bounded source fixture, not a registered checkpoint or an alternative model loader.
struct TextPanel {
    artifact::Reader reader;
    std::vector<std::int32_t> tokens, global_rows, local_rows;
    std::vector<float> residual;
    std::uint64_t row_count = 0;

    explicit TextPanel(const std::string& root) : reader(root + "/qwen4-text-panel.ninfer") {
        if (reader.identity() != artifact::ArtifactIdentity{
                "qwen4/native-text-qualification", "nvidia-source-33"}) {
            throw std::runtime_error("invalid native text panel identity");
        }
        auto read_i32 = [&](const char* name, std::vector<std::uint64_t> shape) {
            const auto bytes = payload(name, artifact::NumericFormat::I32, shape);
            std::vector<std::int32_t> result(bytes.size()/4);
            for (std::size_t i=0; i<result.size(); ++i) {
                std::uint32_t bits=0;
                for (int b=0; b<4; ++b) { bits |= std::to_integer<std::uint32_t>(bytes[4*i+b]) << (8*b); }
                if (bits > INT32_MAX) { throw std::runtime_error("negative native panel index"); }
                result[i]=static_cast<std::int32_t>(bits);
            }
            return result;
        };
        tokens=read_i32("token.ids",{33});
        global_rows=read_i32("ple.global_rows",{33,16});
        local_rows=read_i32("ple.local_rows",{33,16});
        std::vector<std::int32_t> unique=global_rows;
        std::sort(unique.begin(),unique.end());
        unique.erase(std::unique(unique.begin(),unique.end()),unique.end());
        row_count=unique.size();
        for(std::size_t i=0;i<local_rows.size();++i) {
            if(global_rows[i]>=320001536 || local_rows[i]>=static_cast<std::int32_t>(row_count) ||
               unique[local_rows[i]]!=global_rows[i]) { throw std::runtime_error("native text row mapping mismatch"); }
        }
        const auto embedding=payload("token.embeddings",artifact::NumericFormat::BF16,{33,2560});
        residual.resize(33*10240);
        for(int t=0;t<33;++t) for(int branch=0;branch<4;++branch) for(int d=0;d<2560;++d) {
            const auto index=2*(t*2560+d);
            const auto bits=std::to_integer<unsigned>(embedding[index]) |
                (std::to_integer<unsigned>(embedding[index+1])<<8);
            residual[t*10240+branch*2560+d]=bf16_to_f32(bits);
        }
    }

    std::span<const std::byte> payload(const char* name,artifact::NumericFormat format,
                                      const std::vector<std::uint64_t>& shape) const {
        const auto* object=reader.find(name);
        const auto* tensor=object?std::get_if<artifact::TensorDescriptor>(object):nullptr;
        if(!tensor || tensor->format!=format || tensor->shape!=shape ||
           tensor->layout!=artifact::StorageLayout::ContiguousLeV1) {
            throw std::runtime_error("invalid native text panel field");
        }
        return reader.payload(*object).data;
    }
};

} // namespace ninfer::test::qwen4_sequence
