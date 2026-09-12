#include "artifact/reader.h"
#include "ops/linear/linear_test_common.h"

#include <array>
#include <cstdlib>
#include <cstring>
#include <iostream>

namespace {
using namespace ninfer;
namespace qualification = ninfer::test::linear;
const artifact::Reader* source;
const artifact::TensorDescriptor* tensor;

struct MatrixCase { const char* name; int n; int k; };
constexpr std::array<MatrixCase, 12> matrices{{
    {"blk.0.attn_qkv.weight", 10240, 2560},
    {"blk.0.attn_gate.weight", 6144, 2560},
    {"blk.0.ssm_out.weight", 2560, 6144},
    {"blk.2.attn_qkv.weight", 10240, 2560},
    {"blk.3.attn_q.weight", 12288, 2560},
    {"blk.3.attn_k.weight", 512, 2560},
    {"blk.0.hc_attn_down.weight", 320, 10240},
    {"blk.0.hc_attn_up.weight", 10240, 320},
    {"blk.0.ffn_gate_exps.weight/expert-0", 640, 2560},
    {"blk.0.ffn_gate_exps.weight/expert-511", 640, 2560},
    {"blk.0.ffn_down_exps.weight/expert-0", 2560, 640},
    {"blk.0.ffn_down_exps.weight/expert-511", 2560, 640},
}};

void validate_inventory(const artifact::Reader& reader) {
    if (reader.objects().size() != 23) {
        throw std::runtime_error("bounded Qwen4 qualification requires all 23 matrices");
    }
    for (const auto& matrix : matrices) {
        for (const auto format : {artifact::NumericFormat::NVFP4,
                                  artifact::NumericFormat::FP8_E4M3FN_ROW_BF16S}) {
            if (format == artifact::NumericFormat::NVFP4 && matrix.n % 128) { continue; }
            const auto name = std::string(matrix.name) + "/" + std::string(artifact::format_name(format));
            const auto* object = reader.find(name);
            const auto* descriptor = object ? std::get_if<artifact::TensorDescriptor>(object) : nullptr;
            if (!descriptor || descriptor->format != format ||
                descriptor->shape != std::vector<std::uint64_t>{
                    static_cast<std::uint64_t>(matrix.n), static_cast<std::uint64_t>(matrix.k)}) {
                throw std::runtime_error("missing or incorrect real qualification matrix: " + name);
            }
        }
    }
}

test::quantized_weight::PackedWeight real_weight(int n, int k, unsigned seed) {
    const bool fp4 = tensor->format == artifact::NumericFormat::NVFP4;
    auto weight = fp4 ? qualification::make_nvfp4_weight(n, k, seed)
                      : qualification::make_fp8_weight(n, k, seed);
    const auto bytes = source->payload(tensor->name).data;
    if (bytes.size() != weight.payload.size()) {
        throw std::runtime_error("real projection fixture geometry mismatch");
    }
    std::memcpy(weight.payload.data(), bytes.data(), bytes.size());
    if (fp4) {
        std::memcpy(&weight.weight.weight_scale_divisor,
                    bytes.data() + weight.weight_divisor_offset, sizeof(float));
        weight.weight.input_scale_divisor = 1.0F;
    }
    // The public test fixture independently decodes these stored bytes and scales.
    // No converter's reconstructed output is used as the arithmetic oracle.
    return weight;
}
} // namespace

int main(int argc, char** argv) {
    const char* path = argc == 2 ? argv[1] : std::getenv("NINFER_QWEN4_MIXED_PROJECTIONS");
    if (!path) { std::cout << "No bounded Qwen4 projection fixture configured\n"; return 77; }
    if (!qualification::cuda_available()) { return 77; }
    try {
        artifact::Reader reader(path);
        if (reader.identity() != artifact::ArtifactIdentity{"qwen4/layer-qualification", "requantized-gguf"}) {
            throw std::runtime_error("not a bounded Qwen4 qualification artifact");
        }
        validate_inventory(reader);
        source = &reader;
        int failures = 0;
        for (const auto& object : reader.objects()) {
            tensor = std::get_if<artifact::TensorDescriptor>(&object);
            if (!tensor || tensor->shape.size() != 2 ||
                (tensor->format != artifact::NumericFormat::NVFP4 &&
                 tensor->format != artifact::NumericFormat::FP8_E4M3FN_ROW_BF16S)) {
                throw std::runtime_error("unexpected projection fixture object");
            }
            const int n = tensor->shape[0], k = tensor->shape[1];
            const std::array<qualification::Invocation, 3> a16{{
                {1}, {17}, {33, qualification::CallForm::Policy, ops::LinearPolicy::A16Only, true}}};
            failures += qualification::run_shape(tensor->name, qualification::ActivationCompute::A16,
                real_weight, {n, k, 479, qualification::Comparison::Full, true, a16});
            const std::array<qualification::Invocation, 3> wide_a16{{
                {129}, {512}, {4096}}};
            failures += qualification::run_shape(tensor->name, qualification::ActivationCompute::A16,
                real_weight, {n, k, 479, qualification::Comparison::Sampled, true, wide_a16});
            const bool fp4 = tensor->format == artifact::NumericFormat::NVFP4;
            const auto policy = fp4 ? ops::LinearPolicy::AllowA4 : ops::LinearPolicy::AllowA8;
            const std::array<qualification::Invocation, 3> wide{{
                {129, qualification::CallForm::Policy, policy},
                {512, qualification::CallForm::Policy, policy},
                {4096, qualification::CallForm::Policy, policy}}};
            failures += qualification::run_shape(tensor->name,
                fp4 ? qualification::ActivationCompute::A4 : qualification::ActivationCompute::A8,
                real_weight, {n, k, 479, qualification::Comparison::Sampled, true, wide});
        }
        std::cout << (failures ? "FAIL" : "PASS") << " 23 real Qwen4 native projections\n";
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
