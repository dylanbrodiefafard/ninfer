#include "targets/qwen4/native_sequence_components.h"
#include "targets/qwen4/native_text_panel.h"
#include "ops/op_tester.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>

using namespace ninfer::test;
using namespace ninfer::test::qwen4_sequence;

// An explicit offline assessment input, not an inference entry point or CTest.
int main(int argc, char** argv) {
    if (argc != 2) { throw std::invalid_argument("expected new capture directory"); }
    const char* root = std::getenv("NINFER_QWEN4_NATIVE_LAYERS");
    if (!root) { return 77; }
    if (require_cuda() != 0) { return 1; }
    const std::filesystem::path directory(argv[1]);
    if (!std::filesystem::create_directory(directory)) {
        throw std::runtime_error("capture directory already exists");
    }
    auto save = [&](const std::string& name, const Result& value) {
        for (bool reference : {false, true}) {
            const auto& data = reference ? value.reference : value.actual;
            std::ofstream file(directory / (name + (reference ? ".reference.f32" : ".actual.f32")),
                               std::ios::binary);
            file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
            if (!file) { throw std::runtime_error("capture write failed"); }
        }
    };
    const TextPanel panel(root);
    Result residual; residual.actual = panel.residual; residual.reference = residual.actual;
    int failures = 0;
    for (int layer = 0; layer < 4; ++layer) {
        const auto path = std::string(root) + "/qwen4-layer-" + std::to_string(layer) + ".ninfer";
        if (layer == 1) {
            residual = ple(root, residual, false, false, true);
            failures += residual.failures;
        }
        const auto gr = read(path, layer, residual);
        if (layer == 3) {
            save("residual", residual); save("mixed", gr.mixed); save("write", gr.scale);
        }
        const auto mixer = layer == 3 ? qsa(path, gr.mixed, false) : gdn(path, layer, gr.mixed, false);
        const auto attention = inject(residual, mixer, gr.scale);
        const auto mlp = read(path, layer, attention, "mlp");
        const auto expert = moe(path, layer, mlp.mixed);
        residual = inject(attention, expert, mlp.scale);
        failures += gr.mixed.failures + gr.scale.failures + mixer.failures + attention.failures +
            mlp.mixed.failures + mlp.scale.failures + expert.failures + residual.failures;
        failures += verify_reduction("capture accumulated layer" + std::to_string(layer),
            wide(residual.actual), wide(residual.reference), ReductionCriterion{.02, .005, .02});
        if (layer == 3) { save("qsa", mixer); save("final", residual); }
    }
    std::ofstream metadata(directory / "capture.json");
    metadata << "{\"profile\":\"qwen4-native-text33-layer3-kv-assessment\","
                "\"tokens\":[";
    for (std::size_t i = 0; i < panel.tokens.size(); ++i) {
        if (i) { metadata << ','; }
        metadata << panel.tokens[i];
    }
    metadata << "],\"prefix_failures\":" << failures << ",\"positions\":\"t+axis\","
                "\"dtype\":\"little-endian-f32-representing-bf16\","
                "\"boundary\":\"BF16 source projections, BF16 PLE/cache, W4A16 experts; "
                "independent accumulated reference alongside GPU public boundaries\"}\n";
    if (!metadata) { throw std::runtime_error("capture metadata write failed"); }
    return failures ? 1 : 0;
}
