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
    const char* root = std::getenv("NINFER_QWEN4_NATIVE_LAYERS");
    if (!root) { return 77; }
    if (require_cuda() != 0) { return 1; }
    const bool row_z2=(argc==4 || argc==5) && std::string(argv[1])=="--gdn-row-fp8-z2";
    if(row_z2 || ((argc>=4 && argc<=7) && (std::string(argv[1])=="--gdn-input" ||
                                std::string(argv[1])=="--moe-input"))) {
        const bool is_gdn=std::string(argv[1])=="--gdn-input";
        if(is_gdn && argc>6) throw std::invalid_argument("invalid GDN diagnostic arguments");
        const int layer=row_z2?2:std::stoi(argv[2]);
        const std::filesystem::path path(argv[row_z2?2:3]);
        const auto bytes=std::filesystem::file_size(path);
        if(layer<0 || layer>(is_gdn?2:3) || bytes==0 || bytes%(2560*4) || bytes>4096ULL*2560*4)
            throw std::invalid_argument("invalid native component diagnostic input");
        Result input;input.actual.resize(bytes/4);
        std::ifstream file(path,std::ios::binary);
        file.read(reinterpret_cast<char*>(input.actual.data()),bytes);
        if(!file)throw std::runtime_error("truncated native component diagnostic input");
        for(float value:input.actual) if(!std::isfinite(value) || represented(value)!=value)
            throw std::runtime_error("native component diagnostic requires represented BF16 input");
        input.reference=input.actual;
        const auto source=std::string(root)+"/qwen4-layer-"+std::to_string(layer)+".ninfer";
        if(row_z2 && argc==5 && std::string(argv[4])!="0" && std::string(argv[4])!="1")
            throw std::invalid_argument("row-FP8 Z2 activation selector must be 0 or 1");
        const auto result=row_z2 ? gdn_row_fp8_z2(source,argv[3],input,false,argc==5 && std::string(argv[4])=="1") :
            is_gdn ? (argc==4 ? gdn(source,layer,input,false) :
            gdn_calibrated(root,source,layer,input,false,std::stoi(argv[4]),argc==6?std::stoi(argv[5]):(layer==0?7:2))) :
            (argc==4 ? moe(source,layer,input) :
            moe_calibrated(root,source,layer,input,false,std::stoi(argv[4]),false,
                argc>=6?std::stoi(argv[5]):7,argc==7?argv[6]:""));
        return result.failures?1:0;
    }
    if (argc != 2) { throw std::invalid_argument("expected new capture directory"); }
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
        const auto label = "layer"+std::to_string(layer);
        save(label+".residual_input",residual);
        if (layer == 1) {
            residual = ple(root, residual, false, false, true);
            failures += residual.failures;
            save(label+".ple_output",residual);
        }
        const auto gr = read(path, layer, residual);
        save(label+".attn_input",gr.mixed);
        if (layer == 3) {
            save("residual", residual); save("mixed", gr.mixed); save("write", gr.scale);
        }
        const auto mixer = layer == 3 ? qsa(path, gr.mixed, false) : gdn(path, layer, gr.mixed, false);
        save(label+".mixer_output",mixer);
        if(layer!=3) {
            Result conv;conv.actual=mixer.conv_actual;conv.reference=mixer.conv_reference;
            Result state;state.actual=mixer.recurrent_actual;state.reference=mixer.recurrent_reference;
            save(label+".conv_state",conv);save(label+".recurrent_state",state);
        }
        const auto attention = inject(residual, mixer, gr.scale);
        const auto mlp = read(path, layer, attention, "mlp");
        save(label+".moe_input",mlp.mixed);
        const auto expert = moe(path, layer, mlp.mixed);
        save(label+".moe_output",expert);
        for(const auto& discrete : {std::pair{label+".mixer_ids",&mixer.discrete_ids},
                                    std::pair{label+".expert_ids",&expert.discrete_ids},
                                    std::pair{label+".expert_reference_ids",&expert.reference_discrete_ids}}) {
            std::ofstream ids(directory/(discrete.first+".i32"),std::ios::binary);
            ids.write(reinterpret_cast<const char*>(discrete.second->data()),discrete.second->size()*sizeof(int));
            if(!ids) throw std::runtime_error("capture IDs write failed");
        }
        residual = inject(attention, expert, mlp.scale);
        save(label+".residual_output",residual);
        failures += gr.mixed.failures + gr.scale.failures + mixer.failures + attention.failures +
            mlp.mixed.failures + mlp.scale.failures + expert.failures + residual.failures;
        failures += verify_reduction("capture accumulated layer" + std::to_string(layer),
            wide(residual.actual), wide(residual.reference), ReductionCriterion{.02, .005, .02});
        if (layer == 3) { save("qsa", mixer); save("final", residual); }
    }
    std::ofstream metadata(directory / "capture.json");
    metadata << "{\"profile\":\"" << (panel.tokens.size()==33 ?
                "qwen4-native-text33-layer3-kv-assessment" : "qwen4-native-resident-corpus") << "\","
                "\"tokens\":[";
    for (std::size_t i = 0; i < panel.tokens.size(); ++i) {
        if (i) { metadata << ','; }
        metadata << panel.tokens[i];
    }
    metadata << "],\"prefix_failures\":" << failures << ",\"positions\":\"t+axis\","
                "\"dtype\":\"little-endian-f32; BF16 public activations/conv, FP32 recurrent state\","
                "\"boundary\":\"BF16 source projections, BF16 PLE/cache, W4A16 experts; "
                "independent accumulated reference alongside GPU public boundaries\"}\n";
    if (!metadata) { throw std::runtime_error("capture metadata write failed"); }
    return failures ? 1 : 0;
}
