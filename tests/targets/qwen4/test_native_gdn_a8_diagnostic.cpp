#include "targets/qwen4/native_sequence_components.h"
#include "targets/qwen4/native_text_panel.h"
#include <cstdlib>
#include <iostream>
using namespace ninfer::test::qwen4_sequence;
int main() {
    const char* root=std::getenv("NINFER_QWEN4_NATIVE_LAYERS");
    if(!root) { return 77; }
    if(ninfer::test::require_cuda()!=0) { return 1; }
    TextPanel panel(root);
    Result residual; residual.actual=panel.residual; residual.reference=residual.actual;
    const auto input=read(std::string(root)+"/qwen4-layer-0.ninfer",0,residual);
    return input.mixed.failures+input.scale.failures+gdn_a8_input_diagnostic(root,input.mixed);
}
