#include "targets/qwen4/verifier.h"

#include "core/device.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <vector>

namespace verifier = ninfer::targets::qwen4::verifier;

int main() {
    const char* configured = std::getenv("NINFER_QWEN4_VERIFY_WEIGHTS");
    if (configured == nullptr || *configured == '\0') {
        std::cout << "skip: NINFER_QWEN4_VERIFY_WEIGHTS is not set\n";
        return 77;
    }
    const std::filesystem::path path(configured);
    if (!std::filesystem::is_regular_file(path)) {
        std::cerr << "verifier artifact is not a regular file: " << path << '\n';
        return 1;
    }

    try {
        ninfer::DeviceContext device(0);
        std::unique_ptr<verifier::LoadedModel> model = verifier::LoadedModel::load(path, device);
        const auto& stats = model->backing().stats();
        if (stats.tensor_count != verifier::kDeviceTensorCount ||
            stats.h2d_bytes != verifier::kDevicePayloadBytes ||
            stats.mapped_tensor_count != verifier::kMappedTensorCount ||
            stats.mapped_tensor_bytes != verifier::kMappedTensorBytes ||
            model->view().ple.table.bytes != 28'800'138'240ULL ||
            model->view().ple.table.data == nullptr) {
            std::cerr << "verifier materialization totals or retained PLE mapping changed\n";
            return 1;
        }

        std::size_t gdn_count = 0;
        std::size_t qsa_count = 0;
        for (std::size_t layer = 0; layer < verifier::kLayerCount; ++layer) {
            const auto& view = model->view().layers[layer];
            gdn_count += view.gdn.has_value();
            qsa_count += view.qsa.has_value();
            if ((view.gdn.has_value() == view.qsa.has_value()) ||
                view.attention_gr.norm.dtype != ninfer::DType::FP32 ||
                view.attention_gr.inject.ne[0] != 10240 ||
                view.attention_gr.inject.ne[1] != 4 ||
                view.attention_gr.down.qtype != ninfer::QType::GGML_Q8_0 ||
                view.moe.routed_gate_up.gate.data() == nullptr ||
                view.moe.routed_gate_up.up.data() == nullptr) {
                std::cerr << "typed layer view is incomplete at layer " << layer << '\n';
                return 1;
            }
        }
        if (gdn_count != verifier::kGdnLayerCount || qsa_count != verifier::kQsaLayerCount ||
            model->view().layers[0].gdn->a.ne[0] != 2560 ||
            model->view().layers[0].gdn->a.ne[1] != 48 ||
            model->view().layers[0].gdn->conv.ne[0] != 4 ||
            model->view().layers[0].gdn->conv.ne[1] != 10240 ||
            model->view().ple.conv.ne[0] != 4 || model->view().ple.conv.ne[1] != 10240 ||
            model->view().layers[2].gdn->qkv.qtype != ninfer::QType::GGML_Q6_K ||
            model->view().layers[0].gdn->qkv.qtype != ninfer::QType::GGML_Q5_K ||
            model->view().token_embedding.qtype != ninfer::QType::GGML_Q4_K ||
            model->view().output_head.qtype != ninfer::QType::GGML_Q4_K) {
            std::cerr << "typed verifier model topology or formats changed\n";
            return 1;
        }

        verifier::State state;
        state.reset(device.stream);
        state.embed_token(model->view().token_embedding, 0, device.stream);
        device.synchronize();
        std::vector<std::uint16_t> residual(4 * 2560);
        CUDA_CHECK(cudaMemcpy(residual.data(), state.residual().data,
                              residual.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));
        for (int branch = 1; branch < 4; ++branch) {
            if (!std::equal(residual.begin(), residual.begin() + 2560,
                            residual.begin() + branch * 2560)) {
                std::cerr << "initial embedding was not copied to every residual branch\n";
                return 1;
            }
        }
        const std::array<std::int32_t, 2> expected_history = {
            verifier::kPleResetToken, verifier::kPleResetToken};
        std::array<std::int32_t, 2> history{};
        CUDA_CHECK(cudaMemcpy(history.data(), state.ple_token_history().data, sizeof(history),
                              cudaMemcpyDeviceToHost));
        if (history != expected_history || state.workspace_bytes() == 0 ||
            state.expert_host_stage_bytes() != verifier::kExpertStageBytes) {
            std::cerr << "verifier reset or fixed state capacity changed\n";
            return 1;
        }
        std::cout << "OK qwen4 verifier artifact: device_payload=" << stats.h2d_bytes
                  << " mapped=" << stats.mapped_tensor_bytes
                  << " persistent_state=" << verifier::kPersistentStateBytes
                  << " stage=" << verifier::kExpertStageBytes << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "verifier artifact load failed: " << error.what() << '\n';
        return 1;
    }
}
