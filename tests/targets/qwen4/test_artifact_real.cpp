#include "targets/qwen4/verifier.h"

#include "core/device.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace verifier = ninfer::targets::qwen4::verifier;
using ninfer::test::f32_to_bf16;

namespace {

constexpr std::size_t kEmbeddingWidth = 2560;
constexpr std::size_t kQ4KBlockValues = 256;
constexpr std::size_t kQ4KBlockBytes = 144;
constexpr std::size_t kEmbeddingRowBytes =
    kEmbeddingWidth / kQ4KBlockValues * kQ4KBlockBytes;
constexpr std::array<std::uint64_t, 3> kNgramMultiplier = {
    23703573157769ULL, 20109073645365ULL, 8052911324071ULL};
constexpr std::array<std::int32_t, 16> kNgramPrime = {
    20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
    20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171};
constexpr std::array<std::int32_t, 16> kNgramOffset = {
    0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
    160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114,
    300001275};

std::uint16_t read_u16(const std::uint8_t* bytes) {
    return static_cast<std::uint16_t>(bytes[0]) |
           (static_cast<std::uint16_t>(bytes[1]) << 8U);
}

double binary16(std::uint16_t word) {
    const double sign = (word & 0x8000U) != 0 ? -1.0 : 1.0;
    const unsigned exponent = (word >> 10U) & 0x1fU;
    const unsigned fraction = word & 0x3ffU;
    if (exponent == 0) { return sign * std::ldexp(static_cast<double>(fraction), -24); }
    if (exponent == 31) {
        return fraction == 0 ? sign * std::numeric_limits<double>::infinity()
                             : std::numeric_limits<double>::quiet_NaN();
    }
    return sign * std::ldexp(static_cast<double>(1024U + fraction),
                             static_cast<int>(exponent) - 25);
}

std::pair<int, int> q4_k_scale_min(const std::uint8_t* packed, int group) {
    if (group < 4) { return {packed[group] & 63, packed[group + 4] & 63}; }
    return {(packed[group + 4] & 15) | ((packed[group - 4] >> 6U) << 4U),
            (packed[group + 4] >> 4U) | ((packed[group] >> 6U) << 4U)};
}

double decode_q4_k(const std::uint8_t* row, std::size_t column) {
    const auto* block = row + (column / kQ4KBlockValues) * kQ4KBlockBytes;
    const int item = static_cast<int>(column % kQ4KBlockValues);
    const int group = item / 32;
    const int lane = item % 32;
    const auto [scale, minimum] = q4_k_scale_min(block + 4, group);
    const int packed = block[16 + 32 * (group / 2) + lane];
    const int code = group % 2 == 0 ? packed & 15 : packed >> 4;
    return binary16(read_u16(block)) * scale * code -
           binary16(read_u16(block + 2)) * minimum;
}

std::array<std::int32_t, ninfer::ops::kPleHeads> ple_rows(std::int32_t token) {
    const std::uint64_t lag = static_cast<std::uint64_t>(verifier::kPleResetToken);
    const std::uint64_t mixed2 = static_cast<std::uint64_t>(token) * kNgramMultiplier[0] ^
                                 lag * kNgramMultiplier[1];
    const std::uint64_t mixed3 = mixed2 ^ lag * kNgramMultiplier[2];
    std::array<std::int32_t, ninfer::ops::kPleHeads> rows{};
    for (std::size_t head = 0; head < rows.size(); ++head) {
        const std::int64_t mixed = std::bit_cast<std::int64_t>(head < 8 ? mixed2 : mixed3);
        std::int64_t remainder = mixed % kNgramPrime[head];
        if (remainder < 0) { remainder += kNgramPrime[head]; }
        rows[head] = kNgramOffset[head] + static_cast<std::int32_t>(remainder);
    }
    return rows;
}

double decode_iq4_nl(const std::uint8_t* block, int index) {
    constexpr std::array<int, 16> codebook = {
        -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
    const std::uint8_t packed = block[2 + (index & 15)];
    const std::uint8_t code = index < 16 ? packed & 15U : packed >> 4U;
    return binary16(read_u16(block)) * codebook[code];
}

} // namespace

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
        std::vector<std::uint8_t> encoded_row(kEmbeddingRowBytes);
        CUDA_CHECK(cudaMemcpy(encoded_row.data(), model->view().token_embedding.qdata,
                              encoded_row.size(), cudaMemcpyDeviceToHost));
        std::vector<std::uint16_t> expected(kEmbeddingWidth);
        for (std::size_t column = 0; column < expected.size(); ++column) {
            expected[column] = f32_to_bf16(static_cast<float>(
                decode_q4_k(encoded_row.data(), column)));
        }

        std::vector<std::uint16_t> residual(4 * kEmbeddingWidth);
        CUDA_CHECK(cudaMemcpy(residual.data(), state.residual().data,
                              residual.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost));
        for (std::size_t branch = 0; branch < 4; ++branch) {
            const auto branch_begin = residual.begin() + branch * kEmbeddingWidth;
            if (!std::equal(expected.begin(), expected.end(), branch_begin)) {
                std::cerr << "initial embedding branch " << branch
                          << " did not exactly match the independent Q4_K row oracle\n";
                return 1;
            }
        }

        const auto rows = ple_rows(48);
        ninfer::DeviceBuffer ple_embedding_storage(
            static_cast<std::size_t>(ninfer::ops::kPleHeads) * ninfer::ops::kPleRowWidth *
            sizeof(std::uint16_t));
        ninfer::Tensor ple_embedding(
            ple_embedding_storage.p, ninfer::DType::BF16,
            {ninfer::ops::kPleRowWidth, ninfer::ops::kPleHeads});
        ninfer::Tensor staged_rows = state.ple_device_rows();
        ninfer::ops::ple_iq4_nl_stage_rows(
            model->view().ple.table, rows, state.ple_host_rows(), state.ple_host_rows_bytes(),
            staged_rows, device.stream);
        ninfer::ops::ple_iq4_nl_decode_rows(staged_rows, ple_embedding, device.stream);
        device.synchronize();

        std::array<std::uint8_t, ninfer::ops::kPleStagedBytes> expected_staged{};
        for (std::size_t head = 0; head < rows.size(); ++head) {
            std::memcpy(expected_staged.data() + head * ninfer::ops::kPleIq4NlRowBytes,
                        model->view().ple.table.data +
                            static_cast<std::uint64_t>(rows[head]) *
                                ninfer::ops::kPleIq4NlRowBytes,
                        ninfer::ops::kPleIq4NlRowBytes);
        }
        std::array<std::uint8_t, ninfer::ops::kPleStagedBytes> host_staged{};
        std::array<std::uint8_t, ninfer::ops::kPleStagedBytes> device_staged{};
        std::memcpy(host_staged.data(), state.ple_host_rows(), host_staged.size());
        CUDA_CHECK(cudaMemcpy(device_staged.data(), staged_rows.data, device_staged.size(),
                              cudaMemcpyDeviceToHost));
        if (host_staged != expected_staged || device_staged != expected_staged) {
            std::cerr << "real PLE staging did not exactly copy the independently addressed rows\n";
            return 1;
        }
        std::vector<std::uint16_t> actual_ple(
            static_cast<std::size_t>(ninfer::ops::kPleHeads) * ninfer::ops::kPleRowWidth);
        CUDA_CHECK(cudaMemcpy(actual_ple.data(), ple_embedding.data,
                              actual_ple.size() * sizeof(std::uint16_t),
                              cudaMemcpyDeviceToHost));
        std::vector<std::uint16_t> expected_ple(actual_ple.size());
        for (int head = 0; head < ninfer::ops::kPleHeads; ++head) {
            const auto* row = expected_staged.data() +
                              static_cast<std::size_t>(head) * ninfer::ops::kPleIq4NlRowBytes;
            for (int dimension = 0; dimension < ninfer::ops::kPleRowWidth; ++dimension) {
                expected_ple[static_cast<std::size_t>(head) * ninfer::ops::kPleRowWidth +
                             dimension] = f32_to_bf16(static_cast<float>(decode_iq4_nl(
                    row + (dimension / ninfer::ops::kPleIq4NlBlockValues) *
                              ninfer::ops::kPleIq4NlBlockBytes,
                    dimension % ninfer::ops::kPleIq4NlBlockValues)));
            }
        }
        if (actual_ple != expected_ple) {
            std::cerr << "real PLE rows did not exactly match the independent IQ4_NL oracle\n";
            return 1;
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
