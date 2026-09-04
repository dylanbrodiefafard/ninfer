#include "targets/qwen4/verifier.h"

#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
namespace verifier = ninfer::targets::qwen4::verifier;

namespace {

constexpr std::int32_t kRows = 3;
constexpr std::int32_t kWidth = 2560;
constexpr std::int32_t kBlockValues = 256;
constexpr std::int32_t kBlockBytes = 144;
constexpr std::int32_t kRowBytes = kWidth / kBlockValues * kBlockBytes;

std::uint16_t read_u16(const std::uint8_t* bytes) {
    return static_cast<std::uint16_t>(bytes[0]) |
           (static_cast<std::uint16_t>(bytes[1]) << 8U);
}

double f16(std::uint16_t word) {
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

std::pair<int, int> scale_min(const std::uint8_t* packed, int group) {
    if (group < 4) { return {packed[group] & 63, packed[group + 4] & 63}; }
    return {(packed[group + 4] & 15) | ((packed[group - 4] >> 6U) << 4U),
            (packed[group + 4] >> 4U) | ((packed[group] >> 6U) << 4U)};
}

double decode(const std::uint8_t* row, int column) {
    const auto* block = row + (column / kBlockValues) * kBlockBytes;
    const int item = column % kBlockValues;
    const int group = item / 32;
    const int lane = item % 32;
    const auto [scale, minimum] = scale_min(block + 4, group);
    const int packed = block[16 + 32 * (group / 2) + lane];
    const int code = group % 2 == 0 ? packed & 15 : packed >> 4;
    return f16(read_u16(block)) * scale * code - f16(read_u16(block + 2)) * minimum;
}

Weight weight_view(void* data, std::size_t bytes) {
    Weight weight{};
    weight.payload = data;
    weight.payload_bytes = bytes;
    weight.qdata = data;
    weight.qtype = QType::GGML_Q4_K;
    weight.layout = QuantLayout::GgmlBlockRow;
    weight.group_size = kBlockValues;
    weight.group = kBlockValues;
    weight.n = kRows;
    weight.k = kWidth;
    weight.ndim = 2;
    weight.shape[0] = weight.padded_shape[0] = kRows;
    weight.shape[1] = weight.padded_shape[1] = kWidth;
    return weight;
}

bool all_zero(const void* data, std::size_t bytes) {
    const auto values = from_device<std::uint8_t>(data, bytes);
    return std::all_of(values.begin(), values.end(), [](std::uint8_t value) { return value == 0; });
}

} // namespace

int main() {
    if (const int unavailable = require_cuda()) { return unavailable; }
    std::vector<std::uint8_t> encoded(static_cast<std::size_t>(kRows) * kRowBytes);
    std::mt19937 generator(9191U);
    for (auto& byte : encoded) { byte = static_cast<std::uint8_t>(generator() & 0xffU); }
    for (std::size_t offset = 0; offset < encoded.size(); offset += kBlockBytes) {
        encoded[offset] = 0x00;
        encoded[offset + 1] = 0x2c;
        encoded[offset + 2] = 0x00;
        encoded[offset + 3] = 0x28;
    }
    DeviceBuffer device_embedding = to_device(encoded);
    const Weight embedding = weight_view(device_embedding.p, device_embedding.bytes);
    verifier::State state;
    state.reset(nullptr);
    state.embed_token(embedding, 1, nullptr);
    cuda_synchronize();

    int failures = 0;
    std::size_t gdn_count = 0;
    std::size_t qsa_count = 0;
    for (std::size_t layer = 0; layer < verifier::kLayerCount; ++layer) {
        if (state.gdn()[layer]) {
            ++gdn_count;
            if (!all_zero(state.gdn()[layer]->conv.data, state.gdn()[layer]->conv.bytes()) ||
                !all_zero(state.gdn()[layer]->recurrence.data,
                          state.gdn()[layer]->recurrence.bytes())) {
                std::cerr << "FAIL nonzero reset GDN state\n";
                ++failures;
                break;
            }
        }
        if (state.qsa()[layer]) {
            ++qsa_count;
            const auto& qsa = *state.qsa()[layer];
            if (qsa.k_codes.ne[1] != verifier::kQsaCapacity ||
                !all_zero(qsa.k_codes.data, qsa.k_codes.bytes()) ||
                !all_zero(qsa.positions.data, qsa.positions.bytes())) {
                std::cerr << "FAIL invalid reset QSA state\n";
                ++failures;
                break;
            }
        }
    }
    if (gdn_count != verifier::kGdnLayerCount || qsa_count != verifier::kQsaLayerCount ||
        state.workspace_bytes() == 0 || state.expert_host_stage_bytes() != verifier::kExpertStageBytes ||
        state.expert_device_stage().bytes() != verifier::kExpertStageBytes ||
        state.ple_host_rows_bytes() != ops::kPleStagedBytes ||
        state.ple_device_rows().bytes() != ops::kPleStagedBytes ||
        state.mixed_x().bytes() != 2560 * 2 || state.block_output().bytes() != 2560 * 2 ||
        state.write_scale().bytes() != 4 * 2) {
        std::cerr << "FAIL verifier state ownership/capacity mismatch\n";
        ++failures;
    }
    const auto history = from_device<std::int32_t>(state.ple_token_history().data, 2);
    if (history != std::vector<std::int32_t>({verifier::kPleResetToken,
                                             verifier::kPleResetToken}) ||
        !all_zero(state.ple_conv_state().data, state.ple_conv_state().bytes())) {
        std::cerr << "FAIL PLE reset state\n";
        ++failures;
    }

    const auto residual_bits = from_device<std::uint16_t>(state.residual().data, 4 * kWidth);
    std::vector<std::uint16_t> expected(kWidth);
    const auto* row = encoded.data() + kRowBytes;
    for (int column = 0; column < kWidth; ++column) {
        expected[column] = f32_to_bf16(static_cast<float>(decode(row, column)));
    }
    for (int branch = 0; branch < 4; ++branch) {
        const std::vector<std::uint16_t> actual(residual_bits.begin() + branch * kWidth,
                                                residual_bits.begin() + (branch + 1) * kWidth);
        failures += verify_exact("Qwen4 verifier initial residual branch", actual, expected);
    }
    std::cout << (failures == 0 ? "OK" : "FAIL") << " qwen4_verifier_state\n";
    return failures == 0 ? 0 : 1;
}
