#include "ninfer/ops/ggml_embedding.h"

#include "ops/op_tester.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kRows = 4;
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
    return f16(read_u16(block)) * scale * code -
           f16(read_u16(block + 2)) * minimum;
}

Weight view(void* data) {
    Weight weight{};
    weight.payload = data;
    weight.payload_bytes = static_cast<std::uint64_t>(kRows) * kRowBytes;
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

int run_row(const std::vector<std::uint8_t>& encoded, Weight weight, int row_id) {
    GuardedDeviceBuffer output(static_cast<std::size_t>(kWidth) * sizeof(std::uint16_t));
    output.fill(0xff);
    Tensor out(output.data(), DType::BF16, {kWidth});
    ops::ggml_q4_k_embedding_row(weight, row_id, out, nullptr);
    cuda_synchronize();

    std::vector<std::uint16_t> expected(kWidth);
    const auto* row = encoded.data() + static_cast<std::size_t>(row_id) * kRowBytes;
    for (int column = 0; column < kWidth; ++column) {
        expected[column] = f32_to_bf16(static_cast<float>(decode(row, column)));
    }
    int failures = verify_exact("Q4_K embedding row", from_device<std::uint16_t>(output.data(),
                                                                                  kWidth),
                                expected);
    failures += output.verify_guards("Q4_K embedding row");
    return failures;
}

int run_batch(const std::vector<std::uint8_t>& encoded, Weight weight,
              std::int32_t tokens) {
    std::vector<std::int32_t> token_ids(static_cast<std::size_t>(tokens));
    for (std::int32_t token = 0; token < tokens; ++token) {
        token_ids[static_cast<std::size_t>(token)] = (token * 3 + 1) % kRows;
    }
    DeviceBuffer device_ids = to_device(token_ids);
    const std::size_t count =
        static_cast<std::size_t>(kWidth) * static_cast<std::size_t>(tokens);
    GuardedDeviceBuffer output(count * sizeof(std::uint16_t));
    output.fill(0xff);
    cuda_synchronize();
    Tensor ids(device_ids.p, DType::I32, {tokens});
    Tensor out(output.data(), DType::BF16, {kWidth, tokens});
    ops::ggml_q4_k_embedding(weight, ids, out, nullptr);
    cuda_synchronize();

    std::vector<std::uint16_t> expected(count);
    for (std::int32_t token = 0; token < tokens; ++token) {
        const auto* row = encoded.data() +
                          static_cast<std::size_t>(token_ids[static_cast<std::size_t>(token)]) *
                              kRowBytes;
        for (std::int32_t column = 0; column < kWidth; ++column) {
            expected[static_cast<std::size_t>(token) * kWidth + column] =
                f32_to_bf16(static_cast<float>(decode(row, column)));
        }
    }
    const std::string label = "Q4_K embedding batch T=" + std::to_string(tokens);
    int failures =
        verify_exact(label.c_str(), from_device<std::uint16_t>(output.data(), count), expected);
    failures += output.verify_guards(label);
    return failures;
}

template <class Function>
int expect_invalid(Function&& function, const char* label) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        return 0;
    }
    std::cerr << label << " was accepted\n";
    return 1;
}

int validation_cases(Weight weight) {
    constexpr std::int32_t tokens = 2;
    const std::vector<std::int32_t> token_ids = {0, kRows - 1};
    DeviceBuffer device_ids = to_device(token_ids);
    DeviceBuffer device_out(static_cast<std::size_t>(kWidth) * tokens *
                            sizeof(std::uint16_t));
    Tensor ids(device_ids.p, DType::I32, {tokens});
    Tensor out(device_out.p, DType::BF16, {kWidth, tokens});
    int failures = 0;

    Tensor wrong_dtype_ids(device_ids.p, DType::BF16, {tokens});
    failures += expect_invalid(
        [&] { ops::ggml_q4_k_embedding(weight, wrong_dtype_ids, out, nullptr); },
        "BF16 token ids");
    Tensor wrong_tokens_out(device_out.p, DType::BF16, {kWidth});
    failures += expect_invalid(
        [&] { ops::ggml_q4_k_embedding(weight, ids, wrong_tokens_out, nullptr); },
        "output token mismatch");
    Tensor aliased_out(device_ids.p, DType::BF16, {kWidth, tokens});
    failures += expect_invalid(
        [&] { ops::ggml_q4_k_embedding(weight, ids, aliased_out, nullptr); },
        "output/id alias");
    Tensor too_many_ids(device_ids.p, DType::I32, {4097});
    failures += expect_invalid(
        [&] { ops::ggml_q4_k_embedding(weight, too_many_ids, out, nullptr); },
        "T above verifier ceiling");
    Tensor empty_ids = ids;
    empty_ids.ne[0] = 0;
    failures += expect_invalid(
        [&] { ops::ggml_q4_k_embedding(weight, empty_ids, out, nullptr); },
        "zero-token input");
    return failures;
}

} // namespace

int main() {
    if (const int unavailable = require_cuda()) { return unavailable; }
    std::vector<std::uint8_t> encoded(static_cast<std::size_t>(kRows) * kRowBytes);
    std::mt19937 generator(9137U);
    for (auto& byte : encoded) { byte = static_cast<std::uint8_t>(generator() & 0xffU); }
    for (std::size_t offset = 0; offset < encoded.size(); offset += kBlockBytes) {
        encoded[offset] = 0x00;
        encoded[offset + 1] = 0x2c; // binary16 0.0625
        encoded[offset + 2] = 0x00;
        encoded[offset + 3] = 0x28; // binary16 0.03125
    }
    DeviceBuffer device = to_device(encoded);
    Weight weight = view(device.p);
    int failures = run_row(encoded, weight, 0);
    failures += run_row(encoded, weight, 2);
    failures += run_row(encoded, weight, kRows - 1);
    for (const std::int32_t tokens : {1, 16, 17, 128, 4096}) {
        failures += run_batch(encoded, weight, tokens);
    }
    failures += validation_cases(weight);
    for (const int invalid : {-1, kRows}) {
        try {
            DeviceBuffer output(static_cast<std::size_t>(kWidth) * sizeof(std::uint16_t));
            Tensor out(output.p, DType::BF16, {kWidth});
            ops::ggml_q4_k_embedding_row(weight, invalid, out, nullptr);
            std::cerr << "FAIL Q4_K embedding accepted invalid token id\n";
            ++failures;
        } catch (const std::invalid_argument&) {}
    }
    std::cout << (failures == 0 ? "OK" : "FAIL") << " ggml_embedding\n";
    return failures == 0 ? 0 : 1;
}
