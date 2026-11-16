#include "ninfer/ops/stream_mean.h"
#include "ops/op_tester.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kHidden  = ops::kStreamMeanHidden;
constexpr std::int32_t kStreams = ops::kStreamMeanStreams;
// The five Flash-Next DFlash2 target layers share one feature matrix of 5 * 2560 rows.
constexpr std::int32_t kFeatureRows = 5 * kHidden;

// One BF16 rounding of the mean: half an ulp (2^-9) relative, with margin to the 3.95e-3 BF16
// criterion. Fixtures keep the four-term FP32 sum exact, so cancellation to zero is exact too.
constexpr PointwiseCriterion stream_mean_criterion() {
    return {/*absolute*/ 0.0, /*relative*/ 3.95e-3};
}

std::vector<std::uint16_t> encode_bf16(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) bits[i] = f32_to_bf16(values[i]);
    return bits;
}

// Writes the mean into feature rows [slot * 2560, (slot + 1) * 2560) of a [12800,T] matrix
// (column stride 25600 bytes) and checks every other feature byte is untouched; slot < 0 uses a
// contiguous [2560,T] output.
int run_case(std::int32_t tokens, std::int32_t slot, std::uint32_t seed, bool cancel) {
    const std::size_t residual_count = static_cast<std::size_t>(kHidden) * kStreams * tokens;
    std::vector<float> residual(residual_count);
    fill_uniform(residual, seed, -8.0f, 8.0f);
    round_to_bf16(residual);
    if (cancel) {
        // Streams 2 and 3 negate streams 0 and 1 in every even channel: the mean is exactly zero.
        for (std::int32_t t = 0; t < tokens; ++t) {
            for (std::int32_t d = 0; d < kHidden; d += 2) {
                const std::size_t base       = static_cast<std::size_t>(t) * kStreams * kHidden + d;
                residual[base + 2 * kHidden] = -residual[base];
                residual[base + 3 * kHidden] = -residual[base + kHidden];
            }
        }
    }
    std::vector<double> expected(static_cast<std::size_t>(kHidden) * tokens);
    for (std::int32_t t = 0; t < tokens; ++t) {
        for (std::int32_t d = 0; d < kHidden; ++d) {
            double sum = 0.0;
            for (std::int32_t i = 0; i < kStreams; ++i) {
                sum += static_cast<double>(
                    residual[(static_cast<std::size_t>(t) * kStreams + i) * kHidden + d]);
            }
            expected[static_cast<std::size_t>(t) * kHidden + d] = sum / kStreams;
        }
    }

    const auto residual_bits = encode_bf16(residual);
    GuardedDeviceBuffer device_residual(residual_bits.size() * sizeof(std::uint16_t));
    device_residual.copy_from_host(residual_bits.data(), device_residual.bytes());
    const bool strided          = slot >= 0;
    const std::int32_t out_rows = strided ? kFeatureRows : kHidden;
    const std::size_t out_count = static_cast<std::size_t>(out_rows) * tokens;
    GuardedDeviceBuffer device_out(out_count * sizeof(std::uint16_t));
    device_out.fill(0xcd);
    Tensor residual_t(device_residual.data(), DType::BF16, {kHidden, kStreams, tokens});
    Tensor matrix(device_out.data(), DType::BF16, {out_rows, tokens});
    Tensor out_t = strided ? matrix.slice(0, slot * kHidden, kHidden) : matrix;
    ops::stream_mean(residual_t, out_t, nullptr);
    cuda_synchronize();

    const auto out_bits = from_device<std::uint16_t>(device_out.data(), out_count);
    std::vector<double> got(expected.size());
    int untouched_failures = 0;
    for (std::int32_t t = 0; t < tokens; ++t) {
        for (std::int32_t row = 0; row < out_rows; ++row) {
            const std::uint16_t bits = out_bits[static_cast<std::size_t>(t) * out_rows + row];
            const std::int32_t local = strided ? row - slot * kHidden : row;
            if (local >= 0 && local < kHidden) {
                got[static_cast<std::size_t>(t) * kHidden + local] = bf16_to_f32(bits);
            } else if (bits != 0xcdcd) {
                ++untouched_failures;
            }
        }
    }
    const std::string label = "stream_mean T=" + std::to_string(tokens) +
                              " slot=" + std::to_string(slot) + (cancel ? " cancel" : "");
    int failures            = verify_pointwise(label, got, expected, stream_mean_criterion());
    if (untouched_failures != 0) {
        std::cerr << label << ": " << untouched_failures << " bytes outside out changed\n";
        ++failures;
    }
    failures += verify_exact(
        "stream_mean residual unchanged",
        from_device<std::uint16_t>(device_residual.data(), residual_bits.size()), residual_bits);
    failures += device_residual.verify_guards("stream_mean residual");
    failures += device_out.verify_guards("stream_mean out");
    return failures;
}

int run_invalid_cases() {
    GuardedDeviceBuffer residual(static_cast<std::size_t>(kHidden) * kStreams * 2 * 2);
    GuardedDeviceBuffer out(static_cast<std::size_t>(kHidden) * 3 * 2);
    Tensor residual_t(residual.data(), DType::BF16, {kHidden, kStreams, 2});
    int failures = 0;
    failures += expect_invalid_argument(
        [&] {
            Tensor wrong(out.data(), DType::BF16, {kHidden, 3});
            ops::stream_mean(residual_t, wrong, nullptr);
        },
        "stream_mean accepted a mismatched token count");
    failures += expect_invalid_argument(
        [&] {
            Tensor overlap(residual.data(), DType::BF16, {kHidden, 2});
            ops::stream_mean(residual_t, overlap, nullptr);
        },
        "stream_mean accepted out overlapping residual");
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cerr << "FAIL: no usable CUDA device\n";
        return 1;
    }
    int failures = 0;
    failures += run_case(1, -1, 11u, false);
    failures += run_case(48, -1, 13u, true);
    failures += run_case(1, 0, 17u, false);
    failures += run_case(8, 4, 19u, false);
    failures += run_case(48, 2, 23u, true);
    failures += run_case(4096, 3, 29u, false);
    failures += run_invalid_cases();
    std::cout << (failures ? "FAIL" : "OK") << " stream_mean\n";
    return failures ? 1 : 0;
}
