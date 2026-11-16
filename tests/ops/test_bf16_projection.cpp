#include "ninfer/ops/bf16_projection.h"
#include "ops/op_tester.h"

#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kInputRows = 2560;

// FP32 outputs. The GEMV (T=1 at N >= 2560): IEEE FP32 fused multiply-adds over 2560 BF16
// products, a few FP32 ulps of the reference norm. The tensor-core MMA routes (every other point):
// the accumulator absorbs at most 160 K16 partial sums whose additions may truncate rather than
// round, a consistently signed error of up to about one ulp of the running sum per step; the bound
// allows 160 such steps at u = 2^-24 with a factor of 2 margin (about 2e-5 relative), and gross
// errors to 1e-4 of the largest reference magnitude. The small-T route's chains are 20 steps long
// and its eight per-warp partials add seven rounded IEEE FP32 steps, inside the same bound. The
// unspecified rounding inside each atom (16 products and the accumulate) is covered by the margin.
constexpr ReductionCriterion gemv_criterion() {
    return {/*relative_l2*/ 2.0e-6, /*gross_absolute*/ 0.0,
            /*gross_relative_to_max_reference*/ 1.0e-5};
}

constexpr ReductionCriterion mma_criterion() {
    return {/*relative_l2*/ 2.0e-5, /*gross_absolute*/ 0.0,
            /*gross_relative_to_max_reference*/ 1.0e-4};
}

// BF16 outputs round the FP32 accumulator once: half a BF16 ulp is at most 2^-8 of the magnitude
// (at the bottom of a binade), and an accumulator within its FP32 error of a rounding midpoint may
// round to the other neighbour; the absolute term covers that accumulation error. It is a
// statistical bound for these fixtures, not a worst case: with x uniform in [-2,2] and w in
// [-0.05,0.05], every partial sum is a random walk of standard deviation at most about 1.7, so
// partial sums stay below 8 and 160 accumulation steps of 2^-24 * 8 total under 1e-4.
constexpr PointwiseCriterion bf16_output_criterion() {
    return {/*absolute*/ 1.0e-4, /*relative*/ 3.91e-3};
}

std::vector<std::uint16_t> encode_bf16(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { bits[i] = f32_to_bf16(values[i]); }
    return bits;
}

// Device placement of x and the weight: at a 256-byte boundary, or 16 bytes past one, the weakest
// alignment the contract admits (the small-T route then reads its fragments in 16-byte halves).
enum class OperandPlacement : std::uint8_t {
    Aligned256,
    Offset16,
};

int run_case(std::int32_t output_rows, std::int32_t tokens, std::uint32_t seed,
             OperandPlacement placement = OperandPlacement::Aligned256) {
    const std::string label = "bf16_projection [" + std::to_string(output_rows) +
                              ",2560] T=" + std::to_string(tokens) +
                              (placement == OperandPlacement::Offset16 ? " (16-byte offset)" : "");
    std::vector<float> x(static_cast<std::size_t>(kInputRows) * tokens);
    std::vector<float> weight(static_cast<std::size_t>(output_rows) * kInputRows);
    fill_uniform(x, seed, -2.0F, 2.0F);
    fill_uniform(weight, seed + 1, -0.05F, 0.05F);
    round_to_bf16(x);
    round_to_bf16(weight);

    std::vector<double> expected(static_cast<std::size_t>(output_rows) * tokens);
    for (std::int32_t t = 0; t < tokens; ++t) {
        for (std::int32_t n = 0; n < output_rows; ++n) {
            double sum = 0.0;
            for (std::int32_t k = 0; k < kInputRows; ++k) {
                sum += static_cast<double>(weight[static_cast<std::size_t>(n) * kInputRows + k]) *
                       x[static_cast<std::size_t>(t) * kInputRows + k];
            }
            expected[static_cast<std::size_t>(t) * output_rows + n] = sum;
        }
    }

    const auto x_bits        = encode_bf16(x);
    const auto weight_bits   = encode_bf16(weight);
    const DType output_dtype = ops::bf16_projection_output_dtype(output_rows);
    const std::size_t y_bytes =
        expected.size() * (output_dtype == DType::BF16 ? sizeof(std::uint16_t) : sizeof(float));
    // The payload follows its leading guard; allocations are 256-byte aligned.
    const std::size_t guard_bytes = placement == OperandPlacement::Offset16 ? 256 + 16 : 256;
    GuardedDeviceBuffer device_x(x_bits.size() * sizeof(std::uint16_t), guard_bytes);
    GuardedDeviceBuffer device_weight(weight_bits.size() * sizeof(std::uint16_t), guard_bytes);
    GuardedDeviceBuffer device_y(y_bytes);
    device_x.copy_from_host(x_bits.data(), device_x.bytes());
    device_weight.copy_from_host(weight_bits.data(), device_weight.bytes());

    Tensor x_tensor(device_x.data(), DType::BF16, {kInputRows, tokens});
    Tensor weight_tensor(device_weight.data(), DType::BF16, {kInputRows, output_rows});
    Tensor y_tensor(device_y.data(), output_dtype, {output_rows, tokens});
    ops::bf16_projection(x_tensor, weight_tensor, y_tensor, nullptr);
    cuda_synchronize();

    int failures = 0;
    if (output_dtype == DType::BF16) {
        failures += verify_pointwise(label, from_device_bf16(device_y.data(), expected.size()),
                                     expected, bf16_output_criterion());
    } else {
        const auto y_words = from_device<float>(device_y.data(), expected.size());
        const std::vector<double> got(y_words.begin(), y_words.end());
        const bool gemv = tokens == 1 && output_rows >= 2560;
        failures +=
            verify_reduction(label, got, expected, gemv ? gemv_criterion() : mma_criterion());
    }
    failures += device_y.verify_guards(label + " y");
    failures += verify_exact((label + " x unchanged").c_str(),
                             from_device<std::uint16_t>(device_x.data(), x_bits.size()), x_bits);
    return failures;
}

} // namespace

int main() {
    if (require_cuda() != 0) { return 1; }
    try {
        int failures = 0;
        // Routes: the small-T MMA for T <= 48 (T = 1 too below 2560 rows; tiles of 16 tokens, and
        // 512/544 rows widen their CTAs above 32 tokens), the GEMV for T = 1 at 2560 and 10240
        // rows, and the GEMM beyond (128-token tiles; every T >= 2 at 10240 rows).
        for (const std::int32_t rows : {96, 128, 512, 544, 2560}) {
            for (const std::int32_t tokens :
                 {1, 2, 3, 4, 8, 15, 16, 17, 20, 32, 33, 47, 48, 49, 128, 300}) {
                failures += run_case(rows, tokens, 17U * static_cast<std::uint32_t>(rows + tokens));
            }
        }
        for (const std::int32_t tokens : {1, 2, 3, 4, 8, 15, 16, 20, 48, 49, 130}) {
            failures += run_case(10240, tokens, 991U + static_cast<std::uint32_t>(tokens));
        }
        // Each small-T schedule (per-tile 8-row, per-tile 16-row, whole-T) with 16-byte operands.
        for (const auto& [rows, tokens] :
             {std::pair{96, 20}, std::pair{128, 1}, std::pair{544, 16}, std::pair{544, 33},
              std::pair{512, 40}, std::pair{2560, 16}, std::pair{2560, 17}, std::pair{2560, 40}}) {
            failures += run_case(rows, tokens, 5U + static_cast<std::uint32_t>(rows + tokens),
                                 OperandPlacement::Offset16);
        }
        failures += expect_invalid_argument(
            [] {
                // Admission rejects the shape before any pointer is examined.
                Tensor x(nullptr, DType::BF16, {kInputRows, 1});
                Tensor weight(nullptr, DType::BF16, {kInputRows, 256});
                Tensor y(nullptr, DType::FP32, {256, 1});
                ops::bf16_projection(x, weight, y, nullptr);
            },
            "bf16_projection accepted an unadmitted [256,2560] weight");
        std::cout << (failures == 0 ? "OK" : "FAIL") << " bf16_projection\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "bf16_projection: " << error.what() << '\n';
        return 1;
    }
}
