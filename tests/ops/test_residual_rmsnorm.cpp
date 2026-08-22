#include "ninfer/ops/residual_rmsnorm.h"
#include "ops/norm_test_common.h"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::norm;

namespace {

// Residual stream is the same add as residual_add. Normalized output uses the offset-RMSNorm
// arithmetic profile.
constexpr PointwiseCriterion residual_stream_criterion() {
    return {/*absolute*/ 0.0, /*relative*/ 3.95e-3};
}

// Fused residual + offset RMSNorm from public BF16 y/x/weight. The kernel reduces the
// stored BF16 residual, so this independent FP64 fused formula is a looser profile than
// standalone rmsnorm (observed rel_l2≈2.51e-3, max_abs≈0.015 on D=5120).
constexpr ReductionCriterion residual_rmsnorm_out_criterion() {
    return {/*relative_l2*/ 3.0e-3, /*gross_absolute*/ 2.0e-4,
            /*gross_relative_to_max_reference*/ 5.0e-3};
}

void fused_oracle(const std::vector<float>& y, const std::vector<float>& x,
                  const std::vector<float>& weight, const Shape& shape,
                  std::vector<double>& residual, std::vector<double>& normalized) {
    residual.resize(x.size());
    normalized.resize(x.size());
    const auto row_count = static_cast<std::int64_t>(shape.rows) * shape.tokens;
    for (std::int64_t row = 0; row < row_count; ++row) {
        const std::size_t base = static_cast<std::size_t>(row) * shape.d;
        double sum_squares     = 0.0;
        for (std::int32_t column = 0; column < shape.d; ++column) {
            const double value = static_cast<double>(x[base + column]) +
                                 static_cast<double>(y[base + column]);
            residual[base + column] = value;
            sum_squares += value * value;
        }
        const double inverse = 1.0 / std::sqrt(sum_squares / static_cast<double>(shape.d) + kEps);
        for (std::int32_t column = 0; column < shape.d; ++column) {
            const double gain = static_cast<double>(weight[column]) + 1.0;
            normalized[base + column] = residual[base + column] * inverse * gain;
        }
    }
}

int run_case(const char* label, const Shape& shape, std::uint32_t seed, float input_scale = 4.0F,
             bool cancel = false) {
    const std::size_t count = shape.elements();
    std::vector<float> y(count), x(count), weight(shape.d);
    fill_uniform(y, seed, -input_scale, input_scale);
    fill_uniform(x, seed + 1U, -input_scale, input_scale);
    fill_uniform(weight, seed + 2U, -0.5F, 0.5F);
    round_to_bf16(y);
    round_to_bf16(x);
    round_to_bf16(weight);
    if (cancel) {
        for (std::size_t i = 0; i < count; ++i) { x[i] = -y[i]; }
        round_to_bf16(x);
    }

    std::vector<double> residual_ref;
    std::vector<double> normalized_ref;
    fused_oracle(y, x, weight, shape, residual_ref, normalized_ref);

    DeviceInput device_y      = make_input(y, false);
    DeviceInput device_weight = make_input(weight, false);
    GuardedDeviceBuffer device_x(count * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_out(count * sizeof(std::uint16_t));
    std::vector<std::uint16_t> x_bits(count);
    for (std::size_t i = 0; i < count; ++i) { x_bits[i] = f32_to_bf16(x[i]); }
    device_x.copy_from_host(x_bits.data(), device_x.bytes());
    device_out.fill(0xff);

    Tensor y_tensor      = tensor_for(device_y.data, shape);
    Tensor x_tensor      = tensor_for(device_x.data(), shape);
    Tensor weight_tensor(device_weight.data, DType::BF16, {shape.d});
    Tensor out_tensor    = tensor_for(device_out.data(), shape);
    ops::residual_rmsnorm(y_tensor, x_tensor, weight_tensor, kEps, out_tensor, nullptr);
    cuda_synchronize();

    int failures = verify_reduction(std::string(label) + " out",
                                    from_device_bf16(device_out.data(), count), normalized_ref,
                                    residual_rmsnorm_out_criterion());
    failures += verify_pointwise(std::string(label) + " residual",
                                 from_device_bf16(device_x.data(), count), residual_ref,
                                 residual_stream_criterion());
    failures += verify_output_storage(std::string(label) + " out storage", device_out, false);
    failures += device_x.verify_guards((std::string(label) + " residual").c_str());
    failures += verify_preserved(std::string(label) + " preserves y", device_y);
    failures += verify_preserved(std::string(label) + " preserves weight", device_weight);
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cerr << "FAIL: no usable CUDA device\n";
        return 1;
    }

    int failures = 0;
    failures += run_case("residual_rmsnorm [5120,1]", {5120, 1}, 2101U);
    failures += run_case("residual_rmsnorm [5120,4]", {5120, 4}, 2102U);
    failures += run_case("residual_rmsnorm [5120,64]", {5120, 64}, 2103U);
    failures += run_case("residual_rmsnorm [5120,128]", {5120, 128}, 2104U);
    failures += run_case("residual_rmsnorm near-zero [5120,4]", {5120, 4}, 2105U, 1.0e-5F);
    failures += run_case("residual_rmsnorm cancel [5120,4]", {5120, 4}, 2106U, 4.0F, true);
    std::cout << (failures ? "FAIL" : "OK") << " residual_rmsnorm\n";
    return failures ? 1 : 0;
}
