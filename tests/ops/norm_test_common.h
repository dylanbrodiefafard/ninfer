#pragma once

#include "ops/op_tester.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ninfer::test::norm {

inline constexpr float kEps = 1.0e-6F;

struct Shape {
    std::int32_t d;
    std::int32_t rows;
    std::int32_t tokens = 1;

    std::size_t elements() const {
        return static_cast<std::size_t>(d) * static_cast<std::size_t>(rows) *
               static_cast<std::size_t>(tokens);
    }
};

inline Tensor tensor_for(void* data, const Shape& shape) {
    if (shape.tokens == 1) return Tensor(data, DType::BF16, {shape.d, shape.rows});
    return Tensor(data, DType::BF16, {shape.d, shape.rows, shape.tokens});
}

inline constexpr ReductionCriterion rmsnorm_bf16_criterion() {
    return {/*relative_l2*/ 1.85e-3, /*gross_absolute*/ 1.0e-5,
            /*gross_relative_to_max_reference*/ 3.4e-3};
}

// The RMS family oracle evaluates each complete row naively in FP64 from represented BF16
// values. Output rounding and private reduction or staging choices are absent from this formula.
inline std::vector<double> rmsnorm_oracle(const std::vector<float>& input,
                                          const std::vector<float>& weight, const Shape& shape,
                                          bool unit_offset, float eps = kEps) {
    std::vector<double> output(input.size());
    const auto row_count = static_cast<std::int64_t>(shape.rows) * shape.tokens;
    for (std::int64_t row = 0; row < row_count; ++row) {
        const std::size_t base = static_cast<std::size_t>(row) * shape.d;
        double sum_squares     = 0.0;
        for (std::int32_t column = 0; column < shape.d; ++column) {
            const double value = input[base + column];
            sum_squares += value * value;
        }
        const double inverse = 1.0 / std::sqrt(sum_squares / static_cast<double>(shape.d) + eps);
        for (std::int32_t column = 0; column < shape.d; ++column) {
            const double gain     = static_cast<double>(weight[column]) + (unit_offset ? 1.0 : 0.0);
            output[base + column] = static_cast<double>(input[base + column]) * inverse * gain;
        }
    }
    return output;
}

struct DeviceInput {
    DeviceBuffer storage;
    void* data = nullptr;
    std::vector<std::uint16_t> expected;
};

inline DeviceInput make_input(const std::vector<float>& values, bool bf16x2_unaligned) {
    const std::size_t leading = bf16x2_unaligned ? 1 : 0;
    DeviceInput input;
    input.expected.resize(leading + values.size(), 0x5a5aU);
    for (std::size_t index = 0; index < values.size(); ++index) {
        input.expected[leading + index] = f32_to_bf16(values[index]);
    }
    input.storage = to_device(input.expected);
    input.data    = static_cast<std::uint16_t*>(input.storage.p) + leading;
    return input;
}

inline int verify_preserved(const std::string& label, const DeviceInput& input) {
    return verify_exact(label.c_str(),
                        from_device<std::uint16_t>(input.storage, input.expected.size()),
                        input.expected);
}

inline int verify_output_storage(const std::string& label, const GuardedDeviceBuffer& output,
                                 bool bf16x2_unaligned) {
    int failures = output.verify_guards(label);
    if (bf16x2_unaligned) {
        failures +=
            verify_exact((label + " prefix").c_str(), from_device<std::uint16_t>(output.data(), 1),
                         std::vector<std::uint16_t>{0xffffU});
    }
    return failures;
}

} // namespace ninfer::test::norm
