#include "ninfer/ops/rmsnorm.h"
#include "ops/norm_test_common.h"
#include "core/decode_graph.h"
#include "core/device.h"

#include <array>
#include <cstdint>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::norm;

namespace {

constexpr ReductionCriterion dual_rmsnorm_bf16_criterion() {
    // Nearest BF16 rounding can require max_reference/256 at a binade boundary. The gross
    // coefficient adds 9.375e-5 for the FP32 normalization profile; the RMS normwise bound is
    // unchanged. This panel-local profile admits the full independent offset-gain fixtures.
    return {/*relative_l2*/ 1.85e-3, /*gross_absolute*/ 1.0e-5,
            /*gross_relative_to_max_reference*/ 4.0e-3};
}

int run_case(int tokens, float eps, float scale0, float scale1, bool unaligned, bool capture,
             const DeviceContext& context) {
    constexpr int kWidth = 5120;
    const Shape shape{kWidth, tokens};
    const std::size_t elements = shape.elements();
    std::vector<float> input0(elements), input1(elements), weight0(kWidth), weight1(kWidth);
    fill_uniform(input0, 501U, -scale0, scale0);
    fill_uniform(input1, 503U, -scale1, scale1);
    fill_uniform(weight0, 505U, -1.5F, 0.5F);
    fill_uniform(weight1, 507U, -0.5F, 1.5F);
    // Exact zero gain and opposite signs exercise independent offset weights.
    weight0[0] = -1.0F;
    weight1[0] = 1.0F;
    round_to_bf16(input0);
    round_to_bf16(input1);
    round_to_bf16(weight0);
    round_to_bf16(weight1);
    auto reference0           = rmsnorm_oracle(input0, weight0, shape, true, eps);
    auto reference1           = rmsnorm_oracle(input1, weight1, shape, true, eps);
    auto device_input0        = make_input(input0, unaligned);
    auto device_input1        = make_input(input1, false);
    auto device_weight0       = make_input(weight0, false);
    auto device_weight1       = make_input(weight1, unaligned);
    const std::size_t leading = unaligned ? sizeof(std::uint16_t) : 0;
    GuardedDeviceBuffer output0(leading + elements * sizeof(std::uint16_t));
    GuardedDeviceBuffer output1(elements * sizeof(std::uint16_t));
    // Guard construction initializes on the default stream; compute uses a nonblocking stream.
    cuda_synchronize();
    auto* output_data0 = static_cast<std::uint8_t*>(output0.data()) + leading;
    Tensor x0          = tensor_for(device_input0.data, shape);
    Tensor x1          = tensor_for(device_input1.data, shape);
    Tensor w0(device_weight0.data, DType::BF16, {kWidth});
    Tensor w1(device_weight1.data, DType::BF16, {kWidth});
    Tensor y0         = tensor_for(output_data0, shape);
    Tensor y1         = tensor_for(output1.data(), shape);
    const auto launch = [&] {
        ops::dual_offset_rmsnorm(x0, w0, x1, w1, eps, y0, y1, context.stream);
    };
    DecodeGraphDefinition definition;
    DecodeGraphExecutable executable;
    if (capture) {
        definition.capture(context.stream, launch);
        executable.instantiate(definition);
    }
    const std::string label = "dual offset RMS T=" + std::to_string(tokens) +
                              (unaligned ? " unaligned" : " aligned") +
                              (capture ? " graph" : " eager");
    int failures            = 0;
    for (int replay = 0; replay < (capture ? 3 : 1); ++replay) {
        // Fresh output sentinels expose incomplete writes on every replay.
        CUDA_CHECK(cudaMemsetAsync(output0.data(), 0xff, output0.bytes(), context.stream));
        CUDA_CHECK(cudaMemsetAsync(output1.data(), 0xff, output1.bytes(), context.stream));
        if (capture) {
            executable.launch(context.stream);
        } else {
            launch();
        }
        context.synchronize();
        failures += verify_reduction(label + " panel0", from_device_bf16(output_data0, elements),
                                     reference0, dual_rmsnorm_bf16_criterion());
        failures += verify_reduction(label + " panel1", from_device_bf16(output1.data(), elements),
                                     reference1, dual_rmsnorm_bf16_criterion());
        failures += verify_output_storage(label + " output0", output0, unaligned);
        failures += verify_output_storage(label + " output1", output1, false);
        failures += verify_preserved(label + " input0", device_input0);
        failures += verify_preserved(label + " input1", device_input1);
        failures += verify_preserved(label + " weight0", device_weight0);
        failures += verify_preserved(label + " weight1", device_weight1);
        if (capture && replay < 2) {
            // Reuse captured addresses with changed panel contents, as compact request rows do.
            input0.swap(input1);
            const std::size_t prefix0 = unaligned ? 1 : 0;
            for (std::size_t index = 0; index < elements; ++index) {
                device_input0.expected[prefix0 + index] = f32_to_bf16(input0[index]);
                device_input1.expected[index]           = f32_to_bf16(input1[index]);
            }
            device_input0.storage.copy_from_host(device_input0.expected.data(),
                                                 device_input0.storage.bytes);
            device_input1.storage.copy_from_host(device_input1.expected.data(),
                                                 device_input1.storage.bytes);
            reference0 = rmsnorm_oracle(input0, weight0, shape, true, eps);
            reference1 = rmsnorm_oracle(input1, weight1, shape, true, eps);
        }
    }
    return failures;
}

} // namespace

int main() {
    try {
        if (require_cuda() != 0) { return 1; }
        const DeviceContext context;
        int failures = 0;
        constexpr std::array extents{1, 2, 3, 4, 5, 6, 8, 10, 12, 18, 24, 30, 36, 128};
        for (const int tokens : extents) {
            failures += run_case(tokens, kEps, 4.0F, 0.125F, false, tokens == 6, context);
        }
        failures += run_case(2, kEps, 0.0F, 1.0e-5F, false, false, context);
        failures += run_case(6, 1.0e-3F, 1.0e-5F, 3.0F, false, true, context);
        failures += run_case(1, kEps, 4.0F, 0.125F, true, false, context);
        failures += run_case(36, kEps, 4.0F, 0.125F, true, true, context);
        std::cout << (failures ? "FAIL" : "OK") << " dual offset RMSNorm\n";
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << "dual offset RMSNorm: " << error.what() << '\n';
        return 1;
    }
}
