#include "ninfer/ops/attn_input_proj.h"

#include "ops/input_projection_test_common.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::input_projection;

namespace {

// This criterion belongs to the complete A16 attention-input-projection Op.
constexpr ReductionCriterion kAttnInputProjA16Tolerance{2.9e-3, 4.0e-3, 4.5e-3};

int verify_output(std::string_view label, const GuardedBf16Tensor& output,
                  const quantized_weight::PackedWeight& weight, std::int32_t weight_row_offset,
                  std::int32_t output_rows, const std::vector<float>& activation,
                  std::int32_t hidden, std::int32_t tokens) {
    int failures = output.verify_guards(label);
    failures += output.verify_fully_written(label);
    const std::vector<double> actual =
        gather_rows(output.values(), output_rows, 0, output_rows, tokens);
    const std::vector<double> expected =
        projection_oracle(weight, weight_row_offset, output_rows, activation, hidden, tokens);
    failures += compare(label, actual, expected, kAttnInputProjA16Tolerance);
    return failures;
}

int run_q4_q5_case(DevicePackedWeight& query_key, DevicePackedWeight& gate_value,
                   std::int32_t tokens) {
    constexpr std::int32_t kHidden      = 5120;
    constexpr std::int32_t kQRows       = 6144;
    constexpr std::int32_t kKvRows      = 1024;
    const std::vector<float> activation = make_bf16_activation(kHidden, tokens, 101U + tokens);
    const std::vector<std::uint16_t> activation_bits = bf16_bits(activation);
    DeviceBuffer device_activation                   = to_device(activation_bits);

    GuardedBf16Tensor query(kQRows, tokens);
    GuardedBf16Tensor gate(kQRows, tokens);
    GuardedBf16Tensor key(kKvRows, tokens);
    GuardedBf16Tensor value(kKvRows, tokens);
    Tensor x(device_activation.p, DType::BF16, {kHidden, tokens});
    Tensor q = query.tensor();
    Tensor g = gate.tensor();
    Tensor k = key.tensor();
    Tensor v = value.tensor();
    WorkspaceArena workspace(1);

    ops::attn_input_proj(x, query_key.view(), gate_value.view(), q, g, k, v, workspace, nullptr);
    cuda_synchronize();

    const std::string suffix = " Q4/Q5 A16 T=" + std::to_string(tokens);
    int failures             = 0;
    failures += verify_output("attn q" + suffix, query, query_key.host, 0, kQRows, activation,
                              kHidden, tokens);
    failures += verify_output("attn k" + suffix, key, query_key.host, kQRows, kKvRows, activation,
                              kHidden, tokens);
    failures += verify_output("attn gate" + suffix, gate, gate_value.host, 0, kQRows, activation,
                              kHidden, tokens);
    failures += verify_output("attn value" + suffix, value, gate_value.host, kQRows, kKvRows,
                              activation, kHidden, tokens);
    failures += verify_preserved("attn x" + suffix, device_activation, activation_bits);
    failures += query_key.verify_preserved("attn query/key" + suffix);
    failures += gate_value.verify_preserved("attn gate/value" + suffix);
    return failures;
}

int run_q4_q5() {
    constexpr std::int32_t kHidden = 5120;
    constexpr std::int32_t kParent = 7168;
    DevicePackedWeight query_key(
        quantized_weight::make_patterned_weight(QType::Q4G64_F16S, kParent, kHidden, 103U));
    DevicePackedWeight gate_value(
        quantized_weight::make_patterned_weight(QType::Q5G64_F16S, kParent, kHidden, 107U));

    int failures = 0;
    for (const std::int32_t tokens : {1, 2, 16, 17}) {
        failures += run_q4_q5_case(query_key, gate_value, tokens);
    }
    return failures;
}

int run_w8_target_case(DevicePackedWeight& parent, std::int32_t tokens) {
    constexpr std::int32_t kHidden      = 2048;
    constexpr std::int32_t kQRows       = 4096;
    constexpr std::int32_t kKvRows      = 512;
    const std::vector<float> activation = make_bf16_activation(kHidden, tokens, 201U + tokens);
    const std::vector<std::uint16_t> activation_bits = bf16_bits(activation);
    DeviceBuffer device_activation                   = to_device(activation_bits);

    GuardedBf16Tensor query(kQRows, tokens);
    GuardedBf16Tensor gate(kQRows, tokens);
    GuardedBf16Tensor key(kKvRows, tokens);
    GuardedBf16Tensor value(kKvRows, tokens);
    Tensor x(device_activation.p, DType::BF16, {kHidden, tokens});
    Tensor q = query.tensor();
    Tensor g = gate.tensor();
    Tensor k = key.tensor();
    Tensor v = value.tensor();
    WorkspaceArena workspace(1);

    ops::attn_input_proj(x, parent.view(), q, g, k, v, workspace, nullptr);
    cuda_synchronize();

    const std::string suffix = " W8 target A16 T=" + std::to_string(tokens);
    int failures             = 0;
    failures += verify_output("attn q" + suffix, query, parent.host, 0, kQRows, activation, kHidden,
                              tokens);
    failures += verify_output("attn k" + suffix, key, parent.host, kQRows, kKvRows, activation,
                              kHidden, tokens);
    failures += verify_output("attn gate" + suffix, gate, parent.host, kQRows + kKvRows, kQRows,
                              activation, kHidden, tokens);
    failures += verify_output("attn value" + suffix, value, parent.host, 2 * kQRows + kKvRows,
                              kKvRows, activation, kHidden, tokens);
    failures += verify_preserved("attn x" + suffix, device_activation, activation_bits);
    failures += parent.verify_preserved("attn parent weight" + suffix);
    return failures;
}

int run_w8_target() {
    constexpr std::int32_t kHidden = 2048;
    DevicePackedWeight parent(
        quantized_weight::make_patterned_weight(QType::W8G32_F16S, 9216, kHidden, 211U));
    int failures = 0;
    for (const std::int32_t tokens : {1, 2, 17, 129}) {
        failures += run_w8_target_case(parent, tokens);
    }
    return failures;
}

int run_w8_companion_case(DevicePackedWeight& parent, std::int32_t tokens) {
    constexpr std::int32_t kHidden      = 2048;
    constexpr std::int32_t kQRows       = 4096;
    constexpr std::int32_t kKvRows      = 1024;
    const std::vector<float> activation = make_bf16_activation(kHidden, tokens, 301U + tokens);
    const std::vector<std::uint16_t> activation_bits = bf16_bits(activation);
    DeviceBuffer device_activation                   = to_device(activation_bits);

    GuardedBf16Tensor query(kQRows, tokens);
    GuardedBf16Tensor key(kKvRows, tokens);
    GuardedBf16Tensor value(kKvRows, tokens);
    Tensor x(device_activation.p, DType::BF16, {kHidden, tokens});
    Tensor q = query.tensor();
    Tensor k = key.tensor();
    Tensor v = value.tensor();
    WorkspaceArena workspace(1);

    ops::attn_input_proj(x, parent.view(), q, k, v, workspace, nullptr);
    cuda_synchronize();

    const std::string suffix = " W8 companion A16 T=" + std::to_string(tokens);
    int failures             = 0;
    failures += verify_output("attn q" + suffix, query, parent.host, 0, kQRows, activation, kHidden,
                              tokens);
    failures += verify_output("attn k" + suffix, key, parent.host, kQRows, kKvRows, activation,
                              kHidden, tokens);
    failures += verify_output("attn value" + suffix, value, parent.host, kQRows + kKvRows, kKvRows,
                              activation, kHidden, tokens);
    failures += verify_preserved("attn x" + suffix, device_activation, activation_bits);
    failures += parent.verify_preserved("attn parent weight" + suffix);
    return failures;
}

int run_w8_companion() {
    constexpr std::int32_t kHidden = 2048;
    DevicePackedWeight parent(
        quantized_weight::make_patterned_weight(QType::W8G32_F16S, 6144, kHidden, 307U));
    int failures = 0;
    // One public numerical case from every registered companion A16 T region.
    for (const std::int32_t tokens : {1, 2, 97, 193, 289, 321, 385, 449}) {
        failures += run_w8_companion_case(parent, tokens);
    }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    failures += run_q4_q5();
    failures += run_w8_target();
    failures += run_w8_companion();
    std::cout << (failures == 0 ? "OK" : "FAIL") << " attn_input_proj\n";
    return failures == 0 ? 0 : 1;
}
