#include "ninfer/ops/grouped_dynamic_conv.h"

#include "ops/direct_bf16_weight.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::direct_bf16_weight;

namespace {

constexpr std::int32_t kD = ops::kGroupedDynamicConvHidden;
constexpr std::int32_t kG = ops::kGroupedDynamicConvGroups;
constexpr std::int32_t kGroup = ops::kGroupedDynamicConvGroupSize;
constexpr std::int32_t kProj = ops::kGroupedDynamicConvProjRows;

constexpr ReductionCriterion prepare_criterion() {
    return {/*relative_l2*/ 1.0 / 256.0, /*gross_absolute*/ 1.0 / 256.0,
            /*gross_relative_to_max_reference*/ 2.0 / 256.0};
}
constexpr PointwiseCriterion finish_criterion() {
    return {/*absolute*/ 1.0 / 256.0, /*relative*/ 4.0 / 256.0};
}

std::vector<std::uint16_t> encode_bf16(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) bits[i] = f32_to_bf16(values[i]);
    return bits;
}

std::size_t hid_index(std::int32_t d, std::int32_t t, std::int32_t b, std::int32_t tokens) {
    return (static_cast<std::size_t>(b) * tokens + t) * kD + d;
}

std::size_t dyn_index(std::int32_t g, std::int32_t offset, std::int32_t t, std::int32_t b,
                      std::int32_t tokens) {
    return ((static_cast<std::size_t>(b) * tokens + t) * 2 + offset) * kG + g;
}

std::size_t base_index(std::int32_t d, std::int32_t offset, std::int32_t phase) {
    return static_cast<std::size_t>(phase) * (kD * 2) + static_cast<std::size_t>(offset) * kD + d;
}

void convolve_oracle(const std::vector<float>& hidden, const std::vector<float>& base,
                     const std::vector<double>& dynamic, std::int32_t tokens, std::int32_t batch,
                     std::int32_t phase, std::vector<double>& out) {
    out.assign(static_cast<std::size_t>(kD) * tokens * batch, 0.0);
    for (std::int32_t b = 0; b < batch; ++b) {
        for (std::int32_t t = 0; t < tokens; ++t) {
            for (std::int32_t d = 0; d < kD; ++d) {
                const std::int32_t g = d / kGroup;
                const double x0      = static_cast<double>(hidden[hid_index(d, t, b, tokens)]);
                const double x1 =
                    t == 0 ? 0.0 : static_cast<double>(hidden[hid_index(d, t - 1, b, tokens)]);
                const double b0 = static_cast<double>(base[base_index(d, 0, phase)]);
                const double b1 = static_cast<double>(base[base_index(d, 1, phase)]);
                const double d0 = dynamic[dyn_index(g, 0, t, b, tokens)];
                const double d1 = dynamic[dyn_index(g, 1, t, b, tokens)];
                out[hid_index(d, t, b, tokens)] = (b0 + d0) * x0 + (b1 + d1) * x1;
            }
        }
    }
}

void projection_oracle(const HostWeight& weight, const std::vector<float>& hidden,
                       std::int32_t tokens, std::int32_t batch, std::vector<double>& proj) {
    proj.assign(static_cast<std::size_t>(kProj) * tokens * batch, 0.0);
    for (std::int32_t b = 0; b < batch; ++b) {
        for (std::int32_t t = 0; t < tokens; ++t) {
            std::vector<float> column(static_cast<std::size_t>(kD));
            for (std::int32_t d = 0; d < kD; ++d) {
                column[static_cast<std::size_t>(d)] = hidden[hid_index(d, t, b, tokens)];
            }
            const std::size_t col = static_cast<std::size_t>(b) * tokens + t;
            for (std::int32_t n = 0; n < kProj; ++n) {
                proj[col * kProj + n] = dot_fp64(weight, n, column);
            }
        }
    }
}

void split_projection(const std::vector<double>& proj, std::int32_t tokens, std::int32_t batch,
                      std::int32_t phase, std::vector<double>& dynamic) {
    dynamic.assign(static_cast<std::size_t>(kG) * 2 * tokens * batch, 0.0);
    for (std::int32_t b = 0; b < batch; ++b) {
        for (std::int32_t t = 0; t < tokens; ++t) {
            const std::size_t col = static_cast<std::size_t>(b) * tokens + t;
            for (std::int32_t offset = 0; offset < 2; ++offset) {
                for (std::int32_t g = 0; g < kG; ++g) {
                    dynamic[dyn_index(g, offset, t, b, tokens)] =
                        proj[col * kProj + phase * 640 + offset * kG + g];
                }
            }
        }
    }
}

int run_prepare_case(const char* label, std::int32_t tokens, std::int32_t batch,
                     std::uint32_t seed) {
    HostWeight host_weight = make_patterned(kProj, kD, seed);
    DeviceWeight device_weight(std::move(host_weight));

    std::vector<float> hidden(static_cast<std::size_t>(kD) * tokens * batch);
    std::vector<float> base(static_cast<std::size_t>(kD) * 4);
    fill_uniform(hidden, seed + 11, -1.0f, 1.0f);
    fill_uniform(base, seed + 23, -0.5f, 0.5f);
    round_to_bf16(hidden);
    round_to_bf16(base);

    std::vector<double> proj;
    projection_oracle(device_weight.host, hidden, tokens, batch, proj);
    std::vector<double> prepare_dyn;
    std::vector<double> finish_dyn;
    split_projection(proj, tokens, batch, 0, prepare_dyn);
    split_projection(proj, tokens, batch, 1, finish_dyn);
    std::vector<double> expected_prepared;
    convolve_oracle(hidden, base, prepare_dyn, tokens, batch, 0, expected_prepared);

    const auto hidden_bits = encode_bf16(hidden);
    const auto base_bits   = encode_bf16(base);
    GuardedDeviceBuffer device_hidden(hidden_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_base(base_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_prepared(hidden_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_finish(finish_dyn.size() * sizeof(std::uint16_t));
    device_hidden.copy_from_host(hidden_bits.data(), device_hidden.bytes());
    device_base.copy_from_host(base_bits.data(), device_base.bytes());
    device_prepared.fill(0xcd);
    device_finish.fill(0xcd);

    Tensor hidden_t = batch == 1 ? Tensor(device_hidden.data(), DType::BF16, {kD, tokens})
                                 : Tensor(device_hidden.data(), DType::BF16, {kD, tokens, batch});
    Tensor base_t(device_base.data(), DType::BF16, {kD, 2, 2});
    Tensor prepared_t = batch == 1 ? Tensor(device_prepared.data(), DType::BF16, {kD, tokens})
                                   : Tensor(device_prepared.data(), DType::BF16, {kD, tokens, batch});
    Tensor finish_t =
        batch == 1 ? Tensor(device_finish.data(), DType::BF16, {kG, 2, tokens})
                   : Tensor(device_finish.data(), DType::BF16, {kG, 2, tokens, batch});

    const std::size_t workspace_bytes =
        ops::grouped_dynamic_conv_prepare_workspace_capacity_bytes(QType::BF16_CTRL, tokens, tokens,
                                                                   batch);
    WorkspaceArena workspace(std::max<std::size_t>(256, workspace_bytes));
    ops::grouped_dynamic_conv_prepare(hidden_t, base_t, device_weight.view(), prepared_t, finish_t,
                                      workspace, nullptr);
    cuda_synchronize();

    int failures = verify_reduction(label, from_device_bf16(device_prepared.data(), hidden.size()),
                                    expected_prepared, prepare_criterion());
    const std::string finish_label = std::string(label) + " finish_dynamic";
    failures += verify_reduction(finish_label.c_str(),
                                 from_device_bf16(device_finish.data(), finish_dyn.size()),
                                 finish_dyn, prepare_criterion());
    failures += device_hidden.verify_guards("grouped_dynamic_conv hidden");
    failures += device_prepared.verify_guards("grouped_dynamic_conv prepared");
    failures += device_finish.verify_guards("grouped_dynamic_conv finish_dynamic");
    failures += device_weight.verify_preserved("grouped_dynamic_conv weight");
    return failures;
}

int run_finish_case(const char* label, std::int32_t tokens, std::int32_t batch,
                    std::uint32_t seed) {
    std::vector<float> hidden(static_cast<std::size_t>(kD) * tokens * batch);
    std::vector<float> base(static_cast<std::size_t>(kD) * 4);
    std::vector<float> dynamic(static_cast<std::size_t>(kG) * 2 * tokens * batch);
    fill_uniform(hidden, seed + 31, -1.0f, 1.0f);
    fill_uniform(base, seed + 41, -0.5f, 0.5f);
    fill_uniform(dynamic, seed + 51, -0.25f, 0.25f);
    round_to_bf16(hidden);
    round_to_bf16(base);
    round_to_bf16(dynamic);

    std::vector<double> dynamic64(dynamic.size());
    for (std::size_t i = 0; i < dynamic.size(); ++i) {
        dynamic64[i] = static_cast<double>(dynamic[i]);
    }
    std::vector<double> expected;
    convolve_oracle(hidden, base, dynamic64, tokens, batch, 1, expected);

    const auto hidden_bits  = encode_bf16(hidden);
    const auto base_bits    = encode_bf16(base);
    const auto dynamic_bits = encode_bf16(dynamic);
    GuardedDeviceBuffer device_hidden(hidden_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_base(base_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_dynamic(dynamic_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_out(hidden_bits.size() * sizeof(std::uint16_t));
    device_hidden.copy_from_host(hidden_bits.data(), device_hidden.bytes());
    device_base.copy_from_host(base_bits.data(), device_base.bytes());
    device_dynamic.copy_from_host(dynamic_bits.data(), device_dynamic.bytes());
    device_out.fill(0xcd);

    Tensor hidden_t = batch == 1 ? Tensor(device_hidden.data(), DType::BF16, {kD, tokens})
                                 : Tensor(device_hidden.data(), DType::BF16, {kD, tokens, batch});
    Tensor base_t(device_base.data(), DType::BF16, {kD, 2, 2});
    Tensor dyn_t = batch == 1 ? Tensor(device_dynamic.data(), DType::BF16, {kG, 2, tokens})
                              : Tensor(device_dynamic.data(), DType::BF16, {kG, 2, tokens, batch});
    Tensor out_t = batch == 1 ? Tensor(device_out.data(), DType::BF16, {kD, tokens})
                              : Tensor(device_out.data(), DType::BF16, {kD, tokens, batch});
    ops::grouped_dynamic_conv_finish(hidden_t, base_t, dyn_t, out_t, nullptr);
    cuda_synchronize();

    int failures = verify_pointwise(label, from_device_bf16(device_out.data(), hidden.size()),
                                    expected, finish_criterion());
    failures += device_out.verify_guards("grouped_dynamic_conv finish out");
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    int failures = 0;
    failures += run_finish_case("grouped_dynamic_conv_finish T=8", 8, 1, 101u);
    failures += run_finish_case("grouped_dynamic_conv_finish T=1", 1, 1, 102u);
    failures += run_finish_case("grouped_dynamic_conv_finish T=4 B=2", 4, 2, 103u);
    failures += run_prepare_case("grouped_dynamic_conv_prepare T=8", 8, 1, 201u);
    failures += run_prepare_case("grouped_dynamic_conv_prepare T=1", 1, 1, 202u);
    std::cout << (failures ? "FAIL" : "OK") << " grouped_dynamic_conv\n";
    return failures ? 1 : 0;
}
