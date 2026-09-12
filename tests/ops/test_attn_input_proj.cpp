#include "ninfer/ops/attn_input_proj.h"
#include "core/device.h"
#include "core/decode_graph.h"

#include "ops/direct_bf16_weight.h"
#include "ops/input_projection_test_common.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <span>
#include <string>
#include <thread>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::direct_bf16_weight;
using namespace ninfer::test::input_projection;

namespace {

// This criterion belongs to the complete A16 attention-input-projection Op.
constexpr ReductionCriterion kAttnInputProjA16Tolerance{2.9e-3, 4.0e-3, 4.5e-3};
constexpr ReductionCriterion kAttnInputProjA4Tolerance{0.16, 1.0 / 256.0, 0.16};
// Retain the original seven grid points while stabilizing the distribution-level A4 criterion.
constexpr std::int32_t kA4SampleRows = 31;

int verify_output(std::string_view label, const GuardedBf16Tensor& output,
                  const quantized_weight::PackedWeight& weight, std::int32_t weight_row_offset,
                  std::int32_t output_rows, const std::vector<float>& activation,
                  std::int32_t hidden, std::int32_t tokens,
                  const ReductionCriterion& criterion = kAttnInputProjA16Tolerance,
                  std::int32_t sample_count           = 7) {
    int failures = output.verify_guards(label);
    failures += output.verify_fully_written(label);
    const std::vector<double> actual =
        gather_rows(output.values(), output_rows, 0, output_rows, tokens, sample_count);
    const std::vector<double> expected = projection_oracle(
        weight, weight_row_offset, output_rows, activation, hidden, tokens, sample_count);
    failures += compare(label, actual, expected, criterion);
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
    ops::attn_input_proj(x, query_key.view(), gate_value.view(), q, g, k, v, nullptr);
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
    for (const std::int32_t tokens : {1, 16, 17, 21, 48}) {
        failures += run_q4_q5_case(query_key, gate_value, tokens);
    }
    return failures;
}

std::vector<double> bf16_attention_oracle(const HostWeight& weight,
                                          std::span<const float> activation) {
    std::vector<double> result(static_cast<std::size_t>(weight.n));
    const unsigned available   = std::max(1U, std::thread::hardware_concurrency());
    const std::int32_t threads = std::min(weight.n, static_cast<std::int32_t>(available));
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(threads));
    for (std::int32_t thread = 0; thread < threads; ++thread) {
        const std::int32_t begin =
            static_cast<std::int32_t>((static_cast<std::int64_t>(weight.n) * thread) / threads);
        const std::int32_t end = static_cast<std::int32_t>(
            (static_cast<std::int64_t>(weight.n) * (thread + 1)) / threads);
        workers.emplace_back([&, begin, end] {
            for (std::int32_t row = begin; row < end; ++row) {
                result[static_cast<std::size_t>(row)] = dot_fp64(weight, row, activation);
            }
        });
    }
    for (std::thread& worker : workers) { worker.join(); }
    return result;
}

int verify_direct_output(std::string_view label, const GuardedBf16Tensor& output,
                         std::span<const double> expected) {
    int failures = output.verify_guards(label);
    failures += output.verify_fully_written(label);
    failures +=
        compare(label, output.values(), std::vector<double>(expected.begin(), expected.end()),
                kAttnInputProjA16Tolerance);
    return failures;
}

std::vector<std::int32_t> sampled_tokens(std::int32_t tokens) {
    if (tokens <= 32) {
        std::vector<std::int32_t> result(static_cast<std::size_t>(tokens));
        for (std::int32_t token = 0; token < tokens; ++token) {
            result[static_cast<std::size_t>(token)] = token;
        }
        return result;
    }
    std::vector<std::int32_t> result{0, 1, tokens / 2, tokens - 2, tokens - 1};
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

int verify_direct_output_sampled(std::string_view label, const GuardedBf16Tensor& output,
                                 const HostWeight& weight, std::int32_t parent_row_offset,
                                 std::int32_t output_rows, const std::vector<float>& activation,
                                 std::int32_t hidden, std::int32_t tokens) {
    int failures = output.verify_guards(label);
    failures += output.verify_fully_written(label);
    const std::vector<std::int32_t> rows          = sampled_rows(output_rows);
    const std::vector<std::int32_t> token_samples = sampled_tokens(tokens);
    const std::vector<double> values              = output.values();
    std::vector<double> actual;
    std::vector<double> expected;
    actual.reserve(rows.size() * token_samples.size());
    expected.reserve(actual.capacity());
    for (const std::int32_t local_row : rows) {
        for (const std::int32_t token : token_samples) {
            actual.push_back(values[static_cast<std::size_t>(token) * output_rows + local_row]);
            expected.push_back(dot_fp64(
                weight, parent_row_offset + local_row,
                std::span<const float>(activation.data() + static_cast<std::size_t>(token) * hidden,
                                       hidden)));
        }
    }
    failures += compare(label, actual, expected, kAttnInputProjA16Tolerance);
    return failures;
}

int run_bf16_target_case(DeviceWeight& parent, std::int32_t tokens) {
    constexpr std::int32_t kHidden      = 5120;
    constexpr std::int32_t kQRows       = 6144;
    constexpr std::int32_t kKvRows      = 1024;
    constexpr std::int32_t kParentRows  = 14336;
    const std::vector<float> activation = make_bf16_activation(kHidden, tokens, 317U + tokens);
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
    ops::attn_input_proj(x, parent.view(), q, g, k, v, nullptr);
    cuda_synchronize();

    constexpr std::int32_t kKeyBegin   = kQRows;
    constexpr std::int32_t kGateBegin  = kKeyBegin + kKvRows;
    constexpr std::int32_t kValueBegin = kGateBegin + kQRows;
    const std::string suffix           = " BF16 A16 T=" + std::to_string(tokens);
    int failures                       = 0;
    if (tokens == 1) {
        const std::vector<double> expected = bf16_attention_oracle(parent.host, activation);
        failures += verify_direct_output(
            "attn q" + suffix, query,
            std::span<const double>(expected.data(), static_cast<std::size_t>(kQRows)));
        failures +=
            verify_direct_output("attn k" + suffix, key,
                                 std::span<const double>(expected.data() + kKeyBegin,
                                                         static_cast<std::size_t>(kKvRows)));
        failures += verify_direct_output("attn gate" + suffix, gate,
                                         std::span<const double>(expected.data() + kGateBegin,
                                                                 static_cast<std::size_t>(kQRows)));
        failures +=
            verify_direct_output("attn value" + suffix, value,
                                 std::span<const double>(expected.data() + kValueBegin,
                                                         static_cast<std::size_t>(kKvRows)));
    } else {
        failures += verify_direct_output_sampled("attn q" + suffix, query, parent.host, 0, kQRows,
                                                 activation, kHidden, tokens);
        failures += verify_direct_output_sampled("attn k" + suffix, key, parent.host, kKeyBegin,
                                                 kKvRows, activation, kHidden, tokens);
        failures += verify_direct_output_sampled("attn gate" + suffix, gate, parent.host,
                                                 kGateBegin, kQRows, activation, kHidden, tokens);
        failures += verify_direct_output_sampled("attn value" + suffix, value, parent.host,
                                                 kValueBegin, kKvRows, activation, kHidden, tokens);
    }
    failures += verify_preserved("attn x" + suffix, device_activation, activation_bits);
    failures += parent.verify_preserved("attn parent" + suffix);
    return failures;
}

int run_bf16_w5_panels(DeviceWeight& parent, std::int32_t tokens) {
    constexpr std::int32_t kPanel  = 5;
    constexpr std::int32_t kHidden = 5120;
    constexpr std::int32_t kQRows  = 6144;
    constexpr std::int32_t kKvRows = 1024;
    const std::vector<float> activation = make_bf16_activation(kHidden, tokens, 359U + tokens);
    const std::vector<std::uint16_t> activation_bits = bf16_bits(activation);
    DeviceBuffer device_activation = to_device(activation_bits);

    GuardedBf16Tensor packed_q(kQRows, tokens), packed_gate(kQRows, tokens);
    GuardedBf16Tensor packed_k(kKvRows, tokens), packed_value(kKvRows, tokens);
    GuardedBf16Tensor panel_q(kQRows, tokens), panel_gate(kQRows, tokens);
    GuardedBf16Tensor panel_k(kKvRows, tokens), panel_value(kKvRows, tokens);
    Tensor x(device_activation.p, DType::BF16, {kHidden, tokens});
    Tensor pq = packed_q.tensor(), pg = packed_gate.tensor(), pk = packed_k.tensor();
    Tensor pv = packed_value.tensor(), cq = panel_q.tensor(), cg = panel_gate.tensor();
    Tensor ck = panel_k.tensor(), cv = panel_value.tensor();
    DeviceArena workspace(256);
    ops::attn_input_proj(x, parent.view(), pq, pg, pk, pv, ops::LinearPolicy::A16Only, workspace,
                         nullptr);
    for (std::int32_t offset = 0; offset < tokens; offset += kPanel) {
        Tensor panel_x = x.slice(1, offset, kPanel);
        Tensor out_q = cq.slice(1, offset, kPanel), out_g = cg.slice(1, offset, kPanel);
        Tensor out_k = ck.slice(1, offset, kPanel), out_v = cv.slice(1, offset, kPanel);
        ops::attn_input_proj(panel_x, parent.view(), out_q, out_g, out_k, out_v,
                             ops::LinearPolicy::A16Only, workspace, nullptr);
    }
    cuda_synchronize();

    int failures = 0;
    const auto exact = [&](std::string_view name, const GuardedBf16Tensor& packed,
                           const GuardedBf16Tensor& panels) {
        if (packed.bits() == panels.bits()) { return; }
        std::cerr << "attn " << name << " BF16 A16 W5 panels T=" << tokens
                  << ": packed output differs from panels\n";
        ++failures;
    };
    exact("q", packed_q, panel_q);
    exact("gate", packed_gate, panel_gate);
    exact("k", packed_k, panel_k);
    exact("value", packed_value, panel_value);
    for (const auto& [name, output] :
         std::initializer_list<std::pair<std::string_view, const GuardedBf16Tensor*>>{
             {"packed q", &packed_q},       {"packed gate", &packed_gate},
             {"packed k", &packed_k},       {"packed value", &packed_value},
             {"panel q", &panel_q},         {"panel gate", &panel_gate},
             {"panel k", &panel_k},         {"panel value", &panel_value},
         }) {
        failures += output->verify_guards(name);
        failures += output->verify_fully_written(name);
    }
    failures += verify_preserved("attn BF16 W5 panels x", device_activation, activation_bits);
    failures += parent.verify_preserved("attn BF16 W5 panels parent");
    return failures;
}

int run_bf16_target() {
    constexpr std::int32_t kHidden     = 5120;
    constexpr std::int32_t kParentRows = 14336;
    DeviceWeight parent(make_patterned(kParentRows, kHidden, 313U));
    int failures = 0;
    if (ops::attn_input_proj_workspace_capacity_bytes(QType::BF16_CTRL, kParentRows, kHidden,
                                                      ops::LinearPolicy::A16Only, 1, 1024) != 0) {
        std::cerr << "BF16 attention input workspace interval is not zero-capacity\n";
        ++failures;
    }
    for (const std::int32_t tokens : {1, 2, 10, 15, 20, 22, 23, 128}) {
        failures += run_bf16_target_case(parent, tokens);
    }
    for (const std::int32_t tokens : {10, 15, 20}) {
        failures += run_bf16_w5_panels(parent, tokens);
    }
    return failures;
}

int run_nvfp4_target_case(DevicePackedWeight& parent, std::int32_t tokens,
                          ops::LinearPolicy policy = ops::LinearPolicy::A16Only) {
    constexpr std::int32_t kHidden = 5120;
    constexpr std::int32_t kQRows  = 6144;
    constexpr std::int32_t kKvRows = 1024;
    const std::vector<float> activation =
        make_bf16_activation(kHidden, tokens, 337U + static_cast<std::uint32_t>(tokens));
    const std::vector<std::uint16_t> activation_bits = bf16_bits(activation);
    DeviceBuffer device_activation                   = to_device(activation_bits);

    GuardedBf16Tensor query(kQRows, tokens);
    GuardedBf16Tensor gate(kQRows, tokens);
    GuardedBf16Tensor key(kKvRows, tokens);
    GuardedBf16Tensor value(kKvRows, tokens);
    Tensor x(device_activation.p, DType::BF16, {kHidden, tokens});
    Tensor q                   = query.tensor();
    Tensor g                   = gate.tensor();
    Tensor k                   = key.tensor();
    Tensor v                   = value.tensor();
    const std::size_t capacity = ops::attn_input_proj_workspace_capacity_bytes(
        QType::NVFP4, 14336, kHidden, policy, tokens, tokens);
    DeviceArena workspace(std::max<std::size_t>(capacity, 256));
    ops::attn_input_proj(x, parent.view(), q, g, k, v, policy, workspace, nullptr);
    cuda_synchronize();

    constexpr std::int32_t kKeyBegin   = kQRows;
    constexpr std::int32_t kGateBegin  = kKeyBegin + kKvRows;
    constexpr std::int32_t kValueBegin = kGateBegin + kQRows;
    int failures                       = 0;
    const bool a4                      = policy == ops::LinearPolicy::AllowA4 && tokens >= 4;
    const ReductionCriterion& criterion =
        a4 ? kAttnInputProjA4Tolerance : kAttnInputProjA16Tolerance;
    const std::int32_t sample_count = a4 ? kA4SampleRows : 7;
    const std::string suffix =
        std::string(" NVFP4 ") + (a4 ? "A4" : "A16") + " T=" + std::to_string(tokens);
    failures += verify_output("attn q" + suffix, query, parent.host, 0, kQRows, activation, kHidden,
                              tokens, criterion, sample_count);
    failures += verify_output("attn k" + suffix, key, parent.host, kKeyBegin, kKvRows, activation,
                              kHidden, tokens, criterion, sample_count);
    failures += verify_output("attn gate" + suffix, gate, parent.host, kGateBegin, kQRows,
                              activation, kHidden, tokens, criterion, sample_count);
    failures += verify_output("attn value" + suffix, value, parent.host, kValueBegin, kKvRows,
                              activation, kHidden, tokens, criterion, sample_count);
    failures += verify_preserved("attn x" + suffix, device_activation, activation_bits);
    failures += parent.verify_preserved("attn parent" + suffix);
    return failures;
}

int run_nvfp4_panels(DevicePackedWeight& parent, std::int32_t tokens, std::int32_t panel) {
    constexpr std::int32_t kHidden = 5120;
    constexpr std::int32_t kQRows  = 6144;
    constexpr std::int32_t kKvRows = 1024;
    const std::vector<float> activation = make_bf16_activation(kHidden, tokens, 373U + tokens);
    const std::vector<std::uint16_t> activation_bits = bf16_bits(activation);
    DeviceBuffer device_activation = to_device(activation_bits);

    GuardedBf16Tensor packed_q(kQRows, tokens), packed_gate(kQRows, tokens);
    GuardedBf16Tensor packed_k(kKvRows, tokens), packed_value(kKvRows, tokens);
    GuardedBf16Tensor panel_q(kQRows, tokens), panel_gate(kQRows, tokens);
    GuardedBf16Tensor panel_k(kKvRows, tokens), panel_value(kKvRows, tokens);
    Tensor x(device_activation.p, DType::BF16, {kHidden, tokens});
    Tensor pq = packed_q.tensor(), pg = packed_gate.tensor(), pk = packed_k.tensor();
    Tensor pv = packed_value.tensor(), cq = panel_q.tensor(), cg = panel_gate.tensor();
    Tensor ck = panel_k.tensor(), cv = panel_value.tensor();
    DeviceArena workspace(256);
    ops::attn_input_proj(x, parent.view(), pq, pg, pk, pv, ops::LinearPolicy::A16Only, workspace,
                         nullptr);
    for (std::int32_t offset = 0; offset < tokens; offset += panel) {
        Tensor panel_x = x.slice(1, offset, panel);
        Tensor out_q = cq.slice(1, offset, panel), out_g = cg.slice(1, offset, panel);
        Tensor out_k = ck.slice(1, offset, panel), out_v = cv.slice(1, offset, panel);
        ops::attn_input_proj(panel_x, parent.view(), out_q, out_g, out_k, out_v,
                             ops::LinearPolicy::A16Only, workspace, nullptr);
    }
    cuda_synchronize();

    int failures = 0;
    const auto exact = [&](std::string_view name, const GuardedBf16Tensor& packed,
                           const GuardedBf16Tensor& panels) {
        if (packed.bits() == panels.bits()) { return; }
        std::cerr << "attn " << name << " NVFP4 A16 W" << panel << " panels T=" << tokens
                  << ": packed output differs from panels\n";
        ++failures;
    };
    exact("q", packed_q, panel_q);
    exact("gate", packed_gate, panel_gate);
    exact("k", packed_k, panel_k);
    exact("value", packed_value, panel_value);
    for (const auto& [name, output] :
         std::initializer_list<std::pair<std::string_view, const GuardedBf16Tensor*>>{
             {"packed q", &packed_q},       {"packed gate", &packed_gate},
             {"packed k", &packed_k},       {"packed value", &packed_value},
             {"panel q", &panel_q},         {"panel gate", &panel_gate},
             {"panel k", &panel_k},         {"panel value", &panel_value},
         }) {
        failures += output->verify_guards(name);
        failures += output->verify_fully_written(name);
    }
    failures += verify_preserved("attn NVFP4 panels x", device_activation, activation_bits);
    failures += parent.verify_preserved("attn NVFP4 panels parent");
    return failures;
}

int run_nvfp4_packed_column0(DevicePackedWeight& parent, std::int32_t tokens) {
    constexpr std::int32_t kHidden = 5120;
    constexpr std::int32_t kQRows  = 6144;
    constexpr std::int32_t kKvRows = 1024;
    const std::vector<float> activation = make_bf16_activation(kHidden, tokens, 337U + tokens);
    const std::vector<std::uint16_t> activation_bits = bf16_bits(activation);
    DeviceBuffer device_activation                   = to_device(activation_bits);

    GuardedBf16Tensor packed_q(kQRows, tokens);
    GuardedBf16Tensor packed_gate(kQRows, tokens);
    GuardedBf16Tensor packed_key(kKvRows, tokens);
    GuardedBf16Tensor packed_value(kKvRows, tokens);
    GuardedBf16Tensor decode_q(kQRows, 1);
    GuardedBf16Tensor decode_gate(kQRows, 1);
    GuardedBf16Tensor decode_key(kKvRows, 1);
    GuardedBf16Tensor decode_value(kKvRows, 1);

    Tensor packed_x(device_activation.p, DType::BF16, {kHidden, tokens});
    Tensor decode_x(device_activation.p, DType::BF16, {kHidden, 1});
    Tensor pq = packed_q.tensor();
    Tensor pg = packed_gate.tensor();
    Tensor pk = packed_key.tensor();
    Tensor pv = packed_value.tensor();
    Tensor dq = decode_q.tensor();
    Tensor dg = decode_gate.tensor();
    Tensor dk = decode_key.tensor();
    Tensor dv = decode_value.tensor();

    const std::size_t packed_capacity = ops::attn_input_proj_workspace_capacity_bytes(
        QType::NVFP4, 14336, kHidden, ops::LinearPolicy::AllowA4, tokens, tokens);
    const std::size_t decode_capacity = ops::attn_input_proj_workspace_capacity_bytes(
        QType::NVFP4, 14336, kHidden, ops::LinearPolicy::A16Only, 1, 1);
    DeviceArena packed_workspace(std::max<std::size_t>(packed_capacity, 256));
    DeviceArena decode_workspace(std::max<std::size_t>(decode_capacity, 256));
    ops::attn_input_proj(packed_x, parent.view(), pq, pg, pk, pv, ops::LinearPolicy::AllowA4,
                         packed_workspace, nullptr);
    ops::attn_input_proj(decode_x, parent.view(), dq, dg, dk, dv, ops::LinearPolicy::A16Only,
                         decode_workspace, nullptr);
    cuda_synchronize();

    const std::string suffix =
        std::string(" NVFP4 A4 packed-col0 T=") + std::to_string(tokens);
    int failures = 0;
    failures += compare_column0("attn q" + suffix, packed_q, decode_q, kQRows,
                                kAttnInputProjA16Tolerance);
    failures += compare_column0("attn gate" + suffix, packed_gate, decode_gate, kQRows,
                                kAttnInputProjA16Tolerance);
    failures += compare_column0("attn k" + suffix, packed_key, decode_key, kKvRows,
                                kAttnInputProjA16Tolerance);
    failures += compare_column0("attn value" + suffix, packed_value, decode_value, kKvRows,
                                kAttnInputProjA16Tolerance);
    return failures;
}

int run_nvfp4_target() {
    constexpr std::int32_t kHidden     = 5120;
    constexpr std::int32_t kParentRows = 14336;
    quantized_weight::PatternedWeightOptions options;
    options.weight_scale_divisor = 0.125F;
    options.input_scale_divisor  = 3.5F;
    DevicePackedWeight parent(
        quantized_weight::make_patterned_weight(QType::NVFP4, kParentRows, kHidden, 331U, options));

    int failures = 0;
    for (const std::int32_t tokens : {1, 2, 10, 15, 16, 20, 32, 33}) {
        failures += run_nvfp4_target_case(parent, tokens);
    }
    for (const std::int32_t panel : {4, 5, 6}) {
        for (std::int32_t batch = 2; batch <= 4; ++batch) {
            failures += run_nvfp4_panels(parent, panel * batch, panel);
        }
    }
    for (const std::int32_t tokens : {1, 4, 15, 36, 1024}) {
        failures += run_nvfp4_target_case(parent, tokens, ops::LinearPolicy::AllowA4);
    }
    for (const std::int32_t tokens : {2}) {
        failures += run_nvfp4_packed_column0(parent, tokens);
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
    ops::attn_input_proj(x, parent.view(), q, g, k, v, nullptr);
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
    for (const std::int32_t tokens : {1, 2, 64, 65, 129}) {
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
    ops::attn_input_proj(x, parent.view(), q, k, v, nullptr);
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

constexpr ReductionCriterion kFp8AttnInputProjA16Tolerance{1.0 / 256.0, 1.0 / 256.0, 2.0 / 256.0};
constexpr ReductionCriterion kAttnInputProjA8Tolerance{0.04, 1.0 / 256.0, 0.06};
int run_target_projection_case(DevicePackedWeight& parent, DevicePackedWeight* gate_value,
                               int tokens, ops::LinearPolicy policy, bool replay = false) {
    constexpr int hidden = 5120, qrows = 6144, kvrows = 1024;
    const bool dual      = gate_value != nullptr;
    auto activation      = make_bf16_activation(hidden, tokens, 101U + tokens);
    auto activation_bits = bf16_bits(activation);
    DeviceBuffer input   = to_device(activation_bits);
    GuardedBf16Tensor query(qrows, tokens), gate(qrows, tokens), key(kvrows, tokens),
        value(kvrows, tokens);
    Tensor x(input.p, DType::BF16, {hidden, tokens}), q = query.tensor(), g = gate.tensor(),
                                                      k = key.tensor(), v = value.tensor();
    const auto capacity =
        dual ? 0
             : ops::attn_input_proj_workspace_capacity_bytes(QType::FP8_E4M3FN_ROW_BF16S, 14336,
                                                             hidden, policy, tokens, tokens);
    GuardedDeviceBuffer scratch(std::max<std::size_t>(capacity, 1));
    DeviceArena workspace(DeviceSpan{scratch.data(), std::max<std::size_t>(capacity, 1)});
    DeviceContext device;
    const auto launch = [&] {
        if (dual)
            ops::attn_input_proj(x, parent.view(), gate_value->view(), q, g, k, v, device.stream);
        else if (policy == ops::LinearPolicy::A16Only && (tokens % 2))
            ops::attn_input_proj(x, parent.view(), q, g, k, v, device.stream);
        else
            ops::attn_input_proj(x, parent.view(), q, g, k, v, policy, workspace, device.stream);
    };
    DecodeGraphDefinition definition;
    DecodeGraphExecutable graph;
    cuda_synchronize();
    if (replay) {
        definition.capture(device.stream, launch);
        graph.instantiate(definition);
    }
    int failures = 0;
    for (int phase = 0; phase < (replay ? 2 : 1); ++phase) {
        if (phase) {
            for (auto& v : activation) v = -v;
            activation_bits = bf16_bits(activation);
            input.copy_from_host(activation_bits.data(), input.bytes);
        }
        scratch.fill(phase ? 0xa5 : 0x5a);
        cuda_synchronize();
        for (Tensor* out : {&q, &g, &k, &v})
            CUDA_CHECK(cudaMemsetAsync(out->data, 0xff, std::size_t(out->ne[0]) * tokens * 2,
                                       device.stream));
        if (replay)
            graph.launch(device.stream);
        else
            launch();
        cuda_synchronize(device.stream);
        const bool a8            = policy == ops::LinearPolicy::AllowA8;
        const auto criterion     = dual ? kAttnInputProjA16Tolerance
                                   : a8 ? kAttnInputProjA8Tolerance
                                        : kFp8AttnInputProjA16Tolerance;
        const int sample_count   = (a8 || replay) ? 31 : 7;
        const std::string suffix = std::string(dual ? " Q4/Q5" : " FP8") +
                                   (a8 ? " allow-a8" : " a16") + " T=" + std::to_string(tokens) +
                                   " phase=" + std::to_string(phase);
        failures += verify_output("attn q" + suffix, query, parent.host, 0, qrows, activation,
                                  hidden, tokens, criterion, sample_count);
        failures += verify_output("attn k" + suffix, key, parent.host, qrows, kvrows, activation,
                                  hidden, tokens, criterion, sample_count);
        failures += verify_output("attn gate" + suffix, gate, dual ? gate_value->host : parent.host,
                                  dual ? 0 : 7168, qrows, activation, hidden, tokens, criterion,
                                  sample_count);
        failures += verify_output("attn value" + suffix, value,
                                  dual ? gate_value->host : parent.host, dual ? 6144 : 13312,
                                  kvrows, activation, hidden, tokens, criterion, sample_count);
        failures += verify_preserved("attn input" + suffix, input, activation_bits);
        failures += scratch.verify_guards(suffix);
        if (workspace.used() != 0 || workspace.peak_used() > capacity) {
            std::cerr << "attention projection workspace exceeds query or leaks a scope\n";
            ++failures;
        }
    }
    failures += parent.verify_preserved("attn parent");
    if (dual) failures += gate_value->verify_preserved("attn gate/value");
    return failures;
}

int run_fp8_target() {
    constexpr std::int32_t kHidden = 5120;
    constexpr std::int32_t kRows   = 14336;
    DevicePackedWeight parent(
        quantized_weight::make_patterned_weight(QType::FP8_E4M3FN_ROW_BF16S, kRows, kHidden, 349U));

    int failures = 0;
    for (auto policy : {ops::LinearPolicy::A16Only, ops::LinearPolicy::AllowA8}) {
        std::size_t peak = 0;
        for (int t = 1; t <= 128; ++t) {
            peak = std::max(peak, ops::attn_input_proj_workspace_capacity_bytes(
                                      QType::FP8_E4M3FN_ROW_BF16S, kRows, kHidden, policy, t, t));
            failures += run_target_projection_case(parent, nullptr, t, policy);
        }
        const auto interval = ops::attn_input_proj_workspace_capacity_bytes(
            QType::FP8_E4M3FN_ROW_BF16S, kRows, kHidden, policy, 1, 128);
        if (interval != peak || (policy == ops::LinearPolicy::A16Only && interval != 0)) {
            std::cerr << "FP8 attention projection workspace interval mismatch\n";
            ++failures;
        }
        for (int t : {129, 144, 145, 160, 161, 192, 193, 256, 257, 1024})
            failures += run_target_projection_case(parent, nullptr, t, policy);
        for (int t : {1,  4,  5,  6,  8,  9,  16,  24,  25,  32,  33,  34,
                      64, 65, 80, 81, 96, 97, 128, 129, 144, 145, 160, 161})
            failures += run_target_projection_case(parent, nullptr, t, policy, true);
    }
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cerr << "FAIL: no usable CUDA device\n";
        return 1;
    }

    int failures = 0;
    failures += run_fp8_target();
    failures += run_q4_q5();
    failures += run_bf16_target();
    failures += run_nvfp4_target();
    failures += run_w8_target();
    failures += run_w8_companion();
    std::cout << (failures == 0 ? "OK" : "FAIL") << " attn_input_proj\n";
    return failures == 0 ? 0 : 1;
}
