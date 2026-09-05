#include "ninfer/ops/gated_delta_net.h"
#include "ops/gdn_ref.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr int kStateDim = 128;

constexpr ReductionCriterion gated_delta_net_output_bf16_criterion() {
    return {/*relative_l2=*/4.1e-3, /*gross_absolute=*/5.0e-6,
            /*gross_relative_to_max_reference=*/5.5e-3};
}

constexpr ReductionCriterion gated_delta_net_state_fp32_criterion() {
    return {/*relative_l2=*/2.7e-3, /*gross_absolute=*/1.0e-5,
            /*gross_relative_to_max_reference=*/3.9e-3};
}

struct Case {
    const char* name;
    int qk_heads;
    int value_heads;
    int tokens;
    bool normalize_qk;
    bool near_zero_qk = false;
};

void fill_uniform(std::vector<float>& values, std::mt19937& generator, float low, float high) {
    std::uniform_real_distribution<float> distribution(low, high);
    for (float& value : values) { value = distribution(generator); }
}

void normalize_rows(std::vector<float>& values, int width) {
    const std::size_t rows = values.size() / static_cast<std::size_t>(width);
    for (std::size_t row = 0; row < rows; ++row) {
        float* base  = values.data() + row * static_cast<std::size_t>(width);
        double sumsq = 0.0;
        for (int d = 0; d < width; ++d) {
            const double value = static_cast<double>(base[d]);
            sumsq += value * value;
        }
        const double inv = 1.0 / std::sqrt(sumsq);
        for (int d = 0; d < width; ++d) {
            base[d] = static_cast<float>(static_cast<double>(base[d]) * inv);
        }
    }
}

gdn_ref::Inputs make_inputs(const Case& test_case, std::uint32_t seed) {
    gdn_ref::Inputs in;
    in.head_dim    = kStateDim;
    in.qk_heads    = test_case.qk_heads;
    in.value_heads = test_case.value_heads;
    in.tokens      = test_case.tokens;

    const std::size_t qk_size =
        static_cast<std::size_t>(kStateDim * test_case.qk_heads * test_case.tokens);
    const std::size_t value_size =
        static_cast<std::size_t>(kStateDim * test_case.value_heads * test_case.tokens);
    const std::size_t state_size =
        static_cast<std::size_t>(kStateDim * kStateDim * test_case.value_heads);
    in.q.resize(qk_size);
    in.k.resize(qk_size);
    in.v.resize(value_size);
    in.g.resize(static_cast<std::size_t>(test_case.value_heads * test_case.tokens));
    in.beta.resize(static_cast<std::size_t>(test_case.value_heads * test_case.tokens));
    in.state.resize(state_size);

    std::mt19937 generator(seed);
    fill_uniform(in.q, generator, -1.0f, 1.0f);
    fill_uniform(in.k, generator, -1.0f, 1.0f);
    fill_uniform(in.v, generator, -0.5f, 0.5f);
    fill_uniform(in.g, generator, -0.10f, -0.005f);
    fill_uniform(in.beta, generator, 0.05f, 0.95f);
    fill_uniform(in.state, generator, -0.02f, 0.02f);

    if (test_case.near_zero_qk) {
        for (float& value : in.q) { value *= 1.0e-4f; }
        for (float& value : in.k) { value *= 1.0e-4f; }
    } else if (!test_case.normalize_qk) {
        // Raw-Q/K mode still receives a stable, entirely valid public input. This host-side
        // generation choice is not part of the oracle.
        normalize_rows(in.q, kStateDim);
        normalize_rows(in.k, kStateDim);
    }

    round_to_bf16(in.q);
    round_to_bf16(in.k);
    round_to_bf16(in.v);
    return in;
}

std::vector<std::uint16_t> bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { bits[i] = f32_to_bf16(values[i]); }
    return bits;
}

std::vector<double> doubles(const std::vector<float>& values) {
    return std::vector<double>(values.begin(), values.end());
}

template <typename T>
int verify_exact(const std::string& label, const std::vector<T>& got,
                 const std::vector<T>& expected) {
    return ninfer::test::verify_exact(label.c_str(), got, expected);
}

int verify_recurrence(const std::string& label, std::span<const double> got,
                      std::span<const double> expected, const ReductionCriterion& criterion) {
    return verify_reduction(label.c_str(), got, expected, criterion);
}

std::vector<double> bf16_doubles(const std::vector<std::uint16_t>& bits) {
    std::vector<double> values(bits.size());
    for (std::size_t index = 0; index < bits.size(); ++index) {
        values[index] = static_cast<double>(bf16_to_f32(bits[index]));
    }
    return values;
}

std::vector<double> read_f32(const void* device, std::size_t count) {
    return doubles(from_device<float>(device, count));
}

int verify_common_inputs_unchanged(const std::string& label, const gdn_ref::Inputs& in,
                                   const DeviceBuffer& q, const DeviceBuffer& k,
                                   const DeviceBuffer& v, const DeviceBuffer& g,
                                   const DeviceBuffer& beta) {
    int failures = 0;
    failures += verify_exact(label + " q unchanged", from_device<std::uint16_t>(q, in.q.size()),
                             bf16_bits(in.q));
    failures += verify_exact(label + " k unchanged", from_device<std::uint16_t>(k, in.k.size()),
                             bf16_bits(in.k));
    failures += verify_exact(label + " v unchanged", from_device<std::uint16_t>(v, in.v.size()),
                             bf16_bits(in.v));
    failures += verify_exact(label + " g unchanged", from_device<float>(g, in.g.size()), in.g);
    failures +=
        verify_exact(label + " beta unchanged", from_device<float>(beta, in.beta.size()), in.beta);
    return failures;
}

struct DeviceInputs {
    explicit DeviceInputs(const gdn_ref::Inputs& in)
        : q(to_device_bf16(in.q)), k(to_device_bf16(in.k)), v(to_device_bf16(in.v)),
          g(to_device_f32(in.g)), beta(to_device_f32(in.beta)) {}

    DeviceBuffer q;
    DeviceBuffer k;
    DeviceBuffer v;
    DeviceBuffer g;
    DeviceBuffer beta;
};

struct RouteEvidence {
    std::string name;
    std::vector<std::uint16_t> output;
    std::vector<float> state;
    int failures = 0;
};

RouteEvidence run_partition_route(const Case& test_case, const gdn_ref::Inputs& in,
                                  const DeviceInputs& device,
                                  const std::vector<std::int32_t>& chunks,
                                  std::string name) {
    std::int32_t total = 0;
    std::int32_t maximum_chunk = 0;
    for (const std::int32_t chunk : chunks) {
        total += chunk;
        maximum_chunk = std::max(maximum_chunk, chunk);
    }
    RouteEvidence evidence{std::move(name)};
    if (total != test_case.tokens || maximum_chunk <= 0) {
        std::cerr << evidence.name << ": invalid route partition\n";
        evidence.failures = 1;
        return evidence;
    }

    GuardedDeviceBuffer state(in.state.size() * sizeof(float));
    GuardedDeviceBuffer output(in.v.size() * sizeof(std::uint16_t));
    state.copy_from_host(in.state.data(), state.bytes());
    output.fill(0xff);
    Tensor q(device.q.p, DType::BF16,
             {kStateDim, test_case.qk_heads, test_case.tokens});
    Tensor k(device.k.p, DType::BF16,
             {kStateDim, test_case.qk_heads, test_case.tokens});
    Tensor v(device.v.p, DType::BF16,
             {kStateDim, test_case.value_heads, test_case.tokens});
    Tensor g(device.g.p, DType::FP32, {test_case.value_heads, test_case.tokens});
    Tensor beta(device.beta.p, DType::FP32, {test_case.value_heads, test_case.tokens});
    Tensor state_tensor(state.data(), DType::FP32,
                        {kStateDim, kStateDim, test_case.value_heads});
    Tensor output_tensor(output.data(), DType::BF16,
                         {kStateDim, test_case.value_heads, test_case.tokens});
    const std::size_t workspace_bytes = ops::gated_delta_net_workspace_capacity_bytes(
        test_case.qk_heads, test_case.value_heads, test_case.normalize_qk, 1, maximum_chunk);
    WorkspaceArena workspace(std::max<std::size_t>(workspace_bytes, 256));
    const float scale = 1.0f / std::sqrt(static_cast<float>(kStateDim));

    std::int32_t offset = 0;
    for (const std::int32_t chunk : chunks) {
        Tensor q_chunk = q.slice(2, offset, chunk);
        Tensor k_chunk = k.slice(2, offset, chunk);
        Tensor v_chunk = v.slice(2, offset, chunk);
        Tensor g_chunk = g.slice(1, offset, chunk);
        Tensor beta_chunk = beta.slice(1, offset, chunk);
        Tensor output_chunk = output_tensor.slice(2, offset, chunk);
        ops::gated_delta_net(q_chunk, k_chunk, v_chunk, g_chunk, beta_chunk, scale,
                             test_case.normalize_qk, workspace, state_tensor, output_chunk,
                             nullptr);
        offset += chunk;
    }
    cuda_synchronize();

    evidence.output = from_device<std::uint16_t>(output.data(), in.v.size());
    evidence.state = from_device<float>(state.data(), in.state.size());
    evidence.failures += state.verify_guards((evidence.name + " state").c_str());
    evidence.failures += output.verify_guards((evidence.name + " output").c_str());
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << evidence.name << ": workspace query/execution high-water mismatch\n";
        ++evidence.failures;
    }
    return evidence;
}

int qwen4_maximum_schedule_case() {
    constexpr std::int32_t kTokens = 4096;
    constexpr std::int32_t kHeads = 48;
    const Case test_case{"Qwen4 exact-geometry maximum prefill", kHeads, kHeads, kTokens, true};
    const gdn_ref::Inputs in = make_inputs(test_case, 16496u);
    const float scale = 1.0f / std::sqrt(static_cast<float>(kStateDim));
    const gdn_ref::Result reference = gdn_ref::evaluate(in, static_cast<double>(scale), true);
    const DeviceInputs device(in);

    std::vector<std::int32_t> scalar_chunks(kTokens, 1);
    std::vector<std::int32_t> aligned_chunks(kTokens / 64, 64);
    std::vector<RouteEvidence> routes;
    routes.reserve(4);
    routes.push_back(run_partition_route(test_case, in, device, scalar_chunks, "scalar T=1"));
    routes.push_back(run_partition_route(test_case, in, device, {kTokens}, "one-shot T=4096"));
    routes.push_back(
        run_partition_route(test_case, in, device, aligned_chunks, "aligned 64x64"));
    routes.push_back(run_partition_route(test_case, in, device, {64, 1, 1986, 1, 2044},
                                         "Program 64+1+1986+1+2044"));

    constexpr std::array<std::int32_t, 9> kPrefixWidths = {
        1, 63, 64, 65, 127, 128, 129, 2051, 2052};
    constexpr std::array<std::int32_t, 10> kDiagnosticTokens = {
        0, 62, 63, 64, 126, 127, 128, 2050, 2051, kTokens - 1};
    const std::size_t values_per_token = static_cast<std::size_t>(kStateDim * kHeads);
    int failures = 0;
    for (const RouteEvidence& route : routes) {
        failures += route.failures;
        const std::vector<double> output = bf16_doubles(route.output);
        const std::vector<double> state = doubles(route.state);
        const std::string label = std::string(test_case.name) + " " + route.name;
        failures += verify_recurrence(label + " output oracle", output, reference.out,
                                      gated_delta_net_output_bf16_criterion());
        failures += verify_recurrence(label + " state oracle", state, reference.final_state,
                                      gated_delta_net_state_fp32_criterion());
        for (const std::int32_t width : kPrefixWidths) {
            const std::size_t count = static_cast<std::size_t>(width) * values_per_token;
            failures += verify_recurrence(
                label + " output prefix T=" + std::to_string(width),
                std::span<const double>(output.data(), count),
                std::span<const double>(reference.out.data(), count),
                gated_delta_net_output_bf16_criterion());
        }
        if (error_stats_enabled()) {
            double peak_relative_l2 = -1.0;
            double peak_absolute = -1.0;
            std::int32_t peak_relative_token = -1;
            std::int32_t peak_absolute_token = -1;
            for (std::int32_t token = 0; token < kTokens; ++token) {
                const std::size_t begin = static_cast<std::size_t>(token) * values_per_token;
                const ReductionStats stats = compute_reduction_stats(
                    output.data() + begin, reference.out.data() + begin,
                    static_cast<std::int64_t>(values_per_token));
                if (stats.relative_l2 > peak_relative_l2) {
                    peak_relative_l2 = stats.relative_l2;
                    peak_relative_token = token;
                }
                if (stats.maximum_absolute_error > peak_absolute) {
                    peak_absolute = stats.maximum_absolute_error;
                    peak_absolute_token = token;
                }
            }
            std::cout << "GDN_TOKEN_ERROR_SUMMARY route=" << route.name
                      << " peak_relative_token=" << peak_relative_token
                      << " peak_relative_l2=" << peak_relative_l2
                      << " peak_absolute_token=" << peak_absolute_token
                      << " peak_absolute=" << peak_absolute << '\n';
            for (const std::int32_t token : kDiagnosticTokens) {
                const std::size_t begin = static_cast<std::size_t>(token) * values_per_token;
                const ReductionStats stats = compute_reduction_stats(
                    output.data() + begin, reference.out.data() + begin,
                    static_cast<std::int64_t>(values_per_token));
                report_reduction_stats(label + " output token t=" + std::to_string(token),
                                       static_cast<std::int64_t>(values_per_token), stats,
                                       gated_delta_net_output_bf16_criterion());
            }
        }
    }

    failures += verify_common_inputs_unchanged(std::string(test_case.name), in, device.q, device.k,
                                               device.v, device.g, device.beta);
    if (error_stats_enabled()) {
        const auto report_pair = [&](std::size_t reference_index, std::size_t actual_index) {
            const RouteEvidence& route_reference = routes[reference_index];
            const RouteEvidence& route_actual = routes[actual_index];
            const auto mismatch = std::mismatch(
                route_reference.output.begin(), route_reference.output.end(),
                route_actual.output.begin(), route_actual.output.end());
            const bool state_bit_identical =
                route_reference.state.size() == route_actual.state.size() &&
                std::memcmp(route_reference.state.data(), route_actual.state.data(),
                            route_reference.state.size() * sizeof(float)) == 0;
            if (mismatch.first == route_reference.output.end()) {
                std::cout << "GDN_SCHEDULE_DIAGNOSTIC reference=" << route_reference.name
                          << " actual=" << route_actual.name
                          << " output_bit_identical=true state_bit_identical="
                          << (state_bit_identical ? "true" : "false") << '\n';
                return;
            }
            const std::size_t index =
                static_cast<std::size_t>(mismatch.first - route_reference.output.begin());
            const std::size_t token = index / values_per_token;
            const std::size_t within = index - token * values_per_token;
            std::cout << "GDN_SCHEDULE_DIAGNOSTIC reference=" << route_reference.name
                      << " actual=" << route_actual.name
                      << " output_bit_identical=false state_bit_identical="
                      << (state_bit_identical ? "true" : "false") << " first_token=" << token
                      << " first_head=" << within / kStateDim
                      << " first_dimension=" << within % kStateDim << '\n';
        };
        report_pair(0, 1);
        report_pair(1, 2);
        report_pair(1, 3);
    }
    return failures;
}

int inplace_case(const Case& test_case, std::uint32_t seed) {
    const gdn_ref::Inputs in = make_inputs(test_case, seed);
    const float scale        = 1.0f / std::sqrt(static_cast<float>(kStateDim));
    const gdn_ref::Result ref =
        gdn_ref::evaluate(in, static_cast<double>(scale), test_case.normalize_qk);
    DeviceInputs device(in);
    GuardedDeviceBuffer state(in.state.size() * sizeof(float));
    GuardedDeviceBuffer out(in.v.size() * sizeof(std::uint16_t));
    state.copy_from_host(in.state.data(), state.bytes());
    out.fill(0xff);

    Tensor q(device.q.p, DType::BF16, {kStateDim, test_case.qk_heads, test_case.tokens});
    Tensor k(device.k.p, DType::BF16, {kStateDim, test_case.qk_heads, test_case.tokens});
    Tensor v(device.v.p, DType::BF16, {kStateDim, test_case.value_heads, test_case.tokens});
    Tensor g(device.g.p, DType::FP32, {test_case.value_heads, test_case.tokens});
    Tensor beta(device.beta.p, DType::FP32, {test_case.value_heads, test_case.tokens});
    Tensor state_tensor(state.data(), DType::FP32, {kStateDim, kStateDim, test_case.value_heads});
    Tensor out_tensor(out.data(), DType::BF16,
                      {kStateDim, test_case.value_heads, test_case.tokens});
    const std::size_t workspace_bytes = ops::gated_delta_net_workspace_capacity_bytes(
        test_case.qk_heads, test_case.value_heads, test_case.normalize_qk, test_case.tokens,
        test_case.tokens);
    WorkspaceArena workspace(std::max<std::size_t>(workspace_bytes, 256));

    ops::gated_delta_net(q, k, v, g, beta, scale, test_case.normalize_qk, workspace, state_tensor,
                         out_tensor, nullptr);
    cuda_synchronize();

    const std::string label = std::string(test_case.name) + " inplace";
    int failures            = 0;
    failures += verify_recurrence(label + " out", from_device_bf16(out.data(), in.v.size()),
                                  ref.out, gated_delta_net_output_bf16_criterion());
    failures += verify_recurrence(label + " state", read_f32(state.data(), in.state.size()),
                                  ref.final_state, gated_delta_net_state_fp32_criterion());
    failures += state.verify_guards((label + " state").c_str());
    failures += out.verify_guards((label + " out").c_str());
    failures += verify_common_inputs_unchanged(label, in, device.q, device.k, device.v, device.g,
                                               device.beta);
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    return failures;
}

int distinct_state_case(const Case& test_case, std::uint32_t seed) {
    const gdn_ref::Inputs in = make_inputs(test_case, seed);
    const float scale        = 1.0f / std::sqrt(static_cast<float>(kStateDim));
    const gdn_ref::Result ref =
        gdn_ref::evaluate(in, static_cast<double>(scale), test_case.normalize_qk);
    DeviceInputs device(in);
    GuardedDeviceBuffer state_in(in.state.size() * sizeof(float));
    GuardedDeviceBuffer state_out(in.state.size() * sizeof(float));
    GuardedDeviceBuffer out(in.v.size() * sizeof(std::uint16_t));
    state_in.copy_from_host(in.state.data(), state_in.bytes());
    state_out.fill(0xff);
    out.fill(0xff);

    Tensor q(device.q.p, DType::BF16, {kStateDim, test_case.qk_heads, test_case.tokens});
    Tensor k(device.k.p, DType::BF16, {kStateDim, test_case.qk_heads, test_case.tokens});
    Tensor v(device.v.p, DType::BF16, {kStateDim, test_case.value_heads, test_case.tokens});
    Tensor g(device.g.p, DType::FP32, {test_case.value_heads, test_case.tokens});
    Tensor beta(device.beta.p, DType::FP32, {test_case.value_heads, test_case.tokens});
    Tensor state_in_tensor(state_in.data(), DType::FP32,
                           {kStateDim, kStateDim, test_case.value_heads});
    Tensor state_out_tensor(state_out.data(), DType::FP32,
                            {kStateDim, kStateDim, test_case.value_heads});
    Tensor out_tensor(out.data(), DType::BF16,
                      {kStateDim, test_case.value_heads, test_case.tokens});
    const std::size_t workspace_bytes = ops::gated_delta_net_workspace_capacity_bytes(
        test_case.qk_heads, test_case.value_heads, test_case.normalize_qk, test_case.tokens,
        test_case.tokens);
    WorkspaceArena workspace(std::max<std::size_t>(workspace_bytes, 256));

    ops::gated_delta_net(q, k, v, g, beta, scale, test_case.normalize_qk, workspace,
                         state_in_tensor, state_out_tensor, out_tensor, nullptr);
    cuda_synchronize();

    const std::string label = std::string(test_case.name) + " distinct-state";
    int failures            = 0;
    failures += verify_recurrence(label + " out", from_device_bf16(out.data(), in.v.size()),
                                  ref.out, gated_delta_net_output_bf16_criterion());
    failures += verify_recurrence(label + " state", read_f32(state_out.data(), in.state.size()),
                                  ref.final_state, gated_delta_net_state_fp32_criterion());
    failures += verify_exact(label + " state-in unchanged",
                             from_device<float>(state_in.data(), in.state.size()), in.state);
    failures += state_in.verify_guards((label + " state-in").c_str());
    failures += state_out.verify_guards((label + " state-out").c_str());
    failures += out.verify_guards((label + " out").c_str());
    failures += verify_common_inputs_unchanged(label, in, device.q, device.k, device.v, device.g,
                                               device.beta);
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << label << ": workspace query/execution high-water mismatch\n";
        ++failures;
    }
    return failures;
}

int partition_case(const Case& test_case, std::initializer_list<std::int32_t> chunks,
                   std::uint32_t seed) {
    std::int32_t partition_tokens = 0;
    std::int32_t maximum_chunk    = 0;
    for (const std::int32_t chunk : chunks) {
        partition_tokens += chunk;
        maximum_chunk = std::max(maximum_chunk, chunk);
    }
    if (partition_tokens != test_case.tokens || maximum_chunk <= 0) {
        std::cerr << "invalid gated_delta_net partition\n";
        return 1;
    }

    const gdn_ref::Inputs in = make_inputs(test_case, seed);
    const float scale        = 1.0f / std::sqrt(static_cast<float>(kStateDim));
    const gdn_ref::Result ref =
        gdn_ref::evaluate(in, static_cast<double>(scale), test_case.normalize_qk);
    DeviceInputs device(in);
    GuardedDeviceBuffer full_state(in.state.size() * sizeof(float));
    GuardedDeviceBuffer partition_state(in.state.size() * sizeof(float));
    GuardedDeviceBuffer full_out(in.v.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer partition_out(in.v.size() * sizeof(std::uint16_t));
    full_state.copy_from_host(in.state.data(), full_state.bytes());
    partition_state.copy_from_host(in.state.data(), partition_state.bytes());
    full_out.fill(0xff);
    partition_out.fill(0xff);

    Tensor q(device.q.p, DType::BF16, {kStateDim, test_case.qk_heads, test_case.tokens});
    Tensor k(device.k.p, DType::BF16, {kStateDim, test_case.qk_heads, test_case.tokens});
    Tensor v(device.v.p, DType::BF16, {kStateDim, test_case.value_heads, test_case.tokens});
    Tensor g(device.g.p, DType::FP32, {test_case.value_heads, test_case.tokens});
    Tensor beta(device.beta.p, DType::FP32, {test_case.value_heads, test_case.tokens});
    Tensor full_state_tensor(full_state.data(), DType::FP32,
                             {kStateDim, kStateDim, test_case.value_heads});
    Tensor partition_state_tensor(partition_state.data(), DType::FP32,
                                  {kStateDim, kStateDim, test_case.value_heads});
    Tensor full_out_tensor(full_out.data(), DType::BF16,
                           {kStateDim, test_case.value_heads, test_case.tokens});
    Tensor partition_out_tensor(partition_out.data(), DType::BF16,
                                {kStateDim, test_case.value_heads, test_case.tokens});
    const std::size_t full_workspace_bytes = ops::gated_delta_net_workspace_capacity_bytes(
        test_case.qk_heads, test_case.value_heads, test_case.normalize_qk, test_case.tokens,
        test_case.tokens);
    const std::size_t partition_workspace_bytes = ops::gated_delta_net_workspace_capacity_bytes(
        test_case.qk_heads, test_case.value_heads, test_case.normalize_qk, 1, maximum_chunk);
    WorkspaceArena full_workspace(std::max<std::size_t>(full_workspace_bytes, 256));
    WorkspaceArena partition_workspace(std::max<std::size_t>(partition_workspace_bytes, 256));

    ops::gated_delta_net(q, k, v, g, beta, scale, test_case.normalize_qk, full_workspace,
                         full_state_tensor, full_out_tensor, nullptr);
    std::int32_t offset = 0;
    for (const std::int32_t chunk : chunks) {
        Tensor q_chunk   = q.slice(2, offset, chunk);
        Tensor k_chunk   = k.slice(2, offset, chunk);
        Tensor v_chunk   = v.slice(2, offset, chunk);
        Tensor g_chunk   = g.slice(1, offset, chunk);
        Tensor b_chunk   = beta.slice(1, offset, chunk);
        Tensor out_chunk = partition_out_tensor.slice(2, offset, chunk);
        ops::gated_delta_net(q_chunk, k_chunk, v_chunk, g_chunk, b_chunk, scale,
                             test_case.normalize_qk, partition_workspace, partition_state_tensor,
                             out_chunk, nullptr);
        offset += chunk;
    }
    cuda_synchronize();

    std::string partition_name;
    for (const std::int32_t chunk : chunks) {
        if (!partition_name.empty()) partition_name += '+';
        partition_name += std::to_string(chunk);
    }
    const std::string label = std::string(test_case.name) + " partition " + partition_name;
    const std::vector<double> full_output = from_device_bf16(full_out.data(), in.v.size());
    const std::vector<double> partition_output =
        from_device_bf16(partition_out.data(), in.v.size());
    const std::vector<double> full_state_values = read_f32(full_state.data(), in.state.size());
    const std::vector<double> partition_state_values =
        read_f32(partition_state.data(), in.state.size());
    int failures = 0;
    failures += verify_recurrence(label + " full out oracle", full_output, ref.out,
                                  gated_delta_net_output_bf16_criterion());
    failures += verify_recurrence(label + " partition out oracle", partition_output, ref.out,
                                  gated_delta_net_output_bf16_criterion());
    failures += verify_recurrence(label + " full state oracle", full_state_values, ref.final_state,
                                  gated_delta_net_state_fp32_criterion());
    failures += verify_recurrence(label + " partition state oracle", partition_state_values,
                                  ref.final_state, gated_delta_net_state_fp32_criterion());
    failures += full_state.verify_guards((label + " full state").c_str());
    failures += partition_state.verify_guards((label + " partition state").c_str());
    failures += full_out.verify_guards((label + " full out").c_str());
    failures += partition_out.verify_guards((label + " partition out").c_str());
    failures += verify_common_inputs_unchanged(label, in, device.q, device.k, device.v, device.g,
                                               device.beta);
    return failures;
}

int snapshot_case(const Case& test_case, int slots, int initial_slot, int snapshot_base_slot,
                  std::uint32_t seed) {
    const gdn_ref::Inputs in = make_inputs(test_case, seed);
    const float scale        = 1.0f / std::sqrt(static_cast<float>(kStateDim));
    const gdn_ref::Result ref =
        gdn_ref::evaluate(in, static_cast<double>(scale), test_case.normalize_qk, true);
    const std::size_t state_size = in.state.size();
    std::vector<float> initial_states(state_size * static_cast<std::size_t>(slots), 17.0f);
    std::copy(in.state.begin(), in.state.end(),
              initial_states.begin() + static_cast<std::size_t>(initial_slot) * state_size);

    DeviceInputs device(in);
    GuardedDeviceBuffer states(initial_states.size() * sizeof(float));
    GuardedDeviceBuffer out(in.v.size() * sizeof(std::uint16_t));
    states.copy_from_host(initial_states.data(), states.bytes());
    out.fill(0xff);
    DeviceBuffer device_initial_slot       = to_device_i32({initial_slot});
    DeviceBuffer device_snapshot_base_slot = to_device_i32({snapshot_base_slot});

    Tensor q(device.q.p, DType::BF16, {kStateDim, test_case.qk_heads, test_case.tokens});
    Tensor k(device.k.p, DType::BF16, {kStateDim, test_case.qk_heads, test_case.tokens});
    Tensor v(device.v.p, DType::BF16, {kStateDim, test_case.value_heads, test_case.tokens});
    Tensor g(device.g.p, DType::FP32, {test_case.value_heads, test_case.tokens});
    Tensor beta(device.beta.p, DType::FP32, {test_case.value_heads, test_case.tokens});
    Tensor states_tensor(states.data(), DType::FP32,
                         {kStateDim, kStateDim, test_case.value_heads, slots});
    Tensor initial_slot_tensor(device_initial_slot.p, DType::I32, {1});
    Tensor snapshot_base_slot_tensor(device_snapshot_base_slot.p, DType::I32, {1});
    Tensor out_tensor(out.data(), DType::BF16,
                      {kStateDim, test_case.value_heads, test_case.tokens});
    ops::gated_delta_net_snapshot(q, k, v, g, beta, scale, test_case.normalize_qk, states_tensor,
                                  Tensor{}, initial_slot_tensor, snapshot_base_slot_tensor,
                                  out_tensor, nullptr);
    cuda_synchronize();

    const std::string label             = std::string(test_case.name) + " snapshot";
    const std::vector<float> got_states = from_device<float>(states.data(), initial_states.size());
    const auto got_updated_begin =
        got_states.begin() + static_cast<std::size_t>(snapshot_base_slot) * state_size;
    const auto got_updated_end =
        got_updated_begin + static_cast<std::size_t>(test_case.tokens) * state_size;
    int failures = 0;
    failures += verify_recurrence(label + " out", from_device_bf16(out.data(), in.v.size()),
                                  ref.out, gated_delta_net_output_bf16_criterion());
    failures += verify_recurrence(label + " updated state slots",
                                  doubles(std::vector<float>(got_updated_begin, got_updated_end)),
                                  ref.snapshots, gated_delta_net_state_fp32_criterion());
    const auto initial_updated_begin =
        initial_states.begin() + static_cast<std::size_t>(snapshot_base_slot) * state_size;
    const auto initial_updated_end =
        initial_updated_begin + static_cast<std::size_t>(test_case.tokens) * state_size;
    failures += verify_exact(label + " slots before destination unchanged",
                             std::vector<float>(got_states.begin(), got_updated_begin),
                             std::vector<float>(initial_states.begin(), initial_updated_begin));
    failures += verify_exact(label + " slots after destination unchanged",
                             std::vector<float>(got_updated_end, got_states.end()),
                             std::vector<float>(initial_updated_end, initial_states.end()));
    failures +=
        verify_exact(label + " initial-slot scalar unchanged",
                     from_device_i32(device_initial_slot, 1), std::vector<int>{initial_slot});
    failures += verify_exact(label + " snapshot-base scalar unchanged",
                             from_device_i32(device_snapshot_base_slot, 1),
                             std::vector<int>{snapshot_base_slot});
    failures += states.verify_guards((label + " states").c_str());
    failures += out.verify_guards((label + " out").c_str());
    failures += verify_common_inputs_unchanged(label, in, device.q, device.k, device.v, device.g,
                                               device.beta);
    return failures;
}

int batched_snapshot_case(const Case& test_case, const std::vector<int>& initial_slots,
                          const std::vector<int>& snapshot_bases,
                          const std::vector<int>& valid_columns, int slots, std::uint32_t seed) {
    const int batch   = static_cast<int>(initial_slots.size());
    const int width   = test_case.tokens;
    const bool masked = !valid_columns.empty();
    const float scale = 1.0f / std::sqrt(static_cast<float>(kStateDim));
    const std::size_t qk_row_size =
        static_cast<std::size_t>(kStateDim * test_case.qk_heads * width);
    const std::size_t value_row_size =
        static_cast<std::size_t>(kStateDim * test_case.value_heads * width);
    const std::size_t gate_row_size = static_cast<std::size_t>(test_case.value_heads * width);
    const std::size_t state_size =
        static_cast<std::size_t>(kStateDim * kStateDim * test_case.value_heads);

    gdn_ref::Inputs aggregate;
    aggregate.head_dim    = kStateDim;
    aggregate.qk_heads    = test_case.qk_heads;
    aggregate.value_heads = test_case.value_heads;
    aggregate.tokens      = static_cast<std::int64_t>(width) * batch;
    aggregate.q.reserve(qk_row_size * static_cast<std::size_t>(batch));
    aggregate.k.reserve(qk_row_size * static_cast<std::size_t>(batch));
    aggregate.v.reserve(value_row_size * static_cast<std::size_t>(batch));
    aggregate.g.reserve(gate_row_size * static_cast<std::size_t>(batch));
    aggregate.beta.reserve(gate_row_size * static_cast<std::size_t>(batch));

    std::vector<gdn_ref::Inputs> rows;
    rows.reserve(static_cast<std::size_t>(batch));
    std::vector<float> initial_states(state_size * static_cast<std::size_t>(slots), 0.125f);
    for (int row = 0; row < batch; ++row) {
        gdn_ref::Inputs input =
            make_inputs(test_case, seed + static_cast<std::uint32_t>(row) * 97U);
        aggregate.q.insert(aggregate.q.end(), input.q.begin(), input.q.end());
        aggregate.k.insert(aggregate.k.end(), input.k.begin(), input.k.end());
        aggregate.v.insert(aggregate.v.end(), input.v.begin(), input.v.end());
        aggregate.g.insert(aggregate.g.end(), input.g.begin(), input.g.end());
        aggregate.beta.insert(aggregate.beta.end(), input.beta.begin(), input.beta.end());
        std::copy(input.state.begin(), input.state.end(),
                  initial_states.begin() +
                      static_cast<std::size_t>(initial_slots[static_cast<std::size_t>(row)]) *
                          state_size);
        rows.push_back(std::move(input));
    }

    std::vector<gdn_ref::Result> references;
    references.reserve(static_cast<std::size_t>(batch));
    std::vector<double> expected_output(value_row_size * static_cast<std::size_t>(batch), 0.0);
    std::vector<bool> written_slots(static_cast<std::size_t>(slots), false);
    for (int row = 0; row < batch; ++row) {
        const int valid = masked ? valid_columns[static_cast<std::size_t>(row)] : width;
        gdn_ref::Inputs oracle_input = rows[static_cast<std::size_t>(row)];
        oracle_input.tokens          = valid;
        oracle_input.q.resize(static_cast<std::size_t>(kStateDim * test_case.qk_heads * valid));
        oracle_input.k.resize(static_cast<std::size_t>(kStateDim * test_case.qk_heads * valid));
        oracle_input.v.resize(static_cast<std::size_t>(kStateDim * test_case.value_heads * valid));
        oracle_input.g.resize(static_cast<std::size_t>(test_case.value_heads * valid));
        oracle_input.beta.resize(static_cast<std::size_t>(test_case.value_heads * valid));
        gdn_ref::Result reference = gdn_ref::evaluate(oracle_input, static_cast<double>(scale),
                                                      test_case.normalize_qk, true);
        std::copy(reference.out.begin(), reference.out.end(),
                  expected_output.begin() + static_cast<std::size_t>(row) * value_row_size);
        for (int column = 0; column < valid; ++column) {
            written_slots[static_cast<std::size_t>(snapshot_bases[static_cast<std::size_t>(row)] +
                                                   column)] = true;
        }
        references.push_back(std::move(reference));
    }

    DeviceInputs device(aggregate);
    GuardedDeviceBuffer states(initial_states.size() * sizeof(float));
    GuardedDeviceBuffer out(aggregate.v.size() * sizeof(std::uint16_t));
    states.copy_from_host(initial_states.data(), states.bytes());
    out.fill(0xff);
    DeviceBuffer device_initial_slots  = to_device(initial_slots);
    DeviceBuffer device_snapshot_bases = to_device(snapshot_bases);
    DeviceBuffer device_valid_columns;
    if (masked) { device_valid_columns = to_device(valid_columns); }

    Tensor q(device.q.p, DType::BF16, {kStateDim, test_case.qk_heads, width, batch});
    Tensor k(device.k.p, DType::BF16, {kStateDim, test_case.qk_heads, width, batch});
    Tensor v(device.v.p, DType::BF16, {kStateDim, test_case.value_heads, width, batch});
    Tensor g(device.g.p, DType::FP32, {test_case.value_heads, width, batch});
    Tensor beta(device.beta.p, DType::FP32, {test_case.value_heads, width, batch});
    Tensor states_tensor(states.data(), DType::FP32,
                         {kStateDim, kStateDim, test_case.value_heads, slots});
    Tensor valid_tensor;
    if (masked) { valid_tensor = Tensor(device_valid_columns.p, DType::I32, {batch}); }
    Tensor initial_tensor(device_initial_slots.p, DType::I32, {batch});
    Tensor bases_tensor(device_snapshot_bases.p, DType::I32, {batch});
    Tensor out_tensor(out.data(), DType::BF16, {kStateDim, test_case.value_heads, width, batch});
    ops::gated_delta_net_snapshot(q, k, v, g, beta, scale, test_case.normalize_qk, states_tensor,
                                  valid_tensor, initial_tensor, bases_tensor, out_tensor, nullptr);
    cuda_synchronize();

    const std::string label = std::string(test_case.name) +
                              " batched snapshot B=" + std::to_string(batch) +
                              (masked ? " masked" : " dense");
    int failures                         = 0;
    const std::vector<double> got_output = from_device_bf16(out.data(), aggregate.v.size());
    failures += verify_recurrence(label + " out", got_output, expected_output,
                                  gated_delta_net_output_bf16_criterion());
    if (masked) {
        const std::vector<std::uint16_t> output_bits =
            from_device<std::uint16_t>(out.data(), aggregate.v.size());
        for (int row = 0; row < batch; ++row) {
            for (int column = valid_columns[static_cast<std::size_t>(row)]; column < width;
                 ++column) {
                const std::size_t begin =
                    static_cast<std::size_t>(row) * value_row_size +
                    static_cast<std::size_t>(column) * kStateDim * test_case.value_heads;
                const std::size_t end =
                    begin + static_cast<std::size_t>(kStateDim * test_case.value_heads);
                if (!std::all_of(output_bits.begin() + begin, output_bits.begin() + end,
                                 [](std::uint16_t value) { return value == 0; })) {
                    std::cerr << label << ": invalid output tail is not exact zero\n";
                    ++failures;
                    row = batch;
                    break;
                }
            }
        }
    }

    const std::vector<float> got_states = from_device<float>(states.data(), initial_states.size());
    for (int row = 0; row < batch; ++row) {
        const int valid = masked ? valid_columns[static_cast<std::size_t>(row)] : width;
        const std::size_t begin =
            static_cast<std::size_t>(snapshot_bases[static_cast<std::size_t>(row)]) * state_size;
        failures += verify_recurrence(
            label + " row " + std::to_string(row) + " snapshots",
            doubles(std::vector<float>(got_states.begin() + begin,
                                       got_states.begin() + begin +
                                           static_cast<std::size_t>(valid) * state_size)),
            references[static_cast<std::size_t>(row)].snapshots,
            gated_delta_net_state_fp32_criterion());
    }
    for (int slot = 0; slot < slots; ++slot) {
        if (written_slots[static_cast<std::size_t>(slot)]) continue;
        const std::size_t begin = static_cast<std::size_t>(slot) * state_size;
        failures += verify_exact(
            label + " untouched slot " + std::to_string(slot),
            std::vector<float>(got_states.begin() + begin, got_states.begin() + begin + state_size),
            std::vector<float>(initial_states.begin() + begin,
                               initial_states.begin() + begin + state_size));
    }
    failures +=
        verify_exact(label + " initial selectors unchanged",
                     from_device_i32(device_initial_slots, initial_slots.size()), initial_slots);
    failures +=
        verify_exact(label + " snapshot bases unchanged",
                     from_device_i32(device_snapshot_bases, snapshot_bases.size()), snapshot_bases);
    if (masked) {
        failures += verify_exact(label + " valid columns unchanged",
                                 from_device_i32(device_valid_columns, valid_columns.size()),
                                 valid_columns);
    }
    failures += states.verify_guards((label + " states").c_str());
    failures += out.verify_guards((label + " out").c_str());
    failures += verify_common_inputs_unchanged(label, aggregate, device.q, device.k, device.v,
                                               device.g, device.beta);
    return failures;
}

int contract_rejection_cases() {
    DeviceBuffer q_buffer(kStateDim * 8 * sizeof(std::uint16_t));
    DeviceBuffer k_buffer(kStateDim * 8 * sizeof(std::uint16_t));
    DeviceBuffer v_buffer(kStateDim * 8 * sizeof(std::uint16_t));
    DeviceBuffer g_buffer(8 * sizeof(float));
    DeviceBuffer beta_buffer(8 * sizeof(float));
    DeviceBuffer state_buffer(kStateDim * kStateDim * 8 * sizeof(float));
    DeviceBuffer out_buffer(kStateDim * 8 * sizeof(std::uint16_t));
    WorkspaceArena workspace(256);
    const float scale = 1.0f / std::sqrt(static_cast<float>(kStateDim));

    auto is_rejected = [&](int activation_dim, int state_dim, int qk_heads, int value_heads) {
        Tensor q(q_buffer.p, DType::BF16, {activation_dim, qk_heads, 1});
        Tensor k(k_buffer.p, DType::BF16, {activation_dim, qk_heads, 1});
        Tensor v(v_buffer.p, DType::BF16, {activation_dim, value_heads, 1});
        Tensor g(g_buffer.p, DType::FP32, {value_heads, 1});
        Tensor beta(beta_buffer.p, DType::FP32, {value_heads, 1});
        Tensor state(state_buffer.p, DType::FP32, {state_dim, state_dim, value_heads});
        Tensor out(out_buffer.p, DType::BF16, {activation_dim, value_heads, 1});
        try {
            ops::gated_delta_net(q, k, v, g, beta, scale, true, workspace, state, out, nullptr);
        } catch (const std::invalid_argument&) { return true; }
        cuda_synchronize();
        return false;
    };

    int failures = 0;
    if (!is_rejected(64, kStateDim, 4, 8)) {
        std::cerr << "gated_delta_net accepted Q/K/V head dimension 64\n";
        ++failures;
    }
    if (!is_rejected(kStateDim, 64, 4, 8)) {
        std::cerr << "gated_delta_net accepted state dimension 64\n";
        ++failures;
    }
    if (!is_rejected(kStateDim, kStateDim, 4, 6)) {
        std::cerr << "gated_delta_net accepted a non-divisible head map\n";
        ++failures;
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

    for (const bool normalize_qk : {false, true}) {
        const std::size_t interval =
            ops::gated_delta_net_workspace_capacity_bytes(16, 48, normalize_qk, 63, 65);
        const std::size_t witness =
            ops::gated_delta_net_workspace_capacity_bytes(16, 48, normalize_qk, 65, 65);
        if (interval != witness) {
            std::cerr << "gated_delta_net interval capacity missed the chunk boundary\n";
            ++failures;
        }
    }
    const std::size_t before_fusion =
        ops::gated_delta_net_workspace_capacity_bytes(48, 48, true, 447, 447);
    const std::size_t after_fusion =
        ops::gated_delta_net_workspace_capacity_bytes(48, 48, true, 448, 448);
    const std::size_t crossing =
        ops::gated_delta_net_workspace_capacity_bytes(48, 48, true, 447, 448);
    if (crossing != std::max(before_fusion, after_fusion)) {
        std::cerr << "gated_delta_net interval capacity missed the fused-Q crossover\n";
        ++failures;
    }
    try {
        (void)ops::gated_delta_net_workspace_capacity_bytes(16, 48, true, 0, 65);
        std::cerr << "gated_delta_net accepted an invalid token interval\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
    try {
        (void)ops::gated_delta_net_workspace_capacity_bytes(4, 6, true, 1, 65);
        std::cerr << "gated_delta_net workspace accepted a non-divisible head map\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
    failures += contract_rejection_cases();

    // Registered 27B/35B-A3B geometries, public state forms, and the recurrent/chunk/tail route
    // boundary are all qualified directly against the same complete FP64 recurrence.
    failures += inplace_case({"27b decode fused-qk-norm", 16, 48, 1, true}, 12001u);
    failures += distinct_state_case({"27b raw-qk small-T", 16, 48, 7, false}, 12007u);
    failures += distinct_state_case({"35b pre-chunk fused-qk-norm", 16, 32, 63, true}, 12063u);
    failures += distinct_state_case({"27b exact chunk fused-qk-norm", 16, 48, 64, true}, 12064u);
    failures += distinct_state_case({"27b exact chunk raw-qk", 16, 48, 64, false}, 12164u);
    failures += inplace_case({"35b chunk-tail fused-qk-norm", 16, 32, 65, true}, 12065u);
    failures += distinct_state_case({"generic grouped-map chunk-tail", 3, 12, 65, true}, 12365u);
    failures += distinct_state_case({"27b two-chunk fused-qk-norm", 16, 48, 128, true}, 12128u);
    failures += distinct_state_case({"27b production-tail fused-qk-norm", 16, 48, 3404, true},
                                    15404u);
    failures += distinct_state_case(
        {"Qwen4 materialized-Q crossover predecessor", 48, 48, 447, true}, 15447u);
    failures +=
        distinct_state_case({"Qwen4 fused-Q crossover", 48, 48, 448, true}, 15448u);
    failures +=
        distinct_state_case({"Qwen4 fused-Q crossover tail", 48, 48, 449, true}, 15449u);
    failures +=
        distinct_state_case({"Qwen4 materialized-K crossover predecessor", 48, 48, 511, true},
                            15511u);
    failures +=
        distinct_state_case({"Qwen4 fused-K crossover", 48, 48, 512, true}, 15512u);
    failures +=
        distinct_state_case({"Qwen4 fused-K crossover tail", 48, 48, 513, true}, 15513u);
    failures += distinct_state_case({"27b grouped K-fusion exclusion", 16, 48, 512, true},
                                    16512u);
    failures += inplace_case({"35b two-chunk raw-qk", 16, 32, 128, false}, 12228u);
    failures += partition_case({"27b chunk boundary", 16, 48, 128, true}, {64, 64}, 12428u);
    failures += partition_case({"27b chunk tail", 16, 48, 65, true}, {64, 1}, 12465u);
    // Qwen4 expands Q/K to the 48 value heads before recurrence. Exercise that exact geometry at
    // maximum width, qualifying scalar, one-shot, aligned, and Program boundary schedules
    // independently against one complete FP64 recurrence.
    failures += qwen4_maximum_schedule_case();

    // Snapshot is a separate public state transition. Nonzero source slots also prove that the
    // selected initial state, not slot zero, seeds the complete recurrence.
    failures += snapshot_case({"27b verify fused-qk-norm", 16, 48, 4, true}, 8, 7, 1, 12104u);
    failures += snapshot_case({"35b verify fused-qk-norm near-zero", 16, 32, 4, true, true}, 8, 6,
                              1, 12204u);
    failures +=
        batched_snapshot_case({"35b ordinary", 16, 32, 1, true}, {8, 9, 10, 11, 12, 13, 14, 15},
                              {0, 1, 2, 3, 4, 5, 6, 7}, {}, 16, 13001u);
    failures += batched_snapshot_case({"27b MTP", 16, 48, 6, true}, {18, 19, 20}, {0, 6, 12},
                                      {6, 3, 1}, 21, 13006u);
    // Row 0's initial state is its final destination; every state tile must load it before write.
    failures += batched_snapshot_case({"35b DFlash", 16, 32, 16, true}, {15, 33}, {0, 16}, {16, 7},
                                      34, 13016u);

    std::cout << (failures == 0 ? "OK" : "FAIL") << " gated_delta_net correctness\n";
    return failures == 0 ? 0 : 1;
}
