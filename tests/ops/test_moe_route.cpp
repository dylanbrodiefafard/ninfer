#include "core/decode_graph.h"
#include "core/device.h"
#include "ninfer/ops/moe_route.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kRows    = ops::kMoeControlRows;
constexpr std::int32_t kExperts = ops::kMoeExperts;
constexpr std::int32_t kTopK    = ops::kMoeTopK;

// Weights are ratios of FP32 exponentials of differences of FP32 logits (a few ulps); the shared
// scale is one FP32 sigmoid.
constexpr PointwiseCriterion weight_criterion() {
    return {/*absolute*/ 1.0e-7, /*relative*/ 2.0e-6};
}

int run_case(std::int32_t tokens, std::uint32_t seed, cudaStream_t stream,
             std::int32_t packed_width = 1, std::int32_t rows = 0) {
    std::vector<float> logits(static_cast<std::size_t>(kRows) * tokens);
    fill_uniform(logits, seed, -6.0F, 6.0F);
    // Exact ties: some tokens repeat values across experts, including at the top-10 boundary.
    for (std::int32_t t = 0; t < tokens; t += 3) {
        float* row = logits.data() + static_cast<std::size_t>(t) * kRows;
        for (std::int32_t e = 0; e < kExperts; e += 7) { row[e] = 5.5F; }
    }
    // More than ten tied maxima, spread over lanes and slots: the lowest ten ids win, each 0.1.
    for (std::int32_t e = 3; e < kExperts; e += 41) { logits[e] = 7.0F; }
    std::vector<std::int32_t> expected_ids(static_cast<std::size_t>(kTopK) * tokens);
    std::vector<double> expected_weights(expected_ids.size()), expected_scale(tokens);
    for (std::int32_t t = 0; t < tokens; ++t) {
        const float* row = logits.data() + static_cast<std::size_t>(t) * kRows;
        double maximum   = row[0];
        for (std::int32_t e = 1; e < kExperts; ++e) { maximum = std::max(maximum, double(row[e])); }
        std::vector<double> p(kExperts);
        double sum = 0.0;
        for (std::int32_t e = 0; e < kExperts; ++e) {
            sum += p[e] = std::exp(double(row[e]) - maximum);
        }
        for (auto& value : p) { value /= sum; }
        std::vector<std::int32_t> order(kExperts);
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(),
                         [&](std::int32_t a, std::int32_t b) { return p[a] > p[b]; });
        double selected = 0.0;
        for (std::int32_t k = 0; k < kTopK; ++k) { selected += p[order[k]]; }
        for (std::int32_t k = 0; k < kTopK; ++k) {
            expected_ids[std::size_t(t) * kTopK + k]     = order[k];
            expected_weights[std::size_t(t) * kTopK + k] = p[order[k]] / selected;
        }
        expected_scale[t] = 1.0 / (1.0 + std::exp(-double(row[ops::kMoeSharedGateRow])));
    }

    GuardedDeviceBuffer device_logits(logits.size() * sizeof(float));
    GuardedDeviceBuffer device_ids(expected_ids.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer device_weights(expected_weights.size() * sizeof(float));
    GuardedDeviceBuffer device_scale(static_cast<std::size_t>(tokens) * sizeof(float));
    device_logits.copy_from_host(logits.data(), device_logits.bytes());
    Tensor logits_tensor(device_logits.data(), DType::FP32, {kRows, tokens});
    Tensor ids_tensor(device_ids.data(), DType::I32, {kTopK, tokens});
    Tensor weights_tensor(device_weights.data(), DType::FP32, {kTopK, tokens});
    Tensor scale_tensor(device_scale.data(), DType::FP32, {tokens});
    ops::moe_route(logits_tensor, ids_tensor, weights_tensor, scale_tensor, {}, 1, stream);
    cuda_synchronize(stream);

    const std::string label = "moe_route T=" + std::to_string(tokens);
    int failures = verify_exact((label + " ids").c_str(),
                                from_device<std::int32_t>(device_ids.data(), expected_ids.size()),
                                expected_ids);
    const auto weights = from_device<float>(device_weights.data(), expected_weights.size());
    failures +=
        verify_pointwise(label + " weights", std::vector<double>(weights.begin(), weights.end()),
                         expected_weights, weight_criterion());
    const auto scale = from_device<float>(device_scale.data(), static_cast<std::size_t>(tokens));
    failures +=
        verify_pointwise(label + " shared scale", std::vector<double>(scale.begin(), scale.end()),
                         expected_scale, weight_criterion());
    if (rows != 0) {
        GuardedDeviceBuffer device_valid(static_cast<std::size_t>(rows) * sizeof(std::int32_t));
        Tensor valid_tensor(device_valid.data(), DType::I32, {rows});
        DecodeGraphDefinition definition;
        DecodeGraphExecutable graph;
        definition.capture(stream, [&] {
            ops::moe_route(logits_tensor, ids_tensor, weights_tensor, scale_tensor, valid_tensor,
                           packed_width, stream);
        });
        graph.instantiate(definition);
        // Replay with different counts at the same addresses, including full-width rows.
        for (std::int32_t pass = 0; pass < 2; ++pass) {
            std::vector<std::int32_t> valid(rows);
            for (std::int32_t b = 0; b < rows; ++b) { valid[b] = 1 + (b + pass) % packed_width; }
            device_valid.copy_from_host(valid.data(), device_valid.bytes());
            graph.launch(stream);
            cuda_synchronize(stream);
            const auto ids    = from_device<std::int32_t>(device_ids.data(), expected_ids.size());
            const auto routed = from_device<float>(device_weights.data(), expected_weights.size());
            const auto shared = from_device<float>(device_scale.data(), expected_scale.size());
            std::vector<std::int32_t> exact_ids(ids.size());
            std::vector<float> exact_weights(routed.size()), exact_scale(shared.size());
            std::vector<double> oracle_weights(routed.size()), oracle_scale(shared.size());
            for (std::int32_t t = 0; t < tokens; ++t) {
                const auto b      = t / packed_width;
                const auto c      = t % packed_width;
                const auto source = b * packed_width + std::min(c, valid[b] - 1);
                for (std::int32_t k = 0; k < kTopK; ++k) {
                    const auto dst      = static_cast<std::size_t>(t) * kTopK + k;
                    const auto src      = static_cast<std::size_t>(source) * kTopK + k;
                    exact_ids[dst]      = expected_ids[src];
                    exact_weights[dst]  = weights[src];
                    oracle_weights[dst] = expected_weights[src];
                }
                exact_scale[t]  = scale[source];
                oracle_scale[t] = expected_scale[source];
            }
            const std::string packed_label = label + " W=" + std::to_string(packed_width) +
                                             " B=" + std::to_string(rows) +
                                             " replay=" + std::to_string(pass);
            failures += verify_exact((packed_label + " ids").c_str(), ids, exact_ids);
            failures +=
                verify_exact((packed_label + " exact weights").c_str(), routed, exact_weights);
            failures += verify_exact((packed_label + " exact scale").c_str(), shared, exact_scale);
            failures += verify_pointwise(packed_label + " oracle weights",
                                         std::vector<double>(routed.begin(), routed.end()),
                                         oracle_weights, weight_criterion());
            failures += verify_pointwise(packed_label + " oracle scale",
                                         std::vector<double>(shared.begin(), shared.end()),
                                         oracle_scale, weight_criterion());
            failures += device_valid.verify_guards(packed_label + " counts");
            failures +=
                verify_exact((packed_label + " preserved counts").c_str(),
                             from_device<std::int32_t>(device_valid.data(), valid.size()), valid);
        }
    }
    failures += device_logits.verify_guards(label + " logits");
    failures += device_ids.verify_guards(label + " ids");
    failures += device_weights.verify_guards(label + " weights");
    failures += device_scale.verify_guards(label + " shared scale");
    return failures;
}

} // namespace

int main() {
    if (require_cuda() != 0) { return 1; }
    try {
        DeviceContext context;
        int failures = 0;
        for (const std::int32_t tokens : {1, 5, 4099}) {
            failures += run_case(tokens, 77U + static_cast<std::uint32_t>(tokens), context.stream);
        }
        for (const std::int32_t width : {1, 2, 3, 4}) {
            for (const std::int32_t rows : {1, 5, 12}) {
                failures += run_case(width * rows, 143U, context.stream, width, rows);
            }
        }
        std::cout << (failures == 0 ? "OK" : "FAIL") << " moe_route\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "moe_route: " << error.what() << '\n';
        return 1;
    }
}
