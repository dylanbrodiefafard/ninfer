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

int run_case(std::int32_t tokens, std::uint32_t seed) {
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
    ops::moe_route(logits_tensor, ids_tensor, weights_tensor, scale_tensor, nullptr);
    cuda_synchronize();

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
    failures += device_ids.verify_guards(label + " ids");
    failures += device_weights.verify_guards(label + " weights");
    failures += device_scale.verify_guards(label + " shared scale");
    return failures;
}

} // namespace

int main() {
    if (require_cuda() != 0) { return 1; }
    try {
        int failures = 0;
        for (const std::int32_t tokens : {1, 5, 4099}) {
            failures += run_case(tokens, 77U + static_cast<std::uint32_t>(tokens));
        }
        std::cout << (failures == 0 ? "OK" : "FAIL") << " moe_route\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "moe_route: " << error.what() << '\n';
        return 1;
    }
}
