#include "ninfer/ops/vision_pos_embed.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

// The independent oracle reduces four represented products in FP64. Production's natural FP32
// association can select the adjacent staged BF16 value; after residual cancellation that one
// position ULP can be several output ULPs. This absolute-plus-relative pointwise envelope is the
// measured bound for that reduction profile. Separate exact fixtures below make the two semantic
// BF16 casts non-negotiable on every kernel route.
constexpr PointwiseCriterion kVisionPositionEmbeddingTolerance{
    /*absolute=*/4.1e-3,
    /*relative=*/3.95e-3,
};

std::vector<std::uint16_t> as_bf16_bits(const std::vector<float>& values) {
    std::vector<std::uint16_t> bits(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) { bits[i] = f32_to_bf16(values[i]); }
    return bits;
}

std::uint16_t f64_to_bf16(double value) {
    // Search around the FP32-derived candidate, then select the nearest represented BF16 value
    // directly in FP64. This avoids an FP64 -> FP32 -> BF16 double-rounding boundary in the
    // independent mathematical oracle.
    const std::uint16_t center = f32_to_bf16(static_cast<float>(value));
    std::uint16_t best         = center;
    double best_distance       = std::abs(value - static_cast<double>(bf16_to_f32(center)));
    for (int delta = -2; delta <= 2; ++delta) {
        const int candidate_int = static_cast<int>(center) + delta;
        if (candidate_int < 0 || candidate_int > std::numeric_limits<std::uint16_t>::max()) {
            continue;
        }
        const auto candidate = static_cast<std::uint16_t>(candidate_int);
        const double represented = static_cast<double>(bf16_to_f32(candidate));
        if (!std::isfinite(represented)) { continue; }
        const double distance = std::abs(value - represented);
        if (distance < best_distance ||
            (distance == best_distance && (candidate & 1u) == 0u && (best & 1u) != 0u)) {
            best          = candidate;
            best_distance = distance;
        }
    }
    return best;
}

int run_case(std::int32_t dimensions, std::int32_t table_rows, std::int32_t patches,
             std::uint32_t seed) {

    std::vector<float> table_values(static_cast<std::size_t>(dimensions) * table_rows);
    std::vector<float> x_values(static_cast<std::size_t>(dimensions) * patches);
    std::vector<float> weights(static_cast<std::size_t>(4) * patches);
    std::vector<std::int32_t> indices(static_cast<std::size_t>(4) * patches);
    fill_uniform(table_values, seed, -2.0f, 2.0f);
    fill_uniform(x_values, seed + 1u, -8.0f, 8.0f);
    fill_uniform(weights, seed + 2u, 0.05f, 1.0f);
    round_to_bf16(table_values);
    round_to_bf16(x_values);

    for (std::int32_t patch = 0; patch < patches; ++patch) {
        double sum = 0.0;
        for (std::int32_t corner = 0; corner < 4; ++corner) {
            const std::size_t control = static_cast<std::size_t>(patch) * 4 + corner;
            indices[control] =
                static_cast<std::int32_t>((patch * 37 + corner * 101 + 11) % table_rows);
            sum += static_cast<double>(weights[control]);
        }
        for (std::int32_t corner = 0; corner < 4; ++corner) {
            const std::size_t control = static_cast<std::size_t>(patch) * 4 + corner;
            weights[control] = static_cast<float>(static_cast<double>(weights[control]) / sum);
        }
    }

    std::vector<double> reference(x_values.size());
    for (std::int32_t patch = 0; patch < patches; ++patch) {
        for (std::int32_t dimension = 0; dimension < dimensions; ++dimension) {
            double position = 0.0;
            for (std::int32_t corner = 0; corner < 4; ++corner) {
                const std::size_t control = static_cast<std::size_t>(patch) * 4 + corner;
                position +=
                    static_cast<double>(
                        table_values[static_cast<std::size_t>(indices[control]) * dimensions +
                                     dimension]) *
                    static_cast<double>(weights[control]);
            }
            const std::size_t output = static_cast<std::size_t>(patch) * dimensions + dimension;
            const float staged_position = bf16_to_f32(f64_to_bf16(position));
            reference[output] = static_cast<double>(
                bf16_to_f32(f32_to_bf16(x_values[output] + staged_position)));
        }
    }

    const auto table_bits = as_bf16_bits(table_values);
    const auto x_bits     = as_bf16_bits(x_values);
    GuardedDeviceBuffer device_table(table_bits.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_indices(indices.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer device_weights(weights.size() * sizeof(float));
    GuardedDeviceBuffer device_x(x_bits.size() * sizeof(std::uint16_t));
    device_table.copy_from_host(table_bits.data(), table_bits.size() * sizeof(std::uint16_t));
    device_indices.copy_from_host(indices.data(), indices.size() * sizeof(std::int32_t));
    device_weights.copy_from_host(weights.data(), weights.size() * sizeof(float));
    device_x.copy_from_host(x_bits.data(), x_bits.size() * sizeof(std::uint16_t));

    Tensor table_tensor(device_table.data(), DType::BF16, {dimensions, table_rows});
    Tensor indices_tensor(device_indices.data(), DType::I32, {4, patches});
    Tensor weights_tensor(device_weights.data(), DType::FP32, {4, patches});
    Tensor x_tensor(device_x.data(), DType::BF16, {dimensions, patches});
    ops::vision_pos_embed_add(table_tensor, indices_tensor, weights_tensor, x_tensor, nullptr);
    cuda_synchronize();

    const std::string label = "vision_pos_embed D=" + std::to_string(dimensions) +
                              " R=" + std::to_string(table_rows) +
                              " P=" + std::to_string(patches);
    int failures = verify_pointwise(label.c_str(), from_device_bf16(device_x.data(), x_bits.size()),
                                    reference, kVisionPositionEmbeddingTolerance);
    failures += verify_exact((label + " preserves table").c_str(),
                             from_device<std::uint16_t>(device_table.data(), table_bits.size()),
                             table_bits);
    failures +=
        verify_exact((label + " preserves indices").c_str(),
                     from_device<std::int32_t>(device_indices.data(), indices.size()), indices);
    failures += verify_exact((label + " preserves weights").c_str(),
                             from_device<float>(device_weights.data(), weights.size()), weights);
    failures += device_table.verify_guards((label + " table").c_str());
    failures += device_indices.verify_guards((label + " indices").c_str());
    failures += device_weights.verify_guards((label + " weights").c_str());
    failures += device_x.verify_guards((label + " x").c_str());
    return failures;
}

int run_staging_boundary_case(std::int32_t dimensions, std::int32_t patches) {
    constexpr float kTableValue = -1.8125f;
    constexpr float kWeight     = -0.7315477728843689f;
    constexpr float kResidual   = -0.02685546875f;
    constexpr std::uint16_t kExpected = 0x3fa7; // 1.3046875
    constexpr std::uint16_t kOldFused = 0x3fa6; // 1.296875

    std::vector<std::uint16_t> table(static_cast<std::size_t>(dimensions),
                                     f32_to_bf16(kTableValue));
    std::vector<std::uint16_t> x(static_cast<std::size_t>(dimensions) * patches,
                                 f32_to_bf16(kResidual));
    std::vector<std::int32_t> indices(static_cast<std::size_t>(4) * patches, 0);
    std::vector<float> weights(static_cast<std::size_t>(4) * patches, 0.0f);
    for (std::int32_t patch = 0; patch < patches; ++patch) {
        weights[static_cast<std::size_t>(patch) * 4] = kWeight;
    }
    if (f32_to_bf16(kResidual + kTableValue * kWeight) != kOldFused ||
        f32_to_bf16(kResidual + bf16_to_f32(f32_to_bf16(kTableValue * kWeight))) != kExpected) {
        std::cerr << "FAIL: invalid position staging boundary fixture\n";
        return 1;
    }

    GuardedDeviceBuffer device_table(table.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer device_indices(indices.size() * sizeof(std::int32_t));
    GuardedDeviceBuffer device_weights(weights.size() * sizeof(float));
    GuardedDeviceBuffer device_x(x.size() * sizeof(std::uint16_t));
    device_table.copy_from_host(table.data(), table.size() * sizeof(std::uint16_t));
    device_indices.copy_from_host(indices.data(), indices.size() * sizeof(std::int32_t));
    device_weights.copy_from_host(weights.data(), weights.size() * sizeof(float));
    device_x.copy_from_host(x.data(), x.size() * sizeof(std::uint16_t));

    Tensor table_tensor(device_table.data(), DType::BF16, {dimensions, 1});
    Tensor indices_tensor(device_indices.data(), DType::I32, {4, patches});
    Tensor weights_tensor(device_weights.data(), DType::FP32, {4, patches});
    Tensor x_tensor(device_x.data(), DType::BF16, {dimensions, patches});
    ops::vision_pos_embed_add(table_tensor, indices_tensor, weights_tensor, x_tensor, nullptr);
    cuda_synchronize();

    const auto output = from_device<std::uint16_t>(device_x.data(), x.size());
    int failures = 0;
    for (std::size_t i = 0; i < output.size(); ++i) {
        if (output[i] != kExpected) {
            std::cerr << "FAIL: position staging boundary D=" << dimensions
                      << " P=" << patches << " element " << i << " expected 0x" << std::hex
                      << kExpected << " got 0x" << output[i] << std::dec << '\n';
            ++failures;
            break;
        }
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
    failures += run_case(1152, 2304, 17, 1u);
    failures += run_case(1152, 2304, 1024, 11u);
    failures += run_case(7, 23, 19, 21u);
    failures += run_staging_boundary_case(1152, 1);
    failures += run_staging_boundary_case(1152, 1024);
    failures += run_staging_boundary_case(7, 1);
    std::cout << (failures ? "FAIL" : "OK") << " vision_pos_embed\n";
    return failures ? 1 : 0;
}
