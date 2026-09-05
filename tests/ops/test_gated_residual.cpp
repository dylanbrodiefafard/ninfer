#include "ninfer/ops/gated_residual.h"

#include "ops/op_tester.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kHidden = 2560;
constexpr std::int32_t kBranches = 4;
constexpr std::int32_t kFlat = kHidden * kBranches;
constexpr std::int32_t kRank = 320;
constexpr std::size_t kQ8BlockBytes = 34;

constexpr ReductionCriterion kReadCriterion{/*relative_l2=*/6.0e-3,
                                             /*gross_absolute=*/4.0e-3,
                                             /*gross_relative_to_max_reference=*/5.0e-3};
constexpr ReductionCriterion kScaleCriterion{/*relative_l2=*/3.5e-3,
                                              /*gross_absolute=*/1.5e-3,
                                              /*gross_relative_to_max_reference=*/3.0e-3};
constexpr ReductionCriterion kInjectCriterion{/*relative_l2=*/3.0e-3,
                                               /*gross_absolute=*/2.0e-3,
                                               /*gross_relative_to_max_reference=*/2.0e-3};

double sigmoid(double value) {
    if (value >= 0.0) { return 1.0 / (1.0 + std::exp(-value)); }
    const double exponential = std::exp(value);
    return exponential / (1.0 + exponential);
}

double silu(double value) { return value * sigmoid(value); }

std::uint16_t read_u16(const std::uint8_t* bytes) {
    return static_cast<std::uint16_t>(bytes[0]) |
           (static_cast<std::uint16_t>(bytes[1]) << 8U);
}

void write_u16(std::uint8_t* bytes, std::uint16_t value) {
    bytes[0] = static_cast<std::uint8_t>(value & 0xffU);
    bytes[1] = static_cast<std::uint8_t>(value >> 8U);
}

double half_to_double(std::uint16_t word) {
    const double sign = (word & 0x8000U) != 0 ? -1.0 : 1.0;
    const unsigned exponent = (word >> 10U) & 0x1fU;
    const unsigned fraction = word & 0x3ffU;
    if (exponent == 0) { return sign * std::ldexp(static_cast<double>(fraction), -24); }
    if (exponent == 31) {
        return fraction == 0 ? sign * std::numeric_limits<double>::infinity()
                             : std::numeric_limits<double>::quiet_NaN();
    }
    return sign * std::ldexp(static_cast<double>(1024U + fraction),
                             static_cast<int>(exponent) - 25);
}

int signed_byte(std::uint8_t value) {
    return value < 128U ? static_cast<int>(value) : static_cast<int>(value) - 256;
}

std::vector<std::uint8_t> make_q8(std::int32_t rows, std::int32_t columns, std::uint32_t seed) {
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(rows) * (columns / 32) * kQ8BlockBytes);
    std::mt19937 generator(seed);
    std::uniform_int_distribution<int> distribution(-31, 31);
    for (std::size_t offset = 0; offset < bytes.size(); offset += kQ8BlockBytes) {
        write_u16(bytes.data() + offset, 0x1800U); // exactly 2^-9
        for (int i = 0; i < 32; ++i) {
            bytes[offset + 2 + i] = static_cast<std::uint8_t>(distribution(generator));
        }
    }
    return bytes;
}

Weight q8_weight(void* data, std::uint64_t bytes, std::int32_t rows, std::int32_t columns) {
    Weight weight{};
    weight.payload = data;
    weight.payload_bytes = bytes;
    weight.qdata = data;
    weight.qtype = QType::GGML_Q8_0;
    weight.group_size = 32;
    weight.group = 32;
    weight.layout = QuantLayout::GgmlBlockRow;
    weight.ndim = 2;
    weight.n = rows;
    weight.k = columns;
    weight.shape[0] = rows;
    weight.shape[1] = columns;
    weight.padded_shape[0] = rows;
    weight.padded_shape[1] = columns;
    return weight;
}

std::vector<double> q8_project(const std::vector<std::uint8_t>& weight, std::int32_t rows,
                               std::int32_t columns, const std::vector<double>& input) {
    const std::size_t row_bytes = static_cast<std::size_t>(columns / 32) * kQ8BlockBytes;
    std::vector<double> output(rows);
    for (std::int32_t row = 0; row < rows; ++row) {
        const auto* row_data = weight.data() + static_cast<std::size_t>(row) * row_bytes;
        double sum = 0.0;
        for (std::int32_t block = 0; block < columns / 32; ++block) {
            const auto* bytes = row_data + static_cast<std::size_t>(block) * kQ8BlockBytes;
            const double scale = half_to_double(read_u16(bytes));
            for (int item = 0; item < 32; ++item) {
                sum += scale * signed_byte(bytes[2 + item]) *
                       input[static_cast<std::size_t>(block) * 32 + item];
            }
        }
        output[row] = sum;
    }
    return output;
}

struct Fixture {
    std::vector<float> norm;
    std::vector<std::uint8_t> down;
    std::vector<std::uint8_t> up;
    std::vector<float> write;
    DeviceBuffer d_norm;
    DeviceBuffer d_down;
    DeviceBuffer d_up;
    DeviceBuffer d_write;

    Fixture()
        : norm(kFlat), down(make_q8(kRank, kFlat, 7102U)),
          up(make_q8(kFlat, kRank, 7103U)), write(static_cast<std::size_t>(kBranches) * kFlat) {
        fill_uniform(norm, 7101U, 0.80F, 1.20F);
        fill_uniform(write, 7104U, -0.003F, 0.003F);
        d_norm = to_device_f32(norm);
        d_down = to_device(down);
        d_up = to_device(up);
        d_write = to_device_f32(write);
    }
};

struct OracleResult {
    std::vector<double> mixed;
    std::vector<double> scale;
};

OracleResult oracle(const Fixture& fixture, const std::vector<float>& residual,
                    std::int32_t tokens) {
    OracleResult result{std::vector<double>(static_cast<std::size_t>(kHidden) * tokens),
                        std::vector<double>(static_cast<std::size_t>(kBranches) * tokens)};
    for (std::int32_t token = 0; token < tokens; ++token) {
        const std::size_t token_base = static_cast<std::size_t>(token) * kFlat;
        std::vector<double> normalized(kFlat);
        for (std::int32_t branch = 0; branch < kBranches; ++branch) {
            const std::size_t base = static_cast<std::size_t>(branch) * kHidden;
            double sum_squares = 0.0;
            for (std::int32_t d = 0; d < kHidden; ++d) {
                const double value = residual[token_base + base + d];
                sum_squares += value * value;
            }
            const double inverse = 1.0 / std::sqrt(sum_squares / kHidden + 1.0e-6);
            for (std::int32_t d = 0; d < kHidden; ++d) {
                normalized[base + d] =
                    residual[token_base + base + d] * inverse * fixture.norm[base + d];
            }
        }

        std::vector<double> low_rank = q8_project(fixture.down, kRank, kFlat, normalized);
        for (double& value : low_rank) { value = silu(value / 4.0); }
        const std::vector<double> up = q8_project(fixture.up, kFlat, kRank, low_rank);
        for (std::int32_t d = 0; d < kHidden; ++d) {
            double mixed = 0.0;
            for (std::int32_t branch = 0; branch < kBranches; ++branch) {
                const std::size_t index = static_cast<std::size_t>(branch) * kHidden + d;
                mixed += sigmoid(up[index]) * normalized[index];
            }
            result.mixed[static_cast<std::size_t>(token) * kHidden + d] = mixed / 4.0;
        }
        for (std::int32_t branch = 0; branch < kBranches; ++branch) {
            double projected = 0.0;
            const std::size_t base = static_cast<std::size_t>(branch) * kFlat;
            for (std::int32_t k = 0; k < kFlat; ++k) {
                projected += fixture.write[base + k] * normalized[k];
            }
            result.scale[static_cast<std::size_t>(token) * kBranches + branch] =
                2.0 * sigmoid(projected / 4.0);
        }
    }
    return result;
}

std::vector<std::uint16_t> bits(const void* device, std::size_t words) {
    return from_device<std::uint16_t>(device, words);
}

int run_case(const Fixture& fixture, std::int32_t tokens, const char* label) {
    std::vector<float> residual(static_cast<std::size_t>(kFlat) * tokens);
    std::vector<float> block_output(static_cast<std::size_t>(kHidden) * tokens);
    fill_uniform(residual, 7201U + tokens, -0.70F, 0.70F);
    fill_uniform(block_output, 7202U + tokens, -0.30F, 0.30F);
    round_to_bf16(residual);
    round_to_bf16(block_output);
    const OracleResult expected = oracle(fixture, residual, tokens);

    DeviceBuffer d_residual = to_device_bf16(residual);
    DeviceBuffer d_block = to_device_bf16(block_output);
    GuardedDeviceBuffer d_mixed(static_cast<std::size_t>(kHidden) * tokens * 2);
    GuardedDeviceBuffer d_read_only(static_cast<std::size_t>(kHidden) * tokens * 2);
    GuardedDeviceBuffer d_scale(static_cast<std::size_t>(kBranches) * tokens * 2);
    GuardedDeviceBuffer d_injected(static_cast<std::size_t>(kFlat) * tokens * 2);
    d_mixed.fill(0xcd);
    d_read_only.fill(0xcd);
    d_scale.fill(0xcd);
    d_injected.fill(0xcd);

    Tensor residual_tensor(d_residual.p, DType::BF16, {kHidden, kBranches, tokens});
    Tensor norm_tensor(fixture.d_norm.p, DType::FP32, {kFlat});
    Tensor write_tensor(fixture.d_write.p, DType::FP32, {kFlat, kBranches});
    Tensor mixed_tensor(d_mixed.data(), DType::BF16, {kHidden, tokens});
    Tensor read_only_tensor(d_read_only.data(), DType::BF16, {kHidden, tokens});
    Tensor scale_tensor(d_scale.data(), DType::BF16, {kBranches, tokens});
    const Weight down_weight = q8_weight(fixture.d_down.p, fixture.d_down.bytes, kRank, kFlat);
    const Weight up_weight = q8_weight(fixture.d_up.p, fixture.d_up.bytes, kFlat, kRank);
    const std::size_t workspace_bytes = ops::gated_residual_workspace_capacity_bytes(tokens);
    WorkspaceArena workspace(workspace_bytes);

    ops::gated_residual_read_write(residual_tensor, norm_tensor, down_weight, up_weight,
                                   write_tensor, mixed_tensor, scale_tensor, workspace, nullptr);
    ops::gated_residual_read(residual_tensor, norm_tensor, down_weight, up_weight,
                             read_only_tensor, workspace, nullptr);
    cuda_synchronize();

    const auto actual_mixed = from_device_bf16(d_mixed.data(), expected.mixed.size());
    const auto actual_scale = from_device_bf16(d_scale.data(), expected.scale.size());
    int failures = verify_reduction(std::string(label) + " Q8 read", actual_mixed, expected.mixed,
                                    kReadCriterion);
    failures += verify_reduction(std::string(label) + " FP32 write", actual_scale, expected.scale,
                                 kScaleCriterion);
    const std::string read_only_label = std::string(label) + " final read-only form";
    failures += verify_exact(read_only_label.c_str(),
                             bits(d_read_only.data(), expected.mixed.size()),
                             bits(d_mixed.data(), expected.mixed.size()));

    Tensor block_tensor(d_block.p, DType::BF16, {kHidden, tokens});
    Tensor injected_tensor(d_injected.data(), DType::BF16, {kHidden, kBranches, tokens});
    DeviceBuffer d_in_place = to_device_bf16(residual);
    Tensor in_place_tensor(d_in_place.p, DType::BF16, {kHidden, kBranches, tokens});
    ops::gated_residual_inject(residual_tensor, block_tensor, scale_tensor, injected_tensor, nullptr);
    ops::gated_residual_inject(in_place_tensor, block_tensor, scale_tensor, in_place_tensor, nullptr);
    cuda_synchronize();

    std::vector<double> inject_expected(residual.size());
    for (std::int32_t token = 0; token < tokens; ++token) {
        for (std::int32_t branch = 0; branch < kBranches; ++branch) {
            for (std::int32_t d = 0; d < kHidden; ++d) {
                const std::size_t index = static_cast<std::size_t>(token) * kFlat +
                                          static_cast<std::size_t>(branch) * kHidden + d;
                inject_expected[index] =
                    residual[index] +
                    actual_scale[static_cast<std::size_t>(token) * kBranches + branch] *
                        block_output[static_cast<std::size_t>(token) * kHidden + d];
            }
        }
    }
    failures += verify_reduction(
        std::string(label) + " inject",
        from_device_bf16(d_injected.data(), inject_expected.size()), inject_expected,
        kInjectCriterion);
    const std::string inplace_label = std::string(label) + " in-place inject";
    failures += verify_exact(inplace_label.c_str(),
                             bits(d_in_place.p, residual.size()),
                             bits(d_injected.data(), residual.size()));
    std::vector<std::uint16_t> expected_residual_bits(residual.size());
    std::transform(residual.begin(), residual.end(), expected_residual_bits.begin(), f32_to_bf16);
    failures += verify_exact("gated residual input preservation",
                             bits(d_residual.p, residual.size()), expected_residual_bits);
    failures += d_mixed.verify_guards("gated residual mixed");
    failures += d_read_only.verify_guards("gated residual read-only");
    failures += d_scale.verify_guards("gated residual scale");
    failures += d_injected.verify_guards("gated residual injected");
    if (workspace.used() != 0 || workspace.peak_used() != workspace_bytes) {
        std::cerr << "FAIL gated residual workspace query/high-water mismatch\n";
        ++failures;
    }
    return failures;
}

} // namespace

int main() {
    if (const int unavailable = require_cuda()) { return unavailable; }
    Fixture fixture;
    int failures = run_case(fixture, 1, "gated residual T=1");
    failures += run_case(fixture, 3, "gated residual uneven T=3");
    failures += run_case(fixture, 64, "gated residual T=64");
    failures += run_case(fixture, 65, "gated residual T=65");
    try {
        const std::size_t broad = ops::gated_residual_workspace_capacity_bytes(4096);
        if (broad <= ops::gated_residual_workspace_capacity_bytes(65)) {
            std::cerr << "FAIL gated residual broad workspace capacity\n";
            ++failures;
        }
    } catch (const std::invalid_argument&) {
        std::cerr << "FAIL gated residual rejected T=4096 capacity\n";
        ++failures;
    }
    try {
        (void)ops::gated_residual_workspace_capacity_bytes(4097);
        std::cerr << "FAIL gated residual accepted T=4097 capacity\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
    std::cout << (failures ? "FAIL" : "OK") << " gated_residual\n";
    return failures == 0 ? 0 : 1;
}
