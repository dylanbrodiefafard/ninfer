#include "ninfer/ops/gdn_input_proj.h"

#include "ops/input_projection_test_common.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::input_projection;

namespace {

// This criterion belongs to the complete A16 GDN-input-projection Op.
constexpr ReductionCriterion kGdnInputProjA16Tolerance{3.0e-3, 4.0e-3, 3.5e-3};
constexpr ReductionCriterion kGdnInputProjA4Tolerance{0.16, 4.0e-3, 0.16};
constexpr std::int32_t kGdnInputProjA4SampleRows = 31;

int verify_output_range_sampled(std::string_view label, const GuardedBf16Tensor& output,
                                std::int32_t full_rows, std::int32_t output_row_offset,
                                std::int32_t output_rows,
                                const quantized_weight::PackedWeight& weight,
                                std::int32_t weight_row_offset,
                                const std::vector<float>& activation, std::int32_t hidden,
                                std::int32_t tokens, const ReductionCriterion& criterion,
                                std::int32_t sample_count) {
    const std::vector<double> values     = output.values();
    const std::vector<std::int32_t> rows = sampled_rows(output_rows, sample_count);
    std::vector<std::int32_t> selected_tokens;
    for (const std::int32_t token :
         {0, 1, tokens / 4, tokens / 2, (3 * tokens) / 4, tokens - 2, tokens - 1}) {
        if (token >= 0 && token < tokens &&
            std::find(selected_tokens.begin(), selected_tokens.end(), token) ==
                selected_tokens.end()) {
            selected_tokens.push_back(token);
        }
    }
    std::vector<double> actual;
    std::vector<double> expected;
    actual.reserve(rows.size() * selected_tokens.size());
    expected.reserve(rows.size() * selected_tokens.size());
    for (const std::int32_t local_row : rows) {
        const std::int32_t output_row = output_row_offset + local_row;
        const std::int32_t weight_row = weight_row_offset + local_row;
        for (const std::int32_t token : selected_tokens) {
            actual.push_back(values[static_cast<std::size_t>(token) * full_rows + output_row]);
            expected.push_back(quantized_weight::dot_fp64(
                weight, weight_row, activation.data() + static_cast<std::size_t>(token) * hidden,
                hidden));
        }
    }
    return compare(label, actual, expected, criterion);
}

int run_nvfp4_case(DevicePackedWeight& parent, std::int32_t tokens, ops::LinearPolicy policy) {
    const std::int32_t hidden           = parent.host.weight.k;
    constexpr std::int32_t kQkvRows     = 10240;
    constexpr std::int32_t kZRows       = 6144;
    constexpr std::int32_t kRows        = kQkvRows + kZRows;
    const std::vector<float> activation = make_bf16_activation(hidden, tokens, 601U + tokens);
    const std::vector<std::uint16_t> activation_bits = bf16_bits(activation);
    DeviceBuffer device_activation                   = to_device(activation_bits);
    GuardedBf16Tensor qkv(kQkvRows, tokens);
    GuardedBf16Tensor z(kZRows, tokens);
    Tensor x                   = Tensor(device_activation.p, DType::BF16, {hidden, tokens});
    Tensor qkv_output          = qkv.tensor();
    Tensor z_output            = z.tensor();
    const std::size_t capacity = ops::gdn_input_proj_workspace_capacity_bytes(
        QType::NVFP4, kRows, hidden, policy, tokens, tokens);
    WorkspaceArena workspace(std::max<std::size_t>(capacity, 256));
    ops::gdn_input_proj(x, parent.view(), qkv_output, z_output, policy, workspace, nullptr);
    cuda_synchronize();

    const bool a4                       = policy == ops::LinearPolicy::AllowA4 && tokens >= 3;
    const ReductionCriterion& criterion = a4 ? kGdnInputProjA4Tolerance : kGdnInputProjA16Tolerance;
    const std::int32_t sample_count     = a4 ? kGdnInputProjA4SampleRows : 7;
    const std::string suffix            = std::string(" NVFP4 K=") + std::to_string(hidden) +
                                          (a4 ? " A4" : " A16") + " T=" + std::to_string(tokens);
    int failures                        = qkv.verify_guards("gdn qkv" + suffix);
    failures += z.verify_guards("gdn z" + suffix);
    failures += qkv.verify_fully_written("gdn qkv" + suffix);
    failures += z.verify_fully_written("gdn z" + suffix);
    failures +=
        verify_output_range_sampled("gdn query" + suffix, qkv, kQkvRows, 0, 2048, parent.host, 0,
                                    activation, hidden, tokens, criterion, sample_count);
    failures +=
        verify_output_range_sampled("gdn key" + suffix, qkv, kQkvRows, 2048, 2048, parent.host,
                                    2048, activation, hidden, tokens, criterion, sample_count);
    failures +=
        verify_output_range_sampled("gdn value" + suffix, qkv, kQkvRows, 4096, 6144, parent.host,
                                    4096, activation, hidden, tokens, criterion, sample_count);
    failures +=
        verify_output_range_sampled("gdn z" + suffix, z, kZRows, 0, kZRows, parent.host, kQkvRows,
                                    activation, hidden, tokens, criterion, sample_count);
    if (workspace.peak_used() != capacity) {
        std::cerr << "gdn workspace" << suffix << ": query/execution high-water mismatch\n";
        ++failures;
    }
    failures += verify_preserved("gdn x" + suffix, device_activation, activation_bits);
    failures += parent.verify_preserved("gdn parent weight" + suffix);
    return failures;
}

int run_nvfp4_packed_column0(DevicePackedWeight& parent, std::int32_t tokens) {
    constexpr std::int32_t kHidden      = 5120;
    constexpr std::int32_t kQkvRows     = 10240;
    constexpr std::int32_t kZRows       = 6144;
    constexpr std::int32_t kRows        = kQkvRows + kZRows;
    const std::vector<float> activation = make_bf16_activation(kHidden, tokens, 601U + tokens);
    const std::vector<std::uint16_t> activation_bits = bf16_bits(activation);
    DeviceBuffer device_activation                   = to_device(activation_bits);
    GuardedBf16Tensor packed_qkv(kQkvRows, tokens);
    GuardedBf16Tensor packed_z(kZRows, tokens);
    GuardedBf16Tensor decode_qkv(kQkvRows, 1);
    GuardedBf16Tensor decode_z(kZRows, 1);
    Tensor packed_x(device_activation.p, DType::BF16, {kHidden, tokens});
    Tensor decode_x(device_activation.p, DType::BF16, {kHidden, 1});
    Tensor pqkv                       = packed_qkv.tensor();
    Tensor pz                         = packed_z.tensor();
    Tensor dqkv                       = decode_qkv.tensor();
    Tensor dz                         = decode_z.tensor();
    const std::size_t packed_capacity = ops::gdn_input_proj_workspace_capacity_bytes(
        QType::NVFP4, kRows, kHidden, ops::LinearPolicy::AllowA4, tokens, tokens);
    const std::size_t decode_capacity = ops::gdn_input_proj_workspace_capacity_bytes(
        QType::NVFP4, kRows, kHidden, ops::LinearPolicy::A16Only, 1, 1);
    WorkspaceArena packed_workspace(std::max<std::size_t>(packed_capacity, 256));
    WorkspaceArena decode_workspace(std::max<std::size_t>(decode_capacity, 256));
    ops::gdn_input_proj(packed_x, parent.view(), pqkv, pz, ops::LinearPolicy::AllowA4,
                        packed_workspace, nullptr);
    ops::gdn_input_proj(decode_x, parent.view(), dqkv, dz, ops::LinearPolicy::A16Only,
                        decode_workspace, nullptr);
    cuda_synchronize();
    const std::string suffix = std::string(" NVFP4 A4 packed-col0 T=") + std::to_string(tokens);
    int failures             = 0;
    failures += compare_column0("gdn qkv" + suffix, packed_qkv, decode_qkv, kQkvRows,
                                kGdnInputProjA16Tolerance);
    failures +=
        compare_column0("gdn z" + suffix, packed_z, decode_z, kZRows, kGdnInputProjA16Tolerance);
    return failures;
}

int run_nvfp4() {
    constexpr std::int32_t kHidden = 5120;
    constexpr std::int32_t kRows   = 16384;
    quantized_weight::PatternedWeightOptions options;
    options.weight_scale_divisor = 0.125F;
    options.input_scale_divisor  = 3.5F;
    DevicePackedWeight parent(
        quantized_weight::make_patterned_weight(QType::NVFP4, kRows, kHidden, 607U, options));
    int failures = 0;
    // A16Only is registered at T=1 and below the W4A4 cutover.
    failures += run_nvfp4_case(parent, 1, ops::LinearPolicy::A16Only);
    failures += run_nvfp4_case(parent, 2, ops::LinearPolicy::A16Only);
    for (const std::int32_t tokens :
         {1,   2,   3,   4,   5,   6,   8,   10,  12,   15,   18,   24,   36,
          100, 240, 255, 256, 257, 400, 600, 777, 1024, 1025, 1279, 2048, 4096}) {
        failures += run_nvfp4_case(parent, tokens, ops::LinearPolicy::AllowA4);
    }
    for (const std::int32_t tokens : {2}) { failures += run_nvfp4_packed_column0(parent, tokens); }
    return failures;
}

// The [16384,2560] parent admits only A16Only: the T=1 GEMV and the BF16-activation MMA at every
// T>=2, whose wide tiles change at each eight-token panel count 1..8 (T=64) and repeat beyond.
// Distinct parent rows make a misrouted q/k/v/z range visible to the oracle.
int run_nvfp4_2560() {
    constexpr std::int32_t kHidden = 2560;
    constexpr std::int32_t kRows   = 16384;
    DevicePackedWeight parent(
        with_distinct_nvfp4_rows(make_nvfp4_a16_weight(kRows, kHidden, 609U)));
    int failures = 0;
    for (const std::int32_t tokens :
         {1, 2, 3, 8, 9, 16, 17, 24, 25, 32, 33, 40, 41, 48, 49, 56, 57, 64, 65, 300, 1000}) {
        failures += run_nvfp4_case(parent, tokens, ops::LinearPolicy::A16Only);
    }
    if (ops::gdn_input_proj_workspace_capacity_bytes(QType::NVFP4, kRows, kHidden,
                                                     ops::LinearPolicy::A16Only, 1, 4096) != 0) {
        std::cerr << "NVFP4 K=2560 gdn input A16 workspace interval is not zero-capacity\n";
        ++failures;
    }
    failures += expect_invalid_argument(
        [&] {
            return ops::gdn_input_proj_workspace_capacity_bytes(QType::NVFP4, kRows, kHidden,
                                                                ops::LinearPolicy::AllowA4, 1, 1);
        },
        "NVFP4 K=2560 gdn input accepted AllowA4");
    return failures;
}

constexpr ReductionCriterion kFp8GdnInputProjA16Tolerance{1.0 / 256.0, 1.0 / 256.0, 2.0 / 256.0};
constexpr ReductionCriterion kFp8GdnInputProjA8Tolerance{0.04, 1.0 / 256.0, 0.06};
constexpr int kA8SampleRows = 31;

// a16_sample_count bounds the rows an A16 case compares per q/k/v/z range; a count of at least
// 6144, the widest range, compares every row.
int run_fp8_case(DevicePackedWeight& parent, std::int32_t tokens, ops::LinearPolicy policy,
                 bool convenience = false, std::int32_t a16_sample_count = 7) {
    const std::int32_t hidden       = parent.host.weight.k;
    constexpr std::int32_t kQkvRows = 10240;
    constexpr std::int32_t kZRows   = 6144;
    constexpr std::int32_t kRows    = kQkvRows + kZRows;
    const std::vector<float> activation =
        make_bf16_activation(hidden, tokens, 617U + static_cast<std::uint32_t>(tokens));
    const std::vector<std::uint16_t> activation_bits = bf16_bits(activation);
    DeviceBuffer device_activation                   = to_device(activation_bits);
    GuardedBf16Tensor qkv(kQkvRows, tokens);
    GuardedBf16Tensor z(kZRows, tokens);
    Tensor x                   = Tensor(device_activation.p, DType::BF16, {hidden, tokens});
    Tensor qkv_output          = qkv.tensor();
    Tensor z_output            = z.tensor();
    const std::size_t capacity = ops::gdn_input_proj_workspace_capacity_bytes(
        QType::FP8_E4M3FN_ROW_BF16S, kRows, hidden, policy, tokens, tokens);
    WorkspaceArena workspace(std::max<std::size_t>(capacity, 256));
    if (convenience) {
        ops::gdn_input_proj(x, parent.view(), qkv_output, z_output, nullptr);
    } else {
        ops::gdn_input_proj(x, parent.view(), qkv_output, z_output, policy, workspace, nullptr);
    }
    cuda_synchronize();

    const bool a8 = policy == ops::LinearPolicy::AllowA8 && tokens >= 8;
    const ReductionCriterion& criterion =
        a8 ? kFp8GdnInputProjA8Tolerance : kFp8GdnInputProjA16Tolerance;
    const std::int32_t sample_count = a8 ? kA8SampleRows : a16_sample_count;
    const std::string suffix        = std::string(" FP8 K=") + std::to_string(hidden) +
                                      (a8 ? " A8" : " A16") + " T=" + std::to_string(tokens);
    int failures                    = qkv.verify_guards("gdn qkv" + suffix);
    failures += z.verify_guards("gdn z" + suffix);
    failures += qkv.verify_fully_written("gdn qkv" + suffix);
    failures += z.verify_fully_written("gdn z" + suffix);
    failures +=
        verify_output_range_sampled("gdn query" + suffix, qkv, kQkvRows, 0, 2048, parent.host, 0,
                                    activation, hidden, tokens, criterion, sample_count);
    failures +=
        verify_output_range_sampled("gdn key" + suffix, qkv, kQkvRows, 2048, 2048, parent.host,
                                    2048, activation, hidden, tokens, criterion, sample_count);
    failures +=
        verify_output_range_sampled("gdn value" + suffix, qkv, kQkvRows, 4096, 6144, parent.host,
                                    4096, activation, hidden, tokens, criterion, sample_count);
    failures +=
        verify_output_range_sampled("gdn z" + suffix, z, kZRows, 0, kZRows, parent.host, kQkvRows,
                                    activation, hidden, tokens, criterion, sample_count);
    if (workspace.peak_used() != capacity) {
        std::cerr << "gdn workspace" << suffix << ": query/execution high-water mismatch\n";
        ++failures;
    }
    failures += verify_preserved("gdn x" + suffix, device_activation, activation_bits);
    failures += parent.verify_preserved("gdn parent weight" + suffix);
    return failures;
}

int run_fp8() {
    constexpr std::int32_t kHidden = 5120;
    constexpr std::int32_t kRows   = 16384;
    DevicePackedWeight parent(
        quantized_weight::make_patterned_weight(QType::FP8_E4M3FN_ROW_BF16S, kRows, kHidden, 613U));

    int failures = 0;
    for (int columns : {5, 8, 16, 24, 32, 33, 64, 65, 96, 97, 128, 129}) {
        failures += run_fp8_case(parent, columns, ops::LinearPolicy::A16Only);
    }
    const std::size_t one = ops::gdn_input_proj_workspace_capacity_bytes(
        QType::FP8_E4M3FN_ROW_BF16S, kRows, kHidden, ops::LinearPolicy::AllowA8, 1, 1);
    const std::size_t seven = ops::gdn_input_proj_workspace_capacity_bytes(
        QType::FP8_E4M3FN_ROW_BF16S, kRows, kHidden, ops::LinearPolicy::AllowA8, 7, 7);
    const std::size_t eight = ops::gdn_input_proj_workspace_capacity_bytes(
        QType::FP8_E4M3FN_ROW_BF16S, kRows, kHidden, ops::LinearPolicy::AllowA8, 8, 8);
    const std::size_t forty_eight = ops::gdn_input_proj_workspace_capacity_bytes(
        QType::FP8_E4M3FN_ROW_BF16S, kRows, kHidden, ops::LinearPolicy::AllowA8, 48, 48);
    const std::size_t hot_interval = ops::gdn_input_proj_workspace_capacity_bytes(
        QType::FP8_E4M3FN_ROW_BF16S, kRows, kHidden, ops::LinearPolicy::AllowA8, 1, 48);
    const std::size_t exact_1024 = ops::gdn_input_proj_workspace_capacity_bytes(
        QType::FP8_E4M3FN_ROW_BF16S, kRows, kHidden, ops::LinearPolicy::AllowA8, 1024, 1024);
    const std::size_t a16 = ops::gdn_input_proj_workspace_capacity_bytes(
        QType::FP8_E4M3FN_ROW_BF16S, kRows, kHidden, ops::LinearPolicy::A16Only, 1, 2048);
    if (one != 0 || seven != 0 || eight == 0 || forty_eight <= eight ||
        hot_interval != forty_eight || exact_1024 <= forty_eight || a16 != 0) {
        std::cerr << "FP8 gdn input workspace interval contract mismatch\n";
        ++failures;
    }

    failures += run_fp8_case(parent, 1, ops::LinearPolicy::A16Only, true);
    failures += run_fp8_case(parent, 2, ops::LinearPolicy::A16Only);
    for (const std::int32_t tokens : {1, 2, 7, 8, 48, 65, 1024}) {
        failures += run_fp8_case(parent, tokens, ops::LinearPolicy::AllowA8);
    }
    return failures;
}

// The [16384,2560] parent admits only A16Only: the T=1 GEMV, every SIMT width T=2..8, the 16-,
// 24-, and 32-column small-T MMA tiles, and the 64- and 128-column GEMM tiers, each on both sides
// of its boundary. Every parent row is compared, and distinct parent rows make any misrouted row
// range visible to the oracle.
int run_fp8_2560() {
    constexpr std::int32_t kHidden   = 2560;
    constexpr std::int32_t kRows     = 16384;
    constexpr std::int32_t kEveryRow = 6144;
    DevicePackedWeight parent(with_distinct_fp8_rows(quantized_weight::make_patterned_weight(
        QType::FP8_E4M3FN_ROW_BF16S, kRows, kHidden, 619U)));
    int failures = 0;
    for (const std::int32_t tokens :
         {1, 2, 3, 5, 6, 7, 8, 9, 16, 17, 24, 25, 32, 33, 64, 65, 127, 128, 129, 300, 1000}) {
        failures += run_fp8_case(parent, tokens, ops::LinearPolicy::A16Only, false, kEveryRow);
    }
    failures += run_fp8_case(parent, 4, ops::LinearPolicy::A16Only, true, kEveryRow);
    if (ops::gdn_input_proj_workspace_capacity_bytes(QType::FP8_E4M3FN_ROW_BF16S, kRows, kHidden,
                                                     ops::LinearPolicy::A16Only, 1, 4096) != 0) {
        std::cerr << "FP8 K=2560 gdn input A16 workspace interval is not zero-capacity\n";
        ++failures;
    }
    failures += expect_invalid_argument(
        [&] {
            return ops::gdn_input_proj_workspace_capacity_bytes(
                QType::FP8_E4M3FN_ROW_BF16S, kRows, kHidden, ops::LinearPolicy::AllowA8, 1, 1);
        },
        "FP8 K=2560 gdn input workspace accepted AllowA8");
    failures += expect_invalid_argument(
        [&] {
            return ops::gdn_input_proj_workspace_capacity_bytes(
                QType::FP8_E4M3FN_ROW_BF16S, kRows, kHidden, ops::LinearPolicy::AllowA4, 1, 1);
        },
        "FP8 K=2560 gdn input workspace accepted AllowA4");
    failures += expect_invalid_argument(
        [&] {
            const std::vector<std::uint16_t> zeros(static_cast<std::size_t>(kHidden), 0);
            DeviceBuffer device_x = to_device(zeros);
            GuardedBf16Tensor qkv(10240, 1);
            GuardedBf16Tensor z(6144, 1);
            Tensor x(device_x.p, DType::BF16, {kHidden, 1});
            Tensor qkv_output = qkv.tensor();
            Tensor z_output   = z.tensor();
            WorkspaceArena workspace(256);
            ops::gdn_input_proj(x, parent.view(), qkv_output, z_output, ops::LinearPolicy::AllowA8,
                                workspace, nullptr);
            return 0;
        },
        "FP8 K=2560 gdn input accepted AllowA8");
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cerr << "FAIL: no usable CUDA device\n";
        return 1;
    }

    int failures = 0;
    failures += run_fp8();
    failures += run_fp8_2560();
    failures += run_nvfp4();
    failures += run_nvfp4_2560();
    std::cout << (failures == 0 ? "OK" : "FAIL") << " gdn_input_proj\n";
    return failures == 0 ? 0 : 1;
}
