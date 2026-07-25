#include "ops/linear/w8/w8_dispatch.h"
#include "ops/linear_pair/w8/w8_pair_plan.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

using ninfer::ops::LinearPolicy;
using namespace ninfer::ops::detail;

int failures = 0;

template <class Fn>
void expect_invalid(const char* label, Fn&& fn) {
    try {
        fn();
        std::cerr << label << ": expected invalid_argument\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
}

struct LinearCase {
    std::int32_t n;
    std::int32_t k;
    std::int32_t t;
    W8Launch launch;
};

void pure_linear_routes() {
    constexpr std::array cases{
        LinearCase{5120, 10240, 4, launch_w8_simt_r8_c4},
        LinearCase{5120, 10240, 5, launch_w8_simt_r8_c8},
        LinearCase{5120, 10240, 16, launch_w8_simt_r8_c8},
        LinearCase{5120, 10240, 17, launch_w8_mma_r64_c128},
        LinearCase{1024, 5120, 4, launch_w8_simt_r8_c4},
        LinearCase{1024, 5120, 5, launch_w8_simt_r8_c8},
        LinearCase{1024, 5120, 17, launch_w8_mma_r32_c128},
        LinearCase{6144, 5120, 17, launch_w8_mma_r64_c128},
        LinearCase{14336, 5120, 8, launch_w8_simt_r8_c8},
        LinearCase{14336, 5120, 9, launch_w8_mma_r64_c128},
        LinearCase{34816, 5120, 8, launch_w8_simt_r8_c8},
        LinearCase{34816, 5120, 9, launch_w8_mma_r64_c128},
        LinearCase{5120, 6144, 16, launch_w8_simt_r8_c8},
        LinearCase{5120, 6144, 17, launch_w8_mma_r64_c128},
        LinearCase{5120, 17408, 16, launch_w8_simt_r8_c8},
        LinearCase{5120, 17408, 17, launch_w8_mma_r64_c128},
        LinearCase{2048, 4096, 56, launch_w8_simt_r8_c4},
        LinearCase{2048, 4096, 57, launch_w8_mma_r32_c128},
        LinearCase{2048, 4096, 895, launch_w8_mma_r32_c128},
        LinearCase{2048, 4096, 896, launch_w8_mma_r64_c128},
        LinearCase{1024, 2048, 16, launch_w8_simt_r8_c8},
        LinearCase{1024, 2048, 17, launch_w8_mma_r32_c128},
        LinearCase{9216, 2048, 13, launch_w8_simt_r8_c4},
        LinearCase{9216, 2048, 14, launch_w8_mma_r32_c128},
        LinearCase{9216, 2048, 128, launch_w8_mma_r32_c128},
        LinearCase{9216, 2048, 129, launch_w8_mma_r64_c128},
        LinearCase{12288, 2048, 16, launch_w8_simt_r8_c4},
        LinearCase{12288, 2048, 17, launch_w8_mma_r64_c128},
        LinearCase{2048, 4608, 14, launch_w8_simt_r8_c4},
        LinearCase{2048, 4608, 15, launch_w8_mma_r32_c128},
        LinearCase{2048, 4608, 16, launch_w8_simt_r8_c4},
        LinearCase{2048, 4608, 17, launch_w8_mma_r32_c128},
        LinearCase{2048, 4608, 32, launch_w8_simt_r8_c4},
        LinearCase{2048, 4608, 33, launch_w8_mma_r32_c128},
        LinearCase{2048, 4608, 871, launch_w8_mma_r32_c128},
        LinearCase{2048, 4608, 872, launch_w8_mma_r64_c128},
        LinearCase{4608, 4608, 8, launch_w8_simt_r8_c4},
        LinearCase{4608, 4608, 9, launch_w8_mma_r32_c128},
        LinearCase{4608, 4608, 12, launch_w8_simt_r8_c4},
        LinearCase{4608, 4608, 13, launch_w8_mma_r32_c128},
        LinearCase{4608, 4608, 256, launch_w8_mma_r32_c128},
        LinearCase{4608, 4608, 257, launch_w8_mma_r64_c128},
        LinearCase{5120, 4608, 4, launch_w8_simt_r8_c4},
        LinearCase{5120, 4608, 5, launch_w8_simt_r8_c8},
        LinearCase{5120, 4608, 6, launch_w8_mma_r64_c128},
        LinearCase{2048, 16384, 1, launch_w8_decode_r4},
        LinearCase{2048, 16384, 2, launch_w8_exact_t_splitk},
        LinearCase{2048, 16384, 32, launch_w8_exact_t_splitk},
        LinearCase{2048, 16384, 33, launch_w8_exact_t_composite},
        LinearCase{2048, 16384, 88, launch_w8_exact_t_composite},
        LinearCase{2048, 16384, 89, launch_w8_medium_splitk_c96},
        LinearCase{2048, 16384, 97, launch_w8_medium_splitk_c128},
        LinearCase{2048, 16384, 129, launch_w8_medium_splitk_c144},
        LinearCase{2048, 16384, 145, launch_w8_mma_r32_c128},
        LinearCase{2048, 16384, 256, launch_w8_mma_r32_c64},
        LinearCase{2048, 16384, 385, launch_w8_mma_r32_c96},
        LinearCase{2048, 16384, 481, launch_w8_exact_mma_r32_c96},
        LinearCase{2048, 16384, 482, launch_w8_mma_r32_c128},
        LinearCase{2048, 16384, 641, launch_w8_exact_mma_r32_c128},
        LinearCase{2048, 16384, 669, launch_w8_mma_r48_c96},
        LinearCase{2048, 16384, 673, launch_w8_exact_mma_r48_c96},
        LinearCase{2048, 16384, 674, launch_w8_mma_r48_c64},
        LinearCase{2048, 16384, 705, launch_w8_mma_r48_c112},
        LinearCase{2048, 16384, 785, launch_w8_mma_r48_c128},
        LinearCase{2048, 16384, 897, launch_w8_exact_mma_r48_c128},
        LinearCase{2048, 16384, 913, launch_w8_mma_r64_c96},
        LinearCase{2048, 16384, 961, launch_w8_exact_mma_r64_c96},
        LinearCase{2048, 16384, 1008, launch_w8_mma_r64_c112},
        LinearCase{2048, 16384, 1009, launch_w8_mma_r64_c128},
        LinearCase{2048, 16384, 1120, launch_w8_mma_r64_c112},
        LinearCase{2048, 16384, 1121, launch_w8_mma_r64_c128},
        LinearCase{2048, 16384, 1281, launch_w8_exact_mma_r64_c128},
        LinearCase{2048, 16384, 1314, launch_w8_mma_r128_c64},
        LinearCase{2048, 16384, 1345, launch_w8_mma_r96_c96},
        LinearCase{2048, 16384, 1441, launch_w8_exact_mma_r96_c96},
        LinearCase{2048, 16384, 1501, launch_w8_mma_r128_c80},
        LinearCase{2048, 16384, 1681, launch_w8_exact_mma_r128_c80},
        LinearCase{2048, 16384, 1746, launch_w8_mma_r48_c128},
        LinearCase{2048, 16384, 1792, launch_w8_mma_r64_c128},
        LinearCase{2048, 16384, 1793, launch_w8_mma_r48_c128},
        LinearCase{2048, 16384, 1920, launch_w8_mma_r64_c128},
        LinearCase{2048, 16384, 1921, launch_w8_exact_mma_r64_c128},
        LinearCase{2048, 16384, 1954, launch_w8_mma_r64_c96},
        LinearCase{2048, 16384, 2017, launch_w8_exact_mma_r64_c96},
        LinearCase{2048, 16384, 2049, launch_w8_mma_r96_c96},
        LinearCase{2048, 16384, 2113, launch_w8_mma_r64_c128},
    };

    for (const LinearCase& test : cases) {
        const W8Launch direct = select_w8_a16_launch(test.n, test.k, test.t);
        const W8Launch a16    = select_w8_launch(test.n, test.k, test.t, LinearPolicy::A16Only);
        const W8Launch a8     = select_w8_launch(test.n, test.k, test.t, LinearPolicy::AllowA8);
        if (direct != test.launch || a16 != test.launch || a8 != test.launch) {
            std::cerr << "wrong W8 route [" << test.n << ',' << test.k << "] T=" << test.t << '\n';
            ++failures;
        }
    }

    expect_invalid("T=0", [] { (void)select_w8_a16_launch(5120, 10240, 0); });
    expect_invalid("unsupported shape", [] { (void)select_w8_a16_launch(5121, 10240, 1); });
    expect_invalid("vision T limit", [] { (void)select_w8_a16_launch(2048, 4608, 32769); });
    expect_invalid("AllowA4",
                   [] { (void)select_w8_launch(5120, 10240, 1, LinearPolicy::AllowA4); });
}

struct PairRange {
    std::int32_t first;
    std::int32_t last;
    W8PairScheduleId schedule;
};

void pair_routes() {
    constexpr std::array k5120{
        PairRange{1, 4, W8PairScheduleId::TwoSimtR8C4},
        PairRange{5, 56, W8PairScheduleId::TwoSimtR8C8},
        PairRange{57, std::numeric_limits<std::int32_t>::max(), W8PairScheduleId::DualMmaR32C128},
    };
    constexpr std::array k2048{
        PairRange{1, 1, W8PairScheduleId::DualDecodeR4},
        PairRange{2, 32, W8PairScheduleId::DualSplitKMmaExactT},
        PairRange{33, 48, W8PairScheduleId::DualSplitKMediumC48},
        PairRange{49, 64, W8PairScheduleId::DualSplitKMediumC64},
        PairRange{65, 80, W8PairScheduleId::DualSplitKMediumC80},
        PairRange{81, 88, W8PairScheduleId::DualSplitKMediumC88},
        PairRange{89, 96, W8PairScheduleId::DualSplitKMediumC96},
        PairRange{97, 104, W8PairScheduleId::DualSplitKMediumC104},
        PairRange{105, 112, W8PairScheduleId::DualSplitKMediumC112},
        PairRange{113, 128, W8PairScheduleId::DualSplitKMediumC128},
        PairRange{129, 160, W8PairScheduleId::DualSplitKMediumC160},
        PairRange{161, 192, W8PairScheduleId::DualSplitKMediumC192},
        PairRange{193, 384, W8PairScheduleId::ConcatMmaR32C64},
        PairRange{385, 480, W8PairScheduleId::ConcatMmaR32C96},
        PairRange{481, 640, W8PairScheduleId::ConcatMmaR32C128},
        PairRange{641, 641, W8PairScheduleId::ExactConcatMmaR32C128},
        PairRange{642, 672, W8PairScheduleId::ConcatMmaR48C96},
        PairRange{673, 680, W8PairScheduleId::ExactConcatMmaR32C96},
        PairRange{681, 784, W8PairScheduleId::ConcatMmaR48C112},
        PairRange{785, 896, W8PairScheduleId::ConcatMmaR48C128},
        PairRange{897, 960, W8PairScheduleId::ConcatMmaR96C64},
        PairRange{961, 976, W8PairScheduleId::ExactConcatMmaR64C96},
        PairRange{977, 1280, W8PairScheduleId::ConcatMmaR64C128},
        PairRange{1281, 1316, W8PairScheduleId::ExactConcatMmaR64C128},
        PairRange{1317, 1344, W8PairScheduleId::ConcatMmaR128C64},
        PairRange{1345, 1345, W8PairScheduleId::ExactConcatMmaR128C64},
        PairRange{1346, 1440, W8PairScheduleId::ConcatMmaR96C96},
        PairRange{1441, 1466, W8PairScheduleId::ExactConcatMmaR96C96},
        PairRange{1467, 1680, W8PairScheduleId::ConcatMmaR128C80},
        PairRange{1681, 1708, W8PairScheduleId::ExactConcatMmaR128C80},
        PairRange{1709, 1920, W8PairScheduleId::ConcatMmaR48C128},
        PairRange{1921, 1922, W8PairScheduleId::ExactConcatMmaR64C128},
        PairRange{1923, 2016, W8PairScheduleId::ConcatMmaR64C96},
        PairRange{2017, 2018, W8PairScheduleId::ExactConcatMmaR64C96},
        PairRange{2019, 2208, W8PairScheduleId::ConcatMmaR96C96},
        PairRange{2209, 2270, W8PairScheduleId::ExactConcatMmaR96C96},
        PairRange{2271, std::numeric_limits<std::int32_t>::max(),
                  W8PairScheduleId::ConcatMmaR64C128},
    };
    const auto check = [](std::int32_t k, const auto& ranges) {
        for (const PairRange& range : ranges) {
            for (const std::int32_t t : {range.first, range.last}) {
                const W8PairPlan plan = w8_pair_resolve_plan({1024, k, k, t});
                if (plan.schedule != range.schedule || plan.workspace_bytes != 0) {
                    std::cerr << "wrong W8 Pair route K=" << k << " T=" << t << '\n';
                    ++failures;
                }
            }
        }
    };
    check(5120, k5120);
    check(2048, k2048);

    expect_invalid("pair T=0", [] { (void)w8_pair_resolve_plan({1024, 2048, 2048, 0}); });
    expect_invalid("pair unsupported K", [] { (void)w8_pair_resolve_plan({1024, 4096, 4096, 1}); });
    expect_invalid("pair padded K", [] { (void)w8_pair_resolve_plan({1024, 2048, 2112, 1}); });
}

} // namespace

int main() {
    pure_linear_routes();
    pair_routes();
    std::cout << (failures == 0 ? "OK" : "FAIL") << " W8 selectors\n";
    return failures == 0 ? 0 : 1;
}
