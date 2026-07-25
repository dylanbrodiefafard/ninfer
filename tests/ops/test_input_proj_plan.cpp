#include "ninfer/ops/gdn_input_proj.h"
#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_plan.h"
#include "ops/attn_input_proj/w8/w8_attn_input_plan.h"
#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.h"
#include "ops/gdn_input_proj/w8/w8_gdn_input_plan.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {

using ninfer::ops::detail::Q4Q5AttnInputProblem;
using ninfer::ops::detail::Q4Q5AttnInputScheduleId;
using ninfer::ops::detail::Q4Q5GdnInputProblem;
using ninfer::ops::detail::Q4Q5GdnInputScheduleId;
using ninfer::ops::detail::W8AttnInputProblem;
using ninfer::ops::detail::W8AttnInputScheduleId;
using ninfer::ops::detail::W8GdnInputProblem;
using ninfer::ops::detail::W8GdnInputScheduleId;
using ninfer::ops::detail::W8GdnInputSnapshotScheduleId;

int failures = 0;

template <class Fn>
void expect_invalid(const char* label, Fn&& fn) {
    try {
        fn();
        std::cerr << label << ": expected invalid_argument\n";
        ++failures;
    } catch (const std::invalid_argument&) {}
}

void attn_route_tests() {
    constexpr std::array<std::int32_t, 10> boundaries{
        1, 2, 16, 17, 127, 128, 129, 1024, 1025, 2048,
    };
    for (const std::int32_t cols : boundaries) {
        const Q4Q5AttnInputProblem problem{5120, 6144, 1024, 5120, cols};
        const auto plan         = ninfer::ops::detail::q4_q5_attn_input_resolve_plan(problem);
        const bool parent_split = cols <= 16;
        const auto expected     = parent_split
                                      ? Q4Q5AttnInputScheduleId::ParentSplitFixed
                                      : Q4Q5AttnInputScheduleId::GroupedHomogeneousPairMmaR64C128;
        if (plan.schedule != expected || plan.workspace_bytes != 0) {
            std::cerr << "wrong attention input route C=" << cols << '\n';
            ++failures;
        }
    }
    expect_invalid("attention C0", [] {
        (void)ninfer::ops::detail::q4_q5_attn_input_resolve_plan({5120, 6144, 1024, 5120, 0});
    });
    expect_invalid("attention unsupported shape", [] {
        (void)ninfer::ops::detail::q4_q5_attn_input_resolve_plan({5120, 6144, 2048, 5120, 1});
    });
}

void gdn_route_tests() {
    constexpr std::array<std::int32_t, 10> boundaries{
        1, 2, 16, 17, 127, 128, 129, 1024, 1025, 2048,
    };
    for (const std::int32_t cols : boundaries) {
        const Q4Q5GdnInputProblem problem{5120, 4096, 6144, 10240, 5120, cols};
        const auto plan        = ninfer::ops::detail::q4_q5_gdn_input_resolve_plan(problem);
        const bool independent = cols <= 16;
        const auto expected    = independent ? Q4Q5GdnInputScheduleId::IndependentDirectFixed
                                             : Q4Q5GdnInputScheduleId::GroupedMixedMmaR64C128;
        if (plan.schedule != expected || plan.workspace_bytes != 0) {
            std::cerr << "wrong GDN input route C=" << cols << '\n';
            ++failures;
        }
    }
    expect_invalid("GDN C0", [] {
        (void)ninfer::ops::detail::q4_q5_gdn_input_resolve_plan({5120, 4096, 6144, 10240, 5120, 0});
    });
    expect_invalid("GDN unsupported shape", [] {
        (void)ninfer::ops::detail::q4_q5_gdn_input_resolve_plan({5120, 4096, 6144, 10241, 5120, 1});
    });
}

void w8_attn_route_tests() {
    struct Case {
        std::int32_t cols;
        W8AttnInputScheduleId schedule;
    };

    constexpr std::array<Case, 23> target_cases{{
        {1, W8AttnInputScheduleId::DecodeR8Direct},
        {2, W8AttnInputScheduleId::SplitKMmaDirect},
        {3, W8AttnInputScheduleId::SplitKMmaDirect},
        {4, W8AttnInputScheduleId::SplitKMmaDirect},
        {5, W8AttnInputScheduleId::SplitKMmaDirect},
        {6, W8AttnInputScheduleId::SplitKMmaDirect},
        {7, W8AttnInputScheduleId::SplitKMmaDirect},
        {8, W8AttnInputScheduleId::SplitKMmaDirect},
        {9, W8AttnInputScheduleId::SplitKMmaDirect},
        {10, W8AttnInputScheduleId::SplitKMmaDirect},
        {11, W8AttnInputScheduleId::SplitKMmaDirect},
        {12, W8AttnInputScheduleId::SplitKMmaDirect},
        {13, W8AttnInputScheduleId::SplitKMmaDirect},
        {14, W8AttnInputScheduleId::SplitKMmaDirect},
        {15, W8AttnInputScheduleId::SplitKMmaDirect},
        {16, W8AttnInputScheduleId::SplitKMmaDirect},
        {17, W8AttnInputScheduleId::MmaR32C128},
        {127, W8AttnInputScheduleId::MmaR32C128},
        {128, W8AttnInputScheduleId::MmaR32C128},
        {129, W8AttnInputScheduleId::MmaR64C128},
        {256, W8AttnInputScheduleId::MmaR64C128},
        {1024, W8AttnInputScheduleId::MmaR64C128},
        {2048, W8AttnInputScheduleId::MmaR64C128},
    }};
    constexpr std::array<Case, 23> companion_cases{{
        {1, W8AttnInputScheduleId::DecodeR8Direct},
        {2, W8AttnInputScheduleId::SplitKMmaDirect},
        {16, W8AttnInputScheduleId::SplitKMmaDirect},
        {32, W8AttnInputScheduleId::SplitKMmaDirect},
        {96, W8AttnInputScheduleId::SplitKMmaDirect},
        {97, W8AttnInputScheduleId::MmaR32C64},
        {192, W8AttnInputScheduleId::MmaR32C64},
        {193, W8AttnInputScheduleId::MmaR64C96},
        {288, W8AttnInputScheduleId::MmaR64C96},
        {289, W8AttnInputScheduleId::MmaR64C64},
        {320, W8AttnInputScheduleId::MmaR64C64},
        {321, W8AttnInputScheduleId::MmaR64C128},
        {384, W8AttnInputScheduleId::MmaR64C128},
        {385, W8AttnInputScheduleId::MmaR128C64},
        {448, W8AttnInputScheduleId::MmaR128C64},
        {449, W8AttnInputScheduleId::MmaR128C80},
        {480, W8AttnInputScheduleId::MmaR128C80},
        {481, W8AttnInputScheduleId::MmaR128C80},
        {512, W8AttnInputScheduleId::MmaR128C80},
        {560, W8AttnInputScheduleId::MmaR128C80},
        {561, W8AttnInputScheduleId::MmaR64C128},
        {1024, W8AttnInputScheduleId::MmaR64C128},
        {2048, W8AttnInputScheduleId::MmaR64C128},
    }};
    const auto check_cases = [&](const auto& cases, W8AttnInputProblem problem) {
        for (const Case test : cases) {
            problem.cols    = test.cols;
            const auto plan = ninfer::ops::detail::w8_attn_input_resolve_plan(problem);
            if (plan.schedule != test.schedule || plan.workspace_bytes != 0) {
                std::cerr << "wrong W8 attention input route rows=" << problem.parent_rows
                          << " C=" << test.cols << '\n';
                ++failures;
            }
        }
    };
    check_cases(target_cases, {2048, 4096, 512, 9216, 2048, 1});
    check_cases(companion_cases, {2048, 4096, 1024, 6144, 2048, 1});
    expect_invalid("W8 attention C0", [] {
        (void)ninfer::ops::detail::w8_attn_input_resolve_plan({2048, 4096, 512, 9216, 2048, 0});
    });
    expect_invalid("W8 attention unsupported shape", [] {
        (void)ninfer::ops::detail::w8_attn_input_resolve_plan({2048, 4096, 512, 9217, 2048, 1});
    });
    expect_invalid("W8 companion attention mixed geometry", [] {
        (void)ninfer::ops::detail::w8_attn_input_resolve_plan({2048, 4096, 1024, 9216, 2048, 1});
    });
}

void w8_gdn_route_tests() {
    struct Case {
        std::int32_t cols;
        W8GdnInputScheduleId schedule;
    };

    constexpr std::array<Case, 17> cases{{
        {1, W8GdnInputScheduleId::DecodeR8Direct},
        {2, W8GdnInputScheduleId::SplitKMmaDirect},
        {4, W8GdnInputScheduleId::SplitKMmaDirect},
        {16, W8GdnInputScheduleId::SplitKMmaDirect},
        {17, W8GdnInputScheduleId::SplitKMmaDirect},
        {32, W8GdnInputScheduleId::SplitKMmaDirect},
        {33, W8GdnInputScheduleId::SplitKMmaDirect},
        {48, W8GdnInputScheduleId::SplitKMmaDirect},
        {49, W8GdnInputScheduleId::SplitKMmaDirect},
        {64, W8GdnInputScheduleId::SplitKMmaDirect},
        {65, W8GdnInputScheduleId::SplitKMmaDirect},
        {96, W8GdnInputScheduleId::SplitKMmaDirect},
        {97, W8GdnInputScheduleId::MmaR64C128},
        {127, W8GdnInputScheduleId::MmaR64C128},
        {128, W8GdnInputScheduleId::MmaR64C128},
        {129, W8GdnInputScheduleId::MmaR64C128},
        {1024, W8GdnInputScheduleId::MmaR64C128},
    }};
    for (const Case test : cases) {
        const W8GdnInputProblem problem{2048, 8192, 4096, 12288, 2048, test.cols};
        const auto plan = ninfer::ops::detail::w8_gdn_input_resolve_plan(problem);
        if (plan.schedule != test.schedule || plan.workspace_bytes != 0) {
            std::cerr << "wrong W8 GDN input route C=" << test.cols << '\n';
            ++failures;
        }
    }
    for (const std::int32_t cols : {1, 2, 6, 7, 16, 17, 128}) {
        const auto plan = ninfer::ops::detail::w8_gdn_input_snapshot_resolve_plan(
            {2048, 8192, 4096, 12288, 2048, cols});
        const W8GdnInputSnapshotScheduleId expected =
            cols == 1    ? W8GdnInputSnapshotScheduleId::DecodeFused
            : cols <= 16 ? W8GdnInputSnapshotScheduleId::SplitKMmaFused
                         : W8GdnInputSnapshotScheduleId::Composed;
        if (plan.schedule != expected) {
            std::cerr << "wrong W8 GDN snapshot route C=" << cols << '\n';
            ++failures;
        }
    }
    expect_invalid("W8 GDN C0", [] {
        (void)ninfer::ops::detail::w8_gdn_input_resolve_plan({2048, 8192, 4096, 12288, 2048, 0});
    });
    expect_invalid("W8 GDN unsupported shape", [] {
        (void)ninfer::ops::detail::w8_gdn_input_resolve_plan({2048, 8192, 4095, 12288, 2048, 1});
    });
}

void workspace_tests() {
    struct Case {
        std::int32_t capacity;
        std::size_t bytes;
    };

    constexpr std::array<Case, 8> cases{{
        {1, 0},
        {2, 0},
        {16, 0},
        {17, 0},
        {128, 0},
        {1024, 0},
        {1025, 0},
        {2048, 0},
    }};
    for (const Case test : cases) {
        const std::size_t actual =
            ninfer::ops::gdn_input_proj_workspace_bytes(4096, 6144, test.capacity);
        if (actual != test.bytes) {
            std::cerr << "GDN input workspace C=" << test.capacity << ": expected " << test.bytes
                      << ", got " << actual << '\n';
            ++failures;
        }
    }
    for (const Case test : cases) {
        const std::size_t actual =
            ninfer::ops::gdn_input_proj_workspace_bytes(8192, 4096, test.capacity);
        if (actual != test.bytes) {
            std::cerr << "W8 GDN input workspace C=" << test.capacity << ": expected " << test.bytes
                      << ", got " << actual << '\n';
            ++failures;
        }
    }
    const std::size_t w8_t16 =
        ninfer::ops::gdn_input_proj_conv_snapshot_workspace_bytes(2048, 2048, 4096, 16);
    const std::size_t w8_t17 =
        ninfer::ops::gdn_input_proj_conv_snapshot_workspace_bytes(2048, 2048, 4096, 17);
    if (w8_t16 != 0 || w8_t17 != 2ULL * 8192 * 17 * sizeof(std::uint16_t)) {
        std::cerr << "wrong W8 GDN snapshot workspace frontier\n";
        ++failures;
    }
    expect_invalid("workspace unsupported rows",
                   [] { (void)ninfer::ops::gdn_input_proj_workspace_bytes(4095, 6144, 1); });
    expect_invalid("workspace C0",
                   [] { (void)ninfer::ops::gdn_input_proj_workspace_bytes(4096, 6144, 0); });
}

} // namespace

int main() {
    attn_route_tests();
    gdn_route_tests();
    w8_attn_route_tests();
    w8_gdn_route_tests();
    workspace_tests();
    std::cout << (failures == 0 ? "OK" : "FAIL") << " input projection plans\n";
    return failures == 0 ? 0 : 1;
}
