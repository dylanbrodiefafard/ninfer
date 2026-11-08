#include "ops/linear_pair/w8/w8_pair_plan.h"

#include "ops/linear_pair/w8/w8_pair_kernels.h"
#include "ops/common/token_slices.h"

#include <array>
#include <cstdint>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kAnyCols = std::numeric_limits<std::int32_t>::max();

struct W8PairRouteSpec {
    std::int32_t first;
    std::int32_t last;
    W8PairScheduleId schedule;
};

constexpr std::array<W8PairRouteSpec, 3> kK5120Routes{{
    {1, 4, W8PairScheduleId::TwoSimtR8C4},
    {5, 56, W8PairScheduleId::TwoSimtR8C8},
    {57, kAnyCols, W8PairScheduleId::DualMmaR32C128},
}};

template <std::size_t N>
constexpr bool routes_are_closed(const std::array<W8PairRouteSpec, N>& routes) noexcept {
    std::int64_t expected = 1;
    for (const W8PairRouteSpec& route : routes) {
        if (route.first != expected || route.last < route.first) { return false; }
        expected = static_cast<std::int64_t>(route.last) + 1;
    }
    return routes.back().last == kAnyCols && expected == static_cast<std::int64_t>(kAnyCols) + 1;
}

static_assert(routes_are_closed(kK5120Routes),
              "W8 pair routes must be exact, contiguous, and closed");

bool uses_mma(W8PairScheduleId schedule) noexcept {
    return schedule == W8PairScheduleId::DualMmaR32C128;
}

std::int32_t schedule_rows(W8PairScheduleId schedule) {
    switch (schedule) {
    case W8PairScheduleId::TwoSimtR8C4:
    case W8PairScheduleId::TwoSimtR8C8:
        return 8;
    case W8PairScheduleId::DualMmaR32C128:
        return 32;
    }
    throw std::logic_error("w8 pair: unknown schedule");
}

std::int32_t schedule_cols(W8PairScheduleId schedule) {
    switch (schedule) {
    case W8PairScheduleId::TwoSimtR8C4:
        return 4;
    case W8PairScheduleId::TwoSimtR8C8:
        return 8;
    case W8PairScheduleId::DualMmaR32C128:
        return 128;
    }
    throw std::logic_error("w8 pair: unknown schedule");
}

void require_pair_weights(const Weight& first_weight, const Weight& second_weight, std::int32_t k) {
    const auto valid = [k](const Weight& w) {
        const std::uint64_t groups = static_cast<std::uint64_t>(k / 32);
        const std::uint64_t payload_bytes =
            static_cast<std::uint64_t>(1024) * k + static_cast<std::uint64_t>(1024) * groups * 2;
        return w.qtype == QType::W8G32_F16S && w.layout == QuantLayout::RowSplit &&
               w.scale_dtype == DType::FP16 && w.group == 32 && w.group_size == 32 && w.ndim == 2 &&
               w.n == 1024 && w.k == k && w.shape[0] == 1024 && w.shape[1] == k &&
               w.padded_shape[0] == 1024 && w.padded_shape[1] == k &&
               w.payload_bytes >= payload_bytes && w.qdata != nullptr && w.qhigh == nullptr &&
               w.high_plane_bytes == 0 && w.scales != nullptr;
    };
    if (!valid(first_weight) || !valid(second_weight) || first_weight.n != second_weight.n ||
        first_weight.k != second_weight.k ||
        first_weight.padded_shape[1] != second_weight.padded_shape[1]) {
        throw std::invalid_argument("w8 pair: weights must be matching W8G32 RowSplit matrices");
    }
}

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void require_pair_operands(const Tensor& x, const Weight& first_weight, const Weight& second_weight,
                           const Tensor& first_out, const Tensor& second_out,
                           bool require_scale_16) {
    if (!aligned_to(x.data, 16) || !aligned_to(first_out.data, 16) ||
        !aligned_to(second_out.data, 16) || !aligned_to(first_weight.qdata, 16) ||
        !aligned_to(second_weight.qdata, 16) || !aligned_to(first_weight.scales, 4) ||
        !aligned_to(second_weight.scales, 4)) {
        throw std::invalid_argument(
            "w8 pair: requires 16-byte x/out/code and 4-byte scale alignment");
    }
    if (require_scale_16 &&
        (!aligned_to(first_weight.scales, 16) || !aligned_to(second_weight.scales, 16))) {
        throw std::invalid_argument("w8 pair MMA: scale planes must be 16-byte aligned");
    }
}

} // namespace

const char* w8_pair_schedule_name(W8PairScheduleId schedule) {
    switch (schedule) {
    case W8PairScheduleId::TwoSimtR8C4:
        return "w8_pair.two_simt.r8.c4";
    case W8PairScheduleId::TwoSimtR8C8:
        return "w8_pair.two_simt.r8.c8";
    case W8PairScheduleId::DualMmaR32C128:
        return "w8_pair.dual_mma.r32.c128";
    }
    return "w8_pair.unknown";
}

W8PairProblem w8_pair_problem(const Tensor& x, const Weight& first_weight,
                              const Tensor& first_out) noexcept {
    return {first_out.ne[0], x.ne[0], first_weight.padded_shape[1], x.ne[1]};
}

bool w8_pair_admits(const W8PairProblem& problem) noexcept {
    return problem.rows == 1024 && problem.k == 5120 && problem.padded_k == problem.k &&
           problem.cols >= 1;
}

W8PairPlan w8_pair_resolve_plan(const W8PairProblem& problem) {
    if (!w8_pair_admits(problem)) {
        throw std::invalid_argument("w8 pair: exact problem or column count is not admitted");
    }
    for (const W8PairRouteSpec& route : kK5120Routes) {
        if (problem.cols >= route.first && problem.cols <= route.last) { return {route.schedule}; }
    }
    throw std::logic_error("w8 pair: admitted problem has no covering route");
}

namespace {

bool tiled_use_full(W8PairScheduleId schedule, const W8PairProblem& problem) {
    const bool tile_aligned = (problem.rows % schedule_rows(schedule)) == 0 &&
                              (problem.cols % schedule_cols(schedule)) == 0;
    if (schedule == W8PairScheduleId::TwoSimtR8C4 || schedule == W8PairScheduleId::TwoSimtR8C8) {
        return tile_aligned;
    }
    return tile_aligned && problem.k == problem.padded_k && (problem.k % 64) == 0;
}

void launch_tiled(W8PairScheduleId schedule, bool full, const Tensor& x, const Weight& first_weight,
                  const Weight& second_weight, Tensor& first_out, Tensor& second_out,
                  cudaStream_t stream) {
    const std::int32_t tile_cols = schedule_cols(schedule);
    for_each_token_slice(x.ne[1], tile_cols, [&](std::int32_t offset, std::int32_t count) {
        const Tensor x_slice = x.slice(1, offset, count);
        Tensor first_slice   = first_out.slice(1, offset, count);
        Tensor second_slice  = second_out.slice(1, offset, count);
        switch (schedule) {
        case W8PairScheduleId::TwoSimtR8C4:
            w8_pair_simt_r8_c4_launch(full, x_slice, first_weight, second_weight, first_slice,
                                      second_slice, stream);
            return;
        case W8PairScheduleId::TwoSimtR8C8:
            w8_pair_simt_r8_c8_launch(full, x_slice, first_weight, second_weight, first_slice,
                                      second_slice, stream);
            return;
        case W8PairScheduleId::DualMmaR32C128:
            w8_pair_gemm_mma_launch(full, x_slice, first_weight, second_weight, first_slice,
                                    second_slice, stream);
            return;
        }
        throw std::logic_error("w8 pair: unknown schedule");
    });
}

} // namespace

void w8_pair_execute_plan(W8PairPlan plan, const Tensor& x, const Weight& first_weight,
                          const Weight& second_weight, Tensor& first_out, Tensor& second_out,
                          cudaStream_t stream) {
    const W8PairProblem problem = w8_pair_problem(x, first_weight, first_out);
    const W8PairPlan resolved   = w8_pair_resolve_plan(problem);
    if (resolved.schedule != plan.schedule) {
        throw std::invalid_argument("w8 pair: plan does not match the exact problem");
    }

    require_pair_weights(first_weight, second_weight, x.ne[0]);
    require_pair_operands(x, first_weight, second_weight, first_out, second_out,
                          uses_mma(plan.schedule));

    launch_tiled(plan.schedule, tiled_use_full(plan.schedule, problem), x, first_weight,
                 second_weight, first_out, second_out, stream);
}

void w8_pair_dispatch(const Tensor& x, const Weight& first_weight, const Weight& second_weight,
                      Tensor& first_out, Tensor& second_out, cudaStream_t stream) {
    const W8PairPlan plan = w8_pair_resolve_plan(w8_pair_problem(x, first_weight, first_out));
    w8_pair_execute_plan(plan, x, first_weight, second_weight, first_out, second_out, stream);
}

} // namespace ninfer::ops::detail
