#include "ops/linear_add/q5/q5_linear_add_plan.h"

#include "ninfer/ops/linear.h"
#include "ninfer/ops/residual_add.h"
#include "ops/linear_add/q5/q5_linear_add_kernels.h"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kAnyCols = std::numeric_limits<std::int32_t>::max();

struct ColsSet {
    std::int32_t first;
    std::int32_t last;

    constexpr bool contains(std::int32_t cols) const noexcept {
        return cols >= first && cols <= last;
    }
};

struct SupportSpec {
    std::int32_t rows;
    std::int32_t k;
    std::int32_t padded_k;
};

struct RouteSpec {
    ColsSet cols;
    Q5LinearAddScheduleId schedule;
};

constexpr std::array<SupportSpec, 2> kSupports{{
    {5120, 6144, 6144},
    {5120, 17408, 17408},
}};

constexpr std::array<RouteSpec, 4> kRoutes{{
    {{1, 1}, Q5LinearAddScheduleId::GemvResidual},
    {{2, 24}, Q5LinearAddScheduleId::Materialized},
    {{25, 128}, Q5LinearAddScheduleId::MmaResidualR64C64},
    {{129, kAnyCols}, Q5LinearAddScheduleId::MmaResidualR64C128},
}};

constexpr bool catalog_is_closed() noexcept {
    std::int64_t expected = 1;
    for (const RouteSpec& route : kRoutes) {
        if (route.cols.first != expected || route.cols.last < route.cols.first) { return false; }
        expected = static_cast<std::int64_t>(route.cols.last) + 1;
    }
    return kRoutes.back().cols.last == kAnyCols &&
           expected == static_cast<std::int64_t>(kAnyCols) + 1;
}

static_assert(catalog_is_closed(), "Q5 LinearAdd routes must be exact, contiguous, and closed");

bool supported_shape(const Q5LinearAddProblem& problem) noexcept {
    for (const SupportSpec& support : kSupports) {
        if (problem.rows == support.rows && problem.k == support.k &&
            problem.padded_k == support.padded_k) {
            return true;
        }
    }
    return false;
}

std::size_t checked_matrix_bytes(std::int32_t rows, std::int32_t cols) {
    const std::size_t r = static_cast<std::size_t>(rows);
    const std::size_t c = static_cast<std::size_t>(cols);
    if (c != 0 && r > std::numeric_limits<std::size_t>::max() / c) {
        throw std::overflow_error("q5 linear_add: workspace element count overflows size_t");
    }
    const std::size_t elements = r * c;
    if (elements > std::numeric_limits<std::size_t>::max() / sizeof(std::uint16_t)) {
        throw std::overflow_error("q5 linear_add: workspace byte count overflows size_t");
    }
    return elements * sizeof(std::uint16_t);
}

} // namespace

const char* q5_linear_add_schedule_name(Q5LinearAddScheduleId schedule) noexcept {
    switch (schedule) {
    case Q5LinearAddScheduleId::GemvResidual:
        return "linear_add.q5.gemv.residual";
    case Q5LinearAddScheduleId::Materialized:
        return "linear_add.q5.materialized";
    case Q5LinearAddScheduleId::MmaResidualR64C64:
        return "linear_add.q5.mma.r64.c64.cta_collective_residual";
    case Q5LinearAddScheduleId::MmaResidualR64C128:
        return "linear_add.q5.mma.r64.c128.cta_collective_residual";
    }
    return "linear_add.q5.unknown";
}

bool q5_linear_add_admits(const Q5LinearAddProblem& problem) noexcept {
    return supported_shape(problem) && problem.cols >= 1;
}

Q5LinearAddPlan q5_linear_add_resolve_plan(const Q5LinearAddProblem& problem) {
    if (!q5_linear_add_admits(problem)) {
        throw std::invalid_argument("q5 linear_add: exact problem or column count is not admitted");
    }

    for (const RouteSpec& route : kRoutes) {
        if (!route.cols.contains(problem.cols)) { continue; }
        Q5LinearAddPlan plan{
            route.schedule,
            0,
        };
        switch (route.schedule) {
        case Q5LinearAddScheduleId::GemvResidual:
            return plan;
        case Q5LinearAddScheduleId::Materialized: {
            plan.workspace_bytes = checked_matrix_bytes(problem.rows, problem.cols);
            return plan;
        }
        case Q5LinearAddScheduleId::MmaResidualR64C64:
        case Q5LinearAddScheduleId::MmaResidualR64C128:
            return plan;
        }
    }
    throw std::logic_error("q5 linear_add: admitted problem has no covering route");
}

std::size_t q5_linear_add_capacity_workspace_bytes(std::int32_t rows, std::int32_t k,
                                                   std::int32_t padded_k, std::int32_t max_cols) {
    const Q5LinearAddProblem maximum_problem{rows, k, padded_k, max_cols};
    (void)q5_linear_add_resolve_plan(maximum_problem);

    std::size_t maximum = 0;
    for (const RouteSpec& route : kRoutes) {
        if (route.cols.first > max_cols) { continue; }
        const std::int32_t endpoint = std::min(route.cols.last, max_cols);
        maximum                     = std::max(
            maximum, q5_linear_add_resolve_plan({rows, k, padded_k, endpoint}).workspace_bytes);
    }
    return maximum;
}

void q5_linear_add_execute_plan(const Q5LinearAddPlan& plan, const Tensor& x, const Weight& w,
                                Tensor& residual_out, WorkspaceArena& ws, cudaStream_t stream) {
    const Q5LinearAddProblem problem{residual_out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    const Q5LinearAddPlan resolved = q5_linear_add_resolve_plan(problem);
    if (resolved.schedule != plan.schedule || resolved.workspace_bytes != plan.workspace_bytes) {
        throw std::invalid_argument("q5 linear_add: plan does not match the exact problem");
    }

    switch (plan.schedule) {
    case Q5LinearAddScheduleId::GemvResidual:
        q5_linear_add_gemv_residual_launch(x, w, residual_out, stream);
        return;
    case Q5LinearAddScheduleId::Materialized: {
        auto scratch_scope = ws.scope();
        Tensor projected   = ws.alloc(DType::BF16, {problem.rows, problem.cols});
        linear(x, w, projected, ws, stream);
        residual_add(projected, residual_out, stream);
        return;
    }
    case Q5LinearAddScheduleId::MmaResidualR64C64:
        q5_linear_add_mma_r64_c64_launch(x, w, residual_out, stream);
        return;
    case Q5LinearAddScheduleId::MmaResidualR64C128:
        q5_linear_add_mma_r64_c128_launch(x, w, residual_out, stream);
        return;
    }
    throw std::logic_error("q5 linear_add: unknown schedule");
}

void q5_linear_add_dispatch(const Tensor& x, const Weight& w, Tensor& residual_out,
                            WorkspaceArena& ws, cudaStream_t stream) {
    const Q5LinearAddProblem problem{residual_out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    const Q5LinearAddPlan plan = q5_linear_add_resolve_plan(problem);
    q5_linear_add_execute_plan(plan, x, w, residual_out, ws, stream);
}

} // namespace ninfer::ops::detail
