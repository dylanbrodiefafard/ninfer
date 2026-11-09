#include "ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.h"

#include "ninfer/ops/rmsnorm.h"

#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

inline constexpr std::int32_t kAnyCols = std::numeric_limits<std::int32_t>::max();

struct ColsSet {
    std::int32_t first;
    std::int32_t last;

    constexpr bool contains(std::int32_t cols) const noexcept {
        return cols >= first && cols <= last;
    }
};

struct RouteSpec {
    ColsSet cols;
    Bf16GdnGatingScheduleId schedule;
};

constexpr std::array<RouteSpec, 5> k27Routes{{
    // A fixed split-40 reduction covers decode and compact verify batches. The narrow token
    // tiles share identical arithmetic, including packed C=1..12 execution with T<=48.
    {{1, 48}, Bf16GdnGatingScheduleId::MmaCooperativeSplit40},
    // As token tiles double, halve SplitK. This keeps the cooperative grid near 192 CTAs instead
    // of making T a launch limit. Once the unsplit grid has enough independent work, it also
    // removes the cooperative-residency constraint.
    {{49, 1024}, Bf16GdnGatingScheduleId::MmaCooperativeSplit8},
    {{1025, 2048}, Bf16GdnGatingScheduleId::MmaCooperativeSplit4},
    {{2049, 4096}, Bf16GdnGatingScheduleId::MmaCooperativeSplit2},
    {{4097, kAnyCols}, Bf16GdnGatingScheduleId::MmaUnsplit},
}};

template <std::size_t N>
constexpr bool catalog_is_closed(const std::array<RouteSpec, N>& routes,
                                 std::int32_t last) noexcept {
    std::int64_t expected = 1;
    for (const RouteSpec& route : routes) {
        if (route.cols.first != expected || route.cols.last < route.cols.first) { return false; }
        expected = static_cast<std::int64_t>(route.cols.last) + 1;
    }
    return routes.back().cols.last == last && expected == static_cast<std::int64_t>(last) + 1;
}

static_assert(catalog_is_closed(k27Routes, kAnyCols));

// Every registered schedule tiles the long-context token axis in 128-column MMA tiles.
inline constexpr std::int32_t kMmaTileCols = 128;

std::int32_t schedule_split_k(Bf16GdnGatingScheduleId schedule) {
    switch (schedule) {
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit40:
        return 40;
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit8:
        return 8;
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit4:
        return 4;
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit2:
        return 2;
    case Bf16GdnGatingScheduleId::MmaUnsplit:
        return 1;
    }
    throw std::logic_error("BF16 GDN gating: unknown schedule");
}

bool cooperative_grid_is_resident(Bf16GdnGatingScheduleId schedule, std::int32_t cols) {
    // BN128 uses 40 KiB of dynamic shared memory. Split8 uses 71 registers with 256 threads;
    // split4/2 use 62 registers with 512 threads. Each specialization admits two CTAs/SM, hence
    // 340 resident CTAs device-wide. There are three 16-row tiles per token tile.
    constexpr std::int64_t kRowTiles     = 3;
    constexpr std::int64_t kResidentCtas = 340;
    const std::int64_t column_tiles =
        (static_cast<std::int64_t>(cols) + kMmaTileCols - 1) / kMmaTileCols;
    return column_tiles * kRowTiles * schedule_split_k(schedule) <= kResidentCtas;
}

bool candidate_is_legal(Bf16GdnGatingScheduleId schedule, const Bf16GdnGatingProblem& problem) {
    if (!bf16_gdn_gating_admits(problem)) { return false; }
    switch (schedule) {
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit40:
        // 120 CTAs, at most six warps and 20 KiB shared per CTA: one CTA/SM suffices
        // on the 170-SM RTX 5090, independently of the token-tile specialization.
        return problem.cols <= 48;
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit8:
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit4:
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit2:
        return cooperative_grid_is_resident(schedule, problem.cols);
    case Bf16GdnGatingScheduleId::MmaUnsplit:
        return true;
    }
    return false;
}

std::size_t checked_partial_bytes(std::int32_t heads, std::int32_t split_k, std::int32_t cols) {
    const std::size_t logical_rows = 2 * static_cast<std::size_t>(heads);
    const std::size_t split        = static_cast<std::size_t>(split_k);
    const std::size_t tokens       = static_cast<std::size_t>(cols);
    if (tokens > std::numeric_limits<std::size_t>::max() / logical_rows ||
        split > std::numeric_limits<std::size_t>::max() / (tokens * logical_rows)) {
        throw std::overflow_error("BF16 GDN gating workspace element count overflows size_t");
    }
    const std::size_t elements = split * tokens * logical_rows;
    if (elements > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        throw std::overflow_error("BF16 GDN gating workspace byte count overflows size_t");
    }
    return elements * sizeof(float);
}

Bf16GdnGatingPlan make_plan(Bf16GdnGatingScheduleId schedule, const Bf16GdnGatingProblem& problem) {
    const Bf16GdnGatingTokenVariant variant =
        schedule == Bf16GdnGatingScheduleId::MmaCooperativeSplit40 ||
                problem.cols % kMmaTileCols != 0
            ? Bf16GdnGatingTokenVariant::Predicated
            : Bf16GdnGatingTokenVariant::Full;
    const std::int32_t split_k = schedule_split_k(schedule);
    const std::size_t workspace =
        split_k > 1 ? checked_partial_bytes(problem.heads, split_k, problem.cols) : 0;
    return {schedule, variant, workspace};
}

void execute_resolved(const Bf16GdnGatingPlan& plan, const Tensor& x, const Weight& a_weight,
                      const Weight& b_weight, const Tensor& A_log, const Tensor& dt_bias,
                      WorkspaceArena& ws, Tensor& g, Tensor& beta, cudaStream_t stream) {
    auto scratch_scope = ws.scope();
    DeviceSpan scratch{};
    if (plan.workspace_bytes != 0) { scratch = ws.alloc_bytes(plan.workspace_bytes); }

    switch (plan.schedule) {
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit40:
        bf16_gdn_gating_proj_mma_split40_launch(x, a_weight, b_weight, A_log, dt_bias, scratch.data,
                                                g, beta, stream);
        return;
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit8:
        bf16_gdn_gating_proj_mma_split8_launch(plan.token_variant, x, a_weight, b_weight, A_log,
                                               dt_bias, scratch.data, g, beta, stream);
        return;
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit4:
        bf16_gdn_gating_proj_mma_split4_launch(plan.token_variant, x, a_weight, b_weight, A_log,
                                               dt_bias, scratch.data, g, beta, stream);
        return;
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit2:
        bf16_gdn_gating_proj_mma_split2_launch(plan.token_variant, x, a_weight, b_weight, A_log,
                                               dt_bias, scratch.data, g, beta, stream);
        return;
    case Bf16GdnGatingScheduleId::MmaUnsplit:
        bf16_gdn_gating_proj_mma_unsplit_launch(plan.token_variant, x, a_weight, b_weight, A_log,
                                                dt_bias, g, beta, stream);
        return;
    }
    throw std::logic_error("BF16 GDN gating: unknown schedule");
}

std::size_t route_capacity(const Bf16GdnGatingProblem& base, std::int32_t min_cols,
                           std::int32_t max_cols) {
    std::size_t maximum = 0;
    for (const RouteSpec& route : k27Routes) {
        if (route.cols.last < min_cols || route.cols.first > max_cols) { continue; }
        const std::int32_t endpoint = std::min(route.cols.last, max_cols);
        maximum                     = std::max(
            maximum,
            bf16_gdn_gating_resolve_plan({base.heads, base.input_rows, endpoint}).workspace_bytes);
    }
    return maximum;
}

} // namespace

const char* bf16_gdn_gating_schedule_name(Bf16GdnGatingScheduleId schedule) noexcept {
    switch (schedule) {
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit40:
        return "gdn_gating_proj.bf16.mma.cooperative_split40";
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit8:
        return "gdn_gating_proj.bf16.mma.cooperative_split8";
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit4:
        return "gdn_gating_proj.bf16.mma.cooperative_split4";
    case Bf16GdnGatingScheduleId::MmaCooperativeSplit2:
        return "gdn_gating_proj.bf16.mma.cooperative_split2";
    case Bf16GdnGatingScheduleId::MmaUnsplit:
        return "gdn_gating_proj.bf16.mma.unsplit";
    }
    return "gdn_gating_proj.bf16.unknown";
}

bool bf16_gdn_gating_admits(const Bf16GdnGatingProblem& problem) noexcept {
    return problem.cols >= 1 && problem.heads == 48 && problem.input_rows == 5120;
}

Bf16GdnGatingPlan bf16_gdn_gating_resolve_candidate(Bf16GdnGatingScheduleId schedule,
                                                    const Bf16GdnGatingProblem& problem) {
    if (!candidate_is_legal(schedule, problem)) {
        throw std::invalid_argument("BF16 GDN gating: candidate is not legal for exact problem");
    }
    return make_plan(schedule, problem);
}

Bf16GdnGatingPlan bf16_gdn_gating_resolve_packed_plan(const Bf16GdnGatingProblem& problem) {
    if (!bf16_gdn_gating_admits(problem) || problem.cols < 2 ||
        problem.cols > kBf16GdnGatingPackedMaxCols) {
        throw std::invalid_argument("BF16 GDN gating: packed plan requires T=2..48");
    }
    return make_plan(Bf16GdnGatingScheduleId::MmaCooperativeSplit40, problem);
}

Bf16GdnGatingPlan bf16_gdn_gating_resolve_plan(const Bf16GdnGatingProblem& problem) {
    if (!bf16_gdn_gating_admits(problem)) {
        throw std::invalid_argument(
            "BF16 GDN gating: exact problem or column count is not admitted");
    }
    for (const RouteSpec& route : k27Routes) {
        if (route.cols.contains(problem.cols)) {
            return bf16_gdn_gating_resolve_candidate(route.schedule, problem);
        }
    }
    throw std::logic_error("BF16 GDN gating: admitted problem has no covering route");
}

std::size_t bf16_gdn_gating_capacity_workspace_bytes(std::int32_t heads, std::int32_t input_rows,
                                                     std::int32_t min_cols, std::int32_t max_cols) {
    if (min_cols <= 0 || max_cols < min_cols) {
        throw std::invalid_argument("BF16 GDN gating: invalid column interval");
    }
    const Bf16GdnGatingProblem base{heads, input_rows, 1};
    (void)bf16_gdn_gating_resolve_plan({heads, input_rows, min_cols});
    (void)bf16_gdn_gating_resolve_plan({heads, input_rows, max_cols});
    return route_capacity(base, min_cols, max_cols);
}

std::size_t bf16_gdn_norm_gating_packed_sequences_capacity_workspace_bytes(
    std::int32_t sequence_width, std::int32_t min_batch, std::int32_t max_batch) {
    if (sequence_width <= 0 || min_batch <= 0 || max_batch < min_batch ||
        max_batch > kBf16GdnGatingPackedMaxBatch ||
        sequence_width > std::numeric_limits<std::int32_t>::max() / max_batch) {
        throw std::invalid_argument(
            "BF16 GDN packed norm/control: invalid width or batch interval");
    }
    std::size_t maximum = 0;
    for (std::int32_t batch = min_batch; batch <= max_batch; ++batch) {
        if (bf16_gdn_gating_packed_aggregates(sequence_width, batch)) {
            maximum = std::max(
                maximum, bf16_gdn_gating_resolve_packed_plan({48, 5120, sequence_width * batch})
                             .workspace_bytes);
        } else {
            maximum = std::max(maximum, bf16_gdn_gating_capacity_workspace_bytes(
                                            48, 5120, sequence_width, sequence_width));
        }
    }
    return maximum;
}

void bf16_gdn_gating_execute_plan(const Bf16GdnGatingPlan& plan, const Tensor& x,
                                  const Weight& a_weight, const Weight& b_weight,
                                  const Tensor& A_log, const Tensor& dt_bias, WorkspaceArena& ws,
                                  Tensor& g, Tensor& beta, cudaStream_t stream) {
    const Bf16GdnGatingProblem problem{g.ne[0], x.ne[0], x.ne[1]};
    const Bf16GdnGatingPlan resolved = bf16_gdn_gating_resolve_plan(problem);
    if (resolved.schedule != plan.schedule || resolved.token_variant != plan.token_variant ||
        resolved.workspace_bytes != plan.workspace_bytes) {
        throw std::invalid_argument("BF16 GDN gating: plan does not match the exact problem");
    }
    execute_resolved(plan, x, a_weight, b_weight, A_log, dt_bias, ws, g, beta, stream);
}

void bf16_gdn_gating_execute_candidate(Bf16GdnGatingScheduleId schedule, const Tensor& x,
                                       const Weight& a_weight, const Weight& b_weight,
                                       const Tensor& A_log, const Tensor& dt_bias,
                                       WorkspaceArena& ws, Tensor& g, Tensor& beta,
                                       cudaStream_t stream) {
    const Bf16GdnGatingProblem problem{g.ne[0], x.ne[0], x.ne[1]};
    const Bf16GdnGatingPlan plan = bf16_gdn_gating_resolve_candidate(schedule, problem);
    execute_resolved(plan, x, a_weight, b_weight, A_log, dt_bias, ws, g, beta, stream);
}

void bf16_gdn_gating_dispatch(const Tensor& x, const Weight& a_weight, const Weight& b_weight,
                              const Tensor& A_log, const Tensor& dt_bias, WorkspaceArena& ws,
                              Tensor& g, Tensor& beta, cudaStream_t stream) {
    const Bf16GdnGatingPlan plan = bf16_gdn_gating_resolve_plan({g.ne[0], x.ne[0], x.ne[1]});
    bf16_gdn_gating_execute_plan(plan, x, a_weight, b_weight, A_log, dt_bias, ws, g, beta, stream);
}

namespace {

// The standalone unit-offset RMSNorm ahead of the control projection, publishing h's A8 or A4
// activation as well when the caller requests it.
void normalize(const Tensor& x, const Tensor& norm_weight, float eps, Tensor& h,
               Bf16GdnHiddenActivation h_activation, cudaStream_t stream) {
    if (h_activation.a8 != nullptr) {
        rmsnorm_a8(x, norm_weight, eps, &h, *h_activation.a8, stream);
    } else if (h_activation.a4 != nullptr) {
        rmsnorm_a4(x, norm_weight, eps, &h, *h_activation.a4, stream);
    } else {
        rmsnorm(x, norm_weight, eps, true, h, stream);
    }
}

} // namespace

void bf16_gdn_norm_gating_dispatch(const Tensor& x, const Tensor& norm_weight, float eps, Tensor& h,
                                   Bf16GdnHiddenActivation h_activation, const Weight& a_weight,
                                   const Weight& b_weight, const Tensor& A_log,
                                   const Tensor& dt_bias, WorkspaceArena& ws, Tensor& g,
                                   Tensor& beta, cudaStream_t stream) {
    const Bf16GdnGatingPlan plan = bf16_gdn_gating_resolve_plan({g.ne[0], x.ne[0], x.ne[1]});
    normalize(x, norm_weight, eps, h, h_activation, stream);
    execute_resolved(plan, h, a_weight, b_weight, A_log, dt_bias, ws, g, beta, stream);
}

void bf16_gdn_norm_gating_packed_dispatch(const Tensor& x, const Tensor& norm_weight, float eps,
                                          Tensor& h, A8Activation* h_activation,
                                          const Weight& a_weight, const Weight& b_weight,
                                          const Tensor& A_log, const Tensor& dt_bias,
                                          WorkspaceArena& ws, Tensor& g, Tensor& beta,
                                          cudaStream_t stream) {
    const Bf16GdnGatingPlan plan = bf16_gdn_gating_resolve_packed_plan({g.ne[0], x.ne[0], x.ne[1]});
    normalize(x, norm_weight, eps, h, {h_activation, nullptr}, stream);
    execute_resolved(plan, h, a_weight, b_weight, A_log, dt_bias, ws, g, beta, stream);
}

} // namespace ninfer::ops::detail
