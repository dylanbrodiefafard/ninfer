#include "ops/linear_add/fp8/fp8_linear_add_plan.h"

#include "core/device.h"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_output.cuh"
#include "ops/linear/fp8/fp8_small_t.cuh"
#include "ops/linear_add/fp8/fp8_linear_add_epilogue.cuh"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

using Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

// SIMT column extent controls the accumulator register count; retain exact small extents,
// with common register/load profiles and no per-shape cache hints or isolated token exceptions.
template <class Geometry, int ActiveTokens>
struct Fp8LinearAddSmallTProductionSchedule {
    static_assert(ActiveTokens >= kFp8FirstSmallT && ActiveTokens <= kFp8LastSmallT);
    static constexpr int kWarpsPerCta   = ActiveTokens <= 19 ? 8 : 4;
    static constexpr int kRowsPerWarp   = ActiveTokens <= 5 ? 1 : 2;
    static constexpr int kValuesPerLane = ActiveTokens <= 19 ? 16 : 8;
    using Type = Fp8SmallTSchedule<kWarpsPerCta, kRowsPerWarp, kValuesPerLane, ActiveTokens, 1,
                                   Fp8SmallTActivationAccess::TokenPacked, Fp8CodeCache::Default, 1,
                                   Fp8SmallTBlockOrder::RowsContiguous, 1>;
};

template <class Geometry, int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& residual, cudaStream_t stream) {
    using Schedule = typename Fp8LinearAddSmallTProductionSchedule<Geometry, ActiveTokens>::Type;
    constexpr int kTokenTiles = (ActiveTokens + Schedule::kTokenTile - 1) / Schedule::kTokenTile;
    constexpr int kBlocks     = (Geometry::kOutputRows / Schedule::kRowsPerCta) * kTokenTiles;
    auto* output              = static_cast<__nv_bfloat16*>(residual.data);
    fp8_small_t_kernel<Geometry, ActiveTokens, Schedule>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales),
            Fp8ContiguousOutput{output, Geometry::kOutputRows},
            Fp8AddResidualEpilogue{output, Geometry::kOutputRows});
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Launch, sizeof...(Offsets)>{
        &launch_exact<Geometry, kFp8FirstSmallT + static_cast<int>(Offsets)>...};
}

// One launcher per A16 width below the projection's A8 width.
template <class Geometry>
void launch_below_a8(const Tensor& x, const Weight& weight, Tensor& residual, cudaStream_t stream) {
    constexpr std::int32_t kFirstA8 = fp8_linear_add_first_a8(Geometry::kInputRows);
    static_assert(kFirstA8 - 1 <= kFp8LastSmallT);
    static constexpr auto kLaunchers =
        make_launchers<Geometry>(std::make_index_sequence<kFirstA8 - kFp8FirstSmallT>{});
    const std::int32_t tokens = x.ne[1];
    if (tokens < kFp8FirstSmallT || tokens >= kFirstA8) {
        throw std::invalid_argument("fp8 linear_add small-T: T is at or above the A8 width");
    }
    kLaunchers[static_cast<std::size_t>(tokens - kFp8FirstSmallT)](x, weight, residual, stream);
}

} // namespace

void fp8_linear_add_small_t_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                   cudaStream_t stream) {
    switch (resolve_fp8_problem(weight.n, weight.k)) {
    case Fp8Problem::Residual6144:
        launch_below_a8<Fp8Residual6144Geometry>(x, weight, residual, stream);
        return;
    case Fp8Problem::Residual17408:
        launch_below_a8<Fp8Residual17408Geometry>(x, weight, residual, stream);
        return;
    case Fp8Problem::AttnInput:
    case Fp8Problem::GdnInput:
    case Fp8Problem::MlpGateUp:
    case Fp8Problem::Vocabulary:
        break;
    }
    throw std::invalid_argument("fp8 linear_add small-T: unsupported problem");
}

} // namespace ninfer::ops::detail
