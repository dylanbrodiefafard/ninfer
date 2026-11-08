#include "ops/linear/nvfp4/nvfp4_launch.h"

#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_small_t.cuh"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

using Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

template <class Geometry, int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Schedule = typename Nvfp4LinearSmallTProductionSchedule<Geometry, ActiveTokens>::Type;
    constexpr int kTokenTiles = (ActiveTokens + Schedule::kTokenTile - 1) / Schedule::kTokenTile;
    constexpr int kBlocks     = (Geometry::kOutputRows / Schedule::kRowsPerCta) * kTokenTiles;

    const Nvfp4ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data),
                                       Geometry::kOutputRows};
    const float inverse_weight_divisor = 1.0F / weight.weight_scale_divisor;
    nvfp4_small_t_kernel<Geometry, ActiveTokens, Schedule>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(
            Nvfp4PackedActivation<Geometry>{static_cast<const __nv_bfloat16*>(x.data)},
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), inverse_weight_divisor,
            Nvfp4IdentityEpilogue{}, output);
    CUDA_CHECK(cudaGetLastError());
}

// The exact-T launcher at an admitted width, nullptr elsewhere, so only admitted widths
// instantiate.
template <class Geometry, Nvfp4Problem Problem, int ActiveTokens>
constexpr Launch admitted_launcher() {
    if constexpr (nvfp4_linear_small_t_admitted(Problem, ActiveTokens)) {
        return &launch_exact<Geometry, ActiveTokens>;
    } else {
        return nullptr;
    }
}

template <class Geometry, Nvfp4Problem Problem, std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Launch, sizeof...(Offsets)>{
        admitted_launcher<Geometry, Problem, kNvfp4FirstSmallT + static_cast<int>(Offsets)>()...};
}

template <class Geometry, Nvfp4Problem Problem>
void launch_admitted(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    static constexpr auto kLaunchers = make_launchers<Geometry, Problem>(
        std::make_index_sequence<kNvfp4LastSmallT - kNvfp4FirstSmallT + 1>{});
    const std::int32_t tokens = x.ne[1];
    const Launch launch = tokens >= kNvfp4FirstSmallT && tokens <= kNvfp4LastSmallT
                              ? kLaunchers[static_cast<std::size_t>(tokens - kNvfp4FirstSmallT)]
                              : nullptr;
    if (launch == nullptr) {
        throw std::invalid_argument("nvfp4 linear SmallT: T is not registered for this problem");
    }
    launch(x, weight, out, stream);
}

} // namespace

void launch_nvfp4_small_t(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    switch (resolve_nvfp4_problem(weight.n, weight.k)) {
    case Nvfp4Problem::Residual17408:
        launch_admitted<Nvfp4Residual17408Geometry, Nvfp4Problem::Residual17408>(x, weight, out,
                                                                                 stream);
        return;
    case Nvfp4Problem::DflashConvProj:
        launch_admitted<Nvfp4DflashConvProjGeometry, Nvfp4Problem::DflashConvProj>(x, weight, out,
                                                                                   stream);
        return;
    case Nvfp4Problem::DflashSelector:
        launch_admitted<Nvfp4DflashSelectorGeometry, Nvfp4Problem::DflashSelector>(x, weight, out,
                                                                                   stream);
        return;
    case Nvfp4Problem::MtpFc:
        launch_admitted<Nvfp4MtpFcGeometry, Nvfp4Problem::MtpFc>(x, weight, out, stream);
        return;
    case Nvfp4Problem::AttnInput:
    case Nvfp4Problem::GdnInput:
    case Nvfp4Problem::MlpGateUp:
    case Nvfp4Problem::Residual6144:
    case Nvfp4Problem::DflashFeature:
    case Nvfp4Problem::DflashQkv:
    case Nvfp4Problem::DflashAttnOut:
        break;
    }
    throw std::invalid_argument("nvfp4 linear SmallT: the problem has no SmallT route");
}

using SplitKLaunch = void (*)(const Tensor&, const Tensor&, const Weight&, Tensor&, cudaStream_t);

template <int ActiveTokens>
void launch_splitk_exact(const Tensor& embedding, const Tensor& hidden, const Weight& weight,
                         Tensor& out, cudaStream_t stream) {
    using Geometry = Nvfp4MtpFcGeometry;
    using Schedule = typename Nvfp4LinearSmallTProductionSchedule<Geometry, ActiveTokens>::Type;
    constexpr int kValuesPerPhase = Schedule::kWarpsPerRow * 32 * Schedule::kValuesPerLane;
    static_assert((Geometry::kInputRows / 2) % kValuesPerPhase == 0);
    constexpr int kTokenTiles = (ActiveTokens + Schedule::kTokenTile - 1) / Schedule::kTokenTile;
    constexpr int kBlocks     = (Geometry::kOutputRows / Schedule::kRowsPerCta) * kTokenTiles;

    const Nvfp4ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data),
                                       Geometry::kOutputRows};
    const float inverse_weight_divisor = 1.0F / weight.weight_scale_divisor;
    nvfp4_small_t_kernel<Geometry, ActiveTokens, Schedule>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(
            Nvfp4SplitKActivation<Geometry>{static_cast<const __nv_bfloat16*>(embedding.data),
                                            static_cast<const __nv_bfloat16*>(hidden.data)},
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), inverse_weight_divisor,
            Nvfp4IdentityEpilogue{}, output);
    CUDA_CHECK(cudaGetLastError());
}

template <std::size_t... Offsets>
constexpr auto make_splitk_launchers(std::index_sequence<Offsets...>) {
    return std::array<SplitKLaunch, sizeof...(Offsets)>{
        &launch_splitk_exact<kNvfp4FirstSmallT + static_cast<int>(Offsets)>...};
}

void launch_nvfp4_small_t_splitk(const Tensor& embedding, const Tensor& hidden,
                                 const Weight& weight, Tensor& out, cudaStream_t stream) {
    static constexpr auto kLaunchers =
        make_splitk_launchers(std::make_index_sequence<kNvfp4LastSmallT - kNvfp4FirstSmallT + 1>{});
    kLaunchers[static_cast<std::size_t>(embedding.ne[1] - kNvfp4FirstSmallT)](embedding, hidden,
                                                                              weight, out, stream);
}

} // namespace ninfer::ops::detail
