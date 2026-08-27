#include "ops/linear/nvfp4/nvfp4_launch.h"

#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_small_t.cuh"

#include <array>
#include <cstddef>
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

template <class Geometry, std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Launch, sizeof...(Offsets)>{
        &launch_exact<Geometry, kNvfp4FirstSmallT + static_cast<int>(Offsets)>...};
}

template <class Geometry>
const auto& launchers() {
    static constexpr auto kLaunchers = make_launchers<Geometry>(
        std::make_index_sequence<kNvfp4LastSmallT - kNvfp4FirstSmallT + 1>{});
    return kLaunchers;
}

} // namespace

void launch_nvfp4_small_t(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const std::size_t index = static_cast<std::size_t>(x.ne[1] - kNvfp4FirstSmallT);
    switch (resolve_nvfp4_problem(weight.n, weight.k)) {
    case Nvfp4Problem::AttnInput:
        launchers<Nvfp4AttnInputGeometry>()[index](x, weight, out, stream);
        return;
    case Nvfp4Problem::GdnInput:
        launchers<Nvfp4GdnInputGeometry>()[index](x, weight, out, stream);
        return;
    case Nvfp4Problem::MlpGateUp:
        launchers<Nvfp4MlpGateUpGeometry>()[index](x, weight, out, stream);
        return;
    case Nvfp4Problem::Residual6144:
        launchers<Nvfp4Residual6144Geometry>()[index](x, weight, out, stream);
        return;
    case Nvfp4Problem::Residual17408:
        launchers<Nvfp4Residual17408Geometry>()[index](x, weight, out, stream);
        return;
    case Nvfp4Problem::DflashFeature:
        launchers<Nvfp4DflashFeatureGeometry>()[index](x, weight, out, stream);
        return;
    case Nvfp4Problem::DflashQkv:
        launchers<Nvfp4DflashQkvGeometry>()[index](x, weight, out, stream);
        return;
    case Nvfp4Problem::DflashAttnOut:
        launchers<Nvfp4DflashAttnOutGeometry>()[index](x, weight, out, stream);
        return;
    case Nvfp4Problem::DflashConvProj:
        launchers<Nvfp4DflashConvProjGeometry>()[index](x, weight, out, stream);
        return;
    case Nvfp4Problem::DflashSelector:
        launchers<Nvfp4DflashSelectorGeometry>()[index](x, weight, out, stream);
        return;
    case Nvfp4Problem::MtpFc:
        launchers<Nvfp4MtpFcGeometry>()[index](x, weight, out, stream);
        return;
    }
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
    static constexpr auto kLaunchers = make_splitk_launchers(
        std::make_index_sequence<kNvfp4LastSmallT - kNvfp4FirstSmallT + 1>{});
    kLaunchers[static_cast<std::size_t>(embedding.ne[1] - kNvfp4FirstSmallT)](embedding, hidden,
                                                                             weight, out, stream);
}

} // namespace ninfer::ops::detail
