#include "ops/linear/nvfp4/nvfp4_launch.h"

#include "core/device.h"
#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_small_t.cuh"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

using Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

template <class Geometry, int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Schedule = typename Nvfp4LinearSmallTProductionSchedule<Geometry, ActiveTokens>::Type;
    constexpr int kTokenTiles = (ActiveTokens + Schedule::kTokenTile - 1) /
                                 Schedule::kTokenTile;
    constexpr int kBlocks = (Geometry::kOutputRows / Schedule::kRowsPerCta) * kTokenTiles;

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

void launch_nvfp4_exact_geometry_small_t(const Tensor& x, const Weight& weight, Tensor& out,
                                         cudaStream_t stream) {
    const std::size_t index = static_cast<std::size_t>(x.ne[1] - kNvfp4FirstSmallT);
    switch (resolve_nvfp4_problem(weight.n, weight.k)) {
    case Nvfp4Problem::N10240K2560:
        return launchers<Nvfp4N10240K2560Geometry>()[index](x, weight, out, stream);
    case Nvfp4Problem::N6144K2560:
        return launchers<Nvfp4N6144K2560Geometry>()[index](x, weight, out, stream);
    case Nvfp4Problem::N12288K2560:
        return launchers<Nvfp4N12288K2560Geometry>()[index](x, weight, out, stream);
    case Nvfp4Problem::N512K2560:
        return launchers<Nvfp4N512K2560Geometry>()[index](x, weight, out, stream);
    case Nvfp4Problem::N2560K6144:
        return launchers<Nvfp4N2560K6144Geometry>()[index](x, weight, out, stream);
    case Nvfp4Problem::N640K2560:
        return launchers<Nvfp4N640K2560Geometry>()[index](x, weight, out, stream);
    case Nvfp4Problem::N1280K2560:
        return launchers<Nvfp4N1280K2560Geometry>()[index](x, weight, out, stream);
    case Nvfp4Problem::N2560K640:
        return launchers<Nvfp4N2560K640Geometry>()[index](x, weight, out, stream);
    case Nvfp4Problem::N10240K320:
        return launchers<Nvfp4N10240K320Geometry>()[index](x, weight, out, stream);
    case Nvfp4Problem::N2560K2560:
        return launchers<Nvfp4N2560K2560Geometry>()[index](x, weight, out, stream);
    case Nvfp4Problem::N248320K2560:
        return launchers<Nvfp4N248320K2560Geometry>()[index](x, weight, out, stream);
    default:
        throw std::invalid_argument("nvfp4 exact-geometry SmallT: unsupported problem");
    }
}

} // namespace ninfer::ops::detail
