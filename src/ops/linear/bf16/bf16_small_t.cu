#include "ops/linear/bf16/bf16_launch.h"

#include "core/device.h"
#include "ops/linear/bf16/bf16_config.h"
#include "ops/linear/bf16/bf16_small_t.cuh"

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace ninfer::ops::detail {
namespace {

template <int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Geometry = Bf16LinearControlGeometry;
    using Schedule = typename Bf16LinearSmallTProductionSchedule<ActiveTokens>::Type;
    static_assert((Geometry::kOutputRows % Schedule::kRowsPerCta) == 0);

    const Bf16SmallTContiguousOutput output{static_cast<__nv_bfloat16*>(out.data),
                                            Geometry::kOutputRows};
    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    bf16_small_t_inner_kernel<Geometry, ActiveTokens, Schedule>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const __nv_bfloat16*>(weight.qdata), output);
    CUDA_CHECK(cudaGetLastError());
}

template <std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Bf16Launch, sizeof...(Offsets)>{
        &launch_exact<kBf16SmallTMinTokens + static_cast<int>(Offsets)>...};
}

constexpr auto kLaunchers =
    make_launchers(std::make_index_sequence<kBf16SmallTMaxTokens - kBf16SmallTMinTokens + 1>{});

} // namespace

void launch_bf16_small_t(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    kLaunchers[x.ne[1] - kBf16SmallTMinTokens](x, weight, out, stream);
}

} // namespace ninfer::ops::detail
