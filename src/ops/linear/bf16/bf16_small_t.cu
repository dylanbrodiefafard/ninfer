#include "ops/linear/bf16/bf16_launch.h"

#include "core/device.h"
#include "ops/linear/bf16/bf16_config.h"
#include "ops/linear/bf16/bf16_small_t.cuh"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

constexpr int kFirstSmallT = 2;
constexpr int kLastSmallT  = 32;
using Launch               = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

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
    return std::array<Launch, sizeof...(Offsets)>{
        &launch_exact<kFirstSmallT + static_cast<int>(Offsets)>...};
}

constexpr auto kLaunchers =
    make_launchers(std::make_index_sequence<kLastSmallT - kFirstSmallT + 1>{});

} // namespace

void launch_bf16_small_t(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    if (x.ne[1] < kFirstSmallT || x.ne[1] > kLastSmallT) {
        throw std::invalid_argument("bf16 linear small-T requires T in [2,32]");
    }
    kLaunchers[x.ne[1] - kFirstSmallT](x, weight, out, stream);
}

} // namespace ninfer::ops::detail
