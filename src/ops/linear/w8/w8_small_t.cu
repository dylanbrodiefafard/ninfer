#include "ops/linear/w8/w8_launch.h"

#include "core/device.h"
#include "ops/linear/w8/w8_config.h"
#include "ops/linear/w8/w8_small_t_mma.cuh"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

template <class Geometry, int ActiveTokens>
void launch_exact(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Schedule = typename W8LinearSmallTProductionSchedule<Geometry, ActiveTokens>::Type;
    static_assert((Geometry::kOutputRows % Schedule::kRowsPerCta) == 0);
    static_assert((Geometry::kInputRows % Schedule::kGroupK) == 0);

    const W8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), Geometry::kOutputRows};
    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    w8_small_t_mma_kernel<Geometry, ActiveTokens, Schedule>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), output);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<W8Launch, sizeof...(Offsets)>{
        &launch_exact<Geometry, kW8VocabularyFirstSmallT + static_cast<int>(Offsets)>...};
}

constexpr auto kVocabularyLaunchers = make_launchers<W8VocabularyProjectionGeometry>(
    std::make_index_sequence<kW8VocabularyLastSmallT - kW8VocabularyFirstSmallT + 1>{});

} // namespace

void launch_w8_small_t(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    if (weight.n != W8VocabularyProjectionGeometry::kOutputRows ||
        weight.k != W8VocabularyProjectionGeometry::kInputRows ||
        weight.padded_shape[1] != W8VocabularyProjectionGeometry::kInputRows ||
        x.ne[1] < kW8VocabularyFirstSmallT || x.ne[1] > kW8VocabularyLastSmallT) {
        throw std::invalid_argument("W8 Linear small-T: unsupported exact problem");
    }
    const std::size_t index = static_cast<std::size_t>(x.ne[1] - kW8VocabularyFirstSmallT);
    kVocabularyLaunchers[index](x, weight, out, stream);
}

} // namespace ninfer::ops::detail
