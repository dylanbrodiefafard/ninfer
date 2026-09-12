#include "ops/linear/fp8/fp8_launch.h"

#include "core/device.h"
#include "ops/linear/fp8/fp8_a16_small_t_mma.cuh"
#include "ops/linear/fp8/fp8_config.h"
#include "ops/linear/fp8/fp8_output.cuh"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>

namespace ninfer::ops::detail {
namespace {

using Launch = void (*)(const Tensor&, const Weight&, Tensor&, cudaStream_t);

template <class Geometry, int ActiveTokens>
void launch_tile(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    using Schedule =
        typename Fp8VocabularyA16SmallTMmaProductionSchedule<Geometry, ActiveTokens>::Type;
    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    const Fp8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), Geometry::kOutputRows};
    fp8_a16_small_t_mma_kernel<Geometry, ActiveTokens, Schedule, Fp8ContiguousOutput, true>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales), output, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Launch, sizeof...(Offsets)>{
        &launch_tile<Geometry, 8 * (static_cast<int>(Offsets) + 1)>...};
}

template <class Geometry>
const auto& launchers() {
    static constexpr auto kLaunchers = make_launchers<Geometry>(std::make_index_sequence<6>{});
    return kLaunchers;
}

template <class Geometry>
void launch_registered(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const std::size_t index = static_cast<std::size_t>((x.ne[1] - 1) / 8);
    launchers<Geometry>()[index](x, weight, out, stream);
}

} // namespace

void launch_fp8_vocabulary_a16_small_t(const Tensor& x, const Weight& weight, Tensor& out,
                                       cudaStream_t stream) {
    if (weight.n != Fp8VocabularyGeometry::kOutputRows ||
        (weight.k != Fp8VocabularyGeometry::kInputRows &&
         weight.k != Fp8Vocabulary2560Geometry::kInputRows) ||
        x.ne[1] < kFp8VocabularyFirstA16SmallTMmaT ||
        x.ne[1] > kFp8VocabularyLastA16SmallTMmaT) {
        throw std::invalid_argument("fp8 vocabulary A16 small-T MMA: invalid exact problem");
    }
    if (weight.k == Fp8VocabularyGeometry::kInputRows) {
        launch_registered<Fp8VocabularyGeometry>(x, weight, out, stream);
        return;
    }
    launch_registered<Fp8Vocabulary2560Geometry>(x, weight, out, stream);
}

} // namespace ninfer::ops::detail
