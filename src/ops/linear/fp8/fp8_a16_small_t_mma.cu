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
    static_assert((Geometry::kInputRows % Schedule::kGroupK) == 0);
    constexpr int kBlocks = Geometry::kOutputRows / Schedule::kRowsPerCta;
    const Fp8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), Geometry::kOutputRows};
    fp8_a16_small_t_mma_kernel<Geometry, ActiveTokens, Schedule, Fp8ContiguousOutput, true>
        <<<kBlocks, Schedule::kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales), output, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

// One launcher per eight-column tile tier; each masks the columns beyond T.
template <class Geometry, std::size_t... Offsets>
constexpr auto make_launchers(std::index_sequence<Offsets...>) {
    return std::array<Launch, sizeof...(Offsets)>{
        &launch_tile<Geometry, 8 * (static_cast<int>(Offsets) + 1)>...};
}

constexpr std::size_t kTileTiers = kFp8VocabularyLastA16SmallTMmaT / 8;
static_assert(kFp8VocabularyFirstA16SmallTMmaT == 1 && kFp8VocabularyLastA16SmallTMmaT % 8 == 0);

template <class Geometry>
constexpr auto kLaunchers = make_launchers<Geometry>(std::make_index_sequence<kTileTiers>{});

} // namespace

void launch_fp8_vocabulary_a16_small_t(const Tensor& x, const Weight& weight, Tensor& out,
                                       cudaStream_t stream) {
    if (!is_fp8_linear_problem(weight.n, weight.k) || x.ne[1] < kFp8VocabularyFirstA16SmallTMmaT ||
        x.ne[1] > kFp8VocabularyLastA16SmallTMmaT) {
        throw std::invalid_argument("fp8 vocabulary A16 small-T MMA: invalid exact problem");
    }
    const auto index = static_cast<std::size_t>((x.ne[1] - 1) / 8);
    switch (resolve_fp8_problem(weight.n, weight.k)) {
    case Fp8Problem::Vocabulary5120:
        kLaunchers<Fp8Vocabulary5120Geometry>[index](x, weight, out, stream);
        return;
    case Fp8Problem::Vocabulary2560:
        kLaunchers<Fp8Vocabulary2560Geometry>[index](x, weight, out, stream);
        return;
    case Fp8Problem::AttnInput:
    case Fp8Problem::GdnInput:
    case Fp8Problem::MlpGateUp:
    case Fp8Problem::Residual6144:
    case Fp8Problem::Residual17408:
    case Fp8Problem::FlashNextGdnInput:
    case Fp8Problem::FlashNextMixerOut:
        break;
    }
    throw std::invalid_argument("fp8 vocabulary A16 small-T MMA: invalid exact problem");
}

} // namespace ninfer::ops::detail
