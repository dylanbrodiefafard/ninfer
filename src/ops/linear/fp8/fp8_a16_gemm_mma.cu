#include "ops/linear/fp8/fp8_launch.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/linear/fp8/fp8_a16_gemm_mma.cuh"
#include "ops/linear/fp8/fp8_config.h"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// RTX 5090 cold-cache winners for the exact [248320,5120] vocabulary problem. The 128-token
// schedule is the large-T computation core. The 64- and 96-token schedules avoid executing a
// mostly empty final token tile; dispatch emits at most one such tail launch.
using Main128 = Fp8A16GemmSchedule<64, 128, 64, 64, 16, 2, 2>;
using Tail64  = Fp8A16GemmSchedule<128, 64, 64, 64, 16, 2, 2>;
using Tail96  = Fp8A16GemmSchedule<64, 96, 64, 64, 16, 2, 2>;
using ExactGeometry128 = Fp8A16GemmSchedule<64, 128, 64, 32, 16, 2, 2>;

template <class Geometry, class Schedule, bool FullTokens>
void launch_slice(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    static_assert((Geometry::kOutputRows % Schedule::kBlockRows) == 0);
    static_assert((Geometry::kInputRows % Schedule::kBlockK) == 0);
    constexpr int row_tiles = Geometry::kOutputRows / Schedule::kBlockRows;
    const int token_tiles   = div_up(x.ne[1], Schedule::kBlockTokens);
    const dim3 grid(static_cast<unsigned>(row_tiles), static_cast<unsigned>(token_tiles), 1U);
    const Fp8ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data), Geometry::kOutputRows};
    fp8_a16_gemm_mma_kernel<Geometry, Schedule, FullTokens>
        <<<grid, Schedule::kThreads, Schedule::kSharedBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const __nv_bfloat16*>(weight.scales), output, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, class Schedule>
void launch_schedule(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    for_each_token_slice(x.ne[1], Schedule::kBlockTokens,
                         [&](std::int32_t offset, std::int32_t count) {
                             const Tensor input = x.slice(1, offset, count);
                             Tensor output      = out.slice(1, offset, count);
                             if ((count % Schedule::kBlockTokens) == 0) {
                                 launch_slice<Geometry, Schedule, true>(input, weight, output, stream);
                             } else {
                                 launch_slice<Geometry, Schedule, false>(input, weight, output, stream);
                             }
                         });
}

template <class Geometry>
void launch_tail(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    if (x.ne[1] < kFp8VocabularyFirstA16GemmT) {
        launch_fp8_vocabulary_a16_small_t(x, weight, out, stream);
    } else if (x.ne[1] <= Tail64::kBlockTokens) {
        launch_schedule<Geometry, Tail64>(x, weight, out, stream);
    } else {
        launch_schedule<Geometry, Tail96>(x, weight, out, stream);
    }
}

template <class Geometry>
void launch_registered(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const std::int32_t tokens = x.ne[1];
    if (tokens <= Tail64::kBlockTokens) {
        launch_schedule<Geometry, Tail64>(x, weight, out, stream);
        return;
    }
    if (tokens <= Tail96::kBlockTokens) {
        launch_schedule<Geometry, Tail96>(x, weight, out, stream);
        return;
    }
    if (tokens <= Main128::kBlockTokens) {
        launch_schedule<Geometry, Main128>(x, weight, out, stream);
        return;
    }

    if ((tokens >= 161 && tokens <= 192) || (tokens >= 257 && tokens <= 288)) {
        launch_schedule<Geometry, Tail96>(x, weight, out, stream);
        return;
    }

    const std::int32_t tail = tokens % Main128::kBlockTokens;
    if (tail == 0 || tail > Tail96::kBlockTokens) {
        launch_schedule<Geometry, Main128>(x, weight, out, stream);
        return;
    }

    const std::int32_t prefix = tokens - tail;
    const Tensor input_prefix = x.slice(1, 0, prefix);
    Tensor output_prefix      = out.slice(1, 0, prefix);
    launch_schedule<Geometry, Main128>(input_prefix, weight, output_prefix, stream);

    const Tensor input_tail = x.slice(1, prefix, tail);
    Tensor output_tail      = out.slice(1, prefix, tail);
    launch_tail<Geometry>(input_tail, weight, output_tail, stream);
}

} // namespace

void launch_fp8_vocabulary_a16_gemm(const Tensor& x, const Weight& weight, Tensor& out,
                                    cudaStream_t stream) {
    if (weight.n != Fp8VocabularyGeometry::kOutputRows ||
        (weight.k != Fp8VocabularyGeometry::kInputRows &&
         weight.k != Fp8Vocabulary2560Geometry::kInputRows) ||
        x.ne[1] < kFp8VocabularyFirstA16GemmT) {
        throw std::invalid_argument("fp8 vocabulary A16 GEMM: invalid exact problem");
    }
    if (weight.k == Fp8VocabularyGeometry::kInputRows) {
        launch_registered<Fp8VocabularyGeometry>(x, weight, out, stream);
        return;
    }
    launch_registered<Fp8Vocabulary2560Geometry>(x, weight, out, stream);
}

void launch_fp8_exact_geometry_a16_gemm(const Tensor& x, const Weight& weight, Tensor& out,
                                        cudaStream_t stream) {
    const Fp8Problem problem = resolve_fp8_problem(weight.n, weight.k);
    if (!is_fp8_exact_geometry_problem(problem) ||
        x.ne[1] < fp8_exact_a16_gemm_first_t(problem)) {
        throw std::invalid_argument("fp8 exact-geometry A16 GEMM: invalid token extent");
    }
    switch (problem) {
    case Fp8Problem::Rows10240K2560:
        launch_schedule<Fp8Rows10240K2560Geometry, ExactGeometry128>(x, weight, out, stream);
        return;
    case Fp8Problem::Rows6144K2560:
        launch_schedule<Fp8Rows6144K2560Geometry, ExactGeometry128>(x, weight, out, stream);
        return;
    case Fp8Problem::Rows12288K2560:
        launch_schedule<Fp8Rows12288K2560Geometry, ExactGeometry128>(x, weight, out, stream);
        return;
    case Fp8Problem::Rows512K2560:
        launch_schedule<Fp8Rows512K2560Geometry, ExactGeometry128>(x, weight, out, stream);
        return;
    case Fp8Problem::Rows2560K6144:
        launch_schedule<Fp8Rows2560K6144Geometry, ExactGeometry128>(x, weight, out, stream);
        return;
    case Fp8Problem::Rows640K2560:
        launch_schedule<Fp8Rows640K2560Geometry, ExactGeometry128>(x, weight, out, stream);
        return;
    case Fp8Problem::Rows1280K2560:
        launch_schedule<Fp8Rows1280K2560Geometry, ExactGeometry128>(x, weight, out, stream);
        return;
    case Fp8Problem::Rows2560K640:
        launch_schedule<Fp8Rows2560K640Geometry, ExactGeometry128>(x, weight, out, stream);
        return;
    case Fp8Problem::Rows320K10240:
        launch_schedule<Fp8Rows320K10240Geometry, ExactGeometry128>(x, weight, out, stream);
        return;
    case Fp8Problem::Rows10240K320:
        launch_schedule<Fp8Rows10240K320Geometry, ExactGeometry128>(x, weight, out, stream);
        return;
    case Fp8Problem::Rows2560K2560:
        launch_schedule<Fp8Rows2560K2560Geometry, ExactGeometry128>(x, weight, out, stream);
        return;
    case Fp8Problem::AttnInput:
    case Fp8Problem::GdnInput:
    case Fp8Problem::MlpGateUp:
    case Fp8Problem::Vocabulary:
    case Fp8Problem::Residual6144:
    case Fp8Problem::Residual17408:
    case Fp8Problem::Vocabulary2560:
        break;
    }
    throw std::invalid_argument("fp8 exact-geometry A16 GEMM: invalid exact problem");
}

} // namespace ninfer::ops::detail
