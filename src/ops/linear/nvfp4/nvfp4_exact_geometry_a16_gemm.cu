#include "ops/linear/nvfp4/nvfp4_launch.h"

#include "core/device.h"
#include "ops/common/token_slices.h"
#include "ops/linear/nvfp4/nvfp4_a16_gemm_mma.cuh"
#include "ops/linear/nvfp4/nvfp4_config.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// One persistent NVFP4 tile is decoded per activation panel. These schedules reuse the validated
// SM120 BF16-MMA panel shapes used by the row-scaled FP8 family while retaining NVFP4's exact
// persistent byte layout and direct scale addressing.
using Main128 = Nvfp4A16GemmSchedule<64, 128, 64, 64, 16, 2, 2>;
using Tail64  = Nvfp4A16GemmSchedule<128, 64, 64, 64, 16, 2, 2>;
using Tail96  = Nvfp4A16GemmSchedule<64, 96, 64, 64, 16, 2, 2>;

template <class Geometry, class Schedule, bool FullTokens>
void launch_slice(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    static_assert((Geometry::kOutputRows % Schedule::kBlockRows) == 0);
    static_assert((Geometry::kInputRows % Schedule::kBlockK) == 0);
    constexpr int row_tiles = Geometry::kOutputRows / Schedule::kBlockRows;
    const int token_tiles   = (x.ne[1] + Schedule::kBlockTokens - 1) / Schedule::kBlockTokens;
    const dim3 grid(static_cast<unsigned>(row_tiles), static_cast<unsigned>(token_tiles), 1U);
    const Nvfp4ContiguousOutput output{static_cast<__nv_bfloat16*>(out.data),
                                       Geometry::kOutputRows};
    nvfp4_a16_gemm_mma_kernel<Geometry, Schedule, FullTokens>
        <<<grid, Schedule::kThreads, Schedule::kSharedBytes, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), 1.0F / weight.weight_scale_divisor,
            output, x.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

template <class Geometry, class Schedule>
void launch_schedule(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    for_each_token_slice(x.ne[1], Schedule::kBlockTokens,
                         [&](std::int32_t offset, std::int32_t count) {
                             const Tensor input = x.slice(1, offset, count);
                             Tensor output      = out.slice(1, offset, count);
                             if ((count % Schedule::kBlockTokens) == 0) {
                                 launch_slice<Geometry, Schedule, true>(input, weight, output,
                                                                        stream);
                             } else {
                                 launch_slice<Geometry, Schedule, false>(input, weight, output,
                                                                         stream);
                             }
                         });
}

template <class Geometry>
void launch_tail(const Tensor& x, const Weight& weight, Tensor& out, std::int32_t first_gemm_t,
                 cudaStream_t stream) {
    if (x.ne[1] < first_gemm_t) {
        if (x.ne[1] == 1) {
            launch_nvfp4_exact_geometry_decode(x, weight, out, stream);
        } else {
            launch_nvfp4_exact_geometry_small_t(x, weight, out, stream);
        }
    } else if (x.ne[1] <= Tail64::kBlockTokens) {
        launch_schedule<Geometry, Tail64>(x, weight, out, stream);
    } else {
        launch_schedule<Geometry, Tail96>(x, weight, out, stream);
    }
}

template <class Geometry>
void launch_registered(const Tensor& x, const Weight& weight, Tensor& out,
                       std::int32_t first_gemm_t, cudaStream_t stream) {
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
    launch_tail<Geometry>(input_tail, weight, output_tail, first_gemm_t, stream);
}

} // namespace

void launch_nvfp4_exact_geometry_a16_gemm(const Tensor& x, const Weight& weight, Tensor& out,
                                           cudaStream_t stream) {
    const Nvfp4Problem problem = resolve_nvfp4_problem(weight.n, weight.k);
    const std::int32_t first_gemm_t = nvfp4_exact_a16_gemm_first_t(problem);
    if (x.ne[1] < first_gemm_t) {
        throw std::invalid_argument("nvfp4 exact-geometry A16 GEMM: invalid token extent");
    }
    switch (problem) {
    case Nvfp4Problem::N10240K2560:
        return launch_registered<Nvfp4N10240K2560Geometry>(x, weight, out, first_gemm_t, stream);
    case Nvfp4Problem::N6144K2560:
        return launch_registered<Nvfp4N6144K2560Geometry>(x, weight, out, first_gemm_t, stream);
    case Nvfp4Problem::N12288K2560:
        return launch_registered<Nvfp4N12288K2560Geometry>(x, weight, out, first_gemm_t, stream);
    case Nvfp4Problem::N512K2560:
        return launch_registered<Nvfp4N512K2560Geometry>(x, weight, out, first_gemm_t, stream);
    case Nvfp4Problem::N2560K6144:
        return launch_registered<Nvfp4N2560K6144Geometry>(x, weight, out, first_gemm_t, stream);
    case Nvfp4Problem::N640K2560:
        return launch_registered<Nvfp4N640K2560Geometry>(x, weight, out, first_gemm_t, stream);
    case Nvfp4Problem::N1280K2560:
        return launch_registered<Nvfp4N1280K2560Geometry>(x, weight, out, first_gemm_t, stream);
    case Nvfp4Problem::N2560K640:
        return launch_registered<Nvfp4N2560K640Geometry>(x, weight, out, first_gemm_t, stream);
    case Nvfp4Problem::N10240K320:
        return launch_registered<Nvfp4N10240K320Geometry>(x, weight, out, first_gemm_t, stream);
    case Nvfp4Problem::N2560K2560:
        return launch_registered<Nvfp4N2560K2560Geometry>(x, weight, out, first_gemm_t, stream);
    case Nvfp4Problem::N248320K2560:
        return launch_registered<Nvfp4N248320K2560Geometry>(x, weight, out, first_gemm_t, stream);
    default:
        throw std::invalid_argument("nvfp4 exact-geometry A16 GEMM: unsupported problem");
    }
}

} // namespace ninfer::ops::detail
