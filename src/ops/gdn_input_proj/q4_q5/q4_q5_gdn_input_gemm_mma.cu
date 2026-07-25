#include "ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_kernels.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/rowsplit_grouped_mma.cuh"
#include "ops/common/token_slices.h"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

RowSplitGroupedMmaJob make_job(const Weight& weight, Tensor& out, std::int32_t row_offset) {
    return RowSplitGroupedMmaJob{
        static_cast<const std::uint8_t*>(weight.qdata),
        static_cast<const std::uint8_t*>(weight.qhigh),
        static_cast<const std::uint8_t*>(weight.scales),
        static_cast<__nv_bfloat16*>(out.data),
        weight.n,
        out.ne[0],
        row_offset,
        weight.qtype == QType::Q5G64_F16S,
    };
}

void launch_slice(bool full, const Tensor& x, const Weight& qk_weight, const Weight& v_weight,
                  Tensor& qkv, cudaStream_t stream) {
    using Schedule                 = GemmCfg<64, 128, 64, 64, 16, 2, 1, false, true, true>;
    const RowSplitGroupedMmaJob qk = make_job(qk_weight, qkv, 0);
    const RowSplitGroupedMmaJob v  = make_job(v_weight, qkv, qk_weight.n);
    RowSplitGroupedMmaJob empty{};
    const int tiles = div_up(qk.n, Schedule::BM) + div_up(v.n, Schedule::BM);
    const int cols  = x.ne[1];
    const dim3 grid(static_cast<unsigned>(tiles),
                    static_cast<unsigned>(div_up(cols, Schedule::BN)));

    if (full) {
        rowsplit_grouped_mma_kernel<Schedule, true, RowSplitGroupedMmaCodec::Mixed, 2>
            <<<grid, Schedule::THREADS, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data), qk,
                                                     v, empty, empty, x.ne[0], cols, x.ne[0]);
    } else {
        rowsplit_grouped_mma_kernel<Schedule, false, RowSplitGroupedMmaCodec::Mixed, 2>
            <<<grid, Schedule::THREADS, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data), qk,
                                                     v, empty, empty, x.ne[0], cols, x.ne[0]);
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void q4_q5_gdn_input_grouped_mma_launch(const Tensor& x, const Weight& qk_weight,
                                        const Weight& v_weight, Tensor& qkv, cudaStream_t stream) {
    constexpr std::int32_t kTileCols = 128;
    const bool full                  = (x.ne[1] % kTileCols) == 0;
    for_each_token_slice(x.ne[1], kTileCols, [&](std::int32_t offset, std::int32_t count) {
        const Tensor x_slice = x.slice(1, offset, count);
        Tensor qkv_slice     = qkv.slice(1, offset, count);
        launch_slice(full, x_slice, qk_weight, v_weight, qkv_slice, stream);
    });
}

} // namespace ninfer::ops::detail
