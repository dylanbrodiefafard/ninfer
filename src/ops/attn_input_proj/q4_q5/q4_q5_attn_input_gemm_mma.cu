#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_kernels.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/rowsplit_grouped_mma.cuh"
#include "ops/common/token_slices.h"

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

RowSplitGroupedMmaJob make_job(const Weight& weight, std::int32_t row_begin, std::int32_t row_count,
                               Tensor& out) {
    if (row_begin < 0 || row_count <= 0 || row_begin + row_count > weight.n ||
        out.ne[0] != row_count) {
        throw std::invalid_argument("Q4/Q5 attention input grouped MMA row view is invalid");
    }
    const std::int64_t groups = weight.padded_shape[1] / 64;
    const auto* codes         = static_cast<const std::uint8_t*>(weight.qdata) +
                        static_cast<std::int64_t>(row_begin) * groups * 32;
    const auto* high   = weight.qtype == QType::Q5G64_F16S
                             ? static_cast<const std::uint8_t*>(weight.qhigh) +
                                 static_cast<std::int64_t>(row_begin) * groups * 8
                             : nullptr;
    const auto* scales = static_cast<const std::uint8_t*>(weight.scales) +
                         static_cast<std::int64_t>(row_begin) * groups * 2;
    return RowSplitGroupedMmaJob{
        codes,     high,      scales, static_cast<__nv_bfloat16*>(out.data),
        row_count, out.ne[0], 0,      weight.qtype == QType::Q5G64_F16S,
    };
}

template <class Cfg, RowSplitGroupedMmaCodec Codec>
void launch_pair(bool full, const Tensor& x, RowSplitGroupedMmaJob first,
                 RowSplitGroupedMmaJob second, cudaStream_t stream) {
    const int tiles = div_up(first.n, Cfg::BM) + div_up(second.n, Cfg::BM);
    const int cols  = x.ne[1];
    const dim3 grid(static_cast<unsigned>(tiles), static_cast<unsigned>(div_up(cols, Cfg::BN)));
    RowSplitGroupedMmaJob empty{};

    if (full) {
        rowsplit_grouped_mma_kernel<Cfg, true, Codec, 2>
            <<<grid, Cfg::THREADS, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data), first,
                                                second, empty, empty, x.ne[0], cols, x.ne[0]);
    } else {
        rowsplit_grouped_mma_kernel<Cfg, false, Codec, 2>
            <<<grid, Cfg::THREADS, 0, stream>>>(static_cast<const __nv_bfloat16*>(x.data), first,
                                                second, empty, empty, x.ne[0], cols, x.ne[0]);
    }
    CUDA_CHECK(cudaGetLastError());
}

void launch_slice(bool full, const Tensor& x, const Weight& query_key_weight,
                  const Weight& gate_value_weight, Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                  cudaStream_t stream) {
    using Schedule = GemmCfg<64, 128, 64, 64, 32, 2, 1, false, true, true>;
    launch_pair<Schedule, RowSplitGroupedMmaCodec::Q4>(
        full, x, make_job(query_key_weight, 0, 6144, q), make_job(query_key_weight, 6144, 1024, k),
        stream);
    launch_pair<Schedule, RowSplitGroupedMmaCodec::Q5>(
        full, x, make_job(gate_value_weight, 0, 6144, gate),
        make_job(gate_value_weight, 6144, 1024, v), stream);
}

} // namespace

void q4_q5_attn_input_grouped_mma_launch(const Tensor& x, const Weight& query_key_weight,
                                         const Weight& gate_value_weight, Tensor& q, Tensor& gate,
                                         Tensor& k, Tensor& v, cudaStream_t stream) {
    constexpr std::int32_t kTileCols = 128;
    const bool full                  = (x.ne[1] % kTileCols) == 0;
    for_each_token_slice(x.ne[1], kTileCols, [&](std::int32_t offset, std::int32_t count) {
        const Tensor x_slice = x.slice(1, offset, count);
        Tensor q_slice       = q.slice(1, offset, count);
        Tensor gate_slice    = gate.slice(1, offset, count);
        Tensor k_slice       = k.slice(1, offset, count);
        Tensor v_slice       = v.slice(1, offset, count);
        launch_slice(full, x_slice, query_key_weight, gate_value_weight, q_slice, gate_slice,
                     k_slice, v_slice, stream);
    });
}

} // namespace ninfer::ops::detail
