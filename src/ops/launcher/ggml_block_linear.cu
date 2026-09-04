#include "ops/launcher/ggml_block_linear.h"

#include "core/device.h"
#include "ops/kernel/ggml_block_linear.cuh"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

template <QType Format>
void launch_format(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    constexpr std::uint64_t block_values = GgmlDecoder<Format>::block_values;
    constexpr std::uint64_t block_bytes  = GgmlDecoder<Format>::block_bytes;
    const auto row_bytes = static_cast<std::uint64_t>(w.k) / block_values * block_bytes;
    ggml_block_linear_kernel<Format><<<w.n, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x.data), static_cast<const std::uint8_t*>(w.qdata),
        static_cast<__nv_bfloat16*>(out.data), w.k, row_bytes);
}

} // namespace

void ggml_block_linear_launch(const Tensor& x, const Weight& w, Tensor& out,
                              cudaStream_t stream) {
    switch (w.qtype) {
    case QType::GGML_Q8_0:
        launch_format<QType::GGML_Q8_0>(x, w, out, stream);
        break;
    case QType::GGML_Q4_K:
        launch_format<QType::GGML_Q4_K>(x, w, out, stream);
        break;
    case QType::GGML_Q5_K:
        launch_format<QType::GGML_Q5_K>(x, w, out, stream);
        break;
    case QType::GGML_Q6_K:
        launch_format<QType::GGML_Q6_K>(x, w, out, stream);
        break;
    case QType::GGML_IQ1_S:
        launch_format<QType::GGML_IQ1_S>(x, w, out, stream);
        break;
    case QType::GGML_IQ2_XXS:
        launch_format<QType::GGML_IQ2_XXS>(x, w, out, stream);
        break;
    case QType::GGML_IQ4_NL:
        launch_format<QType::GGML_IQ4_NL>(x, w, out, stream);
        break;
    default:
        break;
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
