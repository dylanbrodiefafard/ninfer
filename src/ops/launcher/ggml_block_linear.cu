#include "ops/launcher/ggml_block_linear.h"

#include "core/device.h"
#include "ops/kernel/ggml_block_linear.cuh"

#include <cstdint>

namespace ninfer::ops::detail {
namespace {

template <QType Format, int Columns, int TileTokens, int MinimumBlocksPerSm>
void launch_replay(const Tensor& x, const Weight& w, Tensor& out, std::uint64_t row_bytes,
                   cudaStream_t stream) {
    const dim3 grid(static_cast<unsigned>(w.n),
                    static_cast<unsigned>(x.ne[1] / TileTokens));
    constexpr int shared_bytes = (Columns + TileTokens * 8) * static_cast<int>(sizeof(float));
    ggml_block_linear_replay_kernel<Format, Columns, TileTokens, MinimumBlocksPerSm>
        <<<grid, 256, shared_bytes, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(w.qdata), static_cast<__nv_bfloat16*>(out.data),
            w.n, row_bytes);
}

template <QType Format>
void launch_format(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    constexpr std::uint64_t block_values = GgmlDecoder<Format>::block_values;
    constexpr std::uint64_t block_bytes  = GgmlDecoder<Format>::block_bytes;
    const auto row_bytes = static_cast<std::uint64_t>(w.k) / block_values * block_bytes;
    const std::int32_t tokens = x.ne[1];
    if constexpr (Format == QType::GGML_Q5_K) {
        if (uses_qwen4_q5_k_weight_replay(Format, w.n, w.k, tokens)) {
            launch_replay<QType::GGML_Q5_K, 2560, kGgmlQ5KReplayTokens, 5>(
                x, w, out, row_bytes, stream);
            return;
        }
    }
    if constexpr (Format == QType::GGML_Q6_K) {
        if (uses_qwen4_q6_k_output_replay(Format, w.n, w.k, tokens)) {
            launch_replay<QType::GGML_Q6_K, 6144, kGgmlQ6KReplayTokens, 3>(
                x, w, out, row_bytes, stream);
            return;
        }
    }
    if (tokens == 1) {
        ggml_block_linear_kernel<Format><<<w.n, 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(w.qdata),
            static_cast<__nv_bfloat16*>(out.data), w.k, row_bytes);
    } else {
        const dim3 grid(
            static_cast<unsigned>(w.n),
            static_cast<unsigned>((tokens + kGgmlBlockLinearAggregateTokens - 1) /
                                  kGgmlBlockLinearAggregateTokens));
        ggml_block_linear_aggregate_kernel<Format><<<grid, 256, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x.data),
            static_cast<const std::uint8_t*>(w.qdata),
            static_cast<__nv_bfloat16*>(out.data), w.n, w.k, tokens, row_bytes);
    }
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
