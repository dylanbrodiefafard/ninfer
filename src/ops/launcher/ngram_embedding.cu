#include "ops/launcher/ngram_embedding.h"

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/kernel/ngram_embedding.cuh"

namespace ninfer::ops::detail {

void ngram_row_ids_launch(const Tensor& input_ids, const Tensor& valid_tokens,
                          const Tensor& old_history, const NgramRowLaunchConfig& config,
                          Tensor& row_ids, Tensor& new_history, cudaStream_t stream) {
    constexpr int block        = 256;
    const std::int64_t entries = static_cast<std::int64_t>(input_ids.ne[0]) * input_ids.ne[1] * 16;
    const int row_grid         = static_cast<int>(div_up(entries, static_cast<std::int64_t>(block)));
    ngram_row_ids_kernel<<<row_grid, block, 0, stream>>>(
        static_cast<const std::int32_t*>(input_ids.data),
        static_cast<const std::int32_t*>(valid_tokens.data),
        static_cast<const std::int32_t*>(old_history.data), config,
        static_cast<std::int32_t*>(row_ids.data), input_ids.ne[0], input_ids.ne[1]);
    CUDA_CHECK(cudaGetLastError());

    constexpr int history_block = 32;
    const int history_grid       = div_up(input_ids.ne[1], history_block);
    ngram_history_kernel<<<history_grid, history_block, 0, stream>>>(
        static_cast<const std::int32_t*>(input_ids.data),
        static_cast<const std::int32_t*>(valid_tokens.data),
        static_cast<const std::int32_t*>(old_history.data),
        static_cast<std::int32_t*>(new_history.data), input_ids.ne[0], input_ids.ne[1]);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
