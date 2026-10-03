#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops::detail {

// T=2..48 share the aggregate MMA tiles. Every BF16 MMA schedule reduces each column in one K
// order, so packed verify aggregates match their per-request panels at every width.
inline constexpr std::int32_t kBf16LinearAddAggregateMmaEnd = 48;

enum class Bf16LinearAddScheduleId : std::uint8_t {
    Decode,
    AggregateMma,
    Mma,
};

bool bf16_linear_add_admits(std::int32_t output_rows, std::int32_t input_rows,
                            std::int32_t tokens) noexcept;
Bf16LinearAddScheduleId bf16_linear_add_select(std::int32_t output_rows, std::int32_t input_rows,
                                               std::int32_t tokens);
const char* bf16_linear_add_schedule_name(Bf16LinearAddScheduleId schedule) noexcept;

void bf16_linear_add_decode_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                   cudaStream_t stream);
void bf16_linear_add_aggregate_mma_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                          cudaStream_t stream);
void bf16_linear_add_mma_launch(const Tensor& x, const Weight& weight, Tensor& residual,
                                cudaStream_t stream);

void bf16_linear_add_dispatch(const Tensor& x, const Weight& weight, Tensor& residual,
                              cudaStream_t stream);

} // namespace ninfer::ops::detail
